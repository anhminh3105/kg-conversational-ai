# Evaluation Suite -- PrimeKG Drug-Disease

Automated evaluation of the KG-RAG system using the PrimeKG drug-disease subset
(indication, contraindication, off-label use). Compares system
configurations under controlled knowledge-availability conditions (Full / Partial tiers).

## Quick Start

```bash
# 1. Configure LLM backend
source export_dual_llm.sh          # local primary + remote validation (needs GPU)
# OR
source export_dual_remote_llm.sh   # both LLMs via API (no GPU needed)

# 2. Run the full pipeline (split → import → QA gen → evaluate)
bash scripts/eval/run_eval_pipeline.sh

# 3. Results appear in data/eval/eval_N*_*.json
```

## Pipeline Overview

```
   data/kg_drug_disease.csv          (PrimeKG drug-disease subset, ~43 K rows)
              │
              ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │ split_primekb.py                                                    │
   │   • drug-aware sampling (--max-rows)                                │
   │   • per-relation entity-disjoint split (--test-ratio)               │
   │   • Full / Partial tier assignment (--full-ratio)                   │
   │   • Held-out triple selection for Partial tier (--partial-include)  │
   └─────────────────────────────────────────────────────────────────────┘
              │
              ├─► train.csv               (KG content -> Neo4j)
              ├─► test.csv                (full gold ground truth -> QA)
              ├─► tier_assignments.json   ({drug, relation} -> tier + answers)
              └─► split_stats.json        (hyperparameters + counts)
              │
              ▼
   import_primekb_to_neo4j.py --input train.csv --clear        →  Neo4j
              │
              ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │ generate_qa.py                                                      │
   │   • forward + reverse questions per relation                        │
   │   • tier-routed prompts (standard vs "list ALL ...")                │
   │   • atomic resume cache via --output                                │
   └─────────────────────────────────────────────────────────────────────┘
              │
              ▼
   qa_dataset.json    (id, question, gold_answers, kg_answers,
                       held_out_answers, tier, relation, direction)
              │
              ▼
   evaluate.py --configs without-validation with-validation
              │
              ▼
   eval_N{n}_{configs}_{tiers}_{timestamp}.json   (per-config + per-tier metrics)
```

## Dataset Preparation & Tiered Train / Test Split

`split_primekb.py` is the heart of the evaluation pipeline: it takes the
PrimeKG drug-disease subset and produces an entity-disjoint train/test split
plus a tier assignment that controls *exactly* which gold triples each test
question can or cannot recover from Neo4j alone. The design lets us measure
two distinct system capabilities with one dataset:

1. **Retrieval** -- can the system find a gold triple that *is* present in
  the KG?  (Full tier)
2. **Knowledge gap recovery** -- can the system recover a gold triple that
  has been *deliberately removed* from the KG, using its parametric LLM
   knowledge plus dual-LLM validation?  (Partial tier)

### 1. Splitting Algorithm (4 stages, in order)

```
                  data/kg_drug_disease.csv
                            │
                            ▼
        ┌──────────────────────────────────────────┐
        │ STAGE 1 — Drug-aware row sampling        │
        │  Keep all triples for each sampled drug  │
        │  together until --max-rows is reached    │
        └──────────────────────────────────────────┘
                            │
                            ▼
        ┌──────────────────────────────────────────┐
        │ STAGE 2 — Per-relation entity-disjoint   │
        │  train / test split                      │
        │  (no drug appears in both sides for the  │
        │  same relation)                          │
        └──────────────────────────────────────────┘
                            │
                            ▼
        ┌──────────────────────────────────────────┐
        │ STAGE 3 — Test-drug tier assignment      │
        │  full    = 100 % of triples in train     │
        │  partial = subset in train + rest held   │
        │            out (only drugs with ≥ 2      │
        │            triples are eligible)         │
        └──────────────────────────────────────────┘
                            │
                            ▼
        ┌──────────────────────────────────────────┐
        │ STAGE 4 — Per-drug held-out selection    │
        │  (Partial tier only, controlled by       │
        │  --partial-include)                      │
        └──────────────────────────────────────────┘
                            │
                            ▼
   train.csv + test.csv + tier_assignments.json + split_stats.json
```

#### Stage 1 -- Drug-aware row sampling (`--max-rows`)

The full drug-disease subset has roughly 43 K rows. To keep evaluation cheap,
the splitter caps the input at `--max-rows` (default `1000`, `0` = no cap).
Naive row-level sampling would shred multi-triple drug groups, leaving most
test drugs with a single triple per relation -- which makes the Partial tier
impossible to populate.

`_drug_aware_sample()` instead samples *complete drugs*: it shuffles the
unique `x_name` values and keeps drugs (with all their triples) until the
cumulative row count meets `--max-rows`. The resulting subset preserves the
multi-triple structure required by the Partial tier.

#### Stage 2 -- Per-relation, entity-disjoint train/test split (`--test-ratio`)

For each of the three drug-disease relations
(`indication`, `contraindication`, `off-label use`) the splitter performs an
**entity-disjoint** split via `entity_disjoint_split()`:

- Shuffle the unique drug names for that relation.
- Walk the shuffled list, accumulating drugs into the test bucket until the
cumulative triple count reaches `--test-ratio` × (relation row count).
- The remaining drugs form the **background train** rows for that relation
(i.e. drugs that never appear in test and provide unrelated context in the
KG).

This guarantees that no drug ever appears in both train and test for the
same relation, so test gold answers cannot leak into train through the same
drug.

##### Why entity-disjoint per relation?

The split is the foundation of every Gold Recall number in the report. Naive
row-level sampling would silently break the metric in four ways:

1. **Prevents same-drug leakage.** If drug *D* appeared in both `train.csv`
   (some of its triples) and `test.csv` (others), then *D*'s training triples
   would land in Neo4j *and* serve as gold answers for the test question
   about *D*. Config B's KG search would always retrieve them, inflating
   Gold Recall to ≈ 1.0 regardless of whether the system can actually
   recover *unknown* answers.
2. **Bounds Config B's retrieval ceiling.** With per-relation
   entity-disjointness, every triple in `held_out_answers` is *guaranteed
   absent* from Neo4j for that relation-direction. Config B's recall is
   therefore bounded by what was deliberately kept (`kg_answers`), and only
   Config C's propose / validate loop can recover the held-out answers --
   which is *exactly* the diagnostic point of the eval.
3. **Per-relation, not global, by design.** A drug can be train-only for
   `indication` and test-only for `contraindication` because
   `entity_disjoint_split()` runs once per relation. This keeps the
   relation-level Gold Recall rows in the report comparable: each row
   measures recovery on truly unseen edges of that semantic type, not
   residual retrieval of leaked edges of another type.
4. **Carries through to reverse-direction questions.** Reverse questions
   ("which drugs target disease *X*?") inherit the most-restrictive tier
   across contributing drugs (see Stage 3). Without per-relation disjointness
   on the forward side, a Partial reverse question could still be "answered"
   by retrieving the training triples of a Full-tier neighbour drug, leaking
   the gold via a different traversal path.

In short: entity-disjointness is what turns Gold Recall from a memorisation
score into a knowledge-recovery score.

##### Subject-only scope

Disjointness applies to the **subject** (`x_name`, drug) only. Diseases
(`y_name`) stay shared -- they are the KG's connective tissue, and
splitting on them would shred graph density. Reverse-axis overlap is
handled by the any-Partial inheritance rule (see § QA Dataset Generation
> Reverse-direction tier inheritance).

##### Role of the train background (80 %)

Background drugs (in `train.csv`, never in `test.csv`) never appear in
`gold_answers`, so they don't move Gold Recall directly. They exist to
keep the retrieval environment realistic: Neo4j stays ~93 % full
(real PrimeKG density), vector / graph search competes with real noise,
and per-question latency (Config B ~23 s, Config C ~173 s) is measured
against the full KG. Strip them out and the numbers stay the same but
the measurement happens in a sterile bubble.

##### Alternative considered: tier-only on all drugs

Skip Stage 2 and apply Full / Partial tiering to *every* sampled drug
(e.g. 80 % Full / 20 % Partial).

**Wins**

- **Honest hallucination metric** -- real PrimeKG drugs no longer count
  as hallucinations just because they weren't sampled into test.
- **Larger golds and test set** -- reverse gold grows ~1-2 → ~5-10
  drugs; test set ~194 → ~900 questions (tighter CIs).
- **Larger Full baseline** -- Full tier ~6 % (= 30 % × 20 %) → ~80 %
  of drugs, sharpening the retriever-health diagnostic.
- **Simpler** -- one stage, no `background` concept.

**Costs**

- **~5× eval throughput** -- 5-9 h runs vs 1-2 h today.
- **Diluted Δ(B → C)** -- Config B's retrieval cap rises ~50 % → ~90 %,
  so the same recovery work shows up against a higher (more honest)
  baseline.

**Migration.** Keep current as the primary benchmark while throughput is
the binding constraint; revisit tier-only as a v2 methodology appendix
when it relaxes.

#### Stage 3 -- Tier assignment for test drugs (`--full-ratio`)

`assign_tiers()` partitions the test drugs of each relation into Full and
Partial:

- Drugs with **only one triple** for that relation are forced into the Full
tier (they have nothing to hold out).
- The remaining multi-triple drugs are shuffled and the first
`max(1, len × --full-ratio)` go to **Full**, the rest to **Partial**.

The default `--full-ratio 0.30` therefore yields roughly a 30 / 70 Full /
Partial split *of the eligible multi-triple drugs* -- single-triple drugs
inflate the Full count slightly above 30 % in practice (see Stage 4 example
below).

#### Stage 4 -- Held-out triple selection (`--partial-include`)

For each Partial-tier drug, `--partial-include` (default `0.50`) of its
triples are **kept** in `train.csv` (so the KG sees some context for that
drug) and the remaining triples are **held out** from train but still
recorded in `test.csv`. Specifically:

```python
n_keep = int(len(triples_for_drug) * partial_include)
sampled  = triples.sample(n=min(n_keep, len(triples)), random_state=seed)
held_out = triples.drop(sampled.index)
```

Note that `int()` truncates: a Partial drug with 3 triples and the default
`partial_include=0.5` keeps `int(3 * 0.5) = 1` triple in train and holds
out 2. This intentional truncation ensures Partial drugs always have at
least one held-out answer when they have ≥ 2 triples.

#### Tier visibility — what `train.csv` and `test.csv` see

Once Stages 1-4 finish, every test drug ends up in exactly one of two
visibility regimes. The contrast is the whole point of the tier
design — it is what makes Config B vs Config C a clean A/B test for
knowledge-gap recovery rather than a memorisation test.

```
                        train.csv  (-> Neo4j)         test.csv (gold)
                        ─────────────────────         ────────────────
   Full-tier drug   ┌─► every (drug, r, *) triple ◄─► same complete set
                    │       (kg_answers = gold_answers,
                    │        held_out_answers = [])
                    │
   Partial-tier ────┤   subset of the drug's          all (drug, r, *)
   drug             │   (drug, r, *) triples          triples (Stage 4
                    │   (kg_answers ⊂ gold_answers,    gives test the
                    └─  the rest are removed)         held-out portion)
```

Concretely, for a single test drug **D** and one relation **r** with gold
diseases `{X, Y, Z}`:

| Drug **D**'s tier for **r** | What lands in `train.csv` (Neo4j) | What `test.csv` carries | `kg_answers` | `held_out_answers` |
| --------------------------- | --------------------------------- | ----------------------- | ------------ | ------------------ |
| **Full**                    | `(D, r, X), (D, r, Y), (D, r, Z)` | same three rows         | `{X, Y, Z}`  | `[]`               |
| **Partial** (`partial_include=0.5`, 3 triples → keep 1) | `(D, r, X)` *only* | all three rows | `{X}`        | `{Y, Z}`           |

So:

- Train **and** test see the *complete* gold answer set for Full-tier drugs.
   `gold_answers = kg_answers` and Config B can in principle hit Gold Recall
   1.0 by retrieving them straight from Neo4j.
- Train sees only a *subset* of the gold answer set for Partial-tier drugs;
   test still carries the complete set. The held-out portion
   (`held_out_answers`) is invisible to retrieval and can only be supplied
   by Config C's propose / validate loop. This is what bounds Config B's
   recall and turns the Config B → Config C delta into a recovery signal.

The Stage 2 entity-disjointness rule is what *guarantees* this contrast
holds: because no drug ever appears in both train and test for the same
relation, the only triples Config B can retrieve for a test drug are the
ones Stage 3 / Stage 4 *deliberately* placed back into `train.csv`. There
is no "background" pollution from the same drug's other triples leaking
in via the train side.

### 2. Outputs (in `--output-dir`, default `data/eval/`)


| File                    | Content                                                                                                                                      |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `train.csv`             | Background train rows + Full-tier full triples + Partial-tier kept subset. This is what gets imported into Neo4j.                            |
| `test.csv`              | All triples for every test drug (Full + Partial), including the held-out portion. Used by `generate_qa.py` to derive multi-answer gold sets. |
| `tier_assignments.json` | Per-(drug, relation) tier label, `kg_answers` (in train.csv), and `held_out_answers` (kept out of train.csv).                                |
| `split_stats.json`      | Hyperparameters, row counts, per-relation and per-tier breakdowns.                                                                           |


#### `tier_assignments.json` schema

```jsonc
{
  "<drug_name>": {
    "<relation>": {
      "tier":             "full" | "partial",
      "kg_answers":       ["<disease_1>", "<disease_2>", ...],   // present in train.csv
      "held_out_answers": ["<disease_3>", ...]                    // present in test.csv only
    },
    ...
  },
  ...
}
```

For **Full-tier** drugs, `held_out_answers` is always `[]` and `kg_answers`
is the complete gold set. For **Partial-tier** drugs, the union
`kg_answers ∪ held_out_answers` is the complete gold set, and the system
must recover at least the held-out portion via dual-LLM validation to score
above Config B on Gold Recall.

#### Worked example (`split_stats.json` from the default 1000-row run)

```jsonc
{
  "seed": 42,                       // np.random.default_rng(seed)
  "test_ratio_requested": 0.2,
  "max_rows": 1000,
  "full_ratio": 0.3,
  "partial_include": 0.5,
  "input_file": "data/kg_drug_disease.csv",
  "total_rows_after_sampling": 970, // < 1000 because drug-aware sampling stopped before the cap
  "train_rows": 884,                // background + Full-full + Partial-kept (de-duped)
  "test_rows":  220,
  "relations": {
    "contraindication": {
      "total": 649, "train_background": 495, "test": 154,
      "train_drugs": 21, "test_drugs": 14,
      "tiers": {
        "full":    { "drugs":  4, "triples":  32 },
        "partial": { "drugs": 10, "triples": 122 }
      }
    },
    "indication":      { /* ... */ },
    "off-label use":   { /* ... */ }
  }
}
```

Reading the contraindication row top-to-bottom:

- 649 contraindication rows survived drug-aware sampling.
- 35 unique drugs hold those rows; 21 went to **train background** (495
rows of *background* triples that never appear in test), 14 became
**test drugs** (154 rows of *test* triples).
- Of the 14 test drugs, 4 went to **Full** tier and 10 went to **Partial**.
All 32 Full-tier triples are also written to `train.csv` (so
`kg_answers = gold_answers` for those drugs). For the 10 Partial drugs,
Stage 4 keeps `int(n × 0.5)` *per drug* in `train.csv` (so a drug with
3 triples contributes 1 to train and 2 to `held_out_answers`); the rest
of the 122 Partial triples stay only in `test.csv` and form the
recovery target for Config C.

Net effect on `train.csv` for contraindication: 495 (background) + 32
(Full-full) + ≈ 50 (Partial-kept, depending on per-drug truncation) ≈ 577
contraindication rows imported into Neo4j, while `test.csv` carries the
full 154 test rows for QA generation.

### 3. Usage

```bash
# Default: 1000-row drug-aware sample, 80/20 train/test, 30/70 Full/Partial
python scripts/eval/split_primekb.py --input data/kg_drug_disease.csv

# Larger sample (5 K rows), fewer held-out answers per Partial drug
python scripts/eval/split_primekb.py \
    --input data/kg_drug_disease.csv \
    --max-rows 5000 \
    --partial-include 0.7

# Use the entire 43 K-row drug-disease subset (slow but most realistic)
python scripts/eval/split_primekb.py \
    --input data/kg_drug_disease.csv \
    --max-rows 0
```

### 4. CLI Reference


| Flag                | Default                    | Description                                                                    |
| ------------------- | -------------------------- | ------------------------------------------------------------------------------ |
| `--input`           | `data/kg_drug_disease.csv` | Path to the drug-disease subset CSV produced by `scripts/download_primekb.py`  |
| `--output-dir`      | `data/eval`                | Directory where the four output files are written                              |
| `--test-ratio`      | `0.2`                      | Per-relation target fraction of rows to send to the test bucket (Stage 2)      |
| `--max-rows`        | `1000`                     | Cap total input rows via drug-aware sampling (Stage 1). `0` disables sampling. |
| `--full-ratio`      | `0.30`                     | Fraction of multi-triple test drugs assigned to **Full** tier (Stage 3)        |
| `--partial-include` | `0.50`                     | Fraction of each Partial-tier drug's triples included in train (Stage 4)       |
| `--seed`            | `42`                       | Random seed for sampling, shuffling, and held-out selection                    |


### 5. Reproducibility & determinism

All randomness flows from a single `--seed`:

- `np.random.default_rng(seed)` drives drug-aware sampling, the per-relation
shuffle in `entity_disjoint_split()`, and the multi-triple-drug shuffle in
`assign_tiers()`.
- The per-drug held-out selection in Stage 4 uses
`pd.DataFrame.sample(random_state=seed)` (note: not the same RNG, but
derived from the same `--seed` value).

Re-running `split_primekb.py` with the same `--seed`, same `--input`, and
the same hyperparameters produces byte-identical `train.csv`, `test.csv`,
and `tier_assignments.json`. This is what makes the eval suite a true
benchmark: every Config B vs Config C comparison runs against the same
gold-answer ground truth.

## QA Dataset Generation

`generate_qa.py` consumes `test.csv` (and `tier_assignments.json` next to
it) and produces a natural-language QA dataset whose questions are
**tier-aware** and bidirectional. The LLM only synthesises the question
text; gold answers, KG answers, and held-out answers are derived
deterministically from the splitter's outputs, so the gold ground truth
never depends on the question-writing LLM.

### 1. How questions are constructed

For each relation, `generate_qa.py` walks `test.csv` *twice* — once
grouped by the **drug** column (`x_name`) for forward questions, once
grouped by the **disease** column (`y_name`) for reverse questions. The
LLM only writes the natural-language phrasing; the gold-answer
ground truth is derived deterministically from the splitter's outputs,
so it never depends on the question-writing model.


| Direction | Group key          | Gold answer set                                                  | Example phrasing                                |
| --------- | ------------------ | ---------------------------------------------------------------- | ----------------------------------------------- |
| `forward` | `x_name` (drug)    | All **diseases** linked to that drug for the relation, in test   | "What conditions is *Drug X* indicated for?"    |
| `reverse` | `y_name` (disease) | All **drugs** linked to that disease for the relation, in test   | "Which drugs are indicated for *Disease Y*?"    |


`--max-questions-per-relation` (default `100`) caps each direction-group
independently before any LLM calls are made, so the dataset size is bounded
by `2 × 100 × n_relations = 600` in the worst case.

#### How `gold_answers`, `kg_answers`, and `held_out_answers` are derived

For a **forward** question the entity is the test drug **D**, and all
three answer lists come straight from `tier_assignments.json[D][r]`
(they were already computed by `split_primekb.py`):

```
gold_answers      = kg_answers ∪ held_out_answers   (the whole gold set in test.csv)
kg_answers        = tier_assignments[D][r]["kg_answers"]
held_out_answers  = tier_assignments[D][r]["held_out_answers"]
tier              = tier_assignments[D][r]["tier"]
```

The Full / Partial visibility table from §1 (Splitting Algorithm) carries
through directly:

- Full-tier drug → `kg_answers = gold_answers`, `held_out_answers = []`.
   Config B can in principle retrieve every gold disease from Neo4j.
- Partial-tier drug → `kg_answers ⊂ gold_answers`. The diseases in
   `held_out_answers` are not in Neo4j and must be recovered by Config C.

For a **reverse** question the entity is a *disease* **X**, which has
no direct row in `tier_assignments.json`. The script reconstructs the
three lists by walking every contributing drug **D** in the gold answer
list and checking, *per-drug*, whether its edge to **X** sits in
`kg_answers` or `held_out_answers`:

```python
for drug in gold_drugs_for(X, r):                              # built from test.csv
    drug_kg, drug_held = tier_assignments[drug][r]["kg_answers"], \
                         tier_assignments[drug][r]["held_out_answers"]
    if X in drug_kg:    entity_kg.append(drug)                 # drug -> X is in Neo4j
    if X in drug_held:  entity_held.append(drug)               # drug -> X is held out

gold_answers     = list(gold_drugs_for(X, r))
kg_answers       = entity_kg
held_out_answers = entity_held
```

The same Stage 2 entity-disjointness guarantee that bounds the forward
metric also bounds the reverse one: a drug whose `(drug, r, X)` edge is
in `train.csv` lands in `kg_answers`; the rest land in `held_out_answers`
and remain Config C's recovery target. Because each drug's tier was
fixed by `split_primekb.py` (never recomputed at QA time), the per-drug
visibility carries through to the reverse axis without any new
randomness.

### 2. Tier routing for prompts (Full vs Partial)

The script inspects each entity's tier *before* dispatching it to the LLM
and routes it to one of two prompt sets:


| Tier prompt set            | When it is used                         | Prompt style                                                                                                    |
| -------------------------- | --------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `RELATION_CONTEXT`         | Entity is **Full** for that relation    | Standard, single-fact phrasing ("What condition is *Drug X* indicated for?")                                    |
| `PARTIAL_RELATION_CONTEXT` | Entity is **Partial** for that relation | Completeness-demanding phrasing ("List **ALL** conditions...", "Provide a comprehensive list of every drug...") |


The completeness-demanding phrasing is a deliberate evaluation design
choice: it is engineered to trip the agent's INSUFFICIENT assessment in
Config B (which then refuses) and trigger the propose/validate loop in
Config C. The delta in Gold Recall between the two configs on Partial-tier
questions is the **knowledge-gap recovery signal**.

Within a single `(relation, direction)` batch, the LLM is called *up to
twice* -- once for Full entities, once for Partial entities -- so the
prompt context for each batch matches the tier of every entity in it.

### 3. Reverse-direction tier inheritance

For `direction=reverse` the entity is a *disease*, not a drug, so it has no
direct entry in `tier_assignments.json`. The script aggregates the tier of
the contributing drugs:

```python
tiers = { tier_of(drug, relation) for drug in drugs_indicating_this_disease }
inherited = "partial" if "partial" in tiers else "full"
```

In other words: a reverse question is treated as **Partial** if *any* drug
in its gold answer list is a Partial-tier drug for that relation. This
mirrors what the system actually has to recover: even one missing drug in
the KG turns the reverse question into a knowledge-gap recovery test.

#### Worked example — the *any-Partial* rule in practice

Suppose the test split contains three drugs all linked to *Fever* via
`indication`:

| Drug          | Tier for `indication` | `kg_answers` for that drug                | `held_out_answers` for that drug |
| ------------- | --------------------- | ----------------------------------------- | -------------------------------- |
| Aspirin       | Full                  | `{Fever, Pain, Inflammation}`             | `[]`                             |
| Ibuprofen     | Partial               | `{Pain}`                                  | `{Fever}`                        |
| Paracetamol   | Full                  | `{Fever, Pain}`                           | `[]`                             |

When `generate_qa.py` builds the reverse question
*"Which drugs are indicated for Fever?"*, it walks the three contributing
drugs and re-projects each one's tier record onto the disease in
question:

```
for D in [Aspirin, Ibuprofen, Paracetamol]:                # from test.csv
    if Fever ∈ tier_assignments[D]["indication"]["kg_answers"]:
        kg_answers.append(D)                               # drug -> Fever in Neo4j
    if Fever ∈ tier_assignments[D]["indication"]["held_out_answers"]:
        held_out_answers.append(D)                         # drug -> Fever held out
```

Result for the reverse question record:

```jsonc
{
  "entity":           "Fever",
  "direction":        "reverse",
  "relation":         "indication",
  "gold_answers":     ["Aspirin", "Ibuprofen", "Paracetamol"],   // every drug in test.csv linked to Fever
  "kg_answers":       ["Aspirin", "Paracetamol"],                // (drug -> Fever) edges that are in train.csv
  "held_out_answers": ["Ibuprofen"],                             // (drug -> Fever) edges held out from train.csv
  "tier":             "partial"                                  // any-Partial rule: Ibuprofen is Partial
}
```

What this example pins down:

- **Gold reconstruction is per-drug, not per-disease.** Each contributing
   drug carries its own tier record, and the reverse question simply
   re-projects those records onto the disease. The tier of "Fever as a
   reverse question" is *derived*, never stored.
- **Config B is naturally bounded.** Aspirin and Paracetamol are Full and
   their `(drug, indication, Fever)` edges live in `train.csv`, so they
   land in `kg_answers` and Config B can retrieve them. Ibuprofen's edge
   was held out by Stage 4, so Config B cannot retrieve it.
- **The any-Partial rule is the right rule.** A single missing edge on
   the reverse side is enough to leave the gold answer set incomplete
   from Neo4j's perspective. Tagging the question Partial makes
   `generate_qa.py` use the completeness-demanding `PARTIAL_RELATION_CONTEXT`
   prompt, which trips Config B into INSUFFICIENT and routes Config C
   into its propose / validate path — i.e. the tier label matches the
   recovery work the question actually requires.
- **No new visibility is introduced.** Reverse questions reuse the *same*
   `kg_answers` / `held_out_answers` records that `split_primekb.py` wrote
   for each forward `(drug, relation)` pair. There is no separate reverse
   split, no second tier assignment, and no chance for a Partial held-out
   edge to silently appear as a Full answer on the reverse axis.

### 4. Cache & resume semantics (`--no-cache`)

`qa_dataset.json` is *both* the final output and the incremental cache.
After every batch the full record list is rewritten to disk, so an
interrupted run can resume cheaply:

```bash
# First run -- generates everything
python scripts/eval/generate_qa.py

# Interrupted half-way? Just rerun -- already-generated entities are skipped
python scripts/eval/generate_qa.py

# Force a clean rebuild (ignores existing qa_dataset.json)
python scripts/eval/generate_qa.py --no-cache
```

Resume keys are `f"{entity}|{relation}|{direction}"`. IDs (`q0001`,
`q0002`, ...) are reassigned after the final batch, so they are always
contiguous regardless of how many resumes happened.

If the LLM batch call fails *or* returns malformed JSON for a particular
entity, the script falls back to a deterministic template
(`FALLBACK_TEMPLATES` for Full, `PARTIAL_FALLBACK_TEMPLATES` for Partial).
A warning is logged and the record is still emitted with the templated
question, so a single API hiccup never blocks the rest of the batch.

### 5. Backend selection

```bash
# Local LLM (Qwen-3+ recommended for thinking mode)
source export_local_qwen3.sh
python scripts/eval/generate_qa.py

# Remote LLM (Gemma 4 / Gemini)
source export_dual_remote_llm.sh
python scripts/eval/generate_qa.py --remote-llm
```

`--remote-llm` overrides `USE_LOCAL_LLM=true`, which is useful when both a
local and a remote backend are configured in the same shell.

### 6. CLI Reference


| Flag                           | Default                     | Description                                               |
| ------------------------------ | --------------------------- | --------------------------------------------------------- |
| `--test-csv`                   | `data/eval/test.csv`        | Path to the test split CSV produced by `split_primekb.py` |
| `--output`                     | `data/eval/qa_dataset.json` | Output JSON path; also the resume cache                   |
| `--max-questions-per-relation` | `100`                       | Cap *per direction* (so up to 200 per relation total)     |
| `--batch-size`                 | `15`                        | Entities per LLM API call (per tier-routed sub-batch)     |
| `--no-cache`                   | off                         | Force regeneration; ignore any existing `--output` file   |
| `--remote-llm`                 | off                         | Use remote LLM even when `USE_LOCAL_LLM=true`             |
| `--seed`                       | `42`                        | Random seed for the per-relation entity shuffle           |


### 7. Output Format

Each entry in `qa_dataset.json` carries everything `evaluate.py` needs to
score it without re-reading `tier_assignments.json`:

```json
{
  "id":               "q0001",
  "entity":           "Mirabegron",
  "question":         "List ALL diseases and conditions for which Mirabegron is contraindicated.",
  "gold_answers":     ["hypertension", "overactive bladder"],
  "relation":         "contraindication",
  "direction":        "forward",
  "tier":             "partial",
  "kg_answers":       ["hypertension"],
  "held_out_answers": ["overactive bladder"],
  "source_triplets":  ["(Mirabegron, contraindication, hypertension)",
                       "(Mirabegron, contraindication, overactive bladder)"]
}
```

Field semantics:


| Field              | Meaning                                                                                                                                        |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `gold_answers`     | The complete ground truth (everything in `test.csv` for this entity+relation+direction). The denominator of Gold Recall.                       |
| `kg_answers`       | Subset of `gold_answers` that is also in `train.csv` (i.e. retrievable from Neo4j). For Full-tier records this equals `gold_answers`.          |
| `held_out_answers` | Subset of `gold_answers` that is **only** in `test.csv` (i.e. removed from Neo4j). The recovery target for Config C.                           |
| `tier`             | `"full"` if all gold answers are in the KG, `"partial"` if at least one is held out (computed via the inheritance rule for reverse questions). |


## Evaluation

`evaluate.py` runs the KG-RAG system on each question and computes metrics by
comparing system predictions against the gold answers.

### Configurations


| Config | Aliases              | Description                               | Behavior on INSUFFICIENT assessment          |
| ------ | -------------------- | ----------------------------------------- | -------------------------------------------- |
| **A**  | `rag`                | Pure RAG -- vector retrieval only         | N/A (no assessment)                          |
| **B**  | `without-validation` | Agent with KG tools, no remote validation | Returns refusal (no answer attempt)          |
| **C**  | `with-validation`    | Agent + dual-LLM validated expansion      | Proposes triplets, validates with remote LLM |


Config names and their single-letter aliases (`A`, `B`, `C`) are interchangeable
on the command line.

**Config B behavior**: Uses a simplified assessment prompt that checks whether
the retrieved facts are relevant to the question (without comparing against the
LLM's own knowledge). If INSUFFICIENT (no relevant facts found), returns a
refusal instead of hallucinating. If SUFFICIENT, generates an answer from the
retrieved facts only.

**Config C behavior**: Uses the full assessment prompt that demands completeness.
When the LLM detects missing information, it proposes triplets which are then
validated by the remote LLM before being used in the answer.

### Usage

```bash
# Run Configs B and C (default)
python scripts/eval/evaluate.py --verbose

# Run a specific config
python scripts/eval/evaluate.py --configs C --output-dir data/eval --verbose

# Filter by tier
python scripts/eval/evaluate.py --configs B C --tier partial

# Re-evaluate specific questions (targeted debugging)
python scripts/eval/evaluate.py --configs C \
  --questions "q0014,q0034,q0039" --output-dir data/eval --verbose
```

### CLI Reference


| Flag               | Default                               | Description                                                |
| ------------------ | ------------------------------------- | ---------------------------------------------------------- |
| `--qa-dataset`     | `data/eval/qa_dataset.json`           | Path to QA dataset JSON                                    |
| `--configs`        | `without-validation with-validation`  | Configs to run (space-separated)                           |
| `--output`         | auto-generated                        | Explicit path for report JSON                              |
| `--output-dir`     | `data/eval`                           | Directory for auto-named report                            |
| `--store-type`     | `neo4j`                               | Backend for Config A retrieval (`neo4j` or `faiss`)        |
| `--index-dir`      | `./output/rag_primekb`                | FAISS index directory (Config A with `--store-type faiss`) |
| `--neo4j-uri`      | `bolt://localhost:7687`               | Neo4j Bolt URI                                             |
| `--neo4j-password` | env `NEO4J_PASSWORD` or `password123` | Neo4j password                                             |
| `--tier`           | `all`                                 | Filter questions by tier (`full`, `partial`, `all`)        |
| `--questions`      | all                                   | Comma-separated question IDs to evaluate                   |
| `--verbose`        | off                                   | Print per-question progress                                |


### Metrics


| Metric                  | Description                                                                                                                                     |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **Gold Recall (mean)**  | Fraction of ALL gold answers found in the prediction (averaged across questions). Primary metric for measuring retrieval + answer completeness. |
| **Answer Rate**         | Fraction of questions that received a non-refusal answer                                                                                        |
| **Hallucination Proxy** | Fraction of mentioned gold answers not grounded in retrieved KG facts                                                                           |
| **Format Failures**     | Count of questions where the LLM did not produce the expected status line after 3 retries                                                       |
| **Mean Latency**        | Average wall-clock seconds per question                                                                                                         |
| **Errors**              | Count of questions that raised an unrecoverable exception                                                                                       |


Metrics are reported **overall**, **per-relation** (indication, contraindication,
off-label use), and **per-tier** (full, partial).

### Thought Block Handling

Some models (e.g. Gemma 4) wrap their reasoning in `<thought>...</thought>` tags.
The system handles this transparently:

- `<thought>` blocks are stripped from verbose log output for readability
- The status line (`status: Accepted/Refused`) is detected anywhere in the
response after stripping thought blocks
- Assessment JSON parsing strips thought blocks before extracting JSON

Set `SHOW_THOUGHT_BLOCKS=true` to include thought blocks in verbose output
for debugging:

```bash
SHOW_THOUGHT_BLOCKS=true python scripts/eval/evaluate.py --configs B --verbose
```

### Format Failure Handling

The evaluator expects each LLM answer to contain a `status:` line. When this is
missing (common with some remote models), the question is retried up to 3 times.
If all retries fail, the question is marked as a **format failure**:

- Not counted as an error or scored in standard metrics.
- Reported separately in the summary with its question IDs.
- Useful for identifying model compatibility issues.

### Output File Naming

Reports are auto-named with these components:

```
eval_N{count}_{config_names}_{tiers}_{YYYYMMDD_HHMMSS}.json
```

Example: `eval_N194_noval_val_f30_p70_20260414_032644.json`

- `N194` -- 194 questions evaluated
- `noval_val` -- Configs B (`without-validation`) and C (`with-validation`).
Aliases used here: `rag` for Config A, `noval` for B, `val` for C.
- `f30_p70` -- 30 % Full, 70 % Partial tier ratio. The 30 is read from
`split_stats.json#full_ratio`; the 70 is `100 - 30`.
- Trailing UTC timestamp guarantees uniqueness when the same configuration
is rerun.

If `split_stats.json` is missing (e.g. evaluating an arbitrary
`qa_dataset.json` that did not come from `split_primekb.py`), the
`f{n}_p{n}` block is omitted from the filename. Override the entire path
with `--output` if you want a stable, non-timestamped report file.

## Automation Script

`run_eval_pipeline.sh` orchestrates the full workflow in a single command.
It handles Neo4j startup, data splitting, QA generation, Neo4j import,
and evaluation.

### Usage

```bash
# Default: 1000 rows, Configs B + C
bash scripts/eval/run_eval_pipeline.sh

# Custom row limit
bash scripts/eval/run_eval_pipeline.sh --max-rows 500

# Skip QA generation (reuse existing qa_dataset.json)
bash scripts/eval/run_eval_pipeline.sh --skip-qa-gen

# Run only Config B
bash scripts/eval/run_eval_pipeline.sh --configs "without-validation"
```

### CLI Reference


| Flag            | Default                              | Description                      |
| --------------- | ------------------------------------ | -------------------------------- |
| `--max-rows`    | `1000` (or env `MAX_ROWS`)           | Cap input CSV rows               |
| `--skip-qa-gen` | off                                  | Reuse existing `qa_dataset.json` |
| `--configs`     | `without-validation with-validation` | Configs to evaluate              |


### Environment Variables

The pipeline and agent read these from the environment (set by `source export_*.sh`):


| Variable                  | Default                          | Description                                                |
| ------------------------- | -------------------------------- | ---------------------------------------------------------- |
| `NEO4J_HOME`              | `~/tools/neo4j-community-5.26.0` | Neo4j installation path                                    |
| `NEO4J_URI`               | `bolt://localhost:7687`          | Neo4j Bolt URI                                             |
| `NEO4J_PASSWORD`          | `password123`                    | Neo4j password                                             |
| `USE_LOCAL_LLM`           | --                               | `true` for local GPU, `false` for API                      |
| `SHOW_THOUGHT_BLOCKS`     | `false`                          | Set to `true` to show `<thought>` blocks in verbose output |
| `OPENAI_MODEL`            | --                               | Primary LLM model name                                     |
| `REMOTE_LLM_MODEL`        | --                               | Validation LLM model name (Config C)                       |
| `OPENAI_HTTP_TIMEOUT`     | `600` (SDK default)              | HTTP timeout (seconds) for primary LLM API calls           |
| `REMOTE_LLM_HTTP_TIMEOUT` | `600` (SDK default)              | HTTP timeout (seconds) for validation LLM API calls        |


### What the Script Does

1. **Prerequisites** -- Checks Python deps, verifies or starts Neo4j, verifies LLM env.
2. **Data prep** -- Downloads `kg_drug_disease.csv` if missing.
3. **Split** -- Runs `split_primekb.py` with drug-aware tier assignment.
4. **QA generation** -- Runs `generate_qa.py` (skippable with `--skip-qa-gen`).
5. **Neo4j import** -- Clears Neo4j and imports `train.csv`.
6. **Evaluate** -- Runs `evaluate.py` for the specified configs.
7. **Summary** -- Prints output file locations.

## LLM Configuration for Evaluation


| Script                      | GPU | Primary LLM          | Validation LLM       | Best for                          |
| --------------------------- | --- | -------------------- | -------------------- | --------------------------------- |
| `export_dual_llm.sh`        | Yes | Local (Qwen 2.5-7B)  | Remote (Gemma)       | Full Config C with local proposer |
| `export_dual_remote_llm.sh` | No  | Remote (Gemma 4-26b) | Remote (Gemma 4-31b) | Config B + C without GPU          |
| `export_google_ai.sh`       | No  | Remote (Gemini)      | --                   | Config A/B, QA generation         |
| `export_local_qwen3.sh`     | Yes | Local (Qwen3)        | --                   | QA generation with thinking mode  |


### API Rate Limits

When using Google AI Studio's free tier for both primary and validation LLMs
(dual-remote mode), expect slower runs due to rate limiting. Config C makes
4-8 API calls per question, so a dataset of 194 questions may take several hours.
Strategies to mitigate this:

- Use `--questions` to evaluate small batches.
- Use `--tier partial` to focus on the most informative tier.
- Run Config B first (fewer API calls) to establish a baseline.

## Troubleshooting

### Common Errors


| Error                                         | Cause                                        | Fix                                                                                                                     |
| --------------------------------------------- | -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `QA dataset not found`                        | `generate_qa.py` hasn't been run             | Run `python scripts/eval/generate_qa.py`                                                                                |
| `Neo4j AuthError`                             | Password mismatch                            | See main README's Neo4j troubleshooting section                                                                         |
| `FormatError` after 3 retries                 | LLM not producing `status:` line             | Try a different model; check `format_failures` in report                                                                |
| `CUDA out of memory`                          | GPU too small or occupied                    | Use `export_dual_remote_llm.sh` for GPU-free mode                                                                       |
| `401 Client Error` (HuggingFace)              | Gated model access                           | Set `HF_TOKEN` in env; accept license at huggingface.co                                                                 |
| `Developer instruction not enabled`           | Model doesn't support system prompts         | Switch to Gemma 4+ or Gemini models                                                                                     |
| `Max retries (3) exceeded: Request timed out` | API rate limit or slow response              | Increase `OPENAI_HTTP_TIMEOUT` / `REMOTE_LLM_HTTP_TIMEOUT` (e.g., 300-600s); re-run failed questions with `--questions` |
| Stale triplets inflating Config C scores      | Config C persisted triplets from a prior run | Clear Neo4j first: `run_eval_clean.sh` or `import_primekb_to_neo4j.py --clear`                                          |


### Re-running Failed Questions

If some questions fail due to API timeouts or format errors, re-run only those:

```bash
# Check which questions failed
python -c "
import json
data = json.load(open('data/eval/eval_report.json'))
for cfg in data['configs']:
    fails = [r['qid'] for r in data['configs'][cfg]['results']
             if r.get('error') or r.get('format_failure')]
    if fails:
        print(f'{cfg}: {\",\".join(fails)}')
"

# Re-evaluate only those
python scripts/eval/evaluate.py --configs C --questions "q0014,q0034" --verbose
```

### Merging Supplementary Results

After re-running failed questions, merge their results into the original report
without modifying it. The merge script replaces error entries with the new
scored results and recomputes aggregate metrics:

```bash
python scripts/eval/merge_eval_reports.py \
  --original data/eval/eval_N194_original.json \
  --supplement data/eval/eval_N18_supplement.json \
  --output data/eval/eval_N194_merged.json
```

The `--supplement` flag accepts glob patterns (e.g., `data/eval/eval_N*_val_*.json`).
The merge prints a before/after comparison for all metrics.

### Clean Evaluation

`run_eval_clean.sh` clears Neo4j, reimports `train.csv`, and runs evaluation
in one step -- useful for ensuring no stale triplets (persisted by Config C
in previous runs) contaminate the results:

```bash
bash scripts/eval/run_eval_clean.sh
```

### Full Evaluation Report

See `[data/eval/EVALUATION_REPORT.md](../../data/eval/EVALUATION_REPORT.md)`
for the comprehensive analysis of all evaluation runs, including per-tier and
per-relation breakdowns, the impact of agent-level fixes, and the definitive
Config B vs Config C comparison.

## Related: Interactive Sampler

For *interactive* (REPL) testing of the agent -- rather than the fully
automated batch evaluation described above -- see
`[scripts/sample_test_questions.py](../sample_test_questions.py)`. It samples
a reproducible handful of drug-disease questions and writes them to
`data/eval/interactive_test_questions.{txt,json}` using the same
`FALLBACK_TEMPLATES` as `generate_qa.py`, so the rendered questions are
consistent with the eval suite.

The sampler shares its inputs with this directory via `--source`:


| `--source` (sampler) | Reads from                        | Best for                                                                                                                             |
| -------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `partial` (default)  | `data/eval/tier_assignments.json` | Dual-LLM validation testing -- partial-tier drugs have guaranteed held-out answers, so the agent's knowledge-gap path fires reliably |
| `train`              | `data/eval/train.csv`             | Sanity-checking RAG against facts already in Neo4j                                                                                   |
| `test`               | `data/eval/test.csv`              | Drug-level knowledge-gap behaviour (full drug held out from Neo4j)                                                                   |
| `kg`                 | `data/kg_drug_disease.csv`        | Anything pre-split (no eval prereqs)                                                                                                 |


`--source train`, `test`, and `partial` (default) all require this
directory's artifacts to have been produced by
`[split_primekb.py](split_primekb.py)`; the sampler exits with a one-line
remediation message if any file is missing.

For `--source partial`, each emitted JSON record additionally carries
`tier`, `kg_triples`, and `held_out_triples` so the user can see at a
glance which gold answers should be retrievable from Neo4j and which
require the dual-LLM proposal/validation path.

Combine the sampler with the `--no-persist` flag on
`[scripts/interactive_agent.py](../interactive_agent.py)` to run dual-LLM
validation in **read-only** mode -- the remote LLM still validates proposed
triplets, but nothing is written back to Neo4j between runs, so the same
seed + same questions exercise the same KG state every time:

```bash
# 1. Sample a reproducible batch (default --source partial)
python scripts/sample_test_questions.py --n 12 --seed 42

# 2. Run the agent without mutating the graph
source export_dual_llm.sh
python scripts/interactive_agent.py --validate --no-persist --verbose

# 3. Paste questions from data/eval/interactive_test_questions.txt
```

See the [main README §5 (Tutorial C)](../../README.md#tutorial-c-sample-test-questions-scriptssample_test_questionspy)
for the full flag reference.