# Changelog

## [Unreleased] - 2026-04-23

### Changed -- evaluation suite redesign (dataset preparation & splitting)

- **Two-tier (Full / Partial) drug-aware split** in
  `scripts/eval/split_primekb.py`. The previous three-tier
  (Full / Partial / Zero) design was replaced because the Zero tier produced
  uninformative all-or-nothing scores and the original Partial tier kept at
  least one gold triple in train (`max(1, ...)`), which made it equivalent to
  Full for single-triple drugs. The new design:

  1. **Stage 1 -- drug-aware row sampling** (`--max-rows`, default `1000`):
     keeps complete drug groups together so multi-triple drugs survive
     sub-sampling. Naive row-level sampling would shred most drugs to a
     single triple per relation, making the Partial tier infeasible.
  2. **Stage 2 -- per-relation entity-disjoint split** (`--test-ratio`,
     default `0.20`): no drug appears in both train and test for the same
     relation, removing same-drug leakage of gold answers.
  3. **Stage 3 -- tier assignment** (`--full-ratio`, default `0.30`):
     multi-triple test drugs are split 30 / 70 between Full and Partial;
     single-triple drugs are forced to Full because they have nothing to
     hold out.
  4. **Stage 4 -- per-drug held-out selection** (`--partial-include`,
     default `0.50`): for each Partial drug,
     `n_keep = int(len * partial_include)` triples stay in `train.csv`
     and the rest are recorded as `held_out_answers`. The `int()`
     truncation guarantees that any Partial drug with ≥ 2 triples has at
     least one held-out answer.

- **`tier_assignments.json` schema** now records the full evaluation
  ground truth per `(drug, relation)`:

  ```jsonc
  { "<drug>": { "<relation>": {
      "tier":             "full" | "partial",
      "kg_answers":       [...],   // present in train.csv (retrievable)
      "held_out_answers": [...]    // only in test.csv (recovery target)
  }}}
  ```

  Each `qa_dataset.json` record carries a copy of these fields so
  `evaluate.py` can score Gold Recall and Hallucination without re-reading
  the splits.

- **Tier-aware QA generation** in `scripts/eval/generate_qa.py`. Each
  `(entity, relation, direction)` is routed to one of two prompt sets:
  `RELATION_CONTEXT` for Full-tier entities (standard phrasing) and
  `PARTIAL_RELATION_CONTEXT` for Partial-tier entities (completeness-
  demanding phrasing -- "List **ALL** ...", "Provide a comprehensive list
  of every ..."). The Partial phrasing is engineered to trip the agent's
  INSUFFICIENT assessment, which is what makes the Config B vs Config C
  recovery comparison sharp. Reverse-direction questions inherit the tier
  of the most-restrictive contributing drug
  (`partial` if any contributing drug is Partial, else `full`).

- **`generate_qa.py` is now resumable.** The output file (`qa_dataset.json`)
  doubles as an incremental cache: completed `(entity, relation, direction)`
  triples are skipped on restart and IDs are reassigned contiguously after
  the final batch. `--no-cache` forces a clean rebuild. If a single batch
  fails or returns malformed JSON, the script falls back to a deterministic
  template instead of dropping the entity.

- **Gold Recall** is the new primary metric, replacing Exact Match
  (binary) and Token F1. Defined as
  `count(gold_answers found in prediction) / count(all gold_answers)`,
  reported as a per-question float and averaged for the headline number.
  It coincides with the old KG Recall on Full-tier questions and is
  strictly more informative on Partial-tier questions because it counts
  recovered held-out answers.

- **Auto-named report files** in `evaluate.py`. Reports are written to
  `data/eval/eval_N{count}_{configs}_f{full_pct}_p{partial_pct}_{YYYYMMDD_HHMMSS}.json`
  -- e.g. `eval_N194_noval_val_f30_p70_20260417_055917.json`. The
  `f30_p70` block is derived from `split_stats.json#full_ratio`
  (`fr = round(full_ratio * 100)`, `pr = 100 - fr`); the block is
  omitted entirely if no `split_stats.json` is found next to the QA
  dataset. `--output` overrides the auto-name with an explicit path.

### Added -- evaluation suite tooling

- **`scripts/eval/run_eval_pipeline.sh`** -- one-shot orchestration of the
  full pipeline (Neo4j start → split → QA generation → Neo4j import →
  evaluation), with `--max-rows`, `--skip-qa-gen`, and `--configs` flags.

- **`scripts/eval/run_eval_clean.sh`** -- "clean re-eval" helper that
  clears Neo4j, reimports `train.csv`, and re-runs evaluation in a single
  step. Useful when comparing Config B vs Config C without leftover
  triplets persisted by previous Config C runs.

- **`scripts/eval/merge_eval_reports.py`** -- merges supplementary
  evaluation results into an existing report. Used to recover from
  per-question API timeouts: re-run only the failed question IDs with
  `evaluate.py --questions q0014,q0034,...`, then merge the supplementary
  JSON into the original. The merge replaces error entries with the new
  scored results, recomputes overall / per-relation / per-tier metrics,
  and writes a new file (the original is never modified).

- **`scripts/eval/evaluate.py` per-question re-runs.** New `--questions`
  flag accepts a comma-separated list of question IDs to evaluate, and
  `--tier {full,partial,all}` filters by tier. Together with
  `merge_eval_reports.py` this enables targeted recovery from API
  rate-limit failures.

- **`scripts/sample_test_questions.py`** -- deterministic sampler that
  produces a small, reproducible batch of drug-focused questions for
  interactive REPL testing. Reads from one of four sources:
  `data/kg_drug_disease.csv`, `train.csv`, `test.csv`, or
  `tier_assignments.json` (Partial entries only). Pair with
  `interactive_agent.py --validate --no-persist --expand-mode auto` for
  reproducible read-only dual-LLM testing.

- **`SHOW_THOUGHT_BLOCKS` env var** (default `false`). When `true`,
  verbose evaluation logs include `<thought>...</thought>` reasoning
  blocks emitted by some models (e.g. Gemma 4). The blocks are still
  stripped before answer parsing in either case, so this only controls
  log readability.

### Initial evaluation suite (superseded by the redesign above) - 2026-04-06

The original implementation introduced the suite with a 3-tier
(Full / Partial / Zero) split, Exact Match + Token F1 metrics, and a
fixed `data/eval/eval_results.json` output path. All three are
superseded by the redesign documented above. See
[`scripts/eval/README.md`](../scripts/eval/README.md) for the current
CLI reference and [`data/eval/EVALUATION_REPORT.md`](../data/eval/EVALUATION_REPORT.md)
for the methodology and results discussion.

### Added -- new components from the original 2026-04-06 release

- **Drug-disease evaluation suite** (`scripts/eval/`) -- end-to-end pipeline
  for quantitatively evaluating the KG-RAG system on the PrimeKG drug-disease
  subset across configurations A (pure RAG), B (agent without validation),
  and C (agent + dual-LLM validation). See above for the current metrics
  and output layout.

- **`scripts/eval/split_primekb.py`** -- entity-disjoint train/test split.
  Groups edges by drug entity and ensures no drug appears in both splits
  for the same relation type. The four-stage drug-aware tiered design
  documented above replaced the original stratified split.

- **`scripts/eval/generate_qa.py`** -- LLM-generated QA dataset from test
  triplets. Supports local LLM (default, via `export_local_qwen3.sh`) or
  remote API (`--remote-llm` with `export_google_ai.sh`). Now tier-aware
  (see above) with deterministic gold-answer derivation and an atomic
  resume cache; robust JSON parsing with fallback templates; batched API
  calls with configurable `--batch-size`.

- **`export_local_qwen3.sh`** -- local LLM configuration for the eval QA
  generation pipeline. Defaults to Qwen/Qwen2.5-7B-Instruct with 4-bit
  quantization.

- **`scripts/eval/evaluate.py`** -- runs the system against the QA dataset
  in configs A (pure RAG), B (agent), and C (agent + dual-LLM validation).
  Computes overall, per-relation, and per-tier metrics. Reports are
  auto-named per the convention documented above (the original fixed
  `data/eval/eval_results.json` path is no longer used).

- **Drug-disease subset extraction** in `scripts/download_primekb.py` -- new
  `--extract` flag (default `drug-disease`) filters `kg.csv` to drug-disease
  edges only and saves to `data/kg_drug_disease.csv`. Use `--no-extract` to
  skip.

## [Unreleased] - 2026-03-03

### Added

- **LLM-based automatic predicate selection** (`--auto_filter`) -- before
  retrieval, the LLM inspects the user's query and the set of available
  relationship types in the index, then selects the 1-3 most relevant
  predicates.  This replaces the need for manual `--predicate_filter` flags.

- **`rag/predicate_selector.py`** -- `PredicateSelector` class that prompts
  the LLM (local or API, via `openai_chat_completion`) and parses the JSON
  response into a list of predicate strings.

- **Metadata-based post-filtering** on search results:
  - `FaissStore.search_filtered()` -- over-fetches from FAISS, then filters by
    `predicate` and/or `node_type` metadata before returning top-k.
  - `KGRagIndexer.get_unique_predicates()` -- returns all unique predicate
    values present in the loaded index.

- **CLI flags** on `scripts/index_rag.py` (search & generate modes):
  - `--predicate_filter` -- explicit comma-separated predicate filter
  - `--node_type_filter` -- explicit comma-separated node-type filter
  - `--auto_filter` -- enable LLM-driven automatic predicate selection

- **Neo4j backend support in CLI** (`--store_type neo4j`) -- `scripts/index_rag.py`
  now exposes the existing `KGRagIndexer` Neo4j backend through CLI flags.
  Users can index, search, and generate using Neo4j as the storage backend
  instead of FAISS. Connection parameters are resolved from CLI flags or
  environment variables (`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`).

- **CLI flags** on `scripts/index_rag.py` (all modes):
  - `--store_type {faiss,neo4j}` -- storage backend selection (default: faiss)
  - `--neo4j_uri` -- Neo4j Bolt URI (default: env `NEO4J_URI` or `bolt://localhost:7687`)
  - `--neo4j_user` -- Neo4j username (default: env `NEO4J_USER` or `neo4j`)
  - `--neo4j_password` -- Neo4j password (default: env `NEO4J_PASSWORD`)
  - `--neo4j_database` -- Neo4j database name (default: `neo4j`)

### Previously added

- **PrimeKG dataset support** -- the RAG pipeline now accepts PrimeKG's `kg.csv`
  (~8.1 M biomedical relationships) as a data source alongside the existing EDC
  `canon_kg.txt` format.

- **`rag/primekb_loader.py`** -- `PrimeKBLoader` class that reads PrimeKG CSV,
  maps columns (`x_name` -> subject, `display_relation` -> predicate,
  `y_name` -> object), and produces `Triplet` objects. Supports filtering by
  `node_types`, `relation_types`, and `max_rows`.

- **`get_loader()` factory** in `rag/triplet_loader.py` -- auto-detects data
  format (`edc` / `primekb`) by file extension and returns the appropriate
  loader.

- **`scripts/download_primekb.py`** -- downloads `kg.csv` (and optionally
  `drug_features.csv`, `disease_features.csv`) from Harvard Dataverse with
  progress display and dataset summary.

- **`scripts/import_primekb_to_neo4j.py`** -- imports PrimeKG into Neo4j as a
  native graph using batched UNWIND queries. Creates node labels from
  `x_type`/`y_type` and relationship types from `display_relation`.

- **`rag/edc/schemas/primekb_schema.csv`** -- PrimeKG relation definitions for
  the `TripletExpander` schema-constrained expansion.

- **CLI flags** on `scripts/index_rag.py`:
  - `--format {auto,edc,primekb}` -- select input format (default: auto-detect)
  - `--node_types` -- comma-separated PrimeKG node type filter
  - `--relation_types` -- comma-separated PrimeKG relation type filter
  - `--max_rows` -- limit CSV rows loaded

### Changed

- **`KGRagIndexer.index_from_path()`** now accepts `fmt`, `node_types`,
  `relation_types`, and `max_rows` parameters. Internally delegates to
  `get_loader()` instead of directly instantiating `TripletLoader`.

- **`rag/__init__.py`** exports `PrimeKBLoader` and `get_loader`.

- **`README.md`** updated with PrimeKG download/indexing/Neo4j-import
  instructions and revised project structure.
