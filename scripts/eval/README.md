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
kg_drug_disease.csv  (PrimeKG raw data)
        │
        ▼
  split_primekb.py   →  train.csv  +  test.csv  +  tier_assignments.json
        │
        ▼
  import_primekb_to_neo4j.py  →  Neo4j (train set only)
        │
        ▼
  generate_qa.py     →  qa_dataset.json  (questions from test set)
        │
        ▼
  evaluate.py        →  eval_N{n}_{configs}_{tiers}_{timestamp}.json
```

## Tiered Train / Test Split

`split_primekb.py` partitions the drug-disease dataset using **drug-aware
sampling** (keeps all triples for each selected drug together) and assigns
drugs to one of two tiers:

| Tier | Default share | What lands in Neo4j | Purpose |
|------|:------------:|---------------------|---------|
| **Full** | 30% | 100% of drug's triples | Baseline -- gold answers fully in KG |
| **Partial** | 70% | Subset of drug's triples (some held out) | Core test -- system must recover gaps |

Only drugs with >= 2 triples per relation are eligible for the Partial tier
(single-triple drugs automatically go to Full). Held-out answers are recorded
in `tier_assignments.json` so evaluate.py can measure recovery.

### Usage

```bash
python scripts/eval/split_primekb.py --input data/kg_drug_disease.csv
```

### CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `data/kg_drug_disease.csv` | Path to drug-disease CSV |
| `--output-dir` | `data/eval` | Directory for output files |
| `--test-ratio` | `0.2` | Fraction of data for the test set |
| `--max-rows` | `1000` | Cap total input rows (0 = no cap); uses drug-aware sampling |
| `--full-ratio` | `0.30` | Fraction of multi-triple test drugs assigned to Full tier |
| `--partial-include` | `0.50` | Fraction of Partial-tier drug triples included in train |
| `--seed` | `42` | Random seed for reproducibility |

### Output Files

| File | Content |
|------|---------|
| `train.csv` | Training triples loaded into Neo4j |
| `test.csv` | Test triples used for QA generation |
| `tier_assignments.json` | Per-drug per-relation mapping with tier, `kg_answers`, and `held_out_answers` |
| `split_stats.json` | Per-relation and per-tier row counts |

## QA Dataset Generation

`generate_qa.py` feeds test triples to an LLM and produces natural-language
questions with gold answers. Questions are **tier-aware**:

- **Full tier**: standard questions (forward and reverse directions)
- **Partial tier**: completeness-demanding questions ("list ALL ...", "comprehensive
  list of every ...") designed to trigger the LLM's `INSUFFICIENT` assessment
  when the KG has only a subset of answers

Both tiers generate questions in forward and reverse directions for data
augmentation and variability.

```bash
# Local LLM
source export_local_qwen3.sh
python scripts/eval/generate_qa.py

# Remote LLM (e.g. Gemma)
source export_dual_remote_llm.sh
python scripts/eval/generate_qa.py --remote-llm
```

The output file doubles as an incremental cache: if the process is interrupted,
re-running it resumes from where it left off (skip `--no-cache` to append).

### CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--test-csv` | `data/eval/test.csv` | Path to test split CSV |
| `--output` | `data/eval/qa_dataset.json` | Output JSON path |
| `--max-questions-per-relation` | `100` | Max forward + reverse questions per relation type |
| `--batch-size` | `15` | Entities per LLM API call |
| `--no-cache` | off | Force regeneration, ignore existing output |
| `--remote-llm` | off | Use remote LLM even when `USE_LOCAL_LLM=true` |
| `--seed` | `42` | Random seed |

### Output Format

Each entry in `qa_dataset.json`:

```json
{
  "id": "q0001",
  "question": "List ALL diseases and conditions for which Mirabegron is contraindicated.",
  "gold_answers": ["hypertension", "overactive bladder"],
  "relation": "contraindication",
  "direction": "forward",
  "entity": "Mirabegron",
  "tier": "partial",
  "kg_answers": ["hypertension"],
  "held_out_answers": ["overactive bladder"]
}
```

## Evaluation

`evaluate.py` runs the KG-RAG system on each question and computes metrics by
comparing system predictions against the gold answers.

### Configurations

| Config | Aliases | Description | Behavior on INSUFFICIENT assessment |
|--------|---------|-------------|-------------------------------------|
| **A** | `rag` | Pure RAG -- vector retrieval only | N/A (no assessment) |
| **B** | `without-validation` | Agent with KG tools, no remote validation | Returns refusal (no answer attempt) |
| **C** | `with-validation` | Agent + dual-LLM validated expansion | Proposes triplets, validates with remote LLM |

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

| Flag | Default | Description |
|------|---------|-------------|
| `--qa-dataset` | `data/eval/qa_dataset.json` | Path to QA dataset JSON |
| `--configs` | `without-validation with-validation` | Configs to run (space-separated) |
| `--output` | auto-generated | Explicit path for report JSON |
| `--output-dir` | `data/eval` | Directory for auto-named report |
| `--store-type` | `neo4j` | Backend for Config A retrieval (`neo4j` or `faiss`) |
| `--index-dir` | `./output/rag_primekb` | FAISS index directory (Config A with `--store-type faiss`) |
| `--neo4j-uri` | `bolt://localhost:7687` | Neo4j Bolt URI |
| `--neo4j-password` | env `NEO4J_PASSWORD` or `password123` | Neo4j password |
| `--tier` | `all` | Filter questions by tier (`full`, `partial`, `all`) |
| `--questions` | all | Comma-separated question IDs to evaluate |
| `--verbose` | off | Print per-question progress |

### Metrics

| Metric | Description |
|--------|-------------|
| **Gold Recall (mean)** | Fraction of ALL gold answers found in the prediction (averaged across questions). Primary metric for measuring retrieval + answer completeness. |
| **Answer Rate** | Fraction of questions that received a non-refusal answer |
| **Hallucination Proxy** | Fraction of mentioned gold answers not grounded in retrieved KG facts |
| **Format Failures** | Count of questions where the LLM did not produce the expected status line after 3 retries |
| **Mean Latency** | Average wall-clock seconds per question |
| **Errors** | Count of questions that raised an unrecoverable exception |

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
- `noval_val` -- Configs B (no validation) and C (validation)
- `f30_p70` -- 30% Full, 70% Partial tier split
- Timestamp for uniqueness

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

| Flag | Default | Description |
|------|---------|-------------|
| `--max-rows` | `1000` (or env `MAX_ROWS`) | Cap input CSV rows |
| `--skip-qa-gen` | off | Reuse existing `qa_dataset.json` |
| `--configs` | `without-validation with-validation` | Configs to evaluate |

### Environment Variables

The pipeline and agent read these from the environment (set by `source export_*.sh`):

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_HOME` | `~/tools/neo4j-community-5.26.0` | Neo4j installation path |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt URI |
| `NEO4J_PASSWORD` | `password123` | Neo4j password |
| `USE_LOCAL_LLM` | -- | `true` for local GPU, `false` for API |
| `SHOW_THOUGHT_BLOCKS` | `false` | Set to `true` to show `<thought>` blocks in verbose output |
| `OPENAI_MODEL` | -- | Primary LLM model name |
| `REMOTE_LLM_MODEL` | -- | Validation LLM model name (Config C) |

### What the Script Does

1. **Prerequisites** -- Checks Python deps, verifies or starts Neo4j, verifies LLM env.
2. **Data prep** -- Downloads `kg_drug_disease.csv` if missing.
3. **Split** -- Runs `split_primekb.py` with drug-aware tier assignment.
4. **QA generation** -- Runs `generate_qa.py` (skippable with `--skip-qa-gen`).
5. **Neo4j import** -- Clears Neo4j and imports `train.csv`.
6. **Evaluate** -- Runs `evaluate.py` for the specified configs.
7. **Summary** -- Prints output file locations.

## LLM Configuration for Evaluation

| Script | GPU | Primary LLM | Validation LLM | Best for |
|--------|:---:|-------------|----------------|----------|
| `export_dual_llm.sh` | Yes | Local (Qwen 2.5-7B) | Remote (Gemma) | Full Config C with local proposer |
| `export_dual_remote_llm.sh` | No | Remote (Gemma 4-26b) | Remote (Gemma 4-31b) | Config B + C without GPU |
| `export_google_ai.sh` | No | Remote (Gemini) | -- | Config A/B, QA generation |
| `export_local_qwen3.sh` | Yes | Local (Qwen3) | -- | QA generation with thinking mode |

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

| Error | Cause | Fix |
|-------|-------|-----|
| `QA dataset not found` | `generate_qa.py` hasn't been run | Run `python scripts/eval/generate_qa.py` |
| `Neo4j AuthError` | Password mismatch | See main README's Neo4j troubleshooting section |
| `FormatError` after 3 retries | LLM not producing `status:` line | Try a different model; check `format_failures` in report |
| `CUDA out of memory` | GPU too small or occupied | Use `export_dual_remote_llm.sh` for GPU-free mode |
| `401 Client Error` (HuggingFace) | Gated model access | Set `HF_TOKEN` in env; accept license at huggingface.co |
| `Developer instruction not enabled` | Model doesn't support system prompts | Switch to Gemma 4+ or Gemini models |

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
