# Evaluation Suite -- PrimeKG Drug-Disease

Automated evaluation of the KG-RAG system using the PrimeKG drug-disease subset
(indication, contraindication, off-label use). Compares three system
configurations under controlled knowledge-availability conditions (Full / Partial / Zero tiers).

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

`split_primekb.py` partitions the drug-disease dataset so that test-set
entities appear in the training KG at varying levels:

| Tier | Default share | What lands in Neo4j | Purpose |
|------|:------------:|---------------------|---------|
| **Full** | 10% | 100% of entity triples | Baseline -- gold answers fully available |
| **Partial** | 80% | 50% of entity triples | Core test -- system must reason over gaps |
| **Zero** | 10% | 0 triples | Stress test -- entities absent entirely |

This setup evaluates whether the system (and specifically Config C's validation
loop) can recover useful facts when the KG is incomplete or empty for a given
entity, rather than only testing recall of data already stored.

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
| `--max-rows` | `1000` | Cap total input rows (0 = no cap) |
| `--full-ratio` | `0.10` | Fraction of test entities assigned to Full tier |
| `--zero-ratio` | `0.10` | Fraction of test entities assigned to Zero tier |
| `--partial-include` | `0.50` | Fraction of Partial-tier entity triples included in train |
| `--seed` | `42` | Random seed for reproducibility |

### Output Files

| File | Content |
|------|---------|
| `train.csv` | Training triples loaded into Neo4j |
| `test.csv` | Test triples used for QA generation |
| `tier_assignments.json` | Entity-to-tier mapping (`{entity: "full"|"partial"|"zero"}`) |
| `split_stats.json` | Per-relation and per-tier row counts |

## QA Dataset Generation

`generate_qa.py` feeds test triples to an LLM and produces natural-language
questions with gold answers. Each question records its entity tier from the
split, so evaluation can be broken down by knowledge availability.

```bash
# Local LLM
source export_local_qwen3.sh
python scripts/eval/generate_qa.py

# Remote LLM (e.g. Gemini)
source export_google_ai.sh
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
  "qid": "q0001",
  "question": "What drugs are indicated for diabetes mellitus?",
  "gold_answers": ["metformin", "insulin glargine"],
  "relation": "indication",
  "direction": "forward",
  "entity": "diabetes mellitus",
  "tier": "partial"
}
```

## Evaluation

`evaluate.py` runs the KG-RAG system on each question and computes metrics by
comparing system predictions against the gold answers.

### Configurations

| Config | Aliases | Description | LLM calls per question |
|--------|---------|-------------|:----------------------:|
| **A** | `rag` | Pure RAG -- vector retrieval only | 1 |
| **B** | `without-validation` | Agent with KG tools, no remote validation | 2-4 |
| **C** | `with-validation` | Agent + dual-LLM validated expansion | 4-8 |

Config names and their single-letter aliases (`A`, `B`, `C`) are interchangeable
on the command line.

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
| `--tier` | `all` | Filter questions by tier (`full`, `partial`, `zero`, `all`) |
| `--questions` | all | Comma-separated question IDs to evaluate |
| `--verbose` | off | Print per-question progress |

### Metrics

| Metric | Description |
|--------|-------------|
| **Exact Match** | Fraction of predictions that contain at least one gold answer (case-insensitive) |
| **KG Recall** | Fraction of gold answers found in the system's retrieved KG triples |
| **Answer Rate** | Fraction of questions that received a non-refusal answer |
| **Hallucination Proxy** | Fraction of answered questions where no gold answer appears in the prediction |
| **Format Failures** | Count of questions where the LLM did not produce the expected status line after 3 retries |
| **Mean Latency** | Average wall-clock seconds per question |
| **Errors** | Count of questions that raised an unrecoverable exception |

Metrics are reported **overall**, **per-relation** (indication, contraindication,
off-label use), and **per-tier** (full, partial, zero).

### Format Failure Handling

The evaluator expects each LLM answer to contain a `Status:` line. When this is
missing (common with some remote models), the question is retried up to 3 times.
If all retries fail, the question is marked as a **format failure**:

- Not counted as an error or scored in standard metrics.
- Reported separately in the summary with its question IDs.
- Useful for identifying model compatibility issues.

### Output File Naming

Reports are auto-named with these components:

```
eval_N{count}_{config_names}_{tier_ratios}_{YYYYMMDD_HHMMSS}.json
```

Example: `eval_N84_noval_val_f10_p80_z10_20260412_181739.json`

- `N84` -- 84 questions evaluated
- `noval_val` -- Configs B (no validation) and C (validation)
- `f10_p80_z10` -- 10% Full, 80% Partial, 10% Zero tier split
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

The script reads these from the environment (set by `source export_*.sh`):

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_HOME` | `~/tools/neo4j-community-5.26.0` | Neo4j installation path |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt URI |
| `NEO4J_PASSWORD` | `password123` | Neo4j password |
| `USE_LOCAL_LLM` | -- | `true` for local GPU, `false` for API |

### What the Script Does

1. **Prerequisites** -- Checks Python deps, verifies or starts Neo4j, verifies LLM env.
2. **Data prep** -- Downloads `kg_drug_disease.csv` if missing.
3. **Split** -- Runs `split_primekb.py` with tiered assignment.
4. **QA generation** -- Runs `generate_qa.py` (skippable with `--skip-qa-gen`).
5. **Neo4j import** -- Clears Neo4j and imports `train.csv`.
6. **Evaluate** -- Runs `evaluate.py` for the specified configs.
7. **Summary** -- Prints output file locations.

## LLM Configuration for Evaluation

| Script | GPU | Primary LLM | Validation LLM | Best for |
|--------|:---:|-------------|----------------|----------|
| `export_dual_llm.sh` | Yes | Local (Qwen 2.5-7B) | Remote (Gemini) | Full Config C with local proposer |
| `export_dual_remote_llm.sh` | No | Remote (Gemini) | Remote (Gemini) | Config C without GPU access |
| `export_google_ai.sh` | No | Remote (Gemini) | -- | Config A/B, QA generation |
| `export_local_qwen3.sh` | Yes | Local (Qwen3) | -- | QA generation with thinking mode |

### API Rate Limits

When using Google AI Studio's free tier for both primary and validation LLMs
(dual-remote mode), expect slower runs due to rate limiting. Config C makes
4-8 API calls per question, so a dataset of 84 questions may take several hours.
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
| `FormatError` after 3 retries | LLM not producing `Status:` line | Try a different model; check `format_failures` in report |
| `CUDA out of memory` | GPU too small or occupied | Use `export_dual_remote_llm.sh` for GPU-free mode |
| `401 Client Error` (HuggingFace) | Gated model access | Set `HF_TOKEN` in env; accept license at huggingface.co |
| `Developer instruction not enabled` | Model doesn't support system prompts | Switch to a model that supports system prompts (e.g. Gemini, Qwen) |

### Re-running Failed Questions

If some questions fail due to API timeouts or format errors, re-run only those:

```bash
# Check which questions failed
python -c "
import json
data = json.load(open('data/eval/eval_N84_noval_val_f10_p80_z10_20260412_181739.json'))
for cfg in data['configs']:
    fails = [r['qid'] for r in data['configs'][cfg]['results']
             if r.get('error') or r.get('format_failure')]
    if fails:
        print(f'{cfg}: {\",\".join(fails)}')
"

# Re-evaluate only those
python scripts/eval/evaluate.py --configs C --questions "q0014,q0034" --verbose
```
