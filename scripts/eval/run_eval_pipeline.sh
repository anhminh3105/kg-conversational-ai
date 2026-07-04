#!/usr/bin/env bash
#
# End-to-end evaluation pipeline: data splitting -> QA generation -> evaluation
#
# Runs two configs by default:
#   without-validation  – Agent without remote LLM validation
#   with-validation     – Agent with remote LLM validation (propose + fact-check)
#
# Usage:
#   bash scripts/eval/run_eval_pipeline.sh
#   bash scripts/eval/run_eval_pipeline.sh --max-rows 500
#   bash scripts/eval/run_eval_pipeline.sh --skip-qa-gen
#   bash scripts/eval/run_eval_pipeline.sh --configs "without-validation"
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ── Defaults ──────────────────────────────────────────────────────────────────
MAX_ROWS="${MAX_ROWS:-1000}"
NEO4J_HOME="${NEO4J_HOME:-$HOME/tools/neo4j-community-5.26.0}"
NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-password123}"
EVAL_DIR="data/eval"
CONFIGS="without-validation with-validation"
SKIP_QA_GEN=0
export SHOW_THOUGHT_BLOCKS="${SHOW_THOUGHT_BLOCKS:-false}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-rows) MAX_ROWS="$2"; shift 2 ;;
        --skip-qa-gen) SKIP_QA_GEN=1; shift ;;
        --configs) CONFIGS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

log() { echo -e "\n\033[1;36m==> $1\033[0m"; }
warn() { echo -e "\033[1;33mWARN: $1\033[0m"; }
fail() { echo -e "\033[1;31mERROR: $1\033[0m"; exit 1; }

# ── 1. Check prerequisites ───────────────────────────────────────────────────
log "Checking prerequisites"

if ! command -v python &>/dev/null; then
    fail "python not found in PATH"
fi

python -c "import pandas, numpy" 2>/dev/null || fail "Missing Python deps: pip install pandas numpy"

# Neo4j check
check_neo4j() {
    python -c "
from neo4j import GraphDatabase
d = GraphDatabase.driver('$NEO4J_URI', auth=('neo4j','$NEO4J_PASSWORD'))
d.verify_connectivity()
d.close()
print('OK')
" 2>/dev/null
}

if ! check_neo4j | grep -q OK; then
    log "Neo4j not reachable at $NEO4J_URI -- attempting to start"
    if [[ -x "$NEO4J_HOME/bin/neo4j" ]]; then
        "$NEO4J_HOME/bin/neo4j" start
        echo "Waiting for Neo4j to start..."
        for i in $(seq 1 30); do
            sleep 2
            if check_neo4j | grep -q OK; then
                echo "Neo4j is ready."
                break
            fi
            if [[ $i -eq 30 ]]; then
                fail "Neo4j did not start within 60 seconds"
            fi
        done
    else
        fail "Neo4j not running and NEO4J_HOME ($NEO4J_HOME) not found. Set NEO4J_HOME or start Neo4j manually."
    fi
fi

echo "Neo4j: OK at $NEO4J_URI"

# LLM check -- source config if not already set
if [[ -z "${USE_LOCAL_LLM:-}" && -z "${OPENAI_API_KEY:-}" && -z "${OPENAI_KEY:-}" ]]; then
    if [[ -f "$PROJECT_ROOT/export_dual_remote_llm.sh" ]]; then
        log "Sourcing export_dual_remote_llm.sh for LLM configuration"
        source "$PROJECT_ROOT/export_dual_remote_llm.sh"
    elif [[ -f "$PROJECT_ROOT/export_dual_llm.sh" ]]; then
        log "Sourcing export_dual_llm.sh for LLM configuration"
        source "$PROJECT_ROOT/export_dual_llm.sh"
    else
        fail "No LLM configured. Run: source export_dual_remote_llm.sh (or export_dual_llm.sh)"
    fi
fi
echo "LLM env: OK (USE_LOCAL_LLM=${USE_LOCAL_LLM:-unset}, PRIMARY=${OPENAI_MODEL:-unset}, VALIDATION=${REMOTE_LLM_MODEL:-unset})"

# ── 2. Data preparation ─────────────────────────────────────────────────────
log "Step 1: Ensuring drug-disease CSV exists"

INPUT_CSV="data/kg_drug_disease.csv"
if [[ ! -f "$INPUT_CSV" ]]; then
    log "Downloading PrimeKG drug-disease subset"
    python scripts/download_primekg.py --skip_summary
fi

if [[ ! -f "$INPUT_CSV" ]]; then
    fail "Expected $INPUT_CSV after download, but it does not exist."
fi
echo "Input: $INPUT_CSV ($(wc -l < "$INPUT_CSV") lines)"

# ── 3. Split with tiers ─────────────────────────────────────────────────────
log "Step 2: Splitting data with tier assignment (max_rows=$MAX_ROWS)"

python scripts/eval/split_primekg.py \
    --input "$INPUT_CSV" \
    --output-dir "$EVAL_DIR" \
    --max-rows "$MAX_ROWS"

echo "Train: $(wc -l < "$EVAL_DIR/train.csv") lines"
echo "Test:  $(wc -l < "$EVAL_DIR/test.csv") lines"

if [[ ! -f "$EVAL_DIR/tier_assignments.json" ]]; then
    fail "tier_assignments.json not created by split script"
fi

# ── 4. QA generation ────────────────────────────────────────────────────────
if [[ "$SKIP_QA_GEN" -eq 1 && -f "$EVAL_DIR/qa_dataset.json" ]]; then
    log "Step 3: Skipping QA generation (--skip-qa-gen, using existing dataset)"
else
    log "Step 3: Generating QA dataset from test split"
    python scripts/eval/generate_qa.py \
        --test-csv "$EVAL_DIR/test.csv" \
        --output "$EVAL_DIR/qa_dataset.json" \
        --no-cache
fi

QA_COUNT=$(python -c "import json; print(len(json.load(open('$EVAL_DIR/qa_dataset.json'))))")
echo "QA dataset: $QA_COUNT questions"

# ── 5. Import into Neo4j & evaluate ─────────────────────────────────────────
log "Step 4: Importing train.csv into Neo4j (with --clear)"

python scripts/import_primekg_to_neo4j.py \
    --input "$EVAL_DIR/train.csv" \
    --uri "$NEO4J_URI" \
    --password "$NEO4J_PASSWORD" \
    --clear

# ── 6. Run evaluation ───────────────────────────────────────────────────────
log "Step 5: Running evaluation (configs: $CONFIGS)"

python scripts/eval/evaluate.py \
    --qa-dataset "$EVAL_DIR/qa_dataset.json" \
    --configs $CONFIGS \
    --output-dir "$EVAL_DIR" \
    --neo4j-uri "$NEO4J_URI" \
    --neo4j-password "$NEO4J_PASSWORD" \
    --verbose

# ── 7. Summary ──────────────────────────────────────────────────────────────
log "Pipeline complete"
echo ""
echo "Output files:"
echo "  $EVAL_DIR/train.csv              - KG data loaded into Neo4j"
echo "  $EVAL_DIR/test.csv               - Test triples"
echo "  $EVAL_DIR/tier_assignments.json   - Entity tier mapping"
echo "  $EVAL_DIR/qa_dataset.json         - QA dataset with tier tags"
echo "  $EVAL_DIR/eval_N*_*.json         - Full evaluation report (timestamped)"
echo ""
echo "To re-run evaluation only (skip data prep):"
echo "  python scripts/eval/evaluate.py --configs $CONFIGS --verbose"
echo ""
echo "Environment:"
echo "  SHOW_THOUGHT_BLOCKS=$SHOW_THOUGHT_BLOCKS  (set to 'true' to show <thought> blocks in logs)"
