#!/usr/bin/env bash
#
# Neo4j clear + evaluation only (no CSV download, split, or QA generation).
# Expects existing train split and QA dataset under the eval directory.
#
# Runs two configs by default:
#   without-validation  – Agent without remote LLM validation
#   with-validation     – Agent with remote LLM validation (propose + fact-check)
#
# Usage:
#   bash scripts/eval/run_eval_clean.sh
#   bash scripts/eval/run_eval_clean.sh --configs "without-validation"
#   bash scripts/eval/run_eval_clean.sh --eval-dir data/eval --qa-dataset data/eval/qa_dataset.json
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ── Defaults ──────────────────────────────────────────────────────────────────
NEO4J_HOME="${NEO4J_HOME:-$HOME/tools/neo4j-community-5.26.0}"
NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-password123}"
EVAL_DIR="data/eval"
CONFIGS="without-validation with-validation"
QA_DATASET=""
export SHOW_THOUGHT_BLOCKS="${SHOW_THOUGHT_BLOCKS:-false}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --configs) CONFIGS="$2"; shift 2 ;;
        --eval-dir) EVAL_DIR="$2"; shift 2 ;;
        --qa-dataset) QA_DATASET="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$QA_DATASET" ]]; then
    QA_DATASET="$EVAL_DIR/qa_dataset.json"
fi

log() { echo -e "\n\033[1;36m==> $1\033[0m"; }
fail() { echo -e "\033[1;31mERROR: $1\033[0m"; exit 1; }

# ── 1. Check prerequisites ───────────────────────────────────────────────────
log "Checking prerequisites"

if ! command -v python &>/dev/null; then
    fail "python not found in PATH"
fi

python -c "import pandas, numpy" 2>/dev/null || fail "Missing Python deps: pip install pandas numpy"

TRAIN_CSV="$EVAL_DIR/train.csv"
if [[ ! -f "$TRAIN_CSV" ]]; then
    fail "Missing $TRAIN_CSV — run the full pipeline or split/import first."
fi
if [[ ! -f "$QA_DATASET" ]]; then
    fail "Missing QA dataset: $QA_DATASET — generate it or pass --qa-dataset."
fi

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

QA_COUNT=$(python -c "import json, sys; print(len(json.load(open(sys.argv[1]))))" "$QA_DATASET")
echo "Eval dir: $EVAL_DIR"
echo "Train CSV: $TRAIN_CSV ($(wc -l < "$TRAIN_CSV") lines)"
echo "QA dataset: $QA_DATASET ($QA_COUNT questions)"

# ── 2. Import into Neo4j (clear) ──────────────────────────────────────────────
log "Importing train.csv into Neo4j (with --clear)"

python scripts/import_primekb_to_neo4j.py \
    --input "$TRAIN_CSV" \
    --uri "$NEO4J_URI" \
    --password "$NEO4J_PASSWORD" \
    --clear

# ── 3. Run evaluation ─────────────────────────────────────────────────────────
log "Running evaluation (configs: $CONFIGS)"

python scripts/eval/evaluate.py \
    --qa-dataset "$QA_DATASET" \
    --configs $CONFIGS \
    --output-dir "$EVAL_DIR" \
    --neo4j-uri "$NEO4J_URI" \
    --neo4j-password "$NEO4J_PASSWORD" \
    --verbose

# ── 4. Summary ────────────────────────────────────────────────────────────────
log "Neo4j clear + evaluation complete"
echo ""
echo "Output files:"
echo "  $EVAL_DIR/eval_N*_*.json         - Full evaluation report (timestamped)"
echo ""
echo "Split / QA generation were not run. To re-run evaluation only (no Neo4j reload):"
echo "  python scripts/eval/evaluate.py --qa-dataset \"$QA_DATASET\" --configs $CONFIGS --verbose"
echo ""
echo "Environment:"
echo "  SHOW_THOUGHT_BLOCKS=$SHOW_THOUGHT_BLOCKS  (set to 'true' to show <thought> blocks in logs)"
