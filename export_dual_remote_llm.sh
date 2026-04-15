#!/bin/bash
# =============================================================================
# Dual-Remote-LLM Configuration Script
# =============================================================================
# Routes BOTH the primary LLM and the validation LLM through remote APIs,
# so no local GPU is required. Useful when the GPU is unavailable or occupied.
#
#   - Primary LLM:    Queries, knowledge assessment, answer generation
#   - Validation LLM: Triplet fact-checking (Config C)
#
# Usage: source export_dual_remote_llm.sh
#
# Prerequisites:
#   - API key for remote LLM service (Google AI, OpenAI, or SambaNova)
#   - No GPU required
# =============================================================================

# =============================================================================
# PRIMARY LLM CONFIGURATION (remote — no GPU needed)
# =============================================================================
export USE_LOCAL_LLM=false

# OpenAI-compatible API used for primary generation
# (Google AI Studio via OpenAI compatibility layer)
export OPENAI_KEY=""
export OPENAI_API_BASE="https://generativelanguage.googleapis.com/v1beta/openai/"
export OPENAI_MODEL="gemma-4-26b-a4b-it"

# Per-request HTTP timeout for the OpenAI Python SDK (seconds). Without this, the
# SDK may wait a very long time on stalled connections. Use 60 for a 1-minute cap.
export OPENAI_HTTP_TIMEOUT=60
export REMOTE_LLM_HTTP_TIMEOUT=60

# Embedder Model — small enough to run on CPU or a shared GPU
export LOCAL_EMBEDDER_MODEL="BAAI/bge-small-en-v1.5"

# =============================================================================
# VALIDATION LLM CONFIGURATION (remote — triplet fact-checking)
# =============================================================================
export REMOTE_LLM_API_KEY=""
export REMOTE_LLM_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
export REMOTE_LLM_MODEL="gemma-4-31b-it"

# =============================================================================
# NEO4J CONFIGURATION
# =============================================================================
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password123"

# =============================================================================
# Print Configuration Summary
# =============================================================================
echo "=============================================="
echo "Dual-Remote-LLM Configuration Loaded"
echo "=============================================="
echo ""
echo "PRIMARY LLM (remote — queries & answer generation):"
echo "  USE_LOCAL_LLM:        $USE_LOCAL_LLM"
echo "  OPENAI_MODEL:         $OPENAI_MODEL"
echo "  OPENAI_API_BASE:      $OPENAI_API_BASE"
echo "  OPENAI_HTTP_TIMEOUT:  ${OPENAI_HTTP_TIMEOUT:-unset}"
if [ -n "$OPENAI_KEY" ] && [ "$OPENAI_KEY" != "YOUR_API_KEY" ]; then
    echo "  OPENAI_KEY:           ${OPENAI_KEY:0:8}..."
else
    echo "  OPENAI_KEY:           (not set - please configure!)"
fi
echo "  LOCAL_EMBEDDER_MODEL: $LOCAL_EMBEDDER_MODEL"
echo ""
echo "VALIDATION LLM (remote — triplet fact-checking):"
echo "  REMOTE_LLM_MODEL:    $REMOTE_LLM_MODEL"
echo "  REMOTE_LLM_BASE_URL: $REMOTE_LLM_BASE_URL"
echo "  REMOTE_LLM_HTTP_TIMEOUT: ${REMOTE_LLM_HTTP_TIMEOUT:-unset}"
if [ -n "$REMOTE_LLM_API_KEY" ] && [ "$REMOTE_LLM_API_KEY" != "YOUR_API_KEY" ]; then
    echo "  REMOTE_LLM_API_KEY:  ${REMOTE_LLM_API_KEY:0:8}..."
else
    echo "  REMOTE_LLM_API_KEY:  (not set - please configure!)"
fi
echo ""
echo "NEO4J:"
echo "  NEO4J_URI:            $NEO4J_URI"
echo "  NEO4J_USER:           $NEO4J_USER"
echo "=============================================="
echo ""
echo "No GPU required. All LLM calls go through the remote API."
echo ""
echo "Usage examples:"
echo "  python scripts/demo_mcp_agent.py --validated-expand"
echo "  python scripts/eval/evaluate.py --configs C --output-dir data/eval --verbose"
echo ""
