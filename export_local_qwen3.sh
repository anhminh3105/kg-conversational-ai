#!/bin/bash
# =============================================================================
# Local LLM Configuration for Eval QA Generation
# =============================================================================
# Configures a local HuggingFace model for the eval QA generation pipeline.
#
# Usage:
#   source export_local_qwen3.sh                    # configure only
#   source export_local_qwen3.sh --run-generate-qa  # configure + run QA generation
#
# Default model: Qwen/Qwen2.5-7B-Instruct (~5 GB VRAM with 4-bit quantization)
# Override:      LOCAL_LLM_MODEL=... source export_local_qwen3.sh
#
# Note: Qwen3.5 models require transformers >= 5.2 (Python >= 3.10).
#       With Python 3.9, use Qwen2.5 models instead.
# =============================================================================

# Enable local LLM mode (remote API vars are kept so --remote-llm still works)
export USE_LOCAL_LLM=true

# =============================================================================
# Model Configuration
# =============================================================================
export LOCAL_LLM_MODEL="${LOCAL_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export LOCAL_EMBEDDER_MODEL="BAAI/bge-small-en-v1.5"
export LOCAL_LLM_QUANTIZE="4bit"

# Thinking mode - set to "true" to enable chain-of-thought reasoning (Qwen3+ models)
export LOCAL_LLM_ENABLE_THINKING="${LOCAL_LLM_ENABLE_THINKING:-false}"

# =============================================================================
# Summary
# =============================================================================
echo "=============================================="
echo "Local LLM Configuration Loaded"
echo "=============================================="
echo "  USE_LOCAL_LLM:        $USE_LOCAL_LLM"
echo "  LOCAL_LLM_MODEL:      $LOCAL_LLM_MODEL"
echo "  LOCAL_EMBEDDER_MODEL: $LOCAL_EMBEDDER_MODEL"
echo "  LOCAL_LLM_QUANTIZE:   $LOCAL_LLM_QUANTIZE"
echo "  ENABLE_THINKING:      $LOCAL_LLM_ENABLE_THINKING"
echo "=============================================="

# =============================================================================
# Optional: run QA generation immediately
# =============================================================================
if [[ "$1" == "--run-generate-qa" ]]; then
    echo ""
    echo "Running QA generation with local LLM ..."
    echo "(Using --no-cache to regenerate all questions with the local model)"
    echo ""
    python scripts/eval/generate_qa.py --no-cache "$@"
fi
