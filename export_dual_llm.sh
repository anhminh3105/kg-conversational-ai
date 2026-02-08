#!/bin/bash
# =============================================================================
# Dual-LLM Configuration Script
# =============================================================================
# This script configures environment variables for running the dual-LLM
# knowledge expansion workflow:
#   - Local LLM: For knowledge graph querying and triplet proposal
#   - Remote LLM: For triplet validation and fact-checking
#
# Usage: source export_dual_llm.sh
#
# Prerequisites:
#   - GPU with 6GB+ VRAM for local LLM (with 4-bit quantization)
#   - API key for remote LLM service (Google AI, OpenAI, or SambaNova)
# =============================================================================

# =============================================================================
# LOCAL LLM CONFIGURATION (for queries and triplet generation)
# =============================================================================
# Enable local LLM mode for primary operations
export USE_LOCAL_LLM=true

# LLM Model - Qwen2.5-7B with 4-bit quantization (~5GB VRAM)
export LOCAL_LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"

# Embedder Model - Lightweight and fast
export LOCAL_EMBEDDER_MODEL="BAAI/bge-small-en-v1.5"

# Quantization - 4bit recommended for GPU with 16GB or less
export LOCAL_LLM_QUANTIZE="4bit"

# =============================================================================
# REMOTE LLM CONFIGURATION (for triplet validation)
# =============================================================================
# The remote LLM is used to validate and fact-check triplets proposed by
# the local LLM. This provides a second opinion from a more capable model.
#
# Supported providers:
#   - Google AI Studio (recommended - free tier available)
#   - OpenAI
#   - SambaNova
#   - Any OpenAI-compatible API
# =============================================================================

# Option 1: Google AI Studio (recommended)
# Get your API key from: https://aistudio.google.com/apikey
export REMOTE_LLM_API_KEY=""
export REMOTE_LLM_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
export REMOTE_LLM_MODEL="gemini-2.5-flash-lite"

# Option 2: OpenAI (uncomment to use)
# export REMOTE_LLM_API_KEY="YOUR_OPENAI_API_KEY"
# export REMOTE_LLM_BASE_URL="https://api.openai.com/v1"
# export REMOTE_LLM_MODEL="gpt-4o-mini"

# Option 3: SambaNova (uncomment to use)
# export REMOTE_LLM_API_KEY="YOUR_SAMBANOVA_API_KEY"
# export REMOTE_LLM_BASE_URL="https://api.sambanova.ai/v1"
# export REMOTE_LLM_MODEL="Meta-Llama-3.1-70B-Instruct"

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
echo "Dual-LLM Configuration Loaded"
echo "=============================================="
echo ""
echo "LOCAL LLM (queries & triplet generation):"
echo "  USE_LOCAL_LLM:        $USE_LOCAL_LLM"
echo "  LOCAL_LLM_MODEL:      $LOCAL_LLM_MODEL"
echo "  LOCAL_EMBEDDER_MODEL: $LOCAL_EMBEDDER_MODEL"
echo "  LOCAL_LLM_QUANTIZE:   $LOCAL_LLM_QUANTIZE"
echo ""
echo "REMOTE LLM (validation & fact-checking):"
echo "  REMOTE_LLM_MODEL:     $REMOTE_LLM_MODEL"
echo "  REMOTE_LLM_BASE_URL:  $REMOTE_LLM_BASE_URL"
if [ -n "$REMOTE_LLM_API_KEY" ] && [ "$REMOTE_LLM_API_KEY" != "YOUR_GOOGLE_AI_API_KEY" ]; then
    echo "  REMOTE_LLM_API_KEY:   ${REMOTE_LLM_API_KEY:0:8}..."
else
    echo "  REMOTE_LLM_API_KEY:   (not set - please configure!)"
fi
echo ""
echo "NEO4J:"
echo "  NEO4J_URI:            $NEO4J_URI"
echo "  NEO4J_USER:           $NEO4J_USER"
echo "=============================================="
echo ""
echo "To run the dual-LLM demo:"
echo "  python scripts/demo_mcp_agent.py --validated-expand"
echo ""
echo "To run with verbose output:"
echo "  python scripts/demo_mcp_agent.py --validated-expand --verbose"
echo ""
