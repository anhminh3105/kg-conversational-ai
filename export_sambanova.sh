#!/bin/bash
# SambaNova API credentials

# Disable local LLM mode (in case it was previously enabled)
unset USE_LOCAL_LLM

export OPENAI_KEY=""
export OPENAI_API_BASE="https://api.sambanova.ai/v1"
export OPENAI_MODEL="Meta-Llama-3.1-8B-Instruct"
export EMBEDDER_MODEL="BAAI/bge-small-en-v1.5"

echo "SambaNova credentials exported successfully"
echo "  USE_LOCAL_LLM: (disabled)"
echo "  OPENAI_KEY: ${OPENAI_KEY:0:8}..."
echo "  OPENAI_API_BASE: $OPENAI_API_BASE"
echo "  OPENAI_MODEL: $OPENAI_MODEL"
echo "  EMBEDDER_MODEL: $EMBEDDER_MODEL"
