#!/bin/bash
# Google AI Studio credentials (OpenAI-compatible endpoint)

# Disable local LLM mode (in case it was previously enabled)
unset USE_LOCAL_LLM

export OPENAI_KEY=""
export OPENAI_API_BASE="https://generativelanguage.googleapis.com/v1beta/openai/"
# export OPENAI_MODEL="gemini-2.5-flash-lite"
export OPENAI_MODEL="gemini-2.5-flash"
export EMBEDDER_MODEL="BAAI/bge-small-en-v1.5"
echo "Google AI Studio credentials exported successfully"
echo "  USE_LOCAL_LLM: (disabled)"
echo "  OPENAI_KEY: ${OPENAI_KEY:0:8}..."
echo "  OPENAI_API_BASE: $OPENAI_API_BASE"
echo "  OPENAI_MODEL: $OPENAI_MODEL"
echo "  EMBEDDER_MODEL: $EMBEDDER_MODEL"
