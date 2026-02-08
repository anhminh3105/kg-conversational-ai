"""
Prompt loader utilities for RAG system.

This module provides functions to load prompt templates from text files,
enabling easier maintenance and editing of prompts without modifying Python code.

Usage:
    from rag.prompts import load_prompt, get_prompt_path
    
    # Load a prompt template
    prompt = load_prompt("mcp_agent_system")
    
    # Load and format a prompt with variables
    prompt = load_prompt("mcp_agent_system", tools_json=json.dumps(tools))
    
    # Get the path to a prompt file
    path = get_prompt_path("mcp_agent_system")
"""

from .prompt_utils import (
    load_prompt,
    get_prompt_path,
    clear_cache,
    list_prompts,
    PROMPTS_DIR,
)

# Pre-defined prompt names for IDE autocompletion
PROMPT_NAMES = {
    "mcp_agent_system": "System prompt for MCP Agent with full tool definitions",
    "mcp_agent_lite_system": "System prompt for lightweight MCP Agent",
    "answer_generation": "Prompt for generating answers from facts",
    "answer_generation_with_new_facts": "Answer prompt noting new validated facts",
    "knowledge_assessment": "Prompt for assessing knowledge sufficiency",
    "triplet_reproposal": "Prompt for re-proposing rejected triplets",
    "triplet_validation": "Prompt for validating proposed triplets",
    "persistence_justification": "Prompt for local LLM to decide which triplets to persist",
    "kg_qa_fallback": "Fallback prompt for KG question answering",
    "kg_qa_system": "System prompt for KG QA",
    "triplet_expansion_fallback": "Fallback prompt for triplet expansion",
}

__all__ = [
    "load_prompt",
    "get_prompt_path",
    "clear_cache",
    "list_prompts",
    "PROMPTS_DIR",
    "PROMPT_NAMES",
]
