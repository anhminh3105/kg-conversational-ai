"""
Internal utilities for loading and caching prompt templates.

All public access should go through rag.prompts (the __init__.py).
"""

import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Directory containing prompt files
PROMPTS_DIR = Path(__file__).parent

# Cache for loaded prompts
_prompt_cache: Dict[str, str] = {}


def get_prompt_path(name: str) -> Path:
    """
    Get the full path to a prompt file.

    Args:
        name: Prompt name (without .txt extension)

    Returns:
        Path to the prompt file
    """
    return PROMPTS_DIR / f"{name}.txt"


def load_prompt(
    name: str,
    use_cache: bool = True,
    **format_kwargs: Any,
) -> str:
    """
    Load a prompt template from file.

    Args:
        name: Prompt name (without .txt extension)
        use_cache: Whether to cache loaded prompts (default: True)
        **format_kwargs: Variables to format into the prompt template

    Returns:
        The prompt text, optionally formatted with provided variables

    Raises:
        FileNotFoundError: If the prompt file doesn't exist
        ValueError: If the prompt file is empty or contains only whitespace

    Example:
        # Load without formatting
        prompt = load_prompt("mcp_agent_system")

        # Load with variable substitution
        prompt = load_prompt("mcp_agent_system", tools_json=json.dumps(tools))
    """
    # Check cache first
    if use_cache and name in _prompt_cache and not format_kwargs:
        return _prompt_cache[name]

    path = get_prompt_path(name)

    try:
        with open(path, "r", encoding="utf-8") as f:
            template = f.read()
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {path}")
        raise

    if not template.strip():
        raise ValueError(f"Prompt file is empty: {path}")

    logger.debug(f"Loaded prompt template: {name}")

    # Cache the raw template
    if use_cache:
        _prompt_cache[name] = template

    # Format if kwargs provided
    if format_kwargs:
        return template.format(**format_kwargs)

    return template


def clear_cache() -> None:
    """Clear the prompt cache."""
    _prompt_cache.clear()
    logger.debug("Prompt cache cleared")


def list_prompts() -> list:
    """
    List all available prompt files.

    Returns:
        List of prompt names (without .txt extension)
    """
    return [p.stem for p in PROMPTS_DIR.glob("*.txt")]
