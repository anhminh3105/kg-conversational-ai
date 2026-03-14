"""
LLM-based predicate selection for filtered KG retrieval.

Given a user query and the set of relationship types available in the index,
asks the LLM to pick the most relevant predicate(s).  This turns manual
``--predicate_filter`` flags into an automatic step.
"""

import json
import logging
from typing import List, Optional

from .edc.edc.utils.llm_utils import openai_chat_completion

logger = logging.getLogger(__name__)

SELECTOR_SYSTEM_PROMPT = (
    "You are a biomedical knowledge-graph expert.  "
    "Your ONLY job is to select relationship types that are relevant to the "
    "user's question.  Return a JSON array of strings -- nothing else."
)

SELECTOR_USER_TEMPLATE = """\
Given the following question and the list of available relationship types \
in a biomedical knowledge graph, select the 1-3 relationship types that \
are most relevant to answering the question.

Question: {query}

Available relationships:
{predicate_list}

Return ONLY a JSON array of the selected relationship names, e.g. ["treats", "indication"]. \
Do NOT include explanation.\
"""


class PredicateSelector:
    """Select relevant predicates for a query by prompting the LLM."""

    def __init__(self, temperature: float = 0.0, max_tokens: int = 128):
        self.temperature = temperature
        self.max_tokens = max_tokens

    def select(
        self,
        query: str,
        available_predicates: List[str],
    ) -> Optional[List[str]]:
        """
        Ask the LLM to choose relevant predicates.

        Returns:
            A list of predicate strings, or ``None`` when the LLM response
            cannot be parsed (caller should fall back to unfiltered search).
        """
        if not available_predicates:
            return None

        predicate_list = "\n".join(f"- {p}" for p in available_predicates)
        user_msg = SELECTOR_USER_TEMPLATE.format(
            query=query,
            predicate_list=predicate_list,
        )

        logger.info("Predicate selection -- prompting LLM with query: %s", query)
        # logger.info("Available predicates (%d): %s", len(available_predicates), available_predicates)
        logger.info("System prompt: %s", SELECTOR_SYSTEM_PROMPT)
        logger.info("User prompt:\n%s", user_msg)

        try:
            raw = openai_chat_completion(
                system_prompt=SELECTOR_SYSTEM_PROMPT,
                history=[{"role": "user", "content": user_msg}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception:
            logger.exception("LLM call failed during predicate selection")
            return None

        logger.info("LLM raw response: %s", raw)
        selected = self._parse_response(raw, available_predicates)
        # logger.info("Parsed predicate selection: %s", selected)
        return selected

    @staticmethod
    def _parse_response(
        raw: str,
        available_predicates: List[str],
    ) -> Optional[List[str]]:
        """Extract a list of predicates from the LLM's text output."""
        raw = raw.strip()

        # Try direct JSON parse first
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list) and all(isinstance(p, str) for p in parsed):
                return [p for p in parsed if p]
        except json.JSONDecodeError:
            pass

        # Fallback: extract the first JSON array in the response
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(raw[start : end + 1])
                if isinstance(parsed, list):
                    return [str(p) for p in parsed if p]
            except json.JSONDecodeError:
                pass

        # Last resort: comma-separated plain text
        available_lower = {p.lower(): p for p in available_predicates}
        candidates = [tok.strip().strip("\"'") for tok in raw.split(",")]
        matched = [
            available_lower[c.lower()]
            for c in candidates
            if c.lower() in available_lower
        ]
        if matched:
            return matched

        logger.warning("Could not parse predicate selection from LLM response: %s", raw)
        return None
