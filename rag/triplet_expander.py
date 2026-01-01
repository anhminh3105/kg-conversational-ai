"""
Triplet Expander: Uses LLM to generate additional related triplets.

This module provides functionality to expand sparse retrieved triplets
by leveraging LLM's parametric knowledge to generate additional facts
that are semantically related to the user's query and retrieved context.

Uses schema-constrained prompting to ensure generated predicates
match the knowledge graph ontology.
"""

import logging
import re
from typing import List, Tuple, Optional, Set
from pathlib import Path

from .edc.edc.utils.llm_utils import openai_chat_completion

logger = logging.getLogger(__name__)

# Default template path (in rag/edc/prompt_templates/)
DEFAULT_TEMPLATE_PATH = Path(__file__).parent / "edc" / "prompt_templates" / "triplet_expansion.txt"

# Fallback template if file not found
FALLBACK_EXPANSION_TEMPLATE = """You are a knowledge graph expert. Given a user's question and some existing knowledge graph triplets, generate additional relevant triplets that would help answer the question.

=== USER QUESTION ===
{question}

=== EXISTING TRIPLETS ===
{existing_triplets}

=== VALID RELATIONS ===
You may ONLY use the following predicates (relations):
{schema_relations}

=== INSTRUCTIONS ===
1. Generate {max_triplets} additional triplets that are relevant to answering the question.
2. Each triplet must use ONLY predicates from the valid relations list above.
3. Focus on facts that would directly help answer the user's question.
4. Use entities that are related to those in the existing triplets.
5. Format each triplet as: (subject, predicate, object)
6. Output ONLY the triplets, one per line.

=== ADDITIONAL TRIPLETS ===
"""


class TripletExpander:
    """
    Expands retrieved triplets using LLM to generate additional related facts.
    
    Uses schema-constrained prompting to produce valid (subject, predicate, object) triplets.
    This leverages the LLM's parametric knowledge to "complete" the knowledge graph
    with facts that are semantically related to the query and retrieved context.
    
    Usage:
        expander = TripletExpander(schema_path="./schemas/webnlg_schema.csv")
        
        # Expand triplets
        expanded = expander.expand(
            query="Where is Alan Shepard from?",
            retrieved_triplets=[("Alan_Shepard", "birthPlace", "New_Hampshire")],
            max_new_triplets=5
        )
    """
    
    def __init__(
        self,
        schema_path: Optional[str] = None,
        template_path: Optional[str] = None,
        max_schema_relations: int = 50,
    ):
        """
        Initialize the triplet expander.
        
        Args:
            schema_path: Path to CSV file with relation definitions (relation,description)
            template_path: Path to custom prompt template
            max_schema_relations: Maximum number of schema relations to include in prompt
        """
        self.template = self._load_template(template_path)
        self.schema_relations = self._load_schema(schema_path)
        self.max_schema_relations = max_schema_relations
        
        logger.info(f"Initialized TripletExpander with {len(self.schema_relations)} schema relations")
    
    def _load_template(self, template_path: Optional[str] = None) -> str:
        """Load the expansion prompt template from file."""
        path = Path(template_path) if template_path else DEFAULT_TEMPLATE_PATH
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                template = f.read()
            logger.debug(f"Loaded expansion template from {path}")
            return template
        except FileNotFoundError:
            logger.warning(f"Template not found at {path}, using fallback")
            return FALLBACK_EXPANSION_TEMPLATE
    
    def _load_schema(self, schema_path: Optional[str] = None) -> List[Tuple[str, str]]:
        """
        Load schema relations from CSV file.
        
        Returns:
            List of (relation_name, description) tuples
        """
        if schema_path is None:
            # Try default path relative to this module
            default_schema = Path(__file__).parent / "edc" / "schemas" / "webnlg_schema.csv"
            if default_schema.exists():
                schema_path = str(default_schema)
            else:
                logger.warning("No schema file found. Using empty schema.")
                return []
        
        relations = []
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and "," in line:
                        # Split on first comma only
                        parts = line.split(",", 1)
                        relation = parts[0].strip()
                        description = parts[1].strip() if len(parts) > 1 else ""
                        relations.append((relation, description))
            logger.debug(f"Loaded {len(relations)} relations from {schema_path}")
        except FileNotFoundError:
            logger.warning(f"Schema file not found: {schema_path}")
        
        return relations
    
    def _format_triplets(self, triplets: List[Tuple[str, str, str]]) -> str:
        """Format triplets for prompt."""
        if not triplets:
            return "No existing triplets available."
        
        lines = []
        for s, p, o in triplets:
            # Clean up underscores for readability
            s_clean = s.replace("_", " ")
            p_clean = p.replace("_", " ")
            o_clean = o.replace("_", " ")
            lines.append(f"({s_clean}, {p_clean}, {o_clean})")
        
        return "\n".join(lines)
    
    def _select_relevant_relations(
        self,
        query: str,
        triplets: List[Tuple[str, str, str]],
    ) -> List[str]:
        """
        Select the most relevant schema relations for the query.
        
        Uses simple keyword matching to prioritize relations.
        """
        if not self.schema_relations:
            return []
        
        query_lower = query.lower()
        
        # Extract predicates from existing triplets
        existing_predicates = set()
        for _, p, _ in triplets:
            existing_predicates.add(p.lower().replace("_", " "))
        
        # Score relations by relevance
        scored_relations = []
        for relation, description in self.schema_relations:
            score = 0
            rel_lower = relation.lower()
            desc_lower = description.lower()
            
            # Boost if already used in triplets
            if rel_lower in existing_predicates:
                score += 10
            
            # Check for keyword matches in query
            query_words = set(query_lower.split())
            rel_words = set(rel_lower.replace("_", " ").split())
            desc_words = set(desc_lower.split())
            
            # Match with query
            score += len(query_words & rel_words) * 3
            score += len(query_words & desc_words) * 1
            
            scored_relations.append((score, relation, description))
        
        # Sort by score (descending) and take top N
        scored_relations.sort(key=lambda x: x[0], reverse=True)
        
        # Format as "relation: description"
        result = []
        for score, relation, description in scored_relations[:self.max_schema_relations]:
            if description:
                result.append(f"- {relation}: {description}")
            else:
                result.append(f"- {relation}")
        
        return result
    
    def _parse_triplets_from_response(self, response: str) -> List[Tuple[str, str, str]]:
        """
        Parse triplets from LLM response.
        
        Expects format: (subject, predicate, object)
        """
        triplets = []
        
        # Match triplet patterns: (subject, predicate, object)
        pattern = r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)'
        matches = re.findall(pattern, response)
        
        for match in matches:
            subject = match[0].strip().replace(" ", "_")
            predicate = match[1].strip().replace(" ", "_")
            obj = match[2].strip().replace(" ", "_")
            
            # Basic validation
            if subject and predicate and obj:
                triplets.append((subject, predicate, obj))
        
        logger.debug(f"Parsed {len(triplets)} triplets from LLM response")
        return triplets
    
    def _validate_triplets(
        self,
        triplets: List[Tuple[str, str, str]],
        valid_relations: Optional[Set[str]] = None,
    ) -> List[Tuple[str, str, str]]:
        """
        Validate triplets against schema relations.
        
        Args:
            triplets: List of triplets to validate
            valid_relations: Set of valid relation names (optional)
            
        Returns:
            List of validated triplets
        """
        if valid_relations is None:
            valid_relations = {r.lower().replace(" ", "_") for r, _ in self.schema_relations}
        
        validated = []
        for s, p, o in triplets:
            p_normalized = p.lower().replace(" ", "_")
            
            # Check if predicate is in schema
            if p_normalized in valid_relations or not valid_relations:
                validated.append((s, p, o))
            else:
                logger.debug(f"Filtered out triplet with invalid predicate: {p}")
        
        return validated
    
    def expand(
        self,
        query: str,
        retrieved_triplets: List[Tuple[str, str, str]],
        max_new_triplets: int = 10,
        temperature: float = 0.3,
        validate_schema: bool = True,
    ) -> List[Tuple[str, str, str]]:
        """
        Expand triplets using LLM to generate additional related facts.
        
        Args:
            query: User's question
            retrieved_triplets: Triplets retrieved from FAISS
            max_new_triplets: Maximum number of new triplets to generate
            temperature: LLM sampling temperature (lower = more focused)
            validate_schema: Whether to filter triplets with invalid predicates
            
        Returns:
            List of new triplets (does not include original retrieved triplets)
        """
        logger.info(f"Expanding triplets for query: {query[:50]}...")
        
        # Format existing triplets
        existing_str = self._format_triplets(retrieved_triplets)
        
        # Select relevant schema relations
        relevant_relations = self._select_relevant_relations(query, retrieved_triplets)
        relations_str = "\n".join(relevant_relations) if relevant_relations else "Any valid relation"
        
        # Build prompt
        prompt = self.template.format(
            question=query,
            existing_triplets=existing_str,
            schema_relations=relations_str,
            max_triplets=max_new_triplets,
        )
        
        # Build messages for chat completion
        system_prompt = (
            "You are a knowledge graph expert that generates factual triplets. "
            "Only generate triplets that are likely to be true based on the context provided. "
            "Be concise and precise."
        )
        
        messages = [{"role": "user", "content": prompt}]
        
        # Call LLM
        try:
            response = openai_chat_completion(
                system_prompt=system_prompt,
                history=messages,
                temperature=temperature,
                max_tokens=512,
            )
            
            logger.debug(f"LLM response: {response[:200]}...")
            
            # Parse triplets from response
            new_triplets = self._parse_triplets_from_response(response)
            
            # Validate against schema if requested
            if validate_schema and self.schema_relations:
                new_triplets = self._validate_triplets(new_triplets)
            
            # Limit to max requested
            new_triplets = new_triplets[:max_new_triplets]
            
            logger.info(f"Generated {len(new_triplets)} new triplets")
            return new_triplets
            
        except Exception as e:
            logger.error(f"Error during triplet expansion: {e}")
            return []
    
    def expand_and_combine(
        self,
        query: str,
        retrieved_triplets: List[Tuple[str, str, str]],
        max_new_triplets: int = 10,
        temperature: float = 0.3,
        validate_schema: bool = True,
    ) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
        """
        Expand triplets and return both original and expanded.
        
        Args:
            query: User's question
            retrieved_triplets: Triplets retrieved from FAISS
            max_new_triplets: Maximum number of new triplets to generate
            temperature: LLM sampling temperature
            validate_schema: Whether to filter triplets with invalid predicates
            
        Returns:
            Tuple of (combined_triplets, expanded_triplets_only)
        """
        expanded = self.expand(
            query=query,
            retrieved_triplets=retrieved_triplets,
            max_new_triplets=max_new_triplets,
            temperature=temperature,
            validate_schema=validate_schema,
        )
        
        # Combine: original first, then expanded
        combined = list(retrieved_triplets) + expanded
        
        return combined, expanded


def get_triplet_expander(
    schema_path: Optional[str] = None,
    template_path: Optional[str] = None,
) -> TripletExpander:
    """
    Factory function to create a TripletExpander.
    
    Args:
        schema_path: Optional path to schema CSV file
        template_path: Optional path to prompt template
        
    Returns:
        Configured TripletExpander instance
    """
    return TripletExpander(
        schema_path=schema_path,
        template_path=template_path,
    )


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.DEBUG)
    
    # Demo usage
    parser = argparse.ArgumentParser(description="Test Triplet Expander")
    parser.add_argument("--query", default="Where was Alan Shepard born?", help="Test query")
    parser.add_argument("--schema", default=None, help="Path to schema CSV")
    args = parser.parse_args()
    
    # Create expander
    expander = TripletExpander(schema_path=args.schema)
    
    # Test with sample triplets
    sample_triplets = [
        ("Alan_Shepard", "birthPlace", "New_Hampshire"),
        ("Alan_Shepard", "nationality", "American"),
    ]
    
    print(f"\nQuery: {args.query}")
    print(f"\nExisting triplets:")
    for t in sample_triplets:
        print(f"  {t}")
    
    # Expand
    expanded = expander.expand(
        query=args.query,
        retrieved_triplets=sample_triplets,
        max_new_triplets=5,
    )
    
    print(f"\nExpanded triplets:")
    for t in expanded:
        print(f"  {t}")
