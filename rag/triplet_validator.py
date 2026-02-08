"""
Triplet Validator: Uses remote LLM to validate and correct proposed triplets.

This module provides functionality to validate triplets proposed by the local LLM
using a remote LLM service (e.g., Google AI, OpenAI, SambaNova). The remote LLM
acts as a fact-checker, verifying accuracy and suggesting corrections.

The validation workflow:
1. Local LLM proposes triplets based on query and context
2. This module sends proposals to remote LLM for validation
3. Remote LLM returns: validated, corrected, or rejected triplets
4. Validated/corrected triplets can be persisted to the knowledge graph
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

from .prompts import load_prompt

logger = logging.getLogger(__name__)

DEFAULT_VALIDATION_TEMPLATE = load_prompt("triplet_validation")


@dataclass
class ValidatedTriplet:
    """A triplet that has been validated by the remote LLM."""
    subject: str
    predicate: str
    object: str
    status: str  # "validated" or "corrected"
    reason: str
    original: Optional[Tuple[str, str, str]] = None  # Original if corrected
    
    def to_tuple(self) -> Tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "status": self.status,
            "reason": self.reason,
        }
        if self.original:
            result["original"] = list(self.original)
        return result


@dataclass
class RejectedTriplet:
    """A triplet that was rejected by the remote LLM."""
    subject: str
    predicate: str
    object: str
    reason: str
    
    def to_tuple(self) -> Tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "reason": self.reason,
        }


@dataclass
class ValidationResult:
    """Result from triplet validation."""
    validated: List[ValidatedTriplet] = field(default_factory=list)
    rejected: List[RejectedTriplet] = field(default_factory=list)
    query: str = ""
    timestamp: str = ""
    remote_model: str = ""
    local_model: str = ""
    
    @property
    def all_accepted(self) -> List[ValidatedTriplet]:
        """All triplets that passed validation (validated + corrected)."""
        return self.validated
    
    @property
    def accepted_triplets(self) -> List[Tuple[str, str, str]]:
        """Get accepted triplets as tuples."""
        return [v.to_tuple() for v in self.validated]
    
    @property
    def rejected_triplets(self) -> List[Tuple[str, str, str]]:
        """Get rejected triplets as tuples."""
        return [r.to_tuple() for r in self.rejected]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "validated": [v.to_dict() for v in self.validated],
            "rejected": [r.to_dict() for r in self.rejected],
            "query": self.query,
            "timestamp": self.timestamp,
            "remote_model": self.remote_model,
            "local_model": self.local_model,
            "summary": {
                "total_proposed": len(self.validated) + len(self.rejected),
                "accepted": len(self.validated),
                "rejected": len(self.rejected),
            }
        }
    
    def get_user_notification(self) -> str:
        """Generate a user-friendly notification about validated facts."""
        if not self.validated:
            return ""
        
        lines = [
            "",
            "---",
            f"Note: This answer includes {len(self.validated)} new fact(s) that were generated and validated:",
        ]
        
        for v in self.validated:
            status_tag = "[validated]" if v.status == "validated" else "[corrected]"
            triplet_str = f"({v.subject}, {v.predicate}, {v.object})"
            if v.status == "corrected" and v.original:
                orig_str = f"({v.original[0]}, {v.original[1]}, {v.original[2]})"
                lines.append(f"- {triplet_str} {status_tag} (from {orig_str})")
            else:
                lines.append(f"- {triplet_str} {status_tag}")
        
        lines.append("")
        lines.append("These facts have been added to the knowledge base.")
        
        return "\n".join(lines)


class TripletValidator:
    """
    Validates proposed triplets using a remote LLM service.
    
    This class sends proposed triplets to a remote LLM (via OpenAI-compatible API)
    for factual verification. The remote LLM acts as a fact-checker, returning
    validated, corrected, or rejected triplets.
    
    Usage:
        validator = TripletValidator()
        
        result = validator.validate(
            proposed_triplets=[("Einstein", "born_in", "Germany")],
            query="Where was Einstein born?",
            existing_facts=["(Einstein, field, Physics)"],
        )
        
        # Access results
        print(result.accepted_triplets)  # Validated + corrected
        print(result.rejected_triplets)  # Rejected ones
    """
    
    def __init__(
        self,
        template: Optional[str] = None,
        local_model_name: str = "local_llm",
    ):
        """
        Initialize the triplet validator.
        
        Args:
            template: Custom validation prompt template
            local_model_name: Name of the local LLM (for metadata)
        """
        self.template = template or DEFAULT_VALIDATION_TEMPLATE
        self.local_model_name = local_model_name
        
        logger.info("Initialized TripletValidator")
    
    def _format_triplets(self, triplets: List[Tuple[str, str, str]]) -> str:
        """Format triplets for the prompt."""
        lines = []
        for i, (s, p, o) in enumerate(triplets, 1):
            # Clean up underscores for readability
            s_clean = s.replace("_", " ")
            p_clean = p.replace("_", " ")
            o_clean = o.replace("_", " ")
            lines.append(f"{i}. ({s_clean}, {p_clean}, {o_clean})")
        return "\n".join(lines)
    
    def _format_existing_facts(self, facts: List[str]) -> str:
        """Format existing facts for the prompt."""
        if not facts:
            return "No existing facts available."
        return "\n".join(f"- {fact}" for fact in facts)
    
    def _parse_validation_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the JSON response from the remote LLM.
        
        Handles various response formats and extracts the validation results.
        """
        # Try to find JSON in the response
        json_patterns = [
            r'```json\s*(.*?)\s*```',  # JSON code block
            r'```\s*(.*?)\s*```',       # Generic code block
            r'(\{[\s\S]*\})',           # Raw JSON object
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                try:
                    json_str = match.group(1)
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue
        
        # If no JSON found, try to parse the whole response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse validation response as JSON: {response[:200]}...")
            return {"validated": [], "corrected": [], "rejected": []}
    
    def _normalize_triplet(self, triplet: Any) -> Optional[Tuple[str, str, str]]:
        """Normalize a triplet from various formats."""
        if isinstance(triplet, (list, tuple)) and len(triplet) >= 3:
            return (
                str(triplet[0]).strip().replace(" ", "_"),
                str(triplet[1]).strip().replace(" ", "_"),
                str(triplet[2]).strip().replace(" ", "_"),
            )
        return None
    
    def validate(
        self,
        proposed_triplets: List[Tuple[str, str, str]],
        query: str,
        existing_facts: Optional[List[str]] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> ValidationResult:
        """
        Validate proposed triplets using the remote LLM.
        
        Args:
            proposed_triplets: List of (subject, predicate, object) tuples to validate
            query: The user's original question (for context)
            existing_facts: List of existing fact strings from the knowledge graph
            temperature: LLM sampling temperature
            max_tokens: Maximum tokens for the response
            
        Returns:
            ValidationResult containing validated, corrected, and rejected triplets
        """
        from .edc.edc.utils.llm_utils import remote_chat_completion, get_remote_model_name
        
        if not proposed_triplets:
            return ValidationResult(query=query)
        
        logger.info(f"Validating {len(proposed_triplets)} proposed triplets")
        
        # Build the validation prompt
        prompt = self.template.format(
            query=query,
            existing_facts=self._format_existing_facts(existing_facts or []),
            proposed_triplets=self._format_triplets(proposed_triplets),
        )
        
        # System prompt for the validator
        system_prompt = (
            "You are a knowledge validation expert. Verify factual accuracy of proposed "
            "knowledge graph triplets. Be strict - only validate what you are confident is correct. "
            "Always respond with valid JSON."
        )
        
        messages = [{"role": "user", "content": prompt}]
        
        # Call remote LLM for validation
        try:
            response = remote_chat_completion(
                system_prompt=system_prompt,
                history=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            remote_model = get_remote_model_name()
            logger.debug(f"Remote LLM response: {response[:300]}...")
            
        except Exception as e:
            logger.error(f"Remote LLM validation failed: {e}")
            # On failure, reject all triplets
            return ValidationResult(
                rejected=[
                    RejectedTriplet(
                        subject=s, predicate=p, object=o,
                        reason=f"Validation failed: {str(e)}"
                    )
                    for s, p, o in proposed_triplets
                ],
                query=query,
                timestamp=datetime.now().isoformat(),
                remote_model="error",
                local_model=self.local_model_name,
            )
        
        # Parse the response
        parsed = self._parse_validation_response(response)
        
        # Build result
        result = ValidationResult(
            query=query,
            timestamp=datetime.now().isoformat(),
            remote_model=remote_model,
            local_model=self.local_model_name,
        )
        
        # Process rejected triplets FIRST to build a rejection set.
        # This ensures that if the remote LLM returns the same triplet in
        # both 'validated' and 'rejected' arrays (an observed LLM bug),
        # the rejection takes priority (conservative/safe approach).
        rejected_keys = set()
        for item in parsed.get("rejected", []):
            triplet = item.get("triplet") if isinstance(item, dict) else item
            normalized = self._normalize_triplet(triplet)
            if normalized:
                rejected_keys.add(normalized)
                result.rejected.append(RejectedTriplet(
                    subject=normalized[0],
                    predicate=normalized[1],
                    object=normalized[2],
                    reason=item.get("reason", "") if isinstance(item, dict) else "",
                ))
        
        # Process validated triplets, skipping any that were also rejected
        for item in parsed.get("validated", []):
            triplet = item.get("triplet") if isinstance(item, dict) else item
            normalized = self._normalize_triplet(triplet)
            if normalized:
                if normalized in rejected_keys:
                    logger.warning(
                        f"Triplet {normalized} appeared in both 'validated' and "
                        f"'rejected' arrays from remote LLM. Treating as rejected."
                    )
                    continue
                result.validated.append(ValidatedTriplet(
                    subject=normalized[0],
                    predicate=normalized[1],
                    object=normalized[2],
                    status="validated",
                    reason=item.get("reason", "") if isinstance(item, dict) else "",
                ))
        
        # Process corrected triplets, skipping any that were also rejected
        for item in parsed.get("corrected", []):
            if isinstance(item, dict):
                corrected = self._normalize_triplet(item.get("corrected"))
                original = self._normalize_triplet(item.get("original"))
                if corrected:
                    if corrected in rejected_keys:
                        logger.warning(
                            f"Corrected triplet {corrected} was also rejected "
                            f"by remote LLM. Treating as rejected."
                        )
                        continue
                    result.validated.append(ValidatedTriplet(
                        subject=corrected[0],
                        predicate=corrected[1],
                        object=corrected[2],
                        status="corrected",
                        reason=item.get("reason", ""),
                        original=original,
                    ))
        
        logger.info(
            f"Validation complete: {len(result.validated)} accepted, "
            f"{len(result.rejected)} rejected"
        )
        
        return result
    
    def validate_batch(
        self,
        batches: List[Dict[str, Any]],
        temperature: float = 0.1,
    ) -> List[ValidationResult]:
        """
        Validate multiple batches of triplets.
        
        Args:
            batches: List of dicts with 'proposed_triplets', 'query', 'existing_facts'
            temperature: LLM sampling temperature
            
        Returns:
            List of ValidationResult objects
        """
        results = []
        for batch in batches:
            result = self.validate(
                proposed_triplets=batch.get("proposed_triplets", []),
                query=batch.get("query", ""),
                existing_facts=batch.get("existing_facts"),
                temperature=temperature,
            )
            results.append(result)
        return results


def get_triplet_validator(
    template: Optional[str] = None,
    local_model_name: str = "local_llm",
) -> TripletValidator:
    """
    Factory function to create a TripletValidator.
    
    Args:
        template: Optional custom validation template
        local_model_name: Name of the local LLM for metadata
        
    Returns:
        Configured TripletValidator instance
    """
    return TripletValidator(
        template=template,
        local_model_name=local_model_name,
    )


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.DEBUG)
    
    parser = argparse.ArgumentParser(description="Test Triplet Validator")
    parser.add_argument("--query", default="Where was Einstein born?", help="Test query")
    args = parser.parse_args()
    
    # Test with sample triplets
    validator = TripletValidator()
    
    sample_triplets = [
        ("Albert_Einstein", "born_in", "Ulm_Germany"),
        ("Albert_Einstein", "died_in", "Princeton"),
        ("Albert_Einstein", "invented", "Light_Bulb"),  # Should be rejected
    ]
    
    existing_facts = [
        "(Albert_Einstein, field, Physics)",
        "(Albert_Einstein, known_for, Relativity)",
    ]
    
    print(f"\nQuery: {args.query}")
    print(f"\nProposed triplets:")
    for t in sample_triplets:
        print(f"  {t}")
    
    print("\nValidating with remote LLM...")
    
    try:
        result = validator.validate(
            proposed_triplets=sample_triplets,
            query=args.query,
            existing_facts=existing_facts,
        )
        
        print(f"\n=== Validation Result ===")
        print(f"Accepted: {len(result.validated)}")
        for v in result.validated:
            print(f"  [{v.status}] {v.to_tuple()}: {v.reason}")
        
        print(f"\nRejected: {len(result.rejected)}")
        for r in result.rejected:
            print(f"  {r.to_tuple()}: {r.reason}")
        
        print(result.get_user_notification())
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure remote LLM is configured (source export_google_ai.sh or similar)")
