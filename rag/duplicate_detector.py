"""
Duplicate Detection Module for Triplet Persistence.

Provides functionality to detect both exact and semantically similar triplets
to avoid storing redundant information in the knowledge graph.
"""

import logging
from typing import List, Tuple, Set, Optional
from dataclasses import dataclass

import numpy as np

from .triplet_loader import Triplet
from .representation import EmbeddableItem, humanize

logger = logging.getLogger(__name__)


@dataclass
class DuplicateCheckResult:
    """
    Result from duplicate detection.
    
    Attributes:
        new_triplets: Triplets that are not duplicates
        exact_duplicates: Triplets that exactly match existing ones
        semantic_duplicates: Triplets that are semantically similar to existing ones
    """
    new_triplets: List[Tuple[str, str, str]]
    exact_duplicates: List[Tuple[str, str, str]]
    semantic_duplicates: List[Tuple[Tuple[str, str, str], float]]  # (triplet, similarity_score)
    
    def __repr__(self) -> str:
        return (
            f"DuplicateCheckResult(new={len(self.new_triplets)}, "
            f"exact_dup={len(self.exact_duplicates)}, "
            f"semantic_dup={len(self.semantic_duplicates)})"
        )


class DuplicateDetector:
    """
    Detects duplicate triplets using exact matching and semantic similarity.
    
    Uses the indexer's embedder for semantic comparison and FAISS store
    for efficient similarity search.
    
    Usage:
        detector = DuplicateDetector(indexer, similarity_threshold=0.95)
        result = detector.filter_duplicates(new_triplets)
        # result.new_triplets contains only non-duplicate triplets
    """
    
    def __init__(
        self,
        indexer,  # KGRagIndexer - avoid circular import
        similarity_threshold: float = 0.95,
    ):
        """
        Initialize the duplicate detector.
        
        Args:
            indexer: A loaded KGRagIndexer instance
            similarity_threshold: Cosine similarity threshold for semantic duplicates (0.0-1.0)
        """
        self.indexer = indexer
        self.similarity_threshold = similarity_threshold
        
        # Build set of existing triplet keys for fast exact matching
        self._existing_keys: Optional[Set[Tuple[str, str, str]]] = None
        
        logger.info(f"Initialized DuplicateDetector with threshold={similarity_threshold}")
    
    def _get_existing_keys(self) -> Set[Tuple[str, str, str]]:
        """
        Build and cache set of existing triplet keys from the index metadata.
        
        Returns:
            Set of (subject, predicate, object) tuples
        """
        if self._existing_keys is not None:
            return self._existing_keys
        
        self._existing_keys = set()
        
        # Extract triplet keys from store metadata
        if self.indexer._store is not None:
            for meta in self.indexer.store.metadata:
                subject = meta.get("subject", "")
                predicate = meta.get("predicate", "")
                obj = meta.get("object", "")
                
                if subject and predicate and obj:
                    # Store both normalized and non-normalized versions
                    key = (subject, predicate, obj)
                    self._existing_keys.add(key)
                    
                    # Also add normalized version for comparison
                    key_norm = (
                        subject.replace("_", " ").lower(),
                        predicate.replace("_", " ").lower(),
                        obj.replace("_", " ").lower(),
                    )
                    self._existing_keys.add(key_norm)
        
        logger.debug(f"Built existing keys cache with {len(self._existing_keys)} entries")
        return self._existing_keys
    
    def _normalize_triplet(self, triplet: Tuple[str, str, str]) -> Tuple[str, str, str]:
        """Normalize a triplet for comparison."""
        s, p, o = triplet
        return (
            s.replace("_", " ").lower().strip(),
            p.replace("_", " ").lower().strip(),
            o.replace("_", " ").lower().strip(),
        )
    
    def _is_exact_duplicate(self, triplet: Tuple[str, str, str]) -> bool:
        """
        Check if a triplet exactly matches an existing one.
        
        Args:
            triplet: (subject, predicate, object) tuple
            
        Returns:
            True if exact duplicate exists
        """
        existing = self._get_existing_keys()
        
        # Check original form
        if triplet in existing:
            return True
        
        # Check normalized form
        normalized = self._normalize_triplet(triplet)
        return normalized in existing
    
    def _triplet_to_text(self, triplet: Tuple[str, str, str]) -> str:
        """Convert triplet to embeddable text format."""
        s, p, o = triplet
        return f"{humanize(s)} {humanize(p)} {humanize(o)}"
    
    def _check_semantic_duplicate(
        self,
        triplet: Tuple[str, str, str],
    ) -> Optional[float]:
        """
        Check if a triplet is semantically similar to existing ones.
        
        Args:
            triplet: (subject, predicate, object) tuple
            
        Returns:
            Highest similarity score if above threshold, None otherwise
        """
        if self.indexer._store is None or self.indexer.store.size == 0:
            return None
        
        # Embed the triplet
        text = self._triplet_to_text(triplet)
        embedding = self.indexer.embedder.embed_texts(
            [text],
            show_progress=False,
            normalize=True,
        )[0]
        
        # Search for similar triplets
        results = self.indexer.store.search(embedding, top_k=1)
        
        if results and results[0].score >= self.similarity_threshold:
            return results[0].score
        
        return None
    
    def filter_duplicates(
        self,
        triplets: List[Tuple[str, str, str]],
        check_semantic: bool = True,
    ) -> DuplicateCheckResult:
        """
        Filter out duplicate triplets from the input list.
        
        Checks for both exact matches and semantic similarity.
        
        Args:
            triplets: List of (subject, predicate, object) tuples
            check_semantic: Whether to check semantic similarity (slower)
            
        Returns:
            DuplicateCheckResult with new, exact duplicate, and semantic duplicate lists
        """
        new_triplets = []
        exact_duplicates = []
        semantic_duplicates = []
        
        # Track triplets we've already seen in this batch
        seen_in_batch: Set[Tuple[str, str, str]] = set()
        
        for triplet in triplets:
            normalized = self._normalize_triplet(triplet)
            
            # Skip if already processed in this batch
            if normalized in seen_in_batch:
                exact_duplicates.append(triplet)
                continue
            
            seen_in_batch.add(normalized)
            
            # Check exact duplicate
            if self._is_exact_duplicate(triplet):
                exact_duplicates.append(triplet)
                logger.debug(f"Exact duplicate: {triplet}")
                continue
            
            # Check semantic duplicate
            if check_semantic:
                similarity = self._check_semantic_duplicate(triplet)
                if similarity is not None:
                    semantic_duplicates.append((triplet, similarity))
                    logger.debug(f"Semantic duplicate (score={similarity:.3f}): {triplet}")
                    continue
            
            # Not a duplicate
            new_triplets.append(triplet)
        
        result = DuplicateCheckResult(
            new_triplets=new_triplets,
            exact_duplicates=exact_duplicates,
            semantic_duplicates=semantic_duplicates,
        )
        
        logger.info(
            f"Duplicate check: {len(triplets)} input -> "
            f"{len(new_triplets)} new, {len(exact_duplicates)} exact dup, "
            f"{len(semantic_duplicates)} semantic dup"
        )
        
        return result
    
    def invalidate_cache(self) -> None:
        """Invalidate the cached existing keys (call after adding new triplets)."""
        self._existing_keys = None
        logger.debug("Invalidated duplicate detector cache")


def get_duplicate_detector(
    indexer,
    similarity_threshold: float = 0.95,
) -> DuplicateDetector:
    """
    Factory function to create a DuplicateDetector.
    
    Args:
        indexer: A loaded KGRagIndexer instance
        similarity_threshold: Similarity threshold for semantic duplicates
        
    Returns:
        Configured DuplicateDetector instance
    """
    return DuplicateDetector(
        indexer=indexer,
        similarity_threshold=similarity_threshold,
    )
