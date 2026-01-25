"""
Main KG-RAG Indexer that orchestrates the full pipeline.

Combines:
1. TripletLoader - Load and parse triplets from canon_kg.txt
2. TripletRepresenter - Convert to embeddable format
3. Embedder - Generate embeddings
4. Store - FAISS or Neo4j for storage and retrieval

Supports multiple storage backends:
- FAISS: Fast vector similarity search (default)
- Neo4j: Graph database with vector search and graph traversal

Usage:
    # FAISS backend (default)
    indexer = KGRagIndexer()
    indexer.index_from_path("./output/tmp")
    
    # Neo4j backend
    indexer = KGRagIndexer(
        store_type="neo4j",
        neo4j_uri="bolt://localhost:7687",
        neo4j_password="password123",
    )
    indexer.index_from_path("./output/tmp")
"""

import os
import json
import logging
from typing import List, Optional, Dict, Any, Tuple, Union, TYPE_CHECKING

from .triplet_loader import TripletLoader, Triplet
from .representation import TripletRepresenter, RepresentationMode, EmbeddableItem, get_representer, humanize
from .embedder import Embedder, get_embedder, DEFAULT_MODEL
from .faiss_store import FaissStore, SearchResult

# Type alias for store backends (Neo4jStore imported lazily)
# Using Any here to avoid circular import issues
if TYPE_CHECKING:
    from .neo4j_store import Neo4jStore
    StoreType = Union[FaissStore, Neo4jStore]
else:
    StoreType = Any

logger = logging.getLogger(__name__)


def _get_neo4j_store():
    """Lazy import Neo4jStore to avoid import errors when neo4j is not installed."""
    from .neo4j_store import Neo4jStore
    return Neo4jStore


class KGRagIndexer:
    """
    Knowledge Graph RAG Indexer.
    
    Full pipeline for indexing KG triplets for retrieval:
    1. Load triplets from EDC pipeline output (canon_kg.txt)
    2. Convert to embeddable representations
    3. Generate embeddings
    4. Store in FAISS index or Neo4j database
    
    Supports two storage backends:
    - "faiss": Fast vector similarity search (default, no external dependencies)
    - "neo4j": Graph database with vector search + graph traversal capabilities
    
    Usage:
        # FAISS backend (default)
        indexer = KGRagIndexer()
        indexer.index_from_path("./output/tmp", mode="triplet_text")
        indexer.save("./output/rag")
        
        # Neo4j backend
        indexer = KGRagIndexer(
            store_type="neo4j",
            neo4j_uri="bolt://localhost:7687",
            neo4j_password="password123",
        )
        indexer.index_from_path("./output/tmp", mode="triplet_text")
        
        # Later, for retrieval:
        indexer = KGRagIndexer.load("./output/rag")
        results = indexer.search("Where is Trane located?")
    """
    
    def __init__(
        self,
        embedding_model: Optional[str] = None,
        device: Optional[str] = None,
        use_gpu_faiss: bool = False,
        store_type: str = "faiss",
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: Optional[str] = None,
        neo4j_database: str = "neo4j",
    ):
        """
        Initialize the indexer.
        
        Args:
            embedding_model: Sentence transformer model name
            device: Device for embeddings ("cuda", "cpu", or None for auto)
            use_gpu_faiss: Whether to use GPU for FAISS (only for faiss backend)
            store_type: Storage backend - "faiss" or "neo4j"
            neo4j_uri: Neo4j Bolt URI (only for neo4j backend)
            neo4j_user: Neo4j username (only for neo4j backend)
            neo4j_password: Neo4j password (only for neo4j backend, defaults to env var)
            neo4j_database: Neo4j database name (only for neo4j backend)
        """
        self.embedding_model_name = embedding_model or DEFAULT_MODEL
        self.device = device
        self.use_gpu_faiss = use_gpu_faiss
        
        # Store configuration
        self.store_type = store_type.lower()
        if self.store_type not in ("faiss", "neo4j"):
            raise ValueError(f"store_type must be 'faiss' or 'neo4j', got '{store_type}'")
        
        # Neo4j configuration
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password or os.environ.get("NEO4J_PASSWORD", "password123")
        self.neo4j_database = neo4j_database
        
        # Components (lazy-loaded)
        self._embedder: Optional[Embedder] = None
        self._store: Optional[StoreType] = None
        
        # State
        self.triplets: List[Triplet] = []
        self.items: List[EmbeddableItem] = []
        self.representation_mode: Optional[str] = None
        
    @property
    def embedder(self) -> Embedder:
        """Lazy-load the embedder."""
        if self._embedder is None:
            self._embedder = get_embedder(
                model_name=self.embedding_model_name,
                device=self.device,
            )
        return self._embedder
    
    @property
    def store(self) -> StoreType:
        """Get the store (must be initialized first)."""
        if self._store is None:
            raise ValueError("Store not initialized. Call index_from_path() or load() first.")
        return self._store
    
    def _create_store(self, embedding_dim: int) -> StoreType:
        """
        Create the appropriate store based on store_type.
        
        Args:
            embedding_dim: Dimension of embeddings
            
        Returns:
            FaissStore or Neo4jStore instance
        """
        if self.store_type == "faiss":
            return FaissStore(
                embedding_dim=embedding_dim,
                use_gpu=self.use_gpu_faiss,
            )
        else:  # neo4j
            Neo4jStore = _get_neo4j_store()
            return Neo4jStore(
                uri=self.neo4j_uri,
                user=self.neo4j_user,
                password=self.neo4j_password,
                database=self.neo4j_database,
                embedding_dim=embedding_dim,
            )
    
    def index_from_path(
        self,
        path: str,
        mode: str = "triplet_text",
        batch_size: int = 32,
        show_progress: bool = True,
        deduplicate: bool = True,
        normalize: bool = True,
    ) -> "KGRagIndexer":
        """
        Index triplets from EDC pipeline output (canon_kg.txt).
        
        Automatically finds the latest canon_kg.txt in the output directory.
        
        Args:
            path: Path to canon_kg.txt or EDC output directory
            mode: Representation mode ("triplet_text" or "entity_context")
            batch_size: Batch size for embedding
            show_progress: Whether to show progress bars
            deduplicate: If True, remove duplicate triplets
            normalize: If True, normalize entity names (replace underscores with spaces)
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Starting indexing pipeline for {path}")
        logger.info(f"Using mode: {mode}")
        
        self.representation_mode = mode
        
        # Step 1: Load triplets from canon_kg.txt
        logger.info("Step 1: Loading triplets from canon_kg.txt...")
        loader = TripletLoader(path)
        loader.load()
        self.triplets = loader.parse(deduplicate=deduplicate, normalize=normalize)
        logger.info(f"Loaded {len(self.triplets)} unique triplets")
        
        if len(self.triplets) == 0:
            raise ValueError("No triplets found. Check your input file/directory.")
        
        # Step 2: Convert to representations
        logger.info("Step 2: Converting to representations...")
        representer = get_representer(mode)
        self.items = representer.convert(self.triplets)
        logger.info(f"Created {len(self.items)} embeddable items")
        
        if len(self.items) == 0:
            raise ValueError(
                "No embeddable items created. "
                "Use 'triplet_text' or 'entity_context' mode."
            )
        
        # Step 3: Generate embeddings
        logger.info("Step 3: Generating embeddings...")
        embeddings = self.embedder.embed_items(
            self.items,
            batch_size=batch_size,
            show_progress=show_progress,
            normalize=True,
        )
        logger.info(f"Generated embeddings with shape {embeddings.shape}")
        
        # Step 4: Store in index
        store_name = "Neo4j" if self.store_type == "neo4j" else "FAISS"
        logger.info(f"Step 4: Storing in {store_name} index...")
        self._store = self._create_store(self.embedder.embedding_dim)
        self._store.add(embeddings, self.items)
        logger.info(f"Indexed {self._store.size} vectors in {store_name}")
        
        logger.info("Indexing complete!")
        return self
    
    def index_triplets(
        self,
        triplets: List[Triplet],
        mode: str = "triplet_text",
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> "KGRagIndexer":
        """
        Index triplets directly (without loading from file).
        
        Args:
            triplets: List of Triplet objects
            mode: Representation mode
            batch_size: Batch size for embedding
            show_progress: Whether to show progress bars
            
        Returns:
            Self for method chaining
        """
        self.representation_mode = mode
        self.triplets = triplets
        
        # Step 2: Convert to representations
        representer = get_representer(mode)
        self.items = representer.convert(self.triplets)
        
        # Step 3: Generate embeddings
        embeddings = self.embedder.embed_items(
            self.items,
            batch_size=batch_size,
            show_progress=show_progress,
            normalize=True,
        )
        
        # Step 4: Store in index
        self._store = self._create_store(self.embedder.embedding_dim)
        self._store.add(embeddings, self.items)
        
        return self
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        use_instruction: bool = True,
    ) -> List[SearchResult]:
        """
        Search for relevant triplets given a query.
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            use_instruction: Whether to use instruction prefix for query
            
        Returns:
            List of SearchResult objects
        """
        # Embed query
        query_embedding = self.embedder.embed_query(
            query,
            use_instruction=use_instruction,
            normalize=True,
        )
        
        # Search
        results = self.store.search(query_embedding, top_k=top_k)
        
        return results
    
    def search_batch(
        self,
        queries: List[str],
        top_k: int = 10,
        use_instruction: bool = True,
    ) -> List[List[SearchResult]]:
        """
        Batch search for multiple queries.
        
        Args:
            queries: List of query strings
            top_k: Number of results per query
            use_instruction: Whether to use instruction prefix
            
        Returns:
            List of lists of SearchResult objects
        """
        # Embed queries
        query_embeddings = self.embedder.embed_queries(
            queries,
            use_instruction=use_instruction,
            normalize=True,
        )
        
        # Batch search
        results = self.store.search_batch(query_embeddings, top_k=top_k)
        
        return results
    
    def save(self, directory: str, prefix: str = "kg_triplets") -> None:
        """
        Save the index and metadata to disk.
        
        For FAISS: Saves index file and metadata JSON
        For Neo4j: Saves configuration only (data persists in database)
        
        Args:
            directory: Output directory
            prefix: Filename prefix
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save store-specific data
        self.store.save(directory, prefix=prefix)
        
        # Save config (includes store type for proper loading)
        config = {
            "store_type": self.store_type,
            "embedding_model": self.embedding_model_name,
            "representation_mode": self.representation_mode,
            "num_triplets": len(self.triplets),
            "num_items": len(self.items),
            "embedding_dim": self.embedder.embedding_dim,
        }
        
        # Add Neo4j-specific config if needed
        if self.store_type == "neo4j":
            config["neo4j_uri"] = self.neo4j_uri
            config["neo4j_user"] = self.neo4j_user
            config["neo4j_database"] = self.neo4j_database
            # Note: password is NOT saved for security
        
        config_path = os.path.join(directory, f"{prefix}_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved {self.store_type} index config to {directory}")
    
    @classmethod
    def load(
        cls,
        directory: str,
        prefix: str = "kg_triplets",
        device: Optional[str] = None,
        use_gpu_faiss: bool = False,
        neo4j_password: Optional[str] = None,
    ) -> "KGRagIndexer":
        """
        Load a saved index.
        
        Automatically detects the store type from saved configuration.
        
        Args:
            directory: Directory containing saved files
            prefix: Filename prefix
            device: Device for embeddings
            use_gpu_faiss: Whether to use GPU for FAISS (faiss backend only)
            neo4j_password: Neo4j password (neo4j backend only, or from env)
            
        Returns:
            Loaded KGRagIndexer instance
        """
        # Load config
        config_path = os.path.join(directory, f"{prefix}_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Detect store type (default to faiss for backward compatibility)
        store_type = config.get("store_type", "faiss")
        
        # Create indexer with appropriate settings
        indexer = cls(
            embedding_model=config.get("embedding_model"),
            device=device,
            use_gpu_faiss=use_gpu_faiss,
            store_type=store_type,
            neo4j_uri=config.get("neo4j_uri", "bolt://localhost:7687"),
            neo4j_user=config.get("neo4j_user", "neo4j"),
            neo4j_password=neo4j_password,
            neo4j_database=config.get("neo4j_database", "neo4j"),
        )
        
        indexer.representation_mode = config.get("representation_mode")
        
        # Load the appropriate store
        if store_type == "faiss":
            indexer._store = FaissStore.load(
                directory,
                prefix=prefix,
                use_gpu=use_gpu_faiss,
            )
        else:  # neo4j
            Neo4jStore = _get_neo4j_store()
            indexer._store = Neo4jStore.load(
                directory,
                prefix=prefix,
                password=indexer.neo4j_password,
            )
        
        logger.info(f"Loaded {store_type} index with {indexer.store.size} vectors")
        
        return indexer
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed data."""
        stats = {
            "store_type": self.store_type,
            "num_triplets": len(self.triplets),
            "num_items": len(self.items),
            "num_vectors": self.store.size if self._store else 0,
            "embedding_model": self.embedding_model_name,
            "embedding_dim": self.embedder.embedding_dim if self._embedder else None,
            "representation_mode": self.representation_mode,
        }
        
        if self.store_type == "neo4j":
            stats["neo4j_uri"] = self.neo4j_uri
            stats["neo4j_database"] = self.neo4j_database
        
        return stats
    
    def graph_search(
        self,
        entity: str,
        relationship: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform graph traversal search (Neo4j backend only).
        
        Find all triplets involving an entity, optionally filtered by relationship.
        
        Args:
            entity: Entity name to search for
            relationship: Optional relationship type to filter by
            
        Returns:
            List of triplet dictionaries
            
        Raises:
            ValueError: If called on FAISS backend
        """
        if self.store_type != "neo4j":
            raise ValueError("graph_search() is only available with Neo4j backend")
        
        return self.store.graph_search(entity, relationship)
    
    def graph_expand(
        self,
        entity: str,
        depth: int = 2,
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Expand from an entity to find connected triplets (Neo4j backend only).
        
        Args:
            entity: Starting entity
            depth: Maximum number of hops
            max_results: Maximum number of results
            
        Returns:
            List of triplet dictionaries
            
        Raises:
            ValueError: If called on FAISS backend
        """
        if self.store_type != "neo4j":
            raise ValueError("graph_expand() is only available with Neo4j backend")
        
        return self.store.graph_expand(entity, depth=depth, max_results=max_results)
    
    def add_expanded_triplets(
        self,
        triplets: List[Tuple[str, str, str]],
        check_duplicates: bool = True,
        similarity_threshold: float = 0.95,
        source_tag: str = "llm_expanded",
    ) -> int:
        """
        Add expanded triplets to the existing index.
        
        Filters out duplicates (both exact and semantic) before adding.
        
        Args:
            triplets: List of (subject, predicate, object) tuples
            check_duplicates: Whether to check for duplicates before adding
            similarity_threshold: Cosine similarity threshold for semantic duplicates
            source_tag: Tag to mark the source of these triplets in metadata
            
        Returns:
            Number of triplets actually added (after duplicate filtering)
        """
        if not triplets:
            return 0
        
        if self._store is None:
            raise ValueError("Store not initialized. Cannot add triplets to uninitialized index.")
        
        logger.info(f"Adding {len(triplets)} expanded triplets to index")
        
        # Filter duplicates if requested
        triplets_to_add = triplets
        if check_duplicates:
            from .duplicate_detector import DuplicateDetector
            
            detector = DuplicateDetector(self, similarity_threshold=similarity_threshold)
            result = detector.filter_duplicates(triplets)
            triplets_to_add = result.new_triplets
            
            if len(result.exact_duplicates) > 0 or len(result.semantic_duplicates) > 0:
                logger.info(
                    f"Filtered {len(result.exact_duplicates)} exact duplicates and "
                    f"{len(result.semantic_duplicates)} semantic duplicates"
                )
        
        if not triplets_to_add:
            logger.info("No new triplets to add after duplicate filtering")
            return 0
        
        # Convert triplets to embeddable items
        items = []
        for s, p, o in triplets_to_add:
            # Create text representation (same as triplet_text mode)
            text = f"{humanize(s)} {humanize(p)} {humanize(o)}"
            
            metadata = {
                "subject": s,
                "predicate": p,
                "object": o,
                "source_text": "",
                "document": f"({s}, {p}, {o})",
                "representation_mode": "triplet_text",
                "source": source_tag,  # Mark as expanded triplet
            }
            
            items.append(EmbeddableItem(text=text, metadata=metadata))
        
        # Generate embeddings
        texts = [item.text for item in items]
        embeddings = self.embedder.embed_texts(
            texts,
            show_progress=False,
            normalize=True,
        )
        
        # Add to store
        self._store.add(embeddings, items)
        
        # Update internal state
        for s, p, o in triplets_to_add:
            self.triplets.append(Triplet(subject=s, predicate=p, obj=o))
        self.items.extend(items)
        
        logger.info(f"Added {len(triplets_to_add)} new triplets to index (total: {self._store.size})")
        
        return len(triplets_to_add)
    
    def get_existing_triplet_keys(self) -> List[Tuple[str, str, str]]:
        """
        Get all existing triplet keys for duplicate checking.
        
        Returns:
            List of (subject, predicate, object) tuples
        """
        if self._store is None:
            return []
        return self.store.get_triplet_keys()