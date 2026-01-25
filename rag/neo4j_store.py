"""
Neo4j Knowledge Graph Store for KG-RAG.

Provides graph-native storage with optional vector similarity search.
Uses Neo4j's native vector index (Neo4j 5.11+) for semantic search.

This module implements the same interface as FaissStore but uses Neo4j,
enabling graph traversal capabilities alongside vector similarity search.
"""

import os
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

from neo4j import GraphDatabase
import numpy as np

from .representation import EmbeddableItem

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from a similarity search."""
    score: float
    metadata: Dict[str, Any]
    rank: int
    
    def __repr__(self) -> str:
        doc = self.metadata.get("document", "")[:50]
        return f"SearchResult(rank={self.rank}, score={self.score:.4f}, doc='{doc}...')"


class Neo4jStore:
    """
    Knowledge Graph store using Neo4j.
    
    Features:
    - Native graph traversal for relationship queries
    - Vector similarity via Neo4j vector index (5.11+)
    - Cypher query generation for complex questions
    - Same interface as FaissStore for drop-in replacement
    
    Usage:
        store = Neo4jStore(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password123",
            embedding_dim=384,
        )
        
        # Add triplets
        store.add(embeddings, items)
        
        # Vector similarity search
        results = store.search(query_embedding, top_k=10)
        
        # Graph traversal search
        results = store.graph_search("Einstein", relationship="discovered")
    """
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password123",
        database: str = "neo4j",
        embedding_dim: int = 384,
        create_indexes: bool = True,
    ):
        """
        Initialize the Neo4j store.
        
        Args:
            uri: Neo4j Bolt URI
            user: Neo4j username
            password: Neo4j password
            database: Database name
            embedding_dim: Dimension of the embeddings
            create_indexes: Whether to create indexes on initialization
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.embedding_dim = embedding_dim
        
        # Connect to Neo4j
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # Verify connection
        try:
            with self.driver.session(database=database) as session:
                session.run("RETURN 1")
            logger.info(f"Connected to Neo4j at {uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
        
        # Create indexes
        if create_indexes:
            self._ensure_indexes()
    
    def _ensure_indexes(self):
        """Create necessary indexes including vector index."""
        with self.driver.session(database=self.database) as session:
            # Create constraint for unique triplet identification
            try:
                session.run("""
                    CREATE CONSTRAINT triplet_unique IF NOT EXISTS
                    FOR (t:Triplet) REQUIRE (t.subject, t.predicate, t.object) IS UNIQUE
                """)
                logger.info("Created unique constraint on Triplet nodes")
            except Exception as e:
                logger.debug(f"Constraint creation: {e}")
            
            # Create index on subject for faster graph traversal
            try:
                session.run("""
                    CREATE INDEX triplet_subject IF NOT EXISTS
                    FOR (t:Triplet) ON (t.subject)
                """)
                logger.info("Created index on subject")
            except Exception as e:
                logger.debug(f"Subject index: {e}")
            
            # Create index on object for faster graph traversal
            try:
                session.run("""
                    CREATE INDEX triplet_object IF NOT EXISTS
                    FOR (t:Triplet) ON (t.object)
                """)
                logger.info("Created index on object")
            except Exception as e:
                logger.debug(f"Object index: {e}")
            
            # Create vector index for semantic search (Neo4j 5.11+)
            try:
                session.run(f"""
                    CREATE VECTOR INDEX triplet_embedding IF NOT EXISTS
                    FOR (t:Triplet)
                    ON t.embedding
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: {self.embedding_dim},
                            `vector.similarity_function`: 'cosine'
                        }}
                    }}
                """)
                logger.info(f"Created vector index with dimension {self.embedding_dim}")
            except Exception as e:
                logger.warning(f"Vector index creation failed (requires Neo4j 5.11+): {e}")
    
    def add(
        self,
        embeddings: np.ndarray,
        items: List[EmbeddableItem],
    ) -> None:
        """
        Add embeddings and their metadata to the store.
        
        Args:
            embeddings: numpy array of shape (n, embedding_dim)
            items: List of EmbeddableItem objects (must match embeddings length)
        """
        if len(embeddings) != len(items):
            raise ValueError(
                f"Embeddings length ({len(embeddings)}) must match items length ({len(items)})"
            )
        
        # Ensure embeddings are float32
        embeddings = embeddings.astype(np.float32)
        
        added_count = 0
        with self.driver.session(database=self.database) as session:
            for embedding, item in zip(embeddings, items):
                meta = item.metadata
                try:
                    session.run("""
                        MERGE (t:Triplet {
                            subject: $subject,
                            predicate: $predicate,
                            object: $object
                        })
                        SET t.document = $document,
                            t.source_text = $source_text,
                            t.embedding = $embedding,
                            t.representation_mode = $mode,
                            t.source = $source
                    """, {
                        "subject": meta.get("subject", ""),
                        "predicate": meta.get("predicate", ""),
                        "object": meta.get("object", ""),
                        "document": meta.get("document", ""),
                        "source_text": meta.get("source_text", ""),
                        "embedding": embedding.tolist(),
                        "mode": meta.get("representation_mode", "triplet_text"),
                        "source": meta.get("source", "original"),
                    })
                    added_count += 1
                except Exception as e:
                    logger.warning(f"Failed to add triplet: {e}")
        
        logger.info(f"Added {added_count} triplets to Neo4j. Total: {self.size}")
    
    def add_with_metadata(
        self,
        embeddings: np.ndarray,
        metadata_list: List[Dict[str, Any]],
    ) -> None:
        """
        Add embeddings with raw metadata dictionaries (no EmbeddableItem required).
        
        Useful for adding expanded triplets directly.
        
        Args:
            embeddings: numpy array of shape (n, embedding_dim)
            metadata_list: List of metadata dictionaries
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError(
                f"Embeddings length ({len(embeddings)}) must match metadata length ({len(metadata_list)})"
            )
        
        # Create EmbeddableItem wrappers
        items = [
            EmbeddableItem(text="", metadata=meta)
            for meta in metadata_list
        ]
        
        self.add(embeddings, items)
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """
        Search for similar vectors using Neo4j vector index.
        
        Args:
            query_embedding: Query vector of shape (embedding_dim,) or (1, embedding_dim)
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        # Reshape if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = query_embedding.astype(np.float32)
        
        results = []
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run("""
                    CALL db.index.vector.queryNodes(
                        'triplet_embedding',
                        $top_k,
                        $embedding
                    )
                    YIELD node, score
                    RETURN node.subject AS subject,
                           node.predicate AS predicate,
                           node.object AS object,
                           node.document AS document,
                           node.source_text AS source_text,
                           node.representation_mode AS mode,
                           node.source AS source,
                           score
                """, {
                    "top_k": top_k,
                    "embedding": query_embedding[0].tolist(),
                })
                
                for rank, record in enumerate(result):
                    results.append(SearchResult(
                        score=float(record["score"]),
                        metadata={
                            "subject": record["subject"],
                            "predicate": record["predicate"],
                            "object": record["object"],
                            "document": record["document"],
                            "source_text": record["source_text"] or "",
                            "representation_mode": record["mode"] or "triplet_text",
                            "source": record["source"] or "original",
                        },
                        rank=rank,
                    ))
            except Exception as e:
                logger.error(f"Vector search failed: {e}")
                # Fallback to brute-force search if vector index not available
                results = self._brute_force_search(query_embedding[0], top_k)
        
        return results
    
    def _brute_force_search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> List[SearchResult]:
        """
        Fallback brute-force cosine similarity search.
        
        Used when vector index is not available (Neo4j < 5.11).
        """
        logger.warning("Using brute-force search (vector index not available)")
        
        results = []
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (t:Triplet)
                WHERE t.embedding IS NOT NULL
                RETURN t.subject AS subject,
                       t.predicate AS predicate,
                       t.object AS object,
                       t.document AS document,
                       t.source_text AS source_text,
                       t.representation_mode AS mode,
                       t.source AS source,
                       t.embedding AS embedding
            """)
            
            # Compute similarities
            scored_results = []
            for record in result:
                embedding = np.array(record["embedding"], dtype=np.float32)
                # Cosine similarity
                score = float(np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding) + 1e-8
                ))
                scored_results.append((score, record))
            
            # Sort by score descending
            scored_results.sort(key=lambda x: x[0], reverse=True)
            
            # Take top_k
            for rank, (score, record) in enumerate(scored_results[:top_k]):
                results.append(SearchResult(
                    score=score,
                    metadata={
                        "subject": record["subject"],
                        "predicate": record["predicate"],
                        "object": record["object"],
                        "document": record["document"],
                        "source_text": record["source_text"] or "",
                        "representation_mode": record["mode"] or "triplet_text",
                        "source": record["source"] or "original",
                    },
                    rank=rank,
                ))
        
        return results
    
    def search_batch(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 10,
    ) -> List[List[SearchResult]]:
        """
        Batch search for multiple queries.
        
        Args:
            query_embeddings: Query vectors of shape (n_queries, embedding_dim)
            top_k: Number of results per query
            
        Returns:
            List of lists of SearchResult objects
        """
        all_results = []
        for query_embedding in query_embeddings:
            results = self.search(query_embedding, top_k=top_k)
            all_results.append(results)
        
        return all_results
    
    def graph_search(
        self,
        entity: str,
        relationship: Optional[str] = None,
        max_results: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Graph-native search - find related triplets via graph traversal.
        
        This is where Neo4j shines over FAISS - you can traverse relationships!
        
        Args:
            entity: Entity name to search for (as subject or object)
            relationship: Optional predicate to filter by
            max_results: Maximum number of results
            
        Returns:
            List of triplet dictionaries
        """
        with self.driver.session(database=self.database) as session:
            if relationship:
                query = """
                    MATCH (t:Triplet)
                    WHERE (toLower(t.subject) = toLower($entity) 
                           OR toLower(t.object) = toLower($entity))
                    AND toLower(t.predicate) = toLower($relationship)
                    RETURN t.subject AS subject, t.predicate AS predicate, 
                           t.object AS object, t.document AS document
                    LIMIT $limit
                """
                params = {"entity": entity, "relationship": relationship, "limit": max_results}
            else:
                query = """
                    MATCH (t:Triplet)
                    WHERE toLower(t.subject) = toLower($entity) 
                          OR toLower(t.object) = toLower($entity)
                    RETURN t.subject AS subject, t.predicate AS predicate,
                           t.object AS object, t.document AS document
                    LIMIT $limit
                """
                params = {"entity": entity, "limit": max_results}
            
            result = session.run(query, params)
            return [dict(record) for record in result]
    
    def graph_expand(
        self,
        entity: str,
        depth: int = 2,
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Expand from an entity to find connected triplets within N hops.
        
        Args:
            entity: Starting entity
            depth: Maximum number of hops
            max_results: Maximum number of results
            
        Returns:
            List of triplet dictionaries
        """
        with self.driver.session(database=self.database) as session:
            # First hop: direct connections
            results = []
            visited_entities = {entity.lower()}
            current_entities = [entity]
            
            for _ in range(depth):
                if not current_entities:
                    break
                
                # Find all triplets involving current entities
                query = """
                    MATCH (t:Triplet)
                    WHERE toLower(t.subject) IN $entities 
                          OR toLower(t.object) IN $entities
                    RETURN DISTINCT t.subject AS subject, t.predicate AS predicate,
                           t.object AS object, t.document AS document
                    LIMIT $limit
                """
                
                result = session.run(query, {
                    "entities": [e.lower() for e in current_entities],
                    "limit": max_results - len(results),
                })
                
                next_entities = []
                for record in result:
                    triplet = dict(record)
                    results.append(triplet)
                    
                    # Collect new entities for next hop
                    for entity_field in ["subject", "object"]:
                        e = triplet[entity_field].lower()
                        if e not in visited_entities:
                            visited_entities.add(e)
                            next_entities.append(triplet[entity_field])
                
                current_entities = next_entities
                
                if len(results) >= max_results:
                    break
            
            return results[:max_results]
    
    def cypher_search(self, cypher_query: str) -> List[Dict[str, Any]]:
        """
        Execute a raw Cypher query (for MCP tool use).
        
        Args:
            cypher_query: Cypher query string
            
        Returns:
            List of result dictionaries
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher_query)
            return [dict(record) for record in result]
    
    def get_triplet_keys(self) -> List[Tuple[str, str, str]]:
        """
        Extract all triplet keys for duplicate checking.
        
        Returns:
            List of (subject, predicate, object) tuples
        """
        keys = []
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (t:Triplet)
                RETURN t.subject AS subject, t.predicate AS predicate, t.object AS object
            """)
            
            for record in result:
                keys.append((
                    record["subject"],
                    record["predicate"],
                    record["object"],
                ))
        
        return keys
    
    def find_similar(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.95,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Find vectors similar to the query above a threshold.
        
        Args:
            query_embedding: Query vector
            threshold: Minimum similarity score
            top_k: Maximum number of results to check
            
        Returns:
            List of SearchResult objects with score >= threshold
        """
        results = self.search(query_embedding, top_k=top_k)
        return [r for r in results if r.score >= threshold]
    
    def has_similar(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.95,
    ) -> bool:
        """
        Check if any similar vector exists above the threshold.
        
        Args:
            query_embedding: Query vector
            threshold: Minimum similarity score
            
        Returns:
            True if at least one similar vector exists
        """
        similar = self.find_similar(query_embedding, threshold=threshold, top_k=1)
        return len(similar) > 0
    
    @property
    def size(self) -> int:
        """Number of triplets in the store."""
        with self.driver.session(database=self.database) as session:
            result = session.run("MATCH (t:Triplet) RETURN count(t) AS count")
            return result.single()["count"]
    
    def save(self, directory: str, prefix: str = "kg_triplets") -> Tuple[str, str]:
        """
        Save store configuration (Neo4j data persists automatically).
        
        Args:
            directory: Directory to save config to
            prefix: Filename prefix
            
        Returns:
            Tuple of (config_path, None) - Neo4j handles data persistence
        """
        import json
        os.makedirs(directory, exist_ok=True)
        
        config_path = os.path.join(directory, f"{prefix}_neo4j_config.json")
        config = {
            "store_type": "neo4j",
            "uri": self.uri,
            "user": self.user,
            "database": self.database,
            "embedding_dim": self.embedding_dim,
            "size": self.size,
        }
        
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved Neo4j store config to {config_path}")
        
        return config_path, None
    
    @classmethod
    def load(
        cls,
        directory: str,
        prefix: str = "kg_triplets",
        password: Optional[str] = None,
    ) -> "Neo4jStore":
        """
        Load a Neo4j store from saved configuration.
        
        Args:
            directory: Directory containing saved config
            prefix: Filename prefix
            password: Neo4j password (required, not stored in config)
            
        Returns:
            Loaded Neo4jStore instance
        """
        import json
        
        config_path = os.path.join(directory, f"{prefix}_neo4j_config.json")
        
        with open(config_path, "r") as f:
            config = json.load(f)
        
        if password is None:
            password = os.environ.get("NEO4J_PASSWORD", "password123")
        
        store = cls(
            uri=config["uri"],
            user=config["user"],
            password=password,
            database=config["database"],
            embedding_dim=config["embedding_dim"],
            create_indexes=False,  # Indexes should already exist
        )
        
        logger.info(f"Loaded Neo4j store with {store.size} triplets")
        
        return store
    
    def clear(self) -> None:
        """Delete all triplets from the store."""
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (t:Triplet) DELETE t")
        logger.info("Cleared all triplets from Neo4j store")
    
    def close(self):
        """Close the Neo4j driver connection."""
        self.driver.close()
        logger.info("Closed Neo4j connection")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    # Simple test
    import tempfile
    
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Neo4jStore...")
    print("Note: Requires Neo4j running at bolt://localhost:7687")
    
    try:
        # Create store
        store = Neo4jStore(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password123",
            embedding_dim=384,
        )
        
        print(f"Connected! Current store size: {store.size}")
        
        # Test graph search
        results = store.graph_search("Einstein")
        print(f"\nGraph search for 'Einstein': {len(results)} results")
        for r in results[:3]:
            print(f"  ({r['subject']}, {r['predicate']}, {r['object']})")
        
        store.close()
        print("\nTest completed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        print("Make sure Neo4j is running and accessible.")
