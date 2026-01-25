#!/usr/bin/env python3
"""
Migrate FAISS index to Neo4j Knowledge Graph Store.

This script transfers triplet data and embeddings from an existing FAISS index
to a Neo4j database, enabling graph traversal capabilities alongside vector search.

Usage:
    # Basic migration
    python scripts/migrate_faiss_to_neo4j.py --faiss-dir ./output/rag
    
    # With custom Neo4j connection
    python scripts/migrate_faiss_to_neo4j.py \
        --faiss-dir ./output/rag \
        --neo4j-uri bolt://localhost:7687 \
        --neo4j-user neo4j \
        --neo4j-password mypassword
    
    # Re-embed triplets (if embeddings are missing or need updating)
    python scripts/migrate_faiss_to_neo4j.py \
        --faiss-dir ./output/rag \
        --re-embed \
        --embedding-model BAAI/bge-small-en-v1.5

Environment Variables:
    NEO4J_PASSWORD: Neo4j password (alternative to --neo4j-password)
    LOCAL_EMBEDDER_MODEL: Default embedding model
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag.faiss_store import FaissStore
from rag.neo4j_store import Neo4jStore
from rag.representation import EmbeddableItem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def migrate_faiss_to_neo4j(
    faiss_dir: str,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password123",
    neo4j_database: str = "neo4j",
    batch_size: int = 100,
    re_embed: bool = False,
    embedding_model: Optional[str] = None,
    clear_existing: bool = False,
    prefix: str = "kg_triplets",
) -> int:
    """
    Migrate FAISS index data to Neo4j.
    
    Args:
        faiss_dir: Directory containing saved FAISS index
        neo4j_uri: Neo4j Bolt URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        neo4j_database: Neo4j database name
        batch_size: Number of triplets to process at a time
        re_embed: Whether to re-generate embeddings
        embedding_model: Model to use for re-embedding
        clear_existing: Whether to clear existing Neo4j data
        prefix: FAISS file prefix
        
    Returns:
        Number of triplets migrated
    """
    logger.info("=" * 60)
    logger.info("FAISS to Neo4j Migration")
    logger.info("=" * 60)
    
    # Load FAISS store
    logger.info(f"Loading FAISS index from {faiss_dir}")
    try:
        faiss_store = FaissStore.load(faiss_dir, prefix=prefix)
    except FileNotFoundError:
        logger.error(f"FAISS index not found at {faiss_dir}")
        logger.error(f"Expected files: {prefix}.faiss, {prefix}_meta.json")
        return 0
    
    logger.info(f"Loaded {faiss_store.size} vectors from FAISS")
    logger.info(f"Embedding dimension: {faiss_store.embedding_dim}")
    
    # Connect to Neo4j
    logger.info(f"Connecting to Neo4j at {neo4j_uri}")
    try:
        neo4j_store = Neo4jStore(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password,
            database=neo4j_database,
            embedding_dim=faiss_store.embedding_dim,
        )
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        logger.error("Make sure Neo4j is running and accessible")
        return 0
    
    # Clear existing data if requested
    if clear_existing:
        logger.warning("Clearing existing Neo4j triplet data...")
        neo4j_store.clear()
    
    initial_size = neo4j_store.size
    logger.info(f"Neo4j store initial size: {initial_size}")
    
    # Get metadata and prepare for migration
    metadata = faiss_store.metadata
    logger.info(f"Processing {len(metadata)} triplets...")
    
    # Load embedder if re-embedding
    embedder = None
    if re_embed:
        from rag.embedder import get_embedder
        
        if embedding_model is None:
            embedding_model = os.environ.get(
                "LOCAL_EMBEDDER_MODEL",
                "BAAI/bge-small-en-v1.5"
            )
        
        logger.info(f"Loading embedding model: {embedding_model}")
        embedder = get_embedder(model_name=embedding_model)
    
    # Migrate in batches
    migrated_count = 0
    skipped_count = 0
    error_count = 0
    
    for i in tqdm(range(0, len(metadata), batch_size), desc="Migrating"):
        batch_meta = metadata[i:i + batch_size]
        
        # Prepare items for this batch
        items = []
        embeddings_list = []
        
        for j, meta in enumerate(batch_meta):
            # Skip invalid entries
            if not meta.get("subject") or not meta.get("predicate") or not meta.get("object"):
                skipped_count += 1
                continue
            
            # Create EmbeddableItem
            item = EmbeddableItem(
                text=meta.get("document", ""),
                metadata=meta,
            )
            items.append(item)
            
            # Get or generate embedding
            if re_embed and embedder:
                # Generate new embedding
                text = f"{meta.get('subject', '')} {meta.get('predicate', '')} {meta.get('object', '')}"
                text = text.replace("_", " ")
                embedding = embedder.embed_text(text, normalize=True)
                embeddings_list.append(embedding)
            else:
                # Try to reconstruct from FAISS (limited support)
                # FAISS doesn't directly expose embeddings, so we need to query
                idx = i + j
                if idx < faiss_store.index.ntotal:
                    try:
                        # Reconstruct embedding from index
                        embedding = faiss_store.index.reconstruct(idx)
                        embeddings_list.append(embedding)
                    except Exception:
                        # If reconstruction fails, we need to re-embed
                        if embedder is None:
                            from rag.embedder import get_embedder
                            embedder = get_embedder()
                        
                        text = f"{meta.get('subject', '')} {meta.get('predicate', '')} {meta.get('object', '')}"
                        embedding = embedder.embed_text(text.replace("_", " "), normalize=True)
                        embeddings_list.append(embedding)
        
        if not items:
            continue
        
        # Convert to numpy array
        embeddings = np.array(embeddings_list, dtype=np.float32)
        
        # Add to Neo4j
        try:
            neo4j_store.add(embeddings, items)
            migrated_count += len(items)
        except Exception as e:
            logger.warning(f"Failed to add batch: {e}")
            error_count += len(items)
    
    # Summary
    final_size = neo4j_store.size
    new_triplets = final_size - initial_size
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("Migration Summary")
    logger.info("=" * 60)
    logger.info(f"Total metadata entries: {len(metadata)}")
    logger.info(f"Migrated: {migrated_count}")
    logger.info(f"Skipped (invalid): {skipped_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"New triplets in Neo4j: {new_triplets}")
    logger.info(f"Total Neo4j triplets: {final_size}")
    logger.info("=" * 60)
    
    neo4j_store.close()
    
    return migrated_count


def verify_migration(
    faiss_dir: str,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password123",
    neo4j_database: str = "neo4j",
    sample_queries: int = 5,
    prefix: str = "kg_triplets",
) -> bool:
    """
    Verify migration by comparing search results.
    
    Args:
        faiss_dir: Directory containing saved FAISS index
        neo4j_uri: Neo4j Bolt URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        neo4j_database: Neo4j database name
        sample_queries: Number of random queries to test
        prefix: FAISS file prefix
        
    Returns:
        True if verification passes
    """
    logger.info("Verifying migration...")
    
    from rag.embedder import get_embedder
    
    # Load both stores
    faiss_store = FaissStore.load(faiss_dir, prefix=prefix)
    neo4j_store = Neo4jStore(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
        database=neo4j_database,
        embedding_dim=faiss_store.embedding_dim,
        create_indexes=False,
    )
    
    embedder = get_embedder()
    
    # Check counts
    faiss_count = faiss_store.size
    neo4j_count = neo4j_store.size
    
    logger.info(f"FAISS count: {faiss_count}")
    logger.info(f"Neo4j count: {neo4j_count}")
    
    if neo4j_count < faiss_count * 0.9:
        logger.warning("Neo4j has significantly fewer entries than FAISS")
    
    # Test sample queries
    test_queries = [
        "What is the capital of France?",
        "Who discovered relativity?",
        "Where is the Eiffel Tower located?",
        "What year was the company founded?",
        "Who is the CEO?",
    ][:sample_queries]
    
    logger.info(f"\nTesting {len(test_queries)} sample queries...")
    
    all_passed = True
    for query in test_queries:
        query_embedding = embedder.embed_query(query, normalize=True)
        
        faiss_results = faiss_store.search(query_embedding, top_k=3)
        neo4j_results = neo4j_store.search(query_embedding, top_k=3)
        
        logger.info(f"\nQuery: {query}")
        logger.info(f"  FAISS results: {len(faiss_results)}")
        logger.info(f"  Neo4j results: {len(neo4j_results)}")
        
        if len(faiss_results) > 0 and len(neo4j_results) == 0:
            logger.warning("  WARNING: Neo4j returned no results!")
            all_passed = False
    
    neo4j_store.close()
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Migrate FAISS index to Neo4j Knowledge Graph Store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic migration
    python scripts/migrate_faiss_to_neo4j.py --faiss-dir ./output/rag

    # With verification
    python scripts/migrate_faiss_to_neo4j.py --faiss-dir ./output/rag --verify

    # Clear and re-migrate
    python scripts/migrate_faiss_to_neo4j.py --faiss-dir ./output/rag --clear --re-embed
        """
    )
    
    parser.add_argument(
        "--faiss-dir",
        required=True,
        help="Directory containing the FAISS index files"
    )
    parser.add_argument(
        "--prefix",
        default="kg_triplets",
        help="FAISS file prefix (default: kg_triplets)"
    )
    parser.add_argument(
        "--neo4j-uri",
        default="bolt://localhost:7687",
        help="Neo4j Bolt URI (default: bolt://localhost:7687)"
    )
    parser.add_argument(
        "--neo4j-user",
        default="neo4j",
        help="Neo4j username (default: neo4j)"
    )
    parser.add_argument(
        "--neo4j-password",
        default=None,
        help="Neo4j password (default: from NEO4J_PASSWORD env or 'password123')"
    )
    parser.add_argument(
        "--neo4j-database",
        default="neo4j",
        help="Neo4j database name (default: neo4j)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for migration (default: 100)"
    )
    parser.add_argument(
        "--re-embed",
        action="store_true",
        help="Re-generate embeddings during migration"
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding model for re-embedding (default: from env or bge-small)"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing Neo4j triplet data before migration"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify migration after completion"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without actually migrating"
    )
    
    args = parser.parse_args()
    
    # Get password from args or environment
    neo4j_password = args.neo4j_password
    if neo4j_password is None:
        neo4j_password = os.environ.get("NEO4J_PASSWORD", "password123")
    
    if args.dry_run:
        logger.info("DRY RUN - No changes will be made")
        logger.info(f"Would migrate from: {args.faiss_dir}")
        logger.info(f"To Neo4j at: {args.neo4j_uri}")
        
        # Just load and show stats
        try:
            faiss_store = FaissStore.load(args.faiss_dir, prefix=args.prefix)
            logger.info(f"FAISS contains {faiss_store.size} vectors")
            logger.info(f"Embedding dimension: {faiss_store.embedding_dim}")
            
            # Show sample metadata
            if faiss_store.metadata:
                logger.info("\nSample triplets:")
                for meta in faiss_store.metadata[:5]:
                    s = meta.get("subject", "?")
                    p = meta.get("predicate", "?")
                    o = meta.get("object", "?")
                    logger.info(f"  ({s}, {p}, {o})")
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
        
        return
    
    # Run migration
    migrated = migrate_faiss_to_neo4j(
        faiss_dir=args.faiss_dir,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=neo4j_password,
        neo4j_database=args.neo4j_database,
        batch_size=args.batch_size,
        re_embed=args.re_embed,
        embedding_model=args.embedding_model,
        clear_existing=args.clear,
        prefix=args.prefix,
    )
    
    if migrated > 0 and args.verify:
        logger.info("\nRunning verification...")
        success = verify_migration(
            faiss_dir=args.faiss_dir,
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=neo4j_password,
            neo4j_database=args.neo4j_database,
            prefix=args.prefix,
        )
        
        if success:
            logger.info("\nVerification PASSED")
        else:
            logger.warning("\nVerification completed with warnings")


if __name__ == "__main__":
    main()
