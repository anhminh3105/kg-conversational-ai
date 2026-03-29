#!/usr/bin/env python3
"""
Convert native PrimeKG graph nodes into :Triplet nodes for demo compatibility.

The native PrimeKG importer (import_primekb_to_neo4j.py) creates typed nodes
(:Drug, :Disease, :GeneProtein, etc.) with typed relationships (-[:TREATS]->,
-[:ASSOCIATED_WITH]->).  The demo scripts (demo_mcp_agent.py, interactive_agent.py)
and the RAG pipeline expect flat :Triplet nodes with subject/predicate/object
properties.  This script bridges the two schemas by reading native PrimeKG
relationships and writing corresponding :Triplet nodes with embeddings.

Usage:
    # Basic conversion (all native PrimeKG relationships)
    python scripts/convert_primekb_to_triplets.py

    # Filter by node types
    python scripts/convert_primekb_to_triplets.py --node-types drug,disease

    # Filter by relationship types
    python scripts/convert_primekb_to_triplets.py --relation-types TREATS,ASSOCIATED_WITH

    # Limit rows and clear existing Triplet nodes first
    python scripts/convert_primekb_to_triplets.py --max-rows 50000 --clear-triplets

    # Custom Neo4j connection
    python scripts/convert_primekb_to_triplets.py --neo4j-password mypassword

Prerequisites:
    - Neo4j running with PrimeKG data imported via import_primekb_to_neo4j.py
    - pip install sentence-transformers neo4j numpy tqdm
"""

import argparse
import logging
import os
import sys
import time
from typing import List, Optional

import numpy as np
from neo4j import GraphDatabase
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from rag.embedder import get_embedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BATCH_SIZE = 500
EMBED_BATCH_SIZE = 64


def detect_primekb_nodes(driver, database: str = "neo4j") -> int:
    """Check how many native PrimeKG nodes exist (nodes with primekb_id)."""
    with driver.session(database=database) as session:
        result = session.run(
            "MATCH (n) WHERE n.primekb_id IS NOT NULL RETURN count(n) AS cnt"
        )
        return result.single()["cnt"]


def detect_triplet_nodes(driver, database: str = "neo4j") -> int:
    """Check how many :Triplet nodes already exist."""
    with driver.session(database=database) as session:
        result = session.run("MATCH (t:Triplet) RETURN count(t) AS cnt")
        return result.single()["cnt"]


def fetch_native_relationships(
    driver,
    database: str = "neo4j",
    max_rows: Optional[int] = None,
    node_types: Optional[List[str]] = None,
    relation_types: Optional[List[str]] = None,
) -> List[dict]:
    """
    Fetch relationships from native PrimeKG graph.

    Returns list of dicts with keys: subject, predicate, object,
    subject_type, object_type.
    """
    clauses = ["WHERE a.primekb_id IS NOT NULL"]
    params = {}

    if node_types:
        nt = [t.lower().strip() for t in node_types]
        clauses.append(
            "(toLower(a.node_type) IN $node_types "
            "OR toLower(b.node_type) IN $node_types)"
        )
        params["node_types"] = nt

    if relation_types:
        rt = [r.upper().replace(" ", "_") for r in relation_types]
        clauses.append("type(r) IN $relation_types")
        params["relation_types"] = rt

    where = " AND ".join(clauses)
    limit = f" LIMIT {max_rows}" if max_rows else ""

    query = f"""
        MATCH (a)-[r]->(b)
        {where}
        RETURN a.name AS subject,
               type(r) AS predicate,
               b.name AS object,
               a.node_type AS subject_type,
               b.node_type AS object_type
        {limit}
    """

    logger.info("Querying native PrimeKG relationships...")
    with driver.session(database=database) as session:
        result = session.run(query, params)
        rows = [dict(record) for record in result]

    logger.info(f"Fetched {len(rows)} relationships")
    return rows


def clear_triplet_nodes(driver, database: str = "neo4j") -> int:
    """Remove all existing :Triplet nodes. Returns count deleted."""
    with driver.session(database=database) as session:
        result = session.run(
            "MATCH (t:Triplet) WITH t LIMIT 50000 DETACH DELETE t RETURN count(*) AS cnt"
        )
        total = result.single()["cnt"]

        while total > 0:
            result = session.run(
                "MATCH (t:Triplet) WITH t LIMIT 50000 DETACH DELETE t "
                "RETURN count(*) AS cnt"
            )
            batch = result.single()["cnt"]
            if batch == 0:
                break
            total += batch

    logger.info(f"Cleared {total} Triplet nodes")
    return total


def ensure_triplet_indexes(driver, database: str = "neo4j", embedding_dim: int = 384):
    """Create indexes for Triplet nodes (matches Neo4jStore._ensure_indexes)."""
    with driver.session(database=database) as session:
        for stmt, desc in [
            (
                "CREATE CONSTRAINT triplet_unique IF NOT EXISTS "
                "FOR (t:Triplet) REQUIRE (t.subject, t.predicate, t.object) IS UNIQUE",
                "unique constraint",
            ),
            (
                "CREATE INDEX triplet_subject IF NOT EXISTS "
                "FOR (t:Triplet) ON (t.subject)",
                "subject index",
            ),
            (
                "CREATE INDEX triplet_object IF NOT EXISTS "
                "FOR (t:Triplet) ON (t.object)",
                "object index",
            ),
        ]:
            try:
                session.run(stmt)
                logger.info(f"  Created {desc}")
            except Exception as e:
                logger.debug(f"  {desc}: {e}")

        try:
            session.run(
                f"CREATE VECTOR INDEX triplet_embedding IF NOT EXISTS "
                f"FOR (t:Triplet) ON t.embedding "
                f"OPTIONS {{indexConfig: {{"
                f"`vector.dimensions`: {embedding_dim}, "
                f"`vector.similarity_function`: 'cosine'"
                f"}}}}"
            )
            logger.info(f"  Created vector index (dim={embedding_dim})")
        except Exception as e:
            logger.warning(f"  Vector index: {e}")


def write_triplets(
    driver,
    rows: List[dict],
    embeddings: np.ndarray,
    database: str = "neo4j",
) -> int:
    """Write :Triplet nodes with embeddings to Neo4j in batches."""
    written = 0
    with driver.session(database=database) as session:
        for i in tqdm(range(0, len(rows), BATCH_SIZE), desc="Writing triplets"):
            batch_rows = rows[i : i + BATCH_SIZE]
            batch_embs = embeddings[i : i + BATCH_SIZE]

            batch = []
            for row, emb in zip(batch_rows, batch_embs):
                batch.append({
                    "subject": row["subject"],
                    "predicate": row["predicate"],
                    "object": row["object"],
                    "document": f"({row['subject']}, {row['predicate']}, {row['object']})",
                    "source_text": (
                        f"subject_type={row.get('subject_type', '')}; "
                        f"object_type={row.get('object_type', '')}"
                    ),
                    "embedding": emb.tolist(),
                    "source": "primekb_converted",
                    "mode": "triplet_text",
                })

            session.run(
                """
                UNWIND $batch AS row
                MERGE (t:Triplet {
                    subject: row.subject,
                    predicate: row.predicate,
                    object: row.object
                })
                ON CREATE SET
                    t.document = row.document,
                    t.source_text = row.source_text,
                    t.embedding = row.embedding,
                    t.source = row.source,
                    t.representation_mode = row.mode
                ON MATCH SET
                    t.embedding = row.embedding,
                    t.source = row.source
                """,
                {"batch": batch},
            )
            written += len(batch)

    return written


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert native PrimeKG graph nodes into :Triplet nodes "
            "for demo script compatibility"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/convert_primekb_to_triplets.py
  python scripts/convert_primekb_to_triplets.py --node-types drug,disease
  python scripts/convert_primekb_to_triplets.py --max-rows 50000 --clear-triplets
        """,
    )
    parser.add_argument(
        "--neo4j-uri", default="bolt://localhost:7687", help="Neo4j Bolt URI"
    )
    parser.add_argument("--neo4j-user", default="neo4j", help="Neo4j username")
    parser.add_argument(
        "--neo4j-password", default=None, help="Neo4j password (default: env or password123)"
    )
    parser.add_argument(
        "--node-types", default=None,
        help="Comma-separated node types to include (e.g. drug,disease)",
    )
    parser.add_argument(
        "--relation-types", default=None,
        help="Comma-separated relationship types (e.g. TREATS,ASSOCIATED_WITH)",
    )
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Maximum relationships to convert",
    )
    parser.add_argument(
        "--clear-triplets", action="store_true",
        help="Remove existing :Triplet nodes before conversion",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("LOCAL_EMBEDDER_MODEL", "BAAI/bge-small-en-v1.5"),
        help="Sentence transformer model for embeddings",
    )
    args = parser.parse_args()

    password = args.neo4j_password or os.environ.get("NEO4J_PASSWORD", "password123")
    node_types = (
        [t.strip() for t in args.node_types.split(",")]
        if args.node_types else None
    )
    relation_types = (
        [r.strip() for r in args.relation_types.split(",")]
        if args.relation_types else None
    )

    print("=" * 60)
    print("PrimeKG Native Graph -> :Triplet Node Converter")
    print("=" * 60)

    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, password))

    try:
        # Verify native PrimeKG data exists
        n_primekb = detect_primekb_nodes(driver)
        n_triplets = detect_triplet_nodes(driver)
        print(f"\n  Native PrimeKG nodes: {n_primekb}")
        print(f"  Existing Triplet nodes: {n_triplets}")

        if n_primekb == 0:
            print("\n  ERROR: No native PrimeKG nodes found.")
            print("  Run import_primekb_to_neo4j.py first:")
            print("    python scripts/import_primekb_to_neo4j.py --input data/kg.csv")
            sys.exit(1)

        # Optionally clear existing Triplet nodes
        if args.clear_triplets and n_triplets > 0:
            print(f"\n  Clearing {n_triplets} existing Triplet nodes...")
            clear_triplet_nodes(driver)

        # Fetch native relationships
        rows = fetch_native_relationships(
            driver,
            max_rows=args.max_rows,
            node_types=node_types,
            relation_types=relation_types,
        )

        if not rows:
            print("\n  No relationships found after filtering.")
            sys.exit(0)

        # Deduplicate by (subject, predicate, object)
        seen = set()
        unique_rows = []
        for r in rows:
            key = (r["subject"], r["predicate"], r["object"])
            if key not in seen:
                seen.add(key)
                unique_rows.append(r)
        rows = unique_rows
        print(f"\n  Unique relationships to convert: {len(rows)}")

        # Generate embeddings
        print(f"\n  Loading embedder: {args.embedding_model}")
        embedder = get_embedder(model_name=args.embedding_model)

        texts = [
            f"{r['subject'].replace('_', ' ')} "
            f"{r['predicate'].replace('_', ' ')} "
            f"{r['object'].replace('_', ' ')}"
            for r in rows
        ]

        print(f"  Generating embeddings for {len(texts)} triplets...")
        t0 = time.time()
        embeddings = embedder.embed_texts(
            texts, batch_size=EMBED_BATCH_SIZE, show_progress=True, normalize=True
        )
        embed_time = time.time() - t0
        print(f"  Embeddings generated in {embed_time:.1f}s (shape: {embeddings.shape})")

        # Ensure indexes
        print("\n  Ensuring Triplet indexes...")
        ensure_triplet_indexes(driver, embedding_dim=embedder.embedding_dim)

        # Write Triplet nodes
        print(f"\n  Writing {len(rows)} Triplet nodes to Neo4j...")
        t0 = time.time()
        written = write_triplets(driver, rows, embeddings)
        write_time = time.time() - t0
        print(f"  Wrote {written} Triplet nodes in {write_time:.1f}s")

        # Final stats
        n_triplets_after = detect_triplet_nodes(driver)
        print(f"\n  Total Triplet nodes now: {n_triplets_after}")

        print("\n" + "=" * 60)
        print("Conversion complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("  # Run the MCP agent demo")
        print("  python scripts/demo_mcp_agent.py --simple")
        print("")
        print("  # Start an interactive session")
        print("  python scripts/interactive_agent.py --lite")
        print("")
        print("  # Visualize the knowledge graph")
        print("  python scripts/visualize_kg.py --output outputs/primekb_triplets.png")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
