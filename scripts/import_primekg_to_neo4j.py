#!/usr/bin/env python
"""
Import PrimeKG CSV directly into Neo4j as flat :Triplet nodes with embeddings.

Reads a PrimeKG-format CSV (columns: x_name, display_relation, y_name, x_type,
y_type, ...) and writes :Triplet nodes that the RAG pipeline and demo scripts
(demo_mcp_agent.py, interactive_agent.py) expect.

Each :Triplet node has: subject, predicate, object, document, source_text,
embedding (vector), source, and representation_mode properties.

By default, the script reads data/eval/train.csv (the train split of the
drug-disease subset).  Pass --input to override.

Usage:
    # Default: import the drug-disease subset
    python scripts/import_primekb_to_neo4j.py

    # Import the full PrimeKG dataset
    python scripts/import_primekb_to_neo4j.py --input data/kg.csv

    # Limit rows for a quick test
    python scripts/import_primekb_to_neo4j.py --max-rows 500

    # Filter by relation types
    python scripts/import_primekb_to_neo4j.py --relation-types contraindication,indication

    # Clear existing Triplet nodes before import
    python scripts/import_primekb_to_neo4j.py --clear

    # Custom Neo4j credentials
    python scripts/import_primekb_to_neo4j.py --uri bolt://myhost:7687 --password secret

Prerequisites:
    - Neo4j running (bolt://localhost:7687 by default)
    - pip install sentence-transformers neo4j numpy pandas tqdm
"""

import argparse
import logging
import os
import re
import sys
import time
from typing import List, Optional

import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Load rag.embedder directly to avoid rag/__init__.py pulling in LLM deps.
# Only rag/__init__ drags in LLM utilities; the embedder chain is standalone.
import importlib.util as _ilu
import types as _types

_rag_dir = os.path.join(project_root, "rag")

_rag_pkg = _types.ModuleType("rag")
_rag_pkg.__path__ = [_rag_dir]
_rag_pkg.__package__ = "rag"
sys.modules["rag"] = _rag_pkg

def _load_submodule(name: str, filepath: str):
    spec = _ilu.spec_from_file_location(name, filepath, submodule_search_locations=[])
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_load_submodule("rag.triplet_loader", os.path.join(_rag_dir, "triplet_loader.py"))
_load_submodule("rag.representation", os.path.join(_rag_dir, "representation.py"))
_embedder_mod = _load_submodule("rag.embedder", os.path.join(_rag_dir, "embedder.py"))
get_embedder = _embedder_mod.get_embedder

logger = logging.getLogger(__name__)

WRITE_BATCH_SIZE = 500
EMBED_BATCH_SIZE = 64

DEFAULT_INPUT = os.path.join(project_root, "data", "eval", "train.csv")


def _neo4j_rel_type(display_relation: str) -> str:
    """Convert a PrimeKG display_relation to an upper-case Neo4j-style name.

    'treats' -> 'TREATS',  'associated with' -> 'ASSOCIATED_WITH'
    """
    cleaned = re.sub(r"[^a-zA-Z0-9 ]", " ", display_relation)
    return "_".join(cleaned.upper().split())


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_data(
    path: str,
    max_rows: Optional[int] = None,
    node_types: Optional[List[str]] = None,
    relation_types: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load a PrimeKG CSV and optionally filter by node / relation types."""
    logger.info("Reading %s ...", path)
    df = pd.read_csv(path, nrows=max_rows)
    logger.info("Loaded %d rows", len(df))

    if node_types:
        nt = {t.lower().strip() for t in node_types}
        df = df[df["x_type"].str.lower().isin(nt) | df["y_type"].str.lower().isin(nt)]
        logger.info("Filtered to %d rows by node types %s", len(df), nt)

    if relation_types:
        rt = {r.lower().strip() for r in relation_types}
        df = df[df["display_relation"].str.lower().isin(rt)]
        logger.info("Filtered to %d rows by relation types %s", len(df), rt)

    return df


def deduplicate_triplets(df: pd.DataFrame) -> List[dict]:
    """Extract unique (subject, predicate, object) rows from the DataFrame."""
    seen: set = set()
    rows: List[dict] = []
    for _, r in df.iterrows():
        subj = str(r["x_name"])
        pred = _neo4j_rel_type(str(r["display_relation"]))
        obj = str(r["y_name"])
        key = (subj, pred, obj)
        if key not in seen:
            seen.add(key)
            rows.append({
                "subject": subj,
                "predicate": pred,
                "object": obj,
                "subject_type": str(r.get("x_type", "")),
                "object_type": str(r.get("y_type", "")),
            })
    return rows


# ---------------------------------------------------------------------------
# Neo4j helpers
# ---------------------------------------------------------------------------

def count_triplet_nodes(driver, database: str = "neo4j") -> int:
    with driver.session(database=database) as session:
        result = session.run("MATCH (t:Triplet) RETURN count(t) AS cnt")
        return result.single()["cnt"]


def clear_all_nodes(driver, database: str = "neo4j") -> int:
    """Delete ALL nodes and relationships in the database in batches."""
    total = 0
    while True:
        with driver.session(database=database) as session:
            result = session.run(
                "MATCH (n) WITH n LIMIT 50000 "
                "DETACH DELETE n RETURN count(*) AS cnt"
            )
            batch = result.single()["cnt"]
        if batch == 0:
            break
        total += batch
    logger.info("Cleared %d nodes from database", total)
    return total


def ensure_triplet_indexes(driver, database: str = "neo4j", embedding_dim: int = 384):
    """Create constraints and indexes for :Triplet nodes."""
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
                logger.info("  Created %s", desc)
            except Exception as e:
                logger.debug("  %s: %s", desc, e)

        try:
            session.run(
                "CREATE VECTOR INDEX triplet_embedding IF NOT EXISTS "
                "FOR (t:Triplet) ON t.embedding "
                "OPTIONS {indexConfig: {"
                f"`vector.dimensions`: {embedding_dim}, "
                "`vector.similarity_function`: 'cosine'"
                "}}"
            )
            logger.info("  Created vector index (dim=%d)", embedding_dim)
        except Exception as e:
            logger.warning("  Vector index: %s", e)


def write_triplets(
    driver,
    rows: List[dict],
    embeddings: np.ndarray,
    database: str = "neo4j",
) -> int:
    """Write :Triplet nodes with embeddings to Neo4j in batches."""
    written = 0
    with driver.session(database=database) as session:
        for i in tqdm(range(0, len(rows), WRITE_BATCH_SIZE), desc="Writing triplets"):
            batch_rows = rows[i : i + WRITE_BATCH_SIZE]
            batch_embs = embeddings[i : i + WRITE_BATCH_SIZE]

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
                    "source": "primekb",
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Import PrimeKG CSV into Neo4j as :Triplet nodes with embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/import_primekb_to_neo4j.py
  python scripts/import_primekb_to_neo4j.py --input data/kg.csv --max-rows 50000
  python scripts/import_primekb_to_neo4j.py --relation-types contraindication,indication
  python scripts/import_primekb_to_neo4j.py --clear --max-rows 1000
        """,
    )
    parser.add_argument(
        "--input", default=DEFAULT_INPUT,
        help="Path to PrimeKG CSV (default: data/eval/train.csv)",
    )
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j Bolt URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument(
        "--password",
        default=os.environ.get("NEO4J_PASSWORD", "password123"),
        help="Neo4j password (default: env NEO4J_PASSWORD or password123)",
    )
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Limit number of CSV rows loaded",
    )
    parser.add_argument(
        "--node-types", default=None,
        help="Comma-separated node types to include (e.g. drug,disease)",
    )
    parser.add_argument(
        "--relation-types", default=None,
        help="Comma-separated relation types (e.g. contraindication,indication)",
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="Wipe ALL nodes in the Neo4j database before import",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("LOCAL_EMBEDDER_MODEL", "BAAI/bge-small-en-v1.5"),
        help="Sentence-transformer model for embeddings (default: BAAI/bge-small-en-v1.5)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    node_types = (
        [t.strip() for t in args.node_types.split(",")]
        if args.node_types else None
    )
    relation_types = (
        [r.strip() for r in args.relation_types.split(",")]
        if args.relation_types else None
    )

    # --- Load CSV -----------------------------------------------------------
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        print("Run  python scripts/download_primekb.py  first to download the dataset.")
        sys.exit(1)

    df = load_data(
        args.input,
        max_rows=args.max_rows,
        node_types=node_types,
        relation_types=relation_types,
    )
    if len(df) == 0:
        print("No data to import after filtering.")
        sys.exit(0)

    print("=" * 60)
    print("PrimeKG CSV -> Neo4j :Triplet Importer")
    print("=" * 60)
    print(f"  Input:  {args.input}  ({len(df)} rows)")

    # --- Deduplicate --------------------------------------------------------
    rows = deduplicate_triplets(df)
    print(f"  Unique triplets: {len(rows)}")

    relations = sorted({r["predicate"] for r in rows})
    print(f"  Relations: {relations}")

    # --- Connect to Neo4j ---------------------------------------------------
    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))

    try:
        existing = count_triplet_nodes(driver)
        print(f"  Existing Triplet nodes in Neo4j: {existing}")

        if args.clear:
            print("\n  Clearing ALL nodes from the database...")
            cleared = clear_all_nodes(driver)
            print(f"  Deleted {cleared} nodes.")

        # --- Generate embeddings --------------------------------------------
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
            texts, batch_size=EMBED_BATCH_SIZE, show_progress=True, normalize=True,
        )
        embed_time = time.time() - t0
        print(f"  Embeddings done in {embed_time:.1f}s  (shape: {embeddings.shape})")

        # --- Indexes --------------------------------------------------------
        print("\n  Ensuring Triplet indexes...")
        ensure_triplet_indexes(driver, embedding_dim=embedder.embedding_dim)

        # --- Write ----------------------------------------------------------
        print(f"\n  Writing {len(rows)} Triplet nodes to Neo4j...")
        t0 = time.time()
        written = write_triplets(driver, rows, embeddings)
        write_time = time.time() - t0
        print(f"  Wrote {written} Triplet nodes in {write_time:.1f}s")

        total = count_triplet_nodes(driver)
        print(f"\n  Total Triplet nodes now: {total}")

        print("\n" + "=" * 60)
        print("Import complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("  python scripts/demo_mcp_agent.py --simple")
        print("  python scripts/interactive_agent.py --lite")
        print("  python scripts/visualize_kg.py --output outputs/primekb_triplets.png")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
