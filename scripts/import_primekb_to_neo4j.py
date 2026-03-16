#!/usr/bin/env python
"""
Import PrimeKG dataset into Neo4j as a native graph.

Creates nodes with labels derived from x_type/y_type (e.g. :Drug, :Disease,
:Gene) and relationships from display_relation (e.g. -[:TREATS]->).

Uses batched UNWIND queries for performance.

Usage:
    python scripts/import_primekb_to_neo4j.py --input data/kg.csv
    python scripts/import_primekb_to_neo4j.py --input data/kg.csv --max_rows 50000
    python scripts/import_primekb_to_neo4j.py --input data/kg.csv --node_types drug,disease
"""

import argparse
import logging
import os
import re
import sys
import time
from typing import List, Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.import_kg_data_from_json import Neo4jConnection

logger = logging.getLogger(__name__)

BATCH_SIZE = 2000


def _neo4j_label(raw_type: str) -> str:
    """Convert a PrimeKG type string to a valid Neo4j label.

    Examples:
        'gene/protein' -> 'GeneProtein'
        'biological_process' -> 'BiologicalProcess'
        'effect/phenotype' -> 'EffectPhenotype'
    """
    cleaned = re.sub(r"[^a-zA-Z0-9 ]", " ", raw_type)
    return "".join(w.capitalize() for w in cleaned.split())


def _neo4j_rel_type(display_relation: str) -> str:
    """Convert a PrimeKG display_relation to a Neo4j relationship type.

    Examples:
        'treats' -> 'TREATS'
        'associated with' -> 'ASSOCIATED_WITH'
    """
    cleaned = re.sub(r"[^a-zA-Z0-9 ]", " ", display_relation)
    return "_".join(cleaned.upper().split())


def load_data(
    path: str,
    max_rows: Optional[int] = None,
    node_types: Optional[List[str]] = None,
    relation_types: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load and optionally filter the PrimeKG CSV."""
    logger.info(f"Reading {path} ...")
    df = pd.read_csv(path, nrows=max_rows)
    logger.info(f"Loaded {len(df)} rows")

    if node_types:
        nt = {t.lower().strip() for t in node_types}
        df = df[df["x_type"].str.lower().isin(nt) | df["y_type"].str.lower().isin(nt)]
        logger.info(f"Filtered to {len(df)} rows by node types {nt}")

    if relation_types:
        rt = {r.lower().strip() for r in relation_types}
        df = df[df["display_relation"].str.lower().isin(rt)]
        logger.info(f"Filtered to {len(df)} rows by relation types {rt}")

    return df


def create_indexes(conn: Neo4jConnection, labels: set[str]) -> None:
    """Create uniqueness constraints / indexes for each node label."""
    for label in sorted(labels):
        neo_label = _neo4j_label(label)
        try:
            conn.query(
                f"CREATE CONSTRAINT IF NOT EXISTS "
                f"FOR (n:{neo_label}) REQUIRE n.primekb_id IS UNIQUE"
            )
            logger.info(f"  Index on :{neo_label}(primekb_id)")
        except Exception as e:
            logger.debug(f"Index creation note for {neo_label}: {e}")


def import_nodes(conn: Neo4jConnection, df: pd.DataFrame) -> int:
    """Create nodes from both x and y sides of the dataframe. Returns count."""
    x_nodes = (
        df[["x_index", "x_id", "x_type", "x_name", "x_source"]]
        .drop_duplicates(subset=["x_index"])
        .rename(columns={"x_index": "idx", "x_id": "ext_id", "x_type": "ntype", "x_name": "name", "x_source": "source"})
    )
    y_nodes = (
        df[["y_index", "y_id", "y_type", "y_name", "y_source"]]
        .drop_duplicates(subset=["y_index"])
        .rename(columns={"y_index": "idx", "y_id": "ext_id", "y_type": "ntype", "y_name": "name", "y_source": "source"})
    )
    all_nodes = pd.concat([x_nodes, y_nodes]).drop_duplicates(subset=["idx"])
    logger.info(f"Importing {len(all_nodes)} unique nodes ...")

    count = 0
    for ntype, group in all_nodes.groupby("ntype"):
        label = _neo4j_label(ntype)
        records = group.to_dict("records")

        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i : i + BATCH_SIZE]
            conn.query(
                f"UNWIND $batch AS row "
                f"MERGE (n:{label} {{primekb_id: row.idx}}) "
                f"ON CREATE SET n.name = row.name, n.ext_id = row.ext_id, "
                f"n.source = row.source, n.node_type = row.ntype",
                {"batch": batch},
            )
            count += len(batch)

        logger.info(f"  :{label} -> {len(records)} nodes")

    return count


def import_relationships(conn: Neo4jConnection, df: pd.DataFrame) -> int:
    """Create relationships using UNWIND batches. Returns count."""
    logger.info(f"Importing {len(df)} relationships ...")
    count = 0

    for rel_name, group in df.groupby("display_relation"):
        rel_type = _neo4j_rel_type(rel_name)
        x_label = _neo4j_label(group["x_type"].iloc[0])
        y_label = _neo4j_label(group["y_type"].iloc[0])

        records = group[["x_index", "y_index"]].rename(
            columns={"x_index": "x_idx", "y_index": "y_idx"}
        ).to_dict("records")

        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i : i + BATCH_SIZE]
            conn.query(
                f"UNWIND $batch AS row "
                f"MATCH (a {{primekb_id: row.x_idx}}) "
                f"MATCH (b {{primekb_id: row.y_idx}}) "
                f"MERGE (a)-[r:{rel_type}]->(b)",
                {"batch": batch},
            )
            count += len(batch)

        logger.info(f"  -[:{rel_type}]-> {len(records)} rels ({x_label} -> {y_label})")

    return count


def show_stats(conn: Neo4jConnection) -> None:
    """Print basic graph statistics."""
    res = conn.query("MATCH (n) RETURN count(n) AS cnt")
    print(f"\n  Total nodes: {res[0]['cnt']}")

    res = conn.query("MATCH ()-[r]->() RETURN count(r) AS cnt")
    print(f"  Total relationships: {res[0]['cnt']}")

    res = conn.query("MATCH (n) RETURN DISTINCT labels(n) AS lbl, count(n) AS cnt ORDER BY cnt DESC LIMIT 15")
    print("\n  Top node labels:")
    for r in res:
        print(f"    {r['lbl']}: {r['cnt']}")

    res = conn.query("MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS cnt ORDER BY cnt DESC LIMIT 15")
    print("\n  Top relationship types:")
    for r in res:
        print(f"    {r['t']}: {r['cnt']}")


def main():
    parser = argparse.ArgumentParser(
        description="Import PrimeKG into Neo4j as a native graph",
    )
    parser.add_argument("--input", required=True, help="Path to PrimeKG kg.csv")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j Bolt URI")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD", "password123"))
    parser.add_argument("--max_rows", type=int, default=None, help="Limit rows loaded")
    parser.add_argument("--node_types", default=None, help="Comma-separated node types")
    parser.add_argument("--relation_types", default=None, help="Comma-separated relation types")
    parser.add_argument("--clear", action="store_true", help="Clear database before import")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    node_types = [t.strip() for t in args.node_types.split(",")] if args.node_types else None
    relation_types = [r.strip() for r in args.relation_types.split(",")] if args.relation_types else None

    df = load_data(args.input, max_rows=args.max_rows, node_types=node_types, relation_types=relation_types)

    if len(df) == 0:
        print("No data to import after filtering.")
        sys.exit(0)

    print("=" * 60)
    print("PrimeKG -> Neo4j Importer")
    print("=" * 60)

    conn = Neo4jConnection(args.uri, args.user, args.password)
    try:
        if args.clear:
            logger.info("Clearing database ...")
            conn.query("MATCH (n) DETACH DELETE n")

        all_types = set(df["x_type"].unique()) | set(df["y_type"].unique())
        create_indexes(conn, all_types)

        t0 = time.time()
        n_nodes = import_nodes(conn, df)
        n_rels = import_relationships(conn, df)
        elapsed = time.time() - t0

        print(f"\nImported {n_nodes} nodes, {n_rels} relationships in {elapsed:.1f}s")
        show_stats(conn)

        print("\n" + "=" * 60)
        print("Import complete!")
        print("=" * 60)
        print("\nNext steps with native PrimeKG graph:")
        print("  # Convert to :Triplet nodes for demo scripts:")
        print("  python scripts/convert_primekb_to_triplets.py")
        print("")
        print("  # Then run demos:")
        print("  python scripts/demo_mcp_agent.py --simple")
        print("  python scripts/interactive_agent.py --lite")
        print("")
        print("  # Or visualize the native graph directly:")
        print("  python scripts/visualize_kg.py --schema primekb")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
