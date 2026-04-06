#!/usr/bin/env python
"""
Split the PrimeKG drug-disease subset into train/test sets.

Produces a stratified, entity-disjoint split: for each relation type, drugs
(x_name) are grouped and split so that no drug appears in both train and test
for the same relation.  This prevents trivial lookup during evaluation.

Outputs (to --output-dir, default data/eval/):
    train.csv        -- training split, used as default input by
                        import_primekb_to_neo4j.py to populate Neo4j
    test.csv         -- held-out test split for evaluation
    split_stats.json -- per-relation split statistics

Usage:
    python scripts/eval/split_primekb.py --input data/kg_drug_disease.csv
    python scripts/eval/split_primekb.py --input data/kg_drug_disease.csv --test-ratio 0.3
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def entity_disjoint_split(
    df: pd.DataFrame,
    relation: str,
    test_ratio: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows for *one* relation so that no drug appears in both sets.

    Groups edges by x_name (drug), shuffles the groups, and assigns groups to
    test until the test-set edge count reaches ``test_ratio`` of the total.
    """
    rel_df = df[df["display_relation"] == relation].copy()
    drugs = rel_df["x_name"].unique()
    rng.shuffle(drugs)

    target_test = int(len(rel_df) * test_ratio)
    test_drugs: set[str] = set()
    running = 0
    for drug in drugs:
        if running >= target_test:
            break
        count = int((rel_df["x_name"] == drug).sum())
        test_drugs.add(drug)
        running += count

    test_mask = rel_df["x_name"].isin(test_drugs)
    return rel_df[~test_mask], rel_df[test_mask]


def split_dataset(
    input_path: str,
    output_dir: str,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    """Load, split, save, and return stats."""
    rng = np.random.default_rng(seed)

    df = pd.read_csv(input_path, low_memory=False)
    logger.info(f"Loaded {len(df):,} rows from {input_path}")

    relations = sorted(df["display_relation"].unique())
    logger.info(f"Relations: {relations}")

    train_parts, test_parts = [], []
    per_relation_stats: dict[str, dict] = {}

    for rel in relations:
        train_rel, test_rel = entity_disjoint_split(df, rel, test_ratio, rng)
        train_parts.append(train_rel)
        test_parts.append(test_rel)

        train_drugs = set(train_rel["x_name"].unique())
        test_drugs = set(test_rel["x_name"].unique())
        overlap = train_drugs & test_drugs

        per_relation_stats[rel] = {
            "total": len(train_rel) + len(test_rel),
            "train": len(train_rel),
            "test": len(test_rel),
            "train_drugs": len(train_drugs),
            "test_drugs": len(test_drugs),
            "drug_overlap": len(overlap),
            "test_ratio_actual": round(
                len(test_rel) / (len(train_rel) + len(test_rel)), 4
            ),
        }
        logger.info(
            f"  {rel}: {per_relation_stats[rel]['train']} train / "
            f"{per_relation_stats[rel]['test']} test  "
            f"(overlap drugs: {len(overlap)})"
        )

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    stats = {
        "seed": seed,
        "test_ratio_requested": test_ratio,
        "input_file": str(input_path),
        "total_rows": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "relations": per_relation_stats,
    }
    stats_path = os.path.join(output_dir, "split_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(
        f"Saved: {train_path} ({len(train_df):,}), "
        f"{test_path} ({len(test_df):,}), {stats_path}"
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split PrimeKG drug-disease subset into train/test",
    )
    parser.add_argument(
        "--input",
        default="data/kg_drug_disease.csv",
        help="Path to the drug-disease CSV (default: data/kg_drug_disease.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/eval",
        help="Directory for train.csv, test.csv, split_stats.json (default: data/eval)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of data for the test set (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not os.path.exists(args.input):
        logger.error(
            f"Input file not found: {args.input}\n"
            "Run:  python scripts/download_primekb.py          to create it.\n"
            "  Or: python scripts/download_primekb.py --extract drug-disease"
        )
        sys.exit(1)

    stats = split_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print("\n" + "=" * 50)
    print("Split summary")
    print("=" * 50)
    print(f"  Train rows : {stats['train_rows']:,}")
    print(f"  Test rows  : {stats['test_rows']:,}")
    for rel, rs in stats["relations"].items():
        print(f"  {rel:25s}  train={rs['train']:>6,}  test={rs['test']:>5,}  "
              f"ratio={rs['test_ratio_actual']:.2%}  drug_overlap={rs['drug_overlap']}")
    print(f"\nOutput dir: {args.output_dir}")


if __name__ == "__main__":
    main()
