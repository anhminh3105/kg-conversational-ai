#!/usr/bin/env python
"""
Split the PrimeKG drug-disease subset into train/test with evaluation tiers.

For each relation type, drugs (x_name) are grouped and split so that no drug
appears in both train and test for the same relation.  Test entities are then
assigned to one of three evaluation tiers:

    Full    (default 10%) -- all gold triples included in train.csv
    Partial (default 80%) -- ~50% of gold triples included in train.csv
    Zero    (default 10%) -- no gold triples included in train.csv

Outputs (to --output-dir, default data/eval/):
    train.csv              -- KG to load into Neo4j (background + tier-appropriate gold)
    test.csv               -- all test entity triples (for QA generation)
    tier_assignments.json  -- {entity: {relation: tier}} mapping
    split_stats.json       -- per-relation and per-tier statistics

Usage:
    python scripts/eval/split_primekb.py --input data/kg_drug_disease.csv
    python scripts/eval/split_primekb.py --input data/kg_drug_disease.csv --max-rows 500
    python scripts/eval/split_primekb.py --input data/kg_drug_disease.csv --test-ratio 0.3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def entity_disjoint_split(
    df: pd.DataFrame,
    relation: str,
    test_ratio: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows for *one* relation so that no drug appears in both sets."""
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


def assign_tiers(
    test_drugs: list[str],
    rng: np.random.Generator,
    full_ratio: float = 0.10,
    zero_ratio: float = 0.10,
) -> dict[str, str]:
    """Assign each test drug to a tier: full, partial, or zero."""
    drugs = list(test_drugs)
    rng.shuffle(drugs)
    n = len(drugs)
    n_full = max(1, int(n * full_ratio))
    n_zero = max(1, int(n * zero_ratio))

    assignments: dict[str, str] = {}
    for i, drug in enumerate(drugs):
        if i < n_full:
            assignments[drug] = "full"
        elif i < n_full + n_zero:
            assignments[drug] = "zero"
        else:
            assignments[drug] = "partial"
    return assignments


def split_dataset(
    input_path: str,
    output_dir: str,
    test_ratio: float = 0.2,
    seed: int = 42,
    max_rows: Optional[int] = None,
    full_ratio: float = 0.10,
    zero_ratio: float = 0.10,
    partial_include: float = 0.50,
) -> dict:
    """Load, split, assign tiers, save, and return stats."""
    rng = np.random.default_rng(seed)

    df = pd.read_csv(input_path, low_memory=False)
    logger.info(f"Loaded {len(df):,} rows from {input_path}")

    if max_rows and max_rows < len(df):
        total = len(df)
        parts = []
        for rel, grp in df.groupby("display_relation"):
            n = max(1, int(max_rows * len(grp) / total))
            parts.append(grp.sample(n=min(n, len(grp)), random_state=seed))
        df = pd.concat(parts, ignore_index=True)
        logger.info(f"Sampled to {len(df):,} rows (--max-rows {max_rows})")

    relations = sorted(df["display_relation"].unique())
    logger.info(f"Relations: {relations}")

    train_parts, test_parts = [], []
    tier_map: dict[str, dict] = {}
    per_relation_stats: dict[str, dict] = {}

    for rel in relations:
        train_rel, test_rel = entity_disjoint_split(df, rel, test_ratio, rng)

        test_drugs = list(test_rel["x_name"].unique())
        drug_tiers = assign_tiers(test_drugs, rng, full_ratio, zero_ratio)

        full_drugs = {d for d, t in drug_tiers.items() if t == "full"}
        partial_drugs = {d for d, t in drug_tiers.items() if t == "partial"}

        tier_train_rows = []

        # Full tier: include ALL test triples; record all y_names as kg_answers
        if full_drugs:
            full_df = test_rel[test_rel["x_name"].isin(full_drugs)]
            tier_train_rows.append(full_df)
            for drug in full_drugs:
                answers = sorted(full_df.loc[full_df["x_name"] == drug, "y_name"].unique().tolist())
                tier_map.setdefault(drug, {})[rel] = {"tier": "full", "kg_answers": answers}

        # Partial tier: include ~partial_include fraction; record sampled y_names
        if partial_drugs:
            partial_df = test_rel[test_rel["x_name"].isin(partial_drugs)]
            partial_parts = []
            for drug, grp in partial_df.groupby("x_name"):
                n = max(1, int(len(grp) * partial_include))
                sampled = grp.sample(n=min(n, len(grp)), random_state=seed)
                partial_parts.append(sampled)
                answers = sorted(sampled["y_name"].unique().tolist())
                tier_map.setdefault(drug, {})[rel] = {"tier": "partial", "kg_answers": answers}
            if partial_parts:
                tier_train_rows.append(pd.concat(partial_parts, ignore_index=True))

        # Zero tier: no triples included; kg_answers is empty
        for drug, tier in drug_tiers.items():
            if tier == "zero":
                tier_map.setdefault(drug, {})[rel] = {"tier": "zero", "kg_answers": []}

        train_parts.append(train_rel)
        if tier_train_rows:
            train_parts.extend(tier_train_rows)
        test_parts.append(test_rel)

        # Stats
        tier_counts = {"full": 0, "partial": 0, "zero": 0}
        for t in drug_tiers.values():
            tier_counts[t] += 1

        tier_triple_counts = {
            "full": int(test_rel[test_rel["x_name"].isin(full_drugs)].shape[0]),
            "partial": int(test_rel[test_rel["x_name"].isin(partial_drugs)].shape[0]),
            "zero": int(test_rel[
                test_rel["x_name"].isin(
                    {d for d, t in drug_tiers.items() if t == "zero"}
                )
            ].shape[0]),
        }

        per_relation_stats[rel] = {
            "total": len(train_rel) + len(test_rel),
            "train_background": len(train_rel),
            "test": len(test_rel),
            "train_drugs": len(train_rel["x_name"].unique()),
            "test_drugs": len(test_drugs),
            "tiers": {
                "full": {"drugs": tier_counts["full"], "triples": tier_triple_counts["full"]},
                "partial": {"drugs": tier_counts["partial"], "triples": tier_triple_counts["partial"]},
                "zero": {"drugs": tier_counts["zero"], "triples": tier_triple_counts["zero"]},
            },
        }
        logger.info(
            f"  {rel}: {len(train_rel)} bg-train / {len(test_rel)} test  "
            f"tiers: full={tier_counts['full']} partial={tier_counts['partial']} "
            f"zero={tier_counts['zero']}"
        )

    train_df = pd.concat(train_parts, ignore_index=True).drop_duplicates(
        subset=["x_name", "display_relation", "y_name"]
    )
    test_df = pd.concat(test_parts, ignore_index=True)

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    tier_path = os.path.join(output_dir, "tier_assignments.json")
    with open(tier_path, "w") as f:
        json.dump(tier_map, f, indent=2)

    stats = {
        "seed": seed,
        "test_ratio_requested": test_ratio,
        "max_rows": max_rows,
        "full_ratio": full_ratio,
        "zero_ratio": zero_ratio,
        "partial_include": partial_include,
        "input_file": str(input_path),
        "total_rows_after_sampling": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "relations": per_relation_stats,
    }
    stats_path = os.path.join(output_dir, "split_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(
        f"Saved: {train_path} ({len(train_df):,}), "
        f"{test_path} ({len(test_df):,}), {tier_path}, {stats_path}"
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split PrimeKG drug-disease subset into train/test with evaluation tiers",
    )
    parser.add_argument(
        "--input",
        default="data/kg_drug_disease.csv",
        help="Path to the drug-disease CSV (default: data/kg_drug_disease.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/eval",
        help="Directory for outputs (default: data/eval)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of data for the test set (default: 0.2)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=1000,
        help="Cap total input rows for fast processing (default: 1000, 0=no cap)",
    )
    parser.add_argument(
        "--full-ratio",
        type=float,
        default=0.10,
        help="Fraction of test entities in the Full tier (default: 0.10)",
    )
    parser.add_argument(
        "--zero-ratio",
        type=float,
        default=0.10,
        help="Fraction of test entities in the Zero tier (default: 0.10)",
    )
    parser.add_argument(
        "--partial-include",
        type=float,
        default=0.50,
        help="Fraction of Partial-tier entity triples to include in train (default: 0.50)",
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
        max_rows=args.max_rows or None,
        full_ratio=args.full_ratio,
        zero_ratio=args.zero_ratio,
        partial_include=args.partial_include,
    )

    print("\n" + "=" * 60)
    print("Split summary")
    print("=" * 60)
    print(f"  Input rows (after sampling): {stats['total_rows_after_sampling']:,}")
    print(f"  Train rows (KG):             {stats['train_rows']:,}")
    print(f"  Test rows  (QA):             {stats['test_rows']:,}")
    for rel, rs in stats["relations"].items():
        tiers = rs["tiers"]
        print(
            f"  {rel:25s}  bg={rs['train_background']:>5,}  test={rs['test']:>5,}  "
            f"F={tiers['full']['drugs']} P={tiers['partial']['drugs']} "
            f"Z={tiers['zero']['drugs']}"
        )
    print(f"\nOutput dir: {args.output_dir}")


if __name__ == "__main__":
    main()
