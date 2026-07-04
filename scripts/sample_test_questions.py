#!/usr/bin/env python
"""
Sample a handful of drug-related test questions from PrimeKG for use with
the interactive agent (dual-LLM validation, optionally with --no-persist).

The sampler reads from one of four related sources:

  - ``data/kg_drug_disease.csv``           (--source kg)
  - ``data/eval/train.csv``                (--source train)
  - ``data/eval/test.csv``                 (--source test)
  - ``data/eval/tier_assignments.json``    (--source partial, default)

The ``train.csv``, ``test.csv``, and ``tier_assignments.json`` artifacts are
produced by ``scripts/eval/split_primekg.py``. The default (``partial``)
pulls from drugs that the split labelled as Partial tier -- i.e. drugs with
some gold answers in Neo4j and some held out -- which reliably triggers the
dual-LLM validation path when paired with
``scripts/interactive_agent.py --validate --no-persist``.

Questions are stratified across the three drug-disease relations present
in the dataset:

  - indication
  - contraindication
  - off-label use

For the CSV sources (kg/train/test), questions are split ~50/50 between
forward (drug -> disease) and reverse (disease -> drug) phrasings. For the
partial source, only forward (drug -> disease) questions are emitted since
the tier assignments are keyed per drug.

Templates match the deterministic ``FALLBACK_TEMPLATES`` used by
``scripts/eval/generate_qa.py`` so the sampler is reproducible and does not
require any LLM calls.

Outputs (written to ``data/eval/`` by default):

  - interactive_test_questions.txt   one question per line, grouped by
                                     source/relation for readability
  - interactive_test_questions.json  structured records with drug, relation,
                                     direction, and gold triples for reference

Use ``--format {txt,json,both}`` to pick which file(s) to emit (default:
``both``), or ``--txt-path`` / ``--json-path`` to override individual paths.

Examples:
    python scripts/sample_test_questions.py                       # default: partial
    python scripts/sample_test_questions.py --source kg --n 12
    python scripts/sample_test_questions.py --source train --n 9
    python scripts/sample_test_questions.py --source test --format txt
    python scripts/sample_test_questions.py --txt-path data/eval/qs.txt
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


RELATIONS = ["indication", "contraindication", "off-label use"]

# Mirrors scripts/eval/generate_qa.py FALLBACK_TEMPLATES so the two stay in
# sync; keep phrasing identical for reproducibility across eval and demos.
FALLBACK_TEMPLATES: dict[str, dict[str, str]] = {
    "indication": {
        "forward": "What conditions is {entity} indicated for?",
        "reverse": "Which drugs are indicated for {entity}?",
    },
    "contraindication": {
        "forward": "What are the contraindications for {entity}?",
        "reverse": "Which drugs are contraindicated for {entity}?",
    },
    "off-label use": {
        "forward": "What are the off-label uses of {entity}?",
        "reverse": "Which drugs are used off-label for {entity}?",
    },
}

# Source identifier used in the output JSON/txt to show which file the row
# came from (also controls how missing-prerequisite errors are worded).
SOURCE_LABELS: dict[str, str] = {
    "kg": "kg_edge",
    "train": "train_edge",
    "test": "test_edge",
    "partial": "partial_edge",
}


def _split_counts(n: int, buckets: int) -> list[int]:
    """Distribute ``n`` items across ``buckets`` as evenly as possible."""
    if buckets <= 0 or n <= 0:
        return [0] * max(buckets, 0)
    base = [n // buckets] * buckets
    for i in range(n % buckets):
        base[i] += 1
    return base


def _load_csv(csv_path: Path, label: str) -> pd.DataFrame:
    """Load a drug-disease CSV and filter to the three supported relations.

    Shared by ``--source kg/train/test``; the schema is identical across
    ``data/kg_drug_disease.csv`` and the ``train.csv``/``test.csv`` output
    of ``scripts/eval/split_primekg.py``.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    required = {
        "display_relation", "x_name", "y_name",
        "x_type", "y_type", "x_index",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"{label} file {csv_path} is missing expected columns: "
            f"{sorted(missing)}"
        )
    mask = (
        (df["x_type"] == "drug")
        & (df["y_type"] == "disease")
        & (df["display_relation"].isin(RELATIONS))
    )
    return df.loc[mask].copy()


def _choose_entities(
    df: pd.DataFrame,
    relation: str,
    column: str,
    n: int,
    rng: random.Random,
    min_edges: int = 2,
) -> list[str]:
    """Pick `n` distinct entity names with >= min_edges for `relation`.

    Falls back to any entities with at least one edge if not enough meet the
    min_edges threshold. Returns [] when `n <= 0` or no matching rows exist.
    """
    if n <= 0:
        return []
    sub = df[df["display_relation"] == relation]
    counts = sub[column].value_counts()
    eligible = counts[counts >= min_edges].index.tolist()
    if len(eligible) >= n:
        return rng.sample(eligible, n)
    all_entities = counts.index.tolist()
    if len(all_entities) <= n:
        return all_entities
    return rng.sample(all_entities, n)


def _gold_triples(
    df: pd.DataFrame, entity: str, relation: str, column: str,
) -> list[dict[str, str]]:
    rows = df[(df[column] == entity) & (df["display_relation"] == relation)]
    return [
        {
            "subject": row["x_name"],
            "predicate": row["display_relation"],
            "object": row["y_name"],
        }
        for _, row in rows.iterrows()
    ]


def sample_kg_questions(
    df: pd.DataFrame, n: int, seed: int, source_label: str,
) -> list[dict[str, Any]]:
    """Stratified sample for CSV-backed sources (kg/train/test).

    Balances across relations with a ~50/50 forward/reverse split per
    relation. ``source_label`` tags the emitted records so the .txt output
    groups them by origin.
    """
    rng = random.Random(seed)
    per_relation = _split_counts(n, len(RELATIONS))

    records: list[dict[str, Any]] = []
    qid = 0
    for relation, k in zip(RELATIONS, per_relation):
        if k <= 0:
            continue
        k_forward = k // 2 + (k % 2)
        k_reverse = k - k_forward

        for drug in _choose_entities(df, relation, "x_name", k_forward, rng):
            qid += 1
            records.append({
                "id": f"q{qid:03d}",
                "question": FALLBACK_TEMPLATES[relation]["forward"].format(
                    entity=drug
                ),
                "entity": drug,
                "entity_type": "drug",
                "relation": relation,
                "direction": "forward",
                "source": source_label,
                "gold_triples": _gold_triples(df, drug, relation, "x_name"),
            })

        for disease in _choose_entities(df, relation, "y_name", k_reverse, rng):
            qid += 1
            records.append({
                "id": f"q{qid:03d}",
                "question": FALLBACK_TEMPLATES[relation]["reverse"].format(
                    entity=disease
                ),
                "entity": disease,
                "entity_type": "disease",
                "relation": relation,
                "direction": "reverse",
                "source": source_label,
                "gold_triples": _gold_triples(df, disease, relation, "y_name"),
            })
    return records


def _collect_partial_pool(
    tier_assignments: dict[str, Any],
) -> list[dict[str, Any]]:
    """Flatten tier_assignments.json to a list of Partial-tier entries."""
    pool: list[dict[str, Any]] = []
    for drug, relations in tier_assignments.items():
        if not isinstance(relations, dict):
            continue
        for relation, info in relations.items():
            if relation not in FALLBACK_TEMPLATES or not isinstance(info, dict):
                continue
            if info.get("tier") != "partial":
                continue
            pool.append({
                "drug": drug,
                "relation": relation,
                "kg_answers": list(info.get("kg_answers", [])),
                "held_out_answers": list(info.get("held_out_answers", [])),
            })
    return pool


def _allocate_partial_budget(
    effective_n: int, bucket_sizes: dict[str, int],
) -> dict[str, int]:
    """Spread ``effective_n`` picks across relations without overflow.

    Starts with an even split across relations, clips each bucket to its
    available size, then redistributes any leftover budget to relations
    that still have headroom. Keeps stratification balanced when possible
    but honours ``--n`` even when some relations run dry.
    """
    relations = list(bucket_sizes.keys())
    target = dict(zip(relations, _split_counts(effective_n, len(relations))))
    alloc = {r: min(target[r], bucket_sizes[r]) for r in relations}

    leftover = effective_n - sum(alloc.values())
    while leftover > 0:
        progress = False
        for r in relations:
            if alloc[r] < bucket_sizes[r]:
                alloc[r] += 1
                leftover -= 1
                progress = True
                if leftover == 0:
                    break
        if not progress:
            break
    return alloc


def sample_partial_questions(
    tier_assignments: dict[str, Any], n: int, seed: int,
) -> list[dict[str, Any]]:
    """Stratified sample of Partial-tier (drug, relation) pairs.

    Emits forward-only drug-centric questions. Each record carries both the
    KG-retrievable gold (``kg_triples``) and the held-out gold
    (``held_out_triples``), with ``gold_triples`` being their union.
    """
    rng = random.Random(seed)
    pool = _collect_partial_pool(tier_assignments)

    if not pool:
        return []

    by_relation: dict[str, list[dict[str, Any]]] = {
        r: [] for r in RELATIONS
    }
    for entry in pool:
        by_relation.setdefault(entry["relation"], []).append(entry)

    # Cap n at the available pool and warn if the caller asked for more.
    effective_n = min(n, len(pool))
    if effective_n < n:
        print(
            f"[warning] --source partial has only {len(pool)} eligible "
            f"(drug, relation) pairs; capping --n from {n} to {effective_n}",
            flush=True,
        )

    bucket_sizes = {r: len(by_relation[r]) for r in RELATIONS}
    alloc = _allocate_partial_budget(effective_n, bucket_sizes)

    records: list[dict[str, Any]] = []
    qid = 0
    for relation in RELATIONS:
        k = alloc[relation]
        if k <= 0:
            continue
        bucket = by_relation.get(relation, [])
        if not bucket:
            continue
        picks = rng.sample(bucket, min(k, len(bucket)))
        for entry in picks:
            drug = entry["drug"]
            kg_triples = [
                {"subject": drug, "predicate": relation, "object": obj}
                for obj in entry["kg_answers"]
            ]
            held_out_triples = [
                {"subject": drug, "predicate": relation, "object": obj}
                for obj in entry["held_out_answers"]
            ]
            qid += 1
            records.append({
                "id": f"q{qid:03d}",
                "question": FALLBACK_TEMPLATES[relation]["forward"].format(
                    entity=drug
                ),
                "entity": drug,
                "entity_type": "drug",
                "relation": relation,
                "direction": "forward",
                "source": SOURCE_LABELS["partial"],
                "tier": "partial",
                "gold_triples": kg_triples + held_out_triples,
                "kg_triples": kg_triples,
                "held_out_triples": held_out_triples,
            })
    return records


def _write_txt(records: list[dict[str, Any]], path: Path) -> None:
    """Write questions grouped by source/relation, one per line, header-commented."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        grouped.setdefault(f"{rec['source']}/{rec['relation']}", []).append(rec)

    with path.open("w") as f:
        for key in sorted(grouped):
            f.write(f"# {key}\n")
            for rec in grouped[key]:
                f.write(rec["question"] + "\n")
            f.write("\n")


def _write_json(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w") as f:
        json.dump({"count": len(records), "questions": records}, f, indent=2)


def write_outputs(
    records: list[dict[str, Any]],
    out_dir: Path,
    fmt: str = "both",
    txt_path: Path | None = None,
    json_path: Path | None = None,
) -> tuple[Path | None, Path | None]:
    """Write questions in the requested format(s).

    Args:
        records: Sampled question records.
        out_dir: Default directory when explicit paths are not given.
        fmt: One of ``"txt"``, ``"json"``, ``"both"``.
        txt_path: Optional explicit path for the text file.
        json_path: Optional explicit path for the JSON file.

    Returns:
        ``(txt_path, json_path)`` where each element is ``None`` if that
        format was not requested.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    written_txt: Path | None = None
    written_json: Path | None = None

    if fmt in ("txt", "both"):
        target = txt_path or (out_dir / "interactive_test_questions.txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_txt(records, target)
        written_txt = target

    if fmt in ("json", "both"):
        target = json_path or (out_dir / "interactive_test_questions.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_json(records, target)
        written_json = target

    return written_txt, written_json


def _resolve_source_path(source: str, kg_csv: Path, split_dir: Path) -> Path:
    """Return the concrete file path for a given --source value."""
    if source == "kg":
        return kg_csv
    if source == "train":
        return split_dir / "train.csv"
    if source == "test":
        return split_dir / "test.csv"
    if source == "partial":
        return split_dir / "tier_assignments.json"
    raise SystemExit(f"Unknown --source value: {source!r}")


def _missing_source_error(source: str, path: Path) -> str:
    """Build a clear, single-step remediation message for a missing source."""
    if source == "kg":
        return (
            f"ERROR: KG file not found: {path}\n"
            f"--source kg requires the PrimeKG drug-disease subset. Run:\n"
            f"    python scripts/download_primekg.py --skip_summary"
        )
    return (
        f"ERROR: {path.name} not found at {path}\n"
        f"--source {source} requires the eval split to have been run:\n"
        f"    python scripts/eval/split_primekg.py "
        f"--input data/kg_drug_disease.csv\n"
        f"Or use --source kg (works without the eval split)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sample a handful of drug-related test questions from PrimeKG "
            "for use with the interactive agent."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Default: partial-tier drugs (from tier_assignments.json)\n"
            "  python scripts/sample_test_questions.py\n"
            "\n"
            "  # Sample from the full PrimeKG drug-disease subset\n"
            "  python scripts/sample_test_questions.py --source kg --n 12\n"
            "\n"
            "  # Sample from the train / test splits\n"
            "  python scripts/sample_test_questions.py --source train --n 9\n"
            "  python scripts/sample_test_questions.py --source test  --n 9\n"
            "\n"
            "  # Text-only output (copy-paste friendly)\n"
            "  python scripts/sample_test_questions.py --format txt\n"
            "\n"
            "  # Custom text path (implies --format txt)\n"
            "  python scripts/sample_test_questions.py \\\n"
            "      --txt-path data/eval/my_interactive_questions.txt\n"
            "\n"
            "Using the output with the interactive agent:\n"
            "  source export_dual_llm.sh\n"
            "  python scripts/interactive_agent.py --validate --no-persist --verbose\n"
            "  # then paste questions from data/eval/interactive_test_questions.txt\n"
        ),
    )
    parser.add_argument(
        "--source", choices=["kg", "train", "test", "partial"],
        default="partial",
        help=(
            "Where to sample from: 'kg' for the full PrimeKG subset, "
            "'train'/'test' for the eval split CSVs, or 'partial' (default) "
            "for drugs labelled Partial tier in tier_assignments.json "
            "(guaranteed held-out answers -> best for dual-LLM testing)."
        ),
    )
    parser.add_argument(
        "--n", type=int, default=12,
        help="Total number of questions to sample (default: 12)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="RNG seed (default: 42)",
    )
    parser.add_argument(
        "--kg-csv", default="data/kg_drug_disease.csv",
        help=(
            "Drug-disease KG CSV used when --source kg "
            "(default: data/kg_drug_disease.csv)"
        ),
    )
    parser.add_argument(
        "--split-dir", default="data/eval",
        help=(
            "Directory containing train.csv, test.csv, and "
            "tier_assignments.json; used when --source is train/test/partial "
            "(default: data/eval)"
        ),
    )
    parser.add_argument(
        "--out-dir", default="data/eval",
        help="Output directory for default filenames (default: data/eval)",
    )
    parser.add_argument(
        "--format", choices=["txt", "json", "both"], default="both",
        help=(
            "Which output format(s) to write: 'txt' for the flat "
            "one-question-per-line text file, 'json' for the structured "
            "records with gold triples, or 'both' (default)."
        ),
    )
    parser.add_argument(
        "--txt-path", default=None,
        help=(
            "Explicit path for the text output (overrides "
            "--out-dir/interactive_test_questions.txt). Implies --format txt "
            "if --format was left at default and --json-path was not set."
        ),
    )
    parser.add_argument(
        "--json-path", default=None,
        help=(
            "Explicit path for the JSON output (overrides "
            "--out-dir/interactive_test_questions.json)."
        ),
    )

    args = parser.parse_args()

    fmt = args.format
    if args.txt_path and not args.json_path and fmt == "both":
        fmt = "txt"
    elif args.json_path and not args.txt_path and fmt == "both":
        fmt = "json"

    source_path = _resolve_source_path(
        args.source, Path(args.kg_csv), Path(args.split_dir)
    )
    if not source_path.exists():
        raise SystemExit(_missing_source_error(args.source, source_path))

    if args.source == "partial":
        with source_path.open() as f:
            tier_assignments = json.load(f)
        records = sample_partial_questions(
            tier_assignments, n=args.n, seed=args.seed,
        )
        if not records:
            raise SystemExit(
                f"No partial-tier entries found in {source_path}. "
                f"Re-run the split with more data:\n"
                f"    python scripts/eval/split_primekg.py "
                f"--input data/kg_drug_disease.csv"
            )
    else:
        df = _load_csv(source_path, label=args.source)
        if df.empty:
            raise SystemExit(
                f"No drug->disease rows with relations {RELATIONS} in "
                f"{source_path}"
            )
        records = sample_kg_questions(
            df, n=args.n, seed=args.seed,
            source_label=SOURCE_LABELS[args.source],
        )

    txt_path, json_path = write_outputs(
        records,
        Path(args.out_dir),
        fmt=fmt,
        txt_path=Path(args.txt_path) if args.txt_path else None,
        json_path=Path(args.json_path) if args.json_path else None,
    )

    print(
        f"Sampled {len(records)} questions from --source {args.source} "
        f"({source_path})"
    )
    if txt_path is not None:
        print(f"  txt:  {txt_path}")
    if json_path is not None:
        print(f"  json: {json_path}")


if __name__ == "__main__":
    main()
