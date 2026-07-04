#!/usr/bin/env python
"""
Merge a supplementary evaluation report into an original report.

Replaces error entries in the original with scored results from the
supplement (matched by qid), recomputes aggregate metrics, and writes
the merged report to a new file.  The original report is never modified.

Usage:
    python scripts/eval/merge_eval_reports.py \
        --original data/eval/eval_N194_noval_val_f30_p70_20260415_065258.json \
        --supplement data/eval/eval_N18_val_*.json \
        --output data/eval/eval_N194_merged.json
"""

import argparse
import glob
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any


def _metrics(items: list[dict]) -> dict[str, Any]:
    n = len(items)
    if n == 0:
        return {}
    fmt_fails = [r for r in items if r.get("format_failure")]
    valid = [r for r in items if not r.get("error") and not r.get("format_failure")]
    n_valid = len(valid)
    errors = sum(1 for r in items if r.get("error"))
    if n_valid == 0:
        return {
            "count": n, "scored_count": 0,
            "errors": errors, "format_failures": len(fmt_fails),
        }
    answered = [r for r in valid if not r.get("is_refusal")]
    m = {
        "count": n,
        "scored_count": n_valid,
        "gold_recall_mean": round(
            sum(r.get("gold_recall", 0) for r in valid) / n_valid, 4
        ),
        "answer_rate": round(len(answered) / n_valid, 4),
        "hallucination_mean": round(
            sum(r.get("hallucination", 0) for r in valid) / n_valid, 4
        ),
        "latency_mean_s": round(
            sum(r.get("latency_s", 0) for r in valid) / n_valid, 2
        ),
        "errors": errors,
        "format_failures": len(fmt_fails),
    }
    val_counts = [r.get("validated_triplets_count", 0) for r in valid]
    rej_counts = [r.get("rejected_triplets_count", 0) for r in valid]
    if any(val_counts) or any(rej_counts):
        m["validated_mean"] = round(sum(val_counts) / n_valid, 2)
        m["rejected_mean"] = round(sum(rej_counts) / n_valid, 2)
        m["persisted_mean"] = round(
            sum(r.get("persisted_count", 0) for r in valid) / n_valid, 2
        )
    return m


def _aggregate(results: list[dict]) -> dict[str, Any]:
    by_relation: dict[str, list[dict]] = defaultdict(list)
    by_tier: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_relation[r.get("relation", "unknown")].append(r)
        by_tier[r.get("tier", "unknown")].append(r)
    return {
        "overall": _metrics(results),
        "by_relation": {
            rel: _metrics(items) for rel, items in sorted(by_relation.items())
        },
        "by_tier": {
            tier: _metrics(items) for tier, items in sorted(by_tier.items())
        },
    }


def _print_comparison(label: str, before: dict, after: dict):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    keys = [
        ("gold_recall_mean", "Gold Recall (mean)"),
        ("answer_rate", "Answer Rate"),
        ("hallucination_mean", "Hallucination (mean)"),
        ("latency_mean_s", "Latency (s, mean)"),
        ("errors", "Errors"),
        ("scored_count", "Scored questions"),
    ]
    print(f"{'Metric':35s} {'Before':>10s} {'After':>10s} {'Delta':>10s}")
    print("-" * 70)
    for key, name in keys:
        bv = before.get(key, 0)
        av = after.get(key, 0)
        delta = av - bv
        if isinstance(bv, float):
            print(f"{name:35s} {bv:>10.4f} {av:>10.4f} {delta:>+10.4f}")
        else:
            print(f"{name:35s} {bv:>10d} {av:>10d} {delta:>+10d}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge supplementary evaluation results into an original report."
    )
    parser.add_argument(
        "--original", required=True,
        help="Path to the original evaluation JSON report",
    )
    parser.add_argument(
        "--supplement", required=True,
        help="Path (or glob pattern) to the supplementary JSON report",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path for merged report (default: auto-generated)",
    )
    args = parser.parse_args()

    matches = glob.glob(args.supplement)
    if not matches:
        print(f"ERROR: No files matching '{args.supplement}'", file=sys.stderr)
        sys.exit(1)
    supplement_path = sorted(matches)[-1]

    with open(args.original) as f:
        original = json.load(f)
    with open(supplement_path) as f:
        supplement = json.load(f)

    print(f"Original:   {args.original}")
    print(f"Supplement: {supplement_path}")

    merged = json.loads(json.dumps(original))

    for cfg_name, cfg_data in merged["configs"].items():
        supp_cfg = supplement.get("configs", {}).get(cfg_name)
        if not supp_cfg:
            continue

        supp_by_qid = {r["qid"]: r for r in supp_cfg["results"]}

        replaced = 0
        still_error = 0
        new_results = []
        for r in cfg_data["results"]:
            if r.get("error") and r["qid"] in supp_by_qid:
                replacement = supp_by_qid[r["qid"]]
                if not replacement.get("error"):
                    new_results.append(replacement)
                    replaced += 1
                else:
                    new_results.append(replacement)
                    still_error += 1
            else:
                new_results.append(r)

        cfg_data["results"] = new_results
        cfg_data["aggregate"] = _aggregate(new_results)

        print(f"\n  [{cfg_name}] Replaced {replaced} error results, "
              f"{still_error} still errored")

        _print_comparison(
            f"Config C ({cfg_name}) — Before vs After merge",
            original["configs"][cfg_name]["aggregate"]["overall"],
            cfg_data["aggregate"]["overall"],
        )

        for tier in cfg_data["aggregate"].get("by_tier", {}):
            orig_tier = original["configs"][cfg_name]["aggregate"].get("by_tier", {}).get(tier, {})
            merged_tier = cfg_data["aggregate"]["by_tier"][tier]
            _print_comparison(
                f"  Tier: {tier}",
                orig_tier,
                merged_tier,
            )

    if args.output:
        output_path = args.output
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        base = args.original.rsplit(".", 1)[0]
        output_path = f"{base}_merged_{ts}.json"

    merged["merge_info"] = {
        "original": args.original,
        "supplement": supplement_path,
        "merged_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"\nMerged report written to: {output_path}")


if __name__ == "__main__":
    main()
