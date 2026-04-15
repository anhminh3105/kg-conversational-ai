#!/usr/bin/env python
"""
Evaluate the KG-RAG system on the PrimeKG drug-disease QA dataset.

All configs use Neo4j as the knowledge store.

    rag                  – Pure RAG (KGRagGenerator + Neo4j, no expansion)
    without-validation   – Agent without remote LLM validation
    with-validation      – Agent with remote LLM validation (propose + fact-check)

Metrics: Gold Recall, Answer Rate, Hallucination Proxy, Latency.

Prerequisites:
    source export_local_qwen3.sh   # or export_google_ai.sh
    # Neo4j must be running with PrimeKG data indexed

Usage:
    python scripts/eval/evaluate.py --qa-dataset data/eval/qa_dataset.json
    python scripts/eval/evaluate.py --qa-dataset data/eval/qa_dataset.json --configs without-validation
    python scripts/eval/evaluate.py --qa-dataset data/eval/qa_dataset.json --configs with-validation --verbose
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# QA Judger – scoring helpers
# ---------------------------------------------------------------------------

_STATUS_LINE_RE = re.compile(r"^status:\s*(accepted|refused)\s*$", re.IGNORECASE | re.MULTILINE)
_THOUGHT_RE = re.compile(r"<thought>.*?</thought>", re.DOTALL)

_QUALIFIER_RE = re.compile(r"\s*\([^)]*\)\s*$")


class FormatError(Exception):
    """Raised when the LLM answer is missing the required status line."""


def _strip_thought_blocks(text: str) -> str:
    """Remove <thought>...</thought> blocks produced by some models (e.g. Gemma 4)."""
    return _THOUGHT_RE.sub("", text).strip()


def _strip_status_line(text: str) -> str:
    """Remove 'status: …' line and clean special characters."""
    text = _strip_thought_blocks(text)
    text = _STATUS_LINE_RE.sub("", text)
    text = re.sub(r"[\n\r\t]+", " ", text)
    text = re.sub(r"\s*[-*•]\s+", " ", text)
    text = re.sub(r"[()[\]{}<>]", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _normalize(text: str) -> str:
    """Strip trailing parenthetical qualifiers like '(disease)' for matching."""
    return _QUALIFIER_RE.sub("", text).strip()


def _strip_non_alnum(text: str) -> str:
    """Remove all non-alphanumeric ASCII characters except spaces."""
    return re.sub(r"[^a-zA-Z0-9 ]+", "", text)


def _parse_retrieved_facts(retrieved_facts: list[str]) -> list[dict]:
    """Parse the raw retrieved-facts strings into a flat list of fact dicts."""
    import ast
    facts: list[dict] = []
    for raw in retrieved_facts:
        try:
            data = ast.literal_eval(raw)
            if isinstance(data, dict):
                facts.extend(data.get("facts", []))
        except Exception:
            pass
    return facts


def _best_retrieval_score(entity_clean: str, parsed_facts: list[dict]) -> float:
    """Return the highest retrieval score among facts that mention *entity_clean*."""
    best = 0.0
    for f in parsed_facts:
        subj = _strip_non_alnum(f.get("subject", "")).lower()
        obj = _strip_non_alnum(f.get("object", "")).lower()
        if entity_clean in (subj, obj):
            best = max(best, float(f.get("score", 0)))
    return best


@dataclass
class QAJudger:
    """Stateless scorer for a single QA pair."""

    @staticmethod
    def gold_recall(prediction: str, gold_answers: list[str]) -> float:
        """Fraction of ALL gold answers found in the prediction."""
        if not gold_answers:
            return 0.0
        pred_clean = _strip_non_alnum(prediction).lower()
        found = sum(
            1 for a in gold_answers
            if _strip_non_alnum(_normalize(a)).lower() in pred_clean
        )
        return found / len(gold_answers)

    @staticmethod
    def is_refusal(prediction: str) -> bool:
        clean = _strip_thought_blocks(prediction).strip()
        match = _STATUS_LINE_RE.search(clean)
        if match:
            return match.group(1).lower() == "refused"
        raise FormatError(f"Missing status line in response")

    @staticmethod
    def hallucination_score(
        prediction: str,
        gold_answers: list[str],
        kg_answers: list[str],
        retrieved_facts: list[str],
        retrieval_score_threshold: float = 0.75,
    ) -> float:
        """Fraction of mentioned gold answers NOT grounded in KG.

        A mentioned gold answer is grounded if it appears in *kg_answers*
        (exact match) **or** if the entity appears in a retrieved fact whose
        retrieval score >= *retrieval_score_threshold* (semantic grounding).
        """
        pred_clean = _strip_non_alnum(prediction).lower()
        kg_set = {_strip_non_alnum(_normalize(a)).lower() for a in kg_answers}

        mentioned = [
            g for g in gold_answers
            if _strip_non_alnum(_normalize(g)).lower() in pred_clean
        ]
        if not mentioned:
            return 0.0

        parsed_facts = _parse_retrieved_facts(retrieved_facts)

        not_grounded = 0
        for g in mentioned:
            g_clean = _strip_non_alnum(_normalize(g)).lower()
            if g_clean in kg_set:
                continue
            best_score = _best_retrieval_score(g_clean, parsed_facts)
            if best_score < retrieval_score_threshold:
                not_grounded += 1
        return not_grounded / len(mentioned)


# ---------------------------------------------------------------------------
# Per-question result
# ---------------------------------------------------------------------------

@dataclass
class QuestionResult:
    qid: str
    question: str
    relation: str
    direction: str
    gold_answers: list[str]
    tier: str = "partial"
    prediction: str = ""
    gold_recall: float = 0.0
    kg_answers: list[str] = field(default_factory=list)
    is_refusal: bool = False
    hallucination: float = 0.0
    latency_s: float = 0.0
    config: str = ""
    retrieved_facts: list[str] = field(default_factory=list)
    format_failure: bool = False
    # with-validation specific
    validated_triplets_count: int = 0
    rejected_triplets_count: int = 0
    persisted_count: int = 0
    error: str = ""


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------

def _run_rag(
    questions: list[dict],
    store_type: str,
    neo4j_uri: str,
    neo4j_password: str,
    index_dir: str,
    verbose: bool,
) -> list[QuestionResult]:
    """Pure RAG via KGRagGenerator (Neo4j or FAISS backend)."""
    try:
        from rag.kg_rag_indexer import KGRagIndexer
        from rag.generator import KGRagGenerator
    except Exception as e:
        logger.error(
            f"Cannot import RAG modules: {e}\n"
            "  Ensure an LLM backend is configured:\n"
            "    source export_local_qwen3.sh   (local)\n"
            "    source export_google_ai.sh     (remote)"
        )
        return []

    try:
        if store_type == "faiss":
            indexer = KGRagIndexer.load(index_dir)
            logger.info(f"rag: loaded FAISS index from {index_dir}")
        else:
            indexer = KGRagIndexer(
                store_type="neo4j",
                neo4j_uri=neo4j_uri,
                neo4j_password=neo4j_password,
            )
            logger.info(f"rag: connected to Neo4j at {neo4j_uri}")
    except Exception as e:
        logger.error(f"rag: failed to initialize {store_type} store: {e}")
        return []

    generator = KGRagGenerator(indexer)
    judger = QAJudger()
    results: list[QuestionResult] = []

    for i, q in enumerate(questions):
        q_kg_answers = q.get("kg_answers", [])
        qr = QuestionResult(
            qid=q["id"],
            question=q["question"],
            relation=q["relation"],
            direction=q["direction"],
            gold_answers=q["gold_answers"],
            tier=q.get("tier", "partial"),
            kg_answers=q_kg_answers,
            config="rag",
        )
        try:
            t0 = time.time()
            gen = generator.generate(
                query=q["question"],
                top_k=10,
                expand_triplets=False,
            )
            qr.latency_s = time.time() - t0
            qr.is_refusal = judger.is_refusal(gen.answer)
            qr.prediction = _strip_status_line(gen.answer)

            retrieved_strs = [
                f"{s} {p} {o}" for s, p, o in gen.sources
            ]
            qr.retrieved_facts = retrieved_strs
            qr.gold_recall = judger.gold_recall(qr.prediction, q["gold_answers"])
            qr.hallucination = judger.hallucination_score(
                qr.prediction, q["gold_answers"], q_kg_answers, retrieved_strs
            )
        except Exception as e:
            qr.error = str(e)
            logger.warning(f"rag q={q['id']}: {e}")

        results.append(qr)
        if verbose:
            _print_progress("rag", i + 1, len(questions), qr)

    return results


def _run_agent(
    questions: list[dict],
    enable_validation: bool,
    neo4j_uri: str,
    neo4j_password: str,
    verbose: bool,
) -> list[QuestionResult]:
    """Agent runner, with or without remote LLM validation."""
    config_label = "with-validation" if enable_validation else "without-validation"
    try:
        from rag.mcp_agent import create_mcp_agent_with_validation
    except Exception as e:
        logger.error(
            f"Cannot import agent module: {e}\n"
            "  Ensure an LLM backend is configured:\n"
            "    source export_local_qwen3.sh   (local)\n"
            "    source export_google_ai.sh     (remote)"
        )
        return []

    try:
        agent = create_mcp_agent_with_validation(
            neo4j_uri=neo4j_uri,
            neo4j_password=neo4j_password,
            enable_validation=enable_validation,
            auto_expand=True,
        )
    except Exception as e:
        logger.error(f"{config_label}: failed to create agent: {e}")
        return []

    judger = QAJudger()
    results: list[QuestionResult] = []
    max_format_retries = 3

    for i, q in enumerate(questions):
        q_kg_answers = q.get("kg_answers", [])
        qr = QuestionResult(
            qid=q["id"],
            question=q["question"],
            relation=q["relation"],
            direction=q["direction"],
            gold_answers=q["gold_answers"],
            tier=q.get("tier", "partial"),
            kg_answers=q_kg_answers,
            config=config_label,
        )

        for attempt in range(1, max_format_retries + 1):
            try:
                t0 = time.time()
                res = agent.run(query=q["question"], verbose=verbose)
                qr.latency_s = time.time() - t0
                qr.is_refusal = judger.is_refusal(res.answer)
                qr.prediction = _strip_status_line(res.answer)

                retrieved_strs = [
                    str(tc.get("result", ""))
                    for tc in res.tool_calls
                    if tc.get("tool") in ("search_knowledge", "search_knowledge_graph")
                ]
                qr.retrieved_facts = retrieved_strs
                qr.gold_recall = judger.gold_recall(qr.prediction, q["gold_answers"])
                qr.hallucination = judger.hallucination_score(
                    qr.prediction, q["gold_answers"], q_kg_answers, retrieved_strs
                )
                qr.validated_triplets_count = len(getattr(res, "validated_triplets", []))
                qr.rejected_triplets_count = len(getattr(res, "rejected_triplets", []))
                qr.persisted_count = getattr(res, "persisted_count", 0)
                break
            except FormatError as e:
                if attempt < max_format_retries:
                    logger.warning(
                        f"{config_label} q={q['id']}: attempt {attempt}/{max_format_retries} "
                        f"- {e}, retrying..."
                    )
                    continue
                qr.format_failure = True
                qr.prediction = _strip_status_line(res.answer)
                logger.warning(
                    f"{config_label} q={q['id']}: format failure after "
                    f"{max_format_retries} attempts - skipping scoring"
                )
            except Exception as e:
                qr.error = str(e)
                logger.warning(f"{config_label} q={q['id']}: {e}")
                break

        results.append(qr)
        if verbose:
            _print_progress(config_label, i + 1, len(questions), qr)

    return results


def _print_progress(config: str, idx: int, total: int, qr: QuestionResult) -> None:
    if qr.format_failure:
        tag = "FMT_FAIL"
    elif qr.error:
        tag = "ERR"
    else:
        tag = f"GR={qr.gold_recall:.2f}"
    print(
        f"  [{config}] {idx}/{total}  {tag}  "
        f"lat={qr.latency_s:.1f}s  {qr.question[:60]}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def _aggregate(results: list[QuestionResult]) -> dict[str, Any]:
    """Compute aggregate metrics overall, per-relation, and per-tier."""
    if not results:
        return {}

    def _metrics(items: list[QuestionResult]) -> dict[str, Any]:
        n = len(items)
        if n == 0:
            return {}
        fmt_fails = [r for r in items if r.format_failure]
        valid = [r for r in items if not r.error and not r.format_failure]
        n_valid = len(valid)
        errors = sum(1 for r in items if r.error)
        if n_valid == 0:
            return {
                "count": n, "scored_count": 0,
                "errors": errors, "format_failures": len(fmt_fails),
            }
        answered = [r for r in valid if not r.is_refusal]
        m = {
            "count": n,
            "scored_count": n_valid,
            "gold_recall_mean": round(
                sum(r.gold_recall for r in valid) / n_valid, 4
            ),
            "answer_rate": round(len(answered) / n_valid, 4),
            "hallucination_mean": round(
                sum(r.hallucination for r in valid) / n_valid, 4
            ),
            "latency_mean_s": round(sum(r.latency_s for r in valid) / n_valid, 2),
            "errors": errors,
            "format_failures": len(fmt_fails),
        }
        val_counts = [r.validated_triplets_count for r in valid]
        rej_counts = [r.rejected_triplets_count for r in valid]
        if any(val_counts) or any(rej_counts):
            m["validated_mean"] = round(sum(val_counts) / n_valid, 2)
            m["rejected_mean"] = round(sum(rej_counts) / n_valid, 2)
            m["persisted_mean"] = round(
                sum(r.persisted_count for r in valid) / n_valid, 2
            )
        return m

    by_relation: dict[str, list[QuestionResult]] = defaultdict(list)
    by_tier: dict[str, list[QuestionResult]] = defaultdict(list)
    for r in results:
        by_relation[r.relation].append(r)
        by_tier[r.tier].append(r)

    agg: dict[str, Any] = {
        "overall": _metrics(results),
        "by_relation": {
            rel: _metrics(items) for rel, items in sorted(by_relation.items())
        },
        "by_tier": {
            tier: _metrics(items) for tier, items in sorted(by_tier.items())
        },
    }

    return agg


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

_CONFIG_SHORT = {
    "rag": "rag",
    "without-validation": "noval",
    "with-validation": "val",
}


def _build_output_path(output_dir: str, num_questions: int,
                       split_stats: Optional[dict],
                       configs: Optional[list[str]] = None) -> str:
    """Build a descriptive filename like eval_N84_noval_val_f30_p70_20260410_153012.json"""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    parts = [f"eval_N{num_questions}"]
    if configs:
        cfg_tags = [_CONFIG_SHORT.get(c, c) for c in configs]
        parts.append("_".join(cfg_tags))
    if split_stats:
        fr = round(split_stats.get("full_ratio", 0) * 100)
        pr = 100 - fr
        parts.append(f"f{fr}_p{pr}")
    parts.append(ts)
    return os.path.join(output_dir, "_".join(parts) + ".json")


def evaluate(
    qa_path: str,
    configs: list[str],
    output_path: Optional[str] = None,
    output_dir: str = "data/eval",
    store_type: str = "neo4j",
    index_dir: str = "./output/rag_primekb",
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_password: str = "password123",
    verbose: bool = False,
    tier_filter: str = "all",
    split_stats_path: Optional[str] = None,
    question_ids: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Run evaluation and return the full report dict."""
    with open(qa_path) as f:
        questions = json.load(f)
    logger.info(f"Loaded {len(questions)} questions from {qa_path}")

    if tier_filter != "all":
        questions = [q for q in questions if q.get("tier", "partial") == tier_filter]
        logger.info(f"Filtered to {len(questions)} questions (tier={tier_filter})")

    if question_ids:
        id_set = set(question_ids)
        questions = [q for q in questions if q["id"] in id_set]
        logger.info(f"Filtered to {len(questions)} questions by ID: {question_ids}")

    split_stats: Optional[dict] = None
    if split_stats_path is None:
        default_stats = os.path.join(os.path.dirname(qa_path), "split_stats.json")
        if os.path.exists(default_stats):
            split_stats_path = default_stats
    if split_stats_path and os.path.exists(split_stats_path):
        with open(split_stats_path) as f:
            split_stats = json.load(f)

    _ALIASES = {"A": "rag", "B": "without-validation", "C": "with-validation"}
    resolved_configs = [_ALIASES.get(c.upper(), c.lower()) for c in configs]

    if output_path is None:
        output_path = _build_output_path(
            output_dir, len(questions), split_stats, resolved_configs,
        )

    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "qa_dataset": qa_path,
        "num_questions": len(questions),
        "tier_filter": tier_filter,
        "split_stats": split_stats,
        "configs": {},
    }

    for cfg in resolved_configs:
        print(f"\n{'='*60}")
        print(f"Running: {cfg}")
        print(f"{'='*60}")

        if cfg == "rag":
            results = _run_rag(
                questions,
                store_type=store_type,
                neo4j_uri=neo4j_uri,
                neo4j_password=neo4j_password,
                index_dir=index_dir,
                verbose=verbose,
            )
        elif cfg == "without-validation":
            results = _run_agent(
                questions,
                enable_validation=False,
                neo4j_uri=neo4j_uri,
                neo4j_password=neo4j_password,
                verbose=verbose,
            )
        elif cfg == "with-validation":
            results = _run_agent(
                questions,
                enable_validation=True,
                neo4j_uri=neo4j_uri,
                neo4j_password=neo4j_password,
                verbose=verbose,
            )
        else:
            logger.warning(f"Unknown config '{cfg}', skipping")
            continue

        if not results:
            logger.warning(f"{cfg}: produced no results (prerequisites missing?)")
            report["configs"][cfg] = {"error": "no results (check prerequisites)"}
            continue

        agg = _aggregate(results)
        report["configs"][cfg] = {
            "aggregate": agg,
            "results": [asdict(r) for r in results],
        }

    report["output_path"] = output_path
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Report saved to {output_path}")

    return report


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def _print_summary(report: dict[str, Any]) -> None:
    """Print a readable comparison table to stdout."""
    configs = report.get("configs", {})
    if not configs:
        print("No results to display.")
        return

    cfg_labels = sorted(configs.keys())
    col_width = max(16, max((len(c) + 2 for c in cfg_labels), default=16))
    header = f"{'Metric':30s}" + "".join(f"{c:>{col_width}s}" for c in cfg_labels)

    total_width = 30 + col_width * len(cfg_labels)

    print("\n" + "=" * total_width)
    print("EVALUATION SUMMARY")
    print("=" * total_width)
    print(header)
    print("-" * total_width)

    metrics = [
        ("Gold Recall (mean)", "gold_recall_mean"),
        ("Answer Rate", "answer_rate"),
        ("Hallucination (mean)", "hallucination_mean"),
        ("Latency (s, mean)", "latency_mean_s"),
        ("Errors", "errors"),
        ("Format Failures", "format_failures"),
    ]

    for label, key in metrics:
        row = f"{label:30s}"
        for c in cfg_labels:
            agg = configs[c].get("aggregate", {}).get("overall", {})
            val = agg.get(key, "N/A")
            if isinstance(val, float):
                row += f"{val:>{col_width}.4f}"
            else:
                row += f"{str(val):>{col_width}s}"
        print(row)

    # Per-relation breakdown
    all_relations: set[str] = set()
    for c in cfg_labels:
        by_rel = configs[c].get("aggregate", {}).get("by_relation", {})
        all_relations.update(by_rel.keys())

    if all_relations:
        print(f"\n{'--- Per-relation Gold Recall ---':^{total_width}}")
        for rel in sorted(all_relations):
            row = f"  {rel:28s}"
            for c in cfg_labels:
                by_rel = configs[c].get("aggregate", {}).get("by_relation", {})
                val = by_rel.get(rel, {}).get("gold_recall_mean", "N/A")
                if isinstance(val, float):
                    row += f"{val:>{col_width}.4f}"
                else:
                    row += f"{str(val):>{col_width}s}"
            print(row)

    # Per-tier breakdown
    all_tiers: set[str] = set()
    for c in cfg_labels:
        by_tier = configs[c].get("aggregate", {}).get("by_tier", {})
        all_tiers.update(by_tier.keys())

    if all_tiers:
        print(f"\n{'--- Per-tier Gold Recall ---':^{total_width}}")
        for tier in sorted(all_tiers):
            row = f"  {tier:28s}"
            for c in cfg_labels:
                by_tier = configs[c].get("aggregate", {}).get("by_tier", {})
                tier_data = by_tier.get(tier, {})
                cnt = tier_data.get("count", 0)
                val = tier_data.get("gold_recall_mean", "N/A")
                if isinstance(val, float):
                    row += f"{val:>{col_width - 5}.4f} ({cnt})"
                else:
                    row += f"{str(val):>{col_width}s}"
            print(row)

    # Validation stats (if present)
    has_val = any(
        "validated_mean" in configs[c].get("aggregate", {}).get("overall", {})
        for c in cfg_labels
    )
    if has_val:
        print(f"\n{'--- Validation Stats ---':^{total_width}}")
        for key_label, key_name in [
            ("Validated (mean)", "validated_mean"),
            ("Rejected (mean)", "rejected_mean"),
            ("Persisted (mean)", "persisted_mean"),
        ]:
            row = f"  {key_label:28s}"
            for c in cfg_labels:
                overall = configs[c].get("aggregate", {}).get("overall", {})
                val = overall.get(key_name, "")
                if isinstance(val, (int, float)):
                    row += f"{val:>{col_width}.2f}"
                else:
                    row += f"{'':>{col_width}s}"
            print(row)

    # List format-failure question IDs per config
    for c in cfg_labels:
        cfg_results = configs[c].get("results", [])
        ff_ids = [r["qid"] for r in cfg_results if r.get("format_failure")]
        if ff_ids:
            print(f"\n  Format-failure questions ({c}): {', '.join(ff_ids)}")

    print("=" * total_width)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate KG-RAG system on PrimeKG drug-disease QA",
    )
    parser.add_argument(
        "--qa-dataset",
        default="data/eval/qa_dataset.json",
        help="Path to QA dataset JSON (default: data/eval/qa_dataset.json)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["without-validation", "with-validation"],
        help="Configs to run: rag, without-validation, with-validation "
             "(default: without-validation with-validation). "
             "Legacy aliases: A=rag, B=without-validation, C=with-validation",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output report JSON path. If omitted, an auto-generated name "
             "with dataset size, split ratios, and timestamp is used.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/eval",
        help="Directory for auto-generated report filename (default: data/eval)",
    )
    parser.add_argument(
        "--store-type",
        choices=["neo4j", "faiss"],
        default="neo4j",
        help="Backend for 'rag' config retrieval (default: neo4j)",
    )
    parser.add_argument(
        "--index-dir",
        default="./output/rag_primekb",
        help="FAISS index directory, used only with --store-type faiss "
             "(default: ./output/rag_primekb)",
    )
    parser.add_argument(
        "--neo4j-uri",
        default="bolt://localhost:7687",
        help="Neo4j Bolt URI (default: bolt://localhost:7687)",
    )
    parser.add_argument(
        "--neo4j-password",
        default=None,
        help="Neo4j password (default: from NEO4J_PASSWORD env or 'password123')",
    )
    parser.add_argument(
        "--tier",
        choices=["all", "full", "partial"],
        default="all",
        help="Filter questions by tier (default: all)",
    )
    parser.add_argument(
        "--questions",
        default=None,
        help="Comma-separated question IDs to evaluate (e.g. q0014,q0034,q0039). "
             "If omitted, all questions are evaluated.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-question progress",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not os.path.exists(args.qa_dataset):
        logger.error(
            f"QA dataset not found: {args.qa_dataset}\n"
            "Run:  python scripts/eval/generate_qa.py   first."
        )
        sys.exit(1)

    neo4j_password = args.neo4j_password or os.environ.get(
        "NEO4J_PASSWORD", "password123"
    )

    qids = [s.strip() for s in args.questions.split(",")] if args.questions else None

    report = evaluate(
        qa_path=args.qa_dataset,
        configs=args.configs,
        output_path=args.output,
        output_dir=args.output_dir,
        store_type=args.store_type,
        index_dir=args.index_dir,
        neo4j_uri=args.neo4j_uri,
        neo4j_password=neo4j_password,
        verbose=args.verbose,
        tier_filter=args.tier,
        question_ids=qids,
    )

    _print_summary(report)
    print(f"\nFull report: {report['output_path']}")


if __name__ == "__main__":
    main()
