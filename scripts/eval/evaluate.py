#!/usr/bin/env python
"""
Evaluate the KG-RAG system on the PrimeKG drug-disease QA dataset.

All configs use Neo4j as the knowledge store.

    Config A  – Pure RAG          (KGRagGenerator + Neo4j, no expansion)
    Config B  – Agent, no valid.  (MCPAgentWithValidation, validation off)
    Config C  – Full system       (MCPAgentWithValidation, validation on)

Metrics: Exact Match, Token F1, Answer Rate, Hallucination Proxy, Latency.

Prerequisites:
    source export_local_qwen3.sh   # or export_google_ai.sh
    # Neo4j must be running with PrimeKG data indexed

Usage:
    python scripts/eval/evaluate.py --qa-dataset data/eval/qa_dataset.json
    python scripts/eval/evaluate.py --qa-dataset data/eval/qa_dataset.json --configs A B
    python scripts/eval/evaluate.py --qa-dataset data/eval/qa_dataset.json --configs C --verbose
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
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

_REFUSAL_PATTERNS = re.compile(
    r"(i don.t know|i cannot|no information|not enough|"
    r"unable to answer|cannot determine|no relevant)",
    re.IGNORECASE,
)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


@dataclass
class QAJudger:
    """Stateless scorer for a single QA pair."""

    @staticmethod
    def exact_match(prediction: str, gold_answers: list[str]) -> bool:
        pred_lower = prediction.lower()
        return any(g.lower() in pred_lower for g in gold_answers)

    @staticmethod
    def token_f1(prediction: str, gold_answers: list[str]) -> float:
        pred_tokens = set(_tokenize(prediction))
        if not pred_tokens:
            return 0.0
        best = 0.0
        for gold in gold_answers:
            gold_tokens = set(_tokenize(gold))
            if not gold_tokens:
                continue
            common = pred_tokens & gold_tokens
            if not common:
                continue
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(gold_tokens)
            f1 = 2 * precision * recall / (precision + recall)
            best = max(best, f1)
        return best

    @staticmethod
    def is_refusal(prediction: str) -> bool:
        return bool(_REFUSAL_PATTERNS.search(prediction)) or len(prediction.strip()) == 0

    @staticmethod
    def hallucination_score(
        prediction: str,
        gold_answers: list[str],
        retrieved_facts: list[str],
    ) -> float:
        """Fraction of capitalised entity-like tokens not in gold or facts."""
        pred_tokens = set(_tokenize(prediction))
        known_tokens: set[str] = set()
        for text in gold_answers + retrieved_facts:
            known_tokens.update(_tokenize(text))
        if not pred_tokens:
            return 0.0
        novel = pred_tokens - known_tokens
        return len(novel) / len(pred_tokens)


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
    prediction: str = ""
    exact_match: bool = False
    token_f1: float = 0.0
    is_refusal: bool = False
    hallucination: float = 0.0
    latency_s: float = 0.0
    config: str = ""
    # Agent-specific (Config B/C)
    validated_triplets_count: int = 0
    rejected_triplets_count: int = 0
    persisted_count: int = 0
    error: str = ""


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------

def _run_config_a(
    questions: list[dict],
    store_type: str,
    neo4j_uri: str,
    neo4j_password: str,
    index_dir: str,
    verbose: bool,
) -> list[QuestionResult]:
    """Config A: Pure RAG via KGRagGenerator (Neo4j or FAISS backend)."""
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
            logger.info(f"Config A: loaded FAISS index from {index_dir}")
        else:
            indexer = KGRagIndexer(
                store_type="neo4j",
                neo4j_uri=neo4j_uri,
                neo4j_password=neo4j_password,
            )
            logger.info(f"Config A: connected to Neo4j at {neo4j_uri}")
    except Exception as e:
        logger.error(f"Config A: failed to initialize {store_type} store: {e}")
        return []

    generator = KGRagGenerator(indexer)
    judger = QAJudger()
    results: list[QuestionResult] = []

    for i, q in enumerate(questions):
        qr = QuestionResult(
            qid=q["id"],
            question=q["question"],
            relation=q["relation"],
            direction=q["direction"],
            gold_answers=q["gold_answers"],
            config="A",
        )
        try:
            t0 = time.time()
            gen = generator.generate(
                query=q["question"],
                top_k=10,
                expand_triplets=False,
            )
            qr.latency_s = time.time() - t0
            qr.prediction = gen.answer

            retrieved_strs = [
                f"{s} {p} {o}" for s, p, o in gen.sources
            ]
            qr.exact_match = judger.exact_match(gen.answer, q["gold_answers"])
            qr.token_f1 = judger.token_f1(gen.answer, q["gold_answers"])
            qr.is_refusal = judger.is_refusal(gen.answer)
            qr.hallucination = judger.hallucination_score(
                gen.answer, q["gold_answers"], retrieved_strs
            )
        except Exception as e:
            qr.error = str(e)
            logger.warning(f"Config A q={q['id']}: {e}")

        results.append(qr)
        if verbose:
            _print_progress("A", i + 1, len(questions), qr)

    return results


def _run_config_bc(
    questions: list[dict],
    enable_validation: bool,
    neo4j_uri: str,
    neo4j_password: str,
    verbose: bool,
) -> list[QuestionResult]:
    """Config B or C via MCPAgentWithValidation."""
    config_label = "C" if enable_validation else "B"
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
        logger.error(f"Config {config_label}: failed to create agent: {e}")
        return []

    judger = QAJudger()
    results: list[QuestionResult] = []

    for i, q in enumerate(questions):
        qr = QuestionResult(
            qid=q["id"],
            question=q["question"],
            relation=q["relation"],
            direction=q["direction"],
            gold_answers=q["gold_answers"],
            config=config_label,
        )
        try:
            t0 = time.time()
            res = agent.run(query=q["question"], verbose=verbose)
            qr.latency_s = time.time() - t0
            qr.prediction = res.answer

            retrieved_strs = [
                str(tc.get("result", ""))
                for tc in res.tool_calls
                if tc.get("tool") in ("search_knowledge", "search_knowledge_graph")
            ]
            qr.exact_match = judger.exact_match(res.answer, q["gold_answers"])
            qr.token_f1 = judger.token_f1(res.answer, q["gold_answers"])
            qr.is_refusal = judger.is_refusal(res.answer)
            qr.hallucination = judger.hallucination_score(
                res.answer, q["gold_answers"], retrieved_strs
            )
            qr.validated_triplets_count = len(getattr(res, "validated_triplets", []))
            qr.rejected_triplets_count = len(getattr(res, "rejected_triplets", []))
            qr.persisted_count = getattr(res, "persisted_count", 0)
        except Exception as e:
            qr.error = str(e)
            logger.warning(f"Config {config_label} q={q['id']}: {e}")

        results.append(qr)
        if verbose:
            _print_progress(config_label, i + 1, len(questions), qr)

    return results


def _print_progress(config: str, idx: int, total: int, qr: QuestionResult) -> None:
    em_str = "Y" if qr.exact_match else "N"
    print(
        f"  [{config}] {idx}/{total}  EM={em_str}  F1={qr.token_f1:.2f}  "
        f"lat={qr.latency_s:.1f}s  {qr.question[:60]}"
    )


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def _aggregate(results: list[QuestionResult]) -> dict[str, Any]:
    """Compute aggregate metrics overall and per-relation."""
    if not results:
        return {}

    def _metrics(items: list[QuestionResult]) -> dict[str, Any]:
        n = len(items)
        if n == 0:
            return {}
        answered = [r for r in items if not r.is_refusal and not r.error]
        return {
            "count": n,
            "exact_match": round(sum(r.exact_match for r in items) / n, 4),
            "token_f1_mean": round(sum(r.token_f1 for r in items) / n, 4),
            "answer_rate": round(len(answered) / n, 4),
            "hallucination_mean": round(
                sum(r.hallucination for r in items) / n, 4
            ),
            "latency_mean_s": round(sum(r.latency_s for r in items) / n, 2),
            "errors": sum(1 for r in items if r.error),
        }

    by_relation: dict[str, list[QuestionResult]] = defaultdict(list)
    for r in results:
        by_relation[r.relation].append(r)

    agg: dict[str, Any] = {
        "overall": _metrics(results),
        "by_relation": {
            rel: _metrics(items) for rel, items in sorted(by_relation.items())
        },
    }

    # Agent-specific aggregates (configs B/C)
    val_counts = [r.validated_triplets_count for r in results]
    rej_counts = [r.rejected_triplets_count for r in results]
    if any(val_counts) or any(rej_counts):
        n = len(results)
        agg["validation"] = {
            "validated_mean": round(sum(val_counts) / n, 2),
            "rejected_mean": round(sum(rej_counts) / n, 2),
            "persisted_mean": round(
                sum(r.persisted_count for r in results) / n, 2
            ),
        }

    return agg


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    qa_path: str,
    configs: list[str],
    output_path: str,
    store_type: str = "neo4j",
    index_dir: str = "./output/rag_primekb",
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_password: str = "password123",
    verbose: bool = False,
) -> dict[str, Any]:
    """Run evaluation and return the full report dict."""
    with open(qa_path) as f:
        questions = json.load(f)
    logger.info(f"Loaded {len(questions)} questions from {qa_path}")

    report: dict[str, Any] = {
        "qa_dataset": qa_path,
        "num_questions": len(questions),
        "configs": {},
    }

    for cfg in configs:
        cfg = cfg.upper()
        print(f"\n{'='*60}")
        print(f"Running Config {cfg}")
        print(f"{'='*60}")

        if cfg == "A":
            results = _run_config_a(
                questions,
                store_type=store_type,
                neo4j_uri=neo4j_uri,
                neo4j_password=neo4j_password,
                index_dir=index_dir,
                verbose=verbose,
            )
        elif cfg == "B":
            results = _run_config_bc(
                questions,
                enable_validation=False,
                neo4j_uri=neo4j_uri,
                neo4j_password=neo4j_password,
                verbose=verbose,
            )
        elif cfg == "C":
            results = _run_config_bc(
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
            logger.warning(f"Config {cfg} produced no results (prerequisites missing?)")
            report["configs"][cfg] = {"error": "no results (check prerequisites)"}
            continue

        agg = _aggregate(results)
        report["configs"][cfg] = {
            "aggregate": agg,
            "results": [asdict(r) for r in results],
        }

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
    header = f"{'Metric':30s}" + "".join(f"{'Config '+c:>16s}" for c in cfg_labels)

    print("\n" + "=" * (30 + 16 * len(cfg_labels)))
    print("EVALUATION SUMMARY")
    print("=" * (30 + 16 * len(cfg_labels)))
    print(header)
    print("-" * (30 + 16 * len(cfg_labels)))

    metrics = [
        ("Exact Match", "exact_match"),
        ("Token F1 (mean)", "token_f1_mean"),
        ("Answer Rate", "answer_rate"),
        ("Hallucination (mean)", "hallucination_mean"),
        ("Latency (s, mean)", "latency_mean_s"),
        ("Errors", "errors"),
    ]

    for label, key in metrics:
        row = f"{label:30s}"
        for c in cfg_labels:
            agg = configs[c].get("aggregate", {}).get("overall", {})
            val = agg.get(key, "N/A")
            if isinstance(val, float):
                row += f"{val:>16.4f}"
            else:
                row += f"{str(val):>16s}"
        print(row)

    # Per-relation breakdown
    all_relations: set[str] = set()
    for c in cfg_labels:
        by_rel = configs[c].get("aggregate", {}).get("by_relation", {})
        all_relations.update(by_rel.keys())

    if all_relations:
        print(f"\n{'--- Per-relation Exact Match ---':^{30 + 16 * len(cfg_labels)}}")
        for rel in sorted(all_relations):
            row = f"  {rel:28s}"
            for c in cfg_labels:
                by_rel = configs[c].get("aggregate", {}).get("by_relation", {})
                val = by_rel.get(rel, {}).get("exact_match", "N/A")
                if isinstance(val, float):
                    row += f"{val:>16.4f}"
                else:
                    row += f"{str(val):>16s}"
            print(row)

    # Validation stats (if present)
    has_val = any(
        "validation" in configs[c].get("aggregate", {})
        for c in cfg_labels
    )
    if has_val:
        print(f"\n{'--- Validation Stats ---':^{30 + 16 * len(cfg_labels)}}")
        for key_label, key_name in [
            ("Validated (mean)", "validated_mean"),
            ("Rejected (mean)", "rejected_mean"),
            ("Persisted (mean)", "persisted_mean"),
        ]:
            row = f"  {key_label:28s}"
            for c in cfg_labels:
                val_stats = configs[c].get("aggregate", {}).get("validation", {})
                val = val_stats.get(key_name, "")
                if isinstance(val, (int, float)):
                    row += f"{val:>16.2f}"
                else:
                    row += f"{'':>16s}"
            print(row)

    print("=" * (30 + 16 * len(cfg_labels)))


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
        default=["A", "B", "C"],
        help="Configs to run: A (pure-rag), B (agent-no-val), C (full) (default: A B C)",
    )
    parser.add_argument(
        "--output",
        default="data/eval/eval_report.json",
        help="Output report JSON (default: data/eval/eval_report.json)",
    )
    parser.add_argument(
        "--store-type",
        choices=["neo4j", "faiss"],
        default="neo4j",
        help="Backend for Config A retrieval (default: neo4j)",
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

    report = evaluate(
        qa_path=args.qa_dataset,
        configs=args.configs,
        output_path=args.output,
        store_type=args.store_type,
        index_dir=args.index_dir,
        neo4j_uri=args.neo4j_uri,
        neo4j_password=neo4j_password,
        verbose=args.verbose,
    )

    _print_summary(report)
    print(f"\nFull report: {args.output}")


if __name__ == "__main__":
    main()
