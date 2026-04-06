#!/usr/bin/env python
"""
Generate QA pairs from the PrimeKG drug-disease test split using an LLM.

For each relation type (indication, contraindication, off-label use) the
script uses an LLM to generate natural questions in both forward
(drug -> disease) and reverse (disease -> drug) directions.
Gold answers are multi-answer sets derived by grouping the test triplets.

Supports two modes:
  Remote LLM:  source export_google_ai.sh   (Gemini / OpenAI-compatible API)
  Local  LLM:  source export_local_qwen3.sh (Qwen 3.5-9B or any HuggingFace model)

Usage:
    python scripts/eval/generate_qa.py --test-csv data/eval/test.csv
    python scripts/eval/generate_qa.py --test-csv data/eval/test.csv --no-cache
    python scripts/eval/generate_qa.py --test-csv data/eval/test.csv --batch-size 20
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from typing import Optional

# Ensure the project root is on sys.path so `from rag...` imports work.
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Relation / direction metadata for the LLM prompt
# ---------------------------------------------------------------------------

RELATION_CONTEXT: dict[str, dict[str, str]] = {
    "indication": {
        "forward": (
            "The relation is 'indication': the drug is approved or used to "
            "treat certain diseases/conditions. Generate a question asking "
            "what diseases or conditions the given DRUG is indicated for. "
            "The question must mention the drug name."
        ),
        "reverse": (
            "The relation is 'indication': certain drugs are approved to "
            "treat this disease. Generate a question asking which DRUGS are "
            "indicated for the given DISEASE. The question must mention the "
            "disease name."
        ),
    },
    "contraindication": {
        "forward": (
            "The relation is 'contraindication': the drug should NOT be "
            "used in patients with certain conditions. Generate a question "
            "asking what diseases or conditions make this DRUG unsafe or "
            "contraindicated. The question must mention the drug name."
        ),
        "reverse": (
            "The relation is 'contraindication': certain drugs should be "
            "avoided for patients with this disease. Generate a question "
            "asking which DRUGS are contraindicated for the given DISEASE. "
            "The question must mention the disease name."
        ),
    },
    "off-label use": {
        "forward": (
            "The relation is 'off-label use': the drug is used in clinical "
            "practice for conditions it is NOT officially approved for. "
            "Generate a question asking what off-label uses the given DRUG "
            "has. The question must mention the drug name."
        ),
        "reverse": (
            "The relation is 'off-label use': certain drugs are prescribed "
            "off-label for this disease. Generate a question asking which "
            "DRUGS are used off-label for the given DISEASE. The question "
            "must mention the disease name."
        ),
    },
}

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

SYSTEM_PROMPT = (
    "You are a biomedical QA dataset generator. You produce natural, varied "
    "questions about drug-disease relationships from a knowledge graph. "
    "Each question should sound like something a medical student or clinician "
    "would ask. Vary phrasing across questions -- do not repeat the same "
    "sentence structure. Never reveal the answer in the question."
)


# ---------------------------------------------------------------------------
# Environment check
# ---------------------------------------------------------------------------

def _is_local_llm() -> bool:
    """Return True when the local-LLM path will be used."""
    return os.environ.get("USE_LOCAL_LLM", "").lower() == "true"


def _force_remote_llm() -> None:
    """Override env so the LLM module routes to the remote API."""
    os.environ["USE_LOCAL_LLM"] = ""


def _check_llm_env(remote: bool) -> None:
    """Exit early with a helpful message if the chosen backend is not configured.

    Args:
        remote: If True, require remote API vars regardless of USE_LOCAL_LLM.
    """
    if remote:
        _force_remote_llm()

    if _is_local_llm():
        model = os.environ.get("LOCAL_LLM_MODEL")
        if not model:
            print(
                "ERROR: USE_LOCAL_LLM is set but LOCAL_LLM_MODEL is missing.\n\n"
                "  Run:  source export_local_qwen3.sh\n",
                file=sys.stderr,
            )
            sys.exit(1)
        logger.info(f"LLM configured (local): model={model}")
        return

    key = os.environ.get("OPENAI_KEY") or os.environ.get("OPENAI_API_KEY")
    base = os.environ.get("OPENAI_API_BASE")
    model = os.environ.get("OPENAI_MODEL")
    if not key or not base or not model:
        print(
            "ERROR: No LLM backend configured.\n"
            "This script requires an LLM to generate questions.\n\n"
            "  Remote:  source export_google_ai.sh\n"
            "  Local:   source export_local_qwen3.sh\n",
            file=sys.stderr,
        )
        sys.exit(1)
    logger.info(f"LLM configured (remote): model={model}, base={base[:40]}...")


# ---------------------------------------------------------------------------
# Batched LLM question generation
# ---------------------------------------------------------------------------

def _generate_questions_batch(
    entities: list[str],
    relation: str,
    direction: str,
) -> dict[str, str]:
    """Call the LLM to generate one question per entity. Returns {entity: question}."""
    from rag.edc.edc.utils.llm_utils import openai_chat_completion

    context = RELATION_CONTEXT[relation][direction]
    numbered = "\n".join(f"{i+1}. {e}" for i, e in enumerate(entities))

    user_msg = (
        f"{context}\n\n"
        f"Generate one natural-sounding question for EACH entity below.\n\n"
        f"Entities:\n{numbered}\n\n"
        f"Return ONLY a JSON array of objects: "
        f'[{{"entity": "...", "question": "..."}}]\n'
        f"Do NOT include the answer in the question. "
        f"Vary the phrasing across questions."
    )

    history = [{"role": "user", "content": user_msg}]

    try:
        raw = openai_chat_completion(
            system_prompt=SYSTEM_PROMPT,
            history=history,
            temperature=0.7,
            max_tokens=2048,
        )
    except Exception as e:
        logger.warning(f"LLM call failed for {relation}/{direction} batch: {e}")
        return {}

    return _parse_llm_response(raw, entities)


def _parse_llm_response(raw: str, expected_entities: list[str]) -> dict[str, str]:
    """Parse the LLM JSON response into an {entity: question} mapping."""
    if not raw or not raw.strip():
        logger.warning("Empty LLM response")
        return {}

    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")

    items = None
    # Attempt 1: parse the whole cleaned string
    try:
        items = json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 2: find the outermost JSON array
    if items is None:
        match = re.search(r"\[\s*\{.*\}\s*\]", cleaned, re.DOTALL)
        if match:
            try:
                items = json.loads(match.group())
            except json.JSONDecodeError:
                pass

    # Attempt 3: line-by-line extraction for {"entity":..., "question":...}
    if items is None:
        items = []
        for m in re.finditer(
            r'\{\s*"entity"\s*:\s*"([^"]+)"\s*,\s*"question"\s*:\s*"([^"]+)"\s*\}',
            cleaned,
        ):
            items.append({"entity": m.group(1), "question": m.group(2)})
        if not items:
            logger.warning("Could not parse LLM response as JSON")
            logger.debug(f"Raw response (first 500 chars): {raw[:500]}")
            return {}

    if not isinstance(items, list):
        logger.warning(f"Expected list, got {type(items)}")
        return {}

    result: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        entity = item.get("entity", "")
        question = item.get("question", "")
        if entity and question:
            result[entity] = question

    return result


def _generate_all_questions(
    entity_lists: dict[tuple[str, str], list[str]],
    batch_size: int,
    cache_path: str | None,
    use_cache: bool,
) -> dict[str, str]:
    """Generate questions for all (relation, direction, entity) combos.

    Returns a flat dict keyed by "entity|relation|direction" -> question.
    """
    # Load existing cache (supports incremental resumption)
    questions: dict[str, str] = {}
    if use_cache and cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            questions = json.load(f)

        needed_keys = set()
        for (relation, direction), entities in entity_lists.items():
            for e in entities:
                needed_keys.add(f"{e}|{relation}|{direction}")

        if needed_keys.issubset(set(questions.keys())):
            logger.info(f"Cache hit: all {len(needed_keys)} questions found in {cache_path}")
            return questions
        else:
            missing = len(needed_keys - set(questions.keys()))
            logger.info(f"Cache partial: {len(questions)} cached, {missing} still needed")

    total_batches = sum(
        (len(entities) + batch_size - 1) // batch_size
        for entities in entity_lists.values()
    )
    batch_num = 0
    batches_called = 0

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)

    for (relation, direction), entities in entity_lists.items():
        for i in range(0, len(entities), batch_size):
            batch = entities[i : i + batch_size]
            batch_num += 1

            # Skip entities already in cache
            uncached = [
                e for e in batch
                if f"{e}|{relation}|{direction}" not in questions
            ]
            if not uncached:
                logger.info(
                    f"  Batch {batch_num}/{total_batches}: "
                    f"{relation}/{direction} -- all cached, skipping"
                )
                continue

            # Rate-limit: wait between remote API calls (not needed for local LLM)
            if batches_called > 0 and not _is_local_llm():
                time.sleep(12)

            logger.info(
                f"  Batch {batch_num}/{total_batches}: "
                f"{relation}/{direction} ({len(uncached)} entities)"
            )

            generated = _generate_questions_batch(uncached, relation, direction)
            batches_called += 1

            for entity in uncached:
                key = f"{entity}|{relation}|{direction}"
                if entity in generated:
                    questions[key] = generated[entity]
                else:
                    fallback = FALLBACK_TEMPLATES[relation][direction]
                    questions[key] = fallback.format(entity=entity)
                    logger.warning(
                        f"  Fallback for '{entity}' ({relation}/{direction})"
                    )

            # Incremental save after each batch
            if cache_path:
                with open(cache_path, "w") as f:
                    json.dump(questions, f, indent=2)

    if cache_path:
        logger.info(f"Saved {len(questions)} questions to cache: {cache_path}")

    return questions


# ---------------------------------------------------------------------------
# Build QA items (same output format as before)
# ---------------------------------------------------------------------------

def _build_questions(
    df: pd.DataFrame,
    max_per_relation: int,
    rng: random.Random,
    batch_size: int,
    cache_path: str | None,
    use_cache: bool,
) -> list[dict]:
    """Build QA items using LLM-generated questions."""

    relations = sorted(RELATION_CONTEXT.keys())

    # First pass: collect the entities we need questions for
    entity_lists: dict[tuple[str, str], list[str]] = {}
    grouped_data: dict[str, dict] = {}

    for relation in relations:
        rel_df = df[df["display_relation"] == relation]
        if rel_df.empty:
            logger.warning(f"No test rows for relation '{relation}', skipping")
            continue

        forward_groups = (
            rel_df.groupby("x_name")["y_name"]
            .apply(lambda s: sorted(s.unique().tolist()))
            .to_dict()
        )
        fwd_keys = list(forward_groups.keys())
        rng.shuffle(fwd_keys)
        fwd_keys = fwd_keys[:max_per_relation]
        entity_lists[(relation, "forward")] = fwd_keys

        reverse_groups = (
            rel_df.groupby("y_name")["x_name"]
            .apply(lambda s: sorted(s.unique().tolist()))
            .to_dict()
        )
        rev_keys = list(reverse_groups.keys())
        rng.shuffle(rev_keys)
        rev_keys = rev_keys[:max_per_relation]
        entity_lists[(relation, "reverse")] = rev_keys

        grouped_data[relation] = {
            "forward_groups": forward_groups,
            "forward_keys": fwd_keys,
            "reverse_groups": reverse_groups,
            "reverse_keys": rev_keys,
        }

    # Second pass: generate all questions via LLM
    total_entities = sum(len(v) for v in entity_lists.values())
    logger.info(
        f"Generating questions for {total_entities} entities "
        f"across {len(entity_lists)} (relation, direction) combos ..."
    )
    question_map = _generate_all_questions(
        entity_lists, batch_size, cache_path, use_cache
    )

    # Third pass: assemble QA items
    questions: list[dict] = []
    qid = 0

    for relation in relations:
        if relation not in grouped_data:
            continue
        data = grouped_data[relation]

        for drug in data["forward_keys"]:
            diseases = data["forward_groups"][drug]
            key = f"{drug}|{relation}|forward"
            q_text = question_map.get(
                key, FALLBACK_TEMPLATES[relation]["forward"].format(entity=drug)
            )
            questions.append({
                "id": f"q{qid:04d}",
                "question": q_text,
                "gold_answers": diseases,
                "relation": relation,
                "direction": "forward",
                "source_triplets": [
                    f"({drug}, {relation}, {d})" for d in diseases
                ],
            })
            qid += 1

        for disease in data["reverse_keys"]:
            drugs = data["reverse_groups"][disease]
            key = f"{disease}|{relation}|reverse"
            q_text = question_map.get(
                key, FALLBACK_TEMPLATES[relation]["reverse"].format(entity=disease)
            )
            questions.append({
                "id": f"q{qid:04d}",
                "question": q_text,
                "gold_answers": drugs,
                "relation": relation,
                "direction": "reverse",
                "source_triplets": [
                    f"({dr}, {relation}, {disease})" for dr in drugs
                ],
            })
            qid += 1

    return questions


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_qa(
    test_csv: str,
    output_path: str,
    max_per_relation: int = 100,
    seed: int = 42,
    batch_size: int = 15,
    cache_path: str | None = None,
    use_cache: bool = True,
) -> list[dict]:
    """Read test CSV, generate LLM questions, build QA, write JSON."""
    rng = random.Random(seed)

    df = pd.read_csv(test_csv, low_memory=False)
    logger.info(f"Loaded {len(df):,} test rows from {test_csv}")

    if cache_path is None:
        cache_path = os.path.join(
            os.path.dirname(output_path), "qa_questions_cache.json"
        )

    questions = _build_questions(
        df, max_per_relation, rng, batch_size, cache_path, use_cache
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(questions, f, indent=2)

    logger.info(f"Generated {len(questions)} QA pairs -> {output_path}")
    return questions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate QA pairs from PrimeKG drug-disease test split (LLM-powered)",
    )
    parser.add_argument(
        "--test-csv",
        default="data/eval/test.csv",
        help="Path to the test split CSV (default: data/eval/test.csv)",
    )
    parser.add_argument(
        "--output",
        default="data/eval/qa_dataset.json",
        help="Output JSON path (default: data/eval/qa_dataset.json)",
    )
    parser.add_argument(
        "--max-questions-per-relation",
        type=int,
        default=100,
        help="Max forward + max reverse questions per relation (default: 100 each)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=15,
        help="Entities per LLM API call (default: 15)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force regeneration, ignoring any existing cache",
    )
    parser.add_argument(
        "--remote-llm",
        action="store_true",
        help="Force remote LLM (e.g. Gemini) even when USE_LOCAL_LLM is set. "
             "Requires: source export_google_ai.sh",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    _check_llm_env(remote=args.remote_llm)

    if not os.path.exists(args.test_csv):
        logger.error(
            f"Test CSV not found: {args.test_csv}\n"
            "Run:  python scripts/eval/split_primekb.py   first."
        )
        sys.exit(1)

    questions = generate_qa(
        test_csv=args.test_csv,
        output_path=args.output,
        max_per_relation=args.max_questions_per_relation,
        seed=args.seed,
        batch_size=args.batch_size,
        use_cache=not args.no_cache,
    )

    # --- Summary ---
    from collections import Counter
    by_rel = Counter(q["relation"] for q in questions)
    by_dir = Counter(q["direction"] for q in questions)

    print("\n" + "=" * 50)
    print("QA generation summary")
    print("=" * 50)
    print(f"  Total questions: {len(questions)}")
    for rel, cnt in sorted(by_rel.items()):
        print(f"  {rel:25s}  {cnt}")
    print(f"  Forward: {by_dir['forward']}  Reverse: {by_dir['reverse']}")
    avg_answers = (
        sum(len(q["gold_answers"]) for q in questions) / len(questions)
        if questions
        else 0
    )
    print(f"  Avg gold answers per question: {avg_answers:.1f}")
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
