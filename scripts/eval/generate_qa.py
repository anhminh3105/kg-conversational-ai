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

PARTIAL_RELATION_CONTEXT: dict[str, dict[str, str]] = {
    "indication": {
        "forward": (
            "The relation is 'indication'. Generate a COMPREHENSIVE question "
            "asking for ALL diseases or conditions the given DRUG is indicated "
            "for. The question MUST explicitly demand a complete or exhaustive "
            "list. Use phrasing like 'list all', 'comprehensive list of every', "
            "'what are all the', etc. The question must mention the drug name."
        ),
        "reverse": (
            "The relation is 'indication'. Generate a COMPREHENSIVE question "
            "asking for ALL drugs that are indicated for the given DISEASE. "
            "The question MUST explicitly demand a complete list. Use phrasing "
            "like 'list every drug', 'all medications approved for', etc. "
            "The question must mention the disease name."
        ),
    },
    "contraindication": {
        "forward": (
            "The relation is 'contraindication'. Generate a COMPREHENSIVE "
            "question asking for ALL diseases or conditions that make the "
            "given DRUG unsafe or contraindicated. The question MUST demand "
            "a complete/exhaustive list. Use phrasing like 'list all', "
            "'every contraindication', etc. The question must mention the "
            "drug name."
        ),
        "reverse": (
            "The relation is 'contraindication'. Generate a COMPREHENSIVE "
            "question asking for ALL drugs that are contraindicated for the "
            "given DISEASE. The question MUST demand a complete list. "
            "The question must mention the disease name."
        ),
    },
    "off-label use": {
        "forward": (
            "The relation is 'off-label use'. Generate a COMPREHENSIVE "
            "question asking for ALL off-label uses the given DRUG has. "
            "The question MUST demand a complete or exhaustive list. "
            "The question must mention the drug name."
        ),
        "reverse": (
            "The relation is 'off-label use'. Generate a COMPREHENSIVE "
            "question asking for ALL drugs that are used off-label for the "
            "given DISEASE. The question MUST demand a complete list. "
            "The question must mention the disease name."
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

PARTIAL_FALLBACK_TEMPLATES: dict[str, dict[str, str]] = {
    "indication": {
        "forward": "Provide a comprehensive list of ALL diseases and conditions that {entity} is indicated for.",
        "reverse": "List every drug that is indicated for treating {entity}.",
    },
    "contraindication": {
        "forward": "List ALL diseases and conditions for which {entity} is contraindicated.",
        "reverse": "List every drug that is contraindicated for {entity}.",
    },
    "off-label use": {
        "forward": "Provide a comprehensive list of ALL known off-label uses of {entity}.",
        "reverse": "List every drug that is used off-label for {entity}.",
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
    context_dict: dict[str, dict[str, str]] | None = None,
) -> dict[str, str]:
    """Call the LLM to generate one question per entity. Returns {entity: question}."""
    from rag.edc.edc.utils.llm_utils import openai_chat_completion

    ctx = context_dict or RELATION_CONTEXT
    context = ctx[relation][direction]
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


# ---------------------------------------------------------------------------
# Build QA records (single-pass: generate questions + assemble records)
# ---------------------------------------------------------------------------

def _record_key(rec: dict) -> str:
    """Derive a unique resume key from a QA record."""
    return f"{rec['entity']}|{rec['relation']}|{rec['direction']}"


def _build_record(
    entity: str,
    question: str,
    gold_answers: list[str],
    relation: str,
    direction: str,
    tier: str = "partial",
    kg_answers: list[str] | None = None,
    held_out_answers: list[str] | None = None,
) -> dict:
    """Build a single QA record dict (ID is assigned later)."""
    if direction == "forward":
        triplets = [f"({entity}, {relation}, {a})" for a in gold_answers]
    else:
        triplets = [f"({a}, {relation}, {entity})" for a in gold_answers]
    return {
        "id": "",
        "entity": entity,
        "question": question,
        "gold_answers": gold_answers,
        "relation": relation,
        "direction": direction,
        "tier": tier,
        "kg_answers": kg_answers if kg_answers is not None else [],
        "held_out_answers": held_out_answers if held_out_answers is not None else [],
        "source_triplets": triplets,
    }


def _save_records(records: list[dict], path: str) -> None:
    """Write the full records list to disk (atomic-ish via write+flush)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)


def _load_tier_assignments(output_path: str) -> dict[str, dict[str, dict]]:
    """Load tier_assignments.json from the same directory as the output.

    Returns {entity: {relation: {"tier": str, "kg_answers": list}}} or
    legacy format {entity: {relation: str}} (auto-detected).
    """
    tier_path = os.path.join(os.path.dirname(output_path) or ".", "tier_assignments.json")
    if os.path.exists(tier_path):
        with open(tier_path) as f:
            return json.load(f)
    return {}


def _get_tier_info(
    tier_assignments: dict,
    entity: str,
    relation: str,
) -> tuple[str, list[str], list[str]]:
    """Extract tier, kg_answers, and held_out_answers for an entity+relation.

    Handles both old format (value is str) and new format (value is dict).
    Returns (tier, kg_answers, held_out_answers).
    """
    entry = tier_assignments.get(entity, {}).get(relation, {})
    if isinstance(entry, str):
        return entry, [], []
    if isinstance(entry, dict):
        return (
            entry.get("tier", "partial"),
            entry.get("kg_answers", []),
            entry.get("held_out_answers", []),
        )
    return "partial", [], []


def _build_questions(
    df: pd.DataFrame,
    max_per_relation: int,
    rng: random.Random,
    batch_size: int,
    output_path: str,
    use_cache: bool,
) -> list[dict]:
    """Generate LLM questions and assemble complete QA records in one pass.

    The output file doubles as an incremental cache: if *use_cache* is True
    and *output_path* already exists, previously generated records are loaded
    and their entities are skipped.
    """
    tier_assignments = _load_tier_assignments(output_path)
    relations = sorted(RELATION_CONTEXT.keys())

    # -- Collect entity lists and gold-answer groups from the test CSV ------
    entity_lists: dict[tuple[str, str], list[str]] = {}
    answer_groups: dict[tuple[str, str], dict[str, list[str]]] = {}

    for relation in relations:
        rel_df = df[df["display_relation"] == relation]
        if rel_df.empty:
            logger.warning(f"No test rows for relation '{relation}', skipping")
            continue

        fwd = (
            rel_df.groupby("x_name")["y_name"]
            .apply(lambda s: sorted(s.unique().tolist()))
            .to_dict()
        )
        fwd_keys = list(fwd.keys())
        rng.shuffle(fwd_keys)
        fwd_keys = fwd_keys[:max_per_relation]
        entity_lists[(relation, "forward")] = fwd_keys
        answer_groups[(relation, "forward")] = fwd

        rev = (
            rel_df.groupby("y_name")["x_name"]
            .apply(lambda s: sorted(s.unique().tolist()))
            .to_dict()
        )
        rev_keys = list(rev.keys())
        rng.shuffle(rev_keys)
        rev_keys = rev_keys[:max_per_relation]
        entity_lists[(relation, "reverse")] = rev_keys
        answer_groups[(relation, "reverse")] = rev

    # -- Resume: load existing records if available -------------------------
    records: list[dict] = []
    done_keys: set[str] = set()

    if use_cache and os.path.exists(output_path):
        try:
            with open(output_path) as f:
                records = json.load(f)
            done_keys = {_record_key(r) for r in records}
            logger.info(
                f"Resumed {len(records)} existing records from {output_path}"
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning(f"Could not resume from {output_path}: {exc}")
            records, done_keys = [], set()

    # -- Pre-compute per-entity tier for routing prompts --------------------
    def _entity_tier_for_prompt(entity: str, relation: str, direction: str) -> str:
        """Determine which prompt set to use for an entity."""
        if direction == "forward":
            tier, _, _ = _get_tier_info(tier_assignments, entity, relation)
            return tier
        # Reverse: aggregate from contributing drugs
        drug_list = answer_groups.get((relation, direction), {}).get(entity, [])
        tiers = set()
        for drug in drug_list:
            t, _, _ = _get_tier_info(tier_assignments, drug, relation)
            tiers.add(t)
        if "partial" in tiers:
            return "partial"
        return "full"

    # -- Generate + build records in one pass -------------------------------
    total_entities = sum(len(v) for v in entity_lists.values())
    total_batches = sum(
        (len(ents) + batch_size - 1) // batch_size
        for ents in entity_lists.values()
    )
    logger.info(
        f"Generating questions for {total_entities} entities "
        f"across {len(entity_lists)} (relation, direction) combos ..."
    )

    batch_num = 0
    batches_called = 0

    for (relation, direction), entities in entity_lists.items():
        golds = answer_groups[(relation, direction)]

        for i in range(0, len(entities), batch_size):
            batch = entities[i : i + batch_size]
            batch_num += 1

            uncached = [
                e for e in batch
                if f"{e}|{relation}|{direction}" not in done_keys
            ]
            if not uncached:
                logger.info(
                    f"  Batch {batch_num}/{total_batches}: "
                    f"{relation}/{direction} -- all cached, skipping"
                )
                continue

            if batches_called > 0 and not _is_local_llm():
                time.sleep(12)

            # Split entities by tier so each batch uses the right prompt
            full_ents = [e for e in uncached
                         if _entity_tier_for_prompt(e, relation, direction) != "partial"]
            partial_ents = [e for e in uncached
                            if _entity_tier_for_prompt(e, relation, direction) == "partial"]

            logger.info(
                f"  Batch {batch_num}/{total_batches}: "
                f"{relation}/{direction} ({len(full_ents)} full, {len(partial_ents)} partial)"
            )

            generated: dict[str, str] = {}
            if full_ents:
                generated.update(
                    _generate_questions_batch(full_ents, relation, direction,
                                              context_dict=RELATION_CONTEXT)
                )
                batches_called += 1
                if partial_ents and not _is_local_llm():
                    time.sleep(12)
            if partial_ents:
                generated.update(
                    _generate_questions_batch(partial_ents, relation, direction,
                                              context_dict=PARTIAL_RELATION_CONTEXT)
                )
                batches_called += 1

            for entity in uncached:
                entity_prompt_tier = _entity_tier_for_prompt(entity, relation, direction)
                fb = (PARTIAL_FALLBACK_TEMPLATES if entity_prompt_tier == "partial"
                      else FALLBACK_TEMPLATES)

                if entity in generated:
                    q_text = generated[entity]
                else:
                    q_text = fb[relation][direction].format(entity=entity)
                    logger.warning(
                        f"  Fallback for '{entity}' ({relation}/{direction})"
                    )

                if direction == "forward":
                    entity_tier, entity_kg, entity_held = _get_tier_info(
                        tier_assignments, entity, relation
                    )
                else:
                    entity_kg = []
                    entity_held = []
                    tier_labels = set()
                    for drug in golds[entity]:
                        drug_tier, drug_kg, drug_held = _get_tier_info(
                            tier_assignments, drug, relation
                        )
                        tier_labels.add(drug_tier)
                        if entity in drug_kg:
                            entity_kg.append(drug)
                        if entity in drug_held:
                            entity_held.append(drug)
                    if "partial" in tier_labels:
                        entity_tier = "partial"
                    elif "full" in tier_labels:
                        entity_tier = "full"
                    else:
                        entity_tier = "partial"

                records.append(_build_record(
                    entity=entity,
                    question=q_text,
                    gold_answers=golds[entity],
                    relation=relation,
                    direction=direction,
                    tier=entity_tier,
                    kg_answers=entity_kg,
                    held_out_answers=entity_held,
                ))
                done_keys.add(f"{entity}|{relation}|{direction}")

            _save_records(records, output_path)

    # -- Re-number IDs so they are contiguous -------------------------------
    for idx, rec in enumerate(records):
        rec["id"] = f"q{idx:04d}"
    _save_records(records, output_path)

    logger.info(f"Generated {len(records)} QA pairs -> {output_path}")
    return records


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_qa(
    test_csv: str,
    output_path: str,
    max_per_relation: int = 100,
    seed: int = 42,
    batch_size: int = 15,
    use_cache: bool = True,
) -> list[dict]:
    """Read test CSV, generate LLM questions, build QA, write JSON.

    The *output_path* file serves as both the final dataset and an incremental
    cache.  Pass *use_cache=False* to regenerate from scratch.
    """
    rng = random.Random(seed)

    df = pd.read_csv(test_csv, low_memory=False)
    logger.info(f"Loaded {len(df):,} test rows from {test_csv}")

    return _build_questions(
        df, max_per_relation, rng, batch_size, output_path, use_cache
    )


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
        help="Force regeneration, ignoring any existing output file",
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
        use_cache=(not args.no_cache),
    )

    # --- Summary ---
    from collections import Counter
    by_rel = Counter(q["relation"] for q in questions)
    by_dir = Counter(q["direction"] for q in questions)
    by_tier = Counter(q["tier"] for q in questions)

    print("\n" + "=" * 50)
    print("QA generation summary")
    print("=" * 50)
    print(f"  Total questions: {len(questions)}")
    for rel, cnt in sorted(by_rel.items()):
        print(f"  {rel:25s}  {cnt}")
    print(f"  Forward: {by_dir['forward']}  Reverse: {by_dir['reverse']}")
    for tier, cnt in sorted(by_tier.items()):
        print(f"  Tier {tier:10s}: {cnt}")
    avg_answers = (
        sum(len(q["gold_answers"]) for q in questions) / len(questions)
        if questions
        else 0
    )
    avg_held = (
        sum(len(q.get("held_out_answers", [])) for q in questions) / len(questions)
        if questions
        else 0
    )
    print(f"  Avg gold answers per question: {avg_answers:.1f}")
    print(f"  Avg held-out answers per question: {avg_held:.1f}")
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
