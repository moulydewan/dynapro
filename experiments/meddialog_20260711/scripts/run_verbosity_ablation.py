#!/usr/bin/env python3
"""Build deterministic original/repeat/padded conversations for judge testing."""

import argparse
import copy
import hashlib
import json
import random
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.judge_common import load_records, write_json_atomic  # noqa: E402


DEFAULT_SUMMARY = (
    ROOT
    / "output/judgments_score_band_ablation/"
    "five_methods_banded_v4flash_thinking_high_000_099_20260712/"
    "four_eval_summary.json"
)
METHODS = ("dynapro", "dynapro_medical", "proact", "generic_proact", "none")
VARIANTS = ("original", "original_repeat", "padded")
SEED = 20260712
MIN_ORIGINAL_WORDS = 80
PADDING_TARGET_RATIO = 1.45
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[’'_-][A-Za-z0-9]+)*")
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])\d[\d,.:/%+-]*(?![A-Za-z0-9])")
URL_RE = re.compile(r"https?://[^\s)>\]}]+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!])\s+(?=[A-Z0-9*#])")
PADDING_HEADER = "To restate only the same information:"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a deterministic repetition-only verbosity-bias ablation."
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def word_count(text):
    return len(WORD_RE.findall(text))


def assistant_messages(conversation):
    return [turn["content"] for turn in conversation if turn.get("role") == "assistant"]


def assistant_word_count(conversation):
    return sum(word_count(text) for text in assistant_messages(conversation))


def substantive_numeric_expressions(text):
    without_list_numbers = re.sub(
        r"(?m)^(\s*(?:(?:[-*]|#{1,6})\s*)?)\d+[.)]\s+",
        r"\1",
        text,
    )
    return set(NUMBER_RE.findall(without_list_numbers))


def load_score_lookup(summary):
    return {
        (row["method"], row["source_index"]): row["total_score"]
        for row in summary["per_case"]
        if row.get("status") == "ok"
    }


def select_cases(summary):
    rng = random.Random(SEED)
    score_lookup = load_score_lookup(summary)
    used_source_indices = set()
    selected = []
    for method in METHODS:
        records = load_records(Path(summary["inputs"][method]), method)
        eligible = []
        for source_index, record in records.items():
            words = assistant_word_count(record["conversation"])
            if words < MIN_ORIGINAL_WORDS:
                continue
            eligible.append(
                {
                    "source_index": source_index,
                    "record": record,
                    "words": words,
                    "prior_total_score": score_lookup[(method, source_index)],
                }
            )
        eligible.sort(key=lambda item: (item["words"], item["source_index"]))
        for rank, item in enumerate(eligible):
            item["length_tertile"] = min(2, (rank * 3) // len(eligible))
        median_score = statistics.median(item["prior_total_score"] for item in eligible)
        for item in eligible:
            item["score_stratum"] = (
                "low" if item["prior_total_score"] <= median_score else "high"
            )
        for length_tertile in range(3):
            for score_stratum in ("low", "high"):
                candidates = [
                    item
                    for item in eligible
                    if item["length_tertile"] == length_tertile
                    and item["score_stratum"] == score_stratum
                ]
                if not candidates:
                    raise RuntimeError(
                        f"No {method} candidate for length={length_tertile}, "
                        f"score={score_stratum}"
                    )
                rng.shuffle(candidates)
                unique = [
                    item
                    for item in candidates
                    if item["source_index"] not in used_source_indices
                ]
                chosen = (unique or candidates)[0]
                used_source_indices.add(chosen["source_index"])
                selected.append(
                    {
                        "ablation_index": len(selected),
                        "source_method": method,
                        "source_index": chosen["source_index"],
                        "length_tertile": length_tertile,
                        "score_stratum": score_stratum,
                        "prior_total_score": chosen["prior_total_score"],
                        "original_words": chosen["words"],
                        "record": chosen["record"],
                    }
                )
    if len(selected) != 30 or len(used_source_indices) != 30:
        raise RuntimeError(
            f"Expected 30 cases with distinct source indices; got "
            f"cases={len(selected)}, unique={len(used_source_indices)}"
        )
    return selected


def repeatable_units(text):
    units = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        for part in SENTENCE_SPLIT_RE.split(stripped):
            unit = part.strip()
            if word_count(unit) < 5 or "?" in unit:
                continue
            if re.fullmatch(r"[#*\s_-]+", unit):
                continue
            units.append(unit)
    return units


def build_padded_conversation(original_conversation):
    original_messages = assistant_messages(original_conversation)
    original_words = sum(word_count(text) for text in original_messages)
    target_words = round(original_words * PADDING_TARGET_RATIO)
    units_by_turn = [repeatable_units(text) for text in original_messages]
    candidate_units = [
        (turn_index, unit)
        for offset in range(max((len(units) for units in units_by_turn), default=0))
        for turn_index, units in enumerate(units_by_turn)
        if offset < len(units)
        for unit in (units[offset],)
    ]
    if not candidate_units:
        raise ValueError("Conversation has no non-question declarative unit to repeat")
    additions = [[] for _ in original_messages]
    padded_words = original_words
    position = 0
    while padded_words < target_words:
        turn_index, unit = candidate_units[position % len(candidate_units)]
        additions[turn_index].append(unit)
        padded_words += word_count(unit)
        if len(additions[turn_index]) == 1:
            padded_words += word_count(PADDING_HEADER)
        position += 1
        if position > 10000:
            raise RuntimeError("Padding loop did not converge")

    rewritten_messages = []
    audit = []
    for turn_index, (original, repeated) in enumerate(
        zip(original_messages, additions), 1
    ):
        if repeated:
            appended = "\n".join(f"- {unit}" for unit in repeated)
            padded = f"{original}\n\n{PADDING_HEADER}\n{appended}"
        else:
            padded = original
        if not padded.startswith(original):
            raise AssertionError("Padded message does not preserve the original prefix")
        if any(unit not in original or "?" in unit for unit in repeated):
            raise AssertionError("Padding contains a non-original or question-bearing unit")
        if original.count("?") != padded.count("?"):
            raise AssertionError("Padding changed question-mark count")
        if substantive_numeric_expressions(original) != substantive_numeric_expressions(padded):
            raise AssertionError("Padding introduced a new numeric expression")
        if set(URL_RE.findall(original)) != set(URL_RE.findall(padded)):
            raise AssertionError("Padding introduced a new URL")
        rewritten_messages.append(padded)
        audit.append(
            {
                "assistant_turn": turn_index,
                "repeated_units": [
                    {
                        "text": unit,
                        "word_count": word_count(unit),
                        "sha256": hashlib.sha256(unit.encode()).hexdigest(),
                    }
                    for unit in repeated
                ],
            }
        )

    result = []
    assistant_index = 0
    for turn in original_conversation:
        copied = copy.deepcopy(turn)
        if copied["role"] == "assistant":
            copied["content"] = rewritten_messages[assistant_index]
            assistant_index += 1
        result.append(copied)
    final_words = assistant_word_count(result)
    if final_words <= original_words:
        raise AssertionError("Padded conversation is not longer")
    return result, {
        "original_words": original_words,
        "padded_words": final_words,
        "padded_ratio": final_words / original_words,
        "target_ratio": PADDING_TARGET_RATIO,
        "turn_audit": audit,
    }


def make_condition_record(case, condition, conversation):
    record = copy.deepcopy(case["record"])
    record.update(
        {
            "conv_id": case["ablation_index"] + 1,
            "source_index": case["ablation_index"],
            "method": condition,
            "experiment": "verbosity_bias_ablation",
            "source_method": case["source_method"],
            "source_source_index": case["source_index"],
            "conversation": copy.deepcopy(conversation),
        }
    )
    return record


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    selected = select_cases(summary)
    condition_names = [f"condition_{letter}" for letter in "abc"]
    random.Random(SEED + 1).shuffle(condition_names)
    condition_by_variant = dict(zip(VARIANTS, condition_names))
    records_by_condition = {condition: [] for condition in condition_names}
    manifest_cases = []

    for case in selected:
        original = case["record"]["conversation"]
        padded, padding_audit = build_padded_conversation(original)
        conversations = {
            "original": original,
            "original_repeat": original,
            "padded": padded,
        }
        for variant, conversation in conversations.items():
            condition = condition_by_variant[variant]
            records_by_condition[condition].append(
                make_condition_record(case, condition, conversation)
            )
        manifest_cases.append(
            {
                **{key: value for key, value in case.items() if key != "record"},
                "padding": padding_audit,
            }
        )

    condition_files = {}
    for condition, records in records_by_condition.items():
        path = args.output_dir / f"{condition}.json"
        write_json_atomic(path, records)
        condition_files[condition] = str(path.resolve())
    write_json_atomic(
        args.output_dir / "selected_cases.json",
        [
            {key: value for key, value in case.items() if key != "record"}
            for case in selected
        ],
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": str(
            (
                ROOT
                / "experiments/meddialog_20260711/protocols/"
                "VERBOSITY_ABLATION_PROTOCOL.md"
            ).resolve()
        ),
        "source_summary": str(args.summary.resolve()),
        "sampling_seed": SEED,
        "padding_target_ratio": PADDING_TARGET_RATIO,
        "padding_header": PADDING_HEADER,
        "condition_by_variant": condition_by_variant,
        "condition_files": condition_files,
        "counts": {
            "base_conversations": len(selected),
            "conditions": len(VARIANTS),
            "judge_conversations": len(selected) * len(VARIANTS),
        },
        "cases": manifest_cases,
    }
    write_json_atomic(args.output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "base_conversations": len(selected),
                "judge_conversations": len(selected) * len(VARIANTS),
                "condition_by_variant": condition_by_variant,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
