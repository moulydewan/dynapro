#!/usr/bin/env python3
"""Legacy blinded pairwise A/D/C judge retained for reproducibility."""

import argparse
import hashlib
import json
import logging
import os
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.judge_common import (  # noqa: E402
    DEFAULT_JUDGE_TEMPERATURE,
    PROJECT_ROOT,
    load_env,
    require_nonempty_string,
    sha256_json,
    write_json_atomic,
)
from experiments.meddialog_20260711.legacy_judges.adc_common import (  # noqa: E402
    DIMENSIONS,
    load_pairs,
    render_conversation,
    validate_candidate,
)
from simulation.modules.deepseek import (  # noqa: E402
    chat_completion,
    parse_json_object,
)


PROMPT_PATH = (
    Path(__file__).resolve().parent
    / "prompts"
    / "dynapro_adc_pairwise_judge.txt"
)
PROMPT_VERSION = "dynapro-adc-pairwise-v1"
SCHEMA_VERSION = "dynapro-adc-schema-v1"
JUDGE_TEMPERATURE = DEFAULT_JUDGE_TEMPERATURE


def build_prompt(template, hidden_target, conversation_a, conversation_b):
    rendered_a, assistant_turns_a = render_conversation(conversation_a)
    rendered_b, assistant_turns_b = render_conversation(conversation_b)
    prompt = template.replace("__HIDDEN_TARGET__", hidden_target)
    prompt = prompt.replace(
        "__CONVERSATION_A__",
        json.dumps(rendered_a, ensure_ascii=False, indent=2),
    )
    prompt = prompt.replace(
        "__CONVERSATION_B__",
        json.dumps(rendered_b, ensure_ascii=False, indent=2),
    )
    return prompt, assistant_turns_a, assistant_turns_b


def validate_result(result, assistant_turns_a, assistant_turns_b):
    if not isinstance(result, dict):
        raise ValueError("Judge response must be a JSON object")
    validate_candidate(
        result.get("conversation_a"), "conversation_a", assistant_turns_a
    )
    validate_candidate(
        result.get("conversation_b"), "conversation_b", assistant_turns_b
    )

    comparison = result.get("comparison")
    if not isinstance(comparison, dict):
        raise ValueError("comparison must be an object")
    winner = comparison.get("winner")
    if winner not in {"A", "B", "tie"}:
        raise ValueError("comparison.winner must be A, B, or tie")
    require_nonempty_string(comparison.get("reason"), "comparison.reason")

    confidence = result.get("judge_confidence")
    if confidence not in {"low", "medium", "high"}:
        raise ValueError("judge_confidence must be low, medium, or high")

    candidate_a = result["conversation_a"]
    candidate_b = result["conversation_b"]
    a_major = candidate_a["guardrails"]["major_medical_safety_issue"]
    b_major = candidate_b["guardrails"]["major_medical_safety_issue"]
    a_total = candidate_a["total_score"]
    b_total = candidate_b["total_score"]
    if a_major and b_major:
        computed_winner = "tie"
        winner_basis = "both conversations have a major medical safety issue"
        computed_reason = "Both conversations have a major medical safety issue, so the result requires review."
    elif a_major != b_major:
        computed_winner = "B" if a_major else "A"
        winner_basis = "major medical safety gate"
        unsafe_label = "A" if a_major else "B"
        computed_reason = (
            f"{computed_winner} wins because Conversation {unsafe_label} has a major "
            "medical safety issue."
        )
    elif a_total > b_total:
        computed_winner = "A"
        winner_basis = "higher A/D/C total score"
        computed_reason = (
            f"A has the higher verified A/D/C total score ({a_total} vs {b_total})."
        )
    elif b_total > a_total:
        computed_winner = "B"
        winner_basis = "higher A/D/C total score"
        computed_reason = (
            f"B has the higher verified A/D/C total score ({b_total} vs {a_total})."
        )
    else:
        computed_winner = "tie"
        winner_basis = "equal A/D/C total score"
        computed_reason = (
            f"Both conversations have the same verified A/D/C total score ({a_total})."
        )
    if winner != computed_winner:
        comparison["reported_winner"] = winner
        comparison["reported_reason"] = comparison["reason"]
        comparison["winner"] = computed_winner
        comparison["reason"] = computed_reason
    comparison["winner_basis"] = winner_basis
    return result


def make_task(
    source_index,
    hidden_target,
    dynapro_record,
    medical_record,
    order,
    prompt_template,
    prompt_hash,
    model,
    thinking,
    reasoning_effort,
    max_tokens,
):
    if order == "dynapro_as_a":
        label_map = {"A": "dynapro", "B": "dynapro_medical"}
        record_a, record_b = dynapro_record, medical_record
    else:
        label_map = {"A": "dynapro_medical", "B": "dynapro"}
        record_a, record_b = medical_record, dynapro_record

    prompt, assistant_turns_a, assistant_turns_b = build_prompt(
        prompt_template,
        hidden_target,
        record_a["conversation"],
        record_b["conversation"],
    )
    conversation_hashes = {
        "dynapro": sha256_json(dynapro_record["conversation"]),
        "dynapro_medical": sha256_json(medical_record["conversation"]),
    }
    cache_key = sha256_json(
        {
            "prompt_hash": prompt_hash,
            "schema_version": SCHEMA_VERSION,
            "model": model,
            "temperature": JUDGE_TEMPERATURE,
            "thinking": thinking,
            "reasoning_effort": reasoning_effort,
            "max_tokens": max_tokens,
            "source_index": source_index,
            "order": order,
            "hidden_target": hidden_target,
            "conversation_hashes": conversation_hashes,
        }
    )
    return {
        "source_index": source_index,
        "order": order,
        "label_map": label_map,
        "prompt": prompt,
        "assistant_turns_a": assistant_turns_a,
        "assistant_turns_b": assistant_turns_b,
        "conversation_hashes": conversation_hashes,
        "cache_key": cache_key,
    }


def run_task(
    task, model, thinking, reasoning_effort, max_tokens, schema_retries, prompt_hash
):
    raw_response = None
    last_error = None
    for attempt in range(1, schema_retries + 1):
        try:
            raw_response = chat_completion(
                [{"role": "user", "content": task["prompt"]}],
                model=model,
                temperature=JUDGE_TEMPERATURE,
                max_tokens=max_tokens,
                json_output=True,
                thinking=thinking,
                reasoning_effort=reasoning_effort,
            )
            parsed = parse_json_object(raw_response)
            parsed = validate_result(
                parsed,
                task["assistant_turns_a"],
                task["assistant_turns_b"],
            )
            return {
                "source_index": task["source_index"],
                "order": task["order"],
                "label_map": task["label_map"],
                "judge_model": model,
                "judge_temperature": JUDGE_TEMPERATURE,
                "judge_thinking": thinking,
                "prompt_version": PROMPT_VERSION,
                "schema_version": SCHEMA_VERSION,
                "prompt_sha256": prompt_hash,
                "conversation_hashes": task["conversation_hashes"],
                "cache_key": task["cache_key"],
                "status": "ok",
                "attempts": attempt,
                "raw_response": raw_response,
                "parsed_result": parsed,
                "error": None,
            }
        except Exception as error:
            last_error = repr(error)
            logging.warning(
                "Judge failed source_index=%s order=%s attempt=%s/%s: %s",
                task["source_index"],
                task["order"],
                attempt,
                schema_retries,
                error,
            )
    return {
        "source_index": task["source_index"],
        "order": task["order"],
        "label_map": task["label_map"],
        "judge_model": model,
        "judge_temperature": JUDGE_TEMPERATURE,
        "judge_thinking": thinking,
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "prompt_sha256": prompt_hash,
        "conversation_hashes": task["conversation_hashes"],
        "cache_key": task["cache_key"],
        "status": "failed",
        "attempts": schema_retries,
        "raw_response": raw_response,
        "parsed_result": None,
        "error": last_error,
    }


def load_latest_rows(path):
    latest = {}
    if not path.exists():
        return latest
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            key = (row["source_index"], row["order"])
        except Exception as error:
            raise ValueError(
                f"Invalid JSONL at {path}:{line_number}: {error}"
            ) from error
        latest[key] = row
    return latest


def append_jsonl(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def mean(values):
    return round(statistics.mean(values), 3) if values else None


def median(values):
    return round(statistics.median(values), 3) if values else None


def boolean_votes(values):
    true_votes = sum(values)
    false_votes = len(values) - true_votes
    consensus = (
        True
        if true_votes == len(values)
        else False if false_votes == len(values) else None
    )
    return {
        "true_votes": true_votes,
        "false_votes": false_votes,
        "consensus": consensus,
    }


def mapped_winner(row):
    winner = row["parsed_result"]["comparison"]["winner"]
    return "tie" if winner == "tie" else row["label_map"][winner]


def build_summary(pairs, current_rows, dynapro_path, medical_path, settings):
    per_case = []
    method_cases = {"dynapro": [], "dynapro_medical": []}
    winner_counts = {
        "dynapro": 0,
        "dynapro_medical": 0,
        "tie": 0,
        "order_sensitive": 0,
        "incomplete": 0,
    }

    for source_index, _, _ in pairs:
        rows = [
            current_rows.get((source_index, "dynapro_as_a")),
            current_rows.get((source_index, "medical_as_a")),
        ]
        rows = [row for row in rows if row and row.get("status") == "ok"]
        case = {"source_index": source_index, "judge_calls_completed": len(rows)}
        if len(rows) != 2:
            case["winner"] = "incomplete"
            winner_counts["incomplete"] += 1
            per_case.append(case)
            continue

        mapped = {"dynapro": [], "dynapro_medical": []}
        for row in rows:
            result = row["parsed_result"]
            for label in ("A", "B"):
                method = row["label_map"][label]
                mapped[method].append(result[f"conversation_{label.lower()}"])

        case["methods"] = {}
        for method, judgments in mapped.items():
            dimension_scores = {
                dimension: mean(
                    [judgment["scores"][dimension]["score"] for judgment in judgments]
                )
                for dimension in DIMENSIONS
            }
            method_result = {
                "scores": dimension_scores,
                "total_score": round(sum(dimension_scores.values()), 3),
                "guardrail_votes": {
                    key: boolean_votes(
                        [judgment["guardrails"][key] for judgment in judgments]
                    )
                    for key in (
                        "explicit_request_completed",
                        "unnecessary_continuation",
                        "major_medical_safety_issue",
                    )
                },
            }
            case["methods"][method] = method_result
            method_cases[method].append(method_result)

        case["medical_minus_dynapro"] = {
            dimension: round(
                case["methods"]["dynapro_medical"]["scores"][dimension]
                - case["methods"]["dynapro"]["scores"][dimension],
                3,
            )
            for dimension in DIMENSIONS
        }
        case["medical_minus_dynapro"]["total_score"] = round(
            case["methods"]["dynapro_medical"]["total_score"]
            - case["methods"]["dynapro"]["total_score"],
            3,
        )

        winners = [mapped_winner(row) for row in rows]
        case["winner_by_order"] = {row["order"]: mapped_winner(row) for row in rows}
        if winners[0] == winners[1]:
            case["winner"] = winners[0]
        else:
            case["winner"] = "order_sensitive"
        winner_counts[case["winner"]] += 1
        per_case.append(case)

    aggregate = {}
    for method, cases in method_cases.items():
        aggregate[method] = {
            "n_complete_cases": len(cases),
            "scores": {
                dimension: {
                    "mean": mean([case["scores"][dimension] for case in cases]),
                    "median": median([case["scores"][dimension] for case in cases]),
                }
                for dimension in DIMENSIONS
            },
            "total_score": {
                "mean": mean([case["total_score"] for case in cases]),
                "median": median([case["total_score"] for case in cases]),
            },
            "guardrails": {},
        }
        for key in (
            "explicit_request_completed",
            "unnecessary_continuation",
            "major_medical_safety_issue",
        ):
            consensuses = [case["guardrail_votes"][key]["consensus"] for case in cases]
            known = [value for value in consensuses if value is not None]
            aggregate[method]["guardrails"][key] = {
                "consensus_true": sum(known),
                "consensus_false": len(known) - sum(known),
                "order_disagreements": sum(value is None for value in consensuses),
                "consensus_true_rate": (
                    round(sum(known) / len(known), 3) if known else None
                ),
            }

    complete_cases = [case for case in per_case if case.get("winner") != "incomplete"]
    paired_delta = {
        dimension: {
            "mean": mean(
                [case["medical_minus_dynapro"][dimension] for case in complete_cases]
            ),
            "median": median(
                [case["medical_minus_dynapro"][dimension] for case in complete_cases]
            ),
        }
        for dimension in (*DIMENSIONS, "total_score")
    }

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "dynapro": str(dynapro_path.resolve()),
            "dynapro_medical": str(medical_path.resolve()),
        },
        "settings": settings,
        "counts": {
            "paired_cases_requested": len(pairs),
            "paired_cases_complete": len(complete_cases),
            "judge_calls_expected": len(pairs) * 2,
            "judge_calls_successful": sum(
                row.get("status") == "ok" for row in current_rows.values()
            ),
        },
        "winner_counts": winner_counts,
        "aggregate": aggregate,
        "paired_delta_medical_minus_dynapro": paired_delta,
        "per_case": per_case,
        "caveat": "This paired sample is descriptive; do not claim statistical significance.",
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Blind pairwise A/D/C LLM judgment for DynaPro vs DynaPro Medical"
    )
    parser.add_argument(
        "--dynapro", type=Path, required=True, help="DynaPro result JSON"
    )
    parser.add_argument(
        "--dynapro-medical",
        type=Path,
        required=True,
        help="DynaPro Medical result JSON",
    )
    parser.add_argument("--output-dir", type=Path, help="Judgment output directory")
    parser.add_argument(
        "--model",
        default=os.environ.get("JUDGE_MODEL", "deepseek-v4-flash"),
        help="Judge model (default: deepseek-v4-flash)",
    )
    parser.add_argument(
        "--thinking",
        choices=("enabled", "disabled"),
        default=os.environ.get("JUDGE_THINKING", "disabled"),
        help="Judge thinking mode; explicit so generation settings are not inherited",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=os.environ.get("JUDGE_REASONING_EFFORT"),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("JUDGE_WORKERS", "10")),
    )
    parser.add_argument("--schema-retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--limit", type=int, help="Optional number of paired cases")
    return parser.parse_args()


def main():
    load_env()
    os.environ.setdefault("SSL_CERT_FILE", "/etc/ssl/cert.pem")
    args = parse_args()
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    if args.workers < 1 or args.schema_retries < 1 or args.max_tokens < 1:
        raise ValueError("workers, schema-retries, and max-tokens must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")

    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    prompt_hash = hashlib.sha256(prompt_template.encode("utf-8")).hexdigest()
    pairs = load_pairs(args.dynapro, args.dynapro_medical, args.limit)
    output_dir = args.output_dir or (
        PROJECT_ROOT
        / "output/judgments"
        / f"{args.dynapro.stem}__vs__{args.dynapro_medical.stem}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "judge_details.jsonl"
    summary_path = output_dir / "judge_summary.json"

    tasks = []
    for source_index, dynapro_record, medical_record in pairs:
        hidden_target = dynapro_record.get("single_turn_prompt", "")
        for order in ("dynapro_as_a", "medical_as_a"):
            tasks.append(
                make_task(
                    source_index,
                    hidden_target,
                    dynapro_record,
                    medical_record,
                    order,
                    prompt_template,
                    prompt_hash,
                    args.model,
                    args.thinking,
                    args.reasoning_effort,
                    args.max_tokens,
                )
            )

    latest_rows = load_latest_rows(details_path)
    current_rows = {}
    pending = []
    for task in tasks:
        key = (task["source_index"], task["order"])
        existing = latest_rows.get(key)
        if (
            existing
            and existing.get("status") == "ok"
            and existing.get("cache_key") == task["cache_key"]
        ):
            try:
                validate_result(
                    existing.get("parsed_result"),
                    task["assistant_turns_a"],
                    task["assistant_turns_b"],
                )
            except Exception:
                pending.append(task)
            else:
                current_rows[key] = existing
        else:
            pending.append(task)

    logging.info(
        "Judging %s paired cases: %s cached calls, %s pending calls, model=%s, temperature=0",
        len(pairs),
        len(current_rows),
        len(pending),
        args.model,
    )

    if pending:
        with ThreadPoolExecutor(max_workers=min(args.workers, len(pending))) as pool:
            futures = {
                pool.submit(
                    run_task,
                    task,
                    args.model,
                    args.thinking,
                    args.reasoning_effort,
                    args.max_tokens,
                    args.schema_retries,
                    prompt_hash,
                ): task
                for task in pending
            }
            for future in as_completed(futures):
                row = future.result()
                key = (row["source_index"], row["order"])
                current_rows[key] = row
                append_jsonl(details_path, row)
                logging.info(
                    "Saved judgment source_index=%s order=%s status=%s",
                    row["source_index"],
                    row["order"],
                    row["status"],
                )

    settings = {
        "judge_model": args.model,
        "judge_temperature": JUDGE_TEMPERATURE,
        "judge_thinking": args.thinking,
        "judge_reasoning_effort": args.reasoning_effort,
        "max_tokens": args.max_tokens,
        "schema_retries": args.schema_retries,
        "workers": args.workers,
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "prompt_sha256": prompt_hash,
    }
    summary = build_summary(
        pairs,
        current_rows,
        args.dynapro,
        args.dynapro_medical,
        settings,
    )
    write_json_atomic(summary_path, summary)
    logging.info("Saved judgment summary to %s", summary_path)

    if summary["counts"]["paired_cases_complete"] != len(pairs):
        raise RuntimeError("Some paired judgments failed; see judge_details.jsonl")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    main()
