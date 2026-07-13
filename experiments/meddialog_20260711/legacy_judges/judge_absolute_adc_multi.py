#!/usr/bin/env python3
"""Legacy multi-method absolute A/D/C judge retained for reproducibility."""

import argparse
import hashlib
import logging
import os
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
    load_records,
    write_json_atomic,
)
from experiments.meddialog_20260711.legacy_judges.adc_common import (  # noqa: E402
    DIMENSIONS,
)
from experiments.meddialog_20260711.legacy_judges.judge_absolute_adc import (  # noqa: E402
    PROMPT_PATH,
    PROMPT_VERSION,
    SCHEMA_VERSION,
    append_jsonl,
    load_latest_rows,
    make_task,
    mean,
    median,
    run_task,
    validate_result,
)


JUDGE_TEMPERATURE = DEFAULT_JUDGE_TEMPERATURE
COMMON_GENERATION_FIELDS = (
    "provider",
    "user_model",
    "assistant_model",
    "user_temperature",
    "assistant_temperature",
    "thinking",
    "reasoning_effort",
    "max_new_turns",
    "natural_termination_enabled",
    "task_desc",
    "single_turn_completion",
    "single_turn_metadata",
)
RESERVED_METHOD_NAMES = {"tie", "incomplete"}


def parse_input_specs(values):
    inputs = {}
    for value in values:
        method, separator, raw_path = value.partition("=")
        method = method.strip()
        raw_path = raw_path.strip()
        if not separator or not method or not raw_path:
            raise ValueError(
                f"Invalid --input {value!r}; expected a non-empty METHOD=PATH"
            )
        if method in RESERVED_METHOD_NAMES:
            raise ValueError(f"Reserved --input method name {method!r}")
        if method in inputs:
            raise ValueError(f"Duplicate --input method {method!r}")
        inputs[method] = Path(raw_path)
    if len(inputs) < 2:
        raise ValueError("At least two --input METHOD=PATH arguments are required")
    return inputs


def validate_record_sets(records_by_method, limit=None):
    if len(records_by_method) < 2:
        raise ValueError("At least two methods are required")
    methods = tuple(records_by_method)
    reference_method = methods[0]
    reference_indices = set(records_by_method[reference_method])

    for method in methods[1:]:
        indices = set(records_by_method[method])
        if indices != reference_indices:
            only_reference = sorted(reference_indices - indices)
            only_method = sorted(indices - reference_indices)
            raise ValueError(
                "Input files do not contain identical source indices: "
                f"{reference_method}_only={only_reference}, "
                f"{method}_only={only_method}"
            )

    groups = []
    for source_index in sorted(reference_indices):
        records = {
            method: records_by_method[method][source_index] for method in methods
        }
        hidden_targets = {
            method: record.get("single_turn_prompt")
            for method, record in records.items()
        }
        if any(
            not isinstance(target, str) or not target.strip()
            for target in hidden_targets.values()
        ):
            raise ValueError(
                f"source_index={source_index} has a missing/empty hidden target"
            )
        if len(set(hidden_targets.values())) != 1:
            raise ValueError(
                f"source_index={source_index} has mismatched hidden targets"
            )

        for field in COMMON_GENERATION_FIELDS:
            missing = [method for method, record in records.items() if field not in record]
            if missing:
                raise ValueError(
                    f"source_index={source_index} cannot verify common generation "
                    f"settings; field={field!r} missing from methods={missing}"
                )
            values = [records[method][field] for method in methods]
            if any(value != values[0] for value in values[1:]):
                raise ValueError(
                    f"source_index={source_index} has mismatched generation "
                    f"setting {field!r}"
                )

        groups.append((source_index, records))

    return groups[:limit] if limit is not None else groups


def load_groups(inputs, limit=None):
    records_by_method = {
        method: load_records(path, method) for method, path in inputs.items()
    }
    return validate_record_sets(records_by_method, limit)


def derive_winner(results, methods):
    unflagged = [
        method
        for method in methods
        if not results[method]["guardrails"]["major_medical_safety_issue"]
    ]
    if not unflagged:
        return "tie", list(methods)

    highest = max(results[method]["total_score"] for method in unflagged)
    leaders = [
        method for method in unflagged if results[method]["total_score"] == highest
    ]
    if len(leaders) == 1:
        return leaders[0], leaders
    return "tie", leaders


def build_summary(groups, rows, inputs, settings):
    methods = tuple(inputs)
    per_case = []
    method_results = {method: [] for method in methods}
    winner_counts = {method: 0 for method in methods}
    winner_counts.update({"tie": 0, "incomplete": 0})

    for source_index, _ in groups:
        case_rows = {method: rows.get((source_index, method)) for method in methods}
        if any(
            row is None or row.get("status") != "ok"
            for row in case_rows.values()
        ):
            per_case.append({"source_index": source_index, "winner": "incomplete"})
            winner_counts["incomplete"] += 1
            continue

        results = {
            method: case_rows[method]["parsed_result"] for method in methods
        }
        for method in methods:
            method_results[method].append(results[method])

        winner, leaders = derive_winner(results, methods)
        winner_counts[winner] += 1
        case = {
            "source_index": source_index,
            "winner": winner,
            "scores": {
                method: {
                    dimension: results[method]["scores"][dimension]["score"]
                    for dimension in DIMENSIONS
                }
                | {"total_score": results[method]["total_score"]}
                for method in methods
            },
            "guardrails": {
                method: results[method]["guardrails"] for method in methods
            },
        }
        if winner == "tie":
            case["tied_methods"] = leaders
        per_case.append(case)

    aggregate = {}
    for method, results in method_results.items():
        aggregate[method] = {
            "n_complete_cases": len(results),
            "scores": {
                dimension: {
                    "mean": mean(
                        [result["scores"][dimension]["score"] for result in results]
                    ),
                    "median": median(
                        [result["scores"][dimension]["score"] for result in results]
                    ),
                }
                for dimension in DIMENSIONS
            },
            "total_score": {
                "mean": mean([result["total_score"] for result in results]),
                "median": median([result["total_score"] for result in results]),
            },
            "guardrail_true_rate": {
                key: mean([result["guardrails"][key] for result in results])
                for key in (
                    "explicit_request_completed",
                    "unnecessary_continuation",
                    "major_medical_safety_issue",
                )
            },
        }

    complete_cases = [case for case in per_case if case["winner"] != "incomplete"]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {method: str(path.resolve()) for method, path in inputs.items()},
        "methods": list(methods),
        "settings": settings,
        "counts": {
            "cases_requested": len(groups),
            "cases_complete": len(complete_cases),
            "judge_calls_expected": len(groups) * len(methods),
            "judge_calls_successful": sum(
                row.get("status") == "ok" for row in rows.values()
            ),
        },
        "winner_rule": (
            "A method with a major medical safety issue is ineligible when at "
            "least one unflagged method exists. Among eligible methods, the unique "
            "highest total score wins; equal highest scores are a tie. If every "
            "method is flagged, the case is a tie."
        ),
        "winner_counts": winner_counts,
        "aggregate": aggregate,
        "per_case": per_case,
        "caveat": (
            "This standalone absolute judgment is descriptive; do not claim "
            "statistical significance."
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone multi-method absolute A/D/C judgment"
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="METHOD=PATH",
        help="Repeat for each method to judge",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--model", default=os.environ.get("JUDGE_MODEL", "deepseek-v4-flash")
    )
    parser.add_argument(
        "--thinking",
        choices=("enabled", "disabled"),
        default=os.environ.get("JUDGE_THINKING", "disabled"),
    )
    parser.add_argument(
        "--reasoning-effort", default=os.environ.get("JUDGE_REASONING_EFFORT")
    )
    parser.add_argument(
        "--workers", type=int, default=int(os.environ.get("JUDGE_WORKERS", "10"))
    )
    parser.add_argument("--schema-retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--limit", type=int)
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

    inputs = parse_input_specs(args.input)
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    prompt_hash = hashlib.sha256(prompt_template.encode("utf-8")).hexdigest()
    groups = load_groups(inputs, args.limit)
    output_dir = args.output_dir or (
        PROJECT_ROOT
        / "output/judgments_absolute"
        / "__vs__".join(path.stem for path in inputs.values())
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "absolute_judge_details.jsonl"
    summary_path = output_dir / "absolute_judge_summary.json"

    tasks = []
    for source_index, records in groups:
        for method, record in records.items():
            tasks.append(
                make_task(
                    source_index,
                    method,
                    record,
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
        key = (task["source_index"], task["method"])
        existing = latest_rows.get(key)
        if (
            existing
            and existing.get("status") == "ok"
            and existing.get("cache_key") == task["cache_key"]
        ):
            try:
                validate_result(existing.get("parsed_result"), task["assistant_turns"])
            except Exception:
                pending.append(task)
            else:
                current_rows[key] = existing
        else:
            pending.append(task)

    logging.info(
        "Absolute judging %s conversations across methods=%s: %s cached, %s "
        "pending, model=%s, temperature=0",
        len(tasks),
        ",".join(inputs),
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
                current_rows[(row["source_index"], row["method"])] = row
                append_jsonl(details_path, row)
                logging.info(
                    "Saved absolute judgment source_index=%s method=%s status=%s",
                    row["source_index"],
                    row["method"],
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
    summary = build_summary(groups, current_rows, inputs, settings)
    write_json_atomic(summary_path, summary)
    logging.info("Saved multi-method absolute summary to %s", summary_path)

    if summary["counts"]["cases_complete"] != len(groups):
        raise RuntimeError("Some absolute judgments failed; see details JSONL")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    main()
