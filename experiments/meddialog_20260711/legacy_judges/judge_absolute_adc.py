#!/usr/bin/env python3
"""Legacy two-method absolute A/D/C judge retained for reproducibility."""

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
    / "dynapro_adc_absolute_judge.txt"
)
PROMPT_VERSION = "dynapro-adc-absolute-v1"
SCHEMA_VERSION = "dynapro-adc-absolute-schema-v1"
METHODS = ("dynapro", "dynapro_medical")
JUDGE_TEMPERATURE = DEFAULT_JUDGE_TEMPERATURE


def build_prompt(template, hidden_target, conversation):
    rendered, assistant_turns = render_conversation(conversation)
    prompt = template.replace("__HIDDEN_TARGET__", hidden_target)
    prompt = prompt.replace(
        "__CONVERSATION__", json.dumps(rendered, ensure_ascii=False, indent=2)
    )
    return prompt, assistant_turns


def validate_result(result, assistant_turns):
    validate_candidate(result, "conversation", assistant_turns)
    confidence = result.get("judge_confidence")
    if confidence not in {"low", "medium", "high"}:
        raise ValueError("judge_confidence must be low, medium, or high")
    return result


def make_task(
    source_index,
    method,
    record,
    template,
    prompt_hash,
    model,
    thinking,
    reasoning_effort,
    max_tokens,
):
    hidden_target = record["single_turn_prompt"]
    prompt, assistant_turns = build_prompt(
        template, hidden_target, record["conversation"]
    )
    conversation_hash = sha256_json(record["conversation"])
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
            "hidden_target": hidden_target,
            "conversation_hash": conversation_hash,
        }
    )
    return {
        "source_index": source_index,
        "method": method,
        "prompt": prompt,
        "assistant_turns": assistant_turns,
        "conversation_hash": conversation_hash,
        "cache_key": cache_key,
    }


def run_task(
    task,
    model,
    thinking,
    reasoning_effort,
    max_tokens,
    schema_retries,
    prompt_hash,
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
            parsed = validate_result(
                parse_json_object(raw_response), task["assistant_turns"]
            )
            return result_row(
                task,
                model,
                thinking,
                prompt_hash,
                "ok",
                attempt,
                raw_response,
                parsed,
                None,
            )
        except Exception as error:
            last_error = repr(error)
            logging.warning(
                "Absolute judge failed source_index=%s method=%s attempt=%s/%s: %s",
                task["source_index"],
                task["method"],
                attempt,
                schema_retries,
                error,
            )
    return result_row(
        task,
        model,
        thinking,
        prompt_hash,
        "failed",
        schema_retries,
        raw_response,
        None,
        last_error,
    )


def result_row(
    task,
    model,
    thinking,
    prompt_hash,
    status,
    attempts,
    raw_response,
    parsed_result,
    error,
):
    return {
        "source_index": task["source_index"],
        "method": task["method"],
        "judge_model": model,
        "judge_temperature": JUDGE_TEMPERATURE,
        "judge_thinking": thinking,
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "prompt_sha256": prompt_hash,
        "conversation_hash": task["conversation_hash"],
        "cache_key": task["cache_key"],
        "status": status,
        "attempts": attempts,
        "raw_response": raw_response,
        "parsed_result": parsed_result,
        "error": error,
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
            key = (row["source_index"], row["method"])
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


def derived_winner(dynapro, medical):
    dynapro_major = dynapro["guardrails"]["major_medical_safety_issue"]
    medical_major = medical["guardrails"]["major_medical_safety_issue"]
    if dynapro_major and medical_major:
        return "tie"
    if dynapro_major != medical_major:
        return "dynapro_medical" if dynapro_major else "dynapro"
    if dynapro["total_score"] > medical["total_score"]:
        return "dynapro"
    if medical["total_score"] > dynapro["total_score"]:
        return "dynapro_medical"
    return "tie"


def build_summary(pairs, rows, dynapro_path, medical_path, settings):
    per_case = []
    method_results = {method: [] for method in METHODS}
    winner_counts = {"dynapro": 0, "dynapro_medical": 0, "tie": 0, "incomplete": 0}

    for source_index, _, _ in pairs:
        case_rows = {method: rows.get((source_index, method)) for method in METHODS}
        if any(row is None or row.get("status") != "ok" for row in case_rows.values()):
            per_case.append({"source_index": source_index, "winner": "incomplete"})
            winner_counts["incomplete"] += 1
            continue

        results = {method: case_rows[method]["parsed_result"] for method in METHODS}
        for method in METHODS:
            method_results[method].append(results[method])

        winner = derived_winner(results["dynapro"], results["dynapro_medical"])
        winner_counts[winner] += 1
        delta = {
            dimension: results["dynapro_medical"]["scores"][dimension]["score"]
            - results["dynapro"]["scores"][dimension]["score"]
            for dimension in DIMENSIONS
        }
        delta["total_score"] = (
            results["dynapro_medical"]["total_score"]
            - results["dynapro"]["total_score"]
        )
        per_case.append(
            {
                "source_index": source_index,
                "winner": winner,
                "scores": {
                    method: {
                        dimension: results[method]["scores"][dimension]["score"]
                        for dimension in DIMENSIONS
                    }
                    | {"total_score": results[method]["total_score"]}
                    for method in METHODS
                },
                "guardrails": {
                    method: results[method]["guardrails"] for method in METHODS
                },
                "medical_minus_dynapro": delta,
            }
        )

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
                row.get("status") == "ok" for row in rows.values()
            ),
        },
        "winner_counts": winner_counts,
        "aggregate": aggregate,
        "paired_delta_medical_minus_dynapro": paired_delta,
        "per_case": per_case,
        "caveat": "This standalone absolute judgment is descriptive; do not claim statistical significance.",
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone absolute A/D/C judgment for DynaPro conversations"
    )
    parser.add_argument("--dynapro", type=Path, required=True)
    parser.add_argument("--dynapro-medical", type=Path, required=True)
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

    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    prompt_hash = hashlib.sha256(prompt_template.encode("utf-8")).hexdigest()
    pairs = load_pairs(args.dynapro, args.dynapro_medical, args.limit)
    output_dir = args.output_dir or (
        PROJECT_ROOT
        / "output/judgments_absolute"
        / f"{args.dynapro.stem}__vs__{args.dynapro_medical.stem}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "absolute_judge_details.jsonl"
    summary_path = output_dir / "absolute_judge_summary.json"

    tasks = []
    for source_index, dynapro_record, medical_record in pairs:
        for method, record in (
            ("dynapro", dynapro_record),
            ("dynapro_medical", medical_record),
        ):
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
        "Absolute judging %s conversations: %s cached, %s pending, model=%s, temperature=0",
        len(tasks),
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
    summary = build_summary(
        pairs,
        current_rows,
        args.dynapro,
        args.dynapro_medical,
        settings,
    )
    write_json_atomic(summary_path, summary)
    logging.info("Saved absolute judgment summary to %s", summary_path)

    if summary["counts"]["paired_cases_complete"] != len(pairs):
        raise RuntimeError("Some absolute judgments failed; see details JSONL")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    main()
