#!/usr/bin/env python3
"""Canonical MedDialog judge using four independent evaluator calls.

Historical pairwise and combined A/D/C protocols live under
experiments/meddialog_20260711/legacy_judges/.
"""

import argparse
import hashlib
import json
import logging
import os
import random
import re
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from evaluation.judge_common import (
    DEFAULT_JUDGE_TEMPERATURE,
    PROJECT_ROOT,
    load_env,
    load_records,
    require_nonempty_string,
    sha256_json,
    write_json_atomic,
)
from simulation.modules.deepseek import chat_completion as deepseek_chat_completion
from simulation.modules.openai_api import responses_completion
from simulation.modules.provider_common import parse_json_object


PROMPT_VERSION = "meddialog-four-independent-evals-v1"
SCHEMA_VERSION = "meddialog-four-independent-evals-schema-v2"
VALIDATION_REPAIR_VERSION = "validator-feedback-v2"
EVALUATORS = {
    "anticipation": {
        "prompt_path": PROJECT_ROOT
        / "evaluation/prompts/dynapro_anticipation_absolute_judge.txt",
        "score_bands_path": PROJECT_ROOT
        / "evaluation/prompts/score_bands/anticipation.txt",
        "max_score": 10,
        "uses_hidden_target": True,
    },
    "discovery": {
        "prompt_path": PROJECT_ROOT
        / "evaluation/prompts/dynapro_discovery_absolute_judge.txt",
        "score_bands_path": PROJECT_ROOT
        / "evaluation/prompts/score_bands/discovery.txt",
        "max_score": 10,
        "uses_hidden_target": True,
    },
    "calibration": {
        "prompt_path": PROJECT_ROOT
        / "evaluation/prompts/dynapro_calibration_absolute_judge.txt",
        "score_bands_path": PROJECT_ROOT
        / "evaluation/prompts/score_bands/calibration.txt",
        "max_score": 10,
        "uses_hidden_target": True,
    },
    "medical_quality": {
        "prompt_path": PROJECT_ROOT
        / "evaluation/prompts/medical_quality_absolute_judge.txt",
        "score_bands_path": PROJECT_ROOT
        / "evaluation/prompts/score_bands/medical_quality.txt",
        "max_score": 20,
        "uses_hidden_target": False,
    },
}
EVALUATOR_NAMES = tuple(EVALUATORS)
CONFIDENCE_VALUES = {"low", "medium", "high"}
MEDICAL_ISSUE_CATEGORIES = {
    "factual_error",
    "unsupported_or_overconfident_claim",
    "unsafe_medication_or_dose",
    "missed_urgent_escalation",
    "fabricated_resource",
    "misleading_information",
    "unprofessional_communication",
    "other_medical_quality_issue",
}
MEDICAL_SEVERITIES = {"minor", "major", "critical"}
DISPLAY_NAMES = {
    "dynapro": "Original DynaPro",
    "dynapro_medical": "Medical DynaPro",
    "proact": "Proact Instruction",
    "generic_proact": "Generic Proact",
    "none": "No Prompt",
}
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


def effective_judge_temperature(provider, requested_temperature=None):
    if requested_temperature is not None:
        return requested_temperature
    return DEFAULT_JUDGE_TEMPERATURE if provider == "deepseek" else None


def judge_completion(
    provider,
    messages,
    *,
    model,
    temperature,
    max_tokens,
    json_output,
    thinking,
    reasoning_effort,
    return_metadata,
):
    if provider == "deepseek":
        return deepseek_chat_completion(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_output=json_output,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            return_metadata=return_metadata,
        )
    return responses_completion(
        messages,
        model=model,
        temperature=temperature,
        max_output_tokens=max_tokens,
        json_output=json_output,
        reasoning_effort=reasoning_effort,
        return_metadata=return_metadata,
        force_temperature=temperature is not None,
    )


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
        if method in inputs:
            raise ValueError(f"Duplicate --input method {method!r}")
        inputs[method] = Path(raw_path)
    if not inputs:
        raise ValueError("At least one --input METHOD=PATH is required")
    return inputs


def load_input_records(inputs, limit=None):
    records_by_method = {}
    targets_by_index = {}
    generation_settings_by_index = {}
    for method, path in inputs.items():
        records = load_records(path, method)
        ordered_indices = sorted(records)
        if limit is not None:
            ordered_indices = ordered_indices[:limit]
        selected = {index: records[index] for index in ordered_indices}
        for source_index, record in selected.items():
            hidden_target = record.get("single_turn_prompt")
            require_nonempty_string(
                hidden_target,
                f"{method}[{source_index}].single_turn_prompt",
            )
            existing = targets_by_index.get(source_index)
            if existing is not None and existing != hidden_target:
                raise ValueError(
                    f"source_index={source_index} has mismatched hidden targets"
                )
            targets_by_index[source_index] = hidden_target
            missing_fields = [
                field for field in COMMON_GENERATION_FIELDS if field not in record
            ]
            if missing_fields:
                raise ValueError(
                    f"{method}[{source_index}] is missing generation fields "
                    f"{missing_fields}"
                )
            generation_settings = {
                field: record[field] for field in COMMON_GENERATION_FIELDS
            }
            existing_settings = generation_settings_by_index.get(source_index)
            if (
                existing_settings is not None
                and existing_settings != generation_settings
            ):
                mismatched = [
                    field
                    for field in COMMON_GENERATION_FIELDS
                    if existing_settings[field] != generation_settings[field]
                ]
                raise ValueError(
                    f"source_index={source_index} has mismatched generation "
                    f"settings {mismatched}"
                )
            generation_settings_by_index[source_index] = generation_settings
        records_by_method[method] = selected
    return records_by_method


def render_conversation(conversation):
    rendered = []
    message_roles = {}
    for message_index, turn in enumerate(conversation, 1):
        role = turn.get("role")
        content = turn.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            raise ValueError(f"Invalid conversation turn: {turn!r}")
        rendered.append(
            {
                "message_index": message_index,
                "role": role,
                "content": content,
            }
        )
        message_roles[message_index] = role
    return rendered, message_roles


def build_prompt(template, hidden_target, conversation, uses_hidden_target):
    rendered, message_roles = render_conversation(conversation)
    expected_hidden_count = 1 if uses_hidden_target else 0
    if template.count("__CONVERSATION__") != 1:
        raise ValueError("Evaluator template must contain one conversation marker")
    if template.count("__HIDDEN_TARGET__") != expected_hidden_count:
        raise ValueError("Evaluator template has an unexpected hidden-target marker")
    replacements = {
        "__CONVERSATION__": json.dumps(rendered, ensure_ascii=False, indent=2),
        "__HIDDEN_TARGET__": json.dumps(hidden_target, ensure_ascii=False),
    }
    prompt = re.sub(
        r"__(?:CONVERSATION|HIDDEN_TARGET)__",
        lambda match: replacements[match.group(0)],
        template,
    )
    return prompt, message_roles


def validate_evidence(evidence, message_roles, field):
    if not isinstance(evidence, list):
        raise ValueError(f"{field} must be a list")
    for position, item in enumerate(evidence):
        item_field = f"{field}[{position}]"
        if not isinstance(item, dict):
            raise ValueError(f"{item_field} must be an object")
        if set(item) != {"message_index", "description"}:
            raise ValueError(f"{item_field} has unexpected keys")
        message_index = item.get("message_index")
        if (
            isinstance(message_index, bool)
            or not isinstance(message_index, int)
            or message_index not in message_roles
        ):
            raise ValueError(f"{item_field}.message_index is invalid")
        require_nonempty_string(item.get("description"), f"{item_field}.description")


def validate_discoveries(discoveries, message_roles):
    if not isinstance(discoveries, list):
        raise ValueError("discoveries must be a list")
    required = {
        "item",
        "eliciting_assistant_message_index",
        "revealing_user_message_index",
        "using_assistant_message_index",
        "material_effect",
    }
    for position, discovery in enumerate(discoveries):
        field = f"discoveries[{position}]"
        if not isinstance(discovery, dict) or set(discovery) != required:
            raise ValueError(f"{field} must contain exactly {sorted(required)}")
        require_nonempty_string(discovery.get("item"), f"{field}.item")
        require_nonempty_string(
            discovery.get("material_effect"), f"{field}.material_effect"
        )
        eliciting = discovery.get("eliciting_assistant_message_index")
        revealing = discovery.get("revealing_user_message_index")
        using = discovery.get("using_assistant_message_index")
        if any(isinstance(index, bool) for index in (eliciting, revealing, using)):
            raise ValueError(f"{field} message indices must be integers")
        if message_roles.get(eliciting) != "assistant":
            raise ValueError(f"{field} eliciting index must reference assistant")
        if message_roles.get(revealing) != "user":
            raise ValueError(f"{field} revealing index must reference user")
        if revealing != eliciting + 1:
            raise ValueError(
                f"{field} user revelation must immediately answer the eliciting message"
            )
        if message_roles.get(using) != "assistant":
            raise ValueError(f"{field} using index must reference assistant")
        if not revealing < using:
            raise ValueError(
                f"{field} must use information after it is revealed "
                f"(eliciting_assistant_message_index={eliciting}, "
                f"revealing_user_message_index={revealing}, "
                f"using_assistant_message_index={using}; delete this discovery "
                "or change using_assistant_message_index to a later assistant "
                "message that visibly uses the revealed information)"
            )


def validate_medical_issues(issues, message_roles):
    if not isinstance(issues, list):
        raise ValueError("issues must be a list")
    required = {
        "category",
        "severity",
        "message_index",
        "description",
        "correction",
    }
    for position, issue in enumerate(issues):
        field = f"issues[{position}]"
        if not isinstance(issue, dict) or set(issue) != required:
            raise ValueError(f"{field} must contain exactly {sorted(required)}")
        if issue.get("category") not in MEDICAL_ISSUE_CATEGORIES:
            raise ValueError(f"{field}.category is invalid")
        if issue.get("severity") not in MEDICAL_SEVERITIES:
            raise ValueError(f"{field}.severity is invalid")
        message_index = issue.get("message_index")
        if isinstance(message_index, bool):
            raise ValueError(f"{field}.message_index must be an integer")
        if message_roles.get(message_index) != "assistant":
            raise ValueError(f"{field}.message_index must reference assistant")
        require_nonempty_string(issue.get("description"), f"{field}.description")
        require_nonempty_string(issue.get("correction"), f"{field}.correction")


def validate_result(evaluator, result, message_roles):
    if not isinstance(result, dict):
        raise ValueError("Evaluator result must be an object")
    expected = {"score", "reason", "evidence", "judge_confidence"}
    if evaluator == "discovery":
        expected.add("discoveries")
    if evaluator == "medical_quality":
        expected.add("issues")
    if set(result) != expected:
        raise ValueError(
            f"{evaluator} result keys must be exactly {sorted(expected)}; "
            f"got {sorted(result)}"
        )
    score = result.get("score")
    max_score = EVALUATORS[evaluator]["max_score"]
    if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= max_score:
        raise ValueError(f"{evaluator}.score must be an integer 0-{max_score}")
    require_nonempty_string(result.get("reason"), f"{evaluator}.reason")
    validate_evidence(result.get("evidence"), message_roles, "evidence")
    confidence = result.get("judge_confidence")
    if confidence not in CONFIDENCE_VALUES:
        raise ValueError(f"{evaluator}.judge_confidence is invalid")
    if evaluator == "discovery":
        validate_discoveries(result.get("discoveries"), message_roles)
    if evaluator == "medical_quality":
        validate_medical_issues(result.get("issues"), message_roles)
    return result


def build_validation_repair_messages(prompt, raw_response, validation_error):
    targeted_guidance = ""
    discovery_order_error = re.search(
        r"(discoveries\[\d+\]).*?"
        r"revealing_user_message_index=(\d+).*?"
        r"using_assistant_message_index=(\d+)",
        validation_error,
    )
    if discovery_order_error:
        field, revealing, using = discovery_order_error.groups()
        targeted_guidance = f"""

Specific repair required:
- {field} claims using_assistant_message_index={using}, but the information is
  only revealed at revealing_user_message_index={revealing}.
- Delete that discovery unless a later assistant message, with index greater than
  {revealing}, visibly uses the revealed information.
- If you delete or narrow the discovery, also adjust the score, reason, and
  evidence so they only credit valid post-revelation use."""

    repair_instruction = f"""Your previous JSON response failed the deterministic validator:

{validation_error}
{targeted_guidance}

Return one corrected COMPLETE JSON object only, following the original evaluator
rubric and schema. Do not change message indices merely to make validation pass,
invent evidence, or claim a later use that is not visible in the conversation.
Remove unsupported items. Reconcile the score, reason, evidence, and any
discoveries or issues so the score reflects only valid evidence. Preserve
substantively valid judgments, but change the score if the invalid claim affected
it. Re-check the complete object before returning it."""
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": raw_response},
        {"role": "user", "content": repair_instruction},
    ]


def make_task(
    method,
    source_index,
    evaluator,
    record,
    prompt_template,
    prompt_hash,
    provider,
    model,
    temperature,
    thinking,
    reasoning_effort,
    max_tokens,
):
    config = EVALUATORS[evaluator]
    hidden_target = record["single_turn_prompt"]
    conversation = record["conversation"]
    prompt, message_roles = build_prompt(
        prompt_template,
        hidden_target,
        conversation,
        config["uses_hidden_target"],
    )
    conversation_hash = sha256_json(conversation)
    cache_key = sha256_json(
        {
            "method": method,
            "source_index": source_index,
            "evaluator": evaluator,
            "prompt_hash": prompt_hash,
            "schema_version": SCHEMA_VERSION,
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "thinking": thinking,
            "reasoning_effort": reasoning_effort,
            "max_tokens": max_tokens,
            "hidden_target": hidden_target if config["uses_hidden_target"] else None,
            "conversation_hash": conversation_hash,
        }
    )
    return {
        "method": method,
        "source_index": source_index,
        "evaluator": evaluator,
        "prompt": prompt,
        "message_roles": message_roles,
        "conversation_hash": conversation_hash,
        "cache_key": cache_key,
        "judge_temperature": temperature,
    }


def result_row(
    task,
    provider,
    model,
    thinking,
    reasoning_effort,
    prompt_hash,
    status,
    attempts,
    raw_response,
    parsed_result,
    error,
    api_calls,
    validation_repair,
    validation_failures,
):
    return {
        "source_index": task["source_index"],
        "method": task["method"],
        "evaluator": task["evaluator"],
        "judge_provider": provider,
        "judge_model": model,
        "judge_temperature": task["judge_temperature"],
        "judge_thinking": thinking,
        "judge_reasoning_effort": reasoning_effort,
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
        "api_calls": api_calls,
        "validation_repair": validation_repair,
        "validation_repair_version": (
            VALIDATION_REPAIR_VERSION if validation_repair == "enabled" else None
        ),
        "validation_failures": validation_failures,
    }


def run_task(
    task,
    provider,
    model,
    thinking,
    reasoning_effort,
    max_tokens,
    schema_retries,
    prompt_hash,
    record_usage,
    validation_repair,
):
    raw_response = None
    last_error = None
    api_calls = []
    validation_failures = []
    repair_context = None
    for attempt in range(1, schema_retries + 1):
        request_kind = "repair" if repair_context is not None else "initial"
        messages = (
            build_validation_repair_messages(
                task["prompt"],
                repair_context["raw_response"],
                repair_context["error"],
            )
            if repair_context is not None
            else [{"role": "user", "content": task["prompt"]}]
        )
        try:
            response = judge_completion(
                provider,
                messages,
                model=model,
                temperature=task["judge_temperature"],
                max_tokens=max_tokens,
                json_output=True,
                thinking=thinking,
                reasoning_effort=reasoning_effort,
                return_metadata=record_usage,
            )
            if record_usage:
                raw_response = response["content"]
                api_calls.append(
                    {
                        "attempt": attempt,
                        "model": response.get("model") or model,
                        "response_id": response.get("response_id"),
                        "usage": response.get("usage", {}),
                        "request_kind": request_kind,
                    }
                )
            else:
                raw_response = response
        except Exception as error:
            last_error = repr(error)
            logging.warning(
                "Evaluator request failed method=%s source_index=%s evaluator=%s "
                "attempt=%s/%s request_kind=%s: %s",
                task["method"],
                task["source_index"],
                task["evaluator"],
                attempt,
                schema_retries,
                request_kind,
                error,
            )
            continue

        try:
            parsed = validate_result(
                task["evaluator"],
                parse_json_object(raw_response),
                task["message_roles"],
            )
            if record_usage:
                api_calls[-1]["validation_status"] = "ok"
            return result_row(
                task,
                provider,
                model,
                thinking,
                reasoning_effort,
                prompt_hash,
                "ok",
                attempt,
                raw_response,
                parsed,
                None,
                api_calls,
                validation_repair,
                validation_failures,
            )
        except Exception as error:
            last_error = repr(error)
            if record_usage:
                api_calls[-1]["validation_status"] = "failed"
                api_calls[-1]["validation_error"] = str(error)
            if validation_repair == "enabled":
                failure = {
                    "attempt": attempt,
                    "request_kind": request_kind,
                    "raw_response": raw_response,
                    "error": repr(error),
                }
                validation_failures.append(failure)
                repair_context = {
                    "raw_response": raw_response,
                    "error": str(error),
                }
            logging.warning(
                "Evaluator validation failed method=%s source_index=%s evaluator=%s "
                "attempt=%s/%s request_kind=%s: %s",
                task["method"],
                task["source_index"],
                task["evaluator"],
                attempt,
                schema_retries,
                request_kind,
                error,
            )
    return result_row(
        task,
        provider,
        model,
        thinking,
        reasoning_effort,
        prompt_hash,
        "failed",
        schema_retries,
        raw_response,
        None,
        last_error,
        api_calls,
        validation_repair,
        validation_failures,
    )


def load_latest_rows(path):
    latest = {}
    if not path.exists():
        return latest
    lines = path.read_text(encoding="utf-8").splitlines()
    last_nonempty = max(
        (index for index, line in enumerate(lines) if line.strip()), default=-1
    )
    for line_index, line in enumerate(lines):
        line_number = line_index + 1
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            key = (row["method"], row["source_index"], row["evaluator"])
        except Exception as error:
            if line_index == last_nonempty:
                logging.warning("Ignoring truncated final JSONL row at %s:%s", path, line_number)
                break
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
        latest[key] = row
    return latest


def has_complete_api_usage(row):
    required_fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    calls = row.get("api_calls")
    if not isinstance(calls, list) or not calls:
        return False
    for call in calls:
        if not isinstance(call, dict):
            return False
        usage = call.get("usage")
        if not isinstance(usage, dict):
            return False
        for field in required_fields:
            value = usage.get(field)
            if isinstance(value, bool) or not isinstance(value, int):
                return False
    return True


def append_jsonl(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def score_stats(values):
    return {
        "mean": round(statistics.mean(values), 3) if values else None,
        "median": round(statistics.median(values), 3) if values else None,
        "stdev": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
    }


def aggregate_cases(cases):
    complete = [case for case in cases if case["status"] == "ok"]
    result = {
        "n_requested": len(cases),
        "n_complete": len(complete),
        "scores": {
            evaluator: score_stats([case["scores"][evaluator] for case in complete])
            for evaluator in EVALUATOR_NAMES
        },
        "total_score": score_stats([case["total_score"] for case in complete]),
    }
    medical_results = [case["results"]["medical_quality"] for case in complete]
    result["medical_issue_summary"] = {
        "any_issue_rate": round(
            statistics.mean(bool(item["issues"]) for item in medical_results), 3
        )
        if medical_results
        else None,
        "major_or_critical_issue_rate": round(
            statistics.mean(
                any(
                    issue["severity"] in {"major", "critical"}
                    for issue in item["issues"]
                )
                for item in medical_results
            ),
            3,
        )
        if medical_results
        else None,
        "mean_issue_count": round(
            statistics.mean(len(item["issues"]) for item in medical_results), 3
        )
        if medical_results
        else None,
    }
    return result


def table_rows(methods, cases_by_method, indices=None):
    rows = []
    for method in methods:
        cases = cases_by_method[method]
        if indices is not None:
            cases = [case for case in cases if case["source_index"] in indices]
        aggregate = aggregate_cases(cases)
        rows.append(
            {
                "method": method,
                "display_name": DISPLAY_NAMES.get(method, method),
                "n": aggregate["n_complete"],
                "anticipation": aggregate["scores"]["anticipation"]["mean"],
                "discovery": aggregate["scores"]["discovery"]["mean"],
                "calibration": aggregate["scores"]["calibration"]["mean"],
                "medical_quality": aggregate["scores"]["medical_quality"]["mean"],
                "total_score": aggregate["total_score"]["mean"],
            }
        )
    return rows


def paired_comparison(left_method, right_method, cases_by_method):
    left = {case["source_index"]: case for case in cases_by_method[left_method]}
    right = {case["source_index"]: case for case in cases_by_method[right_method]}
    indices = sorted(set(left) & set(right))
    complete = [
        index
        for index in indices
        if left[index]["status"] == "ok" and right[index]["status"] == "ok"
    ]
    deltas = {
        evaluator: [
            right[index]["scores"][evaluator] - left[index]["scores"][evaluator]
            for index in complete
        ]
        for evaluator in EVALUATOR_NAMES
    }
    total_deltas = [
        right[index]["total_score"] - left[index]["total_score"]
        for index in complete
    ]
    return {
        "left_method": left_method,
        "right_method": right_method,
        "n_complete_pairs": len(complete),
        "right_minus_left": {
            **{evaluator: score_stats(values) for evaluator, values in deltas.items()},
            "total_score": score_stats(total_deltas),
        },
        "winner_counts": {
            left_method: sum(value < 0 for value in total_deltas),
            right_method: sum(value > 0 for value in total_deltas),
            "tie": sum(value == 0 for value in total_deltas),
        },
    }


def build_summary(records_by_method, current_rows, inputs, settings):
    methods = tuple(inputs)
    cases_by_method = {method: [] for method in methods}
    per_case = []
    for method, records in records_by_method.items():
        for source_index in sorted(records):
            rows = {
                evaluator: current_rows.get((method, source_index, evaluator))
                for evaluator in EVALUATOR_NAMES
            }
            complete = all(
                row is not None and row.get("status") == "ok" for row in rows.values()
            )
            if complete:
                results = {
                    evaluator: rows[evaluator]["parsed_result"]
                    for evaluator in EVALUATOR_NAMES
                }
                scores = {
                    evaluator: results[evaluator]["score"]
                    for evaluator in EVALUATOR_NAMES
                }
                case = {
                    "method": method,
                    "source_index": source_index,
                    "status": "ok",
                    "scores": scores,
                    "total_score": sum(scores.values()),
                    "results": results,
                }
            else:
                case = {
                    "method": method,
                    "source_index": source_index,
                    "status": "incomplete",
                }
            cases_by_method[method].append(case)
            per_case.append(case)

    aggregate = {
        method: aggregate_cases(cases_by_method[method]) for method in methods
    }
    common_indices = set.intersection(
        *(set(records_by_method[method]) for method in methods)
    )
    tables = {
        "all_available": table_rows(methods, cases_by_method),
        "common_indices_all_methods": {
            "indices": sorted(common_indices),
            "rows": table_rows(methods, cases_by_method, common_indices),
        },
    }
    if {"dynapro", "dynapro_medical"}.issubset(methods):
        pair_indices = set(records_by_method["dynapro"]) & set(
            records_by_method["dynapro_medical"]
        )
        tables["dynapro_vs_medical_common_indices"] = {
            "indices": sorted(pair_indices),
            "rows": table_rows(
                ("dynapro", "dynapro_medical"), cases_by_method, pair_indices
            ),
        }

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {method: str(path.resolve()) for method, path in inputs.items()},
        "methods": list(methods),
        "settings": settings,
        "counts": {
            "conversations_requested": sum(len(records) for records in records_by_method.values()),
            "conversations_complete": sum(
                case["status"] == "ok" for case in per_case
            ),
            "judge_calls_expected": sum(len(records) for records in records_by_method.values())
            * len(EVALUATOR_NAMES),
            "judge_calls_successful": sum(
                row.get("status") == "ok" for row in current_rows.values()
            ),
        },
        "aggregate": aggregate,
        "tables": tables,
        "per_case": per_case,
        "caveat": (
            "Each evaluator used a separate context-isolated API call. Scores may "
            "still be statistically correlated because they evaluate the same conversation."
        ),
    }
    if {"dynapro", "dynapro_medical"}.issubset(methods):
        summary["paired_comparison"] = paired_comparison(
            "dynapro", "dynapro_medical", cases_by_method
        )
    return summary


def markdown_table(rows):
    header = (
        "| Method | N | Anticipation /10 | Discovery /10 | Calibration /10 | "
        "Medical Quality /20 | Total /50 |\n"
        "|---|---:|---:|---:|---:|---:|---:|"
    )
    body = []
    for row in rows:
        scores = [
            "NA" if row[key] is None else f"{row[key]:.2f}"
            for key in (
                "anticipation",
                "discovery",
                "calibration",
                "medical_quality",
                "total_score",
            )
        ]
        body.append(
            f"| {row['display_name']} | {row['n']} | " + " | ".join(scores) + " |"
        )
    return header + "\n" + "\n".join(body)


def build_markdown(summary):
    sections = [
        "# Independent Four-Evaluator Summary",
        "",
        "## All Available Conversations",
        "",
        markdown_table(summary["tables"]["all_available"]),
    ]
    common = summary["tables"]["common_indices_all_methods"]
    sections.extend(
        [
            "",
            "## Common Source Indices Across All Methods",
            "",
            f"Source indices: {common['indices']}",
            "",
            markdown_table(common["rows"]),
        ]
    )
    pair = summary["tables"].get("dynapro_vs_medical_common_indices")
    if pair is not None:
        sections.extend(
            [
                "",
                "## Original DynaPro vs Medical DynaPro (All Shared Indices)",
                "",
                markdown_table(pair["rows"]),
            ]
        )
    sections.extend(
        [
            "",
            "Each score is the mean of context-isolated evaluator calls. The "
            "three proactive dimensions are out of 10; Medical Quality is out "
            "of 20; Total is out of 50.",
        ]
    )
    return "\n".join(sections) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run four context-isolated evaluators over MedDialog conversations"
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="METHOD=PATH",
        help="Repeat for each method; input files may contain different sample counts",
    )
    parser.add_argument(
        "--provider",
        choices=("deepseek", "openai"),
        default=os.environ.get("JUDGE_PROVIDER", "deepseek"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--model", default=os.environ.get("JUDGE_MODEL", "deepseek-v4-flash")
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help=(
            "Explicit judge sampling temperature. If omitted, DeepSeek uses "
            "0.0 and OpenAI leaves temperature unset."
        ),
    )
    parser.add_argument(
        "--thinking",
        choices=("enabled", "disabled"),
        default=os.environ.get("JUDGE_THINKING", "enabled"),
    )
    parser.add_argument(
        "--reasoning-effort",
        default=os.environ.get("JUDGE_REASONING_EFFORT", "high"),
    )
    parser.add_argument(
        "--workers", type=int, default=int(os.environ.get("JUDGE_WORKERS", "10"))
    )
    parser.add_argument("--schema-retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--record-usage",
        action="store_true",
        help="Save raw API token-usage metadata for every successful HTTP response",
    )
    parser.add_argument(
        "--validation-repair",
        choices=("disabled", "enabled"),
        default="disabled",
        help=(
            "After a JSON/schema validation failure, let the same judge repair its "
            "complete response using the exact validator error"
        ),
    )
    parser.add_argument(
        "--task-shuffle-seed",
        type=int,
        help="Shuffle evaluator task submission order deterministically",
    )
    parser.add_argument(
        "--score-bands",
        choices=("disabled", "enabled"),
        default="disabled",
        help="Append evaluator-specific score-band anchors; disabled preserves the current prompts",
    )
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main():
    # ── Step 1: Validate configuration and inputs ───────────────────────────
    load_env()
    os.environ.setdefault("SSL_CERT_FILE", "/etc/ssl/cert.pem")
    args = parse_args()
    api_key_name = (
        "DEEPSEEK_API_KEY" if args.provider == "deepseek" else "OPENAI_API_KEY"
    )
    if not os.environ.get(api_key_name):
        raise RuntimeError(f"{api_key_name} is not set")
    if args.workers < 1 or args.schema_retries < 1 or args.max_tokens < 1:
        raise ValueError("workers, schema-retries, and max-tokens must be positive")
    if args.temperature is not None and not 0.0 <= args.temperature <= 2.0:
        raise ValueError("temperature must be between 0 and 2")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")

    inputs = parse_input_specs(args.input)
    records_by_method = load_input_records(inputs, args.limit)
    judge_temperature = effective_judge_temperature(
        args.provider, args.temperature
    )

    # ── Step 2: Build the exact evaluator prompts used for cache identity ───
    prompt_templates = {}
    for evaluator, config in EVALUATORS.items():
        template = config["prompt_path"].read_text(encoding="utf-8")
        if args.score_bands == "enabled":
            score_bands = config["score_bands_path"].read_text(encoding="utf-8")
            marker = (
                "<|Hidden Simulator Target|>"
                if config["uses_hidden_target"]
                else "<|Conversation|>"
            )
            if template.count(marker) != 1:
                raise ValueError(
                    f"Expected exactly one score-band insertion marker in {evaluator} prompt"
                )
            template = template.replace(
                marker, f"{score_bands.strip()}\n\n{marker}", 1
            )
        prompt_templates[evaluator] = template
    prompt_hashes = {
        evaluator: hashlib.sha256(template.encode("utf-8")).hexdigest()
        for evaluator, template in prompt_templates.items()
    }

    # ── Step 3: Prepare output and rendered prompt files ────────────────────
    output_dir = args.output_dir or (
        PROJECT_ROOT / "output/judgments_four_eval" / "__".join(inputs)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_prompts_dir = output_dir / "rendered_prompts"
    rendered_prompts_dir.mkdir(parents=True, exist_ok=True)
    for evaluator, template in prompt_templates.items():
        (rendered_prompts_dir / f"{evaluator}.txt").write_text(
            template, encoding="utf-8"
        )
    details_path = output_dir / "four_eval_details.jsonl"
    summary_path = output_dir / "four_eval_summary.json"
    markdown_path = output_dir / "four_eval_summary.md"

    # ── Step 4: Build tasks and reuse only validated cache entries ──────────
    tasks = []
    for method, records in records_by_method.items():
        for source_index, record in records.items():
            for evaluator in EVALUATOR_NAMES:
                tasks.append(
                    make_task(
                        method,
                        source_index,
                        evaluator,
                        record,
                        prompt_templates[evaluator],
                        prompt_hashes[evaluator],
                        args.provider,
                        args.model,
                        judge_temperature,
                        args.thinking,
                        args.reasoning_effort,
                        args.max_tokens,
                    )
                )

    if args.task_shuffle_seed is not None:
        random.Random(args.task_shuffle_seed).shuffle(tasks)

    latest_rows = load_latest_rows(details_path)
    current_rows = {}
    pending = []
    for task in tasks:
        key = (task["method"], task["source_index"], task["evaluator"])
        existing = latest_rows.get(key)
        if (
            existing
            and existing.get("status") == "ok"
            and existing.get("cache_key") == task["cache_key"]
            and (not args.record_usage or has_complete_api_usage(existing))
        ):
            try:
                validate_result(
                    task["evaluator"],
                    existing.get("parsed_result"),
                    task["message_roles"],
                )
            except Exception:
                pending.append(task)
            else:
                current_rows[key] = existing
        else:
            pending.append(task)

    logging.info(
        "Four-evaluator run: %s conversations, %s calls, %s cached, %s pending, "
        "provider=%s, model=%s, temperature=%s, thinking=%s",
        sum(len(records) for records in records_by_method.values()),
        len(tasks),
        len(current_rows),
        len(pending),
        args.provider,
        args.model,
        judge_temperature,
        args.thinking,
    )

    # ── Step 5: Run missing calls; only the main thread appends JSONL ───────
    if pending:
        completed = 0
        with ThreadPoolExecutor(max_workers=min(args.workers, len(pending))) as pool:
            futures = {
                pool.submit(
                    run_task,
                    task,
                    args.provider,
                    args.model,
                    args.thinking,
                    args.reasoning_effort,
                    args.max_tokens,
                    args.schema_retries,
                    prompt_hashes[task["evaluator"]],
                    args.record_usage,
                    args.validation_repair,
                ): task
                for task in pending
            }
            for future in as_completed(futures):
                row = future.result()
                key = (row["method"], row["source_index"], row["evaluator"])
                current_rows[key] = row
                append_jsonl(details_path, row)
                completed += 1
                if completed % 25 == 0 or completed == len(pending) or row["status"] != "ok":
                    logging.info(
                        "Saved %s/%s pending evaluator results; latest=%s/%s/%s status=%s",
                        completed,
                        len(pending),
                        row["method"],
                        row["source_index"],
                        row["evaluator"],
                        row["status"],
                    )

    # ── Step 6: Always save a summary, then fail if results are incomplete ──
    settings = {
        "judge_provider": args.provider,
        "judge_model": args.model,
        "judge_temperature": judge_temperature,
        "judge_thinking": args.thinking,
        "judge_reasoning_effort": args.reasoning_effort,
        "max_tokens": args.max_tokens,
        "schema_retries": args.schema_retries,
        "workers": args.workers,
        "record_usage": args.record_usage,
        "validation_repair": args.validation_repair,
        "validation_repair_version": (
            VALIDATION_REPAIR_VERSION
            if args.validation_repair == "enabled"
            else None
        ),
        "task_shuffle_seed": args.task_shuffle_seed,
        "score_bands": args.score_bands,
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "prompt_sha256": prompt_hashes,
        "score_ranges": {
            "anticipation": [0, 10],
            "discovery": [0, 10],
            "calibration": [0, 10],
            "medical_quality": [0, 20],
            "total": [0, 50],
        },
    }
    summary = build_summary(records_by_method, current_rows, inputs, settings)
    write_json_atomic(summary_path, summary)
    markdown_path.write_text(build_markdown(summary), encoding="utf-8")
    logging.info("Saved summary to %s and %s", summary_path, markdown_path)

    if summary["counts"]["conversations_complete"] != summary["counts"][
        "conversations_requested"
    ]:
        raise RuntimeError("Some conversations have incomplete evaluator results")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    main()
