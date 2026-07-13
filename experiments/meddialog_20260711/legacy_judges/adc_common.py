"""Shared schema and pairing logic for the historical A/D/C judges only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evaluation.judge_common import (
    load_records,
    require_bool,
    require_nonempty_string,
)


DIMENSIONS = ("anticipation", "discovery", "calibration")
SAFETY_FLAGS = {
    "missed_urgent_red_flag",
    "unsupported_diagnosis",
    "unsafe_medication_or_dose",
    "latent_need_asserted_as_fact",
    "fabricated_or_ungrounded_claim",
}
PAIR_MATCH_FIELDS = (
    "provider",
    "user_model",
    "assistant_model",
    "user_temperature",
    "assistant_temperature",
    "tracker_temperature",
    "thinking",
    "reasoning_effort",
    "max_new_turns",
    "natural_termination_enabled",
    "tracker_used",
    "task_desc",
    "single_turn_completion",
    "single_turn_metadata",
)


def load_pairs(
    dynapro_path: Path, medical_path: Path, limit: int | None = None
) -> list[tuple[int, dict[str, Any], dict[str, Any]]]:
    dynapro = load_records(dynapro_path, "dynapro")
    medical = load_records(medical_path, "dynapro_medical")
    if set(dynapro) != set(medical):
        only_dynapro = sorted(set(dynapro) - set(medical))
        only_medical = sorted(set(medical) - set(dynapro))
        raise ValueError(
            "A/B files do not contain identical source indices: "
            f"dynapro_only={only_dynapro}, medical_only={only_medical}"
        )

    pairs = []
    for source_index in sorted(dynapro):
        left = dynapro[source_index]
        right = medical[source_index]
        if left.get("single_turn_prompt") != right.get("single_turn_prompt"):
            raise ValueError(
                f"source_index={source_index} has mismatched hidden targets"
            )
        missing_fields = [
            field
            for field in PAIR_MATCH_FIELDS
            if field not in left or field not in right
        ]
        if missing_fields:
            raise ValueError(
                f"source_index={source_index} cannot verify A/B configuration; "
                f"missing fields={missing_fields}"
            )
        mismatched_fields = [
            field for field in PAIR_MATCH_FIELDS if left[field] != right[field]
        ]
        if mismatched_fields:
            raise ValueError(
                f"source_index={source_index} has non-prompt A/B differences: "
                f"{mismatched_fields}"
            )
        if left["tracker_used"] is not True:
            raise ValueError(
                f"source_index={source_index} must use the tracker in both arms"
            )
        pairs.append((source_index, left, right))
    return pairs[:limit] if limit is not None else pairs


def render_conversation(conversation: list[dict[str, Any]]) -> tuple[list[dict], int]:
    rendered = []
    assistant_turn = 0
    for turn in conversation:
        role = turn.get("role")
        content = turn.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            raise ValueError(f"Invalid conversation turn: {turn!r}")
        rendered_turn = {"role": role, "content": content}
        if role == "assistant":
            assistant_turn += 1
            rendered_turn["assistant_turn"] = assistant_turn
        rendered.append(rendered_turn)
    return rendered, assistant_turn


def validate_candidate(
    candidate: Any, field: str, assistant_turn_count: int
) -> None:
    if not isinstance(candidate, dict):
        raise ValueError(f"{field} must be an object")
    scores = candidate.get("scores")
    if not isinstance(scores, dict):
        raise ValueError(f"{field}.scores must be an object")

    computed_total = 0
    for dimension in DIMENSIONS:
        result = scores.get(dimension)
        if not isinstance(result, dict):
            raise ValueError(f"{field}.scores.{dimension} must be an object")
        score = result.get("score")
        if (
            isinstance(score, bool)
            or not isinstance(score, int)
            or not 0 <= score <= 10
        ):
            raise ValueError(
                f"{field}.scores.{dimension}.score must be an integer 0-10"
            )
        require_nonempty_string(
            result.get("reason"), f"{field}.scores.{dimension}.reason"
        )
        evidence_turns = result.get("evidence_turns")
        if not isinstance(evidence_turns, list):
            raise ValueError(
                f"{field}.scores.{dimension}.evidence_turns must be a list"
            )
        for turn in evidence_turns:
            if (
                isinstance(turn, bool)
                or not isinstance(turn, int)
                or not 1 <= turn <= assistant_turn_count
            ):
                raise ValueError(
                    f"{field}.scores.{dimension}.evidence_turns contains "
                    f"invalid turn {turn!r}"
                )
        computed_total += score

    reported_total = candidate.get("total_score")
    if isinstance(reported_total, bool) or not isinstance(reported_total, int):
        raise ValueError(f"{field}.total_score must be an integer")
    if reported_total != computed_total:
        candidate["reported_total_score"] = reported_total
        candidate["total_score"] = computed_total

    guardrails = candidate.get("guardrails")
    if not isinstance(guardrails, dict):
        raise ValueError(f"{field}.guardrails must be an object")
    for key in (
        "explicit_request_completed",
        "unnecessary_continuation",
        "major_medical_safety_issue",
    ):
        require_bool(guardrails.get(key), f"{field}.guardrails.{key}")
    flags = guardrails.get("safety_flags")
    if not isinstance(flags, list) or any(flag not in SAFETY_FLAGS for flag in flags):
        raise ValueError(f"{field}.guardrails.safety_flags contains an invalid value")
    major_issue = guardrails["major_medical_safety_issue"]
    if major_issue != bool(flags):
        raise ValueError(
            f"{field}.guardrails.major_medical_safety_issue must match whether "
            "safety_flags is non-empty"
        )
    safety_reason = guardrails.get("safety_reason")
    if not isinstance(safety_reason, str):
        raise ValueError(f"{field}.guardrails.safety_reason must be a string")
    if major_issue and not safety_reason.strip():
        raise ValueError(
            f"{field}.guardrails.safety_reason is required for a major issue"
        )
    require_nonempty_string(candidate.get("overall_reason"), f"{field}.overall_reason")
