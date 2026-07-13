#!/usr/bin/env python3
"""Compare completed GPT-5.6 Luna low/high MedDialog trajectories."""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SIM = ROOT / "output" / "openai_simulations"
JUDGE = ROOT / "output" / "judgments_openai_generation"

LOW_SIM = {
    "dynapro_medical": SIM / "dynapro_medical_gpt56luna_low_prompt98e58426_workers4_v2_completed100_000_099_20260712" / "meddialog_dynapro_medical_gpt-5.6-luna-reasoning-low_00_99_complete.json",
    "dynapro": SIM / "gpt56luna_low_dynapro1c64335f_proactb75ee522_none_workers4_completed100_000_099_20260712" / "meddialog_dynapro_gpt-5.6-luna-reasoning-low_00_99_complete.json",
    "proact": SIM / "gpt56luna_low_dynapro1c64335f_proactb75ee522_none_workers4_completed100_000_099_20260712" / "meddialog_proact_gpt-5.6-luna-reasoning-low_00_99_complete.json",
    "none": SIM / "gpt56luna_low_dynapro1c64335f_proactb75ee522_none_workers4_completed100_000_099_20260712" / "meddialog_none_gpt-5.6-luna-reasoning-low_00_99_complete.json",
}
HIGH_DIR = SIM / "gpt56luna_high_dynapromedical98e58426_dynapro1c64335f_proactb75ee522_none_workers4_completed100_000_099_20260712"
HIGH_SIM = {
    "dynapro_medical": HIGH_DIR / "meddialog_dynapro_medical_gpt-5.6-luna-reasoning-high_00_99_complete.json",
    "dynapro": HIGH_DIR / "meddialog_dynapro_gpt-5.6-luna-reasoning-high_00_99_complete.json",
    "proact": HIGH_DIR / "meddialog_proact_gpt-5.6-luna-reasoning-high_00_99_complete.json",
    "none": HIGH_DIR / "meddialog_none_gpt-5.6-luna-reasoning-high_00_99_complete.json",
}
LOW_JUDGE = [
    JUDGE / "dynapro_medical_gpt56luna_low_prompt98e58426_completed100_eval_v4flash_thinking_high_banded_workers40_000_099_20260712" / "four_eval_details.jsonl",
    JUDGE / "gpt56luna_low_dynapro_proact_none_completed100_eval_v4flash_thinking_high_banded_workers40_000_099_20260712" / "four_eval_details.jsonl",
]
HIGH_JUDGE = [
    JUDGE / "gpt56luna_high_eval_v4flash_thinking_high_banded_000_099_20260712" / "four_eval_details.jsonl"
]

METHOD_LABELS = {
    "dynapro_medical": "DynaPro + Medical",
    "dynapro": "Original DynaPro",
    "proact": "Proact Instruction",
    "none": "Baseline / No Prompt",
}
DIMENSIONS = ("anticipation", "discovery", "calibration", "medical_quality")
WORD_RE = re.compile(r"\b[\w’'-]+\b")
LIST_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)
BREVITY_RE = re.compile(r"\b(short|brief|concise|simple|few sentences|not too much detail)\b", re.I)


def load_records(path: Path) -> dict[int, dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["source_index"]): row for row in records}


def load_judgments(paths: list[Path]) -> dict[tuple[str, int, str], dict]:
    out: dict[tuple[str, int, str], dict] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["status"] == "ok":
                    out[(row["method"], int(row["source_index"]), row["evaluator"])] = row["parsed_result"]
    return out


def words(text: str) -> int:
    return len(WORD_RE.findall(text))


def summarize_record(record: dict) -> dict[str, float | int | bool]:
    assistants = [m["content"] for m in record["conversation"] if m["role"] == "assistant"]
    users = [m["content"] for m in record["conversation"] if m["role"] == "user"]
    assistant_words = [words(text) for text in assistants]
    return {
        "assistant_turns": len(assistants),
        "assistant_total_words": sum(assistant_words),
        "assistant_first_words": assistant_words[0],
        "assistant_question_turns": sum("?" in text for text in assistants),
        "assistant_question_marks": sum(text.count("?") for text in assistants),
        "assistant_list_turns": sum(bool(LIST_RE.search(text)) for text in assistants),
        "assistant_list_lines": sum(len(LIST_RE.findall(text)) for text in assistants),
        "user_followup_words": sum(words(text) for text in users[1:]),
        "early_termination": len(assistants) < 3,
    }


def mean(rows: list[dict], key: str) -> float:
    return statistics.mean(float(row[key]) for row in rows)


def correlation(xs: list[float], ys: list[float]) -> float:
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in xs) * sum((y - y_mean) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else 0.0


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    low_judge = load_judgments(LOW_JUDGE)
    high_judge = load_judgments(HIGH_JUDGE)
    structural_rows: list[dict] = []
    comparability_rows: list[dict] = []
    case_rows: list[dict] = []
    judge_rows: list[dict] = []

    for method in METHOD_LABELS:
        low = load_records(LOW_SIM[method])
        high = load_records(HIGH_SIM[method])
        if sorted(low) != list(range(100)) or sorted(high) != list(range(100)):
            raise ValueError(f"{method}: expected source_index 0-99")

        # Reuse the same structural summaries in the aggregate and paired tables.
        low_summaries = [summarize_record(low[i]) for i in range(100)]
        high_summaries = [summarize_record(high[i]) for i in range(100)]
        for effort, effort_summaries in (("low", low_summaries), ("high", high_summaries)):
            turns = [int(row["assistant_turns"]) for row in effort_summaries]
            structural_rows.append({
                "method": method,
                "method_label": METHOD_LABELS[method],
                "reasoning": effort,
                "n": 100,
                "mean_assistant_turns": round(mean(effort_summaries, "assistant_turns"), 4),
                "one_turn_conversations": turns.count(1),
                "two_turn_conversations": turns.count(2),
                "three_turn_conversations": turns.count(3),
                "early_termination_rate": round(mean(effort_summaries, "early_termination"), 4),
                "mean_total_assistant_words": round(mean(effort_summaries, "assistant_total_words"), 4),
                "median_total_assistant_words": statistics.median(float(r["assistant_total_words"]) for r in effort_summaries),
                "mean_first_response_words": round(mean(effort_summaries, "assistant_first_words"), 4),
                "mean_user_followup_words": round(mean(effort_summaries, "user_followup_words"), 4),
                "question_turn_rate": round(sum(float(r["assistant_question_turns"]) for r in effort_summaries) / sum(float(r["assistant_turns"]) for r in effort_summaries), 4),
                "conversations_with_question_rate": round(sum(bool(r["assistant_question_turns"]) for r in effort_summaries) / 100, 4),
                "question_marks": int(sum(float(r["assistant_question_marks"]) for r in effort_summaries)),
                "list_turn_rate": round(sum(float(r["assistant_list_turns"]) for r in effort_summaries) / sum(float(r["assistant_turns"]) for r in effort_summaries), 4),
                "list_lines_per_1000_assistant_words": round(1000 * sum(float(r["assistant_list_lines"]) for r in effort_summaries) / sum(float(r["assistant_total_words"]) for r in effort_summaries), 4),
            })

        exact = 0
        seed_exact = 0
        ratios: list[float] = []
        low_brevity = 0
        high_brevity = 0
        for index in range(100):
            low_first = low[index]["conversation"][0]["content"]
            high_first = high[index]["conversation"][0]["content"]
            exact += low_first == high_first
            seed_exact += low[index]["single_turn_prompt"] == high[index]["single_turn_prompt"]
            ratios.append(difflib.SequenceMatcher(None, low_first, high_first).ratio())
            low_metrics = low_summaries[index]
            high_metrics = high_summaries[index]
            low_brevity += bool(BREVITY_RE.search(low_first))
            high_brevity += bool(BREVITY_RE.search(high_first))

            deltas = {
                dim: high_judge[(method, index, dim)]["score"] - low_judge[(method, index, dim)]["score"]
                for dim in DIMENSIONS
            }
            case_rows.append({
                "method": method,
                "source_index": index,
                "initial_user_exact_match": low_first == high_first,
                "initial_user_sequence_similarity": round(ratios[-1], 6),
                "assistant_turn_delta": int(high_metrics["assistant_turns"]) - int(low_metrics["assistant_turns"]),
                "assistant_total_word_delta": int(high_metrics["assistant_total_words"]) - int(low_metrics["assistant_total_words"]),
                "first_response_word_delta": int(high_metrics["assistant_first_words"]) - int(low_metrics["assistant_first_words"]),
                "question_turn_delta": int(high_metrics["assistant_question_turns"]) - int(low_metrics["assistant_question_turns"]),
                "anticipation_delta": deltas["anticipation"],
                "discovery_delta": deltas["discovery"],
                "calibration_delta": deltas["calibration"],
                "medical_quality_delta": deltas["medical_quality"],
                "total_judge_delta": sum(deltas.values()),
            })

        comparability_rows.append({
            "method": method,
            "method_label": METHOD_LABELS[method],
            "paired_seeds_exact": seed_exact,
            "initial_user_messages_exact": exact,
            "mean_initial_user_sequence_similarity": round(statistics.mean(ratios), 4),
            "median_initial_user_sequence_similarity": round(statistics.median(ratios), 4),
            "low_initial_brevity_requests": low_brevity,
            "high_initial_brevity_requests": high_brevity,
        })

        method_cases = [row for row in case_rows if row["method"] == method]
        turn_deltas = [float(row["assistant_turn_delta"]) for row in method_cases]
        discovery_deltas = [float(row["discovery_delta"]) for row in method_cases]
        total_deltas = [float(row["total_judge_delta"]) for row in method_cases]
        total_se = statistics.stdev(total_deltas) / math.sqrt(len(total_deltas))
        judge_rows.append({
            "method": method,
            "method_label": METHOD_LABELS[method],
            "n": 100,
            "mean_anticipation_delta": round(statistics.mean(float(row["anticipation_delta"]) for row in method_cases), 4),
            "mean_discovery_delta": round(statistics.mean(discovery_deltas), 4),
            "mean_calibration_delta": round(statistics.mean(float(row["calibration_delta"]) for row in method_cases), 4),
            "mean_medical_quality_delta": round(statistics.mean(float(row["medical_quality_delta"]) for row in method_cases), 4),
            "mean_total_judge_delta": round(statistics.mean(total_deltas), 4),
            "total_delta_ci95_low": round(statistics.mean(total_deltas) - 1.984 * total_se, 4),
            "total_delta_ci95_high": round(statistics.mean(total_deltas) + 1.984 * total_se, 4),
            "turn_delta_discovery_correlation": round(correlation(turn_deltas, discovery_deltas), 4),
            "more_turn_pairs": sum(delta > 0 for delta in turn_deltas),
            "same_turn_pairs": sum(delta == 0 for delta in turn_deltas),
            "fewer_turn_pairs": sum(delta < 0 for delta in turn_deltas),
            "mean_discovery_delta_more_turns": round(statistics.mean(float(row["discovery_delta"]) for row in method_cases if float(row["assistant_turn_delta"]) > 0), 4),
            "mean_discovery_delta_same_turns": round(statistics.mean(float(row["discovery_delta"]) for row in method_cases if float(row["assistant_turn_delta"]) == 0), 4),
            "mean_discovery_delta_fewer_turns": round(statistics.mean(float(row["discovery_delta"]) for row in method_cases if float(row["assistant_turn_delta"]) < 0), 4),
        })

    write_csv(args.output_dir / "structural_summary.csv", structural_rows)
    write_csv(args.output_dir / "initial_prompt_comparability.csv", comparability_rows)
    write_csv(args.output_dir / "paired_case_deltas.csv", case_rows)
    write_csv(args.output_dir / "judge_and_turn_diagnostics.csv", judge_rows)
    summary = {
        "population": "source_index 0-99, same MedDialog seeds, four methods",
        "comparison": "GPT-5.6 Luna reasoning high versus low for the whole simulation pipeline",
        "structural_summary": structural_rows,
        "initial_prompt_comparability": comparability_rows,
        "judge_and_turn_diagnostics": judge_rows,
        "case_delta_rows": len(case_rows),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
