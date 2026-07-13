#!/usr/bin/env python3
"""Compare GPT-5.6 Luna high and DeepSeek V4 Flash thinking/high generators."""

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
OUT = ROOT / "output"
GPT_DIR = OUT / "openai_simulations" / "gpt56luna_high_dynapromedical98e58426_dynapro1c64335f_proactb75ee522_none_workers4_completed100_000_099_20260712"

SIMULATIONS = {
    "dynapro_medical": {
        "v4": OUT / "thinking_ablation" / "dynapro_medical_v4flash_prompt98e58426_thinking_on_000_099_20260712" / "meddialog_dynapro_medical_deepseek-v4-flash-thinking_00_99.json",
        "gpt": GPT_DIR / "meddialog_dynapro_medical_gpt-5.6-luna-reasoning-high_00_99_complete.json",
    },
    "dynapro": {
        "v4": OUT / "simulations" / "v4flash_dynapro_vs_medical_fc97fb17_000_099_20260712" / "meddialog_dynapro_deepseek-v4-flash-thinking_00_99.json",
        "gpt": GPT_DIR / "meddialog_dynapro_gpt-5.6-luna-reasoning-high_00_99_complete.json",
    },
    "proact": {
        "v4": OUT / "simulations" / "v4flash_3methods_000_099_20260712" / "meddialog_proact_deepseek-v4-flash-thinking_00_99.json",
        "gpt": GPT_DIR / "meddialog_proact_gpt-5.6-luna-reasoning-high_00_99_complete.json",
    },
    "none": {
        "v4": OUT / "simulations" / "v4flash_3methods_000_099_20260712" / "meddialog_none_deepseek-v4-flash-thinking_00_99.json",
        "gpt": GPT_DIR / "meddialog_none_gpt-5.6-luna-reasoning-high_00_99_complete.json",
    },
}

GPT_EVAL = OUT / "eval" / "gpt56luna_high_four_methods_completed100_all_metrics_20260712" / "comparison.json"
V4_EVAL_ALL = OUT / "eval" / "v4flash_five_methods_evalpy_all_metrics_20260712" / "comparison.json"
V4_EVAL_MED = OUT / "eval" / "v4flash_dynapro_medical_prompt98_evalpy_all_metrics_20260712" / "comparison.json"
GPT_JUDGE = OUT / "judgments_openai_generation" / "gpt56luna_high_eval_v4flash_thinking_high_banded_000_099_20260712" / "four_eval_summary.json"
V4_JUDGE_ALL = OUT / "judgments_score_band_ablation" / "five_methods_banded_v4flash_thinking_high_000_099_20260712" / "four_eval_summary.json"
V4_JUDGE_MED = OUT / "judgments_thinking_ablation" / "dynapro_medical_v4flash_prompt98e58426_simthink_on_evalthink_on_banded_000_099_20260712" / "four_eval_summary.json"

LABELS = {
    "dynapro_medical": "DynaPro + Medical",
    "dynapro": "Original DynaPro",
    "proact": "Proact Instruction",
    "none": "Baseline / No Prompt",
}
DIMENSIONS = ("anticipation", "discovery", "calibration", "medical_quality")
WORD_RE = re.compile(r"\b[\w’'-]+\b")
LIST_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def records(path: Path) -> dict[int, dict]:
    return {int(row["source_index"]): row for row in json.loads(path.read_text())}


def summarize(record: dict) -> dict:
    assistant = [m["content"] for m in record["conversation"] if m["role"] == "assistant"]
    word_counts = [len(WORD_RE.findall(text)) for text in assistant]
    return {
        "turns": len(assistant),
        "total_words": sum(word_counts),
        "first_words": word_counts[0],
        "question_turns": sum("?" in text for text in assistant),
        "list_turns": sum(bool(LIST_RE.search(text)) for text in assistant),
    }


def judge_cases(summary: dict, method: str) -> dict[int, dict]:
    return {
        int(row["source_index"]): row
        for row in summary["per_case"]
        if row["method"] == method and row["status"] == "ok"
    }


def mean(values) -> float:
    return statistics.mean(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gpt_eval = json.loads(GPT_EVAL.read_text())
    v4_eval_all = json.loads(V4_EVAL_ALL.read_text())
    v4_eval_med = json.loads(V4_EVAL_MED.read_text())["dynapro_medical_v4flash_prompt98"]
    gpt_judge = json.loads(GPT_JUDGE.read_text())
    v4_judge_all = json.loads(V4_JUDGE_ALL.read_text())
    v4_judge_med = json.loads(V4_JUDGE_MED.read_text())

    generator_rows, delta_rows, judge_rows, comparable_rows, case_rows = [], [], [], [], []
    for method in LABELS:
        v4, gpt = records(SIMULATIONS[method]["v4"]), records(SIMULATIONS[method]["gpt"])
        if sorted(v4) != list(range(100)) or sorted(gpt) != list(range(100)):
            raise ValueError(f"{method}: expected source_index 0-99")
        if not all(v4[i]["single_turn_prompt"] == gpt[i]["single_turn_prompt"] for i in range(100)):
            raise ValueError(f"{method}: source prompts differ")

        eval_key = "baseline" if method == "none" else method
        v4_eval = v4_eval_med if method == "dynapro_medical" else v4_eval_all[eval_key]
        summaries = {}
        method_generator_rows = []
        for model, data, eval_row in (
            ("V4 Flash High", v4, v4_eval),
            ("GPT-5.6 High", gpt, gpt_eval[eval_key]),
        ):
            rows = [summarize(data[i]) for i in range(100)]
            summaries[model] = rows
            total_turns = sum(row["turns"] for row in rows)
            summary_row = {
                "method": method,
                "method_label": LABELS[method],
                "generator": model,
                "n": 100,
                "mean_turns": round(mean(row["turns"] for row in rows), 4),
                "early_termination_rate": round(mean(row["turns"] < 3 for row in rows), 4),
                "mean_total_words": round(mean(row["total_words"] for row in rows), 4),
                "mean_first_words": round(mean(row["first_words"] for row in rows), 4),
                "question_turn_rate": round(sum(row["question_turns"] for row in rows) / total_turns, 4),
                "conversations_with_question_rate": round(mean(row["question_turns"] > 0 for row in rows), 4),
                "list_turn_rate": round(sum(row["list_turns"] for row in rows) / total_turns, 4),
                "bertscore_f1": eval_row["bertscore"]["mean_f1"],
                "corpus_bleu": eval_row["corpus_bleu"],
                "evalpy_clarify_rate": eval_row["clarify_rate"],
            }
            generator_rows.append(summary_row)
            method_generator_rows.append(summary_row)

        v4_rows, gpt_rows = summaries["V4 Flash High"], summaries["GPT-5.6 High"]
        v4_summary_row, gpt_summary_row = method_generator_rows
        delta_rows.append({
            "method": method,
            "method_label": LABELS[method],
            "turn_delta": round(mean(gpt_rows[i]["turns"] - v4_rows[i]["turns"] for i in range(100)), 4),
            "total_word_delta": round(mean(gpt_rows[i]["total_words"] - v4_rows[i]["total_words"] for i in range(100)), 4),
            "first_word_delta": round(mean(gpt_rows[i]["first_words"] - v4_rows[i]["first_words"] for i in range(100)), 4),
            "question_turn_rate_delta": round(gpt_summary_row["question_turn_rate"] - v4_summary_row["question_turn_rate"], 4),
            "early_termination_rate_delta": round(gpt_summary_row["early_termination_rate"] - v4_summary_row["early_termination_rate"], 4),
            "bertscore_f1_delta": round(gpt_summary_row["bertscore_f1"] - v4_summary_row["bertscore_f1"], 4),
        })

        exact_initial = sum(v4[i]["conversation"][0]["content"] == gpt[i]["conversation"][0]["content"] for i in range(100))
        initial_similarity = mean(difflib.SequenceMatcher(None, v4[i]["conversation"][0]["content"], gpt[i]["conversation"][0]["content"]).ratio() for i in range(100))
        comparable_rows.append({
            "method": method,
            "method_label": LABELS[method],
            "same_source_prompts": 100,
            "exact_same_initial_user_messages": exact_initial,
            "mean_initial_user_sequence_similarity": round(initial_similarity, 4),
            "v4_user_temperature": v4[0]["user_temperature"],
            "v4_assistant_temperature": v4[0]["assistant_temperature"],
            "gpt_user_temperature": gpt[0]["user_temperature"],
            "gpt_assistant_temperature": gpt[0]["assistant_temperature"],
        })

        v4_summary = v4_judge_med if method == "dynapro_medical" else v4_judge_all
        v4_case, gpt_case = judge_cases(v4_summary, method), judge_cases(gpt_judge, method)
        deltas = {dimension: [] for dimension in DIMENSIONS}
        total_deltas = []
        for index in range(100):
            row = {"method": method, "source_index": index}
            for dimension in DIMENSIONS:
                value = gpt_case[index]["scores"][dimension] - v4_case[index]["scores"][dimension]
                deltas[dimension].append(value)
                row[f"{dimension}_delta"] = value
            row["total_delta"] = gpt_case[index]["total_score"] - v4_case[index]["total_score"]
            total_deltas.append(row["total_delta"])
            case_rows.append(row)
        se = statistics.stdev(total_deltas) / math.sqrt(100)
        gpt_agg = gpt_judge["aggregate"][method]
        v4_agg = v4_summary["aggregate"][method]
        judge_rows.append({
            "method": method,
            "method_label": LABELS[method],
            "v4_total": v4_agg["total_score"]["mean"],
            "gpt_total": gpt_agg["total_score"]["mean"],
            "total_delta": round(mean(total_deltas), 4),
            "total_ci95_low": round(mean(total_deltas) - 1.984 * se, 4),
            "total_ci95_high": round(mean(total_deltas) + 1.984 * se, 4),
            "anticipation_delta": round(mean(deltas["anticipation"]), 4),
            "discovery_delta": round(mean(deltas["discovery"]), 4),
            "calibration_delta": round(mean(deltas["calibration"]), 4),
            "medical_quality_delta": round(mean(deltas["medical_quality"]), 4),
            "v4_medical_issue_rate": v4_agg["medical_issue_summary"]["any_issue_rate"],
            "gpt_medical_issue_rate": gpt_agg["medical_issue_summary"]["any_issue_rate"],
        })

    write_csv(args.output_dir / "generator_summary.csv", generator_rows)
    write_csv(args.output_dir / "generator_deltas.csv", delta_rows)
    write_csv(args.output_dir / "judge_deltas.csv", judge_rows)
    write_csv(args.output_dir / "comparability.csv", comparable_rows)
    write_csv(args.output_dir / "paired_judge_case_deltas.csv", case_rows)
    summary = {
        "comparison": "GPT-5.6 Luna high minus DeepSeek V4 Flash thinking/high",
        "scope": "four methods, source_index 0-99, same MedDialog source prompts",
        "generator_summary": generator_rows,
        "generator_deltas": delta_rows,
        "judge_deltas": judge_rows,
        "comparability": comparable_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
