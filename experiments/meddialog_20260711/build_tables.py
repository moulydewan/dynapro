#!/usr/bin/env python3
"""Rebuild the four canonical MedDialog result tables from saved artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST_PATH = HERE / "manifest.json"
TABLE_DIR = HERE / "tables"

METHOD_ORDER = ["dynapro_medical", "dynapro", "proact", "none"]
SHORT_MODEL_LABEL = {"v4": "V4", "gpt": "GPT"}
VARIANT_ORDER = ["original", "original_repeat", "padded"]


def load_json(path: str | Path) -> Any:
    target = Path(path)
    if not target.is_absolute():
        target = ROOT / target
    with target.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: str) -> dict[str, dict[str, str]]:
    with (ROOT / path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {row["variant"]: row for row in rows}


def f(value: Any, digits: int) -> str:
    return f"{float(value):.{digits}f}"


def temperature_label(value: Any) -> str:
    if value is None:
        return "unset"
    return f(float(value), 1)


def score_values(aggregate: dict[str, Any]) -> list[str]:
    scores = aggregate["scores"]
    return [
        f(scores["anticipation"]["mean"], 2),
        f(scores["discovery"]["mean"], 2),
        f(scores["calibration"]["mean"], 2),
        f(scores["medical_quality"]["mean"], 2),
        f(aggregate["total_score"]["mean"], 2),
    ]


def verify_hashes(manifest: dict[str, Any]) -> None:
    entries: list[tuple[str, str]] = [
        (manifest["dataset"]["path"], manifest["dataset"]["sha256"]),
        (
            manifest["shared_generation_prompts"]["user_simulator"]["path"],
            manifest["shared_generation_prompts"]["user_simulator"]["sha256"],
        ),
    ]
    entries.extend(
        (method["assistant_prompt"], method["assistant_prompt_sha256"])
        for method in manifest["methods"]
        if method["assistant_prompt"]
    )
    entries.extend(
        (method["assistant_prompt"], method["assistant_prompt_sha256"])
        for method in manifest.get("excluded_methods", [])
        if method["assistant_prompt"]
    )
    entries.extend(
        (artifact["path"], artifact["sha256"])
        for artifact in manifest["generation_artifacts"]
    )

    for relative_path, expected in entries:
        digest = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(
                f"SHA-256 mismatch for {relative_path}: expected {expected}, got {digest}"
            )


def build_original_eval(
    manifest: dict[str, Any], method_labels: dict[str, str]
) -> tuple[list[str], list[list[str]]]:
    headers = [
        "Generator",
        "Method",
        "N",
        "Corpus BLEU-4",
        "Mean Sentence BLEU",
        "BERTScore F1",
        "BERTScore F1 SD",
        "Mean Turns",
        "Words / Turn",
        "Clarify Rate",
        "Source Summary",
    ]
    rows: list[list[str]] = []
    generator_labels = {
        key: value["label"]
        for key, value in manifest["generation_configurations"].items()
    }

    for source in manifest["evaluations"]["original_eval"]:
        payload = load_json(source["source_summary"])
        for item in source["rows"]:
            metrics = payload[item["source_key"]]
            rows.append(
                [
                    generator_labels[source["generator_key"]],
                    method_labels[item["method_key"]],
                    str(metrics["n_samples"]),
                    f(metrics["corpus_bleu"], 4),
                    f(metrics["mean_sentence_bleu"], 4),
                    f(metrics["bertscore"]["mean_f1"], 4),
                    f(metrics["bertscore"]["std_f1"], 4),
                    f(metrics["mean_num_turns"], 2),
                    f(metrics["mean_avg_turn_len"], 1),
                    f(metrics["clarify_rate"], 2),
                    source["source_summary"],
                ]
            )
    return headers, rows


def build_judge_2x2(
    manifest: dict[str, Any], method_labels: dict[str, str]
) -> tuple[list[str], list[list[str]]]:
    headers = [
        "Generator",
        "Judge",
        "Method",
        "N",
        "Anticipation /10",
        "Discovery /10",
        "Calibration /10",
        "Medical Quality /20",
        "Total /50",
        "Judge Temperature",
        "Judge Reasoning",
        "Status",
        "Source Summary",
    ]
    rows: list[list[str]] = []

    for source in manifest["evaluations"]["judge_2x2"]:
        payload = load_json(source["source_summary"])
        settings = payload["settings"]
        for method_key in source["method_keys"]:
            aggregate = payload["aggregate"][method_key]
            complete = aggregate["n_complete"] == aggregate["n_requested"]
            rows.append(
                [
                    SHORT_MODEL_LABEL[source["generation_key"]],
                    SHORT_MODEL_LABEL[source["judge_key"]],
                    method_labels[method_key],
                    str(aggregate["n_complete"]),
                    *score_values(aggregate),
                    temperature_label(settings.get("judge_temperature")),
                    settings.get("judge_reasoning_effort") or "unset",
                    "complete" if complete else "provisional",
                    source["source_summary"],
                ]
            )
    return headers, rows


def build_verbosity(manifest: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    headers = [
        "Judge",
        "Condition",
        "N",
        "Mean Words",
        "Length Ratio",
        "Anticipation /10",
        "Discovery /10",
        "Calibration /10",
        "Medical Quality /20",
        "Content /40",
        "Total /50",
        "Judge Temperature",
        "Judge Reasoning",
        "Status",
        "Source Summary",
        "Score Statistics",
        "Length Statistics",
    ]
    rows: list[list[str]] = []
    section = manifest["evaluations"]["verbosity_ablation"]
    ablation_manifest = load_json(section["manifest"])
    condition_by_variant = ablation_manifest["condition_by_variant"]

    for judge in section["judges"]:
        summary = load_json(judge["source_summary"])
        score_stats = read_csv(judge["score_statistics"])
        length_stats = read_csv(judge["length_statistics"])
        settings = summary["settings"]

        for variant in VARIANT_ORDER:
            score = score_stats[variant]
            length = length_stats[variant]
            source_aggregate = summary["aggregate"][condition_by_variant[variant]]
            if int(score["n"]) != source_aggregate["n_complete"]:
                raise ValueError(f"Verbosity N mismatch for {judge['judge_key']} {variant}")
            if abs(
                float(score["total_score_mean"])
                - float(source_aggregate["total_score"]["mean"])
            ) > 0.00051:
                raise ValueError(
                    f"Verbosity total mismatch for {judge['judge_key']} {variant}"
                )

            rows.append(
                [
                    judge["judge_label"],
                    section["condition_labels"][variant],
                    score["n"],
                    f(length["mean_visible_words"], 2),
                    f(length["mean_ratio_to_original"], 3),
                    f(score["anticipation_mean"], 2),
                    f(score["discovery_mean"], 2),
                    f(score["calibration_mean"], 2),
                    f(score["medical_quality_mean"], 2),
                    f(score["content_score_mean"], 2),
                    f(score["total_score_mean"], 2),
                    temperature_label(settings.get("judge_temperature")),
                    settings.get("judge_reasoning_effort") or "unset",
                    "complete",
                    judge["source_summary"],
                    judge["score_statistics"],
                    judge["length_statistics"],
                ]
            )
    return headers, rows


def build_temperature(manifest: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    headers = [
        "Generator",
        "Judge Configuration",
        "N Requested",
        "N Complete",
        "Judge Calls Successful",
        "Judge Calls Expected",
        "Anticipation /10",
        "Discovery /10",
        "Calibration /10",
        "Medical Quality /20",
        "Total /50",
        "Status",
        "Missing Result",
        "Source Summary",
    ]
    rows: list[list[str]] = []

    for source in manifest["evaluations"]["temperature_ablation"]:
        payload = load_json(source["source_summary"])
        aggregate = payload["aggregate"]["dynapro_medical"]
        requested = int(aggregate["n_requested"])
        complete = int(aggregate["n_complete"])
        expected_calls = requested * 4
        if complete == requested:
            successful_calls = expected_calls
        elif len(payload["methods"]) == 1:
            successful_calls = int(payload["counts"]["judge_calls_successful"])
        else:
            raise ValueError(
                "Cannot infer method-level successful calls from a partial multi-method summary"
            )
        status = source.get(
            "status", "complete" if complete == requested else "provisional"
        )

        rows.append(
            [
                SHORT_MODEL_LABEL[source["generation_key"]],
                source["configuration_label"],
                str(requested),
                str(complete),
                str(successful_calls),
                str(expected_calls),
                *score_values(aggregate),
                status,
                source.get("missing_result", ""),
                source["source_summary"],
            ]
        )
    return headers, rows


def render_csv(headers: list[str], rows: list[list[str]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(headers)
    writer.writerows(rows)
    return buffer.getvalue()


def validate_tables(tables: dict[str, tuple[list[str], list[list[str]]]]) -> None:
    expected_counts = {
        "original_eval.csv": 8,
        "judge_2x2.csv": 16,
        "verbosity_ablation.csv": 6,
        "temperature_ablation.csv": 8,
    }
    for filename, expected in expected_counts.items():
        actual = len(tables[filename][1])
        if actual != expected:
            raise ValueError(f"{filename}: expected {expected} rows, found {actual}")

    original = tables["original_eval.csv"][1]
    judge = tables["judge_2x2.csv"][1]
    verbosity = tables["verbosity_ablation.csv"][1]
    temperature = tables["temperature_ablation.csv"][1]

    assert original[0][5] == "0.8229"
    assert judge[0][8] == "38.97"
    assert not any(row[2] == "Generic" for row in judge)
    assert verbosity[-1][10] == "30.53"
    assert temperature[-1][3] == "99"
    assert temperature[-1][4:6] == ["399", "400"]
    assert temperature[-1][10:12] == ["40.12", "provisional"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that committed CSV files match the canonical sources without rewriting them.",
    )
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Skip SHA-256 validation of the dataset, prompts, and generation artifacts.",
    )
    args = parser.parse_args()

    manifest = load_json(MANIFEST_PATH)
    if not args.skip_hash_check:
        verify_hashes(manifest)

    method_labels = {method["key"]: method["label"] for method in manifest["methods"]}
    if list(method_labels) != METHOD_ORDER:
        raise ValueError("Canonical methods are missing or out of order in manifest.json")

    tables = {
        "original_eval.csv": build_original_eval(manifest, method_labels),
        "judge_2x2.csv": build_judge_2x2(manifest, method_labels),
        "verbosity_ablation.csv": build_verbosity(manifest),
        "temperature_ablation.csv": build_temperature(manifest),
    }
    validate_tables(tables)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rendered = {
        filename: render_csv(headers, rows)
        for filename, (headers, rows) in tables.items()
    }
    if args.check:
        mismatches = [
            filename
            for filename, expected in rendered.items()
            if not (TABLE_DIR / filename).exists()
            or (TABLE_DIR / filename).read_text(encoding="utf-8") != expected
        ]
        if mismatches:
            raise SystemExit("Out-of-date tables: " + ", ".join(mismatches))
        print("All four tables match their canonical sources.")
        return

    for filename, content in rendered.items():
        (TABLE_DIR / filename).write_text(content, encoding="utf-8")
        print(f"wrote {TABLE_DIR / filename} ({len(tables[filename][1])} rows)")


if __name__ == "__main__":
    main()
