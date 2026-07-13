#!/usr/bin/env bash
set -euo pipefail

# Runs exactly the four DynaPro + Medical temperature-ablation cells over the
# two frozen generator corpora. The judge runner resumes completed evaluator
# tasks from each output directory if this script is interrupted.

cd "$(dirname "$0")/../../.."
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/cert.pem}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

V4_INPUT="output/thinking_ablation/dynapro_medical_v4flash_prompt98e58426_thinking_on_000_099_20260712/meddialog_dynapro_medical_deepseek-v4-flash-thinking_00_99.json"
GPT_INPUT="output/openai_simulations/gpt56luna_high_dynapromedical98e58426_dynapro1c64335f_proactb75ee522_none_workers4_completed100_000_099_20260712/meddialog_dynapro_medical_gpt-5.6-luna-reasoning-high_00_99_complete.json"
OUT_ROOT="output/judgments_temperature_ablation"

V4GEN_V4_OUT="$OUT_ROOT/v4gen_dynapro_medical_eval_v4flash_temp1_high_banded_000_099_20260713"
GPTGEN_V4_OUT="$OUT_ROOT/gptgen_dynapro_medical_eval_v4flash_temp1_high_banded_000_099_20260713"
V4GEN_GPT_OUT="$OUT_ROOT/v4gen_dynapro_medical_eval_gpt56luna_temp0_reasoning_none_banded_000_099_20260713"
GPTGEN_GPT_OUT="$OUT_ROOT/gptgen_dynapro_medical_eval_gpt56luna_temp0_reasoning_none_banded_000_099_20260713"

run_judge() {
  # The judge itself does not use boto3/tqdm. Stub those optional simulation
  # imports so this works in the lightweight analysis Python environment.
  "$PYTHON_BIN" -c '
import runpy
import sys
import types

boto3 = types.ModuleType("boto3")
boto3.client = lambda *args, **kwargs: None
sys.modules["boto3"] = boto3

tqdm = types.ModuleType("tqdm")
tqdm.tqdm = lambda values, *args, **kwargs: values
sys.modules["tqdm"] = tqdm

runpy.run_path("judge_meddialog_four_evals.py", run_name="__main__")
' "$@"
}

COMMON_ARGS=(
  --thinking enabled
  --schema-retries 3
  --max-tokens 8192
  --record-usage
  --task-shuffle-seed 20260712
  --score-bands enabled
)

# V4 judge: high reasoning, explicit temperature 1.0.
run_judge \
  --input "dynapro_medical=$V4_INPUT" \
  --provider deepseek \
  --model deepseek-v4-flash \
  --temperature 1.0 \
  --reasoning-effort high \
  --workers 20 \
  --output-dir "$V4GEN_V4_OUT" \
  "${COMMON_ARGS[@]}"

run_judge \
  --input "dynapro_medical=$GPT_INPUT" \
  --provider deepseek \
  --model deepseek-v4-flash \
  --temperature 1.0 \
  --reasoning-effort high \
  --workers 20 \
  --output-dir "$GPTGEN_V4_OUT" \
  "${COMMON_ARGS[@]}"

# GPT-5.6 Luna rejects temperature with high reasoning. The accepted fallback
# is no reasoning plus explicit temperature 0.0, so this changes two settings.
run_judge \
  --input "dynapro_medical=$V4_INPUT" \
  --provider openai \
  --model gpt-5.6-luna \
  --temperature 0 \
  --reasoning-effort none \
  --workers 4 \
  --output-dir "$V4GEN_GPT_OUT" \
  "${COMMON_ARGS[@]}"

run_judge \
  --input "dynapro_medical=$GPT_INPUT" \
  --provider openai \
  --model gpt-5.6-luna \
  --temperature 0 \
  --reasoning-effort none \
  --validation-repair enabled \
  --workers 4 \
  --output-dir "$GPTGEN_GPT_OUT" \
  "${COMMON_ARGS[@]}"

"$PYTHON_BIN" - \
  "$V4GEN_V4_OUT/four_eval_summary.json" \
  "$V4GEN_GPT_OUT/four_eval_summary.json" \
  "$GPTGEN_V4_OUT/four_eval_summary.json" \
  "$GPTGEN_GPT_OUT/four_eval_summary.json" <<'PY'
import json
import sys

labels = (
    "V4 generation -> V4 judge (temp 1, high)",
    "V4 generation -> GPT judge (temp 0, none)",
    "GPT generation -> V4 judge (temp 1, high)",
    "GPT generation -> GPT judge (temp 0, none)",
)
print("\nCondition\tAnticipation\tDiscovery\tCalibration\tMedical\tTotal")
for label, path in zip(labels, sys.argv[1:]):
    summary = json.load(open(path, encoding="utf-8"))
    counts = summary["counts"]
    if counts["judge_calls_successful"] != 400:
        raise RuntimeError(f"Incomplete result: {path}: {counts}")
    row = summary["tables"]["all_available"][0]
    print(
        f"{label}\t{row['anticipation']:.2f}\t{row['discovery']:.2f}\t"
        f"{row['calibration']:.2f}\t{row['medical_quality']:.2f}\t"
        f"{row['total_score']:.2f}"
    )
PY
