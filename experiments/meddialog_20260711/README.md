# MedDialog experiment registry (July 11, 2026)

This directory is the stable index for the MedDialog experiments described in the July 11 notes. The trajectories and judge details were created on July 12–13. No existing file under `output/` was moved, renamed, or deleted.

The Git-shareable canonical simulation trajectories and their per-sample
`evaluate.py` results are in [`results/`](results/README.md). This bundle keeps
only the eight matched generator-method arms and excludes smoke, retry, legacy,
and superseded JSON files.

## Quick use

The four report-ready tables are in [`tables/`](tables/):

- [`original_eval.csv`](tables/original_eval.csv): 2 generators × 4 methods = 8 rows
- [`judge_2x2.csv`](tables/judge_2x2.csv): 2 generators × 2 judges × 4 methods = 16 rows
- [`verbosity_ablation.csv`](tables/verbosity_ablation.csv): 2 judges × 3 conditions = 6 rows
- [`temperature_ablation.csv`](tables/temperature_ablation.csv): 2 generators × 4 judge configurations = 8 rows

Every row includes its canonical source path. Rebuild and verify them with:

```bash
python3 experiments/meddialog_20260711/build_tables.py
python3 experiments/meddialog_20260711/build_tables.py --check
```

The builder uses only the Python standard library. By default, it also verifies the SHA-256 hashes of the dataset, prompts, and eight canonical generation files.

[`manifest.json`](manifest.json) is the machine-readable registry. It freezes the dataset, method names, prompt hashes, generator files, evaluator sources, and known caveats.

## Code layout

The repository's original tracked layout is intentionally unchanged. The active conversation runner and active four-evaluator judge remain at repository root. Shared judge infrastructure now lives in `evaluation/judge_common.py`; it contains no scoring protocol. Only dated experiment code and historical judges are grouped here:

```text
evaluation/
├── judge_common.py
└── prompts/

meddialog_20260711/
├── analysis/
│   ├── analyze_generator_model_content.py
│   └── analyze_reasoning_content.py
├── legacy_judges/
│   ├── README.md
│   ├── adc_common.py
│   ├── judge_pairwise_adc.py
│   ├── judge_absolute_adc.py
│   └── judge_absolute_adc_multi.py
├── protocols/
│   └── VERBOSITY_ABLATION_PROTOCOL.md
├── scripts/
│   ├── run_temperature_ablation_medical_only.sh
│   └── run_verbosity_ablation.py
├── tables/
├── build_tables.py
├── manifest.json
└── README.md
```

The verbosity builder imports the existing judge utilities and should therefore use the project environment:

```bash
.venv/bin/python experiments/meddialog_20260711/scripts/run_verbosity_ablation.py --help
```

Judge dependencies now flow in one direction:

```text
evaluation/judge_common.py                 # no scoring logic
├── judge_meddialog_four_evals.py          # ACTIVE judge
├── scripts/run_verbosity_ablation.py
└── legacy_judges/adc_common.py            # historical A/D/C schema
    ├── judge_pairwise_adc.py
    ├── judge_absolute_adc.py
    └── judge_absolute_adc_multi.py
        └── reuses judge_absolute_adc.py
```

The active judge never imports a legacy protocol.

## Canonical scope

- Dataset: MedDialog English train file, loaded source indices 0–99 (`N=100` per method).
- Generators: V4 Flash High and GPT-5.6 High.
- Matched methods:
  - `dynapro_medical` — **DynaPro + Medical**; medical DynaPro prompt and tracker.
  - `dynapro` — **Original DynaPro**; original DynaPro prompt and tracker.
  - `proact` — **Proact**; Proact prompt, no tracker.
  - `none` — **Baseline**; no wrapper prompt, no tracker.

The original notes said “Generic” in one place, but the final matched four-method experiment uses **Original DynaPro**, not Generic. V4 Generic artifacts exist; there is no matched GPT-5.6 High Generic generation run, so Generic is excluded from these tables.

## Where the underlying data lives

### Dataset and code

| Purpose | Canonical path |
|---|---|
| MedDialog train data | `data/raw/meddialog/english-train.json` |
| Dataset loader | `data/meddialog.py` |
| Conversation runner | `run_meddialog.py` |
| Original metric evaluator | `evaluate.py` |
| Four independent LLM evaluators | `judge_meddialog_four_evals.py` |
| Shared judge I/O and record validation | `evaluation/judge_common.py` |
| Historical A/D/C judge protocols | `experiments/meddialog_20260711/legacy_judges/` |
| Generator-content analysis | `experiments/meddialog_20260711/analysis/analyze_generator_model_content.py` |
| Reasoning-content analysis | `experiments/meddialog_20260711/analysis/analyze_reasoning_content.py` |
| Temperature runner | `experiments/meddialog_20260711/scripts/run_temperature_ablation_medical_only.sh` |
| Verbosity protocol | `experiments/meddialog_20260711/protocols/VERBOSITY_ABLATION_PROTOCOL.md` |
| Verbosity dataset builder | `experiments/meddialog_20260711/scripts/run_verbosity_ablation.py` |

### Canonical conversation files

| Generator | Method | Saved conversations |
|---|---|---|
| V4 | DynaPro + Medical | `output/thinking_ablation/dynapro_medical_v4flash_prompt98e58426_thinking_on_000_099_20260712/meddialog_dynapro_medical_deepseek-v4-flash-thinking_00_99.json` |
| V4 | Original DynaPro | `output/simulations/v4flash_dynapro_vs_medical_fc97fb17_000_099_20260712/meddialog_dynapro_deepseek-v4-flash-thinking_00_99.json` |
| V4 | Proact | `output/simulations/v4flash_3methods_000_099_20260712/meddialog_proact_deepseek-v4-flash-thinking_00_99.json` |
| V4 | Baseline | `output/simulations/v4flash_3methods_000_099_20260712/meddialog_none_deepseek-v4-flash-thinking_00_99.json` |
| GPT | DynaPro + Medical | `output/openai_simulations/gpt56luna_high_dynapromedical98e58426_dynapro1c64335f_proactb75ee522_none_workers4_completed100_000_099_20260712/meddialog_dynapro_medical_gpt-5.6-luna-reasoning-high_00_99_complete.json` |
| GPT | Original DynaPro | `output/openai_simulations/gpt56luna_high_dynapromedical98e58426_dynapro1c64335f_proactb75ee522_none_workers4_completed100_000_099_20260712/meddialog_dynapro_gpt-5.6-luna-reasoning-high_00_99_complete.json` |
| GPT | Proact | `output/openai_simulations/gpt56luna_high_dynapromedical98e58426_dynapro1c64335f_proactb75ee522_none_workers4_completed100_000_099_20260712/meddialog_proact_gpt-5.6-luna-reasoning-high_00_99_complete.json` |
| GPT | Baseline | `output/openai_simulations/gpt56luna_high_dynapromedical98e58426_dynapro1c64335f_proactb75ee522_none_workers4_completed100_000_099_20260712/meddialog_none_gpt-5.6-luna-reasoning-high_00_99_complete.json` |

The four GPT `*_complete.json` files are the canonical merged copies. Six invalid base trajectories were replaced with successful retries. The provenance is recorded in:

`output/openai_simulations/gpt56luna_high_dynapromedical98e58426_dynapro1c64335f_proactb75ee522_none_workers4_completed100_000_099_20260712/retry_merge_manifest.json`

Do not replace these with the unmerged base or retry fragments.

## Generator settings

The saved conversation records show:

| Generator | Model | Reasoning | Temperatures | New-turn limit |
|---|---|---|---|---:|
| V4 Flash High | `deepseek-v4-flash` | thinking enabled, high | user `1.0`; assistant `0.9`; tracker `0.0` when used | 6 |
| GPT-5.6 High | `gpt-5.6-luna` | high | user, assistant, and tracker omitted/unset | 6 |

Natural termination was enabled. The raw JSON records do **not** save the component max-token limits. Current source defaults are 1,024 for the user simulator, 2,048 for the assistant, and 8,192 for the tracker; therefore, the claim “every simulator used max_token 8192” is not supported by the saved generation records. The LLM **judge** does record `max_tokens=8192`.

The report's turn-limit statement comes from the canonical conversation JSON files, not from a separate saved summary. With `max_new_turns=6`, three assistant turns indicate that the run reached the interaction cap. If this statistic will be reported formally, it should be exported as its own reproducible table rather than inferred from mean turns alone.

## Original metrics

`evaluate.py` compares the final assistant turn with `single_turn_completion` from MedDialog. It reports corpus BLEU-4, mean sentence BLEU, BERTScore F1, assistant-turn count, mean assistant words per turn, and clarify rate.

Clarify rate is only a heuristic: the first assistant turn must contain `?` and have fewer than 150 words. A long turn that asks a valid question is counted as non-clarifying.

The canonical inputs for [`tables/original_eval.csv`](tables/original_eval.csv) are:

1. V4 DynaPro + Medical: `output/eval/v4flash_dynapro_medical_prompt98_evalpy_all_metrics_20260712/comparison.json`
2. Other three V4 methods: `output/eval/v4flash_five_methods_evalpy_all_metrics_20260712/comparison.json`
3. Four GPT methods: `output/eval/gpt56luna_high_four_methods_completed100_all_metrics_20260712/comparison.json`

The convenient `output/generator_model_comparison/.../generator_summary.csv` omits mean sentence BLEU and BERTScore standard deviation, so it is not the canonical full table source.

## Four-evaluator LLM judge

Each conversation is scored by four independent API calls:

- Anticipation: 0–10
- Discovery: 0–10
- Calibration: 0–10
- Medical quality: 0–20
- Total: 0–50

The evaluator prompt sources are under `evaluation/prompts/`, and score-band additions are under `evaluation/prompts/score_bands/`. The exact rendered prompt hashes are frozen in [`manifest.json`](manifest.json). The four calls share a conversation, so their scores should not be treated as statistically independent observations.

`judge_meddialog_four_evals.py` depends only on `evaluation/judge_common.py` for generic I/O and validation primitives. It does not import any historical judge. The three earlier A/D/C protocols and their dependency graph are documented in [`legacy_judges/README.md`](legacy_judges/README.md).

Canonical [`tables/judge_2x2.csv`](tables/judge_2x2.csv) sources:

| Generation | Judge | Source summary |
|---|---|---|
| V4 | V4 | Medical row: `output/judgments_thinking_ablation/dynapro_medical_v4flash_prompt98e58426_simthink_on_evalthink_on_banded_000_099_20260712/four_eval_summary.json`; other three rows: `output/judgments_score_band_ablation/five_methods_banded_v4flash_thinking_high_000_099_20260712/four_eval_summary.json` |
| V4 | GPT | `output/judgments_v4_generation/v4flash_high_eval_gpt56high_thinking_high_banded_000_099_20260712/four_eval_summary.json` |
| GPT | V4 | `output/judgments_openai_generation/gpt56luna_high_eval_v4flash_thinking_high_banded_000_099_20260712/four_eval_summary.json` |
| GPT | GPT | `output/judgments_openai_generation/gpt56luna_high_eval_gpt56high_thinking_high_banded_000_099_20260712/four_eval_summary.json` |

The V4-generation/V4-judge medical score is **38.97** from the prompt-hash-`98e58426` run. The older five-method medical row is 39.00 and is not used. All 16 canonical rows have `N=100`.

The original V4 judge used temperature `0` with high reasoning. The original GPT judge used high reasoning with temperature **omitted/unset**, not explicitly zero.

Keep each `four_eval_details.jsonl` beside its summary: it contains per-call results and is also the resume cache.

## Verbosity ablation

Canonical root:

`output/verbosity_ablation/deterministic_padding_n30_20260712/`

Thirty base conversations were evaluated under three conditions:

- `condition_a` → Original
- `condition_c` → Original Repeat (identical content; measures repeated-judge variation)
- `condition_b` → Padded (deterministic repetition of existing information)

The padded responses average 1.486× the original visible words. Both judges evaluated 90 condition-conversations (360 independent evaluator calls each) with shuffle seed `20260712`. V4 used temperature `0`/high reasoning; GPT used temperature unset/high reasoning.

This experiment supports the limited conclusion that adding redundant length did not increase scores under this construction. It does not prove the judges are free of every possible length bias.

## Temperature experiment

[`tables/temperature_ablation.csv`](tables/temperature_ablation.csv) combines the four previous DynaPro + Medical cells with four reruns:

- V4 judge: temperature `0 → 1`, with high reasoning held fixed. This is the cleaner temperature comparison.
- GPT judge: temperature `unset → 0`, but reasoning also changed from `high → none`. This comparison is confounded and cannot isolate temperature.

The GPT-generation/GPT-judge `temp=0, reasoning=none` row is provisional: 399/400 judge calls succeeded. `source_index=58` is missing the Discovery evaluator, so the reported total `40.12` is based on `N=99` complete conversations. No successful repair result was obtained.

The four “previous” rows reuse existing judge results; they are not new judge calls.

## File-handling rules

- `output/` stays in place because scripts contain dated paths, summaries contain provenance paths, and judge detail files support resume behavior.
- Some immutable historical files under `output/` record the code's pre-organization absolute path. They are preserved as provenance; the current paths are the ones in this README and the registry manifest.
- `output/` and `data/raw/` are ignored by Git. This registry makes the layout understandable, but it does not back up those large files. Archive them separately before changing machines.
- `.cache/` is ignored and can be regenerated; it is not experimental evidence.
- Do not commit `.env` or API keys.
- Prototype, smoke, low-reasoning, and superseded artifacts remain on disk for provenance but are excluded from the canonical tables. Their classifications are recorded in [`manifest.json`](manifest.json).
