# Legacy MedDialog judges

These programs preserve earlier A/D/C evaluation protocols. They are not used
for the canonical July 11 tables.

| Program | Historical protocol |
|---|---|
| `judge_pairwise_adc.py` | Blinded A/B comparison of Original DynaPro and DynaPro + Medical |
| `judge_absolute_adc.py` | Absolute A/D/C scoring for the two DynaPro variants |
| `judge_absolute_adc_multi.py` | Multi-method extension of the same absolute A/D/C protocol |

The matching historical prompt templates are kept in `prompts/`. They are not
loaded by the active four-evaluator judge.

All three use `adc_common.py` for the historical A/D/C schema. Generic project
paths, environment loading, record validation, hashing, and atomic JSON writes
come from `evaluation/judge_common.py`.

The active evaluator is the repository-root `judge_meddialog_four_evals.py`,
which makes four independent Anticipation, Discovery, Calibration, and Medical
Quality calls. It does not import anything from this legacy directory.
