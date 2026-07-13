# Shareable simulation and eval results

This directory contains the JSON artifacts intended for Git sharing. It is a
curated copy of the canonical July 11 MedDialog experiment; the original files
under `output/` remain unchanged.

## Contents

```text
results/
├── simulations/
│   ├── v4/
│   └── gpt/
└── eval/
    ├── v4/
    └── gpt/
```

Each model directory has the same four methods:

- `dynapro_medical.json`: DynaPro + Medical
- `dynapro.json`: Original DynaPro
- `proact.json`: Proact
- `baseline.json`: no assistant prompt

The eight simulation files contain 100 records each (`source_index=0..99`).
Each record includes the complete user-simulator/assistant conversation and
run metadata. The two DynaPro methods also contain the per-turn parsed Tracker
states in `intent_states`; Proact and Baseline do not use the Tracker.

Each eval file corresponds one-to-one with a simulation file. It contains:

- `aggregate`: corpus and mean metrics for the 100 conversations;
- `per_sample`: BLEU, BERTScore, turn count, response length, and clarification
  measurements for every conversation;
- `input_file`: the repository-relative path to its simulation JSON.

The raw three-way `comparison.json` files were not copied because the older V4
comparison also includes Generic and a superseded DynaPro + Medical run. The
eight per-method eval files contain all canonical aggregate and per-sample
metrics without those extra rows.

Judge outputs, temperature and verbosity ablations, smoke tests, retries, and
legacy runs are intentionally excluded from this first shareable bundle.
