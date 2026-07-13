#!/usr/bin/env python3
"""Run matched MedDialog conversation simulations.

Configuration stays in environment variables so existing experiment commands
continue to work. Results are checkpointed after every completed conversation.
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from data import datasets_info
from simulation.modules.deepseek import DeepSeekBoto3
from simulation.modules.openai_api import OpenAIResponsesBoto3
import simulation.modules.llm_collaborator as collaborator_module
import simulation.modules.tracker as tracker_module
import simulation.modules.user_simulator as user_module
from simulation.simulate import ChatSessionSimulator


ROOT = Path(__file__).resolve().parent
DATASET_NAME = "meddialog"
USER_TEMPERATURE = 1.0
ASSISTANT_TEMPERATURE = 0.9
TRACKER_TEMPERATURE = 0.0

EXPERIMENTS = ("none", "proact", "generic_proact", "dynapro", "dynapro_medical")
# These two methods differ only in the assistant prompt; both use the tracker.
TRACKED_EXPERIMENTS = frozenset(("dynapro", "dynapro_medical"))
DEFAULT_EXPERIMENTS = ("dynapro", "dynapro_medical")


def load_env():
    path = Path(os.environ.get("ENV_FILE", ROOT / ".env"))
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key, value.strip().strip("'\""))


def load_provider():
    """Configure the API adapter and return metadata saved with every result."""
    provider = os.environ.get("LLM_PROVIDER", "deepseek")

    if provider == "deepseek":
        facade, key_name = DeepSeekBoto3(), "DEEPSEEK_API_KEY"
        user_model = assistant_model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")
        thinking, reasoning_effort = (
            os.environ.get("DEEPSEEK_THINKING"),
            os.environ.get("DEEPSEEK_REASONING_EFFORT"),
        )
    elif provider == "openai":
        facade, key_name = OpenAIResponsesBoto3(), "OPENAI_API_KEY"
        user_model = assistant_model = os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")
        reasoning_effort = os.environ.get("OPENAI_REASONING_EFFORT", "low")
        thinking = "disabled" if reasoning_effort == "none" else "enabled"
    elif provider == "bedrock":
        facade = key_name = None
        user_model, assistant_model = (
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        )
        thinking = reasoning_effort = None
    else:
        raise ValueError("LLM_PROVIDER must be 'bedrock', 'deepseek', or 'openai'")

    if facade:
        # The original components call boto3. This shared facade redirects all
        # three to DeepSeek or OpenAI without changing the simulation itself.
        for module in (user_module, collaborator_module, tracker_module):
            module.boto3 = facade
    if key_name and not os.environ.get(key_name):
        raise RuntimeError(f"{key_name} is not set")

    user_temperature, assistant_temperature, tracker_temperature = (
        (None, None, None)
        if provider == "openai"
        else (USER_TEMPERATURE, ASSISTANT_TEMPERATURE, TRACKER_TEMPERATURE)
    )
    return {
        "provider": provider,
        "user_model": user_model,
        "assistant_model": assistant_model,
        "user_temperature": user_temperature,
        "assistant_temperature": assistant_temperature,
        "tracker_temperature": tracker_temperature,
        "thinking": thinking,
        "reasoning_effort": reasoning_effort,
    }


def run_config():
    """Read and validate the batch settings used by existing launch commands."""
    start = int(os.environ.get("START_INDEX", "0"))
    count = int(os.environ.get("NUM_ARTICLES", "10"))
    max_new_turns = int(os.environ.get("MAX_NEW_TURNS", "6"))
    workers = int(os.environ.get("CONVERSATION_WORKERS", "40"))
    requested = os.environ.get("EXPERIMENTS", ",".join(DEFAULT_EXPERIMENTS))
    experiments = tuple(item.strip() for item in requested.split(",") if item.strip())

    if start < 0 or count < 1 or max_new_turns < 1 or workers < 1:
        raise ValueError(
            "START_INDEX >= 0, NUM_ARTICLES >= 1, MAX_NEW_TURNS >= 1, "
            "and CONVERSATION_WORKERS >= 1 are required"
        )
    if not experiments or set(experiments) - set(EXPERIMENTS):
        raise ValueError(
            f"EXPERIMENTS must contain only {EXPERIMENTS}; got {experiments}"
        )
    return start, count, max_new_turns, workers, experiments


def write_json_atomic(path, value):
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temp_path.replace(path)


def simulate_one(experiment, idx, example, generation, max_new_turns):
    use_tracker = experiment in TRACKED_EXPERIMENTS
    started = time.monotonic()
    logging.info(
        "Starting experiment=%s method=%s tracker=%s source_index=%s",
        experiment,
        experiment,
        use_tracker,
        idx,
    )

    session = ChatSessionSimulator().run_chat_simulation(
        task_desc=datasets_info[DATASET_NAME]["task_desc"],
        single_turn_prompt=example["single_turn_prompt"],
        chat_history=[],
        user_generation_kwargs={"model": generation["user_model"], "temperature": USER_TEMPERATURE},
        assistant_generation_kwargs={
            "model": generation["assistant_model"],
            "temperature": ASSISTANT_TEMPERATURE,
        },
        num_samples=1,
        max_new_turns=max_new_turns,
        proact_prompt_ratio=1.0,
        method=experiment,
        use_tracker=use_tracker,
        region="us-east-1",
        max_workers=1,
        verbose=False,
    )[0]

    conversation = session["conversation"] if use_tracker else session
    intent_states = session["intent_states"] if use_tracker else []

    metadata = {
        **generation,
        "tracker_temperature": generation["tracker_temperature"] if use_tracker else None,
        "max_new_turns": max_new_turns,
        "natural_termination_enabled": True,
    }
    return {
        "conv_id": idx + 1,
        "source_index": idx,
        "experiment": experiment,
        "method": experiment,
        "task_desc": datasets_info[DATASET_NAME]["task_desc"],
        **metadata,
        "tracker_used": use_tracker,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "single_turn_prompt": example["single_turn_prompt"],
        "single_turn_completion": example["single_turn_completion"],
        "single_turn_metadata": example["single_turn_metadata"],
        "conversation": conversation,
        "intent_states": intent_states,
    }


def main():
    # ── Step 1: Read configuration ──────────────────────────────────────────
    load_env()
    os.environ.setdefault("SSL_CERT_FILE", "/etc/ssl/cert.pem")
    generation = load_provider()
    start, count, max_new_turns, conversation_workers, experiments = run_config()

    # ── Step 2: Load the requested MedDialog rows ───────────────────────────
    dataset = datasets_info[DATASET_NAME]["class"]().to_hf_dataset()["train"]
    if start + count > len(dataset):
        raise ValueError(
            f"Requested rows {start}:{start + count}, but train has {len(dataset)} rows"
        )

    # ── Step 3: Prepare output paths without overwriting prior runs ─────────
    output_dir = Path(os.environ.get("OUTPUT_DIR", ROOT / "output/simulations"))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_tag = generation["assistant_model"].replace("/", "_")
    if generation["provider"] == "openai":
        thinking_tag = f"reasoning-{generation['reasoning_effort']}"
    else:
        thinking_tag = (
            "thinking" if generation["thinking"] == "enabled" else "nonthinking"
        )
    run_tag = f"{model_tag}-{thinking_tag}_{start:02d}_{start + count - 1:02d}"
    output_files = {
        experiment: output_dir / f"meddialog_{experiment}_{run_tag}.json"
        for experiment in experiments
    }
    existing = [path for path in output_files.values() if path.exists()]
    if existing and os.environ.get("ALLOW_OVERWRITE") != "1":
        raise FileExistsError(
            "Refusing to overwrite: " + ", ".join(str(path) for path in existing)
        )

    # ── Step 4: Run conversations and checkpoint each success ───────────────
    tasks = [(experiment, idx) for idx in range(start, start + count) for experiment in experiments]
    records = {experiment: {} for experiment in experiments}
    errors = {experiment: {} for experiment in experiments}
    workers = min(conversation_workers, len(tasks))
    started = time.monotonic()

    logging.info(
        "Running %s conversations (%s) with %s workers and max_new_turns=%s",
        len(tasks),
        ", ".join(f"{experiment}={count}" for experiment in experiments),
        workers,
        max_new_turns,
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                simulate_one, experiment, idx, dataset[idx], generation, max_new_turns
            ): (experiment, idx)
            for experiment, idx in tasks
        }
        for future in as_completed(futures):
            experiment, idx = futures[future]
            try:
                records[experiment][idx] = future.result()
            except Exception as error:
                errors[experiment][idx] = repr(error)
                logging.exception(
                    "experiment=%s source_index=%s failed", experiment, idx
                )
                continue
            ordered = [record for _, record in sorted(records[experiment].items())]
            write_json_atomic(output_files[experiment], ordered)

    summary = {
        **generation,
        "max_new_turns": max_new_turns,
        "natural_termination_enabled": True,
        "conversation_workers": workers,
        "total_elapsed_seconds": round(time.monotonic() - started, 2),
        "experiments": {
            experiment: {
                "method": experiment,
                "use_tracker": experiment in TRACKED_EXPERIMENTS,
                "completed": len(records[experiment]),
                "failed_source_indices": sorted(errors[experiment]),
                "errors": errors[experiment],
                "output_file": str(output_files[experiment]),
            }
            for experiment in experiments
        },
    }
    summary_file = output_dir / f"meddialog_compare_{run_tag}_summary.json"
    write_json_atomic(summary_file, summary)
    logging.info("Saved summary to %s", summary_file)

    incomplete = [
        f"{experiment}={len(records[experiment])}/{count}"
        for experiment in experiments
        if len(records[experiment]) != count
    ]
    if incomplete:
        raise RuntimeError("Incomplete simulations: " + ", ".join(incomplete))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    main()
