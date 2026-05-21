import json
import os
import logging
from data import datasets_info
from simulation.simulate import ChatSessionSimulator

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
USER_MODEL_ID      = 'us.anthropic.claude-haiku-4-5-20251001-v1:0'
ASSISTANT_MODEL_ID = 'us.anthropic.claude-sonnet-4-5-20250929-v1:0'
REGION             = 'us-east-1'
MAX_NEW_TURNS      = 4
NUM_SAMPLES        = 1
NUM_ARTICLES       = 10
OUTPUT_DIR         = 'output/simulations'
OUTPUT_FILE        = os.path.join(OUTPUT_DIR, 'medium_baseline.json')

# ── Step 1: Load dataset ───────────────────────────────────────────────────────
logger.info("Loading Medium dataset...")
dataset_cls = datasets_info["medium"]["class"]
task_desc   = datasets_info["medium"]["task_desc"]
dataset     = dataset_cls().to_hf_dataset()
train       = dataset["train"]

# ── Step 2: Run simulation for each article ────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
simulator = ChatSessionSimulator()
data_list = []

for idx in range(NUM_ARTICLES):
    example = train[idx]
    single_turn_prompt     = example["single_turn_prompt"]
    single_turn_completion = example["single_turn_completion"]
    single_turn_metadata   = example["single_turn_metadata"]

    logger.info(f"\n[{idx+1}/{NUM_ARTICLES}] Article: {single_turn_prompt[:80]}...")

    try:
        sessions = simulator.run_chat_simulation(
            task_desc=task_desc,
            single_turn_prompt=single_turn_prompt,
            chat_history=[],
            user_generation_kwargs={"model": USER_MODEL_ID},
            assistant_generation_kwargs={"model": ASSISTANT_MODEL_ID, "temperature": 0.9},
            num_samples=NUM_SAMPLES,
            max_new_turns=MAX_NEW_TURNS,
            region=REGION,
            verbose=True,
        )

        # ── Append conversation to list ────────────────────────────────────
        data_list.append({
            "conv_id":                idx + 1,
            "task_desc":              task_desc,
            "single_turn_prompt":     single_turn_prompt,
            "single_turn_completion": single_turn_completion,
            "single_turn_metadata":   single_turn_metadata,
            "conversation":           sessions[0],
        })

        # ── Save after each conversation like CollabLLM ────────────────────
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(data_list, f, indent=2)

        logger.info(f"Saved {len(data_list)} conversations to {OUTPUT_FILE}")

    except Exception as e:
        logger.error(f"Error on article {idx+1}: {e}")
        continue

logger.info(f"\nDone. {len(data_list)} simulations saved to {OUTPUT_FILE}")