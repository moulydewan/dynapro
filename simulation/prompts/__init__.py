from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent

TERMINATION_SIGNAL = "[[TERMINATE CHAT]]"


def _read_prompt(filename):
    return (PROMPT_DIR / filename).read_text(encoding="utf-8")


USER_SIMULATOR_PROMPT = _read_prompt("user_simulator.txt")
PROACT_MODEL_PROMPT = _read_prompt("proact_instruction.txt")
DYNAPRO_ASSISTANT_PROMPT = _read_prompt("dynapro_assistant.txt")
DYNAPRO_MEDICAL_ASSISTANT_PROMPT = _read_prompt("dynapro_medical_assistant.txt")
GENERIC_PROACT_PROMPT = _read_prompt("generic_proact.txt")
