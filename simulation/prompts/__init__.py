import os
import os.path as osp

current_dir = osp.dirname(__file__)

TERMINATION_SIGNAL = "[[TERMINATE CHAT]]"

with open(osp.join(current_dir, 'user_simulator.txt'), 'r') as f:
    USER_SIMULATOR_PROMPT = f.read()

with open(osp.join(current_dir, 'proact_instruction.txt'), 'r') as f:
    PROACT_MODEL_PROMPT = f.read()

with open(osp.join(current_dir, 'dynapro_assistant.txt'), 'r') as f:
    DYNAPRO_ASSISTANT_PROMPT = f.read()

with open(osp.join(current_dir, 'generic_proact.txt'), 'r') as f:
    GENERIC_PROACT_PROMPT = f.read()