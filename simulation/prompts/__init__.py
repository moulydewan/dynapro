import os
import os.path as osp

current_dir = osp.dirname(__file__)

TERMINATION_SIGNAL = "[[TERMINATE CHAT]]"

with open(osp.join(current_dir, 'user_simulator.txt'), 'r') as f:
    USER_SIMULATOR_PROMPT = f.read()