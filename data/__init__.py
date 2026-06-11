from .single_turn import SingleTurnDataset
from .medium import Medium
from .math_hard import MathHard
from .bigcodebench import BigCodeBench

datasets_info = {
    "medium": {
        "task_desc": "document editing",
        "class": Medium,
    },
    "math_hard": {
        "task_desc": "math tutoring",
        "class": MathHard,
    },
    "bigcodebench": {
        "task_desc": "code generation",
        "class": BigCodeBench,
    },
}