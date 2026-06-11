#data/bigcodebench.py
from __future__ import annotations

import json
import random
from typing import List, Dict, Any

from datasets import load_dataset
from data.single_turn import SingleTurnDataset


class BigCodeBench(SingleTurnDataset):

    def __init__(
        self,
        repo_id: str = "bigcode/bigcodebench",
        *,
        train_ratio: float = 0.8,
        seed: int = 42,
    ):
        self.train_ratio = train_ratio
        self.seed = seed

        raw_ds = load_dataset(repo_id, split="v0.1.2")
        processed = self._preprocess(raw_ds)
        super().__init__(processed, eval_ratio=1.0 - train_ratio, seed=seed)

    def _preprocess(self, raw_ds) -> List[Dict[str, Any]]:
        n_total = len(raw_ds)
        n_train = int(n_total * self.train_ratio)

        random.seed(self.seed)
        indices = list(range(n_total))
        random.shuffle(indices)
        split_map = {idx: ("train" if i < n_train else "test") for i, idx in enumerate(indices)}

        processed = []
        for idx, row in enumerate(raw_ds):
            ground_truth = {
                "dataset": "bigcodebench",
                "task_id": row["task_id"],
                "test": row["test"],
                "entry_point": row["entry_point"],
                "answer": row["code_prompt"] + row["canonical_solution"],
            }
            processed.append({
                "prompt": row["instruct_prompt"],
                "completion": json.dumps(ground_truth),
                "split": split_map[idx],
                "task_id": row["task_id"],
                "entry_point": row["entry_point"],
                "test": row["test"],
                "extraction_requirement": f"Your extraction should be executable code without any need of processing. You should start with the following code:\n\n{row['code_prompt']}\n",
            })

        return processed