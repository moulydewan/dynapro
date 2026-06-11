from __future__ import annotations
from typing import Dict, List, Any
from datasets import load_dataset
from data.single_turn import SingleTurnDataset


class MathHard(SingleTurnDataset):
    """
    MATH-Hard → SingleTurnDataset adaptor.

    Loads from lighteval/MATH-Hard (Level 5 problems only — baked into dataset).
    HF dataset already has train/test splits, so no ratio math needed.

    Each example:
        prompt     = the math problem
        completion = the reference solution
        split      = "train" / "test" (from HF)
        level      = difficulty level (all Level 5)
        type       = problem category (Algebra, Geometry, etc.)
    """

    def __init__(
        self,
        repo_id: str = "lighteval/MATH-Hard",
        *,
        eval_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.eval_ratio = eval_ratio
        self.seed = seed

        raw_ds = load_dataset(repo_id)
        processed = self._preprocess(raw_ds)
        super().__init__(processed, eval_ratio=eval_ratio, seed=seed)

    @staticmethod
    def _preprocess(raw_ds) -> List[Dict[str, Any]]:
        processed = []

        for split_name, split in raw_ds.items():
            for row in split:
                processed.append({
                    # Required by SingleTurnDataset
                    "prompt":     row["problem"],
                    "completion": row["solution"],
                    # split comes from HF directly — no manual splitting needed
                    "split":      split_name,
                    # Metadata
                    "level":      row.get("level"),
                    "type":       row.get("type"),
                })

        return processed