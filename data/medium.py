from __future__ import annotations
import random
from typing import List, Dict, Any

import tiktoken
from datasets import load_dataset
from tqdm import tqdm

from data.single_turn import SingleTurnDataset


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string, disallowed_special=()))


class Medium(SingleTurnDataset):
    """
    Medium articles → SingleTurnDataset
    
    Each example:
        single_turn_prompt     = "Please write an article ... about {title}"
        single_turn_completion = the full article text  (ground truth)
        single_turn_metadata   = {url, authors, timestamp, tags, num_tokens}
    """

    def __init__(
        self,
        repo_id: str = "Kamaljp/medium_articles",
        *,
        train_ratio: float = 0.020,
        test_ratio: float = 0.005,
        max_tokens: int = 512,
        seed: int = 42,
    ):
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.max_tokens = max_tokens
        self.seed = seed

        raw = load_dataset(repo_id)
        processed = self._preprocess(raw)
        super().__init__(processed, eval_ratio=test_ratio, seed=seed)

    def _preprocess(self, raw_ds) -> List[Dict[str, Any]]:
        # keep only rows with a timestamp, sort by it (most recent = tail)
        full = [row for row in raw_ds["train"] if row.get("timestamp") is not None]
        full.sort(key=lambda x: x["timestamp"])

        n_total  = len(full)
        n_test   = int(n_total * self.test_ratio)
        n_train  = int(n_total * self.train_ratio)

        recent = full[-(n_train + n_test):]
        random.Random(self.seed).shuffle(recent)

        test_set  = recent[:n_test]
        train_set = recent[n_test:]

        split_map = {id(row): "test"  for row in test_set}
        split_map.update({id(row): "train" for row in train_set})

        examples: List[Dict[str, Any]] = []
        for row in tqdm(recent, desc="Processing Medium articles"):
            tokens = num_tokens_from_string(row["text"])
            if tokens > self.max_tokens:
                continue

            examples.append({
                # Required by SingleTurnDataset
                "prompt":     f"Please write an article in less than 500 words about:\n\n{row['title']}\n\n{row['text']}",
                "completion": row["text"],
                "split":      split_map[id(row)],
                # Metadata
                "url":        row.get("url"),
                "authors":    row.get("authors"),
                "timestamp":  row.get("timestamp"),
                "tags":       row.get("tags"),
                "num_tokens": tokens,
            })

        return examples