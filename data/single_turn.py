# Base single turn dataset, can be used for any dataset to convert it into a HuggingFace DatasetDict.

from datasets import Dataset, DatasetDict
from typing import List, Dict, Any
import random


class SingleTurnDataset:
    """A dataset wrapper for single-turn chat data with HuggingFace integration."""
    
    def __init__(self, data: List[Dict[str, Any]], eval_ratio: float = 0.1, seed: int = 42):
        if not data:
            raise ValueError("Data cannot be empty")
        
        required_fields = {'prompt', 'completion'}
        self.fields = set(data[0].keys())
        
        if not required_fields.issubset(self.fields):
            missing = required_fields - self.fields
            raise ValueError(f"Missing required fields: {missing}")
        
        for i, entry in enumerate(data):
            if set(entry.keys()) != self.fields:
                raise ValueError(f"Entry {i} has inconsistent keys. "
                               f"Expected: {self.fields}, Got: {set(entry.keys())}")
        
        self.data = data
        self.eval_ratio = eval_ratio
        self.seed = seed

    def to_hf_dataset(self) -> DatasetDict:
        if 'split' in self.fields:
            splits = [entry['split'] for entry in self.data]
            unique_splits = list(set(splits))
            split_indices = {
                split: [i for i, x in enumerate(splits) if x == split] 
                for split in unique_splits
            }
        else:
            random.seed(self.seed)
            eval_size = int(len(self.data) * self.eval_ratio)
            eval_indices = random.sample(range(len(self.data)), k=min(eval_size, len(self.data)))
            train_indices = list(set(range(len(self.data))) - set(eval_indices))
            split_indices = {
                'train': train_indices,
                'eval': eval_indices
            }
        
        metadata_fields = self.fields - {'prompt', 'completion', 'split'}
        
        dataset_dict = {}
        for split, indices in split_indices.items():
            if not indices:
                continue
            dataset_dict[split] = Dataset.from_dict({
                "single_turn_prompt": [self.data[i]['prompt'] for i in indices],
                "single_turn_completion": [self.data[i]['completion'] for i in indices],
                "single_turn_metadata": [
                    {field: self.data[i][field] for field in metadata_fields}
                    for i in indices
                ]
            })
        
        return DatasetDict(dataset_dict)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves a specific chat entry by index.

        Args:
            idx: The index of the chat entry to retrieve.

        Returns:
            The chat entry at the specified index.
        
        Raises:
            IndexError: If index is out of range.
        """
        return self.data[idx]

    def __len__(self) -> int:
        """
        Returns the number of chat entries in the dataset.

        Returns:
            The number of chat entries in the dataset.
        """
        return len(self.data)
    
    def get_splits_info(self) -> Dict[str, int]:
        """
        Returns information about data splits.
        
        Returns:
            Dictionary mapping split names to their sizes.
        """
        if 'split' in self.fields:
            splits = [entry['split'] for entry in self.data]
            split_counts = {}
            for split in set(splits):
                split_counts[split] = splits.count(split)
            return split_counts
        else:
            eval_size = int(len(self.data) * self.eval_ratio)
            return {
                'train': len(self.data) - eval_size,
                'eval': eval_size
            }