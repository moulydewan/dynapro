"""Load MedDialog as the single-turn prompt/completion format used here."""

import json
from pathlib import Path

from data.single_turn import SingleTurnDataset


DATA_DIR = Path(__file__).resolve().parent / "raw" / "meddialog"


class MedDialog(SingleTurnDataset):
    def __init__(self):
        data = []

        for split in ("train", "dev", "test"):
            path = DATA_DIR / f"english-{split}.json"
            with path.open(encoding="utf-8") as f:
                rows = json.load(f)

            for source_index, row in enumerate(rows):
                utterances = row["utterances"]
                # This benchmark uses only the first patient/doctor pair. Keep
                # num_utterances so multi-turn source rows remain auditable.
                valid = (
                    len(utterances) >= 2
                    and utterances[0].startswith("patient: ")
                    and utterances[1].startswith("doctor: ")
                )
                if not valid:
                    raise ValueError(f"Invalid dialogue at {path}:{source_index}")

                data.append({
                    "prompt": utterances[0].removeprefix("patient: ").strip(),
                    "completion": utterances[1].removeprefix("doctor: ").strip(),
                    "split": split,
                    "description": row["description"],
                    "source_index": source_index,
                    "num_utterances": len(utterances),
                })

        super().__init__(data)
