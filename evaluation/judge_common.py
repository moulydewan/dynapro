"""Shared infrastructure for MedDialog judge protocols.

This module contains only project paths, environment loading, record validation,
and JSON I/O. Scoring prompts and protocol-specific validation belong in the
individual judge programs.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JUDGE_TEMPERATURE = 0.0


def load_env() -> None:
    path = PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key, value.strip().strip("'\""))


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temp_path.replace(path)


def load_records(path: Path, expected_method: str) -> dict[int, dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must contain a non-empty JSON list")

    records: dict[int, dict[str, Any]] = {}
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"{path} contains a non-object record")
        method = item.get("method")
        if method is not None and method != expected_method:
            raise ValueError(
                f"{path} contains method={method!r}; expected {expected_method!r}"
            )
        source_index = item.get("source_index")
        if source_index is None:
            conv_id = item.get("conv_id")
            if not isinstance(conv_id, int):
                raise ValueError(
                    f"Record is missing integer source_index/conv_id: {item}"
                )
            source_index = conv_id - 1
        if not isinstance(source_index, int) or source_index < 0:
            raise ValueError(f"Invalid source_index={source_index!r} in {path}")
        if source_index in records:
            raise ValueError(f"Duplicate source_index={source_index} in {path}")
        conversation = item.get("conversation")
        if not isinstance(conversation, list) or not conversation:
            raise ValueError(
                f"source_index={source_index} has no conversation in {path}"
            )
        if not any(turn.get("role") == "assistant" for turn in conversation):
            raise ValueError(
                f"source_index={source_index} has no assistant turn in {path}"
            )
        records[source_index] = item
    return records


def require_nonempty_string(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def require_bool(value: Any, field: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
