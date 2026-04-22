from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import DEVICE
from .paths import path_for_json as path_for_json_relative


def save_model(model: nn.Module, filename: Path) -> None:
    filename = Path(filename)
    tmp_filename = filename.with_name(f"{filename.name}.tmp")
    print(f"Saving model to {filename}...")

    tmp_filename.unlink(missing_ok=True)
    try:
        torch.save(model, tmp_filename)
        os.replace(tmp_filename, filename)
    except Exception:
        tmp_filename.unlink(missing_ok=True)
        raise


def load_model(filename: Path) -> nn.Module:
    print(f"Loading model from {filename}...")
    loaded_model = torch.load(filename, map_location=DEVICE, weights_only=False)
    if not isinstance(loaded_model, nn.Module):
        raise TypeError(f"Serialized object at {filename} is not a torch.nn.Module")
    loaded_model = loaded_model.to(DEVICE)
    loaded_model.eval()
    return loaded_model


def release_torch_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def path_to_json(path: Path | None) -> str | None:
    if path is None:
        return None
    return path_for_json_relative(path)


def write_setup_file(payload: dict[str, object], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Setup saved to {output_path}")


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return path_to_json(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def load_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Existing JSON is invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Existing JSON root must be an object: {path}")
    return payload


def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")
    print(f"JSON saved to {path}")


