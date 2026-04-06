"""Load ``configs/elsa_models.yaml`` and resolve per-model data paths."""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable, Dict, List

import yaml

from elsa_support.paths import CONV_ROOT


def load_elsa_models_config(path: str | Path | None = None) -> Dict[str, Any]:
    p = Path(path) if path else CONV_ROOT / "configs" / "elsa_models.yaml"
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_model_ids(cfg: Dict[str, Any]) -> List[str]:
    return sorted(cfg.get("models", {}).keys())


def get_model_entry(cfg: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    models = cfg.get("models", {})
    if model_id not in models:
        raise KeyError(
            f"Unknown model {model_id!r}. Valid: {', '.join(sorted(models))}"
        )
    return models[model_id]


def resolve_data_paths(entry: Dict[str, Any], conv_root: Path | None = None) -> Dict[str, str]:
    root = conv_root or CONV_ROOT
    base = root / entry["data_subdir"]
    return {
        "connectionpath": str(base / entry["connection"]),
        "mappingpath": str(base / entry["mapping"]),
        "occupyPath": str(base / entry["occupy"]),
    }


def import_runner(module_name: str, callable_name: str) -> Callable[..., Any]:
    mod = importlib.import_module(module_name)
    return getattr(mod, callable_name)
