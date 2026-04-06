"""Paths relative to the transformer simulator root."""
from pathlib import Path

_PKG = Path(__file__).resolve().parent
TRANSFORMER_ROOT = _PKG.parent
CONFIG_YAML = TRANSFORMER_ROOT / "configs" / "Config.yaml"
