"""Paths relative to the convolution simulator root (kept out of project root listing)."""
from pathlib import Path

_PKG = Path(__file__).resolve().parent
CONV_ROOT = _PKG.parent
CONFIG_YAML = CONV_ROOT / "configs" / "Config.yaml"
