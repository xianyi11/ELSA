"""Lightweight summaries from ``calculateInfo`` dict (transformer simulator)."""
from __future__ import annotations

from typing import Any, Dict


def compute_transformer_metrics(info: Dict[str, Any], plot: bool = False) -> None:
    del plot  # optional matplotlib hooks not implemented
    total_e = 0.0
    total_a = 0.0
    max_lat = 0.0
    for k, v in info.items():
        if not isinstance(v, dict):
            continue
        total_e += float(v.get("Total Energy", 0) or 0)
        total_a += float(v.get("Total Area", 0) or 0)
        max_lat = max(max_lat, float(v.get("Total Latency", 0) or 0))
    print(
        "[transformer metrics] summed Total Energy (approx layers): %.6e, "
        "summed Total Area: %.6e, max Total Latency: %.1f"
        % (total_e, total_a, max_lat)
    )


def compute_transformer_metrics_from_path(path: str, plot: bool = False) -> None:
    import torch

    try:
        info = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        info = torch.load(path, map_location="cpu")
    compute_transformer_metrics(info, plot=plot)
