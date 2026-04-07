#!/usr/bin/env python3
"""Multi-model entry: dispatch to Compilers/VSACompiler using configs/elsa_models.yaml."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Optional

TRANS_ROOT = Path(__file__).resolve().parent
CONV_ROOT = TRANS_ROOT.parent / "convolution"
sys.path.insert(0, str(TRANS_ROOT))

import elsa_support  # noqa: F401 — legacy ``partition`` shim for torch.load

from elsa_support.elsa_model_loader import (
    get_model_entry,
    import_runner,
    list_model_ids,
    load_elsa_models_config,
    resolve_data_paths,
)
from elsa_support.paths import TRANSFORMER_ROOT as ELSA_TRANS_ROOT


def _default_calculate_info_path(model_id: str, entry: dict, time_step: int, output_dir: Path) -> Path:
    del model_id, time_step
    name = entry.get("default_calculate_info", "calculateInfo.pth")
    return output_dir / name


def _default_hardware_config_path(explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit)
    return ELSA_TRANS_ROOT / "configs" / "Config.yaml"


def _frequency_hz_from_config(config_path: Path) -> float:
    """Clock from ``Config.yaml`` top-level ``frequency`` (MHz) → Hz (matches convolution)."""
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mhz = float(cfg.get("frequency", 200))
    return mhz * 1e6


def _import_compute_resnet50_metrics():
    """Load convolution ``compute_resnet50_metrics`` without shadowing ``transformer/elsa_support``."""
    mod_path = CONV_ROOT / "elsa_support" / "resnet50_metrics.py"
    spec = importlib.util.spec_from_file_location("elsa_conv_resnet50_metrics", mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load metrics module: {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_resnet50_metrics


def run_transformer_metrics(
    info: Dict[str, Any],
    *,
    model_id: str,
    config_path: Path,
    path_hint: Optional[str] = None,
    plot: bool = False,
) -> Dict[str, float]:
    """
    PE tile energy + NoC ``traffic_cost``, latency from last layer / frequency from ``config_path``.
    Uses ``compute_resnet50_metrics`` with ``transformer_noc_reference=True`` so NoC energy matches
    legacy ViT scripts (diagonal of ``transmitTraffic`` cleared before cost; no ``/2`` on traffic).
    """
    freq_hz = _frequency_hz_from_config(config_path)
    compute_resnet50_metrics = _import_compute_resnet50_metrics()
    return compute_resnet50_metrics(
        info,
        frequency=freq_hz,
        model_id=model_id,
        path_hint=path_hint,
        plot=plot,
        verbose=True,
        # Match legacy ViT scripts: zero diagonal on traffic matrix, no /2 on NoC energy.
        transformer_noc_reference=True,
    )


def main() -> None:
    cfg = load_elsa_models_config()
    models = list_model_ids(cfg)
    default_model = "vit_small" if "vit_small" in models else (models[0] if models else "vit_small")

    parser = argparse.ArgumentParser(
        description="ELSA transformer simulator: pick --model to run VSACompiler pipeline.",
    )
    parser.add_argument(
        "--model",
        default=default_model,
        choices=models,
        help="Which network / runner to run (see configs/elsa_models.yaml)",
    )
    parser.add_argument(
        "--elsa-models-yaml",
        default=None,
        help="Override path to elsa_models.yaml (default: transformer/configs/elsa_models.yaml)",
    )
    parser.add_argument(
        "--datapath",
        default=None,
        help="Exported layer binary directory (default: from YAML for --model)",
    )
    parser.add_argument(
        "--connectionpath",
        default=None,
        help="Layer connection .pth (default: datas/<model>/... from YAML)",
    )
    parser.add_argument(
        "--mappingpath",
        default=None,
        help="NoC mapping .pth (default from YAML)",
    )
    parser.add_argument(
        "--occupyPath",
        default=None,
        help="NoC link occupancy .pth (default from YAML)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Hardware configs/Config.yaml (default: configs/Config.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for .pth (default: transformer/outputs)",
    )
    parser.add_argument(
        "--Time_step",
        type=int,
        default=None,
        help="Reserved for API parity; simulation uses T from exported tensors (default from YAML)",
    )
    parser.add_argument(
        "--metrics-mode",
        choices=["none", "file", "run"],
        default="run",
        help=(
            "none: only compilation; "
            "file: load calculateInfo from --info-path and print metrics; "
            "run: full simulation then aggregate metrics"
        ),
    )
    parser.add_argument(
        "--info-path",
        default=None,
        help="calculateInfo *.pth for --metrics-mode file",
    )
    parser.add_argument(
        "--metrics-plots",
        action="store_true",
        help="Energy/area pie charts (matplotlib) if supported by metrics",
    )
    args = parser.parse_args()

    if args.elsa_models_yaml:
        cfg = load_elsa_models_config(args.elsa_models_yaml)
    entry = get_model_entry(cfg, args.model)
    paths = resolve_data_paths(entry, ELSA_TRANS_ROOT)

    datapath = args.datapath or entry["datapath"]
    dp = Path(datapath)
    if not dp.is_absolute():
        # Configs use paths relative to transformer root.
        dp = (ELSA_TRANS_ROOT / dp).resolve()
    # Many exported zips contain a single top-level folder with the same name.
    if dp.is_dir():
        has_act_files = any(dp.glob("act_*"))
        children = [p for p in dp.iterdir() if p.is_dir()]
        if (not has_act_files) and len(children) == 1:
            dp = children[0]
    datapath = str(dp)
    connectionpath = args.connectionpath or paths["connectionpath"]
    mappingpath = args.mappingpath or paths["mappingpath"]
    occupyPath = args.occupyPath or paths["occupyPath"]
    time_step = (
        args.Time_step
        if args.Time_step is not None
        else int(entry["default_time_step"])
    )

    hw_config = _default_hardware_config_path(args.config)

    output_dir = Path(args.output_dir or (ELSA_TRANS_ROOT / "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.metrics_mode == "file":
        if args.info_path:
            info_path = Path(args.info_path)
        else:
            info_path = _default_calculate_info_path(
                args.model, entry, time_step, output_dir
            )
        import torch

        try:
            info = torch.load(str(info_path), map_location="cpu", weights_only=False)
        except TypeError:
            info = torch.load(str(info_path), map_location="cpu")
        run_transformer_metrics(
            info,
            model_id=args.model,
            config_path=hw_config,
            path_hint=str(info_path),
            plot=args.metrics_plots,
        )
        return

    runner = import_runner(entry["runner_module"], entry["runner_callable"])
    info = runner(
        datapath=datapath,
        connectionpath=connectionpath,
        mappingpath=mappingpath,
        occupyPath=occupyPath,
        config_path=args.config,
        time_step=time_step,
        output_dir=str(output_dir),
    )
    if args.metrics_mode == "run":
        run_transformer_metrics(
            info,
            model_id=args.model,
            config_path=hw_config,
            path_hint=None,
            plot=args.metrics_plots,
        )


if __name__ == "__main__":
    main()
