#!/usr/bin/env python3
"""Multi-model entry: dispatch to Compilers/ELSACompiler_*.py using configs/elsa_models.yaml."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

CONV_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(CONV_ROOT))

import elsa_support  # noqa: F401 — legacy ``partition`` shim for torch.load

from elsa_support.elsa_model_loader import (
    get_model_entry,
    import_runner,
    list_model_ids,
    load_elsa_models_config,
    resolve_data_paths,
)
from elsa_support.paths import CONV_ROOT as ELSA_CONV_ROOT


def _default_calculate_info_path(model_id: str, entry: dict, time_step: int, output_dir: Path) -> Path:
    name = entry["default_calculate_info"]
    return output_dir / name


def main() -> None:
    cfg = load_elsa_models_config()
    models = list_model_ids(cfg)
    default_model = "resnet50" if "resnet50" in models else (models[0] if models else "resnet50")

    parser = argparse.ArgumentParser(
        description="ELSA simulator: pick --model to run the matching ELSACompiler and metrics.",
    )
    parser.add_argument(
        "--model",
        default=default_model,
        choices=models,
        help="Which network / compiler to run (see configs/elsa_models.yaml)",
    )
    parser.add_argument(
        "--elsa-models-yaml",
        default=None,
        help="Override path to elsa_models.yaml (default: convolution/configs/elsa_models.yaml)",
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
        help="Hardware Config.yaml (default: configs/Config.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for .pth (default: convolution/outputs)",
    )
    parser.add_argument(
        "--Time_step",
        type=int,
        default=None,
        help="Simulation time steps (default: default_time_step in YAML for this model)",
    )
    parser.add_argument(
        "--metrics-mode",
        choices=["none", "file", "run"],
        default="run",
        help=(
            "none: only compilation; "
            "file: load calculateInfo from --info-path (or default under output-dir) and print metrics; "
            "run: compile then aggregate metrics from returned dict"
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
        help="Energy/area pie charts (matplotlib)",
    )
    args = parser.parse_args()

    if args.elsa_models_yaml:
        cfg = load_elsa_models_config(args.elsa_models_yaml)
    entry = get_model_entry(cfg, args.model)
    paths = resolve_data_paths(entry, ELSA_CONV_ROOT)

    datapath = args.datapath or entry["datapath"]
    connectionpath = args.connectionpath or paths["connectionpath"]
    mappingpath = args.mappingpath or paths["mappingpath"]
    occupyPath = args.occupyPath or paths["occupyPath"]
    time_step = (
        args.Time_step
        if args.Time_step is not None
        else int(entry["default_time_step"])
    )

    output_dir = Path(args.output_dir or (ELSA_CONV_ROOT / "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.metrics_mode == "file":
        if args.info_path:
            info_path = Path(args.info_path)
        else:
            info_path = _default_calculate_info_path(
                args.model, entry, time_step, output_dir
            )
        from elsa_support.resnet50_metrics import compute_resnet50_metrics_from_path

        compute_resnet50_metrics_from_path(
            str(info_path),
            plot=args.metrics_plots,
            model_id=args.model,
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
        from elsa_support.resnet50_metrics import compute_resnet50_metrics

        compute_resnet50_metrics(info, plot=args.metrics_plots, model_id=args.model)


if __name__ == "__main__":
    main()
