#!/usr/bin/env python3
"""
Figure 17: run convolution + transformer ``run_elsa.py``, normalize vs Eyeriss baselines,
then plot Energy Saving (top) and Speed Up (bottom).

Baseline tables are from the provided spreadsheets (latency + energy per accelerator).
Normalization:
  speedup_arch       = Eyeriss_latency / arch_latency   (``_BASE_LAT``)
  energy_saving_arch = Eyeriss_energy / arch_energy (``_BASE_ENERGY``)
ELSA column: same totals as ``convolution/run_elsa.py`` → ``elsa_support.resnet50_metrics.compute_resnet50_metrics``
(**energy** = tile sum + NoC ``traffic_cost`` + ResNet50-only ``energy_base_offset``; **latency** = last layer cycles / frequency). Other accelerators’ table cells are unchanged.

Rows W1–W6: VGG16-CIFAR10, VGG16-CIFAR100, VGG16-DVS, ResNet18, ResNet34, ResNet50.
Geomean: geometric mean across W1–W6 for each bar. Transformer (ViT-small) is run and
saved to ``figure17_vit_elsa.json`` (not in W1–W6 / Geomean because published table has no ViT row).

CLI: ``--print-matrices`` prints two 7×6 tables (W1–W6 + Geomean × six accelerators); ``--matrices-txt PATH``
saves the same text. In code, use ``matrices_for_print(energy_rows, speed_rows)`` or ``numpy.asarray``::

    E, S = matrices_for_print(energy_rows, speed_rows)  # each 7×6
    # import numpy as np
    # np.set_printoptions(precision=6, suppress=True)
    # print(np.array(E))
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

# --- Root paths ---
ROOT = Path(__file__).resolve().parent
CONV_ROOT = ROOT / "convolution"
TRANS_ROOT = ROOT / "transformer"
FIGURE17_FONT_DIR = ROOT / "figure17_cache" / "font"

# Wi -> convolution ``--model`` id
CONV_MODEL_ORDER: List[Tuple[str, str]] = [
    ("W1", "vgg16_cifar10"),
    ("W2", "vgg16_cifar100"),
    ("W3", "vgg16_cifar10dvs"),
    ("W4", "resnet18"),
    ("W5", "resnet34"),
    ("W6", "resnet50"),
]

# Published baselines: keys must match ``works`` first five + Eyeriss for division.
# NOTE: Latency vs Energy for the two spreadsheets were swapped in an earlier revision;
# ``_BASE_LAT`` is used for Speed Up, ``_BASE_ENERGY`` for Energy Saving (top subplot).

# Latency table (used for Speed Up = Eyeriss_lat / arch_lat)
_BASE_LAT: Dict[str, Dict[str, float]] = {
    "W1": {
        "Eyeriss": 25.43,
        "SpinalFlow": 0.964207,
        "TrueNorth": 11.37931,
        "Darwin": 9.88024,
        "PAICORE": 0.570934,
        "C-DNN": 0.781047,
    },
    "W2": {
        "Eyeriss": 25.46,
        "SpinalFlow": 0.964207,
        "TrueNorth": 11.37931,
        "Darwin": 9.88024,
        "PAICORE": 0.570934,
        "C-DNN": 0.781047,
    },
    "W3": {
        "Eyeriss": 37.0176,
        "SpinalFlow": 2.264427,
        "TrueNorth": 26.72414,
        "Darwin": 23.20359,
        "PAICORE": 1.34083,
        "C-DNN": 1.834276,
    },
    "W4": {
        "Eyeriss": 84.518,
        "SpinalFlow": 5.303141,
        "TrueNorth": 62.58621,
        "Darwin": 54.34132,
        "PAICORE": 3.140138,
        "C-DNN": 4.307072,
    },
    "W5": {
        "Eyeriss": 159.265,
        "SpinalFlow": 10.75237,
        "TrueNorth": 126.8966,
        "Darwin": 110.1796,
        "PAICORE": 6.366782,
        "C-DNN": 8.732795,
    },
    "W6": {
        "Eyeriss": 203.166,
        "SpinalFlow": 11.95033,
        "TrueNorth": 141.0345,
        "Darwin": 122.4551,
        "PAICORE": 7.076125,
        "C-DNN": 9.705743,
    },
}

# Energy table (used for Energy Saving = Eyeriss_energy / arch_energy)
_BASE_ENERGY: Dict[str, Dict[str, float]] = {
    "W1": {
        "Eyeriss": 3.079,
        "SpinalFlow": 0.156584,
        "TrueNorth": 1.65,
        "Darwin": 5.791373,
        "PAICORE": 0.570934,
        "C-DNN": 0.026087,
    },
    "W2": {
        "Eyeriss": 3.103,
        "SpinalFlow": 0.156584,
        "TrueNorth": 1.65,
        "Darwin": 5.899067,
        "PAICORE": 0.570934,
        "C-DNN": 0.026087,
    },
    "W3": {
        "Eyeriss": 4.651,
        "SpinalFlow": 0.367734,
        "TrueNorth": 3.875,
        "Darwin": 14.85778,
        "PAICORE": 1.34083,
        "C-DNN": 0.061265,
    },
    "W4": {
        "Eyeriss": 3.934,
        "SpinalFlow": 0.868537,
        "TrueNorth": 9.075,
        "Darwin": 17.6402,
        "PAICORE": 3.140138,
        "C-DNN": 0.148163,
    },
    "W5": {
        "Eyeriss": 7.46,
        "SpinalFlow": 1.761,
        "TrueNorth": 18.4,
        "Darwin": 51.55701,
        "PAICORE": 6.366782,
        "C-DNN": 0.300408,
    },
    "W6": {
        "Eyeriss": 10.676,
        "SpinalFlow": 1.957198,
        "TrueNorth": 20.45,
        "Darwin": 54.89796,
        "PAICORE": 7.076125,
        "C-DNN": 0.333878,
    },
}

PUBLISHED_ARCHES = ["SpinalFlow", "TrueNorth", "Darwin", "PAICORE", "C-DNN"]
WORKS = ["SpinalFlow", "TrueNorth", "Darwin", "PAICORE", "C-DNN", "ELSA"]

# ELSA totals come from ``compute_resnet50_metrics`` (default 200 MHz in that module).

def _geom_mean(values: Sequence[float]) -> float:
    xs = [float(v) for v in values if v is not None and v > 0]
    if not xs:
        return float("nan")
    return float(math.exp(sum(math.log(x) for x in xs) / len(xs)))


def _load_torch(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _default_calculate_info_name(model_id: str, entry: dict, time_step: int) -> str:
    if model_id == "resnet34":
        return f"calculateInfoConv_ResNet34_a4w4_with_Noc_T={time_step}.pth"
    return str(entry["default_calculate_info"])


def run_elsa_subprocess(sim_root: Path, model: str, out_sub: Path) -> None:
    out_sub.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(sim_root / "run_elsa.py"),
        "--model",
        model,
        "--metrics-mode",
        "none",
        "--output-dir",
        str(out_sub),
    ]
    subprocess.run(cmd, cwd=str(sim_root), check=True)


def load_metrics_from_run(
    sim_root: Path, model_id: str, out_sub: Path
) -> Tuple[float, float, float, float, float]:
    """Load calculateInfo .pth and apply the same aggregation as ``run_elsa.py`` (``compute_resnet50_metrics``).

    Returns ``(energy_total_J, latency_cycles_last, latency_ms, energy_mJ, frequency_Hz)``.
    Energy includes per-layer tile sum, NoC ``traffic_cost``, and ResNet50-only ``energy_base_offset``
    (see ``elsa_support/resnet50_metrics.py``).
    """
    import sys
    import yaml

    cfg_path = sim_root / "configs" / "elsa_models.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    entry = cfg["models"][model_id]
    ts = int(entry["default_time_step"])
    name = _default_calculate_info_name(model_id, entry, ts)
    p = out_sub / name
    if not p.exists():
        cands = sorted(out_sub.glob("calculateInfo*.pth"), key=lambda x: x.stat().st_mtime)
        if not cands:
            raise FileNotFoundError(f"No calculateInfo*.pth under {out_sub}")
        p = cands[-1]
    info = _load_torch(p)

    conv_root = str(CONV_ROOT)
    if conv_root not in sys.path:
        sys.path.insert(0, conv_root)
    from elsa_support.resnet50_metrics import compute_resnet50_metrics

    out = compute_resnet50_metrics(
        info,
        verbose=False,
        plot=False,
        model_id=model_id,
        path_hint=str(p),
        transformer_noc_reference=(model_id == "vit_small"),
    )
    return (
        float(out["energy_total_J"]),
        float(out["latency_cycles_last_layer"]),
        float(out["latency_last_layer_ms"]),
        float(out["energy_total_mj"]),
        float(out["frequency"]),
    )


def published_row_speedup(wi: str) -> List[float]:
    b = _BASE_LAT[wi]
    e = b["Eyeriss"]
    return [e / b[a] for a in PUBLISHED_ARCHES]


def published_row_energy_saving(wi: str) -> List[float]:
    b = _BASE_ENERGY[wi]
    e = b["Eyeriss"]
    return [e / b[a] for a in PUBLISHED_ARCHES]


def elsa_speedup_energy(
    wi: str, elsa_latency_ms: float, elsa_energy_mj: float
) -> Tuple[float, float]:
    """ELSA side uses **ms** (latency) and **mJ** (energy); published baselines unchanged."""
    if elsa_latency_ms <= 0 or elsa_energy_mj <= 0:
        raise ValueError(
            f"Invalid ELSA totals for {wi}: lat_ms={elsa_latency_ms}, energy_mJ={elsa_energy_mj}"
        )
    sp = _BASE_LAT[wi]["Eyeriss"] / elsa_latency_ms
    es = _BASE_ENERGY[wi]["Eyeriss"] / elsa_energy_mj
    return sp, es


def build_rows(
    elsa_by_wi: Dict[str, Tuple[float, float]]
) -> Tuple[List[List[float]], List[List[float]]]:
    """Returns (energy_saving_rows, speedup_rows) each 6x6 for W1..W6."""
    energy_rows: List[List[float]] = []
    speed_rows: List[List[float]] = []
    for wi, _mid in CONV_MODEL_ORDER:
        e_pub = published_row_energy_saving(wi)
        s_pub = published_row_speedup(wi)
        lat_ms, en_mj = elsa_by_wi[wi]
        sp, es = elsa_speedup_energy(wi, lat_ms, en_mj)
        energy_rows.append(e_pub + [es])
        speed_rows.append(s_pub + [sp])
    return energy_rows, speed_rows


def geomean_rows(rows: List[List[float]]) -> List[float]:
    return [_geom_mean([rows[r][c] for r in range(6)]) for c in range(6)]


def matrices_for_print(
    energy_rows: List[List[float]], speed_rows: List[List[float]]
) -> Tuple[List[List[float]], List[List[float]]]:
    """7×6 matrices: rows W1..W6 + Geomean, columns ``WORKS``."""
    e_geo = geomean_rows(energy_rows)
    s_geo = geomean_rows(speed_rows)
    E = [list(energy_rows[i]) for i in range(6)] + [e_geo]
    S = [list(speed_rows[i]) for i in range(6)] + [s_geo]
    return E, S


def print_data_matrices(
    energy_rows: List[List[float]],
    speed_rows: List[List[float]],
    *,
    to_file: Path | None = None,
) -> None:
    """Print Energy Saving and Speed Up as numeric matrices (same as the bar chart)."""
    row_labels = ["W1", "W2", "W3", "W4", "W5", "W6", "Geomean"]
    E, S = matrices_for_print(energy_rows, speed_rows)

    lines: List[str] = []

    def emit(s: str = "") -> None:
        lines.append(s)
        print(s)

    emit()
    emit("Energy Saving  [rows: network, cols: " + ", ".join(WORKS) + "]")
    emit("-" * 88)
    emit(f"{'':10s}" + "".join(f"{h:>14s}" for h in WORKS))
    for lab, row in zip(row_labels, E):
        emit(f"{lab:10s}" + "".join(f"{v:14.6f}" for v in row))

    emit()
    emit("Speed Up  [rows: network, cols: " + ", ".join(WORKS) + "]")
    emit("-" * 88)
    emit(f"{'':10s}" + "".join(f"{h:>14s}" for h in WORKS))
    for lab, row in zip(row_labels, S):
        emit(f"{lab:10s}" + "".join(f"{v:14.6f}" for v in row))
    emit()

    if to_file is not None:
        to_file.write_text("\n".join(lines), encoding="utf-8")


def _register_figure17_fonts() -> None:
    """Load all TTFs under ``figure17_cache/font`` so matplotlib uses local Times New Roman."""
    from matplotlib import font_manager as fm

    if not FIGURE17_FONT_DIR.is_dir():
        return
    for path in sorted(FIGURE17_FONT_DIR.glob("*.[Tt][Tt][Ff]")):
        try:
            fm.fontManager.addfont(str(path))
        except OSError:
            pass


def plot_figure(
    energy_rows: List[List[float]],
    speed_rows: List[List[float]],
    out_pdf: Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    _register_figure17_fonts()
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False

    group_names = ["W1", "W2", "W3", "W4", "W5", "W6", "Geomean"]
    bar_width = 0.8
    group_gap = 1
    colors = sns.color_palette("flare", n_colors=10)
    hatches = ["", "//////"]
    works = WORKS
    glen = 6

    x1 = np.arange(glen)
    x2 = np.arange(glen) + glen + group_gap
    x3 = np.arange(glen) + 2 * glen + 2 * group_gap
    x4 = np.arange(glen) + 3 * glen + 3 * group_gap
    x5 = np.arange(glen) + 4 * glen + 4 * group_gap
    x6 = np.arange(glen) + 5 * glen + 5 * group_gap
    x7 = np.arange(glen) + 6 * glen + 6 * group_gap

    def draw_groups(
        ax,
        rows: List[List[float]],
        yfac: float,
        ylim: float,
        ylabel: str,
        *,
        show_legend: bool = True,
    ) -> None:
        legend_added = [False] * len(colors)
        groups = [rows[0], rows[1], rows[2], rows[3], rows[4], rows[5], geomean_rows(rows)]
        xs = [x1, x2, x3, x4, x5, x6, x7]
        for row, xv in zip(groups, xs):
            for i, val in enumerate(row):
                if val == 0 or math.isnan(val):
                    continue
                ax.bar(
                    xv[i],
                    val,
                    width=bar_width,
                    color=colors[i],
                    label=f"{works[i]}" if not legend_added[i] else "",
                    hatch=hatches[i % 2],
                )
                ax.text(
                    xv[i] - 0.1,
                    val * yfac,
                    f"{val:10.1f}",
                    ha="center",
                    va="bottom",
                    rotation=270,
                )
                legend_added[i] = True

        ax.set_yscale("log", base=2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(0, ylim)
        ax.set_ylabel(ylabel)
        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles,
                labels,
                ncol=6,
                bbox_to_anchor=(0.95, 1.0),
                frameon=False,
                loc="lower right",
                columnspacing=1.0,
                handletextpad=0.5,
            )

    plt.figure(figsize=(8, 3.5))
    ax1 = plt.subplot(211)
    draw_groups(ax1, energy_rows, yfac=1.1, ylim=400, ylabel="Energy Saving")

    ax2 = plt.subplot(212)
    draw_groups(
        ax2, speed_rows, yfac=1.2, ylim=400, ylabel="Speed Up", show_legend=False
    )
    group_centers = [
        np.mean(x1),
        np.mean(x2),
        np.mean(x3),
        np.mean(x4),
        np.mean(x5),
        np.mean(x6),
        np.mean(x7),
    ]
    ax2.set_xticks(group_centers, group_names)

    plt.subplots_adjust(hspace=0.1)
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.0, dpi=400)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Figure 17 (ELSA vs baselines).")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "figure17_cache",
        help="Per-model simulator outputs (calculateInfo .pth)",
    )
    parser.add_argument(
        "--skip-sim",
        action="store_true",
        help="Only load from cache-dir (must exist from a previous run).",
    )
    parser.add_argument(
        "--skip-transformer",
        action="store_true",
        help="Do not run transformer/vit_small.",
    )
    parser.add_argument(
        "--out-pdf",
        type=Path,
        default=ROOT / "SNN_benchMark_Comparison.png",
        help="Output PDF path",
    )
    parser.add_argument(
        "--print-matrices",
        action="store_true",
        help="Print Energy Saving / Speed Up as 7×6 matrices to stdout.",
    )
    parser.add_argument(
        "--matrices-txt",
        type=Path,
        default=None,
        help="Also write the same matrix text to this file (implies formatted dump).",
    )
    args = parser.parse_args()

    cache: Path = args.cache_dir
    cache.mkdir(parents=True, exist_ok=True)

    # Per Wi: (latency_ms, energy_mJ) after physical-unit conversion for ELSA only
    elsa_by_wi: Dict[str, Tuple[float, float]] = {}
    elsa_raw_by_wi: Dict[str, Dict[str, float]] = {}

    for wi, mid in CONV_MODEL_ORDER:
        sub = cache / "convolution" / mid
        if not args.skip_sim:
            run_elsa_subprocess(CONV_ROOT, mid, sub)
        total_e_j, total_l_cyc, lat_ms, en_mj, freq_hz = load_metrics_from_run(
            CONV_ROOT, mid, sub
        )
        elsa_by_wi[wi] = (lat_ms, en_mj)
        elsa_raw_by_wi[wi] = {
            "total_latency_cycles": total_l_cyc,
            "total_energy_J": total_e_j,
            "total_latency_ms": lat_ms,
            "total_energy_mJ": en_mj,
            "clock_MHz": freq_hz / 1e6,
            "frequency_Hz": freq_hz,
        }

    energy_rows, speed_rows = build_rows(elsa_by_wi)

    payload = {
        "elsa_raw_by_wi": elsa_raw_by_wi,
        "energy_saving_rows": {f"W{i+1}": energy_rows[i] for i in range(6)},
        "speedup_rows": {f"W{i+1}": speed_rows[i] for i in range(6)},
        "geomean_energy_saving": geomean_rows(energy_rows),
        "geomean_speedup": geomean_rows(speed_rows),
    }

    if not args.skip_transformer:
        vit_dir = cache / "transformer" / "vit_small"
        if not args.skip_sim:
            run_elsa_subprocess(TRANS_ROOT, "vit_small", vit_dir)
        try:
            v_e_j, v_l_cyc, v_ms, v_mj, v_freq = load_metrics_from_run(
                TRANS_ROOT, "vit_small", vit_dir
            )
            payload["vit_small"] = {
                "total_latency_cycles": v_l_cyc,
                "total_energy_J": v_e_j,
                "total_latency_ms": v_ms,
                "total_energy_mJ": v_mj,
                "clock_MHz": v_freq / 1e6,
                "frequency_Hz": v_freq,
            }
        except Exception as e:
            payload["vit_small_error"] = str(e)

    out_json = args.out_pdf.with_suffix(".json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    plot_figure(energy_rows, speed_rows, args.out_pdf)
    print(f"Wrote {args.out_pdf} and {out_json}")

    if args.print_matrices or args.matrices_txt is not None:
        print_data_matrices(
            energy_rows,
            speed_rows,
            to_file=args.matrices_txt,
        )
        if args.matrices_txt is not None:
            print(f"Matrix text also saved to {args.matrices_txt}")


if __name__ == "__main__":
    main()
