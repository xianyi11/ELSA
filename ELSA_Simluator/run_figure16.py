#!/usr/bin/env python3
"""
Figure 16 (QANN-style): Energy Saving, Speed Up, Norm. AEDP vs Eyeriss [20].

Layout and *n/a* handling match the reference script: nine logical slots
``works[0..8]``; bars are drawn only where ``val != 0``. Group widths are fixed
(``grouplen``: 6 / 6 / 6 / 7 / 6).

- Published **energy** (figure 1) + **latency** (figure 2) per row give
  ``energy_saving = E_eye/E_arch`` and ``speed_up = L_eye/L_arch``.

- **Norm. AEDP** for non-ELSA accelerators is taken **directly from Fig. 16** (hardcoded);
  **ELSA** energy/latency come from the simulator and **Norm. AEDP (ELSA)** uses::

    norm_AEDP_elsa = (A_elsa / A_eyeriss) * (E_elsa/E_eye) * (L_elsa/L_eye)

  with ``A_elsa = 100.23 mm^2``, ``A_eyeriss = 7.749 mm^2``.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_figure17 as f17

CONV_ROOT = f17.CONV_ROOT
TRANS_ROOT = f17.TRANS_ROOT

# Normalized AEDP for ELSA only: AEDP ∝ A·E·D → scale by area ratio vs Eyeriss.
AREA_ELSA_MM2 = 100.23
AREA_EYERISS_MM2 = 7.749 # 4.76 mm^2 for the 16 MB eDRAM@28nm

R18 = "resnet18"
R34 = "resnet34"
R50 = "resnet50"
VIT = "vit_small"
GEO = "geomean_resnet"
RESNET_ROWS: Tuple[str, ...] = (R18, R34, R50)

# Norm. AEDP from Fig. 16 (hardcoded). Keys = ``WORKS`` index; omit ELSA (8).
# ResNet rows use ViTALiTy (index 5), not LLH-CIM, per reference figure.
_AEDP_FIG16: Dict[str, Dict[int, float]] = {
    R18: {0: 1.0, 1: 9.0e-2, 2: 2.3e-2, 3: 1.0e-2, 5: 1.9e-2},
    R34: {0: 1.0, 1: 1.0e-1, 2: 2.4e-2, 3: 1.2e-2, 5: 2.2e-2},
    R50: {0: 1.0, 1: 7.0e-2, 2: 1.6e-2, 3: 8.2e-3, 5: 1.5e-2},
    VIT: {0: 1.0, 1: 9.4e-2, 2: 4.6e-2, 4: 2.4e-1, 5: 2.1e-2, 6: 1.2e-1},
    GEO: {0: 1.0, 1: 8.7e-2, 2: 2.1e-2, 3: 1.0e-2, 5: 1.9e-2},
}

# Reference legend / color indices (9 slots; last is ELSA).
WORKS = [
    "Eyeriss",
    "Eyeriss v2",
    "ANT",
    "S-CONV",
    "Sanger",
    "ViTALiTy",
    "AIOQAB",
    "LLH-CIM",
    "ELSA",
]
ELSA_IDX = 8
N_WORKS = len(WORKS)

# Energy (figure 1). Omitted keys = n/a for that row (``0`` in vectors).
_BASE_ENERGY: Dict[str, Dict[str, float]] = {
    R18: {
        "Eyeriss": 3.934,
        "Eyeriss v2": 1.5539,
        "AdaFloat": 5.744,
        "OLAccel": 3.961,
        "BitFusion": 4.925,
        "ANT": 2.032,
        "S-CONV": 0.7398,
        "LLH-CIM": 0.175362,
    },
    R34: {
        "Eyeriss": 7.46,
        "Eyeriss v2": 3.1507,
        "AdaFloat": 11.389,
        "OLAccel": 8.356,
        "BitFusion": 10.698,
        "ANT": 3.984,
        "S-CONV": 1.5,
        "LLH-CIM": 0.355556,
    },
    R50: {
        "Eyeriss": 10.676,
        "Eyeriss v2": 3.50171,
        "AdaFloat": 6.801,
        "OLAccel": 8.365,
        "BitFusion": 8.626,
        "ANT": 4.345,
        "S-CONV": 1.667,
        "LLH-CIM": 0.395169,
    },
    VIT: {
        "Eyeriss": 8.272,
        "Eyeriss v2": 3.638,
        "AdaFloat": 15.802,
        "OLAccel": 9.472,
        "BitFusion": 12.4,
        "ANT": 6.46,
        "Sanger": 23.29,
        "ViTALiTy": 6.799,
        "AIOQAB": 4.749,
    },
}

# Latency (figure 2).
_BASE_LAT: Dict[str, Dict[str, float]] = {
    R18: {
        "Eyeriss": 84.518,
        "Eyeriss v2": 23.63,
        "AdaFloat": 12.9384,
        "OLAccel": 11.52162,
        "BitFusion": 8.786215,
        "ANT": 3.1034,
        "S-CONV": 4.892,
        "LLH-CIM": 58.17308,
    },
    R34: {
        "Eyeriss": 159.265,
        "Eyeriss v2": 47.91,
        "AdaFloat": 25.75505,
        "OLAccel": 23.48708,
        "BitFusion": 19.385,
        "ANT": 5.85321,
        "S-CONV": 9.92,
        "LLH-CIM": 117.9487,
    },
    R50: {
        "Eyeriss": 203.166,
        "Eyeriss v2": 53.2,
        "AdaFloat": 28.44372,
        "OLAccel": 24.02814,
        "BitFusion": 15.39397,
        "ANT": 6.76706,
        "S-CONV": 11.02522,
        "LLH-CIM": 131.0897,
    },
    VIT: {
        "Eyeriss": 210.51,
        "Eyeriss v2": 55.3,
        "AdaFloat": 27.14588,
        "OLAccel": 23.93682,
        "BitFusion": 22.61054,
        "ANT": 10.4,
        "Sanger": 13.81025,
        "ViTALiTy": 4.131,
        "AIOQAB": 64.26735,
    },
}

# Reference: fixed bar counts per group (must match number of non-zero ``val``).
GROUPLEN: Dict[str, int] = {
    "ResNet_18": 6,
    "ResNet_34": 6,
    "ResNet_50": 6,
    "Vit_small": 7,
    "Geomean": 6,
}
GROUP_GAP = 1.0
BAR_WIDTH = 0.8

# Legend reorder + ncol as in reference.
LEGEND_NEW_ORDER = [0, 1, 2, 3, 6, 7, 8, 4, 5]
LEGEND_NCOL = 5


def _geom_mean(values: Sequence[float]) -> float:
    xs = [float(v) for v in values if v is not None and v > 0]
    if not xs:
        return 0.0
    return float(math.exp(sum(math.log(x) for x in xs) / len(xs)))


def _energy_speed_from_row(row_key: str) -> Tuple[List[float], List[float]]:
    """Energy saving and speed-up for ``works[0..7]`` from tables; slot 8 filled by ELSA."""
    e_row = _BASE_ENERGY[row_key]
    l_row = _BASE_LAT[row_key]
    e_eye = e_row["Eyeriss"]
    l_eye = l_row["Eyeriss"]
    es = [0.0] * N_WORKS
    sp = [0.0] * N_WORKS
    es[0] = sp[0] = 1.0
    for i in range(1, ELSA_IDX):
        name = WORKS[i]
        if name not in e_row or name not in l_row:
            continue
        ea = e_row[name]
        la = l_row[name]
        if ea <= 0 or la <= 0:
            continue
        es[i] = e_eye / ea
        sp[i] = l_eye / la
    return es, sp


def _aedp_vector_from_fig16(row_key: str) -> List[float]:
    """Published Norm. AEDP from Fig. 16; index ``ELSA_IDX`` left 0 for ``_insert_elsa``."""
    out = [0.0] * N_WORKS
    for idx, val in _AEDP_FIG16.get(row_key, {}).items():
        out[idx] = val
    return out


def _insert_elsa(
    es: List[float],
    sp: List[float],
    ad: List[float],
    row_key: str,
    lat_ms: float,
    en_mj: float,
) -> None:
    e_eye = _BASE_ENERGY[row_key]["Eyeriss"]
    l_eye = _BASE_LAT[row_key]["Eyeriss"]
    if lat_ms <= 0 or en_mj <= 0:
        raise ValueError(f"Invalid ELSA totals for {row_key}: lat_ms={lat_ms}, energy_mJ={en_mj}")
    es[ELSA_IDX] = e_eye / en_mj
    sp[ELSA_IDX] = l_eye / lat_ms
    area_ratio = AREA_ELSA_MM2 / AREA_EYERISS_MM2
    ad[ELSA_IDX] = area_ratio * (en_mj / e_eye) * (lat_ms / l_eye)


def _geomean_es_sp(
    triples: Sequence[Tuple[List[float], List[float], List[float]]]
) -> Tuple[List[float], List[float]]:
    out_e = [0.0] * N_WORKS
    out_s = [0.0] * N_WORKS
    for i in range(N_WORKS):
        out_e[i] = _geom_mean([t[0][i] for t in triples])
        out_s[i] = _geom_mean([t[1][i] for t in triples])
    return out_e, out_s


def _flip_row_major(items: List, ncol: int):
    return list(itertools.chain(*[items[i::ncol] for i in range(ncol)]))


def _build_x_arrays():
    import numpy as np

    g18 = GROUPLEN["ResNet_18"]
    g34 = GROUPLEN["ResNet_34"]
    g50 = GROUPLEN["ResNet_50"]
    gv = GROUPLEN["Vit_small"]
    gg = GROUPLEN["Geomean"]

    x1 = np.arange(g18, dtype=float)
    x2 = np.arange(g34, dtype=float) + g18 + GROUP_GAP
    x3 = np.arange(g50, dtype=float) + g34 + g18 + 2 * GROUP_GAP
    x4 = np.arange(gv, dtype=float) + g50 + g34 + g18 + 3 * GROUP_GAP
    x5 = np.arange(gg, dtype=float) + gv + g50 + g34 + g18 + 4 * GROUP_GAP
    return x1, x2, x3, x4, x5


def plot_figure(
    groups_es: List[List[float]],
    groups_sp: List[List[float]],
    groups_ad: List[List[float]],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    f17._register_figure17_fonts()
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False

    colors = sns.color_palette("flare", n_colors=10)
    hatches = ["", "//////"]

    x1, x2, x3, x4, x5 = _build_x_arrays()
    xs_list = [x1, x2, x3, x4, x5]

    group_names_bottom = [
        "ResNet-18\n(W4)",
        "ResNet-34\n(W5)",
        "ResNet-50\n(W6)",
        "Vit-Small\n(W7)",
        "Geomean\n@ResNet",
    ]

    plt.figure(figsize=(8, 3))

    # --- ax1 Energy Saving ---
    ax1 = plt.subplot(311)
    legend_labels_added = [False] * len(colors)

    def draw_energy_group(xv: np.ndarray, vec: List[float]) -> None:
        xi = 0
        for i, val in enumerate(vec):
            if val != 0:
                ax1.bar(
                    xv[xi],
                    val,
                    width=BAR_WIDTH,
                    color=colors[i],
                    label=f"{WORKS[i]}" if not legend_labels_added[i] else "",
                    hatch=hatches[i % 2],
                )
                ax1.text(
                    xv[xi] - 0.15,
                    val * 1.2,
                    f"{val:10.1f}",
                    ha="center",
                    va="bottom",
                    rotation=270,
                )
                legend_labels_added[i] = True
                xi += 1

    draw_energy_group(x1, groups_es[0])
    draw_energy_group(x2, groups_es[1])
    draw_energy_group(x3, groups_es[2])
    draw_energy_group(x4, groups_es[3])
    draw_energy_group(x5, groups_es[4])

    ax1.set_yscale("log", base=2)
    handles, labels = ax1.get_legend_handles_labels()
    handles = [handles[i] for i in LEGEND_NEW_ORDER if i < len(handles)]
    labels = [labels[i] for i in LEGEND_NEW_ORDER if i < len(labels)]
    handles = _flip_row_major(handles, LEGEND_NCOL)
    labels = _flip_row_major(labels, LEGEND_NCOL)
    ax1.legend(
        handles,
        labels,
        ncol=LEGEND_NCOL,
        bbox_to_anchor=(0.95, 1.1),
        frameon=False,
        loc="lower right",
    )
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_ylim(0, 64)
    ax1.set_ylabel("Energy Saving", y=0.6)


    # --- ax2 Speed Up ---
    ax2 = plt.subplot(312)
    legend_labels_added2 = [False] * len(colors)

    def draw_speed_group(gix: int, xv: np.ndarray, vec: List[float]) -> None:
        xi = 0
        for i, val in enumerate(vec):
            if val != 0:
                ax2.bar(
                    xv[xi],
                    val,
                    width=BAR_WIDTH,
                    color=colors[i],
                    label=f"{WORKS[i]}" if not legend_labels_added2[i] else "",
                    hatch=hatches[i % 2],
                )
                if gix == 2 and i == ELSA_IDX and val >= 100:
                    txt = f"{val:.0f}"
                else:
                    txt = f"{val:10.1f}"
                ax2.text(
                    xv[xi] - 0.15,
                    val * 1.2,
                    txt,
                    ha="center",
                    va="bottom",
                    rotation=270,
                )
                legend_labels_added2[i] = True
                xi += 1

    draw_speed_group(0, x1, groups_sp[0])
    draw_speed_group(1, x2, groups_sp[1])
    draw_speed_group(2, x3, groups_sp[2])
    draw_speed_group(3, x4, groups_sp[3])
    draw_speed_group(4, x5, groups_sp[4])

    ax2.set_yscale("log", base=2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_ylim(0, 400)
    ax2.set_ylabel("Speed Up")

    # --- ax3 Norm. AEDP ---
    ax3 = plt.subplot(313)
    legend_labels_added3 = [False] * len(colors)

    def aedp_text_yfac(gix: int) -> float:
        return 1.1 if gix in (3, 4) else 1.2

    def draw_aedp_group(gix: int, xv: np.ndarray, vec: List[float]) -> None:
        yfac = aedp_text_yfac(gix)
        xi = 0
        for i, val in enumerate(vec):
            if val != 0:
                ax3.bar(
                    xv[xi],
                    val,
                    width=BAR_WIDTH,
                    color=colors[i],
                    label=f"{WORKS[i]}" if not legend_labels_added3[i] else "",
                    hatch=hatches[i % 2],
                )
                if val == 1:
                    t = f"{val:.1f}"
                else:
                    t = f"{val:.1e}"
                ax3.text(
                    xv[xi] - 0.15,
                    val * yfac,
                    t,
                    ha="center",
                    va="bottom",
                    rotation=270,
                )
                legend_labels_added3[i] = True
                xi += 1

    draw_aedp_group(0, x1, groups_ad[0])
    draw_aedp_group(1, x2, groups_ad[1])
    draw_aedp_group(2, x3, groups_ad[2])
    draw_aedp_group(3, x4, groups_ad[3])
    draw_aedp_group(4, x5, groups_ad[4])

    group_centers = [float(np.mean(x)) for x in xs_list]
    ax3.set_yscale("log", base=2)
    ax3.set_xticks(group_centers, group_names_bottom)
    ax3.set_yticks([])
    ax3.set_ylim(0, 400)
    ax3.set_ylabel("Norm. AEDP")

    fmt = out_path.suffix.lower()
    if fmt in (".png", ".jpg", ".jpeg"):
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0.0, dpi=400)
    else:
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0.0)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Figure 16 (QANN-style comparison).")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "tracer_files",
        help="Per-model simulator outputs (calculateInfo .pth)",
    )
    parser.add_argument(
        "--skip-sim",
        action="store_true",
        help="Only load from cache-dir.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "QANN_benchMark_Comparison_new.png",
        help="Output image path (.pdf or .png)",
    )
    args = parser.parse_args()
    cache: Path = args.cache_dir
    cache.mkdir(parents=True, exist_ok=True)

    elsa: Dict[str, Tuple[float, float]] = {}
    raw: Dict[str, dict] = {}
    for row_key, mid in ((R18, "resnet18"), (R34, "resnet34"), (R50, "resnet50")):
        sub = cache / "convolution" / mid
        if not args.skip_sim:
            f17.run_elsa_subprocess(CONV_ROOT, mid, sub)
        e_j, l_cyc, lat_ms, en_mj, freq = f17.load_metrics_from_run(CONV_ROOT, mid, sub)
        elsa[row_key] = (lat_ms, en_mj)
        raw[row_key] = {
            "model": mid,
            "total_latency_ms": lat_ms,
            "total_energy_mJ": en_mj,
            "total_latency_cycles": l_cyc,
            "total_energy_J": e_j,
            "frequency_Hz": freq,
        }

    vit_sub = cache / "transformer" / "vit_small"
    if not args.skip_sim:
        f17.run_elsa_subprocess(TRANS_ROOT, "vit_small", vit_sub)
    v_e_j, v_l_cyc, v_ms, v_mj, v_freq = f17.load_metrics_from_run(
        TRANS_ROOT, "vit_small", vit_sub
    )
    raw["vit_small"] = {
        "model": "vit_small",
        "total_latency_ms": v_ms,
        "total_energy_mJ": v_mj,
        "total_latency_cycles": v_l_cyc,
        "total_energy_J": v_e_j,
        "frequency_Hz": v_freq,
    }

    triples: List[Tuple[List[float], List[float], List[float]]] = []
    for rk in RESNET_ROWS:
        es, sp = _energy_speed_from_row(rk)
        ad = _aedp_vector_from_fig16(rk)
        lat_ms, en_mj = elsa[rk]
        _insert_elsa(es, sp, ad, rk, lat_ms, en_mj)
        triples.append((es, sp, ad))

    vit_es, vit_sp = _energy_speed_from_row(VIT)
    vit_ad = _aedp_vector_from_fig16(VIT)
    _insert_elsa(vit_es, vit_sp, vit_ad, VIT, v_ms, v_mj)
    vit_t = (vit_es, vit_sp, vit_ad)

    geo_es, geo_sp = _geomean_es_sp(triples)
    geo_ad = _aedp_vector_from_fig16(GEO)
    geo_ad[ELSA_IDX] = _geom_mean([t[2][ELSA_IDX] for t in triples])
    geo = (geo_es, geo_sp, geo_ad)

    groups_es = [triples[0][0], triples[1][0], triples[2][0], vit_t[0], geo[0]]
    groups_sp = [triples[0][1], triples[1][1], triples[2][1], vit_t[1], geo[1]]
    groups_ad = [triples[0][2], triples[1][2], triples[2][2], vit_t[2], geo[2]]

    out_json = args.out.with_suffix(".json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "area_mm2": {"elsa": AREA_ELSA_MM2, "eyeriss": AREA_EYERISS_MM2},
                "elsa_raw": raw,
                "energy_saving": {
                    "resnet18": groups_es[0],
                    "resnet34": groups_es[1],
                    "resnet50": groups_es[2],
                    "vit_small": groups_es[3],
                    "geomean_resnet": groups_es[4],
                },
                "speedup": {
                    "resnet18": groups_sp[0],
                    "resnet34": groups_sp[1],
                    "resnet50": groups_sp[2],
                    "vit_small": groups_sp[3],
                    "geomean_resnet": groups_sp[4],
                },
                "norm_aedp": {
                    "resnet18": groups_ad[0],
                    "resnet34": groups_ad[1],
                    "resnet50": groups_ad[2],
                    "vit_small": groups_ad[3],
                    "geomean_resnet": groups_ad[4],
                },
                "works": WORKS,
            },
            f,
            indent=2,
        )

    plot_figure(groups_es, groups_sp, groups_ad, args.out)
    print(f"Wrote {args.out} and {out_json}")


if __name__ == "__main__":
    main()
