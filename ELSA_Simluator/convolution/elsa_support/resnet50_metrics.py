"""
Aggregate energy / latency / area / throughput metrics from a ``calculateInfo`` dict.

Logic matches the legacy analysis script (ResNet50 Conv + NoC); optional transformer
breakdown keys are read with ``.get`` and default to 0.
"""
from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any, Dict, Optional

import torch

DEFAULT_FREQUENCY = 2e8
DEFAULT_ENERGY_BASE = 2.4950907339734142e-05 - 9e-6
DEFAULT_TILE_NUM = 36
DEFAULT_TILE_AREA = 2.71
DEFAULT_FLIT_SIZE = 256
DEFAULT_TILE_H = 6
DEFAULT_TILE_W = 6

DEFAULT_AVG_UNICAST = 3.89713e-11
DEFAULT_AVG_UNITCAST_INNER = 1.4684275e-11
DEFAULT_AVG_UNICAST_HOP = 3.0
DEFAULT_ROUTER_TO_LOCAL = 1.38525e-13


def _path_suggests_vgg(path: str) -> bool:
    return "vgg" in path.lower()


def _model_id_suggests_vgg(model_id: Optional[str]) -> bool:
    """YAML / CLI model ids are e.g. ``vgg16_cifar10`` — reliable when layer keys omit \"vgg\"."""
    if not model_id:
        return False
    return "vgg" in str(model_id).lower()


def _looks_like_vgg_calculate_info(info: Dict[str, Any]) -> bool:
    """Heuristic: VGG exports use ``features.*`` + ``classifier`` names; path may contain ``vgg``."""
    keys = [str(k) for k in info.keys() if k != "transmitTraffic"]
    joined = " ".join(keys).lower()
    if "vgg" in joined:
        return True
    has_features = any(k.startswith("features.") for k in keys)
    has_classifier = any("classifier" in k for k in keys)
    return bool(has_features and has_classifier)


def _resolve_energy_base_offset(
    energy_base_offset: Optional[float],
    info: Dict[str, Any],
    *,
    path_hint: Optional[str] = None,
    model_id: Optional[str] = None,
) -> float:
    """
    ResNet-style runs use ``DEFAULT_ENERGY_BASE`` as a fixed offset; VGG should not (use 0).

    If ``energy_base_offset`` is not ``None``, it is returned unchanged.
    Otherwise: 0 when ``--model`` / path / layer keys suggest VGG, else ``DEFAULT_ENERGY_BASE``.
    """
    if energy_base_offset is not None:
        return float(energy_base_offset)
    # if _model_id_suggests_vgg(model_id):
    #     return 0.0
    # if path_hint is not None and _path_suggests_vgg(path_hint):
    #     return 0.0
    # if _looks_like_vgg_calculate_info(info):
    #     return 0.0
    if model_id == "resnet50":
        return DEFAULT_ENERGY_BASE
    else:
        return 0.0


def _load_calculate_info_pth(path: str, map_location: str) -> Any:
    """Load legacy calculateInfo dicts (arbitrary pickled objects, not weights-only tensors)."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*weights_only.*",
            category=FutureWarning,
        )
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)


def _f(x: Any, default: float = 0.0) -> float:
    if x is None:
        return default
    if torch.is_tensor(x):
        return float(x.detach().cpu().item()) if x.numel() else default
    return float(x)


def compute_resnet50_metrics(
    info: Dict[str, Any],
    *,
    frequency: float = DEFAULT_FREQUENCY,
    energy_base_offset: Optional[float] = None,
    path_hint: Optional[str] = None,
    model_id: Optional[str] = None,
    tile_num: int = DEFAULT_TILE_NUM,
    tile_area: float = DEFAULT_TILE_AREA,
    flitsize: int = DEFAULT_FLIT_SIZE,
    tile_h: int = DEFAULT_TILE_H,
    tile_w: int = DEFAULT_TILE_W,
    avg_unicast: float = DEFAULT_AVG_UNICAST,
    avg_unitcast_inner: float = DEFAULT_AVG_UNITCAST_INNER,
    avg_unicast_hop: float = DEFAULT_AVG_UNICAST_HOP,
    router_to_local: float = DEFAULT_ROUTER_TO_LOCAL,
    verbose: bool = True,
    plot: bool = False,
    transformer_noc_reference: bool = False,
) -> Dict[str, float]:
    """
    Parameters
    ----------
    info :
        Dict as produced by ``run_resnet50_vary_t`` / ``torch.load`` on calculateInfo*.pth.
        Must include per-layer entries with ``Total Latency``, ``Total Energy``, etc.
        ``transmitTraffic`` may be popped; copied internally.
    plot :
        If True, try to show energy/area pie charts (requires matplotlib).
    energy_base_offset :
        If ``None`` (default), uses ``0`` for VGG-style ``calculateInfo`` and
        ``DEFAULT_ENERGY_BASE`` for ResNet-style; pass a float to override.
    path_hint :
        Optional file path used when loading ``calculateInfo`` (e.g. filename
        contains ``vgg``); improves VGG detection when layer names are generic.
    model_id :
        Optional ``configs/elsa_models.yaml`` model key (e.g. ``vgg16_cifar10``).
        Passed automatically by ``run_elsa.py``; layer names alone often omit ``vgg``.
    transformer_noc_reference :
        If ``True`` (ViT / legacy transformer scripts): zero ``transmitTraffic`` diagonal
        before NoC cost, and **do not** halve ``traffic_cost`` or ``total_traffic``.
        Convolution ResNet path uses ``False`` (symmetric traffic double-counting removed by ``/2``).
    """
    info = deepcopy(info)
    energy_base_offset = _resolve_energy_base_offset(
        energy_base_offset, info, path_hint=path_hint, model_id=model_id
    )
    transmit_traffic = info.pop("transmitTraffic", None)
    if transmit_traffic is None:
        transmit_traffic = [[0.0]]
    if not torch.is_tensor(transmit_traffic):
        transmit_traffic = torch.tensor(transmit_traffic, dtype=torch.float64)
    else:
        transmit_traffic = transmit_traffic.double()

    energy = energy_base_offset
    ops = 0.0
    area_total = 0.0
    spike_number = 0.0
    all_layer_timestep = []
    effective_area_per_layer = []
    calculate_cycle = 0.0
    total_cycle_for_pipeline = 0.0
    total_sparsity = 0.0
    index1 = 0
    total_input_flit = 0.0
    latency_last = 0.0

    for key in list(info.keys()):
        if "Total Latency" not in info[key]:
            continue
        index1 += 1
        total_sparsity += 1 - _f(info[key].get("sparsity", 0))
        latency_last = _f(info[key]["Total Latency"])
        total_cycle_for_pipeline += latency_last
        energy += _f(info[key]["Total Energy"])
        ops += _f(info[key].get("numbers of operation", 0)) * 2
        total_input_flit += _f(info[key].get("InputFlitNum", 0))
        if "numbers of spikes" in info[key]:
            spike_number += _f(info[key].get("one spike operation", 0))

        for inkey in info[key].keys():
            if "Area" in inkey and inkey != "Total Area":
                area_total += _f(info[key][inkey])

        layer_per_timestep = []
        ptc = info[key]["perTimeStepCycle"]
        effective_area_per_layer.append(_f(info[key]["Total Area"]))
        for t in range(len(ptc)):
            begin = 0.0
            end = _f(ptc[t])
            layer_per_timestep.append((begin, end))
            if end - begin > 0:
                calculate_cycle += end - begin
        all_layer_timestep.append(layer_per_timestep)

    new_effective = []
    for i, a in enumerate(effective_area_per_layer):
        if (i % 10 == 8 or i % 10 == 9) and i != 0:
            continue
        new_effective.append(a)
    layer_num = len(new_effective)
    update_tile_area = deepcopy(new_effective)
    tile_allocate = [1] * layer_num if layer_num else []
    if layer_num > 0:
        tile_allocate[0] = 0
    cur_tile_num = layer_num
    idx = 0
    while tile_num < cur_tile_num and layer_num > 0:
        if idx >= layer_num - 1:
            break
        if update_tile_area[idx] + update_tile_area[idx + 1] < tile_area:
            area_combine = update_tile_area[idx] + update_tile_area[idx + 1]
            update_tile_area[idx + 1] = area_combine
            update_tile_area[idx] = area_combine
            tile_allocate[idx + 1] = 0
            idx += 1
            cur_tile_num -= 1
        else:
            idx += 1

    _order = [-1]
    for i in range(tile_h):
        if i % 2 == 0:
            for j in range(tile_w):
                _order.append(i * tile_w + j)
        else:
            for j in range(tile_w):
                _order.append((i + 1) * tile_w - j - 1)

    total_traffic = 0.0
    for i in range(transmit_traffic.shape[0]):
        for j in range(transmit_traffic.shape[1]):
            total_traffic += float(transmit_traffic[i, j])

    if transformer_noc_reference:
        tt_for_cost = transmit_traffic.clone()
        d = min(tt_for_cost.shape[0], tt_for_cost.shape[1])
        for i in range(d):
            tt_for_cost[i, i] = 0.0
        tt_for_balance = tt_for_cost.float()
    else:
        tt_for_cost = transmit_traffic
        tt_for_balance = transmit_traffic.float()

    pos = tt_for_balance[tt_for_balance > 0]
    traffic_balance = pos.reshape(-1).std().item() if pos.numel() else 0.0

    traffic_cost = 0.0
    for i in range(tt_for_cost.shape[0]):
        for j in range(tt_for_cost.shape[1]):
            v = float(tt_for_cost[i, j])
            if i == j:
                traffic_cost += (v / 256.0) * (
                    router_to_local + avg_unitcast_inner / avg_unicast_hop
                )
            else:
                traffic_cost += (v / 256.0) * avg_unicast / avg_unicast_hop
    if not transformer_noc_reference:
        traffic_cost /= 2.0
        total_traffic /= 2.0

    energy_tile = energy
    energy_total = energy + traffic_cost

    tops_per_w = (ops / energy_total) / 1e12 if energy_total > 0 else float("nan")
    tsops_per_w = (spike_number * 2 / energy_total) / 1e12 if energy_total > 0 else float("nan")
    gops = (ops * frequency / latency_last) / 1e9 if latency_last > 0 else float("nan")
    gsops = (spike_number * 2 * frequency / latency_last) / 1e9 if latency_last > 0 else float("nan")
    gops_per_mm2 = (ops * frequency / latency_last) / area_total / 1e9 if latency_last > 0 and area_total > 0 else float("nan")

    avg_sparsity = total_sparsity / index1 if index1 else 0.0
    pipe_ratio = (calculate_cycle / total_cycle_for_pipeline * 100) if total_cycle_for_pipeline > 0 else 0.0
    inject_rate = (
        (total_input_flit / (latency_last * tile_num)) * 100
        if latency_last > 0 and tile_num > 0
        else 0.0
    )

    # Component breakdown (optional keys for transformer / extended blocks)
    membrane_energy = membrane_area = 0.0
    weight_buffer_energy = weight_buffer_area = 0.0
    input_buffer_energy = input_buffer_area = 0.0
    spike_tracer_energy = spike_tracer_area = 0.0
    fire_component_energy = fire_component_area = 0.0
    adder_energy = adder_area = 0.0
    aysn_img2col_energy = aysn_img2col_area = 0.0
    vsa_order_energy = vsa_order_area = 0.0
    vsa_arb_area = vsa_arb_energy = 0.0
    flit_gen_area = flit_gen_energy = 0.0
    flit_dec_area = flit_dec_energy = 0.0
    routing_engine_area = routing_engine_energy = 0.0
    crossbar_area = crossbar_energy = 0.0
    layernorm_area = layernorm_energy = 0.0
    addition_area = addition_energy = 0.0
    softmax_area = softmax_energy = 0.0
    tsram_area = 0.0
    sram_area = 0.0

    for key in info.keys():
        if "Total Latency" not in info[key]:
            continue
        lk = info[key]
        if "addition" in key:
            layernorm_area += _f(lk.get("LayernormFunctionArea", 0))
            layernorm_energy += _f(lk.get("LayernormFunctionEnergy", 0))
            addition_area += _f(lk.get("AdditionFunctionArea", 0))
            addition_energy += _f(lk.get("AdditionFunctionEnergy", 0))
        if "attn_qkMulti" in key:
            softmax_area += _f(lk.get("softmaxFunctionArea", 0))
            softmax_energy += _f(lk.get("softmaxFunctionEnergy", 0))

        if "addition" not in key:
            membrane_energy += _f(lk.get("mambraneEnergy", 0))
            membrane_area += _f(lk.get("membraneArea", 0))
            weight_buffer_energy += _f(lk.get("weightBufferEnergy", 0))
            weight_buffer_area += _f(lk.get("weightBufferArea", 0))
            input_buffer_energy += _f(lk.get("inputBufferEnergy", 0))
            input_buffer_area += _f(lk.get("inputBufferArea", 0))
            spike_tracer_energy += _f(lk.get("spikeTracerEnergy", 0))
            spike_tracer_area += _f(lk.get("spikeTracerArea", 0))
            fire_component_energy += _f(lk.get("fireComponentEnergy", 0))
            fire_component_area += _f(lk.get("fireComponentArea", 0))
            adder_energy += _f(lk.get("adderVectorEnergy", 0))
            adder_area += _f(lk.get("adderVectorArea", 0))
            if "attn_qkMulti" in key:
                tsram_area += (
                    _f(lk.get("weightBufferArea", 0))
                    + _f(lk.get("inputBufferArea", 0))
                    + _f(lk.get("membraneArea", 0))
                )
                sram_area += _f(lk.get("spikeTracerArea", 0))
            else:
                sram_area += (
                    _f(lk.get("weightBufferArea", 0))
                    + _f(lk.get("inputBufferArea", 0))
                    + _f(lk.get("membraneArea", 0))
                    + _f(lk.get("spikeTracerArea", 0))
                )

        if "AysnImg2ColEnergy" in lk:
            aysn_img2col_energy += _f(lk.get("AysnImg2ColEnergy", 0))
            aysn_img2col_area += _f(lk.get("AysnImg2ColArea", 0))
            vsa_order_energy += _f(lk.get("VSAOrderControllerEnergy", 0))
            vsa_order_area += _f(lk.get("VSAOrderControllerArea", 0))
            vsa_arb_area += _f(lk.get("VSAUpdateArbiterArea", 0))
            vsa_arb_energy += _f(lk.get("VSAUpdateArbiterEnergy", 0))
        flit_gen_area += _f(lk.get("FlitGeneratorArea", 0))
        flit_gen_energy += _f(lk.get("FlitGeneratorEnergy", 0))
        flit_dec_area += _f(lk.get("FlitDecoderArea", 0))
        flit_dec_energy += _f(lk.get("FlitDecoderEnergy", 0))
        routing_engine_area += _f(lk.get("RoutingEngineArea", 0))
        routing_engine_energy += _f(lk.get("RoutingEngineEnergy", 0))
        crossbar_area += _f(lk.get("CrossbarSwitchArea", 0))
        crossbar_energy += _f(lk.get("CrossbarSwitchEnergy", 0))

    router_energy = (
        aysn_img2col_energy
        + vsa_order_energy
        + vsa_arb_energy
        + flit_gen_energy
        + flit_dec_energy
        + routing_engine_energy
        + crossbar_energy
        + layernorm_energy
        + addition_energy
        + softmax_energy
    )
    router_area = (
        aysn_img2col_area
        + vsa_order_area
        + vsa_arb_area
        + flit_gen_area
        + flit_dec_area
        + routing_engine_area
        + crossbar_area
        + softmax_area
    )

    out: Dict[str, float] = {
        "area_mm2": area_total,
        "energy_tile_J": energy_tile,
        "energy_noc_J": traffic_cost,
        "energy_total_J": energy_total,
        "latency_last_layer_s": latency_last / frequency,
        "latency_last_layer_ms": (latency_last / frequency) * 1000,
        "total_traffic_mb": total_traffic / (1024 * 1024 * 8),
        "ops_gops": ops / 1e9,
        "spike_ops_gsops": spike_number * 2 / 1e9,
        "tops_per_w": tops_per_w,
        "tsops_per_w": tsops_per_w,
        "gops_throughput": gops,
        "gsops_throughput": gsops,
        "gops_per_mm2": gops_per_mm2,
        "pipeline_calculate_ratio_pct": pipe_ratio,
        "avg_sparsity": avg_sparsity,
        "total_input_flit": total_input_flit,
        "inject_rate_pct": inject_rate,
        "traffic_balance": traffic_balance,
        "ops": ops,
        "frequency": frequency,
        "latency_cycles_last_layer": latency_last,
        "tile_energy_mj": energy_tile * 1000,
        "noc_energy_mj": traffic_cost * 1000,
        "energy_total_mj": energy_total * 1000,
        "tsram_area": tsram_area,
        "sram_area": sram_area,
        "router_energy_breakdown": router_energy,
        "membrane_energy": membrane_energy,
    }

    if verbose:
        print(f"Area:{area_total} mm^2")
        print(f"Energy:{energy_total * 1000} mJ")
        print(f"Latency:{(latency_last / frequency) * 1000} ms")
        print(f"operation:{ops / 1e9} GOPS")
        print(f"spiking operation:{spike_number * 2 / 1e9} GSOPS")
        print(f"TOPS/W:{tops_per_w}")
        print(f"TSOPS/W:{tsops_per_w}")
        print(f"GOPS:{gops}")
        print(f"GSOPS:{gsops}")
        print(f"GOPs/mm^2:{gops_per_mm2}")
    if plot:
        try:
            import matplotlib.pyplot as plt

            labels = [
                "membraneEnergy",
                "inputBufferEnergy",
                "weightBufferEnergy",
                "spikeTracerEnergy",
                "FireComponentEnergy",
                "AdderEnergy",
                "routerEnergy",
                "NoCEnergy",
            ]
            plt.figure(figsize=(9, 9))
            plt.pie(pie_sizes, labels=labels, autopct="%1.1f%%", shadow=True, startangle=90)
            plt.axis("equal")
            plt.show()

            plt.figure(figsize=(9, 9))
            labels2 = [
                "membrane",
                "weightBuffer",
                "inputBuffer",
                "spikeTracer",
                "FireComponent",
                "\n\nAdder",
                "\n\n\n\nRouter",
            ]
            sizes2 = [
                membrane_area,
                weight_buffer_area,
                input_buffer_area,
                spike_tracer_area,
                fire_component_area,
                adder_area,
                router_area,
            ]
            plt.pie(sizes2, labels=labels2, autopct="%1.1f%%", shadow=True, startangle=90)
            plt.axis("equal")
            plt.show()
        except Exception as e:
            print(f"[metrics] plot skipped: {e}")

    return out


def compute_resnet50_metrics_from_path(
    path: str,
    *,
    map_location: str = "cpu",
    **kwargs: Any,
) -> Dict[str, float]:
    info = _load_calculate_info_pth(path, map_location)
    kwargs = {**kwargs, "path_hint": path}
    return compute_resnet50_metrics(info, **kwargs)
