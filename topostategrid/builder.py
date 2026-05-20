"""Homogeneous bus-branch graph construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from .parser import ParsedCase, discover_opfdata_examples, parse_matpower_case, parse_opfdata_sample

NODE_FEATURE_NAMES = [
    "bus_status",
    "bus_type",
    "pd",
    "qd",
    "vm",
    "va",
    "vmax",
    "vmin",
    "normalized_demand",
]

EDGE_FEATURE_NAMES = [
    "component_type",
    "r",
    "x",
    "b_from",
    "b_to",
    "rate_a",
    "pf",
    "qf",
    "pt",
    "qt",
    "loading_ratio",
    "outage_flag",
]


def build_graph(parsed: ParsedCase, attach_proxy_label: bool = False) -> Data:
    """Build a PyTorch Geometric Data object from a parsed case."""

    if parsed.source_type == "opfdata":
        data = _build_opfdata_graph(parsed)
    elif parsed.source_type == "matpower":
        data = _build_matpower_graph(parsed)
    else:
        raise ValueError(f"Unsupported parsed case type: {parsed.source_type}")

    if attach_proxy_label:
        from .labels import attach_stress_proxy_labels

        attach_stress_proxy_labels(data)
    return data


def build_graph_from_opfdata_json(path: str | Path, attach_proxy_label: bool = True) -> Data:
    """Parse one OPFData JSON sample and build one graph."""

    return build_graph(parse_opfdata_sample(path), attach_proxy_label=attach_proxy_label)


def build_graphs_from_opfdata(
    root: str | Path = "data/opfdata",
    network_id: str | None = None,
    limit: int | None = None,
    attach_proxy_label: bool = True,
) -> list[Data]:
    """Build multiple graph samples from extracted OPFData JSON scenarios."""

    paths = discover_opfdata_examples(root=root, network_id=network_id, limit=limit)
    return [build_graph_from_opfdata_json(path, attach_proxy_label=attach_proxy_label) for path in paths]


def build_graph_from_matpower(path: str | Path, attach_proxy_label: bool = False) -> Data:
    """Build a static graph from a MATPOWER/PGLib `.m` case file."""

    return build_graph(parse_matpower_case(path), attach_proxy_label=attach_proxy_label)


def _build_opfdata_graph(parsed: ParsedCase) -> Data:
    grid_nodes = parsed.grid.get("nodes") or {}
    grid_edges = parsed.grid.get("edges") or {}
    solution_nodes = parsed.solution.get("nodes") or {}
    solution_edges = parsed.solution.get("edges") or {}

    bus = _as_2d_array(grid_nodes.get("bus"), width=4)
    n_bus = bus.shape[0]
    if n_bus == 0:
        raise ValueError(f"OPFData sample has no bus nodes: {parsed.path}")

    bus_status = _column(bus, 0, default=1.0)
    bus_type = _column(bus, 1, default=0.0)
    vmin = _column(bus, 2, default=0.0)
    vmax = _column(bus, 3, default=0.0)

    sol_bus = _as_2d_array(solution_nodes.get("bus"), width=2, rows=n_bus)
    va = _column(sol_bus, 0, rows=n_bus)
    vm = _column(sol_bus, 1, rows=n_bus)

    pd, qd = _aggregate_component_to_bus(
        grid_nodes.get("load"),
        grid_edges.get("load_link"),
        n_bus,
        feature_indices=(0, 1),
    )
    demand_norm = _normalized_abs(pd)

    x = np.column_stack([bus_status, bus_type, pd, qd, vm, va, vmax, vmin, demand_norm])
    edge_index, edge_attr = _opfdata_edges(grid_edges, solution_edges, n_bus)

    return _make_data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        parsed=parsed,
        notes="OPFData JSON scenario; load values and solved states are scenario-dependent.",
    )


def _build_matpower_graph(parsed: ParsedCase) -> Data:
    bus = _as_2d_array(parsed.grid.get("bus"), width=13)
    branch = _as_2d_array(parsed.grid.get("branch"), width=13)
    n_bus = bus.shape[0]
    if n_bus == 0:
        raise ValueError(f"MATPOWER case has no bus table: {parsed.path}")

    base = parsed.base_mva if parsed.base_mva else 100.0
    bus_ids = bus[:, 0].astype(int)
    bus_to_idx = {bus_id: idx for idx, bus_id in enumerate(bus_ids)}

    pd = _column(bus, 2, rows=n_bus) / base
    qd = _column(bus, 3, rows=n_bus) / base
    vm = _column(bus, 7, rows=n_bus)
    va = np.deg2rad(_column(bus, 8, rows=n_bus))
    vmax = _column(bus, 11, rows=n_bus)
    vmin = _column(bus, 12, rows=n_bus)
    demand_norm = _normalized_abs(pd)
    x = np.column_stack([np.ones(n_bus), _column(bus, 1, rows=n_bus), pd, qd, vm, va, vmax, vmin, demand_norm])

    edges: list[list[int]] = []
    attrs: list[list[float]] = []
    for row in branch:
        f_bus = int(row[0])
        t_bus = int(row[1])
        if f_bus not in bus_to_idx or t_bus not in bus_to_idx:
            continue
        src = bus_to_idx[f_bus]
        dst = bus_to_idx[t_bus]
        r = _value(row, 2)
        x_val = _value(row, 3)
        b = _value(row, 4)
        rate = _value(row, 5) / base if _value(row, 5) else 0.0
        ratio = _value(row, 8)
        shift = _value(row, 9)
        status = _value(row, 10, default=1.0)
        outage = 0.0 if status > 0 else 1.0
        component_type = 1.0 if ratio not in (0.0, 1.0) or shift != 0.0 else 0.0
        attr = [component_type, r, x_val, b, b, rate, 0.0, 0.0, 0.0, 0.0, 0.0, outage]
        _append_bidirectional(edges, attrs, src, dst, attr)

    edge_index = np.asarray(edges, dtype=np.int64).T if edges else np.empty((2, 0), dtype=np.int64)
    edge_attr = np.asarray(attrs, dtype=np.float32) if attrs else np.empty((0, len(EDGE_FEATURE_NAMES)), dtype=np.float32)

    return _make_data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        parsed=parsed,
        notes="Static MATPOWER/PGLib case; solved operating-state flow fields are unavailable.",
    )


def _opfdata_edges(grid_edges: dict[str, Any], solution_edges: dict[str, Any], n_bus: int) -> tuple[np.ndarray, np.ndarray]:
    edges: list[list[int]] = []
    attrs: list[list[float]] = []

    for edge_kind in ("ac_line", "transformer"):
        static = grid_edges.get(edge_kind) or {}
        solution = solution_edges.get(edge_kind) or {}
        senders = static.get("senders") or []
        receivers = static.get("receivers") or []
        static_features = static.get("features") or []
        solution_features = solution.get("features") or []

        for idx, (src, dst) in enumerate(zip(senders, receivers)):
            if src >= n_bus or dst >= n_bus:
                continue
            static_feature = static_features[idx] if idx < len(static_features) else []
            solution_feature = solution_features[idx] if idx < len(solution_features) else []
            attr = _opfdata_edge_attr(edge_kind, static_feature, solution_feature)
            reverse = attr.copy()
            reverse[3], reverse[4] = attr[4], attr[3]
            reverse[6], reverse[7], reverse[8], reverse[9] = attr[8], attr[9], attr[6], attr[7]
            edges.append([int(src), int(dst)])
            attrs.append(attr)
            edges.append([int(dst), int(src)])
            attrs.append(reverse)

    edge_index = np.asarray(edges, dtype=np.int64).T if edges else np.empty((2, 0), dtype=np.int64)
    edge_attr = np.asarray(attrs, dtype=np.float32) if attrs else np.empty((0, len(EDGE_FEATURE_NAMES)), dtype=np.float32)
    return edge_index, edge_attr


def _opfdata_edge_attr(edge_kind: str, static_feature: list[float], solution_feature: list[float]) -> list[float]:
    if edge_kind == "ac_line":
        component_type = 0.0
        b_from = _value(static_feature, 2)
        b_to = _value(static_feature, 3)
        r = _value(static_feature, 4)
        x_val = _value(static_feature, 5)
        rate = _value(static_feature, 6)
    else:
        component_type = 1.0
        b_from = 0.0
        b_to = 0.0
        r = _value(static_feature, 2)
        x_val = _value(static_feature, 3)
        rate = _value(static_feature, 4)

    pf = _value(solution_feature, 0)
    qf = _value(solution_feature, 1)
    pt = _value(solution_feature, 2)
    qt = _value(solution_feature, 3)
    loading = _loading_ratio(pf, qf, pt, qt, rate)
    return [component_type, r, x_val, b_from, b_to, rate, pf, qf, pt, qt, loading, 0.0]


def _aggregate_component_to_bus(
    features: Any,
    link: dict[str, Any] | None,
    n_bus: int,
    feature_indices: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    out_a = np.zeros(n_bus, dtype=float)
    out_b = np.zeros(n_bus, dtype=float)
    component_features = _as_2d_array(features, width=max(feature_indices) + 1)
    if component_features.size == 0 or not link:
        return out_a, out_b

    pairs = _component_to_bus_pairs(
        link.get("senders") or [],
        link.get("receivers") or [],
        component_count=component_features.shape[0],
        bus_count=n_bus,
    )
    for component_idx, bus_idx in pairs:
        out_a[bus_idx] += _value(component_features[component_idx], feature_indices[0])
        out_b[bus_idx] += _value(component_features[component_idx], feature_indices[1])
    return out_a, out_b


def _component_to_bus_pairs(
    senders: list[int],
    receivers: list[int],
    component_count: int,
    bus_count: int,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for sender, receiver in zip(senders, receivers):
        if 0 <= sender < component_count and 0 <= receiver < bus_count:
            pairs.append((int(sender), int(receiver)))
        elif 0 <= receiver < component_count and 0 <= sender < bus_count:
            pairs.append((int(receiver), int(sender)))
    return pairs


def _make_data(
    x: np.ndarray,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    parsed: ParsedCase,
    notes: str,
) -> Data:
    data = Data(
        x=torch.as_tensor(np.nan_to_num(x, nan=0.0), dtype=torch.float32),
        edge_index=torch.as_tensor(edge_index, dtype=torch.long),
        edge_attr=torch.as_tensor(np.nan_to_num(edge_attr, nan=0.0), dtype=torch.float32),
    )
    data.y = torch.tensor([-1], dtype=torch.long)
    data.y_cls = torch.tensor([-1], dtype=torch.long)
    data.y_reg = torch.tensor([0.0], dtype=torch.float32)
    data.risk_score = torch.tensor([0.0], dtype=torch.float32)
    data.has_label = torch.tensor([False], dtype=torch.bool)
    data.label_state = "missing"
    data.num_nodes = int(x.shape[0])
    data.network_id = parsed.network_id
    data.sample_id = parsed.sample_id
    data.timestamp = parsed.timestamp if parsed.timestamp is not None else ""
    data.scenario_id = parsed.scenario_id if parsed.scenario_id is not None else ""
    data.contingency_id = parsed.contingency_id if parsed.contingency_id is not None else ""
    data.base_mva = float(parsed.base_mva)
    data.source_type = parsed.source_type
    data.source_format = parsed.source_type
    data.source_path = parsed.path if parsed.path is not None else ""
    data.node_feature_names = list(NODE_FEATURE_NAMES)
    data.edge_feature_names = list(EDGE_FEATURE_NAMES)
    data.metadata_json = _metadata_to_json(parsed.metadata)
    data.metadata = data.metadata_json
    data.construction_notes = notes
    data.label_notes = ""
    return data


def _append_bidirectional(
    edges: list[list[int]],
    attrs: list[list[float]],
    src: int,
    dst: int,
    attr: list[float],
) -> None:
    edges.append([int(src), int(dst)])
    attrs.append(attr)
    reverse = attr.copy()
    reverse[3], reverse[4] = attr[4], attr[3]
    reverse[6], reverse[7], reverse[8], reverse[9] = attr[8], attr[9], attr[6], attr[7]
    edges.append([int(dst), int(src)])
    attrs.append(reverse)


def _loading_ratio(pf: float, qf: float, pt: float, qt: float, rate: float) -> float:
    if rate <= 0:
        return 0.0
    s_from = float(np.hypot(pf, qf))
    s_to = float(np.hypot(pt, qt))
    return max(s_from, s_to) / rate


def _normalized_abs(values: np.ndarray) -> np.ndarray:
    scale = float(np.nanmax(np.abs(values))) if values.size else 0.0
    if scale <= 0:
        return np.zeros_like(values, dtype=float)
    return values / scale


def _as_2d_array(values: Any, width: int, rows: int | None = None) -> np.ndarray:
    if values is None:
        row_count = 0 if rows is None else rows
        return np.zeros((row_count, width), dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        row_count = 0 if rows is None else rows
        return np.zeros((row_count, width), dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if rows is not None and arr.shape[0] < rows:
        padded = np.zeros((rows, max(width, arr.shape[1])), dtype=float)
        padded[: arr.shape[0], : arr.shape[1]] = arr
        arr = padded
    return arr


def _column(arr: np.ndarray, idx: int, default: float = 0.0, rows: int | None = None) -> np.ndarray:
    row_count = rows if rows is not None else arr.shape[0]
    if arr.size == 0 or idx >= arr.shape[1]:
        return np.full(row_count, default, dtype=float)
    col = arr[:row_count, idx].astype(float)
    if len(col) < row_count:
        padded = np.full(row_count, default, dtype=float)
        padded[: len(col)] = col
        return padded
    return col


def _value(row: Any, idx: int, default: float = 0.0) -> float:
    try:
        if idx >= len(row):
            return default
        value = float(row[idx])
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _metadata_to_json(metadata: dict[str, Any]) -> str:
    return json.dumps(dict(metadata), sort_keys=True, default=str)
