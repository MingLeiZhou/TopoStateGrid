"""Optional pandapower network support."""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np

from .builder import _append_bidirectional, _make_data, _normalized_abs
from .parser import ParsedCase


PANDAPOWER_INSTALL_MESSAGE = 'pandapower support is optional; install it with: pip install -e ".[pandapower]"'


def build_graph_from_pandapower(
    net: Any,
    network_id: str = "pandapower_net",
    sample_id: str = "sample_0",
    timestamp: str | None = None,
    scenario_id: str | None = None,
    contingency_id: str | None = None,
):
    """Convert a pandapower net object into a homogeneous bus-branch PyG graph."""

    if importlib.util.find_spec("pandapower") is None:
        raise ImportError(PANDAPOWER_INSTALL_MESSAGE)
    if not hasattr(net, "bus"):
        raise ValueError("pandapower net object must provide a bus table")
    if net.bus.empty:
        raise ValueError("pandapower net bus table must not be empty")

    bus_ids = list(net.bus.index)
    bus_to_idx = {bus_id: idx for idx, bus_id in enumerate(bus_ids)}
    bus_status = _table_column(net.bus, "in_service", default=1.0, index=bus_ids)
    bus_type = np.ones(len(bus_ids), dtype=float)
    _mark_generator_buses(net, bus_type, bus_to_idx)
    _mark_ext_grid_buses(net, bus_type, bus_to_idx)

    pd, qd = _aggregate_loads(net, bus_to_idx, len(bus_ids))
    vm = _result_column(net, "res_bus", "vm_pu", bus_ids)
    va = _result_column(net, "res_bus", "va_degree", bus_ids)
    vmax = _table_column(net.bus, "max_vm_pu", default=0.0, index=bus_ids)
    vmin = _table_column(net.bus, "min_vm_pu", default=0.0, index=bus_ids)
    x = np.column_stack([bus_status, bus_type, pd, qd, vm, va, vmax, vmin, _normalized_abs(pd)])

    edges: list[list[int]] = []
    attrs: list[list[float]] = []
    _append_line_edges(net, bus_to_idx, edges, attrs)
    _append_trafo_edges(net, bus_to_idx, edges, attrs)
    if edges:
        edge_index = np.asarray(edges, dtype=np.int64).T
        edge_attr = np.asarray(attrs, dtype=np.float32)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, 12), dtype=np.float32)

    parsed = ParsedCase(
        source_type="pandapower",
        network_id=network_id,
        sample_id=sample_id,
        base_mva=float(getattr(net, "sn_mva", 100.0) or 100.0),
        grid={},
        solution={},
        metadata={
            "input_format": "pandapower",
            "num_buses": len(bus_ids),
            "num_lines": len(getattr(net, "line", [])) if hasattr(net, "line") else 0,
            "num_trafos": len(getattr(net, "trafo", [])) if hasattr(net, "trafo") else 0,
        },
        timestamp=timestamp,
        scenario_id=scenario_id or sample_id,
        contingency_id=contingency_id,
    )
    return _make_data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        parsed=parsed,
        notes=(
            "pandapower graph; line rate_a uses max_i_ka as an approximate rating proxy "
            "when no direct MVA rating is available."
        ),
    )


def _append_line_edges(net: Any, bus_to_idx: dict[Any, int], edges: list[list[int]], attrs: list[list[float]]) -> None:
    if not hasattr(net, "line") or net.line.empty:
        return
    for line_idx, row in net.line.iterrows():
        from_bus = row.get("from_bus")
        to_bus = row.get("to_bus")
        if from_bus not in bus_to_idx or to_bus not in bus_to_idx:
            raise ValueError(f"pandapower line {line_idx!r} references an unknown bus")
        length = _finite(row.get("length_km", 1.0), default=1.0)
        r = _finite(row.get("r_ohm_per_km", 0.0)) * length
        x = _finite(row.get("x_ohm_per_km", 0.0)) * length
        b = _finite(row.get("c_nf_per_km", 0.0)) * length
        rate = _finite(row.get("max_i_ka", 0.0))
        outage = 0.0 if bool(row.get("in_service", True)) else 1.0
        res = getattr(net, "res_line", None)
        attr = [
            0.0,
            r,
            x,
            b,
            b,
            rate,
            _res_value(res, line_idx, "p_from_mw"),
            _res_value(res, line_idx, "q_from_mvar"),
            _res_value(res, line_idx, "p_to_mw"),
            _res_value(res, line_idx, "q_to_mvar"),
            _res_value(res, line_idx, "loading_percent") / 100.0,
            outage,
        ]
        _append_bidirectional(edges, attrs, bus_to_idx[from_bus], bus_to_idx[to_bus], attr)


def _append_trafo_edges(net: Any, bus_to_idx: dict[Any, int], edges: list[list[int]], attrs: list[list[float]]) -> None:
    if not hasattr(net, "trafo") or net.trafo.empty:
        return
    for trafo_idx, row in net.trafo.iterrows():
        hv_bus = row.get("hv_bus")
        lv_bus = row.get("lv_bus")
        if hv_bus not in bus_to_idx or lv_bus not in bus_to_idx:
            raise ValueError(f"pandapower transformer {trafo_idx!r} references an unknown bus")
        outage = 0.0 if bool(row.get("in_service", True)) else 1.0
        res = getattr(net, "res_trafo", None)
        attr = [
            1.0,
            _finite(row.get("vkr_percent", 0.0)),
            _finite(row.get("vk_percent", 0.0)),
            0.0,
            0.0,
            _finite(row.get("sn_mva", 0.0)),
            _res_value(res, trafo_idx, "p_hv_mw"),
            _res_value(res, trafo_idx, "q_hv_mvar"),
            _res_value(res, trafo_idx, "p_lv_mw"),
            _res_value(res, trafo_idx, "q_lv_mvar"),
            _res_value(res, trafo_idx, "loading_percent") / 100.0,
            outage,
        ]
        _append_bidirectional(edges, attrs, bus_to_idx[hv_bus], bus_to_idx[lv_bus], attr)


def _aggregate_loads(net: Any, bus_to_idx: dict[Any, int], n_bus: int) -> tuple[np.ndarray, np.ndarray]:
    pd = np.zeros(n_bus, dtype=float)
    qd = np.zeros(n_bus, dtype=float)
    if not hasattr(net, "load") or net.load.empty:
        return pd, qd
    for load_idx, row in net.load.iterrows():
        if not bool(row.get("in_service", True)):
            continue
        bus = row.get("bus")
        if bus not in bus_to_idx:
            raise ValueError(f"pandapower load {load_idx!r} references an unknown bus")
        idx = bus_to_idx[bus]
        pd[idx] += _finite(row.get("p_mw", 0.0))
        qd[idx] += _finite(row.get("q_mvar", 0.0))
    return pd, qd


def _mark_generator_buses(net: Any, bus_type: np.ndarray, bus_to_idx: dict[Any, int]) -> None:
    if not hasattr(net, "gen") or net.gen.empty:
        return
    for _, row in net.gen.iterrows():
        if not bool(row.get("in_service", True)):
            continue
        bus = row.get("bus")
        if bus in bus_to_idx:
            bus_type[bus_to_idx[bus]] = 2.0


def _mark_ext_grid_buses(net: Any, bus_type: np.ndarray, bus_to_idx: dict[Any, int]) -> None:
    if not hasattr(net, "ext_grid") or net.ext_grid.empty:
        return
    for _, row in net.ext_grid.iterrows():
        if not bool(row.get("in_service", True)):
            continue
        bus = row.get("bus")
        if bus in bus_to_idx:
            bus_type[bus_to_idx[bus]] = 3.0


def _table_column(table: Any, column: str, default: float, index: list[Any]) -> np.ndarray:
    values = np.full(len(index), default, dtype=float)
    if column not in table.columns:
        return values
    for pos, idx in enumerate(index):
        if idx in table.index:
            values[pos] = _finite(table.at[idx, column], default=default)
    return values


def _result_column(net: Any, table_name: str, column: str, index: list[Any]) -> np.ndarray:
    result = getattr(net, table_name, None)
    values = np.zeros(len(index), dtype=float)
    if result is None or column not in result.columns:
        return values
    for pos, idx in enumerate(index):
        if idx in result.index:
            values[pos] = _finite(result.at[idx, column])
    return values


def _res_value(result: Any, index: Any, column: str) -> float:
    if result is None or column not in result.columns or index not in result.index:
        return 0.0
    return _finite(result.at[index, column])


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if np.isfinite(numeric) else default
