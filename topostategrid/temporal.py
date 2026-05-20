"""Temporal window helpers for ordered graph samples."""

from __future__ import annotations

import math
from typing import Any, Sequence


def make_temporal_windows(
    graphs: Sequence[Any],
    input_window: int = 6,
    forecast_horizon: int = 1,
    target: str = "y",
    sort_by_timestamp: bool = True,
) -> list[dict[str, Any]]:
    """Create simple past-graphs-to-future-target windows.

    When ``sort_by_timestamp`` is true and all graphs carry valid comparable
    timestamps, windows are built in chronological order. Otherwise, the input
    order is treated as the temporal or scenario order.
    """

    if input_window <= 0:
        raise ValueError("input_window must be positive")
    if forecast_horizon <= 0:
        raise ValueError("forecast_horizon must be positive")

    ordered_graphs = _order_graphs(graphs, sort_by_timestamp=sort_by_timestamp)
    windows: list[dict[str, Any]] = []
    total = len(ordered_graphs)
    required = input_window + forecast_horizon
    if total < required:
        return windows

    for start in range(total - required + 1):
        end = start + input_window
        target_index = end + forecast_horizon - 1
        target_graph = ordered_graphs[target_index]
        if not hasattr(target_graph, target):
            raise AttributeError(f"Target graph at index {target_index} has no attribute {target!r}")
        context_graphs = list(ordered_graphs[start:end])
        windows.append(
            {
                "graphs": context_graphs,
                "target": getattr(target_graph, target),
                "target_graph": target_graph,
                "target_index": target_index,
                "timestamps": [getattr(graph, "timestamp", None) for graph in context_graphs],
                "target_timestamp": getattr(target_graph, "timestamp", None),
                "sample_ids": [getattr(graph, "sample_id", None) for graph in context_graphs],
                "target_sample_id": getattr(target_graph, "sample_id", None),
            }
        )
    return windows


def _order_graphs(graphs: Sequence[Any], sort_by_timestamp: bool) -> list[Any]:
    ordered = list(graphs)
    if not sort_by_timestamp:
        return ordered

    timestamps = [getattr(graph, "timestamp", None) for graph in ordered]
    if not timestamps or not all(_is_valid_timestamp(timestamp) for timestamp in timestamps):
        return ordered
    try:
        return sorted(ordered, key=lambda graph: getattr(graph, "timestamp"))
    except TypeError:
        return ordered


def _is_valid_timestamp(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "nan", "none", "null", "nat"}
    try:
        return not math.isnan(float(value))
    except (TypeError, ValueError):
        return True
