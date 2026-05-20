"""Graph builders for pandas DataFrame and CSV table inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .builder import EDGE_FEATURE_NAMES, _append_bidirectional, _make_data, _normalized_abs
from .parser import ParsedCase


BUS_STATE_COLUMNS = ("pd", "qd", "vm", "va")
BRANCH_STATE_COLUMNS = ("pf", "qf", "pt", "qt", "loading_ratio")


def build_graph_from_tables(
    bus_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    bus_state_df: pd.DataFrame | None = None,
    branch_state_df: pd.DataFrame | None = None,
    network_id: str = "table_network",
    sample_id: str = "sample_0",
    timestamp: str | None = None,
    scenario_id: str | None = None,
    contingency_id: str | None = None,
    bus_id_col: str = "bus_id",
    from_bus_col: str = "from_bus",
    to_bus_col: str = "to_bus",
):
    """Build a homogeneous bus-branch graph from pandas DataFrame tables."""

    bus_df = _require_dataframe(bus_df, "bus_df").copy()
    branch_df = _require_dataframe(branch_df, "branch_df").copy()
    bus_state_df = None if bus_state_df is None else _require_dataframe(bus_state_df, "bus_state_df").copy()
    branch_state_df = None if branch_state_df is None else _require_dataframe(branch_state_df, "branch_state_df").copy()

    _require_columns(bus_df, [bus_id_col], "bus_df")
    _require_columns(branch_df, [from_bus_col, to_bus_col], "branch_df")
    if bus_df.empty:
        raise ValueError("bus_df must not be empty")
    if branch_df.empty:
        raise ValueError("branch_df must not be empty")
    if bus_df[bus_id_col].duplicated().any():
        raise ValueError(f"bus_df contains duplicate bus ids in column {bus_id_col!r}")

    effective_timestamp = timestamp
    bus_state_df, bus_timestamp = _filter_state_by_timestamp(bus_state_df, timestamp, "bus_state_df")
    effective_timestamp = _merge_timestamp(effective_timestamp, bus_timestamp)
    branch_state_df, branch_timestamp = _filter_state_by_timestamp(branch_state_df, timestamp, "branch_state_df")
    effective_timestamp = _merge_timestamp(effective_timestamp, branch_timestamp)

    bus_ids = list(bus_df[bus_id_col])
    bus_to_idx = {bus_id: idx for idx, bus_id in enumerate(bus_ids)}
    _validate_branch_bus_refs(branch_df, bus_to_idx, from_bus_col, to_bus_col)

    bus_values = _bus_feature_frame(bus_df, bus_id_col)
    if bus_state_df is not None and not bus_state_df.empty:
        _apply_bus_state(bus_values, bus_state_df, bus_to_idx, bus_id_col)

    pd_values = bus_values["pd"].to_numpy(dtype=float)
    x = np.column_stack(
        [
            bus_values["bus_status"],
            bus_values["bus_type"],
            bus_values["pd"],
            bus_values["qd"],
            bus_values["vm"],
            bus_values["va"],
            bus_values["vmax"],
            bus_values["vmin"],
            _normalized_abs(pd_values),
        ]
    )

    branch_values = _branch_feature_frame(branch_df, from_bus_col, to_bus_col)
    if branch_state_df is not None and not branch_state_df.empty:
        _apply_branch_state(branch_values, branch_df, branch_state_df, from_bus_col, to_bus_col)

    edges: list[list[int]] = []
    attrs: list[list[float]] = []
    for idx, row in branch_values.iterrows():
        src = bus_to_idx[branch_df.loc[idx, from_bus_col]]
        dst = bus_to_idx[branch_df.loc[idx, to_bus_col]]
        attr = [float(row[name]) for name in EDGE_FEATURE_NAMES]
        _append_bidirectional(edges, attrs, src, dst, attr)

    edge_index = np.asarray(edges, dtype=np.int64).T
    edge_attr = np.asarray(attrs, dtype=np.float32)
    parsed = ParsedCase(
        source_type="tables",
        network_id=network_id,
        sample_id=sample_id,
        base_mva=100.0,
        grid={},
        solution={},
        metadata={
            "input_format": "pandas",
            "num_buses": len(bus_df),
            "num_branches": len(branch_df),
        },
        timestamp=effective_timestamp,
        scenario_id=scenario_id or sample_id,
        contingency_id=contingency_id,
    )
    return _make_data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        parsed=parsed,
        notes="Pandas table graph; state tables override matching static columns when provided.",
    )


def build_graph_from_csv_tables(
    bus_csv: str | Path,
    branch_csv: str | Path,
    bus_state_csv: str | Path | None = None,
    branch_state_csv: str | Path | None = None,
    **kwargs: Any,
):
    """Read CSV tables and build a homogeneous bus-branch graph."""

    bus_df = pd.read_csv(bus_csv)
    branch_df = pd.read_csv(branch_csv)
    bus_state_df = pd.read_csv(bus_state_csv) if bus_state_csv is not None else None
    branch_state_df = pd.read_csv(branch_state_csv) if branch_state_csv is not None else None
    return build_graph_from_tables(bus_df, branch_df, bus_state_df, branch_state_df, **kwargs)


def _require_dataframe(value: Any, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame")
    return value


def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required column(s): {', '.join(missing)}")


def _filter_state_by_timestamp(
    df: pd.DataFrame | None,
    timestamp: str | None,
    name: str,
) -> tuple[pd.DataFrame | None, str | None]:
    if df is None or df.empty or "timestamp" not in df.columns:
        return df, None

    timestamps = [value for value in df["timestamp"].dropna().unique().tolist()]
    if timestamp is not None:
        filtered = df[df["timestamp"] == timestamp].copy()
        if filtered.empty:
            raise ValueError(f"{name} contains timestamp values but no rows match timestamp={timestamp!r}")
        return filtered, timestamp
    if len(timestamps) > 1:
        raise ValueError(f"{name} contains multiple timestamps; pass timestamp=... to select one")
    if len(timestamps) == 1:
        selected = timestamps[0]
        return df[df["timestamp"] == selected].copy(), str(selected)
    return df, None


def _merge_timestamp(current: str | None, candidate: str | None) -> str | None:
    if candidate is None:
        return current
    if current is None:
        return candidate
    if str(current) != str(candidate):
        raise ValueError(f"State tables refer to conflicting timestamps: {current!r} and {candidate!r}")
    return current


def _bus_feature_frame(bus_df: pd.DataFrame, bus_id_col: str) -> pd.DataFrame:
    features = pd.DataFrame(index=bus_df.index)
    defaults = {
        "bus_status": 0.0,
        "bus_type": 0.0,
        "pd": 0.0,
        "qd": 0.0,
        "vm": 0.0,
        "va": 0.0,
        "vmax": 0.0,
        "vmin": 0.0,
    }
    for column, default in defaults.items():
        features[column] = _numeric_column(bus_df, column, default, f"bus_df.{column}")
    features[bus_id_col] = bus_df[bus_id_col].values
    return features


def _branch_feature_frame(branch_df: pd.DataFrame, from_bus_col: str, to_bus_col: str) -> pd.DataFrame:
    features = pd.DataFrame(index=branch_df.index)
    for column in EDGE_FEATURE_NAMES:
        features[column] = _numeric_column(branch_df, column, 0.0, f"branch_df.{column}")
    features[from_bus_col] = branch_df[from_bus_col].values
    features[to_bus_col] = branch_df[to_bus_col].values
    if "branch_id" in branch_df.columns:
        features["branch_id"] = branch_df["branch_id"].values
        if branch_df["branch_id"].duplicated().any():
            raise ValueError("branch_df contains duplicate branch_id values")
    return features


def _apply_bus_state(
    bus_values: pd.DataFrame,
    bus_state_df: pd.DataFrame,
    bus_to_idx: dict[Any, int],
    bus_id_col: str,
) -> None:
    _require_columns(bus_state_df, [bus_id_col], "bus_state_df")
    if bus_state_df[bus_id_col].duplicated().any():
        raise ValueError(f"bus_state_df contains duplicate bus ids in column {bus_id_col!r}")
    unknown = [bus_id for bus_id in bus_state_df[bus_id_col] if bus_id not in bus_to_idx]
    if unknown:
        raise ValueError(f"bus_state_df references unknown bus id(s): {unknown}")

    state = bus_state_df.set_index(bus_id_col)
    for column in BUS_STATE_COLUMNS:
        if column in state.columns:
            values = _numeric_column(state, column, 0.0, f"bus_state_df.{column}")
            for bus_id, value in values.items():
                row_idx = bus_values.index[bus_to_idx[bus_id]]
                bus_values.loc[row_idx, column] = value


def _apply_branch_state(
    branch_values: pd.DataFrame,
    branch_df: pd.DataFrame,
    branch_state_df: pd.DataFrame,
    from_bus_col: str,
    to_bus_col: str,
) -> None:
    if "branch_id" in branch_df.columns and "branch_id" in branch_state_df.columns:
        _apply_branch_state_by_key(branch_values, branch_state_df, "branch_id", "branch_state_df.branch_id")
        return

    _require_columns(branch_state_df, [from_bus_col, to_bus_col], "branch_state_df")
    if branch_df.duplicated([from_bus_col, to_bus_col]).any():
        raise ValueError("branch_state_df join by from/to is ambiguous because branch_df has duplicate pairs")
    if branch_state_df.duplicated([from_bus_col, to_bus_col]).any():
        raise ValueError("branch_state_df contains duplicate from/to pairs")

    key_to_idx = {
        (row[from_bus_col], row[to_bus_col]): idx
        for idx, row in branch_df.iterrows()
    }
    for _, row in branch_state_df.iterrows():
        key = (row[from_bus_col], row[to_bus_col])
        if key not in key_to_idx:
            raise ValueError(f"branch_state_df references unknown branch pair: {key!r}")
        target_idx = key_to_idx[key]
        for column in BRANCH_STATE_COLUMNS:
            if column in branch_state_df.columns:
                branch_values.loc[target_idx, column] = _numeric_scalar(row[column], f"branch_state_df.{column}")


def _apply_branch_state_by_key(
    branch_values: pd.DataFrame,
    branch_state_df: pd.DataFrame,
    key_col: str,
    key_name: str,
) -> None:
    if branch_state_df[key_col].duplicated().any():
        raise ValueError(f"branch_state_df contains duplicate {key_col} values")
    state = branch_state_df.set_index(key_col)
    static_keys = set(branch_values[key_col].tolist())
    unknown = [key for key in state.index.tolist() if key not in static_keys]
    if unknown:
        raise ValueError(f"branch_state_df references unknown branch_id value(s): {unknown}")
    for branch_id, row in state.iterrows():
        target_idx = branch_values.index[branch_values[key_col] == branch_id][0]
        for column in BRANCH_STATE_COLUMNS:
            if column in state.columns:
                branch_values.loc[target_idx, column] = _numeric_scalar(row[column], f"{key_name}.{column}")


def _validate_branch_bus_refs(
    branch_df: pd.DataFrame,
    bus_to_idx: dict[Any, int],
    from_bus_col: str,
    to_bus_col: str,
) -> None:
    unknown: list[Any] = []
    for value in list(branch_df[from_bus_col]) + list(branch_df[to_bus_col]):
        if value not in bus_to_idx:
            unknown.append(value)
    if unknown:
        raise ValueError(f"branch_df references unknown bus id(s): {unknown}")


def _numeric_column(df: pd.DataFrame, column: str, default: float, name: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    try:
        values = pd.to_numeric(df[column], errors="raise").astype(float)
    except Exception as exc:
        raise ValueError(f"Column {name} contains non-numeric values") from exc
    values = values.fillna(default)
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"Column {name} contains non-finite values")
    return values


def _numeric_scalar(value: Any, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Column {name} contains non-numeric values") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"Column {name} contains non-finite values")
    return numeric
