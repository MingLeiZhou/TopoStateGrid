"""Label attachment utilities."""

from __future__ import annotations

from typing import Any

import torch
from torch_geometric.data import Data


def attach_labels(
    data: Data,
    y: Any | None = None,
    y_cls: Any | None = None,
    y_reg: Any | None = None,
    risk_score: Any | None = None,
) -> Data:
    """Attach user-provided labels to a graph in-place and return it."""

    if y is not None:
        data.y = _tensor_1d(y)
    if y_cls is not None:
        data.y_cls = _tensor_1d(y_cls, dtype=torch.long)
        if y is None:
            data.y = data.y_cls
    if y_reg is not None:
        data.y_reg = _tensor_1d(y_reg)
        if y is None and y_cls is None:
            data.y = data.y_reg
    if risk_score is not None:
        data.risk_score = _tensor_1d(risk_score)
    return data


def attach_stress_proxy_labels(
    data: Data,
    threshold: float = 1.0,
    loading_feature: str = "loading_ratio",
    overwrite: bool = False,
) -> Data:
    """Attach temporary stress labels derived from max branch loading.

    This is a proxy label for prototyping graph construction. It is not a real
    cascading-failure or reliability target. Existing labels are preserved by
    default; pass ``overwrite=True`` to replace them intentionally.
    """

    existing = _existing_label_fields(data)
    if existing and not overwrite:
        raise ValueError(
            "Proxy label attachment would overwrite existing label fields "
            f"{existing}; pass overwrite=True to replace them."
        )

    feature_names = getattr(data, "edge_feature_names", [])
    if loading_feature not in feature_names or data.edge_attr.numel() == 0:
        risk = 0.0
    else:
        idx = feature_names.index(loading_feature)
        values = data.edge_attr[:, idx]
        finite_values = values[torch.isfinite(values)]
        risk = float(finite_values.max().item()) if finite_values.numel() else 0.0

    data.risk_score = torch.tensor([risk], dtype=torch.float32)
    data.y_reg = torch.tensor([risk], dtype=torch.float32)
    data.y_cls = torch.tensor([1 if risk > threshold else 0], dtype=torch.long)
    data.y = data.y_cls
    data.label_notes = (
        "Temporary proxy: y_cls = 1 if max loading_ratio exceeds the threshold; "
        "not a cascading-failure ground-truth label."
    )
    return data


def _tensor_1d(value: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1)
    return tensor


def _existing_label_fields(data: Data) -> list[str]:
    existing: list[str] = []
    for name in ("y", "y_cls", "y_reg", "risk_score"):
        if hasattr(data, name) and getattr(data, name) is not None:
            existing.append(name)
    return existing
