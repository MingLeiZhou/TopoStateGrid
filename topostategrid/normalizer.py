"""Feature normalization without training-set leakage."""

from __future__ import annotations

from typing import Any, Sequence

import torch


class FeatureNormalizer:
    """Standardize node and edge features using statistics fit on train graphs."""

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps
        self.node_mean: torch.Tensor | None = None
        self.node_std: torch.Tensor | None = None
        self.edge_mean: torch.Tensor | None = None
        self.edge_std: torch.Tensor | None = None

    def fit(self, train_dataset: Sequence[Any]) -> "FeatureNormalizer":
        """Fit normalization statistics from the training split only."""

        node_tensors = [graph.x for graph in train_dataset if hasattr(graph, "x") and graph.x.numel()]
        edge_tensors = [graph.edge_attr for graph in train_dataset if hasattr(graph, "edge_attr") and graph.edge_attr.numel()]
        self.node_mean, self.node_std = _fit_stats(node_tensors, self.eps)
        self.edge_mean, self.edge_std = _fit_stats(edge_tensors, self.eps)
        return self

    def transform(self, dataset: Sequence[Any], in_place: bool = False) -> list[Any]:
        """Transform a dataset using the fitted statistics."""

        transformed: list[Any] = []
        for graph in dataset:
            item = graph if in_place else graph.clone()
            if self.node_mean is not None and hasattr(item, "x") and item.x.numel():
                item.x = _standardize(item.x, self.node_mean, self.node_std)
            if self.edge_mean is not None and hasattr(item, "edge_attr") and item.edge_attr.numel():
                item.edge_attr = _standardize(item.edge_attr, self.edge_mean, self.edge_std)
            item.normalized = True
            transformed.append(item)
        return transformed

    def fit_transform(self, train_dataset: Sequence[Any], in_place: bool = False) -> list[Any]:
        """Fit on and transform a training dataset."""

        self.fit(train_dataset)
        return self.transform(train_dataset, in_place=in_place)


def _fit_stats(tensors: list[torch.Tensor], eps: float) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not tensors:
        return None, None
    values = torch.cat([tensor.detach().float() for tensor in tensors], dim=0)
    finite = torch.isfinite(values)
    nan_values = torch.where(finite, values, torch.full_like(values, float("nan")))
    mean = torch.nanmean(nan_values, dim=0)
    mean = torch.nan_to_num(mean, nan=0.0)
    centered = torch.where(finite, values - mean, torch.full_like(values, float("nan")))
    var = torch.nanmean(centered * centered, dim=0)
    std = torch.sqrt(torch.nan_to_num(var, nan=0.0))
    std = torch.where(std < eps, torch.ones_like(std), std)
    return mean, std


def _standardize(values: torch.Tensor, mean: torch.Tensor | None, std: torch.Tensor | None) -> torch.Tensor:
    if mean is None or std is None:
        return values
    clean = torch.where(torch.isfinite(values), values, mean.to(values.device))
    return (clean - mean.to(values.device)) / std.to(values.device)
