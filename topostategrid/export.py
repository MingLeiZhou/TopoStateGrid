"""Disk export helpers for generated graph datasets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import torch


def save_graphs(graphs: Sequence[Any], path: str | Path) -> Path:
    """Save graphs with torch.save."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(list(graphs), path)
    return path


def load_graphs(path: str | Path) -> list[Any]:
    """Load graphs saved by `save_graphs` across PyTorch default changes."""

    path = Path(path)
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def write_metadata_csv(graphs: Sequence[Any], path: str | Path) -> Path:
    """Write a compact metadata table for graph samples."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "index",
        "network_id",
        "sample_id",
        "timestamp",
        "scenario_id",
        "contingency_id",
        "source_type",
        "num_nodes",
        "num_edges",
        "num_node_features",
        "num_edge_features",
        "risk_score",
        "y",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx, graph in enumerate(graphs):
            has_label = _has_label(graph)
            writer.writerow(
                {
                    "index": idx,
                    "network_id": getattr(graph, "network_id", ""),
                    "sample_id": getattr(graph, "sample_id", ""),
                    "timestamp": getattr(graph, "timestamp", ""),
                    "scenario_id": getattr(graph, "scenario_id", ""),
                    "contingency_id": getattr(graph, "contingency_id", ""),
                    "source_type": getattr(graph, "source_type", ""),
                    "num_nodes": getattr(graph, "num_nodes", graph.x.shape[0] if hasattr(graph, "x") else ""),
                    "num_edges": graph.edge_index.shape[1] if hasattr(graph, "edge_index") else "",
                    "num_node_features": graph.x.shape[1] if hasattr(graph, "x") and graph.x.ndim == 2 else "",
                    "num_edge_features": graph.edge_attr.shape[1]
                    if hasattr(graph, "edge_attr") and graph.edge_attr.ndim == 2
                    else "",
                    "risk_score": _first_scalar(getattr(graph, "risk_score", "")) if has_label else "",
                    "y": _first_scalar(getattr(graph, "y", "")) if has_label else "",
                }
            )
    return path


def save_split_json(split: dict[str, Any], path: str | Path) -> Path:
    """Save split indices as JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(split, handle, indent=2)
    return path


def export_dataset(
    graphs: Sequence[Any],
    output_dir: str | Path = "outputs",
    split: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write graphs, metadata, optional split, and a generated README."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "graphs": save_graphs(graphs, output_dir / "graphs.pt"),
        "metadata": write_metadata_csv(graphs, output_dir / "metadata.csv"),
        "readme": _write_generated_readme(graphs, output_dir / "README_generated.md"),
    }
    if split is not None:
        paths["split"] = save_split_json(split, output_dir / "split.json")
    return paths


def _write_generated_readme(graphs: Sequence[Any], path: Path) -> Path:
    networks = sorted({str(getattr(graph, "network_id", "")) for graph in graphs})
    path.write_text(
        "\n".join(
            [
                "# TopoStateGrid Generated Dataset",
                "",
                f"Graphs: {len(graphs)}",
                f"Networks: {', '.join(networks) if networks else 'unknown'}",
                "",
                "Each graph is a PyTorch Geometric Data object with bus nodes, bidirectional branch edges,",
                "node features, edge features, preserved metadata, and optional proxy labels.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _first_scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return ""
        return value.detach().cpu().reshape(-1)[0].item()
    return value


def _has_label(graph: Any) -> bool:
    value = getattr(graph, "has_label", False)
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return False
        return bool(value.detach().cpu().reshape(-1)[0].item())
    return bool(value)
