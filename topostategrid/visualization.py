"""Visualization helpers for generated graph samples."""

from __future__ import annotations

import importlib
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def render_graph_sequence(
    graphs: Any | Sequence[Any],
    output_path: str | Path,
    node_value: str | int | None = "vm",
    edge_value: str | int | None = "loading_ratio",
    fps: int = 2,
    layout: str = "spring",
    seed: int = 42,
    figsize: tuple[float, float] = (6.0, 5.0),
    dpi: int = 120,
    title: str | None = None,
    show_labels: bool = False,
    fixed_layout: bool = True,
) -> Path:
    """Render one graph or a graph sequence to GIF or MP4.

    The renderer is intended for lightweight inspection of constructed graph
    datasets. Nodes are colored by a node feature such as ``vm`` and edges by
    an edge feature such as ``loading_ratio``. It does not simulate grid
    dynamics; it only visualizes existing graph samples.
    """

    output_path = Path(output_path)
    suffix = output_path.suffix.lower()
    if suffix not in {".gif", ".mp4", ".m4v"}:
        raise ValueError("output_path must end with .gif, .mp4, or .m4v")
    if fps <= 0:
        raise ValueError("fps must be positive")

    graph_list = _coerce_graphs(graphs)
    if not graph_list:
        raise ValueError("graphs must contain at least one graph")

    plt, animation, nx = _require_visual_dependencies()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    node_idx = _feature_index(graph_list[0], node_value, "node_feature_names", "x")
    edge_idx = _feature_index(graph_list[0], edge_value, "edge_feature_names", "edge_attr")
    node_range = _feature_range(graph_list, "x", node_idx)
    edge_range = _feature_range(graph_list, "edge_attr", edge_idx)
    if fixed_layout:
        first_layout = _make_layout(nx, graph_list[0], layout=layout, seed=seed)
        first_node_count = _num_nodes(graph_list[0])
        layouts = [
            first_layout if _num_nodes(graph) == first_node_count else _make_layout(nx, graph, layout=layout, seed=seed)
            for graph in graph_list
        ]
    else:
        layouts = [_make_layout(nx, graph, layout=layout, seed=seed) for graph in graph_list]

    fig, ax = plt.subplots(figsize=figsize)

    def update(frame_idx: int) -> None:
        graph = graph_list[frame_idx]
        pos = layouts[frame_idx]
        node_count = _num_nodes(graph)
        edge_pairs, edge_values = _undirected_edge_values(graph, edge_idx)
        graph_nx = nx.Graph()
        graph_nx.add_nodes_from(range(node_count))
        graph_nx.add_edges_from(edge_pairs)

        ax.clear()
        node_colors = _node_values(graph, node_idx)
        node_size = 280 if node_count <= 30 else max(40, 5000 / max(node_count, 1))

        nx.draw_networkx_nodes(
            graph_nx,
            pos,
            ax=ax,
            node_color=node_colors,
            cmap=plt.cm.viridis,
            vmin=node_range[0],
            vmax=node_range[1],
            node_size=node_size,
            linewidths=0.6,
            edgecolors="#1f2937",
        )
        if edge_pairs:
            widths = _edge_widths(edge_values)
            nx.draw_networkx_edges(
                graph_nx,
                pos,
                edgelist=edge_pairs,
                ax=ax,
                edge_color=edge_values,
                edge_cmap=plt.cm.plasma,
                edge_vmin=edge_range[0],
                edge_vmax=edge_range[1],
                width=widths,
                alpha=0.85,
            )
        if show_labels and node_count <= 80:
            nx.draw_networkx_labels(graph_nx, pos, ax=ax, font_size=7)

        frame_title = _frame_title(graph, frame_idx, title)
        ax.set_title(frame_title, fontsize=11)
        ax.text(
            0.01,
            0.01,
            f"node: {_feature_label(node_value)} | edge: {_feature_label(edge_value)}",
            transform=ax.transAxes,
            fontsize=8,
            color="#374151",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.85},
        )
        ax.set_axis_off()
        ax.margins(0.15)

    movie = animation.FuncAnimation(fig, update, frames=len(graph_list), interval=1000 / fps, repeat=True)
    if suffix == ".gif":
        writer = animation.PillowWriter(fps=fps)
    else:
        if not animation.writers.is_available("ffmpeg"):
            plt.close(fig)
            raise RuntimeError("MP4 rendering requires ffmpeg. Use a .gif output path or install ffmpeg.")
        writer = animation.FFMpegWriter(fps=fps)
    movie.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return output_path


def _coerce_graphs(graphs: Any | Sequence[Any]) -> list[Any]:
    if hasattr(graphs, "x") and hasattr(graphs, "edge_index"):
        return [graphs]
    if isinstance(graphs, (str, bytes, Path)):
        raise ValueError("graphs must be a graph object or a sequence of graph objects")
    return list(graphs)


def _require_visual_dependencies():
    _prepare_matplotlib_config()
    try:
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        plt = importlib.import_module("matplotlib.pyplot")
        animation = importlib.import_module("matplotlib.animation")
        nx = importlib.import_module("networkx")
        importlib.import_module("PIL")
    except ImportError as exc:
        raise ImportError(
            "Graph rendering requires optional visualization dependencies. "
            'Install them with: pip install -e ".[visual]"'
        ) from exc
    return plt, animation, nx


def _prepare_matplotlib_config() -> None:
    if "MPLCONFIGDIR" in os.environ:
        return
    default_config = Path.home() / ".matplotlib"
    if default_config.exists() and os.access(default_config, os.W_OK):
        return
    config_dir = Path(tempfile.gettempdir()) / "topostategrid-matplotlib"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(config_dir)


def _feature_index(graph: Any, selector: str | int | None, names_attr: str, tensor_attr: str) -> int | None:
    if selector is None:
        return None
    tensor = getattr(graph, tensor_attr, None)
    if tensor is None or getattr(tensor, "ndim", 0) != 2:
        return None
    width = int(tensor.shape[1])
    if isinstance(selector, int):
        if selector < 0 or selector >= width:
            raise ValueError(f"Feature index {selector} is out of bounds for {tensor_attr} with width {width}")
        return selector
    names = list(getattr(graph, names_attr, []) or [])
    if selector not in names:
        raise ValueError(f"Unknown feature {selector!r}; available {names_attr}: {names}")
    return names.index(selector)


def _feature_range(graphs: Sequence[Any], tensor_attr: str, feature_idx: int | None) -> tuple[float, float]:
    values: list[float] = []
    for graph in graphs:
        tensor = getattr(graph, tensor_attr, None)
        if tensor is None or feature_idx is None or getattr(tensor, "numel", lambda: 0)() == 0:
            continue
        arr = tensor.detach().cpu().numpy()[:, feature_idx].astype(float)
        values.extend(arr[np.isfinite(arr)].tolist())
    if not values:
        return 0.0, 1.0
    low = float(min(values))
    high = float(max(values))
    if math.isclose(low, high):
        margin = abs(low) * 0.05 + 1.0
        return low - margin, high + margin
    return low, high


def _make_layout(nx: Any, graph: Any, layout: str, seed: int) -> dict[int, np.ndarray]:
    node_count = _num_nodes(graph)
    edges = _undirected_edge_values(graph, edge_feature_idx=None)[0]
    graph_nx = nx.Graph()
    graph_nx.add_nodes_from(range(node_count))
    graph_nx.add_edges_from(edges)
    if layout == "spring":
        return nx.spring_layout(graph_nx, seed=seed)
    if layout == "circular":
        return nx.circular_layout(graph_nx)
    if layout == "kamada_kawai":
        return nx.kamada_kawai_layout(graph_nx)
    raise ValueError("layout must be one of: spring, circular, kamada_kawai")


def _num_nodes(graph: Any) -> int:
    explicit = getattr(graph, "num_nodes", None)
    if explicit is not None:
        return int(explicit)
    return int(graph.x.shape[0])


def _node_values(graph: Any, feature_idx: int | None) -> np.ndarray:
    node_count = _num_nodes(graph)
    if feature_idx is None or not hasattr(graph, "x") or graph.x.numel() == 0:
        return np.zeros(node_count, dtype=float)
    values = graph.x.detach().cpu().numpy()[:, feature_idx].astype(float)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _undirected_edge_values(graph: Any, edge_feature_idx: int | None) -> tuple[list[tuple[int, int]], list[float]]:
    if not hasattr(graph, "edge_index") or graph.edge_index.numel() == 0:
        return [], []
    edge_index = graph.edge_index.detach().cpu().numpy()
    edge_attr = getattr(graph, "edge_attr", None)
    edge_values: dict[tuple[int, int], list[float]] = {}
    for edge_pos in range(edge_index.shape[1]):
        src = int(edge_index[0, edge_pos])
        dst = int(edge_index[1, edge_pos])
        pair = (src, dst) if src <= dst else (dst, src)
        if edge_feature_idx is None or edge_attr is None or edge_attr.numel() == 0:
            value = 0.0
        else:
            value = float(edge_attr[edge_pos, edge_feature_idx].detach().cpu().item())
            if not np.isfinite(value):
                value = 0.0
        edge_values.setdefault(pair, []).append(value)
    pairs = sorted(edge_values)
    values = [float(np.mean(edge_values[pair])) for pair in pairs]
    return pairs, values


def _edge_widths(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    low = float(np.min(arr))
    high = float(np.max(arr))
    if math.isclose(low, high):
        return [2.0 for _ in values]
    scaled = (arr - low) / (high - low)
    return (1.0 + 3.0 * scaled).tolist()


def _frame_title(graph: Any, frame_idx: int, title: str | None) -> str:
    parts = [title or "TopoStateGrid graph sequence", f"frame {frame_idx}"]
    sample_id = str(getattr(graph, "sample_id", "") or "")
    timestamp = str(getattr(graph, "timestamp", "") or "")
    if sample_id:
        parts.append(sample_id)
    if timestamp:
        parts.append(timestamp)
    return " | ".join(parts)


def _feature_label(selector: str | int | None) -> str:
    if selector is None:
        return "constant"
    return str(selector)
