"""Visualization helpers for generated graph samples."""

from __future__ import annotations

import importlib
import html as html_lib
import json
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


def render_graph_html(
    graphs: Any | Sequence[Any],
    output_path: str | Path,
    node_value: str | int | None = "vm",
    edge_value: str | int | None = "loading_ratio",
    layout: str = "spring",
    seed: int = 42,
    title: str | None = None,
    width: int = 1100,
    height: int = 760,
    show_labels: bool = False,
    fixed_layout: bool = True,
) -> Path:
    """Render one graph or a graph sequence to an interactive HTML file.

    The HTML output is self-contained and supports frame selection, playback,
    pan, zoom, and hover tooltips for node and edge features.
    """

    output_path = Path(output_path)
    if output_path.suffix.lower() not in {".html", ".htm"}:
        raise ValueError("output_path must end with .html or .htm")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    graph_list = _coerce_graphs(graphs)
    if not graph_list:
        raise ValueError("graphs must contain at least one graph")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    node_idx = _feature_index(graph_list[0], node_value, "node_feature_names", "x")
    edge_idx = _feature_index(graph_list[0], edge_value, "edge_feature_names", "edge_attr")
    node_range = _feature_range(graph_list, "x", node_idx)
    edge_range = _feature_range(graph_list, "edge_attr", edge_idx)

    layouts = _html_layouts(graph_list, layout=layout, seed=seed, fixed_layout=fixed_layout)
    frames = [
        _html_frame_payload(
            graph=graph,
            frame_idx=frame_idx,
            positions=_normalize_positions(layouts[frame_idx], width=width, height=height),
            node_feature_idx=node_idx,
            edge_feature_idx=edge_idx,
        )
        for frame_idx, graph in enumerate(graph_list)
    ]
    payload = {
        "title": title or "TopoStateGrid interactive graph",
        "width": width,
        "height": height,
        "nodeValue": _feature_label(node_value),
        "edgeValue": _feature_label(edge_value),
        "nodeRange": node_range,
        "edgeRange": edge_range,
        "showLabels": bool(show_labels),
        "frames": frames,
    }
    output_path.write_text(_html_document(payload), encoding="utf-8")
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


def _html_layouts(graphs: Sequence[Any], layout: str, seed: int, fixed_layout: bool) -> list[dict[int, np.ndarray]]:
    if layout == "circular":
        first_layout = _circular_layout(_num_nodes(graphs[0]))
        if fixed_layout:
            first_node_count = _num_nodes(graphs[0])
            return [
                first_layout if _num_nodes(graph) == first_node_count else _circular_layout(_num_nodes(graph))
                for graph in graphs
            ]
        return [_circular_layout(_num_nodes(graph)) for graph in graphs]

    try:
        nx = importlib.import_module("networkx")
    except ImportError as exc:
        raise ImportError(
            "HTML graph rendering requires networkx for spring/kamada_kawai layouts. "
            'Install it with: pip install -e ".[visual]" or pass layout="circular".'
        ) from exc

    if fixed_layout:
        first_layout = _make_layout(nx, graphs[0], layout=layout, seed=seed)
        first_node_count = _num_nodes(graphs[0])
        return [
            first_layout if _num_nodes(graph) == first_node_count else _make_layout(nx, graph, layout=layout, seed=seed)
            for graph in graphs
        ]
    return [_make_layout(nx, graph, layout=layout, seed=seed) for graph in graphs]


def _circular_layout(node_count: int) -> dict[int, np.ndarray]:
    if node_count <= 0:
        return {}
    angles = np.linspace(0.0, 2.0 * np.pi, node_count, endpoint=False)
    return {idx: np.asarray([math.cos(angle), math.sin(angle)], dtype=float) for idx, angle in enumerate(angles)}


def _normalize_positions(
    positions: dict[int, np.ndarray],
    width: int,
    height: int,
    padding: float = 48.0,
) -> dict[int, tuple[float, float]]:
    if not positions:
        return {}
    arr = np.asarray(list(positions.values()), dtype=float)
    min_xy = np.nanmin(arr, axis=0)
    max_xy = np.nanmax(arr, axis=0)
    span = np.maximum(max_xy - min_xy, 1e-9)
    scaled: dict[int, tuple[float, float]] = {}
    for node_id, xy in positions.items():
        norm = (np.asarray(xy, dtype=float) - min_xy) / span
        x = padding + norm[0] * max(width - 2.0 * padding, 1.0)
        y = padding + (1.0 - norm[1]) * max(height - 2.0 * padding, 1.0)
        scaled[int(node_id)] = (float(x), float(y))
    return scaled


def _html_frame_payload(
    graph: Any,
    frame_idx: int,
    positions: dict[int, tuple[float, float]],
    node_feature_idx: int | None,
    edge_feature_idx: int | None,
) -> dict[str, Any]:
    node_names = list(getattr(graph, "node_feature_names", []) or [])
    edge_names = list(getattr(graph, "edge_feature_names", []) or [])
    node_values = _node_values(graph, node_feature_idx)
    nodes = []
    node_count = _num_nodes(graph)
    for node_id in range(node_count):
        x, y = positions.get(node_id, (0.0, 0.0))
        nodes.append(
            {
                "id": node_id,
                "x": _json_float(x),
                "y": _json_float(y),
                "value": _json_float(node_values[node_id] if node_id < len(node_values) else 0.0),
                "features": _feature_payload(getattr(graph, "x", None), node_id, node_names),
            }
        )
    return {
        "index": frame_idx,
        "network_id": str(getattr(graph, "network_id", "") or ""),
        "sample_id": str(getattr(graph, "sample_id", "") or ""),
        "timestamp": str(getattr(graph, "timestamp", "") or ""),
        "scenario_id": str(getattr(graph, "scenario_id", "") or ""),
        "nodes": nodes,
        "edges": _html_edge_payload(graph, edge_feature_idx, edge_names),
    }


def _feature_payload(tensor: Any, row_idx: int, names: Sequence[str]) -> dict[str, float]:
    if tensor is None or getattr(tensor, "ndim", 0) != 2 or row_idx >= int(tensor.shape[0]):
        return {}
    row = tensor[row_idx].detach().cpu().numpy().astype(float)
    features: dict[str, float] = {}
    for col_idx, value in enumerate(row):
        name = names[col_idx] if col_idx < len(names) else f"feature_{col_idx}"
        features[str(name)] = _json_float(value)
    return features


def _html_edge_payload(graph: Any, edge_feature_idx: int | None, edge_names: Sequence[str]) -> list[dict[str, Any]]:
    if not hasattr(graph, "edge_index") or graph.edge_index.numel() == 0:
        return []
    edge_index = graph.edge_index.detach().cpu().numpy()
    edge_attr = getattr(graph, "edge_attr", None)
    grouped: dict[tuple[int, int], dict[str, Any]] = {}
    for edge_pos in range(edge_index.shape[1]):
        src = int(edge_index[0, edge_pos])
        dst = int(edge_index[1, edge_pos])
        pair = (src, dst) if src <= dst else (dst, src)
        record = grouped.setdefault(pair, {"source": pair[0], "target": pair[1], "values": [], "directions": []})
        if edge_feature_idx is None or edge_attr is None or edge_attr.numel() == 0:
            value = 0.0
        else:
            value = _json_float(edge_attr[edge_pos, edge_feature_idx].detach().cpu().item())
        record["values"].append(value)
        features = _feature_payload(edge_attr, edge_pos, edge_names)
        features["direction"] = f"{src}->{dst}"
        record["directions"].append({"from": src, "to": dst, "features": features})

    edges = []
    for pair in sorted(grouped):
        record = grouped[pair]
        values = [float(value) for value in record["values"]]
        edges.append(
            {
                "source": int(record["source"]),
                "target": int(record["target"]),
                "value": _json_float(float(np.mean(values)) if values else 0.0),
                "directions": record["directions"],
            }
        )
    return edges


def _json_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(numeric):
        return 0.0
    return round(numeric, 6)


def _html_document(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html_lib.escape(str(payload["title"]))}</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f8fafc;
      color: #172033;
    }}
    body {{ margin: 0; }}
    .app {{
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }}
    header {{
      padding: 16px 20px 12px;
      border-bottom: 1px solid #d8dee9;
      background: #ffffff;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 20px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 14px;
      align-items: center;
      font-size: 13px;
    }}
    button {{
      border: 1px solid #b8c2d4;
      background: #ffffff;
      color: #172033;
      border-radius: 6px;
      padding: 6px 10px;
      cursor: pointer;
    }}
    button:hover {{ background: #eef2f7; }}
    input[type="range"] {{ width: min(440px, 70vw); }}
    .meta {{
      color: #506070;
      min-width: 220px;
    }}
    .stage {{
      position: relative;
      overflow: hidden;
      background: #f8fafc;
    }}
    svg {{
      width: 100%;
      height: calc(100vh - 88px);
      display: block;
      cursor: grab;
      user-select: none;
      touch-action: none;
    }}
    svg:active {{ cursor: grabbing; }}
    .edge {{
      stroke-linecap: round;
      opacity: 0.72;
      cursor: pointer;
    }}
    .node {{
      stroke: #172033;
      stroke-width: 0.8;
      cursor: pointer;
    }}
    .node-label {{
      font-size: 9px;
      fill: #172033;
      pointer-events: none;
      text-anchor: middle;
      dominant-baseline: central;
    }}
    .tooltip {{
      position: absolute;
      max-width: 420px;
      max-height: min(560px, 78vh);
      overflow: auto;
      padding: 10px 12px;
      border: 1px solid #c8d1df;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.96);
      box-shadow: 0 14px 40px rgba(15, 23, 42, 0.16);
      font-size: 12px;
      line-height: 1.4;
      display: none;
      pointer-events: none;
      white-space: normal;
    }}
    .tooltip strong {{ display: block; margin-bottom: 6px; font-size: 13px; }}
    .tooltip table {{ border-collapse: collapse; width: 100%; }}
    .tooltip td {{ padding: 2px 8px 2px 0; border-bottom: 1px solid #eef2f7; vertical-align: top; }}
    .tooltip td:first-child {{ color: #506070; white-space: nowrap; }}
    .legend {{
      position: absolute;
      right: 16px;
      bottom: 16px;
      padding: 9px 10px;
      border: 1px solid #c8d1df;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.9);
      font-size: 12px;
      color: #334155;
    }}
  </style>
</head>
<body>
  <div class="app">
    <header>
      <h1 id="title"></h1>
      <div class="controls">
        <button id="playBtn" type="button">Play</button>
        <button id="resetBtn" type="button">Reset view</button>
        <label>Frame <input id="frameSlider" type="range" min="0" value="0"></label>
        <span id="frameLabel" class="meta"></span>
      </div>
    </header>
    <main class="stage">
      <svg id="graph"></svg>
      <div id="tooltip" class="tooltip"></div>
      <div class="legend">
        <div>Node color: <strong id="nodeLegend"></strong></div>
        <div>Edge color/width: <strong id="edgeLegend"></strong></div>
        <div>Wheel: zoom | Drag: pan | Hover: state</div>
      </div>
    </main>
  </div>
  <script>
    const payload = {payload_json};
    const svg = document.getElementById("graph");
    const tooltip = document.getElementById("tooltip");
    const slider = document.getElementById("frameSlider");
    const playBtn = document.getElementById("playBtn");
    const resetBtn = document.getElementById("resetBtn");
    const frameLabel = document.getElementById("frameLabel");
    document.getElementById("title").textContent = payload.title;
    document.getElementById("nodeLegend").textContent = payload.nodeValue;
    document.getElementById("edgeLegend").textContent = payload.edgeValue;

    const svgNS = "http://www.w3.org/2000/svg";
    const baseViewBox = {{ x: 0, y: 0, width: payload.width, height: payload.height }};
    let viewBox = {{ ...baseViewBox }};
    let frameIndex = 0;
    let timer = null;
    let dragging = false;
    let dragStart = null;
    let viewStart = null;

    svg.setAttribute("viewBox", viewBoxString());
    slider.max = Math.max(payload.frames.length - 1, 0);

    function viewBoxString() {{
      return `${{viewBox.x}} ${{viewBox.y}} ${{viewBox.width}} ${{viewBox.height}}`;
    }}

    function setViewBox() {{
      svg.setAttribute("viewBox", viewBoxString());
    }}

    function clearSvg() {{
      while (svg.firstChild) svg.removeChild(svg.firstChild);
    }}

    function norm(value, range) {{
      const low = Number(range[0]);
      const high = Number(range[1]);
      if (!Number.isFinite(value) || Math.abs(high - low) < 1e-12) return 0.5;
      return Math.max(0, Math.min(1, (value - low) / (high - low)));
    }}

    function mix(a, b, t) {{
      return Math.round(a + (b - a) * t);
    }}

    function colorScale(value, range, stops) {{
      const t = norm(value, range);
      const scaled = t * (stops.length - 1);
      const idx = Math.min(Math.floor(scaled), stops.length - 2);
      const local = scaled - idx;
      const a = stops[idx];
      const b = stops[idx + 1];
      return `rgb(${{mix(a[0], b[0], local)}}, ${{mix(a[1], b[1], local)}}, ${{mix(a[2], b[2], local)}})`;
    }}

    function nodeColor(value) {{
      return colorScale(value, payload.nodeRange, [[68, 1, 84], [33, 145, 140], [253, 231, 37]]);
    }}

    function edgeColor(value) {{
      return colorScale(value, payload.edgeRange, [[13, 8, 135], [204, 71, 120], [240, 249, 33]]);
    }}

    function edgeWidth(value) {{
      return 1.0 + 4.0 * norm(value, payload.edgeRange);
    }}

    function formatValue(value) {{
      if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(6).replace(/0+$/, "").replace(/\\.$/, "");
      return String(value);
    }}

    function featureTable(features) {{
      const rows = Object.entries(features || {{}})
        .map(([key, value]) => `<tr><td>${{escapeHtml(key)}}</td><td>${{escapeHtml(formatValue(value))}}</td></tr>`)
        .join("");
      return `<table>${{rows}}</table>`;
    }}

    function escapeHtml(value) {{
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }}

    function nodeTooltip(node, frame) {{
      return `<strong>Bus node ${{node.id}}</strong>
        <div>network: ${{escapeHtml(frame.network_id || "")}}</div>
        <div>sample: ${{escapeHtml(frame.sample_id || "")}}</div>
        <div>timestamp: ${{escapeHtml(frame.timestamp || "")}}</div>
        ${{featureTable(node.features)}}`;
    }}

    function edgeTooltip(edge, frame) {{
      const directions = edge.directions.map(direction => {{
        return `<div style="margin-top: 8px;"><strong>Line ${{direction.from}} → ${{direction.to}}</strong>${{featureTable(direction.features)}}</div>`;
      }}).join("");
      return `<strong>Physical edge ${{edge.source}} - ${{edge.target}}</strong>
        <div>network: ${{escapeHtml(frame.network_id || "")}}</div>
        <div>sample: ${{escapeHtml(frame.sample_id || "")}}</div>
        <div>timestamp: ${{escapeHtml(frame.timestamp || "")}}</div>
        <div>${{escapeHtml(payload.edgeValue)}}: ${{escapeHtml(formatValue(edge.value))}}</div>
        ${{directions}}`;
    }}

    function showTooltip(evt, html) {{
      tooltip.innerHTML = html;
      tooltip.style.display = "block";
      moveTooltip(evt);
    }}

    function moveTooltip(evt) {{
      const rect = svg.getBoundingClientRect();
      const left = Math.min(evt.clientX - rect.left + 16, rect.width - tooltip.offsetWidth - 12);
      const top = Math.min(evt.clientY - rect.top + 16, rect.height - tooltip.offsetHeight - 12);
      tooltip.style.left = `${{Math.max(8, left)}}px`;
      tooltip.style.top = `${{Math.max(8, top)}}px`;
    }}

    function hideTooltip() {{
      tooltip.style.display = "none";
    }}

    function pointFor(nodeMap, id) {{
      return nodeMap.get(id) || {{ x: 0, y: 0 }};
    }}

    function updateFrame(nextIndex) {{
      frameIndex = Number(nextIndex);
      const frame = payload.frames[frameIndex];
      slider.value = String(frameIndex);
      frameLabel.textContent = `${{frameIndex + 1}}/${{payload.frames.length}}  ${{frame.sample_id || ""}}  ${{frame.timestamp || ""}}`;
      clearSvg();

      const nodeMap = new Map(frame.nodes.map(node => [node.id, node]));
      for (const edge of frame.edges) {{
        const source = pointFor(nodeMap, edge.source);
        const target = pointFor(nodeMap, edge.target);
        const line = document.createElementNS(svgNS, "line");
        line.setAttribute("class", "edge");
        line.setAttribute("x1", source.x);
        line.setAttribute("y1", source.y);
        line.setAttribute("x2", target.x);
        line.setAttribute("y2", target.y);
        line.setAttribute("stroke", edgeColor(edge.value));
        line.setAttribute("stroke-width", edgeWidth(edge.value));
        line.addEventListener("mousemove", evt => showTooltip(evt, edgeTooltip(edge, frame)));
        line.addEventListener("mouseleave", hideTooltip);
        svg.appendChild(line);
      }}

      const radius = Math.max(2.8, Math.min(8, 80 / Math.sqrt(Math.max(frame.nodes.length, 1))));
      for (const node of frame.nodes) {{
        const circle = document.createElementNS(svgNS, "circle");
        circle.setAttribute("class", "node");
        circle.setAttribute("cx", node.x);
        circle.setAttribute("cy", node.y);
        circle.setAttribute("r", radius);
        circle.setAttribute("fill", nodeColor(node.value));
        circle.addEventListener("mousemove", evt => showTooltip(evt, nodeTooltip(node, frame)));
        circle.addEventListener("mouseleave", hideTooltip);
        svg.appendChild(circle);
        if (payload.showLabels && frame.nodes.length <= 120) {{
          const label = document.createElementNS(svgNS, "text");
          label.setAttribute("class", "node-label");
          label.setAttribute("x", node.x);
          label.setAttribute("y", node.y);
          label.textContent = node.id;
          svg.appendChild(label);
        }}
      }}
    }}

    slider.addEventListener("input", () => updateFrame(slider.value));
    playBtn.addEventListener("click", () => {{
      if (timer) {{
        clearInterval(timer);
        timer = null;
        playBtn.textContent = "Play";
        return;
      }}
      playBtn.textContent = "Pause";
      timer = setInterval(() => updateFrame((frameIndex + 1) % payload.frames.length), 900);
    }});
    resetBtn.addEventListener("click", () => {{
      viewBox = {{ ...baseViewBox }};
      setViewBox();
    }});

    svg.addEventListener("wheel", evt => {{
      evt.preventDefault();
      const scale = evt.deltaY < 0 ? 0.9 : 1.1;
      const rect = svg.getBoundingClientRect();
      const mouseX = viewBox.x + (evt.clientX - rect.left) / rect.width * viewBox.width;
      const mouseY = viewBox.y + (evt.clientY - rect.top) / rect.height * viewBox.height;
      viewBox.width *= scale;
      viewBox.height *= scale;
      viewBox.x = mouseX - (evt.clientX - rect.left) / rect.width * viewBox.width;
      viewBox.y = mouseY - (evt.clientY - rect.top) / rect.height * viewBox.height;
      setViewBox();
    }}, {{ passive: false }});

    svg.addEventListener("pointerdown", evt => {{
      dragging = true;
      dragStart = {{ x: evt.clientX, y: evt.clientY }};
      viewStart = {{ ...viewBox }};
      svg.setPointerCapture(evt.pointerId);
    }});
    svg.addEventListener("pointermove", evt => {{
      if (!dragging) return;
      const rect = svg.getBoundingClientRect();
      const dx = (evt.clientX - dragStart.x) / rect.width * viewStart.width;
      const dy = (evt.clientY - dragStart.y) / rect.height * viewStart.height;
      viewBox.x = viewStart.x - dx;
      viewBox.y = viewStart.y - dy;
      setViewBox();
    }});
    svg.addEventListener("pointerup", evt => {{
      dragging = false;
      svg.releasePointerCapture(evt.pointerId);
    }});
    svg.addEventListener("pointerleave", () => {{
      dragging = false;
      hideTooltip();
    }});

    updateFrame(0);
  </script>
</body>
</html>
"""
