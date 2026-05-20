"""Render a 20-second GIF from a large pandapower case300 graph sequence."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from topostategrid import build_graph_from_pandapower, render_graph_sequence, save_graphs, write_metadata_csv  # noqa: E402


FRAME_COUNT = 20
FPS = 1
OUTPUT_DIR = Path("outputs")
GRAPH_PATH = OUTPUT_DIR / "graphs_pandapower_case300_sequence.pt"
METADATA_PATH = OUTPUT_DIR / "metadata_pandapower_case300_sequence.csv"
GIF_PATH = OUTPUT_DIR / "topostategrid_case300_20s.gif"


def main() -> None:
    try:
        import pandapower as pp
        import pandapower.networks as pn
    except ImportError:
        print('pandapower is optional; install it with: pip install -e ".[pandapower]"')
        return

    net = pn.case300()
    if len(net.bus) <= 200:
        raise RuntimeError(f"Expected more than 200 buses, got {len(net.bus)}")

    base_load_p = net.load["p_mw"].copy() if not net.load.empty and "p_mw" in net.load else None
    base_load_q = net.load["q_mvar"].copy() if not net.load.empty and "q_mvar" in net.load else None

    graphs = []
    for frame_idx in range(FRAME_COUNT):
        scale = 1.0 + 0.03 * np.sin(2.0 * np.pi * frame_idx / FRAME_COUNT)
        if base_load_p is not None:
            net.load["p_mw"] = base_load_p * scale
        if base_load_q is not None:
            net.load["q_mvar"] = base_load_q * scale

        _run_power_flow(pp, net)

        graphs.append(
            build_graph_from_pandapower(
                net,
                network_id="pandapower_case300",
                sample_id=f"case300_state_{frame_idx:02d}",
                timestamp=f"frame_{frame_idx:02d}",
                scenario_id=f"load_scale_{scale:.4f}",
            )
        )

    first_graph = graphs[0]
    if first_graph.num_nodes <= 200:
        raise RuntimeError(f"Converted graph should have more than 200 nodes, got {first_graph.num_nodes}")
    if first_graph.edge_index.shape[1] == 0:
        raise RuntimeError("Converted graph has no edges")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_graphs(graphs, GRAPH_PATH)
    write_metadata_csv(graphs, METADATA_PATH)
    rendered = render_graph_sequence(
        graphs,
        GIF_PATH,
        node_value="vm",
        edge_value="loading_ratio",
        fps=FPS,
        layout="spring",
        seed=7,
        figsize=(8.5, 7.0),
        dpi=120,
        title="TopoStateGrid pandapower case300",
        show_labels=False,
        fixed_layout=True,
    )

    print(f"network_id: {first_graph.network_id}")
    print(f"frames: {len(graphs)}")
    print(f"duration_seconds: {len(graphs) / FPS:.0f}")
    print(f"num_nodes: {first_graph.num_nodes}")
    print(f"num_directed_edges: {first_graph.edge_index.shape[1]}")
    print(f"node_feature_shape: {tuple(first_graph.x.shape)}")
    print(f"edge_feature_shape: {tuple(first_graph.edge_attr.shape)}")
    print(f"saved_graphs: {GRAPH_PATH}")
    print(f"metadata: {METADATA_PATH}")
    print(f"rendered_gif: {rendered}")


def _run_power_flow(pp, net) -> None:
    attempts = [
        {
            "algorithm": "fdbx",
            "init": "flat",
            "calculate_voltage_angles": False,
            "max_iteration": 100,
            "numba": False,
        },
        {
            "algorithm": "iwamoto_nr",
            "init": "auto",
            "calculate_voltage_angles": True,
            "max_iteration": 100,
            "numba": False,
        },
        {
            "algorithm": "iwamoto_nr",
            "init": "flat",
            "calculate_voltage_angles": False,
            "max_iteration": 100,
            "numba": False,
        },
    ]
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                pp.runpp(net, **kwargs)
            return
        except Exception as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


if __name__ == "__main__":
    main()
