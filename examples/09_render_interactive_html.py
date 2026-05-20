"""Render an interactive HTML graph viewer."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from topostategrid import build_graph_from_tables, load_graphs, render_graph_html  # noqa: E402


CASE300_GRAPHS = Path("outputs") / "graphs_pandapower_case300_sequence.pt"
CASE300_HTML = Path("outputs") / "topostategrid_case300_interactive.html"
TOY_HTML = Path("outputs") / "topostategrid_interactive.html"


def main() -> None:
    if CASE300_GRAPHS.exists():
        graphs = load_graphs(CASE300_GRAPHS)
        output_path = CASE300_HTML
        title = "TopoStateGrid case300 interactive viewer"
    else:
        graphs = _toy_sequence()
        output_path = TOY_HTML
        title = "TopoStateGrid interactive viewer"

    rendered = render_graph_html(
        graphs,
        output_path,
        node_value="vm",
        edge_value="loading_ratio",
        layout="spring",
        title=title,
        show_labels=False,
    )

    first_graph = graphs[0]
    print(f"frames: {len(graphs)}")
    print(f"num_nodes: {first_graph.num_nodes}")
    print(f"num_directed_edges: {first_graph.edge_index.shape[1]}")
    print(f"rendered_html: {rendered}")


def _toy_sequence():
    bus_df = pd.DataFrame(
        {
            "bus_id": [1, 2, 3, 4],
            "bus_type": [3, 1, 1, 1],
            "pd": [0.0, 1.0, 0.8, 0.6],
            "qd": [0.0, 0.2, 0.1, 0.08],
            "vm": [1.0, 0.99, 0.98, 0.97],
        }
    )
    branch_df = pd.DataFrame(
        {
            "branch_id": ["l12", "l23", "l34", "l14"],
            "from_bus": [1, 2, 3, 1],
            "to_bus": [2, 3, 4, 4],
            "r": [0.01, 0.02, 0.02, 0.03],
            "x": [0.05, 0.06, 0.04, 0.08],
            "rate_a": [1.0, 1.0, 0.8, 0.9],
        }
    )
    graphs = []
    for idx, loading in enumerate([0.15, 0.35, 0.55, 0.75, 0.95]):
        bus_state_df = pd.DataFrame(
            {
                "bus_id": [1, 2, 3, 4],
                "pd": [0.0, 1.0 + idx * 0.12, 0.8 + idx * 0.1, 0.6 + idx * 0.08],
                "qd": [0.0, 0.2 + idx * 0.03, 0.1 + idx * 0.02, 0.08 + idx * 0.02],
                "vm": [1.0, 0.995 - idx * 0.015, 0.985 - idx * 0.02, 0.975 - idx * 0.025],
                "va": [0.0, -0.01 * idx, -0.02 * idx, -0.03 * idx],
            }
        )
        branch_state_df = pd.DataFrame(
            {
                "branch_id": ["l12", "l23", "l34", "l14"],
                "pf": [0.15 + idx * 0.05, 0.12 + idx * 0.04, 0.08 + idx * 0.03, 0.07 + idx * 0.02],
                "qf": [0.03 + idx * 0.02, 0.02 + idx * 0.015, 0.02 + idx * 0.01, 0.01 + idx * 0.01],
                "pt": [-0.14 - idx * 0.05, -0.11 - idx * 0.04, -0.07 - idx * 0.03, -0.06 - idx * 0.02],
                "qt": [-0.03 - idx * 0.02, -0.02 - idx * 0.015, -0.02 - idx * 0.01, -0.01 - idx * 0.01],
                "loading_ratio": [loading, loading + 0.08, loading + 0.04, loading + 0.12],
            }
        )
        graphs.append(
            build_graph_from_tables(
                bus_df,
                branch_df,
                bus_state_df=bus_state_df,
                branch_state_df=branch_state_df,
                network_id="toy_interactive",
                sample_id=f"state_{idx}",
                timestamp=f"frame_{idx}",
            )
        )
    return graphs


if __name__ == "__main__":
    main()
