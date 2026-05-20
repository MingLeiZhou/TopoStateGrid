"""Render a small TopoStateGrid graph sequence to GIF."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from topostategrid import build_graph_from_tables, render_graph_sequence  # noqa: E402


def main() -> None:
    bus_df = pd.DataFrame(
        {
            "bus_id": [1, 2, 3],
            "bus_type": [3, 1, 1],
            "pd": [0.0, 1.0, 0.8],
            "qd": [0.0, 0.2, 0.1],
            "vm": [1.0, 0.99, 0.98],
        }
    )
    branch_df = pd.DataFrame(
        {
            "branch_id": ["l12", "l23"],
            "from_bus": [1, 2],
            "to_bus": [2, 3],
            "r": [0.01, 0.02],
            "x": [0.05, 0.06],
            "rate_a": [1.0, 1.0],
        }
    )

    graphs = []
    for idx, loading in enumerate([0.15, 0.35, 0.55, 0.75, 0.95]):
        bus_state_df = pd.DataFrame(
            {
                "bus_id": [1, 2, 3],
                "pd": [0.0, 1.0 + 0.15 * idx, 0.8 + 0.12 * idx],
                "qd": [0.0, 0.2 + 0.03 * idx, 0.1 + 0.02 * idx],
                "vm": [1.0, 0.995 - 0.018 * idx, 0.985 - 0.026 * idx],
                "va": [0.0, -0.02 * idx, -0.04 * idx],
            }
        )
        branch_state_df = pd.DataFrame(
            {
                "branch_id": ["l12", "l23"],
                "pf": [0.2 + 0.08 * idx, 0.16 + 0.06 * idx],
                "qf": [0.05 + 0.02 * idx, 0.04 + 0.02 * idx],
                "pt": [-0.19 - 0.08 * idx, -0.15 - 0.06 * idx],
                "qt": [-0.04 - 0.02 * idx, -0.03 - 0.02 * idx],
                "loading_ratio": [loading, loading + 0.04],
            }
        )
        graphs.append(
            build_graph_from_tables(
                bus_df,
                branch_df,
                bus_state_df=bus_state_df,
                branch_state_df=branch_state_df,
                network_id="toy_animation",
                sample_id=f"state_{idx}",
                timestamp=f"2026-01-01T00:0{idx}:00",
            )
        )

    output_path = Path("outputs") / "topostategrid_sequence.gif"
    try:
        rendered = render_graph_sequence(
            graphs,
            output_path,
            node_value="vm",
            edge_value="loading_ratio",
            fps=2,
            title="TopoStateGrid operating-state sequence",
            show_labels=True,
        )
    except ImportError as exc:
        print(str(exc))
        return

    print(f"frames: {len(graphs)}")
    print(f"rendered: {rendered}")


if __name__ == "__main__":
    main()
