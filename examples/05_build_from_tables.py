"""Build a graph from in-memory pandas tables."""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from topostategrid import build_graph_from_tables, save_graphs, write_metadata_csv  # noqa: E402


def main() -> None:
    bus_df = pd.DataFrame(
        {
            "bus_id": [1, 2, 3],
            "bus_status": [1, 1, 1],
            "bus_type": [3, 1, 1],
            "pd": [0.0, 1.5, 0.8],
            "qd": [0.0, 0.4, 0.2],
            "vm": [1.0, 0.99, 0.98],
            "va": [0.0, -1.0, -2.0],
            "vmax": [1.1, 1.1, 1.1],
            "vmin": [0.9, 0.9, 0.9],
        }
    )
    branch_df = pd.DataFrame(
        {
            "branch_id": ["line_12", "line_23"],
            "from_bus": [1, 2],
            "to_bus": [2, 3],
            "r": [0.01, 0.02],
            "x": [0.05, 0.06],
            "rate_a": [1.0, 1.0],
            "loading_ratio": [0.3, 0.4],
        }
    )
    graph = build_graph_from_tables(bus_df, branch_df, network_id="toy_3bus", sample_id="sample_0")

    output_dir = ROOT / "outputs"
    save_graphs([graph], output_dir / "graphs_tables.pt")
    write_metadata_csv([graph], output_dir / "metadata_tables.csv")

    print(f"num_nodes: {graph.num_nodes}")
    print(f"num_edges: {graph.edge_index.shape[1]}")
    print(f"node_feature_shape: {tuple(graph.x.shape)}")
    print(f"edge_feature_shape: {tuple(graph.edge_attr.shape)}")
    print(f"source_format: {graph.source_format}")


if __name__ == "__main__":
    main()
