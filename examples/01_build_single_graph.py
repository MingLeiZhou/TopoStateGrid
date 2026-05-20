"""Build one PyTorch Geometric graph from a local OPFData sample."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from topostategrid import (  # noqa: E402
    build_graph_from_matpower,
    build_graph_from_opfdata_json,
    discover_opfdata_examples,
    export_dataset,
    load_graphs,
)


def main() -> None:
    opf_paths = discover_opfdata_examples(ROOT / "data" / "opfdata", network_id="pglib_opf_case14_ieee", limit=1)
    if opf_paths:
        graph = build_graph_from_opfdata_json(opf_paths[0], attach_proxy_label=True)
        source = opf_paths[0]
    else:
        source = ROOT / "data" / "pglib" / "pglib_opf_case118_ieee.m"
        graph = build_graph_from_matpower(source)

    paths = export_dataset([graph], ROOT / "outputs")
    loaded = load_graphs(paths["graphs"])

    metadata = {
        "network_id": graph.network_id,
        "sample_id": graph.sample_id,
        "timestamp": graph.timestamp,
        "scenario_id": graph.scenario_id,
        "contingency_id": graph.contingency_id,
        "source_type": graph.source_type,
    }

    print(f"source: {source}")
    print(f"num_nodes: {graph.num_nodes}")
    print(f"num_edges: {graph.edge_index.shape[1]}")
    print(f"node_feature_shape: {tuple(graph.x.shape)}")
    print(f"edge_feature_shape: {tuple(graph.edge_attr.shape)}")
    print(f"metadata: {metadata}")
    print(f"saved_graphs: {paths['graphs']}")
    print(f"loaded_graph_count: {len(loaded)}")


if __name__ == "__main__":
    main()
