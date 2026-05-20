"""Build a graph from a small pandapower network when pandapower is installed."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from topostategrid import build_graph_from_pandapower, save_graphs, write_metadata_csv  # noqa: E402


def main() -> None:
    try:
        import pandapower as pp
    except ImportError:
        print('pandapower is optional; install it with: pip install -e ".[pandapower]"')
        return

    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=110, max_vm_pu=1.1, min_vm_pu=0.9)
    b2 = pp.create_bus(net, vn_kv=110, max_vm_pu=1.1, min_vm_pu=0.9)
    b3 = pp.create_bus(net, vn_kv=110, max_vm_pu=1.1, min_vm_pu=0.9)
    pp.create_ext_grid(net, b1)
    pp.create_load(net, b2, p_mw=10.0, q_mvar=3.0)
    pp.create_line_from_parameters(net, b1, b2, 1.0, 0.1, 0.2, 0.0, 0.4)
    pp.create_line_from_parameters(net, b2, b3, 1.0, 0.1, 0.2, 0.0, 0.4)
    try:
        pp.runpp(net)
    except Exception as exc:
        print(f"pandapower power flow did not converge; building graph from static tables and available results: {exc}")

    graph = build_graph_from_pandapower(net, network_id="pandapower_3bus", sample_id="sample_0")
    output_dir = ROOT / "outputs"
    save_graphs([graph], output_dir / "graphs_pandapower.pt")
    write_metadata_csv([graph], output_dir / "metadata_pandapower.csv")

    print(f"num_nodes: {graph.num_nodes}")
    print(f"num_edges: {graph.edge_index.shape[1]}")
    print(f"node_feature_shape: {tuple(graph.x.shape)}")
    print(f"edge_feature_shape: {tuple(graph.edge_attr.shape)}")
    print(f"source_format: {graph.source_format}")


if __name__ == "__main__":
    main()
