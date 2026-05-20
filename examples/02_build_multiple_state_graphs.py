"""Build multiple OPFData scenario graphs for one benchmark network."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from topostategrid import build_graphs_from_opfdata, create_random_split, save_graphs, save_split_json, write_metadata_csv  # noqa: E402


def main() -> None:
    graphs = build_graphs_from_opfdata(
        ROOT / "data" / "opfdata",
        network_id="pglib_opf_case14_ieee",
        limit=16,
        attach_proxy_label=True,
    )
    split = create_random_split(graphs, seed=7)

    output_dir = ROOT / "outputs"
    save_graphs(graphs, output_dir / "graphs_multi.pt")
    write_metadata_csv(graphs, output_dir / "metadata_multi.csv")
    save_split_json(split, output_dir / "split_random.json")

    print(f"graphs: {len(graphs)}")
    print(f"network_id: {graphs[0].network_id if graphs else 'none'}")
    print(f"node_feature_shape: {tuple(graphs[0].x.shape) if graphs else 'none'}")
    print(f"edge_feature_shape: {tuple(graphs[0].edge_attr.shape) if graphs else 'none'}")
    print(f"split: {split}")


if __name__ == "__main__":
    main()
