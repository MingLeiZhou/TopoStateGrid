"""Create random, time-ordered, and Leave-One-Network-Out splits."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from topostategrid import (  # noqa: E402
    build_graphs_from_opfdata,
    create_lono_split,
    create_random_split,
    create_time_based_split,
    save_split_json,
)


def main() -> None:
    graphs = []
    graphs.extend(build_graphs_from_opfdata(ROOT / "data" / "opfdata", network_id="pglib_opf_case14_ieee", limit=5))
    graphs.extend(build_graphs_from_opfdata(ROOT / "data" / "opfdata", network_id="pglib_opf_case30_ieee", limit=5))

    output_dir = ROOT / "outputs"
    random_split = create_random_split(graphs, seed=3)
    time_split = create_time_based_split(graphs)
    lono_split = create_lono_split(graphs, test_network="pglib_opf_case30_ieee")

    save_split_json(random_split, output_dir / "split_random.json")
    save_split_json(time_split, output_dir / "split_time.json")
    save_split_json(lono_split, output_dir / "split_lono.json")

    print(f"graphs: {len(graphs)}")
    print(f"random_split: {random_split}")
    print(f"time_split: {time_split}")
    print(f"lono_split: {lono_split}")


if __name__ == "__main__":
    main()
