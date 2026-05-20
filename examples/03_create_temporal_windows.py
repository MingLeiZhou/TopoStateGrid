"""Create simple graph windows over ordered OPFData scenarios."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from topostategrid import build_graphs_from_opfdata, make_temporal_windows  # noqa: E402
import torch  # noqa: E402


def main() -> None:
    graphs = build_graphs_from_opfdata(
        ROOT / "data" / "opfdata",
        network_id="pglib_opf_case14_ieee",
        limit=10,
        attach_proxy_label=True,
    )
    windows = make_temporal_windows(graphs, input_window=3, forecast_horizon=1, target="risk_score")

    output_path = ROOT / "outputs" / "temporal_windows.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(windows, output_path)

    print(f"graphs: {len(graphs)}")
    print(f"windows: {len(windows)}")
    if windows:
        first = windows[0]
        print(f"first_window_sample_ids: {first['sample_ids']}")
        print(f"first_target_sample_id: {first['target_sample_id']}")
        print(f"first_target: {first['target'].reshape(-1).tolist()}")
    print(f"saved_windows: {output_path}")


if __name__ == "__main__":
    main()
