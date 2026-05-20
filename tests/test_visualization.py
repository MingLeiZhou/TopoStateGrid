import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from topostategrid import build_graph_from_tables, render_graph_html, render_graph_sequence


VISUAL_DEPS_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("matplotlib", "networkx", "PIL")
)


def _toy_sequence():
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
    for idx, loading in enumerate([0.2, 0.6, 0.9]):
        bus_state_df = pd.DataFrame(
            {
                "bus_id": [1, 2, 3],
                "vm": [1.0, 0.99 - 0.02 * idx, 0.98 - 0.03 * idx],
                "pd": [0.0, 1.0 + idx * 0.1, 0.8 + idx * 0.1],
            }
        )
        branch_state_df = pd.DataFrame(
            {
                "branch_id": ["l12", "l23"],
                "loading_ratio": [loading, loading + 0.05],
            }
        )
        graphs.append(
            build_graph_from_tables(
                bus_df,
                branch_df,
                bus_state_df=bus_state_df,
                branch_state_df=branch_state_df,
                network_id="viz_toy",
                sample_id=f"sample_{idx}",
                timestamp=f"2026-01-01T00:0{idx}:00",
            )
        )
    return graphs


class VisualizationTest(unittest.TestCase):
    def test_invalid_animation_suffix_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "output_path"):
                render_graph_sequence(_toy_sequence(), Path(tmpdir) / "sequence.txt")

    def test_render_graph_html(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = render_graph_html(
                _toy_sequence(),
                Path(tmpdir) / "sequence.html",
                layout="circular",
                title="Interactive QA",
            )
            self.assertTrue(path.exists())
            html = path.read_text(encoding="utf-8")
            self.assertIn("Interactive QA", html)
            self.assertIn("loading_ratio", html)
            self.assertIn("sample_0", html)
            self.assertIn("Wheel: zoom", html)

    @unittest.skipUnless(VISUAL_DEPS_AVAILABLE, "visualization dependencies are not installed")
    def test_render_graph_sequence_to_gif(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = render_graph_sequence(
                _toy_sequence(),
                Path(tmpdir) / "sequence.gif",
                fps=1,
                title="QA visualization",
            )
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
