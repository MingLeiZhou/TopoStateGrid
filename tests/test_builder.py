from pathlib import Path
import tempfile
import unittest

from topostategrid import (
    FeatureNormalizer,
    ParsedCase,
    attach_stress_proxy_labels,
    build_graph,
    create_lono_split,
    create_random_split,
    load_graphs,
    make_temporal_windows,
    save_graphs,
)


def _synthetic_opfdata_case(sample_id="example_1", network_id="case2"):
    return ParsedCase(
        source_type="opfdata",
        network_id=network_id,
        sample_id=sample_id,
        base_mva=100.0,
        grid={
            "nodes": {
                "bus": [[1.0, 3.0, 0.94, 1.06], [1.0, 1.0, 0.94, 1.06]],
                "load": [[0.5, 0.2]],
            },
            "edges": {
                "load_link": {"senders": [0], "receivers": [1], "features": []},
                "ac_line": {
                    "senders": [0],
                    "receivers": [1],
                    "features": [[-0.5, 0.5, 0.01, 0.01, 0.02, 0.06, 1.0, 1.0, 1.0]],
                },
            },
        },
        solution={
            "nodes": {"bus": [[0.0, 1.0], [-0.1, 0.98]]},
            "edges": {"ac_line": {"senders": [0], "receivers": [1], "features": [[1.2, 0.1, -1.1, -0.05]]}},
        },
        scenario_id=sample_id,
    )


class BuilderTest(unittest.TestCase):
    def test_build_opfdata_graph_shapes_and_metadata(self):
        graph = build_graph(_synthetic_opfdata_case(), attach_proxy_label=True)

        self.assertEqual(graph.x.shape, (2, 9))
        self.assertEqual(graph.edge_index.shape, (2, 2))
        self.assertEqual(graph.edge_attr.shape, (2, 12))
        self.assertEqual(graph.network_id, "case2")
        self.assertEqual(graph.sample_id, "example_1")
        self.assertEqual(graph.y.item(), 1)
        self.assertGreater(graph.risk_score.item(), 1.0)

    def test_save_and_load_graphs(self):
        graph = build_graph(_synthetic_opfdata_case(), attach_proxy_label=True)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = save_graphs([graph], Path(tmp_dir) / "graphs.pt")
            loaded = load_graphs(path)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].network_id, "case2")
        self.assertTrue(loaded[0].edge_index.equal(graph.edge_index))

    def test_splits_normalizer_and_temporal_windows(self):
        graphs = [
            build_graph(_synthetic_opfdata_case(f"example_{idx}", "case_a" if idx < 3 else "case_b"))
            for idx in range(5)
        ]
        for graph in graphs:
            attach_stress_proxy_labels(graph)

        random_split = create_random_split(graphs, seed=1)
        self.assertEqual(sorted(random_split), ["test", "train", "val"])
        self.assertEqual(sum(len(value) for value in random_split.values()), len(graphs))

        lono_split = create_lono_split(graphs, test_network="case_b")
        self.assertEqual(lono_split["test"], [3, 4])
        self.assertEqual(lono_split["train"], [0, 1, 2])

        normalizer = FeatureNormalizer().fit([graphs[idx] for idx in lono_split["train"]])
        transformed = normalizer.transform(graphs)
        self.assertEqual(transformed[0].x.shape, graphs[0].x.shape)
        self.assertEqual(transformed[0].edge_attr.shape, graphs[0].edge_attr.shape)

        windows = make_temporal_windows(graphs, input_window=2, forecast_horizon=1, target="risk_score")
        self.assertEqual(len(windows), 3)
        self.assertEqual(len(windows[0]["graphs"]), 2)


if __name__ == "__main__":
    unittest.main()
