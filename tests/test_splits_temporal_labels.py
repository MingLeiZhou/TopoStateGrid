import unittest

from topostategrid import (
    attach_stress_proxy_labels,
    create_lono_split,
    create_random_split,
    create_time_based_split,
    make_temporal_windows,
)
import torch
from torch_geometric.data import Data


class SplitTemporalLabelTest(unittest.TestCase):
    def test_lono_normal_two_network_split(self):
        dataset = [Data(network_id="case14"), Data(network_id="case14"), Data(network_id="case30")]

        split = create_lono_split(dataset, test_network="case30")

        self.assertEqual(split["train"], [0, 1])
        self.assertEqual(split["val"], [])
        self.assertEqual(split["test"], [2])

    def test_lono_unknown_test_network_raises(self):
        dataset = [Data(network_id="case14")]

        with self.assertRaisesRegex(ValueError, "not present"):
            create_lono_split(dataset, test_network="missing")

    def test_lono_single_network_train_empty_raises(self):
        dataset = [Data(network_id="case14"), Data(network_id="case14")]

        with self.assertRaisesRegex(ValueError, "empty train"):
            create_lono_split(dataset, test_network="case14")

    def test_lono_missing_network_id_raises(self):
        with self.assertRaisesRegex(ValueError, "missing a valid network_id"):
            create_lono_split([Data()], test_network="case14")

    def test_time_split_sorts_all_valid_timestamps(self):
        dataset = [
            Data(timestamp="2024-01-03"),
            Data(timestamp="2024-01-01"),
            Data(timestamp="2024-01-02"),
        ]

        split = create_time_based_split(dataset, train_ratio=1 / 3, val_ratio=1 / 3, test_ratio=1 / 3)

        self.assertEqual(split, {"train": [1], "val": [2], "test": [0]})

    def test_time_split_empty_timestamp_falls_back_to_input_order(self):
        dataset = [Data(timestamp="2024-01-02"), Data(timestamp=""), Data(timestamp="2024-01-01")]

        split = create_time_based_split(dataset, train_ratio=1 / 3, val_ratio=1 / 3, test_ratio=1 / 3)

        self.assertEqual(split, {"train": [0], "val": [1], "test": [2]})

    def test_time_split_none_timestamp_falls_back_to_input_order(self):
        dataset = [Data(timestamp="2024-01-02"), Data(timestamp=None), Data(timestamp="2024-01-01")]

        split = create_time_based_split(dataset, train_ratio=1 / 3, val_ratio=1 / 3, test_ratio=1 / 3)

        self.assertEqual(split, {"train": [0], "val": [1], "test": [2]})

    def test_time_split_nan_like_timestamp_falls_back_to_input_order(self):
        dataset = [Data(timestamp="2024-01-02"), Data(timestamp=float("nan")), Data(timestamp="2024-01-01")]

        split = create_time_based_split(dataset, train_ratio=1 / 3, val_ratio=1 / 3, test_ratio=1 / 3)

        self.assertEqual(split, {"train": [0], "val": [1], "test": [2]})

    def test_temporal_windows_sort_out_of_order_timestamps(self):
        graphs = [
            Data(y=torch.tensor([2]), timestamp="2024-01-02", sample_id="middle"),
            Data(y=torch.tensor([1]), timestamp="2024-01-01", sample_id="early"),
            Data(y=torch.tensor([3]), timestamp="2024-01-03", sample_id="late"),
        ]

        windows = make_temporal_windows(graphs, input_window=1, forecast_horizon=1)

        self.assertEqual([(w["sample_ids"], w["target_sample_id"]) for w in windows], [(["early"], "middle"), (["middle"], "late")])
        self.assertEqual(windows[0]["target"].item(), 2)

    def test_temporal_windows_preserve_input_order_without_timestamps(self):
        graphs = [
            Data(y=torch.tensor([1]), timestamp="", sample_id="first"),
            Data(y=torch.tensor([2]), timestamp="", sample_id="second"),
            Data(y=torch.tensor([3]), timestamp="", sample_id="third"),
        ]

        windows = make_temporal_windows(graphs, input_window=1, forecast_horizon=1)

        self.assertEqual([(w["sample_ids"], w["target_sample_id"]) for w in windows], [(["first"], "second"), (["second"], "third")])

    def test_temporal_windows_too_short_sequence_returns_empty(self):
        windows = make_temporal_windows([Data(y=torch.tensor([1]), sample_id="only")], input_window=2, forecast_horizon=1)

        self.assertEqual(windows, [])

    def test_temporal_windows_forecast_horizon_selects_future_target(self):
        graphs = [
            Data(y=torch.tensor([1]), timestamp="2024-01-01", sample_id="t1"),
            Data(y=torch.tensor([2]), timestamp="2024-01-02", sample_id="t2"),
            Data(y=torch.tensor([3]), timestamp="2024-01-03", sample_id="t3"),
            Data(y=torch.tensor([4]), timestamp="2024-01-04", sample_id="t4"),
        ]

        windows = make_temporal_windows(graphs, input_window=2, forecast_horizon=2)

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["sample_ids"], ["t1", "t2"])
        self.assertEqual(windows[0]["target_sample_id"], "t4")
        self.assertEqual(windows[0]["target"].item(), 4)

    def test_random_split_tiny_datasets_raise_by_default(self):
        with self.assertRaisesRegex(ValueError, "too small"):
            create_random_split([Data()], seed=1)
        with self.assertRaisesRegex(ValueError, "too small"):
            create_random_split([Data(), Data()], seed=1)

    def test_random_split_allow_empty_keeps_legacy_behavior(self):
        split = create_random_split([Data()], seed=1, allow_empty=True)

        self.assertEqual(split, {"train": [], "val": [], "test": [0]})

    def test_random_split_normal_dataset_is_reproducible_and_non_empty(self):
        dataset = [Data() for _ in range(10)]

        split_a = create_random_split(dataset, seed=42)
        split_b = create_random_split(dataset, seed=42)

        self.assertEqual(split_a, split_b)
        self.assertTrue(split_a["train"])
        self.assertTrue(split_a["val"])
        self.assertTrue(split_a["test"])

    def test_proxy_labels_attach_when_no_existing_labels(self):
        graph = self._graph_for_proxy_label()

        attach_stress_proxy_labels(graph)

        self.assertEqual(graph.y.item(), 1)
        self.assertGreater(graph.risk_score.item(), 1.0)

    def test_proxy_labels_do_not_overwrite_existing_y_by_default(self):
        graph = self._graph_for_proxy_label()
        graph.y = torch.tensor([99])

        with self.assertRaisesRegex(ValueError, "overwrite existing label fields"):
            attach_stress_proxy_labels(graph)
        self.assertEqual(graph.y.item(), 99)

    def test_proxy_labels_do_not_overwrite_existing_y_cls_by_default(self):
        graph = self._graph_for_proxy_label()
        graph.y_cls = torch.tensor([0])

        with self.assertRaisesRegex(ValueError, "overwrite existing label fields"):
            attach_stress_proxy_labels(graph)
        self.assertEqual(graph.y_cls.item(), 0)

    def test_proxy_labels_overwrite_when_requested(self):
        graph = self._graph_for_proxy_label()
        graph.y = torch.tensor([99])

        attach_stress_proxy_labels(graph, overwrite=True)

        self.assertEqual(graph.y.item(), 1)
        self.assertNotEqual(graph.y.item(), 99)

    def _graph_for_proxy_label(self):
        graph = Data(edge_attr=torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.2, 0.0]]))
        graph.edge_feature_names = [
            "component_type",
            "r",
            "x",
            "b_from",
            "b_to",
            "rate_a",
            "pf",
            "qf",
            "pt",
            "qt",
            "loading_ratio",
            "outage_flag",
        ]
        return graph


if __name__ == "__main__":
    unittest.main()
