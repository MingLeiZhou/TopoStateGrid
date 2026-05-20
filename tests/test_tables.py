from pathlib import Path
import tempfile
import unittest

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from topostategrid import build_graph_from_csv_tables, build_graph_from_tables


class TableBuilderTest(unittest.TestCase):
    def test_dataframe_graph_shapes_features_metadata_and_batching(self):
        graph = build_graph_from_tables(
            *_table_fixture(),
            network_id="toy_3bus",
            sample_id="sample_a",
            timestamp="2025-01-01T00:00:00",
        )

        self.assertEqual(graph.num_nodes, 3)
        self.assertEqual(graph.x.shape, (3, 9))
        self.assertEqual(graph.edge_index.shape, (2, 4))
        self.assertEqual(graph.edge_attr.shape, (4, 12))
        self.assertEqual(graph.node_feature_names, [
            "bus_status",
            "bus_type",
            "pd",
            "qd",
            "vm",
            "va",
            "vmax",
            "vmin",
            "normalized_demand",
        ])
        self.assertEqual(graph.edge_feature_names, [
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
        ])
        self.assertEqual(graph.network_id, "toy_3bus")
        self.assertEqual(graph.sample_id, "sample_a")
        self.assertEqual(graph.source_format, "tables")
        self.assertEqual(graph.timestamp, "2025-01-01T00:00:00")
        self.assertIsInstance(graph.metadata_json, str)
        self.assertTrue(torch.isfinite(graph.x).all())
        self.assertTrue(torch.isfinite(graph.edge_attr).all())
        self.assertTrue(torch.isfinite(graph.y_reg).all())
        self.assertTrue(torch.isfinite(graph.risk_score).all())
        batch = next(iter(DataLoader([graph], batch_size=1)))
        self.assertEqual(batch.num_graphs, 1)

    def test_state_tables_override_static_values_and_preserve_timestamp(self):
        bus_df, branch_df = _table_fixture()
        bus_state_df = pd.DataFrame(
            {
                "bus_id": [2],
                "pd": [9.0],
                "qd": [3.0],
                "vm": [0.97],
                "va": [-2.0],
                "timestamp": ["t1"],
            }
        )
        branch_state_df = pd.DataFrame(
            {
                "branch_id": ["b12"],
                "pf": [7.0],
                "qf": [2.0],
                "pt": [-6.5],
                "qt": [-1.8],
                "loading_ratio": [0.9],
                "timestamp": ["t1"],
            }
        )

        graph = build_graph_from_tables(bus_df, branch_df, bus_state_df, branch_state_df)

        self.assertEqual(graph.timestamp, "t1")
        self.assertAlmostEqual(graph.x[1, 2].item(), 9.0)
        self.assertAlmostEqual(graph.x[1, 3].item(), 3.0)
        self.assertAlmostEqual(graph.x[1, 4].item(), 0.97)
        self.assertAlmostEqual(graph.edge_attr[0, 6].item(), 7.0)
        self.assertAlmostEqual(graph.edge_attr[0, 10].item(), 0.9)

    def test_zero_demand_normalization_is_finite_zero(self):
        bus_df, branch_df = _table_fixture()
        bus_df["pd"] = 0.0
        graph = build_graph_from_tables(bus_df, branch_df)

        self.assertTrue(torch.equal(graph.x[:, 8], torch.zeros(3)))

    def test_dataframe_validation_errors(self):
        bus_df, branch_df = _table_fixture()
        with self.assertRaisesRegex(ValueError, "bus_id"):
            build_graph_from_tables(bus_df.drop(columns=["bus_id"]), branch_df)
        with self.assertRaisesRegex(ValueError, "from_bus"):
            build_graph_from_tables(bus_df, branch_df.drop(columns=["from_bus"]))
        with self.assertRaisesRegex(ValueError, "to_bus"):
            build_graph_from_tables(bus_df, branch_df.drop(columns=["to_bus"]))

        bad_branch = branch_df.copy()
        bad_branch.loc[0, "to_bus"] = 99
        with self.assertRaisesRegex(ValueError, "unknown bus"):
            build_graph_from_tables(bus_df, bad_branch)

        duplicate_bus = pd.concat([bus_df, bus_df.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate bus"):
            build_graph_from_tables(duplicate_bus, branch_df)

        bad_numeric = bus_df.copy()
        bad_numeric["pd"] = bad_numeric["pd"].astype(object)
        bad_numeric.loc[0, "pd"] = "bad"
        with self.assertRaisesRegex(ValueError, "non-numeric"):
            build_graph_from_tables(bad_numeric, branch_df)

    def test_csv_tables_match_dataframe_shapes_and_state_override(self):
        bus_df, branch_df = _table_fixture()
        bus_state_df = pd.DataFrame({"bus_id": [3], "pd": [4.5]})
        branch_state_df = pd.DataFrame({"branch_id": ["b23"], "loading_ratio": [0.75]})
        df_graph = build_graph_from_tables(bus_df, branch_df, bus_state_df, branch_state_df, network_id="csv_equiv")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            bus_csv = tmp / "bus.csv"
            branch_csv = tmp / "branch.csv"
            bus_state_csv = tmp / "bus_state.csv"
            branch_state_csv = tmp / "branch_state.csv"
            bus_df.to_csv(bus_csv, index=False)
            branch_df.to_csv(branch_csv, index=False)
            bus_state_df.to_csv(bus_state_csv, index=False)
            branch_state_df.to_csv(branch_state_csv, index=False)
            csv_graph = build_graph_from_csv_tables(
                bus_csv,
                branch_csv,
                bus_state_csv,
                branch_state_csv,
                network_id="csv_equiv",
            )

        self.assertEqual(csv_graph.x.shape, df_graph.x.shape)
        self.assertEqual(csv_graph.edge_attr.shape, df_graph.edge_attr.shape)
        self.assertEqual(csv_graph.node_feature_names, df_graph.node_feature_names)
        self.assertEqual(csv_graph.edge_feature_names, df_graph.edge_feature_names)
        self.assertEqual(csv_graph.network_id, "csv_equiv")
        self.assertAlmostEqual(csv_graph.x[2, 2].item(), 4.5)
        self.assertAlmostEqual(csv_graph.edge_attr[2, 10].item(), 0.75)


def _table_fixture():
    bus_df = pd.DataFrame(
        {
            "bus_id": [1, 2, 3],
            "bus_status": [1, 1, 1],
            "bus_type": [3, 1, 1],
            "pd": [0.0, 2.0, 1.0],
            "qd": [0.0, 0.5, 0.3],
            "vm": [1.0, 0.99, 0.98],
            "va": [0.0, -1.0, -2.0],
            "vmax": [1.1, 1.1, 1.1],
            "vmin": [0.9, 0.9, 0.9],
        }
    )
    branch_df = pd.DataFrame(
        {
            "branch_id": ["b12", "b23"],
            "from_bus": [1, 2],
            "to_bus": [2, 3],
            "component_type": [0, 0],
            "r": [0.01, 0.02],
            "x": [0.05, 0.06],
            "b_from": [0.0, 0.0],
            "b_to": [0.0, 0.0],
            "rate_a": [1.0, 1.5],
            "pf": [0.5, 0.3],
            "qf": [0.1, 0.05],
            "pt": [-0.48, -0.29],
            "qt": [-0.09, -0.04],
            "loading_ratio": [0.4, 0.2],
            "outage_flag": [0, 0],
        }
    )
    return bus_df, branch_df


if __name__ == "__main__":
    unittest.main()
