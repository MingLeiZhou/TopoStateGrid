import importlib.util
import tempfile
from pathlib import Path
import unittest

import pandas as pd
from torch_geometric.loader import DataLoader

from topostategrid import (
    build_graph_from_csv_tables,
    build_graph_from_matpower,
    build_graph_from_pandapower,
    build_graph_from_tables,
)


class CrossSourceBatchingTest(unittest.TestCase):
    def test_matpower_dataframe_csv_and_optional_pandapower_batch_together(self):
        bus_df, branch_df = _table_fixture()
        graphs = [
            build_graph_from_matpower(Path("data/pglib/pglib_opf_case118_ieee.m")),
            build_graph_from_tables(bus_df, branch_df, network_id="table_graph"),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            bus_csv = tmp / "bus.csv"
            branch_csv = tmp / "branch.csv"
            bus_df.to_csv(bus_csv, index=False)
            branch_df.to_csv(branch_csv, index=False)
            graphs.append(build_graph_from_csv_tables(bus_csv, branch_csv, network_id="csv_graph"))

        if importlib.util.find_spec("pandapower") is not None:
            graphs.append(build_graph_from_pandapower(_pandapower_fixture(), network_id="pp_graph"))

        batch = next(iter(DataLoader(graphs, batch_size=len(graphs))))

        self.assertEqual(batch.num_graphs, len(graphs))
        self.assertEqual(batch.x.shape[1], 9)
        self.assertEqual(batch.edge_attr.shape[1], 12)


def _table_fixture():
    bus_df = pd.DataFrame(
        {
            "bus_id": [1, 2, 3],
            "bus_status": [1, 1, 1],
            "bus_type": [3, 1, 1],
            "pd": [0.0, 1.0, 0.0],
            "qd": [0.0, 0.2, 0.0],
        }
    )
    branch_df = pd.DataFrame(
        {
            "branch_id": ["a", "b"],
            "from_bus": [1, 2],
            "to_bus": [2, 3],
            "r": [0.01, 0.02],
            "x": [0.05, 0.06],
        }
    )
    return bus_df, branch_df


def _pandapower_fixture():
    import pandapower as pp

    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=110)
    b2 = pp.create_bus(net, vn_kv=110)
    b3 = pp.create_bus(net, vn_kv=110)
    pp.create_ext_grid(net, b1)
    pp.create_load(net, b2, p_mw=1.0, q_mvar=0.2)
    pp.create_line_from_parameters(net, b1, b2, 1.0, 0.1, 0.2, 0.0, 0.4)
    pp.create_line_from_parameters(net, b2, b3, 1.0, 0.1, 0.2, 0.0, 0.4)
    try:
        pp.runpp(net)
    except Exception:
        pass
    return net


if __name__ == "__main__":
    unittest.main()
