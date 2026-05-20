import importlib.util
import unittest
from unittest import mock


class PandapowerBuilderTest(unittest.TestCase):
    def test_missing_pandapower_dependency_message(self):
        from topostategrid.pandapower import build_graph_from_pandapower

        with mock.patch("topostategrid.pandapower.importlib.util.find_spec", return_value=None):
            with self.assertRaisesRegex(ImportError, r'pandapower.*pip install -e "\.\[pandapower\]"'):
                build_graph_from_pandapower(object())

    @unittest.skipIf(importlib.util.find_spec("pandapower") is None, "pandapower is not installed")
    def test_small_pandapower_network_graph(self):
        import pandapower as pp
        from torch_geometric.loader import DataLoader
        from topostategrid import build_graph_from_pandapower

        net = pp.create_empty_network()
        b1 = pp.create_bus(net, vn_kv=110, max_vm_pu=1.1, min_vm_pu=0.9)
        b2 = pp.create_bus(net, vn_kv=110, max_vm_pu=1.1, min_vm_pu=0.9)
        b3 = pp.create_bus(net, vn_kv=110, max_vm_pu=1.1, min_vm_pu=0.9)
        pp.create_ext_grid(net, b1)
        pp.create_load(net, b2, p_mw=10.0, q_mvar=3.0)
        pp.create_line_from_parameters(
            net,
            b1,
            b2,
            length_km=1.0,
            r_ohm_per_km=0.1,
            x_ohm_per_km=0.2,
            c_nf_per_km=0.0,
            max_i_ka=0.4,
        )
        pp.create_line_from_parameters(
            net,
            b2,
            b3,
            length_km=1.0,
            r_ohm_per_km=0.1,
            x_ohm_per_km=0.2,
            c_nf_per_km=0.0,
            max_i_ka=0.4,
        )
        try:
            pp.runpp(net)
        except Exception:
            pass

        graph = build_graph_from_pandapower(net, network_id="pp_3bus")

        self.assertEqual(graph.num_nodes, 3)
        self.assertEqual(graph.x.shape, (3, 9))
        self.assertEqual(graph.edge_index.shape, (2, 4))
        self.assertEqual(graph.edge_attr.shape, (4, 12))
        self.assertEqual(graph.network_id, "pp_3bus")
        self.assertEqual(graph.source_format, "pandapower")
        self.assertTrue(graph.x.isfinite().all())
        self.assertTrue(graph.edge_attr.isfinite().all())
        if hasattr(net, "res_bus") and not net.res_bus.empty:
            self.assertAlmostEqual(graph.x[0, 4].item(), float(net.res_bus.loc[b1, "vm_pu"]), places=4)
        if hasattr(net, "res_line") and not net.res_line.empty and "loading_percent" in net.res_line:
            self.assertAlmostEqual(graph.edge_attr[0, 10].item(), float(net.res_line.iloc[0]["loading_percent"]) / 100.0, places=4)
        batch = next(iter(DataLoader([graph], batch_size=1)))
        self.assertEqual(batch.num_graphs, 1)


if __name__ == "__main__":
    unittest.main()
