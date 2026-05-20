from pathlib import Path
import tempfile
import unittest

from topostategrid import parse_matpower_case, parse_opfdata_sample


class ParserTest(unittest.TestCase):
    def test_matpower_accepts_commas_comments_and_empty_branch(self):
        parsed = self._parse_matpower(
            """
            mpc.baseMVA = 100;
            mpc.bus = [
              1, 3, 0, 0, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9; % slack bus
            ];
            mpc.branch = [ ];
            """
        )

        self.assertEqual(parsed.base_mva, 100.0)
        self.assertEqual(parsed.grid["bus"].shape, (1, 13))
        self.assertEqual(parsed.grid["branch"].shape, (0, 0))

    def test_matpower_accepts_whitespace_and_scientific_notation(self):
        parsed = self._parse_matpower(
            """
            mpc.baseMVA = 1e2;
            mpc.bus = [
              1 3 1e-1 2E-1 0 0 1 1.0 0 230 1 1.1 0.9;
              2 1 0 0 0 0 1 9.8e-1 -1 230 1 1.1 0.9;
            ];
            mpc.branch = [
              1 2 1e-2 6e-2 0 100 100 100 0 0 1 -30 30;
            ];
            """
        )

        self.assertEqual(parsed.grid["bus"].shape, (2, 13))
        self.assertEqual(parsed.grid["branch"].shape, (1, 13))
        self.assertAlmostEqual(parsed.grid["bus"][0, 2], 0.1)
        self.assertAlmostEqual(parsed.grid["branch"][0, 3], 0.06)

    def test_matpower_missing_bus_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "missing required matrix 'mpc.bus'"):
            self._parse_matpower("mpc.branch = [ ];")

    def test_matpower_missing_branch_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "missing required matrix 'mpc.branch'"):
            self._parse_matpower("mpc.bus = [1 3 0 0 0 0 1 1 0 230 1 1.1 0.9;];")

    def test_matpower_malformed_row_lengths_raise_clear_error(self):
        with self.assertRaisesRegex(ValueError, "inconsistent row lengths"):
            self._parse_matpower(
                """
                mpc.bus = [
                  1 3 0 0 0 0 1 1 0 230 1 1.1 0.9;
                  2 1 0;
                ];
                mpc.branch = [ ];
                """
            )

    def test_opfdata_malformed_json_has_path_context(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "example_bad.json"
            path.write_text("{bad json", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Malformed OPFData JSON"):
                parse_opfdata_sample(path)

    def test_opfdata_missing_required_fields_raise_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "example_missing.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing required field 'grid'"):
                parse_opfdata_sample(path)

    def test_opfdata_empty_bus_nodes_raise_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "example_empty_bus.json"
            path.write_text('{"grid": {"nodes": {"bus": []}, "edges": {}}}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "empty 'grid.nodes.bus'"):
                parse_opfdata_sample(path)

    def _parse_matpower(self, text: str):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "case_test.m"
            path.write_text(text, encoding="utf-8")
            return parse_matpower_case(path)


if __name__ == "__main__":
    unittest.main()
