# QA Validation Report for TopoStateGrid v1.1

## Summary

- QA decision: Pass with issues.
- Overall result: v1.1 development claims are mostly verified by independent commands and regression probes.
- MVP usability: usable as an MVP graph-construction package for OPFData, MATPOWER/PGLib, pandas DataFrame tables, CSV tables, and pandapower `net` objects in this environment.
- Fully verified claims:
  - Package imports as `topostategrid` and reports `1.1.0`.
  - Standard tests pass.
  - Existing examples pass.
  - New table and pandapower examples pass.
  - DataFrame and CSV graph construction work.
  - pandapower graph construction works with installed pandapower 3.4.0.
  - Cross-source batching works, including OPFData + MATPOWER + DataFrame + CSV + pandapower.
  - Metadata is now batch-safe JSON string metadata.
- Partially verified:
  - pandapower missing-dependency branch was not directly exercised because pandapower is installed.
- Failed claimed features: 0.
- Newly found issues: 2 low-risk documentation/environment issues.

## Environment

Working directory:

```text
/Users/jumiray/Projects/GridGraphKit
```

Import/version command:

```bash
python -c "import topostategrid; print(topostategrid.__version__ if hasattr(topostategrid, '__version__') else 'import ok')"
```

Observed output:

```text
1.1.0
```

Package-version probe command:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python - <<'PY'
import sys
import torch
import torch_geometric
import numpy
import pandas
print("python", sys.version)
print("torch", torch.__version__)
print("torch_geometric", torch_geometric.__version__)
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
try:
 import pandapower
 print("pandapower", pandapower.__version__)
except Exception as e:
 print("pandapower MISSING/ERROR", type(e).__name__, e)
PY
```

Observed output:

```text
python 3.12.9
torch 2.11.0
torch_geometric 2.7.0
numpy 2.4.3
pandas 2.3.3
pandapower 3.4.0
```

Note: `KMP_DUPLICATE_LIB_OK=TRUE` was used for the package-version probe because this environment previously showed OpenMP runtime conflicts when importing numeric/PyTorch packages together.

## Development Report Checklist

Source file reviewed first:

```text
v1.1_development_report.md
```

Extracted claims:

- New APIs:
  - `build_graph_from_pandapower`
  - `build_graph_from_tables`
  - `build_graph_from_csv_tables`
- Claimed package version: `1.1.0`.
- Claimed tests:
  - `python -m unittest discover -s tests -q`: `Ran 39 tests ... OK (skipped=1)`
  - `MPLCONFIGDIR=/private/tmp/topostategrid-mpl pytest -q`: `38 passed, 1 skipped, 3 warnings`
  - `python -m compileall -q topostategrid examples tests`: passed
- Claimed examples:
  - `examples/01_build_single_graph.py`: passed
  - `examples/02_build_multiple_state_graphs.py`: passed
  - `examples/03_create_temporal_windows.py`: passed
  - `examples/04_create_splits.py`: passed
  - `examples/05_build_from_tables.py`: passed, 3 nodes / 4 edges
  - `examples/06_build_from_pandapower.py`: passed, 3 nodes / 4 edges when pandapower installed
- Claimed limitations:
  - Homogeneous bus-branch graph only
  - No GNN model
  - No cascading-failure simulator
  - pandapower line `rate_a` uses `max_i_ka` as approximate proxy
  - No `.mat` support
  - No heterogeneous graph support
  - No real cascading-failure labels

## Standard Test Results

### Import Test

Result: passed.

```text
1.1.0
```

### Unittest

Command:

```bash
python -m unittest discover -s tests -q
```

Result:

```text
Ran 39 tests in 9.547s
OK (skipped=1)
```

Additional observed output:

- Matplotlib config directory warning because `/Users/jumiray/.matplotlib` is not writable.
- pandapower/numba performance warning because `numba` is not installed.

These warnings did not fail the test run.

### Pytest

Command:

```bash
MPLCONFIGDIR=/private/tmp/topostategrid-mpl pytest -q
```

Result:

```text
38 passed, 1 skipped, 3 warnings in 4.16s
```

Warnings were third-party `torch_geometric` / `torch` deprecation warnings.

### Compileall

Command:

```bash
python -m compileall -q topostategrid examples tests
```

Result: passed.

## Example Script Results

### `examples/01_build_single_graph.py`

QA status: Verified.

Observed:

```text
num_nodes: 14
num_edges: 40
node_feature_shape: (14, 9)
edge_feature_shape: (40, 12)
loaded_graph_count: 1
```

Created/updated:

- `outputs/graphs.pt`
- `outputs/metadata.csv`
- `outputs/README_generated.md`

### `examples/02_build_multiple_state_graphs.py`

QA status: Verified.

Observed:

```text
graphs: 16
network_id: pglib_opf_case14_ieee
node_feature_shape: (14, 9)
edge_feature_shape: (40, 12)
```

Created/updated:

- `outputs/graphs_multi.pt`
- `outputs/metadata_multi.csv`
- `outputs/split_random.json`

### `examples/03_create_temporal_windows.py`

QA status: Verified.

Observed:

```text
graphs: 10
windows: 7
first_window_sample_ids: ['example_0', 'example_1', 'example_2']
first_target_sample_id: example_3
```

Created/updated:

- `outputs/temporal_windows.pt`

### `examples/04_create_splits.py`

QA status: Verified.

Observed:

```text
graphs: 10
random_split: {'train': [1, 5, 6, 0, 9, 4], 'val': [7, 2], 'test': [8, 3]}
time_split: {'train': [0, 1, 2, 3, 4, 5], 'val': [6, 7], 'test': [8, 9]}
lono_split: {'train': [0, 1, 2, 3, 4], 'val': [], 'test': [5, 6, 7, 8, 9]}
```

Created/updated:

- `outputs/split_random.json`
- `outputs/split_time.json`
- `outputs/split_lono.json`

### `examples/05_build_from_tables.py`

QA status: Verified.

Observed:

```text
num_nodes: 3
num_edges: 4
node_feature_shape: (3, 9)
edge_feature_shape: (4, 12)
source_format: tables
```

Created/updated:

- `outputs/graphs_tables.pt`
- `outputs/metadata_tables.csv`

### `examples/06_build_from_pandapower.py`

QA status: Verified when pandapower is installed.

Command:

```bash
MPLCONFIGDIR=/private/tmp/topostategrid-mpl python examples/06_build_from_pandapower.py
```

Observed:

```text
num_nodes: 3
num_edges: 4
node_feature_shape: (3, 9)
edge_feature_shape: (4, 12)
source_format: pandapower
```

Created/updated:

- `outputs/graphs_pandapower.pt`
- `outputs/metadata_pandapower.csv`

## New API Validation

### `build_graph_from_tables`

QA status: Verified.

Independent probes covered:

- 3-bus / 2-branch DataFrame graph.
- Shape checks: `x=(3, 9)`, `edge_index=(2, 4)`, `edge_attr=(4, 12)`.
- Finite tensor checks.
- Node-index bounds.
- Bidirectional edge checks.
- Metadata JSON string checks.
- Placeholder label fields for unlabeled graphs:
  - `has_label=[False]`
  - `y=[-1]`
  - `y_cls=[-1]`
- Bidirectional edge-attribute reversal:
  - `b_from` / `b_to` swapped.
  - `pf/qf` and `pt/qt` swapped.
- State table overrides:
  - Bus state overrode `pd`, `qd`, `vm`, and `va`.
  - Branch state overrode `pf`, `qf`, `pt`, `qt`, and `loading_ratio`.
  - Single timestamp was preserved.
- Error cases:
  - Multiple state timestamps without explicit `timestamp` raised `ValueError`.
  - Conflicting bus/branch state timestamps raised `ValueError`.
  - Empty branch table raised `ValueError`.
  - Unknown bus reference raised `ValueError`.
  - Duplicate `branch_id` raised `ValueError`.
  - Non-numeric branch column raised `ValueError`.

### `build_graph_from_csv_tables`

QA status: Verified.

Independent probes covered:

- Temporary CSV bus, branch, bus-state, and branch-state files.
- CSV graph matched DataFrame graph shape.
- State override worked from CSV.
- Missing CSV file surfaced as `FileNotFoundError`.

### `build_graph_from_pandapower`

QA status: Verified for installed pandapower.

Independent probes covered:

- Built a 3-bus / 2-line pandapower network.
- `pp.runpp(net)` was attempted before graph conversion.
- Result graph:
  - `x=(3, 9)`
  - `edge_attr=(4, 12)`
  - `source_format="pandapower"`
  - finite tensors

Partially verified:

- Missing-dependency behavior was not directly tested because pandapower 3.4.0 is installed. The repository unit test also skips this branch under the installed dependency.

## Cross-Source Batching Validation

QA status: Verified.

Independent batch included:

- OPFData graph
- MATPOWER/PGLib graph
- DataFrame table graph
- CSV table graph
- pandapower graph

Observed:

```text
num_graphs=5
batch.x.shape=(141, 9)
batch.edge_attr.shape=(424, 12)
```

This verifies the v1.1 claim that metadata is now batch-safe across source types.

Metadata checks:

- OPFData: `metadata` and `metadata_json` are strings.
- MATPOWER: `metadata` and `metadata_json` are strings.
- Tables: `metadata` and `metadata_json` are strings.
- pandapower: `metadata` and `metadata_json` are strings.

Observed examples:

```text
opfdata metadata_json: {"objective": 2265.953939003096}
matpower metadata_json: {"format": "MATPOWER/PGLib", "static_only": true}
tables metadata_json: {"input_format": "pandas", "num_branches": 1, "num_buses": 2}
pandapower metadata_json: {"input_format": "pandapower", "num_buses": 2, "num_lines": 1, "num_trafos": 0}
```

## Export / Reload Validation

QA status: Verified.

Loaded output artifacts:

- `outputs/graphs.pt`: list length 1
- `outputs/graphs_multi.pt`: list length 16
- `outputs/graphs_tables.pt`: list length 1
- `outputs/graphs_pandapower.pt`: list length 1
- `outputs/temporal_windows.pt`: list length 7

Metadata CSV behavior:

- Unlabeled table and pandapower examples leave `risk_score` and `y` blank.
- Proxy-labeled OPFData examples include `risk_score` and `y`.

## Documentation Consistency

Verified as consistent:

- README states package import name is `topostategrid`.
- README lists v1.1 supported inputs: OPFData, MATPOWER/PGLib, pandapower, DataFrame, CSV.
- README documents pandapower optional install extra.
- README documents `metadata` as JSON string for PyG batching safety.
- README documents placeholder labels via `has_label=False`.
- README lists `split_time.json`, `graphs_tables.pt`, and `graphs_pandapower.pt`.
- README states no GNN, no cascading-failure simulator, no `.mat`, and no heterogeneous graph construction.

Mismatches / caveats:

- `v1.1_development_report.md` reports `numpy: 1.26.4` and `pandas: 3.0.2`, but the actual validation environment reports `numpy: 2.4.3` and `pandas: 2.3.3`.
- The development report says OPFData/MATPOWER mixed batching was already covered by the QA fix path; I re-tested this under v1.1 as part of broader cross-source batching and it now passes.

## New Issues Found

### QA-001: Development report environment versions do not match actual environment

Severity: Low  
Area: docs / environment  
Reproduction steps:
1. Read `v1.1_development_report.md`.
2. Run package version probe in this environment.

Expected behavior:
The development report environment section should match the actual validation environment or be clearly marked as historical.

Actual behavior:
Development report lists:

```text
numpy: 1.26.4
pandas: 3.0.2
```

Actual validation observed:

```text
numpy: 2.4.3
pandas: 2.3.3
```

Suggested fix:
Update the report or release notes to distinguish development-time environment from current QA environment.

### QA-002: Direct unittest command emits environment warnings unless MPLCONFIGDIR/numba are handled

Severity: Low  
Area: environment / docs / tests  
Reproduction steps:
1. Run:

```bash
python -m unittest discover -s tests -q
```

Expected behavior:
The command should pass cleanly or documented optional dependency warnings should be expected.

Actual behavior:
Tests pass, but output includes:

- Matplotlib cache warning because `/Users/jumiray/.matplotlib` is not writable.
- pandapower warning that `numba` cannot be imported and execution may be slow.

Suggested fix:
Document `MPLCONFIGDIR=/private/tmp/topostategrid-mpl` for unittest too, or configure tests to avoid noisy pandapower imports when possible. If speed matters, document optional `numba` for pandapower workflows.

## Final QA Decision

Pass with issues.

v1.1 functionality is verified enough for MVP graph construction. The new DataFrame, CSV, and pandapower paths work; existing OPFData and MATPOWER paths still work; cross-source PyG batching now works with batch-safe JSON metadata and placeholder label tensors.

Recommended next steps:

1. Update `v1.1_development_report.md` environment versions or mark them as development-machine values.
2. Add a CI/test note for `MPLCONFIGDIR` and optional pandapower/numba warnings.
3. If strict optional-dependency behavior matters, add a small test that monkeypatches `importlib.util.find_spec("pandapower")` to verify the missing-dependency branch even when pandapower is installed.
4. Continue with the documented roadmap only after preserving this cross-source batching test coverage.

## Resolution Notes

Status after fixes: addressed.

- QA-001 fixed by updating `v1.1_development_report.md` to match the current validation environment: `numpy 2.4.3`, `pandas 2.3.3`, and `pandapower 3.4.0`.
- QA-002 addressed by documenting `MPLCONFIGDIR=/private/tmp/topostategrid-mpl` for test/render runs and by documenting the optional pandapower `numba` performance warning in `README.md`.
- Added a monkeypatch unit test for the pandapower missing-dependency branch, so the behavior is verified even when pandapower is installed.
- Added `render_graph_sequence` plus `examples/07_render_graph_animation.py` to render constructed graph sequences to GIF/MP4 for visual inspection.
