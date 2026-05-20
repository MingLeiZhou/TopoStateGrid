# QA Validation Report for TopoStateGrid v1.0

## Summary

- Overall result: Pass with issues.
- v1.0 fixes verified: BUG-001 through BUG-007 and BUG-009 were independently verified with commands and probes.
- Partially verified/documented: BUG-008, BUG-010, and BUG-011 remain partial/documentation-only fixes as claimed.
- MVP graph construction usability: usable for bundled OPFData case14/case30 graph construction, static MATPOWER/PGLib case118 graph construction, examples, tests, export/reload, labels, splits, temporal windows, and normalization. A new mixed-source batching issue remains for datasets containing both OPFData and MATPOWER graphs.
- Verified fixes count: 8 fully verified, 3 partially verified/documented.
- Failed claimed fixes count: 0.
- Newly found issues: 3.

## Environment

- Working directory: `/Users/jumiray/Projects/GridGraphKit`
- OS: Darwin/macOS arm64.
- Import command:

```bash
python -c "import topostategrid; print(topostategrid.__version__ if hasattr(topostategrid, '__version__') else 'import ok')"
```

Observed output:

```text
1.0.0
```

- Requested package-version probe initially failed with an OpenMP runtime conflict:

```text
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

- Rerun with `KMP_DUPLICATE_LIB_OK=TRUE` succeeded:

```text
python 3.12.9 | packaged by conda-forge | (main, Mar  4 2025, 22:44:42) [Clang 18.1.8 ]
torch 2.11.0
torch_geometric 2.7.0
numpy 2.4.3
pandas 3.0.2
```

Commands used included:

```bash
python -m unittest discover -s tests -q
pytest -q
python -m compileall -q topostategrid examples tests
python examples/01_build_single_graph.py
python examples/02_build_multiple_state_graphs.py
python examples/03_create_temporal_windows.py
python examples/04_create_splits.py
```

## Verification of Claimed Fixes

### BUG-001: MATPOWER parser rejects comma-delimited MATLAB matrix rows

Claimed status: Fixed  
QA status: Verified  
Evidence:
- Created temporary MATPOWER fixtures covering comma-delimited rows, whitespace-delimited rows, semicolons, inline `%` comments, scientific notation, multi-line matrices, and explicit empty `mpc.branch = [ ];`.
- Valid syntax fixture parsed and built: `bus=(2, 13)`, `branch=(1, 13)`, `edge_attr=(2, 12)`.
- Empty branch fixture parsed and built: `branch=(0, 0)`, `edge_index=(2, 0)`.
- Missing `mpc.bus`, missing `mpc.branch`, and malformed row lengths raised path-aware `ValueError`.

### BUG-002: LONO split silently returns invalid empty splits

Claimed status: Fixed  
QA status: Verified  
Evidence:
- Normal two-network split returned `{'train': [0, 1], 'val': [], 'test': [2]}`.
- Unknown test network raised `ValueError: test_network 'z' is not present in the dataset`.
- Single-network default raised `ValueError: Leave-One-Network-Out split produced an empty train split`.
- Missing `network_id` raised `ValueError: Graph at index 1 is missing a valid network_id`.
- `allow_empty=True` returned `{'train': [], 'val': [], 'test': [0]}` intentionally.

### BUG-003: Time-based split treats empty string timestamps as valid

Claimed status: Fixed  
QA status: Verified  
Evidence:
- Valid timestamps sorted chronologically.
- Empty string, `None`, `"nan"`, and `float("nan")` triggered input-order fallback.
- Mixed non-comparable timestamp types also fell back to input order.
- No empty-string-first sorting was observed.

### BUG-004: `make_temporal_windows` does not sort by timestamp

Claimed status: Fixed  
QA status: Verified  
Evidence:
- Out-of-order timestamped graphs produced windows in chronological order: `[(['early'], 'late'), (['late'], 'last')]`.
- `sort_by_timestamp=False` preserved input order.
- Graphs without timestamps preserved input order.
- Too-short sequences returned `[]`.
- `forecast_horizon=2` selected the correct future target.

### BUG-005: Proxy label attachment overwrites existing labels

Claimed status: Fixed  
QA status: Verified  
Evidence:
- Graph without labels received PyG-compatible `y`, `y_cls`, `y_reg`, and `risk_score` tensors.
- Existing `y`, `y_cls`, `y_reg`, or `risk_score` each raised a `ValueError` unless `overwrite=True`.
- `overwrite=True` intentionally replaced labels.
- README states proxy labels are not cascading-failure ground truth.

### BUG-006: Random split can produce empty train/validation sets for small datasets

Claimed status: Fixed  
QA status: Verified  
Evidence:
- Dataset size 1 and size 2 raised `ValueError` by default.
- `allow_empty=True` preserved old truncation behavior for size 1.
- Dataset size 10 returned non-empty deterministic splits with a fixed seed.
- Note: the new non-empty allocation changes default 10-item split sizes to 6/2/2 for 0.7/0.15/0.15, rather than the previous truncation-style 7/1/2. This appears intentional from the implementation but differs from some old report output.

### BUG-007: Parser validation is weak for malformed OPFData and missing MATPOWER tables

Claimed status: Fixed  
QA status: Verified  
Evidence:
- Malformed JSON raised path-aware `ValueError`.
- Missing `grid`, missing `grid.nodes`, missing `grid.edges`, and empty `grid.nodes.bus` raised path-aware `ValueError`.
- Missing MATPOWER `mpc.bus` and missing `mpc.branch` raised path-aware `ValueError`.

### BUG-008: OPFData discovery scans the whole tree before applying `limit`

Claimed status: Partially fixed  
QA status: Partially verified  
Evidence:
- `discover_opfdata_examples("data/opfdata", network_id="pglib_opf_case14_ieee", limit=3)` returned exactly examples 0, 1, and 2 deterministically across repeated calls.
- Code inspection shows the `network_id` path uses `_discover_opfdata_for_network` and returns once `limit` is reached.
- Global discovery without `network_id` remains tree-scanning and deterministic, matching the documented limitation.

### BUG-009: pytest optional dependency

Claimed status: Fixed  
QA status: Verified  
Evidence:
- `pyproject.toml` includes `[project.optional-dependencies] test = ["pytest"]`.
- README documents `python -m pip install -e ".[test]"`.
- `pytest -q` ran successfully: `30 passed, 3 warnings in 4.82s`.

### BUG-010: Examples depend on repository-local data layout

Claimed status: Partially fixed  
QA status: Partially verified  
Evidence:
- README documents that examples assume repository-local `data/`.
- Normal example runs passed with the current repository data.
- Practical missing-data simulation by copying examples to a temp repo without `data/` showed examples do not fail gracefully:
  - `01_build_single_graph.py` raised a traceback after falling back to a missing MATPOWER file.
  - `02_build_multiple_state_graphs.py` and `04_create_splits.py` raised tracebacks from split validation on empty graph lists.
  - `03_create_temporal_windows.py` wrote an empty windows file.
- This is consistent with the claim that the issue is only documented, not fully fixed.

### BUG-011: Repeated example runs overwrite outputs

Claimed status: Partially fixed  
QA status: Partially verified  
Evidence:
- README documents that example scripts overwrite corresponding files in `outputs/`.
- Re-running examples recreated the same output paths.
- No output versioning or overwrite protection was implemented, matching the claimed remaining limitation.

## Regression Test Results

### Import

Passed. Actual version is `1.0.0`.

### Unit Tests

Command:

```bash
python -m unittest discover -s tests -q
```

Result:

```text
Ran 30 tests in 0.034s
OK
```

### Pytest

Command:

```bash
pytest -q
```

Result:

```text
30 passed, 3 warnings in 4.82s
```

Warnings were from third-party `torch_geometric`/`torch` deprecations.

### Compileall

Command:

```bash
python -m compileall -q topostategrid examples tests
```

Result: passed.

### Example Scripts

- `python examples/01_build_single_graph.py`: passed.
  - Built case14 graph: `num_nodes: 14`, `num_edges: 40`, `x=(14, 9)`, `edge_attr=(40, 12)`.
  - Wrote/reloaded `outputs/graphs.pt`; also writes `metadata.csv` and `README_generated.md`.
- `python examples/02_build_multiple_state_graphs.py`: passed.
  - Built 16 case14 graphs.
  - `x=(14, 9)`, `edge_attr=(40, 12)`.
  - Wrote `graphs_multi.pt`, `metadata_multi.csv`, `split_random.json`.
- `python examples/03_create_temporal_windows.py`: passed.
  - Built 10 graphs and 7 windows.
  - First context: `example_0`, `example_1`, `example_2`; target `example_3`.
  - Wrote `temporal_windows.pt`.
- `python examples/04_create_splits.py`: passed.
  - Built 10 graphs.
  - Wrote `split_random.json`, `split_time.json`, `split_lono.json`.

### Graph Object Validation

OPFData-only graphs passed:
- OPFData same-network batching passed.
- OPFData mixed case14/case30 batching passed.
- Shapes and finite tensors were valid.

MATPOWER-only graph passed:
- Single MATPOWER graph batched successfully.
- case118 shape: `x=(118, 9)`, `edge_index=(2, 372)`, `edge_attr=(372, 12)`.

Mixed OPFData + MATPOWER batching failed:

```text
FAIL opf + matpower KeyError 'objective'
```

Details:
- OPFData graph metadata: `{'objective': ...}`
- MATPOWER graph metadata: `{'format': 'MATPOWER/PGLib', 'static_only': True}`
- PyG collation fails because `metadata` dictionaries have different keys.

### Export/Reload

Passed:
- `load_graphs("outputs/graphs.pt")`: list length 1.
- `load_graphs("outputs/graphs_multi.pt")`: list length 16.
- `load_graphs("outputs/temporal_windows.pt")`: list length 7.

### Normalization

Passed:
- Fit used training graph statistics.
- Zero standard deviation features were replaced with `1.0` std.
- Transform on validation graph worked.
- Labels and metadata were preserved.
- `in_place=False` did not mutate the original validation graph.

## New Issues Found

### QA-001: Mixed OPFData and MATPOWER graphs fail PyG DataLoader batching

Severity: High  
Area: builder / graph object correctness  
Reproduction steps:
1. Build one OPFData graph:

```python
from topostategrid import build_graph_from_opfdata_json, discover_opfdata_examples
p = discover_opfdata_examples("data/opfdata", network_id="pglib_opf_case14_ieee", limit=1)[0]
g_opf = build_graph_from_opfdata_json(p, attach_proxy_label=True)
```

2. Build one MATPOWER graph:

```python
from topostategrid import build_graph_from_matpower
g_m = build_graph_from_matpower("data/pglib/pglib_opf_case118_ieee.m")
```

3. Batch them:

```python
from torch_geometric.loader import DataLoader
next(iter(DataLoader([g_opf, g_m], batch_size=2)))
```

Expected behavior:
Supported graph outputs with the same feature schema should be batchable, especially for cross-network datasets.

Actual behavior:

```text
KeyError: 'objective'
```

The failure is caused by heterogeneous `data.metadata` dictionaries. OPFData metadata contains `objective`; MATPOWER metadata contains `format` and `static_only`.

Suggested fix:
Avoid storing arbitrary heterogeneous dicts as PyG-collated attributes, or normalize metadata keys across graph sources. Options include converting metadata to a JSON string, storing source-specific metadata under consistently keyed fields, or documenting that callers must pass `exclude_keys=["metadata"]` to `DataLoader`.

### QA-002: Requested package-version probe can abort with OpenMP runtime conflict

Severity: Medium  
Area: environment / docs  
Reproduction steps:

```bash
python - <<'PY'
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
PY
```

Expected behavior:
Environment validation should run cleanly.

Actual behavior:

```text
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

The probe succeeded only after setting `KMP_DUPLICATE_LIB_OK=TRUE`.

Suggested fix:
Document the known macOS/conda OpenMP conflict and preferred dependency installation approach. Avoid recommending `KMP_DUPLICATE_LIB_OK` as a normal fix because it is an unsafe workaround.

### QA-003: Examples do not fail gracefully when repository-local data is missing

Severity: Low  
Area: examples / docs  
Reproduction steps:
1. Copy example scripts to a temporary directory without `data/`.
2. Run with `PYTHONPATH` pointing at this package.

Expected behavior:
Since README documents the local-data assumption, a clear user-facing error would be preferable when data is absent.

Actual behavior:
- `01_build_single_graph.py` raises a traceback after trying a missing fallback `.m` file.
- `02_build_multiple_state_graphs.py` and `04_create_splits.py` raise tracebacks when split validation receives empty datasets.
- `03_create_temporal_windows.py` silently writes an empty `temporal_windows.pt`.

Suggested fix:
Add explicit data-existence checks and short error messages in examples, or provide a tiny committed fixture dataset.

## Documentation Mismatches

- `buglist_v1.0.md` reports `numpy: 1.26.4`, but the current validation environment reports `numpy: 2.4.3`.
- `buglist_v1.0.md` example split outputs for `examples/04_create_splits.py` no longer match the current run. Current random/time split sizes are 6/2/2 for 10 graphs, not 7/1/2. This appears to be due to the new non-empty split allocation and is not necessarily a functional bug.
- README says examples write files “including” the listed outputs, but the list omits `split_time.json`, which `examples/04_create_splits.py` creates. This is minor because “including” is not exhaustive.

## Final QA Decision

Pass with issues.

The v1.0 bugfix claims are mostly correct: the high and medium issues from the original bug list were verified as fixed, and the documented partial fixes are accurately described. The project is usable as an MVP graph-construction package for the bundled examples and same-source graph datasets.

Recommended next steps:
1. Fix mixed OPFData + MATPOWER batching by making graph metadata PyG-collation-safe.
2. Add example data-existence checks or a tiny fixture dataset for portable examples.
3. Document the macOS/conda OpenMP conflict if it is reproducible for other users.
4. Update `buglist_v1.0.md` or release notes to reflect the actual environment versions and current split outputs.
