# Bug List for TopoStateGrid

## Summary

- Overall project status: the package imports, the available OPFData case14/case30 samples build valid PyTorch Geometric graphs, the static PGLib case118 `.m` file builds a valid graph, and generated graphs can be batched with `torch_geometric.loader.DataLoader`.
- Examples: all four example scripts ran successfully and created the expected files in `outputs/`.
- Tests: `python -m unittest discover -s tests -q` passes 3 tests. `pytest -q` could not run because `pytest` is not installed in this environment.
- Most serious issues: MATPOWER parsing is fragile for valid comma-delimited MATLAB matrix syntax; LONO split edge cases silently produce empty train/test splits; temporal/time split behavior around timestamps is inconsistent with expected robustness; proxy labels overwrite existing labels without warning.
- MVP usability: usable for the bundled MVP graph construction paths, but fragile for external `.m` files, split edge cases, and temporal workflows.

## Environment

- Python version: Python 3.12.9
- OS: Darwin Millers-MacBook-Pro.local 25.4.0 arm64
- Working directory: `/Users/jumiray/Projects/GridGraphKit`
- Package import command:

```bash
python -c "import topostategrid; print(topostategrid.__version__ if hasattr(topostategrid, '__version__') else 'import ok')"
```

Output:

```text
0.1.0
```

- Key package versions:
  - `torch`: 2.11.0
  - `torch_geometric`: 2.7.0
  - `numpy`: 1.26.4
  - `pandas`: 3.0.2
  - `scipy`: 1.17.1
  - `pytest`: missing as a command and package metadata

- One ad hoc probe that imported `torch`, `torch_geometric.data.Data`, and `topostategrid` together failed once with an OpenMP runtime conflict:

```text
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

The same checks continued with `KMP_DUPLICATE_LIB_OK=TRUE`. This appears environment-related, but it is worth documenting in installation notes if other macOS users hit it.

## Critical Bugs

No critical bugs were reproduced. Core import and MVP graph construction worked for the available local OPFData and MATPOWER/PGLib data.

## High Priority Bugs

### BUG-001: MATPOWER parser rejects comma-delimited MATLAB matrix rows

Severity: High  
Area: parser  
Status: Open  
Reproduction steps:
1. Create a MATPOWER-style file with comma-delimited `mpc.bus` rows:

```matlab
mpc.baseMVA = 100;
mpc.bus = [
1, 3, 0, 0, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9;
];
mpc.branch = [ ];
```

2. Run `parse_matpower_case(path)`.

Expected behavior:
The parser should accept realistic MATLAB matrix syntax with commas, whitespace, semicolons, comments, scientific notation, and multi-line rows.

Actual behavior:
The parser splits only on whitespace and passes tokens like `1,` to `float()`.

Error message:

```text
ValueError: could not convert string to float: '1,'
```

Suggested fix:
Normalize commas to whitespace before tokenization in `_parse_mpc_matrix`, and add parser tests covering comma-delimited rows, comments, semicolons, scientific notation, and empty matrices.

### BUG-002: LONO split silently returns invalid empty splits for unknown or single-network datasets

Severity: High  
Area: splits  
Status: Open  
Reproduction steps:
1. Run:

```python
from torch_geometric.data import Data
from topostategrid import create_lono_split

ds = [Data(network_id="a"), Data(network_id="a")]
print(create_lono_split(ds, test_network="missing"))
print(create_lono_split(ds, test_network="a"))
```

Expected behavior:
Unknown `test_network` should be handled explicitly, e.g. with a clear `ValueError`. A dataset with only the held-out network should also fail clearly or document that train will be empty.

Actual behavior:

```text
lono unknown {'train': [0, 1], 'val': [], 'test': []}
lono only network {'train': [], 'val': [], 'test': [0, 1]}
```

This can silently produce unusable evaluation splits.

Suggested fix:
Validate that `test_network` exists and that both train and test contain at least one graph, unless an explicit `allow_empty=True` option is provided.

## Medium Priority Bugs

### BUG-003: Time-based split treats empty string timestamps as valid timestamps

Severity: Medium  
Area: splits / temporal  
Status: Open  
Reproduction steps:
1. Build OPFData graphs from local samples. `_make_data` stores missing timestamps as `""`.
2. Run `create_time_based_split`.
3. Or run:

```python
from torch_geometric.data import Data
from topostategrid import create_time_based_split

ds = [Data(timestamp=""), Data(timestamp="2024-01-01"), Data(timestamp="2023-01-01")]
print(create_time_based_split(ds, train_ratio=.34, val_ratio=.33, test_ratio=.33))
```

Expected behavior:
If timestamps are missing, the function should fall back to input order as documented.

Actual behavior:
Because `"" is not None`, the function sorts the empty timestamp before real timestamps:

```text
{'train': [0], 'val': [], 'test': [2, 1]}
```

Suggested fix:
Treat `None` and empty strings as missing timestamps. Only sort by timestamp when all timestamps are non-empty and comparable.

### BUG-004: `make_temporal_windows` does not sort by timestamp when timestamps exist

Severity: Medium  
Area: temporal  
Status: Open  
Reproduction steps:
1. Run:

```python
from torch_geometric.data import Data
import torch
from topostategrid import make_temporal_windows

graphs = [
    Data(y=torch.tensor([0]), timestamp="2024-01-02", sample_id="late"),
    Data(y=torch.tensor([1]), timestamp="2024-01-01", sample_id="early"),
    Data(y=torch.tensor([2]), timestamp="2024-01-03", sample_id="last"),
]
print([(w["sample_ids"], w["target_sample_id"]) for w in make_temporal_windows(graphs, input_window=1)])
```

Expected behavior:
For timestamped graphs, windows should be built in chronological order, or documentation should clearly state that callers must pre-sort.

Actual behavior:

```text
[(['late'], 'early'), (['early'], 'last')]
```

The first window uses `late` to predict `early`, which is temporally reversed.

Suggested fix:
Either sort by timestamp when all timestamps are available, or rename/document the function as input-order-only and add a helper or assertion for chronological ordering.

### BUG-005: Proxy label attachment overwrites existing labels without warning

Severity: Medium  
Area: labels  
Status: Open  
Reproduction steps:
1. Run:

```python
import torch
from torch_geometric.data import Data
from topostategrid import attach_stress_proxy_labels

g = Data(edge_attr=torch.tensor([[0.,0,0,0,0,0,0,0,0,0,1.2,0.]]))
g.edge_feature_names = ["component_type","r","x","b_from","b_to","rate_a","pf","qf","pt","qt","loading_ratio","outage_flag"]
g.y = torch.tensor([99])
attach_stress_proxy_labels(g)
print(g.y)
```

Expected behavior:
Existing labels should be preserved unless the caller explicitly opts into overwrite, or the function should warn.

Actual behavior:

```text
y [1]
```

The previous `y=[99]` was replaced silently.

Suggested fix:
Add an `overwrite: bool = False` argument, raise or warn when labels already exist, and document the in-place behavior.

### BUG-006: Random split can produce empty train/validation sets for small datasets

Severity: Medium  
Area: splits  
Status: Open  
Reproduction steps:

```python
from torch_geometric.data import Data
from topostategrid import create_random_split
print(create_random_split([Data()], seed=1))
```

Expected behavior:
For very small datasets, either enforce minimum split sizes, warn clearly, or document that integer truncation can produce empty splits.

Actual behavior:

```text
{'train': [], 'val': [], 'test': [0]}
```

Suggested fix:
Validate minimum dataset sizes for requested ratios or document the truncation behavior and provide an option for minimum split counts.

### BUG-007: Parser has weak validation for missing MATPOWER tables and malformed OPFData

Severity: Medium  
Area: parser / builder  
Status: Open  
Reproduction steps:
1. MATPOWER: parse a file without `mpc.bus` or `mpc.branch`.
2. OPFData: parse malformed JSON or a file with empty bus nodes.

Expected behavior:
The parser should raise clear, domain-specific errors that name the missing required section and source path.

Actual behavior:
- Missing MATPOWER matrices become empty arrays in `_parse_mpc_matrix`; downstream code may fail later or build a graph with zero edges without a warning.
- Malformed JSON raises a raw `JSONDecodeError`.
- Empty OPFData bus nodes raise `ValueError: OPFData sample has no bus nodes: <path>`, which is clearer than the JSON case.

Suggested fix:
Validate required parser outputs in `parse_*` functions, wrap JSON errors with path context, and decide whether missing `mpc.branch` is allowed for isolated-bus cases or should be rejected.

### BUG-008: OPFData discovery scans the entire data tree before applying `limit`

Severity: Medium  
Area: parser / examples / performance  
Status: Open  
Reproduction steps:
1. Run any example script.
2. Observe that each script calls `discover_opfdata_examples(..., limit=N)`.
3. The implementation first expands all `**/group_*/example_*.json` files, sorts them, and only then applies `limit`.

Expected behavior:
Examples with small limits should not need to enumerate and sort every local sample if the data tree is large.

Actual behavior:
The local tree contained 23,587 OPFData JSON files. The examples still completed, but this approach is slow and memory-heavy for larger datasets.

Suggested fix:
Use a more targeted discovery strategy by network path when `network_id` is supplied, or stream candidates with a bounded heap/early stopping after stable ordering can be guaranteed.

## Low Priority Bugs / Documentation Issues

### BUG-009: `pytest` compatibility is documented, but pytest is not installed or declared as an optional test dependency

Severity: Low  
Area: docs / tests / environment  
Status: Open  
Reproduction steps:

```bash
pytest -q
```

Expected behavior:
Either `pytest` should be available in the documented environment, or the README/packaging should tell users how to install test extras.

Actual behavior:

```text
zsh:1: command not found: pytest
```

Suggested fix:
Add a test extra such as `TopoStateGrid[test]` with `pytest`, or keep the README focused on `unittest` and mention pytest as optional.

### BUG-010: Examples are portable within the repository but depend on bundled local data layout

Severity: Low  
Area: examples / docs  
Status: Open  
Reproduction steps:
1. Inspect examples.
2. They resolve `ROOT = Path(__file__).resolve().parents[1]` and look under `ROOT / "data" / "opfdata"`.

Expected behavior:
Examples should either ship a tiny fixture dataset or accept an input path argument so they can run after installation without the full local data tree.

Actual behavior:
The examples assume the repository data layout and silently fall back only in `01_build_single_graph.py`. Other examples can produce empty graph lists or fail later if the large data folder is absent.

Suggested fix:
Add CLI arguments for input/output paths and include a tiny committed fixture used by examples and tests.

### BUG-011: Repeated example runs overwrite outputs without warning or versioning

Severity: Low  
Area: export / examples  
Status: Open  
Reproduction steps:
1. Run the examples repeatedly.
2. Inspect `outputs/`.

Expected behavior:
Repeated runs should either clearly document overwrite behavior, write to run-specific directories, or require an explicit overwrite flag for irreversible outputs.

Actual behavior:
Files such as `graphs.pt`, `graphs_multi.pt`, `metadata.csv`, and split JSON files are overwritten.

Suggested fix:
Document overwrite behavior and consider optional timestamped output directories or an `overwrite` flag for user-facing scripts.

## Successful Checks

- Import succeeded and reported version `0.1.0`.
- Example scripts succeeded:
  - `python examples/01_build_single_graph.py`: created `outputs/graphs.pt`, `outputs/metadata.csv`, `outputs/README_generated.md`; built one case14 graph with 14 nodes, 40 directed edges, node shape `(14, 9)`, edge shape `(40, 12)`.
  - `python examples/02_build_multiple_state_graphs.py`: created `outputs/graphs_multi.pt`, `outputs/metadata_multi.csv`, `outputs/split_random.json`; built 16 case14 graphs.
  - `python examples/03_create_temporal_windows.py`: created `outputs/temporal_windows.pt`; built 7 windows from 10 graphs with input window 3 and horizon 1.
  - `python examples/04_create_splits.py`: created `outputs/split_random.json`, `outputs/split_time.json`, `outputs/split_lono.json`; built 10 graphs across case14 and case30.
- Unit tests:
  - `python -m unittest discover -s tests -q`: 3 tests passed.
- Local data detection:
  - OPFData JSON count: 23,587
  - OPFData networks: `pglib_opf_case14_ieee` with 15,000 samples, `pglib_opf_case30_ieee` with 8,587 samples
  - MATPOWER/PGLib cases: `data/pglib/pglib_opf_case118_ieee.m`
- Graph object checks:
  - case14 OPFData: `x=(14, 9)`, `edge_index=(2, 40)`, `edge_attr=(40, 12)`, finite tensors, bidirectional edges present.
  - case30 OPFData: `x=(30, 9)`, `edge_index=(2, 82)`, `edge_attr=(82, 12)`, finite tensors, bidirectional edges present.
  - case118 MATPOWER: `x=(118, 9)`, `edge_index=(2, 372)`, `edge_attr=(372, 12)`, finite tensors, max node index 117, static loading ratios all zero.
  - PyG `DataLoader` batched four generated graphs successfully.
- Export/reload:
  - `load_graphs` loaded `outputs/graphs.pt`, `outputs/graphs_multi.pt`, and `outputs/temporal_windows.pt`.
  - `metadata.csv` row count matched `graphs.pt` length.
  - `metadata_multi.csv` row count matched `graphs_multi.pt` length.
  - Split JSON indices had no duplicates and stayed within dataset length for the generated examples.

## Test Coverage Gaps

- No tests use real local OPFData files.
- No tests use the real `pglib_opf_case118_ieee.m` file.
- No parser tests cover comments, commas, scientific notation, optional MATPOWER columns, missing matrices, malformed JSON, missing fields, `None`, NaN, or empty arrays.
- No tests assert exact node feature order or edge feature order against README names.
- No tests check OPFData load aggregation against a known fixture with multiple loads per bus.
- No tests check bidirectional edge flow reversal values.
- No tests cover `rate_a=0` or missing rate handling beyond implicit behavior.
- No tests cover LONO unknown network or single-network datasets.
- No tests cover timestamp sorting, empty timestamp fallback, or out-of-order temporal windows.
- No tests check label overwrite behavior.
- No tests check that normalization uses only training graphs with deliberately different train/test distributions.
- No tests check that normalization does not mutate inputs when `in_place=False` beyond shape preservation.
- No tests cover output overwrite behavior or portability of relative paths.

## Recommended Next Steps

1. Harden MATPOWER parsing and add parser fixtures for realistic syntax variants.
2. Add split validation for LONO and small datasets so invalid experiments fail loudly.
3. Fix timestamp handling: distinguish missing timestamps from empty strings, and document or enforce chronological ordering for temporal windows.
4. Add overwrite protection or warnings for proxy labels.
5. Add small committed fixtures for OPFData and MATPOWER so tests and examples do not depend on the developer's full local data tree.
6. Expand unit tests around feature order, edge reversals, loading ratios, normalization leakage, export/reload, and malformed inputs.
7. Document test dependencies and the macOS OpenMP runtime issue if reproducible for other users.
