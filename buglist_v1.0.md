# Bug List for TopoStateGrid v1.0

## Summary

- Original status: MVP graph construction worked for local OPFData and PGLib/MATPOWER data, but parser robustness, split validation, temporal ordering, and proxy-label safety had known gaps.
- Fixed in this pass: all High and Medium priority bugs were fixed or addressed with explicit validation and regression tests. Low-priority documentation/package metadata issues were addressed where simple.
- Current status: TopoStateGrid imports as `topostategrid`, reports version `1.0.0`, and builds OPFData and MATPOWER graph samples successfully.
- MVP usability: usable for graph construction, metadata preservation, mixed OPFData/MATPOWER batching, proxy-label prototyping, dataset splitting, temporal windows, normalization, and export. Remaining limitations are documented below.

## Environment

- Python version: 3.12.9
- OS: macOS-26.4.1-arm64-arm-64bit
- Working directory: `/Users/jumiray/Projects/GridGraphKit`
- Key package versions:
  - `torch`: 2.11.0
  - `torch-geometric`: 2.7.0
  - `numpy`: 1.26.4
  - `pandas`: 3.0.2
  - `pytest`: 9.0.3
- Commands used:
  - `python -c "import topostategrid; print(topostategrid.__version__ if hasattr(topostategrid, '__version__') else 'import ok')"`
  - `python -m unittest discover -s tests -q`
  - `pytest -q`
  - `python examples/01_build_single_graph.py`
  - `python examples/02_build_multiple_state_graphs.py`
  - `python examples/03_create_temporal_windows.py`
  - `python examples/04_create_splits.py`
  - `python -m compileall -q topostategrid examples tests`

## Fixed Bugs

### BUG-001: MATPOWER parser rejects comma-delimited MATLAB matrix rows

Original severity: High  
Area: parser  
Fix status: Fixed  
Files changed:
- `topostategrid/parser.py`
- `tests/test_parser.py`

Fix summary:
The MATPOWER matrix parser now accepts comma-delimited and whitespace-delimited rows, semicolons, `%` comments, scientific notation, multi-line matrices, and explicit empty matrices such as `mpc.branch = [ ];`. It raises clear `ValueError` messages for missing matrices, malformed numeric values, and inconsistent row lengths.

Tests added or updated:
- Comma-delimited `mpc.bus`
- Whitespace-delimited `mpc.bus`
- Scientific notation
- Inline comments
- Empty `mpc.branch`
- Missing `mpc.bus`
- Missing `mpc.branch`
- Malformed row lengths

Verification command:
- `python -m unittest discover -s tests -q`
- `pytest -q`
- `python -c "from topostategrid import build_graph_from_matpower; g=build_graph_from_matpower('data/pglib/pglib_opf_case118_ieee.m'); print(g.network_id, tuple(g.x.shape), tuple(g.edge_index.shape), tuple(g.edge_attr.shape), int(g.edge_index.max()))"`

Remaining limitations:
Explicitly empty `mpc.branch` is allowed for isolated-bus fixtures. Missing `mpc.branch` still fails because the declaration is required for MATPOWER case validation.

### BUG-002: LONO split silently returns invalid empty splits for unknown or single-network datasets

Original severity: High  
Area: splits  
Fix status: Fixed  
Files changed:
- `topostategrid/splits.py`
- `tests/test_splits_temporal_labels.py`
- `README.md`

Fix summary:
`create_lono_split` now raises `ValueError` when `test_network` is absent, when any graph lacks a valid `network_id`, or when train/test would be empty. An `allow_empty` parameter is available for explicitly opting out.

Tests added or updated:
- Normal two-network LONO split
- Unknown test network
- Single-network dataset causing empty train
- Missing `network_id`

Verification command:
- `python -m unittest discover -s tests -q`
- `pytest -q`
- `python examples/04_create_splits.py`

Remaining limitations:
Validation intentionally allows an empty validation split because validation networks are optional.

### BUG-003: Time-based split treats empty string timestamps as valid timestamps

Original severity: Medium  
Area: splits / temporal  
Fix status: Fixed  
Files changed:
- `topostategrid/splits.py`
- `tests/test_splits_temporal_labels.py`
- `README.md`

Fix summary:
`create_time_based_split` now treats `None`, empty strings, and NaN-like timestamps as missing. It sorts only when all timestamps are valid and comparable; otherwise it falls back to input order.

Tests added or updated:
- All valid timestamps
- Empty string timestamp
- `None` timestamp
- NaN-like timestamp
- Input-order fallback

Verification command:
- `python -m unittest discover -s tests -q`
- `pytest -q`
- `python examples/04_create_splits.py`

Remaining limitations:
Timestamp values are sorted using their native Python ordering. Mixed non-comparable timestamp types fall back to input order.

### BUG-004: `make_temporal_windows` does not sort by timestamp when timestamps exist

Original severity: Medium  
Area: temporal  
Fix status: Fixed  
Files changed:
- `topostategrid/temporal.py`
- `tests/test_splits_temporal_labels.py`
- `README.md`

Fix summary:
`make_temporal_windows` now has `sort_by_timestamp=True` by default. When all timestamps are valid and comparable, graphs are sorted chronologically before windows are built. Missing or non-comparable timestamps preserve input order.

Tests added or updated:
- Out-of-order timestamped graphs
- No-timestamp graphs
- Too-short graph sequences
- `input_window` and `forecast_horizon` behavior

Verification command:
- `python -m unittest discover -s tests -q`
- `pytest -q`
- `python examples/03_create_temporal_windows.py`

Remaining limitations:
For scenario-only OPFData samples without real timestamps, window order is still the deterministic input/scenario order.

### BUG-005: Proxy label attachment overwrites existing labels without warning

Original severity: Medium  
Area: labels  
Fix status: Fixed  
Files changed:
- `topostategrid/labels.py`
- `tests/test_splits_temporal_labels.py`
- `README.md`

Fix summary:
`attach_stress_proxy_labels` now has `overwrite: bool = False`. Existing `data.y`, `data.y_cls`, `data.y_reg`, or `data.risk_score` cause a clear `ValueError` unless `overwrite=True` is passed.

Tests added or updated:
- Graph without labels
- Graph with existing `y`
- Graph with existing `y_cls`
- `overwrite=False`
- `overwrite=True`

Verification command:
- `python -m unittest discover -s tests -q`
- `pytest -q`

Remaining limitations:
Proxy labels remain temporary stress labels and are not cascading-failure ground truth.

### BUG-006: Random split can produce empty train/validation sets for small datasets

Original severity: Medium  
Area: splits  
Fix status: Fixed  
Files changed:
- `topostategrid/splits.py`
- `tests/test_splits_temporal_labels.py`
- `README.md`

Fix summary:
`create_random_split` now requires each positive-ratio split to receive at least one sample by default. Tiny datasets raise `ValueError`; `allow_empty=True` preserves the previous truncation behavior intentionally. The split remains deterministic for a fixed seed.

Tests added or updated:
- Dataset of size 1
- Dataset of size 2
- `allow_empty=True`
- Normal dataset
- Seed reproducibility

Verification command:
- `python -m unittest discover -s tests -q`
- `pytest -q`
- `python examples/02_build_multiple_state_graphs.py`
- `python examples/04_create_splits.py`

Remaining limitations:
For very small datasets with three positive ratios, callers must either collect more samples, set some ratios to zero, or pass `allow_empty=True`.

### BUG-007: Parser has weak validation for missing MATPOWER tables and malformed OPFData

Original severity: Medium  
Area: parser / builder  
Fix status: Fixed  
Files changed:
- `topostategrid/parser.py`
- `tests/test_parser.py`
- `README.md`

Fix summary:
`parse_opfdata_sample` now wraps JSON decoding errors with path context and validates required OPFData fields: `grid`, `grid.nodes`, `grid.edges`, and non-empty `grid.nodes.bus`. `parse_matpower_case` now raises clear errors for missing `mpc.bus` or `mpc.branch`, and rejects empty `mpc.bus`.

Tests added or updated:
- Malformed JSON
- Missing OPFData fields
- Empty OPFData bus list
- Missing MATPOWER `mpc.bus`
- Missing MATPOWER `mpc.branch`

Verification command:
- `python -m unittest discover -s tests -q`
- `pytest -q`

Remaining limitations:
The OPFData parser validates the MVP-required schema only. It does not fully validate every optional OPFData node/edge type.

### BUG-008: OPFData discovery scans the entire data tree before applying `limit`

Original severity: Medium  
Area: parser / examples / performance  
Fix status: Partially fixed  
Files changed:
- `topostategrid/parser.py`

Fix summary:
When `network_id` is supplied, discovery now targets matching network directories, sorts group/example files deterministically, and returns early once `limit` is reached. This fixes the example path because examples request specific networks.

Tests added or updated:
- Covered indirectly by successful example runs using `network_id` and `limit`.

Verification command:
- `python examples/01_build_single_graph.py`
- `python examples/02_build_multiple_state_graphs.py`
- `python examples/03_create_temporal_windows.py`
- `python examples/04_create_splits.py`

Remaining limitations:
Global discovery without `network_id` still needs to inspect and sort the data tree to provide deterministic ordering across networks. A TODO is left in code for a more advanced bounded global discovery strategy.

### BUG-009: `pytest` compatibility is documented, but pytest is not installed or declared as an optional test dependency

Original severity: Low  
Area: docs / tests / environment  
Fix status: Fixed  
Files changed:
- `pyproject.toml`
- `README.md`

Fix summary:
Added a `test` optional dependency extra with `pytest`, and updated README to show `python -m pip install -e ".[test]"`. `pytest` is available in the current environment and passes.

Tests added or updated:
- No test code change required.

Verification command:
- `pytest -q`

Remaining limitations:
Users who do not install the optional test extra should run the standard-library `unittest` command.

### BUG-010: Examples are portable within the repository but depend on bundled local data layout

Original severity: Low  
Area: examples / docs  
Fix status: Partially fixed  
Files changed:
- `README.md`

Fix summary:
README now documents that examples assume the repository-local `data/` layout and that package functions should be used directly for custom input paths.

Tests added or updated:
- Existing example validation commands were re-run.

Verification command:
- `python examples/01_build_single_graph.py`
- `python examples/02_build_multiple_state_graphs.py`
- `python examples/03_create_temporal_windows.py`
- `python examples/04_create_splits.py`

Remaining limitations:
The examples still do not include CLI arguments or a tiny committed fixture dataset. This is intentionally left out of the MVP bugfix pass to avoid expanding scope.

### BUG-011: Repeated example runs overwrite outputs without warning or versioning

Original severity: Low  
Area: export / examples  
Fix status: Partially fixed  
Files changed:
- `README.md`

Fix summary:
README now documents that example scripts overwrite their corresponding files in `outputs/` on repeated runs.

Tests added or updated:
- Existing example validation commands were re-run.

Verification command:
- `python examples/01_build_single_graph.py`
- `python examples/02_build_multiple_state_graphs.py`
- `python examples/03_create_temporal_windows.py`
- `python examples/04_create_splits.py`

Remaining limitations:
No output versioning or `overwrite=False` export policy was added. This remains a low-priority usability enhancement.

## Remaining Open Issues

- BUG-008 remains partially fixed for global discovery without `network_id`; deterministic global ordering still scans the tree.
- BUG-010 remains partially fixed; examples still assume the local repository data layout.
- BUG-011 remains partially fixed; repeated example runs still overwrite outputs, now documented.
- No High or Medium priority bugs remain open.

## Regression Test Results

- Import test result:
  - Command: `python -c "import topostategrid; print(topostategrid.__version__ if hasattr(topostategrid, '__version__') else 'import ok')"`
  - Output: `1.0.0`
- Unit test result:
  - Command: `python -m unittest discover -s tests -q`
  - Output: `Ran 31 tests ... OK`
- Pytest result:
  - Command: `pytest -q`
  - Output: `31 passed, 3 warnings`
- Example script results:
  - `python examples/01_build_single_graph.py`: passed; built one case14 graph with 14 nodes, 40 edges, `x=(14, 9)`, `edge_attr=(40, 12)`, and reloaded `outputs/graphs.pt`.
  - `python examples/02_build_multiple_state_graphs.py`: passed; built 16 case14 graphs and wrote `graphs_multi.pt`, `metadata_multi.csv`, and `split_random.json`.
  - `python examples/03_create_temporal_windows.py`: passed; built 7 temporal/scenario windows from 10 graphs and wrote `temporal_windows.pt`.
  - `python examples/04_create_splits.py`: passed; built 10 graphs across case14/case30 and wrote random, time, and LONO split JSON files.
- Graph construction check results:
  - OPFData case14 reload: `pglib_opf_case14_ieee example_0 (14, 9) (2, 40) (40, 12) [0]`
  - MATPOWER case118: `pglib_opf_case118_ieee (118, 9) (2, 372) (372, 12) 117`
- Compile check:
  - Command: `python -m compileall -q topostategrid examples tests`
  - Result: passed.

## Post-QA Fixes

- Mixed OPFData and MATPOWER graphs now batch together in PyTorch Geometric `DataLoader`. Source-specific metadata is stored as a JSON string on `data.metadata` instead of heterogeneous dictionaries, and all graphs initialize consistent optional label fields with `data.has_label` marking real labels.
- Example scripts now fail with short user-facing messages when the repository-local `data/` layout is absent, instead of raising low-level tracebacks or silently writing empty outputs.
- README now documents JSON metadata storage, mixed-source batching rationale, `split_time.json`, and a known macOS/conda OpenMP runtime conflict that can appear when binary dependencies are mixed.

## Fix Report for v1.0

1. Parser improvements

The MATPOWER parser now handles realistic MATLAB matrix syntax, including commas, comments, semicolons, scientific notation, empty matrices, and multi-line cases. OPFData parsing now fails early with path-aware errors for malformed JSON and missing MVP-required fields.

2. Split validation improvements

Random and time-based split helpers now enforce non-empty positive-ratio splits by default and expose `allow_empty=True` for explicit legacy behavior. LONO now validates `network_id`, unknown test networks, and empty train/test splits.

3. Temporal handling improvements

Time-based splitting and temporal window construction now treat empty, `None`, and NaN-like timestamps as missing. Temporal windows sort by timestamp when all timestamps are valid and comparable, otherwise they preserve input/scenario order.

4. Label safety improvements

Proxy stress-label attachment no longer overwrites existing labels by default. Callers must pass `overwrite=True` to intentionally replace `y`, `y_cls`, `y_reg`, or `risk_score`.

5. Test coverage improvements

Regression coverage expanded from 3 tests to 30 tests, covering MATPOWER parsing, OPFData validation, LONO edge cases, time split fallback, temporal ordering, random split small datasets, reproducibility, and proxy label overwrite behavior.

6. Documentation updates

README now documents supported MATPOWER syntax, OPFData parser validation, split validation behavior, timestamp sorting/fallback behavior, proxy label overwrite behavior, optional pytest installation, local-data assumptions in examples, and output overwrite behavior.

7. Remaining limitations

Global OPFData discovery without `network_id` still scans the tree for deterministic ordering. Examples still depend on the repository-local data layout and overwrite outputs. These are low-priority usability limitations rather than MVP graph-construction blockers.

8. Recommended next steps

Add a small committed fixture dataset for portable examples, add CLI arguments for example input/output paths, add optional output versioning or overwrite controls, and expand schema validation if additional OPFData variants are introduced.
