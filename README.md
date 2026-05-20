# TopoStateGrid

TopoStateGrid is a physically informed graph construction method that converts power-grid topology, component attributes, and operating-state variables into machine-learning-ready graph datasets.

The Python package import name is `topostategrid`.

## Scope

TopoStateGrid focuses on physically grounded, state-dependent, and optionally time-indexed graph dataset construction for power-system machine learning. PowerGraph can be used as a reference dataset, and pandapower can be used as a parsing or simulation tool, but the main output is a reusable graph-construction pipeline.

This prototype does not build a GNN model, does not implement a cascading-failure simulator, and does not claim to be the first power-grid graph dataset tool.

## Graph Definition

Each graph sample represents:

```text
G_t = (V, E, X_t, A_t, y_t)
```

where:

- `V` are bus nodes.
- `E` are physical line and transformer branches.
- `X_t` contains node features for scenario or time `t`.
- `A_t` contains edge features for scenario or time `t`.
- `y_t` is an optional label.

For the MVP, TopoStateGrid builds a homogeneous bus-branch graph and exports a PyTorch Geometric `Data` object with:

- `data.x`
- `data.edge_index`
- `data.edge_attr`
- `data.y`, optional label value; unlabeled graphs use `data.has_label=False` with placeholder label tensors for PyG batching
- `data.network_id`
- `data.sample_id`
- `data.timestamp`, optional
- `data.scenario_id`, optional
- `data.contingency_id`, optional
- `data.metadata`, a JSON string for source-specific metadata

Edges are stored bidirectionally so message passing can use both branch directions.

Source-specific metadata is stored as a JSON string rather than a Python dict so OPFData and MATPOWER graphs remain batchable together with PyTorch Geometric `DataLoader`.

## Supported Inputs

Supported input sources in v1.1:

- OPFData JSON
- MATPOWER / PGLib `.m`
- pandapower `net` object
- pandas DataFrame tables
- CSV tables

Current working input paths:

- Extracted OPFData JSON samples under `data/opfdata/**/group_*/example_*.json`
- Static MATPOWER/PGLib `.m` files with `mpc.bus` and `mpc.branch` tables

The MATPOWER parser accepts common matrix syntax: comma-delimited or whitespace-delimited rows, semicolons, `%` comments, scientific notation, multi-line matrices, and explicit empty matrices such as `mpc.branch = [ ];`. Missing required `mpc.bus` or `mpc.branch` declarations raise `ValueError`; an explicitly present empty `mpc.branch` is allowed for isolated-bus fixtures.

The OPFData parser validates that JSON is well-formed and that `grid.nodes.bus` is present and non-empty. Malformed JSON and missing required fields raise `ValueError` with the source path included.

The local environment used for this prototype contains extracted OPFData samples for `pglib_opf_case14_ieee` and `pglib_opf_case30_ieee`, plus a static PGLib MATPOWER case for `pglib_opf_case118_ieee`.

pandapower support is optional. Install it with:

```bash
python -m pip install -e ".[pandapower]"
```

The pandapower converter supports bus nodes and line/transformer branch edges. For lines, `rate_a` uses `max_i_ka` as an approximate rating proxy when no direct MVA rating is available. The graph remains homogeneous bus-branch only.

Graph rendering support is also optional. Install it with:

```bash
python -m pip install -e ".[visual]"
```

The renderer writes GIF or MP4 files from existing graph samples for inspection. It does not simulate grid dynamics.

## Features

Node features:

```text
bus_status, bus_type, pd, qd, vm, va, vmax, vmin, normalized_demand
```

For OPFData, `pd` and `qd` are aggregated from load nodes through `load_link` edges. `vm` and `va` are read from solved bus states when available. Missing values are filled with zero after NaN-safe conversion.

Edge features:

```text
component_type, r, x, b_from, b_to, rate_a, pf, qf, pt, qt, loading_ratio, outage_flag
```

`component_type` is `0` for AC lines and `1` for transformers. OPFData solution flows are used when present. Static MATPOWER/PGLib cases include physical branch attributes, but solved flow fields are set to zero unless supplied by another source.

## Static Topology vs Operating State

Topology and component attributes come from buses, lines, transformers, and branch parameters. Operating state comes from scenario-dependent demand, solved bus voltage, solved branch flow, and derived loading ratio.

For the same network, `edge_index` can remain fixed across scenarios while `data.x` and `data.edge_attr` vary by sample. This supports later supervised GNNs, contrastive or masked-feature self-supervision, and temporal forecasting when ordered timestamps are available.

## Labels

`topostategrid.labels.attach_stress_proxy_labels` can attach temporary proxy labels:

```text
risk_score = max_line_loading_ratio
y_cls = 1 if max_line_loading_ratio > 1.0 else 0
y_reg = risk_score
```

This is only a stress proxy for graph-construction experiments. It is not a real cascading-failure target.

Proxy label attachment is in-place and will not overwrite existing `data.y`, `data.y_cls`, `data.y_reg`, or `data.risk_score` by default. Pass `overwrite=True` only when replacing existing labels is intentional.

## Splits

Implemented split strategies:

- Random split
- Time-based split when timestamps exist, otherwise input order
- Leave-One-Network-Out split with `create_lono_split(dataset, test_network="...")`

LONO is useful for cross-topology evaluation, for example training on `case14` and testing on `case30` or `case118`.

Random and time-based splits require each positive-ratio split to receive at least one graph by default. Tiny datasets raise `ValueError`; pass `allow_empty=True` to permit empty splits. LONO raises `ValueError` when the test network is absent, when graph objects lack `network_id`, or when train/test would be empty.

Time-based splitting treats `None`, empty strings, and NaN-like timestamps as missing. It sorts only when all timestamps are valid and comparable; otherwise it falls back to input order. Temporal windows use the same timestamp rule by default through `make_temporal_windows(..., sort_by_timestamp=True)`.

## Normalization

`FeatureNormalizer` fits node and edge feature statistics only on the training split, then transforms train/validation/test graphs using the same statistics. This avoids data leakage from validation or test graphs.

## Usage

Build one graph:

```bash
python examples/01_build_single_graph.py
```

Build multiple scenario graphs:

```bash
python examples/02_build_multiple_state_graphs.py
```

Create temporal windows over ordered samples:

```bash
python examples/03_create_temporal_windows.py
```

Create random, ordered, and LONO splits:

```bash
python examples/04_create_splits.py
```

Render a small graph-state sequence to GIF:

```bash
python examples/07_render_graph_animation.py
```

Render a 20-second GIF from pandapower's 300-bus benchmark:

```bash
python examples/08_render_large_pandapower_gif.py
```

Run tests:

```bash
python -m unittest discover -s tests -q
```

The tests are also compatible with `pytest` if it is installed.

Install optional test tooling with:

```bash
python -m pip install -e ".[test]"
```

On systems where the default matplotlib cache directory is not writable, use a writable cache directory for tests or rendering:

```bash
MPLCONFIGDIR=/private/tmp/topostategrid-mpl python -m unittest discover -s tests -q
```

On some macOS/conda environments, importing `torch`, `torch_geometric`, and numeric packages in one probe may expose an OpenMP runtime conflict from binary dependencies. Prefer a clean, consistent conda or virtualenv environment and avoid mixing package channels where possible.

pandapower may warn that `numba` is not installed. That warning only affects pandapower runtime speed; install `numba` separately if pandapower performance matters.

## Output Files

The examples write to `outputs/`, including:

- `graphs.pt`
- `metadata.csv`
- `graphs_multi.pt`
- `metadata_multi.csv`
- `split_random.json`
- `split_time.json`
- `split_lono.json`
- `temporal_windows.pt`
- `graphs_tables.pt`
- `graphs_pandapower.pt`, when pandapower is installed
- `topostategrid_sequence.gif`, when visualization dependencies are installed
- `topostategrid_case300_20s.gif`, when pandapower and visualization dependencies are installed
- `README_generated.md`

Use `topostategrid.export.load_graphs` to load `.pt` files because it handles recent PyTorch `weights_only` defaults.

The example scripts assume the repository-local `data/` layout used by this prototype and overwrite their corresponding files in `outputs/` on repeated runs. Use the package functions directly when you need custom input paths or run-specific output directories.

## v1.1 Table And pandapower Examples

Build from pandas DataFrames:

```python
import pandas as pd
from topostategrid import build_graph_from_tables

bus_df = pd.DataFrame({
    "bus_id": [1, 2, 3],
    "bus_type": [3, 1, 1],
    "pd": [0.0, 1.5, 0.8],
    "qd": [0.0, 0.4, 0.2],
})
branch_df = pd.DataFrame({
    "from_bus": [1, 2],
    "to_bus": [2, 3],
    "r": [0.01, 0.02],
    "x": [0.05, 0.06],
})

data = build_graph_from_tables(
    bus_df,
    branch_df,
    network_id="toy_3bus",
    sample_id="sample_0",
)
```

Build from CSV tables:

```python
from topostategrid import build_graph_from_csv_tables

data = build_graph_from_csv_tables(
    "bus.csv",
    "branch.csv",
    network_id="toy_3bus",
)
```

Build from pandapower:

```python
import pandapower as pp
from topostategrid import build_graph_from_pandapower

net = pp.create_empty_network()
b1 = pp.create_bus(net, vn_kv=110)
b2 = pp.create_bus(net, vn_kv=110)
b3 = pp.create_bus(net, vn_kv=110)
pp.create_ext_grid(net, b1)
pp.create_load(net, b2, p_mw=10.0, q_mvar=3.0)
pp.create_line_from_parameters(net, b1, b2, 1.0, 0.1, 0.2, 0.0, 0.4)
pp.create_line_from_parameters(net, b2, b3, 1.0, 0.1, 0.2, 0.0, 0.4)
pp.runpp(net)

data = build_graph_from_pandapower(net, network_id="pandapower_3bus")
```

Render constructed graph samples to GIF:

```python
from topostategrid import render_graph_sequence

render_graph_sequence(
    [data],
    "outputs/topostategrid_sequence.gif",
    node_value="vm",
    edge_value="loading_ratio",
)
```

TopoStateGrid v1.1 still does not include a GNN model, cascading-failure simulator, `.mat` support, or heterogeneous graph construction.
