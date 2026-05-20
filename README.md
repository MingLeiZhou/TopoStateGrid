# TopoStateGrid

TopoStateGrid is a physically informed graph construction method that converts power-grid topology, component attributes, and operating-state variables into machine-learning-ready graph datasets.

The Python import name is:

```python
import topostategrid
```

TopoStateGrid focuses on reusable graph dataset construction for power-system machine learning. It does not include a GNN model, a cascading-failure simulator, `.mat` support, or heterogeneous graph construction.

## What It Builds

Each graph sample represents:

```text
G_t = (V, E, X_t, A_t, y_t)
```

where:

- `V` are bus nodes.
- `E` are physical line and transformer connections.
- `X_t` contains node features for a scenario or timestamp.
- `A_t` contains edge features for a scenario or timestamp.
- `y_t` is an optional label.

The output is a homogeneous bus-branch PyTorch Geometric `Data` object:

```text
data.x
data.edge_index
data.edge_attr
data.y
data.network_id
data.sample_id
data.timestamp
data.scenario_id
data.contingency_id
data.node_feature_names
data.edge_feature_names
data.metadata
```

Edges are stored bidirectionally so message passing can use both branch directions. Source-specific metadata is stored as a JSON string, not a Python dictionary, so graphs from different input sources can be batched together with PyG `DataLoader`.

## Installation

From PyPI:

```bash
python -m pip install TopoStateGrid
```

Optional extras:

```bash
python -m pip install "TopoStateGrid[pandapower]"
python -m pip install "TopoStateGrid[visual]"
python -m pip install "TopoStateGrid[test]"
```

From a local checkout:

```bash
git clone https://github.com/MingLeiZhou/TopoStateGrid.git
cd TopoStateGrid
python -m pip install -e ".[test,pandapower,visual]"
```

## Supported Inputs

TopoStateGrid v1.1 supports:

| Source | Status | Notes |
| --- | --- | --- |
| OPFData JSON | Supported | Uses extracted JSON samples such as `grid.nodes.bus` and solved state fields when available. |
| MATPOWER / PGLib `.m` | Supported | Parses `mpc.bus` and `mpc.branch`; static files have zero flow fields unless solved state data are supplied elsewhere. |
| pandapower `net` | Supported, optional | Requires `TopoStateGrid[pandapower]`; supports buses, lines, and transformers. |
| pandas DataFrame tables | Supported | Useful for custom bus/branch/state tables. |
| CSV tables | Supported | Thin wrapper around DataFrame support. |

The MATPOWER parser accepts comma-delimited or whitespace-delimited rows, semicolons, `%` comments, scientific notation, multi-line matrices, and explicit empty matrices such as `mpc.branch = [ ];`.

## Feature Schema

Node feature order:

```text
bus_status, bus_type, pd, qd, vm, va, vmax, vmin, normalized_demand
```

Edge feature order:

```text
component_type, r, x, b_from, b_to, rate_a, pf, qf, pt, qt, loading_ratio, outage_flag
```

Missing optional numeric values are filled with zero after validation and NaN-safe conversion.

For pandapower line data, `rate_a` uses `max_i_ka` as an approximate rating proxy when no direct MVA rating is available.

## Quick Start

Build a graph from pandas tables:

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

print(data.x.shape)
print(data.edge_index.shape)
print(data.edge_attr.shape)
```

Build from CSV:

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

Batch graphs from different sources:

```python
from torch_geometric.loader import DataLoader

loader = DataLoader([graph_a, graph_b, graph_c], batch_size=3)
batch = next(iter(loader))
```

## Operating-State Graphs

TopoStateGrid separates static topology from operating state:

- Static topology: buses, lines, transformers, branch parameters.
- Operating state: demand, voltage, angle, branch flow, loading ratio.

For one network, `edge_index` can remain fixed while `data.x` and `data.edge_attr` change across scenarios or timestamps. This supports later supervised prediction, self-supervised pretraining, temporal forecasting, and cross-topology evaluation.

## Temporal Windows

Create windows from ordered graph samples:

```python
from topostategrid import make_temporal_windows

windows = make_temporal_windows(
    graphs,
    input_window=6,
    forecast_horizon=1,
    target="y",
)
```

If all timestamps are valid and comparable, temporal utilities sort by timestamp. Otherwise they preserve input order.

## Labels

TopoStateGrid can attach temporary stress proxy labels:

```text
risk_score = max_line_loading_ratio
y_cls = 1 if max_line_loading_ratio > 1.0 else 0
y_reg = risk_score
```

This is only a proxy for graph-construction experiments. It is not a real cascading-failure target.

Proxy labels do not overwrite existing `data.y`, `data.y_cls`, `data.y_reg`, or `data.risk_score` unless `overwrite=True` is passed.

## Splits And Normalization

Implemented split strategies:

- Random split
- Time-based split
- Leave-One-Network-Out split

LONO supports cross-topology evaluation, for example training on `case14`, `case30`, and `case57`, then testing on `case118`.

`FeatureNormalizer` fits node and edge statistics only on the training split and then transforms train/validation/test graphs with the same statistics to avoid data leakage.

## Visualization

TopoStateGrid includes optional GIF/MP4 rendering for inspecting constructed graph sequences:

```python
from topostategrid import render_graph_sequence

render_graph_sequence(
    graphs,
    "outputs/topostategrid_sequence.gif",
    node_value="vm",
    edge_value="loading_ratio",
)
```

The renderer visualizes existing graph samples. It does not simulate grid dynamics.

Large pandapower example:

```bash
python examples/08_render_large_pandapower_gif.py
```

This script uses pandapower `case300`, converts it to a 300-node TopoStateGrid graph sequence, and renders a 20-second GIF.

## Example Scripts

| Script | Purpose |
| --- | --- |
| `examples/01_build_single_graph.py` | Build one local OPFData graph and save it. |
| `examples/02_build_multiple_state_graphs.py` | Build multiple OPFData scenario graphs. |
| `examples/03_create_temporal_windows.py` | Create temporal graph windows. |
| `examples/04_create_splits.py` | Create random, time-based, and LONO splits. |
| `examples/05_build_from_tables.py` | Build a graph from in-memory pandas tables. |
| `examples/06_build_from_pandapower.py` | Build a graph from a small pandapower network. |
| `examples/07_render_graph_animation.py` | Render a small graph-state sequence to GIF. |
| `examples/08_render_large_pandapower_gif.py` | Render a 20-second GIF from pandapower `case300`. |

The example scripts write generated artifacts to `outputs/`, which is intentionally ignored by git.

## Testing

Run the standard test suite:

```bash
python -m unittest discover -s tests -q
```

Or with pytest:

```bash
pytest -q
```

On systems where the default matplotlib cache directory is not writable, set a writable cache directory:

```bash
MPLCONFIGDIR=/private/tmp/topostategrid-mpl pytest -q
```

pandapower may warn that `numba` is not installed. That warning only affects pandapower runtime speed.

## Output Files

Common generated files:

```text
outputs/
├── graphs.pt
├── metadata.csv
├── graphs_multi.pt
├── metadata_multi.csv
├── split_random.json
├── split_time.json
├── split_lono.json
├── temporal_windows.pt
├── graphs_tables.pt
├── graphs_pandapower.pt
├── topostategrid_sequence.gif
├── topostategrid_case300_20s.gif
└── README_generated.md
```

Use `topostategrid.export.load_graphs` to load `.pt` graph files because it handles recent PyTorch `weights_only` defaults.

## Research Positioning

TopoStateGrid is not positioned as a wrapper around PowerGraph or pandapower.

PowerGraph can be used as a reference dataset, and pandapower can be used as a parsing or simulation tool. TopoStateGrid's main output is a reusable graph-construction pipeline for physically grounded, state-dependent, and optionally time-indexed power-grid graph datasets.

## Limitations

- Homogeneous bus-branch graph only.
- No GNN model.
- No cascading-failure simulator.
- No `.mat` support.
- No heterogeneous component graph.
- No real cascading-failure labels.
- pandapower line rating mapping may be approximate when only `max_i_ka` is available.
- MP4 rendering requires ffmpeg; GIF rendering uses matplotlib, networkx, and Pillow.
