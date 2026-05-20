# TopoStateGrid Prototype Report

TopoStateGrid is a physically informed graph construction method that converts power-grid topology, component attributes, and operating-state variables into machine-learning-ready graph datasets.

## Local Data Found

The repository contains local power-system data under `data/`:

- OPFData archives:
  - `data/opfdata/dataset_release_1__pglib_opf_case14_ieee_0.tar.gz`
  - `data/opfdata/dataset_release_1__pglib_opf_case30_ieee_0.tar.gz`
  - `data/opfdata/dataset_release_1__pglib_opf_case57_ieee_0.tar.gz`
  - `data/opfdata/dataset_release_1__pglib_opf_case118_ieee_0.tar.gz`
- Extracted OPFData JSON scenarios:
  - `pglib_opf_case14_ieee`: 15000 JSON examples
  - `pglib_opf_case30_ieee`: 8587 JSON examples
- Static PGLib/MATPOWER case:
  - `data/pglib/pglib_opf_case118_ieee.m`

The Python environment has `torch`, `torch_geometric`, `numpy`, and `pandas`. `pandapower` was not installed, so the prototype uses OPFData JSON first and a small MATPOWER parser as a static fallback.

## Successfully Parsed Format

The primary working format is extracted OPFData JSON:

```text
data/opfdata/.../pglib_opf_case14_ieee/group_0/example_*.json
```

Each JSON file contains:

- `grid`: static and scenario input data
- `solution`: solved bus and branch state
- `metadata`: objective value and any optional metadata

The first runnable example builds a graph from `pglib_opf_case14_ieee`.

`examples/01_build_single_graph.py` generated and reloaded `outputs/graphs.pt` successfully. The generated case14 graph has 14 bus nodes, 40 bidirectional edges, 9 node features, and 12 edge features.

## Graph Construction

The prototype builds a homogeneous PyTorch Geometric `Data` object:

- Nodes are buses.
- Edges are AC lines and transformers.
- Edges are bidirectional.
- Load nodes are aggregated onto their connected bus through OPFData `load_link` edges.
- OPFData branch solution fields are merged into edge attributes.
- Metadata is preserved on the `Data` object.

Expected fields are populated:

- `data.x`
- `data.edge_index`
- `data.edge_attr`
- `data.y` when proxy labels are attached
- `data.network_id`
- `data.sample_id`
- `data.timestamp`
- `data.scenario_id`
- `data.contingency_id`

## Node Features Used

The prototype uses:

```text
bus_status, bus_type, pd, qd, vm, va, vmax, vmin, normalized_demand
```

For OPFData:

- `bus_status`, `bus_type`, `vmin`, and `vmax` come from bus node features.
- `pd` and `qd` come from load nodes aggregated to buses.
- `vm` and `va` come from solved bus states.
- `normalized_demand` is `pd / max(abs(pd))` within the graph.

Missing or unavailable values are filled with zero using NaN-safe conversion.

## Edge Features Used

The prototype uses:

```text
component_type, r, x, b_from, b_to, rate_a, pf, qf, pt, qt, loading_ratio, outage_flag
```

For OPFData:

- AC line and transformer static features provide branch parameters and limits.
- Solution edge features provide directional active and reactive flows.
- `loading_ratio` is computed as the larger apparent power magnitude at either branch end divided by `rate_a`.
- `outage_flag` is currently zero for OPFData samples because no contingency/outage field was found in the inspected JSON.

For static MATPOWER/PGLib:

- Bus and branch tables are parsed.
- Solved flow fields are unavailable and set to zero.
- `outage_flag` is inferred from branch status.

## Timestamps and Multiple Operating States

Multiple operating states are available through OPFData examples. The local extracted data has many scenario samples for `case14` and `case30`.

No real timestamp field was found in the inspected OPFData metadata. The package therefore treats these as scenario-indexed graph samples, not verified chronological time series.

## Temporal Windows

`make_temporal_windows` is implemented and demonstrated in `examples/03_create_temporal_windows.py`.

Because the inspected local OPFData samples do not include timestamps, the demonstration uses sorted scenario order. This verifies the window data structure but should not be interpreted as a real temporal forecasting benchmark until timestamped data is supplied.

## Labels Attached

Temporary stress proxy labels are implemented:

```text
risk_score = max_line_loading_ratio
y_cls = 1 if risk_score > 1.0 else 0
y_reg = risk_score
```

These labels are useful for testing graph construction, saving, splitting, and normalization. They are not real cascading-failure labels.

## Splits and Normalization

Implemented split utilities:

- `create_random_split`
- `create_time_based_split`
- `create_lono_split`

`FeatureNormalizer` fits only on training graphs and transforms node and edge features afterward, which avoids validation/test leakage.

## Problems and Missing Data

- `pandapower` is not installed locally, so pandapower network support was deferred.
- The inspected OPFData metadata did not include real timestamps.
- Contingency identifiers and explicit outage flags were not present in the inspected JSON.
- OPFData feature names are not embedded in the files, so the parser uses documented positional assumptions based on the observed schema.
- Static PGLib/MATPOWER files do not include solved operating-state flows unless another solver/parser is added later.

## Recommended Next Steps

1. Add formal schema documentation for each supported OPFData feature position.
2. Add pandapower input support if pandapower becomes available in the environment.
3. Extract or generate solved operating states for static PGLib/MATPOWER cases.
4. Add explicit contingency metadata support when local contingency samples are available.
5. Add real target labels from cascading, overload, stability, or OPF feasibility studies.
6. Add dataset classes compatible with `torch_geometric.data.InMemoryDataset`.
7. Use Leave-One-Network-Out experiments for cross-topology evaluation.
8. Add self-supervised pretraining tasks such as masked bus demand, masked voltage, masked branch flow, or topology-aware contrastive pairs.

## Research Positioning

TopoStateGrid is not simply a wrapper around PowerGraph or pandapower. It focuses on physically grounded, state-dependent, and optionally time-indexed graph dataset construction for power-system machine learning. PowerGraph can be used as a reference dataset, and pandapower can be used as a parsing or simulation tool, but the main output is a reusable graph-construction pipeline.
