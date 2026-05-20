"""TopoStateGrid: physically informed graph construction for power-grid ML."""

from .builder import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    build_graph,
    build_graph_from_matpower,
    build_graph_from_opfdata_json,
    build_graphs_from_opfdata,
)
from .export import export_dataset, load_graphs, save_graphs, save_split_json, write_metadata_csv
from .labels import attach_labels, attach_stress_proxy_labels
from .normalizer import FeatureNormalizer
from .parser import (
    ParsedCase,
    discover_opfdata_examples,
    list_local_power_data,
    parse_matpower_case,
    parse_opfdata_sample,
)
from .splits import create_lono_split, create_random_split, create_time_based_split
from .temporal import make_temporal_windows

__version__ = "1.0.0"

__all__ = [
    "EDGE_FEATURE_NAMES",
    "NODE_FEATURE_NAMES",
    "FeatureNormalizer",
    "ParsedCase",
    "attach_labels",
    "attach_stress_proxy_labels",
    "build_graph",
    "build_graph_from_matpower",
    "build_graph_from_opfdata_json",
    "build_graphs_from_opfdata",
    "create_lono_split",
    "create_random_split",
    "create_time_based_split",
    "discover_opfdata_examples",
    "export_dataset",
    "list_local_power_data",
    "load_graphs",
    "make_temporal_windows",
    "parse_matpower_case",
    "parse_opfdata_sample",
    "save_graphs",
    "save_split_json",
    "write_metadata_csv",
]
