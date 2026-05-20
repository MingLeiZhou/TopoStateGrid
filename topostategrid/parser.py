"""Input parsers for local power-system benchmark data."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from json import JSONDecodeError
from pathlib import Path
import re
from typing import Any

import numpy as np


@dataclass
class ParsedCase:
    """Normalized parser output consumed by the graph builder."""

    source_type: str
    network_id: str
    sample_id: str
    base_mva: float
    grid: dict[str, Any]
    solution: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    path: str | None = None
    timestamp: Any | None = None
    scenario_id: str | None = None
    contingency_id: str | None = None


def discover_opfdata_examples(
    root: str | Path = "data/opfdata",
    network_id: str | None = None,
    limit: int | None = None,
) -> list[Path]:
    """Return local OPFData JSON example files in stable scenario order."""

    root = Path(root)
    if network_id is not None:
        paths = _discover_opfdata_for_network(root, network_id, limit)
        return paths

    # TODO: Global deterministic discovery still needs to inspect the tree
    # before applying a limit because ordering spans networks and group dirs.
    paths = [p for p in root.glob("**/group_*/example_*.json") if p.is_file()]
    paths.sort(key=_opfdata_sort_key)
    if limit is not None:
        return paths[:limit]
    return paths


def parse_opfdata_sample(path: str | Path) -> ParsedCase:
    """Parse one extracted OPFData JSON scenario."""

    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except JSONDecodeError as exc:
        raise ValueError(f"Malformed OPFData JSON at {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"OPFData sample at {path} must be a JSON object")
    if not isinstance(raw.get("grid"), dict):
        raise ValueError(f"OPFData sample at {path} is missing required field 'grid'")
    if not isinstance(raw["grid"].get("nodes"), dict):
        raise ValueError(f"OPFData sample at {path} is missing required field 'grid.nodes'")
    if not isinstance(raw["grid"].get("edges"), dict):
        raise ValueError(f"OPFData sample at {path} is missing required field 'grid.edges'")
    if "bus" not in raw["grid"]["nodes"]:
        raise ValueError(f"OPFData sample at {path} is missing required field 'grid.nodes.bus'")
    if not raw["grid"]["nodes"]["bus"]:
        raise ValueError(f"OPFData sample at {path} has empty 'grid.nodes.bus'")

    metadata = dict(raw.get("metadata") or {})
    grid = dict(raw.get("grid") or {})
    base_mva = _first_number(grid.get("context"), default=100.0)
    sample_id = path.stem

    timestamp = (
        metadata.get("timestamp")
        or metadata.get("time")
        or metadata.get("datetime")
        or raw.get("timestamp")
    )
    scenario_id = metadata.get("scenario_id") or sample_id
    contingency_id = metadata.get("contingency_id") or metadata.get("contingency")

    return ParsedCase(
        source_type="opfdata",
        network_id=infer_network_id_from_path(path),
        sample_id=sample_id,
        base_mva=float(base_mva),
        grid=grid,
        solution=dict(raw.get("solution") or {}),
        metadata=metadata,
        path=str(path),
        timestamp=timestamp,
        scenario_id=scenario_id,
        contingency_id=contingency_id,
    )


def parse_matpower_case(path: str | Path) -> ParsedCase:
    """Parse the bus and branch tables from a MATPOWER/PGLib `.m` case file."""

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    base_match = re.search(r"mpc\.baseMVA\s*=\s*([0-9eE+\-.]+)\s*;", text)
    base_mva = float(base_match.group(1)) if base_match else 100.0

    bus = _parse_mpc_matrix(text, "bus", source=path, required=True)
    branch = _parse_mpc_matrix(text, "branch", source=path, required=True)
    if bus.shape[0] == 0:
        raise ValueError(f"MATPOWER case at {path} has empty required matrix 'mpc.bus'")

    return ParsedCase(
        source_type="matpower",
        network_id=path.stem,
        sample_id=path.stem,
        base_mva=base_mva,
        grid={"bus": bus, "branch": branch},
        solution={},
        metadata={"format": "MATPOWER/PGLib", "static_only": True},
        path=str(path),
        scenario_id=path.stem,
    )


def list_local_power_data(root: str | Path = ".") -> dict[str, Any]:
    """Summarize local data discovered by the prototype."""

    root = Path(root)
    opf_json = discover_opfdata_examples(root / "data" / "opfdata")
    networks: dict[str, int] = {}
    for path in opf_json:
        networks[infer_network_id_from_path(path)] = networks.get(infer_network_id_from_path(path), 0) + 1

    return {
        "opfdata_archives": sorted(str(p) for p in (root / "data" / "opfdata").glob("*.tar.gz")),
        "opfdata_json_count": len(opf_json),
        "opfdata_networks": networks,
        "matpower_cases": sorted(str(p) for p in (root / "data" / "pglib").glob("*.m")),
    }


def infer_network_id_from_path(path: str | Path) -> str:
    """Infer the benchmark network name from an OPFData or PGLib path."""

    path = Path(path)
    for part in path.parts:
        if part.startswith("pglib_opf_case") and not part.endswith(".tar.gz"):
            return part
    name = path.stem
    if name.startswith("example_") and len(path.parts) >= 3:
        return path.parts[-3]
    return name


def _opfdata_sort_key(path: Path) -> tuple[str, str, int, str]:
    group = path.parent.name
    return (infer_network_id_from_path(path), group, _example_number(path), str(path))


def _example_number(path: Path) -> int:
    match = re.search(r"example_(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def _first_number(value: Any, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        for item in value:
            found = _first_number(item, default=np.nan)
            if not np.isnan(found):
                return float(found)
    return float(default)


def _discover_opfdata_for_network(root: Path, network_id: str, limit: int | None) -> list[Path]:
    network_dirs: list[Path]
    if root.is_dir() and root.name == network_id:
        network_dirs = [root]
    else:
        network_dirs = sorted((p for p in root.rglob(network_id) if p.is_dir()), key=str)

    paths: list[Path] = []
    for network_dir in network_dirs:
        for group_dir in sorted(network_dir.glob("group_*"), key=lambda p: p.name):
            examples = sorted(
                (p for p in group_dir.glob("example_*.json") if p.is_file()),
                key=lambda p: (_example_number(p), str(p)),
            )
            for example in examples:
                paths.append(example)
                if limit is not None and len(paths) >= limit:
                    return paths
    return paths


def _parse_mpc_matrix(
    text: str,
    name: str,
    source: Path | None = None,
    required: bool = False,
) -> np.ndarray:
    pattern = rf"mpc\.{re.escape(name)}\s*=\s*\[(.*?)\]\s*;"
    match = re.search(pattern, text, flags=re.S)
    if not match:
        if required:
            location = f" at {source}" if source is not None else ""
            raise ValueError(f"MATPOWER case{location} is missing required matrix 'mpc.{name}'")
        return np.empty((0, 0), dtype=float)

    body = match.group(1)
    clean_lines = [line.split("%", 1)[0].replace(",", " ") for line in body.splitlines()]
    clean_body = "\n".join(clean_lines).strip()
    if not clean_body:
        return np.empty((0, 0), dtype=float)

    row_chunks = clean_body.split(";") if ";" in clean_body else clean_body.splitlines()
    rows: list[list[float]] = []
    for chunk in row_chunks:
        tokens = chunk.strip().split()
        if not tokens:
            continue
        try:
            rows.append([float(token) for token in tokens])
        except ValueError as exc:
            location = f" at {source}" if source is not None else ""
            raise ValueError(f"Malformed numeric value in MATPOWER matrix 'mpc.{name}'{location}: {chunk!r}") from exc

    if not rows:
        return np.empty((0, 0), dtype=float)
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        location = f" at {source}" if source is not None else ""
        raise ValueError(f"Malformed MATPOWER matrix 'mpc.{name}'{location}: inconsistent row lengths")
    return np.asarray(rows, dtype=float)
