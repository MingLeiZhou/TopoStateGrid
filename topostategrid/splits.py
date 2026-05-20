"""Dataset split strategies."""

from __future__ import annotations

import math
import random
from typing import Any, Sequence


def create_random_split(
    dataset: Sequence[Any],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 0,
    allow_empty: bool = False,
) -> dict[str, list[int]]:
    """Create index-based random train/val/test splits.

    By default, each split with a positive ratio must receive at least one item.
    Pass ``allow_empty=True`` to keep the older truncation behavior for tiny
    datasets.
    """

    _validate_ratios(train_ratio, val_ratio, test_ratio)
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    return _split_ordered_indices(indices, train_ratio, val_ratio, test_ratio, allow_empty=allow_empty)


def create_time_based_split(
    dataset: Sequence[Any],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    allow_empty: bool = False,
) -> dict[str, list[int]]:
    """Create ordered splits by timestamp when available, otherwise by input order.

    ``None``, empty strings, and NaN-like timestamp values are treated as
    missing. Timestamp sorting is used only when every graph has a valid and
    mutually comparable timestamp.
    """

    _validate_ratios(train_ratio, val_ratio, test_ratio)
    indexed = list(enumerate(dataset))
    timestamps = [getattr(graph, "timestamp", None) for _, graph in indexed]
    if timestamps and all(_is_valid_timestamp(timestamp) for timestamp in timestamps):
        try:
            indexed.sort(key=lambda item: getattr(item[1], "timestamp"))
        except TypeError:
            pass
    indices = [idx for idx, _ in indexed]
    return _split_ordered_indices(indices, train_ratio, val_ratio, test_ratio, allow_empty=allow_empty)


def create_lono_split(
    dataset: Sequence[Any],
    test_network: str,
    val_networks: list[str] | None = None,
    allow_empty: bool = False,
) -> dict[str, list[int]]:
    """Create a Leave-One-Network-Out split by `data.network_id`.

    The default behavior fails loudly when the requested test network is absent
    or when the train/test split would be empty.
    """

    val_networks = val_networks or []
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []

    for idx, graph in enumerate(dataset):
        network_id = getattr(graph, "network_id", None)
        if not _is_valid_network_id(network_id):
            raise ValueError(f"Graph at index {idx} is missing a valid network_id")
        if network_id == test_network:
            test.append(idx)
        elif network_id in val_networks:
            val.append(idx)
        else:
            train.append(idx)

    if not allow_empty:
        network_ids = {getattr(graph, "network_id", None) for graph in dataset}
        if test_network not in network_ids:
            raise ValueError(f"test_network {test_network!r} is not present in the dataset")
        if not train:
            raise ValueError("Leave-One-Network-Out split produced an empty train split")
        if not test:
            raise ValueError("Leave-One-Network-Out split produced an empty test split")
    return {"train": train, "val": val, "test": test}


def _split_ordered_indices(
    indices: list[int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    allow_empty: bool = False,
) -> dict[str, list[int]]:
    n_items = len(indices)
    if allow_empty:
        train_count = int(n_items * train_ratio)
        val_count = int(n_items * val_ratio)
        test_count = n_items - train_count - val_count
    else:
        train_count, val_count, test_count = _non_empty_counts(n_items, train_ratio, val_ratio, test_ratio)

    train_end = train_count
    val_end = train_end + val_count
    return {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end : val_end + test_count],
    }


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if train_ratio < 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("Split ratios must be non-negative")
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")


def _non_empty_counts(
    n_items: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    ratios = [train_ratio, val_ratio, test_ratio]
    positive = [ratio > 0 for ratio in ratios]
    min_counts = [1 if is_positive else 0 for is_positive in positive]
    required = sum(min_counts)
    if n_items < required:
        raise ValueError(
            "Dataset is too small for non-empty train/val/test splits with the requested positive ratios; "
            "pass allow_empty=True to permit empty splits."
        )

    remaining = n_items - required
    counts = min_counts[:]
    if remaining == 0:
        return counts[0], counts[1], counts[2]

    positive_ratio_total = sum(ratio for ratio in ratios if ratio > 0)
    extras = [
        (remaining * ratio / positive_ratio_total) if ratio > 0 else 0.0
        for ratio in ratios
    ]
    floors = [math.floor(extra) for extra in extras]
    counts = [count + extra for count, extra in zip(counts, floors)]
    leftover = remaining - sum(floors)
    order = sorted(range(3), key=lambda idx: (extras[idx] - floors[idx], ratios[idx]), reverse=True)
    for idx in order[:leftover]:
        counts[idx] += 1
    return counts[0], counts[1], counts[2]


def _is_valid_timestamp(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "nan", "none", "null", "nat"}
    try:
        return not math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def _is_valid_network_id(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""
