from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    data_dir: str
    num_train_datasets: int
    max_samples_0: int
    max_samples_1: int


DATASETS: Dict[str, DatasetSpec] = {
    "amlworld": DatasetSpec(
        key="amlworld",
        display_name="AMLWorld",
        data_dir=str(ROOT / "Data" / "AMLWorld" / "Data"),
        num_train_datasets=6,
        max_samples_0=9800,
        max_samples_1=150,
    ),
}

DATASET_ALIASES: Dict[str, str] = {}

ONLY_TO_DATASETS = {
    "all": ["amlworld"],
    "amlworld": ["amlworld"],
}


def resolve_dataset_keys(selection: Optional[str]) -> List[str]:
    if selection is None:
        return ["amlworld"]
    key = selection.lower()
    if key in ONLY_TO_DATASETS:
        return ONLY_TO_DATASETS[key]
    if key in DATASET_ALIASES:
        return [DATASET_ALIASES[key]]
    if key in DATASETS:
        return [key]
    raise ValueError(
        f"Unknown dataset selection {selection!r}; "
        f"expected one of {sorted(ONLY_TO_DATASETS)} or a dataset key."
    )


def get_dataset_spec(key: str) -> DatasetSpec:
    resolved = DATASET_ALIASES.get(key.lower(), key.lower())
    if resolved not in DATASETS:
        raise KeyError(f"Unknown dataset key: {key!r}")
    return DATASETS[resolved]
