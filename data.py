"""Dataset registry."""

from __future__ import annotations

from pathlib import Path

from datasets_bindingdb import build_datasets as build_bindingdb
from datasets_davis import build_datasets as build_davis
from datasets_kiba import build_datasets as build_kiba
from datasets_common import DatasetBundle

DATASET_BUILDERS = {
    "davis": build_davis,
    "kiba": build_kiba,
    "bindingdb": build_bindingdb,
}
DEFAULT_FOLDS = {"davis": 0, "kiba": 0, "bindingdb": 1}


def resolve_fold(dataset: str, fold: int | None) -> int:
    return DEFAULT_FOLDS[dataset] if fold is None else fold


def build_dataset_bundle(
    dataset: str,
    data_root: str | Path,
    pretrained_root: str | Path,
    fold: int,
    shuffle_affinity: bool,
    seed: int,
) -> DatasetBundle:
    try:
        builder = DATASET_BUILDERS[dataset]
    except KeyError as error:
        raise ValueError(f"Unsupported dataset: {dataset}") from error

    return builder(
        data_root=data_root,
        pretrained_root=pretrained_root,
        fold=fold,
        shuffle_affinity=shuffle_affinity,
        seed=seed,
    )
