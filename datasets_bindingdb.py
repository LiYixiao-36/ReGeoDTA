"""BindingDB dataset loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from datasets_common import DTADataset, DatasetBundle, load_drug_graphs, load_protein_features, make_sample

PROTEIN_LENGTH = 1000
AUPR_THRESHOLD = 7.0
TRAIN_FILE = "bindingdb_train.csv"
TEST_FILE = "bindingdb_test.csv"


def _numeric_id(value: str, prefix: str) -> int:
    value = str(value)
    if not value.startswith(prefix):
        raise ValueError(f"Expected identifier beginning with '{prefix}', got {value!r}.")
    return int(value[len(prefix):])


def _build_samples(frame: pd.DataFrame, drug_graphs, protein_records):
    required_columns = {"d", "p", "affinity"}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"BindingDB CSV is missing columns: {sorted(missing)}")

    samples = []
    for row in frame.itertuples(index=False):
        drug_id = str(getattr(row, "d"))
        protein_id = str(getattr(row, "p"))
        samples.append(make_sample(drug_graphs[drug_id], protein_records[protein_id], getattr(row, "affinity")))
    return samples


def build_datasets(
    data_root: str | Path,
    pretrained_root: str | Path,
    fold: int = 1,
    shuffle_affinity: bool = False,
    seed: int = 2027,
) -> DatasetBundle:
    if shuffle_affinity:
        raise ValueError("--shuffle-affinity is not supported for BindingDB.")
    del seed
    data_root = Path(data_root)
    pretrained_root = Path(pretrained_root)

    train_frame = pd.read_csv(data_root / TRAIN_FILE)
    test_frame = pd.read_csv(data_root / TEST_FILE)
    all_frame = pd.concat([train_frame, test_frame], ignore_index=True)

    protein_ids = sorted({str(value) for value in all_frame["p"]}, key=lambda value: _numeric_id(value, "p"))
    drug_ids = sorted({str(value) for value in all_frame["d"]}, key=lambda value: _numeric_id(value, "d"))

    protein_records = load_protein_features(
        protein_ids, pretrained_root, PROTEIN_LENGTH, "Loading BindingDB proteins"
    )
    drug_graphs = load_drug_graphs(
        drug_ids,
        pretrained_root,
        "drug_node_features",
        normalize=True,
        description="Loading BindingDB drugs",
    )

    train_pool = _build_samples(train_frame, drug_graphs, protein_records)
    test_samples = _build_samples(test_frame, drug_graphs, protein_records)
    if fold < 0 or fold > 4:
        raise ValueError(f"fold must be in [0, 4], got {fold}.")

    fold_size = len(train_pool) // 5
    folds = [train_pool[index * fold_size:(index + 1) * fold_size] for index in range(4)]
    folds.append(train_pool[4 * fold_size:])
    val_samples = folds[fold]
    train_samples = [sample for fold_id, values in enumerate(folds) if fold_id != fold for sample in values]

    return DatasetBundle(
        train=DTADataset(train_samples),
        val=DTADataset(val_samples),
        test=DTADataset(test_samples),
        aupr_threshold=AUPR_THRESHOLD,
        protein_length=PROTEIN_LENGTH,
    )
