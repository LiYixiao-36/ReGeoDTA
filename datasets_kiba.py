"""KIBA dataset loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from datasets_common import (
    DTADataset,
    DatasetBundle,
    load_drug_graphs,
    load_fold_indices,
    load_ordered_json,
    load_protein_features,
    make_sample,
    shuffle_affinities,
    split_folds,
)

PROTEIN_LENGTH = 1000
AUPR_THRESHOLD = 12.1
AFFINITY_FILE = "kiba_binding_affinity_v2.txt"


def build_datasets(
    data_root: str | Path,
    pretrained_root: str | Path,
    fold: int = 0,
    shuffle_affinity: bool = False,
    seed: int = 2027,
) -> DatasetBundle:
    data_root = Path(data_root)
    pretrained_root = Path(pretrained_root)

    drugs = load_ordered_json(data_root / "ligands_iso.txt")
    proteins = load_ordered_json(data_root / "proteins.txt")
    affinities = pd.read_csv(data_root / AFFINITY_FILE, sep=r"\s+", header=None, encoding="latin1")

    protein_records = load_protein_features(
        proteins.keys(), pretrained_root, PROTEIN_LENGTH, "Loading KIBA proteins"
    )
    drug_graphs = load_drug_graphs(
        drugs.keys(), pretrained_root, "drug_node_features", normalize=True, description="Loading KIBA drugs"
    )

    samples = []
    for drug_index, drug_id in enumerate(drugs.keys()):
        for protein_index, protein_id in enumerate(proteins.keys()):
            affinity = affinities.iloc[drug_index, protein_index]
            if not pd.isna(affinity):
                samples.append(make_sample(drug_graphs[drug_id], protein_records[protein_id], affinity))

    if shuffle_affinity:
        samples = shuffle_affinities(samples, seed)

    train_folds, test_indices = load_fold_indices(data_root)
    train_samples, val_samples, test_samples = split_folds(samples, train_folds, test_indices, fold)
    return DatasetBundle(
        train=DTADataset(train_samples),
        val=DTADataset(val_samples),
        test=DTADataset(test_samples),
        aupr_threshold=AUPR_THRESHOLD,
        protein_length=PROTEIN_LENGTH,
    )
