"""Dataset metadata loaders for pretraining-data generation."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DATASETS = ("davis", "kiba", "bindingdb")


@dataclass(frozen=True)
class DrugRecord:
    drug_id: str
    smiles: str


@dataclass(frozen=True)
class ProteinRecord:
    protein_id: str
    sequence: str


@dataclass(frozen=True)
class DatasetSpec:
    protein_length: int


DATASET_SPECS = {
    "davis": DatasetSpec(protein_length=1200),
    "kiba": DatasetSpec(protein_length=1000),
    "bindingdb": DatasetSpec(protein_length=1000),
}


def get_dataset_spec(dataset: str) -> DatasetSpec:
    try:
        return DATASET_SPECS[dataset]
    except KeyError as error:
        raise ValueError(f"Unsupported dataset: {dataset!r}. Expected one of {DATASETS}.") from error


def _load_ordered_json(path: Path) -> OrderedDict:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file, object_pairs_hook=OrderedDict)


def _validate_text(value, field: str, identifier: str) -> str:
    if pd.isna(value):
        raise ValueError(f"Missing {field} for {identifier}.")
    text = str(value).strip()
    if not text:
        raise ValueError(f"Empty {field} for {identifier}.")
    return text


def _load_json_dataset(data_root: Path) -> tuple[list[DrugRecord], list[ProteinRecord]]:
    drugs = _load_ordered_json(data_root / "ligands_iso.txt")
    proteins = _load_ordered_json(data_root / "proteins.txt")

    drug_records = [
        DrugRecord(drug_id=str(drug_id), smiles=_validate_text(smiles, "SMILES", str(drug_id)))
        for drug_id, smiles in drugs.items()
    ]
    protein_records = [
        ProteinRecord(
            protein_id=str(protein_id),
            sequence=_validate_text(sequence, "protein sequence", str(protein_id)),
        )
        for protein_id, sequence in proteins.items()
    ]
    return drug_records, protein_records


def _merge_unique_mapping(
    frames: Iterable[pd.DataFrame],
    id_column: str,
    value_column: str,
    entity_name: str,
) -> OrderedDict[str, str]:
    mapping: OrderedDict[str, str] = OrderedDict()
    for frame in frames:
        for row in frame[[id_column, value_column]].itertuples(index=False, name=None):
            identifier = _validate_text(row[0], f"{entity_name} ID", entity_name)
            value = _validate_text(row[1], value_column, identifier)
            previous = mapping.get(identifier)
            if previous is None:
                mapping[identifier] = value
            elif previous != value:
                raise ValueError(
                    f"Inconsistent {entity_name} mapping for {identifier!r}: "
                    f"the same ID is associated with multiple {value_column} values."
                )
    return mapping


def _load_bindingdb(data_root: Path) -> tuple[list[DrugRecord], list[ProteinRecord]]:
    required_columns = {"d", "p", "compound_iso_smiles", "target_sequence"}
    frames = []
    for filename in ("bindingdb_train.csv", "bindingdb_test.csv"):
        path = data_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required BindingDB file not found: {path}")
        frame = pd.read_csv(path)
        missing = required_columns.difference(frame.columns)
        if missing:
            raise ValueError(f"{filename} is missing columns: {sorted(missing)}")
        frames.append(frame)

    drug_mapping = _merge_unique_mapping(frames, "d", "compound_iso_smiles", "drug")
    protein_mapping = _merge_unique_mapping(frames, "p", "target_sequence", "protein")

    drug_records = [DrugRecord(drug_id=drug_id, smiles=smiles) for drug_id, smiles in drug_mapping.items()]
    protein_records = [
        ProteinRecord(protein_id=protein_id, sequence=sequence)
        for protein_id, sequence in protein_mapping.items()
    ]
    return drug_records, protein_records


def load_dataset_records(
    dataset: str,
    data_root: str | Path,
) -> tuple[list[DrugRecord], list[ProteinRecord]]:
    """Load the unique drug and protein entities required by a dataset."""
    data_root = Path(data_root)
    if dataset in {"davis", "kiba"}:
        return _load_json_dataset(data_root)
    if dataset == "bindingdb":
        return _load_bindingdb(data_root)
    raise ValueError(f"Unsupported dataset: {dataset!r}. Expected one of {DATASETS}.")


def load_drug_records(dataset: str, data_root: str | Path) -> list[DrugRecord]:
    drugs, _ = load_dataset_records(dataset, data_root)
    return drugs


def load_protein_records(dataset: str, data_root: str | Path) -> list[ProteinRecord]:
    _, proteins = load_dataset_records(dataset, data_root)
    return proteins
