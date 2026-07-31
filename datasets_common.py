"""Shared dataset utilities for Davis, KIBA, and BindingDB."""

from __future__ import annotations

import ast
import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

CONV_KERNEL = 8
CONV_STRIDE = 4
CONV_PADDING = 2


@dataclass(frozen=True)
class DatasetBundle:
    train: Dataset
    val: Dataset
    test: Dataset
    aupr_threshold: float
    protein_length: int


class DTADataset(Dataset):
    """Dataset returning drug graph, protein tensors, masks, and affinity."""

    def __init__(
        self,
        samples: Sequence[tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]],
    ) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        drug, protein, key_mask, position, query_mask, affinity = self.samples[index]
        return drug, protein, key_mask, position, query_mask, torch.tensor(affinity, dtype=torch.float32)


def load_tensor(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Required tensor file not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_ordered_json(path: Path) -> OrderedDict:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file, object_pairs_hook=OrderedDict)


def find_fold_file(data_root: Path, filename: str) -> Path:
    candidates = [data_root / "folds" / filename, data_root / filename]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find {filename} under {data_root} or {data_root / 'folds'}.")


def load_fold_indices(data_root: Path) -> tuple[list[list[int]], list[int]]:
    train_path = find_fold_file(data_root, "train_fold_setting1.txt")
    test_path = find_fold_file(data_root, "test_fold_setting1.txt")
    with train_path.open("r", encoding="utf-8") as file:
        train_folds = ast.literal_eval(file.read())
    with test_path.open("r", encoding="utf-8") as file:
        test_indices = ast.literal_eval(file.read())
    return train_folds, test_indices


def extract_ca_coordinates(pdb_path: Path) -> torch.Tensor:
    if not pdb_path.is_file():
        raise FileNotFoundError(f"Required PDB file not found: {pdb_path}")

    coordinates = []
    with pdb_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("ATOM") and " CA " in line:
                coordinates.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))

    if not coordinates:
        raise ValueError(f"No C-alpha coordinates were found in {pdb_path}.")
    return torch.tensor(coordinates, dtype=torch.float32)


def process_pos_and_mask(coords_3d: torch.Tensor, target_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build downsampled key masks and relative distance-direction features."""
    real_len = min(coords_3d.size(0), target_len)
    real_coords = coords_3d[:real_len]
    key_len = (target_len + 2 * CONV_PADDING - CONV_KERNEL) // CONV_STRIDE + 1

    query_mask = torch.zeros(target_len, dtype=torch.bool)
    query_mask[:real_len] = True
    key_mask = torch.zeros(key_len, dtype=torch.bool)
    key_coords = torch.zeros(key_len, 3, dtype=coords_3d.dtype)

    for key_index in range(key_len):
        padded_start = key_index * CONV_STRIDE - CONV_PADDING
        start = max(0, padded_start)
        end = min(real_len, padded_start + CONV_KERNEL)
        if start < end:
            key_mask[key_index] = True
            key_coords[key_index] = real_coords[start:end].mean(dim=0)

    query_coords = torch.zeros(target_len, 3, dtype=coords_3d.dtype)
    query_coords[:real_len] = real_coords
    difference = query_coords.unsqueeze(1) - key_coords.unsqueeze(0)
    distance = torch.sqrt(torch.sum(difference**2, dim=-1) + 1e-8)
    direction = difference / (distance.unsqueeze(-1) + 1e-8)
    position = torch.cat([distance.unsqueeze(-1), direction], dim=-1)

    valid_mask = query_mask.unsqueeze(1) & key_mask.unsqueeze(0)
    position = position * valid_mask.unsqueeze(-1)

    if not torch.isfinite(position).all():
        raise ValueError("Relative position features contain NaN or Inf values.")
    return key_mask, position, query_mask


def load_protein_features(
    protein_ids: Iterable[str], pretrained_root: Path, target_len: int, description: str
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    protein_dir = pretrained_root / "prottrans_result"
    structure_dir = pretrained_root / "esm_result"
    records = {}

    for protein_id in tqdm(list(protein_ids), desc=description):
        embedding = load_tensor(protein_dir / f"{protein_id}.tensor").float()
        coordinates = extract_ca_coordinates(structure_dir / f"{protein_id}.pdb")
        real_len = min(embedding.size(0), coordinates.size(0), target_len)
        if real_len == 0:
            raise ValueError(f"Protein {protein_id} has no aligned embedding/coordinate positions.")

        padded_embedding = torch.zeros(target_len, embedding.size(1), dtype=embedding.dtype)
        padded_embedding[:real_len] = embedding[:real_len]
        key_mask, position, query_mask = process_pos_and_mask(coordinates[:real_len], target_len)
        records[str(protein_id)] = (padded_embedding, key_mask, position, query_mask)

    return records


def normalize_drug_features(graphs: dict[str, Data], eps: float = 1e-8) -> dict[str, Data]:
    if not graphs:
        return graphs

    feature_dims = {graph.x.size(1) for graph in graphs.values()}
    if feature_dims != {11}:
        raise ValueError(f"Every drug node feature matrix must have 11 columns, got {sorted(feature_dims)}.")

    concatenated = torch.cat([graph.x[:, 1:10] for graph in graphs.values()], dim=0)
    minimum = concatenated.min(dim=0, keepdim=True).values
    maximum = concatenated.max(dim=0, keepdim=True).values
    feature_range = maximum - minimum
    feature_range[feature_range < eps] = 1.0

    for graph in graphs.values():
        graph.x = torch.cat(
            [graph.x[:, :1], (graph.x[:, 1:10] - minimum) / feature_range, graph.x[:, 10:11]], dim=1
        )
    return graphs


def load_drug_graphs(
    drug_ids: Iterable[str], pretrained_root: Path, node_dir_name: str, normalize: bool, description: str
) -> dict[str, Data]:
    node_dir = pretrained_root / node_dir_name
    edge_dir = pretrained_root / "drug_index"
    graphs = {}

    for drug_id in tqdm(list(drug_ids), desc=description):
        node_features = load_tensor(node_dir / f"{drug_id}.tensor").float()
        edge_index = load_tensor(edge_dir / f"{drug_id}.tensor").long()
        graphs[str(drug_id)] = Data(x=node_features, edge_index=edge_index, num_nodes=node_features.size(0))

    return normalize_drug_features(graphs) if normalize else graphs


def make_sample(drug: Data, protein_record, affinity: float):
    protein, key_mask, position, query_mask = protein_record
    return drug, protein, key_mask, position, query_mask, float(affinity)


def split_folds(samples: Sequence, train_folds: Sequence[Sequence[int]], test_indices: Sequence[int], fold: int):
    if fold < 0 or fold >= len(train_folds):
        raise ValueError(f"fold must be in [0, {len(train_folds) - 1}], got {fold}.")

    val_indices = list(train_folds[fold])
    train_indices = [index for fold_id, indices in enumerate(train_folds) if fold_id != fold for index in indices]
    return (
        [samples[index] for index in train_indices],
        [samples[index] for index in val_indices],
        [samples[index] for index in test_indices],
    )


def shuffle_affinities(samples: Sequence[tuple], seed: int) -> list[tuple]:
    affinities = [sample[-1] for sample in samples]
    random.Random(seed).shuffle(affinities)
    return [(*sample[:-1], affinity) for sample, affinity in zip(samples, affinities)]
