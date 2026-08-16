"""Generate drug atom features and graph edge indices."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from rdkit import Chem, rdBase
from tqdm import tqdm

from datasets import DATASETS, load_drug_records
from utils import ensure_output_dir, should_skip


rdBase.DisableLog("rdApp.warning")

ATOM_TYPE_MAP = {
    "C": 1,
    "N": 2,
    "O": 3,
    "F": 4,
    "Cl": 5,
    "Br": 6,
    "I": 7,
    "S": 8,
    "P": 9,
    "Other": 10,
}


def get_atom_features(atom: Chem.Atom) -> torch.Tensor:
    symbol_index = ATOM_TYPE_MAP.get(atom.GetSymbol(), ATOM_TYPE_MAP["Other"])
    degree = float(atom.GetDegree())
    total_hydrogens = float(atom.GetTotalNumHs())
    implicit_valence = float(atom.GetImplicitValence())
    aromaticity = 1.0 if atom.GetIsAromatic() else -1.0

    aromatic_factor = (aromaticity + 1.0) / 2.0
    non_aromatic_factor = (-aromaticity + 1.0) / 2.0
    features = [
        float(symbol_index),
        degree,
        total_hydrogens,
        implicit_valence,
        degree * aromatic_factor,
        total_hydrogens * aromatic_factor,
        implicit_valence * aromatic_factor,
        degree * non_aromatic_factor,
        total_hydrogens * non_aromatic_factor,
        implicit_valence * non_aromatic_factor,
        aromaticity,
    ]
    return torch.tensor(features, dtype=torch.float32)


def smiles_to_graph(smiles: str) -> tuple[torch.Tensor, torch.Tensor]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles!r}")

    atoms = list(mol.GetAtoms())
    if not atoms:
        raise ValueError(f"SMILES contains no atoms: {smiles!r}")

    node_features = torch.stack([get_atom_features(atom) for atom in atoms])

    edges: list[list[int]] = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        edges.append([begin, end])
        edges.append([end, begin])

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return node_features, edge_index


def generate_drug_graphs(
    dataset: str,
    data_root: str | Path,
    output_root: str | Path,
    skip_existing: bool = True,
) -> None:
    records = load_drug_records(dataset, data_root)
    dataset_output = Path(output_root) / dataset
    node_dir = ensure_output_dir(dataset_output / "drug_node_features")
    edge_dir = ensure_output_dir(dataset_output / "drug_index")

    max_atoms = 0
    generated = 0
    skipped = 0

    for record in tqdm(records, desc=f"Generating {dataset} drug graphs"):
        node_path = node_dir / f"{record.drug_id}.tensor"
        edge_path = edge_dir / f"{record.drug_id}.tensor"
        if should_skip(node_path, skip_existing) and should_skip(edge_path, skip_existing):
            skipped += 1
            continue

        node_features, edge_index = smiles_to_graph(record.smiles)
        max_atoms = max(max_atoms, node_features.size(0))
        torch.save(node_features, node_path)
        torch.save(edge_index, edge_path)
        generated += 1

    print(
        f"dataset={dataset} drugs={len(records)} generated={generated} "
        f"skipped={skipped} max_atoms={max_atoms}"
    )
    print(f"node_features={node_dir}")
    print(f"edge_index={edge_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate ReGeoDTA drug graph tensors.")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data-root", required=True, help="Directory containing the raw dataset files.")
    parser.add_argument(
        "--output-root",
        required=True,
        help="Root directory under which <dataset>/drug_node_features and drug_index are created.",
    )
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generate_drug_graphs(args.dataset, args.data_root, args.output_root, args.skip_existing)


if __name__ == "__main__":
    main()
