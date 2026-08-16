"""Generate ESMFold protein structures as PDB files."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from datasets import DATASETS, get_dataset_spec, load_protein_records
from utils import ensure_output_dir, resolve_device, should_skip


def generate_esm_structures(
    dataset: str,
    data_root: str | Path,
    output_root: str | Path,
    model_path: str | Path,
    device_name: str = "auto",
    chunk_size: int = 256,
    start_index: int = 0,
    end_index: int | None = None,
    skip_existing: bool = True,
) -> None:
    records = load_protein_records(dataset, data_root)
    spec = get_dataset_spec(dataset)
    selected = records[start_index:end_index]
    output_dir = ensure_output_dir(Path(output_root) / dataset / "esm_result")
    device = resolve_device(device_name)

    try:
        from transformers import AutoTokenizer, EsmForProteinFolding
    except ImportError as error:
        raise RuntimeError(
            "transformers is required for ESMFold generation. Install the preprocessing requirements first."
        ) from error

    print(f"device={device}")
    print(f"loading ESMFold model={model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = EsmForProteinFolding.from_pretrained(model_path, local_files_only=True).to(device)
    model.eval()
    if device.type == "cuda":
        model.esm = model.esm.half()
    model.trunk.set_chunk_size(chunk_size)

    generated = 0
    skipped = 0
    for record in tqdm(selected, desc=f"Generating {dataset} ESMFold structures"):
        output_path = output_dir / f"{record.protein_id}.pdb"
        if should_skip(output_path, skip_existing):
            skipped += 1
            continue

        sequence = record.sequence[: spec.protein_length]
        if not sequence:
            raise ValueError(f"Protein {record.protein_id!r} has an empty sequence after truncation.")

        with torch.no_grad():
            tokenized = tokenizer(
                [sequence],
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"].to(device)
            output = model(tokenized)
            cpu_output = {key: value.cpu() for key, value in output.items()}
            pdb_string = model.output_to_pdb(cpu_output)[0]

        output_path.write_text(pdb_string, encoding="utf-8")
        generated += 1

        del tokenized, output, cpu_output
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(
        f"dataset={dataset} proteins={len(records)} selected={len(selected)} "
        f"generated={generated} skipped={skipped} protein_length={spec.protein_length}"
    )
    print(f"esm_result={output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate ReGeoDTA ESMFold PDB structures.")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data-root", required=True, help="Directory containing the raw dataset files.")
    parser.add_argument(
        "--output-root",
        required=True,
        help="Root directory under which <dataset>/esm_result is created.",
    )
    parser.add_argument("--model-path", required=True, help="Local ESMFold model directory.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generate_esm_structures(
        dataset=args.dataset,
        data_root=args.data_root,
        output_root=args.output_root,
        model_path=args.model_path,
        device_name=args.device,
        chunk_size=args.chunk_size,
        start_index=args.start_index,
        end_index=args.end_index,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
