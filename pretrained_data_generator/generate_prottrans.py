"""Generate ProtT5 residue embeddings for proteins."""

from __future__ import annotations

import argparse
import gc
import re
from pathlib import Path

import torch
from tqdm import tqdm

from datasets import DATASETS, get_dataset_spec, load_protein_records
from utils import ensure_output_dir, resolve_device, should_skip


NON_STANDARD_AA_PATTERN = re.compile(r"[UZOB]")


def _prepare_sequence(sequence: str) -> str:
    sequence = NON_STANDARD_AA_PATTERN.sub("X", sequence)
    return " ".join(sequence)


def generate_prottrans_embeddings(
    dataset: str,
    data_root: str | Path,
    output_root: str | Path,
    model_path: str | Path,
    device_name: str = "auto",
    batch_size: int = 1,
    start_index: int = 0,
    end_index: int | None = None,
    skip_existing: bool = True,
) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    records = load_protein_records(dataset, data_root)
    spec = get_dataset_spec(dataset)
    selected = records[start_index:end_index]
    output_dir = ensure_output_dir(Path(output_root) / dataset / "prottrans_result")
    device = resolve_device(device_name)

    try:
        from transformers import T5EncoderModel, T5Tokenizer
    except ImportError as error:
        raise RuntimeError(
            "transformers is required for ProtTrans generation. Install the preprocessing requirements first."
        ) from error

    print(f"device={device}")
    print(f"loading ProtTrans model={model_path}")
    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False, local_files_only=True)
    model = T5EncoderModel.from_pretrained(model_path, local_files_only=True).to(device)
    model.eval()
    gc.collect()

    generated = 0
    skipped = 0

    for batch_start in tqdm(
        range(0, len(selected), batch_size),
        desc=f"Generating {dataset} ProtTrans embeddings",
    ):
        batch_records = selected[batch_start : batch_start + batch_size]
        pending = [
            record
            for record in batch_records
            if not should_skip(output_dir / f"{record.protein_id}.tensor", skip_existing)
        ]
        skipped += len(batch_records) - len(pending)
        if not pending:
            continue

        sequences = [
            _prepare_sequence(record.sequence[: spec.protein_length])
            for record in pending
        ]
        encoded = tokenizer.batch_encode_plus(
            sequences,
            add_special_tokens=True,
            padding=True,
        )
        input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long, device=device)
        attention_mask = torch.tensor(encoded["attention_mask"], dtype=torch.long, device=device)

        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.cpu()

        attention_mask_cpu = attention_mask.cpu()
        for index, record in enumerate(pending):
            sequence_length = int(attention_mask_cpu[index].sum().item())
            residue_embedding = embedding[index, : sequence_length - 1].contiguous()
            expected_length = min(len(record.sequence), spec.protein_length)
            if residue_embedding.size(0) != expected_length:
                raise RuntimeError(
                    f"ProtTrans length mismatch for {record.protein_id!r}: "
                    f"expected {expected_length}, got {residue_embedding.size(0)}."
                )
            torch.save(residue_embedding, output_dir / f"{record.protein_id}.tensor")
            generated += 1

        del input_ids, attention_mask, attention_mask_cpu, embedding
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(
        f"dataset={dataset} proteins={len(records)} selected={len(selected)} "
        f"generated={generated} skipped={skipped} protein_length={spec.protein_length}"
    )
    print(f"prottrans_result={output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate ReGeoDTA ProtTrans embeddings.")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data-root", required=True, help="Directory containing the raw dataset files.")
    parser.add_argument(
        "--output-root",
        required=True,
        help="Root directory under which <dataset>/prottrans_result is created.",
    )
    parser.add_argument("--model-path", required=True, help="Local ProtT5 model directory.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generate_prottrans_embeddings(
        dataset=args.dataset,
        data_root=args.data_root,
        output_root=args.output_root,
        model_path=args.model_path,
        device_name=args.device,
        batch_size=args.batch_size,
        start_index=args.start_index,
        end_index=args.end_index,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
