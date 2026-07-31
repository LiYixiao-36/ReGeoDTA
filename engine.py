"""Training and evaluation workflows."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from data import build_dataset_bundle, resolve_fold
from metrics import compute_metrics
from model import ReGeoDTA
from utils import resolve_device, seed_worker, set_seed


def _loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    num_workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
    pin_memory: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory and device.type == "cuda",
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = persistent_workers
    return DataLoader(dataset, **kwargs)


def _move_batch(batch, device: torch.device):
    drug, protein, key_mask, position, query_mask, affinity = batch
    non_blocking = device.type == "cuda"
    return (
        drug.to(device),
        protein.to(device, non_blocking=non_blocking),
        key_mask.to(device, non_blocking=non_blocking),
        position.to(device, non_blocking=non_blocking),
        query_mask.to(device, non_blocking=non_blocking),
        affinity.to(device, non_blocking=non_blocking),
    )


def _prediction(model: nn.Module, protein, drug, position, key_mask, query_mask) -> torch.Tensor:
    output = model(protein, drug, position, key_mask, query_mask)
    return output[0] if isinstance(output, (tuple, list)) else output


def _load_state_dict(path: str | Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        state_dict = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location=device)

    if not isinstance(state_dict, dict) or not all(isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise TypeError("The model file must contain a plain PyTorch state_dict.")
    return state_dict


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        drug, protein, key_mask, position, query_mask, affinity = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = _prediction(model, protein, drug, position, key_mask, query_mask)
        loss = criterion(prediction.float(), affinity.float())
        loss.backward()
        optimizer.step()

        batch_size = affinity.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("The training DataLoader is empty.")
    return total_loss / total_samples


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    aupr_threshold: float,
) -> dict[str, float]:
    model.eval()
    predictions = []
    labels = []

    for batch in loader:
        drug, protein, key_mask, position, query_mask, affinity = _move_batch(batch, device)
        prediction = _prediction(model, protein, drug, position, key_mask, query_mask)
        predictions.append(prediction.detach().cpu())
        labels.append(affinity.detach().cpu())

    if not predictions:
        raise RuntimeError("The evaluation DataLoader is empty.")

    y_pred = torch.cat(predictions).view(-1).numpy()
    y_true = torch.cat(labels).view(-1).numpy()
    return compute_metrics(y_true, y_pred, aupr_threshold)


def _metric_value(value: float) -> str:
    return "nan" if value != value else f"{value:.6f}"


def _format_metrics(prefix: str, metrics: dict[str, float]) -> str:
    fields = " ".join(f"{name}={_metric_value(value)}" for name, value in metrics.items())
    return f"{prefix} {fields}"


def _print_and_log(line: str, log_path: Path) -> None:
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as file:
        file.write(line + "\n")


def _model_config(args: Namespace) -> dict:
    return {
        "protein_kernel": tuple(args.protein_kernel),
        "head_num": args.head_num,
        "dropout_rate": args.dropout,
        "graph_mid_dim": args.graph_mid_dim,
    }


def run_train(args: Namespace) -> None:
    set_seed(args.seed, deterministic=args.deterministic)
    device = resolve_device(args.device)
    fold = resolve_fold(args.dataset, args.fold)
    output_dir = Path(args.output_dir or Path("runs") / args.dataset / f"fold_{fold}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    log_path.write_text("", encoding="utf-8")

    print(f"device={device}")
    print(f"loading dataset={args.dataset} fold={fold}")
    bundle = build_dataset_bundle(
        args.dataset, args.data_root, args.pretrained_root, fold, args.shuffle_affinity, args.seed
    )
    print(f"samples: train={len(bundle.train)} val={len(bundle.val)} test={len(bundle.test)}")

    train_loader = _loader(
        bundle.train, args.batch_size, True, device, args.num_workers, args.prefetch_factor,
        args.persistent_workers, args.pin_memory, args.seed,
    )
    val_loader = _loader(
        bundle.val, args.batch_size, False, device, args.num_workers, args.prefetch_factor,
        args.persistent_workers, args.pin_memory, args.seed,
    )
    test_loader = _loader(
        bundle.test, args.batch_size, False, device, args.num_workers, args.prefetch_factor,
        args.persistent_workers, args.pin_memory, args.seed,
    )

    model_config = _model_config(args)
    model = ReGeoDTA(**model_config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    scheduler_start = args.scheduler_start_epoch
    if scheduler_start is None:
        scheduler_start = args.epochs // 2
    if scheduler_start < 0 or scheduler_start >= args.epochs:
        raise ValueError("scheduler-start-epoch must be between 0 and epochs - 1.")
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - scheduler_start), eta_min=args.eta_min)

    best_mse = float("inf")
    best_epoch = -1
    model_path = output_dir / f"ReGeoDTA_{args.dataset}.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_model(model, val_loader, device, bundle.aupr_threshold)
        line = _format_metrics(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} lr={optimizer.param_groups[0]['lr']:.8g}",
            {f"val_{name}": value for name, value in val_metrics.items()},
        )
        _print_and_log(line, log_path)

        if val_metrics["mse"] < best_mse:
            best_mse = val_metrics["mse"]
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)

        if epoch > scheduler_start:
            scheduler.step()

    best_model = ReGeoDTA(**model_config).to(device)
    best_model.load_state_dict(_load_state_dict(model_path, device))
    best_model.eval()

    test_metrics = evaluate_model(best_model, test_loader, device, bundle.aupr_threshold)
    _print_and_log(_format_metrics(f"best_epoch={best_epoch:03d} test", test_metrics), log_path)
    print(f"model={model_path}")
    print(f"metric_log={log_path}")


def run_evaluate(args: Namespace) -> None:
    set_seed(args.seed, deterministic=args.deterministic)
    device = resolve_device(args.device)

    if args.dataset is None:
        raise ValueError("--dataset is required when evaluating a plain state_dict model file.")

    dataset = args.dataset
    fold = resolve_fold(dataset, args.fold)
    shuffle_affinity = bool(args.shuffle_affinity) if args.shuffle_affinity is not None else False

    print(f"device={device}")
    print(f"loading dataset={dataset} fold={fold}")
    bundle = build_dataset_bundle(
        dataset, args.data_root, args.pretrained_root, fold, shuffle_affinity, args.seed
    )
    selected_dataset = bundle.val if args.split == "val" else bundle.test
    loader = _loader(
        selected_dataset, args.batch_size, False, device, args.num_workers, args.prefetch_factor,
        args.persistent_workers, args.pin_memory, args.seed,
    )

    model = ReGeoDTA(**_model_config(args)).to(device)
    model.load_state_dict(_load_state_dict(args.checkpoint, device))
    model.eval()

    metrics = evaluate_model(model, loader, device, bundle.aupr_threshold)
    print(_format_metrics(f"split={args.split}", metrics))
