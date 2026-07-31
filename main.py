"""Command-line entry point for ReGeoDTA training and evaluation."""

from __future__ import annotations

import argparse


DATASETS = ("davis", "kiba", "bindingdb")


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=3)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)


def add_data_arguments(parser: argparse.ArgumentParser, dataset_required: bool) -> None:
    parser.add_argument("--dataset", choices=DATASETS, required=dataset_required)
    parser.add_argument("--data-root", required=True, help="Directory containing the raw dataset files.")
    parser.add_argument(
        "--pretrained-root", required=True,
        help="Directory containing prottrans_result, esm_result, drug features, and drug_index.",
    )
    parser.add_argument(
        "--fold", type=int, default=None,
        help="Validation fold. Defaults: Davis=0, KIBA=0, BindingDB=1.",
    )


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protein-kernel", type=int, nargs=3, default=(5, 9, 13), metavar=("K1", "K2", "K3"))
    parser.add_argument("--head-num", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--graph-mid-dim", type=int, default=96)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or evaluate ReGeoDTA.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train ReGeoDTA and save the best checkpoint.")
    add_data_arguments(train_parser, dataset_required=True)
    add_runtime_arguments(train_parser)
    add_model_arguments(train_parser)
    train_parser.add_argument("--output-dir", default=None)
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--eta-min", type=float, default=1e-8)
    train_parser.add_argument("--scheduler-start-epoch", type=int, default=None)
    train_parser.add_argument(
        "--shuffle-affinity", action="store_true",
        help="Shuffle affinity labels before fold selection. Intended only for controlled randomization experiments.",
    )
    train_parser.set_defaults(handler_name="train")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a checkpoint on validation or test data.")
    add_data_arguments(evaluate_parser, dataset_required=False)
    add_runtime_arguments(evaluate_parser)
    add_model_arguments(evaluate_parser)
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--split", choices=("val", "test"), default="test")
    evaluate_parser.add_argument("--batch-size", type=int, default=64)
    evaluate_parser.add_argument(
        "--shuffle-affinity", action=argparse.BooleanOptionalAction, default=None,
        help="Override the affinity-shuffling setting stored in a new-format checkpoint.",
    )
    evaluate_parser.set_defaults(handler_name="evaluate")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.handler_name == "train":
        from engine import run_train

        run_train(args)
    else:
        from engine import run_evaluate

        run_evaluate(args)


if __name__ == "__main__":
    main()
