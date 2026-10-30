"""Command-line entry point for training and evaluating the OnHW recognizer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from smartpen.hwr.training import evaluate_checkpoint, run_training


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a CNN-BiLSTM-CTC model on OnHW-Words500."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train and report held-out writer metrics")
    train.add_argument("--data", required=True, help="official OnHW ZIP or extracted folder")
    train.add_argument("--output", required=True, help="directory for checkpoint and metrics")
    train.add_argument("--fold", type=int, default=0, choices=range(5))
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--seed", type=int, default=1337)
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train.add_argument("--num-threads", type=int, default=4)
    train.add_argument("--max-train-samples", type=int)
    train.add_argument("--max-validation-samples", type=int)

    evaluate = commands.add_parser("evaluate", help="evaluate a saved model on the held-out fold")
    evaluate.add_argument("--data", required=True, help="official OnHW ZIP or extracted folder")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--fold", type=int, choices=range(5))
    evaluate.add_argument("--seed", type=int)
    evaluate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    evaluate.add_argument("--batch-size", type=int, default=32)
    evaluate.add_argument("--num-threads", type=int, default=4)
    evaluate.add_argument("--max-validation-samples", type=int)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "train":
        report = run_training(
            data_path=args.data,
            output_dir=args.output,
            fold=args.fold,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            device_name=args.device,
            max_train_samples=args.max_train_samples,
            max_validation_samples=args.max_validation_samples,
            num_threads=args.num_threads,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    result = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        fold=args.fold,
        max_validation_samples=args.max_validation_samples,
        seed=args.seed,
        device_name=args.device,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
