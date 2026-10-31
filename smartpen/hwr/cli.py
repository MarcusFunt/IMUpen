"""Command-line entry point for training and evaluating the OnHW recognizer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from smartpen.hwr.data import PERSONAL_CHARSET
from smartpen.hwr.augment import AUGMENTATION_NAMES
from smartpen.hwr.echwr_training import run_personal_echwr_comparison
from smartpen.hwr.model import MODEL_TYPES
from smartpen.hwr.personal import (
    run_onhw_encoder_pretraining,
    run_personal_comparison,
    run_personal_ts2vec_comparison,
)
from smartpen.hwr.training import (
    evaluate_checkpoint,
    run_augmentation_ablation,
    run_comparison,
    run_training,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate CTC handwriting models on OnHW-Words500."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train and report held-out writer metrics")
    _add_training_arguments(train)
    train.add_argument("--model-type", choices=MODEL_TYPES, default="cnn_bilstm_ctc")

    compare = commands.add_parser(
        "compare", help="train M4 and M5 with identical data and optimization settings"
    )
    _add_training_arguments(compare)

    augment_ablate = commands.add_parser(
        "augment-ablate",
        help="run M6-B baseline and independent training-only IMU augmentation ablations",
    )
    _add_training_arguments(augment_ablate)
    augment_ablate.add_argument("--model-type", choices=MODEL_TYPES, default="resnet_bilstm_ctc")
    augment_ablate.add_argument(
        "--include-combined",
        action="store_true",
        help="also run a combined augmentation experiment after all isolated variants",
    )
    augment_ablate.add_argument(
        "--concat-delimiter",
        default="",
        help="label boundary for concatenation (OnHW vocabulary requires the default empty string)",
    )
    augment_ablate.add_argument("--concat-gap-frames", type=int, default=8)

    pretrain = commands.add_parser(
        "pretrain", help="pretrain the M5 encoder on the licensed OnHW dataset"
    )
    _add_training_arguments(pretrain)

    personalize = commands.add_parser(
        "personalize",
        help="compare OnHW encoder fine-tuning with training from scratch on recorder data",
    )
    personalize.add_argument("--recordings", required=True, help="RecordingStore output root")
    personalize.add_argument("--pretrained-checkpoint", required=True)
    personalize.add_argument("--output", required=True, help="directory for split, checkpoints, and metrics")
    personalize.add_argument("--charset", default=PERSONAL_CHARSET)
    personalize.add_argument("--validation-fraction", type=float, default=0.2)
    personalize.add_argument("--split-manifest", help="reuse an existing session_split.json")
    personalize.add_argument("--epochs", type=int, default=20)
    personalize.add_argument("--batch-size", type=int, default=16)
    personalize.add_argument("--seed", type=int, default=1337)
    personalize.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    personalize.add_argument("--num-threads", type=int, default=4)
    personalize.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="use CUDA float16 autocast (defaults to enabled on CUDA)",
    )

    personalize_ssl = commands.add_parser(
        "personalize-ssl",
        help="compare TS2Vec-style personal self-supervision with labeled training from scratch",
    )
    personalize_ssl.add_argument("--recordings", required=True, help="labeled RecordingStore root")
    personalize_ssl.add_argument(
        "--unlabeled-recordings",
        required=True,
        help="accepted recorder sequences for SSL; prompt labels are ignored",
    )
    personalize_ssl.add_argument("--output", required=True)
    personalize_ssl.add_argument("--charset", default=PERSONAL_CHARSET)
    personalize_ssl.add_argument("--validation-fraction", type=float, default=0.2)
    personalize_ssl.add_argument("--split-manifest", help="reuse an existing session_split.json")
    personalize_ssl.add_argument("--ssl-epochs", type=int, default=20)
    personalize_ssl.add_argument("--epochs", type=int, default=20, help="supervised fine-tuning epochs")
    personalize_ssl.add_argument("--batch-size", type=int, default=16)
    personalize_ssl.add_argument("--max-unlabeled-samples", type=int)
    personalize_ssl.add_argument("--seed", type=int, default=1337)
    personalize_ssl.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    personalize_ssl.add_argument("--num-threads", type=int, default=4)
    personalize_ssl.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="use CUDA float16 autocast (defaults to enabled on CUDA)",
    )

    personalize_echwr = commands.add_parser(
        "personalize-echwr",
        help="compare CTC, in-batch contrastive, and ECHWR-style hard-negative training",
    )
    personalize_echwr.add_argument("--recordings", required=True, help="labeled RecordingStore root")
    personalize_echwr.add_argument("--output", required=True)
    personalize_echwr.add_argument(
        "--pretrained-checkpoint",
        help="optional M6-A OnHW encoder checkpoint shared by every comparison variant",
    )
    personalize_echwr.add_argument("--charset", default=PERSONAL_CHARSET)
    personalize_echwr.add_argument("--validation-fraction", type=float, default=0.2)
    personalize_echwr.add_argument("--split-manifest", help="reuse an existing session_split.json")
    personalize_echwr.add_argument("--epochs", type=int, default=20)
    personalize_echwr.add_argument("--batch-size", type=int, default=16)
    personalize_echwr.add_argument("--max-train-samples", type=int)
    personalize_echwr.add_argument("--max-validation-samples", type=int)
    personalize_echwr.add_argument("--seed", type=int, default=1337)
    personalize_echwr.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    personalize_echwr.add_argument("--num-threads", type=int, default=4)
    personalize_echwr.add_argument("--batch-contrastive-weight", type=float, default=1.0)
    personalize_echwr.add_argument("--error-contrastive-weight", type=float, default=1.0)
    personalize_echwr.add_argument("--negative-sets", type=int, default=1)
    personalize_echwr.add_argument(
        "--augmentation", choices=AUGMENTATION_NAMES, default="none",
        help="same M6-B training-only augmentation for all three variants",
    )
    personalize_echwr.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="use CUDA float16 autocast (defaults to enabled on CUDA)",
    )

    evaluate = commands.add_parser("evaluate", help="evaluate a saved model on the held-out fold")
    evaluate.add_argument("--data", required=True, help="official OnHW ZIP or extracted folder")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--fold", type=int, choices=range(5))
    evaluate.add_argument("--seed", type=int)
    evaluate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    evaluate.add_argument("--batch-size", type=int, default=32)
    evaluate.add_argument("--num-threads", type=int, default=4)
    evaluate.add_argument("--max-validation-samples", type=int)
    evaluate.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="use CUDA float16 autocast (defaults to enabled on CUDA)",
    )
    return parser


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", required=True, help="official OnHW ZIP or extracted folder")
    parser.add_argument("--output", required=True, help="directory for checkpoint and metrics")
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="use CUDA float16 autocast (defaults to enabled on CUDA)",
    )


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
            model_type=args.model_type,
            mixed_precision=args.mixed_precision,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if args.command == "compare":
        comparison = run_comparison(
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
            mixed_precision=args.mixed_precision,
        )
        print(json.dumps(comparison, ensure_ascii=False, indent=2))
        return

    if args.command == "augment-ablate":
        report = run_augmentation_ablation(
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
            mixed_precision=args.mixed_precision,
            model_type=args.model_type,
            include_combined=args.include_combined,
            concat_delimiter=args.concat_delimiter,
            concat_gap_frames=args.concat_gap_frames,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if args.command == "pretrain":
        report = run_onhw_encoder_pretraining(
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
            mixed_precision=args.mixed_precision,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if args.command == "personalize":
        comparison = run_personal_comparison(
            recordings_dir=args.recordings,
            pretrained_checkpoint=args.pretrained_checkpoint,
            output_dir=args.output,
            charset=args.charset,
            validation_fraction=args.validation_fraction,
            split_manifest_path=args.split_manifest,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            device_name=args.device,
            num_threads=args.num_threads,
            mixed_precision=args.mixed_precision,
        )
        print(json.dumps(comparison, ensure_ascii=False, indent=2))
        return

    if args.command == "personalize-ssl":
        comparison = run_personal_ts2vec_comparison(
            recordings_dir=args.recordings,
            unlabeled_recordings_dir=args.unlabeled_recordings,
            output_dir=args.output,
            charset=args.charset,
            validation_fraction=args.validation_fraction,
            split_manifest_path=args.split_manifest,
            ssl_epochs=args.ssl_epochs,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_unlabeled_samples=args.max_unlabeled_samples,
            seed=args.seed,
            device_name=args.device,
            num_threads=args.num_threads,
            mixed_precision=args.mixed_precision,
        )
        print(json.dumps(comparison, ensure_ascii=False, indent=2))
        return

    if args.command == "personalize-echwr":
        comparison = run_personal_echwr_comparison(
            recordings_dir=args.recordings,
            output_dir=args.output,
            pretrained_checkpoint=args.pretrained_checkpoint,
            charset=args.charset,
            validation_fraction=args.validation_fraction,
            split_manifest_path=args.split_manifest,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_train_samples=args.max_train_samples,
            max_validation_samples=args.max_validation_samples,
            seed=args.seed,
            device_name=args.device,
            num_threads=args.num_threads,
            mixed_precision=args.mixed_precision,
            batch_contrastive_weight=args.batch_contrastive_weight,
            error_contrastive_weight=args.error_contrastive_weight,
            negative_sets=args.negative_sets,
            augmentation=args.augmentation,
        )
        print(json.dumps(comparison, ensure_ascii=False, indent=2))
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
        mixed_precision=args.mixed_precision,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
