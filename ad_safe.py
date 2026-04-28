#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import ad_safe_lib as ad_safe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ad safety classifier."
    )
    parser.add_argument("--list-base-models", action="store_true", help="List supported base models and exit")
    parser.add_argument("--setup", type=str, default=argparse.SUPPRESS, help="Load run configuration from a setup JSON file")
    parser.add_argument("--model-path", type=str, default=argparse.SUPPRESS, help="Load a pre-trained model")
    parser.add_argument("--model-path-last", action="store_true", default=argparse.SUPPRESS, help="Use the newest '*-model.pt' checkpoint in artefacts/ad_safe_runs as --model-path")
    parser.add_argument("--train-split", default=argparse.SUPPRESS, help="Dataset split folder used as the source for resplitting during training")
    parser.add_argument("--train-fraction", type=float, default=argparse.SUPPRESS, help="Stratified fraction of the train split to use before normal training splits")
    parser.add_argument("--eval-split", default=argparse.SUPPRESS, help="Dataset split folder used for final metrics and figures; defaults to --train-split")
    parser.add_argument("--base-model", choices=sorted(ad_safe.SUPPORTED_BACKBONES), default=argparse.SUPPRESS, help="Base backbone to use when creating a new model")
    parser.add_argument(
        "--unfreeze-all",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Train all model layers instead of only the classification head",
    )
    parser.add_argument(
        "--unfreeze-top",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of last backbone blocks to unfreeze in addition to the classification head",
    )
    parser.add_argument(
        "--unfreeze",
        type=str,
        default=argparse.SUPPRESS,
        help="Comma-separated backbone block names to unfreeze in addition to the classification head",
    )
    parser.add_argument("--teacher-model-path", type=str, default=argparse.SUPPRESS, help="Optional checkpoint to use as a frozen teacher during training")
    parser.add_argument("--distillation-alpha", type=float, default=argparse.SUPPRESS, help="Weight for teacher KL loss when --teacher-model-path is provided")
    parser.add_argument("--distillation-temperature", type=float, default=argparse.SUPPRESS, help="Softmax temperature for teacher distillation")
    parser.add_argument("--cooldown-every-epochs", type=int, default=argparse.SUPPRESS, help="Run cooldown after every N global epochs. Use 0 to disable periodic cooldown.")
    parser.add_argument("--cooldown-seconds", type=float, default=argparse.SUPPRESS, help="Maximum cooldown duration in seconds. Required for enabled cooldown.")
    parser.add_argument("--gpu-max-temp", type=int, default=argparse.SUPPRESS, help="Start cooldown after an epoch when GPU temperature is at least this Celsius value. Use 0 to disable.")
    parser.add_argument("--gpu-resume-temp", type=int, default=argparse.SUPPRESS, help="Resume after GPU temperature is at or below this Celsius value. Defaults to --gpu-max-temp minus 5 when omitted.")
    parser.add_argument("--gpu-temp-check-seconds", type=float, default=argparse.SUPPRESS, help="Seconds between GPU temperature checks during cooldown.")
    parser.add_argument("--epochs", type=int, default=argparse.SUPPRESS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS, help="Training batch size")
    parser.add_argument(
        "--learning-rate",
        type=str,
        default=argparse.SUPPRESS,
        help="Optimizer learning rate, or comma-separated learning rates to use per split round",
    )
    parser.add_argument(
        "--learning-rate-multiplier",
        type=float,
        default=argparse.SUPPRESS,
        help="Multiplier applied to a single learning rate after each split round",
    )
    parser.add_argument("--resplit-runs", type=int, default=argparse.SUPPRESS, help="How many times to resplit train data and continue training")
    parser.add_argument("--patience", type=int, default=argparse.SUPPRESS, help="Early stopping patience (epochs to wait for improvement)")
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS, help="Random seed. Pass 0 for a fresh random seed")
    return parser.parse_args()


def print_supported_base_models() -> None:
    print("Supported base models:")
    for definition, native_input_size, backbone_parameter_count, block_names in ad_safe.list_supported_backbone_infos():
        print(
            f"- {definition.name}: native_input={native_input_size or definition.native_input_size}, "
            f"backbone_params={ad_safe.format_parameter_count(backbone_parameter_count)}, "
            f"available_blocks={len(block_names)}, "
            f"blocks={', '.join(block_names) if block_names else '<none>'}"
        )


def validate_fraction(value: object, *, field_name: str) -> float:
    fraction = float(value)
    if fraction <= 0 or fraction > 1:
        raise ValueError(f"{field_name} must be in the range (0, 1]")
    return fraction


def resolve_model_path(model_path_arg: str | None) -> Path | None:
    if model_path_arg is None:
        return None
    return ad_safe.resolve_required_existing_path(model_path_arg, field_name="model path")


def resolve_teacher_model_path(teacher_model_path_arg: str | None) -> str | None:
    teacher_model_path = resolve_model_path(teacher_model_path_arg)
    return str(teacher_model_path.resolve()) if teacher_model_path is not None else None


def resolve_latest_model_path() -> Path:
    candidates = sorted(ad_safe.AD_SAFE_RUNS_DIR.glob("*-model.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No '*-model.pt' checkpoints found in {ad_safe.AD_SAFE_RUNS_DIR}"
        )
    return candidates[-1]


def resolve_setup_path(setup_path_arg: str | None) -> Path | None:
    if setup_path_arg is None:
        return None
    return ad_safe.resolve_required_existing_path(setup_path_arg, field_name="setup path")


def build_run_plan(
    *,
    run_id: str,
    config: ad_safe.TrainingConfig,
    cooldown_config: ad_safe.CooldownConfig,
    train_split: str,
    eval_split: str,
    train_fraction: float,
    seed: int,
    original_model_path: Path | None,
    unfreeze_all: bool,
    unfreeze_top: int,
) -> ad_safe.RunPlan:
    phase = ad_safe.PhaseSpec(
        job_index=0,
        phase_index=0,
        prefix=run_id,
        title="main",
        requested_seed=seed,
        config=config,
        unfreeze_all=unfreeze_all,
        unfreeze_top=unfreeze_top,
        model_filename=f"{run_id}-model.pt",
        history_filename=f"{run_id}-training_history.png",
        json_filename=f"{run_id}-phase.json",
        signature={"builder": "ad_safe"},
    )
    ad_safe.ensure_artifact_dirs()
    return ad_safe.RunPlan(
        output_dir=ad_safe.AD_SAFE_RUNS_DIR,
        run_id=run_id,
        train_split=train_split,
        eval_splits=(eval_split,),
        jobs=(
            ad_safe.JobSpec(
                job_index=0,
                job_id="main",
                title=config.base_model,
                backbone=config.base_model,
                phases=(phase,),
                initial_model_path=original_model_path,
                metadata={"builder": "ad_safe"},
            ),
        ),
        cooldown=cooldown_config,
        resume=False,
        force=True,
        setup_path=ad_safe.AD_SAFE_RUNS_DIR / f"{run_id}-setup.json",
        metrics_csv_path=ad_safe.AD_SAFE_RUNS_DIR / f"{run_id}-accuracy.csv",
        check_foreign_contract=False,
        train_source=ad_safe.DatasetSourceSpec(
            name=train_split,
            fraction=train_fraction,
            seed=seed,
        ),
        metadata={
            "builder": "ad_safe",
            "original_model_path": ad_safe.path_to_json(original_model_path),
            "train_fraction": train_fraction,
        },
    )


def main() -> None:
    args = parse_args()
    cli_values = vars(args)

    if args.list_base_models:
        print_supported_base_models()
        return
    if "model_path" in cli_values and bool(cli_values.get("model_path_last", False)):
        raise ValueError("Use only one of --model-path or --model-path-last")

    setup_path = resolve_setup_path(cli_values.get("setup"))
    setup_values = ad_safe.load_setup_values(setup_path) if setup_path is not None else {}
    merged_values = ad_safe.merge_setup_and_cli_values(cli_values, setup_values)

    config = ad_safe.build_training_config(merged_values)
    config = replace(
        config,
        teacher_model_path=resolve_teacher_model_path(config.teacher_model_path),
    )
    cooldown_config = ad_safe.build_cooldown_config(
        every_epochs=int(merged_values["cooldown_every_epochs"]),
        seconds=float(merged_values["cooldown_seconds"]),
        gpu_max_temp=int(merged_values["gpu_max_temp"]),
        gpu_resume_temp=int(merged_values["gpu_resume_temp"]),
        gpu_temp_check_seconds=float(merged_values["gpu_temp_check_seconds"]),
    )
    seed = ad_safe.resolve_effective_seed(merged_values["seed"])
    ad_safe.set_seed(seed)
    train_split = merged_values["train_split"]
    eval_split = merged_values["eval_split"] or train_split
    train_fraction = validate_fraction(merged_values["train_fraction"], field_name="--train-fraction")
    unfreeze_all = bool(merged_values["unfreeze_all"])
    unfreeze_top = int(merged_values["unfreeze_top"])
    run_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    original_model_path = (
        resolve_latest_model_path()
        if bool(merged_values["model_path_last"])
        else resolve_model_path(merged_values["model_path"])
    )

    print(f"Using device: {ad_safe.DEVICE}")
    print(f"Using seed: {seed}")
    print(f"Using batch size: {config.batch_size}")
    print(f"Using resplit runs: {config.resplit_runs}")
    print(f"Train fraction: {train_fraction}")
    print(f"Cooldown: {cooldown_config.to_json()}")
    if setup_path is not None:
        print(f"Using setup: {setup_path}")
    print(f"Training source split: {train_split}")
    print(f"Evaluation split: {eval_split}")
    print(f"Original model path: {original_model_path}")

    if train_split is None and original_model_path is None and eval_split is not None:
        raise ValueError("Provide --train-split to train or --model-path to evaluate an existing model")
    if train_split is None and original_model_path is None and eval_split is None:
        raise ValueError("Provide --train-split, --eval-split, or --model-path")
    if train_split is None and original_model_path is not None and eval_split is None:
        ad_safe.describe_model_checkpoint(
            model_path=original_model_path,
            config=config,
            unfreeze_all=unfreeze_all,
            unfreeze_top=unfreeze_top,
        )
        return
    if eval_split is None:
        raise ValueError("--eval-split is required when --train-split is not provided")

    final_model_path = original_model_path
    if train_split is not None:
        run_result = ad_safe.run_training_plan(
            build_run_plan(
                run_id=run_id,
                config=config,
                cooldown_config=cooldown_config,
                train_split=train_split,
                eval_split=eval_split,
                train_fraction=train_fraction,
                seed=seed,
                original_model_path=original_model_path,
                unfreeze_all=unfreeze_all,
                unfreeze_top=unfreeze_top,
            )
        )
        final_model_path = run_result.phase_results[-1].model_path
    elif final_model_path is not None:
        ad_safe.run_evaluation_plan(
            ad_safe.EvaluationPlan(
                models=(
                    ad_safe.ModelEvalSpec(
                        path=final_model_path,
                        title=final_model_path.name,
                    ),
                ),
                datasets=(ad_safe.DatasetEvalSpec(name=eval_split, batch_size=config.batch_size),),
                write_csv=False,
                print_results=True,
                title="Results",
                sort_key=None,
            )
        )
    else:
        raise ValueError("No model is available for evaluation")

    ad_safe.generate_single_model_artifacts(
        model_path=final_model_path,
        config=config,
        eval_split=eval_split,
        output_dir=ad_safe.AD_SAFE_RUNS_DIR,
        output_prefix=run_id,
        unfreeze_all=unfreeze_all,
        unfreeze_top=unfreeze_top,
    )


if __name__ == "__main__":
    main()
