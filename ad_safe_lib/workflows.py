from __future__ import annotations

from pathlib import Path

from .artifacts import load_model, release_torch_memory
from .backbones import configure_trainable_layers, describe_model, finalize_training_config
from .config import TrainingConfig
from .data import load_dataset
from .figures import (
    generate_adversarial_example,
    generate_class_reversal_figure,
    generate_test_figure,
    save_figure,
)


def describe_model_checkpoint(
    *,
    model_path: Path,
    config: TrainingConfig,
    unfreeze_all: bool = False,
    unfreeze_top: int = 0,
) -> TrainingConfig:
    model = load_model(model_path)
    finalized_config = finalize_training_config(
        model,
        config,
        unfreeze_all=unfreeze_all,
        unfreeze_top=unfreeze_top,
    )
    configure_trainable_layers(model, unfreeze=finalized_config.unfreeze)
    describe_model(model, finalized_config)
    del model
    release_torch_memory()
    return finalized_config


def generate_single_model_artifacts(
    *,
    model_path: Path,
    config: TrainingConfig,
    eval_split: str,
    output_dir: Path,
    output_prefix: str,
    unfreeze_all: bool = False,
    unfreeze_top: int = 0,
) -> None:
    model = load_model(model_path)
    finalized_config = finalize_training_config(
        model,
        config,
        unfreeze_all=unfreeze_all,
        unfreeze_top=unfreeze_top,
    )
    configure_trainable_layers(model, unfreeze=finalized_config.unfreeze)
    describe_model(model, finalized_config)
    eval_dataset = load_dataset(eval_split)

    print(f"\nGenerating sample figure from {eval_split}...")
    save_figure(
        generate_test_figure(
            model=model,
            sample_dataset=eval_dataset,
        ),
        output_dir / f"{output_prefix}-test.png",
    )

    print(f"\nGenerating adversarial attack example from {eval_split}...")
    save_figure(
        generate_adversarial_example(
            model=model,
            sample_dataset=eval_dataset,
            epsilon=finalized_config.adv_epsilon,
            num_steps=max(finalized_config.adv_steps, 1),
        ),
        output_dir / f"{output_prefix}_adversarial.png",
    )

    print("\nGenerating class reversal figure...")
    save_figure(
        generate_class_reversal_figure(model=model),
        output_dir / f"{output_prefix}_class_reversal.png",
    )
    del model
    release_torch_memory()
