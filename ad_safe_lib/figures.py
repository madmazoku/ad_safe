from __future__ import annotations

import os
import random
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib


def is_notebook_environment() -> bool:
    try:
        from IPython import get_ipython
    except ImportError:
        return False

    shell = get_ipython()
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


if not is_notebook_environment():
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset
from matplotlib.figure import Figure

from .backbones import forward_logits
from .config import CLASS_NAMES, DEFAULT_ADV_STEPS, DEFAULT_TARGET_STEPS, DEVICE, TrainingConfig, TrainingHistoryEntry
from .data import get_dataset_classes
from .reversal import ClassReversalContext, ClassReversalStrategy, RandomRestartTargetClassStrategy
from .training import AdversarialAttackStrategy, MinimalFlipPgdStrategy, generate_adversarial_perturbation


def evaluate_samples(model: nn.Module, *samples: Tensor) -> tuple[tuple[Tensor, Tensor], ...]:
    model.eval()
    evaluations: list[tuple[Tensor, Tensor]] = []

    with torch.inference_mode():
        for sample in samples:
            logits = forward_logits(model, sample)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            confs = probs.gather(1, preds.unsqueeze(1)).squeeze(1)
            evaluations.append((preds, confs))

    return tuple(evaluations)


def save_figure(fig: Figure, output_path: Path) -> None:
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    print(f"Figure saved to {output_path}")


def format_info_block_lines(
    items: list[tuple[str, str]],
    *,
    width: int = 38,
    indent: int = 2,
) -> str:
    wrapped_lines: list[str] = []
    subsequent_indent = " " * indent

    for key, value in items:
        prefix = f"{key}="
        wrapped_value = textwrap.wrap(
            value,
            width=max(12, width - len(prefix)),
            break_long_words=False,
            break_on_hyphens=False,
            subsequent_indent=subsequent_indent,
        )
        if not wrapped_value:
            wrapped_lines.append(prefix)
            continue

        wrapped_lines.append(f"{prefix}{wrapped_value[0]}")
        wrapped_lines.extend(wrapped_value[1:])

    return "\n".join(wrapped_lines)


def generate_training_history_figure(
    history: list[TrainingHistoryEntry],
    config: TrainingConfig,
    train_split: str,
    seed: int,
    input_model_path: Path | None,
    output_model_path: Path,
) -> Figure:
    if not history:
        raise ValueError("Training history is empty")

    epoch_history = [entry for entry in history if not entry.is_baseline]
    baseline_history = [entry for entry in history if entry.is_baseline]
    global_epochs = [entry.global_epoch for entry in epoch_history]
    val_scores = [entry.val_accuracy for entry in epoch_history]
    best_epochs = [entry.global_epoch for entry in epoch_history if entry.is_new_best]
    best_scores = [entry.score for entry in epoch_history if entry.is_new_best]

    fig = plt.figure(figsize=(13.5, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=(5.8, 0.9), wspace=0.08)
    score_ax = fig.add_subplot(gs[0, 0])
    info_ax = fig.add_subplot(gs[0, 1])

    score_ax.plot(global_epochs, val_scores, marker="o", linewidth=2, label="val accuracy")

    if best_epochs:
        score_ax.scatter(
            best_epochs,
            best_scores,
            color="crimson",
            s=90,
            zorder=5,
            label="new best checkpoint",
        )

    round_starts: dict[int, int] = {}
    for entry in baseline_history:
        round_starts.setdefault(entry.split_round, int(entry.global_epoch + 0.5))

    baseline_label_drawn = False
    for split_round, start_epoch in sorted(round_starts.items()):
        round_entries = [entry for entry in epoch_history if entry.split_round == split_round]
        if round_entries:
            end_epoch = int(round_entries[-1].global_epoch)
        else:
            end_epoch = start_epoch
        baseline_entry = next(entry for entry in baseline_history if entry.split_round == split_round)
        score_ax.hlines(
            y=baseline_entry.score,
            xmin=start_epoch,
            xmax=end_epoch,
            colors="0.35",
            linestyles=":",
            linewidth=2,
            alpha=0.6,
            label="split baseline" if not baseline_label_drawn else None,
        )
        baseline_label_drawn = True

        if split_round > 1:
            score_ax.axvline(start_epoch - 0.5, color="0.75", linestyle="--", linewidth=1)
        score_ax.text(
            start_epoch,
            1.005,
            f"split {split_round}",
            transform=score_ax.get_xaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=9,
        )

    score_ax.set_ylabel("Accuracy / score")
    score_ax.set_xlabel("Global epoch")
    score_ax.grid(True, alpha=0.3)
    score_ax.legend(loc="lower right")
    info_ax.axis("off")

    params_text = format_info_block_lines(
        [
            ("train_split", train_split),
            ("base_model", config.base_model),
            ("epochs", str(config.epochs)),
            ("resplit_runs", str(config.resplit_runs)),
            ("batch_size", str(config.batch_size)),
            ("learning_rate", ",".join(str(value) for value in config.learning_rate)),
            ("learning_rate_multiplier", str(config.learning_rate_multiplier)),
            ("unfreeze", ",".join(config.unfreeze) if config.unfreeze else "<head only>"),
            (
                "teacher_model",
                Path(config.teacher_model_path).name if config.teacher_model_path is not None else "<none>",
            ),
            ("distill_alpha", str(config.distillation_alpha)),
            ("distill_temp", str(config.distillation_temperature)),
            ("seed", str(seed)),
            (
                "input_model",
                Path(input_model_path).name if input_model_path is not None else "<new model>",
            ),
            ("output_model", Path(output_model_path).name),
        ],
        width=34,
        indent=2,
    )
    info_ax.text(
        0.02,
        0.98,
        params_text,
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9},
        transform=info_ax.transAxes,
    )

    fig.suptitle("Training History", y=0.97)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.11, top=0.86, wspace=0.1)
    plt.close(fig)
    return fig


def generate_adversarial_example(
    model: nn.Module,
    sample_dataset: Dataset,
    epsilon: float = 0.15,
    num_steps: int = DEFAULT_ADV_STEPS,
    max_tries: int = 24,
    require_correct_original: bool = True,
    sample_index: int | None = None,
    strategy: AdversarialAttackStrategy | None = None,
    search_steps: int = 10,
    refinement_steps: int = 6,
) -> Figure:
    criterion = nn.CrossEntropyLoss()
    if len(sample_dataset) == 0:
        raise ValueError("Sample dataset is empty")
    if max_tries <= 0:
        raise ValueError("max_tries must be positive")
    if sample_index is not None and (sample_index < 0 or sample_index >= len(sample_dataset)):
        raise ValueError(
            f"sample_index must be in range [0, {len(sample_dataset) - 1}], got {sample_index}"
        )
    attack_strategy = strategy or MinimalFlipPgdStrategy(
        max_epsilon=epsilon,
        num_steps=num_steps,
        search_steps=search_steps,
        refinement_steps=refinement_steps,
    )

    class_names = get_dataset_classes(sample_dataset)
    if sample_index is None:
        candidate_indices = random.sample(
            range(len(sample_dataset)),
            k=min(max_tries, len(sample_dataset)),
        )
    else:
        candidate_indices = [sample_index]
    attempts = []

    for sample_index in candidate_indices:
        image_tensor, label_idx = sample_dataset[sample_index]
        true_label = class_names[label_idx]
        target_label_idx = CLASS_NAMES.index(true_label)
        original_tensor = image_tensor.unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            original_logits = forward_logits(model, original_tensor)
            original_probs = torch.softmax(original_logits, dim=1)
            original_pred = int(original_logits.argmax(dim=1).item())
            original_conf = float(original_probs[0, original_pred].item())
            original_true_conf = float(original_probs[0, target_label_idx].item())

        if require_correct_original and original_pred != target_label_idx:
            attempts.append(
                {
                    "sample_index": sample_index,
                    "image_tensor": image_tensor,
                    "original_tensor": original_tensor,
                    "adversarial_tensor": None,
                    "target_label_idx": target_label_idx,
                    "true_label": true_label,
                    "original_pred": original_pred,
                    "original_conf": original_conf,
                    "original_true_conf": original_true_conf,
                    "adversarial_pred": None,
                    "adversarial_conf": None,
                    "adversarial_true_conf": None,
                    "attack_success": False,
                    "skipped": True,
                }
            )
            continue

        attack_result = generate_adversarial_perturbation(
            model=model,
            x_original=original_tensor,
            criterion=criterion,
            target_labels=torch.tensor([target_label_idx], device=DEVICE),
            strategy=attack_strategy,
        )
        attempt = {
            "sample_index": sample_index,
            "image_tensor": image_tensor,
            "original_tensor": original_tensor,
            "adversarial_tensor": attack_result.adversarial_tensor,
            "target_label_idx": target_label_idx,
            "true_label": true_label,
            "original_pred": original_pred,
            "original_conf": original_conf,
            "original_true_conf": original_true_conf,
            "adversarial_pred": attack_result.prediction,
            "adversarial_conf": attack_result.confidence,
            "adversarial_true_conf": attack_result.true_confidence,
            "attack_success": attack_result.attack_success,
            "attack_epsilon": attack_result.epsilon,
            "attack_linf": attack_result.linf,
            "attack_rms": attack_result.rms,
            "attack_mae": attack_result.mae,
            "skipped": False,
        }
        attempts.append(attempt)
        if attack_result.attack_success:
            break

    attacked_attempts = [
        attempt
        for attempt in attempts
        if attempt["adversarial_tensor"] is not None
    ]
    if not attacked_attempts:
        sample_index = candidate_indices[0]
        image_tensor, label_idx = sample_dataset[sample_index]
        true_label = class_names[label_idx]
        target_label_idx = CLASS_NAMES.index(true_label)
        original_tensor = image_tensor.unsqueeze(0).to(DEVICE)
        attack_result = generate_adversarial_perturbation(
            model=model,
            x_original=original_tensor,
            criterion=criterion,
            target_labels=torch.tensor([target_label_idx], device=DEVICE),
            strategy=attack_strategy,
        )
        selected_attempt = {
            "sample_index": sample_index,
            "image_tensor": image_tensor,
            "original_tensor": original_tensor,
            "adversarial_tensor": attack_result.adversarial_tensor,
            "target_label_idx": target_label_idx,
            "true_label": true_label,
            "original_pred": attack_result.original_prediction,
            "original_conf": attack_result.original_confidence,
            "original_true_conf": attack_result.original_true_confidence,
            "adversarial_pred": attack_result.prediction,
            "adversarial_conf": attack_result.confidence,
            "adversarial_true_conf": attack_result.true_confidence,
            "attack_success": attack_result.attack_success,
            "attack_epsilon": attack_result.epsilon,
            "attack_linf": attack_result.linf,
            "attack_rms": attack_result.rms,
            "attack_mae": attack_result.mae,
            "skipped": False,
        }
    else:
        successful_attempts = [
            attempt for attempt in attacked_attempts if attempt["attack_success"]
        ]
        selected_attempt = (
            successful_attempts[0]
            if successful_attempts
            else min(
                attacked_attempts,
                key=lambda attempt: float(attempt["adversarial_true_conf"]),
            )
        )

    sample_index = int(selected_attempt["sample_index"])
    image_tensor = selected_attempt["image_tensor"]
    adversarial_tensor = selected_attempt["adversarial_tensor"]
    true_label = str(selected_attempt["true_label"])
    original_pred = int(selected_attempt["original_pred"])
    original_conf = float(selected_attempt["original_conf"])
    adversarial_pred = int(selected_attempt["adversarial_pred"])
    adversarial_conf = float(selected_attempt["adversarial_conf"])
    attack_success = bool(selected_attempt["attack_success"])
    attack_epsilon = float(selected_attempt["attack_epsilon"])
    attack_linf = float(selected_attempt["attack_linf"])
    attack_rms = float(selected_attempt["attack_rms"])
    attack_mae = float(selected_attempt["attack_mae"])

    original_img_np = image_tensor.permute(1, 2, 0).numpy()
    adversarial_img_np = adversarial_tensor[0].detach().cpu().permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(original_img_np)
    axes[0].set_title(
        f"Original: {true_label}\nPred: {CLASS_NAMES[original_pred]} ({original_conf:.3f})"
    )
    axes[0].axis("off")

    axes[1].imshow(adversarial_img_np)
    axes[1].set_title(
        f"Adversarial\nPred: {CLASS_NAMES[adversarial_pred]} ({adversarial_conf:.3f})\n"
        f"eps={attack_epsilon:.4f} Linf={attack_linf:.4f}"
    )
    axes[1].axis("off")

    fig.suptitle(
        f"Attack successful: {attack_success} | "
        f"strategy={type(attack_strategy).__name__} tries={len(attempts)}"
    )
    fig.tight_layout()
    plt.close(fig)
    print(f"  True label: {true_label}")
    print(f"  Sample index: {sample_index}")
    print(f"  Original prediction: {CLASS_NAMES[original_pred]} (confidence: {original_conf:.3f})")
    print(f"  Adversarial prediction: {CLASS_NAMES[adversarial_pred]} (confidence: {adversarial_conf:.3f})")
    print(f"  Epsilon: {attack_epsilon:.6f}")
    print(f"  Distance: Linf={attack_linf:.6f} RMS={attack_rms:.6f} MAE={attack_mae:.6f}")
    print(f"  Attack successful: {attack_success}")
    return fig


def generate_class_reversal_figure(
    model: nn.Module,
    step_size: float = 0.03,
    num_steps: int = DEFAULT_TARGET_STEPS,
    num_restarts: int = 4,
    strategy: ClassReversalStrategy | None = None,
) -> Figure:
    criterion = nn.CrossEntropyLoss()
    reversal_strategy = strategy or RandomRestartTargetClassStrategy(
        step_size=step_size,
        num_steps=num_steps,
        num_restarts=num_restarts,
    )
    print("Generating class reversal figure from random pixel initialization...")

    fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(6 * len(CLASS_NAMES), 5))
    if len(CLASS_NAMES) == 1:
        axes = [axes]
    context = ClassReversalContext(model=model, criterion=criterion)

    for idx, class_name in enumerate(CLASS_NAMES):
        result = reversal_strategy.run(context, idx)
        generated_tensor = result.image_tensor
        generated_img_np = generated_tensor[0].detach().cpu().permute(1, 2, 0).numpy()

        axes[idx].imshow(generated_img_np)
        axes[idx].set_title(
            f"Target: {class_name}\n"
            f"Pred: {CLASS_NAMES[result.prediction]} ({result.confidence:.3f})\n"
            f"target_conf={result.target_confidence:.3f}"
        )
        axes[idx].axis("off")
        print(
            f"  Target {class_name}: pred={CLASS_NAMES[result.prediction]} "
            f"pred_conf={result.confidence:.3f} target_conf={result.target_confidence:.3f} "
            f"margin={result.margin:.3f} restart={result.restart_index + 1}/{result.num_restarts}"
        )

    fig.suptitle(
        f"Class reversal | strategy={type(reversal_strategy).__name__}"
    )
    fig.tight_layout()
    plt.close(fig)
    return fig


def generate_test_figure(
    model: nn.Module,
    sample_dataset: Dataset,
) -> Figure:
    if len(sample_dataset) == 0:
        raise ValueError("Sample dataset is empty")

    sample_index = random.randrange(len(sample_dataset))
    model.eval()
    image_tensor, label_idx = sample_dataset[sample_index]
    input_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = forward_logits(model, input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).item()
        confidence = probs[0, pred].item()

    class_names = get_dataset_classes(sample_dataset)
    true_label = class_names[label_idx]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_tensor.permute(1, 2, 0).numpy())
    ax.set_title(f"True: {true_label}, Pred: {CLASS_NAMES[pred]} ({confidence:.3f})")
    ax.axis("off")
    fig.tight_layout()
    plt.close(fig)
    return fig
