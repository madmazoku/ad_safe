from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor, nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .artifacts import load_model, release_torch_memory, save_model
from .backbones import forward_logits
from .config import CLASS_NAMES, DEFAULT_ADV_STEPS, DEVICE, TrainingConfig, TrainingHistoryEntry, get_learning_rate_for_split
from .cooldown import EpochEndHandler
from .data import LabeledDatasetView, PreparedTrainingDataset, make_data_loader, split_train_dataset, to_device
from .enrichment import EnrichmentJobSpec, run_enrichment_jobs
from .metrics import DefaultValidationMetricComparator, ValidationComparison, evaluate_validation_score


def compute_distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    temperature: float,
) -> Tensor:
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)


def compute_supervised_training_loss(
    *,
    student_logits: Tensor,
    labels: Tensor,
    criterion: nn.Module,
    config: TrainingConfig,
    teacher_logits: Tensor | None,
) -> tuple[Tensor, Tensor | None]:
    hard_loss = criterion(student_logits, labels)
    if teacher_logits is None:
        return hard_loss, None

    distillation_loss = compute_distillation_loss(
        student_logits,
        teacher_logits,
        temperature=config.distillation_temperature,
    )
    combined_loss = (
        (1.0 - config.distillation_alpha) * hard_loss
        + config.distillation_alpha * distillation_loss
    )
    return combined_loss, distillation_loss


def load_teacher_model(teacher_model_path: str | None) -> nn.Module | None:
    if teacher_model_path is None:
        return None

    teacher_model = load_model(Path(teacher_model_path))
    teacher_model.eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad_(False)
    return teacher_model


def validate_teacher_logits(logits: Tensor, batch_size: int) -> None:
    if logits.ndim != 2 or logits.shape[0] != batch_size or logits.shape[1] != len(CLASS_NAMES):
        raise ValueError(
            "Teacher model must return logits with shape "
            f"(batch, {len(CLASS_NAMES)}), got {tuple(logits.shape)}"
        )


def precompute_teacher_logits(
    *,
    teacher_model_path: str | None,
    dataset: Dataset,
    batch_size: int,
) -> Tensor | None:
    teacher_model = load_teacher_model(teacher_model_path)
    if teacher_model is None:
        return None

    print(f"Precomputing teacher logits: {teacher_model_path}")
    data_loader, = make_data_loader(dataset, batch_size=batch_size, shuffle=False)
    logits_batches: list[Tensor] = []
    teacher_model.eval()

    with torch.inference_mode():
        for images, _ in tqdm(data_loader, desc="Teacher logits", leave=False):
            images, = to_device(images)
            logits = forward_logits(teacher_model, images)
            validate_teacher_logits(logits, images.shape[0])
            logits_batches.append(logits.detach())

    del teacher_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not logits_batches:
        raise ValueError("Cannot precompute teacher logits for an empty training dataset")
    return torch.cat(logits_batches, dim=0).to(DEVICE)


def prepare_training_dataset(
    *,
    full_train_dataset: Dataset,
    config: TrainingConfig,
) -> PreparedTrainingDataset:
    prepared_dataset = PreparedTrainingDataset(full_train_dataset)
    teacher_logits = precompute_teacher_logits(
        teacher_model_path=config.teacher_model_path,
        dataset=full_train_dataset,
        batch_size=config.batch_size,
    )
    prepared_dataset.set_teacher_logits(teacher_logits)

    if teacher_logits is not None:
        print(f"Using cached teacher logits: {config.teacher_model_path}")
        print(f"Distillation alpha: {config.distillation_alpha}")
        print(f"Distillation temperature: {config.distillation_temperature}")

    return prepared_dataset


def train_model_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    config: TrainingConfig,
) -> tuple[float, float | None]:
    model.train()
    total_loss = 0.0
    total_distill_loss = 0.0
    has_teacher_logits = False

    for batch in tqdm(
        train_loader,
        desc="Training",
        leave=False,
    ):
        if config.teacher_model_path is not None:
            images, labels, teacher_logits = batch
            has_teacher_logits = True
            teacher_logits, = to_device(teacher_logits)
        else:
            images, labels = batch
            teacher_logits = None
        images, labels = to_device(images, labels)
        optimizer.zero_grad(set_to_none=True)
        clean_logits = forward_logits(model, images)
        hard_clean_loss = criterion(clean_logits, labels)
        clean_loss, distill_loss = compute_supervised_training_loss(
            student_logits=clean_logits,
            labels=labels,
            criterion=criterion,
            config=config,
            teacher_logits=teacher_logits,
        )
        loss = clean_loss
        if distill_loss is not None:
            total_distill_loss += distill_loss.item()

        loss.backward()
        optimizer.step()
        total_loss += hard_clean_loss.item()

    avg_train_loss = total_loss / len(train_loader)
    avg_distill_train_loss = total_distill_loss / len(train_loader) if has_teacher_logits else None
    return avg_train_loss, avg_distill_train_loss


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    config: TrainingConfig,
    best_model_path: Path,
    split_round: int,
    global_epoch_start: int,
    training_start_time: float,
    epoch_end_handlers: tuple[EpochEndHandler, ...] = (),
) -> tuple[nn.Module, list[TrainingHistoryEntry]]:
    print("Starting training.")
    best_epoch = 0
    patience_counter = 0
    best_model_path = Path(best_model_path)
    comparator = DefaultValidationMetricComparator()
    history: list[TrainingHistoryEntry] = []
    best_metrics = evaluate_validation_score(
        model=model,
        val_loader=val_loader,
    )
    best_score = comparator.score(best_metrics)
    baseline_elapsed = time.time() - training_start_time
    history.append(
        TrainingHistoryEntry(
            split_round=split_round,
            epoch_in_round=0,
            global_epoch=global_epoch_start + 0.5,
            train_loss=None,
            val_accuracy=best_metrics.accuracy,
            val_auc=best_metrics.auc,
            val_nll=best_metrics.nll,
            val_unsafe_recall=best_metrics.unsafe_recall,
            val_avg_wrong_conf=best_metrics.avg_wrong_conf,
            score=best_score,
            elapsed_seconds=baseline_elapsed,
            is_baseline=True,
            is_new_best=False,
        )
    )
    print(f"Initial val metrics: {best_metrics}")

    for epoch in range(1, config.epochs + 1):
        avg_train_loss, avg_distill_train_loss = train_model_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
        )
        val_metrics = evaluate_validation_score(
            model=model,
            val_loader=val_loader,
        )
        score = comparator.score(val_metrics)
        comparison = comparator.compare(val_metrics, best_metrics)
        global_epoch = global_epoch_start + epoch
        elapsed = time.time() - training_start_time
        log_parts = [
            f"Split {split_round}/{config.resplit_runs}",
            f"Epoch {epoch}/{config.epochs}",
            f"global_epoch={global_epoch}",
            f"train={avg_train_loss:.4f}",
            f"val={val_metrics.accuracy:.4f}",
        ]
        if val_metrics.auc is not None:
            log_parts.append(f"auc={val_metrics.auc:.4f}")
        log_parts.append(f"nll={val_metrics.nll:.4f}")
        if avg_distill_train_loss is not None:
            log_parts.append(f"distill_train={avg_distill_train_loss:.4f}")
        log_parts.extend(
            [
                f"best={best_score:.4f} @ epoch {best_epoch}",
                f"comparison={comparison.value}",
                f"time={elapsed:.2f}s",
            ]
        )

        print(" | ".join(log_parts))

        is_new_best = comparison == ValidationComparison.BETTER
        entry = TrainingHistoryEntry(
            split_round=split_round,
            epoch_in_round=epoch,
            global_epoch=global_epoch,
            train_loss=avg_train_loss,
            val_accuracy=val_metrics.accuracy,
            val_auc=val_metrics.auc,
            val_nll=val_metrics.nll,
            val_unsafe_recall=val_metrics.unsafe_recall,
            val_avg_wrong_conf=val_metrics.avg_wrong_conf,
            score=score,
            elapsed_seconds=elapsed,
            is_baseline=False,
            is_new_best=is_new_best,
        )
        history.append(entry)

        if is_new_best:
            best_metrics = val_metrics
            best_score = score
            best_epoch = epoch
            patience_counter = 0
            save_model(model, best_model_path)
        elif comparison == ValidationComparison.EQUIVALENT:
            patience_counter = 0
        else:
            patience_counter += 1

        for handler in epoch_end_handlers:
            handler.on_epoch_end(entry)

        if patience_counter >= config.patience:
            print(f"Early stopping at epoch {epoch} (patience={config.patience})")
            break

    if patience_counter > 0:
        model = load_model(best_model_path)
        if best_epoch == 0:
            print(
                f"Loaded best baseline model "
                f"(score: {best_score:.4f})"
            )
        else:
            print(
                f"Loaded best model from epoch {best_epoch} "
                f"(score: {best_score:.4f})"
            )
    else:
        if best_epoch == 0:
            print(
                f"Best baseline model already in memory "
                f"(score: {best_score:.4f})"
            )
        else:
            print(
                f"Best model already in memory from epoch {best_epoch} "
                f"(score: {best_score:.4f})"
            )

    return model, history


class AdversarialAttackStrategy(Protocol):
    def run_with_result(self, context: "AdversarialAttackContext") -> AdversarialPerturbationResult:
        raise NotImplementedError


@dataclass(frozen=True)
class AdversarialPerturbationResult:
    adversarial_tensor: Tensor
    epsilon: float
    attack_success: bool
    original_prediction: int
    original_confidence: float
    original_true_confidence: float
    prediction: int
    confidence: float
    true_confidence: float
    linf: float
    rms: float
    mae: float


@dataclass(frozen=True)
class AdversarialAttackContext:
    model: nn.Module
    x_original: Tensor
    criterion: nn.Module
    target_labels: Tensor

    def require_single_sample(self) -> int:
        if self.x_original.shape[0] != 1 or self.target_labels.shape[0] != 1:
            raise ValueError("AdversarialAttackResult expects a single sample")
        return int(self.target_labels[0].item())

    def predict_single(self, sample_tensor: Tensor, true_label: int) -> tuple[int, float, float]:
        with torch.inference_mode():
            logits = forward_logits(self.model, sample_tensor)
            probs = torch.softmax(logits, dim=1)
            prediction = int(logits.argmax(dim=1).item())
            confidence = float(probs[0, prediction].item())
            true_confidence = float(probs[0, true_label].item())
        return prediction, confidence, true_confidence

    def make_result(
        self,
        *,
        adversarial_tensor: Tensor,
        epsilon: float,
        attack_success: bool,
        original_prediction: int,
        original_confidence: float,
        original_true_confidence: float,
        prediction: int,
        confidence: float,
        true_confidence: float,
    ) -> AdversarialPerturbationResult:
        delta = (adversarial_tensor - self.x_original).detach()
        return AdversarialPerturbationResult(
            adversarial_tensor=adversarial_tensor,
            epsilon=epsilon,
            attack_success=attack_success,
            original_prediction=original_prediction,
            original_confidence=original_confidence,
            original_true_confidence=original_true_confidence,
            prediction=prediction,
            confidence=confidence,
            true_confidence=true_confidence,
            linf=float(delta.abs().max().item()),
            rms=float(torch.sqrt(torch.mean(delta * delta)).item()),
            mae=float(delta.abs().mean().item()),
        )

    def describe_attack(self, *, adversarial_tensor: Tensor, epsilon: float) -> AdversarialPerturbationResult:
        true_label = self.require_single_sample()
        original_prediction, original_confidence, original_true_confidence = self.predict_single(
            self.x_original,
            true_label,
        )
        prediction, confidence, true_confidence = self.predict_single(
            adversarial_tensor,
            true_label,
        )
        return self.make_result(
            adversarial_tensor=adversarial_tensor,
            epsilon=epsilon,
            attack_success=original_prediction == true_label and prediction != true_label,
            original_prediction=original_prediction,
            original_confidence=original_confidence,
            original_true_confidence=original_true_confidence,
            prediction=prediction,
            confidence=confidence,
            true_confidence=true_confidence,
        )


@dataclass(frozen=True)
class BudgetedPgdStrategy:
    epsilon: float = 0.15
    num_steps: int = DEFAULT_ADV_STEPS

    def run(self, context: AdversarialAttackContext) -> Tensor:
        if self.epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")

        x_adv = context.x_original.detach().clone()
        step_size = self.epsilon / self.num_steps
        was_training = context.model.training
        context.model.eval()

        for _ in range(self.num_steps):
            x_adv.requires_grad_(True)
            context.model.zero_grad(set_to_none=True)
            logits = forward_logits(context.model, x_adv)
            loss = context.criterion(logits, context.target_labels)
            grad = torch.autograd.grad(loss, x_adv)[0]

            with torch.no_grad():
                x_adv = x_adv + step_size * grad.sign()
                x_adv = torch.clamp(x_adv, context.x_original - self.epsilon, context.x_original + self.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
                x_adv = x_adv.detach()

        context.model.train(was_training)
        return x_adv

    def run_with_result(self, context: AdversarialAttackContext) -> AdversarialPerturbationResult:
        adversarial_tensor = self.run(context)
        return context.describe_attack(
            adversarial_tensor=adversarial_tensor,
            epsilon=self.epsilon,
        )


@dataclass(frozen=True)
class MinimalFlipPgdStrategy:
    max_epsilon: float = 0.15
    num_steps: int = DEFAULT_ADV_STEPS
    search_steps: int = 10
    refinement_steps: int = 6

    def run(self, context: AdversarialAttackContext) -> Tensor:
        return self.run_with_result(context).adversarial_tensor

    def run_with_result(self, context: AdversarialAttackContext) -> AdversarialPerturbationResult:
        if self.max_epsilon <= 0:
            raise ValueError("max_epsilon must be positive")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if self.search_steps <= 0:
            raise ValueError("search_steps must be positive")
        if self.refinement_steps < 0:
            raise ValueError("refinement_steps must be non-negative")

        true_label = context.require_single_sample()
        original_prediction, original_confidence, original_true_confidence = context.predict_single(
            context.x_original,
            true_label,
        )
        if original_prediction != true_label:
            return context.make_result(
                adversarial_tensor=context.x_original.detach().clone(),
                epsilon=0.0,
                attack_success=False,
                original_prediction=original_prediction,
                original_confidence=original_confidence,
                original_true_confidence=original_true_confidence,
                prediction=original_prediction,
                confidence=original_confidence,
                true_confidence=original_true_confidence,
            )

        def attack_at(epsilon: float) -> tuple[Tensor, int, float, float]:
            adversarial_tensor = BudgetedPgdStrategy(
                epsilon=epsilon,
                num_steps=self.num_steps,
            ).run(context)
            prediction, confidence, true_confidence = context.predict_single(
                adversarial_tensor,
                true_label,
            )
            return adversarial_tensor, prediction, confidence, true_confidence

        probe_epsilons = [
            self.max_epsilon * (2 ** (probe_index + 1 - self.search_steps))
            for probe_index in range(self.search_steps)
        ]
        probe_epsilons[-1] = self.max_epsilon
        lower_epsilon = 0.0
        upper_epsilon: float | None = None
        upper_result: tuple[Tensor, int, float, float] | None = None

        for epsilon in probe_epsilons:
            result = attack_at(epsilon)
            _, prediction, _, _ = result
            if prediction != true_label:
                upper_epsilon = epsilon
                upper_result = result
                break
            lower_epsilon = epsilon

        if upper_epsilon is None or upper_result is None:
            adversarial_tensor, prediction, confidence, true_confidence = attack_at(self.max_epsilon)
            return context.make_result(
                adversarial_tensor=adversarial_tensor,
                epsilon=self.max_epsilon,
                attack_success=False,
                original_prediction=original_prediction,
                original_confidence=original_confidence,
                original_true_confidence=original_true_confidence,
                prediction=prediction,
                confidence=confidence,
                true_confidence=true_confidence,
            )

        for _ in range(self.refinement_steps):
            mid_epsilon = (lower_epsilon + upper_epsilon) / 2.0
            result = attack_at(mid_epsilon)
            _, prediction, _, _ = result
            if prediction != true_label:
                upper_epsilon = mid_epsilon
                upper_result = result
            else:
                lower_epsilon = mid_epsilon

        adversarial_tensor, prediction, confidence, true_confidence = upper_result
        return context.make_result(
            adversarial_tensor=adversarial_tensor,
            epsilon=upper_epsilon,
            attack_success=True,
            original_prediction=original_prediction,
            original_confidence=original_confidence,
            original_true_confidence=original_true_confidence,
            prediction=prediction,
            confidence=confidence,
            true_confidence=true_confidence,
        )


def generate_adversarial_perturbation(
    model: nn.Module,
    x_original: Tensor,
    criterion: nn.Module,
    target_labels: Tensor | None = None,
    strategy: AdversarialAttackStrategy | None = None,
) -> AdversarialPerturbationResult:
    if target_labels is None:
        raise ValueError("target_labels must be provided for adversarial perturbation")

    attack_strategy = strategy or BudgetedPgdStrategy()
    context = AdversarialAttackContext(
        model=model,
        x_original=x_original,
        criterion=criterion,
        target_labels=target_labels,
    )
    return attack_strategy.run_with_result(context)


def train_model_across_resplits(
    *,
    model: nn.Module,
    full_train_dataset: Dataset,
    config: TrainingConfig,
    best_model_path: Path,
    criterion: nn.Module | None = None,
    training_start_time: float | None = None,
    split_log_prefix: str = "",
    epoch_end_handlers: tuple[EpochEndHandler, ...] = (),
    enrichment_jobs: tuple[EnrichmentJobSpec, ...] = (),
    teacher_model: nn.Module | None = None,
) -> tuple[nn.Module, list[TrainingHistoryEntry]]:
    criterion = nn.CrossEntropyLoss() if criterion is None else criterion
    prepared_train_dataset = prepare_training_dataset(
        full_train_dataset=full_train_dataset,
        config=config,
    )
    training_start_time = time.time() if training_start_time is None else training_start_time
    history: list[TrainingHistoryEntry] = []
    completed_epochs = 0
    split_prefix = f"{split_log_prefix} " if split_log_prefix else ""

    for split_round in range(1, config.resplit_runs + 1):
        print(f"\n{split_prefix}Split round {split_round}/{config.resplit_runs}")
        enriched_train_dataset: PreparedTrainingDataset | None = None
        train_loader: DataLoader | None = None
        val_loader: DataLoader | None = None
        try:
            train_dataset, val_dataset, test_dataset = split_train_dataset(prepared_train_dataset)

            # Apply enrichment to train_dataset if configured
            if enrichment_jobs:
                print(f"Applying enrichment with {len(enrichment_jobs)} jobs...")
                enriched_train_dataset = run_enrichment_jobs(
                    jobs=enrichment_jobs,
                    subset=train_dataset,
                    student_model=model,
                    teacher_model=teacher_model,
                    teacher_logits=prepared_train_dataset.teacher_logits,
                )
                train_dataset = enriched_train_dataset

            train_loader, = make_data_loader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                pin_memory=(
                    False
                    if prepared_train_dataset.teacher_logits is not None
                    and prepared_train_dataset.teacher_logits.device.type == "cuda"
                    else None
                ),
            )
            val_loader, = make_data_loader(
                LabeledDatasetView(val_dataset),
                batch_size=config.batch_size,
                shuffle=False,
            )
            print(f"Train samples: {len(train_dataset)}")
            print(f"Val samples: {len(val_dataset)}")
            print(f"Test samples: {len(test_dataset)}")
            split_learning_rate = get_learning_rate_for_split(config, split_round)
            print(f"Learning rate for split {split_round}: {split_learning_rate}")

            optimizer = optim.Adam(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=split_learning_rate,
            )
            model, round_history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                config=config,
                best_model_path=best_model_path,
                split_round=split_round,
                global_epoch_start=completed_epochs,
                training_start_time=training_start_time,
                epoch_end_handlers=epoch_end_handlers,
            )
            history.extend(round_history)
            completed_epochs += sum(1 for entry in round_history if not entry.is_baseline)
        finally:
            # Explicitly drop large split-specific objects before the next round.
            del train_loader
            del val_loader
            del enriched_train_dataset
            if "train_dataset" in locals():
                del train_dataset
            if "val_dataset" in locals():
                del val_dataset
            if "test_dataset" in locals():
                del test_dataset
            gc.collect()
            release_torch_memory()

    return model, history
