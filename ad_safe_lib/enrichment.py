from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import Iterable
import gc

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Subset
from torchvision import transforms
from tqdm.auto import tqdm

from .artifacts import release_torch_memory
from .config import DEVICE
from .data import PreparedTrainingDataset, make_data_loader, to_device
from .backbones import forward_logits


@dataclass(frozen=True)
class AdversarialAttackContext:
    model: nn.Module
    x_original: Tensor
    criterion: nn.Module
    target_labels: Tensor


@dataclass(frozen=True)
class BudgetedPgdStrategy:
    epsilon: float = 0.15
    num_steps: int = 5

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
                x_adv = torch.clamp(
                    x_adv,
                    context.x_original - self.epsilon,
                    context.x_original + self.epsilon,
                )
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
                x_adv = x_adv.detach()

        context.model.train(was_training)
        return x_adv


def generate_budgeted_pgd_perturbation(
    *,
    model: nn.Module,
    x_original: Tensor,
    criterion: nn.Module,
    target_labels: Tensor,
    strategy: BudgetedPgdStrategy,
) -> Tensor:
    context = AdversarialAttackContext(
        model=model,
        x_original=x_original,
        criterion=criterion,
        target_labels=target_labels,
    )
    return strategy.run(context)


class EnrichmentStrategy(ABC):
    """Base class for enrichment strategies.

    The runner owns dataset iteration and progress reporting. Strategies only
    transform a provided batch and return derived images paired with source
    positions inside the input subset.
    """

    @abstractmethod
    def generate_batch(
        self,
        *,
        images: Tensor,
        labels: Tensor,
        source_positions: Tensor,
        student_model: nn.Module | None,
        teacher_model: nn.Module | None,
    ) -> Iterable[tuple[int, Tensor]]:
        """Yield ``(source_position, derived_image)`` for one source batch."""
        raise NotImplementedError


class StrictInheritanceStrategy(EnrichmentStrategy):
    """Base for one-to-one transforms that inherit source labels/logits."""

    @abstractmethod
    def transform_batch(self, images: Tensor) -> Tensor:
        """Transform a batch of images (N, C, H, W) on whatever device they are on."""
        raise NotImplementedError

    def generate_batch(
        self,
        *,
        images: Tensor,
        labels: Tensor,
        source_positions: Tensor,
        student_model: nn.Module | None,
        teacher_model: nn.Module | None,
    ) -> Iterable[tuple[int, Tensor]]:
        del labels, student_model, teacher_model
        derived = self.transform_batch(images.to(DEVICE)).detach().cpu()
        for source_pos, image in zip(source_positions, derived, strict=True):
            yield int(source_pos.item()), image


@dataclass(frozen=True)
class HorizontalFlipStrategy(StrictInheritanceStrategy):
    """Horizontal flip (left-right reflection)."""

    def transform_batch(self, images: Tensor) -> Tensor:
        return transforms.functional.hflip(images)


@dataclass(frozen=True)
class VerticalFlipStrategy(StrictInheritanceStrategy):
    """Vertical flip (top-bottom reflection). Opt-in due to label validity risk."""

    def transform_batch(self, images: Tensor) -> Tensor:
        return transforms.functional.vflip(images)


@dataclass(frozen=True)
class ScaleStrategy(StrictInheritanceStrategy):
    """Scale/zoom: multiply image size by a factor in range (factor_min, factor_max)."""

    factor_min: float = 0.9
    factor_max: float = 1.1

    def transform_batch(self, images: Tensor) -> Tensor:
        factor = torch.empty(1).uniform_(self.factor_min, self.factor_max).item()
        _, _, h, w = images.shape
        new_h = max(1, int(h * factor))
        new_w = max(1, int(w * factor))
        resized = transforms.functional.resize(images, [new_h, new_w])
        if factor > 1.0:
            return transforms.functional.center_crop(resized, [h, w])
        h_pad = h - new_h
        w_pad = w - new_w
        top_pad = h_pad // 2
        left_pad = w_pad // 2
        return transforms.functional.pad(resized, [left_pad, top_pad, w_pad - left_pad, h_pad - top_pad])


@dataclass(frozen=True)
class GaussianBlurStrategy(StrictInheritanceStrategy):
    """Gaussian blur with kernel size and sigma range."""

    kernel_size: int = 5
    sigma_min: float = 0.1
    sigma_max: float = 2.0

    def transform_batch(self, images: Tensor) -> Tensor:
        sigma = torch.empty(1).uniform_(self.sigma_min, self.sigma_max).item()
        return transforms.functional.gaussian_blur(images, kernel_size=self.kernel_size, sigma=sigma)


@dataclass(frozen=True)
class PerspectiveStrategy(StrictInheritanceStrategy):
    """Mild perspective distortion."""

    distortion_scale: float = 0.2

    def transform_batch(self, images: Tensor) -> Tensor:
        return transforms.RandomPerspective(distortion_scale=self.distortion_scale, p=1.0)(images)


@dataclass(frozen=True)
class RotateStrategy(StrictInheritanceStrategy):
    """Rotation by specified angles. Opt-in due to label validity risk in ads."""

    angles: tuple[int, ...] = (90, 180, 270)

    def transform_batch(self, images: Tensor) -> Tensor:
        angle = self.angles[torch.randint(0, len(self.angles), (1,)).item()]
        return transforms.functional.rotate(images, angle)


@dataclass(frozen=True)
class GrayscaleStrategy(StrictInheritanceStrategy):
    """Convert to grayscale. Opt-in due to label validity risk in colorful ads."""

    def transform_batch(self, images: Tensor) -> Tensor:
        return transforms.functional.rgb_to_grayscale(images, num_output_channels=3)


@dataclass(frozen=True)
class BrightnessStrategy(StrictInheritanceStrategy):
    """Random brightness adjustment. Simulates screen calibration variance in ad delivery."""

    factor_min: float = 0.6
    factor_max: float = 1.4

    def transform_batch(self, images: Tensor) -> Tensor:
        factor = torch.empty(1).uniform_(self.factor_min, self.factor_max).item()
        return transforms.functional.adjust_brightness(images, factor)


@dataclass(frozen=True)
class ContrastStrategy(StrictInheritanceStrategy):
    """Random contrast adjustment. Simulates compression artifacts and display variation."""

    factor_min: float = 0.6
    factor_max: float = 1.4

    def transform_batch(self, images: Tensor) -> Tensor:
        factor = torch.empty(1).uniform_(self.factor_min, self.factor_max).item()
        return transforms.functional.adjust_contrast(images, factor)


@dataclass(frozen=True)
class SaturationStrategy(StrictInheritanceStrategy):
    """Random saturation adjustment. Ads vary heavily in color processing across pipelines."""

    factor_min: float = 0.5
    factor_max: float = 1.5

    def transform_batch(self, images: Tensor) -> Tensor:
        factor = torch.empty(1).uniform_(self.factor_min, self.factor_max).item()
        return transforms.functional.adjust_saturation(images, factor)


@dataclass(frozen=True)
class SharpnessStrategy(StrictInheritanceStrategy):
    """Random sharpness adjustment. Simulates downscaling and compression in ad delivery."""

    factor_min: float = 0.0
    factor_max: float = 2.0

    def transform_batch(self, images: Tensor) -> Tensor:
        factor = torch.empty(1).uniform_(self.factor_min, self.factor_max).item()
        return transforms.functional.adjust_sharpness(images, factor)


@dataclass(frozen=True)
class RandomErasingStrategy(StrictInheritanceStrategy):
    """Erase a random rectangle patch. Simulates occlusion in banner and overlay placements."""

    scale_min: float = 0.02
    scale_max: float = 0.2
    ratio_min: float = 0.3
    ratio_max: float = 3.3

    def transform_batch(self, images: Tensor) -> Tensor:
        eraser = transforms.RandomErasing(
            p=1.0,
            scale=(self.scale_min, self.scale_max),
            ratio=(self.ratio_min, self.ratio_max),
            value=0,
        )
        return torch.stack([eraser(image) for image in images])


@dataclass(frozen=True)
class AdversarialStrategy(EnrichmentStrategy):
    """Adversarial perturbation using PGD attack."""

    epsilon: float = 0.05
    steps: int = 5

    def generate_batch(
        self,
        *,
        images: Tensor,
        labels: Tensor,
        source_positions: Tensor,
        student_model: nn.Module | None,
        teacher_model: nn.Module | None,
    ) -> Iterable[tuple[int, Tensor]]:
        """Yield successful adversarial samples for one source batch."""
        del teacher_model
        if student_model is None:
            raise ValueError("AdversarialStrategy requires student_model")

        student_model.eval()
        criterion = nn.CrossEntropyLoss()

        images, labels = to_device(images, labels)
        source_positions = source_positions.to(labels.device)

        with torch.inference_mode():
            logits = forward_logits(student_model, images)
            predictions = logits.argmax(dim=1)
        correct_mask = predictions == labels
        if not correct_mask.any():
            return

        correct_images = images[correct_mask]
        correct_labels = labels[correct_mask]
        correct_source_positions = source_positions[correct_mask]

        perturbed = generate_budgeted_pgd_perturbation(
            model=student_model,
            x_original=correct_images,
            criterion=criterion,
            target_labels=correct_labels,
            strategy=BudgetedPgdStrategy(epsilon=self.epsilon, num_steps=self.steps),
        )
        with torch.inference_mode():
            adv_logits = forward_logits(student_model, perturbed)
            adv_predictions = adv_logits.argmax(dim=1)
        successful_mask = adv_predictions != correct_labels

        for image, source_pos in zip(
            perturbed[successful_mask],
            correct_source_positions[successful_mask],
            strict=True,
        ):
            yield int(source_pos.item()), image.detach().cpu()


@dataclass(frozen=True)
class EnrichmentPhaseSpec:
    """Specification of one enrichment phase (one strategy applied)."""

    strategy: EnrichmentStrategy


@dataclass(frozen=True)
class EnrichmentJobSpec:
    """Specification of an enrichment job (sequence of phases to apply)."""

    phases: tuple[EnrichmentPhaseSpec, ...]
    input_replay_fraction: float = 1.0


@dataclass(frozen=True)
class EnrichmentRunReport:
    input_count: int
    replayed_input_count: int
    derived_sample_count: int
    output_count: int
    input_replay_fraction: float


def _get_subset_teacher_logits(subset: Subset, source_pos: int) -> Tensor | None:
    source_index = int(subset.indices[source_pos])
    if hasattr(subset.dataset, "get_teacher_logits"):
        return subset.dataset.get_teacher_logits(source_index)
    return None


def _select_replay_indices(sample_count: int, replay_fraction: float, seed: int) -> list[int]:
    if replay_fraction < 0 or replay_fraction > 1:
        raise ValueError("input_replay_fraction must be in the range [0, 1]")
    if replay_fraction == 1.0:
        return list(range(sample_count))
    if replay_fraction == 0.0:
        return []

    replay_count = min(sample_count, int(round(sample_count * replay_fraction)))
    if replay_count <= 0:
        return []
    generator = torch.Generator()
    generator.manual_seed(seed)
    replay_indices = torch.randperm(sample_count, generator=generator)[:replay_count].tolist()
    replay_indices.sort()
    return replay_indices


def _phase_source_positions(batch_start: int, batch_size: int) -> Tensor:
    return torch.arange(batch_start, batch_start + batch_size, dtype=torch.long)


def _run_enrichment_phase(
    *,
    phase_idx: int,
    strategy: EnrichmentStrategy,
    subset: Subset,
    prepared: PreparedTrainingDataset,
    student_model: nn.Module | None,
    teacher_model: nn.Module | None,
    teacher_logits: torch.Tensor | None,
    batch_size: int = 32,
) -> int:
    data_loader, = make_data_loader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
    )
    generated_count = 0
    seen_count = 0
    strategy_name = strategy.__class__.__name__
    progress = tqdm(total=len(subset), desc=f"Phase {phase_idx}: {strategy_name}", leave=False)
    try:
        for batch in data_loader:
            images, labels = batch[0], batch[1]
            current_batch_size = images.shape[0]
            source_positions = _phase_source_positions(seen_count, current_batch_size)
            source_labels = {
                int(source_pos.item()): int(label.item())
                for source_pos, label in zip(source_positions, labels, strict=True)
            }

            for source_pos, derived_image in strategy.generate_batch(
                images=images,
                labels=labels,
                source_positions=source_positions,
                student_model=student_model,
                teacher_model=teacher_model,
            ):
                source_teacher_logits = (
                    _get_subset_teacher_logits(subset, source_pos)
                    if teacher_logits is not None
                    else None
                )
                prepared.add_sample(
                    image=derived_image.detach().cpu(),
                    label=source_labels[source_pos],
                    teacher_logits=source_teacher_logits,
                )
                generated_count += 1

            seen_count += current_batch_size
            progress.update(current_batch_size)
    finally:
        progress.close()

    return generated_count


def run_enrichment_job(
    job: EnrichmentJobSpec,
    subset: Subset,
    student_model: nn.Module | None,
    teacher_model: nn.Module | None,
    teacher_logits: torch.Tensor | None,
    *,
    replay_seed: int,
) -> tuple[PreparedTrainingDataset, EnrichmentRunReport]:
    """Execute one enrichment job on a dataset subset.
    
    All phases read the same subset snapshot. New samples accumulate.
    All derived samples inherit their source sample's label and teacher logits.
    
    Args:
        job: EnrichmentJobSpec defining phases.
        subset: Input dataset subset.
        student_model: Student model (passed to all strategies).
        teacher_model: Teacher model (passed to all strategies).
        teacher_logits: Precomputed teacher logits for the base dataset.
        
    Returns:
        PreparedTrainingDataset with original samples + all derived samples.
    """
    replay_indices = _select_replay_indices(len(subset), job.input_replay_fraction, replay_seed)
    prepared = PreparedTrainingDataset(subset, base_indices=replay_indices)
    if teacher_logits is not None and hasattr(subset.dataset, "get_teacher_logits"):
        subset_logits_items: list[Tensor] = []
        for source_pos in replay_indices:
            source_logits = _get_subset_teacher_logits(subset, source_pos)
            if source_logits is None:
                subset_logits_items = []
                break
            subset_logits_items.append(source_logits)
        if subset_logits_items:
            prepared.set_teacher_logits(torch.stack(subset_logits_items))

    print(
        "  Replay input samples: "
        f"{len(replay_indices)}/{len(subset)} "
        f"(fraction={job.input_replay_fraction:.3f})"
    )

    total_added = 0
    for phase_idx, phase in enumerate(job.phases):
        generated_count = _run_enrichment_phase(
            phase_idx=phase_idx,
            strategy=phase.strategy,
            subset=subset,
            prepared=prepared,
            student_model=student_model,
            teacher_model=teacher_model,
            teacher_logits=teacher_logits,
        )
        total_added += generated_count
        print(
            f"  Phase {phase_idx}: "
            f"{phase.strategy.__class__.__name__} generated {generated_count} samples"
        )
        gc.collect()
        release_torch_memory()

    if len(prepared) == 0:
        raise ValueError("Enrichment job produced an empty training dataset. Increase input_replay_fraction.")

    report = EnrichmentRunReport(
        input_count=len(subset),
        replayed_input_count=len(replay_indices),
        derived_sample_count=total_added,
        output_count=len(prepared),
        input_replay_fraction=job.input_replay_fraction,
    )
    print(
        "Job enrichment complete: "
        f"input={report.input_count}, "
        f"replayed={report.replayed_input_count}, "
        f"derived={report.derived_sample_count}, "
        f"output={report.output_count}, "
        f"multiplier={report.output_count / max(1, report.input_count):.3f}x"
    )
    return prepared, report


def run_enrichment_jobs(
    jobs: tuple[EnrichmentJobSpec, ...],
    subset: Subset,
    student_model: nn.Module | None,
    teacher_model: nn.Module | None,
    teacher_logits: torch.Tensor | None,
    *,
    replay_seed: int,
) -> tuple[PreparedTrainingDataset, tuple[EnrichmentRunReport, ...]]:
    """Execute a sequence of enrichment jobs, chaining outputs.
    
    Job N's output becomes job N+1's input.
    
    Args:
        jobs: Tuple of EnrichmentJobSpec to execute in order.
        subset: Initial input subset.
        student_model: Student model.
        teacher_model: Teacher model.
        teacher_logits: Teacher logits for the base dataset.
        
    Returns:
        Final enriched PreparedTrainingDataset after all jobs.
    """
    current: Subset = subset
    result: PreparedTrainingDataset | None = None
    reports: list[EnrichmentRunReport] = []
    for job_idx, job in enumerate(jobs):
        print(f"\nEnrichment job {job_idx}:")
        result, report = run_enrichment_job(
            job,
            current,
            student_model,
            teacher_model,
            teacher_logits,
            replay_seed=replay_seed + job_idx,
        )
        reports.append(report)
        current = Subset(result, list(range(len(result))))
        gc.collect()
        release_torch_memory()

    if result is None:
        return PreparedTrainingDataset(subset), ()
    return result, tuple(reports)
