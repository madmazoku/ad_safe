from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Subset
from torchvision import transforms

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
    """Base class for all enrichment strategies.
    Defines the contract for generating derived samples from a dataset subset.
    """

    @abstractmethod
    def run(
        self,
        subset: Subset,
        student_model: nn.Module | None,
        teacher_model: nn.Module | None,
    ) -> list[tuple[int, Tensor]]:
        """Generate derived sample images from the input subset.
        
        Args:
            subset: Dataset subset to generate samples from.
            student_model: Student model (may be None if not needed by strategy).
            teacher_model: Teacher model (may be None if not needed by strategy).
            
        Returns:
            List of derived image tensors. Caller assigns labels and teacher logits
            from corresponding source samples.
        """
        raise NotImplementedError


class StrictInheritanceStrategy(EnrichmentStrategy):
    """Base for v1 strategies: returns only image tensors.
    Executor always inherits label and teacher logits from source sample.
    Concrete strategies inherit from this class and implement run().
    """

    def _iter_source_samples(self, subset: Subset) -> Iterable[tuple[int, Tensor]]:
        """Yield (source_position_in_subset, image_tensor)."""
        for source_pos in range(len(subset)):
            item = subset[source_pos]
            image = item[0]
            yield source_pos, image


@dataclass(frozen=True)
class HorizontalFlipStrategy(StrictInheritanceStrategy):
    """Horizontal flip (left-right reflection)."""

    def run(
        self,
        subset: Subset,
        student_model: nn.Module | None,
        teacher_model: nn.Module | None,
    ) -> list[tuple[int, Tensor]]:
        flip_transform = transforms.RandomHorizontalFlip(p=1.0)
        return [
            (source_pos, flip_transform(img))
            for source_pos, img in self._iter_source_samples(subset)
        ]


@dataclass(frozen=True)
class VerticalFlipStrategy(StrictInheritanceStrategy):
    """Vertical flip (top-bottom reflection). Opt-in due to label validity risk."""

    def run(
        self,
        subset: Subset,
        student_model: nn.Module | None,
        teacher_model: nn.Module | None,
    ) -> list[tuple[int, Tensor]]:
        flip_transform = transforms.RandomVerticalFlip(p=1.0)
        return [
            (source_pos, flip_transform(img))
            for source_pos, img in self._iter_source_samples(subset)
        ]


@dataclass(frozen=True)
class ScaleStrategy(StrictInheritanceStrategy):
    """Scale/zoom: multiply image size by a factor in range (factor_min, factor_max)."""

    factor_min: float = 0.9
    factor_max: float = 1.1

    def run(
        self,
        subset: Subset,
        student_model: nn.Module | None,
        teacher_model: nn.Module | None,
    ) -> list[tuple[int, Tensor]]:
        result = []
        for source_pos, img in self._iter_source_samples(subset):
            # img is shape (C, H, W)
            factor = torch.empty(1).uniform_(self.factor_min, self.factor_max).item()
            new_h = max(1, int(img.shape[1] * factor))
            new_w = max(1, int(img.shape[2] * factor))
            resizer = transforms.Resize((new_h, new_w))
            scaled = resizer(img)
            # Crop back to original size if scaled up, pad back if scaled down
            if factor > 1.0:
                restore_transform = transforms.CenterCrop((img.shape[1], img.shape[2]))
            else:
                h_pad = img.shape[1] - new_h
                w_pad = img.shape[2] - new_w
                top_pad = h_pad // 2
                left_pad = w_pad // 2
                restore_transform = transforms.Pad(
                    (left_pad, top_pad, w_pad - left_pad, h_pad - top_pad)
                )
            result.append((source_pos, restore_transform(scaled)))
        return result


@dataclass(frozen=True)
class GaussianBlurStrategy(StrictInheritanceStrategy):
    """Gaussian blur with kernel size and sigma range."""

    kernel_size: int = 5
    sigma_min: float = 0.1
    sigma_max: float = 2.0

    def run(
        self,
        subset: Subset,
        student_model: nn.Module | None,
        teacher_model: nn.Module | None,
    ) -> list[tuple[int, Tensor]]:
        result = []
        for source_pos, img in self._iter_source_samples(subset):
            sigma = torch.empty(1).uniform_(self.sigma_min, self.sigma_max).item()
            blur = transforms.GaussianBlur(kernel_size=self.kernel_size, sigma=(sigma, sigma))
            result.append((source_pos, blur(img)))
        return result


@dataclass(frozen=True)
class PerspectiveStrategy(StrictInheritanceStrategy):
    """Mild perspective distortion."""

    distortion_scale: float = 0.2

    def run(
        self,
        subset: Subset,
        student_model: nn.Module | None,
        teacher_model: nn.Module | None,
    ) -> list[tuple[int, Tensor]]:
        perspective = transforms.RandomPerspective(distortion_scale=self.distortion_scale, p=1.0)
        return [
            (source_pos, perspective(img))
            for source_pos, img in self._iter_source_samples(subset)
        ]


@dataclass(frozen=True)
class RotateStrategy(StrictInheritanceStrategy):
    """Rotation by specified angles. Opt-in due to label validity risk in ads."""

    angles: tuple[int, ...] = (90, 180, 270)

    def run(
        self,
        subset: Subset,
        student_model: nn.Module | None,
        teacher_model: nn.Module | None,
    ) -> list[tuple[int, Tensor]]:
        result = []
        for source_pos, img in self._iter_source_samples(subset):
            angle = self.angles[torch.randint(0, len(self.angles), (1,)).item()]
            rotated = transforms.functional.rotate(img, angle)
            result.append((source_pos, rotated))
        return result


@dataclass(frozen=True)
class GrayscaleStrategy(StrictInheritanceStrategy):
    """Convert to grayscale. Opt-in due to label validity risk in colorful ads."""

    def run(
        self,
        subset: Subset,
        student_model: nn.Module | None,
        teacher_model: nn.Module | None,
    ) -> list[tuple[int, Tensor]]:
        to_gray = transforms.Grayscale(num_output_channels=3)
        return [
            (source_pos, to_gray(img))
            for source_pos, img in self._iter_source_samples(subset)
        ]


@dataclass(frozen=True)
class AdversarialStrategy(StrictInheritanceStrategy):
    """Adversarial perturbation using PGD attack."""

    epsilon: float = 0.05
    steps: int = 5

    def run(
        self,
        subset: Subset,
        student_model: nn.Module | None,
        teacher_model: nn.Module | None,
    ) -> list[tuple[int, Tensor]]:
        """Generate adversarial samples via PGD attack.
        
        Args:
            subset: Input dataset subset.
            student_model: Model to attack (required for adversarial).
            teacher_model: Unused (inherited for interface uniformity).
            
        Returns:
            List of successful adversarial image tensors.
        """
        if student_model is None:
            raise ValueError("AdversarialStrategy requires student_model")

        batch_size = 32
        adversarial_samples: list[tuple[int, Tensor]] = []
        student_model.eval()
        criterion = nn.CrossEntropyLoss()

        data_loader, = make_data_loader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=False,
        )
        seen_count = 0
        for batch in data_loader:
            images, labels = batch[0], batch[1]
            images, labels = to_device(images, labels)
            batch_len = images.shape[0]

            with torch.inference_mode():
                logits = forward_logits(student_model, images)
                predictions = logits.argmax(dim=1)
            correct_mask = predictions == labels
            if not correct_mask.any():
                seen_count += batch_len
                continue

            correct_images = images[correct_mask]
            correct_labels = labels[correct_mask]
            source_positions = torch.arange(
                seen_count,
                seen_count + batch_len,
                device=labels.device,
            )[correct_mask]

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
                source_positions[successful_mask],
                strict=True,
            ):
                adversarial_samples.append((int(source_pos.item()), image.detach().cpu()))

            seen_count += batch_len

        return adversarial_samples


@dataclass(frozen=True)
class EnrichmentPhaseSpec:
    """Specification of one enrichment phase (one strategy applied)."""

    strategy: EnrichmentStrategy


@dataclass(frozen=True)
class EnrichmentJobSpec:
    """Specification of an enrichment job (sequence of phases to apply)."""

    phases: tuple[EnrichmentPhaseSpec, ...]


def run_enrichment_job(
    job: EnrichmentJobSpec,
    subset: Subset,
    student_model: nn.Module | None,
    teacher_model: nn.Module | None,
    teacher_logits: torch.Tensor | None,
) -> PreparedTrainingDataset:
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
    prepared = PreparedTrainingDataset(subset)
    if teacher_logits is not None and hasattr(subset.dataset, "get_teacher_logits"):
        subset_logits_items: list[Tensor] = []
        for source_index in subset.indices:
            source_logits = subset.dataset.get_teacher_logits(int(source_index))
            if source_logits is None:
                subset_logits_items = []
                break
            subset_logits_items.append(source_logits)
        if subset_logits_items:
            prepared.set_teacher_logits(torch.stack(subset_logits_items))

    # Process each phase
    total_added = 0
    for phase_idx, phase in enumerate(job.phases):
        phase_samples = phase.strategy.run(subset, student_model, teacher_model)
        print(f"  Phase {phase_idx}: {phase.strategy.__class__.__name__} generated {len(phase_samples)} samples")

        for source_pos, derived_image in phase_samples:
            source_item = subset[source_pos]
            source_label = int(source_item[1])
            source_teacher_logits = (
                prepared.get_teacher_logits(source_pos)
                if teacher_logits is not None
                else None
            )

            prepared.add_sample(
                image=derived_image.detach().cpu(),
                label=source_label,
                teacher_logits=source_teacher_logits,
            )
            total_added += 1

    print(f"Job enrichment complete: {total_added} samples added")
    return prepared


def run_enrichment_jobs(
    jobs: tuple[EnrichmentJobSpec, ...],
    subset: Subset,
    student_model: nn.Module | None,
    teacher_model: nn.Module | None,
    teacher_logits: torch.Tensor | None,
) -> PreparedTrainingDataset:
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
    for job_idx, job in enumerate(jobs):
        print(f"\nEnrichment job {job_idx}:")
        result = run_enrichment_job(job, current, student_model, teacher_model, teacher_logits)
        current = Subset(result, list(range(len(result))))

    if result is None:
        return PreparedTrainingDataset(subset)
    return result
