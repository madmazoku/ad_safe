from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn
from torchvision.transforms import functional as TF

from .backbones import forward_logits
from .config import DEFAULT_TARGET_STEPS, DEVICE, IMAGE_SIZE


@dataclass(frozen=True)
class ClassReversalResult:
    image_tensor: Tensor
    target_label: int
    prediction: int
    confidence: float
    target_confidence: float
    margin: float
    restart_index: int
    num_restarts: int


@dataclass(frozen=True)
class ClassReversalContext:
    model: nn.Module
    criterion: nn.Module

    def make_random_image_tensor(self) -> Tensor:
        return torch.rand((1, 3, IMAGE_SIZE, IMAGE_SIZE), device=DEVICE)

    def score(self, image_tensor: Tensor, target_label: int) -> tuple[int, float, float, float]:
        with torch.inference_mode():
            logits = forward_logits(self.model, image_tensor)
            probs = torch.softmax(logits, dim=1)
            prediction = int(logits.argmax(dim=1).item())
            confidence = float(probs[0, prediction].item())
            target_confidence = float(probs[0, target_label].item())
            other_class_indices = [
                class_index
                for class_index in range(logits.shape[1])
                if class_index != target_label
            ]
            if other_class_indices:
                margin = float(
                    (
                        logits[:, target_label]
                        - torch.logsumexp(logits[:, other_class_indices], dim=1)
                    ).item()
                )
            else:
                margin = float(logits[:, target_label].item())
        return prediction, confidence, target_confidence, margin


class ClassReversalStrategy(Protocol):
    def run(self, context: ClassReversalContext, target_label: int) -> ClassReversalResult:
        raise NotImplementedError


@dataclass(frozen=True)
class RandomRestartTargetClassStrategy:
    step_size: float = 0.03
    num_steps: int = DEFAULT_TARGET_STEPS
    num_restarts: int = 4
    blur_every_steps: int = 12
    range_normalize: bool = True

    def run(self, context: ClassReversalContext, target_label: int) -> ClassReversalResult:
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if self.num_restarts <= 0:
            raise ValueError("num_restarts must be positive")
        if self.blur_every_steps < 0:
            raise ValueError("blur_every_steps must be non-negative")

        best_result: ClassReversalResult | None = None
        for restart_index in range(self.num_restarts):
            generated_tensor = self._generate_from_initial(
                context=context,
                x_initial=context.make_random_image_tensor(),
                target_label=target_label,
            )
            prediction, confidence, target_confidence, margin = context.score(
                generated_tensor,
                target_label,
            )
            candidate = ClassReversalResult(
                image_tensor=generated_tensor,
                target_label=target_label,
                prediction=prediction,
                confidence=confidence,
                target_confidence=target_confidence,
                margin=margin,
                restart_index=restart_index,
                num_restarts=self.num_restarts,
            )
            if best_result is None or self._result_key(candidate) > self._result_key(best_result):
                best_result = candidate

        if best_result is None:
            raise RuntimeError("Class reversal produced no candidates")
        return best_result

    def _generate_from_initial(
        self,
        *,
        context: ClassReversalContext,
        x_initial: Tensor,
        target_label: int,
    ) -> Tensor:
        x_target = TF.gaussian_blur(x_initial.detach().clone(), kernel_size=9)
        target_labels = torch.tensor([target_label], device=x_initial.device)
        was_training = context.model.training
        context.model.eval()

        try:
            for step_index in range(self.num_steps):
                x_target.requires_grad_(True)
                context.model.zero_grad(set_to_none=True)
                logits = forward_logits(context.model, x_target)
                target_logit = logits[:, target_label]
                other_class_indices = [
                    class_index
                    for class_index in range(logits.shape[1])
                    if class_index != target_label
                ]
                if other_class_indices:
                    other_logit = torch.logsumexp(logits[:, other_class_indices], dim=1)
                    margin_loss = -(target_logit - other_logit).mean()
                else:
                    margin_loss = -target_logit.mean()
                loss = 0.5 * context.criterion(logits, target_labels) + 0.5 * margin_loss
                grad = torch.autograd.grad(loss, x_target)[0]

                with torch.no_grad():
                    x_target = x_target - self.step_size * grad.sign()
                    x_target = torch.clamp(x_target, 0.0, 1.0)
                    if self.blur_every_steps and (step_index + 1) % self.blur_every_steps == 0:
                        x_target = TF.gaussian_blur(x_target, kernel_size=5)
                    if self.range_normalize:
                        x_target = self._normalize_channel_range(x_target)
                    x_target = x_target.detach()
        finally:
            context.model.train(was_training)

        return x_target

    @staticmethod
    def _normalize_channel_range(image_tensor: Tensor) -> Tensor:
        channel_min = image_tensor.amin(dim=(2, 3), keepdim=True)
        channel_max = image_tensor.amax(dim=(2, 3), keepdim=True)
        channel_range = channel_max - channel_min
        return torch.where(
            channel_range > 1e-6,
            (image_tensor - channel_min) / channel_range,
            image_tensor,
        )

    @staticmethod
    def _result_key(result: ClassReversalResult) -> tuple[bool, float, float]:
        return (
            result.prediction == result.target_label,
            result.target_confidence,
            result.margin,
        )
