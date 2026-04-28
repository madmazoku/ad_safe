from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from .config import (
    DEFAULT_COOLDOWN_EVERY_EPOCHS,
    DEFAULT_COOLDOWN_SECONDS,
    DEFAULT_GPU_MAX_TEMP,
    DEFAULT_GPU_RESUME_TEMP,
    DEFAULT_GPU_TEMP_CHECK_SECONDS,
    DEVICE,
    TrainingHistoryEntry,
)


@dataclass(frozen=True)
class CooldownConfig:
    every_epochs: int = DEFAULT_COOLDOWN_EVERY_EPOCHS
    seconds: float = DEFAULT_COOLDOWN_SECONDS
    gpu_max_temp: int = DEFAULT_GPU_MAX_TEMP
    gpu_resume_temp: int = DEFAULT_GPU_RESUME_TEMP
    gpu_temp_check_seconds: float = DEFAULT_GPU_TEMP_CHECK_SECONDS

    @property
    def uses_temperature(self) -> bool:
        return self.gpu_max_temp > 0

    @property
    def enabled(self) -> bool:
        return self.every_epochs > 0 or self.uses_temperature

    def to_json(self) -> dict[str, object]:
        return {
            "every_epochs": self.every_epochs,
            "seconds": self.seconds,
            "gpu_max_temp": self.gpu_max_temp,
            "gpu_resume_temp": self.gpu_resume_temp,
            "gpu_temp_check_seconds": self.gpu_temp_check_seconds,
        }


class EpochEndHandler(ABC):
    @abstractmethod
    def on_epoch_end(self, entry: TrainingHistoryEntry) -> None:
        raise NotImplementedError


class CooldownEpochEndHandler(EpochEndHandler):
    def __init__(
        self,
        *,
        config: CooldownConfig,
        backbone_name: str,
        phase_title: str,
    ) -> None:
        self.config = config
        self.backbone_name = backbone_name
        self.phase_title = phase_title
        self._nvml = None
        self._handle = None
        if self.config.uses_temperature:
            if DEVICE.type != "cuda":
                raise RuntimeError("--gpu-max-temp requires CUDA")
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

    def on_epoch_end(self, entry: TrainingHistoryEntry) -> None:
        if not self.config.enabled:
            return

        current_temp = self._read_gpu_temp()
        reasons = []
        global_epoch = int(entry.global_epoch)
        if self.config.every_epochs > 0 and global_epoch % self.config.every_epochs == 0:
            reasons.append(f"global_epoch={global_epoch}")
        if current_temp is not None and current_temp >= self.config.gpu_max_temp:
            reasons.append(f"gpu_temp={current_temp}C")
        if not reasons:
            return

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        started = time.time()
        deadline = started + self.config.seconds
        print(
            f"Cooldown start | {self.backbone_name} {self.phase_title} | "
            f"reason={','.join(reasons)} | max_wait={self.config.seconds:.1f}s"
        )

        while True:
            current_temp = self._read_gpu_temp()
            if current_temp is not None and current_temp <= self.config.gpu_resume_temp:
                break

            remaining_seconds = deadline - time.time()
            if remaining_seconds <= 0:
                break

            if current_temp is None:
                print(f"Cooldown waiting {remaining_seconds:.1f}s")
            else:
                print(
                    f"Cooldown waiting | gpu_temp={current_temp}C | "
                    f"resume_temp={self.config.gpu_resume_temp}C | "
                    f"remaining={remaining_seconds:.1f}s"
                )
            time.sleep(min(self.config.gpu_temp_check_seconds, remaining_seconds))

        final_temp = self._read_gpu_temp()
        waited_seconds = time.time() - started
        temp_part = "unknown" if final_temp is None else f"{final_temp}C"
        print(f"Cooldown end | waited={waited_seconds:.1f}s | gpu_temp={temp_part}")

    def _read_gpu_temp(self) -> int | None:
        if self._nvml is None or self._handle is None:
            return None
        return int(self._nvml.nvmlDeviceGetTemperature(self._handle, self._nvml.NVML_TEMPERATURE_GPU))


def build_cooldown_config(
    *,
    every_epochs: int = DEFAULT_COOLDOWN_EVERY_EPOCHS,
    seconds: float = DEFAULT_COOLDOWN_SECONDS,
    gpu_max_temp: int = DEFAULT_GPU_MAX_TEMP,
    gpu_resume_temp: int = DEFAULT_GPU_RESUME_TEMP,
    gpu_temp_check_seconds: float = DEFAULT_GPU_TEMP_CHECK_SECONDS,
) -> CooldownConfig:
    if gpu_max_temp > 0 and gpu_resume_temp == 0:
        gpu_resume_temp = gpu_max_temp - 5

    config = CooldownConfig(
        every_epochs=int(every_epochs),
        seconds=float(seconds),
        gpu_max_temp=int(gpu_max_temp),
        gpu_resume_temp=int(gpu_resume_temp),
        gpu_temp_check_seconds=float(gpu_temp_check_seconds),
    )
    if config.every_epochs < 0:
        raise ValueError("--cooldown-every-epochs must be non-negative")
    if config.seconds < 0:
        raise ValueError("--cooldown-seconds must be non-negative")
    if config.gpu_max_temp < 0:
        raise ValueError("--gpu-max-temp must be non-negative")
    if config.gpu_resume_temp < 0:
        raise ValueError("--gpu-resume-temp must be non-negative")
    if config.gpu_temp_check_seconds <= 0:
        raise ValueError("--gpu-temp-check-seconds must be positive")
    if config.enabled and config.seconds <= 0:
        raise ValueError("--cooldown-seconds must be positive when cooldown is enabled")
    if config.uses_temperature and config.gpu_resume_temp >= config.gpu_max_temp:
        raise ValueError("--gpu-resume-temp must be lower than --gpu-max-temp")
    return config

