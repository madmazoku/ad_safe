from __future__ import annotations

import json
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from .paths import AD_SAFETY_DATASET_DIR, CHALLENGE_DIR


SCRIPT_DIR = CHALLENGE_DIR
DATA_DIR = AD_SAFETY_DATASET_DIR
CLASS_NAMES = ["safe", "unsafe"]
IMAGE_SIZE = 299
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 3
DEFAULT_RESPLIT_RUNS = 1
DEFAULT_ADV_STEPS = 5
DEFAULT_TARGET_STEPS = 20
DEFAULT_LR = 1e-4
DEFAULT_PATIENCE = 10
DEFAULT_ADV_EPSILON = 0.15
DEFAULT_DISTILLATION_ALPHA = 0.3
DEFAULT_DISTILLATION_TEMPERATURE = 2.0
DEFAULT_COOLDOWN_EVERY_EPOCHS = 0
DEFAULT_COOLDOWN_SECONDS = 0.0
DEFAULT_GPU_MAX_TEMP = 0
DEFAULT_GPU_RESUME_TEMP = 0
DEFAULT_GPU_TEMP_CHECK_SECONDS = 15.0
VALID_RATIO = 0.10
TEST_RATIO = 0.05
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class TrainingConfig:
    base_model: str = "inception_v3"
    epochs: int = DEFAULT_EPOCHS
    patience: int = DEFAULT_PATIENCE
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: tuple[float, ...] = (DEFAULT_LR,)
    learning_rate_multiplier: float = 1.0
    resplit_runs: int = DEFAULT_RESPLIT_RUNS
    unfreeze: tuple[str, ...] = ()
    teacher_model_path: str | None = None
    distillation_alpha: float = DEFAULT_DISTILLATION_ALPHA
    distillation_temperature: float = DEFAULT_DISTILLATION_TEMPERATURE


@dataclass(frozen=True)
class TrainingHistoryEntry:
    split_round: int
    epoch_in_round: int
    global_epoch: float
    train_loss: float | None
    val_accuracy: float
    val_auc: float | None
    val_nll: float | None
    val_unsafe_recall: float | None
    val_avg_wrong_conf: float | None
    score: float
    elapsed_seconds: float
    is_baseline: bool
    is_new_best: bool


def make_seed() -> int:
    return random.SystemRandom().randrange(2**32)


def parse_learning_rates(learning_rate_arg: str) -> tuple[float, ...]:
    learning_rates = tuple(
        float(part.strip()) for part in learning_rate_arg.split(",") if part.strip()
    )
    if not learning_rates:
        raise ValueError("--learning-rate must contain at least one positive value")
    if any(learning_rate <= 0 for learning_rate in learning_rates):
        raise ValueError("--learning-rate values must be positive")
    return learning_rates


def normalize_learning_rates_value(value: object) -> tuple[float, ...]:
    if isinstance(value, str):
        return parse_learning_rates(value)
    if isinstance(value, (int, float)):
        return parse_learning_rates(str(value))
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("learning_rate must contain at least one positive value")
        return parse_learning_rates(",".join(str(item) for item in value))
    raise ValueError("learning_rate must be a number, string, or list of numbers")


def normalize_unfreeze_value(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(block_name.strip() for block_name in value.split(",") if block_name.strip())
    if isinstance(value, (list, tuple)):
        if not all(isinstance(item, str) for item in value):
            raise ValueError("unfreeze must be a list of strings")
        return tuple(item.strip() for item in value if item.strip())
    raise ValueError("unfreeze must be a comma-separated string or a list of strings")


def get_learning_rate_for_split(config: TrainingConfig, split_round: int) -> float:
    if split_round <= 0:
        raise ValueError("split_round must be positive")

    if len(config.learning_rate) == 1:
        return config.learning_rate[0] * (config.learning_rate_multiplier ** (split_round - 1))

    learning_rate_index = min(split_round - 1, len(config.learning_rate) - 1)
    return config.learning_rate[learning_rate_index]


def get_default_batch_size(device: torch.device = DEVICE) -> int:
    if device.type != "cuda" or not torch.cuda.is_available():
        return DEFAULT_BATCH_SIZE

    total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    if total_vram_gb >= 32:
        return 256
    if total_vram_gb >= 16:
        return 128
    if total_vram_gb >= 8:
        return 64
    return DEFAULT_BATCH_SIZE


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_setup_values(setup_path: Path | None) -> dict[str, object]:
    if setup_path is None:
        return {}
    
    try:
        setup_data = json.loads(setup_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Setup JSON is invalid: {setup_path}") from exc

    if not isinstance(setup_data, dict):
        raise ValueError("Setup JSON root must be an object")

    training_config = setup_data.get("training_config", {})
    if training_config is None:
        training_config = {}
    if not isinstance(training_config, dict):
        raise ValueError("training_config must be an object when present")

    cooldown_config = setup_data.get("cooldown", {})
    if cooldown_config is None:
        cooldown_config = {}
    if not isinstance(cooldown_config, dict):
        raise ValueError("cooldown must be an object when present")

    def get_optional_value(container: dict[str, object], field_name: str, expected_type: type) -> object:
        value = container.get(field_name)
        if value is None:
            return None
        if isinstance(expected_type, tuple):
            if not isinstance(value, expected_type):
                raise ValueError(f"{field_name} must be one of {expected_type} or null")
        else:
            if not isinstance(value, expected_type):
                raise ValueError(f"{field_name} must be {expected_type.__name__} or null")
        if expected_type == float or (isinstance(expected_type, tuple) and float in expected_type):
            return float(value)
        return value

    learning_rate_value = training_config.get("learning_rate")
    if learning_rate_value is not None:
        learning_rate_value = normalize_learning_rates_value(learning_rate_value)

    unfreeze_value = training_config.get("unfreeze")
    if unfreeze_value is not None:
        unfreeze_value = normalize_unfreeze_value(unfreeze_value)

    train_fraction = setup_data.get("train_fraction")

    return {
        "seed": get_optional_value(setup_data, "seed", int),
        "train_split": get_optional_value(setup_data, "train_split", str),
        "eval_split": get_optional_value(setup_data, "eval_split", str),
        "train_fraction": None if train_fraction is None else float(train_fraction),
        "base_model": get_optional_value(setup_data, "base_model", str),
        "model_path": get_optional_value(setup_data, "original_model_path", str),
        "epochs": get_optional_value(training_config, "epochs", int),
        "patience": get_optional_value(training_config, "patience", int),
        "batch_size": get_optional_value(training_config, "batch_size", int),
        "learning_rate": learning_rate_value,
        "learning_rate_multiplier": get_optional_value(training_config, "learning_rate_multiplier", (int, float)),
        "resplit_runs": get_optional_value(training_config, "resplit_runs", int),
        "unfreeze": unfreeze_value,
        "teacher_model_path": get_optional_value(training_config, "teacher_model_path", str),
        "distillation_alpha": get_optional_value(training_config, "distillation_alpha", (int, float)),
        "distillation_temperature": get_optional_value(training_config, "distillation_temperature", (int, float)),
        "cooldown_every_epochs": get_optional_value(cooldown_config, "every_epochs", int),
        "cooldown_seconds": get_optional_value(cooldown_config, "seconds", (int, float)),
        "gpu_max_temp": get_optional_value(cooldown_config, "gpu_max_temp", int),
        "gpu_resume_temp": get_optional_value(cooldown_config, "gpu_resume_temp", int),
        "gpu_temp_check_seconds": get_optional_value(cooldown_config, "gpu_temp_check_seconds", (int, float)),
    }


def merge_setup_and_cli_values(cli_values: Mapping[str, object], setup_values: dict[str, object]) -> dict[str, object]:
    def pick_value(name: str, default: object) -> object:
        if name in cli_values:
            return cli_values[name]
        if name in setup_values and setup_values[name] is not None:
            return setup_values[name]
        return default

    return {
        "setup_path": cli_values.get("setup", None),
        "list_base_models": cli_values.get("list_base_models", False),
        "model_path": pick_value("model_path", None),
        "model_path_last": cli_values.get("model_path_last", False),
        "train_split": pick_value("train_split", None),
        "eval_split": pick_value("eval_split", None),
        "train_fraction": pick_value("train_fraction", 1.0),
        "base_model": pick_value("base_model", "inception_v3"),
        "unfreeze_all": pick_value("unfreeze_all", False),
        "unfreeze_top": pick_value("unfreeze_top", 0),
        "unfreeze": pick_value("unfreeze", ()),
        "teacher_model_path": pick_value("teacher_model_path", None),
        "distillation_alpha": pick_value("distillation_alpha", DEFAULT_DISTILLATION_ALPHA),
        "distillation_temperature": pick_value("distillation_temperature", DEFAULT_DISTILLATION_TEMPERATURE),
        "cooldown_every_epochs": pick_value("cooldown_every_epochs", DEFAULT_COOLDOWN_EVERY_EPOCHS),
        "cooldown_seconds": pick_value("cooldown_seconds", DEFAULT_COOLDOWN_SECONDS),
        "gpu_max_temp": pick_value("gpu_max_temp", DEFAULT_GPU_MAX_TEMP),
        "gpu_resume_temp": pick_value("gpu_resume_temp", DEFAULT_GPU_RESUME_TEMP),
        "gpu_temp_check_seconds": pick_value("gpu_temp_check_seconds", DEFAULT_GPU_TEMP_CHECK_SECONDS),
        "epochs": pick_value("epochs", DEFAULT_EPOCHS),
        "batch_size": pick_value("batch_size", None),
        "learning_rate": pick_value("learning_rate", (DEFAULT_LR,)),
        "learning_rate_multiplier": pick_value("learning_rate_multiplier", 1.0),
        "resplit_runs": pick_value("resplit_runs", DEFAULT_RESPLIT_RUNS),
        "patience": pick_value("patience", DEFAULT_PATIENCE),
        "seed": pick_value("seed", None),
    }


def resolve_effective_seed(seed_value: object) -> int:
    if seed_value is None:
        return make_seed()
    if not isinstance(seed_value, int):
        raise ValueError("seed must be an integer or null")
    if seed_value == 0:
        return make_seed()
    if seed_value < 0:
        raise ValueError("seed must be a non-negative integer or null")
    return seed_value


def build_training_config(values: dict[str, object]) -> TrainingConfig:
    batch_size_value = values["batch_size"]
    batch_size = batch_size_value if batch_size_value is not None else get_default_batch_size()
    learning_rates = normalize_learning_rates_value(values["learning_rate"])
    learning_rate_multiplier = float(values["learning_rate_multiplier"])
    explicit_unfreeze = normalize_unfreeze_value(values["unfreeze"])
    epochs = int(values["epochs"])
    teacher_model_path = values["teacher_model_path"]
    distillation_alpha = float(values["distillation_alpha"])
    distillation_temperature = float(values["distillation_temperature"])
    resplit_runs = int(values["resplit_runs"])
    patience = int(values["patience"])
    unfreeze_top = int(values["unfreeze_top"])
    unfreeze_all = bool(values["unfreeze_all"])

    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if epochs <= 0:
        raise ValueError("--epochs must be positive")
    if teacher_model_path is not None and not isinstance(teacher_model_path, str):
        raise ValueError("--teacher-model-path must be a string or null")
    if distillation_alpha < 0 or distillation_alpha > 1:
        raise ValueError("--distillation-alpha must be between 0 and 1")
    if distillation_temperature <= 0:
        raise ValueError("--distillation-temperature must be positive")
    if resplit_runs <= 0:
        raise ValueError("--resplit-runs must be positive")
    if patience < 0:
        raise ValueError("--patience must be non-negative")
    if unfreeze_top < 0:
        raise ValueError("--unfreeze-top must be non-negative")
    if learning_rate_multiplier <= 0:
        raise ValueError("--learning-rate-multiplier must be positive")
    if len(learning_rates) > 1 and values["learning_rate_multiplier"] != 1.0:
        raise ValueError("--learning-rate-multiplier cannot be used with multiple --learning-rate values")
    unfreeze_mode_count = int(unfreeze_all) + int(unfreeze_top > 0) + int(bool(explicit_unfreeze))
    if unfreeze_mode_count > 1:
        raise ValueError("Use only one of --unfreeze-all, --unfreeze-top, or --unfreeze")

    return TrainingConfig(
        base_model=str(values["base_model"]),
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        learning_rate=learning_rates,
        learning_rate_multiplier=learning_rate_multiplier,
        resplit_runs=resplit_runs,
        unfreeze=explicit_unfreeze,
        teacher_model_path=teacher_model_path,
        distillation_alpha=distillation_alpha,
        distillation_temperature=distillation_temperature,
    )


def config_to_json_dict(config: TrainingConfig) -> dict[str, object]:
    return {
        "epochs": config.epochs,
        "patience": config.patience,
        "batch_size": config.batch_size,
        "learning_rate": list(config.learning_rate),
        "learning_rate_multiplier": config.learning_rate_multiplier,
        "resplit_runs": config.resplit_runs,
        "unfreeze": list(config.unfreeze),
        "teacher_model_path": config.teacher_model_path,
        "distillation_alpha": config.distillation_alpha,
        "distillation_temperature": config.distillation_temperature,
    }
