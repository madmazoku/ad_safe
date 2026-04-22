from __future__ import annotations

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
CHALLENGE_DIR = PACKAGE_DIR.parent
DATASETS_DIR = CHALLENGE_DIR / "datasets"
AD_SAFETY_DATASET_DIR = DATASETS_DIR / "ml_bootcamp_adsafety_dataset"
ARTEFACTS_DIR = CHALLENGE_DIR / "artefacts"
AD_SAFE_RUNS_DIR = ARTEFACTS_DIR / "ad_safe_runs"
SWEEP_CONFIGS_DIR = ARTEFACTS_DIR / "sweep_configs"
PROD_MODELS_DIR = ARTEFACTS_DIR / "prod_models"
SMOKE_MODELS_DIR = ARTEFACTS_DIR / "smoke_models"

PATH_SEARCH_ROOTS = (
    Path.cwd(),
    AD_SAFE_RUNS_DIR,
    SWEEP_CONFIGS_DIR,
    ARTEFACTS_DIR,
    CHALLENGE_DIR,
)


def path_for_json(path: Path | None) -> str | None:
    if path is None:
        return None
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(CHALLENGE_DIR).as_posix()
    except ValueError:
        return str(resolved_path)


def resolve_existing_path(path_value: str | Path | None, *, search_roots: tuple[Path, ...] = PATH_SEARCH_ROOTS) -> Path | None:
    if path_value is None:
        return None

    path = Path(path_value)
    if path.is_absolute():
        candidates = [path]
        try:
            challenge_relative = path.resolve().relative_to(CHALLENGE_DIR)
        except ValueError:
            challenge_relative = None
        if challenge_relative is not None:
            candidates.append(AD_SAFE_RUNS_DIR / challenge_relative)
            candidates.append(ARTEFACTS_DIR / challenge_relative)
            candidates.append(DATASETS_DIR / challenge_relative)
            if len(challenge_relative.parts) >= 2 and challenge_relative.parts[:2] == ("artefacts", "ad_safe"):
                candidates.append(ARTEFACTS_DIR / "ad_safe_runs" / Path(*challenge_relative.parts[2:]))
    else:
        candidates = [root / path for root in search_roots]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def resolve_required_existing_path(
    path_value: str | Path,
    *,
    field_name: str,
    search_roots: tuple[Path, ...] = PATH_SEARCH_ROOTS,
) -> Path:
    resolved_path = resolve_existing_path(path_value, search_roots=search_roots)
    if resolved_path is None:
        raise FileNotFoundError(f"Specified {field_name} does not exist: {path_value}")
    return resolved_path


def ensure_artifact_dirs() -> None:
    AD_SAFE_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    SWEEP_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    PROD_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SMOKE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
