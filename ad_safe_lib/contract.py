from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .config import SCRIPT_DIR


def run_foreign_contract_check(model_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "check_ad_safe_contract.py"),
            str(model_path),
        ],
        check=True,
    )
