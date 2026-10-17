"""Runtime environment probes shared by the doctor, artifacts, and service.

Kept free of the modeling stack so that it imports even when scikit-learn
or LightGBM is broken.
"""

from __future__ import annotations

import importlib
import os

from ledgerclf.config import FORCE_FALLBACK_ENV, Limits


def lightgbm_status(limits: Limits | None = None) -> tuple[bool, str]:
    """Report whether LightGBM imports, honoring the forced-fallback variable.

    Args:
        limits: Provides the maximum length of the error detail.

    Returns:
        Tuple of (available, reason).
    """
    limits = limits or Limits()
    if os.environ.get(FORCE_FALLBACK_ENV) == "1":
        return False, f"{FORCE_FALLBACK_ENV}=1 forces the fallback model"
    try:
        lightgbm = importlib.import_module("lightgbm")
    except (ImportError, OSError) as error:
        first_line = next(iter(str(error).splitlines()), "no message")
        return False, f"{type(error).__name__}: {first_line[: limits.message_detail_chars]}"
    return True, f"lightgbm {lightgbm.__version__}"
