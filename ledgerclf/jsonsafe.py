"""Conversion of numpy and pandas values to strict, JSON-safe Python values.

A bottom-layer helper shared by predictions, explanations, reports, and the API.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def to_jsonable(value: Any) -> Any:
    """Convert numpy and pandas values to native, JSON-safe Python values.

    NaN and infinities become None so that ``json.dumps`` output stays strict.
    """
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.bool_ | bool):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating | float):
        number = float(value)
        return number if math.isfinite(number) else None
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value
