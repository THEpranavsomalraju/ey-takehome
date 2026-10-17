"""Environment and artifact diagnostics (docs/05 section 10).

Every check returns PASS, WARN, or FAIL with a plain-English fix. The same
checks back ``python -m ledgerclf doctor`` and ``PredictionService.health()``.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import os
import platform
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from ledgerclf.config import FORCE_FALLBACK_ENV, SUPPORTED_PYTHON, Settings

logger = logging.getLogger(__name__)

RUNTIME_PACKAGES: tuple[tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("scikit-learn", "sklearn"),
    ("joblib", "joblib"),
    ("openpyxl", "openpyxl"),
    ("xlrd", "xlrd"),
)
LIGHTGBM_FIXES = {
    "Darwin": "Install the OpenMP runtime: brew install libomp",
    "Linux": "Install the OpenMP runtime: sudo apt-get install -y libgomp1",
    "Windows": "Install the latest Microsoft Visual C++ Redistributable (x64).",
}


class Status(StrEnum):
    """Outcome of one doctor check."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class CheckResult:
    """Result of one diagnostic check."""

    name: str
    status: Status
    detail: str
    fix: str = ""

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-safe dict."""
        return {
            "name": self.name,
            "status": str(self.status),
            "detail": self.detail,
            "fix": self.fix,
        }


@dataclass
class DoctorReport:
    """All check results plus the overall exit code."""

    checks: list[CheckResult] = field(default_factory=list)
    exit_code: int = 0

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe dict."""
        return {"exit_code": self.exit_code, "checks": [c.to_dict() for c in self.checks]}


def required_pins(settings: Settings) -> dict[str, str]:
    """Read exact pins from requirements.txt (empty when the file is absent)."""
    path = settings.paths.root / "requirements.txt"
    if not path.exists():
        return {}
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            name, version = line.split("==", 1)
            pins[name.strip().lower()] = version.strip()
    return pins


def check_python() -> CheckResult:
    """Check 1: Python version inside the supported range."""
    version = sys.version_info[:2]
    low, high = SUPPORTED_PYTHON
    detail = f"Python {platform.python_version()}"
    if low <= version <= high:
        return CheckResult("python", Status.PASS, detail)
    return CheckResult(
        "python",
        Status.FAIL,
        detail + " is outside the supported range.",
        f"Install Python {low[0]}.{low[1]} to {high[0]}.{high[1]} (3.12 recommended).",
    )


def check_platform() -> CheckResult:
    """Check 2: operating system and architecture (informational)."""
    return CheckResult("platform", Status.PASS, f"{platform.system()} {platform.machine()}")


def check_packages(settings: Settings) -> list[CheckResult]:
    """Check 3: each runtime package imports; versions compared with the pins."""
    pins = required_pins(settings)
    results = []
    for dist_name, module_name in RUNTIME_PACKAGES:
        try:
            importlib.import_module(module_name)
            installed = importlib.metadata.version(dist_name)
        except Exception as error:
            results.append(
                CheckResult(
                    f"package {dist_name}",
                    Status.FAIL,
                    f"cannot import: {type(error).__name__}",
                    "Run: python -m pip install -r requirements.txt",
                )
            )
            continue
        pinned = pins.get(dist_name)
        if pinned and pinned != installed:
            results.append(
                CheckResult(
                    f"package {dist_name}",
                    Status.WARN,
                    f"{installed} installed, {pinned} pinned",
                    "For exact reproduction run: python -m pip install -r requirements.txt",
                )
            )
        else:
            results.append(CheckResult(f"package {dist_name}", Status.PASS, installed))
    return results


def lightgbm_status() -> tuple[bool, str]:
    """Return (available, reason). Honors the forced-fallback environment variable."""
    if os.environ.get(FORCE_FALLBACK_ENV) == "1":
        return False, f"{FORCE_FALLBACK_ENV}=1 forces the fallback model"
    try:
        lightgbm = importlib.import_module("lightgbm")
    except (ImportError, OSError) as error:
        return False, f"{type(error).__name__}: {str(error).splitlines()[0][:160]}"
    return True, f"lightgbm {lightgbm.__version__}"


def check_lightgbm() -> CheckResult:
    """Check 4: LightGBM import. A failure activates the fallback model."""
    available, reason = lightgbm_status()
    if available:
        return CheckResult("lightgbm", Status.PASS, reason)
    fix = LIGHTGBM_FIXES.get(platform.system(), "Reinstall lightgbm from requirements.txt.")
    return CheckResult(
        "lightgbm",
        Status.WARN,
        f"not available ({reason}); predictions use the fallback logistic model",
        fix,
    )


def check_writable(settings: Settings) -> CheckResult:
    """Check 8: write permission for artifacts/ (needed only for retraining)."""
    target = settings.paths.artifacts
    probe_dir = target if target.exists() else target.parent
    try:
        with tempfile.TemporaryFile(dir=probe_dir):
            pass
    except OSError:
        return CheckResult(
            "artifacts writable",
            Status.WARN,
            f"cannot write to {probe_dir.name}/",
            "Prediction still works. Copy the project to a writable folder to retrain.",
        )
    return CheckResult("artifacts writable", Status.PASS, f"{target.name}/ is writable")


def check_holdout_files(settings: Settings) -> CheckResult:
    """Check 7: holdout feature files present."""
    expected = [settings.paths.holdout / f"holdout_features.{ext}" for ext in ("xlsx", "csv")]
    missing = [p.name for p in expected if not p.exists()]
    if missing:
        return CheckResult(
            "holdout files",
            Status.WARN,
            "missing: " + ", ".join(missing),
            "Run: python -m ledgerclf split",
        )
    return CheckResult("holdout files", Status.PASS, "holdout_features.xlsx and .csv present")


ModelChecks = Callable[[Settings], list[CheckResult]]


def run_doctor(
    settings: Settings, require_model: bool = False, model_checks: ModelChecks | None = None
) -> DoctorReport:
    """Run every check and compute the exit code.

    Args:
        settings: Project settings.
        require_model: Treat missing artifacts as a failure (gate from Phase 6).
        model_checks: Callable producing artifact, reference, and self-test
            checks (checks 5, 6, 9). Injected by the CLI so that this module
            stays importable without the modeling stack.

    Returns:
        The report. Exit code 0 when the environment passes and a model works
        (or artifacts do not exist yet and ``require_model`` is False), else 2.
    """
    checks = [check_python(), check_platform(), *check_packages(settings), check_lightgbm()]
    if model_checks is not None:
        checks.extend(model_checks(settings))
    checks.extend([check_holdout_files(settings), check_writable(settings)])
    failed = any(c.status is Status.FAIL for c in checks)
    missing_model = any(c.name == "artifacts" and c.status is not Status.PASS for c in checks)
    exit_code = 2 if failed or (require_model and missing_model) else 0
    return DoctorReport(checks, exit_code)


def format_report(report: DoctorReport) -> list[str]:
    """Render the report as ASCII lines for the CLI."""
    lines = [f"{c.status:<5} {c.name}: {c.detail}" for c in report.checks]
    problems = [c for c in report.checks if c.status is not Status.PASS and c.fix]
    counts = {s: sum(c.status is s for c in report.checks) for s in Status}
    lines.append("")
    passed, warned, failed = (counts[s] for s in (Status.PASS, Status.WARN, Status.FAIL))
    lines.append(f"Summary: {passed} pass, {warned} warn, {failed} fail")
    for check in problems:
        lines.append(f"Fix for {check.name}: {check.fix}")
    return lines
