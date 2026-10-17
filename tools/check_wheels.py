"""Verify binary wheel availability for every pin across the OS and Python matrix.

Usage:
    python tools/check_wheels.py

Runs ``pip install --dry-run --only-binary=:all:`` per (Python, platform)
pair for all pins at once, then per package when a pair fails, so that each
gap names the exact package. Needs network access (development only).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENT_FILES = ("requirements.txt", "requirements-dev.txt")
PYTHONS = ("3.11", "3.12", "3.13", "3.14")
PLATFORMS: dict[str, tuple[str, ...]] = {
    "windows-x64": ("win_amd64",),
    "macos-arm64": tuple(f"macosx_{v}_0_arm64" for v in (11, 12, 13, 14, 15)),
    "macos-x86_64": tuple(f"macosx_10_{v}_x86_64" for v in (9, 13, 14, 15))
    + tuple(f"macosx_{v}_0_x86_64" for v in (11, 12, 13, 14)),
    "linux-x86_64": ("manylinux2014_x86_64", "manylinux_2_17_x86_64", "manylinux_2_28_x86_64"),
}


def read_pins() -> list[str]:
    """Return every exact pin from the runtime and development requirement files."""
    pins: list[str] = []
    for name in REQUIREMENT_FILES:
        for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if "==" in line:
                pins.append(line)
    return pins


def resolves(pins: list[str], python: str, tags: tuple[str, ...]) -> tuple[bool, str]:
    """Return (ok, stderr tail) for resolving pins as binary wheels on one target."""
    with tempfile.TemporaryDirectory() as target:
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--dry-run",
            "--quiet",
            "--ignore-installed",
            "--no-deps",
            "--only-binary=:all:",
            "--python-version",
            python,
            "--implementation",
            "cp",
            "--target",
            target,
        ]
        for tag in tags:
            command.extend(["--platform", tag])
        completed = subprocess.run(
            [*command, *pins], capture_output=True, text=True, encoding="utf-8", check=False
        )
    return completed.returncode == 0, completed.stderr.strip()[-300:]


def main() -> int:
    """Print the wheel matrix and return 1 when any pin lacks a wheel."""
    pins = read_pins()
    print(f"Checking {len(pins)} pins x {len(PYTHONS)} Pythons x {len(PLATFORMS)} platforms")
    gaps: list[str] = []
    header = f"{'python':<8}" + "".join(f"{name:<15}" for name in PLATFORMS)
    print(header)
    for python in PYTHONS:
        cells = []
        for platform_name, tags in PLATFORMS.items():
            ok, _ = resolves(pins, python, tags)
            if not ok:
                missing = [pin for pin in pins if not resolves([pin], python, tags)[0]]
                gaps.extend(
                    f"{pin} missing for Python {python} on {platform_name}" for pin in missing
                )
            cells.append("ok" if ok else "GAP")
        print(f"{python:<8}" + "".join(f"{cell:<15}" for cell in cells), flush=True)
    if gaps:
        print("\nGaps:")
        for gap in gaps:
            print(f"- {gap}")
        return 1
    print("\nNo gaps: every pin has a binary wheel for every Python and platform.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
