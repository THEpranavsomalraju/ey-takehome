"""Build the submission ZIP from an explicit include list (docs/03 Phase 9).

Usage:
    python tools/package_submission.py --last LAST --first FIRST [--with-dev-notes] [--out DIR]

Writes ``<out>/LAST, FIRST_AI Technical Assessment.zip`` with one top-level
folder of the same name. Entries are sorted and carry a fixed timestamp, so
the same tree always gives the same bytes. The build stops before writing if
any included file holds an API key or other secret.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUFFIX = "_AI Technical Assessment"
FIXED_TIMESTAMP = (2026, 10, 18, 0, 0, 0)
INCLUDE_FILES = (
    "README.md",
    "MODEL_CARD.md",
    "DECISIONS.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-research.txt",
    "Dockerfile",
    ".dockerignore",
    ".gitignore",
)
INCLUDE_DIRS = (
    ".github",
    "artifacts",
    "data",
    "docs",
    "frontend",
    "ledgerclf",
    "notebooks",
    "relay",
    "reports",
    "research",
    "tests",
    "tools",
)
DEV_NOTES = ("CLAUDE.md", "PROGRESS.md", ".claude")
EXCLUDE_PATTERNS = (
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".DS_Store",
    "Thumbs.db",
    ".env",
    ".env.*",
    ".vercel",
    "node_modules",
    ".ipynb_checkpoints",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".hypothesis",
    "catboost_info",
    "*.egg-info",
    ".coverage",
    ".coverage.*",
)
SECRET_PATTERNS = (
    re.compile(rb"sk-ant-[A-Za-z0-9_\-]{16,}"),
    re.compile(rb"VERCEL_OIDC_TOKEN\s*=\s*\S"),
    re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


def _excluded(relative: Path) -> bool:
    return any(
        fnmatch.fnmatch(part, pattern) for part in relative.parts for pattern in EXCLUDE_PATTERNS
    )


def collect(with_dev_notes: bool) -> list[Path]:
    """Return the repo-relative files to package, sorted."""
    roots = [*INCLUDE_FILES, *INCLUDE_DIRS, *(DEV_NOTES if with_dev_notes else ())]
    files: set[Path] = set()
    for name in roots:
        path = ROOT / name
        candidates = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        files.update(p.relative_to(ROOT) for p in candidates if not _excluded(p.relative_to(ROOT)))
    return sorted(files, key=lambda p: p.as_posix())


def _local_secret_values() -> list[bytes]:
    """Values of the local .env files, so the scan also catches keys of unknown formats."""
    values = []
    for env in (ROOT / ".env", ROOT / "relay" / ".env.local"):
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                _, sep, value = line.partition("=")
                value = value.strip().strip('"').strip("'")
                if sep and len(value) >= 16:
                    values.append(value.encode("utf-8"))
    return values


def find_secrets(files: list[Path]) -> list[str]:
    """Return the files that contain a secret pattern or a local .env value."""
    known = _local_secret_values()
    hits = []
    for relative in files:
        data = (ROOT / relative).read_bytes()
        if any(p.search(data) for p in SECRET_PATTERNS) or any(v in data for v in known):
            hits.append(relative.as_posix())
    return hits


def build(files: list[Path], target: Path, folder: str) -> None:
    """Write the ZIP with sorted entries and a fixed timestamp."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in files:
            info = zipfile.ZipInfo(f"{folder}/{relative.as_posix()}", date_time=FIXED_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, (ROOT / relative).read_bytes())


def main(argv: list[str] | None = None) -> int:
    """Collect, scan, and write the submission ZIP."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--last", required=True, help="last name")
    parser.add_argument("--first", required=True, help="first name")
    parser.add_argument(
        "--with-dev-notes", action="store_true", help="also include " + ", ".join(DEV_NOTES)
    )
    parser.add_argument("--out", default=str(ROOT / "dist"), help="output folder (default: dist)")
    args = parser.parse_args(argv)
    folder = f"{args.last.strip()}, {args.first.strip()}{SUFFIX}"
    files = collect(args.with_dev_notes)
    secrets = find_secrets(files)
    if secrets:
        print("Refusing to package: secrets found in " + ", ".join(secrets), flush=True)
        return 1
    target = Path(args.out) / f"{folder}.zip"
    build(files, target, folder)
    size_mb = target.stat().st_size / 1_000_000
    print(f"Wrote {target} ({len(files)} files, {size_mb:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
