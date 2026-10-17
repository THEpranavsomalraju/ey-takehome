"""Execute every notebook top to bottom in a fresh kernel and save the outputs.

Usage:
    python tools/run_notebooks.py [NOTEBOOK ...]

Fails on the first cell error. Run it after any change that touches the
notebooks or the reports they read.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"
CELL_TIMEOUT_SECONDS = 600


def run(path: Path) -> float:
    """Execute one notebook in place with its own directory as the working directory."""
    notebook = nbformat.read(path, as_version=4)
    started = time.time()
    client = NotebookClient(
        notebook,
        timeout=CELL_TIMEOUT_SECONDS,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()
    nbformat.write(notebook, path)
    return time.time() - started


def main() -> int:
    """Run the named notebooks, or every notebook, and report timings."""
    paths = [Path(p).resolve() for p in sys.argv[1:]] or sorted(NOTEBOOKS.glob("*.ipynb"))
    for path in paths:
        seconds = run(path)
        print(f"PASS {path.relative_to(ROOT).as_posix()} in {seconds:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
