"""Enable ``python -m ledgerclf``."""

from __future__ import annotations

import sys

from ledgerclf.cli import main

if __name__ == "__main__":
    sys.exit(main())
