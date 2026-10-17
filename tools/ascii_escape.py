"""Rewrite non-ASCII characters in Python sources as escape sequences.

Usage:
    python tools/ascii_escape.py [PATH ...]

Python sources must stay ASCII (docs/05 section 5.5). Run this after pasting
text that contains accented characters or symbols into a string literal.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIRS = ("ledgerclf", "tests", "tools", "research", "dashboard")


def escape(text: str) -> str:
    """Return text with every non-ASCII character written as a Python escape."""
    out = []
    for char in text:
        code = ord(char)
        if code < 128:
            out.append(char)
        elif code <= 0xFFFF:
            out.append(f"\\u{code:04x}")
        else:
            out.append(f"\\U{code:08x}")
    return "".join(out)


def main() -> int:
    """Escape every non-ASCII Python source under the given paths."""
    targets = [Path(p).resolve() for p in sys.argv[1:]] or [ROOT / d for d in DEFAULT_DIRS]
    files = [f for t in targets for f in ([t] if t.is_file() else sorted(t.rglob("*.py")))]
    for path in files:
        text = path.read_text(encoding="utf-8")
        if not text.isascii():
            path.write_text(escape(text), encoding="utf-8")
            print(f"escaped {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
