"""Command-line interface: ``python -m ledgerclf <command>``.

The only module allowed to print. Output is ASCII-only so that it works on
any console encoding (docs/05 section 5.5). Exit codes: 0 success, 1 user
input error, 2 artifact or environment error, 3 internal error.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence

from ledgerclf import __version__
from ledgerclf.config import Settings, get_settings
from ledgerclf.exceptions import ArtifactError, LedgerClfError
from ledgerclf.logging_setup import configure

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_USER_ERROR = 1
EXIT_ENVIRONMENT_ERROR = 2
EXIT_INTERNAL_ERROR = 3

Handler = Callable[[argparse.Namespace, Settings], int]


def echo(text: str = "") -> None:
    """Print one line, escaping non-ASCII characters."""
    print(text.encode("ascii", "backslashreplace").decode("ascii"), flush=True)


def _cmd_doctor(args: argparse.Namespace, settings: Settings) -> int:
    from ledgerclf.doctor import format_report, run_doctor

    report = run_doctor(settings, require_model=args.require_model, model_checks=_model_checks())
    for line in format_report(report):
        echo(line)
    return report.exit_code


def _model_checks() -> Callable[[Settings], list] | None:
    try:
        from ledgerclf.artifacts import doctor_model_checks
    except ImportError:
        return None
    return doctor_model_checks


def _cmd_split(args: argparse.Namespace, settings: Settings) -> int:
    from ledgerclf.splitting import make_holdout_split

    result = make_holdout_split(settings)
    manifest = result.manifest
    rows = manifest["rows"]
    echo(f"Holdout seed: {manifest['seed']} (seeds tried: {manifest['seeds_tried']})")
    echo(f"Rows: {rows['train']} training, {rows['holdout']} holdout, {rows['total']} total")
    for side in ("train", "holdout"):
        counts = manifest["class_counts"][side]
        echo(f"{side:>8}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    echo("Wrote data/splits/ and data/holdout/")
    return EXIT_OK


COMMANDS: dict[str, tuple[str, Handler]] = {
    "doctor": ("environment and artifact health check", _cmd_doctor),
    "split": ("label cleanup, group key, holdout seed rule, holdout files", _cmd_split),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ledgerclf", description="Accounts payable ledger classifier."
    )
    parser.add_argument("--version", action="version", version=f"ledgerclf {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug logging")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for name, (help_text, _) in COMMANDS.items():
        sub = commands.add_parser(name, help=help_text, description=help_text)
        _add_arguments(name, sub)
    return parser


def _add_arguments(name: str, sub: argparse.ArgumentParser) -> None:
    if name == "doctor":
        sub.add_argument(
            "--require-model", action="store_true", help="fail when trained artifacts are missing"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the process exit code.

    Args:
        argv: Arguments without the program name (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code: 0 success, 1 user input error, 2 artifact or environment
        error, 3 internal error.
    """
    args = _build_parser().parse_args(argv)
    configure(logging.DEBUG if args.verbose else logging.WARNING)
    handler = COMMANDS[args.command][1]
    try:
        return handler(args, get_settings())
    except ArtifactError as error:
        echo(f"Error: {error.message}")
        return EXIT_ENVIRONMENT_ERROR
    except LedgerClfError as error:
        echo(f"Error: {error.message}")
        return EXIT_USER_ERROR
    except KeyboardInterrupt:
        echo("Interrupted.")
        return EXIT_USER_ERROR
    except Exception:
        logger.debug("Unexpected error", exc_info=True)
        echo(
            "Internal error. Rerun with --verbose for details, then report the output. "
            "Your data and artifacts were not changed."
        )
        return EXIT_INTERNAL_ERROR


if __name__ == "__main__":
    sys.exit(main())
