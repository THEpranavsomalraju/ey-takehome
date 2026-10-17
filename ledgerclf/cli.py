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
from pathlib import Path

from ledgerclf import __version__
from ledgerclf.config import (
    EXIT_ENVIRONMENT_ERROR,
    EXIT_INTERNAL_ERROR,
    EXIT_OK,
    EXIT_USER_ERROR,
    Settings,
    get_settings,
)
from ledgerclf.exceptions import ArtifactError, LedgerClfError
from ledgerclf.logging_setup import configure

logger = logging.getLogger(__name__)

Handler = Callable[[argparse.Namespace, Settings], int]


def echo(text: str = "") -> None:
    """Print one line, escaping non-ASCII characters so any console can show it.

    Args:
        text: The line to print.
    """
    print(text.encode("ascii", "backslashreplace").decode("ascii"), flush=True)


def _cmd_doctor(args: argparse.Namespace, settings: Settings) -> int:
    from ledgerclf.doctor import format_report, run_doctor

    report = run_doctor(settings, require_model=args.require_model)
    for line in format_report(report):
        echo(line)
    return report.exit_code


def _cmd_split(args: argparse.Namespace, settings: Settings) -> int:
    from ledgerclf.splitting import make_holdout_split

    result = make_holdout_split(settings, force=args.force)
    manifest = result.manifest
    rows = manifest["rows"]
    echo(f"Holdout seed: {manifest['seed']} (seeds tried: {manifest['seeds_tried']})")
    echo(f"Rows: {rows['train']} training, {rows['holdout']} holdout, {rows['total']} total")
    for side in ("train", "holdout"):
        counts = manifest["class_counts"][side]
        echo(f"{side:>8}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    for directory in (settings.paths.splits, settings.paths.holdout):
        echo(f"Wrote {directory.relative_to(settings.paths.root).as_posix()}/")
    return EXIT_OK


def _cmd_eda(args: argparse.Namespace, settings: Settings) -> int:
    from ledgerclf.eda import run_eda
    from ledgerclf.preprocessing_reports import run_preprocessing_reports

    written = run_eda(settings)
    echo(f"Wrote {len(written)} EDA reports to reports/eda/ (labels: training portion only).")
    written = run_preprocessing_reports(settings)
    echo(f"Wrote {len(written)} preprocessing reports to reports/preprocessing/.")
    return EXIT_OK


def _cmd_benchmark(args: argparse.Namespace, settings: Settings) -> int:
    from ledgerclf.benchmark import run_benchmark

    written = run_benchmark(settings, quick=args.quick)
    mode = "quick" if args.quick else "full"
    echo(f"Wrote {len(written)} benchmark reports to reports/benchmark/ ({mode} run).")
    return EXIT_OK


def _cmd_tune(args: argparse.Namespace, settings: Settings) -> int:
    from ledgerclf.tuning import run_tuning

    written = run_tuning(settings, quick=args.quick)
    mode = "quick" if args.quick else "full"
    echo(f"Wrote {len(written)} tuning reports to reports/tuning/ ({mode} run).")
    return EXIT_OK


def _cmd_select(args: argparse.Namespace, settings: Settings) -> int:
    from ledgerclf.selection import run_selection

    written = run_selection(settings)
    echo(f"Wrote {len(written)} selection reports and artifacts/champion.json.")
    return EXIT_OK


def _cmd_train(args: argparse.Namespace, settings: Settings) -> int:
    from ledgerclf.training import run_training

    summary = run_training(settings)
    echo(
        f"Trained champion {summary['champion']} and fallback {summary['fallback']} on "
        f"{summary['training_rows']} training rows. Wrote artifacts/."
    )
    return EXIT_OK


def _cmd_predict(args: argparse.Namespace, settings: Settings) -> int:
    from ledgerclf.service import PredictionService

    service = PredictionService.load(auto_train=False, settings=settings)
    path = Path(args.file)
    result = service.predict_file(path.read_bytes() if path.is_file() else b"", path.name)
    summary = result.summary()
    echo(
        f"Model: {summary['model_used']}"
        + (f" (fallback: {result.fallback_reason})" if result.fallback_reason else "")
    )
    echo(f"Rows predicted: {summary['rows']}; review required: {summary['review_required']}")
    for cls, count in summary["predicted_counts"].items():
        echo(f"  {cls}: {count}")
    for issue in result.report.issues:
        echo(f"  {issue.severity}: {issue.message}")
    if result.metrics:
        echo(
            f"Accuracy on {result.metrics['labeled_rows']} labeled rows: "
            f"{result.metrics['accuracy']:.4f}"
        )
    if args.out:
        result.predictions.to_csv(args.out, index=False, encoding="utf-8-sig")
        echo(f"Wrote {args.out}")
    return EXIT_OK


def _cmd_explain(args: argparse.Namespace, settings: Settings) -> int:
    import json

    from ledgerclf.service import PredictionService

    service = PredictionService.load(auto_train=False, settings=settings)
    path = Path(args.file)
    result = service.predict_file(path.read_bytes() if path.is_file() else b"", path.name)
    payload = service.explain(result, args.row)
    echo(payload["fallback_text"])
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        echo(f"Wrote {args.out}")
    return EXIT_OK


def _cmd_evaluate_holdout(args: argparse.Namespace, settings: Settings) -> int:
    from ledgerclf.holdout_eval import run_holdout_evaluation
    from ledgerclf.splitting import issue_holdout_token

    token = issue_holdout_token(confirmed=args.confirm, reason=args.reason)
    summary = run_holdout_evaluation(settings, token, rerun=args.rerun)
    for name in ("champion", "fallback_logistic"):
        metrics = summary.get(name, {})
        if "accuracy" in metrics:
            echo(
                f"{name}: holdout accuracy {metrics['accuracy']:.4f}, "
                f"macro_f1_5 {metrics['macro_f1_5']:.4f}"
            )
    echo("Wrote reports/holdout/")
    return EXIT_OK


COMMANDS: dict[str, tuple[str, Handler]] = {
    "doctor": ("environment and artifact health check", _cmd_doctor),
    "split": ("label cleanup, group key, holdout seed rule, holdout files", _cmd_split),
    "eda": ("EDA and preprocessing reports on the training portion", _cmd_eda),
    "benchmark": ("leaderboard, ablations, validation schemes", _cmd_benchmark),
    "tune": ("logged hyperparameter search and nested CV", _cmd_tune),
    "select": ("champion decision, calibration, thresholds, flag policy", _cmd_select),
    "train": ("final fit on the training portion, artifacts, golden files", _cmd_train),
    "evaluate-holdout": (
        "single holdout evaluation (ask the project owner first)",
        _cmd_evaluate_holdout,
    ),
    "predict": ("predict an Excel or CSV file", _cmd_predict),
    "explain": ("explain one predicted row of a file", _cmd_explain),
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
    if name in ("predict", "explain"):
        sub.add_argument("file", help="Excel (.xlsx, .xls) or CSV file with Col1 to Col7")
        sub.add_argument("--out", help="write predictions (CSV) or the explanation (JSON) here")
    if name == "explain":
        sub.add_argument(
            "--row", type=int, required=True, help="row number in the predictions table"
        )
    if name == "evaluate-holdout":
        sub.add_argument("--confirm", action="store_true", help="confirm the single holdout look")
        sub.add_argument("--rerun", action="store_true", help="allow a second, logged evaluation")
        sub.add_argument(
            "--reason", default="final evaluation", help="reason written to the access log"
        )
    if name in ("benchmark", "tune"):
        sub.add_argument("--quick", action="store_true", help="reduced budget for development")
    if name == "split":
        sub.add_argument(
            "--force", action="store_true", help="replace an existing split even if it changes"
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
        echo("Internal error. Rerun the same command with --verbose and report the output.")
        return EXIT_INTERNAL_ERROR


if __name__ == "__main__":
    sys.exit(main())
