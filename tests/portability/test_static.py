"""Static portability checks (docs/05 section 6): AST and text scans of the source tree."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.portability

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRS = ("ledgerclf", "tools", "tests", "research", "dashboard")
ABSOLUTE_PATH = re.compile(r"""["'](?:[A-Za-z]:\\\\|/Users/|/home/|/opt/|C:/)""")


def _python_files() -> list[Path]:
    files: list[Path] = []
    for name in SOURCE_DIRS:
        files.extend(sorted((ROOT / name).rglob("*.py")))
    return files


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _calls(tree: ast.Module):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            yield node


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _is_binary_mode(node: ast.Call) -> bool:
    # Builtin open(file, mode) takes the mode second; Path.open(mode) takes it first.
    position = slice(0, 1) if isinstance(node.func, ast.Attribute) else slice(1, 2)
    mode_args = [*node.args[position], *(k.value for k in node.keywords if k.arg == "mode")]
    for arg in mode_args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and "b" in arg.value:
            return True
    return False


def test_text_open_calls_pass_encoding():
    offenders = []
    for path in _python_files():
        for node in _calls(_parse(path)):
            name = _call_name(node)
            if name not in ("open", "read_text", "write_text"):
                continue
            if name == "open" and _is_binary_mode(node):
                continue
            if name == "open" and isinstance(node.func, ast.Attribute) and node.func.attr == "open":
                owner = ast.unparse(node.func.value)
                if any(
                    token in owner
                    for token in ("zipfile", "ZipFile", "archive", "Image", "webbrowser")
                ):
                    continue
            if not any(k.arg == "encoding" for k in node.keywords):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} {name}()")
    assert not offenders, offenders


def test_no_shell_execution():
    offenders = []
    for path in _python_files():
        for node in _calls(_parse(path)):
            if any(
                k.arg == "shell" and getattr(k.value, "value", False) is True for k in node.keywords
            ):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} shell=True")
            if ast.unparse(node.func) in ("os.system", "os.popen"):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} {ast.unparse(node.func)}")
    assert not offenders, offenders


def test_no_absolute_paths_in_source():
    offenders = [
        f"{path.relative_to(ROOT)}:{number}"
        for path in _python_files()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if ABSOLUTE_PATH.search(line) and "noqa: absolute-path-example" not in line
    ]
    assert not offenders, offenders


def test_python_sources_are_ascii():
    offenders = []
    for path in _python_files():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.isascii():
                offenders.append(f"{path.relative_to(ROOT)}:{number}")
    assert not offenders, offenders


def test_lightgbm_never_imported_at_module_level():
    offenders = []
    for path in sorted((ROOT / "ledgerclf").rglob("*.py")):
        for node in _parse(path).body:
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(name.split(".")[0] == "lightgbm" for name in names):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not offenders, offenders


def test_no_locale_module():
    offenders = [
        str(path.relative_to(ROOT))
        for path in _python_files()
        if re.search(
            r"^\s*(import locale|from locale import)", path.read_text(encoding="utf-8"), re.M
        )
    ]
    assert not offenders, offenders


def test_repo_paths_stay_short():
    long_paths = [
        p.relative_to(ROOT).as_posix()
        for p in ROOT.rglob("*")
        if ".venv" not in p.parts
        and ".git" not in p.parts
        and "node_modules" not in p.parts
        and len(p.relative_to(ROOT).as_posix()) > 100
    ]
    assert not long_paths, long_paths
