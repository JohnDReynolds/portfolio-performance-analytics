"""Run the project's local health checks with the repository virtual environment.

The script is intentionally local-checkout oriented. It refuses to run unless invoked
with ``./.venv/bin/python`` so type checkers, tests, and developer tools all see the
same dependency environment that VS Code and the project documentation expect.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_VENV_PYTHON = _PROJECT_ROOT / ".venv" / "bin" / "python"
_CHECK_CACHE_DIR = _PROJECT_ROOT / ".cache" / "check_project"
_MPLCONFIGDIR = _CHECK_CACHE_DIR / "matplotlib"


def _format_command(command: Sequence[str | Path]) -> str:
    """Return a readable command line for status output."""
    return " ".join(str(part) for part in command)


def _require_venv_python() -> None:
    """Validate that this script is running under the project virtual environment.

    Raises:
        SystemExit: If ``./.venv/bin/python`` is missing or the current interpreter is
            not the project virtual-environment interpreter.
    """
    if not _VENV_PYTHON.exists():
        raise SystemExit(
            "Missing .venv/bin/python. Create the project virtual environment before "
            "running scripts/check_project.py."
        )

    if Path(sys.executable).resolve() != _VENV_PYTHON.resolve():
        raise SystemExit(
            "Run this check with the project virtual environment:\n"
            "  ./.venv/bin/python scripts/check_project.py"
        )


def _run(command: Sequence[str | Path]) -> None:
    """Run a project check command and stop on failure.

    Args:
        command: Command and arguments to execute.

    Raises:
        subprocess.CalledProcessError: If the command exits with a non-zero status.
    """
    print(f"\n==> {_format_command(command)}", flush=True)
    _MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["XDG_CACHE_HOME"] = str(_CHECK_CACHE_DIR)
    env["MPLCONFIGDIR"] = str(_MPLCONFIGDIR)
    subprocess.run(
        [str(part) for part in command],
        cwd=_PROJECT_ROOT,
        check=True,
        env=env,
    )


def _run_build_check() -> None:
    """Build wheel and source distribution into a temporary directory."""
    try:
        with tempfile.TemporaryDirectory(prefix="ppar-build-check-") as temp_dir:
            _run(
                [
                    _VENV_PYTHON,
                    "-m",
                    "build",
                    "--wheel",
                    "--sdist",
                    "--no-isolation",
                    "--outdir",
                    temp_dir,
                ]
            )
    finally:
        # setuptools creates these intermediate directories in the project checkout
        # even when the final artifacts are written to a temporary output directory.
        for generated_path in (_PROJECT_ROOT / "build", _PROJECT_ROOT / "ppar.egg-info"):
            shutil.rmtree(generated_path, ignore_errors=True)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run project health checks using the mandatory .venv/bin/python interpreter."
        )
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Also build wheel and sdist into a temporary directory.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run the faster routine check set: tests, pyright errors, and pylint errors.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip the unit test suite.",
    )
    parser.add_argument(
        "--skip-types",
        action="store_true",
        help="Skip mypy and pyright.",
    )
    parser.add_argument(
        "--skip-pylint",
        action="store_true",
        help="Skip pylint error checks.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run selected project checks.

    Args:
        argv: Optional command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code. Returns ``0`` when all selected checks pass.
    """
    _require_venv_python()
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    if not args.skip_tests:
        _run([_VENV_PYTHON, "-m", "unittest", "discover", "tests"])

    if not args.skip_types:
        if not args.quick:
            _run([_VENV_PYTHON, "-m", "mypy"])
        _run([_VENV_PYTHON, "-m", "pyright", "--level", "error"])

    if not args.skip_pylint:
        # Existing design/refactor warnings are handled separately; this gate catches
        # pylint error-level regressions without requiring unrelated cleanup first.
        _run([_VENV_PYTHON, "-m", "pylint", "--errors-only", "ppar", "scripts", "tests"])

    if args.build:
        _run_build_check()

    print("\nAll selected project checks passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
