"""Run packaged performance-comparison demo health checks.

This source-checkout helper consolidates the existing demo guardrails into one
command. It intentionally runs the real validators and demo commands instead of
re-implementing their checks, so failures keep their original detailed output.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_VENV_PYTHON = _PROJECT_ROOT / ".venv" / "bin" / "python"


def _format_command(command: Sequence[str | Path]) -> str:
    """Return a readable command line for status output."""
    return " ".join(str(part) for part in command)


def _require_venv_python() -> None:
    """Validate that this script is running under the project virtual environment.

    Raises:
        SystemExit: If ``./.venv/bin/python`` is missing or is not the current
            interpreter.
    """
    if not _VENV_PYTHON.exists():
        raise SystemExit(
            "Missing .venv/bin/python. Create the project virtual environment before "
            "running scripts/check_performance_comparison_demo_health.py."
        )

    if Path(sys.executable).resolve() != _VENV_PYTHON.resolve():
        raise SystemExit(
            "Run this check with the project virtual environment:\n"
            "  ./.venv/bin/python scripts/check_performance_comparison_demo_health.py"
        )


def _run(command: Sequence[str | Path]) -> None:
    """Run one demo health command and stop on failure.

    Args:
        command: Command and arguments to execute.

    Raises:
        subprocess.CalledProcessError: If the command exits nonzero.
    """
    print(f"\n==> {_format_command(command)}", flush=True)
    subprocess.run(
        [str(part) for part in command],
        cwd=_PROJECT_ROOT,
        check=True,
    )


def _run_setup_generated_smoke_tests() -> None:
    """Create a setup workspace and run its copied Python scripts.

    The copied scripts are the user-visible Python examples installed by
    ``ppar setup``. Running them here keeps the health check focused on the
    onboarding surface instead of the older source-checkout demo modules.
    """
    with tempfile.TemporaryDirectory(prefix="ppar_setup_smoke_") as directory:
        site_directory = Path(directory) / "my_ppar_data"
        comparison_directory = site_directory / "performance_comparison"

        _run(
            [
                _VENV_PYTHON,
                "-m",
                "ppar.cli",
                "setup",
                site_directory,
                "--include-generic-analytics",
            ]
        )
        _run([_VENV_PYTHON, site_directory / "analytics" / "run_analytics.py"])
        _run(
            [
                _VENV_PYTHON,
                comparison_directory / "run_portfolio_comparison.py",
            ]
        )
        _run(
            [
                _VENV_PYTHON,
                comparison_directory / "run_security_comparison.py",
            ]
        )
        _run(
            [
                _VENV_PYTHON,
                site_directory / "generic_analytics" / "run_generic_analytics.py",
            ]
        )
        _run(
            [
                _VENV_PYTHON,
                "-m",
                "ppar.performance_comparison.cli.validate_bundle",
                comparison_directory / "output" / "portfolio",
            ]
        )
        _run(
            [
                _VENV_PYTHON,
                "-m",
                "ppar.performance_comparison.cli.validate_bundle",
                comparison_directory / "output" / "security",
            ]
        )


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run packaged performance-comparison demo health checks."
    )
    parser.add_argument(
        "--skip-rebuild-audit",
        action="store_true",
        help="Skip the operational demo rebuild drift audit.",
    )
    parser.add_argument(
        "--skip-extract-availability",
        action="store_true",
        help="Skip checking the rendered Axys/APX demo extract-availability contract.",
    )
    parser.add_argument(
        "--skip-bundles",
        action="store_true",
        help="Skip generating and validating portfolio/security demo bundles.",
    )
    parser.add_argument(
        "--skip-demo-matrix",
        action="store_true",
        help="Skip packaged scenario-matrix validation.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run selected performance-comparison demo health checks.

    Args:
        argv: Optional command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code. Returns ``0`` when all selected checks pass.
    """
    _require_venv_python()
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    if not args.skip_rebuild_audit:
        _run(
            [
                _VENV_PYTHON,
                "scripts/operational_demo_data/"
                "rebuild_performance_comparison_demo_data.py",
            ]
        )

    if not args.skip_extract_availability:
        _run(
            [
                _VENV_PYTHON,
                "scripts/render_demo_extract_availability.py",
                "--check",
            ]
        )

    if not args.skip_bundles:
        _run_setup_generated_smoke_tests()

    if not args.skip_demo_matrix:
        _run([_VENV_PYTHON, "-m", "ppar.performance_comparison.cli.validate_demo_matrix"])

    print("\nPackaged performance-comparison demo health checks passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
