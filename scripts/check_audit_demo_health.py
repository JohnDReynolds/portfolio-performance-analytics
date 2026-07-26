"""Run packaged Audit demo health checks.

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
            "running scripts/check_audit_demo_health.py."
        )

    if Path(sys.executable).resolve() != _VENV_PYTHON.resolve():
        raise SystemExit(
            "Run this check with the project virtual environment:\n"
            "  ./.venv/bin/python scripts/check_audit_demo_health.py"
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
        audit_directory = Path(directory) / "my_ppar_audit"

        _run(
            [
                _VENV_PYTHON,
                "-m",
                "ppar.cli",
                "setup",
                audit_directory,
            ]
        )
        _run(
            [
                _VENV_PYTHON,
                audit_directory / "run_audit.py",
            ]
        )
        _run(
            [
                _VENV_PYTHON,
                "-m",
                "ppar.audit.cli.validate_bundle",
                audit_directory / "output" / "portfolio",
            ]
        )
        _run(
            [
                _VENV_PYTHON,
                "-m",
                "ppar.audit.cli.validate_bundle",
                audit_directory / "output" / "security",
            ]
        )


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run packaged Audit demo health checks."
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
    parser.add_argument(
        "--write-packaged-assets",
        action="store_true",
        help=(
            "Rewrite tracked packaged Audit CSV assets after validating the "
            "derived demo data."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run selected Audit demo health checks.

    Args:
        argv: Optional command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code. Returns ``0`` when all selected checks pass.
    """
    _require_venv_python()
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    if not args.skip_rebuild_audit:
        rebuild_command: list[str | Path] = [
            _VENV_PYTHON,
            "scripts/operational_demo_data/rebuild_audit_demo_data.py",
        ]
        if args.write_packaged_assets:
            rebuild_command.append("--write")
        _run(rebuild_command)

    if not args.skip_extract_availability:
        _run(
            [
                _VENV_PYTHON,
                "scripts/render_demo_extract_availability.py",
                "--check",
            ]
        )
        _run(
            [
                _VENV_PYTHON,
                "scripts/render_transaction_semantics_matrix.py",
                "--check",
            ]
        )

    if not args.skip_bundles:
        _run_setup_generated_smoke_tests()

    if not args.skip_demo_matrix:
        _run([_VENV_PYTHON, "scripts/validate_demo_matrix.py"])

    print("\nPackaged Audit demo health checks passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
