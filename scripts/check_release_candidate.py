"""Run the release-candidate demo, packaging, and health-check sequence.

This source-checkout helper is the maintained replacement for pasting a long
shell checklist into a terminal. It runs commands one at a time, prints the
exact command being run, and stops at the first failure.
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
_CHECK_CACHE_DIR = _PROJECT_ROOT / ".cache" / "check_release_candidate"
_COMPARISON_YAML = (
    _PROJECT_ROOT
    / "ppar"
    / "setup_templates"
    / "axysapx_performance_comparison"
    / "axysapx_performance_comparison.yaml"
)
_OUTPUT_DIRECTORIES = (
    _PROJECT_ROOT / "_demo_output" / "generic_analytics_data_generation",
    _PROJECT_ROOT / "_demo_output" / "performance_comparison_portfolio",
    _PROJECT_ROOT / "_demo_output" / "performance_comparison_security",
    _PROJECT_ROOT / "_demo_output" / "readme_image_cache",
)


def _format_command(command: Sequence[str | Path]) -> str:
    """Return a readable command line for status output.

    Args:
        command: Command and arguments that will be executed.

    Returns:
        A readable command string for logs.
    """
    return " ".join(str(part) for part in command)


def _require_venv_python() -> None:
    """Validate that this script is running under the repository virtual environment.

    Raises:
        SystemExit: If ``./.venv/bin/python`` is missing or is not the current
            interpreter.
    """
    if not _VENV_PYTHON.exists():
        raise SystemExit(
            "Missing .venv/bin/python. Create the project virtual environment before "
            "running scripts/check_release_candidate.py."
        )

    if Path(sys.executable).resolve() != _VENV_PYTHON.resolve():
        raise SystemExit(
            "Run this check with the project virtual environment:\n"
            "  ./.venv/bin/python scripts/check_release_candidate.py"
        )


def _run(command: Sequence[str | Path]) -> None:
    """Run one release-candidate command and stop on failure.

    Args:
        command: Command and arguments to execute.

    Raises:
        subprocess.CalledProcessError: If the command exits nonzero.
    """
    print(f"\n==> {_format_command(command)}", flush=True)
    _CHECK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("XDG_CACHE_HOME", str(_CHECK_CACHE_DIR))
    env.setdefault("MPLCONFIGDIR", str(_CHECK_CACHE_DIR / "matplotlib"))
    subprocess.run(
        [str(part) for part in command],
        cwd=_PROJECT_ROOT,
        check=True,
        env=env,
    )


def _clean_generated_output() -> None:
    """Remove ignored generated-output directories used by this check."""
    print("\n==> Cleaning generated release-candidate output", flush=True)
    for output_directory in _OUTPUT_DIRECTORIES:
        shutil.rmtree(output_directory, ignore_errors=True)


def _run_generic_data_generation() -> None:
    """Generate and validate optional candidate generic analytics demo data."""
    _run(
        [
            _VENV_PYTHON,
            "scripts/generic_analytics_demo_data/"
            "generate_mega_cap_analytics_demo_data.py",
        ]
    )
    _run(
        [
            _VENV_PYTHON,
            "scripts/generic_analytics_demo_data/"
            "validate_generated_analytics_demo_data.py",
        ]
    )


def _run_demo_data_audit(*, write_packaged_assets: bool) -> None:
    """Audit or rewrite packaged Axys/APX performance-comparison demo data.

    Args:
        write_packaged_assets: Whether to pass ``--write`` to the rebuild
            script, updating tracked packaged CSV files after intentional
            source-data changes.
    """
    command: list[str | Path] = [
        _VENV_PYTHON,
        "scripts/operational_demo_data/rebuild_performance_comparison_demo_data.py",
    ]
    if write_packaged_assets:
        command.append("--write")
    _run(command)


def _run_extract_availability_check() -> None:
    """Check that rendered Axys/APX extract-availability docs are current."""
    _run(
        [
            _VENV_PYTHON,
            "scripts/render_demo_extract_availability.py",
            "--check",
        ]
    )


def _run_report_bundle_checks() -> None:
    """Generate and validate portfolio and security demo report bundles."""
    bundle_specs = (
        (
            "portfolio",
            _PROJECT_ROOT / "_demo_output" / "performance_comparison_portfolio",
        ),
        (
            "security",
            _PROJECT_ROOT / "_demo_output" / "performance_comparison_security",
        ),
    )
    for comparison_level, output_directory in bundle_specs:
        _run(
            [
                _VENV_PYTHON,
                "-m",
                "ppar.performance_comparison.cli.report_bundle",
                _COMPARISON_YAML,
                output_directory,
                "--comparison-level",
                comparison_level,
                "--include-workbook",
            ]
        )
        _run(
            [
                _VENV_PYTHON,
                "-m",
                "ppar.performance_comparison.cli.validate_bundle",
                output_directory,
            ]
        )


def _run_setup_smoke_tests() -> None:
    """Run scripts copied by ``ppar setup`` in a temporary site workspace."""
    with tempfile.TemporaryDirectory(prefix="ppar_release_site_") as directory:
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
        _run([_VENV_PYTHON, comparison_directory / "run_portfolio_comparison.py"])
        _run([_VENV_PYTHON, comparison_directory / "run_security_comparison.py"])
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


def _run_readme_image_refresh() -> None:
    """Refresh README images from current generated demo outputs."""
    _run([_VENV_PYTHON, "scripts/render_readme_images.py"])


def _run_project_checks(*, quick: bool, build: bool) -> None:
    """Run the consolidated project health check.

    Args:
        quick: Whether to use the faster project-check profile.
        build: Whether to include wheel and source-distribution build checks.
    """
    command: list[str | Path] = [_VENV_PYTHON, "scripts/check_project.py"]
    if quick:
        command.append("--quick")
    if build:
        command.append("--build")
    _run(command)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the maintained release-candidate check sequence.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete ignored _demo_output directories before running checks.",
    )
    parser.add_argument(
        "--include-generic-data-generation",
        action="store_true",
        help=(
            "Also run the Yahoo-dependent generic analytics data generator and "
            "validator. This is intentionally outside the default deterministic path."
        ),
    )
    parser.add_argument(
        "--write-packaged-assets",
        action="store_true",
        help=(
            "Rewrite tracked packaged Axys/APX performance-comparison CSV assets "
            "when the rebuild script derives intentional changes."
        ),
    )
    parser.add_argument(
        "--refresh-images",
        action="store_true",
        help="Refresh tracked README images under docs/images/readme.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use check_project.py --quick for the final project check.",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Include the final wheel/sdist build check.",
    )
    parser.add_argument(
        "--skip-project-check",
        action="store_true",
        help="Skip the final scripts/check_project.py pass.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run release-candidate checks.

    Args:
        argv: Optional command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code. Returns ``0`` when all selected checks pass.
    """
    _require_venv_python()
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    if args.clean_output:
        _clean_generated_output()

    if args.include_generic_data_generation:
        _run_generic_data_generation()

    _run_demo_data_audit(write_packaged_assets=args.write_packaged_assets)
    _run_extract_availability_check()
    _run_report_bundle_checks()
    _run_setup_smoke_tests()
    _run([_VENV_PYTHON, "-m", "ppar.performance_comparison.cli.validate_demo_matrix"])

    if args.refresh_images:
        _run_readme_image_refresh()

    if not args.skip_project_check:
        _run_project_checks(quick=args.quick, build=args.build)

    print("\nRelease-candidate checks passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
