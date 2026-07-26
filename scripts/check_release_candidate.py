"""Run the release-candidate demo, packaging, and health-check sequence.

This source-checkout helper is the maintained replacement for pasting a long
shell checklist into a terminal. It runs commands one at a time, prints the
exact command being run, and stops at the first failure.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
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
_AUDIT_DEMO_WORKSPACE = _PROJECT_ROOT / "_demo_output" / "audit_workspace"
_AUDIT_OUTPUT_ROOT = _PROJECT_ROOT / "_demo_output" / "audit"
_OUTPUT_DIRECTORIES = (
    _PROJECT_ROOT / "_demo_output" / "generic_analytics_data_generation",
    _AUDIT_DEMO_WORKSPACE,
    _AUDIT_OUTPUT_ROOT,
    _PROJECT_ROOT / "_demo_output" / "readme_image_cache",
)


@dataclass
class ReleaseCandidateRunner:
    """Run release-candidate commands while tracking status for a concise summary.

    Attributes:
        verbose: Whether to stream subcommand output directly to the terminal.
        completed_phases: Names of phases that completed successfully.
        skipped_items: Notes describing checks intentionally skipped by options.
        changed_asset_notes: Notes describing options that may modify tracked files.
    """

    verbose: bool = False
    completed_phases: list[str] = field(default_factory=list)
    skipped_items: list[str] = field(default_factory=list)
    changed_asset_notes: list[str] = field(default_factory=list)

    def phase(self, number: int, name: str) -> None:
        """Print a numbered release-candidate phase heading.

        Args:
            number: Phase number in the release-candidate sequence.
            name: Human-readable phase name.
        """
        print(f"\n{number}. {name}", flush=True)

    def complete(self, name: str) -> None:
        """Record and print a successful phase completion.

        Args:
            name: Human-readable phase name.
        """
        self.completed_phases.append(name)
        print(f"   OK: {name}", flush=True)

    def skip(self, note: str) -> None:
        """Record and print an intentionally skipped item.

        Args:
            note: Explanation of what was skipped.
        """
        self.skipped_items.append(note)
        print(f"   SKIP: {note}", flush=True)

    def asset_note(self, note: str) -> None:
        """Record an option that may change tracked generated assets.

        Args:
            note: Explanation of the possible tracked-file change.
        """
        self.changed_asset_notes.append(note)

    def run(self, command: Sequence[str | Path]) -> None:
        """Run one release-candidate command and stop on failure.

        Args:
            command: Command and arguments to execute.

        Raises:
            subprocess.CalledProcessError: If the command exits nonzero.
        """
        print(f"   RUN: {_format_command(command)}", flush=True)
        _CHECK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.setdefault("XDG_CACHE_HOME", str(_CHECK_CACHE_DIR))
        env.setdefault("MPLCONFIGDIR", str(_CHECK_CACHE_DIR / "matplotlib"))
        if self.verbose:
            subprocess.run(
                [str(part) for part in command],
                cwd=_PROJECT_ROOT,
                check=True,
                env=env,
            )
            return

        completed = subprocess.run(
            [str(part) for part in command],
            cwd=_PROJECT_ROOT,
            check=False,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if completed.returncode != 0:
            print("\nCommand failed. Captured output:", flush=True)
            print(completed.stdout, flush=True)
            raise subprocess.CalledProcessError(
                completed.returncode,
                [str(part) for part in command],
                output=completed.stdout,
            )

    def print_summary(self) -> None:
        """Print release-candidate summary details."""
        print("\nRelease-candidate checks passed.", flush=True)
        print("\nCompleted:", flush=True)
        for phase in self.completed_phases:
            print(f"  - {phase}", flush=True)
        if self.skipped_items:
            print("\nSkipped:", flush=True)
            for item in self.skipped_items:
                print(f"  - {item}", flush=True)
        if self.changed_asset_notes:
            print("\nTracked assets may have changed:", flush=True)
            for item in self.changed_asset_notes:
                print(f"  - {item}", flush=True)


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


def _clean_generated_output(runner: ReleaseCandidateRunner) -> None:
    """Remove ignored generated-output directories used by this check."""
    runner.phase(0, "Clean generated output")
    for output_directory in _OUTPUT_DIRECTORIES:
        shutil.rmtree(output_directory, ignore_errors=True)
    runner.complete("Clean generated output")


def _run_generic_data_generation(runner: ReleaseCandidateRunner) -> None:
    """Generate and validate optional candidate generic analytics demo data."""
    runner.run(
        [
            _VENV_PYTHON,
            "-m",
            "scripts.generic_analytics_demo_data.generate_mega_cap_analytics_demo_data",
        ]
    )
    runner.run(
        [
            _VENV_PYTHON,
            "-m",
            "scripts.generic_analytics_demo_data.validate_generated_analytics_demo_data",
        ]
    )


def _run_audit_demo_health(
    runner: ReleaseCandidateRunner,
    *,
    write_packaged_assets: bool,
) -> None:
    """Run the canonical packaged Audit health check.

    Args:
        runner: Release-candidate command runner.
        write_packaged_assets: Whether to pass ``--write`` to the rebuild
            script, updating tracked packaged CSV files after intentional
            source-data changes.
    """
    command: list[str | Path] = [
        _VENV_PYTHON,
        "scripts/check_audit_demo_health.py",
    ]
    if write_packaged_assets:
        command.append("--write-packaged-assets")
    runner.run(command)


def _run_audit_demo_report_checks(runner: ReleaseCandidateRunner) -> None:
    """Generate both maintained demo reports through ``ppar audit``."""
    runner.run(
        [
            _VENV_PYTHON,
            "-m",
            "ppar.cli",
            "setup",
            _AUDIT_DEMO_WORKSPACE,
            "--overwrite",
        ]
    )
    runner.run(
        [
            _VENV_PYTHON,
            "-m",
            "ppar.cli",
            "audit",
            _AUDIT_DEMO_WORKSPACE,
            "--output-directory",
            _AUDIT_OUTPUT_ROOT,
        ]
    )
    for comparison_level in ("portfolio", "security"):
        runner.run(
            [
                _VENV_PYTHON,
                "-m",
                "ppar.audit.cli.validate_bundle",
                _AUDIT_OUTPUT_ROOT / comparison_level,
            ]
        )


def _run_analytics_setup_smoke_tests(runner: ReleaseCandidateRunner) -> None:
    """Run both Analytics scripts copied by ``ppar setup``."""
    with tempfile.TemporaryDirectory(prefix="ppar_release_site_") as directory:
        analytics_directory = Path(directory) / "my_ppar_analytics"
        generic_directory = Path(directory) / "my_ppar_generic_analytics"

        runner.run(
            [
                _VENV_PYTHON,
                "-m",
                "ppar.cli",
                "setup",
                analytics_directory,
                "--analytics",
            ]
        )
        runner.run(
            [
                _VENV_PYTHON,
                "-m",
                "ppar.cli",
                "setup",
                generic_directory,
                "--generic-analytics",
            ]
        )
        runner.run([_VENV_PYTHON, analytics_directory / "run_analytics.py"])
        runner.run([_VENV_PYTHON, generic_directory / "run_generic_analytics.py"])


def _release_asset_refresh_scope(
    *,
    build: bool,
    refresh_images: bool,
) -> tuple[bool, bool]:
    """Return whether a release-candidate run refreshes images and the PDF.

    Args:
        build: Whether distributable package artifacts will be built.
        refresh_images: Whether the caller explicitly requested README images.

    Returns:
        A tuple of ``(refresh_images, refresh_pdf)``. A release build always
        refreshes the PDF so the product overview cannot lag behind README.md.
    """
    return refresh_images, build or refresh_images


def _run_readme_asset_refresh(
    runner: ReleaseCandidateRunner,
    *,
    refresh_images: bool,
) -> None:
    """Refresh selected README assets and always refresh the product PDF.

    Args:
        runner: Release-candidate command runner.
        refresh_images: Whether to regenerate README PNG and JPG assets before
            regenerating the PDF.
    """
    if refresh_images:
        runner.run([_VENV_PYTHON, "scripts/render_readme_images.py"])
    runner.run([_VENV_PYTHON, "scripts/render_readme_pdf.py"])


def _run_project_checks(
    runner: ReleaseCandidateRunner,
    *,
    quick: bool,
    build: bool,
) -> None:
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
    runner.run(command)


def _run_scale_regression_check(runner: ReleaseCandidateRunner) -> None:
    """Run the hard 500x Analytics and Audit release-candidate scale gate."""
    runner.run([_VENV_PYTHON, "scripts/check_scale.py", "--scale", "500"])


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
            "Rewrite tracked packaged Axys/APX Audit CSV assets "
            "when the rebuild script derives intentional changes."
        ),
    )
    parser.add_argument(
        "--refresh-images",
        action="store_true",
        help=(
            "Refresh tracked README images and the root PPAR.pdf. Release builds "
            "refresh PPAR.pdf even when this option is omitted."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use check_project.py --quick for the final project check.",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help=(
            "Refresh PPAR.pdf and include the final wheel/sdist build check."
        ),
    )
    parser.add_argument(
        "--skip-project-check",
        action="store_true",
        help="Skip the final scripts/check_project.py pass.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stream full subcommand output instead of only command names.",
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
    runner = ReleaseCandidateRunner(verbose=args.verbose)

    if args.clean_output:
        _clean_generated_output(runner)
    else:
        runner.skip("Generated-output cleanup; use --clean-output to remove caches first.")

    runner.phase(1, "Optional generic analytics data generation")
    if args.include_generic_data_generation:
        _run_generic_data_generation(runner)
        runner.complete("Generic analytics candidate data generated and validated")
    else:
        runner.skip(
            "Yahoo-dependent generic analytics data generation; use "
            "--include-generic-data-generation."
        )

    runner.phase(2, "Run packaged Audit health checks")
    _run_audit_demo_health(
        runner,
        write_packaged_assets=args.write_packaged_assets,
    )
    if args.write_packaged_assets:
        runner.asset_note(
            "Packaged Axys/APX Audit CSV assets may have been rewritten."
        )
    runner.complete("Packaged Audit health checks")

    runner.phase(3, "Generate and validate Audit demo reports")
    _run_audit_demo_report_checks(runner)
    runner.complete("Portfolio/security Audit demo reports")

    runner.phase(4, "Smoke-test Analytics setup workspaces")
    _run_analytics_setup_smoke_tests(runner)
    runner.complete("Analytics setup workspace scripts")

    runner.phase(5, "Release-asset refresh")
    refresh_images, refresh_pdf = _release_asset_refresh_scope(
        build=args.build,
        refresh_images=args.refresh_images,
    )
    if refresh_pdf:
        _run_readme_asset_refresh(runner, refresh_images=refresh_images)
        if refresh_images:
            runner.asset_note(
                "README images under docs/images/readme and PPAR.pdf may have been "
                "refreshed."
            )
            runner.complete("README images and product-overview PDF")
        else:
            runner.asset_note("PPAR.pdf was refreshed from the current README.md.")
            runner.complete("Product-overview PDF")
    else:
        runner.skip(
            "Release-asset refresh; --build refreshes PPAR.pdf and "
            "--refresh-images refreshes both images and PDF."
        )

    runner.phase(6, "Run release-candidate scale regression checks")
    _run_scale_regression_check(runner)
    runner.complete("Analytics/Audit scale regression checks at 500x")

    runner.phase(7, "Run project checks")
    if not args.skip_project_check:
        _run_project_checks(runner, quick=args.quick, build=args.build)
        runner.complete("Project checks")
    else:
        runner.skip("Project checks; --skip-project-check was supplied.")

    runner.print_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
