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
import sysconfig
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


def _run(
    command: Sequence[str | Path],
    *,
    cwd: Path = _PROJECT_ROOT,
    isolate_python_path: bool = False,
) -> None:
    """Run a project check command and stop on failure.

    Args:
        command: Command and arguments to execute.
        cwd: Directory from which to run the command.
        isolate_python_path: Whether to remove ``PYTHONPATH`` so an installed-
            package check cannot import from the source checkout through the
            caller's environment.

    Raises:
        subprocess.CalledProcessError: If the command exits with a non-zero status.
    """
    print(f"\n==> {_format_command(command)}", flush=True)
    _MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["XDG_CACHE_HOME"] = str(_CHECK_CACHE_DIR)
    env["MPLCONFIGDIR"] = str(_MPLCONFIGDIR)
    env["PIP_CACHE_DIR"] = str(_CHECK_CACHE_DIR / "pip")
    if isolate_python_path:
        env.pop("PYTHONPATH", None)
    subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
        check=True,
        env=env,
    )


def _virtual_environment_command(venv_path: Path, command_name: str) -> Path:
    """Return one executable path inside a virtual environment."""
    command_directory = "Scripts" if os.name == "nt" else "bin"
    suffix = ".exe" if os.name == "nt" else ""
    return venv_path / command_directory / f"{command_name}{suffix}"


def _virtual_environment_site_packages(venv_path: Path) -> Path:
    """Return the site-packages directory inside a virtual environment."""
    if os.name == "nt":
        return venv_path / "Lib" / "site-packages"
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return venv_path / "lib" / version / "site-packages"


def _run_installed_wheel_smoke(wheel_path: Path, smoke_root: Path) -> None:
    """Install and exercise a built wheel outside the source checkout.

    Args:
        wheel_path: Wheel produced by the current build.
        smoke_root: Temporary directory that will receive the virtual
            environment, setup workspace, and generated output.

    Notes:
        The temporary environment reuses the already-verified release
        environment's dependencies, then installs only the candidate wheel into
        its own site-packages. Commands run outside the checkout with
        ``PYTHONPATH`` removed, and an explicit origin check proves that ``ppar``
        resolves from the temporary environment.
    """
    smoke_root.mkdir(parents=True, exist_ok=True)
    venv_path = smoke_root / "venv"
    audit_path = smoke_root / "my_ppar_audit"
    analytics_path = smoke_root / "my_ppar_analytics"
    _run(
        [
            _VENV_PYTHON,
            "-m",
            "venv",
            venv_path,
        ],
        cwd=smoke_root,
        isolate_python_path=True,
    )
    smoke_python = _virtual_environment_command(venv_path, "python")
    smoke_ppar = _virtual_environment_command(venv_path, "ppar")
    _run(
        [
            smoke_python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            wheel_path,
        ],
        cwd=smoke_root,
        isolate_python_path=True,
    )
    dependency_site_packages = Path(sysconfig.get_paths()["purelib"]).resolve()
    dependency_link = (
        _virtual_environment_site_packages(venv_path)
        / "_ppar_release_dependencies.pth"
    )
    dependency_link.parent.mkdir(parents=True, exist_ok=True)
    dependency_link.write_text(
        f"{dependency_site_packages}\n",
        encoding="utf-8",
    )
    _run(
        [smoke_python, "-m", "pip", "check"],
        cwd=smoke_root,
        isolate_python_path=True,
    )
    origin_check = (
        "from pathlib import Path; import ppar, sys; "
        "package_path = Path(ppar.__file__).resolve(); "
        "environment_path = Path(sys.prefix).resolve(); "
        "assert package_path.is_relative_to(environment_path), "
        "f'{package_path} is not installed under {environment_path}'"
    )
    _run(
        [smoke_python, "-c", origin_check],
        cwd=smoke_root,
        isolate_python_path=True,
    )
    _run(
        [smoke_ppar, "setup", audit_path],
        cwd=smoke_root,
        isolate_python_path=True,
    )
    _run(
        [smoke_ppar, "audit", audit_path],
        cwd=smoke_root,
        isolate_python_path=True,
    )
    _run(
        [smoke_ppar, "setup", analytics_path, "--analytics"],
        cwd=smoke_root,
        isolate_python_path=True,
    )
    _run(
        [smoke_ppar, "analytics", analytics_path],
        cwd=smoke_root,
        isolate_python_path=True,
    )
    for comparison_level in ("portfolio", "security"):
        _run(
            [
                smoke_python,
                "-m",
                "ppar.audit.cli.validate_bundle",
                audit_path / "output" / comparison_level,
            ],
            cwd=smoke_root,
            isolate_python_path=True,
        )


def _run_build_check() -> None:
    """Build distributions and smoke-test the installed candidate wheel."""
    try:
        with tempfile.TemporaryDirectory(prefix="ppar-build-check-") as temp_dir:
            build_directory = Path(temp_dir)
            _run(
                [
                    _VENV_PYTHON,
                    "-m",
                    "build",
                    "--wheel",
                    "--sdist",
                    "--no-isolation",
                    "--outdir",
                    build_directory,
                ]
            )
            wheel_paths = sorted(build_directory.glob("*.whl"))
            if len(wheel_paths) != 1:
                raise RuntimeError(
                    "Build check expected exactly one wheel, found "
                    f"{len(wheel_paths)}."
                )
            _run_installed_wheel_smoke(
                wheel_paths[0],
                build_directory / "installed-wheel-smoke",
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
        help=(
            "Also build wheel and sdist, then install and smoke-test the wheel "
            "outside the source checkout."
        ),
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
