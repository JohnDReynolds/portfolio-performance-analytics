"""Create a one-folder site setup for performance comparison."""

from __future__ import annotations

# Python imports
import argparse
from importlib.resources import files
from pathlib import Path
import sys
from typing import Any, Final

# Third-party imports
import yaml

# Project imports
from ppar.errors import PpaError
from ppar.performance_comparison.config_validation import validate_config
from ppar.performance_comparison.specification import PORTFOLIO_COMPARISON_LEVEL
import ppar.utilities as util

_CONFIG_FILE_NAME: Final[str] = "ppar.yaml"
_SNAPSHOT_A_DIR: Final[str] = "snapshot_a"
_SNAPSHOT_B_DIR: Final[str] = "snapshot_b"
_PACKAGED_AXYS_RESOURCE: Final[str] = "ppar.demos.data"
_PACKAGED_AXYS_DIRECTORY: Final[str] = "axys_performance_comparison"
_PACKAGED_COMPARISON_YAML: Final[str] = "axys_performance_comparison.yaml"
_PORTFOLIO_SETUP_FILES: Final[tuple[str, ...]] = (
    "portperf.csv",
    "holdings.csv",
    "transactions.csv",
)


def main(argv: list[str] | None = None) -> int:
    """Run the performance-comparison setup command.

    Args:
        argv: Optional command-line arguments excluding the executable name.

    Returns:
        Process exit code. ``0`` indicates that the starter folders and
        ``ppar.yaml`` exist. If portfolio source files are present, setup also
        validates the portfolio configuration.
    """
    args = _argument_parser().parse_args(argv)
    try:
        result = run_setup(
            args.site_directory,
            overwrite=args.overwrite,
        )
    except PpaError as error:
        print(f"Setup failed: {error}", file=sys.stderr)
        return 1

    _print_success(result)
    return 0


def run_setup(
    site_directory: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path | str]:
    """Create ``ppar.yaml`` and validate portfolio setup when files exist.

    Args:
        site_directory: Folder containing ``snapshot_a`` and ``snapshot_b``.
        overwrite: Whether to replace an existing ``ppar.yaml``.

    Returns:
        Paths and status labels for the site folder and generated config file.
        When portfolio source files are not present yet, the result contains
        setup guidance instead of validation status.

    Raises:
        PpaError: If the site path is an existing non-directory, if
            ``ppar.yaml`` cannot be written, or if portfolio validation fails.
    """
    site_path = Path(site_directory).expanduser()
    setup_status = _ensure_site_layout(site_path)
    config_path = site_path / _CONFIG_FILE_NAME
    config_status = _ensure_config(config_path, overwrite=overwrite)

    result: dict[str, Path | str] = {
        "site_directory": site_path,
        "setup_status": setup_status,
        "config_path": config_path,
        "config_status": config_status,
    }
    missing_files = _missing_snapshot_files(site_path)
    if missing_files:
        result["missing_files"] = "\n".join(missing_files)
        return result

    validate_config(config_path)
    result["validation_status"] = "portfolio-ready"
    return result


def _argument_parser() -> argparse.ArgumentParser:
    """Return the setup argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Create a ppar.yaml setup for a folder with snapshot_a and "
            "snapshot_b extracts."
        ),
    )
    parser.add_argument(
        "site_directory",
        type=Path,
        help="Folder containing snapshot_a and snapshot_b.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing ppar.yaml with the packaged starter config.",
    )
    return parser


def _ensure_site_layout(site_path: Path) -> str:
    """Create missing site/snapshot folders and return a status label."""
    if site_path.exists() and not site_path.is_dir():
        raise PpaError(
            f"{site_path} exists but is not a directory.",
            802,
        )

    created: list[Path] = []
    if not site_path.exists():
        site_path.mkdir(parents=True)
        created.append(site_path)

    for snapshot_dir in (_SNAPSHOT_A_DIR, _SNAPSHOT_B_DIR):
        path = site_path / snapshot_dir
        if path.exists() and not path.is_dir():
            raise PpaError(
                f"{path} exists but is not a directory.",
                802,
            )
        if not path.exists():
            path.mkdir()
            created.append(path)
    return "created" if created else "existing"


def _missing_snapshot_files(site_path: Path) -> list[str]:
    """Return user-facing labels for expected portfolio files not present yet."""
    missing_files: list[str] = []
    for snapshot_dir in (_SNAPSHOT_A_DIR, _SNAPSHOT_B_DIR):
        snapshot_path = site_path / snapshot_dir
        for file_name in _PORTFOLIO_SETUP_FILES:
            if not (snapshot_path / file_name).exists():
                missing_files.append(f"{snapshot_dir}/{file_name}")
    return missing_files


def _ensure_config(config_path: Path, *, overwrite: bool) -> str:
    """Create ``ppar.yaml`` when needed and return its status label."""
    if config_path.exists() and not overwrite:
        return "existing"

    existed_before = config_path.exists()
    config_values = _starter_config_values()
    with open(config_path, "w", encoding=util.ENCODING) as file:
        yaml.safe_dump(
            config_values,
            file,
            sort_keys=False,
            allow_unicode=False,
        )
    return "updated" if existed_before else "written"


def _starter_config_values() -> dict[str, Any]:
    """Return a one-file starter configuration based on the packaged Axys YAML."""
    resource = files(_PACKAGED_AXYS_RESOURCE).joinpath(
        _PACKAGED_AXYS_DIRECTORY,
        _PACKAGED_COMPARISON_YAML,
    )
    values = yaml.safe_load(resource.read_text(encoding=util.ENCODING))
    if not isinstance(values, dict):
        raise PpaError("Packaged starter YAML must be a mapping.", 504)

    snapshots = values.get("snapshots")
    if not isinstance(snapshots, dict):
        raise PpaError("Packaged starter YAML must define snapshots.", 504)
    snapshots["a"] = {
        "label": _SNAPSHOT_A_DIR,
        "path": _SNAPSHOT_A_DIR,
    }
    snapshots["b"] = {
        "label": _SNAPSHOT_B_DIR,
        "path": _SNAPSHOT_B_DIR,
    }

    comparison = values.get("comparison")
    if not isinstance(comparison, dict):
        comparison = {}
        values["comparison"] = comparison
    comparison["name"] = "PPAR performance comparison"
    comparison["level"] = PORTFOLIO_COMPARISON_LEVEL
    values.pop("security_return_reconstruction", None)
    return values


def _print_success(result: dict[str, Path | str]) -> None:
    """Print a concise user handoff."""
    print("PPAR setup complete")
    print(f"Site folder: {result['site_directory']}")
    print(f"Config: {result['config_path']} ({result['config_status']})")
    if "missing_files" in result:
        print("Next step: add these portfolio source files, then run setup again:")
        print(result["missing_files"])
        return

    print("Portfolio setup validated.")
    print(f"Next step: ppar report {result['site_directory']}")


if __name__ == "__main__":
    raise SystemExit(main())
