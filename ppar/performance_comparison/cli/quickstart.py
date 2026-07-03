"""Create a one-folder starter setup for performance comparison."""

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
from ppar.performance_comparison import (
    compare_snapshots,
    write_performance_comparison_report_bundle,
)
from ppar.performance_comparison.config_validation import validate_config
from ppar.performance_comparison.specification import (
    PORTFOLIO_COMPARISON_LEVEL,
    SECURITY_COMPARISON_LEVEL,
)
import ppar.utilities as util

_CONFIG_FILE_NAME: Final[str] = "ppar.yaml"
_SNAPSHOT_A_DIR: Final[str] = "snapshot_a"
_SNAPSHOT_B_DIR: Final[str] = "snapshot_b"
_OUTPUT_DIR: Final[str] = "output"
_PACKAGED_AXYS_RESOURCE: Final[str] = "ppar.demos.data.axys"
_PACKAGED_COMPARISON_YAML: Final[str] = "axys_performance_comparison.yaml"
_EXPECTED_SNAPSHOT_FILES: Final[tuple[str, ...]] = (
    "portperf.csv",
    "secperf.csv",
    "holdings.csv",
    "transactions.csv",
)
_REPORT_CHOICES: Final[tuple[str, ...]] = (
    "both",
    PORTFOLIO_COMPARISON_LEVEL,
    SECURITY_COMPARISON_LEVEL,
    "none",
)


def main(argv: list[str] | None = None) -> int:
    """Run the performance-comparison quickstart command.

    Args:
        argv: Optional command-line arguments excluding the executable name.

    Returns:
        Process exit code. ``0`` indicates that either starter folders were
        created or the starter config was valid and requested report bundles
        were written.
    """
    args = _argument_parser().parse_args(argv)
    try:
        result = run_quickstart(
            args.site_directory,
            reports=args.reports,
            overwrite=args.overwrite,
        )
    except PpaError as error:
        print(f"Quickstart failed: {error}", file=sys.stderr)
        return 1

    _print_success(result)
    return 0


def run_quickstart(
    site_directory: Path,
    *,
    reports: str = "both",
    overwrite: bool = False,
) -> dict[str, Path | str]:
    """Create ``ppar.yaml`` and optional report bundles for a site folder.

    Args:
        site_directory: Folder containing ``snapshot_a`` and ``snapshot_b``.
        reports: Report family to generate: ``"both"``, ``"portfolio"``,
            ``"security"``, or ``"none"``.
        overwrite: Whether to replace an existing ``ppar.yaml``.

    Returns:
        Paths and status labels for the site folder, config file, and any
        generated report bundles. When snapshot files are not present yet, the
        result contains setup guidance instead of report paths.

    Raises:
        PpaError: If the site path is an existing non-directory, if
            ``ppar.yaml`` cannot be written, or if validation/report generation
            fails.
    """
    if reports not in _REPORT_CHOICES:
        raise PpaError(
            f"reports must be one of: {', '.join(_REPORT_CHOICES)}.",
            504,
        )

    site_path = Path(site_directory).expanduser()
    setup_status = _ensure_site_layout(site_path)
    missing_files = _missing_snapshot_files(site_path)
    if missing_files:
        return {
            "site_directory": site_path,
            "setup_status": setup_status,
            "missing_files": "\n".join(missing_files),
        }

    config_path = site_path / _CONFIG_FILE_NAME
    config_status = _ensure_config(config_path, overwrite=overwrite)

    validate_config(config_path)
    result: dict[str, Path | str] = {
        "site_directory": site_path,
        "config_path": config_path,
        "config_status": config_status,
    }

    if reports in ("both", PORTFOLIO_COMPARISON_LEVEL):
        result["portfolio_report"] = _write_report_bundle(
            config_path,
            site_path / _OUTPUT_DIR / PORTFOLIO_COMPARISON_LEVEL,
            comparison_level=PORTFOLIO_COMPARISON_LEVEL,
        )
    if reports in ("both", SECURITY_COMPARISON_LEVEL):
        result["security_report"] = _write_report_bundle(
            config_path,
            site_path / _OUTPUT_DIR / SECURITY_COMPARISON_LEVEL,
            comparison_level=SECURITY_COMPARISON_LEVEL,
        )
    return result


def _argument_parser() -> argparse.ArgumentParser:
    """Return the quickstart argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Create ppar.yaml and report bundles for a folder with snapshot_a "
            "and snapshot_b extracts."
        ),
    )
    parser.add_argument(
        "site_directory",
        type=Path,
        help="Folder containing snapshot_a and snapshot_b.",
    )
    parser.add_argument(
        "--reports",
        choices=_REPORT_CHOICES,
        default="both",
        help="Report family to generate. Defaults to both.",
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
    """Return user-facing labels for expected files not present yet."""
    missing_files: list[str] = []
    for snapshot_dir in (_SNAPSHOT_A_DIR, _SNAPSHOT_B_DIR):
        snapshot_path = site_path / snapshot_dir
        for file_name in _EXPECTED_SNAPSHOT_FILES:
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
    resource = files(_PACKAGED_AXYS_RESOURCE).joinpath(_PACKAGED_COMPARISON_YAML)
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
    return values


def _write_report_bundle(
    config_path: Path,
    output_directory: Path,
    *,
    comparison_level: str,
) -> Path:
    """Write one report bundle and return the workbook path."""
    findings = compare_snapshots(
        config_path,
        comparison_level=comparison_level,
    )
    title = (
        "Portfolio Performance Comparison"
        if comparison_level == PORTFOLIO_COMPARISON_LEVEL
        else "Security Performance Comparison"
    )
    paths = write_performance_comparison_report_bundle(
        findings,
        output_directory,
        title=title,
        include_workbook=True,
        comparison_path=config_path,
        comparison_level=comparison_level,
    )
    workbook = paths.get("review_workbook")
    if workbook is None:
        raise PpaError(f"Report bundle did not write report.xlsx in {output_directory}.", 999)
    return workbook


def _print_success(result: dict[str, Path | str]) -> None:
    """Print a concise user handoff."""
    if "missing_files" in result:
        print("PPAR quickstart folders are ready")
        print(f"Site folder: {result['site_directory']}")
        print("Next step: add these source files, then run quickstart again:")
        print(result["missing_files"])
        return

    print("PPAR quickstart complete")
    print(f"Site folder: {result['site_directory']}")
    print(f"Config: {result['config_path']} ({result['config_status']})")
    if "portfolio_report" in result:
        print(f"Portfolio report: {result['portfolio_report']}")
    if "security_report" in result:
        print(f"Security report: {result['security_report']}")


if __name__ == "__main__":
    raise SystemExit(main())
