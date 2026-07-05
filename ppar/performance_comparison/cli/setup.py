"""Create an Axys/APX starter workspace for PPAR."""

from __future__ import annotations

# Python imports
import argparse
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
import sys
from typing import Final

# Project imports
from ppar.errors import PpaError
from ppar.performance_comparison.config_validation import validate_config
import ppar.utilities as util

_CONFIG_FILE_NAME: Final[str] = "ppar.yaml"
_ANALYTICS_DIRECTORY: Final[str] = "analytics"
_PERFORMANCE_COMPARISON_DIRECTORY: Final[str] = "performance_comparison"
_SNAPSHOT_A_DIR: Final[str] = "snapshot_a"
_SNAPSHOT_B_DIR: Final[str] = "snapshot_b"
_PACKAGED_DEMO_RESOURCE: Final[str] = "ppar.demos.data"
_PACKAGED_ANALYTICS_DIRECTORY: Final[str] = "axysapx_analytics"
_PACKAGED_COMPARISON_DIRECTORY: Final[str] = "axysapx_performance_comparison"
_PACKAGED_ANALYTICS_YAML: Final[str] = "axysapx_analytics.yaml"
_PACKAGED_COMPARISON_YAML: Final[str] = "axysapx_performance_comparison.yaml"
_PACKAGED_SETUP_GUIDE: Final[str] = "SETUP.md"
_ANALYTICS_SETUP_FILES: Final[tuple[str, ...]] = (
    "portperf.csv",
    "secperf.csv",
)
_PORTFOLIO_SETUP_FILES: Final[tuple[str, ...]] = (
    "portperf.csv",
    "holdings.csv",
    "transactions.csv",
)


def main(argv: list[str] | None = None) -> int:
    """Run the Axys/APX setup command.

    Args:
        argv: Optional command-line arguments excluding the executable name.

    Returns:
        Process exit code. ``0`` indicates that the starter workspace exists
        and performance comparison setup validates.
    """
    args = _argument_parser().parse_args(argv)
    if args.guide:
        print(_setup_guide_text())
        return 0
    if args.site_directory is None:
        print(
            "Setup failed: site_directory is required unless --guide is used.",
            file=sys.stderr,
        )
        return 1
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
    """Create an Axys/APX starter workspace with analytics and comparison folders.

    Args:
        site_directory: Folder that will receive ``analytics`` and
            ``performance_comparison`` subfolders.
        overwrite: Whether to replace existing starter files.

    Returns:
        Paths and status labels for the starter workspace.

    Raises:
        PpaError: If a destination path is an existing non-directory, if starter
            files cannot be written, or if performance-comparison validation
            fails.
    """
    site_path = Path(site_directory).expanduser()
    setup_status = _ensure_directory(site_path)
    analytics_path = site_path / _ANALYTICS_DIRECTORY
    comparison_path = site_path / _PERFORMANCE_COMPARISON_DIRECTORY
    readme_status = _write_text_file(
        site_path / "README.md",
        _starter_readme_text(site_path),
        overwrite=overwrite,
    )
    analytics_status = _ensure_analytics_starter(analytics_path, overwrite=overwrite)
    comparison_status = _ensure_comparison_starter(
        comparison_path,
        overwrite=overwrite,
    )
    comparison_config_path = comparison_path / _CONFIG_FILE_NAME

    result: dict[str, Path | str] = {
        "site_directory": site_path,
        "setup_status": setup_status,
        "readme_path": site_path / "README.md",
        "readme_status": readme_status,
        "analytics_directory": analytics_path,
        "analytics_status": analytics_status,
        "analytics_config_path": analytics_path / _CONFIG_FILE_NAME,
        "comparison_directory": comparison_path,
        "comparison_status": comparison_status,
        "comparison_config_path": comparison_config_path,
    }
    missing_files = _missing_snapshot_files(comparison_path)
    if missing_files:
        result["missing_files"] = "\n".join(missing_files)
        return result

    validate_config(comparison_config_path)
    result["validation_status"] = "performance-comparison-ready"
    return result


def _argument_parser() -> argparse.ArgumentParser:
    """Return the setup argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Create an Axys/APX starter workspace with analytics and "
            "performance-comparison folders."
        ),
    )
    parser.add_argument(
        "site_directory",
        nargs="?",
        type=Path,
        help="Folder that will receive analytics and performance_comparison.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing starter files with packaged starter files.",
    )
    parser.add_argument(
        "--guide",
        action="store_true",
        help="Print the Axys/APX setup guide and exit without creating files.",
    )
    return parser


def _setup_guide_text() -> str:
    """Return the packaged setup guide text."""
    resource = files(_PACKAGED_DEMO_RESOURCE).joinpath(
        _PACKAGED_COMPARISON_DIRECTORY,
        _PACKAGED_SETUP_GUIDE,
    )
    return resource.read_text(encoding=util.ENCODING).rstrip()


def _ensure_directory(directory: Path) -> str:
    """Create a directory when needed and return a status label."""
    if directory.exists() and not directory.is_dir():
        raise PpaError(
            f"{directory} exists but is not a directory.",
            802,
        )
    if directory.exists():
        return "existing"
    directory.mkdir(parents=True)
    return "created"


def _missing_snapshot_files(site_path: Path) -> list[str]:
    """Return user-facing labels for expected portfolio files not present yet."""
    missing_files: list[str] = []
    for snapshot_dir in (_SNAPSHOT_A_DIR, _SNAPSHOT_B_DIR):
        snapshot_path = site_path / snapshot_dir
        for file_name in _PORTFOLIO_SETUP_FILES:
            if not (snapshot_path / file_name).exists():
                missing_files.append(
                    f"{_PERFORMANCE_COMPARISON_DIRECTORY}/{snapshot_dir}/{file_name}"
                )
    return missing_files


def _ensure_analytics_starter(directory: Path, *, overwrite: bool) -> str:
    """Copy Axys/APX analytics starter files into ``directory``."""
    status = _ensure_directory(directory)
    source_directory = files(_PACKAGED_DEMO_RESOURCE).joinpath(
        _PACKAGED_ANALYTICS_DIRECTORY
    )
    config_status = _copy_resource_file(
        source_directory.joinpath(_PACKAGED_ANALYTICS_YAML),
        directory / _CONFIG_FILE_NAME,
        overwrite=overwrite,
    )
    for file_name in _ANALYTICS_SETUP_FILES:
        _copy_resource_file(
            source_directory.joinpath(file_name),
            directory / file_name,
            overwrite=overwrite,
        )
    return _combined_status(status, config_status)


def _ensure_comparison_starter(directory: Path, *, overwrite: bool) -> str:
    """Copy Axys/APX performance-comparison starter files into ``directory``."""
    status = _ensure_directory(directory)
    source_directory = files(_PACKAGED_DEMO_RESOURCE).joinpath(
        _PACKAGED_COMPARISON_DIRECTORY
    )
    config_status = _write_text_file(
        directory / _CONFIG_FILE_NAME,
        _starter_comparison_config_text(
            source_directory.joinpath(_PACKAGED_COMPARISON_YAML)
        ),
        overwrite=overwrite,
    )
    for snapshot_dir in (_SNAPSHOT_A_DIR, _SNAPSHOT_B_DIR):
        _copy_resource_tree(
            source_directory.joinpath(snapshot_dir),
            directory / snapshot_dir,
            overwrite=overwrite,
        )
    return _combined_status(status, config_status)


def _starter_comparison_config_text(resource: Traversable) -> str:
    """Return the documented one-file performance-comparison starter YAML."""
    text = resource.read_text(encoding=util.ENCODING)
    text = text.replace(
        "name: Axys/APX performance comparison demo",
        "name: PPAR performance comparison",
    )
    text = text.replace(
        "    vendor: axys\n    schema: axysapx_column_mappings.yaml\n",
        "    vendor: axysapx\n",
    )
    return text


def _starter_readme_text(site_path: Path) -> str:
    """Return the local setup README copied into a user's starter workspace."""
    return f"""# PPAR

This folder was created by:

```bash
ppar setup {site_path}
```

## Folder Map

```text
{site_path.name}/
  analytics/
    ppar.yaml
    portperf.csv
    secperf.csv
  performance_comparison/
    ppar.yaml
    snapshot_a/
      portperf.csv
      holdings.csv
      transactions.csv
      secperf.csv
    snapshot_b/
      portperf.csv
      holdings.csv
      transactions.csv
      secperf.csv
```

## Running

### Analytics

Analytics produces holdings-based multi-period performance attribution,
contribution, and benchmark-relative ex-post risk statistics. It uses the
Brinson-Fachler methodology for calculating attribution effects, and uses the
Carino method for logarithmically-smoothing cumulative effects over multi-period
time frames.

```bash
ppar analytics {site_path / _ANALYTICS_DIRECTORY}
```

Success means the `analytics/output` folder contains attribution HTML, chart PNG,
and risk-statistics HTML files.

### Performance Comparison

Performance Comparison answers the question "Why did my reported performance
change?". It does this by comparing two source-data snapshots, and then
attributing the performance difference to specific changes in holdings market
values and transaction cash flows.

```bash
ppar performance_comparison {site_path / _PERFORMANCE_COMPARISON_DIRECTORY}
```

Success means the `performance_comparison/output` folder contains:

```text
portfolio/report.xlsx
security/report.xlsx
```

If you only want to create the portfolio reports, you can add the option
`--report portfolio`.

If you only want to create the security reports, you can add the option
`--report security`.

## Customizing

### Analytics

1. Replace `analytics/portperf.csv` with your own portfolio-performance IMEX CSV.
2. Replace `analytics/secperf.csv` with your own security-performance IMEX CSV.
3. Edit `analytics/ppar.yaml` with your own specifications.
4. Run `ppar analytics {site_path / _ANALYTICS_DIRECTORY}`.

### Performance Comparison

1. Replace the CSVs in `performance_comparison/snapshot_a`.
2. Replace the CSVs in `performance_comparison/snapshot_b` with your own data.
3. Edit `performance_comparison/ppar.yaml` with your own specifications.
4. Run `ppar performance_comparison {site_path / _PERFORMANCE_COMPARISON_DIRECTORY}`.

## YAML Files

The YAML files are intentionally heavily commented. Treat them as the working
user manual for filenames, required sections, transaction rules, and local
overrides.
"""


def _copy_resource_tree(
    source_directory: Traversable,
    destination_directory: Path,
    *,
    overwrite: bool,
) -> None:
    """Copy resource files from one packaged directory to a local directory."""
    _ensure_directory(destination_directory)
    for resource in source_directory.iterdir():
        destination = destination_directory / resource.name
        if resource.is_dir():
            _copy_resource_tree(resource, destination, overwrite=overwrite)
        elif resource.is_file():
            _copy_resource_file(resource, destination, overwrite=overwrite)


def _copy_resource_file(
    resource: Traversable,
    destination: Path,
    *,
    overwrite: bool,
) -> str:
    """Copy one packaged resource file and return its status label."""
    if destination.exists() and not overwrite:
        return "existing"
    existed_before = destination.exists()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(resource.read_bytes())
    return "updated" if existed_before else "written"


def _write_text_file(destination: Path, text: str, *, overwrite: bool) -> str:
    """Write a text starter file and return its status label."""
    if destination.exists() and not overwrite:
        return "existing"
    existed_before = destination.exists()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding=util.ENCODING)
    return "updated" if existed_before else "written"


def _combined_status(directory_status: str, config_status: str) -> str:
    """Return a compact status label for a starter folder."""
    if directory_status == "created":
        return "created"
    return config_status


def _print_success(result: dict[str, Path | str]) -> None:
    """Print a concise user handoff."""
    print(f"PPAR setup complete: {result['site_directory']}")
    print()
    print("To run Analytics:")
    print(f"  ppar analytics {result['analytics_directory']}")
    print()
    print("To run Performance Comparison:")
    print(f"  ppar performance_comparison {result['comparison_directory']}")
    print()
    if "missing_files" in result:
        print("Next step: add these portfolio source files, then run setup again:")
        print(result["missing_files"])
        return

    print("To customize with your own data:")
    print(f"  Open {result['site_directory'] / 'README.md'}")


if __name__ == "__main__":
    raise SystemExit(main())
