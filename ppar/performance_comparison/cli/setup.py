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
    return f"""# PPAR Setup

This folder was created by:

```bash
ppar setup {site_path}
```

Use these commands to run the starter reports:

```bash
ppar analytics {site_path / _ANALYTICS_DIRECTORY}
ppar performance_comparison {site_path / _PERFORMANCE_COMPARISON_DIRECTORY}
```

`ppar perfcomp` is a shorter alias for `ppar performance_comparison`.

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

Analytics:

```bash
ppar analytics {site_path / _ANALYTICS_DIRECTORY}
```

Success means the `analytics/output` folder contains attribution HTML, chart PNG,
and risk-statistics HTML files.

Performance Comparison:

```bash
ppar performance_comparison {site_path / _PERFORMANCE_COMPARISON_DIRECTORY}
```

By default, this writes portfolio and security comparison reports when the
required source files are available. Use `--report portfolio`,
`--report security`, or `--report both` when you want to choose explicitly.

Success means the `performance_comparison/output` folder contains one or both of:

```text
portfolio/report.xlsx
portfolio/report.html
security/report.xlsx
security/report.html
```

## Customizing

### Analytics

Purpose: explain portfolio performance versus a benchmark with attribution,
contribution, charts, and risk statistics.

1. Replace `analytics/portperf.csv` with a portfolio-performance IMEX/export CSV.
2. Replace `analytics/secperf.csv` with a security-performance IMEX/export CSV.
3. Edit `analytics/ppar.yaml`.
4. Run `ppar analytics {site_path / _ANALYTICS_DIRECTORY}`.

The CSVs should keep the starter headers when possible. If your export headers
are different, add or adjust the column-mapping sections in `analytics/ppar.yaml`.

### Performance Comparison

Purpose: compare two Axys/APX source-data snapshots and explain changed reported
performance.

1. Replace the CSVs in `performance_comparison/snapshot_a`.
2. Replace the CSVs in `performance_comparison/snapshot_b`.
3. Edit `performance_comparison/ppar.yaml`.
4. Run `ppar performance_comparison {site_path / _PERFORMANCE_COMPARISON_DIRECTORY}`.

For the first pass, focus on IMEX/export CSVs for:

- `portperf.csv`: portfolio returns by portfolio and period.
- `secperf.csv`: security returns by portfolio, security, and period.
- `holdings.csv`: beginning and ending holdings values.
- `transactions.csv`: dated transaction rows used for Modified Dietz flows and
  performance-impacting activity.

Keep native transaction codes and security identifiers case-sensitive.

## YAML Files

The YAML files are intentionally heavily commented. Treat them as the working
user manual for filenames, required sections, transaction rules, and local
overrides.

Start by editing only:

- portfolio and benchmark codes in `analytics/ppar.yaml`;
- filenames or column mappings if your CSV headers differ;
- transaction rules only after your site evidence proves a local override.

Avoid changing transaction meanings from code alone. Ambiguous Axys/APX-style
codes need source/destination, special-security, REP/report, or reviewed local
evidence before they should be treated as external flows, fees, income, or
transfers.
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
    print(f"PPAR setup complete in: {result['site_directory']}")
    print(f"Local README: {result['readme_path']} ({result['readme_status']})")
    print()
    print(
        "Analytics folder: "
        f"{result['analytics_directory']} ({result['analytics_status']})"
    )
    print(f"Analytics config: {result['analytics_config_path']}")
    print(f"Analytics run command: ppar analytics {result['analytics_directory']}")
    print()
    print(
        "Performance comparison folder: "
        f"{result['comparison_directory']} ({result['comparison_status']})"
    )
    print(f"Performance comparison config: {result['comparison_config_path']}")
    print(
        "Performance comparison run command: "
        f"ppar performance_comparison {result['comparison_directory']}"
    )
    print()
    if "missing_files" in result:
        print("Next step: add these portfolio source files, then run setup again:")
        print(result["missing_files"])
        return

    print(
        "Open "
        f"{result['site_directory'] / 'README.md'} "
        '(section "Customizing") to customize with your own data.'
    )


if __name__ == "__main__":
    raise SystemExit(main())
