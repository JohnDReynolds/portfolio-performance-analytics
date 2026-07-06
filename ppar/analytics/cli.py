"""Command-line reporting for analytics configured from a local site folder."""

from __future__ import annotations

# Python imports
import argparse
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Final

# Third-party imports
import yaml

# Project imports
from ppar._chart_console import quiet_matplotlib_startup
from ppar.errors import PpaError
import ppar.utilities as util

_CONFIG_FILE_NAME: Final[str] = "ppar.yaml"
_ANALYTICS_SECTION: Final[str] = "analytics"
_DEFAULT_OUTPUT_DIRECTORY: Final[str] = "output"
_DEFAULT_CLASSIFICATION: Final[str] = "Economic Sector"


def main(argv: list[str] | None = None) -> int:
    """Run analytics reports from a local ``ppar.yaml`` site folder.

    Args:
        argv: Optional command-line arguments excluding the executable name.

    Returns:
        Process exit code. ``0`` indicates that analytics artifacts were written.
    """
    args = _argument_parser().parse_args(argv)
    try:
        run_analytics(
            _default_site_directory(args.site_directory),
            portfolio_code=args.portfolio,
            benchmark_code=args.benchmark,
            frequency_value=args.frequency,
            output_directory=args.output,
        )
    except (PpaError, ValueError) as error:
        print(f"Analytics failed: {error}", file=sys.stderr)
        return 1

    return 0


def run_analytics(
    site_directory: Path | str,
    *,
    portfolio_code: str | None = None,
    benchmark_code: str | None = None,
    frequency_value: str | None = None,
    output_directory: Path | None = None,
) -> Path:
    """Write analytics artifacts for a configured Axys/APX analytics folder.

    Args:
        site_directory: Folder containing ``ppar.yaml`` and analytics CSV files.
            Accepts a ``Path`` or string path.
        portfolio_code: Optional portfolio code override.
        benchmark_code: Optional benchmark portfolio code override.
        frequency_value: Optional reporting frequency override.
        output_directory: Optional output directory override.

    Returns:
        Directory that received analytics artifacts.

    Raises:
        PpaError: If required settings are missing or source data cannot be read.
        ValueError: If the reporting frequency cannot be interpreted.
    """
    site_path = Path(site_directory).expanduser()
    config_path = (site_path / _CONFIG_FILE_NAME).resolve()
    config_values = _load_config_values(config_path)
    analytics_settings = _analytics_settings(config_values)

    selected_portfolio = portfolio_code or _required_setting(
        analytics_settings,
        "portfolio",
        _ANALYTICS_SECTION,
    )
    selected_benchmark = benchmark_code or _required_setting(
        analytics_settings,
        "benchmark",
        _ANALYTICS_SECTION,
    )
    selected_frequency_value = frequency_value or str(
        analytics_settings.get("frequency", "quarterly")
    )
    selected_output = output_directory or site_path / str(
        analytics_settings.get("output_directory", _DEFAULT_OUTPUT_DIRECTORY)
    )
    classification_name = _classification_name(config_values)

    selected_output.mkdir(parents=True, exist_ok=True)
    original_cache_env = {
        "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR"),
        "XDG_CACHE_HOME": os.environ.get("XDG_CACHE_HOME"),
    }
    try:
        with tempfile.TemporaryDirectory(prefix="ppar_chart_cache_") as cache_directory:
            cache_path = Path(cache_directory)
            os.environ.setdefault("MPLCONFIGDIR", str(cache_path / "matplotlib"))
            os.environ.setdefault("XDG_CACHE_HOME", str(cache_path / "cache"))
            quiet_matplotlib_startup()

            # Import after cache env vars are set; analytics/chart modules may
            # initialize Matplotlib during package import on some systems.
            from ppar.axys import AxysData
            from ppar.demos.analytics_demo_outputs import (
                demo_frequency_from_string,
                write_analytics_demo_outputs,
            )

            selected_frequency = demo_frequency_from_string(selected_frequency_value)
            axys_data = AxysData(config_path)
            portfolio = axys_data.get_portfolio(selected_portfolio)
            benchmark = axys_data.get_portfolio(selected_benchmark)
            analytics = portfolio.to_analytics(benchmark, frequency=selected_frequency)
            written_paths = write_analytics_demo_outputs(
                analytics,
                selected_output,
                sector_classification_name=classification_name,
            )
    finally:
        for key, original_value in original_cache_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value

    if written_paths:
        print("Open these files to review analytics output:")
        for path in written_paths:
            print(f"  {path}")
    return selected_output


def _argument_parser() -> argparse.ArgumentParser:
    """Return the analytics argument parser."""
    parser = argparse.ArgumentParser(
        prog="ppar analytics",
        description="Write Axys/APX analytics reports from a configured site folder.",
        epilog=(
            "Examples:\n"
            "  ppar analytics ./my_ppar_data/analytics\n"
            "  ppar analytics"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "site_directory",
        nargs="?",
        type=Path,
        help=(
            "Folder containing analytics ppar.yaml and source CSV files. "
            "Defaults to the current folder."
        ),
    )
    parser.add_argument(
        "--portfolio",
        help="Portfolio code. Defaults to analytics.portfolio in ppar.yaml.",
    )
    parser.add_argument(
        "--benchmark",
        help="Benchmark portfolio code. Defaults to analytics.benchmark in ppar.yaml.",
    )
    parser.add_argument(
        "-f",
        "--frequency",
        help="Reporting frequency: monthly, quarterly, yearly, or m/q/y.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory. Defaults to analytics.output_directory in ppar.yaml.",
    )
    return parser


def _default_site_directory(site_directory: Path | None) -> Path:
    """Return the explicit or conventional analytics site directory."""
    if site_directory is not None:
        return site_directory
    return Path.cwd()


def _load_config_values(config_path: Path) -> dict[str, Any]:
    """Load an analytics YAML file and return its root mapping."""
    if not config_path.exists():
        raise PpaError(
            f"{config_path} does not exist. Run from the analytics folder "
            "or pass the folder.",
            504,
        )
    with open(config_path, "r", encoding=util.ENCODING) as file:
        try:
            values: Any = yaml.safe_load(file)
        except Exception as error:
            raise PpaError(f"Invalid YAML in {config_path}: {error}", 504) from error
    if not isinstance(values, dict):
        raise PpaError(f"{config_path} must contain a YAML mapping.", 504)
    return values


def _analytics_settings(config_values: dict[str, Any]) -> dict[str, Any]:
    """Return the optional analytics settings mapping."""
    settings = config_values.get(_ANALYTICS_SECTION, {})
    if not isinstance(settings, dict):
        raise PpaError(f"{_ANALYTICS_SECTION} must be a mapping.", 504)
    return settings


def _required_setting(
    settings: dict[str, Any],
    key: str,
    section_name: str,
) -> str:
    """Return a required string setting from a named YAML section."""
    value = settings.get(key)
    if not isinstance(value, str) or not value:
        raise PpaError(f"{section_name}.{key} must be set in ppar.yaml.", 504)
    return value


def _classification_name(config_values: dict[str, Any]) -> str:
    """Return the configured default classification for attribution output."""
    defaults = config_values.get("defaults", {})
    if not isinstance(defaults, dict):
        raise PpaError("defaults must be a mapping.", 504)
    value = defaults.get("classification", _DEFAULT_CLASSIFICATION)
    if not isinstance(value, str) or not value:
        raise PpaError("defaults.classification must be a string.", 504)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
