"""Command-line reporting for analytics configured from a local site folder."""

from __future__ import annotations

# Python imports
import argparse
from pathlib import Path
import sys
from typing import Any, Final

# Third-party imports
import yaml

# Project imports
from ppar.analytics import Analytics
from ppar.analytics.attribution import Chart, View
from ppar.analytics.frequency import Frequency
from ppar.axys import AxysData
from ppar.errors import PpaError
import ppar.utilities as util

_CONFIG_FILE_NAME: Final[str] = "ppar.yaml"
_ANALYTICS_SECTION: Final[str] = "analytics"
_DEFAULT_OUTPUT_DIRECTORY: Final[str] = "output"


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

    selected_frequency = _frequency_from_string(selected_frequency_value)
    axys_data = AxysData(config_path)
    portfolio = axys_data.get_portfolio(selected_portfolio)
    benchmark = axys_data.get_portfolio(selected_benchmark)
    analytics = portfolio.to_analytics(benchmark, frequency=selected_frequency)
    written_paths = _write_analytics_outputs(analytics, selected_output)

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


def _frequency_from_string(value: str | None) -> Frequency:
    """Return a reporting frequency from a lenient user string."""
    normalized = "" if value is None else value.strip().lower()
    if not normalized:
        return Frequency.QUARTERLY
    first_character = normalized[0]
    if first_character == "m":
        return Frequency.MONTHLY
    if first_character == "q":
        return Frequency.QUARTERLY
    if first_character == "y":
        return Frequency.YEARLY
    raise ValueError(
        "frequency must start with 'm', 'q', or 'y' "
        "(monthly, quarterly, or yearly)"
    )


def _write_analytics_outputs(
    analytics: Analytics,
    output_directory: Path,
) -> list[Path]:
    """Write the analytics HTML, PNG, and risk outputs used by the CLI."""
    written_paths: list[Path] = []

    attribution_by_security = analytics.get_attribution("Security")
    written_paths.append(
        _write_html(
            output_directory,
            "security_overall_attribution.html",
            attribution_by_security.to_html(View.OVERALL_ATTRIBUTION),
        )
    )

    attribution_by_sector = analytics.get_attribution()
    for view in (View.CUMULATIVE_ATTRIBUTION, View.OVERALL_ATTRIBUTION):
        written_paths.append(
            _write_html(
                output_directory,
                f"sector_{view.name.lower()}.html",
                attribution_by_sector.to_html(view),
            )
        )

    for chart in (
        Chart.OVERALL_CONTRIBUTION,
        Chart.OVERALL_ATTRIBUTION,
        Chart.SUBPERIOD_ATTRIBUTION,
        Chart.HEATMAP_ACTIVE_CONTRIBUTION,
        Chart.HEATMAP_ATTRIBUTION,
        Chart.CUMULATIVE_ATTRIBUTION,
        Chart.CUMULATIVE_RETURN,
    ):
        written_paths.append(
            _write_png(
                output_directory,
                f"sector_{chart.name.lower()}.png",
                attribution_by_sector.to_chart(chart),
            )
        )

    risk_statistics = analytics.get_riskstatistics()
    written_paths.append(
        _write_html(output_directory, "risk_statistics.html", risk_statistics.to_html())
    )
    return written_paths


def _write_html(output_directory: Path, file_name: str, html: str) -> Path:
    """Write one HTML artifact."""
    path = output_directory / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding=util.ENCODING)
    return path


def _write_png(output_directory: Path, file_name: str, png: bytes) -> Path:
    """Write one PNG artifact."""
    path = output_directory / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
