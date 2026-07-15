"""Command-line reporting for analytics configured from a local site folder."""

from __future__ import annotations

# Python imports
import argparse
from dataclasses import dataclass
import datetime as dt
from pathlib import Path
import sys
from typing import Any, Final

# Third-party imports
import polars as pl
import yaml

# Project imports
from ppar.analytics import Analytics
from ppar.analytics.attribution import Chart, View
from ppar.analytics.frequency import Frequency
import ppar.analytics.schema as cols
from ppar.axys import AxysData
from ppar.errors import PpaError
import ppar.utilities as util

_CONFIG_FILE_NAME: Final[str] = "ppar.yaml"
_ANALYTICS_SECTION: Final[str] = "analytics"
_DEFAULT_OUTPUT_DIRECTORY: Final[str] = "output"


@dataclass(frozen=True)
class AnalyticsRunSettings:
    """Resolved scalar settings for one standard Performance Analytics run."""

    portfolio_code: str
    benchmark_code: str
    frequency: Frequency
    output_directory: Path
    from_date: dt.date | None
    thru_date: dt.date | None
    classification_name: str | None
    annual_minimum_acceptable_return: float
    annual_risk_free_rate: float
    confidence_level: float
    portfolio_value: float
    currency_symbol: str


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
            from_date=args.from_date,
            thru_date=args.thru_date,
            classification_name=args.classification,
            annual_minimum_acceptable_return=args.minimum_acceptable_return,
            annual_risk_free_rate=args.risk_free_rate,
            confidence_level=args.confidence_level,
            portfolio_value=args.portfolio_value,
            currency_symbol=args.currency_symbol,
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
    from_date: str | None = None,
    thru_date: str | None = None,
    classification_name: str | None = None,
    annual_minimum_acceptable_return: float | None = None,
    annual_risk_free_rate: float | None = None,
    confidence_level: float | None = None,
    portfolio_value: float | None = None,
    currency_symbol: str | None = None,
) -> Path:
    """Write analytics artifacts for a configured Axys/APX analytics folder.

    Args:
        site_directory: Folder containing ``ppar.yaml`` and analytics CSV files.
            Accepts a ``Path`` or string path.
        portfolio_code: Optional portfolio code override.
        benchmark_code: Optional benchmark portfolio code override.
        frequency_value: Optional reporting frequency override.
        output_directory: Optional output directory override.
        from_date: Optional inclusive starting date override.
        thru_date: Optional inclusive ending date override.
        classification_name: Optional attribution classification override.
        annual_minimum_acceptable_return: Optional downside-risk assumption.
        annual_risk_free_rate: Optional risk-adjusted-return assumption.
        confidence_level: Optional value-at-risk confidence level.
        portfolio_value: Optional value used for value-at-risk presentation.
        currency_symbol: Optional value-at-risk currency symbol.

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

    settings = _resolve_settings(
        site_path,
        analytics_settings,
        portfolio_code=portfolio_code,
        benchmark_code=benchmark_code,
        frequency_value=frequency_value,
        output_directory=output_directory,
        from_date=from_date,
        thru_date=thru_date,
        classification_name=classification_name,
        annual_minimum_acceptable_return=annual_minimum_acceptable_return,
        annual_risk_free_rate=annual_risk_free_rate,
        confidence_level=confidence_level,
        portfolio_value=portfolio_value,
        currency_symbol=currency_symbol,
    )
    axys_data = AxysData(config_path)
    portfolios = axys_data.get_portfolios(
        (settings.portfolio_code, settings.benchmark_code),
        from_date=settings.from_date,
        thru_date=settings.thru_date,
        classification_name=settings.classification_name,
    )
    portfolio = portfolios[settings.portfolio_code]
    benchmark = portfolios[settings.benchmark_code]
    analytics = portfolio.to_analytics(
        benchmark,
        frequency=settings.frequency,
        annual_minimum_acceptable_return=settings.annual_minimum_acceptable_return,
        annual_risk_free_rate=settings.annual_risk_free_rate,
        confidence_level=settings.confidence_level,
        portfolio_value=(settings.portfolio_value, settings.currency_symbol),
    )
    security_classification = pl.concat(
        [
            axys_data.get_classification_sources(
                "Security", portfolio
            ).classification_data_source,
            axys_data.get_classification_sources(
                "Security", benchmark
            ).classification_data_source,
        ],
        how="vertical",
    ).unique(subset=[cols.IDENTIFIER], keep="any")
    written_paths = _write_analytics_outputs(
        analytics,
        settings.output_directory,
        security_classification,
    )

    if written_paths:
        print("Open these files to review analytics output:")
        for path in written_paths:
            print(f"  {path}")
    return settings.output_directory


def _argument_parser(
    *,
    prog: str = "ppar analytics",
    include_site_directory: bool = True,
) -> argparse.ArgumentParser:
    """Return the analytics argument parser."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Write Axys/APX analytics reports from a configured site folder.",
        epilog=(
            (
                "Examples:\n"
                "  ppar analytics ./my_ppar_data/analytics\n"
                "  ppar analytics"
            )
            if include_site_directory
            else (
                "Examples:\n"
                "  python run_analytics.py\n"
                "  python run_analytics.py --frequency monthly"
            )
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    if include_site_directory:
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
        help="Portfolio code. Defaults to YAML analytics.portfolio in ppar.yaml.",
    )
    parser.add_argument(
        "--benchmark",
        help="Benchmark code. Defaults to YAML analytics.benchmark in ppar.yaml.",
    )
    parser.add_argument(
        "-f",
        "--frequency",
        help=(
            "Reporting frequency: monthly, quarterly, yearly, or m/q/y. "
            "Defaults to YAML analytics.frequency in ppar.yaml, then the "
            "Python default quarterly."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output directory. Defaults to YAML analytics.output_directory in "
            "ppar.yaml, then the Python default output."
        ),
    )
    parser.add_argument(
        "--from-date",
        help=(
            "Inclusive YYYY-MM-DD start. Defaults to YAML analytics.from_date, "
            "then YAML defaults.from_date in ppar.yaml."
        ),
    )
    parser.add_argument(
        "--thru-date",
        help=(
            "Inclusive YYYY-MM-DD end. Defaults to YAML analytics.thru_date, "
            "then YAML defaults.thru_date in ppar.yaml."
        ),
    )
    parser.add_argument(
        "--classification",
        help=(
            "Attribution classification. Defaults to YAML "
            "analytics.classification, then YAML defaults.classification in "
            "ppar.yaml."
        ),
    )
    parser.add_argument(
        "--minimum-acceptable-return",
        type=float,
        help=(
            "Annual downside-risk target. Defaults to "
            "YAML analytics.annual_minimum_acceptable_return in ppar.yaml, "
            "then the Python default 0.0."
        ),
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        help=(
            "Annual risk-free rate. Defaults to YAML "
            "analytics.annual_risk_free_rate in ppar.yaml, then the Python "
            "default 0.03."
        ),
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        help=(
            "Value-at-risk confidence. Defaults to YAML "
            "analytics.confidence_level in ppar.yaml, then the Python default "
            "0.95."
        ),
    )
    parser.add_argument(
        "--portfolio-value",
        type=float,
        help=(
            "Value-at-risk amount. Defaults to YAML analytics.portfolio_value "
            "in ppar.yaml, then the Python default 100000."
        ),
    )
    parser.add_argument(
        "--currency-symbol",
        help=(
            "Value-at-risk currency. Defaults to YAML analytics.currency_symbol "
            "in ppar.yaml, then the Python default '$'."
        ),
    )
    return parser


def _default_site_directory(site_directory: Path | None) -> Path:
    """Return the explicit or conventional analytics site directory."""
    if site_directory is not None:
        return site_directory
    return Path.cwd()


def script_run_settings(
    site_directory: Path,
    argv: list[str] | None = None,
) -> AnalyticsRunSettings:
    """Resolve script arguments using the same rules as ``ppar analytics``.

    Args:
        site_directory: Analytics folder containing ``ppar.yaml``.
        argv: Optional CLI-style overrides excluding the script name.

    Returns:
        Fully resolved and validated settings for the visible Python workflow.
    """
    arguments = _argument_parser(
        prog="python run_analytics.py",
        include_site_directory=False,
    ).parse_args(argv)
    config_path = (site_directory / _CONFIG_FILE_NAME).resolve()
    values = _analytics_settings(_load_config_values(config_path))
    return _resolve_settings(
        site_directory,
        values,
        portfolio_code=arguments.portfolio,
        benchmark_code=arguments.benchmark,
        frequency_value=arguments.frequency,
        output_directory=arguments.output,
        from_date=arguments.from_date,
        thru_date=arguments.thru_date,
        classification_name=arguments.classification,
        annual_minimum_acceptable_return=arguments.minimum_acceptable_return,
        annual_risk_free_rate=arguments.risk_free_rate,
        confidence_level=arguments.confidence_level,
        portfolio_value=arguments.portfolio_value,
        currency_symbol=arguments.currency_symbol,
    )


def write_html_file(output_directory: Path, file_name: str, html: str) -> Path:
    """Write one standard HTML artifact for a visible Python workflow."""
    return _write_html(output_directory, file_name, html)


def write_png_file(output_directory: Path, file_name: str, png: bytes) -> Path:
    """Write one standard PNG artifact for a visible Python workflow."""
    return _write_png(output_directory, file_name, png)


def _load_config_values(config_path: Path) -> dict[str, Any]:
    """Load an analytics YAML file and return its root mapping."""
    if not config_path.exists():
        raise PpaError(
            f"{config_path} does not exist. Run from the analytics folder "
            "or pass the folder. For first-time setup, run: "
            "ppar setup ./my_ppar_data",
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


def _resolve_settings(
    site_path: Path,
    values: dict[str, Any],
    *,
    portfolio_code: str | None,
    benchmark_code: str | None,
    frequency_value: str | None,
    output_directory: Path | None,
    from_date: str | None,
    thru_date: str | None,
    classification_name: str | None,
    annual_minimum_acceptable_return: float | None,
    annual_risk_free_rate: float | None,
    confidence_level: float | None,
    portfolio_value: float | None,
    currency_symbol: str | None,
) -> AnalyticsRunSettings:
    """Resolve CLI overrides, YAML values, and library defaults."""
    confidence = _float_setting(
        confidence_level,
        values,
        "confidence_level",
        util.DEFAULT_CONFIDENCE_LEVEL,
    )
    value = _float_setting(
        portfolio_value,
        values,
        "portfolio_value",
        util.DEFAULT_PORTFOLIO_VALUE,
    )
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be greater than 0 and less than 1")
    if value <= 0.0:
        raise ValueError("portfolio_value must be greater than 0")
    selected_currency = currency_symbol or str(
        values.get("currency_symbol", util.DEFAULT_CURRENCY_SYMBOL)
    )
    if not selected_currency:
        raise ValueError("currency_symbol must not be empty")
    return AnalyticsRunSettings(
        portfolio_code=portfolio_code
        or _required_setting(values, "portfolio", _ANALYTICS_SECTION),
        benchmark_code=benchmark_code
        or _required_setting(values, "benchmark", _ANALYTICS_SECTION),
        frequency=_frequency_from_string(
            frequency_value or str(values.get("frequency", "quarterly"))
        ),
        output_directory=output_directory
        or site_path / str(values.get("output_directory", _DEFAULT_OUTPUT_DIRECTORY)),
        from_date=_optional_date(from_date or values.get("from_date"), "from_date"),
        thru_date=_optional_date(thru_date or values.get("thru_date"), "thru_date"),
        classification_name=classification_name
        or _optional_string_setting(values, "classification"),
        annual_minimum_acceptable_return=_float_setting(
            annual_minimum_acceptable_return,
            values,
            "annual_minimum_acceptable_return",
            util.DEFAULT_ANNUAL_MINIMUM_ACCEPTABLE_RETURN,
        ),
        annual_risk_free_rate=_float_setting(
            annual_risk_free_rate,
            values,
            "annual_risk_free_rate",
            util.DEFAULT_ANNUAL_RISK_FREE_RATE,
        ),
        confidence_level=confidence,
        portfolio_value=value,
        currency_symbol=selected_currency,
    )


def _float_setting(
    override: float | None,
    values: dict[str, Any],
    name: str,
    default: float,
) -> float:
    """Return one float setting using CLI, YAML, then library precedence."""
    value = override if override is not None else values.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"analytics.{name} must be numeric")
    return float(value)


def _optional_string_setting(values: dict[str, Any], name: str) -> str | None:
    """Return one optional non-empty string setting."""
    value = values.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"analytics.{name} must be a non-empty string")
    return value


def _optional_date(value: Any, name: str) -> dt.date | None:
    """Return one optional date setting converted to a native date."""
    if value is None:
        return None
    try:
        return util.convert_to_date(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"analytics.{name} must be a valid date") from error


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
    security_classification: util.ClassificationDataSource,
) -> list[Path]:
    """Write the analytics HTML, PNG, and risk outputs used by the CLI.

    Args:
        analytics: Configured analytics calculation.
        output_directory: Directory that receives report artifacts.
        security_classification: Security identifier/display-name lookup.

    Returns:
        Paths written by the standard analytics workflow.
    """
    written_paths: list[Path] = []

    attribution_by_security = analytics.get_attribution(
        "Security",
        security_classification,
    )
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
