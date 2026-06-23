"""Shared output rendering for bundled analytics demos."""

from __future__ import annotations

# Python imports
import argparse
from collections.abc import Sequence
from pathlib import Path

# Project imports
from ppar.analytics import Analytics
from ppar.analytics.attribution import Attribution, Chart, View
from ppar.analytics.frequency import Frequency
import ppar.utilities as util

DEFAULT_DEMO_FREQUENCY = Frequency.QUARTERLY


def demo_frequency_from_string(value: str | None) -> Frequency:
    """Return a demo reporting frequency from a lenient user string.

    Args:
        value: Optional frequency string. Values whose first nonblank character
            is ``"m"``, ``"q"``, or ``"y"`` map to monthly, quarterly, or
            yearly reporting. Blank values default to quarterly.

    Returns:
        Reporting frequency for analytics demo output.

    Raises:
        ValueError: If ``value`` does not start with ``m``, ``q``, or ``y``.
    """
    normalized = "" if value is None else value.strip().lower()
    if not normalized:
        return DEFAULT_DEMO_FREQUENCY
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


def parse_demo_frequency_argument(
    argv: Sequence[str] | None = None,
    *,
    description: str,
) -> Frequency:
    """Parse a common analytics-demo frequency argument.

    Args:
        argv: Optional argument sequence. ``None`` reads from ``sys.argv``.
        description: Argument parser description for the calling demo.

    Returns:
        Parsed reporting frequency. Defaults to quarterly.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-f",
        "--frequency",
        default="quarterly",
        help=(
            "Reporting frequency. Any value starting with m, q, or y is accepted. "
            "Defaults to quarterly."
        ),
    )
    args = parser.parse_args(argv)
    try:
        return demo_frequency_from_string(args.frequency)
    except ValueError as error:
        parser.error(str(error))
        raise AssertionError("argparse should exit before this point") from error


def frequency_display_name(frequency: Frequency) -> str:
    """Return a concise display name for a demo reporting frequency."""
    if frequency == Frequency.MONTHLY:
        return "monthly"
    if frequency == Frequency.YEARLY:
        return "yearly"
    return "quarterly"


def write_analytics_demo_outputs(
    analytics: Analytics,
    output_directory: Path,
    *,
    sector_classification_name: str = "Economic Sector",
    sector_classification_data_source: util.ClassificationDataSource | None = None,
    sector_mapping_data_sources: Sequence[util.MappingDataSource | None] | None = None,
) -> list[Path]:
    """Write the common analytics demo tables, charts, and risk report.

    Args:
        analytics: Analytics object already initialized from the desired data
            ingestion path.
        output_directory: Directory where demo artifacts should be written.
        sector_classification_name: Reporting classification used for the
            sector-level attribution artifacts.
        sector_classification_data_source: Optional classification source for
            sector-level attribution. Omit when ``analytics`` was created with
            default attribution sources, such as through ``AxysData``.
        sector_mapping_data_sources: Optional portfolio and benchmark mappings
            from security to sector. Omit when ``analytics`` was created with
            default attribution sources.

    Returns:
        Paths written by the renderer.
    """
    written_paths: list[Path] = []

    attribution_by_security = analytics.get_attribution("Security")
    written_paths.append(
        _write_html(
            output_directory,
            "security_overall_attribution.html",
            attribution_by_security.to_html(View.OVERALL_ATTRIBUTION),
        )
    )

    attribution_by_sector = _sector_attribution(
        analytics,
        sector_classification_name,
        sector_classification_data_source,
        sector_mapping_data_sources,
    )
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

    # Exercise alternate output formats that applications commonly use for
    # their own presentation layers.
    view = View.OVERALL_ATTRIBUTION
    _ = attribution_by_sector.to_html(view)
    _ = attribution_by_sector.to_json(view)
    _ = attribution_by_sector.to_pandas(view)
    _ = attribution_by_sector.to_polars(view)
    table = attribution_by_sector.to_table(view)
    _ = table.as_raw_html(make_page=False)
    _ = attribution_by_sector.to_xml(view)

    return written_paths


def print_analytics_demo_handoff(output_directory: Path, paths: Sequence[Path]) -> None:
    """Print generated analytics demo artifact paths.

    Args:
        output_directory: Directory that received the rendered demo artifacts.
        paths: Paths written by ``write_analytics_demo_outputs``.
    """
    print(f"Analytics demo output written to: {output_directory.resolve()}")
    if paths:
        print("Open these files to review the demo output:")
        for path in paths:
            print(f"- {path.resolve()}")


def _sector_attribution(
    analytics: Analytics,
    sector_classification_name: str,
    sector_classification_data_source: util.ClassificationDataSource | None,
    sector_mapping_data_sources: Sequence[util.MappingDataSource | None] | None,
) -> Attribution:
    """Return sector attribution from explicit or Analytics-default sources."""
    if sector_classification_data_source is None and sector_mapping_data_sources is None:
        return analytics.get_attribution()
    return analytics.get_attribution(
        sector_classification_name,
        sector_classification_data_source,
        sector_mapping_data_sources,
    )


def _write_html(output_directory: Path, file_name: str, html: str) -> Path:
    """Write one demo HTML artifact."""
    path = output_directory / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding=util.ENCODING)
    return path


def _write_png(output_directory: Path, file_name: str, png: bytes) -> Path:
    """Write one demo PNG artifact."""
    path = output_directory / file_name
    path.write_bytes(png)
    return path
