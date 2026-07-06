"""Run the optional generic analytics sample from Python."""

from __future__ import annotations

# Python imports
from pathlib import Path

# Project imports
from ppar.analytics import Analytics
from ppar.analytics.attribution import Chart, View
from ppar.analytics.frequency import Frequency
import ppar.utilities as util


SITE_DIRECTORY = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = SITE_DIRECTORY / "output"
CLASSIFICATION_NAME = "Economic Sector"


def main() -> None:
    """Create analytics output from the optional generic setup data."""
    analytics = Analytics(
        SITE_DIRECTORY / "performance" / "Mega-Cap Alpha Portfolio.csv",
        SITE_DIRECTORY / "performance" / "Mega-Cap Benchmark.csv",
        portfolio_classification_name="Security",
        benchmark_classification_name="Security",
        frequency=Frequency.QUARTERLY,
    )
    classification_data_source = (
        SITE_DIRECTORY / "classifications" / f"{CLASSIFICATION_NAME}.csv"
    )
    mapping_data_sources = (
        SITE_DIRECTORY / "mappings" / "Security--to--Economic Sector.csv",
        SITE_DIRECTORY / "mappings" / "Security--to--Economic Sector.csv",
    )

    written_paths: list[Path] = []

    attribution_by_security = analytics.get_attribution("Security")
    written_paths.append(
        _write_html(
            "security_overall_attribution.html",
            attribution_by_security.to_html(View.OVERALL_ATTRIBUTION),
        )
    )

    attribution_by_sector = analytics.get_attribution(
        CLASSIFICATION_NAME,
        classification_data_source,
        mapping_data_sources,
    )
    for view in (View.CUMULATIVE_ATTRIBUTION, View.OVERALL_ATTRIBUTION):
        written_paths.append(
            _write_html(
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
                f"sector_{chart.name.lower()}.png",
                attribution_by_sector.to_chart(chart),
            )
        )

    risk_statistics = analytics.get_riskstatistics()
    written_paths.append(_write_html("risk_statistics.html", risk_statistics.to_html()))

    print("Open these files to review analytics output:")
    for path in written_paths:
        print(f"  {path.resolve()}")


def _write_html(file_name: str, html: str) -> Path:
    """Write one HTML review file."""
    path = OUTPUT_DIRECTORY / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding=util.ENCODING)
    return path


def _write_png(file_name: str, png: bytes) -> Path:
    """Write one PNG chart file."""
    path = OUTPUT_DIRECTORY / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    return path


if __name__ == "__main__":
    main()
