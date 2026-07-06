"""Run the local PPAR analytics setup from Python.

This script is installed by ``ppar setup`` next to ``ppar.yaml`` and the
analytics CSV files. It keeps the workflow visible: load data, calculate
analytics, write review files, and print the files to open.
"""

from __future__ import annotations

# Python imports
from pathlib import Path

# Project imports
from ppar.analytics.attribution import Chart, View
from ppar.analytics.frequency import Frequency
from ppar.axys import AxysData
import ppar.utilities as util


SITE_DIRECTORY = Path(__file__).resolve().parent
CONFIG_PATH = SITE_DIRECTORY / "ppar.yaml"
OUTPUT_DIRECTORY = SITE_DIRECTORY / "output"

# These defaults match the starter ``ppar.yaml``. You can change the values here
# for an automation-specific override, but most sites should change the YAML.
PORTFOLIO_CODE = "MEGA_ALPHA"
BENCHMARK_CODE = "MEGA_BENCH"
FREQUENCY = Frequency.QUARTERLY


def main() -> None:
    """Create analytics output from the local setup-site CSV files."""
    source_data = AxysData(CONFIG_PATH)
    portfolio = source_data.get_portfolio(PORTFOLIO_CODE)
    benchmark = source_data.get_portfolio(BENCHMARK_CODE)
    analytics = portfolio.to_analytics(benchmark, frequency=FREQUENCY)

    written_paths: list[Path] = []

    attribution_by_security = analytics.get_attribution("Security")
    written_paths.append(
        _write_html(
            "security_overall_attribution.html",
            attribution_by_security.to_html(View.OVERALL_ATTRIBUTION),
        )
    )

    attribution_by_sector = analytics.get_attribution()
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
