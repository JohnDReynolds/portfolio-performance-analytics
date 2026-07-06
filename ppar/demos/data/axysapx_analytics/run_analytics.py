"""Run the local PPAR analytics setup from Python."""

from __future__ import annotations

# Python imports
from pathlib import Path

# Project imports
from ppar.analytics.attribution import Chart, View
from ppar.analytics.frequency import Frequency
from ppar.axys import AxysData
import ppar.utilities as util


# Anchor every relative path to the folder containing this script. That lets
# you run the script from any current working directory without changing paths.
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
    # ``AxysData`` reads ``ppar.yaml`` and uses its ``files`` and ``columns``
    # sections to locate and interpret the local IMEX CSV files.
    source_data = AxysData(CONFIG_PATH)

    # These two portfolio objects are the inputs to attribution: one managed
    # portfolio and one benchmark. Change the codes above or in YAML as your
    # site setup evolves.
    portfolio = source_data.get_portfolio(PORTFOLIO_CODE)
    benchmark = source_data.get_portfolio(BENCHMARK_CODE)

    # ``to_analytics`` calculates the return, contribution, attribution, and
    # risk data used by the review files below.
    analytics = portfolio.to_analytics(benchmark, frequency=FREQUENCY)

    written_paths: list[Path] = []

    # Security attribution is useful for checking which individual positions
    # drove the active return over the selected reporting window.
    attribution_by_security = analytics.get_attribution("Security")
    written_paths.append(
        _write_html(
            "security_overall_attribution.html",
            attribution_by_security.to_html(View.OVERALL_ATTRIBUTION),
        )
    )

    # The default attribution level in the starter setup is sector. These HTML
    # tables are easy to inspect first because they are compact and sortable.
    attribution_by_sector = analytics.get_attribution()
    for view in (View.CUMULATIVE_ATTRIBUTION, View.OVERALL_ATTRIBUTION):
        written_paths.append(
            _write_html(
                f"sector_{view.name.lower()}.html",
                attribution_by_sector.to_html(view),
            )
        )

    # PNG charts are written separately so they can be used in decks, emails,
    # README files, or other materials outside the HTML tables.
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

    # Risk statistics provide a separate cross-check on the same return stream:
    # tracking error, information ratio, drawdown, and related review metrics.
    risk_statistics = analytics.get_riskstatistics()
    written_paths.append(_write_html("risk_statistics.html", risk_statistics.to_html()))

    print("Open these files to review analytics output:")
    for path in written_paths:
        print(f"  {path.resolve()}")


def _write_html(file_name: str, html: str) -> Path:
    """Write one HTML review file."""
    path = OUTPUT_DIRECTORY / file_name
    # ``exist_ok=True`` makes repeat runs overwrite output files without needing
    # manual cleanup between review cycles.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding=util.ENCODING)
    return path


def _write_png(file_name: str, png: bytes) -> Path:
    """Write one PNG chart file."""
    path = OUTPUT_DIRECTORY / file_name
    # Chart renderers return PNG bytes, so ``write_bytes`` avoids any text
    # encoding assumptions.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    return path


if __name__ == "__main__":
    main()
