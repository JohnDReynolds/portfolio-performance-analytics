"""Regenerate checked-in HTML expected-result files.

Run this after intentional changes to the lightweight HTML renderer.
"""

# Imports below the repository path bootstrap are intentional for direct execution.
# pylint: disable=wrong-import-order,wrong-import-position

# Python Imports
import datetime as dt
import io
from pathlib import Path
import sys

# Third-Party Imports
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Project Imports
from ppar import Analytics, Frequency  # noqa: E402
from ppar.attribution import View  # noqa: E402
import ppar.columns as cols  # noqa: E402
import ppar.utilities as util  # noqa: E402
from tests import test_utilities as test_util  # noqa: E402

EXPECTED_RESULTS_DIR = REPO_ROOT / "tests" / "expected_results"


def main() -> None:
    """Regenerate attribution and risk-statistics HTML expected results."""
    _write_attribution_html()
    _write_riskstatistics_html()


def _write_attribution_html() -> None:
    """Regenerate all attribution HTML expected-result files."""
    portfolio_df = pl.scan_csv(
        source=REPO_ROOT / "tests/data/performance/Mega-Cap Portfolio.csv",
        try_parse_dates=True,
    ).collect()
    benchmark_df = (
        pl.scan_csv(
            source=REPO_ROOT / "tests/data/performance/Large-Cap Portfolio.csv",
            try_parse_dates=True,
        )
        .collect()
        .to_pandas()
    )
    analytics = Analytics(
        portfolio_df,
        benchmark_df,
        portfolio_name="Mega-Cap Portfolio",
        benchmark_name="Large-Cap Portfolio",
        portfolio_classification_name="Security",
        benchmark_classification_name="Security",
        beginning_date="2024-01-31",
    )

    for classification_name in ("Security", "Economic Sector"):
        attribution = test_util.get_attribution(analytics, classification_name)
        for view in View:
            columns_to_sort: str | list[str] = util.EMPTY
            sort_descendings: bool | list[bool] = False
            if view == View.SUBPERIOD_ATTRIBUTION:
                columns_to_sort = [
                    cols.BEGINNING_DATE,
                    cols.PORTFOLIO_WEIGHT,
                    cols.CLASSIFICATION_IDENTIFIER,
                ]
                sort_descendings = [True, False, False]

            file_path = EXPECTED_RESULTS_DIR / f"{view.value}_{classification_name}.html"
            _write_text(
                file_path,
                attribution.to_html(view, columns_to_sort, sort_descendings),
            )


def _write_riskstatistics_html() -> None:
    """Regenerate the risk-statistics HTML expected-result file."""
    analytics = Analytics(
        REPO_ROOT / "tests/data/performance/Mega-Cap Portfolio.csv",
        REPO_ROOT / "tests/data/performance/Large-Cap Portfolio.csv",
        portfolio_classification_name="Security",
        benchmark_classification_name="Security",
        beginning_date=dt.date(2021, 12, 31),
        ending_date=dt.date(2023, 3, 31),
        frequency=Frequency.QUARTERLY,
        annual_minimum_acceptable_return=-0.16,
    )
    _write_text(
        EXPECTED_RESULTS_DIR / "riskstatistics.html",
        analytics.get_riskstatistics().to_html(),
    )


def _write_text(file_path: Path, text: str) -> None:
    """Write UTF-8 text with stable newline behavior."""
    with io.open(file_path, "w", encoding=util.ENCODING, newline="\n") as file:
        file.write(text)
    print(file_path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
