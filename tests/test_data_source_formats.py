"""Integration tests for supported user-facing data-source representations."""

# Direct cache access verifies source-format integration state.
# pylint: disable=protected-access
# pyright: reportPrivateUsage=false

# Python Imports
import datetime as dt
from collections.abc import Sequence
from pathlib import Path
import tempfile
import unittest

# Third-Party Imports
import pandas as pd
import polars as pl

# Test Imports
from tests import test_utilities as test_util

# Project Imports
from ppar.analytics import Analytics
from ppar.analytics.attribution import View
import ppar.analytics.schema as cols
import ppar.utilities as util

_SECURITY_NAMES = {"AAPL": "Apple Inc.", "MSFT": "Microsoft"}
_SECTOR_MAPPINGS = {"AAPL": "IT", "MSFT": "IT"}


class TestDataSourceFormats(unittest.TestCase):
    """Verify equivalent supported data sources produce equivalent output."""

    def test_classification_data_and_mapping_data(self) -> None:
        """Dictionary-based classification and mapping data sources work together."""
        analytics = Analytics(
            test_util.performance_data_path("Big 2"),
            test_util.performance_data_path("Big 2"),
            portfolio_classification_name="Security",
            benchmark_classification_name="Security",
        )
        expected_html = test_util.get_attribution(analytics, "Economic Sector").to_html(
            View.OVERALL_ATTRIBUTION
        )

        html = test_util.get_attribution(
            analytics,
            "Economic Sector",
            mapping_data_source={"AAPL": "IT", "MSFT": "IT"},
            classification_data_source={"IT": "Information Technology"},
        ).to_html(View.OVERALL_ATTRIBUTION)

        self.assertEqual(
            test_util.html_table_lines(expected_html),
            test_util.html_table_lines(html),
        )

    def test_classification_data_formats(self) -> None:
        """Classification input supports path, dictionary, pandas, and Polars forms."""
        analytics = Analytics(
            test_util.performance_data_path("Big 2"),
            test_util.performance_data_path("Big 2"),
            portfolio_classification_name="Security",
            benchmark_classification_name="Security",
        )
        with tempfile.TemporaryDirectory() as temp_directory:
            security_path = _write_two_column_csv(
                Path(temp_directory) / "Security.csv",
                _SECURITY_NAMES,
            )
            expected_html = test_util.get_attribution(
                analytics,
                "Security",
                classification_data_source=security_path,
            ).to_html(View.OVERALL_ATTRIBUTION)
            classification_sources: tuple[util.ClassificationDataSource, ...] = (
                security_path,
                _SECURITY_NAMES,
                pd.DataFrame(
                    {"c1": ["AAPL", "MSFT"], "c2": ["Apple Inc.", "Microsoft"]}
                ),
                pl.DataFrame(
                    {"c1": ["AAPL", "MSFT"], "c2": ["Apple Inc.", "Microsoft"]}
                ),
            )

            for classification_data_source in classification_sources:
                analytics._attributions = {}
                html = test_util.get_attribution(
                    analytics,
                    "Security",
                    classification_data_source=classification_data_source,
                ).to_html(View.OVERALL_ATTRIBUTION)

                self.assertEqual(
                    test_util.html_table_lines(expected_html),
                    test_util.html_table_lines(html),
                )

    def test_mapping_data_formats(self) -> None:
        """Mapping input supports path, dictionary, pandas, and Polars forms."""
        analytics = Analytics(
            test_util.performance_data_path("Big 2"),
            test_util.performance_data_path("Big 2"),
            portfolio_classification_name="Security",
            benchmark_classification_name="Security",
        )
        with tempfile.TemporaryDirectory() as temp_directory:
            mapping_path = _write_two_column_csv(
                Path(temp_directory) / "Security--to--Economic Sector.csv",
                _SECTOR_MAPPINGS,
            )
            expected_html = test_util.get_attribution(
                analytics,
                "Economic Sector",
                mapping_data_source=mapping_path,
            ).to_html(View.OVERALL_ATTRIBUTION)
            mapping_sources: tuple[util.MappingDataSource, ...] = (
                mapping_path,
                _SECTOR_MAPPINGS,
                pd.DataFrame({"c1": ["AAPL", "MSFT"], "c2": ["IT", "IT"]}),
                pl.DataFrame({"c1": ["AAPL", "MSFT"], "c2": ["IT", "IT"]}),
            )

            for mapping_data_source in mapping_sources:
                analytics._attributions = {}
                html = test_util.get_attribution(
                    analytics,
                    "Economic Sector",
                    mapping_data_source=mapping_data_source,
                ).to_html(View.OVERALL_ATTRIBUTION)

                self.assertEqual(
                    test_util.html_table_lines(expected_html),
                    test_util.html_table_lines(html),
                )

    def test_performance_data_formats(self) -> None:
        """Performance input supports DataFrame forms and embedded security names."""
        expected_analytics = Analytics(
            test_util.performance_data_path("Big 2"),
            test_util.performance_data_path("Big 2"),
            portfolio_classification_name="Security",
            benchmark_classification_name="Security",
            from_date=dt.date(2024, 1, 1),
            thru_date="2024-02-29",
        )
        expected_html = expected_analytics.get_attribution().to_html(
            View.OVERALL_ATTRIBUTION
        )
        performance_dict: dict[str, Sequence[dt.date | str | float]] = {
            cols.FROM_DATE: [dt.date(2024, 1, 1)] * 2 + [dt.date(2024, 2, 1)] * 2,
            cols.THRU_DATE: [dt.date(2024, 1, 31)] * 2 + [dt.date(2024, 2, 29)] * 2,
            cols.IDENTIFIER: ["AAPL", "MSFT", "AAPL", "MSFT"],
            cols.NAME: ["Apple Inc.", "Microsoft", "Apple Inc.", "Microsoft"],
            cols.WEIGHT: [0.5, 0.5, 0.5, 0.5],
            cols.RETURN: [-0.0422272121, 0.0572811503, -0.019793881, 0.0403944092],
        }

        for data_frame_library in (pd, pl):
            analytics = Analytics(
                data_frame_library.DataFrame(performance_dict),
                data_frame_library.DataFrame(performance_dict),
                portfolio_name="Big 2",
                benchmark_name="Big 2",
                portfolio_classification_name="Security",
                benchmark_classification_name="Security",
            )
            html = analytics.get_attribution().to_html(View.OVERALL_ATTRIBUTION)
            self.assertEqual(
                test_util.html_table_lines(html),
                test_util.html_table_lines(expected_html),
            )

        analytics = Analytics(
            test_util.performance_data_path("Big 2"),
            test_util.performance_data_path("Big 2"),
            from_date=dt.date(2024, 1, 1),
            thru_date="2024-02-29",
        )
        html = analytics.get_attribution().to_html(View.OVERALL_ATTRIBUTION)
        self.assertIn("Apple Inc.", html)
        self.assertIn("Microsoft", html)

    def test_no_classification_name_uses_default_attribution(self) -> None:
        """Attribution may render when no classification name is supplied."""
        analytics = Analytics(
            test_util.performance_data_path("abcde_portfolio1"),
            test_util.performance_data_path("abcde_portfolio1"),
        )

        html = test_util.get_attribution(analytics).to_html(View.OVERALL_ATTRIBUTION)

        self.assertIn("Overall Attribution", html)


def _write_two_column_csv(path: Path, values: dict[str, str]) -> Path:
    """Write a small two-column CSV fixture and return its path."""
    path.write_text(
        "".join(f"{identifier},{name}\n" for identifier, name in values.items()),
        encoding=util.ENCODING,
    )
    return path


if __name__ == "__main__":
    unittest.main()
