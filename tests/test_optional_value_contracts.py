"""Contracts for optional values normalized to ``None``."""

# Python Imports
import datetime as dt
import unittest

# Third-Party Imports
import polars as pl

# Project Imports
from ppar.analytics import Analytics
from ppar.analytics.attribution import View
import ppar.analytics.schema as cols
import ppar.demos.generic_analytics_data_sources as demo_data
from ppar.analytics.performance import Performance
import ppar.utilities as util


def _performance_rows() -> pl.DataFrame:
    """Return minimal narrow performance rows for sentinel compatibility tests."""
    return pl.DataFrame(
        {
            cols.FROM_DATE: [dt.date(2024, 1, 1)] * 2,
            cols.THRU_DATE: [dt.date(2024, 2, 1)] * 2,
            cols.IDENTIFIER: ["A", "B"],
            cols.RETURN: [0.10, -0.05],
            cols.WEIGHT: [0.60, 0.40],
        }
    )


class TestOptionalValueContracts(unittest.TestCase):
    """Verify optional values use ``None`` while blank input remains accepted."""

    def test_optional_strings_normalize_to_none(self) -> None:
        """Omitted and blank optional strings normalize to ``None``."""
        for value in (None, "", "   "):
            with self.subTest(value=value):
                self.assertIsNone(util.normalize_optional_string(value))

        self.assertEqual(util.normalize_optional_string("Security"), "Security")
        self.assertEqual(util.normalize_optional_string("_empty_"), "_empty_")

    def test_omitted_performance_metadata_uses_none(self) -> None:
        """Absent public performance metadata is stored as ``None``."""
        performance = Performance(_performance_rows())

        self.assertIsNone(performance.name)
        self.assertIsNone(performance.classification_name)

    def test_blank_attribution_arguments_match_omitted_arguments(self) -> None:
        """Blank string arguments select the same output as omissions."""
        implicit = Analytics(_performance_rows()).get_attribution()
        explicit = Analytics(_performance_rows()).get_attribution(
            classification_name="",
            classification_data_source="",
            mapping_data_sources=("", ""),
            classification_label="",
        )

        for view in View:
            with self.subTest(view=view):
                self.assertTrue(implicit.to_polars(view).equals(explicit.to_polars(view)))

        implicit_html = implicit.to_table(View.OVERALL_ATTRIBUTION).as_raw_html(make_page=False)
        explicit_html = explicit.to_table(View.OVERALL_ATTRIBUTION).as_raw_html(make_page=False)
        self.assertEqual(implicit_html, explicit_html)
        self.assertNotIn("_empty_", implicit_html)

    def test_demo_omitted_sources_return_none(self) -> None:
        """Demo source helpers expose ``None`` when no source is requested."""
        analytics = Analytics(_performance_rows())

        self.assertIsNone(demo_data.classification_data_source())
        self.assertEqual(
            demo_data.mapping_data_sources(analytics),
            (None, None),
        )

    def test_blank_sort_string_matches_omitted_sorting(self) -> None:
        """A blank sorting argument preserves default output ordering."""
        attribution = Analytics(_performance_rows()).get_attribution()

        default_output = attribution.to_polars(View.OVERALL_ATTRIBUTION)
        legacy_output = attribution.to_polars(
            View.OVERALL_ATTRIBUTION,
            columns_to_sort="",
        )

        self.assertTrue(default_output.equals(legacy_output))


if __name__ == "__main__":
    unittest.main()
