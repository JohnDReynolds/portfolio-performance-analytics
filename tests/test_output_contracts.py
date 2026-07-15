"""Focused tests for machine-readable attribution and risk-statistics outputs."""

# Python Imports
import datetime as dt
import io
import json
from pathlib import Path
import tempfile
import unittest
from xml.etree import ElementTree

# Third-Party Imports
import numpy as np
import pandas as pd
import polars as pl

# Project Imports
from ppar.analytics import Analytics
from ppar.analytics.attribution import Attribution, View
import ppar.analytics.schema as cols
from ppar.analytics.frequency import Frequency
from ppar.analytics.html_table import ColumnSpec, HtmlTable, SpannerSpec, attribution_table
from ppar.analytics.riskstatistics import RiskStatistics
import ppar.errors as errs
from ppar.errors import PpaError


def _attribution() -> Attribution:
    """Return a small classified attribution result for output tests."""
    performance = pl.DataFrame(
        {
            cols.FROM_DATE: [dt.date(2024, 1, 1)] * 2 + [dt.date(2024, 2, 1)] * 2,
            cols.THRU_DATE: [dt.date(2024, 1, 31)] * 2 + [dt.date(2024, 2, 29)] * 2,
            cols.IDENTIFIER: ["A", "B", "A", "B"],
            cols.RETURN: [0.10, -0.05, 0.02, 0.03],
            cols.WEIGHT: [0.60, 0.40, 0.40, 0.60],
        }
    )
    return Analytics(
        performance,
        portfolio_classification_name="Security",
    ).get_attribution(classification_data_source={"A": "Alpha", "B": "Beta"})


def _risk_statistics() -> RiskStatistics:
    """Return monthly statistics with stable values and a custom currency label."""
    portfolio_returns = np.array(
        [0.01, -0.02, 0.03, 0.01, -0.01, 0.02] * 2,
        dtype=np.float64,
    )
    benchmark_returns = np.array(
        [0.005, -0.01, 0.02, 0.015, -0.005, 0.01] * 2,
        dtype=np.float64,
    )
    return RiskStatistics(
        (portfolio_returns, benchmark_returns),
        Frequency.MONTHLY,
        portfolio_value=(250_000.0, "$"),
    )


class TestHtmlTableOutputs(unittest.TestCase):
    """Verify the internal HTML table renderer's formatting contract."""

    def test_renderer_escapes_groups_and_formats_values(self) -> None:
        """Rendered HTML escapes text and formats grouped numeric output."""
        table = HtmlTable(
            pl.DataFrame(
                {
                    "Category": ["Group <A>", "Group <A>", "Group & B"],
                    "Name": ["Alpha <One>", "Beta & Two", "Gamma"],
                    "Date": [
                        dt.date(2024, 1, 31),
                        dt.date(2024, 2, 29),
                        dt.date(2024, 3, 31),
                    ],
                    "Return": [0.012345, -0.02, float("nan")],
                    "VaR": [1234.56, 0.0, None],
                }
            ),
            columns=(
                ColumnSpec("Name", "Name", align="left"),
                ColumnSpec("Date", "Thru", format="date", align="center"),
                ColumnSpec("Return", "Return", format="number"),
                ColumnSpec("VaR", "Value At Risk", format="currency"),
            ),
            title="Portfolio <A>",
            subtitle="Risk & Return",
            spanners=(SpannerSpec("Metrics", ("Return", "VaR")),),
            group_column="Category",
            stub_column="Name",
        )

        html = table.as_raw_html()

        self.assertTrue(html.startswith("<!DOCTYPE html>"))
        self.assertIn("Portfolio &lt;A&gt;", html)
        self.assertIn("Risk &amp; Return", html)
        self.assertIn("Group &lt;A&gt;", html)
        self.assertIn("Beta &amp; Two", html)
        self.assertIn('class="ppar_spanner" colspan="2">Metrics</th>', html)
        self.assertIn("2024-02-29", html)
        self.assertIn("0.0123", html)
        self.assertIn("&minus;0.0200", html)
        self.assertIn("$1,235", html)
        self.assertEqual(html.count("<NA>"), 2)
        self.assertIn('<th scope="row" class="ppar_row ppar_left', html)

    def test_renderer_can_emit_table_fragment(self) -> None:
        """Rendered HTML may omit page scaffolding for embedding."""
        table = HtmlTable(
            pl.DataFrame({"Name": ["Alpha"], "Value": [1.25]}),
            columns=(
                ColumnSpec("Name", align="left"),
                ColumnSpec("Value", format="number"),
            ),
        )

        html = table.as_raw_html(make_page=False)

        self.assertFalse(html.startswith("<!DOCTYPE html>"))
        self.assertIn('<table class="ppar_table">', html)
        self.assertIn("1.2500", html)

    def test_attribution_table_rejects_unknown_view_name(self) -> None:
        """Unknown attribution table layouts are rejected explicitly."""
        with self.assertRaisesRegex(ValueError, "Unknown attribution view"):
            attribution_table(pl.DataFrame(), "Unexpected View", ("Title", "Subtitle"))


class TestAttributionOutputs(unittest.TestCase):
    """Verify attribution outputs retain the calculated tabular contract."""

    def test_json_contains_all_overall_rows_and_columns(self) -> None:
        """JSON output preserves the overall attribution view dimensions."""
        attribution = _attribution()
        expected = attribution.to_polars(View.OVERALL_ATTRIBUTION)

        exported = json.loads(attribution.to_json(View.OVERALL_ATTRIBUTION))

        self.assertEqual(set(exported), set(expected.columns))
        self.assertEqual(len(exported[cols.CLASSIFICATION_NAME]), expected.height)
        self.assertEqual(
            exported[cols.CLASSIFICATION_NAME][str(expected.height - 1)],
            "Total",
        )

    def test_xml_preserves_subperiod_dates_and_row_count(self) -> None:
        """XML output includes one element per subperiod summary row."""
        attribution = _attribution()
        expected = attribution.to_polars(View.SUBPERIOD_SUMMARY)

        root = ElementTree.fromstring(attribution.to_xml(View.SUBPERIOD_SUMMARY))
        rows = root.findall("row")

        self.assertEqual(len(rows), expected.height)
        self.assertTrue(
            rows[0]
            .findtext(cols.FROM_DATE, "")
            .startswith(expected[cols.FROM_DATE].item(0).isoformat())
        )
        self.assertTrue(
            rows[-1]
            .findtext(cols.THRU_DATE, "")
            .startswith(expected[cols.THRU_DATE].item(-1).isoformat())
        )

    def test_csv_respects_sorting_for_detail_view(self) -> None:
        """CSV output applies requested ordering to detail records."""
        attribution = _attribution()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "attribution.csv"
            attribution.write_csv(
                View.SUBPERIOD_ATTRIBUTION,
                output_path,
                columns_to_sort=[cols.FROM_DATE, cols.CLASSIFICATION_NAME],
                sort_descendings=[False, True],
            )
            output = pl.read_csv(output_path, try_parse_dates=True)

        names = output.filter(pl.col(cols.FROM_DATE) == dt.date(2024, 1, 1))[
            cols.CLASSIFICATION_NAME
        ].to_list()
        self.assertEqual(names, ["Beta", "Alpha"])

    def test_to_table_returns_html_table(self) -> None:
        """Attribution exposes the shared internal HTML table object."""
        table = _attribution().to_table(View.OVERALL_ATTRIBUTION)
        html = table.as_raw_html(make_page=False)

        self.assertIsInstance(table, HtmlTable)
        self.assertFalse(html.startswith("<!DOCTYPE html>"))
        self.assertIn('<table class="ppar_table">', html)
        self.assertIn("Overall Attribution", html)
        self.assertIn("Portfolio", html)

    def test_csv_accepts_path_object(self) -> None:
        """Attribution CSV output accepts pathlib output paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "attribution.csv"
            _attribution().write_csv(View.OVERALL_ATTRIBUTION, output_path)

            self.assertTrue(output_path.is_file())

    def test_invalid_view_and_sort_options_raise_ppar_errors(self) -> None:
        """Output validation fails before leaking enum, key, or Polars errors."""
        attribution = _attribution()
        with self.assertRaisesRegex(PpaError, errs.ERRORS[205]):
            attribution.to_polars("bad view")  # type: ignore[arg-type]
        with self.assertRaisesRegex(PpaError, errs.ERRORS[806]):
            attribution.to_polars(
                View.OVERALL_ATTRIBUTION,
                columns_to_sort="missing_column",
            )
        with self.assertRaisesRegex(PpaError, errs.ERRORS[806]):
            attribution.to_polars(
                View.OVERALL_ATTRIBUTION,
                columns_to_sort=[cols.CLASSIFICATION_NAME, cols.PORTFOLIO_RETURN],
                sort_descendings=[True],
            )
        with self.assertRaisesRegex(PpaError, errs.ERRORS[806]):
            attribution.to_polars(
                View.OVERALL_ATTRIBUTION,
                columns_to_sort=1,  # type: ignore[arg-type]
            )

    def test_invalid_float_precision_raises_before_serialization(self) -> None:
        """JSON and CSV precision use one bounded public contract."""
        attribution = _attribution()
        for precision in (-1, 16, True):
            with self.subTest(precision=precision):
                with self.assertRaisesRegex(PpaError, errs.ERRORS[806]):
                    attribution.to_json(
                        View.SUBPERIOD_SUMMARY,
                        float_precision=precision,
                    )


class TestRiskStatisticsOutputs(unittest.TestCase):
    """Verify risk-statistics structured outputs retain labels and values."""

    def test_json_preserves_value_at_risk_label_and_value(self) -> None:
        """JSON includes the currency-formatted value-at-risk statistic."""
        risk_statistics = _risk_statistics()
        expected = risk_statistics.to_polars()
        value_at_risk_label = "Monthly Value At Risk for $250,000"

        exported = pd.read_json(io.StringIO(risk_statistics.to_json()))
        exported_row = exported.loc[exported["column"] == value_at_risk_label]
        expected_value = expected.filter(pl.col("column") == value_at_risk_label)[
            "Portfolio"
        ].item()

        self.assertEqual(len(exported_row), 1)
        self.assertAlmostEqual(float(exported_row["Portfolio"].iloc[0]), expected_value, places=7)

    def test_xml_preserves_statistic_categories(self) -> None:
        """XML includes category values from the in-memory statistics table."""
        risk_statistics = _risk_statistics()

        root = ElementTree.fromstring(risk_statistics.to_xml())
        categories = {row.findtext("Category") for row in root.findall("row")}

        self.assertEqual(
            categories,
            {
                "Absolute Risk",
                "Downside Risk",
                "Benchmark-Relative Risk",
                "Risk-Adjusted Performance",
                "Regression",
            },
        )

    def test_csv_uses_requested_float_precision(self) -> None:
        """CSV output rounds statistic numeric values to the requested precision."""
        risk_statistics = _risk_statistics()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "riskstatistics.csv"
            risk_statistics.write_csv(output_path, float_precision=3)
            text = output_path.read_text(encoding="utf-8")

        mean_row = next(line for line in text.splitlines() if "Monthly Mean Return" in line)
        self.assertIn(",0.007,", mean_row)

    def test_to_table_returns_html_table(self) -> None:
        """Risk statistics expose the shared internal HTML table object."""
        table = _risk_statistics().to_table()
        full_html = table.as_raw_html()
        fragment_html = table.as_raw_html(make_page=False)

        self.assertIsInstance(table, HtmlTable)
        self.assertTrue(full_html.startswith("<!DOCTYPE html>"))
        self.assertFalse(fragment_html.startswith("<!DOCTYPE html>"))
        self.assertIn("Ex-Post Risk Statistics", full_html)
        self.assertIn("Absolute Risk", full_html)

    def test_csv_accepts_path_object(self) -> None:
        """Risk-statistics CSV output accepts pathlib output paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "riskstatistics.csv"
            _risk_statistics().write_csv(output_path)

            self.assertTrue(output_path.is_file())

    def test_invalid_float_precision_raises_ppar_error(self) -> None:
        """Risk serialization shares the bounded precision contract."""
        with self.assertRaisesRegex(PpaError, errs.ERRORS[806]):
            _risk_statistics().to_json(float_precision=16)


if __name__ == "__main__":
    unittest.main()
