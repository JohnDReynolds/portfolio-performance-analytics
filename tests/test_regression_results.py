"""Fixture-based regression tests for calculated report results and exports."""

# Legacy compatibility checks exercise internal calculated frames.
# pylint: disable=protected-access
# pyright: reportPrivateUsage=false

# Python Imports
import datetime as dt
from pathlib import Path
import tempfile
import unittest

# Third-Party Imports
import polars as pl

# Test Imports
from tests import test_utilities as test_util

# Project Imports
from ppar.analytics import Analytics
from ppar.attribution import Chart, View
import ppar.columns as cols
from ppar.frequency import Frequency
import ppar.utilities as util

_EXPECTED_RESULTS_DIRECTORIES = [
    "tests/expected_results",
    "../tests/expected_results",
    "expected_results",
]


class TestRegressionResults(unittest.TestCase):
    """Verify fixture-based calculation values and stored CSV baselines."""

    def test_abcde_different_subperiod_contributions(self) -> None:
        """Five-asset calculations remain stable when source subperiods differ."""
        analytics = Analytics(
            test_util.performance_data_path("abcde_portfolio1"),
            test_util.performance_data_path("abcde_portfolio2"),
        )
        attribution = test_util.get_attribution(analytics)
        contribution = attribution.to_polars(View.SUBPERIOD_ATTRIBUTION)

        portfolio_contributions = contribution[cols.PORTFOLIO_CONTRIB_SIMPLE]
        benchmark_contributions = contribution[cols.BENCHMARK_CONTRIB_SIMPLE]
        expected_portfolio = [
            0.03696005216365282,
            -0.05010275600837092,
            0.015611261376729373,
            0.029019065603398495,
            0.07704845163844518,
        ]

        for actual, expected in zip(portfolio_contributions[:5], expected_portfolio):
            self.assertTrue(util.are_near(actual, expected))
        self.assertTrue(util.are_near(benchmark_contributions.item(3), 0.001314124548289089))

    def test_abcde_identical_subperiod_results(self) -> None:
        """Five-asset calculations remain stable when source subperiods match."""
        analytics = Analytics(
            test_util.performance_data_path("abcde_portfolio1"),
            test_util.performance_data_path("abcde_benchmark1"),
        )
        attribution = test_util.get_attribution(analytics)
        subperiods = attribution.to_polars(View.SUBPERIOD_SUMMARY)
        detail = attribution.to_polars(View.OVERALL_ATTRIBUTION)

        self.assertTrue(
            util.are_near(subperiods[cols.PORTFOLIO_RETURN].item(0), 0.03638750268034727)
        )
        self.assertTrue(
            util.are_near(subperiods[cols.PORTFOLIO_RETURN].item(1), 0.004100599095234386)
        )
        self.assertTrue(
            util.are_near(subperiods[cols.BENCHMARK_RETURN].item(0), 0.03964350666619861)
        )
        self.assertTrue(
            util.are_near(subperiods[cols.BENCHMARK_RETURN].item(2), 0.06673607157200062)
        )
        self.assertTrue(
            util.are_near(detail[cols.ALLOCATION_EFFECT_SMOOTHED].item(1), -0.0000097757165254280)
        )
        self.assertTrue(
            util.are_near(detail[cols.SELECTION_EFFECT_SMOOTHED].item(1), -0.0016362229861442853)
        )

        constructed_detail = attribution._construct_df_for_detail_views(
            View.SUBPERIOD_ATTRIBUTION
        ).collect()
        self.assertTrue(
            util.are_near(
                constructed_detail[cols.PORTFOLIO_CONTRIB_SMOOTHED].item(4), 0.01900264215424944
            )
        )
        self.assertTrue(
            util.are_near(
                constructed_detail[cols.BENCHMARK_CONTRIB_SMOOTHED].item(4), 0.019577639459518823
            )
        )

    def test_attribution_csv_results_and_serialization_paths(self) -> None:
        """Attribution views retain their CSV baselines and serializable paths."""
        portfolio_df = pl.read_csv(
            test_util.performance_data_path("Mega-Cap Portfolio"),
            try_parse_dates=True,
        )
        benchmark_df = pl.read_csv(
            test_util.performance_data_path("Large-Cap Portfolio"),
            try_parse_dates=True,
        ).to_pandas()
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
                columns_to_sort: str | list[str] | None = None
                sort_descendings: bool | list[bool] = False
                if view == View.SUBPERIOD_ATTRIBUTION:
                    columns_to_sort = [
                        cols.BEGINNING_DATE,
                        cols.PORTFOLIO_WEIGHT,
                        cols.CLASSIFICATION_IDENTIFIER,
                    ]
                    sort_descendings = [True, False, False]

                file_name = f"{view.value}_{classification_name}.csv"
                with tempfile.TemporaryDirectory() as temp_dir:
                    output_path = Path(temp_dir) / file_name
                    attribution.write_csv(view, output_path, columns_to_sort, sort_descendings)
                    output = pl.read_csv(output_path)
                expected_path = test_util.resolve_file_path(
                    _EXPECTED_RESULTS_DIRECTORIES,
                    file_name,
                )
                self.assertTrue(output.equals(pl.read_csv(expected_path)))

                if classification_name == "Economic Sector":
                    attribution.to_json(view)
                    attribution.to_xml(view)

            if classification_name == "Economic Sector":
                for chart in Chart:
                    columns_to_sort = None
                    sort_descendings = False
                    if chart == Chart.OVERALL_ATTRIBUTION:
                        columns_to_sort = [cols.CLASSIFICATION_NAME]
                        sort_descendings = True
                    attribution.to_chart(chart, columns_to_sort, sort_descendings)

    def test_selected_attribution_values(self) -> None:
        """Selected classified attribution values remain stable across report views."""
        analytics = Analytics(
            test_util.performance_data_path("Mega-Cap Portfolio"),
            test_util.performance_data_path("Large-Cap Portfolio"),
            portfolio_classification_name="Security",
            benchmark_classification_name="Security",
            beginning_date=dt.date(2023, 10, 31),
            frequency=Frequency.MONTHLY,
        )
        economic_sector = test_util.get_attribution(analytics, "Economic Sector")
        security = test_util.get_attribution(analytics, "Security")

        self.assertTrue(
            util.are_near(
                economic_sector.to_polars(View.OVERALL_ATTRIBUTION)[cols.TOTAL_EFFECT_SMOOTHED][3],
                0.0047794523621773125,
            )
        )
        self.assertTrue(
            util.are_near(
                security.to_polars(View.OVERALL_ATTRIBUTION)[cols.ALLOCATION_EFFECT_SMOOTHED][0],
                -0.017280820318116667,
            )
        )
        self.assertTrue(
            util.are_near(
                economic_sector.to_polars(View.SUBPERIOD_ATTRIBUTION)[cols.ACTIVE_RETURN][12],
                0.03931589249102954,
            )
        )
        self.assertTrue(
            util.are_near(
                security.to_polars(View.SUBPERIOD_ATTRIBUTION)[cols.BENCHMARK_CONTRIB_SIMPLE][11],
                0.0002353459131385708,
            )
        )
        self.assertTrue(
            util.are_near(
                economic_sector.to_polars(View.SUBPERIOD_SUMMARY)[cols.TOTAL_EFFECT_SIMPLE][3],
                0.129471631945489,
            )
        )
        self.assertTrue(
            util.are_near(
                security.to_polars(View.SUBPERIOD_SUMMARY)[cols.ACTIVE_CONTRIB_SIMPLE][3],
                0.1294716319583555,
            )
        )
        self.assertTrue(
            util.are_near(economic_sector._df[cols.TOTAL_EFFECT_SMOOTHED][3], 0.1585372255258416)
        )
        self.assertTrue(
            util.are_near(economic_sector._df[cols.ACTIVE_CONTRIB_SMOOTHED][3], 0.1463257464885667)
        )

    def test_riskstatistics_csv_results_and_serialization_paths(self) -> None:
        """Risk-statistics output retains its CSV baseline and serialization paths."""
        analytics = Analytics(
            test_util.performance_data_path("Mega-Cap Portfolio"),
            test_util.performance_data_path("Large-Cap Portfolio"),
            portfolio_classification_name="Security",
            benchmark_classification_name="Security",
            beginning_date=dt.date(2021, 12, 31),
            ending_date=dt.date(2023, 3, 31),
            frequency=Frequency.QUARTERLY,
            annual_minimum_acceptable_return=-0.16,
        )
        risk_statistics = analytics.get_riskstatistics()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "riskstatistics.csv"
            risk_statistics.write_csv(output_path)
            output = pl.read_csv(output_path)
        expected_path = test_util.resolve_file_path(
            _EXPECTED_RESULTS_DIRECTORIES,
            "riskstatistics.csv",
        )

        self.assertTrue(output.equals(pl.read_csv(expected_path)))
        risk_statistics.to_json()
        risk_statistics.to_xml()

    def test_short_positions(self) -> None:
        """Short-position inputs process successfully through attribution."""
        analytics = Analytics(
            test_util.performance_data_path("case_short"),
            test_util.performance_data_path("Big 2"),
        )

        self.assertEqual(
            len(test_util.get_attribution(analytics).to_polars(View.SUBPERIOD_SUMMARY)),
            5,
        )


if __name__ == "__main__":
    unittest.main()
