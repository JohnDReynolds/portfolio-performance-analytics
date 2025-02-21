""" Tests """

## Overrides for pylint and pylance
# pyright: reportPrivateUsage=false
# pylint: disable=protected-access
# pylint: disable=too-many-lines
# pylint: disable=wrong-import-order
# pylint: disable=wrong-import-position

# Python Imports
import datetime as dt
import io
import math
import os
import sys
import tempfile
import unittest

# Third-Party Imports
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl

# # Project Imports
from ppa.analytics import Analytics
from ppa.attribution import Chart, View
import ppa.columns as cols
import ppa.errors as errs
from ppa.frequency import Frequency
from ppa.riskstatistics import RiskStatistics
from ppa.performance import Performance
import ppa.utilities as util

# Add the tests directory to the Python path (PYTHONPATH) so that it can find test_util_sources.py.
# Note that this is also done in .pylintrc and in .vscode/settings.json.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "tests")))
import test_utilities as test_util

# Directory Constants
_DATA_DIRECTORIES = ("tests/data/", "../tests/data/", "data/")
_CLASSIFICATION_DIRECTORIES = [f"{dir}classifications" for dir in _DATA_DIRECTORIES]
_EXPECTED_RESULTS_DIRECTORIES = [
    "tests/expected_results",
    "../tests/expected_results",
    "expected_results",
]
_MAPPING_DIRECTORIES = [f"{dir}mappings" for dir in _DATA_DIRECTORIES]


class Test(unittest.TestCase):
    """The Test Class containing all tests."""

    ############################## Performance Exceptions ##############################
    def test_102(self):
        """Test error 102."""
        self.assertTrue(
            _performance_exception(
                self, "error_102.csv", errs.ERROR_102_ENDING_DATES_ARE_NOT_UNIQUE
            )
        )

    def test_103(self):
        """Test error 103."""
        self.assertTrue(
            _performance_exception(
                self,
                "Magnificent 7.csv",
                errs.ERROR_103_NO_PERFORMANCE_ROWS,
                ending_date=dt.date(1900, 1, 31),
            )
        )

    def test_104(self):
        """Test error 104."""
        self.assertTrue(
            _attribution_exception(
                self,
                "error_104.csv",
                "aapl_daily.csv",
                errs.ERROR_104_MISSING_VALUES,
            )
        )

    def test_105(self):
        """Test error 105."""
        self.assertTrue(
            _performance_exception(
                self, "error_105.csv", errs.ERROR_105_BEGINNING_DATES_GREATER_THAN_ENDING_DATES
            )
        )

    def test_106(self):
        """Test error 106."""
        self.assertTrue(
            _performance_exception(self, "error_106.csv", errs.ERROR_106_DISCONTINUOS_TIME_PERIODS)
        )

    def test_107(self):
        """Test error 107."""
        self.assertTrue(
            _performance_exception(
                self, "error_107.csv", errs.ERROR_107_RETURN_COLUMNS_NOT_EQUAL_TO_WEIGHT_COLUMNS
            )
        )

    def test_108(self):
        """Test error 108."""
        self.assertTrue(
            _attribution_exception(
                self, "error_108.csv", "aapl_daily.csv", errs.ERROR_108_WEIGHTS_DO_NOT_SUM_TO_1
            )
        )

    def test_109(self):
        """Test error 109."""
        self.assertTrue(
            _performance_exception(self, "error_109.csv", errs.ERROR_109_NO_RETURNS_OR_WEIGHTS)
        )

    def test_110(self):
        """Test error 110: Bad floating point number."""
        self.assertTrue(
            _performance_exception(
                self,
                "error_110.csv",
                errs.ERROR_110_INVALID_PERFORMANCE_DATA_FORMAT,
            )
        )

    def test_111(self):
        """Test error 111."""
        self.assertTrue(
            _performance_exception(
                self,
                "aapl_daily.csv",
                errs.ERROR_111_INVALID_DATES,
                beginning_date=dt.date(1901, 1, 31),
                ending_date=dt.date(1901, 1, 30),
            )
        )

    ############################## Attribution Exceptions ##############################
    def test_202(self):
        """Test error 202."""
        self.assertTrue(
            _attribution_exception(
                self,
                "abcde_portfolio1.csv",
                "Magnificent 7.csv",
                errs.ERROR_202_NO_REPORTABLE_DATES,
            )
        )

    def test_203(self):
        """Test error 203."""
        self.assertTrue(
            _attribution_exception(
                self,
                "error_203.csv",
                "error_203.csv",
                errs.ERROR_203_UNDEFINED_RETURN,
            )
        )

    def test_204(self):
        """Test error 204."""
        self.assertTrue(
            _attribution_exception(
                self,
                "Magnificent 7",
                "Large-Cap Portfolio",
                errs.ERROR_204_TOO_MANY_HTML_ROWS,
                view=View.SUBPERIOD_ATTRIBUTION,
            )
        )

    ############################## Analytics Exceptions ##############################
    def test_252(self):
        """Test error 252: The classification name must be specified."""
        self.assertTrue(
            _attribution_exception(
                self,
                "abcde_portfolio1",
                "abcde_portfolio1",
                errs.ERROR_252_MUST_SPECIFY_CLASSIFICATION_NAME,
                portfolio_classification_name="Security",
                benchmark_classification_name="Security",
            )
        )

    ############################## Classification Exceptions ##############################
    def test_302(self):
        """Test error 302: The classification dataframe must contain 2 columns."""
        self.assertTrue(
            _attribution_exception(
                self,
                "abcde_portfolio1",
                "abcde_portfolio1",
                errs.ERROR_302_CLASSIFICATION_MUST_CONTAIN_2_COLUMNS,
                portfolio_classification_name="Security",
                benchmark_classification_name="Security",
                classification_name="Security",
                classification_data_source=pd.DataFrame({"col1": ["a", "b", "c"]}),
            )
        )

    ############################## Mapping Exceptions ##############################
    def test_353(self):
        """Test error 353: The mapping dataframe must contain 2 columns."""
        self.assertTrue(
            _attribution_exception(
                self,
                "abcde_portfolio1",
                "abcde_portfolio1",
                errs.ERROR_353_MAPPING_MUST_CONTAIN_2_COLUMNS,
                portfolio_classification_name="Security",
                benchmark_classification_name="Security",
                classification_name="Gics Sector",
                mapping_data_source=pd.DataFrame({"col1": ["a", "b", "c"]}),
            )
        )

    ############################## Ex-Post Risk Exceptions ##############################
    def test_402(self):
        """Test error 402."""
        self.assertTrue(
            _riskstatistics_exception(
                self,
                errs.ERROR_402_INVALID_FREQUENCY,
                (np.array([np.nan, np.nan]), np.array([np.nan, np.nan])),
                Frequency.AS_OFTEN_AS_POSSIBLE,
            )
        )

    def test_403(self):
        """Test error 403."""
        self.assertTrue(
            _riskstatistics_exception(
                self,
                errs.ERROR_403_INSUFFICIENT_QUANTITY_OF_RETURNS,
                (np.array([np.nan]), np.array([np.nan])),
                Frequency.MONTHLY,
            )
        )

    def test_404(self):
        """Test error 404."""
        self.assertTrue(
            _riskstatistics_exception(
                self,
                errs.ERROR_404_PORTFOLIO_BENCHMARK_RETURNS_QTY_NOT_EQUAL,
                (np.array([np.nan, np.nan]), np.array([np.nan, np.nan, np.nan])),
                Frequency.MONTHLY,
            )
        )

    def test_405(self):
        """Test error 405."""
        self.assertTrue(
            _riskstatistics_exception(
                self,
                errs.ERROR_405_NAN_VALUES,
                (np.array([1.0, 2.0]), np.array([1.0, np.nan])),
                Frequency.MONTHLY,
            )
        )

    ############################## General Exceptions ##############################
    def test_802(self):
        """Test error 802."""
        self.assertTrue(
            _attribution_exception(
                self,
                "_does_not_exist_",
                "aapl_daily.csv",
                errs.ERROR_802_FILE_DOES_NOT_EXIST,
            )
        )

    def test_803(self):
        """Test error 803."""
        self.assertTrue(
            _performance_exception(
                self, "aapl_daily.csv", errs.ERROR_803_CANNOT_CONVERT_TO_A_DATE, "2020-aa-bb"
            )
        )

    ############################## Test cases for utilities.py ##############################
    def test_are_near(self) -> None:
        """
        Test whether the are_near function correctly determines if two floats
        are within a given tolerance of each other.
        """
        self.assertTrue(util.are_near(1.0000000000001, 1.0, util.Tolerance.HIGH))
        self.assertFalse(util.are_near(1.0001, 1.0, util.Tolerance.LOW))

    def test_carino_linking_coefficient_assertion(self) -> None:
        """
        Test the carino_linking_coefficient function where the returns are invalid
        (i.e., <= -1.0), expecting an AssertionError.
        """
        with self.assertRaises(AssertionError) as cm:
            _ = util.carino_linking_coefficient(-1.0, 0.03)
        self.assertIn(errs.ERROR_203_UNDEFINED_RETURN, str(cm.exception))

        with self.assertRaises(AssertionError) as cm:
            _ = util.carino_linking_coefficient(0.05, -1.0)
        self.assertIn(errs.ERROR_203_UNDEFINED_RETURN, str(cm.exception))

    def test_carino_linking_coefficient_valid(self) -> None:
        """
        Test the carino_linking_coefficient function with valid portfolio and
        benchmark returns. Should return a float value without raising an assertion.
        """
        result = util.carino_linking_coefficient(0.05, 0.03)
        self.assertIsInstance(result, float, "Carino linking coefficient should be a float.")

    def test_col_names(self) -> None:
        """
        Test the col_names function to ensure it transforms suffixes properly.
        """
        from_columns = ["Port_ret", "Bench_ret"]
        transformed = list(cols.col_names(from_columns, "_wgt"))
        self.assertEqual(transformed, ["Port_wgt", "Bench_wgt"])

    def test_date_str(self) -> None:
        """
        Test date_str to ensure it formats the date as YYYY-MM-DD.
        """
        test_date = dt.date(2023, 1, 5)
        self.assertEqual(util.date_str(test_date), "2023-01-05")

    def test_file_basename_without_extension(self) -> None:
        """
        Test file_basename_without_extension to ensure it returns only the file name
        without the directory or extension.
        """
        path = "/some/path/to/myfile.csv"
        base_name = util.file_basename_without_extension(path)
        self.assertEqual(base_name, "myfile")

    def test_file_path_exists(self) -> None:
        """
        Test the _file_path_exists function with both an existing and non-existing file.
        (This demonstrates testing a private-like function if you import it.)
        """
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            temp_name = tmp_file.name
        try:
            self.assertTrue(util.file_path_exists(temp_name))
        finally:
            os.remove(temp_name)

        self.assertFalse(util.file_path_exists("not_a_real_file.xyz"))

    def test_load_dictionary_from_csv(self) -> None:
        """
        Test the load_dictionary_from_csv function by creating a temporary CSV file.
        """
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tmp_file:
            tmp_file.write("key1,value1\nkey2,value2\nkey3,value3")
            temp_name = tmp_file.name

        try:
            dct = util.load_dictionary_from_csv(temp_name)
            self.assertEqual(dct, {"key1": "value1", "key2": "value2", "key3": "value3"})
        finally:
            os.remove(temp_name)

    def test_logarithmic_linking_coefficient_series(self) -> None:
        """
        Test the logarithmic_linking_coefficient_series function using polars Series
        for overall_returns and subperiod returns.
        """
        overall_returns_series = pl.Series([0.02, 0.03, 0.05])
        returns_series = pl.Series([0.01, 0.02, 0.025])
        result_series = util.logarithmic_linking_coefficient_series(
            overall_returns_series, returns_series
        )
        self.assertIsInstance(result_series, pl.Series)
        self.assertEqual(result_series.len(), 3)

    def test_logarithmic_linking_coefficients(self) -> None:
        """
        Test the logarithmic_linking_coefficients function using polars Series.
        """
        returns_series = pl.Series([0.01, 0.02, 0.03])
        overall_return = 0.06
        result_series = util.logarithmic_linking_coefficients(overall_return, returns_series)
        self.assertIsInstance(result_series, pl.Series)
        self.assertEqual(result_series.len(), 3, "There should be 3 linking coefficients.")

    def test_near_zero(self) -> None:
        """
        Test near_zero function to ensure it detects values near 0 based on a tolerance.
        """
        self.assertTrue(util.near_zero(0.0000000000001, util.Tolerance.HIGH))
        self.assertFalse(util.near_zero(0.001, util.Tolerance.LOW))

    ############################## Test Various Data Formats ##############################
    def test_classification_data_and_mapping_data(self):
        """Test passing different formats of both classification_data and mapping_data.."""
        # Get the expected html using test_util.get_attribution().
        analytics = Analytics(
            test_util.performance_data_path("Big 2"),
            test_util.performance_data_path("Big 2"),
            portfolio_classification_name="Security",
            benchmark_classification_name="Security",
        )
        expected_html = test_util.get_attribution(analytics, "Gics Sub-Industry").to_html(
            View.OVERALL_ATTRIBUTION
        )

        # Get the same html, except instead of using test_util.get_attribution(), which reads a csv
        # file, specify mapping_data and classification_data as python dictionairies.
        html = test_util.get_attribution(
            analytics,
            "Gics Sub-Industry",
            mapping_data_source={"AAPL": "45202030", "MSFT": "45103020"},
            classification_data_source={
                "45103020": "Systems Software",
                "45202030": "Technology Hardware, Storage & Peripherals",
            },
        ).to_html(View.OVERALL_ATTRIBUTION)

        # Assert that the new results are equal to the expected results.
        assert test_util.html_table_lines(expected_html) == test_util.html_table_lines(html)

    def test_classification_data_formats(self):
        """Test the various formats of classification_data."""
        # Get the expected html using test_util.get_attribution().
        analytics = Analytics(
            test_util.performance_data_path("Big 2"),
            test_util.performance_data_path("Big 2"),
            portfolio_classification_name="Security",
            benchmark_classification_name="Security",
        )
        expected_html = test_util.get_attribution(analytics, "Security").to_html(
            View.OVERALL_ATTRIBUTION
        )

        # Test 4 different methods of specifying classification data.
        for i in range(4):
            if i == 0:
                # hard-coded csv file
                classification_data = util.resolve_file_path(
                    _CLASSIFICATION_DIRECTORIES, "Security.csv"
                )
            elif i == 1:
                # dictionary
                classification_data = {"AAPL": "Apple Inc.", "MSFT": "Microsoft"}
            else:
                classification_dict = {"c1": ["AAPL", "MSFT"], "c2": ["Apple Inc.", "Microsoft"]}
                if i == 2:
                    # pandas dataframe
                    classification_data = pd.DataFrame(classification_dict)
                else:  # i == 3
                    # polars dataframe
                    classification_data = pl.DataFrame(classification_dict)

            # Clear the cache
            analytics._attributions = {}

            # Assert the resulting attribution
            html = test_util.get_attribution(
                analytics, "Security", classification_data_source=classification_data
            ).to_html(View.OVERALL_ATTRIBUTION)
            assert test_util.html_table_lines(expected_html) == test_util.html_table_lines(html)

    def test_mapping_data_formats(self):
        """Test the various formats of mapping_data."""
        # Get the expected html using test_util.get_attribution().
        analytics = Analytics(
            test_util.performance_data_path("Big 2"),
            test_util.performance_data_path("Big 2"),
            portfolio_classification_name="Security",
            benchmark_classification_name="Security",
        )
        expected_html = test_util.get_attribution(analytics, "Gics Sub-Industry").to_html(
            View.OVERALL_ATTRIBUTION
        )

        # Test 4 different methods of specifying mapping data.
        for i in range(4):
            if i == 0:
                # hard-coded csv file
                mapping_data = util.resolve_file_path(
                    _MAPPING_DIRECTORIES, "Security--to--Gics Sub-Industry.csv"
                )
            elif i == 1:
                # dictionary
                mapping_data = {"AAPL": "45202030", "MSFT": "45103020"}
            else:
                map_dict: dict[str, list[str]] = {
                    "c1": ["AAPL", "MSFT"],
                    "c2": ["45202030", "45103020"],
                }
                if i == 2:
                    # pandas dataframe
                    mapping_data = pd.DataFrame(map_dict)
                else:  # i == 3
                    # polars dataframe
                    mapping_data = pl.DataFrame(map_dict)

            # Clear the cache
            analytics._attributions = {}

            # Assert the resulting attribution
            html = test_util.get_attribution(
                analytics, "Gics Sub-Industry", mapping_data_source=mapping_data
            ).to_html(View.OVERALL_ATTRIBUTION)
            assert test_util.html_table_lines(expected_html) == test_util.html_table_lines(html)

    def test_performance_data_formats(self):
        """Test the various formats of performance data."""
        # Get the expected html using test_util.get_attribution()
        analytics = Analytics(
            test_util.performance_data_path("Big 2"),
            test_util.performance_data_path("Big 2"),
            beginning_date=dt.date(2023, 12, 31),
            ending_date="2024-02-29",
        )
        expected_html = test_util.get_attribution(analytics).to_html(View.OVERALL_ATTRIBUTION)

        # Create a dictionary of performance data that can be turned into a DataFrame.
        performance_dict: dict[str, list[dt.date | float]] = {
            cols.BEGINNING_DATE: [dt.date(2023, 12, 31), dt.date(2024, 1, 31)],
            cols.ENDING_DATE: [dt.date(2024, 1, 31), dt.date(2024, 2, 29)],
            "aapl.wgt": [0.5, 0.5],
            "msft.wgt": [0.5, 0.5],
            "aapl.ret": [-0.0422272121, -0.019793881],
            "msft.ret": [0.0572811503, 0.0403944092],
        }

        for pdl in (pd, pl):
            # Get the analytics with the performance being either a pandas or a polars dataframe.
            analytics = Analytics(
                pdl.DataFrame(performance_dict),
                pdl.DataFrame(performance_dict),
                portfolio_name="Big 2",
                benchmark_name="Big 2",
            )
            # Assert the results.
            html = test_util.get_attribution(analytics).to_html(View.OVERALL_ATTRIBUTION)
            assert test_util.html_table_lines(html) == test_util.html_table_lines(expected_html)

    ############################## Test Charts ##############################
    def test_charts(self):
        """Test just to make sure that all of the charts run and do not fail."""
        for portfolio_benchmark in (("Big 2", "Magnificent 7"),):
            analytics = Analytics(
                test_util.performance_data_path(portfolio_benchmark[0]),
                test_util.performance_data_path(portfolio_benchmark[1]),
                portfolio_classification_name="Security",
                benchmark_classification_name="Security",
                beginning_date="2022-12-31",
                ending_date="2024-02-29",
                frequency=Frequency.MONTHLY,
            )
            attribution = test_util.get_attribution(analytics, "Gics Industry")
            for chart in Chart:
                print("Testing Gics Industry Chart", chart.value)
                attribution.to_chart(chart)

    ############################## Test Calculations and Auditing ##############################
    def test_abcde1(self):
        """Test basic attribution calculations for 5 assets with different subperiods."""
        analytics = Analytics(
            test_util.performance_data_path("abcde_portfolio1"),
            test_util.performance_data_path("abcde_portfolio2"),
        )

        # Portfolio contributions
        contribs = test_util.get_attribution(analytics).to_polars(View.SUBPERIOD_ATTRIBUTION)[
            cols.PORTFOLIO_CONTRIB_SIMPLE
        ]
        assert util.are_near(contribs.item(0), 0.03696005216365282)
        assert util.are_near(contribs.item(1), -0.05010275600837092)
        assert util.are_near(contribs.item(2), 0.015611261376729373)
        assert util.are_near(contribs.item(3), 0.029019065603398495)
        assert util.are_near(contribs.item(4), 0.07704845163844518)

        # Benchmark contributions
        contribs = test_util.get_attribution(analytics).to_polars(View.SUBPERIOD_ATTRIBUTION)[
            cols.BENCHMARK_CONTRIB_SIMPLE
        ]
        assert util.are_near(contribs.item(3), 0.001314124548289089)

    def test_abcde2(self):
        """Test basic attribution calculations for 5 assets with identical sub-eriods."""
        # Get the analytics
        analytics = Analytics(
            test_util.performance_data_path("abcde_portfolio1"),
            test_util.performance_data_path("abcde_benchmark1"),
        )

        # Get the attribution
        attribution = test_util.get_attribution(analytics)

        # Assert SUBPERIOD_SUMMARY
        subperiods = attribution.to_polars(View.SUBPERIOD_SUMMARY)
        assert util.are_near(subperiods[cols.PORTFOLIO_RETURN].item(0), 0.03638750268034727)
        assert util.are_near(subperiods[cols.PORTFOLIO_RETURN].item(1), 0.004100599095234386)
        assert util.are_near(subperiods[cols.BENCHMARK_RETURN].item(0), 0.03964350666619861)
        assert util.are_near(subperiods[cols.BENCHMARK_RETURN].item(2), 0.06673607157200062)

        # Assert OVERALL_ATTRIBUTION
        detail = attribution.to_polars(View.OVERALL_ATTRIBUTION)
        assert util.are_near(
            detail[cols.ALLOCATION_EFFECT_SMOOTHED].item(1), -0.0000097757165254280
        )
        assert util.are_near(
            detail[cols.SELECTION_EFFECT_SMOOTHED].item(1), -0.0016362229861442853
        )

        # Backwards compatibile test.
        df = attribution._construct_df_for_detail_views(View.SUBPERIOD_ATTRIBUTION).collect()
        assert util.are_near(df[cols.PORTFOLIO_CONTRIB_SMOOTHED].item(4), 0.01900264215424944)
        assert util.are_near(df[cols.BENCHMARK_CONTRIB_SMOOTHED].item(4), 0.019577639459518823)

    def test_attribution_content(self):
        """Test multiple attribution views and formats."""
        # Get the portfolio data as a Polars dataframe.
        portfolio_path = str(test_util.performance_data_path("Mega-Cap Portfolio"))
        portfolio_df = pl.scan_csv(source=portfolio_path, try_parse_dates=True).collect()

        # Get the benchmark data as a Pandas dataframe.
        benchmark_path = str(test_util.performance_data_path("Large-Cap Portfolio"))
        benchmark_df = (
            pl.scan_csv(source=benchmark_path, try_parse_dates=True).collect().to_pandas()
        )

        # Get the analytics
        analytics = Analytics(
            portfolio_df,
            benchmark_df,
            portfolio_name="Mega-Cap Portfolio",
            benchmark_name="Large-Cap Portfolio",
            portfolio_classification_name="Security",
            benchmark_classification_name="Security",
            beginning_date="2024-01-31",
        )

        # Assert each view.
        for view in View:
            # Assert each view/classification.
            for classification_name in ("Security", "Gics Sector"):
                print("Asserting View Content", view, classification_name)
                # Get the attribution
                attribution = test_util.get_attribution(analytics, classification_name)

                # Assert the view csv file
                file_name = f"{view.value}_{classification_name}.csv"
                test_file_path = os.path.join(tempfile.gettempdir(), file_name)
                attribution.write_csv(view, test_file_path)
                test_results = pl.read_csv(test_file_path)
                expected_file_path = util.resolve_file_path(
                    _EXPECTED_RESULTS_DIRECTORIES, file_name
                )
                expected_results = pl.read_csv(expected_file_path)
                # if not test_results.equals(expected_results):
                #     pause_it = 9
                assert test_results.equals(expected_results)
                os.remove(test_file_path)

                # Assert the view html file
                html = attribution.to_html(view)
                file_name = f"{view.value}_{classification_name}.html"
                test_file_path = os.path.join(tempfile.gettempdir(), file_name)
                with io.open(test_file_path, "w", encoding=util.ENCODING, newline="\n") as f:
                    f.write(html)
                test_results = test_util.read_html_table(test_file_path)
                expected_file_path = util.resolve_file_path(
                    _EXPECTED_RESULTS_DIRECTORIES, file_name
                )
                expected_results = test_util.read_html_table(expected_file_path)
                # if test_results != expected_results:
                #     pause_it = 9
                assert test_results == expected_results
                os.remove(test_file_path)

                # Just get the json and xml to make sure they do not fail.
                _ = attribution.to_json(view)
                _ = attribution.to_xml(view)

    def test_audit(self):
        """Test auditing a broad range of performance, classifications, frequencies and views."""

        # Declare the performance files for the portfolio/benchmark.
        file_names = (
            "aapl_daily",
            "Big 2",
            "Large-Cap Portfolio",
            "mag7_daily",
            "Magnificent 7",
        )

        # Iterate through different combinations of portfolio/benchmark/frequency trios.
        for file_name1 in file_names:
            # Portfolio
            for file_name2 in file_names:
                # Benchmark
                for frequency in Frequency:
                    # Frequency
                    print("Auditing", file_name1, file_name2, frequency)
                    analytics = Analytics(
                        test_util.performance_data_path(file_name1),
                        test_util.performance_data_path(file_name2),
                        portfolio_classification_name="Security",
                        benchmark_classification_name="Security",
                        frequency=frequency,
                    )

                    # Create a first attribution instance so analytisc.audit() below will have two
                    # attributions to audit side-by-side for common column equivalency.
                    _ = test_util.get_attribution(analytics, "Gics Sector")

                    # Test the Views
                    if frequency in (
                        Frequency.AS_OFTEN_AS_POSSIBLE,
                        Frequency.MONTHLY,
                    ):
                        # Test Gics Industry Group views
                        attribution = test_util.get_attribution(analytics, "Gics Industry Group")
                        for view in View:
                            attribution._audit_view(view)
                    else:
                        # Test Security views
                        attribution = test_util.get_attribution(analytics, "Security")
                        for view in View:
                            attribution._audit_view(view)
                        # Test Risk Statistics
                        risk_statistics = analytics.get_riskstatistics()
                        risk_statistics._audit()
                        risk_statistics.to_table()

                    # Audit the analytics
                    analytics.audit()

    def test_calculations(self):
        """Test basic calculations."""

        # Get the analytics
        analytics = Analytics(
            test_util.performance_data_path("Mega-Cap Portfolio"),
            test_util.performance_data_path("Large-Cap Portfolio"),
            portfolio_classification_name="Security",
            benchmark_classification_name="Security",
            beginning_date=dt.date(2023, 10, 31),
            frequency=Frequency.MONTHLY,
        )

        # Get the attributions
        gics = test_util.get_attribution(analytics, "Gics Sector")
        security = test_util.get_attribution(analytics, "Security")

        # Assert View.OVERALL_ATTRIBUTION
        content = gics.to_polars(View.OVERALL_ATTRIBUTION)
        assert util.are_near(content[cols.TOTAL_EFFECT_SMOOTHED][0], 0.0047794523621773125)
        content = security.to_polars(View.OVERALL_ATTRIBUTION)
        assert util.are_near(content[cols.ALLOCATION_EFFECT_SMOOTHED][0], -0.017280820318116667)

        # Assert View.SUBPERIOD_ATTRIBUTION
        content = gics.to_polars(View.SUBPERIOD_ATTRIBUTION)
        assert util.are_near(content[cols.ACTIVE_RETURN][12], 0.007395457599899)
        content = security.to_polars(View.SUBPERIOD_ATTRIBUTION)
        assert util.are_near(content[cols.BENCHMARK_CONTRIB_SIMPLE][11], 0.0002353459131385708)

        # Assert View.SUBPERIOD_SUMMARY
        content = gics.to_polars(View.SUBPERIOD_SUMMARY)
        assert util.are_near(content[cols.TOTAL_EFFECT_SIMPLE][3], 0.129471631945489)
        content = security.to_polars(View.SUBPERIOD_SUMMARY)
        assert util.are_near(content[cols.ACTIVE_CONTRIB_SIMPLE][3], 0.1294716319583555)

        # Backwards compatibile test.
        df = gics._df
        assert util.are_near(df[cols.TOTAL_EFFECT_SMOOTHED][3], 0.1585372255258416)
        assert util.are_near(df[cols.ACTIVE_CONTRIB_SMOOTHED][3], 0.1463257464885667)

    def test_riskstatistics_content(self):
        """Test risk statistics csv and html."""
        # Get the analytics.
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

        # Get the risk statistics.
        riskstatistics = analytics.get_riskstatistics()

        # Assert the expected results in the riskstatistics.csv file
        file_name = "riskstatistics.csv"
        test_file_path = os.path.join(tempfile.gettempdir(), file_name)
        riskstatistics.write_csv(test_file_path)
        test_results: pl.DataFrame = pl.read_csv(test_file_path)
        expected_file_path = util.resolve_file_path(_EXPECTED_RESULTS_DIRECTORIES, file_name)
        expected_results: pl.DataFrame = pl.read_csv(expected_file_path)
        # if not test_results.equals(expected_results):
        #     pause_it = 9
        assert test_results.equals(expected_results)
        os.remove(test_file_path)

        # Assert the expected results in the riskstatistics.html file
        html = riskstatistics.to_html()
        file_name = "riskstatistics.html"
        test_file_path = os.path.join(tempfile.gettempdir(), file_name)
        with io.open(test_file_path, "w", encoding=util.ENCODING, newline="\n") as f:
            f.write(html)
        test_results2: list[str] = test_util.read_html_table(test_file_path)
        expected_file_path = util.resolve_file_path(_EXPECTED_RESULTS_DIRECTORIES, file_name)
        expected_results2: list[str] = test_util.read_html_table(expected_file_path)
        # if test_results2 != expected_results2:
        #     pause_it = 9
        assert test_results2 == expected_results2
        os.remove(test_file_path)

        # Just get the json and xml to make sure they do not fail.
        _ = riskstatistics.to_json()
        _ = riskstatistics.to_xml()

    ############################## Test Frequencies and Dates ##############################
    def test_crazy_frequency(self):
        """Test the consolidation of odd mis-matched frequencies."""
        analytics = Analytics(
            test_util.performance_data_path("case_mixed_frequency"),
            test_util.performance_data_path("case_crazy_frequency"),
        )
        assert len(test_util.get_attribution(analytics).to_pandas(View.SUBPERIOD_SUMMARY)) == 3

    def test_daily_to_monthly(self):
        """Test consolidating daily to monthly."""
        # Get the analytics.
        analytics = Analytics(
            test_util.performance_data_path("big2_daily"),
            test_util.performance_data_path("Big 2"),
            beginning_date=dt.date(2020, 12, 31),
            frequency=Frequency.MONTHLY,
        )

        # Get the attribution
        attribution = test_util.get_attribution(analytics)

        # Assert SUBPERIOD_ATTRIBUTION
        df = attribution.to_polars(View.SUBPERIOD_ATTRIBUTION)
        assert df[cols.BEGINNING_DATE].item(0) == dt.date(2020, 12, 31)
        assert df[cols.ENDING_DATE].item(4) == dt.date(2021, 3, 31)
        assert util.are_near(df[cols.TOTAL_EFFECT_SIMPLE].item(3), 0.0012545960452570828)
        assert util.are_near(df[cols.SELECTION_EFFECT_SIMPLE].item(14), 0.001057705826113624)

        # Backwards compatibile test.
        df = attribution._construct_df_for_detail_views(View.SUBPERIOD_ATTRIBUTION).collect()
        assert util.are_near(df[cols.TOTAL_EFFECT_SMOOTHED].item(3), 0.002038295249203867)
        assert util.are_near(df[cols.SELECTION_EFFECT_SMOOTHED].item(14), 0.0015709213702753996)

    def test_daily_to_quarterly(self):
        """Test consolidating daily to quarterly."""
        # Get the analytics.
        analytics = Analytics(
            test_util.performance_data_path("big2_daily"),
            test_util.performance_data_path("Big 2"),
            beginning_date=dt.date(2020, 12, 31),
            frequency=Frequency.QUARTERLY,
        )

        # Get the attribution.
        attribution = test_util.get_attribution(analytics)

        # Assert SUBPERIOD_SUMMARY
        df = attribution.to_polars(View.SUBPERIOD_SUMMARY)
        assert df[cols.BEGINNING_DATE].item(0) == dt.date(2020, 12, 31)
        assert df[cols.ENDING_DATE].item(4) == dt.date(2022, 3, 31)
        assert util.are_near(df[cols.TOTAL_EFFECT_SIMPLE].item(3), -0.0020721529010043226)
        assert util.are_near(df[cols.PORTFOLIO_RETURN].item(8), 0.2401702546346276)

        # Backwards compatibile test.
        df = attribution._df
        assert util.are_near(df[cols.TOTAL_EFFECT_SMOOTHED].item(3), -0.002740959239265768)
        assert util.are_near(df[cols.PORTFOLIO_RETURN].item(8), 0.2401702546346276)

    def test_map_mixed_frequency(self):
        """Test mapping and consolidating mixed frequencies."""
        # Get the analytics
        analytics = Analytics(
            test_util.performance_data_path("Magnificent 7"),
            test_util.performance_data_path("gics_sector_daily"),
            portfolio_classification_name="Security",
            benchmark_classification_name="Gics Sector",
        )

        # Assert Gics Sector attribution classification identifiers.
        classifications = test_util.get_attribution(analytics, "Gics Sector").to_polars(
            View.OVERALL_ATTRIBUTION
        )[cols.CLASSIFICATION_IDENTIFIER]
        assert classifications.item(0) == "10"
        assert classifications.item(1) == "15"
        assert classifications.item(2) == "25"
        assert classifications.item(3) == "35"
        assert classifications.item(4) == "45"
        assert classifications.item(5) == "50"
        assert test_util.get_attribution(analytics, "Gics Sector").to_polars(
            View.SUBPERIOD_SUMMARY
        ).shape == (141, 11)

    def test_mixed_frequency(self):
        """Test mixed frequencies."""
        analytics = Analytics(
            test_util.performance_data_path("case_mixed_frequency"),
            test_util.performance_data_path("case_monthly_frequency"),
        )
        assert len(test_util.get_attribution(analytics).to_polars(View.SUBPERIOD_SUMMARY)) == 3

    def test_monthly_to_yearly(self):
        """Test consolidating monthly to yearly."""
        analytics = Analytics(
            test_util.performance_data_path("Big 2"),
            test_util.performance_data_path("big2_daily"),
            beginning_date=dt.date(2020, 12, 31),
            frequency=Frequency.YEARLY,
        )

        df = test_util.get_attribution(analytics).to_polars(View.SUBPERIOD_SUMMARY)
        assert len(df) == 3
        assert df[cols.BEGINNING_DATE].item(0) == dt.date(2020, 12, 31)
        assert df[cols.ENDING_DATE].item(2) == dt.date(2023, 12, 31)

    def test_specify_dates(self):
        """Test date filtering."""
        perf = Performance(
            test_util.performance_data_path("case_adjust_beginning_dates"),
            beginning_date="2023-01-31",
            ending_date="2023-02-28",
        )
        assert perf.df[cols.BEGINNING_DATE].item(0) == dt.date(2023, 1, 31) and perf.df[
            cols.ENDING_DATE
        ].item(1) == dt.date(2023, 2, 28)

    ############################## Miscellaneous Tests ##############################
    def test_no_classification_name(self):
        """Test not specifying any classification name."""
        analytics = Analytics(
            test_util.performance_data_path("abcde_portfolio1"),
            test_util.performance_data_path("abcde_portfolio1"),
        )
        test_util.get_attribution(analytics).to_html(View.OVERALL_ATTRIBUTION)

    def test_non_annualizability(self):
        """Test nan for non-annualizability less than 1 year."""
        expostrisk = RiskStatistics(
            (np.array([1, 2, 3]), np.array([4, 5, 6])), Frequency.QUARTERLY
        )
        df = expostrisk.to_polars()
        assert math.isnan(df["Portfolio"].item(2))
        assert math.isnan(df["Benchmark"].item(2))

    def test_short_positions(self):
        """Test short positiions."""
        analytics = Analytics(
            test_util.performance_data_path("case_short"), test_util.performance_data_path("Big 2")
        )
        assert len(test_util.get_attribution(analytics).to_polars(View.SUBPERIOD_SUMMARY)) == 5


######################### Module-Wide Functions ########################
def _attribution_exception(
    test: Test,
    file_name1: str,
    file_name2: str,
    error_message: str,
    portfolio_classification_name: str = util.EMPTY,
    benchmark_classification_name: str = util.EMPTY,
    classification_name: str = util.EMPTY,
    classification_data_source: util.TypeClassificationDataSource = util.EMPTY,
    mapping_data_source: util.TypeMappingDataSource = util.EMPTY,
    view: View | None = None,
):
    """Test Attribution exception."""
    with test.assertRaises(Exception) as context:
        # Get the analytics
        analytics = Analytics(
            test_util.performance_data_path(file_name1),
            test_util.performance_data_path(file_name2),
            portfolio_classification_name=portfolio_classification_name,
            benchmark_classification_name=benchmark_classification_name,
        )

        # Get the attribution
        attribution = test_util.get_attribution(
            analytics,
            classification_name,
            classification_data_source=classification_data_source,
            mapping_data_source=mapping_data_source,
        )

        # Check for view.
        if view is not None:
            attribution.to_html(view)

    # Print the exception
    print(str(context.exception))

    # Assert that the exception starts with the expected error message
    return str(context.exception).startswith(error_message)


def _performance_exception(
    test: Test,
    file_name: str,
    error_message: str,
    beginning_date: str | dt.date = dt.date.min,
    ending_date: str | dt.date = dt.date.max,
):
    """Test Performance exception."""
    with test.assertRaises(Exception) as context:
        Performance(
            test_util.performance_data_path(file_name),
            beginning_date=beginning_date,
            ending_date=ending_date,
        )
    print(str(context.exception))
    return str(context.exception).startswith(error_message)


def _riskstatistics_exception(
    test: Test,
    error_message: str,
    returns: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    frequency: Frequency,
    minimum_acceptable_return: float = 0,
):
    """Test RiskStatistics exception."""
    with test.assertRaises(Exception) as context:
        RiskStatistics(returns, frequency, minimum_acceptable_return)
    print(str(context.exception))
    return str(context.exception).startswith(error_message)


#################################
if __name__ == "__main__":
    unittest.main()
