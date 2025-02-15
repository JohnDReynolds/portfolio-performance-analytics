"""
Copyright 2025 John D. Reynolds, All Rights Reserved.

This software and associated documentation files (the “Software”) are provided for reference or
archival purposes only. No license is granted to any person to copy, modify, publish, distribute,
sublicense, or sell copies of the Software, or to use it for any commercial or non-commercial
purpose, without the express prior written permission of John D. Reynolds.

The RiskStatistics class calculates the ex-post risk statistics enumerated in the "Statistic" enum.

The public methods to retrieve the resulting statistics are:
    1. to_html()
    2. to_pandas()
    3. to_polars()
    4. to_table()
    5. write_csv()
"""

# Python Imports
import datetime as dt
from enum import Enum
import math

# Third-Party Imports
import great_tables as gt
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
from scipy.stats import norm  # type: ignore

# Project Imports
import ppa.columns as cols
import ppa.errors as errs
from ppa.frequency import Frequency, periods_per_year
from ppa.performance import Performance
import ppa.utilities as util

# Constants
_DEFAULT_OUTPUT_PRECISION = 8


class Statistic(Enum):
    """An enumeration of the different types of statistics.  Arranged in view order."""

    RANGE = "Range"
    STANDARD_DEVIATION = "Standard Deviation"
    STANDARD_DEVIATION_ANNUALIZED = "Annualized Standard Deviation"
    DOWNSIDE_PROBABILITY = "Downside Probability"  # aka "Shortfall Risk"
    EXPECTED_DOWNSIDE_VALUE = "Expected Downside Value"
    DOWNSIDE_DEVIATION = "Downside Deviation"
    DOWNSIDE_DEVIATION_ANNUALIZED = "Annualized Downside Deviation"
    VALUE_AT_RISK = "Value At Risk (VAR)"
    CORRELATION = "Correlation"
    R_SQUARED = "R-Squared"  # aka "Coefficient Of Determination"
    TRACKING_ERROR = "Tracking Error"
    TRACKING_ERROR_ANNUALIZED = "Annualized Tracking Error"
    SHARPE_RATIO = "Sharpe Ratio"
    SHARPE_RATIO_ANNUALIZED = "Annualized Sharpe Ratio"
    SORTINO_RATIO = "Sortino Ratio"
    SORTINO_RATIO_ANNUALIZED = "Annualized Sortino Ratio"
    INFORMATION_RATIO = "Information Ratio"
    M_SQUARED = "M_Squared"  # aka "Modigliani-Modigliani"
    TREYNOR_RATIO = "Treynor Ratio"
    BETA = "Beta"
    ALPHA = "Alpha"
    ALPHA_ANNUALIZED = "Annualized Alpha"
    JENSENS_ALPHA = "Jensens Alpha"
    JENSENS_ALPHA_ANNUALIZED = "Annualized Jensens Alpha"


# View categories.
_CATEGORIES = pl.Series(
    "Category",
    (["Absolute Risk"] * 3)
    + (["Downside Risk"] * 5)
    + (["Benchmark-Relative Risk"] * 4)
    + (["Risk-Adjusted Performance"] * 7)
    + (["Regression"] * 5),
)

# The minimum quantity of returns in order to calculate the statistics.
_MINIMUM_QUANTITY_OF_RETURNS = 2


class RiskStatistics:
    """
    The RiskStatistics class calculates the ex-post risk statistics enumerated in the
    "Statistic" enum.

    The public methods to retrieve the resulting statistics are:
        1. to_html()
        2. to_pandas()
        3. to_polars()
        4. to_table()
        5. write_csv()
    """

    def __init__(
        self,
        returns: (
            tuple[Performance, Performance]
            | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        ),
        frequency: Frequency,
        annual_minimum_acceptable_return: float = util.DEFAULT_ANNUAL_MINIMUM_ACCEPTABLE_RETURN,
        annual_risk_free_rate: float = util.DEFAULT_ANNUAL_RISK_FREE_RATE,
        confidence_level: float = util.DEFAULT_CONFIDENCE_LEVEL,
        portfolio_value: float = util.DEFAULT_PORTFOLIO_VALUE,
    ):
        """
        The constructor.  Calculates the ex-post risk statistics enumerated in the "Statistic"
        enum.  Stores them in the dictionary self._df.

        Args:
            returns (tuple[Performance, Performance]  |
                tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]):
                Two Performances or numpy arrays of periodic returns.  0 = Portfolio, 1 = Benchmark
            frequency (Frequency): The frequency of both the portfolio returns and the
                benchmark returns.  Either:
                1. Frequency.MONTHLY
                2. Frequency.QUARTERLY
                3. Frequency.YEARLY
            annual_minimum_acceptable_return (float, optional): The minimum acceptable return used
                for calculating "downside" satistics.
                Defaults to util.DEFAULT_ANNUAL_MINIMUM_ACCEPTABLE_RETURN.
            annual_risk_free_rate (float, optional): The annual risk-free rate used for
                calculating statistics that involve a risk-free rates.
                Defaults to util.DEFAULT_ANNUAL_RISK_FREE_RATE.
            confidence_level (float, optional): The confidence level for calculating the
                value-at-risk (VAR).  Defaults to util.DEFAULT_CONFIDENCE_LEVEL.
            portfolio_value (float, optional): The portfolio value (stated in a currency) for
                calculating the value-at-risk (VAR).  Defaults to util.DEFAULT_PORTFOLIO_VALUE.

        Raises:
            errs.PpaError: Error if frequency is invalid.
        """
        # Set and validate the frequency.
        self._frequency = frequency
        assert (
            self._frequency != Frequency.AS_OFTEN_AS_POSSIBLE
        ), f"{errs.ERROR_402_INVALID_FREQUENCY}{self._frequency}"

        # Set the dates, names and returns depending on the input parameters.
        self._performances: tuple[Performance, Performance] = tuple()
        if isinstance(returns[0], Performance) and isinstance(returns[1], Performance):
            self._performances = returns  # type: ignore
            self._beginning_date = returns[0].df[cols.BEGINNING_DATE][0]
            self._ending_date = returns[0].df[cols.ENDING_DATE][-1]
            self._portfolio_name = returns[0].name
            self._benchmark_name = returns[1].name
            portfolio_returns = returns[0].df[cols.TOTAL_RETURN].to_numpy()
            benchmark_returns = returns[1].df[cols.TOTAL_RETURN].to_numpy()
        elif isinstance(returns[0], np.ndarray) and isinstance(returns[1], np.ndarray):
            self._beginning_date = dt.date.min
            self._ending_date = dt.date.max
            self._portfolio_name = "Portfolio"
            self._benchmark_name = "Benchmark"
            portfolio_returns = returns[0]
            benchmark_returns = returns[1]
        else:
            # Should never reach here.
            raise errs.PpaError(
                f"{errs.ERROR_999_UNEXPECTED}Unknown returns type in RiskStatistics constructor."
            )

        quantity_of_returns = len(portfolio_returns)

        # Validate that the portfolio and benchmark have the same quantity of returns.
        assert quantity_of_returns == len(benchmark_returns), (
            f"{errs.ERROR_404_PORTFOLIO_BENCHMARK_RETURNS_QTY_NOT_EQUAL}"
            f"{quantity_of_returns} <> {len(benchmark_returns)}"
        )

        # Validate that there are enough returns.
        assert (
            _MINIMUM_QUANTITY_OF_RETURNS <= quantity_of_returns
        ), f"{errs.ERROR_403_INSUFFICIENT_QUANTITY_OF_RETURNS}{quantity_of_returns}"

        # Validate that there are not any NaN values.
        assert not np.any(np.isnan(portfolio_returns)) and not np.any(
            np.isnan(benchmark_returns)
        ), f"{errs.ERROR_405_NAN_VALUES}"

        # Set the annualization coefficient.  It is not kosher to annualize numbers that are for
        # less than a year, so in that case, use NaN.
        qty_periods_per_year = periods_per_year(self._frequency)
        annualization_coefficient = (
            np.nan
            if quantity_of_returns < qty_periods_per_year
            else math.sqrt(qty_periods_per_year)
        )

        # Set the minimum acceptable return and risk free return to correspond with the frequency.
        # Note that a lot of the old literature just divides the annual rate by the quantity of the
        # periods per year, which is probably a shortcut used before computers.  We use the correct
        # de-annualized formula.
        frequency_mar = RiskStatistics._deannualize_return(
            annual_minimum_acceptable_return, qty_periods_per_year
        )
        frequency_rfr = RiskStatistics._deannualize_return(
            annual_risk_free_rate, qty_periods_per_year
        )

        # Calculate common values used below.
        active_returns = portfolio_returns - benchmark_returns
        benchmark_mean = np.mean(benchmark_returns)

        # Initialize the dictionaries for the statistics.
        statistic_values: dict[str, list[np.float64]] = {}

        # Calculate the statistics.
        for idx, rets in enumerate((portfolio_returns, benchmark_returns)):
            # Calculate the basic statistics.
            mean = float(np.mean(rets))  # typecast for pylint
            stddev = float(np.std(rets))  # typecast for pylint

            # Calculate the downside_deviation that is used for DOWNSIDE_DEVIATION and
            # DOWNSIDE_DEVIATION_ANNUALIZED.  Note that downside_returns will include
            # (return - mar) for all values below the mar, and then zeros for returns equal to or
            # above the mar.
            downside_returns = np.clip(rets - frequency_mar, a_min=-np.inf, a_max=0)
            downside_deviation = np.sqrt(np.mean(downside_returns**2))

            # Calculate returns_below_mar that is used for DOWNSIDE_PROBABILITY and
            # EXPECTED_DOWNSIDE_VALUE.  Note that returns_below_mar will only include
            # (return - mar) for values below the mar.  It will not include returns equal to or
            # above the mar.
            returns_below_mar = rets[rets < frequency_mar] - frequency_mar

            # Calculate the risk-free ratios.
            excess_returns_mean, sharpe_ratio, sortino_ratio = (
                RiskStatistics._calculate_risk_free_ratios(rets, frequency_rfr)
            )

            if idx == 0:  # Portfolio
                # Calculate the active returns over the benchmark.
                active_returns_stddev = np.std(active_returns)
                # Calculate the beta.  Note that the "capm beta" would be:
                # _beta(portfolio_returns - frequency_rfr, benchmark_returns - frequency_rfr)
                # But since you are only supporting a constant risk-free rate for all periods
                # (frequency_rfr), the capm_beta will be identical to the regular beta that does
                # not use the risk-adjusted returns.
                beta = RiskStatistics._beta(portfolio_returns, benchmark_returns)
                # Calculate the alpha
                alpha = mean - (beta * benchmark_mean)
                # Calculate the correlation coefficient.
                correlation_coefficient = np.corrcoef(portfolio_returns, benchmark_returns)[
                    0, 1
                ]  # type: ignore
                # Calculate Jensen's Alpha
                jensens_alpha = excess_returns_mean - (beta * (benchmark_mean - frequency_rfr))
            else:  # Benchmark
                # These are all NaN for the benchmark.
                active_returns_stddev = np.nan
                beta = np.nan
                alpha = np.nan
                correlation_coefficient = np.nan
                jensens_alpha = np.nan

            for statistic in Statistic:
                if idx == 0:
                    # Allocate the statistic: 0 = Portfolio, 1 = Benchmark.
                    statistic_values[statistic.value] = []

                # Set the appropriate statistic value.
                match statistic:
                    case Statistic.ALPHA:
                        value = alpha
                    case Statistic.ALPHA_ANNUALIZED:
                        value = annualization_coefficient * alpha
                    case Statistic.BETA:
                        value = beta
                    case Statistic.CORRELATION:
                        value = correlation_coefficient
                    case Statistic.DOWNSIDE_DEVIATION:
                        value = downside_deviation
                    case Statistic.DOWNSIDE_DEVIATION_ANNUALIZED:
                        value = annualization_coefficient * downside_deviation
                    case Statistic.DOWNSIDE_PROBABILITY:
                        value = len(returns_below_mar) / quantity_of_returns
                    case Statistic.EXPECTED_DOWNSIDE_VALUE:
                        value = sum(returns_below_mar) / quantity_of_returns
                    case Statistic.INFORMATION_RATIO:
                        if idx == 0:
                            value = (
                                np.inf
                                if active_returns_stddev == 0
                                else np.mean(active_returns) / active_returns_stddev
                            )
                        else:
                            value = np.nan
                    case Statistic.JENSENS_ALPHA:
                        value = jensens_alpha
                    case Statistic.JENSENS_ALPHA_ANNUALIZED:
                        value = annualization_coefficient * jensens_alpha
                    case Statistic.M_SQUARED:
                        if idx == 0:
                            # M2 = (Sharpe Ratio * Benchmark Standard Deviation) + Risk-Free Rate,
                            value = (sharpe_ratio * np.std(benchmark_returns)) + frequency_rfr
                        else:
                            value = np.nan
                    case Statistic.R_SQUARED:
                        value = correlation_coefficient**2
                    case Statistic.RANGE:
                        value = np.max(rets) - np.min(rets)
                    case Statistic.SHARPE_RATIO:
                        value = sharpe_ratio
                    case Statistic.SHARPE_RATIO_ANNUALIZED:
                        value = annualization_coefficient * sharpe_ratio
                    case Statistic.SORTINO_RATIO:
                        value = sortino_ratio
                    case Statistic.SORTINO_RATIO_ANNUALIZED:
                        value = annualization_coefficient * sortino_ratio
                    case Statistic.STANDARD_DEVIATION:
                        value = stddev
                    case Statistic.STANDARD_DEVIATION_ANNUALIZED:
                        value = annualization_coefficient * stddev
                    case Statistic.TRACKING_ERROR:
                        value = active_returns_stddev
                    case Statistic.TRACKING_ERROR_ANNUALIZED:
                        value = annualization_coefficient * active_returns_stddev
                    case Statistic.TREYNOR_RATIO:
                        if idx == 0:
                            value = excess_returns_mean / beta
                        else:
                            value = np.nan
                    case Statistic.VALUE_AT_RISK:
                        value = RiskStatistics._parametric_var(
                            mean, stddev, confidence_level, portfolio_value
                        )

                statistic_values[statistic.value].append(value)  # type: ignore

        # Create the final DataFrame.
        self._df = (
            pl.DataFrame(statistic_values)
            # Transpose 2 rows to 2 Portfolio and Benchmark columns.
            .transpose(include_header=True, column_names=("Portfolio", "Benchmark"))
            .lazy()
            # Add the Difference column.
            .with_columns((pl.col("Portfolio") - pl.col("Benchmark")).alias("Difference"))
            # Add the Category column.
            .with_columns(_CATEGORIES)
            .collect()
        )

    def _audit(self) -> None:
        """Audit the RiskStatistics instance (self)."""
        # Audit the portfolio/benchmark pair of performances.
        Performance.audit_performances(self._performances, self._beginning_date, self._ending_date)

    @staticmethod
    def _beta(
        portfolio_returns: npt.NDArray[np.float64], benchmark_returns: npt.NDArray[np.float64]
    ) -> float:
        """
        Calculates the beta between the portfolio and the benchmark.

        Args:
            portfolio_returns (npt.NDArray[np.float64]): The portfolio returns.
            benchmark_returns (npt.NDArray[np.float64]): The benchmark returns.

        Returns:
            float: The beta between the portfolio and the benchmark.
        """
        covariance_matrix = np.cov(portfolio_returns, benchmark_returns)
        covariance = covariance_matrix[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        return covariance / benchmark_variance

    @staticmethod
    def _calculate_risk_free_ratios(
        returns: npt.NDArray[np.float64], frequency_rfr: float
    ) -> tuple[float, float, float]:
        """
        Calculate the statistics that are adjusted by the risk-free rate.

        Args:
            returns (npt.NDArray[np.float64]): The array of returns.
            frequency_rfr (float): The risk-free rate corresponding to the returns frequency.

        Returns:
            tuple[float, float, float]: excess_returns_mean, sharpe_ratio, sortino_ratio
        """
        # Calculate the excess returns for the Sortino Ratio and Sharpe Ratio.
        # Note that excess_returns_downside will only include excess_returns < 0.
        # It will not include excess_returns => 0.
        # Note also that rets.std() == excess_returns.std() as long as the risk-free
        # returns are constant for all periods, which they are.  In the future, you might want
        # to support an array of risk-free rates for each period.
        excess_returns = returns - frequency_rfr
        excess_returns_mean = excess_returns.mean()
        sharpe_ratio = excess_returns_mean / excess_returns.std()
        excess_returns_downside = excess_returns[excess_returns < 0]
        excess_returns_downside_std = (
            0.0 if len(excess_returns_downside) == 0 else excess_returns_downside.std()
        )
        sortino_ratio = (
            np.inf
            if excess_returns_downside_std == 0.0
            else excess_returns_mean / excess_returns_downside.std()
        )
        return excess_returns_mean, sharpe_ratio, sortino_ratio

    @staticmethod
    def _deannualize_return(annual_return: float, qty_periods_per_year: int) -> float:
        """
        De-annualizes the annual_return to correspond with the frequency.

        Args:
            annual_return (float): The annual return.
            qty_periods_per_year: The quantity of periods per year.

        Returns:
            float: The de-annualized return corresponding with the frequency.
        """
        return (1 + annual_return) ** (1 / qty_periods_per_year) - 1

    @staticmethod
    def _parametric_var(
        mean: float,
        stddev: float,
        confidence_level: float,
        portfolio_value: float,
    ) -> float:
        """
        Calculate Parametric Value at Risk (VaR).

        Args:
            mean (float): Mean return of the portfolio (stated in frequency).
            stddev (float): Standard deviation of the portfolio returns (stated in frequency).
            confidence_level (float): Confidence level (e.g., 0.95 for 95% confidence).
            portfolio_value (float): Total currency value of the portfolio.

        Returns:
            float: Value at Risk (VaR).
        """
        # Calculate the z-score for the given confidence level. typecast for pylint
        z_score = float(norm.ppf(1 - confidence_level))  # type: ignore

        # Calculate the VaR
        var = portfolio_value * (mean - (z_score * stddev))

        # Return the absolute value.
        return abs(var)

    def to_html(self) -> str:
        """
        Returns the view in an html format.

        Args:
            view (View): The desired View.

        Returns:
            str: The view in an html format.
        """
        return self.to_table().as_raw_html(make_page=True)

    def to_json(self, float_precision: int = _DEFAULT_OUTPUT_PRECISION) -> str:
        """
        Returns the view as a json string.

        Args:
            float_precision (int, optional): The quantity of decimal places.
                Defaults to _DEFAULT_OUTPUT_PRECISION.

        Returns:
            str: The view as a json string.
        """
        return self.to_pandas().to_json(double_precision=float_precision)  # type: ignore

    def to_pandas(self) -> pd.DataFrame:
        """
        Returns the view as a pandas DataFrame.

        Args:
            view (View): The desired View.

        Returns:
            str: The view as a pandas DataFrame.
        """
        return self._df.to_pandas()

    def to_polars(self) -> pl.DataFrame:
        """
        Returns the view as a polars DataFrame.

        Args:
            view (View): The desired View.

        Returns:
            str: The view as a polars DataFrame.
        """
        return self._df

    def to_table(self) -> gt.GT:
        """
        Returns a "great_table" of the view.

        Args:
            view (View): The desired View.

        Returns:
            gt.GT: A "great_table" of the view.
        """
        # Set the title and subtitle.
        title = f"{self._portfolio_name} vs {self._benchmark_name}"
        subtitle = (
            f"Ex-Post Risk Statistics: {self._frequency.value} from {self._beginning_date} to "
            f"{self._ending_date}"
        )

        # Return the formatted table.
        return (
            gt.GT(self._df.to_pandas())
            .tab_header(title=title, subtitle=subtitle)
            .tab_stub(rowname_col="column", groupname_col="Category")
            .fmt_number(columns=["Portfolio", "Benchmark", "Difference"], decimals=4)
            .opt_row_striping()
        )

    def write_csv(self, file_path: str, float_precision: int = _DEFAULT_OUTPUT_PRECISION) -> None:
        """
        Writes a csv file of the view.

        Args:
            view (View): The desired View.
            file_path (str): The file path of the csv file to be written to.
            float_precision (int, optional): The quantity of decimal places.
                Defaults to util._DEFAULT_OUTPUT_PRECISION.
        """
        self._df.write_csv(file_path, float_precision=float_precision)

    def to_xml(self) -> str:
        """
        Returns the view as an xml string.

        Returns:
            str: The view as an xml string.
        """
        return self.to_pandas().to_xml()
