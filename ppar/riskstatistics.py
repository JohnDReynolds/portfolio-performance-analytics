"""Calculate and format ex-post portfolio risk statistics.

This module contains the ``Statistic`` enumeration and the ``RiskStatistics``
class. ``RiskStatistics`` calculates absolute risk, downside risk,
benchmark-relative risk, risk-adjusted performance, and regression statistics
for a portfolio and benchmark return series.
"""

# Python Imports
import datetime as dt
from enum import Enum
import math
from typing import cast

# Third-Party Imports
import great_tables as gt
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
from scipy.stats import norm  # type: ignore

# Project Imports
import ppar.columns as cols
from ppar.errors import PpaError
from ppar.frequency import Frequency, periods_per_year
from ppar.performance import Performance
import ppar.utilities as util

# Constants
_DEFAULT_OUTPUT_PRECISION = 8


class Statistic(Enum):
    """Enumeration of supported ex-post risk statistics.

    The values are arranged in the order used by the formatted output views.
    """

    # Absolute Risk
    RETURN_RANGE = "Return Range"
    MEAN_RETURN = "Mean Return"
    MEAN_RETURN_ANNUALIZED = "Annualized Mean Return"
    STANDARD_DEVIATION = "Standard Deviation"
    STANDARD_DEVIATION_ANNUALIZED = "Annualized Standard Deviation"
    # Downside Rirk
    DOWNSIDE_PROBABILITY = "Downside Probability"  # aka "Shortfall Risk"
    EXPECTED_DOWNSIDE_VALUE = "Expected Downside Value"
    DOWNSIDE_DEVIATION = "Downside Deviation"
    DOWNSIDE_DEVIATION_ANNUALIZED = "Annualized Downside Deviation"
    VALUE_AT_RISK = "Value At Risk"
    # Benchmark-Relative Risk
    CORRELATION = "Correlation"
    R_SQUARED = "R-Squared"  # aka "Coefficient Of Determination"
    TRACKING_ERROR = "Tracking Error"
    TRACKING_ERROR_ANNUALIZED = "Annualized Tracking Error"
    # Risk-Adjusted Performance
    SHARPE_RATIO = "Sharpe Ratio"
    SHARPE_RATIO_ANNUALIZED = "Annualized Sharpe Ratio"
    SORTINO_RATIO = "Sortino Ratio"
    SORTINO_RATIO_ANNUALIZED = "Annualized Sortino Ratio"
    INFORMATION_RATIO = "Information Ratio"
    M_SQUARED = "M_Squared"  # aka "Modigliani-Modigliani"
    TREYNOR_RATIO = "Treynor Ratio"
    # Regression
    BETA = "Beta"  # slope
    ALPHA = "Alpha"  # intercept
    ALPHA_ANNUALIZED = "Annualized Alpha"
    JENSENS_ALPHA = "Jensens Alpha"
    JENSENS_ALPHA_ANNUALIZED = "Annualized Jensens Alpha"


# View categories.
_CATEGORIES = pl.Series(
    "Category",
    (["Absolute Risk"] * 5)
    + (["Downside Risk"] * 5)
    + (["Benchmark-Relative Risk"] * 4)
    + (["Risk-Adjusted Performance"] * 7)
    + (["Regression"] * 5),
)

# The minimum quantity of returns in order to calculate the statistics.
_MINIMUM_QUANTITY_OF_RETURNS = 2


class RiskStatistics:
    """Calculate ex-post risk statistics for portfolio and benchmark returns.

    The class accepts either two ``Performance`` instances or two NumPy arrays
    of periodic returns. It calculates the statistics enumerated by
    ``Statistic`` and stores the formatted results in a Polars DataFrame.

    Public output methods:
        1. ``to_html()``
        2. ``to_json()``
        3. ``to_pandas()``
        4. ``to_polars()``
        5. ``to_table()``
        6. ``to_xml()``
        7. ``write_csv()``
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
        portfolio_value: tuple[float, str] = (
            util.DEFAULT_PORTFOLIO_VALUE,
            util.DEFAULT_CURRENCY_SYMBOL,
        ),
    ):
        """Initialize and calculate risk statistics.

        Args:
            returns: Either a tuple of two ``Performance`` instances or a
                tuple of two NumPy arrays of periodic returns. Index ``0`` is
                the portfolio and index ``1`` is the benchmark.
            frequency: Frequency of the portfolio and benchmark returns.
                Supported values are ``Frequency.MONTHLY``,
                ``Frequency.QUARTERLY``, and ``Frequency.YEARLY``.
            annual_minimum_acceptable_return: Annual minimum acceptable return
                used for downside-risk calculations.
            annual_risk_free_rate: Annual risk-free rate used for
                risk-adjusted performance statistics.
            confidence_level: Confidence level used for value-at-risk
                calculations.
            portfolio_value: Tuple containing the portfolio value and currency
                symbol used when calculating and displaying value at risk.

        Raises:
            PpaError: If ``frequency`` is
                ``Frequency.AS_OFTEN_AS_POSSIBLE``, ``returns`` is not a
                supported tuple type, the portfolio and benchmark return
                series have different lengths, there are fewer than two
                returns, or either return series contains NaN values.
        """
        # Set and validate the frequency.
        self._frequency = frequency
        if self._frequency == Frequency.AS_OFTEN_AS_POSSIBLE:
            raise PpaError(f"{self._frequency}", 402)

        # Set the currency symbol used when presenting the VaR.
        self._currency_symbol = portfolio_value[1]

        # Set the dates, names and returns depending on the input parameters.
        if isinstance(returns[0], Performance) and isinstance(returns[1], Performance):
            self._beginning_date = returns[0].df[cols.BEGINNING_DATE][0]
            self._ending_date = returns[0].df[cols.ENDING_DATE][-1]
            self._portfolio_name = returns[0].name
            self._benchmark_name = returns[1].name
            self._portfolio_returns = returns[0].df[cols.TOTAL_RETURN].to_numpy()
            self._benchmark_returns = returns[1].df[cols.TOTAL_RETURN].to_numpy()
            self._performances_to_audit: tuple[Performance, Performance] = returns  # type: ignore
        elif isinstance(returns[0], np.ndarray) and isinstance(returns[1], np.ndarray):
            self._beginning_date = dt.date.min
            self._ending_date = dt.date.max
            self._portfolio_name = "Portfolio"
            self._benchmark_name = "Benchmark"
            self._portfolio_returns = returns[0]
            self._benchmark_returns = returns[1]
            self._performances_to_audit: tuple[Performance, Performance] = tuple()  # type: ignore
        else:
            # Should never reach here.
            raise PpaError("Unknown returns type in RiskStatistics constructor.", 999)

        # Now that self._portfolio_returns has been established, set self._quantity_of_returns.
        self._quantity_of_returns = len(self._portfolio_returns)

        # Validate that the portfolio and benchmark have the same quantity of returns.
        if self._quantity_of_returns != len(self._benchmark_returns):
            raise PpaError(f"{self._quantity_of_returns} <> {len(self._benchmark_returns)}", 404)

        # Validate that there are enough returns.
        if self._quantity_of_returns < _MINIMUM_QUANTITY_OF_RETURNS:
            raise PpaError(f"{self._quantity_of_returns}", 403)

        # Validate that there are not any NaN values.
        if np.any(np.isnan(self._portfolio_returns)) or np.any(np.isnan(self._benchmark_returns)):
            raise PpaError("", 405)

        # Get all statistic values.
        statistic_values = self._calculate_all_statistics(
            annual_minimum_acceptable_return,
            annual_risk_free_rate,
            confidence_level,
            portfolio_value[0],
        )

        # Create self._df from the statistic_values dictionary.
        self._df = pl.DataFrame(statistic_values)

        # Rename the non-annualized column names so the frequency is prepended, and
        # the currency symbol is prepended for the portfolio_value.
        self._df.columns = [
            self._frequency_column_name(col, portfolio_value[0]) for col in self._df.columns
        ]

        # Create the final DataFrame.
        self._df = (
            self._df  ############################pl.DataFrame(statistic_values)
            # Transpose 2 rows to 2 Portfolio and Benchmark columns.
            .transpose(include_header=True, column_names=("Portfolio", "Benchmark"))
            .lazy()
            # Add the Difference column.
            .with_columns((pl.col("Portfolio") - pl.col("Benchmark")).alias("Difference"))
            # Add the Category column.
            .with_columns(_CATEGORIES)
            .collect()
        )

    def _annualize_return(self, mean_frequency_return: float, qty_periods_per_year: int) -> float:
        """Annualize a mean periodic return.

        Args:
            mean_frequency_return: Mean return for the input frequency.
            qty_periods_per_year: Number of periods per year for the input
                frequency.

        Returns:
            Annualized return, or ``np.nan`` when the return series contains
            fewer than one year's worth of observations.
        """
        # Cannot annualize if you do not have at least a years worth of returns, so return np.nan.
        return (
            np.nan
            if self._quantity_of_returns < qty_periods_per_year
            else ((1 + mean_frequency_return) ** qty_periods_per_year) - 1
        )

    def _audit(self) -> None:
        """Audit the source performance pair.

        Raises:
            PpaError: Raised by ``Performance.audit_performances()`` if the
                source portfolio and benchmark performances fail validation.
        """
        # Audit the portfolio/benchmark pair of performances.
        Performance.audit_performances(
            self._performances_to_audit, self._beginning_date, self._ending_date
        )

    @staticmethod
    def _beta(
        portfolio_returns: npt.NDArray[np.float64], benchmark_returns: npt.NDArray[np.float64]
    ) -> float:
        """Calculate beta between portfolio and benchmark returns.

        Args:
            portfolio_returns: Portfolio periodic returns.
            benchmark_returns: Benchmark periodic returns.

        Returns:
            Portfolio beta relative to the benchmark.
        """
        covariance_matrix = np.cov(portfolio_returns, benchmark_returns)
        covariance = covariance_matrix[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        return cast(float, covariance / benchmark_variance)  # cast for mypy

    def _calculate_all_statistics(
        self,
        annual_minimum_acceptable_return: float,
        annual_risk_free_rate: float,
        confidence_level: float,
        portfolio_value: float,
    ) -> dict[str, list[np.float64]]:
        """Calculate all statistic values for portfolio and benchmark.

        Args:
            annual_minimum_acceptable_return: Annual minimum acceptable return
                used for downside-risk calculations.
            annual_risk_free_rate: Annual risk-free rate used for
                risk-adjusted performance statistics.
            confidence_level: Confidence level used for value-at-risk
                calculations.
            portfolio_value: Portfolio currency value used for value-at-risk
                calculations.

        Returns:
            Dictionary keyed by statistic display name. Each value contains
            the portfolio statistic followed by the benchmark statistic.

        Raises:
            PpaError: Raised by ``periods_per_year()`` if the stored frequency
                is unsupported for annualization.
        """
        # Set the annualization coefficient.  It is not kosher to annualize numbers that are for
        # less than a year, so in that case, use NaN.
        qty_periods_per_year = periods_per_year(self._frequency)
        annualization_coefficient = (
            np.nan
            if self._quantity_of_returns < qty_periods_per_year
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

        # Calculate other common values used below.
        active_returns = self._portfolio_returns - self._benchmark_returns
        benchmark_mean = np.mean(self._benchmark_returns)

        # Initialize the dictionaries for the statistics.
        statistic_values: dict[str, list[np.float64]] = {}

        # Calculate the statistics.
        for idx, rets in enumerate((self._portfolio_returns, self._benchmark_returns)):
            # Calculate the basic statistics.
            mean = cast(float, np.mean(rets))
            stddev = cast(float, np.std(rets))

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
                beta = RiskStatistics._beta(self._portfolio_returns, self._benchmark_returns)
                # Calculate the alpha
                alpha = mean - (beta * benchmark_mean)
                # Calculate the correlation coefficient.
                correlation_coefficient = np.corrcoef(
                    self._portfolio_returns, self._benchmark_returns
                )[
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
                        value = len(returns_below_mar) / self._quantity_of_returns
                    case Statistic.EXPECTED_DOWNSIDE_VALUE:
                        value = sum(returns_below_mar) / self._quantity_of_returns
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
                            value = (
                                sharpe_ratio * np.std(self._benchmark_returns)
                            ) + frequency_rfr
                        else:
                            value = np.nan
                    case Statistic.MEAN_RETURN:
                        value = mean
                    case Statistic.MEAN_RETURN_ANNUALIZED:
                        value = self._annualize_return(mean, qty_periods_per_year)
                    case Statistic.R_SQUARED:
                        value = correlation_coefficient**2
                    case Statistic.RETURN_RANGE:
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

        # Return all statistic values.
        return statistic_values

    @staticmethod
    def _calculate_risk_free_ratios(
        returns: npt.NDArray[np.float64], frequency_rfr: float
    ) -> tuple[float, float, float]:
        """Calculate risk-free-rate-adjusted performance ratios.

        Args:
            returns: Periodic return series.
            frequency_rfr: Risk-free rate converted to the same periodic
                frequency as ``returns``.

        Returns:
            Tuple containing mean excess return, Sharpe ratio, and Sortino
            ratio.
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
        """Convert an annual return to a periodic return.

        Args:
            annual_return: Annual return to convert.
            qty_periods_per_year: Number of periods per year.

        Returns:
            Periodic return corresponding to ``qty_periods_per_year``.
        """
        return cast(
            float, ((1 + annual_return) ** (1 / qty_periods_per_year)) - 1
        )  # cast for mypy

    def _frequency_column_name(self, column_name: str, portfolio_value: float) -> str:
        """Build a display column name for a statistic.

        Non-annualized statistics are prefixed with the reporting frequency.
        Value-at-risk is also labeled with the portfolio currency value.

        Args:
            column_name: Base statistic column name.
            portfolio_value: Portfolio currency value used for value-at-risk
                labeling.

        Returns:
            Display column name.
        """
        if column_name == Statistic.VALUE_AT_RISK.value:
            return (
                f"{self._frequency.value} {column_name} for "
                f"{self._currency_symbol}{portfolio_value:,.0f}"
            )
        if not column_name.startswith("Annualized"):
            return f"{self._frequency.value} {column_name}"
        return column_name

    @staticmethod
    def _parametric_var(
        mean: float,
        stddev: float,
        confidence_level: float,
        portfolio_value: float,
    ) -> float:
        """Calculate parametric value at risk.

        Args:
            mean: Mean periodic return.
            stddev: Standard deviation of periodic returns.
            confidence_level: Confidence level, such as ``0.95`` for 95%.
            portfolio_value: Portfolio currency value.

        Returns:
            Parametric value at risk as a positive currency amount.
        """
        # Calculate the z-score for the given confidence level.
        z_score = norm.ppf(1 - confidence_level)  # type: ignore

        # Calculate the VaR
        var = portfolio_value * (mean - (z_score * stddev))

        # Even though a VaR represents a potential loss, it is typically reported as a positive
        # number.  Some models present it as a negative number to emphasize the potential loss
        # (e.g., VaR = -$1 million means a potential $1 million loss).  The sign convention depends
        # on how the institution or analyst defines it.  We will return it as a positive number.
        return cast(float, var)

    def to_html(self) -> str:
        """Return the statistics table as an HTML page string.

        Returns:
            HTML string containing the formatted risk-statistics table.
        """
        return self.to_table().as_raw_html(make_page=True)

    def to_json(self, float_precision: int = _DEFAULT_OUTPUT_PRECISION) -> str:
        """Return the statistics as a JSON string.

        Args:
            float_precision: Number of decimal places to include for floating
                point values.

        Returns:
            JSON representation of the risk-statistics DataFrame.
        """
        return self.to_pandas().to_json(double_precision=float_precision)  # type: ignore

    def to_pandas(self) -> pd.DataFrame:
        """Return the statistics as a pandas DataFrame.

        Returns:
            pandas DataFrame containing the calculated risk statistics.
        """
        return self._df.to_pandas()

    def to_polars(self) -> pl.DataFrame:
        """Return the statistics as a Polars DataFrame.

        Returns:
            Polars DataFrame containing the calculated risk statistics.
        """
        return self._df

    def to_table(self) -> gt.GT:
        """Return the statistics as a formatted Great Tables table.

        Returns:
            ``great_tables.GT`` object containing the formatted
            risk-statistics table.
        """
        # Set the title and subtitle.
        title = f"{self._portfolio_name} vs {self._benchmark_name}"
        subtitle = (
            f"Ex-Post Risk Statistics: {self._frequency.value} from {self._beginning_date} to "
            f"{self._ending_date}"
        )

        # Return the formatted table.
        column_names = ["Portfolio", "Benchmark", "Difference"]
        return (
            gt.GT(self._df.to_pandas())
            .tab_header(title=title, subtitle=subtitle)
            .tab_stub(rowname_col="column", groupname_col="Category")
            # Format the statistics to 4 decimals (bps)
            .fmt_number(columns=column_names, decimals=4)  # pyright: ignore
            # Format the VaR to 0 decimals since it represents a currency amount.
            .fmt_number(
                columns=column_names,
                pattern=f"{self._currency_symbol}{{x}}",  # the actual str {x}
                decimals=0,
                rows=9,
            )
            .opt_row_striping()
        )

    def to_xml(self) -> str:
        """Return the statistics as an XML string.

        Returns:
            XML representation of the risk-statistics DataFrame.
        """
        return self.to_pandas().to_xml()

    def write_csv(self, file_path: str, float_precision: int = _DEFAULT_OUTPUT_PRECISION) -> None:
        """Write the statistics to a CSV file.

        Args:
            file_path: Output CSV file path.
            float_precision: Number of decimal places to write for floating
                point values.
        """
        self._df.write_csv(file_path, float_precision=float_precision)
