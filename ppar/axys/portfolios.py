"""Load reconciled Axys portfolio performance outputs."""

from __future__ import annotations

# Python imports
from collections.abc import Callable
from dataclasses import dataclass
import datetime as dt
from typing import TYPE_CHECKING

# Third-party imports
import polars as pl

# Project imports
from ppar.axys import reconciliation
from ppar.axys.performance_sources import AxysPerformanceSourceLoader
from ppar.axys.specification import AxysSpecification
import ppar.columns as cols
from ppar.errors import PpaError
from ppar.frequency import Frequency
import ppar.utilities as util

if TYPE_CHECKING:
    from ppar.analytics import Analytics
    from ppar.axys.supporting_sources import AxysClassificationSources

_ANALYTICS_REQUIRED_COLUMNS = {
    cols.FROM_DATE,
    cols.THRU_DATE,
    cols.IDENTIFIER,
    cols.RETURN,
    cols.WEIGHT,
}
_SECPERF_CLASSIFICATION_NAME = "Security"
_PortfolioErrorMessage = Callable[[str, str | None], str]


@dataclass(frozen=True)
class AxysPortfolio:
    """Contain the reconciled performance output for one portfolio.

    Attributes:
        portfolio_code: Identifier used to select the portfolio in Axys sources.
        portfolio_name: Display name supplied to analytics output.
        secperf: Reconciled security-level performance rows accepted by
            :class:`ppar.analytics.Analytics`.
        classification_sources: Optional classification source bundle requested
            alongside the portfolio.
    """

    portfolio_code: str
    portfolio_name: str
    secperf: pl.DataFrame
    classification_sources: AxysClassificationSources | None = None

    def to_analytics(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        benchmark_data_source: util.PerformanceDataSource | None = None,
        benchmark_name: str | None = None,
        portfolio_classification_name: str = _SECPERF_CLASSIFICATION_NAME,
        benchmark_classification_name: str | None = None,
        from_date: str | dt.date = dt.date.min,
        thru_date: str | dt.date = dt.date.max,
        frequency: Frequency = Frequency.AS_OFTEN_AS_POSSIBLE,
        annual_minimum_acceptable_return: float = (
            util.DEFAULT_ANNUAL_MINIMUM_ACCEPTABLE_RETURN
        ),
        annual_risk_free_rate: float = util.DEFAULT_ANNUAL_RISK_FREE_RATE,
        confidence_level: float = util.DEFAULT_CONFIDENCE_LEVEL,
        portfolio_value: tuple[float, str] = (
            util.DEFAULT_PORTFOLIO_VALUE,
            util.DEFAULT_CURRENCY_SYMBOL,
        ),
    ) -> Analytics:
        """Return an Analytics instance for this reconciled Axys portfolio.

        Args:
            benchmark_data_source: Optional benchmark performance data source. When
                omitted, Analytics reuses the portfolio data as its benchmark.
            benchmark_name: Benchmark display name used in output titles.
            portfolio_classification_name: Classification name associated with Axys
                security-performance rows. Defaults to ``"Security"``.
            benchmark_classification_name: Classification name associated with the
                benchmark performance data.
            from_date: Earliest allowed from date.
            thru_date: Latest allowed thru date.
            frequency: Reporting frequency used to consolidate subperiods.
            annual_minimum_acceptable_return: Annual minimum acceptable return used in
                downside-risk calculations.
            annual_risk_free_rate: Annual risk-free rate used in risk statistics that
                require a risk-free return.
            confidence_level: Confidence level used when calculating value at risk.
            portfolio_value: Tuple containing the portfolio value and its currency
                symbol for value-at-risk calculations.

        Returns:
            Analytics instance initialized with this portfolio's reconciled
            security-performance rows, display name, and optional default attribution
            sources.

        Raises:
            PpaError: If Analytics validation fails.
        """
        # Import lazily so Axys portfolio containers do not force analytics imports
        # unless the convenience adapter is used.
        from ppar.analytics import Analytics  # pylint: disable=import-outside-toplevel

        return Analytics(
            portfolio_data_source=self.secperf,
            benchmark_data_source=benchmark_data_source,
            portfolio_name=self.portfolio_name,
            benchmark_name=benchmark_name,
            portfolio_classification_name=portfolio_classification_name,
            benchmark_classification_name=benchmark_classification_name,
            from_date=from_date,
            thru_date=thru_date,
            frequency=frequency,
            default_attribution_sources=self.classification_sources,
            annual_minimum_acceptable_return=annual_minimum_acceptable_return,
            annual_risk_free_rate=annual_risk_free_rate,
            confidence_level=confidence_level,
            portfolio_value=portfolio_value,
        )

    @property
    def required_classification_sources(self) -> AxysClassificationSources:
        """Return requested classification sources or raise a clear error.

        Returns:
            Classification source bundle requested with this portfolio.

        Raises:
            PpaError: If the portfolio was loaded without a classification.
        """
        if self.classification_sources is None:
            raise PpaError(
                "AxysPortfolio was loaded without classification sources. "
                "Pass classification_name to AxysData.get_portfolio().",
                999,
            )
        return self.classification_sources


class AxysPortfolioLoader:  # pylint: disable=too-few-public-methods
    """Load and reconcile requested Axys portfolios.

    Attributes:
        _specification: Parsed Axys source configuration.
        _loader: Source loader used to read portfolio and security performance.
        _error_message: Callback used to add facade-level validation context.
        _portperf_path: Portfolio-performance CSV path.
        _secperf_path: Security-performance CSV path.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        specification: AxysSpecification,
        loader: AxysPerformanceSourceLoader,
        error_message: _PortfolioErrorMessage,
        portperf_path: util.PathLike,
        secperf_path: util.PathLike,
    ) -> None:
        """Initialize a portfolio loader.

        Args:
            specification: Parsed Axys configuration used for display-name
                settings.
            loader: Source loader used to read portfolio and security data.
            error_message: Callback used to add facade-level source context.
            portperf_path: Portfolio-performance CSV path.
            secperf_path: Security-performance CSV path.
        """
        self._specification = specification
        self._loader = loader
        self._error_message = error_message
        self._portperf_path = portperf_path
        self._secperf_path = secperf_path

    def load(self, portfolio_codes: tuple[str, ...] | None) -> dict[str, AxysPortfolio]:
        """Return reconciled security performance for requested portfolios.

        Args:
            portfolio_codes: Portfolio codes to load, or ``None`` to discover
                all codes from the portfolio-performance source.

        Returns:
            Reconciled portfolio output keyed by portfolio code.

        Raises:
            PpaError: If input data cannot be loaded, common periods cannot be
                found, or security returns cannot be reconciled to portfolio
                returns.
        """
        if not portfolio_codes:
            portperf = self._loader.load(self._portperf_path, "portperf_columns")
            portfolio_codes = tuple(portperf[cols.PORTFOLIO_CODE].unique().sort().to_list())

        return {
            portfolio.portfolio_code: portfolio
            for portfolio_code in portfolio_codes
            if (portfolio := self._load_one(portfolio_code)) is not None
        }

    def _load_one(self, portfolio_code: str) -> AxysPortfolio | None:
        """Return one reconciled portfolio, or ``None`` when it has no rows.

        Args:
            portfolio_code: Portfolio code to load.

        Returns:
            Reconciled portfolio output, or ``None`` when no portfolio rows
            match the requested code.

        Raises:
            PpaError: If common periods cannot be found or security returns
                cannot be reconciled to portfolio returns.
        """
        portperf = self._loader.load(self._portperf_path, "portperf_columns", portfolio_code)
        if portperf.is_empty():
            return None

        secperf = self._loader.load(self._secperf_path, "secperf_columns", portfolio_code)
        portperf, secperf = reconciliation.filter_to_common_periods(
            portperf,
            secperf,
            lambda message, code=portfolio_code: self._error_message(message, code),
        )
        secperf, unreconciled_periods = reconciliation.derive_secperf_for_all_periods(
            portperf,
            secperf,
            lambda message, code=portfolio_code: self._error_message(message, code),
        )
        difference = reconciliation.unreconciled_difference(unreconciled_periods)
        if reconciliation.exceeds_fatal_tolerance(difference):
            raise PpaError(
                self._error_message(
                    f"Returns difference across unreconciled periods is {difference}",
                    portfolio_code,
                ),
                503,
            )

        portfolio_name = str(portperf[cols.PORTFOLIO_NAME][0])
        if self._specification.prefix_portfolio_code:
            portfolio_name = (
                f"{portfolio_code}{self._specification.prefix_portfolio_code}{portfolio_name}"
            )
        return AxysPortfolio(
            portfolio_code,
            portfolio_name,
            secperf.select(_ANALYTICS_REQUIRED_COLUMNS),
        )
