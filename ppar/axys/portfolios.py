"""Load reconciled Axys portfolio performance outputs."""

from __future__ import annotations

# Python imports
from collections.abc import Callable
from dataclasses import dataclass

# Third-party imports
import polars as pl

# Project imports
from ppar.axys import reconciliation
from ppar.axys.performance_sources import AxysPerformanceSourceLoader
from ppar.axys.specification import AxysSpecification
import ppar.columns as cols
from ppar.errors import PpaError
import ppar.utilities as util

_ANALYTICS_REQUIRED_COLUMNS = {
    cols.BEGINNING_DATE,
    cols.ENDING_DATE,
    cols.IDENTIFIER,
    cols.RETURN,
    cols.WEIGHT,
}
_PortfolioErrorMessage = Callable[[str, str | None], str]


@dataclass(frozen=True)
class AxysPortfolio:
    """Contain the reconciled performance output for one portfolio.

    Attributes:
        portfolio_code: Identifier used to select the portfolio in Axys sources.
        portfolio_name: Display name supplied to analytics output.
        secperf: Reconciled security-level performance rows accepted by
            :class:`ppar.analytics.Analytics`.
    """

    portfolio_code: str
    portfolio_name: str
    secperf: pl.DataFrame


class AxysPortfolioLoader:
    """Load and reconcile requested Axys portfolios.

    Attributes:
        _specification: Parsed Axys source configuration.
        _loader: Source loader used to read portfolio and security performance.
        _error_message: Callback used to add facade-level validation context.
        _portperf_path: Portfolio-performance CSV path.
        _secperf_path: Security-performance CSV path.
    """

    def __init__(
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
