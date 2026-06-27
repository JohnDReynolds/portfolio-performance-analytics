"""Portfolio return-reconstruction diagnostics for performance comparison."""

from __future__ import annotations

# Python imports
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# Third-party imports
import polars as pl

# Project imports
from ppar.errors import PpaError
from ppar.performance_comparison import schema as pc_cols
from ppar.performance_comparison.holdings import HoldingsLoader
from ppar.performance_comparison.modified_dietz import modified_dietz_flow_weight
from ppar.performance_comparison.portfolio_performance import PortfolioPerformanceLoader
from ppar.performance_comparison.security_performance import SecurityPerformanceLoader
from ppar.performance_comparison.specification import (
    PerformanceComparisonSpecification,
    PortfolioReturnReconstruction,
    SecurityReturnReconstruction,
)
from ppar.performance_comparison.transactions import TransactionsLoader
import ppar.utilities as util

RECONSTRUCTION_REVIEW_KEY: Final[str] = "review_key"
RECONSTRUCTION_PORTFOLIO_ID: Final[str] = pc_cols.PORTFOLIO_ID
RECONSTRUCTION_SECURITY_ID: Final[str] = pc_cols.SECURITY_ID
RECONSTRUCTION_FROM_DATE: Final[str] = pc_cols.FROM_DATE
RECONSTRUCTION_THRU_DATE: Final[str] = pc_cols.THRU_DATE
REPORTED_RETURN_A: Final[str] = "reported_return_a"
REPORTED_RETURN_B: Final[str] = "reported_return_b"
REPORTED_RETURN_DIFFERENCE: Final[str] = "reported_return_difference"
DERIVED_RETURN_A: Final[str] = "derived_return_a"
DERIVED_RETURN_B: Final[str] = "derived_return_b"
DERIVED_RETURN_DIFFERENCE: Final[str] = "derived_return_difference"
RECONSTRUCTION_DIFFERENCE: Final[str] = "reconstruction_difference"
DERIVED_NUMERATOR_A: Final[str] = "derived_numerator_a"
DERIVED_NUMERATOR_B: Final[str] = "derived_numerator_b"
DERIVED_NUMERATOR_DIFFERENCE: Final[str] = "derived_numerator_difference"
DERIVED_DENOMINATOR_A: Final[str] = "derived_denominator_a"
DERIVED_DENOMINATOR_B: Final[str] = "derived_denominator_b"
DERIVED_DENOMINATOR_DIFFERENCE: Final[str] = "derived_denominator_difference"
BEGIN_VALUE_A: Final[str] = "begin_value_a"
BEGIN_VALUE_B: Final[str] = "begin_value_b"
BEGIN_VALUE_DIFFERENCE: Final[str] = "begin_value_difference"
END_VALUE_A: Final[str] = "end_value_a"
END_VALUE_B: Final[str] = "end_value_b"
END_VALUE_DIFFERENCE: Final[str] = "end_value_difference"
NET_FLOW_A: Final[str] = "net_flow_a"
NET_FLOW_B: Final[str] = "net_flow_b"
NET_FLOW_DIFFERENCE: Final[str] = "net_flow_difference"
WEIGHTED_FLOW_A: Final[str] = "weighted_flow_a"
WEIGHTED_FLOW_B: Final[str] = "weighted_flow_b"
WEIGHTED_FLOW_DIFFERENCE: Final[str] = "weighted_flow_difference"
INCOME_A: Final[str] = "income_a"
INCOME_B: Final[str] = "income_b"
INCOME_DIFFERENCE: Final[str] = "income_difference"
BEGIN_VALUE_DATE_A: Final[str] = "begin_value_date_a"
BEGIN_VALUE_DATE_B: Final[str] = "begin_value_date_b"
END_VALUE_DATE_A: Final[str] = "end_value_date_a"
END_VALUE_DATE_B: Final[str] = "end_value_date_b"
RECONSTRUCTION_STATUS: Final[str] = "reconstruction_status"
RECONSTRUCTION_CATEGORY: Final[str] = "reconstruction_category"
RECONSTRUCTION_COMMENTS: Final[str] = "reconstruction_comments"
RECONSTRUCTION_CHECK_TYPE: Final[str] = "reconstruction_check_type"
RECONSTRUCTION_ROW_COUNT: Final[str] = "row_count"

RECONSTRUCTION_STATUS_ALIGNED: Final[str] = "Aligned"
RECONSTRUCTION_STATUS_DIFFERENT: Final[str] = "Different"
RECONSTRUCTION_STATUS_MISSING_INPUTS: Final[str] = "Missing Inputs"
RECONSTRUCTION_CATEGORY_ALIGNED: Final[str] = "Aligned"
RECONSTRUCTION_CATEGORY_MISSING_INPUTS: Final[str] = "Missing Inputs"
RECONSTRUCTION_CATEGORY_SOURCE_INPUTS_CHANGED: Final[str] = "Source Inputs Changed"
RECONSTRUCTION_CATEGORY_FORMULA_DIFFERENCE: Final[str] = "Formula Difference"
RECONSTRUCTION_CHECK_TYPE_PORTFOLIO: Final[str] = "Portfolio Return"
RECONSTRUCTION_CHECK_TYPE_SECURITY: Final[str] = "Security Return"

PORTFOLIO_RETURN_RECONSTRUCTION_COLUMNS: Final[tuple[str, ...]] = (
    RECONSTRUCTION_REVIEW_KEY,
    RECONSTRUCTION_PORTFOLIO_ID,
    RECONSTRUCTION_FROM_DATE,
    RECONSTRUCTION_THRU_DATE,
    REPORTED_RETURN_A,
    REPORTED_RETURN_B,
    REPORTED_RETURN_DIFFERENCE,
    DERIVED_RETURN_A,
    DERIVED_RETURN_B,
    DERIVED_RETURN_DIFFERENCE,
    RECONSTRUCTION_DIFFERENCE,
    DERIVED_NUMERATOR_A,
    DERIVED_NUMERATOR_B,
    DERIVED_NUMERATOR_DIFFERENCE,
    DERIVED_DENOMINATOR_A,
    DERIVED_DENOMINATOR_B,
    DERIVED_DENOMINATOR_DIFFERENCE,
    BEGIN_VALUE_A,
    BEGIN_VALUE_B,
    BEGIN_VALUE_DIFFERENCE,
    END_VALUE_A,
    END_VALUE_B,
    END_VALUE_DIFFERENCE,
    NET_FLOW_A,
    NET_FLOW_B,
    NET_FLOW_DIFFERENCE,
    WEIGHTED_FLOW_A,
    WEIGHTED_FLOW_B,
    WEIGHTED_FLOW_DIFFERENCE,
    BEGIN_VALUE_DATE_A,
    BEGIN_VALUE_DATE_B,
    END_VALUE_DATE_A,
    END_VALUE_DATE_B,
    RECONSTRUCTION_STATUS,
    RECONSTRUCTION_CATEGORY,
    RECONSTRUCTION_COMMENTS,
)

SECURITY_RETURN_RECONSTRUCTION_COLUMNS: Final[tuple[str, ...]] = (
    RECONSTRUCTION_REVIEW_KEY,
    RECONSTRUCTION_PORTFOLIO_ID,
    RECONSTRUCTION_SECURITY_ID,
    RECONSTRUCTION_FROM_DATE,
    RECONSTRUCTION_THRU_DATE,
    REPORTED_RETURN_A,
    REPORTED_RETURN_B,
    REPORTED_RETURN_DIFFERENCE,
    DERIVED_RETURN_A,
    DERIVED_RETURN_B,
    DERIVED_RETURN_DIFFERENCE,
    RECONSTRUCTION_DIFFERENCE,
    DERIVED_NUMERATOR_A,
    DERIVED_NUMERATOR_B,
    DERIVED_NUMERATOR_DIFFERENCE,
    DERIVED_DENOMINATOR_A,
    DERIVED_DENOMINATOR_B,
    DERIVED_DENOMINATOR_DIFFERENCE,
    BEGIN_VALUE_A,
    BEGIN_VALUE_B,
    BEGIN_VALUE_DIFFERENCE,
    END_VALUE_A,
    END_VALUE_B,
    END_VALUE_DIFFERENCE,
    NET_FLOW_A,
    NET_FLOW_B,
    NET_FLOW_DIFFERENCE,
    WEIGHTED_FLOW_A,
    WEIGHTED_FLOW_B,
    WEIGHTED_FLOW_DIFFERENCE,
    INCOME_A,
    INCOME_B,
    INCOME_DIFFERENCE,
    BEGIN_VALUE_DATE_A,
    BEGIN_VALUE_DATE_B,
    END_VALUE_DATE_A,
    END_VALUE_DATE_B,
    RECONSTRUCTION_STATUS,
    RECONSTRUCTION_CATEGORY,
    RECONSTRUCTION_COMMENTS,
)

RECONSTRUCTION_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    RECONSTRUCTION_CHECK_TYPE,
    RECONSTRUCTION_STATUS,
    RECONSTRUCTION_CATEGORY,
    RECONSTRUCTION_ROW_COUNT,
)


@dataclass(frozen=True)
class _SnapshotReturnInputs:
    """Return inputs derived for one portfolio-period snapshot."""

    reported_return: float | None
    derived_return: float | None
    derived_numerator: float | None
    derived_denominator: float | None
    begin_value: float | None
    end_value: float | None
    net_flow: float | None
    weighted_flow: float | None
    income: float | None
    begin_value_date: dt.date | None
    end_value_date: dt.date | None
    comments: tuple[str, ...]


def portfolio_return_reconstruction_checks(
    comparison_path: util.PathLike | None,
) -> pl.DataFrame:
    """Return portfolio return-reconstruction diagnostics for a comparison YAML.

    Args:
        comparison_path: Path to a performance comparison YAML file. ``None``
            returns an empty table because reconstruction needs source data.

    Returns:
        Stable Polars table with one row per portfolio-period that can be
        checked against the configured reconstruction rules. Empty when the YAML
        does not opt into ``portfolio_return_reconstruction``.

    Raises:
        PpaError: If opted-in YAML is malformed or required source files are
            unavailable.
    """
    if comparison_path is None:
        return _empty_portfolio_return_reconstruction_checks()

    specification = PerformanceComparisonSpecification(comparison_path)
    reconstruction = specification.portfolio_return_reconstruction
    if reconstruction is None:
        return _empty_portfolio_return_reconstruction_checks()

    engine = _PortfolioReturnReconstructionEngine(specification, reconstruction)
    return engine.checks()


def security_return_reconstruction_checks(
    comparison_path: util.PathLike | None,
) -> pl.DataFrame:
    """Return security return-reconstruction diagnostics for a comparison YAML.

    Args:
        comparison_path: Path to a performance comparison YAML file. ``None``
            returns an empty table because reconstruction needs source data.

    Returns:
        Stable Polars table with one row per portfolio-security-period that can
        be checked against configured security reconstruction rules. Empty when
        the YAML does not opt into ``security_return_reconstruction``.

    Raises:
        PpaError: If opted-in YAML is malformed or required source files are
            unavailable.
    """
    if comparison_path is None:
        return _empty_security_return_reconstruction_checks()

    specification = PerformanceComparisonSpecification(comparison_path)
    reconstruction = specification.security_return_reconstruction
    if reconstruction is None:
        return _empty_security_return_reconstruction_checks()

    engine = _SecurityReturnReconstructionEngine(specification, reconstruction)
    return engine.checks()


def return_reconstruction_summary(
    comparison_path: util.PathLike | None,
) -> pl.DataFrame:
    """Return summary counts for available return-reconstruction diagnostics.

    Args:
        comparison_path: Path to a performance comparison YAML file. ``None``
            returns an empty summary.

    Returns:
        Stable summary table grouped by reconstruction check type, status, and
        diagnostic category. Empty when neither reconstruction check is enabled.
    """
    tables = []
    portfolio_checks = portfolio_return_reconstruction_checks(comparison_path)
    if not portfolio_checks.is_empty():
        tables.append(
            _summary_counts(
                portfolio_checks,
                check_type=RECONSTRUCTION_CHECK_TYPE_PORTFOLIO,
            )
        )
    security_checks = security_return_reconstruction_checks(comparison_path)
    if not security_checks.is_empty():
        tables.append(
            _summary_counts(
                security_checks,
                check_type=RECONSTRUCTION_CHECK_TYPE_SECURITY,
            )
        )
    if not tables:
        return _empty_return_reconstruction_summary()
    return pl.concat(tables).select(RECONSTRUCTION_SUMMARY_COLUMNS)


class _PortfolioReturnReconstructionEngine:
    """Compute portfolio-level Modified Dietz reconstruction diagnostics."""

    def __init__(
        self,
        specification: PerformanceComparisonSpecification,
        reconstruction: PortfolioReturnReconstruction,
    ) -> None:
        self._specification = specification
        self._reconstruction = reconstruction
        self._portfolio_loader = PortfolioPerformanceLoader(specification)
        self._holdings_loader = HoldingsLoader(specification)
        self._transactions_loader = TransactionsLoader(specification)

    def checks(self) -> pl.DataFrame:
        """Return the configured reconstruction check table."""
        portfolio_a = self._portfolio_loader.load("a")
        portfolio_b = self._portfolio_loader.load("b")
        holdings_a = self._required_holdings("a")
        holdings_b = self._required_holdings("b")
        transactions_a = self._required_transactions("a")
        transactions_b = self._required_transactions("b")

        rows = []
        for key in sorted(_portfolio_period_keys(portfolio_a, portfolio_b)):
            period_a = _row_by_key(portfolio_a, key)
            period_b = _row_by_key(portfolio_b, key)
            if period_a is None or period_b is None:
                continue
            inputs_a = self._snapshot_inputs(period_a, holdings_a, transactions_a)
            inputs_b = self._snapshot_inputs(period_b, holdings_b, transactions_b)
            rows.append(_reconstruction_row(key, inputs_a, inputs_b, self._tolerance()))

        if not rows:
            return _empty_portfolio_return_reconstruction_checks()
        return pl.DataFrame(rows).select(PORTFOLIO_RETURN_RECONSTRUCTION_COLUMNS)

    def _required_holdings(self, snapshot_key: str) -> pl.DataFrame:
        """Return loaded holdings or raise a reconstruction-specific error."""
        holdings = self._holdings_loader.load(snapshot_key)  # type: ignore[arg-type]
        if holdings is None:
            raise PpaError(
                f"{self._specification.path}: portfolio_return_reconstruction "
                "requires files.holdings.",
                504,
            )
        return holdings

    def _required_transactions(self, snapshot_key: str) -> pl.DataFrame:
        """Return loaded transactions or raise a reconstruction-specific error."""
        transactions = self._transactions_loader.load(snapshot_key)  # type: ignore[arg-type]
        if transactions is None:
            raise PpaError(
                f"{self._specification.path}: portfolio_return_reconstruction "
                "requires files.transactions.",
                504,
            )
        return transactions

    def _snapshot_inputs(
        self,
        period_row: dict[str, object],
        holdings: pl.DataFrame,
        transactions: pl.DataFrame,
    ) -> _SnapshotReturnInputs:
        """Return reconstructed inputs for one snapshot period row."""
        portfolio_id = str(period_row[pc_cols.PORTFOLIO_ID])
        from_date = _date_value(period_row[pc_cols.FROM_DATE])
        thru_date = _date_value(period_row[pc_cols.THRU_DATE])
        begin_date = _begin_holding_date(holdings, portfolio_id, from_date)
        end_date = _end_holding_date(holdings, portfolio_id, thru_date)
        comments: list[str] = []

        begin_value = None
        if begin_date is None:
            comments.append("missing beginning holdings value")
        else:
            begin_value = _portfolio_holding_value(holdings, portfolio_id, begin_date)

        end_value = None
        if end_date is None:
            comments.append("missing ending holdings value")
        else:
            end_value = _portfolio_holding_value(holdings, portfolio_id, end_date)

        net_flow, weighted_flow = self._portfolio_flows(
            transactions,
            portfolio_id,
            from_date,
            thru_date,
        )

        derived_numerator = None
        derived_denominator = None
        derived_return = None
        if begin_value is not None and end_value is not None:
            derived_numerator = end_value - begin_value - net_flow
            derived_denominator = begin_value + weighted_flow
            if derived_denominator == 0:
                comments.append("zero Modified Dietz denominator")
            else:
                derived_return = derived_numerator / derived_denominator

        return _SnapshotReturnInputs(
            reported_return=_float_or_none(period_row.get(pc_cols.PORTFOLIO_RETURN)),
            derived_return=derived_return,
            derived_numerator=derived_numerator,
            derived_denominator=derived_denominator,
            begin_value=begin_value,
            end_value=end_value,
            net_flow=net_flow,
            weighted_flow=weighted_flow,
            income=0.0,
            begin_value_date=begin_date,
            end_value_date=end_date,
            comments=tuple(comments),
        )

    def _portfolio_flows(
        self,
        transactions: pl.DataFrame,
        portfolio_id: str,
        from_date: dt.date,
        thru_date: dt.date,
    ) -> tuple[float, float]:
        """Return net and weighted portfolio external flows for a period."""
        if transactions.is_empty():
            return 0.0, 0.0

        rows = transactions.filter(
            (pl.col(pc_cols.PORTFOLIO_ID) == portfolio_id)
            & pl.col(pc_cols.TRANSACTION_DATE).is_between(from_date, thru_date)
            & pl.col(pc_cols.TRANSACTION_CATEGORY).is_in(
                list(self._reconstruction.flow_categories)
            )
        )
        net_flow = 0.0
        weighted_flow = 0.0
        for row in rows.iter_rows(named=True):
            amount = _float_or_none(row.get(pc_cols.AMOUNT))
            if amount is None:
                continue
            flow_date = _date_value(row[pc_cols.TRANSACTION_DATE])
            weight = modified_dietz_flow_weight(
                from_date=from_date,
                thru_date=thru_date,
                flow_date=flow_date,
                inclusion_rule=self._reconstruction.inclusion_rule,
            )
            net_flow += amount
            weighted_flow += amount * weight
        return net_flow, weighted_flow

    def _tolerance(self) -> float:
        """Return the configured return tolerance for reconstruction checks."""
        tolerances = self._specification.values.get("tolerances")
        if isinstance(tolerances, dict):
            tolerance = _float_or_none(tolerances.get("return"))
            if tolerance is not None:
                return tolerance
        return 1e-6


class _SecurityReturnReconstructionEngine:
    """Compute security-level Modified Dietz reconstruction diagnostics."""

    def __init__(
        self,
        specification: PerformanceComparisonSpecification,
        reconstruction: SecurityReturnReconstruction,
    ) -> None:
        self._specification = specification
        self._reconstruction = reconstruction
        self._security_loader = SecurityPerformanceLoader(specification)
        self._holdings_loader = HoldingsLoader(specification)
        self._transactions_loader = TransactionsLoader(specification)

    def checks(self) -> pl.DataFrame:
        """Return the configured security reconstruction check table."""
        security_a = self._required_security_performance("a")
        security_b = self._required_security_performance("b")
        holdings_a = self._required_holdings("a")
        holdings_b = self._required_holdings("b")
        transactions_a = self._required_transactions("a")
        transactions_b = self._required_transactions("b")

        rows = []
        for key in sorted(_security_period_keys(security_a, security_b)):
            period_a = _security_row_by_key(security_a, key)
            period_b = _security_row_by_key(security_b, key)
            if period_a is None or period_b is None:
                continue
            inputs_a = self._snapshot_inputs(period_a, holdings_a, transactions_a)
            inputs_b = self._snapshot_inputs(period_b, holdings_b, transactions_b)
            rows.append(
                _security_reconstruction_row(
                    key,
                    inputs_a,
                    inputs_b,
                    self._tolerance(),
                )
            )

        if not rows:
            return _empty_security_return_reconstruction_checks()
        return pl.DataFrame(rows).select(SECURITY_RETURN_RECONSTRUCTION_COLUMNS)

    def _required_security_performance(self, snapshot_key: str) -> pl.DataFrame:
        """Return loaded security performance or raise a reconstruction error."""
        security_performance = self._security_loader.load(snapshot_key)  # type: ignore[arg-type]
        if security_performance is None:
            raise PpaError(
                f"{self._specification.path}: security_return_reconstruction "
                "requires files.security_performance.",
                504,
            )
        return security_performance

    def _required_holdings(self, snapshot_key: str) -> pl.DataFrame:
        """Return loaded holdings or raise a reconstruction-specific error."""
        holdings = self._holdings_loader.load(snapshot_key)  # type: ignore[arg-type]
        if holdings is None:
            raise PpaError(
                f"{self._specification.path}: security_return_reconstruction "
                "requires files.holdings.",
                504,
            )
        return holdings

    def _required_transactions(self, snapshot_key: str) -> pl.DataFrame:
        """Return loaded transactions or raise a reconstruction-specific error."""
        transactions = self._transactions_loader.load(snapshot_key)  # type: ignore[arg-type]
        if transactions is None:
            raise PpaError(
                f"{self._specification.path}: security_return_reconstruction "
                "requires files.transactions.",
                504,
            )
        return transactions

    def _snapshot_inputs(
        self,
        period_row: dict[str, object],
        holdings: pl.DataFrame,
        transactions: pl.DataFrame,
    ) -> _SnapshotReturnInputs:
        """Return reconstructed inputs for one security-period row."""
        portfolio_id = str(period_row[pc_cols.PORTFOLIO_ID])
        security_id = str(period_row[pc_cols.SECURITY_ID])
        from_date = _date_value(period_row[pc_cols.FROM_DATE])
        thru_date = _date_value(period_row[pc_cols.THRU_DATE])
        begin_date = _begin_holding_date(holdings, portfolio_id, from_date)
        end_date = _end_holding_date(holdings, portfolio_id, thru_date)
        comments: list[str] = []

        begin_value = None
        if begin_date is None:
            comments.append("missing beginning holdings value")
        else:
            begin_value = _security_holding_value(
                holdings,
                portfolio_id,
                security_id,
                begin_date,
            )
            if begin_value is None:
                comments.append("missing beginning security holding")

        end_value = None
        if end_date is None:
            comments.append("missing ending holdings value")
        else:
            end_value = _security_holding_value(
                holdings,
                portfolio_id,
                security_id,
                end_date,
            )
            if end_value is None:
                comments.append("missing ending security holding")

        net_flow, weighted_flow = self._security_flows(
            transactions,
            portfolio_id,
            security_id,
            from_date,
            thru_date,
        )
        income = self._security_income(
            transactions,
            portfolio_id,
            security_id,
            from_date,
            thru_date,
        )

        derived_numerator = None
        derived_denominator = None
        derived_return = None
        if begin_value is not None and end_value is not None:
            derived_numerator = end_value - begin_value - net_flow + income
            derived_denominator = begin_value + weighted_flow
            if derived_denominator == 0:
                comments.append("zero Modified Dietz denominator")
            else:
                derived_return = derived_numerator / derived_denominator

        return _SnapshotReturnInputs(
            reported_return=_float_or_none(period_row.get(pc_cols.SECURITY_RETURN)),
            derived_return=derived_return,
            derived_numerator=derived_numerator,
            derived_denominator=derived_denominator,
            begin_value=begin_value,
            end_value=end_value,
            net_flow=net_flow,
            weighted_flow=weighted_flow,
            income=income,
            begin_value_date=begin_date,
            end_value_date=end_date,
            comments=tuple(comments),
        )

    def _security_flows(
        self,
        transactions: pl.DataFrame,
        portfolio_id: str,
        security_id: str,
        from_date: dt.date,
        thru_date: dt.date,
    ) -> tuple[float, float]:
        """Return net and weighted security-level buy/sell flows for a period."""
        if transactions.is_empty():
            return 0.0, 0.0

        rows = transactions.filter(
            (pl.col(pc_cols.PORTFOLIO_ID) == portfolio_id)
            & (pl.col(pc_cols.SECURITY_ID) == security_id)
            & pl.col(pc_cols.TRANSACTION_DATE).is_between(from_date, thru_date)
            & pl.col(pc_cols.TRANSACTION_CATEGORY).is_in(
                list(self._reconstruction.flow_categories)
            )
        )
        net_flow = 0.0
        weighted_flow = 0.0
        for row in rows.iter_rows(named=True):
            amount = _float_or_none(row.get(pc_cols.AMOUNT))
            if amount is None:
                continue
            security_flow = -amount
            flow_date = _date_value(row[pc_cols.TRANSACTION_DATE])
            weight = modified_dietz_flow_weight(
                from_date=from_date,
                thru_date=thru_date,
                flow_date=flow_date,
                inclusion_rule=self._reconstruction.inclusion_rule,
            )
            net_flow += security_flow
            weighted_flow += security_flow * weight
        return net_flow, weighted_flow

    def _security_income(
        self,
        transactions: pl.DataFrame,
        portfolio_id: str,
        security_id: str,
        from_date: dt.date,
        thru_date: dt.date,
    ) -> float:
        """Return performance income transactions for one security-period."""
        if transactions.is_empty():
            return 0.0

        rows = transactions.filter(
            (pl.col(pc_cols.PORTFOLIO_ID) == portfolio_id)
            & (pl.col(pc_cols.SECURITY_ID) == security_id)
            & pl.col(pc_cols.TRANSACTION_DATE).is_between(from_date, thru_date)
            & pl.col(pc_cols.TRANSACTION_CATEGORY).is_in(
                list(self._reconstruction.income_categories)
            )
        )
        income = 0.0
        for row in rows.iter_rows(named=True):
            amount = _float_or_none(row.get(pc_cols.AMOUNT))
            if amount is not None:
                income += amount
        return income

    def _tolerance(self) -> float:
        """Return the configured return tolerance for reconstruction checks."""
        tolerances = self._specification.values.get("tolerances")
        if isinstance(tolerances, dict):
            tolerance = _float_or_none(tolerances.get("return"))
            if tolerance is not None:
                return tolerance
        return 1e-6


def _portfolio_period_keys(
    portfolio_a: pl.DataFrame,
    portfolio_b: pl.DataFrame,
) -> set[tuple[str, dt.date, dt.date]]:
    """Return portfolio-period keys present in either snapshot."""
    keys: set[tuple[str, dt.date, dt.date]] = set()
    for frame in (portfolio_a, portfolio_b):
        for row in frame.iter_rows(named=True):
            keys.add(
                (
                    str(row[pc_cols.PORTFOLIO_ID]),
                    _date_value(row[pc_cols.FROM_DATE]),
                    _date_value(row[pc_cols.THRU_DATE]),
                )
            )
    return keys


def _security_period_keys(
    security_a: pl.DataFrame,
    security_b: pl.DataFrame,
) -> set[tuple[str, str, dt.date, dt.date]]:
    """Return portfolio-security-period keys present in either snapshot."""
    keys: set[tuple[str, str, dt.date, dt.date]] = set()
    for frame in (security_a, security_b):
        for row in frame.iter_rows(named=True):
            keys.add(
                (
                    str(row[pc_cols.PORTFOLIO_ID]),
                    str(row[pc_cols.SECURITY_ID]),
                    _date_value(row[pc_cols.FROM_DATE]),
                    _date_value(row[pc_cols.THRU_DATE]),
                )
            )
    return keys


def _row_by_key(
    frame: pl.DataFrame,
    key: tuple[str, dt.date, dt.date],
) -> dict[str, object] | None:
    """Return a portfolio-period row by key, or ``None`` when absent."""
    portfolio_id, from_date, thru_date = key
    rows = frame.filter(
        (pl.col(pc_cols.PORTFOLIO_ID) == portfolio_id)
        & (pl.col(pc_cols.FROM_DATE) == from_date)
        & (pl.col(pc_cols.THRU_DATE) == thru_date)
    )
    if rows.is_empty():
        return None
    return rows.row(0, named=True)


def _security_row_by_key(
    frame: pl.DataFrame,
    key: tuple[str, str, dt.date, dt.date],
) -> dict[str, object] | None:
    """Return a security-period row by key, or ``None`` when absent."""
    portfolio_id, security_id, from_date, thru_date = key
    rows = frame.filter(
        (pl.col(pc_cols.PORTFOLIO_ID) == portfolio_id)
        & (pl.col(pc_cols.SECURITY_ID) == security_id)
        & (pl.col(pc_cols.FROM_DATE) == from_date)
        & (pl.col(pc_cols.THRU_DATE) == thru_date)
    )
    if rows.is_empty():
        return None
    return rows.row(0, named=True)


def _begin_holding_date(
    holdings: pl.DataFrame,
    portfolio_id: str,
    from_date: dt.date,
) -> dt.date | None:
    """Return the latest available holding date before a period starts."""
    rows = holdings.filter(
        (pl.col(pc_cols.PORTFOLIO_ID) == portfolio_id)
        & (pl.col(pc_cols.HOLDING_DATE) < from_date)
    )
    if rows.is_empty():
        return None
    return _date_value(rows.select(pl.max(pc_cols.HOLDING_DATE)).item())


def _end_holding_date(
    holdings: pl.DataFrame,
    portfolio_id: str,
    thru_date: dt.date,
) -> dt.date | None:
    """Return the holding date used as ending value for a period."""
    rows = holdings.filter(
        (pl.col(pc_cols.PORTFOLIO_ID) == portfolio_id)
        & (pl.col(pc_cols.HOLDING_DATE) == thru_date)
    )
    if rows.is_empty():
        return None
    return thru_date


def _portfolio_holding_value(
    holdings: pl.DataFrame,
    portfolio_id: str,
    holding_date: dt.date,
) -> float:
    """Return portfolio holding value including accrued interest."""
    rows = holdings.filter(
        (pl.col(pc_cols.PORTFOLIO_ID) == portfolio_id)
        & (pl.col(pc_cols.HOLDING_DATE) == holding_date)
    )
    value_expression = pl.col(pc_cols.MARKET_VALUE).cast(pl.Float64).fill_null(0.0)
    if pc_cols.ACCRUED in rows.columns:
        value_expression = value_expression + pl.col(pc_cols.ACCRUED).cast(
            pl.Float64
        ).fill_null(0.0)
    return float(rows.select(value_expression.sum()).item())


def _security_holding_value(
    holdings: pl.DataFrame,
    portfolio_id: str,
    security_id: str,
    holding_date: dt.date,
) -> float | None:
    """Return one security holding value including accrued interest."""
    rows = holdings.filter(
        (pl.col(pc_cols.PORTFOLIO_ID) == portfolio_id)
        & (pl.col(pc_cols.SECURITY_ID) == security_id)
        & (pl.col(pc_cols.HOLDING_DATE) == holding_date)
    )
    if rows.is_empty():
        return None
    value_expression = pl.col(pc_cols.MARKET_VALUE).cast(pl.Float64).fill_null(0.0)
    if pc_cols.ACCRUED in rows.columns:
        value_expression = value_expression + pl.col(pc_cols.ACCRUED).cast(
            pl.Float64
        ).fill_null(0.0)
    return float(rows.select(value_expression.sum()).item())


def _reconstruction_row(
    key: tuple[str, dt.date, dt.date],
    inputs_a: _SnapshotReturnInputs,
    inputs_b: _SnapshotReturnInputs,
    tolerance: float,
) -> dict[str, object]:
    """Return one portfolio-period reconstruction check row."""
    portfolio_id, from_date, thru_date = key
    reported_difference = _difference(inputs_a.reported_return, inputs_b.reported_return)
    derived_difference = _difference(inputs_a.derived_return, inputs_b.derived_return)
    reconstruction_difference = _difference(derived_difference, reported_difference)
    begin_value_difference = _difference(inputs_a.begin_value, inputs_b.begin_value)
    end_value_difference = _difference(inputs_a.end_value, inputs_b.end_value)
    net_flow_difference = _difference(inputs_a.net_flow, inputs_b.net_flow)
    weighted_flow_difference = _difference(
        inputs_a.weighted_flow,
        inputs_b.weighted_flow,
    )
    derived_numerator_difference = _difference(
        inputs_a.derived_numerator,
        inputs_b.derived_numerator,
    )
    derived_denominator_difference = _difference(
        inputs_a.derived_denominator,
        inputs_b.derived_denominator,
    )
    comments = [*inputs_a.comments, *inputs_b.comments]
    status = _reconstruction_status(reconstruction_difference, comments, tolerance)
    component_changes = _component_change_comments(
        begin_value_difference=begin_value_difference,
        end_value_difference=end_value_difference,
        net_flow_difference=net_flow_difference,
        weighted_flow_difference=weighted_flow_difference,
        derived_numerator_difference=derived_numerator_difference,
        derived_denominator_difference=derived_denominator_difference,
    )
    category = _reconstruction_category(status, comments, component_changes)
    return {
        RECONSTRUCTION_REVIEW_KEY: f"{portfolio_id}::{from_date}::{thru_date}",
        RECONSTRUCTION_PORTFOLIO_ID: portfolio_id,
        RECONSTRUCTION_FROM_DATE: from_date,
        RECONSTRUCTION_THRU_DATE: thru_date,
        REPORTED_RETURN_A: inputs_a.reported_return,
        REPORTED_RETURN_B: inputs_b.reported_return,
        REPORTED_RETURN_DIFFERENCE: reported_difference,
        DERIVED_RETURN_A: inputs_a.derived_return,
        DERIVED_RETURN_B: inputs_b.derived_return,
        DERIVED_RETURN_DIFFERENCE: derived_difference,
        RECONSTRUCTION_DIFFERENCE: reconstruction_difference,
        DERIVED_NUMERATOR_A: inputs_a.derived_numerator,
        DERIVED_NUMERATOR_B: inputs_b.derived_numerator,
        DERIVED_NUMERATOR_DIFFERENCE: derived_numerator_difference,
        DERIVED_DENOMINATOR_A: inputs_a.derived_denominator,
        DERIVED_DENOMINATOR_B: inputs_b.derived_denominator,
        DERIVED_DENOMINATOR_DIFFERENCE: derived_denominator_difference,
        BEGIN_VALUE_A: inputs_a.begin_value,
        BEGIN_VALUE_B: inputs_b.begin_value,
        BEGIN_VALUE_DIFFERENCE: begin_value_difference,
        END_VALUE_A: inputs_a.end_value,
        END_VALUE_B: inputs_b.end_value,
        END_VALUE_DIFFERENCE: end_value_difference,
        NET_FLOW_A: inputs_a.net_flow,
        NET_FLOW_B: inputs_b.net_flow,
        NET_FLOW_DIFFERENCE: net_flow_difference,
        WEIGHTED_FLOW_A: inputs_a.weighted_flow,
        WEIGHTED_FLOW_B: inputs_b.weighted_flow,
        WEIGHTED_FLOW_DIFFERENCE: weighted_flow_difference,
        BEGIN_VALUE_DATE_A: inputs_a.begin_value_date,
        BEGIN_VALUE_DATE_B: inputs_b.begin_value_date,
        END_VALUE_DATE_A: inputs_a.end_value_date,
        END_VALUE_DATE_B: inputs_b.end_value_date,
        RECONSTRUCTION_STATUS: status,
        RECONSTRUCTION_CATEGORY: category,
        RECONSTRUCTION_COMMENTS: _reconstruction_comments(
            status,
            comments,
            reconstruction_difference,
            component_changes=component_changes,
        ),
    }


def _security_reconstruction_row(
    key: tuple[str, str, dt.date, dt.date],
    inputs_a: _SnapshotReturnInputs,
    inputs_b: _SnapshotReturnInputs,
    tolerance: float,
) -> dict[str, object]:
    """Return one security-period reconstruction check row."""
    portfolio_id, security_id, from_date, thru_date = key
    reported_difference = _difference(inputs_a.reported_return, inputs_b.reported_return)
    derived_difference = _difference(inputs_a.derived_return, inputs_b.derived_return)
    reconstruction_difference = _difference(derived_difference, reported_difference)
    begin_value_difference = _difference(inputs_a.begin_value, inputs_b.begin_value)
    end_value_difference = _difference(inputs_a.end_value, inputs_b.end_value)
    net_flow_difference = _difference(inputs_a.net_flow, inputs_b.net_flow)
    weighted_flow_difference = _difference(
        inputs_a.weighted_flow,
        inputs_b.weighted_flow,
    )
    income_difference = _difference(inputs_a.income, inputs_b.income)
    derived_numerator_difference = _difference(
        inputs_a.derived_numerator,
        inputs_b.derived_numerator,
    )
    derived_denominator_difference = _difference(
        inputs_a.derived_denominator,
        inputs_b.derived_denominator,
    )
    comments = [*inputs_a.comments, *inputs_b.comments]
    status = _reconstruction_status(reconstruction_difference, comments, tolerance)
    component_changes = _component_change_comments(
        begin_value_difference=begin_value_difference,
        end_value_difference=end_value_difference,
        net_flow_difference=net_flow_difference,
        weighted_flow_difference=weighted_flow_difference,
        income_difference=income_difference,
        derived_numerator_difference=derived_numerator_difference,
        derived_denominator_difference=derived_denominator_difference,
    )
    category = _reconstruction_category(status, comments, component_changes)
    return {
        RECONSTRUCTION_REVIEW_KEY: (
            f"{portfolio_id}::{from_date}::{thru_date}::{security_id}"
        ),
        RECONSTRUCTION_PORTFOLIO_ID: portfolio_id,
        RECONSTRUCTION_SECURITY_ID: security_id,
        RECONSTRUCTION_FROM_DATE: from_date,
        RECONSTRUCTION_THRU_DATE: thru_date,
        REPORTED_RETURN_A: inputs_a.reported_return,
        REPORTED_RETURN_B: inputs_b.reported_return,
        REPORTED_RETURN_DIFFERENCE: reported_difference,
        DERIVED_RETURN_A: inputs_a.derived_return,
        DERIVED_RETURN_B: inputs_b.derived_return,
        DERIVED_RETURN_DIFFERENCE: derived_difference,
        RECONSTRUCTION_DIFFERENCE: reconstruction_difference,
        DERIVED_NUMERATOR_A: inputs_a.derived_numerator,
        DERIVED_NUMERATOR_B: inputs_b.derived_numerator,
        DERIVED_NUMERATOR_DIFFERENCE: derived_numerator_difference,
        DERIVED_DENOMINATOR_A: inputs_a.derived_denominator,
        DERIVED_DENOMINATOR_B: inputs_b.derived_denominator,
        DERIVED_DENOMINATOR_DIFFERENCE: derived_denominator_difference,
        BEGIN_VALUE_A: inputs_a.begin_value,
        BEGIN_VALUE_B: inputs_b.begin_value,
        BEGIN_VALUE_DIFFERENCE: begin_value_difference,
        END_VALUE_A: inputs_a.end_value,
        END_VALUE_B: inputs_b.end_value,
        END_VALUE_DIFFERENCE: end_value_difference,
        NET_FLOW_A: inputs_a.net_flow,
        NET_FLOW_B: inputs_b.net_flow,
        NET_FLOW_DIFFERENCE: net_flow_difference,
        WEIGHTED_FLOW_A: inputs_a.weighted_flow,
        WEIGHTED_FLOW_B: inputs_b.weighted_flow,
        WEIGHTED_FLOW_DIFFERENCE: weighted_flow_difference,
        INCOME_A: inputs_a.income,
        INCOME_B: inputs_b.income,
        INCOME_DIFFERENCE: income_difference,
        BEGIN_VALUE_DATE_A: inputs_a.begin_value_date,
        BEGIN_VALUE_DATE_B: inputs_b.begin_value_date,
        END_VALUE_DATE_A: inputs_a.end_value_date,
        END_VALUE_DATE_B: inputs_b.end_value_date,
        RECONSTRUCTION_STATUS: status,
        RECONSTRUCTION_CATEGORY: category,
        RECONSTRUCTION_COMMENTS: _reconstruction_comments(
            status,
            comments,
            reconstruction_difference,
            component_changes=component_changes,
        ),
    }


def _reconstruction_status(
    reconstruction_difference: float | None,
    comments: list[str],
    tolerance: float,
) -> str:
    """Return review status for one reconstruction check row."""
    if comments or reconstruction_difference is None:
        return RECONSTRUCTION_STATUS_MISSING_INPUTS
    if abs(reconstruction_difference) <= tolerance:
        return RECONSTRUCTION_STATUS_ALIGNED
    return RECONSTRUCTION_STATUS_DIFFERENT


def _reconstruction_comments(
    status: str,
    comments: list[str],
    reconstruction_difference: float | None,
    *,
    component_changes: tuple[str, ...],
) -> str:
    """Return reviewer-facing reconstruction comments."""
    if comments:
        return f"Missing reconstruction inputs: {'; '.join(dict.fromkeys(comments))}."
    if status == RECONSTRUCTION_STATUS_ALIGNED:
        if not component_changes:
            return "Reported and derived return differences agree."
        return (
            "Reported and derived return differences agree. Changed "
            f"reconstruction inputs: {', '.join(component_changes)}."
        )
    if reconstruction_difference is None:
        return "Derived or reported return difference is unavailable."
    if component_changes:
        return (
            "Reported and derived return differences do not agree. Changed "
            f"reconstruction inputs: {', '.join(component_changes)}. Review "
            "vendor methodology or missing reconstruction inputs if unexpected."
        )
    return (
        "Reported and derived return differences do not agree; no changed "
        "reconstruction formula input was isolated."
    )


def _reconstruction_category(
    status: str,
    comments: list[str],
    component_changes: tuple[str, ...],
) -> str:
    """Return a compact diagnostic category for one reconstruction row."""
    if comments or status == RECONSTRUCTION_STATUS_MISSING_INPUTS:
        return RECONSTRUCTION_CATEGORY_MISSING_INPUTS
    if status == RECONSTRUCTION_STATUS_ALIGNED:
        return RECONSTRUCTION_CATEGORY_ALIGNED
    if component_changes:
        return RECONSTRUCTION_CATEGORY_SOURCE_INPUTS_CHANGED
    return RECONSTRUCTION_CATEGORY_FORMULA_DIFFERENCE


def _summary_counts(
    checks: pl.DataFrame,
    *,
    check_type: str,
) -> pl.DataFrame:
    """Return reconstruction summary counts for one check table."""
    if checks.is_empty():
        return _empty_return_reconstruction_summary()
    return (
        checks.group_by(RECONSTRUCTION_STATUS, RECONSTRUCTION_CATEGORY)
        .len(RECONSTRUCTION_ROW_COUNT)
        .with_columns(pl.lit(check_type).alias(RECONSTRUCTION_CHECK_TYPE))
        .sort(
            RECONSTRUCTION_CHECK_TYPE,
            RECONSTRUCTION_STATUS,
            RECONSTRUCTION_CATEGORY,
        )
        .select(RECONSTRUCTION_SUMMARY_COLUMNS)
    )


def _component_change_comments(
    *,
    begin_value_difference: float | None,
    end_value_difference: float | None,
    net_flow_difference: float | None,
    weighted_flow_difference: float | None,
    derived_numerator_difference: float | None,
    derived_denominator_difference: float | None,
    income_difference: float | None = None,
) -> tuple[str, ...]:
    """Return concise names for changed reconstruction formula components."""
    component_values = (
        ("beginning value", begin_value_difference),
        ("ending value", end_value_difference),
        ("net flow", net_flow_difference),
        ("weighted flow", weighted_flow_difference),
        ("income", income_difference),
        ("derived numerator", derived_numerator_difference),
        ("derived denominator", derived_denominator_difference),
    )
    return tuple(
        label
        for label, value in component_values
        if value is not None and abs(value) > 0.0000005
    )


def _difference(value_a: float | None, value_b: float | None) -> float | None:
    """Return ``value_b - value_a`` when both values are available."""
    if value_a is None or value_b is None:
        return None
    return value_b - value_a


def _date_value(value: object) -> dt.date:
    """Return a date object for Polars/Python date values."""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    raise TypeError(f"Expected date value, got {type(value)!r}.")


def _float_or_none(value: object) -> float | None:
    """Return a float for non-boolean numeric values."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _empty_portfolio_return_reconstruction_checks() -> pl.DataFrame:
    """Return an empty portfolio return-reconstruction check table."""
    return pl.DataFrame(
        schema={
            RECONSTRUCTION_REVIEW_KEY: pl.String,
            RECONSTRUCTION_PORTFOLIO_ID: pl.String,
            RECONSTRUCTION_FROM_DATE: pl.Date,
            RECONSTRUCTION_THRU_DATE: pl.Date,
            REPORTED_RETURN_A: pl.Float64,
            REPORTED_RETURN_B: pl.Float64,
            REPORTED_RETURN_DIFFERENCE: pl.Float64,
            DERIVED_RETURN_A: pl.Float64,
            DERIVED_RETURN_B: pl.Float64,
            DERIVED_RETURN_DIFFERENCE: pl.Float64,
            RECONSTRUCTION_DIFFERENCE: pl.Float64,
            DERIVED_NUMERATOR_A: pl.Float64,
            DERIVED_NUMERATOR_B: pl.Float64,
            DERIVED_NUMERATOR_DIFFERENCE: pl.Float64,
            DERIVED_DENOMINATOR_A: pl.Float64,
            DERIVED_DENOMINATOR_B: pl.Float64,
            DERIVED_DENOMINATOR_DIFFERENCE: pl.Float64,
            BEGIN_VALUE_A: pl.Float64,
            BEGIN_VALUE_B: pl.Float64,
            BEGIN_VALUE_DIFFERENCE: pl.Float64,
            END_VALUE_A: pl.Float64,
            END_VALUE_B: pl.Float64,
            END_VALUE_DIFFERENCE: pl.Float64,
            NET_FLOW_A: pl.Float64,
            NET_FLOW_B: pl.Float64,
            NET_FLOW_DIFFERENCE: pl.Float64,
            WEIGHTED_FLOW_A: pl.Float64,
            WEIGHTED_FLOW_B: pl.Float64,
            WEIGHTED_FLOW_DIFFERENCE: pl.Float64,
            BEGIN_VALUE_DATE_A: pl.Date,
            BEGIN_VALUE_DATE_B: pl.Date,
            END_VALUE_DATE_A: pl.Date,
            END_VALUE_DATE_B: pl.Date,
            RECONSTRUCTION_STATUS: pl.String,
            RECONSTRUCTION_CATEGORY: pl.String,
            RECONSTRUCTION_COMMENTS: pl.String,
        }
    )


def _empty_security_return_reconstruction_checks() -> pl.DataFrame:
    """Return an empty security return-reconstruction check table."""
    return pl.DataFrame(
        schema={
            RECONSTRUCTION_REVIEW_KEY: pl.String,
            RECONSTRUCTION_PORTFOLIO_ID: pl.String,
            RECONSTRUCTION_SECURITY_ID: pl.String,
            RECONSTRUCTION_FROM_DATE: pl.Date,
            RECONSTRUCTION_THRU_DATE: pl.Date,
            REPORTED_RETURN_A: pl.Float64,
            REPORTED_RETURN_B: pl.Float64,
            REPORTED_RETURN_DIFFERENCE: pl.Float64,
            DERIVED_RETURN_A: pl.Float64,
            DERIVED_RETURN_B: pl.Float64,
            DERIVED_RETURN_DIFFERENCE: pl.Float64,
            RECONSTRUCTION_DIFFERENCE: pl.Float64,
            DERIVED_NUMERATOR_A: pl.Float64,
            DERIVED_NUMERATOR_B: pl.Float64,
            DERIVED_NUMERATOR_DIFFERENCE: pl.Float64,
            DERIVED_DENOMINATOR_A: pl.Float64,
            DERIVED_DENOMINATOR_B: pl.Float64,
            DERIVED_DENOMINATOR_DIFFERENCE: pl.Float64,
            BEGIN_VALUE_A: pl.Float64,
            BEGIN_VALUE_B: pl.Float64,
            BEGIN_VALUE_DIFFERENCE: pl.Float64,
            END_VALUE_A: pl.Float64,
            END_VALUE_B: pl.Float64,
            END_VALUE_DIFFERENCE: pl.Float64,
            NET_FLOW_A: pl.Float64,
            NET_FLOW_B: pl.Float64,
            NET_FLOW_DIFFERENCE: pl.Float64,
            WEIGHTED_FLOW_A: pl.Float64,
            WEIGHTED_FLOW_B: pl.Float64,
            WEIGHTED_FLOW_DIFFERENCE: pl.Float64,
            INCOME_A: pl.Float64,
            INCOME_B: pl.Float64,
            INCOME_DIFFERENCE: pl.Float64,
            BEGIN_VALUE_DATE_A: pl.Date,
            BEGIN_VALUE_DATE_B: pl.Date,
            END_VALUE_DATE_A: pl.Date,
            END_VALUE_DATE_B: pl.Date,
            RECONSTRUCTION_STATUS: pl.String,
            RECONSTRUCTION_CATEGORY: pl.String,
            RECONSTRUCTION_COMMENTS: pl.String,
        }
    )


def _empty_return_reconstruction_summary() -> pl.DataFrame:
    """Return an empty return-reconstruction summary table."""
    return pl.DataFrame(
        schema={
            RECONSTRUCTION_CHECK_TYPE: pl.String,
            RECONSTRUCTION_STATUS: pl.String,
            RECONSTRUCTION_CATEGORY: pl.String,
            RECONSTRUCTION_ROW_COUNT: pl.UInt32,
        }
    )
