"""Compare normalized performance snapshot datasets."""

from __future__ import annotations

# Python imports
import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, cast

# Third-party imports
import polars as pl

# Project imports
from ppar.errors import PpaError
from ppar.performance_comparison import columns as pc_cols
from ppar.performance_comparison.cash import CashLoader
from ppar.performance_comparison.findings import (
    CONFIDENCE_HIGH,
    CONTEXT,
    DIRECT_INPUT,
    IMPACT_POLICY_PORTFOLIO_SOURCE_FIELD,
    IMPACT_POLICY_SECURITY_CONTRIBUTION,
    IMPACT_POLICY_SECURITY_RETURN_WEIGHTED,
    RELATED_OUTPUT,
    TARGET_OUTPUT,
    TRANSACTION_IMPACT_POLICY_EXTERNAL_FLOW_EVIDENCE_ONLY,
    TRANSACTION_IMPACT_POLICY_PERFORMANCE_AMOUNT_DELTA,
    TRANSACTION_MATCH_STATUS_ID_MATCH,
    TRANSACTION_MATCH_STATUS_ID_UNMATCHED,
    TRANSACTION_MATCH_STATUS_STRICT_FALLBACK_UNMATCHED,
    PC_CASH_MV,
    PC_FX_RATE,
    PC_PORT_FLOW,
    PC_PORT_MV,
    PC_PORT_RET,
    PC_ROW_ADD,
    PC_ROW_DROP,
    PC_SEC_ADD,
    PC_SEC_CONTR,
    PC_SEC_DROP,
    PC_SEC_RET,
    PC_SEC_WGT,
    PC_TXN_ADD,
    PC_TXN_AMT,
    PC_TXN_DROP,
    PC_TXN_PRICE,
    PC_TXN_QTY,
    PC_REF_CLASS,
    PC_REF_ID,
    PC_POS_MV,
    PC_POS_QTY,
    PC_POS_ACCR,
    PC_PRICE,
    SEVERITY_INFORMATIONAL,
    SEVERITY_MATERIAL,
    Finding,
)
from ppar.performance_comparison.fx_rates import FxRatesLoader
from ppar.performance_comparison.period_linking import (
    period_context_for_dated_evidence,
    portfolio_periods_from_snapshots,
    security_period_contexts_for_dated_evidence,
    security_periods_from_snapshots,
)
from ppar.performance_comparison.portfolio_performance import PortfolioPerformanceLoader
from ppar.performance_comparison.positions import PositionsLoader
from ppar.performance_comparison.prices import PricesLoader
from ppar.performance_comparison.rules import apply_suppressions
from ppar.performance_comparison.security_performance import SecurityPerformanceLoader
from ppar.performance_comparison.security_master import SecurityMasterLoader
from ppar.performance_comparison.specification import PerformanceComparisonSpecification
from ppar.performance_comparison.transactions import (
    TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL,
    TRANSACTION_PERFORMANCE_FLOW_SIGN_PERFORMANCE,
    TransactionsLoader,
)

_PORTFOLIO_KEY_COLUMNS: Final[tuple[str, str, str]] = (
    pc_cols.PORTFOLIO_ID,
    pc_cols.FROM_DATE,
    pc_cols.THRU_DATE,
)
_SECURITY_KEY_COLUMNS: Final[tuple[str, str, str, str]] = (
    pc_cols.PORTFOLIO_ID,
    pc_cols.SECURITY_ID,
    pc_cols.FROM_DATE,
    pc_cols.THRU_DATE,
)
_SECURITY_MASTER_KEY_COLUMNS: Final[tuple[str]] = (pc_cols.SECURITY_ID,)
_POSITIONS_KEY_COLUMNS: Final[tuple[str, str, str]] = (
    pc_cols.PORTFOLIO_ID,
    pc_cols.SECURITY_ID,
    pc_cols.POSITION_DATE,
)
_CASH_KEY_COLUMNS: Final[tuple[str, str, str]] = (
    pc_cols.PORTFOLIO_ID,
    pc_cols.CASH_DATE,
    pc_cols.CURRENCY,
)
_PRICE_KEY_COLUMNS: Final[tuple[str, str, str, str]] = (
    pc_cols.SECURITY_ID,
    pc_cols.PRICE_DATE,
    pc_cols.CURRENCY,
    pc_cols.PRICE_SOURCE,
)
_FX_RATE_KEY_COLUMNS: Final[tuple[str, str, str, str, str]] = (
    pc_cols.FROM_CURRENCY,
    pc_cols.TO_CURRENCY,
    pc_cols.RATE_DATE,
    pc_cols.RATE_SOURCE,
    pc_cols.RATE_TYPE,
)
_TRANSACTION_ID_KEY_COLUMNS: Final[tuple[str]] = (pc_cols.TRANSACTION_ID,)
_TRANSACTION_FALLBACK_KEY_COLUMNS: Final[tuple[str, ...]] = (
    pc_cols.PORTFOLIO_ID,
    pc_cols.SECURITY_ID,
    pc_cols.TRANSACTION_DATE,
    pc_cols.SETTLEMENT_DATE,
    pc_cols.TRANSACTION_CODE,
    pc_cols.QUANTITY,
    pc_cols.PRICE,
    pc_cols.AMOUNT,
)
_PORTFOLIO_COMPARE_COLUMNS: Final[dict[str, str]] = {
    pc_cols.PORTFOLIO_RETURN: PC_PORT_RET,
    pc_cols.BEGIN_MARKET_VALUE: PC_PORT_MV,
    pc_cols.END_MARKET_VALUE: PC_PORT_MV,
    pc_cols.FLOW: PC_PORT_FLOW,
    pc_cols.INCOME: PC_PORT_FLOW,
    pc_cols.GAIN_LOSS: PC_PORT_FLOW,
}
_SECURITY_COMPARE_COLUMNS: Final[dict[str, str]] = {
    pc_cols.SECURITY_RETURN: PC_SEC_RET,
    pc_cols.WEIGHT: PC_SEC_WGT,
    pc_cols.CONTRIBUTION: PC_SEC_CONTR,
}
_SECURITY_MASTER_COMPARE_COLUMNS: Final[dict[str, str]] = {
    pc_cols.SECURITY_NAME: PC_REF_ID,
    pc_cols.TICKER: PC_REF_ID,
    pc_cols.CUSIP: PC_REF_ID,
    pc_cols.ISIN: PC_REF_ID,
    pc_cols.CURRENCY: PC_REF_ID,
    pc_cols.COUNTRY: PC_REF_CLASS,
    pc_cols.SECTOR: PC_REF_CLASS,
    pc_cols.INDUSTRY: PC_REF_CLASS,
    pc_cols.ASSET_CLASS: PC_REF_CLASS,
}
_POSITIONS_COMPARE_COLUMNS: Final[dict[str, str]] = {
    pc_cols.QUANTITY: PC_POS_QTY,
    pc_cols.MARKET_VALUE: PC_POS_MV,
    pc_cols.ACCRUED: PC_POS_ACCR,
}
_CASH_COMPARE_COLUMNS: Final[dict[str, str]] = {
    pc_cols.CASH_BALANCE: PC_CASH_MV,
    pc_cols.MARKET_VALUE: PC_CASH_MV,
}
_PRICE_COMPARE_COLUMNS: Final[dict[str, str]] = {
    pc_cols.PRICE: PC_PRICE,
}
_FX_RATE_COMPARE_COLUMNS: Final[dict[str, str]] = {
    pc_cols.FX_RATE: PC_FX_RATE,
}
_TRANSACTION_COMPARE_COLUMNS: Final[dict[str, str]] = {
    pc_cols.AMOUNT: PC_TXN_AMT,
    pc_cols.QUANTITY: PC_TXN_QTY,
    pc_cols.PRICE: PC_TXN_PRICE,
}
_DIRECT_INPUT_DATASETS: Final[frozenset[str]] = frozenset(
    {
        pc_cols.PRICES,
        pc_cols.FX_RATES,
        pc_cols.TRANSACTIONS,
        pc_cols.POSITIONS,
        pc_cols.CASH,
    }
)
_TRANSACTION_IMPACT_METHODS_KEY: Final[str] = "transaction_impact_methods"
_CONTRIBUTION_IMPACT_METHODS_KEY: Final[str] = "contribution_impact_methods"
_PORTFOLIO_SOURCE_FIELD_KEY: Final[str] = "portfolio_source_field"
_SECURITY_CONTRIBUTION_KEY: Final[str] = "security_contribution"
_SECURITY_RETURN_KEY: Final[str] = "security_return"
_EXTERNAL_FLOW_KEY: Final[str] = "external_flow"
_PERFORMANCE_KEY: Final[str] = "performance"
_METHOD_KEY: Final[str] = "method"
_SOURCE_FIELDS_KEY: Final[str] = "source_fields"
_WEIGHT_SOURCE_KEY: Final[str] = "weight_source"
_EVIDENCE_ONLY_METHOD: Final[str] = "evidence_only"
_VENDOR_CONTRIBUTION_DELTA_METHOD: Final[str] = "vendor_contribution_delta"
_SECURITY_RETURN_DELTA_TIMES_WEIGHT_METHOD: Final[str] = (
    "security_return_delta_times_weight"
)
_SOURCE_FIELD_DELTA_OVER_BEGIN_MV_METHOD: Final[str] = (
    "source_field_delta_over_begin_market_value"
)
_MODIFIED_DIETZ_METHOD: Final[str] = "modified_dietz"
_TRANSACTION_AMOUNT_DELTA_METHOD: Final[str] = (
    "transaction_amount_delta_over_return_denominator"
)
_FLOW_TIMING_KEY: Final[str] = "flow_timing"
_DAY_COUNT_KEY: Final[str] = "day_count"
_INCLUSION_RULE_KEY: Final[str] = "inclusion_rule"
_DENOMINATOR_SOURCE_KEY: Final[str] = "denominator_source"
_DOUBLE_COUNT_POLICY_KEY: Final[str] = "double_count_policy"
_MODIFIED_DIETZ_REQUIRED_KEYS: Final[frozenset[str]] = frozenset(
    {
        _METHOD_KEY,
        _FLOW_TIMING_KEY,
        _DAY_COUNT_KEY,
        _INCLUSION_RULE_KEY,
        _DENOMINATOR_SOURCE_KEY,
        _DOUBLE_COUNT_POLICY_KEY,
    }
)
_PERFORMANCE_AMOUNT_REQUIRED_KEYS: Final[frozenset[str]] = frozenset(
    {
        _METHOD_KEY,
        _DENOMINATOR_SOURCE_KEY,
    }
)
_MODIFIED_DIETZ_ALLOWED_VALUES: Final[dict[str, frozenset[str]]] = {
    _FLOW_TIMING_KEY: frozenset({"trade_date", "settlement_date"}),
    _DAY_COUNT_KEY: frozenset({"actual_days"}),
    _INCLUSION_RULE_KEY: frozenset({"beginning_of_day", "end_of_day"}),
    _DENOMINATOR_SOURCE_KEY: frozenset({"begin_market_value"}),
    _DOUBLE_COUNT_POLICY_KEY: frozenset({"cross_check_only"}),
}
_RESERVED_EXTERNAL_FLOW_METHODS: Final[frozenset[str]] = frozenset(
    {
        _MODIFIED_DIETZ_METHOD,
        "subperiod_linked",
        "unweighted_flow_delta",
    }
)
_PERFORMANCE_AMOUNT_ALLOWED_VALUES: Final[dict[str, frozenset[str]]] = {
    _DENOMINATOR_SOURCE_KEY: frozenset({"begin_market_value"}),
}
_PORTFOLIO_SOURCE_FIELD_REQUIRED_KEYS: Final[frozenset[str]] = frozenset(
    {
        _METHOD_KEY,
        _DENOMINATOR_SOURCE_KEY,
        _SOURCE_FIELDS_KEY,
    }
)
_SECURITY_CONTRIBUTION_REQUIRED_KEYS: Final[frozenset[str]] = frozenset(
    {_METHOD_KEY}
)
_SECURITY_RETURN_REQUIRED_KEYS: Final[frozenset[str]] = frozenset(
    {
        _METHOD_KEY,
        _WEIGHT_SOURCE_KEY,
    }
)
_PORTFOLIO_SOURCE_FIELD_ALLOWED_VALUES: Final[dict[str, frozenset[str]]] = {
    _DENOMINATOR_SOURCE_KEY: frozenset({"begin_market_value"}),
}
_SECURITY_RETURN_ALLOWED_VALUES: Final[dict[str, frozenset[str]]] = {
    _WEIGHT_SOURCE_KEY: frozenset({"snapshot_a_weight"}),
}
_PORTFOLIO_SOURCE_FIELD_ALLOWED_SOURCE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        pc_cols.INCOME,
        pc_cols.GAIN_LOSS,
    }
)
_DEFAULT_TOLERANCES: Final[dict[str, float]] = {
    "return": 1e-6,
    "contribution": 1e-6,
    "weight": 1e-6,
    "market_value": 0.01,
    "quantity": 1e-6,
    "price": 1e-6,
    "fx_rate": 1e-8,
}
_COLUMN_TOLERANCE_KEYS: Final[dict[str, str]] = {
    pc_cols.PORTFOLIO_RETURN: "return",
    pc_cols.BEGIN_MARKET_VALUE: "market_value",
    pc_cols.END_MARKET_VALUE: "market_value",
    pc_cols.FLOW: "market_value",
    pc_cols.INCOME: "market_value",
    pc_cols.GAIN_LOSS: "market_value",
    pc_cols.SECURITY_RETURN: "return",
    pc_cols.WEIGHT: "weight",
    pc_cols.CONTRIBUTION: "contribution",
    pc_cols.QUANTITY: "quantity",
    pc_cols.MARKET_VALUE: "market_value",
    pc_cols.ACCRUED: "market_value",
    pc_cols.CASH_BALANCE: "market_value",
    pc_cols.PRICE: "price",
    pc_cols.FX_RATE: "fx_rate",
    pc_cols.AMOUNT: "market_value",
}


@dataclass(frozen=True)
class _TransactionImpactPolicy:
    """Carry explicitly configured transaction impact-method settings.

    Attributes:
        method: YAML method name.
        finding_label: Stable finding-table label exposed to reports.
        flow_timing: Date field used to time external flows.
        day_count: Day-count convention for timing weights.
        inclusion_rule: Beginning/end-of-day flow inclusion convention.
        denominator_source: YAML-selected return denominator source.
        double_count_policy: Rule for handling overlap with portfolio-level
            flow deltas.
    """

    method: str
    finding_label: str
    flow_timing: str | None = None
    day_count: str | None = None
    inclusion_rule: str | None = None
    denominator_source: str | None = None
    double_count_policy: str | None = None


@dataclass(frozen=True)
class _ModifiedDietzEligibility:
    """Describe whether one external-flow row has all Modified Dietz inputs.

    Attributes:
        eligible: Whether the row has every explicitly configured input needed
            for a Modified Dietz cross-check estimate.
        missing_inputs: Human-readable missing or disqualifying inputs.
        flow_date: YAML-selected transaction flow date, when available.
    """

    eligible: bool
    missing_inputs: tuple[str, ...] = ()
    flow_date: dt.date | None = None


class PerformanceComparison:
    """Compare performance snapshots and return finding records.

    Attributes:
        _specification: Parsed comparison specification.
        _portfolio_loader: Loader for normalized portfolio performance rows.
        _security_loader: Loader for normalized security performance rows.
        _security_master_loader: Loader for normalized security master rows.
        _positions_loader: Loader for normalized position rows.
        _cash_loader: Loader for normalized cash rows.
        _prices_loader: Loader for normalized price rows.
        _fx_rates_loader: Loader for normalized FX rate rows.
        _transactions_loader: Loader for normalized transaction rows.
        _transaction_impact_policies: YAML-configured transaction impact
            policies keyed by performance-flow treatment.
        _contribution_impact_policies: YAML-configured contribution impact
            policy labels keyed by dataset and source column.
    """

    def __init__(self, specification: PerformanceComparisonSpecification) -> None:
        """Initialize a performance comparison.

        Args:
            specification: Parsed comparison specification.
        """
        self._specification = specification
        self._portfolio_loader = PortfolioPerformanceLoader(specification)
        self._security_loader = SecurityPerformanceLoader(specification)
        self._security_master_loader = SecurityMasterLoader(specification)
        self._positions_loader = PositionsLoader(specification)
        self._cash_loader = CashLoader(specification)
        self._prices_loader = PricesLoader(specification)
        self._fx_rates_loader = FxRatesLoader(specification)
        self._transactions_loader = TransactionsLoader(specification)
        self._transaction_impact_policies = _transaction_impact_policies(
            specification
        )
        self._contribution_impact_policies = _contribution_impact_policies(
            specification
        )

    def compare_portfolio_performance(self) -> list[Finding]:
        """Compare portfolio performance rows for snapshots A and B.

        Returns:
            Findings for added/dropped rows and material portfolio field
            changes.
        """
        snapshot_a = self._portfolio_loader.load("a")
        snapshot_b = self._portfolio_loader.load("b")
        findings = self._row_presence_findings(snapshot_a, snapshot_b)
        findings.extend(self._changed_value_findings(snapshot_a, snapshot_b))
        return findings

    def compare(self) -> list[Finding]:
        """Compare all currently supported normalized datasets.

        Returns:
            Portfolio findings plus security performance findings when the
            optional security performance dataset is available.
        """
        findings = self.compare_portfolio_performance()
        findings.extend(self.compare_security_performance())
        findings.extend(self.compare_security_master())
        findings.extend(self.compare_positions())
        findings.extend(self.compare_cash())
        findings.extend(self.compare_prices())
        findings.extend(self.compare_fx_rates())
        findings.extend(self.compare_transactions())
        return apply_suppressions(findings, self._specification)

    def compare_security_performance(self) -> list[Finding]:
        """Compare security performance rows for snapshots A and B.

        Returns:
            Findings for added/dropped rows and material security return,
            weight, and contribution changes. Returns an empty list when the
            optional security performance dataset is unavailable.
        """
        snapshot_a = self._security_loader.load("a")
        snapshot_b = self._security_loader.load("b")
        if snapshot_a is None or snapshot_b is None:
            return []

        findings = self._row_presence_findings(
            snapshot_a,
            snapshot_b,
            _SECURITY_KEY_COLUMNS,
            PC_SEC_ADD,
            PC_SEC_DROP,
            pc_cols.SECURITY_PERFORMANCE,
            "Security performance row appears only in snapshot B.",
            "Security performance row appears only in snapshot A.",
        )
        findings.extend(
            self._changed_value_findings(
                snapshot_a,
                snapshot_b,
                _SECURITY_KEY_COLUMNS,
                _SECURITY_COMPARE_COLUMNS,
                pc_cols.SECURITY_PERFORMANCE,
            )
        )
        return findings

    def compare_security_master(self) -> list[Finding]:
        """Compare security master rows for snapshots A and B.

        Returns:
            Findings for added/dropped rows and changed security reference or
            classification fields. Returns an empty list when the optional
            security master dataset is unavailable.
        """
        snapshot_a = self._security_master_loader.load("a")
        snapshot_b = self._security_master_loader.load("b")
        if snapshot_a is None or snapshot_b is None:
            return []

        findings = self._row_presence_findings(
            snapshot_a,
            snapshot_b,
            _SECURITY_MASTER_KEY_COLUMNS,
            PC_ROW_ADD,
            PC_ROW_DROP,
            pc_cols.SECURITY_MASTER,
            "Security master row appears only in snapshot B.",
            "Security master row appears only in snapshot A.",
        )
        findings.extend(
            self._changed_object_findings(
                snapshot_a,
                snapshot_b,
                _SECURITY_MASTER_KEY_COLUMNS,
                _SECURITY_MASTER_COMPARE_COLUMNS,
                pc_cols.SECURITY_MASTER,
            )
        )
        return findings

    def compare_positions(self) -> list[Finding]:
        """Compare position rows for snapshots A and B.

        Returns:
            Findings for added/dropped rows and material position quantity or
            market value changes. Returns an empty list when the optional
            positions dataset is unavailable.
        """
        snapshot_a = self._positions_loader.load("a")
        snapshot_b = self._positions_loader.load("b")
        if snapshot_a is None or snapshot_b is None:
            return []

        portfolio_periods = self._portfolio_periods()
        findings = self._row_presence_findings(
            snapshot_a,
            snapshot_b,
            _POSITIONS_KEY_COLUMNS,
            PC_ROW_ADD,
            PC_ROW_DROP,
            pc_cols.POSITIONS,
            "Position row appears only in snapshot B.",
            "Position row appears only in snapshot A.",
        )
        findings.extend(
            self._changed_value_findings(
                snapshot_a,
                snapshot_b,
                _POSITIONS_KEY_COLUMNS,
                _POSITIONS_COMPARE_COLUMNS,
                pc_cols.POSITIONS,
                portfolio_periods,
            )
        )
        return findings

    def compare_cash(self) -> list[Finding]:
        """Compare cash rows for snapshots A and B.

        Returns:
            Findings for added/dropped rows and material cash balance or market
            value changes. Returns an empty list when the optional cash dataset
            is unavailable.
        """
        snapshot_a = self._cash_loader.load("a")
        snapshot_b = self._cash_loader.load("b")
        if snapshot_a is None or snapshot_b is None:
            return []

        portfolio_periods = self._portfolio_periods()
        key_columns = self._cash_key_columns(snapshot_a, snapshot_b)
        findings = self._row_presence_findings(
            snapshot_a,
            snapshot_b,
            key_columns,
            PC_ROW_ADD,
            PC_ROW_DROP,
            pc_cols.CASH,
            "Cash row appears only in snapshot B.",
            "Cash row appears only in snapshot A.",
        )
        findings.extend(
            self._changed_value_findings(
                snapshot_a,
                snapshot_b,
                key_columns,
                _CASH_COMPARE_COLUMNS,
                pc_cols.CASH,
                portfolio_periods,
            )
        )
        return findings

    def compare_prices(self) -> list[Finding]:
        """Compare price rows for snapshots A and B.

        Returns:
            Findings for added/dropped rows and material price changes. Returns
            an empty list when the optional prices dataset is unavailable.
        """
        snapshot_a = self._prices_loader.load("a")
        snapshot_b = self._prices_loader.load("b")
        if snapshot_a is None or snapshot_b is None:
            return []

        key_columns = self._optional_key_columns(snapshot_a, snapshot_b, _PRICE_KEY_COLUMNS)
        security_periods = self._security_periods()
        findings = self._row_presence_findings(
            snapshot_a,
            snapshot_b,
            key_columns,
            PC_ROW_ADD,
            PC_ROW_DROP,
            pc_cols.PRICES,
            "Price row appears only in snapshot B.",
            "Price row appears only in snapshot A.",
        )
        findings.extend(
            self._changed_value_findings(
                snapshot_a,
                snapshot_b,
                key_columns,
                _PRICE_COMPARE_COLUMNS,
                pc_cols.PRICES,
                security_periods=security_periods,
            )
        )
        return findings

    def compare_fx_rates(self) -> list[Finding]:
        """Compare FX rate rows for snapshots A and B.

        Returns:
            Findings for added/dropped rows and material FX rate changes.
            Returns an empty list when the optional FX rates dataset is
            unavailable.
        """
        snapshot_a = self._fx_rates_loader.load("a")
        snapshot_b = self._fx_rates_loader.load("b")
        if snapshot_a is None or snapshot_b is None:
            return []

        key_columns = self._optional_key_columns(
            snapshot_a,
            snapshot_b,
            _FX_RATE_KEY_COLUMNS,
        )
        findings = self._row_presence_findings(
            snapshot_a,
            snapshot_b,
            key_columns,
            PC_ROW_ADD,
            PC_ROW_DROP,
            pc_cols.FX_RATES,
            "FX rate row appears only in snapshot B.",
            "FX rate row appears only in snapshot A.",
        )
        findings.extend(
            self._changed_value_findings(
                snapshot_a,
                snapshot_b,
                key_columns,
                _FX_RATE_COMPARE_COLUMNS,
                pc_cols.FX_RATES,
            )
        )
        return findings

    def compare_transactions(self) -> list[Finding]:
        """Compare transaction rows for snapshots A and B.

        Returns:
            Findings for added/dropped transaction rows. When both snapshots
            contain transaction identifiers, matching rows are also compared
            for material amount changes.
        """
        snapshot_a = self._transactions_loader.load("a")
        snapshot_b = self._transactions_loader.load("b")
        if snapshot_a is None or snapshot_b is None:
            return []

        key_columns = self._transaction_key_columns(snapshot_a, snapshot_b)
        portfolio_periods = self._portfolio_periods()
        return_denominators = self._portfolio_period_return_denominators()
        findings = self._row_presence_findings(
            snapshot_a,
            snapshot_b,
            key_columns,
            PC_TXN_ADD,
            PC_TXN_DROP,
            pc_cols.TRANSACTIONS,
            "Transaction row appears only in snapshot B.",
            "Transaction row appears only in snapshot A.",
            self._transaction_unmatched_status(key_columns),
        )
        if key_columns == _TRANSACTION_ID_KEY_COLUMNS:
            findings.extend(
                self._changed_value_findings(
                    snapshot_a,
                    snapshot_b,
                    key_columns,
                    _TRANSACTION_COMPARE_COLUMNS,
                    pc_cols.TRANSACTIONS,
                    portfolio_periods,
                    return_denominators=return_denominators,
                    transaction_match_status=TRANSACTION_MATCH_STATUS_ID_MATCH,
                )
            )
        return findings

    def _row_presence_findings(
        self,
        snapshot_a: pl.DataFrame,
        snapshot_b: pl.DataFrame,
        key_columns: tuple[str, ...] = _PORTFOLIO_KEY_COLUMNS,
        add_code: str = PC_ROW_ADD,
        drop_code: str = PC_ROW_DROP,
        dataset: str = pc_cols.PORTFOLIO_PERFORMANCE,
        add_message: str = "Portfolio performance row appears only in snapshot B.",
        drop_message: str = "Portfolio performance row appears only in snapshot A.",
        transaction_match_status: object | None = None,
    ) -> list[Finding]:
        """Return findings for portfolio rows present in only one snapshot."""
        self._validate_unique_keys(snapshot_a, key_columns, dataset, "snapshot A")
        self._validate_unique_keys(snapshot_b, key_columns, dataset, "snapshot B")
        rows_a = self._row_keys(snapshot_a, key_columns)
        rows_b = self._row_keys(snapshot_b, key_columns)
        source_file = self._source_file(dataset)
        findings: list[Finding] = []

        for row_key in sorted(rows_b - rows_a, key=self._sortable_key):
            findings.append(
                self._key_finding(
                    add_code,
                    row_key,
                    key_columns,
                    dataset,
                    source_file,
                    add_message,
                    transaction_match_status,
                )
            )
        for row_key in sorted(rows_a - rows_b, key=self._sortable_key):
            findings.append(
                self._key_finding(
                    drop_code,
                    row_key,
                    key_columns,
                    dataset,
                    source_file,
                    drop_message,
                    transaction_match_status,
                )
            )
        return findings

    def _changed_object_findings(
        self,
        snapshot_a: pl.DataFrame,
        snapshot_b: pl.DataFrame,
        key_columns: tuple[str, ...],
        compare_columns: dict[str, str],
        dataset: str,
    ) -> list[Finding]:
        """Return findings for changed nonnumeric values on matching rows."""
        shared_columns = [
            column
            for column in compare_columns
            if column in snapshot_a.columns and column in snapshot_b.columns
        ]
        if not shared_columns:
            return []

        source_file = self._source_file(dataset)
        joined = snapshot_a.join(
            snapshot_b,
            on=list(key_columns),
            how="inner",
            suffix="_b",
        )
        findings: list[Finding] = []
        for row in joined.iter_rows(named=True):
            for column in shared_columns:
                snapshot_a_value = row[column]
                snapshot_b_value = row[f"{column}_b"]
                if snapshot_a_value == snapshot_b_value:
                    continue
                findings.append(
                    Finding(
                        code=compare_columns[column],
                        severity=SEVERITY_INFORMATIONAL,
                        confidence=CONFIDENCE_HIGH,
                        dataset=dataset,
                        evidence_role=self._evidence_role(
                            compare_columns[column],
                            dataset,
                            column,
                        ),
                        portfolio_id=row.get(pc_cols.PORTFOLIO_ID),
                        security_id=row.get(pc_cols.SECURITY_ID),
                        from_date=row.get(pc_cols.FROM_DATE),
                        thru_date=row.get(pc_cols.THRU_DATE),
                        source_file=source_file,
                        source_column=column,
                        transaction_category=self._transaction_category(row, dataset),
                        cash_flow_sign=self._transaction_cash_flow_sign(row, dataset),
                        performance_flow_sign=self._transaction_performance_flow_sign(
                            row,
                            dataset,
                        ),
                        transaction_semantics_source=(
                            self._transaction_semantics_source(row, dataset)
                        ),
                            transaction_impact_policy=(
                                self._transaction_impact_policy(row, dataset)
                            ),
                            transaction_impact_diagnostic=(
                                self._transaction_impact_diagnostic(
                                    row,
                                    dataset,
                                    column,
                                    row.get(pc_cols.PORTFOLIO_ID),
                                    row.get(pc_cols.FROM_DATE),
                                    row.get(pc_cols.THRU_DATE),
                                    None,
                                )
                            ),
                            transaction_impact_diagnostic_estimate=(
                                self._transaction_impact_diagnostic_estimate(
                                    row,
                                    dataset,
                                    column,
                                    row.get(pc_cols.PORTFOLIO_ID),
                                    row.get(pc_cols.FROM_DATE),
                                    row.get(pc_cols.THRU_DATE),
                                    None,
                                    None,
                                )
                            ),
                            snapshot_a_value=snapshot_a_value,
                            snapshot_b_value=snapshot_b_value,
                            message=f"{dataset} {column!r} changed.",
                    )
                )
        return findings

    def _changed_value_findings(
        self,
        snapshot_a: pl.DataFrame,
        snapshot_b: pl.DataFrame,
        key_columns: tuple[str, ...] = _PORTFOLIO_KEY_COLUMNS,
        compare_columns: dict[str, str] | None = None,
        dataset: str = pc_cols.PORTFOLIO_PERFORMANCE,
        portfolio_periods: pl.DataFrame | None = None,
        security_periods: pl.DataFrame | None = None,
        return_denominators: Mapping[tuple[object, object, object], float] | None = None,
        transaction_match_status: object | None = None,
    ) -> list[Finding]:
        """Return findings for material value changes on matching rows."""
        compare_columns = compare_columns or _PORTFOLIO_COMPARE_COLUMNS
        shared_columns = [
            column
            for column in compare_columns
            if column in snapshot_a.columns and column in snapshot_b.columns
        ]
        if not shared_columns:
            return []

        source_file = self._source_file(dataset)
        joined = snapshot_a.join(
            snapshot_b,
            on=list(key_columns),
            how="inner",
            suffix="_b",
        )
        findings: list[Finding] = []
        for row in joined.iter_rows(named=True):
            finding_contexts = self._changed_value_contexts(
                row,
                dataset,
                portfolio_periods,
                security_periods,
            )
            for column in shared_columns:
                snapshot_a_value = row[column]
                snapshot_b_value = row[f"{column}_b"]
                delta = self._numeric_delta(snapshot_a_value, snapshot_b_value)
                if delta is None:
                    continue
                if abs(delta) <= self._tolerance(column):
                    continue
                for portfolio_id, from_date, thru_date in finding_contexts:
                    return_denominator = self._return_denominator(
                        row,
                        dataset,
                        portfolio_id,
                        from_date,
                        thru_date,
                        return_denominators,
                    )
                    findings.append(
                        Finding(
                            code=compare_columns[column],
                            severity=SEVERITY_MATERIAL,
                            confidence=CONFIDENCE_HIGH,
                            dataset=dataset,
                            evidence_role=self._evidence_role(
                                compare_columns[column],
                                dataset,
                                column,
                            ),
                            portfolio_id=portfolio_id,
                            security_id=row.get(pc_cols.SECURITY_ID),
                            from_date=from_date,
                            thru_date=thru_date,
                            source_file=source_file,
                            source_column=column,
                            transaction_category=self._transaction_category(
                                row,
                                dataset,
                            ),
                            cash_flow_sign=self._transaction_cash_flow_sign(
                                row,
                                dataset,
                            ),
                            performance_flow_sign=(
                                self._transaction_performance_flow_sign(row, dataset)
                            ),
                            transaction_semantics_source=(
                                self._transaction_semantics_source(row, dataset)
                            ),
                            transaction_match_status=transaction_match_status,
                            impact_policy=self._contribution_impact_policy(
                                dataset,
                                column,
                            ),
                            transaction_impact_policy=(
                                self._transaction_impact_policy(row, dataset)
                            ),
                            transaction_impact_diagnostic=(
                                self._transaction_impact_diagnostic(
                                    row,
                                    dataset,
                                    column,
                                    portfolio_id,
                                    from_date,
                                    thru_date,
                                    return_denominator,
                                )
                            ),
                            transaction_impact_diagnostic_estimate=(
                                self._transaction_impact_diagnostic_estimate(
                                    row,
                                    dataset,
                                    column,
                                    portfolio_id,
                                    from_date,
                                    thru_date,
                                    return_denominator,
                                    delta,
                                )
                            ),
                            snapshot_a_value=snapshot_a_value,
                            snapshot_b_value=snapshot_b_value,
                            delta_b_minus_a=delta,
                            return_denominator=return_denominator,
                            return_weight=self._return_weight(row, dataset),
                            message=f"{dataset} {column!r} changed.",
                        )
                    )
        return findings

    @staticmethod
    def _row_keys(frame: pl.DataFrame, key_columns: tuple[str, ...]) -> set[tuple[object, ...]]:
        """Return key tuples from a normalized comparison frame."""
        return set(frame.select(key_columns).iter_rows())

    @staticmethod
    def _validate_unique_keys(
        frame: pl.DataFrame,
        key_columns: tuple[str, ...],
        dataset: str,
        snapshot_label: str,
    ) -> None:
        """Raise if a normalized frame has duplicate comparison keys."""
        duplicate_keys = (
            frame.group_by(list(key_columns))
            .len(name="_duplicate_count")
            .filter(pl.col("_duplicate_count") > 1)
        )
        if duplicate_keys.is_empty():
            return

        duplicate_key = duplicate_keys.select(key_columns).row(0, named=True)
        key_names = ", ".join(key_columns)
        raise PpaError(
            (
                f"{dataset} contains duplicate {snapshot_label} rows for "
                f"key columns {key_names}: {duplicate_key}"
            ),
            112,
        )

    @staticmethod
    def _sortable_key(row_key: tuple[object, ...]) -> tuple[str, ...]:
        """Return a deterministic string sort key for finding output."""
        return tuple(str(value) for value in row_key)

    @staticmethod
    def _transaction_category(row: dict[str, object], dataset: str) -> object | None:
        """Return normalized transaction category context for transaction rows."""
        if dataset != pc_cols.TRANSACTIONS:
            return None
        return row.get(pc_cols.TRANSACTION_CATEGORY)

    @staticmethod
    def _transaction_cash_flow_sign(
        row: dict[str, object],
        dataset: str,
    ) -> object | None:
        """Return normalized transaction cash-flow semantics for transaction rows."""
        if dataset != pc_cols.TRANSACTIONS:
            return None
        return row.get(pc_cols.CASH_FLOW_SIGN)

    @staticmethod
    def _transaction_performance_flow_sign(
        row: dict[str, object],
        dataset: str,
    ) -> object | None:
        """Return normalized transaction performance-flow semantics."""
        if dataset != pc_cols.TRANSACTIONS:
            return None
        return row.get(pc_cols.PERFORMANCE_FLOW_SIGN)

    @staticmethod
    def _transaction_semantics_source(
        row: dict[str, object],
        dataset: str,
    ) -> object | None:
        """Return transaction semantics provenance for transaction rows."""
        if dataset != pc_cols.TRANSACTIONS:
            return None
        return row.get(pc_cols.TRANSACTION_SEMANTICS_SOURCE)

    def _transaction_impact_policy(
        self,
        row: dict[str, object],
        dataset: str,
    ) -> object | None:
        """Return YAML-configured transaction impact policy for a row."""
        if dataset != pc_cols.TRANSACTIONS:
            return None
        performance_flow_sign = row.get(pc_cols.PERFORMANCE_FLOW_SIGN)
        if performance_flow_sign == TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL:
            policy_key = _EXTERNAL_FLOW_KEY
        elif performance_flow_sign == TRANSACTION_PERFORMANCE_FLOW_SIGN_PERFORMANCE:
            policy_key = _PERFORMANCE_KEY
        else:
            return None
        policy = self._transaction_impact_policies.get(policy_key)
        if policy is None:
            return None
        return policy.finding_label

    def _transaction_impact_diagnostic(
        self,
        row: Mapping[str, object],
        dataset: str,
        column: str,
        portfolio_id: object | None,
        from_date: object | None,
        thru_date: object | None,
        denominator: object | None,
    ) -> object | None:
        """Return review-only transaction impact eligibility diagnostics."""
        if dataset != pc_cols.TRANSACTIONS or column != pc_cols.AMOUNT:
            return None
        if row.get(pc_cols.PERFORMANCE_FLOW_SIGN) != TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL:
            return None

        policy = self._transaction_impact_policies.get(_EXTERNAL_FLOW_KEY)
        if policy is None:
            return "external-flow impact method missing"
        if policy.method == _EVIDENCE_ONLY_METHOD:
            return "external-flow evidence-only policy"

        eligibility = _modified_dietz_external_flow_eligibility(
            row=row,
            policy=policy,
            portfolio_id=portfolio_id,
            from_date=from_date,
            thru_date=thru_date,
            denominator=denominator,
        )
        if eligibility.eligible:
            return "modified_dietz cross-check estimate"
        missing = ", ".join(eligibility.missing_inputs)
        return f"modified_dietz missing inputs: {missing}"

    def _transaction_impact_diagnostic_estimate(
        self,
        row: Mapping[str, object],
        dataset: str,
        column: str,
        portfolio_id: object | None,
        from_date: object | None,
        thru_date: object | None,
        denominator: object | None,
        delta: object | None,
    ) -> float | None:
        """Return a review-only Modified Dietz cross-check estimate."""
        policy = self._transaction_impact_policies.get(_EXTERNAL_FLOW_KEY)
        delta_float = _modified_dietz_float(delta)
        denominator_float = _modified_dietz_float(denominator)
        eligibility = _modified_dietz_external_flow_eligibility(
            row=row,
            policy=policy,
            portfolio_id=portfolio_id,
            from_date=from_date,
            thru_date=thru_date,
            denominator=denominator,
        )
        required_inputs_available = all(
            [
                dataset == pc_cols.TRANSACTIONS,
                column == pc_cols.AMOUNT,
                eligibility.eligible,
                eligibility.flow_date is not None,
                delta_float is not None,
                denominator_float is not None,
                isinstance(from_date, dt.date),
                isinstance(thru_date, dt.date),
                policy is not None,
                policy is not None and policy.inclusion_rule is not None,
            ]
        )
        if not required_inputs_available:
            return None
        assert eligibility.flow_date is not None
        assert isinstance(from_date, dt.date)
        assert isinstance(thru_date, dt.date)
        assert policy is not None
        assert policy.inclusion_rule is not None
        assert delta_float is not None
        assert denominator_float is not None
        return _modified_dietz_external_flow_impact(
            flow_delta=delta_float,
            denominator=denominator_float,
            from_date=from_date,
            thru_date=thru_date,
            flow_date=eligibility.flow_date,
            inclusion_rule=policy.inclusion_rule,
        )

    @staticmethod
    def _return_denominator(
        row: Mapping[str, object],
        dataset: str,
        portfolio_id: object | None,
        from_date: object | None,
        thru_date: object | None,
        return_denominators: Mapping[tuple[object, object, object], float] | None,
    ) -> float | None:
        """Return beginning market value for approximate return impacts."""
        if dataset != pc_cols.PORTFOLIO_PERFORMANCE:
            if (
                dataset == pc_cols.TRANSACTIONS
                and portfolio_id is not None
                and from_date is not None
                and thru_date is not None
                and return_denominators is not None
            ):
                return return_denominators.get((portfolio_id, from_date, thru_date))
            return None
        value = row.get(pc_cols.BEGIN_MARKET_VALUE)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    @staticmethod
    def _return_weight(
        row: Mapping[str, object],
        dataset: str,
    ) -> float | None:
        """Return snapshot A security weight for approximate return impacts."""
        if dataset != pc_cols.SECURITY_PERFORMANCE:
            return None
        value = row.get(pc_cols.WEIGHT)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    @staticmethod
    def _changed_value_contexts(
        row: Mapping[str, object],
        dataset: str,
        portfolio_periods: pl.DataFrame | None,
        security_periods: pl.DataFrame | None,
    ) -> list[tuple[object | None, object | None, object | None]]:
        """Return portfolio and period contexts for changed-value findings."""
        security_period_contexts = security_period_contexts_for_dated_evidence(
            row,
            dataset,
            security_periods,
        )
        if security_period_contexts:
            return security_period_contexts

        period_context = period_context_for_dated_evidence(
            row,
            dataset,
            portfolio_periods,
        )
        return [(row.get(pc_cols.PORTFOLIO_ID), period_context[0], period_context[1])]

    def _portfolio_period_return_denominators(
        self,
    ) -> dict[tuple[object, object, object], float]:
        """Return snapshot A beginning market value keyed by portfolio period."""
        snapshot_a = self._portfolio_loader.load("a")
        denominators: dict[tuple[object, object, object], float] = {}
        for row in snapshot_a.iter_rows(named=True):
            value = row.get(pc_cols.BEGIN_MARKET_VALUE)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            key = (
                row.get(pc_cols.PORTFOLIO_ID),
                row.get(pc_cols.FROM_DATE),
                row.get(pc_cols.THRU_DATE),
            )
            denominators[key] = float(value)
        return denominators

    def _portfolio_periods(self) -> pl.DataFrame:
        """Return portfolio period rows from both snapshots for evidence linking."""
        return portfolio_periods_from_snapshots(
            self._portfolio_loader.load("a"),
            self._portfolio_loader.load("b"),
        )

    def _security_periods(self) -> pl.DataFrame | None:
        """Return security period rows from both snapshots for evidence linking."""
        snapshot_a = self._security_loader.load("a")
        snapshot_b = self._security_loader.load("b")
        if snapshot_a is None or snapshot_b is None:
            return None
        return security_periods_from_snapshots(snapshot_a, snapshot_b)

    @staticmethod
    def _cash_key_columns(
        snapshot_a: pl.DataFrame,
        snapshot_b: pl.DataFrame,
    ) -> tuple[str, ...]:
        """Return cash comparison key columns, including currency when present."""
        return PerformanceComparison._optional_key_columns(
            snapshot_a,
            snapshot_b,
            _CASH_KEY_COLUMNS,
        )

    @staticmethod
    def _optional_key_columns(
        snapshot_a: pl.DataFrame,
        snapshot_b: pl.DataFrame,
        candidate_key_columns: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Return key columns present in both snapshots."""
        return tuple(
            column
            for column in candidate_key_columns
            if column in snapshot_a.columns and column in snapshot_b.columns
        )

    @staticmethod
    def _transaction_key_columns(
        snapshot_a: pl.DataFrame,
        snapshot_b: pl.DataFrame,
    ) -> tuple[str, ...]:
        """Return transaction ID key when available, else composite fallback."""
        if (
            pc_cols.TRANSACTION_ID in snapshot_a.columns
            and pc_cols.TRANSACTION_ID in snapshot_b.columns
        ):
            return _TRANSACTION_ID_KEY_COLUMNS
        return PerformanceComparison._optional_key_columns(
            snapshot_a,
            snapshot_b,
            _TRANSACTION_FALLBACK_KEY_COLUMNS,
        )

    @staticmethod
    def _transaction_unmatched_status(key_columns: tuple[str, ...]) -> str:
        """Return the unmatched transaction diagnostic for a comparison key."""
        if key_columns == _TRANSACTION_ID_KEY_COLUMNS:
            return TRANSACTION_MATCH_STATUS_ID_UNMATCHED
        return TRANSACTION_MATCH_STATUS_STRICT_FALLBACK_UNMATCHED

    @staticmethod
    def _key_finding(
        code: str,
        row_key: tuple[object, ...],
        key_columns: tuple[str, ...],
        dataset: str,
        source_file: str | None,
        message: str,
        transaction_match_status: object | None = None,
    ) -> Finding:
        """Return a row-presence finding from a portfolio key tuple."""
        portfolio_id: object | None = None
        security_id: object | None = None
        from_date: object | None = None
        thru_date: object | None = None
        row_context = dict(zip(key_columns, row_key, strict=True))
        portfolio_id = row_context.get(pc_cols.PORTFOLIO_ID)
        security_id = row_context.get(pc_cols.SECURITY_ID)
        from_date = row_context.get(pc_cols.FROM_DATE)
        thru_date = row_context.get(pc_cols.THRU_DATE)

        return Finding(
            code=code,
            severity=SEVERITY_INFORMATIONAL,
            confidence=CONFIDENCE_HIGH,
            dataset=dataset,
            evidence_role=PerformanceComparison._evidence_role(code, dataset, None),
            portfolio_id=portfolio_id,
            security_id=security_id,
            from_date=from_date,
            thru_date=thru_date,
            source_file=source_file,
            transaction_match_status=transaction_match_status,
            message=message,
        )

    @staticmethod
    def _evidence_role(
        code: str,
        dataset: str,
        source_column: str | None,
    ) -> str:
        """Return the explanation role for a finding."""
        if dataset == pc_cols.PORTFOLIO_PERFORMANCE:
            if (
                code in {PC_ROW_ADD, PC_ROW_DROP, PC_PORT_RET}
                and source_column in {None, pc_cols.PORTFOLIO_RETURN}
            ):
                return TARGET_OUTPUT
            return DIRECT_INPUT
        if dataset == pc_cols.SECURITY_PERFORMANCE:
            return RELATED_OUTPUT
        if dataset in _DIRECT_INPUT_DATASETS:
            return DIRECT_INPUT
        return CONTEXT

    @staticmethod
    def _numeric_delta(snapshot_a_value: object, snapshot_b_value: object) -> float | None:
        """Return numeric B-minus-A delta, or ``None`` when not comparable."""
        if snapshot_a_value is None or snapshot_b_value is None:
            return None
        if isinstance(snapshot_a_value, dt.date) or isinstance(snapshot_b_value, dt.date):
            return None
        try:
            return float(cast(Any, snapshot_b_value)) - float(cast(Any, snapshot_a_value))
        except (TypeError, ValueError):
            return None

    def _tolerance(self, column: str) -> float:
        """Return configured tolerance for a portfolio comparison column."""
        tolerances = self._specification.values.get("tolerances", {})
        tolerance_key = _COLUMN_TOLERANCE_KEYS[column]
        if isinstance(tolerances, dict):
            configured_tolerance = tolerances.get(tolerance_key)
            if isinstance(configured_tolerance, int | float):
                return float(configured_tolerance)
        return _DEFAULT_TOLERANCES[tolerance_key]

    def _source_file(self, dataset: str) -> str | None:
        """Return the configured relative source file for a dataset."""
        comparison_file = self._specification.files.get(dataset)
        if comparison_file is None:
            return None
        return comparison_file.relative_path.as_posix()

    def _contribution_impact_policy(
        self,
        dataset: str,
        source_column: str,
    ) -> str | None:
        """Return the YAML-selected contribution-impact policy for a field."""
        return self._contribution_impact_policies.get((dataset, source_column))


def _transaction_impact_policies(
    specification: PerformanceComparisonSpecification,
) -> dict[str, _TransactionImpactPolicy]:
    """Return validated YAML-configured transaction impact policies.

    Args:
        specification: Parsed comparison specification.

    Returns:
        Transaction impact policies keyed by normalized performance-flow
        treatment. Missing configuration returns an empty mapping.

    Raises:
        PpaError: If transaction impact method configuration is malformed or
            names an unsupported method.
    """
    methods_value = specification.values.get(_TRANSACTION_IMPACT_METHODS_KEY, {})
    if methods_value is None:
        return {}
    if not isinstance(methods_value, dict):
        raise PpaError(
            (
                f"{specification.path}: {_TRANSACTION_IMPACT_METHODS_KEY} "
                "must be a mapping."
            ),
            504,
        )

    unsupported_keys = set(methods_value) - {_EXTERNAL_FLOW_KEY, _PERFORMANCE_KEY}
    if unsupported_keys:
        unsupported = ", ".join(sorted(str(key) for key in unsupported_keys))
        raise PpaError(
            (
                f"{specification.path}: unsupported "
                f"{_TRANSACTION_IMPACT_METHODS_KEY} keys: {unsupported}."
            ),
            504,
        )

    policies: dict[str, _TransactionImpactPolicy] = {}
    external_flow_value = methods_value.get(_EXTERNAL_FLOW_KEY)
    if external_flow_value is not None and not isinstance(external_flow_value, dict):
        raise PpaError(
            (
                f"{specification.path}: "
                f"{_TRANSACTION_IMPACT_METHODS_KEY}.{_EXTERNAL_FLOW_KEY} "
                "must be a mapping."
            ),
            504,
        )
    if isinstance(external_flow_value, dict):
        policies[_EXTERNAL_FLOW_KEY] = _validated_external_flow_policy(
            specification,
            external_flow_value,
        )

    performance_value = methods_value.get(_PERFORMANCE_KEY)
    if performance_value is not None and not isinstance(performance_value, dict):
        raise PpaError(
            (
                f"{specification.path}: "
                f"{_TRANSACTION_IMPACT_METHODS_KEY}.{_PERFORMANCE_KEY} "
                "must be a mapping."
            ),
            504,
        )
    if isinstance(performance_value, dict):
        policies[_PERFORMANCE_KEY] = _validated_performance_amount_policy(
            specification,
            performance_value,
        )
    return policies


def _contribution_impact_policies(
    specification: PerformanceComparisonSpecification,
) -> dict[tuple[str, str], str]:
    """Return validated YAML-selected contribution impact policies.

    Args:
        specification: Parsed comparison specification.

    Returns:
        Policy labels keyed by ``(dataset, source_column)``. Missing
        configuration returns an empty mapping, which leaves candidate rows as
        evidence-only.

    Raises:
        PpaError: If contribution impact method configuration is malformed or
            names an unsupported method.
    """
    methods_value = specification.values.get(_CONTRIBUTION_IMPACT_METHODS_KEY, {})
    if methods_value is None:
        return {}
    if not isinstance(methods_value, dict):
        raise PpaError(
            (
                f"{specification.path}: {_CONTRIBUTION_IMPACT_METHODS_KEY} "
                "must be a mapping."
            ),
            504,
        )

    supported_keys = {
        _PORTFOLIO_SOURCE_FIELD_KEY,
        _SECURITY_CONTRIBUTION_KEY,
        _SECURITY_RETURN_KEY,
    }
    unsupported_keys = set(methods_value) - supported_keys
    if unsupported_keys:
        unsupported = ", ".join(sorted(str(key) for key in unsupported_keys))
        raise PpaError(
            (
                f"{specification.path}: unsupported "
                f"{_CONTRIBUTION_IMPACT_METHODS_KEY} keys: {unsupported}."
            ),
            504,
        )

    policies: dict[tuple[str, str], str] = {}
    portfolio_source_field_value = methods_value.get(_PORTFOLIO_SOURCE_FIELD_KEY)
    if portfolio_source_field_value is not None:
        policies.update(
            _validated_portfolio_source_field_policy(
                specification,
                portfolio_source_field_value,
            )
        )

    security_contribution_value = methods_value.get(_SECURITY_CONTRIBUTION_KEY)
    if security_contribution_value is not None:
        policies.update(
            _validated_security_contribution_policy(
                specification,
                security_contribution_value,
            )
        )

    security_return_value = methods_value.get(_SECURITY_RETURN_KEY)
    if security_return_value is not None:
        policies.update(
            _validated_security_return_policy(
                specification,
                security_return_value,
            )
        )
    return policies


def _validated_portfolio_source_field_policy(
    specification: PerformanceComparisonSpecification,
    policy_value: object,
) -> dict[tuple[str, str], str]:
    """Validate portfolio source-field contribution policy configuration."""
    policy = _require_policy_mapping(
        specification,
        _CONTRIBUTION_IMPACT_METHODS_KEY,
        _PORTFOLIO_SOURCE_FIELD_KEY,
        policy_value,
    )
    _validate_policy_keys(
        specification,
        _CONTRIBUTION_IMPACT_METHODS_KEY,
        _PORTFOLIO_SOURCE_FIELD_KEY,
        policy,
        _PORTFOLIO_SOURCE_FIELD_REQUIRED_KEYS,
    )
    _validate_policy_method(
        specification,
        _CONTRIBUTION_IMPACT_METHODS_KEY,
        _PORTFOLIO_SOURCE_FIELD_KEY,
        policy,
        _SOURCE_FIELD_DELTA_OVER_BEGIN_MV_METHOD,
    )
    _validate_allowed_policy_values(
        specification,
        _CONTRIBUTION_IMPACT_METHODS_KEY,
        _PORTFOLIO_SOURCE_FIELD_KEY,
        policy,
        _PORTFOLIO_SOURCE_FIELD_ALLOWED_VALUES,
    )
    source_fields = policy[_SOURCE_FIELDS_KEY]
    if not isinstance(source_fields, list) or not source_fields:
        raise PpaError(
            (
                f"{specification.path}: "
                f"{_CONTRIBUTION_IMPACT_METHODS_KEY}.{_PORTFOLIO_SOURCE_FIELD_KEY}."
                f"{_SOURCE_FIELDS_KEY} must be a non-empty list."
            ),
            504,
        )
    if any(not isinstance(field, str) for field in source_fields):
        raise PpaError(
            (
                f"{specification.path}: "
                f"{_CONTRIBUTION_IMPACT_METHODS_KEY}.{_PORTFOLIO_SOURCE_FIELD_KEY}."
                f"{_SOURCE_FIELDS_KEY} values must be strings."
            ),
            504,
        )
    unsupported_fields = set(source_fields) - _PORTFOLIO_SOURCE_FIELD_ALLOWED_SOURCE_FIELDS
    if unsupported_fields:
        unsupported = ", ".join(sorted(str(field) for field in unsupported_fields))
        allowed = ", ".join(sorted(_PORTFOLIO_SOURCE_FIELD_ALLOWED_SOURCE_FIELDS))
        raise PpaError(
            (
                f"{specification.path}: "
                f"{_CONTRIBUTION_IMPACT_METHODS_KEY}.{_PORTFOLIO_SOURCE_FIELD_KEY}."
                f"{_SOURCE_FIELDS_KEY} contains unsupported fields: {unsupported}. "
                f"Allowed fields: {allowed}."
            ),
            504,
        )
    return {
        (pc_cols.PORTFOLIO_PERFORMANCE, str(field)): IMPACT_POLICY_PORTFOLIO_SOURCE_FIELD
        for field in source_fields
    }


def _validated_security_contribution_policy(
    specification: PerformanceComparisonSpecification,
    policy_value: object,
) -> dict[tuple[str, str], str]:
    """Validate vendor contribution-delta policy configuration."""
    policy = _require_policy_mapping(
        specification,
        _CONTRIBUTION_IMPACT_METHODS_KEY,
        _SECURITY_CONTRIBUTION_KEY,
        policy_value,
    )
    _validate_policy_keys(
        specification,
        _CONTRIBUTION_IMPACT_METHODS_KEY,
        _SECURITY_CONTRIBUTION_KEY,
        policy,
        _SECURITY_CONTRIBUTION_REQUIRED_KEYS,
    )
    _validate_policy_method(
        specification,
        _CONTRIBUTION_IMPACT_METHODS_KEY,
        _SECURITY_CONTRIBUTION_KEY,
        policy,
        _VENDOR_CONTRIBUTION_DELTA_METHOD,
    )
    return {
        (pc_cols.SECURITY_PERFORMANCE, pc_cols.CONTRIBUTION): (
            IMPACT_POLICY_SECURITY_CONTRIBUTION
        )
    }


def _validated_security_return_policy(
    specification: PerformanceComparisonSpecification,
    policy_value: object,
) -> dict[tuple[str, str], str]:
    """Validate weighted security-return policy configuration."""
    policy = _require_policy_mapping(
        specification,
        _CONTRIBUTION_IMPACT_METHODS_KEY,
        _SECURITY_RETURN_KEY,
        policy_value,
    )
    _validate_policy_keys(
        specification,
        _CONTRIBUTION_IMPACT_METHODS_KEY,
        _SECURITY_RETURN_KEY,
        policy,
        _SECURITY_RETURN_REQUIRED_KEYS,
    )
    _validate_policy_method(
        specification,
        _CONTRIBUTION_IMPACT_METHODS_KEY,
        _SECURITY_RETURN_KEY,
        policy,
        _SECURITY_RETURN_DELTA_TIMES_WEIGHT_METHOD,
    )
    _validate_allowed_policy_values(
        specification,
        _CONTRIBUTION_IMPACT_METHODS_KEY,
        _SECURITY_RETURN_KEY,
        policy,
        _SECURITY_RETURN_ALLOWED_VALUES,
    )
    return {
        (pc_cols.SECURITY_PERFORMANCE, pc_cols.SECURITY_RETURN): (
            IMPACT_POLICY_SECURITY_RETURN_WEIGHTED
        )
    }


def _require_policy_mapping(
    specification: PerformanceComparisonSpecification,
    root_key: str,
    policy_key: str,
    policy_value: object,
) -> Mapping[str, object]:
    """Return a YAML policy mapping or raise a contract error."""
    if isinstance(policy_value, dict):
        return policy_value
    raise PpaError(
        (
            f"{specification.path}: {root_key}.{policy_key} "
            "must be a mapping."
        ),
        504,
    )


def _validate_policy_keys(
    specification: PerformanceComparisonSpecification,
    root_key: str,
    policy_key: str,
    policy: Mapping[str, object],
    required_keys: frozenset[str],
) -> None:
    """Validate one explicit policy has exactly the supported keys."""
    unsupported_keys = set(policy) - required_keys
    if unsupported_keys:
        unsupported = ", ".join(sorted(str(key) for key in unsupported_keys))
        raise PpaError(
            (
                f"{specification.path}: {root_key}.{policy_key} "
                f"has unsupported keys: {unsupported}."
            ),
            504,
        )
    missing_keys = required_keys - set(policy)
    if missing_keys:
        missing = ", ".join(sorted(str(key) for key in missing_keys))
        raise PpaError(
            (
                f"{specification.path}: {root_key}.{policy_key} "
                f"is missing required keys: {missing}."
            ),
            504,
        )


def _validate_policy_method(
    specification: PerformanceComparisonSpecification,
    root_key: str,
    policy_key: str,
    policy: Mapping[str, object],
    expected_method: str,
) -> None:
    """Validate one explicit policy selects the only supported method."""
    if policy.get(_METHOD_KEY) != expected_method:
        raise PpaError(
            (
                f"{specification.path}: {root_key}.{policy_key}."
                f"{_METHOD_KEY} must be {expected_method!r}."
            ),
            504,
        )


def _validate_allowed_policy_values(
    specification: PerformanceComparisonSpecification,
    root_key: str,
    policy_key: str,
    policy: Mapping[str, object],
    allowed_values_by_key: Mapping[str, frozenset[str]],
) -> None:
    """Validate one explicit policy's constrained option values."""
    for key, allowed_values in allowed_values_by_key.items():
        value = policy.get(key)
        if value not in allowed_values:
            allowed = ", ".join(sorted(allowed_values))
            raise PpaError(
                (
                    f"{specification.path}: {root_key}.{policy_key}."
                    f"{key} must be one of: {allowed}."
                ),
                504,
            )


def _validated_external_flow_policy(
    specification: PerformanceComparisonSpecification,
    external_flow_value: Mapping[str, object],
) -> _TransactionImpactPolicy:
    """Validate and preserve the external-flow YAML policy."""
    method = external_flow_value.get(_METHOD_KEY)
    if method is None:
        raise PpaError(
            (
                f"{specification.path}: "
                f"{_TRANSACTION_IMPACT_METHODS_KEY}.{_EXTERNAL_FLOW_KEY}."
                f"{_METHOD_KEY} is required."
            ),
            504,
        )
    if method == _MODIFIED_DIETZ_METHOD:
        return _validated_modified_dietz_policy(
            specification,
            external_flow_value,
        )
    if method != _EVIDENCE_ONLY_METHOD:
        _raise_unsupported_external_flow_method(specification, method)

    return _TransactionImpactPolicy(
        method=_EVIDENCE_ONLY_METHOD,
        finding_label=TRANSACTION_IMPACT_POLICY_EXTERNAL_FLOW_EVIDENCE_ONLY,
    )


def _validated_performance_amount_policy(
    specification: PerformanceComparisonSpecification,
    performance_value: Mapping[str, object],
) -> _TransactionImpactPolicy:
    """Validate the performance transaction-amount impact YAML policy."""
    unsupported_keys = set(performance_value) - _PERFORMANCE_AMOUNT_REQUIRED_KEYS
    if unsupported_keys:
        unsupported = ", ".join(sorted(str(key) for key in unsupported_keys))
        raise PpaError(
            (
                f"{specification.path}: "
                f"{_TRANSACTION_IMPACT_METHODS_KEY}.{_PERFORMANCE_KEY} "
                f"has unsupported keys: {unsupported}."
            ),
            504,
        )

    missing_keys = _PERFORMANCE_AMOUNT_REQUIRED_KEYS - set(performance_value)
    if missing_keys:
        missing = ", ".join(sorted(str(key) for key in missing_keys))
        raise PpaError(
            (
                f"{specification.path}: "
                f"{_TRANSACTION_IMPACT_METHODS_KEY}.{_PERFORMANCE_KEY} "
                f"is missing required keys: {missing}."
            ),
            504,
        )

    method = performance_value.get(_METHOD_KEY)
    if method != _TRANSACTION_AMOUNT_DELTA_METHOD:
        raise PpaError(
            (
                f"{specification.path}: "
                f"{_TRANSACTION_IMPACT_METHODS_KEY}.{_PERFORMANCE_KEY}."
                f"{_METHOD_KEY} must be {_TRANSACTION_AMOUNT_DELTA_METHOD!r}."
            ),
            504,
        )
    for key, allowed_values in _PERFORMANCE_AMOUNT_ALLOWED_VALUES.items():
        value = performance_value.get(key)
        if value not in allowed_values:
            allowed = ", ".join(sorted(allowed_values))
            raise PpaError(
                (
                    f"{specification.path}: "
                    f"{_TRANSACTION_IMPACT_METHODS_KEY}.{_PERFORMANCE_KEY}."
                    f"{key} must be one of: {allowed}."
                ),
                504,
            )
    return _TransactionImpactPolicy(
        method=_TRANSACTION_AMOUNT_DELTA_METHOD,
        finding_label=TRANSACTION_IMPACT_POLICY_PERFORMANCE_AMOUNT_DELTA,
        denominator_source=str(performance_value[_DENOMINATOR_SOURCE_KEY]),
    )


def _validated_modified_dietz_policy(
    specification: PerformanceComparisonSpecification,
    external_flow_value: Mapping[str, object],
) -> _TransactionImpactPolicy:
    """Validate and preserve the Modified Dietz YAML policy shape."""
    unsupported_keys = set(external_flow_value) - _MODIFIED_DIETZ_REQUIRED_KEYS
    if unsupported_keys:
        unsupported = ", ".join(sorted(str(key) for key in unsupported_keys))
        raise PpaError(
            (
                f"{specification.path}: "
                f"{_TRANSACTION_IMPACT_METHODS_KEY}.{_EXTERNAL_FLOW_KEY} "
                f"has unsupported modified_dietz keys: {unsupported}."
            ),
            504,
        )

    missing_keys = _MODIFIED_DIETZ_REQUIRED_KEYS - set(external_flow_value)
    if missing_keys:
        missing = ", ".join(sorted(str(key) for key in missing_keys))
        raise PpaError(
            (
                f"{specification.path}: "
                f"{_TRANSACTION_IMPACT_METHODS_KEY}.{_EXTERNAL_FLOW_KEY} "
                f"is missing required modified_dietz keys: {missing}."
            ),
            504,
        )

    for key, allowed_values in _MODIFIED_DIETZ_ALLOWED_VALUES.items():
        value = external_flow_value.get(key)
        if value not in allowed_values:
            allowed = ", ".join(sorted(allowed_values))
            raise PpaError(
                (
                    f"{specification.path}: "
                    f"{_TRANSACTION_IMPACT_METHODS_KEY}.{_EXTERNAL_FLOW_KEY}."
                    f"{key} must be one of: {allowed}."
                ),
                504,
            )

    return _TransactionImpactPolicy(
        method=_MODIFIED_DIETZ_METHOD,
        finding_label="external_flow:modified_dietz",
        flow_timing=cast(str, external_flow_value[_FLOW_TIMING_KEY]),
        day_count=cast(str, external_flow_value[_DAY_COUNT_KEY]),
        inclusion_rule=cast(str, external_flow_value[_INCLUSION_RULE_KEY]),
        denominator_source=cast(str, external_flow_value[_DENOMINATOR_SOURCE_KEY]),
        double_count_policy=cast(str, external_flow_value[_DOUBLE_COUNT_POLICY_KEY]),
    )


def _modified_dietz_external_flow_eligibility(
    *,
    row: Mapping[str, object],
    policy: _TransactionImpactPolicy | None,
    portfolio_id: object | None,
    from_date: object | None,
    thru_date: object | None,
    denominator: object | None,
) -> _ModifiedDietzEligibility:
    """Return whether a transaction row has explicit Modified Dietz inputs.

    This is a guardrail for the diagnostic-only Modified Dietz path. It
    deliberately does not make the estimate part of regular contribution
    totals.
    """
    missing_inputs: list[str] = []
    flow_date = _modified_dietz_flow_date(row, policy)

    if row.get(pc_cols.PERFORMANCE_FLOW_SIGN) != TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL:
        missing_inputs.append("external performance-flow semantics")
    if policy is None or policy.method != _MODIFIED_DIETZ_METHOD:
        missing_inputs.append("modified_dietz policy")
    elif policy.double_count_policy != "cross_check_only":
        missing_inputs.append("cross_check_only double-count policy")
    if flow_date is None:
        missing_inputs.append("flow date")
    if portfolio_id is None:
        missing_inputs.append("portfolio")
    if not isinstance(from_date, dt.date) or not isinstance(thru_date, dt.date):
        missing_inputs.append("portfolio period")
    if not _usable_modified_dietz_denominator(denominator):
        missing_inputs.append("nonzero begin_market_value denominator")

    if (
        flow_date is not None
        and isinstance(from_date, dt.date)
        and isinstance(thru_date, dt.date)
        and not from_date <= flow_date <= thru_date
    ):
        missing_inputs.append("in-period flow date")

    return _ModifiedDietzEligibility(
        eligible=not missing_inputs,
        missing_inputs=tuple(missing_inputs),
        flow_date=flow_date,
    )


def _modified_dietz_external_flow_impact(
    *,
    flow_delta: float,
    denominator: float,
    from_date: dt.date,
    thru_date: dt.date,
    flow_date: dt.date,
    inclusion_rule: str,
) -> float:
    """Return a Modified Dietz cross-check estimate for one external flow."""
    flow_weight = _modified_dietz_flow_weight(
        from_date=from_date,
        thru_date=thru_date,
        flow_date=flow_date,
        inclusion_rule=inclusion_rule,
    )
    return flow_delta * flow_weight / denominator


def _modified_dietz_flow_weight(
    *,
    from_date: dt.date,
    thru_date: dt.date,
    flow_date: dt.date,
    inclusion_rule: str,
) -> float:
    """Return the actual-days Modified Dietz flow weight."""
    period_days = (thru_date - from_date).days + 1
    if period_days <= 0:
        raise ValueError("period must include at least one day")
    if not from_date <= flow_date <= thru_date:
        raise ValueError("flow_date must be inside the period")

    remaining_days = (thru_date - flow_date).days
    if inclusion_rule == "beginning_of_day":
        remaining_days += 1
    elif inclusion_rule != "end_of_day":
        raise ValueError("inclusion_rule must be beginning_of_day or end_of_day")
    return remaining_days / period_days


def _modified_dietz_flow_date(
    row: Mapping[str, object],
    policy: _TransactionImpactPolicy | None,
) -> dt.date | None:
    """Return the YAML-selected transaction flow date for Modified Dietz."""
    if policy is None:
        return None
    flow_date_column_by_timing = {
        "trade_date": pc_cols.TRANSACTION_DATE,
        "settlement_date": pc_cols.SETTLEMENT_DATE,
    }
    if policy.flow_timing is None:
        return None
    date_column = flow_date_column_by_timing.get(policy.flow_timing)
    if date_column is None:
        return None
    value = row.get(date_column)
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return None


def _usable_modified_dietz_denominator(value: object) -> bool:
    """Return whether a configured Modified Dietz denominator is usable."""
    number = _modified_dietz_float(value)
    return number is not None and number != 0


def _usable_modified_dietz_number(value: object) -> bool:
    """Return whether a value can be used in Modified Dietz arithmetic."""
    return _modified_dietz_float(value) is not None


def _modified_dietz_float(value: object) -> float | None:
    """Return a float for non-boolean numeric Modified Dietz values."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _raise_unsupported_external_flow_method(
    specification: PerformanceComparisonSpecification,
    method: object,
) -> None:
    """Raise for external-flow methods that are not implemented yet."""
    method_text = str(method)
    reserved_note = ""
    if method_text in _RESERVED_EXTERNAL_FLOW_METHODS:
        reserved_note = " The method name is reserved but not implemented."
    raise PpaError(
        (
            f"{specification.path}: "
            f"{_TRANSACTION_IMPACT_METHODS_KEY}.{_EXTERNAL_FLOW_KEY}."
            f"{_METHOD_KEY} must be {_EVIDENCE_ONLY_METHOD!r} until an "
            "external-flow impact formula is explicitly supported."
            f"{reserved_note}"
        ),
        504,
    )
