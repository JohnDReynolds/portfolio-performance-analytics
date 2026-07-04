"""Compare normalized performance snapshot datasets."""

from __future__ import annotations

# Python imports
import datetime as dt
from collections.abc import Mapping
from typing import Any, Final, cast

# Third-party imports
import polars as pl

# Project imports
from ppar.errors import PpaError
from ppar.performance_comparison import schema as pc_cols
from ppar.performance_comparison import field_roles as _field_roles
from ppar.performance_comparison.cash import CashLoader
from ppar.performance_comparison.findings import (
    CONFIDENCE_HIGH,
    CONTEXT,
    DIRECT_INPUT,
    EvidenceRole,
    IMPACT_POLICY_HOLDING_QUANTITY_UNIT_MARKET_VALUE,
    RELATED_OUTPUT,
    TARGET_OUTPUT,
    TransactionMatchStatus,
    TRANSACTION_MATCH_STATUS_ADDED_IN_SNAPSHOT_B,
    TRANSACTION_MATCH_STATUS_AMBIGUOUS_FALLBACK_MATCH,
    TRANSACTION_MATCH_STATUS_ID_MATCH,
    TRANSACTION_MATCH_STATUS_MISSING_FROM_SNAPSHOT_B,
    TRANSACTION_MATCH_STATUS_SINGLETON_FALLBACK_MATCH,
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
    PC_TXN_AMBIG,
    PC_TXN_AMT,
    PC_TXN_COMM,
    PC_TXN_DROP,
    PC_TXN_PRICE,
    PC_TXN_QTY,
    PC_REF_CLASS,
    PC_REF_ID,
    PC_HOLD_MV,
    PC_HOLD_COST,
    PC_HOLD_QTY,
    PC_HOLD_ACCR,
    PC_PRICE,
    SEVERITY_INFORMATIONAL,
    SEVERITY_MATERIAL,
    Finding,
)
from ppar.performance_comparison.fx_rates import FxRatesLoader
from ppar.performance_comparison.period_linking import (
    period_context_for_dated_evidence,
    portfolio_periods_from_snapshots,
)
from ppar.performance_comparison.policies import (
    _EVIDENCE_ONLY_METHOD,
    _EXTERNAL_FLOW_KEY,
    _PERFORMANCE_KEY,
    _TRANSACTION_COMMISSION_KEY,
    _TRANSACTION_PRICE_KEY,
    _TRANSACTION_QUANTITY_KEY,
    _cash_impact_policies,
    _contribution_impact_policies,
    _evidence_only_impact_policies,
    _fx_rate_impact_policies,
    _is_evidence_only_policy_label,
    _modified_dietz_external_flow_eligibility,
    _holding_impact_policies,
    _price_impact_policies,
    _security_return_impact_policies,
    _transaction_impact_policies,
)
from ppar.performance_comparison.modified_dietz import (
    modified_dietz_external_flow_impact as _modified_dietz_external_flow_impact,
    modified_dietz_float as _modified_dietz_float,
)
from ppar.performance_comparison.portfolio_performance import PortfolioPerformanceLoader
from ppar.performance_comparison.holdings import HoldingsLoader
from ppar.performance_comparison.rules import apply_suppressions
from ppar.performance_comparison.security_performance import SecurityPerformanceLoader
from ppar.performance_comparison.specification import (
    SECURITY_COMPARISON_LEVEL,
    PerformanceComparisonSpecification,
)
from ppar.performance_comparison.transactions import (
    TRANSACTION_CATEGORY_BUY,
    TRANSACTION_CATEGORY_SELL,
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
_HOLDINGS_KEY_COLUMNS: Final[tuple[str, str, str]] = (
    pc_cols.PORTFOLIO_ID,
    pc_cols.SECURITY_ID,
    pc_cols.HOLDING_DATE,
)
_CASH_KEY_COLUMNS: Final[tuple[str, str, str]] = (
    pc_cols.PORTFOLIO_ID,
    pc_cols.CASH_DATE,
    pc_cols.CURRENCY,
)
_FX_RATE_KEY_COLUMNS: Final[tuple[str, str, str, str, str]] = (
    pc_cols.FROM_CURRENCY,
    pc_cols.TO_CURRENCY,
    pc_cols.RATE_DATE,
    pc_cols.RATE_SOURCE,
    pc_cols.RATE_TYPE,
)
_TRANSACTION_ID_KEY_COLUMNS: Final[tuple[str]] = (pc_cols.TRANSACTION_ID,)
_TRANSACTION_SINGLETON_FALLBACK_KEY_COLUMNS: Final[tuple[str, str, str, str]] = (
    pc_cols.PORTFOLIO_ID,
    pc_cols.SECURITY_ID,
    pc_cols.TRANSACTION_DATE,
    pc_cols.TRANSACTION_CODE,
)
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
_HOLDINGS_COMPARE_COLUMNS: Final[dict[str, str]] = {
    pc_cols.QUANTITY: PC_HOLD_QTY,
    pc_cols.PRICE: PC_PRICE,
    pc_cols.MARKET_VALUE: PC_HOLD_MV,
    pc_cols.COST: PC_HOLD_COST,
    pc_cols.ACCRUED: PC_HOLD_ACCR,
}
_CASH_COMPARE_COLUMNS: Final[dict[str, str]] = {
    pc_cols.CASH_BALANCE: PC_CASH_MV,
    pc_cols.MARKET_VALUE: PC_CASH_MV,
}
_FX_RATE_COMPARE_COLUMNS: Final[dict[str, str]] = {
    pc_cols.FX_RATE: PC_FX_RATE,
}
_TRANSACTION_COMPARE_COLUMNS: Final[dict[str, str]] = {
    pc_cols.AMOUNT: PC_TXN_AMT,
    pc_cols.QUANTITY: PC_TXN_QTY,
    pc_cols.PRICE: PC_TXN_PRICE,
    pc_cols.COMMISSION: PC_TXN_COMM,
}
_DIRECT_INPUT_DATASETS: Final[frozenset[str]] = frozenset(
    {
        pc_cols.FX_RATES,
        pc_cols.TRANSACTIONS,
        pc_cols.HOLDINGS,
        pc_cols.CASH,
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
    pc_cols.COST: "market_value",
    pc_cols.ACCRUED: "market_value",
    pc_cols.CASH_BALANCE: "market_value",
    pc_cols.PRICE: "price",
    pc_cols.FX_RATE: "fx_rate",
    pc_cols.AMOUNT: "market_value",
    pc_cols.COMMISSION: "market_value",
}




class PerformanceComparison:
    """Compare performance snapshots and return finding records.

    Attributes:
        _specification: Parsed comparison specification.
        _portfolio_loader: Loader for normalized portfolio performance rows.
        _security_loader: Loader for normalized security performance rows.
        _holdings_loader: Loader for normalized holding rows.
        _cash_loader: Loader for normalized cash rows.
        _fx_rates_loader: Loader for normalized FX rate rows.
        _transactions_loader: Loader for normalized transaction rows.
        _transaction_impact_policies: YAML-configured transaction impact
            policies keyed by performance-flow treatment.
        _contribution_impact_policies: YAML-configured contribution impact
            policy labels keyed by dataset and source column.
        _holding_impact_policies: YAML-configured holding impact policy
            labels keyed by source column.
        _price_impact_policies: YAML-configured price impact policy labels
            keyed by source column.
        _cash_impact_policies: YAML-configured cash impact policy labels keyed
            by source column.
        _fx_rate_impact_policies: YAML-configured FX rate impact policy labels
            keyed by source column.
        _evidence_only_impact_policies: YAML-configured evidence-only policy
            labels keyed by dataset and source column.
    """

    def __init__(self, specification: PerformanceComparisonSpecification) -> None:
        """Initialize a performance comparison.

        Args:
            specification: Parsed comparison specification.
        """
        self._specification = specification
        self._portfolio_loader = PortfolioPerformanceLoader(specification)
        self._security_loader = SecurityPerformanceLoader(specification)
        self._holdings_loader = HoldingsLoader(specification)
        self._cash_loader = CashLoader(specification)
        self._fx_rates_loader = FxRatesLoader(specification)
        self._transactions_loader = TransactionsLoader(specification)
        self._transaction_impact_policies = _transaction_impact_policies(
            specification
        )
        self._security_return_impact_policies = _security_return_impact_policies(
            specification
        )
        self._contribution_impact_policies = _contribution_impact_policies(
            specification
        )
        self._holding_impact_policies = _holding_impact_policies(specification)
        self._price_impact_policies = _price_impact_policies(specification)
        self._cash_impact_policies = _cash_impact_policies(specification)
        self._fx_rate_impact_policies = _fx_rate_impact_policies(specification)
        self._evidence_only_impact_policies = _evidence_only_impact_policies(
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
            Findings for the configured primary performance-result dataset plus
            shared source-data findings.
        """
        findings = self._primary_performance_findings()
        findings.extend(self.compare_holdings())
        findings.extend(self.compare_cash())
        findings.extend(self.compare_fx_rates())
        findings.extend(self.compare_transactions())
        return apply_suppressions(findings, self._specification)

    def _primary_performance_findings(self) -> list[Finding]:
        """Return findings for the configured primary performance-result layer."""
        if self._specification.comparison_level == SECURITY_COMPARISON_LEVEL:
            return self.compare_security_performance()
        return self.compare_portfolio_performance()

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

    def compare_holdings(self) -> list[Finding]:
        """Compare holding rows for snapshots A and B.

        Returns:
            Findings for added/dropped rows and material holding quantity or
            market value changes. Returns an empty list when the optional
            holdings dataset is unavailable.
        """
        snapshot_a = self._holdings_loader.load("a")
        snapshot_b = self._holdings_loader.load("b")
        if snapshot_a is None or snapshot_b is None:
            return []

        portfolio_periods = self._portfolio_periods()
        return_denominators = self._portfolio_period_return_denominators()
        return_weights = self._security_period_return_weights()
        findings = self._row_presence_findings(
            snapshot_a,
            snapshot_b,
            _HOLDINGS_KEY_COLUMNS,
            PC_ROW_ADD,
            PC_ROW_DROP,
            pc_cols.HOLDINGS,
            "Holding row appears only in snapshot B.",
            "Holding row appears only in snapshot A.",
        )
        findings.extend(
            self._changed_value_findings(
                snapshot_a,
                snapshot_b,
                _HOLDINGS_KEY_COLUMNS,
                _HOLDINGS_COMPARE_COLUMNS,
                pc_cols.HOLDINGS,
                portfolio_periods,
                return_denominators=return_denominators,
                return_weights=return_weights,
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
        return_denominators = self._portfolio_period_return_denominators()
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
                return_denominators=return_denominators,
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
        fallback_periods = self._single_changed_portfolio_return_periods()
        findings = self._transaction_row_presence_findings(
            snapshot_a,
            snapshot_b,
            key_columns,
            portfolio_periods,
            return_denominators,
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
                    transaction_fallback_periods=fallback_periods,
                )
            )
        else:
            singleton_a, singleton_b = self._transaction_singleton_match_frames(
                snapshot_a,
                snapshot_b,
            )
            findings.extend(
                self._changed_value_findings(
                    singleton_a,
                    singleton_b,
                    _TRANSACTION_SINGLETON_FALLBACK_KEY_COLUMNS,
                    _TRANSACTION_COMPARE_COLUMNS,
                    pc_cols.TRANSACTIONS,
                    portfolio_periods,
                    return_denominators=return_denominators,
                    transaction_match_status=(
                        TRANSACTION_MATCH_STATUS_SINGLETON_FALLBACK_MATCH
                    ),
                    transaction_fallback_periods=fallback_periods,
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
        transaction_match_status: TransactionMatchStatus | None = None,
        add_transaction_match_status: TransactionMatchStatus | None = None,
        drop_transaction_match_status: TransactionMatchStatus | None = None,
    ) -> list[Finding]:
        """Return findings for portfolio rows present in only one snapshot."""
        self._validate_unique_keys(snapshot_a, key_columns, dataset, "snapshot A")
        self._validate_unique_keys(snapshot_b, key_columns, dataset, "snapshot B")
        rows_a = self._row_keys(snapshot_a, key_columns)
        rows_b = self._row_keys(snapshot_b, key_columns)
        source_file = self._source_file(dataset)
        findings: list[Finding] = []
        add_status = add_transaction_match_status or transaction_match_status
        drop_status = drop_transaction_match_status or transaction_match_status

        for row_key in sorted(rows_b - rows_a, key=self._sortable_key):
            findings.append(
                self._key_finding(
                    add_code,
                    row_key,
                    key_columns,
                    dataset,
                    source_file,
                    add_message,
                    add_status,
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
                    drop_status,
                )
            )
        return findings

    def _transaction_row_presence_findings(
        self,
        snapshot_a: pl.DataFrame,
        snapshot_b: pl.DataFrame,
        key_columns: tuple[str, ...],
        portfolio_periods: pl.DataFrame | None,
        return_denominators: Mapping[tuple[object, object, object], float] | None,
    ) -> list[Finding]:
        """Return transaction add/drop findings and fallback ambiguity diagnostics."""
        if key_columns == _TRANSACTION_ID_KEY_COLUMNS:
            self._validate_unique_keys(
                snapshot_a,
                key_columns,
                pc_cols.TRANSACTIONS,
                "snapshot A",
            )
            self._validate_unique_keys(
                snapshot_b,
                key_columns,
                pc_cols.TRANSACTIONS,
                "snapshot B",
            )
        else:
            self._validate_complete_keys(
                snapshot_a,
                key_columns,
                pc_cols.TRANSACTIONS,
                "snapshot A",
            )
            self._validate_complete_keys(
                snapshot_b,
                key_columns,
                pc_cols.TRANSACTIONS,
                "snapshot B",
            )
            self._validate_complete_keys(
                snapshot_a,
                _TRANSACTION_SINGLETON_FALLBACK_KEY_COLUMNS,
                pc_cols.TRANSACTIONS,
                "snapshot A",
            )
            self._validate_complete_keys(
                snapshot_b,
                _TRANSACTION_SINGLETON_FALLBACK_KEY_COLUMNS,
                pc_cols.TRANSACTIONS,
                "snapshot B",
            )

            singleton_a, singleton_b = self._transaction_singleton_match_frames(
                snapshot_a,
                snapshot_b,
            )
            snapshot_a = self._without_transaction_singleton_matches(
                snapshot_a,
                singleton_a,
            )
            snapshot_b = self._without_transaction_singleton_matches(
                snapshot_b,
                singleton_b,
            )

        findings = self._transaction_presence_findings_for_side(
            snapshot_b,
            snapshot_a,
            key_columns,
            PC_TXN_ADD,
            "Transaction row appears only in snapshot B.",
            TRANSACTION_MATCH_STATUS_ADDED_IN_SNAPSHOT_B,
            "b",
            portfolio_periods,
            return_denominators,
        )
        findings.extend(
            self._transaction_presence_findings_for_side(
                snapshot_a,
                snapshot_b,
                key_columns,
                PC_TXN_DROP,
                "Transaction row appears only in snapshot A.",
                TRANSACTION_MATCH_STATUS_MISSING_FROM_SNAPSHOT_B,
                "a",
                portfolio_periods,
                return_denominators,
            )
        )
        if key_columns == _TRANSACTION_ID_KEY_COLUMNS:
            return findings

        findings.extend(
            self._transaction_fallback_ambiguity_findings(
                snapshot_a,
                snapshot_b,
                _TRANSACTION_SINGLETON_FALLBACK_KEY_COLUMNS,
            )
        )
        return findings

    def _transaction_singleton_match_frames(
        self,
        snapshot_a: pl.DataFrame,
        snapshot_b: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Return no-ID transaction rows eligible for singleton fallback matching."""
        singleton_keys = self._transaction_singleton_match_keys(snapshot_a, snapshot_b)
        if singleton_keys.is_empty():
            return snapshot_a.head(0), snapshot_b.head(0)
        return (
            snapshot_a.join(
                singleton_keys,
                on=list(_TRANSACTION_SINGLETON_FALLBACK_KEY_COLUMNS),
                how="inner",
            ),
            snapshot_b.join(
                singleton_keys,
                on=list(_TRANSACTION_SINGLETON_FALLBACK_KEY_COLUMNS),
                how="inner",
            ),
        )

    @staticmethod
    def _transaction_singleton_match_keys(
        snapshot_a: pl.DataFrame,
        snapshot_b: pl.DataFrame,
    ) -> pl.DataFrame:
        """Return exact fallback keys with one transaction row on each side."""
        key_columns = list(_TRANSACTION_SINGLETON_FALLBACK_KEY_COLUMNS)
        counts_a = snapshot_a.group_by(key_columns).len(name="snapshot_a_count")
        counts_b = snapshot_b.group_by(key_columns).len(name="snapshot_b_count")
        return (
            counts_a.join(counts_b, on=key_columns, how="inner")
            .filter(
                (pl.col("snapshot_a_count") == 1)
                & (pl.col("snapshot_b_count") == 1)
            )
            .select(key_columns)
        )

    @staticmethod
    def _without_transaction_singleton_matches(
        snapshot: pl.DataFrame,
        singleton_rows: pl.DataFrame,
    ) -> pl.DataFrame:
        """Return transaction rows excluding singleton fallback matched rows."""
        if singleton_rows.is_empty():
            return snapshot
        return snapshot.join(
            singleton_rows.select(list(_TRANSACTION_SINGLETON_FALLBACK_KEY_COLUMNS)),
            on=list(_TRANSACTION_SINGLETON_FALLBACK_KEY_COLUMNS),
            how="anti",
        )

    def _transaction_presence_findings_for_side(
        self,
        source_snapshot: pl.DataFrame,
        other_snapshot: pl.DataFrame,
        key_columns: tuple[str, ...],
        code: str,
        message: str,
        transaction_match_status: TransactionMatchStatus,
        snapshot_side: str,
        portfolio_periods: pl.DataFrame | None,
        return_denominators: Mapping[tuple[object, object, object], float] | None,
    ) -> list[Finding]:
        """Return transaction findings for rows present in only one snapshot."""
        other_keys = other_snapshot.select(key_columns).unique()
        unmatched = source_snapshot.join(
            other_keys,
            on=list(key_columns),
            how="anti",
        )
        if unmatched.is_empty():
            return []

        rows = sorted(
            unmatched.iter_rows(named=True),
            key=lambda row: self._sortable_key(
                tuple(row[column] for column in key_columns)
            ),
        )
        findings: list[Finding] = []
        for row in rows:
            findings.extend(
                self._transaction_presence_findings_for_row(
                    row,
                    code,
                    message,
                    transaction_match_status,
                    snapshot_side,
                    portfolio_periods,
                    return_denominators,
                )
            )
        return findings

    def _transaction_presence_findings_for_row(
        self,
        row: Mapping[str, object],
        code: str,
        message: str,
        transaction_match_status: TransactionMatchStatus,
        snapshot_side: str,
        portfolio_periods: pl.DataFrame | None,
        return_denominators: Mapping[tuple[object, object, object], float] | None,
    ) -> list[Finding]:
        """Return dated transaction add/drop findings from one source row."""
        amount_finding = self._transaction_presence_finding(
            row,
            code,
            message,
            transaction_match_status,
            snapshot_side,
            portfolio_periods,
            return_denominators,
            pc_cols.AMOUNT,
        )
        component_findings = [
            self._transaction_presence_finding(
                row,
                _TRANSACTION_COMPARE_COLUMNS[column],
                message,
                transaction_match_status,
                snapshot_side,
                portfolio_periods,
                return_denominators,
                column,
            )
            for column in (pc_cols.QUANTITY, pc_cols.PRICE, pc_cols.COMMISSION)
            if self._transaction_presence_field_delta(row.get(column), snapshot_side)
            not in (None, 0.0)
        ]
        return [amount_finding, *component_findings]

    def _transaction_presence_finding(
        self,
        row: Mapping[str, object],
        code: str,
        message: str,
        transaction_match_status: TransactionMatchStatus,
        snapshot_side: str,
        portfolio_periods: pl.DataFrame | None,
        return_denominators: Mapping[tuple[object, object, object], float] | None,
        source_column: str,
    ) -> Finding:
        """Return a dated transaction add/drop finding for one source field."""
        from_date, thru_date = period_context_for_dated_evidence(
            row,
            pc_cols.TRANSACTIONS,
            portfolio_periods,
        )
        portfolio_id = row.get(pc_cols.PORTFOLIO_ID)
        source_value = row.get(source_column)
        delta = self._transaction_presence_field_delta(source_value, snapshot_side)
        snapshot_a_value = source_value if snapshot_side == "a" else None
        snapshot_b_value = source_value if snapshot_side == "b" else None
        return_denominator = self._return_denominator(
            row,
            pc_cols.TRANSACTIONS,
            portfolio_id,
            from_date,
            thru_date,
            return_denominators,
        )
        impact_policy = self._impact_policy(pc_cols.TRANSACTIONS, source_column)
        transaction_impact_policy = self._transaction_impact_policy(
            row,
            pc_cols.TRANSACTIONS,
            source_column,
        )
        return Finding(
            code=code,
            severity=SEVERITY_INFORMATIONAL,
            confidence=CONFIDENCE_HIGH,
            dataset=pc_cols.TRANSACTIONS,
            evidence_role=self._evidence_role(
                code,
                pc_cols.TRANSACTIONS,
                source_column,
            ),
            portfolio_id=portfolio_id,
            security_id=row.get(pc_cols.SECURITY_ID),
            from_date=from_date,
            thru_date=thru_date,
            input_date=self._input_date(row, pc_cols.TRANSACTIONS),
            source_file=self._source_file(pc_cols.TRANSACTIONS),
            source_column=source_column,
            transaction_code=self._transaction_code(row, pc_cols.TRANSACTIONS),
            transaction_category=self._transaction_category(row, pc_cols.TRANSACTIONS),
            cash_flow_sign=self._transaction_cash_flow_sign(row, pc_cols.TRANSACTIONS),
            performance_flow_sign=self._transaction_performance_flow_sign(
                row,
                pc_cols.TRANSACTIONS,
            ),
            transaction_semantics_source=self._transaction_semantics_source(
                row,
                pc_cols.TRANSACTIONS,
            ),
            transaction_match_status=transaction_match_status,
            impact_policy=impact_policy,
            transaction_impact_policy=transaction_impact_policy,
            transaction_impact_diagnostic=self._transaction_impact_diagnostic(
                row,
                pc_cols.TRANSACTIONS,
                source_column,
                portfolio_id,
                from_date,
                thru_date,
                return_denominator,
            ),
            transaction_impact_diagnostic_estimate=(
                self._transaction_impact_diagnostic_estimate(
                    row,
                    pc_cols.TRANSACTIONS,
                    source_column,
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
            message=message,
        )

    @staticmethod
    def _transaction_presence_field_delta(
        value: object,
        snapshot_side: str,
    ) -> float | None:
        """Return B-minus-A field delta for a transaction presence finding."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        amount_float = float(value)
        if snapshot_side == "a":
            return -amount_float
        if snapshot_side == "b":
            return amount_float
        return None

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
                impact_policy = self._impact_policy(dataset, column)
                transaction_impact_policy = self._transaction_impact_policy(
                    row,
                    dataset,
                    column,
                )
                findings.append(
                    Finding(
                        code=compare_columns[column],
                        severity=SEVERITY_INFORMATIONAL,
                        confidence=CONFIDENCE_HIGH,
                        dataset=dataset,
                        evidence_role=self._changed_value_evidence_role(
                            compare_columns[column],
                            dataset,
                            column,
                            impact_policy,
                            transaction_impact_policy,
                        ),
                        portfolio_id=row.get(pc_cols.PORTFOLIO_ID),
                        security_id=row.get(pc_cols.SECURITY_ID),
                        from_date=row.get(pc_cols.FROM_DATE),
                        thru_date=row.get(pc_cols.THRU_DATE),
                        input_date=self._input_date(row, dataset),
                        source_file=source_file,
                        source_column=column,
                        transaction_code=self._transaction_code(row, dataset),
                        transaction_category=self._transaction_category(row, dataset),
                        cash_flow_sign=self._transaction_cash_flow_sign(row, dataset),
                        performance_flow_sign=self._transaction_performance_flow_sign(
                            row,
                            dataset,
                        ),
                        transaction_semantics_source=(
                            self._transaction_semantics_source(row, dataset)
                        ),
                        impact_policy=impact_policy,
                        transaction_impact_policy=transaction_impact_policy,
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
        return_weights: (
            Mapping[tuple[object, object, object, object], float] | None
        ) = None,
        transaction_match_status: TransactionMatchStatus | None = None,
        transaction_fallback_periods: (
            Mapping[object, tuple[object | None, object | None]] | None
        ) = None,
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
                transaction_fallback_periods,
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
                    impact_policy = self._impact_policy(dataset, column)
                    transaction_impact_policy = self._transaction_impact_policy(
                        row,
                        dataset,
                        column,
                    )
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
                            evidence_role=self._changed_value_evidence_role(
                                compare_columns[column],
                                dataset,
                                column,
                                impact_policy,
                                transaction_impact_policy,
                            ),
                            portfolio_id=portfolio_id,
                            security_id=row.get(pc_cols.SECURITY_ID),
                            from_date=from_date,
                            thru_date=thru_date,
                            input_date=self._input_date(row, dataset),
                            source_file=source_file,
                            source_column=column,
                            transaction_code=self._transaction_code(row, dataset),
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
                            impact_policy=impact_policy,
                            transaction_impact_policy=transaction_impact_policy,
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
                            return_weight=self._return_weight(
                                row,
                                dataset,
                                portfolio_id,
                                from_date,
                                thru_date,
                                return_weights,
                            ),
                            impact_input_value=self._impact_input_value(
                                row,
                                dataset,
                                column,
                                impact_policy,
                            ),
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
        PerformanceComparison._validate_complete_keys(
            frame,
            key_columns,
            dataset,
            snapshot_label,
        )

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
    def _validate_complete_keys(
        frame: pl.DataFrame,
        key_columns: tuple[str, ...],
        dataset: str,
        snapshot_label: str,
    ) -> None:
        """Raise if a normalized frame has missing comparison key values."""
        null_key_rows = frame.filter(
            pl.any_horizontal(pl.col(column).is_null() for column in key_columns)
        )
        if not null_key_rows.is_empty():
            null_key = null_key_rows.select(key_columns).row(0, named=True)
            key_names = ", ".join(key_columns)
            raise PpaError(
                (
                    f"{dataset} contains missing {snapshot_label} comparison "
                    f"key values for key columns {key_names}: {null_key}"
                ),
                112,
            )

    def _transaction_fallback_ambiguity_findings(
        self,
        snapshot_a: pl.DataFrame,
        snapshot_b: pl.DataFrame,
        key_columns: tuple[str, ...],
    ) -> list[Finding]:
        """Return ambiguity diagnostics for duplicate fallback transaction keys."""
        duplicate_keys = self._duplicate_transaction_fallback_keys(
            snapshot_a,
            snapshot_b,
            key_columns,
        )
        if duplicate_keys.is_empty():
            return []

        source_file = self._source_file(pc_cols.TRANSACTIONS)
        findings: list[Finding] = []
        for row in duplicate_keys.iter_rows(named=True):
            row_key = tuple(row[column] for column in key_columns)
            row_context = dict(zip(key_columns, row_key, strict=True))
            findings.append(
                Finding(
                    code=PC_TXN_AMBIG,
                    severity=SEVERITY_INFORMATIONAL,
                    confidence=CONFIDENCE_HIGH,
                    dataset=pc_cols.TRANSACTIONS,
                    evidence_role=CONTEXT,
                    portfolio_id=row_context.get(pc_cols.PORTFOLIO_ID),
                    security_id=row_context.get(pc_cols.SECURITY_ID),
                    from_date=row_context.get(pc_cols.FROM_DATE),
                    thru_date=row_context.get(pc_cols.THRU_DATE),
                    input_date=self._input_date(row_context, pc_cols.TRANSACTIONS),
                    source_file=source_file,
                    transaction_match_status=(
                        TRANSACTION_MATCH_STATUS_AMBIGUOUS_FALLBACK_MATCH
                    ),
                    snapshot_a_value=row.get("snapshot_a_count"),
                    snapshot_b_value=row.get("snapshot_b_count"),
                    message=(
                        "Duplicate strict fallback transaction keys make row "
                        "pairing ambiguous; review as separate same-key activity."
                    ),
                )
            )
        return findings

    @staticmethod
    def _duplicate_transaction_fallback_keys(
        snapshot_a: pl.DataFrame,
        snapshot_b: pl.DataFrame,
        key_columns: tuple[str, ...],
    ) -> pl.DataFrame:
        """Return fallback keys duplicated in either transaction snapshot."""
        counts_a = snapshot_a.group_by(list(key_columns)).len(name="snapshot_a_count")
        counts_b = snapshot_b.group_by(list(key_columns)).len(name="snapshot_b_count")
        duplicate_keys = counts_a.join(
            counts_b,
            on=list(key_columns),
            how="full",
            coalesce=True,
        )
        if duplicate_keys.is_empty():
            return duplicate_keys
        return duplicate_keys.with_columns(
            pl.col("snapshot_a_count").fill_null(0),
            pl.col("snapshot_b_count").fill_null(0),
        ).filter(
            (pl.col("snapshot_a_count") > 1) | (pl.col("snapshot_b_count") > 1)
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
    def _transaction_code(row: dict[str, object], dataset: str) -> object | None:
        """Return source transaction code context for transaction rows."""
        if dataset != pc_cols.TRANSACTIONS:
            return None
        return row.get(pc_cols.TRANSACTION_CODE)

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
        source_column: str,
    ) -> object | None:
        """Return YAML-configured transaction impact policy for a row."""
        if dataset != pc_cols.TRANSACTIONS:
            return None
        if _field_roles.is_input_component(dataset, source_column):
            policy = self._transaction_impact_policies.get(source_column)
            if policy is not None:
                return policy.finding_label
        if source_column in {
            _TRANSACTION_QUANTITY_KEY,
            _TRANSACTION_PRICE_KEY,
            _TRANSACTION_COMMISSION_KEY,
        }:
            policy = self._transaction_impact_policies.get(source_column)
            if policy is not None:
                return policy.finding_label
        if (
            self._specification.comparison_level == SECURITY_COMPARISON_LEVEL
            and source_column == pc_cols.AMOUNT
            and row.get(pc_cols.TRANSACTION_CATEGORY)
            in {TRANSACTION_CATEGORY_BUY, TRANSACTION_CATEGORY_SELL}
            and row.get(pc_cols.PERFORMANCE_FLOW_SIGN)
            == TRANSACTION_PERFORMANCE_FLOW_SIGN_PERFORMANCE
        ):
            policy = self._security_return_impact_policies.get(pc_cols.TRANSACTIONS)
            if policy is not None:
                return policy.finding_label
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
                dataset in {pc_cols.HOLDINGS, pc_cols.TRANSACTIONS, pc_cols.CASH}
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
        portfolio_id: object | None,
        from_date: object | None,
        thru_date: object | None,
        return_weights: Mapping[tuple[object, object, object, object], float] | None,
    ) -> float | None:
        """Return snapshot A security weight for approximate return impacts."""
        if dataset == pc_cols.HOLDINGS:
            if (
                portfolio_id is None
                or from_date is None
                or thru_date is None
                or return_weights is None
            ):
                return None
            return return_weights.get(
                (
                    portfolio_id,
                    row.get(pc_cols.SECURITY_ID),
                    from_date,
                    thru_date,
                )
            )
        if dataset != pc_cols.SECURITY_PERFORMANCE:
            return None
        value = row.get(pc_cols.WEIGHT)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    @staticmethod
    def _impact_input_value(
        row: Mapping[str, object],
        dataset: str,
        source_column: str,
        impact_policy: object | None,
    ) -> float | None:
        """Return method-specific input value used for impact estimates."""
        if (
            dataset != pc_cols.HOLDINGS
            or source_column != pc_cols.QUANTITY
            or impact_policy != IMPACT_POLICY_HOLDING_QUANTITY_UNIT_MARKET_VALUE
        ):
            return None
        quantity = row.get(pc_cols.QUANTITY)
        market_value = row.get(pc_cols.MARKET_VALUE)
        if (
            isinstance(quantity, bool)
            or isinstance(market_value, bool)
            or not isinstance(quantity, (int, float))
            or not isinstance(market_value, (int, float))
            or float(quantity) == 0.0
        ):
            return None
        return float(market_value) / float(quantity)

    @staticmethod
    def _changed_value_contexts(
        row: Mapping[str, object],
        dataset: str,
        portfolio_periods: pl.DataFrame | None,
        security_periods: pl.DataFrame | None,
        transaction_fallback_periods: (
            Mapping[object, tuple[object | None, object | None]] | None
        ) = None,
    ) -> list[tuple[object | None, object | None, object | None]]:
        """Return portfolio and period contexts for changed-value findings."""
        del security_periods

        period_context = period_context_for_dated_evidence(
            row,
            dataset,
            portfolio_periods,
        )
        if (
            dataset == pc_cols.TRANSACTIONS
            and period_context == (None, None)
            and transaction_fallback_periods is not None
        ):
            fallback_period = transaction_fallback_periods.get(row.get(pc_cols.PORTFOLIO_ID))
            if fallback_period is not None:
                return [(row.get(pc_cols.PORTFOLIO_ID), fallback_period[0], fallback_period[1])]
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

    def _single_changed_portfolio_return_periods(
        self,
    ) -> dict[object, tuple[object | None, object | None]]:
        """Return one changed portfolio-return period per portfolio when unambiguous."""
        snapshot_a = self._portfolio_loader.load("a")
        snapshot_b = self._portfolio_loader.load("b")
        if pc_cols.PORTFOLIO_RETURN not in snapshot_a.columns:
            return {}
        if pc_cols.PORTFOLIO_RETURN not in snapshot_b.columns:
            return {}

        joined = snapshot_a.join(
            snapshot_b,
            on=list(_PORTFOLIO_KEY_COLUMNS),
            how="inner",
            suffix="_b",
        )
        changed_periods_by_portfolio: dict[object, list[tuple[object, object]]] = {}
        for row in joined.iter_rows(named=True):
            delta = self._numeric_delta(
                row[pc_cols.PORTFOLIO_RETURN],
                row[f"{pc_cols.PORTFOLIO_RETURN}_b"],
            )
            if delta is None or abs(delta) <= self._tolerance(pc_cols.PORTFOLIO_RETURN):
                continue
            portfolio_id = row[pc_cols.PORTFOLIO_ID]
            period = (row[pc_cols.FROM_DATE], row[pc_cols.THRU_DATE])
            changed_periods_by_portfolio.setdefault(portfolio_id, []).append(period)

        return {
            portfolio_id: periods[0]
            for portfolio_id, periods in changed_periods_by_portfolio.items()
            if len(periods) == 1
        }

    def _portfolio_periods(self) -> pl.DataFrame:
        """Return portfolio period rows from both snapshots for evidence linking."""
        return portfolio_periods_from_snapshots(
            self._portfolio_loader.load("a"),
            self._portfolio_loader.load("b"),
        )

    def _security_period_return_weights(
        self,
    ) -> dict[tuple[object, object, object, object], float]:
        """Return snapshot A security weights keyed by portfolio/security period."""
        snapshot_a = self._security_loader.load("a")
        if snapshot_a is None or pc_cols.WEIGHT not in snapshot_a.columns:
            return {}
        weights: dict[tuple[object, object, object, object], float] = {}
        for row in snapshot_a.iter_rows(named=True):
            value = row.get(pc_cols.WEIGHT)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            key = (
                row.get(pc_cols.PORTFOLIO_ID),
                row.get(pc_cols.SECURITY_ID),
                row.get(pc_cols.FROM_DATE),
                row.get(pc_cols.THRU_DATE),
            )
            weights[key] = float(value)
        return weights

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
    def _key_finding(
        code: str,
        row_key: tuple[object, ...],
        key_columns: tuple[str, ...],
        dataset: str,
        source_file: str | None,
        message: str,
        transaction_match_status: TransactionMatchStatus | None = None,
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
            input_date=PerformanceComparison._input_date(row_context, dataset),
            source_file=source_file,
            transaction_match_status=transaction_match_status,
            message=message,
        )

    @staticmethod
    def _input_date(row: Mapping[str, object], dataset: str) -> object | None:
        """Return the date represented by an input row."""
        date_column_by_dataset = {
            pc_cols.HOLDINGS: pc_cols.HOLDING_DATE,
            pc_cols.TRANSACTIONS: pc_cols.TRANSACTION_DATE,
            pc_cols.FX_RATES: pc_cols.RATE_DATE,
            pc_cols.CASH: pc_cols.CASH_DATE,
        }
        date_column = date_column_by_dataset.get(dataset)
        if date_column is None:
            return None
        return row.get(date_column)

    @staticmethod
    def _evidence_role(
        code: str,
        dataset: str,
        source_column: str | None,
    ) -> EvidenceRole:
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
        if dataset == pc_cols.HOLDINGS and source_column == pc_cols.COST:
            return CONTEXT
        if dataset == pc_cols.TRANSACTIONS and source_column == pc_cols.COMMISSION:
            return CONTEXT
        if dataset in _DIRECT_INPUT_DATASETS:
            return DIRECT_INPUT
        return CONTEXT

    def _changed_value_evidence_role(
        self,
        code: str,
        dataset: str,
        source_column: str,
        impact_policy: object | None,
        transaction_impact_policy: object | None,
    ) -> EvidenceRole:
        """Return the evidence role for one changed-value finding."""
        if _field_roles.is_context(dataset, source_column):
            return CONTEXT
        return self._evidence_role(code, dataset, source_column)

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

    def _impact_policy(
        self,
        dataset: str,
        source_column: str,
    ) -> str | None:
        """Return the YAML-selected non-transaction impact policy for a field."""
        evidence_only_policy = self._evidence_only_impact_policies.get(
            (dataset, source_column)
        )
        if evidence_only_policy is not None:
            return evidence_only_policy
        if dataset == pc_cols.HOLDINGS:
            if source_column == pc_cols.PRICE:
                policy = self._price_impact_policies.get(source_column)
                if policy is not None:
                    return policy
            policy = self._holding_impact_policies.get(source_column)
            if policy is not None:
                return policy
        if dataset == pc_cols.CASH:
            policy = self._cash_impact_policies.get(source_column)
            if policy is not None:
                return policy
        if dataset == pc_cols.FX_RATES:
            policy = self._fx_rate_impact_policies.get(source_column)
            if policy is not None:
                return policy
        if _field_roles.is_reported_performance_component(dataset, source_column):
            return None
        policy = self._contribution_impact_policy(dataset, source_column)
        if policy is not None:
            return policy
        return None
