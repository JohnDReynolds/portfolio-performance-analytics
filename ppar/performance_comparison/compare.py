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
from ppar.performance_comparison import columns as pc_cols
from ppar.performance_comparison.cash import CashLoader
from ppar.performance_comparison.findings import (
    CONFIDENCE_HIGH,
    CONTEXT,
    DIRECT_INPUT,
    RELATED_OUTPUT,
    TARGET_OUTPUT,
    TRANSACTION_IMPACT_POLICY_EXTERNAL_FLOW_EVIDENCE_ONLY,
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
_EXTERNAL_FLOW_KEY: Final[str] = "external_flow"
_METHOD_KEY: Final[str] = "method"
_EVIDENCE_ONLY_METHOD: Final[str] = "evidence_only"
_RESERVED_EXTERNAL_FLOW_METHODS: Final[frozenset[str]] = frozenset(
    {
        "modified_dietz",
        "subperiod_linked",
        "unweighted_flow_delta",
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
                            transaction_impact_policy=(
                                self._transaction_impact_policy(row, dataset)
                            ),
                            snapshot_a_value=snapshot_a_value,
                            snapshot_b_value=snapshot_b_value,
                            delta_b_minus_a=delta,
                            return_denominator=self._return_denominator(
                                row,
                                dataset,
                                portfolio_id,
                                from_date,
                                thru_date,
                                return_denominators,
                            ),
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
        if row.get(pc_cols.PERFORMANCE_FLOW_SIGN) != TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL:
            return None
        return self._transaction_impact_policies.get(_EXTERNAL_FLOW_KEY)

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
    def _key_finding(
        code: str,
        row_key: tuple[object, ...],
        key_columns: tuple[str, ...],
        dataset: str,
        source_file: str | None,
        message: str,
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


def _transaction_impact_policies(
    specification: PerformanceComparisonSpecification,
) -> dict[str, str]:
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

    unsupported_keys = set(methods_value) - {_EXTERNAL_FLOW_KEY}
    if unsupported_keys:
        unsupported = ", ".join(sorted(str(key) for key in unsupported_keys))
        raise PpaError(
            (
                f"{specification.path}: unsupported "
                f"{_TRANSACTION_IMPACT_METHODS_KEY} keys: {unsupported}."
            ),
            504,
        )

    policies: dict[str, str] = {}
    external_flow_value = methods_value.get(_EXTERNAL_FLOW_KEY)
    if external_flow_value is None:
        return policies
    if not isinstance(external_flow_value, dict):
        raise PpaError(
            (
                f"{specification.path}: "
                f"{_TRANSACTION_IMPACT_METHODS_KEY}.{_EXTERNAL_FLOW_KEY} "
                "must be a mapping."
            ),
            504,
        )

    method = external_flow_value.get(_METHOD_KEY)
    if method is None:
        raise PpaError(
            (
                f"{specification.path}: "
                f"{_TRANSACTION_IMPACT_METHODS_KEY}.{_EXTERNAL_FLOW_KEY}."
                f"{_METHOD_KEY} is required and must be "
                f"{_EVIDENCE_ONLY_METHOD!r} until an external-flow impact "
                "formula is explicitly supported."
            ),
            504,
        )
    if method != _EVIDENCE_ONLY_METHOD:
        raise PpaError(
            (
                f"{specification.path}: "
                f"{_TRANSACTION_IMPACT_METHODS_KEY}.{_EXTERNAL_FLOW_KEY}."
                f"{_METHOD_KEY} must be {_EVIDENCE_ONLY_METHOD!r} until an "
                "external-flow impact formula is explicitly supported."
            ),
            504,
        )

    policies[_EXTERNAL_FLOW_KEY] = TRANSACTION_IMPACT_POLICY_EXTERNAL_FLOW_EVIDENCE_ONLY
    return policies
