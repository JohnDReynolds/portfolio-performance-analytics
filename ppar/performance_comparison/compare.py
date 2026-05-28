"""Compare normalized performance snapshot datasets."""

from __future__ import annotations

# Python imports
import datetime as dt
from typing import Any, Final, cast

# Third-party imports
import polars as pl

# Project imports
from ppar.performance_comparison import columns as pc_cols
from ppar.performance_comparison.findings import (
    CONFIDENCE_HIGH,
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
    PC_REF_CLASS,
    PC_REF_ID,
    SEVERITY_INFORMATIONAL,
    SEVERITY_MATERIAL,
    Finding,
)
from ppar.performance_comparison.portfolio_performance import PortfolioPerformanceLoader
from ppar.performance_comparison.security_performance import SecurityPerformanceLoader
from ppar.performance_comparison.security_master import SecurityMasterLoader
from ppar.performance_comparison.specification import PerformanceComparisonSpecification

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
_DEFAULT_TOLERANCES: Final[dict[str, float]] = {
    "return": 1e-6,
    "contribution": 1e-6,
    "weight": 1e-6,
    "market_value": 0.01,
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
}


class PerformanceComparison:
    """Compare performance snapshots and return finding records.

    Attributes:
        _specification: Parsed comparison specification.
        _portfolio_loader: Loader for normalized portfolio performance rows.
        _security_loader: Loader for normalized security performance rows.
        _security_master_loader: Loader for normalized security master rows.
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
        return findings

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
        rows_a = self._row_keys(snapshot_a, key_columns)
        rows_b = self._row_keys(snapshot_b, key_columns)
        findings: list[Finding] = []

        for row_key in sorted(rows_b - rows_a, key=self._sortable_key):
            findings.append(
                self._key_finding(
                    add_code,
                    row_key,
                    dataset,
                    add_message,
                )
            )
        for row_key in sorted(rows_a - rows_b, key=self._sortable_key):
            findings.append(
                self._key_finding(
                    drop_code,
                    row_key,
                    dataset,
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
                        portfolio_id=row.get(pc_cols.PORTFOLIO_ID),
                        security_id=row.get(pc_cols.SECURITY_ID),
                        from_date=row.get(pc_cols.FROM_DATE),
                        thru_date=row.get(pc_cols.THRU_DATE),
                        source_column=column,
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
                delta = self._numeric_delta(snapshot_a_value, snapshot_b_value)
                if delta is None:
                    continue
                if abs(delta) <= self._tolerance(column):
                    continue
                findings.append(
                    Finding(
                        code=compare_columns[column],
                        severity=SEVERITY_MATERIAL,
                        confidence=CONFIDENCE_HIGH,
                        dataset=dataset,
                        portfolio_id=row[pc_cols.PORTFOLIO_ID],
                        security_id=row.get(pc_cols.SECURITY_ID),
                        from_date=row[pc_cols.FROM_DATE],
                        thru_date=row[pc_cols.THRU_DATE],
                        source_column=column,
                        snapshot_a_value=snapshot_a_value,
                        snapshot_b_value=snapshot_b_value,
                        delta_b_minus_a=delta,
                        message=f"{dataset} {column!r} changed.",
                    )
                )
        return findings

    @staticmethod
    def _row_keys(frame: pl.DataFrame, key_columns: tuple[str, ...]) -> set[tuple[object, ...]]:
        """Return key tuples from a normalized comparison frame."""
        return set(frame.select(key_columns).iter_rows())

    @staticmethod
    def _sortable_key(row_key: tuple[object, ...]) -> tuple[str, ...]:
        """Return a deterministic string sort key for finding output."""
        return tuple(str(value) for value in row_key)

    @staticmethod
    def _key_finding(
        code: str,
        row_key: tuple[object, ...],
        dataset: str,
        message: str,
    ) -> Finding:
        """Return a row-presence finding from a portfolio key tuple."""
        portfolio_id: object | None = None
        security_id: object | None = None
        from_date: object | None = None
        thru_date: object | None = None
        if len(row_key) == 1:
            security_id = row_key[0]
        elif len(row_key) == 3:
            portfolio_id, from_date, thru_date = row_key
        elif len(row_key) == 4:
            portfolio_id, security_id, from_date, thru_date = row_key

        return Finding(
            code=code,
            severity=SEVERITY_INFORMATIONAL,
            confidence=CONFIDENCE_HIGH,
            dataset=dataset,
            portfolio_id=portfolio_id,
            security_id=security_id,
            from_date=from_date,
            thru_date=thru_date,
            message=message,
        )

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
