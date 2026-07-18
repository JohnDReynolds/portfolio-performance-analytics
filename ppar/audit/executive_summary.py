"""Build the two-table Audit Executive Summary.

The summary contains quantities only. Performance quantities use mutually
exclusive status buckets, while Data Issues quantities are grouped by stable
issue type and sorted from largest to smallest.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, TypedDict

import polars as pl

import ppar.utilities as util
from ppar.errors import PpaError
from ppar.audit import schema as pc_cols
from ppar.audit.data_issues import checks as data_issue_checks
from ppar.audit.data_issues.vocabulary import DataIssueType
from ppar.audit.portfolio_performance import PortfolioPerformanceLoader
from ppar.audit.security_performance import SecurityPerformanceLoader
from ppar.audit.specification import (
    AuditSpecification,
    PORTFOLIO_COMPARISON_LEVEL,
    SECURITY_COMPARISON_LEVEL,
)

SUMMARY_SECTION: Final[str] = "summary_section"
SUMMARY_LABEL: Final[str] = "summary_label"
TOTAL_QUANTITY: Final[str] = "total_quantity"
NO_PERFORMANCE_DIFFERENCES: Final[str] = "no_performance_differences"
FULLY_EXPLAINED_DIFFERENCES: Final[str] = "fully_explained_differences"
PARTLY_EXPLAINED_DIFFERENCES: Final[str] = "partly_explained_differences"
UNEXPLAINED_DIFFERENCES: Final[str] = "unexplained_differences"
SETUP_INCOMPLETE: Final[str] = "setup_incomplete"
DATA_ISSUE_QUANTITY: Final[str] = "data_issue_quantity"

EXECUTIVE_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    SUMMARY_SECTION,
    SUMMARY_LABEL,
    TOTAL_QUANTITY,
    NO_PERFORMANCE_DIFFERENCES,
    FULLY_EXPLAINED_DIFFERENCES,
    PARTLY_EXPLAINED_DIFFERENCES,
    UNEXPLAINED_DIFFERENCES,
    SETUP_INCOMPLETE,
    DATA_ISSUE_QUANTITY,
)

PERFORMANCE_SECTION: Final[str] = "Performance Differences"
DATA_ISSUES_SECTION: Final[str] = "Data Issues"
PERFORMANCE_TABLE_CAPTION: Final[str] = "Performance Differences Summary"
DATA_ISSUES_TABLE_CAPTION: Final[str] = "Data Issues Summary"
PERFORMANCE_HEADERS: Final[tuple[str, ...]] = (
    "",
    "Total Quantity",
    "No Performance Differences",
    "Fully Explained Differences",
    "Partly Explained Differences",
    "Unexplained Differences",
    "Setup Incomplete",
)
DATA_ISSUES_HEADERS: Final[tuple[str, ...]] = ("Issue Type", "Quantity")

_NO_DIFFERENCE = "No Performance Differences"
_FULLY_EXPLAINED = "Fully Explained"
_PARTLY_EXPLAINED = "Partly Explained"
_UNEXPLAINED = "Unexplained"
_SETUP_INCOMPLETE = "Missing YAML Specifications"
_STATUS_PRECEDENCE: Final[Mapping[str, int]] = {
    _NO_DIFFERENCE: 0,
    _FULLY_EXPLAINED: 1,
    _PARTLY_EXPLAINED: 2,
    _UNEXPLAINED: 3,
    _SETUP_INCOMPLETE: 4,
}
_PERFORMANCE_VALUE_COLUMNS: Final[tuple[str, ...]] = (
    TOTAL_QUANTITY,
    NO_PERFORMANCE_DIFFERENCES,
    FULLY_EXPLAINED_DIFFERENCES,
    PARTLY_EXPLAINED_DIFFERENCES,
    UNEXPLAINED_DIFFERENCES,
    SETUP_INCOMPLETE,
)


class ExecutiveSummaryDisplayTable(TypedDict):
    """One rendered Executive Summary quantity table."""

    columns: list[str]
    rows: list[list[str]]


@dataclass(frozen=True)
class ExecutiveSummaryContext:
    """Evaluated comparison scope used to count unchanged review units.

    Attributes:
        comparison_level: Portfolio or security comparison level.
        evaluated_unit_keys: Union of primary performance keys in both snapshots.
    """

    comparison_level: str
    evaluated_unit_keys: tuple[tuple[object, ...], ...] = ()


def executive_summary_context(
    comparison_path: util.PathLike | None,
    comparison_level: str,
) -> ExecutiveSummaryContext:
    """Return the evaluated primary-performance scope for the summary.

    Args:
        comparison_path: Optional Audit YAML path.
        comparison_level: Portfolio or security comparison level.

    Returns:
        Comparison context. When no YAML path is available, the evaluated scope
        is left empty and the summary uses the known review rows only.
    """
    _assert_supported_level(comparison_level)
    if comparison_path is None:
        return ExecutiveSummaryContext(comparison_level=comparison_level)
    specification = AuditSpecification(
        comparison_path,
        comparison_level=comparison_level,
    )
    frames = _primary_performance_frames(specification, comparison_level)
    key_columns = _unit_key_columns(comparison_level)
    keys = {
        tuple(row[column] for column in key_columns)
        for frame in frames
        for row in frame.select(key_columns).iter_rows(named=True)
    }
    return ExecutiveSummaryContext(
        comparison_level=comparison_level,
        evaluated_unit_keys=tuple(sorted(keys, key=_sortable_key)),
    )


def executive_summary_table(
    primary_changes: pl.DataFrame,
    data_issues: pl.DataFrame,
    *,
    context: ExecutiveSummaryContext,
) -> pl.DataFrame:
    """Return the canonical quantitative Executive Summary table.

    Args:
        primary_changes: Reconciled Performance Differences table.
        data_issues: Existing canonical Data Issues table.
        context: Evaluated primary-performance scope.

    Returns:
        Two performance quantity rows followed by Data Issues type quantities.

    Raises:
        PpaError: If a review status, issue type, or comparison level is unknown.
    """
    _assert_supported_level(context.comparison_level)
    changed_statuses = _changed_unit_statuses(primary_changes, context.comparison_level)
    evaluated_units = set(context.evaluated_unit_keys) | set(changed_statuses)
    unit_statuses = {
        key: changed_statuses.get(key, _NO_DIFFERENCE) for key in evaluated_units
    }
    rows = [
        _performance_row(
            "Portfolios",
            _portfolio_statuses(unit_statuses),
        ),
        _performance_row(
            _period_label(context.comparison_level),
            tuple(unit_statuses.values()),
        ),
        *_data_issue_rows(data_issues),
    ]
    return pl.DataFrame(rows, schema=_summary_schema())


def executive_summary_display_tables(
    table: pl.DataFrame,
) -> dict[str, ExecutiveSummaryDisplayTable]:
    """Return the two deterministic display payloads for HTML/XLSX parity."""
    performance = table.filter(pl.col(SUMMARY_SECTION) == PERFORMANCE_SECTION)
    data_issues = table.filter(pl.col(SUMMARY_SECTION) == DATA_ISSUES_SECTION)
    return {
        PERFORMANCE_TABLE_CAPTION: {
            "columns": list(PERFORMANCE_HEADERS),
            "rows": [
                [
                    str(row[SUMMARY_LABEL]),
                    *[str(row[column]) for column in _PERFORMANCE_VALUE_COLUMNS],
                ]
                for row in performance.iter_rows(named=True)
            ],
        },
        DATA_ISSUES_TABLE_CAPTION: {
            "columns": list(DATA_ISSUES_HEADERS),
            "rows": [
                [str(row[SUMMARY_LABEL]), str(row[DATA_ISSUE_QUANTITY])]
                for row in data_issues.iter_rows(named=True)
            ],
        },
    }


def _primary_performance_frames(
    specification: AuditSpecification,
    comparison_level: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return normalized Snapshot A and B primary-performance frames."""
    if comparison_level == PORTFOLIO_COMPARISON_LEVEL:
        portfolio_loader = PortfolioPerformanceLoader(specification)
        return portfolio_loader.load("a"), portfolio_loader.load("b")
    security_loader = SecurityPerformanceLoader(specification)
    snapshot_a = security_loader.load("a")
    snapshot_b = security_loader.load("b")
    if snapshot_a is None or snapshot_b is None:
        raise PpaError("Security performance input is unavailable.", None)
    return snapshot_a, snapshot_b


def _changed_unit_statuses(
    primary_changes: pl.DataFrame,
    comparison_level: str,
) -> dict[tuple[object, ...], str]:
    """Return one validated review status per changed primary unit."""
    key_columns = _unit_key_columns(comparison_level)
    statuses: dict[tuple[object, ...], str] = {}
    for row in primary_changes.iter_rows(named=True):
        status = str(row.get("review_status", ""))
        if status == "No differences":
            continue
        if status not in _STATUS_PRECEDENCE or status == _NO_DIFFERENCE:
            raise PpaError(
                f"Executive Summary received unknown performance status: {status!r}",
                None,
            )
        key = tuple(row.get(column) for column in key_columns)
        statuses[key] = status
    return statuses


def _portfolio_statuses(
    unit_statuses: Mapping[tuple[object, ...], str],
) -> tuple[str, ...]:
    """Roll unit statuses to one mutually exclusive worst status per portfolio."""
    by_portfolio: dict[object, list[str]] = {}
    for key, status in unit_statuses.items():
        by_portfolio.setdefault(key[0], []).append(status)
    return tuple(
        max(statuses, key=_STATUS_PRECEDENCE.__getitem__)
        for _, statuses in sorted(by_portfolio.items(), key=lambda item: str(item[0]))
    )


def _performance_row(label: str, statuses: Sequence[str]) -> dict[str, object]:
    """Return one performance quantity row whose buckets foot to total."""
    counts = Counter(statuses)
    row: dict[str, object] = {
        SUMMARY_SECTION: PERFORMANCE_SECTION,
        SUMMARY_LABEL: label,
        TOTAL_QUANTITY: len(statuses),
        NO_PERFORMANCE_DIFFERENCES: counts[_NO_DIFFERENCE],
        FULLY_EXPLAINED_DIFFERENCES: counts[_FULLY_EXPLAINED],
        PARTLY_EXPLAINED_DIFFERENCES: counts[_PARTLY_EXPLAINED],
        UNEXPLAINED_DIFFERENCES: counts[_UNEXPLAINED],
        SETUP_INCOMPLETE: counts[_SETUP_INCOMPLETE],
        DATA_ISSUE_QUANTITY: None,
    }
    bucket_total = sum(
        counts[status]
        for status in (
            _NO_DIFFERENCE,
            _FULLY_EXPLAINED,
            _PARTLY_EXPLAINED,
            _UNEXPLAINED,
            _SETUP_INCOMPLETE,
        )
    )
    if bucket_total != len(statuses):
        raise PpaError("Executive Summary performance quantities do not reconcile.", None)
    return row


def _data_issue_rows(data_issues: pl.DataFrame) -> list[dict[str, object]]:
    """Return issue-type quantities sorted descending with a stable tie-breaker."""
    counts: Counter[DataIssueType] = Counter()
    for row in data_issues.iter_rows(named=True):
        raw_type = str(row.get(data_issue_checks.ISSUE_TYPE))
        try:
            issue_type = DataIssueType(raw_type)
        except ValueError as error:
            raise PpaError(
                f"Executive Summary received unknown Data Issues issue type: {raw_type!r}",
                None,
            ) from error
        counts[issue_type] += 1
    return [
        {
            SUMMARY_SECTION: DATA_ISSUES_SECTION,
            SUMMARY_LABEL: issue_type.value,
            TOTAL_QUANTITY: None,
            NO_PERFORMANCE_DIFFERENCES: None,
            FULLY_EXPLAINED_DIFFERENCES: None,
            PARTLY_EXPLAINED_DIFFERENCES: None,
            UNEXPLAINED_DIFFERENCES: None,
            SETUP_INCOMPLETE: None,
            DATA_ISSUE_QUANTITY: quantity,
        }
        for issue_type, quantity in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0].value),
        )
    ]


def _unit_key_columns(comparison_level: str) -> tuple[str, ...]:
    """Return primary review-unit key columns for one comparison level."""
    _assert_supported_level(comparison_level)
    columns = [pc_cols.PORTFOLIO_ID]
    if comparison_level == SECURITY_COMPARISON_LEVEL:
        columns.append(pc_cols.SECURITY_ID)
    columns.extend((pc_cols.FROM_DATE, pc_cols.THRU_DATE))
    return tuple(columns)


def _period_label(comparison_level: str) -> str:
    """Return the report-level period quantity label."""
    return (
        "Portfolio Periods"
        if comparison_level == PORTFOLIO_COMPARISON_LEVEL
        else "Security Periods"
    )


def _assert_supported_level(comparison_level: str) -> None:
    """Raise when a comparison level cannot support the summary."""
    if comparison_level not in {
        PORTFOLIO_COMPARISON_LEVEL,
        SECURITY_COMPARISON_LEVEL,
    }:
        raise PpaError(f"Unsupported comparison level: {comparison_level!r}", None)


def _summary_schema() -> dict[str, type[pl.DataType]]:
    """Return the stable canonical Executive Summary schema."""
    return {
        SUMMARY_SECTION: pl.String,
        SUMMARY_LABEL: pl.String,
        TOTAL_QUANTITY: pl.Int64,
        NO_PERFORMANCE_DIFFERENCES: pl.Int64,
        FULLY_EXPLAINED_DIFFERENCES: pl.Int64,
        PARTLY_EXPLAINED_DIFFERENCES: pl.Int64,
        UNEXPLAINED_DIFFERENCES: pl.Int64,
        SETUP_INCOMPLETE: pl.Int64,
        DATA_ISSUE_QUANTITY: pl.Int64,
    }


def _sortable_key(values: tuple[object, ...]) -> tuple[str, ...]:
    """Return a deterministic representation for evaluated source keys."""
    return tuple(str(value) for value in values)
