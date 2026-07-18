"""Tests for the quantitative Audit Executive Summary."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
import unittest

import polars as pl

from ppar.errors import PpaError
from ppar.audit import compare_snapshots
from ppar.audit import executive_summary
from ppar.audit import review_model
from ppar.audit import workbook_tables
from ppar.audit.data_issues import checks as data_issue_checks
from ppar.audit.performance_comparison import findings

_RESTATEMENT_PATH = Path("tests/data/axys/validation/ppar_audit_restatement.yaml")


class TestAuditExecutiveSummary(unittest.TestCase):
    """Executive quantities remain mutually exclusive and deterministic."""

    def test_summary_is_first_and_uses_full_evaluated_scope(self) -> None:
        """The canonical first artifact includes unchanged source review units."""
        sheets = workbook_tables.audit_review_workbook_sheets(
            compare_snapshots(_RESTATEMENT_PATH),
            comparison_path=_RESTATEMENT_PATH,
        )

        self.assertEqual(sheets[0].artifact_name, review_model.EXECUTIVE_SUMMARY_ARTIFACT)
        self.assertEqual(sheets[0].sheet_name, review_model.EXECUTIVE_SUMMARY_SHEET)
        self.assertEqual(
            sheets[0].table.columns,
            list(executive_summary.EXECUTIVE_SUMMARY_COLUMNS),
        )
        performance = sheets[0].table.filter(
            pl.col(executive_summary.SUMMARY_SECTION)
            == executive_summary.PERFORMANCE_SECTION
        )
        period_row = performance.row(1, named=True)
        status_total = sum(
            int(period_row[column])
            for column in (
                executive_summary.NO_PERFORMANCE_DIFFERENCES,
                executive_summary.FULLY_EXPLAINED_DIFFERENCES,
                executive_summary.PARTLY_EXPLAINED_DIFFERENCES,
                executive_summary.UNEXPLAINED_DIFFERENCES,
                executive_summary.SETUP_INCOMPLETE,
            )
        )
        self.assertEqual(period_row[executive_summary.TOTAL_QUANTITY], status_total)
        self.assertGreater(
            period_row[executive_summary.NO_PERFORMANCE_DIFFERENCES],
            0,
        )
        self.assertEqual(
            [sheet.sheet_name for sheet in sheets[1:4]],
            [
                review_model.PERFORMANCE_DIFFERENCES_SHEET,
                review_model.PERFORMANCE_DIFFERENCE_CAUSES_SHEET,
                review_model.DATA_ISSUES_SHEET,
            ],
        )

    def test_performance_buckets_and_portfolio_rollup_are_mutually_exclusive(self) -> None:
        """Worst-status portfolio rollup and period counts both foot to total."""
        evaluated_keys = (
            ("P1", dt.date(2026, 1, 1), dt.date(2026, 1, 31)),
            ("P1", dt.date(2026, 2, 1), dt.date(2026, 2, 28)),
            ("P2", dt.date(2026, 1, 1), dt.date(2026, 1, 31)),
            ("P3", dt.date(2026, 1, 1), dt.date(2026, 1, 31)),
            ("P4", dt.date(2026, 1, 1), dt.date(2026, 1, 31)),
        )
        primary = pl.DataFrame(
            [
                _primary_row("P1", 1, "Fully Explained"),
                _primary_row("P1", 2, "Partly Explained"),
                _primary_row("P2", 1, "Unexplained"),
                _primary_row("P3", 1, "Missing YAML Specifications"),
            ]
        )
        summary = executive_summary.executive_summary_table(
            primary,
            _data_issues_table(),
            context=executive_summary.ExecutiveSummaryContext(
                comparison_level="portfolio",
                evaluated_unit_keys=evaluated_keys,
            ),
        )
        performance = summary.filter(
            pl.col(executive_summary.SUMMARY_SECTION)
            == executive_summary.PERFORMANCE_SECTION
        )

        portfolio = performance.row(0, named=True)
        periods = performance.row(1, named=True)
        self.assertEqual(
            _performance_quantities(portfolio),
            (4, 1, 0, 1, 1, 1),
        )
        self.assertEqual(
            _performance_quantities(periods),
            (5, 1, 1, 1, 1, 1),
        )

    def test_data_issues_are_sorted_by_quantity_then_issue_type(self) -> None:
        """Data Issues show stable issue types in descending quantity order."""
        summary = executive_summary.executive_summary_table(
            pl.DataFrame(),
            _data_issues_table(),
            context=executive_summary.ExecutiveSummaryContext("portfolio"),
        )
        data_rows = summary.filter(
            pl.col(executive_summary.SUMMARY_SECTION)
            == executive_summary.DATA_ISSUES_SECTION
        )
        self.assertEqual(
            data_rows.select(
                executive_summary.SUMMARY_LABEL,
                executive_summary.DATA_ISSUE_QUANTITY,
            ).rows(),
            [
                ("holdings_price_range", 3),
                ("dividend_rate", 2),
                ("pa_sa_rate", 2),
            ],
        )

    def test_unknown_status_and_issue_type_fail_closed(self) -> None:
        """The quantity summary cannot silently discard unknown product values."""
        context = executive_summary.ExecutiveSummaryContext("portfolio")
        with self.assertRaisesRegex(PpaError, "unknown performance status"):
            executive_summary.executive_summary_table(
                pl.DataFrame([_primary_row("P1", 1, "not_registered")]),
                pl.DataFrame(schema={data_issue_checks.ISSUE_TYPE: pl.String}),
                context=context,
            )
        with self.assertRaisesRegex(PpaError, "unknown Data Issues issue type"):
            executive_summary.executive_summary_table(
                pl.DataFrame(),
                pl.DataFrame([{data_issue_checks.ISSUE_TYPE: "not_registered"}]),
                context=context,
            )


def _primary_row(portfolio: str, month: int, status: str) -> dict[str, object]:
    """Return a minimal primary performance row for summary tests."""
    return {
        findings.PORTFOLIO_ID: portfolio,
        findings.FROM_DATE: dt.date(2026, month, 1),
        findings.THRU_DATE: dt.date(2026, month, 28 if month == 2 else 31),
        "review_status": status,
    }


def _data_issues_table() -> pl.DataFrame:
    """Return deterministic issue-type counts with one tie."""
    return pl.DataFrame(
        {
            data_issue_checks.ISSUE_TYPE: [
                "pa_sa_rate",
                "dividend_rate",
                "holdings_price_range",
                "holdings_price_range",
                "pa_sa_rate",
                "dividend_rate",
                "holdings_price_range",
            ]
        }
    )


def _performance_quantities(row: dict[str, object]) -> tuple[object, ...]:
    """Return ordered performance quantities from a canonical row."""
    return tuple(
        row[column]
        for column in (
            executive_summary.TOTAL_QUANTITY,
            executive_summary.NO_PERFORMANCE_DIFFERENCES,
            executive_summary.FULLY_EXPLAINED_DIFFERENCES,
            executive_summary.PARTLY_EXPLAINED_DIFFERENCES,
            executive_summary.UNEXPLAINED_DIFFERENCES,
            executive_summary.SETUP_INCOMPLETE,
        )
    )


if __name__ == "__main__":
    unittest.main()
