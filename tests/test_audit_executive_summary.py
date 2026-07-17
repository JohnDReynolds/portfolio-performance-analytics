"""Tests for the canonical Audit Executive Summary."""

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
from ppar.audit.performance_comparison import explain
from ppar.audit.performance_comparison import findings

_RESTATEMENT_PATH = Path("tests/data/axys/validation/ppar_audit_restatement.yaml")
_BASELINE_PATH = Path("tests/data/axys/validation/ppar_audit.yaml")


class TestAuditExecutiveSummary(unittest.TestCase):
    """The summary remains bounded, reconciled, and fail-closed."""

    def test_summary_is_first_and_uses_configured_context(self) -> None:
        """Both renderers receive the same first canonical review artifact."""
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
        snapshot_row = sheets[0].table.filter(
            pl.col(executive_summary.SUMMARY_ITEM) == "What was compared?"
        ).row(0, named=True)
        self.assertEqual(
            snapshot_row[executive_summary.SUMMARY_RESULT],
            "axys_a to axys_b_restatement",
        )
        self.assertEqual(
            [sheet.sheet_name for sheet in sheets[1:4]],
            [
                review_model.PERFORMANCE_DIFFERENCES_SHEET,
                review_model.PERFORMANCE_DIFFERENCE_CAUSES_SHEET,
                review_model.DATA_ISSUES_SHEET,
            ],
        )
        bottom_line = sheets[0].table.row(0, named=True)
        self.assertEqual(bottom_line[executive_summary.SUMMARY_ITEM], "Bottom line")
        self.assertIn(
            "only partly explained. This requires review",
            bottom_line[executive_summary.SUMMARY_RESULT],
        )
        serialized = sheets[0].table.write_csv()
        self.assertNotIn("market_value_or_holding", serialized)
        self.assertNotIn("portfolio_market_value_continuity", serialized)

    def test_empty_performance_state_is_honest(self) -> None:
        """A placeholder detail row does not become a changed review unit."""
        summary = workbook_tables.audit_review_workbook_sheets(
            compare_snapshots(_BASELINE_PATH),
            comparison_path=_BASELINE_PATH,
        )[0].table

        bottom_line = summary.row(0, named=True)
        changed = summary.filter(
            pl.col(executive_summary.SUMMARY_ITEM) == "What changed?"
        ).row(0, named=True)
        self.assertIn(
            "No reported performance changes were found",
            bottom_line[executive_summary.SUMMARY_RESULT],
        )
        self.assertEqual(
            changed[executive_summary.SUMMARY_RESULT],
            "0 changed portfolio periods",
        )

    def test_priority_first_view_is_capped_at_ten(self) -> None:
        """The fixed display limit does not depend on YAML or input size."""
        primary = pl.DataFrame(
            [
                {
                    findings.PORTFOLIO_ID: f"PORT_{index:02d}",
                    findings.FROM_DATE: dt.date(2026, 1, 1),
                    findings.THRU_DATE: dt.date(2026, 1, 31),
                    "performance_change": index / 1000,
                    "estimated_cause_total": 0.0,
                    "unexplained_change": index / 1000,
                    "review_status": "Unexplained",
                    "review_key": f"KEY_{index:02d}",
                }
                for index in range(12)
            ]
        )
        summary = executive_summary.executive_summary_table(
            primary,
            pl.DataFrame(),
            pl.DataFrame(schema={data_issue_checks.ISSUE_TYPE: pl.String}),
            pl.DataFrame(),
            context=executive_summary.ExecutiveSummaryContext(
                comparison_level="portfolio",
                snapshot_a_label="A",
                snapshot_b_label="B",
            ),
        )

        priority = summary.filter(pl.col(executive_summary.REVIEW_KEY) != "")
        self.assertEqual(priority.height, executive_summary.PRIORITY_REVIEW_UNIT_LIMIT)
        self.assertEqual(priority.row(0, named=True)[executive_summary.REVIEW_KEY], "KEY_11")

    def test_unknown_issue_type_and_cause_area_fail_closed(self) -> None:
        """Summary generation cannot silently classify unknown product values."""
        context = executive_summary.ExecutiveSummaryContext("portfolio", "A", "B")
        unknown_issue = pl.DataFrame(
            [{data_issue_checks.ISSUE_TYPE: "not_registered"}]
        )
        with self.assertRaisesRegex(PpaError, "unknown Data Issues issue type"):
            executive_summary.executive_summary_table(
                pl.DataFrame(),
                pl.DataFrame(),
                unknown_issue,
                pl.DataFrame(),
                context=context,
            )

        unknown_cause = pl.DataFrame(
            [
                {
                    findings.PORTFOLIO_ID: "PORT_A",
                    findings.FROM_DATE: dt.date(2026, 1, 1),
                    findings.THRU_DATE: dt.date(2026, 1, 31),
                    explain.ROOT_CAUSE_AREA: "not_registered",
                }
            ]
        )
        with self.assertRaisesRegex(PpaError, "unknown cause area"):
            executive_summary.executive_summary_table(
                pl.DataFrame(),
                unknown_cause,
                pl.DataFrame(schema={data_issue_checks.ISSUE_TYPE: pl.String}),
                pl.DataFrame(),
                context=context,
            )


if __name__ == "__main__":
    unittest.main()
