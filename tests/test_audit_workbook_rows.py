"""Tests for Audit workbook row classification and source identity."""

# Python imports
from pathlib import Path
import unittest
from unittest import mock

# Third-party imports
import polars as pl
from polars.testing import assert_frame_equal

# Project imports
from ppar.audit import workbook_rows
from ppar.audit import workbook_tables
from ppar.audit import compare_snapshots
from ppar.audit.workbook_tables import (
    _workbook_changed_item_row,
    _workbook_raw_audit_trail_table,
    _workbook_with_primary_review_key,
)

_RESTATEMENT_AUDIT_PATH = Path(
    "tests/data/axys/validation/ppar_audit_restatement.yaml"
)


class TestAuditWorkbookRows(unittest.TestCase):
    """Verify row presentation preserves classification and source identity."""

    def test_raw_audit_trail_preserves_non_presentation_source_columns(self) -> None:
        """Bulk source-column projection preserves every raw finding value."""
        findings = compare_snapshots(_RESTATEMENT_AUDIT_PATH)
        keyed_findings = _workbook_with_primary_review_key(
            findings,
            workbook_tables.PORTFOLIO_COMPARISON_LEVEL,
        )
        first_row = keyed_findings.row(0, named=True)
        presentation_columns = set(_workbook_changed_item_row(first_row))
        raw_columns = [
            column
            for column in keyed_findings.columns
            if column not in presentation_columns
        ]

        raw_audit_trail = _workbook_raw_audit_trail_table(findings)
        source_columns = ["review_key", *raw_columns]
        expected = keyed_findings.select(source_columns).sort(source_columns)
        actual = raw_audit_trail.select(source_columns).sort(source_columns)

        assert_frame_equal(actual, expected)

    def test_changed_item_row_classifies_finding_once(self) -> None:
        """Changed-item construction reuses one presentation classification."""
        findings = compare_snapshots(_RESTATEMENT_AUDIT_PATH)
        row = findings.row(0, named=True)
        classify = workbook_rows.workbook_row_kind

        with mock.patch(
            "ppar.audit.workbook_rows.workbook_row_kind",
            wraps=classify,
        ) as classify_spy:
            changed_item = _workbook_changed_item_row(row)

        self.assertEqual(classify_spy.call_count, 1)
        self.assertEqual(changed_item["portfolio_id"], row["portfolio_id"])

    def test_raw_audit_trail_reuses_wording_without_losing_identity(self) -> None:
        """Repeated presentation inputs retain row-specific audit identities."""
        source = compare_snapshots(_RESTATEMENT_AUDIT_PATH).head(1)
        duplicate = source.with_columns(
            pl.lit("SECOND_PORTFOLIO").alias("portfolio_id"),
            pl.lit("source:second-portfolio").alias("source_record_locator"),
        )
        findings = pl.concat([source, duplicate])

        with mock.patch.object(
            workbook_tables,
            "_workbook_changed_item_row",
            wraps=_workbook_changed_item_row,
        ) as changed_item_spy:
            raw_audit_trail = _workbook_raw_audit_trail_table(findings)

        self.assertEqual(changed_item_spy.call_count, 1)
        self.assertEqual(
            set(raw_audit_trail["portfolio_id"].to_list()),
            {source["portfolio_id"][0], "SECOND_PORTFOLIO"},
        )
        self.assertIn(
            "source:second-portfolio",
            raw_audit_trail["source_record_locator"].to_list(),
        )
        self.assertEqual(raw_audit_trail["review_key"].n_unique(), 2)


if __name__ == "__main__":
    unittest.main()
