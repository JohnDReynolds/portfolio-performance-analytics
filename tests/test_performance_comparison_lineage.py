"""Tests for Phase 4 lineage and fail-closed policy coverage."""

from __future__ import annotations

# Python imports
import datetime as dt
from pathlib import Path
import unittest

# Third-party imports
import polars as pl

# Project imports
from ppar.errors import PpaError
from ppar.performance_comparison import conservation
from ppar.performance_comparison import field_roles
from ppar.performance_comparison import findings as pc_findings
from ppar.performance_comparison import lineage
from ppar.performance_comparison import schema as pc_cols
from ppar.performance_comparison.runner import (
    compare_snapshots,
    validate_yaml_setup_complete,
)
from ppar.performance_comparison.workbook_tables import (
    _workbook_underlying_causes_table,
)

_COMPARISON_PATH = Path(
    "tests/data/axys/validation/ppar_performance_comparison_restatement.yaml"
)


class TestPerformanceComparisonLineage(unittest.TestCase):
    """Verify Phase 4 fails closed at lineage and policy boundaries."""

    def test_source_locator_uses_logical_keys_not_changed_values(self) -> None:
        """The same logical record keeps its locator when values change."""
        row_a = {
            pc_cols.PORTFOLIO_ID: "PORT_A",
            pc_cols.SECURITY_ID: "AAPL",
            pc_cols.HOLDING_DATE: dt.date(2026, 1, 31),
            pc_cols.MARKET_VALUE: 100.0,
        }
        row_b = {**row_a, pc_cols.MARKET_VALUE: 125.0}
        key_columns = (
            pc_cols.PORTFOLIO_ID,
            pc_cols.SECURITY_ID,
            pc_cols.HOLDING_DATE,
        )

        locator_a = lineage.source_record_locator(
            pc_cols.HOLDINGS,
            "holdings.csv",
            row_a,
            key_columns,
        )
        locator_b = lineage.source_record_locator(
            pc_cols.HOLDINGS,
            "holdings.csv",
            row_b,
            key_columns,
        )

        self.assertEqual(locator_a, locator_b)

    def test_generated_findings_and_causes_have_bidirectional_lineage(self) -> None:
        """Every generated finding and source-backed cause is traceable."""
        findings = compare_snapshots(_COMPARISON_PATH)
        causes = _workbook_underlying_causes_table(findings)

        lineage.assert_finding_source_lineage(findings)
        lineage.assert_bidirectional_report_lineage(findings, causes)
        source_causes = causes.filter(
            pl.col(lineage.SOURCE_LINEAGE_TYPE) == lineage.SOURCE_FINDING_LINEAGE
        )
        self.assertGreater(source_causes.height, 0)
        self.assertEqual(
            source_causes[lineage.SOURCE_FINDING_FINGERPRINTS].null_count(),
            0,
        )

    def test_untraceable_cause_stops_processing(self) -> None:
        """A report cause cannot point to an invented source record."""
        findings = compare_snapshots(_COMPARISON_PATH)
        causes = _workbook_underlying_causes_table(findings)
        source_cause = causes.filter(
            pl.col(lineage.SOURCE_LINEAGE_TYPE) == lineage.SOURCE_FINDING_LINEAGE
        ).head(1)
        stripped = source_cause.drop(
            lineage.SOURCE_LINEAGE_TYPE,
            lineage.SOURCE_FINDING_FINGERPRINTS,
        ).with_columns(
            pl.lit("source:holdings:invented").alias(
                pc_findings.SOURCE_RECORD_LOCATOR
            )
        )

        with self.assertRaisesRegex(PpaError, "SN-05 bidirectional-lineage"):
            lineage.cause_lineage_table(stripped, findings)

    def test_source_cause_links_every_duplicate_finding_fingerprint(self) -> None:
        """Repeated findings remain visible as sorted unique lineage identities."""
        locator = "source:holdings:duplicate"
        findings = pl.DataFrame(
            {
                pc_findings.FINDING_CODE: ["PC-HOLD-MV", "PC-HOLD-MV"],
                pc_findings.DATASET: [pc_cols.HOLDINGS, pc_cols.HOLDINGS],
                pc_findings.SOURCE_RECORD_LOCATOR: [locator, locator],
            }
        )
        audit_trail = conservation.finding_audit_trail(findings)
        causes = pl.DataFrame(
            {
                pc_findings.FINDING_CODE: ["PC-HOLD-MV"],
                pc_findings.DATASET: [pc_cols.HOLDINGS],
                pc_findings.SOURCE_COLUMN: [pc_cols.MARKET_VALUE],
                pc_findings.SOURCE_RECORD_LOCATOR: [locator],
            }
        )

        result = lineage.cause_lineage_table(
            causes,
            findings,
            finding_audit_trail=audit_trail,
        )

        expected = "|".join(
            sorted(audit_trail[conservation.FINDING_FINGERPRINT].to_list())
        )
        self.assertEqual(result[lineage.SOURCE_FINDING_FINGERPRINTS][0], expected)

    def test_persisted_cross_artifact_lineage_rejects_unknown_fingerprint(self) -> None:
        """Bundle lineage cannot reference a fingerprint absent from findings."""
        findings = compare_snapshots(_COMPARISON_PATH)
        finding_audit = conservation.finding_audit_trail(findings)
        causes = _workbook_underlying_causes_table(findings)
        source_causes = causes.filter(
            pl.col(lineage.SOURCE_LINEAGE_TYPE) == lineage.SOURCE_FINDING_LINEAGE
        ).with_columns(
            pl.lit("invented-fingerprint").alias(
                lineage.SOURCE_FINDING_FINGERPRINTS
            )
        )

        issues = lineage.persisted_cross_artifact_lineage_issues(
            finding_audit,
            source_causes,
        )

        self.assertIn(
            "cause lineage references a fingerprint outside findings.csv",
            issues,
        )

    def test_comparison_surface_rejects_unclassified_field(self) -> None:
        """A newly compared field must receive an accounting role first."""
        with self.assertRaisesRegex(PpaError, "SN-12 fail-closed policy"):
            field_roles.assert_comparison_fields_classified(
                {pc_cols.HOLDINGS: ("new_performance_value",)}
            )

    def test_suppression_cannot_hide_unclassified_field(self) -> None:
        """YAML suppression cannot substitute for an accounting-role decision."""
        findings = pl.DataFrame(
            {
                pc_findings.DATASET: [pc_cols.HOLDINGS],
                pc_findings.SOURCE_COLUMN: ["new_performance_value"],
                pc_findings.SUPPRESSED: [True],
            }
        )

        with self.assertRaisesRegex(PpaError, "SN-12 fail-closed policy"):
            validate_yaml_setup_complete(findings)

    def test_policy_requirement_is_derived_from_field_role(self) -> None:
        """Input roles require policy while reported and context roles do not."""
        self.assertTrue(
            field_roles.requires_explicit_impact_policy(
                pc_cols.HOLDINGS,
                pc_cols.BASE_MARKET_VALUE,
            )
        )
        self.assertTrue(
            field_roles.requires_explicit_impact_policy(
                pc_cols.FX_RATES,
                pc_cols.FX_RATE,
            )
        )
        self.assertFalse(
            field_roles.requires_explicit_impact_policy(
                pc_cols.PORTFOLIO_PERFORMANCE,
                pc_cols.INCOME,
            )
        )
        self.assertFalse(
            field_roles.requires_explicit_impact_policy(
                pc_cols.HOLDINGS,
                pc_cols.COST,
            )
        )


if __name__ == "__main__":
    unittest.main()
