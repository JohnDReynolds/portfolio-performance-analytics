"""Tests for performance comparison explanation helpers."""

# Python imports
from datetime import date
from pathlib import Path
import unittest

# Third-party imports
import polars as pl

# Project imports
from ppar.performance_comparison import (
    Finding,
    compare_snapshots,
    findings_to_polars,
    portfolio_period_cause_summary,
    portfolio_period_contribution_candidates,
    portfolio_period_evidence_breakdown,
    portfolio_period_flow_cross_check_reconciliation,
    portfolio_period_impact_coverage_summary,
    portfolio_period_summary,
    portfolio_period_transaction_cross_checks,
    rank_portfolio_period_evidence,
    security_period_evidence_breakdown,
    security_period_summary,
    transaction_activity_summary,
    transaction_matching_diagnostics,
)
from ppar.performance_comparison import explain as pc_explain
from ppar.performance_comparison.explain import (
    CASH_FINDING_COUNT,
    AMOUNT_DELTA,
    CHANGED_FIELDS,
    CONTEXT_FINDING_COUNT,
    CROSS_CHECK_ABSOLUTE_ESTIMATE_TOTAL,
    CROSS_CHECK_COUNT,
    CROSS_CHECK_ESTIMATE_TOTAL,
    CROSS_CHECK_ONLY,
    CROSS_CHECK_TREATMENT,
    CROSS_CHECK_MINUS_FLOW_IMPACT,
    DIRECT_INPUT_FINDING_COUNT,
    EVIDENCE_GROUP,
    ESTIMATED_CAUSE_AREA_COUNT,
    ESTIMATED_RETURN_IMPACT,
    ESTIMATED_RETURN_IMPACT_TOTAL,
    EVIDENCE_ONLY_AREAS,
    EVIDENCE_ONLY_CAUSE_AREA_COUNT,
    FINDING_COUNT,
    FX_RATE_FINDING_COUNT,
    HAS_SUPPRESSED_FINDINGS,
    IMPACT_POLICY,
    IMPACT_BASIS,
    IMPACT_BASIS_NO_ESTIMATE,
    IMPACT_BASIS_PORTFOLIO_SOURCE_FIELD,
    IMPACT_BASIS_SECURITY_CONTRIBUTION,
    IMPACT_BASIS_SECURITY_RETURN_WEIGHTED,
    IMPACT_BASIS_TRANSACTION_PERFORMANCE_AMOUNT,
    IMPACT_COVERAGE_REVIEW_NOTE,
    IMPACT_COVERAGE_STATUS,
    IMPACT_COVERAGE_STATUS_MISSING_INPUTS,
    IMPACT_CONFIDENCE,
    IMPACT_CONFIDENCE_LOW,
    IMPACT_CONFIDENCE_MEDIUM,
    IMPACT_MESSAGE,
    IMPACT_METHOD,
    IMPACT_METHOD_TRANSACTION_AMOUNT_DELTA_OVER_DENOMINATOR,
    IMPACT_METHOD_VENDOR_CONTRIBUTION_DELTA,
    LOW_CONFIDENCE_ESTIMATE_COUNT,
    MEDIUM_CONFIDENCE_ESTIMATE_COUNT,
    MISSING_IMPACT_INPUTS,
    PORTFOLIO_PERIOD_CAUSE_SUMMARY_COLUMNS,
    PORTFOLIO_FINDING_COUNT,
    PORTFOLIO_PERIOD_CONTRIBUTION_CANDIDATE_COLUMNS,
    PORTFOLIO_PERIOD_EVIDENCE_BREAKDOWN_COLUMNS,
    PORTFOLIO_PERIOD_FLOW_CROSS_CHECK_RECONCILIATION_COLUMNS,
    PORTFOLIO_PERIOD_EVIDENCE_RANKING_COLUMNS,
    PORTFOLIO_PERIOD_IMPACT_COVERAGE_COLUMNS,
    PORTFOLIO_PERIOD_SUMMARY_COLUMNS,
    PORTFOLIO_PERIOD_TRANSACTION_CROSS_CHECK_COLUMNS,
    PORTFOLIO_RETURN_DELTA,
    PORTFOLIO_FLOW_DELTA,
    PORTFOLIO_FLOW_IMPACT_ESTIMATE,
    POSITION_FINDING_COUNT,
    PRICE_DELTA,
    PRICE_FINDING_COUNT,
    PRIORITY_SCORE,
    QUANTITY_DELTA,
    REFERENCE_FINDING_COUNT,
    RECONCILIATION_STATUS,
    RECONCILIATION_STATUS_ALIGNED,
    RECONCILIATION_STATUS_MISSING_PORTFOLIO_FLOW_DELTA,
    RECONCILIATION_STATUS_MISSING_TRANSACTION_CROSS_CHECK,
    RELATED_OUTPUT_FINDING_COUNT,
    REVIEW_RANK,
    ROOT_CAUSE_AREA,
    ROOT_CAUSE_CASH,
    ROOT_CAUSE_MARKET_VALUE_OR_POSITION,
    ROOT_CAUSE_PORTFOLIO_PERFORMANCE_INPUT,
    ROOT_CAUSE_PRICE,
    ROOT_CAUSE_AREA_COUNT,
    ROOT_CAUSE_SECURITY_RETURN_OR_CONTRIBUTION,
    ROOT_CAUSE_TRANSACTION_ACTIVITY,
    SECURITY_FINDING_COUNT,
    SECURITY_PERIOD_EVIDENCE_BREAKDOWN_COLUMNS,
    SECURITY_PERIOD_SUMMARY_COLUMNS,
    SECURITY_RETURN_DELTA,
    TRANSACTION_FINDING_COUNT,
    TRANSACTION_ACTIVITY_SUMMARY_COLUMNS,
    TRANSACTION_IMPACT_DIAGNOSTICS,
    TRANSACTION_IMPACT_POLICIES,
    TRANSACTION_MATCHING_DIAGNOSTIC_COLUMNS,
    TRANSACTION_MATCH_REVIEW_NOTE,
    TRANSACTION_SEMANTICS_SOURCES,
    TOP_CODES,
)
from ppar.performance_comparison import columns as pc_cols
from ppar.performance_comparison.findings import (
    CONTEXT,
    DATASET,
    DIRECT_INPUT,
    EVIDENCE_ROLE,
    FINDING_CODE,
    FROM_DATE,
    CONFIDENCE_HIGH,
    PC_PORT_RET,
    PC_PORT_FLOW,
    PC_SEC_CONTR,
    PC_SEC_RET,
    PC_TXN_AMT,
    PORTFOLIO_ID,
    RELATED_OUTPUT,
    RETURN_DENOMINATOR,
    SECURITY_ID,
    SEVERITY_MATERIAL,
    SOURCE_COLUMN,
    TARGET_OUTPUT,
    THRU_DATE,
    TRANSACTION_IMPACT_DIAGNOSTIC,
    TRANSACTION_IMPACT_DIAGNOSTIC_ESTIMATE,
    TRANSACTION_IMPACT_POLICY,
    TRANSACTION_IMPACT_POLICY_EXTERNAL_FLOW_EVIDENCE_ONLY,
    TRANSACTION_IMPACT_POLICY_PERFORMANCE_AMOUNT_DELTA,
    TRANSACTION_CATEGORY,
    TRANSACTION_MATCH_STATUS,
    TRANSACTION_MATCH_STATUS_ID_MATCH,
    TRANSACTION_MATCH_STATUS_STRICT_FALLBACK_UNMATCHED,
    TRANSACTION_SEMANTICS_SOURCE,
)

_BASELINE_COMPARISON_PATH = Path("tests/data/axys/ppar_performance_comparison.yaml")
_RESTATEMENT_COMPARISON_PATH = Path(
    "tests/data/axys/ppar_performance_comparison_restatement.yaml"
)
_RESTATEMENT_TRANSACTION_RULES_PATH = Path(
    "tests/data/axys/ppar_performance_comparison_restatement_transaction_rules.yaml"
)
_SUPPRESSED_COMPARISON_PATH = Path(
    "tests/data/axys/ppar_performance_comparison_suppressed.yaml"
)


class TestPerformanceComparisonExplain(unittest.TestCase):
    """Verify explanation-oriented performance comparison tables."""

    _baseline_findings: pl.DataFrame
    _restatement_findings: pl.DataFrame
    _suppressed_findings: pl.DataFrame

    @classmethod
    def setUpClass(cls) -> None:
        """Cache expensive snapshot comparisons for explanation tests."""
        cls._baseline_findings = compare_snapshots(_BASELINE_COMPARISON_PATH)
        cls._restatement_findings = compare_snapshots(_RESTATEMENT_COMPARISON_PATH)
        cls._suppressed_findings = compare_snapshots(_SUPPRESSED_COMPARISON_PATH)

    def _baseline(self) -> pl.DataFrame:
        """Return baseline findings for one test."""
        return self._baseline_findings.clone()

    def _restatement(self) -> pl.DataFrame:
        """Return restatement findings for one test."""
        return self._restatement_findings.clone()

    def _suppressed(self) -> pl.DataFrame:
        """Return suppressed fixture findings for one test."""
        return self._suppressed_findings.clone()

    def _transaction_estimate_findings(
        self,
        *,
        cash_flow_sign: str | None = "negative",
        performance_flow_sign: str | None = "performance",
        transaction_semantics_source: str | None = "source",
        transaction_impact_policy: str | None = (
            TRANSACTION_IMPACT_POLICY_PERFORMANCE_AMOUNT_DELTA
        ),
        transaction_impact_diagnostic: str | None = None,
        transaction_impact_diagnostic_estimate: float | None = None,
        return_denominator: float | None = 1000.0,
        from_date: date | None = date(2025, 5, 30),
        thru_date: date | None = date(2025, 5, 30),
    ) -> pl.DataFrame:
        """Return restatement findings with transaction impact inputs overridden."""
        return self._restatement().with_columns(
            pl.when(pl.col(DATASET) == pc_cols.TRANSACTIONS)
            .then(pl.lit(cash_flow_sign))
            .otherwise(pl.col(pc_cols.CASH_FLOW_SIGN))
            .alias(pc_cols.CASH_FLOW_SIGN),
            pl.when(pl.col(DATASET) == pc_cols.TRANSACTIONS)
            .then(pl.lit(performance_flow_sign))
            .otherwise(pl.col(pc_cols.PERFORMANCE_FLOW_SIGN))
            .alias(pc_cols.PERFORMANCE_FLOW_SIGN),
            pl.when(pl.col(DATASET) == pc_cols.TRANSACTIONS)
            .then(pl.lit(transaction_semantics_source))
            .otherwise(pl.col(TRANSACTION_SEMANTICS_SOURCE))
            .alias(TRANSACTION_SEMANTICS_SOURCE),
            pl.when(pl.col(DATASET) == pc_cols.TRANSACTIONS)
            .then(pl.lit(transaction_impact_policy))
            .otherwise(pl.col(TRANSACTION_IMPACT_POLICY))
            .alias(TRANSACTION_IMPACT_POLICY),
            pl.when(pl.col(DATASET) == pc_cols.TRANSACTIONS)
            .then(pl.lit(transaction_impact_diagnostic))
            .otherwise(pl.col(TRANSACTION_IMPACT_DIAGNOSTIC))
            .alias(TRANSACTION_IMPACT_DIAGNOSTIC),
            pl.when(pl.col(DATASET) == pc_cols.TRANSACTIONS)
            .then(pl.lit(transaction_impact_diagnostic_estimate).cast(pl.Float64))
            .otherwise(pl.col(TRANSACTION_IMPACT_DIAGNOSTIC_ESTIMATE))
            .alias(TRANSACTION_IMPACT_DIAGNOSTIC_ESTIMATE),
            pl.when(pl.col(DATASET) == pc_cols.TRANSACTIONS)
            .then(pl.lit(return_denominator).cast(pl.Float64))
            .otherwise(pl.col(RETURN_DENOMINATOR))
            .alias(RETURN_DENOMINATOR),
            pl.when(pl.col(DATASET) == pc_cols.TRANSACTIONS)
            .then(pl.lit(from_date).cast(pl.Date))
            .otherwise(pl.col(FROM_DATE))
            .alias(FROM_DATE),
            pl.when(pl.col(DATASET) == pc_cols.TRANSACTIONS)
            .then(pl.lit(thru_date).cast(pl.Date))
            .otherwise(pl.col(THRU_DATE))
            .alias(THRU_DATE),
        )

    def test_explain_module_exports_public_explanation_helpers(self) -> None:
        """Explanation helpers are available directly from the explain module."""
        self.assertIs(pc_explain.portfolio_period_cause_summary, portfolio_period_cause_summary)
        self.assertIs(pc_explain.portfolio_period_summary, portfolio_period_summary)
        self.assertIs(
            pc_explain.portfolio_period_contribution_candidates,
            portfolio_period_contribution_candidates,
        )
        self.assertIs(
            pc_explain.portfolio_period_evidence_breakdown,
            portfolio_period_evidence_breakdown,
        )
        self.assertIs(
            pc_explain.portfolio_period_flow_cross_check_reconciliation,
            portfolio_period_flow_cross_check_reconciliation,
        )
        self.assertIs(
            pc_explain.portfolio_period_impact_coverage_summary,
            portfolio_period_impact_coverage_summary,
        )
        self.assertIs(
            pc_explain.portfolio_period_transaction_cross_checks,
            portfolio_period_transaction_cross_checks,
        )
        self.assertIs(
            pc_explain.rank_portfolio_period_evidence,
            rank_portfolio_period_evidence,
        )
        self.assertIs(pc_explain.security_period_summary, security_period_summary)
        self.assertIs(
            pc_explain.security_period_evidence_breakdown,
            security_period_evidence_breakdown,
        )
        self.assertIs(pc_explain.transaction_activity_summary, transaction_activity_summary)
        self.assertIs(
            pc_explain.transaction_matching_diagnostics,
            transaction_matching_diagnostics,
        )

    def test_portfolio_period_summary_groups_related_evidence(self) -> None:
        """Portfolio-period summary groups findings around return deltas."""
        findings = self._restatement()

        summary = portfolio_period_summary(findings)
        row = summary.row(0, named=True)

        self.assertEqual(summary.columns, list(PORTFOLIO_PERIOD_SUMMARY_COLUMNS))
        self.assertEqual(summary.height, 1)
        self.assertEqual(row[PORTFOLIO_ID], "PORT_A")
        self.assertAlmostEqual(row[PORTFOLIO_RETURN_DELTA], 0.0005)
        self.assertEqual(row[PORTFOLIO_FINDING_COUNT], 3)
        self.assertEqual(row[DIRECT_INPUT_FINDING_COUNT], 11)
        self.assertEqual(row[RELATED_OUTPUT_FINDING_COUNT], 5)
        self.assertEqual(row[CONTEXT_FINDING_COUNT], 1)
        self.assertEqual(row[POSITION_FINDING_COUNT], 4)
        self.assertEqual(row[CASH_FINDING_COUNT], 2)
        self.assertEqual(row[TRANSACTION_FINDING_COUNT], 3)
        self.assertEqual(row[PRICE_FINDING_COUNT], 1)
        self.assertEqual(row[FX_RATE_FINDING_COUNT], 0)
        self.assertEqual(row[REFERENCE_FINDING_COUNT], 0)
        self.assertEqual(row[FINDING_COUNT], 18)
        self.assertFalse(row[HAS_SUPPRESSED_FINDINGS])

    def test_portfolio_period_summary_tracks_suppressed_related_evidence(self) -> None:
        """Portfolio-period summary flags suppressed related findings."""
        findings = self._suppressed()

        active_summary = portfolio_period_summary(findings)
        audit_summary = portfolio_period_summary(findings, include_suppressed=True)

        self.assertTrue(active_summary.row(0, named=True)[HAS_SUPPRESSED_FINDINGS])
        self.assertEqual(active_summary.row(0, named=True)[RELATED_OUTPUT_FINDING_COUNT], 4)
        self.assertEqual(audit_summary.row(0, named=True)[RELATED_OUTPUT_FINDING_COUNT], 5)

    def test_portfolio_period_evidence_breakdown_returns_role_and_dataset_counts(
        self,
    ) -> None:
        """Evidence breakdown returns readable role and dataset count rows."""
        findings = self._restatement()

        breakdown = portfolio_period_evidence_breakdown(findings)

        self.assertEqual(
            breakdown.columns,
            list(PORTFOLIO_PERIOD_EVIDENCE_BREAKDOWN_COLUMNS),
        )
        self.assertEqual(breakdown.height, 11)
        self.assertEqual(_breakdown_count(breakdown, TARGET_OUTPUT, None), 1)
        self.assertEqual(_breakdown_count(breakdown, DIRECT_INPUT, None), 11)
        self.assertEqual(_breakdown_count(breakdown, RELATED_OUTPUT, None), 5)
        self.assertEqual(_breakdown_count(breakdown, CONTEXT, None), 1)
        self.assertEqual(
            _breakdown_count(breakdown, TARGET_OUTPUT, "portfolio_performance"),
            1,
        )
        self.assertEqual(
            _breakdown_count(breakdown, DIRECT_INPUT, "portfolio_performance"),
            2,
        )
        self.assertEqual(_breakdown_count(breakdown, DIRECT_INPUT, "prices"), 1)
        self.assertEqual(_breakdown_count(breakdown, DIRECT_INPUT, "transactions"), 3)
        self.assertEqual(_breakdown_count(breakdown, DIRECT_INPUT, "positions"), 3)
        self.assertEqual(_breakdown_count(breakdown, DIRECT_INPUT, "cash"), 2)
        self.assertEqual(
            _breakdown_count(breakdown, RELATED_OUTPUT, "security_performance"),
            5,
        )

    def test_portfolio_period_evidence_breakdown_tracks_suppressed_counts(self) -> None:
        """Evidence breakdown counts suppressed rows only when requested."""
        findings = self._suppressed()

        active_breakdown = portfolio_period_evidence_breakdown(findings)
        audit_breakdown = portfolio_period_evidence_breakdown(
            findings,
            include_suppressed=True,
        )

        self.assertEqual(_breakdown_count(active_breakdown, RELATED_OUTPUT, None), 4)
        self.assertEqual(_breakdown_count(audit_breakdown, RELATED_OUTPUT, None), 5)

    def test_portfolio_period_evidence_breakdown_returns_stable_empty_table(self) -> None:
        """No portfolio return deltas produce an empty breakdown table."""
        findings = self._baseline()

        breakdown = portfolio_period_evidence_breakdown(findings)

        self.assertTrue(breakdown.is_empty())
        self.assertEqual(
            breakdown.columns,
            list(PORTFOLIO_PERIOD_EVIDENCE_BREAKDOWN_COLUMNS),
        )

    def test_rank_portfolio_period_evidence_prioritizes_direct_inputs(self) -> None:
        """Evidence ranking sorts direct inputs ahead of related outputs."""
        findings = self._restatement()

        ranking = rank_portfolio_period_evidence(findings)
        first_row = ranking.row(0, named=True)
        direct_input_scores = ranking.filter(
            pl.col(EVIDENCE_ROLE) == DIRECT_INPUT
        ).get_column(PRIORITY_SCORE)
        related_output_scores = ranking.filter(
            pl.col(EVIDENCE_ROLE) == RELATED_OUTPUT
        ).get_column(PRIORITY_SCORE)

        self.assertEqual(
            ranking.columns,
            list(PORTFOLIO_PERIOD_EVIDENCE_RANKING_COLUMNS),
        )
        self.assertEqual(ranking.height, 17)
        self.assertEqual(first_row[REVIEW_RANK], 1)
        self.assertEqual(first_row[EVIDENCE_ROLE], DIRECT_INPUT)
        direct_input_score_values = [int(value) for value in direct_input_scores.to_list()]
        related_output_score_values = [
            int(value) for value in related_output_scores.to_list()
        ]
        self.assertGreater(
            min(direct_input_score_values),
            max(related_output_score_values),
        )
        self.assertNotIn(TARGET_OUTPUT, ranking.get_column(EVIDENCE_ROLE).to_list())

    def test_portfolio_period_contribution_candidates_estimates_contribution(
        self,
    ) -> None:
        """Contribution candidates estimate vendor contribution deltas."""
        findings = self._restatement()

        candidates = portfolio_period_contribution_candidates(findings)
        contribution = candidates.filter(pl.col(FINDING_CODE) == PC_SEC_CONTR).row(
            0,
            named=True,
        )

        self.assertEqual(
            candidates.columns,
            list(PORTFOLIO_PERIOD_CONTRIBUTION_CANDIDATE_COLUMNS),
        )
        self.assertEqual(candidates.height, 17)
        self.assertAlmostEqual(contribution[ESTIMATED_RETURN_IMPACT], 0.00058425)
        self.assertEqual(
            contribution[IMPACT_POLICY],
            "security_contribution:vendor_contribution_delta",
        )
        self.assertEqual(contribution[IMPACT_BASIS], IMPACT_BASIS_SECURITY_CONTRIBUTION)
        self.assertEqual(contribution[IMPACT_CONFIDENCE], IMPACT_CONFIDENCE_MEDIUM)
        self.assertEqual(
            contribution[IMPACT_METHOD],
            IMPACT_METHOD_VENDOR_CONTRIBUTION_DELTA,
        )

    def test_portfolio_period_contribution_candidates_estimates_source_field(
        self,
    ) -> None:
        """Contribution candidates estimate supported portfolio source fields."""
        findings = self._restatement()

        candidates = portfolio_period_contribution_candidates(findings)
        source_field = candidates.filter(
            (pl.col(SOURCE_COLUMN) == pc_cols.GAIN_LOSS)
            & (pl.col(IMPACT_BASIS) == IMPACT_BASIS_PORTFOLIO_SOURCE_FIELD)
        ).row(0, named=True)

        self.assertAlmostEqual(
            source_field[ESTIMATED_RETURN_IMPACT],
            500.0 / 999915.0,
        )
        self.assertEqual(source_field[IMPACT_CONFIDENCE], IMPACT_CONFIDENCE_LOW)
        self.assertEqual(
            source_field[IMPACT_POLICY],
            "portfolio_source_field:source_field_delta_over_begin_market_value",
        )
        self.assertIn("beginning market value", source_field[IMPACT_MESSAGE])

    def test_portfolio_period_contribution_candidates_estimates_security_return(
        self,
    ) -> None:
        """Contribution candidates estimate weighted security return deltas."""
        findings = self._restatement()

        candidates = portfolio_period_contribution_candidates(findings)
        security_return = candidates.filter(
            (pl.col(FINDING_CODE) == PC_SEC_RET)
            & (pl.col(IMPACT_BASIS) == IMPACT_BASIS_SECURITY_RETURN_WEIGHTED)
        ).row(0, named=True)

        self.assertAlmostEqual(
            security_return[ESTIMATED_RETURN_IMPACT],
            0.01 * 0.05319463,
        )
        self.assertEqual(
            security_return[IMPACT_POLICY],
            "security_return:security_return_delta_times_weight",
        )
        self.assertEqual(security_return[IMPACT_CONFIDENCE], IMPACT_CONFIDENCE_LOW)
        self.assertIn("portfolio weight", security_return[IMPACT_MESSAGE])

    def test_contribution_candidates_require_explicit_yaml_policy(self) -> None:
        """Portfolio and security contribution estimates require YAML policy."""
        findings = self._restatement().with_columns(
            pl.when(pl.col(DATASET) != pc_cols.TRANSACTIONS)
            .then(pl.lit(None))
            .otherwise(pl.col(IMPACT_POLICY))
            .alias(IMPACT_POLICY)
        )

        candidates = portfolio_period_contribution_candidates(findings)
        gated_rows = candidates.filter(
            pl.col(FINDING_CODE).is_in([PC_SEC_CONTR, PC_SEC_RET])
            | (
                (pl.col(DATASET) == pc_cols.PORTFOLIO_PERFORMANCE)
                & pl.col(SOURCE_COLUMN).is_in([pc_cols.INCOME, pc_cols.GAIN_LOSS])
            )
        )

        self.assertGreater(gated_rows.height, 0)
        self.assertTrue(
            gated_rows.get_column(ESTIMATED_RETURN_IMPACT).null_count()
            == gated_rows.height
        )
        self.assertEqual(
            set(gated_rows.get_column(IMPACT_BASIS).to_list()),
            {IMPACT_BASIS_NO_ESTIMATE},
        )

    def test_portfolio_period_contribution_candidates_keeps_no_estimate_rows(
        self,
    ) -> None:
        """Contribution candidates keep rows without defensible impact estimates."""
        findings = self._restatement()

        candidates = portfolio_period_contribution_candidates(findings)
        no_estimate = candidates.filter(
            pl.col(IMPACT_BASIS) == IMPACT_BASIS_NO_ESTIMATE
        )
        row = no_estimate.row(0, named=True)

        self.assertEqual(no_estimate.height, 14)
        self.assertIsNone(row[ESTIMATED_RETURN_IMPACT])
        self.assertEqual(row[IMPACT_CONFIDENCE], IMPACT_CONFIDENCE_LOW)
        self.assertIsNone(row[IMPACT_METHOD])

    def test_portfolio_period_contribution_candidates_returns_stable_empty_table(
        self,
    ) -> None:
        """No portfolio return deltas produce an empty contribution table."""
        findings = self._baseline()

        candidates = portfolio_period_contribution_candidates(findings)

        self.assertTrue(candidates.is_empty())
        self.assertEqual(
            candidates.columns,
            list(PORTFOLIO_PERIOD_CONTRIBUTION_CANDIDATE_COLUMNS),
        )

    def test_portfolio_period_cause_summary_rolls_up_contribution_estimates(
        self,
    ) -> None:
        """Cause summary rolls up currently defensible contribution estimates."""
        findings = self._restatement()

        summary = portfolio_period_cause_summary(findings)
        contribution_row = summary.filter(
            pl.col(ROOT_CAUSE_AREA) == ROOT_CAUSE_SECURITY_RETURN_OR_CONTRIBUTION
        ).row(0, named=True)

        self.assertEqual(
            summary.columns,
            list(PORTFOLIO_PERIOD_CAUSE_SUMMARY_COLUMNS),
        )
        self.assertEqual(summary.height, 6)
        self.assertEqual(contribution_row[FINDING_COUNT], 5)
        self.assertAlmostEqual(contribution_row[ESTIMATED_RETURN_IMPACT], 0.00058425)
        self.assertEqual(
            contribution_row[IMPACT_BASIS],
            IMPACT_BASIS_SECURITY_CONTRIBUTION,
        )
        self.assertEqual(contribution_row[IMPACT_CONFIDENCE], IMPACT_CONFIDENCE_MEDIUM)
        self.assertIn("PC-SEC-CONTR", contribution_row[TOP_CODES])
        self.assertIn("vendor contribution deltas", contribution_row[IMPACT_MESSAGE])
        self.assertIn("review cross-checks", contribution_row[IMPACT_MESSAGE])

    def test_portfolio_period_cause_summary_prefers_vendor_contribution(
        self,
    ) -> None:
        """Security cause summary avoids double-counting weighted returns."""
        findings = self._restatement()

        summary = portfolio_period_cause_summary(findings)
        contribution_row = summary.filter(
            pl.col(ROOT_CAUSE_AREA) == ROOT_CAUSE_SECURITY_RETURN_OR_CONTRIBUTION
        ).row(0, named=True)

        self.assertAlmostEqual(contribution_row[ESTIMATED_RETURN_IMPACT], 0.00058425)
        self.assertEqual(
            contribution_row[IMPACT_BASIS],
            IMPACT_BASIS_SECURITY_CONTRIBUTION,
        )

    def test_portfolio_period_cause_summary_uses_weighted_return_fallback(
        self,
    ) -> None:
        """Security cause summary uses weighted return when contribution is absent."""
        findings = self._restatement().filter(pl.col(FINDING_CODE) != PC_SEC_CONTR)

        summary = portfolio_period_cause_summary(findings)
        contribution_row = summary.filter(
            pl.col(ROOT_CAUSE_AREA) == ROOT_CAUSE_SECURITY_RETURN_OR_CONTRIBUTION
        ).row(0, named=True)

        self.assertAlmostEqual(
            contribution_row[ESTIMATED_RETURN_IMPACT],
            0.01 * 0.05319463,
        )
        self.assertEqual(
            contribution_row[IMPACT_BASIS],
            IMPACT_BASIS_SECURITY_RETURN_WEIGHTED,
        )
        self.assertEqual(contribution_row[IMPACT_CONFIDENCE], IMPACT_CONFIDENCE_LOW)
        self.assertNotIn("vendor contribution deltas", contribution_row[IMPACT_MESSAGE])

    def test_portfolio_period_cause_summary_keeps_transactions_evidence_only(
        self,
    ) -> None:
        """Transaction activity remains unestimated without sign semantics."""
        findings = self._restatement()

        summary = portfolio_period_cause_summary(findings)
        transaction_row = summary.filter(
            pl.col(ROOT_CAUSE_AREA) == ROOT_CAUSE_TRANSACTION_ACTIVITY
        ).row(0, named=True)

        self.assertEqual(transaction_row[FINDING_COUNT], 3)
        self.assertIsNone(transaction_row[ESTIMATED_RETURN_IMPACT])
        self.assertEqual(transaction_row[IMPACT_BASIS], IMPACT_BASIS_NO_ESTIMATE)
        self.assertEqual(transaction_row[IMPACT_CONFIDENCE], IMPACT_CONFIDENCE_LOW)
        self.assertIn("Missing impact inputs", transaction_row[IMPACT_MESSAGE])
        self.assertNotIn("return denominator", transaction_row[IMPACT_MESSAGE])
        self.assertIn(
            "transaction sign and flow semantics",
            transaction_row[IMPACT_MESSAGE],
        )
        self.assertNotIn("normalized transaction category", transaction_row[IMPACT_MESSAGE])

    def test_portfolio_period_cause_summary_includes_direct_input_buckets(
        self,
    ) -> None:
        """Cause summary keeps direct input buckets visible for review."""
        findings = self._restatement()

        summary = portfolio_period_cause_summary(findings)
        cause_areas = set(summary.get_column(ROOT_CAUSE_AREA).to_list())

        self.assertEqual(
            cause_areas,
            {
                ROOT_CAUSE_CASH,
                ROOT_CAUSE_MARKET_VALUE_OR_POSITION,
                ROOT_CAUSE_PORTFOLIO_PERFORMANCE_INPUT,
                ROOT_CAUSE_PRICE,
                ROOT_CAUSE_SECURITY_RETURN_OR_CONTRIBUTION,
                ROOT_CAUSE_TRANSACTION_ACTIVITY,
            },
        )

    def test_portfolio_period_cause_summary_groups_by_portfolio_period(
        self,
    ) -> None:
        """Cause summary rows remain tied to one portfolio-period target delta."""
        findings = self._restatement()

        summary = portfolio_period_cause_summary(findings)
        key_rows = summary.select([PORTFOLIO_ID, "from_date", "thru_date"]).unique()

        self.assertEqual(key_rows.height, 1)
        key_row = key_rows.row(0, named=True)
        self.assertEqual(key_row[PORTFOLIO_ID], "PORT_A")
        self.assertEqual(str(key_row["from_date"]), "2025-05-30")
        self.assertEqual(str(key_row["thru_date"]), "2025-05-30")

    def test_portfolio_period_cause_summary_keeps_buckets_separate(
        self,
    ) -> None:
        """Cause summary separates direct inputs from related-output buckets."""
        findings = self._restatement()

        summary = portfolio_period_cause_summary(findings)
        counts_by_cause = {
            row[ROOT_CAUSE_AREA]: row[FINDING_COUNT]
            for row in summary.iter_rows(named=True)
        }

        self.assertEqual(
            counts_by_cause,
            {
                ROOT_CAUSE_CASH: 2,
                ROOT_CAUSE_MARKET_VALUE_OR_POSITION: 4,
                ROOT_CAUSE_PORTFOLIO_PERFORMANCE_INPUT: 2,
                ROOT_CAUSE_PRICE: 1,
                ROOT_CAUSE_SECURITY_RETURN_OR_CONTRIBUTION: 5,
                ROOT_CAUSE_TRANSACTION_ACTIVITY: 3,
            },
        )

    def test_portfolio_period_impact_coverage_summary_counts_estimate_coverage(
        self,
    ) -> None:
        """Impact coverage summarizes estimated and evidence-only cause areas."""
        findings = self._restatement()

        coverage = portfolio_period_impact_coverage_summary(findings)
        row = coverage.row(0, named=True)

        self.assertEqual(
            coverage.columns,
            list(PORTFOLIO_PERIOD_IMPACT_COVERAGE_COLUMNS),
        )
        self.assertEqual(coverage.height, 1)
        self.assertEqual(row[PORTFOLIO_ID], "PORT_A")
        self.assertAlmostEqual(row[PORTFOLIO_RETURN_DELTA], 0.0005)
        self.assertEqual(row[ROOT_CAUSE_AREA_COUNT], 6)
        self.assertEqual(row[ESTIMATED_CAUSE_AREA_COUNT], 2)
        self.assertEqual(row[EVIDENCE_ONLY_CAUSE_AREA_COUNT], 4)
        self.assertEqual(row[LOW_CONFIDENCE_ESTIMATE_COUNT], 1)
        self.assertEqual(row[MEDIUM_CONFIDENCE_ESTIMATE_COUNT], 1)
        self.assertAlmostEqual(
            row[ESTIMATED_RETURN_IMPACT_TOTAL],
            0.0010842925036128068,
        )
        self.assertIn(ROOT_CAUSE_TRANSACTION_ACTIVITY, row[EVIDENCE_ONLY_AREAS])
        self.assertIn(ROOT_CAUSE_PRICE, row[EVIDENCE_ONLY_AREAS])
        self.assertEqual(row[TRANSACTION_SEMANTICS_SOURCES], "unknown: 3")
        self.assertNotIn("return denominator", row[MISSING_IMPACT_INPUTS])
        self.assertIn(
            "transaction sign and flow semantics",
            row[MISSING_IMPACT_INPUTS],
        )
        self.assertIn("return-impact method", row[MISSING_IMPACT_INPUTS])
        self.assertEqual(row[IMPACT_COVERAGE_STATUS], IMPACT_COVERAGE_STATUS_MISSING_INPUTS)
        self.assertEqual(
            row[IMPACT_COVERAGE_REVIEW_NOTE],
            "Resolve missing inputs before relying on impact totals.",
        )
        self.assertIn("2 cause area(s) have estimates", row[IMPACT_MESSAGE])

    def test_portfolio_period_impact_coverage_summary_returns_stable_empty_table(
        self,
    ) -> None:
        """No portfolio return deltas produce an empty coverage summary."""
        findings = self._baseline()

        coverage = portfolio_period_impact_coverage_summary(findings)

        self.assertTrue(coverage.is_empty())
        self.assertEqual(
            coverage.columns,
            list(PORTFOLIO_PERIOD_IMPACT_COVERAGE_COLUMNS),
        )

    def test_portfolio_period_cause_summary_works_without_security_performance(
        self,
    ) -> None:
        """Portfolio cause summary does not require security performance evidence."""
        findings = self._restatement()
        portfolio_only_findings = findings.filter(
            pl.col(DATASET) != "security_performance"
        )

        summary = portfolio_period_cause_summary(portfolio_only_findings)
        cause_areas = set(summary.get_column(ROOT_CAUSE_AREA).to_list())

        self.assertEqual(summary.height, 5)
        self.assertEqual(
            cause_areas,
            {
                ROOT_CAUSE_CASH,
                ROOT_CAUSE_MARKET_VALUE_OR_POSITION,
                ROOT_CAUSE_PORTFOLIO_PERFORMANCE_INPUT,
                ROOT_CAUSE_PRICE,
                ROOT_CAUSE_TRANSACTION_ACTIVITY,
            },
        )
        self.assertNotIn(ROOT_CAUSE_SECURITY_RETURN_OR_CONTRIBUTION, cause_areas)

    def test_portfolio_period_cause_summary_returns_stable_empty_table(
        self,
    ) -> None:
        """No portfolio return deltas produce an empty cause summary table."""
        findings = self._baseline()

        summary = portfolio_period_cause_summary(findings)

        self.assertTrue(summary.is_empty())
        self.assertEqual(
            summary.columns,
            list(PORTFOLIO_PERIOD_CAUSE_SUMMARY_COLUMNS),
        )

    def test_transaction_activity_summary_groups_changed_fields_by_category(
        self,
    ) -> None:
        """Transaction activity summary groups changed fields by category."""
        findings = self._restatement()

        summary = transaction_activity_summary(findings)
        row = summary.row(0, named=True)

        self.assertEqual(
            summary.columns,
            list(TRANSACTION_ACTIVITY_SUMMARY_COLUMNS),
        )
        self.assertEqual(summary.height, 1)
        self.assertEqual(row[PORTFOLIO_ID], "PORT_A")
        self.assertEqual(row[SECURITY_ID], "AAPL")
        self.assertEqual(str(row["from_date"]), "2025-05-30")
        self.assertEqual(str(row["thru_date"]), "2025-05-30")
        self.assertEqual(row[TRANSACTION_CATEGORY], "buy")
        self.assertEqual(row[FINDING_COUNT], 3)
        self.assertEqual(row[CHANGED_FIELDS], "amount, quantity, price")
        self.assertAlmostEqual(row[AMOUNT_DELTA], -100.0)
        self.assertAlmostEqual(row[QUANTITY_DELTA], 1.0)
        self.assertAlmostEqual(row[PRICE_DELTA], 0.5)
        self.assertEqual(row[TRANSACTION_SEMANTICS_SOURCES], "unknown: 3")
        self.assertEqual(
            row[MISSING_IMPACT_INPUTS],
            "transaction sign and flow semantics",
        )

    def test_transaction_matching_diagnostics_explain_id_and_fallback_counts(
        self,
    ) -> None:
        """Transaction matching diagnostics expose conservative match status counts."""
        findings = self._restatement()

        diagnostics = transaction_matching_diagnostics(findings)
        row = diagnostics.row(0, named=True)

        self.assertEqual(
            diagnostics.columns,
            list(TRANSACTION_MATCHING_DIAGNOSTIC_COLUMNS),
        )
        self.assertEqual(diagnostics.height, 1)
        self.assertEqual(row[TRANSACTION_MATCH_STATUS], TRANSACTION_MATCH_STATUS_ID_MATCH)
        self.assertEqual(row[FINDING_COUNT], 3)
        self.assertIn("transaction_id", row[TRANSACTION_MATCH_REVIEW_NOTE])

        fallback_findings = findings.with_columns(
            pl.when(pl.col(DATASET) == pc_cols.TRANSACTIONS)
            .then(pl.lit(TRANSACTION_MATCH_STATUS_STRICT_FALLBACK_UNMATCHED))
            .otherwise(pl.col(TRANSACTION_MATCH_STATUS))
            .alias(TRANSACTION_MATCH_STATUS)
        )
        fallback_row = transaction_matching_diagnostics(fallback_findings).row(
            0,
            named=True,
        )

        self.assertEqual(
            fallback_row[TRANSACTION_MATCH_STATUS],
            TRANSACTION_MATCH_STATUS_STRICT_FALLBACK_UNMATCHED,
        )
        self.assertIn("strict fallback keys", fallback_row[TRANSACTION_MATCH_REVIEW_NOTE])
        self.assertIn("rather than inferring an edit", fallback_row[TRANSACTION_MATCH_REVIEW_NOTE])

    def test_transaction_rules_fixture_summarizes_yaml_semantics(self) -> None:
        """Axys YAML transaction rules flow into transaction summaries."""
        findings = compare_snapshots(_RESTATEMENT_TRANSACTION_RULES_PATH)

        activity = transaction_activity_summary(findings)
        activity_row = activity.row(0, named=True)
        coverage = portfolio_period_impact_coverage_summary(findings)
        coverage_row = coverage.row(0, named=True)

        self.assertEqual(activity_row[TRANSACTION_SEMANTICS_SOURCES], "mixed: 3")
        self.assertEqual(activity_row[MISSING_IMPACT_INPUTS], "")
        self.assertNotIn(
            "transaction sign and flow semantics",
            activity_row[IMPACT_MESSAGE],
        )
        self.assertEqual(coverage_row[TRANSACTION_SEMANTICS_SOURCES], "mixed: 3")
        self.assertNotIn("return denominator", coverage_row[MISSING_IMPACT_INPUTS])
        self.assertNotIn(
            "transaction sign and flow semantics",
            coverage_row[MISSING_IMPACT_INPUTS],
        )

    def test_transaction_activity_summary_uses_available_sign_semantics(self) -> None:
        """Recognized sign semantics remove only the sign missing-input flag."""
        findings = self._restatement().with_columns(
            pl.when(pl.col(DATASET) == pc_cols.TRANSACTIONS)
            .then(pl.lit("positive"))
            .otherwise(pl.col(pc_cols.CASH_FLOW_SIGN))
            .alias(pc_cols.CASH_FLOW_SIGN),
            pl.when(pl.col(DATASET) == pc_cols.TRANSACTIONS)
            .then(pl.lit("external"))
            .otherwise(pl.col(pc_cols.PERFORMANCE_FLOW_SIGN))
            .alias(pc_cols.PERFORMANCE_FLOW_SIGN),
            pl.when(pl.col(DATASET) == pc_cols.TRANSACTIONS)
            .then(pl.lit(None).cast(pl.Float64))
            .otherwise(pl.col(RETURN_DENOMINATOR))
            .alias(RETURN_DENOMINATOR),
            pl.when(pl.col(DATASET) == pc_cols.TRANSACTIONS)
            .then(pl.lit("source"))
            .otherwise(pl.col(TRANSACTION_SEMANTICS_SOURCE))
            .alias(TRANSACTION_SEMANTICS_SOURCE),
            pl.when(pl.col(DATASET) == pc_cols.TRANSACTIONS)
            .then(pl.lit(TRANSACTION_IMPACT_POLICY_PERFORMANCE_AMOUNT_DELTA))
            .otherwise(pl.col(TRANSACTION_IMPACT_POLICY))
            .alias(TRANSACTION_IMPACT_POLICY),
        )

        summary = transaction_activity_summary(findings)
        row = summary.row(0, named=True)

        self.assertEqual(row[TRANSACTION_SEMANTICS_SOURCES], "source: 3")
        self.assertEqual(row[MISSING_IMPACT_INPUTS], "return denominator")
        self.assertEqual(row[IMPACT_BASIS], IMPACT_BASIS_NO_ESTIMATE)
        self.assertIn("evidence-only", row[IMPACT_MESSAGE])
        self.assertNotIn("transaction sign and flow semantics", row[IMPACT_MESSAGE])

        cause_summary = portfolio_period_cause_summary(findings)
        transaction_row = cause_summary.filter(
            pl.col(ROOT_CAUSE_AREA) == ROOT_CAUSE_TRANSACTION_ACTIVITY
        ).row(0, named=True)
        self.assertIn("return denominator", transaction_row[IMPACT_MESSAGE])
        self.assertNotIn(
            "transaction sign and flow semantics",
            transaction_row[IMPACT_MESSAGE],
        )

    def test_transaction_amount_candidate_estimates_performance_flow(self) -> None:
        """Performance-treated transaction amount deltas get a low-confidence estimate."""
        findings = self._restatement().with_columns(
            pl.when(pl.col(DATASET) == pc_cols.TRANSACTIONS)
            .then(pl.lit("negative"))
            .otherwise(pl.col(pc_cols.CASH_FLOW_SIGN))
            .alias(pc_cols.CASH_FLOW_SIGN),
            pl.when(pl.col(DATASET) == pc_cols.TRANSACTIONS)
            .then(pl.lit("performance"))
            .otherwise(pl.col(pc_cols.PERFORMANCE_FLOW_SIGN))
            .alias(pc_cols.PERFORMANCE_FLOW_SIGN),
            pl.when(pl.col(DATASET) == pc_cols.TRANSACTIONS)
            .then(pl.lit(1000.0))
            .otherwise(pl.col(RETURN_DENOMINATOR))
            .alias(RETURN_DENOMINATOR),
            pl.when(pl.col(DATASET) == pc_cols.TRANSACTIONS)
            .then(pl.lit("source"))
            .otherwise(pl.col(TRANSACTION_SEMANTICS_SOURCE))
            .alias(TRANSACTION_SEMANTICS_SOURCE),
            pl.when(pl.col(DATASET) == pc_cols.TRANSACTIONS)
            .then(pl.lit(TRANSACTION_IMPACT_POLICY_PERFORMANCE_AMOUNT_DELTA))
            .otherwise(pl.col(TRANSACTION_IMPACT_POLICY))
            .alias(TRANSACTION_IMPACT_POLICY),
        )

        candidates = portfolio_period_contribution_candidates(findings)
        transaction_amount = candidates.filter(
            (pl.col(DATASET) == pc_cols.TRANSACTIONS)
            & (pl.col(SOURCE_COLUMN) == pc_cols.AMOUNT)
        ).row(0, named=True)

        self.assertAlmostEqual(transaction_amount[ESTIMATED_RETURN_IMPACT], -0.1)
        self.assertEqual(
            transaction_amount[IMPACT_BASIS],
            IMPACT_BASIS_TRANSACTION_PERFORMANCE_AMOUNT,
        )
        self.assertEqual(transaction_amount[IMPACT_CONFIDENCE], IMPACT_CONFIDENCE_LOW)
        self.assertEqual(
            transaction_amount[IMPACT_METHOD],
            IMPACT_METHOD_TRANSACTION_AMOUNT_DELTA_OVER_DENOMINATOR,
        )
        self.assertIn("source-signed transaction amount", transaction_amount[IMPACT_MESSAGE])
        self.assertIn("Transaction semantics source: source", transaction_amount[IMPACT_MESSAGE])

        summary = portfolio_period_cause_summary(findings)
        transaction_row = summary.filter(
            pl.col(ROOT_CAUSE_AREA) == ROOT_CAUSE_TRANSACTION_ACTIVITY
        ).row(0, named=True)
        self.assertAlmostEqual(transaction_row[ESTIMATED_RETURN_IMPACT], -0.1)
        self.assertEqual(
            transaction_row[IMPACT_BASIS],
            IMPACT_BASIS_TRANSACTION_PERFORMANCE_AMOUNT,
        )

        activity = transaction_activity_summary(findings)
        activity_row = activity.row(0, named=True)
        self.assertEqual(activity_row[TRANSACTION_SEMANTICS_SOURCES], "source: 3")
        self.assertEqual(activity_row[MISSING_IMPACT_INPUTS], "")
        self.assertIn("modeled impact inputs", activity_row[IMPACT_MESSAGE])

    def test_transaction_amount_requires_explicit_performance_policy(self) -> None:
        """Performance transaction amounts need a YAML-selected impact method."""
        findings = self._transaction_estimate_findings(transaction_impact_policy=None)

        candidates = portfolio_period_contribution_candidates(findings)
        transaction_amount = candidates.filter(
            (pl.col(DATASET) == pc_cols.TRANSACTIONS)
            & (pl.col(SOURCE_COLUMN) == pc_cols.AMOUNT)
        ).row(0, named=True)
        activity = transaction_activity_summary(findings)
        activity_row = activity.row(0, named=True)

        self.assertIsNone(transaction_amount[ESTIMATED_RETURN_IMPACT])
        self.assertEqual(transaction_amount[IMPACT_BASIS], IMPACT_BASIS_NO_ESTIMATE)
        self.assertEqual(activity_row[MISSING_IMPACT_INPUTS], "transaction impact method")

    def test_transaction_amount_candidate_reports_mixed_semantics_source(self) -> None:
        """Transaction estimates disclose when YAML rules helped supply semantics."""
        findings = self._transaction_estimate_findings(
            transaction_semantics_source="mixed"
        )

        candidates = portfolio_period_contribution_candidates(findings)
        transaction_amount = candidates.filter(
            (pl.col(DATASET) == pc_cols.TRANSACTIONS)
            & (pl.col(SOURCE_COLUMN) == pc_cols.AMOUNT)
        ).row(0, named=True)

        self.assertIn(
            "mixed source and YAML transaction_rules",
            transaction_amount[IMPACT_MESSAGE],
        )

    def test_external_transaction_amount_stays_unestimated(self) -> None:
        """External-flow transaction amounts need a separate impact method."""
        findings = self._transaction_estimate_findings(performance_flow_sign="external")

        candidates = portfolio_period_contribution_candidates(findings)
        transaction_amount = candidates.filter(
            (pl.col(DATASET) == pc_cols.TRANSACTIONS)
            & (pl.col(SOURCE_COLUMN) == pc_cols.AMOUNT)
        ).row(0, named=True)

        self.assertIsNone(transaction_amount[ESTIMATED_RETURN_IMPACT])
        self.assertEqual(transaction_amount[IMPACT_BASIS], IMPACT_BASIS_NO_ESTIMATE)
        self.assertIsNone(transaction_amount[IMPACT_METHOD])

        activity = transaction_activity_summary(findings)
        activity_row = activity.row(0, named=True)
        self.assertEqual(
            activity_row[MISSING_IMPACT_INPUTS],
            "external-flow impact method",
        )
        self.assertIn("external-flow impact method", activity_row[IMPACT_MESSAGE])

    def test_external_transaction_evidence_only_policy_stays_unestimated(self) -> None:
        """Explicit YAML evidence-only policy documents external-flow treatment."""
        findings = self._transaction_estimate_findings(
            performance_flow_sign="external",
            transaction_impact_policy=(
                TRANSACTION_IMPACT_POLICY_EXTERNAL_FLOW_EVIDENCE_ONLY
            ),
            transaction_impact_diagnostic="external-flow evidence-only policy",
        )

        candidates = portfolio_period_contribution_candidates(findings)
        transaction_amount = candidates.filter(
            (pl.col(DATASET) == pc_cols.TRANSACTIONS)
            & (pl.col(SOURCE_COLUMN) == pc_cols.AMOUNT)
        ).row(0, named=True)

        self.assertIsNone(transaction_amount[ESTIMATED_RETURN_IMPACT])
        self.assertEqual(transaction_amount[IMPACT_BASIS], IMPACT_BASIS_NO_ESTIMATE)
        self.assertEqual(
            transaction_amount[TRANSACTION_IMPACT_POLICY],
            TRANSACTION_IMPACT_POLICY_EXTERNAL_FLOW_EVIDENCE_ONLY,
        )
        self.assertEqual(
            transaction_amount[TRANSACTION_IMPACT_DIAGNOSTIC],
            "external-flow evidence-only policy",
        )

        activity = transaction_activity_summary(findings)
        activity_row = activity.row(0, named=True)
        self.assertEqual(
            activity_row[MISSING_IMPACT_INPUTS],
            "external-flow evidence-only policy",
        )
        self.assertIn("evidence-only policy", activity_row[IMPACT_MESSAGE])

    def test_external_transaction_diagnostic_refines_missing_inputs(self) -> None:
        """Transaction diagnostics make external-flow method gaps reviewable."""
        findings = self._transaction_estimate_findings(
            performance_flow_sign="external",
            transaction_impact_diagnostic=(
                "modified_dietz missing inputs: flow date, in-period flow date"
            ),
        )

        activity = transaction_activity_summary(findings)
        activity_row = activity.row(0, named=True)

        self.assertEqual(
            activity_row[MISSING_IMPACT_INPUTS],
            "modified_dietz flow date, modified_dietz in-period flow date",
        )

    def test_transaction_amount_requires_usable_denominator(self) -> None:
        """Missing or zero denominators keep transaction amount rows unestimated."""
        for denominator in (None, 0.0):
            with self.subTest(denominator=denominator):
                findings = self._transaction_estimate_findings(
                    return_denominator=denominator
                )

                candidates = portfolio_period_contribution_candidates(findings)
                transaction_amount = candidates.filter(
                    (pl.col(DATASET) == pc_cols.TRANSACTIONS)
                    & (pl.col(SOURCE_COLUMN) == pc_cols.AMOUNT)
                ).row(0, named=True)
                activity = transaction_activity_summary(findings)
                activity_row = activity.row(0, named=True)

                self.assertIsNone(transaction_amount[ESTIMATED_RETURN_IMPACT])
                self.assertEqual(
                    transaction_amount[IMPACT_BASIS],
                    IMPACT_BASIS_NO_ESTIMATE,
                )
                self.assertEqual(
                    activity_row[MISSING_IMPACT_INPUTS],
                    "return denominator",
                )

    def test_transaction_amount_rejects_unmodeled_semantics(self) -> None:
        """Only performance-treated positive/negative cash signs estimate today."""
        scenarios = [
            ("negative", "neutral", "neutral-flow impact method"),
            ("negative", "unknown", "transaction sign and flow semantics"),
            ("none", "performance", "no-cash transaction impact method"),
            ("unknown", "performance", "transaction sign and flow semantics"),
        ]
        for cash_flow_sign, performance_flow_sign, missing_input in scenarios:
            with self.subTest(
                cash_flow_sign=cash_flow_sign,
                performance_flow_sign=performance_flow_sign,
            ):
                findings = self._transaction_estimate_findings(
                    cash_flow_sign=cash_flow_sign,
                    performance_flow_sign=performance_flow_sign,
                )

                candidates = portfolio_period_contribution_candidates(findings)
                transaction_amount = candidates.filter(
                    (pl.col(DATASET) == pc_cols.TRANSACTIONS)
                    & (pl.col(SOURCE_COLUMN) == pc_cols.AMOUNT)
                ).row(0, named=True)
                activity = transaction_activity_summary(findings)
                activity_row = activity.row(0, named=True)

                self.assertIsNone(transaction_amount[ESTIMATED_RETURN_IMPACT])
                self.assertEqual(
                    transaction_amount[IMPACT_BASIS],
                    IMPACT_BASIS_NO_ESTIMATE,
                )
                self.assertIsNone(transaction_amount[IMPACT_METHOD])
                self.assertEqual(activity_row[MISSING_IMPACT_INPUTS], missing_input)

    def test_portfolio_period_transaction_cross_checks_summarizes_diagnostics(
        self,
    ) -> None:
        """Cross-check summaries roll up diagnostic estimates by portfolio period."""
        findings = self._transaction_estimate_findings(
            performance_flow_sign="external",
            transaction_impact_policy="external_flow:modified_dietz",
            transaction_impact_diagnostic="modified_dietz cross-check estimate",
            transaction_impact_diagnostic_estimate=0.005,
        )

        cross_checks = portfolio_period_transaction_cross_checks(findings)
        row = cross_checks.row(0, named=True)

        self.assertEqual(
            cross_checks.columns,
            list(PORTFOLIO_PERIOD_TRANSACTION_CROSS_CHECK_COLUMNS),
        )
        self.assertEqual(row[PORTFOLIO_ID], "PORT_A")
        self.assertEqual(
            row[TRANSACTION_IMPACT_POLICIES],
            "external_flow:modified_dietz",
        )
        self.assertEqual(row[CROSS_CHECK_TREATMENT], CROSS_CHECK_ONLY)
        self.assertEqual(row[CROSS_CHECK_COUNT], 3)
        self.assertAlmostEqual(row[CROSS_CHECK_ESTIMATE_TOTAL], 0.015)
        self.assertAlmostEqual(row[CROSS_CHECK_ABSOLUTE_ESTIMATE_TOTAL], 0.015)
        self.assertEqual(row[CHANGED_FIELDS], "amount, quantity, price")
        self.assertEqual(
            row[TRANSACTION_IMPACT_DIAGNOSTICS],
            "modified_dietz cross-check estimate",
        )
        self.assertIn("review-only", row[IMPACT_MESSAGE])

    def test_portfolio_period_transaction_cross_checks_returns_stable_empty_table(
        self,
    ) -> None:
        """Empty cross-check summaries preserve stable columns."""
        findings = self._restatement()

        cross_checks = portfolio_period_transaction_cross_checks(findings)

        self.assertEqual(
            cross_checks.columns,
            list(PORTFOLIO_PERIOD_TRANSACTION_CROSS_CHECK_COLUMNS),
        )
        self.assertTrue(cross_checks.is_empty())

    def test_portfolio_period_flow_cross_check_reconciliation_aligns_estimates(
        self,
    ) -> None:
        """Flow/cross-check reconciliation compares review-only estimates."""
        findings = findings_to_polars(
            [
                _portfolio_flow_finding(delta=10.0, denominator=1000.0),
                _transaction_cross_check_finding(estimate=0.01),
            ]
        )

        reconciliation = portfolio_period_flow_cross_check_reconciliation(findings)
        row = reconciliation.row(0, named=True)

        self.assertEqual(
            reconciliation.columns,
            list(PORTFOLIO_PERIOD_FLOW_CROSS_CHECK_RECONCILIATION_COLUMNS),
        )
        self.assertEqual(row[PORTFOLIO_ID], "PORT_A")
        self.assertEqual(row[PORTFOLIO_FLOW_DELTA], 10.0)
        self.assertEqual(row[PORTFOLIO_FLOW_IMPACT_ESTIMATE], 0.01)
        self.assertEqual(row[CROSS_CHECK_ESTIMATE_TOTAL], 0.01)
        self.assertEqual(row[CROSS_CHECK_MINUS_FLOW_IMPACT], 0.0)
        self.assertEqual(row[RECONCILIATION_STATUS], RECONCILIATION_STATUS_ALIGNED)

    def test_portfolio_period_flow_cross_check_reconciliation_names_missing_sides(
        self,
    ) -> None:
        """Reconciliation rows stay explicit when one side is unavailable."""
        flow_only = portfolio_period_flow_cross_check_reconciliation(
            findings_to_polars([_portfolio_flow_finding(delta=10.0, denominator=1000.0)])
        )
        cross_check_only = portfolio_period_flow_cross_check_reconciliation(
            findings_to_polars([_transaction_cross_check_finding(estimate=0.01)])
        )

        self.assertEqual(
            flow_only.row(0, named=True)[RECONCILIATION_STATUS],
            RECONCILIATION_STATUS_MISSING_TRANSACTION_CROSS_CHECK,
        )
        self.assertEqual(
            cross_check_only.row(0, named=True)[RECONCILIATION_STATUS],
            RECONCILIATION_STATUS_MISSING_PORTFOLIO_FLOW_DELTA,
        )

    def test_portfolio_period_flow_cross_check_reconciliation_empty_table(
        self,
    ) -> None:
        """Empty reconciliation output preserves stable columns."""
        reconciliation = portfolio_period_flow_cross_check_reconciliation(
            findings_to_polars([])
        )

        self.assertEqual(
            reconciliation.columns,
            list(PORTFOLIO_PERIOD_FLOW_CROSS_CHECK_RECONCILIATION_COLUMNS),
        )
        self.assertTrue(reconciliation.is_empty())

    def test_transaction_activity_summary_is_evidence_only(self) -> None:
        """Transaction activity summary does not estimate return impact."""
        findings = self._restatement()

        summary = transaction_activity_summary(findings)
        row = summary.row(0, named=True)

        self.assertEqual(row[IMPACT_BASIS], IMPACT_BASIS_NO_ESTIMATE)
        self.assertEqual(row[IMPACT_CONFIDENCE], IMPACT_CONFIDENCE_LOW)
        self.assertIn("evidence-only", row[IMPACT_MESSAGE])
        self.assertIn("Missing impact inputs", row[IMPACT_MESSAGE])

    def test_transaction_activity_summary_does_not_infer_ambiguous_period(
        self,
    ) -> None:
        """Transaction activity summary only borrows unambiguous target periods."""
        findings = self._restatement().with_columns(
            pl.when(pl.col(DATASET) == pc_cols.TRANSACTIONS)
            .then(pl.lit(None).cast(pl.Date))
            .otherwise(pl.col(FROM_DATE))
            .alias(FROM_DATE),
            pl.when(pl.col(DATASET) == pc_cols.TRANSACTIONS)
            .then(pl.lit(None).cast(pl.Date))
            .otherwise(pl.col(THRU_DATE))
            .alias(THRU_DATE),
        )
        second_target_period = (
            findings.filter(pl.col(FINDING_CODE) == PC_PORT_RET)
            .head(1)
            .with_columns(
                pl.lit(date(2025, 5, 31)).alias(FROM_DATE),
                pl.lit(date(2025, 5, 31)).alias(THRU_DATE),
            )
        )

        summary = transaction_activity_summary(pl.concat([findings, second_target_period]))
        row = summary.row(0, named=True)

        self.assertIsNone(row[FROM_DATE])
        self.assertIsNone(row[THRU_DATE])
        self.assertIn("portfolio period", row[MISSING_IMPACT_INPUTS])

    def test_transaction_activity_summary_returns_stable_empty_table(self) -> None:
        """No transaction findings produce an empty transaction summary."""
        findings = self._baseline()

        summary = transaction_activity_summary(findings)

        self.assertTrue(summary.is_empty())
        self.assertEqual(
            summary.columns,
            list(TRANSACTION_ACTIVITY_SUMMARY_COLUMNS),
        )

    def test_rank_portfolio_period_evidence_tracks_suppressed_findings(self) -> None:
        """Evidence ranking includes suppressed evidence only when requested."""
        findings = self._suppressed()

        active_ranking = rank_portfolio_period_evidence(findings)
        audit_ranking = rank_portfolio_period_evidence(
            findings,
            include_suppressed=True,
        )

        self.assertEqual(active_ranking.height, 16)
        self.assertEqual(audit_ranking.height, 17)
        self.assertEqual(
            active_ranking.filter(pl.col(FINDING_CODE) == PC_SEC_RET).height,
            0,
        )
        self.assertEqual(
            audit_ranking.filter(pl.col(FINDING_CODE) == PC_SEC_RET).height,
            1,
        )

    def test_rank_portfolio_period_evidence_works_without_security_performance(
        self,
    ) -> None:
        """Portfolio evidence ranking does not require security performance."""
        findings = self._restatement()
        portfolio_only_findings = findings.filter(
            pl.col(DATASET) != "security_performance"
        )

        ranking = rank_portfolio_period_evidence(portfolio_only_findings)

        self.assertEqual(ranking.height, 12)
        self.assertEqual(
            set(ranking.get_column(EVIDENCE_ROLE).to_list()),
            {DIRECT_INPUT, CONTEXT},
        )
        self.assertEqual(
            ranking.filter(pl.col(SOURCE_COLUMN) == pc_cols.COST).row(0, named=True)[
                EVIDENCE_ROLE
            ],
            CONTEXT,
        )
        self.assertNotIn("security_performance", ranking.get_column(DATASET).to_list())

    def test_rank_portfolio_period_evidence_returns_stable_empty_table(self) -> None:
        """No portfolio return deltas produce an empty ranking table."""
        findings = self._baseline()

        ranking = rank_portfolio_period_evidence(findings)

        self.assertTrue(ranking.is_empty())
        self.assertEqual(
            ranking.columns,
            list(PORTFOLIO_PERIOD_EVIDENCE_RANKING_COLUMNS),
        )

    def test_portfolio_period_summary_returns_stable_empty_table(self) -> None:
        """No portfolio return deltas produce an empty stable summary."""
        findings = self._baseline()

        summary = portfolio_period_summary(findings)

        self.assertTrue(summary.is_empty())
        self.assertEqual(summary.columns, list(PORTFOLIO_PERIOD_SUMMARY_COLUMNS))

    def test_security_period_summary_groups_related_evidence(self) -> None:
        """Security-period summary groups findings around security returns."""
        findings = self._restatement()

        summary = security_period_summary(findings)
        row = summary.row(0, named=True)

        self.assertEqual(summary.columns, list(SECURITY_PERIOD_SUMMARY_COLUMNS))
        self.assertEqual(summary.height, 1)
        self.assertEqual(row[PORTFOLIO_ID], "PORT_A")
        self.assertEqual(row[SECURITY_ID], "AAPL")
        self.assertAlmostEqual(row[SECURITY_RETURN_DELTA], 0.01)
        self.assertEqual(row[SECURITY_FINDING_COUNT], 3)
        self.assertEqual(row[DIRECT_INPUT_FINDING_COUNT], 7)
        self.assertEqual(row[RELATED_OUTPUT_FINDING_COUNT], 3)
        self.assertEqual(row[CONTEXT_FINDING_COUNT], 3)
        self.assertEqual(row[POSITION_FINDING_COUNT], 4)
        self.assertEqual(row[TRANSACTION_FINDING_COUNT], 3)
        self.assertEqual(row[PRICE_FINDING_COUNT], 1)
        self.assertEqual(row[REFERENCE_FINDING_COUNT], 2)
        self.assertEqual(row[FINDING_COUNT], 13)
        self.assertFalse(row[HAS_SUPPRESSED_FINDINGS])

    def test_security_period_summary_tracks_suppressed_related_evidence(self) -> None:
        """Security-period summary flags suppressed related findings."""
        findings = self._suppressed()

        active_summary = security_period_summary(findings)
        audit_summary = security_period_summary(findings, include_suppressed=True)

        self.assertTrue(active_summary.row(0, named=True)[HAS_SUPPRESSED_FINDINGS])
        self.assertEqual(active_summary.row(0, named=True)[RELATED_OUTPUT_FINDING_COUNT], 2)
        self.assertEqual(audit_summary.row(0, named=True)[RELATED_OUTPUT_FINDING_COUNT], 3)
        self.assertEqual(active_summary.row(0, named=True)[FINDING_COUNT], 12)
        self.assertEqual(audit_summary.row(0, named=True)[FINDING_COUNT], 13)

    def test_security_period_summary_returns_stable_empty_table(self) -> None:
        """No security return deltas produce an empty stable summary."""
        findings = self._baseline()

        summary = security_period_summary(findings)

        self.assertTrue(summary.is_empty())
        self.assertEqual(summary.columns, list(SECURITY_PERIOD_SUMMARY_COLUMNS))

    def test_security_period_evidence_breakdown_returns_role_and_dataset_counts(
        self,
    ) -> None:
        """Security evidence breakdown returns role and dataset count rows."""
        findings = self._restatement()

        breakdown = security_period_evidence_breakdown(findings)

        self.assertEqual(
            breakdown.columns,
            list(SECURITY_PERIOD_EVIDENCE_BREAKDOWN_COLUMNS),
        )
        self.assertEqual(breakdown.height, 10)
        self.assertEqual(_breakdown_count(breakdown, TARGET_OUTPUT, None), 1)
        self.assertEqual(_breakdown_count(breakdown, DIRECT_INPUT, None), 7)
        self.assertEqual(_breakdown_count(breakdown, RELATED_OUTPUT, None), 2)
        self.assertEqual(_breakdown_count(breakdown, CONTEXT, None), 3)
        self.assertEqual(
            _breakdown_count(breakdown, TARGET_OUTPUT, "security_performance"),
            1,
        )
        self.assertEqual(
            _breakdown_count(breakdown, RELATED_OUTPUT, "security_performance"),
            2,
        )
        self.assertEqual(_breakdown_count(breakdown, DIRECT_INPUT, "prices"), 1)
        self.assertEqual(_breakdown_count(breakdown, DIRECT_INPUT, "transactions"), 3)
        self.assertEqual(_breakdown_count(breakdown, DIRECT_INPUT, "positions"), 3)
        self.assertEqual(_breakdown_count(breakdown, CONTEXT, "security_master"), 2)

    def test_security_period_evidence_breakdown_tracks_suppressed_counts(self) -> None:
        """Security evidence breakdown counts suppressed rows only when requested."""
        findings = self._suppressed()

        active_breakdown = security_period_evidence_breakdown(findings)
        audit_breakdown = security_period_evidence_breakdown(
            findings,
            include_suppressed=True,
        )

        self.assertEqual(_breakdown_count(active_breakdown, TARGET_OUTPUT, None), 0)
        self.assertEqual(_breakdown_count(audit_breakdown, TARGET_OUTPUT, None), 1)
        self.assertEqual(_breakdown_count(active_breakdown, RELATED_OUTPUT, None), 2)
        self.assertEqual(_breakdown_count(audit_breakdown, RELATED_OUTPUT, None), 2)

    def test_security_period_evidence_breakdown_returns_stable_empty_table(self) -> None:
        """No security return deltas produce an empty security breakdown table."""
        findings = self._baseline()

        breakdown = security_period_evidence_breakdown(findings)

        self.assertTrue(breakdown.is_empty())
        self.assertEqual(
            breakdown.columns,
            list(SECURITY_PERIOD_EVIDENCE_BREAKDOWN_COLUMNS),
        )


def _breakdown_count(
    breakdown: pl.DataFrame,
    evidence_group: str,
    dataset: str | None,
) -> int:
    """Return one evidence breakdown count from a Polars table."""
    dataset_filter = (
        pl.col(DATASET).is_null()
        if dataset is None
        else pl.col(DATASET) == dataset
    )
    row = breakdown.filter(
        (pl.col(EVIDENCE_GROUP) == evidence_group) & dataset_filter
    ).row(0, named=True)
    return int(row[FINDING_COUNT])


def _portfolio_flow_finding(*, delta: float, denominator: float | None) -> Finding:
    """Return a synthetic portfolio flow finding for reconciliation tests."""
    return Finding(
        code=PC_PORT_FLOW,
        severity=SEVERITY_MATERIAL,
        confidence=CONFIDENCE_HIGH,
        dataset=pc_cols.PORTFOLIO_PERFORMANCE,
        evidence_role=DIRECT_INPUT,
        portfolio_id="PORT_A",
        from_date=date(2025, 5, 1),
        thru_date=date(2025, 5, 31),
        source_column=pc_cols.FLOW,
        delta_b_minus_a=delta,
        return_denominator=denominator,
        message="portfolio_performance 'flow' changed.",
    )


def _transaction_cross_check_finding(*, estimate: float) -> Finding:
    """Return a synthetic transaction cross-check finding."""
    return Finding(
        code=PC_TXN_AMT,
        severity=SEVERITY_MATERIAL,
        confidence=CONFIDENCE_HIGH,
        dataset=pc_cols.TRANSACTIONS,
        evidence_role=DIRECT_INPUT,
        portfolio_id="PORT_A",
        security_id="AAPL",
        from_date=date(2025, 5, 1),
        thru_date=date(2025, 5, 31),
        source_column=pc_cols.AMOUNT,
        transaction_category="external_flow",
        transaction_impact_policy="external_flow:modified_dietz",
        transaction_impact_diagnostic="modified_dietz cross-check estimate",
        transaction_impact_diagnostic_estimate=estimate,
        delta_b_minus_a=10.0,
        message="transactions 'amount' changed.",
    )


if __name__ == "__main__":
    unittest.main()
