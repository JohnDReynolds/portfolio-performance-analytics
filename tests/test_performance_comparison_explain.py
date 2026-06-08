"""Tests for performance comparison explanation helpers."""

# Python imports
from pathlib import Path
import unittest

# Third-party imports
import polars as pl

# Project imports
from ppar.performance_comparison import (
    compare_snapshots,
    portfolio_period_cause_summary,
    portfolio_period_contribution_candidates,
    portfolio_period_evidence_breakdown,
    portfolio_period_summary,
    rank_portfolio_period_evidence,
    security_period_evidence_breakdown,
    security_period_summary,
    transaction_activity_summary,
)
from ppar.performance_comparison import explain as pc_explain
from ppar.performance_comparison.explain import (
    CASH_FINDING_COUNT,
    AMOUNT_DELTA,
    CHANGED_FIELDS,
    CONTEXT_FINDING_COUNT,
    DIRECT_INPUT_FINDING_COUNT,
    EVIDENCE_GROUP,
    ESTIMATED_RETURN_IMPACT,
    FINDING_COUNT,
    FX_RATE_FINDING_COUNT,
    HAS_SUPPRESSED_FINDINGS,
    IMPACT_BASIS,
    IMPACT_BASIS_NO_ESTIMATE,
    IMPACT_BASIS_SECURITY_CONTRIBUTION,
    IMPACT_CONFIDENCE,
    IMPACT_CONFIDENCE_LOW,
    IMPACT_CONFIDENCE_MEDIUM,
    IMPACT_MESSAGE,
    IMPACT_METHOD,
    IMPACT_METHOD_VENDOR_CONTRIBUTION_DELTA,
    PORTFOLIO_PERIOD_CAUSE_SUMMARY_COLUMNS,
    PORTFOLIO_FINDING_COUNT,
    PORTFOLIO_PERIOD_CONTRIBUTION_CANDIDATE_COLUMNS,
    PORTFOLIO_PERIOD_EVIDENCE_BREAKDOWN_COLUMNS,
    PORTFOLIO_PERIOD_EVIDENCE_RANKING_COLUMNS,
    PORTFOLIO_PERIOD_SUMMARY_COLUMNS,
    PORTFOLIO_RETURN_DELTA,
    POSITION_FINDING_COUNT,
    PRICE_DELTA,
    PRICE_FINDING_COUNT,
    PRIORITY_SCORE,
    QUANTITY_DELTA,
    REFERENCE_FINDING_COUNT,
    RELATED_OUTPUT_FINDING_COUNT,
    REVIEW_RANK,
    ROOT_CAUSE_AREA,
    ROOT_CAUSE_CASH,
    ROOT_CAUSE_MARKET_VALUE_OR_POSITION,
    ROOT_CAUSE_PORTFOLIO_PERFORMANCE_INPUT,
    ROOT_CAUSE_PRICE,
    ROOT_CAUSE_SECURITY_RETURN_OR_CONTRIBUTION,
    ROOT_CAUSE_TRANSACTION_ACTIVITY,
    SECURITY_FINDING_COUNT,
    SECURITY_PERIOD_EVIDENCE_BREAKDOWN_COLUMNS,
    SECURITY_PERIOD_SUMMARY_COLUMNS,
    SECURITY_RETURN_DELTA,
    TRANSACTION_FINDING_COUNT,
    TRANSACTION_ACTIVITY_SUMMARY_COLUMNS,
    TOP_CODES,
)
from ppar.performance_comparison.findings import (
    CONTEXT,
    DATASET,
    DIRECT_INPUT,
    EVIDENCE_ROLE,
    FINDING_CODE,
    PC_SEC_CONTR,
    PC_SEC_RET,
    PORTFOLIO_ID,
    RELATED_OUTPUT,
    SECURITY_ID,
    TARGET_OUTPUT,
    TRANSACTION_CATEGORY,
)

_BASELINE_COMPARISON_PATH = Path("tests/data/axys/ppar_performance_comparison.yaml")
_RESTATEMENT_COMPARISON_PATH = Path(
    "tests/data/axys/ppar_performance_comparison_restatement.yaml"
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
            pc_explain.rank_portfolio_period_evidence,
            rank_portfolio_period_evidence,
        )
        self.assertIs(pc_explain.security_period_summary, security_period_summary)
        self.assertIs(
            pc_explain.security_period_evidence_breakdown,
            security_period_evidence_breakdown,
        )
        self.assertIs(pc_explain.transaction_activity_summary, transaction_activity_summary)

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
        self.assertEqual(row[CONTEXT_FINDING_COUNT], 0)
        self.assertEqual(row[POSITION_FINDING_COUNT], 3)
        self.assertEqual(row[CASH_FINDING_COUNT], 2)
        self.assertEqual(row[TRANSACTION_FINDING_COUNT], 3)
        self.assertEqual(row[PRICE_FINDING_COUNT], 1)
        self.assertEqual(row[FX_RATE_FINDING_COUNT], 0)
        self.assertEqual(row[REFERENCE_FINDING_COUNT], 0)
        self.assertEqual(row[FINDING_COUNT], 17)
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
        self.assertEqual(_breakdown_count(breakdown, CONTEXT, None), 0)
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
        self.assertEqual(ranking.height, 16)
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
        self.assertEqual(candidates.height, 16)
        self.assertAlmostEqual(contribution[ESTIMATED_RETURN_IMPACT], 0.00058425)
        self.assertEqual(contribution[IMPACT_BASIS], IMPACT_BASIS_SECURITY_CONTRIBUTION)
        self.assertEqual(contribution[IMPACT_CONFIDENCE], IMPACT_CONFIDENCE_MEDIUM)
        self.assertEqual(
            contribution[IMPACT_METHOD],
            IMPACT_METHOD_VENDOR_CONTRIBUTION_DELTA,
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

        self.assertEqual(no_estimate.height, 15)
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
        self.assertIn("transaction-type sign", transaction_row[IMPACT_MESSAGE])

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
                ROOT_CAUSE_MARKET_VALUE_OR_POSITION: 3,
                ROOT_CAUSE_PORTFOLIO_PERFORMANCE_INPUT: 2,
                ROOT_CAUSE_PRICE: 1,
                ROOT_CAUSE_SECURITY_RETURN_OR_CONTRIBUTION: 5,
                ROOT_CAUSE_TRANSACTION_ACTIVITY: 3,
            },
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
        self.assertIsNone(row["from_date"])
        self.assertIsNone(row["thru_date"])
        self.assertEqual(row[TRANSACTION_CATEGORY], "buy")
        self.assertEqual(row[FINDING_COUNT], 3)
        self.assertEqual(row[CHANGED_FIELDS], "amount, quantity, price")
        self.assertAlmostEqual(row[AMOUNT_DELTA], -100.0)
        self.assertAlmostEqual(row[QUANTITY_DELTA], 1.0)
        self.assertAlmostEqual(row[PRICE_DELTA], 0.5)

    def test_transaction_activity_summary_is_evidence_only(self) -> None:
        """Transaction activity summary does not estimate return impact."""
        findings = self._restatement()

        summary = transaction_activity_summary(findings)
        row = summary.row(0, named=True)

        self.assertEqual(row[IMPACT_BASIS], IMPACT_BASIS_NO_ESTIMATE)
        self.assertEqual(row[IMPACT_CONFIDENCE], IMPACT_CONFIDENCE_LOW)
        self.assertIn("evidence-only", row[IMPACT_MESSAGE])

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

        self.assertEqual(active_ranking.height, 15)
        self.assertEqual(audit_ranking.height, 16)
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

        self.assertEqual(ranking.height, 11)
        self.assertEqual(set(ranking.get_column(EVIDENCE_ROLE).to_list()), {DIRECT_INPUT})
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
        self.assertEqual(row[CONTEXT_FINDING_COUNT], 2)
        self.assertEqual(row[POSITION_FINDING_COUNT], 3)
        self.assertEqual(row[TRANSACTION_FINDING_COUNT], 3)
        self.assertEqual(row[PRICE_FINDING_COUNT], 1)
        self.assertEqual(row[REFERENCE_FINDING_COUNT], 2)
        self.assertEqual(row[FINDING_COUNT], 12)
        self.assertFalse(row[HAS_SUPPRESSED_FINDINGS])

    def test_security_period_summary_tracks_suppressed_related_evidence(self) -> None:
        """Security-period summary flags suppressed related findings."""
        findings = self._suppressed()

        active_summary = security_period_summary(findings)
        audit_summary = security_period_summary(findings, include_suppressed=True)

        self.assertTrue(active_summary.row(0, named=True)[HAS_SUPPRESSED_FINDINGS])
        self.assertEqual(active_summary.row(0, named=True)[RELATED_OUTPUT_FINDING_COUNT], 2)
        self.assertEqual(audit_summary.row(0, named=True)[RELATED_OUTPUT_FINDING_COUNT], 3)
        self.assertEqual(active_summary.row(0, named=True)[FINDING_COUNT], 11)
        self.assertEqual(audit_summary.row(0, named=True)[FINDING_COUNT], 12)

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
        self.assertEqual(_breakdown_count(breakdown, CONTEXT, None), 2)
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


if __name__ == "__main__":
    unittest.main()
