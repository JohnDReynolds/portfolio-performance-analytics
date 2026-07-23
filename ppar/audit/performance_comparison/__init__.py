"""Explain reported performance differences between two Audit snapshots."""

from ppar.audit.performance_comparison.compare import PerformanceComparison
from ppar.audit.performance_comparison.explain import (
    portfolio_period_cause_summary,
    portfolio_period_contribution_candidates,
    portfolio_period_evidence_breakdown,
    portfolio_period_impact_coverage_summary,
    portfolio_period_summary,
    portfolio_period_transaction_cross_checks,
    rank_portfolio_period_evidence,
    security_period_evidence_breakdown,
    security_period_summary,
    transaction_activity_summary,
    transaction_matching_diagnostics,
)
from ppar.audit.performance_comparison.findings import (
    CONTEXT,
    DIRECT_INPUT,
    EVIDENCE_ROLE,
    Finding,
    RELATED_OUTPUT,
    TARGET_OUTPUT,
    findings_to_polars,
)
from ppar.audit.performance_comparison.rules import (
    SuppressionRule,
    apply_suppressions,
)
from ppar.audit.performance_comparison.vocabulary import CauseArea

__all__ = [
    "CONTEXT",
    "DIRECT_INPUT",
    "EVIDENCE_ROLE",
    "RELATED_OUTPUT",
    "TARGET_OUTPUT",
    "CauseArea",
    "Finding",
    "PerformanceComparison",
    "SuppressionRule",
    "apply_suppressions",
    "findings_to_polars",
    "portfolio_period_cause_summary",
    "portfolio_period_contribution_candidates",
    "portfolio_period_evidence_breakdown",
    "portfolio_period_impact_coverage_summary",
    "portfolio_period_summary",
    "portfolio_period_transaction_cross_checks",
    "rank_portfolio_period_evidence",
    "security_period_evidence_breakdown",
    "security_period_summary",
    "transaction_activity_summary",
    "transaction_matching_diagnostics",
]
