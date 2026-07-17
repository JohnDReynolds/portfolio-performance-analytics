"""Explain reported performance differences between two Audit snapshots."""

from ppar.audit.performance_comparison.compare import PerformanceComparison
from ppar.audit.performance_comparison.explain import (
    IMPACT_BASIS_PORTFOLIO_SOURCE_FIELD,
    IMPACT_BASIS_SECURITY_RETURN_WEIGHTED,
    IMPACT_METHOD_SECURITY_RETURN_DELTA_TIMES_WEIGHT,
    IMPACT_METHOD_SOURCE_FIELD_DELTA_OVER_BEGIN_MV,
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
    "IMPACT_BASIS_PORTFOLIO_SOURCE_FIELD",
    "IMPACT_BASIS_SECURITY_RETURN_WEIGHTED",
    "IMPACT_METHOD_SECURITY_RETURN_DELTA_TIMES_WEIGHT",
    "IMPACT_METHOD_SOURCE_FIELD_DELTA_OVER_BEGIN_MV",
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
    "portfolio_period_flow_cross_check_reconciliation",
    "portfolio_period_impact_coverage_summary",
    "portfolio_period_summary",
    "portfolio_period_transaction_cross_checks",
    "rank_portfolio_period_evidence",
    "security_period_evidence_breakdown",
    "security_period_summary",
    "transaction_activity_summary",
    "transaction_matching_diagnostics",
]
