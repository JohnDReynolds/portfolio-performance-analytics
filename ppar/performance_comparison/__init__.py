"""Compare performance snapshots and explain restatements.

The package root intentionally exposes workflow entrypoints, source-data
loaders, core finding objects, and report-bundle handoff helpers. More surgical
policy, validation, and evidence-pack helpers remain importable as direct
submodules so the root API does not grow with every internal review aid.
"""

from ppar.performance_comparison.bundle import (
    REPORT_BUNDLE_REQUIRED_ARTIFACTS,
    report_bundle_contract,
    report_bundle_validation_issues,
)
from ppar.performance_comparison import schema
from ppar.performance_comparison.compare import PerformanceComparison
from ppar.performance_comparison.explain import (
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
from ppar.performance_comparison.findings import (
    CONTEXT,
    DIRECT_INPUT,
    EVIDENCE_ROLE,
    Finding,
    RELATED_OUTPUT,
    TARGET_OUTPUT,
    findings_to_polars,
)
from ppar.performance_comparison.fx_rates import FxRatesLoader
from ppar.performance_comparison.portfolio_performance import PortfolioPerformanceLoader
from ppar.performance_comparison.holdings import HoldingsLoader
from ppar.performance_comparison.report import (
    write_performance_comparison_report_bundle,
    write_performance_comparison_review_workbook,
)
from ppar.performance_comparison.rules import SuppressionRule, apply_suppressions
from ppar.performance_comparison.runner import (
    compact_findings_table,
    compare_snapshots,
    summarize_findings,
    validate_causal_attribution_ready,
    validate_yaml_setup_complete,
)
from ppar.performance_comparison.security_performance import SecurityPerformanceLoader
from ppar.performance_comparison.specification import (
    ComparisonFile,
    ComparisonSnapshot,
    PerformanceComparisonSpecification,
)
from ppar.performance_comparison.transactions import TransactionsLoader

__all__ = [
    # Source-data loaders and comparison specification objects.
    "ComparisonFile",
    "ComparisonSnapshot",
    "FxRatesLoader",
    "HoldingsLoader",
    "PerformanceComparisonSpecification",
    "PortfolioPerformanceLoader",
    "SecurityPerformanceLoader",
    "TransactionsLoader",
    # Core comparison and finding model.
    "CONTEXT",
    "DIRECT_INPUT",
    "EVIDENCE_ROLE",
    "Finding",
    "PerformanceComparison",
    "RELATED_OUTPUT",
    "TARGET_OUTPUT",
    "compact_findings_table",
    "compare_snapshots",
    "findings_to_polars",
    "schema",
    "summarize_findings",
    "validate_causal_attribution_ready",
    "validate_yaml_setup_complete",
    # Report and bundle handoff helpers.
    "REPORT_BUNDLE_REQUIRED_ARTIFACTS",
    "report_bundle_contract",
    "report_bundle_validation_issues",
    "write_performance_comparison_report_bundle",
    "write_performance_comparison_review_workbook",
    # Suppression and explanation helpers kept for compatibility with existing
    # review automation.
    "IMPACT_BASIS_PORTFOLIO_SOURCE_FIELD",
    "IMPACT_BASIS_SECURITY_RETURN_WEIGHTED",
    "IMPACT_METHOD_SECURITY_RETURN_DELTA_TIMES_WEIGHT",
    "IMPACT_METHOD_SOURCE_FIELD_DELTA_OVER_BEGIN_MV",
    "SuppressionRule",
    "apply_suppressions",
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
