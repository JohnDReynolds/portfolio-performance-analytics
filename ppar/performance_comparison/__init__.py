"""Compare performance snapshots and explain restatements."""

from ppar.performance_comparison.cash import CashLoader
from ppar.performance_comparison import columns
from ppar.performance_comparison.compare import PerformanceComparison
from ppar.performance_comparison.explain import (
    IMPACT_BASIS_PORTFOLIO_SOURCE_FIELD,
    IMPACT_BASIS_SECURITY_RETURN_WEIGHTED,
    IMPACT_METHOD_SECURITY_RETURN_DELTA_TIMES_WEIGHT,
    IMPACT_METHOD_SOURCE_FIELD_DELTA_OVER_BEGIN_MV,
    portfolio_period_cause_summary,
    portfolio_period_contribution_candidates,
    portfolio_period_evidence_breakdown,
    portfolio_period_impact_coverage_summary,
    portfolio_period_summary,
    rank_portfolio_period_evidence,
    security_period_evidence_breakdown,
    security_period_summary,
    transaction_activity_summary,
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
from ppar.performance_comparison.positions import PositionsLoader
from ppar.performance_comparison.prices import PricesLoader
from ppar.performance_comparison.report import (
    performance_comparison_html_report,
    performance_comparison_markdown_report,
    write_performance_comparison_html_report,
    write_performance_comparison_markdown_report,
    write_performance_comparison_report_bundle,
)
from ppar.performance_comparison.rules import SuppressionRule, apply_suppressions
from ppar.performance_comparison.runner import (
    compact_findings_table,
    compare_snapshots,
    summarize_findings,
)
from ppar.performance_comparison.security_performance import SecurityPerformanceLoader
from ppar.performance_comparison.security_master import SecurityMasterLoader
from ppar.performance_comparison.specification import (
    ComparisonFile,
    ComparisonSnapshot,
    PerformanceComparisonSpecification,
)
from ppar.performance_comparison.transactions import TransactionsLoader

__all__ = [
    "CashLoader",
    "ComparisonFile",
    "ComparisonSnapshot",
    "CONTEXT",
    "DIRECT_INPUT",
    "EVIDENCE_ROLE",
    "Finding",
    "FxRatesLoader",
    "IMPACT_BASIS_PORTFOLIO_SOURCE_FIELD",
    "IMPACT_BASIS_SECURITY_RETURN_WEIGHTED",
    "IMPACT_METHOD_SECURITY_RETURN_DELTA_TIMES_WEIGHT",
    "IMPACT_METHOD_SOURCE_FIELD_DELTA_OVER_BEGIN_MV",
    "PerformanceComparison",
    "PortfolioPerformanceLoader",
    "PerformanceComparisonSpecification",
    "PositionsLoader",
    "PricesLoader",
    "SecurityPerformanceLoader",
    "SecurityMasterLoader",
    "SuppressionRule",
    "RELATED_OUTPUT",
    "TARGET_OUTPUT",
    "TransactionsLoader",
    "apply_suppressions",
    "columns",
    "compact_findings_table",
    "compare_snapshots",
    "findings_to_polars",
    "portfolio_period_cause_summary",
    "portfolio_period_contribution_candidates",
    "portfolio_period_evidence_breakdown",
    "portfolio_period_impact_coverage_summary",
    "portfolio_period_summary",
    "performance_comparison_html_report",
    "performance_comparison_markdown_report",
    "rank_portfolio_period_evidence",
    "security_period_evidence_breakdown",
    "security_period_summary",
    "summarize_findings",
    "transaction_activity_summary",
    "write_performance_comparison_html_report",
    "write_performance_comparison_markdown_report",
    "write_performance_comparison_report_bundle",
]
