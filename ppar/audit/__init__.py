"""Audit performance restatements and source-data integrity.

The package root exposes shared workflow entrypoints, source-data loaders, and
report-bundle handoff helpers. Specialized APIs live under the
``performance_comparison`` and ``data_issues`` sub-feature packages.
"""

from ppar.audit.bundle import (
    REPORT_BUNDLE_REQUIRED_ARTIFACTS,
    report_bundle_contract,
    report_bundle_validation_issues,
)
from ppar.audit import schema
from ppar.audit.fx_rates import FxRatesLoader
from ppar.audit.portfolio_performance import PortfolioPerformanceLoader
from ppar.audit.holdings import HoldingsLoader
from ppar.audit.report import (
    write_audit_report_bundle,
    write_audit_review_workbook,
)
from ppar.audit.runner import (
    compact_findings_table,
    compare_snapshots,
    summarize_findings,
    validate_causal_attribution_ready,
    validate_yaml_setup_complete,
)
from ppar.audit.security_performance import SecurityPerformanceLoader
from ppar.audit.specification import (
    ComparisonFile,
    ComparisonSnapshot,
    AuditSpecification,
)
from ppar.audit.transactions import TransactionsLoader
__all__ = [
    # Source-data loaders and comparison specification objects.
    "ComparisonFile",
    "ComparisonSnapshot",
    "FxRatesLoader",
    "HoldingsLoader",
    "AuditSpecification",
    "PortfolioPerformanceLoader",
    "SecurityPerformanceLoader",
    "TransactionsLoader",
    "compact_findings_table",
    "compare_snapshots",
    "schema",
    "summarize_findings",
    "validate_causal_attribution_ready",
    "validate_yaml_setup_complete",
    # Report and bundle handoff helpers.
    "REPORT_BUNDLE_REQUIRED_ARTIFACTS",
    "report_bundle_contract",
    "report_bundle_validation_issues",
    "write_audit_report_bundle",
    "write_audit_review_workbook",
]
