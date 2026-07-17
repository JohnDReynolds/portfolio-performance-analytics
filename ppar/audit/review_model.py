"""Shared vocabulary for performance comparison review artifacts.

This module contains presentation names that must stay aligned across the
generated XLSX workbook, HTML report, bundle README, validators, and tests.
Keeping the names here avoids drift when the review model is renamed.
"""

from __future__ import annotations

from typing import Final

from ppar.performance_comparison.specification import (
    PORTFOLIO_COMPARISON_LEVEL,
    SECURITY_COMPARISON_LEVEL,
)

HTML_REPORT_ARTIFACT: Final[str] = "html_report"
REVIEW_WORKBOOK_ARTIFACT: Final[str] = "review_workbook"
PORTFOLIO_AUDIT_FILE_STEM: Final[str] = "portfolio_audit"
SECURITY_AUDIT_FILE_STEM: Final[str] = "security_audit"

PERFORMANCE_DIFFERENCES_ARTIFACT: Final[str] = "performance_differences"
PERFORMANCE_DIFFERENCE_CAUSES_ARTIFACT: Final[str] = "performance_difference_causes"
CAUSE_LINEAGE_ARTIFACT: Final[str] = "cause_lineage"
X_REF_ISSUES_ARTIFACT: Final[str] = "x_ref_issues"
SOURCE_DETAIL_ARTIFACT: Final[str] = "raw_audit_trail"
TRANSACTION_MATCHING_DIAGNOSTICS_ARTIFACT: Final[str] = (
    "transaction_matching_diagnostics"
)
RECONSTRUCTION_SUMMARY_ARTIFACT: Final[str] = "reconstruction_summary"
RETURN_RECONSTRUCTION_CHECKS_ARTIFACT: Final[str] = "return_reconstruction_checks"
SECURITY_RETURN_RECONSTRUCTION_CHECKS_ARTIFACT: Final[str] = (
    "security_return_reconstruction_checks"
)

PERFORMANCE_DIFFERENCES_SHEET: Final[str] = "Performance Differences"
RECONSTRUCTION_SUMMARY_SHEET: Final[str] = "Reconstruction Summary"
RETURN_RECONSTRUCTION_CHECKS_SHEET: Final[str] = "Return Reconstruction Checks"
SECURITY_RETURN_RECONSTRUCTION_CHECKS_SHEET: Final[str] = "Security Return Checks"
PERFORMANCE_DIFFERENCE_CAUSES_SHEET: Final[str] = "Performance Difference Causes"
X_REF_ISSUES_SHEET: Final[str] = "Data Audit Issues"
TRANSACTION_MATCHING_DIAGNOSTICS_SHEET: Final[str] = (
    "Transaction Match Diagnostics"
)
SOURCE_DETAIL_SHEET: Final[str] = "Source Detail"
REVIEW_ORDER_SECTION: Final[str] = "Review Order"


def audit_file_stem(comparison_level: str) -> str:
    """Return the audit report filename stem for a comparison level.

    Args:
        comparison_level: Portfolio or security comparison level.

    Returns:
        ``portfolio_audit`` or ``security_audit``.

    Raises:
        ValueError: If the comparison level is unsupported.
    """
    if comparison_level == PORTFOLIO_COMPARISON_LEVEL:
        return PORTFOLIO_AUDIT_FILE_STEM
    if comparison_level == SECURITY_COMPARISON_LEVEL:
        return SECURITY_AUDIT_FILE_STEM
    raise ValueError(f"Unsupported comparison level: {comparison_level!r}")


def html_report_file_name(comparison_level: str) -> str:
    """Return the HTML audit filename for a comparison level."""
    return f"{audit_file_stem(comparison_level)}.html"


def review_workbook_file_name(comparison_level: str) -> str:
    """Return the XLSX audit filename for a comparison level."""
    return f"{audit_file_stem(comparison_level)}.xlsx"

PRIMARY_REVIEW_SHEETS: Final[tuple[str, ...]] = (
    PERFORMANCE_DIFFERENCES_SHEET,
)
SHARED_REVIEW_SHEETS: Final[tuple[str, ...]] = (
    PERFORMANCE_DIFFERENCE_CAUSES_SHEET,
    X_REF_ISSUES_SHEET,
)
EXPECTED_REVIEW_SHEETS: Final[tuple[str, ...]] = (
    *PRIMARY_REVIEW_SHEETS,
    *SHARED_REVIEW_SHEETS,
)
