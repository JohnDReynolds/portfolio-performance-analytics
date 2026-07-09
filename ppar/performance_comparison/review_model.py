"""Shared vocabulary for performance comparison review artifacts.

This module contains presentation names that must stay aligned across the
generated XLSX workbook, HTML report, bundle README, validators, and tests.
Keeping the names here avoids drift when the review model is renamed.
"""

from __future__ import annotations

from typing import Final

HTML_REPORT_ARTIFACT: Final[str] = "html_report"
REVIEW_WORKBOOK_ARTIFACT: Final[str] = "review_workbook"
REVIEW_WORKBOOK_FILE_NAME: Final[str] = "report.xlsx"

PERFORMANCE_DIFFERENCES_ARTIFACT: Final[str] = "performance_differences"
PERFORMANCE_DIFFERENCE_CAUSES_ARTIFACT: Final[str] = "performance_difference_causes"
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

PRIMARY_REVIEW_SHEETS: Final[tuple[str, ...]] = (
    PERFORMANCE_DIFFERENCES_SHEET,
)
SHARED_REVIEW_SHEETS: Final[tuple[str, ...]] = (
    PERFORMANCE_DIFFERENCE_CAUSES_SHEET,
    X_REF_ISSUES_SHEET,
    SOURCE_DETAIL_SHEET,
)
EXPECTED_REVIEW_SHEETS: Final[tuple[str, ...]] = (
    *PRIMARY_REVIEW_SHEETS,
    *SHARED_REVIEW_SHEETS,
)
