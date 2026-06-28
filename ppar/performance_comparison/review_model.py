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
OTHER_DATA_DIFFERENCES_ARTIFACT: Final[str] = "other_data_differences"
RAW_AUDIT_TRAIL_ARTIFACT: Final[str] = "raw_audit_trail"
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
OTHER_DATA_DIFFERENCES_SHEET: Final[str] = "Other Data Differences"
RAW_AUDIT_TRAIL_SHEET: Final[str] = "Raw Audit Trail"
REVIEW_ORDER_SECTION: Final[str] = "Review Order"

PRIMARY_REVIEW_SHEETS: Final[tuple[str, ...]] = (
    PERFORMANCE_DIFFERENCES_SHEET,
)
SHARED_REVIEW_SHEETS: Final[tuple[str, ...]] = (
    PERFORMANCE_DIFFERENCE_CAUSES_SHEET,
    OTHER_DATA_DIFFERENCES_SHEET,
    RAW_AUDIT_TRAIL_SHEET,
)
EXPECTED_REVIEW_SHEETS: Final[tuple[str, ...]] = (
    *PRIMARY_REVIEW_SHEETS,
    *SHARED_REVIEW_SHEETS,
)
