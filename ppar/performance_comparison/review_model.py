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
IDENTIFIABLE_CAUSES_ARTIFACT: Final[str] = "identifiable_causes"
OTHER_EVIDENCE_ARTIFACT: Final[str] = "other_evidence"
RAW_AUDIT_TRAIL_ARTIFACT: Final[str] = "raw_audit_trail"

PERFORMANCE_DIFFERENCES_SHEET: Final[str] = "Performance Differences"
IDENTIFIABLE_CAUSES_SHEET: Final[str] = "Identifiable Causes"
OTHER_EVIDENCE_SHEET: Final[str] = "Other Evidence"
RAW_AUDIT_TRAIL_SHEET: Final[str] = "Raw Audit Trail"
REVIEW_ORDER_SECTION: Final[str] = "Review Order"

PRIMARY_REVIEW_SHEETS: Final[tuple[str, ...]] = (
    PERFORMANCE_DIFFERENCES_SHEET,
)
SHARED_REVIEW_SHEETS: Final[tuple[str, ...]] = (
    IDENTIFIABLE_CAUSES_SHEET,
    OTHER_EVIDENCE_SHEET,
    RAW_AUDIT_TRAIL_SHEET,
)
EXPECTED_REVIEW_SHEETS: Final[tuple[str, ...]] = (
    *PRIMARY_REVIEW_SHEETS,
    *SHARED_REVIEW_SHEETS,
)
