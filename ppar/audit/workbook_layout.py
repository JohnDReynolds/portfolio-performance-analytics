"""Define reviewer-facing Audit workbook columns, labels, and tooltips."""

from __future__ import annotations

# Python imports
from collections.abc import Sequence

# Third-party imports
import polars as pl

# Project imports
from ppar.audit.data_issues import checks as data_issue_checks
from ppar.audit.performance_comparison import explain
from ppar.audit.performance_comparison import findings
from ppar.audit.performance_comparison import return_reconstruction
from ppar.audit.review_keys import REVIEW_KEY

__all__ = [
    "REVIEW_KEY",
    "workbook_column_labels",
    "workbook_column_tooltip",
    "workbook_portfolio_changes_columns",
    "workbook_return_reconstruction_columns",
    "workbook_return_reconstruction_summary_columns",
    "workbook_security_changes_columns",
    "workbook_security_return_reconstruction_columns",
    "workbook_sorted_table",
    "workbook_underlying_cause_columns",
]

REVIEW_STATUS = "review_status"
REVIEW_CUES = "review_cues"
SUGGESTED_NEXT_STEP = "suggested_next_step"
REVIEW_DETAIL_ARTIFACTS = "review_detail_artifacts"
PERFORMANCE_CHANGE = "performance_change"
ESTIMATED_CAUSE_TOTAL = "estimated_cause_total"
UNEXPLAINED_CHANGE = "unexplained_change"
USE = "use"
USE_PRIORITY = "_use_priority"
CHANGE_LABEL = "change_label"
CHANGE = "change"
DATASET_FIELD = "dataset_field"
ROW_TYPE = "row_type"
INPUT_ROLE = "input_role"
AS_OF_DATE = "as_of_date"
ESTIMATED_IMPACT = "estimated_impact"
IMPACT_STATUS = "impact_status"
REVIEW_NOTE = "review_note"
REVIEW_GUIDANCE = "review_guidance"
CONTEXT_USE = "context_use"
REVIEW_PRIORITY = "review_priority"
REVIEW_PRIORITY_REASON = "review_priority_reason"
RETURN_IMPACT_TREATMENT = "return_impact_treatment"


def workbook_portfolio_changes_columns() -> tuple[str, ...]:
    """Return portfolio-level Performance Differences worksheet columns."""
    return (
        findings.PORTFOLIO_ID,
        findings.FROM_DATE,
        findings.THRU_DATE,
        PERFORMANCE_CHANGE,
        ESTIMATED_CAUSE_TOTAL,
        UNEXPLAINED_CHANGE,
        REVIEW_STATUS,
        REVIEW_NOTE,
        REVIEW_KEY,
    )


def workbook_security_changes_columns() -> tuple[str, ...]:
    """Return security-level Performance Differences worksheet columns."""
    return (
        findings.PORTFOLIO_ID,
        findings.FROM_DATE,
        findings.THRU_DATE,
        findings.SECURITY_ID,
        PERFORMANCE_CHANGE,
        ESTIMATED_CAUSE_TOTAL,
        UNEXPLAINED_CHANGE,
        REVIEW_STATUS,
        REVIEW_NOTE,
        REVIEW_KEY,
    )


def workbook_underlying_cause_columns() -> tuple[str, ...]:
    """Return Performance Difference Causes worksheet columns."""
    return (
        findings.PORTFOLIO_ID,
        findings.FROM_DATE,
        findings.THRU_DATE,
        AS_OF_DATE,
        DATASET_FIELD,
        findings.SECURITY_ID,
        findings.SNAPSHOT_A_VALUE,
        findings.SNAPSHOT_B_VALUE,
        CHANGE,
        ESTIMATED_IMPACT,
        REVIEW_GUIDANCE,
        REVIEW_KEY,
    )


def workbook_return_reconstruction_columns() -> tuple[str, ...]:
    """Return Return Reconstruction Checks worksheet columns."""
    return (
        return_reconstruction.RECONSTRUCTION_PORTFOLIO_ID,
        return_reconstruction.RECONSTRUCTION_FROM_DATE,
        return_reconstruction.RECONSTRUCTION_THRU_DATE,
        return_reconstruction.REPORTED_RETURN_A,
        return_reconstruction.REPORTED_RETURN_B,
        return_reconstruction.REPORTED_RETURN_DIFFERENCE,
        return_reconstruction.DERIVED_RETURN_A,
        return_reconstruction.DERIVED_RETURN_B,
        return_reconstruction.DERIVED_RETURN_DIFFERENCE,
        return_reconstruction.RECONSTRUCTION_DIFFERENCE,
        return_reconstruction.RECONSTRUCTION_STATUS,
        return_reconstruction.RECONSTRUCTION_CATEGORY,
        return_reconstruction.RECONSTRUCTION_COMMENTS,
        return_reconstruction.DERIVED_NUMERATOR_A,
        return_reconstruction.DERIVED_NUMERATOR_B,
        return_reconstruction.DERIVED_NUMERATOR_DIFFERENCE,
        return_reconstruction.DERIVED_DENOMINATOR_A,
        return_reconstruction.DERIVED_DENOMINATOR_B,
        return_reconstruction.DERIVED_DENOMINATOR_DIFFERENCE,
        return_reconstruction.BEGIN_VALUE_A,
        return_reconstruction.BEGIN_VALUE_B,
        return_reconstruction.BEGIN_VALUE_DIFFERENCE,
        return_reconstruction.END_VALUE_A,
        return_reconstruction.END_VALUE_B,
        return_reconstruction.END_VALUE_DIFFERENCE,
        return_reconstruction.NET_FLOW_A,
        return_reconstruction.NET_FLOW_B,
        return_reconstruction.NET_FLOW_DIFFERENCE,
        return_reconstruction.WEIGHTED_FLOW_A,
        return_reconstruction.WEIGHTED_FLOW_B,
        return_reconstruction.WEIGHTED_FLOW_DIFFERENCE,
        return_reconstruction.BEGIN_VALUE_DATE_A,
        return_reconstruction.BEGIN_VALUE_DATE_B,
        return_reconstruction.END_VALUE_DATE_A,
        return_reconstruction.END_VALUE_DATE_B,
        return_reconstruction.RECONSTRUCTION_REVIEW_KEY,
    )


def workbook_security_return_reconstruction_columns() -> tuple[str, ...]:
    """Return Security Return Reconstruction Checks worksheet columns."""
    return (
        return_reconstruction.RECONSTRUCTION_PORTFOLIO_ID,
        return_reconstruction.RECONSTRUCTION_SECURITY_ID,
        return_reconstruction.RECONSTRUCTION_FROM_DATE,
        return_reconstruction.RECONSTRUCTION_THRU_DATE,
        return_reconstruction.REPORTED_RETURN_A,
        return_reconstruction.REPORTED_RETURN_B,
        return_reconstruction.REPORTED_RETURN_DIFFERENCE,
        return_reconstruction.DERIVED_RETURN_A,
        return_reconstruction.DERIVED_RETURN_B,
        return_reconstruction.DERIVED_RETURN_DIFFERENCE,
        return_reconstruction.RECONSTRUCTION_DIFFERENCE,
        return_reconstruction.RECONSTRUCTION_STATUS,
        return_reconstruction.RECONSTRUCTION_CATEGORY,
        return_reconstruction.RECONSTRUCTION_COMMENTS,
        return_reconstruction.DERIVED_NUMERATOR_A,
        return_reconstruction.DERIVED_NUMERATOR_B,
        return_reconstruction.DERIVED_NUMERATOR_DIFFERENCE,
        return_reconstruction.DERIVED_DENOMINATOR_A,
        return_reconstruction.DERIVED_DENOMINATOR_B,
        return_reconstruction.DERIVED_DENOMINATOR_DIFFERENCE,
        return_reconstruction.BEGIN_VALUE_A,
        return_reconstruction.BEGIN_VALUE_B,
        return_reconstruction.BEGIN_VALUE_DIFFERENCE,
        return_reconstruction.END_VALUE_A,
        return_reconstruction.END_VALUE_B,
        return_reconstruction.END_VALUE_DIFFERENCE,
        return_reconstruction.NET_FLOW_A,
        return_reconstruction.NET_FLOW_B,
        return_reconstruction.NET_FLOW_DIFFERENCE,
        return_reconstruction.WEIGHTED_FLOW_A,
        return_reconstruction.WEIGHTED_FLOW_B,
        return_reconstruction.WEIGHTED_FLOW_DIFFERENCE,
        return_reconstruction.INCOME_A,
        return_reconstruction.INCOME_B,
        return_reconstruction.INCOME_DIFFERENCE,
        return_reconstruction.BEGIN_VALUE_DATE_A,
        return_reconstruction.BEGIN_VALUE_DATE_B,
        return_reconstruction.END_VALUE_DATE_A,
        return_reconstruction.END_VALUE_DATE_B,
        return_reconstruction.RECONSTRUCTION_REVIEW_KEY,
    )


def workbook_return_reconstruction_summary_columns() -> tuple[str, ...]:
    """Return Reconstruction Summary worksheet columns."""
    return (
        return_reconstruction.RECONSTRUCTION_CHECK_TYPE,
        return_reconstruction.RECONSTRUCTION_STATUS,
        return_reconstruction.RECONSTRUCTION_CATEGORY,
        return_reconstruction.RECONSTRUCTION_ROW_COUNT,
    )


def workbook_sorted_table(
    table: pl.DataFrame,
    columns: Sequence[str],
) -> pl.DataFrame:
    """Return a workbook table sorted by available reviewer-facing columns."""
    sort_columns = [column for column in columns if column in table.columns]
    if not sort_columns or table.is_empty():
        return table
    return table.sort(sort_columns, nulls_last=True)


def workbook_column_labels() -> dict[str, str]:
    """Return shared user-facing labels for review workbook columns."""
    return {
        REVIEW_KEY: "Review Key",
        findings.PORTFOLIO_ID: "Portfolio",
        findings.SECURITY_ID: "Security",
        data_issue_checks.SNAPSHOT: "Snapshot",
        data_issue_checks.ISSUE_TYPE: "Issue Type",
        data_issue_checks.VALUE_A: "Reference Value",
        data_issue_checks.VALUE_B: "Observed Value",
        data_issue_checks.DIFFERENCE: "Difference",
        data_issue_checks.TOLERANCE: "Tolerance",
        findings.FROM_DATE: "From Date",
        findings.THRU_DATE: "Thru Date",
        PERFORMANCE_CHANGE: "Performance Difference",
        ESTIMATED_CAUSE_TOTAL: "Explained Difference",
        UNEXPLAINED_CHANGE: "Unexplained Difference",
        AS_OF_DATE: "As Of Date",
        USE: "Purpose",
        CHANGE_LABEL: "What Changed",
        DATASET_FIELD: "Dataset.Field",
        ROW_TYPE: "Row Type",
        CHANGE: "B - A Difference",
        ESTIMATED_IMPACT: "Performance Difference Explained",
        IMPACT_STATUS: "Impact Status",
        REVIEW_NOTE: "Explanation",
        REVIEW_GUIDANCE: "Explanation",
        explain.PORTFOLIO_RETURN_DELTA: "Return Delta",
        REVIEW_STATUS: "Status",
        REVIEW_CUES: "Review Cues",
        SUGGESTED_NEXT_STEP: "Suggested Next Step",
        REVIEW_DETAIL_ARTIFACTS: "Review Detail Artifacts",
        CONTEXT_USE: "Context Use",
        REVIEW_PRIORITY: "Review Priority",
        REVIEW_PRIORITY_REASON: "Review Priority Reason",
        RETURN_IMPACT_TREATMENT: "Return Impact Treatment",
        findings.FINDING_CODE: "Code",
        findings.DATASET: "Source Dataset",
        findings.SOURCE_COLUMN: "Input Field",
        findings.MESSAGE: "Message",
        findings.SEVERITY: "Severity",
        findings.CONFIDENCE: "Confidence",
        findings.EVIDENCE_ROLE: "Evidence Role",
        findings.SOURCE_FILE: "Source File",
        findings.TRANSACTION_CATEGORY: "Transaction Category",
        findings.TRANSACTION_MATCH_STATUS: "Transaction Match Status",
        explain.TRANSACTION_MATCH_CONFIDENCE: "Match Confidence",
        explain.TRANSACTION_MATCH_INTERPRETATION: "Match Interpretation",
        explain.TRANSACTION_MATCH_REVIEW_NOTE: "Review Note",
        findings.SNAPSHOT_A_VALUE: "Snapshot A Value",
        findings.SNAPSHOT_B_VALUE: "Snapshot B Value",
        findings.DELTA_B_MINUS_A: "Delta B Minus A",
        findings.IMPACT_INPUT_VALUE: "Impact Input Value",
        findings.SUPPRESSED: "Suppressed",
        explain.ROOT_CAUSE_AREA: "Cause Area",
        explain.FINDING_COUNT: "Finding Count",
        explain.IMPACT_BASIS: "Impact Basis",
        explain.IMPACT_CONFIDENCE: "Confidence",
        explain.TOP_CODES: "Top Codes",
        explain.IMPACT_MESSAGE: "Impact Message",
        explain.REVIEW_RANK: "Review Rank",
        return_reconstruction.RECONSTRUCTION_REVIEW_KEY: "Review Key",
        return_reconstruction.REPORTED_RETURN_A: "Reported Return A",
        return_reconstruction.REPORTED_RETURN_B: "Reported Return B",
        return_reconstruction.REPORTED_RETURN_DIFFERENCE: "Reported Difference",
        return_reconstruction.DERIVED_RETURN_A: "Derived Return A",
        return_reconstruction.DERIVED_RETURN_B: "Derived Return B",
        return_reconstruction.DERIVED_RETURN_DIFFERENCE: "Derived Difference",
        return_reconstruction.RECONSTRUCTION_DIFFERENCE: "Reconstruction Difference",
        return_reconstruction.DERIVED_NUMERATOR_A: "Derived Numerator A",
        return_reconstruction.DERIVED_NUMERATOR_B: "Derived Numerator B",
        return_reconstruction.DERIVED_NUMERATOR_DIFFERENCE: (
            "Derived Numerator Difference"
        ),
        return_reconstruction.DERIVED_DENOMINATOR_A: "Derived Denominator A",
        return_reconstruction.DERIVED_DENOMINATOR_B: "Derived Denominator B",
        return_reconstruction.DERIVED_DENOMINATOR_DIFFERENCE: (
            "Derived Denominator Difference"
        ),
        return_reconstruction.BEGIN_VALUE_A: "Beginning Value A",
        return_reconstruction.BEGIN_VALUE_B: "Beginning Value B",
        return_reconstruction.BEGIN_VALUE_DIFFERENCE: "Beginning Value Difference",
        return_reconstruction.END_VALUE_A: "Ending Value A",
        return_reconstruction.END_VALUE_B: "Ending Value B",
        return_reconstruction.END_VALUE_DIFFERENCE: "Ending Value Difference",
        return_reconstruction.NET_FLOW_A: "Net Flow A",
        return_reconstruction.NET_FLOW_B: "Net Flow B",
        return_reconstruction.NET_FLOW_DIFFERENCE: "Net Flow Difference",
        return_reconstruction.WEIGHTED_FLOW_A: "Weighted Flow A",
        return_reconstruction.WEIGHTED_FLOW_B: "Weighted Flow B",
        return_reconstruction.WEIGHTED_FLOW_DIFFERENCE: "Weighted Flow Difference",
        return_reconstruction.INCOME_A: "Income A",
        return_reconstruction.INCOME_B: "Income B",
        return_reconstruction.INCOME_DIFFERENCE: "Income Difference",
        return_reconstruction.BEGIN_VALUE_DATE_A: "Beginning Value Date A",
        return_reconstruction.BEGIN_VALUE_DATE_B: "Beginning Value Date B",
        return_reconstruction.END_VALUE_DATE_A: "Ending Value Date A",
        return_reconstruction.END_VALUE_DATE_B: "Ending Value Date B",
        return_reconstruction.RECONSTRUCTION_STATUS: "Status",
        return_reconstruction.RECONSTRUCTION_CATEGORY: "Diagnostic Category",
        return_reconstruction.RECONSTRUCTION_COMMENTS: "Comments",
        return_reconstruction.RECONSTRUCTION_CHECK_TYPE: "Check Type",
        return_reconstruction.RECONSTRUCTION_ROW_COUNT: "Row Count",
    }


def workbook_column_tooltip(column: str) -> str:
    """Return explanatory header text for a workbook/report column.

    Args:
        column: Internal workbook-table column name.

    Returns:
        Reviewer-facing explanation suitable for XLSX comments and HTML header
        tooltips.
    """
    tooltips = {
        REVIEW_KEY: "Stable performance-period key used to connect workbook rows.",
        findings.PORTFOLIO_ID: (
            "Portfolio identifier from the compared source-data."
        ),
        findings.FROM_DATE: "Beginning date of the affected performance period.",
        findings.THRU_DATE: "Ending date of the affected performance period.",
        findings.SECURITY_ID: (
            "Security identifier, when the discrepancy is security-level."
        ),
        data_issue_checks.SNAPSHOT: (
            "Snapshot whose internal source-data is being checked."
        ),
        data_issue_checks.ISSUE_TYPE: (
            "Type of cross-reference consistency issue."
        ),
        data_issue_checks.VALUE_A: (
            "Expected value or minimum rate found for this consistency check."
        ),
        data_issue_checks.VALUE_B: (
            "Observed value or maximum rate found for this consistency check."
        ),
        data_issue_checks.DIFFERENCE: (
            "Observed value minus expected value, or maximum rate minus minimum rate."
        ),
        data_issue_checks.TOLERANCE: (
            "Configured threshold before the consistency check raises an issue."
        ),
        findings.SEVERITY: "Materiality/severity assigned to this discrepancy.",
        PERFORMANCE_CHANGE: (
            "Snapshot B reported performance minus snapshot A reported performance."
        ),
        ESTIMATED_CAUSE_TOTAL: (
            'Total performance difference explained by "Performance Difference '
            'Causes" sheet rows.'
        ),
        UNEXPLAINED_CHANGE: "Performance difference less explained difference.",
        USE: "Workbook row category used for sorting and compatibility.",
        CHANGE_LABEL: "Plain-English changed data item.",
        DATASET_FIELD: (
            "Changed input field, shown as dataset.field. In detailed datasets, "
            "unqualified monetary fields use the row currency and base_ fields "
            "use portfolio base currency. Portfolio-performance monetary fields "
            "are inherently base-currency values."
        ),
        ROW_TYPE: "Internal reviewer role used for row coloring and sorting.",
        CHANGE: "Snapshot B value minus snapshot A value for the compared item.",
        AS_OF_DATE: (
            "Date represented by the input row. Holding rows use the period Thru Date."
        ),
        ESTIMATED_IMPACT: (
            "Decimal performance difference explained by this underlying input row."
        ),
        IMPACT_STATUS: (
            "Whether this row has an additive estimate, is missing an impact method, "
            "or is review-only."
        ),
        REVIEW_NOTE: "Plain-language explanation for this changed item.",
        REVIEW_GUIDANCE: (
            "Plain-language explanation of what changed and how this row relates "
            "to the performance difference."
        ),
        explain.PORTFOLIO_RETURN_DELTA: (
            "Snapshot B reported performance minus snapshot A reported performance."
        ),
        REVIEW_STATUS: "Reviewer triage status for this performance difference.",
        explain.ROOT_CAUSE_AREA: (
            "Coarse explanation bucket for a group of findings."
        ),
        explain.FINDING_COUNT: "Number of finding rows grouped into this cause.",
        explain.IMPACT_BASIS: "Method basis used to estimate return impact.",
        explain.IMPACT_CONFIDENCE: "Confidence level for the estimated impact.",
        explain.TOP_CODES: "Most relevant finding codes represented by this row.",
        explain.IMPACT_MESSAGE: "Explanation of the impact estimate or limitation.",
        explain.REVIEW_RANK: "Priority rank within the portfolio period.",
        findings.FINDING_CODE: "Stable finding code for the discrepancy type.",
        findings.CONFIDENCE: (
            "Confidence level for the finding or impact interpretation."
        ),
        findings.DATASET: (
            "Normalized dataset where the source-data discrepancy was found."
        ),
        findings.EVIDENCE_ROLE: (
            "Whether the finding is target output, direct input, related output, "
            "or context."
        ),
        findings.SOURCE_FILE: (
            "Source file path or dataset file where applicable."
        ),
        findings.SOURCE_COLUMN: (
            "Normalized source column that changed or was relevant."
        ),
        findings.TRANSACTION_CATEGORY: (
            "Normalized transaction category, when applicable."
        ),
        findings.CASH_FLOW_SIGN: (
            "Configured or source cash-flow sign, when applicable."
        ),
        findings.PERFORMANCE_FLOW_SIGN: (
            "Configured or source performance-flow sign, when applicable."
        ),
        findings.TRANSACTION_SEMANTICS_SOURCE: (
            "Where transaction sign/category semantics came from."
        ),
        findings.TRANSACTION_MATCH_STATUS: (
            "How transaction rows were matched between snapshots."
        ),
        explain.TRANSACTION_MATCH_CONFIDENCE: (
            "Reviewer-facing confidence tier for interpreting transaction row identity."
        ),
        explain.TRANSACTION_MATCH_INTERPRETATION: (
            "Short description of what the match status permits reviewers to infer."
        ),
        explain.TRANSACTION_MATCH_REVIEW_NOTE: (
            "Plain-language note explaining the transaction match status."
        ),
        findings.IMPACT_POLICY: (
            "Contribution/return impact policy used for this finding."
        ),
        findings.TRANSACTION_IMPACT_POLICY: (
            "Transaction impact policy used for this finding."
        ),
        findings.TRANSACTION_IMPACT_DIAGNOSTIC: (
            "Review-only transaction diagnostic name, when available."
        ),
        findings.TRANSACTION_IMPACT_DIAGNOSTIC_ESTIMATE: (
            "Review-only transaction diagnostic estimate, when available."
        ),
        findings.SNAPSHOT_A_VALUE: "Value observed in snapshot A.",
        findings.SNAPSHOT_B_VALUE: "Value observed in snapshot B.",
        findings.DELTA_B_MINUS_A: (
            "Numeric difference calculated as snapshot B minus A."
        ),
        findings.RETURN_DENOMINATOR: (
            "Denominator used for return-impact estimates, when configured."
        ),
        findings.RETURN_WEIGHT: (
            "Weight used for security return-impact estimates, when available."
        ),
        findings.IMPACT_INPUT_VALUE: (
            "Additional numeric input used by the selected impact method, when needed."
        ),
        findings.MESSAGE: "Human-readable finding detail.",
        findings.SUPPRESSED: (
            "Whether a configured suppression marked this finding hidden."
        ),
    }
    return tooltips.get(
        column,
        f"Workbook column derived from normalized ppar field `{column}`.",
    )
