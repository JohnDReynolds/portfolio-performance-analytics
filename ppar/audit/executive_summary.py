"""Build the canonical Audit Executive Summary review table.

The summary is a bounded presentation and navigation layer over existing
validated review tables. It deliberately performs no financial calculation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import polars as pl

import ppar.utilities as util
from ppar.errors import PpaError
from ppar.audit import rendering
from ppar.audit import review_model
from ppar.audit.data_issues import checks as data_issue_checks
from ppar.audit.data_issues.vocabulary import (
    DATA_ISSUE_REGISTRY,
    DataIssueCategory,
    DataIssueType,
)
from ppar.audit.performance_comparison import explain
from ppar.audit.performance_comparison import findings
from ppar.audit.performance_comparison.vocabulary import CauseArea
from ppar.audit.specification import (
    AuditSpecification,
    PORTFOLIO_COMPARISON_LEVEL,
    SECURITY_COMPARISON_LEVEL,
)

SUMMARY_SECTION: Final[str] = "summary_section"
SUMMARY_ITEM: Final[str] = "summary_item"
SUMMARY_RESULT: Final[str] = "summary_result"
SUMMARY_DETAIL: Final[str] = "summary_detail"
REVIEW_DESTINATION: Final[str] = "review_destination"
REVIEW_KEY: Final[str] = "review_key"
EXECUTIVE_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    SUMMARY_SECTION,
    SUMMARY_ITEM,
    SUMMARY_RESULT,
    SUMMARY_DETAIL,
    REVIEW_DESTINATION,
    REVIEW_KEY,
)
PRIORITY_REVIEW_UNIT_LIMIT: Final[int] = 10

_STATUS_ORDER = {
    "Unexplained": 0,
    "Partly Explained": 1,
    "Missing YAML Specifications": 2,
    "Fully Explained": 3,
}
_PERFORMANCE_DESTINATION = review_model.PERFORMANCE_DIFFERENCES_SHEET
_CAUSE_DESTINATION = review_model.PERFORMANCE_DIFFERENCE_CAUSES_SHEET
_DATA_ISSUES_DESTINATION = review_model.DATA_ISSUES_SHEET
_CAUSE_AREA_LABELS: Final[Mapping[CauseArea, str]] = {
    CauseArea.SECURITY_RETURN_OR_CONTRIBUTION: "Security returns or contributions",
    CauseArea.MARKET_VALUE_OR_HOLDING: "Market values or holdings",
    CauseArea.TRANSACTION_ACTIVITY: "Transaction activity",
    CauseArea.FX_RATE: "Foreign-exchange rates",
    CauseArea.PORTFOLIO_PERFORMANCE_INPUT: "Portfolio performance inputs",
    CauseArea.CLASSIFICATION_OR_REFERENCE: "Classification or reference data",
    CauseArea.UNEXPLAINED: "Unexplained",
}
_ISSUE_CATEGORY_LABELS: Final[Mapping[DataIssueCategory, str]] = {
    DataIssueCategory.CONTINUITY: "Data continuity",
    DataIssueCategory.DUPLICATE: "Duplicate rows",
    DataIssueCategory.PRICE: "Price consistency",
    DataIssueCategory.INCOME: "Income consistency",
    DataIssueCategory.ACCRUED_INTEREST: "Accrued-interest consistency",
    DataIssueCategory.POSITION_VALUE: "Position-value consistency",
    DataIssueCategory.CORPORATE_ACTION: "Corporate-action consistency",
}
_ISSUE_TYPE_LABELS: Final[Mapping[DataIssueType, str]] = {
    DataIssueType.DUPLICATE_TRANSACTIONS: "Duplicate transactions",
    DataIssueType.DIVIDEND_RATE: "Dividend-rate consistency",
    DataIssueType.HOLDINGS_ACCRUED_RATE: "Holdings accrued-interest consistency",
    DataIssueType.HOLDINGS_PRICE_RANGE: "Holdings price consistency",
    DataIssueType.MISSING_DIVIDEND: "Potential missing dividends",
    DataIssueType.PA_SA_RATE: "Purchase/sale accrued-interest consistency",
    DataIssueType.PORTFOLIO_MARKET_VALUE_CONTINUITY: (
        "Portfolio market-value continuity"
    ),
    DataIssueType.SECURITY_MARKET_VALUE_CONTINUITY: (
        "Security market-value continuity"
    ),
    DataIssueType.TRANSACTIONS_PRICE_RANGE: "Transaction price consistency",
}


@dataclass(frozen=True)
class ExecutiveSummaryContext:
    """Configuration context displayed in the Executive Summary.

    Attributes:
        comparison_level: Portfolio or security review level.
        snapshot_a_label: User-facing Snapshot A label.
        snapshot_b_label: User-facing Snapshot B label.
    """

    comparison_level: str
    snapshot_a_label: str
    snapshot_b_label: str


def executive_summary_context(
    comparison_path: util.PathLike | None,
    comparison_level: str,
) -> ExecutiveSummaryContext:
    """Return summary context from the already validated comparison inputs.

    Args:
        comparison_path: Optional Audit YAML path.
        comparison_level: Portfolio or security review level.

    Returns:
        Context containing the report level and available snapshot labels.
    """
    if comparison_path is None:
        return ExecutiveSummaryContext(
            comparison_level=comparison_level,
            snapshot_a_label="Snapshot A",
            snapshot_b_label="Snapshot B",
        )
    specification = AuditSpecification(
        comparison_path,
        comparison_level=comparison_level,
    )
    return ExecutiveSummaryContext(
        comparison_level=comparison_level,
        snapshot_a_label=specification.snapshot_a.label,
        snapshot_b_label=specification.snapshot_b.label,
    )


def executive_summary_table(
    primary_changes: pl.DataFrame,
    cause_summary: pl.DataFrame,
    data_issues: pl.DataFrame,
    impact_coverage: pl.DataFrame,
    *,
    context: ExecutiveSummaryContext,
) -> pl.DataFrame:
    """Return the bounded canonical Executive Summary table.

    Args:
        primary_changes: Reconciled Performance Differences table.
        cause_summary: Existing cause-area summary table.
        data_issues: Existing canonical Data Issues table.
        impact_coverage: Existing impact-coverage summary table.
        context: Report-level and snapshot-label context.

    Returns:
        Ordered summary rows shared by CSV, XLSX, and HTML.

    Raises:
        PpaError: If a supposedly canonical issue type or cause area is unknown.
    """
    changed_rows = _changed_primary_rows(primary_changes)
    rows = [
        _bottom_line_row(changed_rows, data_issues, context.comparison_level),
        *_performance_rows(changed_rows, impact_coverage, context.comparison_level),
        *_cause_rows(cause_summary, context.comparison_level),
        *_data_issue_rows(data_issues),
        *_next_step_rows(changed_rows, data_issues),
        *_context_rows(changed_rows, context),
    ]
    return pl.DataFrame(rows, schema={column: pl.String for column in EXECUTIVE_SUMMARY_COLUMNS})


def _row(
    section: str,
    item: str,
    result: object,
    detail: str,
    destination: str = "",
    review_key: str = "",
) -> dict[str, str]:
    """Return one canonical summary row."""
    return {
        SUMMARY_SECTION: section,
        SUMMARY_ITEM: item,
        SUMMARY_RESULT: rendering.format_value(result),
        SUMMARY_DETAIL: detail,
        REVIEW_DESTINATION: destination,
        REVIEW_KEY: review_key,
    }


def _bottom_line_row(
    changed_rows: list[dict[str, object]],
    data_issues: pl.DataFrame,
    comparison_level: str,
) -> dict[str, str]:
    """Return the plain-English first answer a reviewer should read."""
    statuses = Counter(str(row.get("review_status", "")) for row in changed_rows)
    unit = _review_unit_label(comparison_level, len(changed_rows))
    if not changed_rows:
        performance_message = "No reported performance changes were found."
    else:
        attention_phrases = _performance_attention_phrases(statuses)
        if attention_phrases:
            attention_count = sum(
                statuses[status]
                for status in (
                    "Unexplained",
                    "Partly Explained",
                    "Missing YAML Specifications",
                )
            )
            review_sentence = (
                "This requires review."
                if attention_count == 1
                else f"These {attention_count} require review."
            )
            performance_message = (
                f"{len(changed_rows)} {unit} changed; "
                f"{_joined_phrases(attention_phrases)}. {review_sentence}"
            )
        else:
            performance_message = (
                f"{len(changed_rows)} {unit} changed; all are fully explained by "
                "supported evidence."
            )
    continuity_count = _continuity_count(data_issues)
    if continuity_count:
        data_message = (
            f" Separately, {continuity_count:,} data-continuity rows require attention."
        )
    elif data_issues.height:
        data_message = f" Separately, {data_issues.height:,} data-quality rows require attention."
    else:
        data_message = " No separate data-quality rows require attention."
    return _row(
        "At a Glance",
        "Bottom line",
        performance_message + data_message,
        "Use the review steps below for the shortest path to supporting evidence.",
        _PERFORMANCE_DESTINATION if changed_rows else _DATA_ISSUES_DESTINATION,
    )


def _performance_attention_phrases(statuses: Counter[str]) -> list[str]:
    """Return plain-English status phrases that require reviewer attention."""
    phrases: list[str] = []
    for status, singular, plural in (
        ("Unexplained", "is unexplained", "are unexplained"),
        ("Partly Explained", "is only partly explained", "are only partly explained"),
        (
            "Missing YAML Specifications",
            "lacks explanation setup",
            "lack explanation setup",
        ),
    ):
        count = statuses[status]
        if count:
            phrases.append(f"{count} {singular if count == 1 else plural}")
    return phrases


def _joined_phrases(phrases: list[str]) -> str:
    """Join short phrases with a readable final conjunction."""
    if len(phrases) < 2:
        return phrases[0]
    return ", ".join(phrases[:-1]) + f" and {phrases[-1]}"


def _changed_primary_rows(table: pl.DataFrame) -> list[dict[str, object]]:
    """Return real changed review units, excluding honest empty-state rows."""
    return [
        row
        for row in table.iter_rows(named=True)
        if row.get("review_status") != "No differences"
        and row.get("performance_change") is not None
    ]


def _report_level_label(comparison_level: str) -> str:
    """Return the user-facing report-level label."""
    if comparison_level == PORTFOLIO_COMPARISON_LEVEL:
        return "Portfolio"
    if comparison_level == SECURITY_COMPARISON_LEVEL:
        return "Security"
    raise PpaError(f"Unsupported comparison level: {comparison_level!r}", None)


def _review_unit_label(comparison_level: str, count: int) -> str:
    """Return a readable singular or plural review-unit label."""
    base = (
        "portfolio period"
        if comparison_level == PORTFOLIO_COMPARISON_LEVEL
        else "security period"
    )
    return base if count == 1 else f"{base}s"


def _scope_result(rows: list[dict[str, object]], comparison_level: str) -> str:
    """Return a compact changed entity/period scope result."""
    unit = (
        "portfolio period"
        if comparison_level == PORTFOLIO_COMPARISON_LEVEL
        else "security period"
    )
    return f"{len(rows)} changed {unit}{'' if len(rows) == 1 else 's'}"


def _scope_detail(rows: list[dict[str, object]], comparison_level: str) -> str:
    """Return entity and period cardinality available from changed review units."""
    if not rows:
        return "No changed review units are present; detailed sheets retain the complete evidence."
    entity_columns = [findings.PORTFOLIO_ID]
    if comparison_level == SECURITY_COMPARISON_LEVEL:
        entity_columns.append(findings.SECURITY_ID)
    entities = {tuple(row.get(column) for column in entity_columns) for row in rows}
    periods = {
        (row.get(findings.FROM_DATE), row.get(findings.THRU_DATE)) for row in rows
    }
    entity_label = "entity" if len(entities) == 1 else "entities"
    period_label = "period" if len(periods) == 1 else "periods"
    return (
        f"{len(entities)} affected {entity_label} across {len(periods)} distinct "
        f"{period_label}."
    )


def _performance_rows(
    changed_rows: list[dict[str, object]],
    impact_coverage: pl.DataFrame,
    comparison_level: str,
) -> list[dict[str, str]]:
    """Return performance overview and bounded priority-unit rows."""
    statuses = Counter(str(row.get("review_status", "")) for row in changed_rows)
    count = len(changed_rows)
    rows = [
        _row(
            "Performance",
            "What changed?",
            f"{count} changed {_review_unit_label(comparison_level, count)}",
            (
                f"Fully explained: {statuses['Fully Explained']}; partly explained: "
                f"{statuses['Partly Explained']}; unexplained: "
                f"{statuses['Unexplained']}."
            ),
            _PERFORMANCE_DESTINATION,
        )
    ]
    limited_count = _method_limited_count(impact_coverage)
    if limited_count:
        coverage_verb = "has" if limited_count == 1 else "have"
        rows.append(
            _row(
                "Performance",
                "Can every supporting cause be quantified?",
                (
                    f"No — {limited_count} changed "
                    f"{_review_unit_label(comparison_level, limited_count)} "
                    f"{coverage_verb} incomplete estimates"
                ),
                (
                    "Some supporting causes have incomplete estimates, evidence only, "
                    "or missing inputs."
                ),
                _CAUSE_DESTINATION,
            )
        )
    if not changed_rows:
        return rows
    for index, row in enumerate(_priority_rows(changed_rows), start=1):
        rows.append(
            _row(
                "Performance",
                "Review next" if index == 1 else f"Then review #{index}",
                _primary_identity(row, comparison_level),
                (
                    f"{row.get('review_status')}. Reported return change: "
                    f"{_percentage_points(row.get('performance_change'))}; explained: "
                    f"{_percentage_points(row.get('estimated_cause_total'))}; unexplained: "
                    f"{_percentage_points(row.get('unexplained_change'))}."
                ),
                _PERFORMANCE_DESTINATION,
                str(row.get(REVIEW_KEY, "")),
            )
        )
    return rows


def _percentage_points(value: object) -> str:
    """Return a return difference as signed percentage points for display."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value) * 100:+.6f} percentage points"
    return "not available"


def _method_limited_count(coverage: pl.DataFrame) -> int:
    """Return the existing count of non-complete impact-coverage units."""
    if explain.IMPACT_COVERAGE_STATUS not in coverage.columns:
        return 0
    return sum(
        row.get(explain.IMPACT_COVERAGE_STATUS)
        != explain.IMPACT_COVERAGE_STATUS_COMPLETE_ESTIMATES
        for row in coverage.iter_rows(named=True)
    )


def _priority_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return at most ten changed units in deterministic review order."""
    review_needed_rows = [
        row for row in rows if row.get("review_status") != "Fully Explained"
    ]
    return sorted(
        review_needed_rows,
        key=lambda row: (
            _STATUS_ORDER.get(str(row.get("review_status")), len(_STATUS_ORDER)),
            -_absolute_numeric(row.get("unexplained_change")),
            -_absolute_numeric(row.get("performance_change")),
            str(row.get(REVIEW_KEY, "")),
        ),
    )[:PRIORITY_REVIEW_UNIT_LIMIT]


def _absolute_numeric(value: object) -> float:
    """Return an absolute numeric value for deterministic priority sorting."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return abs(float(value))
    return 0.0


def _primary_identity(row: Mapping[str, object], comparison_level: str) -> str:
    """Return a stable, readable review-unit identity."""
    values = [
        row.get(findings.PORTFOLIO_ID),
        row.get(findings.FROM_DATE),
        row.get(findings.THRU_DATE),
    ]
    if comparison_level == SECURITY_COMPARISON_LEVEL:
        values.insert(1, row.get(findings.SECURITY_ID))
    return " | ".join(rendering.format_value(value) for value in values)


def _cause_rows(
    cause_summary: pl.DataFrame,
    comparison_level: str,
) -> list[dict[str, str]]:
    """Return stable cause-area affected-unit counts."""
    affected: dict[CauseArea, set[tuple[object, ...]]] = defaultdict(set)
    for row in cause_summary.iter_rows(named=True):
        try:
            area = CauseArea(str(row.get(explain.ROOT_CAUSE_AREA)))
        except ValueError as error:
            raise PpaError(
                f"Executive Summary received unknown cause area: "
                f"{row.get(explain.ROOT_CAUSE_AREA)!r}",
                None,
            ) from error
        key = [
            row.get(findings.PORTFOLIO_ID),
            row.get(findings.FROM_DATE),
            row.get(findings.THRU_DATE),
        ]
        if comparison_level == SECURITY_COMPARISON_LEVEL:
            key.insert(1, row.get(findings.SECURITY_ID))
        affected[area].add(tuple(key))
    if not affected:
        return [
            _row(
                "Explanation",
                "What supports the explanation?",
                "No supported cause is available",
                "Review the detailed evidence for unresolved changes.",
                _CAUSE_DESTINATION,
            )
        ]
    return [
        _row(
            "Explanation",
            "What supports the explanation?",
            _CAUSE_AREA_LABELS[area],
            (
                f"This evidence affects {len(affected[area])} "
                f"{_review_unit_label(comparison_level, len(affected[area]))}."
            ),
            _CAUSE_DESTINATION,
        )
        for area in CauseArea
        if area in affected
    ]


def _data_issue_rows(data_issues: pl.DataFrame) -> list[dict[str, str]]:
    """Return concise, plain-English Data Issues attention rows."""
    type_counts: Counter[DataIssueType] = Counter()
    category_counts: Counter[DataIssueCategory] = Counter()
    for row in data_issues.iter_rows(named=True):
        raw_type = str(row.get(data_issue_checks.ISSUE_TYPE))
        try:
            issue_type = DataIssueType(raw_type)
        except ValueError as error:
            raise PpaError(
                f"Executive Summary received unknown Data Issues issue type: {raw_type!r}",
                None,
            ) from error
        definition = DATA_ISSUE_REGISTRY[issue_type]
        type_counts[issue_type] += 1
        category_counts[definition.category] += 1
    if data_issues.is_empty():
        return [
            _row(
                "Data Quality",
                "Does other source-data need attention?",
                "No",
                "No separate Data Issues rows were found.",
                _DATA_ISSUES_DESTINATION,
            )
        ]
    rows = [
        _row(
            "Data Quality",
            "Does other source-data need attention?",
            f"Yes — {data_issues.height:,} rows",
            (
                "These are source-data checks, separate from the performance "
                "explanation; they are not validated incident counts."
            ),
            _DATA_ISSUES_DESTINATION,
        )
    ]
    rows.extend(
        _row(
            "Data Quality",
            (
                "Review continuity first"
                if category == DataIssueCategory.CONTINUITY
                else "Also review"
            ),
            f"{_ISSUE_CATEGORY_LABELS[category]} — {category_counts[category]:,} rows",
            _issue_type_breakdown(category, type_counts),
            _DATA_ISSUES_DESTINATION,
        )
        for category in DataIssueCategory
        if category_counts[category]
    )
    return rows


def _issue_type_breakdown(
    category: DataIssueCategory,
    type_counts: Counter[DataIssueType],
) -> str:
    """Return readable issue-type counts within one stable category."""
    parts = [
        f"{_ISSUE_TYPE_LABELS[issue_type]}: {type_counts[issue_type]:,}"
        for issue_type in DataIssueType
        if type_counts[issue_type]
        and DATA_ISSUE_REGISTRY[issue_type].category == category
    ]
    return "; ".join(parts) + "."


def _continuity_count(data_issues: pl.DataFrame) -> int:
    """Return mandatory continuity rows while failing closed on unknown types."""
    count = 0
    for row in data_issues.iter_rows(named=True):
        raw_type = str(row.get(data_issue_checks.ISSUE_TYPE))
        try:
            issue_type = DataIssueType(raw_type)
        except ValueError as error:
            raise PpaError(
                f"Executive Summary received unknown Data Issues issue type: {raw_type!r}",
                None,
            ) from error
        if DATA_ISSUE_REGISTRY[issue_type].category == DataIssueCategory.CONTINUITY:
            count += 1
    return count


def _next_step_rows(
    changed_rows: list[dict[str, object]],
    data_issues: pl.DataFrame,
) -> list[dict[str, str]]:
    """Return deterministic next-review cues without combining attention types."""
    statuses = Counter(str(row.get("review_status", "")) for row in changed_rows)
    performance_attention = statuses["Partly Explained"] + statuses["Unexplained"]
    continuity_count = 0
    other_issue_count = 0
    for row in data_issues.iter_rows(named=True):
        try:
            issue_type = DataIssueType(str(row.get(data_issue_checks.ISSUE_TYPE)))
        except ValueError:
            continue
        if DATA_ISSUE_REGISTRY[issue_type].category == DataIssueCategory.CONTINUITY:
            continuity_count += 1
        else:
            other_issue_count += 1
    rows: list[dict[str, str]] = []
    if performance_attention:
        rows.append(
            _row(
                "Next Steps",
                "What should I do first?",
                "Review the unexplained performance change",
                (
                    "Open Performance Differences, then follow its supporting "
                    "cause and source-evidence links."
                ),
                _PERFORMANCE_DESTINATION,
            )
        )
    if data_issues.height:
        issue_result = (
            f"Review {continuity_count:,} continuity rows"
            if continuity_count
            else f"Review {other_issue_count:,} data-quality rows"
        )
        detail = (
            f"Then review {other_issue_count:,} other data-quality rows."
            if continuity_count and other_issue_count
            else "Open Data Issues for the affected source rows and exact values."
        )
        rows.append(
            _row(
                "Next Steps",
                "What else needs attention?",
                issue_result,
                detail,
                _DATA_ISSUES_DESTINATION,
            )
        )
    if not rows:
        rows.append(
            _row(
                "Next Steps",
                "What should I do next?",
                "No immediate exception review",
                "Retain the report bundle as evidence of the comparison.",
            )
        )
    return rows


def _context_rows(
    changed_rows: list[dict[str, object]],
    context: ExecutiveSummaryContext,
) -> list[dict[str, str]]:
    """Return concise comparison context after the decision-useful content."""
    return [
        _row(
            "Context",
            "What was compared?",
            f"{context.snapshot_a_label} to {context.snapshot_b_label}",
            (
                f"{_report_level_label(context.comparison_level)} report; "
                f"{_scope_result(changed_rows, context.comparison_level)}. "
                f"{_scope_detail(changed_rows, context.comparison_level)}"
            ),
            _PERFORMANCE_DESTINATION,
        ),
        _row(
            "Context",
            "How were supported effects reviewed?",
            "Modified Dietz",
            "The summary reuses existing validated results; it does not recalculate returns.",
            _CAUSE_DESTINATION,
        ),
        _row(
            "Context",
            "What does this report not prove?",
            "It does not certify official performance",
            (
                "PPAR explains supported changes but does not determine which "
                "snapshot is correct."
            ),
        ),
    ]
