"""Guard Audit output against unusably large review tables."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from ppar.errors import PpaError
from ppar.audit import review_model as _pc_review_model
from ppar.audit import schema as pc_cols
from ppar.audit import workbook as _pc_workbook


_MAX_PRIMARY_REVIEW_ROWS = 100_000
_ROW_LIMIT_ARTIFACTS = {
    _pc_review_model.PERFORMANCE_DIFFERENCES_ARTIFACT,
    _pc_review_model.PERFORMANCE_DIFFERENCE_CAUSES_ARTIFACT,
    _pc_review_model.DATA_ISSUES_ARTIFACT,
}


def _top_review_row_contributors(table: pl.DataFrame) -> str:
    """Return compact dimensions explaining an oversized review table."""
    dimensions = (
        ("portfolios", (pc_cols.PORTFOLIO_ID,)),
        ("periods", (pc_cols.FROM_DATE, pc_cols.THRU_DATE)),
        ("as-of dates", ("as_of_date",)),
        ("dataset.fields", ("dataset_field",)),
    )
    summaries: list[str] = []
    for label, columns in dimensions:
        if not all(column in table.columns for column in columns):
            continue
        grouped = (
            table.group_by(columns)
            .len(name="_rows")
            .sort(
                ["_rows", *columns],
                descending=[True, *([False] * len(columns))],
                nulls_last=True,
            )
            .head(3)
        )
        values = []
        for row in grouped.iter_rows(named=True):
            key = " to ".join(
                "<blank>" if row[column] is None else str(row[column])
                for column in columns
            )
            values.append(f"{key} ({int(row['_rows']):,})")
        if values:
            summaries.append(f"{label}: {', '.join(values)}")
    return "; ".join(summaries)


def assert_review_output_row_limit(
    sheets: Sequence[_pc_workbook.ReviewWorkbookSheet],
    *,
    comparison_level: str,
) -> None:
    """Stop an oversized report before writing any bundle artifacts.

    Args:
        sheets: Canonical review sheets shared by HTML, XLSX, and CSV output.
        comparison_level: Portfolio or security report level used in the error.

    Raises:
        PpaError: If a primary reviewer-facing table exceeds 100,000 rows.
    """
    oversized = [
        sheet
        for sheet in sheets
        if sheet.artifact_name in _ROW_LIMIT_ARTIFACTS
        and sheet.table.height > _MAX_PRIMARY_REVIEW_ROWS
    ]
    if not oversized:
        return

    table_messages = []
    for sheet in oversized:
        contributors = _top_review_row_contributors(sheet.table)
        contributor_message = (
            f" Largest contributors: {contributors}." if contributors else ""
        )
        table_messages.append(
            f'{sheet.sheet_name} would contain {sheet.table.height:,} rows '
            f"(limit {_MAX_PRIMARY_REVIEW_ROWS:,}).{contributor_message}"
        )
    raise PpaError(
        f"Audit output row limit exceeded for the {comparison_level} report. "
        "No files were written for the oversized report. "
        + " ".join(table_messages)
        + " Reduce the portfolio or date scope, or correct the upstream differences, "
        "then rerun Audit.",
        None,
    )
