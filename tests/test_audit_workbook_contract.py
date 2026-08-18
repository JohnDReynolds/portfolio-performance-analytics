"""Contract tests for the performance-comparison review workbook."""

from __future__ import annotations

# Python imports
import datetime as dt
import importlib
import json
from pathlib import Path
import tempfile
from typing import Any
import unittest

# Third-party imports
import polars as pl

# Test imports
from tests import test_utilities as test_util

# Project imports
from ppar.errors import PpaError
from ppar.audit import (
    compare_snapshots,
    write_audit_report_bundle as _write_audit_report_bundle,
)
from ppar.audit import review_model as _pc_review_model
from ppar.audit import workbook_tables as _pc_workbook_tables

_PORTFOLIO_COMPARISON_PATH = Path(
    "ppar/setup_templates/axys_apx_audit/axys_apx_audit.yaml"
)

_EXPECTED_PORTFOLIO_SHEETS = [
    _pc_review_model.EXECUTIVE_SUMMARY_SHEET,
    _pc_review_model.PERFORMANCE_DIFFERENCES_SHEET,
    _pc_review_model.PERFORMANCE_DIFFERENCE_CAUSES_SHEET,
    _pc_review_model.DATA_ISSUES_SHEET,
]
_EXPECTED_SECURITY_SHEETS = list(_EXPECTED_PORTFOLIO_SHEETS)
_EXPECTED_DIAGNOSTIC_SHEETS = [
    _pc_review_model.EXECUTIVE_SUMMARY_SHEET,
    _pc_review_model.PERFORMANCE_DIFFERENCES_SHEET,
    _pc_review_model.PERFORMANCE_DIFFERENCE_CAUSES_SHEET,
    _pc_review_model.DATA_ISSUES_SHEET,
    _pc_review_model.RECONSTRUCTION_SUMMARY_SHEET,
    _pc_review_model.RETURN_RECONSTRUCTION_CHECKS_SHEET,
    _pc_review_model.SECURITY_RETURN_RECONSTRUCTION_CHECKS_SHEET,
]
_COMMON_LEFT_HEADERS = [
    "Portfolio",
    "From Date",
    "Thru Date",
    "As Of Date",
    "Dataset.Field",
    "Security",
]
_IDENTIFIABLE_LEFT_HEADERS = [
    "Portfolio",
    "From Date",
    "Thru Date",
    "As Of Date",
    "Dataset.Field",
    "Security",
]
_EXPECTED_NON_FULLY_EXPLAINED_PORTFOLIO_ROWS = {
    ("BALANCED", "2026-05-09", "2026-05-14", "Partly Explained"),
    ("INCOME", "2026-04-01", "2026-04-30", "Unexplained"),
}


def write_audit_report_bundle(*args: Any, **kwargs: Any) -> dict[str, Path]:
    """Write a bundle and extract support for workbook contract assertions."""
    paths = _write_audit_report_bundle(*args, **kwargs)
    output_directory = Path(args[1] if len(args) > 1 else kwargs["output_directory"])
    return test_util.extract_audit_support(paths, output_directory)


def _header_values(worksheet: Any) -> list[object]:
    """Return worksheet header values."""
    return [_normalized_header(cell.value) for cell in worksheet[1]]


def _raw_header_values(worksheet: Any) -> list[object]:
    """Return worksheet header values without normalizing display line breaks."""
    return [cell.value for cell in worksheet[1]]


def _header_comment(worksheet: Any, header: str) -> str:
    """Return the comment text for a named worksheet header."""
    for cell in worksheet[1]:
        if _normalized_header(cell.value) == header:
            if cell.comment is None:
                raise AssertionError(f"Header {header!r} has no comment.")
            return str(cell.comment.text)
    raise AssertionError(f"Header {header!r} not found.")


def _normalized_header(value: object) -> object:
    """Return an Excel header with intentional line breaks normalized."""
    if isinstance(value, str):
        return " ".join(value.split())
    return value


def _column_values(worksheet: Any, column: str) -> list[object]:
    """Return nonblank values from a worksheet column below the header row."""
    return [
        worksheet[f"{column}{row}"].value
        for row in range(2, worksheet.max_row + 1)
        if worksheet[f"{column}{row}"].value is not None
    ]


def _sheet_rows(worksheet: Any) -> list[tuple[object, ...]]:
    """Return worksheet data rows."""
    return list(worksheet.iter_rows(min_row=2, values_only=True))


def _numeric_value(value: object) -> float:
    """Return an XLSX cell value as a float after asserting it is numeric."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssertionError(f"Expected numeric workbook value, got {value!r}.")
    return float(value)


def _workbook_date_text(value: object) -> str:
    """Return an ISO date string from an XLSX cell value."""
    return str(value)[:10]


def _assert_print_layout(test_case: unittest.TestCase, workbook: Any) -> None:
    """Assert every reviewer sheet has bounded, one-page-wide print metadata."""
    for worksheet in workbook.worksheets:
        with test_case.subTest(sheet=worksheet.title):
            test_case.assertEqual(worksheet.page_setup.orientation, "landscape")
            test_case.assertEqual(worksheet.page_setup.fitToWidth, 1)
            test_case.assertTrue(worksheet.sheet_properties.pageSetUpPr.fitToPage)
            test_case.assertTrue(worksheet.print_area)
            if worksheet.title == _pc_review_model.EXECUTIVE_SUMMARY_SHEET:
                test_case.assertEqual(worksheet.page_setup.fitToHeight, 1)
            else:
                test_case.assertEqual(worksheet.page_setup.fitToHeight, 0)
                test_case.assertEqual(worksheet.print_title_rows, "$1:$1")


class TestAuditWorkbookContract(unittest.TestCase):
    """Validate reviewer-facing workbook presentation invariants."""

    def test_audit_file_names_are_level_specific(self) -> None:
        """Audit artifact names identify their portfolio or security level."""
        self.assertEqual(
            _pc_review_model.html_report_file_name("portfolio"),
            "portfolio_audit.html",
        )
        self.assertEqual(
            _pc_review_model.review_workbook_file_name("portfolio"),
            "portfolio_audit.xlsx",
        )
        self.assertEqual(
            _pc_review_model.html_report_file_name("security"),
            "security_audit.html",
        )
        self.assertEqual(
            _pc_review_model.review_workbook_file_name("security"),
            "security_audit.xlsx",
        )
        with self.assertRaisesRegex(ValueError, "Unsupported comparison level"):
            _pc_review_model.audit_file_stem("account")

    def test_portfolio_explanation_invariant_rejects_arithmetic_mismatch(self) -> None:
        """Cause impacts must equal the portfolio-period explained difference."""
        primary = pl.DataFrame(
            [
                {
                    "portfolio_id": "TEST",
                    "from_date": dt.date(2026, 1, 1),
                    "thru_date": dt.date(2026, 1, 31),
                    "performance_change": 0.01,
                    "estimated_cause_total": 0.01,
                    "review_status": "Fully Explained",
                }
            ]
        )
        causes = pl.DataFrame(
            [
                {
                    "portfolio_id": "TEST",
                    "from_date": dt.date(2026, 1, 1),
                    "thru_date": dt.date(2026, 1, 31),
                    "estimated_impact": 0.009,
                }
            ]
        )

        with self.assertRaisesRegex(PpaError, "causes total"):
            # pylint: disable=protected-access
            _pc_workbook_tables._assert_portfolio_explanation_invariants(
                primary,
                causes,
                (),
            )

    def test_fully_explained_rounding_boundary_reconciles_visible_values(self) -> None:
        """Sub-precision residuals cannot make visible Fully Explained cells differ."""
        period = {
            "portfolio_id": "TEST",
            "from_date": dt.date(2036, 2, 28),
            "thru_date": dt.date(2036, 3, 31),
        }
        primary = pl.DataFrame(
            [
                {
                    **period,
                    "performance_change": 0.0015805693,
                    "estimated_cause_total": 0.00158047626573,
                    "review_status": "Fully Explained",
                }
            ]
        )
        causes = pl.DataFrame(
            [{**period, "estimated_impact": 0.00158047626573}]
        )

        # pylint: disable=protected-access
        _pc_workbook_tables._assert_portfolio_explanation_invariants(
            primary,
            causes,
            (),
        )
        displayed_causes = (
            _pc_workbook_tables._workbook_reconcile_displayed_explained_values(
                primary,
                causes,
            )
        )
        displayed_primary = (
            _pc_workbook_tables._workbook_reconcile_displayed_primary_values(primary)
        )
        _pc_workbook_tables._assert_displayed_portfolio_explanation_reconciliation(
            displayed_primary,
            displayed_causes,
        )

        self.assertEqual(
            displayed_primary["estimated_cause_total"].item(),
            0.001581,
        )
        self.assertEqual(displayed_causes["estimated_impact"].item(), 0.001581)

    def test_portfolio_explanation_invariant_rejects_missing_dietz_component(
        self,
    ) -> None:
        """A beginning-value effect cannot vanish even when totals still foot."""
        period = {
            "portfolio_id": "TEST",
            "from_date": dt.date(2026, 1, 1),
            "thru_date": dt.date(2026, 1, 31),
        }
        primary = pl.DataFrame(
            [
                {
                    **period,
                    "performance_change": 0.01,
                    "estimated_cause_total": 0.01,
                    "review_status": "Fully Explained",
                }
            ]
        )
        causes = pl.DataFrame([{**period, "estimated_impact": 0.01}])
        formula_rows = [
            {
                **period,
                "source_column": "beginning_market_value",
                "estimated_impact": 0.01,
            }
        ]

        with self.assertRaisesRegex(PpaError, "beginning_market_value"):
            # pylint: disable=protected-access
            _pc_workbook_tables._assert_portfolio_explanation_invariants(
                primary,
                causes,
                formula_rows,
            )

    def test_visible_explanation_invariant_rejects_missing_source_facts(self) -> None:
        """Final cause rows retain their field, direction, and transaction code."""
        causes = pl.DataFrame(
            [
                {
                    "portfolio_id": "TEST",
                    "from_date": dt.date(2026, 1, 1),
                    "thru_date": dt.date(2026, 1, 31),
                    "dataset": "transactions",
                    "dataset_field": "transactions.amount",
                    "transaction_code": "dv",
                    "change": 10.0,
                    "review_guidance": "Cash balance increased by 10.00.",
                }
            ]
        )

        with self.assertRaisesRegex(PpaError, "visible explanation invariant"):
            # pylint: disable=protected-access
            _pc_workbook_tables._assert_visible_explanation_contract(
                causes,
                comparison_level="portfolio",
            )

    def test_review_workbook_contract_remains_reviewer_oriented(self) -> None:
        """Generated workbook uses stable, action-oriented sheets and columns."""
        openpyxl: Any = importlib.import_module("openpyxl")

        findings = compare_snapshots(
            _PORTFOLIO_COMPARISON_PATH,
            comparison_level="portfolio",
        )
        # Transaction metadata remains internal, but it must still make every
        # transaction-associated reviewer explanation self-identifying.
        cause_table = _pc_workbook_tables._workbook_underlying_causes_table(
            findings,
            comparison_path=_PORTFOLIO_COMPARISON_PATH,
        )
        transaction_rows = cause_table.filter(
            pl.col("dataset") == "transactions"
        ).select("transaction_code", "review_guidance")
        self.assertFalse(transaction_rows.is_empty())
        self.assertTrue(
            all(
                code not in (None, "")
                for code in transaction_rows["transaction_code"]
            )
        )
        transaction_codes = set(transaction_rows["transaction_code"])
        self.assertTrue({"ai", "ti"}.issubset(transaction_codes))
        self.assertTrue(
            all(
                str(row["review_guidance"]).startswith(
                    f"{row['transaction_code']}:"
                )
                for row in transaction_rows.iter_rows(named=True)
            )
        )
        balanced_msft_rows = cause_table.filter(
            (pl.col("portfolio_id") == "BALANCED")
            & (pl.col("from_date") == dt.date(2026, 5, 9))
            & (pl.col("thru_date") == dt.date(2026, 5, 14))
            & (pl.col("security_id") == "csusMSFT")
        )
        self.assertIn(
            "holdings.market_value",
            balanced_msft_rows["dataset_field"].to_list(),
        )
        self.assertNotIn(
            "holdings.base_market_value",
            balanced_msft_rows["dataset_field"].to_list(),
        )
        foreign_base_rows = cause_table.filter(
            (pl.col("dataset_field") == "holdings.base_market_value")
            & (pl.col("security_id") == "csgbSHEL.L")
        )
        self.assertFalse(foreign_base_rows.is_empty())
        self.assertTrue(
            all(
                value is not None
                for value in foreign_base_rows["estimated_impact"].to_list()
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = write_audit_report_bundle(
                findings,
                Path(directory) / "bundle",
                include_workbook=True,
                comparison_path=_PORTFOLIO_COMPARISON_PATH,
            )
            self.assertNotIn(_pc_review_model.RECONSTRUCTION_SUMMARY_ARTIFACT, paths)
            self.assertNotIn(
                _pc_review_model.RETURN_RECONSTRUCTION_CHECKS_ARTIFACT,
                paths,
            )
            self.assertNotIn(
                _pc_review_model.SECURITY_RETURN_RECONSTRUCTION_CHECKS_ARTIFACT,
                paths,
            )

            readme = paths["readme"].read_text(encoding="utf-8")
            self.assertIn("source_detail.csv", readme)
            self.assertNotIn("## Primary Review Artifact", readme)
            self.assertNotIn("Open `portfolio_audit.xlsx` first", readme)
            self.assertNotIn("same review model in a browser", readme)
            html_report = paths["html_report"].read_text(encoding="utf-8")
            self.assertIn("explained causes, supporting evidence", readme)
            self.assertIn("Dataset.Field", html_report)
            self.assertNotIn(">Review Key</th>", html_report)
            self.assertNotIn("Source Dataset", html_report)
            self.assertNotIn("Source-Data Dataset", html_report)
            self.assertIn(
                "Changed input field, shown as dataset.field.",
                html_report,
            )
            self.assertIn(
                'title="Count of reported performance differences accounted for '
                "by supported, quantified causes within the configured tolerance.\"",
                html_report,
            )
            self.assertIn(
                'title="The reported performance difference is accounted for by '
                "supported, quantified causes within the configured tolerance.\">"
                "Fully Explained</td>",
                html_report,
            )
            self.assertNotIn("Normalized source dataset", html_report)
            self.assertNotIn(
                "Browser view for reviewing this performance-comparison bundle.",
                html_report,
            )
            self.assertNotIn('class="pc-contents-list"', html_report)
            self.assertNotIn("same review model", html_report)
            self.assertNotIn("Browser review surface", html_report)
            self.assertNotIn("Transaction Match Diagnostics", html_report)
            self.assertNotIn("Match Confidence", html_report)
            self.assertNotIn(">Transaction Code</th>", html_report)
            self.assertNotIn(">Transaction Category</th>", html_report)
            self.assertIn(">ai:", html_report)
            self.assertIn(">ti:", html_report)
            self.assertNotIn("pc-value-explained-cause", html_report)
            self.assertNotIn("pc-value-explained-impact", html_report)
            self.assertIn("pc-fill-explained-cause", html_report)
            self.assertIn("pc-fill-possible-cause", html_report)
            self.assertIn("pc-fill-review-needed", html_report)

            source_detail = pl.read_csv(
                paths["source_detail"],
                infer_schema_length=None,
            )
            retained_msft_base_value = source_detail.filter(
                (pl.col("portfolio_id") == "BALANCED")
                & (pl.col("from_date") == "2026-05-09")
                & (pl.col("thru_date") == "2026-05-14")
                & (pl.col("dataset_field") == "holdings.base_market_value")
                & (pl.col("security_id") == "csusMSFT")
            )
            self.assertFalse(retained_msft_base_value.is_empty())

            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            review_summary = json.loads(paths["review_summary"].read_text(encoding="utf-8"))
            for summary in (manifest, review_summary):
                transaction_semantics = summary["transaction_semantics"]
                self.assertIn("by", transaction_semantics["observed_codes"])
                self.assertIn("sl", transaction_semantics["observed_codes"])
                self.assertNotIn("BY", transaction_semantics["observed_codes"])
                self.assertNotIn("SL", transaction_semantics["observed_codes"])

            workbook = openpyxl.load_workbook(
                paths["review_workbook"],
                data_only=True,
            )
            try:
                _assert_print_layout(self, workbook)
                self.assertEqual(workbook.sheetnames, _EXPECTED_PORTFOLIO_SHEETS)
                self.assertEqual(
                    _header_values(workbook["Performance Differences"]),
                    [
                        "Portfolio",
                        "From Date",
                        "Thru Date",
                        "Performance Difference",
                        "Explained Difference",
                        "Unexplained Difference",
                        "Status",
                        "Comments",
                    ],
                )
                self.assertEqual(
                    _header_values(workbook["Performance Difference Causes"]),
                    [
                        *_IDENTIFIABLE_LEFT_HEADERS,
                        "Snapshot A Value",
                        "Snapshot B Value",
                        "B - A Difference",
                        "Performance Difference Explained",
                        "Explanation",
                    ],
                )
                self.assertNotIn(
                    "Row Type",
                    _header_values(workbook["Performance Difference Causes"]),
                )
                self.assertIn(
                    "Performance\nDifference\nExplained",
                    _raw_header_values(workbook["Performance Difference Causes"]),
                )
                self.assertNotIn("Source Detail", workbook.sheetnames)
                self.assertEqual(
                    _header_values(workbook["Data Issues"]),
                    [
                        "Snapshot",
                        "Portfolio",
                        "As Of Date",
                        "Dataset.Field",
                        "Security",
                        "Issue Type",
                        "Category",
                        "Reference Value",
                        "Observed Value",
                        "Difference",
                        "Tolerance",
                        "Explanation",
                    ],
                )

                portfolio_differences = _column_values(
                    workbook["Performance Differences"],
                    "D",
                )
                portfolio_rows = _sheet_rows(workbook["Performance Differences"])
                portfolio_codes = {row[0] for row in portfolio_rows}
                statuses = {row[6] for row in portfolio_rows}
                self.assertTrue(portfolio_differences)
                self.assertEqual(
                    portfolio_codes,
                    {
                        "ALPHA",
                        "BALANCED",
                        "BALANCED_CONTRIBUTION",
                        "INCOME",
                    },
                )
                self.assertEqual(
                    statuses,
                    {"Fully Explained", "Partly Explained", "Unexplained"},
                )
                partly_explained_rows = [
                    row for row in portfolio_rows if row[6] == "Partly Explained"
                ]
                unexplained_rows = [row for row in portfolio_rows if row[6] == "Unexplained"]
                self.assertEqual(
                    {
                        (
                            row[0],
                            _workbook_date_text(row[1]),
                            _workbook_date_text(row[2]),
                        )
                        for row in partly_explained_rows
                    },
                    {("BALANCED", "2026-05-09", "2026-05-14")},
                )
                self.assertEqual(
                    {
                        (
                            row[0],
                            _workbook_date_text(row[1]),
                            _workbook_date_text(row[2]),
                        )
                        for row in unexplained_rows
                    },
                    {("INCOME", "2026-04-01", "2026-04-30")},
                )
                for row in portfolio_rows:
                    if row[6] == "Fully Explained":
                        self.assertAlmostEqual(
                            _numeric_value(row[3]),
                            _numeric_value(row[4]),
                            places=6,
                        )
                        self.assertIsNone(row[5])
                    else:
                        self.assertIsNotNone(row[5])
                balanced_may = partly_explained_rows[0]
                balanced_may_review_note = str(balanced_may[7])
                self.assertIn(
                    "Possible cause: csusJPM transactions.amount increased "
                    "by 200.00 on 2026-05-12. Add YAML configuration to count "
                    "it as explained.",
                    balanced_may_review_note,
                )
                self.assertNotIn(
                    "portfolio_performance.gain_loss",
                    balanced_may_review_note,
                )
                self.assertNotIn("holdings.cost", balanced_may_review_note)
                income_april = unexplained_rows[0]
                income_april_review_note = str(income_april[7])
                self.assertIn(
                    "The Unexplained Difference may be due to missing source-data, "
                    "source-file timing differences, or vendor methodology that "
                    "does not match the YAML specifications.",
                    income_april_review_note,
                )
                self.assertNotIn("holdings.cost", income_april_review_note)
                self.assertNotIn(
                    "Review `source_detail.csv`",
                    income_april_review_note,
                )
                self.assertTrue(
                    all(
                        isinstance(value, (int, float)) and not isinstance(value, bool)
                        for value in portfolio_differences
                    )
                )
                self.assertEqual(
                    workbook["Performance Differences"]["D2"].number_format,
                    "0.000000",
                )
                underlying_rows = _sheet_rows(workbook["Performance Difference Causes"])
                cause_totals: dict[tuple[object, object, object], float] = {}
                for cause_row in underlying_rows:
                    cause_key = cause_row[0], cause_row[1], cause_row[2]
                    explained_value = cause_row[9]
                    if explained_value is None:
                        continue
                    cause_totals[cause_key] = cause_totals.get(
                        cause_key,
                        0.0,
                    ) + _numeric_value(explained_value)
                for portfolio_row in portfolio_rows:
                    if portfolio_row[6] != "Fully Explained":
                        continue
                    period_key = portfolio_row[0], portfolio_row[1], portfolio_row[2]
                    self.assertEqual(
                        round(cause_totals.get(period_key, 0.0), 6),
                        round(_numeric_value(portfolio_row[4]), 6),
                    )
                alpha_february_rows = [
                    row
                    for row in underlying_rows
                    if row[0] == "ALPHA"
                    and str(row[1])[:10] == "2026-01-31"
                    and str(row[2])[:10] == "2026-02-27"
                ]
                self.assertTrue(
                    any(
                        row[4] == "holdings.market_value"
                        and row[5] == "causCASHUSD"
                        and row[10]
                        == ("causCASHUSD ending holdings.market_value decreased by " "2,008.00.")
                        for row in alpha_february_rows
                    )
                )
                self.assertFalse(
                    any(row[4] == "no_underlying_causes_found" for row in alpha_february_rows)
                )
                self.assertTrue(
                    all(
                        row[10] in (None, "")
                        or "No additive underlying cause" in str(row[10])
                        or "No identifiable cause" in str(row[10])
                        or "shown for review" in str(row[10])
                        or ('"Performance Differences"."Explained Difference"' in str(row[10]))
                        or "Input for changed" in str(row[10])
                        or "related performance input" in str(row[10])
                        or "changed transactions.amount" in str(row[10])
                        or "transactions.base_amount" in str(row[10])
                        or "changed holdings.market_value" in str(row[10])
                        or "calculated portfolio-return difference" in str(row[10])
                        or "Configured transaction impact method is present" in str(row[10])
                        or "Modified Dietz" in str(row[10])
                        or "Supporting detail for changed holdings value" in str(row[10])
                        or "External flow" in str(row[10])
                        or "Helped explain" in str(row[10])
                        or "Caused cash-balance" in str(row[10])
                        or "performance calculation through" in str(row[10])
                        or "split factor" in str(row[10])
                        or "Add YAML configuration to count it as explained" in str(row[10])
                        or "ending holdings." in str(row[10])
                        or "beginning holdings." in str(row[10])
                        or "Review-only evidence" in str(row[10])
                        for row in underlying_rows
                    )
                )
                underlying_fields = {row[4] for row in underlying_rows}
                self.assertTrue(
                    {
                        "holdings.market_value",
                        "holdings.quantity",
                        "transactions.amount",
                        "transactions.commission",
                        "transactions.price",
                        "transactions.quantity",
                    }.issubset(underlying_fields)
                )
                self.assertNotIn(
                    "Row Type",
                    _header_values(workbook["Performance Difference Causes"]),
                )
                balanced_msft_cause_fields = {
                    row[4]
                    for row in underlying_rows
                    if row[0] == "BALANCED"
                    and _workbook_date_text(row[1]) == "2026-05-09"
                    and _workbook_date_text(row[2]) == "2026-05-14"
                    and row[5] == "csusMSFT"
                }
                self.assertIn("holdings.market_value", balanced_msft_cause_fields)
                self.assertNotIn(
                    "holdings.base_market_value",
                    balanced_msft_cause_fields,
                )
                self.assertTrue(
                    any(
                        row[4] == "holdings.base_market_value"
                        and row[5] == "csgbSHEL.L"
                        and row[9] is not None
                        for row in underlying_rows
                    )
                )
                ai_row = next(
                    row
                    for row in underlying_rows
                    if row[0] == "INCOME"
                    and row[4] == "transactions.amount"
                    and row[5] == "causMARGIN"
                    and str(row[3])[:10] == "2026-01-22"
                )
                self.assertTrue(str(ai_row[10]).startswith("ai:"))
                ti_row = next(
                    row
                    for row in underlying_rows
                    if row[0] == "BALANCED"
                    and row[4] == "transactions.amount"
                    and row[5] == "csusJPM"
                    and str(row[3])[:10] == "2026-03-20"
                )
                self.assertTrue(str(ti_row[10]).startswith("ti:"))
                jpm_dividend_row = next(
                    row
                    for row in underlying_rows
                    if row[0] == "BALANCED"
                    and row[4] == "transactions.amount"
                    and row[5] == "csusJPM"
                    and str(row[1])[:10] == "2026-04-01"
                    and str(row[2])[:10] == "2026-04-10"
                    and str(row[10]).startswith("dv:")
                )
                self.assertEqual(str(jpm_dividend_row[3])[:10], "2026-04-06")
                self.assertEqual(
                    jpm_dividend_row[10],
                    (
                        "dv: csusJPM transactions.amount increased by 10.58. This "
                        "affects the performance calculation through cash-balance "
                        "ending holdings.market_value."
                    ),
                )
                sap_base_dividend_row = next(
                    row
                    for row in underlying_rows
                    if row[0] == "BALANCED"
                    and row[4] == "transactions.base_amount"
                    and row[5] == "cseuSAP.DE"
                    and str(row[1])[:10] == "2026-04-11"
                    and str(row[2])[:10] == "2026-04-16"
                )
                self.assertEqual(
                    sap_base_dividend_row[10],
                    (
                        "dv: cseuSAP.DE transactions.base_amount increased by 32.40. "
                        "This affects the performance calculation through "
                        "cash-balance ending holdings.base_market_value. Local amount "
                        "changed from EUR 120.00 to EUR 150.00. The implied conversion "
                        "ratio remained 1.080000 USD per EUR."
                    ),
                )
                eur_cash_quantity_row = next(
                    row
                    for row in underlying_rows
                    if row[0] == "BALANCED"
                    and row[4] == "holdings.quantity"
                    and row[5] == "causCASHEUR"
                    and str(row[1])[:10] == "2026-04-11"
                    and str(row[2])[:10] == "2026-04-16"
                )
                self.assertAlmostEqual(_numeric_value(eur_cash_quantity_row[8]), 30.0)
                self.assertEqual(
                    eur_cash_quantity_row[10],
                    (
                        "causCASHEUR ending holdings.quantity increased by 30.00. "
                        "This affects the performance calculation through "
                        "holdings.base_market_value."
                    ),
                )
                eur_base_change_fields = {
                    row[4]
                    for row in underlying_rows
                    if row[0] == "BALANCED"
                    and str(row[1])[:10] == "2026-04-11"
                    and str(row[2])[:10] == "2026-04-16"
                    and row[8] is not None
                    and abs(_numeric_value(row[8]) - 32.4) <= 0.005
                }
                self.assertEqual(
                    eur_base_change_fields,
                    {
                        "holdings.base_market_value",
                        "transactions.base_amount",
                    },
                )
                price_rows = [
                    row
                    for row in underlying_rows
                    if row[4] == "holdings.price"
                ]
                self.assertTrue(price_rows)
                self.assertTrue(
                    all(
                        str(row[10]).endswith(
                            "This affects the performance calculation through "
                            "holdings.market_value."
                        )
                        for row in price_rows
                    )
                )
                income_fee_row = next(
                    row
                    for row in underlying_rows
                    if row[0] == "INCOME"
                    and row[4] == "transactions.amount"
                    and row[5] == "causCASHUSD"
                    and str(row[1])[:10] == "2026-01-01"
                    and str(row[2])[:10] == "2026-01-30"
                )
                self.assertEqual(
                    income_fee_row[10],
                    (
                        "dp: causCASHUSD transactions.amount decreased by 50.00. This "
                        "affects the performance calculation through cash-balance "
                        "ending holdings.market_value."
                    ),
                )
                alpha_withdrawal_row = next(
                    row
                    for row in underlying_rows
                    if row[0] == "ALPHA"
                    and row[4] == "transactions.amount"
                    and row[5] == "causCASHUSD"
                    and str(row[1])[:10] == "2026-01-31"
                    and str(row[2])[:10] == "2026-02-27"
                )
                self.assertEqual(
                    alpha_withdrawal_row[10],
                    (
                        "lo: causCASHUSD transactions.amount decreased by 2,000.00. "
                        "This affects the performance calculation through weighted "
                        "external flow, which decreased by 785.71."
                    ),
                )
                transaction_component_guidance = [
                    str(row[10])
                    for row in underlying_rows
                    if row[4]
                    in {
                        "transactions.commission",
                        "transactions.price",
                        "transactions.quantity",
                    }
                ]
                transaction_guidance = [
                    str(row[10])
                    for row in underlying_rows
                    if str(row[4]).startswith("transactions.")
                ]
                self.assertFalse(
                    any(
                        "Helped explain the changed transactions.amount" in guidance
                        for guidance in transaction_component_guidance
                    )
                )
                self.assertFalse(
                    any(
                        guidance.startswith(("by: Caused ", "sl: Caused "))
                        for guidance in transaction_component_guidance
                    )
                )
                self.assertTrue(
                    any(
                        guidance.startswith("by: ")
                        and "performance calculation through" in guidance
                        for guidance in transaction_component_guidance
                    )
                )
                self.assertTrue(
                    any(
                        guidance.startswith("sl: ")
                        and "performance calculation through" in guidance
                        for guidance in transaction_component_guidance
                    )
                )
                income_tnote_buy_quantity_guidance = {
                    str(row[10])
                    for row in underlying_rows
                    if row[0] == "INCOME"
                    and row[4] == "transactions.quantity"
                    and row[5] == "fius91282Y5Y1"
                    and str(row[1])[:10] == "2026-01-31"
                    and str(row[2])[:10] == "2026-02-13"
                }
                income_tnote_sell_quantity_guidance = {
                    str(row[10])
                    for row in underlying_rows
                    if row[0] == "INCOME"
                    and row[4] == "transactions.quantity"
                    and row[5] == "fius91282Y5Y1"
                    and str(row[1])[:10] == "2026-02-14"
                    and str(row[2])[:10] == "2026-02-27"
                }
                self.assertIn(
                    (
                        "by: fius91282Y5Y1 transactions.quantity increased by 5.00. "
                        "This affects the performance calculation through "
                        "transactions.amount and holdings.market_value."
                    ),
                    income_tnote_buy_quantity_guidance,
                )
                self.assertIn(
                    (
                        "sl: fius91282Y5Y1 transactions.quantity increased by 3.00. "
                        "This affects the performance calculation through "
                        "transactions.amount and holdings.market_value."
                    ),
                    income_tnote_sell_quantity_guidance,
                )
                income_tnote_buy_amount_guidance = {
                    str(row[10])
                    for row in underlying_rows
                    if row[0] == "INCOME"
                    and row[4] == "transactions.amount"
                    and row[5] == "fius91282Y5Y1"
                    and str(row[1])[:10] == "2026-01-31"
                    and str(row[2])[:10] == "2026-02-13"
                }
                income_tnote_sell_amount_guidance = {
                    str(row[10])
                    for row in underlying_rows
                    if row[0] == "INCOME"
                    and row[4] == "transactions.amount"
                    and row[5] == "fius91282Y5Y1"
                    and str(row[1])[:10] == "2026-02-14"
                    and str(row[2])[:10] == "2026-02-27"
                }
                self.assertIn(
                    (
                        "pa: fius91282Y5Y1 transactions.amount decreased by 42.50. "
                        "This affects the performance calculation through "
                        "cash-balance ending holdings.market_value."
                    ),
                    income_tnote_buy_amount_guidance,
                )
                self.assertIn(
                    (
                        "sa: fius91282Y5Y1 transactions.amount increased by 37.25. "
                        "This affects the performance calculation through "
                        "cash-balance ending holdings.market_value."
                    ),
                    income_tnote_sell_amount_guidance,
                )
                income_tnote_buy_accrued_guidance = {
                    str(row[10])
                    for row in underlying_rows
                    if row[0] == "INCOME"
                    and row[4] == "holdings.accrued"
                    and row[5] == "fius91282Y5Y1"
                    and str(row[1])[:10] == "2026-01-31"
                    and str(row[2])[:10] == "2026-02-13"
                }
                income_tnote_sell_accrued_guidance = {
                    str(row[10])
                    for row in underlying_rows
                    if row[0] == "INCOME"
                    and row[4] == "holdings.accrued"
                    and row[5] == "fius91282Y5Y1"
                    and str(row[1])[:10] == "2026-02-14"
                    and str(row[2])[:10] == "2026-02-27"
                }
                self.assertEqual(
                    income_tnote_buy_accrued_guidance,
                    {"fius91282Y5Y1 ending holdings.accrued increased by 0.48."},
                )
                self.assertEqual(
                    income_tnote_sell_accrued_guidance,
                    {
                        (
                            "fius91282Y5Y1 beginning holdings.accrued increased by 0.48."
                        ),
                        "fius91282Y5Y1 ending holdings.accrued increased by 0.19.",
                    },
                )
                self.assertFalse(
                    any(
                        guidance.startswith(("pa:", "sa:"))
                        for guidance in (
                            income_tnote_buy_accrued_guidance | income_tnote_sell_accrued_guidance
                        )
                    )
                )
                self.assertFalse(
                    any(
                        guidance.startswith(("BY:", "SL:", "DV:", "DP:", "IN:", "WD:"))
                        for guidance in transaction_guidance
                    )
                )
                self.assertTrue(
                    any(guidance.startswith("dv: ") for guidance in transaction_guidance)
                )
                raw_table = _pc_workbook_tables._workbook_raw_audit_trail_table(
                    findings,
                    comparison_path=_PORTFOLIO_COMPARISON_PATH,
                ).select(_pc_workbook_tables._workbook_raw_audit_columns(findings))
                raw_rows = list(raw_table.iter_rows())
                raw_fields = {row[4] for row in raw_rows}
                self.assertNotIn("holdings.cost", raw_fields)
                self.assertTrue(
                    any(
                        row[0] == "BALANCED"
                        and _workbook_date_text(row[1]) == "2026-05-09"
                        and _workbook_date_text(row[2]) == "2026-05-14"
                        and row[4] == "transactions.amount"
                        and row[5] == "csusJPM"
                        and row[9]
                        == (
                            "rc: csusJPM transactions.amount increased by 200.00. "
                            "This affects the performance calculation through "
                            "cash-balance ending holdings.market_value."
                        )
                        for row in raw_rows
                    )
                )
            finally:
                workbook.close()

    def test_security_review_workbook_uses_security_as_primary_level(self) -> None:
        """Security comparison workbooks start with security-period differences."""
        openpyxl: Any = importlib.import_module("openpyxl")

        findings = compare_snapshots(
            _PORTFOLIO_COMPARISON_PATH,
            comparison_level="security",
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = write_audit_report_bundle(
                findings,
                Path(directory) / "bundle",
                include_workbook=True,
                comparison_path=_PORTFOLIO_COMPARISON_PATH,
                comparison_level="security",
            )
            readme = paths["readme"].read_text(encoding="utf-8")
            self.assertIn(
                "the exact security periods that changed",
                readme,
            )
            self.assertNotIn("the exact performance periods that changed", readme)
            self.assertIn("`security_audit.xlsx` or `security_audit.html`", readme)
            self.assertIn("`audit_support.zip`", readme)

            workbook = openpyxl.load_workbook(
                paths["review_workbook"],
                data_only=True,
            )
            try:
                _assert_print_layout(self, workbook)
                self.assertEqual(workbook.sheetnames, _EXPECTED_SECURITY_SHEETS)
                self.assertEqual(
                    _header_values(workbook["Performance Differences"]),
                    [
                        "Portfolio",
                        "From Date",
                        "Thru Date",
                        "Security",
                        "Performance Difference",
                        "Explained Difference",
                        "Unexplained Difference",
                        "Status",
                        "Comments",
                    ],
                )
                security_rows = _sheet_rows(workbook["Performance Differences"])
                self.assertEqual(
                    {row[0] for row in security_rows},
                    {
                        "ALPHA",
                        "BALANCED",
                        "BALANCED_CONTRIBUTION",
                        "INCOME",
                    },
                )
                self.assertEqual(
                    {row[3] for row in security_rows},
                    {
                        "csusAAPL",
                        "causCASHUSD",
                        "csusCVNA",
                        "csusJPM",
                        "csusMSFT",
                        "fius91282Y2Y1",
                        "fius91282Y5Y1",
                        "cseuSAP.DE",
                        "csgbSHEL.L",
                        "causCASHEUR",
                        "fius36225MBS1",
                    },
                )
                self.assertEqual(
                    {row[7] for row in security_rows},
                    {"Fully Explained", "Partly Explained", "Unexplained"},
                )
                self.assertEqual(
                    {
                        (
                            row[0],
                            row[3],
                            _workbook_date_text(row[1]),
                            _workbook_date_text(row[2]),
                        )
                        for row in security_rows
                        if row[7] == "Partly Explained"
                    },
                    {("BALANCED", "csusMSFT", "2026-05-09", "2026-05-14")},
                )
                self.assertEqual(
                    {
                        (
                            row[0],
                            row[3],
                            _workbook_date_text(row[1]),
                            _workbook_date_text(row[2]),
                        )
                        for row in security_rows
                        if row[7] == "Unexplained"
                    },
                    {
                        ("BALANCED", "csusJPM", "2026-05-09", "2026-05-14"),
                        ("INCOME", "fius91282Y5Y1", "2026-04-01", "2026-04-30"),
                    },
                )
                jpm_possible_cause_row = next(
                    row
                    for row in security_rows
                    if row[0] == "BALANCED"
                    and row[3] == "csusJPM"
                    and _workbook_date_text(row[1]) == "2026-05-09"
                    and _workbook_date_text(row[2]) == "2026-05-14"
                )
                self.assertIn(
                    "Possible cause: csusJPM transactions.amount increased "
                    "by 200.00 on 2026-05-12. Add YAML configuration to count "
                    "it as explained.",
                    str(jpm_possible_cause_row[8]),
                )
                fully_explained_rows = [
                    row for row in security_rows if row[7] == "Fully Explained"
                ]
                self.assertTrue(fully_explained_rows)
                for row in fully_explained_rows:
                    with self.subTest(portfolio=row[0], security=row[3]):
                        self.assertAlmostEqual(
                            _numeric_value(row[4]),
                            _numeric_value(row[5]),
                            places=6,
                        )
                        self.assertIsNone(row[6])
                aapl_trade_row = next(
                    row
                    for row in security_rows
                    if row[0] == "ALPHA"
                    and str(row[1])[:10] == "2026-02-28"
                    and str(row[2])[:10] == "2026-03-31"
                    and row[3] == "csusAAPL"
                )
                self.assertEqual(aapl_trade_row[7], "Fully Explained")
                self.assertAlmostEqual(
                    _numeric_value(aapl_trade_row[4]),
                    _numeric_value(aapl_trade_row[5]),
                    places=6,
                )
                self.assertIsNone(aapl_trade_row[6])
                aapl_price_periods = {
                    ("ALPHA", "2026-05-01", "2026-05-29"),
                    ("BALANCED", "2026-02-28", "2026-03-31"),
                    ("INCOME", "2026-05-01", "2026-05-08"),
                }
                self.assertTrue(
                    aapl_price_periods.issubset(
                        {
                            (
                                row[0],
                                _workbook_date_text(row[1]),
                                _workbook_date_text(row[2]),
                            )
                            for row in security_rows
                            if row[3] == "csusAAPL"
                        }
                    )
                )
                aapl_price_rows = [
                    row
                    for row in security_rows
                    if row[3] == "csusAAPL"
                    and (row[0], _workbook_date_text(row[1]), _workbook_date_text(row[2]))
                    in aapl_price_periods
                ]
                for row in aapl_price_rows:
                    self.assertEqual(row[7], "Fully Explained")
                    self.assertAlmostEqual(
                        _numeric_value(row[4]),
                        _numeric_value(row[5]),
                        places=6,
                    )
                    self.assertIsNone(row[6])
                tnote_row = next(row for row in security_rows if row[3] == "fius91282Y2Y1")
                self.assertEqual(tnote_row[7], "Fully Explained")
                self.assertAlmostEqual(
                    _numeric_value(tnote_row[4]),
                    0.005414,
                    places=6,
                )
                self.assertAlmostEqual(
                    _numeric_value(tnote_row[5]),
                    _numeric_value(tnote_row[4]),
                    places=6,
                )
                self.assertIsNone(tnote_row[6])
                raw_table = _pc_workbook_tables._workbook_raw_audit_trail_table(
                    findings,
                    comparison_path=_PORTFOLIO_COMPARISON_PATH,
                    comparison_level="security",
                ).select(_pc_workbook_tables._workbook_raw_audit_columns(findings))
                raw_rows = list(raw_table.iter_rows())
                self.assertNotIn("holdings.cost", {row[4] for row in raw_rows})
                raw_sort_keys = [
                    (
                        row[0],
                        str(row[1])[:10],
                        str(row[2])[:10],
                        str(row[3])[:10],
                        row[4],
                        row[5],
                    )
                    for row in raw_rows
                ]
                self.assertEqual(raw_sort_keys, sorted(raw_sort_keys))
                underlying_rows = _sheet_rows(workbook["Performance Difference Causes"])
                balanced_msft_cause_fields = {
                    row[4]
                    for row in underlying_rows
                    if row[0] == "BALANCED"
                    and _workbook_date_text(row[1]) == "2026-05-09"
                    and _workbook_date_text(row[2]) == "2026-05-14"
                    and row[5] == "csusMSFT"
                }
                self.assertIn("holdings.market_value", balanced_msft_cause_fields)
                self.assertNotIn(
                    "holdings.base_market_value",
                    balanced_msft_cause_fields,
                )
                underlying_sort_keys = [
                    (
                        row[0],
                        str(row[1])[:10],
                        str(row[2])[:10],
                        str(row[3])[:10],
                        row[4],
                        row[5],
                    )
                    for row in underlying_rows
                ]
                self.assertEqual(underlying_sort_keys, sorted(underlying_sort_keys))
                self.assertTrue(
                    {
                        ("transactions.amount", "fius91282Y2Y1"),
                        ("holdings.market_value", "csusAAPL"),
                        ("transactions.amount", "csusAAPL"),
                    }.issubset({(row[4], row[5]) for row in underlying_rows})
                )
                self.assertTrue(
                    {
                        ("holdings.market_value", "fius91282Y2Y1"),
                        ("holdings.quantity", "fius91282Y2Y1"),
                    }.issubset({(row[4], row[5]) for row in underlying_rows})
                )
                tnote_interest_row = next(
                    row
                    for row in underlying_rows
                    if row[4] == "transactions.amount"
                    and row[5] == "fius91282Y2Y1"
                    and str(row[1])[:10] == "2026-05-15"
                    and str(row[2])[:10] == "2026-05-15"
                    and str(row[3])[:10] == "2026-05-15"
                )
                self.assertEqual(str(tnote_interest_row[3])[:10], "2026-05-15")
                self.assertEqual(
                    tnote_interest_row[10],
                    (
                        "in: fius91282Y2Y1 transactions.amount increased by 80.00. "
                        "This affects the performance calculation through income."
                    ),
                )
                cash_fee_row = next(
                    row
                    for row in underlying_rows
                    if row[0] == "INCOME"
                    and row[4] == "transactions.amount"
                    and row[5] == "causCASHUSD"
                    and str(row[1])[:10] == "2026-01-01"
                    and str(row[2])[:10] == "2026-01-30"
                    and str(row[3])[:10] == "2026-01-20"
                )
                self.assertEqual(
                    cash_fee_row[10],
                    (
                        "dp: causCASHUSD transactions.amount decreased by 50.00. "
                        "This affects the performance calculation through income."
                    ),
                )
                jpm_dividend_row = next(
                    row
                    for row in underlying_rows
                    if row[0] == "BALANCED"
                    and row[4] == "transactions.amount"
                    and row[5] == "csusJPM"
                    and str(row[1])[:10] == "2026-04-01"
                    and str(row[2])[:10] == "2026-04-10"
                    and str(row[10]).startswith("dv:")
                )
                self.assertEqual(
                    jpm_dividend_row[10],
                    (
                        "dv: csusJPM transactions.amount increased by 10.58. This "
                        "affects the performance calculation through income."
                    ),
                )
            finally:
                workbook.close()

            workbook_with_comments = openpyxl.load_workbook(
                paths["review_workbook"],
                read_only=False,
                data_only=True,
            )
            try:
                executive_sheet = workbook_with_comments["Executive Summary"]
                differences_sheet = workbook_with_comments["Performance Differences"]
                causes_sheet = workbook_with_comments["Performance Difference Causes"]

                self.assertIsNotNone(executive_sheet["D2"].comment)
                assert executive_sheet["D2"].comment is not None
                self.assertIn(
                    "accounted for by supported, quantified causes",
                    executive_sheet["D2"].comment.text,
                )
                self.assertEqual(
                    _header_comment(differences_sheet, "Performance Difference"),
                    ("Snapshot B reported performance minus snapshot A " "reported performance."),
                )
                self.assertNotIn("Review Key", _header_values(differences_sheet))
                self.assertIn(
                    "Fully Explained:",
                    _header_comment(differences_sheet, "Status"),
                )
                self.assertEqual(
                    _header_comment(
                        causes_sheet,
                        "Performance Difference Explained",
                    ),
                    (
                        "A supported, quantified cause included in Explained "
                        "Difference. A blank value is not counted."
                    ),
                )
            finally:
                workbook_with_comments.close()

    def test_review_workbook_can_include_reconstruction_diagnostics(self) -> None:
        """Reconstruction diagnostic sheets are available by explicit opt-in."""
        openpyxl: Any = importlib.import_module("openpyxl")

        findings = compare_snapshots(
            _PORTFOLIO_COMPARISON_PATH,
            comparison_level="portfolio",
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = write_audit_report_bundle(
                findings,
                Path(directory) / "bundle",
                include_workbook=True,
                comparison_path=_PORTFOLIO_COMPARISON_PATH,
                include_reconstruction_diagnostics=True,
            )
            self.assertIn(_pc_review_model.RECONSTRUCTION_SUMMARY_ARTIFACT, paths)
            self.assertIn(
                _pc_review_model.RETURN_RECONSTRUCTION_CHECKS_ARTIFACT,
                paths,
            )
            self.assertIn(
                _pc_review_model.SECURITY_RETURN_RECONSTRUCTION_CHECKS_ARTIFACT,
                paths,
            )
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            self.assertTrue(manifest["options"]["include_reconstruction_diagnostics"])
            self.assertEqual(
                manifest["review_entrypoints"]["return_reconstruction"],
                [
                    "supporting_files/reconstruction_summary.csv",
                    "supporting_files/return_reconstruction_checks.csv",
                    "supporting_files/security_return_reconstruction_checks.csv",
                ],
            )

            workbook = openpyxl.load_workbook(
                paths["review_workbook"],
                read_only=True,
                data_only=True,
            )
            try:
                self.assertEqual(workbook.sheetnames, _EXPECTED_DIAGNOSTIC_SHEETS)
            finally:
                workbook.close()

            html_report = paths["html_report"].read_text(encoding="utf-8")
            self.assertLess(
                html_report.index(_pc_review_model.PERFORMANCE_DIFFERENCE_CAUSES_SHEET),
                html_report.index(_pc_review_model.RECONSTRUCTION_SUMMARY_SHEET),
            )
            self.assertNotIn(
                f"<h2>{_pc_review_model.SOURCE_DETAIL_SHEET}</h2>",
                html_report,
            )
