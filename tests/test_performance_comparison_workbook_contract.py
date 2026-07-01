"""Contract tests for the performance-comparison review workbook."""

from __future__ import annotations

# Python imports
import importlib
import json
from pathlib import Path
import tempfile
from typing import Any
import unittest

# Project imports
from ppar.performance_comparison import (
    compare_snapshots,
    write_performance_comparison_report_bundle,
)
from ppar.performance_comparison import review_model as _pc_review_model

_PORTFOLIO_COMPARISON_PATH = Path(
    "ppar/demos/data/axys/ppar_performance_comparison.yaml"
)

_EXPECTED_PORTFOLIO_SHEETS = [
    _pc_review_model.PERFORMANCE_DIFFERENCES_SHEET,
    _pc_review_model.PERFORMANCE_DIFFERENCE_CAUSES_SHEET,
    _pc_review_model.OTHER_DATA_DIFFERENCES_SHEET,
    _pc_review_model.RAW_AUDIT_TRAIL_SHEET,
]
_EXPECTED_SECURITY_SHEETS = list(_EXPECTED_PORTFOLIO_SHEETS)
_EXPECTED_DIAGNOSTIC_SHEETS = [
    _pc_review_model.PERFORMANCE_DIFFERENCES_SHEET,
    _pc_review_model.RECONSTRUCTION_SUMMARY_SHEET,
    _pc_review_model.RETURN_RECONSTRUCTION_CHECKS_SHEET,
    _pc_review_model.SECURITY_RETURN_RECONSTRUCTION_CHECKS_SHEET,
    _pc_review_model.PERFORMANCE_DIFFERENCE_CAUSES_SHEET,
    _pc_review_model.OTHER_DATA_DIFFERENCES_SHEET,
    _pc_review_model.RAW_AUDIT_TRAIL_SHEET,
]
_COMMON_LEFT_HEADERS = [
    "Portfolio",
    "From Date",
    "Thru Date",
    "Source Dataset",
    "Input Field",
    "Security",
]
_IDENTIFIABLE_LEFT_HEADERS = [
    "Portfolio",
    "From Date",
    "Thru Date",
    "As Of Date",
    "Dataset Field",
    "Security",
]
_NON_ADDITIVE_HEADERS = [
    *_IDENTIFIABLE_LEFT_HEADERS,
    "Snapshot A Value",
    "Snapshot B Value",
    "B - A Difference",
    "Explanation",
    "Review Key",
]
_EXPECTED_NON_FULLY_EXPLAINED_PORTFOLIO_ROWS = {
    ("BALANCED", "2026-05-01", "2026-05-29", "Partly Explained"),
    ("INCOME", "2026-04-01", "2026-04-30", "Unexplained"),
}


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


class TestPerformanceComparisonWorkbookContract(unittest.TestCase):
    """Validate reviewer-facing workbook presentation invariants."""

    def test_review_workbook_contract_remains_reviewer_oriented(self) -> None:
        """Generated workbook uses stable, action-oriented sheets and columns."""
        openpyxl: Any = importlib.import_module("openpyxl")

        findings = compare_snapshots(_PORTFOLIO_COMPARISON_PATH)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_performance_comparison_report_bundle(
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
            self.assertIn("Raw Audit Trail", readme)
            self.assertNotIn("## Primary Review Artifact", readme)
            self.assertNotIn("Open `report.xlsx` first", readme)
            self.assertNotIn("same review model in a browser", readme)
            html_report = paths["html_report"].read_text(encoding="utf-8")
            self.assertIn("source-data differences", readme)
            self.assertIn("Source Dataset", html_report)
            self.assertNotIn("Source-Data Dataset", html_report)
            self.assertIn(
                "Normalized dataset where the source-data discrepancy was found.",
                html_report,
            )
            self.assertNotIn("Normalized source dataset", html_report)
            self.assertIn(
                "Browser view for reviewing this performance-comparison bundle.",
                html_report,
            )
            self.assertNotIn("same review model", html_report)
            self.assertNotIn("Browser review surface", html_report)
            self.assertNotIn("Transaction Match Diagnostics", html_report)
            self.assertNotIn("Match Confidence", html_report)

            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            review_summary = json.loads(
                paths["review_summary"].read_text(encoding="utf-8")
            )
            for summary in (manifest, review_summary):
                transaction_semantics = summary["transaction_semantics"]
                self.assertIn("by", transaction_semantics["observed_codes"])
                self.assertIn("sl", transaction_semantics["observed_codes"])
                self.assertNotIn("BY", transaction_semantics["observed_codes"])
                self.assertNotIn("SL", transaction_semantics["observed_codes"])

            workbook = openpyxl.load_workbook(
                paths["review_workbook"],
                read_only=True,
                data_only=True,
            )
            try:
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
                        "Review Key",
                    ],
                )
                self.assertEqual(
                    _header_values(workbook["Performance Difference Causes"])[:13],
                    [
                        *_IDENTIFIABLE_LEFT_HEADERS,
                        "Snapshot A Value",
                        "Snapshot B Value",
                        "B - A Difference",
                        "Performance Difference Explained",
                        "Explanation",
                        "Review Key",
                    ],
                )
                self.assertIn(
                    "Performance\nDifference\nExplained",
                    _raw_header_values(workbook["Performance Difference Causes"]),
                )
                self.assertEqual(
                    _header_values(workbook["Other Data Differences"]),
                    _NON_ADDITIVE_HEADERS,
                )
                self.assertEqual(
                    _header_values(workbook["Raw Audit Trail"])[:7],
                    [*_COMMON_LEFT_HEADERS, "Transaction Category"],
                )
                self.assertEqual(
                    _header_values(workbook["Raw Audit Trail"])[-1],
                    "Review Key",
                )

                portfolio_differences = _column_values(
                    workbook["Performance Differences"],
                    "D",
                )
                portfolio_rows = _sheet_rows(workbook["Performance Differences"])
                portfolio_codes = {row[0] for row in portfolio_rows}
                statuses = {row[6] for row in portfolio_rows}
                self.assertTrue(portfolio_differences)
                self.assertEqual(portfolio_codes, {"ALPHA", "BALANCED", "INCOME"})
                self.assertEqual(statuses, {"Fully Explained"})
                self.assertTrue(all(row[6] == "Fully Explained" for row in portfolio_rows))
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
                        and row[5] == "CASH_USD"
                        and row[10]
                        == (
                            "CASH_USD beginning holdings.market_value decreased by "
                            "1,500.00."
                        )
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
                        or (
                            '"Performance Differences"."Explained Difference"'
                            in str(row[10])
                        )
                        or "Input for changed" in str(row[10])
                        or "related performance input" in str(row[10])
                        or "changed transactions.amount" in str(row[10])
                        or "changed holdings.market_value" in str(row[10])
                        or "calculated portfolio-return difference" in str(row[10])
                        or "Configured transaction impact method is present" in str(row[10])
                        or "Modified Dietz" in str(row[10])
                        or "Supporting detail for changed holdings value" in str(row[10])
                        or "External flow" in str(row[10])
                        or "Helped explain" in str(row[10])
                        or "Caused cash-balance" in str(row[10])
                        or "Caused transactions.amount" in str(row[10])
                        or "transactions.amount to" in str(row[10])
                        or "ending holdings." in str(row[10])
                        or "beginning holdings." in str(row[10])
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
                jpm_dividend_row = next(
                    row
                    for row in underlying_rows
                    if row[4] == "transactions.amount"
                    and row[5] == "JPM"
                    and str(row[1])[:10] == "2026-04-01"
                    and str(row[2])[:10] == "2026-04-30"
                )
                self.assertEqual(str(jpm_dividend_row[3])[:10], "2026-04-06")
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
                        "transactions.amount to increase by 0.25" in guidance
                        or "transactions.amount to increase by 0.15" in guidance
                        for guidance in transaction_component_guidance
                    )
                )
                self.assertTrue(
                    any(
                        guidance.startswith("by: Helped explain")
                        or (
                            guidance.startswith("by: Caused ")
                            and "transactions.amount to" in guidance
                        )
                        for guidance in transaction_component_guidance
                    )
                )
                self.assertTrue(
                    any(
                        guidance.startswith("sl: Helped explain")
                        or (
                            guidance.startswith("sl: Caused ")
                            and "transactions.amount to" in guidance
                        )
                        for guidance in transaction_component_guidance
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
                context_rows = _sheet_rows(workbook["Other Data Differences"])
                context_fields = {row[4] for row in context_rows}
                self.assertTrue(
                    {
                        "holdings.cost",
                    }.issubset(context_fields)
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
            paths = write_performance_comparison_report_bundle(
                findings,
                Path(directory) / "bundle",
                include_workbook=True,
                comparison_path=_PORTFOLIO_COMPARISON_PATH,
                comparison_level="security",
            )
            readme = paths["readme"].read_text(encoding="utf-8")
            self.assertIn("explain each security period", readme)
            self.assertIn("follow a security period across CSV artifacts", readme)
            self.assertNotIn("explain each performance period", readme)

            workbook = openpyxl.load_workbook(
                paths["review_workbook"],
                read_only=True,
                data_only=True,
            )
            try:
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
                        "Review Key",
                    ],
                )
                review_keys = _column_values(workbook["Performance Differences"], "J")
                self.assertTrue(review_keys)
                self.assertTrue(any(str(key).endswith("::AAPL") for key in review_keys))
                self.assertTrue(
                    any(str(key).endswith("::TNOTE2Y") for key in review_keys)
                )
                security_rows = _sheet_rows(workbook["Performance Differences"])
                self.assertEqual(
                    {row[0] for row in security_rows},
                    {"ALPHA", "BALANCED", "INCOME"},
                )
                self.assertEqual(
                    {row[3] for row in security_rows},
                    {"AAPL", "CASH_USD", "JPM", "MSFT", "TNOTE2Y"},
                )
                self.assertEqual({row[7] for row in security_rows}, {"Fully Explained"})
                fully_explained_rows = [
                    row for row in security_rows if row[7] == "Fully Explained"
                ]
                self.assertTrue(fully_explained_rows)
                for row in fully_explained_rows:
                    with self.subTest(review_key=row[9]):
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
                    and row[3] == "AAPL"
                )
                self.assertEqual(aapl_trade_row[7], "Fully Explained")
                self.assertAlmostEqual(
                    _numeric_value(aapl_trade_row[4]),
                    _numeric_value(aapl_trade_row[5]),
                    places=6,
                )
                self.assertIsNone(aapl_trade_row[6])
                aapl_price_rows = [
                    row
                    for row in security_rows
                    if row[3] == "AAPL"
                    and str(row[1])[:10] == "2026-05-01"
                    and str(row[2])[:10] == "2026-05-29"
                ]
                self.assertEqual(len(aapl_price_rows), 3)
                for row in aapl_price_rows:
                    self.assertEqual(row[7], "Fully Explained")
                    self.assertAlmostEqual(
                        _numeric_value(row[4]),
                        _numeric_value(row[5]),
                        places=6,
                    )
                    self.assertIsNone(row[6])
                tnote_row = next(
                    row for row in security_rows if row[3] == "TNOTE2Y"
                )
                self.assertAlmostEqual(
                    _numeric_value(tnote_row[4]),
                    (128.0 + 50.0 + 80.0) / 64000.0,
                    places=6,
                )
                context_rows = _sheet_rows(workbook["Other Data Differences"])
                self.assertTrue(
                    {
                        ("holdings.cost", "AAPL"),
                        ("holdings.cost", "CASH_USD"),
                        ("holdings.cost", "MSFT"),
                        ("holdings.cost", "TNOTE2Y"),
                    }.issubset(
                        {(row[4], row[5]) for row in context_rows}
                    )
                )
                underlying_rows = _sheet_rows(workbook["Performance Difference Causes"])
                underlying_sort_keys = [
                    (row[0], str(row[1])[:10], str(row[2])[:10], row[5], row[4])
                    for row in underlying_rows
                ]
                self.assertEqual(underlying_sort_keys, sorted(underlying_sort_keys))
                self.assertTrue(
                    {
                        ("transactions.amount", "TNOTE2Y"),
                        ("holdings.market_value", "AAPL"),
                        ("transactions.amount", "AAPL"),
                    }.issubset(
                        {(row[4], row[5]) for row in underlying_rows}
                    )
                )
                self.assertTrue(
                    {
                        ("holdings.market_value", "TNOTE2Y"),
                        ("holdings.quantity", "TNOTE2Y"),
                    }.issubset(
                        {(row[4], row[5]) for row in underlying_rows}
                    )
                )
                tnote_interest_row = next(
                    row
                    for row in underlying_rows
                    if row[4] == "transactions.amount"
                    and row[5] == "TNOTE2Y"
                    and str(row[1])[:10] == "2026-05-01"
                    and str(row[2])[:10] == "2026-05-29"
                )
                self.assertEqual(str(tnote_interest_row[3])[:10], "2026-05-15")
            finally:
                workbook.close()

            workbook_with_comments = openpyxl.load_workbook(
                paths["review_workbook"],
                read_only=False,
                data_only=True,
            )
            try:
                differences_sheet = workbook_with_comments["Performance Differences"]
                causes_sheet = workbook_with_comments["Performance Difference Causes"]

                self.assertEqual(
                    _header_comment(differences_sheet, "Performance Difference"),
                    (
                        "Snapshot B reported performance minus snapshot A "
                        "reported performance."
                    ),
                )
                self.assertEqual(
                    _header_comment(differences_sheet, "Review Key"),
                    "Stable performance-period key used to connect workbook rows.",
                )
                self.assertEqual(
                    _header_comment(differences_sheet, "Status"),
                    "Reviewer triage status for this performance difference.",
                )
                self.assertEqual(
                    _header_comment(
                        causes_sheet,
                        "Performance Difference Explained",
                    ),
                    (
                        "Decimal performance difference explained by this "
                        "underlying input row."
                    ),
                )
            finally:
                workbook_with_comments.close()

    def test_review_workbook_can_include_reconstruction_diagnostics(self) -> None:
        """Reconstruction diagnostic sheets are available by explicit opt-in."""
        openpyxl: Any = importlib.import_module("openpyxl")

        findings = compare_snapshots(_PORTFOLIO_COMPARISON_PATH)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_performance_comparison_report_bundle(
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
            self.assertTrue(
                manifest["options"]["include_reconstruction_diagnostics"]
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
