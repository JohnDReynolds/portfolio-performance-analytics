"""Contract tests for the performance-comparison review workbook."""

from __future__ import annotations

# Python imports
import importlib
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
    "ppar/demos/data/axys/ppar_performance_comparison_portfolio.yaml"
)
_SECURITY_SPEC_COMPARISON_PATH = Path(
    "ppar/demos/data/axys/ppar_performance_comparison_security.yaml"
)

_EXPECTED_PORTFOLIO_SHEETS = list(_pc_review_model.EXPECTED_REVIEW_SHEETS)
_EXPECTED_SECURITY_SHEETS = list(_pc_review_model.EXPECTED_REVIEW_SHEETS)
_COMMON_LEFT_HEADERS = [
    "Portfolio",
    "From Date",
    "Thru Date",
    "Input Dataset",
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
    *_COMMON_LEFT_HEADERS,
    "Snapshot A Value",
    "Snapshot B Value",
    "B - A Difference",
    "What Changed",
    "Review Guidance",
    "Review Key",
]


def _header_values(worksheet: Any) -> list[object]:
    """Return worksheet header values."""
    return [_normalized_header(cell.value) for cell in worksheet[1]]


def _raw_header_values(worksheet: Any) -> list[object]:
    """Return worksheet header values without normalizing display line breaks."""
    return [cell.value for cell in worksheet[1]]


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

            readme = paths["readme"].read_text(encoding="utf-8")
            self.assertIn("Raw Audit Trail sheet", readme)

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
                    _header_values(workbook["Identifiable Causes"])[:14],
                    [
                        *_IDENTIFIABLE_LEFT_HEADERS,
                        "Snapshot A Value",
                        "Snapshot B Value",
                        "B - A Difference",
                        "Performance Difference Explained",
                        "Related Performance Difference",
                        "Review Guidance",
                        "Review Key",
                    ],
                )
                self.assertIn(
                    "Performance\nDifference\nExplained",
                    _raw_header_values(workbook["Identifiable Causes"]),
                )
                self.assertEqual(
                    _header_values(workbook["Other Evidence"]),
                    _NON_ADDITIVE_HEADERS,
                )
                self.assertEqual(
                    _header_values(workbook["Raw Audit Trail"])[:6],
                    _COMMON_LEFT_HEADERS,
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
                self.assertEqual(
                    statuses,
                    {"Fully Explained", "Unexplained"},
                )
                self.assertEqual(len(portfolio_rows), 10)
                self.assertEqual(
                    sum(1 for row in portfolio_rows if row[6] == "Fully Explained"),
                    9,
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
                underlying_rows = _sheet_rows(workbook["Identifiable Causes"])
                self.assertTrue(
                    all(
                        row[11] in (None, "")
                        or "No additive underlying cause" in str(row[11])
                        or "No identifiable cause" in str(row[11])
                        or "shown for review" in str(row[11])
                        or "not included in explained difference" in str(row[11])
                        or "Input for changed" in str(row[11])
                        or "related performance input" in str(row[11])
                        or "counted portfolio external-flow transaction" in str(row[11])
                        or "changed transactions.amount" in str(row[11])
                        or "changed holdings.market_value" in str(row[11])
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
                context_rows = _sheet_rows(workbook["Other Evidence"])
                context_fields = {(row[3], row[4]) for row in context_rows}
                self.assertTrue(
                    {
                        ("holdings", "cost"),
                    }.issubset(context_fields)
                )
            finally:
                workbook.close()

    def test_security_review_workbook_uses_security_as_primary_level(self) -> None:
        """Security comparison workbooks start with security-period differences."""
        openpyxl: Any = importlib.import_module("openpyxl")

        findings = compare_snapshots(_SECURITY_SPEC_COMPARISON_PATH)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_performance_comparison_report_bundle(
                findings,
                Path(directory) / "bundle",
                include_workbook=True,
                comparison_path=_SECURITY_SPEC_COMPARISON_PATH,
                comparison_level="security",
            )

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
                self.assertEqual(
                    {row[7] for row in security_rows},
                    {"Fully Explained", "Unexplained"},
                )
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
                        self.assertAlmostEqual(
                            _numeric_value(row[6]),
                            0.0,
                            places=6,
                        )
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
                self.assertAlmostEqual(
                    _numeric_value(aapl_trade_row[6]),
                    0.0,
                    places=6,
                )
                aapl_price_rows = [
                    row
                    for row in security_rows
                    if row[3] == "AAPL"
                    and str(row[1])[:10] == "2026-05-01"
                    and str(row[2])[:10] == "2026-05-29"
                ]
                self.assertEqual(len(aapl_price_rows), 3)
                for row in aapl_price_rows:
                    self.assertAlmostEqual(
                        _numeric_value(row[4]),
                        (162.61 - 161.0) / 161.0,
                    )
                tnote_row = next(
                    row for row in security_rows if row[3] == "TNOTE2Y"
                )
                self.assertAlmostEqual(
                    _numeric_value(tnote_row[4]),
                    (128.0 + 50.0 + 80.0) / 64000.0,
                    places=6,
                )
                context_rows = _sheet_rows(workbook["Other Evidence"])
                self.assertTrue(
                    {
                        ("holdings", "cost", "AAPL"),
                        ("holdings", "cost", "CASH_USD"),
                        ("holdings", "cost", "MSFT"),
                        ("holdings", "cost", "TNOTE2Y"),
                    }.issubset(
                        {(row[3], row[4], row[5]) for row in context_rows}
                    )
                )
                underlying_rows = _sheet_rows(workbook["Identifiable Causes"])
                underlying_sort_keys = [
                    (row[0], str(row[1])[:10], str(row[2])[:10], row[5], row[4])
                    for row in underlying_rows
                ]
                self.assertEqual(underlying_sort_keys, sorted(underlying_sort_keys))
                self.assertTrue(
                    {
                        ("transactions.amount", "TNOTE2Y"),
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
