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

_FULL_SPEC_COMPARISON_PATH = Path(
    "ppar/demos/data/axys/ppar_performance_comparison_full_spec.yaml"
)
_SECURITY_SPEC_COMPARISON_PATH = Path(
    "ppar/demos/data/axys/ppar_performance_comparison_security_full_spec.yaml"
)

_EXPECTED_PORTFOLIO_SHEETS = [
    "Portfolio Differences",
    "Underlying Causes",
    "Reported Performance Checks",
    "Context",
    "Raw Audit Trail",
]
_EXPECTED_SECURITY_SHEETS = [
    "Security Differences",
    "Underlying Causes",
    "Reported Performance Checks",
    "Context",
    "Raw Audit Trail",
]
_COMMON_LEFT_HEADERS = [
    "Portfolio",
    "From Date",
    "Thru Date",
    "Dataset",
    "Source Column",
    "Security",
]
_NON_ADDITIVE_HEADERS = [
    *_COMMON_LEFT_HEADERS,
    "Snapshot A Value",
    "Snapshot B Value",
    "B - A Difference",
    "What Changed",
    "Next Action",
    "Review Key",
]


def _header_values(worksheet: Any) -> list[object]:
    """Return worksheet header values."""
    return [cell.value for cell in worksheet[1]]


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


class TestPerformanceComparisonWorkbookContract(unittest.TestCase):
    """Validate reviewer-facing workbook presentation invariants."""

    def test_review_workbook_contract_remains_reviewer_oriented(self) -> None:
        """Generated workbook uses stable, action-oriented sheets and columns."""
        openpyxl: Any = importlib.import_module("openpyxl")

        findings = compare_snapshots(_FULL_SPEC_COMPARISON_PATH)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_performance_comparison_report_bundle(
                findings,
                Path(directory) / "bundle",
                include_workbook=True,
                comparison_path=_FULL_SPEC_COMPARISON_PATH,
            )

            readme = paths["readme"].read_text(encoding="utf-8")
            self.assertIn("Reported Performance Checks sheet", readme)
            self.assertIn("Raw Audit Trail sheet", readme)
            self.assertNotIn("Derived Checks", readme)

            workbook = openpyxl.load_workbook(
                paths["review_workbook"],
                read_only=True,
                data_only=True,
            )
            try:
                self.assertEqual(workbook.sheetnames, _EXPECTED_PORTFOLIO_SHEETS)
                self.assertNotIn("Security Differences", workbook.sheetnames)
                self.assertNotIn("Derived Checks", workbook.sheetnames)

                self.assertEqual(
                    _header_values(workbook["Portfolio Differences"]),
                    [
                        "Portfolio",
                        "From Date",
                        "Thru Date",
                        "Performance Difference",
                        "Explained Difference",
                        "Unexplained Difference",
                        "Status",
                        "Next Action",
                        "Review Key",
                    ],
                )
                self.assertEqual(
                    _header_values(workbook["Underlying Causes"])[:13],
                    [
                        *_COMMON_LEFT_HEADERS,
                        "Snapshot A Value",
                        "Snapshot B Value",
                        "B - A Difference",
                        "Impact Input Value",
                        "Performance Difference Explained",
                        "Required YAML Setup",
                        "Review Key",
                    ],
                )
                self.assertEqual(
                    _header_values(workbook["Reported Performance Checks"]),
                    _NON_ADDITIVE_HEADERS,
                )
                self.assertEqual(
                    _header_values(workbook["Context"]),
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
                    workbook["Portfolio Differences"],
                    "D",
                )
                portfolio_rows = _sheet_rows(workbook["Portfolio Differences"])
                portfolio_codes = {row[0] for row in portfolio_rows}
                statuses = {row[6] for row in portfolio_rows}
                self.assertTrue(portfolio_differences)
                self.assertEqual(portfolio_codes, {"ALPHA", "BALANCED", "INCOME"})
                self.assertEqual(
                    statuses,
                    {"Fully Explained", "Partly Explained", "Unexplained"},
                )
                self.assertEqual(len(portfolio_rows), 6)
                self.assertEqual(
                    sum(1 for row in portfolio_rows if row[6] == "Fully Explained"),
                    4,
                )
                self.assertTrue(
                    all(
                        isinstance(value, (int, float)) and not isinstance(value, bool)
                        for value in portfolio_differences
                    )
                )
                self.assertEqual(
                    workbook["Portfolio Differences"]["D2"].number_format,
                    "0.######",
                )
                underlying_rows = _sheet_rows(workbook["Underlying Causes"])
                self.assertFalse(
                    any(row[3] == "no_underlying_cause_found" for row in underlying_rows)
                )
                self.assertTrue(
                    all(
                        row[11] in (None, "None")
                        or "No additive underlying cause" in str(row[11])
                        or "not included in explained difference" in str(row[11])
                        or "related performance input row is selected" in str(row[11])
                        for row in underlying_rows
                    )
                )
                underlying_fields = {(row[3], row[4]) for row in underlying_rows}
                self.assertTrue(
                    {
                        ("positions", "market_value"),
                        ("positions", "quantity"),
                        ("transactions", "commission"),
                        ("transactions", "price"),
                        ("transactions", "quantity"),
                    }.issubset(underlying_fields)
                )
                context_rows = _sheet_rows(workbook["Context"])
                context_fields = {(row[3], row[4]) for row in context_rows}
                self.assertEqual(
                    {
                        ("positions", "cost"),
                    },
                    context_fields,
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
                self.assertNotIn("Portfolio Differences", workbook.sheetnames)
                self.assertEqual(
                    _header_values(workbook["Security Differences"]),
                    [
                        "Portfolio",
                        "From Date",
                        "Thru Date",
                        "Security",
                        "Performance Difference",
                        "Explained Difference",
                        "Unexplained Difference",
                        "Status",
                        "Next Action",
                        "Review Key",
                    ],
                )
                review_keys = _column_values(workbook["Security Differences"], "J")
                self.assertTrue(review_keys)
                self.assertTrue(any(str(key).endswith("::AAPL") for key in review_keys))
                self.assertTrue(
                    any(str(key).endswith("::TNOTE2Y") for key in review_keys)
                )
                security_rows = _sheet_rows(workbook["Security Differences"])
                self.assertEqual(
                    {row[0] for row in security_rows},
                    {"ALPHA", "INCOME"},
                )
                self.assertEqual(
                    {row[3] for row in security_rows},
                    {"AAPL", "TNOTE2Y"},
                )
                context_rows = _sheet_rows(workbook["Context"])
                self.assertEqual(
                    {(row[3], row[4], row[5]) for row in context_rows},
                    {
                        ("positions", "cost", "TNOTE2Y"),
                    },
                )
                underlying_rows = _sheet_rows(workbook["Underlying Causes"])
                self.assertTrue(
                    {
                        ("positions", "market_value", "TNOTE2Y"),
                        ("positions", "quantity", "TNOTE2Y"),
                    }.issubset(
                        {(row[3], row[4], row[5]) for row in underlying_rows}
                    )
                )
            finally:
                workbook.close()
