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

_EXPECTED_SHEETS = [
    "Portfolio Differences",
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


class TestPerformanceComparisonWorkbookContract(unittest.TestCase):
    """Validate reviewer-facing workbook presentation invariants."""

    def test_review_workbook_contract_remains_reviewer_oriented(self) -> None:
        """Generated workbook uses stable, action-oriented sheets and columns."""
        openpyxl: Any = importlib.import_module("openpyxl")

        findings = compare_snapshots(
            _FULL_SPEC_COMPARISON_PATH,
            require_causal_attribution=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = write_performance_comparison_report_bundle(
                findings,
                Path(directory) / "bundle",
                include_workbook=True,
                comparison_path=_FULL_SPEC_COMPARISON_PATH,
                require_causal_attribution=True,
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
                self.assertEqual(workbook.sheetnames, _EXPECTED_SHEETS)
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
                self.assertTrue(portfolio_differences)
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
            finally:
                workbook.close()
