"""Tests for performance comparison suppression rules."""

# Python imports
from pathlib import Path
import tempfile
import unittest

# Third-party imports
import polars as pl
import yaml

# Test imports
from tests import test_utilities as test_util

# Project imports
from ppar.errors import PpaError
from ppar.audit import compare_snapshots, summarize_findings
from ppar.audit.performance_comparison.findings import (
    FINDING_CODE,
    PC_SEC_RET,
    SECURITY_ID,
    SOURCE_COLUMN,
    SUPPRESSED,
)
from ppar.audit import schema as pc_cols

_AXYS_DATA_PATH = Path("tests/data/axys/snapshots").resolve()
_SUPPRESSED_COMPARISON_PATH = Path(
    "tests/data/axys/validation/ppar_audit_suppressed.yaml"
)


def _write_suppression_specification(
    directory: Path,
    suppressions: object,
) -> Path:
    """Write a comparison specification with configurable suppressions."""
    configuration = {
        "comparison": {"level": "portfolio"},
        "snapshots": {
            "a": {"path": str(_AXYS_DATA_PATH / "axys_a")},
            "b": {"path": str(_AXYS_DATA_PATH / "axys_b_restatement")},
        },
        "files": {
            "portfolio_performance": "portperf.csv",
            "security_performance": "secperf.csv",
        },
        "extract_contract": {
            "enforce_ambiguous_axys_flows": True,
        },
        "tolerances": {
            "return": 0.000001,
            "market_value": 0.01,
            "quantity": 0.000001,
            "price": 0.000001,
            "split_factor": 0.00000001,
            "fx_rate": 0.00000001,
        },
        "suppressions": suppressions,
    }
    specification_path = directory / "ppar_audit.yaml"
    test_util.write_audit_test_yaml(specification_path, configuration)
    return specification_path


class TestPerformanceComparisonRules(unittest.TestCase):
    """Verify finding suppression rule behavior."""

    def test_suppression_marks_matching_findings_only(self) -> None:
        """Exact-match suppressions mark matching findings as suppressed."""
        findings = compare_snapshots(_SUPPRESSED_COMPARISON_PATH)
        security_return_findings = findings.filter(
            (pl.col(FINDING_CODE) == PC_SEC_RET)
            & (pl.col(SECURITY_ID) == "AAPL")
            & (pl.col(SOURCE_COLUMN) == pc_cols.SECURITY_RETURN)
        )
        unsuppressed_security_findings = findings.filter(
            (pl.col(FINDING_CODE) != PC_SEC_RET)
            & (pl.col(SECURITY_ID) == "AAPL")
        )

        self.assertEqual(security_return_findings.height, 1)
        self.assertEqual(
            security_return_findings.get_column(SUPPRESSED).to_list(),
            [True],
        )
        self.assertGreater(unsuppressed_security_findings.height, 0)
        self.assertEqual(
            unsuppressed_security_findings.get_column(SUPPRESSED).unique().to_list(),
            [False],
        )

        summaries = summarize_findings(findings)
        suppression_counts = {
            row[SUPPRESSED]: row["count"]
            for row in summaries["by_suppressed"].iter_rows(named=True)
        }
        self.assertEqual(suppression_counts[True], 1)
        self.assertGreater(suppression_counts[False], 0)

    def test_compare_snapshots_can_exclude_suppressed_findings(self) -> None:
        """The public runner can return only active unsuppressed findings."""
        all_findings = compare_snapshots(_SUPPRESSED_COMPARISON_PATH)
        active_findings = compare_snapshots(
            _SUPPRESSED_COMPARISON_PATH,
            include_suppressed=False,
        )
        suppressed_security_returns = active_findings.filter(
            (pl.col(FINDING_CODE) == PC_SEC_RET)
            & (pl.col(SECURITY_ID) == "AAPL")
            & (pl.col(SOURCE_COLUMN) == pc_cols.SECURITY_RETURN)
        )

        self.assertEqual(all_findings.height - active_findings.height, 1)
        self.assertTrue(suppressed_security_returns.is_empty())
        self.assertEqual(active_findings.get_column(SUPPRESSED).unique().to_list(), [False])

    def test_nonmatching_suppression_does_not_mark_findings(self) -> None:
        """Suppression criteria must all match before a finding is suppressed."""
        suppressions = [{"code": PC_SEC_RET, "security_id": "MSFT"}]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _write_suppression_specification(Path(temp_dir), suppressions)

            findings = compare_snapshots(path)

            self.assertEqual(findings.get_column(SUPPRESSED).unique().to_list(), [False])

    def test_invalid_suppression_shape_raises_error_504(self) -> None:
        """Suppression entries must be mappings with string codes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _write_suppression_specification(Path(temp_dir), [{"reason": "bad"}])

            with self.assertRaises(PpaError) as context:
                compare_snapshots(path)

            self.assertTrue(str(context.exception).startswith("Error 504"))
            self.assertIn("suppressions[0].code", str(context.exception))


if __name__ == "__main__":
    unittest.main()
