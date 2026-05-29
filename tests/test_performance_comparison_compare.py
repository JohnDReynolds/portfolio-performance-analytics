"""Tests for portfolio-level performance comparison findings."""

# Python imports
from pathlib import Path
import tempfile
from typing import cast
import unittest

# Third-party imports
import yaml

# Project imports
from ppar.errors import PpaError
from ppar.performance_comparison import (
    PerformanceComparison,
    PerformanceComparisonSpecification,
    findings_to_polars,
)
from ppar.performance_comparison import columns as pc_cols
from ppar.performance_comparison.findings import (
    DELTA_B_MINUS_A,
    FINDING_CODE,
    PC_CASH_MV,
    PC_PORT_MV,
    PC_PORT_RET,
    PC_POS_MV,
    PC_POS_QTY,
    PC_PRICE,
    PC_SEC_ADD,
    PC_SEC_CONTR,
    PC_SEC_DROP,
    PC_SEC_RET,
    PC_SEC_WGT,
    PC_TXN_ADD,
    PC_TXN_AMT,
    PC_TXN_DROP,
    PC_REF_CLASS,
    PC_REF_ID,
    SECURITY_ID,
    SOURCE_FILE,
    SOURCE_COLUMN,
)

_BASELINE_COMPARISON_PATH = Path("tests/data/axys/ppar_performance_comparison.yaml")
_RESTATEMENT_COMPARISON_PATH = Path(
    "tests/data/axys/ppar_performance_comparison_restatement.yaml"
)


def _write_transaction_fallback_specification(directory: Path) -> Path:
    """Write a minimal transaction comparison fixture without transaction ids."""
    for snapshot_name, amount in (("snapshot_a", "100.00"), ("snapshot_b", "110.00")):
        snapshot_path = directory / snapshot_name
        snapshot_path.mkdir()
        (snapshot_path / "portperf.csv").write_text(
            "PORTFOLIO_CODE,FROM_DATE,THRU_DATE,PORT_RETURN\n"
            "PORT_A,2025-05-01,2025-05-31,0.01\n",
            encoding="utf-8",
        )
        (snapshot_path / "transactions.csv").write_text(
            "PORT,SEC,TRADE_DATE,SETTLE_DATE,TRAN,QTY,PRICE,AMOUNT\n"
            f"PORT_A,AAPL,2025-05-15,2025-05-16,BUY,1,100.00,{amount}\n",
            encoding="utf-8",
        )

    specification = {
        "snapshots": {
            "a": {"path": "snapshot_a"},
            "b": {"path": "snapshot_b"},
        },
        "files": {
            "portfolio_performance": "portperf.csv",
            "transactions": "transactions.csv",
        },
    }
    specification_path = directory / "ppar_performance_comparison.yaml"
    specification_path.write_text(yaml.safe_dump(specification), encoding="utf-8")
    return specification_path


def _write_duplicate_portfolio_specification(directory: Path) -> Path:
    """Write a minimal comparison fixture with duplicate portfolio keys."""
    for snapshot_name in ("snapshot_a", "snapshot_b"):
        snapshot_path = directory / snapshot_name
        snapshot_path.mkdir()
        rows = [
            "PORTFOLIO_CODE,FROM_DATE,THRU_DATE,PORT_RETURN",
            "PORT_A,2025-05-01,2025-05-31,0.01",
        ]
        if snapshot_name == "snapshot_a":
            rows.append("PORT_A,2025-05-01,2025-05-31,0.02")
        (snapshot_path / "portperf.csv").write_text(
            "\n".join(rows) + "\n",
            encoding="utf-8",
        )

    specification = {
        "snapshots": {
            "a": {"path": "snapshot_a"},
            "b": {"path": "snapshot_b"},
        },
        "files": {"portfolio_performance": "portperf.csv"},
    }
    specification_path = directory / "ppar_performance_comparison.yaml"
    specification_path.write_text(yaml.safe_dump(specification), encoding="utf-8")
    return specification_path


class TestPerformanceComparison(unittest.TestCase):
    """Verify portfolio performance comparison findings."""

    def test_identical_baseline_snapshots_have_no_portfolio_findings(self) -> None:
        """The baseline fixture compares identical A/B snapshots."""
        specification = PerformanceComparisonSpecification(_BASELINE_COMPARISON_PATH)

        findings = PerformanceComparison(specification).compare_portfolio_performance()

        self.assertEqual(findings, [])

    def test_restatement_fixture_reports_portfolio_return_change(self) -> None:
        """The restatement fixture reports controlled portfolio-level changes."""
        specification = PerformanceComparisonSpecification(_RESTATEMENT_COMPARISON_PATH)

        findings = PerformanceComparison(specification).compare_portfolio_performance()
        finding_dicts = [finding.to_dict() for finding in findings]
        return_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_PORT_RET
            and finding[SOURCE_COLUMN] == pc_cols.PORTFOLIO_RETURN
        ]
        end_mv_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_PORT_MV
            and finding[SOURCE_COLUMN] == pc_cols.END_MARKET_VALUE
        ]

        self.assertEqual(len(return_findings), 1)
        self.assertEqual(return_findings[0][SOURCE_FILE], "portperf.csv")
        self.assertAlmostEqual(cast(float, return_findings[0][DELTA_B_MINUS_A]), 0.0005)
        self.assertEqual(len(end_mv_findings), 1)
        self.assertAlmostEqual(cast(float, end_mv_findings[0][DELTA_B_MINUS_A]), 500.0)

    def test_duplicate_portfolio_comparison_keys_raise_error_112(self) -> None:
        """Duplicate comparison keys are invalid because joins would multiply rows."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _write_duplicate_portfolio_specification(Path(temp_dir))
            specification = PerformanceComparisonSpecification(path)

            with self.assertRaises(PpaError) as context:
                PerformanceComparison(specification).compare_portfolio_performance()

            self.assertTrue(str(context.exception).startswith("Error 112"))
            self.assertIn("portfolio_performance", str(context.exception))
            self.assertIn("snapshot A", str(context.exception))

    def test_identical_baseline_snapshots_have_no_security_findings(self) -> None:
        """The baseline fixture compares identical security performance rows."""
        specification = PerformanceComparisonSpecification(_BASELINE_COMPARISON_PATH)

        findings = PerformanceComparison(specification).compare_security_performance()

        self.assertEqual(findings, [])

    def test_restatement_fixture_reports_security_changes(self) -> None:
        """The restatement fixture reports controlled security-level changes."""
        specification = PerformanceComparisonSpecification(_RESTATEMENT_COMPARISON_PATH)

        findings = PerformanceComparison(specification).compare_security_performance()
        finding_dicts = [finding.to_dict() for finding in findings]
        aapl_return_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_SEC_RET
            and finding[SECURITY_ID] == "AAPL"
            and finding[SOURCE_COLUMN] == pc_cols.SECURITY_RETURN
        ]
        aapl_weight_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_SEC_WGT
            and finding[SECURITY_ID] == "AAPL"
            and finding[SOURCE_COLUMN] == pc_cols.WEIGHT
        ]
        aapl_contribution_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_SEC_CONTR
            and finding[SECURITY_ID] == "AAPL"
            and finding[SOURCE_COLUMN] == pc_cols.CONTRIBUTION
        ]
        add_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_SEC_ADD
            and finding[SECURITY_ID] == "RESTATED_SEC"
        ]
        drop_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_SEC_DROP
            and finding[SECURITY_ID] == "PFE"
        ]

        self.assertEqual(len(aapl_return_findings), 1)
        self.assertAlmostEqual(
            cast(float, aapl_return_findings[0][DELTA_B_MINUS_A]),
            0.01,
        )
        self.assertEqual(len(aapl_weight_findings), 1)
        self.assertAlmostEqual(
            cast(float, aapl_weight_findings[0][DELTA_B_MINUS_A]),
            0.001,
        )
        self.assertEqual(len(aapl_contribution_findings), 1)
        self.assertAlmostEqual(
            cast(float, aapl_contribution_findings[0][DELTA_B_MINUS_A]),
            0.00058425,
        )
        self.assertEqual(len(add_findings), 1)
        self.assertEqual(len(drop_findings), 1)

    def test_compare_combines_portfolio_and_security_findings(self) -> None:
        """Combined comparison returns all currently supported finding groups."""
        specification = PerformanceComparisonSpecification(_RESTATEMENT_COMPARISON_PATH)

        finding_dicts = [
            finding.to_dict() for finding in PerformanceComparison(specification).compare()
        ]
        finding_codes = {finding[FINDING_CODE] for finding in finding_dicts}

        self.assertIn(PC_PORT_RET, finding_codes)
        self.assertIn(PC_SEC_RET, finding_codes)
        self.assertIn(PC_SEC_ADD, finding_codes)
        self.assertIn(PC_SEC_DROP, finding_codes)
        self.assertIn(PC_POS_QTY, finding_codes)
        self.assertIn(PC_CASH_MV, finding_codes)
        self.assertIn(PC_PRICE, finding_codes)
        self.assertIn(PC_TXN_AMT, finding_codes)

    def test_combined_findings_convert_to_polars(self) -> None:
        """Combined findings can be converted to a stable Polars table."""
        specification = PerformanceComparisonSpecification(_RESTATEMENT_COMPARISON_PATH)
        findings = PerformanceComparison(specification).compare()

        frame = findings_to_polars(findings)

        self.assertFalse(frame.is_empty())
        self.assertIn(FINDING_CODE, frame.columns)
        self.assertIn(DELTA_B_MINUS_A, frame.columns)
        self.assertIn(SOURCE_FILE, frame.columns)

    def test_baseline_combined_compare_has_empty_polars_output(self) -> None:
        """Identical baseline snapshots produce an empty stable finding table."""
        specification = PerformanceComparisonSpecification(_BASELINE_COMPARISON_PATH)
        findings = PerformanceComparison(specification).compare()

        frame = findings_to_polars(findings)

        self.assertTrue(frame.is_empty())
        self.assertIn(FINDING_CODE, frame.columns)
        self.assertIn(SOURCE_FILE, frame.columns)

    def test_identical_baseline_snapshots_have_no_security_master_findings(self) -> None:
        """The baseline fixture compares identical security master rows."""
        specification = PerformanceComparisonSpecification(_BASELINE_COMPARISON_PATH)

        findings = PerformanceComparison(specification).compare_security_master()

        self.assertEqual(findings, [])

    def test_restatement_fixture_reports_security_master_changes(self) -> None:
        """The restatement fixture reports controlled security master changes."""
        specification = PerformanceComparisonSpecification(_RESTATEMENT_COMPARISON_PATH)

        findings = PerformanceComparison(specification).compare_security_master()
        finding_dicts = [finding.to_dict() for finding in findings]
        aapl_name_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_REF_ID
            and finding[SECURITY_ID] == "AAPL"
            and finding[SOURCE_COLUMN] == pc_cols.SECURITY_NAME
        ]
        aapl_sector_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_REF_CLASS
            and finding[SECURITY_ID] == "AAPL"
            and finding[SOURCE_COLUMN] == pc_cols.SECTOR
        ]
        added_reference_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == "PC-ROW-ADD"
            and finding[SECURITY_ID] == "RESTATED_SEC"
        ]

        self.assertEqual(len(aapl_name_findings), 1)
        self.assertEqual(
            aapl_name_findings[0]["snapshot_b_value"],
            "Apple Inc Restated Name",
        )
        self.assertEqual(len(aapl_sector_findings), 1)
        self.assertEqual(aapl_sector_findings[0]["snapshot_b_value"], "TECH_RESTATED")
        self.assertEqual(len(added_reference_findings), 1)

    def test_identical_baseline_snapshots_have_no_position_findings(self) -> None:
        """The baseline fixture compares identical position rows."""
        specification = PerformanceComparisonSpecification(_BASELINE_COMPARISON_PATH)

        findings = PerformanceComparison(specification).compare_positions()

        self.assertEqual(findings, [])

    def test_restatement_fixture_reports_position_changes(self) -> None:
        """The restatement fixture reports controlled position-level changes."""
        specification = PerformanceComparisonSpecification(_RESTATEMENT_COMPARISON_PATH)

        findings = PerformanceComparison(specification).compare_positions()
        finding_dicts = [finding.to_dict() for finding in findings]
        quantity_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_POS_QTY
            and finding[SECURITY_ID] == "AAPL"
            and finding[SOURCE_COLUMN] == pc_cols.QUANTITY
        ]
        market_value_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_POS_MV
            and finding[SECURITY_ID] == "AAPL"
            and finding[SOURCE_COLUMN] == pc_cols.MARKET_VALUE
        ]

        self.assertEqual(len(quantity_findings), 1)
        self.assertAlmostEqual(
            cast(float, quantity_findings[0][DELTA_B_MINUS_A]),
            10.0,
        )
        self.assertEqual(len(market_value_findings), 1)
        self.assertAlmostEqual(
            cast(float, market_value_findings[0][DELTA_B_MINUS_A]),
            2648.56,
        )

    def test_identical_baseline_snapshots_have_no_cash_findings(self) -> None:
        """The baseline fixture compares identical cash rows."""
        specification = PerformanceComparisonSpecification(_BASELINE_COMPARISON_PATH)

        findings = PerformanceComparison(specification).compare_cash()

        self.assertEqual(findings, [])

    def test_restatement_fixture_reports_cash_changes(self) -> None:
        """The restatement fixture reports controlled cash-level changes."""
        specification = PerformanceComparisonSpecification(_RESTATEMENT_COMPARISON_PATH)

        findings = PerformanceComparison(specification).compare_cash()
        finding_dicts = [finding.to_dict() for finding in findings]
        cash_balance_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_CASH_MV
            and finding[SOURCE_COLUMN] == pc_cols.CASH_BALANCE
        ]
        market_value_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_CASH_MV
            and finding[SOURCE_COLUMN] == pc_cols.MARKET_VALUE
        ]

        self.assertEqual(len(cash_balance_findings), 1)
        self.assertAlmostEqual(
            cast(float, cash_balance_findings[0][DELTA_B_MINUS_A]),
            500.0,
        )
        self.assertEqual(len(market_value_findings), 1)
        self.assertAlmostEqual(
            cast(float, market_value_findings[0][DELTA_B_MINUS_A]),
            500.0,
        )

    def test_identical_baseline_snapshots_have_no_price_findings(self) -> None:
        """The baseline fixture compares identical price rows."""
        specification = PerformanceComparisonSpecification(_BASELINE_COMPARISON_PATH)

        findings = PerformanceComparison(specification).compare_prices()

        self.assertEqual(findings, [])

    def test_restatement_fixture_reports_price_changes(self) -> None:
        """The restatement fixture reports controlled price changes."""
        specification = PerformanceComparisonSpecification(_RESTATEMENT_COMPARISON_PATH)

        findings = PerformanceComparison(specification).compare_prices()
        finding_dicts = [finding.to_dict() for finding in findings]
        price_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_PRICE
            and finding[SECURITY_ID] == "AAPL"
            and finding[SOURCE_COLUMN] == pc_cols.PRICE
        ]

        self.assertEqual(len(price_findings), 1)
        self.assertAlmostEqual(
            cast(float, price_findings[0][DELTA_B_MINUS_A]),
            1.0,
        )

    def test_identical_baseline_snapshots_have_no_transaction_findings(self) -> None:
        """The baseline fixture compares identical transaction rows."""
        specification = PerformanceComparisonSpecification(_BASELINE_COMPARISON_PATH)

        findings = PerformanceComparison(specification).compare_transactions()

        self.assertEqual(findings, [])

    def test_restatement_fixture_reports_transaction_amount_change(self) -> None:
        """The restatement fixture reports controlled transaction amount changes."""
        specification = PerformanceComparisonSpecification(_RESTATEMENT_COMPARISON_PATH)

        findings = PerformanceComparison(specification).compare_transactions()
        finding_dicts = [finding.to_dict() for finding in findings]
        amount_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_TXN_AMT
            and finding[SECURITY_ID] == "AAPL"
            and finding[SOURCE_COLUMN] == pc_cols.AMOUNT
        ]

        self.assertEqual(len(amount_findings), 1)
        self.assertAlmostEqual(
            cast(float, amount_findings[0][DELTA_B_MINUS_A]),
            -100.0,
        )

    def test_transaction_fallback_key_treats_amount_change_as_add_drop(self) -> None:
        """Transaction amount changes require a stable transaction id."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_transaction_fallback_specification(Path(temp_dir))
            specification = PerformanceComparisonSpecification(specification_path)

            findings = PerformanceComparison(specification).compare_transactions()
            finding_codes = [finding.to_dict()[FINDING_CODE] for finding in findings]

            self.assertEqual(finding_codes.count(PC_TXN_ADD), 1)
            self.assertEqual(finding_codes.count(PC_TXN_DROP), 1)
            self.assertNotIn(PC_TXN_AMT, finding_codes)


if __name__ == "__main__":
    unittest.main()
