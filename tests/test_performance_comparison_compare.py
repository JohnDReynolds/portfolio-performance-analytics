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
    CASH_FLOW_SIGN,
    CONTEXT,
    DELTA_B_MINUS_A,
    DIRECT_INPUT,
    EVIDENCE_ROLE,
    FINDING_CODE,
    FROM_DATE,
    PORTFOLIO_ID,
    PC_CASH_MV,
    PC_FX_RATE,
    PC_PORT_MV,
    PC_POS_ACCR,
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
    PC_TXN_PRICE,
    PC_TXN_QTY,
    PC_REF_CLASS,
    PC_REF_ID,
    RELATED_OUTPUT,
    RETURN_DENOMINATOR,
    SECURITY_ID,
    SOURCE_FILE,
    SOURCE_COLUMN,
    TARGET_OUTPUT,
    THRU_DATE,
    TRANSACTION_CATEGORY,
    PERFORMANCE_FLOW_SIGN,
    Finding,
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


def _write_transaction_period_specification(directory: Path) -> Path:
    """Write a minimal transaction comparison fixture with transaction ids."""
    for snapshot_name, amount in (("snapshot_a", "100.00"), ("snapshot_b", "110.00")):
        snapshot_path = directory / snapshot_name
        snapshot_path.mkdir()
        (snapshot_path / "portperf.csv").write_text(
            "PORTFOLIO_CODE,FROM_DATE,THRU_DATE,PORT_RETURN\n"
            "PORT_A,2025-05-01,2025-05-31,0.01\n",
            encoding="utf-8",
        )
        (snapshot_path / "transactions.csv").write_text(
            "TRANSACTION_ID,PORT,SEC,TRADE_DATE,SETTLE_DATE,TRAN,QTY,PRICE,"
            "AMOUNT,CASH_FLOW_SIGN,PERFORMANCE_FLOW_SIGN\n"
            f"TXN1,PORT_A,AAPL,2025-05-15,2025-05-16,BUY,1,100.00,{amount},"
            "cash out,external\n",
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


def _write_transaction_outside_period_specification(directory: Path) -> Path:
    """Write a minimal transaction fixture whose trade date is outside period."""
    for snapshot_name, amount in (("snapshot_a", "100.00"), ("snapshot_b", "110.00")):
        snapshot_path = directory / snapshot_name
        snapshot_path.mkdir()
        (snapshot_path / "portperf.csv").write_text(
            "PORTFOLIO_CODE,FROM_DATE,THRU_DATE,BEGIN_MV,PORT_RETURN\n"
            "PORT_A,2025-05-01,2025-05-31,1000.00,0.01\n",
            encoding="utf-8",
        )
        (snapshot_path / "transactions.csv").write_text(
            "TRANSACTION_ID,PORT,SEC,TRADE_DATE,SETTLE_DATE,TRAN,QTY,PRICE,"
            "AMOUNT,CASH_FLOW_SIGN,PERFORMANCE_FLOW_SIGN\n"
            f"TXN1,PORT_A,AAPL,2025-06-15,2025-06-16,BUY,1,100.00,{amount},"
            "cash out,performance\n",
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


def _write_position_period_specification(directory: Path) -> Path:
    """Write a minimal position comparison fixture with a containing period."""
    for snapshot_name, market_value in (
        ("snapshot_a", "1000.00"),
        ("snapshot_b", "1010.00"),
    ):
        snapshot_path = directory / snapshot_name
        snapshot_path.mkdir()
        (snapshot_path / "portperf.csv").write_text(
            "PORTFOLIO_CODE,FROM_DATE,THRU_DATE,PORT_RETURN\n"
            "PORT_A,2025-05-01,2025-05-31,0.01\n",
            encoding="utf-8",
        )
        (snapshot_path / "positions.csv").write_text(
            "PORT,SEC,POSITION_DATE,QTY,MKT_VAL\n"
            f"PORT_A,AAPL,2025-05-31,10,{market_value}\n",
            encoding="utf-8",
        )

    specification = {
        "snapshots": {
            "a": {"path": "snapshot_a"},
            "b": {"path": "snapshot_b"},
        },
        "files": {
            "portfolio_performance": "portperf.csv",
            "positions": "positions.csv",
        },
    }
    specification_path = directory / "ppar_performance_comparison.yaml"
    specification_path.write_text(yaml.safe_dump(specification), encoding="utf-8")
    return specification_path


def _write_cash_period_specification(directory: Path) -> Path:
    """Write a minimal cash comparison fixture with a containing period."""
    for snapshot_name, cash_balance in (
        ("snapshot_a", "1000.00"),
        ("snapshot_b", "1010.00"),
    ):
        snapshot_path = directory / snapshot_name
        snapshot_path.mkdir()
        (snapshot_path / "portperf.csv").write_text(
            "PORTFOLIO_CODE,FROM_DATE,THRU_DATE,PORT_RETURN\n"
            "PORT_A,2025-05-01,2025-05-31,0.01\n",
            encoding="utf-8",
        )
        (snapshot_path / "cash.csv").write_text(
            "PORT,CASH_DATE,CURRENCY,CASH_BALANCE\n"
            f"PORT_A,2025-05-31,USD,{cash_balance}\n",
            encoding="utf-8",
        )

    specification = {
        "snapshots": {
            "a": {"path": "snapshot_a"},
            "b": {"path": "snapshot_b"},
        },
        "files": {
            "portfolio_performance": "portperf.csv",
            "cash": "cash.csv",
        },
    }
    specification_path = directory / "ppar_performance_comparison.yaml"
    specification_path.write_text(yaml.safe_dump(specification), encoding="utf-8")
    return specification_path


def _write_multi_portfolio_price_specification(directory: Path) -> Path:
    """Write a fixture where one price change affects two portfolio periods."""
    for snapshot_name, price in (("snapshot_a", "100.00"), ("snapshot_b", "101.00")):
        snapshot_path = directory / snapshot_name
        snapshot_path.mkdir()
        (snapshot_path / "portperf.csv").write_text(
            "PORTFOLIO_CODE,FROM_DATE,THRU_DATE,PORT_RETURN\n"
            "PORT_A,2025-05-01,2025-05-31,0.01\n"
            "PORT_B,2025-05-01,2025-05-31,0.02\n",
            encoding="utf-8",
        )
        (snapshot_path / "secperf.csv").write_text(
            "PORTFOLIO_CODE,SEC,FROM_DATE,THRU_DATE,SEC_RETURN\n"
            "PORT_A,AAPL,2025-05-01,2025-05-31,0.01\n"
            "PORT_B,AAPL,2025-05-01,2025-05-31,0.02\n",
            encoding="utf-8",
        )
        (snapshot_path / "prices.csv").write_text(
            "SEC,PRICE_DATE,PRICE\n"
            f"AAPL,2025-05-31,{price}\n",
            encoding="utf-8",
        )

    specification = {
        "snapshots": {
            "a": {"path": "snapshot_a"},
            "b": {"path": "snapshot_b"},
        },
        "files": {
            "portfolio_performance": "portperf.csv",
            "security_performance": "secperf.csv",
            "prices": "prices.csv",
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

    _baseline_specification: PerformanceComparisonSpecification
    _restatement_specification: PerformanceComparisonSpecification
    _baseline_combined_findings: list[Finding]
    _restatement_combined_findings: list[Finding]
    _baseline_portfolio_findings: list[Finding]
    _restatement_portfolio_findings: list[Finding]
    _baseline_security_findings: list[Finding]
    _restatement_security_findings: list[Finding]
    _baseline_security_master_findings: list[Finding]
    _restatement_security_master_findings: list[Finding]
    _baseline_position_findings: list[Finding]
    _restatement_position_findings: list[Finding]
    _baseline_cash_findings: list[Finding]
    _restatement_cash_findings: list[Finding]
    _baseline_price_findings: list[Finding]
    _restatement_price_findings: list[Finding]
    _baseline_fx_rate_findings: list[Finding]
    _restatement_fx_rate_findings: list[Finding]
    _baseline_transaction_findings: list[Finding]
    _restatement_transaction_findings: list[Finding]

    @classmethod
    def setUpClass(cls) -> None:
        """Cache shared fixture comparisons for the class."""
        cls._baseline_specification = PerformanceComparisonSpecification(
            _BASELINE_COMPARISON_PATH
        )
        cls._restatement_specification = PerformanceComparisonSpecification(
            _RESTATEMENT_COMPARISON_PATH
        )
        baseline = PerformanceComparison(cls._baseline_specification)
        restatement = PerformanceComparison(cls._restatement_specification)

        cls._baseline_combined_findings = baseline.compare()
        cls._restatement_combined_findings = restatement.compare()
        cls._baseline_portfolio_findings = baseline.compare_portfolio_performance()
        cls._restatement_portfolio_findings = restatement.compare_portfolio_performance()
        cls._baseline_security_findings = baseline.compare_security_performance()
        cls._restatement_security_findings = restatement.compare_security_performance()
        cls._baseline_security_master_findings = baseline.compare_security_master()
        cls._restatement_security_master_findings = restatement.compare_security_master()
        cls._baseline_position_findings = baseline.compare_positions()
        cls._restatement_position_findings = restatement.compare_positions()
        cls._baseline_cash_findings = baseline.compare_cash()
        cls._restatement_cash_findings = restatement.compare_cash()
        cls._baseline_price_findings = baseline.compare_prices()
        cls._restatement_price_findings = restatement.compare_prices()
        cls._baseline_fx_rate_findings = baseline.compare_fx_rates()
        cls._restatement_fx_rate_findings = restatement.compare_fx_rates()
        cls._baseline_transaction_findings = baseline.compare_transactions()
        cls._restatement_transaction_findings = restatement.compare_transactions()

    def test_identical_baseline_snapshots_have_no_portfolio_findings(self) -> None:
        """The baseline fixture compares identical A/B snapshots."""
        findings = list(self._baseline_portfolio_findings)

        self.assertEqual(findings, [])

    def test_restatement_fixture_reports_portfolio_return_change(self) -> None:
        """The restatement fixture reports controlled portfolio-level changes."""
        findings = list(self._restatement_portfolio_findings)
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
        findings = list(self._baseline_security_findings)

        self.assertEqual(findings, [])

    def test_restatement_fixture_reports_security_changes(self) -> None:
        """The restatement fixture reports controlled security-level changes."""
        findings = list(self._restatement_security_findings)
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
        finding_dicts = [
            finding.to_dict() for finding in self._restatement_combined_findings
        ]
        finding_codes = {finding[FINDING_CODE] for finding in finding_dicts}

        self.assertIn(PC_PORT_RET, finding_codes)
        self.assertIn(PC_SEC_RET, finding_codes)
        self.assertIn(PC_SEC_ADD, finding_codes)
        self.assertIn(PC_SEC_DROP, finding_codes)
        self.assertIn(PC_POS_QTY, finding_codes)
        self.assertIn(PC_POS_ACCR, finding_codes)
        self.assertIn(PC_CASH_MV, finding_codes)
        self.assertIn(PC_PRICE, finding_codes)
        self.assertIn(PC_FX_RATE, finding_codes)
        self.assertIn(PC_TXN_AMT, finding_codes)
        self.assertIn(PC_TXN_QTY, finding_codes)
        self.assertIn(PC_TXN_PRICE, finding_codes)

    def test_combined_findings_convert_to_polars(self) -> None:
        """Combined findings can be converted to a stable Polars table."""
        findings = list(self._restatement_combined_findings)

        frame = findings_to_polars(findings)

        self.assertFalse(frame.is_empty())
        self.assertIn(FINDING_CODE, frame.columns)
        self.assertIn(EVIDENCE_ROLE, frame.columns)
        self.assertIn(DELTA_B_MINUS_A, frame.columns)
        self.assertIn(SOURCE_FILE, frame.columns)

    def test_restatement_findings_have_explanation_roles(self) -> None:
        """Comparison assigns explicit evidence roles to finding families."""
        findings = list(self._restatement_combined_findings)
        frame = findings_to_polars(findings)

        role_by_code = {
            row[FINDING_CODE]: row[EVIDENCE_ROLE]
            for row in frame.select(FINDING_CODE, EVIDENCE_ROLE).iter_rows(named=True)
        }

        self.assertEqual(role_by_code[PC_PORT_RET], TARGET_OUTPUT)
        self.assertEqual(role_by_code[PC_PORT_MV], DIRECT_INPUT)
        self.assertEqual(role_by_code[PC_SEC_RET], RELATED_OUTPUT)
        self.assertEqual(role_by_code[PC_POS_QTY], DIRECT_INPUT)
        self.assertEqual(role_by_code[PC_CASH_MV], DIRECT_INPUT)
        self.assertEqual(role_by_code[PC_PRICE], DIRECT_INPUT)
        self.assertEqual(role_by_code[PC_FX_RATE], DIRECT_INPUT)
        self.assertEqual(role_by_code[PC_TXN_AMT], DIRECT_INPUT)
        self.assertEqual(role_by_code[PC_REF_ID], CONTEXT)

    def test_baseline_combined_compare_has_empty_polars_output(self) -> None:
        """Identical baseline snapshots produce an empty stable finding table."""
        findings = list(self._baseline_combined_findings)

        frame = findings_to_polars(findings)

        self.assertTrue(frame.is_empty())
        self.assertIn(FINDING_CODE, frame.columns)
        self.assertIn(EVIDENCE_ROLE, frame.columns)
        self.assertIn(SOURCE_FILE, frame.columns)

    def test_identical_baseline_snapshots_have_no_security_master_findings(self) -> None:
        """The baseline fixture compares identical security master rows."""
        findings = list(self._baseline_security_master_findings)

        self.assertEqual(findings, [])

    def test_restatement_fixture_reports_security_master_changes(self) -> None:
        """The restatement fixture reports controlled security master changes."""
        findings = list(self._restatement_security_master_findings)
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
        findings = list(self._baseline_position_findings)

        self.assertEqual(findings, [])

    def test_restatement_fixture_reports_position_changes(self) -> None:
        """The restatement fixture reports controlled position-level changes."""
        findings = list(self._restatement_position_findings)
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
        accrued_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_POS_ACCR
            and finding[SECURITY_ID] == "AAPL"
            and finding[SOURCE_COLUMN] == pc_cols.ACCRUED
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
        self.assertEqual(len(accrued_findings), 1)
        self.assertAlmostEqual(
            cast(float, accrued_findings[0][DELTA_B_MINUS_A]),
            6.25,
        )

    def test_position_changes_link_to_containing_portfolio_period(self) -> None:
        """Changed position rows inherit the containing portfolio period."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_position_period_specification(Path(temp_dir))
            specification = PerformanceComparisonSpecification(specification_path)

            findings = PerformanceComparison(specification).compare_positions()
            finding_dicts = [finding.to_dict() for finding in findings]
            market_value_finding = next(
                finding
                for finding in finding_dicts
                if finding[FINDING_CODE] == PC_POS_MV
            )

            self.assertEqual(str(market_value_finding[FROM_DATE]), "2025-05-01")
            self.assertEqual(str(market_value_finding[THRU_DATE]), "2025-05-31")

    def test_identical_baseline_snapshots_have_no_cash_findings(self) -> None:
        """The baseline fixture compares identical cash rows."""
        findings = list(self._baseline_cash_findings)

        self.assertEqual(findings, [])

    def test_restatement_fixture_reports_cash_changes(self) -> None:
        """The restatement fixture reports controlled cash-level changes."""
        findings = list(self._restatement_cash_findings)
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

    def test_cash_changes_link_to_containing_portfolio_period(self) -> None:
        """Changed cash rows inherit the containing portfolio period."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_cash_period_specification(Path(temp_dir))
            specification = PerformanceComparisonSpecification(specification_path)

            findings = PerformanceComparison(specification).compare_cash()
            finding_dicts = [finding.to_dict() for finding in findings]
            cash_balance_finding = next(
                finding
                for finding in finding_dicts
                if finding[FINDING_CODE] == PC_CASH_MV
            )

            self.assertEqual(str(cash_balance_finding[FROM_DATE]), "2025-05-01")
            self.assertEqual(str(cash_balance_finding[THRU_DATE]), "2025-05-31")

    def test_identical_baseline_snapshots_have_no_price_findings(self) -> None:
        """The baseline fixture compares identical price rows."""
        findings = list(self._baseline_price_findings)

        self.assertEqual(findings, [])

    def test_restatement_fixture_reports_price_changes(self) -> None:
        """The restatement fixture reports controlled price changes."""
        findings = list(self._restatement_price_findings)
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
        self.assertEqual(price_findings[0][PORTFOLIO_ID], "PORT_A")
        self.assertEqual(str(price_findings[0][FROM_DATE]), "2025-05-30")
        self.assertEqual(str(price_findings[0][THRU_DATE]), "2025-05-30")

    def test_price_changes_expand_to_matching_portfolio_periods(self) -> None:
        """One security price change links to every matching portfolio period."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _write_multi_portfolio_price_specification(Path(temp_dir))
            specification = PerformanceComparisonSpecification(path)

            findings = PerformanceComparison(specification).compare_prices()
            price_findings = [
                finding.to_dict()
                for finding in findings
                if finding.code == PC_PRICE
            ]
            contexts = sorted(
                (
                    finding[PORTFOLIO_ID],
                    str(finding[FROM_DATE]),
                    str(finding[THRU_DATE]),
                    finding[SECURITY_ID],
                    finding[DELTA_B_MINUS_A],
                )
                for finding in price_findings
            )

            self.assertEqual(
                contexts,
                [
                    ("PORT_A", "2025-05-01", "2025-05-31", "AAPL", 1.0),
                    ("PORT_B", "2025-05-01", "2025-05-31", "AAPL", 1.0),
                ],
            )

    def test_identical_baseline_snapshots_have_no_fx_rate_findings(self) -> None:
        """The baseline fixture compares identical FX rate rows."""
        findings = list(self._baseline_fx_rate_findings)

        self.assertEqual(findings, [])

    def test_restatement_fixture_reports_fx_rate_changes(self) -> None:
        """The restatement fixture reports controlled FX rate changes."""
        findings = list(self._restatement_fx_rate_findings)
        finding_dicts = [finding.to_dict() for finding in findings]
        fx_rate_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_FX_RATE
            and finding[SOURCE_COLUMN] == pc_cols.FX_RATE
        ]

        self.assertEqual(len(fx_rate_findings), 1)
        self.assertAlmostEqual(
            cast(float, fx_rate_findings[0][DELTA_B_MINUS_A]),
            0.005,
        )

    def test_identical_baseline_snapshots_have_no_transaction_findings(self) -> None:
        """The baseline fixture compares identical transaction rows."""
        findings = list(self._baseline_transaction_findings)

        self.assertEqual(findings, [])

    def test_restatement_fixture_reports_transaction_changes(self) -> None:
        """The restatement fixture reports controlled transaction changes."""
        findings = list(self._restatement_transaction_findings)
        finding_dicts = [finding.to_dict() for finding in findings]
        amount_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_TXN_AMT
            and finding[SECURITY_ID] == "AAPL"
            and finding[SOURCE_COLUMN] == pc_cols.AMOUNT
        ]
        quantity_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_TXN_QTY
            and finding[SECURITY_ID] == "AAPL"
            and finding[SOURCE_COLUMN] == pc_cols.QUANTITY
        ]
        price_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_TXN_PRICE
            and finding[SECURITY_ID] == "AAPL"
            and finding[SOURCE_COLUMN] == pc_cols.PRICE
        ]

        self.assertEqual(len(amount_findings), 1)
        self.assertEqual(amount_findings[0][TRANSACTION_CATEGORY], "buy")
        self.assertIsNone(amount_findings[0][FROM_DATE])
        self.assertIsNone(amount_findings[0][THRU_DATE])
        self.assertAlmostEqual(
            cast(float, amount_findings[0][DELTA_B_MINUS_A]),
            -100.0,
        )
        self.assertEqual(len(quantity_findings), 1)
        self.assertEqual(quantity_findings[0][TRANSACTION_CATEGORY], "buy")
        self.assertAlmostEqual(
            cast(float, quantity_findings[0][DELTA_B_MINUS_A]),
            1.0,
        )
        self.assertEqual(len(price_findings), 1)
        self.assertEqual(price_findings[0][TRANSACTION_CATEGORY], "buy")
        self.assertAlmostEqual(
            cast(float, price_findings[0][DELTA_B_MINUS_A]),
            0.5,
        )

    def test_transaction_changes_link_to_containing_portfolio_period(self) -> None:
        """Changed transaction rows inherit the containing portfolio period."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_transaction_period_specification(Path(temp_dir))
            specification = PerformanceComparisonSpecification(specification_path)

            findings = PerformanceComparison(specification).compare_transactions()
            finding_dicts = [finding.to_dict() for finding in findings]
            amount_finding = next(
                finding
                for finding in finding_dicts
                if finding[FINDING_CODE] == PC_TXN_AMT
            )

            self.assertEqual(str(amount_finding[FROM_DATE]), "2025-05-01")
            self.assertEqual(str(amount_finding[THRU_DATE]), "2025-05-31")
            self.assertEqual(amount_finding[TRANSACTION_CATEGORY], "buy")
            self.assertEqual(amount_finding[CASH_FLOW_SIGN], "negative")
            self.assertEqual(amount_finding[PERFORMANCE_FLOW_SIGN], "external")

    def test_transaction_outside_period_does_not_get_denominator(self) -> None:
        """Out-of-period transaction rows do not inherit a return denominator."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_transaction_outside_period_specification(
                Path(temp_dir)
            )
            specification = PerformanceComparisonSpecification(specification_path)

            findings = PerformanceComparison(specification).compare_transactions()
            finding_dicts = [finding.to_dict() for finding in findings]
            amount_finding = next(
                finding
                for finding in finding_dicts
                if finding[FINDING_CODE] == PC_TXN_AMT
            )

            self.assertIsNone(amount_finding[FROM_DATE])
            self.assertIsNone(amount_finding[THRU_DATE])
            self.assertIsNone(amount_finding[RETURN_DENOMINATOR])
            self.assertEqual(amount_finding[CASH_FLOW_SIGN], "negative")
            self.assertEqual(amount_finding[PERFORMANCE_FLOW_SIGN], "performance")

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
