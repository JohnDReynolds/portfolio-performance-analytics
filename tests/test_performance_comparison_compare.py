"""Tests for portfolio-level performance comparison findings."""

# Python imports
import datetime as dt
from pathlib import Path
import tempfile
from typing import cast
import unittest

# Third-party imports
import polars as pl
import yaml

# Project imports
from ppar.errors import PpaError
from ppar.performance_comparison import (
    PerformanceComparison,
    PerformanceComparisonSpecification,
    findings_to_polars,
)
from ppar.performance_comparison import columns as pc_cols
from ppar.performance_comparison.compare import (
    _modified_dietz_external_flow_eligibility,
    _transaction_impact_policies,
    _validated_modified_dietz_policy,
)
from ppar.performance_comparison.explain import (
    ESTIMATED_RETURN_IMPACT,
    portfolio_period_contribution_candidates,
)
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
    TRANSACTION_IMPACT_DIAGNOSTIC,
    TRANSACTION_IMPACT_DIAGNOSTIC_ESTIMATE,
    TRANSACTION_IMPACT_POLICY,
    TRANSACTION_IMPACT_POLICY_EXTERNAL_FLOW_EVIDENCE_ONLY,
    TRANSACTION_IMPACT_POLICY_PERFORMANCE_AMOUNT_DELTA,
    TRANSACTION_CATEGORY,
    TRANSACTION_MATCH_STATUS,
    TRANSACTION_MATCH_STATUS_ID_MATCH,
    TRANSACTION_MATCH_STATUS_STRICT_FALLBACK_UNMATCHED,
    TRANSACTION_SEMANTICS_SOURCE,
    PERFORMANCE_FLOW_SIGN,
    Finding,
)

_BASELINE_COMPARISON_PATH = Path("tests/data/axys/ppar_performance_comparison.yaml")
_RESTATEMENT_COMPARISON_PATH = Path(
    "tests/data/axys/ppar_performance_comparison_restatement.yaml"
)
_RESTATEMENT_TRANSACTION_RULES_PATH = Path(
    "tests/data/axys/ppar_performance_comparison_restatement_transaction_rules.yaml"
)


def _write_transaction_fallback_specification(directory: Path) -> Path:
    """Write a minimal transaction comparison fixture without transaction ids."""
    for snapshot_name, amount in (("snapshot_a", "100.00"), ("snapshot_b", "110.00")):
        snapshot_path = directory / snapshot_name
        snapshot_path.mkdir()
        (snapshot_path / "portperf.csv").write_text(
            "PORTFOLIO_CODE,FROM_DATE,THRU_DATE,BEGIN_MV,PORT_RETURN\n"
            "PORT_A,2025-05-01,2025-05-31,1000.00,0.01\n",
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
            "PORTFOLIO_CODE,FROM_DATE,THRU_DATE,BEGIN_MV,PORT_RETURN\n"
            "PORT_A,2025-05-01,2025-05-31,1000.00,0.01\n",
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
        self.assertEqual(
            amount_findings[0][TRANSACTION_MATCH_STATUS],
            TRANSACTION_MATCH_STATUS_ID_MATCH,
        )
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

    def test_restatement_transaction_rules_fixture_carries_yaml_semantics(
        self,
    ) -> None:
        """YAML transaction rules fill sign/flow semantics in Axys findings."""
        specification = PerformanceComparisonSpecification(
            _RESTATEMENT_TRANSACTION_RULES_PATH
        )

        findings = PerformanceComparison(specification).compare_transactions()
        changed_fields = {
            finding.to_dict()[SOURCE_COLUMN]: finding.to_dict()
            for finding in findings
            if finding.to_dict()[SECURITY_ID] == "AAPL"
        }

        self.assertEqual(
            set(changed_fields),
            {pc_cols.AMOUNT, pc_cols.QUANTITY, pc_cols.PRICE},
        )
        for finding in changed_fields.values():
            self.assertEqual(finding[TRANSACTION_CATEGORY], "buy")
            self.assertEqual(finding[CASH_FLOW_SIGN], "negative")
            self.assertEqual(finding[PERFORMANCE_FLOW_SIGN], "performance")
            self.assertEqual(finding[TRANSACTION_SEMANTICS_SOURCE], "mixed")

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
            self.assertEqual(amount_finding[TRANSACTION_SEMANTICS_SOURCE], "source")

    def test_transaction_external_flow_policy_is_loaded_from_yaml(self) -> None:
        """Explicit external-flow impact policy is carried into findings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_transaction_period_specification(Path(temp_dir))
            configuration = yaml.safe_load(specification_path.read_text(encoding="utf-8"))
            configuration["transaction_impact_methods"] = {
                "external_flow": {"method": "evidence_only"}
            }
            specification_path.write_text(
                yaml.safe_dump(configuration),
                encoding="utf-8",
            )
            specification = PerformanceComparisonSpecification(specification_path)

            findings = PerformanceComparison(specification).compare_transactions()
            amount_finding = next(
                finding.to_dict()
                for finding in findings
                if finding.to_dict()[FINDING_CODE] == PC_TXN_AMT
            )

            self.assertEqual(
                amount_finding[TRANSACTION_IMPACT_POLICY],
                TRANSACTION_IMPACT_POLICY_EXTERNAL_FLOW_EVIDENCE_ONLY,
            )
            self.assertEqual(
                amount_finding[TRANSACTION_IMPACT_DIAGNOSTIC],
                "external-flow evidence-only policy",
            )
            policies = _transaction_impact_policies(specification)
            external_flow_policy = policies["external_flow"]
            self.assertEqual(external_flow_policy.method, "evidence_only")
            self.assertEqual(
                external_flow_policy.finding_label,
                TRANSACTION_IMPACT_POLICY_EXTERNAL_FLOW_EVIDENCE_ONLY,
            )
            self.assertIsNone(external_flow_policy.flow_timing)

    def test_transaction_performance_policy_is_loaded_from_yaml(self) -> None:
        """Explicit performance amount impact policy is carried into findings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_transaction_period_specification(Path(temp_dir))
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                transaction_path = Path(temp_dir) / snapshot_name / "transactions.csv"
                transaction_path.write_text(
                    transaction_path.read_text(encoding="utf-8").replace(
                        "cash out,external",
                        "cash out,performance",
                    ),
                    encoding="utf-8",
                )
            configuration = yaml.safe_load(specification_path.read_text(encoding="utf-8"))
            configuration["transaction_impact_methods"] = {
                "performance": {
                    "method": "transaction_amount_delta_over_return_denominator",
                    "denominator_source": "begin_market_value",
                },
            }
            specification_path.write_text(
                yaml.safe_dump(configuration),
                encoding="utf-8",
            )
            specification = PerformanceComparisonSpecification(specification_path)

            findings = PerformanceComparison(specification).compare_transactions()
            amount_finding = next(
                finding.to_dict()
                for finding in findings
                if finding.to_dict()[FINDING_CODE] == PC_TXN_AMT
            )

            self.assertEqual(
                amount_finding[TRANSACTION_IMPACT_POLICY],
                TRANSACTION_IMPACT_POLICY_PERFORMANCE_AMOUNT_DELTA,
            )
            policies = _transaction_impact_policies(specification)
            performance_policy = policies["performance"]
            self.assertEqual(
                performance_policy.method,
                "transaction_amount_delta_over_return_denominator",
            )
            self.assertEqual(
                performance_policy.finding_label,
                TRANSACTION_IMPACT_POLICY_PERFORMANCE_AMOUNT_DELTA,
            )
            self.assertEqual(performance_policy.denominator_source, "begin_market_value")

    def test_transaction_modified_dietz_policy_preserves_explicit_yaml_fields(
        self,
    ) -> None:
        """Modified Dietz policy keeps every explicit YAML convention."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_transaction_period_specification(Path(temp_dir))
            specification = PerformanceComparisonSpecification(specification_path)
            external_flow_value = {
                "method": "modified_dietz",
                "flow_timing": "settlement_date",
                "day_count": "actual_days",
                "inclusion_rule": "end_of_day",
                "denominator_source": "begin_market_value",
                "double_count_policy": "cross_check_only",
            }

            policy = _validated_modified_dietz_policy(
                specification,
                external_flow_value,
            )

            self.assertEqual(policy.method, "modified_dietz")
            self.assertEqual(policy.flow_timing, "settlement_date")
            self.assertEqual(policy.day_count, "actual_days")
            self.assertEqual(policy.inclusion_rule, "end_of_day")
            self.assertEqual(policy.denominator_source, "begin_market_value")
            self.assertEqual(policy.double_count_policy, "cross_check_only")

    def test_transaction_modified_dietz_eligibility_accepts_complete_inputs(
        self,
    ) -> None:
        """Modified Dietz eligibility requires explicit row, period, and policy inputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_transaction_period_specification(Path(temp_dir))
            specification = PerformanceComparisonSpecification(specification_path)
            policy = _validated_modified_dietz_policy(
                specification,
                {
                    "method": "modified_dietz",
                    "flow_timing": "trade_date",
                    "day_count": "actual_days",
                    "inclusion_rule": "beginning_of_day",
                    "denominator_source": "begin_market_value",
                    "double_count_policy": "cross_check_only",
                },
            )
            row = {
                pc_cols.PERFORMANCE_FLOW_SIGN: "external",
                pc_cols.TRANSACTION_DATE: dt.date(2025, 5, 15),
                pc_cols.SETTLEMENT_DATE: dt.date(2025, 5, 16),
            }

            eligibility = _modified_dietz_external_flow_eligibility(
                row=row,
                policy=policy,
                portfolio_id="PORT_A",
                from_date=dt.date(2025, 5, 1),
                thru_date=dt.date(2025, 5, 31),
                denominator=10000.0,
            )

            self.assertTrue(eligibility.eligible)
            self.assertEqual(eligibility.missing_inputs, ())
            self.assertEqual(eligibility.flow_date, dt.date(2025, 5, 15))

    def test_transaction_modified_dietz_eligibility_reports_missing_inputs(
        self,
    ) -> None:
        """Modified Dietz eligibility names missing inputs instead of assuming them."""
        row = {
            pc_cols.PERFORMANCE_FLOW_SIGN: "performance",
            pc_cols.TRANSACTION_DATE: dt.date(2025, 6, 1),
        }

        eligibility = _modified_dietz_external_flow_eligibility(
            row=row,
            policy=None,
            portfolio_id=None,
            from_date=None,
            thru_date=dt.date(2025, 5, 31),
            denominator=0.0,
        )

        self.assertFalse(eligibility.eligible)
        self.assertIn(
            "external performance-flow semantics",
            eligibility.missing_inputs,
        )
        self.assertIn("modified_dietz policy", eligibility.missing_inputs)
        self.assertIn("flow date", eligibility.missing_inputs)
        self.assertIn("portfolio", eligibility.missing_inputs)
        self.assertIn("portfolio period", eligibility.missing_inputs)
        self.assertIn(
            "nonzero begin_market_value denominator",
            eligibility.missing_inputs,
        )

    def test_transaction_modified_dietz_eligibility_uses_settlement_date(
        self,
    ) -> None:
        """The flow date comes from the YAML-selected timing convention."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_transaction_period_specification(Path(temp_dir))
            specification = PerformanceComparisonSpecification(specification_path)
            policy = _validated_modified_dietz_policy(
                specification,
                {
                    "method": "modified_dietz",
                    "flow_timing": "settlement_date",
                    "day_count": "actual_days",
                    "inclusion_rule": "end_of_day",
                    "denominator_source": "begin_market_value",
                    "double_count_policy": "cross_check_only",
                },
            )
            row = {
                pc_cols.PERFORMANCE_FLOW_SIGN: "external",
                pc_cols.TRANSACTION_DATE: dt.date(2025, 5, 15),
                pc_cols.SETTLEMENT_DATE: dt.date(2025, 5, 16),
            }

            eligibility = _modified_dietz_external_flow_eligibility(
                row=row,
                policy=policy,
                portfolio_id="PORT_A",
                from_date=dt.date(2025, 5, 1),
                thru_date=dt.date(2025, 5, 31),
                denominator=10000.0,
            )

            self.assertTrue(eligibility.eligible)
            self.assertEqual(eligibility.flow_date, dt.date(2025, 5, 16))

    def test_transaction_modified_dietz_eligibility_rejects_out_of_period_flow(
        self,
    ) -> None:
        """External-flow dates must fall inside the linked portfolio period."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_transaction_period_specification(Path(temp_dir))
            specification = PerformanceComparisonSpecification(specification_path)
            policy = _validated_modified_dietz_policy(
                specification,
                {
                    "method": "modified_dietz",
                    "flow_timing": "trade_date",
                    "day_count": "actual_days",
                    "inclusion_rule": "beginning_of_day",
                    "denominator_source": "begin_market_value",
                    "double_count_policy": "cross_check_only",
                },
            )
            row = {
                pc_cols.PERFORMANCE_FLOW_SIGN: "external",
                pc_cols.TRANSACTION_DATE: dt.date(2025, 6, 1),
            }

            eligibility = _modified_dietz_external_flow_eligibility(
                row=row,
                policy=policy,
                portfolio_id="PORT_A",
                from_date=dt.date(2025, 5, 1),
                thru_date=dt.date(2025, 5, 31),
                denominator=10000.0,
            )

            self.assertFalse(eligibility.eligible)
            self.assertIn("in-period flow date", eligibility.missing_inputs)

    def test_transaction_external_flow_policy_rejects_unsupported_method(self) -> None:
        """Unsupported external-flow methods fail instead of implying a formula."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_transaction_period_specification(Path(temp_dir))
            configuration = yaml.safe_load(specification_path.read_text(encoding="utf-8"))
            configuration["transaction_impact_methods"] = {
                "external_flow": {"method": "not_a_supported_method"}
            }
            specification_path.write_text(
                yaml.safe_dump(configuration),
                encoding="utf-8",
            )

            with self.assertRaises(PpaError) as context:
                PerformanceComparison(
                    PerformanceComparisonSpecification(specification_path)
                )

            self.assertIn("external_flow.method", str(context.exception))

    def test_transaction_modified_dietz_design_contract_validates_fields(self) -> None:
        """Modified Dietz YAML fields have explicit allowed cross-check values."""
        scenarios = [
            ({"flow_timing": "activity_date"}, "external_flow.flow_timing"),
            ({"day_count": "business_days"}, "external_flow.day_count"),
            ({"inclusion_rule": "midday"}, "external_flow.inclusion_rule"),
            (
                {"denominator_source": "average_market_value"},
                "external_flow.denominator_source",
            ),
            ({"double_count_policy": "aggregate"}, "external_flow.double_count_policy"),
            ({"unsupported_key": "value"}, "unsupported modified_dietz keys"),
        ]

        for overrides, expected_message in scenarios:
            with self.subTest(overrides=overrides):
                with tempfile.TemporaryDirectory() as temp_dir:
                    specification_path = _write_transaction_period_specification(
                        Path(temp_dir)
                    )
                    configuration = yaml.safe_load(
                        specification_path.read_text(encoding="utf-8")
                    )
                    modified_dietz = {
                        "method": "modified_dietz",
                        "flow_timing": "trade_date",
                        "day_count": "actual_days",
                        "inclusion_rule": "beginning_of_day",
                        "denominator_source": "begin_market_value",
                        "double_count_policy": "cross_check_only",
                    }
                    modified_dietz.update(overrides)
                    configuration["transaction_impact_methods"] = {
                        "external_flow": modified_dietz
                    }
                    specification_path.write_text(
                        yaml.safe_dump(configuration),
                        encoding="utf-8",
                    )

                    with self.assertRaises(PpaError) as context:
                        PerformanceComparison(
                            PerformanceComparisonSpecification(specification_path)
                        )

                    self.assertIn(expected_message, str(context.exception))

    def test_transaction_external_flow_future_methods_remain_rejected(self) -> None:
        """Future method names are reserved until their formulas are implemented."""
        for method in ("subperiod_linked", "unweighted_flow_delta"):
            with self.subTest(method=method):
                with tempfile.TemporaryDirectory() as temp_dir:
                    specification_path = _write_transaction_period_specification(
                        Path(temp_dir)
                    )
                    configuration = yaml.safe_load(
                        specification_path.read_text(encoding="utf-8")
                    )
                    configuration["transaction_impact_methods"] = {
                        "external_flow": {"method": method}
                    }
                    specification_path.write_text(
                        yaml.safe_dump(configuration),
                        encoding="utf-8",
                    )

                    with self.assertRaises(PpaError) as context:
                        PerformanceComparison(
                            PerformanceComparisonSpecification(specification_path)
                        )

                    self.assertIn("external_flow.method", str(context.exception))
                    self.assertIn(
                        "reserved but not implemented",
                        str(context.exception),
                    )

    def test_transaction_modified_dietz_cross_check_estimate_is_loaded(self) -> None:
        """A fully shaped Modified Dietz policy emits review-only estimates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_transaction_period_specification(Path(temp_dir))
            configuration = yaml.safe_load(specification_path.read_text(encoding="utf-8"))
            configuration["transaction_impact_methods"] = {
                "external_flow": {
                    "method": "modified_dietz",
                    "flow_timing": "trade_date",
                    "day_count": "actual_days",
                    "inclusion_rule": "beginning_of_day",
                    "denominator_source": "begin_market_value",
                    "double_count_policy": "cross_check_only",
                }
            }
            specification_path.write_text(
                yaml.safe_dump(configuration),
                encoding="utf-8",
            )

            specification = PerformanceComparisonSpecification(specification_path)
            findings = PerformanceComparison(specification).compare_transactions()
            amount_finding = next(
                finding.to_dict()
                for finding in findings
                if finding.to_dict()[FINDING_CODE] == PC_TXN_AMT
            )

            self.assertEqual(
                amount_finding[TRANSACTION_IMPACT_POLICY],
                "external_flow:modified_dietz",
            )
            self.assertEqual(
                amount_finding[TRANSACTION_IMPACT_DIAGNOSTIC],
                "modified_dietz cross-check estimate",
            )
            self.assertAlmostEqual(
                cast(float, amount_finding[TRANSACTION_IMPACT_DIAGNOSTIC_ESTIMATE]),
                10.0 * (17.0 / 31.0) / 1000.0,
            )
            policies = _transaction_impact_policies(specification)
            self.assertEqual(
                policies["external_flow"].finding_label,
                "external_flow:modified_dietz",
            )

    def test_transaction_modified_dietz_cross_check_missing_inputs_are_reported(
        self,
    ) -> None:
        """Modified Dietz stays diagnostic-only when row-level inputs are missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_transaction_period_specification(Path(temp_dir))
            configuration = yaml.safe_load(specification_path.read_text(encoding="utf-8"))
            configuration["transaction_impact_methods"] = {
                "external_flow": {
                    "method": "modified_dietz",
                    "flow_timing": "settlement_date",
                    "day_count": "actual_days",
                    "inclusion_rule": "beginning_of_day",
                    "denominator_source": "begin_market_value",
                    "double_count_policy": "cross_check_only",
                }
            }
            specification_path.write_text(
                yaml.safe_dump(configuration),
                encoding="utf-8",
            )

            findings = PerformanceComparison(
                PerformanceComparisonSpecification(specification_path)
            ).compare_transactions()
            amount_finding = next(
                finding.to_dict()
                for finding in findings
                if finding.to_dict()[FINDING_CODE] == PC_TXN_AMT
            )

            self.assertEqual(
                amount_finding[TRANSACTION_IMPACT_DIAGNOSTIC],
                "modified_dietz cross-check estimate",
            )
            self.assertAlmostEqual(
                cast(float, amount_finding[TRANSACTION_IMPACT_DIAGNOSTIC_ESTIMATE]),
                10.0 * (16.0 / 31.0) / 1000.0,
            )

    def test_transaction_modified_dietz_out_of_period_stays_unestimated(self) -> None:
        """Modified Dietz cross-check estimates require in-period flow dates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_transaction_outside_period_specification(
                Path(temp_dir)
            )
            for snapshot_name, amount in (
                ("snapshot_a", "100.00"),
                ("snapshot_b", "110.00"),
            ):
                transaction_path = (
                    Path(temp_dir) / snapshot_name / "transactions.csv"
                )
                transaction_path.write_text(
                    "TRANSACTION_ID,PORT,SEC,TRADE_DATE,SETTLE_DATE,TRAN,QTY,"
                    "PRICE,AMOUNT,CASH_FLOW_SIGN,PERFORMANCE_FLOW_SIGN\n"
                    "TXN1,PORT_A,AAPL,2025-06-15,2025-06-16,BUY,1,100.00,"
                    f"{amount},cash out,external\n",
                    encoding="utf-8",
                )
            configuration = yaml.safe_load(specification_path.read_text(encoding="utf-8"))
            configuration["transaction_impact_methods"] = {
                "external_flow": {
                    "method": "modified_dietz",
                    "flow_timing": "trade_date",
                    "day_count": "actual_days",
                    "inclusion_rule": "beginning_of_day",
                    "denominator_source": "begin_market_value",
                    "double_count_policy": "cross_check_only",
                }
            }
            specification_path.write_text(
                yaml.safe_dump(configuration),
                encoding="utf-8",
            )

            findings = PerformanceComparison(
                PerformanceComparisonSpecification(specification_path)
            ).compare_transactions()
            amount_finding = next(
                finding.to_dict()
                for finding in findings
                if finding.to_dict()[FINDING_CODE] == PC_TXN_AMT
            )

            self.assertIn(
                "modified_dietz missing inputs",
                cast(str, amount_finding[TRANSACTION_IMPACT_DIAGNOSTIC]),
            )
            self.assertIsNone(
                amount_finding[TRANSACTION_IMPACT_DIAGNOSTIC_ESTIMATE]
            )

    def test_transaction_modified_dietz_cross_check_is_not_contribution_estimate(
        self,
    ) -> None:
        """Modified Dietz diagnostics do not populate regular impact totals."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_transaction_period_specification(Path(temp_dir))
            (Path(temp_dir) / "snapshot_b" / "portperf.csv").write_text(
                "PORTFOLIO_CODE,FROM_DATE,THRU_DATE,BEGIN_MV,PORT_RETURN\n"
                "PORT_A,2025-05-01,2025-05-31,1000.00,0.02\n",
                encoding="utf-8",
            )
            configuration = yaml.safe_load(specification_path.read_text(encoding="utf-8"))
            configuration["transaction_impact_methods"] = {
                "external_flow": {
                    "method": "modified_dietz",
                    "flow_timing": "trade_date",
                    "day_count": "actual_days",
                    "inclusion_rule": "beginning_of_day",
                    "denominator_source": "begin_market_value",
                    "double_count_policy": "cross_check_only",
                }
            }
            specification_path.write_text(
                yaml.safe_dump(configuration),
                encoding="utf-8",
            )

            findings = findings_to_polars(
                PerformanceComparison(
                    PerformanceComparisonSpecification(specification_path)
                ).compare()
            )
            candidates = portfolio_period_contribution_candidates(findings)
            transaction_amount = candidates.filter(
                (pl.col(FINDING_CODE) == PC_TXN_AMT)
            ).row(0, named=True)

            self.assertIsNone(transaction_amount[ESTIMATED_RETURN_IMPACT])
            self.assertIsNotNone(
                transaction_amount[TRANSACTION_IMPACT_DIAGNOSTIC_ESTIMATE]
            )

    def test_transaction_impact_methods_reject_malformed_yaml(self) -> None:
        """Transaction impact method YAML must use the supported contract."""
        scenarios = [
            ("not-a-mapping", "must be a mapping"),
            ({"unsupported": {"method": "evidence_only"}}, "unsupported"),
            ({"external_flow": "evidence_only"}, "external_flow must be a mapping"),
            ({"external_flow": {}}, "external_flow.method is required"),
            (
                {"external_flow": {"method": "modified_dietz"}},
                "missing required modified_dietz keys",
            ),
            ({"performance": "estimate"}, "performance must be a mapping"),
            ({"performance": {}}, "performance is missing required keys"),
            (
                {
                    "performance": {
                        "method": "unsupported",
                        "denominator_source": "begin_market_value",
                    }
                },
                "performance.method must be",
            ),
            (
                {
                    "performance": {
                        "method": "transaction_amount_delta_over_return_denominator",
                        "denominator_source": "ending_market_value",
                    }
                },
                "performance.denominator_source must be one of",
            ),
        ]

        for transaction_impact_methods, expected_message in scenarios:
            with self.subTest(transaction_impact_methods=transaction_impact_methods):
                with tempfile.TemporaryDirectory() as temp_dir:
                    specification_path = _write_transaction_period_specification(
                        Path(temp_dir)
                    )
                    configuration = yaml.safe_load(
                        specification_path.read_text(encoding="utf-8")
                    )
                    configuration["transaction_impact_methods"] = (
                        transaction_impact_methods
                    )
                    specification_path.write_text(
                        yaml.safe_dump(configuration),
                        encoding="utf-8",
                    )

                    with self.assertRaises(PpaError) as context:
                        PerformanceComparison(
                            PerformanceComparisonSpecification(specification_path)
                        )

                    self.assertIn(expected_message, str(context.exception))

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
            self.assertEqual(amount_finding[TRANSACTION_SEMANTICS_SOURCE], "source")

    def test_transaction_fallback_key_treats_amount_change_as_add_drop(self) -> None:
        """Transaction amount changes require a stable transaction id."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_transaction_fallback_specification(Path(temp_dir))
            specification = PerformanceComparisonSpecification(specification_path)

            findings = PerformanceComparison(specification).compare_transactions()
            finding_dicts = [finding.to_dict() for finding in findings]
            finding_codes = [finding[FINDING_CODE] for finding in finding_dicts]

            self.assertEqual(finding_codes.count(PC_TXN_ADD), 1)
            self.assertEqual(finding_codes.count(PC_TXN_DROP), 1)
            self.assertNotIn(PC_TXN_AMT, finding_codes)
            self.assertEqual(
                {
                    finding[TRANSACTION_MATCH_STATUS]
                    for finding in finding_dicts
                    if finding[FINDING_CODE] in {PC_TXN_ADD, PC_TXN_DROP}
                },
                {TRANSACTION_MATCH_STATUS_STRICT_FALLBACK_UNMATCHED},
            )


if __name__ == "__main__":
    unittest.main()
