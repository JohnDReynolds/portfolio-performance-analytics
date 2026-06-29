"""Tests for packaged performance-comparison demo data accounting guardrails."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

from ppar.performance_comparison.config_validation import validate_config


_REPO_ROOT = Path(__file__).resolve().parents[1]
_AUDIT_SCRIPT_PATH = _REPO_ROOT / "scripts" / "audit_performance_comparison_demo_data.py"
_REBUILD_SCRIPT_PATH = (
    _REPO_ROOT
    / "scripts"
    / "operational_demo_data"
    / "rebuild_performance_comparison_demo_data.py"
)


def _load_audit_module():
    """Load the demo-data audit script as a test module."""
    spec = importlib.util.spec_from_file_location(
        "audit_performance_comparison_demo_data",
        _AUDIT_SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("Could not load performance-comparison demo-data audit.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_rebuild_module():
    """Load the demo-data rebuild script as a test module."""
    spec = importlib.util.spec_from_file_location(
        "rebuild_performance_comparison_demo_data",
        _REBUILD_SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("Could not load performance-comparison demo-data rebuild.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestPerformanceComparisonDemoDataAudit(unittest.TestCase):
    """Verify packaged demo data remains internally consistent."""

    def test_packaged_demo_transaction_rules_cover_observed_codes(self) -> None:
        """Packaged demo YAML explicitly defines every observed transaction code."""
        comparison_path = (
            _REPO_ROOT
            / "ppar"
            / "demos"
            / "data"
            / "axys"
            / "ppar_performance_comparison.yaml"
        )

        summary = validate_config(comparison_path)

        self.assertEqual(summary["transaction_codes_without_yaml_rules"], "none")

    def test_packaged_performance_comparison_demo_data_foots(self) -> None:
        """Packaged demo data has no accidental accounting or residual issues."""
        audit_module = _load_audit_module()

        issues = audit_module.audit_demo_data()

        self.assertEqual(issues, [])

    def test_packaged_holdings_are_derived_from_transaction_and_holding_scenarios(
        self,
    ) -> None:
        """Snapshot B holdings match transaction-derived plus explicit scenarios."""
        rebuild_module = _load_rebuild_module()

        summary = rebuild_module.rebuild_demo_performance_files(
            rebuild_module._DEFAULT_AXYS_DIRECTORY,
            write=False,
        )
        snapshots = {snapshot["snapshot"]: snapshot for snapshot in summary["snapshots"]}

        self.assertEqual(snapshots["axys_full_spec_a"]["max_holdings_numeric_delta"], 0.0)
        self.assertEqual(snapshots["axys_full_spec_b"]["max_transaction_numeric_delta"], 0.0)
        self.assertFalse(snapshots["axys_full_spec_b"]["has_transaction_field_drift"])
        self.assertEqual(snapshots["axys_full_spec_b"]["max_holdings_numeric_delta"], 0.0)
        self.assertEqual(snapshots["axys_full_spec_b"]["transaction_scenario_rows"], 7)
        self.assertEqual(
            snapshots["axys_full_spec_b"]["transaction_scenarios_by_type"],
            {
                "BUY": 1,
                "DIV": 1,
                "FEE": 1,
                "INT": 1,
                "SELL": 1,
                "SPLIT": 1,
                "WD": 1,
            },
        )
        self.assertEqual(snapshots["axys_full_spec_b"]["transaction_derived_holding_rows"], 8)
        self.assertEqual(
            snapshots["axys_full_spec_b"]["transaction_derived_holdings_by_type"],
            {
                "BUY": 2,
                "DIV": 1,
                "FEE": 1,
                "INT": 1,
                "SELL": 2,
                "WD": 1,
            },
        )
        self.assertEqual(snapshots["axys_full_spec_b"]["holding_scenario_rows"], 7)
        self.assertEqual(
            snapshots["axys_full_spec_b"]["holding_scenarios_by_type"],
            {
                "accrual_correction": 1,
                "cash_balance_correction": 1,
                "cost_only_correction": 1,
                "quantity_valuation_correction": 1,
                "valuation_mark": 3,
            },
        )
        self.assertFalse(snapshots["axys_full_spec_b"]["has_transaction_drift"])
        self.assertFalse(snapshots["axys_full_spec_b"]["has_holdings_drift"])

    def test_remaining_holding_scenarios_are_classified(self) -> None:
        """Residual holding scenarios have explicit scenario types."""
        rebuild_module = _load_rebuild_module()

        scenarios = rebuild_module._load_holding_scenarios(
            rebuild_module._DEFAULT_HOLDING_SCENARIOS_PATH,
        )
        type_counts = {}
        for adjustment in scenarios.for_snapshot("axys_full_spec_b"):
            type_counts[adjustment.scenario_type] = (
                type_counts.get(adjustment.scenario_type, 0) + 1
            )

        self.assertEqual(
            type_counts,
            {
                "valuation_mark": 3,
                "cash_balance_correction": 1,
                "quantity_valuation_correction": 1,
                "accrual_correction": 1,
                "cost_only_correction": 1,
            },
        )

    def test_scenario_coverage_audit_detects_lost_examples(self) -> None:
        """Scenario coverage guardrail catches missing demo story examples."""
        rebuild_module = _load_rebuild_module()
        summary = {
            "snapshots": [
                {
                    "snapshot": "axys_full_spec_b",
                    "transaction_scenarios_by_type": {"BUY": 1},
                    "transaction_derived_holdings_by_type": {},
                    "holding_scenarios_by_type": {},
                }
            ]
        }

        issues = rebuild_module._audit_scenario_coverage(summary)

        self.assertGreaterEqual(len(issues), 1)
        self.assertEqual(issues[0].check, "scenario_coverage")

    def test_transaction_scenarios_create_expected_holding_impacts(self) -> None:
        """Transaction changes create the expected cash and security adjustments."""
        rebuild_module = _load_rebuild_module()
        axys_directory = rebuild_module._DEFAULT_AXYS_DIRECTORY
        base_holdings = pd.read_csv(axys_directory / "axys_full_spec_a" / "holdings.csv")
        base_transactions = pd.read_csv(
            axys_directory / "axys_full_spec_a" / "transactions.csv"
        )
        rebuilt_transactions = pd.read_csv(
            axys_directory / "axys_full_spec_b" / "transactions.csv"
        )
        periods = pd.read_csv(axys_directory / "axys_full_spec_b" / "portperf.csv")

        adjustments = rebuild_module._transaction_derived_holding_adjustments(
            "axys_full_spec_b",
            base_holdings=base_holdings,
            base_transactions=base_transactions,
            current_transactions=rebuilt_transactions,
            periods=periods,
        )
        by_scenario = {adjustment.scenario: adjustment for adjustment in adjustments}

        self.assertEqual(len(adjustments), 8)
        self.assertNotIn(
            "BALANCED0503 SPLIT transaction changes cash balance.",
            by_scenario,
        )
        self._assert_adjustment(
            by_scenario["ALPHA0203 WD transaction changes cash balance."],
            portfolio="ALPHA",
            security="CASH_USD",
            holding_date="2026-01-30",
            deltas={"QTY": -1500.0, "MKT_VAL": -1500.0, "COST": -1500.0},
        )
        self._assert_adjustment(
            by_scenario["INCOME0203 FEE transaction changes cash balance."],
            portfolio="INCOME",
            security="CASH_USD",
            holding_date="2026-01-30",
            deltas={"QTY": -50.0, "MKT_VAL": -50.0, "COST": -50.0},
        )
        self._assert_adjustment(
            by_scenario["BALANCED0502 DIV transaction changes cash balance."],
            portfolio="BALANCED",
            security="CASH_USD",
            holding_date="2026-04-30",
            deltas={"QTY": 117.07, "MKT_VAL": 117.07, "COST": 117.07},
        )
        self._assert_adjustment(
            by_scenario["INCOME0603 INT transaction changes cash balance."],
            portfolio="INCOME",
            security="CASH_USD",
            holding_date="2026-05-29",
            deltas={"QTY": 80.0, "MKT_VAL": 80.0, "COST": 80.0},
        )
        self._assert_adjustment(
            by_scenario["ALPHA0401 BUY transaction changes ending holding."],
            portfolio="ALPHA",
            security="AAPL",
            holding_date="2026-03-31",
            deltas={"QTY": 1.1372, "MKT_VAL": 183.0892, "COST": 183.0892},
        )
        self._assert_adjustment(
            by_scenario["ALPHA0401 BUY transaction changes cash balance."],
            portfolio="ALPHA",
            security="CASH_USD",
            holding_date="2026-03-31",
            deltas={"QTY": -196.98, "MKT_VAL": -196.98, "COST": -196.98},
        )
        self._assert_adjustment(
            by_scenario["BALANCED0203 SELL transaction changes ending holding."],
            portfolio="BALANCED",
            security="MSFT",
            holding_date="2026-01-30",
            deltas={"QTY": -2.0, "MKT_VAL": -228.0, "COST": -224.58001579710972},
        )
        self._assert_adjustment(
            by_scenario["BALANCED0203 SELL transaction changes cash balance."],
            portfolio="BALANCED",
            security="CASH_USD",
            holding_date="2026-01-30",
            deltas={"QTY": 226.0, "MKT_VAL": 226.0, "COST": 226.0},
        )

    def test_holding_scenario_file_requires_exact_columns(self) -> None:
        """Scenario CSV shape errors fail before any demo files are rebuilt."""
        rebuild_module = _load_rebuild_module()

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bad_scenarios.csv"
            path.write_text("snapshot,PORT\naxys_full_spec_b,ALPHA\n")

            with self.assertRaisesRegex(ValueError, "columns must exactly match"):
                rebuild_module._load_holding_scenarios(path)

    def test_holding_scenario_file_rejects_base_snapshot_adjustments(self) -> None:
        """Scenario rows must target derived snapshots, not the base snapshot."""
        rebuild_module = _load_rebuild_module()
        columns = ",".join(rebuild_module._HOLDING_SCENARIO_COLUMNS)
        row = (
            "axys_full_spec_a,cash_balance_correction,ALPHA,CASH_USD,2026-01-30,"
            "1,0,1,1,0,Invalid base adjustment\n"
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bad_scenarios.csv"
            path.write_text(f"{columns}\n{row}")

            with self.assertRaisesRegex(ValueError, "derived snapshots"):
                rebuild_module._load_holding_scenarios(path)

    def test_holding_scenario_file_validates_type_delta_patterns(self) -> None:
        """Scenario type controls which holding fields may change."""
        rebuild_module = _load_rebuild_module()
        columns = ",".join(rebuild_module._HOLDING_SCENARIO_COLUMNS)
        row = (
            "axys_full_spec_b,valuation_mark,ALPHA,AAPL,2026-05-29,"
            "1,0,1,0,0,Invalid valuation mark quantity change\n"
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bad_scenarios.csv"
            path.write_text(f"{columns}\n{row}")

            with self.assertRaisesRegex(ValueError, "changed fields"):
                rebuild_module._load_holding_scenarios(path)

    def test_transaction_scenario_file_requires_exact_columns(self) -> None:
        """Transaction scenario CSV shape errors fail before rebuilds."""
        rebuild_module = _load_rebuild_module()

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bad_transaction_scenarios.csv"
            path.write_text("snapshot,TRANSACTION_ID\naxys_full_spec_b,ALPHA0203\n")

            with self.assertRaisesRegex(ValueError, "columns must exactly match"):
                rebuild_module._load_transaction_scenarios(path)

    def test_transaction_scenario_file_rejects_base_snapshot_adjustments(self) -> None:
        """Transaction scenario rows must target derived snapshots."""
        rebuild_module = _load_rebuild_module()
        columns = ",".join(rebuild_module._TRANSACTION_SCENARIO_COLUMNS)
        row = "axys_full_spec_a,ALPHA0203,0,0,-1,0,Invalid base adjustment\n"

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bad_transaction_scenarios.csv"
            path.write_text(f"{columns}\n{row}")

            with self.assertRaisesRegex(ValueError, "derived snapshots"):
                rebuild_module._load_transaction_scenarios(path)

    def _assert_adjustment(
        self,
        adjustment,
        *,
        portfolio: str,
        security: str,
        holding_date: str,
        deltas: dict[str, float],
    ) -> None:
        """Assert one derived holding adjustment has the expected accounting impact."""
        self.assertEqual(adjustment.portfolio, portfolio)
        self.assertEqual(adjustment.security, security)
        self.assertEqual(adjustment.holding_date, holding_date)
        for column, expected_delta in deltas.items():
            self.assertAlmostEqual(adjustment.deltas[column], expected_delta, places=6)
        self.assertEqual(adjustment.deltas["PRICE"], 0.0)
        self.assertEqual(adjustment.deltas["ACCRUED"], 0.0)


if __name__ == "__main__":
    unittest.main()
