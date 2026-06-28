"""Tests for packaged performance-comparison demo data accounting guardrails."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


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

    def test_packaged_performance_comparison_demo_data_foots(self) -> None:
        """Packaged demo data has no accidental accounting or residual issues."""
        audit_module = _load_audit_module()

        issues = audit_module.audit_demo_data()

        self.assertEqual(issues, [])

    def test_packaged_holdings_are_derived_from_scenario_adjustments(self) -> None:
        """Snapshot B holdings match snapshot A plus explicit scenario rows."""
        rebuild_module = _load_rebuild_module()

        summary = rebuild_module.rebuild_demo_performance_files(
            rebuild_module._DEFAULT_AXYS_DIRECTORY,
            write=False,
        )
        snapshots = {snapshot["snapshot"]: snapshot for snapshot in summary["snapshots"]}

        self.assertEqual(snapshots["axys_full_spec_a"]["max_holdings_numeric_delta"], 0.0)
        self.assertEqual(snapshots["axys_full_spec_b"]["max_holdings_numeric_delta"], 0.0)
        self.assertFalse(snapshots["axys_full_spec_b"]["has_holdings_drift"])

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
            "axys_full_spec_a,ALPHA,CASH_USD,2026-01-30,"
            "1,0,1,1,0,Invalid base adjustment\n"
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bad_scenarios.csv"
            path.write_text(f"{columns}\n{row}")

            with self.assertRaisesRegex(ValueError, "derived snapshots"):
                rebuild_module._load_holding_scenarios(path)


if __name__ == "__main__":
    unittest.main()
