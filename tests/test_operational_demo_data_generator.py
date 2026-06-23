"""Tests for operational demo data derivation helpers."""

# Python Imports
import importlib.util
from pathlib import Path
import tempfile
import unittest

# Third-Party Imports
import pandas as pd

# Project Imports
from ppar.performance_comparison import compare_snapshots


_REPO_ROOT = Path(__file__).resolve().parents[1]
_GENERATOR_PATH = (
    _REPO_ROOT / "scripts" / "operational_demo_data" / "derive_operational_demo_data.py"
)
_SOURCE_PATH = (
    _REPO_ROOT / "ppar" / "demos" / "data" / "performance" / "Mega-Cap Alpha Portfolio.csv"
)


def _load_generator():
    """Load the operational demo generator as a test module."""
    spec = importlib.util.spec_from_file_location("derive_operational_demo_data", _GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("Could not load operational demo generator.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestOperationalDemoDataGenerator(unittest.TestCase):
    """Verify the phase-1 operational demo data prototype."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load generator module and source fixture once for the test class."""
        cls.generator = _load_generator()
        cls.source = pd.read_csv(_SOURCE_PATH, parse_dates=["from_date", "thru_date"])

    def test_derives_recent_periods_with_cash_and_fixed_income(self) -> None:
        """Generated performance has equities plus split cash/fixed-income rows."""
        performance = self.generator.derive_operational_performance(
            self.source,
            equity_count=8,
            period_count=4,
            cash_sleeve_floor=0.04,
        )

        period_count = performance[["from_date", "thru_date"]].drop_duplicates().shape[0]
        weights = performance.groupby(["from_date", "thru_date"])["weight"].sum()

        self.assertEqual(period_count, 4)
        self.assertLess(float((weights - 1.0).abs().max()), 1e-12)
        self.assertEqual(
            set(performance[performance["sector"].eq("Cash")]["identifier"]),
            {"CASHBAL", "TBILL13W", "TNOTE2Y", "TNOTE5Y"},
        )
        self.assertEqual(
            performance[performance["asset_class"].eq("Equity")]["identifier"].nunique(),
            8,
        )

    def test_axys_exports_include_accrual_and_cash_rows(self) -> None:
        """Generated Axys-style frames contain expected operational examples."""
        performance = self.generator.derive_operational_performance(
            self.source,
            equity_count=6,
            period_count=3,
        )
        axys = self.generator.build_axys_exports(performance)

        self.assertIn("positions_holdings", axys)
        self.assertIn("cash", axys)
        self.assertIn("transactions", axys)
        self.assertGreater(float(axys["positions_holdings"]["ACCRUED"].max()), 0.0)
        self.assertEqual(set(axys["cash"]["CURRENCY"]), {"USD"})
        self.assertGreater(len(axys["transactions"]), 0)
        self.assertIn("TNOTE2Y", set(axys["sec_ref"]["SECURITY_ID"]))

    def test_restatement_snapshot_contains_controlled_differences(self) -> None:
        """Snapshot B includes source-data differences across key datasets."""
        performance = self.generator.derive_operational_performance(
            self.source,
            equity_count=8,
            period_count=4,
        )
        snapshot_a = self.generator.build_axys_exports(performance)
        snapshot_b = self.generator.build_restatement_snapshot(snapshot_a)

        latest_date = snapshot_a["portperf"]["THRU_DATE"].max()
        aapl_price_a = _value(
            snapshot_a["prices"],
            snapshot_a["prices"]["PRICE_DATE"].astype(str).eq(str(latest_date))
            & snapshot_a["prices"]["SEC"].eq("AAPL"),
            "PRICE",
        )
        aapl_price_b = _value(
            snapshot_b["prices"],
            snapshot_b["prices"]["PRICE_DATE"].astype(str).eq(str(latest_date))
            & snapshot_b["prices"]["SEC"].eq("AAPL"),
            "PRICE",
        )
        tnote_accrued_a = _value(
            snapshot_a["positions_holdings"],
            snapshot_a["positions_holdings"]["SEC"].eq("TNOTE2Y"),
            "ACCRUED",
        )
        tnote_accrued_b = _value(
            snapshot_b["positions_holdings"],
            snapshot_b["positions_holdings"]["SEC"].eq("TNOTE2Y"),
            "ACCRUED",
        )
        nvda_cost_a = _value(
            snapshot_a["positions_holdings"],
            snapshot_a["positions_holdings"]["SEC"].eq("NVDA"),
            "COST",
        )
        nvda_cost_b = _value(
            snapshot_b["positions_holdings"],
            snapshot_b["positions_holdings"]["SEC"].eq("NVDA"),
            "COST",
        )

        self.assertGreater(aapl_price_b, aapl_price_a)
        self.assertGreater(tnote_accrued_b, tnote_accrued_a)
        self.assertGreater(nvda_cost_b, nvda_cost_a)
        self.assertGreater(
            snapshot_b["portperf"]["PORT_RETURN"].iloc[-1],
            snapshot_a["portperf"]["PORT_RETURN"].iloc[-1],
        )

    def test_write_outputs_creates_reviewable_files(self) -> None:
        """Generated prototype writes source, snapshots, YAMLs, and summary paths."""
        performance = self.generator.derive_operational_performance(
            self.source,
            equity_count=5,
            period_count=3,
        )
        snapshot_a = self.generator.build_axys_exports(performance)
        snapshot_b = self.generator.build_restatement_snapshot(snapshot_a)

        with tempfile.TemporaryDirectory() as temp_directory:
            paths = self.generator.write_outputs(
                performance,
                snapshot_a,
                snapshot_b,
                Path(temp_directory),
            )
            summary = self.generator.summarize_outputs(performance, paths)

            self.assertTrue(Path(paths["source_performance"]).exists())
            self.assertTrue(Path(paths["axys_a_positions_holdings"]).exists())
            self.assertTrue(Path(paths["axys_b_positions_holdings"]).exists())
            self.assertTrue(Path(paths["portfolio_comparison_yaml"]).exists())
            self.assertTrue(Path(paths["security_comparison_yaml"]).exists())
            self.assertEqual(summary["period_count"], 3)
            self.assertEqual(summary["equity_count"], 5)

    def test_generated_comparison_yamls_produce_expected_findings(self) -> None:
        """Generated A/B snapshots run through performance comparison."""
        performance = self.generator.derive_operational_performance(
            self.source,
            equity_count=8,
            period_count=4,
        )
        snapshot_a = self.generator.build_axys_exports(performance)
        snapshot_b = self.generator.build_restatement_snapshot(snapshot_a)

        with tempfile.TemporaryDirectory() as temp_directory:
            paths = self.generator.write_outputs(
                performance,
                snapshot_a,
                snapshot_b,
                Path(temp_directory),
            )
            portfolio_findings = compare_snapshots(
                Path(paths["portfolio_comparison_yaml"]),
                require_causal_attribution=True,
            )
            security_findings = compare_snapshots(
                Path(paths["security_comparison_yaml"]),
                require_causal_attribution=True,
            )

            self.assertEqual(
                set(portfolio_findings["dataset"]),
                {"cash", "portfolio_performance", "positions", "prices", "transactions"},
            )
            self.assertIn("security_performance", set(security_findings["dataset"]))
            self.assertGreater(len(portfolio_findings), 0)
            self.assertGreater(len(security_findings), 0)


def _value(frame: pd.DataFrame, mask: pd.Series, column: str) -> float:
    """Return one numeric value from a filtered frame."""
    values = frame.loc[mask, column]
    if values.empty:
        raise AssertionError(f"Missing expected value for {column}.")
    return float(values.iloc[0])


if __name__ == "__main__":
    unittest.main()
