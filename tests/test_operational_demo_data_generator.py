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
    _REPO_ROOT
    / "ppar"
    / "setup_templates"
    / "generic_analytics"
    / "performance"
    / "Mega-Cap Alpha Portfolio.csv"
)
_SECURITY_REFERENCE_PATH = (
    _REPO_ROOT
    / "ppar"
    / "setup_templates"
    / "generic_analytics"
    / "classifications"
    / "Security.csv"
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
        cls.security_reference = pd.read_csv(
            _SECURITY_REFERENCE_PATH,
            header=None,
            names=["identifier", "name"],
        )

    def test_derives_recent_periods_with_cash_and_fixed_income(self) -> None:
        """Generated performance has equities plus split cash/fixed-income rows."""
        performance = self.generator.derive_operational_performance(
            self.source,
            security_reference=self.security_reference,
            equity_count=8,
            period_count=4,
            cash_sleeve_floor=0.04,
        )

        period_count = performance[["from_date", "thru_date"]].drop_duplicates().shape[0]
        weights = performance.groupby(["portfolio_code", "from_date", "thru_date"])["weight"].sum()

        self.assertEqual(period_count, 4)
        self.assertEqual(set(performance["portfolio_code"]), {"ALPHA", "BALANCED", "INCOME"})
        self.assertLess(float((weights - 1.0).abs().max()), 1e-12)
        self.assertEqual(
            set(performance[performance["sector"].eq("Cash")]["identifier"]),
            {"CASH_USD", "36225MBS1", "912797AA1", "91282Y2Y1", "91282Y5Y1"},
        )
        self.assertEqual(
            performance[performance["asset_class"].eq("Equity")]["identifier"].nunique(),
            8,
        )

    def test_axys_exports_include_accrual_and_cash_holding_rows(self) -> None:
        """Generated Axys-style frames contain expected operational examples."""
        performance = self.generator.derive_operational_performance(
            self.source,
            security_reference=self.security_reference,
            equity_count=6,
            period_count=3,
        )
        axys = self.generator.build_axys_exports(performance)

        self.assertIn("holdings", axys)
        self.assertNotIn("cash", axys)
        self.assertIn("transactions", axys)
        self.assertGreater(float(axys["holdings"]["ACCRUED"].max()), 0.0)
        self.assertIn("CASH_USD", set(axys["holdings"]["SEC"]))
        self.assertGreater(len(axys["transactions"]), 0)
        self.assertIn("91282Y2Y1", set(axys["sec_ref"]["SECURITY_ID"]))

    def test_restatement_snapshot_contains_controlled_differences(self) -> None:
        """Snapshot B includes source-data differences across key datasets."""
        performance = self.generator.derive_operational_performance(
            self.source,
            security_reference=self.security_reference,
            equity_count=8,
            period_count=4,
        )
        snapshot_a = self.generator.build_axys_exports(performance)
        snapshot_b = self.generator.build_restatement_snapshot(snapshot_a)

        latest_date = snapshot_a["portperf"]["THRU_DATE"].max()
        aapl_price_a = _value(
            snapshot_a["holdings"],
            snapshot_a["holdings"]["PORT"].eq("ALPHA")
            & snapshot_a["holdings"]["SEC"].eq("AAPL")
            & snapshot_a["holdings"]["HOLDING_DATE"].astype(str).eq(str(latest_date)),
            "PRICE",
        )
        aapl_price_b = _value(
            snapshot_b["holdings"],
            snapshot_b["holdings"]["PORT"].eq("ALPHA")
            & snapshot_b["holdings"]["SEC"].eq("AAPL")
            & snapshot_b["holdings"]["HOLDING_DATE"].astype(str).eq(str(latest_date)),
            "PRICE",
        )
        tnote_accrued_a = _value(
            snapshot_a["holdings"],
            snapshot_a["holdings"]["PORT"].eq("INCOME")
            & snapshot_a["holdings"]["SEC"].eq("91282Y2Y1")
            & snapshot_a["holdings"]["HOLDING_DATE"].astype(str).eq(
                str(latest_date)
            ),
            "ACCRUED",
        )
        tnote_accrued_b = _value(
            snapshot_b["holdings"],
            snapshot_b["holdings"]["PORT"].eq("INCOME")
            & snapshot_b["holdings"]["SEC"].eq("91282Y2Y1")
            & snapshot_b["holdings"]["HOLDING_DATE"].astype(str).eq(
                str(latest_date)
            ),
            "ACCRUED",
        )
        tnote_quantity_a = _value(
            snapshot_a["holdings"],
            snapshot_a["holdings"]["PORT"].eq("INCOME")
            & snapshot_a["holdings"]["SEC"].eq("91282Y2Y1")
            & snapshot_a["holdings"]["HOLDING_DATE"].astype(str).eq(
                str(latest_date)
            ),
            "QTY",
        )
        tnote_quantity_b = _value(
            snapshot_b["holdings"],
            snapshot_b["holdings"]["PORT"].eq("INCOME")
            & snapshot_b["holdings"]["SEC"].eq("91282Y2Y1")
            & snapshot_b["holdings"]["HOLDING_DATE"].astype(str).eq(
                str(latest_date)
            ),
            "QTY",
        )
        tnote_cost_a = _value(
            snapshot_a["holdings"],
            snapshot_a["holdings"]["PORT"].eq("INCOME")
            & snapshot_a["holdings"]["SEC"].eq("91282Y2Y1")
            & snapshot_a["holdings"]["HOLDING_DATE"].astype(str).eq(
                str(latest_date)
            ),
            "COST",
        )
        tnote_cost_b = _value(
            snapshot_b["holdings"],
            snapshot_b["holdings"]["PORT"].eq("INCOME")
            & snapshot_b["holdings"]["SEC"].eq("91282Y2Y1")
            & snapshot_b["holdings"]["HOLDING_DATE"].astype(str).eq(
                str(latest_date)
            ),
            "COST",
        )
        cash_market_value_a = _value(
            snapshot_a["holdings"],
            snapshot_a["holdings"]["PORT"].eq("ALPHA")
            & snapshot_a["holdings"]["SEC"].eq("CASH_USD")
            & snapshot_a["holdings"]["HOLDING_DATE"].astype(str).eq(
                str(latest_date)
            ),
            "MKT_VAL",
        )
        cash_market_value_b = _value(
            snapshot_b["holdings"],
            snapshot_b["holdings"]["PORT"].eq("ALPHA")
            & snapshot_b["holdings"]["SEC"].eq("CASH_USD")
            & snapshot_b["holdings"]["HOLDING_DATE"].astype(str).eq(
                str(latest_date)
            ),
            "MKT_VAL",
        )
        alpha_buy_a = (
            snapshot_a["transactions"]["PORT"].eq("ALPHA")
            & snapshot_a["transactions"]["TRAN"].eq("by")
        )
        alpha_buy_b = (
            snapshot_b["transactions"]["PORT"].eq("ALPHA")
            & snapshot_b["transactions"]["TRAN"].eq("by")
        )

        self.assertGreater(aapl_price_b, aapl_price_a)
        self.assertGreater(tnote_accrued_b, tnote_accrued_a)
        self.assertGreater(tnote_quantity_b, tnote_quantity_a)
        self.assertGreater(tnote_cost_b, tnote_cost_a)
        self.assertGreater(cash_market_value_b, cash_market_value_a)
        self.assertGreater(
            float(snapshot_b["transactions"].loc[alpha_buy_b, "QTY"].max()),
            float(snapshot_a["transactions"].loc[alpha_buy_a, "QTY"].max()),
        )
        self.assertGreater(
            float(snapshot_b["transactions"].loc[alpha_buy_b, "PRICE"].max()),
            float(snapshot_a["transactions"].loc[alpha_buy_a, "PRICE"].max()),
        )
        self.assertGreater(
            float(snapshot_b["transactions"].loc[alpha_buy_b, "COMMISSION"].max()),
            float(snapshot_a["transactions"].loc[alpha_buy_a, "COMMISSION"].max()),
        )
        self.assertGreater(
            snapshot_b["portperf"]["PORT_RETURN"].iloc[-1],
            snapshot_a["portperf"]["PORT_RETURN"].iloc[-1],
        )

    def test_write_outputs_creates_reviewable_files(self) -> None:
        """Generated prototype writes source, snapshots, YAMLs, and summary paths."""
        performance = self.generator.derive_operational_performance(
            self.source,
            security_reference=self.security_reference,
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
            self.assertTrue(Path(paths["axys_a_holdings"]).exists())
            self.assertTrue(Path(paths["axys_b_holdings"]).exists())
            self.assertTrue(Path(paths["comparison_yaml"]).exists())
            self.assertEqual(summary["period_count"], 3)
            self.assertEqual(summary["equity_count"], 5)
            self.assertEqual(summary["portfolio_codes"], ["ALPHA", "BALANCED", "INCOME"])

    def test_generated_comparison_yamls_produce_expected_findings(self) -> None:
        """Generated A/B snapshots run through performance comparison."""
        performance = self.generator.derive_operational_performance(
            self.source,
            security_reference=self.security_reference,
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
            comparison_path = Path(paths["comparison_yaml"])
            portfolio_findings = compare_snapshots(
                comparison_path,
                comparison_level="portfolio",
            )
            security_findings = compare_snapshots(
                comparison_path,
                comparison_level="security",
            )

            self.assertEqual(
                set(portfolio_findings["dataset"]),
                {"portfolio_performance", "holdings", "transactions"},
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
