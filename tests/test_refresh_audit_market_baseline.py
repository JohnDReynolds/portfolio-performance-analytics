"""Tests for refreshing the packaged Audit real-market baseline."""

from __future__ import annotations

# Python imports
import importlib.util
from pathlib import Path
import sys
import unittest

# Third-party imports
import pandas as pd


_REFRESH_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "operational_demo_data"
    / "refresh_audit_market_baseline.py"
)


def _load_refresh_module():
    """Load the market-baseline refresh script as a test module."""
    spec = importlib.util.spec_from_file_location(
        "refresh_audit_market_baseline",
        _REFRESH_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("Could not load Audit market-baseline refresh script.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestRefreshAuditMarketBaseline(unittest.TestCase):
    """Protect real-price refresh and the isolated stale-price anomaly."""

    def test_refresh_injects_only_named_stale_price_field(self) -> None:
        """The anomaly copies price after real market value is calculated."""
        refresh = _load_refresh_module()
        holdings = pd.DataFrame(
            {
                "PORT": ["ALPHA", "ALPHA"],
                "SEC": ["GOOGL", "GOOGL"],
                "HOLDING_DATE": ["2025-12-31", "2026-01-30"],
                "CURRENCY": ["USD", "USD"],
                "BASE_CURRENCY": ["USD", "USD"],
                "QTY": [10.0, 10.0],
                "PRICE": [313.0, 338.0],
                "MKT_VAL": [3130.0, 3380.0],
                "BASE_MKT_VAL": [3130.0, 3380.0],
                "ACCRUED": [0.0, 0.0],
            }
        )
        transactions = pd.DataFrame(
            {
                column: pd.Series(dtype=str)
                for column in (
                    "PORT",
                    "SEC",
                    "TRANSACTION_DATE",
                    "TRAN",
                    "QTY",
                    "AMOUNT",
                    "CURRENCY",
                )
            }
        )
        market_history = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-12-31", "2026-01-30"]),
                "identifier": ["GOOGL", "GOOGL"],
                "unadjusted_close": [313.0, 338.0],
            }
        )

        refreshed = refresh.refresh_holdings(
            holdings,
            transactions,
            market_history,
        )
        target = refreshed.loc[
            refreshed["HOLDING_DATE"].astype(str).eq("2026-01-30")
        ].iloc[0]

        self.assertEqual(float(target["PRICE"]), 313.0)
        self.assertEqual(float(target["MKT_VAL"]), 3380.0)
        self.assertEqual(float(target["BASE_MKT_VAL"]), 3380.0)


if __name__ == "__main__":
    unittest.main()
