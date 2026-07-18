"""Tests for shared Analytics and Audit demo market history."""

from pathlib import Path
import tempfile

import pandas as pd
import pytest

from ppar._demo_market_data import (
    adjusted_period_return,
    load_market_history,
    price_on_or_before,
    raw_total_period_return,
    reconcile_total_returns,
)


def test_load_reconstructs_contemporaneous_closes_from_split_history() -> None:
    """Yahoo split-normalized closes become as-traded historical closes."""
    history = _split_and_dividend_history().drop(columns="unadjusted_close")
    with tempfile.TemporaryDirectory() as temporary_directory:
        path = Path(temporary_directory) / "market_history.csv"
        history.to_csv(path, index=False)

        loaded = load_market_history(path)

    assert price_on_or_before(loaded, "DEMO", "2026-01-02") == pytest.approx(102.0)
    assert price_on_or_before(loaded, "DEMO", "2026-01-03") == pytest.approx(52.0)


def test_raw_corporate_action_return_reconciles_to_adjusted_close() -> None:
    """Split and dividend mechanics independently reproduce adjusted return."""
    history = _split_and_dividend_history()

    calculated = raw_total_period_return(history, "DEMO", "2026-01-02", "2026-01-04")
    adjusted = adjusted_period_return(history, "DEMO", "2026-01-02", "2026-01-04")
    reconciliation = reconcile_total_returns(
        history,
        ["DEMO"],
        pd.DataFrame({"from_date": ["2026-01-02"], "thru_date": ["2026-01-04"]}),
    )

    assert calculated == pytest.approx(0.07)
    assert adjusted == pytest.approx(0.07)
    assert reconciliation.loc[0, "status"] == "pass"


def test_total_return_reconciliation_stops_above_failure_tolerance() -> None:
    """A material corporate-action mismatch stops deterministic generation."""
    history = _split_and_dividend_history()
    history.loc[history["date"].eq(pd.Timestamp("2026-01-04")), "adjusted_close"] = 52.5

    with pytest.raises(ValueError, match="failure tolerance"):
        reconcile_total_returns(
            history,
            ["DEMO"],
            pd.DataFrame(
                {"from_date": ["2026-01-02"], "thru_date": ["2026-01-04"]}
            ),
        )


def _split_and_dividend_history() -> pd.DataFrame:
    """Return a small history with one two-for-one split and dividend."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
            ),
            "identifier": ["DEMO"] * 4,
            "yahoo_symbol": ["DEMO"] * 4,
            "raw_close": [50.0, 51.0, 52.0, 52.5],
            "unadjusted_close": [100.0, 102.0, 52.0, 52.5],
            "adjusted_close": [50.0, 51.0, 52.0, 53.5],
            "dividend": [0.0, 0.0, 0.0, 1.0],
            "split_factor": [0.0, 0.0, 2.0, 0.0],
            "repaired": [False] * 4,
        }
    )
