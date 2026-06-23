"""Probe benchmark-weight calibration for temporary analytics demo data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import generate_analytics_demo_data as generator


WORKSPACE = Path("_demo_output") / "analytics_data_generation"
CACHE_DIRECTORY = WORKSPACE / "cache"
OUTPUT_DIRECTORY = WORKSPACE / "generated_files"


def main() -> None:
    """Compare candidate benchmark weighting approaches against SPY."""
    holdings = generator._load_holdings(
        cache_directory=CACHE_DIRECTORY,
        refresh=False,
        holdings_source="spy",
        holdings_url=None,
        seed_holdings_path=None,
        top_holdings=200,
    )
    source_weights = generator._benchmark_weights(
        holdings=holdings,
        cache_directory=CACHE_DIRECTORY,
        refresh=False,
        allow_market_cap_fetch=False,
        top_holdings=200,
    )
    holdings = generator._holdings_with_weights(holdings, source_weights)
    holdings = generator._top_holdings(holdings, 200)
    prices = generator._load_monthly_prices(
        holdings=holdings,
        cache_directory=CACHE_DIRECTORY,
        refresh=False,
        years=10,
    )
    prices = generator._filter_prices(prices)
    holdings = [holding for holding in holdings if holding.ticker in prices.columns]
    returns = prices.pct_change().dropna(how="all")
    returns = returns.dropna(axis="columns", how="any")
    holdings = [holding for holding in holdings if holding.ticker in returns.columns]
    returns = returns[[holding.ticker for holding in holdings]]

    target = _load_spy_target_returns(returns.index)
    source = generator._normalize_weights(source_weights.reindex(returns.columns))
    dynamic = generator._benchmark_weight_model(
        returns=returns,
        prices=prices,
        holdings=holdings,
        static_weights=source_weights,
    )

    results = {
        "target_spy": _stats(target),
        "static_source": _stats(returns.dot(source)),
        "dynamic_spy_shares": _stats(generator._period_returns(returns, dynamic)),
    }
    for cumulative_strength in (0.0, 1.0, 5.0, 20.0, 100.0):
        weights = _calibrate_static_weights(
            returns=returns,
            target=target,
            prior=source,
            prior_strength=0.001,
            cumulative_strength=cumulative_strength,
        )
        result_key = f"calibrated_static_cum_{cumulative_strength:g}"
        results[result_key] = _stats(returns.dot(weights))
        results[result_key]["l1_distance_from_source_weights"] = round(
            float((weights - source).abs().sum()),
            6,
        )
        results[result_key]["largest_weight"] = round(float(weights.max()), 6)

    output_path = OUTPUT_DIRECTORY / "benchmark_calibration_probe.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Wrote {output_path}")


def _load_spy_target_returns(index: pd.DatetimeIndex) -> pd.Series:
    """Load cached SPY adjusted-close returns aligned to generated periods."""
    cache_path = CACHE_DIRECTORY / "spy_target_returns_10y.csv"
    if cache_path.exists() and cache_path.stat().st_size > 20:
        target = pd.read_csv(cache_path, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        import yfinance as yf

        raw = yf.download(
            tickers=["SPY"],
            period="10y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        monthly = close.resample("ME").last()
        target = monthly.pct_change().dropna()
        if target.empty:
            raise SystemExit("SPY target return download did not return usable data.")
        target.to_frame("return").to_csv(cache_path)
    aligned = target.reindex(index).dropna()
    if aligned.empty:
        raise SystemExit("SPY target returns do not overlap generated return periods.")
    return aligned


def _calibrate_static_weights(
    returns: pd.DataFrame,
    target: pd.Series,
    prior: pd.Series,
    prior_strength: float,
    cumulative_strength: float,
) -> pd.Series:
    """Fit nonnegative static weights to target returns with a source-weight prior."""
    common_index = returns.index.intersection(target.index)
    aligned_returns = returns.loc[common_index]
    aligned_target = target.loc[common_index]
    prior = prior.reindex(aligned_returns.columns).fillna(0.0)
    prior = generator._normalize_weights(prior)
    matrix = aligned_returns.to_numpy()
    target_values = aligned_target.to_numpy()
    prior_values = prior.to_numpy()
    target_cumulative = np.prod(1.0 + target_values) - 1.0

    def objective(weights: np.ndarray) -> float:
        candidate = matrix.dot(weights)
        tracking_error = np.mean((candidate - target_values) ** 2)
        prior_distance = np.mean((weights - prior_values) ** 2)
        candidate_cumulative = np.prod(1.0 + candidate) - 1.0
        cumulative_error = (candidate_cumulative - target_cumulative) ** 2
        return (
            tracking_error
            + prior_strength * prior_distance
            + cumulative_strength * cumulative_error
        )

    constraints = [{"type": "eq", "fun": lambda weights: weights.sum() - 1.0}]
    bounds = [(0.0, 1.0) for _ in prior_values]
    result = minimize(
        objective,
        prior_values,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not result.success:
        print(f"Calibration warning: {result.message}")
    return pd.Series(result.x, index=aligned_returns.columns)


def _stats(returns: pd.Series) -> dict[str, float]:
    """Return compact return diagnostics."""
    returns = returns.dropna()
    cumulative = float(np.prod(1.0 + returns) - 1.0)
    years = len(returns) / 12.0
    annualized = float((1.0 + cumulative) ** (1.0 / years) - 1.0)
    volatility = float(returns.std(ddof=0) * np.sqrt(12.0))
    sharpe = annualized / volatility if volatility else np.nan
    return {
        "periods": int(len(returns)),
        "cumulative_return": round(cumulative, 6),
        "annualized_return": round(annualized, 6),
        "annualized_volatility": round(volatility, 6),
        "annualized_sharpe": round(float(sharpe), 6),
    }


if __name__ == "__main__":
    main()
