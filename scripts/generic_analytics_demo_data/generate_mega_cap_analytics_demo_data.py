"""Generate temporary analytics demo data from historical OEF holdings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Final

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BMonthEnd
import requests

try:
    import yfinance as yf
except ImportError as error:  # pragma: no cover - local generation convenience.
    raise SystemExit("yfinance is required in the local .venv for this probe.") from error


WORKSPACE: Final = Path("_demo_output") / "generic_analytics_data_generation"
DEFAULT_CACHE_DIRECTORY: Final = WORKSPACE / "cache" / "oef"
DEFAULT_OUTPUT_DIRECTORY: Final = WORKSPACE / "generated_oef_files"
OEF_PRODUCT_DATA_URL: Final = (
    "https://www.blackrock.com/varnish-api/blk-one01-product-data/"
    "product-data/api/v2/get-product-data"
)
MONTHS_PER_YEAR: Final = 12
SECTOR_CODES: Final = {
    "Cash": "CA",
    "Communication Services": "CO",
    "Consumer Discretionary": "CD",
    "Consumer Staples": "CS",
    "Energy": "EN",
    "Financials": "FI",
    "Health Care": "HC",
    "Industrials": "IN",
    "Information Technology": "IT",
    "Materials": "MA",
    "Real Estate": "RE",
    "Utilities": "UT",
}
SECTOR_ALIASES: Final = {
    "Cash and/or Derivatives": "Cash",
    "Communication": "Communication Services",
    "Communications": "Communication Services",
}
CASH_IDENTIFIER: Final = "CASH_USD"
CASH_RETURN_PROXY: Final = "BIL"


def main() -> None:
    """Generate OEF-based temporary analytics demo files."""
    args = _parse_args()
    args.cache_directory.mkdir(parents=True, exist_ok=True)
    args.output_directory.mkdir(parents=True, exist_ok=True)

    requested_dates = _business_month_ends(args.years)
    holdings = _load_oef_holdings_history(
        requested_dates,
        args.cache_directory,
        args.refresh,
        args.sleep_seconds,
    )
    missing_dates = _missing_requested_dates(requested_dates, holdings)
    if missing_dates:
        replacements = _load_nearby_replacements(
            missing_dates,
            args.cache_directory,
            args.refresh,
            args.sleep_seconds,
        )
        if not replacements.empty:
            holdings = pd.concat([holdings, replacements], ignore_index=True)
    if holdings.empty:
        raise SystemExit("No OEF holdings snapshots were available.")
    _validate_monthly_continuity(
        requested_dates,
        holdings,
        args.allow_missing_months,
    )

    prices = _load_prices(
        sorted(holdings["identifier"].unique()),
        args.cache_directory,
        args.refresh,
        args.years,
    )
    benchmark, portfolio, selected_tilt = _build_performance_rows(
        holdings,
        prices,
        args.alpha_tilt,
    )
    output_paths = _write_outputs(benchmark, portfolio, holdings, args.output_directory)
    summary = _summarize(benchmark, portfolio, holdings, selected_tilt, output_paths)
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    _print_summary(summary)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--allow-missing-months", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--alpha-tilt", type=float, default=0.8)
    parser.add_argument("--cache-directory", type=Path, default=DEFAULT_CACHE_DIRECTORY)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    return parser.parse_args()


def _business_month_ends(years: int) -> pd.DatetimeIndex:
    """Return completed business month-end dates for the requested lookback."""
    last = pd.Timestamp.today().normalize().replace(day=1) - pd.Timedelta(days=1)
    last_business = last if last.weekday() < 5 else last - BMonthEnd()
    first = last_business - pd.DateOffset(years=years) + BMonthEnd()
    return pd.date_range(first, last_business, freq="BME")


def _load_oef_holdings_history(
    asof_dates: pd.DatetimeIndex,
    cache_directory: Path,
    refresh: bool,
    sleep_seconds: float,
) -> pd.DataFrame:
    """Download or load OEF holdings snapshots for each requested date."""
    rows: list[pd.DataFrame] = []
    for asof_date in asof_dates:
        asof = asof_date.strftime("%Y%m%d")
        cache_path = cache_directory / f"oef_holdings_{asof}.json"
        if cache_path.exists() and not refresh:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            payload = _download_oef_holdings_payload(asof)
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
            time.sleep(sleep_seconds)
        frame = _holdings_frame_from_payload(payload)
        if frame.empty:
            print(f"No holdings rows for {asof}; skipping.")
            continue
        rows.append(frame)
        print(f"OK {asof} {len(frame)}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _load_nearby_replacements(
    missing_dates: list[pd.Timestamp],
    cache_directory: Path,
    refresh: bool,
    sleep_seconds: float,
) -> pd.DataFrame:
    """Load nearby holdings snapshots for missing requested month-ends."""
    rows: list[pd.DataFrame] = []
    for missing_date in missing_dates:
        replacement = _find_nearby_replacement(
            missing_date,
            cache_directory,
            refresh,
            sleep_seconds,
        )
        if replacement is not None:
            rows.append(replacement)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _find_nearby_replacement(
    missing_date: pd.Timestamp,
    cache_directory: Path,
    refresh: bool,
    sleep_seconds: float,
) -> pd.DataFrame | None:
    """Return the closest available holdings snapshot within seven days."""
    candidates: list[tuple[int, pd.DataFrame]] = []
    for offset in _nearby_offsets():
        candidate_date = missing_date + pd.Timedelta(days=offset)
        asof = candidate_date.strftime("%Y%m%d")
        cache_path = cache_directory / f"oef_holdings_{asof}.json"
        if cache_path.exists() and not refresh:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            payload = _download_oef_holdings_payload(asof)
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
            time.sleep(sleep_seconds)
        frame = _holdings_frame_from_payload(payload)
        if not frame.empty:
            frame = frame.copy()
            frame["requested_as_of_date"] = missing_date.normalize()
            candidates.append((offset, frame))
    if not candidates:
        return None
    offset, frame = sorted(candidates, key=lambda item: (abs(item[0]), item[0]))[0]
    print(
        "Using nearby OEF holdings snapshot "
        f"{frame['as_of_date'].iloc[0].date()} for missing "
        f"{missing_date.date()} ({offset:+d} days)."
    )
    return frame


def _nearby_offsets() -> list[int]:
    """Return nearby day offsets for repairing one missing month-end."""
    offsets = []
    for distance in range(1, 8):
        offsets.extend([-distance, distance])
    return offsets


def _missing_requested_dates(
    requested_dates: pd.DatetimeIndex,
    holdings: pd.DataFrame,
) -> list[pd.Timestamp]:
    """Return requested dates not present in the holdings snapshots."""
    if holdings.empty:
        return list(requested_dates)
    available_months = set(holdings["as_of_date"].dt.to_period("M"))
    return [
        date
        for date in requested_dates
        if date.to_period("M") not in available_months
    ]


def _validate_monthly_continuity(
    requested_dates: pd.DatetimeIndex,
    holdings: pd.DataFrame,
    allow_missing_months: bool,
) -> None:
    """Fail fast when requested monthly holdings snapshots are unavailable."""
    available_months = set(holdings["as_of_date"].dt.to_period("M"))
    missing_dates = [
        date.strftime("%Y-%m-%d")
        for date in requested_dates
        if date.to_period("M") not in available_months
    ]
    if not missing_dates:
        return

    message = (
        "OEF holdings history is missing requested month-end snapshots: "
        + ", ".join(missing_dates)
    )
    if allow_missing_months:
        print(f"WARNING: {message}")
        return
    raise SystemExit(message)


def _download_oef_holdings_payload(asof: str) -> dict:
    """Download one OEF holdings payload from BlackRock's product-data API."""
    params = {
        "appSubType": "ISHARES",
        "appType": "PRODUCT_PAGE",
        "locale": "en_US",
        "targetSite": "us-ishares",
        "userType": "individual",
        "portfolioId": "239723",
        "component": "holdings",
        "asOfDate": asof,
    }
    response = requests.get(
        OEF_PRODUCT_DATA_URL,
        params=params,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json,*/*"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _holdings_frame_from_payload(payload: dict) -> pd.DataFrame:
    """Return normalized holdings rows from one product-data payload."""
    data_points = (
        payload["componentsByNameMap"]["holdings"]["containersByNameMap"]["all"][
            "dataPointsByNameMap"
        ]
    )
    fields = {
        "identifier": "ticker",
        "name": "issueName",
        "sector": "sectorName",
        "weight": "holdingPercent",
        "market_value": "marketValue",
        "shares": "unitsHeld",
        "as_of_date": "asOfDate",
    }
    ticker_values = data_points.get("ticker", {}).get("value")
    if not isinstance(ticker_values, list) or not ticker_values:
        return pd.DataFrame()
    length = len(ticker_values)
    output: dict[str, list[object]] = {}
    for output_name, source_name in fields.items():
        values = data_points.get(source_name, {}).get("value")
        formatted = data_points.get(source_name, {}).get("formattedValue")
        column_values = values if isinstance(values, list) else formatted
        if not isinstance(column_values, list):
            column_values = [column_values] * length
        output[output_name] = column_values

    frame = pd.DataFrame(output)
    frame["identifier"] = frame["identifier"].map(_normalize_ticker)
    frame["name"] = frame["name"].astype(str).str.strip().str.title()
    frame["sector"] = frame["sector"].astype(str).str.strip().replace(SECTOR_ALIASES)
    cash_rows = frame["sector"].eq("Cash")
    frame.loc[cash_rows, "identifier"] = CASH_IDENTIFIER
    frame.loc[cash_rows, "name"] = "US Dollar Cash"
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce") / 100.0
    frame["market_value"] = pd.to_numeric(frame["market_value"], errors="coerce")
    frame["shares"] = pd.to_numeric(frame["shares"], errors="coerce")
    frame["as_of_date"] = frame["as_of_date"].map(_parse_asof_date)
    frame["requested_as_of_date"] = frame["as_of_date"]
    frame = frame.dropna(subset=["identifier", "weight", "as_of_date"])
    frame = frame[frame["identifier"] != ""]
    frame = frame[frame["sector"].isin(SECTOR_CODES)]
    frame = _aggregate_cash_rows(frame)
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")
    frame["market_value"] = pd.to_numeric(frame["market_value"], errors="coerce")
    frame["shares"] = pd.to_numeric(frame["shares"], errors="coerce")
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"], errors="coerce")
    frame["requested_as_of_date"] = pd.to_datetime(
        frame["requested_as_of_date"],
        errors="coerce",
    )
    return frame


def _aggregate_cash_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate BlackRock cash and derivative rows into one cash balance row."""
    cash = frame[frame["identifier"].eq(CASH_IDENTIFIER)]
    non_cash = frame[~frame["identifier"].eq(CASH_IDENTIFIER)]
    if cash.empty:
        return frame
    cash_row = cash.iloc[0].copy()
    cash_row["weight"] = cash["weight"].sum()
    cash_row["market_value"] = cash["market_value"].sum()
    cash_row["shares"] = np.nan
    return pd.concat([non_cash, cash_row.to_frame().T], ignore_index=True)


def _parse_asof_date(value: object) -> pd.Timestamp:
    """Parse BlackRock date values that may be YYYYMMDD integers or text."""
    if value is None or pd.isna(value):
        return pd.NaT
    text = str(value).strip()
    if text.isdigit() and len(text) == 8:
        return pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(text, errors="coerce")


def _normalize_ticker(value: object) -> str:
    """Normalize source ticker symbols for Yahoo Finance."""
    return str(value).strip().upper().replace(".", "-")


def _load_prices(
    tickers: list[str],
    cache_directory: Path,
    refresh: bool,
    years: int,
) -> pd.DataFrame:
    """Load monthly adjusted prices for all holdings tickers."""
    cache_path = cache_directory / f"oef_monthly_prices_{years}y.csv"
    if cache_path.exists() and not refresh:
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if set(tickers).issubset(cached.columns):
            return cached
        missing = sorted(set(tickers) - set(cached.columns))
        print(
            "Cached OEF prices are missing requested tickers; refreshing: "
            + ", ".join(missing[:10])
            + ("..." if len(missing) > 10 else "")
        )
    yahoo_tickers = [
        CASH_RETURN_PROXY if ticker == CASH_IDENTIFIER else "BRK-B" if ticker == "BRKB" else ticker
        for ticker in tickers
    ]
    raw = yf.download(
        tickers=yahoo_tickers,
        period=f"{years + 1}y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close = close.rename(columns={"BRK-B": "BRKB", CASH_RETURN_PROXY: CASH_IDENTIFIER})
    monthly = close.resample("BME").last().dropna(how="all")
    monthly.to_csv(cache_path)
    return monthly


def _build_performance_rows(
    holdings: pd.DataFrame,
    prices: pd.DataFrame,
    alpha_tilt: float,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Build benchmark and alpha portfolio narrow performance rows."""
    available = sorted(set(holdings["identifier"]).intersection(prices.columns))
    prices = prices[available]
    returns = prices.pct_change().dropna(how="all")
    returns = returns.dropna(axis="columns", how="any")
    holdings = holdings[holdings["identifier"].isin(returns.columns)]
    common_dates = sorted(set(holdings["requested_as_of_date"]).intersection(returns.index))

    benchmark_frames: list[pd.DataFrame] = []
    portfolio_frames: list[pd.DataFrame] = []
    previous_date: pd.Timestamp | None = None
    for period_end in common_dates[1:]:
        start_holdings_date = common_dates[common_dates.index(period_end) - 1]
        period_holdings = holdings[
            holdings["requested_as_of_date"].eq(start_holdings_date)
        ].copy()
        period_returns = returns.loc[period_end]
        period_holdings = period_holdings[
            period_holdings["identifier"].isin(period_returns.index)
        ].copy()
        period_holdings["weight"] = period_holdings["weight"] / period_holdings["weight"].sum()
        if previous_date is None:
            previous_date = start_holdings_date
        from_date = (previous_date + pd.Timedelta(days=1)).date()
        thru_date = period_end.date()
        benchmark_frames.append(
            _performance_frame(period_holdings, period_returns, from_date, thru_date)
        )
        portfolio_holdings = _alpha_tilt_holdings(period_holdings, returns, alpha_tilt)
        portfolio_frames.append(
            _performance_frame(portfolio_holdings, period_returns, from_date, thru_date)
        )
        previous_date = period_end

    benchmark = pd.concat(benchmark_frames, ignore_index=True)
    portfolio = pd.concat(portfolio_frames, ignore_index=True)
    return benchmark, portfolio, alpha_tilt


def _alpha_tilt_holdings(
    holdings: pd.DataFrame,
    returns: pd.DataFrame,
    alpha_tilt: float,
) -> pd.DataFrame:
    """Tilt weights toward stronger realized names while preserving sector weights."""
    score = returns.mean() / returns.std(ddof=0).replace(0.0, np.nan)
    score = score.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    output = holdings.copy()
    output["score"] = output["identifier"].map(score).fillna(0.0)
    output["candidate_weight"] = output["weight"] * (1.0 + alpha_tilt * output["score"])
    output["candidate_weight"] = output["candidate_weight"].clip(lower=0.0)
    normalized_parts = []
    for _, group in output.groupby("sector", sort=False):
        target = group["weight"].sum()
        candidate_total = group["candidate_weight"].sum()
        group = group.copy()
        if candidate_total <= 0.0:
            group["weight"] = group["weight"]
        else:
            group["weight"] = group["candidate_weight"] / candidate_total * target
        normalized_parts.append(group)
    return pd.concat(normalized_parts, ignore_index=True).drop(
        columns=["score", "candidate_weight"]
    )


def _performance_frame(
    holdings: pd.DataFrame,
    returns: pd.Series,
    from_date,
    thru_date,
) -> pd.DataFrame:
    """Return ppar narrow-format performance rows for one period."""
    frame = holdings[["identifier", "weight"]].copy()
    frame["return"] = frame["identifier"].map(returns)
    frame["from_date"] = from_date
    frame["thru_date"] = thru_date
    return frame[["from_date", "thru_date", "identifier", "weight", "return"]]


def _write_outputs(
    benchmark: pd.DataFrame,
    portfolio: pd.DataFrame,
    holdings: pd.DataFrame,
    output_directory: Path,
) -> dict[str, str]:
    """Write generated performance, classification, and mapping files."""
    performance_directory = output_directory / "performance"
    classification_directory = output_directory / "classifications"
    mapping_directory = output_directory / "mappings"
    for directory in (performance_directory, classification_directory, mapping_directory):
        directory.mkdir(parents=True, exist_ok=True)

    benchmark_path = performance_directory / "Generated OEF Benchmark.csv"
    portfolio_path = performance_directory / "Generated OEF Alpha Portfolio.csv"
    security_path = classification_directory / "Generated OEF Security.csv"
    sector_path = classification_directory / "Generated OEF Economic Sector.csv"
    mapping_path = (
        mapping_directory
        / "Generated OEF Security--to--Generated OEF Economic Sector.csv"
    )
    benchmark.to_csv(benchmark_path, index=False)
    portfolio.to_csv(portfolio_path, index=False)

    securities = holdings.sort_values("as_of_date").drop_duplicates("identifier", keep="last")
    securities[["identifier", "name"]].to_csv(security_path, index=False, header=False)
    pd.DataFrame(
        sorted(
            {(SECTOR_CODES[sector], sector) for sector in securities["sector"].unique()},
            key=lambda row: row[1],
        )
    ).to_csv(sector_path, index=False, header=False)
    pd.DataFrame(
        [(row.identifier, SECTOR_CODES[row.sector]) for row in securities.itertuples()],
    ).to_csv(mapping_path, index=False, header=False)

    return {
        "benchmark_performance": str(benchmark_path),
        "portfolio_performance": str(portfolio_path),
        "security_classification": str(security_path),
        "sector_classification": str(sector_path),
        "security_to_sector_mapping": str(mapping_path),
    }


def _summarize(
    benchmark: pd.DataFrame,
    portfolio: pd.DataFrame,
    holdings: pd.DataFrame,
    selected_tilt: float,
    output_paths: dict[str, str],
) -> dict[str, object]:
    """Return compact summary metrics for generated files."""
    benchmark_returns = _period_returns(benchmark)
    portfolio_returns = _period_returns(portfolio)
    benchmark_cumulative = _cumulative_return(benchmark_returns)
    portfolio_cumulative = _cumulative_return(portfolio_returns)
    return {
        "usable_security_count": int(holdings["identifier"].nunique()),
        "holdings_snapshot_count": int(holdings["as_of_date"].nunique()),
        "period_count": int(len(benchmark_returns)),
        "from_date": str(benchmark["from_date"].min()),
        "thru_date": str(benchmark["thru_date"].max()),
        "selected_tilt": selected_tilt,
        "portfolio_cumulative_return": round(portfolio_cumulative, 6),
        "benchmark_cumulative_return": round(benchmark_cumulative, 6),
        "active_return": round(portfolio_cumulative - benchmark_cumulative, 6),
        "portfolio_annualized_sharpe": round(_annualized_sharpe(portfolio_returns), 6),
        "benchmark_annualized_sharpe": round(_annualized_sharpe(benchmark_returns), 6),
        "output_paths": output_paths,
    }


def _period_returns(frame: pd.DataFrame) -> pd.Series:
    """Return period-level weighted returns from narrow performance rows."""
    return (
        frame.assign(contribution=frame["weight"] * frame["return"])
        .groupby(["from_date", "thru_date"], sort=True)["contribution"]
        .sum()
    )


def _cumulative_return(returns: pd.Series) -> float:
    """Return cumulative compounded return."""
    return float((1.0 + returns).prod() - 1.0)


def _annualized_sharpe(returns: pd.Series) -> float:
    """Return simple annualized Sharpe with zero risk-free rate."""
    volatility = returns.std(ddof=0)
    if volatility <= 0.0 or np.isnan(volatility):
        return np.nan
    return float(returns.mean() / volatility * np.sqrt(MONTHS_PER_YEAR))


def _print_summary(summary: dict[str, object]) -> None:
    print("Generated temporary OEF analytics demo data")
    for key, value in summary.items():
        if key != "output_paths":
            print(f"{key}: {value}")
    print("Output files:")
    for path in summary["output_paths"].values():
        print(f"- {path}")


if __name__ == "__main__":
    main()
