"""Shared cached market history for the maintained Analytics and Audit demos.

This module is internal maintainer infrastructure. It keeps network-dependent
market-data refreshes separate from deterministic demo construction and does
not make yfinance a PPAR runtime dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Mapping, Sequence, cast

import numpy as np
import pandas as pd


MARKET_HISTORY_COLUMNS: Final = (
    "date",
    "identifier",
    "yahoo_symbol",
    "raw_close",
    "unadjusted_close",
    "adjusted_close",
    "dividend",
    "split_factor",
    "repaired",
)
RETURN_WARNING_TOLERANCE: Final = 0.001
RETURN_FAILURE_TOLERANCE: Final = 0.01
_DOWNLOAD_FIELDS: Final = {
    "Adj Close": "adjusted_close",
    "Close": "raw_close",
    "Dividends": "dividend",
    "Stock Splits": "split_factor",
    "Repaired?": "repaired",
}


def load_market_history(path: Path) -> pd.DataFrame:
    """Load and validate normalized cached market history.

    Args:
        path: CSV cache created by :func:`ensure_market_history`.

    Returns:
        Market observations ordered by identifier and date.

    Raises:
        FileNotFoundError: If the cache does not exist.
        ValueError: If the cache schema, keys, or numeric values are invalid.
    """
    history = pd.read_csv(path, parse_dates=["date"])
    if "unadjusted_close" not in history.columns and {
        "identifier",
        "date",
        "raw_close",
        "split_factor",
    }.issubset(history.columns):
        history = _with_unadjusted_closes(history)
    missing = set(MARKET_HISTORY_COLUMNS).difference(history.columns)
    if missing:
        raise ValueError(f"Market-history cache is missing columns: {sorted(missing)}")
    history = cast(
        pd.DataFrame,
        history.loc[:, MARKET_HISTORY_COLUMNS],
    ).copy()
    history["identifier"] = history["identifier"].astype(str)
    history["yahoo_symbol"] = history["yahoo_symbol"].astype(str)
    for column in (
        "raw_close",
        "unadjusted_close",
        "adjusted_close",
        "dividend",
        "split_factor",
    ):
        history[column] = pd.to_numeric(history[column], errors="coerce")
    history["dividend"] = history["dividend"].fillna(0.0)
    history["split_factor"] = history["split_factor"].fillna(0.0)
    history["repaired"] = history["repaired"].fillna(False).astype(bool)
    invalid_prices = (
        history["raw_close"].le(0.0)
        | history["unadjusted_close"].le(0.0)
        | history["adjusted_close"].le(0.0)
    )
    if bool(invalid_prices.any()):
        invalid = cast(
            pd.DataFrame,
            history.loc[invalid_prices, ["identifier", "date"]],
        ).head(5)
        raise ValueError(
            "Market-history cache contains nonpositive prices: "
            f"{invalid.to_dict(orient='records')}"
        )
    if bool(history.duplicated(subset=["identifier", "date"]).any()):
        raise ValueError("Market-history cache contains duplicate identifier/date rows.")
    return history.sort_values(by=["identifier", "date"]).reset_index(drop=True)


def ensure_market_history(
    path: Path,
    identifier_to_symbol: Mapping[str, str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return cached history, downloading only when coverage is insufficient.

    Args:
        path: Shared normalized CSV cache path.
        identifier_to_symbol: Demo identifier to Yahoo symbol mapping. Multiple
            identifiers may deliberately use the same public-market proxy.
        start: First required calendar date, inclusive.
        end: Last required calendar date, inclusive.
        refresh: Whether to redownload every requested identifier.

    Returns:
        Cached history for all requested identifiers and dates.

    Raises:
        RuntimeError: If yfinance is unavailable or does not return complete
            usable coverage.
    """
    requested = {str(key): str(value) for key, value in identifier_to_symbol.items()}
    cached = _empty_history()
    if path.exists():
        cached = load_market_history(path)
    missing = set(requested) if refresh else _identifiers_needing_refresh(
        cached,
        requested,
        start=start,
        end=end,
    )
    if missing:
        downloaded = download_market_history(
            {identifier: requested[identifier] for identifier in sorted(missing)},
            start=start,
            end=end,
        )
        retained = cached.loc[~cached["identifier"].isin(missing)]
        cached = pd.concat([retained, downloaded], ignore_index=True)
        cached = cached.sort_values(["identifier", "date"]).reset_index(drop=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        cached.to_csv(path, index=False, date_format="%Y-%m-%d")
    unresolved = _identifiers_needing_refresh(
        cached,
        requested,
        start=start,
        end=end,
    )
    if unresolved:
        raise RuntimeError(
            "yFinance market history does not cover the requested interval for: "
            f"{sorted(unresolved)}"
        )
    return cached.loc[
        cached["identifier"].isin(requested)
        & cached["date"].between(start - pd.Timedelta(days=10), end)
    ].reset_index(drop=True)


def download_market_history(
    identifier_to_symbol: Mapping[str, str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Download normalized raw, adjusted, dividend, and split observations.

    Args:
        identifier_to_symbol: Demo identifier to Yahoo symbol mapping.
        start: First required date, inclusive.
        end: Last required date, inclusive.

    Returns:
        One normalized daily row per identifier and trading date.

    Raises:
        RuntimeError: If yfinance is not installed or returns no usable data.
    """
    try:
        import yfinance as yf  # pylint: disable=import-outside-toplevel
    except ImportError as error:  # pragma: no cover - maintainer environment.
        raise RuntimeError(
            "yfinance is required only when refreshing the demo market-history cache."
        ) from error

    symbols = sorted(set(identifier_to_symbol.values()))
    raw = yf.download(
        tickers=symbols,
        start=(start - pd.Timedelta(days=10)).date().isoformat(),
        end=(end + pd.Timedelta(days=2)).date().isoformat(),
        interval="1d",
        auto_adjust=False,
        actions=True,
        repair=True,
        progress=False,
        threads=True,
    )
    if raw is None or raw.empty:
        raise RuntimeError("yFinance returned no market-history observations.")
    frames: list[pd.DataFrame] = []
    for identifier, symbol in identifier_to_symbol.items():
        frame = _symbol_frame(raw, symbol)
        if frame.empty:
            continue
        frame.insert(0, "identifier", identifier)
        frame.insert(1, "yahoo_symbol", symbol)
        frame.insert(0, "date", pd.to_datetime(frame.index).tz_localize(None))
        frames.append(frame.reset_index(drop=True))
    if not frames:
        raise RuntimeError("yFinance returned no usable symbol history.")
    history = pd.concat(frames, ignore_index=True)
    normalized = _with_unadjusted_closes(history)
    return cast(pd.DataFrame, normalized.loc[:, MARKET_HISTORY_COLUMNS])


def price_on_or_before(
    history: pd.DataFrame,
    identifier: str,
    date: object,
    *,
    column: str = "unadjusted_close",
) -> float:
    """Return the last available price on or before a calendar date.

    Args:
        history: Normalized market history.
        identifier: Demo security identifier.
        date: Calendar valuation date.
        column: ``unadjusted_close``, Yahoo-source ``raw_close``, or
            ``adjusted_close``.

    Returns:
        The most recent available positive price.

    Raises:
        ValueError: If the requested price is unavailable.
    """
    if column not in {"unadjusted_close", "raw_close", "adjusted_close"}:
        raise ValueError(f"Unsupported market price column: {column}")
    target = _timestamp(date).normalize()
    rows = history.loc[
        history["identifier"].eq(identifier) & history["date"].le(target)
    ]
    if rows.empty:
        raise ValueError(
            f"No {column} is available for {identifier} on or before {target.date()}."
        )
    value = _as_float(rows.iloc[-1][column])
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"Invalid {column} for {identifier} on or before {target.date()}.")
    return value


def adjusted_period_return(
    history: pd.DataFrame,
    identifier: str,
    from_date: object,
    thru_date: object,
) -> float:
    """Return adjusted-close total return for a reporting period.

    The beginning valuation is the last trading observation strictly before
    ``from_date``; the ending valuation is the last observation on or before
    ``thru_date``.
    """
    beginning = price_on_or_before(
        history,
        identifier,
        _timestamp(from_date) - pd.Timedelta(days=1),
        column="adjusted_close",
    )
    ending = price_on_or_before(
        history,
        identifier,
        thru_date,
        column="adjusted_close",
    )
    return ending / beginning - 1.0


def raw_total_period_return(
    history: pd.DataFrame,
    identifier: str,
    from_date: object,
    thru_date: object,
) -> float:
    """Return an as-traded close total return independent of adjusted close.

    The calculation applies reported split factors to reconstructed
    contemporaneous closes and adds reported per-share cash dividends.
    """
    from_timestamp = _timestamp(from_date).normalize()
    thru_timestamp = _timestamp(thru_date).normalize()
    rows = history.loc[
        history["identifier"].eq(identifier) & history["date"].le(thru_timestamp)
    ].sort_values("date")
    beginning_rows = rows.loc[rows["date"].lt(from_timestamp)]
    if beginning_rows.empty:
        raise ValueError(f"No beginning market observation is available for {identifier}.")
    beginning_index = beginning_rows.index[-1]
    start_position = rows.index.get_loc(beginning_index)
    scoped = rows.iloc[start_position:]
    if len(scoped) < 2:
        raise ValueError(f"No ending market observation is available for {identifier}.")
    total_factor = 1.0
    previous_close = _as_float(scoped.iloc[0]["unadjusted_close"])
    for row in scoped.iloc[1:].itertuples(index=False):
        reported_split = _as_float(row.split_factor)
        split_factor = reported_split if reported_split > 0.0 else 1.0
        current_value = _as_float(row.unadjusted_close) * split_factor
        current_value += _as_float(row.dividend)
        total_factor *= current_value / previous_close
        previous_close = _as_float(row.unadjusted_close)
    return total_factor - 1.0


def reconcile_total_returns(
    history: pd.DataFrame,
    identifiers: Sequence[str],
    periods: pd.DataFrame,
    *,
    warning_tolerance: float = RETURN_WARNING_TOLERANCE,
    failure_tolerance: float = RETURN_FAILURE_TOLERANCE,
) -> pd.DataFrame:
    """Compare raw corporate-action returns with adjusted-price returns.

    Args:
        history: Normalized daily market history.
        identifiers: Identifiers to validate.
        periods: DataFrame containing ``from_date`` and ``thru_date``.
        warning_tolerance: Absolute return difference recorded as a warning.
        failure_tolerance: Absolute return difference that stops generation.

    Returns:
        Reconciliation details for every identifier and period.

    Raises:
        ValueError: If tolerances are invalid or a difference exceeds the hard
            failure tolerance.
    """
    if warning_tolerance < 0.0 or failure_tolerance <= warning_tolerance:
        raise ValueError("Return tolerances must satisfy 0 <= warning < failure.")
    rows: list[dict[str, object]] = []
    unique_periods = periods[["from_date", "thru_date"]].drop_duplicates()
    for period in unique_periods.itertuples(index=False):
        for identifier in identifiers:
            adjusted = adjusted_period_return(
                history,
                identifier,
                period.from_date,
                period.thru_date,
            )
            calculated = raw_total_period_return(
                history,
                identifier,
                period.from_date,
                period.thru_date,
            )
            difference = abs(calculated - adjusted)
            rows.append(
                {
                    "identifier": identifier,
                    "from_date": _timestamp(period.from_date).date(),
                    "thru_date": _timestamp(period.thru_date).date(),
                    "calculated_total_return": calculated,
                    "adjusted_price_return": adjusted,
                    "absolute_difference": difference,
                    "status": "fail"
                    if difference > failure_tolerance
                    else "warning"
                    if difference > warning_tolerance
                    else "pass",
                }
            )
    reconciliation = pd.DataFrame(rows)
    failures = reconciliation.loc[reconciliation["status"].eq("fail")]
    if not failures.empty:
        evidence = failures.sort_values("absolute_difference", ascending=False).head(10)
        raise ValueError(
            "Calculated total returns exceed the adjusted-price failure tolerance "
            f"of {failure_tolerance:.2%}: {evidence.to_dict(orient='records')}"
        )
    return reconciliation


def _symbol_frame(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Return normalized fields for one downloaded Yahoo symbol."""
    values: dict[str, pd.Series] = {}
    for source, target in _DOWNLOAD_FIELDS.items():
        series = _download_series(raw, source, symbol)
        if series is None:
            default: object = False if target == "repaired" else 0.0
            series = pd.Series(default, index=raw.index)
        values[target] = series
    frame = pd.DataFrame(values, index=raw.index)
    frame = frame.dropna(subset=["raw_close", "adjusted_close"])
    return frame


def _with_unadjusted_closes(history: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct contemporaneous closes from Yahoo's split-normalized Close."""
    output = history.copy()
    output["date"] = pd.to_datetime(output["date"]).dt.tz_localize(None)
    parts: list[pd.DataFrame] = []
    for _, rows in output.groupby("identifier", sort=False):
        rows = rows.sort_values("date").copy()
        factors = pd.to_numeric(rows["split_factor"], errors="coerce").fillna(0.0)
        factors = factors.where(factors.gt(0.0), 1.0)
        future_factors = factors.iloc[::-1].cumprod().iloc[::-1] / factors
        rows["unadjusted_close"] = (
            pd.to_numeric(rows["raw_close"], errors="coerce") * future_factors
        )
        parts.append(rows)
    return pd.concat(parts, ignore_index=True) if parts else output


def _download_series(
    raw: pd.DataFrame,
    field: str,
    symbol: str,
) -> pd.Series | None:
    """Return one field/symbol series from yfinance's variable column shapes."""
    if isinstance(raw.columns, pd.MultiIndex):
        key = (field, symbol)
        return raw[key] if key in raw.columns else None
    if field not in raw.columns:
        return None
    return raw[field]


def _identifiers_needing_refresh(
    history: pd.DataFrame,
    identifier_to_symbol: Mapping[str, str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> set[str]:
    """Return identifiers without adequate cached date and symbol coverage."""
    missing: set[str] = set()
    for identifier, symbol in identifier_to_symbol.items():
        rows = history.loc[history["identifier"].eq(identifier)]
        if rows.empty or set(rows["yahoo_symbol"]) != {symbol}:
            missing.add(identifier)
            continue
        earliest = pd.Timestamp(rows["date"].min())
        latest = pd.Timestamp(rows["date"].max())
        if earliest > start + pd.Timedelta(days=7) or latest < end - pd.Timedelta(days=7):
            missing.add(identifier)
    return missing


def _empty_history() -> pd.DataFrame:
    """Return an empty market-history frame with the normalized schema."""
    return pd.DataFrame(columns=MARKET_HISTORY_COLUMNS)


def _timestamp(value: object) -> pd.Timestamp:
    """Return a Timestamp from a dynamically typed tabular value."""
    return pd.Timestamp(str(value))


def _as_float(value: object) -> float:
    """Return a float from a dynamically typed tabular value."""
    return float(str(value))
