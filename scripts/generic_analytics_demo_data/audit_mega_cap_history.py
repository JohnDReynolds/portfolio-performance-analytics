"""Audit Mega-Cap source holdings-date continuity and price coverage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import pandas as pd
from pandas.tseries.offsets import BMonthEnd

import generate_mega_cap_analytics_demo_data as generator


WORKSPACE: Final = Path("_demo_output") / "generic_analytics_data_generation"
CACHE_DIRECTORY: Final = WORKSPACE / "cache" / "oef"
OUTPUT_DIRECTORY: Final = WORKSPACE / "generated_oef_files"


def main() -> None:
    """Audit candidate Mega-Cap source history and write a compact JSON report."""
    CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    requested = generator._business_month_ends(10)
    requested_status = []
    replacements = []
    for requested_date in requested:
        requested_frame = _load_or_download(requested_date)
        has_rows = not requested_frame.empty
        requested_status.append(
            {
                "requested_date": requested_date.strftime("%Y-%m-%d"),
                "has_rows": has_rows,
                "row_count": int(len(requested_frame)),
            }
        )
        if has_rows:
            continue
        replacement = _find_nearby_replacement(requested_date)
        replacements.append(
            {
                "requested_date": requested_date.strftime("%Y-%m-%d"),
                "replacement_date": (
                    None
                    if replacement is None
                    else replacement["date"].strftime("%Y-%m-%d")
                ),
                "replacement_row_count": None if replacement is None else replacement["rows"],
                "replacement_day_offset": None if replacement is None else replacement["offset"],
            }
        )

    holdings = _load_available_holdings(requested)
    price_coverage = {
        "10_year": _price_coverage(holdings, years=10),
        "5_year": _price_coverage(holdings, years=5),
    }
    report = {
        "requested_month_end_count": int(len(requested)),
        "available_month_end_count": int(sum(item["has_rows"] for item in requested_status)),
        "missing_month_ends": [
            item["requested_date"] for item in requested_status if not item["has_rows"]
        ],
        "nearby_replacements": replacements,
        "price_coverage": price_coverage,
    }
    output_path = OUTPUT_DIRECTORY / "oef_history_audit.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {output_path}")


def _find_nearby_replacement(requested_date: pd.Timestamp) -> dict[str, object] | None:
    """Find the nearest available holdings date within a conservative window."""
    candidates = []
    for offset in _nearby_offsets():
        candidate_date = requested_date + pd.Timedelta(days=offset)
        frame = _load_or_download(candidate_date)
        if not frame.empty:
            candidates.append(
                {
                    "date": candidate_date,
                    "rows": int(len(frame)),
                    "offset": int(offset),
                }
            )
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (abs(item["offset"]), item["offset"]))[0]


def _nearby_offsets() -> list[int]:
    """Return day offsets for nearby business-day and month-end probing."""
    offsets = [0]
    for distance in range(1, 8):
        offsets.extend([-distance, distance])
    offsets.extend([-31, -30, -29, 29, 30, 31])
    return offsets


def _load_or_download(asof_date: pd.Timestamp) -> pd.DataFrame:
    """Load or download one OEF snapshot for audit purposes."""
    asof = asof_date.strftime("%Y%m%d")
    cache_path = CACHE_DIRECTORY / f"oef_holdings_{asof}.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        try:
            payload = generator._download_oef_holdings_payload(asof)
        except Exception as error:  # pragma: no cover - network variability.
            print(f"FAILED {asof} {type(error).__name__}: {error}")
            return pd.DataFrame()
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return generator._holdings_frame_from_payload(payload)


def _load_available_holdings(asof_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Return all available holdings for the requested dates."""
    frames = [_load_or_download(date) for date in asof_dates]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _price_coverage(holdings: pd.DataFrame, years: int) -> dict[str, object]:
    """Return price coverage diagnostics for holdings in a lookback window."""
    if holdings.empty:
        return {}
    last = holdings["as_of_date"].max()
    first = last - pd.DateOffset(years=years) + BMonthEnd()
    scoped = holdings[holdings["as_of_date"].between(first, last)]
    tickers = sorted(scoped["identifier"].unique())
    prices = generator._load_prices(
        tickers,
        generator.DEFAULT_MARKET_HISTORY_PATH,
        refresh=False,
        start=first,
        end=last,
    )
    available = sorted(set(tickers).intersection(prices.columns))
    missing = sorted(set(tickers) - set(available))
    return {
        "from_date": str(first.date()),
        "thru_date": str(last.date()),
        "holding_security_count": int(len(tickers)),
        "priced_security_count": int(len(available)),
        "missing_price_count": int(len(missing)),
        "missing_price_tickers": missing,
    }


if __name__ == "__main__":
    main()
