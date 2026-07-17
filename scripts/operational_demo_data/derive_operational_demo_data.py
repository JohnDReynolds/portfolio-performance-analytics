"""Derive small operational demo data from the Mega-Cap analytics demo.

The generated data is a prototype source for future Axys/APX and performance
comparison demos. It intentionally writes only under ``_demo_output`` and does
not replace packaged demo inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

import pandas as pd


_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
_DEFAULT_SOURCE_PATH: Final = (
    _REPO_ROOT
    / "ppar"
    / "setup_templates"
    / "generic_analytics"
    / "performance"
    / "Mega-Cap Alpha Portfolio.csv"
)
_DEFAULT_SECURITY_REFERENCE_PATH: Final = (
    _REPO_ROOT
    / "ppar"
    / "setup_templates"
    / "generic_analytics"
    / "classifications"
    / "Security.csv"
)
_DEFAULT_OUTPUT_DIRECTORY: Final = (
    _REPO_ROOT / "_demo_output" / "operational_demo_data_generation"
)
_AXYS_SCHEMA_PATH: Final = (
    _REPO_ROOT
    / "ppar"
    / "setup_templates"
    / "axys_apx_audit"
    / "axys_apx_column_mappings.yaml"
)
_PORTFOLIOS: Final = (
    ("ALPHA", "Mega-Cap Alpha", 1.00, 0.04),
    ("BALANCED", "Mega-Cap Balanced", 0.72, 0.16),
    ("INCOME", "Mega-Cap Income", 0.48, 0.32),
)
_BASE_MARKET_VALUE: Final = 1_000_000.0
_EQUITY_COUNT: Final = 10
_PERIOD_COUNT: Final = 6
_CASH_IDENTIFIER: Final = "CASHUSD"
_CASH_SLEEVE_FLOOR: Final = 0.04
_CVNA_SPLIT_IDENTIFIER: Final = "CVNA"
_CVNA_SPLIT_NAME: Final = "Carvana Co."
_CVNA_SYNTHETIC_WEIGHT: Final = 0.001
_CVNA_SYNTHETIC_RETURN: Final = 0.012
_CVNA_SPLIT_ADJUSTED_PRICE: Final = 58.00
_FIXED_INCOME_SLEEVE: Final = (
    (_CASH_IDENTIFIER, "US Dollar Cash", "Cash", 0.40, 0.00025, 1.0, 0.0),
    ("912797AA1", "13 Week Treasury Bill", "Treasury Bills", 0.25, 0.0038, 99.35, 0.0),
    ("91282Y2Y1", "2 Year Treasury Note", "Treasury Notes", 0.15, 0.0032, 100.15, 65.0),
    ("91282Y5Y1", "5 Year Treasury Note", "Treasury Notes", 0.15, 0.0027, 98.80, 95.0),
    ("36225MBS1", "Agency MBS Pool", "Mortgage-Backed Securities", 0.05, 0.0035, 97.25, 120.0),
)
_JPM_DIVIDEND_IDENTIFIER: Final = "JPM"
_JPM_DIVIDEND_EX_DATE: Final = "2026-04-06"
_JPM_DIVIDEND_PAY_DATE: Final = "2026-04-30"
_JPM_PRIOR_DIVIDEND_PER_SHARE: Final = 1.40
_JPM_CURRENT_DIVIDEND_PER_SHARE: Final = 1.50
_INTEREST_NOTE_TRANSACTION_ID: Final = "INCOME0603"
_INTEREST_NOTE_IDENTIFIER: Final = "91282Y2Y1"
_INTEREST_NOTE_DATE: Final = "2026-05-15"
_INTEREST_NOTE_PERIOD_END: Final = "2026-05-29"
_INTEREST_NOTE_PRIOR_AMOUNT: Final = 1_200.0
_INTEREST_NOTE_CURRENT_AMOUNT: Final = 1_280.0


def main() -> None:
    """Write first-pass operational demo data under ``_demo_output``."""
    args = _parse_args()
    source = pd.read_csv(args.source_path, parse_dates=["from_date", "thru_date"])
    security_reference = pd.read_csv(
        args.security_reference_path,
        header=None,
        names=["identifier", "name"],
    )
    performance = derive_operational_performance(
        source,
        security_reference=security_reference,
        equity_count=args.equity_count,
        period_count=args.period_count,
        cash_sleeve_floor=args.cash_sleeve_floor,
    )
    snapshot_a = build_axys_exports(performance)
    snapshot_b = build_restatement_snapshot(snapshot_a)
    output_paths = write_outputs(performance, snapshot_a, snapshot_b, args.output_directory)
    summary = summarize_outputs(performance, output_paths)
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def derive_operational_performance(
    source: pd.DataFrame,
    *,
    security_reference: pd.DataFrame,
    equity_count: int = _EQUITY_COUNT,
    period_count: int = _PERIOD_COUNT,
    cash_sleeve_floor: float = _CASH_SLEEVE_FLOOR,
) -> pd.DataFrame:
    """Return a compact operational performance set from Mega-Cap rows.

    Args:
        source: Packaged Mega-Cap Alpha Portfolio performance rows.
        security_reference: Security identifiers and display names from the
            packaged ``Security.csv`` reference file.
        equity_count: Number of high-weight equity names to keep.
        period_count: Number of recent monthly periods to retain.
        cash_sleeve_floor: Minimum visible cash/fixed-income sleeve weight.

    Returns:
        Performance rows with selected equities plus cash, T-bill, and T-note
        synthetic rows. Weights sum to 1.0 within each period.

    Raises:
        ValueError: If source data is missing required rows or columns.
    """
    required_columns = {"from_date", "thru_date", "identifier", "weight", "return"}
    missing_columns = required_columns.difference(source.columns)
    if missing_columns:
        raise ValueError(f"Source data is missing columns: {sorted(missing_columns)}")
    reference_columns = {"identifier", "name"}
    missing_reference_columns = reference_columns.difference(security_reference.columns)
    if missing_reference_columns:
        raise ValueError(
            "Security reference data is missing columns: "
            f"{sorted(missing_reference_columns)}"
        )
    source = source.merge(
        security_reference.loc[:, ["identifier", "name"]],
        on="identifier",
        how="left",
        validate="many_to_one",
    )
    missing_names = sorted(
        source.loc[source["name"].isna(), "identifier"].astype(str).unique()
    )
    if missing_names:
        raise ValueError(f"Security reference data is missing names for: {missing_names}")

    periods = (
        source[["from_date", "thru_date"]]
        .drop_duplicates()
        .sort_values(["from_date", "thru_date"])
        .tail(period_count)
    )
    if len(periods) < period_count:
        raise ValueError("Source data does not contain enough periods.")

    period_keys = set(map(tuple, periods[["from_date", "thru_date"]].itertuples(index=False)))
    recent = source[
        source[["from_date", "thru_date"]].apply(tuple, axis=1).isin(period_keys)
    ].copy()
    recent = _with_synthetic_cvna_rows(recent, periods)
    selected_equities = select_equity_identifiers(recent, equity_count)
    rows: list[dict[str, object]] = []
    for portfolio_code, portfolio_name, equity_return_scale, sleeve_floor in _PORTFOLIOS:
        for period in periods.itertuples(index=False):
            period_rows = recent[
                recent["from_date"].eq(period.from_date)
                & recent["thru_date"].eq(period.thru_date)
            ]
            equity_rows = period_rows[period_rows["identifier"].isin(selected_equities)]
            if equity_rows.empty:
                raise ValueError("No selected equity rows found for period.")
            original_cash_weight = float(
                period_rows.loc[
                    period_rows["identifier"].eq(_CASH_IDENTIFIER),
                    "weight",
                ].sum()
            )
            sleeve_weight = max(original_cash_weight, cash_sleeve_floor, sleeve_floor)
            equity_scale = (1.0 - sleeve_weight) / float(equity_rows["weight"].sum())
            for _, equity in equity_rows.sort_values("identifier").iterrows():
                rows.append(
                    {
                        "portfolio_code": portfolio_code,
                        "portfolio_name": portfolio_name,
                        "from_date": period.from_date,
                        "thru_date": period.thru_date,
                        "identifier": equity["identifier"],
                        "weight": float(equity["weight"]) * equity_scale,
                        "return": float(equity["return"]) * equity_return_scale,
                        "name": equity["name"],
                        "asset_class": "Equity",
                        "sector": "Equity",
                        "source": "Mega-Cap Alpha Portfolio",
                    }
                )
            for (
                identifier,
                name,
                asset_class,
                share,
                synthetic_return,
                _,
                _,
            ) in _FIXED_INCOME_SLEEVE:
                rows.append(
                    {
                        "portfolio_code": portfolio_code,
                        "portfolio_name": portfolio_name,
                        "from_date": period.from_date,
                        "thru_date": period.thru_date,
                        "identifier": identifier,
                        "weight": sleeve_weight * share,
                        "return": synthetic_return,
                        "name": name,
                        "asset_class": asset_class,
                        "sector": "Cash",
                        "source": "Synthetic cash/fixed-income sleeve",
                    }
                )

    performance = pd.DataFrame(rows)
    _validate_period_weights(performance)
    return performance.sort_values(
        ["portfolio_code", "from_date", "identifier"],
    ).reset_index(drop=True)


def select_equity_identifiers(source: pd.DataFrame, equity_count: int) -> list[str]:
    """Return high-weight non-cash identifiers from source rows."""
    if equity_count <= 0:
        raise ValueError("equity_count must be positive.")
    candidates = source[~source["identifier"].eq(_CASH_IDENTIFIER)]
    identifiers = (
        candidates.groupby("identifier", as_index=False)["weight"]
        .mean()
        .sort_values(["weight", "identifier"], ascending=[False, True])
        .head(equity_count)["identifier"]
        .tolist()
    )
    available_identifiers = set(candidates["identifier"])
    for required_identifier in (_JPM_DIVIDEND_IDENTIFIER, _CVNA_SPLIT_IDENTIFIER):
        if required_identifier not in available_identifiers:
            continue
        if required_identifier in identifiers:
            continue
        replaceable_index = _last_replaceable_identifier_index(
            identifiers,
            required_identifiers={_JPM_DIVIDEND_IDENTIFIER, _CVNA_SPLIT_IDENTIFIER},
        )
        identifiers[replaceable_index] = required_identifier
        identifiers = sorted(identifiers)
    if len(identifiers) < equity_count:
        raise ValueError("Source data does not contain enough equity identifiers.")
    return identifiers


def _with_synthetic_cvna_rows(recent: pd.DataFrame, periods: pd.DataFrame) -> pd.DataFrame:
    """Return recent rows with a small CVNA position for the split demo."""
    if _CVNA_SPLIT_IDENTIFIER in set(recent["identifier"]):
        return recent

    rows: list[dict[str, object]] = []
    for period in periods.itertuples(index=False):
        rows.append(
            {
                "from_date": period.from_date,
                "thru_date": period.thru_date,
                "identifier": _CVNA_SPLIT_IDENTIFIER,
                "weight": _CVNA_SYNTHETIC_WEIGHT,
                "return": _CVNA_SYNTHETIC_RETURN,
                "name": _CVNA_SPLIT_NAME,
            }
        )
    return pd.concat([recent, pd.DataFrame(rows)], ignore_index=True)


def _last_replaceable_identifier_index(
    identifiers: list[str],
    *,
    required_identifiers: set[str],
) -> int:
    """Return the last identifier index that can be replaced by a demo fixture."""
    for index in range(len(identifiers) - 1, -1, -1):
        if identifiers[index] not in required_identifiers:
            return index
    raise ValueError("No replaceable equity identifier is available.")


def build_axys_exports(performance: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return Axys/APX-style CSV frames from compact performance rows."""
    secref = _security_master(performance)
    secperf = _security_performance(performance)
    portperf = _portfolio_performance(performance)
    holdings = _positions(performance)
    transactions = _transactions(performance)
    return {
        "secref": secref,
        "secperf": secperf,
        "portperf": portperf,
        "holdings": holdings,
        "transactions": transactions,
    }


def build_restatement_snapshot(axys: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Return a controlled snapshot B with reviewable source-data differences.

    Args:
        axys: Snapshot A Axys/APX-style frames from :func:`build_axys_exports`.

    Returns:
        Snapshot B frames with small deterministic restatements across holdings,
        transactions, and performance rows.
    """
    snapshot = {name: frame.copy(deep=True) for name, frame in axys.items()}
    dates = sorted(snapshot["portperf"]["THRU_DATE"].unique())
    if len(dates) < 3:
        raise ValueError("Restatement scenarios require at least three periods.")
    early_date = dates[-3]
    middle_date = dates[-2]
    latest_date = dates[-1]

    # Fully explained: a holding price correction affects every portfolio that
    # holds AAPL, with portfolio impact flowing through holdings.market_value.
    aapl_market_value_deltas = {}
    for portfolio_code, _, _, _ in _PORTFOLIOS:
        aapl_market_value_deltas[portfolio_code] = _adjust_holding_price_multiplier(
            snapshot["holdings"],
            portfolio_code,
            "AAPL",
            target_date=latest_date,
            price_multiplier=1.01,
        )
        _adjust_security_performance(
            snapshot["secperf"],
            portfolio_code,
            latest_date,
            "AAPL",
            return_delta=0.01,
        )
    # Fully explained: JPM paid a $1.50/share dividend with an ex-date of
    # 2026-04-06. Snapshot A uses the prior $1.40/share rate; snapshot B
    # corrects the dividend amount, cash holding, and reported performance.
    jpm_dividend_delta = _adjust_jpm_dividend_to_current_rate(
        snapshot["transactions"],
        snapshot["secperf"],
        "BALANCED",
        middle_date,
    )
    _adjust_cash_holding(
        snapshot["holdings"],
        "BALANCED",
        middle_date,
        cash_delta=jpm_dividend_delta,
    )
    _apply_portfolio_return_delta(
        snapshot["portperf"],
        "BALANCED",
        middle_date,
        return_delta=jpm_dividend_delta / _BASE_MARKET_VALUE,
    )
    # Fully explained: a 91282Y2Y1 interest receipt was corrected from $1,200
    # to $1,280 on the coupon date. The $80 correction increases cash,
    # security income, and reported performance.
    tnote_interest_delta = _adjust_interest_note_to_current_amount(
        snapshot["transactions"],
        snapshot["secperf"],
    )
    _adjust_cash_holding(
        snapshot["holdings"],
        "INCOME",
        _INTEREST_NOTE_PERIOD_END,
        cash_delta=tnote_interest_delta,
    )
    _apply_portfolio_return_delta(
        snapshot["portperf"],
        "INCOME",
        _INTEREST_NOTE_PERIOD_END,
        return_delta=tnote_interest_delta / _BASE_MARKET_VALUE,
    )

    # Partly explained: the trade amount changed and explains part of the
    # period, while changed trade components give a realistic review clue for
    # the remaining difference without being additive on their own.
    _adjust_transaction_fields(
        snapshot["transactions"],
        "ALPHA",
        early_date,
        "by",
        quantity_delta=1.0,
        price_delta=0.15,
        commission_delta=10.0,
        recalculate_buy_amount=True,
    )
    _apply_portfolio_return_delta(
        snapshot["portperf"],
        "ALPHA",
        early_date,
        return_delta=-0.00028,
    )

    # Fully explained: cash-as-holding market value explains a separate
    # INCOME period change.
    _adjust_cash_holding(
        snapshot["holdings"],
        "INCOME",
        middle_date,
        cash_delta=300.0,
    )
    _apply_portfolio_return_delta(
        snapshot["portperf"],
        "INCOME",
        middle_date,
        return_delta=300.0 / _BASE_MARKET_VALUE,
    )

    # Fully explained: accrued interest explains the INCOME period change.
    tnote_market_value_a = _holding_value(
        snapshot["holdings"],
        "INCOME",
        "91282Y2Y1",
        latest_date,
        "MKT_VAL",
    )
    _adjust_holding(
        snapshot["holdings"],
        "INCOME",
        "91282Y2Y1",
        target_date=latest_date,
        quantity_multiplier=1.002,
        cost_delta=200.0,
        accrued_delta=50.0,
    )
    tnote_market_value_b = _holding_value(
        snapshot["holdings"],
        "INCOME",
        "91282Y2Y1",
        latest_date,
        "MKT_VAL",
    )
    _adjust_security_performance(
        snapshot["secperf"],
        "INCOME",
        latest_date,
        "91282Y2Y1",
        return_delta=0.0004,
    )
    _apply_portfolio_return_delta(
            snapshot["portperf"],
            "INCOME",
            latest_date,
            return_delta=round(
            aapl_market_value_deltas["INCOME"] / _BASE_MARKET_VALUE
            + (tnote_market_value_b - tnote_market_value_a) / _BASE_MARKET_VALUE
            + 50.0 / _BASE_MARKET_VALUE,
            10,
        ),
    )

    # Fully explained: holding market value changed by itself, as when a vendor
    # restates a holding value but leaves component quantity and price unchanged.
    _adjust_holding(
        snapshot["holdings"],
        "BALANCED",
        "MSFT",
        target_date=latest_date,
        market_value_delta=425.0,
    )
    _apply_portfolio_return_delta(
            snapshot["portperf"],
            "BALANCED",
            latest_date,
            return_delta=round(
            aapl_market_value_deltas["BALANCED"] / _BASE_MARKET_VALUE
            + 425.0 / _BASE_MARKET_VALUE,
            10,
        ),
    )

    # Reviewable component-only holding difference: price changed on the
    # holdings row while market value stayed the same.
    _adjust_holding(
        snapshot["holdings"],
        "INCOME",
        "91282Y5Y1",
        target_date=middle_date,
        price_delta=0.35,
    )

    _adjust_cash_holding(
        snapshot["holdings"],
        "ALPHA",
        latest_date,
        cash_delta=1_500.0,
    )
    _apply_portfolio_return_delta(
            snapshot["portperf"],
            "ALPHA",
            latest_date,
            return_delta=round(
            aapl_market_value_deltas["ALPHA"] / _BASE_MARKET_VALUE
            + 1_500.0 / _BASE_MARKET_VALUE,
            10,
        ),
    )
    return snapshot


def write_outputs(
    performance: pd.DataFrame,
    snapshot_a: dict[str, pd.DataFrame],
    snapshot_b: dict[str, pd.DataFrame],
    output_directory: Path,
) -> dict[str, str]:
    """Write generated prototype files and return their paths."""
    output_directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    source_path = output_directory / "source_performance.csv"
    performance.to_csv(source_path, index=False)
    paths["source_performance"] = str(source_path)
    schema_path = output_directory / "axys_apx_column_mappings.yaml"
    schema_path.write_text(_AXYS_SCHEMA_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    paths["axys_schema"] = str(schema_path)
    for snapshot_name, frames in (("axys_a", snapshot_a), ("axys_b", snapshot_b)):
        snapshot_directory = output_directory / snapshot_name
        snapshot_directory.mkdir(parents=True, exist_ok=True)
        for name, frame in frames.items():
            path = snapshot_directory / f"{name}.csv"
            frame.to_csv(path, index=False)
            paths[f"{snapshot_name}_{name}"] = str(path)
    comparison_yaml_path = output_directory / "axys_apx_audit.yaml"
    comparison_yaml_path.write_text(_comparison_yaml(), encoding="utf-8")
    paths["comparison_yaml"] = str(comparison_yaml_path)
    return paths


def summarize_outputs(
    performance: pd.DataFrame,
    output_paths: dict[str, str],
) -> dict[str, object]:
    """Return a compact summary for generated prototype review."""
    period_returns = (
        performance.assign(contribution=performance["weight"] * performance["return"])
        .groupby(["from_date", "thru_date"], sort=True)["contribution"]
        .sum()
    )
    sleeve = performance[performance["sector"].eq("Cash")]
    return {
        "portfolio_codes": sorted(performance["portfolio_code"].unique()),
        "period_count": int(performance[["from_date", "thru_date"]].drop_duplicates().shape[0]),
        "security_count": int(performance["identifier"].nunique()),
        "equity_count": int(
            performance[performance["asset_class"].eq("Equity")]["identifier"].nunique()
        ),
        "cash_and_fixed_income_identifiers": sorted(sleeve["identifier"].unique()),
        "average_cash_and_fixed_income_weight": round(float(sleeve["weight"].mean() * 4), 8),
        "cumulative_return": round(float((1.0 + period_returns).prod() - 1.0), 8),
        "from_date": str(performance["from_date"].min().date()),
        "thru_date": str(performance["thru_date"].max().date()),
        "outputs": output_paths,
    }


def _security_master(performance: pd.DataFrame) -> pd.DataFrame:
    securities = performance.sort_values("thru_date").drop_duplicates("identifier", keep="last")
    rows = []
    for _, security in securities.sort_values("identifier").iterrows():
        rows.append(
            {
                "SECURITY_ID": security["identifier"],
                "SECURITY_NAME": security["name"],
                "ASSET_CLASS_CODE": _asset_class_code(security["asset_class"]),
                "ASSET_CLASS_DESC": security["asset_class"],
                "SECTOR_CODE": _sector_code(security["sector"]),
                "SECTOR_DESC": security["sector"],
                "COUNTRY_CODE": "US",
                "COUNTRY_DESC": "United States",
                "CURRENCY_CODE": "USD",
                "CURRENCY_DESC": "US Dollar",
                "INDUSTRY_CODE": _sector_code(security["sector"]),
                "INDUSTRY_DESC": security["sector"],
            }
        )
    return pd.DataFrame(rows)


def _security_performance(performance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in performance.iterrows():
        begin_mv = _BASE_MARKET_VALUE * float(row["weight"])
        sec_return = float(row["return"])
        rows.append(
            {
                "DATE": row["thru_date"].date(),
                "END_MV": round(begin_mv * (1.0 + sec_return), 2),
                "INCOME": round(begin_mv * _income_component(row["identifier"], sec_return), 2),
                "GAIN_LOSS": round(
                    begin_mv * (sec_return - _income_component(row["identifier"], sec_return)),
                    2,
                ),
                "PORTFOLIO_CODE": row["portfolio_code"],
                "PORTFOLIO_NAME": row["portfolio_name"],
                "SECURITY_ID": row["identifier"],
                "ASSET_CLASS": row["asset_class"],
                "SECTOR": row["sector"],
                "COUNTRY": "US",
                "CALENDAR_MONTH": "",
                "FROM_DATE": row["from_date"].date(),
                "THRU_DATE": row["thru_date"].date(),
                "PERIOD_TYPE": "",
                "PERIOD_SEQ_IN_MONTH": "",
                "PERIODS_IN_MONTH": "",
                "BEGIN_WEIGHT": round(float(row["weight"]), 10),
                "BEGIN_MV": round(begin_mv, 2),
                "SEC_RETURN": round(sec_return, 10),
                "CONTRIBUTION": round(float(row["weight"]) * sec_return, 10),
                "SEC_SUM_WT_X_RET": "",
                "DIFFERENCE": "",
                "MATCH_TYPE": "",
            }
        )
    return pd.DataFrame(rows)


def _portfolio_performance(performance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, group in performance.groupby(
        ["portfolio_code", "portfolio_name", "from_date", "thru_date"],
        sort=True,
    ):
        portfolio_code, portfolio_name, from_date, thru_date = period
        contribution = group["weight"] * group["return"]
        portfolio_return = float(contribution.sum())
        income = sum(
            _BASE_MARKET_VALUE
            * float(row["weight"])
            * _income_component(row["identifier"], float(row["return"]))
            for _, row in group.iterrows()
        )
        rows.append(
            {
                "DATE": thru_date.date(),
                "END_MV": round(_BASE_MARKET_VALUE * (1.0 + portfolio_return), 2),
                "FLOW": 0.0,
                "INCOME": round(income, 2),
                "GAIN_LOSS": round(_BASE_MARKET_VALUE * portfolio_return - income, 2),
                "PORTFOLIO_CODE": portfolio_code,
                "PORTFOLIO_NAME": portfolio_name,
                "CALENDAR_MONTH": "",
                "FROM_DATE": from_date.date(),
                "THRU_DATE": thru_date.date(),
                "PERIOD_TYPE": "",
                "PERIOD_SEQ_IN_MONTH": "",
                "PERIODS_IN_MONTH": "",
                "BEGIN_MV": _BASE_MARKET_VALUE,
                "PORT_RETURN": round(portfolio_return, 10),
                "SEC_SUM_WT_X_RET": round(portfolio_return, 10),
                "DIFFERENCE": 0.0,
                "MATCH_TYPE": "derived",
            }
        )
    return pd.DataFrame(rows)


def _positions(performance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in performance.itertuples(index=False):
        if row.identifier == _CASH_IDENTIFIER:
            continue
        price = _price_for(row.identifier)
        market_value = _BASE_MARKET_VALUE * float(row.weight)
        rows.append(
            {
                "PORT": row.portfolio_code,
                "SEC": row.identifier,
                "HOLDING_DATE": row.thru_date.date(),
                "QTY": round(market_value / price, 4),
                "PRICE": round(price, 4),
                "MKT_VAL": round(market_value, 2),
                "COST": round(market_value * 0.985, 2),
                "ACCRUED": round(_accrued_for(row.identifier, market_value), 2),
            }
        )
    cash_rows = performance[performance["identifier"].eq(_CASH_IDENTIFIER)]
    for row in cash_rows.itertuples(index=False):
        market_value = _BASE_MARKET_VALUE * float(row.weight)
        rows.append(
            {
                "PORT": row.portfolio_code,
                "SEC": _CASH_IDENTIFIER,
                "HOLDING_DATE": row.thru_date.date(),
                "QTY": round(market_value, 4),
                "PRICE": 1.0,
                "MKT_VAL": round(market_value, 2),
                "COST": round(market_value, 2),
                "ACCRUED": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _transactions(performance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for portfolio_code, group in performance.groupby("portfolio_code", sort=True):
        equities = (
            group[group["asset_class"].eq("Equity")]
            .drop_duplicates("identifier")
            .sort_values("identifier")
            .head(2)
        )
        periods = (
            group[["from_date", "thru_date"]]
            .drop_duplicates()
            .sort_values("from_date")
            .itertuples(index=False)
        )
        for period_index, period in enumerate(periods, start=1):
            for index, row in enumerate(equities.itertuples(index=False), start=1):
                gross_amount = 4_000.0 * index
                transaction_code = "by" if index == 1 else "dv"
                transaction_identifier = row.identifier
                quantity = (
                    round(gross_amount / _price_for(transaction_identifier), 4)
                    if transaction_code == "by"
                    else 0.0
                )
                price = (
                    _price_for(transaction_identifier)
                    if transaction_code == "by"
                    else 0.0
                )
                commission = 4.95 if transaction_code == "by" else 0.0
                transaction_date = period.from_date + pd.Timedelta(days=5)
                settle_date = period.from_date + pd.Timedelta(days=6)
                amount = (
                    _buy_transaction_amount(quantity, price, commission)
                    if transaction_code == "by"
                    else round(gross_amount * 0.0125, 2)
                )
                if (
                    portfolio_code == "BALANCED"
                    and str(period.thru_date.date()) == _JPM_DIVIDEND_PAY_DATE
                    and transaction_code == "dv"
                ):
                    transaction_identifier = _JPM_DIVIDEND_IDENTIFIER
                    transaction_date = pd.Timestamp(_JPM_DIVIDEND_EX_DATE)
                    settle_date = pd.Timestamp(_JPM_DIVIDEND_PAY_DATE)
                    amount = _jpm_dividend_amount(group, _JPM_PRIOR_DIVIDEND_PER_SHARE)
                rows.append(
                    {
                        "TRANSACTION_ID": f"{portfolio_code}{period_index:02d}{index:02d}",
                        "PORT": portfolio_code,
                        "TRANSACTION_DATE": transaction_date.date(),
                        "SETTLE_DATE": settle_date.date(),
                        "SEC": transaction_identifier,
                        "TRAN": transaction_code,
                        "SEC_TYPE": _security_type_for_transaction(
                            transaction_identifier
                        ),
                        "SRC_DEST_TYPE": (
                            "$cash" if transaction_code == "by" else "$income"
                        ),
                        "SRC_DEST_SYMBOL": (
                            _CASH_IDENTIFIER if transaction_code == "by" else "$cash"
                        ),
                        "SPECIAL_SEC_TYPE": "",
                        "SPECIAL_SEC_SYMBOL": "",
                        "QTY": quantity,
                        "PRICE": price,
                        "AMOUNT": amount,
                        "COMMISSION": commission,
                    }
                )
            if portfolio_code == "ALPHA" and str(period.thru_date.date()) == "2026-01-30":
                rows.append(
                    _transaction_row(
                        transaction_id="ALPHA0203",
                        portfolio_code=portfolio_code,
                        transaction_date="2026-01-20",
                        settle_date="2026-01-20",
                        security_id=_CASH_IDENTIFIER,
                        transaction_code="wd",
                        source_destination_type="$pty",
                        source_destination_symbol="$cash",
                        amount=-500.0,
                    )
                )
            if portfolio_code == "BALANCED" and str(period.thru_date.date()) == "2026-01-30":
                rows.append(
                    _transaction_row(
                        transaction_id="BALANCED0203",
                        portfolio_code=portfolio_code,
                        transaction_date="2026-01-15",
                        settle_date="2026-01-15",
                        security_id="MSFT",
                        transaction_code="sl",
                        source_destination_type="$cash",
                        source_destination_symbol=_CASH_IDENTIFIER,
                        quantity=10.0,
                        price=114.0,
                        amount=1135.05,
                        commission=4.95,
                    )
                )
            if portfolio_code == "INCOME" and str(period.thru_date.date()) == "2026-01-30":
                rows.append(
                    _transaction_row(
                        transaction_id="INCOME0203",
                        portfolio_code=portfolio_code,
                        transaction_date="2026-01-20",
                        settle_date="2026-01-20",
                        security_id=_CASH_IDENTIFIER,
                        transaction_code="dp",
                        source_destination_type="$cash",
                        source_destination_symbol="$cash",
                        special_security_type="exus",
                        special_security_symbol="custfee",
                        amount=-50.0,
                    )
                )
            if (
                portfolio_code == "INCOME"
                and str(period.thru_date.date()) == _INTEREST_NOTE_PERIOD_END
            ):
                rows.append(
                    {
                        "TRANSACTION_ID": _INTEREST_NOTE_TRANSACTION_ID,
                        "PORT": portfolio_code,
                        "TRANSACTION_DATE": pd.Timestamp(
                            _INTEREST_NOTE_DATE
                        ).date(),
                        "SETTLE_DATE": pd.Timestamp(_INTEREST_NOTE_DATE).date(),
                        "SEC": _INTEREST_NOTE_IDENTIFIER,
                        "TRAN": "in",
                        "SEC_TYPE": _security_type_for_transaction(
                            _INTEREST_NOTE_IDENTIFIER
                        ),
                        "SRC_DEST_TYPE": "$income",
                        "SRC_DEST_SYMBOL": "$cash",
                        "SPECIAL_SEC_TYPE": "",
                        "SPECIAL_SEC_SYMBOL": "",
                        "QTY": 0.0,
                        "PRICE": 0.0,
                        "AMOUNT": _INTEREST_NOTE_PRIOR_AMOUNT,
                        "COMMISSION": 0.0,
                    }
                )
    return pd.DataFrame(rows)


def _transaction_row(
    *,
    transaction_id: str,
    portfolio_code: str,
    transaction_date: str,
    settle_date: str,
    security_id: str,
    transaction_code: str,
    source_destination_type: str,
    source_destination_symbol: str,
    quantity: float = 0.0,
    price: float = 0.0,
    amount: float = 0.0,
    commission: float = 0.0,
    special_security_type: str = "",
    special_security_symbol: str = "",
) -> dict[str, object]:
    """Return one Axys/APX-style demo transaction row."""
    return {
        "TRANSACTION_ID": transaction_id,
        "PORT": portfolio_code,
        "TRANSACTION_DATE": pd.Timestamp(transaction_date).date(),
        "SETTLE_DATE": pd.Timestamp(settle_date).date(),
        "SEC": security_id,
        "TRAN": transaction_code,
        "SEC_TYPE": _security_type_for_transaction(security_id),
        "SRC_DEST_TYPE": source_destination_type,
        "SRC_DEST_SYMBOL": source_destination_symbol,
        "SPECIAL_SEC_TYPE": special_security_type,
        "SPECIAL_SEC_SYMBOL": special_security_symbol,
        "QTY": quantity,
        "PRICE": price,
        "AMOUNT": amount,
        "COMMISSION": commission,
    }


def _security_type_for_transaction(identifier: str) -> str:
    """Return a compact Axys/APX-style security type for demo transaction rows."""
    if identifier == _CASH_IDENTIFIER:
        return "caus"
    if identifier == "36225MBS1":
        return "mbus"
    if identifier in {"912797AA1", "91282Y2Y1", "91282Y5Y1"}:
        return "fius"
    return "csus"


def _jpm_dividend_amount(performance: pd.DataFrame, dividend_per_share: float) -> float:
    """Return the JPM dividend amount for the BALANCED dividend demo phase."""
    jpm_rows = performance[
        performance["identifier"].eq(_JPM_DIVIDEND_IDENTIFIER)
        & performance["thru_date"].dt.date.astype(str).eq(_JPM_DIVIDEND_PAY_DATE)
    ]
    if jpm_rows.empty:
        raise ValueError("Could not find BALANCED JPM performance row for dividend demo.")
    shares = (
        _BASE_MARKET_VALUE
        * float(jpm_rows.iloc[0]["weight"])
        / _price_for(_JPM_DIVIDEND_IDENTIFIER)
    )
    return round(shares * dividend_per_share, 2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, default=_DEFAULT_SOURCE_PATH)
    parser.add_argument(
        "--security-reference-path",
        type=Path,
        default=_DEFAULT_SECURITY_REFERENCE_PATH,
    )
    parser.add_argument("--output-directory", type=Path, default=_DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--equity-count", type=int, default=_EQUITY_COUNT)
    parser.add_argument("--period-count", type=int, default=_PERIOD_COUNT)
    parser.add_argument("--cash-sleeve-floor", type=float, default=_CASH_SLEEVE_FLOOR)
    return parser.parse_args()


def _validate_period_weights(performance: pd.DataFrame) -> None:
    errors = (
        performance.groupby(["portfolio_code", "from_date", "thru_date"])["weight"]
        .sum()
        .sub(1.0)
        .abs()
    )
    max_error = float(errors.max())
    if max_error > 1e-12:
        raise ValueError(f"Generated period weights do not sum to 1.0: {max_error}")


def _adjust_holding(
    holdings: pd.DataFrame,
    portfolio_code: str,
    identifier: str,
    *,
    target_date: object | None = None,
    quantity_multiplier: float | None = None,
    quantity_delta: float = 0.0,
    price_delta: float = 0.0,
    market_value_delta: float = 0.0,
    cost_delta: float = 0.0,
    accrued_delta: float = 0.0,
) -> None:
    """Apply one holding restatement in place."""
    mask = holdings["PORT"].eq(portfolio_code) & holdings["SEC"].eq(identifier)
    if target_date is not None:
        mask &= holdings["HOLDING_DATE"].astype(str).eq(str(target_date))
    if not bool(mask.any()):
        raise ValueError(f"Could not find holding row for {portfolio_code} {identifier}.")
    if quantity_multiplier is not None:
        holdings.loc[mask, "QTY"] = (holdings.loc[mask, "QTY"] * quantity_multiplier).round(4)
        holdings.loc[mask, "MKT_VAL"] = (
            holdings.loc[mask, "MKT_VAL"] * quantity_multiplier
        ).round(2)
    if quantity_delta:
        holdings.loc[mask, "QTY"] = (holdings.loc[mask, "QTY"] + quantity_delta).round(4)
    if price_delta:
        holdings.loc[mask, "PRICE"] = (holdings.loc[mask, "PRICE"] + price_delta).round(4)
    if market_value_delta:
        holdings.loc[mask, "MKT_VAL"] = (
            holdings.loc[mask, "MKT_VAL"] + market_value_delta
        ).round(2)
    if cost_delta:
        holdings.loc[mask, "COST"] = (holdings.loc[mask, "COST"] + cost_delta).round(2)
    if accrued_delta:
        holdings.loc[mask, "ACCRUED"] = (
            holdings.loc[mask, "ACCRUED"] + accrued_delta
        ).round(2)


def _adjust_holding_price_multiplier(
    holdings: pd.DataFrame,
    portfolio_code: str,
    identifier: str,
    *,
    target_date: object,
    price_multiplier: float,
) -> float:
    """Apply a holding price restatement and return the market-value delta."""
    mask = (
        holdings["PORT"].eq(portfolio_code)
        & holdings["SEC"].eq(identifier)
        & holdings["HOLDING_DATE"].astype(str).eq(str(target_date))
    )
    if not bool(mask.any()):
        raise ValueError(f"Could not find holding row for {portfolio_code} {identifier}.")
    old_market_value = float(holdings.loc[mask, "MKT_VAL"].iloc[0])
    holdings.loc[mask, "PRICE"] = (holdings.loc[mask, "PRICE"] * price_multiplier).round(2)
    holdings.loc[mask, "MKT_VAL"] = (
        holdings.loc[mask, "QTY"] * holdings.loc[mask, "PRICE"]
    ).round(2)
    return float(holdings.loc[mask, "MKT_VAL"].iloc[0]) - old_market_value


def _adjust_cash_holding(
    holdings: pd.DataFrame,
    portfolio_code: str,
    target_date: object,
    *,
    cash_delta: float,
) -> None:
    """Apply one cash-holding market value restatement in place."""
    mask = (
        holdings["PORT"].eq(portfolio_code)
        & holdings["SEC"].eq(_CASH_IDENTIFIER)
        & holdings["HOLDING_DATE"].astype(str).eq(str(target_date))
    )
    if not bool(mask.any()):
        raise ValueError(
            f"Could not find cash holding for {portfolio_code} on {target_date}."
        )
    holdings.loc[mask, "MKT_VAL"] = (holdings.loc[mask, "MKT_VAL"] + cash_delta).round(2)
    holdings.loc[mask, "QTY"] = (holdings.loc[mask, "QTY"] + cash_delta).round(4)
    holdings.loc[mask, "COST"] = (holdings.loc[mask, "COST"] + cash_delta).round(2)


def _adjust_transaction_amount(
    transactions: pd.DataFrame,
    portfolio_code: str,
    target_date: object,
    transaction_code: str,
    *,
    amount_delta: float,
) -> None:
    """Apply one transaction amount restatement in place."""
    transaction_dates = pd.to_datetime(transactions["TRANSACTION_DATE"])
    target = pd.Timestamp(target_date)
    mask = (
        transactions["PORT"].eq(portfolio_code)
        & transactions["TRAN"].eq(transaction_code)
        & transaction_dates.dt.to_period("M").eq(target.to_period("M"))
    )
    if not bool(mask.any()):
        raise ValueError(
            f"Could not find {transaction_code} transaction for {portfolio_code} "
            f"in {target:%Y-%m}."
        )
    first_index = transactions.index[mask][0]
    transactions.loc[first_index, "AMOUNT"] = round(
        float(transactions.loc[first_index, "AMOUNT"]) + amount_delta,
        2,
    )


def _adjust_jpm_dividend_to_current_rate(
    transactions: pd.DataFrame,
    security_performance: pd.DataFrame,
    portfolio_code: str,
    target_date: object,
) -> float:
    """Correct the JPM dividend transaction to the current rate and return delta."""
    mask = (
        transactions["PORT"].eq(portfolio_code)
        & transactions["SEC"].eq(_JPM_DIVIDEND_IDENTIFIER)
        & transactions["TRAN"].eq("dv")
        & transactions["SETTLE_DATE"].astype(str).eq(_JPM_DIVIDEND_PAY_DATE)
    )
    if not bool(mask.any()):
        raise ValueError("Could not find JPM dividend transaction for restatement.")
    row_index = transactions.index[mask][0]
    old_amount = float(transactions.loc[row_index, "AMOUNT"])
    shares = old_amount / _JPM_PRIOR_DIVIDEND_PER_SHARE
    new_amount = round(shares * _JPM_CURRENT_DIVIDEND_PER_SHARE, 2)
    amount_delta = round(new_amount - old_amount, 2)
    transactions.loc[row_index, "AMOUNT"] = new_amount
    _adjust_security_income(
        security_performance,
        portfolio_code,
        target_date,
        _JPM_DIVIDEND_IDENTIFIER,
        income_delta=amount_delta,
    )
    return amount_delta


def _adjust_interest_note_to_current_amount(
    transactions: pd.DataFrame,
    security_performance: pd.DataFrame,
) -> float:
    """Correct the 91282Y2Y1 interest transaction and return the amount delta."""
    mask = (
        transactions["TRANSACTION_ID"].eq(_INTEREST_NOTE_TRANSACTION_ID)
        & transactions["PORT"].eq("INCOME")
    )
    if not bool(mask.any()):
        raise ValueError("Could not find 91282Y2Y1 interest transaction.")
    row_index = transactions.index[mask][0]
    old_amount = float(transactions.loc[row_index, "AMOUNT"])
    amount_delta = round(_INTEREST_NOTE_CURRENT_AMOUNT - old_amount, 2)
    transactions.loc[row_index, "AMOUNT"] = _INTEREST_NOTE_CURRENT_AMOUNT
    _adjust_security_income(
        security_performance,
        "INCOME",
        _INTEREST_NOTE_PERIOD_END,
        _INTEREST_NOTE_IDENTIFIER,
        income_delta=amount_delta,
    )
    return amount_delta


def _adjust_transaction_fields(
    transactions: pd.DataFrame,
    portfolio_code: str,
    target_date: object,
    transaction_code: str,
    *,
    quantity_delta: float = 0.0,
    price_delta: float = 0.0,
    commission_delta: float = 0.0,
    recalculate_buy_amount: bool = False,
) -> None:
    """Apply review-context transaction field restatements in place."""
    transaction_dates = pd.to_datetime(transactions["TRANSACTION_DATE"])
    target = pd.Timestamp(target_date)
    mask = (
        transactions["PORT"].eq(portfolio_code)
        & transactions["TRAN"].eq(transaction_code)
        & transaction_dates.dt.to_period("M").eq(target.to_period("M"))
    )
    if not bool(mask.any()):
        raise ValueError(
            f"Could not find {transaction_code} transaction for {portfolio_code} "
            f"in {target:%Y-%m}."
        )
    first_index = transactions.index[mask][0]
    transactions.loc[first_index, "QTY"] = round(
        float(transactions.loc[first_index, "QTY"]) + quantity_delta,
        4,
    )
    transactions.loc[first_index, "PRICE"] = round(
        float(transactions.loc[first_index, "PRICE"]) + price_delta,
        4,
    )
    transactions.loc[first_index, "COMMISSION"] = round(
        float(transactions.loc[first_index, "COMMISSION"]) + commission_delta,
        2,
    )
    if recalculate_buy_amount and transactions.loc[first_index, "TRAN"] == "by":
        transactions.loc[first_index, "AMOUNT"] = _buy_transaction_amount(
            float(transactions.loc[first_index, "QTY"]),
            float(transactions.loc[first_index, "PRICE"]),
            float(transactions.loc[first_index, "COMMISSION"]),
        )


def _buy_transaction_amount(quantity: float, price: float, commission: float) -> float:
    """Return signed cash amount for a buy transaction including commission."""
    return -round(quantity * price + commission, 2)


def _adjust_security_performance(
    security_performance: pd.DataFrame,
    portfolio_code: str,
    target_date: object,
    identifier: str,
    *,
    return_delta: float,
) -> None:
    """Apply one security-performance restatement in place."""
    mask = (
        security_performance["PORTFOLIO_CODE"].eq(portfolio_code)
        & security_performance["THRU_DATE"].astype(str).eq(str(target_date))
        & security_performance["SECURITY_ID"].eq(identifier)
    )
    if not bool(mask.any()):
        raise ValueError(
            f"Could not find security row for {portfolio_code} {identifier} on {target_date}."
        )
    begin_mv = security_performance.loc[mask, "BEGIN_MV"].astype(float)
    security_performance.loc[mask, "SEC_RETURN"] = (
        security_performance.loc[mask, "SEC_RETURN"].astype(float) + return_delta
    ).round(10)
    security_performance.loc[mask, "CONTRIBUTION"] = (
        security_performance.loc[mask, "BEGIN_WEIGHT"].astype(float)
        * security_performance.loc[mask, "SEC_RETURN"].astype(float)
    ).round(10)
    security_performance.loc[mask, "END_MV"] = (
        begin_mv * (1.0 + security_performance.loc[mask, "SEC_RETURN"].astype(float))
    ).round(2)
    income_component = security_performance.loc[mask, "INCOME"].astype(float)
    security_performance.loc[mask, "GAIN_LOSS"] = (
        security_performance.loc[mask, "END_MV"].astype(float) - begin_mv - income_component
    ).round(2)


def _adjust_security_income(
    security_performance: pd.DataFrame,
    portfolio_code: str,
    target_date: object,
    identifier: str,
    *,
    income_delta: float,
) -> None:
    """Apply one security-income correction and keep synthetic returns aligned."""
    mask = (
        security_performance["PORTFOLIO_CODE"].eq(portfolio_code)
        & security_performance["THRU_DATE"].astype(str).eq(str(target_date))
        & security_performance["SECURITY_ID"].eq(identifier)
    )
    if not bool(mask.any()):
        raise ValueError(
            f"Could not find security row for {portfolio_code} {identifier} on {target_date}."
        )
    begin_mv = security_performance.loc[mask, "BEGIN_MV"].astype(float)
    return_delta = income_delta / float(begin_mv.iloc[0])
    security_performance.loc[mask, "INCOME"] = (
        security_performance.loc[mask, "INCOME"].astype(float) + income_delta
    ).round(2)
    security_performance.loc[mask, "SEC_RETURN"] = (
        security_performance.loc[mask, "SEC_RETURN"].astype(float) + return_delta
    ).round(10)
    security_performance.loc[mask, "CONTRIBUTION"] = (
        security_performance.loc[mask, "BEGIN_WEIGHT"].astype(float)
        * security_performance.loc[mask, "SEC_RETURN"].astype(float)
    ).round(10)
    security_performance.loc[mask, "END_MV"] = (
        begin_mv * (1.0 + security_performance.loc[mask, "SEC_RETURN"].astype(float))
    ).round(2)
    security_performance.loc[mask, "GAIN_LOSS"] = (
        security_performance.loc[mask, "END_MV"].astype(float)
        - begin_mv
        - security_performance.loc[mask, "INCOME"].astype(float)
    ).round(2)


def _security_weight(
    security_performance: pd.DataFrame,
    portfolio_code: str,
    target_date: object,
    identifier: str,
) -> float:
    """Return the beginning weight for one security-performance row."""
    mask = (
        security_performance["PORTFOLIO_CODE"].eq(portfolio_code)
        & security_performance["THRU_DATE"].astype(str).eq(str(target_date))
        & security_performance["SECURITY_ID"].eq(identifier)
    )
    if not bool(mask.any()):
        raise ValueError(
            f"Could not find security weight for {portfolio_code} {identifier} on {target_date}."
        )
    return float(security_performance.loc[mask, "BEGIN_WEIGHT"].iloc[0])


def _holding_value(
    holdings: pd.DataFrame,
    portfolio_code: str,
    identifier: str,
    target_date: object,
    column: str,
) -> float:
    """Return one numeric value from a dated holding row."""
    mask = (
        holdings["PORT"].eq(portfolio_code)
        & holdings["SEC"].eq(identifier)
        & holdings["HOLDING_DATE"].astype(str).eq(str(target_date))
    )
    if not bool(mask.any()):
        raise ValueError(
            f"Could not find holding value for {portfolio_code} {identifier} "
            f"on {target_date}."
        )
    return float(holdings.loc[mask, column].iloc[0])


def _apply_portfolio_return_delta(
    portfolio_performance: pd.DataFrame,
    portfolio_code: str,
    target_date: object,
    *,
    return_delta: float,
) -> None:
    """Apply a reported portfolio return restatement in place."""
    mask = (
        portfolio_performance["PORTFOLIO_CODE"].eq(portfolio_code)
        & portfolio_performance["THRU_DATE"].astype(str).eq(str(target_date))
    )
    if not bool(mask.any()):
        raise ValueError(f"Could not find portfolio row for {portfolio_code} on {target_date}.")
    portfolio_performance.loc[mask, "PORT_RETURN"] = (
        portfolio_performance.loc[mask, "PORT_RETURN"].astype(float) + return_delta
    ).round(10)
    portfolio_performance.loc[mask, "SEC_SUM_WT_X_RET"] = (
        portfolio_performance.loc[mask, "SEC_SUM_WT_X_RET"].astype(float) + return_delta
    ).round(10)
    portfolio_performance.loc[mask, "END_MV"] = (
        _BASE_MARKET_VALUE
        * (1.0 + portfolio_performance.loc[mask, "PORT_RETURN"].astype(float))
    ).round(2)
    portfolio_performance.loc[mask, "GAIN_LOSS"] = (
        portfolio_performance.loc[mask, "END_MV"].astype(float)
        - _BASE_MARKET_VALUE
        - portfolio_performance.loc[mask, "INCOME"].astype(float)
    ).round(2)


def _apply_reported_portfolio_return_delta(
    portfolio_performance: pd.DataFrame,
    portfolio_code: str,
    target_date: object,
    *,
    return_delta: float,
) -> None:
    """Apply a reported return restatement without changing additive source fields."""
    mask = (
        portfolio_performance["PORTFOLIO_CODE"].eq(portfolio_code)
        & portfolio_performance["THRU_DATE"].astype(str).eq(str(target_date))
    )
    if not bool(mask.any()):
        raise ValueError(f"Could not find portfolio row for {portfolio_code} on {target_date}.")
    portfolio_performance.loc[mask, "PORT_RETURN"] = (
        portfolio_performance.loc[mask, "PORT_RETURN"].astype(float) + return_delta
    ).round(10)


def _comparison_yaml() -> str:
    """Return generated performance-comparison YAML text."""
    return f"""# Generated operational performance-comparison prototype.
comparison:
  name: Mega-Cap operational performance comparison prototype
snapshots:
  a:
    label: operational_axys_a
    path: axys_a
    vendor: axys
    schema: axys_apx_column_mappings.yaml

  b:
    label: operational_axys_b
    path: axys_b
    vendor: axys
    schema: axys_apx_column_mappings.yaml

files:
  portfolio_performance: portperf.csv
  security_performance: secperf.csv
  holdings: holdings.csv
  transactions: transactions.csv

transaction_rules:
  by:
    transaction_category: buy
    cash_flow_sign: negative
    performance_flow_sign: performance
  dv:
    transaction_category: income
    cash_flow_sign: positive
    performance_flow_sign: performance

security_return_impact_methods:
  transactions:
    method: modified_dietz
    flow_timing: transaction_date
    day_count: actual_days
    inclusion_rule: beginning_of_day

tolerances:
  return: 0.000001
  contribution: 0.000001
  weight: 0.000001
  market_value: 0.01
"""


def _asset_class_code(asset_class: str) -> str:
    return {
        "Cash": "CASH",
        "Equity": "EQ",
        "Mortgage-Backed Securities": "MBS",
        "Treasury Bills": "TBILL",
        "Treasury Notes": "TNOTE",
    }.get(asset_class, "OTHER")


def _sector_code(sector: str) -> str:
    return {
        "Cash": "CA",
        "Equity": "EQ",
        "Mortgage-Backed Securities": "MB",
        "Treasury Bills": "TB",
        "Treasury Notes": "TN",
    }.get(sector, "OT")


def _price_for(identifier: str) -> float:
    if identifier == _CASH_IDENTIFIER:
        return 1.0
    if identifier == _CVNA_SPLIT_IDENTIFIER:
        return _CVNA_SPLIT_ADJUSTED_PRICE
    for sleeve_identifier, _, _, _, _, price, _ in _FIXED_INCOME_SLEEVE:
        if identifier == sleeve_identifier:
            return price
    return 100.0 + (sum(ord(char) for char in identifier) % 75)


def _holding_security_id(identifier: str) -> str:
    """Return the security identifier used by holdings-style holding exports."""
    if identifier == _CASH_IDENTIFIER:
        return _CASH_IDENTIFIER
    return identifier


def _accrued_for(identifier: str, market_value: float) -> float:
    for sleeve_identifier, _, _, _, _, _, accrued_per_million in _FIXED_INCOME_SLEEVE:
        if identifier == sleeve_identifier:
            return market_value / 1_000_000.0 * accrued_per_million
    return 0.0


def _income_component(identifier: str, security_return: float) -> float:
    if identifier in {"36225MBS1", "912797AA1", "91282Y2Y1", "91282Y5Y1", _CASH_IDENTIFIER}:
        return min(security_return, max(security_return, 0.0))
    return 0.0


if __name__ == "__main__":
    main()
