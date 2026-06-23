"""Derive small operational demo data from the Mega-Cap analytics demo.

The generated data is a prototype source for future Axys and performance
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
    _REPO_ROOT / "ppar" / "demos" / "data" / "performance" / "Mega-Cap Alpha Portfolio.csv"
)
_DEFAULT_OUTPUT_DIRECTORY: Final = (
    _REPO_ROOT / "_demo_output" / "operational_demo_data_generation"
)
_AXYS_SCHEMA_PATH: Final = (
    _REPO_ROOT / "ppar" / "demos" / "data" / "axys" / "axys_column_mappings.yaml"
)
_PORTFOLIO_CODE: Final = "MEGA_ALPHA_OPS"
_PORTFOLIO_NAME: Final = "Mega-Cap Alpha Operational Demo"
_BASE_MARKET_VALUE: Final = 1_000_000.0
_EQUITY_COUNT: Final = 10
_PERIOD_COUNT: Final = 6
_CASHBAL_IDENTIFIER: Final = "CASHBAL"
_CASH_SLEEVE_FLOOR: Final = 0.04
_FIXED_INCOME_SLEEVE: Final = (
    ("CASHBAL", "Cash Balance", "Cash", 0.40, 0.00025, 1.0, 0.0),
    ("TBILL13W", "13 Week Treasury Bill", "Treasury Bills", 0.25, 0.0038, 99.35, 0.0),
    ("TNOTE2Y", "2 Year Treasury Note", "Treasury Notes", 0.20, 0.0032, 100.15, 65.0),
    ("TNOTE5Y", "5 Year Treasury Note", "Treasury Notes", 0.15, 0.0027, 98.80, 95.0),
)


def main() -> None:
    """Write first-pass operational demo data under ``_demo_output``."""
    args = _parse_args()
    source = pd.read_csv(args.source_path, parse_dates=["from_date", "thru_date"])
    performance = derive_operational_performance(
        source,
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
    equity_count: int = _EQUITY_COUNT,
    period_count: int = _PERIOD_COUNT,
    cash_sleeve_floor: float = _CASH_SLEEVE_FLOOR,
) -> pd.DataFrame:
    """Return a compact operational performance set from Mega-Cap rows.

    Args:
        source: Packaged Mega-Cap Alpha Portfolio performance rows.
        equity_count: Number of high-weight equity names to keep.
        period_count: Number of recent monthly periods to retain.
        cash_sleeve_floor: Minimum visible cash/fixed-income sleeve weight.

    Returns:
        Performance rows with selected equities plus cash, T-bill, and T-note
        synthetic rows. Weights sum to 1.0 within each period.

    Raises:
        ValueError: If source data is missing required rows or columns.
    """
    required_columns = {"from_date", "thru_date", "identifier", "weight", "return", "name"}
    missing_columns = required_columns.difference(source.columns)
    if missing_columns:
        raise ValueError(f"Source data is missing columns: {sorted(missing_columns)}")

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
    selected_equities = select_equity_identifiers(recent, equity_count)
    rows: list[dict[str, object]] = []
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
                period_rows["identifier"].eq(_CASHBAL_IDENTIFIER),
                "weight",
            ].sum()
        )
        sleeve_weight = max(original_cash_weight, cash_sleeve_floor)
        equity_scale = (1.0 - sleeve_weight) / float(equity_rows["weight"].sum())
        for _, equity in equity_rows.sort_values("identifier").iterrows():
            rows.append(
                {
                    "from_date": period.from_date,
                    "thru_date": period.thru_date,
                    "identifier": equity["identifier"],
                    "weight": float(equity["weight"]) * equity_scale,
                    "return": float(equity["return"]),
                    "name": equity["name"],
                    "asset_class": "Equity",
                    "sector": "Equity",
                    "source": "Mega-Cap Alpha Portfolio",
                }
            )
        for identifier, name, asset_class, share, synthetic_return, _, _ in _FIXED_INCOME_SLEEVE:
            rows.append(
                {
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
    return performance.sort_values(["from_date", "identifier"]).reset_index(drop=True)


def select_equity_identifiers(source: pd.DataFrame, equity_count: int) -> list[str]:
    """Return high-weight non-cash identifiers from source rows."""
    if equity_count <= 0:
        raise ValueError("equity_count must be positive.")
    candidates = source[~source["identifier"].eq(_CASHBAL_IDENTIFIER)]
    identifiers = (
        candidates.groupby("identifier", as_index=False)["weight"]
        .mean()
        .sort_values(["weight", "identifier"], ascending=[False, True])
        .head(equity_count)["identifier"]
        .tolist()
    )
    if len(identifiers) < equity_count:
        raise ValueError("Source data does not contain enough equity identifiers.")
    return identifiers


def build_axys_exports(performance: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return Axys-style CSV frames from compact performance rows."""
    sec_ref = _security_master(performance)
    secperf = _security_performance(performance)
    portperf = _portfolio_performance(performance)
    positions = _positions(performance)
    prices = _prices(performance)
    cash = _cash(performance)
    transactions = _transactions(performance)
    return {
        "sec_ref": sec_ref,
        "secperf": secperf,
        "portperf": portperf,
        "positions_holdings": positions,
        "prices": prices,
        "cash": cash,
        "transactions": transactions,
    }


def build_restatement_snapshot(axys: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Return a controlled snapshot B with reviewable source-data differences.

    Args:
        axys: Snapshot A Axys-style frames from :func:`build_axys_exports`.

    Returns:
        Snapshot B frames with small deterministic restatements across prices,
        positions, cash, transactions, and performance rows.
    """
    snapshot = {name: frame.copy(deep=True) for name, frame in axys.items()}
    latest_date = snapshot["portperf"]["THRU_DATE"].max()

    _adjust_latest_price(snapshot["prices"], latest_date, "AAPL", 1.02)
    _adjust_latest_position(snapshot["positions_holdings"], "NVDA", quantity_multiplier=1.01)
    _adjust_latest_cost(snapshot["positions_holdings"], "NVDA", cost_delta=100.0)
    _adjust_latest_accrual(snapshot["positions_holdings"], "TNOTE2Y", accrued_delta=25.0)
    _adjust_latest_cash(snapshot["cash"], latest_date, cash_delta=1_500.0)
    _adjust_transaction_amount(snapshot["transactions"], "DIV", amount_delta=125.0)
    _adjust_security_performance(snapshot["secperf"], latest_date, "AAPL", return_delta=0.0020)
    _adjust_security_performance(snapshot["secperf"], latest_date, "TNOTE2Y", return_delta=0.0005)
    _recalculate_portfolio_performance_from_security(snapshot["portperf"], snapshot["secperf"])
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
    schema_path = output_directory / "axys_column_mappings.yaml"
    schema_path.write_text(_AXYS_SCHEMA_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    paths["axys_schema"] = str(schema_path)
    for snapshot_name, frames in (("axys_a", snapshot_a), ("axys_b", snapshot_b)):
        snapshot_directory = output_directory / snapshot_name
        snapshot_directory.mkdir(parents=True, exist_ok=True)
        for name, frame in frames.items():
            path = snapshot_directory / f"{name}.csv"
            frame.to_csv(path, index=False)
            paths[f"{snapshot_name}_{name}"] = str(path)
    portfolio_yaml_path = output_directory / "ppar_performance_comparison_portfolio.yaml"
    security_yaml_path = output_directory / "ppar_performance_comparison_security.yaml"
    portfolio_yaml_path.write_text(_comparison_yaml(level=None), encoding="utf-8")
    security_yaml_path.write_text(_comparison_yaml(level="security"), encoding="utf-8")
    paths["portfolio_comparison_yaml"] = str(portfolio_yaml_path)
    paths["security_comparison_yaml"] = str(security_yaml_path)
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
        "portfolio_code": _PORTFOLIO_CODE,
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
                "PORTFOLIO_CODE": _PORTFOLIO_CODE,
                "PORTFOLIO_NAME": _PORTFOLIO_NAME,
                "SECURITY_ID": row["identifier"],
                "SECURITY_NAME": row["name"],
                "ASSET_CLASS": row["asset_class"],
                "SECTOR": row["sector"],
                "COUNTRY": "US",
                "CURRENCY": "USD",
                "PERIOD_ID": "",
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
    for period, group in performance.groupby(["from_date", "thru_date"], sort=True):
        from_date, thru_date = period
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
                "PORTFOLIO_CODE": _PORTFOLIO_CODE,
                "PORTFOLIO_NAME": _PORTFOLIO_NAME,
                "PERIOD_ID": "",
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
    latest = performance[performance["thru_date"].eq(performance["thru_date"].max())]
    for row in latest.itertuples(index=False):
        price = _price_for(row.identifier)
        market_value = _BASE_MARKET_VALUE * float(row.weight)
        rows.append(
            {
                "PORT": _PORTFOLIO_CODE,
                "SEC": row.identifier,
                "POSITION_DATE": row.thru_date.date(),
                "QTY": round(market_value / price, 4),
                "MKT_VAL": round(market_value, 2),
                "COST": round(market_value * 0.985, 2),
                "ACCRUED": round(_accrued_for(row.identifier, market_value), 2),
            }
        )
    return pd.DataFrame(rows)


def _prices(performance: pd.DataFrame) -> pd.DataFrame:
    dates = performance["thru_date"].drop_duplicates().sort_values()
    identifiers = sorted(performance["identifier"].unique())
    rows = []
    for date_index, date in enumerate(dates):
        for identifier in identifiers:
            rows.append(
                {
                    "SEC": identifier,
                    "PRICE_DATE": date.date(),
                    "PRICE": round(_price_for(identifier) * (1.0 + date_index * 0.002), 4),
                }
            )
    return pd.DataFrame(rows)


def _cash(performance: pd.DataFrame) -> pd.DataFrame:
    cash_rows = performance[performance["identifier"].eq(_CASHBAL_IDENTIFIER)]
    return pd.DataFrame(
        {
            "PORT": _PORTFOLIO_CODE,
            "CASH_DATE": cash_rows["thru_date"].dt.date,
            "CURRENCY": "USD",
            "CASH_BALANCE": (cash_rows["weight"] * _BASE_MARKET_VALUE).round(2),
            "MARKET_VALUE": (cash_rows["weight"] * _BASE_MARKET_VALUE).round(2),
        }
    )


def _transactions(performance: pd.DataFrame) -> pd.DataFrame:
    first_period = performance["from_date"].min()
    equities = (
        performance[performance["asset_class"].eq("Equity")]
        .drop_duplicates("identifier")
        .sort_values("identifier")
        .head(2)
    )
    rows = []
    for index, row in enumerate(equities.itertuples(index=False), start=1):
        amount = 10_000.0 * index
        rows.append(
            {
                "TRANSACTION_ID": f"OPS{index:03d}",
                "PORT": _PORTFOLIO_CODE,
                "TRANSACTION_DATE": first_period.date(),
                "SETTLE_DATE": (first_period + pd.Timedelta(days=1)).date(),
                "SEC": row.identifier,
                "TRAN": "BUY" if index == 1 else "DIV",
                "QTY": round(amount / _price_for(row.identifier), 4) if index == 1 else 0.0,
                "PRICE": _price_for(row.identifier) if index == 1 else 0.0,
                "AMOUNT": -amount if index == 1 else round(amount * 0.0125, 2),
                "COMMISSION": 4.95 if index == 1 else 0.0,
                "BROKER": "DEMO",
            }
        )
    return pd.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, default=_DEFAULT_SOURCE_PATH)
    parser.add_argument("--output-directory", type=Path, default=_DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--equity-count", type=int, default=_EQUITY_COUNT)
    parser.add_argument("--period-count", type=int, default=_PERIOD_COUNT)
    parser.add_argument("--cash-sleeve-floor", type=float, default=_CASH_SLEEVE_FLOOR)
    return parser.parse_args()


def _validate_period_weights(performance: pd.DataFrame) -> None:
    errors = performance.groupby(["from_date", "thru_date"])["weight"].sum().sub(1.0).abs()
    max_error = float(errors.max())
    if max_error > 1e-12:
        raise ValueError(f"Generated period weights do not sum to 1.0: {max_error}")


def _adjust_latest_price(
    prices: pd.DataFrame,
    latest_date: object,
    identifier: str,
    multiplier: float,
) -> None:
    """Apply one latest-period price restatement in place."""
    mask = prices["PRICE_DATE"].astype(str).eq(str(latest_date)) & prices["SEC"].eq(identifier)
    prices.loc[mask, "PRICE"] = (prices.loc[mask, "PRICE"] * multiplier).round(4)


def _adjust_latest_position(
    positions: pd.DataFrame,
    identifier: str,
    *,
    quantity_multiplier: float,
) -> None:
    """Apply one position quantity and market-value restatement in place."""
    mask = positions["SEC"].eq(identifier)
    positions.loc[mask, "QTY"] = (positions.loc[mask, "QTY"] * quantity_multiplier).round(4)
    positions.loc[mask, "MKT_VAL"] = (
        positions.loc[mask, "MKT_VAL"] * quantity_multiplier
    ).round(2)


def _adjust_latest_accrual(
    positions: pd.DataFrame,
    identifier: str,
    *,
    accrued_delta: float,
) -> None:
    """Apply one fixed-income accrued-interest restatement in place."""
    mask = positions["SEC"].eq(identifier)
    positions.loc[mask, "ACCRUED"] = (positions.loc[mask, "ACCRUED"] + accrued_delta).round(2)


def _adjust_latest_cost(
    positions: pd.DataFrame,
    identifier: str,
    *,
    cost_delta: float,
) -> None:
    """Apply one evidence-only position cost restatement in place."""
    mask = positions["SEC"].eq(identifier)
    positions.loc[mask, "COST"] = (positions.loc[mask, "COST"] + cost_delta).round(2)


def _adjust_latest_cash(cash: pd.DataFrame, latest_date: object, *, cash_delta: float) -> None:
    """Apply one latest-period cash balance restatement in place."""
    mask = cash["CASH_DATE"].astype(str).eq(str(latest_date))
    for column in ("CASH_BALANCE", "MARKET_VALUE"):
        cash.loc[mask, column] = (cash.loc[mask, column] + cash_delta).round(2)


def _adjust_transaction_amount(
    transactions: pd.DataFrame,
    transaction_code: str,
    *,
    amount_delta: float,
) -> None:
    """Apply one transaction amount restatement in place."""
    mask = transactions["TRAN"].eq(transaction_code)
    first_index = transactions.index[mask][0]
    transactions.loc[first_index, "AMOUNT"] = round(
        float(transactions.loc[first_index, "AMOUNT"]) + amount_delta,
        2,
    )


def _adjust_security_performance(
    security_performance: pd.DataFrame,
    latest_date: object,
    identifier: str,
    *,
    return_delta: float,
) -> None:
    """Apply one security-performance restatement in place."""
    mask = (
        security_performance["THRU_DATE"].astype(str).eq(str(latest_date))
        & security_performance["SECURITY_ID"].eq(identifier)
    )
    if not bool(mask.any()):
        raise ValueError(f"Could not find latest security row for {identifier}.")
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


def _recalculate_portfolio_performance_from_security(
    portfolio_performance: pd.DataFrame,
    security_performance: pd.DataFrame,
) -> None:
    """Recalculate portfolio rows from security rows in place."""
    grouped = security_performance.groupby(["FROM_DATE", "THRU_DATE"], sort=True)
    for (from_date, thru_date), group in grouped:
        mask = (
            portfolio_performance["FROM_DATE"].astype(str).eq(str(from_date))
            & portfolio_performance["THRU_DATE"].astype(str).eq(str(thru_date))
        )
        if not bool(mask.any()):
            continue
        income = float(group["INCOME"].astype(float).sum())
        gain_loss = float(group["GAIN_LOSS"].astype(float).sum())
        portfolio_return = float(group["CONTRIBUTION"].astype(float).sum())
        current_return = float(portfolio_performance.loc[mask, "PORT_RETURN"].iloc[0])
        if abs(portfolio_return - current_return) < 1e-10:
            continue
        portfolio_performance.loc[mask, "INCOME"] = round(income, 2)
        portfolio_performance.loc[mask, "GAIN_LOSS"] = round(gain_loss, 2)
        portfolio_performance.loc[mask, "END_MV"] = round(
            _BASE_MARKET_VALUE * (1.0 + portfolio_return),
            2,
        )
        portfolio_performance.loc[mask, "PORT_RETURN"] = round(portfolio_return, 10)
        portfolio_performance.loc[mask, "SEC_SUM_WT_X_RET"] = round(portfolio_return, 10)


def _comparison_yaml(level: str | None) -> str:
    """Return generated performance-comparison YAML text."""
    level_line = "" if level is None else f"  level: {level}\n"
    return f"""# Generated operational performance-comparison prototype.
comparison:
  name: Mega-Cap operational performance comparison prototype
{level_line}
snapshots:
  a:
    label: operational_axys_a
    path: axys_a
    vendor: axys
    schema: axys_column_mappings.yaml

  b:
    label: operational_axys_b
    path: axys_b
    vendor: axys
    schema: axys_column_mappings.yaml

files:
  portfolio_performance: portperf.csv
  security_performance: secperf.csv
  positions: positions_holdings.csv
  prices: prices.csv
  transactions: transactions.csv
  cash: cash.csv

contribution_impact_methods:
  portfolio_source_field:
    method: source_field_delta_over_begin_market_value
    denominator_source: begin_market_value
    source_fields:
      - income
      - gain_loss
  security_contribution:
    method: vendor_contribution_delta
  security_return:
    method: security_return_delta_times_weight
    weight_source: snapshot_a_weight

position_impact_methods:
  market_value:
    method: market_value_delta_over_return_denominator
    denominator_source: begin_market_value
  accrued:
    method: accrued_delta_over_return_denominator
    denominator_source: begin_market_value
  quantity:
    method: quantity_delta_times_snapshot_a_unit_market_value_over_return_denominator
    denominator_source: begin_market_value
  cost:
    method: evidence_only

price_impact_methods:
  price:
    method: price_delta_over_snapshot_a_price_times_weight
    weight_source: snapshot_a_weight

cash_impact_methods:
  cash_balance:
    method: cash_delta_over_return_denominator
    denominator_source: begin_market_value
  market_value:
    method: cash_delta_over_return_denominator
    denominator_source: begin_market_value

transaction_rules:
  BUY:
    transaction_category: buy
    cash_flow_sign: negative
    performance_flow_sign: performance
  DIV:
    transaction_category: income
    cash_flow_sign: positive
    performance_flow_sign: performance

transaction_impact_methods:
  external_flow:
    method: evidence_only
  performance:
    method: transaction_amount_delta_over_return_denominator
    denominator_source: begin_market_value
  quantity:
    method: evidence_only
  price:
    method: evidence_only
  commission:
    method: evidence_only

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
        "Treasury Bills": "TBILL",
        "Treasury Notes": "TNOTE",
    }.get(asset_class, "OTHER")


def _sector_code(sector: str) -> str:
    return {
        "Cash": "CA",
        "Equity": "EQ",
        "Treasury Bills": "TB",
        "Treasury Notes": "TN",
    }.get(sector, "OT")


def _price_for(identifier: str) -> float:
    for sleeve_identifier, _, _, _, _, price, _ in _FIXED_INCOME_SLEEVE:
        if identifier == sleeve_identifier:
            return price
    return 100.0 + (sum(ord(char) for char in identifier) % 75)


def _accrued_for(identifier: str, market_value: float) -> float:
    for sleeve_identifier, _, _, _, _, _, accrued_per_million in _FIXED_INCOME_SLEEVE:
        if identifier == sleeve_identifier:
            return market_value / 1_000_000.0 * accrued_per_million
    return 0.0


def _income_component(identifier: str, security_return: float) -> float:
    if identifier in {"TBILL13W", "TNOTE2Y", "TNOTE5Y", _CASHBAL_IDENTIFIER}:
        return min(security_return, max(security_return, 0.0))
    return 0.0


if __name__ == "__main__":
    main()
