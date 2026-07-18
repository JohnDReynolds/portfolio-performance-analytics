"""Refresh packaged Audit baseline prices and position roll-forwards.

The refresh preserves the curated review-period structure and intentional
Snapshot A anomalies. It updates normal market observations from the shared
yFinance cache, rolls quantities and cash through baseline transactions, and
recalibrates holding-scenario deltas before the ordinary Audit rebuild runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import pandas as pd

from ppar._demo_market_data import ensure_market_history, price_on_or_before


_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
_DEFAULT_AUDIT_DIRECTORY: Final = (
    _REPO_ROOT / "ppar" / "setup_templates" / "axys_apx_audit"
)
_DEFAULT_MARKET_HISTORY_PATH: Final = (
    _REPO_ROOT / "_demo_output" / "demo_market_data" / "yfinance_market_history.csv"
)
_DEFAULT_HOLDING_SCENARIOS_PATH: Final = (
    Path(__file__).resolve().parent / "audit_holding_scenarios.csv"
)
_DEFAULT_TRANSACTION_SCENARIOS_PATH: Final = (
    Path(__file__).resolve().parent / "audit_transaction_scenarios.csv"
)
_MARKET_PROXY_BY_IDENTIFIER: Final = {
    "CASHUSD": "BIL",
    "912797AA1": "BIL",
    "91282Y2Y1": "SHY",
    "91282Y5Y1": "IEI",
    "36225MBS1": "MBB",
}
_NON_MARKET_IDENTIFIERS: Final = {"CASHUSD", "CASHEUR", "CASHGBP"}
_QUANTITY_SIGNS: Final = {"by": 1.0, "cs": 1.0, "sl": -1.0, "ss": -1.0}
_CONTRIBUTION_HOLDINGS: Final = {
    "2026-02-28": 100_000.0,
    "2026-03-31": 101_000.0,
}


def main() -> None:
    """Refresh the shared cache and optionally rewrite packaged baseline rows."""
    args = _parse_args()
    snapshot_a = args.audit_directory / "snapshot_a"
    holdings = pd.read_csv(snapshot_a / "holdings.csv")
    transactions = pd.read_csv(snapshot_a / "transactions.csv")
    identifiers = sorted(
        set(holdings["SEC"]).union(transactions["SEC"]).difference(
            _NON_MARKET_IDENTIFIERS
        )
    )
    identifier_to_symbol = {
        identifier: _MARKET_PROXY_BY_IDENTIFIER.get(identifier, identifier)
        for identifier in identifiers
    }
    dates = pd.concat(
        [
            pd.to_datetime(holdings["HOLDING_DATE"]),
            pd.to_datetime(transactions["TRANSACTION_DATE"]),
        ],
        ignore_index=True,
    )
    market_history = ensure_market_history(
        args.market_history_path,
        identifier_to_symbol,
        start=dates.min(),
        end=dates.max(),
        refresh=args.refresh_market_history,
    )
    refreshed_transactions = refresh_transactions(transactions, market_history)
    refreshed_holdings = refresh_holdings(
        holdings,
        refreshed_transactions,
        market_history,
    )
    refreshed_scenarios = recalibrate_holding_scenarios(
        pd.read_csv(args.holding_scenarios_path),
        refreshed_holdings,
    )
    refreshed_transaction_scenarios = recalibrate_transaction_scenarios(
        pd.read_csv(args.transaction_scenarios_path),
        refreshed_transactions,
        market_history,
    )
    if args.write:
        refreshed_transactions.to_csv(snapshot_a / "transactions.csv", index=False)
        refreshed_holdings.to_csv(snapshot_a / "holdings.csv", index=False)
        refreshed_scenarios.to_csv(args.holding_scenarios_path, index=False)
        refreshed_transaction_scenarios.to_csv(
            args.transaction_scenarios_path,
            index=False,
        )
    print(
        {
            "mode": "write" if args.write else "check",
            "holding_rows": len(refreshed_holdings),
            "transaction_rows": len(refreshed_transactions),
            "scenario_rows": len(refreshed_scenarios),
            "transaction_scenario_rows": len(refreshed_transaction_scenarios),
            "market_history_path": str(args.market_history_path),
        }
    )


def refresh_transactions(
    transactions: pd.DataFrame,
    market_history: pd.DataFrame,
) -> pd.DataFrame:
    """Return baseline trades marked at their date-specific as-traded closes."""
    output = transactions.copy()
    transaction_dates = pd.to_datetime(output["TRANSACTION_DATE"])
    for index, row in output.iterrows():
        identifier = str(row["SEC"])
        transaction_code = str(row["TRAN"]).lower()
        if identifier in _NON_MARKET_IDENTIFIERS or identifier not in set(
            market_history["identifier"]
        ):
            continue
        if transaction_code not in _QUANTITY_SIGNS or float(row["PRICE"]) <= 0.0:
            continue
        price = price_on_or_before(market_history, identifier, transaction_dates[index])
        quantity = float(row["QTY"])
        commission = float(row["COMMISSION"])
        rounded_price = round(price, 4)
        output.loc[index, "PRICE"] = rounded_price
        if transaction_code in {"by", "cs"}:
            gross_cash = max(abs(float(row["AMOUNT"])) - commission, 0.0)
            quantity = gross_cash / rounded_price if gross_cash else quantity
            quantity = round(quantity, 4)
            amount = -(quantity * rounded_price + commission)
        else:
            amount = quantity * rounded_price - commission
        output.loc[index, "QTY"] = quantity
        output.loc[index, "AMOUNT"] = round(amount, 2)
        if "BASE_AMOUNT" in output.columns and str(row["CURRENCY"]) == str(
            row["BASE_CURRENCY"]
        ):
            output.loc[index, "BASE_AMOUNT"] = round(amount, 2)
    return output


def refresh_holdings(
    holdings: pd.DataFrame,
    transactions: pd.DataFrame,
    market_history: pd.DataFrame,
) -> pd.DataFrame:
    """Return Snapshot A holdings rolled from opening balances and activity."""
    output = holdings.copy()
    output["HOLDING_DATE"] = pd.to_datetime(output["HOLDING_DATE"])
    prepared_transactions = transactions.copy()
    prepared_transactions["TRANSACTION_DATE"] = pd.to_datetime(
        prepared_transactions["TRANSACTION_DATE"]
    )
    refreshed_parts: list[pd.DataFrame] = []
    for (portfolio, identifier), rows in output.groupby(["PORT", "SEC"], sort=False):
        rows = rows.sort_values("HOLDING_DATE").copy()
        if portfolio == "BALANCED_CONTRIBUTION":
            rows["QTY"] = rows["HOLDING_DATE"].dt.strftime("%Y-%m-%d").map(
                _CONTRIBUTION_HOLDINGS
            )
            rows["PRICE"] = 1.0
            rows["MKT_VAL"] = rows["QTY"]
            rows["BASE_MKT_VAL"] = rows["QTY"]
            rows["ACCRUED"] = 0.0
            refreshed_parts.append(rows)
            continue
        first = rows.iloc[0]
        is_cash = identifier in _NON_MARKET_IDENTIFIERS
        first_price = (
            1.0
            if is_cash
            else price_on_or_before(market_history, identifier, first["HOLDING_DATE"])
        )
        quantity = float(first["MKT_VAL"]) / first_price
        previous_date = pd.Timestamp(first["HOLDING_DATE"])
        for row_index, row in rows.iterrows():
            holding_date = pd.Timestamp(row["HOLDING_DATE"])
            if holding_date > previous_date:
                interval = prepared_transactions.loc[
                    prepared_transactions["PORT"].eq(portfolio)
                    & prepared_transactions["TRANSACTION_DATE"].gt(previous_date)
                    & prepared_transactions["TRANSACTION_DATE"].le(holding_date)
                ]
                if is_cash:
                    quantity += _cash_change(interval, identifier)
                else:
                    security_activity = interval.loc[interval["SEC"].eq(identifier)]
                    signs = security_activity["TRAN"].str.lower().map(_QUANTITY_SIGNS)
                    quantity += float(
                        (
                            pd.to_numeric(security_activity["QTY"], errors="coerce")
                            * signs.fillna(0.0)
                        ).sum()
                    )
            actual_price = (
                1.0
                if is_cash
                else price_on_or_before(market_history, identifier, holding_date)
            )
            displayed_price = 0.0 if float(row["PRICE"]) <= 0.0 else actual_price
            market_value = quantity * actual_price
            base_ratio = _base_value_ratio(row)
            accrued_ratio = _accrued_value_ratio(row)
            rows.loc[row_index, "QTY"] = round(quantity, 4)
            rows.loc[row_index, "PRICE"] = round(displayed_price, 4)
            rows.loc[row_index, "MKT_VAL"] = round(market_value, 2)
            rows.loc[row_index, "BASE_MKT_VAL"] = round(market_value * base_ratio, 2)
            rows.loc[row_index, "ACCRUED"] = round(market_value * accrued_ratio, 2)
            previous_date = holding_date
        refreshed_parts.append(rows)
    refreshed = pd.concat(refreshed_parts, ignore_index=True)
    refreshed["HOLDING_DATE"] = refreshed["HOLDING_DATE"].dt.date
    return refreshed.loc[:, holdings.columns]


def recalibrate_holding_scenarios(
    scenarios: pd.DataFrame,
    baseline_holdings: pd.DataFrame,
) -> pd.DataFrame:
    """Return holding corrections calibrated to refreshed prices and quantities."""
    output = scenarios.copy()
    for index, scenario in output.iterrows():
        mask = (
            baseline_holdings["PORT"].eq(scenario["PORT"])
            & baseline_holdings["SEC"].eq(scenario["SEC"])
            & baseline_holdings["HOLDING_DATE"].astype(str).eq(
                str(scenario["HOLDING_DATE"])
            )
        )
        if int(mask.sum()) != 1:
            raise ValueError(
                "Holding scenario must match one refreshed baseline row: "
                f"{scenario['PORT']}/{scenario['SEC']}/{scenario['HOLDING_DATE']}."
            )
        holding = baseline_holdings.loc[mask].iloc[0]
        price = float(holding["PRICE"])
        quantity = float(holding["QTY"])
        scenario_type = str(scenario["scenario_type"])
        if scenario_type == "valuation_mark":
            price_delta = round(price * 0.01, 4)
            output.loc[index, "PRICE_delta"] = price_delta
            output.loc[index, "MKT_VAL_delta"] = round(quantity * price_delta, 2)
        elif scenario_type == "quantity_valuation_correction":
            quantity_delta = float(scenario["QTY_delta"])
            if str(scenario["SEC"]) == "CVNA":
                quantity_delta = round(quantity * 4.0, 4)
                output.loc[index, "QTY_delta"] = quantity_delta
            output.loc[index, "MKT_VAL_delta"] = round(quantity_delta * price, 2)
        elif scenario_type == "accrual_correction":
            market_value_delta = float(scenario["MKT_VAL_delta"])
            output.loc[index, "QTY_delta"] = round(market_value_delta / price, 4)
    return output


def recalibrate_transaction_scenarios(
    scenarios: pd.DataFrame,
    baseline_transactions: pd.DataFrame,
    market_history: pd.DataFrame,
) -> pd.DataFrame:
    """Return trade scenarios calibrated to dated prices and cash amounts."""
    output = scenarios.copy()
    baseline = _with_transaction_ids(baseline_transactions).set_index("TRANSACTION_ID")
    market_identifiers = set(market_history["identifier"])
    for index, scenario in output.iterrows():
        action = str(scenario["action"])
        if action == "insert":
            identifier = str(scenario["SEC"])
            transaction_code = str(scenario["TRAN"]).lower()
            if (
                transaction_code not in _QUANTITY_SIGNS
                or identifier not in market_identifiers
                or float(scenario["PRICE"]) <= 0.0
            ):
                continue
            price = round(
                price_on_or_before(
                    market_history,
                    identifier,
                    scenario["TRANSACTION_DATE"],
                ),
                4,
            )
            quantity = float(scenario["QTY"])
            commission = float(scenario["COMMISSION"])
            amount = _trade_amount(transaction_code, quantity, price, commission)
            output.loc[index, "PRICE"] = price
            output.loc[index, "AMOUNT"] = amount
            continue
        if action != "adjust":
            continue
        transaction_id = str(scenario["TRANSACTION_ID"])
        if transaction_id not in baseline.index:
            continue
        base = baseline.loc[transaction_id]
        transaction_code = str(base["TRAN"]).lower()
        if transaction_code not in _QUANTITY_SIGNS:
            continue
        quantity = float(base["QTY"]) + float(scenario["QTY_delta"])
        price = float(base["PRICE"]) + float(scenario["PRICE_delta"])
        commission = float(base["COMMISSION"]) + float(
            scenario["COMMISSION_delta"]
        )
        amount = _trade_amount(transaction_code, quantity, price, commission)
        output.loc[index, "AMOUNT_delta"] = round(amount - float(base["AMOUNT"]), 2)
    return output


def _with_transaction_ids(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return baseline rows with the rebuild script's deterministic IDs."""
    output = transactions.copy()
    period_index_by_portfolio: dict[str, dict[pd.Period, int]] = {}
    row_count_by_period: dict[tuple[str, pd.Period], int] = {}
    identifiers: list[str] = []
    for row in output.itertuples(index=False):
        portfolio = str(row.PORT)
        month = pd.Timestamp(row.TRANSACTION_DATE).to_period("M")
        portfolio_periods = period_index_by_portfolio.setdefault(portfolio, {})
        if month not in portfolio_periods:
            portfolio_periods[month] = len(portfolio_periods) + 1
        key = (portfolio, month)
        row_count_by_period[key] = row_count_by_period.get(key, 0) + 1
        identifiers.append(
            f"{portfolio}{portfolio_periods[month]:02d}{row_count_by_period[key]:02d}"
        )
    output.insert(0, "TRANSACTION_ID", identifiers)
    return output


def _trade_amount(
    transaction_code: str,
    quantity: float,
    price: float,
    commission: float,
) -> float:
    """Return signed trade cash using the packaged transaction convention."""
    gross = quantity * price
    return round(-(gross + commission), 2) if transaction_code in {"by", "cs"} else round(
        gross - commission,
        2,
    )


def _cash_change(transactions: pd.DataFrame, cash_identifier: str) -> float:
    """Return transaction cash movement for one local-currency cash balance."""
    if cash_identifier == "CASHUSD":
        rows = transactions.loc[transactions["CURRENCY"].eq("USD")]
    elif cash_identifier == "CASHEUR":
        rows = transactions.loc[transactions["CURRENCY"].eq("EUR")]
    else:
        rows = transactions.loc[transactions["CURRENCY"].eq("GBP")]
    return float(pd.to_numeric(rows["AMOUNT"], errors="coerce").fillna(0.0).sum())


def _base_value_ratio(row: pd.Series) -> float:
    """Return the existing local-to-base valuation ratio."""
    market_value = float(row["MKT_VAL"])
    return float(row["BASE_MKT_VAL"]) / market_value if market_value else 1.0


def _accrued_value_ratio(row: pd.Series) -> float:
    """Return the existing accrued-to-market-value ratio."""
    market_value = float(row["MKT_VAL"])
    return float(row["ACCRUED"]) / market_value if market_value else 0.0


def _parse_args() -> argparse.Namespace:
    """Return command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-directory", type=Path, default=_DEFAULT_AUDIT_DIRECTORY)
    parser.add_argument(
        "--market-history-path",
        type=Path,
        default=_DEFAULT_MARKET_HISTORY_PATH,
    )
    parser.add_argument(
        "--holding-scenarios-path",
        type=Path,
        default=_DEFAULT_HOLDING_SCENARIOS_PATH,
    )
    parser.add_argument(
        "--transaction-scenarios-path",
        type=Path,
        default=_DEFAULT_TRANSACTION_SCENARIOS_PATH,
    )
    parser.add_argument("--refresh-market-history", action="store_true")
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
