"""Build and validate holdings for the maintained operational demo.

This internal maintainer module keeps position roll-forward mechanics separate
from the scenario construction performed by the operational demo generator.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

from scripts.demo_support.market_data import price_on_or_before


_BASE_MARKET_VALUE: Final = 1_000_000.0
_CASH_IDENTIFIER: Final = "CASHUSD"
_ACCRUED_PER_UNIT: Final = {
    "91282Y2Y1": 0.065,
    "91282Y5Y1": 0.095,
    "36225MBS1": 0.120,
}


def build_operational_holdings(
    performance: pd.DataFrame,
    market_history: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Build holdings and validate price and quantity behavior.

    Args:
        performance: Operational security performance rows.
        market_history: Shared normalized market history.
        transactions: Axys/APX-style operational transactions.

    Returns:
        Axys/APX-style dated holding rows.

    Raises:
        ValueError: If quantities do not roll through trades and splits or an
            equity has an unexplained multi-period constant price.
    """
    holdings = _positions(performance, market_history, transactions)
    _validate_holding_rollforward(holdings, transactions, market_history)
    _validate_equity_price_variation(holdings, performance)
    return holdings


def _positions(
    performance: pd.DataFrame,
    market_history: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Roll holdings through dated trades, cash activity, and stock splits."""
    rows: list[dict[str, object]] = []
    transaction_rows = transactions.copy()
    transaction_rows["TRANSACTION_DATE"] = pd.to_datetime(
        transaction_rows["TRANSACTION_DATE"]
    )
    for portfolio_key, portfolio_rows in performance.groupby(
        "portfolio_code",
        sort=True,
    ):
        portfolio_code = str(portfolio_key)
        dates = sorted(portfolio_rows["thru_date"].unique())
        securities = sorted(portfolio_rows["identifier"].unique())
        for identifier_value in securities:
            identifier = str(identifier_value)
            security_rows = portfolio_rows.loc[
                portfolio_rows["identifier"].eq(identifier)
            ].sort_values("thru_date")
            first = security_rows.iloc[0]
            first_date = _timestamp(dates[0])
            first_price = (
                1.0
                if identifier == _CASH_IDENTIFIER
                else price_on_or_before(market_history, identifier, first_date)
            )
            quantity = _BASE_MARKET_VALUE * _as_float(first["weight"]) / first_price
            previous_date = first_date
            for date_index, holding_date_value in enumerate(dates):
                holding_date = _timestamp(holding_date_value)
                if date_index:
                    if identifier == _CASH_IDENTIFIER:
                        quantity += _cash_change_between(
                            transaction_rows,
                            portfolio_code,
                            previous_date,
                            holding_date,
                        )
                    else:
                        quantity *= _split_factor_between(
                            market_history,
                            identifier,
                            previous_date,
                            holding_date,
                        )
                        quantity += _quantity_change_between(
                            transaction_rows,
                            portfolio_code,
                            identifier,
                            previous_date,
                            holding_date,
                        )
                price = (
                    1.0
                    if identifier == _CASH_IDENTIFIER
                    else price_on_or_before(market_history, identifier, holding_date)
                )
                market_value = quantity * price
                rows.append(
                    {
                        "PORT": portfolio_code,
                        "SEC": identifier,
                        "HOLDING_DATE": holding_date.date(),
                        "QTY": round(quantity, 4),
                        "PRICE": round(price, 4),
                        "MKT_VAL": round(market_value, 2),
                        "COST": round(market_value * 0.985, 2),
                        "ACCRUED": round(
                            accrued_income_for(identifier, quantity),
                            2,
                        ),
                    }
                )
                previous_date = holding_date
    return pd.DataFrame(rows)


def _split_factor_between(
    market_history: pd.DataFrame,
    identifier: str,
    previous_date: pd.Timestamp,
    holding_date: pd.Timestamp,
) -> float:
    """Return the cumulative reported split factor between holding dates."""
    actions = market_history.loc[
        market_history["identifier"].eq(identifier)
        & market_history["date"].gt(previous_date)
        & market_history["date"].le(holding_date)
        & market_history["split_factor"].gt(0.0),
        "split_factor",
    ]
    return _as_float(actions.prod()) if not actions.empty else 1.0


def _quantity_change_between(
    transactions: pd.DataFrame,
    portfolio_code: str,
    identifier: str,
    previous_date: pd.Timestamp,
    holding_date: pd.Timestamp,
) -> float:
    """Return signed trade quantity between two holding dates."""
    rows = transactions.loc[
        transactions["PORT"].eq(portfolio_code)
        & transactions["SEC"].eq(identifier)
        & transactions["TRANSACTION_DATE"].gt(previous_date)
        & transactions["TRANSACTION_DATE"].le(holding_date)
    ]
    signs = rows["TRAN"].map({"by": 1.0, "cs": 1.0, "sl": -1.0, "ss": -1.0})
    return float((rows["QTY"] * signs.fillna(0.0)).sum())


def _cash_change_between(
    transactions: pd.DataFrame,
    portfolio_code: str,
    previous_date: pd.Timestamp,
    holding_date: pd.Timestamp,
) -> float:
    """Return net base-currency cash activity between holding dates."""
    rows = transactions.loc[
        transactions["PORT"].eq(portfolio_code)
        & transactions["TRANSACTION_DATE"].gt(previous_date)
        & transactions["TRANSACTION_DATE"].le(holding_date)
    ]
    return float(pd.to_numeric(rows["AMOUNT"], errors="coerce").fillna(0.0).sum())


def _validate_holding_rollforward(
    holdings: pd.DataFrame,
    transactions: pd.DataFrame,
    market_history: pd.DataFrame,
) -> None:
    """Fail when a holding quantity is not explained by trades or splits."""
    prepared_transactions = transactions.copy()
    prepared_transactions["TRANSACTION_DATE"] = pd.to_datetime(
        prepared_transactions["TRANSACTION_DATE"]
    )
    failures: list[dict[str, object]] = []
    for (portfolio_key, identifier_key), rows in holdings.groupby(["PORT", "SEC"]):
        portfolio_code = str(portfolio_key)
        identifier = str(identifier_key)
        rows = rows.sort_values("HOLDING_DATE")
        previous = None
        for row in rows.itertuples(index=False):
            if previous is None:
                previous = row
                continue
            previous_date = _timestamp(previous.HOLDING_DATE)
            holding_date = _timestamp(row.HOLDING_DATE)
            if identifier == _CASH_IDENTIFIER:
                expected = _as_float(previous.QTY) + _cash_change_between(
                    prepared_transactions,
                    portfolio_code,
                    previous_date,
                    holding_date,
                )
            else:
                expected = _as_float(previous.QTY) * _split_factor_between(
                    market_history,
                    identifier,
                    previous_date,
                    holding_date,
                )
                expected += _quantity_change_between(
                    prepared_transactions,
                    portfolio_code,
                    identifier,
                    previous_date,
                    holding_date,
                )
            actual_quantity = _as_float(row.QTY)
            difference = abs(actual_quantity - expected)
            if difference > 0.001:
                failures.append(
                    {
                        "portfolio": portfolio_code,
                        "identifier": identifier,
                        "holding_date": holding_date.date(),
                        "expected_quantity": expected,
                        "actual_quantity": actual_quantity,
                        "difference": difference,
                    }
                )
            previous = row
    if failures:
        raise ValueError(f"Holding quantity roll-forward failures: {failures[:10]}")


def _validate_equity_price_variation(
    holdings: pd.DataFrame,
    performance: pd.DataFrame,
) -> None:
    """Fail on repeated multi-period constant prices for public equities."""
    equity_identifiers = set(
        performance.loc[performance["asset_class"].eq("Equity"), "identifier"]
    )
    failures: list[dict[str, object]] = []
    for identifier, rows in holdings.loc[
        holdings["SEC"].isin(equity_identifiers)
    ].groupby("SEC"):
        prices = (
            rows.drop_duplicates("HOLDING_DATE")
            .sort_values("HOLDING_DATE")["PRICE"]
            .astype(float)
        )
        repeated = prices.diff().eq(0.0)
        if bool(repeated.rolling(2).sum().ge(2).any()):
            failures.append({"identifier": identifier, "prices": prices.tolist()})
    if failures:
        raise ValueError(f"Unexplained multi-period constant equity prices: {failures}")


def accrued_income_for(identifier: str, quantity: float) -> float:
    """Return consistent accrued income for one fixed-income holding.

    Args:
        identifier: Demo security identifier.
        quantity: Holding units on the observation date.

    Returns:
        Accrued income based on the security's synthetic per-unit accrual rate.

    Notes:
        The Data Issues contract compares accrued income per unit across
        portfolios. A per-unit fixture therefore represents that contract more
        faithfully than the former market-value percentage, which changed with
        the proxy price and produced false cross-portfolio findings.
    """
    return quantity * _ACCRUED_PER_UNIT.get(identifier, 0.0)


def _timestamp(value: object) -> pd.Timestamp:
    """Return a Timestamp from a dynamically typed tabular value."""
    return pd.Timestamp(str(value))


def _as_float(value: object) -> float:
    """Return a float from a dynamically typed tabular value."""
    return float(str(value))
