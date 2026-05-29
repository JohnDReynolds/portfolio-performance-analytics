"""Define default source-column aliases for normalized comparison datasets."""

from __future__ import annotations

# Python imports
from typing import Final

# Project imports
from ppar.performance_comparison import columns as pc_cols
from ppar.performance_comparison.source_loader import ColumnAliases

PORTFOLIO_PERFORMANCE_REQUIRED_ALIASES: Final[ColumnAliases] = {
    pc_cols.PORTFOLIO_ID: (
        "PORTFOLIO_ID",
        "PORTFOLIO_CODE",
        "PORT",
        "PORTFOLIO",
        "ACCOUNT",
        "ACCT",
    ),
    pc_cols.FROM_DATE: ("FROM_DATE",),
    pc_cols.THRU_DATE: ("THRU_DATE",),
    pc_cols.PORTFOLIO_RETURN: ("PORT_RETURN", "RETURN", "RET"),
}
PORTFOLIO_PERFORMANCE_OPTIONAL_ALIASES: Final[ColumnAliases] = {
    pc_cols.PORTFOLIO_NAME: ("PORTFOLIO_NAME",),
    pc_cols.BEGIN_MARKET_VALUE: ("BEGIN_MV", "BEG_MV", "BMV", "BEGIN_VALUE"),
    pc_cols.END_MARKET_VALUE: ("END_MV", "EMV", "ENDING_VALUE"),
    pc_cols.FLOW: ("FLOW", "NET_FLOW", "CONTRIB_WITHDRAW", "CASH_FLOW"),
    pc_cols.INCOME: ("INCOME", "INC", "DIV_INT", "INV_INCOME"),
    pc_cols.GAIN_LOSS: ("GAIN_LOSS", "GL", "GAIN", "REAL_UNREAL_GL"),
    pc_cols.PERIOD_ID: ("PERIOD_ID",),
    pc_cols.CURRENCY: ("CURRENCY", "CURRENCY_CODE", "CURR", "CCY"),
}

SECURITY_PERFORMANCE_REQUIRED_ALIASES: Final[ColumnAliases] = {
    pc_cols.PORTFOLIO_ID: PORTFOLIO_PERFORMANCE_REQUIRED_ALIASES[pc_cols.PORTFOLIO_ID],
    pc_cols.SECURITY_ID: ("SECURITY_ID", "SEC", "SECURITY", "SEC_ID", "SECNO"),
    pc_cols.FROM_DATE: ("FROM_DATE",),
    pc_cols.THRU_DATE: ("THRU_DATE",),
    pc_cols.SECURITY_RETURN: ("SEC_RETURN", "RETURN", "RET"),
}
SECURITY_PERFORMANCE_OPTIONAL_ALIASES: Final[ColumnAliases] = {
    pc_cols.SECURITY_NAME: ("SECURITY_NAME",),
    pc_cols.WEIGHT: ("BEGIN_WEIGHT", "WEIGHT", "WGT", "PCT_ASSETS", "PERCENT_ASSETS"),
    pc_cols.CONTRIBUTION: (
        "CONTRIBUTION",
        "CONTRIB",
        "CTR",
        "RET_CONTRIB",
        "CONTRIBUTION_W_X_R",
    ),
    pc_cols.BEGIN_MARKET_VALUE: ("BEGIN_MV", "BEG_MV", "BMV", "BEGIN_VALUE"),
    pc_cols.END_MARKET_VALUE: ("END_MV", "EMV", "ENDING_VALUE"),
    pc_cols.INCOME: ("INCOME", "INC", "DIV_INT", "INV_INCOME"),
    pc_cols.GAIN_LOSS: ("GAIN_LOSS", "GL", "GAIN", "REAL_UNREAL_GL"),
    pc_cols.PERIOD_ID: ("PERIOD_ID",),
    pc_cols.CURRENCY: ("CURRENCY", "CURRENCY_CODE", "CURR", "CCY"),
}

SECURITY_MASTER_REQUIRED_ALIASES: Final[ColumnAliases] = {
    pc_cols.SECURITY_ID: SECURITY_PERFORMANCE_REQUIRED_ALIASES[pc_cols.SECURITY_ID],
}
SECURITY_MASTER_OPTIONAL_ALIASES: Final[ColumnAliases] = {
    pc_cols.SECURITY_NAME: (
        "SECURITY_NAME",
        "DESC",
        "DESCRIPTION",
        "NAME",
        "SEC_DESC",
    ),
    pc_cols.TICKER: ("TICKER", "SYMBOL", "TICKER_SYMBOL"),
    pc_cols.CUSIP: ("CUSIP", "CUSIP_NO", "CUSIP_NUMBER"),
    pc_cols.ISIN: ("ISIN",),
    pc_cols.CURRENCY: ("CURRENCY_CODE", "CURRENCY", "CURR", "CCY", "LOCAL_CCY"),
    pc_cols.COUNTRY: ("COUNTRY_CODE", "COUNTRY", "CNTRY", "ISSUE_COUNTRY"),
    pc_cols.SECTOR: ("SECTOR_CODE", "SECTOR"),
    pc_cols.INDUSTRY: ("INDUSTRY_CODE", "INDUSTRY", "IND"),
    pc_cols.ASSET_CLASS: (
        "ASSET_CLASS_CODE",
        "ASSET_CLASS",
        "SEC_TYPE",
        "ASSET_TYPE",
        "INV_TYPE",
    ),
}

PRICES_REQUIRED_ALIASES: Final[ColumnAliases] = {
    pc_cols.SECURITY_ID: SECURITY_PERFORMANCE_REQUIRED_ALIASES[pc_cols.SECURITY_ID],
    pc_cols.PRICE_DATE: ("PRICE_DATE",),
    pc_cols.PRICE: ("PRICE", "PX", "CLOSE_PRICE", "MARKET_PRICE"),
}
PRICES_OPTIONAL_ALIASES: Final[ColumnAliases] = {
    pc_cols.CURRENCY: ("CURRENCY", "CURRENCY_CODE", "CURR", "CCY"),
    pc_cols.PRICE_SOURCE: ("PRICE_SOURCE", "SOURCE", "SRC", "VENDOR"),
    pc_cols.PRICE_TYPE: ("PRICE_TYPE",),
}

TRANSACTIONS_REQUIRED_ALIASES: Final[ColumnAliases] = {
    pc_cols.PORTFOLIO_ID: PORTFOLIO_PERFORMANCE_REQUIRED_ALIASES[pc_cols.PORTFOLIO_ID],
    pc_cols.SECURITY_ID: SECURITY_PERFORMANCE_REQUIRED_ALIASES[pc_cols.SECURITY_ID],
    pc_cols.TRANSACTION_DATE: ("TRANSACTION_DATE", "TRADE_DATE", "TRD_DATE"),
}
TRANSACTIONS_OPTIONAL_ALIASES: Final[ColumnAliases] = {
    pc_cols.TRANSACTION_ID: ("TRANSACTION_ID", "TRAN_ID", "TXN_ID"),
    pc_cols.SETTLEMENT_DATE: (
        "SETTLEMENT_DATE",
        "SETTLE_DATE",
        "SETTLE",
        "SET_DATE",
        "STL_DATE",
    ),
    pc_cols.TRANSACTION_CODE: (
        "TRANSACTION_CODE",
        "TRAN",
        "TRAN_CODE",
        "TRANS_CODE",
        "ACTIVITY",
    ),
    pc_cols.QUANTITY: ("QUANTITY", "QTY", "UNITS", "SHARES"),
    pc_cols.PRICE: ("PRICE", "PX", "TRADE_PRICE"),
    pc_cols.AMOUNT: ("AMOUNT", "AMT", "NET_AMOUNT", "NET_AMT"),
    pc_cols.COMMISSION: ("COMMISSION", "COMM", "COMMISH"),
    pc_cols.CURRENCY: ("CURRENCY", "CURRENCY_CODE", "CURR", "CCY"),
    pc_cols.BROKER: ("BROKER", "BRKR", "BROKER_CODE"),
}

POSITIONS_REQUIRED_ALIASES: Final[ColumnAliases] = {
    pc_cols.PORTFOLIO_ID: PORTFOLIO_PERFORMANCE_REQUIRED_ALIASES[pc_cols.PORTFOLIO_ID],
    pc_cols.SECURITY_ID: SECURITY_PERFORMANCE_REQUIRED_ALIASES[pc_cols.SECURITY_ID],
    pc_cols.POSITION_DATE: ("POSITION_DATE",),
}
POSITIONS_OPTIONAL_ALIASES: Final[ColumnAliases] = {
    pc_cols.QUANTITY: ("QUANTITY", "QTY", "UNITS", "SHARES"),
    pc_cols.PRICE: ("PRICE", "PX", "MARKET_PRICE"),
    pc_cols.MARKET_VALUE: ("MARKET_VALUE", "MV", "MKT_VAL"),
    pc_cols.COST: ("COST", "COST_BASIS", "BOOK_COST", "TAX_COST", "ORIG_COST"),
    pc_cols.ACCRUED: ("ACCRUED", "ACCRUED_INCOME", "ACCRUED_INT", "ACCRUAL"),
    pc_cols.CURRENCY: ("CURRENCY", "CURRENCY_CODE", "CURR", "CCY"),
}

CASH_REQUIRED_ALIASES: Final[ColumnAliases] = {
    pc_cols.PORTFOLIO_ID: PORTFOLIO_PERFORMANCE_REQUIRED_ALIASES[pc_cols.PORTFOLIO_ID],
    pc_cols.CASH_DATE: ("CASH_DATE", "BALANCE_DATE"),
}
CASH_OPTIONAL_ALIASES: Final[ColumnAliases] = {
    pc_cols.CURRENCY: ("CURRENCY", "CURRENCY_CODE", "CURR", "CCY"),
    pc_cols.CASH_BALANCE: ("CASH_BALANCE", "CASH", "CASH_BAL", "BALANCE"),
    pc_cols.MARKET_VALUE: ("MARKET_VALUE", "MV", "BASE_VALUE"),
}
