"""Define default source-column aliases for normalized comparison datasets."""

from __future__ import annotations

# Python imports
from typing import Final

# Project imports
from ppar.performance_comparison import schema as pc_cols
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
    pc_cols.BASE_CURRENCY: ("BASE_CURRENCY", "BASE_CCY", "PORTFOLIO_CURRENCY"),
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
    pc_cols.BASE_CURRENCY: ("BASE_CURRENCY", "BASE_CURR", "BASE_CCY"),
}

SPLITS_REQUIRED_ALIASES: Final[ColumnAliases] = {
    pc_cols.SECURITY_ID: SECURITY_PERFORMANCE_REQUIRED_ALIASES[pc_cols.SECURITY_ID],
    pc_cols.SPLIT_DATE: ("SPLIT_DATE", "SPLITDATE", "EFFECTIVE_DATE"),
    pc_cols.SPLIT_FACTOR: ("SPLIT_FACTOR", "SPLITFACTOR", "FACTOR"),
}
SPLITS_OPTIONAL_ALIASES: Final[ColumnAliases] = {
    pc_cols.SECURITY_NAME: ("SECURITY_NAME",),
    pc_cols.TICKER: ("TICKER", "SYMBOL"),
}

FX_RATES_REQUIRED_ALIASES: Final[ColumnAliases] = {
    pc_cols.FROM_CURRENCY: ("FROM_CURRENCY", "FROM_CCY", "BASE_CURRENCY", "BASE_CCY"),
    pc_cols.TO_CURRENCY: ("TO_CURRENCY", "TO_CCY", "QUOTE_CURRENCY", "QUOTE_CCY"),
    pc_cols.RATE_DATE: ("RATE_DATE", "FX_DATE"),
    pc_cols.FX_RATE: ("FX_RATE", "RATE", "EXCHANGE_RATE"),
}
FX_RATES_OPTIONAL_ALIASES: Final[ColumnAliases] = {
    pc_cols.PORTFOLIO_ID: PORTFOLIO_PERFORMANCE_REQUIRED_ALIASES[pc_cols.PORTFOLIO_ID],
    pc_cols.LOCAL_EXPOSURE: (
        "LOCAL_EXPOSURE",
        "LOCAL_MARKET_VALUE",
        "LOCAL_MV",
    ),
    pc_cols.RATE_SOURCE: ("RATE_SOURCE", "SOURCE", "SRC", "VENDOR"),
    pc_cols.RATE_TYPE: ("RATE_TYPE",),
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
    pc_cols.SECURITY_TYPE: (
        "SECURITY_TYPE",
        "SEC_TYPE",
        "TRANSACTION_SECURITY_TYPE",
        "TRAN_SEC_TYPE",
    ),
    pc_cols.SOURCE_DESTINATION_TYPE: (
        "SOURCE_DESTINATION_TYPE",
        "SRC_DEST_TYPE",
        "SOURCE_DEST_TYPE",
        "SRCDESTTYPE",
    ),
    pc_cols.SOURCE_DESTINATION_SYMBOL: (
        "SOURCE_DESTINATION_SYMBOL",
        "SRC_DEST_SYMBOL",
        "SOURCE_DEST_SYMBOL",
        "SRCDESTSYMBOL",
    ),
    pc_cols.SPECIAL_SECURITY_TYPE: (
        "SPECIAL_SECURITY_TYPE",
        "SPECIAL_SEC_TYPE",
        "SPEC_SEC_TYPE",
    ),
    pc_cols.SPECIAL_SECURITY_SYMBOL: (
        "SPECIAL_SECURITY_SYMBOL",
        "SPECIAL_SEC_SYMBOL",
        "SPEC_SEC_SYMBOL",
    ),
    pc_cols.TRANSACTION_CATEGORY: (
        "TRANSACTION_CATEGORY",
        "TXN_CATEGORY",
        "ACTIVITY_CATEGORY",
    ),
    pc_cols.CASH_FLOW_SIGN: ("CASH_FLOW_SIGN",),
    pc_cols.PERFORMANCE_FLOW_SIGN: ("PERFORMANCE_FLOW_SIGN",),
    pc_cols.QUANTITY: ("QUANTITY", "QTY", "UNITS", "SHARES"),
    pc_cols.PRICE: ("PRICE", "PX", "TRADE_PRICE"),
    pc_cols.AMOUNT: ("AMOUNT", "AMT", "NET_AMOUNT", "NET_AMT"),
    pc_cols.BASE_AMOUNT: ("BASE_AMOUNT", "BASE_AMT", "AMOUNT_BASE"),
    pc_cols.COMMISSION: ("COMMISSION", "COMM", "COMMISH"),
    pc_cols.CURRENCY: ("CURRENCY", "CURRENCY_CODE", "CURR", "CCY"),
    pc_cols.BASE_CURRENCY: ("BASE_CURRENCY", "BASE_CCY", "PORTFOLIO_CURRENCY"),
    pc_cols.BROKER: ("BROKER", "BRKR", "BROKER_CODE"),
}

HOLDINGS_REQUIRED_ALIASES: Final[ColumnAliases] = {
    pc_cols.PORTFOLIO_ID: PORTFOLIO_PERFORMANCE_REQUIRED_ALIASES[pc_cols.PORTFOLIO_ID],
    pc_cols.SECURITY_ID: SECURITY_PERFORMANCE_REQUIRED_ALIASES[pc_cols.SECURITY_ID],
    pc_cols.HOLDING_DATE: ("HOLDING_DATE", "POSITION_DATE"),
}
HOLDINGS_OPTIONAL_ALIASES: Final[ColumnAliases] = {
    pc_cols.QUANTITY: ("QUANTITY", "QTY", "UNITS", "SHARES"),
    pc_cols.PRICE: ("PRICE", "PX", "MARKET_PRICE"),
    pc_cols.MARKET_VALUE: ("MARKET_VALUE", "MV", "MKT_VAL"),
    pc_cols.BASE_MARKET_VALUE: (
        "BASE_MARKET_VALUE",
        "BASE_MV",
        "BASE_MKT_VAL",
    ),
    pc_cols.COST: ("COST", "COST_BASIS", "BOOK_COST", "TAX_COST", "ORIG_COST"),
    pc_cols.ACCRUED: ("ACCRUED", "ACCRUED_INCOME", "ACCRUED_INT", "ACCRUAL"),
    pc_cols.CURRENCY: ("CURRENCY", "CURRENCY_CODE", "CURR", "CCY"),
    pc_cols.BASE_CURRENCY: ("BASE_CURRENCY", "BASE_CCY", "PORTFOLIO_CURRENCY"),
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
