"""Load normalized transaction comparison sources."""

from __future__ import annotations

# Third-party imports
import polars as pl

# Project imports
from ppar.performance_comparison import aliases
from ppar.performance_comparison import columns as pc_cols
from ppar.performance_comparison import source_loader
from ppar.performance_comparison.portfolio_performance import SnapshotKey
from ppar.performance_comparison.specification import PerformanceComparisonSpecification
import ppar.utilities as util

TRANSACTION_CATEGORY_EXTERNAL_FLOW = "external_flow"
TRANSACTION_CATEGORY_INCOME = "income"
TRANSACTION_CATEGORY_FEE_EXPENSE = "fee_expense"
TRANSACTION_CATEGORY_BUY = "buy"
TRANSACTION_CATEGORY_SELL = "sell"
TRANSACTION_CATEGORY_TRANSFER = "transfer"
TRANSACTION_CATEGORY_CORPORATE_ACTION = "corporate_action"
TRANSACTION_CATEGORY_UNKNOWN = "unknown"

_CATEGORY_NORMALIZATION: dict[str, str] = {
    "activity": TRANSACTION_CATEGORY_UNKNOWN,
    "buy": TRANSACTION_CATEGORY_BUY,
    "buy_security": TRANSACTION_CATEGORY_BUY,
    "cash_contribution": TRANSACTION_CATEGORY_EXTERNAL_FLOW,
    "cash_deposit": TRANSACTION_CATEGORY_EXTERNAL_FLOW,
    "cash_flow": TRANSACTION_CATEGORY_EXTERNAL_FLOW,
    "cash_withdrawal": TRANSACTION_CATEGORY_EXTERNAL_FLOW,
    "contribution": TRANSACTION_CATEGORY_EXTERNAL_FLOW,
    "corporate_action": TRANSACTION_CATEGORY_CORPORATE_ACTION,
    "deposit": TRANSACTION_CATEGORY_EXTERNAL_FLOW,
    "distribution": TRANSACTION_CATEGORY_EXTERNAL_FLOW,
    "div": TRANSACTION_CATEGORY_INCOME,
    "dividend": TRANSACTION_CATEGORY_INCOME,
    "expense": TRANSACTION_CATEGORY_FEE_EXPENSE,
    "external_flow": TRANSACTION_CATEGORY_EXTERNAL_FLOW,
    "fee": TRANSACTION_CATEGORY_FEE_EXPENSE,
    "fee_expense": TRANSACTION_CATEGORY_FEE_EXPENSE,
    "income": TRANSACTION_CATEGORY_INCOME,
    "int": TRANSACTION_CATEGORY_INCOME,
    "interest": TRANSACTION_CATEGORY_INCOME,
    "merger": TRANSACTION_CATEGORY_CORPORATE_ACTION,
    "sell": TRANSACTION_CATEGORY_SELL,
    "sell_security": TRANSACTION_CATEGORY_SELL,
    "spin": TRANSACTION_CATEGORY_CORPORATE_ACTION,
    "spinoff": TRANSACTION_CATEGORY_CORPORATE_ACTION,
    "split": TRANSACTION_CATEGORY_CORPORATE_ACTION,
    "transfer": TRANSACTION_CATEGORY_TRANSFER,
    "xfer": TRANSACTION_CATEGORY_TRANSFER,
    "withdrawal": TRANSACTION_CATEGORY_EXTERNAL_FLOW,
}

_TRANSACTION_CODE_CATEGORIES: dict[str, str] = {
    "BUY": TRANSACTION_CATEGORY_BUY,
    "SELL": TRANSACTION_CATEGORY_SELL,
    "DIV": TRANSACTION_CATEGORY_INCOME,
    "INT": TRANSACTION_CATEGORY_INCOME,
    "INCOME": TRANSACTION_CATEGORY_INCOME,
    "FEE": TRANSACTION_CATEGORY_FEE_EXPENSE,
    "EXP": TRANSACTION_CATEGORY_FEE_EXPENSE,
    "DEP": TRANSACTION_CATEGORY_EXTERNAL_FLOW,
    "WD": TRANSACTION_CATEGORY_EXTERNAL_FLOW,
    "WITHDRAWAL": TRANSACTION_CATEGORY_EXTERNAL_FLOW,
    "XFER": TRANSACTION_CATEGORY_TRANSFER,
    "TRANSFER": TRANSACTION_CATEGORY_TRANSFER,
    "SPLIT": TRANSACTION_CATEGORY_CORPORATE_ACTION,
    "MERGER": TRANSACTION_CATEGORY_CORPORATE_ACTION,
    "SPIN": TRANSACTION_CATEGORY_CORPORATE_ACTION,
}


def normalize_transaction_category(value: object) -> str:
    """Return a normalized transaction category label.

    Args:
        value: Source category or transaction code value.

    Returns:
        One of the normalized transaction category labels. Unknown, blank, and
        missing values return ``"unknown"``.
    """
    if value is None:
        return TRANSACTION_CATEGORY_UNKNOWN
    normalized_value = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized_value:
        return TRANSACTION_CATEGORY_UNKNOWN
    return _CATEGORY_NORMALIZATION.get(normalized_value, TRANSACTION_CATEGORY_UNKNOWN)


def transaction_category_from_code(value: object) -> str:
    """Return a normalized transaction category inferred from a transaction code.

    Args:
        value: Source transaction code value.

    Returns:
        One of the normalized transaction category labels. Unknown, blank, and
        missing values return ``"unknown"``.
    """
    if value is None:
        return TRANSACTION_CATEGORY_UNKNOWN
    normalized_value = str(value).strip().upper()
    if not normalized_value:
        return TRANSACTION_CATEGORY_UNKNOWN
    return _TRANSACTION_CODE_CATEGORIES.get(
        normalized_value,
        normalize_transaction_category(normalized_value),
    )


class TransactionsLoader:
    """Load normalized transaction rows for comparison snapshots.

    Attributes:
        _specification: Parsed comparison specification.
    """

    def __init__(self, specification: PerformanceComparisonSpecification) -> None:
        """Initialize the transaction loader.

        Args:
            specification: Parsed comparison specification containing resolved
                snapshot and file paths.
        """
        self._specification = specification

    def load(self, snapshot_key: SnapshotKey) -> pl.DataFrame | None:
        """Load one snapshot's normalized transaction rows.

        Args:
            snapshot_key: Snapshot side to load, either ``"a"`` or ``"b"``.

        Returns:
            Transaction rows with normalized comparison column names, or
            ``None`` when the optional dataset is omitted or missing.

        Raises:
            PpaError: If the source exists but required columns cannot be
                resolved.
        """
        path = source_loader.optional_file_path(
            self._specification,
            pc_cols.TRANSACTIONS,
            snapshot_key,
        )
        if path is None or not util.file_path_exists(path):
            return None

        frame = source_loader.read_mapped_csv(
            path,
            pc_cols.TRANSACTIONS_COLUMNS,
            pc_cols.TRANSACTIONS,
            aliases.TRANSACTIONS_REQUIRED_ALIASES,
            aliases.TRANSACTIONS_OPTIONAL_ALIASES,
            self._specification.path,
        )
        date_columns = [
            column
            for column in (pc_cols.TRANSACTION_DATE, pc_cols.SETTLEMENT_DATE)
            if column in frame.columns
        ]
        frame = frame.with_columns(
            pl.col(date_columns).str.strptime(pl.Date, "%Y-%m-%d", strict=True),
        )
        return _with_transaction_category(frame)


def _with_transaction_category(frame: pl.DataFrame) -> pl.DataFrame:
    """Return transaction rows with a normalized transaction category column."""
    if pc_cols.TRANSACTION_CATEGORY in frame.columns:
        return frame.with_columns(
            pl.col(pc_cols.TRANSACTION_CATEGORY)
            .map_elements(normalize_transaction_category, return_dtype=pl.String)
            .alias(pc_cols.TRANSACTION_CATEGORY),
        )
    if pc_cols.TRANSACTION_CODE in frame.columns:
        return frame.with_columns(
            pl.col(pc_cols.TRANSACTION_CODE)
            .map_elements(transaction_category_from_code, return_dtype=pl.String)
            .alias(pc_cols.TRANSACTION_CATEGORY),
        )
    return frame.with_columns(
        pl.lit(TRANSACTION_CATEGORY_UNKNOWN).alias(pc_cols.TRANSACTION_CATEGORY)
    )
