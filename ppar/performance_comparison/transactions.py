"""Load normalized transaction comparison sources.

The transaction category, sign, and semantics constants are intentionally
public. They define the YAML/source vocabulary used to avoid inferred
performance-flow assumptions.
"""

from __future__ import annotations

# Python imports
from collections.abc import Mapping

# Third-party imports
import polars as pl

# Project imports
from ppar.errors import PpaError
from ppar.performance_comparison import aliases
from ppar.performance_comparison import columns as pc_cols
from ppar.performance_comparison import source_loader
from ppar.performance_comparison.portfolio_performance import SnapshotKey
from ppar.performance_comparison.specification import PerformanceComparisonSpecification
import ppar.utilities as util

__all__ = [
    "TRANSACTION_CATEGORY_EXTERNAL_FLOW",
    "TRANSACTION_CATEGORY_INCOME",
    "TRANSACTION_CATEGORY_FEE_EXPENSE",
    "TRANSACTION_CATEGORY_BUY",
    "TRANSACTION_CATEGORY_SELL",
    "TRANSACTION_CATEGORY_TRANSFER",
    "TRANSACTION_CATEGORY_CORPORATE_ACTION",
    "TRANSACTION_CATEGORY_UNKNOWN",
    "TRANSACTION_CASH_FLOW_SIGN_POSITIVE",
    "TRANSACTION_CASH_FLOW_SIGN_NEGATIVE",
    "TRANSACTION_CASH_FLOW_SIGN_NONE",
    "TRANSACTION_CASH_FLOW_SIGN_UNKNOWN",
    "TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL",
    "TRANSACTION_PERFORMANCE_FLOW_SIGN_PERFORMANCE",
    "TRANSACTION_PERFORMANCE_FLOW_SIGN_NEUTRAL",
    "TRANSACTION_PERFORMANCE_FLOW_SIGN_UNKNOWN",
    "TRANSACTION_SEMANTICS_SOURCE_SOURCE",
    "TRANSACTION_SEMANTICS_SOURCE_YAML_RULE",
    "TRANSACTION_SEMANTICS_SOURCE_MIXED",
    "TRANSACTION_SEMANTICS_SOURCE_UNKNOWN",
    "normalize_transaction_category",
    "transaction_category_from_code",
    "normalize_transaction_cash_flow_sign",
    "normalize_transaction_performance_flow_sign",
    "transaction_impact_semantics_available",
    "TransactionsLoader",
]

TRANSACTION_CATEGORY_EXTERNAL_FLOW = "external_flow"
TRANSACTION_CATEGORY_INCOME = "income"
TRANSACTION_CATEGORY_FEE_EXPENSE = "fee_expense"
TRANSACTION_CATEGORY_BUY = "buy"
TRANSACTION_CATEGORY_SELL = "sell"
TRANSACTION_CATEGORY_TRANSFER = "transfer"
TRANSACTION_CATEGORY_CORPORATE_ACTION = "corporate_action"
TRANSACTION_CATEGORY_UNKNOWN = "unknown"
TRANSACTION_CASH_FLOW_SIGN_POSITIVE = "positive"
TRANSACTION_CASH_FLOW_SIGN_NEGATIVE = "negative"
TRANSACTION_CASH_FLOW_SIGN_NONE = "none"
TRANSACTION_CASH_FLOW_SIGN_UNKNOWN = "unknown"
TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL = "external"
TRANSACTION_PERFORMANCE_FLOW_SIGN_PERFORMANCE = "performance"
TRANSACTION_PERFORMANCE_FLOW_SIGN_NEUTRAL = "neutral"
TRANSACTION_PERFORMANCE_FLOW_SIGN_UNKNOWN = "unknown"
TRANSACTION_SEMANTICS_SOURCE_SOURCE = "source"
TRANSACTION_SEMANTICS_SOURCE_YAML_RULE = "yaml_rule"
TRANSACTION_SEMANTICS_SOURCE_MIXED = "mixed"
TRANSACTION_SEMANTICS_SOURCE_UNKNOWN = "unknown"
_TRANSACTION_RULES_KEY = "transaction_rules"

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

_CASH_FLOW_SIGN_NORMALIZATION: dict[str, str] = {
    "0": TRANSACTION_CASH_FLOW_SIGN_NONE,
    "cash_in": TRANSACTION_CASH_FLOW_SIGN_POSITIVE,
    "cash_out": TRANSACTION_CASH_FLOW_SIGN_NEGATIVE,
    "deposit": TRANSACTION_CASH_FLOW_SIGN_POSITIVE,
    "in": TRANSACTION_CASH_FLOW_SIGN_POSITIVE,
    "inflow": TRANSACTION_CASH_FLOW_SIGN_POSITIVE,
    "minus": TRANSACTION_CASH_FLOW_SIGN_NEGATIVE,
    "negative": TRANSACTION_CASH_FLOW_SIGN_NEGATIVE,
    "neutral": TRANSACTION_CASH_FLOW_SIGN_NONE,
    "no_cash_flow": TRANSACTION_CASH_FLOW_SIGN_NONE,
    "none": TRANSACTION_CASH_FLOW_SIGN_NONE,
    "out": TRANSACTION_CASH_FLOW_SIGN_NEGATIVE,
    "outflow": TRANSACTION_CASH_FLOW_SIGN_NEGATIVE,
    "plus": TRANSACTION_CASH_FLOW_SIGN_POSITIVE,
    "positive": TRANSACTION_CASH_FLOW_SIGN_POSITIVE,
    "withdrawal": TRANSACTION_CASH_FLOW_SIGN_NEGATIVE,
}

_PERFORMANCE_FLOW_SIGN_NORMALIZATION: dict[str, str] = {
    "0": TRANSACTION_PERFORMANCE_FLOW_SIGN_NEUTRAL,
    "external": TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL,
    "external_flow": TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL,
    "excluded": TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL,
    "included": TRANSACTION_PERFORMANCE_FLOW_SIGN_PERFORMANCE,
    "market": TRANSACTION_PERFORMANCE_FLOW_SIGN_PERFORMANCE,
    "neutral": TRANSACTION_PERFORMANCE_FLOW_SIGN_NEUTRAL,
    "none": TRANSACTION_PERFORMANCE_FLOW_SIGN_NEUTRAL,
    "performance": TRANSACTION_PERFORMANCE_FLOW_SIGN_PERFORMANCE,
    "performance_effect": TRANSACTION_PERFORMANCE_FLOW_SIGN_PERFORMANCE,
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


def normalize_transaction_cash_flow_sign(value: object) -> str:
    """Return normalized source-supplied transaction cash-flow direction.

    Args:
        value: Source cash-flow sign or direction label.

    Returns:
        ``"positive"``, ``"negative"``, ``"none"``, or ``"unknown"``.
        Unknown, blank, and missing values return ``"unknown"``.
    """
    return _normalize_transaction_semantic_label(
        value,
        _CASH_FLOW_SIGN_NORMALIZATION,
        TRANSACTION_CASH_FLOW_SIGN_UNKNOWN,
    )


def normalize_transaction_performance_flow_sign(value: object) -> str:
    """Return normalized transaction performance-flow treatment.

    Args:
        value: Source performance-flow sign or treatment label.

    Returns:
        ``"external"``, ``"performance"``, ``"neutral"``, or ``"unknown"``.
        Unknown, blank, and missing values return ``"unknown"``.
    """
    return _normalize_transaction_semantic_label(
        value,
        _PERFORMANCE_FLOW_SIGN_NORMALIZATION,
        TRANSACTION_PERFORMANCE_FLOW_SIGN_UNKNOWN,
    )


def transaction_impact_semantics_available(row: Mapping[str, object]) -> bool:
    """Return whether a transaction row carries modeled sign/flow semantics.

    Args:
        row: Normalized transaction row.

    Returns:
        ``True`` only when both optional semantic fields are present and
        normalized to recognized non-unknown values. This helper is an
        eligibility check for future impact methods; it does not imply that a
        return-impact estimate is currently calculated.
    """
    cash_flow_sign = row.get(pc_cols.CASH_FLOW_SIGN)
    performance_flow_sign = row.get(pc_cols.PERFORMANCE_FLOW_SIGN)
    return (
        cash_flow_sign
        in {
            TRANSACTION_CASH_FLOW_SIGN_POSITIVE,
            TRANSACTION_CASH_FLOW_SIGN_NEGATIVE,
            TRANSACTION_CASH_FLOW_SIGN_NONE,
        }
        and performance_flow_sign
        in {
            TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL,
            TRANSACTION_PERFORMANCE_FLOW_SIGN_PERFORMANCE,
            TRANSACTION_PERFORMANCE_FLOW_SIGN_NEUTRAL,
        }
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
        return _with_transaction_rules(
            _with_transaction_semantics(_with_transaction_category(frame)),
            self._transaction_rules(),
        )

    def _transaction_rules(self) -> dict[str, dict[str, str]]:
        """Return normalized YAML transaction rules keyed by transaction code."""
        rules_value = self._specification.values.get(_TRANSACTION_RULES_KEY, {})
        if not isinstance(rules_value, dict):
            raise PpaError(
                f"{self._specification.path}: transaction_rules must be a mapping.",
                504,
            )

        rules: dict[str, dict[str, str]] = {}
        for raw_code, raw_rule in rules_value.items():
            if not isinstance(raw_code, str) or not raw_code.strip():
                raise PpaError(
                    f"{self._specification.path}: transaction rule keys must be strings.",
                    504,
                )
            if not isinstance(raw_rule, dict):
                raise PpaError(
                    (
                        f"{self._specification.path}: transaction_rules.{raw_code} "
                        "must be a mapping."
                    ),
                    504,
                )
            rules[raw_code.strip().upper()] = _normalized_transaction_rule(raw_rule)
        return rules


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


def _with_transaction_semantics(frame: pl.DataFrame) -> pl.DataFrame:
    """Return transaction rows with normalized optional sign semantics."""
    expressions = []
    if pc_cols.CASH_FLOW_SIGN in frame.columns:
        expressions.append(
            pl.col(pc_cols.CASH_FLOW_SIGN)
            .map_elements(normalize_transaction_cash_flow_sign, return_dtype=pl.String)
            .alias(pc_cols.CASH_FLOW_SIGN)
        )
    if pc_cols.PERFORMANCE_FLOW_SIGN in frame.columns:
        expressions.append(
            pl.col(pc_cols.PERFORMANCE_FLOW_SIGN)
            .map_elements(
                normalize_transaction_performance_flow_sign,
                return_dtype=pl.String,
            )
            .alias(pc_cols.PERFORMANCE_FLOW_SIGN)
        )
    if not expressions:
        return frame
    return frame.with_columns(*expressions)


def _with_transaction_rules(
    frame: pl.DataFrame,
    rules: dict[str, dict[str, str]],
) -> pl.DataFrame:
    """Return transaction rows with missing semantics filled from YAML rules."""
    if pc_cols.CASH_FLOW_SIGN not in frame.columns:
        frame = frame.with_columns(
            pl.lit(TRANSACTION_CASH_FLOW_SIGN_UNKNOWN).alias(pc_cols.CASH_FLOW_SIGN)
        )
    if pc_cols.PERFORMANCE_FLOW_SIGN not in frame.columns:
        frame = frame.with_columns(
            pl.lit(TRANSACTION_PERFORMANCE_FLOW_SIGN_UNKNOWN).alias(
                pc_cols.PERFORMANCE_FLOW_SIGN
            )
        )
    if pc_cols.TRANSACTION_SEMANTICS_SOURCE not in frame.columns:
        frame = frame.with_columns(
            pl.lit(TRANSACTION_SEMANTICS_SOURCE_UNKNOWN).alias(
                pc_cols.TRANSACTION_SEMANTICS_SOURCE
            )
        )

    if not rules or pc_cols.TRANSACTION_CODE not in frame.columns:
        rows = [
            _row_with_transaction_semantics_source(dict(row), False, row)
            for row in frame.iter_rows(named=True)
        ]
        return pl.DataFrame(rows).select(frame.columns)

    rows = [_row_with_transaction_rule(row, rules) for row in frame.iter_rows(named=True)]
    return pl.DataFrame(rows).select(frame.columns)


def _row_with_transaction_rule(
    row: dict[str, object],
    rules: dict[str, dict[str, str]],
) -> dict[str, object]:
    """Return one transaction row with YAML rule values filling unknown fields."""
    original_row = dict(row)
    raw_code = row.get(pc_cols.TRANSACTION_CODE)
    if raw_code is None:
        return _row_with_transaction_semantics_source(dict(row), False, original_row)
    rule = rules.get(str(raw_code).strip().upper())
    if rule is None:
        return _row_with_transaction_semantics_source(dict(row), False, original_row)

    updated_row = dict(row)
    yaml_filled = False
    if updated_row.get(pc_cols.TRANSACTION_CATEGORY) == TRANSACTION_CATEGORY_UNKNOWN:
        updated_row[pc_cols.TRANSACTION_CATEGORY] = rule[pc_cols.TRANSACTION_CATEGORY]
        yaml_filled = rule[pc_cols.TRANSACTION_CATEGORY] != TRANSACTION_CATEGORY_UNKNOWN
    if updated_row.get(pc_cols.CASH_FLOW_SIGN) == TRANSACTION_CASH_FLOW_SIGN_UNKNOWN:
        updated_row[pc_cols.CASH_FLOW_SIGN] = rule[pc_cols.CASH_FLOW_SIGN]
        yaml_filled = yaml_filled or (
            rule[pc_cols.CASH_FLOW_SIGN] != TRANSACTION_CASH_FLOW_SIGN_UNKNOWN
        )
    if (
        updated_row.get(pc_cols.PERFORMANCE_FLOW_SIGN)
        == TRANSACTION_PERFORMANCE_FLOW_SIGN_UNKNOWN
    ):
        updated_row[pc_cols.PERFORMANCE_FLOW_SIGN] = rule[pc_cols.PERFORMANCE_FLOW_SIGN]
        yaml_filled = yaml_filled or (
            rule[pc_cols.PERFORMANCE_FLOW_SIGN]
            != TRANSACTION_PERFORMANCE_FLOW_SIGN_UNKNOWN
        )
    return _row_with_transaction_semantics_source(
        updated_row,
        yaml_filled,
        original_row,
    )


def _row_with_transaction_semantics_source(
    row: dict[str, object],
    yaml_filled: bool,
    original_row: Mapping[str, object],
) -> dict[str, object]:
    """Return one row tagged with transaction semantics provenance."""
    # The provenance label is intentionally conservative: a row is only usable
    # for transaction impact estimates when both sign fields are recognized.
    if not transaction_impact_semantics_available(row):
        row[pc_cols.TRANSACTION_SEMANTICS_SOURCE] = TRANSACTION_SEMANTICS_SOURCE_UNKNOWN
        return row

    source_supplied = _has_source_transaction_semantics(original_row)
    if yaml_filled and source_supplied:
        row[pc_cols.TRANSACTION_SEMANTICS_SOURCE] = TRANSACTION_SEMANTICS_SOURCE_MIXED
    elif yaml_filled:
        row[pc_cols.TRANSACTION_SEMANTICS_SOURCE] = TRANSACTION_SEMANTICS_SOURCE_YAML_RULE
    else:
        row[pc_cols.TRANSACTION_SEMANTICS_SOURCE] = TRANSACTION_SEMANTICS_SOURCE_SOURCE
    return row


def _has_source_transaction_semantics(row: Mapping[str, object]) -> bool:
    """Return whether the source row carried any recognized transaction semantics."""
    return (
        _recognized_transaction_category(row.get(pc_cols.TRANSACTION_CATEGORY))
        or _recognized_cash_flow_sign(row.get(pc_cols.CASH_FLOW_SIGN))
        or _recognized_performance_flow_sign(row.get(pc_cols.PERFORMANCE_FLOW_SIGN))
    )


def _recognized_transaction_category(value: object) -> bool:
    """Return whether a normalized transaction category is recognized."""
    return value not in {None, "", TRANSACTION_CATEGORY_UNKNOWN}


def _recognized_cash_flow_sign(value: object) -> bool:
    """Return whether a normalized transaction cash-flow sign is recognized."""
    return value in {
        TRANSACTION_CASH_FLOW_SIGN_POSITIVE,
        TRANSACTION_CASH_FLOW_SIGN_NEGATIVE,
        TRANSACTION_CASH_FLOW_SIGN_NONE,
    }


def _recognized_performance_flow_sign(value: object) -> bool:
    """Return whether a normalized transaction performance-flow sign is recognized."""
    return value in {
        TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL,
        TRANSACTION_PERFORMANCE_FLOW_SIGN_PERFORMANCE,
        TRANSACTION_PERFORMANCE_FLOW_SIGN_NEUTRAL,
    }


def _normalized_transaction_rule(rule: Mapping[str, object]) -> dict[str, str]:
    """Return one normalized YAML transaction semantics rule."""
    return {
        pc_cols.TRANSACTION_CATEGORY: normalize_transaction_category(
            rule.get(pc_cols.TRANSACTION_CATEGORY)
        ),
        pc_cols.CASH_FLOW_SIGN: normalize_transaction_cash_flow_sign(
            rule.get(pc_cols.CASH_FLOW_SIGN)
        ),
        pc_cols.PERFORMANCE_FLOW_SIGN: normalize_transaction_performance_flow_sign(
            rule.get(pc_cols.PERFORMANCE_FLOW_SIGN)
        ),
    }


def _normalize_transaction_semantic_label(
    value: object,
    normalization: Mapping[str, str],
    unknown_value: str,
) -> str:
    """Return a normalized transaction semantic label."""
    if value is None:
        return unknown_value
    normalized_value = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized_value:
        return unknown_value
    return normalization.get(normalized_value, unknown_value)
