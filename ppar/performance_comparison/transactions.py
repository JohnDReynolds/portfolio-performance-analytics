"""Load normalized transaction comparison sources.

The transaction category, sign, and semantics constants are intentionally
public. They define the YAML/source vocabulary used to avoid inferred
performance-flow assumptions.
"""

from __future__ import annotations

# Python imports
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

# Third-party imports
import polars as pl

# Project imports
from ppar.errors import PpaError
from ppar.performance_comparison import aliases
from ppar.performance_comparison import schema as pc_cols
from ppar.performance_comparison.base_currency import with_authoritative_base_currency
from ppar.performance_comparison.currency_basis import normalize_currency_columns
from ppar.performance_comparison.extract_contract import (
    validate_transaction_extract_contract,
)
from ppar.performance_comparison import source_loader
from ppar.performance_comparison.portfolio_performance import (
    PortfolioPerformanceLoader,
    SnapshotKey,
)
from ppar.performance_comparison.specification import PerformanceComparisonSpecification
import ppar.utilities as util

__all__ = [
    "TransactionCategory",
    "TransactionCashFlowSign",
    "TransactionPerformanceFlowSign",
    "TransactionSemanticsSource",
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

class TransactionCategory(StrEnum):
    """Supported normalized transaction category labels."""

    EXTERNAL_FLOW = "external_flow"
    INCOME = "income"
    FEE_EXPENSE = "fee_expense"
    BUY = "buy"
    SELL = "sell"
    TRANSFER = "transfer"
    CORPORATE_ACTION = "corporate_action"
    UNKNOWN = "unknown"


class TransactionCashFlowSign(StrEnum):
    """Supported normalized transaction cash-flow direction labels."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NONE = "none"
    UNKNOWN = "unknown"


class TransactionPerformanceFlowSign(StrEnum):
    """Supported normalized transaction performance-flow treatment labels."""

    EXTERNAL = "external"
    PERFORMANCE = "performance"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class TransactionSemanticsSource(StrEnum):
    """Supported transaction sign/flow semantics provenance labels."""

    SOURCE = "source"
    YAML_RULE = "yaml_rule"
    MIXED = "mixed"
    UNKNOWN = "unknown"


TRANSACTION_CATEGORY_EXTERNAL_FLOW = TransactionCategory.EXTERNAL_FLOW.value
TRANSACTION_CATEGORY_INCOME = TransactionCategory.INCOME.value
TRANSACTION_CATEGORY_FEE_EXPENSE = TransactionCategory.FEE_EXPENSE.value
TRANSACTION_CATEGORY_BUY = TransactionCategory.BUY.value
TRANSACTION_CATEGORY_SELL = TransactionCategory.SELL.value
TRANSACTION_CATEGORY_TRANSFER = TransactionCategory.TRANSFER.value
TRANSACTION_CATEGORY_CORPORATE_ACTION = TransactionCategory.CORPORATE_ACTION.value
TRANSACTION_CATEGORY_UNKNOWN = TransactionCategory.UNKNOWN.value
TRANSACTION_CASH_FLOW_SIGN_POSITIVE = TransactionCashFlowSign.POSITIVE.value
TRANSACTION_CASH_FLOW_SIGN_NEGATIVE = TransactionCashFlowSign.NEGATIVE.value
TRANSACTION_CASH_FLOW_SIGN_NONE = TransactionCashFlowSign.NONE.value
TRANSACTION_CASH_FLOW_SIGN_UNKNOWN = TransactionCashFlowSign.UNKNOWN.value
TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL = TransactionPerformanceFlowSign.EXTERNAL.value
TRANSACTION_PERFORMANCE_FLOW_SIGN_PERFORMANCE = (
    TransactionPerformanceFlowSign.PERFORMANCE.value
)
TRANSACTION_PERFORMANCE_FLOW_SIGN_NEUTRAL = TransactionPerformanceFlowSign.NEUTRAL.value
TRANSACTION_PERFORMANCE_FLOW_SIGN_UNKNOWN = TransactionPerformanceFlowSign.UNKNOWN.value
TRANSACTION_SEMANTICS_SOURCE_SOURCE = TransactionSemanticsSource.SOURCE.value
TRANSACTION_SEMANTICS_SOURCE_YAML_RULE = TransactionSemanticsSource.YAML_RULE.value
TRANSACTION_SEMANTICS_SOURCE_MIXED = TransactionSemanticsSource.MIXED.value
TRANSACTION_SEMANTICS_SOURCE_UNKNOWN = TransactionSemanticsSource.UNKNOWN.value
_TRANSACTION_RULES_KEY = "transaction_rules"
_TRANSACTION_RULE_WHEN_KEY = "when"

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
    "BY": TRANSACTION_CATEGORY_BUY,
    "BUY": TRANSACTION_CATEGORY_BUY,
    "SL": TRANSACTION_CATEGORY_SELL,
    "SELL": TRANSACTION_CATEGORY_SELL,
    "DV": TRANSACTION_CATEGORY_INCOME,
    "DIV": TRANSACTION_CATEGORY_INCOME,
    "IN": TRANSACTION_CATEGORY_INCOME,
    "INT": TRANSACTION_CATEGORY_INCOME,
    "INCOME": TRANSACTION_CATEGORY_INCOME,
    "FEE": TRANSACTION_CATEGORY_FEE_EXPENSE,
    "EXP": TRANSACTION_CATEGORY_FEE_EXPENSE,
    "DEP": TRANSACTION_CATEGORY_EXTERNAL_FLOW,
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


@dataclass(frozen=True)
class _TransactionRule:
    """One normalized transaction semantics rule from YAML."""

    when: Mapping[str, str]
    values: Mapping[str, str]


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
        cached = source_loader.cached_normalized_frame(
            self._specification.path,
            pc_cols.TRANSACTIONS,
            snapshot_key,
            path,
        )
        if cached is not None:
            return cached

        frame = source_loader.read_mapped_csv(
            path,
            pc_cols.TRANSACTIONS_COLUMNS,
            pc_cols.TRANSACTIONS,
            aliases.TRANSACTIONS_REQUIRED_ALIASES,
            aliases.TRANSACTIONS_OPTIONAL_ALIASES,
            self._specification.path,
        )
        validate_transaction_extract_contract(
            frame,
            path=path,
            specification_path=self._specification.path,
            specification_values=self._specification.values,
        )
        date_columns = [
            column
            for column in (pc_cols.TRANSACTION_DATE, pc_cols.SETTLEMENT_DATE)
            if column in frame.columns
        ]
        frame = frame.with_columns(
            pl.col(date_columns).str.strptime(pl.Date, "%Y-%m-%d", strict=True),
        )
        frame = source_loader.require_numeric_columns(
            frame,
            columns=(
                pc_cols.QUANTITY,
                pc_cols.PRICE,
                pc_cols.AMOUNT,
                pc_cols.BASE_AMOUNT,
                pc_cols.COMMISSION,
            ),
            dataset_name=pc_cols.TRANSACTIONS,
            path=path,
            specification_path=self._specification.path,
        )
        frame = _with_transaction_rules(
            _with_transaction_semantics(_with_transaction_category(frame)),
            self._transaction_rules(),
        )
        _validate_transaction_semantics(
            frame,
            path=path,
            specification_path=self._specification.path,
        )
        frame = normalize_currency_columns(
            with_authoritative_base_currency(
                frame,
                PortfolioPerformanceLoader(self._specification).load(snapshot_key),
                dataset_name=pc_cols.TRANSACTIONS,
                path=path,
                specification_path=self._specification.path,
            )
        )
        return source_loader.cache_normalized_frame(
            self._specification.path,
            pc_cols.TRANSACTIONS,
            snapshot_key,
            path,
            frame,
        )

    def _transaction_rules(self) -> dict[str, tuple[_TransactionRule, ...]]:
        """Return normalized YAML transaction rules keyed by transaction code."""
        rules_value = self._specification.values.get(_TRANSACTION_RULES_KEY, {})
        if not isinstance(rules_value, dict):
            raise PpaError(
                f"{self._specification.path}: transaction_rules must be a mapping.",
                504,
            )

        rules: dict[str, tuple[_TransactionRule, ...]] = {}
        for raw_code, raw_rule in rules_value.items():
            if not isinstance(raw_code, str) or not raw_code.strip():
                raise PpaError(
                    f"{self._specification.path}: transaction rule keys must be strings.",
                    504,
                )
            try:
                rules[raw_code.strip().upper()] = _normalized_transaction_rules(raw_rule)
            except ValueError as error:
                raise PpaError(
                    (
                        f"{self._specification.path}: transaction_rules.{raw_code} "
                        f"{error}"
                    ),
                    504,
                ) from error
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
    rules: dict[str, tuple[_TransactionRule, ...]],
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


def _validate_transaction_semantics(
    frame: pl.DataFrame,
    *,
    path: util.PathLike,
    specification_path: util.PathLike,
) -> None:
    """Raise when transaction rows do not have a known transaction category."""
    unknown_rows = frame.filter(
        pl.col(pc_cols.TRANSACTION_CATEGORY) == TRANSACTION_CATEGORY_UNKNOWN
    )
    if unknown_rows.height == 0:
        return

    raise PpaError(
        (
            f"{specification_path}: transactions file {path} contains rows with "
            "unknown transaction codes or categories. Every transaction row must "
            "resolve transaction_category from a recognized source category, known "
            "transaction_code, or transaction_rules entry. Add transaction_rules "
            "entries for custom transaction_code values, provide recognized "
            "source categories, or include the IMEX context fields needed by "
            "conditional transaction_rules. If IMEX cannot expose those fields, "
            "consider a REP/report extract for transaction classification. "
            f"Sample rows: {_transaction_semantics_error_samples(unknown_rows)}"
        ),
        504,
    )


def _transaction_semantics_error_samples(frame: pl.DataFrame) -> str:
    """Return compact row samples for transaction semantics validation errors."""
    sample_columns = [
        column
        for column in (
            pc_cols.PORTFOLIO_ID,
            pc_cols.SECURITY_ID,
            pc_cols.TRANSACTION_DATE,
            pc_cols.TRANSACTION_CODE,
            pc_cols.SECURITY_TYPE,
            pc_cols.SOURCE_DESTINATION_TYPE,
            pc_cols.SOURCE_DESTINATION_SYMBOL,
            pc_cols.SPECIAL_SECURITY_TYPE,
            pc_cols.SPECIAL_SECURITY_SYMBOL,
            pc_cols.TRANSACTION_CATEGORY,
            pc_cols.CASH_FLOW_SIGN,
            pc_cols.PERFORMANCE_FLOW_SIGN,
        )
        if column in frame.columns
    ]
    samples = []
    for row in frame.select(sample_columns).head(5).iter_rows(named=True):
        samples.append(
            ", ".join(f"{column}={row[column]}" for column in sample_columns)
        )
    return "; ".join(samples)


def _row_with_transaction_rule(
    row: dict[str, object],
    rules: dict[str, tuple[_TransactionRule, ...]],
) -> dict[str, object]:
    """Return one transaction row with YAML rule values filling unknown fields."""
    original_row = dict(row)
    raw_code = row.get(pc_cols.TRANSACTION_CODE)
    if raw_code is None:
        return _row_with_transaction_semantics_source(dict(row), False, original_row)
    rule = _matching_transaction_rule(row, rules.get(str(raw_code).strip().upper()))
    if rule is None:
        return _row_with_transaction_semantics_source(dict(row), False, original_row)

    updated_row = dict(row)
    yaml_filled = False
    if updated_row.get(pc_cols.TRANSACTION_CATEGORY) == TRANSACTION_CATEGORY_UNKNOWN:
        category = rule.values[pc_cols.TRANSACTION_CATEGORY]
        updated_row[pc_cols.TRANSACTION_CATEGORY] = category
        yaml_filled = category != TRANSACTION_CATEGORY_UNKNOWN
    if updated_row.get(pc_cols.CASH_FLOW_SIGN) == TRANSACTION_CASH_FLOW_SIGN_UNKNOWN:
        cash_flow_sign = rule.values[pc_cols.CASH_FLOW_SIGN]
        updated_row[pc_cols.CASH_FLOW_SIGN] = cash_flow_sign
        yaml_filled = yaml_filled or (
            cash_flow_sign != TRANSACTION_CASH_FLOW_SIGN_UNKNOWN
        )
    if (
        updated_row.get(pc_cols.PERFORMANCE_FLOW_SIGN)
        == TRANSACTION_PERFORMANCE_FLOW_SIGN_UNKNOWN
    ):
        performance_flow_sign = rule.values[pc_cols.PERFORMANCE_FLOW_SIGN]
        updated_row[pc_cols.PERFORMANCE_FLOW_SIGN] = performance_flow_sign
        yaml_filled = yaml_filled or (
            performance_flow_sign != TRANSACTION_PERFORMANCE_FLOW_SIGN_UNKNOWN
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


def _matching_transaction_rule(
    row: Mapping[str, object],
    rules: tuple[_TransactionRule, ...] | None,
) -> _TransactionRule | None:
    """Return the first YAML rule whose conditions match one transaction row."""
    if rules is None:
        return None
    for rule in rules:
        if _transaction_rule_matches(row, rule):
            return rule
    return None


def _transaction_rule_matches(
    row: Mapping[str, object],
    rule: _TransactionRule,
) -> bool:
    """Return whether all normalized YAML ``when`` conditions match a row."""
    for column, expected_value in rule.when.items():
        actual_value = row.get(column)
        if _normalized_transaction_rule_condition(actual_value) != expected_value:
            return False
    return True


def _normalized_transaction_rules(raw_rule: object) -> tuple[_TransactionRule, ...]:
    """Return normalized YAML transaction semantics rules for one code."""
    if isinstance(raw_rule, list):
        rules = raw_rule
    elif isinstance(raw_rule, dict):
        nested_rules = raw_rule.get("rules")
        rules = nested_rules if isinstance(nested_rules, list) else [raw_rule]
    else:
        raise ValueError("must be a mapping or list of mappings.")
    if not rules:
        raise ValueError("must define at least one rule.")

    normalized_rules = []
    for rule in rules:
        if not isinstance(rule, dict):
            raise ValueError("must contain mapping rules.")
        normalized_rules.append(_normalized_transaction_rule(rule))
    return tuple(normalized_rules)


def _normalized_transaction_rule(rule: Mapping[str, object]) -> _TransactionRule:
    """Return one normalized YAML transaction semantics rule."""
    when = _normalized_transaction_rule_conditions(
        rule.get(_TRANSACTION_RULE_WHEN_KEY, {})
    )
    values = {
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
    return _TransactionRule(when=when, values=values)


def _normalized_transaction_rule_conditions(raw_conditions: object) -> dict[str, str]:
    """Return normalized YAML transaction rule conditions."""
    if raw_conditions in (None, ""):
        return {}
    if not isinstance(raw_conditions, dict):
        raise ValueError("when must be a mapping.")
    supported_columns = set(pc_cols.TRANSACTIONS_COLUMNS)
    conditions: dict[str, str] = {}
    for raw_column, raw_value in raw_conditions.items():
        if not isinstance(raw_column, str) or not raw_column.strip():
            raise ValueError("when keys must be normalized transaction column names.")
        column = raw_column.strip()
        if column not in supported_columns:
            raise ValueError(
                "when keys must be normalized transaction column names; "
                f"unsupported key {column!r}."
            )
        conditions[column] = _normalized_transaction_rule_condition(raw_value)
    return conditions


def _normalized_transaction_rule_condition(value: object) -> str:
    """Return normalized scalar value used for conditional rule matching."""
    if value is None:
        return ""
    return str(value).strip().lower()


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
