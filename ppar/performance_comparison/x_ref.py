"""Cross-reference consistency checks for performance-comparison reports.

The checks in this module are intentionally separate from ``source_detail.csv``.
Source-detail rows explain changed source rows between Snapshot A and Snapshot B;
Data Audit Issues look for internally inconsistent source-data across the union
of both snapshots.
"""

from __future__ import annotations

# Python imports
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import datetime as dt
from typing import Any, Final

# Third-party imports
import polars as pl

# Project imports
import ppar.utilities as util
from ppar.performance_comparison import schema as pc_cols
from ppar.performance_comparison.holdings import HoldingsLoader
from ppar.performance_comparison.portfolio_performance import (
    PortfolioPerformanceLoader,
    SnapshotKey,
)
from ppar.performance_comparison.security_performance import SecurityPerformanceLoader
from ppar.performance_comparison.specification import PerformanceComparisonSpecification
from ppar.performance_comparison.transactions import TransactionsLoader

SNAPSHOT: Final[str] = "snapshot"
AS_OF_DATE: Final[str] = "as_of_date"
ISSUE_TYPE: Final[str] = "issue_type"
DATASET_FIELD: Final[str] = "dataset_field"
VALUE_A: Final[str] = "value_a"
VALUE_B: Final[str] = "value_b"
DIFFERENCE: Final[str] = "difference"
TOLERANCE: Final[str] = "tolerance"
EXPLANATION: Final[str] = "explanation"
REVIEW_KEY: Final[str] = "review_key"

ISSUE_DUPLICATE_TRANSACTIONS: Final[str] = "duplicate_transactions"
ISSUE_DIVIDEND_RATE: Final[str] = "dividend_rate"
ISSUE_HOLDINGS_ACCRUED_RATE: Final[str] = "holdings_accrued_rate"
ISSUE_HOLDINGS_PRICE_RANGE: Final[str] = "holdings_price_range"
ISSUE_MISSING_DIVIDEND: Final[str] = "missing_dividend"
ISSUE_PA_SA_RATE: Final[str] = "pa_sa_rate"
ISSUE_PORTFOLIO_MV_CONTINUITY: Final[str] = "portfolio_market_value_continuity"
ISSUE_SECURITY_MV_CONTINUITY: Final[str] = "security_market_value_continuity"
ISSUE_TRANSACTIONS_PRICE_RANGE: Final[str] = "transactions_price_range"

X_REF_ISSUE_COLUMNS: Final[tuple[str, ...]] = (
    SNAPSHOT,
    pc_cols.PORTFOLIO_ID,
    AS_OF_DATE,
    DATASET_FIELD,
    pc_cols.SECURITY_ID,
    ISSUE_TYPE,
    VALUE_A,
    VALUE_B,
    DIFFERENCE,
    TOLERANCE,
    EXPLANATION,
    REVIEW_KEY,
)

_X_REF_CONFIG_KEY: Final[str] = "data_audit_checks"
_SNAPSHOT_A_LABEL: Final[str] = "Snapshot A"
_SNAPSHOT_B_LABEL: Final[str] = "Snapshot B"
_BUY_CODES: Final[frozenset[str]] = frozenset({"by"})
_DIVIDEND_CODES: Final[frozenset[str]] = frozenset({"dv"})
_ACCRUAL_CODES: Final[frozenset[str]] = frozenset({"pa", "sa"})
_DEFAULT_PERCENT_TOLERANCE: Final[float] = 0.0
_DEFAULT_ABSOLUTE_TOLERANCE: Final[float] = 0.0
_FILTER_FIELD_ALIASES: Final[dict[str, str]] = {
    "snapshot": SNAPSHOT,
    "portfolio": pc_cols.PORTFOLIO_ID,
    "portfolio_id": pc_cols.PORTFOLIO_ID,
    "security": pc_cols.SECURITY_ID,
    "security_id": pc_cols.SECURITY_ID,
    "security_type": pc_cols.SECURITY_TYPE,
    "asset_class": pc_cols.ASSET_CLASS,
    "transaction_code": pc_cols.TRANSACTION_CODE,
}

_ISSUE_SCHEMA: Final[dict[str, type[pl.DataType]]] = {
    SNAPSHOT: pl.String,
    pc_cols.PORTFOLIO_ID: pl.String,
    AS_OF_DATE: pl.Date,
    DATASET_FIELD: pl.String,
    pc_cols.SECURITY_ID: pl.String,
    ISSUE_TYPE: pl.String,
    VALUE_A: pl.Float64,
    VALUE_B: pl.Float64,
    DIFFERENCE: pl.Float64,
    TOLERANCE: pl.String,
    EXPLANATION: pl.String,
    REVIEW_KEY: pl.String,
}

_CONTINUITY_PRIOR_THRU_DATE: Final[str] = "__ppar_continuity_prior_thru_date"
_CONTINUITY_PRIOR_END_VALUE: Final[str] = "__ppar_continuity_prior_end_value"
_CONTINUITY_CURRENT_BEGIN_VALUE: Final[str] = (
    "__ppar_continuity_current_begin_value"
)
_CONTINUITY_THRESHOLD: Final[str] = "__ppar_continuity_threshold"


@dataclass(frozen=True)
class _Tolerance:
    """Numeric tolerance used to decide whether a consistency issue exists.

    Attributes:
        absolute: Absolute threshold in source-data units.
        percent: Percentage threshold, expressed as a percent rather than a
            decimal. For example, ``1.0`` means one percent.
    """

    absolute: float
    percent: float

    def threshold(self, reference_value: float) -> float:
        """Return the larger absolute or percent-based threshold."""
        return max(self.absolute, abs(reference_value) * self.percent / 100.0)

    def description(self) -> str:
        """Return a concise reviewer-facing tolerance description."""
        parts: list[str] = []
        if self.absolute:
            parts.append(_format_tolerance_number(self.absolute))
        if self.percent:
            parts.append(f"{_format_tolerance_number(self.percent)}%")
        if not parts:
            return "0"
        if len(parts) == 1:
            return parts[0]
        return f"greater of {parts[0]} or {parts[1]}"


@dataclass(frozen=True)
class _RowFilter:
    """Precompiled include/exclude filters for one data-audit check.

    Attributes:
        only: Filters that every row must match.
        exclude: Filters where any match excludes the row.

    Notes:
        YAML filter normalization is invariant for a check. Compiling it once
        avoids repeating that work for every source row at large sites.
    """

    only: tuple[tuple[str, frozenset[str]], ...]
    exclude: tuple[tuple[str, frozenset[str]], ...]

    def allows(self, row: Mapping[str, object]) -> bool:
        """Return whether one row passes the compiled filters."""
        if self.only and not all(
            _text(row.get(column_name)).lower() in values
            for column_name, values in self.only
        ):
            return False
        if not self.exclude:
            return True
        return not any(
            _text(row.get(column_name)).lower() in values
            for column_name, values in self.exclude
        )


def x_ref_issues_table(comparison_path: util.PathLike | None) -> pl.DataFrame:
    """Return cross-reference consistency issues for a comparison YAML.

    Args:
        comparison_path: Performance-comparison YAML path. When omitted, an
            empty table is returned because the source snapshots are unavailable.

    Returns:
        Data Audit Issues rows built from the union of Snapshot A and Snapshot B.
    """
    if comparison_path is None:
        return _empty_issues_table()

    specification = PerformanceComparisonSpecification(comparison_path)
    config = _x_ref_config(specification.values)
    portfolio_performance = _snapshot_frames(
        PortfolioPerformanceLoader(specification),
        pc_cols.PORTFOLIO_PERFORMANCE,
    )
    security_performance = _snapshot_frames(
        SecurityPerformanceLoader(specification),
        pc_cols.SECURITY_PERFORMANCE,
    )
    holdings = _snapshot_frames(HoldingsLoader(specification), pc_cols.HOLDINGS)
    transactions = _snapshot_frames(
        TransactionsLoader(specification),
        pc_cols.TRANSACTIONS,
    )
    rows: list[dict[str, object]] = []
    rows.extend(
        _market_value_continuity_issues(
            portfolio_performance,
            config,
            dataset_name=pc_cols.PORTFOLIO_PERFORMANCE,
        )
    )
    rows.extend(
        _market_value_continuity_issues(
            security_performance,
            config,
            dataset_name=pc_cols.SECURITY_PERFORMANCE,
        )
    )
    if not _config_enabled(config):
        return _issues_table(rows)
    transaction_rows = _snapshot_rows(transactions)
    holding_rows = _snapshot_rows(holdings)
    rows.extend(_duplicate_transaction_issues(transaction_rows, config))
    rows.extend(_holding_price_range_issues(holding_rows, config))
    rows.extend(_transaction_price_range_issues(transaction_rows, config))
    rows.extend(_same_day_rate_issues(transaction_rows, holding_rows, config))
    rows.extend(_missing_dividend_issues(holding_rows, transaction_rows, config))
    rows.extend(_holdings_accrued_rate_issues(holding_rows, config))
    return _issues_table(rows)


def _market_value_continuity_issues(
    performance_frames: Iterable[pl.DataFrame],
    config: Mapping[str, object],
    *,
    dataset_name: str,
) -> list[dict[str, object]]:
    """Return prior-ending versus next-beginning market-value issues.

    Continuity is a mandatory financial-integrity check. Unlike optional Data
    Audit checks, it remains active when ``data_audit_checks.enabled`` is false
    because these values directly participate in Modified Dietz.
    """
    issue_type = (
        ISSUE_SECURITY_MV_CONTINUITY
        if dataset_name == pc_cols.SECURITY_PERFORMANCE
        else ISSUE_PORTFOLIO_MV_CONTINUITY
    )
    tolerance = _tolerance(config, issue_type)
    if tolerance == _Tolerance(absolute=0.0, percent=0.0):
        tolerance = _Tolerance(absolute=0.01, percent=0.0)
    grouping_columns = [SNAPSHOT, pc_cols.PORTFOLIO_ID]
    if dataset_name == pc_cols.SECURITY_PERFORMANCE:
        grouping_columns.append(pc_cols.SECURITY_ID)

    candidates = _market_value_continuity_candidates(
        performance_frames,
        grouping_columns,
        tolerance,
    )
    rows: list[dict[str, object]] = []
    for current in candidates.iter_rows(named=True):
        current_from = _date(current.get(pc_cols.FROM_DATE))
        prior_end = _number(current.get(_CONTINUITY_PRIOR_END_VALUE))
        current_begin = _number(current.get(_CONTINUITY_CURRENT_BEGIN_VALUE))
        assert current_from is not None
        assert prior_end is not None
        assert current_begin is not None
        difference = current_begin - prior_end
        security_id = _text(current.get(pc_cols.SECURITY_ID))
        scope = f"portfolio {current.get(pc_cols.PORTFOLIO_ID)}"
        if security_id:
            scope += f", security {security_id}"
        rows.append(
            _issue_row(
                snapshot=_text(current.get(SNAPSHOT)),
                portfolio_id=_text(current.get(pc_cols.PORTFOLIO_ID)),
                as_of_date=current_from,
                dataset_field=(
                    f"{dataset_name}.end_market_value -> "
                    f"{dataset_name}.begin_market_value"
                ),
                security_id=security_id,
                issue_type=issue_type,
                value_a=prior_end,
                value_b=current_begin,
                difference=difference,
                tolerance=tolerance.description(),
                explanation=(
                    f"SN-04 continuity mismatch for {scope}: prior ending "
                    f"market value {prior_end:,.2f} does not equal next "
                    f"beginning market value {current_begin:,.2f}."
                ),
            )
        )
    return rows


def _market_value_continuity_candidates(
    performance_frames: Iterable[pl.DataFrame],
    grouping_columns: Sequence[str],
    tolerance: _Tolerance,
) -> pl.DataFrame:
    """Return consecutive-period market-value mismatches from one lazy plan."""
    candidate_plans: list[pl.LazyFrame] = []
    sort_columns = [
        *grouping_columns,
        pc_cols.FROM_DATE,
        pc_cols.THRU_DATE,
    ]
    for frame in performance_frames:
        if not {
            pc_cols.BEGIN_MARKET_VALUE,
            pc_cols.END_MARKET_VALUE,
        }.issubset(frame.columns):
            continue
        security_id = (
            pl.col(pc_cols.SECURITY_ID)
            if pc_cols.SECURITY_ID in frame.columns
            else pl.lit(None, dtype=pl.String)
        )
        candidate_plans.append(
            frame.lazy()
            .sort(sort_columns, nulls_last=False)
            .with_columns(
                pl.col(pc_cols.THRU_DATE)
                .shift(1)
                .over(grouping_columns)
                .alias(_CONTINUITY_PRIOR_THRU_DATE),
                pl.col(pc_cols.END_MARKET_VALUE)
                .shift(1)
                .over(grouping_columns)
                .cast(pl.Float64, strict=False)
                .alias(_CONTINUITY_PRIOR_END_VALUE),
                pl.col(pc_cols.BEGIN_MARKET_VALUE)
                .cast(pl.Float64, strict=False)
                .alias(_CONTINUITY_CURRENT_BEGIN_VALUE),
            )
            .filter(
                pl.col(_CONTINUITY_PRIOR_THRU_DATE).is_not_null()
                & pl.col(pc_cols.FROM_DATE).is_not_null()
                & (
                    pl.col(pc_cols.FROM_DATE)
                    == pl.col(_CONTINUITY_PRIOR_THRU_DATE).dt.offset_by("1d")
                )
                & pl.col(_CONTINUITY_PRIOR_END_VALUE).is_finite()
                & pl.col(_CONTINUITY_CURRENT_BEGIN_VALUE).is_finite()
            )
            .with_columns(
                (
                    pl.col(_CONTINUITY_CURRENT_BEGIN_VALUE)
                    - pl.col(_CONTINUITY_PRIOR_END_VALUE)
                ).alias(DIFFERENCE),
                pl.max_horizontal(
                    pl.lit(tolerance.absolute),
                    pl.col(_CONTINUITY_PRIOR_END_VALUE).abs()
                    * tolerance.percent
                    / 100.0,
                ).alias(_CONTINUITY_THRESHOLD),
            )
            .filter(pl.col(DIFFERENCE).abs() > pl.col(_CONTINUITY_THRESHOLD))
            .select(
                SNAPSHOT,
                pc_cols.PORTFOLIO_ID,
                pc_cols.FROM_DATE,
                pc_cols.THRU_DATE,
                security_id.alias(pc_cols.SECURITY_ID),
                _CONTINUITY_PRIOR_END_VALUE,
                _CONTINUITY_CURRENT_BEGIN_VALUE,
            )
        )
    if not candidate_plans:
        return pl.DataFrame()
    return pl.concat(candidate_plans).sort(sort_columns).collect()


def _empty_issues_table() -> pl.DataFrame:
    """Return an empty Data Audit Issues table with stable columns."""
    return pl.DataFrame(schema=_ISSUE_SCHEMA)


def _issues_table(rows: list[dict[str, object]]) -> pl.DataFrame:
    """Return sorted Data Audit Issues rows with stable schema."""
    if not rows:
        return _empty_issues_table()
    return pl.DataFrame(rows, schema=_ISSUE_SCHEMA).sort(
        [SNAPSHOT, pc_cols.PORTFOLIO_ID, AS_OF_DATE, ISSUE_TYPE, pc_cols.SECURITY_ID],
        nulls_last=True,
    )


def _snapshot_frames(loader: Any, dataset_name: str) -> tuple[pl.DataFrame, ...]:
    """Load Snapshot A and Snapshot B frames for one optional dataset."""
    frames: list[pl.DataFrame] = []
    for snapshot_key, snapshot_label in _snapshot_labels():
        frame = loader.load(snapshot_key)
        if frame is not None:
            frames.append(frame.with_columns(pl.lit(snapshot_label).alias(SNAPSHOT)))
    return tuple(frames)


def _snapshot_rows(
    frames: Iterable[pl.DataFrame],
) -> tuple[dict[str, object], ...]:
    """Materialize snapshot rows once for all enabled cross-reference checks."""
    return tuple(row for frame in frames for row in frame.iter_rows(named=True))


def _snapshot_labels() -> tuple[tuple[SnapshotKey, str], ...]:
    """Return snapshot loader keys with reviewer-facing labels."""
    return (("a", _SNAPSHOT_A_LABEL), ("b", _SNAPSHOT_B_LABEL))


def _x_ref_config(specification_values: Mapping[str, object]) -> Mapping[str, object]:
    """Return the raw Data Audit YAML configuration mapping."""
    raw_config = specification_values.get(_X_REF_CONFIG_KEY, {})
    if isinstance(raw_config, Mapping):
        return raw_config
    return {}


def _config_enabled(config: Mapping[str, object]) -> bool:
    """Return whether the Data Audit Issues worksheet is enabled."""
    enabled = config.get("enabled", True)
    return not isinstance(enabled, bool) or enabled


def _check_config(config: Mapping[str, object], check_name: str) -> Mapping[str, object]:
    """Return one check's YAML configuration mapping."""
    raw_config = config.get(check_name, {})
    if isinstance(raw_config, Mapping):
        return raw_config
    return {}


def _check_enabled(config: Mapping[str, object], check_name: str) -> bool:
    """Return whether one X-Ref check is enabled."""
    check_config = _check_config(config, check_name)
    enabled = check_config.get("enabled", True)
    return not isinstance(enabled, bool) or enabled


def _tolerance(config: Mapping[str, object], check_name: str) -> _Tolerance:
    """Return configured tolerance for one consistency check."""
    check_config = _check_config(config, check_name)
    return _Tolerance(
        absolute=_float_config(check_config, "absolute_tolerance", 0.0),
        percent=_float_config(
            check_config,
            "percent_tolerance",
            _DEFAULT_PERCENT_TOLERANCE,
        ),
    )


def _float_config(
    config: Mapping[str, object],
    key: str,
    default: float,
) -> float:
    """Return a non-boolean numeric YAML setting."""
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _format_tolerance_number(value: float) -> str:
    """Return a compact tolerance value for reviewer-facing text."""
    if abs(value) >= 1:
        return f"{value:,.2f}"
    if 0 < abs(value) < 0.0001:
        return f"{value:.12f}".rstrip("0").rstrip(".")
    return f"{value:g}"


def _duplicate_transaction_issues(
    transactions: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
) -> list[dict[str, object]]:
    """Return exact duplicate transaction issues."""
    check_name = ISSUE_DUPLICATE_TRANSACTIONS
    if not _check_enabled(config, check_name):
        return []

    row_filter = _row_filter(config, check_name)
    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in transactions:
        if not row_filter.allows(row):
            continue
        key = (
            _text(row.get(SNAPSHOT)),
            _text(row.get(pc_cols.PORTFOLIO_ID)),
            _date(row.get(pc_cols.TRANSACTION_DATE)),
            _text(row.get(pc_cols.SECURITY_ID)),
            _text(row.get(pc_cols.TRANSACTION_CODE)).lower(),
            _number(row.get(pc_cols.AMOUNT)),
            _number(row.get(pc_cols.QUANTITY)),
            _number(row.get(pc_cols.PRICE)),
        )
        groups.setdefault(key, []).append(row)

    rows: list[dict[str, object]] = []
    for group_rows in groups.values():
        if len(group_rows) < 2:
            continue
        duplicate_count = float(len(group_rows))
        for group_row in group_rows:
            rows.append(
                _issue_row(
                    snapshot=_text(group_row.get(SNAPSHOT)),
                    portfolio_id=_text(group_row.get(pc_cols.PORTFOLIO_ID)),
                    as_of_date=_date(group_row.get(pc_cols.TRANSACTION_DATE)),
                    dataset_field="transactions",
                    security_id=_text(group_row.get(pc_cols.SECURITY_ID)),
                    issue_type=ISSUE_DUPLICATE_TRANSACTIONS,
                    value_a=1.0,
                    value_b=duplicate_count,
                    difference=duplicate_count - 1.0,
                    tolerance="unique row",
                    explanation=(
                        "Duplicate transaction rows have the same portfolio, date, "
                        "security, code, amount, quantity, and price."
                    ),
                )
            )
    return rows


def _holding_price_range_issues(
    holdings: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
) -> list[dict[str, object]]:
    """Return same-day same-security holdings.price range issues."""
    check_name = "holdings_price_range"
    if not _check_enabled(config, check_name):
        return []

    row_filter = _row_filter(config, check_name)
    groups: dict[tuple[str, dt.date, str], list[Mapping[str, object]]] = {}
    for row in holdings:
        if not row_filter.allows(row):
            continue
        holding_date = _date(row.get(pc_cols.HOLDING_DATE))
        security_id = _text(row.get(pc_cols.SECURITY_ID))
        snapshot = _text(row.get(SNAPSHOT))
        price = _number(row.get(pc_cols.PRICE))
        if holding_date is None or price is None or not security_id:
            continue
        groups.setdefault((snapshot, holding_date, security_id), []).append(row)

    return _price_range_issues(
        groups=groups,
        config=config,
        check_name=check_name,
        date_column=pc_cols.HOLDING_DATE,
        dataset_field="holdings.price",
        issue_type=ISSUE_HOLDINGS_PRICE_RANGE,
        explanation_prefix="Same-day same-security holdings.price",
    )


def _transaction_price_range_issues(
    transactions: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
) -> list[dict[str, object]]:
    """Return same-day same-security transactions.price range issues."""
    check_name = "transactions_price_range"
    if not _check_enabled(config, check_name):
        return []

    row_filter = _row_filter(config, check_name)
    groups: dict[tuple[str, dt.date, str], list[Mapping[str, object]]] = {}
    for row in transactions:
        if not row_filter.allows(row):
            continue
        transaction_date = _date(row.get(pc_cols.TRANSACTION_DATE))
        security_id = _text(row.get(pc_cols.SECURITY_ID))
        snapshot = _text(row.get(SNAPSHOT))
        price = _number(row.get(pc_cols.PRICE))
        if transaction_date is None or price is None or not security_id:
            continue
        groups.setdefault((snapshot, transaction_date, security_id), []).append(
            row
        )

    return _price_range_issues(
        groups=groups,
        config=config,
        check_name=check_name,
        date_column=pc_cols.TRANSACTION_DATE,
        dataset_field="transactions.price",
        issue_type=ISSUE_TRANSACTIONS_PRICE_RANGE,
        explanation_prefix="Same-day same-security transactions.price",
    )


def _price_range_issues(
    *,
    groups: Mapping[tuple[str, dt.date, str], list[Mapping[str, object]]],
    config: Mapping[str, object],
    check_name: str,
    date_column: str,
    dataset_field: str,
    issue_type: str,
    explanation_prefix: str,
) -> list[dict[str, object]]:
    """Return price-range issues from same-snapshot same-date security groups."""
    tolerance = _tolerance(config, check_name)
    rows: list[dict[str, object]] = []
    for (snapshot, as_of_date, security_id), group_rows in groups.items():
        prices = [_number(row.get(pc_cols.PRICE)) for row in group_rows]
        numeric_prices = [price for price in prices if price is not None]
        if len(numeric_prices) < 2:
            continue
        minimum_price = min(numeric_prices)
        maximum_price = max(numeric_prices)
        difference = maximum_price - minimum_price
        if difference <= tolerance.threshold(maximum_price):
            continue
        for group_row in group_rows:
            rows.append(
                _issue_row(
                    snapshot=snapshot,
                    portfolio_id=_text(group_row.get(pc_cols.PORTFOLIO_ID)),
                    as_of_date=_date(group_row.get(date_column)),
                    dataset_field=dataset_field,
                    security_id=security_id,
                    issue_type=issue_type,
                    value_a=minimum_price,
                    value_b=maximum_price,
                    difference=difference,
                    tolerance=tolerance.description(),
                    explanation=(
                        f"{explanation_prefix} differs across portfolios "
                        f"for {security_id}."
                    ),
                )
            )
    return rows


def _same_day_rate_issues(
    transactions: Sequence[Mapping[str, object]],
    holdings: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
) -> list[dict[str, object]]:
    """Return same-day same-security transaction-rate issues."""
    enabled_checks = (
        _check_enabled(config, "dividend_rate"),
        _check_enabled(config, "pa_sa_rate"),
    )
    if not any(enabled_checks):
        return []
    holdings_by_key = _holdings_by_portfolio_security(holdings)
    rows: list[dict[str, object]] = []
    if enabled_checks[0]:
        rows.extend(
            _transaction_rate_issues(
                transactions,
                holdings_by_key,
                config,
                check_name="dividend_rate",
                transaction_codes=_DIVIDEND_CODES,
                issue_type=ISSUE_DIVIDEND_RATE,
                dataset_field="transactions.amount",
            )
        )
    if enabled_checks[1]:
        rows.extend(
            _transaction_rate_issues(
                transactions,
                holdings_by_key,
                config,
                check_name="pa_sa_rate",
                transaction_codes=_ACCRUAL_CODES,
                issue_type=ISSUE_PA_SA_RATE,
                dataset_field="transactions.amount",
            )
        )
    return rows


def _transaction_rate_issues(
    transactions: Sequence[Mapping[str, object]],
    holdings_by_key: Mapping[
        tuple[str, str, str], tuple[Mapping[str, object], ...]
    ],
    config: Mapping[str, object],
    *,
    check_name: str,
    transaction_codes: frozenset[str],
    issue_type: str,
    dataset_field: str,
) -> list[dict[str, object]]:
    """Return transaction-rate issues for a configured transaction-code family."""
    if not _check_enabled(config, check_name):
        return []

    tolerance = _tolerance(config, check_name)
    row_filter = _row_filter(config, check_name)
    groups: dict[tuple[str, dt.date, str, str], list[Mapping[str, object]]] = {}
    for row in transactions:
        code = _text(row.get(pc_cols.TRANSACTION_CODE)).lower()
        if code not in transaction_codes:
            continue
        if not row_filter.allows(row):
            continue
        transaction_date = _date(row.get(pc_cols.TRANSACTION_DATE))
        security_id = _text(row.get(pc_cols.SECURITY_ID))
        snapshot = _text(row.get(SNAPSHOT))
        rate = _transaction_rate(row, holdings_by_key=holdings_by_key)
        if rate is None or transaction_date is None or not security_id:
            continue
        groups.setdefault((snapshot, transaction_date, security_id, code), []).append(row)

    rows: list[dict[str, object]] = []
    for (snapshot, transaction_date, security_id, code), group_rows in groups.items():
        rates = [
            _transaction_rate(group_row, holdings_by_key=holdings_by_key)
            for group_row in group_rows
        ]
        numeric_rates = [rate for rate in rates if rate is not None]
        if len(numeric_rates) < 2:
            continue
        minimum_rate = min(numeric_rates)
        maximum_rate = max(numeric_rates)
        difference = maximum_rate - minimum_rate
        if difference <= tolerance.threshold(maximum_rate):
            continue
        for group_row in group_rows:
            rows.append(
                _issue_row(
                    snapshot=snapshot,
                    portfolio_id=_text(group_row.get(pc_cols.PORTFOLIO_ID)),
                    as_of_date=transaction_date,
                    dataset_field=dataset_field,
                    security_id=security_id,
                    issue_type=issue_type,
                    value_a=minimum_rate,
                    value_b=maximum_rate,
                    difference=difference,
                    tolerance=tolerance.description(),
                    explanation=(
                        f"{code}: Same-day same-security rate differs across "
                        f"portfolios for {security_id}."
                    ),
                )
            )
    return rows


def _transaction_rate(
    row: Mapping[str, object],
    *,
    holdings_by_key: Mapping[tuple[str, str, str], tuple[Mapping[str, object], ...]]
    | None = None,
) -> float | None:
    """Return amount-per-unit transaction rate for one row."""
    amount = _number(row.get(pc_cols.AMOUNT))
    quantity = _transaction_rate_quantity(row, holdings_by_key=holdings_by_key)
    if amount is None or quantity is None or quantity == 0:
        return None
    return abs(amount) / abs(quantity)


def _transaction_rate_quantity(
    row: Mapping[str, object],
    *,
    holdings_by_key: Mapping[tuple[str, str, str], tuple[Mapping[str, object], ...]]
    | None,
) -> float | None:
    """Return the quantity basis for a transaction-rate check."""
    quantity = _number(row.get(pc_cols.QUANTITY))
    if quantity is not None and quantity != 0:
        return quantity
    if holdings_by_key is None:
        return None
    transaction_date = _date(row.get(pc_cols.TRANSACTION_DATE))
    if transaction_date is None:
        return None
    key = (
        _text(row.get(SNAPSHOT)),
        _text(row.get(pc_cols.PORTFOLIO_ID)),
        _text(row.get(pc_cols.SECURITY_ID)),
    )
    for holding_row in holdings_by_key.get(key, ()):
        holding_date = _date(holding_row.get(pc_cols.HOLDING_DATE))
        if holding_date is not None and holding_date >= transaction_date:
            return _number(holding_row.get(pc_cols.QUANTITY))
    return None


def _missing_dividend_issues(
    holdings: Sequence[Mapping[str, object]],
    transactions: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
) -> list[dict[str, object]]:
    """Return portfolios that appear to be missing same-date dividends."""
    check_name = "missing_dividend"
    if not _check_enabled(config, check_name):
        return []

    row_filter = _row_filter(config, check_name)
    dividend_rows = [
        row
        for row in transactions
        if _text(row.get(pc_cols.TRANSACTION_CODE)).lower() in _DIVIDEND_CODES
    ]
    holdings_by_security: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in holdings:
        security_key = (
            _text(row.get(SNAPSHOT)),
            _text(row.get(pc_cols.SECURITY_ID)),
        )
        holdings_by_security.setdefault(security_key, []).append(row)
    transactions_by_position: dict[
        tuple[str, str, str], list[Mapping[str, object]]
    ] = {}
    for row in transactions:
        position_key = (
            _text(row.get(SNAPSHOT)),
            _text(row.get(pc_cols.PORTFOLIO_ID)),
            _text(row.get(pc_cols.SECURITY_ID)),
        )
        transactions_by_position.setdefault(position_key, []).append(row)
    dividend_keys = {
        (
            _text(row.get(SNAPSHOT)),
            _text(row.get(pc_cols.PORTFOLIO_ID)),
            _text(row.get(pc_cols.SECURITY_ID)),
            _date(row.get(pc_cols.TRANSACTION_DATE)),
        )
        for row in dividend_rows
    }
    dividend_portfolios_by_key: dict[tuple[str, str, dt.date | None], set[str]] = {}
    for row in dividend_rows:
        dividend_key = (
            _text(row.get(SNAPSHOT)),
            _text(row.get(pc_cols.SECURITY_ID)),
            _date(row.get(pc_cols.TRANSACTION_DATE)),
        )
        dividend_portfolios_by_key.setdefault(dividend_key, set()).add(
            _text(row.get(pc_cols.PORTFOLIO_ID))
        )

    dividend_rows_by_event: dict[
        tuple[str, str, dt.date], list[Mapping[str, object]]
    ] = {}
    for dividend_row in dividend_rows:
        if not row_filter.allows(dividend_row):
            continue
        snapshot = _text(dividend_row.get(SNAPSHOT))
        security_id = _text(dividend_row.get(pc_cols.SECURITY_ID))
        dividend_date = _date(dividend_row.get(pc_cols.TRANSACTION_DATE))
        if dividend_date is None or not security_id:
            continue
        dividend_rows_by_event.setdefault(
            (snapshot, security_id, dividend_date),
            [],
        ).append(dividend_row)

    rows: list[dict[str, object]] = []
    for dividend_key, event_rows in dividend_rows_by_event.items():
        snapshot, security_id, dividend_date = dividend_key
        representative_row = min(
            event_rows,
            key=lambda row: _text(row.get(pc_cols.PORTFOLIO_ID)),
        )
        held_portfolios = _held_portfolios(
            holdings_by_security.get((snapshot, security_id), ()),
            transactions_by_position,
            snapshot=snapshot,
            security_id=security_id,
            dividend_date=dividend_date,
        )
        for portfolio_id in held_portfolios:
            if (
                snapshot,
                portfolio_id,
                security_id,
                dividend_date,
            ) in dividend_keys:
                continue
            if not row_filter.allows(
                {
                    SNAPSHOT: snapshot,
                    pc_cols.PORTFOLIO_ID: portfolio_id,
                    pc_cols.SECURITY_ID: security_id,
                }
            ):
                continue
            rows.append(
                _issue_row(
                    snapshot=snapshot,
                    portfolio_id=portfolio_id,
                    as_of_date=dividend_date,
                    dataset_field="transactions.amount",
                    security_id=security_id,
                    issue_type=ISSUE_MISSING_DIVIDEND,
                    value_a=_transaction_rate(representative_row),
                    value_b=None,
                    difference=None,
                    tolerance="same-date dividend present",
                    explanation=(
                        _missing_dividend_explanation(
                            security_id=security_id,
                            dividend_date=dividend_date,
                            dividend_portfolios=dividend_portfolios_by_key.get(
                                (snapshot, security_id, dividend_date),
                                set(),
                            ),
                        )
                    ),
                )
            )
    return rows


def _missing_dividend_explanation(
    *,
    security_id: str,
    dividend_date: dt.date,
    dividend_portfolios: set[str],
) -> str:
    """Return reviewer-facing missing-dividend explanation text."""
    sorted_portfolios = sorted(
        portfolio for portfolio in dividend_portfolios if portfolio
    )
    if not sorted_portfolios:
        return f"Missing a dividend for {security_id} on {dividend_date}."
    source_text = f"portfolio {sorted_portfolios[0]}"
    if len(sorted_portfolios) > 1:
        source_text = f"{source_text} and other portfolios"
    return (
        f"Missing a dividend for {security_id} on {dividend_date} "
        f"that is in {source_text}."
    )


def _held_portfolios(
    holding_rows: Iterable[Mapping[str, object]],
    transactions_by_position: Mapping[
        tuple[str, str, str], Iterable[Mapping[str, object]]
    ],
    *,
    snapshot: str,
    security_id: str,
    dividend_date: dt.date,
) -> tuple[str, ...]:
    """Return portfolios that conservatively appear eligible for a dividend."""
    by_portfolio: dict[str, list[Mapping[str, object]]] = {}
    for row in holding_rows:
        portfolio_id = _text(row.get(pc_cols.PORTFOLIO_ID))
        by_portfolio.setdefault(portfolio_id, []).append(row)

    held_portfolios: list[str] = []
    for portfolio_id, rows in by_portfolio.items():
        dated_rows = sorted(
            (row for row in rows if _date(row.get(pc_cols.HOLDING_DATE)) is not None),
            key=lambda row: _date(row.get(pc_cols.HOLDING_DATE)) or dt.date.min,
        )
        for previous_row, current_row in zip(dated_rows, dated_rows[1:]):
            previous_date = _date(previous_row.get(pc_cols.HOLDING_DATE))
            current_date = _date(current_row.get(pc_cols.HOLDING_DATE))
            if previous_date is None or current_date is None:
                continue
            if not previous_date < dividend_date <= current_date:
                continue
            if not _missing_dividend_position_qualifies(
                transactions_by_position.get(
                    (snapshot, portfolio_id, security_id), ()
                ),
                start_date=previous_date,
                dividend_date=dividend_date,
                beginning_quantity=_number(previous_row.get(pc_cols.QUANTITY)),
            ):
                continue
            held_portfolios.append(portfolio_id)
            break
    return tuple(sorted(set(held_portfolios)))


def _missing_dividend_position_qualifies(
    transaction_rows: Iterable[Mapping[str, object]],
    *,
    start_date: dt.date,
    dividend_date: dt.date,
    beginning_quantity: float | None,
) -> bool:
    """Return whether a portfolio qualifies for conservative dividend review."""
    has_beginning_position = _positive(beginning_quantity)
    has_pre_dividend_buy = False
    for row in transaction_rows:
        transaction_date = _date(row.get(pc_cols.TRANSACTION_DATE))
        if transaction_date is None or not start_date < transaction_date < dividend_date:
            continue
        transaction_code = _text(row.get(pc_cols.TRANSACTION_CODE)).lower()
        if transaction_code not in _BUY_CODES:
            return False
        if _positive(_number(row.get(pc_cols.QUANTITY))):
            has_pre_dividend_buy = True
    return has_beginning_position or has_pre_dividend_buy


def _holdings_accrued_rate_issues(
    holdings: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
) -> list[dict[str, object]]:
    """Return same-day same-security holdings.accrued rate issues."""
    check_name = "holdings_accrued_rate"
    if not _check_enabled(config, check_name):
        return []

    tolerance = _tolerance(config, check_name)
    row_filter = _row_filter(config, check_name)
    groups: dict[tuple[str, dt.date, str], list[Mapping[str, object]]] = {}
    for row in holdings:
        if not row_filter.allows(row):
            continue
        rate = _holdings_accrued_rate(row)
        holding_date = _date(row.get(pc_cols.HOLDING_DATE))
        security_id = _text(row.get(pc_cols.SECURITY_ID))
        snapshot = _text(row.get(SNAPSHOT))
        if rate is None or holding_date is None or not security_id:
            continue
        groups.setdefault((snapshot, holding_date, security_id), []).append(row)

    rows: list[dict[str, object]] = []
    for (snapshot, holding_date, security_id), group_rows in groups.items():
        rates = [_holdings_accrued_rate(row) for row in group_rows]
        numeric_rates = [rate for rate in rates if rate is not None]
        if len(numeric_rates) < 2:
            continue
        minimum_rate = min(numeric_rates)
        maximum_rate = max(numeric_rates)
        difference = maximum_rate - minimum_rate
        if difference <= tolerance.threshold(maximum_rate):
            continue
        for group_row in group_rows:
            rows.append(
                _issue_row(
                    snapshot=snapshot,
                    portfolio_id=_text(group_row.get(pc_cols.PORTFOLIO_ID)),
                    as_of_date=holding_date,
                    dataset_field="holdings.accrued",
                    security_id=security_id,
                    issue_type=ISSUE_HOLDINGS_ACCRUED_RATE,
                    value_a=minimum_rate,
                    value_b=maximum_rate,
                    difference=difference,
                    tolerance=tolerance.description(),
                    explanation=(
                        "Same-day same-security holdings.accrued per unit differs "
                        f"across portfolios for {security_id}."
                    ),
                )
            )
    return rows


def _holdings_accrued_rate(row: Mapping[str, object]) -> float | None:
    """Return accrued-per-unit rate for one holding row."""
    accrued = _number(row.get(pc_cols.ACCRUED))
    quantity = _number(row.get(pc_cols.QUANTITY))
    if accrued is None or quantity is None or quantity == 0:
        return None
    return accrued / abs(quantity)


def _issue_row(
    *,
    snapshot: str,
    portfolio_id: str,
    as_of_date: dt.date | None,
    dataset_field: str,
    security_id: str,
    issue_type: str,
    value_a: float | None,
    value_b: float | None,
    difference: float | None,
    tolerance: str,
    explanation: str,
) -> dict[str, object]:
    """Return one Data Audit Issues row."""
    return {
        SNAPSHOT: snapshot,
        pc_cols.PORTFOLIO_ID: portfolio_id,
        AS_OF_DATE: as_of_date,
        DATASET_FIELD: dataset_field,
        pc_cols.SECURITY_ID: security_id,
        ISSUE_TYPE: issue_type,
        VALUE_A: value_a,
        VALUE_B: value_b,
        DIFFERENCE: difference,
        TOLERANCE: tolerance,
        EXPLANATION: explanation,
        REVIEW_KEY: "::".join(
            str(part)
            for part in (
                "XREF",
                snapshot,
                portfolio_id,
                as_of_date,
                dataset_field,
                security_id,
                issue_type,
            )
        ),
    }


def _holdings_by_portfolio_security(
    holding_rows: Iterable[Mapping[str, object]],
) -> dict[tuple[str, str, str], tuple[Mapping[str, object], ...]]:
    """Return holding rows keyed by snapshot, portfolio, and security."""
    grouped_rows: dict[tuple[str, str, str], list[Mapping[str, object]]] = {}
    for row in holding_rows:
        key = (
            _text(row.get(SNAPSHOT)),
            _text(row.get(pc_cols.PORTFOLIO_ID)),
            _text(row.get(pc_cols.SECURITY_ID)),
        )
        grouped_rows.setdefault(key, []).append(row)
    return {
        key: tuple(
            sorted(
                rows,
                key=lambda row: _date(row.get(pc_cols.HOLDING_DATE)) or dt.date.max,
            )
        )
        for key, rows in grouped_rows.items()
    }


def _row_allowed(
    row: Mapping[str, object],
    config: Mapping[str, object],
    check_name: str,
) -> bool:
    """Return whether one row passes a check's optional include/exclude filters."""
    return _row_filter(config, check_name).allows(row)


def _row_filter(config: Mapping[str, object], check_name: str) -> _RowFilter:
    """Compile one check's row filters for reuse across all source rows."""
    check_config = _check_config(config, check_name)
    return _RowFilter(
        only=_compiled_filters(check_config.get("only", {})),
        exclude=_compiled_filters(check_config.get("exclude", {})),
    )


def _compiled_filters(
    filter_config: object,
) -> tuple[tuple[str, frozenset[str]], ...]:
    """Normalize a YAML row-filter mapping once for repeated matching."""
    compiled: list[tuple[str, frozenset[str]]] = []
    for field_name, raw_values in _filter_mapping(filter_config):
        values = _text_filter_values(raw_values)
        if values:
            compiled.append((_filter_column_name(field_name), values))
    return tuple(compiled)


def _matches_only(row: Mapping[str, object], filter_config: object) -> bool:
    """Return whether a row matches every configured include filter."""
    filters = _filter_mapping(filter_config)
    if not filters:
        return True
    return all(
        _field_matches(row, field_name, raw_values)
        for field_name, raw_values in filters
    )


def _matches_exclude(row: Mapping[str, object], filter_config: object) -> bool:
    """Return whether a row matches any configured exclude filter."""
    filters = _filter_mapping(filter_config)
    return any(
        _field_matches(row, field_name, raw_values)
        for field_name, raw_values in filters
    )


def _field_matches(
    row: Mapping[str, object],
    field_name: str,
    raw_values: object,
) -> bool:
    """Return whether one row field matches one scalar-or-list filter."""
    values = _text_filter_values(raw_values)
    if not values:
        return False
    column_name = _filter_column_name(field_name)
    return _text(row.get(column_name)).lower() in values


def _filter_mapping(filter_config: object) -> tuple[tuple[str, object], ...]:
    """Return a stable field/value filter mapping from YAML configuration."""
    if not isinstance(filter_config, Mapping):
        return ()
    return tuple(
        (str(field_name), raw_values)
        for field_name, raw_values in filter_config.items()
    )


def _filter_column_name(field_name: str) -> str:
    """Return the normalized row column name for a YAML filter field."""
    normalized_name = field_name.strip().lower()
    if "." in normalized_name:
        normalized_name = normalized_name.rsplit(".", maxsplit=1)[-1]
    return _FILTER_FIELD_ALIASES.get(normalized_name, normalized_name)


def _text_filter_values(value: object) -> frozenset[str]:
    """Return lower-cased exact-match values from a YAML scalar or sequence."""
    if isinstance(value, str):
        return frozenset({value.strip().lower()})
    if isinstance(value, Iterable):
        return frozenset(
            str(item).strip().lower()
            for item in value
            if str(item).strip()
        )
    return frozenset()


def _number(value: object) -> float | None:
    """Return a finite float for numeric source values."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        numeric_value = float(value)
        if numeric_value == numeric_value and numeric_value not in {
            float("inf"),
            float("-inf"),
        }:
            return numeric_value
    return None


def _text(value: object) -> str:
    """Return a stripped text value for source identifiers."""
    if value is None:
        return ""
    return str(value).strip()


def _date(value: object) -> dt.date | None:
    """Return a Python date from a Polars or Python date value."""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return None


def _positive(value: float | None) -> bool:
    """Return whether a numeric value is materially positive."""
    return value is not None and value > 0
