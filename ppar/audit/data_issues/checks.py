"""Source-data consistency checks for Audit reports.

The checks in this module are intentionally separate from ``source_detail.csv``.
Source-detail rows explain changed source rows between Snapshot A and Snapshot B;
Data Issues look for internally inconsistent source-data across the union
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
from ppar.errors import PpaError
from ppar.audit import schema as pc_cols
from ppar.audit.data_issues.config import (
    DATA_ISSUES_CONFIG_KEY,
    required_transaction_columns,
)
from ppar.audit.holdings import HoldingsLoader
from ppar.audit.portfolio_performance import (
    PortfolioPerformanceLoader,
    SnapshotKey,
)
from ppar.audit.security_performance import SecurityPerformanceLoader
from ppar.audit.security_reference import SecurityReferenceLoader
from ppar.audit.splits import SplitsLoader
from ppar.audit.specification import AuditSpecification
from ppar.audit.extract_contract import transaction_semantics_exact_case
from ppar.audit.transaction_policy import (
    transaction_boundary_codes,
    transaction_code_matching_key,
)
from ppar.audit.transactions import TransactionsLoader
from ppar.audit.data_issues.vocabulary import DATA_ISSUE_REGISTRY, DataIssueType

SNAPSHOT: Final[str] = "snapshot"
AS_OF_DATE: Final[str] = "as_of_date"
ISSUE_TYPE: Final[str] = "issue_type"
CATEGORY: Final[str] = "category"
DATASET_FIELD: Final[str] = "dataset_field"
VALUE_A: Final[str] = "value_a"
VALUE_B: Final[str] = "value_b"
DIFFERENCE: Final[str] = "difference"
TOLERANCE: Final[str] = "tolerance"
EXPLANATION: Final[str] = "explanation"
REVIEW_KEY: Final[str] = "review_key"

ISSUE_DUPLICATE_TRANSACTIONS: Final[str] = DataIssueType.DUPLICATE_TRANSACTIONS.value
ISSUE_DELIVER_IN_ORIGINAL_COST_INCOMPLETE: Final[str] = (
    DataIssueType.DELIVER_IN_ORIGINAL_COST_INCOMPLETE.value
)
ISSUE_DIVIDEND_RATE: Final[str] = DataIssueType.DIVIDEND_RATE.value
ISSUE_HOLDINGS_ACCRUED_RATE: Final[str] = DataIssueType.HOLDINGS_ACCRUED_RATE.value
ISSUE_HOLDINGS_NONPOSITIVE_PRICE: Final[str] = (
    DataIssueType.HOLDINGS_NONPOSITIVE_PRICE.value
)
ISSUE_HOLDINGS_PRICE_RANGE: Final[str] = DataIssueType.HOLDINGS_PRICE_RANGE.value
ISSUE_HOLDINGS_STALE_PRICE: Final[str] = DataIssueType.HOLDINGS_STALE_PRICE.value
ISSUE_LARGE_PRICE_VARIATION: Final[str] = (
    DataIssueType.LARGE_PRICE_VARIATION.value
)
ISSUE_MISSING_DIVIDEND: Final[str] = DataIssueType.MISSING_DIVIDEND.value
ISSUE_PA_SA_RATE: Final[str] = DataIssueType.PA_SA_RATE.value
ISSUE_PORTFOLIO_MV_CONTINUITY: Final[str] = (
    DataIssueType.PORTFOLIO_MARKET_VALUE_CONTINUITY.value
)
ISSUE_SECURITY_MV_CONTINUITY: Final[str] = (
    DataIssueType.SECURITY_MARKET_VALUE_CONTINUITY.value
)
ISSUE_TRANSACTION_SECURITY_TYPE_MISMATCH: Final[str] = (
    DataIssueType.TRANSACTION_SECURITY_TYPE_MISMATCH.value
)
ISSUE_TRANSACTIONS_NONPOSITIVE_PRICE: Final[str] = (
    DataIssueType.TRANSACTIONS_NONPOSITIVE_PRICE.value
)
ISSUE_TRANSACTIONS_PRICE_RANGE: Final[str] = (
    DataIssueType.TRANSACTIONS_PRICE_RANGE.value
)

DATA_ISSUE_COLUMNS: Final[tuple[str, ...]] = (
    SNAPSHOT,
    pc_cols.PORTFOLIO_ID,
    AS_OF_DATE,
    DATASET_FIELD,
    pc_cols.SECURITY_ID,
    ISSUE_TYPE,
    CATEGORY,
    VALUE_A,
    VALUE_B,
    DIFFERENCE,
    TOLERANCE,
    EXPLANATION,
    REVIEW_KEY,
)

_DATA_ISSUES_CONFIG_KEY: Final[str] = DATA_ISSUES_CONFIG_KEY
_SNAPSHOT_A_LABEL: Final[str] = "Snapshot A"
_SNAPSHOT_B_LABEL: Final[str] = "Snapshot B"
_BUY_CODES: Final[frozenset[str]] = transaction_boundary_codes("data_issue_buy")
_DIVIDEND_CODES: Final[frozenset[str]] = transaction_boundary_codes(
    "data_issue_dividend"
)
_ACCRUAL_CODES: Final[frozenset[str]] = transaction_boundary_codes(
    "data_issue_accrual"
)
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
    "source_destination_type": pc_cols.SOURCE_DESTINATION_TYPE,
    "source_destination_symbol": pc_cols.SOURCE_DESTINATION_SYMBOL,
}
_SECURITY_REFERENCE_PREFIX: Final[str] = f"{pc_cols.SECURITY_REFERENCE}."
_HOLDING_FILTER_ISSUES: Final[frozenset[str]] = frozenset(
    {
        ISSUE_HOLDINGS_ACCRUED_RATE,
        ISSUE_HOLDINGS_NONPOSITIVE_PRICE,
        ISSUE_HOLDINGS_PRICE_RANGE,
        ISSUE_HOLDINGS_STALE_PRICE,
    }
)
_TRANSACTION_FILTER_ISSUES: Final[frozenset[str]] = frozenset(
    {
        ISSUE_DUPLICATE_TRANSACTIONS,
        ISSUE_DELIVER_IN_ORIGINAL_COST_INCOMPLETE,
        ISSUE_DIVIDEND_RATE,
        ISSUE_MISSING_DIVIDEND,
        ISSUE_PA_SA_RATE,
        ISSUE_TRANSACTION_SECURITY_TYPE_MISMATCH,
        ISSUE_TRANSACTIONS_NONPOSITIVE_PRICE,
        ISSUE_TRANSACTIONS_PRICE_RANGE,
    }
)

_ISSUE_SCHEMA: Final[dict[str, type[pl.DataType]]] = {
    SNAPSHOT: pl.String,
    pc_cols.PORTFOLIO_ID: pl.String,
    AS_OF_DATE: pl.Date,
    DATASET_FIELD: pl.String,
    pc_cols.SECURITY_ID: pl.String,
    ISSUE_TYPE: pl.String,
    CATEGORY: pl.String,
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
class _CompiledFilter:
    """One normalized YAML filter condition.

    Attributes:
        column: Enriched or native row column to inspect.
        values: Accepted scalar strings.
        exact_case: Whether matching preserves case.
    """

    column: str
    values: frozenset[str]
    exact_case: bool

    def matches(self, row: Mapping[str, object]) -> bool:
        """Return whether one row matches this condition."""
        value = _text(row.get(self.column))
        candidate = value if self.exact_case else value.lower()
        return candidate in self.values


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

    only: tuple[_CompiledFilter, ...]
    exclude: tuple[_CompiledFilter, ...]

    def allows(self, row: Mapping[str, object]) -> bool:
        """Return whether one row passes the compiled filters."""
        if self.only and not all(condition.matches(row) for condition in self.only):
            return False
        if not self.exclude:
            return True
        return not any(condition.matches(row) for condition in self.exclude)


@dataclass(frozen=True)
class _LargePriceVariationRule:
    """One validated named rule for period-level price observations.

    Attributes:
        rule_id: Stable configuration and review-key identity.
        minimum_calendar_days: Minimum inclusive performance-period length.
        minimum_tolerance: Minimum decimal price variation, such as ``0.20``.
        holdings_filter: Filters applicable to holdings observations.
        transactions_filter: Filters applicable to transaction observations.
    """

    rule_id: str
    minimum_calendar_days: int
    minimum_tolerance: float
    holdings_filter: _RowFilter
    transactions_filter: _RowFilter


@dataclass(frozen=True)
class _PriceObservation:
    """One raw and split-normalized comparable price observation.

    Attributes:
        observation_date: Holding valuation date or transaction trade date.
        source: Reviewer-facing source and boundary meaning.
        source_rank: Stable tie-break rank; holdings precede transactions.
        source_order: Original row order within the loaded snapshot union.
        raw_price: Positive source price before split normalization.
        adjusted_price: Price expressed on the performance-period ending basis.
        cumulative_split_factor: Product applied to the raw price.
        currency: Normalized source price currency, when supplied.
    """

    observation_date: dt.date
    source: str
    source_rank: int
    source_order: int
    raw_price: float
    adjusted_price: float
    cumulative_split_factor: float
    currency: str


def data_issues_table(comparison_path: util.PathLike | None) -> pl.DataFrame:
    """Return source-data consistency issues for an Audit YAML file.

    Args:
        comparison_path: Audit YAML path. When omitted, an
            empty table is returned because the source snapshots are unavailable.

    Returns:
        Data Issues rows built from the union of Snapshot A and Snapshot B.
    """
    if comparison_path is None:
        return _empty_issues_table()

    specification = AuditSpecification(comparison_path)
    exact_case = transaction_semantics_exact_case(
        specification.values,
        specification_path=specification.path,
    )
    config = _data_issues_config(specification.values)
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
    _validate_required_transaction_columns(
        transactions,
        required_columns=required_transaction_columns(specification.values),
        specification=specification,
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
    reference_requirements = _security_reference_requirements(config)
    if reference_requirements:
        reference_maps = _security_reference_maps(
            specification,
            required_columns=frozenset(
                column
                for columns in reference_requirements.values()
                for column in columns
            ),
        )
        if reference_requirements.get(pc_cols.HOLDINGS):
            holding_rows = _enrich_with_security_reference(
                holding_rows,
                reference_maps,
                reference_requirements[pc_cols.HOLDINGS],
                dataset_name=pc_cols.HOLDINGS,
                specification=specification,
            )
        if reference_requirements.get(pc_cols.TRANSACTIONS):
            transaction_rows = _enrich_with_security_reference(
                transaction_rows,
                reference_maps,
                reference_requirements[pc_cols.TRANSACTIONS],
                dataset_name=pc_cols.TRANSACTIONS,
                specification=specification,
            )
    split_rows: tuple[dict[str, object], ...] = ()
    if _check_enabled(config, ISSUE_LARGE_PRICE_VARIATION):
        split_rows = _snapshot_rows(
            _snapshot_frames(SplitsLoader(specification), pc_cols.SPLITS)
        )
    rows.extend(
        _duplicate_transaction_issues(
            transaction_rows,
            config,
            exact_case=exact_case,
        )
    )
    rows.extend(
        _deliver_in_original_cost_incomplete_issues(
            transaction_rows,
            config,
            exact_case=exact_case,
        )
    )
    rows.extend(_holdings_nonpositive_price_issues(holding_rows, config))
    rows.extend(_holding_price_range_issues(holding_rows, config))
    rows.extend(_holdings_stale_price_issues(holding_rows, config))
    rows.extend(
        _large_price_variation_issues(
            _snapshot_rows(portfolio_performance),
            holding_rows,
            transaction_rows,
            split_rows,
            config,
        )
    )
    rows.extend(_transaction_security_type_mismatch_issues(transaction_rows, config))
    rows.extend(_transactions_nonpositive_price_issues(transaction_rows, config))
    rows.extend(_transaction_price_range_issues(transaction_rows, config))
    rows.extend(
        _same_day_rate_issues(
            transaction_rows,
            holding_rows,
            config,
            exact_case=exact_case,
        )
    )
    rows.extend(
        _missing_dividend_issues(
            holding_rows,
            transaction_rows,
            config,
            exact_case=exact_case,
        )
    )
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
    Issues checks, it remains active when ``data_issues.enabled`` is false
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
        if current_from is None or prior_end is None or current_begin is None:
            raise PpaError(
                "SN-04 continuity candidate contains invalid typed values.",
                999,
                context={
                    "dataset": dataset_name,
                    "portfolio_id": current.get(pc_cols.PORTFOLIO_ID),
                },
            )
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
    """Return an empty Data Issues table with stable columns."""
    return pl.DataFrame(schema=_ISSUE_SCHEMA)


def _issues_table(rows: list[dict[str, object]]) -> pl.DataFrame:
    """Return sorted Data Issues rows with stable schema."""
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


def _validate_required_transaction_columns(
    frames: Sequence[pl.DataFrame],
    *,
    required_columns: frozenset[str],
    specification: AuditSpecification,
) -> None:
    """Fail when an enabled transaction check lacks required source columns."""
    if not required_columns:
        return
    if len(frames) != len(_snapshot_labels()):
        raise PpaError(
            (
                f"{specification.path}: enabled Data Issues checks require "
                "transactions for both snapshots."
            ),
            504,
        )
    for frame, (_, snapshot) in zip(frames, _snapshot_labels()):
        missing_columns = sorted(required_columns - set(frame.columns))
        if not missing_columns:
            continue
        raise PpaError(
            (
                f"{specification.path}: transactions for {snapshot} are missing "
                "columns required by enabled Data Issues checks: "
                f"{', '.join(missing_columns)}."
            ),
            504,
        )


def _deliver_in_original_cost_incomplete_issues(
    transactions: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
    *,
    exact_case: bool,
) -> list[dict[str, object]]:
    """Return configured deliver-ins with incomplete original-cost evidence."""
    check_name = ISSUE_DELIVER_IN_ORIGINAL_COST_INCOMPLETE
    if not _check_enabled(config, check_name):
        return []

    row_filter = _row_filter(
        config,
        check_name,
        exact_native_case=exact_case,
    )
    rows: list[dict[str, object]] = []
    occurrence_counts: dict[tuple[str, ...], int] = {}
    for row in transactions:
        if not row_filter.allows(row):
            continue
        transaction_date = _date(row.get(pc_cols.TRANSACTION_DATE))
        security_id = _text(row.get(pc_cols.SECURITY_ID))
        if transaction_date is None or not security_id:
            continue
        missing_fields = [
            field
            for field in (pc_cols.ORIGINAL_COST, pc_cols.ORIGINAL_COST_DATE)
            if row.get(field) is None
        ]
        if not missing_fields:
            continue
        transaction_code = _text(row.get(pc_cols.TRANSACTION_CODE))
        missing_text = " and ".join(f"transactions.{field}" for field in missing_fields)
        dataset_field = " + ".join(
            f"transactions.{field}" for field in missing_fields
        )
        occurrence_key = (
            _text(row.get(SNAPSHOT)),
            _text(row.get(pc_cols.PORTFOLIO_ID)),
            transaction_date.isoformat(),
            security_id,
            dataset_field,
            transaction_code,
            _text(row.get(pc_cols.SOURCE_DESTINATION_TYPE)),
            _text(row.get(pc_cols.SOURCE_DESTINATION_SYMBOL)),
        )
        occurrence = occurrence_counts.get(occurrence_key, 0)
        occurrence_counts[occurrence_key] = occurrence + 1
        rows.append(
            _issue_row(
                snapshot=_text(row.get(SNAPSHOT)),
                portfolio_id=_text(row.get(pc_cols.PORTFOLIO_ID)),
                as_of_date=transaction_date,
                dataset_field=dataset_field,
                security_id=security_id,
                issue_type=check_name,
                value_a=None,
                value_b=None,
                difference=None,
                tolerance="original cost amount and date supplied",
                explanation=(
                    f"Configured deliver-in {security_id} (code {transaction_code}) "
                    f"has no supplied {missing_text}. In the cited Axys report "
                    "workflow, standard reporting may fall back to trade-date "
                    "market value, so an apparently reasonable value does not "
                    "prove that original cost was supplied."
                ),
                review_identity=_deliver_in_review_identity(row, occurrence),
            )
        )
    return rows


def _deliver_in_review_identity(
    row: Mapping[str, object],
    occurrence: int,
) -> str:
    """Return copy-invariant identity for one configured deliver-in row."""
    transaction_id = _text(row.get(pc_cols.TRANSACTION_ID))
    if transaction_id:
        return transaction_id
    return ":".join(
        (
            _text(row.get(pc_cols.TRANSACTION_CODE)),
            _text(row.get(pc_cols.SOURCE_DESTINATION_TYPE)),
            _text(row.get(pc_cols.SOURCE_DESTINATION_SYMBOL)),
            f"occurrence={occurrence}",
        )
    )


def _snapshot_labels() -> tuple[tuple[SnapshotKey, str], ...]:
    """Return snapshot loader keys with reviewer-facing labels."""
    return (("a", _SNAPSHOT_A_LABEL), ("b", _SNAPSHOT_B_LABEL))


def _security_reference_requirements(
    config: Mapping[str, object],
) -> dict[str, frozenset[str]]:
    """Return reference columns required by enabled holding/transaction filters."""
    requirements: dict[str, set[str]] = {
        pc_cols.HOLDINGS: set(),
        pc_cols.TRANSACTIONS: set(),
    }
    for issue_type in DataIssueType:
        check_name = issue_type.value
        if not _check_enabled(config, check_name):
            continue
        fields = _security_reference_filter_fields(_check_config(config, check_name))
        if check_name == ISSUE_LARGE_PRICE_VARIATION:
            fields = frozenset(
                field
                for rule_config in _large_price_variation_rule_configs(config)
                for field in _security_reference_filter_fields(rule_config)
            )
            requirements[pc_cols.HOLDINGS].update(fields)
            requirements[pc_cols.TRANSACTIONS].update(fields)
            continue
        if check_name in _HOLDING_FILTER_ISSUES:
            requirements[pc_cols.HOLDINGS].update(fields)
        if check_name in _TRANSACTION_FILTER_ISSUES:
            requirements[pc_cols.TRANSACTIONS].update(fields)
    return {
        dataset_name: frozenset(columns)
        for dataset_name, columns in requirements.items()
        if columns
    }


def _security_reference_filter_fields(
    check_config: Mapping[str, object],
) -> frozenset[str]:
    """Return normalized reference fields used by one check's filters."""
    fields: set[str] = set()
    for filter_key in ("only", "exclude"):
        for field_name, _ in _filter_mapping(check_config.get(filter_key, {})):
            normalized = field_name.strip().lower()
            if normalized.startswith(_SECURITY_REFERENCE_PREFIX):
                fields.add(normalized.removeprefix(_SECURITY_REFERENCE_PREFIX))
    return frozenset(fields)


def _security_reference_maps(
    specification: AuditSpecification,
    *,
    required_columns: frozenset[str],
) -> dict[str, dict[str, dict[str, object]]]:
    """Return exact-case security-reference rows keyed by snapshot and security."""
    loader = SecurityReferenceLoader(specification)
    reference_maps: dict[str, dict[str, dict[str, object]]] = {}
    for snapshot_key, snapshot_label in _snapshot_labels():
        frame = loader.load(snapshot_key)
        if frame is None:
            raise PpaError(
                (
                    f"{specification.path}: Data Issues filters reference "
                    "security_reference fields, but files.security_reference is "
                    f"missing for snapshot {snapshot_key}."
                ),
                504,
            )
        missing_columns = sorted(required_columns - set(frame.columns))
        if missing_columns:
            raise PpaError(
                (
                    f"{specification.path}: security_reference for snapshot "
                    f"{snapshot_key} is missing filter columns: "
                    f"{', '.join(missing_columns)}."
                ),
                504,
            )
        reference_maps[snapshot_label] = {
            str(row[pc_cols.SECURITY_ID]): dict(row)
            for row in frame.iter_rows(named=True)
        }
    return reference_maps


def _enrich_with_security_reference(
    rows: Sequence[Mapping[str, object]],
    reference_maps: Mapping[str, Mapping[str, Mapping[str, object]]],
    required_columns: frozenset[str],
    *,
    dataset_name: str,
    specification: AuditSpecification,
) -> tuple[dict[str, object], ...]:
    """Attach required exact-case reference values or fail closed."""
    enriched_rows: list[dict[str, object]] = []
    for row in rows:
        snapshot = _text(row.get(SNAPSHOT))
        security_id = _text(row.get(pc_cols.SECURITY_ID))
        reference_row = reference_maps.get(snapshot, {}).get(security_id)
        if reference_row is None:
            raise PpaError(
                (
                    f"{specification.path}: {dataset_name} security_id "
                    f"{security_id!r} in {snapshot} has no exact-case "
                    "security_reference row required by a Data Issues filter."
                ),
                504,
            )
        enriched_row = dict(row)
        for column in required_columns:
            value = reference_row.get(column)
            if value is None or not str(value).strip():
                raise PpaError(
                    (
                        f"{specification.path}: security_reference security_id "
                        f"{security_id!r} in {snapshot} has no value for "
                        f"{column!r}, required by a Data Issues filter."
                    ),
                    504,
                )
            enriched_row[f"{_SECURITY_REFERENCE_PREFIX}{column}"] = value
        enriched_rows.append(enriched_row)
    return tuple(enriched_rows)


def _data_issues_config(
    specification_values: Mapping[str, object],
) -> Mapping[str, object]:
    """Return the normalized Data Issues configuration mapping."""
    raw_config = specification_values.get(_DATA_ISSUES_CONFIG_KEY, {})
    if isinstance(raw_config, Mapping):
        return raw_config
    return {}


def _config_enabled(config: Mapping[str, object]) -> bool:
    """Return whether optional Data Issues checks are enabled."""
    enabled = config.get("enabled", True)
    return not isinstance(enabled, bool) or enabled


def _check_config(config: Mapping[str, object], check_name: str) -> Mapping[str, object]:
    """Return one check's YAML configuration mapping."""
    raw_config = config.get(check_name, {})
    if isinstance(raw_config, Mapping):
        return raw_config
    return {}


def _check_enabled(config: Mapping[str, object], check_name: str) -> bool:
    """Return whether one Data Issues check is enabled."""
    check_config = _check_config(config, check_name)
    issue_type = DataIssueType(check_name)
    enabled = check_config.get(
        "enabled",
        DATA_ISSUE_REGISTRY[issue_type].default_enabled,
    )
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


def _minimum_calendar_days(
    config: Mapping[str, object],
    check_name: str,
) -> int:
    """Return a validated observed-price calendar-day threshold."""
    value = _check_config(config, check_name).get("minimum_calendar_days", 28)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return 28
    return value


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
    *,
    exact_case: bool,
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
            transaction_code_matching_key(
                row.get(pc_cols.TRANSACTION_CODE),
                exact_case=exact_case,
            ),
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
    check_name = ISSUE_HOLDINGS_PRICE_RANGE
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


def _holdings_nonpositive_price_issues(
    holdings: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
) -> list[dict[str, object]]:
    """Return configured nonzero holdings with zero or negative prices."""
    check_name = ISSUE_HOLDINGS_NONPOSITIVE_PRICE
    if not _check_enabled(config, check_name):
        return []

    row_filter = _row_filter(config, check_name)
    rows: list[dict[str, object]] = []
    for row in holdings:
        if not row_filter.allows(row):
            continue
        quantity = _number(row.get(pc_cols.QUANTITY))
        observed_price = _number(row.get(pc_cols.PRICE))
        holding_date = _date(row.get(pc_cols.HOLDING_DATE))
        security_id = _text(row.get(pc_cols.SECURITY_ID))
        if (
            quantity is None
            or quantity == 0
            or observed_price is None
            or observed_price > 0
            or holding_date is None
            or not security_id
        ):
            continue
        rows.append(
            _issue_row(
                snapshot=_text(row.get(SNAPSHOT)),
                portfolio_id=_text(row.get(pc_cols.PORTFOLIO_ID)),
                as_of_date=holding_date,
                dataset_field="holdings.price",
                security_id=security_id,
                issue_type=check_name,
                value_a=None,
                value_b=observed_price,
                difference=None,
                tolerance="price must be greater than 0",
                explanation=(
                    "A nonzero holding has a nonpositive holdings.price for "
                    f"{security_id}; review the valuation or exclude an intentional "
                    "pricing convention."
                ),
            )
        )
    return rows


def _transaction_price_range_issues(
    transactions: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
) -> list[dict[str, object]]:
    """Return same-day same-security transactions.price range issues."""
    check_name = ISSUE_TRANSACTIONS_PRICE_RANGE
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


def _holdings_stale_price_issues(
    holdings: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
) -> list[dict[str, object]]:
    """Return unchanged positive prices spanning the configured observed period."""
    check_name = ISSUE_HOLDINGS_STALE_PRICE
    if not _check_enabled(config, check_name):
        return []

    row_filter = _row_filter(config, check_name)
    grouped = _holdings_by_portfolio_security(
        row for row in holdings if row_filter.allows(row)
    )
    minimum_days = _minimum_calendar_days(config, check_name)
    issues: list[dict[str, object]] = []
    for rows in grouped.values():
        unchanged_start_date: dt.date | None = None
        unchanged_price: float | None = None
        for row in rows:
            holding_date = _date(row.get(pc_cols.HOLDING_DATE))
            price = _number(row.get(pc_cols.PRICE))
            quantity = _number(row.get(pc_cols.QUANTITY))
            if (
                holding_date is None
                or price is None
                or price <= 0
                or quantity is None
                or quantity == 0
            ):
                unchanged_start_date = None
                unchanged_price = None
                continue
            if unchanged_price != price or unchanged_start_date is None:
                unchanged_start_date = holding_date
                unchanged_price = price
                continue
            elapsed_days = (holding_date - unchanged_start_date).days
            if elapsed_days < minimum_days:
                continue
            security_id = _text(row.get(pc_cols.SECURITY_ID))
            issues.append(
                _issue_row(
                    snapshot=_text(row.get(SNAPSHOT)),
                    portfolio_id=_text(row.get(pc_cols.PORTFOLIO_ID)),
                    as_of_date=holding_date,
                    dataset_field="holdings.price",
                    security_id=security_id,
                    issue_type=check_name,
                    value_a=unchanged_price,
                    value_b=price,
                    difference=0.0,
                    tolerance=(
                        f"unchanged for at least {minimum_days} calendar days"
                    ),
                    explanation=(
                        f"The same holdings.price {price:g} was observed for "
                        f"{security_id} on {unchanged_start_date.isoformat()} and "
                        f"{holding_date.isoformat()}, {elapsed_days} calendar days "
                        "apart. PPAR did not observe every intervening day; review "
                        "whether the later price is stale or intentionally unchanged."
                    ),
                )
            )
    return issues


def _large_price_variation_issues(
    periods: Sequence[Mapping[str, object]],
    holdings: Sequence[Mapping[str, object]],
    transactions: Sequence[Mapping[str, object]],
    splits: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
) -> list[dict[str, object]]:
    """Return maximum split-normalized price variation per named rule and period."""
    check_name = ISSUE_LARGE_PRICE_VARIATION
    if not _check_enabled(config, check_name):
        return []

    rules = _large_price_variation_rules(config)
    if not rules:
        return []
    period_scopes = _rows_by_portfolio_scope(periods)
    holding_scopes = _indexed_rows_by_portfolio_security(holdings)
    transaction_scopes = _indexed_rows_by_portfolio_security(transactions)
    split_index = _split_factor_index(splits)

    issues: list[dict[str, object]] = []
    for scope in sorted(period_scopes):
        snapshot, portfolio_id = scope
        holding_securities = holding_scopes.get(scope, {})
        transaction_securities = transaction_scopes.get(scope, {})
        security_ids = sorted(
            set(holding_securities).union(transaction_securities)
        )
        prior_thru_date: dt.date | None = None
        for period in period_scopes[scope]:
            from_date = _date(period.get(pc_cols.FROM_DATE))
            thru_date = _date(period.get(pc_cols.THRU_DATE))
            if from_date is None or thru_date is None or thru_date < from_date:
                prior_thru_date = thru_date
                continue
            beginning_holding_date = (
                prior_thru_date
                if prior_thru_date is not None
                and from_date == prior_thru_date + dt.timedelta(days=1)
                else None
            )
            inclusive_period_days = (thru_date - from_date).days + 1
            prior_thru_date = thru_date

            for rule in rules:
                if inclusive_period_days < rule.minimum_calendar_days:
                    continue
                for security_id in security_ids:
                    observations = _period_price_observations(
                        holding_securities.get(security_id, ()),
                        transaction_securities.get(security_id, ()),
                        split_index.get((snapshot, security_id), ()),
                        rule,
                        beginning_holding_date=beginning_holding_date,
                        from_date=from_date,
                        thru_date=thru_date,
                    )
                    if len(observations) < 2:
                        continue
                    currencies = {
                        observation.currency
                        for observation in observations
                        if observation.currency
                    }
                    if len(currencies) > 1:
                        # Raw prices in different currencies are not comparable.
                        continue
                    minimum = min(
                        observations,
                        key=_minimum_price_observation_key,
                    )
                    maximum = min(
                        observations,
                        key=_maximum_price_observation_key,
                    )
                    price_difference = maximum.adjusted_price - minimum.adjusted_price
                    variation = price_difference / minimum.adjusted_price
                    if variation <= rule.minimum_tolerance:
                        continue
                    dataset_field = " + ".join(
                        sorted(
                            {
                                (
                                    "transactions.price"
                                    if observation.source_rank
                                    else "holdings.price"
                                )
                                for observation in observations
                            }
                        )
                    )
                    issues.append(
                        _issue_row(
                            snapshot=snapshot,
                            portfolio_id=portfolio_id,
                            as_of_date=thru_date,
                            dataset_field=dataset_field,
                            security_id=security_id,
                            issue_type=check_name,
                            value_a=minimum.adjusted_price,
                            value_b=maximum.adjusted_price,
                            difference=price_difference,
                            tolerance=(
                                f"rule {rule.rule_id}: variation > "
                                f"{_decimal_percent_text(rule.minimum_tolerance)}"
                            ),
                            explanation=_large_price_variation_explanation(
                                rule,
                                minimum,
                                maximum,
                                observations,
                                from_date=from_date,
                                thru_date=thru_date,
                                inclusive_period_days=inclusive_period_days,
                                variation=variation,
                            ),
                            review_identity=f"rule:{rule.rule_id}",
                        )
                    )
    return issues


def _large_price_variation_rules(
    config: Mapping[str, object],
) -> tuple[_LargePriceVariationRule, ...]:
    """Compile enabled named rules in canonical rule-ID order."""
    rules: list[_LargePriceVariationRule] = []
    for raw_rule in _large_price_variation_rule_configs(config):
        rule_id = _text(raw_rule.get("rule_id"))
        minimum_days = raw_rule.get("minimum_calendar_days")
        minimum_tolerance = raw_rule.get("minimum_tolerance")
        if (
            not rule_id
            or isinstance(minimum_days, bool)
            or not isinstance(minimum_days, int)
            or minimum_days <= 0
            or isinstance(minimum_tolerance, bool)
            or not isinstance(minimum_tolerance, (int, float))
        ):
            continue
        rules.append(
            _LargePriceVariationRule(
                rule_id=rule_id,
                minimum_calendar_days=minimum_days,
                minimum_tolerance=float(minimum_tolerance),
                holdings_filter=_large_price_variation_filter(
                    raw_rule,
                    pc_cols.HOLDINGS,
                ),
                transactions_filter=_large_price_variation_filter(
                    raw_rule,
                    pc_cols.TRANSACTIONS,
                ),
            )
        )
    return tuple(rules)


def _large_price_variation_rule_configs(
    config: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    """Return enabled named-rule mappings in canonical rule-ID order."""
    check_config = _check_config(config, ISSUE_LARGE_PRICE_VARIATION)
    raw_rules = check_config.get("rules", [])
    if not isinstance(raw_rules, list):
        return ()
    rules = [
        rule
        for rule in raw_rules
        if isinstance(rule, Mapping) and rule.get("enabled", True) is not False
    ]
    return tuple(sorted(rules, key=lambda rule: _text(rule.get("rule_id"))))


def _large_price_variation_filter(
    rule_config: Mapping[str, object],
    dataset_name: str,
) -> _RowFilter:
    """Compile filters relevant to one side of the observation union."""
    return _RowFilter(
        only=_compiled_filters(
            _source_filter_mapping(rule_config.get("only", {}), dataset_name)
        ),
        exclude=_compiled_filters(
            _source_filter_mapping(rule_config.get("exclude", {}), dataset_name)
        ),
    )


def _source_filter_mapping(
    filter_config: object,
    dataset_name: str,
) -> dict[str, object]:
    """Return generic and source-specific filters applicable to one dataset."""
    return {
        field_name: raw_values
        for field_name, raw_values in _filter_mapping(filter_config)
        if _filter_applies_to_source(field_name, dataset_name)
    }


def _filter_applies_to_source(field_name: str, dataset_name: str) -> bool:
    """Return whether a rule filter qualifies this observation source."""
    normalized = field_name.strip().lower()
    namespace, separator, normalized_name = normalized.rpartition(".")
    if separator and namespace in {pc_cols.HOLDINGS, pc_cols.TRANSACTIONS}:
        return namespace == dataset_name
    if not separator and normalized_name == pc_cols.TRANSACTION_CODE:
        return dataset_name == pc_cols.TRANSACTIONS
    return True


def _rows_by_portfolio_scope(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str], tuple[Mapping[str, object], ...]]:
    """Group rows by snapshot and portfolio in canonical period order."""
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (
            _text(row.get(SNAPSHOT)),
            _text(row.get(pc_cols.PORTFOLIO_ID)),
        )
        grouped.setdefault(key, []).append(row)
    return {
        key: tuple(
            sorted(
                scope_rows,
                key=lambda row: (
                    _date(row.get(pc_cols.FROM_DATE)) or dt.date.max,
                    _date(row.get(pc_cols.THRU_DATE)) or dt.date.max,
                ),
            )
        )
        for key, scope_rows in grouped.items()
    }


def _indexed_rows_by_portfolio_security(
    rows: Sequence[Mapping[str, object]],
) -> dict[
    tuple[str, str],
    dict[str, tuple[tuple[int, Mapping[str, object]], ...]],
]:
    """Index source rows by snapshot, portfolio, security, and source order."""
    grouped: dict[
        tuple[str, str],
        dict[str, list[tuple[int, Mapping[str, object]]]],
    ] = {}
    for source_order, row in enumerate(rows):
        scope = (
            _text(row.get(SNAPSHOT)),
            _text(row.get(pc_cols.PORTFOLIO_ID)),
        )
        security_id = _text(row.get(pc_cols.SECURITY_ID))
        if not security_id:
            continue
        grouped.setdefault(scope, {}).setdefault(security_id, []).append(
            (source_order, row)
        )
    return {
        scope: {
            security_id: tuple(security_rows)
            for security_id, security_rows in securities.items()
        }
        for scope, securities in grouped.items()
    }


def _split_factor_index(
    splits: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str], tuple[tuple[dt.date, float], ...]]:
    """Return one positive split factor per snapshot, security, and date."""
    grouped: dict[tuple[str, str], dict[dt.date, float]] = {}
    for row in splits:
        snapshot = _text(row.get(SNAPSHOT))
        security_id = _text(row.get(pc_cols.SECURITY_ID))
        split_date = _date(row.get(pc_cols.SPLIT_DATE))
        split_factor = _number(row.get(pc_cols.SPLIT_FACTOR))
        if not snapshot or not security_id or split_date is None:
            continue
        if split_factor is None or split_factor <= 0:
            raise PpaError(
                "large_price_variation requires positive finite split factors.",
                504,
                context={
                    "snapshot": snapshot,
                    "security_id": security_id,
                    "split_date": split_date.isoformat(),
                },
            )
        factors = grouped.setdefault((snapshot, security_id), {})
        prior_factor = factors.get(split_date)
        if prior_factor is not None and prior_factor != split_factor:
            raise PpaError(
                "large_price_variation found conflicting split factors.",
                504,
                context={
                    "snapshot": snapshot,
                    "security_id": security_id,
                    "split_date": split_date.isoformat(),
                },
            )
        factors[split_date] = split_factor
    return {
        key: tuple(sorted(factors.items()))
        for key, factors in grouped.items()
    }


def _period_price_observations(
    holding_rows: Sequence[tuple[int, Mapping[str, object]]],
    transaction_rows: Sequence[tuple[int, Mapping[str, object]]],
    split_factors: Sequence[tuple[dt.date, float]],
    rule: _LargePriceVariationRule,
    *,
    beginning_holding_date: dt.date | None,
    from_date: dt.date,
    thru_date: dt.date,
) -> tuple[_PriceObservation, ...]:
    """Return comparable boundary-holding and inclusive-transaction prices."""
    observations: list[_PriceObservation] = []
    boundary_dates = {thru_date}
    if beginning_holding_date is not None:
        boundary_dates.add(beginning_holding_date)
    for source_order, row in holding_rows:
        holding_date = _date(row.get(pc_cols.HOLDING_DATE))
        if (
            holding_date is None
            or holding_date not in boundary_dates
            or not rule.holdings_filter.allows(row)
        ):
            continue
        source = (
            "beginning-period holdings.price"
            if holding_date == beginning_holding_date
            else "ending-period holdings.price"
        )
        observation = _price_observation(
            row,
            split_factors,
            observation_date=holding_date,
            source=source,
            source_rank=0,
            source_order=source_order,
            thru_date=thru_date,
        )
        if observation is not None:
            observations.append(observation)
    for source_order, row in transaction_rows:
        trade_date = _date(row.get(pc_cols.TRANSACTION_DATE))
        if (
            trade_date is None
            or trade_date < from_date
            or trade_date > thru_date
            or not rule.transactions_filter.allows(row)
        ):
            continue
        observation = _price_observation(
            row,
            split_factors,
            observation_date=trade_date,
            source="transactions.price",
            source_rank=1,
            source_order=source_order,
            thru_date=thru_date,
        )
        if observation is not None:
            observations.append(observation)
    return tuple(observations)


def _price_observation(
    row: Mapping[str, object],
    split_factors: Sequence[tuple[dt.date, float]],
    *,
    observation_date: dt.date,
    source: str,
    source_rank: int,
    source_order: int,
    thru_date: dt.date,
) -> _PriceObservation | None:
    """Return one positive price expressed on the period-ending share basis."""
    raw_price = _number(row.get(pc_cols.PRICE))
    if raw_price is None or raw_price <= 0:
        return None
    cumulative_factor = 1.0
    for split_date, split_factor in split_factors:
        if observation_date < split_date <= thru_date:
            cumulative_factor *= split_factor
    return _PriceObservation(
        observation_date=observation_date,
        source=source,
        source_rank=source_rank,
        source_order=source_order,
        raw_price=raw_price,
        adjusted_price=raw_price / cumulative_factor,
        cumulative_split_factor=cumulative_factor,
        currency=_text(row.get(pc_cols.CURRENCY)).upper(),
    )


def _minimum_price_observation_key(
    observation: _PriceObservation,
) -> tuple[float, dt.date, int, int]:
    """Return deterministic minimum-price and evidence tie-break keys."""
    return (
        observation.adjusted_price,
        observation.observation_date,
        observation.source_rank,
        observation.source_order,
    )


def _maximum_price_observation_key(
    observation: _PriceObservation,
) -> tuple[float, dt.date, int, int]:
    """Return deterministic maximum-price and evidence tie-break keys."""
    return (
        -observation.adjusted_price,
        observation.observation_date,
        observation.source_rank,
        observation.source_order,
    )


def _large_price_variation_explanation(
    rule: _LargePriceVariationRule,
    minimum: _PriceObservation,
    maximum: _PriceObservation,
    observations: Sequence[_PriceObservation],
    *,
    from_date: dt.date,
    thru_date: dt.date,
    inclusive_period_days: int,
    variation: float,
) -> str:
    """Return deterministic evidence and split-normalization guidance."""
    applied_factors = sorted(
        {
            observation.cumulative_split_factor
            for observation in observations
            if observation.cumulative_split_factor != 1.0
        }
    )
    if applied_factors:
        split_text = (
            "Split normalization applied cumulative factor(s) "
            + ", ".join(f"{factor:g}" for factor in applied_factors)
            + "."
        )
    else:
        split_text = (
            "No supplied split factor applied to these observations; review "
            "whether the variation is legitimate or split evidence is missing."
        )
    return (
        f"Rule {rule.rule_id} found a {_decimal_percent_text(variation)} maximum "
        f"price variation in the inclusive {from_date.isoformat()} through "
        f"{thru_date.isoformat()} performance period ({inclusive_period_days} "
        f"calendar days). Minimum {_price_observation_text(minimum)}; maximum "
        f"{_price_observation_text(maximum)}. {split_text}"
    )


def _price_observation_text(observation: _PriceObservation) -> str:
    """Return concise source, date, raw-price, and adjustment evidence."""
    evidence = (
        f"{observation.adjusted_price:g} from {observation.source} on "
        f"{observation.observation_date.isoformat()}"
    )
    if observation.cumulative_split_factor == 1.0:
        return evidence
    return (
        f"{evidence} (raw {observation.raw_price:g}, cumulative split factor "
        f"{observation.cumulative_split_factor:g})"
    )


def _decimal_percent_text(value: float) -> str:
    """Return a decimal tolerance or result as a compact percentage."""
    return f"{value * 100:.2f}".rstrip("0").rstrip(".") + "%"


def _transactions_nonpositive_price_issues(
    transactions: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
) -> list[dict[str, object]]:
    """Return configured nonzero transactions with zero or negative prices."""
    check_name = ISSUE_TRANSACTIONS_NONPOSITIVE_PRICE
    if not _check_enabled(config, check_name):
        return []

    row_filter = _row_filter(config, check_name)
    rows: list[dict[str, object]] = []
    for row in transactions:
        if not row_filter.allows(row):
            continue
        quantity = _number(row.get(pc_cols.QUANTITY))
        observed_price = _number(row.get(pc_cols.PRICE))
        transaction_date = _date(row.get(pc_cols.TRANSACTION_DATE))
        security_id = _text(row.get(pc_cols.SECURITY_ID))
        transaction_code = _text(row.get(pc_cols.TRANSACTION_CODE))
        if (
            quantity is None
            or quantity == 0
            or observed_price is None
            or observed_price > 0
            or transaction_date is None
            or not security_id
        ):
            continue
        rows.append(
            _issue_row(
                snapshot=_text(row.get(SNAPSHOT)),
                portfolio_id=_text(row.get(pc_cols.PORTFOLIO_ID)),
                as_of_date=transaction_date,
                dataset_field="transactions.price",
                security_id=security_id,
                issue_type=check_name,
                value_a=None,
                value_b=observed_price,
                difference=None,
                tolerance="price must be greater than 0",
                explanation=(
                    "A nonzero-quantity transaction in the configured "
                    "price-bearing population has a nonpositive "
                    f"transactions.price for {security_id} (code "
                    f"{transaction_code}); review the trade price or population."
                ),
            )
        )
    return rows


def _transaction_security_type_mismatch_issues(
    transactions: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
) -> list[dict[str, object]]:
    """Return exact-case transaction-versus-reference type mismatches."""
    check_name = ISSUE_TRANSACTION_SECURITY_TYPE_MISMATCH
    if not _check_enabled(config, check_name):
        return []

    row_filter = _row_filter(config, check_name)
    reference_field = f"{_SECURITY_REFERENCE_PREFIX}{pc_cols.SECURITY_TYPE}"
    rows: list[dict[str, object]] = []
    for row in transactions:
        if not row_filter.allows(row):
            continue
        transaction_date = _date(row.get(pc_cols.TRANSACTION_DATE))
        security_id = _text(row.get(pc_cols.SECURITY_ID))
        transaction_type = _text(row.get(pc_cols.SECURITY_TYPE))
        reference_type = _text(row.get(reference_field))
        if (
            transaction_date is None
            or not security_id
            or transaction_type == reference_type
        ):
            continue
        if not transaction_type:
            comparison = (
                "has a blank transactions.security_type while "
                f"security_reference.security_type is {reference_type!r}"
            )
        elif transaction_type.casefold() == reference_type.casefold():
            comparison = (
                f"has transactions.security_type {transaction_type!r}, which differs "
                "only by case from security_reference.security_type "
                f"{reference_type!r}"
            )
        else:
            comparison = (
                f"has transactions.security_type {transaction_type!r} and "
                f"security_reference.security_type {reference_type!r}"
            )
        rows.append(
            _issue_row(
                snapshot=_text(row.get(SNAPSHOT)),
                portfolio_id=_text(row.get(pc_cols.PORTFOLIO_ID)),
                as_of_date=transaction_date,
                dataset_field=(
                    "transactions.security_type -> "
                    "security_reference.security_type"
                ),
                security_id=security_id,
                issue_type=check_name,
                value_a=None,
                value_b=None,
                difference=None,
                tolerance="exact-case equality",
                explanation=(
                    f"Transaction {security_id} {comparison}; review the source "
                    "classification mapping. PPAR does not choose which value is "
                    "correct."
                ),
            )
        )
    return rows


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
    *,
    exact_case: bool,
) -> list[dict[str, object]]:
    """Return same-day same-security transaction-rate issues."""
    enabled_checks = (
        _check_enabled(config, ISSUE_DIVIDEND_RATE),
        _check_enabled(config, ISSUE_PA_SA_RATE),
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
                check_name=ISSUE_DIVIDEND_RATE,
                transaction_codes=_DIVIDEND_CODES,
                issue_type=ISSUE_DIVIDEND_RATE,
                dataset_field="transactions.amount",
                exact_case=exact_case,
            )
        )
    if enabled_checks[1]:
        rows.extend(
            _transaction_rate_issues(
                transactions,
                holdings_by_key,
                config,
                check_name=ISSUE_PA_SA_RATE,
                transaction_codes=_ACCRUAL_CODES,
                issue_type=ISSUE_PA_SA_RATE,
                dataset_field="transactions.amount",
                exact_case=exact_case,
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
    exact_case: bool,
) -> list[dict[str, object]]:
    """Return transaction-rate issues for a configured transaction-code family."""
    if not _check_enabled(config, check_name):
        return []

    tolerance = _tolerance(config, check_name)
    row_filter = _row_filter(config, check_name)
    groups: dict[tuple[str, dt.date, str, str], list[Mapping[str, object]]] = {}
    for row in transactions:
        code = transaction_code_matching_key(
            row.get(pc_cols.TRANSACTION_CODE),
            exact_case=exact_case,
        )
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
    *,
    exact_case: bool,
) -> list[dict[str, object]]:
    """Return portfolios that appear to be missing same-date dividends."""
    check_name = ISSUE_MISSING_DIVIDEND
    if not _check_enabled(config, check_name):
        return []

    row_filter = _row_filter(config, check_name)
    dividend_rows = [
        row
        for row in transactions
        if transaction_code_matching_key(
            row.get(pc_cols.TRANSACTION_CODE),
            exact_case=exact_case,
        )
        in _DIVIDEND_CODES
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
            exact_case=exact_case,
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
                _reference_context_row(
                    representative_row,
                    {
                        SNAPSHOT: snapshot,
                        pc_cols.PORTFOLIO_ID: portfolio_id,
                        pc_cols.SECURITY_ID: security_id,
                    },
                )
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
    exact_case: bool,
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
                exact_case=exact_case,
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
    exact_case: bool,
) -> bool:
    """Return whether a portfolio qualifies for conservative dividend review."""
    has_beginning_position = _positive(beginning_quantity)
    has_pre_dividend_buy = False
    for row in transaction_rows:
        transaction_date = _date(row.get(pc_cols.TRANSACTION_DATE))
        if transaction_date is None or not start_date < transaction_date < dividend_date:
            continue
        transaction_code = transaction_code_matching_key(
            row.get(pc_cols.TRANSACTION_CODE),
            exact_case=exact_case,
        )
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
    check_name = ISSUE_HOLDINGS_ACCRUED_RATE
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
    review_identity: str = "",
) -> dict[str, object]:
    """Return one Data Issues row."""
    issue_category = DATA_ISSUE_REGISTRY[DataIssueType(issue_type)].category.value
    return {
        SNAPSHOT: snapshot,
        pc_cols.PORTFOLIO_ID: portfolio_id,
        AS_OF_DATE: as_of_date,
        DATASET_FIELD: dataset_field,
        pc_cols.SECURITY_ID: security_id,
        ISSUE_TYPE: issue_type,
        CATEGORY: issue_category,
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
                *([review_identity] if review_identity else []),
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


def _row_filter(
    config: Mapping[str, object],
    check_name: str,
    *,
    exact_native_case: bool = False,
) -> _RowFilter:
    """Compile one check's row filters for reuse across all source rows."""
    check_config = _check_config(config, check_name)
    return _RowFilter(
        only=_compiled_filters(
            check_config.get("only", {}),
            exact_native_case=exact_native_case,
        ),
        exclude=_compiled_filters(
            check_config.get("exclude", {}),
            exact_native_case=exact_native_case,
        ),
    )


def _compiled_filters(
    filter_config: object,
    *,
    exact_native_case: bool = False,
) -> tuple[_CompiledFilter, ...]:
    """Normalize a YAML row-filter mapping once for repeated matching."""
    compiled: list[_CompiledFilter] = []
    for field_name, raw_values in _filter_mapping(filter_config):
        exact_case = exact_native_case or field_name.strip().lower().startswith(
            _SECURITY_REFERENCE_PREFIX
        )
        values = _text_filter_values(raw_values, exact_case=exact_case)
        if values:
            compiled.append(
                _CompiledFilter(
                    column=_filter_column_name(field_name),
                    values=values,
                    exact_case=exact_case,
                )
            )
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
    exact_case = field_name.strip().lower().startswith(_SECURITY_REFERENCE_PREFIX)
    values = _text_filter_values(raw_values, exact_case=exact_case)
    if not values:
        return False
    column_name = _filter_column_name(field_name)
    value = _text(row.get(column_name))
    candidate = value if exact_case else value.lower()
    return candidate in values


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
    if normalized_name.startswith(_SECURITY_REFERENCE_PREFIX):
        return normalized_name
    if "." in normalized_name:
        normalized_name = normalized_name.rsplit(".", maxsplit=1)[-1]
    return _FILTER_FIELD_ALIASES.get(normalized_name, normalized_name)


def _text_filter_values(
    value: object,
    *,
    exact_case: bool = False,
) -> frozenset[str]:
    """Return normalized exact-match values from a YAML scalar or sequence."""
    if isinstance(value, (str, int, float, bool)):
        text = str(value).strip()
        text = text if exact_case else text.lower()
        return frozenset({text}) if text else frozenset()
    if isinstance(value, Iterable):
        return frozenset(
            (
                str(item).strip()
                if exact_case
                else str(item).strip().lower()
            )
            for item in value
            if str(item).strip()
        )
    return frozenset()


def _reference_context_row(
    source_row: Mapping[str, object],
    target_row: Mapping[str, object],
) -> dict[str, object]:
    """Copy reference-enrichment fields into a derived filter row."""
    enriched = dict(target_row)
    enriched.update(
        {
            key: value
            for key, value in source_row.items()
            if key.startswith(_SECURITY_REFERENCE_PREFIX)
        }
    )
    return enriched


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
