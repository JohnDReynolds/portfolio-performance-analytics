"""
Loads Axys performance data, optional classification and mapping sources, and performs
reconciliation logic to ensure that sumof(weight * return) across secperf rows matches the portperf
return for each period.

Axys only stores performance in portperf and secperf.  It does not store sector-level performance.
This class will always return security-level performance with optional mappings up to sector that
Analytics will then roll up for sector-level attribution.

"""

from __future__ import annotations

# Python imports
from dataclasses import dataclass
import datetime as dt
import math
import os
from typing import Any, Final, Iterable, Literal

# Third-party imports
import polars as pl
import yaml

# Project imports
import ppar.columns as cols
from ppar.errors import PpaError
import ppar.utilities as util

_FATAL_PERIOD_TOLERANCE = 0.0001  # 1 basis point, which is what the UI always displays.
_PERIOD_TOLERANCE: Final[float] = 0.0000001  # 1/1000 of a basis point

_MATCH_TOLERANCE: Final[float] = 1e-12
_NEAR_ZERO_WEIGHT: Final[float] = 1e-18
_RETURN_EPSILON: Final[float] = 1e-12

_ANALYTICS_REQUIRED_COLUMNS: Final[set[str]] = {
    cols.BEGINNING_DATE,
    cols.ENDING_DATE,
    cols.IDENTIFIER,
    cols.RETURN,
    cols.WEIGHT,
}
_CLASSIFICATION_MAPPING_FIELDS_ALLOWED: Final[set[str]] = {
    "file_path",
    "identifier_column",
    "name_column",
    "is_security_master",
    "filter_column",
    "filter_value",
}
_CLASSIFICATION_MAPPING_FIELDS_REQUIRED: Final[set[str]] = {
    "file_path",
    "identifier_column",
    "name_column",
}
_CLASSIFICATION_MAPPING_COLUMN_NAMES: Final[set[str]] = {
    "identifier_column",
    "name_column",
    "filter_column",
}
_PERIOD_UNIQUE_KEY_COLUMNS: Final[tuple[str, ...]] = (
    cols.PORTFOLIO_CODE,
    cols.BEGINNING_DATE,
    cols.ENDING_DATE,
)
_PORTPERF_REQUIRED_COLUMNS: Final[set[str]] = {
    cols.BEGINNING_DATE,
    cols.ENDING_DATE,
    cols.PORTFOLIO_CODE,
    cols.PORTFOLIO_NAME,
    cols.PORTFOLIO_RETURN,
}
_SECPERF_REQUIRED_COLUMNS: Final[set[str]] = {
    cols.BEGINNING_DATE,
    cols.CONTRIBUTION,
    cols.ENDING_DATE,
    cols.IDENTIFIER,
    cols.PORTFOLIO_CODE,
    cols.RETURN,
    cols.WEIGHT,
}

UnreconciledPeriodType = tuple[tuple[str, dt.date, dt.date], float, float]


@dataclass(frozen=True)
class Portfolio:
    """Contains performance data for a portfolio."""

    portfolio_code: str
    portfolio_name: str
    secperf: pl.DataFrame


class AxysData:
    """Load, validate, and derive Axys-style portperf/secperf data.

    1. Loads portperf and secperf using lazy CSV scans.
    2. Applies optional filters on from_date and thru_date.
    3. Takes the intersection of portperf and secperf periods.
    4. Derives reconciled secperf weights for all periods.
    5. Projects only the required columns.
    6. Optionally creates classification and mapping data sources.

    Attributes:
        classification_data_sources: Dict of classification data sources keyed on
            classification_names.
        directory: The directory of the json specifications file.
        from_date: Optional lower date bound.
        mapping_data_sources: Dict of mapping data sources keyed on mapping_names.
        portfolios: Dict of Portfolio objects keyed on portfolio_code.
        portperf_path: Path to the portperf CSV.
        secperf_path: Path to the secperf CSV.
        specifications: Contents of the json specifications file.
        specifications_path: Path to the json specifications file.
        thru_date: Optional upper date bound.

    Note:
        Secperf is treated as row-grain input. Reconciliation operates on each input row exactly
        as provided and does not require cols.IDENTIFIER to be unique within a period. Multiple
        rows with the same identifier in the same period are valid inputs for this class and are
        processed independently.

        This class does not enforce or validate identifier-level uniqueness. Any validation of
        duplicate identifiers is handled by downstream components (for example, PpaError 112),
        where identifier-level rules are applied if required. This separation is intentional to
        keep this class focused on row-level reconciliation logic and to avoid duplicating
        validation responsibilities.
    """

    def __init__(
        self,
        specifications_path: str,
        portperf_path: str | None = None,
        secperf_path: str | None = None,
        from_date: dt.date | None = None,
        thru_date: dt.date | None = None,
        portfolio_codes: Iterable[str] | str | None = None,
        classification_names: Iterable[str] | str | None = None,
        mapping_names: Iterable[str] | str | None = None,
    ) -> None:
        """Initialize AxysData and load/validate all required data.

        Args:
            specifications_path: Contains column name mappings and processing rules.
            portperf_path: Path to portperf CSV.
            secperf_path: Path to secperf CSV.
            from_date: Lower date bound. Keeps rows where from_date <= cols.BEGINNING_DATE.
            thru_date: Upper date bound. Keeps rows where cols.ENDING_DATE <= thru_date.
            portfolio_codes: Optional portfolio codes used to populate self.portfolios.
                Accepts a string, any iterable of strings, or None. The value is normalized
                internally to a tuple of strings. If None or empty after normalization, all
                unique portfolio codes found in portperf are used.
            classification_names: Optional classification names. Accepts a string, any
                iterable of strings, or None. The value is normalized internally to a tuple
                of strings.
            mapping_names: Optional mapping names. Accepts a string, any iterable of
                strings, or None. The value is normalized internally to a tuple of strings.

        Raises:
            PpaError: If any validation fails or if file/schema validation fails.
        """
        # Set the basic class members.
        # self.directory = os.path.dirname(specifications_path)
        self.from_date: dt.date | None = from_date
        self.portfolios: dict[str, Portfolio] = {}
        self.specifications_path = specifications_path
        self.thru_date: dt.date | None = thru_date

        # Lod the specifications file.
        # self.specifications: dict[str, Any] = util.read_json_file(self.specifications_path)
        with open(self.specifications_path, "r", encoding="utf-8") as f:
            try:
                loaded_yaml: Any = yaml.safe_load(f)
            except Exception as e:
                raise PpaError(self._error_message(f"Invalid YAML: {e}"), 504) from e
        if not isinstance(loaded_yaml, dict):
            raise PpaError(self._error_message("YAML must be a dictionary"), 504)
        self.specifications: dict[str, Any] = loaded_yaml
        del loaded_yaml

        # Determine self.portperf_path and self.secperf_path from specifications if not provided
        # as arguments.  Tried doing this in a loop, but it got too complicated.
        if not portperf_path:
            portperf_path = self.specifications.get("portperf_path")
            if not portperf_path:
                raise PpaError(
                    self._error_message(
                        "portperf_path not in specifications file and not provided as an argument."
                    ),
                    504,
                )
        if not secperf_path:
            secperf_path = self.specifications.get("secperf_path")
            if not secperf_path:
                raise PpaError(
                    self._error_message(
                        "secperf_path not in specifications file and not provided as an argument."
                    ),
                    504,
                )
        self.portperf_path: str = portperf_path
        self.secperf_path: str = secperf_path

        # Get the optional prefix_portfolio_code from the specifications. This can be used to
        # create a more descriptive portfolio_name.
        prefix_portfolio_code = self.specifications.get("settings", {}).get(
            "prefix_portfolio_code"
        )

        # If not portfolio_codes, then get a unique list of all portfolios in the entire file.
        portfolio_codes = util.to_tuple_or_none(portfolio_codes)
        if not portfolio_codes:
            portperf: pl.DataFrame = self._get_performance(self.portperf_path, "portperf_columns")
            portfolio_codes = portperf[cols.PORTFOLIO_CODE].unique().sort().to_list()
            del portperf

        for portfolio_code in portfolio_codes:
            self.portfolio_code: str = portfolio_code

            # Get portperf data.
            self.portperf: pl.DataFrame = self._get_performance(
                self.portperf_path, "portperf_columns"
            )

            # It is very well possible that this portfolio does not have any rows after filtering
            # by dates, and that is OK.  Just skip it silently.
            if self.portperf.is_empty():
                continue

            # Set self.portfolio_name based on prefix_portfolio_code
            self.portfolio_name = self.portperf[cols.PORTFOLIO_NAME][0]
            if prefix_portfolio_code:
                self.portfolio_name = (
                    f"{self.portperf[cols.PORTFOLIO_CODE][0]}{prefix_portfolio_code}"
                    f"{self.portfolio_name}"
                )

            # Get secperf data.
            self.secperf: pl.DataFrame = self._get_performance(
                self.secperf_path, "secperf_columns"
            )

            # Filter portperf and secperf to common periods.  If there are discontinuous periods,
            # then it will later get a 106 error downstream in Analytics.
            self.portperf, self.secperf = self._filter_to_common_periods()

            # Derive the secperf weights.
            unreconciled_periods: set[UnreconciledPeriodType] = (
                self._derive_secperf_for_all_periods()
            )

            # In theory, unreconciled periods should be rare and have minimal return differences.
            # We are intentionally lenient here.  It is very well possible that there may be
            # multiple plus and minus differences that net out to 0, but that is OK because each
            # single period is also checked for _FATAL_PERIOD_TOLERANCE.
            difference = abs(
                sum(t for _, t, _ in unreconciled_periods)
                - sum(a for _, _, a in unreconciled_periods)
            )
            if _FATAL_PERIOD_TOLERANCE < difference:
                raise PpaError(
                    self._error_message(
                        f"Returns difference across unreconciled periods is {difference}"
                    ),
                    503,
                )

            # Only include the columns you need for Analytics.
            self.secperf = self.secperf.select(_ANALYTICS_REQUIRED_COLUMNS)

            self.portfolios[self.portfolio_code] = Portfolio(
                self.portfolio_code, self.portfolio_name, self.secperf
            )

        # You do not need these anymore since they are only used during the constructor.  Deleting
        # to avoid confusion.
        for attr in ("portfolio_code", "portfolio_name", "portperf", "secperf"):
            if hasattr(self, attr):
                delattr(self, attr)

        # Get classification and mapping data_sources.
        if self.portfolios:
            # If not classification_names, then get a set of all classifications in specifications.
            classification_names = util.to_tuple_or_none(classification_names)
            if not classification_names:
                classification_names = self.specifications["classifications"].keys()

            # If not mapping_names, then get a set of all mappings in specifications.
            mapping_names = util.to_tuple_or_none(mapping_names)
            if not mapping_names:
                mapping_names = self.specifications["mappings"].keys()

            # Get all security IDs across all portfolios.
            unique_security_ids = (
                pl.concat([p.secperf[cols.IDENTIFIER] for p in self.portfolios.values()])
                .unique()
                .to_list()
            )

            # Create the classification and mapping data sources.
            for ds_type in ("classification", "mapping"):
                data_sources: dict[str, pl.DataFrame] = {}
                ds_names = classification_names if ds_type == "classification" else mapping_names
                if ds_names:
                    for ds_name in ds_names:
                        data_sources[ds_name] = self._get_classification_or_mapping_data_source(
                            ds_type, ds_name, unique_security_ids
                        )
                if ds_type == "classification":
                    self.classification_data_sources = data_sources
                else:
                    self.mapping_data_sources = data_sources
        else:
            self.classification_data_sources = {}
            self.mapping_data_sources = {}

        # Mandate that there is a classification with is_security_master == True.
        have_security_master = False
        for classification_name in self.classification_data_sources:
            if self.specifications["classifications"][classification_name].get(
                "is_security_master", False
            ):
                have_security_master = True
                break
        if not have_security_master:
            raise PpaError(
                self._error_message("Must have a classification with is_security_master == true"),
                504,
            )

    def _derive_reconciled_weights(
        self,
        secperf_df: pl.DataFrame,
        portfolio_return: float,
    ) -> tuple[list[float], float]:
        """Derive single-period security weights with fallbacks.

        The cols.WEIGHT values that Axys sends are often not usable as-is. Their values may be null,
        non-finite, negative, or not aligned with the contribution and return. In Axys, it is
        sometimes a beginning weight or beginning market value, which is not what you want for
        contribution.  This method derives reconciled weights that are aligned with the contribution
        and return, are nonnegative, and sum to 1.0.  The original cols.WEIGHT is used as a starting
        fallback when it is nonnegative and numerically finite, even if it is not perfectly aligned
        with contribution and return.

        This implementation is intentionally lean:
            - It returns only the normalized adjusted weights and the achieved return.
            - The caller writes cols.WEIGHT into self.secperf.
            - No debug/helper columns are materialized.

        Fallback order for the anchor weight on each row:
            1. contribution / return, when numerically safe and nonnegative
            2. cols.WEIGHT, when nonnegative
            3. equal weight fallback

        The preferred reconciliation method is a multiplicative linear tilt:

            w'_i = w_i * (1 + lambda * r_i) / Z

        where Z is the normalization that forces the adjusted weights to sum to 1.

        If the closed-form tilt is unstable or produces materially negative weights, the
        function falls back to:
            1. bisection on the same tilt family
            2. an exact two-security feasible fallback
            3. anchor weights unchanged, as a final fallback

        Args:
            secperf_df: Single-period secperf DataFrame with required columns:
                cols.CONTRIBUTION, cols.IDENTIFIER, cols.RETURN, cols.WEIGHT
            portfolio_return: Single-period portfolio return from portperf.

        Returns:
            Tuple containing:
                adjusted_weights: Normalized adjusted weights summing to 1.0.
                achieved_return: sum(adjusted_weights * cols.RETURN).

        Raises:
            PpaError: If secperf_df is empty.
        """
        # Unexpected defensive check for safety.
        if secperf_df.is_empty():
            raise PpaError(self._error_message("secperf_df must contain at least one row."), 999)

        # Get the contributions, returns, and weights.  We convert to lists so the remaining logic
        # stays in plain Python.
        contributions = (
            secperf_df[cols.CONTRIBUTION].cast(pl.Float64, strict=False).fill_null(0.0).to_list()
        )
        returns = secperf_df[cols.RETURN].cast(pl.Float64, strict=False).fill_null(0.0).to_list()
        weights = (
            secperf_df[cols.WEIGHT]
            .cast(pl.Float64, strict=False)
            .fill_null(float("nan"))
            .to_list()
        )

        # Defensive processing to ensure the lists contain only floats, with nulls filled as
        # specified.  This is to ensure that the subsequent math operations do not encounter
        # unexpected values.
        contributions = [AxysData._finite_or_default(value, 0.0) for value in contributions]
        returns = [AxysData._finite_or_default(value, 0.0) for value in returns]
        weights = [AxysData._finite_or_default(value, float("nan")) for value in weights]

        derived_weight_raw: list[float | None] = []
        for contribution, sec_return in zip(contributions, returns):
            if abs(sec_return) <= _RETURN_EPSILON:
                derived_weight_raw.append(None)
                continue
            implied_weight = contribution / sec_return
            derived_weight_raw.append(implied_weight if implied_weight >= 0.0 else None)

        anchor_weights: list[float] = []

        for implied_weight, weight in zip(
            derived_weight_raw,
            weights,
        ):
            if implied_weight is not None:
                anchor_weights.append(implied_weight)
                continue

            if weight >= 0.0:
                anchor_weights.append(weight)
                continue

            anchor_weights.append(1.0)

        anchor_total = sum(anchor_weights)
        if anchor_total <= 0.0 or not math.isfinite(anchor_total):
            anchor_weights = [1.0] * len(anchor_weights)
            anchor_total = float(len(anchor_weights))

        anchor_weights = [max(0.0, weight) / anchor_total for weight in anchor_weights]

        adjusted_weights = AxysData._solve_adjusted_weights(
            anchor_weights=anchor_weights,
            returns=returns,
            target_return=portfolio_return,
        )

        adjusted_total = sum(adjusted_weights)
        if not math.isfinite(adjusted_total) or adjusted_total <= _NEAR_ZERO_WEIGHT:
            adjusted_weights = [1.0 / float(len(adjusted_weights))] * len(adjusted_weights)
        else:
            adjusted_weights = [weight / adjusted_total for weight in adjusted_weights]

        achieved_return = AxysData._weighted_return(adjusted_weights, returns)

        return adjusted_weights, achieved_return

    def _derive_secperf_for_all_periods(self) -> set[UnreconciledPeriodType]:
        """Derive reconciled self.secperf weights for all periods.

        This method mutates self.secperf by writing a new cols.WEIGHT column while preserving the
        original row alignment and original row grain. It does not group, collapse, or validate
        secperf rows for identifier uniqueness within a period.  It uses a single preallocated
        Python list for the final adjusted weights rather than building one DataFrame per period.

        Returns:
            Set of tuples:
                ((cols.PORTFOLIO_CODE, cols.BEGINNING_DATE, cols.ENDING_DATE),
                  target_return, achieved_return)

        Raises:
            PpaError: On structural or reconciliation failures, such as duplicate
                portperf periods, missing secperf rows for a retained period,
                fatal per-period return mismatch, or incomplete final weight
                assignment.
        """
        dup_periods = (
            self.portperf.group_by(_PERIOD_UNIQUE_KEY_COLUMNS).len().filter(pl.col("len") > 1)
        )
        if not dup_periods.is_empty():
            raise PpaError(
                self._error_message(
                    f"Duplicate portperf periods: {dup_periods.head(10).to_dicts()}"
                ),
                999,
            )

        secperf_with_row_idx = self.secperf.with_row_index(name="_ROW_IDX")
        secperf_lookup = secperf_with_row_idx.partition_by(
            _PERIOD_UNIQUE_KEY_COLUMNS, as_dict=True
        )

        adjusted_weight_values: list[float] = [float("nan")] * self.secperf.height
        unreconciled_periods: set[UnreconciledPeriodType] = set()

        for portfolio_code, from_date, thru_date, port_return in self.portperf.select(
            [
                cols.PORTFOLIO_CODE,
                cols.BEGINNING_DATE,
                cols.ENDING_DATE,
                cols.PORTFOLIO_RETURN,
            ]
        ).iter_rows():
            key = (str(portfolio_code), from_date, thru_date)
            target_return = float(port_return)

            secperf_period = secperf_lookup.get(key)

            # Unexpected defensive check for safety.
            if secperf_period is None or secperf_period.is_empty():
                raise PpaError(self._error_message(f"No secperf rows for period {key}"), 999)

            adjusted_weights, achieved_return = self._derive_reconciled_weights(
                secperf_period,
                target_return,
            )

            for row_idx, adjusted_weight in zip(
                secperf_period["_ROW_IDX"].to_list(), adjusted_weights
            ):
                adjusted_weight_values[int(row_idx)] = adjusted_weight

            difference = abs(achieved_return - target_return)
            if _FATAL_PERIOD_TOLERANCE < difference:
                raise PpaError(
                    self._error_message(f"Return off by {difference} for period {key}"), 503
                )
            if _PERIOD_TOLERANCE < difference:
                unreconciled_periods.add((key, target_return, achieved_return))

        if any(math.isnan(weight) for weight in adjusted_weight_values):
            raise PpaError(
                self._error_message(
                    f"Incomplete {cols.WEIGHT} assignment. One or more secperf rows were not "
                    "assigned a derived weight."
                ),
                999,
            )

        self.secperf = self.secperf.with_columns(
            pl.Series(name=cols.WEIGHT, values=adjusted_weight_values, dtype=pl.Float64)
        )

        return unreconciled_periods

    def _error_message(self, specific_message: str) -> str:
        """Return a consistently formatted error message with context."""
        context = (
            f"Context: "
            f"specifications_path={self.specifications_path}, "
            f"portperf_path={getattr(self, 'portperf_path', None)}, "
            f"secperf_path={getattr(self, 'secperf_path', None)}, "
            f"portfolio_code={getattr(self, 'portfolio_code', None)}, "
            f"from_date={self.from_date}, "
            f"thru_date={self.thru_date}"
        )
        return f"{specific_message}  |  {context}" if specific_message else context

    def _filter_to_common_periods(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Keep only periods that exist in both portperf and secperf.

        This replaces the previous strict validation behavior. Any portperf period
        with no corresponding secperf rows is removed, and any secperf period with
        no corresponding portperf row is also removed.

        Returns:
            Tuple of:
                filtered_portperf_df: Portperf rows whose period exists in secperf.
                filtered_secperf_df: Secperf rows whose period exists in portperf.

        Raises:
            PpaError: If there are no common periods remaining after filtering.
        """
        portperf_periods: pl.DataFrame = self.portperf.select(_PERIOD_UNIQUE_KEY_COLUMNS).unique()
        secperf_periods: pl.DataFrame = self.secperf.select(_PERIOD_UNIQUE_KEY_COLUMNS).unique()

        common_periods: pl.DataFrame = portperf_periods.join(
            secperf_periods,
            on=_PERIOD_UNIQUE_KEY_COLUMNS,
            how="inner",
        )

        if common_periods.is_empty():
            # Intentionally pass no specific message text here. PpaError 505 provides the
            # descriptive error text elsewhere; _error_message("") adds only contextual details.
            raise PpaError(self._error_message(""), 505)

        filtered_portperf_df = self.portperf.join(
            common_periods,
            on=_PERIOD_UNIQUE_KEY_COLUMNS,
            how="inner",
        )

        filtered_secperf_df = self.secperf.join(
            common_periods,
            on=_PERIOD_UNIQUE_KEY_COLUMNS,
            how="inner",
        )

        return filtered_portperf_df, filtered_secperf_df

    @staticmethod
    def _finite_or_default(value: float | None, default: float) -> float:
        """Return a finite float or a default fallback value."""
        if value is None:
            return default
        return value if math.isfinite(value) else default

    def _get_classification_or_mapping_data_source(
        self,
        ds_type: Literal["classification", "mapping"],
        ds_name: str | None,
        unique_security_ids: list[str],
    ) -> pl.DataFrame:
        """Get the classification or mapping data source."""
        if not ds_name:
            return pl.DataFrame()

        # Get the available data source specifications from the Axys json specifications file.
        data_sources: dict[str, dict[str, Any]] = self.specifications.get(f"{ds_type}s", {})
        if ds_name not in data_sources:
            raise PpaError(
                self._error_message(f"Unknown {ds_type} {ds_name!r}"),
                504,
            )

        # Get the data source specification from the Axys json specifications file.
        data_source: dict[str, Any] = data_sources[ds_name]
        data_source_fields = set(data_source)  # keys

        # Make sure there are no unknown fields.
        unknown_fields = data_source_fields - _CLASSIFICATION_MAPPING_FIELDS_ALLOWED
        if unknown_fields:
            raise PpaError(
                self._error_message(f"Unknown fields for {ds_type} {ds_name!r}: {unknown_fields}"),
                504,
            )

        # Make sure you have the required fields.
        missing_fields = _CLASSIFICATION_MAPPING_FIELDS_REQUIRED - data_source_fields
        if missing_fields:  # not _CLASSIFICATION_MAPPING_FIELDS_REQUIRED.issubset(data_source):
            raise PpaError(
                self._error_message(f"Missing fields for {ds_type} {ds_name!r}: {missing_fields}"),
                504,
            )

        # Make sure that file_path exists.
        file_path: str = data_source["file_path"]
        if not util.has_directory(file_path):
            file_path = os.path.join(os.path.dirname(self.specifications_path), file_path)
        if not util.file_path_exists(file_path):
            raise PpaError(self._error_message(util.file_path_error(file_path)), None)

        # Get the lazy_frame
        lf: pl.LazyFrame = pl.scan_csv(file_path)

        # Make sure that all column names are actually in the lazy frame.
        specified_column_names = {
            data_source[k] for k in _CLASSIFICATION_MAPPING_COLUMN_NAMES if k in data_source
        }
        nonexistent_column_names = specified_column_names - set(lf.collect_schema().names())
        if nonexistent_column_names:
            raise PpaError(
                self._error_message(
                    f"Nonexistent column names for {ds_type} {ds_name!r}: "
                    f"{nonexistent_column_names}"
                ),
                504,
            )

        # Optionally filter on security ID so you do not load the entire security master.
        is_security_master = data_source.get("is_security_master", False)
        if not isinstance(is_security_master, bool):
            raise PpaError(
                self._error_message(
                    f"Invalid is_security_master value for {ds_type} {ds_name!r}: "
                    f"{is_security_master!r} must be a boolean."
                ),
                504,
            )
        if is_security_master:
            # Convert to a plain Python list for is_in()
            lf = lf.filter(pl.col(data_source["identifier_column"]).is_in(unique_security_ids))

        # Additional optional filter.
        if {"filter_column", "filter_value"}.issubset(data_source):
            lf = lf.filter(pl.col(data_source["filter_column"]) == data_source["filter_value"])

        # Rename source columns to the standardized output column names used by this method.
        rename_mappings = {
            data_source["identifier_column"]: "identifier_column",
            data_source["name_column"]: "name_column",
        }

        return (
            lf.collect()
            .rename(rename_mappings)
            .select(("identifier_column", "name_column"))
            .unique(subset=["identifier_column"], keep="any")
        )

    def _get_performance(
        self,
        file_path: str,
        column_name_mappings_name: Literal["portperf_columns", "secperf_columns"],
    ) -> pl.DataFrame:
        """Load a performance CSV and return a DataFrame using internal column names.

        This method reads a CSV file, validates that required columns are defined in the
        JSON specifications and present in the file, and returns a DataFrame with columns
        renamed to the internal names used by this codebase.

        JSON Mapping:
            The JSON mapping (e.g., "portperf_columns", "secperf_columns") must map
            internal column names to CSV column names:

                {internal_name: csv_header_name}

            The mapping is reversed internally so that CSV columns are renamed to
            internal column names.

        Processing:
            - Resolve file path (absolute or relative to specifications directory)
            - Validate required columns exist in the JSON mapping
            - Validate mapped CSV columns exist in the file
            - Load CSV via Polars lazy scan
            - Rename columns to internal names
            - Select required columns
            - Parse date columns
            - Apply optional filters (portfolio_code, from_date, thru_date)

        Args:
            file_path: Path to the CSV file.
            column_name_mappings_name: JSON mapping section name.

        Returns:
            Polars DataFrame with required columns using internal names.

        Raises:
            PpaError: On missing file or missing required columns.
        """
        # Make sure that file_path exists.
        if not util.has_directory(file_path):
            file_path = os.path.join(os.path.dirname(self.specifications_path), file_path)
        if not util.file_path_exists(file_path):
            raise PpaError(self._error_message(util.file_path_error(file_path)), None)

        # Determine the required columns.
        required_columns = (
            _PORTPERF_REQUIRED_COLUMNS
            if column_name_mappings_name == "portperf_columns"
            else _SECPERF_REQUIRED_COLUMNS
        )

        # Read the JSON mapping in its declared direction:
        #     internal_name -> csv_header_name
        json_column_mappings: dict[str, str] = self.specifications.get(
            column_name_mappings_name, {}
        )
        available_columns = set(json_column_mappings)

        # First validate that all required internal columns are present in the JSON mapping.
        missing_columns: set[str] = required_columns - available_columns

        # Reverse the mapping for Polars rename():
        #     csv_header_name -> internal_name
        csv_to_internal_mappings: dict[str, str] = {}

        if not missing_columns:
            csv_to_internal_mappings = {v: k for k, v in json_column_mappings.items()}

            # Validate that the mapped CSV columns actually exist in the input file header.
            header_df: pl.DataFrame = pl.read_csv(file_path, n_rows=0)
            available_columns = {
                csv_to_internal_mappings[col]
                for col in csv_to_internal_mappings
                if col in header_df.columns
            }
            missing_columns = required_columns - available_columns

        # Raise an error if you do not have all of the required columns.
        if missing_columns:
            raise PpaError(
                self._error_message(
                    f"Missing {sorted(missing_columns)} in {file_path!r}.  |  "
                    f"Columns available are: {sorted(available_columns)}"
                ),
                502,
            )

        # Load the mapped required columns.
        lazy_frame: pl.LazyFrame = (
            pl.scan_csv(file_path)
            .rename(csv_to_internal_mappings)
            .select(required_columns)
            .with_columns(
                pl.col(cols.BEGINNING_DATE).str.strptime(pl.Date, "%Y-%m-%d", strict=True),
                pl.col(cols.ENDING_DATE).str.strptime(pl.Date, "%Y-%m-%d", strict=True),
            )
        )

        # Filter on portfolio_code, from_date, and thru_date if specified.
        if hasattr(self, "portfolio_code"):
            lazy_frame = lazy_frame.filter(pl.col(cols.PORTFOLIO_CODE) == self.portfolio_code)

        if self.from_date is not None:
            lazy_frame = lazy_frame.filter(pl.lit(self.from_date) <= pl.col(cols.BEGINNING_DATE))

        if self.thru_date is not None:
            lazy_frame = lazy_frame.filter(pl.col(cols.ENDING_DATE) <= pl.lit(self.thru_date))

        return lazy_frame.collect()

    @staticmethod
    def _solve_adjusted_weights(
        anchor_weights: list[float],
        returns: list[float],
        target_return: float,
    ) -> list[float]:
        """Solve for adjusted weights using fallback methods.

        Args:
            anchor_weights: Nonnegative weights summing to 1.
            returns: Single-period security returns.
            target_return: Desired weighted average return.

        Returns:
            Adjusted nonnegative normalized weights.
        """
        anchor_return = AxysData._weighted_return(anchor_weights, returns)
        if abs(anchor_return - target_return) <= _MATCH_TOLERANCE:
            return anchor_weights

        closed_form_weights = AxysData._solve_closed_form_tilt(
            anchor_weights=anchor_weights,
            returns=returns,
            target_return=target_return,
            near_zero_weight=_NEAR_ZERO_WEIGHT,
        )
        if closed_form_weights is not None:
            closed_form_return = AxysData._weighted_return(closed_form_weights, returns)
            if abs(closed_form_return - target_return) <= 10.0 * _MATCH_TOLERANCE:
                return closed_form_weights

        bisection_weights = AxysData._solve_bisection_tilt(
            anchor_weights=anchor_weights,
            returns=returns,
            target_return=target_return,
        )
        if bisection_weights is not None:
            bisection_return = AxysData._weighted_return(bisection_weights, returns)
            if abs(bisection_return - target_return) <= 10.0 * _MATCH_TOLERANCE:
                return bisection_weights

        two_security_weights = AxysData._solve_two_security_fallback(
            anchor_weights=anchor_weights,
            returns=returns,
            target_return=target_return,
        )
        if two_security_weights is not None:
            return two_security_weights

        return anchor_weights

    @staticmethod
    def _solve_bisection_tilt(
        anchor_weights: list[float],
        returns: list[float],
        target_return: float,
        max_iterations: int = 200,
    ) -> list[float] | None:
        """Solve the same tilt family by bisection over a sampled lambda grid."""
        candidate_lambdas = [
            -1.0e12,
            -1.0e9,
            -1.0e6,
            -1.0e3,
            -1.0,
            -1.0e-3,
            0.0,
            1.0e-3,
            1.0,
            1.0e3,
            1.0e6,
            1.0e9,
            1.0e12,
        ]

        valid_points: list[tuple[float, float]] = []
        for lambda_value in candidate_lambdas:
            weights = AxysData._weights_from_lambda(
                anchor_weights=anchor_weights,
                returns=returns,
                lambda_value=lambda_value,
                near_zero_weight=_NEAR_ZERO_WEIGHT,
            )
            if weights is None:
                continue

            residual = AxysData._weighted_return(weights, returns) - target_return
            if math.isfinite(residual):
                valid_points.append((lambda_value, residual))

        if not valid_points:
            return None

        for lambda_value, residual in valid_points:
            if abs(residual) <= _MATCH_TOLERANCE:
                return AxysData._weights_from_lambda(
                    anchor_weights=anchor_weights,
                    returns=returns,
                    lambda_value=lambda_value,
                    near_zero_weight=_NEAR_ZERO_WEIGHT,
                )

        for left_point, right_point in zip(valid_points[:-1], valid_points[1:]):
            left_lambda, left_residual = left_point
            right_lambda, right_residual = right_point

            if left_residual * right_residual > 0.0:
                continue

            lower_lambda = left_lambda
            upper_lambda = right_lambda
            lower_residual = left_residual

            for _ in range(max_iterations):
                mid_lambda = 0.5 * (lower_lambda + upper_lambda)
                mid_weights = AxysData._weights_from_lambda(
                    anchor_weights=anchor_weights,
                    returns=returns,
                    lambda_value=mid_lambda,
                    near_zero_weight=_NEAR_ZERO_WEIGHT,
                )
                if mid_weights is None:
                    return None

                mid_residual = AxysData._weighted_return(mid_weights, returns) - target_return
                if abs(mid_residual) <= _MATCH_TOLERANCE:
                    return mid_weights

                if lower_residual * mid_residual <= 0.0:
                    upper_lambda = mid_lambda
                else:
                    lower_lambda = mid_lambda
                    lower_residual = mid_residual

            final_lambda = 0.5 * (lower_lambda + upper_lambda)
            return AxysData._weights_from_lambda(
                anchor_weights=anchor_weights,
                returns=returns,
                lambda_value=final_lambda,
                near_zero_weight=_NEAR_ZERO_WEIGHT,
            )

        return None

    @staticmethod
    def _solve_closed_form_tilt(
        anchor_weights: list[float],
        returns: list[float],
        target_return: float,
        near_zero_weight: float,
    ) -> list[float] | None:
        """Solve the linear tilt using the closed-form lambda when possible."""
        anchor_return = AxysData._weighted_return(anchor_weights, returns)
        second_moment = sum(
            weight * sec_return * sec_return for weight, sec_return in zip(anchor_weights, returns)
        )
        denominator = second_moment - (target_return * anchor_return)

        if not math.isfinite(denominator) or abs(denominator) <= near_zero_weight:
            return None

        lambda_value = (target_return - anchor_return) / denominator
        return AxysData._weights_from_lambda(
            anchor_weights=anchor_weights,
            returns=returns,
            lambda_value=lambda_value,
            near_zero_weight=near_zero_weight,
        )

    @staticmethod
    def _solve_two_security_fallback(
        anchor_weights: list[float],
        returns: list[float],
        target_return: float,
    ) -> list[float] | None:
        """Construct an exact long-only solution using one or two securities.

        This is intentionally a last-resort fallback. It is exact whenever the target lies within
        the min/max range of available security returns.  But it can move far away from the anchor
        weights. To keep it less arbitrary, it chooses the bracketing pair with the largest
        combined anchor weight.

        Args:
            anchor_weights: Nonnegative anchor weights summing to 1.
            returns: Security returns.
            target_return: Desired weighted average return.

        Returns:
            Exact nonnegative normalized weights if found, otherwise None.
        """
        for row_index, sec_return in enumerate(returns):
            if abs(sec_return - target_return) <= _MATCH_TOLERANCE:
                weights = [0.0] * len(returns)
                weights[row_index] = 1.0
                return weights

        best_pair: tuple[int, int] | None = None
        best_pair_score = -1.0

        for left_index, left_return in enumerate(returns):
            for right_index in range(left_index + 1, len(returns)):
                right_return = returns[right_index]

                low_return = min(left_return, right_return)
                high_return = max(left_return, right_return)
                if target_return < low_return - _MATCH_TOLERANCE:
                    continue
                if target_return > high_return + _MATCH_TOLERANCE:
                    continue
                if abs(left_return - right_return) <= _MATCH_TOLERANCE:
                    continue

                pair_score = anchor_weights[left_index] + anchor_weights[right_index]
                if pair_score > best_pair_score:
                    best_pair = (left_index, right_index)
                    best_pair_score = pair_score

        if best_pair is None:
            return None

        left_index, right_index = best_pair
        left_return = returns[left_index]
        right_return = returns[right_index]

        if abs(right_return - left_return) <= _MATCH_TOLERANCE:
            return None

        right_weight = (target_return - left_return) / (right_return - left_return)
        left_weight = 1.0 - right_weight

        if left_weight < -_MATCH_TOLERANCE or right_weight < -_MATCH_TOLERANCE:
            return None

        weights = [0.0] * len(returns)
        weights[left_index] = max(0.0, left_weight)
        weights[right_index] = max(0.0, right_weight)

        total_weight = sum(weights)
        if total_weight <= 0.0:
            return None

        return [weight / total_weight for weight in weights]

    @staticmethod
    def _weighted_return(weights: list[float], returns: list[float]) -> float:
        """Return the weighted average of the security returns."""
        return sum(weight * sec_return for weight, sec_return in zip(weights, returns))

    @staticmethod
    def _weights_from_lambda(
        anchor_weights: list[float],
        returns: list[float],
        lambda_value: float,
        near_zero_weight: float,
    ) -> list[float] | None:
        """Compute adjusted weights for a specific lambda value.

        Args:
            anchor_weights: Nonnegative weights summing to 1.
            returns: Security returns.
            lambda_value: Tilt parameter.
            near_zero_weight: Tiny threshold used when cleaning weights.

        Returns:
            A cleaned and renormalized weight vector if lambda is usable; otherwise None.
        """
        anchor_return = AxysData._weighted_return(anchor_weights, returns)
        normalization = 1.0 + (lambda_value * anchor_return)

        if not math.isfinite(normalization) or abs(normalization) <= near_zero_weight:
            return None

        raw_weights: list[float] = []
        for anchor_weight, sec_return in zip(anchor_weights, returns):
            scale = 1.0 + (lambda_value * sec_return)
            if not math.isfinite(scale):
                return None
            raw_weights.append(anchor_weight * scale / normalization)

        if any(weight < -near_zero_weight for weight in raw_weights):
            return None

        cleaned_weights = [0.0 if weight < 0.0 else weight for weight in raw_weights]
        total_weight = sum(cleaned_weights)
        if not math.isfinite(total_weight) or total_weight <= near_zero_weight:
            return None

        return [weight / total_weight for weight in cleaned_weights]
