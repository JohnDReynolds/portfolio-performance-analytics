"""Permissive single-period weight reconciliation helpers for Axys-style secperf data."""

from __future__ import annotations

# Python imports
import datetime as dt
import math
from typing import Final, Sequence

# Third-party imports
import polars as pl

# Project imports
import ppar.columns as cols
from ppar.errors import PpaError
import ppar.utilities as util

_MATCH_TOLERANCE: Final[float] = 1e-12
_NEAR_ZERO_WEIGHT: Final[float] = 1e-18
_OVERALL_PERIODS_TOLERANCE = 0.0001  # 1 basis point
_PERIOD_TOLERANCE: Final[float] = 0.000001  # 1/100 of a basis point
_RETURN_EPSILON: Final[float] = 1e-12

_ANALYTICS_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    cols.BEGINNING_DATE,
    cols.ENDING_DATE,
    cols.RETURN,
    cols.IDENTIFIER,
    cols.WEIGHT,
)

_PORTPERF_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    cols.BEGINNING_DATE,
    cols.ENDING_DATE,
    cols.PORTFOLIO_CODE,
    cols.PORTFOLIO_NAME,
    cols.PORTFOLIO_RETURN,
)

_SECPERF_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    cols.BEGIN_MV,
    cols.BEGIN_WEIGHT,
    cols.CONTRIBUTION_W_X_R,
    cols.BEGINNING_DATE,
    cols.PORTFOLIO_CODE,
    cols.RETURN,
    cols.IDENTIFIER,
    cols.ENDING_DATE,
)

_SECPERF_WEIGHT_RETURN_COLUMNS: Final[tuple[str, ...]] = (
    cols.CONTRIBUTION_W_X_R,
    cols.BEGIN_MV,
    cols.BEGIN_WEIGHT,
    cols.RETURN,
    cols.IDENTIFIER,
)

_PERIOD_KEY_COLUMNS: Final[tuple[str, ...]] = (
    cols.PORTFOLIO_CODE,
    cols.BEGINNING_DATE,
    cols.ENDING_DATE,
)

UnreconciledPeriodType = tuple[tuple[str, dt.date, dt.date], float, float]


class AxysData:
    """Load, validate, and derive Axys-style portperf/secperf data.

      1. Loads portperf and secperf using lazy CSV scans.
      2. Projects only the required columns.
      3. Applies optional filters on portfolio_code, from_date, and thru_date.
      4. Validates secperf uniqueness on:
         cols.PORTFOLIO_CODE, cols.BEGINNING_DATE, cols.ENDING_DATE, cols.IDENTIFIER
      5. Validates secperf has exactly the same distinct periods as portperf.
      6. Derives reconciled secperf weights for all periods and writes them into
         self.secperf as cols.WEIGHT.
      7. Captures unreconciled periods in self.unreconciled_periods.

    Attributes:
        portperf_path: Path to the portperf CSV.
        secperf_path: Path to the secperf CSV.
        portfolio_code: Optional portfolio filter.
        from_date: Optional lower date bound.
        thru_date: Optional upper date bound.
        portperf: Loaded and validated portperf data.
        secperf: Loaded and validated secperf data, with cols.WEIGHT added after derivation.
        unreconciled_periods: Set of tuples containing the following for periods that do not
          reconcile within tolerance:
            ((cols.PORTFOLIO_CODE, cols.BEGINNING_DATE, cols.ENDING_DATE),
              target_return, achieved_return)

    """

    def __init__(
        self,
        portperf_path: str,
        secperf_path: str,
        axysdata_json_path: str | None = None,
        portfolio_code: str | None = None,
        from_date: dt.date | None = None,
        thru_date: dt.date | None = None,
    ) -> None:
        """Initialize AxysData and load/validate all requested data.

        Args:
            portperf_path: Path to portperf CSV.
            secperf_path: Path to secperf CSV.
            axysdata_json_path: Contains column name mappings and processing rules.
            portfolio_code: Optional portfolio filter.
            from_date: Optional lower date bound. Keeps rows where
                from_date <= cols.BEGINNING_DATE.
            thru_date: Optional upper date bound. Keeps rows where
                cols.ENDING_DATE <= thru_date.

        Raises:
            PpaError: If any validation fails or if file/schema validation fails.
        """
        # Get the axysdata specifications from the json if provided.
        axysdata_json: dict[str, dict[str, str]] | None = (
            util.read_dict_from_json_file(axysdata_json_path)
            if axysdata_json_path is not None
            else {}
        )

        # Set the class members.
        self.from_date: dt.date | None = from_date
        # self.portperf_columns: dict[str, str] = axysdata_json.get("portperf_columns", {})
        self.portperf_path: str = portperf_path
        self.portfolio_code: str | None = portfolio_code
        self.processing_rules = axysdata_json.get("processing_rules", {})
        # self.secperf_columns = axysdata_json.get("secperf_columns", {})
        self.secperf_path: str = secperf_path
        self.thru_date: dt.date | None = thru_date

        # Load portperf and secperf data.
        self.portperf: pl.DataFrame = self._scan_csv_selected_columns(
            self.portperf_path,
            _PORTPERF_REQUIRED_COLUMNS,
            axysdata_json.get("portperf_columns", {}),
        )
        self.secperf: pl.DataFrame = self._scan_csv_selected_columns(
            self.secperf_path,
            _SECPERF_REQUIRED_COLUMNS,
            axysdata_json.get("secperf_columns", {}),
        )

        # Validate
        self._validate_secperf_uniqueness(self.secperf)
        # self._validate_portperf_row_date_order(self.portperf)
        # self._validate_portperf_continuous_periods(self.portperf)
        self._validate_secperf_periods_match_portperf(self.portperf, self.secperf)

        # Derive the secperf weights.
        self.unreconciled_periods: set[UnreconciledPeriodType] = (
            self._derive_secperf_for_all_periods()
        )

        # In theory, unreconciled periods should be rare and have minimal return differences.
        difference = abs(
            sum(t for _, t, _ in self.unreconciled_periods)
            - sum(a for _, _, a in self.unreconciled_periods)
        )
        if difference > _OVERALL_PERIODS_TOLERANCE:
            raise PpaError(
                self._error_message(
                    f"Returns difference across unreconciled periods is {difference}"
                ),
                503,
            )

        # Set self.portfolio_name based on self.processing_rules
        self.portfolio_name = self.portperf[cols.PORTFOLIO_NAME][0]
        prefix_portfolio_code = self.processing_rules.get("prefix_portfolio_code")
        if prefix_portfolio_code:
            self.portfolio_name = (
                f"{self.portperf[cols.PORTFOLIO_CODE][0]}{prefix_portfolio_code}"
                f"{self.portfolio_name}"
            )

        # Only include the columns you need for Analytics.
        self.secperf = self.secperf.select(_ANALYTICS_REQUIRED_COLUMNS)

        # You do not need portperf anymore.
        del self.portperf

    def _derive_reconciled_weights(
        self,
        secperf_df: pl.DataFrame,
        portfolio_return: float,
    ) -> tuple[list[float], float]:
        """Derive single-period security weights with fallbacks.

        This implementation is intentionally lean:
            - It returns only the normalized adjusted weights and the achieved return.
            - The caller writes cols.WEIGHT into self.secperf.
            - No debug/helper columns are materialized.

        Fallback order for the anchor weight on each row:
            1. contribution / return, when numerically safe and nonnegative
            2. cols.BEGIN_WEIGHT, when nonnegative
            3. cols.BEGIN_MV-based normalized weight, when nonnegative
            4. equal weight fallback

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
                cols.IDENTIFIER, cols.RETURN, cols.CONTRIBUTION_W_X_R, cols.BEGIN_WEIGHT,
                cols.BEGIN_MV.
            portfolio_return: Single-period portfolio return from portperf.

        Returns:
            Tuple containing:
                adjusted_weights: Normalized adjusted weights summing to 1.0.
                achieved_return: sum(adjusted_weights * cols.RETURN).

        Raises:
            PpaError: If secperf_df is empty.
        """
        if secperf_df.height == 0:
            raise PpaError(self._error_message("secperf_df must contain at least one row."), 999)

        working_df = secperf_df.select(list(_SECPERF_WEIGHT_RETURN_COLUMNS)).with_columns(
            pl.col(cols.RETURN).cast(pl.Float64, strict=False).fill_null(0.0),
            pl.col(cols.CONTRIBUTION_W_X_R).cast(pl.Float64, strict=False).fill_null(0.0),
            pl.col(cols.BEGIN_WEIGHT).cast(pl.Float64, strict=False).fill_null(float("nan")),
            pl.col(cols.BEGIN_MV).cast(pl.Float64, strict=False).fill_null(float("nan")),
        )

        returns = [float(value) for value in working_df[cols.RETURN].to_list()]
        contributions = [float(value) for value in working_df[cols.CONTRIBUTION_W_X_R].to_list()]
        begin_weights = [float(value) for value in working_df[cols.BEGIN_WEIGHT].to_list()]
        begin_market_values = [float(value) for value in working_df[cols.BEGIN_MV].to_list()]

        returns = [AxysData._finite_or_default(value, default=0.0) for value in returns]
        contributions = [
            AxysData._finite_or_default(value, default=0.0) for value in contributions
        ]
        begin_weights = [
            AxysData._finite_or_default(value, default=float("nan")) for value in begin_weights
        ]
        begin_market_values = [
            AxysData._finite_or_default(value, default=float("nan"))
            for value in begin_market_values
        ]

        derived_weight_raw: list[float | None] = []
        for contribution, sec_return in zip(contributions, returns):
            if abs(sec_return) <= _RETURN_EPSILON:
                derived_weight_raw.append(None)
                continue

            implied_weight = contribution / sec_return
            if math.isfinite(implied_weight) and implied_weight >= 0.0:
                derived_weight_raw.append(implied_weight)
            else:
                derived_weight_raw.append(None)

        cleaned_begin_mvs = [
            begin_mv if math.isfinite(begin_mv) and begin_mv >= 0.0 else 0.0
            for begin_mv in begin_market_values
        ]
        total_begin_mv = sum(cleaned_begin_mvs)
        if total_begin_mv > 0.0:
            mv_weights: list[float | None] = [
                begin_mv / total_begin_mv for begin_mv in cleaned_begin_mvs
            ]
        else:
            mv_weights = [None] * len(cleaned_begin_mvs)

        anchor_weights: list[float] = []

        for implied_weight, begin_weight, mv_weight in zip(
            derived_weight_raw,
            begin_weights,
            mv_weights,
        ):
            if implied_weight is not None:
                anchor_weights.append(implied_weight)
                continue

            if math.isfinite(begin_weight) and begin_weight >= 0.0:
                anchor_weights.append(begin_weight)
                continue

            if mv_weight is not None and math.isfinite(mv_weight) and mv_weight >= 0.0:
                anchor_weights.append(mv_weight)
                continue

            anchor_weights.append(1.0)

        anchor_total = sum(anchor_weights)
        if anchor_total <= 0.0 or not math.isfinite(anchor_total):
            anchor_weights = [1.0] * len(anchor_weights)
            anchor_total = float(len(anchor_weights))

        anchor_weights = [max(0.0, weight) / anchor_total for weight in anchor_weights]

        min_return = min(returns)
        max_return = max(returns)
        effective_target = min(max(portfolio_return, min_return), max_return)

        adjusted_weights = AxysData._solve_adjusted_weights(
            anchor_weights=anchor_weights,
            returns=returns,
            target_return=effective_target,
        )

        adjusted_total = sum(adjusted_weights)
        if not math.isfinite(adjusted_total) or adjusted_total <= _NEAR_ZERO_WEIGHT:
            adjusted_weights = [1.0 / float(len(adjusted_weights))] * len(adjusted_weights)
        else:
            adjusted_weights = [weight / adjusted_total for weight in adjusted_weights]

        achieved_return = AxysData._weighted_return(adjusted_weights, returns)

        return adjusted_weights, achieved_return

    def _derive_secperf_for_all_periods(self) -> set[UnreconciledPeriodType]:
        """Derive reconciled secperf weights for all periods.

        This method mutates self.secperf by adding a new cols.WEIGHT column while
        preserving original row alignment. It uses a single preallocated Python list for
        the final adjusted weights rather than building one DataFrame per period.

        Returns:
            Set of tuples:
                ((cols.PORTFOLIO_CODE, cols.BEGINNING_DATE, cols.ENDING_DATE),
                  target_return, achieved_return)

        Raises:
            PpaError: On structural validation failures such as duplicate periods,
                missing secperf rows for a portperf period, or incomplete weight assignment.
        """
        dup_periods = (
            self.portperf.group_by(list(_PERIOD_KEY_COLUMNS)).len().filter(pl.col("len") > 1)
        )
        if dup_periods.height > 0:
            raise PpaError(
                self._error_message(
                    f"Duplicate portperf periods: {dup_periods.head(10).to_dicts()}"
                ),
                999,
            )

        secperf_with_row_idx = self.secperf.with_row_index(name="_ROW_IDX")
        secperf_lookup = secperf_with_row_idx.partition_by(
            list(_PERIOD_KEY_COLUMNS),
            as_dict=True,
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
            if secperf_period is None or secperf_period.height == 0:
                raise PpaError(self._error_message(f"No secperf rows for period {key}"), 999)

            adjusted_weights, achieved_return = self._derive_reconciled_weights(
                secperf_period,
                target_return,
            )

            row_indices = secperf_period["_ROW_IDX"].to_list()
            for row_idx, adjusted_weight in zip(row_indices, adjusted_weights):
                adjusted_weight_values[int(row_idx)] = adjusted_weight

            if _PERIOD_TOLERANCE < abs(achieved_return - target_return):
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
        """Helper to raise a PpaError with consistent formatting."""
        return (
            f"{specific_message}  |  Context: "
            f"portperf_path={self.portperf_path}, secperf_path={self.secperf_path}, "
            f"portfolio_code={self.portfolio_code}, "
            f"from_date={self.from_date}, thru_date={self.thru_date}"
        )

    @staticmethod
    def _finite_or_default(value: float, *, default: float) -> float:
        """Return a finite float or a default fallback value."""
        return value if math.isfinite(value) else default

    def _scan_csv_selected_columns(
        self,
        file_path: str,
        requested_columns: Sequence[str],
        column_name_mappings: dict[str, str],
    ) -> pl.DataFrame:
        """Read a CSV lazily, project only requested columns, and apply instance filters.

        Args:
            file_path: CSV file path.
            requested_columns: Exact column names to read.

        Returns:
            A collected Polars DataFrame containing only the requested columns.

        Raises:
            PpaError: If one or more requested columns are missing.
        """
        # Make sure that file_path exists.
        if not util.file_path_exists(file_path):
            raise PpaError(self._error_message(util.file_path_error(file_path)), None)

        # Default column_name_mappings to map requested_columns to requested_columns.
        if not column_name_mappings:
            column_name_mappings = {col: col for col in requested_columns}

        # Get the mapped column names.
        header_df: pl.DataFrame = pl.read_csv(file_path, n_rows=0)
        available_columns: set[str] = {
            column_name_mappings.get(col, col) for col in header_df.columns
        }

        # Make sure that all of the requested_columns exist.
        missing_columns: list[str] = [
            col for col in requested_columns if col not in available_columns
        ]
        if missing_columns:
            raise PpaError(
                self._error_message(
                    f"Missing {missing_columns} in {file_path!r}.  |  "
                    f"Columns available are: {sorted(available_columns)}"
                ),
                502,
            )

        # Load the mapped requested_columns.
        lazy_frame: pl.LazyFrame = (
            pl.scan_csv(file_path)
            .rename(column_name_mappings)
            .select(list(requested_columns))
            .with_columns(
                pl.col(cols.BEGINNING_DATE).str.strptime(pl.Date, "%Y-%m-%d", strict=True),
                pl.col(cols.ENDING_DATE).str.strptime(pl.Date, "%Y-%m-%d", strict=True),
            )
        )

        if self.portfolio_code is not None:
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

        This is intentionally a last-resort fallback. It is exact whenever the target lies
        within the feasible range of security returns, but it can move far away from the
        anchor weights. To keep it less arbitrary, it chooses the bracketing pair with the
        largest combined anchor weight.

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

    def _validate_secperf_periods_match_portperf(
        self,
        portperf_df: pl.DataFrame,
        secperf_df: pl.DataFrame,
    ) -> None:
        """Validate that secperf has exactly the same distinct periods as portperf."""
        portperf_periods: pl.DataFrame = (
            portperf_df.select(list(_PERIOD_KEY_COLUMNS)).unique().sort(list(_PERIOD_KEY_COLUMNS))
        )

        secperf_periods: pl.DataFrame = (
            secperf_df.select(list(_PERIOD_KEY_COLUMNS)).unique().sort(list(_PERIOD_KEY_COLUMNS))
        )

        missing_in_secperf: pl.DataFrame = portperf_periods.join(
            secperf_periods,
            on=list(_PERIOD_KEY_COLUMNS),
            how="anti",
        ).sort(list(_PERIOD_KEY_COLUMNS))

        extra_in_secperf: pl.DataFrame = secperf_periods.join(
            portperf_periods,
            on=list(_PERIOD_KEY_COLUMNS),
            how="anti",
        ).sort(list(_PERIOD_KEY_COLUMNS))

        if missing_in_secperf.height > 0 or extra_in_secperf.height > 0:
            missing_sample: list[dict[str, object]] = missing_in_secperf.head(10).to_dicts()
            extra_sample: list[dict[str, object]] = extra_in_secperf.head(10).to_dicts()
            raise PpaError(
                self._error_message(
                    "secperf/portperf period validation failed. "
                    "secperf must have exactly the same distinct periods as portperf. "
                    f"Missing periods in secperf: {missing_sample}. "
                    f"Extra periods in secperf: {extra_sample}"
                ),
                999,
            )

    def _validate_secperf_uniqueness(self, secperf_df: pl.DataFrame) -> None:
        """Validate secperf uniqueness at portfolio/period/security grain."""
        duplicate_rows: pl.DataFrame = (
            secperf_df.group_by(
                [
                    cols.PORTFOLIO_CODE,
                    cols.BEGINNING_DATE,
                    cols.ENDING_DATE,
                    cols.IDENTIFIER,
                ]
            )
            .len()
            .filter(pl.col("len") > 1)
            .sort(
                [
                    cols.PORTFOLIO_CODE,
                    cols.BEGINNING_DATE,
                    cols.ENDING_DATE,
                    cols.IDENTIFIER,
                ]
            )
        )

        if duplicate_rows.height > 0:
            sample_rows: list[dict[str, object]] = duplicate_rows.head(10).to_dicts()
            raise PpaError(
                self._error_message(
                    f"secperf uniqueness validation failed. {cols.PORTFOLIO_CODE}, "
                    f"{cols.BEGINNING_DATE}, {cols.ENDING_DATE}, {cols.IDENTIFIER} do not "
                    f"uniquely identify each row. Sample duplicates: {sample_rows}"
                ),
                999,
            )

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
