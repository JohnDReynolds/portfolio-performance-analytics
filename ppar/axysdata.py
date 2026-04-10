"""Permissive single-period weight reconciliation helpers for Axys-style secperf data."""

from __future__ import annotations

# Python imports
import datetime as dt
import math
from typing import Final, Sequence

# Third-party imports
import polars as pl

# Project imports
import ppar.errors as errs
import ppar.utilities as util


_PORTPERF_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "PORTFOLIO_CODE",
    "FROM_DATE",
    "THRU_DATE",
    "PORTFOLIO_NAME",
    "PORT_RETURN",
)

_SECPERF_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "PORTFOLIO_CODE",
    "FROM_DATE",
    "THRU_DATE",
    "SECURITY_ID",
    "SEC_RETURN",
    "CONTRIBUTION_W_X_R",
    "BEGIN_WEIGHT",
    "BEGIN_MV",
)

_SECPERF_WEIGHT_RETURN_COLUMNS: Final[tuple[str, ...]] = (
    "SECURITY_ID",
    "SEC_RETURN",
    "CONTRIBUTION_W_X_R",
    "BEGIN_WEIGHT",
    "BEGIN_MV",
)


class AxysData:
    """Load, validate, and derive Axys-style portperf/secperf data.

    The constructor performs the same work that was previously done by
    load_and_validate_portperf_and_secperf():

      1. Loads portperf and secperf using lazy CSV scans.
      2. Projects only the required columns.
      3. Applies optional filters on portfolio_code, from_date, and thru_date.
      4. Validates secperf uniqueness on:
         PORTFOLIO_CODE, FROM_DATE, THRU_DATE, SECURITY_ID
      5. Validates each portperf row has FROM_DATE <= THRU_DATE.
      6. Validates portperf periods form one continuous range with no gaps or overlaps.
      7. Validates secperf has exactly the same distinct periods as portperf.
      8. Derives reconciled secperf weights for all periods.

    Attributes:
        portperf_path: Path to the portperf CSV.
        secperf_path: Path to the secperf CSV.
        portfolio_code: Optional portfolio filter.
        from_date: Optional lower date bound.
        thru_date: Optional upper date bound.
        portperf: Loaded and validated portperf data.
        secperf: Loaded and validated secperf data.
        derived_secperf: secperf with derived/reconciled weight fields added.
    """

    def __init__(
        self,
        portperf_path: str,
        secperf_path: str,
        portfolio_code: str | None = None,
        from_date: dt.date | None = None,
        thru_date: dt.date | None = None,
    ) -> None:
        """Initialize AxysData and load/validate all requested data.

        Args:
            portperf_path: Path to portperf CSV.
            secperf_path: Path to secperf CSV.
            portfolio_code: Optional portfolio filter.
            from_date: Optional lower date bound. Keeps rows where
                from_date <= FROM_DATE.
            thru_date: Optional upper date bound. Keeps rows where
                THRU_DATE <= thru_date.

        Raises:
            PpaError: If any validation fails or if file/schema validation fails.
        """
        self.portperf_path: str = portperf_path
        self.secperf_path: str = secperf_path
        self.portfolio_code: str | None = portfolio_code
        self.from_date: dt.date | None = from_date
        self.thru_date: dt.date | None = thru_date

        self.portperf: pl.DataFrame = self._scan_csv_selected_columns(
            self.portperf_path,
            _PORTPERF_REQUIRED_COLUMNS,
        )

        self.secperf: pl.DataFrame = self._scan_csv_selected_columns(
            self.secperf_path,
            _SECPERF_REQUIRED_COLUMNS,
        )

        self._validate_secperf_uniqueness(self.secperf)
        self._validate_portperf_row_date_order(self.portperf)
        self._validate_portperf_continuous_periods(self.portperf)
        self._validate_secperf_periods_match_portperf(self.portperf, self.secperf)

        self.derived_secperf: pl.DataFrame = self._derive_secperf_for_all_periods()

    def _derive_reconciled_weights(
        self,
        secperf_df: pl.DataFrame,
        portfolio_return: float,
        *,
        return_epsilon: float = 1e-12,
        match_tolerance: float = 1e-12,
        near_zero_weight: float = 1e-18,
    ) -> pl.DataFrame:
        """Derive single-period security weights with fallbacks.

        This function tries to return a usable result even when the input has weak implied
        weights, null-like numeric values, or a target portfolio return that is awkward for
        the preferred closed-form tilt.

        The function assumes all rows belong to a single portfolio and a single period.

        Fallback order for the anchor weight on each row:
            1. contribution / return, when numerically safe and nonnegative
            2. BEGIN_WEIGHT, when nonnegative
            3. BEGIN_MV-based normalized weight, when nonnegative
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
                SECURITY_ID, SEC_RETURN, CONTRIBUTION_W_X_R, BEGIN_WEIGHT, BEGIN_MV.
            portfolio_return: Single-period portfolio return from portperf.
            return_epsilon: Threshold below which SEC_RETURN is treated as zero when computing
                contribution / return implied weights.
            match_tolerance: Floating-point tolerance for reconciliation checks.
            near_zero_weight: Tiny threshold used when cleaning near-zero adjusted weights.

        Returns:
            A new DataFrame containing the original columns plus:
                DERIVED_WEIGHT_RAW
                ANCHOR_WEIGHT
                ADJUSTED_WEIGHT
                ADJUSTED_CONTRIBUTION_W_X_R
                ANCHOR_SOURCE
                TARGET_PORT_RETURN
                EFFECTIVE_TARGET_PORT_RETURN
                RECON_METHOD
                RECON_NOTES

        Raises:
            PpaError: Required columns are missing, the DataFrame is empty, or parameter values are
                invalid.
        """
        if secperf_df.height == 0:
            raise errs.PpaError(self._error_message("secperf_df must contain at least one row."))

        working_df = (
            secperf_df.select(list(_SECPERF_WEIGHT_RETURN_COLUMNS))
            .with_row_index(name="_ROW_IDX")
            .with_columns(
                pl.col("SEC_RETURN").cast(pl.Float64, strict=False).fill_null(0.0),
                pl.col("CONTRIBUTION_W_X_R").cast(pl.Float64, strict=False).fill_null(0.0),
                pl.col("BEGIN_WEIGHT").cast(pl.Float64, strict=False).fill_null(float("nan")),
                pl.col("BEGIN_MV").cast(pl.Float64, strict=False).fill_null(float("nan")),
            )
        )

        security_ids = [str(value) for value in working_df["SECURITY_ID"].to_list()]
        returns = [float(value) for value in working_df["SEC_RETURN"].to_list()]
        contributions = [float(value) for value in working_df["CONTRIBUTION_W_X_R"].to_list()]
        begin_weights = [float(value) for value in working_df["BEGIN_WEIGHT"].to_list()]
        begin_market_values = [float(value) for value in working_df["BEGIN_MV"].to_list()]

        # Clean all numeric vectors into finite values so later logic can stay simple and typed.
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
            if abs(sec_return) <= return_epsilon:
                derived_weight_raw.append(None)
                continue

            implied_weight = contribution / sec_return
            if math.isfinite(implied_weight) and implied_weight >= 0.0:
                derived_weight_raw.append(implied_weight)
            else:
                derived_weight_raw.append(None)

        # Build BEGIN_MV-based fallback weights. Invalid or negative market values are treated as 0.
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
        anchor_sources: list[str] = []

        for implied_weight, begin_weight, mv_weight in zip(
            derived_weight_raw,
            begin_weights,
            mv_weights,
        ):
            if implied_weight is not None:
                anchor_weights.append(implied_weight)
                anchor_sources.append("CONTRIBUTION_DIV_RETURN")
                continue

            if math.isfinite(begin_weight) and begin_weight >= 0.0:
                anchor_weights.append(begin_weight)
                anchor_sources.append("BEGIN_WEIGHT")
                continue

            if mv_weight is not None and math.isfinite(mv_weight) and mv_weight >= 0.0:
                anchor_weights.append(mv_weight)
                anchor_sources.append("BEGIN_MV")
                continue

            anchor_weights.append(1.0)
            anchor_sources.append("EQUAL_WEIGHT_FALLBACK")

        # If every anchor weight is zero, fall back to equal weights for all rows.
        anchor_total = sum(anchor_weights)
        if anchor_total <= 0.0 or not math.isfinite(anchor_total):
            anchor_weights = [1.0] * len(anchor_weights)
            anchor_sources = ["EQUAL_WEIGHT_FALLBACK_ALL"] * len(anchor_sources)
            anchor_total = float(len(anchor_weights))

        anchor_weights = [max(0.0, weight) / anchor_total for weight in anchor_weights]

        min_return = min(returns)
        max_return = max(returns)
        effective_target = min(max(portfolio_return, min_return), max_return)

        notes: list[str] = []
        if abs(effective_target - portfolio_return) > match_tolerance:
            notes.append("TARGET_CLIPPED_TO_FEASIBLE_RANGE")

        adjusted_weights, recon_method, recon_notes = AxysData._solve_adjusted_weights(
            security_ids=security_ids,
            anchor_weights=anchor_weights,
            returns=returns,
            target_return=effective_target,
            match_tolerance=match_tolerance,
            near_zero_weight=near_zero_weight,
        )
        notes.extend(recon_notes)

        adjusted_contributions = [
            weight * sec_return for weight, sec_return in zip(adjusted_weights, returns)
        ]
        note_text = ";".join(notes) if notes else "OK"
        if note_text != "OK":
            print(note_text)  # TODO: NO!!

        derived_df = pl.DataFrame(
            {
                "_ROW_IDX": working_df["_ROW_IDX"].to_list(),
                "DERIVED_WEIGHT_RAW": derived_weight_raw,
                "ANCHOR_WEIGHT": anchor_weights,
                "ADJUSTED_WEIGHT": adjusted_weights,
                "ADJUSTED_CONTRIBUTION_W_X_R": adjusted_contributions,
                "ANCHOR_SOURCE": anchor_sources,
                "TARGET_PORT_RETURN": [portfolio_return] * len(anchor_weights),
                "EFFECTIVE_TARGET_PORT_RETURN": [effective_target] * len(anchor_weights),
                "RECON_METHOD": [recon_method] * len(anchor_weights),
                "RECON_NOTES": [note_text] * len(anchor_weights),
            }
        )

        return (
            secperf_df.with_row_index(name="_ROW_IDX")
            .join(derived_df, on="_ROW_IDX", how="left")
            .drop("_ROW_IDX")
        )

    def _derive_secperf_for_all_periods(self) -> pl.DataFrame:
        """Derive reconciled secperf weights for all periods in this instance.

        Key optimization:
            - Groups secperf once by (FROM_DATE, THRU_DATE)
            - Avoids repeated filtering inside the loop

        Assumptions:
            - portperf periods are unique
            - secperf rows map cleanly to portperf periods

        Returns:
            secperf_derived with same row count as secperf.

        Raises:
            PpaError on validation failures.
        """
        # --- Validate unique portperf periods ---
        dup_periods = (
            self.portperf.group_by(["FROM_DATE", "THRU_DATE"]).len().filter(pl.col("len") > 1)
        )
        if dup_periods.height > 0:
            raise errs.PpaError(
                self._error_message(
                    f"{errs.ERROR_999_UNEXPECTED}Duplicate portperf periods: "
                    f"{dup_periods.head(10).to_dicts()}"
                )
            )

        # --- Pre-group secperf ONCE (big win) ---
        secperf_groups = self.secperf.partition_by(["FROM_DATE", "THRU_DATE"], as_dict=True)

        # Build lookup: (from_date, thru_date) -> DataFrame
        # Keys from Polars are tuples already
        # Example key: (date1, date2)
        secperf_lookup = secperf_groups

        # --- Process each period ---
        derived_frames: list[pl.DataFrame] = []

        for row in self.portperf.iter_rows(named=True):
            key = (row["FROM_DATE"], row["THRU_DATE"])
            port_return = float(row["PORT_RETURN"])

            secperf_period = secperf_lookup.get(key)

            if secperf_period is None or secperf_period.height == 0:
                raise errs.PpaError(
                    self._error_message(
                        f"{errs.ERROR_999_UNEXPECTED}No secperf rows for period {key}"
                    )
                )

            derived = self._derive_reconciled_weights(secperf_period, port_return)

            # If the derived weights do not give sumof(wgt * ret) =~ portperf.ret, then fail.
            if not self._is_reconciled(derived, tolerance=0.00001):
                raise errs.PpaError(
                    self._error_message(
                        f"{errs.ERROR_503_COULD_NOT_DERIVE_WEIGHTS}Period {key}  |  "
                        f"Failed with {derived['RECON_METHOD'][0]} and {derived['RECON_NOTES'][0]}"
                    )
                )

            derived_frames.append(derived)

        if not derived_frames:
            return self.secperf.head(0)

        secperf_derived = pl.concat(derived_frames, how="vertical", rechunk=True)

        # --- Final defensive check ---
        if secperf_derived.height != self.secperf.height:
            raise errs.PpaError(
                self._error_message(
                    f"{errs.ERROR_999_UNEXPECTED}Row count mismatch: "
                    f"derived={secperf_derived.height}, input={self.secperf.height}"
                )
            )

        return secperf_derived

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

    def _is_reconciled(
        self,
        result_df: pl.DataFrame,
        tolerance: float,
        *,
        weight_col: str = "ADJUSTED_WEIGHT",
        return_col: str = "SEC_RETURN",
        target_col: str = "TARGET_PORT_RETURN",
    ) -> bool:
        """Check if sum(weight * return) matches the target portfolio return.

        This is a minimal, definitive reconciliation check for the output of
        _derive_reconciled_weights(). It computes:

            sum(weight * return) - target_port_return

        and verifies that the absolute difference is within the provided tolerance.

        Args:
            result_df: Polars DataFrame returned by the reconciliation function.
                Must contain at least the weight, return, and target columns.
            tolerance: Absolute tolerance for the reconciliation check.
            weight_col: Column name for weights (default: "ADJUSTED_WEIGHT").
            return_col: Column name for returns (default: "SEC_RETURN").
            target_col: Column name for target portfolio return
                (default: "TARGET_PORT_RETURN").

        Returns:
            True if the reconciliation condition holds within tolerance, False otherwise.

        Raises:
            ValueErrorPpaError: If required columns are missing or the DataFrame is empty.
        """
        # --- Validate inputs (fail fast, defensive) ---
        if result_df.height == 0:
            raise errs.PpaError(self._error_message("result_df must contain at least one row."))

        # --- Compute left-hand side: sum(weight * return) ---
        # Use Polars vectorized multiplication and sum for numerical stability and speed.
        lhs: float = float((result_df[weight_col] * result_df[return_col]).sum())

        # --- Extract right-hand side: portfolio return ---
        # Assumes target is constant across all rows (true for single-period extract).
        rhs: float = float(result_df[target_col][0])

        # --- Perform absolute tolerance check ---
        # This is the definitive reconciliation condition.
        return abs(lhs - rhs) <= tolerance

    def _scan_csv_selected_columns(
        self,
        path: str,
        requested_columns: Sequence[str],
    ) -> pl.DataFrame:
        """Read a CSV lazily, project only requested columns, and apply instance filters.

        Args:
            path: CSV file path.
            requested_columns: Exact column names to read.

        Returns:
            A collected Polars DataFrame containing only the requested columns.

        Raises:
            PpaError: If one or more requested columns are missing, or if no valid columns remain.
        """
        # Assert that the data file path exists.
        if not util.file_path_exists(path):
            raise errs.PpaError(self._error_message(util.file_path_error(path)))

        # Read only the header first so we can fail fast on schema drift.
        header_df: pl.DataFrame = pl.read_csv(path, n_rows=0)
        available_columns: set[str] = set(header_df.columns)

        missing_columns: list[str] = [
            column_name
            for column_name in requested_columns
            if column_name not in available_columns
        ]
        if missing_columns:
            raise errs.PpaError(
                self._error_message(
                    f"{errs.ERROR_502_MISSING_REQUIRED_COLUMNS}Missing {missing_columns} "
                    f"in {path!r}.  |  Columns available are: {sorted(available_columns)}"
                )
            )

        # Build a lazy scan. We project only the required columns first, then cast the
        # date fields explicitly so downstream comparisons and validations are type-safe.
        lazy_frame: pl.LazyFrame = (
            pl.scan_csv(path)
            .select(list(requested_columns))
            .with_columns(
                pl.col("FROM_DATE").str.strptime(pl.Date, "%Y-%m-%d", strict=True),
                pl.col("THRU_DATE").str.strptime(pl.Date, "%Y-%m-%d", strict=True),
            )
        )

        if self.portfolio_code is not None:
            lazy_frame = lazy_frame.filter(pl.col("PORTFOLIO_CODE") == self.portfolio_code)

        if self.from_date is not None:
            lazy_frame = lazy_frame.filter(pl.lit(self.from_date) <= pl.col("FROM_DATE"))

        if self.thru_date is not None:
            lazy_frame = lazy_frame.filter(pl.col("THRU_DATE") <= pl.lit(self.thru_date))

        return lazy_frame.collect()

    @staticmethod
    def _solve_adjusted_weights(
        security_ids: list[str],
        anchor_weights: list[float],
        returns: list[float],
        target_return: float,
        match_tolerance: float,
        near_zero_weight: float,
    ) -> tuple[list[float], str, list[str]]:
        """Solve for adjusted weights using fallback methods.

        Args:
            security_ids: Security identifiers aligned to the weight and return vectors.
            anchor_weights: Nonnegative weights summing to 1.
            returns: Single-period security returns.
            target_return: Desired weighted average return.
            match_tolerance: Floating-point match tolerance.
            near_zero_weight: Tiny threshold used when cleaning weights.

        Returns:
            Tuple containing:
                adjusted_weights
                method_name
                notes
        """
        notes: list[str] = []

        anchor_return = AxysData._weighted_return(anchor_weights, returns)
        if abs(anchor_return - target_return) <= match_tolerance:
            return anchor_weights, "ANCHOR_NO_CHANGE", notes

        closed_form_weights = AxysData._solve_closed_form_tilt(
            anchor_weights=anchor_weights,
            returns=returns,
            target_return=target_return,
            near_zero_weight=near_zero_weight,
        )
        if closed_form_weights is not None:
            closed_form_return = AxysData._weighted_return(closed_form_weights, returns)
            if abs(closed_form_return - target_return) <= 10.0 * match_tolerance:
                return closed_form_weights, "CLOSED_FORM_TILT", notes

        notes.append("CLOSED_FORM_UNAVAILABLE_OR_REJECTED")

        bisection_weights = AxysData._solve_bisection_tilt(
            anchor_weights=anchor_weights,
            returns=returns,
            target_return=target_return,
            match_tolerance=match_tolerance,
            near_zero_weight=near_zero_weight,
        )
        if bisection_weights is not None:
            bisection_return = AxysData._weighted_return(bisection_weights, returns)
            if abs(bisection_return - target_return) <= 10.0 * match_tolerance:
                return bisection_weights, "BISECTION_TILT", notes

        notes.append("BISECTION_UNAVAILABLE_OR_REJECTED")

        # Exact long-only fallback: use at most two securities whose returns bracket the target.
        two_security_weights = AxysData._solve_two_security_fallback(
            security_ids=security_ids,
            anchor_weights=anchor_weights,
            returns=returns,
            target_return=target_return,
            match_tolerance=match_tolerance,
        )
        if two_security_weights is not None:
            return two_security_weights, "TWO_SECURITY_EXACT_FALLBACK", notes

        notes.append("TWO_SECURITY_FALLBACK_UNAVAILABLE")

        # Final fallback. This may not hit the target exactly, but it always returns a
        # valid normalized nonnegative weight vector.
        notes.append("FELL_BACK_TO_ANCHOR_WEIGHTS")
        return anchor_weights, "ANCHOR_FALLBACK", notes

    @staticmethod
    def _solve_bisection_tilt(
        anchor_weights: list[float],
        returns: list[float],
        target_return: float,
        match_tolerance: float,
        near_zero_weight: float,
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
                near_zero_weight=near_zero_weight,
            )
            if weights is None:
                continue

            residual = AxysData._weighted_return(weights, returns) - target_return
            if math.isfinite(residual):
                valid_points.append((lambda_value, residual))

        if not valid_points:
            return None

        for lambda_value, residual in valid_points:
            if abs(residual) <= match_tolerance:
                return AxysData._weights_from_lambda(
                    anchor_weights=anchor_weights,
                    returns=returns,
                    lambda_value=lambda_value,
                    near_zero_weight=near_zero_weight,
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
                    near_zero_weight=near_zero_weight,
                )
                if mid_weights is None:
                    return None

                mid_residual = AxysData._weighted_return(mid_weights, returns) - target_return
                if abs(mid_residual) <= match_tolerance:
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
                near_zero_weight=near_zero_weight,
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
        security_ids: list[str],
        anchor_weights: list[float],
        returns: list[float],
        target_return: float,
        match_tolerance: float,
    ) -> list[float] | None:
        """Construct an exact long-only solution using one or two securities.

        This is intentionally a last-resort fallback. It is exact whenever the target lies
        within the feasible range of security returns, but it can move far away from the
        anchor weights. To keep it less arbitrary, it chooses the bracketing pair with the
        largest combined anchor weight.

        Args:
            security_ids: Security identifiers aligned to the inputs.
            anchor_weights: Nonnegative anchor weights summing to 1.
            returns: Security returns.
            target_return: Desired weighted average return.
            match_tolerance: Floating-point tolerance.

        Returns:
            Exact nonnegative normalized weights if found, otherwise None.
        """
        del security_ids  # The ids are accepted for symmetry and future debugging hooks.

        # Single-security exact match.
        for row_index, sec_return in enumerate(returns):
            if abs(sec_return - target_return) <= match_tolerance:
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
                if target_return < low_return - match_tolerance:
                    continue
                if target_return > high_return + match_tolerance:
                    continue
                if abs(left_return - right_return) <= match_tolerance:
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

        if abs(right_return - left_return) <= match_tolerance:
            return None

        right_weight = (target_return - left_return) / (right_return - left_return)
        left_weight = 1.0 - right_weight

        if left_weight < -match_tolerance or right_weight < -match_tolerance:
            return None

        weights = [0.0] * len(returns)
        weights[left_index] = max(0.0, left_weight)
        weights[right_index] = max(0.0, right_weight)

        total_weight = sum(weights)
        if total_weight <= 0.0:
            return None

        return [weight / total_weight for weight in weights]

    def _validate_portperf_continuous_periods(self, portperf_df: pl.DataFrame) -> None:
        """Validate that portperf periods form one continuous, gap-free, overlap-free range."""
        if portperf_df.height == 0:
            return

        period_df: pl.DataFrame = (
            portperf_df.select(["PORTFOLIO_CODE", "FROM_DATE", "THRU_DATE"])
            .unique()
            .sort(["PORTFOLIO_CODE", "FROM_DATE", "THRU_DATE"])
        )

        # Re-check uniqueness at the period level. This was not explicitly requested, but
        # it is cheap and prevents ambiguous continuity results.
        duplicate_periods: pl.DataFrame = (
            period_df.group_by(["PORTFOLIO_CODE", "FROM_DATE", "THRU_DATE"])
            .len()
            .filter(pl.col("len") > 1)
        )
        if duplicate_periods.height > 0:
            sample_rows: list[dict[str, object]] = duplicate_periods.head(10).to_dicts()
            raise errs.PpaError(
                self._error_message(
                    f"{errs.ERROR_999_UNEXPECTED}portperf period validation failed. "
                    f"Duplicate periods found. Sample duplicates: {sample_rows}"
                )
            )

        continuity_check_df: pl.DataFrame = (
            period_df.with_columns(
                pl.col("FROM_DATE").shift(-1).over("PORTFOLIO_CODE").alias("NEXT_FROM_DATE"),
                pl.col("THRU_DATE").shift(-1).over("PORTFOLIO_CODE").alias("NEXT_THRU_DATE"),
            )
            .with_columns(
                (pl.col("THRU_DATE") + dt.timedelta(days=1)).alias("EXPECTED_NEXT_FROM_DATE")
            )
            .filter(
                pl.col("NEXT_FROM_DATE").is_not_null()
                & (pl.col("NEXT_FROM_DATE") != pl.col("EXPECTED_NEXT_FROM_DATE"))
            )
            .select(
                [
                    "PORTFOLIO_CODE",
                    "FROM_DATE",
                    "THRU_DATE",
                    "NEXT_FROM_DATE",
                    "NEXT_THRU_DATE",
                    "EXPECTED_NEXT_FROM_DATE",
                ]
            )
            .sort(["PORTFOLIO_CODE", "FROM_DATE", "THRU_DATE"])
        )

        if continuity_check_df.height > 0:
            sample_rows: list[dict[str, object]] = continuity_check_df.head(10).to_dicts()
            errs.raise_unexpected(
                "portperf continuity validation failed.  The periods are not continuous; at least "
                f"one gap or overlap exists.  Sample discontinuities: {sample_rows}"
            )

    @staticmethod
    def _validate_portperf_row_date_order(portperf_df: pl.DataFrame) -> None:
        """Validate that each portperf row has FROM_DATE <= THRU_DATE."""
        invalid_rows: pl.DataFrame = portperf_df.filter(
            pl.col("FROM_DATE") > pl.col("THRU_DATE")
        ).sort(["PORTFOLIO_CODE", "FROM_DATE", "THRU_DATE"])

        if invalid_rows.height > 0:
            sample_rows: list[dict[str, object]] = invalid_rows.head(10).to_dicts()
            errs.raise_unexpected(
                "portperf date-order validation failed.  Found rows where FROM_DATE > THRU_DATE. "
                f"Sample invalid rows: {sample_rows}"
            )

    @staticmethod
    def _validate_secperf_periods_match_portperf(
        portperf_df: pl.DataFrame,
        secperf_df: pl.DataFrame,
    ) -> None:
        """Validate that secperf has exactly the same distinct periods as portperf."""
        portperf_periods: pl.DataFrame = (
            portperf_df.select(["PORTFOLIO_CODE", "FROM_DATE", "THRU_DATE"])
            .unique()
            .sort(["PORTFOLIO_CODE", "FROM_DATE", "THRU_DATE"])
        )

        secperf_periods: pl.DataFrame = (
            secperf_df.select(["PORTFOLIO_CODE", "FROM_DATE", "THRU_DATE"])
            .unique()
            .sort(["PORTFOLIO_CODE", "FROM_DATE", "THRU_DATE"])
        )

        missing_in_secperf: pl.DataFrame = portperf_periods.join(
            secperf_periods,
            on=["PORTFOLIO_CODE", "FROM_DATE", "THRU_DATE"],
            how="anti",
        ).sort(["PORTFOLIO_CODE", "FROM_DATE", "THRU_DATE"])

        extra_in_secperf: pl.DataFrame = secperf_periods.join(
            portperf_periods,
            on=["PORTFOLIO_CODE", "FROM_DATE", "THRU_DATE"],
            how="anti",
        ).sort(["PORTFOLIO_CODE", "FROM_DATE", "THRU_DATE"])

        if missing_in_secperf.height > 0 or extra_in_secperf.height > 0:
            missing_sample: list[dict[str, object]] = missing_in_secperf.head(10).to_dicts()
            extra_sample: list[dict[str, object]] = extra_in_secperf.head(10).to_dicts()
            errs.raise_unexpected(
                "secperf/portperf period validation failed. "
                "secperf must have exactly the same distinct periods as portperf. "
                f"Missing periods in secperf: {missing_sample}. "
                f"Extra periods in secperf: {extra_sample}"
            )

    @staticmethod
    def _validate_secperf_uniqueness(secperf_df: pl.DataFrame) -> None:
        """Validate secperf uniqueness at portfolio/period/security grain."""
        duplicate_rows: pl.DataFrame = (
            secperf_df.group_by(["PORTFOLIO_CODE", "FROM_DATE", "THRU_DATE", "SECURITY_ID"])
            .len()
            .filter(pl.col("len") > 1)
            .sort(["PORTFOLIO_CODE", "FROM_DATE", "THRU_DATE", "SECURITY_ID"])
        )

        if duplicate_rows.height > 0:
            sample_rows: list[dict[str, object]] = duplicate_rows.head(10).to_dicts()
            errs.raise_unexpected(
                "secperf uniqueness validation failed. "
                "PORTFOLIO_CODE, FROM_DATE, THRU_DATE, and SECURITY_ID do not uniquely "
                f"identify each row. Sample duplicates: {sample_rows}"
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
