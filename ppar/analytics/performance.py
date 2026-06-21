"""Represent and validate narrow periodic performance data.

This module contains the ``Performance`` class, which loads portfolio,
benchmark, or classification-level performance rows, validates dates and
values, and derives contributions, total returns, overall returns, and
linking coefficients.
"""

# Python Imports
import datetime as dt
from pathlib import Path
from typing import cast, Sequence

# Third-Party Imports
import pandas as pd
import polars as pl

# Project Imports
import ppar.schema as cols
from ppar.errors import PpaError
import ppar.utilities as util


class Performance:
    """Hold narrow identifier-level returns and weights for one performance stream.

    Attributes:
        classification_name: Optional name of the classification represented
            by the performance data.
        classification_items: Optional classification identifier/name pairs
            extracted from input data when a ``name`` column is present.
        df: Alias for ``narrow_df`` retained for callers that access calculated
            performance rows directly.
        error_message_context: Context string included in validation errors.
        identifiers: Sorted identifiers present in the performance rows.
        name: Optional descriptive name for the performance stream.
        narrow_df: Calculated long-form Polars DataFrame containing dates,
            identifier, return, weight, contribution, quantity of days, and
            total return.
        subperiods_have_been_consolidated: Indicates whether lower-frequency
            periods have been consolidated into larger reporting periods.
    """

    def __init__(
        self,
        data_source: util.PerformanceDataSource,
        name: str | None = None,
        classification_name: str | None = None,
        from_date: str | dt.date = dt.date.min,
        thru_date: str | dt.date = dt.date.max,
    ):
        """Initialize a ``Performance`` instance from narrow performance rows.

        Args:
            data_source: Performance data source. This can be a CSV file path,
                a pandas DataFrame, or a Polars DataFrame.
            name: Descriptive name for the performance stream. If omitted for
                a CSV input, the file basename is used.
            classification_name: Name of the represented classification, such
                as ``"Security"`` or ``"Economic Sector"``.
            from_date: Earliest from date to keep.
            thru_date: Latest thru date to keep.

        Data Parameters:
            Input must use one row per period and identifier with these columns::

                from_date, thru_date, identifier, return, weight, name
                2024-01-01, 2024-01-31, AAPL, -0.0422272121, 0.4, Apple Inc.
                2024-01-01, 2024-01-31, MSFT,  0.0572811503, 0.6, Microsoft

            The ``name`` column is optional. For each period, weights must sum
            to ``1.0``.

        Raises:
            PpaError: If input rows cannot be loaded or converted, required
                columns are absent, values are missing, rows or periods are
                duplicated, dates are invalid or overlapping, or period
                weights do not sum to ``1.0``.
        """
        name = util.normalize_optional_string(name)
        self.classification_name = util.normalize_optional_string(classification_name)
        from_date = util.convert_to_date(from_date)
        thru_date = util.convert_to_date(thru_date)
        self.subperiods_have_been_consolidated = False
        self.error_message_context = (
            f"in the file {data_source}"
            if isinstance(data_source, str | Path)
            else f"in the dataframe {name}"
        )
        if from_date > thru_date:
            raise PpaError(
                f"{self.error_message_context}: "
                f"From date {from_date} is after thru date {thru_date}.",
                111,
            )

        self.name, self.narrow_df = self._load_data(
            name, data_source, from_date, thru_date
        )
        if self.narrow_df.is_empty():
            raise PpaError(self.error_message_context, 103)
        self._clean_and_validate_columns()
        self._cast_and_validate_columns()
        self._set_classification_items()
        self._clean_and_validate_dates()
        self._calculate_rows()
        self.identifiers = sorted(self.narrow_df[cols.IDENTIFIER].unique().to_list())
        self._df_overall = pl.DataFrame()
        self.df = self.narrow_df

    def audit(self) -> None:
        """Validate internal consistency of this performance stream.

        Raises:
            PpaError: If weights do not sum to ``1.0``, raw contributions do
                not equal weight multiplied by return, or contributions do not
                sum to the period total return.
        """
        period_totals = self.period_totals()
        summed_weights = self.narrow_df.group_by(cols.DATE_COLUMNS).agg(
            pl.col(cols.WEIGHT).sum().alias(cols.WEIGHT)
        )
        if not (summed_weights[cols.WEIGHT].round(8) == 1.0).all():
            raise PpaError(
                f"{self.error_message_context}: Perf.audit() weights do not sum to 1.0.", 999
            )

        if not self.subperiods_have_been_consolidated:
            contributions = self.narrow_df.with_columns(
                (pl.col(cols.WEIGHT) * pl.col(cols.RETURN)).alias("_expected_contribution")
            )
            if not (
                contributions[cols.CONTRIBUTION].round(11)
                == contributions["_expected_contribution"].round(11)
            ).all():
                raise PpaError(
                    f"{self.error_message_context}: Perf.audit() weight * return != contrib.", 999
                )

        summed_contributions = (
            self.narrow_df.group_by(cols.DATE_COLUMNS)
            .agg(pl.col(cols.CONTRIBUTION).sum().alias(cols.CONTRIBUTION))
            .join(period_totals, on=cols.DATE_COLUMNS)
        )
        if not (
            summed_contributions[cols.CONTRIBUTION].round(11)
            == summed_contributions[cols.TOTAL_RETURN].round(11)
        ).all():
            raise PpaError(
                f"{self.error_message_context}: Perf.audit() sum of contribs != total return.",
                999,
            )

    @staticmethod
    def audit_performances(
        performances: Sequence["Performance"],
        expected_from_date: dt.date,
        expected_thru_date: dt.date,
        common_classification_name: str | None = None,
    ) -> None:
        """Validate a portfolio/benchmark performance pair.

        Args:
            performances: Portfolio and benchmark performance streams.
            expected_from_date: Expected first from date.
            expected_thru_date: Expected final thru date.
            common_classification_name: Optional classification name expected
                on both streams.

        Raises:
            PpaError: If either stream fails its audit, dates or day counts
                differ, the date range differs from the expected range, or a
                required classification does not match.
        """
        common_classification_name = util.normalize_optional_string(common_classification_name)
        portfolio, benchmark = performances
        portfolio.audit()
        benchmark.audit()
        dates_days = [*cols.DATE_COLUMNS, cols.QUANTITY_OF_DAYS]
        portfolio_periods = portfolio.period_totals().select(dates_days)
        benchmark_periods = benchmark.period_totals().select(dates_days)
        if not portfolio_periods.equals(benchmark_periods):
            raise PpaError("audit_perfs(): Portfolio and Benchmark dates are not equal.", 999)
        if not (
            portfolio_periods[cols.FROM_DATE][0] == expected_from_date
            and portfolio_periods[cols.THRU_DATE][-1] == expected_thru_date
        ):
            raise PpaError("audit_perfs(): Date logic error.", 999)
        if common_classification_name is not None:
            if portfolio.classification_name != benchmark.classification_name:
                raise PpaError("audit_perfs(): Common classification name error.", 999)

    def _calculate_df_overall(self) -> pl.DataFrame:
        """Calculate overall narrow rows for the full performance period.

        Returns:
            One overall-period row per identifier, including linked returns,
            day-weighted weights, summed contributions, and common total return.
        """
        overall_from_date = cast(dt.date, self.narrow_df[cols.FROM_DATE].min())
        overall_thru_date = cast(dt.date, self.narrow_df[cols.THRU_DATE].max())
        total_days = (overall_thru_date - overall_from_date).days + 1
        coefficient = (
            pl.lit(1.0) if total_days == 0 else pl.col(cols.QUANTITY_OF_DAYS) / total_days
        )
        overall_total_return = cast(
            float, (self.period_totals()[cols.TOTAL_RETURN] + 1).product() - 1
        )
        return (
            self.narrow_df.group_by(cols.IDENTIFIER)
            .agg(
                pl.col(cols.RETURN).add(1).product().sub(1).alias(cols.RETURN),
                (pl.col(cols.WEIGHT) * coefficient).sum().alias(cols.WEIGHT),
                pl.col(cols.CONTRIBUTION).sum().alias(cols.CONTRIBUTION),
            )
            .with_columns(
                pl.lit(overall_from_date).alias(cols.FROM_DATE),
                pl.lit(overall_thru_date).alias(cols.THRU_DATE),
                pl.lit(total_days).alias(cols.QUANTITY_OF_DAYS),
                pl.lit(overall_total_return).alias(cols.TOTAL_RETURN),
            )
            .select(
                *cols.DATE_COLUMNS,
                cols.QUANTITY_OF_DAYS,
                cols.TOTAL_RETURN,
                cols.IDENTIFIER,
                cols.RETURN,
                cols.WEIGHT,
                cols.CONTRIBUTION,
            )
            .sort(cols.IDENTIFIER)
        )

    def _calculate_rows(self) -> None:
        """Add calculated contribution, elapsed-day, and total-return columns."""
        self.narrow_df = (
            self.narrow_df.with_columns(
                (
                    (pl.col(cols.THRU_DATE) - pl.col(cols.FROM_DATE)).dt.total_days()
                    + 1
                ).alias(cols.QUANTITY_OF_DAYS),
                (pl.col(cols.WEIGHT) * pl.col(cols.RETURN)).alias(cols.CONTRIBUTION),
            )
            .join(
                self.narrow_df.with_columns(
                    (pl.col(cols.WEIGHT) * pl.col(cols.RETURN)).alias(cols.CONTRIBUTION)
                )
                .group_by(cols.DATE_COLUMNS)
                .agg(pl.col(cols.CONTRIBUTION).sum().alias(cols.TOTAL_RETURN)),
                on=cols.DATE_COLUMNS,
            )
            .select(
                *cols.DATE_COLUMNS,
                cols.QUANTITY_OF_DAYS,
                cols.TOTAL_RETURN,
                cols.IDENTIFIER,
                cols.RETURN,
                cols.WEIGHT,
                cols.CONTRIBUTION,
            )
            .sort([cols.THRU_DATE, cols.IDENTIFIER])
        )
        weights = self.narrow_df.group_by(cols.DATE_COLUMNS).agg(pl.col(cols.WEIGHT).sum())
        if not (weights[cols.WEIGHT].round(8) == 1.0).all():
            raise PpaError(self.error_message_context, 108)

    def _cast_and_validate_columns(self) -> None:
        """Cast required narrow columns and reject missing numeric values.

        Raises:
            PpaError: If a required value cannot be converted or is missing.
        """
        dtypes: dict[type[pl.Date] | type[pl.Float64] | type[pl.String], list[str]] = {
            pl.Date: cols.DATE_COLUMNS,
            pl.Float64: [cols.RETURN, cols.WEIGHT],
            pl.String: [cols.IDENTIFIER]
            + ([cols.NAME] if cols.NAME in self.narrow_df.columns else []),
        }
        for dtype, column_names in dtypes.items():
            for column_name in [
                name for name in column_names if self.narrow_df.schema[name] != dtype
            ]:
                try:
                    self.narrow_df = self.narrow_df.with_columns(pl.col(column_name).cast(dtype))
                except pl.exceptions.InvalidOperationError as exception:
                    raise PpaError(
                        f"{self.error_message_context}: Cannot convert the column "
                        f"'{column_name}' to a {dtype}, {str(exception)[:1000]}",
                        110,
                    ) from exception
        float_columns = dtypes[pl.Float64]
        if self.narrow_df.select(
            pl.any_horizontal(pl.all().is_null().any())
            | pl.any_horizontal(pl.col(float_columns).is_nan().any())
        ).item():
            raise PpaError(self.error_message_context, 104)

    def _clean_and_validate_columns(self) -> None:
        """Retain supported narrow input columns and validate required fields.

        Raises:
            PpaError: If any required narrow column is absent.
        """
        required_columns = [*cols.DATE_COLUMNS, cols.IDENTIFIER, cols.RETURN, cols.WEIGHT]
        if not all(column in self.narrow_df.columns for column in required_columns):
            raise PpaError(self.error_message_context, 109)
        optional_columns = [cols.NAME] if cols.NAME in self.narrow_df.columns else []
        self.narrow_df = self.narrow_df.select(*required_columns, *optional_columns)

    def _clean_and_validate_dates(self) -> None:
        """Sort and validate inclusive narrow period dates.

        Raises:
            PpaError: If rows are duplicated, period thru dates conflict,
                from dates are invalid, or periods overlap.
        """
        duplicate_rows = (
            self.narrow_df.group_by([*cols.DATE_COLUMNS, cols.IDENTIFIER])
            .len()
            .filter(pl.col("len") > 1)
        )
        if duplicate_rows.height > 0:
            sample_rows = (
                duplicate_rows.sort([*cols.DATE_COLUMNS, cols.IDENTIFIER]).head(10).to_dicts()
            )
            raise PpaError(f"{self.error_message_context}: {sample_rows}", 112)

        periods = self.narrow_df.select(cols.DATE_COLUMNS).unique().sort(cols.THRU_DATE)
        if periods[cols.THRU_DATE].n_unique() != periods.height:
            raise PpaError(self.error_message_context, 102)
        if (periods[cols.FROM_DATE] > periods[cols.THRU_DATE]).any():
            raise PpaError(self.error_message_context, 105)
        if periods.height > 1 and (
            periods[cols.FROM_DATE][1:] <= periods[cols.THRU_DATE][:-1]
        ).any():
            raise PpaError(self.error_message_context, 106)
        self.narrow_df = self.narrow_df.sort([cols.THRU_DATE, cols.IDENTIFIER])

    def _set_classification_items(self) -> None:
        """Capture identifier/name pairs supplied with narrow input data."""
        if cols.NAME not in self.narrow_df.columns:
            self.classification_items = pl.DataFrame()
            return
        self.classification_items = (
            self.narrow_df.unique(subset=[cols.IDENTIFIER], keep="last")
            .select(
                pl.col(cols.IDENTIFIER).alias(cols.CLASSIFICATION_IDENTIFIER),
                pl.col(cols.NAME).alias(cols.CLASSIFICATION_NAME),
            )
        )
        self.narrow_df = self.narrow_df.drop(cols.NAME)

    def period_totals(self) -> pl.DataFrame:
        """Return one summarized total-return row per reporting period.

        Returns:
            DataFrame containing dates, elapsed days, and total return.
        """
        return self.narrow_df.select(
            *cols.DATE_COLUMNS, cols.QUANTITY_OF_DAYS, cols.TOTAL_RETURN
        ).unique().sort(cols.THRU_DATE)

    def df_overall(self) -> pl.DataFrame:
        """Return cached overall-period narrow identifier rows."""
        if self._df_overall.is_empty():
            self._df_overall = self._calculate_df_overall()
        return self._df_overall

    def linking_coefficients(self) -> pl.Series:
        """Return logarithmic linking coefficients for each reporting period."""
        return util.logarithmic_linking_coefficients(
            self.overall_return(), self.period_totals()[cols.TOTAL_RETURN]
        )

    @staticmethod
    def _load_data(
        name: str | None,
        data_source: util.PerformanceDataSource,
        from_date: dt.date,
        thru_date: dt.date,
    ) -> tuple[str | None, pl.DataFrame]:
        """Load performance rows and apply the requested date bounds.

        Args:
            name: Optional descriptive performance name.
            data_source: CSV path, pandas DataFrame, or Polars DataFrame.
            from_date: Earliest from date to retain.
            thru_date: Latest thru date to retain.

        Returns:
            Resolved optional name and loaded DataFrame.

        Raises:
            PpaError: If a supplied file path does not exist.
        """
        if isinstance(data_source, str | Path):
            path = Path(data_source)
            if not util.file_path_exists(path):
                raise PpaError(util.file_path_error(path), None)
            if name is None:
                name = util.file_basename_without_extension(path)
            lazy_frame = pl.scan_csv(source=path, try_parse_dates=True)
        elif isinstance(data_source, pd.DataFrame):
            lazy_frame = pl.from_pandas(data_source).lazy()
        else:
            lazy_frame = data_source.lazy()
        if from_date != dt.date.min:
            lazy_frame = lazy_frame.filter(from_date <= pl.col(cols.THRU_DATE))
        if thru_date != dt.date.max:
            lazy_frame = lazy_frame.filter(pl.col(cols.THRU_DATE) <= thru_date)
        return name, lazy_frame.collect()

    def overall_return(self) -> float:
        """Return linked total return for the full reporting period."""
        return cast(float, (self.period_totals()[cols.TOTAL_RETURN] + 1).product() - 1)

    def reset_narrow_df(self, df: pl.DataFrame) -> None:
        """Replace calculated narrow rows and invalidate cached summaries.

        Args:
            df: Replacement calculated rows using the narrow performance
                schema.
        """
        self._df_overall = pl.DataFrame()
        self.narrow_df = df.sort([cols.THRU_DATE, cols.IDENTIFIER])
        self.identifiers = sorted(self.narrow_df[cols.IDENTIFIER].unique().to_list())
        self.df = self.narrow_df
