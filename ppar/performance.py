"""Represent and validate periodic performance data.

This module contains the ``Performance`` class, which loads portfolio,
benchmark, or classification-level performance data into a Polars DataFrame,
normalizes supported input layouts, validates dates and columns, and derives
contributions, total returns, overall returns, and linking coefficients.
"""

# Python Imports
import datetime as dt
from typing import cast

# Third-Party Imports
import pandas as pd
import polars as pl

# Project Imports
import ppar.columns as cols
from ppar.columns import CON, RET, WGT
from ppar.errors import PpaError
import ppar.utilities as util


class Performance:
    """Hold asset or classification weights and returns for one performance stream.

    A ``Performance`` instance represents one portfolio, benchmark, or mapped
    classification-level performance stream. Input data can be supplied in
    narrow or wide layout and is normalized to a wide Polars DataFrame with
    one row per period and paired ``.ret`` and ``.wgt`` columns for each
    identifier.

    Attributes:
        classification_name: Name of the classification represented by the
            performance data.
        classification_items: Optional classification identifier/name pairs
            extracted from narrow input data when a ``name`` column is present.
        df: Wide Polars DataFrame containing dates, returns, weights,
            contributions, quantity of days, and total return.
        error_message_context: Context string included in validation errors.
        identifiers: Identifier names derived from the wide return columns.
        name: Descriptive name for the performance stream.
        subperiods_have_been_consolidated: Indicates whether lower-frequency
            periods have been consolidated into larger reporting periods.
    """

    def __init__(
        self,
        data_source: util.PerformanceDataSource,
        name: str = util.EMPTY,
        classification_name: str = util.EMPTY,
        beginning_date: str | dt.date = dt.date.min,
        ending_date: str | dt.date = dt.date.max,
    ):
        """Initialize a ``Performance`` instance.

        Args:
            data_source: Performance data source. This can be a CSV file path,
                a pandas DataFrame, or a Polars DataFrame.
            name: Descriptive name for the performance stream. If omitted for
                a CSV input, the file basename is used.
            classification_name: Name of the classification represented by the
                input data, such as ``"Security"`` or ``"Economic Sector"``.
            beginning_date: Earliest beginning date to keep, either as a
                ``datetime.date`` or a ``yyyy-mm-dd`` string. Defaults to
                ``datetime.date.min``.
            ending_date: Latest ending date to keep, either as a
                ``datetime.date`` or a ``yyyy-mm-dd`` string. Defaults to
                ``datetime.date.max``.

        Data Parameters:
            Input data can be supplied in either narrow or wide layout. In both
            layouts, weights for each time period must sum to ``1.0`` and the
            sum of ``weight * return`` across identifiers must equal the total
            return for the period.

            Narrow layout columns::

                beginning_date, ending_date, identifier, return, weight, name
                2023-12-31, 2024-01-31, AAPL, -0.0422272121, 0.4, Apple Inc.
                2023-12-31, 2024-01-31, MSFT,  0.0572811503, 0.6, Microsoft

            Wide layout columns::

                beginning_date, ending_date, AAPL.ret, MSFT.ret, AAPL.wgt, MSFT.wgt
                2023-12-31, 2024-01-31, -0.0422272121, 0.0572811503, 0.4, 0.6

            The ``name`` column is optional in narrow input. Column order does
            not matter.

        Raises:
            PpaError: If dates cannot be converted, the beginning date is after
                the ending date, the input data source cannot be loaded, no
                rows remain after filtering, required return/weight columns are
                missing or inconsistent, values cannot be cast to required
                types, data contains null or NaN values, dates are duplicated,
                invalid, or discontinuous, narrow data has duplicate
                date/identifier rows, or period weights do not sum to ``1.0``.
        """
        # Convert the dates to dt.date types.
        beginning_date = util.convert_to_date(beginning_date)
        ending_date = util.convert_to_date(ending_date)

        # Set the classification_name
        self.classification_name = classification_name

        # Initialize self.subperiods_have_been_consolidated to False.  It might be set to True in
        # the Analytics class (e.g. if daily is consolidated into monthly)
        self.subperiods_have_been_consolidated = False

        # Set the error message for context.
        self.error_message_context = (
            f"in the file {data_source}"
            if isinstance(data_source, str)
            else f"in the dataframe {name}"
        )

        # Validate the dates.
        if beginning_date > ending_date:
            raise PpaError(
                f"{self.error_message_context}: "
                f"Beginning date {beginning_date} is after ending date {ending_date}.",
                111,
            )

        # Load the data.
        self.name, self.df = Performance._load_data(name, data_source, beginning_date, ending_date)

        # Convert self.df to "wide" format with multiple identifier.ret and identifier.wgt columns.
        # If cols.IDENTIFIER and cols.NAME are in self.df, then create self.classification_items,
        # which might be used later in the Attribution constructor when creating the Classification
        self.df, self.classification_items = self._convert_to_wide_format()

        # Assert that there is at least 1 row.
        if self.df.shape[0] == 0:
            raise PpaError(self.error_message_context, 103)

        # Remove extraneous columns, clean and validate columns.
        self._clean_and_validate_columns()

        # Establish self.identifiers
        self._column_names: dict[str, list[str]] = {}
        self.identifiers: list[str] = []
        self._reset_column_names()

        # Cast the columns to their correct data types and validate that there are not any missing
        # values.
        self._cast_and_validate_columns()

        # Clean and validate the dates.
        self._clean_and_validate_dates()

        # Add the QTY_DAYS, contributions and TOTAL_RETURN columns.
        self.df = (
            self.df.lazy()
            # Calculate QUANTITY_OF_DAYS (the quantity of days in each row).
            .with_columns(
                (pl.col(cols.ENDING_DATE) - pl.col(cols.BEGINNING_DATE))
                .dt.total_days()
                .alias(cols.QUANTITY_OF_DAYS)
            )
            # Calculate the contributions.
            .with_columns(
                [
                    (pl.col(wgt) * pl.col(ret)).alias(f"{wgt[:-4]}.con")
                    for wgt, ret in zip(self.col_names(WGT), self.col_names(RET))
                ]
            )
            # Horizontally sum the contribs for each row to get the total return.
            .with_columns(
                pl.sum_horizontal(self.col_names(CON)).alias(cols.TOTAL_RETURN)
            ).collect()
        )

        # Assert that the weights sum to 1.0.
        if not (self.df[self.col_names(WGT)].sum_horizontal().round(8) == 1.0).all():
            raise PpaError(self.error_message_context, 108)

        # self._df_overall is one row for the entire overall period.
        self._df_overall = pl.DataFrame()

    def audit(self) -> None:
        """Validate internal consistency of this performance stream.

        Raises:
            PpaError: If weights do not sum to ``1.0``, if contribution does
                not equal ``weight * return`` for unconsolidated data, or if
                total return does not equal the horizontal sum of
                contributions.
        """
        # Assert that the weights sum to 1.0
        if not (self.df[self.col_names(WGT)].sum_horizontal().round(8) == 1.0).all():
            raise PpaError(
                f"{self.error_message_context}: Perf.audit() weights do not sum to 1.0.", 999
            )

        # If not perf.subperiods_have_been_consolidated, then validate that weight * return
        # == contrib.  Note that this cannot be direcly checked in the Performance constructor
        # because the subperiods are not consolidated until the Analytics class.
        if not self.subperiods_have_been_consolidated:
            contribs = (self.df[self.col_names(RET)] * self.df[self.col_names(WGT)]).rename(
                lambda column_name: f"{column_name[:-4]}.con"
            )
            if not contribs.equals(self.df[self.col_names(CON)]):
                raise PpaError(
                    f"{self.error_message_context}: Perf.audit() weight * return != contrib.", 999
                )
            if not (
                self.df[cols.TOTAL_RETURN].round(11) == contribs.sum_horizontal().round(11)
            ).all():
                raise PpaError(
                    f"{self.error_message_context}: Perf.audit() sum of contribs != total return.",
                    999,
                )

    @staticmethod
    def audit_performances(
        performances: tuple["Performance", "Performance"],
        expected_beginning_date: dt.date,
        expected_ending_date: dt.date,
        common_classification_name: str = util.EMPTY,
    ) -> None:
        """Validate a portfolio/benchmark performance pair.

        Args:
            performances: Tuple containing the portfolio ``Performance`` at
                index ``0`` and the benchmark ``Performance`` at index ``1``.
            expected_beginning_date: Expected first beginning date for both
                performance streams.
            expected_ending_date: Expected final ending date for both
                performance streams.
            common_classification_name: Optional classification name that both
                performance streams are expected to share.

        Raises:
            PpaError: If either performance fails its own audit, the
                portfolio and benchmark dates or day counts differ, the actual
                date range does not match the expected date range, or a common
                classification name is required but the two performances do
                not share one.
        """
        # Set the portfolio and benchmark
        portfolio = performances[0]
        benchmark = performances[1]

        # Audit each Performance separately.
        portfolio.audit()
        benchmark.audit()

        # Assert that the portfolio and benchmark have the same dates and days.
        dates_days = (cols.BEGINNING_DATE, cols.ENDING_DATE, cols.QUANTITY_OF_DAYS)
        if not portfolio.df[dates_days].equals(benchmark.df[dates_days]):
            raise PpaError("audit_perfs(): Portfolio and Benchmark dates are not equal.", 999)

        # Assert that the portfolio/benchmark dates are equal to the expected dates.
        if not (
            portfolio.df[cols.BEGINNING_DATE][0] == expected_beginning_date
            and portfolio.df[cols.ENDING_DATE][-1] == expected_ending_date
        ):
            raise PpaError("audit_perfs(): Date logic error.", 999)

        # Assert that the portfolio and benchmark both have the same common_classification_name.
        if not util.is_empty(common_classification_name):
            if portfolio.classification_name != benchmark.classification_name:
                raise PpaError("audit_perfs(): Common classification name error.", 999)

    def _calculate_df_overall(self) -> pl.DataFrame:
        """Calculate one overall row for the full performance period.

        Returns:
            A Polars DataFrame containing one row for the full date range,
            including linked returns, summed contributions, day-weighted
            weights, and the overall beginning and ending dates.
        """
        # Overall returns are geometrically linked across subperiods, while overall
        # weights represent average exposure through time. A one-month 90% weight
        # and an eleven-month 10% weight should not average to 50%.
        all_return_col_names = self.col_names(RET) + [cols.TOTAL_RETURN]
        overall_beginning_date = self.df[cols.BEGINNING_DATE][0]
        overall_ending_date = self.df[cols.ENDING_DATE][-1]

        # Weight each period by its share of elapsed calendar days. This assumes
        # period weights describe exposure over the period, not only at an endpoint.
        total_overall_days = (overall_ending_date - overall_beginning_date).days

        # The zero-day branch is a defensive fallback for malformed in-memory data.
        # Normal Performance validation requires beginning_date < ending_date.
        if total_overall_days == 0:
            weight_coefficients = pl.Series(
                name=cols.QUANTITY_OF_DAYS, values=[1.0] * self.df.height
            )
        else:
            weight_coefficients = self.df[cols.QUANTITY_OF_DAYS] / total_overall_days

        # Returns compound through time, contributions add through time, and
        # weights are day-weighted exposures for the full reporting window.
        lf_overall = (
            self.df.lazy()
            .select(all_return_col_names + self.col_names(WGT) + self.col_names(CON))
            .with_columns(
                # Convert return series like +5%, -2% into wealth relatives
                # (1.05 * 0.98) before subtracting 1 in the final selection.
                [pl.col(col).add(1).cum_prod() for col in all_return_col_names]
                +
                # Convert each period weight into its contribution to average exposure.
                [(pl.col(col) * weight_coefficients) for col in self.col_names(WGT)]
            )
            .select(
                [
                    # The final cumulative wealth relative is the linked overall return.
                    pl.col(all_return_col_names).tail(1).sub(1),
                    # Day-weighted exposures and arithmetic contributions are additive.
                    pl.col(self.col_names(WGT)).sum(),
                    pl.col(self.col_names(CON)).sum(),
                ]
            )
            .with_columns(pl.lit(overall_beginning_date).alias(cols.BEGINNING_DATE))
            .with_columns(pl.lit(overall_ending_date).alias(cols.ENDING_DATE))
        )

        # Return df_overall
        return lf_overall.collect()

    def _cast_and_validate_columns(self) -> None:
        """Cast columns to required dtypes and validate missing values.

        Date columns are cast to ``pl.Date``, return and weight columns are
        cast to ``pl.Float64``, and optional classification columns are cast to
        ``pl.String``.

        Raises:
            PpaError: If a column cannot be cast to its required dtype, or if
                the DataFrame contains null values or NaN values in float
                columns.
        """
        # Get a dictionary of the column dtypes.
        column_dtypes: dict[type[pl.Date] | type[pl.Float64] | type[pl.String], list[str]] = {
            pl.Date: cols.DATE_COLUMNS,
            pl.Float64: self.col_names(RET) + self.col_names(WGT),
            pl.String: cols.PERFORMANCE_CLASSIFICATION_COLUMNS,
        }

        # Cache the schema into a local dictionary.  Otherwise polars rebuilds it every time you
        # access it.
        schema = self.df.schema

        # Iterate through the column dtypes.
        for dtype, col_names in column_dtypes.items():
            # Loop through columns with incorrect dtypes and try to cast them to the correct dtype.
            for col_name in [
                col for col in col_names if (col in self.df.columns and schema[col] != dtype)
            ]:
                # Cast the column to the appropriate dtype
                try:
                    self.df = self.df.with_columns(pl.col(col_name).cast(dtype))
                except pl.exceptions.InvalidOperationError as e:
                    raise PpaError(
                        f"{self.error_message_context}: "
                        f"Cannot convert the column '{col_name}' to a {dtype}, {str(e)[:1000]}",
                        110,
                    ) from e

        # Assert that there are not any missing (None) or NaN values.
        if self.df.lazy().select(pl.any_horizontal(pl.all().is_null().any())).collect().item() or (
            self.df.lazy()
            .select(pl.any_horizontal(pl.col(column_dtypes[pl.Float64]).is_nan().any()))
            .collect()
            .item()
        ):
            raise PpaError(self.error_message_context, 104)

    def _clean_and_validate_columns(self) -> None:
        """Keep required columns and validate return/weight column pairs.

        Raises:
            PpaError: If no return columns are present, or if return and
                weight columns do not contain matching identifiers.
        """
        # Create lists of different types of col_names.
        return_col_names = self._col_names_from_schema(RET)
        weight_col_names = self._col_names_from_schema(WGT)

        # Assert that there is at least one return.
        if len(return_col_names) == 0:
            raise PpaError(self.error_message_context, 109)

        # Assert that columns.ret == columns.wgt.  Note that polars does not allow for
        # duplicate col_names.
        identifiers = [col[:-4] for col in return_col_names]
        if identifiers != [col[:-4] for col in weight_col_names]:
            raise PpaError(self.error_message_context, 107)

        # Select only the column names that are needed.  This will drop any un-needed columns.
        self.df = self.df.select(cols.DATE_COLUMNS + return_col_names + weight_col_names)

    def _clean_and_validate_dates(self) -> None:
        """Sort, normalize, and validate period dates.

        Inclusive beginning dates are converted to the package's standard
        non-inclusive beginning-date convention when the entire series appears
        to use inclusive beginning dates.

        Raises:
            PpaError: If ending dates are duplicated, any beginning date is not
                before its ending date, or the time periods are discontinuous.
        """
        # Sort rows by ending_date.
        self.df = self.df.sort(cols.ENDING_DATE)

        # Assert that there are no duplicate ending_dates.
        qty_uniques = (
            self.df.lazy()
            .select(pl.col(cols.ENDING_DATE).n_unique())
            .collect()[cols.ENDING_DATE]
            .item(0)
        )

        # Assert that there are no duplicate ending_dates.
        if self.df.shape[0] != qty_uniques:
            raise PpaError(self.error_message_context, 102)

        # Typically, beginning_date[i] == ending_date[i - 1].  This is non-inclusive of
        # beginning_date, but inclusive of ending_date.  The following block will allow for
        # beginning_date to come in as inclusive, and it will change it to be non-inclusive.
        # For instance when beginning_date is 04/01/24, then this will change it to 03/31/24.
        if 1 < self.df.shape[0]:
            minus_1_day = (
                self.df.lazy().select(self.df[cols.BEGINNING_DATE] - pl.duration(days=1)).collect()
            )
            minus_1_day_df = minus_1_day[1:] != self.df[cols.ENDING_DATE][:-1]
            if minus_1_day_df[cols.BEGINNING_DATE].sum() == 0:
                self.df = (
                    self.df.lazy()
                    .with_columns(beginning_date=minus_1_day[cols.BEGINNING_DATE])
                    .collect()
                )

        # Assert that all beginning_dates < ending_dates
        date_sequences = (
            self.df.lazy()
            .select(self.df[cols.BEGINNING_DATE] >= self.df[cols.ENDING_DATE])
            .collect()
        )
        if date_sequences[cols.BEGINNING_DATE].sum() != 0:
            raise PpaError(self.error_message_context, 105)

        # Assert that there are no discontinuous time periods (date gaps).
        discontinuous_time_periods = (
            self.df.lazy()
            .select(self.df[cols.BEGINNING_DATE][1:] != self.df[cols.ENDING_DATE][:-1])
            .collect()
        )
        if discontinuous_time_periods[cols.BEGINNING_DATE].sum() != 0:
            raise PpaError(self.error_message_context, 106)

    def col_names(self, suffix: str) -> list[str]:
        """Return cached identifier column names for a suffix.

        Args:
            suffix: Column suffix to append to each identifier, such as
                ``".ret"``, ``".wgt"``, or ``".con"``.

        Returns:
            Identifier column names with ``suffix`` appended.
        """
        if suffix not in self._column_names:
            self._column_names[suffix] = [f"{id}{suffix}" for id in self.identifiers]
        return self._column_names[suffix]

    def _col_names_from_schema(self, column_name_suffix: str) -> list[str]:
        """Return sorted DataFrame column names ending with a suffix.

        Args:
            column_name_suffix: Column suffix to search for.

        Returns:
            Sorted column names from ``self.df`` that end with
            ``column_name_suffix``.
        """
        return sorted([name for name in self.df.columns if name.endswith(column_name_suffix)])

    def consolidated_returns(self) -> pl.DataFrame:
        """Return raw or implied returns used in attribution calculations.

        For unconsolidated data, the raw return columns are returned. For
        consolidated data, returns are implied from ``contribution / weight``
        where weight is nonzero; otherwise, the stored return is used.

        Returns:
            A Polars DataFrame containing return columns for each identifier.
        """
        # If the data has not been consolidated, the raw return columns are still aligned
        # with the contributions and can be returned directly.
        if not self.subperiods_have_been_consolidated:
            return self.df[self.col_names(RET)]

        # For consolidated data, raw stored return columns may no longer reflect the
        # contribution/weight relationship. Use implied returns from contribution / weight
        # when possible to preserve consistency in attribution calculations.
        return (
            self.df.select(self.col_names(CON))
            .with_columns(
                pl.when(self.df[f"{contrib[0:-4]}.wgt"] != 0)
                # The weight is not zero, so the implied_return = contrib / weight.
                .then(pl.col(contrib).truediv(self.df[f"{contrib[0:-4]}.wgt"]))
                # The weight is zero, so use the actual return.
                .otherwise(self.df[f"{contrib[0:-4]}.ret"])
                for contrib in self.col_names(CON)
            )
            .rename(lambda column_name: f"{column_name[:-4]}.ret")
        )

    def _convert_to_wide_format(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Convert narrow performance input to the package's wide layout.

        If ``self.df`` is already wide, it is returned unchanged with an empty
        classification-items DataFrame. For narrow input, the method pivots
        ``identifier`` rows into paired ``identifier.ret`` and
        ``identifier.wgt`` columns. If a ``name`` column is present, the method
        also creates classification identifier/name pairs for later use.

        Returns:
            A tuple containing the wide performance DataFrame and the optional
            classification-items DataFrame.

        Raises:
            PpaError: If narrow input contains more than one row for the same
                beginning date, ending date, and identifier.
        """
        # Return self.df if it is empty or already in the wide format.
        if self.df.shape[0] == 0 or not all(
            col in self.df.columns for col in (cols.IDENTIFIER, cols.RETURN, cols.WEIGHT)
        ):
            return self.df, pl.DataFrame()

        # Fail fast if there are multiple rows per (date, identifier).
        duplicate_rows = (
            self.df.group_by([*cols.DATE_COLUMNS, cols.IDENTIFIER]).len().filter(pl.col("len") > 1)
        )
        if duplicate_rows.height > 0:
            sample_rows: list[dict[str, object]] = (
                duplicate_rows.sort([*cols.DATE_COLUMNS, cols.IDENTIFIER]).head(10).to_dicts()
            )
            raise PpaError(f"{self.error_message_context}: {sample_rows}", 112)

        # Create self._classification_items if there is a cols.NAME column.
        # This might be used later if they do not specify a Classification data source.
        classification_items = (
            self.df.unique(subset=[cols.IDENTIFIER], keep="last")
            if cols.NAME in self.df.columns
            else pl.DataFrame()
        )
        if not classification_items.is_empty():
            classification_items = classification_items.select(
                cols.PERFORMANCE_CLASSIFICATION_COLUMNS
            )
            classification_items.columns = cols.CLASSIFICATION_COLUMNS

        # Perform a pivot: use the date columns as the index, and pivot on the identifier.
        # The 'values' are both WEIGHT and RETURN; we use an aggregate function of "first" since
        # there should be one value per date-identifier combination.
        pivoted = self.df.pivot(
            index=cols.DATE_COLUMNS,
            on=cols.IDENTIFIER,
            values=[cols.WEIGHT, cols.RETURN],
            aggregate_function="first",
        ).fill_null(0)

        # The pivot produces columns with names like return_msft and weight_aapl.  So change these
        # to the correct f"{identifier}{RET}" and f"{identifier}{WGT}".
        new_columns: dict[str, str] = {}
        return_prefix = f"{cols.RETURN}_"
        weight_prefix = f"{cols.WEIGHT}_"
        for col in pivoted.columns:
            if col.startswith(return_prefix):
                # Change return_msft to msft.ret
                new_columns[col] = f"{col[len(return_prefix):]}{RET}"
            elif col.startswith(weight_prefix):
                # Change weight_aapl to appl.wgt
                new_columns[col] = f"{col[len(weight_prefix):]}{WGT}"
            else:
                # Leave the date column names unchanged.
                new_columns[col] = col

        return pivoted.rename(new_columns), classification_items

    def df_overall(self) -> pl.DataFrame:
        """Return the cached overall-period DataFrame.

        Returns:
            A one-row Polars DataFrame for the full performance period.
        """
        if self._df_overall.is_empty():
            self._df_overall = self._calculate_df_overall()
        return self._df_overall

    def linking_coefficients(self) -> pl.Series:
        """Return logarithmic linking coefficients for each subperiod.

        Returns:
            A Polars Series containing one linking coefficient per subperiod.

        Raises:
            PpaError: Raised by ``util.logarithmic_linking_coefficients()`` if
                the overall return or any subperiod return is less than or
                equal to ``-1.0``.
        """
        return util.logarithmic_linking_coefficients(
            self.overall_return(), self.df[cols.TOTAL_RETURN]
        )

    @staticmethod
    def _load_data(
        name: str,
        data_source: util.PerformanceDataSource,
        beginning_date: dt.date,
        ending_date: dt.date,
    ) -> tuple[str, pl.DataFrame]:
        """Load performance data into a Polars DataFrame.

        Args:
            name: Descriptive name associated with the performance data.
            data_source: Performance data source. This can be a CSV file path,
                a pandas DataFrame, or a Polars DataFrame.
            beginning_date: Earliest beginning date to keep.
            ending_date: Latest ending date to keep.

        Returns:
            A tuple containing the resolved performance name and loaded Polars
            DataFrame.

        Raises:
            PpaError: If ``data_source`` is a file path that is empty, missing,
                or not a file.
        """
        # Load the data
        if isinstance(data_source, str):
            # Assert that the data file path exists.
            if not util.file_path_exists(data_source):
                raise PpaError(util.file_path_error(data_source), None)
            # Default the name to the file name
            if util.is_empty(name):
                name = util.file_basename_without_extension(data_source)
            # Load the csv file
            lf = pl.scan_csv(source=data_source, try_parse_dates=True)
        elif isinstance(data_source, pd.DataFrame):
            # Convert from pandas to polars
            lf = pl.from_pandas(data_source).lazy()
        else:  # isinstance(data_source, pl.DataFrame):
            # Is already a polars DataFrame
            lf = data_source.lazy()

        # Filter on the dates.
        if beginning_date != dt.date.min:
            lf = lf.filter(beginning_date <= pl.col(cols.BEGINNING_DATE))
        if ending_date != dt.date.max:
            lf = lf.filter(pl.col(cols.ENDING_DATE) <= ending_date)

        # Return the performance name and it's DataFrame.
        return name, lf.collect()

    def overall_return(self) -> float:
        """Return the linked total return for the full performance period.

        Returns:
            Total return for the full date range represented by ``self.df``.
        """
        return cast(float, self.df_overall().item(0, cols.TOTAL_RETURN))  # cast for mypy

    def _reset_column_names(self) -> None:
        """Clear cached column-name groups and rebuild identifiers."""
        # Set self._column_names to empty so they will be forced to be recalculated.
        self._column_names = {}

        # Restablish self.identifiers.
        self.identifiers = [name[:-4] for name in self._col_names_from_schema(RET)]

    def reset_df(self, df: pl.DataFrame, do_reset_column_names: bool = True) -> None:
        """Replace the performance DataFrame and invalidate cached summaries.

        Args:
            df: Replacement performance DataFrame.
            do_reset_column_names: Whether to rebuild cached column-name
                groups and identifiers after replacing ``self.df``.
        """
        # Set self._df_overall to empty so it will be forced to be recalculated.
        self._df_overall = pl.DataFrame()

        # Set self.df with the new dataframe.
        self.df = df

        # Reset the column names.
        if do_reset_column_names:
            self._reset_column_names()
