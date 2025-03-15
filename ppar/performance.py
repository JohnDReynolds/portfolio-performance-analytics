"""
The Performance class contains asset/category weights and returns in a polars DataFrame.
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
import ppar.errors as errs
import ppar.utilities as util


class Performance:
    """
    The Performance class contains asset/category weights and returns in a polars DataFrame.
    """

    def __init__(
        self,
        data_source: util.TypePerformanceDataSource,
        name: str = util.EMPTY,
        classification_name: str = util.EMPTY,
        beginning_date: str | dt.date = dt.date.min,
        ending_date: str | dt.date = dt.date.max,
    ):
        """
        The constructor.

        Args:
            data_source (TypePerformanceDataSource): One of the following:
                1. A csv file path containing the performance data.
                2. A pandas or polars DataFrame containing the performance data.
            name (str, optional): The descriptive name.  Defaults to util.EMPTY.
            classification_name (str, optional): The classification name corresponding to the
                data (e.g. "Economic Sector")  Defaults to util.EMPTY.
            beginning_date (str | dt.date, optional): Beginning date as a python date or a date
                string in the format yyyy-mm-dd.  Defaults to dt.date.min.
            ending_date (str | dt.date, optional): Ending date as a python date or a date string in
                the format yyyy-mm-dd.  Defaults to dt.date.max.
            do_calculate_df_overall (bool, optional): Calculate the summary DataFrame
                self.df_overall. This can be set to False if it will be calculated at a later
                time. For instance, in the Attribution constructor.  Defaults to True.

        Data Parameters:
            Sample input data for the "portfolio_data_source" & "benchmark_data_source" parameters
            can be in either of the 2 below formats.  The weights for each time period must
            sum to 1.0.  The equation SumOf(weight * return) == TotalReturn must be satisfied for
            each period.  The column names must conform to the below formats.  The ordering of the
            columns or rows does not matter.
            1. Narrow Format:
                beginning_date, ending_date, identifier,        return, weight
                2023-12-31,      2024-01-31,       aapl, -0.0422272121,    0.4
                2023-12-31,      2024-01-31,       msft,  0.0572811503,    0.6
                2024-01-31,      2024-02-29,       aapl, -0.019793881,     0.7
                2024-01-31,      2024-02-29,       msft,  0.0403944092,    0.3
            2. Wide Format:
                beginning_date, ending_date,      aapl.ret,     msft.ret, aapl.wgt, msft.wgt
                2023-12-31,      2024-01-31, -0.0422272121, 0.0572811503,      0.4,      0.6
                2024-01-31,      2024-02-29, -0.019793881,  0.0403944092,      0.7,      0.3
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
        assert (
            beginning_date <= ending_date
        ), f"{errs.ERROR_111_INVALID_DATES}{ending_date} < {beginning_date}"

        # Load the data.
        self.name, self.df = Performance._load_data(name, data_source, beginning_date, ending_date)

        # Convert self.df to "wide" format with multiple identifier.ret and identifier.wgt columns.
        # If cols.IDENTIFIER and cols.NAME are in self.df, then create self.classification_items,
        # which might be used later in the Attribution constructor when creating the Classification
        self.df, self.classification_items = self._convert_to_wide_format()

        # Assert that there is at least 1 row.
        assert (
            0 < self.df.shape[0]
        ), f"{errs.ERROR_103_NO_PERFORMANCE_ROWS}{self.error_message_context}"

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
        assert (
            self.df[self.col_names(WGT)].sum_horizontal().round(8) == 1.0
        ).all(), f"{errs.ERROR_108_WEIGHTS_DO_NOT_SUM_TO_1}{self.error_message_context}"

        # self._df_overall is one row for the entire overall period.
        self._df_overall = pl.DataFrame()

    def audit(self) -> None:
        """Audit the Performance (self)."""
        # Assert that the weights sum to 1.0
        assert (
            self.df[self.col_names(WGT)].sum_horizontal().round(8) == 1.0
        ).all(), f"{errs.ERROR_999_UNEXPECTED}Perf.audit(): Weights do not sum to 1.0."

        # If not perf.subperiods_have_been_consolidated, then assert that weight * return
        # == contrib.  Note that this cannot be direcly checked in the Performance constructor
        # because the subperiods are not consolidated until the Analytics class.
        if not self.subperiods_have_been_consolidated:
            contribs = (self.df[self.col_names(RET)] * self.df[self.col_names(WGT)]).rename(
                lambda column_name: f"{column_name[:-4]}.con"
            )
            assert contribs.equals(
                self.df[self.col_names(CON)]
            ), f"{errs.ERROR_999_UNEXPECTED}Perf.audit(): weight * return != contrib."
            assert (
                self.df[cols.TOTAL_RETURN].round(11) == contribs.sum_horizontal().round(11)
            ).all(), f"{errs.ERROR_999_UNEXPECTED}Perf.audit(): contribs != total return."

    @staticmethod
    def audit_performances(
        performances: tuple["Performance", "Performance"],
        expected_beginning_date: dt.date,
        expected_ending_date: dt.date,
        common_classification_name: str = util.EMPTY,
    ) -> None:
        """
        Audit the portfolio/benchmark pair of performances.

        Args:
            performances (tuple[Performance, Performance]): The portfolio & benchmark Performances.
            expected_beginning_date (dt.date): The expected beginning date.
            expected_ending_date (dt.date): The expected ending date.
            common_classification_name (str, optional): The classification name that should be
                shared by both the portfolio and the benchmark.  Defaults to util.EMPTY.
        """
        # Set the portfolio and benchmark
        portfolio = performances[0]
        benchmark = performances[1]

        # Audit each Performance separately.
        portfolio.audit()
        benchmark.audit()

        # Assert that the portfolio and benchmark have the same dates and days.
        dates_days = (cols.BEGINNING_DATE, cols.ENDING_DATE, cols.QUANTITY_OF_DAYS)
        assert portfolio.df[dates_days].equals(
            benchmark.df[dates_days]
        ), f"{errs.ERROR_999_UNEXPECTED}audit_perfs(): Portfolio and Benchmark dates are not equal"

        # Assert that the portfolio/benchmark dates are equal to the expected dates.
        assert (
            portfolio.df[cols.BEGINNING_DATE][0] == expected_beginning_date
            and portfolio.df[cols.ENDING_DATE][-1] == expected_ending_date
        ), f"{errs.ERROR_999_UNEXPECTED}audit_perfs(): Date logic error."

        # Assert that the portfolio and benchmark both have the same common_classification_name.
        if not util.is_empty(common_classification_name):
            assert (
                portfolio.classification_name == benchmark.classification_name
            ), f"{errs.ERROR_999_UNEXPECTED}audit_perfs(): Common classification name error."

    def _calculate_df_overall(self) -> pl.DataFrame:
        """
        Calculate df_overall, which is one total row for the entire overall period.  It is either
        called from the constructor or from the Attribution class after the dates have been
        firmly established.

        Returns:
            pl.DataFrame: df_overall, which is one row for the entire overall period.
        """
        # Pre-calculate values
        all_return_col_names = self.col_names(RET) + [cols.TOTAL_RETURN]
        overall_beginning_date = self.df[cols.BEGINNING_DATE][0]
        overall_ending_date = self.df[cols.ENDING_DATE][-1]
        weight_coefficients = (
            self.df[cols.QUANTITY_OF_DAYS] / (overall_ending_date - overall_beginning_date).days
        )

        # Calculate the overall linked return, sum of contributions, and day-weighted weights.
        lf_overall = (
            self.df.lazy()
            .select(all_return_col_names + self.col_names(WGT) + self.col_names(CON))
            .with_columns(
                # Add 1 to the returns, and then take the cumulative product.
                [pl.col(col).add(1).cum_prod() for col in all_return_col_names]
                +
                # Calculate the day-weighted weights.
                [(pl.col(col) * weight_coefficients) for col in self.col_names(WGT)]
            )
            .select(
                [
                    # Take the final (tail) linked return, and then subtract 1.
                    pl.col(all_return_col_names).tail(1).sub(1),
                    # Sum the weights and contributions.
                    pl.col(self.col_names(WGT)).sum(),
                    pl.col(self.col_names(CON)).sum(),
                ]
            )
            # Add the overall period dates.
            .with_columns(pl.lit(overall_beginning_date).alias(cols.BEGINNING_DATE))
            .with_columns(pl.lit(overall_ending_date).alias(cols.ENDING_DATE))
        )

        # Return df_overall
        return lf_overall.collect()

    def _cast_and_validate_columns(self) -> None:
        """
        Cast the columns to their correct data types.  They might come in as strings or ints.  If
        any columns have null values or any pl.Float64 columns have NaN values, then raise an
        exception.
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
                    raise pl.exceptions.InvalidOperationError(
                        f"{errs.ERROR_110_INVALID_PERFORMANCE_DATA_FORMAT}"
                        f"{self.error_message_context}: "
                        f"Cannot convert the column '{col_name}' to a {dtype}, {str(e)[:1000]}"
                    ) from e

        # Assert that there are not any missing (None) or NaN values.
        assert not (
            self.df.lazy().select(pl.any_horizontal(pl.all().is_null().any())).collect().item()
            or (
                self.df.lazy()
                .select(pl.any_horizontal(pl.col(column_dtypes[pl.Float64]).is_nan().any()))
                .collect()
                .item()
            )
        ), f"{errs.ERROR_104_MISSING_VALUES}{self.error_message_context}"

    def _clean_and_validate_columns(self) -> None:
        """Clean and validate the columns."""
        # Create lists of different types of col_names.
        return_col_names = self._col_names_from_schema(RET)
        weight_col_names = self._col_names_from_schema(WGT)

        # Assert that there is at least one return.
        assert (
            len(return_col_names) != 0
        ), f"{errs.ERROR_109_NO_RETURNS_OR_WEIGHTS}{self.error_message_context}"

        # Assert that columns.ret == columns.wgt.  Note that polars does not allow for
        # duplicate col_names.
        identifiers = [col[:-4] for col in return_col_names]
        assert identifiers == [col[:-4] for col in weight_col_names], (
            f"{errs.ERROR_107_RETURN_COLUMNS_NOT_EQUAL_TO_WEIGHT_COLUMNS}"
            f"{self.error_message_context}"
        )

        # Select only the column names that are needed.  This will drop any un-needed columns.
        self.df = self.df.select(cols.DATE_COLUMNS + return_col_names + weight_col_names)

    def _clean_and_validate_dates(self) -> None:
        """
        Clean and validate all of the beginning dates and ending dates for every row.
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
        assert (
            self.df.shape[0] == qty_uniques
        ), f"{errs.ERROR_102_ENDING_DATES_ARE_NOT_UNIQUE}{self.error_message_context}"

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
        assert date_sequences[cols.BEGINNING_DATE].sum() == 0, (
            f"{errs.ERROR_105_BEGINNING_DATES_GREATER_THAN_ENDING_DATES}"
            f"{self.error_message_context}"
        )

        # Assert that there are no discontinuous time periods (date gaps).
        discontinuous_time_periods = (
            self.df.lazy()
            .select(self.df[cols.BEGINNING_DATE][1:] != self.df[cols.ENDING_DATE][:-1])
            .collect()
        )
        assert (
            discontinuous_time_periods[cols.BEGINNING_DATE].sum() == 0
        ), f"{errs.ERROR_106_DISCONTINUOS_TIME_PERIODS}{self.error_message_context}"

    def col_names(self, suffix: str) -> list[str]:
        """
        Return a list of identifier column names with the suffix appended.

        Args:
            suffix (str): The suffix (e.g. ".ret", ".sel", etc)

        Returns:
            list[str]: A list of identifier column names with the suffix appended.
        """
        if suffix not in self._column_names:
            self._column_names[suffix] = [f"{id}{suffix}" for id in self.identifiers]
        return self._column_names[suffix]

    def _col_names_from_schema(self, column_name_suffix: str) -> list[str]:
        """
        Gets a list of the column names that have column_name_suffix.

        Args:
            col_name_suffix (str): The column name suffix for which to get the column names.

        Returns:
            list[str]: A sorted list of the column names that have column_name_suffix.
        """
        return sorted([name for name in self.df.columns if name.endswith(column_name_suffix)])

    def consolidated_returns(self) -> pl.DataFrame:
        """
        Calculates the implied consolidated returns.  They are only used for calculating the
        multi-period attribution effects.  This only needs to be done if
        self.subperiods_have_been_consolidated, because after consolidation, weight * return !=
        contrib.  So the implied consolidated_return = contrib / weight.

        Returns:
            pl.DataFrame: A DataFrame containing the consolidated returns.
        """
        # If not self.subperiods_have_been_consolidated, then weight * return = contrib.  So the raw
        # returns can be returned.
        if not self.subperiods_have_been_consolidated:
            return self.df[self.col_names(RET)]

        # Return the consolidated returns.
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
        """
        Convert self.df to the "wide" format with multiple identifier.ret and identifier.wgt
        columns.  If cols.IDENTIFIER and cols.NAME are in self.df, then create
        self.classification_items, which might be used later in the Attribution constructor when
        creating the Classification.

        Returns:
            tuple[pl.DataFrame, pl.DataFrame]: The wide self.df and self.classification_items.
        """
        # Return self.df if it is empty or already in the wide format.
        if self.df.shape[0] == 0 or not all(
            col in self.df.columns for col in (cols.IDENTIFIER, cols.RETURN, cols.WEIGHT)
        ):
            return self.df, pl.DataFrame()

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
        """Get the DataFrame representing the overall total period."""
        if self._df_overall.is_empty():
            self._df_overall = self._calculate_df_overall()
        return self._df_overall

    def linking_coefficients(self) -> pl.Series:
        """Return the linking coefficients."""
        return util.logarithmic_linking_coefficients(
            self.overall_return(), self.df[cols.TOTAL_RETURN]
        )

    @staticmethod
    def _load_data(
        name: str,
        data_source: util.TypePerformanceDataSource,
        beginning_date: dt.date,
        ending_date: dt.date,
    ) -> tuple[str, pl.DataFrame]:
        """
        Loads performance data into a polars DataFrame.

        Args:
            name (str, optional): The name assoiated with the data.  Defaults to EMPTY.
            data_source (TypePerformanceDataSource): The performance data source.
            beginning_date (dt.date, optional): The beginning date to filter the data on.
            ending_date (dt.date, optional): The ending date to filter the data on.

        Returns:
            tuple[str, pl.DataFrame]: The performance name and it's data (in a Dataframe).
        """
        # Load the data
        if isinstance(data_source, str):
            # Assert that the data file path exists.
            assert util.file_path_exists(data_source), util.file_path_error(data_source)
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
        """
        Gets the total return for the entire overall time period in self.df.

        Returns:
            float: The total return for the entire overall time period in self.df.
        """
        return cast(float, self.df_overall().item(0, cols.TOTAL_RETURN))  # cast for mypy

    def _reset_column_names(self) -> None:
        """
        Reset the column name instance variables.
        """
        # Set self._column_names to empty so they will be forced to be recalculated.
        self._column_names = {}

        # Restablish self.identifiers.
        self.identifiers = [name[:-4] for name in self._col_names_from_schema(RET)]

    def reset_df(self, df: pl.DataFrame, do_reset_column_names: bool = True) -> None:
        """
        Reset the DataFrame self.df

        Args:
            df (pl.DataFrame): The new dataframe.
            reset_column_names (bool, optional): Reset the column names if they have changed.
                Defaults to True.
        """
        # Set self._df_overall to empty so it will be forced to be recalculated.
        self._df_overall = pl.DataFrame()

        # Set self.df with the new dataframe.
        self.df = df

        # Reset the column names.
        if do_reset_column_names:
            self._reset_column_names()
