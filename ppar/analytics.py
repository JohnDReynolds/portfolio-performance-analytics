"""
The Analytics class reads and validates portfolio and benchmark Performance data, and then
consolidates them into common time periods based on the provided dates and frequency.

The public methods to retrieve the analytical calculations are:
    1. get_attribution()
    2. get_riskstatistics()
"""

# Python Imports
import bisect
from collections import defaultdict
import datetime as dt

# Third-Party Imports
import polars as pl

# Project Imports
from ppar.attribution import Attribution
import ppar.columns as cols
from ppar.columns import CON, RET, WGT
import ppar.errors as errs
from ppar.frequency import Frequency, date_matches_frequency
from ppar.mapping import Mapping
from ppar.performance import Performance
from ppar.riskstatistics import RiskStatistics
import ppar.utilities as util


class Analytics:
    """
    The Analytics class reads and validates portfolio and benchmark Performance data, and then
    consolidates them into common time periods based on the provided dates and frequency.

    The public methods to retrieve the analytical calculations are:
        1. get_attribution()
        2. get_riskstatistics()
    """

    def __init__(
        self,
        # Portfolio and Benchmark parameters
        portfolio_data_source: util.TypePerformanceDataSource,
        benchmark_data_source: util.TypePerformanceDataSource = util.EMPTY,
        portfolio_name: str = util.EMPTY,
        benchmark_name: str = util.EMPTY,
        portfolio_classification_name: str = util.EMPTY,
        benchmark_classification_name: str = util.EMPTY,
        # Date and frequency parameters
        beginning_date: str | dt.date = dt.date.min,
        ending_date: str | dt.date = dt.date.max,
        frequency: Frequency = Frequency.AS_OFTEN_AS_POSSIBLE,
        # RiskStatistics parameters
        annual_minimum_acceptable_return: float = util.DEFAULT_ANNUAL_MINIMUM_ACCEPTABLE_RETURN,
        annual_risk_free_rate: float = util.DEFAULT_ANNUAL_RISK_FREE_RATE,
        confidence_level: float = util.DEFAULT_CONFIDENCE_LEVEL,
        portfolio_value: tuple[float, str] = (
            util.DEFAULT_PORTFOLIO_VALUE,
            util.DEFAULT_CURRENCY_SYMBOL,
        ),
    ):
        """
        The constructor.  Reads and validates portfolio and benchmark Performance data, and
        cnsolidates them into common time periods based on the provided dates and frequency.

        Args:
            portfolio_data_source (TypePerformanceDataSource): One of the following:
                1. A csv file path containing the portfolio performance data.
                2. A pandas or polars DataFrame containing the portfolio performance data.
            benchmark_data_source (TypePerformanceDataSource, optional): One of the following:
                1. A csv file path containing the benchmark performance data.
                2. A pandas or polars DataFrame containing the benchmark performance data.
                Defaults to portfolio_data_source.
            portfolio_name (str, optional): The portfolio name used in view titles.
            benchmark_name (str, optional): The benchmark name used in view titles.
            portfolio_classification_name (str, optional): The classification name that corresponds
                to the portfolio_data. Defaults to util.EMPTY.
            benchmark_classification_name (str, optional): The classification name that corresponds
                to the benchmark_data. Defaults to util.EMPTY.
            beginning_date (str | dt.date, optional): Beginning date as a python date or a date
                string in the format yyyy-mm-dd.  Defaults to dt.date.min.
            ending_date (str | dt.date, optional): Ending date as a python date or a date string in
                the format yyyy-mm-dd.  Defaults to dt.date.max.
            frequency (Frequency, optional): The periodic frequency for which to
                produce Attribution instances.  Can be either:
                1. Frequency.AS_OFTEN_AS_POSSIBLE, meaning "as often as the provided data allows".
                   If daily dates, weights and returns are provided, then daily analytics will be
                   created.  If monthly dates, weights and returns are provided, then monthly
                   analytics will be created, etc.
                2. Frequency.MONTHLY
                3. Frequency.QUARTERLY
                4. Frequency.YEARLY
                Defaults to Frequency.AS_OFTEN_AS_POSSIBLE.
            annual_minimum_acceptable_return (float, optional): The minimum acceptable return used
                for calculating "downside" satistics.
                Defaults to util.DEFAULT_ANNUAL_MINIMUM_ACCEPTABLE_RETURN.
            annual_risk_free_rate (float, optional): The annual risk-free rate used for
                calculating statistics that involve a risk-free rates.
                Defaults to util.DEFAULT_ANNUAL_RISK_FREE_RATE.
            confidence_level (float, optional): The confidence level for calculating the
                value-at-risk (VAR).  Defaults to util.DEFAULT_CONFIDENCE_LEVEL.
            portfolio_value (tuple[float, str], optional): A tuple of the portfolio value and it's
                associated currency that will be used when calculating the value-at-risk (VaR).
                Defaults to (util.DEFAULT_PORTFOLIO_VALUE, util.DEFAULT_CURRENCY_SYMBOL).

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
        # Default the benchmark to the portfolio.  This will allow for "portfolio-only" analysis
        # if they do not have a benchmark.
        if util.is_empty(benchmark_data_source):
            benchmark_data_source = portfolio_data_source

        # Convert the dates to dt.date types.
        beginning_date = util.convert_to_date(beginning_date)
        ending_date = util.convert_to_date(ending_date)

        # Set the simple class variables directly from the constructor parameters.
        self._annual_minimum_acceptable_return = annual_minimum_acceptable_return
        self._annual_risk_free_rate = annual_risk_free_rate
        self._confidence_level = confidence_level
        self._frequency = frequency
        self._portfolio_value = portfolio_value

        # Initialize the internal data structures.
        self._attributions: dict[str, Attribution] = {}  # key = classification_name
        self._riskstatistics: RiskStatistics | None = None

        # Get a tuple of the 2 Performance classes.  portfolio == 0, benchmark == 1.
        self._performances = (
            # Portfolio
            Performance(
                portfolio_data_source,
                name=portfolio_name,
                classification_name=portfolio_classification_name,
                beginning_date=beginning_date,
                ending_date=ending_date,
            ),
            # Benchmark
            Performance(
                benchmark_data_source,
                name=benchmark_name,
                classification_name=benchmark_classification_name,
                beginning_date=beginning_date,
                ending_date=ending_date,
            ),
        )

        # Get the beginning_dates and ending_dates for all subperiods that are common between the
        # two Performances.
        self._subperiod_dates = self._calculate_subperiod_dates(
            f"from {util.date_str(beginning_date)} to {util.date_str(ending_date)}"
        )

        # Now that the dates have been firmly established, remove the extraneous rows (dates) from
        # the Performances.
        for perf in self._performances:
            perf.df = (
                perf.df.lazy()
                .filter(
                    (
                        (self._beginning_date() <= pl.col(cols.BEGINNING_DATE))
                        & (pl.col(cols.ENDING_DATE) <= self._ending_date())
                    )
                )
                .collect()
            )

        # Consolidate multiple subperiods (e.g. daily) into single periods (e.g. monthly) based on
        # self._frequency.
        self._consolidate_all_subperiods()

    def audit(self) -> None:
        """Audit the Analytics (self)."""
        # Audit the portfolio/benchmark pair of performances.  These are the performances that
        # were originally read in the constructor.  Depending on their classifications, they may
        # be differenct than the performances in the attributions.
        Performance.audit_performances(
            self._performances, self._beginning_date(), self._ending_date()
        )

        # Audit the attributions and their associated performances.
        Attribution.audit_attributions(list(self._attributions.values()))

    def _beginning_date(self) -> dt.date:
        """
        Get the overall beginning date.

        Returns:
            dt.date: The overall beginning date.
        """
        return self._subperiod_dates[0][0]

    def _calculate_subperiod_dates(self, message_suffix: str) -> list[tuple[dt.date, dt.date]]:
        """
        Calculate the beginning_dates and ending_dates for all subperiods that are common between
        the 2 self._performances.  This will define the subperiods for which the Attribution and
        RiskStatistics will be calculated.

        Args:
            message_suffix (str): Error message suffix.

        Returns:
            list[tuple[dt.date, dt.date]]: A list of the common beginning dates and ending dates
            for each subperiod.
        """

        def _common_dates(dates1: pl.Series, dates2: pl.Series) -> pl.Series:
            """Return the sorted dates common between dates1 and dates2."""
            # Note that using set intersection is MUCH slower.
            # return sorted(set(dates1) & set(dates2))
            return dates1.filter(dates1.is_in(dates2)).sort()

        def _filter_dates_on_frequency(dates: pl.Series | list[dt.date]) -> list[dt.date]:
            """Filter the dates based on self._frequency."""
            return [date for date in dates if date_matches_frequency(date, self._frequency)]

        # Cache the performance DataFrames.
        df0 = self._performances[0].df
        df1 = self._performances[1].df

        # Compute sorted common beginning and ending dates.
        common_beginning_dates: pl.Series | list[dt.date] = _common_dates(
            df0[cols.BEGINNING_DATE], df1[cols.BEGINNING_DATE]
        )
        common_ending_dates: pl.Series | list[dt.date] = _common_dates(
            df0[cols.ENDING_DATE], df1[cols.ENDING_DATE]
        )

        # Filter the dates based on frequency.
        if self._frequency != Frequency.AS_OFTEN_AS_POSSIBLE:
            common_beginning_dates = _filter_dates_on_frequency(common_beginning_dates)
            common_ending_dates = _filter_dates_on_frequency(common_ending_dates)

        # For each beginning date, find the first ending date that is strictly greater.
        subperiod_dates: list[tuple[dt.date, dt.date]] = []
        idx = 0
        len_common_end_dates = len(common_ending_dates)
        for begin_date in common_beginning_dates:
            if idx < len_common_end_dates and common_ending_dates[idx] <= begin_date:
                # bisect_right returns the insertion point which is the index of the first ending
                # date > b.
                idx = bisect.bisect_right(common_ending_dates, begin_date, lo=idx + 1)
            if idx < len_common_end_dates:
                subperiod_dates.append((begin_date, common_ending_dates[idx]))
                idx += 1

        # Assert that there is at least one subperiod.
        assert 0 < len(subperiod_dates), f"{errs.ERROR_202_NO_REPORTABLE_DATES}{message_suffix}"

        # Return the common beginning and ending dates that define the subperiods.
        return subperiod_dates

    def classification_names(self) -> tuple[str, str]:
        """
        Get a tuple of the classification names:  0=Portfolio, 1=Benchmark

        Returns:
            tuple[str, str]: A tuple of the classification names:  0=Portfolio, 1=Benchmark
        """
        return (
            self._performances[0].classification_name,
            self._performances[1].classification_name,
        )

    def _consolidate_all_subperiods(self) -> None:
        """
        Consolidate multiple subperiods (e.g. daily) into single periods (e.g. monthly) based on
        self._frequency.  This is done for both the portfolio and benchmark Performances in
        self._performances.
        """
        # Iterate through the portfolio and benchmark Performances.
        for performance in self._performances:
            # Assert that performance.df has at least the same quantity of rows as
            # self._subperiod_dates.
            assert len(self._subperiod_dates) <= performance.df.shape[0], (
                f"{errs.ERROR_999_UNEXPECTED}"
                f"{performance.error_message_context} from {util.date_str(self._beginning_date())}"
                f" to {util.date_str(self._ending_date())}"
            )

            # If performance.df has more rows than self._subperiod_dates, then that means that
            # performance.df has subperiod rows that need to be consolidated into the
            # self._subperiod_dates periods.
            if len(self._subperiod_dates) < performance.df.shape[0]:
                # Consolidate the subperiods.
                performance.reset_df(
                    df=self._consolidate_subperiods(performance).collect(),
                    do_reset_column_names=False,
                )

    def _consolidate_subperiods(self, performance: Performance) -> pl.LazyFrame:
        """
        Consolidate performance.df into one row for each subperiod in self._subperiod_dates.
        For instance, consolidate daily to monthly, or monthly to quarterly.

        Args:
            performance (Performance): The Performance instance.

        Returns:
            LazyFrame: _type_: The consolidated Performance LazyFrame.
        """
        # Create a DataFrame, one row per subperiod.
        df_subperiods = (
            pl.DataFrame(
                {
                    "beg_date": [bd for bd, _ in self._subperiod_dates],
                    "end_date": [ed for _, ed in self._subperiod_dates],
                }
            )
            .with_row_index(name="subperiod_id")
            .lazy()
        )

        # Create a LazyFrame that contains all of the performance.df columns as well as the
        # subperiod_id and dates.
        joined_lf = performance.df.lazy().join_asof(
            df_subperiods,
            left_on=cols.BEGINNING_DATE,
            right_on="beg_date",
            strategy="backward",
            by=None,
        )

        # Create a LazyFrame with the subperiod_id and the subperiod_return.
        subperiod_returns = joined_lf.group_by("subperiod_id").agg(
            [pl.col(cols.TOTAL_RETURN).add(1).cum_prod().last().sub(1).alias("subperiod_return")]
        )

        # Join the subperiod_returns.  Since LazyFrame columns cannot have arithmetic performed on
        # themselves, you must collect() here.
        joined_df = joined_lf.join(subperiod_returns, on="subperiod_id").collect()

        # Append the day-weighting coefficients and the linking coefficients.
        joined_lf = joined_df.lazy().with_columns(
            # Append the day-weighting coefficient column.
            (
                joined_df[cols.QUANTITY_OF_DAYS]
                / (joined_df["end_date"] - joined_df["beg_date"]).dt.total_days()
            ).alias("weight_coefficient"),
            # Append the linking coefficient column.
            pl.struct(["subperiod_return", cols.TOTAL_RETURN])
            .map_batches(
                lambda x: util.logarithmic_linking_coefficient_series(
                    x.struct.field("subperiod_return"), x.struct.field(cols.TOTAL_RETURN)
                ),
                return_dtype=pl.Float64,
            )
            .alias("linking_coefficient"),
        )

        # Get the final consolidated subperiods by linking the returns, summing the day-weighted
        # weights, and summing the contributions after applying the linking coefficients.
        consolidated_subperiods_lf = (
            joined_lf.group_by("subperiod_id")
            .agg(
                [
                    # Some of these expressions produce lists of either single values or identical
                    # values.  So take the first().
                    # Dates and Days
                    pl.col("beg_date").first().alias(cols.BEGINNING_DATE),
                    pl.col("end_date").first().alias(cols.ENDING_DATE),
                    pl.col(cols.QUANTITY_OF_DAYS).sum(),
                    # Total Return
                    pl.col(cols.TOTAL_RETURN).add(1).cum_prod().last().sub(1),
                    # Returns
                    pl.col(performance.col_names(RET)).add(1).cum_prod().tail(1).sub(1).first(),
                    # Weights
                    pl.col(performance.col_names(WGT)).mul(pl.col("weight_coefficient")).sum(),
                    # Contributions
                    pl.col(performance.col_names(CON)).mul(pl.col("linking_coefficient")).sum(),
                ]
            )
            .sort("subperiod_id")
        )

        # Mark the performance as being consolidated.
        performance.subperiods_have_been_consolidated = True

        # Collect and return the consolidated subperiods.
        return consolidated_subperiods_lf

    def _ending_date(self) -> dt.date:
        """
        Summary:
            Get the overall ending date.
        Returns:
            dt.date: The overall ending date.
        """
        return self._subperiod_dates[-1][-1]

    def get_attribution(
        self,
        classification_name: str = util.EMPTY,
        classification_data_source: util.TypeClassificationDataSource = util.EMPTY,
        mapping_data_sources: tuple[util.TypeMappingDataSource, util.TypeMappingDataSource] = (
            util.EMPTY,
            util.EMPTY,
        ),
        classification_label: str = util.EMPTY,
    ) -> Attribution:
        """
        Get the Attribution instance associated with the classification_name.

        Args:
            classification_name (str, optional): The classification name for the desired
                Attribution instance.  Defaults to util.EMPTY.
            classification_data_source (TypeClassificationDataSource): One of the following:
                1. A csv file path containing the Classification data.
                2. A dictionary containing the Classification data.
                3. A pandas or polars DataFrame containing the Classification data.
                Defaults to util.EMPTY.
            mapping_data_sources (TypeMappingDataSource, TypeMappingDataSource): A tuple of 2
                mapping data sources: 0 = portfolio, 1 = benchmark.  Each mapping data source can
                be one of the following:
                1. A csv file path containing the Mapping data.
                2. A dictionary containing the Mapping data.
                3. A pandas or polars DataFrame containing the Mapping data.
                Defaults to (util.EMPTY, util.EMPTY)
            classification_label (str, optional): The classification label that will be displayed
                in the tables and charts if the classification_name is empty.  This will happen
                when the performance.classification_items are used.  Defaults to util.EMPTY.

        Data Parameters:
            Sample data for the "classification_data_source" param of a "Security" Classification:
                AAPL, Apple Inc.
                MSFT, Microsft
                ...
            Sample data for the "mapping_data_source" param for "Security" to "Economic Sector":
                AAPL, IT
                GOOG, CS
                ...

        Returns:
            Attribution: The Attribution instance associated with the classification_name.
        """
        # If the classification_name is empty, and the portflio and benchmark have common
        # non-empty classification_names, then set the classificcation_name to that common
        # classification_name.
        if (
            util.is_empty(classification_name)
            and not util.is_empty(self._performances[0].classification_name)
            and self._performances[0].classification_name
            == self._performances[1].classification_name
        ):
            classification_name = self._performances[0].classification_name

        # If the classification_name is unknown, and either the portfolio or benchmark have known
        # classificiation names, then mandate that the classification_name is specified.  Note
        # that this wll still allow for all 3 of the classifications to be unknown.
        assert not (
            util.is_empty(classification_name)
            and (
                (not util.is_empty(self._performances[0].classification_name))
                or (not util.is_empty(self._performances[1].classification_name))
            )
        ), errs.ERROR_252_MUST_SPECIFY_CLASSIFICATION_NAME

        # Return the attribution if it already exists in the cache.
        if classification_name in self._attributions:
            return self._attributions[classification_name]

        # Get the performances for the common classification_name.
        attribution_performances = [
            (
                perf
                if perf.classification_name == classification_name
                else self._map_performance(perf, classification_name, mapping_data_sources[idx])
            )
            for idx, perf in enumerate(self._performances)
        ]

        # Now that both attribution performances are of the same common Classification,
        # calculate the Attribution.
        self._attributions[classification_name] = Attribution(
            (attribution_performances[0], attribution_performances[1]),
            classification_name,
            classification_data_source,
            self._frequency,
            classification_label,
        )

        # Return the Attribution coresponding to classification_name.
        return self._attributions[classification_name]

    def get_riskstatistics(self) -> RiskStatistics:
        """
        Calculates the risk statistics and puts them into the cache self._riskstatistics.

        Returns:
            pl.DataFrame: A DataFrame of the risk statistics.
        """
        # Calculate the risk statistics if they are not already cached.
        if self._riskstatistics is None:
            self._riskstatistics = RiskStatistics(
                self._performances,
                self._frequency,
                self._annual_minimum_acceptable_return,
                self._annual_risk_free_rate,
                self._confidence_level,
                self._portfolio_value,
            )

        # Return the DataFrame of the risk statistics.
        return self._riskstatistics

    def _map_columns(
        self,
        performance: Performance,
        to_froms: defaultdict[str, list[str]],
        suffix: str,
    ) -> pl.LazyFrame:
        """
        Map (sum) columns using the mapping.

        Args:
            performance (Performance): The Performance to be mapped (summed).
            to_froms: defaultdict[str, list[str]]: A reverse mapping from `to_column_name` to a
                list of `from_column_names`.
            suffix (str): The suffix for the column names that will be mapped
                (e.g., '.con' or '.wgt').

        Returns:
            pl.LazyFrame: The resulting mapped columns.
        """
        # Create aggregated columns using Polars expressions
        aggregated_columns = [
            pl.sum_horizontal([pl.col(f"{col}{suffix}") for col in from_columns]).alias(
                f"{to_value}{suffix}"
            )
            for to_value, from_columns in to_froms.items()
        ]

        # Perform the horizontal summations of the expressions.  Note that typically there will
        # only be 10 - 50 expressions (e.g. the qty of "to" columns, e.g. the qty of the reporting
        # "to" classification items).  But if they have 10,000 securities and incomplete mappings,
        # then there could be close to 10,000 expressions, which polars struggles with.  It can run
        # into memory issues, even in lazy mode.  So chunk them into batches.
        batch_size = 1000
        horizontally_summed_lfs: list[pl.LazyFrame] = []
        performance_lf = performance.df.lazy()
        for i in range(0, len(aggregated_columns), batch_size):
            horizontally_summed_lfs.append(
                performance_lf.select(aggregated_columns[i : i + batch_size])
            )

        # Concatenate and return the horizontally_summed_lfs.
        return (
            horizontally_summed_lfs[0]
            if len(horizontally_summed_lfs) == 1
            else pl.concat(horizontally_summed_lfs, how="horizontal")
        )

    def _map_performance(
        self,
        performance: Performance,
        to_classification_name: str,
        mapping_data_source: util.TypeMappingDataSource,
    ) -> Performance:
        """
        Map from the Performance Classification to the to_classification.  For instance, from
        Security to Economic Sector.

        Args:
            performance (Performance): The existing Performance that will be mapped.
            to_classification_name (str): The classification name that will be mapped to.
            mapping_data_source (TypeMappingDataSource): The Mapping data source.

        Data Parameters:
            Sample input for the "mapping_data" parameter for "Security" to "Economic Sector":
                AAPL, IT
                GOOG, CO
                ...

        Returns:
            Performance: The new mapped Performance.
        """
        # Create a reverse mapping from `to_column_name` to a list of `from_column_names`.
        to_froms = Mapping(
            performance.identifiers,
            mapping_data_source,
        ).to_froms

        # Get DataFrames of the resulting mapped columns with the new mapped identifiers as the new
        # column names.  For instance if the roll-up is from security to Economic Sector, then the
        # columns ['aapl.con', 'hpq.con'] will be horizontally summed into a single new column
        # named 'IT'.
        mapped_contribs_lf = self._map_columns(performance, to_froms, CON)
        mapped_weights_lf = self._map_columns(performance, to_froms, WGT)

        # Get the mapped_df.  Note that LazyFrames cannot be divided by one-another, so collect().
        mapped_contribs = mapped_contribs_lf.collect()
        mapped_weights = mapped_weights_lf.collect()
        mapped_lf = (
            # Calulate the returns by dividing contribs / weights.
            (
                (mapped_contribs / mapped_weights)
                .lazy()
                .fill_nan(0.0)
                .fill_null(0.0)
                .rename(lambda column_name: f"{column_name[:-4]}.ret")
            )
            # Add the weights
            .with_columns(mapped_weights)
            # Add the dates
            .with_columns(performance.df[cols.BEGINNING_DATE, cols.ENDING_DATE])
        )

        # Return the new mapped Performance.
        return Performance(
            mapped_lf.collect(), name=performance.name, classification_name=to_classification_name
        )
