"""
Analytics orchestration for portfolio and benchmark performance data.

This module provides the Analytics class, which reads portfolio and benchmark
Performance data, aligns both data sets to common subperiods, optionally
consolidates those subperiods to the requested frequency, and exposes cached
Attribution and RiskStatistics objects.
"""

# Python Imports
import bisect
from collections import defaultdict
import datetime as dt
from typing import Sequence

# Third-Party Imports
import polars as pl

# Project Imports
from ppar.attribution import Attribution
import ppar.columns as cols
from ppar.columns import CON, RET, WGT
from ppar.errors import PpaError
from ppar.frequency import Frequency, date_matches_frequency
from ppar.mapping import Mapping
from ppar.performance import Performance
from ppar.riskstatistics import RiskStatistics
import ppar.utilities as util


class Analytics:
    """
    Coordinate attribution and risk-statistics calculations.

    Analytics validates and aligns portfolio and benchmark Performance data, then
    consolidates that data to the requested reporting frequency. It acts as the
    public entry point for attribution and risk-statistics calculations.
    """

    def __init__(
        self,
        # Portfolio and Benchmark parameters
        portfolio_data_source: util.PerformanceDataSource,
        benchmark_data_source: util.PerformanceDataSource | None = None,
        portfolio_name: str | None = None,
        benchmark_name: str | None = None,
        portfolio_classification_name: str | None = None,
        benchmark_classification_name: str | None = None,
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
        Initialize an Analytics instance.

        Reads portfolio and benchmark performance data, converts the requested date
        bounds to ``datetime.date`` values, aligns the two performance data sets to
        common subperiods, and consolidates subperiods according to ``frequency``.
        When no benchmark data source is supplied, the portfolio data source is reused
        as the benchmark data source so portfolio-only analytics can be calculated.

        Args:
            portfolio_data_source: Portfolio performance data source. This can be a
                CSV file path, a pandas DataFrame, or a Polars DataFrame.
            benchmark_data_source: Benchmark performance data source. This can be a
                CSV file path, a pandas DataFrame, or a Polars DataFrame. Defaults to
                ``None``, which causes the portfolio data source to be reused.
            portfolio_name: Portfolio display name used in output titles.
            benchmark_name: Benchmark display name used in output titles.
            portfolio_classification_name: Classification name associated with the
                portfolio performance data.
            benchmark_classification_name: Classification name associated with the
                benchmark performance data.
            beginning_date: Earliest allowed beginning date, either as a
                ``datetime.date`` or a string in ``yyyy-mm-dd`` format.
            ending_date: Latest allowed ending date, either as a ``datetime.date`` or
                a string in ``yyyy-mm-dd`` format.
            frequency: Reporting frequency used to consolidate subperiods.
            annual_minimum_acceptable_return: Annual minimum acceptable return used in
                downside-risk calculations.
            annual_risk_free_rate: Annual risk-free rate used in risk statistics that
                require a risk-free return.
            confidence_level: Confidence level used when calculating value at risk.
            portfolio_value: Tuple containing the portfolio value and its currency
                symbol for value-at-risk calculations.

        Data Parameters:
            ``portfolio_data_source`` and ``benchmark_data_source`` can use either the
            narrow or wide layouts below. For each time period, weights must sum to
            1.0 and ``sum(weight * return)`` must equal the total return. Column order
            and row order do not matter. The ``name`` column is optional.

            Narrow layout::

                beginning_date, ending_date, identifier, return, weight, name
                2023-12-31, 2024-01-31, AAPL, -0.0422272121, 0.4, Apple Inc.
                2023-12-31, 2024-01-31, MSFT,  0.0572811503, 0.6, Microsoft

            Wide layout::

                beginning_date, ending_date, AAPL.ret, MSFT.ret, AAPL.wgt, MSFT.wgt
                2023-12-31, 2024-01-31, -0.0422272121, 0.0572811503, 0.4, 0.6

        Raises:
            PpaError: If either date cannot be converted, the portfolio and benchmark
                do not share any valid subperiods, there are too few performance rows
                for the calculated subperiods, or a nested Performance validation
                raises ``PpaError``.
        """
        portfolio_name = util.normalize_optional_string(portfolio_name)
        benchmark_name = util.normalize_optional_string(benchmark_name)
        portfolio_classification_name = util.normalize_optional_string(
            portfolio_classification_name
        )
        benchmark_classification_name = util.normalize_optional_string(
            benchmark_classification_name
        )

        # Default the benchmark to the portfolio.  This will allow for "portfolio-only" analysis
        # if they do not have a benchmark.
        if benchmark_data_source is None or util.is_empty_string(benchmark_data_source):
            benchmark_data_source = portfolio_data_source
            benchmark_name = portfolio_name
            benchmark_classification_name = portfolio_classification_name

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
        """
        Audit the Analytics instance.

        Audits the original portfolio and benchmark Performance objects, then audits
        any Attribution objects that have already been created and cached.

        Raises:
            PpaError: If any underlying Performance or Attribution audit fails.
        """
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
        Return the overall beginning date.

        Returns:
            The first beginning date in the aligned subperiod date range.
        """
        return self._subperiod_dates[0][0]

    def _calculate_subperiod_dates(self, message_suffix: str) -> list[tuple[dt.date, dt.date]]:
        """
        Calculate common subperiod dates for portfolio and benchmark data.

        Finds beginning and ending dates that exist in both Performance objects,
        optionally filters those dates to match ``self._frequency``, and pairs each
        beginning date with the next strictly later ending date.

        Args:
            message_suffix: Suffix to include in the ``PpaError`` message when no
                valid subperiods are found.

        Returns:
            A list of ``(beginning_date, ending_date)`` tuples for the aligned
            subperiods.

        Raises:
            PpaError: If no common subperiods can be calculated.
        """

        def _common_dates(dates1: pl.Series, dates2: pl.Series) -> pl.Series:
            """
            Return sorted dates that are present in both input series.

            Args:
                dates1: First date series.
                dates2: Second date series.

            Returns:
                Sorted Polars Series containing dates common to both inputs.
            """
            # Note that using set intersection is MUCH slower.
            # return sorted(set(dates1) & set(dates2))
            return dates1.filter(dates1.is_in(dates2.to_list())).sort()

        def _filter_dates_on_frequency(dates: pl.Series | list[dt.date]) -> list[dt.date]:
            """
            Return dates that match the Analytics reporting frequency.

            Args:
                dates: Dates to filter.

            Returns:
                List of dates that satisfy ``date_matches_frequency`` for
                ``self._frequency``.
            """
            return [date for date in dates if date_matches_frequency(date, self._frequency)]

        # Cache the performance DataFrames.
        df0 = self._performances[0].df
        df1 = self._performances[1].df

        # Compute sorted common beginning and ending dates separately. This ensures
        # the aligned subperiods use only dates that are present in both the
        # portfolio and benchmark streams, and that beginning dates are paired with
        # valid later ending dates.
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
        if len(subperiod_dates) == 0:
            raise PpaError(message_suffix, 202)

        # Return the common beginning and ending dates that define the subperiods.
        return subperiod_dates

    def classification_names(self) -> tuple[str, str]:
        """
        Return the portfolio and benchmark classification names.

        Returns:
            A two-item tuple where item 0 is the portfolio classification name and
            item 1 is the benchmark classification name.
        """
        return (
            self._performances[0].classification_name,
            self._performances[1].classification_name,
        )

    def _consolidate_all_subperiods(self) -> None:
        """
        Consolidate portfolio and benchmark data to the aligned subperiods.

        For each Performance object, verifies that enough rows exist for the aligned
        subperiods. If the Performance contains more rows than the aligned subperiod
        list, consolidates the extra rows to the requested frequency.

        Raises:
            PpaError: If a Performance has fewer rows than the calculated subperiod
                date list.
        """
        # Iterate through the portfolio and benchmark Performances.
        for performance in self._performances:
            # Assert that performance.df has at least the same quantity of rows as
            # self._subperiod_dates.
            if performance.df.shape[0] < len(self._subperiod_dates):
                raise PpaError(
                    f"{performance.error_message_context} from "
                    f"{util.date_str(self._beginning_date())} "
                    f"to {util.date_str(self._ending_date())}",
                    999,
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
        Consolidate a Performance object into the aligned subperiods.

        Combines multiple source rows, such as daily rows, into the subperiods stored
        in ``self._subperiod_dates``. Returns are geometrically linked, weights are
        summed using day-weighting coefficients, and contributions are summed after
        applying logarithmic linking coefficients.

        Args:
            performance: Performance instance to consolidate.

        Returns:
            LazyFrame containing one consolidated row for each aligned subperiod.
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

        # Assign each source period to the reporting period that owns its beginning
        # date. For example, daily rows beginning after 2024-01-31 and before
        # 2024-02-29 are grouped into the February monthly report ending 2024-02-29.
        joined_lf = performance.df.lazy().join_asof(
            df_subperiods,
            left_on=cols.BEGINNING_DATE,
            right_on="beg_date",
            strategy="backward",
            by=None,
        )

        # A reporting-period total return must compound the lower-frequency rows.
        # A +10% day followed by a -10% day is -1%, not 0%.
        subperiod_returns = joined_lf.group_by("subperiod_id").agg(
            [pl.col(cols.TOTAL_RETURN).add(1).cum_prod().last().sub(1).alias("subperiod_return")]
        )

        # The period return is needed beside each source row so each row can get
        # its own contribution-linking coefficient inside the reporting period.
        joined_df = joined_lf.join(subperiod_returns, on="subperiod_id").collect()

        joined_lf = joined_df.lazy().with_columns(
            # Weights are interpreted as period exposures, so consolidation averages
            # them by elapsed days instead of taking the first or last holding weight.
            (
                joined_df[cols.QUANTITY_OF_DAYS]
                / (joined_df["end_date"] - joined_df["beg_date"]).dt.total_days()
            ).alias("weight_coefficient"),
            # Contributions are linked so their sum over source rows equals the
            # geometrically linked reporting-period return. This preserves the
            # additive attribution story while returns themselves compound.
            pl.struct(["subperiod_return", cols.TOTAL_RETURN])
            .map_batches(
                lambda x: util.logarithmic_linking_coefficient_series(
                    x.struct.field("subperiod_return"), x.struct.field(cols.TOTAL_RETURN)
                ),
                return_dtype=pl.Float64,
            )
            .alias("linking_coefficient"),
        )

        # The consolidated row keeps one return/weight/contribution triplet per
        # identifier. Returns are linked, weights are time-weighted, and contributions
        # are log-linked so downstream attribution can still foot to total return.
        consolidated_subperiods_lf = (
            joined_lf.group_by("subperiod_id")
            .agg(
                [
                    pl.col("beg_date").first().alias(cols.BEGINNING_DATE),
                    pl.col("end_date").first().alias(cols.ENDING_DATE),
                    pl.col(cols.QUANTITY_OF_DAYS).sum(),
                    pl.col(cols.TOTAL_RETURN).add(1).cum_prod().last().sub(1),
                    pl.col(performance.col_names(RET)).add(1).cum_prod().tail(1).sub(1).first(),
                    pl.col(performance.col_names(WGT)).mul(pl.col("weight_coefficient")).sum(),
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
        Return the overall ending date.

        Returns:
            The last ending date in the aligned subperiod date range.
        """
        return self._subperiod_dates[-1][-1]

    def get_attribution(
        self,
        classification_name: str | None = None,
        classification_data_source: util.ClassificationDataSource | None = None,
        mapping_data_sources: Sequence[util.MappingDataSource] | None = None,
        classification_label: str | None = None,
    ) -> Attribution:
        """
        Return an Attribution instance for the requested classification.

        Returns a cached Attribution object when available. Otherwise, maps portfolio
        and/or benchmark Performance objects to the requested classification when
        needed, creates the Attribution object, stores it in the cache, and returns it.

        Args:
            classification_name: Classification name for the requested Attribution.
                If omitted and both Performance objects share a common non-empty
                classification name, that common name is used.
            classification_data_source: Optional classification data source. This can
                be a CSV file path, dictionary, pandas DataFrame, or Polars DataFrame.
            mapping_data_sources: Two-item sequence of mapping data sources where item 0
                maps the portfolio and item 1 maps the benchmark. Each source can be a
                CSV file path, dictionary, pandas DataFrame, or Polars DataFrame.
            classification_label: Display label used in tables and charts when the
                classification name is empty and the Performance classification items
                are used directly.

        Data Parameters:
            Example ``classification_data_source`` for a Security classification::

                AAPL, Apple Inc.
                MSFT, Microsoft

            Example ``mapping_data_sources`` data for Security to Economic Sector::

                AAPL, IT
                GOOG, CS

        Returns:
            Attribution instance associated with ``classification_name``.

        Raises:
            PpaError: If ``classification_name`` is required because at least one
                Performance has a known classification name but no target
                classification name is supplied, or if a nested Mapping, Performance,
                or Attribution operation raises ``PpaError``.
        """
        classification_name = util.normalize_optional_string(classification_name)
        classification_label = util.normalize_optional_string(classification_label)
        if classification_data_source is None or util.is_empty_string(classification_data_source):
            classification_data_source = util.EMPTY
        if mapping_data_sources is None:
            mapping_data_sources = (util.EMPTY, util.EMPTY)

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
        if util.is_empty(classification_name) and (
            (not util.is_empty(self._performances[0].classification_name))
            or (not util.is_empty(self._performances[1].classification_name))
        ):
            raise PpaError("", 252)

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
        Return risk statistics for the aligned Performance objects.

        Creates and caches a RiskStatistics instance on first use, then returns the
        cached instance on subsequent calls.

        Returns:
            RiskStatistics instance for the portfolio and benchmark Performance data.
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
        Aggregate Performance columns according to a reverse mapping.

        Horizontally sums contribution or weight columns from source classification
        items into target classification items. Large mappings are processed in
        batches to reduce Polars memory pressure.

        Args:
            performance: Performance object containing columns to aggregate.
            to_froms: Reverse mapping from each target classification item to the
                source classification items that should be summed into it.
            suffix: Column suffix to aggregate, such as ``CON`` or ``WGT``.

        Returns:
            LazyFrame containing the aggregated mapped columns.
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
        mapping_data_source: util.MappingDataSource,
    ) -> Performance:
        """
        Map a Performance object to a different classification.

        Uses the supplied mapping data to roll up contribution and weight columns from
        ``performance.classification_name`` to ``to_classification_name``. Mapped
        returns are calculated as mapped contributions divided by mapped weights, with
        missing or undefined mapped returns filled with 0.0.

        Args:
            performance: Existing Performance object to map.
            to_classification_name: Target classification name.
            mapping_data_source: Mapping data source used to map source identifiers to
                target classification items.

        Data Parameters:
            Example mapping data for Security to Economic Sector::

                AAPL, IT
                GOOG, CO

        Returns:
            New Performance object using ``to_classification_name``.

        Raises:
            PpaError: If the Mapping or resulting Performance cannot be created or
                validated.
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
