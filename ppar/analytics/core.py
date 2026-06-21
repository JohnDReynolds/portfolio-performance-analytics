"""Coordinate analytics for portfolio and benchmark performance data.

This module provides the Analytics class, which reads portfolio and benchmark
Performance data, aligns both data sets to common subperiods, optionally
consolidates those subperiods to the requested frequency, and exposes cached
Attribution and RiskStatistics objects.
"""

# Python Imports
import bisect
import datetime as dt
from typing import cast, Protocol, Sequence

# Third-Party Imports
import polars as pl

# Project Imports
from ppar.analytics.attribution import Attribution
from ppar.analytics.frequency import Frequency, date_matches_frequency
from ppar.analytics.mapping import Mapping
from ppar.analytics.performance import Performance
from ppar.analytics.riskstatistics import RiskStatistics
import ppar.schema as cols
from ppar.errors import PpaError
import ppar.utilities as util


class AttributionSources(Protocol):  # pylint: disable=too-few-public-methods
    """Describe bundled classification sources accepted by Analytics."""

    @property
    def classification_name(self) -> str | None:
        """Return the target classification name."""
        raise NotImplementedError

    @property
    def classification_data_source(self) -> util.ClassificationDataSource | None:
        """Return the classification data source."""
        raise NotImplementedError

    @property
    def mapping_data_sources(self) -> Sequence[util.MappingDataSource | None] | None:
        """Return optional mapping data sources."""
        raise NotImplementedError


class Analytics:  # pylint: disable=too-many-instance-attributes
    """Coordinate attribution and risk-statistics calculations.

    Analytics validates and aligns portfolio and benchmark Performance data, then
    consolidates that data to the requested reporting frequency. It acts as the
    public entry point for attribution and risk-statistics calculations.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        self,
        # Portfolio and Benchmark parameters
        portfolio_data_source: util.PerformanceDataSource,
        benchmark_data_source: util.PerformanceDataSource | None = None,
        portfolio_name: str | None = None,
        benchmark_name: str | None = None,
        portfolio_classification_name: str | None = None,
        benchmark_classification_name: str | None = None,
        # Date and frequency parameters
        from_date: str | dt.date = dt.date.min,
        thru_date: str | dt.date = dt.date.max,
        frequency: Frequency = Frequency.AS_OFTEN_AS_POSSIBLE,
        default_attribution_sources: AttributionSources | None = None,
        # RiskStatistics parameters
        annual_minimum_acceptable_return: float = util.DEFAULT_ANNUAL_MINIMUM_ACCEPTABLE_RETURN,
        annual_risk_free_rate: float = util.DEFAULT_ANNUAL_RISK_FREE_RATE,
        confidence_level: float = util.DEFAULT_CONFIDENCE_LEVEL,
        portfolio_value: tuple[float, str] = (
            util.DEFAULT_PORTFOLIO_VALUE,
            util.DEFAULT_CURRENCY_SYMBOL,
        ),
    ):
        """Initialize an Analytics instance.

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
            from_date: Earliest allowed from date, either as a
                ``datetime.date`` or a string in ``yyyy-mm-dd`` format.
            thru_date: Latest allowed thru date, either as a ``datetime.date`` or
                a string in ``yyyy-mm-dd`` format.
            frequency: Reporting frequency used to consolidate subperiods.
            default_attribution_sources: Optional bundled classification source used
                when ``get_attribution()`` is called without explicit attribution
                source arguments.
            annual_minimum_acceptable_return: Annual minimum acceptable return used in
                downside-risk calculations.
            annual_risk_free_rate: Annual risk-free rate used in risk statistics that
                require a risk-free return.
            confidence_level: Confidence level used when calculating value at risk.
            portfolio_value: Tuple containing the portfolio value and its currency
                symbol for value-at-risk calculations.

        Data Parameters:
            ``portfolio_data_source`` and ``benchmark_data_source`` use the
            narrow layout below. For each time period, weights must sum to
            1.0. Column order and row order do not matter. The ``name`` column
            is optional.

            Narrow layout::

                from_date, thru_date, identifier, return, weight, name
                2024-01-01, 2024-01-31, AAPL, -0.0422272121, 0.4, Apple Inc.
                2024-01-01, 2024-01-31, MSFT,  0.0572811503, 0.6, Microsoft

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
        if benchmark_data_source is None or (
            isinstance(benchmark_data_source, str) and not benchmark_data_source.strip()
        ):
            benchmark_data_source = portfolio_data_source
            benchmark_name = portfolio_name
            benchmark_classification_name = portfolio_classification_name

        # Convert the dates to dt.date types.
        from_date = util.convert_to_date(from_date)
        thru_date = util.convert_to_date(thru_date)

        # Set the simple class variables directly from the constructor parameters.
        self._annual_minimum_acceptable_return = annual_minimum_acceptable_return
        self._annual_risk_free_rate = annual_risk_free_rate
        self._confidence_level = confidence_level
        self._default_attribution_sources = default_attribution_sources
        self._frequency = frequency
        self._portfolio_value = portfolio_value

        # Initialize the internal data structures.
        self._attributions: dict[str | None, Attribution] = {}  # key = classification_name
        self._riskstatistics: RiskStatistics | None = None

        # Get a tuple of the two Performance objects. 0 = portfolio, 1 = benchmark.
        self._performances = (
            # Portfolio
            Performance(
                portfolio_data_source,
                name=portfolio_name,
                classification_name=portfolio_classification_name,
                from_date=from_date,
                thru_date=thru_date,
            ),
            # Benchmark
            Performance(
                benchmark_data_source,
                name=benchmark_name,
                classification_name=benchmark_classification_name,
                from_date=from_date,
                thru_date=thru_date,
            ),
        )

        # Get the from dates and thru dates for all subperiods that are common between the
        # two Performance objects.
        self._subperiod_dates = self._calculate_subperiod_dates(
            f"from {util.date_str(from_date)} to {util.date_str(thru_date)}"
        )

        # Now that the dates have been firmly established, remove extraneous dates from
        # the Performance objects.
        for perf in self._performances:
            perf.reset_narrow_df(
                perf.narrow_df.lazy()
                .filter(
                    (self._from_date() <= pl.col(cols.THRU_DATE))
                    & (pl.col(cols.THRU_DATE) <= self._thru_date())
                )
                .collect()
            )

        # Consolidate multiple subperiods (e.g. daily) into single periods (e.g. monthly) based on
        # self._frequency.
        self._consolidate_all_subperiods()

    def audit(self) -> None:
        """Audit the Analytics instance.

        Audits the original portfolio and benchmark Performance objects, then audits
        any Attribution objects that have already been created and cached.

        Raises:
            PpaError: If any underlying Performance or Attribution audit fails.
        """
        # Audit the portfolio/benchmark pair of performances.  These are the performances that
        # were originally read in the constructor. Depending on their classifications, they may
        # be different than the performances in the attributions.
        Performance.audit_performances(
            self._performances, self._from_date(), self._thru_date()
        )

        # Audit the attributions and their associated performances.
        Attribution.audit_attributions(list(self._attributions.values()))

    def _from_date(self) -> dt.date:
        """Return the overall from date.

        Returns:
            The first from date in the aligned subperiod date range.
        """
        return self._subperiod_dates[0][0]

    def _calculate_subperiod_dates(self, message_suffix: str) -> list[tuple[dt.date, dt.date]]:
        """Calculate common subperiod dates for portfolio and benchmark data.

        Finds from and thru dates that exist in both Performance objects,
        optionally filters those dates to match ``self._frequency``, and pairs each
        from date with the next valid thru date.

        Args:
            message_suffix: Suffix to include in the ``PpaError`` message when no
                valid subperiods are found.

        Returns:
            A list of ``(from_date, thru_date)`` tuples for the aligned
            subperiods.

        Raises:
            PpaError: If no common subperiods can be calculated.
        """

        def _common_dates(dates1: pl.Series, dates2: pl.Series) -> pl.Series:
            """Return sorted dates that are present in both input series.

            Args:
                dates1: First date series.
                dates2: Second date series.

            Returns:
                Sorted Polars Series containing dates common to both inputs.
            """
            return dates1.filter(dates1.is_in(dates2.to_list())).sort()

        def _filter_dates_on_frequency(dates: pl.Series | list[dt.date]) -> list[dt.date]:
            """Return dates that match the Analytics reporting frequency.

            Args:
                dates: Dates to filter.

            Returns:
                List of dates that satisfy ``date_matches_frequency`` for
                ``self._frequency``.
            """
            return [date for date in dates if date_matches_frequency(date, self._frequency)]

        # Cache one row per reporting period from each performance stream.
        df0 = self._performances[0].period_totals()
        df1 = self._performances[1].period_totals()

        common_from_dates: pl.Series | list[dt.date] = _common_dates(
            df0[cols.FROM_DATE], df1[cols.FROM_DATE]
        )
        common_thru_dates: pl.Series | list[dt.date] = _common_dates(
            df0[cols.THRU_DATE], df1[cols.THRU_DATE]
        )

        if self._frequency != Frequency.AS_OFTEN_AS_POSSIBLE:
            common_thru_dates = _filter_dates_on_frequency(common_thru_dates)
            subperiod_dates: list[tuple[dt.date, dt.date]] = []
            next_from_date = max(
                cast(dt.date, df0[cols.FROM_DATE].min()),
                cast(dt.date, df1[cols.FROM_DATE].min()),
            )
            for thru_date in common_thru_dates:
                if next_from_date <= thru_date:
                    subperiod_dates.append((next_from_date, thru_date))
                    next_from_date = thru_date + dt.timedelta(days=1)
        else:
            subperiod_dates = []
            idx = 0
            len_common_thru_dates = len(common_thru_dates)
            for start_date in common_from_dates:
                if idx < len_common_thru_dates and common_thru_dates[idx] < start_date:
                    idx = bisect.bisect_left(common_thru_dates, start_date, lo=idx + 1)
                if idx < len_common_thru_dates:
                    subperiod_dates.append((start_date, common_thru_dates[idx]))
                    idx += 1

        # Assert that there is at least one subperiod.
        if len(subperiod_dates) == 0:
            raise PpaError(message_suffix, 202)

        # Return the common from and thru dates that define the subperiods.
        return subperiod_dates

    def classification_names(self) -> tuple[str | None, str | None]:
        """Return the portfolio and benchmark classification names.

        Returns:
            A two-item tuple where item 0 is the portfolio classification name and
            item 1 is the benchmark classification name.
        """
        return (
            self._performances[0].classification_name,
            self._performances[1].classification_name,
        )

    def _consolidate_all_subperiods(self) -> None:
        """Consolidate portfolio and benchmark data to the aligned subperiods.

        For each Performance object, verifies that enough rows exist for the aligned
        subperiods. If the Performance contains more rows than the aligned subperiod
        list, consolidates the extra rows to the requested frequency.

        Raises:
            PpaError: If a Performance has fewer rows than the calculated subperiod
                date list.
        """
        # Iterate through the portfolio and benchmark Performance objects.
        for performance in self._performances:
            quantity_of_periods = performance.narrow_df.select(cols.DATE_COLUMNS).unique().height
            if quantity_of_periods < len(self._subperiod_dates):
                raise PpaError(
                    f"{performance.error_message_context} from "
                    f"{util.date_str(self._from_date())} "
                    f"to {util.date_str(self._thru_date())}",
                    999,
                )

            if len(self._subperiod_dates) < quantity_of_periods:
                performance.reset_narrow_df(self._consolidate_subperiods(performance))

    def _consolidate_subperiods(self, performance: Performance) -> pl.DataFrame:
        """Consolidate a Performance object into the aligned subperiods.

        Combines multiple source rows, such as daily rows, into the subperiods stored
        in ``self._subperiod_dates``. Returns are geometrically linked, weights are
        summed using day-weighting coefficients, and contributions are summed after
        applying logarithmic linking coefficients.

        Args:
            performance: Performance instance to consolidate.

        Returns:
            DataFrame containing one narrow calculated row per identifier in
            each aligned subperiod.
        """
        # Create a DataFrame, one row per subperiod.
        subperiods = (
            pl.DataFrame(
                {
                    "_subperiod_from_date": [bd for bd, _ in self._subperiod_dates],
                    "_subperiod_thru_date": [ed for _, ed in self._subperiod_dates],
                }
            )
            .with_row_index(name="subperiod_id")
            .with_columns(
                (
                    (
                        pl.col("_subperiod_thru_date")
                        - pl.col("_subperiod_from_date")
                    ).dt.total_days()
                    + 1
                ).alias("_subperiod_quantity_of_days")
            )
            .sort("_subperiod_from_date")
        )

        source_periods = performance.narrow_df.select(
            *cols.DATE_COLUMNS,
            cols.QUANTITY_OF_DAYS,
            cols.TOTAL_RETURN,
        ).unique().sort(cols.THRU_DATE)
        assigned_periods = source_periods.join_asof(
            subperiods,
            left_on=cols.THRU_DATE,
            right_on="_subperiod_from_date",
            strategy="backward",
        )

        # A reporting-period total return must compound the lower-frequency rows.
        # A +10% day followed by a -10% day is -1%, not 0%.
        subperiod_returns = assigned_periods.group_by("subperiod_id").agg(
            pl.col(cols.TOTAL_RETURN).add(1).product().sub(1).alias("subperiod_return")
        )
        subperiod_weight_denominators = assigned_periods.group_by("subperiod_id").agg(
            pl.col(cols.QUANTITY_OF_DAYS).sum().alias("_weight_denominator_days")
        )

        assigned_rows = (
            performance.narrow_df.join(
                assigned_periods.select(
                    *cols.DATE_COLUMNS,
                    "subperiod_id",
                    "_subperiod_from_date",
                    "_subperiod_thru_date",
                    "_subperiod_quantity_of_days",
                ),
                on=cols.DATE_COLUMNS,
            )
            .join(subperiod_returns, on="subperiod_id")
            .join(subperiod_weight_denominators, on="subperiod_id")
            .with_columns(
                # Weights are interpreted as period exposures, so consolidation averages
                # them by elapsed days instead of taking the first or last holding weight.
                (
                    pl.col(cols.QUANTITY_OF_DAYS)
                    / pl.col("_weight_denominator_days")
                ).alias("weight_coefficient"),
                # Contributions are linked so their sum over source rows equals the
                # geometrically linked reporting-period return. This preserves the
                # additive attribution story while returns themselves compound.
                pl.struct(["subperiod_return", cols.TOTAL_RETURN])
                .map_batches(
                    lambda x: util.logarithmic_linking_coefficient_series(
                        x.struct.field("subperiod_return"),
                        x.struct.field(cols.TOTAL_RETURN),
                    ),
                    return_dtype=pl.Float64,
                )
                .alias("linking_coefficient"),
            )
        )

        consolidated_subperiods = (
            assigned_rows.group_by(["subperiod_id", cols.IDENTIFIER])
            .agg(
                pl.col("_subperiod_from_date").first().alias(cols.FROM_DATE),
                pl.col("_subperiod_thru_date").first().alias(cols.THRU_DATE),
                pl.col("_subperiod_quantity_of_days")
                .first()
                .alias(cols.QUANTITY_OF_DAYS),
                pl.col("subperiod_return").first().alias(cols.TOTAL_RETURN),
                pl.col(cols.RETURN).add(1).product().sub(1).alias(cols.RETURN),
                (pl.col(cols.WEIGHT) * pl.col("weight_coefficient")).sum().alias(cols.WEIGHT),
                (pl.col(cols.CONTRIBUTION) * pl.col("linking_coefficient"))
                .sum()
                .alias(cols.CONTRIBUTION),
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

        performance.subperiods_have_been_consolidated = True

        return consolidated_subperiods

    def _thru_date(self) -> dt.date:
        """Return the overall thru date.

        Returns:
            The last thru date in the aligned subperiod date range.
        """
        return self._subperiod_dates[-1][-1]

    def get_attribution(
        self,
        classification_name: str | None = None,
        classification_data_source: util.ClassificationDataSource | None = None,
        mapping_data_sources: Sequence[util.MappingDataSource | None] | None = None,
        classification_label: str | None = None,
    ) -> Attribution:
        """Return an Attribution instance for the requested classification.

        Returns a cached Attribution object when available. If no attribution source
        arguments are supplied and the Analytics instance was initialized with default
        attribution sources, those defaults are used. Otherwise, maps portfolio and/or
        benchmark Performance objects to the requested classification when needed,
        creates the Attribution object, stores it in the cache, and returns it.

        Args:
            classification_name: Classification name for the requested Attribution.
                If omitted and both Performance objects share a common non-empty
                classification name, that common name is used.
            classification_data_source: Optional classification data source. This can
                be a CSV file path, dictionary, pandas DataFrame, or Polars DataFrame.
            mapping_data_sources: Optional two-item sequence of mapping data sources
                where item 0 maps the portfolio and item 1 maps the benchmark. Each
                source can be a CSV file path, dictionary, pandas DataFrame, or Polars
                DataFrame; use ``None`` when a performance already uses the target
                classification.
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
        if isinstance(classification_data_source, str) and not classification_data_source.strip():
            classification_data_source = None
        if (
            classification_name is None
            and classification_data_source is None
            and mapping_data_sources is None
            and self._default_attribution_sources is not None
        ):
            return self.get_attribution_for(
                self._default_attribution_sources,
                classification_label=classification_label,
            )
        if mapping_data_sources is None:
            mapping_data_sources = (None, None)
        else:
            mapping_data_sources = tuple(
                None if isinstance(source, str) and not source.strip() else source
                for source in mapping_data_sources
            )

        # If the classification name is omitted and the portfolio and benchmark share a
        # non-empty classification name, use that common classification.
        if (
            classification_name is None
            and self._performances[0].classification_name is not None
            and self._performances[0].classification_name
            == self._performances[1].classification_name
        ):
            classification_name = self._performances[0].classification_name

        # If the target classification is unknown but either source Performance has a
        # known classification, require an explicit target. This still allows all
        # classifications to be unknown for identifier-level attribution.
        if classification_name is None and (
            (self._performances[0].classification_name is not None)
            or (self._performances[1].classification_name is not None)
        ):
            raise PpaError("", 252)

        # Return the attribution if it already exists in the cache.
        if classification_name in self._attributions:
            return self._attributions[classification_name]

        # Get the performances for the common classification_name.
        if classification_name is None:
            attribution_performances = list(self._performances)
        else:
            attribution_performances = [
                (
                    perf
                    if perf.classification_name == classification_name
                    else self._map_performance(
                        perf, classification_name, mapping_data_sources[idx]
                    )
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

        # Return the Attribution corresponding to classification_name.
        return self._attributions[classification_name]

    def get_attribution_for(
        self,
        sources: AttributionSources,
        classification_label: str | None = None,
    ) -> Attribution:
        """Return an Attribution instance from a bundled source object.

        Args:
            sources: Object containing a classification name, classification data
                source, and optional mapping data sources. Axys classification
                source bundles implement this shape.
            classification_label: Display label used in tables and charts when the
                classification name is empty and the Performance classification items
                are used directly.

        Returns:
            Attribution instance associated with ``sources.classification_name``.

        Raises:
            PpaError: If the bundled source values cannot produce the requested
                Attribution.
        """
        return self.get_attribution(
            classification_name=sources.classification_name,
            classification_data_source=sources.classification_data_source,
            mapping_data_sources=sources.mapping_data_sources,
            classification_label=classification_label,
        )

    def get_riskstatistics(self) -> RiskStatistics:
        """Return risk statistics for the aligned Performance objects.

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

    def _map_performance(
        self,
        performance: Performance,
        to_classification_name: str,
        mapping_data_source: util.MappingDataSource | None,
    ) -> Performance:
        """Map a Performance object to a different classification.

        Uses the supplied mapping data to roll up contribution and weight columns from
        ``performance.classification_name`` to ``to_classification_name``. Mapped
        returns are calculated as mapped contributions divided by mapped weights, with
        missing or undefined mapped returns filled with 0.0.

        Args:
            performance: Existing Performance object to map.
            to_classification_name: Target classification name.
            mapping_data_source: Mapping data source used to map source identifiers to
                target classification items. Must be provided when mapping is needed.

        Data Parameters:
            Example mapping data for Security to Economic Sector::

                AAPL, IT
                GOOG, CO

        Returns:
            New Performance object using ``to_classification_name``.

        Raises:
            PpaError: If the Mapping or resulting Performance cannot be created or
                validated, or if mapping is required but no mapping source is supplied.
        """
        if mapping_data_source is None:
            raise PpaError(util.file_path_error(""), None)

        to_froms = Mapping(
            performance.identifiers,
            mapping_data_source,
        ).to_froms
        to_identifier_by_from = {
            from_identifier: to_identifier
            for to_identifier, from_identifiers in to_froms.items()
            for from_identifier in from_identifiers
        }
        mapped = (
            performance.narrow_df.with_columns(
                pl.col(cols.IDENTIFIER)
                .replace_strict(to_identifier_by_from)
                .alias(cols.IDENTIFIER)
            )
            .group_by([*cols.DATE_COLUMNS, cols.IDENTIFIER])
            .agg(
                pl.col(cols.WEIGHT).sum(),
                pl.col(cols.CONTRIBUTION).sum(),
                pl.col(cols.QUANTITY_OF_DAYS).first(),
                pl.col(cols.TOTAL_RETURN).first(),
            )
            .with_columns(
                pl.when(pl.col(cols.WEIGHT) != 0.0)
                .then(pl.col(cols.CONTRIBUTION) / pl.col(cols.WEIGHT))
                .otherwise(0.0)
                .fill_nan(0.0)
                .alias(cols.RETURN)
            )
            .select(
                *cols.DATE_COLUMNS,
                cols.IDENTIFIER,
                cols.WEIGHT,
                cols.RETURN,
            )
        )

        return Performance(
            mapped, name=performance.name, classification_name=to_classification_name
        )
