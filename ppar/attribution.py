"""Compute and expose Brinson-Fachler attribution results.

This module defines the :class:`Attribution` class, chart/view enumerations, and
helpers used to calculate portfolio-versus-benchmark contribution, allocation,
selection, total attribution effects, and formatted output.

Attribution instances are normally created by
:meth:`ppar.analytics.Analytics.get_attribution`.
"""

# Overrides for pylint
# pylint: disable=too-many-lines


# Python Imports
from enum import Enum
import datetime as dt
from typing import cast, Iterable, Sequence

# Third-Party Imports
import numpy as np
import pandas as pd
import polars as pl

# Project Imports
from ppar.classification import Classification
import ppar.columns as cols
from ppar.errors import PpaError
from ppar.frequency import Frequency
from ppar import html_table
from ppar import output
from ppar.performance import Performance
import ppar.utilities as util

# Constants
_DEFAULT_OUTPUT_PRECISION = 8


class Chart(Enum):
    """Attribution chart types supported by :meth:`Attribution.to_chart`.

    Each enum value is the display label used in chart titles.

    Attributes:
        CUMULATIVE_ATTRIBUTION: Cumulative attribution-effects chart.
        CUMULATIVE_CONTRIBUTION: Cumulative contribution chart.
        CUMULATIVE_RETURN: Cumulative returns chart.
        HEATMAP_ACTIVE_CONTRIBUTION: Active-contribution heatmap.
        HEATMAP_ACTIVE_RETURN: Active-return heatmap.
        HEATMAP_ATTRIBUTION: Total-attribution-effects heatmap.
        HEATMAP_PORTFOLIO_CONTRIBUTION: Portfolio-contribution heatmap.
        HEATMAP_PORTFOLIO_RETURN: Portfolio-return heatmap.
        OVERALL_ATTRIBUTION: Overall attribution chart.
        OVERALL_CONTRIBUTION: Overall contribution comparison chart.
        SUBPERIOD_ATTRIBUTION: Subperiod attribution-effects chart.
        SUBPERIOD_RETURN: Subperiod returns chart.
    """

    CUMULATIVE_ATTRIBUTION = "Cumulative Attribution Effects"
    CUMULATIVE_CONTRIBUTION = "Cumulative Contribution"
    CUMULATIVE_RETURN = "Cumulative Returns"
    HEATMAP_ACTIVE_CONTRIBUTION = "Active Contributions"
    HEATMAP_ACTIVE_RETURN = "Active Returns"
    HEATMAP_ATTRIBUTION = "Total Attribution Effects"
    HEATMAP_PORTFOLIO_CONTRIBUTION = "Portfolio Contributions"
    HEATMAP_PORTFOLIO_RETURN = "Portfolio Returns"
    OVERALL_ATTRIBUTION = "Overall Attribution"
    OVERALL_CONTRIBUTION = "Overall Contribution"
    SUBPERIOD_ATTRIBUTION = "Sub-Period Attribution Effects"
    SUBPERIOD_RETURN = "Sub-Period Returns"


class View(Enum):
    """Tabular attribution views supported by the output methods.

    Each enum value is the display label used in table titles and serialized output.

    Attributes:
        CUMULATIVE_ATTRIBUTION: Cumulative attribution view.
        OVERALL_ATTRIBUTION: Overall attribution view.
        SUBPERIOD_ATTRIBUTION: Per-period classified attribution view.
        SUBPERIOD_SUMMARY: Per-period summary view.
    """

    CUMULATIVE_ATTRIBUTION = "Cumulative Attribution"
    OVERALL_ATTRIBUTION = "Overall Attribution"
    SUBPERIOD_ATTRIBUTION = "Sub-Period Attribution"
    SUBPERIOD_SUMMARY = "Sub-Period Summary"


# Column names that should be equivalent between all Attribution instances for a given Analytics.
_EQUIVALENT_COLUMN_NAMES = (
    cols.BEGINNING_DATE,
    cols.ENDING_DATE,
    cols.QUANTITY_OF_DAYS,
    cols.TOTAL_RETURN,
)

# Various pairs of columns that should be equal to each other for the total row.
_OVERALL_COLUMN_PAIRS_THAT_SHOULD_BE_EQUAL = (
    # Smoothed Contributions
    (cols.PORTFOLIO_RETURN, cols.PORTFOLIO_CONTRIB_SMOOTHED),
    (cols.BENCHMARK_RETURN, cols.BENCHMARK_CONTRIB_SMOOTHED),
    (cols.ACTIVE_RETURN, cols.ACTIVE_CONTRIB_SMOOTHED),
    # Cumulative Returns
    (cols.PORTFOLIO_RETURN, cols.CUMULATIVE_PORTFOLIO_RETURN),
    (cols.BENCHMARK_RETURN, cols.CUMULATIVE_BENCHMARK_RETURN),
    (cols.ACTIVE_RETURN, cols.CUMULATIVE_ACTIVE_RETURN),
    # Cumulative Contributions
    (cols.PORTFOLIO_RETURN, cols.CUMULATIVE_PORTFOLIO_CONTRIB),
    (cols.BENCHMARK_RETURN, cols.CUMULATIVE_BENCHMARK_CONTRIB),
    (cols.ACTIVE_RETURN, cols.CUMULATIVE_ACTIVE_CONTRIB),
    # Attribution Effects
    (cols.ALLOCATION_EFFECT_SMOOTHED, cols.CUMULATIVE_ALLOCATION_EFFECT),
    (cols.SELECTION_EFFECT_SMOOTHED, cols.CUMULATIVE_SELECTION_EFFECT),
    (cols.TOTAL_EFFECT_SMOOTHED, cols.CUMULATIVE_TOTAL_EFFECT),
    # Total Effect
    (cols.ACTIVE_RETURN, cols.TOTAL_EFFECT_SMOOTHED),
    (cols.ACTIVE_RETURN, cols.CUMULATIVE_TOTAL_EFFECT),
)

# Various pairs of simple columns that should be equal to each other.
_SIMPLE_COLUMN_PAIRS_THAT_SHOULD_BE_EQUAL = (
    (cols.PORTFOLIO_RETURN, cols.PORTFOLIO_CONTRIB_SIMPLE),
    (cols.BENCHMARK_RETURN, cols.BENCHMARK_CONTRIB_SIMPLE),
    (cols.ACTIVE_RETURN, cols.ACTIVE_CONTRIB_SIMPLE),
    (cols.ACTIVE_RETURN, cols.TOTAL_EFFECT_SIMPLE),
)

# The column names associated with each View.
_VIEW_COLUMN_NAMES = {
    # View.CUMULATIVE_ATTRIBUTION
    View.CUMULATIVE_ATTRIBUTION: cols.DATE_COLUMNS + cols.VIEW_CUMULATIVE_ATTRIBUTION_COLUMNS,
    # View.OVERALL_ATTRIBUTION
    View.OVERALL_ATTRIBUTION: cols.CLASSIFICATION_COLUMNS + cols.VIEW_OVERALL_ATTRIBUTION_COLUMNS,
    # View.SUBPERIOD_ATTRIBUTION
    View.SUBPERIOD_ATTRIBUTION: cols.DATE_COLUMNS
    + cols.CLASSIFICATION_COLUMNS
    + cols.VIEW_SUBPERIOD_ATTRIBUTION_COLUMNS,
    # View.SUBPERIOD_SUMMARY
    View.SUBPERIOD_SUMMARY: cols.DATE_COLUMNS + cols.VIEW_SUBPERIOD_SUMMARY_COLUMNS,
}


class Attribution:
    """Calculate, store, audit, and format attribution results.

    An ``Attribution`` instance contains portfolio and benchmark ``Performance``
    objects, a ``Classification``, and the resulting contribution and attribution
    effects. It provides public methods for retrieving those results as charts,
    HTML, JSON, pandas DataFrames, Polars DataFrames, XML, and CSV.
    """

    def __init__(
        self,
        performances: Sequence[Performance],
        classification_name: str | None,
        classification_data_source: util.ClassificationDataSource | None,
        frequency: Frequency,
        classification_label: str | None = None,
    ):
        """Initialize an attribution calculation.

        Args:
            performances: A two-item sequence containing the portfolio ``Performance`` at
                index 0 and the benchmark ``Performance`` at index 1.
            classification_name: Optional classification name for which
                contribution and attribution effects are calculated.
            classification_data_source: Optional classification source. May be a CSV
                file path, dictionary, pandas DataFrame, or Polars DataFrame. If
                omitted, classification display data is inferred from the performances.
            frequency: Frequency associated with the attribution periods.
            classification_label: Optional label displayed in tables and charts. If
                empty, the classification name is used.

        Raises:
            PpaError: If classification setup, performance alignment, linking, or
                attribution calculation fails validation.
        """
        classification_label = util.normalize_optional_string(classification_label)

        # Set internal instance variables from the constructor parameters.
        self._classification = Classification(
            classification_name, classification_data_source, performances
        )
        self._frequency = frequency
        self._performances = performances
        self._classification_label = (
            self._classification.name
            if classification_label is None
            else classification_label
        )

        # Make sure that portfolio and benchmark include matching identifier rows.
        self._equalize_columns()

        # Create the Attribution DataFrames.
        self._df, self._detail_df = self._calculate_attribution()
        self._df_overall = self._calculate_df_overall()

    def _add_total_row(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return a DataFrame with a total row appended.

        Args:
            df: DataFrame to summarize.

        Returns:
            DataFrame with one additional bottom row containing totals or linked return
            values, depending on the available columns.
        """
        # Start the total_row as a sum of df.
        total_row = df.sum()

        # The classification identifier will have 'None', so make it blank.
        if cols.CLASSIFICATION_IDENTIFIER in df.columns:
            total_row[0, cols.CLASSIFICATION_IDENTIFIER] = None
            total_row[0, cols.CLASSIFICATION_NAME] = "Total"

        # Add the "Total" label to the total row.
        if cols.BEGINNING_DATE in df.columns:
            # Convert the date columns to strings just so "Total" can be added.
            df = df.with_columns(
                pl.col([cols.BEGINNING_DATE, cols.ENDING_DATE]).dt.strftime(
                    util.DATE_FORMAT_STRING
                )
            )
            total_row = total_row.cast(
                {cols.BEGINNING_DATE: pl.String, cols.ENDING_DATE: pl.String}
            )
            total_row[0, cols.BEGINNING_DATE] = None
            total_row[0, cols.ENDING_DATE] = "Total"

        # Override the returns since they should be linked, not summed.
        if cols.ACTIVE_RETURN in df.columns:
            total_row[0, cols.PORTFOLIO_RETURN] = self._df_overall.item(
                0, cols.PORTFOLIO_RETURN
            )
            total_row[0, cols.BENCHMARK_RETURN] = self._df_overall.item(
                0, cols.BENCHMARK_RETURN
            )
            total_row[0, cols.ACTIVE_RETURN] = (
                total_row.item(0, cols.PORTFOLIO_RETURN)
                - total_row.item(0, cols.BENCHMARK_RETURN)
            )

        # The cumulative column totals are the values in the last row.
        if cols.CUMULATIVE_TOTAL_EFFECT in df.columns:
            for cum_col_name in cols.ALL_CUMULATIVE_COLUMNS:
                total_row[0, cum_col_name] = df[-1, cum_col_name]

        # Concatenate the total_row to the bottom of the df.
        return df.vstack(total_row)

    def audit(self) -> None:
        """Audit this attribution instance for internal consistency.

        Raises:
            PpaError: If the underlying performances are invalid, detailed and overall
                DataFrames have different columns, or attribution columns fail footing
                checks.
        """
        # Audit the portfolio/benchmark pair of performance objects.
        Performance.audit_performances(
            self._performances,
            self._beginning_date(),
            self._ending_date(),
            self._classification.name,
        )

        # Assert that df and df_overall have the same columns.
        if set(self._df.columns) != set(self._df_overall.columns):
            raise PpaError("Attr.audit(): df columns != df_overall columns.", 999)

        # Audit all columns.
        Attribution._audit_columns(self._df, self._df_overall)

    @staticmethod
    def audit_attributions(attributions: Iterable["Attribution"]) -> None:
        """Audit multiple attribution instances for consistency.

        Args:
            attributions: Attribution instances to audit.

        Raises:
            PpaError: If any attribution fails its own audit or if the equivalent
                portfolio/benchmark columns differ across attribution instances.
        """
        # Initialize base_equivalent_columns to empty (for lint).
        base_equivalent_columns: list[pl.DataFrame] = []  # 0 = portfolio, 1 = benchmark

        # Loop through each attribution and validate it.
        for idxa, attribution in enumerate(attributions):
            # Audit each Attribution separately.
            attribution.audit()

            # Get the equivalent columns.
            # pylint: disable=protected-access
            equivalent_columns = [
                attribution._performances[0]
                .narrow_df.select(_EQUIVALENT_COLUMN_NAMES)
                .unique()
                .sort(cols.ENDING_DATE),
                attribution._performances[1]
                .narrow_df.select(_EQUIVALENT_COLUMN_NAMES)
                .unique()
                .sort(cols.ENDING_DATE),
            ]
            # pylint: enable=protected-access

            # Round the TOTAL_RETURN so it can be "equivalently" compared.
            for idxe, _ in enumerate(equivalent_columns):
                equivalent_columns[idxe] = equivalent_columns[idxe].with_columns(
                    pl.col(cols.TOTAL_RETURN).round(11)
                )

            # Assert that the equivalent_columns are equivalent.
            if idxa == 0:
                base_equivalent_columns = equivalent_columns
            else:
                for idxe, equiv in enumerate(equivalent_columns):
                    if not equiv.equals(base_equivalent_columns[idxe]):
                        raise PpaError(
                            f"Attribution.audit_attributions(): Attribution {idxa} equivalent "
                            "columns do not match base equivalent columns.",
                            999,
                        )

    @staticmethod
    def _audit_columns(
        df: pl.DataFrame, df_overall: pl.DataFrame, do_assert_simple_column_pairs: bool = True
    ) -> None:
        """Audit calculated attribution columns.

        Args:
            df: Detailed attribution DataFrame.
            df_overall: Overall attribution DataFrame containing the total row. May be
                empty for views that do not include an overall row.
            do_assert_simple_column_pairs: Whether to assert equality for simple column
                pairs such as active return and simple total effect.

        Raises:
            PpaError: If expected column pairs are not equal or smoothed columns do not
                sum to their overall values.
        """
        # Assert that certain simple column pairs in df should be equal.
        if do_assert_simple_column_pairs:
            for col1, col2 in _SIMPLE_COLUMN_PAIRS_THAT_SHOULD_BE_EQUAL:
                if col1 in df.columns and col2 in df.columns:
                    if not df[col1].round(7).equals(df[col2].round(7)):
                        raise PpaError(f"_audit_columns() df: {col1} <> {col2}.", 999)

        # Audit df_overall.
        if not df_overall.is_empty():
            # Assert that certain column pairs in df_overall should be equal.
            for col1, col2 in _OVERALL_COLUMN_PAIRS_THAT_SHOULD_BE_EQUAL:
                if col1 in df_overall.columns and col2 in df_overall.columns:
                    if not df_overall[col1].round(7).equals(df_overall[col2].round(7)):
                        raise PpaError(f"_audit_columns() df_overall: {col1} <> {col2}.", 999)

            # Assert that the vertical sum of the smoothed columns of df is equal to df_overall.
            for col_name in cols.ALL_SMOOTHED_COLUMNS:
                if not util.are_near(
                    df[col_name].sum(), df_overall[col_name].item(0), util.Tolerance.MEDIUM
                ):
                    raise PpaError(f"_audit_columns: {col_name} does not foot when summed.", 999)

    def _audit_view(self, view: View) -> None:
        """Audit a rendered attribution view.

        Args:
            view: View to audit.

        Raises:
            PpaError: If contribution does not equal weight multiplied by return, or if
                column-level audit checks fail.
        """
        # Get the DataFrame for the view.
        df = self._fetch_dataframe(view)

        # Assert that weight * return == contribution
        for idx, _ in enumerate(self._performances):
            if not self._performances[idx].subperiods_have_been_consolidated:
                needed_columns = (
                    cols.PORTFOLIO_COLUMNS_SIMPLE if idx == 0 else cols.BENCHMARK_COLUMNS_SIMPLE
                )
                if all(col in df.columns for col in needed_columns):
                    contributions = df[needed_columns[0]] * df[needed_columns[1]]
                    if not (df[needed_columns[2]].round(11) == contributions.round(11)).all():
                        raise PpaError("audit_view(): weight * return != contribution", 999)

        # Audit all columns.
        match view:
            case View.SUBPERIOD_ATTRIBUTION | View.SUBPERIOD_SUMMARY:
                # Subperiods.  There is not a total row.
                df_overall = pl.DataFrame()
                # Sub-period, sector-level numbers interact with one-another, so they do not tie.
                do_assert_simple_column_pairs = view != View.SUBPERIOD_ATTRIBUTION
            case _:
                # There is a total row.
                df_overall = df[-1]
                df = df[:-1]
                do_assert_simple_column_pairs = True
        Attribution._audit_columns(df, df_overall, do_assert_simple_column_pairs)

    def _beginning_date(self) -> dt.date:
        """Return the first beginning date in the attribution period.

        Returns:
            Overall beginning date.
        """
        return cast(dt.date, self._performances[0].narrow_df[cols.BEGINNING_DATE].item(0))

    @staticmethod
    def _attribution_performance_rows(
        performance: Performance,
        weight_column: str,
        return_column: str,
        contribution_column: str,
    ) -> pl.DataFrame:
        """Return narrow performance inputs under attribution column names.

        Args:
            performance: Performance stream to reshape for attribution.
            weight_column: Output weight column name.
            return_column: Output return column name.
            contribution_column: Output simple contribution column name.

        Returns:
            Narrow attribution input rows aligned by period and identifier.
        """
        calculated_return = pl.col(cols.RETURN)
        if performance.subperiods_have_been_consolidated:
            calculated_return = (
                pl.when(pl.col(cols.WEIGHT) != 0.0)
                .then(pl.col(cols.CONTRIBUTION) / pl.col(cols.WEIGHT))
                .otherwise(pl.col(cols.RETURN))
            )
        return performance.narrow_df.select(
            *cols.DATE_COLUMNS,
            pl.col(cols.IDENTIFIER).alias(cols.CLASSIFICATION_IDENTIFIER),
            pl.col(cols.WEIGHT).alias(weight_column),
            calculated_return.alias(return_column),
            pl.col(cols.CONTRIBUTION).alias(contribution_column),
        )

    @staticmethod
    def _detail_derived_expressions(include_weight: bool = True) -> list[pl.Expr]:
        """Return expressions for active and total detail measures.

        Args:
            include_weight: Whether the source rows contain weight columns.

        Returns:
            Polars expressions for measures derived from portfolio and benchmark
            values.
        """
        expressions = [
            (pl.col(cols.PORTFOLIO_RETURN) - pl.col(cols.BENCHMARK_RETURN)).alias(
                cols.ACTIVE_RETURN
            ),
            (
                pl.col(cols.PORTFOLIO_CONTRIB_SIMPLE) - pl.col(cols.BENCHMARK_CONTRIB_SIMPLE)
            ).alias(cols.ACTIVE_CONTRIB_SIMPLE),
            (
                pl.col(cols.PORTFOLIO_CONTRIB_SMOOTHED)
                - pl.col(cols.BENCHMARK_CONTRIB_SMOOTHED)
            ).alias(cols.ACTIVE_CONTRIB_SMOOTHED),
            (
                pl.col(cols.ALLOCATION_EFFECT_SIMPLE) + pl.col(cols.SELECTION_EFFECT_SIMPLE)
            ).alias(cols.TOTAL_EFFECT_SIMPLE),
            (
                pl.col(cols.ALLOCATION_EFFECT_SMOOTHED) + pl.col(cols.SELECTION_EFFECT_SMOOTHED)
            ).alias(cols.TOTAL_EFFECT_SMOOTHED),
        ]
        if include_weight:
            expressions.append(
                (pl.col(cols.PORTFOLIO_WEIGHT) - pl.col(cols.BENCHMARK_WEIGHT)).alias(
                    cols.ACTIVE_WEIGHT
                )
            )
        return expressions

    @staticmethod
    def _smoothed_detail_expressions() -> list[pl.Expr]:
        """Return derived expressions required by the overall detail view."""
        return [
            (pl.col(cols.PORTFOLIO_RETURN) - pl.col(cols.BENCHMARK_RETURN)).alias(
                cols.ACTIVE_RETURN
            ),
            (pl.col(cols.PORTFOLIO_WEIGHT) - pl.col(cols.BENCHMARK_WEIGHT)).alias(
                cols.ACTIVE_WEIGHT
            ),
            (
                pl.col(cols.PORTFOLIO_CONTRIB_SMOOTHED)
                - pl.col(cols.BENCHMARK_CONTRIB_SMOOTHED)
            ).alias(cols.ACTIVE_CONTRIB_SMOOTHED),
            (
                pl.col(cols.ALLOCATION_EFFECT_SMOOTHED) + pl.col(cols.SELECTION_EFFECT_SMOOTHED)
            ).alias(cols.TOTAL_EFFECT_SMOOTHED),
        ]

    def _calculate_attribution(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Calculate narrow attribution rows and period summaries.

        Returns:
            A tuple containing period-level summaries and identifier-level
            detail rows.

        Raises:
            PpaError: If a linking coefficient cannot be calculated for the
                portfolio or benchmark returns.
        """
        portfolio, benchmark = self._performances
        period_returns = (
            portfolio.narrow_df.select(*cols.DATE_COLUMNS, cols.TOTAL_RETURN)
            .unique()
            .rename({cols.TOTAL_RETURN: cols.PORTFOLIO_RETURN})
            .join(
                benchmark.narrow_df.select(*cols.DATE_COLUMNS, cols.TOTAL_RETURN)
                .unique()
                .rename({cols.TOTAL_RETURN: cols.BENCHMARK_RETURN}),
                on=cols.DATE_COLUMNS,
            )
            .sort(cols.ENDING_DATE)
        )
        portfolio_overall_return = cast(
            float, (period_returns[cols.PORTFOLIO_RETURN] + 1).product() - 1
        )
        benchmark_overall_return = cast(
            float, (period_returns[cols.BENCHMARK_RETURN] + 1).product() - 1
        )
        active_denominator = util.carino_linking_coefficient(
            portfolio_overall_return, benchmark_overall_return
        )
        period_factors = period_returns.with_columns(
            pl.Series(
                "_portfolio_linking_coefficient",
                util.logarithmic_linking_coefficients(
                    portfolio_overall_return, period_returns[cols.PORTFOLIO_RETURN]
                ),
            ),
            pl.Series(
                "_benchmark_linking_coefficient",
                util.logarithmic_linking_coefficients(
                    benchmark_overall_return, period_returns[cols.BENCHMARK_RETURN]
                ),
            ),
            pl.Series(
                "_active_linking_coefficient",
                [
                    util.carino_linking_coefficient(portfolio_return, benchmark_return)
                    / active_denominator
                    for portfolio_return, benchmark_return in zip(
                        period_returns[cols.PORTFOLIO_RETURN],
                        period_returns[cols.BENCHMARK_RETURN],
                    )
                ],
            ),
        )

        detail = (
            self._attribution_performance_rows(
                portfolio,
                cols.PORTFOLIO_WEIGHT,
                cols.PORTFOLIO_RETURN,
                cols.PORTFOLIO_CONTRIB_SIMPLE,
            )
            .join(
                self._attribution_performance_rows(
                    benchmark,
                    cols.BENCHMARK_WEIGHT,
                    cols.BENCHMARK_RETURN,
                    cols.BENCHMARK_CONTRIB_SIMPLE,
                ),
                on=[*cols.DATE_COLUMNS, cols.CLASSIFICATION_IDENTIFIER],
            )
            .join(period_factors, on=cols.DATE_COLUMNS, suffix="_period")
            .with_columns(
                (
                    (pl.col(cols.PORTFOLIO_WEIGHT) - pl.col(cols.BENCHMARK_WEIGHT))
                    * (
                        pl.col(cols.BENCHMARK_RETURN)
                        - pl.col(f"{cols.BENCHMARK_RETURN}_period")
                    )
                ).alias(cols.ALLOCATION_EFFECT_SIMPLE),
                (
                    pl.col(cols.PORTFOLIO_WEIGHT)
                    * (pl.col(cols.PORTFOLIO_RETURN) - pl.col(cols.BENCHMARK_RETURN))
                ).alias(cols.SELECTION_EFFECT_SIMPLE),
                (
                    pl.col(cols.PORTFOLIO_CONTRIB_SIMPLE)
                    * pl.col("_portfolio_linking_coefficient")
                ).alias(cols.PORTFOLIO_CONTRIB_SMOOTHED),
                (
                    pl.col(cols.BENCHMARK_CONTRIB_SIMPLE)
                    * pl.col("_benchmark_linking_coefficient")
                ).alias(cols.BENCHMARK_CONTRIB_SMOOTHED),
            )
            .with_columns(
                (
                    pl.col(cols.ALLOCATION_EFFECT_SIMPLE)
                    * pl.col("_active_linking_coefficient")
                ).alias(cols.ALLOCATION_EFFECT_SMOOTHED),
                (
                    pl.col(cols.SELECTION_EFFECT_SIMPLE)
                    * pl.col("_active_linking_coefficient")
                ).alias(cols.SELECTION_EFFECT_SMOOTHED),
            )
            .join(
                portfolio.narrow_df.select(
                    *cols.DATE_COLUMNS,
                    pl.col(cols.IDENTIFIER).alias(cols.CLASSIFICATION_IDENTIFIER),
                    pl.col(cols.RETURN).alias("_portfolio_display_return"),
                ),
                on=[*cols.DATE_COLUMNS, cols.CLASSIFICATION_IDENTIFIER],
            )
            .join(
                benchmark.narrow_df.select(
                    *cols.DATE_COLUMNS,
                    pl.col(cols.IDENTIFIER).alias(cols.CLASSIFICATION_IDENTIFIER),
                    pl.col(cols.RETURN).alias("_benchmark_display_return"),
                ),
                on=[*cols.DATE_COLUMNS, cols.CLASSIFICATION_IDENTIFIER],
            )
            .with_columns(
                pl.col("_portfolio_display_return").alias(cols.PORTFOLIO_RETURN),
                pl.col("_benchmark_display_return").alias(cols.BENCHMARK_RETURN),
            )
            .with_columns(self._detail_derived_expressions())
            .select(
                *cols.DATE_COLUMNS,
                cols.CLASSIFICATION_IDENTIFIER,
                cols.PORTFOLIO_WEIGHT,
                cols.PORTFOLIO_RETURN,
                cols.PORTFOLIO_CONTRIB_SIMPLE,
                cols.PORTFOLIO_CONTRIB_SMOOTHED,
                cols.BENCHMARK_WEIGHT,
                cols.BENCHMARK_RETURN,
                cols.BENCHMARK_CONTRIB_SIMPLE,
                cols.BENCHMARK_CONTRIB_SMOOTHED,
                cols.ACTIVE_WEIGHT,
                cols.ACTIVE_RETURN,
                cols.ACTIVE_CONTRIB_SIMPLE,
                cols.ACTIVE_CONTRIB_SMOOTHED,
                cols.ALLOCATION_EFFECT_SIMPLE,
                cols.SELECTION_EFFECT_SIMPLE,
                cols.TOTAL_EFFECT_SIMPLE,
                cols.ALLOCATION_EFFECT_SMOOTHED,
                cols.SELECTION_EFFECT_SMOOTHED,
                cols.TOTAL_EFFECT_SMOOTHED,
            )
            .sort([cols.ENDING_DATE, cols.CLASSIFICATION_IDENTIFIER])
        )
        summary = (
            detail.group_by(cols.DATE_COLUMNS)
            .agg(
                pl.col(cols.PORTFOLIO_CONTRIB_SIMPLE).sum(),
                pl.col(cols.BENCHMARK_CONTRIB_SIMPLE).sum(),
                pl.col(cols.PORTFOLIO_CONTRIB_SMOOTHED).sum(),
                pl.col(cols.BENCHMARK_CONTRIB_SMOOTHED).sum(),
                pl.col(cols.ALLOCATION_EFFECT_SIMPLE).sum(),
                pl.col(cols.SELECTION_EFFECT_SIMPLE).sum(),
                pl.col(cols.ALLOCATION_EFFECT_SMOOTHED).sum(),
                pl.col(cols.SELECTION_EFFECT_SMOOTHED).sum(),
            )
            .join(period_returns, on=cols.DATE_COLUMNS)
            .sort(cols.ENDING_DATE)
            .with_columns(self._detail_derived_expressions(include_weight=False))
        )
        return self._sum_columns_and_rows(summary.lazy()).collect(), detail

    def _calculate_df_overall(self) -> pl.DataFrame:
        """Calculate the overall attribution row.

        Returns:
            DataFrame containing one row for the full attribution period.
        """
        portfolio_overall_return = cast(float, self._df[-1, cols.CUMULATIVE_PORTFOLIO_RETURN])
        benchmark_overall_return = cast(float, self._df[-1, cols.CUMULATIVE_BENCHMARK_RETURN])

        # Start the total row.  Note that sums only apply to the smoothed columns.
        df_overall = self._df.sum()

        # Override the total row date columns.
        df_overall[0, cols.BEGINNING_DATE] = self._df[cols.BEGINNING_DATE][0]
        df_overall[0, cols.ENDING_DATE] = self._df[cols.ENDING_DATE][-1]

        # Override the total row return columns.
        df_overall[0, cols.PORTFOLIO_RETURN] = portfolio_overall_return
        df_overall[0, cols.BENCHMARK_RETURN] = benchmark_overall_return
        df_overall[0, cols.ACTIVE_RETURN] = portfolio_overall_return - benchmark_overall_return

        # Override the total row cumulative columns.
        for col_name in cols.ALL_CUMULATIVE_COLUMNS:
            df_overall[0, col_name] = self._df[-1, col_name]

        # Override the total row simple columns.
        for col_name in cols.ALL_SIMPLE_COLUMNS:
            df_overall[0, col_name] = np.nan

        # Return the instance values.
        return df_overall

    def _construct_df_for_detail_views(self, view: View) -> pl.LazyFrame:
        """Construct the DataFrame used by detailed attribution views.

        Args:
            view: Detailed view to construct. Supported values are
                ``View.SUBPERIOD_ATTRIBUTION`` and ``View.OVERALL_ATTRIBUTION``.

        Returns:
            LazyFrame containing dates, classification identifiers, classification
            names, weights, returns, contributions, and attribution effects.

        Raises:
            PpaError: If ``view`` is not a supported detailed attribution view.
        """
        match view:
            case View.SUBPERIOD_ATTRIBUTION:
                detail = self._detail_df
            case View.OVERALL_ATTRIBUTION:
                portfolio, benchmark = self._performances
                detail = (
                    self._overall_performance_rows(
                        portfolio, cols.PORTFOLIO_WEIGHT, cols.PORTFOLIO_RETURN
                    )
                    .join(
                        self._overall_performance_rows(
                            benchmark, cols.BENCHMARK_WEIGHT, cols.BENCHMARK_RETURN
                        ),
                        on=[*cols.DATE_COLUMNS, cols.CLASSIFICATION_IDENTIFIER],
                    )
                    .join(
                        self._detail_df.group_by(cols.CLASSIFICATION_IDENTIFIER).agg(
                            pl.col(cols.PORTFOLIO_CONTRIB_SMOOTHED).sum(),
                            pl.col(cols.BENCHMARK_CONTRIB_SMOOTHED).sum(),
                            pl.col(cols.ALLOCATION_EFFECT_SMOOTHED).sum(),
                            pl.col(cols.SELECTION_EFFECT_SMOOTHED).sum(),
                        ),
                        on=cols.CLASSIFICATION_IDENTIFIER,
                    )
                    .with_columns(self._smoothed_detail_expressions())
                )
            case _:
                raise PpaError(f"Unhandled View {view} in Attribution._construct_df_detail()", 999)

        return (
            detail.lazy()
            .join(
                self._classification.df.lazy(),
                on=cols.CLASSIFICATION_IDENTIFIER,
                how="left",
            )
            .with_columns(
                pl.col(cols.CLASSIFICATION_NAME).fill_null(pl.col(cols.CLASSIFICATION_IDENTIFIER))
            )
            .sort(cols.BEGINNING_DATE, cols.CLASSIFICATION_IDENTIFIER)
        )

    @staticmethod
    def _overall_performance_rows(
        performance: Performance, weight_column: str, return_column: str
    ) -> pl.DataFrame:
        """Calculate overall identifier returns and weights from narrow rows.

        Args:
            performance: Performance stream to summarize.
            weight_column: Output weight column name.
            return_column: Output return column name.

        Returns:
            One overall-period row per identifier.
        """
        beginning_date = cast(dt.date, performance.narrow_df[cols.BEGINNING_DATE].min())
        ending_date = cast(dt.date, performance.narrow_df[cols.ENDING_DATE].max())
        total_days = (ending_date - beginning_date).days
        weight_coefficient = (
            pl.lit(1.0)
            if total_days == 0
            else pl.col(cols.QUANTITY_OF_DAYS) / total_days
        )
        return (
            performance.narrow_df.group_by(cols.IDENTIFIER)
            .agg(
                pl.col(cols.RETURN).add(1).product().sub(1).alias(return_column),
                (pl.col(cols.WEIGHT) * weight_coefficient).sum().alias(weight_column),
            )
            .rename({cols.IDENTIFIER: cols.CLASSIFICATION_IDENTIFIER})
            .with_columns(
                pl.lit(beginning_date).alias(cols.BEGINNING_DATE),
                pl.lit(ending_date).alias(cols.ENDING_DATE),
            )
            .select(
                *cols.DATE_COLUMNS,
                cols.CLASSIFICATION_IDENTIFIER,
                return_column,
                weight_column,
            )
        )

    def _ending_date(self) -> dt.date:
        """Return the last ending date in the attribution period.

        Returns:
            Overall ending date.
        """
        return cast(dt.date, self._performances[0].narrow_df[cols.ENDING_DATE].item(-1))

    def _equalize_columns(self) -> None:
        """Equalize portfolio and benchmark identifier rows.

        Missing identifiers are added to the opposite performance with zero-valued
        return, weight, and contribution rows so narrow row joins use matching
        identifier sets.
        """
        portfolio, benchmark = self._performances
        for target, source in ((portfolio, benchmark), (benchmark, portfolio)):
            missing_identifiers = sorted(set(source.identifiers) - set(target.identifiers))
            if missing_identifiers:
                periods = target.narrow_df.select(
                    *cols.DATE_COLUMNS,
                    cols.QUANTITY_OF_DAYS,
                    cols.TOTAL_RETURN,
                ).unique()
                missing_rows = (
                    periods.join(
                        source.narrow_df.filter(
                            pl.col(cols.IDENTIFIER).is_in(missing_identifiers)
                        ).select(
                            *cols.DATE_COLUMNS,
                            cols.IDENTIFIER,
                            (pl.col(cols.RETURN) * 0.0).alias(cols.RETURN),
                            (pl.col(cols.WEIGHT) * 0.0).alias(cols.WEIGHT),
                            (pl.col(cols.CONTRIBUTION) * 0.0).alias(cols.CONTRIBUTION),
                        ),
                        on=cols.DATE_COLUMNS,
                    )
                    .select(target.narrow_df.columns)
                )
                target.reset_narrow_df(pl.concat([target.narrow_df, missing_rows]))

    def _fetch_dataframe(
        self,
        view: View,
        columns_to_sort: str | Sequence[str] | None = None,
        sort_descendings: bool | Sequence[bool] = False,
    ) -> pl.DataFrame:
        """Fetch the DataFrame for a view.

        Args:
            view: View to fetch.
            columns_to_sort: Optional column name or sequence of column names to sort
                by. Sorting is ignored for cumulative attribution.
            sort_descendings: Boolean or sequence of booleans indicating whether the
                corresponding sort columns should be sorted descending.

        Returns:
            DataFrame for the requested view, optionally sorted and with a total row
            added for views that require one.

        Raises:
            PpaError: If constructing a detailed view fails validation.
        """
        # Get the base dataframe associated with the view.
        match view:
            case View.CUMULATIVE_ATTRIBUTION | View.SUBPERIOD_SUMMARY:
                lf = self._df.lazy()
            case _:  # View.SUBPERIOD_ATTRIBUTION | View.OVERALL_ATTRIBUTION
                lf = self._construct_df_for_detail_views(view)

        # Select only the needed columns.
        lf = lf.select(_VIEW_COLUMN_NAMES[view])

        # Sort the dataframe.  View.CUMULATIVE_ATTRIBUTION is not sortable, because it has
        # "cumulative" columns that are implicitly chronological.
        if (
            columns_to_sort is not None
            and not (isinstance(columns_to_sort, str) and not columns_to_sort.strip())
            and view != View.CUMULATIVE_ATTRIBUTION
        ):
            lf = lf.sort(by=columns_to_sort, descending=sort_descendings)

        # Must collect() before adding the total_row
        df = lf.collect()

        # Add the total_row
        if view in (View.CUMULATIVE_ATTRIBUTION, View.OVERALL_ATTRIBUTION):
            df = self._add_total_row(df)

        # Return the dataframe.
        return df

    def _sum_columns_and_rows(
        self,
        lf: pl.LazyFrame,
    ) -> pl.LazyFrame:
        """Add cumulative columns to period-level attribution summaries.

        Args:
            lf: LazyFrame containing one summarized attribution row per period.

        Returns:
            LazyFrame with cumulative return, contribution, and attribution columns.
        """
        # Vertically accumulate the cumulative columns.
        lf = lf.with_columns(
            [
                # CUMULATIVE_PORTFOLIO_RETURN
                pl.col(cols.PORTFOLIO_RETURN)
                .add(1)
                .cum_prod()
                .sub(1)
                .alias(cols.CUMULATIVE_PORTFOLIO_RETURN),
                # CUMULATIVE_BENCHMARK_RETURN
                pl.col(cols.BENCHMARK_RETURN)
                .add(1)
                .cum_prod()
                .sub(1)
                .alias(cols.CUMULATIVE_BENCHMARK_RETURN),
                # CUMULATIVE_PORTFOLIO_CONTRIB
                pl.col(cols.PORTFOLIO_CONTRIB_SMOOTHED)
                .cum_sum()
                .alias(cols.CUMULATIVE_PORTFOLIO_CONTRIB),
                # CUMULATIVE_BENCHMARK_CONTRIB
                pl.col(cols.BENCHMARK_CONTRIB_SMOOTHED)
                .cum_sum()
                .alias(cols.CUMULATIVE_BENCHMARK_CONTRIB),
                # CUMULATIVE_ALLOCATION_EFFECT
                pl.col(cols.ALLOCATION_EFFECT_SMOOTHED)
                .cum_sum()
                .alias(cols.CUMULATIVE_ALLOCATION_EFFECT),
                # CUMULATIVE_SELECTION_EFFECT
                pl.col(cols.SELECTION_EFFECT_SMOOTHED)
                .cum_sum()
                .alias(cols.CUMULATIVE_SELECTION_EFFECT),
                # CUMULATIVE_TOTAL_EFFECT
                pl.col(cols.TOTAL_EFFECT_SMOOTHED).cum_sum().alias(cols.CUMULATIVE_TOTAL_EFFECT),
            ]
        )

        # Calculate the active columns.
        # You cannot subtract 2 lazyframe columns, so you need to collect first.
        df = lf.collect()
        lf = (
            df.lazy().with_columns(
                [
                    # Active return (no distinction between simple and smoothed)
                    (df[cols.PORTFOLIO_RETURN] - df[cols.BENCHMARK_RETURN]).alias(
                        cols.ACTIVE_RETURN
                    ),
                    # Cumulative active return
                    (
                        df[cols.CUMULATIVE_PORTFOLIO_RETURN] - df[cols.CUMULATIVE_BENCHMARK_RETURN]
                    ).alias(cols.CUMULATIVE_ACTIVE_RETURN),
                    # Simple active contribution
                    (df[cols.PORTFOLIO_CONTRIB_SIMPLE] - df[cols.BENCHMARK_CONTRIB_SIMPLE]).alias(
                        cols.ACTIVE_CONTRIB_SIMPLE
                    ),
                    # Smoothed (log-linked) active contribution
                    (
                        df[cols.PORTFOLIO_CONTRIB_SMOOTHED] - df[cols.BENCHMARK_CONTRIB_SMOOTHED]
                    ).alias(cols.ACTIVE_CONTRIB_SMOOTHED),
                ]
            )
            # Cumulative active contribution
            .with_columns(
                pl.col(cols.ACTIVE_CONTRIB_SMOOTHED)
                .cum_sum()
                .alias(cols.CUMULATIVE_ACTIVE_CONTRIB)
            )
        )

        # Return the resulting LazyFrame
        return lf

    def _title_lines(self, chart_or_view: Chart | View) -> tuple[str, str]:
        """Return title and subtitle text for a chart or view.

        Args:
            chart_or_view: Chart or View whose display value is used in the subtitle.

        Returns:
            Two-item tuple containing the title and subtitle.
        """
        # Determine if chart_or_view is a Chart or a View
        is_view = isinstance(chart_or_view, View)

        # Line 1: Portfolio Name (vs Benchmark Name)
        portfolio_name = self._performances[0].name or ""
        benchmark_name = self._performances[1].name or ""
        line1 = (
            portfolio_name
            if (
                chart_or_view
                in (Chart.HEATMAP_PORTFOLIO_CONTRIBUTION, Chart.HEATMAP_PORTFOLIO_RETURN)
            )
            else f"{portfolio_name} vs {benchmark_name}"
        )

        # Get the classification description if it is relevant.
        classification_description = (
            f" by {self._classification_label}"
            if (
                (
                    is_view
                    or "Attribution" in chart_or_view.value
                    or "Contribution" in chart_or_view.value
                )
                and self._classification_label is not None
            )
            else ""
        )

        # Line 2: Chart/View name, classification, frequency, dates.
        line2 = (
            f"{chart_or_view.value}{classification_description}: {self._frequency.value}"
            f" from {self._beginning_date()} to {self._ending_date()}"
        )

        # Return the title and subtitle.
        return (line1, line2)

    def to_chart(
        self,
        chart: Chart,
        columns_to_sort: str | Sequence[str] | None = None,
        sort_descendings: bool | Sequence[bool] = False,
    ) -> bytes:
        """Return a PNG chart for the requested attribution chart type.

        Args:
            chart: Chart type to render.
            columns_to_sort: Optional column name or sequence of column names used for
                sortable charts.
            sort_descendings: Boolean or sequence of booleans indicating whether the
                corresponding sort columns should be sorted descending.

        Returns:
            In-memory PNG bytes for the requested chart.

        Raises:
            PpaError: If the underlying view construction or table retrieval fails
                validation.
            ModuleNotFoundError: If optional chart dependencies are not installed.
        """
        # Charting dependencies are optional and needed only when chart output is requested.
        try:
            from ppar import format_chart  # pylint: disable=import-outside-toplevel
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "Chart output requires optional dependencies; install 'ppar[charts]'."
            ) from error

        # Get the title_lines.
        title_lines = self._title_lines(chart)

        # Get the chart.
        match chart:
            case (
                Chart.CUMULATIVE_ATTRIBUTION
                | Chart.CUMULATIVE_CONTRIBUTION
                | Chart.CUMULATIVE_RETURN
            ):
                # Set the DataFrame and remove the last "Total" row.  Note that sorting is not
                # valid for these line charts.
                df = self.to_polars(View.CUMULATIVE_ATTRIBUTION)[:-1]
                # Set the labels and column names.
                match chart:
                    case Chart.CUMULATIVE_ATTRIBUTION:
                        y_axis_label = "Effect"
                        column_names = cols.CUMULATIVE_ATTRIBUTION_COLUMNS
                    case Chart.CUMULATIVE_CONTRIBUTION:
                        y_axis_label = "Contribution"
                        column_names = cols.CUMULATIVE_CONTRIBUTION_COLUMNS
                    case Chart.CUMULATIVE_RETURN:
                        y_axis_label = "Return"
                        column_names = cols.CUMULATIVE_RETURN_COLUMNS
                # Get the chart png
                png = format_chart.cumulative_lines(df, column_names, title_lines, y_axis_label)

            case (
                Chart.HEATMAP_ACTIVE_CONTRIBUTION
                | Chart.HEATMAP_ACTIVE_RETURN
                | Chart.HEATMAP_ATTRIBUTION
                | Chart.HEATMAP_PORTFOLIO_CONTRIBUTION
                | Chart.HEATMAP_PORTFOLIO_RETURN
            ):
                # Set the DataFrame.  Note that sorting is done below in format_chart.heatmap().
                df = self.to_polars(View.SUBPERIOD_ATTRIBUTION)
                # Set the labels and column names.
                match chart:
                    case Chart.HEATMAP_ACTIVE_CONTRIBUTION:
                        column_name = cols.ACTIVE_CONTRIB_SIMPLE
                    case Chart.HEATMAP_ACTIVE_RETURN:
                        column_name = cols.ACTIVE_RETURN
                    case Chart.HEATMAP_ATTRIBUTION:
                        column_name = cols.TOTAL_EFFECT_SIMPLE
                    case Chart.HEATMAP_PORTFOLIO_CONTRIBUTION:
                        column_name = cols.PORTFOLIO_CONTRIB_SIMPLE
                    case Chart.HEATMAP_PORTFOLIO_RETURN:
                        column_name = cols.PORTFOLIO_RETURN
                # Get the sorted chart png.
                png = format_chart.heatmap(
                    df, column_name, title_lines, columns_to_sort, sort_descendings
                )

            case Chart.SUBPERIOD_ATTRIBUTION | Chart.SUBPERIOD_RETURN:
                # Set the DataFrame.  Note that sorting is not valid for these bar charts.
                df = self.to_polars(View.SUBPERIOD_SUMMARY)
                # Set the labels and column names.
                match chart:
                    case Chart.SUBPERIOD_ATTRIBUTION:
                        y_axis_label = "Effect"
                        column_names = cols.ATTRIBUTION_COLUMNS_SIMPLE
                    case Chart.SUBPERIOD_RETURN:
                        y_axis_label = "Return"
                        column_names = cols.RETURN_COLUMNS
                # Get the chart png
                png = format_chart.vertical_bars(df, column_names, title_lines, y_axis_label)

            case Chart.OVERALL_ATTRIBUTION:
                # Set the default sorting.
                if columns_to_sort is None or (
                    isinstance(columns_to_sort, str) and not columns_to_sort.strip()
                ):
                    columns_to_sort = cols.TOTAL_EFFECT_SMOOTHED
                    sort_descendings = True
                # Set the DataFrame and remove the last "Total" row.
                df = self.to_polars(View.OVERALL_ATTRIBUTION, columns_to_sort, sort_descendings)[
                    :-1
                ]
                # Get the chart png
                png = format_chart.overall_attribution(df, title_lines)

            case _:  # Chart.OVERALL_CONTRIBUTION:
                # Set the default sorting.
                if columns_to_sort is None or (
                    isinstance(columns_to_sort, str) and not columns_to_sort.strip()
                ):
                    columns_to_sort = cols.PORTFOLIO_CONTRIB_SMOOTHED
                    sort_descendings = True
                # Set the DataFrame and remove the last "Total" row.
                df = self.to_polars(View.OVERALL_ATTRIBUTION, columns_to_sort, sort_descendings)[
                    :-1
                ]
                # Get the chart png
                png = format_chart.overall_contribution(
                    df,
                    title_lines,
                    self._performances[0].name or "",
                    self._performances[1].name or "",
                )

        # Return the chart png
        return png

    def to_html(
        self,
        view: View,
        columns_to_sort: str | Sequence[str] | None = None,
        sort_descendings: bool | Sequence[bool] = False,
    ) -> str:
        """Return a view as an HTML document string.

        Args:
            view: View to render.
            columns_to_sort: Optional column name or sequence of column names to sort
                by.
            sort_descendings: Boolean or sequence of booleans indicating whether the
                corresponding sort columns should be sorted descending.

        Returns:
            HTML string containing the rendered table.

        Raises:
            PpaError: If the requested table is too large for HTML rendering or view
                construction fails validation.
        """
        df = self._fetch_dataframe(view, columns_to_sort, sort_descendings)
        if 500 < len(df):
            raise PpaError(f"{view.value}, Rows = {len(df)}", 204)
        return html_table.attribution_html(
            df,
            view.value,
            self._title_lines(view),
            self._classification_label,
        )

    def to_json(
        self,
        view: View,
        columns_to_sort: str | Sequence[str] | None = None,
        sort_descendings: bool | Sequence[bool] = False,
        float_precision: int = _DEFAULT_OUTPUT_PRECISION,
    ) -> str:
        """Return a view as a JSON string.

        Args:
            view: View to serialize.
            columns_to_sort: Optional column name or sequence of column names to sort
                by.
            sort_descendings: Boolean or sequence of booleans indicating whether the
                corresponding sort columns should be sorted descending.
            float_precision: Number of decimal places to include for floating-point
                values.

        Returns:
            JSON string for the requested view.

        Raises:
            PpaError: If view construction fails validation.
        """
        return output.to_json(
            self._fetch_dataframe(view, columns_to_sort, sort_descendings),
            float_precision,
            date_format="iso",
        )

    def to_pandas(
        self,
        view: View,
        columns_to_sort: str | Sequence[str] | None = None,
        sort_descendings: bool | Sequence[bool] = False,
    ) -> pd.DataFrame:
        """Return a view as a pandas DataFrame.

        Args:
            view: View to return.
            columns_to_sort: Optional column name or sequence of column names to sort
                by.
            sort_descendings: Boolean or sequence of booleans indicating whether the
                corresponding sort columns should be sorted descending.

        Returns:
            pandas DataFrame for the requested view.

        Raises:
            PpaError: If view construction fails validation.
        """
        return output.to_pandas(self._fetch_dataframe(view, columns_to_sort, sort_descendings))

    def to_polars(
        self,
        view: View,
        columns_to_sort: str | Sequence[str] | None = None,
        sort_descendings: bool | Sequence[bool] = False,
    ) -> pl.DataFrame:
        """Return a view as a Polars DataFrame.

        Args:
            view: View to return.
            columns_to_sort: Optional column name or sequence of column names to sort
                by.
            sort_descendings: Boolean or sequence of booleans indicating whether the
                corresponding sort columns should be sorted descending.

        Returns:
            Polars DataFrame for the requested view.

        Raises:
            PpaError: If view construction fails validation.
        """
        return self._fetch_dataframe(view, columns_to_sort, sort_descendings)

    def to_table(
        self,
        view: View,
        columns_to_sort: str | Sequence[str] | None = None,
        sort_descendings: bool | Sequence[bool] = False,
    ) -> html_table.HtmlTable:
        """Return a lightweight HTML table object for a view.

        Args:
            view: View to render.
            columns_to_sort: Optional column name or sequence of column names to sort
                by.
            sort_descendings: Boolean or sequence of booleans indicating whether the
                corresponding sort columns should be sorted descending.

        Returns:
            HtmlTable object for the requested view.

        Raises:
            PpaError: If the requested view has more than 500 rows or view construction
                fails validation.
        """
        df = self._fetch_dataframe(view, columns_to_sort, sort_descendings)

        # If there are more than a few hundred lines in an html file, then Attribution.to_html()
        # can be VERY slow.  This can occur when requesting html for a View that has one line for
        # each sub-period and classification item.  For instance, if the user requests to see 100
        # days of 100 securities for View.SUBPERIOD_ATTRIBUTION.
        if 500 < len(df):
            raise PpaError(f"{view.value}, Rows = {len(df)}", 204)

        return html_table.attribution_table(
            df,
            view.value,
            self._title_lines(view),
            self._classification_label,
        )

    def to_xml(
        self,
        view: View,
        columns_to_sort: str | Sequence[str] | None = None,
        sort_descendings: bool | Sequence[bool] = False,
    ) -> str:
        """Return a view as an XML string.

        Args:
            view: View to serialize.
            columns_to_sort: Optional column name or sequence of column names to sort
                by.
            sort_descendings: Boolean or sequence of booleans indicating whether the
                corresponding sort columns should be sorted descending.

        Returns:
            XML string for the requested view.

        Raises:
            PpaError: If view construction fails validation.
        """
        return output.to_xml(self._fetch_dataframe(view, columns_to_sort, sort_descendings))

    def write_csv(
        self,
        view: View,
        file_path: util.PathLike,
        columns_to_sort: str | Sequence[str] | None = None,
        sort_descendings: bool | Sequence[bool] = False,
        float_precision: int = _DEFAULT_OUTPUT_PRECISION,
    ) -> None:
        """Write a view to a CSV file.

        Args:
            view: View to write.
            file_path: Path of the CSV file to write.
            columns_to_sort: Optional column name or sequence of column names to sort
                by.
            sort_descendings: Boolean or sequence of booleans indicating whether the
                corresponding sort columns should be sorted descending.
            float_precision: Number of decimal places to write for floating-point
                values.

        Raises:
            PpaError: If view construction fails validation.
        """
        output.write_csv(
            self._fetch_dataframe(view, columns_to_sort, sort_descendings),
            file_path,
            float_precision,
        )
