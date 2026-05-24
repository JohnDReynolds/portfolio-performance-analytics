"""
Compute and expose Brinson-Fachler attribution results.

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
import re
from typing import cast, Iterable, Sequence

# Third-Party Imports
import great_tables as gt
import numpy as np
import pandas as pd
import polars as pl

# Project Imports
from ppar.classification import Classification
import ppar.columns as cols
from ppar.columns import AEL, AES, BCL, BCS, CON, PCL, PCS, RET, SEL, SES, WGT
from ppar.errors import PpaError
import ppar.format_chart as format_chart
import ppar.format_table as format_table
from ppar.frequency import Frequency
from ppar.performance import Performance
import ppar.utilities as util

# Constants
_DEFAULT_OUTPUT_PRECISION = 8


class Chart(Enum):
    """Attribution chart types supported by :meth:`Attribution.to_chart`.

    Each enum value is the display label used in chart titles.
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
    HTML, JSON, pandas DataFrames, Polars DataFrames, Great Tables, XML, and CSV.
    """

    def __init__(
        self,
        performances: tuple[Performance, Performance],
        classification_name: str,
        classification_data_source: util.ClassificationDataSource,
        frequency: Frequency,
        classification_label: str = util.EMPTY,
    ):
        """Initialize an attribution calculation.

        Args:
            performances: A two-item tuple containing the portfolio ``Performance`` at
                index 0 and the benchmark ``Performance`` at index 1.
            classification_name: Classification name for which contribution and
                attribution effects are calculated.
            classification_data_source: Classification source. May be a CSV file path,
                dictionary, pandas DataFrame, or Polars DataFrame. The first column is
                the classification identifier and the second column is the display name.
            frequency: Frequency associated with the attribution periods.
            classification_label: Optional label displayed in tables and charts. If
                empty, the classification name is used.

        Raises:
            PpaError: If classification setup, performance alignment, linking, or
                attribution calculation fails validation.
        """
        # Set internal instance variables from the constructor parameters.
        self._classification = Classification(
            classification_name, classification_data_source, performances
        )
        self._frequency = frequency
        self._performances = performances
        self._classification_label = (
            self._classification.name
            if util.is_empty(classification_label)
            else classification_label
        )

        # Make sure that the portfolio and benchmark performances have the same columns.
        self._equalize_columns()

        # Create the Attribution DataFrames.
        self._df = self._calculate_attribution().collect()
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
            total_row[0, cols.PORTFOLIO_RETURN] = self._performances[0].overall_return()
            total_row[0, cols.BENCHMARK_RETURN] = self._performances[1].overall_return()
            total_row[0, cols.ACTIVE_RETURN] = (
                self._performances[0].overall_return() - self._performances[1].overall_return()
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
                attribution._performances[0].df[_EQUIVALENT_COLUMN_NAMES],
                attribution._performances[1].df[_EQUIVALENT_COLUMN_NAMES],
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
        return cast(dt.date, self._performances[0].df[cols.BEGINNING_DATE].item(0))

    def _calculate_attribution(self) -> pl.LazyFrame:
        """Calculate subperiod contribution and attribution effects.

        Returns:
            LazyFrame containing one row per subperiod with portfolio contribution,
            benchmark contribution, Brinson-Fachler allocation and selection effects,
            smoothed effects, active return, and cumulative columns.

        Raises:
            PpaError: If Carino linking coefficient calculation encounters an invalid
                portfolio or benchmark return.
        """
        # Set the portfolio and benchmark.
        portfolio, benchmark = self._performances

        # Pull the period-level inputs once so the Brinson-Fachler formulas below
        # read as matrix arithmetic across all identifiers.
        portfolio_consolidated_returns = portfolio.consolidated_returns()
        benchmark_consolidated_returns = benchmark.consolidated_returns()
        portfolio_linking_coefficients = portfolio.linking_coefficients()
        benchmark_linking_coefficients = benchmark.linking_coefficients()
        portfolio_overall_return = portfolio.overall_return()
        benchmark_overall_return = benchmark.overall_return()
        portfolio_total_returns = portfolio.df[cols.TOTAL_RETURN]
        benchmark_total_returns = benchmark.df[cols.TOTAL_RETURN]

        # Portfolio and benchmark weights must be materialized on the same column
        # grid before subtracting active weights identifier-by-identifier.
        portfolio_weights = portfolio.df.lazy().select(portfolio.col_names(WGT)).collect()
        benchmark_weights = benchmark.df.lazy().select(benchmark.col_names(WGT)).collect()

        # Carino coefficients translate period attribution effects into an overall
        # arithmetic attribution story. The denominator normalizes the period factors
        # so smoothed allocation + selection sums to the linked active return.
        inverse_denominator = 1.0 / util.carino_linking_coefficient(
            portfolio_overall_return, benchmark_overall_return
        )
        linking_coefficients = pl.Series(
            values=[
                util.carino_linking_coefficient(p, b) * inverse_denominator
                for p, b in zip(portfolio_total_returns, benchmark_total_returns)
            ]
        )

        # Construct lf.
        lf = (
            # Dates
            pl.LazyFrame()
            .with_columns(portfolio.df.select(cols.DATE_COLUMNS))
            # Simple portfolio contribution.
            .with_columns(
                (portfolio.df[portfolio.col_names(CON)]).rename(
                    lambda column_name: f"{column_name[:-4]}{PCS}"
                )
            )
            # Simple benchmark contribution.
            .with_columns(
                (benchmark.df[benchmark.col_names(CON)]).rename(
                    lambda column_name: f"{column_name[:-4]}{BCS}"
                )
            )
            # Brinson-Fachler allocation isolates active weight decisions:
            # (portfolio weight - benchmark weight) times the benchmark sector
            # return relative to the benchmark total return. Overweights in sectors
            # that beat the benchmark total return help performance.
            .with_columns(
                (
                    (benchmark_consolidated_returns - benchmark_total_returns)
                    * (portfolio_weights - benchmark_weights)
                ).rename(lambda column_name: f"{column_name[:-4]}{AES}")
            )
            # Selection isolates security/segment return differences while holding
            # the portfolio weight fixed. The active return decision is measured
            # where the portfolio actually had exposure.
            .with_columns(
                (
                    portfolio_weights
                    * (portfolio_consolidated_returns - benchmark_consolidated_returns)
                ).rename(lambda column_name: f"{column_name[:-4]}{SES}")
            )
            .with_columns(
                [
                    # Smoothed contribution columns foot to linked multi-period return.
                    *[
                        (pl.col(f"{id}{PCS}") * portfolio_linking_coefficients).alias(f"{id}{PCL}")
                        for id in portfolio.identifiers
                    ],
                    *[
                        (pl.col(f"{id}{BCS}") * benchmark_linking_coefficients).alias(f"{id}{BCL}")
                        for id in benchmark.identifiers
                    ],
                    # Smoothed attribution effects use the portfolio-vs-benchmark
                    # Carino factors, not the standalone portfolio/benchmark factors.
                    *[
                        (pl.col(f"{id}{AES}") * linking_coefficients).alias(f"{id}{AEL}")
                        for id in portfolio.identifiers
                    ],
                    *[
                        (pl.col(f"{id}{SES}") * linking_coefficients).alias(f"{id}{SEL}")
                        for id in portfolio.identifiers
                    ],
                    portfolio_total_returns.alias(cols.PORTFOLIO_RETURN),
                    benchmark_total_returns.alias(cols.BENCHMARK_RETURN),
                ]
            )
        )

        # Add roll-up columns such as total allocation, total selection, active
        # contribution, and cumulative linked values used by tables and charts.
        lf = self._sum_columns_and_rows(lf, portfolio)

        # Return lazy version of self.df.
        return lf

    def _calculate_df_overall(self) -> pl.DataFrame:
        """Calculate the overall attribution row.

        Returns:
            DataFrame containing one row for the full attribution period.
        """
        # Set the portfolio and benchmark.
        portfolio, benchmark = self._performances

        # Get pre-computed values.
        portfolio_overall_return = portfolio.overall_return()
        benchmark_overall_return = benchmark.overall_return()

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
        for col_name in (
            cols.ALL_SIMPLE_COLUMNS
            + portfolio.col_names(PCS)
            + benchmark.col_names(BCS)
            + portfolio.col_names(AES)
            + benchmark.col_names(SES)
        ):
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
        # Set the appropriate dataframes based on the view.
        portfolio, benchmark = self._performances
        match view:
            case View.SUBPERIOD_ATTRIBUTION:
                attribution_df, portfolio_df, benchmark_df = (
                    self._df,
                    portfolio.df,
                    benchmark.df,
                )
            case View.OVERALL_ATTRIBUTION:
                attribution_df, portfolio_df, benchmark_df = (
                    self._df_overall,
                    portfolio.df_overall(),
                    benchmark.df_overall(),
                )
            case _:
                raise PpaError(f"Unhandled View {view} in Attribution._construct_df_detail()", 999)

        # Do parameter-driven un-pivots to build the list of LazyFrame columns.
        # This transforms wide identifier-level columns into vertical rows for each
        # classification identifier, enabling the detailed attribution views to
        # include one row per identifier per period.
        columns: list[pl.LazyFrame] = []
        for parms in (
            (
                portfolio_df,
                (portfolio.col_names(RET), portfolio.col_names(WGT)),
                (cols.PORTFOLIO_RETURN, cols.PORTFOLIO_WEIGHT),
            ),
            (
                benchmark_df,
                (benchmark.col_names(RET), benchmark.col_names(WGT)),
                (cols.BENCHMARK_RETURN, cols.BENCHMARK_WEIGHT),
            ),
            (
                attribution_df,
                (
                    f".*{PCS}$",
                    f".*{BCS}$",
                    f".*{PCL}$",
                    f".*{BCL}$",
                    f".*{AES}$",
                    f".*{SES}$",
                    f".*{AEL}$",
                    f".*{SEL}$",
                ),
                (
                    cols.PORTFOLIO_CONTRIB_SIMPLE,
                    cols.BENCHMARK_CONTRIB_SIMPLE,
                    cols.PORTFOLIO_CONTRIB_SMOOTHED,
                    cols.BENCHMARK_CONTRIB_SMOOTHED,
                    cols.ALLOCATION_EFFECT_SIMPLE,
                    cols.SELECTION_EFFECT_SIMPLE,
                    cols.ALLOCATION_EFFECT_SMOOTHED,
                    cols.SELECTION_EFFECT_SMOOTHED,
                ),
            ),
        ):
            # Determine the explicit column list to unpivot on. The third
            # parameter set uses regex-like strings (e.g. ".*\.pcs$") to
            # indicate "all columns ending with the suffix". Polars does not
            # accept raw regex strings as column names, so expand any string
            # patterns to the matching column names from the source frame.
            for idx, col_names in enumerate(parms[1]):
                if isinstance(col_names, (list, tuple)):
                    on_cols = col_names
                else:
                    # col_names is expected to be a regex-like string. Compile
                    # it and filter the available columns from the source DataFrame.
                    pattern = re.compile(col_names)
                    available = parms[0].columns
                    on_cols = [c for c in available if pattern.match(c)]
                if not on_cols:
                    # Nothing to unpivot for this pattern; skip.
                    continue
                columns.append(
                    parms[0]
                    .lazy()
                    .unpivot(
                        on=on_cols,
                        index=[cols.BEGINNING_DATE, cols.ENDING_DATE],
                        value_name=parms[2][idx],
                    )
                    .with_columns(
                        pl.col("variable")
                        .str.slice(0, pl.col("variable").str.len_chars() - 4)
                        .alias(cols.CLASSIFICATION_IDENTIFIER)
                    )
                    .drop("variable")
                )

        # Horizontally join all of the LazyFrame columns into the result.
        result = pl.LazyFrame()
        for idx, column in enumerate(columns):
            if idx == 0:
                # Start with the dates, CLASSIFICATION_IDENTIFIER, and CLASSIFICATION_NAME.
                result = column.join(
                    self._classification.df.lazy(),
                    left_on=cols.CLASSIFICATION_IDENTIFIER,
                    right_on=cols.CLASSIFICATION_IDENTIFIER,
                    how="left",
                )
                # The CLASSIFICATION_NAME will be missing if the CLASSIFICATION_IDENTIFER is not
                # in self._classification.df.  So put the CLASSIFICATION_IDENTIFER in the
                # CLASSIFICATION_NAME.
                result = result.with_columns(
                    pl.col(cols.CLASSIFICATION_NAME).fill_null(
                        pl.col(cols.CLASSIFICATION_IDENTIFIER)
                    )
                )
            else:
                # Then join all of the other columns.
                result = result.join(
                    column,
                    on=[cols.BEGINNING_DATE, cols.ENDING_DATE, cols.CLASSIFICATION_IDENTIFIER],
                )

        # Create "active" columns and "total" columns, which are mathematical expressions of
        # existing columns.
        expressions: list[pl.Expr] = [
            # ACTIVE_RETURN
            (pl.col(cols.PORTFOLIO_RETURN) - pl.col(cols.BENCHMARK_RETURN)).alias(
                cols.ACTIVE_RETURN
            ),
            # ACTIVE_WEIGHT
            (pl.col(cols.PORTFOLIO_WEIGHT) - pl.col(cols.BENCHMARK_WEIGHT)).alias(
                cols.ACTIVE_WEIGHT
            ),
            # ACTIVE_CONTRIB_SIMPLE
            (pl.col(cols.PORTFOLIO_CONTRIB_SIMPLE) - pl.col(cols.BENCHMARK_CONTRIB_SIMPLE)).alias(
                cols.ACTIVE_CONTRIB_SIMPLE
            ),
            # ACTIVE_CONTRIB_SMOOTHED
            (
                pl.col(cols.PORTFOLIO_CONTRIB_SMOOTHED) - pl.col(cols.BENCHMARK_CONTRIB_SMOOTHED)
            ).alias(cols.ACTIVE_CONTRIB_SMOOTHED),
            # TOTAL_EFFECT_SMOOTHED
            (
                pl.col(cols.ALLOCATION_EFFECT_SMOOTHED) + pl.col(cols.SELECTION_EFFECT_SMOOTHED)
            ).alias(cols.TOTAL_EFFECT_SMOOTHED),
            # TOTAL_EFFECT_SIMPLE
            (pl.col(cols.ALLOCATION_EFFECT_SIMPLE) + pl.col(cols.SELECTION_EFFECT_SIMPLE)).alias(
                cols.TOTAL_EFFECT_SIMPLE
            ),
        ]
        result = result.with_columns(expressions).sort(
            cols.BEGINNING_DATE, cols.CLASSIFICATION_IDENTIFIER
        )

        # Return the resulting LazyFrame.
        return result

    def _ending_date(self) -> dt.date:
        """Return the last ending date in the attribution period.

        Returns:
            Overall ending date.
        """
        return cast(dt.date, self._performances[0].df[cols.ENDING_DATE].item(-1))  # cast for mypy

    def _equalize_columns(self) -> None:
        """Equalize portfolio and benchmark return, weight, and contribution columns.

        Missing identifiers are added to the opposite performance with zero-valued
        return, weight, and contribution columns so that portfolio and benchmark matrix
        operations use matching column sets.
        """
        # Set the portfolio and benchmark
        portfolio, benchmark = self._performances

        # Make sure that the portfolio and benchmark have the same return_columns,
        # weight_columns and contrib_columns.
        for target, source in ((portfolio, benchmark), (benchmark, portfolio)):
            missing_return_col_names: list[str] | set[str] = set(source.col_names(RET)) - set(
                target.col_names(RET)
            )
            if 0 < len(missing_return_col_names):
                # Set the missing_col_names.
                missing_return_col_names = list(missing_return_col_names)
                missing_col_names = (
                    missing_return_col_names
                    + cols.col_names(missing_return_col_names, WGT)
                    + cols.col_names(missing_return_col_names, CON)
                )
                # Add the missing_col_names to the dataframe.
                target.reset_df(target.df.hstack(source.df[missing_col_names] * 0))

    def _fetch_dataframe(
        self,
        view: View,
        columns_to_sort: str | Sequence[str] = util.EMPTY,
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
        if not util.is_empty(columns_to_sort) and view != View.CUMULATIVE_ATTRIBUTION:
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
        performance: Performance,
    ) -> pl.LazyFrame:
        """Add horizontal totals and cumulative columns to an attribution LazyFrame.

        Args:
            lf: LazyFrame containing the base attribution columns.
            performance: Portfolio or benchmark Performance instance used to provide
                the common column names.

        Returns:
            LazyFrame with simple and smoothed contribution totals, allocation totals,
            selection totals, total effects, active columns, and cumulative columns.
        """
        # Horizontally sum the contributions, allocation effects and selection effects.
        # parameters = (col_names, alias)
        expressions: list[pl.Expr] = []
        for col_names, alias in (
            (performance.col_names(PCS), cols.PORTFOLIO_CONTRIB_SIMPLE),
            (performance.col_names(BCS), cols.BENCHMARK_CONTRIB_SIMPLE),
            (performance.col_names(PCL), cols.PORTFOLIO_CONTRIB_SMOOTHED),
            (performance.col_names(BCL), cols.BENCHMARK_CONTRIB_SMOOTHED),
            (performance.col_names(AES), cols.ALLOCATION_EFFECT_SIMPLE),
            (performance.col_names(SES), cols.SELECTION_EFFECT_SIMPLE),
            (performance.col_names(AEL), cols.ALLOCATION_EFFECT_SMOOTHED),
            (performance.col_names(SEL), cols.SELECTION_EFFECT_SMOOTHED),
        ):
            expressions.append(pl.sum_horizontal(col_names).alias(alias))
        lf = lf.with_columns(expressions)

        # Horizontally sum the total effects.
        # parameters = (col_names, alias)
        expressions = []
        for col_names, alias in (
            (
                [cols.ALLOCATION_EFFECT_SIMPLE, cols.SELECTION_EFFECT_SIMPLE],
                cols.TOTAL_EFFECT_SIMPLE,
            ),
            (
                [cols.ALLOCATION_EFFECT_SMOOTHED, cols.SELECTION_EFFECT_SMOOTHED],
                cols.TOTAL_EFFECT_SMOOTHED,
            ),
        ):
            expressions.append(pl.sum_horizontal(col_names).alias(alias))
        lf = lf.with_columns(expressions)

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
        line1 = (
            self._performances[0].name
            if (
                chart_or_view
                in (Chart.HEATMAP_PORTFOLIO_CONTRIBUTION, Chart.HEATMAP_PORTFOLIO_RETURN)
            )
            else f"{self._performances[0].name} vs {self._performances[1].name}"
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
                and (not util.is_empty(self._classification_label))
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
        columns_to_sort: str | Sequence[str] = util.EMPTY,
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
        """
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
                if util.is_empty(columns_to_sort):
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
                if util.is_empty(columns_to_sort):
                    columns_to_sort = cols.PORTFOLIO_CONTRIB_SMOOTHED
                    sort_descendings = True
                # Set the DataFrame and remove the last "Total" row.
                df = self.to_polars(View.OVERALL_ATTRIBUTION, columns_to_sort, sort_descendings)[
                    :-1
                ]
                # Get the chart png
                png = format_chart.overall_contribution(
                    df, title_lines, self._performances[0].name, self._performances[1].name
                )

        # Return the chart png
        return png

    def to_html(
        self,
        view: View,
        columns_to_sort: str | Sequence[str] = util.EMPTY,
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
        return self.to_table(view, columns_to_sort, sort_descendings).as_raw_html(make_page=True)

    def to_json(
        self,
        view: View,
        columns_to_sort: str | Sequence[str] = util.EMPTY,
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
        return self.to_pandas(view, columns_to_sort, sort_descendings).to_json(  # type: ignore
            double_precision=float_precision, date_format="iso"
        )

    def to_pandas(
        self,
        view: View,
        columns_to_sort: str | Sequence[str] = util.EMPTY,
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
        return self._fetch_dataframe(view, columns_to_sort, sort_descendings).to_pandas()

    def to_polars(
        self,
        view: View,
        columns_to_sort: str | Sequence[str] = util.EMPTY,
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
        columns_to_sort: str | Sequence[str] = util.EMPTY,
        sort_descendings: bool | Sequence[bool] = False,
    ) -> gt.GT:
        """Return a Great Tables object for a view.

        Args:
            view: View to render.
            columns_to_sort: Optional column name or sequence of column names to sort
                by.
            sort_descendings: Boolean or sequence of booleans indicating whether the
                corresponding sort columns should be sorted descending.

        Returns:
            Great Tables object for the requested view.

        Raises:
            PpaError: If the requested view has more than 500 rows or view construction
                fails validation.
        """
        # Set the df
        df = self._fetch_dataframe(view, columns_to_sort, sort_descendings)

        # If there are more than a few hundred lines in an html file, then Attribution.to_html()
        # can be VERY slow.  This can occur when requesting html for a View that has one line for
        # each sub-period and classification item.  For instance, if the user requests to see 100
        # days of 100 securities for View.SUBPERIOD_ATTRIBUTION.  The underlying problem is that
        # Attribution.to_html() calls "great_tables" GT.as_raw_html(), which is inherently slow.
        # It is designed for small tables.  So there is not much that can be done for this problem.
        if 500 < len(df):
            raise PpaError(f"{view.value}, Rows = {len(df)}", 204)

        # Create a great_table.  It slows down DRAMATICALLY if you do not convert the df to pandas!
        table = gt.GT(df.to_pandas())
        title, subtitle = self._title_lines(view)
        table = table.tab_header(title=title, subtitle=subtitle)

        # Now that you have the table template, create the specific table.
        match view:
            case View.CUMULATIVE_ATTRIBUTION:
                table = format_table.cumulative_attribution(table)
            case View.OVERALL_ATTRIBUTION:
                table = format_table.overall_attribution(table, self._classification_label)
            case View.SUBPERIOD_ATTRIBUTION:
                table = format_table.subperiod_attribution(table, self._classification_label)
            case View.SUBPERIOD_SUMMARY:
                table = format_table.subperiod_summary(table)

        # Return the table.
        return table

    def to_xml(
        self,
        view: View,
        columns_to_sort: str | Sequence[str] = util.EMPTY,
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
        return self.to_pandas(view, columns_to_sort, sort_descendings).to_xml()

    def write_csv(
        self,
        view: View,
        file_path: str,
        columns_to_sort: str | Sequence[str] = util.EMPTY,
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
        self._fetch_dataframe(view, columns_to_sort, sort_descendings).write_csv(
            file_path, float_precision=float_precision
        )
