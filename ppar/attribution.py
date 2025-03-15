"""
The Attribution class contains the portfolio Performance, the benchmark Performance, the
Classification, and the reulting contribution and attribution effects.

An Attribution instance can be created using the method Analytics.get_attribution().

The public methods to retrieve the resulting output are:
    1. to_chart(chart)
    2. to_html(view)
    3. to_json(view)
    4. to_pandas(view)
    5. to_polars(view)
    6. to_table(view)
    7. to_xml(view)
    8. write_csv(view)
"""

## Overrides for pylint
# pylint: disable=too-many-lines


# Python Imports
from enum import Enum
import datetime as dt
from typing import cast

# Third-Party Imports
import great_tables as gt
import numpy as np
import pandas as pd
import polars as pl

# Project Imports
from ppar.classification import Classification
import ppar.columns as cols
from ppar.columns import AEL, AES, BCL, BCS, CON, PCL, PCS, RET, SEL, SES, WGT
import ppar.errors as errs
import ppar.format_chart as format_chart
import ppar.format_table as format_table
from ppar.frequency import Frequency
from ppar.performance import Performance
import ppar.utilities as util

# Constants
_DEFAULT_OUTPUT_PRECISION = 8


class Chart(Enum):
    """
    Summary:
        An enumeration of the different types of attribution charts.
    Args:
        Enum (_type_): enum
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
    """
    Summary:
        An enumeration of the different types of attribution views.
    Args:
        Enum (_type_): enum
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
    """
    The Attribution class contains the portfolio Performance, the benchmark Performance, the
    Classification, and the reulting contribution and attribution effects.

    The public methods to retrieve the resulting output are:
        1. to_chart(chart)
        2. to_html(view)
        3. to_pandas(view)
        4. to_polars(view)
        5. to_table(view)
        6. write_csv(view)
    """

    def __init__(
        self,
        performances: tuple[Performance, Performance],
        classification_name: str,
        classification_data_source: util.TypeClassificationDataSource,
        frequency: Frequency,
        classification_label: str = util.EMPTY,
    ):
        """
        Summary:
            The constructor.  Calculates the contribution and attribution effects.

        Args:
            performances (tuple[Performance, Performance]): portfolio=0, benchmark=1
            classification_name (str): The classification_name for which the contribution and
                attribution effects will be calculated.
            classification_data_source (TypeClassificationDataSource): One of the following:
                1. A csv file path containing the Classification data.
                2. A dictionary containing the Classification data.
                3. A pandas or polars DataFrame containing the Classification data.
            frequency (Frequency): The Frequency.
            classification_label (str, optional): The classification label that will be displayed
                in the tables and charts.  Defaults to util.EMPTY.

        Data Parameters:
            Sample input for the "classification_data" parameter of a "Security" Classification:
                AAPL, Apple Inc.
                MSFT, Microsft
                ...
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

        # Initialize the cached dataframes for each View.
        self._dataframes_for_views: dict[View, pl.DataFrame] = {}

    def _add_total_row(self, df: pl.DataFrame, have_classification_columns: bool) -> pl.DataFrame:
        """
        Add a total row to the bottom of df.

        Args:
            df (pl.DataFrame): The dataframe.
            have_classification_columns (bool): Indicates if the dataframe has classification
              columns.

        Returns:
            pl.DataFrame: The dataframe with a total row added to the bottom.
        """
        # Start the total_row as a sum of df.
        total_row = df.sum()

        # The classification identifier will have 'None', so make it blank.
        if have_classification_columns:
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
        """Audit the Attribution (self)."""
        # Audit the portfolio/benchmark pair of performance objects.
        Performance.audit_performances(
            self._performances,
            self._beginning_date(),
            self._ending_date(),
            self._classification.name,
        )

        # Assert that df and df_overall have the same columns.
        assert (
            self._df.columns == self._df_overall.columns
        ), f"{errs.ERROR_999_UNEXPECTED}Attr._audit(): df columns != df_overall columns."

        # Audit all columns.
        Attribution._audit_columns(self._df, self._df_overall)

    @staticmethod
    def audit_attributions(attributions: list["Attribution"]) -> None:
        """Audit the list of Attributions."""
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
                    assert equiv.equals(base_equivalent_columns[idxe])

    @staticmethod
    def _audit_columns(
        df: pl.DataFrame, df_overall: pl.DataFrame, do_assert_simple_column_pairs: bool = True
    ) -> None:
        """
        Audit the columns in the dataframes.

        Args:
            df (pl.DataFrame): The detailed dataframe.
            df_overall (pl.DataFrame): The overall (total row) dataframe.
            do_assert_simple_columns (bool): Assert _SIMPLE_COLUMN_PAIRS_THAT_SHOULD_BE_EQUAL.
        """
        # Assert that certain simple column pairs in df should be equal.
        if do_assert_simple_column_pairs:
            for col1, col2 in _SIMPLE_COLUMN_PAIRS_THAT_SHOULD_BE_EQUAL:
                if col1 in df.columns and col2 in df.columns:
                    assert (
                        df[col1].round(7).equals(df[col2].round(7))
                    ), f"{errs.ERROR_999_UNEXPECTED}_audit_columns() df: {col1} <> {col2}."

        # Audit df_overall.
        if not df_overall.is_empty():
            # Assert that certain column pairs in df_overall should be equal.
            for col1, col2 in _OVERALL_COLUMN_PAIRS_THAT_SHOULD_BE_EQUAL:
                if col1 in df_overall.columns and col2 in df_overall.columns:
                    assert (
                        df_overall[col1].round(7).equals(df_overall[col2].round(7))
                    ), f"{errs.ERROR_999_UNEXPECTED}_audit_columns() df_overall: {col1} <> {col2}."

            # Assert that the vertical sum of the smoothed columns of df is equal to df_overall.
            for col_name in cols.ALL_SMOOTHED_COLUMNS:
                assert util.are_near(
                    df[col_name].sum(), df_overall[col_name].item(0), util.Tolerance.MEDIUM
                ), f"{errs.ERROR_999_UNEXPECTED}_audit_cols: {col_name} does not foot when summed."

    def _audit_view(self, view: View) -> None:
        """Audit the view."""
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
                    assert (
                        df[needed_columns[2]].round(11) == contributions.round(11)
                    ).all(), (
                        f"{errs.ERROR_999_UNEXPECTED}audit_view(): weight * return != contribution"
                    )

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
        """
        Get the overall beginning date.

        Returns:
            dt.date: The overall beginning date.
        """
        return cast(dt.date, self._performances[0].df[cols.BEGINNING_DATE].item(0))

    def _calculate_attribution(self) -> pl.LazyFrame:
        """
        Calculates the contributions and Attribution effects for the portfolio vs benchmark.

        Returns:
            pl.LazyFrame: Multiple rows (one for each subperiod) with contributions and Attribution
            effects for the portfolio vs benchmark.  This will be self.df.
        """
        # Set the portfolio and benchmark.
        portfolio, benchmark = self._performances

        # Get pre-computed values.
        portfolio_consolidated_returns = portfolio.consolidated_returns()
        benchmark_consolidated_returns = benchmark.consolidated_returns()
        portfolio_linking_coefficients = portfolio.linking_coefficients()  # for contribution
        benchmark_linking_coefficients = benchmark.linking_coefficients()  # for contribution
        portfolio_overall_return = portfolio.overall_return()
        benchmark_overall_return = benchmark.overall_return()
        portfolio_total_returns = portfolio.df[cols.TOTAL_RETURN]  # pl.Series
        benchmark_total_returns = benchmark.df[cols.TOTAL_RETURN]  # pl.Series

        # Must pre-compute the weight dfs because you cannot do arithmetic on 2 LazyFrames.
        portfolio_weights = portfolio.df.lazy().select(portfolio.col_names(WGT)).collect()
        benchmark_weights = benchmark.df.lazy().select(benchmark.col_names(WGT)).collect()

        # Calculate the attribution linking_coefficients
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
            # Simple Brinson-Fachler allocation effects for each subperiod.
            .with_columns(
                (
                    (benchmark_consolidated_returns - benchmark_total_returns)
                    * (portfolio_weights - benchmark_weights)
                ).rename(lambda column_name: f"{column_name[:-4]}{AES}")
            )
            # Simple Brinson-Fachler selection effects for each subperiod.
            .with_columns(
                (
                    portfolio_weights
                    * (portfolio_consolidated_returns - benchmark_consolidated_returns)
                ).rename(lambda column_name: f"{column_name[:-4]}{SES}")
            )
            .with_columns(
                [
                    # Smoothed (log-linked) portfolio contribution
                    *[
                        (pl.col(f"{id}{PCS}") * portfolio_linking_coefficients).alias(f"{id}{PCL}")
                        for id in portfolio.identifiers
                    ],
                    # Smoothed (log-linked) benchmark contribution
                    *[
                        (pl.col(f"{id}{BCS}") * benchmark_linking_coefficients).alias(f"{id}{BCL}")
                        for id in benchmark.identifiers
                    ],
                    # Smoothed (log-lLinked) Brinson-Fachler allocation effects for each subperiod.
                    *[
                        (pl.col(f"{id}{AES}") * linking_coefficients).alias(f"{id}{AEL}")
                        for id in portfolio.identifiers
                    ],
                    # Smoothed (log-lLinked) Brinson-Fachler selection effects for each subperiod.
                    *[
                        (pl.col(f"{id}{SES}") * linking_coefficients).alias(f"{id}{SEL}")
                        for id in portfolio.identifiers
                    ],
                    # Portfolio Return
                    portfolio_total_returns.alias(cols.PORTFOLIO_RETURN),
                    # Benchmark Return
                    benchmark_total_returns.alias(cols.BENCHMARK_RETURN),
                ]
            )
        )

        # Append columns that are the horizontal summations of the contributions and attribution
        # effects.  And vertically sum the cumulative columns.
        lf = self._sum_columns_and_rows(lf, portfolio)

        # Return lazy version of self.df.
        return lf

    def _calculate_df_overall(self) -> pl.DataFrame:
        """
        Calculate df_overall, which is one total row for the entire overall period.

        Returns:
            pl.DataFrame: df_overall, which is one row for the entire overall period.
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
        """
        Constructs the DataFrame for the detailed Views.

        Args:
            view (View): The detailed View.

        Raises:
            errs.PpaError: Unhandled View

        Returns:
            pl.LazyFrame: The appropriate detail LazyFrame.
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
                raise errs.PpaError(
                    f"{errs.ERROR_999_UNEXPECTED}"
                    f"Unhandled View {view} in Attribution._construct_df_detail()"
                )

        # Do parameter-driven un-pivots to build the list of LazyFrame columns.
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
                    f"^*{PCS}$",
                    f"^*{BCS}$",
                    f"^*{PCL}$",
                    f"^*{BCL}$",
                    f"^*{AES}$",
                    f"^*{SES}$",
                    f"^*{AEL}$",
                    f"^*{SEL}$",
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
            for idx, col_names in enumerate(parms[1]):
                columns.append(
                    parms[0]
                    .lazy()
                    .unpivot(
                        on=col_names,
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
        """
        Get the overall ending date.

        Returns:
            dt.date: The overall ending date.
        """
        return cast(dt.date, self._performances[0].df[cols.ENDING_DATE].item(-1))  # cast for mypy

    def _equalize_columns(self) -> None:
        """
        Make sure that the portfolio and benchmark have the same return_columns, weight_columns
        and contrib_columns.  This is necessary so we can do polars matrix math.
            1. If the portfolio is missing an item that is in the benchmark, then the portfolio
            will get the benchmark item with all zero weights, returns and contribs.  And
            vice-versa.
            2. Since all columns are added with zeroes, this has no actual effect.
            3. Note that the portfolio and benchmark will have the exact same col_names after this.
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

    def _fetch_dataframe(self, view: View) -> pl.DataFrame:
        """
        Fetch the DataFrame associated with the view.

        Args:
            view (View): The view.

        Returns:
            pl.DataFrame: The DataFrame associated with the view.
        """
        # Check the cache
        if view in self._dataframes_for_views:
            return self._dataframes_for_views[view]

        # Get the base dataframe associated with the view.
        match view:
            case View.CUMULATIVE_ATTRIBUTION | View.SUBPERIOD_SUMMARY:
                lf = self._df.lazy()
            case _:  # View.SUBPERIOD_ATTRIBUTION | View.OVERALL_ATTRIBUTION
                lf = self._construct_df_for_detail_views(view)

        # Select only the needed columns.
        lf = lf.select(_VIEW_COLUMN_NAMES[view])

        # Determine if the view has classification columns.
        have_classification_columns = cols.CLASSIFICATION_IDENTIFIER in lf.collect_schema().keys()

        # Must collect() before adding the total_row
        df = lf.collect()

        # Add the total_row
        if view in (View.CUMULATIVE_ATTRIBUTION, View.OVERALL_ATTRIBUTION):
            df = self._add_total_row(df, have_classification_columns)

        # Return the dataframe with the total_row.
        self._dataframes_for_views[view] = df
        return self._dataframes_for_views[view]

    def _sum_columns_and_rows(
        self,
        lf: pl.LazyFrame,
        performance: Performance,
    ) -> pl.LazyFrame:
        """
        Horizontally append columns that are the horizontal summations of the contributions and
        attribution effects.  Also create vertical cumulative sums.

        Args:
            lf (pl.LazyFrame): The LazyFrame.
            performance (Performance): Either the portfolio or benchmark Performance instance.
                Since both the portfolio and benchmark have the same column names, either one will
                suffice.

        Returns:
            pl.LazyFrame: The new lf with the new horizontal summation columns and cumulative
            columns added.
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
        """
        Return the title lines for a View or a Chart.

        Args:
            chart_or_view (Chart | View): The type of Chart or View.
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

    def to_chart(self, chart: Chart) -> bytes:
        """
        Returns an in-memory png of the chart corresponding to the chart type.

        Args:
            chart (Chart): The chart type.

        Returns:
            bytes: An in-memory png of the chart corresponding to the chart type.
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
                # Set the DataFrame and remove the last "Total" row.
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
                # Set the DataFrame.
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
                # Get the chart png
                png = format_chart.heatmap(df, column_name, title_lines)

            case Chart.SUBPERIOD_ATTRIBUTION | Chart.SUBPERIOD_RETURN:
                # Set the DataFrame.
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

            case _:  # Chart.OVERALL_ATTRIBUTION | Chart.OVERALL_CONTRIBUTION:
                # Set the DataFrame and remove the last "Total" row.
                df = self.to_polars(View.OVERALL_ATTRIBUTION)[:-1]
                # Get the chart png
                match chart:
                    case Chart.OVERALL_ATTRIBUTION:
                        png = format_chart.overall_attribution(df, title_lines)
                    case Chart.OVERALL_CONTRIBUTION:
                        png = format_chart.overall_contribution(
                            df, title_lines, self._performances[0].name, self._performances[1].name
                        )

        # Return the chart png
        return png

    def to_html(self, view: View) -> str:
        """
        Returns the view as an html string.

        Args:
            view (View): The desired View.

        Returns:
            str: The view as an html string.
        """
        return self.to_table(view).as_raw_html(make_page=True)

    def to_json(self, view: View, float_precision: int = _DEFAULT_OUTPUT_PRECISION) -> str:
        """
        Returns the view as a json string.

        Args:
            view (View): The desired View.
            float_precision (int, optional): The quantity of decimal places.
                Defaults to _DEFAULT_OUTPUT_PRECISION.

        Returns:
            str: The view as a json string.
        """
        return self.to_pandas(view).to_json(double_precision=float_precision)  # type: ignore

    def to_pandas(self, view: View) -> pd.DataFrame:
        """
        Returns the view as a pandas DataFrame.

        Args:
            view (View): The desired View.

        Returns:
            str: The view as a pandas DataFrame.
        """
        return self._fetch_dataframe(view).to_pandas()

    def to_polars(self, view: View) -> pl.DataFrame:
        """
        Returns the view as a polars DataFrame.

        Args:
            view (View): The desired View.

        Returns:
            str: The view as a polars DataFrame.
        """
        return self._fetch_dataframe(view)

    def to_table(self, view: View) -> gt.GT:
        """
        Returns a "great_table" of the view.

        Args:
            view (View): The desired View.

        Returns:
            gt.GT: A "great_table" of the view.
        """
        # Set the df
        df = self._fetch_dataframe(view)

        # If there are more than a few hundred lines in an html file, then Attribution.to_html()
        # can be VERY slow.  This can occur when requesting html for a View that has one line for
        # each sub-period and classification item.  For instance, if the user requests to see 100
        # days of 100 securities for View.SUBPERIOD_ATTRIBUTION.  The underlying problem is that
        # Attribution.to_html() calls "great_tables" GT.as_raw_html(), which is inherently slow.
        # It is designed for small tables.  So there is not much that can be done for this problem.
        assert len(df) < 500, f"{errs.ERROR_204_TOO_MANY_HTML_ROWS}{view.value}, Rows = {len(df)}"

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

    def to_xml(self, view: View) -> str:
        """
        Returns the view as an xml string.

        Args:
            view (View): The desired View.

        Returns:
            str: The view as an xml string.
        """
        return self.to_pandas(view).to_xml()

    def write_csv(
        self,
        view: View,
        file_path: str,
        float_precision: int = _DEFAULT_OUTPUT_PRECISION,
    ) -> None:
        """
        Writes a csv file of the view.

        Args:
            view (View): The desired View.
            file_path (str): The file path of the csv file to be written to.
            float_precision (int, optional): The quantity of decimal places.
                Defaults to _DEFAULT_OUTPUT_PRECISION.
        """
        self._fetch_dataframe(view).write_csv(file_path, float_precision=float_precision)
