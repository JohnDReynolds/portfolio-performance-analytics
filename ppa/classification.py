"""
The Classification class contains a DataFrame of it's associated items.
"""

# Third-Party Imports
import pandas as pd
import polars as pl

# Project Imports
import ppa.columns as cols
import ppa.errors as errs
from ppa.performance import Performance
import ppa.utilities as util

_EMPTY_DF = pl.DataFrame(
    {cols.CLASSIFICATION_IDENTIFIER: (util.EMPTY,), cols.CLASSIFICATION_NAME: (util.EMPTY,)}
)


class Classification:
    """
    The Classification class contains a DataFrame of it's associated items.
    """

    def __init__(
        self,
        name: str,
        data_source: util.TypeClassificationDataSource,
        performances: tuple[Performance, Performance] | None = None,
    ):
        """
        Constructs a DataFrame of the Classification and items corresponding to the name parameter

        Args:
            name (str): The Classification name.
            data_source (TypeClassificationDataSource): One of the following:
                1. A csv file path containing the Classification data.
                2. A dictionary containing the Classification data.
                3. A pandas or polars DataFrame containing the Classification data.
            performances (tuple[Performance, Performance] | None, optional): The portfolio
                Performance and the benchmark Performance. Defaults to None.

        Data Parameters:
            Sample input for the "data_source" parameter of a "Security" Classification would be:
                AAPL, Apple Inc.
                MSFT, Microsft
                ...
        """
        # Set the class members.
        self.name = name

        # Get the 2-column dataframe [cols.CLASSIFICATION_IDENTIFIER, cols.CLASSIFICATION_NAME]
        if isinstance(data_source, str):
            if util.is_empty(data_source):
                self.name, self.df = Classification._load_from_performances(performances)
            else:
                self.df = pl.read_csv(
                    data_source,
                    has_header=False,
                    infer_schema=False,  # Will force both columns to be the default strings (Utf8)
                )
        elif isinstance(data_source, dict):
            self.df = pl.DataFrame(
                {
                    cols.CLASSIFICATION_IDENTIFIER: data_source.keys(),
                    cols.CLASSIFICATION_NAME: data_source.values(),
                }
            )
        elif isinstance(data_source, pd.DataFrame):
            self.df = pl.from_pandas(data_source)
        else:  # isinstance(data_source, pl.DataFrame):
            self.df = data_source

        # Assert that you have 2 columns, and then make sure that they have the correct names.
        assert 2 == len(self.df.columns), errs.ERROR_302_CLASSIFICATION_MUST_CONTAIN_2_COLUMNS
        self.df.columns = [cols.CLASSIFICATION_IDENTIFIER, cols.CLASSIFICATION_NAME]

        # All identifiers need to be strings for classifications, mappings, performances, etc.
        if not isinstance(self.df.schema[cols.CLASSIFICATION_IDENTIFIER], pl.String):
            self.df = self.df.with_columns(self.df[cols.CLASSIFICATION_IDENTIFIER].cast(pl.Utf8))

    @staticmethod
    def _load_from_performances(
        performances: tuple[Performance, Performance] | None,
    ) -> tuple[str, pl.DataFrame]:
        """
        Use the performances.classification_items to construt the Classification dataframe.

        Args:
            performances (tuple[Performance, Performance] | None): portfolio = 0, benchmark = 1

        Returns:
            tuple[str, pl.DataFrame]: The classification self.name and classification self.df.
        """
        # Return empty if there are no performances or the portfolio and benchmark are not of the
        # same classifiation_name.
        if (not performances) or (
            performances[0].classification_name != performances[1].classification_name
        ):
            return util.EMPTY, _EMPTY_DF

        # Get the classification items from the portfolio Performance and benchmark Performance.
        # The "reversed" will process the portfolio after the benchmark.  This is so that when we
        # eventually "uniqueify" the dataframe, it will "keep" the last portfolio item instead of
        # the benchmark item.  This assumes that the user prefers the portfolio data over the
        # benchmark data.  Chances are the portfolio data came from their accounting system and the
        # benchmark data came from an external source.
        dfs = [
            performance.classification_items
            for performance in reversed(performances)
            if not performance.classification_items.is_empty()
        ]

        # Return empty if the performances do not have any classification_items.
        if not dfs:
            return util.EMPTY, _EMPTY_DF

        # Concatenate the dataframes and remove duplicates, keeping the last occurrence.
        classification_items = pl.concat(dfs, how="vertical").unique(
            subset=[cols.CLASSIFICATION_IDENTIFIER], keep="last"
        )

        # Return the classification_name that is common to both the portfolio and the benchmark.
        # Return the dataframe with the classification_items.
        return performances[0].classification_name, classification_items
