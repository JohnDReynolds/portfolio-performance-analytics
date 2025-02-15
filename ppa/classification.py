"""
Copyright 2025 John D. Reynolds, All Rights Reserved.

This software and associated documentation files (the “Software”) are provided for reference or
archival purposes only. No license is granted to any person to copy, modify, publish, distribute,
sublicense, or sell copies of the Software, or to use it for any commercial or non-commercial
purpose, without the express prior written permission of John D. Reynolds.

The Classification class contains a DataFrame of it's associated items.
"""

# Third-Party Imports
import pandas as pd
import polars as pl

# Project Imports
import ppa.columns as cols
import ppa.errors as errs
import ppa.utilities as util


class Classification:
    """
    The Classification class contains a DataFrame of it's associated items.
    """

    def __init__(self, name: str, data_source: util.TypeClassificationDataSource):
        """
        Constructs a DataFrame of the Classification and items corresponding to the name parameter

        Args:
            name (str): The Classification name.
            data_source (TypeClassificationDataSource): One of the following:
                1. A csv file path containing the Classification data.
                2. A dictionary containing the Classification data.
                3. A pandas or polars DataFrame containing the Classification data.

        Data Parameters:
            Sample input for the "data_source" parameter of a "Security" Classification would be:
                AAPL, Apple Inc.
                MSFT, Microsft
                ...
        """
        # Set the class members.
        self.name = name

        # Set the classification items dictionary.
        if isinstance(data_source, dict):
            items = data_source
        elif isinstance(data_source, (pd.DataFrame, pl.DataFrame)):
            # Use the first column as the from-key and the second column as the to-value.
            assert 2 <= len(
                data_source.columns
            ), errs.ERROR_302_CLASSIFICATION_MUST_CONTAIN_2_COLUMNS
            items = dict(
                zip(
                    data_source[data_source.columns[0]],  # type: ignore
                    data_source[data_source.columns[1]],  # type: ignore
                )
            )
        else:  # isinstance(data_source, str):
            items: dict[str, str] = (
                {util.EMPTY: util.EMPTY}
                if util.is_empty(self.name)
                else util.load_dictionary_from_csv(data_source)
            )

        # Convert keys to lower-case.
        items = {key.lower(): value for key, value in items.items()}

        # Set the dictionary to a polars DataFrame for vectorized lookups.
        self.df = pl.DataFrame(
            {
                cols.CLASSIFICATION_IDENTIFIER: list(items.keys()),
                cols.CLASSIFICATION_NAME: list(items.values()),
            }
        )
