"""
Copyright 2025 John D. Reynolds, All Rights Reserved.

This software and associated documentation files (the “Software”) are provided for reference or
archival purposes only. No license is granted to any person to copy, modify, publish, distribute,
sublicense, or sell copies of the Software, or to use it for any commercial or non-commercial
purpose, without the express prior written permission of John D. Reynolds.

The Mapping class supports mapping from one Classification to another.
"""

# Third-party imports
import pandas as pd
import polars as pl

# Project Imports
import ppa.errors as errs
import ppa.utilities as util


class Mapping:
    """
    Mapping class.  Supports mapping from one Classification to another.
    """

    def __init__(
        self,
        from_items_to_map: list[str],
        data_source: util.TypeMappingDataSource,
    ):
        """
        The constructor for creating a mapping from one classification to another.

        Args:
            from_items_to_map (list[str]): A list of the from items to map.
            data_source (TypeMappingDataSource): One of the following:
                1. A csv file path containing the Mapping data.
                2. A dictionary containing the Mapping data.
                3. A pandas or polars DataFrame containing the Mapping data.

        Data Parameters:
            Sample input for the "data_source" parameter for "Security" to "Gics Sub-Industry":
                AAPL, 45202030
                MSFT, 45103020
                ...
        """
        if isinstance(data_source, dict):
            # A dictionary: key = from_item, value = to_item
            mappings = data_source
        elif isinstance(data_source, (pd.DataFrame, pl.DataFrame)):
            # If mapping_data is a DataFrame, then use the first column as the from-key and the
            # second column as the to-value.
            assert 2 <= len(data_source.columns), errs.ERROR_353_MAPPING_MUST_CONTAIN_2_COLUMNS
            mappings = dict(
                zip(
                    data_source[data_source.columns[0]],  # type: ignore
                    data_source[data_source.columns[1]],  # type: ignore
                )
            )
        else:  # isinstance(data_source, str):
            # A csv file path that will be loaded into the mappings dictionary.
            mappings = util.load_dictionary_from_csv(data_source)

        # Put the needed mappings in self.mappings and return.
        self.mappings = util.needed_mappings(mappings, from_items_to_map)
