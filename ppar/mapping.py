"""
The Mapping class supports mapping from one Classification to another.
"""

# Python imports
from collections import defaultdict
from typing import Sequence

# Project Imports
import ppar.columns as cols
import ppar.errors as errs
import ppar.utilities as util


class Mapping:
    """
    Mapping class.  Supports mapping from one Classification to another.
    """

    def __init__(
        self,
        from_items_to_map: Sequence[str],
        data_source: util.MappingDataSource,
    ):
        """
        The constructor for creating a mapping from one classification to another.

        Args:
            from_items_to_map (Sequence[str]): A list of the from items to map.
            data_source (TypeMappingDataSource): One of the following:
                1. The path of a csv file containing the Mapping data.
                2. A python dictionary containing the Mapping data.
                3. A pandas DataFrame containing the Mapping data.
                4. A polars DataFrame containing the Mapping data.

        Data Parameters:
            Here is sample input data for the "data_source" parameter for mapping the "Security"
            classification to the "Economic Sector" classification.  The unique identifier of the
            "from" classification is in the first column, and the unique identifier of the "to"
            classification is in the second column.  There are no column headers.
                AAPL, IT
                GOOG, CO
                XOM,  EN
                ...
        """
        # Load the data source into dataframe with 2 columns: 0=from, 1=to
        from_tos = util.load_datasource(
            data_source,
            column_names=cols.FROM_TO_COLUMNS,
            needed_items=from_items_to_map,
            error_message=errs.ERROR_353_MAPPING_MUST_CONTAIN_2_COLUMNS,
        )

        # Turn the from_tos dataframe into a dictionary.
        mappings = dict(
            zip(
                from_tos[from_tos.columns[0]],
                from_tos[from_tos.columns[1]],
            )
        )

        # If from_item is not in mappings, then add it pointing to itself.
        mappings = {
            from_item: (from_item if from_item not in mappings else mappings[from_item])
            for from_item in from_items_to_map
        }

        # Create a reverse mapping from `to_column_name` to a list of `from_column_names`.
        self.to_froms: defaultdict[str, list[str]] = defaultdict(list)
        for from_value, to_value in mappings.items():
            self.to_froms[to_value].append(from_value)
