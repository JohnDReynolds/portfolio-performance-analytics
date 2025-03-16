"""
This module contains custom functions for the Classification, Mapping, and Performance data
sources.  It has been designed for the test data.  The functions in this module deliver the path
name of csv files containing the data.  Users can alternatively create their own custom data source
functions that query databases and then deliver pandas dataframes, polars dataframes, or python
dictionairies.
"""

# Python Imports
from importlib.resources import files

# Project Imports
from ppar.analytics import Analytics
import ppar.utilities as util

# Directory containing the demo data.
_DEMO_DATA_DIRECTORY = files("ppar.demo_data")


def classification_data_source(
    classification_name: str = util.EMPTY,
) -> util.TypeClassificationDataSource:
    """
    This is a custom function for the Classification data source.  It has been designed for the
    test data.  Users can create their own function(s) to deliver the data.

    Args:
        classification_name (str, optional): The Classification name. Defaults to util.EMPTY.

    Returns:
        str: The path name of the classification file corresponding to classification_name.
    """
    # Return util.EMPTY if the classification_name is empty.
    if util.is_empty(classification_name):
        return util.EMPTY

    # Return the path name to the csv file containing the classification data..
    return str(_DEMO_DATA_DIRECTORY.joinpath(f"classifications/{classification_name}.csv"))


def mapping_data_sources(
    analytics: Analytics, to_classification_name: str = util.EMPTY
) -> tuple[util.TypeMappingDataSource, util.TypeMappingDataSource]:
    """
    This is a custom function for the Mapping data sources.  It has been designed for the
    test data.  Users can create their own function(s) to deliver the data.

    Args:
        analytics (Analytics): The Analytics instance.
        to_classification_name (str, optional): The Classification name to map to.
            Defaults to util.EMPTY.

    Returns:
        tuple[util.TypeMappingDataSource, util.TypeMappingDataSource]: A tuple of 2 mapping
        data sources (0 = Portfolio Data Source, 1 = Benchmark Data Source)
    """
    # Return (util.EMPTY, util.EMPTY) if the classification_name is empty.
    if util.is_empty(to_classification_name):
        return (util.EMPTY, util.EMPTY)

    # Build the tuple of mapping data sources containing the csv file paths.
    mapping_list: list[str] = [
        (
            util.EMPTY
            if from_classification_name == to_classification_name
            else str(
                _DEMO_DATA_DIRECTORY.joinpath(
                    f"mappings/{from_classification_name}--to--{to_classification_name}.csv"
                )
            )
        )
        for from_classification_name in analytics.classification_names()
    ]

    # Return the tuple of mapping data sources containing the csv file paths.
    return (mapping_list[0], mapping_list[1])


def performance_data_source(performance_name: str) -> util.TypePerformanceDataSource:
    """
    This is a custom function for the Performance data source.  It has been designed for the
    test data.  Users can create their own function(s) to deliver the data.

    Args:
        performance_name (str): The performance name.

    Returns:
        TypePerformanceDataSource: The path name of the performance file corresponding to
        performance_name.
    """
    # Return the path name of the performance file corresponding to performance_name.
    return str(_DEMO_DATA_DIRECTORY.joinpath(f"performance/{performance_name}"))
