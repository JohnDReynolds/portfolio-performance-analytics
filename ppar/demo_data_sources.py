"""
This module contains custom functions for the Classification, Mapping, and Performance data
sources.  It has been designed for the test data.  The functions in this module deliver the path
of csv files containing the data.  Users can alternatively create their own custom data source
functions that query databases and then deliver pandas dataframes, polars dataframes, or python
dictionaries.
"""

# Python Imports
from importlib.resources import files
from pathlib import Path

# Project Imports
from ppar.analytics import Analytics
import ppar.utilities as util

# Directory containing the demo data.
_DEMO_DATA_DIRECTORY = files("ppar.demo_data")


def _demo_data_path(relative_path: str) -> Path:
    """Return a filesystem path for a packaged demo-data resource."""
    return Path(str(_DEMO_DATA_DIRECTORY.joinpath(relative_path)))


def classification_data_source(
    classification_name: str | None = None,
) -> util.PathLike:
    """
    This is a custom function for the Classification data source.  It has been designed for the
    test data.  Users can create their own function(s) to deliver the data.

    Args:
        classification_name (str | None, optional): The Classification name.
            Defaults to None.

    Returns:
        Path | str: The classification file path, or util.EMPTY when no classification was
        requested.
    """
    classification_name = util.normalize_optional_string(classification_name)

    # Return util.EMPTY if the classification_name is empty.
    if util.is_empty(classification_name):
        return util.EMPTY

    # Return the path to the csv file containing the classification data.
    return _demo_data_path(f"classifications/{classification_name}.csv")


def mapping_data_sources(
    analytics: Analytics, to_classification_name: str | None = None
) -> tuple[util.MappingDataSource, util.MappingDataSource]:
    """
    This is a custom function for the Mapping data sources.  It has been designed for the
    test data.  Users can create their own function(s) to deliver the data.

    Args:
        analytics (Analytics): The Analytics instance.
        to_classification_name (str, optional): The Classification name to map to.
            Defaults to None.

    Returns:
        tuple[util.MappingDataSource, util.MappingDataSource]: A tuple of 2 mapping
        data sources (0 = Portfolio Data Source, 1 = Benchmark Data Source)
    """
    to_classification_name = util.normalize_optional_string(to_classification_name)

    # Return (util.EMPTY, util.EMPTY) if the classification_name is empty.
    if util.is_empty(to_classification_name):
        return (util.EMPTY, util.EMPTY)

    # Build the tuple of mapping data sources containing the csv file paths.
    mapping_list: list[util.MappingDataSource] = [
        (
            util.EMPTY
            if from_classification_name == to_classification_name
            else _demo_data_path(
                f"mappings/{from_classification_name}--to--{to_classification_name}.csv"
            )
        )
        for from_classification_name in analytics.classification_names()
    ]

    # Return the tuple of mapping data sources containing the csv file paths.
    return (mapping_list[0], mapping_list[1])


def performance_data_source(performance_name: str) -> Path:
    """
    This is a custom function for the Performance data source.  It has been designed for the
    test data.  Users can create their own function(s) to deliver the data.

    Args:
        performance_name (str): The performance name.

    Returns:
        Path: The path of the performance file corresponding to performance_name.
    """
    # Return the path of the performance file corresponding to performance_name.
    return _demo_data_path(f"performance/{performance_name}")
