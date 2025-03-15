"""
This module contains custom functions for the Classification, Mapping, and Performance data
sources.  It has been designed for the test data.  Users are free to create their own
function(s) to deliver the data.

The functions in this file deliver the path name of csv files containing the data.  Users
could alternatively create their own custom data source functions that query databases and then
deliver pandas dataframes, polars dataframes, or python dictionairies.
"""

# Python imports
import os

# Project imports
from ppar.analytics import Analytics
from ppar.attribution import Attribution
import ppar.utilities as util


# Directories containing the test data.
_DATA_DIRECTORIES = ("tests/data/", "../tests/data/", "data/")
_CLASSIFICATION_DIRECTORIES = [f"{dir}classifications" for dir in _DATA_DIRECTORIES]
_MAPPING_DIRECTORIES = [f"{dir}mappings" for dir in _DATA_DIRECTORIES]
_PERFORMANCE_DIRECTORIES = [f"{dir}performance" for dir in _DATA_DIRECTORIES]


def classification_data_path(classification_name: str) -> util.TypeClassificationDataSource:
    """
    This is a custom function for the Classification data source.  It has been designed for the
    test data.  Users are free to create their own function(s) to deliver the data.

    Args:
        classification_name (str): The classification name.

    Returns:
        str: The path name of the classification file corresponding to classification_name.
    """
    if util.is_empty(classification_name):
        return classification_name
    return resolve_file_path(_CLASSIFICATION_DIRECTORIES, classification_name, ".csv")


def get_attribution(
    analytics: Analytics,
    classification_name: str = util.EMPTY,
    classification_data_source: util.TypeClassificationDataSource = util.EMPTY,
    mapping_data_source: util.TypeMappingDataSource = util.EMPTY,
) -> Attribution:
    """Infer file path from the classification_name and then return the attribution."""
    if util.is_empty(classification_data_source):
        classification_data_source = classification_data_path(classification_name)

    if util.is_empty(mapping_data_source):
        mapping_data_sources = mapping_data_paths(analytics, classification_name)
    else:
        mapping_data_sources = (mapping_data_source, mapping_data_source)

    return analytics.get_attribution(
        classification_name,
        classification_data_source,
        mapping_data_sources,
    )


def html_table_lines(html_string: str) -> list[str]:
    """Get just the table lines from the html string."""
    # html_lines = html_string.split("\n")
    lines: list[str] = []
    on_table = False
    for line in html_string.split("\n"):
        if not on_table and line.startswith("<table "):
            on_table = True
        if on_table:
            lines.append(line)
    return lines


def mapping_data_paths(
    analytics: Analytics, to_classification_name: str
) -> tuple[util.TypeMappingDataSource, util.TypeMappingDataSource]:
    """
    This is a custom function for the Mapping data sources.  It has been designed for the
    test data.  Users are free to create their own function(s) to deliver the data.

    Args:
        analytics (Analytics): The Analytics instance.
        to_classification_name (str): The classification name to map to.

    Returns:
        tuple[util.TypeMappingDataSource, util.TypeMappingDataSource]: A tuple of 2 mapping
        data sources (0 = Portfolio Data Source, 1 = Benchmark Data Source)
    """
    if util.is_empty(to_classification_name):
        return (util.EMPTY, util.EMPTY)

    # Build the tuple of mapping data sources containing the csv file paths.
    mapping_list: list[str] = [
        (
            util.EMPTY
            if from_classification_name == to_classification_name
            else resolve_file_path(
                _MAPPING_DIRECTORIES,
                f"{from_classification_name}--to--{to_classification_name}.csv",
            )
        )
        for from_classification_name in analytics.classification_names()
    ]

    return (mapping_list[0], mapping_list[1])


def performance_data_path(performance_name: str) -> util.TypePerformanceDataSource:
    """
    This is a custom function for the Performance data source.  It has been designed for the
    test data.  Users are free to create their own function(s) to deliver the data.

    Args:
        performance_name (str): The performance name.

    Returns:
        TypePerformanceDataSource: The path name of the performance file corresponding to
        performance_name.
    """
    return resolve_file_path(_PERFORMANCE_DIRECTORIES, performance_name, ".csv")


def read_html_table(file_path: str) -> list[str]:
    """Read an html table file without the header."""
    lines: list[str] = []
    with open(file_path, "r", encoding=util.ENCODING) as file:
        on_table = False
        for line in file:
            if not on_table and line.startswith("<table "):
                on_table = True
            if on_table:
                lines.append(line)
    return lines


def resolve_file_path(directories: list[str], file_name: str, suffix: str = util.EMPTY) -> str:
    """
    Determines the file path where file_name is located.

    Args:
        directories (list[str]): A list of potential directories where file_name is located.
        file_name (str): The file name.
        suffix (str): The desired suffix.

    Returns:
        str: The resolved file path.
    """
    # Append ".csv".
    if (not util.is_empty(suffix)) and (not file_name.endswith(suffix)):
        file_name = f"{file_name}{suffix}"

    # Find the file_path
    for directory in directories:
        file_path = os.path.join(directory, file_name)
        if os.path.exists(file_path):
            return file_path

    # Throw exeption if file_path was not found
    assert False, util.file_path_error(file_name)
