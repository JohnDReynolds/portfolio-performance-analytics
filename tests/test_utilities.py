"""
This module contains custom functions for the Classification, Mapping, and Performance data
sources.  It has been designed for the test data.  Users are free to create their own
function(s) to deliver the data.

The functions in this file deliver the path of csv files containing the data.  Users
could alternatively create their own custom data source functions that query databases and then
deliver pandas dataframes, polars dataframes, or python dictionaries.
"""

# Python imports
from pathlib import Path
from typing import Iterable

# Project imports
from ppar.analytics import Analytics
from ppar.attribution import Attribution
from ppar.errors import PpaError
import ppar.utilities as util

# Directories containing the test data.
_DATA_DIRECTORIES = (Path("tests/data"), Path("../tests/data"), Path("data"))
_AXYS_DIRECTORIES = [directory / "axys_perf" for directory in _DATA_DIRECTORIES]
_CLASSIFICATION_DIRECTORIES = [directory / "classifications" for directory in _DATA_DIRECTORIES]
_MAPPING_DIRECTORIES = [directory / "mappings" for directory in _DATA_DIRECTORIES]
_PERFORMANCE_DIRECTORIES = [directory / "performance" for directory in _DATA_DIRECTORIES]


def axys_data_path(file_name: str, suffix: str = ".csv") -> Path:
    """
    This is a custom function for resolving the axys file_path (portperf or secperf or axysdata).

    Args:
        file_name (str): The portperf or secperf file name.

    Returns:
        Path: The path of the axys file corresponding to file_name.
    """
    return resolve_file_path(_AXYS_DIRECTORIES, file_name, suffix)


def classification_data_path(classification_name: str) -> util.PathLike:
    """
    This is a custom function for the Classification data source.  It has been designed for the
    test data.  Users are free to create their own function(s) to deliver the data.

    Args:
        classification_name (str): The classification name.

    Returns:
        Path | str: The classification file path, or util.EMPTY when no classification was
        requested.
    """
    if util.is_empty(classification_name):
        return classification_name
    return resolve_file_path(_CLASSIFICATION_DIRECTORIES, classification_name, ".csv")


def get_attribution(
    analytics: Analytics,
    classification_name: str | None = None,
    classification_data_source: util.ClassificationDataSource | None = None,
    mapping_data_source: util.MappingDataSource | None = None,
) -> Attribution:
    """Infer file path from the classification_name and then return the attribution.

    Args:
        analytics (Analytics): The Analytics instance.
        classification_name (str): The classification name to use when resolving a
            classification data source.
        classification_data_source (util.ClassificationDataSource): Optional
            classification data source to use instead of resolving from the
            classification name.
        mapping_data_source (util.MappingDataSource): Optional mapping data source
            to use for both portfolio and benchmark if supplied.

    Returns:
        Attribution: The resulting attribution object.
    """
    classification_name = util.normalize_optional_string(classification_name)

    if classification_data_source is None or util.is_empty_string(classification_data_source):
        classification_data_source = classification_data_path(classification_name)

    if mapping_data_source is None or util.is_empty_string(mapping_data_source):
        mapping_data_sources = mapping_data_paths(analytics, classification_name)
    else:
        mapping_data_sources = (mapping_data_source, mapping_data_source)

    return analytics.get_attribution(
        classification_name,
        classification_data_source,
        mapping_data_sources,
    )


def html_table_lines(html_string: str) -> list[str]:
    """Get just the table lines from the html string.

    Args:
        html_string (str): The HTML string to scan for the first table.

    Returns:
        list[str]: The lines from the first HTML table found in the input.
    """
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
) -> tuple[util.MappingDataSource, util.MappingDataSource]:
    """
    This is a custom function for the Mapping data sources.  It has been designed for the
    test data.  Users are free to create their own function(s) to deliver the data.

    Args:
        analytics (Analytics): The Analytics instance.
        to_classification_name (str): The classification name to map to.

    Returns:
        tuple[util.MappingDataSource, util.MappingDataSource]: A tuple of 2 mapping
        data sources (0 = Portfolio Data Source, 1 = Benchmark Data Source)
    """
    if util.is_empty(to_classification_name):
        return (util.EMPTY, util.EMPTY)

    # Build the tuple of mapping data sources containing the csv file paths.
    mapping_list: list[util.MappingDataSource] = [
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


def performance_data_path(performance_name: str) -> Path:
    """
    This is a custom function for the Performance data source.  It has been designed for the
    test data.  Users are free to create their own function(s) to deliver the data.

    Args:
        performance_name (str): The performance name.

    Returns:
        Path: The path of the performance file corresponding to performance_name.
    """
    return resolve_file_path(_PERFORMANCE_DIRECTORIES, performance_name, ".csv")


def read_html_table(file_path: util.PathLike) -> list[str]:
    """Read an HTML table file without the header.

    Args:
        file_path: Path to the HTML file containing the table.

    Returns:
        list[str]: The table lines from the file.
    """
    lines: list[str] = []
    with open(Path(file_path), "r", encoding=util.ENCODING) as file:
        on_table = False
        for line in file:
            if not on_table and line.startswith("<table "):
                on_table = True
            if on_table:
                lines.append(line)
    return lines


def resolve_file_path(
    directories: Iterable[util.PathLike], file_name: str, suffix: str = util.EMPTY
) -> Path:
    """
    Determines the file path where file_name is located.

    Args:
        directories: Potential directories where file_name may be located.
        file_name (str): The file name.
        suffix (str): The desired suffix.

    Returns:
        Path: The resolved file path.
    """
    # Append ".csv".
    if (not util.is_empty(suffix)) and (not file_name.endswith(suffix)):
        file_name = f"{file_name}{suffix}"

    # Find the file_path.
    for directory in directories:
        file_path = Path(directory) / file_name
        if file_path.exists():
            return file_path

    # Throw exception if file_path was not found.
    raise PpaError(util.file_path_error(file_name), None)
