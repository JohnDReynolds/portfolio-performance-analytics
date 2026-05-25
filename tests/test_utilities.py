"""
This module contains custom functions for the Classification, Mapping, and Performance data
sources.  It has been designed for the test data.  Users are free to create their own
function(s) to deliver the data.

The functions in this file deliver the path of csv files containing the data.  Users
could alternatively create their own custom data source functions that query databases and then
deliver pandas dataframes, polars dataframes, or python dictionaries.
"""

# Python imports
import datetime as dt
from pathlib import Path
import tempfile
from typing import Iterable
import unittest

# Third-party imports
import polars as pl

# Project imports
from ppar.analytics import Analytics
from ppar.attribution import Attribution
import ppar.columns as cols
import ppar.demo_data_sources as demo_data
import ppar.errors as errs
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


class TestUtilities(unittest.TestCase):
    """Verify utility calculations and file/data-source helpers."""

    def test_are_near(self) -> None:
        """Float nearness respects the selected tolerance."""
        self.assertTrue(util.are_near(1.0000000000001, 1.0, util.Tolerance.HIGH))
        self.assertFalse(util.are_near(1.0001, 1.0, util.Tolerance.LOW))

    def test_carino_linking_coefficient_rejects_undefined_returns(self) -> None:
        """Carino linking reports error 203 for returns at or below negative one."""
        with self.assertRaisesRegex(PpaError, errs.ERRORS[203]):
            util.carino_linking_coefficient(-1.0, 0.03)

        with self.assertRaisesRegex(PpaError, errs.ERRORS[203]):
            util.carino_linking_coefficient(0.05, -1.0)

    def test_carino_linking_coefficient_valid(self) -> None:
        """Valid Carino inputs return a floating-point coefficient."""
        self.assertIsInstance(util.carino_linking_coefficient(0.05, 0.03), float)

    def test_col_names(self) -> None:
        """Column suffix replacement generates output column names."""
        self.assertEqual(
            list(cols.col_names(["Port_ret", "Bench_ret"], "_wgt")),
            ["Port_wgt", "Bench_wgt"],
        )

    def test_date_str(self) -> None:
        """Date formatting uses the package's ISO-style format."""
        self.assertEqual(util.date_str(dt.date(2023, 1, 5)), "2023-01-05")

    def test_file_basename_without_extension(self) -> None:
        """File basenames are extracted from strings and Path instances."""
        path = "/some/path/to/myfile.csv"
        self.assertEqual(util.file_basename_without_extension(path), "myfile")
        self.assertEqual(util.file_basename_without_extension(Path(path)), "myfile")

    def test_file_path_exists(self) -> None:
        """File-existence detection handles existing and missing paths."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_name = temp_file.name
        try:
            self.assertTrue(util.file_path_exists(temp_name))
            self.assertTrue(util.file_path_exists(Path(temp_name)))
        finally:
            Path(temp_name).unlink()

        self.assertFalse(util.file_path_exists("not_a_real_file.xyz"))
        self.assertFalse(util.file_path_exists(Path("not_a_real_file.xyz")))

    def test_empty_file_path_error_is_error_804(self) -> None:
        """An empty requested file path reports error 804."""
        self.assertEqual(util.file_path_error(util.EMPTY), errs.ERRORS[804])

    def test_demo_data_sources_return_paths(self) -> None:
        """Packaged demo data helpers resolve existing Path instances."""
        performance_path = demo_data.performance_data_source("Large-Cap Benchmark.csv")
        classification_path = demo_data.classification_data_source("Security")

        self.assertIsInstance(performance_path, Path)
        self.assertIsInstance(classification_path, Path)
        self.assertTrue(util.file_path_exists(performance_path))
        self.assertTrue(util.file_path_exists(classification_path))

    def test_logarithmic_linking_coefficient_series(self) -> None:
        """Paired Polars Series produce a result for each observation."""
        result = util.logarithmic_linking_coefficient_series(
            pl.Series([0.02, 0.03, 0.05]),
            pl.Series([0.01, 0.02, 0.025]),
        )

        self.assertIsInstance(result, pl.Series)
        self.assertEqual(result.len(), 3)

    def test_logarithmic_linking_coefficients(self) -> None:
        """One total return produces one coefficient per period return."""
        result = util.logarithmic_linking_coefficients(
            0.08,
            pl.Series([0.01, 0.02, 0.03]),
        )

        self.assertIsInstance(result, pl.Series)
        self.assertEqual(result.len(), 3)

    def test_near_zero(self) -> None:
        """Near-zero detection respects the selected tolerance."""
        self.assertTrue(util.near_zero(0.0000000000001, util.Tolerance.HIGH))
        self.assertFalse(util.near_zero(0.001, util.Tolerance.LOW))


if __name__ == "__main__":
    unittest.main()
