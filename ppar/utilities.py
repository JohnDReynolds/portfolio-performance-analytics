"""
This module contains utility functions and system-wide constants.
"""

# Python Imports
import datetime as dt
from enum import Enum
import math
import os
import tempfile
import time
import webbrowser

# Third-Party Imports
import numpy as np
import pandas as pd
import polars as pl

# Project Imports
import ppar.errors as errs

# Types for type-checking.
TypeAllDataSources = str | dict[str, str] | pd.DataFrame | pl.DataFrame
TypeClassificationDataSource = TypeAllDataSources
TypeMappingDataSource = TypeAllDataSources
TypePerformanceDataSource = str | pd.DataFrame | pl.DataFrame

# Miscellaneous Common Constants
DATE_FORMAT_STRING = "%Y-%m-%d"  # yyyy-mm-dd
DEFAULT_ANNUAL_MINIMUM_ACCEPTABLE_RETURN = 0.0
DEFAULT_ANNUAL_RISK_FREE_RATE = 0.03  # 3%
DEFAULT_CONFIDENCE_LEVEL = 0.95  # 95%
DEFAULT_CURRENCY_SYMBOL = "$"
DEFAULT_PORTFOLIO_VALUE = 100_000  # $100,000
ENCODING = "utf-8"
_UNDEFINED_RETURN = -1.0
EMPTY = "_empty_"


class Tolerance(Enum):
    """
    An enumeration of tolerances.

    Args:
        Enum (_type_): enum
    """

    LOW = 0.00000005
    MEDIUM = 0.0000000005
    HIGH = 0.0000000000005


def are_near(f1: float, f2: float, tolerance: Tolerance = Tolerance.HIGH) -> bool:
    """
    Determine if f1 is close to f2.

    Args:
        f1 (float): The first number.
        f2 (float): The second number.
        tolerance (float): The "nearness" tolerance. Defaults to TOLERANCE_HIGH.

    Returns:
        bool: True if f1 is within tolerance of f2, otherwise False.
    """
    return abs(f1 - f2) < tolerance.value


def carino_linking_coefficient(portfolio_return: float, benchmark_return: float) -> float:
    """
    Calculates the Carino linking coefficient used for linking over multiple subperiods.

    Args:
        portfolio_return (float): The portfolio return.
        benchmark_return (float): The benchmark return.

    Returns:
        float: The Carino linking coefficient.
    """
    # Check for invalid returns.  The Log of a number <= 0 is undefined.
    assert (
        _UNDEFINED_RETURN < portfolio_return
    ), f"{errs.ERROR_203_UNDEFINED_RETURN}The portfolio has a return of {portfolio_return:.6f}"
    assert (
        _UNDEFINED_RETURN < benchmark_return
    ), f"{errs.ERROR_203_UNDEFINED_RETURN}The benchmark has a return of {benchmark_return:.6f}"

    # Get the difference between the portfolio_return and the benchmark_return
    return_difference = portfolio_return - benchmark_return

    # If the portfolio and benchmark returns are almost identical, then the standard formula below
    # will give non-sensical results with a tiny-tiny denominator.  So return an alternate formula.
    if near_zero(return_difference):
        return 1.0 / (1.0 + portfolio_return)

    # Return the carino k-factor.
    return (
        math.log(1.0 + portfolio_return) - math.log(1.0 + benchmark_return)
    ) / return_difference


def convert_to_date(date: str | dt.date | dt.datetime) -> dt.date:
    """
    Converts a date string in the format yyyy-mm-dd to a python date.

    Args:
        date_string (str | dt.date | dt.datetime): The date string in yyyy-mm-dd format or a native
        python date or datetime.

    Returns:
        date: A python date.

    Raises:
        ValueError: If the input string is not in the correct format.
    """
    # Return the date if it is already in the proper format.
    if isinstance(date, dt.datetime):
        date = date.date()
    if isinstance(date, dt.date):
        return date

    # Try parsing the string date.
    try:
        return dt.datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError as e:
        raise ValueError(
            f"{errs.ERROR_803_CANNOT_CONVERT_TO_A_DATE}'{date}', must be in the format yyyy-mm-dd"
        ) from e


def date_str(date: dt.date) -> str:
    """
    Returns a string representation of the date.

    Args:
        date (dt.date): The date.

    Returns:
        str: A string representation of the date.
    """
    return date.strftime(DATE_FORMAT_STRING)


def file_basename_without_extension(file_path: str) -> str:
    """
    Gets the simple file name without the directory or extension.

    Args:
        file_path (str): The full file path.

    Returns:
        str: The simple file name without the directory or extension.
    """
    return os.path.basename(file_path).split(".")[0]


def file_path_error(file_path: str) -> str:
    """Return the appropriate error message."""
    if is_empty(file_path):
        return errs.ERROR_804_MISSING_DATA_SOURCE
    return f"{errs.ERROR_802_FILE_PATH_DOES_NOT_EXIST}{file_path}"


def file_path_exists(file_path: str) -> bool:
    """Determine if the file_path exists."""
    if is_empty(file_path):
        return False
    return os.path.exists(file_path) and os.path.isfile(file_path)


def is_empty(data_source: TypeAllDataSources) -> bool:
    """Determine if the data_source is unknown."""
    return isinstance(data_source, str) and (data_source == EMPTY or (not data_source.strip()))


def load_datasource(
    data_source: TypeAllDataSources,
    column_names: list[str],
    needed_items: list[str],
    error_message: str,
) -> pl.DataFrame:
    """
    Load a 2-column data_source into a Polars DataFrame.

    Args:
        data_source (TypeAllDataSources): The data source.
        column_names list[str]: The 2 column names.
        needed_items: list[str]: The needed items corresponding to column_names[0].
        error_message (str): The error message in the case where there are not 2 columns.

    Returns:
        pl.DataFrame: The data_source as a Polars DataFrame.
    """
    # Get the 2-column dataframe.
    if isinstance(data_source, str):
        # Assert that the data file path exists.
        assert file_path_exists(data_source), file_path_error(data_source)
        # Load the data_source in lazy-mode.  infer_schema=False will force both columns to be the
        # default strings (Utf8).  Then filter on needed_items.
        lf = pl.scan_csv(data_source, has_header=False, infer_schema=False)
        column0_name = list(lf.collect_schema().keys())[0]
        df = lf.filter(pl.col(column0_name).is_in(needed_items)).collect()
    elif isinstance(data_source, dict):
        df = pl.DataFrame(
            {
                column_names[0]: data_source.keys(),
                column_names[1]: data_source.values(),
            }
        )
    elif isinstance(data_source, pd.DataFrame):
        df = pl.from_pandas(data_source)
    else:  # isinstance(data_source, pl.DataFrame):
        df = data_source

    # Assert that you have 2 columns.
    assert 2 == len(df.columns), error_message

    # Give the columns consistent names.
    df.columns = column_names

    # Remove duplicates.
    df = df.unique(subset=[df.columns[0]], keep="last")

    # Cast to strings and filter on needed_items.  Note that this was done above in pl.scan_scv
    # if isinstance(data_source, str).
    if not isinstance(data_source, str):
        # All identifiers need to be strings for classifications, mappings, performances, etc.
        for column_name in df.columns:
            if not isinstance(df.schema[column_name], pl.String):
                df = df.with_columns(df[column_name].cast(pl.String))
        # Filter on only the needed_items.
        df = df.filter(pl.col(df.columns[0]).is_in(needed_items))

    # Return the dataframe.
    return df


def logarithmic_linking_coefficients(overall_return: float, returns: pl.Series) -> pl.Series:
    """
    Calculates the linking coefficients for each subperiod when linking over the entire period.

    Args:
        overall_return (float): The return for the overall entire period.
        returns (pl.Series): The return for each subperiod.

    Returns:
        pl.Series: The linking coefficients for each subperiod.
    """
    # A return < -1.0 is undefined.  And the log of a negative number is undefined.  So assert that
    # the return is greater than -1.0.  Note that this logic exactly mimics the logic in
    # logarithmic_smoothing_coefficients(), only it is done for a single value.
    assert _UNDEFINED_RETURN < overall_return, f"{errs.ERROR_203_UNDEFINED_RETURN}{overall_return}"
    denominator = np.log(1.0 + overall_return) / overall_return if overall_return != 0.0 else 1.0

    # Return the logarithmic_linking_coefficients
    return logarithmic_smoothing_coefficients(returns) / denominator


def logarithmic_linking_coefficient_series(
    overall_returns: pl.Series, returns: pl.Series
) -> pl.Series:
    """
    Calculates the linking coefficients for each subperiod when linking over the overall entire
    period.

    Args:
        overall_returns (pl.Series): The return for the overall entire period.
        returns (pl.Series): The return for each subperiod.

    Returns:
        pl.Series: The linking coefficients for each subperiod.
    """
    return logarithmic_smoothing_coefficients(returns) / logarithmic_smoothing_coefficients(
        overall_returns
    )


def logarithmic_smoothing_coefficients(returns: pl.Series) -> pl.Series:
    """
    Calculates the logarithmic smoothing coefficients for each subperiod.

    Args:
        returns (pl.Series): The return for each subperiod.

    Returns:
        pl.Series: The logarithmic smoothing coefficients for each subperiod.
    """
    # A return < -1.0 is undefined.  And the log of a negative number is undefined.  So assert
    # that the returns are greater than -1.0.
    assert (returns > -1.0).all(), errs.ERROR_203_UNDEFINED_RETURN

    ## Method 1: This method works great, but is a little slower than Method 2 below.
    # If the return is 0.0, then dividing by 0.0 will give nan.
    # So a return of 0.0 will correctly yield a coeficient of 1.0.
    # return (returns.log1p() / returns).fill_nan(1)  # pl.log1p() is the same as log(1 + value)

    ## Method 2: This method is slightly faster than Method 1.  And takes advantage of lazy.
    return (
        pl.LazyFrame(returns)
        .with_columns(
            pl.when(pl.col(returns.name) == 0.0)
            .then(1.0)
            .otherwise(pl.col(returns.name).log1p() / pl.col(returns.name))
            .alias(returns.name)
        )
        .collect()
    )[returns.name]


def near_zero(f: float, tolerance: Tolerance = Tolerance.HIGH) -> bool:
    """
    Determines if f is close to 0.0 within the specified tolerance.

    Args:
        f (float): The float to test.
        tolerance (float, optional): The tolerance.  Defaults to TOLERANCE_HIGH.

    Returns:
        bool: True if f is within tolerance, otherwise False.
    """
    return are_near(f, 0, tolerance)


def open_in_browser(html_or_png: str | bytes) -> None:
    """
    Open the html string or png binary in a web browser.

    Args:
        html (str): The html string or png binary.
    """
    # Determine if the file is html or png
    suffix = ".html" if isinstance(html_or_png, str) else ".png"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        # Write html_or_png to a temp_file.
        if isinstance(html_or_png, str):  # suffix == ".html":
            with open(temp_file.name, "w", encoding=ENCODING) as f:
                f.write(html_or_png)
        else:
            with open(temp_file.name, "wb") as f:
                f.write(html_or_png)

        # Some web browsers need the local file name prefixed.  It depends on which web browser
        # (e.g. Safari or Chrome) and the settings and security restrictions.
        url = f"file://{os.path.abspath(temp_file.name)}"

        # Open the file in a browser. Sometimes it takes a while for the file to be fully written
        # and accesible to the browser, so give it 2 seconds before failing.  If you are
        # rapid-firing multiple files to this function, the image browser can get overwhelmed,
        # especially on old win10 machines.  So sleep 0.7 seconds after opening the file.  Note
        # that the os will delete the temp file.
        qty_trys = 10
        for i in range(qty_trys):
            try:
                webbrowser.open(url)
                time.sleep(0.7)
                break
            except Exception as e:  # pylint: disable=broad-exception-caught
                if i == qty_trys - 1:
                    print(f"Could not open the file {url}.  {e}")
                    raise  # Re-raise the exception
                time.sleep(0.2)
