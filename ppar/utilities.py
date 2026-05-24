"""Utility functions, type aliases, and constants used across the package."""

# Python Imports
import datetime as dt
from enum import Enum
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Sequence, TypeAlias
import webbrowser

# Third-Party Imports
import numpy as np
import pandas as pd
import polars as pl

# Project Imports
import ppar.errors as errs
from ppar.errors import PpaError

# Types for type-checking.
PathLike: TypeAlias = str | Path
AllDataSources: TypeAlias = PathLike | dict[str, str] | pd.DataFrame | pl.DataFrame
ClassificationDataSource: TypeAlias = AllDataSources
MappingDataSource: TypeAlias = AllDataSources
PerformanceDataSource: TypeAlias = PathLike | pd.DataFrame | pl.DataFrame

# Miscellaneous Common Constants
DATE_FORMAT_STRING = "%Y-%m-%d"  # yyyy-mm-dd
DEFAULT_ANNUAL_MINIMUM_ACCEPTABLE_RETURN = 0.0
DEFAULT_ANNUAL_RISK_FREE_RATE = 0.03  # 3%
DEFAULT_CONFIDENCE_LEVEL = 0.95  # 95%
DEFAULT_CURRENCY_SYMBOL = "$"
DEFAULT_PORTFOLIO_VALUE = 100_000  # $100,000
ENCODING = "utf-8"
_UNDEFINED_RETURN = -1.0
EMPTY = "_empty_"  # Legacy sentinel. Prefer None in public APIs.


class Tolerance(Enum):
    """Floating-point comparison tolerances.

    Attributes:
        LOW: The loosest comparison tolerance.
        MEDIUM: A moderate comparison tolerance.
        HIGH: The strictest comparison tolerance.
    """

    LOW = 0.00000005
    MEDIUM = 0.0000000005
    HIGH = 0.0000000000005


def are_near(f1: float, f2: float, tolerance: Tolerance = Tolerance.HIGH) -> bool:
    """Return whether two floats are within the specified tolerance.

    Args:
        f1: The first value to compare.
        f2: The second value to compare.
        tolerance: The comparison tolerance to apply.

    Returns:
        True if the absolute difference between ``f1`` and ``f2`` is less than
        ``tolerance``; otherwise, False.
    """
    return abs(f1 - f2) < tolerance.value


def carino_linking_coefficient(portfolio_return: float, benchmark_return: float) -> float:
    """Calculate the Carino linking coefficient for two returns.

    Args:
        portfolio_return: The portfolio return expressed as a decimal.
        benchmark_return: The benchmark return expressed as a decimal.

    Returns:
        The Carino linking coefficient.

    Raises:
        PpaError: If either return is less than or equal to -100%, because the
            logarithmic calculation would be undefined.
    """
    # Check for invalid returns.  The Log of a number <= 0 is undefined.
    if portfolio_return <= _UNDEFINED_RETURN:
        raise PpaError(f"The portfolio has a return of {portfolio_return:.6f}", 203)
    if benchmark_return <= _UNDEFINED_RETURN:
        raise PpaError(f"The benchmark has a return of {benchmark_return:.6f}", 203)

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
    """Convert a supported date value to a ``datetime.date``.

    Args:
        date: A ``datetime.date``, ``datetime.datetime``, or string in
            ``yyyy-mm-dd`` format.

    Returns:
        The value converted to a ``datetime.date``.

    Raises:
        PpaError: If a string value cannot be parsed as ``yyyy-mm-dd``.
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
        raise PpaError(f"{date!r} must be in the format yyyy-mm-dd", 803) from e


def date_str(date: dt.date) -> str:
    """Format a date using the package date format.

    Args:
        date: The date to format.

    Returns:
        The date formatted as ``yyyy-mm-dd``.
    """
    return date.strftime(DATE_FORMAT_STRING)


def file_basename_without_extension(file_path: PathLike) -> str:
    """Return a file name without its directory or extension.

    Args:
        file_path: The file path to evaluate.

    Returns:
        The base file name before the first period in the file name.
    """
    return Path(file_path).name.split(".")[0]


def file_path_error(file_path: PathLike) -> str:
    """Return the appropriate file path error message.

    Args:
        file_path: The file path that failed validation.

    Returns:
        The empty-path error message if ``file_path`` is empty; otherwise, the
        missing-file error message with ``file_path`` appended.
    """
    return errs.ERRORS[804] if is_empty(file_path) else f"{errs.ERRORS[802]}{file_path}"


def file_path_exists(file_path: PathLike) -> bool:
    """Return whether a non-empty file path exists and points to a file.

    Args:
        file_path: The file path to test.

    Returns:
        True if ``file_path`` is non-empty, exists, and is a file; otherwise,
        False.
    """
    if is_empty(file_path):
        return False
    return Path(file_path).is_file()


def has_directory(path_str: PathLike) -> bool:
    """Return whether a path includes an explicit directory component.

    Args:
        path_str: The path string to test.

    Returns:
        True if ``path_str`` has a parent directory other than the current
        directory; otherwise, False.
    """
    return Path(path_str).parent != Path(".")


def is_empty_string(thing: Any) -> bool:
    """Return whether a value is an empty string or legacy empty marker.

    Args:
        thing: The value to test.

    Returns:
        True if ``thing`` is a string equal to ``EMPTY`` or contains only
        whitespace; otherwise, False.
    """
    return isinstance(thing, str) and (thing == EMPTY or (not thing.strip()))


def is_empty(thing: Any) -> bool:
    """Return whether a value is an empty string or legacy empty marker.

    This compatibility alias preserves the historical helper name. Prefer
    ``is_empty_string()`` in new code when the value being tested is string-like.

    Args:
        thing: The value to test.

    Returns:
        True if ``thing`` is a string equal to ``EMPTY`` or contains only
        whitespace; otherwise, False.
    """
    return is_empty_string(thing)


def is_missing(thing: Any) -> bool:
    """Return whether a public optional argument was not supplied.

    Args:
        thing: The value to test.

    Returns:
        True for ``None`` or an empty string marker; otherwise, False.
    """
    return thing is None or is_empty_string(thing)


def normalize_optional_string(value: str | None) -> str:
    """Normalize optional public string arguments to the legacy empty marker.

    Args:
        value: Optional string value supplied by the caller.

    Returns:
        ``EMPTY`` for ``None`` or blank/legacy-empty strings; otherwise, ``value``.
    """
    if value is None or is_empty_string(value):
        return EMPTY
    return value


def load_datasource(
    data_source: AllDataSources,
    column_names: Sequence[str],
    needed_items: Sequence[str],
    error_message: str,
) -> pl.DataFrame:
    """Load a two-column data source into a normalized Polars DataFrame.

    Args:
        data_source: The source data. Supported inputs are a CSV file path, a
            dictionary, a pandas DataFrame, or a Polars DataFrame.
        column_names: The two output column names to assign to the DataFrame.
        needed_items: The allowed values for the first output column. Rows with
            other first-column values are filtered out.
        error_message: The error message to use if the loaded source does not
            contain exactly two columns.

    Returns:
        A two-column Polars DataFrame with normalized column names, duplicate
        first-column values removed, values cast to strings for non-file inputs,
        and rows filtered to ``needed_items``.

    Raises:
        PpaError: If ``data_source`` is a file path that does not point to an
            existing file, or if the loaded source does not contain exactly two
            columns.
    """
    # Get the 2-column dataframe.
    if isinstance(data_source, str | Path):
        data_source = Path(data_source)
        # Assert that the data file path exists.
        if not file_path_exists(data_source):
            raise PpaError(file_path_error(data_source), None)
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
    if len(df.columns) != 2:
        raise PpaError(error_message, None)

    # Give the columns consistent names.
    df.columns = column_names

    # Remove duplicates.
    df = df.unique(subset=[df.columns[0]], keep="last")

    # Cast to strings and filter on needed_items.  Note that this was done above in pl.scan_scv
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
    """Calculate logarithmic linking coefficients for subperiod returns.

    Args:
        overall_return: The total return for the full period, expressed as a
            decimal.
        returns: The subperiod returns, expressed as decimals.

    Returns:
        A Polars Series containing the linking coefficient for each subperiod
        return.

    Raises:
        PpaError: If ``overall_return`` is less than or equal to -100%, or if
            any value in ``returns`` is less than or equal to -100%.
    """
    # A return < -1.0 is undefined.  And the log of a negative number is undefined.  So valiadte
    # that the return is greater than -1.0.  Note that this logic exactly mimics the logic in
    # logarithmic_smoothing_coefficients(), only it is done for a single value.
    if overall_return <= _UNDEFINED_RETURN:
        raise PpaError(f"{overall_return}", 203)
    denominator = np.log(1.0 + overall_return) / overall_return if overall_return != 0.0 else 1.0

    # Return the logarithmic_linking_coefficients
    return logarithmic_smoothing_coefficients(returns) / denominator


def logarithmic_linking_coefficient_series(
    overall_returns: pl.Series, returns: pl.Series
) -> pl.Series:
    """Calculate linking coefficients from series-level overall returns.

    Args:
        overall_returns: The full-period returns to use as denominators,
            expressed as decimals.
        returns: The subperiod returns to link, expressed as decimals.

    Returns:
        A Polars Series containing the linking coefficient for each return.

    Raises:
        PpaError: If any value in ``overall_returns`` or ``returns`` is less
            than or equal to -100%.
    """
    return logarithmic_smoothing_coefficients(returns) / logarithmic_smoothing_coefficients(
        overall_returns
    )


def logarithmic_smoothing_coefficients(returns: pl.Series) -> pl.Series:
    """Calculate logarithmic smoothing coefficients for returns.

    Args:
        returns: The returns to smooth, expressed as decimals.

    Returns:
        A Polars Series containing the logarithmic smoothing coefficient for
        each return.

    Raises:
        PpaError: If any return is less than or equal to -100%.
    """
    # A return < -1.0 is undefined.  And the log of a negative number is undefined.  So validate
    # that the returns are greater than -1.0.
    if not (returns > _UNDEFINED_RETURN).all():
        raise PpaError("", 203)

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
    """Return whether a float is near zero within the specified tolerance.

    Args:
        f: The value to compare with zero.
        tolerance: The comparison tolerance to apply.

    Returns:
        True if ``f`` is within ``tolerance`` of zero; otherwise, False.
    """
    return are_near(f, 0, tolerance)


def open_in_browser(html_or_png: str | bytes) -> None:
    """Write HTML or PNG content to a temp file and open it in a browser.

    Args:
        html_or_png: HTML content as a string, or PNG content as bytes.

    Raises:
        Exception: Re-raises the final exception from ``webbrowser.open`` after
            all retry attempts fail.
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


def to_tuple_or_none(value: Iterable[str] | str | None) -> tuple[str, ...] | None:
    """Normalize a string iterable, single string, or None to a tuple or None.

    Args:
        value: The value to normalize.

    Returns:
        None if ``value`` is None or an empty string; a one-item tuple if
        ``value`` is a non-empty string; otherwise, ``value`` converted to a
        tuple.
    """
    if value is None:
        return None
    if isinstance(value, str):
        if not value:
            return None
        return (value,)
    return tuple(value)
