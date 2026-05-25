"""Serialize calculated result DataFrames to public output formats.

This module contains small shared adapters used by calculation objects that
expose Polars results as pandas, JSON, XML, or CSV output.
"""

# Python Imports
from pathlib import Path
from typing import cast, Literal

# Third-Party Imports
import pandas as pd
import polars as pl

# Project Imports
import ppar.utilities as util


def to_json(
    df: pl.DataFrame,
    float_precision: int,
    date_format: Literal["iso"] | None = None,
) -> str:
    """Return a Polars DataFrame as a JSON string.

    Args:
        df: Source DataFrame to serialize.
        float_precision: Number of decimal places to include for floating-point
            values.
        date_format: Optional pandas date format.

    Returns:
        JSON string containing the serialized DataFrame.
    """
    pandas_df = to_pandas(df)
    if date_format is None:
        return cast(  # pyright: ignore[reportUnnecessaryCast]
            str,
            pandas_df.to_json(double_precision=float_precision),
        )
    return cast(  # pyright: ignore[reportUnnecessaryCast]
        str,
        pandas_df.to_json(double_precision=float_precision, date_format=date_format),
    )


def to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    """Return a Polars DataFrame as a pandas DataFrame.

    Args:
        df: Source DataFrame to convert.

    Returns:
        pandas DataFrame containing the same data.
    """
    return df.to_pandas()


def to_xml(df: pl.DataFrame) -> str:
    """Return a Polars DataFrame as an XML string.

    Args:
        df: Source DataFrame to serialize.

    Returns:
        XML string containing the serialized DataFrame.
    """
    return to_pandas(df).to_xml()


def write_csv(
    df: pl.DataFrame,
    file_path: util.PathLike,
    float_precision: int,
) -> None:
    """Write a Polars DataFrame to a CSV file.

    Args:
        df: Source DataFrame to serialize.
        file_path: Path of the CSV file to write.
        float_precision: Number of decimal places to write for floating-point
            values.
    """
    df.write_csv(Path(file_path), float_precision=float_precision)
