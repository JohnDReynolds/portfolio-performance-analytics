"""Provide bundled data sources used by the demonstration application.

These helpers return paths to packaged CSV fixtures for performance,
classification, and mapping inputs. Applications may instead supply their own
supported paths, dictionaries, or DataFrames to the analytics APIs.
"""

# Python Imports
from importlib.resources import files
from pathlib import Path

# Project Imports
from ppar.analytics import Analytics
import ppar.utilities as util

# Directory containing the demo data.
_DEMO_DATA_DIRECTORY = files("ppar.demos.data")


def _demo_data_path(relative_path: str) -> Path:
    """Return a filesystem path for a packaged demo-data resource.

    Args:
        relative_path: Resource path relative to the demo-data package.

    Returns:
        Filesystem path representing the packaged resource.
    """
    return Path(str(_DEMO_DATA_DIRECTORY.joinpath(relative_path)))


def classification_data_source(
    classification_name: str | None = None,
) -> util.PathLike | None:
    """Return a bundled classification data source.

    Args:
        classification_name: Classification name whose packaged CSV should be
            loaded.

    Returns:
        Classification file path, or ``None`` when no classification was requested.
    """
    classification_name = util.normalize_optional_string(classification_name)

    if classification_name is None:
        return None

    # Return the path to the csv file containing the classification data.
    return _demo_data_path(f"generic_analytics/classifications/{classification_name}.csv")


def mapping_data_sources(
    analytics: Analytics, to_classification_name: str | None = None
) -> tuple[util.MappingDataSource | None, util.MappingDataSource | None]:
    """Return bundled portfolio and benchmark mapping data sources.

    Args:
        analytics: Analytics instance whose input classification names determine
            the required mappings.
        to_classification_name: Destination classification name.

    Returns:
        Two-item tuple of portfolio and benchmark mapping sources. A source is
        ``None`` when its performance already uses the destination
        classification or no destination was requested.
    """
    to_classification_name = util.normalize_optional_string(to_classification_name)

    if to_classification_name is None:
        return (None, None)

    # Build the tuple of mapping data sources containing the csv file paths.
    mapping_list: list[util.MappingDataSource | None] = [
        (
            None
            if from_classification_name == to_classification_name
            else _demo_data_path(
                f"generic_analytics/mappings/{from_classification_name}"
                f"--to--{to_classification_name}.csv"
            )
        )
        for from_classification_name in analytics.classification_names()
    ]

    # Return the tuple of mapping data sources containing the csv file paths.
    return (mapping_list[0], mapping_list[1])


def performance_data_source(performance_name: str) -> Path:
    """Return the bundled CSV path for a named performance source.

    Args:
        performance_name: File name of the packaged performance source.

    Returns:
        Path to the requested packaged performance file.
    """
    # Return the path of the performance file corresponding to performance_name.
    return _demo_data_path(f"generic_analytics/performance/{performance_name}")
