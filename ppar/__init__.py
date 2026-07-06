"""Expose the public portfolio performance analytics API."""

# Python Imports
from importlib.metadata import PackageNotFoundError, version

# Explicitly import the specific members or modules.
# If they are defined below in __all__, then they must be imported here.
from ppar.analytics import Analytics, Attribution, Frequency, RiskStatistics, View

try:
    __version__ = version("ppar")
except PackageNotFoundError:  # pragma: no cover - only expected outside package metadata
    __version__ = "0+unknown"

# Define the public API using __all__
__all__ = [
    "Analytics",
    "Attribution",
    "Frequency",
    "RiskStatistics",
    "View",
    "__version__",
]
