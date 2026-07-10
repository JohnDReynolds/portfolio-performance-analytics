"""Expose the public PPAR package API."""

# Python Imports
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ppar.analytics import Analytics, Attribution, Frequency, RiskStatistics, View

try:
    __version__ = version("ppar")
except PackageNotFoundError:  # pragma: no cover - only expected outside package metadata
    __version__ = "0+unknown"

_ANALYTICS_EXPORTS = {
    "Analytics",
    "Attribution",
    "Frequency",
    "RiskStatistics",
    "View",
}

# Define the public API using __all__.
__all__ = [
    "Analytics",
    "Attribution",
    "Frequency",
    "RiskStatistics",
    "View",
    "__version__",
]


def __getattr__(name: str) -> Any:
    """Return lazy package-root exports.

    Args:
        name: Package-root attribute requested by an importer.

    Returns:
        The requested public Analytics symbol.

    Raises:
        AttributeError: If ``name`` is not a public lazy export.
    """
    if name in _ANALYTICS_EXPORTS:
        analytics = import_module("ppar.analytics")
        value = getattr(analytics, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
