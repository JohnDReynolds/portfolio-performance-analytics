"""Expose the Audit-focused public PPAR package API."""

# Python Imports
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ppar.audit import (
        AuditSpecification,
        compare_snapshots,
        write_audit_report_bundle,
    )

try:
    __version__ = version("ppar")
except PackageNotFoundError:  # pragma: no cover - only expected outside package metadata
    __version__ = "0+unknown"

_AUDIT_EXPORTS = {
    "AuditSpecification",
    "compare_snapshots",
    "write_audit_report_bundle",
}

__all__ = [
    "AuditSpecification",
    "__version__",
    "compare_snapshots",
    "write_audit_report_bundle",
]


def __getattr__(name: str) -> Any:
    """Return lazy package-root exports.

    Args:
        name: Package-root attribute requested by an importer.

    Returns:
        The requested public Audit symbol.

    Raises:
        AttributeError: If ``name`` is not a public lazy export.
    """
    if name in _AUDIT_EXPORTS:
        audit = import_module("ppar.audit")
        value = getattr(audit, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
