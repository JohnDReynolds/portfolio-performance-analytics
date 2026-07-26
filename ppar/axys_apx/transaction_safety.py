"""Fail-closed Axys/APX transaction boundaries.

This module intentionally does not assign economic meaning. Local transaction
meaning belongs in Audit ``transaction_rules``.
"""

from __future__ import annotations

from typing import Final

AMBIGUOUS_FLOW_TRANSACTION_CODES: Final[frozenset[str]] = frozenset(
    {"dp", "li", "lo", "ti", "wd"}
)
