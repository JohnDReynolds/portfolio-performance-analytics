"""Define exact-default source columns for normalized Audit datasets.

Vendor and site-specific source headings belong in explicit schema YAML. When
no schema mapping is configured for a normalized field, Audit accepts only the
exact normalized field name, including case.
"""

from __future__ import annotations

# Project imports
from ppar.audit import schema as pc_cols
from ppar.audit.source_loader import ColumnAliases


def _exact_aliases(columns: tuple[str, ...]) -> ColumnAliases:
    """Return one exact source-name candidate per normalized column."""
    return {column: (column,) for column in columns}


PORTFOLIO_PERFORMANCE_REQUIRED_ALIASES = _exact_aliases(
    pc_cols.PORTFOLIO_PERFORMANCE_REQUIRED_COLUMNS
)
PORTFOLIO_PERFORMANCE_OPTIONAL_ALIASES = _exact_aliases(
    pc_cols.PORTFOLIO_PERFORMANCE_OPTIONAL_COLUMNS
)

SECURITY_PERFORMANCE_REQUIRED_ALIASES = _exact_aliases(
    pc_cols.SECURITY_PERFORMANCE_REQUIRED_COLUMNS
)
SECURITY_PERFORMANCE_OPTIONAL_ALIASES = _exact_aliases(
    pc_cols.SECURITY_PERFORMANCE_OPTIONAL_COLUMNS
)

SPLITS_REQUIRED_ALIASES = _exact_aliases(pc_cols.SPLITS_REQUIRED_COLUMNS)
SPLITS_OPTIONAL_ALIASES = _exact_aliases(pc_cols.SPLITS_OPTIONAL_COLUMNS)

TRANSACTIONS_REQUIRED_ALIASES = _exact_aliases(pc_cols.TRANSACTIONS_REQUIRED_COLUMNS)
TRANSACTIONS_OPTIONAL_ALIASES = _exact_aliases(pc_cols.TRANSACTIONS_OPTIONAL_COLUMNS)

HOLDINGS_REQUIRED_ALIASES = _exact_aliases(pc_cols.HOLDINGS_REQUIRED_COLUMNS)
HOLDINGS_OPTIONAL_ALIASES = _exact_aliases(pc_cols.HOLDINGS_OPTIONAL_COLUMNS)

SECURITY_MASTER_REQUIRED_ALIASES = _exact_aliases(
    pc_cols.SECURITY_MASTER_REQUIRED_COLUMNS
)
SECURITY_MASTER_OPTIONAL_ALIASES = _exact_aliases(
    pc_cols.SECURITY_MASTER_OPTIONAL_COLUMNS
)
