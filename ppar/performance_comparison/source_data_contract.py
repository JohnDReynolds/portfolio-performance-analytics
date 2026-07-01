"""Describe the minimum source-data contract for performance comparison."""

from __future__ import annotations

# Python imports
from dataclasses import dataclass
from typing import Final

# Project imports
from ppar.performance_comparison import schema as pc_cols
from ppar.performance_comparison.specification import (
    PORTFOLIO_COMPARISON_LEVEL,
    SECURITY_COMPARISON_LEVEL,
    PerformanceComparisonSpecification,
)

__all__ = [
    "SourceDataDatasetContract",
    "comparison_required_dataset_names",
    "source_data_contract",
    "source_data_contract_summary",
]


@dataclass(frozen=True)
class SourceDataDatasetContract:
    """Minimum normalized source-data expectations for one dataset.

    Attributes:
        name: Normalized dataset name used in comparison YAML.
        required_columns: Normalized columns required when the dataset is used.
        optional_columns: Normalized columns ppar can use when present.
        required_when: Human-readable rule for when the dataset must be
            configured and present in both snapshots.
    """

    name: str
    required_columns: tuple[str, ...]
    optional_columns: tuple[str, ...]
    required_when: str


_SOURCE_DATA_CONTRACT: Final[tuple[SourceDataDatasetContract, ...]] = (
    SourceDataDatasetContract(
        name=pc_cols.PORTFOLIO_PERFORMANCE,
        required_columns=pc_cols.PORTFOLIO_PERFORMANCE_REQUIRED_COLUMNS,
        optional_columns=pc_cols.PORTFOLIO_PERFORMANCE_OPTIONAL_COLUMNS,
        required_when="portfolio comparison is the primary review level",
    ),
    SourceDataDatasetContract(
        name=pc_cols.SECURITY_PERFORMANCE,
        required_columns=pc_cols.SECURITY_PERFORMANCE_REQUIRED_COLUMNS,
        optional_columns=pc_cols.SECURITY_PERFORMANCE_OPTIONAL_COLUMNS,
        required_when=(
            "security comparison is the primary review level or security return "
            "reconstruction is configured"
        ),
    ),
    SourceDataDatasetContract(
        name=pc_cols.HOLDINGS,
        required_columns=pc_cols.HOLDINGS_REQUIRED_COLUMNS,
        optional_columns=pc_cols.HOLDINGS_OPTIONAL_COLUMNS,
        required_when=(
            "return reconstruction is configured or holding fields are used as "
            "performance explanations"
        ),
    ),
    SourceDataDatasetContract(
        name=pc_cols.TRANSACTIONS,
        required_columns=pc_cols.TRANSACTIONS_REQUIRED_COLUMNS,
        optional_columns=pc_cols.TRANSACTIONS_OPTIONAL_COLUMNS,
        required_when=(
            "return reconstruction is configured or transaction fields are used "
            "as performance explanations"
        ),
    ),
    SourceDataDatasetContract(
        name=pc_cols.SECURITY_MASTER,
        required_columns=pc_cols.SECURITY_MASTER_REQUIRED_COLUMNS,
        optional_columns=pc_cols.SECURITY_MASTER_OPTIONAL_COLUMNS,
        required_when="security-reference context or security-master rules are used",
    ),
    SourceDataDatasetContract(
        name=pc_cols.CASH,
        required_columns=pc_cols.CASH_REQUIRED_COLUMNS,
        optional_columns=pc_cols.CASH_OPTIONAL_COLUMNS,
        required_when="cash rows are used as source-data evidence",
    ),
    SourceDataDatasetContract(
        name=pc_cols.FX_RATES,
        required_columns=pc_cols.FX_RATES_REQUIRED_COLUMNS,
        optional_columns=pc_cols.FX_RATES_OPTIONAL_COLUMNS,
        required_when="FX-rate rows are used as source-data evidence",
    ),
)


def source_data_contract() -> tuple[SourceDataDatasetContract, ...]:
    """Return the normalized source-data contract for comparison inputs."""
    return _SOURCE_DATA_CONTRACT


def comparison_required_dataset_names(
    specification: PerformanceComparisonSpecification,
) -> tuple[str, ...]:
    """Return dataset names that must exist for a resolved comparison spec.

    Args:
        specification: Parsed comparison specification.

    Returns:
        Dataset names that are mandatory for the selected comparison level and
        configured return-reconstruction sections.
    """
    required_names = {pc_cols.PORTFOLIO_PERFORMANCE}
    if specification.comparison_level == SECURITY_COMPARISON_LEVEL:
        required_names = {pc_cols.SECURITY_PERFORMANCE}
    if (
        specification.portfolio_return_reconstruction is not None
        or specification.security_return_reconstruction is not None
    ):
        required_names.update({pc_cols.HOLDINGS, pc_cols.TRANSACTIONS})
    if specification.security_return_reconstruction is not None:
        required_names.add(pc_cols.SECURITY_PERFORMANCE)
    return tuple(sorted(required_names))


def source_data_contract_summary(
    *,
    comparison_level: str = PORTFOLIO_COMPARISON_LEVEL,
    include_reconstruction_sources: bool = False,
) -> dict[str, str]:
    """Return compact source-data contract strings for CLI and tests.

    Args:
        comparison_level: Primary comparison level, ``"portfolio"`` or
            ``"security"``.
        include_reconstruction_sources: Whether to include the holdings and
            transaction datasets required by return reconstruction.

    Returns:
        A compact dictionary containing required dataset and required-column
        summaries.
    """
    required_names = {
        pc_cols.SECURITY_PERFORMANCE
        if comparison_level == SECURITY_COMPARISON_LEVEL
        else pc_cols.PORTFOLIO_PERFORMANCE
    }
    if include_reconstruction_sources:
        required_names.update({pc_cols.HOLDINGS, pc_cols.TRANSACTIONS})
    contracts_by_name = {contract.name: contract for contract in _SOURCE_DATA_CONTRACT}
    required_columns = [
        f"{name}: {', '.join(contracts_by_name[name].required_columns)}"
        for name in sorted(required_names)
    ]
    return {
        "required_datasets": ", ".join(sorted(required_names)),
        "required_columns": "; ".join(required_columns),
    }
