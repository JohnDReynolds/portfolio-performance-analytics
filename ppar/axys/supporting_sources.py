"""Load Axys classification sources for reconciled portfolios."""

from __future__ import annotations

# Python imports
from dataclasses import dataclass

# Third-party imports
import polars as pl

# Project imports
from ppar.axys.classification_sources import AxysClassificationSourceLoader
from ppar.axys.portfolios import AxysPortfolio
from ppar.axys.specification import AxysSpecification
import ppar.columns as cols


@dataclass(frozen=True)
class AxysClassificationSources:
    """Contain one classification and its optional mapping source.

    Attributes:
        classification_name: Requested Axys classification name.
        classification_data_source: Normalized classification source.
        mapping_data_sources: Pair of identical mapping sources for analytics
            attribution calls, or ``None`` when the requested classification is
            already at security grain.
    """

    classification_name: str
    classification_data_source: pl.DataFrame
    mapping_data_sources: tuple[pl.DataFrame, pl.DataFrame] | None


class AxysSupportingSourceLoader:
    """Load one classification and its optional mapping source.

    Attributes:
        _specification: Parsed Axys source configuration.
        _loader: Source loader used to normalize configured source files.
    """

    def __init__(
        self,
        specification: AxysSpecification,
        loader: AxysClassificationSourceLoader,
    ) -> None:
        """Initialize a supporting-source loader.

        Args:
            specification: Parsed Axys configuration used to determine default
                source names.
            loader: Source loader used to normalize configured source files.
        """
        self._specification = specification
        self._loader = loader

    def load_classification_sources(
        self,
        classification_name: str,
        portfolio: AxysPortfolio,
    ) -> AxysClassificationSources:
        """Return one classification and its configured mapping source.

        Args:
            classification_name: Requested classification source name.
            portfolio: Reconciled portfolio whose security identifiers limit
                security-master sources.

        Returns:
            Classification source bundle ready for attribution calls.

        Raises:
            PpaError: If the classification source is unknown, invalid, or
                references an invalid mapping source.
        """
        unique_security_ids = portfolio.secperf[cols.IDENTIFIER].unique().to_list()
        classification = self._loader.load(
            "classification", classification_name, unique_security_ids
        )
        if self._specification.is_security_master(classification_name):
            mapping_data_sources = None
        else:
            mapping_name = self._specification.values["classifications"][classification_name][
                "mapping"
            ]
            mapping = self._loader.load("mapping", mapping_name, unique_security_ids)
            mapping_data_sources = (mapping, mapping)
        return AxysClassificationSources(
            classification_name,
            classification,
            mapping_data_sources,
        )
