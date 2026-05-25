"""Load Axys classification and mapping sources for reconciled portfolios."""

from __future__ import annotations

# Python imports
from dataclasses import dataclass

# Third-party imports
import polars as pl

# Project imports
from ppar.axys.classification_sources import AxysClassificationSourceLoader
from ppar.axys.portfolios import AxysPortfolio
from ppar.axys.specification import AxysSpecification, ErrorMessage
import ppar.columns as cols
from ppar.errors import PpaError


@dataclass(frozen=True)
class AxysSupportingSources:
    """Contain normalized supporting sources for Axys analytics.

    Attributes:
        classifications: Normalized classification sources keyed by requested
            classification name.
        mappings: Normalized mapping sources keyed by requested mapping name.
    """

    classifications: dict[str, pl.DataFrame]
    mappings: dict[str, pl.DataFrame]


class AxysSupportingSourceLoader:
    """Load classification and mapping sources for reconciled portfolios.

    Attributes:
        _specification: Parsed Axys source configuration.
        _loader: Source loader used to normalize configured source files.
        _error_message: Callback used to add facade-level validation context.
    """

    def __init__(
        self,
        specification: AxysSpecification,
        loader: AxysClassificationSourceLoader,
        error_message: ErrorMessage,
    ) -> None:
        """Initialize a supporting-source loader.

        Args:
            specification: Parsed Axys configuration used to determine default
                source names.
            loader: Source loader used to normalize configured source files.
            error_message: Callback used to add facade-level source context.
        """
        self._specification = specification
        self._loader = loader
        self._error_message = error_message

    def load(
        self,
        portfolios: dict[str, AxysPortfolio],
        classification_names: tuple[str, ...] | None,
        mapping_names: tuple[str, ...] | None,
    ) -> AxysSupportingSources:
        """Return requested classification and mapping sources.

        Args:
            portfolios: Reconciled portfolios whose security identifiers limit
                security-master classifications.
            classification_names: Requested classification names, or ``None``
                to load configured defaults.
            mapping_names: Requested mapping names, or ``None`` to load
                configured defaults.

        Returns:
            Normalized classification and mapping sources.

        Raises:
            PpaError: If a requested classification or mapping source is
                unknown, cannot be validated, or does not include a security
                master among the loaded classifications.
        """
        if not portfolios:
            return AxysSupportingSources({}, {})

        unique_security_ids = self._unique_security_ids(portfolios)
        classification_names = classification_names or self._specification.default_source_names(
            "classification"
        )
        mapping_names = mapping_names or self._specification.default_source_names("mapping")
        classifications = {
            name: self._loader.load("classification", name, unique_security_ids)
            for name in classification_names
        }
        mappings = {
            name: self._loader.load("mapping", name, unique_security_ids)
            for name in mapping_names
        }
        self._require_security_master(classifications)
        return AxysSupportingSources(classifications, mappings)

    def _require_security_master(self, classifications: dict[str, pl.DataFrame]) -> None:
        """Validate that requested classifications include a security master.

        Args:
            classifications: Loaded classification sources keyed by configured
                classification name.

        Raises:
            PpaError: If none of the loaded classifications is designated as
                the security master.
        """
        if any(
            self._specification.is_security_master(classification_name)
            for classification_name in classifications
        ):
            return
        raise PpaError(
            self._error_message("Must have a classification with is_security_master == true"),
            504,
        )

    @staticmethod
    def _unique_security_ids(portfolios: dict[str, AxysPortfolio]) -> list[str]:
        """Return distinct security identifiers across reconciled portfolios.

        Args:
            portfolios: Reconciled portfolios to inspect.

        Returns:
            Unique security identifiers present in the portfolio outputs.
        """
        return (
            pl.concat([portfolio.secperf[cols.IDENTIFIER] for portfolio in portfolios.values()])
            .unique()
            .to_list()
        )
