"""Load Axys sources for use by the analytics facade.

This module provides the public ``AxysData`` facade for experimental Axys
source loading. See ``scripts/axys_demo.py`` for a working example.
"""

from __future__ import annotations

# Python imports
from collections.abc import Mapping
from dataclasses import replace
import datetime as dt
from pathlib import Path
from typing import Any

# Project imports
from ppar.axys.classification_sources import AxysClassificationSourceLoader
from ppar.axys.performance_sources import AxysPerformanceSourceLoader
from ppar.axys.portfolios import AxysPortfolio, AxysPortfolioLoader
from ppar.axys.specification import AxysSpecification
from ppar.axys.supporting_sources import (
    AxysClassificationSources,
    AxysSupportingSourceLoader,
)
from ppar.errors import PpaError
import ppar.utilities as util


class AxysData:  # pylint: disable=too-few-public-methods,too-many-instance-attributes
    """Configure Axys inputs and expose portfolio/classification loaders.

    ``AxysData`` remains the public construction facade. Specification parsing
    happens during initialization; portfolio reconciliation and supporting
    source loading happen on demand.

    Attributes:
        specifications_path: Path to the Axys YAML specification file.
        specifications: Parsed specification settings.
        portperf_path: Resolved portfolio-performance CSV path.
        secperf_path: Resolved security-performance CSV path.
        _specification: Parsed Axys specification object.
        _classification_loader: Loader used to normalize classification and
            mapping sources.
        _supporting_source_loader: Loader used to resolve classification
            sources on demand.
    """

    def __init__(
        self,
        specifications_path: util.PathLike,
        portperf_path: util.PathLike | None = None,
        secperf_path: util.PathLike | None = None,
        source_path_overrides: Mapping[str, util.PathLike] | None = None,
    ) -> None:
        """Initialize Axys source configuration.

        Args:
            specifications_path: YAML file describing Axys source paths,
                source-column mappings, classifications, and mappings.
            portperf_path: Optional portfolio-performance CSV path overriding
                the specification setting.
            secperf_path: Optional security-performance CSV path overriding the
                specification setting.
            source_path_overrides: Optional classification or mapping source
                file paths keyed by source name. These override configured
                ``default_file_path`` values.

        Raises:
            PpaError: If required performance paths are missing from both
                arguments and the Axys specification, or a source path override
                references an unknown source.
        """
        self.specifications_path = Path(specifications_path)

        self._specification = AxysSpecification(self.specifications_path, self._error_message)
        self.specifications: dict[str, Any] = self._specification.values
        self.portperf_path = self._specification.performance_path(
            portperf_path, "portperf_path", self._error_message
        )
        self.secperf_path = self._specification.performance_path(
            secperf_path, "secperf_path", self._error_message
        )
        self._classification_loader = AxysClassificationSourceLoader(
            self._specification,
            self._error_message,
            source_path_overrides,
        )

        self._supporting_source_loader = AxysSupportingSourceLoader(
            self._specification,
            self._classification_loader,
        )

    def get_portfolio(
        self,
        portfolio_code: str,
        from_date: dt.date | None = None,
        thru_date: dt.date | None = None,
        classification_name: str | None = None,
    ) -> AxysPortfolio:
        """Return one reconciled Axys portfolio for an optional date window.

        Args:
            portfolio_code: Portfolio code to load from Axys performance
                sources.
            from_date: Optional inclusive earliest beginning date to retain.
            thru_date: Optional inclusive latest ending date to retain.
            classification_name: Optional configured Axys classification to
                load with the returned portfolio.

        Returns:
            Reconciled portfolio output, optionally including classification
            sources for the requested classification.

        Raises:
            PpaError: If the requested portfolio has no rows, common periods
                cannot be found, or security returns cannot be reconciled to
                portfolio returns, or if the requested classification source is
                unknown or invalid.
        """
        portfolios = self._portfolio_loader(from_date, thru_date).load((portfolio_code,))
        if portfolio_code not in portfolios:
            raise PpaError(
                self._error_message(
                    f"No portperf rows for portfolio {portfolio_code!r}",
                    portfolio_code,
                    from_date,
                    thru_date,
                ),
                504,
            )
        portfolio = portfolios[portfolio_code]
        if classification_name is None:
            return portfolio
        return replace(
            portfolio,
            classification_sources=self.get_classification_sources(
                classification_name,
                portfolio,
            ),
        )

    def get_classification_sources(
        self,
        classification_name: str,
        portfolio: AxysPortfolio,
    ) -> AxysClassificationSources:
        """Return one Axys classification and its configured mapping source.

        Args:
            classification_name: Requested classification source name.
            portfolio: Reconciled portfolio whose security identifiers limit
                security-master sources.

        Returns:
            Classification source bundle ready for an attribution call.

        Raises:
            PpaError: If the classification source is unknown, invalid, or
                references an invalid mapping source.
        """
        return self._supporting_source_loader.load_classification_sources(
            classification_name,
            portfolio,
        )

    def _portfolio_loader(
        self,
        from_date: dt.date | None,
        thru_date: dt.date | None,
    ) -> AxysPortfolioLoader:
        """Return a portfolio loader for the requested date window.

        Args:
            from_date: Optional inclusive earliest beginning date to retain.
            thru_date: Optional inclusive latest ending date to retain.

        Returns:
            Portfolio loader using the configured performance paths and date
            filters.
        """
        def error_message(message: str, portfolio_code: str | None = None) -> str:
            """Return error context for this portfolio-loading request."""
            return self._error_message(message, portfolio_code, from_date, thru_date)

        performance_loader = AxysPerformanceSourceLoader(
            self._specification,
            error_message,
            from_date,
            thru_date,
        )
        return AxysPortfolioLoader(
            self._specification,
            performance_loader,
            error_message,
            self.portperf_path,
            self.secperf_path,
        )

    def _error_message(
        self,
        specific_message: str,
        portfolio_code: str | None = None,
        from_date: dt.date | None = None,
        thru_date: dt.date | None = None,
    ) -> str:
        """Return an Axys error detail including source and filter context.

        Args:
            specific_message: Error-specific text to prefix to context.
            portfolio_code: Portfolio involved in the error, when known.
            from_date: Optional inclusive earliest beginning date requested.
            thru_date: Optional inclusive latest ending date requested.

        Returns:
            Error detail text including paths, portfolio code, and date filters.
        """
        context = (
            "Context: "
            f"specifications_path={self.specifications_path}, "
            f"portperf_path={getattr(self, 'portperf_path', None)}, "
            f"secperf_path={getattr(self, 'secperf_path', None)}, "
            f"portfolio_code={portfolio_code}, "
            f"from_date={from_date}, "
            f"thru_date={thru_date}"
        )
        return f"{specific_message}  |  {context}" if specific_message else context
