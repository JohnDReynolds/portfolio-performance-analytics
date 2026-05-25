"""Load Axys sources for use by the analytics facade.

This module provides the public ``AxysData`` facade for experimental Axys
source loading. See ``scripts/axys_demo.py`` for a working example.
"""

from __future__ import annotations

# Python imports
import datetime as dt
from pathlib import Path
from typing import Any, Iterable

# Project imports
from ppar.axys.classification_sources import AxysClassificationSourceLoader
from ppar.axys.performance_sources import AxysPerformanceSourceLoader
from ppar.axys.portfolios import AxysPortfolio, AxysPortfolioLoader
from ppar.axys.specification import AxysSpecification
from ppar.axys.supporting_sources import AxysSupportingSourceLoader
import ppar.utilities as util


class AxysData:  # pylint: disable=too-few-public-methods,too-many-instance-attributes
    """Load Axys inputs, reconcile security returns, and expose analytics sources.

    ``AxysData`` remains the public construction facade. Specification and CSV
    validation are handled by :mod:`ppar.axys.performance_sources` and
    :mod:`ppar.axys.classification_sources`; numerical weight reconciliation is
    handled by :mod:`ppar.axys.reconciliation`.

    Attributes:
        from_date: Optional earliest beginning date retained from Axys data.
        thru_date: Optional latest ending date retained from Axys data.
        specifications_path: Path to the Axys YAML specification file.
        specifications: Parsed specification settings.
        portperf_path: Resolved portfolio-performance CSV path.
        secperf_path: Resolved security-performance CSV path.
        portfolios: Loaded reconciled portfolio output keyed by portfolio code.
        classification_data_sources: Normalized classification sources keyed by
            requested classification name.
        mapping_data_sources: Normalized mapping sources keyed by requested
            mapping name.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        specifications_path: util.PathLike,
        portperf_path: util.PathLike | None = None,
        secperf_path: util.PathLike | None = None,
        from_date: dt.date | None = None,
        thru_date: dt.date | None = None,
        portfolio_codes: Iterable[str] | str | None = None,
        classification_names: Iterable[str] | str | None = None,
        mapping_names: Iterable[str] | str | None = None,
    ) -> None:
        """Initialize reconciled Axys portfolio and supporting data sources.

        Args:
            specifications_path: YAML file describing Axys source paths,
                source-column mappings, classifications, and mappings.
            portperf_path: Optional portfolio-performance CSV path overriding
                the specification setting.
            secperf_path: Optional security-performance CSV path overriding the
                specification setting.
            from_date: Optional inclusive earliest beginning date to retain.
            thru_date: Optional inclusive latest ending date to retain.
            portfolio_codes: Optional portfolio code or iterable of codes to
                load. If omitted, all codes present in portperf are loaded.
            classification_names: Optional classification name or iterable of
                names to load. If omitted, configured classifications are used.
            mapping_names: Optional mapping name or iterable of names to load.
                If omitted, configured mappings are used.

        Raises:
            PpaError: If a configured source cannot be loaded or validated, if
                portperf and secperf cannot be reconciled, or if loaded
                classifications do not include a security master.
        """
        self.from_date = from_date
        self.specifications_path = Path(specifications_path)
        self.thru_date = thru_date
        self.portfolios: dict[str, AxysPortfolio] = {}

        specification = AxysSpecification(self.specifications_path, self._error_message)
        self.specifications: dict[str, Any] = specification.values
        self.portperf_path = specification.performance_path(
            portperf_path, "portperf_path", self._error_message
        )
        self.secperf_path = specification.performance_path(
            secperf_path, "secperf_path", self._error_message
        )
        performance_loader = AxysPerformanceSourceLoader(
            specification, self._error_message, from_date, thru_date
        )
        classification_loader = AxysClassificationSourceLoader(
            specification, self._error_message
        )

        portfolio_loader = AxysPortfolioLoader(
            specification,
            performance_loader,
            self._error_message,
            self.portperf_path,
            self.secperf_path,
        )

        self.portfolios = portfolio_loader.load(
            util.to_tuple_or_none(portfolio_codes),
        )
        supporting_source_loader = AxysSupportingSourceLoader(
            specification,
            classification_loader,
            self._error_message,
        )
        supporting_sources = supporting_source_loader.load(
            self.portfolios,
            util.to_tuple_or_none(classification_names),
            util.to_tuple_or_none(mapping_names),
        )
        self.classification_data_sources = supporting_sources.classifications
        self.mapping_data_sources = supporting_sources.mappings

    def _error_message(self, specific_message: str, portfolio_code: str | None = None) -> str:
        """Return an Axys error detail including source and filter context.

        Args:
            specific_message: Error-specific text to prefix to context.
            portfolio_code: Portfolio involved in the error, when known.

        Returns:
            Error detail text including paths, portfolio code, and date filters.
        """
        context = (
            "Context: "
            f"specifications_path={self.specifications_path}, "
            f"portperf_path={getattr(self, 'portperf_path', None)}, "
            f"secperf_path={getattr(self, 'secperf_path', None)}, "
            f"portfolio_code={portfolio_code}, "
            f"from_date={self.from_date}, "
            f"thru_date={self.thru_date}"
        )
        return f"{specific_message}  |  {context}" if specific_message else context
