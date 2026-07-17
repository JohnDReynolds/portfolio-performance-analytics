"""Cache return-reconstruction diagnostics used by Audit workbook views."""

from __future__ import annotations

# Python imports
from collections.abc import Iterable
import datetime as dt

# Third-party imports
import polars as pl

# Project imports
import ppar.utilities as util
from ppar.audit.performance_comparison import return_reconstruction

__all__ = [
    "WorkbookReconstructionCache",
    "resolved_reconstruction_cache",
]


class WorkbookReconstructionCache:
    """Cache reconstruction diagnostics for one workbook or report build.

    Args:
        comparison_path: Audit YAML path used to load reconstruction inputs.
    """

    def __init__(self, comparison_path: util.PathLike | None) -> None:
        self._comparison_path = comparison_path
        self._input_cache = return_reconstruction._SnapshotDataIndexCache()
        self._portfolio_checks: pl.DataFrame | None = None
        self._security_checks: pl.DataFrame | None = None
        self._security_checks_by_active_keys: dict[
            frozenset[tuple[str, str, dt.date, dt.date]],
            pl.DataFrame,
        ] = {}
        self._summary: pl.DataFrame | None = None

    def portfolio_checks(self) -> pl.DataFrame:
        """Return cached portfolio return-reconstruction checks."""
        if self._portfolio_checks is None:
            self._portfolio_checks = (
                return_reconstruction.portfolio_return_reconstruction_checks(
                    self._comparison_path,
                    _input_cache=self._input_cache,
                )
            )
        return self._portfolio_checks

    def security_checks(
        self,
        active_keys: Iterable[tuple[object, object, object, object]] | None = None,
    ) -> pl.DataFrame:
        """Return cached security return-reconstruction checks.

        Args:
            active_keys: Optional workbook primary keys that constrain security
                diagnostics to active review periods.

        Returns:
            Cached security reconstruction rows for the requested key set.
        """
        if active_keys is not None:
            reconstruction_keys = _security_reconstruction_active_keys(active_keys)
            cache_key = frozenset(reconstruction_keys)
            if cache_key not in self._security_checks_by_active_keys:
                self._security_checks_by_active_keys[cache_key] = (
                    return_reconstruction.security_return_reconstruction_checks(
                        self._comparison_path,
                        active_keys=reconstruction_keys,
                        _input_cache=self._input_cache,
                    )
                )
            return self._security_checks_by_active_keys[cache_key]
        if self._security_checks is None:
            self._security_checks = (
                return_reconstruction.security_return_reconstruction_checks(
                    self._comparison_path,
                    _input_cache=self._input_cache,
                )
            )
        return self._security_checks

    def summary(self) -> pl.DataFrame:
        """Return cached return-reconstruction summary."""
        if self._summary is None:
            self._summary = return_reconstruction.return_reconstruction_summary(
                self._comparison_path,
                _input_cache=self._input_cache,
            )
        return self._summary


def resolved_reconstruction_cache(
    comparison_path: util.PathLike | None,
    reconstruction_cache: WorkbookReconstructionCache | None,
) -> WorkbookReconstructionCache:
    """Return an existing reconstruction cache or create one for direct calls.

    Args:
        comparison_path: Audit YAML path used for a newly created cache.
        reconstruction_cache: Existing build-scoped cache, when available.

    Returns:
        The supplied cache or a new cache bound to ``comparison_path``.
    """
    if reconstruction_cache is not None:
        return reconstruction_cache
    return WorkbookReconstructionCache(comparison_path)


def _security_reconstruction_active_keys(
    active_keys: Iterable[tuple[object, object, object, object]],
) -> set[tuple[str, str, dt.date, dt.date]]:
    """Return reconstruction security keys from workbook primary keys."""
    reconstruction_keys: set[tuple[str, str, dt.date, dt.date]] = set()
    for portfolio_id, from_date, thru_date, security_id in active_keys:
        if not isinstance(from_date, dt.date) or not isinstance(thru_date, dt.date):
            continue
        reconstruction_keys.add(
            (
                str(portfolio_id),
                str(security_id),
                from_date,
                thru_date,
            )
        )
    return reconstruction_keys
