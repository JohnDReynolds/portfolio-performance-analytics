"""Tests for Audit workbook reconstruction source allocation."""

from __future__ import annotations

# Python imports
from pathlib import Path
import unittest
from unittest import mock

# Project imports
from ppar.audit import compare_snapshots
from ppar.audit import workbook_reconstruction
from ppar.audit import workbook_source_allocation
from ppar.audit import workbook_tables

_PORTFOLIO_COMPARISON_PATH = Path(
    "ppar/setup_templates/axys_apx_audit/axys_apx_audit.yaml"
)


class TestWorkbookSourceAllocation(unittest.TestCase):
    """Verify source-allocation ownership and reuse contracts."""

    def test_formula_allocation_searches_candidates_once_per_input(self) -> None:
        """Cause allocation and visibility share one source-candidate search."""
        for comparison_level in ("portfolio", "security"):
            with self.subTest(comparison_level=comparison_level):
                findings = compare_snapshots(
                    _PORTFOLIO_COMPARISON_PATH,
                    comparison_level=comparison_level,
                )
                active_findings = workbook_tables._active_findings(findings)
                table_cache = workbook_tables._WorkbookTableCache(active_findings)
                reconstruction_cache = workbook_reconstruction.WorkbookReconstructionCache(
                    _PORTFOLIO_COMPARISON_PATH
                )
                formula_rows = table_cache.reconstruction_formula_rows(
                    comparison_level,
                    comparison_path=_PORTFOLIO_COMPARISON_PATH,
                    reconstruction_cache=reconstruction_cache,
                )
                candidate_search = workbook_source_allocation._formula_source_candidates

                with mock.patch.object(
                    workbook_source_allocation,
                    "_formula_source_candidates",
                    wraps=candidate_search,
                ) as candidate_search_spy:
                    workbook_tables.audit_review_workbook_sheets(
                        findings,
                        comparison_path=_PORTFOLIO_COMPARISON_PATH,
                        comparison_level=comparison_level,
                        _reconstruction_cache=reconstruction_cache,
                        _table_cache=table_cache,
                    )

                self.assertEqual(candidate_search_spy.call_count, len(formula_rows))


if __name__ == "__main__":
    unittest.main()
