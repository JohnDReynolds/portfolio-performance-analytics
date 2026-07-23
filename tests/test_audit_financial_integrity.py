"""Tests for Phase 3 financial-input integrity safety nets."""

from __future__ import annotations

# Python imports
import datetime as dt
from pathlib import Path
import tempfile
import unittest

# Third-party imports
import polars as pl
import yaml

# Test imports
from tests import test_utilities as test_util

# Project imports
from ppar.errors import PpaError
from ppar.audit import schema as pc_cols
from ppar.audit import conservation
from ppar.audit import financial_integrity
from ppar.audit.performance_comparison import findings as pc_findings
from ppar.audit.portfolio_performance import PortfolioPerformanceLoader
from ppar.audit.period_linking import validate_portfolio_periods
from ppar.audit.runner import compare_snapshots
from ppar.audit.specification import AuditSpecification
from ppar.audit.data_issues import checks as data_issues


class TestAuditFinancialIntegrity(unittest.TestCase):
    """Enforce SN-04, SN-06, and SN-07 at their intended boundaries."""

    def test_changed_evidence_multiset_preserves_unmatched_duplicate_rows(self) -> None:
        """Changed-row detection conserves duplicate multiplicity and null values."""
        frame = pl.DataFrame(
            {
                "key": ["A", "A", "B", "C"],
                "value": [1.0, 1.0, None, 3.0],
            }
        )
        counterpart = pl.DataFrame(
            {
                "key": ["A", "B", "D"],
                "value": [1.0, None, 4.0],
            }
        )

        changed = financial_integrity._rows_not_identical_in_other_snapshot(
            frame,
            counterpart,
        )

        assert changed is not None
        self.assertEqual(
            changed.to_dicts(),
            [{"key": "A", "value": 1.0}, {"key": "C", "value": 3.0}],
        )

    def test_retired_performance_values_are_ignored(self) -> None:
        """Performance-file market values do not create findings or inputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            comparison_path = _write_site(
                root,
                portfolio_rows=(
                    "P1,2026-01-01,2026-01-31,1000,1100,0.10,USD",
                    "P1,2026-02-01,2026-02-28,1090,1200,0.10,USD",
                ),
                extra_config={"data_issues": {"enabled": False}},
            )

            specification = AuditSpecification(comparison_path)
            performance = PortfolioPerformanceLoader(specification).load("a")
            issues = data_issues.data_issues_table(comparison_path)

        self.assertEqual(
            performance.columns,
            [
                *pc_cols.PORTFOLIO_PERFORMANCE_REQUIRED_COLUMNS,
                pc_cols.BASE_CURRENCY,
            ],
        )
        self.assertTrue(issues.is_empty())

    def test_overlapping_performance_periods_fail_source_contract(self) -> None:
        """SN-07 rejects periods that could multiply assign dated evidence."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            comparison_path = _write_site(
                root,
                portfolio_rows=(
                    "P1,2026-01-01,2026-01-31,1000,1100,0.10,USD",
                    "P1,2026-01-15,2026-02-15,1100,1200,0.09,USD",
                ),
            )
            specification = AuditSpecification(comparison_path)

            with self.assertRaisesRegex(PpaError, "SN-07.*overlapping periods"):
                PortfolioPerformanceLoader(specification).load("a")

    def test_security_period_validation_keeps_duplicates_and_scopes_independent(
        self,
    ) -> None:
        """SN-07 leaves duplicate-key handling to Error 112 and scopes securities."""
        periods = pl.DataFrame(
            {
                pc_cols.PORTFOLIO_ID: ["P1", "P1", "P1", "P1"],
                pc_cols.SECURITY_ID: ["S1", "S1", "S2", "S1"],
                pc_cols.FROM_DATE: [
                    dt.date(2026, 1, 1),
                    dt.date(2026, 1, 1),
                    dt.date(2026, 1, 15),
                    dt.date(2026, 2, 1),
                ],
                pc_cols.THRU_DATE: [
                    dt.date(2026, 1, 31),
                    dt.date(2026, 1, 31),
                    dt.date(2026, 2, 15),
                    dt.date(2026, 2, 28),
                ],
            }
        )

        validate_portfolio_periods(
            periods,
            dataset_name=pc_cols.SECURITY_PERFORMANCE,
            path="secperf.csv",
            specification_path="comparison.yaml",
        )

    def test_foreign_countable_value_requires_explicit_base_value(self) -> None:
        """SN-06 rejects a foreign holding value with no base-currency counterpart."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            comparison_path = _write_site(
                root,
                portfolio_rows=(
                    "P1,2026-01-01,2026-01-31,1000,1100,0.10,USD",
                ),
                holdings_rows=(
                    "P1,SAP.DE,2026-01-31,EUR,USD,10,100,1000,",
                ),
            )

            with self.assertRaisesRegex(PpaError, "SN-06.*base_market_value"):
                compare_snapshots(comparison_path)

    def test_invalid_currency_code_fails_source_contract(self) -> None:
        """SN-06 rejects ambiguous currency identifiers before comparison."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            comparison_path = _write_site(
                root,
                portfolio_rows=(
                    "P1,2026-01-01,2026-01-31,1000,1100,0.10,US_DOLLAR",
                ),
            )

            with self.assertRaisesRegex(PpaError, "SN-06.*three-letter currency"):
                compare_snapshots(comparison_path)

    def test_portfolio_fx_quote_must_target_base_currency(self) -> None:
        """SN-06 rejects a portfolio FX pair quoted into the wrong unit."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            comparison_path = _write_site(
                root,
                portfolio_rows=(
                    "P1,2026-01-01,2026-01-31,1000,1100,0.10,USD",
                ),
                fx_rows=("P1,EUR,GBP,2026-01-31,0.85",),
            )

            with self.assertRaisesRegex(PpaError, "SN-06.*to_currency=GBP"):
                compare_snapshots(comparison_path)

    def test_out_of_period_transaction_cannot_own_explained_performance(self) -> None:
        """SN-07 allows visible history but prohibits counting it in a later period."""
        original = pl.DataFrame(
            [
                {
                    pc_findings.FINDING_CODE: "PC-TXN-AMT",
                    pc_findings.DATASET: pc_cols.TRANSACTIONS,
                    pc_findings.SOURCE_COLUMN: pc_cols.AMOUNT,
                    pc_findings.PORTFOLIO_ID: "P1",
                    pc_findings.SECURITY_ID: "AAPL",
                    pc_findings.FROM_DATE: dt.date(2026, 2, 1),
                    pc_findings.THRU_DATE: dt.date(2026, 2, 28),
                    "as_of_date": dt.date(2026, 1, 15),
                    "estimated_impact": 0.01,
                }
            ]
        )
        causes = conservation.cause_conservation_table(
            original,
            comparison_level="portfolio",
        )

        with self.assertRaisesRegex(PpaError, "SN-07.*outside performance period"):
            conservation.assert_cause_conservation(
                original,
                causes,
                comparison_level="portfolio",
            )


def _write_site(
    root: Path,
    *,
    portfolio_rows: tuple[str, ...],
    holdings_rows: tuple[str, ...] = (),
    fx_rows: tuple[str, ...] = (),
    extra_config: dict[str, object] | None = None,
) -> Path:
    """Write a two-snapshot Phase 3 test site and return its YAML path."""
    for snapshot_name in ("snapshot_a", "snapshot_b"):
        snapshot_path = root / snapshot_name
        snapshot_path.mkdir()
        _write_csv(
            snapshot_path / "portperf.csv",
            (
                "PORTFOLIO_CODE,FROM_DATE,THRU_DATE,BEGIN_MV,END_MV,"
                "PORT_RETURN,BASE_CURRENCY"
            ),
            portfolio_rows,
        )
        if holdings_rows:
            _write_csv(
                snapshot_path / "holdings.csv",
                (
                    "PORT,SEC,HOLDING_DATE,CURRENCY,BASE_CURRENCY,QTY,PRICE,"
                    "MKT_VAL,BASE_MKT_VAL"
                ),
                holdings_rows,
            )
        if fx_rows:
            _write_csv(
                snapshot_path / "fx_rates.csv",
                "PORT,FROM_CURRENCY,TO_CURRENCY,RATE_DATE,FX_RATE",
                fx_rows,
            )

    files = {"portfolio_performance": "portperf.csv"}
    if holdings_rows:
        files["holdings"] = "holdings.csv"
    if fx_rows:
        files["fx_rates"] = "fx_rates.csv"
    configuration: dict[str, object] = {
        "comparison": {"level": "portfolio"},
        "snapshots": {
            "a": {"path": "snapshot_a"},
            "b": {"path": "snapshot_b"},
        },
        "files": files,
        "extract_contract": {
            "enforce_ambiguous_axys_flows": True,
            "transaction_semantics_case": "legacy_case_insensitive",
        },
        "tolerances": {
            "return": 0.000001,
            "contribution": 0.000001,
            "weight": 0.000001,
            "market_value": 0.01,
            "quantity": 0.000001,
            "price": 0.000001,
            "split_factor": 0.00000001,
            "fx_rate": 0.00000001,
        },
    }
    if holdings_rows:
        configuration["holding_impact_methods"] = {
            "market_value": {
                "method": "market_value_delta_over_return_denominator",
                "denominator_source": "begin_market_value",
            },
            "accrued": {
                "method": "accrued_delta_over_return_denominator",
                "denominator_source": "begin_market_value",
            },
            "quantity": {
                "method": (
                    "quantity_delta_times_snapshot_a_unit_market_value_over_"
                    "return_denominator"
                ),
                "denominator_source": "begin_market_value",
            },
            "cost": {"method": "evidence_only"},
        }
        configuration["price_impact_methods"] = {
            "price": {
                "method": "price_delta_over_snapshot_a_price_times_weight",
                "weight_source": "snapshot_a_weight",
            }
        }
    if fx_rows:
        configuration["fx_rate_impact_methods"] = {
            "fx_rate": {"method": "evidence_only"},
        }
    if extra_config:
        configuration.update(extra_config)
    comparison_path = root / "ppar.yaml"
    test_util.write_audit_test_yaml(comparison_path, configuration)
    return comparison_path


def _write_csv(path: Path, header: str, rows: tuple[str, ...]) -> None:
    """Write a small CSV fixture."""
    path.write_text("\n".join((header, *rows)) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
