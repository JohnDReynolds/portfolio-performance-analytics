"""Tests for portfolio return-reconstruction diagnostics."""

from __future__ import annotations

# Python imports
import datetime as dt
from pathlib import Path
import tempfile
import unittest
from unittest import mock

# Third-party imports
import polars as pl
import yaml

# Test imports
from tests import test_utilities as test_util

# Project imports
from ppar.errors import PpaError
from ppar.audit.performance_comparison import return_reconstruction as _reconstruction
from ppar.audit.performance_comparison.return_reconstruction import (
    BEGIN_VALUE_B,
    DERIVED_DENOMINATOR_B,
    DERIVED_RETURN_DIFFERENCE,
    DERIVED_NUMERATOR_B,
    END_VALUE_B,
    END_VALUE_DIFFERENCE,
    INCOME_B,
    NET_FLOW_DIFFERENCE,
    NET_FLOW_B,
    RECONSTRUCTION_CATEGORY,
    RECONSTRUCTION_STATUS,
    RECONSTRUCTION_STATUS_ALIGNED,
    RECONSTRUCTION_STATUS_DIFFERENT,
    RECONSTRUCTION_STATUS_MISSING_INPUTS,
    REPORTED_RETURN_DIFFERENCE,
    WEIGHTED_FLOW_B,
    WEIGHTED_FLOW_DIFFERENCE,
    portfolio_return_reconstruction_checks,
    return_reconstruction_summary,
    security_return_reconstruction_checks,
)
from ppar.audit.specification import AuditSpecification

_PORTFOLIO_COMPARISON_PATH = Path(
    "ppar/setup_templates/axys_apx_audit/axys_apx_audit.yaml"
)
_BASELINE_COMPARISON_PATH = Path(
    "tests/data/axys/validation/ppar_audit.yaml"
)
_DEMO_AXYS_APX_DIRECTORY = Path("ppar/setup_templates/axys_apx_audit")
_INTENTIONAL_PORTFOLIO_DIFFERENT_KEYS = {
    ("BALANCED", "2026-05-09", "2026-05-14"),
    ("INCOME", "2026-04-01", "2026-04-30"),
}
_INTENTIONAL_SECURITY_DIFFERENT_KEYS = {
    ("BALANCED", "csusJPM", "2026-05-09", "2026-05-14"),
    ("BALANCED", "csusMSFT", "2026-05-09", "2026-05-14"),
    ("INCOME", "fius91282Y5Y1", "2026-04-01", "2026-04-30"),
}


def _required_yaml_settings() -> dict[str, object]:
    """Return explicit settings that formerly had internal defaults."""
    return {
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
        "transaction_impact_methods": {
            "external_flow": {"method": "evidence_only"},
            "performance": {
                "method": "transaction_amount_delta_over_return_denominator",
                "denominator_source": "begin_market_value",
            },
            "quantity": {"method": "evidence_only"},
            "price": {"method": "evidence_only"},
            "commission": {"method": "evidence_only"},
        },
        "holding_impact_methods": {
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
        },
        "price_impact_methods": {
            "price": {
                "method": "price_delta_over_snapshot_a_price_times_weight",
                "weight_source": "snapshot_a_weight",
            }
        },
    }


def _write_reinvestment_pair_fixture(directory: Path) -> Path:
    """Write a tiny comparison fixture with a dividend reinvestment pair."""
    for snapshot_name in ("snapshot_a", "snapshot_b"):
        (directory / snapshot_name).mkdir()

    holdings_a = pl.DataFrame(
        {
            "PORT": ["P1", "P1", "P1", "P1"],
            "SEC": ["AAPL", "CASHUSD", "AAPL", "CASHUSD"],
            "HOLDING_DATE": [
                "2025-12-31",
                "2025-12-31",
                "2026-01-31",
                "2026-01-31",
            ],
            "QTY": [100.0, 1000.0, 100.0, 1000.0],
            "PRICE": [100.0, 1.0, 100.0, 1.0],
            "MKT_VAL": [10000.0, 1000.0, 10000.0, 1000.0],
            "COST": [10000.0, 1000.0, 10000.0, 1000.0],
            "ACCRUED": [0.0, 0.0, 0.0, 0.0],
        }
    )
    holdings_b = holdings_a.with_columns(
        pl.when((pl.col("SEC") == "AAPL") & (pl.col("HOLDING_DATE") == "2026-01-31"))
        .then(10100.0)
        .otherwise(pl.col("MKT_VAL"))
        .alias("MKT_VAL")
    )
    holdings_a.write_csv(directory / "snapshot_a" / "holdings.csv")
    holdings_b.write_csv(directory / "snapshot_b" / "holdings.csv")

    portperf_rows = {
        "END_MV": [11000.0],
        "FLOW": [0.0],
        "INCOME": [100.0],
        "GAIN_LOSS": [0.0],
        "PORTFOLIO_CODE": ["P1"],
        "PORTFOLIO_NAME": ["Portfolio 1"],
        "FROM_DATE": ["2026-01-01"],
        "THRU_DATE": ["2026-01-31"],
        "BEGIN_MV": [11000.0],
        "PORT_RETURN": [0.0],
    }
    pl.DataFrame(portperf_rows).write_csv(directory / "snapshot_a" / "portperf.csv")
    pl.DataFrame({**portperf_rows, "END_MV": [11100.0]}).write_csv(
        directory / "snapshot_b" / "portperf.csv"
    )

    secperf_rows = {
        "END_MV": [10000.0],
        "INCOME": [0.0],
        "GAIN_LOSS": [0.0],
        "PORTFOLIO_CODE": ["P1"],
        "SECURITY_ID": ["AAPL"],
        "FROM_DATE": ["2026-01-01"],
        "THRU_DATE": ["2026-01-31"],
        "BEGIN_WEIGHT": [10000.0 / 11000.0],
        "BEGIN_MV": [10000.0],
        "SEC_RETURN": [0.0],
        "CONTRIBUTION": [0.0],
    }
    pl.DataFrame(secperf_rows).write_csv(directory / "snapshot_a" / "secperf.csv")
    pl.DataFrame({**secperf_rows, "END_MV": [10100.0], "INCOME": [100.0]}).write_csv(
        directory / "snapshot_b" / "secperf.csv"
    )

    pl.DataFrame(
        {
            "PORT": ["P1", "P1"],
            "SEC": ["AAPL", "AAPL"],
            "TRANSACTION_DATE": ["2026-01-15", "2026-01-15"],
            "TRAN": ["dv", "by"],
            "AMOUNT": [0.0, 0.0],
        }
    ).write_csv(directory / "snapshot_a" / "transactions.csv")
    pl.DataFrame(
        {
            "PORT": ["P1", "P1"],
            "SEC": ["AAPL", "AAPL"],
            "TRANSACTION_DATE": ["2026-01-15", "2026-01-15"],
            "TRAN": ["dv", "by"],
            "AMOUNT": [100.0, -100.0],
        }
    ).write_csv(directory / "snapshot_b" / "transactions.csv")

    configuration = {
        **_required_yaml_settings(),
        "comparison": {
            "name": "Reinvestment pair fixture",
            "level": "portfolio",
        },
        "snapshots": {
            "a": {"label": "a", "path": "snapshot_a"},
            "b": {"label": "b", "path": "snapshot_b"},
        },
        "files": {
            "portfolio_performance": "portperf.csv",
            "security_performance": "secperf.csv",
            "holdings": "holdings.csv",
            "transactions": "transactions.csv",
        },
        "portfolio_return_reconstruction": {
            "method": "modified_dietz",
            "beginning_value_source": "holdings",
            "ending_value_source": "holdings",
            "flow_source": "transactions",
            "flow_timing": "transaction_date",
            "day_count": "actual_days",
            "inclusion_rule": "beginning_of_day",
            "flow_categories": ["external_flow"],
            "income_categories": ["income"],
            "return_basis": "net",
            "sign_convention": "signed_amount",
        },
        "security_return_reconstruction": {
            "method": "modified_dietz",
            "beginning_value_source": "holdings",
            "ending_value_source": "holdings",
            "flow_source": "transactions",
            "flow_timing": "transaction_date",
            "day_count": "actual_days",
            "inclusion_rule": "beginning_of_day",
            "flow_categories": ["buy", "sell"],
            "income_categories": ["income"],
            "return_basis": "net",
            "sign_convention": "signed_amount",
        },
        "transaction_rules": {
            "by": {
                "transaction_category": "buy",
                "cash_flow_sign": "negative",
                "performance_flow_sign": "performance",
            },
            "dv": {
                "transaction_category": "income",
                "cash_flow_sign": "positive",
                "performance_flow_sign": "performance",
            },
        },
    }
    path = directory / "comparison.yaml"
    test_util.write_audit_test_yaml(path, configuration)
    return path


def _write_accrued_value_fixture(
    directory: Path,
    *,
    include_accrued_column: bool,
) -> Path:
    """Write a tiny comparison fixture for holdings valuation contracts."""
    for snapshot_name in ("snapshot_a", "snapshot_b"):
        (directory / snapshot_name).mkdir()

    holdings_rows = {
        "PORT": ["P1", "P1", "P1", "P1"],
        "SEC": ["BOND1", "CASHUSD", "BOND1", "CASHUSD"],
        "HOLDING_DATE": [
            "2025-12-31",
            "2025-12-31",
            "2026-01-31",
            "2026-01-31",
        ],
        "QTY": [10.0, 100.0, 10.0, 100.0],
        "PRICE": [100.0, 1.0, 110.0, 1.0],
        "MKT_VAL": [1000.0, 100.0, 1100.0, 100.0],
        "COST": [1000.0, 100.0, 1000.0, 100.0],
    }
    if include_accrued_column:
        holdings_rows["ACCRUED"] = [25.0, None, 30.0, None]
    holdings = pl.DataFrame(holdings_rows)
    holdings.write_csv(directory / "snapshot_a" / "holdings.csv")
    holdings.write_csv(directory / "snapshot_b" / "holdings.csv")

    portperf_rows = {
        "END_MV": [1230.0 if include_accrued_column else 1200.0],
        "FLOW": [0.0],
        "INCOME": [0.0],
        "GAIN_LOSS": [0.0],
        "PORTFOLIO_CODE": ["P1"],
        "FROM_DATE": ["2026-01-01"],
        "THRU_DATE": ["2026-01-31"],
        "BEGIN_MV": [1125.0 if include_accrued_column else 1100.0],
        "PORT_RETURN": [0.0],
    }
    pl.DataFrame(portperf_rows).write_csv(directory / "snapshot_a" / "portperf.csv")
    pl.DataFrame(portperf_rows).write_csv(directory / "snapshot_b" / "portperf.csv")

    begin_weight = 1025.0 / 1125.0 if include_accrued_column else 1000.0 / 1100.0
    secperf_rows = {
        "END_MV": [1130.0 if include_accrued_column else 1100.0],
        "INCOME": [0.0],
        "GAIN_LOSS": [0.0],
        "PORTFOLIO_CODE": ["P1"],
        "SECURITY_ID": ["BOND1"],
        "FROM_DATE": ["2026-01-01"],
        "THRU_DATE": ["2026-01-31"],
        "BEGIN_WEIGHT": [begin_weight],
        "BEGIN_MV": [1025.0 if include_accrued_column else 1000.0],
        "SEC_RETURN": [0.0],
        "CONTRIBUTION": [0.0],
    }
    pl.DataFrame(secperf_rows).write_csv(directory / "snapshot_a" / "secperf.csv")
    pl.DataFrame(secperf_rows).write_csv(directory / "snapshot_b" / "secperf.csv")

    out_of_period_transactions = pl.DataFrame(
        {
            "PORT": ["P1"],
            "SEC": ["BOND1"],
            "TRANSACTION_DATE": ["2025-01-01"],
            "TRAN": ["by"],
            "AMOUNT": [0.0],
        },
    )
    out_of_period_transactions.write_csv(directory / "snapshot_a" / "transactions.csv")
    out_of_period_transactions.write_csv(directory / "snapshot_b" / "transactions.csv")

    configuration = {
        **_required_yaml_settings(),
        "comparison": {
            "name": "Accrued valuation fixture",
            "level": "portfolio",
        },
        "snapshots": {
            "a": {"label": "a", "path": "snapshot_a"},
            "b": {"label": "b", "path": "snapshot_b"},
        },
        "files": {
            "portfolio_performance": "portperf.csv",
            "security_performance": "secperf.csv",
            "holdings": "holdings.csv",
            "transactions": "transactions.csv",
        },
        "portfolio_return_reconstruction": {
            "method": "modified_dietz",
            "beginning_value_source": "holdings",
            "ending_value_source": "holdings",
            "flow_source": "transactions",
            "flow_timing": "transaction_date",
            "day_count": "actual_days",
            "inclusion_rule": "beginning_of_day",
            "flow_categories": ["external_flow"],
            "income_categories": ["income"],
            "return_basis": "net",
            "sign_convention": "signed_amount",
        },
        "security_return_reconstruction": {
            "method": "modified_dietz",
            "beginning_value_source": "holdings",
            "ending_value_source": "holdings",
            "flow_source": "transactions",
            "flow_timing": "transaction_date",
            "day_count": "actual_days",
            "inclusion_rule": "beginning_of_day",
            "flow_categories": ["buy", "sell"],
            "income_categories": ["income"],
            "return_basis": "net",
            "sign_convention": "signed_amount",
        },
        "transaction_rules": {
            "by": {
                "transaction_category": "buy",
                "cash_flow_sign": "negative",
                "performance_flow_sign": "performance",
            },
        },
    }
    path = directory / "comparison.yaml"
    test_util.write_audit_test_yaml(path, configuration)
    return path


def _remove_fixture_column(
    directory: Path,
    *,
    file_name: str,
    column_name: str,
) -> None:
    """Remove one source column from both snapshots of a generated fixture."""
    for snapshot_name in ("snapshot_a", "snapshot_b"):
        path = directory / snapshot_name / file_name
        pl.read_csv(path).drop(column_name).write_csv(path)


def _blank_fixture_column(
    directory: Path,
    *,
    file_name: str,
    column_name: str,
) -> None:
    """Blank one source column in both snapshots of a generated fixture."""
    for snapshot_name in ("snapshot_a", "snapshot_b"):
        path = directory / snapshot_name / file_name
        frame = pl.read_csv(path)
        frame.with_columns(
            pl.lit(None).cast(frame.schema[column_name]).alias(column_name)
        ).write_csv(path)


def _add_fixture_columns(
    directory: Path,
    *,
    file_name: str,
    columns: dict[str, object],
) -> None:
    """Add constant source columns to both snapshots of a generated fixture."""
    for snapshot_name in ("snapshot_a", "snapshot_b"):
        path = directory / snapshot_name / file_name
        frame = pl.read_csv(path)
        frame.with_columns(
            pl.lit(value).alias(column)
            for column, value in columns.items()
        ).write_csv(path)


def _refresh_fixture_mappings(path: Path) -> None:
    """Refresh generated exact column mappings after a fixture mutation."""
    configuration = yaml.safe_load(path.read_text(encoding="utf-8"))
    test_util.write_audit_test_yaml(path, configuration)


def _comparison_path_with_reconstruction_method(
    directory: Path,
    method: str,
) -> Path:
    """Write a temporary demo comparison YAML with one reconstruction method."""
    configuration = yaml.safe_load(_PORTFOLIO_COMPARISON_PATH.read_text(encoding="utf-8"))
    if not isinstance(configuration, dict):
        raise AssertionError("Expected demo comparison YAML to be a mapping.")
    snapshots = configuration["snapshots"]
    if not isinstance(snapshots, dict):
        raise AssertionError("Expected snapshots to be a mapping.")
    for snapshot in snapshots.values():
        if not isinstance(snapshot, dict):
            raise AssertionError("Expected snapshot to be a mapping.")
        snapshot["path"] = str((_DEMO_AXYS_APX_DIRECTORY / str(snapshot["path"])).resolve())

    for section_name in (
        "portfolio_return_reconstruction",
        "security_return_reconstruction",
    ):
        section = configuration[section_name]
        if not isinstance(section, dict):
            raise AssertionError(f"Expected {section_name} to be a mapping.")
        section["method"] = method
        if method in {"simple_dietz", "modified_simple_dietz"}:
            section.pop("flow_timing", None)
            section.pop("day_count", None)
            section.pop("inclusion_rule", None)

    path = directory / "ppar_audit.yaml"
    test_util.write_audit_test_yaml(path, configuration)
    return path


class TestPerformanceComparisonReturnReconstruction(unittest.TestCase):
    """Verify portfolio return-reconstruction diagnostics."""

    def test_missing_reconstruction_yaml_returns_empty_table(self) -> None:
        """Comparisons opt into reconstruction explicitly."""
        checks = portfolio_return_reconstruction_checks(_BASELINE_COMPARISON_PATH)
        security_checks = security_return_reconstruction_checks(
            _BASELINE_COMPARISON_PATH
        )
        summary = return_reconstruction_summary(_BASELINE_COMPARISON_PATH)

        self.assertTrue(checks.is_empty())
        self.assertTrue(security_checks.is_empty())
        self.assertTrue(summary.is_empty())

    def test_shared_input_cache_builds_each_snapshot_index_once(self) -> None:
        """Portfolio and security checks reuse the same snapshot input indexes."""
        input_cache = _reconstruction._SnapshotDataIndexCache()
        index_builder = _reconstruction._snapshot_data_index

        with mock.patch.object(
            _reconstruction,
            "_snapshot_data_index",
            wraps=index_builder,
        ) as index_builder_spy:
            portfolio_return_reconstruction_checks(
                _PORTFOLIO_COMPARISON_PATH,
                _input_cache=input_cache,
            )
            security_return_reconstruction_checks(
                _PORTFOLIO_COMPARISON_PATH,
                _input_cache=input_cache,
            )

        self.assertEqual(index_builder_spy.call_count, 2)

    def test_demo_reconstruction_checks_show_review_statuses(self) -> None:
        """Packaged demo reconstruction only differs for named residual examples."""
        checks = portfolio_return_reconstruction_checks(_PORTFOLIO_COMPARISON_PATH)

        statuses = set(checks.get_column(RECONSTRUCTION_STATUS).to_list())
        self.assertEqual(
            statuses,
            {
                RECONSTRUCTION_STATUS_ALIGNED,
                RECONSTRUCTION_STATUS_DIFFERENT,
                RECONSTRUCTION_STATUS_MISSING_INPUTS,
            },
        )

        alpha_withdrawal = checks.filter(
            (pl.col("portfolio_id") == "ALPHA")
            & (pl.col("from_date") == pl.date(2026, 1, 1))
            & (pl.col("thru_date") == pl.date(2026, 1, 30))
        ).row(0, named=True)
        self.assertAlmostEqual(
            alpha_withdrawal[REPORTED_RETURN_DIFFERENCE],
            alpha_withdrawal[DERIVED_RETURN_DIFFERENCE],
            places=6,
        )
        self.assertEqual(
            alpha_withdrawal[RECONSTRUCTION_STATUS],
            RECONSTRUCTION_STATUS_ALIGNED,
        )

    def test_demo_portfolio_reconstruction_has_no_differences(self) -> None:
        """Portfolio reconstruction differences are limited to intentional rows."""
        checks = portfolio_return_reconstruction_checks(_PORTFOLIO_COMPARISON_PATH)
        different_rows = checks.filter(
            ~pl.col(RECONSTRUCTION_STATUS).is_in(
                [RECONSTRUCTION_STATUS_ALIGNED, RECONSTRUCTION_STATUS_MISSING_INPUTS]
            )
        )

        self.assertEqual(
            {
                (
                    row["portfolio_id"],
                    row["from_date"].isoformat(),
                    row["thru_date"].isoformat(),
                )
                for row in different_rows.iter_rows(named=True)
            },
            _INTENTIONAL_PORTFOLIO_DIFFERENT_KEYS,
        )

    def test_demo_security_reconstruction_checks_show_flow_inputs(self) -> None:
        """Security reconstruction treats buy/sell rows as security-level flows."""
        checks = security_return_reconstruction_checks(_PORTFOLIO_COMPARISON_PATH)

        statuses = set(checks.get_column(RECONSTRUCTION_STATUS).to_list())
        self.assertEqual(
            statuses,
            {
                RECONSTRUCTION_STATUS_ALIGNED,
                RECONSTRUCTION_STATUS_DIFFERENT,
                RECONSTRUCTION_STATUS_MISSING_INPUTS,
            },
        )
        different_rows = checks.filter(
            ~pl.col(RECONSTRUCTION_STATUS).is_in(
                [RECONSTRUCTION_STATUS_ALIGNED, RECONSTRUCTION_STATUS_MISSING_INPUTS]
            )
        )
        self.assertEqual(
            {
                (
                    row["portfolio_id"],
                    row["security_id"],
                    row["from_date"].isoformat(),
                    row["thru_date"].isoformat(),
                )
                for row in different_rows.iter_rows(named=True)
            },
            _INTENTIONAL_SECURITY_DIFFERENT_KEYS,
        )

        alpha_aapl = checks.filter(
            (pl.col("portfolio_id") == "ALPHA")
            & (pl.col("security_id") == "csusAAPL")
            & (pl.col("from_date") == pl.date(2026, 2, 28))
            & (pl.col("thru_date") == pl.date(2026, 3, 31))
        ).row(0, named=True)
        self.assertGreater(alpha_aapl[END_VALUE_DIFFERENCE], 0.0)
        self.assertGreater(alpha_aapl[NET_FLOW_DIFFERENCE], 0.0)
        self.assertGreater(alpha_aapl[WEIGHTED_FLOW_DIFFERENCE], 0.0)
        self.assertEqual(
            alpha_aapl[RECONSTRUCTION_STATUS],
            RECONSTRUCTION_STATUS_ALIGNED,
        )

    def test_security_reconstruction_checks_can_scope_active_keys(self) -> None:
        """Security reconstruction can limit work to requested review keys."""
        active_key = (
            "ALPHA",
            "csusAAPL",
            dt.date(2026, 2, 28),
            dt.date(2026, 3, 31),
        )
        full_checks = security_return_reconstruction_checks(_PORTFOLIO_COMPARISON_PATH)
        scoped_checks = security_return_reconstruction_checks(
            _PORTFOLIO_COMPARISON_PATH,
            active_keys=[active_key],
        )

        self.assertGreater(full_checks.height, scoped_checks.height)
        self.assertEqual(scoped_checks.height, 1)
        scoped_row = scoped_checks.row(0, named=True)
        self.assertEqual(scoped_row["portfolio_id"], "ALPHA")
        self.assertEqual(scoped_row["security_id"], "csusAAPL")
        self.assertEqual(scoped_row["from_date"].isoformat(), "2026-02-28")
        self.assertEqual(scoped_row["thru_date"].isoformat(), "2026-03-31")

    def test_reinvestment_pair_stays_out_of_portfolio_external_flows(self) -> None:
        """A dv/by reinvestment pair is not treated as a portfolio external flow."""
        with tempfile.TemporaryDirectory() as directory:
            path = _write_reinvestment_pair_fixture(Path(directory))

            portfolio_checks = portfolio_return_reconstruction_checks(path)
            security_checks = security_return_reconstruction_checks(path)

        portfolio_row = portfolio_checks.row(0, named=True)
        self.assertEqual(portfolio_row[NET_FLOW_B], 0.0)
        self.assertEqual(portfolio_row[WEIGHTED_FLOW_B], 0.0)

        security_row = security_checks.row(0, named=True)
        self.assertEqual(security_row[NET_FLOW_B], 100.0)
        self.assertEqual(security_row[INCOME_B], 100.0)
        self.assertEqual(security_row[DERIVED_NUMERATOR_B], 100.0)

    def test_accrued_is_included_in_reconstructed_holding_values(self) -> None:
        """Accrued interest is additive in portfolio and security values."""
        with tempfile.TemporaryDirectory() as directory:
            path = _write_accrued_value_fixture(
                Path(directory),
                include_accrued_column=True,
            )

            portfolio_checks = portfolio_return_reconstruction_checks(path)
            security_checks = security_return_reconstruction_checks(path)

        portfolio_row = portfolio_checks.row(0, named=True)
        self.assertAlmostEqual(portfolio_row[BEGIN_VALUE_B], 1125.0)
        self.assertAlmostEqual(portfolio_row[END_VALUE_B], 1230.0)
        self.assertAlmostEqual(portfolio_row[DERIVED_NUMERATOR_B], 105.0)
        self.assertAlmostEqual(portfolio_row[DERIVED_DENOMINATOR_B], 1125.0)

        security_row = security_checks.row(0, named=True)
        self.assertAlmostEqual(security_row[BEGIN_VALUE_B], 1025.0)
        self.assertAlmostEqual(security_row[END_VALUE_B], 1130.0)
        self.assertAlmostEqual(security_row[DERIVED_NUMERATOR_B], 105.0)
        self.assertAlmostEqual(security_row[DERIVED_DENOMINATOR_B], 1025.0)

    def test_missing_accrued_column_is_treated_as_zero_value(self) -> None:
        """Holdings without accrued use market value alone."""
        with tempfile.TemporaryDirectory() as directory:
            path = _write_accrued_value_fixture(
                Path(directory),
                include_accrued_column=False,
            )

            portfolio_checks = portfolio_return_reconstruction_checks(path)
            security_checks = security_return_reconstruction_checks(path)

        portfolio_row = portfolio_checks.row(0, named=True)
        self.assertAlmostEqual(portfolio_row[BEGIN_VALUE_B], 1100.0)
        self.assertAlmostEqual(portfolio_row[END_VALUE_B], 1200.0)

        security_row = security_checks.row(0, named=True)
        self.assertAlmostEqual(security_row[BEGIN_VALUE_B], 1000.0)
        self.assertAlmostEqual(security_row[END_VALUE_B], 1100.0)

    def test_demo_reconstruction_summary_counts_available_checks(self) -> None:
        """Summary table counts portfolio and security reconstruction checks."""
        summary = return_reconstruction_summary(_PORTFOLIO_COMPARISON_PATH)

        check_types = set(summary.get_column("reconstruction_check_type").to_list())
        self.assertEqual(check_types, {"Portfolio Return", "Security Return"})
        self.assertTrue((summary.get_column("row_count") > 0).all())

    def test_simple_dietz_uses_beginning_value_denominator(self) -> None:
        """Simple Dietz excludes all flow weighting from the denominator."""
        with tempfile.TemporaryDirectory() as directory:
            path = _comparison_path_with_reconstruction_method(
                Path(directory),
                "simple_dietz",
            )

            checks = portfolio_return_reconstruction_checks(path)

        alpha_withdrawal = checks.filter(
            (pl.col("portfolio_id") == "ALPHA")
            & (pl.col("from_date") == pl.date(2026, 1, 1))
            & (pl.col("thru_date") == pl.date(2026, 1, 30))
        ).row(0, named=True)
        self.assertNotEqual(alpha_withdrawal[NET_FLOW_DIFFERENCE], 0.0)
        self.assertEqual(alpha_withdrawal[WEIGHTED_FLOW_DIFFERENCE], 0.0)

    def test_modified_simple_dietz_uses_half_weighted_flows(self) -> None:
        """Modified Simple Dietz uses a 0.5 weight for every included flow."""
        with tempfile.TemporaryDirectory() as directory:
            path = _comparison_path_with_reconstruction_method(
                Path(directory),
                "modified_simple_dietz",
            )

            checks = portfolio_return_reconstruction_checks(path)

        alpha_withdrawal = checks.filter(
            (pl.col("portfolio_id") == "ALPHA")
            & (pl.col("from_date") == pl.date(2026, 1, 1))
            & (pl.col("thru_date") == pl.date(2026, 1, 30))
        ).row(0, named=True)
        self.assertNotEqual(alpha_withdrawal[NET_FLOW_DIFFERENCE], 0.0)
        self.assertAlmostEqual(
            alpha_withdrawal[WEIGHTED_FLOW_DIFFERENCE],
            alpha_withdrawal[NET_FLOW_DIFFERENCE] * 0.5,
        )

    def test_security_modified_simple_dietz_uses_half_weighted_flows(self) -> None:
        """Security reconstruction uses the same Modified Simple Dietz weight."""
        with tempfile.TemporaryDirectory() as directory:
            path = _comparison_path_with_reconstruction_method(
                Path(directory),
                "modified_simple_dietz",
            )

            checks = security_return_reconstruction_checks(path)

        alpha_aapl = checks.filter(
            (pl.col("portfolio_id") == "ALPHA")
            & (pl.col("security_id") == "csusAAPL")
            & (pl.col("from_date") == pl.date(2026, 2, 28))
            & (pl.col("thru_date") == pl.date(2026, 3, 31))
        ).row(0, named=True)
        self.assertNotEqual(alpha_aapl[NET_FLOW_DIFFERENCE], 0.0)
        self.assertAlmostEqual(
            alpha_aapl[WEIGHTED_FLOW_DIFFERENCE],
            alpha_aapl[NET_FLOW_DIFFERENCE] * 0.5,
        )

    def test_performance_calculation_requires_holdings_market_value_column(
        self,
    ) -> None:
        """Configured performance calculation requires holdings.market_value."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write_accrued_value_fixture(
                root,
                include_accrued_column=False,
            )
            _remove_fixture_column(
                root,
                file_name="holdings.csv",
                column_name="MKT_VAL",
            )

            with self.assertRaisesRegex(PpaError, "market_value"):
                portfolio_return_reconstruction_checks(path)

    def test_performance_calculation_rejects_blank_holdings_market_values(
        self,
    ) -> None:
        """Configured performance calculation requires finite holding values."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write_accrued_value_fixture(
                root,
                include_accrued_column=False,
            )
            _blank_fixture_column(
                root,
                file_name="holdings.csv",
                column_name="MKT_VAL",
            )

            with self.assertRaisesRegex(PpaError, "finite value on every row"):
                portfolio_return_reconstruction_checks(path)

    def test_performance_calculation_requires_foreign_base_market_values(
        self,
    ) -> None:
        """Foreign holdings require explicit portfolio-base market values."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write_accrued_value_fixture(
                root,
                include_accrued_column=False,
            )
            _add_fixture_columns(
                root,
                file_name="holdings.csv",
                columns={
                    "CURRENCY": "EUR",
                    "BASE_CURRENCY": "USD",
                    "BASE_MKT_VAL": None,
                },
            )
            _refresh_fixture_mappings(path)

            with self.assertRaisesRegex(PpaError, "base_market_value"):
                portfolio_return_reconstruction_checks(path)

    def test_performance_calculation_requires_transaction_code_column(self) -> None:
        """Configured performance calculation requires transaction_code."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write_accrued_value_fixture(
                root,
                include_accrued_column=False,
            )
            _remove_fixture_column(
                root,
                file_name="transactions.csv",
                column_name="TRAN",
            )

            with self.assertRaisesRegex(PpaError, "transaction_code"):
                portfolio_return_reconstruction_checks(path)

    def test_performance_calculation_rejects_blank_transaction_codes(self) -> None:
        """Configured performance calculation requires a code on every row."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write_accrued_value_fixture(
                root,
                include_accrued_column=False,
            )
            _blank_fixture_column(
                root,
                file_name="transactions.csv",
                column_name="TRAN",
            )

            with self.assertRaisesRegex(PpaError, "must contain a value on every row"):
                portfolio_return_reconstruction_checks(path)

    def test_performance_calculation_requires_transaction_amount_column(self) -> None:
        """Configured performance calculation requires transactions.amount."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write_accrued_value_fixture(
                root,
                include_accrued_column=False,
            )
            _remove_fixture_column(
                root,
                file_name="transactions.csv",
                column_name="AMOUNT",
            )

            with self.assertRaisesRegex(PpaError, "amount"):
                portfolio_return_reconstruction_checks(path)

    def test_performance_calculation_rejects_blank_financial_amounts(self) -> None:
        """External or performance transactions require finite amounts."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write_accrued_value_fixture(
                root,
                include_accrued_column=False,
            )
            _blank_fixture_column(
                root,
                file_name="transactions.csv",
                column_name="AMOUNT",
            )

            with self.assertRaisesRegex(PpaError, "finite value for every"):
                portfolio_return_reconstruction_checks(path)

    def test_performance_calculation_requires_foreign_base_amounts(self) -> None:
        """Foreign transactions require explicit portfolio-base amounts."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write_accrued_value_fixture(
                root,
                include_accrued_column=False,
            )
            _add_fixture_columns(
                root,
                file_name="transactions.csv",
                columns={
                    "AMOUNT": 100.0,
                    "CURRENCY": "EUR",
                    "BASE_CURRENCY": "USD",
                    "BASE_AMOUNT": None,
                },
            )
            _refresh_fixture_mappings(path)

            with self.assertRaisesRegex(PpaError, "base_amount"):
                portfolio_return_reconstruction_checks(path)

    def test_malformed_reconstruction_yaml_fails_up_front(self) -> None:
        """Opted-in reconstruction YAML must include every required field."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        **_required_yaml_settings(),
                        "comparison": {"level": "portfolio"},
                        "portfolio_return_reconstruction": {
                            "method": "modified_dietz",
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(PpaError, "missing required keys"):
                AuditSpecification(path)


if __name__ == "__main__":
    unittest.main()
