"""Tests for portfolio return-reconstruction diagnostics."""

from __future__ import annotations

# Python imports
from pathlib import Path
import tempfile
import unittest

# Third-party imports
import polars as pl
import yaml

# Project imports
from ppar.errors import PpaError
from ppar.performance_comparison.return_reconstruction import (
    DERIVED_RETURN_DIFFERENCE,
    DERIVED_NUMERATOR_B,
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
from ppar.performance_comparison.specification import PerformanceComparisonSpecification

_PORTFOLIO_COMPARISON_PATH = Path(
    "ppar/demos/data/axys_performance_comparison/axys_performance_comparison.yaml"
)
_BASELINE_COMPARISON_PATH = Path(
    "tests/data/axys/validation/ppar_performance_comparison.yaml"
)
_DEMO_AXYS_DIRECTORY = Path("ppar/demos/data/axys_performance_comparison")
_INTENTIONAL_PORTFOLIO_DIFFERENT_KEYS = {
    ("BALANCED", "2026-05-01", "2026-05-29"),
    ("INCOME", "2026-04-01", "2026-04-30"),
}
_INTENTIONAL_SECURITY_DIFFERENT_KEYS = {
    ("BALANCED", "MSFT", "2026-05-01", "2026-05-29"),
    ("INCOME", "TNOTE5Y", "2026-04-01", "2026-04-30"),
}


def _write_reinvestment_pair_fixture(directory: Path) -> Path:
    """Write a tiny comparison fixture with a dividend reinvestment pair."""
    for snapshot_name in ("snapshot_a", "snapshot_b"):
        (directory / snapshot_name).mkdir()

    holdings_a = pl.DataFrame(
        {
            "PORT": ["P1", "P1", "P1", "P1"],
            "SEC": ["AAPL", "CASH_USD", "AAPL", "CASH_USD"],
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
        "comparison": {"name": "Reinvestment pair fixture"},
        "snapshots": {
            "a": {"label": "a", "path": "snapshot_a", "vendor": "axys"},
            "b": {"label": "b", "path": "snapshot_b", "vendor": "axys"},
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
    path.write_text(yaml.safe_dump(configuration), encoding="utf-8")
    return path


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
        snapshot["path"] = str((_DEMO_AXYS_DIRECTORY / str(snapshot["path"])).resolve())
        snapshot["schema"] = str((_DEMO_AXYS_DIRECTORY / str(snapshot["schema"])).resolve())

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

    path = directory / "ppar_performance_comparison.yaml"
    path.write_text(yaml.safe_dump(configuration), encoding="utf-8")
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
            & (pl.col("security_id") == "AAPL")
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
            & (pl.col("security_id") == "AAPL")
            & (pl.col("from_date") == pl.date(2026, 2, 28))
            & (pl.col("thru_date") == pl.date(2026, 3, 31))
        ).row(0, named=True)
        self.assertNotEqual(alpha_aapl[NET_FLOW_DIFFERENCE], 0.0)
        self.assertAlmostEqual(
            alpha_aapl[WEIGHTED_FLOW_DIFFERENCE],
            alpha_aapl[NET_FLOW_DIFFERENCE] * 0.5,
        )

    def test_malformed_reconstruction_yaml_fails_up_front(self) -> None:
        """Opted-in reconstruction YAML must include every required field."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.yaml"
            path.write_text(
                "\n".join(
                    [
                        "portfolio_return_reconstruction:",
                        "  method: modified_dietz",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(PpaError, "missing required keys"):
                PerformanceComparisonSpecification(path)


if __name__ == "__main__":
    unittest.main()
