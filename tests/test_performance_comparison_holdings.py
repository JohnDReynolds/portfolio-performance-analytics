"""Tests for loading normalized holding comparison sources."""

# Python imports
from pathlib import Path
import tempfile
import unittest

# Third-party imports
import polars as pl
import yaml

# Project imports
from ppar.errors import PpaError
from ppar.performance_comparison import (
    PerformanceComparisonSpecification,
    HoldingsLoader,
)
from ppar.performance_comparison import schema as pc_cols

_BASELINE_COMPARISON_PATH = Path("tests/data/axys/validation/ppar_performance_comparison.yaml")


def _write_yaml(directory: Path, contents: object) -> Path:
    """Write comparison YAML contents and return the path."""
    path = directory / "ppar_performance_comparison.yaml"
    path.write_text(yaml.safe_dump(contents), encoding="utf-8")
    return path


def _minimal_specification(directory: Path) -> dict[str, object]:
    """Return a minimal valid comparison specification with portfolio files."""
    for snapshot_name in ("snapshot_a", "snapshot_b"):
        snapshot_path = directory / snapshot_name
        snapshot_path.mkdir()
        pl.DataFrame(
            {
                "PORTFOLIO_CODE": ["P1"],
                "FROM_DATE": ["2025-01-01"],
                "THRU_DATE": ["2025-01-31"],
                "PORT_RETURN": [0.01],
            }
        ).write_csv(snapshot_path / "portperf.csv")
    return {
        "snapshots": {
            "a": {"path": "snapshot_a"},
            "b": {"path": "snapshot_b"},
        },
        "files": {"portfolio_performance": "portperf.csv"},
    }


class TestHoldingsLoader(unittest.TestCase):
    """Verify normalized holding loading for snapshots."""

    def test_load_baseline_snapshot_a_holdings(self) -> None:
        """Holding rows load with normalized internal columns."""
        specification = PerformanceComparisonSpecification(_BASELINE_COMPARISON_PATH)
        frame = HoldingsLoader(specification).load("a")
        assert frame is not None

        self.assertTrue(set(pc_cols.HOLDINGS_REQUIRED_COLUMNS).issubset(frame.columns))
        self.assertIn(pc_cols.MARKET_VALUE, frame.columns)
        self.assertIn(pc_cols.ACCRUED, frame.columns)
        self.assertEqual(frame.schema[pc_cols.HOLDING_DATE], pl.Date)

        target_row = frame.filter(
            (pl.col(pc_cols.PORTFOLIO_ID) == "PORT_A")
            & (pl.col(pc_cols.SECURITY_ID) == "AAPL")
            & (pl.col(pc_cols.HOLDING_DATE) == pl.date(2025, 5, 30))
        ).row(0, named=True)
        self.assertEqual(target_row[pc_cols.QUANTITY], 200.0)
        self.assertAlmostEqual(target_row[pc_cols.MARKET_VALUE], 52971.24)
        self.assertEqual(target_row[pc_cols.CURRENCY], "USD")

    def test_portfolio_base_currency_fills_missing_holding_value(self) -> None:
        """Portfolio performance supplies authoritative holding base currency."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "holdings": "holdings.csv",
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                snapshot_path = directory / snapshot_name
                pl.DataFrame(
                    {
                        "PORTFOLIO_CODE": ["P1"],
                        "FROM_DATE": ["2025-01-01"],
                        "THRU_DATE": ["2025-01-31"],
                        "PORT_RETURN": [0.01],
                        "BASE_CURRENCY": ["usd"],
                    }
                ).write_csv(snapshot_path / "portperf.csv")
                pl.DataFrame(
                    {
                        "PORT": ["P1"],
                        "SEC": ["S1"],
                        "HOLDING_DATE": ["2025-01-31"],
                        "CURRENCY": ["EUR"],
                        "MKT_VAL": [100.0],
                    }
                ).write_csv(snapshot_path / "holdings.csv")
            path = _write_yaml(directory, configuration)

            frame = HoldingsLoader(PerformanceComparisonSpecification(path)).load("a")

        assert frame is not None
        self.assertEqual(frame[pc_cols.BASE_CURRENCY].to_list(), ["USD"])

    def test_base_accrued_alias_loads_as_base_currency_value(self) -> None:
        """Foreign accrued income can supply an explicit base counterpart."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "holdings": "holdings.csv",
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                snapshot_path = directory / snapshot_name
                pl.DataFrame(
                    {
                        "PORTFOLIO_CODE": ["P1"],
                        "FROM_DATE": ["2025-01-01"],
                        "THRU_DATE": ["2025-01-31"],
                        "PORT_RETURN": [0.01],
                        "BASE_CURRENCY": ["USD"],
                    }
                ).write_csv(snapshot_path / "portperf.csv")
                pl.DataFrame(
                    {
                        "PORT": ["P1"],
                        "SEC": ["S1"],
                        "HOLDING_DATE": ["2025-01-31"],
                        "CURRENCY": ["EUR"],
                        "ACCRUED": [10.0],
                        "BASE_ACCRUED_INCOME": [11.0],
                    }
                ).write_csv(snapshot_path / "holdings.csv")
            path = _write_yaml(directory, configuration)

            frame = HoldingsLoader(PerformanceComparisonSpecification(path)).load("a")

        assert frame is not None
        self.assertEqual(frame[pc_cols.ACCRUED].to_list(), [10.0])
        self.assertEqual(frame[pc_cols.BASE_ACCRUED].to_list(), [11.0])

    def test_conflicting_portfolio_base_currencies_raise_error_504(self) -> None:
        """One portfolio cannot declare multiple authoritative currencies."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "holdings": "holdings.csv",
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                snapshot_path = directory / snapshot_name
                pl.DataFrame(
                    {
                        "PORTFOLIO_CODE": ["P1", "P1"],
                        "FROM_DATE": ["2025-01-01", "2025-02-01"],
                        "THRU_DATE": ["2025-01-31", "2025-02-28"],
                        "PORT_RETURN": [0.01, 0.02],
                        "BASE_CURRENCY": ["USD", "EUR"],
                    }
                ).write_csv(snapshot_path / "portperf.csv")
                pl.DataFrame(
                    {
                        "PORT": ["P1"],
                        "SEC": ["S1"],
                        "HOLDING_DATE": ["2025-01-31"],
                        "MKT_VAL": [100.0],
                    }
                ).write_csv(snapshot_path / "holdings.csv")
            path = _write_yaml(directory, configuration)

            with self.assertRaises(PpaError) as context:
                HoldingsLoader(PerformanceComparisonSpecification(path)).load("a")

        message = str(context.exception)
        self.assertTrue(message.startswith("Error 504"))
        self.assertIn("one base_currency per portfolio", message)

    def test_omitted_positions_returns_none(self) -> None:
        """Holdings are optional when omitted from YAML."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            path = _write_yaml(directory, _minimal_specification(directory))
            specification = PerformanceComparisonSpecification(path)

            self.assertIsNone(HoldingsLoader(specification).load("a"))

    def test_missing_optional_positions_returns_none(self) -> None:
        """Missing optional holding files do not block loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "holdings": "missing_positions.csv",
            }
            path = _write_yaml(directory, configuration)
            specification = PerformanceComparisonSpecification(path)

            self.assertIsNone(HoldingsLoader(specification).load("a"))

    def test_missing_required_column_raises_error_502(self) -> None:
        """Existing holding files must contain portfolio, security, and date."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "holdings": "holdings.csv",
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(
                    {
                        "PORT": ["P1"],
                        "SEC": ["S1"],
                        "MV": [10.0],
                    }
                ).write_csv(directory / snapshot_name / "holdings.csv")
            path = _write_yaml(directory, configuration)
            specification = PerformanceComparisonSpecification(path)

            with self.assertRaises(PpaError) as context:
                HoldingsLoader(specification).load("a")

            self.assertTrue(str(context.exception).startswith("Error 502"))
            self.assertIn("holding_date", str(context.exception))

    def test_ambiguous_required_column_raises_error_502(self) -> None:
        """Holding identifier columns must not match multiple aliases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "holdings": "holdings.csv",
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(
                    {
                        "PORT": ["P1"],
                        "PORTFOLIO_ID": ["P1"],
                        "SEC": ["S1"],
                        "HOLDING_DATE": ["2025-01-31"],
                    }
                ).write_csv(directory / snapshot_name / "holdings.csv")
            path = _write_yaml(directory, configuration)
            specification = PerformanceComparisonSpecification(path)

            with self.assertRaises(PpaError) as context:
                HoldingsLoader(specification).load("a")

            self.assertTrue(str(context.exception).startswith("Error 502"))
            self.assertIn("Ambiguous holdings", str(context.exception))

    def test_nonnumeric_market_value_raises_error_502(self) -> None:
        """Malformed holding numeric values fail with field-level context."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "holdings": "holdings.csv",
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                (directory / snapshot_name / "holdings.csv").write_text(
                    "PORT,SEC,HOLDING_DATE,QTY,MKT_VAL\n"
                    "P1,S1,2025-01-31,10,--\n",
                    encoding="utf-8",
                )
            path = _write_yaml(directory, configuration)
            specification = PerformanceComparisonSpecification(path)

            with self.assertRaises(PpaError) as context:
                HoldingsLoader(specification).load("a")

            message = str(context.exception)
            self.assertTrue(message.startswith("Error 502"))
            self.assertIn("holdings", message)
            self.assertIn("market_value", message)
            self.assertIn("--", message)


if __name__ == "__main__":
    unittest.main()
