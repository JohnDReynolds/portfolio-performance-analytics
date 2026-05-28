"""Tests for loading normalized transaction comparison sources."""

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
    TransactionsLoader,
)
from ppar.performance_comparison import columns as pc_cols

_BASELINE_COMPARISON_PATH = Path("tests/data/axys/ppar_performance_comparison.yaml")


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


class TestTransactionsLoader(unittest.TestCase):
    """Verify normalized transaction loading for snapshots."""

    def test_load_baseline_snapshot_a_transactions(self) -> None:
        """Transaction rows load with normalized internal columns."""
        specification = PerformanceComparisonSpecification(_BASELINE_COMPARISON_PATH)
        frame = TransactionsLoader(specification).load("a")
        assert frame is not None

        self.assertTrue(
            set(pc_cols.TRANSACTIONS_REQUIRED_COLUMNS).issubset(frame.columns)
        )
        self.assertIn(pc_cols.SETTLEMENT_DATE, frame.columns)
        self.assertIn(pc_cols.TRANSACTION_CODE, frame.columns)
        self.assertIn(pc_cols.BROKER, frame.columns)
        self.assertEqual(frame.schema[pc_cols.TRANSACTION_DATE], pl.Date)
        self.assertEqual(frame.schema[pc_cols.SETTLEMENT_DATE], pl.Date)

        target_row = frame.filter(
            (pl.col(pc_cols.PORTFOLIO_ID) == "PORT_A")
            & (pl.col(pc_cols.SECURITY_ID) == "AAPL")
            & (pl.col(pc_cols.TRANSACTION_DATE) == pl.date(2025, 5, 1))
        ).row(0, named=True)
        self.assertEqual(target_row[pc_cols.TRANSACTION_CODE], "BUY")
        self.assertEqual(target_row[pc_cols.QUANTITY], 200.0)
        self.assertEqual(target_row[pc_cols.BROKER], "INIT")

    def test_omitted_transactions_returns_none(self) -> None:
        """Transactions are optional when omitted from YAML."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            path = _write_yaml(directory, _minimal_specification(directory))
            specification = PerformanceComparisonSpecification(path)

            self.assertIsNone(TransactionsLoader(specification).load("a"))

    def test_missing_optional_transactions_returns_none(self) -> None:
        """Missing optional transaction files do not block loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "transactions": "missing_transactions.csv",
            }
            path = _write_yaml(directory, configuration)
            specification = PerformanceComparisonSpecification(path)

            self.assertIsNone(TransactionsLoader(specification).load("a"))

    def test_missing_required_column_raises_error_502(self) -> None:
        """Existing transaction files must contain portfolio, security, and date."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "transactions": "transactions.csv",
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(
                    {
                        "PORT": ["P1"],
                        "SEC": ["S1"],
                        "AMOUNT": [10.0],
                    }
                ).write_csv(directory / snapshot_name / "transactions.csv")
            path = _write_yaml(directory, configuration)
            specification = PerformanceComparisonSpecification(path)

            with self.assertRaises(PpaError) as context:
                TransactionsLoader(specification).load("a")

            self.assertTrue(str(context.exception).startswith("Error 502"))
            self.assertIn("transaction_date", str(context.exception))

    def test_ambiguous_required_column_raises_error_502(self) -> None:
        """Transaction identifier columns must not match multiple aliases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "transactions": "transactions.csv",
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(
                    {
                        "PORT": ["P1"],
                        "PORTFOLIO_ID": ["P1"],
                        "SEC": ["S1"],
                        "TRANSACTION_DATE": ["2025-01-31"],
                    }
                ).write_csv(directory / snapshot_name / "transactions.csv")
            path = _write_yaml(directory, configuration)
            specification = PerformanceComparisonSpecification(path)

            with self.assertRaises(PpaError) as context:
                TransactionsLoader(specification).load("a")

            self.assertTrue(str(context.exception).startswith("Error 502"))
            self.assertIn("Ambiguous transactions", str(context.exception))


if __name__ == "__main__":
    unittest.main()
