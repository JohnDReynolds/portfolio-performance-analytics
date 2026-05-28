"""Tests for loading normalized cash comparison sources."""

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
    CashLoader,
    PerformanceComparisonSpecification,
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


class TestCashLoader(unittest.TestCase):
    """Verify normalized cash loading for snapshots."""

    def test_load_baseline_snapshot_a_cash(self) -> None:
        """Cash rows load with normalized internal columns."""
        specification = PerformanceComparisonSpecification(_BASELINE_COMPARISON_PATH)
        frame = CashLoader(specification).load("a")
        assert frame is not None

        self.assertTrue(set(pc_cols.CASH_REQUIRED_COLUMNS).issubset(frame.columns))
        self.assertIn(pc_cols.CASH_BALANCE, frame.columns)
        self.assertIn(pc_cols.MARKET_VALUE, frame.columns)
        self.assertEqual(frame.schema[pc_cols.CASH_DATE], pl.Date)

        target_row = frame.filter(
            (pl.col(pc_cols.PORTFOLIO_ID) == "PORT_A")
            & (pl.col(pc_cols.CASH_DATE) == pl.date(2025, 5, 30))
        ).row(0, named=True)
        self.assertAlmostEqual(target_row[pc_cols.CASH_BALANCE], 26773.49)
        self.assertAlmostEqual(target_row[pc_cols.MARKET_VALUE], 26773.49)
        self.assertEqual(target_row[pc_cols.CURRENCY], "USD")

    def test_omitted_cash_returns_none(self) -> None:
        """Cash is optional when omitted from YAML."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            path = _write_yaml(directory, _minimal_specification(directory))
            specification = PerformanceComparisonSpecification(path)

            self.assertIsNone(CashLoader(specification).load("a"))

    def test_missing_optional_cash_returns_none(self) -> None:
        """Missing optional cash files do not block loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "cash": "missing_cash.csv",
            }
            path = _write_yaml(directory, configuration)
            specification = PerformanceComparisonSpecification(path)

            self.assertIsNone(CashLoader(specification).load("a"))

    def test_missing_required_column_raises_error_502(self) -> None:
        """Existing cash files must contain portfolio and date columns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "cash": "cash.csv",
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(
                    {
                        "PORT": ["P1"],
                        "CASH": [10.0],
                    }
                ).write_csv(directory / snapshot_name / "cash.csv")
            path = _write_yaml(directory, configuration)
            specification = PerformanceComparisonSpecification(path)

            with self.assertRaises(PpaError) as context:
                CashLoader(specification).load("a")

            self.assertTrue(str(context.exception).startswith("Error 502"))
            self.assertIn("cash_date", str(context.exception))

    def test_ambiguous_required_column_raises_error_502(self) -> None:
        """Cash portfolio columns must not match multiple aliases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "cash": "cash.csv",
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(
                    {
                        "PORT": ["P1"],
                        "PORTFOLIO_ID": ["P1"],
                        "CASH_DATE": ["2025-01-31"],
                    }
                ).write_csv(directory / snapshot_name / "cash.csv")
            path = _write_yaml(directory, configuration)
            specification = PerformanceComparisonSpecification(path)

            with self.assertRaises(PpaError) as context:
                CashLoader(specification).load("a")

            self.assertTrue(str(context.exception).startswith("Error 502"))
            self.assertIn("Ambiguous cash", str(context.exception))


if __name__ == "__main__":
    unittest.main()
