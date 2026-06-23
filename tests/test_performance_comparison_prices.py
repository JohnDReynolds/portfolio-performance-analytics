"""Tests for loading normalized price comparison sources."""

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
    PricesLoader,
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


class TestPricesLoader(unittest.TestCase):
    """Verify normalized price loading for snapshots."""

    def test_load_baseline_snapshot_a_prices(self) -> None:
        """Price rows load with normalized internal columns."""
        specification = PerformanceComparisonSpecification(_BASELINE_COMPARISON_PATH)
        frame = PricesLoader(specification).load("a")
        assert frame is not None

        self.assertTrue(set(pc_cols.PRICES_REQUIRED_COLUMNS).issubset(frame.columns))
        self.assertIn(pc_cols.CURRENCY, frame.columns)
        self.assertIn(pc_cols.PRICE_SOURCE, frame.columns)
        self.assertEqual(frame.schema[pc_cols.PRICE_DATE], pl.Date)

        target_row = frame.filter(
            (pl.col(pc_cols.SECURITY_ID) == "AAPL")
            & (pl.col(pc_cols.PRICE_DATE) == pl.date(2025, 5, 1))
        ).row(0, named=True)
        self.assertAlmostEqual(target_row[pc_cols.PRICE], 254.0959)
        self.assertEqual(target_row[pc_cols.CURRENCY], "USD")
        self.assertEqual(target_row[pc_cols.PRICE_SOURCE], "SYNTH")

    def test_omitted_prices_returns_none(self) -> None:
        """Prices are optional when omitted from YAML."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            path = _write_yaml(directory, _minimal_specification(directory))
            specification = PerformanceComparisonSpecification(path)

            self.assertIsNone(PricesLoader(specification).load("a"))

    def test_missing_optional_prices_returns_none(self) -> None:
        """Missing optional price files do not block loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "prices": "missing_prices.csv",
            }
            path = _write_yaml(directory, configuration)
            specification = PerformanceComparisonSpecification(path)

            self.assertIsNone(PricesLoader(specification).load("a"))

    def test_missing_required_column_raises_error_502(self) -> None:
        """Existing price files must contain security, date, and price columns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "prices": "prices.csv",
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(
                    {
                        "SEC": ["S1"],
                        "PRICE_DATE": ["2025-01-31"],
                    }
                ).write_csv(directory / snapshot_name / "prices.csv")
            path = _write_yaml(directory, configuration)
            specification = PerformanceComparisonSpecification(path)

            with self.assertRaises(PpaError) as context:
                PricesLoader(specification).load("a")

            self.assertTrue(str(context.exception).startswith("Error 502"))
            self.assertIn("price", str(context.exception))

    def test_ambiguous_required_column_raises_error_502(self) -> None:
        """Price identifier columns must not match multiple aliases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "prices": "prices.csv",
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(
                    {
                        "SEC": ["S1"],
                        "SECURITY_ID": ["S1"],
                        "PRICE_DATE": ["2025-01-31"],
                        "PRICE": [10.0],
                    }
                ).write_csv(directory / snapshot_name / "prices.csv")
            path = _write_yaml(directory, configuration)
            specification = PerformanceComparisonSpecification(path)

            with self.assertRaises(PpaError) as context:
                PricesLoader(specification).load("a")

            self.assertTrue(str(context.exception).startswith("Error 502"))
            self.assertIn("Ambiguous prices", str(context.exception))

    def test_ambiguous_optional_column_raises_error_502(self) -> None:
        """Optional price columns must not match multiple aliases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "prices": "prices.csv",
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(
                    {
                        "SEC": ["S1"],
                        "PRICE_DATE": ["2025-01-31"],
                        "PRICE": [10.0],
                        "SOURCE": ["A"],
                        "PRICE_SOURCE": ["B"],
                    }
                ).write_csv(directory / snapshot_name / "prices.csv")
            path = _write_yaml(directory, configuration)
            specification = PerformanceComparisonSpecification(path)

            with self.assertRaises(PpaError) as context:
                PricesLoader(specification).load("a")

            self.assertTrue(str(context.exception).startswith("Error 502"))
            self.assertIn("Ambiguous prices", str(context.exception))
            self.assertIn("price_source", str(context.exception))

    def test_nonnumeric_price_raises_error_502(self) -> None:
        """Malformed price numeric values fail with field-level context."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "prices": "prices.csv",
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                (directory / snapshot_name / "prices.csv").write_text(
                    "SEC,PRICE_DATE,PRICE\n"
                    "S1,2025-01-31,N/A\n",
                    encoding="utf-8",
                )
            path = _write_yaml(directory, configuration)
            specification = PerformanceComparisonSpecification(path)

            with self.assertRaises(PpaError) as context:
                PricesLoader(specification).load("a")

            message = str(context.exception)
            self.assertTrue(message.startswith("Error 502"))
            self.assertIn("prices", message)
            self.assertIn("price", message)
            self.assertIn("N/A", message)


if __name__ == "__main__":
    unittest.main()
