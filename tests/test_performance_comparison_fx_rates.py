"""Tests for loading normalized FX rate comparison sources."""

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
    FxRatesLoader,
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


class TestFxRatesLoader(unittest.TestCase):
    """Verify normalized FX rate loading for snapshots."""

    def test_load_baseline_snapshot_a_fx_rates(self) -> None:
        """FX rate rows load with normalized internal columns."""
        specification = PerformanceComparisonSpecification(_BASELINE_COMPARISON_PATH)
        frame = FxRatesLoader(specification).load("a")
        assert frame is not None

        self.assertTrue(set(pc_cols.FX_RATES_REQUIRED_COLUMNS).issubset(frame.columns))
        self.assertIn(pc_cols.RATE_SOURCE, frame.columns)
        self.assertIn(pc_cols.RATE_TYPE, frame.columns)
        self.assertEqual(frame.schema[pc_cols.RATE_DATE], pl.Date)

        target_row = frame.filter(
            (pl.col(pc_cols.FROM_CURRENCY) == "EUR")
            & (pl.col(pc_cols.TO_CURRENCY) == "USD")
        ).row(0, named=True)
        self.assertAlmostEqual(target_row[pc_cols.FX_RATE], 1.0825)
        self.assertEqual(target_row[pc_cols.RATE_SOURCE], "SYNTH")
        self.assertEqual(target_row[pc_cols.RATE_TYPE], "SPOT")

    def test_omitted_fx_rates_returns_none(self) -> None:
        """FX rates are optional when omitted from YAML."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            path = _write_yaml(directory, _minimal_specification(directory))
            specification = PerformanceComparisonSpecification(path)

            self.assertIsNone(FxRatesLoader(specification).load("a"))

    def test_missing_optional_fx_rates_returns_none(self) -> None:
        """Missing optional FX rate files do not block loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "fx_rates": "missing_fx_rates.csv",
            }
            path = _write_yaml(directory, configuration)
            specification = PerformanceComparisonSpecification(path)

            self.assertIsNone(FxRatesLoader(specification).load("a"))

    def test_missing_required_column_raises_error_502(self) -> None:
        """Existing FX rate files must contain currencies, date, and rate."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "fx_rates": "fx_rates.csv",
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(
                    {
                        "FROM_CCY": ["EUR"],
                        "TO_CCY": ["USD"],
                        "RATE_DATE": ["2025-01-31"],
                    }
                ).write_csv(directory / snapshot_name / "fx_rates.csv")
            path = _write_yaml(directory, configuration)
            specification = PerformanceComparisonSpecification(path)

            with self.assertRaises(PpaError) as context:
                FxRatesLoader(specification).load("a")

            self.assertTrue(str(context.exception).startswith("Error 502"))
            self.assertIn("fx_rate", str(context.exception))

    def test_ambiguous_required_column_raises_error_502(self) -> None:
        """FX rate currency columns must not match multiple aliases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "fx_rates": "fx_rates.csv",
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(
                    {
                        "FROM_CCY": ["EUR"],
                        "FROM_CURRENCY": ["EUR"],
                        "TO_CCY": ["USD"],
                        "RATE_DATE": ["2025-01-31"],
                        "FX_RATE": [1.1],
                    }
                ).write_csv(directory / snapshot_name / "fx_rates.csv")
            path = _write_yaml(directory, configuration)
            specification = PerformanceComparisonSpecification(path)

            with self.assertRaises(PpaError) as context:
                FxRatesLoader(specification).load("a")

            self.assertTrue(str(context.exception).startswith("Error 502"))
            self.assertIn("Ambiguous fx rates", str(context.exception))


if __name__ == "__main__":
    unittest.main()
