"""Tests for loading normalized security master comparison sources."""

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
    SecurityMasterLoader,
)
from ppar.performance_comparison import schema as pc_cols

_BASELINE_COMPARISON_PATH = Path("tests/data/axys/validation/ppar_performance_comparison.yaml")
_RESTATEMENT_COMPARISON_PATH = Path(
    "tests/data/axys/validation/ppar_performance_comparison_restatement.yaml"
)


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


class TestSecurityMasterLoader(unittest.TestCase):
    """Verify normalized security master loading for snapshots."""

    def test_load_baseline_snapshot_a_security_master(self) -> None:
        """Security master rows load with normalized internal columns."""
        specification = PerformanceComparisonSpecification(_BASELINE_COMPARISON_PATH)
        frame = SecurityMasterLoader(specification).load("a")
        assert frame is not None

        self.assertTrue(
            set(pc_cols.SECURITY_MASTER_REQUIRED_COLUMNS).issubset(frame.columns)
        )
        self.assertIn(pc_cols.SECURITY_NAME, frame.columns)
        self.assertIn(pc_cols.SECTOR, frame.columns)

        target_row = frame.filter(pl.col(pc_cols.SECURITY_ID) == "AAPL").row(
            0,
            named=True,
        )
        self.assertEqual(target_row[pc_cols.SECURITY_NAME], "Apple Inc")

    def test_restatement_snapshot_b_loads_changed_security_master(self) -> None:
        """The restatement fixture exposes controlled security master changes."""
        specification = PerformanceComparisonSpecification(_RESTATEMENT_COMPARISON_PATH)
        frame = SecurityMasterLoader(specification).load("b")
        assert frame is not None

        target_row = frame.filter(pl.col(pc_cols.SECURITY_ID) == "AAPL").row(
            0,
            named=True,
        )
        self.assertEqual(
            target_row[pc_cols.SECURITY_NAME],
            "Apple Inc Restated Name",
        )
        self.assertEqual(target_row[pc_cols.SECTOR], "TECH_RESTATED")

    def test_omitted_security_master_returns_none(self) -> None:
        """Security master is optional when omitted from YAML."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            path = _write_yaml(directory, _minimal_specification(directory))
            specification = PerformanceComparisonSpecification(path)

            self.assertIsNone(SecurityMasterLoader(specification).load("a"))

    def test_missing_optional_security_master_returns_none(self) -> None:
        """Missing optional security master files do not block loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "security_master": "missing_sec_ref.csv",
            }
            path = _write_yaml(directory, configuration)
            specification = PerformanceComparisonSpecification(path)

            self.assertIsNone(SecurityMasterLoader(specification).load("a"))

    def test_missing_required_column_raises_error_502(self) -> None:
        """Existing security master files must contain security identifiers."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "security_master": "sec_ref.csv",
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame({"SECURITY_NAME": ["Security One"]}).write_csv(
                    directory / snapshot_name / "sec_ref.csv"
                )
            path = _write_yaml(directory, configuration)
            specification = PerformanceComparisonSpecification(path)

            with self.assertRaises(PpaError) as context:
                SecurityMasterLoader(specification).load("a")

            self.assertTrue(str(context.exception).startswith("Error 502"))
            self.assertIn("security_id", str(context.exception))

    def test_ambiguous_required_column_raises_error_502(self) -> None:
        """Security master identifier columns must not match multiple aliases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "security_master": "sec_ref.csv",
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(
                    {
                        "SEC": ["S1"],
                        "SECURITY_ID": ["S1"],
                        "SECURITY_NAME": ["Security One"],
                    }
                ).write_csv(directory / snapshot_name / "sec_ref.csv")
            path = _write_yaml(directory, configuration)
            specification = PerformanceComparisonSpecification(path)

            with self.assertRaises(PpaError) as context:
                SecurityMasterLoader(specification).load("a")

            self.assertTrue(str(context.exception).startswith("Error 502"))
            self.assertIn("Ambiguous security master", str(context.exception))


if __name__ == "__main__":
    unittest.main()
