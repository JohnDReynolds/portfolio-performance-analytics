"""Tests for loading normalized security performance comparison sources."""

# Python imports
from pathlib import Path
import tempfile
import unittest

# Third-party imports
import polars as pl
import yaml

# Project imports
from ppar.errors import PpaError
from ppar.audit import (
    AuditSpecification,
    SecurityPerformanceLoader,
)
from ppar.audit import schema as pc_cols

_BASELINE_COMPARISON_PATH = Path("tests/data/axys/validation/ppar_audit.yaml")
_RESTATEMENT_COMPARISON_PATH = Path(
    "tests/data/axys/validation/ppar_audit_restatement.yaml"
)


def _write_yaml(directory: Path, contents: object) -> Path:
    """Write comparison YAML contents and return the path."""
    path = directory / "ppar_audit.yaml"
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
        "comparison": {"level": "portfolio"},
        "snapshots": {
            "a": {"path": "snapshot_a"},
            "b": {"path": "snapshot_b"},
        },
        "files": {"portfolio_performance": "portperf.csv"},
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


class TestSecurityPerformanceLoader(unittest.TestCase):
    """Verify normalized security performance loading for snapshots."""

    def test_load_baseline_snapshot_a_security_performance(self) -> None:
        """Security performance rows load with normalized internal columns."""
        specification = AuditSpecification(_BASELINE_COMPARISON_PATH)
        frame = SecurityPerformanceLoader(specification).load("a")
        assert frame is not None

        self.assertTrue(
            set(pc_cols.SECURITY_PERFORMANCE_REQUIRED_COLUMNS).issubset(frame.columns)
        )
        self.assertIn(pc_cols.WEIGHT, frame.columns)
        self.assertEqual(frame.schema[pc_cols.FROM_DATE], pl.Date)

        target_row = frame.filter(
            (pl.col(pc_cols.PORTFOLIO_ID) == "PORT_A")
            & (pl.col(pc_cols.SECURITY_ID) == "AAPL")
            & (pl.col(pc_cols.FROM_DATE) == pl.date(2025, 5, 30))
        ).row(0, named=True)
        self.assertEqual(target_row[pc_cols.SECURITY_RETURN], 0.04234740)

    def test_restatement_snapshot_b_loads_changed_security_return(self) -> None:
        """The restatement fixture exposes controlled security changes."""
        specification = AuditSpecification(_RESTATEMENT_COMPARISON_PATH)
        frame = SecurityPerformanceLoader(specification).load("b")
        assert frame is not None

        target_row = frame.filter(
            (pl.col(pc_cols.PORTFOLIO_ID) == "PORT_A")
            & (pl.col(pc_cols.SECURITY_ID) == "AAPL")
            & (pl.col(pc_cols.FROM_DATE) == pl.date(2025, 5, 30))
        ).row(0, named=True)
        self.assertEqual(target_row[pc_cols.SECURITY_RETURN], 0.05234740)
        self.assertEqual(target_row[pc_cols.WEIGHT], 0.05419463)

    def test_omitted_security_performance_returns_none(self) -> None:
        """Security performance is optional when omitted from YAML."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            path = _write_yaml(directory, _minimal_specification(directory))
            specification = AuditSpecification(path)

            self.assertIsNone(SecurityPerformanceLoader(specification).load("a"))

    def test_missing_optional_security_performance_returns_none(self) -> None:
        """Missing optional security performance files do not block loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "security_performance": "missing_secperf.csv",
            }
            path = _write_yaml(directory, configuration)
            specification = AuditSpecification(path)

            self.assertIsNone(SecurityPerformanceLoader(specification).load("a"))

    def test_missing_required_column_raises_error_502(self) -> None:
        """Existing security performance files must contain required fields."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "security_performance": "secperf.csv",
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(
                    {
                        "PORTFOLIO_CODE": ["P1"],
                        "SECURITY_ID": ["S1"],
                        "FROM_DATE": ["2025-01-01"],
                        "THRU_DATE": ["2025-01-31"],
                    }
                ).write_csv(directory / snapshot_name / "secperf.csv")
            path = _write_yaml(directory, configuration)
            specification = AuditSpecification(path)

            with self.assertRaises(PpaError) as context:
                SecurityPerformanceLoader(specification).load("a")

            self.assertTrue(str(context.exception).startswith("Error 502"))
            self.assertIn("security_return", str(context.exception))

    def test_ambiguous_required_column_raises_error_502(self) -> None:
        """Security performance required columns must not match multiple aliases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "security_performance": "secperf.csv",
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(
                    {
                        "PORTFOLIO_CODE": ["P1"],
                        "SEC": ["S1"],
                        "SECURITY_ID": ["S1"],
                        "FROM_DATE": ["2025-01-01"],
                        "THRU_DATE": ["2025-01-31"],
                        "SEC_RETURN": [0.01],
                    }
                ).write_csv(directory / snapshot_name / "secperf.csv")
            path = _write_yaml(directory, configuration)
            specification = AuditSpecification(path)

            with self.assertRaises(PpaError) as context:
                SecurityPerformanceLoader(specification).load("a")

            self.assertTrue(str(context.exception).startswith("Error 502"))
            self.assertIn("Ambiguous security performance", str(context.exception))


if __name__ == "__main__":
    unittest.main()
