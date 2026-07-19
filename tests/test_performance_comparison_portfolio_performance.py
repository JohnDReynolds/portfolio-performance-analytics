"""Tests for loading normalized portfolio performance comparison sources."""

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
    PortfolioPerformanceLoader,
)
from ppar.audit import schema as pc_cols

_BASELINE_COMPARISON_PATH = Path("tests/data/axys/validation/ppar_audit.yaml")
_RESTATEMENT_COMPARISON_PATH = Path(
    "tests/data/axys/validation/ppar_audit_restatement.yaml"
)


def _write_yaml(directory: Path, contents: object) -> Path:
    """Write comparison YAML contents and return the path."""
    if isinstance(contents, dict):
        contents = {
            "comparison": {"level": "portfolio"},
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
            **contents,
        }
    path = directory / "ppar_audit.yaml"
    path.write_text(yaml.safe_dump(contents), encoding="utf-8")
    return path


class TestPortfolioPerformanceLoader(unittest.TestCase):
    """Verify normalized portfolio performance loading for snapshots."""

    def test_load_baseline_snapshot_a_portfolio_performance(self) -> None:
        """Portfolio performance rows load with normalized internal columns."""
        specification = AuditSpecification(_BASELINE_COMPARISON_PATH)
        frame = PortfolioPerformanceLoader(specification).load("a")

        self.assertTrue(
            set(pc_cols.PORTFOLIO_PERFORMANCE_REQUIRED_COLUMNS).issubset(frame.columns)
        )
        self.assertIn(pc_cols.BEGIN_MARKET_VALUE, frame.columns)
        self.assertEqual(frame.schema[pc_cols.FROM_DATE], pl.Date)

        target_row = frame.filter(
            (pl.col(pc_cols.PORTFOLIO_ID) == "PORT_A")
            & (pl.col(pc_cols.FROM_DATE) == pl.date(2025, 5, 30))
        ).row(0, named=True)
        self.assertEqual(target_row[pc_cols.PORTFOLIO_RETURN], 0.00853281)

    def test_restatement_snapshot_b_loads_changed_portfolio_return(self) -> None:
        """The restatement fixture exposes a controlled portfolio return change."""
        specification = AuditSpecification(_RESTATEMENT_COMPARISON_PATH)
        frame = PortfolioPerformanceLoader(specification).load("b")

        target_row = frame.filter(
            (pl.col(pc_cols.PORTFOLIO_ID) == "PORT_A")
            & (pl.col(pc_cols.FROM_DATE) == pl.date(2025, 5, 30))
        ).row(0, named=True)
        self.assertEqual(target_row[pc_cols.PORTFOLIO_RETURN], 0.00903281)
        self.assertEqual(target_row[pc_cols.END_MARKET_VALUE], 996300.47)

    def test_missing_required_column_raises_error_502(self) -> None:
        """Portfolio performance cannot load without required normalized fields."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                snapshot_path = directory / snapshot_name
                snapshot_path.mkdir()
                pl.DataFrame(
                    {
                        "PORTFOLIO_CODE": ["P1"],
                        "FROM_DATE": ["2025-01-01"],
                        "THRU_DATE": ["2025-01-31"],
                    }
                ).write_csv(snapshot_path / "portperf.csv")
            path = _write_yaml(
                directory,
                {
                    "snapshots": {
                        "a": {"path": "snapshot_a"},
                        "b": {"path": "snapshot_b"},
                    },
                    "files": {"portfolio_performance": "portperf.csv"},
                },
            )
            specification = AuditSpecification(path)

            with self.assertRaises(PpaError) as context:
                PortfolioPerformanceLoader(specification).load("a")

            self.assertTrue(str(context.exception).startswith("Error 502"))
            self.assertIn("portfolio_return", str(context.exception))

    def test_ambiguous_required_column_raises_error_502(self) -> None:
        """Required columns must not match multiple source aliases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                snapshot_path = directory / snapshot_name
                snapshot_path.mkdir()
                pl.DataFrame(
                    {
                        "PORT": ["P1"],
                        "PORTFOLIO_CODE": ["P1"],
                        "FROM_DATE": ["2025-01-01"],
                        "THRU_DATE": ["2025-01-31"],
                        "PORT_RETURN": [0.01],
                    }
                ).write_csv(snapshot_path / "portperf.csv")
            path = _write_yaml(
                directory,
                {
                    "snapshots": {
                        "a": {"path": "snapshot_a"},
                        "b": {"path": "snapshot_b"},
                    },
                    "files": {"portfolio_performance": "portperf.csv"},
                },
            )
            specification = AuditSpecification(path)

            with self.assertRaises(PpaError) as context:
                PortfolioPerformanceLoader(specification).load("a")

            self.assertTrue(str(context.exception).startswith("Error 502"))
            self.assertIn("Ambiguous portfolio performance", str(context.exception))

    def test_nonnumeric_return_raises_error_502(self) -> None:
        """Malformed portfolio numeric values fail with field-level context."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                snapshot_path = directory / snapshot_name
                snapshot_path.mkdir()
                (snapshot_path / "portperf.csv").write_text(
                    "PORTFOLIO_CODE,FROM_DATE,THRU_DATE,PORT_RETURN\n"
                    "P1,2025-01-01,2025-01-31,N/A\n",
                    encoding="utf-8",
                )
            path = _write_yaml(
                directory,
                {
                    "snapshots": {
                        "a": {"path": "snapshot_a"},
                        "b": {"path": "snapshot_b"},
                    },
                    "files": {"portfolio_performance": "portperf.csv"},
                },
            )
            specification = AuditSpecification(path)

            with self.assertRaises(PpaError) as context:
                PortfolioPerformanceLoader(specification).load("a")

            message = str(context.exception)
            self.assertTrue(message.startswith("Error 502"))
            self.assertIn("portfolio_performance", message)
            self.assertIn("portfolio_return", message)
            self.assertIn("N/A", message)


if __name__ == "__main__":
    unittest.main()
