"""Tests for loading normalized FX rate comparison sources."""

# Python imports
from pathlib import Path
import tempfile
import unittest

# Third-party imports
import polars as pl
import yaml

# Test imports
from tests import test_utilities as test_util

# Project imports
from ppar.errors import PpaError
from ppar.audit import (
    FxRatesLoader,
    AuditSpecification,
)
from ppar.audit import schema as pc_cols

_BASELINE_COMPARISON_PATH = Path("tests/data/axys/validation/ppar_audit.yaml")


def _write_yaml(directory: Path, contents: object) -> Path:
    """Write comparison YAML contents and return the path."""
    path = directory / "ppar_audit.yaml"
    test_util.write_audit_test_yaml(path, contents)
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
        "fx_rate_impact_methods": {
            "fx_rate": {"method": "evidence_only"}
        },
    }


class TestFxRatesLoader(unittest.TestCase):
    """Verify normalized FX rate loading for snapshots."""

    def test_load_baseline_snapshot_a_fx_rates(self) -> None:
        """FX rate rows load with normalized internal columns."""
        specification = AuditSpecification(_BASELINE_COMPARISON_PATH)
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
            specification = AuditSpecification(path)

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
            specification = AuditSpecification(path)

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
            specification = AuditSpecification(path)

            with self.assertRaises(PpaError) as context:
                FxRatesLoader(specification).load("a")

            self.assertTrue(str(context.exception).startswith("Error 502"))
            self.assertIn("fx_rate", str(context.exception))

    def test_explicit_schema_selects_one_fx_currency_heading(self) -> None:
        """A generated explicit mapping selects one source heading."""
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
            schema = yaml.safe_load(
                (directory / "source_column_mappings.yaml").read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                schema["fx_rates_columns"]["from_currency"],
                "FROM_CURRENCY",
            )

    def test_nonpositive_rate_raises_error_502(self) -> None:
        """FX rates must be finite and strictly positive."""
        for invalid_rate in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(invalid_rate=invalid_rate):
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
                                "FX_RATE": [invalid_rate],
                            }
                        ).write_csv(directory / snapshot_name / "fx_rates.csv")
                    path = _write_yaml(directory, configuration)
                    specification = AuditSpecification(path)

                    with self.assertRaises(PpaError) as context:
                        FxRatesLoader(specification).load("a")

                    message = str(context.exception)
                    self.assertTrue(message.startswith("Error 502"))
                    self.assertIn("finite positive rates", message)

    def test_blank_currency_raises_error_502(self) -> None:
        """FX pair currencies must be present and nonblank."""
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
                        "FROM_CCY": [" "],
                        "TO_CCY": ["USD"],
                        "RATE_DATE": ["2025-01-31"],
                        "FX_RATE": [1.1],
                    }
                ).write_csv(directory / snapshot_name / "fx_rates.csv")
            path = _write_yaml(directory, configuration)
            specification = AuditSpecification(path)

            with self.assertRaises(PpaError) as context:
                FxRatesLoader(specification).load("a")

            message = str(context.exception)
            self.assertTrue(message.startswith("Error 502"))
            self.assertIn("from_currency", message)
            self.assertIn("blank value", message)

    def test_duplicate_pair_date_source_type_raises_error_112(self) -> None:
        """Duplicate normalized FX identities are rejected during loading."""
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
                        "FROM_CCY": ["EUR", "EUR"],
                        "TO_CCY": ["USD", "USD"],
                        "RATE_DATE": ["2025-01-31", "2025-01-31"],
                        "FX_RATE": [1.1, 1.2],
                        "SOURCE": ["CLIENT", "CLIENT"],
                        "RATE_TYPE": ["SPOT", "SPOT"],
                    }
                ).write_csv(directory / snapshot_name / "fx_rates.csv")
            path = _write_yaml(directory, configuration)
            specification = AuditSpecification(path)

            with self.assertRaises(PpaError) as context:
                FxRatesLoader(specification).load("a")

            message = str(context.exception)
            self.assertTrue(message.startswith("Error 112"))
            self.assertIn("duplicate rows", message)

    def test_distinct_rate_sources_can_share_pair_and_date(self) -> None:
        """Source provenance distinguishes otherwise identical FX keys."""
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
                        "FROM_CCY": ["EUR", "EUR"],
                        "TO_CCY": ["USD", "USD"],
                        "RATE_DATE": ["2025-01-31", "2025-01-31"],
                        "FX_RATE": [1.1, 1.2],
                        "SOURCE": ["CLIENT_A", "CLIENT_B"],
                        "RATE_TYPE": ["SPOT", "SPOT"],
                    }
                ).write_csv(directory / snapshot_name / "fx_rates.csv")
            path = _write_yaml(directory, configuration)
            specification = AuditSpecification(path)

            frame = FxRatesLoader(specification).load("a")

            assert frame is not None
            self.assertEqual(frame.height, 2)


if __name__ == "__main__":
    unittest.main()
