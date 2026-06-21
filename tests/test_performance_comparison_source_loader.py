"""Tests for shared performance comparison source loading helpers."""

# Python imports
from pathlib import Path
import tempfile
import unittest

# Third-party imports
import polars as pl
import yaml

# Project imports
from ppar.errors import PpaError
from ppar.performance_comparison import aliases
from ppar.performance_comparison import schema as pc_cols
from ppar.performance_comparison import source_loader
from ppar.performance_comparison.specification import ComparisonSnapshot
from ppar.performance_comparison.specification import PerformanceComparisonSpecification


class TestSourceLoader(unittest.TestCase):
    """Verify shared CSV alias resolution behavior."""

    def test_schema_mapping_overrides_default_aliases(self) -> None:
        """Referenced schema mappings are honored before built-in defaults."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            schema_path = directory / "axys_column_mappings.yaml"
            source_path = directory / "portperf.csv"
            schema_path.write_text(
                yaml.safe_dump(
                    {
                        "portfolio_performance_columns": {
                            "portfolio_code": "ACCT_CODE",
                            "portfolio_return": "GROSS_RETURN",
                        }
                    }
                ),
                encoding="utf-8",
            )
            pl.DataFrame(
                {
                    "ACCT_CODE": ["PORT_A"],
                    "PORTFOLIO_CODE": ["SHOULD_NOT_WIN"],
                    "FROM_DATE": ["2025-05-01"],
                    "THRU_DATE": ["2025-05-31"],
                    "GROSS_RETURN": [0.01],
                }
            ).write_csv(source_path)
            snapshot = ComparisonSnapshot(
                key="a",
                label="snapshot_a",
                path=directory,
                vendor="axys",
                schema_path=schema_path,
            )

            frame = source_loader.read_mapped_csv(
                source_path,
                pc_cols.PORTFOLIO_PERFORMANCE_COLUMNS,
                pc_cols.PORTFOLIO_PERFORMANCE,
                source_loader.aliases_with_schema_overrides(
                    pc_cols.PORTFOLIO_PERFORMANCE,
                    aliases.PORTFOLIO_PERFORMANCE_REQUIRED_ALIASES,
                    snapshot,
                    directory / "comparison.yaml",
                ),
                aliases.PORTFOLIO_PERFORMANCE_OPTIONAL_ALIASES,
                directory / "comparison.yaml",
            )

            self.assertEqual(frame.get_column(pc_cols.PORTFOLIO_ID).to_list(), ["PORT_A"])
            self.assertEqual(
                frame.get_column(pc_cols.PORTFOLIO_RETURN).to_list(),
                [0.01],
            )

    def test_schema_mapping_supports_security_sections(self) -> None:
        """Referenced schema mappings support security performance and master."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            schema_path = directory / "axys_column_mappings.yaml"
            schema_path.write_text(
                yaml.safe_dump(
                    {
                        "security_performance_columns": {
                            "identifier": "CUSTOM_SEC",
                            "return": "CUSTOM_RETURN",
                        },
                        "security_master_columns": {
                            "identifier_column": "CUSTOM_ID",
                            "name_column": "CUSTOM_NAME",
                        },
                    }
                ),
                encoding="utf-8",
            )
            snapshot = ComparisonSnapshot(
                key="a",
                label="snapshot_a",
                path=directory,
                vendor="axys",
                schema_path=schema_path,
            )

            security_performance_aliases = source_loader.aliases_with_schema_overrides(
                pc_cols.SECURITY_PERFORMANCE,
                aliases.SECURITY_PERFORMANCE_REQUIRED_ALIASES,
                snapshot,
                directory / "comparison.yaml",
            )
            security_master_aliases = source_loader.aliases_with_schema_overrides(
                pc_cols.SECURITY_MASTER,
                aliases.SECURITY_MASTER_OPTIONAL_ALIASES,
                snapshot,
                directory / "comparison.yaml",
            )

            self.assertEqual(
                security_performance_aliases[pc_cols.SECURITY_ID],
                ("CUSTOM_SEC",),
            )
            self.assertEqual(
                security_performance_aliases[pc_cols.SECURITY_RETURN],
                ("CUSTOM_RETURN",),
            )
            self.assertEqual(
                security_master_aliases[pc_cols.SECURITY_NAME],
                ("CUSTOM_NAME",),
            )

    def test_read_mapped_csv_raises_error_502_for_missing_required_column(self) -> None:
        """Missing required aliases fail with a clear source resolution error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            path = directory / "source.csv"
            pl.DataFrame({"OTHER": ["x"]}).write_csv(path)

            with self.assertRaises(PpaError) as context:
                source_loader.read_mapped_csv(
                    path,
                    ("security_id",),
                    "test_dataset",
                    {"security_id": ("SECURITY_ID", "SEC")},
                    {},
                    directory / "comparison.yaml",
                )

            message = str(context.exception)
            self.assertTrue(message.startswith("Error 502"))
            self.assertIn("Source column resolution failed", message)
            self.assertIn("Missing", message)
            self.assertIn("security_id", message)

    def test_read_mapped_csv_raises_error_502_for_duplicate_optional_aliases(self) -> None:
        """Multiple aliases for one optional normalized column are ambiguous."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            path = directory / "source.csv"
            pl.DataFrame(
                {
                    "SEC": ["S1"],
                    "SOURCE": ["A"],
                    "PRICE_SOURCE": ["B"],
                }
            ).write_csv(path)

            with self.assertRaises(PpaError) as context:
                source_loader.read_mapped_csv(
                    path,
                    ("security_id", "price_source"),
                    "test_dataset",
                    {"security_id": ("SEC",)},
                    {"price_source": ("PRICE_SOURCE", "SOURCE")},
                    directory / "comparison.yaml",
                )

            message = str(context.exception)
            self.assertTrue(message.startswith("Error 502"))
            self.assertIn("Ambiguous test dataset source columns", message)
            self.assertIn("price_source", message)

    def test_optional_file_path_returns_snapshot_specific_paths(self) -> None:
        """Optional file paths resolve to the requested snapshot side."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            specification_path = _write_source_loader_specification(directory)
            specification = PerformanceComparisonSpecification(specification_path)

            snapshot_a_path = source_loader.optional_file_path(
                specification,
                pc_cols.PRICES,
                "a",
            )
            snapshot_b_path = source_loader.optional_file_path(
                specification,
                pc_cols.PRICES,
                "b",
            )

            self.assertEqual(snapshot_a_path, directory / "snapshot_a" / "prices.csv")
            self.assertEqual(snapshot_b_path, directory / "snapshot_b" / "prices.csv")

    def test_optional_file_path_returns_none_for_omitted_dataset(self) -> None:
        """Omitted optional datasets return ``None``."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            specification_path = _write_source_loader_specification(directory)
            specification = PerformanceComparisonSpecification(specification_path)

            path = source_loader.optional_file_path(
                specification,
                pc_cols.TRANSACTIONS,
                "a",
            )

            self.assertIsNone(path)

    def test_optional_file_path_rejects_unknown_snapshot_key(self) -> None:
        """Unknown neutral snapshot keys fail clearly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            specification_path = _write_source_loader_specification(directory)
            specification = PerformanceComparisonSpecification(specification_path)

            with self.assertRaises(PpaError) as context:
                source_loader.optional_file_path(
                    specification,
                    pc_cols.PRICES,
                    "c",
                )

            self.assertTrue(str(context.exception).startswith("Error 999"))
            self.assertIn("Unknown snapshot key", str(context.exception))


def _write_source_loader_specification(directory: Path) -> Path:
    """Write a minimal source-loader comparison specification."""
    for snapshot_name in ("snapshot_a", "snapshot_b"):
        snapshot_path = directory / snapshot_name
        snapshot_path.mkdir()
        (snapshot_path / "portperf.csv").write_text(
            "PORTFOLIO_CODE,FROM_DATE,THRU_DATE,PORT_RETURN\n"
            "PORT_A,2025-05-01,2025-05-31,0.01\n",
            encoding="utf-8",
        )

    specification = {
        "snapshots": {
            "a": {"path": "snapshot_a"},
            "b": {"path": "snapshot_b"},
        },
        "files": {
            "portfolio_performance": "portperf.csv",
            "prices": "prices.csv",
        },
    }
    specification_path = directory / "ppar_performance_comparison.yaml"
    specification_path.write_text(yaml.safe_dump(specification), encoding="utf-8")
    return specification_path


if __name__ == "__main__":
    unittest.main()
