"""Tests for shared performance comparison source loading helpers."""

# Python imports
from pathlib import Path
import tempfile
import unittest
from unittest import mock

# Third-party imports
import polars as pl
import yaml

# Project imports
from ppar.errors import PpaError
from ppar.axys_apx.security_identity import (
    SecurityIdConstruction,
    security_id_construction,
    with_constructed_security_id,
)
from ppar.audit import aliases
from ppar.audit import schema as pc_cols
from ppar.audit import source_loader
from ppar.audit.portfolio_performance import (
    PortfolioPerformanceLoader,
)
from ppar.audit.specification import ComparisonSnapshot
from ppar.audit.specification import AuditSpecification


class TestSourceLoader(unittest.TestCase):
    """Verify shared CSV alias resolution behavior."""

    def test_source_frame_cache_reads_each_physical_csv_once(self) -> None:
        """One audit-run scope reuses a raw CSV across loader requests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            path = directory / "source.csv"
            pl.DataFrame({"SEC": ["S1"]}).write_csv(path)

            with mock.patch.object(
                source_loader.pl,
                "read_csv",
                wraps=source_loader.pl.read_csv,
            ) as read_csv:
                with source_loader.source_frame_cache():
                    frame = source_loader.read_mapped_csv(
                        path,
                        ("security_id",),
                        "test_dataset",
                        {"security_id": ("SEC",)},
                        {},
                        directory / "comparison.yaml",
                    )
                    for _ in range(2):
                        source_loader.read_mapped_csv(
                            path,
                            ("security_id",),
                            "test_dataset",
                            {"security_id": ("SEC",)},
                            {},
                            directory / "comparison.yaml",
                        )

                with source_loader.source_frame_cache():
                    source_loader.read_mapped_csv(
                        path,
                        ("security_id",),
                        "test_dataset",
                        {"security_id": ("SEC",)},
                        {},
                        directory / "comparison.yaml",
                    )

            self.assertEqual(read_csv.call_count, 2)
            self.assertEqual(frame.to_dicts(), [{"security_id": "S1"}])

    def test_financial_validation_cache_is_scoped_to_one_audit_run(self) -> None:
        """Successful financial validation is reused only inside one run scope."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = Path(temp_dir) / "comparison.yaml"
            self.assertFalse(
                source_loader.financial_validation_is_cached(specification_path)
            )

    def test_normalized_frame_cache_validates_once_per_snapshot(self) -> None:
        """One Audit scope reuses a fully validated normalized dataset."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            specification_path = _write_source_loader_specification(directory)
            specification = AuditSpecification(specification_path)

            with mock.patch.object(
                source_loader,
                "require_numeric_columns",
                wraps=source_loader.require_numeric_columns,
            ) as require_numeric:
                with source_loader.source_frame_cache():
                    first = PortfolioPerformanceLoader(specification).load("a")
                    second = PortfolioPerformanceLoader(specification).load("a")

                third = PortfolioPerformanceLoader(specification).load("a")

            self.assertIs(first, second)
            self.assertIsNot(first, third)
            self.assertEqual(require_numeric.call_count, 2)
            with source_loader.source_frame_cache():
                source_loader.cache_financial_validation(specification_path)
                self.assertTrue(
                    source_loader.financial_validation_is_cached(specification_path)
                )
            self.assertFalse(
                source_loader.financial_validation_is_cached(specification_path)
            )

    def test_schema_mapping_overrides_default_aliases(self) -> None:
        """Referenced schema mappings override common mappings and defaults."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            schema_path = directory / "axys_column_mappings.yaml"
            source_path = directory / "portperf.csv"
            schema_path.write_text(
                yaml.safe_dump(
                    {
                        "files": {
                            "portfolio_performance": {
                                "columns": {
                                    "portfolio_code": "ACCT_CODE",
                                    "portfolio_return": "GROSS_RETURN",
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            pl.DataFrame(
                {
                    "ACCT_CODE": ["PORT_A"],
                    "PORTFOLIO_CODE": ["SHOULD_NOT_WIN"],
                    "from_date": ["2025-05-01"],
                    "thru_date": ["2025-05-31"],
                    "GROSS_RETURN": [0.01],
                }
            ).write_csv(source_path)
            snapshot = ComparisonSnapshot(
                key="a",
                label="snapshot_a",
                path=directory,
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
                    {
                        "files": {
                            "portfolio_performance": {
                                "columns": {
                                    "portfolio_code": "COMMON_PORTFOLIO_CODE",
                                    "portfolio_return": "COMMON_PORTFOLIO_RETURN",
                                }
                            }
                        }
                    },
                ),
                aliases.PORTFOLIO_PERFORMANCE_OPTIONAL_ALIASES,
                directory / "comparison.yaml",
            )

            self.assertEqual(frame.get_column(pc_cols.PORTFOLIO_ID).to_list(), ["PORT_A"])
            self.assertEqual(
                frame.get_column(pc_cols.PORTFOLIO_RETURN).to_list(),
                [0.01],
            )

    def test_referenced_schema_rejects_unknown_mapping_sections(self) -> None:
        """A referenced schema cannot silently ignore unknown top-level keys."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            schema_path = directory / "legacy_schema.yaml"
            schema_path.write_text(
                yaml.safe_dump(
                    {
                        "portfolio_performance_columns": {
                            "portfolio_code": "ACCT_CODE"
                        }
                    }
                ),
                encoding="utf-8",
            )
            snapshot = ComparisonSnapshot(
                key="a",
                label="snapshot_a",
                path=directory,
                schema_path=schema_path,
            )

            with self.assertRaisesRegex(
                PpaError,
                "Referenced schema YAML has unsupported top-level keys",
            ):
                source_loader.aliases_with_schema_overrides(
                    pc_cols.PORTFOLIO_PERFORMANCE,
                    aliases.PORTFOLIO_PERFORMANCE_REQUIRED_ALIASES,
                    snapshot,
                    directory / "comparison.yaml",
                )

    def test_common_audit_mapping_overrides_default_aliases(self) -> None:
        """Nested Audit mappings apply when a snapshot has no schema path."""
        snapshot = ComparisonSnapshot(
            key="a",
            label="snapshot_a",
            path=Path("snapshot_a"),
            schema_path=None,
        )

        resolved_aliases = source_loader.aliases_with_schema_overrides(
            pc_cols.PORTFOLIO_PERFORMANCE,
            aliases.PORTFOLIO_PERFORMANCE_REQUIRED_ALIASES,
            snapshot,
            Path("comparison.yaml"),
            {
                "files": {
                    "portfolio_performance": {
                        "columns": {
                            "portfolio_code": "Portfolio Code",
                            "portfolio_return": "Portfolio Return",
                        }
                    }
                }
            },
        )

        self.assertEqual(
            resolved_aliases[pc_cols.PORTFOLIO_ID],
            ("Portfolio Code",),
        )
        self.assertEqual(
            resolved_aliases[pc_cols.PORTFOLIO_RETURN],
            ("Portfolio Return",),
        )

    def test_default_source_names_reject_legacy_uppercase_headings(self) -> None:
        """Audit does not guess vendor headings without a schema mapping."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            source_path = directory / "portperf.csv"
            pl.DataFrame(
                {
                    "PORTFOLIO_CODE": ["PORT_A"],
                    "FROM_DATE": ["2025-05-01"],
                    "THRU_DATE": ["2025-05-31"],
                    "PORT_RETURN": [0.01],
                }
            ).write_csv(source_path)

            with self.assertRaisesRegex(PpaError, "expected one of.*portfolio_id"):
                source_loader.read_mapped_csv(
                    source_path,
                    pc_cols.PORTFOLIO_PERFORMANCE_COLUMNS,
                    pc_cols.PORTFOLIO_PERFORMANCE,
                    aliases.PORTFOLIO_PERFORMANCE_REQUIRED_ALIASES,
                    aliases.PORTFOLIO_PERFORMANCE_OPTIONAL_ALIASES,
                    directory / "comparison.yaml",
                )

    def test_schema_mapping_supports_security_performance_section(self) -> None:
        """Referenced schema mappings support security performance overrides."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            schema_path = directory / "axys_column_mappings.yaml"
            schema_path.write_text(
                yaml.safe_dump(
                    {
                        "files": {
                            "security_performance": {
                                "columns": {
                                    "identifier": "CUSTOM_SEC",
                                    "security_return": "CUSTOM_RETURN",
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            snapshot = ComparisonSnapshot(
                key="a",
                label="snapshot_a",
                path=directory,
                schema_path=schema_path,
            )

            security_performance_aliases = source_loader.aliases_with_schema_overrides(
                pc_cols.SECURITY_PERFORMANCE,
                aliases.SECURITY_PERFORMANCE_REQUIRED_ALIASES,
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

    def test_schema_constructs_security_id_from_spaced_source_headers(self) -> None:
        """Audit constructs a type-first key from exact source column names."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            specification_path = _write_source_loader_specification(directory)
            schema_path = directory / "axys_column_mappings.yaml"
            schema_path.write_text(
                yaml.safe_dump(
                    {
                        "security_id": {
                            "components": ["security_type", "security_symbol"],
                        },
                        "files": {
                            "security_performance": {
                                "columns": {
                                    "portfolio_code": "Portfolio Code",
                                    "security_symbol": "Security Symbol",
                                    "security_type": "Security Type",
                                    "from_date": "From Date",
                                    "thru_date": "Thru Date",
                                    "security_return": "Security Return",
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            specification_values = yaml.safe_load(
                specification_path.read_text(encoding="utf-8")
            )
            assert isinstance(specification_values, dict)
            specification_values["snapshots"]["a"]["schema"] = str(schema_path)
            specification_path.write_text(
                yaml.safe_dump(specification_values),
                encoding="utf-8",
            )
            source_path = directory / "snapshot_a" / "secperf.csv"
            pl.DataFrame(
                {
                    "Portfolio Code": ["PORT_A", "PORT_A"],
                    "Security Symbol": ["AAPL", "AAPL"],
                    "Security Type": ["csus", "amus"],
                    "From Date": ["2025-05-01", "2025-05-01"],
                    "Thru Date": ["2025-05-31", "2025-05-31"],
                    "Security Return": [0.01, 0.02],
                }
            ).write_csv(source_path)
            specification = AuditSpecification(specification_path)

            frame = source_loader.read_schema_mapped_csv(
                source_path,
                pc_cols.SECURITY_PERFORMANCE_COLUMNS,
                pc_cols.SECURITY_PERFORMANCE,
                aliases.SECURITY_PERFORMANCE_REQUIRED_ALIASES,
                aliases.SECURITY_PERFORMANCE_OPTIONAL_ALIASES,
                specification,
                "a",
            )

            self.assertEqual(
                frame.get_column(pc_cols.SECURITY_ID).to_list(),
                ["csusAAPL", "amusAAPL"],
            )
            self.assertNotIn("security_symbol", frame.columns)
            self.assertNotIn("security_type", frame.columns)

    def test_compact_security_id_rejects_ambiguous_component_pairs(self) -> None:
        """Separator-free keys stop when distinct source pairs would collide."""
        frame = pl.DataFrame(
            {
                "Security Type": ["ab", "a"],
                "Security Symbol": ["c", "bc"],
            }
        )
        construction = SecurityIdConstruction(
            components=("security_type", "security_symbol"),
            source_columns=("Security Type", "Security Symbol"),
            separator="",
        )

        with self.assertRaisesRegex(PpaError, "ambiguous identifier 'abc'"):
            with_constructed_security_id(
                frame,
                construction,
                output_column="security_id",
                dataset_name="security_master",
                source_path="secmast.csv",
                error_message=lambda message: message,
            )

    def test_security_id_components_reject_source_column_captions(self) -> None:
        """Identity components use normalized mapping keys, not CSV captions."""
        with self.assertRaisesRegex(PpaError, "normalized field names"):
            security_id_construction(
                {
                    "security_id": {
                        "components": ["Security Type", "Security Symbol"],
                    }
                },
                "security_performance",
                lambda message: message,
            )

    def test_security_id_defaults_from_axys_apx_file_layout(self) -> None:
        """Mapped type and symbol fields imply the compact Axys/APX identity."""
        construction = security_id_construction(
            {
                "files": {
                    "security_performance": {
                        "columns": {
                            "security_symbol": "Security Symbol",
                            "security_type": "Security Type",
                        }
                    }
                }
            },
            "security_performance",
            lambda message: message,
        )

        self.assertIsNotNone(construction)
        assert construction is not None
        self.assertEqual(
            construction.components,
            ("security_type", "security_symbol"),
        )
        self.assertEqual(
            construction.source_columns,
            ("Security Type", "Security Symbol"),
        )
        self.assertEqual(construction.separator, "")

    def test_security_id_is_not_inferred_from_an_incomplete_layout(self) -> None:
        """Generic layouts still use their existing security_id field."""
        construction = security_id_construction(
            {
                "files": {
                    "security_performance": {
                        "columns": {"security_symbol": "Security Symbol"}
                    }
                }
            },
            "security_performance",
            lambda message: message,
        )

        self.assertIsNone(construction)

    def test_explicit_file_security_id_mapping_precedes_inferred_identity(self) -> None:
        """A direct source ID wins even when type and symbol are also mapped."""
        construction = security_id_construction(
            {
                "files": {
                    "security_performance": {
                        "columns": {
                            "security_id": "source_symbol",
                            "security_symbol": "source_symbol",
                            "security_type": "source_type",
                        }
                    }
                }
            },
            "security_performance",
            lambda message: message,
        )

        self.assertIsNone(construction)

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
                    "SOURCE_ALIAS": ["B"],
                }
            ).write_csv(path)

            with self.assertRaises(PpaError) as context:
                source_loader.read_mapped_csv(
                    path,
                    ("security_id", "source_column"),
                    "test_dataset",
                    {"security_id": ("SEC",)},
                    {"source_column": ("SOURCE_ALIAS", "SOURCE")},
                    directory / "comparison.yaml",
                )

            message = str(context.exception)
            self.assertTrue(message.startswith("Error 502"))
            self.assertIn("Ambiguous test dataset source columns", message)
            self.assertIn("source_column", message)

    def test_optional_file_path_returns_snapshot_specific_paths(self) -> None:
        """Optional file paths resolve to the requested snapshot side."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            specification_path = _write_source_loader_specification(directory)
            specification = AuditSpecification(specification_path)

            snapshot_a_path = source_loader.optional_file_path(
                specification,
                pc_cols.SPLITS,
                "a",
            )
            snapshot_b_path = source_loader.optional_file_path(
                specification,
                pc_cols.SPLITS,
                "b",
            )

            self.assertEqual(snapshot_a_path, directory / "snapshot_a" / "splits.csv")
            self.assertEqual(snapshot_b_path, directory / "snapshot_b" / "splits.csv")

    def test_optional_file_path_returns_none_for_omitted_dataset(self) -> None:
        """Omitted optional datasets return ``None``."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            specification_path = _write_source_loader_specification(directory)
            specification = AuditSpecification(specification_path)

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
            specification = AuditSpecification(specification_path)

            with self.assertRaises(PpaError) as context:
                source_loader.optional_file_path(
                    specification,
                    pc_cols.SPLITS,
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
            "portfolio_id,from_date,thru_date,portfolio_return\n"
            "PORT_A,2025-05-01,2025-05-31,0.01\n",
            encoding="utf-8",
        )

    specification = {
        "comparison": {"level": "portfolio"},
        "snapshots": {
            "a": {"path": "snapshot_a"},
            "b": {"path": "snapshot_b"},
        },
        "files": {
            "portfolio_performance": "portperf.csv",
            "splits": "splits.csv",
        },
        "extract_contract": {
            "enforce_ambiguous_axys_flows": True,
        },
        "tolerances": {
            "return": 0.000001,
            "market_value": 0.01,
            "quantity": 0.000001,
            "price": 0.000001,
            "split_factor": 0.00000001,
        },
    }
    specification_path = directory / "ppar_audit.yaml"
    specification_path.write_text(yaml.safe_dump(specification), encoding="utf-8")
    return specification_path


if __name__ == "__main__":
    unittest.main()
