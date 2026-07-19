"""Tests for performance comparison YAML parsing and path resolution."""

# Python imports
from pathlib import Path
import tempfile
import unittest

# Third-party imports
import yaml

# Project imports
from ppar.errors import PpaError
from ppar.audit import AuditSpecification

_AXYS_COMPARISON_PATH = Path("tests/data/axys/validation/ppar_audit.yaml")
_AXYS_SNAPSHOT_PATH = Path("tests/data/axys/snapshots")
_TEST_AXYS_SCHEMA_PATH = Path("tests/data/axys/axys_column_mappings.yaml")


def _write_yaml(directory: Path, contents: object) -> Path:
    """Write comparison YAML contents and return the path."""
    path = directory / "ppar_audit.yaml"
    path.write_text(yaml.safe_dump(contents), encoding="utf-8")
    return path


def _write_yaml_text(directory: Path, contents: str) -> Path:
    """Write raw comparison YAML text and return the path."""
    path = directory / "ppar_audit.yaml"
    path.write_text(contents, encoding="utf-8")
    return path


def _minimal_specification(directory: Path) -> dict[str, object]:
    """Return a minimal valid comparison specification with fixture files."""
    for snapshot_name in ("snapshot_a", "snapshot_b"):
        snapshot_path = directory / snapshot_name
        snapshot_path.mkdir()
        (snapshot_path / "portperf.csv").write_text("header\n", encoding="utf-8")

    return {
        "snapshots": {
            "a": {"path": "snapshot_a", "schema": "schema.yaml"},
            "b": {"path": "snapshot_b", "schema": "schema.yaml"},
        },
        "files": {"portfolio_performance": "portperf.csv"},
    }


class TestAuditSpecification(unittest.TestCase):
    """Verify comparison specification parsing and file preflight behavior."""

    def test_fixture_comparison_paths_are_resolved(self) -> None:
        """Committed baseline fixture resolves snapshots, schemas, and files."""
        specification = AuditSpecification(_AXYS_COMPARISON_PATH)

        self.assertEqual(specification.snapshot_a.label, "axys_a")
        self.assertEqual(specification.snapshot_b.label, "axys_b")
        self.assertEqual(
            specification.snapshot_a.path.resolve(),
            (_AXYS_SNAPSHOT_PATH / "axys_a").resolve(),
        )
        snapshot_b_schema_path = specification.snapshot_b.schema_path
        if snapshot_b_schema_path is None:
            raise AssertionError("Expected snapshot B schema path to be resolved.")
        self.assertEqual(
            snapshot_b_schema_path.resolve(),
            _TEST_AXYS_SCHEMA_PATH.resolve(),
        )

        portfolio_file = specification.files["portfolio_performance"]
        self.assertTrue(portfolio_file.required)
        self.assertEqual(portfolio_file.relative_path, Path("portperf.csv"))
        self.assertEqual(
            portfolio_file.snapshot_a_path.resolve(),
            (_AXYS_SNAPSHOT_PATH / "axys_a" / "portperf.csv").resolve(),
        )
        self.assertEqual(
            specification.files["transactions"].snapshot_b_path.resolve(),
            (_AXYS_SNAPSHOT_PATH / "axys_b" / "transactions.csv").resolve(),
        )

    def test_optional_missing_file_does_not_raise(self) -> None:
        """Missing optional files do not block specification loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "security_performance": "missing_secperf.csv",
            }
            path = _write_yaml(directory, configuration)

            specification = AuditSpecification(path)

            self.assertFalse(specification.files["security_performance"].required)
            self.assertEqual(
                specification.files["security_performance"].snapshot_a_path,
                directory / "snapshot_a" / "missing_secperf.csv",
            )

    def test_required_optional_missing_file_raises_error_802(self) -> None:
        """Optional files marked required are validated during preflight."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "transactions": {
                    "path": "missing_transactions.csv",
                    "required": True,
                },
            }
            path = _write_yaml(directory, configuration)

            with self.assertRaises(PpaError) as context:
                AuditSpecification(path)

            self.assertTrue(str(context.exception).startswith("Error 802"))
            self.assertIn("files.transactions", str(context.exception))
            self.assertIn("snapshot a", str(context.exception))
            self.assertIn("missing_transactions.csv", str(context.exception))

    def test_reconstruction_source_files_are_required(self) -> None:
        """Return reconstruction source files cannot be silently optional."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["portfolio_return_reconstruction"] = {
                "method": "modified_dietz",
                "beginning_value_source": "holdings",
                "ending_value_source": "holdings",
                "flow_source": "transactions",
                "flow_timing": "transaction_date",
                "day_count": "actual_days",
                "inclusion_rule": "beginning_of_day",
                "flow_categories": ["external_flow"],
                "income_categories": ["income"],
                "return_basis": "net",
                "sign_convention": "signed_amount",
            }
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "holdings": "missing_holdings.csv",
                "transactions": "missing_transactions.csv",
            }
            path = _write_yaml(directory, configuration)

            with self.assertRaises(PpaError) as context:
                AuditSpecification(path)

            self.assertTrue(str(context.exception).startswith("Error 802"))
            self.assertIn("files.holdings", str(context.exception))
            self.assertIn("snapshot a", str(context.exception))
            self.assertIn("missing_holdings.csv", str(context.exception))

    def test_reconstruction_source_files_must_not_opt_out_of_required(self) -> None:
        """Reconstruction-required files cannot override required to false."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["portfolio_return_reconstruction"] = {
                "method": "modified_dietz",
                "beginning_value_source": "holdings",
                "ending_value_source": "holdings",
                "flow_source": "transactions",
                "flow_timing": "transaction_date",
                "day_count": "actual_days",
                "inclusion_rule": "beginning_of_day",
                "flow_categories": ["external_flow"],
                "income_categories": ["income"],
                "return_basis": "net",
                "sign_convention": "signed_amount",
            }
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "holdings": {"path": "holdings.csv", "required": False},
                "transactions": "transactions.csv",
            }
            path = _write_yaml(directory, configuration)

            with self.assertRaisesRegex(PpaError, "required by the comparison contract"):
                AuditSpecification(path)

    def test_duplicate_yaml_section_key_raises(self) -> None:
        """Duplicate YAML sections fail instead of silently overriding values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            _minimal_specification(directory)
            path = _write_yaml_text(
                directory,
                """
snapshots:
  a:
    path: snapshot_a
  b:
    path: snapshot_b
files:
  portfolio_performance: portperf.csv
transaction_impact_methods:
  performance:
    method: evidence_only
transaction_impact_methods:
  performance:
    method: transaction_amount_delta_over_return_denominator
""",
            )

            with self.assertRaisesRegex(PpaError, "duplicate YAML key"):
                AuditSpecification(path)

    def test_duplicate_yaml_method_key_raises(self) -> None:
        """Duplicate method keys inside one semantic slot fail fast."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            _minimal_specification(directory)
            path = _write_yaml_text(
                directory,
                """
snapshots:
  a:
    path: snapshot_a
  b:
    path: snapshot_b
files:
  portfolio_performance: portperf.csv
transaction_impact_methods:
  performance:
    method: evidence_only
    method: transaction_amount_delta_over_return_denominator
""",
            )

            with self.assertRaisesRegex(PpaError, "duplicate YAML key"):
                AuditSpecification(path)

    def test_simple_dietz_reconstruction_rejects_timed_flow_keys(self) -> None:
        """Simple Dietz does not accept timing fields it cannot use."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["portfolio_return_reconstruction"] = {
                "method": "simple_dietz",
                "beginning_value_source": "holdings",
                "ending_value_source": "holdings",
                "flow_source": "transactions",
                "flow_timing": "transaction_date",
                "flow_categories": ["external_flow"],
                "income_categories": ["income"],
                "return_basis": "net",
                "sign_convention": "signed_amount",
            }
            path = _write_yaml(directory, configuration)

            with self.assertRaisesRegex(PpaError, "not valid for method simple_dietz"):
                AuditSpecification(path)

    def test_modified_simple_dietz_reconstruction_omits_timed_flow_keys(self) -> None:
        """Modified Simple Dietz accepts only fields used by its formula."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                snapshot_path = directory / snapshot_name
                (snapshot_path / "holdings.csv").write_text("header\n", encoding="utf-8")
                (snapshot_path / "transactions.csv").write_text(
                    "header\n",
                    encoding="utf-8",
                )
            configuration["portfolio_return_reconstruction"] = {
                "method": "modified_simple_dietz",
                "beginning_value_source": "holdings",
                "ending_value_source": "holdings",
                "flow_source": "transactions",
                "flow_categories": ["external_flow"],
                "income_categories": ["income"],
                "return_basis": "net",
                "sign_convention": "signed_amount",
            }
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "holdings": "holdings.csv",
                "transactions": "transactions.csv",
            }
            path = _write_yaml(directory, configuration)

            specification = AuditSpecification(path)

            reconstruction = specification.portfolio_return_reconstruction
            if reconstruction is None:
                raise AssertionError("Expected reconstruction settings.")
            self.assertEqual(reconstruction.method, "modified_simple_dietz")
            self.assertIsNone(reconstruction.flow_timing)
            self.assertIsNone(reconstruction.day_count)
            self.assertIsNone(reconstruction.inclusion_rule)

    def test_modified_dietz_reconstruction_requires_timed_flow_keys(self) -> None:
        """Modified Dietz requires the timing fields that affect weighting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["portfolio_return_reconstruction"] = {
                "method": "modified_dietz",
                "beginning_value_source": "holdings",
                "ending_value_source": "holdings",
                "flow_source": "transactions",
                "flow_categories": ["external_flow"],
                "income_categories": ["income"],
                "return_basis": "net",
                "sign_convention": "signed_amount",
            }
            path = _write_yaml(directory, configuration)

            with self.assertRaisesRegex(PpaError, "required keys for method modified_dietz"):
                AuditSpecification(path)

    def test_reconstruction_rejects_unknown_keys(self) -> None:
        """Unknown reconstruction keys fail instead of being ignored."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["portfolio_return_reconstruction"] = {
                "method": "simple_dietz",
                "beginning_value_source": "holdings",
                "ending_value_source": "holdings",
                "flow_source": "transactions",
                "flow_categories": ["external_flow"],
                "income_categories": ["income"],
                "return_basis": "net",
                "sign_convention": "signed_amount",
                "surprise": "not-supported",
            }
            path = _write_yaml(directory, configuration)

            with self.assertRaisesRegex(PpaError, "unsupported keys: surprise"):
                AuditSpecification(path)

    def test_portfolio_performance_cannot_configure_required_flag(self) -> None:
        """Portfolio performance requiredness is structural, not configurable."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": {
                    "path": "portperf.csv",
                    "required": True,
                }
            }
            path = _write_yaml(directory, configuration)

            with self.assertRaises(PpaError) as context:
                AuditSpecification(path)

            self.assertTrue(str(context.exception).startswith("Error 504"))
            self.assertIn("must not specify required", str(context.exception))

    def test_missing_portfolio_performance_raises_error_504(self) -> None:
        """Portfolio performance must be listed in the files section."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {"security_performance": "secperf.csv"}
            path = _write_yaml(directory, configuration)

            with self.assertRaises(PpaError) as context:
                AuditSpecification(path)

            self.assertTrue(str(context.exception).startswith("Error 504"))
            self.assertIn("files.portfolio_performance is required", str(context.exception))

    def test_missing_snapshot_b_raises_error_504(self) -> None:
        """Snapshot definitions must include both neutral comparison sides."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            snapshots = configuration["snapshots"]
            assert isinstance(snapshots, dict)
            del snapshots["b"]
            path = _write_yaml(directory, configuration)

            with self.assertRaises(PpaError) as context:
                AuditSpecification(path)

            self.assertTrue(str(context.exception).startswith("Error 504"))
            self.assertIn("snapshots.b must be a mapping", str(context.exception))

    def test_non_mapping_yaml_root_raises_error_504(self) -> None:
        """The comparison YAML root must be a mapping."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _write_yaml(Path(temp_dir), ["not", "a", "mapping"])

            with self.assertRaises(PpaError) as context:
                AuditSpecification(path)

            self.assertTrue(str(context.exception).startswith("Error 504"))
            self.assertIn("YAML must be a dictionary", str(context.exception))

    def test_valid_data_issues_configuration_is_preserved(self) -> None:
        """Strict validation retains valid current values without normalization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            data_issues = {
                "enabled": True,
                "holdings_price_range": {
                    "enabled": True,
                    "only": {"holdings.security_id": [101, "ABC"]},
                    "exclude": {"portfolio": "TEST"},
                    "absolute_tolerance": 0.01,
                    "percent_tolerance": 0.5,
                },
                "portfolio_market_value_continuity": {
                    "absolute_tolerance": 0.01,
                },
                "holdings_nonpositive_price": {
                    "enabled": True,
                    "only": {"security_id": ["ABC"]},
                    "exclude": {"portfolio_id": "TEST"},
                },
                "large_price_variation": {
                    "enabled": True,
                    "rules": [
                        {
                            "rule_id": "common_stock_20_percent",
                            "only": {
                                "transactions.transaction_code": ["by", "sl"],
                            },
                            "minimum_calendar_days": 1,
                            "minimum_tolerance": 0.20,
                        }
                    ],
                },
                "deliver_in_original_cost_incomplete": {
                    "enabled": True,
                    "only": {
                        "transaction_code": ["ti", "si"],
                        "security_type": "csus",
                        "source_destination_type": "$pty",
                        "source_destination_symbol": "external_delivery",
                    },
                },
            }
            configuration["data_issues"] = data_issues
            path = _write_yaml(directory, configuration)

            specification = AuditSpecification(path)

        self.assertEqual(specification.values["data_issues"], data_issues)

    def test_retired_data_audit_checks_key_is_rejected(self) -> None:
        """The retired configuration key fails with an actionable replacement."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["data_audit_checks"] = {"enabled": False}
            path = _write_yaml(directory, configuration)

            with self.assertRaises(PpaError) as context:
                AuditSpecification(path)

        self.assertTrue(str(context.exception).startswith("Error 504"))
        self.assertIn(
            "data_audit_checks is no longer supported; use data_issues instead",
            str(context.exception),
        )

    def test_data_issues_configuration_rejects_malformed_or_unknown_values(self) -> None:
        """Every formerly fail-open Data Issues shape fails at an actionable path."""
        invalid_sections = (
            ([], "data_issues must be a mapping"),
            ({"unknown_issue": {}}, "unknown issue types or unsupported keys"),
            ({"enabled": "yes"}, "data_issues.enabled must be a Boolean"),
            (
                {"holdings_price_range": []},
                "data_issues.holdings_price_range must be a mapping",
            ),
            (
                {"holdings_price_range": {"surprise": True}},
                "data_issues.holdings_price_range has unsupported keys: surprise",
            ),
            (
                {"holdings_price_range": {"enabled": 1}},
                "data_issues.holdings_price_range.enabled must be a Boolean",
            ),
            (
                {"holdings_price_range": {"percent_tolerance": -0.01}},
                "data_issues.holdings_price_range.percent_tolerance must be a "
                "finite nonnegative number",
            ),
            (
                {"holdings_price_range": {"absolute_tolerance": float("inf")}},
                "data_issues.holdings_price_range.absolute_tolerance must be a "
                "finite nonnegative number",
            ),
            (
                {"holdings_price_range": {"only": ["security_id"]}},
                "data_issues.holdings_price_range.only must be a mapping",
            ),
            (
                {"holdings_price_range": {"only": {"price": 10}}},
                "data_issues.holdings_price_range.only.price is not a supported "
                "filter field",
            ),
            (
                {
                    "holdings_price_range": {
                        "only": {"security_reference.transaction_code": "by"}
                    }
                },
                "data_issues.holdings_price_range.only.security_reference."
                "transaction_code is not a supported filter field",
            ),
            (
                {"holdings_price_range": {"only": {"asset_class_code": "EQ"}}},
                "data_issues.holdings_price_range.only.asset_class_code is not a "
                "supported filter field",
            ),
            (
                {"holdings_price_range": {"exclude": {"security_id": {"ABC": 1}}}},
                "data_issues.holdings_price_range.exclude.security_id must be a "
                "scalar value",
            ),
            (
                {"duplicate_transactions": {"absolute_tolerance": 1}},
                "data_issues.duplicate_transactions has unsupported keys: "
                "absolute_tolerance",
            ),
            (
                {"holdings_nonpositive_price": {"enabled": True}},
                "data_issues.holdings_nonpositive_price.only must be a nonempty "
                "mapping when data_issues.holdings_nonpositive_price.enabled is true",
            ),
            (
                {
                    "holdings_nonpositive_price": {
                        "enabled": True,
                        "only": {},
                    }
                },
                "data_issues.holdings_nonpositive_price.only must be a nonempty "
                "mapping when data_issues.holdings_nonpositive_price.enabled is true",
            ),
            (
                {
                    "holdings_nonpositive_price": {
                        "enabled": True,
                        "only": {"security_id": "ABC"},
                        "absolute_tolerance": 0.01,
                    }
                },
                "data_issues.holdings_nonpositive_price has unsupported keys: "
                "absolute_tolerance",
            ),
            (
                {"holdings_stale_price": {"enabled": True}},
                "data_issues.holdings_stale_price.only must be a nonempty mapping "
                "when data_issues.holdings_stale_price.enabled is true",
            ),
            (
                {
                    "holdings_stale_price": {
                        "enabled": True,
                        "only": {"security_id": "ABC"},
                        "minimum_calendar_days": 28,
                    }
                },
                "data_issues.holdings_stale_price.only must include security_"
                "reference.security_type when data_issues.holdings_stale_price."
                "enabled is true",
            ),
            (
                {
                    "holdings_stale_price": {
                        "enabled": True,
                        "only": {"security_reference.security_type": "csus"},
                    }
                },
                "data_issues.holdings_stale_price.minimum_calendar_days is required "
                "when data_issues.holdings_stale_price.enabled is true",
            ),
            (
                {
                    "holdings_stale_price": {
                        "enabled": True,
                        "only": {"security_reference.security_type": "csus"},
                        "minimum_calendar_days": 0,
                    }
                },
                "data_issues.holdings_stale_price.minimum_calendar_days must be a "
                "positive integer",
            ),
            (
                {
                    "holdings_stale_price": {
                        "enabled": True,
                        "only": {"security_reference.security_type": "csus"},
                        "minimum_calendar_days": 28,
                        "percent_tolerance": 1,
                    }
                },
                "data_issues.holdings_stale_price has unsupported keys: "
                "percent_tolerance",
            ),
            (
                {"large_price_variation": {"enabled": True}},
                "data_issues.large_price_variation.rules is required when "
                "data_issues.large_price_variation.enabled is true",
            ),
            (
                {
                    "large_price_variation": {
                        "enabled": True,
                        "rules": {"rule_id": "common_stock"},
                    }
                },
                "data_issues.large_price_variation.rules must be a nonempty list",
            ),
            (
                {
                    "large_price_variation": {
                        "enabled": True,
                        "rules": ["common_stock"],
                    }
                },
                "data_issues.large_price_variation.rules[0] must be a mapping",
            ),
            (
                {
                    "large_price_variation": {
                        "enabled": True,
                        "rules": [{"rule_id": "Common Stock"}],
                    }
                },
                "data_issues.large_price_variation.rules[0].rule_id must be a "
                "lowercase snake-case identifier",
            ),
            (
                {
                    "large_price_variation": {
                        "enabled": True,
                        "rules": [
                            {"rule_id": "common_stock"},
                            {"rule_id": "common_stock"},
                        ],
                    }
                },
                "data_issues.large_price_variation.rules has duplicate rule_id "
                "'common_stock'",
            ),
            (
                {
                    "large_price_variation": {
                        "enabled": True,
                        "rules": [
                            {
                                "rule_id": "common_stock",
                                "minimum_calendar_days": 0,
                            }
                        ],
                    }
                },
                "data_issues.large_price_variation.rules[0].minimum_calendar_days "
                "must be a positive integer",
            ),
            (
                {
                    "large_price_variation": {
                        "enabled": True,
                        "rules": [
                            {
                                "rule_id": "common_stock",
                                "minimum_tolerance": True,
                            }
                        ],
                    }
                },
                "data_issues.large_price_variation.rules[0].minimum_tolerance "
                "must be a finite nonnegative number",
            ),
            (
                {
                    "large_price_variation": {
                        "enabled": True,
                        "rules": [
                            {
                                "rule_id": "common_stock",
                                "only": {"portfolio_performance.portfolio_id": "P1"},
                            }
                        ],
                    }
                },
                "data_issues.large_price_variation.rules[0].only.portfolio_"
                "performance.portfolio_id uses unsupported dataset namespace",
            ),
            (
                {
                    "large_price_variation": {
                        "enabled": True,
                        "rules": [
                            {
                                "rule_id": "common_stock",
                                "percent_tolerance": 20,
                            }
                        ],
                    }
                },
                "data_issues.large_price_variation.rules[0] has unsupported keys: "
                "percent_tolerance",
            ),
            (
                {"transactions_nonpositive_price": {"enabled": True}},
                "data_issues.transactions_nonpositive_price.only must be a nonempty "
                "mapping when data_issues.transactions_nonpositive_price.enabled "
                "is true",
            ),
            (
                {
                    "transactions_nonpositive_price": {
                        "enabled": True,
                        "only": {
                            "security_reference.security_type": "csus",
                        },
                    }
                },
                "data_issues.transactions_nonpositive_price.only must include "
                "transaction_code when data_issues.transactions_nonpositive_price."
                "enabled is true",
            ),
            (
                {
                    "transactions_nonpositive_price": {
                        "enabled": True,
                        "only": {"transactions.transaction_code": ["by", "sl"]},
                    }
                },
                "data_issues.transactions_nonpositive_price.only must include "
                "security_reference.asset_class_code or security_reference."
                "security_type when data_issues.transactions_nonpositive_price."
                "enabled is true",
            ),
            (
                {
                    "transactions_nonpositive_price": {
                        "enabled": True,
                        "only": {
                            "transaction_code": "by",
                            "security_reference.asset_class_code": "EQ",
                        },
                        "percent_tolerance": 1,
                    }
                },
                "data_issues.transactions_nonpositive_price has unsupported keys: "
                "percent_tolerance",
            ),
            (
                {"transaction_security_type_mismatch": {"enabled": True}},
                "data_issues.transaction_security_type_mismatch.only must be a "
                "nonempty mapping when data_issues.transaction_security_type_"
                "mismatch.enabled is true",
            ),
            (
                {
                    "transaction_security_type_mismatch": {
                        "enabled": True,
                        "only": {"security_id": "ABC"},
                    }
                },
                "data_issues.transaction_security_type_mismatch.only must include "
                "security_reference.security_type when data_issues.transaction_"
                "security_type_mismatch.enabled is true",
            ),
            (
                {
                    "transaction_security_type_mismatch": {
                        "enabled": True,
                        "only": {"security_reference.security_type": "csus"},
                        "absolute_tolerance": 0,
                    }
                },
                "data_issues.transaction_security_type_mismatch has unsupported "
                "keys: absolute_tolerance",
            ),
            (
                {"deliver_in_original_cost_incomplete": {"enabled": True}},
                "data_issues.deliver_in_original_cost_incomplete.only must be a "
                "nonempty mapping when data_issues.deliver_in_original_cost_"
                "incomplete.enabled is true",
            ),
            (
                {
                    "deliver_in_original_cost_incomplete": {
                        "enabled": True,
                        "only": {"transaction_code": "ti"},
                    }
                },
                "data_issues.deliver_in_original_cost_incomplete.only must include "
                "security_type, source_destination_symbol, source_destination_type",
            ),
            (
                {
                    "deliver_in_original_cost_incomplete": {
                        "enabled": True,
                        "only": {
                            "transaction_code": "ti",
                            "security_type": "csus",
                            "source_destination_type": "$pty",
                            "source_destination_symbol": "external_delivery",
                        },
                        "absolute_tolerance": 0,
                    }
                },
                "data_issues.deliver_in_original_cost_incomplete has unsupported "
                "keys: absolute_tolerance",
            ),
            (
                {"portfolio_market_value_continuity": {"enabled": False}},
                "data_issues.portfolio_market_value_continuity has unsupported "
                "keys: enabled",
            ),
        )
        for data_issues, expected_message in invalid_sections:
            with self.subTest(expected_message=expected_message):
                with tempfile.TemporaryDirectory() as temp_dir:
                    directory = Path(temp_dir)
                    configuration = _minimal_specification(directory)
                    configuration["data_issues"] = data_issues
                    path = _write_yaml(directory, configuration)

                    with self.assertRaises(PpaError) as context:
                        AuditSpecification(path)

                self.assertTrue(str(context.exception).startswith("Error 504"))
                self.assertIn(expected_message, str(context.exception))


if __name__ == "__main__":
    unittest.main()
