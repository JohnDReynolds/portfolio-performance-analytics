"""Tests for performance comparison YAML parsing and path resolution."""

# Python imports
from pathlib import Path
import tempfile
import unittest

# Third-party imports
import yaml

# Project imports
from ppar.errors import PpaError
from ppar.performance_comparison import PerformanceComparisonSpecification

_AXYS_COMPARISON_PATH = Path("tests/data/axys/validation/ppar_performance_comparison.yaml")
_AXYS_SNAPSHOT_PATH = Path("tests/data/axys/snapshots")
_TEST_AXYS_SCHEMA_PATH = Path("tests/data/axys/axys_column_mappings.yaml")


def _write_yaml(directory: Path, contents: object) -> Path:
    """Write comparison YAML contents and return the path."""
    path = directory / "ppar_performance_comparison.yaml"
    path.write_text(yaml.safe_dump(contents), encoding="utf-8")
    return path


def _write_yaml_text(directory: Path, contents: str) -> Path:
    """Write raw comparison YAML text and return the path."""
    path = directory / "ppar_performance_comparison.yaml"
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


class TestPerformanceComparisonSpecification(unittest.TestCase):
    """Verify comparison specification parsing and file preflight behavior."""

    def test_fixture_comparison_paths_are_resolved(self) -> None:
        """Committed baseline fixture resolves snapshots, schemas, and files."""
        specification = PerformanceComparisonSpecification(_AXYS_COMPARISON_PATH)

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

            specification = PerformanceComparisonSpecification(path)

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
                PerformanceComparisonSpecification(path)

            self.assertTrue(str(context.exception).startswith("Error 802"))
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
                PerformanceComparisonSpecification(path)

            self.assertTrue(str(context.exception).startswith("Error 802"))
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
                PerformanceComparisonSpecification(path)

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
                PerformanceComparisonSpecification(path)

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
                PerformanceComparisonSpecification(path)

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
                PerformanceComparisonSpecification(path)

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

            specification = PerformanceComparisonSpecification(path)

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
                PerformanceComparisonSpecification(path)

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
                PerformanceComparisonSpecification(path)

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
                PerformanceComparisonSpecification(path)

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
                PerformanceComparisonSpecification(path)

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
                PerformanceComparisonSpecification(path)

            self.assertTrue(str(context.exception).startswith("Error 504"))
            self.assertIn("snapshots.b must be a mapping", str(context.exception))

    def test_non_mapping_yaml_root_raises_error_504(self) -> None:
        """The comparison YAML root must be a mapping."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _write_yaml(Path(temp_dir), ["not", "a", "mapping"])

            with self.assertRaises(PpaError) as context:
                PerformanceComparisonSpecification(path)

            self.assertTrue(str(context.exception).startswith("Error 504"))
            self.assertIn("YAML must be a dictionary", str(context.exception))


if __name__ == "__main__":
    unittest.main()
