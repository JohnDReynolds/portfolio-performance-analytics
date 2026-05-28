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

_AXYS_COMPARISON_PATH = Path("tests/data/axys/ppar_performance_comparison.yaml")


def _write_yaml(directory: Path, contents: object) -> Path:
    """Write comparison YAML contents and return the path."""
    path = directory / "ppar_performance_comparison.yaml"
    path.write_text(yaml.safe_dump(contents), encoding="utf-8")
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
            specification.snapshot_a.path,
            Path("tests/data/axys/axys_a"),
        )
        self.assertEqual(
            specification.snapshot_b.schema_path,
            Path("tests/data/axys/ppar_axys.yaml"),
        )

        portfolio_file = specification.files["portfolio_performance"]
        self.assertTrue(portfolio_file.required)
        self.assertEqual(portfolio_file.relative_path, Path("portperf.csv"))
        self.assertEqual(
            portfolio_file.snapshot_a_path,
            Path("tests/data/axys/axys_a/portperf.csv"),
        )
        self.assertEqual(
            specification.files["transactions"].snapshot_b_path,
            Path("tests/data/axys/axys_b/transactions.csv"),
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
