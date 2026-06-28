"""Tests for performance comparison command-line modules."""

# Python imports
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

# Third-party imports
import yaml

_RESTATEMENT_COMPARISON_PATH = Path(
    "tests/data/axys/validation/ppar_performance_comparison_restatement.yaml"
)
_PORTFOLIO_COMPARISON_PATH = Path(
    "ppar/demos/data/axys/ppar_performance_comparison.yaml"
)
_PACKAGED_AXYS_DATA_PATH = Path("ppar/demos/data/axys")
_AXYS_SNAPSHOT_PATH = Path("tests/data/axys/snapshots")
_BUNDLE_MODULE = "ppar.performance_comparison.cli.report_bundle"
_VALIDATE_BUNDLE_MODULE = "ppar.performance_comparison.cli.validate_bundle"
_VALIDATE_CONFIG_MODULE = "ppar.performance_comparison.cli.validate_config"
_VALIDATE_DEMO_MATRIX_MODULE = "ppar.performance_comparison.cli.validate_demo_matrix"


class TestPerformanceComparisonCli(unittest.TestCase):
    """Verify command-line report generation and validation commands."""

    def test_report_cli_modules_expose_help(self) -> None:
        """Report CLI modules expose consistent command-line help."""
        module_expectations = {
            _BUNDLE_MODULE: (
                "Write a performance comparison review artifact bundle."
            ),
            _VALIDATE_BUNDLE_MODULE: (
                "Validate a performance comparison report bundle."
            ),
            _VALIDATE_CONFIG_MODULE: (
                "Validate a performance comparison YAML configuration."
            ),
            _VALIDATE_DEMO_MATRIX_MODULE: (
                "Validate performance comparison scenario coverage."
            ),
        }

        for module_name, expected_description in module_expectations.items():
            with self.subTest(module_name=module_name):
                result = subprocess.run(
                    _module_command(module_name, "--help"),
                    check=True,
                    capture_output=True,
                    text=True,
                )

                self.assertIn(expected_description, result.stdout)
                self.assertIn("-h, --help", result.stdout)
                self.assertEqual(result.stderr, "")

    def test_report_cli_modules_reject_negative_top_evidence_limit(self) -> None:
        """Report CLI modules reject surprising negative evidence-row limits."""
        module_output_args = {
            _BUNDLE_MODULE: ("bundle",),
        }

        with tempfile.TemporaryDirectory() as directory:
            for module_name, output_args in module_output_args.items():
                with self.subTest(module_name=module_name):
                    result = subprocess.run(
                        _module_command(
                            module_name,
                            str(_RESTATEMENT_COMPARISON_PATH),
                            *[str(Path(directory) / value) for value in output_args],
                            "--top-evidence-limit",
                            "-1",
                        ),
                        check=False,
                        capture_output=True,
                        text=True,
                    )

                    self.assertEqual(result.returncode, 2)
                    self.assertIn("--top-evidence-limit", result.stderr)
                    self.assertIn("must be greater than or equal to 0", result.stderr)

    def test_bundle_cli_module_writes_report_bundle(self) -> None:
        """The bundle CLI module writes HTML, CSV, and manifest artifacts."""
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "bundle"

            result = subprocess.run(
                _module_command(
                    _BUNDLE_MODULE,
                    str(_RESTATEMENT_COMPARISON_PATH),
                    str(output_directory),
                    "--title",
                    "Script Bundle Report",
                    "--top-evidence-limit",
                    "2",
                    "--allow-incomplete-yaml",
                ),
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn(str(output_directory), result.stdout)
            self.assertTrue((output_directory / "report.html").exists())
            self.assertFalse((output_directory / "report.md").exists())
            self.assertTrue((output_directory / "findings.csv").exists())
            self.assertTrue((output_directory / "context_evidence_summary.csv").exists())
            self.assertTrue((output_directory / "context_evidence.csv").exists())
            self.assertTrue((output_directory / "impact_coverage.csv").exists())
            self.assertTrue((output_directory / "manifest.json").exists())
            report = (output_directory / "report.html").read_text(encoding="utf-8")
            self.assertIn("<h1>Script Bundle Report</h1>", report)
            self.assertIn("Performance Differences", report)
            self.assertIn("Identifiable Causes", report)

            manifest = json.loads(
                (output_directory / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["counts"]["findings"], 16)
            self.assertEqual(manifest["tables"]["context_evidence_summary"]["rows"], 5)
            self.assertEqual(manifest["tables"]["context_evidence"]["rows"], 5)
            self.assertEqual(manifest["tables"]["top_evidence"]["rows"], 2)
            self.assertEqual(
                manifest["artifacts"]["context_evidence"],
                "context_evidence.csv",
            )
            self.assertEqual(
                manifest["artifacts"]["context_evidence_summary"],
                "context_evidence_summary.csv",
            )
            self.assertEqual(manifest["artifacts"]["html_report"], "report.html")
            self.assertNotIn("report", manifest["artifacts"])

    def test_bundle_cli_module_requires_complete_yaml_by_default(self) -> None:
        """The bundle CLI fails before writing reports from incomplete YAML."""
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "bundle"

            result = subprocess.run(
                _module_command(
                    _BUNDLE_MODULE,
                    str(_RESTATEMENT_COMPARISON_PATH),
                    str(output_directory),
                ),
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 1)
        self.assertIn("YAML setup is incomplete", result.stderr)
        self.assertFalse(output_directory.exists())

    def test_bundle_cli_module_accepts_supported_attribution_setup_alias(self) -> None:
        """The clearer strict-setup alias preserves current strict semantics."""
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "bundle"

            result = subprocess.run(
                _module_command(
                    _BUNDLE_MODULE,
                    str(_PORTFOLIO_COMPARISON_PATH),
                    str(output_directory),
                    "--include-workbook",
                    "--require-supported-attribution-setup",
                ),
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0)
            self.assertIn("Report bundle written to:", result.stdout)
            self.assertLess(
                result.stdout.index("Review workbook written to:"),
                result.stdout.index("HTML report written to:"),
            )
            self.assertTrue((output_directory / "report.xlsx").exists())
            self.assertEqual(result.stderr, "")

    def test_validate_bundle_cli_module_accepts_valid_bundle(self) -> None:
        """The bundle validator CLI module accepts a generated bundle."""
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "bundle"
            self._write_bundle(output_directory)

            result = subprocess.run(
                _module_command(
                    _VALIDATE_BUNDLE_MODULE,
                    str(output_directory),
                ),
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("Bundle validation passed:", result.stdout)
            self.assertIn(str(output_directory), result.stdout)
            self.assertEqual(result.stderr, "")

    def test_validate_bundle_cli_module_reports_invalid_bundle(self) -> None:
        """The bundle validator CLI module exits nonzero for a broken bundle."""
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "bundle"
            self._write_bundle(output_directory)
            top_evidence_path = output_directory / "top_evidence.csv"
            header = top_evidence_path.read_text(encoding="utf-8").splitlines()[0]
            top_evidence_path.write_text(header + "\n", encoding="utf-8")

            result = subprocess.run(
                _module_command(
                    _VALIDATE_BUNDLE_MODULE,
                    str(output_directory),
                ),
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout, "")
            self.assertIn("Bundle validation failed:", result.stderr)
            self.assertIn("table 'top_evidence' row count is 0, expected 10", result.stderr)

    def test_validate_config_cli_module_accepts_valid_yaml(self) -> None:
        """The CLI config validator accepts a valid comparison YAML file."""
        result = subprocess.run(
            _module_command(
                _VALIDATE_CONFIG_MODULE,
                str(_RESTATEMENT_COMPARISON_PATH),
            ),
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("Config validation passed:", result.stdout)
        self.assertIn("Configured datasets:", result.stdout)
        self.assertIn("Missing optional files: none", result.stdout)
        self.assertIn(
            "Contribution impact methods: portfolio_source_field, "
            "security_contribution, security_return",
            result.stdout,
        )
        self.assertIn("Cash impact methods: none", result.stdout)
        self.assertIn("FX rate impact methods: none", result.stdout)
        self.assertIn("Security master impact methods: none", result.stdout)
        self.assertIn("Transaction rules configured: 0", result.stdout)
        self.assertIn("Transaction impact methods: none", result.stdout)
        self.assertIn("Transaction files checked: 2", result.stdout)
        self.assertIn("Transaction semantics sources:", result.stdout)
        self.assertEqual(result.stderr, "")

    def test_validate_config_cli_module_reports_invalid_yaml_contract(self) -> None:
        """The CLI config validator exits nonzero for malformed YAML contracts."""
        with tempfile.TemporaryDirectory() as directory:
            configuration = _absolute_restatement_configuration()
            configuration["transaction_impact_methods"] = {
                "performance": {
                    "method": "unsupported",
                    "denominator_source": "begin_market_value",
                },
            }
            comparison_path = Path(directory) / "comparison.yaml"
            comparison_path.write_text(
                yaml.safe_dump(configuration),
                encoding="utf-8",
            )

            result = subprocess.run(
                _module_command(
                    _VALIDATE_CONFIG_MODULE,
                    str(comparison_path),
                ),
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertIn("Config validation failed:", result.stderr)
        self.assertIn("performance.method must be", result.stderr)

    def test_validate_demo_matrix_cli_module_accepts_packaged_demos(self) -> None:
        """The CLI demo matrix validator confirms packaged scenario coverage."""
        result = subprocess.run(
            _module_command(_VALIDATE_DEMO_MATRIX_MODULE),
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("Demo matrix validation passed:", result.stdout)
        self.assertIn("Clean/no issue", result.stdout)
        self.assertIn("Missing transaction method", result.stdout)
        self.assertIn("Missing transaction rules", result.stdout)
        self.assertIn("Single-restatement transaction rows", result.stdout)
        self.assertIn("Transaction rules amount explanation", result.stdout)
        self.assertIn("Context-only evidence", result.stdout)
        self.assertIn("Portfolio field-role specifications", result.stdout)
        self.assertIn("Security field-role specifications", result.stdout)
        self.assertIn("Suppressed finding", result.stdout)
        self.assertIn("Residual withheld", result.stdout)
        self.assertEqual(result.stderr, "")

    def _write_bundle(self, output_directory: Path) -> None:
        """Write a standard report bundle for CLI validation tests."""
        subprocess.run(
            _module_command(
                _BUNDLE_MODULE,
                str(_RESTATEMENT_COMPARISON_PATH),
                str(output_directory),
                "--allow-incomplete-yaml",
            ),
            check=True,
            capture_output=True,
            text=True,
        )


def _absolute_restatement_configuration() -> dict[str, object]:
    """Return restatement YAML values with absolute fixture paths."""
    configuration = yaml.safe_load(_RESTATEMENT_COMPARISON_PATH.read_text(encoding="utf-8"))
    fixture_directory = _AXYS_SNAPSHOT_PATH.resolve()
    configuration["snapshots"]["a"]["path"] = str(fixture_directory / "axys_a")
    configuration["snapshots"]["b"]["path"] = str(
        fixture_directory / "axys_b_restatement"
    )
    schema_path = _PACKAGED_AXYS_DATA_PATH.resolve() / "axys_column_mappings.yaml"
    configuration["snapshots"]["a"]["schema"] = str(
        schema_path
    )
    configuration["snapshots"]["b"]["schema"] = str(
        schema_path
    )
    return configuration


def _module_command(module_name: str, *args: str) -> list[str]:
    """Return a subprocess command that runs a package CLI module."""
    return [sys.executable, "-m", module_name, *args]


if __name__ == "__main__":
    unittest.main()
