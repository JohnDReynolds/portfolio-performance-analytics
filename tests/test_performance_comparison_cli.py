"""Tests for performance comparison command-line modules."""

# Python imports
import json
import os
from pathlib import Path
import shutil
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
    "ppar/demos/data/axysapx_performance_comparison/axysapx_performance_comparison.yaml"
)
_PACKAGED_AXYS_DATA_PATH = Path("ppar/demos/data/axysapx_performance_comparison")
_AXYS_SNAPSHOT_PATH = Path("tests/data/axys/snapshots")
_BUNDLE_MODULE = "ppar.performance_comparison.cli.report_bundle"
_VALIDATE_BUNDLE_MODULE = "ppar.performance_comparison.cli.validate_bundle"
_VALIDATE_CONFIG_MODULE = "ppar.performance_comparison.cli.validate_config"
_VALIDATE_DEMO_MATRIX_MODULE = "ppar.performance_comparison.cli.validate_demo_matrix"
_SETUP_MODULE = "ppar.performance_comparison.cli.setup"
_SITE_REPORT_MODULE = "ppar.performance_comparison.cli.site_report"
_ANALYTICS_MODULE = "ppar.analytics.cli"
_PPAR_MODULE = "ppar.cli"


class TestPerformanceComparisonCli(unittest.TestCase):
    """Verify command-line report generation and validation commands."""

    def test_report_cli_modules_expose_help(self) -> None:
        """Report CLI modules expose consistent command-line help."""
        module_expectations = {
            _SETUP_MODULE: (
                "Create an Axys/APX starter workspace"
            ),
            _ANALYTICS_MODULE: (
                "Write Axys/APX analytics reports"
            ),
            _SITE_REPORT_MODULE: (
                "Write performance-comparison report bundles"
            ),
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
                if module_name == _BUNDLE_MODULE:
                    self.assertIn(
                        "--include-reconstruction-diagnostics",
                        result.stdout,
                    )
                    self.assertIn("Reconstruction Summary", result.stdout)
                    self.assertIn("Return", result.stdout)
                    self.assertIn("Reconstruction Checks", result.stdout)
                    self.assertIn("Security Return Checks", result.stdout)
                if module_name == _SETUP_MODULE:
                    self.assertIn("--overwrite", result.stdout)
                    self.assertIn("--guide", result.stdout)
                if module_name == _SITE_REPORT_MODULE:
                    self.assertIn("--report", result.stdout)

    def test_top_level_ppar_cli_exposes_setup_analytics_and_comparison_help(self) -> None:
        """The product command separates setup from production report generation."""
        result = subprocess.run(
            _module_command(_PPAR_MODULE, "--help"),
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("usage: ppar <command> [options]", result.stdout)
        self.assertIn("analytics", result.stdout)
        self.assertIn("setup", result.stdout)
        self.assertIn("performance_comparison", result.stdout)
        self.assertIn("Write performance-comparison reports", result.stdout)
        self.assertIn("perfcomp", result.stdout)
        self.assertNotIn("{analytics,setup,performance_comparison,perfcomp}", result.stdout)
        self.assertNotIn("PPAR command-line tools", result.stdout)
        self.assertIn("Examples:", result.stdout)
        self.assertIn("ppar setup ./my_ppar_data", result.stdout)
        self.assertIn("ppar analytics ./my_ppar_data/analytics", result.stdout)
        self.assertNotIn("Set up and run PPAR reports.", result.stdout)
        self.assertNotIn("After setup", result.stdout)
        self.assertEqual(result.stderr, "")

    def test_setup_guide_prints_without_creating_files(self) -> None:
        """Installed users can print setup guidance without knowing package paths."""
        result = subprocess.run(
            _module_command(_SETUP_MODULE, "--guide"),
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("PPAR Axys/APX Setup", result.stdout)
        self.assertIn("ppar setup ./my_ppar_data", result.stdout)
        self.assertIn("ppar performance_comparison ./my_ppar_data", result.stdout)
        self.assertEqual(result.stderr, "")

    def test_setup_writes_one_yaml_config(self) -> None:
        """Setup creates one user-facing YAML per starter workflow."""
        with tempfile.TemporaryDirectory() as directory:
            site_directory = Path(directory) / "my_ppar_data"

            result = subprocess.run(
                _module_command(
                    _SETUP_MODULE,
                    str(site_directory),
                ),
                check=True,
                capture_output=True,
                text=True,
            )

            analytics_path = site_directory / "analytics"
            comparison_path = site_directory / "performance_comparison"
            config_path = comparison_path / "ppar.yaml"
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            self.assertIn("PPAR setup complete:", result.stdout)
            self.assertIn("To run Analytics:", result.stdout)
            self.assertIn("To run Performance Comparison:", result.stdout)
            self.assertIn("To customize with your own data:", result.stdout)
            self.assertNotIn("(created)", result.stdout)
            self.assertNotIn("(written)", result.stdout)
            self.assertTrue((site_directory / "README.md").exists())
            self.assertTrue((analytics_path / "ppar.yaml").exists())
            self.assertTrue((analytics_path / "portperf.csv").exists())
            self.assertTrue((analytics_path / "secperf.csv").exists())
            self.assertEqual(config["snapshots"]["a"]["path"], "snapshot_a")
            self.assertEqual(config["snapshots"]["b"]["path"], "snapshot_b")
            self.assertNotIn("schema", config["snapshots"]["a"])
            self.assertNotIn("schema", config["snapshots"]["b"])
            self.assertIn("security_return_reconstruction", config)
            self.assertFalse(
                (comparison_path / "axysapx_column_mappings.yaml").exists()
            )

    def test_setup_creates_starter_workspace(self) -> None:
        """Setup creates analytics and performance-comparison starter folders."""
        with tempfile.TemporaryDirectory() as directory:
            site_directory = Path(directory) / "my_ppar_data"

            result = subprocess.run(
                _module_command(
                    _SETUP_MODULE,
                    str(site_directory),
                ),
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("PPAR setup complete:", result.stdout)
            self.assertIn("ppar analytics", result.stdout)
            self.assertIn("ppar performance_comparison", result.stdout)
            self.assertNotIn("secperf.csv", result.stdout)
            self.assertTrue((site_directory / "analytics").is_dir())
            self.assertTrue(
                (site_directory / "performance_comparison" / "snapshot_a").is_dir()
            )
            self.assertTrue(
                (site_directory / "performance_comparison" / "snapshot_b").is_dir()
            )
            self.assertTrue((site_directory / "analytics" / "ppar.yaml").exists())
            self.assertTrue(
                (site_directory / "performance_comparison" / "ppar.yaml").exists()
            )
            self.assertFalse((site_directory / "output").exists())

    def test_site_report_writes_both_reports_by_default(self) -> None:
        """The production comparison command writes both workbooks by default."""
        with tempfile.TemporaryDirectory() as directory:
            site_directory = Path(directory) / "my_ppar_data"
            subprocess.run(
                _module_command(_SETUP_MODULE, str(site_directory)),
                check=True,
                capture_output=True,
                text=True,
            )
            comparison_directory = site_directory / "performance_comparison"

            result = subprocess.run(
                _module_command(
                    _SITE_REPORT_MODULE,
                    str(comparison_directory),
                ),
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn(
                "Open these files to review performance_comparison output:",
                result.stdout,
            )
            self.assertIn("output/portfolio/report.xlsx", result.stdout)
            self.assertIn("output/security/report.xlsx", result.stdout)
            self.assertNotIn("output/portfolio/report.html", result.stdout)
            self.assertNotIn("output/security/report.html", result.stdout)
            self.assertTrue((comparison_directory / "ppar.yaml").exists())
            self.assertTrue(
                (
                    comparison_directory / "output" / "portfolio" / "report.xlsx"
                ).exists()
            )
            self.assertTrue(
                (
                    comparison_directory / "output" / "portfolio" / "report.html"
                ).exists()
            )
            self.assertTrue(
                (
                    comparison_directory / "output" / "security" / "report.xlsx"
                ).exists()
            )

    def test_portfolio_reports_skip_unavailable_security_performance(self) -> None:
        """Portfolio reports run without secperf; default output skips security."""
        with tempfile.TemporaryDirectory() as directory:
            site_directory = Path(directory) / "my_ppar_data"
            setup_result = subprocess.run(
                _module_command(_SETUP_MODULE, str(site_directory)),
                check=True,
                capture_output=True,
                text=True,
            )
            comparison_directory = site_directory / "performance_comparison"
            (comparison_directory / "snapshot_a" / "secperf.csv").unlink()
            (comparison_directory / "snapshot_b" / "secperf.csv").unlink()

            portfolio_result = subprocess.run(
                _module_command(
                    _SITE_REPORT_MODULE,
                    str(comparison_directory),
                    "--report",
                    "portfolio",
                ),
                check=True,
                capture_output=True,
                text=True,
            )
            default_result = subprocess.run(
                _module_command(_SITE_REPORT_MODULE, str(comparison_directory)),
                check=True,
                capture_output=True,
                text=True,
            )
            security_result = subprocess.run(
                _module_command(
                    _SITE_REPORT_MODULE,
                    str(comparison_directory),
                    "--report",
                    "security",
                ),
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertIn("PPAR setup complete:", setup_result.stdout)
            self.assertIn("output/portfolio/report.xlsx", portfolio_result.stdout)
            self.assertTrue(
                (
                    comparison_directory / "output" / "portfolio" / "report.xlsx"
                ).exists()
            )
            self.assertIn("output/portfolio/report.xlsx", default_result.stdout)
            self.assertIn(
                "Security output skipped because files.security_performance is not available.",
                default_result.stdout,
            )
            self.assertEqual(security_result.returncode, 1)
            self.assertIn("security_performance", security_result.stderr)

    def test_site_report_writes_security_report_when_requested(self) -> None:
        """The production report command supports security as an opt-in report."""
        with tempfile.TemporaryDirectory() as directory:
            site_directory = Path(directory) / "my_ppar_data"
            subprocess.run(
                _module_command(_SETUP_MODULE, str(site_directory)),
                check=True,
                capture_output=True,
                text=True,
            )
            comparison_directory = site_directory / "performance_comparison"

            result = subprocess.run(
                _module_command(
                    _SITE_REPORT_MODULE,
                    str(comparison_directory),
                    "--report",
                    "security",
                ),
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn(
                "Open these files to review performance_comparison output:",
                result.stdout,
            )
            self.assertIn("output/security/report.xlsx", result.stdout)
            self.assertNotIn("output/security/report.html", result.stdout)
            self.assertTrue(
                (
                    comparison_directory / "output" / "security" / "report.xlsx"
                ).exists()
            )
            self.assertTrue(
                (
                    comparison_directory / "output" / "security" / "report.html"
                ).exists()
            )
            self.assertFalse((comparison_directory / "output" / "portfolio").exists())

    def test_top_level_performance_comparison_aliases_write_reports(self) -> None:
        """The top-level long command and alias both dispatch to comparison reports."""
        with tempfile.TemporaryDirectory() as directory:
            for command_name in ("performance_comparison", "perfcomp"):
                with self.subTest(command_name=command_name):
                    site_directory = Path(directory) / command_name / "my_ppar_data"
                    subprocess.run(
                        _module_command(_PPAR_MODULE, "setup", str(site_directory)),
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    comparison_directory = site_directory / "performance_comparison"

                    result = subprocess.run(
                        _module_command(
                            _PPAR_MODULE,
                            command_name,
                            str(comparison_directory),
                            "--report",
                            "portfolio",
                        ),
                        check=True,
                        capture_output=True,
                        text=True,
                    )

                    self.assertIn(
                        "Open these files to review performance_comparison output:",
                        result.stdout,
                    )
                    self.assertIn("output/portfolio/report.xlsx", result.stdout)
                    self.assertTrue(
                        (
                            comparison_directory
                            / "output"
                            / "portfolio"
                            / "report.xlsx"
                        ).exists()
                    )

    def test_analytics_cli_writes_site_outputs(self) -> None:
        """The production analytics command writes output from setup data."""
        with tempfile.TemporaryDirectory() as directory:
            site_directory = Path(directory) / "my_ppar_data"
            subprocess.run(
                _module_command(_SETUP_MODULE, str(site_directory)),
                check=True,
                capture_output=True,
                text=True,
            )
            analytics_directory = site_directory / "analytics"

            result = subprocess.run(
                _module_command(_ANALYTICS_MODULE, str(analytics_directory)),
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("Open these files to review analytics output:", result.stdout)
            self.assertIn("risk_statistics.html", result.stdout)
            self.assertNotIn("Using quarterly reporting.", result.stdout)
            self.assertNotIn("Time:", result.stdout)
            self.assertNotIn("Analytics output:", result.stdout)
            self.assertTrue(
                (analytics_directory / "output" / "risk_statistics.html").exists()
            )
            self.assertTrue(
                (
                    analytics_directory
                    / "output"
                    / "sector_overall_attribution.html"
                ).exists()
            )

    def test_top_level_commands_default_to_setup_child_folders(self) -> None:
        """Production commands can run without a path from the setup root."""
        with tempfile.TemporaryDirectory() as directory:
            site_directory = Path(directory) / "my_ppar_data"
            subprocess.run(
                _module_command(_PPAR_MODULE, "setup", str(site_directory)),
                check=True,
                capture_output=True,
                text=True,
            )

            analytics_result = subprocess.run(
                _module_command(_PPAR_MODULE, "analytics"),
                cwd=site_directory,
                check=True,
                capture_output=True,
                text=True,
            )
            comparison_result = subprocess.run(
                _module_command(
                    _PPAR_MODULE,
                    "performance_comparison",
                    "--report",
                    "portfolio",
                ),
                cwd=site_directory,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn(
                "Open these files to review analytics output:",
                analytics_result.stdout,
            )
            self.assertIn(
                "Open these files to review performance_comparison output:",
                comparison_result.stdout,
            )
            self.assertTrue(
                (
                    site_directory
                    / "analytics"
                    / "output"
                    / "risk_statistics.html"
                ).exists()
            )
            self.assertTrue(
                (
                    site_directory
                    / "performance_comparison"
                    / "output"
                    / "portfolio"
                    / "report.xlsx"
                ).exists()
            )

    def test_analytics_cli_resolves_relative_site_directory_once(self) -> None:
        """Analytics source paths stay config-relative when the site path is relative."""
        with tempfile.TemporaryDirectory(prefix="ppar_relative_site_") as directory:
            site_directory = Path(directory) / "my_ppar_data"
            relative_site_directory = Path(os.path.relpath(site_directory, Path.cwd()))
            subprocess.run(
                _module_command(_SETUP_MODULE, str(relative_site_directory)),
                check=True,
                capture_output=True,
                text=True,
            )
            analytics_directory = relative_site_directory / "analytics"

            result = subprocess.run(
                _module_command(_ANALYTICS_MODULE, str(analytics_directory)),
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("Open these files to review analytics output:", result.stdout)
            self.assertNotIn("analytics/../", result.stderr)
            self.assertTrue(
                (
                    site_directory
                    / "analytics"
                    / "output"
                    / "risk_statistics.html"
                ).exists()
            )

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
            self.assertTrue((output_directory / "review_summary.json").exists())
            report = (output_directory / "report.html").read_text(encoding="utf-8")
            self.assertIn("<h1>Script Bundle Report</h1>", report)
            self.assertIn("Performance Differences", report)
            self.assertIn("Performance Difference Causes", report)

            manifest = json.loads(
                (output_directory / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["counts"]["findings"], 13)
            self.assertEqual(manifest["tables"]["context_evidence_summary"]["rows"], 2)
            self.assertEqual(manifest["tables"]["context_evidence"]["rows"], 2)
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
            self.assertEqual(
                manifest["artifacts"]["review_summary"],
                "review_summary.json",
            )
            self.assertNotIn("report", manifest["artifacts"])
            review_summary = json.loads(
                (output_directory / "review_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                review_summary["review_basis"],
                "Modified Dietz evidence pack",
            )
            self.assertEqual(review_summary["entrypoints"], manifest["review_entrypoints"])

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
                str(_PORTFOLIO_COMPARISON_PATH),
            ),
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("Config validation passed:", result.stdout)
        self.assertIn("Configured datasets:", result.stdout)
        self.assertIn(
            "Minimum required datasets: holdings, portfolio_performance, "
            "security_performance, transactions",
            result.stdout,
        )
        self.assertIn("Required source-data columns:", result.stdout)
        self.assertIn(
            "portfolio_performance: portfolio_id, from_date, thru_date, "
            "portfolio_return",
            result.stdout,
        )
        self.assertIn(
            "security_performance: portfolio_id, security_id, from_date, "
            "thru_date, security_return",
            result.stdout,
        )
        self.assertIn("Missing optional files: none", result.stdout)
        self.assertIn("Contribution impact methods: none", result.stdout)
        self.assertIn("Cash impact methods: none", result.stdout)
        self.assertIn("FX rate impact methods: none", result.stdout)
        self.assertIn("Transaction rules configured: 11", result.stdout)
        self.assertIn("Transaction impact methods: external_flow, performance", result.stdout)
        self.assertIn("Transaction files checked: 2", result.stdout)
        self.assertIn("Extract contract: packaged:", result.stdout)
        self.assertIn("Enforce ambiguous Axys/APX flows: True", result.stdout)
        self.assertIn(
            "Required transaction context columns: security_type, "
            "source_destination_symbol, source_destination_type, "
            "special_security_symbol, special_security_type",
            result.stdout,
        )
        self.assertIn("Report-bundle source context:", result.stdout)
        self.assertIn("transaction semantics summary", result.stdout)
        self.assertIn(
            "Transaction codes observed: by, dp, dv, in, li, lo, pa, sa, sl, wd",
            result.stdout,
        )
        self.assertIn("Transaction codes without YAML rules: none", result.stdout)
        self.assertIn("Transaction semantics sources:", result.stdout)
        self.assertEqual(result.stderr, "")

    def test_validate_config_cli_module_rejects_incomplete_yaml_by_default(self) -> None:
        """The config validator rejects report YAML that would write misleading output."""
        result = subprocess.run(
            _module_command(
                _VALIDATE_CONFIG_MODULE,
                str(_RESTATEMENT_COMPARISON_PATH),
            ),
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertIn("Config validation failed:", result.stderr)
        self.assertIn("YAML setup is incomplete", result.stderr)
        self.assertIn("transactions.amount", result.stderr)

    def test_validate_config_cli_module_allows_diagnostic_incomplete_yaml(self) -> None:
        """Incomplete YAML validation remains available with an explicit flag."""
        result = subprocess.run(
            _module_command(
                _VALIDATE_CONFIG_MODULE,
                str(_RESTATEMENT_COMPARISON_PATH),
                "--allow-incomplete-yaml",
            ),
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("Config validation passed:", result.stdout)
        self.assertIn("Transaction codes without YAML rules: BUY, DIV, INT, SELL, SPLIT", result.stdout)
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
        self.assertIn("Demo matrix coverage includes ambiguous-flow", result.stdout)
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
        self.assertIn("Ambiguous flow context variants", result.stdout)
        self.assertIn("Code-only failure guard", result.stdout)
        self.assertIn("Reviewed local opt-out", result.stdout)
        self.assertIn("Review-only action quarantine", result.stdout)
        self.assertIn("Capital-return and short-side candidate gates", result.stdout)
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
    schema_path = _PACKAGED_AXYS_DATA_PATH.resolve() / "axysapx_column_mappings.yaml"
    configuration["snapshots"]["a"]["schema"] = str(
        schema_path
    )
    configuration["snapshots"]["b"]["schema"] = str(
        schema_path
    )
    return configuration


def _copy_site_snapshots(directory: Path) -> Path:
    """Copy packaged demo snapshots into a setup-style site folder."""
    site_directory = directory / "my_site_extracts"
    shutil.copytree(
        _PACKAGED_AXYS_DATA_PATH / "snapshot_a",
        site_directory / "snapshot_a",
    )
    shutil.copytree(
        _PACKAGED_AXYS_DATA_PATH / "snapshot_b",
        site_directory / "snapshot_b",
    )
    return site_directory


def _module_command(module_name: str, *args: str) -> list[str]:
    """Return a subprocess command that runs a package CLI module."""
    return [sys.executable, "-m", module_name, *args]


if __name__ == "__main__":
    unittest.main()
