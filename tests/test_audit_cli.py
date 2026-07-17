"""Tests for performance comparison command-line modules."""

# Python imports
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

# Third-party imports
from openpyxl import load_workbook
import yaml

from ppar.errors import PpaError
from ppar.audit.cli import site_report as _site_report

_RESTATEMENT_COMPARISON_PATH = Path(
    "tests/data/axys/validation/ppar_audit_restatement.yaml"
)
_PORTFOLIO_COMPARISON_PATH = Path(
    "ppar/setup_templates/axys_apx_audit/axys_apx_audit.yaml"
)
_PACKAGED_AXYS_APX_DATA_PATH = Path("ppar/setup_templates/axys_apx_audit")
_AXYS_SNAPSHOT_PATH = Path("tests/data/axys/snapshots")
_BUNDLE_MODULE = "ppar.audit.cli.report_bundle"
_VALIDATE_BUNDLE_MODULE = "ppar.audit.cli.validate_bundle"
_VALIDATE_CONFIG_MODULE = "ppar.audit.cli.validate_config"
_VALIDATE_DEMO_MATRIX_MODULE = "ppar.audit.cli.validate_demo_matrix"
_SETUP_MODULE = "ppar.audit.cli.setup"
_SITE_REPORT_MODULE = "ppar.audit.cli.site_report"
_ANALYTICS_MODULE = "ppar.analytics.cli"
_PPAR_MODULE = "ppar.cli"
_DEMO_QUIET_PHRASES = (
    "Using quarterly reporting.",
    "Time:",
    "Analytics demo output written to:",
    "Report bundle written to:",
    "Bundle artifacts:",
    "Portfolio Audit Report",
    "Security Audit Report",
)


class TestAuditCli(unittest.TestCase):
    """Verify command-line report generation and validation commands."""

    def test_site_audit_returns_nonzero_for_output_row_limit(self) -> None:
        """The Audit CLI surfaces an oversized-report failure without success output."""
        stderr = io.StringIO()
        with mock.patch.object(
            _site_report,
            "run_report",
            side_effect=PpaError(
                "Audit output row limit exceeded. "
                "No files were written for the oversized report.",
                None,
            ),
        ):
            with redirect_stderr(stderr):
                exit_code = _site_report.main(["."], prog="ppar audit")

        self.assertEqual(exit_code, 1)
        self.assertIn("Report failed: Audit output row limit exceeded", stderr.getvalue())
        self.assertIn(
            "No files were written for the oversized report",
            stderr.getvalue(),
        )

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
                "Write Audit report bundles"
            ),
            _BUNDLE_MODULE: (
                "Write an Audit review artifact bundle."
            ),
            _VALIDATE_BUNDLE_MODULE: (
                "Validate an Audit report bundle."
            ),
            _VALIDATE_CONFIG_MODULE: (
                "Validate an Audit YAML configuration."
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
                    self.assertIn("--comparison-level", result.stdout)
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
                    self.assertNotIn("--guide", result.stdout)
                    self.assertIn("usage: ppar setup", result.stdout)
                    self.assertIn("ppar setup ./my_ppar_data", result.stdout)
                if module_name == _ANALYTICS_MODULE:
                    self.assertIn("usage: ppar analytics", result.stdout)
                    self.assertIn("ppar analytics ./my_ppar_data/analytics", result.stdout)
                if module_name == _SITE_REPORT_MODULE:
                    self.assertIn("--report", result.stdout)
                    self.assertIn("usage: ppar audit", result.stdout)
                    self.assertIn(
                        "ppar audit ./my_ppar_data/audit",
                        result.stdout,
                    )

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
        self.assertIn("audit", result.stdout)
        self.assertIn("Write Audit reports", result.stdout)
        self.assertIn("Write Performance Analytics reports", result.stdout)
        self.assertLess(
            result.stdout.index("audit"),
            result.stdout.index("analytics"),
        )
        self.assertNotIn("performance_comparison", result.stdout)
        self.assertNotIn("perfcomp", result.stdout)
        self.assertNotIn(
            "{analytics,setup,audit,performance_comparison,perfcomp}",
            result.stdout,
        )
        self.assertNotIn("PPAR command-line tools", result.stdout)
        self.assertIn("Examples:", result.stdout)
        self.assertIn("ppar setup ./my_ppar_data", result.stdout)
        self.assertIn("ppar analytics ./my_ppar_data/analytics", result.stdout)
        self.assertIn(
            "ppar audit ./my_ppar_data/audit",
            result.stdout,
        )
        self.assertNotIn("Set up and run PPAR reports.", result.stdout)
        self.assertNotIn("After setup", result.stdout)
        self.assertEqual(result.stderr, "")

    def test_top_level_ppar_cli_without_args_prints_first_run_handoff(self) -> None:
        """Typing ``ppar`` gives new users the setup command, not parser noise."""
        result = subprocess.run(
            _module_command(_PPAR_MODULE),
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("PPAR creates Axys/APX Audit", result.stdout)
        self.assertIn("Performance Analytics reports.", result.stdout)
        self.assertIn("First-time setup:", result.stdout)
        self.assertIn("ppar setup ./my_ppar_data", result.stdout)
        self.assertIn(
            "ppar audit ./my_ppar_data/audit",
            result.stdout,
        )
        self.assertIn("ppar analytics ./my_ppar_data/analytics", result.stdout)
        self.assertLess(
            result.stdout.index("ppar audit"),
            result.stdout.index("ppar analytics"),
        )
        self.assertNotIn("usage:", result.stdout)
        self.assertEqual(result.stderr, "")

    def test_setup_requires_site_directory(self) -> None:
        """Setup requires an explicit destination folder."""
        result = subprocess.run(
            _module_command(_SETUP_MODULE),
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("usage: ppar setup", result.stderr)
        self.assertIn("site_directory", result.stderr)
        self.assertNotIn("--guide", result.stderr)

    def test_setup_writes_canonical_yaml_configs(self) -> None:
        """Setup creates self-describing YAML for each starter workflow."""
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
            comparison_path = site_directory / "audit"
            config_path = comparison_path / "ppar.yaml"
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            readme = (site_directory / "README.md").read_text(encoding="utf-8")
            self.assertIn("PPAR setup complete:", result.stdout)
            self.assertIn("To run Audit:", result.stdout)
            self.assertIn("To run Performance Analytics:", result.stdout)
            self.assertIn("To customize with your own data:", result.stdout)
            self.assertIn(
                f"Refer to the \"Customizing With Your Own Data\" section in "
                f"{site_directory / 'README.md'}",
                result.stdout,
            )
            self.assertNotIn("(created)", result.stdout)
            self.assertNotIn("(written)", result.stdout)
            self.assertTrue((site_directory / "README.md").exists())
            self.assertIn("## Demos", readme)
            self.assertIn("## Customizing With Your Own Data", readme)
            self.assertIn("## Folder Map", readme)
            self.assertIn(
                "Replace `analytics/portperf.csv` with your own "
                "portfolio-performance export.",
                readme,
            )
            self.assertIn(
                "Replace `analytics/secperf.csv` with your own "
                "security-performance export.",
                readme,
            )
            self.assertIn(
                "Replace `analytics/secref.csv` with your own "
                "security reference export.",
                readme,
            )
            self.assertNotIn("PYTHON_TUTORIAL.md", readme)
            self.assertNotIn("Open the files listed in the command output.", readme)
            self.assertNotIn("Start by replacing the starter CSV files", readme)
            self.assertIn("analytics/run_analytics.py", readme)
            self.assertIn("run_audit.py", readme)
            self.assertIn("ppar analytics -h", readme)
            self.assertIn("python analytics/run_analytics.py -h", readme)
            self.assertIn("ppar audit -h", readme)
            self.assertIn("python audit/run_audit.py -h", readme)
            self.assertNotIn("run_portfolio_comparison.py", readme)
            self.assertNotIn("run_security_comparison.py", readme)
            self.assertIn(
                "If you want to customize the workflows and outputs",
                readme,
            )
            self.assertIn("Audit compares two snapshots", readme)
            self.assertIn("#### Getting Data from Axys/APX", readme)
            self.assertIn("Start with the comments under `files:`", readme)
            self.assertIn("use a REP performance or attribution", readme)
            self.assertIn("try IMEX first", readme)
            self.assertIn("source/destination and special-security context", readme)
            self.assertIn("PPAR-normalized examples", readme)
            self.assertEqual(readme.count("fx_rates.csv"), 2)
            self.assertEqual(readme.count("splits.csv"), 2)
            demo_sequence = [
                "## Demos",
                "### Audit",
                f"ppar audit {site_directory / 'audit'}",
                "### Performance Analytics",
                f"ppar analytics {site_directory / 'analytics'}",
                "## Customizing With Your Own Data",
            ]
            for before, after in zip(demo_sequence, demo_sequence[1:]):
                with self.subTest(before=before, after=after):
                    self.assertLess(readme.index(before), readme.index(after))
            self.assertIn("the original or older source-data snapshot", readme)
            self.assertIn("the newer, corrected, or restated source-data snapshot", readme)
            self.assertLess(
                readme.index("## Demos"),
                readme.index("## Customizing With Your Own Data"),
            )
            self.assertLess(
                readme.index("## Customizing With Your Own Data"),
                readme.index("## Folder Map"),
            )
            self.assertIn("Edit `analytics/ppar.yaml` if", readme)
            self.assertIn("Edit `audit/ppar.yaml`.", readme)
            self.assertTrue((analytics_path / "ppar.yaml").exists())
            self.assertTrue((analytics_path / "portperf.csv").exists())
            self.assertTrue((analytics_path / "secperf.csv").exists())
            self.assertTrue((analytics_path / "secref.csv").exists())
            self.assertTrue((analytics_path / "run_analytics.py").exists())
            analytics_script = (analytics_path / "run_analytics.py").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("ppar.demos", analytics_script)
            self.assertNotIn("TemporaryDirectory", analytics_script)
            self.assertNotIn("MPLCONFIGDIR", analytics_script)
            self.assertNotIn("This script is installed by", analytics_script)
            self.assertNotIn("CONFIG_PATH", analytics_script)
            self.assertIn("SPECIFICATIONS_PATH", analytics_script)
            self.assertIn("AxysData", analytics_script)
            self.assertIn("to_analytics", analytics_script)
            self.assertIn(
                "same command-line options as ``ppar analytics``",
                analytics_script,
            )
            self.assertIn("``python run_analytics.py -h``", analytics_script)
            self.assertTrue((comparison_path / "run_audit.py").exists())
            audit_script = (comparison_path / "run_audit.py").read_text(
                encoding="utf-8"
            )
            self.assertIn(
                "same command-line options as ``ppar audit``",
                audit_script,
            )
            self.assertIn("``python run_audit.py -h``", audit_script)
            self.assertFalse(
                (comparison_path / "run_portfolio_comparison.py").exists()
            )
            self.assertFalse(
                (comparison_path / "run_security_comparison.py").exists()
            )
            self.assertFalse((site_directory / "PYTHON_TUTORIAL.md").exists())
            self.assertFalse((site_directory / "generic_analytics").exists())
            self.assertEqual(config["snapshots"]["a"]["path"], "snapshot_a")
            self.assertEqual(config["snapshots"]["b"]["path"], "snapshot_b")
            for snapshot_name in ("a", "b"):
                snapshot = config["snapshots"][snapshot_name]
                self.assertEqual(snapshot["vendor"], "axys_apx")
                self.assertEqual(
                    snapshot["schema"],
                    "axys_apx_column_mappings.yaml",
                )
            self.assertIn("security_return_reconstruction", config)
            self.assertTrue(
                (comparison_path / "axys_apx_column_mappings.yaml").exists()
            )

    def test_setup_creates_starter_workspace(self) -> None:
        """Setup creates Analytics and Audit starter folders."""
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
            self.assertIn("ppar audit", result.stdout)
            self.assertNotIn("secperf.csv", result.stdout)
            self.assertTrue((site_directory / "analytics").is_dir())
            self.assertFalse((site_directory / "PYTHON_TUTORIAL.md").exists())
            self.assertTrue(
                (site_directory / "analytics" / "run_analytics.py").exists()
            )
            self.assertTrue(
                (site_directory / "audit" / "snapshot_a").is_dir()
            )
            self.assertTrue(
                (site_directory / "audit" / "snapshot_b").is_dir()
            )
            self.assertTrue((site_directory / "analytics" / "ppar.yaml").exists())
            self.assertTrue(
                (site_directory / "audit" / "ppar.yaml").exists()
            )
            self.assertFalse((site_directory / "output").exists())

    def test_setup_rerun_preserves_user_edits_without_overwrite(self) -> None:
        """Setup does not replace local user edits unless overwrite is requested."""
        with tempfile.TemporaryDirectory() as directory:
            site_directory = Path(directory) / "my_ppar_data"
            subprocess.run(
                _module_command(_SETUP_MODULE, str(site_directory)),
                check=True,
                capture_output=True,
                text=True,
            )
            readme_path = site_directory / "README.md"
            analytics_script_path = site_directory / "analytics" / "run_analytics.py"
            audit_script_path = (
                site_directory / "audit" / "run_audit.py"
            )
            analytics_config_path = site_directory / "analytics" / "ppar.yaml"
            audit_config_path = (
                site_directory / "audit" / "ppar.yaml"
            )

            custom_readme = "custom readme\n"
            custom_analytics_script = "# custom analytics script\n"
            custom_audit_script = "# custom audit script\n"
            custom_analytics_config = (
                analytics_config_path.read_text(encoding="utf-8")
                + "\n# custom analytics note\n"
            )
            custom_comparison_config = (
                audit_config_path.read_text(encoding="utf-8")
                + "\n# custom performance comparison note\n"
            )
            readme_path.write_text(custom_readme, encoding="utf-8")
            analytics_script_path.write_text(custom_analytics_script, encoding="utf-8")
            audit_script_path.write_text(custom_audit_script, encoding="utf-8")
            analytics_config_path.write_text(
                custom_analytics_config,
                encoding="utf-8",
            )
            audit_config_path.write_text(
                custom_comparison_config,
                encoding="utf-8",
            )

            subprocess.run(
                _module_command(_SETUP_MODULE, str(site_directory)),
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertEqual(readme_path.read_text(encoding="utf-8"), custom_readme)
            self.assertEqual(
                analytics_script_path.read_text(encoding="utf-8"),
                custom_analytics_script,
            )
            self.assertEqual(
                audit_script_path.read_text(encoding="utf-8"),
                custom_audit_script,
            )
            self.assertEqual(
                analytics_config_path.read_text(encoding="utf-8"),
                custom_analytics_config,
            )
            self.assertEqual(
                audit_config_path.read_text(encoding="utf-8"),
                custom_comparison_config,
            )

    def test_setup_can_include_hidden_generic_analytics_sample(self) -> None:
        """Setup can optionally copy generic analytics infrastructure."""
        with tempfile.TemporaryDirectory() as directory:
            site_directory = Path(directory) / "my_ppar_data"

            result = subprocess.run(
                _module_command(
                    _SETUP_MODULE,
                    "--include-generic-analytics",
                    str(site_directory),
                ),
                check=True,
                capture_output=True,
                text=True,
            )

            generic_directory = site_directory / "generic_analytics"
            self.assertIn("To run Generic Analytics:", result.stdout)
            self.assertIn(
                f"python {generic_directory / 'run_generic_analytics.py'}",
                result.stdout,
            )
            self.assertTrue(
                (generic_directory / "run_generic_analytics.py").exists()
            )
            generic_script = (
                generic_directory / "run_generic_analytics.py"
            ).read_text(encoding="utf-8")
            self.assertNotIn("ppar.demos", generic_script)
            self.assertNotIn("TemporaryDirectory", generic_script)
            self.assertNotIn("MPLCONFIGDIR", generic_script)
            self.assertTrue(
                (
                    generic_directory
                    / "performance"
                    / "Mega-Cap Alpha Portfolio.csv"
                ).exists()
            )
            self.assertTrue(
                (
                    generic_directory
                    / "classifications"
                    / "Economic Sector.csv"
                ).exists()
            )

    def test_setup_installed_python_scripts_run_end_to_end(self) -> None:
        """Copied setup scripts are the canonical Python smoke-test path."""
        with tempfile.TemporaryDirectory() as directory:
            site_directory = Path(directory) / "my_ppar_data"
            subprocess.run(
                _module_command(
                    _PPAR_MODULE,
                    "setup",
                    str(site_directory),
                    "--include-generic-analytics",
                ),
                check=True,
                capture_output=True,
                text=True,
            )

            script_paths = (
                site_directory / "analytics" / "run_analytics.py",
                site_directory
                / "audit"
                / "run_audit.py",
                site_directory / "generic_analytics" / "run_generic_analytics.py",
            )
            for script_path in script_paths:
                with self.subTest(script_path=script_path.name):
                    result = subprocess.run(
                        [sys.executable, str(script_path)],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    self.assertIn("Open these files to review", result.stdout)
                    self.assertEqual(result.stderr, "")

            self.assertTrue(
                (
                    site_directory
                    / "audit"
                    / "output"
                    / "portfolio"
                    / "portfolio_audit.xlsx"
                ).exists()
            )
            self.assertTrue(
                (
                    site_directory
                    / "audit"
                    / "output"
                    / "security"
                    / "security_audit.xlsx"
                ).exists()
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
                    / "generic_analytics"
                    / "output"
                    / "risk_statistics.html"
                ).exists()
            )

    def test_setup_audit_script_matches_default_cli_workflow(self) -> None:
        """The visible Python example stays equivalent to ``ppar audit``."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cli_site = root / "cli_site"
            script_site = root / "script_site"
            for site in (cli_site, script_site):
                subprocess.run(
                    _module_command(_PPAR_MODULE, "setup", str(site)),
                    check=True,
                    capture_output=True,
                    text=True,
                )

            cli_audit = cli_site / "audit"
            script_audit = script_site / "audit"
            subprocess.run(
                _module_command(_PPAR_MODULE, "audit", str(cli_audit)),
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [sys.executable, str(script_audit / "run_audit.py")],
                check=True,
                capture_output=True,
                text=True,
            )

            cli_output = cli_audit / "output"
            script_output = script_audit / "output"
            cli_files = {
                path.relative_to(cli_output)
                for path in cli_output.rglob("*")
                if path.is_file()
            }
            script_files = {
                path.relative_to(script_output)
                for path in script_output.rglob("*")
                if path.is_file()
            }
            self.assertEqual(cli_files, script_files)

            for report_level in ("portfolio", "security"):
                with self.subTest(report_level=report_level):
                    audit_stem = f"{report_level}_audit"
                    relative_html = Path(report_level) / f"{audit_stem}.html"
                    self.assertEqual(
                        (cli_output / relative_html).read_text(encoding="utf-8"),
                        (script_output / relative_html).read_text(encoding="utf-8"),
                    )
                    cli_workbook = load_workbook(
                        cli_output / report_level / f"{audit_stem}.xlsx",
                        read_only=True,
                        data_only=False,
                    )
                    script_workbook = load_workbook(
                        script_output / report_level / f"{audit_stem}.xlsx",
                        read_only=True,
                        data_only=False,
                    )
                    self.assertEqual(cli_workbook.sheetnames, script_workbook.sheetnames)
                    for sheet_name in cli_workbook.sheetnames:
                        self.assertEqual(
                            list(cli_workbook[sheet_name].values),
                            list(script_workbook[sheet_name].values),
                        )
                    cli_workbook.close()
                    script_workbook.close()

            cli_advanced = root / "cli_advanced"
            script_advanced = root / "script_advanced"
            shared_options = [
                "--report",
                "portfolio",
                "--title",
                "Custom Audit",
                "--exclude_suppressed",
                "--no-xlsx-output",
                "--include-reconstruction-diagnostics",
            ]
            subprocess.run(
                _module_command(
                    _PPAR_MODULE,
                    "audit",
                    str(cli_audit),
                    "--output",
                    str(cli_advanced),
                    *shared_options,
                ),
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(script_audit / "run_audit.py"),
                    "--output",
                    str(script_advanced),
                    *shared_options,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            cli_advanced_files = {
                path.relative_to(cli_advanced)
                for path in cli_advanced.rglob("*")
                if path.is_file()
            }
            self.assertEqual(
                cli_advanced_files,
                {
                    path.relative_to(script_advanced)
                    for path in script_advanced.rglob("*")
                    if path.is_file()
                },
            )
            self.assertEqual(
                (cli_advanced / "portfolio" / "portfolio_audit.html").read_text(
                    encoding="utf-8"
                ),
                (script_advanced / "portfolio" / "portfolio_audit.html").read_text(
                    encoding="utf-8"
                ),
            )
            self.assertFalse(
                (cli_advanced / "portfolio" / "portfolio_audit.xlsx").exists()
            )

            audit_help = subprocess.run(
                [sys.executable, str(script_audit / "run_audit.py"), "--help"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            for option in (
                "--report",
                "--output",
                "--title",
                "--no-xlsx-output",
                "--no-html-output",
                "--exclude_suppressed",
                "--include-reconstruction-diagnostics",
                "--require-causal-attribution",
                "--allow-incomplete-yaml",
            ):
                self.assertIn(option, audit_help)

    def test_setup_analytics_script_matches_default_cli_workflow(self) -> None:
        """The visible Python example stays equivalent to ``ppar analytics``."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cli_site = root / "cli_site"
            script_site = root / "script_site"
            for site in (cli_site, script_site):
                subprocess.run(
                    _module_command(_PPAR_MODULE, "setup", str(site)),
                    check=True,
                    capture_output=True,
                    text=True,
                )

            cli_analytics = cli_site / "analytics"
            script_analytics = script_site / "analytics"
            subprocess.run(
                _module_command(_PPAR_MODULE, "analytics", str(cli_analytics)),
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [sys.executable, str(script_analytics / "run_analytics.py")],
                check=True,
                capture_output=True,
                text=True,
            )

            cli_output = cli_analytics / "output"
            script_output = script_analytics / "output"
            cli_files = {
                path.relative_to(cli_output)
                for path in cli_output.rglob("*")
                if path.is_file()
            }
            script_files = {
                path.relative_to(script_output)
                for path in script_output.rglob("*")
                if path.is_file()
            }
            self.assertEqual(cli_files, script_files)
            for relative_path in cli_files:
                with self.subTest(relative_path=relative_path.as_posix()):
                    self.assertEqual(
                        (cli_output / relative_path).read_bytes(),
                        (script_output / relative_path).read_bytes(),
                    )

            cli_override_output = root / "cli_override_output"
            script_override_output = root / "script_override_output"
            subprocess.run(
                _module_command(
                    _PPAR_MODULE,
                    "analytics",
                    str(cli_analytics),
                    "--frequency",
                    "yearly",
                    "--from-date",
                    "2022-01-01",
                    "--thru-date",
                    "2025-12-31",
                    "--classification",
                    "Security",
                    "--minimum-acceptable-return",
                    "0.02",
                    "--risk-free-rate",
                    "0.04",
                    "--confidence-level",
                    "0.90",
                    "--portfolio-value",
                    "250000",
                    "--currency-symbol",
                    "EUR",
                    "--output",
                    str(cli_override_output),
                ),
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(script_analytics / "run_analytics.py"),
                    "--frequency",
                    "yearly",
                    "--from-date",
                    "2022-01-01",
                    "--thru-date",
                    "2025-12-31",
                    "--classification",
                    "Security",
                    "--minimum-acceptable-return",
                    "0.02",
                    "--risk-free-rate",
                    "0.04",
                    "--confidence-level",
                    "0.90",
                    "--portfolio-value",
                    "250000",
                    "--currency-symbol",
                    "EUR",
                    "--output",
                    str(script_override_output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            override_files = {
                path.relative_to(cli_override_output)
                for path in cli_override_output.rglob("*")
                if path.is_file()
            }
            self.assertEqual(
                override_files,
                {
                    path.relative_to(script_override_output)
                    for path in script_override_output.rglob("*")
                    if path.is_file()
                },
            )
            for relative_path in override_files:
                self.assertEqual(
                    (cli_override_output / relative_path).read_bytes(),
                    (script_override_output / relative_path).read_bytes(),
                )

            analytics_help = subprocess.run(
                [sys.executable, str(script_analytics / "run_analytics.py"), "--help"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            for option in (
                "--portfolio",
                "--benchmark",
                "--frequency",
                "--output",
                "--from-date",
                "--thru-date",
                "--classification",
                "--minimum-acceptable-return",
                "--risk-free-rate",
                "--confidence-level",
                "--portfolio-value",
                "--currency-symbol",
            ):
                self.assertIn(option, analytics_help)
            for yaml_setting in (
                "YAML analytics.portfolio in ppar.yaml",
                "YAML analytics.benchmark in ppar.yaml",
                "YAML analytics.frequency in ppar.yaml",
                "YAML analytics.output_directory in ppar.yaml",
                "YAML analytics.from_date",
                "YAML defaults.from_date in ppar.yaml",
                "YAML analytics.thru_date",
                "YAML defaults.thru_date in ppar.yaml",
                "YAML analytics.classification",
                "YAML defaults.classification in ppar.yaml",
                "YAML analytics.annual_minimum_acceptable_return in ppar.yaml",
                "YAML analytics.annual_risk_free_rate in ppar.yaml",
                "YAML analytics.confidence_level in ppar.yaml",
                "YAML analytics.portfolio_value in ppar.yaml",
                "YAML analytics.currency_symbol in ppar.yaml",
            ):
                self.assertIn(yaml_setting, analytics_help)

    def test_setup_audit_script_supports_cli_report_option(self) -> None:
        """The Python audit example supports the CLI report selection option."""
        with tempfile.TemporaryDirectory() as directory:
            site = Path(directory) / "site"
            subprocess.run(
                _module_command(_PPAR_MODULE, "setup", str(site)),
                check=True,
                capture_output=True,
                text=True,
            )
            audit_directory = site / "audit"
            result = subprocess.run(
                [
                    sys.executable,
                    str(audit_directory / "run_audit.py"),
                    "--report",
                    "portfolio",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("output/portfolio/portfolio_audit.xlsx", result.stdout)
            self.assertNotIn("output/security/security_audit.xlsx", result.stdout)
            self.assertTrue((audit_directory / "output" / "portfolio").is_dir())
            self.assertFalse((audit_directory / "output" / "security").exists())

            audit_help = subprocess.run(
                [sys.executable, str(audit_directory / "run_audit.py"), "--help"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            self.assertIn("--report", audit_help)
            self.assertIn("--no-xlsx-output", audit_help)
            self.assertIn("--no-html-output", audit_help)
            self.assertIn(
                "Supplying both options writes a CSV-only audit.",
                audit_help,
            )
            self.assertIn(
                "Use this to focus review output on findings that still require attention.",
                audit_help,
            )
            self.assertIn("{portfolio,security,both}", audit_help)

    def test_public_python_entrypoints_accept_string_site_directories(self) -> None:
        """Programmatic entrypoints accept string paths as well as ``Path`` values."""
        from ppar.analytics.cli import run_analytics
        from ppar.errors import PpaError
        from ppar.audit.cli.site_report import run_report

        with tempfile.TemporaryDirectory() as directory:
            missing_analytics_path = str(Path(directory) / "missing_analytics")
            missing_comparison_path = str(Path(directory) / "missing_comparison")

            with self.assertRaises(PpaError):
                run_analytics(missing_analytics_path)
            with self.assertRaises(PpaError):
                run_report(missing_comparison_path)

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
            audit_directory = site_directory / "audit"

            result = subprocess.run(
                _module_command(
                    _SITE_REPORT_MODULE,
                    str(audit_directory),
                ),
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn(
                "Open these files to review Audit output:",
                result.stdout,
            )
            self.assertIn("output/portfolio/portfolio_audit.xlsx", result.stdout)
            self.assertIn("output/security/security_audit.xlsx", result.stdout)
            self.assertIn("output/portfolio/portfolio_audit.html", result.stdout)
            self.assertIn("output/security/security_audit.html", result.stdout)
            self.assertTrue((audit_directory / "ppar.yaml").exists())
            self.assertTrue(
                (
                    audit_directory
                    / "output"
                    / "portfolio"
                    / "portfolio_audit.xlsx"
                ).exists()
            )
            self.assertTrue(
                (
                    audit_directory
                    / "output"
                    / "portfolio"
                    / "portfolio_audit.html"
                ).exists()
            )
            self.assertTrue(
                (
                    audit_directory
                    / "output"
                    / "security"
                    / "security_audit.xlsx"
                ).exists()
            )
            self.assertTrue(
                (
                    audit_directory
                    / "output"
                    / "security"
                    / "security_audit.html"
                ).exists()
            )

    def test_site_report_can_disable_html_output(self) -> None:
        """The production report command can write an XLSX-only audit."""
        with tempfile.TemporaryDirectory() as directory:
            site_directory = Path(directory) / "my_ppar_data"
            subprocess.run(
                _module_command(_SETUP_MODULE, str(site_directory)),
                check=True,
                capture_output=True,
                text=True,
            )
            audit_directory = site_directory / "audit"

            subprocess.run(
                _module_command(
                    _SITE_REPORT_MODULE,
                    str(audit_directory),
                    "--report",
                    "portfolio",
                    "--no-html-output",
                ),
                check=True,
                capture_output=True,
                text=True,
            )

            output_directory = audit_directory / "output" / "portfolio"
            self.assertTrue((output_directory / "portfolio_audit.xlsx").exists())
            self.assertFalse((output_directory / "portfolio_audit.html").exists())

    def test_site_report_can_write_csv_only_output(self) -> None:
        """Disabling XLSX and HTML promotes the canonical CSV review files."""
        with tempfile.TemporaryDirectory() as directory:
            site_directory = Path(directory) / "my_ppar_data"
            subprocess.run(
                _module_command(_SETUP_MODULE, str(site_directory)),
                check=True,
                capture_output=True,
                text=True,
            )
            audit_directory = site_directory / "audit"

            result = subprocess.run(
                _module_command(
                    _SITE_REPORT_MODULE,
                    str(audit_directory),
                    "--report",
                    "portfolio",
                    "--no-xlsx-output",
                    "--no-html-output",
                ),
                check=True,
                capture_output=True,
                text=True,
            )

            output_directory = audit_directory / "output" / "portfolio"
            self.assertFalse((output_directory / "portfolio_audit.xlsx").exists())
            self.assertFalse((output_directory / "portfolio_audit.html").exists())
            for file_name in (
                "performance_differences.csv",
                "performance_difference_causes.csv",
                "data_issues.csv",
                "source_detail.csv",
            ):
                with self.subTest(file_name=file_name):
                    self.assertTrue((output_directory / file_name).is_file())
            self.assertTrue((output_directory / "audit_support.zip").is_file())
            self.assertIn("performance_differences.csv", result.stdout)
            self.assertIn("performance_difference_causes.csv", result.stdout)
            self.assertIn("data_issues.csv", result.stdout)
            self.assertNotIn("source_detail.csv", result.stdout)

    def test_site_report_shares_reconstruction_cache_between_views(self) -> None:
        """One Audit run reuses reconstruction inputs for both report levels."""
        with tempfile.TemporaryDirectory() as directory:
            site_directory = Path(directory)
            (site_directory / "ppar.yaml").touch()
            comparison_views = mock.Mock()
            comparison_views.findings.side_effect = [
                mock.sentinel.portfolio_findings,
                mock.sentinel.security_findings,
            ]
            with (
                mock.patch.object(
                    _site_report,
                    "AuditComparisonViews",
                    return_value=comparison_views,
                ),
                mock.patch.object(
                    _site_report._data_issue_checks,
                    "data_issues_table",
                    return_value=mock.sentinel.data_issues,
                ),
                mock.patch.object(
                    _site_report,
                    "_write_report_bundle",
                    side_effect=(
                        [site_directory / "portfolio_audit.xlsx"],
                        [site_directory / "security_audit.xlsx"],
                    ),
                ) as write_report_bundle,
            ):
                _site_report.run_report(site_directory)

        portfolio_cache = write_report_bundle.call_args_list[0].kwargs[
            "_reconstruction_cache"
        ]
        security_cache = write_report_bundle.call_args_list[1].kwargs[
            "_reconstruction_cache"
        ]
        self.assertIs(portfolio_cache, security_cache)

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
            audit_directory = site_directory / "audit"
            (audit_directory / "snapshot_a" / "secperf.csv").unlink()
            (audit_directory / "snapshot_b" / "secperf.csv").unlink()

            portfolio_result = subprocess.run(
                _module_command(
                    _SITE_REPORT_MODULE,
                    str(audit_directory),
                    "--report",
                    "portfolio",
                ),
                check=True,
                capture_output=True,
                text=True,
            )
            default_result = subprocess.run(
                _module_command(_SITE_REPORT_MODULE, str(audit_directory)),
                check=True,
                capture_output=True,
                text=True,
            )
            security_result = subprocess.run(
                _module_command(
                    _SITE_REPORT_MODULE,
                    str(audit_directory),
                    "--report",
                    "security",
                ),
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertIn("PPAR setup complete:", setup_result.stdout)
            self.assertIn(
                "output/portfolio/portfolio_audit.xlsx",
                portfolio_result.stdout,
            )
            self.assertTrue(
                (
                    audit_directory
                    / "output"
                    / "portfolio"
                    / "portfolio_audit.xlsx"
                ).exists()
            )
            self.assertTrue(
                (
                    audit_directory
                    / "output"
                    / "portfolio"
                    / "portfolio_audit.html"
                ).exists()
            )
            self.assertIn(
                "output/portfolio/portfolio_audit.xlsx",
                default_result.stdout,
            )
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
            audit_directory = site_directory / "audit"

            result = subprocess.run(
                _module_command(
                    _SITE_REPORT_MODULE,
                    str(audit_directory),
                    "--report",
                    "security",
                ),
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn(
                "Open these files to review Audit output:",
                result.stdout,
            )
            self.assertIn("output/security/security_audit.xlsx", result.stdout)
            self.assertIn("output/security/security_audit.html", result.stdout)
            self.assertTrue(
                (
                    audit_directory
                    / "output"
                    / "security"
                    / "security_audit.xlsx"
                ).exists()
            )
            self.assertTrue(
                (
                    audit_directory
                    / "output"
                    / "security"
                    / "security_audit.html"
                ).exists()
            )
            self.assertFalse((audit_directory / "output" / "portfolio").exists())

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
            self.assertFalse((analytics_directory / "output" / ".matplotlib").exists())
            self.assertFalse((analytics_directory / "output" / ".cache").exists())

    def test_performance_comparison_demo_handoff_matches_quiet_success_contract(
        self,
    ) -> None:
        """Performance-comparison demo handoffs only list the workbook path."""
        from ppar.audit.cli.site_report import (
            _print_success,
        )

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            _print_success(
                {
                    "review_paths": [
                        Path("_demo_output")
                        / "audit_portfolio"
                        / "portfolio_audit.xlsx",
                    ],
                }
            )

        output = stdout.getvalue()
        self.assertIn("Open these files to review Audit output:", output)
        self.assertIn("portfolio_audit.xlsx", output)
        self.assertNotIn("portfolio_audit.html", output)
        self.assertNotIn("manifest.json", output)
        for phrase in _DEMO_QUIET_PHRASES:
            self.assertNotIn(phrase, output)

    def test_top_level_commands_do_not_default_from_setup_root(self) -> None:
        """Production commands require either a workflow folder or an explicit path."""
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
                capture_output=True,
                text=True,
            )
            comparison_result = subprocess.run(
                _module_command(
                    _PPAR_MODULE,
                    "audit",
                    "--report",
                    "portfolio",
                ),
                cwd=site_directory,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(analytics_result.returncode, 0)
            self.assertNotEqual(comparison_result.returncode, 0)
            self.assertIn("Analytics failed:", analytics_result.stderr)
            self.assertIn(
                "Run from the analytics folder or pass the folder.",
                analytics_result.stderr,
            )
            self.assertIn("ppar setup ./my_ppar_data", analytics_result.stderr)
            self.assertIn("Report failed:", comparison_result.stderr)
            self.assertIn(
                "Run from the audit folder or pass the folder.",
                comparison_result.stderr,
            )
            self.assertIn("ppar setup ./my_ppar_data", comparison_result.stderr)

    def test_top_level_commands_default_inside_workflow_folders(self) -> None:
        """Production commands can default to cwd inside their configured folder."""
        with tempfile.TemporaryDirectory() as directory:
            site_directory = Path(directory) / "my_ppar_data"
            subprocess.run(
                _module_command(_PPAR_MODULE, "setup", str(site_directory)),
                check=True,
                capture_output=True,
                text=True,
            )
            analytics_directory = site_directory / "analytics"
            audit_directory = site_directory / "audit"

            analytics_result = subprocess.run(
                _module_command(_PPAR_MODULE, "analytics"),
                cwd=analytics_directory,
                check=True,
                capture_output=True,
                text=True,
            )
            comparison_result = subprocess.run(
                _module_command(
                    _PPAR_MODULE,
                    "audit",
                    "--report",
                    "portfolio",
                ),
                cwd=audit_directory,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn(
                "Open these files to review analytics output:",
                analytics_result.stdout,
            )
            self.assertIn(
                "Open these files to review Audit output:",
                comparison_result.stdout,
            )
            self.assertTrue(
                (analytics_directory / "output" / "risk_statistics.html").exists()
            )
            self.assertTrue(
                (
                    audit_directory
                    / "output"
                    / "portfolio"
                    / "portfolio_audit.xlsx"
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
        """The bundle CLI writes concise output with complete archived support."""
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
            self.assertTrue((output_directory / "portfolio_audit.html").exists())
            self.assertFalse((output_directory / "report.md").exists())
            supporting_files = output_directory / "supporting_files"
            self.assertFalse(supporting_files.exists())
            self.assertTrue((output_directory / "source_detail.csv").exists())
            archive_path = output_directory / "audit_support.zip"
            self.assertTrue(archive_path.exists())
            report = (output_directory / "portfolio_audit.html").read_text(
                encoding="utf-8"
            )
            self.assertIn("<h1>Script Bundle Report</h1>", report)
            self.assertIn("Performance Differences", report)
            self.assertIn("Performance Difference Causes", report)

            with zipfile.ZipFile(archive_path) as archive:
                self.assertIn("supporting_files/findings.csv", archive.namelist())
                manifest = json.loads(
                    archive.read("supporting_files/manifest.json").decode("utf-8")
                )
            self.assertEqual(manifest["counts"]["findings"], 13)
            self.assertEqual(manifest["tables"]["context_evidence_summary"]["rows"], 2)
            self.assertEqual(manifest["tables"]["context_evidence"]["rows"], 2)
            self.assertEqual(manifest["tables"]["top_evidence"]["rows"], 2)
            self.assertEqual(
                manifest["artifacts"]["context_evidence"],
                "supporting_files/context_evidence.csv",
            )
            self.assertEqual(
                manifest["artifacts"]["context_evidence_summary"],
                "supporting_files/context_evidence_summary.csv",
            )
            self.assertEqual(
                manifest["artifacts"]["html_report"],
                "portfolio_audit.html",
            )

    def test_bundle_cli_module_accepts_comparison_level_override(self) -> None:
        """The bundle CLI can write a security report from a shared YAML file."""
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "security_bundle"

            result = subprocess.run(
                _module_command(
                    _BUNDLE_MODULE,
                    str(_PORTFOLIO_COMPARISON_PATH),
                    str(output_directory),
                    "--comparison-level",
                    "security",
                    "--include-workbook",
                    "--expand-all-supporting-files",
                ),
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn(str(output_directory), result.stdout)
            self.assertTrue((output_directory / "security_audit.html").exists())
            self.assertTrue((output_directory / "security_audit.xlsx").exists())
            supporting_files = output_directory / "supporting_files"
            manifest = json.loads(
                (supporting_files / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertGreater(manifest["counts"]["findings"], 0)
            readme = (output_directory / "README.md").read_text(encoding="utf-8")
            self.assertIn(
                "source-data differences additively explain each security period",
                readme,
            )
            self.assertEqual(
                manifest["artifacts"]["review_summary"],
                "supporting_files/review_summary.json",
            )
            self.assertNotIn("report", manifest["artifacts"])
            review_summary = json.loads(
                (supporting_files / "review_summary.json").read_text(
                    encoding="utf-8"
                )
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
            self.assertTrue((output_directory / "portfolio_audit.xlsx").exists())
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
            top_evidence_path = (
                output_directory / "supporting_files" / "top_evidence.csv"
            )
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
        self.assertIn("FX rate impact methods: none", result.stdout)
        self.assertIn("Evidence-only impact methods: fx_rates, splits", result.stdout)
        self.assertIn("Data Issues optional checks enabled:", result.stdout)
        self.assertIn("duplicate_transactions", result.stdout)
        self.assertIn(
            "Data Issues mandatory checks: portfolio_market_value_continuity, "
            "security_market_value_continuity",
            result.stdout,
        )
        self.assertIn(
            "Data Issues policy: mandatory continuity checks remain active; "
            "optional checks are enabled by default",
            result.stdout,
        )
        self.assertIn("Transaction rules configured: 15", result.stdout)
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
            "Transaction codes observed: by, cs, dp, dv, in, li, lo, pa, pd, rc, "
            "sa, sl, ss, wd",
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
        self.assertIn(
            "Transaction codes without YAML rules: BUY, DIV, INT, SELL, SPLIT",
            result.stdout,
        )
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
                "--expand-all-supporting-files",
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
    schema_path = _PACKAGED_AXYS_APX_DATA_PATH.resolve() / "axys_apx_column_mappings.yaml"
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
        _PACKAGED_AXYS_APX_DATA_PATH / "snapshot_a",
        site_directory / "snapshot_a",
    )
    shutil.copytree(
        _PACKAGED_AXYS_APX_DATA_PATH / "snapshot_b",
        site_directory / "snapshot_b",
    )
    return site_directory


def _module_command(module_name: str, *args: str) -> list[str]:
    """Return a subprocess command that runs a package CLI module."""
    return [sys.executable, "-m", module_name, *args]


if __name__ == "__main__":
    unittest.main()
