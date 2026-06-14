"""Tests for the performance comparison Markdown report script."""

# Python imports
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import cast
import unittest

# Third-party imports
import yaml

_RESTATEMENT_COMPARISON_PATH = Path(
    "tests/data/axys/ppar_performance_comparison_restatement.yaml"
)
_SUPPRESSED_COMPARISON_PATH = Path(
    "tests/data/axys/ppar_performance_comparison_suppressed.yaml"
)
_SCRIPT_PATH = Path("scripts/performance_comparison_report.py")
_HTML_SCRIPT_PATH = Path("scripts/performance_comparison_html_report.py")
_BUNDLE_SCRIPT_PATH = Path("scripts/performance_comparison_report_bundle.py")
_VALIDATE_BUNDLE_SCRIPT_PATH = Path("scripts/performance_comparison_validate_bundle.py")
_VALIDATE_CONFIG_SCRIPT_PATH = Path("scripts/performance_comparison_validate_config.py")
_VALIDATE_DEMO_MATRIX_SCRIPT_PATH = Path(
    "scripts/performance_comparison_validate_demo_matrix.py"
)


class TestPerformanceComparisonReportScript(unittest.TestCase):
    """Verify command-line Markdown report generation."""

    def test_report_scripts_expose_help(self) -> None:
        """Report scripts expose consistent command-line help."""
        script_expectations = {
            _SCRIPT_PATH: "Write a Markdown performance comparison report.",
            _HTML_SCRIPT_PATH: "Write an HTML performance comparison report.",
            _BUNDLE_SCRIPT_PATH: (
                "Write a performance comparison review artifact bundle."
            ),
            _VALIDATE_BUNDLE_SCRIPT_PATH: (
                "Validate a performance comparison report bundle."
            ),
            _VALIDATE_CONFIG_SCRIPT_PATH: (
                "Validate a performance comparison YAML configuration."
            ),
            _VALIDATE_DEMO_MATRIX_SCRIPT_PATH: (
                "Validate packaged performance comparison demo scenario coverage."
            ),
        }

        for script_path, expected_description in script_expectations.items():
            with self.subTest(script_path=script_path):
                result = subprocess.run(
                    [sys.executable, str(script_path), "--help"],
                    check=True,
                    capture_output=True,
                    text=True,
                )

                self.assertIn(expected_description, result.stdout)
                self.assertIn("-h, --help", result.stdout)
                self.assertEqual(result.stderr, "")

    def test_report_scripts_reject_negative_top_evidence_limit(self) -> None:
        """Report scripts reject surprising negative evidence-row limits."""
        script_output_args = {
            _SCRIPT_PATH: ("comparison.md",),
            _HTML_SCRIPT_PATH: ("comparison.html",),
            _BUNDLE_SCRIPT_PATH: ("bundle",),
        }

        with tempfile.TemporaryDirectory() as directory:
            for script_path, output_args in script_output_args.items():
                with self.subTest(script_path=script_path):
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(script_path),
                            str(_RESTATEMENT_COMPARISON_PATH),
                            *[str(Path(directory) / value) for value in output_args],
                            "--top-evidence-limit",
                            "-1",
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                    )

                    self.assertEqual(result.returncode, 2)
                    self.assertIn("--top-evidence-limit", result.stderr)
                    self.assertIn("must be greater than or equal to 0", result.stderr)

    def test_report_scripts_reject_non_integer_top_evidence_limit(self) -> None:
        """Report scripts reject non-integer evidence-row limits."""
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    str(_SCRIPT_PATH),
                    str(_RESTATEMENT_COMPARISON_PATH),
                    str(Path(directory) / "comparison.md"),
                    "--top-evidence-limit",
                    "many",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("--top-evidence-limit", result.stderr)
        self.assertIn("must be an integer", result.stderr)

    def test_script_writes_markdown_report(self) -> None:
        """The script writes a report for a comparison YAML file."""
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "reports" / "comparison.md"

            result = subprocess.run(
                [
                    sys.executable,
                    str(_SCRIPT_PATH),
                    str(_RESTATEMENT_COMPARISON_PATH),
                    str(output_path),
                    "--title",
                    "Script Restatement Report",
                    "--top-evidence-limit",
                    "2",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn(str(output_path), result.stdout)
            self.assertTrue(output_path.exists())
            report = output_path.read_text(encoding="utf-8")
            self.assertIn("# Script Restatement Report", report)
            self.assertIn("## Impact Estimate Summary", report)
            self.assertIn("## Residual Status", report)
            self.assertIn("## Transaction Activity", report)
            self.assertIn("## Top Evidence", report)
            self.assertIn("PC-PORT-MV", report)

    def test_script_can_omit_suppressed_appendix(self) -> None:
        """The script can suppress the suppressed-findings appendix section."""
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "comparison.md"

            subprocess.run(
                [
                    sys.executable,
                    str(_SCRIPT_PATH),
                    str(_SUPPRESSED_COMPARISON_PATH),
                    str(output_path),
                    "--no-suppressed-appendix",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            report = output_path.read_text(encoding="utf-8")
            self.assertIn("- Suppressed findings: 1", report)
            self.assertNotIn("## Suppressed Findings Appendix", report)

    def test_html_script_writes_html_report(self) -> None:
        """The HTML script writes a browser-readable report file."""
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "reports" / "comparison.html"

            result = subprocess.run(
                [
                    sys.executable,
                    str(_HTML_SCRIPT_PATH),
                    str(_RESTATEMENT_COMPARISON_PATH),
                    str(output_path),
                    "--title",
                    "Script HTML Report",
                    "--top-evidence-limit",
                    "2",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn(str(output_path), result.stdout)
            self.assertTrue(output_path.exists())
            report = output_path.read_text(encoding="utf-8")
            self.assertIn("<h1>Script HTML Report</h1>", report)
            self.assertIn('id="impact-coverage"', report)
            self.assertIn("security_contribution", report)
            self.assertNotIn("PC-TXN-AMT", _html_section(report, "top-evidence"))

    def test_html_script_can_omit_suppressed_appendix(self) -> None:
        """The HTML script can suppress the suppressed-findings appendix."""
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "comparison.html"

            subprocess.run(
                [
                    sys.executable,
                    str(_HTML_SCRIPT_PATH),
                    str(_SUPPRESSED_COMPARISON_PATH),
                    str(output_path),
                    "--active-only",
                    "--no-suppressed-appendix",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            report = output_path.read_text(encoding="utf-8")
            self.assertIn("<span>Suppressed findings</span>", report)
            self.assertIn("<strong>0</strong>", report)
            self.assertNotIn("Suppressed Findings Appendix</h2>", report)

    def test_bundle_script_writes_report_bundle(self) -> None:
        """The bundle script writes Markdown, CSV, and manifest artifacts."""
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "bundle"

            result = subprocess.run(
                [
                    sys.executable,
                    str(_BUNDLE_SCRIPT_PATH),
                    str(_RESTATEMENT_COMPARISON_PATH),
                    str(output_directory),
                    "--title",
                    "Script Bundle Report",
                    "--top-evidence-limit",
                    "2",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn(str(output_directory), result.stdout)
            self.assertTrue((output_directory / "report.md").exists())
            self.assertTrue((output_directory / "findings.csv").exists())
            self.assertTrue((output_directory / "context_evidence_summary.csv").exists())
            self.assertTrue((output_directory / "context_evidence.csv").exists())
            self.assertTrue((output_directory / "impact_coverage.csv").exists())
            self.assertTrue((output_directory / "manifest.json").exists())
            report = (output_directory / "report.md").read_text(encoding="utf-8")
            self.assertIn("# Script Bundle Report", report)
            self.assertIn("## Context Evidence Summary", report)
            self.assertIn("## Context Evidence", report)

            manifest = json.loads(
                (output_directory / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["counts"]["findings"], 22)
            self.assertEqual(manifest["tables"]["context_evidence_summary"]["rows"], 4)
            self.assertEqual(manifest["tables"]["context_evidence"]["rows"], 4)
            self.assertEqual(manifest["tables"]["top_evidence"]["rows"], 2)
            self.assertEqual(
                manifest["artifacts"]["context_evidence"],
                "context_evidence.csv",
            )
            self.assertEqual(
                manifest["artifacts"]["context_evidence_summary"],
                "context_evidence_summary.csv",
            )
            self.assertEqual(manifest["artifacts"]["report"], "report.md")

    def test_bundle_script_supports_active_only_and_omits_appendix(self) -> None:
        """The bundle script passes active-only and appendix options through."""
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "bundle"

            subprocess.run(
                [
                    sys.executable,
                    str(_BUNDLE_SCRIPT_PATH),
                    str(_SUPPRESSED_COMPARISON_PATH),
                    str(output_directory),
                    "--active-only",
                    "--no-suppressed-appendix",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            report = (output_directory / "report.md").read_text(encoding="utf-8")
            manifest = json.loads(
                (output_directory / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertIn("- Suppressed findings: 0", report)
            self.assertNotIn("## Suppressed Findings Appendix", report)
            self.assertEqual(manifest["counts"]["findings"], 21)
            self.assertEqual(manifest["counts"]["suppressed_findings"], 0)

    def test_validate_bundle_script_accepts_valid_bundle(self) -> None:
        """The bundle validator script accepts a generated bundle."""
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "bundle"
            self._write_bundle(output_directory)

            result = subprocess.run(
                [
                    sys.executable,
                    str(_VALIDATE_BUNDLE_SCRIPT_PATH),
                    str(output_directory),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("Bundle validation passed:", result.stdout)
            self.assertIn(str(output_directory), result.stdout)
            self.assertEqual(result.stderr, "")

    def test_validate_bundle_script_reports_invalid_bundle(self) -> None:
        """The bundle validator script exits nonzero for a broken bundle."""
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "bundle"
            self._write_bundle(output_directory)
            top_evidence_path = output_directory / "top_evidence.csv"
            header = top_evidence_path.read_text(encoding="utf-8").splitlines()[0]
            top_evidence_path.write_text(header + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(_VALIDATE_BUNDLE_SCRIPT_PATH),
                    str(output_directory),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout, "")
            self.assertIn("Bundle validation failed:", result.stderr)
            self.assertIn("table 'top_evidence' row count is 0, expected 10", result.stderr)

    def test_validate_config_script_accepts_valid_yaml(self) -> None:
        """The config validator accepts a valid comparison YAML file."""
        result = subprocess.run(
            [
                sys.executable,
                str(_VALIDATE_CONFIG_SCRIPT_PATH),
                str(_RESTATEMENT_COMPARISON_PATH),
            ],
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
        self.assertIn("Transaction rules configured: 0", result.stdout)
        self.assertIn("Transaction impact methods: none", result.stdout)
        self.assertIn("Transaction files checked: 2", result.stdout)
        self.assertIn("Transaction semantics sources:", result.stdout)
        self.assertEqual(result.stderr, "")

    def test_validate_config_script_reports_missing_optional_files(self) -> None:
        """The config validator previews absent optional files without failing."""
        with tempfile.TemporaryDirectory() as directory:
            configuration = _absolute_restatement_configuration()
            files = cast(dict[str, object], configuration["files"])
            files["prices"] = "missing_prices.csv"
            comparison_path = Path(directory) / "comparison.yaml"
            comparison_path.write_text(
                yaml.safe_dump(configuration),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(_VALIDATE_CONFIG_SCRIPT_PATH),
                    str(comparison_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("Config validation passed:", result.stdout)
        self.assertIn("Missing optional files: prices:a, prices:b", result.stdout)
        self.assertEqual(result.stderr, "")

    def test_validate_config_script_reports_invalid_yaml_contract(self) -> None:
        """The config validator exits nonzero for malformed YAML contracts."""
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
                [
                    sys.executable,
                    str(_VALIDATE_CONFIG_SCRIPT_PATH),
                    str(comparison_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertIn("Config validation failed:", result.stderr)
        self.assertIn("performance.method must be", result.stderr)

    def test_validate_demo_matrix_script_accepts_packaged_demos(self) -> None:
        """The demo matrix validator confirms packaged scenario coverage."""
        result = subprocess.run(
            [
                sys.executable,
                str(_VALIDATE_DEMO_MATRIX_SCRIPT_PATH),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("Demo matrix validation passed:", result.stdout)
        self.assertIn("Clean/no issue", result.stdout)
        self.assertIn("Missing contribution policy", result.stdout)
        self.assertIn("Missing transaction method", result.stdout)
        self.assertIn("Missing denominator", result.stdout)
        self.assertIn("Missing transaction sign/flow semantics", result.stdout)
        self.assertIn("Low-confidence estimate", result.stdout)
        self.assertIn("Context-only evidence", result.stdout)
        self.assertIn("Suppressed finding", result.stdout)
        self.assertIn("Residual withheld", result.stdout)
        self.assertEqual(result.stderr, "")

    def _write_bundle(self, output_directory: Path) -> None:
        """Write a standard report bundle for script validation tests."""
        subprocess.run(
            [
                sys.executable,
                str(_BUNDLE_SCRIPT_PATH),
                str(_RESTATEMENT_COMPARISON_PATH),
                str(output_directory),
            ],
            check=True,
            capture_output=True,
            text=True,
        )


def _absolute_restatement_configuration() -> dict[str, object]:
    """Return restatement YAML values with absolute fixture paths."""
    configuration = yaml.safe_load(_RESTATEMENT_COMPARISON_PATH.read_text(encoding="utf-8"))
    fixture_directory = _RESTATEMENT_COMPARISON_PATH.parent.resolve()
    configuration["snapshots"]["a"]["path"] = str(fixture_directory / "axys_a")
    configuration["snapshots"]["b"]["path"] = str(
        fixture_directory / "axys_b_restatement"
    )
    configuration["snapshots"]["a"]["schema"] = str(
        fixture_directory / "axys_column_mappings.yaml"
    )
    configuration["snapshots"]["b"]["schema"] = str(
        fixture_directory / "axys_column_mappings.yaml"
    )
    return configuration


def _html_section(report: str, section_id: str) -> str:
    """Return one HTML section by id."""
    start = f'<section class="pc-section" id="{section_id}">'
    return report.split(start, maxsplit=1)[1].split("</section>", maxsplit=1)[0]


if __name__ == "__main__":
    unittest.main()
