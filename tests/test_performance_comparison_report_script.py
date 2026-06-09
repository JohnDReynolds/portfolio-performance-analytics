"""Tests for the performance comparison Markdown report script."""

# Python imports
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

_RESTATEMENT_COMPARISON_PATH = Path(
    "tests/data/axys/ppar_performance_comparison_restatement.yaml"
)
_SUPPRESSED_COMPARISON_PATH = Path(
    "tests/data/axys/ppar_performance_comparison_suppressed.yaml"
)
_SCRIPT_PATH = Path("scripts/performance_comparison_report.py")
_HTML_SCRIPT_PATH = Path("scripts/performance_comparison_html_report.py")
_BUNDLE_SCRIPT_PATH = Path("scripts/performance_comparison_report_bundle.py")


class TestPerformanceComparisonReportScript(unittest.TestCase):
    """Verify command-line Markdown report generation."""

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
            self.assertTrue((output_directory / "impact_coverage.csv").exists())
            self.assertTrue((output_directory / "manifest.json").exists())
            report = (output_directory / "report.md").read_text(encoding="utf-8")
            self.assertIn("# Script Bundle Report", report)

            manifest = json.loads(
                (output_directory / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["counts"]["findings"], 21)
            self.assertEqual(manifest["tables"]["top_evidence"]["rows"], 2)
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
            self.assertEqual(manifest["counts"]["findings"], 20)
            self.assertEqual(manifest["counts"]["suppressed_findings"], 0)


def _html_section(report: str, section_id: str) -> str:
    """Return one HTML section by id."""
    start = f'<section class="pc-section" id="{section_id}">'
    return report.split(start, maxsplit=1)[1].split("</section>", maxsplit=1)[0]


if __name__ == "__main__":
    unittest.main()
