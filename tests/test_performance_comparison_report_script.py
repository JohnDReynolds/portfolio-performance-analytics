"""Tests for the performance comparison Markdown report script."""

# Python imports
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


if __name__ == "__main__":
    unittest.main()
