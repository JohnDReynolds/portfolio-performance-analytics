"""Focused tests for the temporary scale-check data expansion helpers."""

# Tests intentionally exercise private script helpers as the stable unit boundary.
# pylint: disable=protected-access

from pathlib import Path
import io
import tempfile
import unittest
from unittest import mock

import polars as pl

from scripts import check_scale


class TestScaleCheck(unittest.TestCase):
    """Verify scale data stays coherent without running expensive workflows."""

    def test_scale_choices_are_tens_from_10_through_100(self) -> None:
        """Manual large-site scale accepts only the ten supported increments."""
        for scale in range(10, 101, 10):
            self.assertEqual(
                check_scale._parse_args(["--scale", str(scale)]).scale,
                scale,
            )
        for scale in (0, 1, 9, 11, 30_000, 101):
            with self.subTest(scale=scale):
                with self.assertRaises(SystemExit):
                    check_scale._parse_args(["--scale", str(scale)])

    def test_large_site_expansion_preserves_original_and_suffixes_copies(self) -> None:
        """Portfolio copies retain source rows and use distinct identifiers."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "performance.csv"
            original = pl.DataFrame(
                {
                    "PORTFOLIO_CODE": ["P1", "P2"],
                    "RETURN": [0.01, 0.02],
                }
            )
            original.write_csv(path)
            expanded = check_scale._expanded_frame(
                path,
                3,
                ("PORTFOLIO_CODE",),
            )
            self.assertTrue(pl.read_csv(path).equals(original))

        self.assertEqual(expanded.height, 6)
        self.assertEqual(
            expanded["PORTFOLIO_CODE"].to_list(),
            [
                "P1",
                "P2",
                "P1_SCALE_001",
                "P2_SCALE_001",
                "P1_SCALE_002",
                "P2_SCALE_002",
            ],
        )
        self.assertEqual(expanded["RETURN"].to_list(), [0.01, 0.02] * 3)

    def test_selected_expansion_scales_financials_and_aligns_references(self) -> None:
        """Unique securities preserve total weight/contribution and lookup coverage."""
        performance = pl.DataFrame(
            {
                "SECURITY_ID": ["A", "B"],
                "BEGIN_WEIGHT": [0.6, 0.4],
                "CONTRIBUTION": [0.06, 0.02],
            }
        )
        reference = pl.DataFrame(
            {
                "SECURITY_ID": ["A", "B"],
                "SECURITY_NAME": ["Alpha", "Beta"],
            }
        )

        expanded_performance, expanded_reference = (
            check_scale._expanded_selected_analytics_frames(
                performance,
                reference,
                10,
            )
        )

        self.assertEqual(expanded_performance.height, 20)
        self.assertEqual(expanded_reference.height, 20)
        self.assertAlmostEqual(expanded_performance["BEGIN_WEIGHT"].sum(), 1.0)
        self.assertAlmostEqual(expanded_performance["CONTRIBUTION"].sum(), 0.08)
        self.assertEqual(
            set(expanded_performance["SECURITY_ID"]),
            set(expanded_reference["SECURITY_ID"]),
        )
        self.assertIn("Alpha_LOAD_09", set(expanded_reference["SECURITY_NAME"]))

    def test_generated_paths_must_stay_inside_workspace(self) -> None:
        """Scale generation rejects paths outside its temporary workspace."""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            check_scale._require_workspace_path(workspace, workspace / "inside")
            with self.assertRaisesRegex(ValueError, "outside its workspace"):
                check_scale._require_workspace_path(workspace, workspace.parent / "outside")

    def test_analytics_scaling_ratio_passes_warns_and_fails(self) -> None:
        """Large-site timing uses the documented 1.05x and 1.10x gates."""
        self.assertEqual(
            check_scale._analytics_scaling_result(2.0, 2.10),
            ("PASS", 1.05),
        )
        self.assertEqual(
            check_scale._analytics_scaling_result(2.0, 2.16),
            ("WARN", 1.08),
        )
        with self.assertRaisesRegex(RuntimeError, "1.10x failure threshold"):
            check_scale._analytics_scaling_result(2.0, 2.21)
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            check_scale._analytics_scaling_result(0.0, 1.0)

    def test_history_expansion_shifts_years_without_changing_values(self) -> None:
        """History copies are chronological and preserve their financial rows."""
        source = pl.DataFrame(
            {
                "FROM_DATE": ["2020-02-01", "2020-03-01"],
                "THRU_DATE": ["2020-02-29", "2020-03-31"],
                "RETURN": [0.01, 0.02],
            }
        )

        expanded = check_scale._expanded_history_frame(source, 3)

        self.assertEqual(expanded.height, 6)
        self.assertEqual(
            expanded["FROM_DATE"].cast(pl.String).to_list(),
            [
                "2020-02-01",
                "2020-03-01",
                "2025-02-01",
                "2025-03-01",
                "2030-02-01",
                "2030-03-01",
            ],
        )
        self.assertEqual(
            expanded["THRU_DATE"].cast(pl.String).to_list()[2],
            "2025-02-28",
        )
        self.assertEqual(expanded["RETURN"].to_list(), [0.01, 0.02] * 3)

    def test_scale_summary_displays_consistent_baseline_ratios_and_caps(self) -> None:
        """Every scenario summary exposes the same comparison fields."""
        output = io.StringIO()
        with mock.patch("sys.stdout", output):
            check_scale._print_scale_result(
                "Example",
                5,
                100,
                500,
                2.0,
                3.0,
                warning_cap=">2x",
                error_cap=">3x",
            )

        self.assertEqual(
            output.getvalue().splitlines(),
            [
                "PASS Example 5x",
                "  baseline 1x: rows=100, time=2.00s",
                "  scaled 5x: rows=500, time=3.00s",
                "  ratios: rows=5.00x, time=1.50x",
                "  time ratio caps: warning=>2x, error=>3x",
            ],
        )

    def test_timeout_summary_reports_lower_bound_without_traceback(self) -> None:
        """Timed-out workloads show their measured lower-bound ratio and caps."""
        output = io.StringIO()
        with mock.patch("sys.stdout", output):
            check_scale._print_timeout_result(
                "Audit large-site",
                50,
                1_000,
                50_000,
                2.0,
                110.0,
                52.5,
                55.0,
            )

        self.assertEqual(
            output.getvalue().splitlines(),
            [
                "FAIL Audit large-site 50x",
                "  baseline 1x: rows=1,000, time=2.00s",
                "  scaled 50x: rows=50,000, time=>110.00s (timed out)",
                "  ratios: rows=50.00x, time=>55.00x",
                "  time ratio caps: warning=>52.50x, error=>55.00x",
            ],
        )

    def test_workload_caps_are_derived_from_scale_factor(self) -> None:
        """Proportional workloads allow five and ten percent above their factor."""
        self.assertEqual(
            check_scale._workload_scaling_result("Example", 10, 2.0, 21.0),
            ("PASS", 10.5, 10.5, 11.0),
        )
        self.assertEqual(
            check_scale._workload_scaling_result("Example", 10, 2.0, 21.6),
            ("WARN", 10.8, 10.5, 11.0),
        )
        with self.assertRaisesRegex(RuntimeError, "11.00x time-ratio error cap"):
            check_scale._workload_scaling_result("Example", 10, 2.0, 22.01)
        self.assertEqual(check_scale._scaled_timeout(2.0, 11.0), 22.0)

    def test_sublinear_analytics_caps_use_reduced_expected_growth(self) -> None:
        """Selected and long-history caps reflect their observed workload shape."""
        self.assertEqual(
            check_scale._sublinear_scaling_result("Selected", 10, 2.0, 4.0),
            ("PASS", 2.0, 2.1, 2.2),
        )
        status, ratio, warning_ratio, error_ratio = (
            check_scale._sublinear_scaling_result("History", 5, 2.0, 3.2)
        )
        self.assertEqual(status, "WARN")
        self.assertEqual(ratio, 1.6)
        self.assertAlmostEqual(warning_ratio, 1.575)
        self.assertAlmostEqual(error_ratio, 1.65)
        with self.assertRaisesRegex(RuntimeError, "1.65x time-ratio error cap"):
            check_scale._sublinear_scaling_result("History", 5, 2.0, 3.31)


if __name__ == "__main__":
    unittest.main()
