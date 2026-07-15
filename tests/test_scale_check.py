"""Focused tests for the temporary scale-check data expansion helpers."""

# Tests intentionally exercise private script helpers as the stable unit boundary.
# pylint: disable=protected-access

from pathlib import Path
import io
import json
import subprocess
import tempfile
import unittest
from unittest import mock
import zipfile

import polars as pl

from scripts import check_scale
from scripts import check_release_candidate
from scripts import audit_scale_contract


class TestScaleCheck(unittest.TestCase):
    """Verify scale data stays coherent without running expensive workflows."""

    def test_scale_choices_include_release_candidate_stress_level(self) -> None:
        """Large-site scale accepts routine increments and the 500x RC gate."""
        for scale in (*range(10, 101, 10), 500):
            self.assertEqual(
                check_scale._parse_args(["--scale", str(scale)]).scale,
                scale,
            )
        for scale in (0, 1, 9, 11, 101, 499, 501, 30_000):
            with self.subTest(scale=scale):
                with self.assertRaises(SystemExit):
                    check_scale._parse_args(["--scale", str(scale)])

    def test_run_surfaces_captured_child_process_error(self) -> None:
        """Failed scale subprocesses retain the diagnostic stderr text."""
        error = subprocess.CalledProcessError(
            1,
            ["example"],
            stderr="specific child-process failure",
        )
        with mock.patch("scripts.check_scale.subprocess.run", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "specific child-process failure"):
                check_scale._run(["example"])

    def test_short_command_timing_uses_median_of_three_samples(self) -> None:
        """A single process-startup outlier cannot determine a tight ratio gate."""
        with mock.patch(
            "scripts.check_scale._run",
            side_effect=[1.7, 4.9, 1.9],
        ) as run:
            elapsed = check_scale._run_median_elapsed(
                ["example"],
                timeout_seconds=7.0,
            )

        self.assertEqual(elapsed, 1.9)
        self.assertEqual(run.call_count, 3)
        run.assert_called_with(["example"], timeout_seconds=7.0)

    def test_process_timeout_adds_grace_without_changing_performance_cap(self) -> None:
        """Process-kill grace is separate from the ratio used to pass or fail."""
        self.assertEqual(check_scale._scaled_timeout(2.0, 11.0), 27.0)
        with self.assertRaisesRegex(RuntimeError, "1.10x failure threshold"):
            check_scale._analytics_scaling_result(2.0, 2.21)

    def test_supporting_csv_reader_accepts_compact_archive(self) -> None:
        """Scale verification reads supporting tables from compact bundles."""
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory)
            archive_path = report_path / "audit_support.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "supporting_files/findings.csv",
                    "portfolio_id,from_date\nP1,2025-01-01\n",
                )

            findings = audit_scale_contract.read_supporting_csv(
                report_path,
                "findings.csv",
            )

        self.assertEqual(findings["portfolio_id"].to_list(), ["P1"])
        self.assertEqual(str(findings.schema["from_date"]), "Date")

    def test_scaled_audit_contract_preserves_every_business_result_copy(self) -> None:
        """Synthetic portfolios must reproduce all contracted financial values."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_report = root / "baseline"
            scaled_report = root / "scaled"
            for report in (baseline_report, scaled_report):
                (report / "supporting_files").mkdir(parents=True)
            baseline = pl.DataFrame(
                {
                    "portfolio_id": ["P1"],
                    "performance_change": [0.0123],
                    "review_status": ["Fully Explained"],
                    "review_key": ["P1::2026-01-01::2026-01-31"],
                }
            )
            scaled = pl.concat(
                [
                    baseline,
                    baseline.with_columns(
                        pl.lit("P1_SCALE_001").alias("portfolio_id"),
                        pl.lit(
                            "P1_SCALE_001::2026-01-01::2026-01-31"
                        ).alias("review_key"),
                    ),
                ]
            )
            file_name = "performance_differences.csv"
            baseline.write_csv(baseline_report / "supporting_files" / file_name)
            scaled.write_csv(scaled_report / "supporting_files" / file_name)

            audit_scale_contract._assert_scaled_table_equivalent(
                baseline_report,
                scaled_report,
                file_name,
                2,
                excluded_columns=(),
            )

            scaled = scaled.with_row_index().with_columns(
                pl.when(pl.col("index") == 1)
                .then(0.0456)
                .otherwise(pl.col("performance_change"))
                .alias("performance_change")
            ).drop("index")
            scaled.write_csv(scaled_report / "supporting_files" / file_name)
            with self.assertRaisesRegex(RuntimeError, "business-result copies"):
                audit_scale_contract._assert_scaled_table_equivalent(
                    baseline_report,
                    scaled_report,
                    file_name,
                    2,
                    excluded_columns=(),
                )

    def test_500x_observational_baseline_records_but_does_not_replace_gate(self) -> None:
        """Maintained observations repeat the unchanged hard-gate parameters."""
        baseline_path = Path(check_scale.__file__).with_name(
            "audit_scale_baseline_500x.json"
        )
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        workload = baseline["workload"]
        gate = baseline["established_gate"]

        self.assertEqual(workload["scale"], 500)
        self.assertEqual(
            workload["scaled_input_rows"],
            workload["portfolio_scaled_input_rows"] * workload["scale"]
            + workload["site_level_input_rows"],
        )
        self.assertEqual(
            workload["baseline_input_rows"],
            workload["portfolio_scaled_input_rows"]
            + workload["site_level_input_rows"],
        )
        self.assertEqual(
            gate["scale_divisor"],
            check_scale._AUDIT_LARGE_SITE_SCALE_DIVISOR,
        )
        self.assertEqual(
            gate["warning_multiplier"],
            check_scale._SCALING_WARNING_MULTIPLIER,
        )
        self.assertEqual(
            gate["failure_multiplier"],
            check_scale._SCALING_FAILURE_MULTIPLIER,
        )
        _, _, warning_ratio, failure_ratio = (
            check_scale._audit_large_site_scaling_result(500, 1.0, 1.0)
        )
        self.assertAlmostEqual(gate["warning_ratio_at_500x"], warning_ratio)
        self.assertAlmostEqual(gate["failure_ratio_at_500x"], failure_ratio)

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

    def test_audit_history_expansion_shifts_every_date_column(self) -> None:
        """Audit history copies keep related performance and source dates aligned."""
        source = pl.DataFrame(
            {
                "PORT": ["P1"],
                "FROM_DATE": ["2024-01-01"],
                "THRU_DATE": ["2024-01-31"],
                "TRANSACTION_DATE": ["2024-01-15"],
                "SETTLE_DATE": ["2024-01-17"],
                "AMOUNT": [100.0],
            }
        )

        expanded = check_scale._expanded_audit_history_frame(source, 3)

        self.assertEqual(expanded.height, 3)
        self.assertEqual(
            expanded["FROM_DATE"].cast(pl.String).to_list(),
            ["2024-01-01", "2029-01-01", "2034-01-01"],
        )
        self.assertEqual(
            expanded["TRANSACTION_DATE"].cast(pl.String).to_list(),
            ["2024-01-15", "2029-01-15", "2034-01-15"],
        )
        self.assertEqual(
            expanded["SETTLE_DATE"].cast(pl.String).to_list(),
            ["2024-01-17", "2029-01-17", "2034-01-17"],
        )
        self.assertEqual(expanded["AMOUNT"].to_list(), [100.0] * 3)

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

    def test_audit_caps_reflect_observed_sublinear_growth(self) -> None:
        """Audit large-site and long-history caps catch meaningful regressions."""
        self.assertEqual(
            check_scale._audit_large_site_scaling_result(100, 1.0, 22.91),
            ("PASS", 22.91, 27.3, 28.6),
        )
        with self.assertRaisesRegex(RuntimeError, "28.60x time-ratio error cap"):
            check_scale._audit_large_site_scaling_result(100, 1.0, 28.61)

        self.assertEqual(
            check_scale._audit_history_scaling_result(1.0, 1.61),
            ("PASS", 1.61, 1.75, 2.0),
        )
        self.assertEqual(
            check_scale._audit_history_scaling_result(1.0, 1.80),
            ("WARN", 1.8, 1.75, 2.0),
        )
        with self.assertRaisesRegex(RuntimeError, "2.00x time-ratio error cap"):
            check_scale._audit_history_scaling_result(1.0, 2.01)

    def test_500x_audit_gate_uses_existing_formula_without_relaxation(self) -> None:
        """The RC stress level derives its caps from the established formula."""
        status, ratio, warning_ratio, error_ratio = (
            check_scale._audit_large_site_scaling_result(500, 1.0, 120.0)
        )

        self.assertEqual(status, "PASS")
        self.assertEqual(ratio, 120.0)
        self.assertAlmostEqual(warning_ratio, 132.3)
        self.assertAlmostEqual(error_ratio, 138.6)

    def test_release_candidate_runs_hard_500x_scale_gate(self) -> None:
        """The maintained RC batch includes the 500x scale command by default."""
        runner = mock.create_autospec(
            check_release_candidate.ReleaseCandidateRunner,
            instance=True,
        )

        check_release_candidate._run_scale_regression_check(runner)

        runner.run.assert_called_once_with(
            [
                check_release_candidate._VENV_PYTHON,
                "scripts/check_scale.py",
                "--scale",
                "500",
            ]
        )


if __name__ == "__main__":
    unittest.main()
