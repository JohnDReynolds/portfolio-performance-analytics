"""Tests for portfolio return-reconstruction diagnostics."""

from __future__ import annotations

# Python imports
from pathlib import Path
import tempfile
import unittest

# Third-party imports
import polars as pl
import yaml

# Project imports
from ppar.errors import PpaError
from ppar.performance_comparison.return_reconstruction import (
    DERIVED_RETURN_DIFFERENCE,
    END_VALUE_DIFFERENCE,
    NET_FLOW_DIFFERENCE,
    RECONSTRUCTION_CATEGORY,
    RECONSTRUCTION_STATUS,
    RECONSTRUCTION_STATUS_ALIGNED,
    RECONSTRUCTION_STATUS_MISSING_INPUTS,
    REPORTED_RETURN_DIFFERENCE,
    WEIGHTED_FLOW_DIFFERENCE,
    portfolio_return_reconstruction_checks,
    return_reconstruction_summary,
    security_return_reconstruction_checks,
)
from ppar.performance_comparison.specification import PerformanceComparisonSpecification

_PORTFOLIO_COMPARISON_PATH = Path(
    "ppar/demos/data/axys/ppar_performance_comparison.yaml"
)
_BASELINE_COMPARISON_PATH = Path(
    "tests/data/axys/validation/ppar_performance_comparison.yaml"
)
_DEMO_AXYS_DIRECTORY = Path("ppar/demos/data/axys")


def _comparison_path_with_reconstruction_method(
    directory: Path,
    method: str,
) -> Path:
    """Write a temporary demo comparison YAML with one reconstruction method."""
    configuration = yaml.safe_load(_PORTFOLIO_COMPARISON_PATH.read_text(encoding="utf-8"))
    if not isinstance(configuration, dict):
        raise AssertionError("Expected demo comparison YAML to be a mapping.")
    snapshots = configuration["snapshots"]
    if not isinstance(snapshots, dict):
        raise AssertionError("Expected snapshots to be a mapping.")
    for snapshot in snapshots.values():
        if not isinstance(snapshot, dict):
            raise AssertionError("Expected snapshot to be a mapping.")
        snapshot["path"] = str((_DEMO_AXYS_DIRECTORY / str(snapshot["path"])).resolve())
        snapshot["schema"] = str((_DEMO_AXYS_DIRECTORY / str(snapshot["schema"])).resolve())

    for section_name in (
        "portfolio_return_reconstruction",
        "security_return_reconstruction",
    ):
        section = configuration[section_name]
        if not isinstance(section, dict):
            raise AssertionError(f"Expected {section_name} to be a mapping.")
        section["method"] = method
        if method in {"simple_dietz", "modified_simple_dietz"}:
            section.pop("flow_timing", None)
            section.pop("day_count", None)
            section.pop("inclusion_rule", None)

    path = directory / "ppar_performance_comparison.yaml"
    path.write_text(yaml.safe_dump(configuration), encoding="utf-8")
    return path


class TestPerformanceComparisonReturnReconstruction(unittest.TestCase):
    """Verify portfolio return-reconstruction diagnostics."""

    def test_missing_reconstruction_yaml_returns_empty_table(self) -> None:
        """Comparisons opt into reconstruction explicitly."""
        checks = portfolio_return_reconstruction_checks(_BASELINE_COMPARISON_PATH)
        security_checks = security_return_reconstruction_checks(
            _BASELINE_COMPARISON_PATH
        )
        summary = return_reconstruction_summary(_BASELINE_COMPARISON_PATH)

        self.assertTrue(checks.is_empty())
        self.assertTrue(security_checks.is_empty())
        self.assertTrue(summary.is_empty())

    def test_demo_reconstruction_checks_show_review_statuses(self) -> None:
        """Packaged demo reconstruction is aligned except missing first opens."""
        checks = portfolio_return_reconstruction_checks(_PORTFOLIO_COMPARISON_PATH)

        statuses = set(checks.get_column(RECONSTRUCTION_STATUS).to_list())
        self.assertEqual(statuses, {RECONSTRUCTION_STATUS_ALIGNED, RECONSTRUCTION_STATUS_MISSING_INPUTS})

        alpha_withdrawal = checks.filter(
            (pl.col("portfolio_id") == "ALPHA")
            & (pl.col("from_date") == pl.date(2026, 1, 1))
            & (pl.col("thru_date") == pl.date(2026, 1, 30))
        ).row(0, named=True)
        self.assertAlmostEqual(
            alpha_withdrawal[REPORTED_RETURN_DIFFERENCE],
            alpha_withdrawal[DERIVED_RETURN_DIFFERENCE],
            places=6,
        )
        self.assertEqual(
            alpha_withdrawal[RECONSTRUCTION_STATUS],
            RECONSTRUCTION_STATUS_ALIGNED,
        )

    def test_demo_portfolio_reconstruction_has_no_differences(self) -> None:
        """Packaged portfolio performance is generated from reconstruction inputs."""
        checks = portfolio_return_reconstruction_checks(_PORTFOLIO_COMPARISON_PATH)
        different_rows = checks.filter(
            ~pl.col(RECONSTRUCTION_STATUS).is_in(
                [RECONSTRUCTION_STATUS_ALIGNED, RECONSTRUCTION_STATUS_MISSING_INPUTS]
            )
        )

        self.assertTrue(different_rows.is_empty())

    def test_demo_security_reconstruction_checks_show_flow_inputs(self) -> None:
        """Security reconstruction treats buy/sell rows as security-level flows."""
        checks = security_return_reconstruction_checks(_PORTFOLIO_COMPARISON_PATH)

        statuses = set(checks.get_column(RECONSTRUCTION_STATUS).to_list())
        self.assertEqual(statuses, {RECONSTRUCTION_STATUS_ALIGNED, RECONSTRUCTION_STATUS_MISSING_INPUTS})

        alpha_aapl = checks.filter(
            (pl.col("portfolio_id") == "ALPHA")
            & (pl.col("security_id") == "AAPL")
            & (pl.col("from_date") == pl.date(2026, 2, 28))
            & (pl.col("thru_date") == pl.date(2026, 3, 31))
        ).row(0, named=True)
        self.assertGreater(alpha_aapl[END_VALUE_DIFFERENCE], 0.0)
        self.assertGreater(alpha_aapl[NET_FLOW_DIFFERENCE], 0.0)
        self.assertGreater(alpha_aapl[WEIGHTED_FLOW_DIFFERENCE], 0.0)
        self.assertEqual(
            alpha_aapl[RECONSTRUCTION_STATUS],
            RECONSTRUCTION_STATUS_ALIGNED,
        )

    def test_demo_reconstruction_summary_counts_available_checks(self) -> None:
        """Summary table counts portfolio and security reconstruction checks."""
        summary = return_reconstruction_summary(_PORTFOLIO_COMPARISON_PATH)

        check_types = set(summary.get_column("reconstruction_check_type").to_list())
        self.assertEqual(check_types, {"Portfolio Return", "Security Return"})
        self.assertTrue((summary.get_column("row_count") > 0).all())

    def test_simple_dietz_uses_beginning_value_denominator(self) -> None:
        """Simple Dietz excludes all flow weighting from the denominator."""
        with tempfile.TemporaryDirectory() as directory:
            path = _comparison_path_with_reconstruction_method(
                Path(directory),
                "simple_dietz",
            )

            checks = portfolio_return_reconstruction_checks(path)

        alpha_withdrawal = checks.filter(
            (pl.col("portfolio_id") == "ALPHA")
            & (pl.col("from_date") == pl.date(2026, 1, 1))
            & (pl.col("thru_date") == pl.date(2026, 1, 30))
        ).row(0, named=True)
        self.assertNotEqual(alpha_withdrawal[NET_FLOW_DIFFERENCE], 0.0)
        self.assertEqual(alpha_withdrawal[WEIGHTED_FLOW_DIFFERENCE], 0.0)

    def test_modified_simple_dietz_uses_half_weighted_flows(self) -> None:
        """Modified Simple Dietz uses a 0.5 weight for every included flow."""
        with tempfile.TemporaryDirectory() as directory:
            path = _comparison_path_with_reconstruction_method(
                Path(directory),
                "modified_simple_dietz",
            )

            checks = portfolio_return_reconstruction_checks(path)

        alpha_withdrawal = checks.filter(
            (pl.col("portfolio_id") == "ALPHA")
            & (pl.col("from_date") == pl.date(2026, 1, 1))
            & (pl.col("thru_date") == pl.date(2026, 1, 30))
        ).row(0, named=True)
        self.assertNotEqual(alpha_withdrawal[NET_FLOW_DIFFERENCE], 0.0)
        self.assertAlmostEqual(
            alpha_withdrawal[WEIGHTED_FLOW_DIFFERENCE],
            alpha_withdrawal[NET_FLOW_DIFFERENCE] * 0.5,
        )

    def test_security_modified_simple_dietz_uses_half_weighted_flows(self) -> None:
        """Security reconstruction uses the same Modified Simple Dietz weight."""
        with tempfile.TemporaryDirectory() as directory:
            path = _comparison_path_with_reconstruction_method(
                Path(directory),
                "modified_simple_dietz",
            )

            checks = security_return_reconstruction_checks(path)

        alpha_aapl = checks.filter(
            (pl.col("portfolio_id") == "ALPHA")
            & (pl.col("security_id") == "AAPL")
            & (pl.col("from_date") == pl.date(2026, 2, 28))
            & (pl.col("thru_date") == pl.date(2026, 3, 31))
        ).row(0, named=True)
        self.assertNotEqual(alpha_aapl[NET_FLOW_DIFFERENCE], 0.0)
        self.assertAlmostEqual(
            alpha_aapl[WEIGHTED_FLOW_DIFFERENCE],
            alpha_aapl[NET_FLOW_DIFFERENCE] * 0.5,
        )

    def test_malformed_reconstruction_yaml_fails_up_front(self) -> None:
        """Opted-in reconstruction YAML must include every required field."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.yaml"
            path.write_text(
                "\n".join(
                    [
                        "portfolio_return_reconstruction:",
                        "  method: modified_dietz",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(PpaError, "missing required keys"):
                PerformanceComparisonSpecification(path)


if __name__ == "__main__":
    unittest.main()
