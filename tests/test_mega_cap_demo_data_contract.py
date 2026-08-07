"""Contract tests for the packaged Mega-Cap analytics demo data."""

# Python Imports
import datetime as dt
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import cast
import unittest

# Third-Party Imports
import pandas as pd

# Project Imports
from ppar.analytics import Analytics
from ppar.analytics.attribution import View
from ppar.analytics.cli import _frequency_from_string
from ppar.analytics.cli import run_analytics
from ppar.analytics.frequency import Frequency
import ppar.analytics.schema as cols
from ppar.axys_apx import AxysData


_PERFORMANCE_DIRECTORY = "ppar/setup_templates/generic_analytics/performance"
_CLASSIFICATION_DIRECTORY = "ppar/setup_templates/generic_analytics/classifications"
_MAPPING_DIRECTORY = "ppar/setup_templates/generic_analytics/mappings"

_PORTFOLIO_PATH = (
    f"{_PERFORMANCE_DIRECTORY}/Mega-Cap Alpha Portfolio.csv"
)
_BENCHMARK_PATH = f"{_PERFORMANCE_DIRECTORY}/Mega-Cap Benchmark.csv"
_SECURITY_PATH = f"{_CLASSIFICATION_DIRECTORY}/Security.csv"
_SECTOR_PATH = f"{_CLASSIFICATION_DIRECTORY}/Economic Sector.csv"
_MAPPING_PATH = f"{_MAPPING_DIRECTORY}/Security--to--Economic Sector.csv"
_AXYS_ANALYTICS_YAML = Path("ppar/setup_templates/axys_apx_analytics/axys_apx_analytics.yaml").resolve()
_AXYS_ANALYTICS_DIRECTORY = Path("ppar/setup_templates/axys_apx_analytics")
_AXYS_ANALYTICS_SECREF = _AXYS_ANALYTICS_DIRECTORY / "secmast.csv"
_AXYS_ANALYTICS_SECPERF = _AXYS_ANALYTICS_DIRECTORY / "secperf.csv"
_EXPECTED_ANALYTICS_ARTIFACTS = {
    "risk_statistics.html",
    "security_overall_attribution.html",
    "sector_cumulative_attribution.html",
    "sector_cumulative_attribution.png",
    "sector_cumulative_return.png",
    "sector_heatmap_active_contribution.png",
    "sector_heatmap_attribution.png",
    "sector_overall_attribution.html",
    "sector_overall_attribution.png",
    "sector_overall_contribution.png",
    "sector_subperiod_attribution.png",
}


class TestMegaCapDemoDataContract(unittest.TestCase):
    """Verify the promoted Mega-Cap demo data remains complete and usable."""

    def test_performance_files_have_sixty_consecutive_months(self) -> None:
        """Portfolio and benchmark contain the same 60 consecutive months."""
        portfolio = _read_performance(_PORTFOLIO_PATH)
        benchmark = _read_performance(_BENCHMARK_PATH)

        portfolio_months = _period_months(portfolio)
        benchmark_months = _period_months(benchmark)
        expected_months = pd.period_range(
            portfolio_months.min(),
            portfolio_months.max(),
            freq="M",
        )

        self.assertEqual(len(portfolio_months), 60)
        self.assertEqual(portfolio_months.tolist(), benchmark_months.tolist())
        self.assertEqual(portfolio_months.tolist(), expected_months.tolist())

    def test_performance_files_do_not_duplicate_security_names(self) -> None:
        """Security.csv is the sole source of user-facing security names."""
        expected_columns = ["from_date", "thru_date", "identifier", "weight", "return"]
        for path in (_PORTFOLIO_PATH, _BENCHMARK_PATH):
            self.assertEqual(list(pd.read_csv(path, nrows=0).columns), expected_columns)

    def test_period_weights_sum_to_one_and_cash_is_present(self) -> None:
        """Every period includes CASHUSD and sums to a complete portfolio."""
        for path in (_PORTFOLIO_PATH, _BENCHMARK_PATH):
            performance = _read_performance(path)
            period_weight_sums = performance.groupby(["from_date", "thru_date"])[
                "weight"
            ].sum()
            cash_rows = performance[performance["identifier"].eq("CASHUSD")]

            self.assertLess(float((period_weight_sums - 1.0).abs().max()), 1e-12)
            self.assertEqual(len(cash_rows), 60)
            self.assertGreater(float(cash_rows["weight"].mean()), 0.0)

    def test_classification_and_mapping_cover_every_identifier(self) -> None:
        """Every performance identifier has a name and economic-sector mapping."""
        identifiers = set(_read_performance(_PORTFOLIO_PATH)["identifier"])
        identifiers.update(_read_performance(_BENCHMARK_PATH)["identifier"])
        securities = pd.read_csv(
            _SECURITY_PATH,
            header=None,
            names=["identifier", "name"],
        )
        sectors = pd.read_csv(_SECTOR_PATH, header=None, names=["sector", "name"])
        mappings = pd.read_csv(
            _MAPPING_PATH,
            header=None,
            names=["identifier", "sector"],
        )

        self.assertTrue(identifiers.issubset(set(securities["identifier"])))
        self.assertTrue(identifiers.issubset(set(mappings["identifier"])))
        self.assertTrue(set(mappings["sector"]).issubset(set(sectors["sector"])))
        self.assertIn("CASHUSD", set(securities["identifier"]))
        self.assertIn("Cash", set(sectors["name"]))
        self.assertEqual(
            mappings.loc[mappings["identifier"].eq("CASHUSD"), "sector"].item(),
            "CA",
        )

    def test_analytics_outputs_preserve_demo_story(self) -> None:
        """Mega-Cap data loads through Analytics and keeps the intended story."""
        analytics = Analytics(
            _PORTFOLIO_PATH,
            _BENCHMARK_PATH,
            portfolio_classification_name="Security",
            benchmark_classification_name="Security",
            frequency=Frequency.MONTHLY,
        )
        security_attribution = analytics.get_attribution(
            "Security",
            _SECURITY_PATH,
        ).to_pandas(
            View.OVERALL_ATTRIBUTION
        )
        sector_attribution = analytics.get_attribution(
            "Economic Sector",
            _SECTOR_PATH,
            (_MAPPING_PATH, _MAPPING_PATH),
        ).to_pandas(View.OVERALL_ATTRIBUTION)
        risk = analytics.get_riskstatistics().to_pandas()

        portfolio_return = _cumulative_return(_read_performance(_PORTFOLIO_PATH))
        benchmark_return = _cumulative_return(_read_performance(_BENCHMARK_PATH))
        portfolio_sharpe = _risk_value(risk, "Annualized Sharpe Ratio", "Portfolio")
        benchmark_sharpe = _risk_value(risk, "Annualized Sharpe Ratio", "Benchmark")

        self.assertGreater(portfolio_return, benchmark_return)
        self.assertGreater(portfolio_sharpe, benchmark_sharpe)
        self.assertGreater(len(security_attribution), 0)
        self.assertIn("Intel Corporation", set(security_attribution["Classification_Name"]))
        self.assertGreater(len(sector_attribution), 0)
        self.assertIn("Cash", set(sector_attribution["Classification_Name"]))

    def test_analytics_readme_risk_story_matches_quarterly_demo(self) -> None:
        """Analytics README risk values remain synchronized with its quarterly demo."""
        analytics = Analytics(
            _PORTFOLIO_PATH,
            _BENCHMARK_PATH,
            portfolio_classification_name="Security",
            benchmark_classification_name="Security",
            frequency=Frequency.QUARTERLY,
        )
        risk = analytics.get_riskstatistics().to_pandas()
        portfolio_sharpe = _risk_value(risk, "Annualized Sharpe Ratio", "Portfolio")
        benchmark_sharpe = _risk_value(risk, "Annualized Sharpe Ratio", "Benchmark")
        portfolio_sortino = _risk_value(risk, "Annualized Sortino Ratio", "Portfolio")
        benchmark_sortino = _risk_value(risk, "Annualized Sortino Ratio", "Benchmark")
        expected_story = (
            f"Sharpe was about {portfolio_sharpe:.2f} versus {benchmark_sharpe:.2f}, and "
            f"Sortino was about {portfolio_sortino:.2f} versus {benchmark_sortino:.2f}."
        )
        readme = " ".join(
            Path("docs/analytics/README.md").read_text(encoding="utf-8").split()
        )

        self.assertIn(expected_story, readme)

    def test_fixed_frequencies_preserve_every_completed_demo_bucket(self) -> None:
        """Weekend source endpoints do not disappear from fixed-frequency output."""
        expected = (
            (Frequency.MONTHLY, 60, dt.date(2026, 5, 31)),
            (Frequency.QUARTERLY, 20, dt.date(2026, 3, 31)),
            (Frequency.YEARLY, 5, dt.date(2025, 12, 31)),
        )
        portfolio_source = _read_performance(_PORTFOLIO_PATH)

        for frequency, expected_count, expected_end in expected:
            with self.subTest(frequency=frequency):
                analytics = Analytics(
                    _PORTFOLIO_PATH,
                    _BENCHMARK_PATH,
                    portfolio_classification_name="Security",
                    benchmark_classification_name="Security",
                    frequency=frequency,
                )
                summary = analytics.get_attribution(
                    "Security",
                    _SECURITY_PATH,
                ).to_polars(View.SUBPERIOD_SUMMARY)

                self.assertEqual(summary.height, expected_count)
                self.assertEqual(summary[cols.THRU_DATE].item(-1), expected_end)
                if frequency == Frequency.QUARTERLY:
                    self.assertTrue(
                        {
                            dt.date(2022, 12, 31),
                            dt.date(2023, 9, 30),
                            dt.date(2023, 12, 31),
                            dt.date(2024, 3, 31),
                            dt.date(2024, 6, 30),
                        }.issubset(summary[cols.THRU_DATE].to_list())
                    )

                included_source = portfolio_source[
                    portfolio_source["thru_date"] <= pd.Timestamp(expected_end)
                ]
                consolidated_return = cast(
                    float,
                    (summary[cols.PORTFOLIO_RETURN] + 1).product() - 1,
                )
                self.assertAlmostEqual(
                    consolidated_return,
                    _cumulative_return(included_source),
                    places=12,
                )

    def test_axys_apx_analytics_fixture_matches_canonical_performance(self) -> None:
        """Axys analytics demo data is a lossless wrapper around Mega-Cap data."""
        axys_data = AxysData(_AXYS_ANALYTICS_YAML)
        axys_portfolio = axys_data.get_portfolio("MEGA_ALPHA")
        axys_benchmark = axys_data.get_portfolio("MEGA_BENCH")

        _assert_same_performance(
            self,
            axys_portfolio.security_performance.to_pandas(),
            _with_axys_apx_security_ids(
                _canonical_security_performance(_PORTFOLIO_PATH)
            ),
        )
        _assert_same_performance(
            self,
            axys_benchmark.security_performance.to_pandas(),
            _with_axys_apx_security_ids(
                _canonical_security_performance(_BENCHMARK_PATH)
            ),
        )

        analytics = axys_portfolio.to_analytics(
            axys_benchmark,
            frequency=Frequency.MONTHLY,
        )
        sector_attribution = analytics.get_attribution().to_pandas(
            View.OVERALL_ATTRIBUTION
        )
        self.assertIn("Cash", set(sector_attribution["Classification_Name"]))

    def test_axys_apx_analytics_separates_performance_and_reference_fields(self) -> None:
        """The analytics starter keeps repeated descriptions out of secperf."""
        secperf_columns = list(pd.read_csv(_AXYS_ANALYTICS_SECPERF, nrows=0).columns)
        self.assertEqual(
            secperf_columns,
            [
                "From Date",
                "Thru Date",
                "Portfolio Code",
                "Security Symbol",
                "Security Type",
                "Beginning Weight",
                "Security Return",
                "Contribution",
            ],
        )

        reference = pd.read_csv(_AXYS_ANALYTICS_SECREF).set_index("Security Symbol")
        self.assertTrue(reference.index.is_unique)
        self.assertEqual(reference.loc["AMZN", "Security Name"], "Amazon.Com Inc")
        self.assertEqual(reference.loc["CRM", "Security Name"], "Salesforce Inc")
        self.assertEqual(reference.loc["GE", "Security Name"], "Ge Aerospace")
        self.assertEqual(reference.loc["RTX", "Security Name"], "Rtx Corp")

    def test_standard_entrypoints_write_equivalent_semantic_artifacts(self) -> None:
        """CLI and Python runner write complete reports using reference names."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            site_directory = Path(temporary_directory) / "analytics"
            shutil.copytree(_AXYS_ANALYTICS_DIRECTORY, site_directory)
            (site_directory / "axys_apx_analytics.yaml").rename(
                site_directory / "ppar.yaml"
            )
            cli_output = Path(temporary_directory) / "cli_output"
            runner_output = Path(temporary_directory) / "runner_output"
            run_analytics(
                site_directory,
                output_directory=cli_output,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(site_directory / "run_analytics.py"),
                    "--output-directory",
                    str(runner_output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertEqual(_artifact_names(cli_output), _EXPECTED_ANALYTICS_ARTIFACTS)
            self.assertEqual(_artifact_names(runner_output), _EXPECTED_ANALYTICS_ARTIFACTS)
            _assert_semantic_analytics_output(self, cli_output)
            _assert_semantic_analytics_output(self, runner_output)

            for file_name in _EXPECTED_ANALYTICS_ARTIFACTS:
                if file_name.endswith(".html"):
                    self.assertEqual(
                        (cli_output / file_name).read_bytes(),
                        (runner_output / file_name).read_bytes(),
                    )

    def test_demo_frequency_parser_accepts_lenient_values(self) -> None:
        """Analytics demo frequency parsing accepts first-letter shortcuts."""
        self.assertEqual(
            _frequency_from_string(None),
            Frequency.AS_OFTEN_AS_POSSIBLE,
        )
        self.assertEqual(_frequency_from_string(""), Frequency.QUARTERLY)
        self.assertEqual(_frequency_from_string("monthly"), Frequency.MONTHLY)
        self.assertEqual(_frequency_from_string("M"), Frequency.MONTHLY)
        self.assertEqual(_frequency_from_string("quarterly"), Frequency.QUARTERLY)
        self.assertEqual(_frequency_from_string("q"), Frequency.QUARTERLY)
        self.assertEqual(_frequency_from_string("yearly"), Frequency.YEARLY)
        self.assertEqual(_frequency_from_string("Y"), Frequency.YEARLY)
        with self.assertRaises(ValueError):
            _frequency_from_string("weekly")


def _read_performance(path: str) -> pd.DataFrame:
    """Return one Mega-Cap performance file with parsed dates."""
    return pd.read_csv(path, parse_dates=["from_date", "thru_date"])


def _canonical_security_performance(path: str) -> pd.DataFrame:
    """Return canonical performance rows in AxysPortfolio output shape."""
    return _read_performance(path)[
        ["from_date", "thru_date", "identifier", "return", "weight"]
    ]


def _with_axys_apx_security_ids(performance: pd.DataFrame) -> pd.DataFrame:
    """Return canonical rows with the starter's type-first security keys."""
    result = performance.copy()
    security_types = result["identifier"].map(
        lambda identifier: "caus" if identifier == "CASHUSD" else "csus"
    )
    result["identifier"] = security_types + result["identifier"]
    return result


def _assert_same_performance(
    test_case: unittest.TestCase,
    actual: pd.DataFrame,
    expected: pd.DataFrame,
) -> None:
    """Assert that two security-performance tables contain the same economics."""
    sort_columns = ["from_date", "thru_date", "identifier"]
    actual_sorted = (
        actual[expected.columns].sort_values(sort_columns).reset_index(drop=True)
    )
    expected_sorted = expected.sort_values(sort_columns).reset_index(drop=True)

    test_case.assertEqual(len(actual_sorted), len(expected_sorted))
    pd.testing.assert_frame_equal(
        actual_sorted,
        expected_sorted,
        check_dtype=False,
        atol=1e-12,
        rtol=1e-12,
    )


def _period_months(performance: pd.DataFrame) -> pd.PeriodIndex:
    """Return sorted unique thru-date months from a performance file."""
    periods = performance[["from_date", "thru_date"]].drop_duplicates()
    thru_dates = periods.sort_values(["from_date", "thru_date"])["thru_date"]
    return pd.PeriodIndex(thru_dates.dt.to_period("M"))


def _cumulative_return(performance: pd.DataFrame) -> float:
    """Return cumulative weighted performance return."""
    period_returns = (
        performance.assign(contribution=performance["weight"] * performance["return"])
        .groupby(["from_date", "thru_date"], sort=True)["contribution"]
        .sum()
        .astype(float)
    )
    cumulative_growth = cast(float, period_returns.add(1.0).prod())
    return cumulative_growth - 1.0


def _risk_value(risk: pd.DataFrame, statistic: str, column: str) -> float:
    """Return one risk-statistics value."""
    row = risk[risk["column"].eq(statistic)]
    if row.empty:
        raise AssertionError(f"Missing risk statistic: {statistic}")
    return float(row[column].iloc[0])


def _artifact_names(output_directory: Path) -> set[str]:
    """Return file names written directly to an analytics output directory."""
    return {path.name for path in output_directory.iterdir() if path.is_file()}


def _assert_semantic_analytics_output(
    test_case: unittest.TestCase,
    output_directory: Path,
) -> None:
    """Assert that generated artifacts contain expected user-facing meaning."""
    for file_name in _EXPECTED_ANALYTICS_ARTIFACTS:
        path = output_directory / file_name
        test_case.assertGreater(path.stat().st_size, 0, file_name)
        if path.suffix == ".png":
            test_case.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n", file_name)

    security_html = (
        output_directory / "security_overall_attribution.html"
    ).read_text(encoding="utf-8")
    sector_html = (
        output_directory / "sector_overall_attribution.html"
    ).read_text(encoding="utf-8")
    risk_html = (output_directory / "risk_statistics.html").read_text(encoding="utf-8")

    test_case.assertIn("Apple Inc", security_html)
    test_case.assertIn("Amazon.Com Inc", security_html)
    # Security identifiers remain visible in their own ID column; they must not
    # leak into the adjacent display-name header when reference lookup fails.
    test_case.assertNotIn(">AAPL</th>", security_html)
    test_case.assertIn("Information Technology", sector_html)
    test_case.assertIn("Mega-Cap Alpha Portfolio", risk_html)
    test_case.assertIn("Mega-Cap Benchmark", risk_html)
    test_case.assertIn("Annualized Sharpe Ratio", risk_html)


if __name__ == "__main__":
    unittest.main()
