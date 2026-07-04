"""Contract tests for the packaged Mega-Cap analytics demo data."""

# Python Imports
from pathlib import Path
from typing import cast
import unittest

# Third-Party Imports
import pandas as pd

# Project Imports
from ppar.analytics import Analytics
from ppar.analytics.attribution import View
from ppar.analytics.frequency import Frequency
from ppar.axys import AxysData
from ppar.demos.analytics_demo_outputs import demo_frequency_from_string


_PERFORMANCE_DIRECTORY = "ppar/demos/data/generic_analytics/performance"
_CLASSIFICATION_DIRECTORY = "ppar/demos/data/generic_analytics/classifications"
_MAPPING_DIRECTORY = "ppar/demos/data/generic_analytics/mappings"

_PORTFOLIO_PATH = (
    f"{_PERFORMANCE_DIRECTORY}/Mega-Cap Alpha Portfolio.csv"
)
_BENCHMARK_PATH = f"{_PERFORMANCE_DIRECTORY}/Mega-Cap Benchmark.csv"
_SECURITY_PATH = f"{_CLASSIFICATION_DIRECTORY}/Security.csv"
_SECTOR_PATH = f"{_CLASSIFICATION_DIRECTORY}/Economic Sector.csv"
_MAPPING_PATH = f"{_MAPPING_DIRECTORY}/Security--to--Economic Sector.csv"
_AXYS_ANALYTICS_YAML = Path("ppar/demos/data/axysapx_analytics/axysapx_analytics.yaml").resolve()


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

    def test_period_weights_sum_to_one_and_cash_is_present(self) -> None:
        """Every period includes CASH_USD and sums to a complete portfolio."""
        for path in (_PORTFOLIO_PATH, _BENCHMARK_PATH):
            performance = _read_performance(path)
            period_weight_sums = performance.groupby(["from_date", "thru_date"])[
                "weight"
            ].sum()
            cash_rows = performance[performance["identifier"].eq("CASH_USD")]

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
        self.assertIn("CASH_USD", set(securities["identifier"]))
        self.assertIn("Cash", set(sectors["name"]))
        self.assertEqual(
            mappings.loc[mappings["identifier"].eq("CASH_USD"), "sector"].item(),
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
        security_attribution = analytics.get_attribution().to_pandas(
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
        self.assertGreater(len(sector_attribution), 0)
        self.assertIn("Cash", set(sector_attribution["Classification_Name"]))

    def test_axysapx_analytics_fixture_matches_canonical_performance(self) -> None:
        """Axys analytics demo data is a lossless wrapper around Mega-Cap data."""
        axys_data = AxysData(_AXYS_ANALYTICS_YAML)
        axys_portfolio = axys_data.get_portfolio("MEGA_ALPHA")
        axys_benchmark = axys_data.get_portfolio("MEGA_BENCH")

        _assert_same_performance(
            self,
            axys_portfolio.security_performance.to_pandas(),
            _canonical_security_performance(_PORTFOLIO_PATH),
        )
        _assert_same_performance(
            self,
            axys_benchmark.security_performance.to_pandas(),
            _canonical_security_performance(_BENCHMARK_PATH),
        )

        analytics = axys_portfolio.to_analytics(
            axys_benchmark,
            frequency=Frequency.MONTHLY,
        )
        sector_attribution = analytics.get_attribution().to_pandas(
            View.OVERALL_ATTRIBUTION
        )
        self.assertIn("Cash", set(sector_attribution["Classification_Name"]))

    def test_demo_frequency_parser_accepts_lenient_values(self) -> None:
        """Analytics demo frequency parsing accepts first-letter shortcuts."""
        self.assertEqual(demo_frequency_from_string(None), Frequency.QUARTERLY)
        self.assertEqual(demo_frequency_from_string(""), Frequency.QUARTERLY)
        self.assertEqual(demo_frequency_from_string("monthly"), Frequency.MONTHLY)
        self.assertEqual(demo_frequency_from_string("M"), Frequency.MONTHLY)
        self.assertEqual(demo_frequency_from_string("quarterly"), Frequency.QUARTERLY)
        self.assertEqual(demo_frequency_from_string("q"), Frequency.QUARTERLY)
        self.assertEqual(demo_frequency_from_string("yearly"), Frequency.YEARLY)
        self.assertEqual(demo_frequency_from_string("Y"), Frequency.YEARLY)
        with self.assertRaises(ValueError):
            demo_frequency_from_string("weekly")


def _read_performance(path: str) -> pd.DataFrame:
    """Return one Mega-Cap performance file with parsed dates."""
    return pd.read_csv(path, parse_dates=["from_date", "thru_date"])


def _canonical_security_performance(path: str) -> pd.DataFrame:
    """Return canonical performance rows in AxysPortfolio output shape."""
    return _read_performance(path)[
        ["from_date", "thru_date", "identifier", "return", "weight"]
    ]


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


if __name__ == "__main__":
    unittest.main()
