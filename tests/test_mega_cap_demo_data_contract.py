"""Contract tests for the packaged Mega-Cap analytics demo data."""

# Python Imports
import unittest

# Third-Party Imports
import pandas as pd

# Project Imports
from ppar.analytics import Analytics
from ppar.analytics.attribution import View
from ppar.analytics.frequency import Frequency


_PERFORMANCE_DIRECTORY = "ppar/demos/data/performance"
_CLASSIFICATION_DIRECTORY = "ppar/demos/data/classifications"
_MAPPING_DIRECTORY = "ppar/demos/data/mappings"

_PORTFOLIO_PATH = (
    f"{_PERFORMANCE_DIRECTORY}/Mega-Cap Alpha Portfolio.csv"
)
_BENCHMARK_PATH = f"{_PERFORMANCE_DIRECTORY}/Mega-Cap Benchmark.csv"
_SECURITY_PATH = f"{_CLASSIFICATION_DIRECTORY}/Mega-Cap Security.csv"
_SECTOR_PATH = f"{_CLASSIFICATION_DIRECTORY}/Mega-Cap Economic Sector.csv"
_MAPPING_PATH = (
    f"{_MAPPING_DIRECTORY}/Mega-Cap Security--to--Mega-Cap Economic Sector.csv"
)


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
        """Every period includes CASHBAL and sums to a complete portfolio."""
        for path in (_PORTFOLIO_PATH, _BENCHMARK_PATH):
            performance = _read_performance(path)
            period_weight_sums = performance.groupby(["from_date", "thru_date"])[
                "weight"
            ].sum()
            cash_rows = performance[performance["identifier"].eq("CASHBAL")]

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
        self.assertIn("CASHBAL", set(securities["identifier"]))
        self.assertIn("Cash", set(sectors["name"]))
        self.assertEqual(
            mappings.loc[mappings["identifier"].eq("CASHBAL"), "sector"].item(),
            "CA",
        )

    def test_analytics_outputs_preserve_demo_story(self) -> None:
        """Mega-Cap data loads through Analytics and keeps the intended story."""
        analytics = Analytics(
            _PORTFOLIO_PATH,
            _BENCHMARK_PATH,
            portfolio_classification_name="Mega-Cap Security",
            benchmark_classification_name="Mega-Cap Security",
            frequency=Frequency.MONTHLY,
        )
        security_attribution = analytics.get_attribution().to_pandas(
            View.OVERALL_ATTRIBUTION
        )
        sector_attribution = analytics.get_attribution(
            "Mega-Cap Economic Sector",
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


def _read_performance(path: str) -> pd.DataFrame:
    """Return one Mega-Cap performance file with parsed dates."""
    return pd.read_csv(path, parse_dates=["from_date", "thru_date"])


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
    )
    return float((1.0 + period_returns).prod() - 1.0)


def _risk_value(risk: pd.DataFrame, statistic: str, column: str) -> float:
    """Return one risk-statistics value."""
    row = risk[risk["column"].eq(statistic)]
    if row.empty:
        raise AssertionError(f"Missing risk statistic: {statistic}")
    return float(row[column].iloc[0])


if __name__ == "__main__":
    unittest.main()
