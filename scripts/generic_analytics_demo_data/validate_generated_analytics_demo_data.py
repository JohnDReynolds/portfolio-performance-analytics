"""Validate generated Mega-Cap analytics demo data through ppar APIs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ppar.analytics import Analytics
from ppar.analytics.attribution import View
from ppar.analytics.frequency import Frequency


WORKSPACE = Path("_demo_output") / "generic_analytics_data_generation"
GENERATED = WORKSPACE / "generated_oef_files"


def main() -> None:
    """Run a compact package-level validation for generated demo files."""
    portfolio_path = GENERATED / "performance" / "Generated OEF Alpha Portfolio.csv"
    benchmark_path = GENERATED / "performance" / "Generated OEF Benchmark.csv"
    sector_path = GENERATED / "classifications" / "Generated OEF Economic Sector.csv"
    mapping_path = (
        GENERATED
        / "mappings"
        / "Generated OEF Security--to--Generated OEF Economic Sector.csv"
    )

    analytics = Analytics(
        portfolio_path,
        benchmark_path,
        portfolio_classification_name="Security",
        benchmark_classification_name="Security",
        frequency=Frequency.MONTHLY,
    )
    security_attribution = analytics.get_attribution()
    security_overall = security_attribution.to_pandas(View.OVERALL_ATTRIBUTION)

    sector_attribution = analytics.get_attribution(
        "Economic Sector",
        sector_path,
        (mapping_path, mapping_path),
    )
    sector_overall = sector_attribution.to_pandas(View.OVERALL_ATTRIBUTION)
    risk = analytics.get_riskstatistics().to_pandas()

    portfolio_return = _cumulative_performance_return(portfolio_path)
    benchmark_return = _cumulative_performance_return(benchmark_path)
    portfolio_sharpe = _risk_value(risk, "Annualized Sharpe Ratio", "Portfolio")
    benchmark_sharpe = _risk_value(risk, "Annualized Sharpe Ratio", "Benchmark")

    if portfolio_return <= benchmark_return:
        raise SystemExit("Expected generated portfolio return to exceed benchmark return.")
    if portfolio_sharpe <= benchmark_sharpe:
        raise SystemExit("Expected generated portfolio Sharpe ratio to exceed benchmark.")
    if security_overall.empty:
        raise SystemExit("Security attribution output is empty.")
    if sector_overall.empty:
        raise SystemExit("Sector attribution output is empty.")

    print("Generated analytics data validation passed.")
    print(f"Portfolio cumulative return: {portfolio_return:.6f}")
    print(f"Benchmark cumulative return: {benchmark_return:.6f}")
    print(f"Portfolio Sharpe ratio: {portfolio_sharpe:.6f}")
    print(f"Benchmark Sharpe ratio: {benchmark_sharpe:.6f}")
    print(f"Security attribution rows: {len(security_overall)}")
    print(f"Sector attribution rows: {len(sector_overall)}")


def _risk_value(risk, statistic: str, column: str) -> float:
    """Return one risk-statistics value by row label and output column."""
    row = risk[risk["column"] == statistic]
    if row.empty:
        raise SystemExit(f"Risk output is missing statistic: {statistic}")
    return float(row[column].iloc[0])


def _cumulative_performance_return(path: Path) -> float:
    """Return cumulative weighted return from one generated performance file."""
    frame = pd.read_csv(path)
    period_returns = (
        frame.assign(contribution=frame["weight"] * frame["return"])
        .groupby(["from_date", "thru_date"], sort=True)["contribution"]
        .sum()
    )
    return float((1.0 + period_returns).prod() - 1.0)


if __name__ == "__main__":
    main()
