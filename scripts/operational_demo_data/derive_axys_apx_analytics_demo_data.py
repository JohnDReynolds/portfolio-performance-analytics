"""Derive Axys/APX-shaped analytics demo files from canonical Mega-Cap CSVs."""

from __future__ import annotations

# Python imports
from pathlib import Path

# Third-party imports
import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parents[2]
_GENERIC_ANALYTICS_DIR = _REPO_ROOT / "ppar" / "setup_templates" / "generic_analytics"
_PERFORMANCE_DIR = _GENERIC_ANALYTICS_DIR / "performance"
_CLASSIFICATION_DIR = _GENERIC_ANALYTICS_DIR / "classifications"
_MAPPING_DIR = _GENERIC_ANALYTICS_DIR / "mappings"
_OUTPUT_DIR = _REPO_ROOT / "ppar" / "setup_templates" / "axys_apx_analytics"
_PORTFOLIOS = {
    "MEGA_ALPHA": {
        "file_name": "Mega-Cap Alpha Portfolio.csv",
        "portfolio_name": "Mega-Cap Alpha Portfolio",
    },
    "MEGA_BENCH": {
        "file_name": "Mega-Cap Benchmark.csv",
        "portfolio_name": "Mega-Cap Benchmark",
    },
}


def main() -> None:
    """Write Axys/APX-shaped analytics demo files into the packaged demo tree."""
    output_paths = write_axys_apx_analytics_demo_data(_OUTPUT_DIR)
    print("Axys/APX analytics demo files written:")
    for path in output_paths:
        print(f"- {path}")


def write_axys_apx_analytics_demo_data(output_directory: Path) -> list[Path]:
    """Write Axys/APX-shaped analytics demo files.

    Args:
        output_directory: Directory that will receive the generated CSV files.

    Returns:
        Paths written by the generator.
    """
    performance_by_portfolio = {
        portfolio_code: _read_performance(config["file_name"])
        for portfolio_code, config in _PORTFOLIOS.items()
    }
    security_reference = _security_reference(performance_by_portfolio)
    portperf = _portfolio_performance(performance_by_portfolio)
    secperf = _security_performance(performance_by_portfolio, security_reference)

    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "portperf": output_directory / "portperf.csv",
        "secperf": output_directory / "secperf.csv",
    }
    portperf.to_csv(paths["portperf"], index=False)
    secperf.to_csv(paths["secperf"], index=False)
    return [paths["portperf"], paths["secperf"]]


def _read_performance(file_name: str) -> pd.DataFrame:
    """Return one canonical performance file with parsed date columns."""
    return pd.read_csv(
        _PERFORMANCE_DIR / file_name,
        parse_dates=["from_date", "thru_date"],
    )


def _security_reference(performance_by_portfolio: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return an Axys/APX-style security reference file."""
    security_names = pd.read_csv(
        _CLASSIFICATION_DIR / "Security.csv",
        header=None,
        names=["SECURITY_ID", "SECURITY_NAME"],
    )
    sectors = pd.read_csv(
        _CLASSIFICATION_DIR / "Economic Sector.csv",
        header=None,
        names=["SECTOR_CODE", "SECTOR_DESC"],
    )
    mappings = pd.read_csv(
        _MAPPING_DIR / "Security--to--Economic Sector.csv",
        header=None,
        names=["SECURITY_ID", "SECTOR_CODE"],
    )
    identifiers = sorted(
        {
            identifier
            for performance in performance_by_portfolio.values()
            for identifier in performance["identifier"].unique()
        }
    )
    return (
        pd.DataFrame({"SECURITY_ID": identifiers})
        .merge(security_names, on="SECURITY_ID", how="left")
        .merge(mappings, on="SECURITY_ID", how="left")
        .merge(sectors, on="SECTOR_CODE", how="left")
        .assign(
            ASSET_CLASS_CODE=lambda frame: frame["SECTOR_CODE"].map(_asset_class_code),
            ASSET_CLASS_DESC=lambda frame: frame["ASSET_CLASS_CODE"].map(
                {
                    "CASH": "Cash",
                    "EQ": "Equity",
                }
            ),
            COUNTRY_CODE="US",
            COUNTRY_DESC="United States",
            CURRENCY_CODE="USD",
            CURRENCY_DESC="US Dollar",
        )
        [
            [
                "SECURITY_ID",
                "SECURITY_NAME",
                "ASSET_CLASS_CODE",
                "ASSET_CLASS_DESC",
                "SECTOR_CODE",
                "SECTOR_DESC",
                "COUNTRY_CODE",
                "COUNTRY_DESC",
                "CURRENCY_CODE",
                "CURRENCY_DESC",
            ]
        ]
        .sort_values("SECURITY_ID")
    )


def _portfolio_performance(
    performance_by_portfolio: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Return Axys/APX-style portfolio performance rows."""
    frames: list[pd.DataFrame] = []
    for portfolio_code, performance in performance_by_portfolio.items():
        portfolio_name = _PORTFOLIOS[portfolio_code]["portfolio_name"]
        period_returns = (
            performance.assign(CONTRIBUTION=performance["weight"] * performance["return"])
            .groupby(["from_date", "thru_date"], as_index=False)["CONTRIBUTION"]
            .sum()
            .rename(columns={"CONTRIBUTION": "PORT_RETURN"})
        )
        frames.append(
            period_returns.assign(
                PORTFOLIO_CODE=portfolio_code,
                PORTFOLIO_NAME=portfolio_name,
            )
        )
    return (
        pd.concat(frames, ignore_index=True)
        .rename(columns={"from_date": "FROM_DATE", "thru_date": "THRU_DATE"})
        [["FROM_DATE", "THRU_DATE", "PORTFOLIO_CODE", "PORTFOLIO_NAME", "PORT_RETURN"]]
        .sort_values(["PORTFOLIO_CODE", "FROM_DATE", "THRU_DATE"])
    )


def _security_performance(
    performance_by_portfolio: dict[str, pd.DataFrame],
    security_reference: pd.DataFrame,
) -> pd.DataFrame:
    """Return Axys/APX-style security performance rows."""
    frames: list[pd.DataFrame] = []
    reference_columns = [
        "SECURITY_ID",
        "ASSET_CLASS_CODE",
        "ASSET_CLASS_DESC",
        "SECTOR_CODE",
        "SECTOR_DESC",
        "COUNTRY_CODE",
        "COUNTRY_DESC",
        "CURRENCY_CODE",
        "CURRENCY_DESC",
    ]
    for portfolio_code, performance in performance_by_portfolio.items():
        portfolio_name = _PORTFOLIOS[portfolio_code]["portfolio_name"]
        frame = (
            performance.rename(
                columns={
                    "from_date": "FROM_DATE",
                    "thru_date": "THRU_DATE",
                    "identifier": "SECURITY_ID",
                    "name": "SECURITY_NAME",
                    "weight": "BEGIN_WEIGHT",
                    "return": "SEC_RETURN",
                }
            )
            .assign(
                PORTFOLIO_CODE=portfolio_code,
                PORTFOLIO_NAME=portfolio_name,
                CONTRIBUTION=lambda data: data["BEGIN_WEIGHT"] * data["SEC_RETURN"],
            )
            .merge(security_reference[reference_columns], on="SECURITY_ID", how="left")
        )
        frames.append(frame)
    return (
        pd.concat(frames, ignore_index=True)[
            [
                "FROM_DATE",
                "THRU_DATE",
                "PORTFOLIO_CODE",
                "PORTFOLIO_NAME",
                "SECURITY_ID",
                "SECURITY_NAME",
                "BEGIN_WEIGHT",
                "SEC_RETURN",
                "CONTRIBUTION",
                "ASSET_CLASS_CODE",
                "ASSET_CLASS_DESC",
                "SECTOR_CODE",
                "SECTOR_DESC",
                "COUNTRY_CODE",
                "COUNTRY_DESC",
                "CURRENCY_CODE",
                "CURRENCY_DESC",
            ]
        ]
        .sort_values(["PORTFOLIO_CODE", "FROM_DATE", "THRU_DATE", "SECURITY_ID"])
    )


def _asset_class_code(sector_code: object) -> str:
    """Return the broad asset-class code for a sector code."""
    return "CASH" if sector_code == "CA" else "EQ"


if __name__ == "__main__":
    main()
