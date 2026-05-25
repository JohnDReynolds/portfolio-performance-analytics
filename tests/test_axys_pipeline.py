"""Focused tests for the AxysData-to-Analytics pipeline using temporary inputs."""

# Python Imports
import datetime as dt
import math
from pathlib import Path
import tempfile
import unittest

# Third-Party Imports
import polars as pl
import yaml

# Project Imports
from ppar.analytics import Analytics
from ppar.attribution import View
from ppar.axys import AxysData
import ppar.columns as cols


def _write_axys_inputs(directory: Path) -> Path:
    """Write minimal Axys-like sources into a temporary test directory."""
    pl.DataFrame(
        {
            "FROM_DATE": ["2023-12-31", "2024-01-31", "2023-12-31"],
            "THRU_DATE": ["2024-01-31", "2024-02-29", "2024-01-31"],
            "PORTFOLIO_CODE": ["P1", "P1", "P2"],
            "PORTFOLIO_NAME": ["Growth", "Growth", "Income"],
            "PORT_RETURN": [0.04, 0.03, 0.02],
        }
    ).write_csv(directory / "portperf.csv")
    pl.DataFrame(
        {
            "FROM_DATE": [
                "2023-12-31",
                "2023-12-31",
                "2024-01-31",
                "2024-01-31",
                "2023-12-31",
            ],
            "THRU_DATE": [
                "2024-01-31",
                "2024-01-31",
                "2024-02-29",
                "2024-02-29",
                "2024-01-31",
            ],
            "PORTFOLIO_CODE": ["P1", "P1", "P1", "P1", "P2"],
            "SECURITY_ID": ["A", "B", "A", "B", "C"],
            "SEC_RETURN": [0.10, -0.05, 0.04, 0.015, 0.02],
            "BEGIN_WEIGHT": [0.50, 0.50, 0.50, 0.50, 1.00],
            "CONTRIBUTION": [0.06, -0.02, 0.024, 0.006, 0.02],
        }
    ).write_csv(directory / "secperf.csv")
    pl.DataFrame(
        {
            "SECURITY_ID": ["A", "B", "C", "UNUSED"],
            "SECURITY_NAME": ["Alpha", "Beta", "Cash", "Unused"],
            "SECTOR_CODE": ["TECH", "DEF", "CASH", "OTHER"],
        }
    ).write_csv(directory / "security_master.csv")
    pl.DataFrame(
        {
            "CODE": ["TECH", "DEF", "CASH", "OTHER"],
            "DESCRIPTION": ["Technology", "Defensive", "Cash", "Other"],
            "TYPE": ["SECTOR", "SECTOR", "SECTOR", "SECTOR"],
        }
    ).write_csv(directory / "classifications.csv")
    specification: dict[str, object] = {
        "settings": {"prefix_portfolio_code": " - "},
        "portperf_path": "portperf.csv",
        "portperf_columns": {
            cols.BEGINNING_DATE: "FROM_DATE",
            cols.ENDING_DATE: "THRU_DATE",
            cols.PORTFOLIO_CODE: "PORTFOLIO_CODE",
            cols.PORTFOLIO_NAME: "PORTFOLIO_NAME",
            cols.PORTFOLIO_RETURN: "PORT_RETURN",
        },
        "secperf_path": "secperf.csv",
        "secperf_columns": {
            cols.BEGINNING_DATE: "FROM_DATE",
            cols.ENDING_DATE: "THRU_DATE",
            cols.IDENTIFIER: "SECURITY_ID",
            cols.PORTFOLIO_CODE: "PORTFOLIO_CODE",
            cols.RETURN: "SEC_RETURN",
            cols.WEIGHT: "BEGIN_WEIGHT",
            cols.CONTRIBUTION: "CONTRIBUTION",
        },
        "classifications": {
            "Security": {
                "file_path": "security_master.csv",
                "identifier_column": "SECURITY_ID",
                "name_column": "SECURITY_NAME",
                "is_security_master": True,
            },
            "Sector": {
                "file_path": "classifications.csv",
                "identifier_column": "CODE",
                "name_column": "DESCRIPTION",
                "filter_column": "TYPE",
                "filter_value": "SECTOR",
            },
        },
        "mappings": {
            "SecurityToSector": {
                "file_path": "security_master.csv",
                "identifier_column": "SECURITY_ID",
                "name_column": "SECTOR_CODE",
                "is_security_master": True,
            }
        },
    }
    specification_path = directory / "axysdata.yaml"
    specification_path.write_text(yaml.safe_dump(specification), encoding="utf-8")
    return specification_path


class TestAxysPipeline(unittest.TestCase):
    """Verify successful Axys loading and downstream attribution behavior."""

    def test_load_reconciles_weights_and_filters_security_sources(self) -> None:
        """Selected security sources and performance are ready for Analytics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data = AxysData(_write_axys_inputs(Path(temp_dir)), portfolio_codes="P1")

            performance = data.portfolios["P1"].secperf
            security = data.classification_data_sources["Security"]
            mapping = data.mapping_data_sources["SecurityToSector"]

            self.assertEqual(data.portfolios["P1"].portfolio_name, "P1 - Growth")
            self.assertEqual(security["identifier_column"].sort().to_list(), ["A", "B"])
            self.assertEqual(mapping["identifier_column"].sort().to_list(), ["A", "B"])
            first_period_weights = performance.filter(
                pl.col(cols.ENDING_DATE) == dt.date(2024, 1, 31)
            )[cols.WEIGHT].to_list()
            self.assertTrue(math.isclose(first_period_weights[0], 0.60, abs_tol=1e-12))
            self.assertTrue(math.isclose(first_period_weights[1], 0.40, abs_tol=1e-12))

    def test_omitted_portfolio_codes_loads_all_available_portfolios(self) -> None:
        """Constructor discovery builds output for each available portfolio code."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data = AxysData(_write_axys_inputs(Path(temp_dir)))

            self.assertEqual(set(data.portfolios), {"P1", "P2"})
            self.assertEqual(data.portfolios["P2"].portfolio_name, "P2 - Income")

    def test_date_filters_apply_before_returning_portfolio_performance(self) -> None:
        """Axys date arguments restrict the periods retained for a portfolio."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data = AxysData(
                _write_axys_inputs(Path(temp_dir)),
                portfolio_codes="P1",
                from_date=dt.date(2024, 1, 31),
                thru_date=dt.date(2024, 2, 29),
            )

            performance = data.portfolios["P1"].secperf

            self.assertEqual(performance.height, 2)
            self.assertEqual(
                performance[cols.ENDING_DATE].unique().to_list(),
                [dt.date(2024, 2, 29)],
            )

    def test_axys_sources_roll_up_through_analytics_to_sector_attribution(self) -> None:
        """Generated classification and mapping sources drive public attribution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data = AxysData(_write_axys_inputs(Path(temp_dir)), portfolio_codes="P1")
            portfolio = data.portfolios["P1"]
            analytics = Analytics(
                portfolio.secperf,
                portfolio_name=portfolio.portfolio_name,
                portfolio_classification_name="Security",
            )

            attribution = analytics.get_attribution(
                "Sector",
                data.classification_data_sources["Sector"],
                (
                    data.mapping_data_sources["SecurityToSector"],
                    data.mapping_data_sources["SecurityToSector"],
                ),
            )
            detail = attribution.to_polars(View.SUBPERIOD_ATTRIBUTION)

            self.assertEqual(
                set(detail[cols.CLASSIFICATION_NAME].to_list()),
                {"Defensive", "Technology"},
            )
            self.assertTrue((detail[cols.ACTIVE_RETURN] == 0.0).all())
            self.assertTrue((detail[cols.TOTAL_EFFECT_SIMPLE] == 0.0).all())


if __name__ == "__main__":
    unittest.main()
