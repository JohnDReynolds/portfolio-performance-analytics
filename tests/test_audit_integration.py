"""Focused integration smoke tests for calculated object audits."""

# Overrides for pylint and pylance
# pylint: disable=protected-access
# pyright: reportPrivateUsage=false

# Python Imports
import unittest

# Test Imports
from tests import test_utilities as test_util

# Project Imports
from ppar.analytics import Analytics
from ppar.attribution import View
from ppar.frequency import Frequency
from ppar.html_table import HtmlTable


class TestAuditIntegration(unittest.TestCase):
    """Exercise representative calculated-object audit pathways."""

    def test_daily_attribution_views_and_analytics_audit(self) -> None:
        """Daily calculations audit raw mapped results and cached attributions."""
        analytics = Analytics(
            test_util.performance_data_path("aapl_daily"),
            test_util.performance_data_path("mag7_daily"),
            portfolio_classification_name="Security",
            benchmark_classification_name="Security",
            frequency=Frequency.AS_OFTEN_AS_POSSIBLE,
        )

        test_util.get_attribution(analytics, "Security")
        economic_sector = test_util.get_attribution(analytics, "Economic Sector")

        for view in View:
            economic_sector._audit_view(view)

        analytics.audit()

    def test_monthly_consolidated_attribution_views_audit(self) -> None:
        """Monthly calculations audit consolidated mapped result views."""
        analytics = Analytics(
            test_util.performance_data_path("aapl_daily"),
            test_util.performance_data_path("mag7_daily"),
            portfolio_classification_name="Security",
            benchmark_classification_name="Security",
            frequency=Frequency.MONTHLY,
        )

        economic_sector = test_util.get_attribution(analytics, "Economic Sector")

        for view in View:
            economic_sector._audit_view(view)

        analytics.audit()

    def test_quarterly_security_views_and_risk_statistics_audit(self) -> None:
        """Quarterly calculations audit security views and risk statistics."""
        analytics = Analytics(
            test_util.performance_data_path("Big 2"),
            test_util.performance_data_path("Large-Cap Portfolio"),
            portfolio_classification_name="Security",
            benchmark_classification_name="Security",
            frequency=Frequency.QUARTERLY,
        )

        security = test_util.get_attribution(analytics, "Security")
        test_util.get_attribution(analytics, "Economic Sector")

        for view in View:
            security._audit_view(view)

        risk_statistics = analytics.get_riskstatistics()
        risk_statistics._audit()
        self.assertIsInstance(risk_statistics.to_table(), HtmlTable)

        analytics.audit()


if __name__ == "__main__":
    unittest.main()
