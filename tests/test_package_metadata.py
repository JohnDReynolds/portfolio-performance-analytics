"""Tests for package metadata maintained alongside runtime dependencies."""

# Python Imports
import subprocess
import sys
import tomllib
import unittest

# Project Imports
import ppar.utilities as util
from ppar import axys, performance_comparison
from ppar.axys import (
    AxysClassificationSources,
    AxysData,
    AxysPortfolio,
    AxysSpecification,
)
from ppar.performance_comparison import (
    CashLoader,
    ComparisonFile,
    ComparisonSnapshot,
    CONTEXT,
    DIRECT_INPUT,
    EVIDENCE_ROLE,
    Finding,
    FxRatesLoader,
    IMPACT_BASIS_PORTFOLIO_SOURCE_FIELD,
    IMPACT_BASIS_SECURITY_RETURN_WEIGHTED,
    IMPACT_METHOD_SECURITY_RETURN_DELTA_TIMES_WEIGHT,
    IMPACT_METHOD_SOURCE_FIELD_DELTA_OVER_BEGIN_MV,
    PerformanceComparison,
    PerformanceComparisonSpecification,
    PortfolioPerformanceLoader,
    PositionsLoader,
    PricesLoader,
    RELATED_OUTPUT,
    SecurityMasterLoader,
    SecurityPerformanceLoader,
    SuppressionRule,
    TARGET_OUTPUT,
    TransactionsLoader,
    apply_suppressions,
    columns,
    compact_findings_table,
    compare_snapshots,
    findings_to_polars,
    performance_comparison_html_report,
    performance_comparison_markdown_report,
    portfolio_period_cause_summary,
    portfolio_period_contribution_candidates,
    portfolio_period_evidence_breakdown,
    portfolio_period_impact_coverage_summary,
    portfolio_period_summary,
    rank_portfolio_period_evidence,
    security_period_evidence_breakdown,
    security_period_summary,
    summarize_findings,
    transaction_activity_summary,
    write_performance_comparison_html_report,
    write_performance_comparison_markdown_report,
    write_performance_comparison_report_bundle,
)


class TestPackageMetadata(unittest.TestCase):
    """Verify package dependency metadata agrees with development requirements."""

    def test_dependency_metadata(self) -> None:
        """Runtime dependencies are represented by the requirements file."""
        with open("pyproject.toml", "rb") as file:
            pyproject = tomllib.load(file)
        pyproject_dependencies = {
            dependency.split(">=", maxsplit=1)[0].lower()
            for dependency in pyproject["project"]["dependencies"]
        }
        with open("requirements.txt", "r", encoding=util.ENCODING) as file:
            requirements_dependencies = {
                line.split(">=", maxsplit=1)[0].strip().lower()
                for line in file
                if line.strip()
            }

        self.assertNotIn("great_tables", pyproject_dependencies)
        self.assertNotIn("great_tables", requirements_dependencies)
        self.assertIn("pyyaml", pyproject_dependencies)
        self.assertTrue(pyproject_dependencies.issubset(requirements_dependencies))

    def test_axys_package_is_included(self) -> None:
        """The Axys subpackage is included in distribution metadata."""
        with open("pyproject.toml", "rb") as file:
            pyproject = tomllib.load(file)

        self.assertIn("ppar.axys", pyproject["tool"]["setuptools"]["packages"])
        self.assertIn(
            "ppar.performance_comparison",
            pyproject["tool"]["setuptools"]["packages"],
        )

    def test_public_axys_import_contract(self) -> None:
        """The documented Axys package exports remain importable."""
        expected_exports = {
            "AxysClassificationSources",
            "AxysData",
            "AxysPortfolio",
            "AxysSpecification",
        }

        self.assertEqual(set(axys.__all__), expected_exports)
        self.assertIs(AxysClassificationSources, axys.AxysClassificationSources)
        self.assertIs(AxysData, axys.AxysData)
        self.assertIs(AxysPortfolio, axys.AxysPortfolio)
        self.assertIs(AxysSpecification, axys.AxysSpecification)

    def test_public_performance_comparison_import_contract(self) -> None:
        """The documented performance comparison exports remain importable."""
        expected_exports = {
            "CashLoader",
            "ComparisonFile",
            "ComparisonSnapshot",
            "CONTEXT",
            "DIRECT_INPUT",
            "EVIDENCE_ROLE",
            "Finding",
            "FxRatesLoader",
            "IMPACT_BASIS_PORTFOLIO_SOURCE_FIELD",
            "IMPACT_BASIS_SECURITY_RETURN_WEIGHTED",
            "IMPACT_METHOD_SECURITY_RETURN_DELTA_TIMES_WEIGHT",
            "IMPACT_METHOD_SOURCE_FIELD_DELTA_OVER_BEGIN_MV",
            "PerformanceComparison",
            "PortfolioPerformanceLoader",
            "PerformanceComparisonSpecification",
            "PositionsLoader",
            "PricesLoader",
            "SecurityPerformanceLoader",
            "SecurityMasterLoader",
            "SuppressionRule",
            "RELATED_OUTPUT",
            "TARGET_OUTPUT",
            "TransactionsLoader",
            "apply_suppressions",
            "columns",
            "compact_findings_table",
            "compare_snapshots",
            "findings_to_polars",
            "portfolio_period_cause_summary",
            "portfolio_period_contribution_candidates",
            "portfolio_period_evidence_breakdown",
            "portfolio_period_impact_coverage_summary",
            "portfolio_period_summary",
            "performance_comparison_html_report",
            "performance_comparison_markdown_report",
            "rank_portfolio_period_evidence",
            "security_period_evidence_breakdown",
            "security_period_summary",
            "summarize_findings",
            "transaction_activity_summary",
            "write_performance_comparison_html_report",
            "write_performance_comparison_markdown_report",
            "write_performance_comparison_report_bundle",
        }

        self.assertEqual(set(performance_comparison.__all__), expected_exports)
        self.assertIs(CashLoader, performance_comparison.CashLoader)
        self.assertIs(ComparisonFile, performance_comparison.ComparisonFile)
        self.assertIs(ComparisonSnapshot, performance_comparison.ComparisonSnapshot)
        self.assertIs(CONTEXT, performance_comparison.CONTEXT)
        self.assertIs(DIRECT_INPUT, performance_comparison.DIRECT_INPUT)
        self.assertIs(EVIDENCE_ROLE, performance_comparison.EVIDENCE_ROLE)
        self.assertIs(Finding, performance_comparison.Finding)
        self.assertIs(FxRatesLoader, performance_comparison.FxRatesLoader)
        self.assertIs(
            IMPACT_BASIS_PORTFOLIO_SOURCE_FIELD,
            performance_comparison.IMPACT_BASIS_PORTFOLIO_SOURCE_FIELD,
        )
        self.assertIs(
            IMPACT_BASIS_SECURITY_RETURN_WEIGHTED,
            performance_comparison.IMPACT_BASIS_SECURITY_RETURN_WEIGHTED,
        )
        self.assertIs(
            IMPACT_METHOD_SECURITY_RETURN_DELTA_TIMES_WEIGHT,
            performance_comparison.IMPACT_METHOD_SECURITY_RETURN_DELTA_TIMES_WEIGHT,
        )
        self.assertIs(
            IMPACT_METHOD_SOURCE_FIELD_DELTA_OVER_BEGIN_MV,
            performance_comparison.IMPACT_METHOD_SOURCE_FIELD_DELTA_OVER_BEGIN_MV,
        )
        self.assertIs(PerformanceComparison, performance_comparison.PerformanceComparison)
        self.assertIs(
            PerformanceComparisonSpecification,
            performance_comparison.PerformanceComparisonSpecification,
        )
        self.assertIs(
            PortfolioPerformanceLoader,
            performance_comparison.PortfolioPerformanceLoader,
        )
        self.assertIs(PositionsLoader, performance_comparison.PositionsLoader)
        self.assertIs(PricesLoader, performance_comparison.PricesLoader)
        self.assertIs(RELATED_OUTPUT, performance_comparison.RELATED_OUTPUT)
        self.assertIs(
            SecurityMasterLoader,
            performance_comparison.SecurityMasterLoader,
        )
        self.assertIs(
            SecurityPerformanceLoader,
            performance_comparison.SecurityPerformanceLoader,
        )
        self.assertIs(SuppressionRule, performance_comparison.SuppressionRule)
        self.assertIs(TARGET_OUTPUT, performance_comparison.TARGET_OUTPUT)
        self.assertIs(TransactionsLoader, performance_comparison.TransactionsLoader)
        self.assertIs(apply_suppressions, performance_comparison.apply_suppressions)
        self.assertIs(columns, performance_comparison.columns)
        self.assertIs(compact_findings_table, performance_comparison.compact_findings_table)
        self.assertIs(compare_snapshots, performance_comparison.compare_snapshots)
        self.assertIs(findings_to_polars, performance_comparison.findings_to_polars)
        self.assertIs(
            performance_comparison_html_report,
            performance_comparison.performance_comparison_html_report,
        )
        self.assertIs(
            performance_comparison_markdown_report,
            performance_comparison.performance_comparison_markdown_report,
        )
        self.assertIs(
            portfolio_period_cause_summary,
            performance_comparison.portfolio_period_cause_summary,
        )
        self.assertIs(
            portfolio_period_contribution_candidates,
            performance_comparison.portfolio_period_contribution_candidates,
        )
        self.assertIs(
            portfolio_period_evidence_breakdown,
            performance_comparison.portfolio_period_evidence_breakdown,
        )
        self.assertIs(
            portfolio_period_impact_coverage_summary,
            performance_comparison.portfolio_period_impact_coverage_summary,
        )
        self.assertIs(
            portfolio_period_summary,
            performance_comparison.portfolio_period_summary,
        )
        self.assertIs(
            rank_portfolio_period_evidence,
            performance_comparison.rank_portfolio_period_evidence,
        )
        self.assertIs(
            security_period_evidence_breakdown,
            performance_comparison.security_period_evidence_breakdown,
        )
        self.assertIs(
            security_period_summary,
            performance_comparison.security_period_summary,
        )
        self.assertIs(summarize_findings, performance_comparison.summarize_findings)
        self.assertIs(
            transaction_activity_summary,
            performance_comparison.transaction_activity_summary,
        )
        self.assertIs(
            write_performance_comparison_html_report,
            performance_comparison.write_performance_comparison_html_report,
        )
        self.assertIs(
            write_performance_comparison_markdown_report,
            performance_comparison.write_performance_comparison_markdown_report,
        )
        self.assertIs(
            write_performance_comparison_report_bundle,
            performance_comparison.write_performance_comparison_report_bundle,
        )

    def test_chart_dependencies_are_optional(self) -> None:
        """Normal package imports do not load optional chart rendering code."""
        with open("pyproject.toml", "rb") as file:
            pyproject = tomllib.load(file)

        chart_dependencies = {
            dependency.split(">=", maxsplit=1)[0].lower()
            for dependency in pyproject["project"]["optional-dependencies"]["charts"]
        }
        core_dependencies = {
            dependency.split(">=", maxsplit=1)[0].lower()
            for dependency in pyproject["project"]["dependencies"]
        }
        command = (
            "import sys; import ppar; "
            "raise SystemExit(1 if 'ppar.format_chart' in sys.modules else 0)"
        )

        self.assertEqual(chart_dependencies, {"matplotlib", "seaborn"})
        self.assertTrue(chart_dependencies.isdisjoint(core_dependencies))
        subprocess.run([sys.executable, "-c", command], check=True)


if __name__ == "__main__":
    unittest.main()
