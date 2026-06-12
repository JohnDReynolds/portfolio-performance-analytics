"""Tests for package metadata maintained alongside runtime dependencies."""

# Python Imports
import ast
from fnmatch import fnmatch
from importlib.resources import files
from pathlib import Path
import subprocess
import sys
import tomllib
import unittest

# Project Imports
import ppar.columns as core_columns
import ppar.errors as core_errors
import ppar.utilities as util
from ppar import axys, performance_comparison
from ppar.axys import (
    AxysClassificationSources,
    AxysData,
    AxysPortfolio,
    AxysSpecification,
)
from ppar.performance_comparison import runner as performance_comparison_runner
from ppar.performance_comparison import report as performance_comparison_report
from ppar.performance_comparison import findings as performance_comparison_findings
from ppar.performance_comparison import (
    transactions as performance_comparison_transactions,
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
    portfolio_period_flow_cross_check_reconciliation,
    portfolio_period_impact_coverage_summary,
    portfolio_period_summary,
    portfolio_period_transaction_cross_checks,
    rank_portfolio_period_evidence,
    security_period_evidence_breakdown,
    security_period_summary,
    summarize_findings,
    transaction_activity_summary,
    write_performance_comparison_html_report,
    write_performance_comparison_markdown_report,
    write_performance_comparison_report_bundle,
)


def _is_type_alias_assignment(node: ast.AnnAssign) -> bool:
    """Return whether an annotated assignment declares a public type alias."""
    annotation = node.annotation
    if isinstance(annotation, ast.Name):
        return annotation.id == "TypeAlias"
    return isinstance(annotation, ast.Attribute) and annotation.attr == "TypeAlias"


def _declared_public_module_names(path: Path) -> set[str]:
    """Return public constants, type aliases, functions, and classes in a module."""
    tree = ast.parse(path.read_text(encoding=util.ENCODING))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if not isinstance(target, ast.Name) or target.id.startswith("_"):
                    continue
                is_public_constant = target.id.isupper()
                is_public_type_alias = (
                    isinstance(node, ast.AnnAssign) and _is_type_alias_assignment(node)
                )
                if is_public_constant or is_public_type_alias:
                    names.add(target.id)
        elif (
            isinstance(node, (ast.ClassDef, ast.FunctionDef))
            and not node.name.startswith("_")
        ):
            names.add(node.name)
    return names


class TestPackageMetadata(unittest.TestCase):
    """Verify package dependency metadata is complete and intentionally scoped."""

    def test_dependency_metadata(self) -> None:
        """Runtime dependencies and optional tooling stay in their metadata groups."""
        with open("pyproject.toml", "rb") as file:
            pyproject = tomllib.load(file)
        pyproject_dependencies = {
            dependency.split(">=", maxsplit=1)[0].lower()
            for dependency in pyproject["project"]["dependencies"]
        }
        optional_dependencies = pyproject["project"]["optional-dependencies"]
        chart_dependencies = {
            dependency.split(">=", maxsplit=1)[0].lower()
            for dependency in optional_dependencies["charts"]
        }
        dev_dependencies = {
            dependency.split(">=", maxsplit=1)[0].lower()
            for dependency in optional_dependencies["dev"]
        }

        self.assertNotIn("great_tables", pyproject_dependencies)
        self.assertIn("pyyaml", pyproject_dependencies)
        self.assertNotIn("matplotlib", pyproject_dependencies)
        self.assertIn("matplotlib", chart_dependencies)
        self.assertIn("seaborn", chart_dependencies)
        self.assertIn("pytest", dev_dependencies)

    def test_distribution_metadata_includes_license_and_build_backend(self) -> None:
        """Build metadata uses current setuptools license fields."""
        with open("pyproject.toml", "rb") as file:
            pyproject = tomllib.load(file)

        self.assertEqual(pyproject["build-system"]["requires"], ["setuptools>=77.0.0"])
        self.assertEqual(pyproject["project"]["license"], "LicenseRef-Proprietary")
        self.assertEqual(pyproject["project"]["license-files"], ["LICENSE"])

    def test_manifest_keeps_source_distribution_resources(self) -> None:
        """The source distribution manifest includes checkout scripts and demo data."""
        manifest = Path("MANIFEST.in").read_text(encoding=util.ENCODING)

        self.assertIn("include scripts/*.py", manifest)
        self.assertIn("recursive-include ppar/demo_data *.csv *.yaml *.md", manifest)
        self.assertNotIn("prune scripts", manifest)

    def test_checkout_scripts_are_sdist_only(self) -> None:
        """Checkout utility scripts are shipped in sdist but not as wheel packages."""
        with open("pyproject.toml", "rb") as file:
            pyproject = tomllib.load(file)
        manifest = Path("MANIFEST.in").read_text(encoding=util.ENCODING)

        self.assertIn("include scripts/*.py", manifest)
        self.assertNotIn("scripts", pyproject["tool"]["setuptools"]["packages"])

    def test_package_data_patterns_cover_demo_resources(self) -> None:
        """Every packaged demo resource is covered by explicit package-data globs."""
        with open("pyproject.toml", "rb") as file:
            pyproject = tomllib.load(file)
        package_data_patterns = pyproject["tool"]["setuptools"]["package-data"]["ppar"]
        demo_resource_paths = [
            path.relative_to("ppar").as_posix()
            for path in Path("ppar/demo_data").rglob("*")
            if path.is_file()
        ]

        self.assertGreater(len(demo_resource_paths), 0)
        for resource_path in demo_resource_paths:
            with self.subTest(resource_path=resource_path):
                self.assertTrue(
                    any(fnmatch(resource_path, pattern) for pattern in package_data_patterns),
                    f"{resource_path} is not covered by package-data patterns.",
                )

    def test_axys_package_is_included(self) -> None:
        """The Axys subpackage is included in distribution metadata."""
        with open("pyproject.toml", "rb") as file:
            pyproject = tomllib.load(file)

        self.assertIn("ppar.axys", pyproject["tool"]["setuptools"]["packages"])
        self.assertIn("ppar.demos", pyproject["tool"]["setuptools"]["packages"])
        self.assertIn(
            "ppar.performance_comparison",
            pyproject["tool"]["setuptools"]["packages"],
        )

    def test_demo_console_scripts_are_explicit(self) -> None:
        """The installed demo commands point to the packaged demo modules."""
        with open("pyproject.toml", "rb") as file:
            pyproject = tomllib.load(file)

        self.assertEqual(
            pyproject["project"]["scripts"],
            {
                "ppar-analytics-demo": "ppar.demos.analytics_demo:main",
                "ppar-axys-analytics-demo": "ppar.demos.axys_analytics_demo:main",
                "ppar-performance-comparison-demo": (
                    "ppar.demos.performance_comparison_demo:main"
                ),
            },
        )

    def test_axys_demo_resources_are_packaged(self) -> None:
        """The Axys demos use packaged resources instead of test fixtures."""
        axys_demo_data = files("ppar.demo_data") / "axys"

        self.assertTrue((axys_demo_data / "axys_column_mappings.yaml").is_file())
        self.assertTrue(
            (axys_demo_data / "ppar_performance_comparison_restatement.yaml").is_file()
        )
        self.assertTrue((axys_demo_data / "axys_a" / "portperf.csv").is_file())
        self.assertTrue((axys_demo_data / "axys_b_restatement" / "secperf.csv").is_file())

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
            "portfolio_period_flow_cross_check_reconciliation",
            "portfolio_period_impact_coverage_summary",
            "portfolio_period_summary",
            "portfolio_period_transaction_cross_checks",
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
            portfolio_period_flow_cross_check_reconciliation,
            performance_comparison.portfolio_period_flow_cross_check_reconciliation,
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
            portfolio_period_transaction_cross_checks,
            performance_comparison.portfolio_period_transaction_cross_checks,
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

    def test_public_performance_comparison_runner_import_contract(self) -> None:
        """The runner module exposes only the compact workflow helper surface."""
        expected_exports = {
            "compact_findings_table",
            "compare_snapshots",
            "summarize_findings",
        }

        self.assertEqual(set(performance_comparison_runner.__all__), expected_exports)
        self.assertIs(
            compact_findings_table,
            performance_comparison_runner.compact_findings_table,
        )
        self.assertIs(compare_snapshots, performance_comparison_runner.compare_snapshots)
        self.assertIs(summarize_findings, performance_comparison_runner.summarize_findings)

    def test_public_performance_comparison_report_import_contract(self) -> None:
        """The report module exposes only report rendering and writing helpers."""
        expected_exports = {
            "performance_comparison_html_report",
            "performance_comparison_markdown_report",
            "write_performance_comparison_html_report",
            "write_performance_comparison_markdown_report",
            "write_performance_comparison_report_bundle",
        }

        self.assertEqual(set(performance_comparison_report.__all__), expected_exports)
        self.assertIs(
            performance_comparison_html_report,
            performance_comparison_report.performance_comparison_html_report,
        )
        self.assertIs(
            performance_comparison_markdown_report,
            performance_comparison_report.performance_comparison_markdown_report,
        )
        self.assertIs(
            write_performance_comparison_html_report,
            performance_comparison_report.write_performance_comparison_html_report,
        )
        self.assertIs(
            write_performance_comparison_markdown_report,
            performance_comparison_report.write_performance_comparison_markdown_report,
        )
        self.assertIs(
            write_performance_comparison_report_bundle,
            performance_comparison_report.write_performance_comparison_report_bundle,
        )
        self.assertNotIn(
            "_report_bundle_validation_issues",
            performance_comparison_report.__all__,
        )

    def test_performance_comparison_vocabulary_exports_are_explicit(self) -> None:
        """Vocabulary modules export every declared public schema/code name."""
        module_paths = {
            columns: Path("ppar/performance_comparison/columns.py"),
            performance_comparison_findings: Path(
                "ppar/performance_comparison/findings.py"
            ),
            performance_comparison_transactions: Path(
                "ppar/performance_comparison/transactions.py"
            ),
        }

        for module, path in module_paths.items():
            with self.subTest(module=module.__name__):
                self.assertEqual(
                    set(module.__all__),
                    _declared_public_module_names(path),
                )

    def test_core_public_exports_are_explicit(self) -> None:
        """Core helper modules export every intentional public module name."""
        module_paths = {
            core_columns: Path("ppar/columns.py"),
            core_errors: Path("ppar/errors.py"),
            util: Path("ppar/utilities.py"),
        }

        for module, path in module_paths.items():
            with self.subTest(module=module.__name__):
                self.assertEqual(
                    set(module.__all__),
                    _declared_public_module_names(path),
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
