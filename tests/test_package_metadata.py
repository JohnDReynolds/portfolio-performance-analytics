"""Tests for package metadata maintained alongside runtime dependencies."""

# Python Imports
import ast
import csv
from fnmatch import fnmatch
import importlib.util
from importlib.resources import files
from pathlib import Path
import subprocess
import sys
import tomllib
import unittest

# Project Imports
import ppar.analytics.schema as core_schema
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
from ppar.performance_comparison.methods import (
    CashImpactMethod,
    ContributionImpactMethod,
    FxRateImpactMethod,
    PositionImpactMethod,
    SecurityMasterImpactMethod,
    TransactionImpactMethod,
)
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
    REPORT_BUNDLE_REQUIRED_ARTIFACTS,
    RELATED_OUTPUT,
    SecurityMasterLoader,
    SecurityPerformanceLoader,
    SuppressionRule,
    TARGET_OUTPUT,
    TransactionsLoader,
    apply_suppressions,
    schema,
    compact_findings_table,
    compare_snapshots,
    findings_to_polars,
    portfolio_period_cause_summary,
    portfolio_period_contribution_candidates,
    portfolio_period_evidence_breakdown,
    portfolio_period_flow_cross_check_reconciliation,
    portfolio_period_impact_coverage_summary,
    portfolio_period_summary,
    portfolio_period_transaction_cross_checks,
    rank_portfolio_period_evidence,
    report_bundle_validation_issues,
    security_period_evidence_breakdown,
    security_period_summary,
    summarize_findings,
    transaction_activity_summary,
    transaction_matching_diagnostics,
    validate_causal_attribution_ready,
    write_performance_comparison_report_bundle,
    write_performance_comparison_review_workbook,
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
        self.assertIn("openpyxl", pyproject_dependencies)
        self.assertNotIn("matplotlib", pyproject_dependencies)
        self.assertIn("matplotlib", chart_dependencies)
        self.assertIn("seaborn", chart_dependencies)
        self.assertNotIn("excel", optional_dependencies)
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
        self.assertIn("recursive-include ppar/demos/data *.csv *.yaml *.md", manifest)
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
            for path in Path("ppar/demos/data").rglob("*")
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
        self.assertIn("ppar.analytics", pyproject["tool"]["setuptools"]["packages"])
        self.assertIn("ppar.demos", pyproject["tool"]["setuptools"]["packages"])
        self.assertIn(
            "ppar.performance_comparison",
            pyproject["tool"]["setuptools"]["packages"],
        )
        self.assertIn(
            "ppar.performance_comparison.cli",
            pyproject["tool"]["setuptools"]["packages"],
        )

    def test_console_scripts_are_explicit(self) -> None:
        """Installed commands point to packaged modules."""
        with open("pyproject.toml", "rb") as file:
            pyproject = tomllib.load(file)

        self.assertEqual(
            pyproject["project"]["scripts"],
            {
                "ppar-analytics-demo": "ppar.demos.analytics_demo:main",
                "ppar-axys-analytics-demo": "ppar.demos.axys_analytics_demo:main",
                "ppar-performance-comparison-portfolio-demo": (
                    "ppar.demos.performance_comparison_portfolio_demo:main"
                ),
                "ppar-performance-comparison-security-demo": (
                    "ppar.demos.performance_comparison_security_demo:main"
                ),
            },
        )

    def test_axys_demo_resources_are_packaged(self) -> None:
        """The Axys demos use packaged resources instead of test fixtures."""
        axys_demo_data = files("ppar.demos.data") / "axys"
        expected_resources = (
            "README.md",
            "axys_column_mappings.yaml",
            "axys_analytics.yaml",
            "ppar_performance_comparison_full_spec.yaml",
            "ppar_performance_comparison_security_full_spec.yaml",
            "axys_analytics/portperf.csv",
            "axys_analytics/secperf.csv",
            "axys_analytics/sec_ref.csv",
            "axys_full_spec_a/portperf.csv",
            "axys_full_spec_a/sec_ref.csv",
            "axys_full_spec_b/transactions.csv",
            "axys_full_spec_b/sec_ref.csv",
        )

        for resource_path in expected_resources:
            with self.subTest(resource_path=resource_path):
                self.assertTrue((axys_demo_data / resource_path).is_file())

    def test_axys_full_spec_demo_uses_operational_mega_cap_data(self) -> None:
        """The user-facing comparison demo packages the promoted operational data."""
        axys_demo_data = files("ppar.demos.data") / "axys"
        positions_path = Path(
            str(axys_demo_data / "axys_full_spec_a" / "positions_holdings.csv")
        )
        sec_ref_path = Path(str(axys_demo_data / "axys_full_spec_a" / "sec_ref.csv"))

        with positions_path.open(encoding=util.ENCODING, newline="") as file:
            position_ids = {row["SEC"] for row in csv.DictReader(file)}
        with sec_ref_path.open(encoding=util.ENCODING, newline="") as file:
            security_ids = {row["SECURITY_ID"] for row in csv.DictReader(file)}

        self.assertIn("AAPL", position_ids)
        self.assertIn("NVDA", position_ids)
        self.assertIn("CASHBAL", position_ids)
        self.assertIn("TBILL13W", position_ids)
        self.assertIn("TNOTE2Y", position_ids)
        self.assertIn("TNOTE5Y", position_ids)
        self.assertTrue(position_ids.issubset(security_ids))

    def test_axys_validation_matrix_documents_problem_scenarios(self) -> None:
        """The Axys validation matrix names the expected review scenarios."""
        matrix = Path("tests/data/axys/README.md").read_text(encoding=util.ENCODING)
        expected_scenarios = {
            "Clean/no issue": "baseline",
            "Missing contribution policy": "policy_gap",
            "Missing transaction method": "policy_gap",
            "Missing denominator": "policy_gap",
            "Missing transaction sign/flow semantics": "policy_gap",
            "Low-confidence estimate": "multi",
            "Context-only evidence": "multi",
            "Modified Dietz cross-check": "modified_dietz",
            "Full YAML specifications": "full_spec",
            "Security full YAML specifications": "security_full_spec",
            "Suppressed finding": "suppressed",
            "Residual withheld": "multi",
            "Large clean background": "multi",
            "Large issue scale": "Future generated fixture",
        }

        self.assertIn("## Scenario Matrix", matrix)
        for scenario, fixture in expected_scenarios.items():
            with self.subTest(scenario=scenario):
                self.assertIn(scenario, matrix)
                self.assertIn(fixture, matrix)
        self.assertIn("Covered", matrix)
        self.assertIn("Planned", matrix)

    def test_axys_demo_readme_documents_supported_yaml_methods(self) -> None:
        """The packaged Axys demo README tracks every public YAML method target."""
        matrix = Path("ppar/demos/data/axys/README.md").read_text(encoding=util.ENCODING)
        expected_methods = {
            ContributionImpactMethod.SOURCE_FIELD_DELTA_OVER_BEGIN_MARKET_VALUE.value,
            ContributionImpactMethod.VENDOR_CONTRIBUTION_DELTA.value,
            ContributionImpactMethod.SECURITY_RETURN_DELTA_TIMES_WEIGHT.value,
            CashImpactMethod.CASH_DELTA_OVER_RETURN_DENOMINATOR.value,
            FxRateImpactMethod.EVIDENCE_ONLY.value,
            PositionImpactMethod.EVIDENCE_ONLY.value,
            PositionImpactMethod[
                "QUANTITY_DELTA_TIMES_SNAPSHOT_A_UNIT_MARKET_VALUE_OVER_RETURN_DENOMINATOR"
            ].value,
            PositionImpactMethod.MARKET_VALUE_DELTA_OVER_RETURN_DENOMINATOR.value,
            PositionImpactMethod.ACCRUED_DELTA_OVER_RETURN_DENOMINATOR.value,
            SecurityMasterImpactMethod.EVIDENCE_ONLY.value,
            TransactionImpactMethod.EVIDENCE_ONLY.value,
            TransactionImpactMethod.MODIFIED_DIETZ.value,
            TransactionImpactMethod.TRANSACTION_AMOUNT_DELTA_OVER_RETURN_DENOMINATOR.value,
        }

        self.assertIn("## Method Coverage Goal", matrix)
        for method in expected_methods:
            with self.subTest(method=method):
                self.assertIn(method, matrix)

    def test_performance_comparison_method_constants_use_enum_values(self) -> None:
        """Report and finding constants stay aligned with YAML method enums."""
        self.assertEqual(
            performance_comparison_findings.IMPACT_POLICY_PORTFOLIO_SOURCE_FIELD,
            (
                "portfolio_source_field:"
                f"{ContributionImpactMethod.SOURCE_FIELD_DELTA_OVER_BEGIN_MARKET_VALUE.value}"
            ),
        )
        self.assertEqual(
            performance_comparison_findings.IMPACT_POLICY_SECURITY_CONTRIBUTION,
            (
                "security_contribution:"
                f"{ContributionImpactMethod.VENDOR_CONTRIBUTION_DELTA.value}"
            ),
        )
        self.assertEqual(
            performance_comparison_findings.IMPACT_POLICY_SECURITY_RETURN_WEIGHTED,
            (
                "security_return:"
                f"{ContributionImpactMethod.SECURITY_RETURN_DELTA_TIMES_WEIGHT.value}"
            ),
        )
        self.assertEqual(
            performance_comparison_findings.IMPACT_POLICY_CASH_BALANCE,
            f"cash_balance:{CashImpactMethod.CASH_DELTA_OVER_RETURN_DENOMINATOR.value}",
        )
        self.assertEqual(
            performance_comparison_findings.IMPACT_POLICY_CASH_MARKET_VALUE,
            (
                "cash_market_value:"
                f"{CashImpactMethod.CASH_DELTA_OVER_RETURN_DENOMINATOR.value}"
            ),
        )
        self.assertEqual(
            performance_comparison_findings.TRANSACTION_IMPACT_POLICY_EXTERNAL_FLOW_EVIDENCE_ONLY,
            f"external_flow:{TransactionImpactMethod.EVIDENCE_ONLY.value}",
        )
        self.assertEqual(
            performance_comparison_findings.TRANSACTION_IMPACT_POLICY_PERFORMANCE_AMOUNT_DELTA,
            (
                "performance:"
                f"{TransactionImpactMethod.TRANSACTION_AMOUNT_DELTA_OVER_RETURN_DENOMINATOR.value}"
            ),
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
            "CashLoader": CashLoader,
            "ComparisonFile": ComparisonFile,
            "ComparisonSnapshot": ComparisonSnapshot,
            "CONTEXT": CONTEXT,
            "DIRECT_INPUT": DIRECT_INPUT,
            "EVIDENCE_ROLE": EVIDENCE_ROLE,
            "Finding": Finding,
            "FxRatesLoader": FxRatesLoader,
            "IMPACT_BASIS_PORTFOLIO_SOURCE_FIELD": IMPACT_BASIS_PORTFOLIO_SOURCE_FIELD,
            "IMPACT_BASIS_SECURITY_RETURN_WEIGHTED": IMPACT_BASIS_SECURITY_RETURN_WEIGHTED,
            "IMPACT_METHOD_SECURITY_RETURN_DELTA_TIMES_WEIGHT": (
                IMPACT_METHOD_SECURITY_RETURN_DELTA_TIMES_WEIGHT
            ),
            "IMPACT_METHOD_SOURCE_FIELD_DELTA_OVER_BEGIN_MV": (
                IMPACT_METHOD_SOURCE_FIELD_DELTA_OVER_BEGIN_MV
            ),
            "PerformanceComparison": PerformanceComparison,
            "PortfolioPerformanceLoader": PortfolioPerformanceLoader,
            "PerformanceComparisonSpecification": PerformanceComparisonSpecification,
            "PositionsLoader": PositionsLoader,
            "PricesLoader": PricesLoader,
            "REPORT_BUNDLE_REQUIRED_ARTIFACTS": REPORT_BUNDLE_REQUIRED_ARTIFACTS,
            "SecurityPerformanceLoader": SecurityPerformanceLoader,
            "SecurityMasterLoader": SecurityMasterLoader,
            "SuppressionRule": SuppressionRule,
            "RELATED_OUTPUT": RELATED_OUTPUT,
            "TARGET_OUTPUT": TARGET_OUTPUT,
            "TransactionsLoader": TransactionsLoader,
            "apply_suppressions": apply_suppressions,
            "schema": schema,
            "compact_findings_table": compact_findings_table,
            "compare_snapshots": compare_snapshots,
            "findings_to_polars": findings_to_polars,
            "portfolio_period_cause_summary": portfolio_period_cause_summary,
            "portfolio_period_contribution_candidates": (
                portfolio_period_contribution_candidates
            ),
            "portfolio_period_evidence_breakdown": portfolio_period_evidence_breakdown,
            "portfolio_period_flow_cross_check_reconciliation": (
                portfolio_period_flow_cross_check_reconciliation
            ),
            "portfolio_period_impact_coverage_summary": (
                portfolio_period_impact_coverage_summary
            ),
            "portfolio_period_summary": portfolio_period_summary,
            "portfolio_period_transaction_cross_checks": (
                portfolio_period_transaction_cross_checks
            ),
            "rank_portfolio_period_evidence": rank_portfolio_period_evidence,
            "report_bundle_validation_issues": report_bundle_validation_issues,
            "security_period_evidence_breakdown": security_period_evidence_breakdown,
            "security_period_summary": security_period_summary,
            "summarize_findings": summarize_findings,
            "validate_causal_attribution_ready": validate_causal_attribution_ready,
            "transaction_activity_summary": transaction_activity_summary,
            "transaction_matching_diagnostics": transaction_matching_diagnostics,
            "write_performance_comparison_report_bundle": (
                write_performance_comparison_report_bundle
            ),
            "write_performance_comparison_review_workbook": (
                write_performance_comparison_review_workbook
            ),
        }

        self.assertEqual(set(performance_comparison.__all__), set(expected_exports))
        for name, imported_object in expected_exports.items():
            with self.subTest(name=name):
                self.assertIs(imported_object, getattr(performance_comparison, name))

    def test_public_performance_comparison_runner_import_contract(self) -> None:
        """The runner module exposes only the compact workflow helper surface."""
        expected_exports = {
            "compact_findings_table",
            "compare_snapshots",
            "summarize_findings",
            "validate_causal_attribution_ready",
        }

        self.assertEqual(set(performance_comparison_runner.__all__), expected_exports)
        self.assertIs(
            compact_findings_table,
            performance_comparison_runner.compact_findings_table,
        )
        self.assertIs(compare_snapshots, performance_comparison_runner.compare_snapshots)
        self.assertIs(summarize_findings, performance_comparison_runner.summarize_findings)
        self.assertIs(
            validate_causal_attribution_ready,
            performance_comparison_runner.validate_causal_attribution_ready,
        )

    def test_public_performance_comparison_report_import_contract(self) -> None:
        """The report module exposes only report rendering and writing helpers."""
        expected_exports = {
            "write_performance_comparison_report_bundle",
            "write_performance_comparison_review_workbook",
        }

        self.assertEqual(set(performance_comparison_report.__all__), expected_exports)
        self.assertIs(
            write_performance_comparison_report_bundle,
            performance_comparison_report.write_performance_comparison_report_bundle,
        )
        self.assertIs(
            write_performance_comparison_review_workbook,
            performance_comparison_report.write_performance_comparison_review_workbook,
        )
        self.assertNotIn(
            "report_bundle_validation_issues",
            performance_comparison_report.__all__,
        )

    def test_performance_comparison_vocabulary_exports_are_explicit(self) -> None:
        """Vocabulary modules export every declared public schema/code name."""
        module_paths = {
            schema: Path("ppar/performance_comparison/schema.py"),
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
            core_schema: Path("ppar/analytics/schema.py"),
            core_errors: Path("ppar/errors.py"),
            util: Path("ppar/utilities.py"),
        }

        for module, path in module_paths.items():
            with self.subTest(module=module.__name__):
                self.assertEqual(
                    set(module.__all__),
                    _declared_public_module_names(path),
                )

    def test_analytics_schema_has_no_top_level_compatibility_module(self) -> None:
        """Analytics schema constants live only under the analytics package."""
        self.assertIsNone(importlib.util.find_spec("ppar.schema"))

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
            "raise SystemExit(1 if 'ppar.analytics.format_chart' in sys.modules else 0)"
        )

        self.assertEqual(chart_dependencies, {"matplotlib", "seaborn"})
        self.assertTrue(chart_dependencies.isdisjoint(core_dependencies))
        subprocess.run([sys.executable, "-c", command], check=True)


if __name__ == "__main__":
    unittest.main()
