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
from typing import cast
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
    HoldingImpactMethod,
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
    HoldingsLoader,
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
    validate_yaml_setup_complete,
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


def _csv_rows_by_key(
    path: Path,
    key_columns: tuple[str, ...],
) -> dict[tuple[str, ...], dict[str, str]]:
    """Return CSV rows keyed by one or more columns."""
    with path.open(encoding=util.ENCODING, newline="") as file:
        return {
            tuple(row[column] for column in key_columns): row
            for row in csv.DictReader(file)
        }


def _float_delta(
    snapshot_a: dict[str, str],
    snapshot_b: dict[str, str],
    column: str,
) -> float:
    """Return numeric snapshot B minus snapshot A for one CSV column."""
    return float(snapshot_b[column]) - float(snapshot_a[column])


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
        dev_dependencies = {
            dependency.split(">=", maxsplit=1)[0].lower()
            for dependency in optional_dependencies["dev"]
        }

        self.assertNotIn("great_tables", pyproject_dependencies)
        self.assertIn("pyyaml", pyproject_dependencies)
        self.assertIn("openpyxl", pyproject_dependencies)
        self.assertIn("matplotlib", pyproject_dependencies)
        self.assertIn("seaborn", pyproject_dependencies)
        self.assertNotIn("charts", optional_dependencies)
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
            "ppar_performance_comparison_portfolio.yaml",
            "ppar_performance_comparison_security.yaml",
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

    def test_axys_portfolio_demo_uses_operational_mega_cap_data(self) -> None:
        """The user-facing comparison demo packages the promoted operational data."""
        axys_demo_data = files("ppar.demos.data") / "axys"
        holdings_path = Path(
            str(axys_demo_data / "axys_full_spec_a" / "holdings.csv")
        )
        portperf_path = Path(str(axys_demo_data / "axys_full_spec_a" / "portperf.csv"))
        sec_ref_path = Path(str(axys_demo_data / "axys_full_spec_a" / "sec_ref.csv"))

        with portperf_path.open(encoding=util.ENCODING, newline="") as file:
            portfolio_codes = {row["PORTFOLIO_CODE"] for row in csv.DictReader(file)}
        with holdings_path.open(encoding=util.ENCODING, newline="") as file:
            holding_ids = {row["SEC"] for row in csv.DictReader(file)}
        with sec_ref_path.open(encoding=util.ENCODING, newline="") as file:
            security_ids = {row["SECURITY_ID"] for row in csv.DictReader(file)}

        self.assertEqual(portfolio_codes, {"ALPHA", "BALANCED", "INCOME"})
        self.assertIn("AAPL", holding_ids)
        self.assertIn("NVDA", holding_ids)
        self.assertIn("CASH_USD", holding_ids)
        self.assertIn("TBILL13W", holding_ids)
        self.assertIn("TNOTE2Y", holding_ids)
        self.assertIn("TNOTE5Y", holding_ids)
        self.assertTrue(holding_ids.issubset(security_ids))

    def test_axys_demo_changed_transactions_have_matching_holdings(self) -> None:
        """Material transaction demo changes have matching month-end holdings evidence."""
        axys_demo_data = files("ppar.demos.data") / "axys"
        snapshot_a = Path(str(axys_demo_data / "axys_full_spec_a"))
        snapshot_b = Path(str(axys_demo_data / "axys_full_spec_b"))

        transactions_a = _csv_rows_by_key(
            snapshot_a / "transactions.csv",
            ("TRANSACTION_ID",),
        )
        transactions_b = _csv_rows_by_key(
            snapshot_b / "transactions.csv",
            ("TRANSACTION_ID",),
        )
        holdings_a = _csv_rows_by_key(
            snapshot_a / "holdings.csv",
            ("PORT", "SEC", "HOLDING_DATE"),
        )
        holdings_b = _csv_rows_by_key(
            snapshot_b / "holdings.csv",
            ("PORT", "SEC", "HOLDING_DATE"),
        )

        changed_trade_cases = (
            ("ALPHA0401", "2026-03-31"),
            ("BALANCED0401", "2026-03-31"),
        )
        for transaction_id, holding_date in changed_trade_cases:
            with self.subTest(transaction_id=transaction_id):
                transaction_a = transactions_a[(transaction_id,)]
                transaction_b = transactions_b[(transaction_id,)]
                portfolio = transaction_a["PORT"]
                security = transaction_a["SEC"]
                holding_a = holdings_a[(portfolio, security, holding_date)]
                holding_b = holdings_b[(portfolio, security, holding_date)]
                cash_a = holdings_a[(portfolio, "CASH_USD", holding_date)]
                cash_b = holdings_b[(portfolio, "CASH_USD", holding_date)]

                quantity_delta = _float_delta(transaction_a, transaction_b, "QTY")
                amount_delta = _float_delta(transaction_a, transaction_b, "AMOUNT")
                month_end_price = float(holding_a["PRICE"])

                self.assertAlmostEqual(
                    _float_delta(holding_a, holding_b, "QTY"),
                    quantity_delta,
                    places=4,
                )
                self.assertAlmostEqual(
                    _float_delta(holding_a, holding_b, "MKT_VAL"),
                    quantity_delta * month_end_price,
                    places=2,
                )
                self.assertAlmostEqual(
                    _float_delta(cash_a, cash_b, "MKT_VAL"),
                    amount_delta,
                    places=2,
                )

        dividend_a = transactions_a[("BALANCED0502",)]
        dividend_b = transactions_b[("BALANCED0502",)]
        cash_a = holdings_a[("BALANCED", "CASH_USD", "2026-04-30")]
        cash_b = holdings_b[("BALANCED", "CASH_USD", "2026-04-30")]
        self.assertAlmostEqual(
            _float_delta(cash_a, cash_b, "MKT_VAL"),
            _float_delta(dividend_a, dividend_b, "AMOUNT"),
            places=2,
        )

    def test_axys_demo_buy_transaction_amounts_include_commission(self) -> None:
        """Packaged BUY rows use a stable signed cash-amount convention."""
        axys_demo_data = files("ppar.demos.data") / "axys"
        for snapshot_name in ("axys_full_spec_a", "axys_full_spec_b"):
            with self.subTest(snapshot=snapshot_name):
                snapshot = Path(str(axys_demo_data / snapshot_name))
                with (snapshot / "transactions.csv").open(
                    encoding=util.ENCODING,
                    newline="",
                ) as file:
                    for row in csv.DictReader(file):
                        if row["TRAN"] != "BUY":
                            continue
                        expected_amount = -round(
                            float(row["QTY"]) * float(row["PRICE"])
                            + float(row["COMMISSION"]),
                            2,
                        )
                        self.assertAlmostEqual(
                            float(row["AMOUNT"]),
                            expected_amount,
                            places=2,
                        )

    def test_axys_demo_security_performance_reconciles_to_holdings(self) -> None:
        """Security performance demo rows stay consistent with holdings and prices."""
        axys_demo_data = files("ppar.demos.data") / "axys"
        for snapshot_name in ("axys_full_spec_a", "axys_full_spec_b"):
            with self.subTest(snapshot=snapshot_name):
                snapshot = Path(str(axys_demo_data / snapshot_name))
                holdings = _csv_rows_by_key(
                    snapshot / "holdings.csv",
                    ("PORT", "SEC", "HOLDING_DATE"),
                )
                portperf = _csv_rows_by_key(
                    snapshot / "portperf.csv",
                    ("PORTFOLIO_CODE", "FROM_DATE", "THRU_DATE"),
                )
                secperf = _csv_rows_by_key(
                    snapshot / "secperf.csv",
                    ("PORTFOLIO_CODE", "SECURITY_ID", "FROM_DATE", "THRU_DATE"),
                )

                holding_keys = {
                    (portfolio, security, holding_date)
                    for portfolio, security, holding_date in holdings
                }
                secperf_holding_keys = {
                    (portfolio, security, thru_date)
                    for portfolio, security, _from_date, thru_date in secperf
                }
                self.assertEqual(holding_keys, secperf_holding_keys)

                for key, row in secperf.items():
                    portfolio, _security, from_date, thru_date = key
                    portfolio_row = portperf[(portfolio, from_date, thru_date)]
                    expected_weight = float(row["BEGIN_MV"]) / float(
                        portfolio_row["BEGIN_MV"]
                    )
                    expected_contribution = expected_weight * float(row["SEC_RETURN"])
                    self.assertAlmostEqual(
                        float(row["BEGIN_WEIGHT"]),
                        expected_weight,
                        places=10,
                    )
                    self.assertAlmostEqual(
                        float(row["CONTRIBUTION"]),
                        expected_contribution,
                        places=9,
                    )

    def test_axys_demo_portfolio_performance_rolls_up_security_performance(
        self,
    ) -> None:
        """Portfolio demo performance rows are derived from security rows."""
        axys_demo_data = files("ppar.demos.data") / "axys"
        for snapshot_name in ("axys_full_spec_a", "axys_full_spec_b"):
            with self.subTest(snapshot=snapshot_name):
                snapshot = Path(str(axys_demo_data / snapshot_name))
                portperf = _csv_rows_by_key(
                    snapshot / "portperf.csv",
                    ("PORTFOLIO_CODE", "FROM_DATE", "THRU_DATE"),
                )
                security_rows: dict[tuple[str, str, str], list[dict[str, str]]] = {}
                with (snapshot / "secperf.csv").open(
                    encoding=util.ENCODING,
                    newline="",
                ) as file:
                    for row in csv.DictReader(file):
                        key = (
                            row["PORTFOLIO_CODE"],
                            row["FROM_DATE"],
                            row["THRU_DATE"],
                        )
                        security_rows.setdefault(key, []).append(row)

                self.assertEqual(set(portperf), set(security_rows))
                for raw_key, portfolio_row in portperf.items():
                    key = cast(tuple[str, str, str], raw_key)
                    with self.subTest(snapshot=snapshot_name, key=key):
                        rows = security_rows[key]
                        self.assertAlmostEqual(
                            sum(float(row["BEGIN_MV"]) for row in rows),
                            float(portfolio_row["BEGIN_MV"]),
                            places=2,
                        )
                        self.assertAlmostEqual(
                            sum(float(row["END_MV"]) for row in rows),
                            float(portfolio_row["END_MV"]),
                            places=2,
                        )
                        self.assertAlmostEqual(
                            sum(float(row["CONTRIBUTION"]) for row in rows),
                            float(portfolio_row["PORT_RETURN"]),
                            places=9,
                        )

    def test_axys_demo_security_return_deltas_match_visible_inputs(self) -> None:
        """Security workbook demo return deltas match the changed source inputs."""
        axys_demo_data = files("ppar.demos.data") / "axys"
        snapshot_a = Path(str(axys_demo_data / "axys_full_spec_a"))
        snapshot_b = Path(str(axys_demo_data / "axys_full_spec_b"))
        prices_a = _csv_rows_by_key(
            snapshot_a / "prices.csv",
            ("SEC", "PRICE_DATE"),
        )
        prices_b = _csv_rows_by_key(
            snapshot_b / "prices.csv",
            ("SEC", "PRICE_DATE"),
        )
        holdings_a = _csv_rows_by_key(
            snapshot_a / "holdings.csv",
            ("PORT", "SEC", "HOLDING_DATE"),
        )
        holdings_b = _csv_rows_by_key(
            snapshot_b / "holdings.csv",
            ("PORT", "SEC", "HOLDING_DATE"),
        )
        secperf_a = _csv_rows_by_key(
            snapshot_a / "secperf.csv",
            ("PORTFOLIO_CODE", "SECURITY_ID", "FROM_DATE", "THRU_DATE"),
        )
        secperf_b = _csv_rows_by_key(
            snapshot_b / "secperf.csv",
            ("PORTFOLIO_CODE", "SECURITY_ID", "FROM_DATE", "THRU_DATE"),
        )

        price_delta = _float_delta(
            prices_a[("AAPL", "2026-05-29")],
            prices_b[("AAPL", "2026-05-29")],
            "PRICE",
        )
        expected_aapl_return_delta = price_delta / float(
            prices_a[("AAPL", "2026-05-29")]["PRICE"]
        )
        for portfolio in ("ALPHA", "BALANCED", "INCOME"):
            with self.subTest(portfolio=portfolio, security="AAPL"):
                key = (portfolio, "AAPL", "2026-05-01", "2026-05-29")
                self.assertAlmostEqual(
                    _float_delta(secperf_a[key], secperf_b[key], "SEC_RETURN"),
                    expected_aapl_return_delta,
                )

        tnote_key = ("INCOME", "TNOTE2Y", "2026-05-01", "2026-05-29")
        tnote_holding_key = ("INCOME", "TNOTE2Y", "2026-05-29")
        expected_tnote_return_delta = (
            _float_delta(
                holdings_a[tnote_holding_key],
                holdings_b[tnote_holding_key],
                "MKT_VAL",
            )
            + _float_delta(
                holdings_a[tnote_holding_key],
                holdings_b[tnote_holding_key],
                "ACCRUED",
            )
        ) / float(holdings_a[tnote_holding_key]["MKT_VAL"])
        self.assertAlmostEqual(
            _float_delta(secperf_a[tnote_key], secperf_b[tnote_key], "SEC_RETURN"),
            expected_tnote_return_delta,
        )

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
            "Portfolio YAML specifications": "portfolio",
            "Security YAML specifications": "security",
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

    def test_axys_demo_readme_documents_field_role_model(self) -> None:
        """The packaged Axys demo README describes the user-facing role model."""
        matrix = Path("ppar/demos/data/axys/README.md").read_text(encoding=util.ENCODING)
        expected_terms = {
            "performance_input",
            "input_component",
            "reported_performance_component",
            "context",
            "transaction_rules",
            TransactionImpactMethod.MODIFIED_DIETZ.value,
            TransactionImpactMethod.TRANSACTION_AMOUNT_DELTA_OVER_RETURN_DENOMINATOR.value,
        }

        self.assertIn("## YAML Policy Decision Guide", matrix)
        for term in expected_terms:
            with self.subTest(term=term):
                self.assertIn(term, matrix)

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
            "HoldingsLoader": HoldingsLoader,
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
            "validate_yaml_setup_complete": validate_yaml_setup_complete,
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
            "validate_yaml_setup_complete",
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
        self.assertIs(
            validate_yaml_setup_complete,
            performance_comparison_runner.validate_yaml_setup_complete,
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

    def test_chart_rendering_module_is_not_eagerly_loaded(self) -> None:
        """Normal package imports do not eagerly load chart rendering code."""
        command = (
            "import sys; import ppar; "
            "raise SystemExit(1 if 'ppar.analytics.format_chart' in sys.modules else 0)"
        )

        subprocess.run([sys.executable, "-c", command], check=True)


if __name__ == "__main__":
    unittest.main()
