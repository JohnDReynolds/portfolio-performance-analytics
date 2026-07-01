"""Tests for package metadata maintained alongside runtime dependencies."""

# Python Imports
import ast
import csv
from fnmatch import fnmatch
import importlib.util
from importlib.resources import files
from pathlib import Path
import re
import subprocess
import sys
import tomllib
from typing import cast
import unittest

# Third-Party Imports
import polars as pl
import yaml

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
from ppar.performance_comparison import backlog_gates as performance_backlog_gates
from ppar.performance_comparison import fixed_income as performance_fixed_income
from ppar.performance_comparison import (
    transaction_summary as performance_transaction_summary,
)
from ppar.performance_comparison import (
    source_data_contract as performance_source_data_contract,
)
from ppar.performance_comparison.transaction_boundary_registry import (
    TRANSACTION_BOUNDARY_REGISTRY,
    registered_transaction_codes,
    transaction_boundary_groups,
)
from ppar.performance_comparison import (
    transaction_boundary_registry as performance_boundary_registry,
)
from ppar.performance_comparison.return_reconstruction import (
    DERIVED_RETURN_DIFFERENCE,
    RECONSTRUCTION_STATUS,
    RECONSTRUCTION_STATUS_ALIGNED,
    RECONSTRUCTION_STATUS_DIFFERENT,
    RECONSTRUCTION_STATUS_MISSING_INPUTS,
    REPORTED_RETURN_DIFFERENCE,
    portfolio_return_reconstruction_checks,
    security_return_reconstruction_checks,
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
    report_bundle_contract,
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

_INTENTIONAL_PORTFOLIO_RECONSTRUCTION_DIFFERENT_KEYS = {
    ("BALANCED", "2026-05-01", "2026-05-29"),
    ("INCOME", "2026-04-01", "2026-04-30"),
}


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


def _load_yaml(path: Path) -> dict[str, object]:
    """Return a YAML mapping loaded from a repository file."""
    with path.open(encoding=util.ENCODING) as file:
        loaded = yaml.safe_load(file)
    if not isinstance(loaded, dict):
        raise AssertionError(f"Expected YAML mapping in {path}.")
    return loaded


def _transaction_codes_in_csv(path: Path) -> set[str]:
    """Return lowercase transaction codes from a fixture CSV."""
    with path.open(encoding=util.ENCODING, newline="") as file:
        return {
            row["TRAN"].strip().lower()
            for row in csv.DictReader(file)
            if row["TRAN"].strip()
        }


def _demo_transactions_by_natural_key(
    snapshot_path: Path,
) -> dict[tuple[str, str, str, str], dict[str, str]]:
    """Return packaged Axys transactions keyed by visible row attributes."""
    return _csv_rows_by_key(
        snapshot_path / "transactions.csv",
        ("PORT", "TRANSACTION_DATE", "SEC", "TRAN"),
    )


def _float_delta(
    snapshot_a: dict[str, str],
    snapshot_b: dict[str, str],
    column: str,
) -> float:
    """Return numeric snapshot B minus snapshot A for one CSV column."""
    return float(snapshot_b[column]) - float(snapshot_a[column])


def _reconstruction_rows_by_key(
    checks: pl.DataFrame,
    key_columns: tuple[str, ...],
) -> dict[tuple[str, ...], dict[str, object]]:
    """Return reconstruction rows keyed by stable stringified columns."""
    return {
        tuple(str(row[column]) for column in key_columns): row
        for row in checks.iter_rows(named=True)
    }


def _reconstruction_float(row: dict[str, object], column: str) -> float:
    """Return a numeric reconstruction value from a Polars row dictionary."""
    value = row[column]
    if not isinstance(value, (int, float)):
        raise AssertionError(f"Expected numeric reconstruction value for {column}.")
    return float(value)


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

    def test_source_data_term_is_standardized(self) -> None:
        """Docs and reviewer-facing code use one source-data prose term."""
        forbidden_patterns = (
            re.compile(r"\binput data\b", flags=re.IGNORECASE),
            re.compile(r"\binput-data\b", flags=re.IGNORECASE),
            re.compile(r"\bsource data\b", flags=re.IGNORECASE),
        )
        scanned_paths = [Path("README.md")]
        for root in (
            Path("docs"),
            Path("ppar/performance_comparison"),
            Path("tests"),
        ):
            scanned_paths.extend(
                path
                for path in root.rglob("*")
                if path.suffix in {".md", ".py"}
            )

        violations = []
        for path in scanned_paths:
            text = path.read_text(encoding=util.ENCODING)
            for line_number, line in enumerate(text.splitlines(), start=1):
                for pattern in forbidden_patterns:
                    if pattern.search(line):
                        violations.append(f"{path}:{line_number}: {pattern.pattern}")

        self.assertEqual([], violations)

    def test_public_demo_review_guidance_matches_current_bundle_shape(self) -> None:
        """Public docs describe report files as review surfaces and CSVs as audit aids."""
        root_readme = Path("README.md").read_text(encoding=util.ENCODING)
        axys_readme = Path("ppar/demos/data/axys/README.md").read_text(
            encoding=util.ENCODING
        )
        normalized_root_readme = " ".join(root_readme.split())
        normalized_axys_readme = " ".join(axys_readme.split())

        for text in (root_readme, axys_readme):
            self.assertIn("use `report.html` for browser review", text.lower())
            self.assertIn("CSV artifacts", text)
            self.assertIn("audit traceability", text)
            self.assertNotIn("same review model in a browser", text)

        self.assertIn("open `report.xlsx` when present", normalized_root_readme)
        self.assertIn("Open `report.xlsx` when present", normalized_axys_readme)

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

    def test_package_data_excludes_generation_internals(self) -> None:
        """Wheel package-data exposes demo inputs, not source-checkout tooling."""
        with open("pyproject.toml", "rb") as file:
            pyproject = tomllib.load(file)
        package_data_patterns = pyproject["tool"]["setuptools"]["package-data"]["ppar"]

        forbidden_fragments = (
            "_demo_output",
            "docs/",
            "scripts/",
            "tests/",
            "operational_demo_data",
            "GENERATION_NOTES",
        )

        for pattern in package_data_patterns:
            with self.subTest(pattern=pattern):
                self.assertFalse(
                    any(fragment in pattern for fragment in forbidden_fragments),
                    f"{pattern} exposes generation or source-checkout internals.",
                )

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

    def test_packaged_axys_demo_files_are_product_inputs(self) -> None:
        """Packaged Axys demo files stay limited to user-facing inputs and notes."""
        allowed_suffixes = {".csv", ".md", ".yaml"}
        axys_demo_files = [
            path
            for path in Path("ppar/demos/data/axys").rglob("*")
            if path.is_file()
        ]

        self.assertGreater(len(axys_demo_files), 0)
        for path in axys_demo_files:
            with self.subTest(path=path.as_posix()):
                self.assertIn(path.suffix, allowed_suffixes)
                self.assertNotIn("_demo_output", path.as_posix())
                self.assertNotIn("operational_demo_data", path.as_posix())

    def test_site_extract_contract_template_is_documented(self) -> None:
        """The site extract-contract starter template remains linked from docs."""
        template_path = Path("docs/axys-apx-reference/templates/site_extract_contract.yaml")
        source_contract = Path("docs/performance_comparison_demo_source_contract.md")
        demo_readme = Path("ppar/demos/data/axys/README.md")

        self.assertTrue(template_path.exists())
        self.assertIn(
            template_path.as_posix(),
            demo_readme.read_text(encoding=util.ENCODING),
        )
        self.assertIn(
            "axys-apx-reference/templates/site_extract_contract.yaml",
            source_contract.read_text(encoding=util.ENCODING),
        )

    def test_transaction_semantics_matrix_is_documented(self) -> None:
        """The Axys transaction semantics matrix remains linked from workflow docs."""
        matrix_path = Path(
            "docs/axys-apx-reference/Appendix_Transaction_Semantics_Matrix.md"
        )
        matrix_yaml_path = Path("docs/axys-apx-reference/transaction_semantics_matrix.yaml")
        boundary_snapshot_path = Path(
            "docs/performance_comparison_transaction_boundary_snapshot.md"
        )
        evidence_pack_review_path = Path(
            "docs/performance_comparison_evidence_pack_review.md"
        )
        source_contract = Path("docs/performance_comparison_demo_source_contract.md")
        roadmap = Path("docs/performance_comparison_roadmap.md")

        self.assertTrue(matrix_path.exists())
        self.assertTrue(matrix_yaml_path.exists())
        self.assertTrue(boundary_snapshot_path.exists())
        self.assertTrue(evidence_pack_review_path.exists())
        self.assertIn(
            "axys-apx-reference/Appendix_Transaction_Semantics_Matrix.md",
            source_contract.read_text(encoding=util.ENCODING),
        )
        self.assertIn(
            "performance_comparison_transaction_boundary_snapshot.md",
            source_contract.read_text(encoding=util.ENCODING),
        )
        self.assertIn(
            "performance_comparison_evidence_pack_review.md",
            source_contract.read_text(encoding=util.ENCODING),
        )
        self.assertIn(
            "axys-apx-reference/Appendix_Transaction_Semantics_Matrix.md",
            roadmap.read_text(encoding=util.ENCODING),
        )
        self.assertIn(
            "performance_comparison_transaction_boundary_snapshot.md",
            roadmap.read_text(encoding=util.ENCODING),
        )
        self.assertIn(
            "performance_comparison_evidence_pack_review.md",
            roadmap.read_text(encoding=util.ENCODING),
        )

    def test_demo_transaction_expansion_gate_is_documented(self) -> None:
        """The roadmap keeps future packaged Axys transaction additions gated."""
        roadmap = Path("docs/performance_comparison_roadmap.md").read_text(
            encoding=util.ENCODING
        )

        self.assertIn("Phase 8A: Realistic Transaction Expansion Gate", roadmap)
        self.assertIn("packaged-demo contribution scenario", roadmap)
        self.assertIn("source/destination type", roadmap)
        self.assertIn("source/destination symbol", roadmap)
        self.assertIn("REP/report\n  semantic fields", roadmap)
        self.assertIn("context, not from an ambiguous code alone", roadmap)
        self.assertIn("generated ending `CASH_USD` holding", roadmap)
        self.assertIn("portfolio Modified Dietz\n  reconstruction", roadmap)
        self.assertIn("without double-counting", roadmap)
        self.assertIn("Implemented contribution recipe", roadmap)
        self.assertIn("inserted transaction scenario", roadmap)
        self.assertIn("`SRC_DEST_TYPE=$pty`", roadmap)
        self.assertIn("positive `AMOUNT`", roadmap)
        self.assertIn("`li`/`lo`, additional `dp`/`wd`", roadmap)
        self.assertIn("actual historical split date", roadmap)

    def test_reinvestment_pair_gate_is_documented(self) -> None:
        """The roadmap keeps dividend-reinvestment pairs test-only until safe."""
        roadmap = Path("docs/performance_comparison_roadmap.md").read_text(
            encoding=util.ENCODING
        )

        self.assertIn("Phase 8B: Reinvestment Pair Feasibility Gate", roadmap)
        self.assertIn("Status: partial test-only coverage", roadmap)
        self.assertIn("`dv` income leg plus a related `by` purchase leg", roadmap)
        self.assertIn("formula-role evidence", roadmap)
        self.assertIn("no portfolio-level external-flow treatment", roadmap)
        self.assertIn("must not count the buy\n  leg as an external contribution", roadmap)
        self.assertIn("must not count\n  the dividend income twice", roadmap)
        self.assertIn("likely related income and buy\n  evidence", roadmap)
        self.assertIn("synthetic reinvestment examples belong in test-only data", roadmap)

    def test_fixed_income_transaction_boundary_gate_is_documented(self) -> None:
        """The roadmap keeps under-evidenced fixed-income transaction rows gated."""
        roadmap = Path("docs/performance_comparison_roadmap.md").read_text(
            encoding=util.ENCODING
        )
        source_contract = Path(
            "docs/performance_comparison_demo_source_contract.md"
        ).read_text(encoding=util.ENCODING)

        self.assertIn("Phase 8C: Fixed-Income Transaction Boundary Gate", roadmap)
        self.assertIn("ordinary `in` interest rows", roadmap)
        self.assertIn("`holdings.accrued`", roadmap)
        self.assertIn("`ai`, `pa`, `sa`, and `pd`", roadmap)
        self.assertIn("local mapping or\n  REP/report semantics", roadmap)
        self.assertIn("quantity or principal\n  exposure", roadmap)
        self.assertIn("before it is treated as performance income", roadmap)
        self.assertIn(
            "two proved fixed-income Modified Dietz inputs",
            source_contract,
        )
        self.assertIn("amortization/accretion engine", source_contract)
        self.assertIn("bond principal schedule", source_contract)

    def test_fixed_income_modified_dietz_phase_is_documented(self) -> None:
        """The roadmap keeps Phase 10 scoped to Modified Dietz formula inputs."""
        roadmap = Path("docs/performance_comparison_roadmap.md").read_text(
            encoding=util.ENCODING
        )

        self.assertIn("Phase 10: Fixed-Income Modified Dietz Boundary", roadmap)
        self.assertIn("Phase 10A: Fixed-Income Formula Boundary", roadmap)
        self.assertIn("ordinary `in` interest transaction amounts", roadmap)
        self.assertIn("configured `holdings.accrued` changes", roadmap)
        self.assertIn("amortization/accretion engines", roadmap)
        self.assertIn("bond principal schedule reconstruction", roadmap)
        self.assertIn("Phase 10B: Test-Only Ordinary Interest + Accrued Audit", roadmap)
        self.assertIn("`INCOME0603` remains an ordinary `in` transaction", roadmap)
        self.assertIn("positive `TNOTE2Y` `holdings.accrued` values", roadmap)
        self.assertIn(
            "Phase 10C: Principal Paydown / Accrued-Interest Backlog Contract",
            roadmap,
        )
        self.assertIn("backlog codes remain `unknown` by code alone", roadmap)
        self.assertIn("Phase 10D: Fixed-Income Reviewer Reporting", roadmap)
        self.assertIn("not silently inferred income or flows", roadmap)

    def test_test_only_transaction_semantics_phase_is_documented(self) -> None:
        """The roadmap keeps Phase 11 in the test-only semantics lane."""
        roadmap = Path("docs/performance_comparison_roadmap.md").read_text(
            encoding=util.ENCODING
        )

        self.assertIn("Phase 11: Test-Only Transaction Semantics Expansion", roadmap)
        self.assertIn("Phase 11A: Reversal / Cancellation Boundary", roadmap)
        self.assertIn("`CXL` stays transfer-neutral", roadmap)
        self.assertIn("`REV` stays transfer-neutral", roadmap)
        self.assertIn("Phase 11B: Expanded `dp` / `wd` Context Matrix", roadmap)
        self.assertIn("fee-like `dp` is performance-impacting", roadmap)
        self.assertIn("sweep-like `wd` stays neutral transfer evidence", roadmap)
        self.assertIn("Phase 11C: Synthetic Corporate-Action Quarantine", roadmap)
        self.assertIn("neutral corporate-action row", roadmap)
        self.assertIn("without turning it into a Modified Dietz formula\ninput", roadmap)
        self.assertIn("Phase 11D: Demo Matrix + Roadmap Reporting", roadmap)
        self.assertIn("review-only action quarantine", roadmap)

    def test_return_capital_and_short_backlog_phase_is_documented(self) -> None:
        """The roadmap and matrix keep high-risk backlog gates explicit."""
        roadmap = Path("docs/performance_comparison_roadmap.md").read_text(
            encoding=util.ENCODING
        )
        matrix_yaml = _load_yaml(
            Path("docs/axys-apx-reference/transaction_semantics_matrix.yaml")
        )

        self.assertIn(
            "Phase 12: Return-of-Capital And Short-Side Backlog Gates",
            roadmap,
        )
        self.assertIn("Phase 12A: Return-of-Capital Policy Boundary", roadmap)
        self.assertIn("Code-only `rc` rows stay `unknown`", roadmap)
        self.assertIn(
            "Phase 12B: Principal / Capital Return Vocabulary Alignment",
            roadmap,
        )
        self.assertIn("`pd` remains aligned with the capital-return gate", roadmap)
        self.assertIn("Phase 12C: Short Sale / Cover Short Evidence Gate", roadmap)
        self.assertIn("Code-only short-side rows stay `unknown`", roadmap)
        self.assertIn("Phase 12D: Matrix + Validator Reporting", roadmap)
        self.assertIn("`Capital-return and short-side backlog\ngates`", roadmap)

        for code in ("rc", "pd", "ss", "cs"):
            with self.subTest(code=code):
                row = matrix_yaml["rows"][code]
                self.assertEqual(row["coverage_status"], "backlog")
                self.assertEqual(row["fixtures"], [])
                self.assertIn("Backlog gate:", row["coverage_notes"])

    def test_matrix_consolidation_phase_is_documented(self) -> None:
        """The roadmap documents the release-readiness consolidation phase."""
        roadmap = Path("docs/performance_comparison_roadmap.md").read_text(
            encoding=util.ENCODING
        )
        snapshot = Path(
            "docs/performance_comparison_transaction_boundary_snapshot.md"
        ).read_text(encoding=util.ENCODING)

        self.assertIn("Phase 13: Matrix Consolidation And Release Readiness", roadmap)
        self.assertIn("Phase 13A: Transaction Boundary Registry", roadmap)
        self.assertIn("transaction_boundary_registry", roadmap)
        self.assertIn("Phase 13B: Demo Matrix Validator Cleanup", roadmap)
        self.assertIn("baseline and\nattribution checks", roadmap)
        self.assertIn("Phase 13C: Roadmap / Matrix Consistency Audit", roadmap)
        self.assertIn("Phase 13D: Pre-Commit Release Snapshot", roadmap)
        self.assertIn("Covered Formula Inputs", snapshot)
        self.assertIn("Context-Required Rows", snapshot)
        self.assertIn("Backlog Gates", snapshot)

    def test_final_review_pack_phase_is_documented(self) -> None:
        """The roadmap and review pack document commit-preparation scope."""
        roadmap = Path("docs/performance_comparison_roadmap.md").read_text(
            encoding=util.ENCODING
        )
        review_pack = Path(
            "docs/performance_comparison_evidence_pack_review.md"
        ).read_text(encoding=util.ENCODING)

        self.assertIn("Phase 14: Final Review Pack And Commit Preparation", roadmap)
        self.assertIn("Phase 14A: Change Inventory", roadmap)
        self.assertIn("Phase 14B: Public API / Package Surface Check", roadmap)
        self.assertIn("Phase 14C: Diff Hygiene Pass", roadmap)
        self.assertIn("Phase 14D: Commit-Ready Validation", roadmap)
        self.assertIn("Change Inventory", review_pack)
        self.assertIn("Public Surface", review_pack)
        self.assertIn("Suggested Commit Message", review_pack)
        self.assertIn("Add performance comparison evidence-pack boundaries", review_pack)

    def test_phase_roadmap_sections_are_unique_after_review_pack(self) -> None:
        """The active roadmap keeps one section per late phase train."""
        roadmap = Path("docs/performance_comparison_roadmap.md").read_text(
            encoding=util.ENCODING
        )

        for phase in range(9, 15):
            with self.subTest(phase=phase):
                self.assertEqual(roadmap.count(f"### Phase {phase}:"), 1)
        self.assertEqual(roadmap.count("## Guiding Principle"), 1)
        self.assertEqual(roadmap.count("## Transaction-Type Backlog"), 1)

    def test_performance_comparison_roadmap_starts_with_current_status(self) -> None:
        """The roadmap separates active backlog from historical phase notes."""
        roadmap = Path("docs/performance_comparison_roadmap.md").read_text(
            encoding=util.ENCODING
        )

        for heading in (
            "## How To Read This Roadmap",
            "## Current Status",
            "## Current Open Items",
            "### Near-Term Deliverables",
            "### Transaction Coverage Expansion",
            "### Longer-Term Deliverables",
            "## Axys Extract Contract Review Map",
            "## Implementation Phases",
        ):
            self.assertIn(heading, roadmap)

        self.assertLess(
            roadmap.index("## Current Open Items"),
            roadmap.index("## Implementation Phases"),
        )
        self.assertIn("The remaining work is backlog expansion", roadmap)
        self.assertIn("The phase notes below are an implementation journal", roadmap)
        self.assertIn("Completed guardrails now cover:", roadmap)
        self.assertIn("Phase 37: Roadmap Readability Refactor", roadmap)
        self.assertIn("Evidence-blocked backlog", roadmap)
        self.assertIn("Policy expansion", roadmap)
        self.assertIn("PyPI wheel package-data surface includes packaged demo", roadmap)
        self.assertIn("excludes source-checkout generation internals", roadmap)
        self.assertIn("minimum source-data contract is documented", roadmap)
        self.assertIn("required datasets and required normalized columns", roadmap)
        self.assertIn("intentional `Unexplained`", roadmap)
        self.assertIn("Do not add \"all transaction types\"", roadmap)
        self.assertIn("Richer APX demo", roadmap)
        self.assertIn("multi-currency data must affect comparison behavior", roadmap)
        self.assertIn("Phase 38: Packaging Surface And Product Boundary Audit", roadmap)
        self.assertIn(
            "Phase 40: YAML Strictness And Misleading-Report Prevention",
            roadmap,
        )
        self.assertNotIn("| Packaging surface |", roadmap)
        self.assertNotIn("| User-facing input contract |", roadmap)
        self.assertNotIn("| YAML strictness |", roadmap)
        self.assertIn("validate_config` now uses the same complete-YAML", roadmap)
        self.assertIn("`--allow-incomplete-yaml`", roadmap)

    def test_evidence_pack_hardening_phase_is_documented(self) -> None:
        """The roadmap keeps the reviewer-readiness train tied to evidence packs."""
        roadmap = Path("docs/performance_comparison_roadmap.md").read_text(
            encoding=util.ENCODING
        )

        self.assertIn("Phase 9: Evidence-Pack Hardening And Reviewer Readiness", roadmap)
        self.assertIn("Phase 9A: Bundle Navigation Manifest", roadmap)
        self.assertIn("reviewer entrypoints", roadmap)
        self.assertIn("comparison YAML path", roadmap)
        self.assertIn("extract-contract summary metadata", roadmap)
        self.assertIn("Phase 9B: Site Extract Readiness", roadmap)
        self.assertIn("missing transaction context columns", roadmap)
        self.assertIn("Phase 9C: Test-Only Semantics Expansion", roadmap)
        self.assertIn("Modified Dietz formula role", roadmap)
        self.assertIn("Phase 9D: Manifest Validation And Extract Context Summary", roadmap)
        self.assertIn("review_entrypoints", roadmap)
        self.assertIn("required\n  transaction context columns", roadmap)
        self.assertIn("observed transaction codes", roadmap)
        self.assertIn("Phase 9E: Bundle Validation Completion", roadmap)
        self.assertIn("source_context.extract_contract", roadmap)
        self.assertIn("Phase 9F: Extract Context Operator Readiness", roadmap)
        self.assertIn("Phase 9G: Shared Transaction Semantics Summary", roadmap)
        self.assertIn("codes without YAML rules", roadmap)
        self.assertIn("Phase 9H: Operator Checklist Docs", roadmap)
        self.assertIn("site_extract_readiness_checklist.md", roadmap)
        self.assertIn("Phase 9I: Test-Only Ambiguous Flow Matrix", roadmap)
        self.assertIn("Ambiguous flow context variants", roadmap)
        self.assertIn("Phase 9J: Code-Only Failure Fixtures", roadmap)
        self.assertIn("Code-only failure guard", roadmap)
        self.assertIn("Phase 9K: Local Opt-Out Boundary", roadmap)
        self.assertIn("Reviewed local opt-out", roadmap)
        self.assertIn("Phase 9L: Demo Matrix Reporting Polish", roadmap)
        self.assertIn("Phase 9M: Evidence-Pack Golden Bundle Fixture", roadmap)
        self.assertIn("manifest_version", roadmap)
        self.assertIn("Phase 9N: README / CLI Review Flow Tightening", roadmap)
        self.assertIn("Phase 9O: Bundle Manifest Regression Contract", roadmap)
        self.assertIn("Phase 9P: Final Phase-9 Consolidation", roadmap)
        self.assertIn("Status: complete.", roadmap)
        self.assertIn("release-ready evidence-pack baseline", roadmap)

    def test_site_extract_readiness_checklist_is_documented(self) -> None:
        """The site extract readiness checklist remains linked from setup docs."""
        checklist = Path("docs/site_extract_readiness_checklist.md")
        source_contract = Path("docs/performance_comparison_demo_source_contract.md")
        template = Path("docs/axys-apx-reference/templates/site_extract_contract.yaml")

        self.assertTrue(checklist.exists())
        checklist_text = checklist.read_text(encoding=util.ENCODING)
        self.assertIn("IMEX With Context Fields", checklist_text)
        self.assertIn("REP/Report Semantic Fallback", checklist_text)
        self.assertIn("Code-Only Failure Mode", checklist_text)
        self.assertIn("Reviewed Local Opt-Out", checklist_text)
        self.assertIn("Handoff Evidence", checklist_text)
        self.assertIn(
            "site_extract_readiness_checklist.md",
            source_contract.read_text(encoding=util.ENCODING),
        )
        self.assertIn(
            "site_extract_readiness_checklist.md",
            template.read_text(encoding=util.ENCODING),
        )

    def test_site_variant_local_opt_out_fixture_is_documented(self) -> None:
        """The reviewed local opt-out fixture is documented as non-default."""
        readme = Path("tests/data/axys/site_variants/README.md").read_text(
            encoding=util.ENCODING
        )
        source_contract = Path("docs/performance_comparison_demo_source_contract.md")
        local_opt_out = Path(
            "tests/data/axys/site_variants/local_opt_out/ppar_performance_comparison.yaml"
        )

        self.assertTrue(local_opt_out.exists())
        self.assertIn("local_opt_out", readme)
        self.assertIn("enforce_ambiguous_axys_flows", readme)
        self.assertIn(
            "local_opt_out",
            source_contract.read_text(encoding=util.ENCODING),
        )

    def test_reinvestment_gate_stays_modified_dietz_scoped(self) -> None:
        """Dividend reinvestment docs avoid requiring accounting-style matching."""
        roadmap = Path("docs/performance_comparison_roadmap.md").read_text(
            encoding=util.ENCODING
        )

        self.assertIn("does\nnot require accounting-style pair matching", roadmap)
        self.assertIn("The required formula boundary is narrower", roadmap)
        self.assertIn("`dv` is income", roadmap)
        self.assertIn("`by` is a\nsecurity-level flow", roadmap)
        self.assertIn("optional\nreviewer polish", roadmap)

    def test_transaction_semantics_matrix_yaml_matches_appendix_codes(self) -> None:
        """The machine-readable transaction matrix stays aligned with the appendix."""
        matrix_yaml = _load_yaml(
            Path("docs/axys-apx-reference/transaction_semantics_matrix.yaml")
        )
        appendix = Path(
            "docs/axys-apx-reference/Appendix_Transaction_Semantics_Matrix.md"
        ).read_text(encoding=util.ENCODING)

        rows = matrix_yaml["rows"]
        required_codes = matrix_yaml["required_matrix_codes"]
        self.assertEqual(set(rows), set(required_codes))

        for code, metadata in rows.items():
            self.assertIn(f"`{code}`", appendix)
            self.assertIn("observed_meaning", metadata)
            self.assertIn("ppar_categories", metadata)
            self.assertIn("coverage_status", metadata)
            self.assertIn("fixtures", metadata)
            self.assertIn("coverage_notes", metadata)

        self.assertEqual(
            set(matrix_yaml["ambiguous_external_flow_codes"]),
            {"li", "lo", "dp", "wd"},
        )

    def test_transaction_semantics_matrix_rows_have_coverage_rationale(self) -> None:
        """Each transaction matrix row explains its coverage or backlog status."""
        matrix_yaml = _load_yaml(
            Path("docs/axys-apx-reference/transaction_semantics_matrix.yaml")
        )
        coverage_statuses = set(matrix_yaml["coverage_statuses"])

        for section_name in ("rows", "pair_patterns"):
            for code, metadata in matrix_yaml[section_name].items():
                with self.subTest(section=section_name, code=code):
                    coverage_status = metadata["coverage_status"]
                    fixtures = metadata.get("fixtures", [])
                    coverage_notes = str(metadata.get("coverage_notes", "")).strip()

                    self.assertIn(coverage_status, coverage_statuses)
                    self.assertTrue(coverage_notes)
                    if coverage_status == "backlog":
                        self.assertEqual(fixtures, [], code)
                        self.assertIn("Backlog", coverage_notes)
                    else:
                        if section_name == "rows":
                            self.assertTrue(fixtures, code)
                        self.assertNotIn("Backlog pending", coverage_notes)

    def test_transaction_boundary_registry_matches_matrix_codes(self) -> None:
        """The boundary registry covers every matrix row and key overlap."""
        matrix_yaml = _load_yaml(
            Path("docs/axys-apx-reference/transaction_semantics_matrix.yaml")
        )
        matrix_codes = set(matrix_yaml["required_matrix_codes"])

        self.assertLessEqual(matrix_codes, registered_transaction_codes())
        self.assertEqual(
            transaction_boundary_groups("in"),
            ("packaged_formula", "fixed_income_safe"),
        )
        self.assertIn(
            "ambiguous_context_required",
            transaction_boundary_groups("wd"),
        )
        self.assertIn("review_only_test", transaction_boundary_groups(";"))
        self.assertIn("context_only", transaction_boundary_groups("exus"))
        self.assertIn("standalone_backlog", transaction_boundary_groups("epus"))
        self.assertIn("fixed_income_backlog", transaction_boundary_groups("pd"))
        self.assertIn("capital_return_backlog", transaction_boundary_groups("pd"))
        self.assertIn("short_side_backlog", transaction_boundary_groups("ss"))
        self.assertEqual(transaction_boundary_groups("not-a-real-code"), ())

        registered_groups = set(TRANSACTION_BOUNDARY_REGISTRY)
        self.assertEqual(
            registered_groups,
            {
                "packaged_formula",
                "fixed_income_safe",
                "ambiguous_context_required",
                "review_only_test",
                "context_only",
                "fixed_income_backlog",
                "capital_return_backlog",
                "short_side_backlog",
                "standalone_backlog",
            },
        )

    def test_demo_transaction_rules_are_known_to_semantics_matrix(self) -> None:
        """Packaged demo transaction rules cannot introduce undocumented codes."""
        matrix_yaml = _load_yaml(
            Path("docs/axys-apx-reference/transaction_semantics_matrix.yaml")
        )
        demo_yaml = _load_yaml(Path("ppar/demos/data/axys/ppar_performance_comparison.yaml"))

        matrix_codes = set(matrix_yaml["rows"])
        demo_codes = set(demo_yaml["transaction_rules"])

        self.assertLessEqual(demo_codes, matrix_codes)

    def test_transaction_semantics_coverage_claims_have_fixtures(self) -> None:
        """Coverage statuses in the machine-readable matrix point to real fixtures."""
        matrix_yaml = _load_yaml(
            Path("docs/axys-apx-reference/transaction_semantics_matrix.yaml")
        )
        packaged_demo_codes = set(
            _load_yaml(Path("ppar/demos/data/axys/ppar_performance_comparison.yaml"))[
                "transaction_rules"
            ]
        )
        imex_context_codes = _transaction_codes_in_csv(
            Path("tests/data/axys/site_variants/imex_context/snapshot_a/transactions.csv")
        )
        rep_semantics_codes = _transaction_codes_in_csv(
            Path("tests/data/axys/site_variants/rep_semantics/snapshot_a/transactions.csv")
        )
        code_only_codes = _transaction_codes_in_csv(
            Path("tests/data/axys/site_variants/imex_code_only/snapshot_a/transactions.csv")
        )
        review_only_codes = _transaction_codes_in_csv(
            Path(
                "tests/data/axys/site_variants/review_only_actions/"
                "snapshot_a/transactions.csv"
            )
        )

        for code, metadata in matrix_yaml["rows"].items():
            fixtures = metadata["fixtures"]
            coverage_status = metadata["coverage_status"]
            if coverage_status == "backlog":
                self.assertEqual(fixtures, [], code)
                continue
            if "packaged_demo" in fixtures:
                self.assertTrue(
                    code in packaged_demo_codes or code in {"exus"},
                    code,
                )
            if "site_variants/imex_context" in fixtures:
                self.assertTrue(code in imex_context_codes or code in {"exus"}, code)
            if "site_variants/rep_semantics" in fixtures:
                self.assertIn(code, rep_semantics_codes, code)
            if "site_variants/review_only_actions" in fixtures:
                self.assertIn(code, review_only_codes, code)

        self.assertLessEqual(
            set(matrix_yaml["ambiguous_external_flow_codes"]),
            imex_context_codes & rep_semantics_codes & code_only_codes,
        )

    def test_review_only_action_fixture_is_documented(self) -> None:
        """Synthetic review-only action rows remain a test-only quarantine."""
        readme = Path("tests/data/axys/site_variants/README.md").read_text(
            encoding=util.ENCODING
        )
        matrix_yaml = _load_yaml(
            Path("docs/axys-apx-reference/transaction_semantics_matrix.yaml")
        )
        fixture = Path(
            "tests/data/axys/site_variants/review_only_actions/"
            "ppar_performance_comparison.yaml"
        )

        self.assertTrue(fixture.exists())
        self.assertIn("review_only_actions", readme)
        self.assertIn("neutral review evidence", readme)
        self.assertEqual(
            matrix_yaml["rows"][";"]["fixtures"],
            ["site_variants/review_only_actions"],
        )
        self.assertIn(
            "synthetic corporate-action markers",
            matrix_yaml["rows"][";"]["coverage_notes"],
        )

    def test_transaction_semantics_matrix_matches_ambiguous_fixture_outputs(self) -> None:
        """Documented ambiguous-code semantics include IMEX-context fixture behavior."""
        matrix_yaml = _load_yaml(
            Path("docs/axys-apx-reference/transaction_semantics_matrix.yaml")
        )
        specification = PerformanceComparisonSpecification(
            Path(
                "tests/data/axys/site_variants/imex_context/"
                "ppar_performance_comparison.yaml"
            )
        )
        frame = TransactionsLoader(specification).load("a")
        assert frame is not None

        for code in matrix_yaml["ambiguous_external_flow_codes"]:
            rows = frame.filter(pl.col(schema.TRANSACTION_CODE) == code)
            documented = matrix_yaml["rows"][code]

            self.assertLessEqual(
                set(rows.get_column(schema.TRANSACTION_CATEGORY).to_list()),
                set(documented["ppar_categories"]),
                code,
            )
            self.assertLessEqual(
                set(rows.get_column(schema.CASH_FLOW_SIGN).to_list()),
                set(documented["cash_flow_signs"]),
                code,
            )
            self.assertLessEqual(
                set(rows.get_column(schema.PERFORMANCE_FLOW_SIGN).to_list()),
                set(documented["performance_flow_signs"]),
                code,
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

    def test_repository_guide_documents_demo_onboarding_commands(self) -> None:
        """The repository guide keeps the Axys demo smoke path discoverable."""
        guide = Path("docs/repository_guide.md").read_text(encoding=util.ENCODING)

        for expected_text in [
            "ppar.demos.performance_comparison_portfolio_demo",
            "ppar.demos.performance_comparison_security_demo",
            "ppar.performance_comparison.cli.validate_bundle",
            "ppar.performance_comparison.cli.validate_config",
            "ppar.performance_comparison.cli.validate_demo_matrix",
            "scripts/check_performance_comparison_demo_health.py",
            "_demo_output/performance_comparison_portfolio",
            "_demo_output/performance_comparison_security",
            "report.xlsx",
            "report_bundle_contract()",
            "All demo commands print the generated artifact paths",
        ]:
            with self.subTest(expected_text=expected_text):
                self.assertIn(expected_text, guide)

    def test_design_notes_document_both_user_facing_comparison_demos(self) -> None:
        """Design docs keep portfolio and security demo paths in sync."""
        design = Path("docs/performance_comparison_design.md").read_text(
            encoding=util.ENCODING
        )

        for expected_text in [
            "ppar.demos.performance_comparison_portfolio_demo",
            "ppar.demos.performance_comparison_security_demo",
            "_demo_output/performance_comparison_portfolio",
            "_demo_output/performance_comparison_security",
            "report.xlsx",
            "report.html",
            "review_summary.json",
            "report_bundle_contract()",
            "not a new transaction-classification or accounting layer",
        ]:
            with self.subTest(expected_text=expected_text):
                self.assertIn(expected_text, design)

    def test_axys_demo_resources_are_packaged(self) -> None:
        """The Axys demos use packaged resources instead of test fixtures."""
        axys_demo_data = files("ppar.demos.data") / "axys"
        expected_resources = (
            "README.md",
            "axys_column_mappings.yaml",
            "axys_analytics.yaml",
            "ppar_performance_comparison.yaml",
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

        transactions_a = _demo_transactions_by_natural_key(snapshot_a)
        transactions_b = _demo_transactions_by_natural_key(snapshot_b)
        holdings_a = _csv_rows_by_key(
            snapshot_a / "holdings.csv",
            ("PORT", "SEC", "HOLDING_DATE"),
        )
        holdings_b = _csv_rows_by_key(
            snapshot_b / "holdings.csv",
            ("PORT", "SEC", "HOLDING_DATE"),
        )

        changed_trade_cases = (
            (("ALPHA", "2026-03-05", "AAPL", "by"), "2026-03-31", ()),
            (("BALANCED", "2026-01-15", "MSFT", "sl"), "2026-01-30", ()),
        )
        for transaction_key, holding_date, other_cash_transaction_keys in changed_trade_cases:
            with self.subTest(transaction_key=transaction_key):
                transaction_a = transactions_a[transaction_key]
                transaction_b = transactions_b[transaction_key]
                portfolio = transaction_a["PORT"]
                security = transaction_a["SEC"]
                holding_a = holdings_a[(portfolio, security, holding_date)]
                holding_b = holdings_b[(portfolio, security, holding_date)]
                cash_a = holdings_a[(portfolio, "CASH_USD", holding_date)]
                cash_b = holdings_b[(portfolio, "CASH_USD", holding_date)]

                quantity_delta = _float_delta(transaction_a, transaction_b, "QTY")
                amount_delta = _float_delta(transaction_a, transaction_b, "AMOUNT")
                if transaction_a["TRAN"] == "sl":
                    holding_quantity_delta = -quantity_delta
                else:
                    holding_quantity_delta = quantity_delta
                month_end_price = float(holding_a["PRICE"])
                expected_cash_delta = amount_delta
                for cash_transaction_key in other_cash_transaction_keys:
                    expected_cash_delta += _float_delta(
                        transactions_a[cash_transaction_key],
                        transactions_b[cash_transaction_key],
                        "AMOUNT",
                    )

                self.assertAlmostEqual(
                    _float_delta(holding_a, holding_b, "QTY"),
                    holding_quantity_delta,
                    places=4,
                )
                self.assertAlmostEqual(
                    _float_delta(holding_a, holding_b, "MKT_VAL"),
                    holding_quantity_delta * month_end_price,
                    places=2,
                )
                self.assertAlmostEqual(
                    _float_delta(cash_a, cash_b, "MKT_VAL"),
                    expected_cash_delta,
                    places=2,
                )

    def test_axys_demo_buy_transaction_amounts_include_commission(self) -> None:
        """Packaged buy rows use a stable signed cash-amount convention."""
        axys_demo_data = files("ppar.demos.data") / "axys"
        for snapshot_name in ("axys_full_spec_a", "axys_full_spec_b"):
            with self.subTest(snapshot=snapshot_name):
                snapshot = Path(str(axys_demo_data / snapshot_name))
                with (snapshot / "transactions.csv").open(
                    encoding=util.ENCODING,
                    newline="",
                ) as file:
                    for row in csv.DictReader(file):
                        if row["TRAN"] != "by":
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

    def test_axys_demo_fee_transactions_reduce_cash_and_income(self) -> None:
        """Packaged fee changes reduce ending cash and cash security income."""
        axys_demo_data = files("ppar.demos.data") / "axys"
        snapshot_a = Path(str(axys_demo_data / "axys_full_spec_a"))
        snapshot_b = Path(str(axys_demo_data / "axys_full_spec_b"))
        transaction_key = ("INCOME", "2026-01-20", "CASH_USD", "dp")
        cash_holding_key = ("INCOME", "CASH_USD", "2026-01-30")
        cash_performance_key = (
            "INCOME",
            "CASH_USD",
            "2026-01-01",
            "2026-01-30",
        )

        transactions_a = _demo_transactions_by_natural_key(snapshot_a)
        transactions_b = _demo_transactions_by_natural_key(snapshot_b)
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

        amount_delta = _float_delta(
            transactions_a[transaction_key],
            transactions_b[transaction_key],
            "AMOUNT",
        )
        self.assertLess(amount_delta, 0)
        self.assertAlmostEqual(
            _float_delta(holdings_a[cash_holding_key], holdings_b[cash_holding_key], "MKT_VAL"),
            amount_delta,
            places=2,
        )
        self.assertAlmostEqual(
            _float_delta(
                secperf_a[cash_performance_key],
                secperf_b[cash_performance_key],
                "INCOME",
            ),
            amount_delta,
            places=2,
        )

    def test_axys_demo_omits_synthetic_future_splits(self) -> None:
        """Packaged full-spec demo avoids fictional future split transactions."""
        axys_demo_data = files("ppar.demos.data") / "axys"
        snapshot_a = Path(str(axys_demo_data / "axys_full_spec_a"))
        snapshot_b = Path(str(axys_demo_data / "axys_full_spec_b"))

        transactions_a = _demo_transactions_by_natural_key(snapshot_a)
        transactions_b = _demo_transactions_by_natural_key(snapshot_b)

        for rows in (transactions_a, transactions_b):
            split_rows = [
                row
                for row in rows.values()
                if row["TRAN"] == ";" and row["SEC"] == "TSLA"
            ]
            self.assertEqual(split_rows, [])

    def test_axys_demo_withdrawal_changes_cash_and_flow_return(self) -> None:
        """Packaged withdrawal changes cash, flow, and reconstructed return."""
        axys_demo_data = files("ppar.demos.data") / "axys"
        snapshot_a = Path(str(axys_demo_data / "axys_full_spec_a"))
        snapshot_b = Path(str(axys_demo_data / "axys_full_spec_b"))
        transaction_key = ("ALPHA", "2026-01-20", "CASH_USD", "wd")
        cash_holding_key = ("ALPHA", "CASH_USD", "2026-01-30")
        cash_performance_key = (
            "ALPHA",
            "CASH_USD",
            "2026-01-01",
            "2026-01-30",
        )
        portfolio_key = ("ALPHA", "2026-01-01", "2026-01-30")

        transactions_a = _demo_transactions_by_natural_key(snapshot_a)
        transactions_b = _demo_transactions_by_natural_key(snapshot_b)
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
        portperf_a = _csv_rows_by_key(
            snapshot_a / "portperf.csv",
            ("PORTFOLIO_CODE", "FROM_DATE", "THRU_DATE"),
        )
        portperf_b = _csv_rows_by_key(
            snapshot_b / "portperf.csv",
            ("PORTFOLIO_CODE", "FROM_DATE", "THRU_DATE"),
        )

        amount_delta = _float_delta(
            transactions_a[transaction_key],
            transactions_b[transaction_key],
            "AMOUNT",
        )
        self.assertEqual(transactions_a[transaction_key]["TRAN"], "wd")
        self.assertLess(amount_delta, 0)
        self.assertAlmostEqual(
            _float_delta(holdings_a[cash_holding_key], holdings_b[cash_holding_key], "MKT_VAL"),
            amount_delta,
            places=2,
        )
        self.assertAlmostEqual(
            _float_delta(
                secperf_a[cash_performance_key],
                secperf_b[cash_performance_key],
                "END_MV",
            ),
            amount_delta,
            places=2,
        )
        self.assertAlmostEqual(
            _float_delta(portperf_a[portfolio_key], portperf_b[portfolio_key], "FLOW"),
            amount_delta,
            places=2,
        )
        comparison_path = Path(str(axys_demo_data / "ppar_performance_comparison.yaml"))
        checks = _reconstruction_rows_by_key(
            portfolio_return_reconstruction_checks(comparison_path),
            ("portfolio_id", "from_date", "thru_date"),
        )
        reconstruction_row = checks[portfolio_key]
        self.assertEqual(
            reconstruction_row[RECONSTRUCTION_STATUS],
            RECONSTRUCTION_STATUS_ALIGNED,
        )
        self.assertAlmostEqual(
            _float_delta(
                portperf_a[portfolio_key],
                portperf_b[portfolio_key],
                "PORT_RETURN",
            ),
            _reconstruction_float(reconstruction_row, DERIVED_RETURN_DIFFERENCE),
            places=9,
        )

    def test_axys_demo_security_performance_reconciles_to_holdings(self) -> None:
        """Security performance demo rows stay consistent with holdings."""
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

    def test_axys_demo_portfolio_performance_matches_reconstruction(
        self,
    ) -> None:
        """Portfolio demo performance rows match configured reconstruction rules."""
        axys_demo_data = files("ppar.demos.data") / "axys"
        comparison_path = Path(str(axys_demo_data / "ppar_performance_comparison.yaml"))
        checks = _reconstruction_rows_by_key(
            portfolio_return_reconstruction_checks(comparison_path),
            ("portfolio_id", "from_date", "thru_date"),
        )
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
                        reconstruction_row = checks[key]
                        allowed_statuses = {
                            RECONSTRUCTION_STATUS_ALIGNED,
                            RECONSTRUCTION_STATUS_MISSING_INPUTS,
                        }
                        if key in _INTENTIONAL_PORTFOLIO_RECONSTRUCTION_DIFFERENT_KEYS:
                            allowed_statuses.add(RECONSTRUCTION_STATUS_DIFFERENT)
                        self.assertIn(
                            reconstruction_row[RECONSTRUCTION_STATUS],
                            allowed_statuses,
                        )
                        if (
                            reconstruction_row[RECONSTRUCTION_STATUS]
                            == RECONSTRUCTION_STATUS_ALIGNED
                        ):
                            self.assertAlmostEqual(
                                _reconstruction_float(
                                    reconstruction_row,
                                    REPORTED_RETURN_DIFFERENCE,
                                ),
                                _reconstruction_float(
                                    reconstruction_row,
                                    DERIVED_RETURN_DIFFERENCE,
                                ),
                                places=9,
                            )

    def test_axys_demo_security_return_deltas_match_reconstruction(self) -> None:
        """Security demo return deltas match configured reconstruction rules."""
        axys_demo_data = files("ppar.demos.data") / "axys"
        snapshot_a = Path(str(axys_demo_data / "axys_full_spec_a"))
        snapshot_b = Path(str(axys_demo_data / "axys_full_spec_b"))
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
        comparison_path = Path(str(axys_demo_data / "ppar_performance_comparison.yaml"))
        checks = _reconstruction_rows_by_key(
            security_return_reconstruction_checks(comparison_path),
            ("portfolio_id", "security_id", "from_date", "thru_date"),
        )

        for portfolio in ("ALPHA", "BALANCED", "INCOME"):
            with self.subTest(portfolio=portfolio, security="AAPL"):
                key = (portfolio, "AAPL", "2026-05-01", "2026-05-29")
                holding_key = (portfolio, "AAPL", "2026-05-29")
                price_delta = _float_delta(
                    holdings_a[holding_key],
                    holdings_b[holding_key],
                    "PRICE",
                )
                self.assertGreater(price_delta, 0.0)
                self.assertAlmostEqual(
                    _float_delta(secperf_a[key], secperf_b[key], "SEC_RETURN"),
                    _reconstruction_float(checks[key], DERIVED_RETURN_DIFFERENCE),
                    places=9,
                )
                self.assertEqual(
                    checks[key][RECONSTRUCTION_STATUS],
                    RECONSTRUCTION_STATUS_ALIGNED,
                )

        tnote_key = ("INCOME", "TNOTE2Y", "2026-05-01", "2026-05-29")
        tnote_holding_key = ("INCOME", "TNOTE2Y", "2026-05-29")
        self.assertNotEqual(
            _float_delta(
                holdings_a[tnote_holding_key],
                holdings_b[tnote_holding_key],
                "MKT_VAL",
            ),
            0.0,
        )
        self.assertAlmostEqual(
            _float_delta(secperf_a[tnote_key], secperf_b[tnote_key], "SEC_RETURN"),
            _reconstruction_float(checks[tnote_key], DERIVED_RETURN_DIFFERENCE),
            places=9,
        )
        self.assertEqual(
            checks[tnote_key][RECONSTRUCTION_STATUS],
            RECONSTRUCTION_STATUS_ALIGNED,
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

    def test_minimum_source_data_contract_is_documented(self) -> None:
        """The source-data contract helper and user-facing docs stay aligned."""
        root_readme = Path("README.md").read_text(encoding=util.ENCODING)
        contract_doc = Path("docs/performance_comparison_demo_source_contract.md").read_text(
            encoding=util.ENCODING
        )
        contracts = performance_source_data_contract.source_data_contract()

        self.assertIn("Minimum source-data contract:", root_readme)
        self.assertIn("required normalized", root_readme)
        self.assertIn("columns must resolve unambiguously", root_readme)
        self.assertIn("## Minimum Source-Data Contract", contract_doc)
        self.assertIn("stops before producing a report", contract_doc)
        self.assertIn("return reconstruction is configured", contract_doc)
        for contract in contracts:
            with self.subTest(dataset=contract.name):
                self.assertIn(f"`{contract.name}`", contract_doc)
                for column in contract.required_columns:
                    self.assertIn(f"`{column}`", contract_doc)

    def test_source_data_contract_module_exports_public_helpers(self) -> None:
        """The source-data contract module exports only its public helpers."""
        expected_exports = {
            "SourceDataDatasetContract",
            "comparison_required_dataset_names",
            "source_data_contract",
            "source_data_contract_summary",
        }

        self.assertEqual(set(performance_source_data_contract.__all__), expected_exports)

    def test_source_data_contract_summary_includes_reconstruction_sources(self) -> None:
        """Contract summaries name formula-source datasets when requested."""
        summary = performance_source_data_contract.source_data_contract_summary(
            include_reconstruction_sources=True,
            include_security_performance=True,
        )

        self.assertEqual(
            summary["required_datasets"],
            "holdings, portfolio_performance, security_performance, transactions",
        )
        self.assertIn(
            "holdings: portfolio_id, security_id, holding_date",
            summary["required_columns"],
        )
        self.assertIn(
            "security_performance: portfolio_id, security_id, from_date, "
            "thru_date, security_return",
            summary["required_columns"],
        )
        self.assertIn(
            "transactions: portfolio_id, security_id, transaction_date",
            summary["required_columns"],
        )

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
            "report_bundle_contract": report_bundle_contract,
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

    def test_performance_comparison_boundary_helpers_are_submodules(self) -> None:
        """Boundary helper modules are importable without top-level export churn."""
        top_level_exports = set(performance_comparison.__all__)
        helper_modules = {
            "backlog_gates": performance_backlog_gates,
            "fixed_income": performance_fixed_income,
            "source_data_contract": performance_source_data_contract,
            "transaction_boundary_registry": performance_boundary_registry,
            "transaction_summary": performance_transaction_summary,
        }

        self.assertEqual(
            performance_fixed_income.fixed_income_transaction_boundary("in"),
            "safe_income",
        )
        self.assertEqual(
            performance_backlog_gates.transaction_backlog_gate("rc"),
            "capital_return_policy",
        )
        self.assertIn(
            "packaged_formula",
            performance_boundary_registry.transaction_boundary_groups("by"),
        )
        self.assertTrue(
            hasattr(performance_transaction_summary, "transaction_semantics_summary")
        )
        self.assertTrue(top_level_exports.isdisjoint(helper_modules))

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
