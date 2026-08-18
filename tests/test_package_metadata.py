"""Tests for package metadata maintained alongside runtime dependencies."""

# Python Imports
import ast
import csv
from datetime import date
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
import ppar
import ppar.analytics.schema as core_schema
import ppar.common as common
import ppar.errors as core_errors
import ppar.utilities as util
from ppar import audit, axys_apx
from ppar.audit import data_issues, performance_comparison
from ppar.axys_apx import (
    AxysClassificationSources,
    AxysData,
    AxysPortfolio,
    AxysSpecification,
)
from ppar.audit import report as audit_report
from ppar.audit import runner as audit_runner
from ppar.audit.performance_comparison import findings as performance_comparison_findings
from ppar.audit.performance_comparison.methods import TransactionImpactMethod
from ppar.audit import transactions as audit_transactions
from ppar.audit import (
    transaction_summary as performance_transaction_summary,
)
from ppar.audit import (
    source_data_contract as performance_source_data_contract,
)
from ppar.axys_apx.transaction_safety import (
    AMBIGUOUS_FLOW_TRANSACTION_CODES,
)
from scripts.transaction_policy_evidence import (
    TRANSACTION_EVIDENCE_GROUPS,
    registered_transaction_evidence_codes,
    transaction_evidence_groups,
)
from ppar.audit.performance_comparison.return_reconstruction import (
    DERIVED_RETURN_DIFFERENCE,
    RECONSTRUCTION_STATUS,
    RECONSTRUCTION_STATUS_ALIGNED,
    RECONSTRUCTION_STATUS_DIFFERENT,
    RECONSTRUCTION_STATUS_MISSING_INPUTS,
    REPORTED_RETURN_DIFFERENCE,
    portfolio_return_reconstruction_checks,
    security_return_reconstruction_checks,
)
from ppar.audit import (
    ComparisonFile,
    ComparisonSnapshot,
    AuditSpecification,
    PortfolioPerformanceLoader,
    HoldingsLoader,
    REPORT_BUNDLE_REQUIRED_ARTIFACTS,
    SecurityPerformanceLoader,
    TransactionsLoader,
    schema,
    compact_findings_table,
    compare_snapshots,
    report_bundle_contract,
    report_bundle_validation_issues,
    summarize_findings,
    validate_causal_attribution_ready,
    validate_yaml_setup_complete,
    write_audit_report_bundle,
    write_audit_review_workbook,
)
from ppar.audit.data_issues import (
    DATA_ISSUE_REGISTRY,
    DataIssueCategory,
    DataIssueDefinition,
    DataIssueType,
)
from ppar.audit.performance_comparison import (
    CONTEXT,
    DIRECT_INPUT,
    EVIDENCE_ROLE,
    RELATED_OUTPUT,
    TARGET_OUTPUT,
    CauseArea,
    Finding,
    PerformanceComparison,
    SuppressionRule,
    apply_suppressions,
    findings_to_polars,
    portfolio_period_cause_summary,
    portfolio_period_contribution_candidates,
    portfolio_period_evidence_breakdown,
    portfolio_period_impact_coverage_summary,
    portfolio_period_summary,
    portfolio_period_transaction_cross_checks,
    rank_portfolio_period_evidence,
    security_period_evidence_breakdown,
    security_period_summary,
    transaction_activity_summary,
    transaction_matching_diagnostics,
)

_INTENTIONAL_PORTFOLIO_RECONSTRUCTION_DIFFERENT_KEYS = {
    ("BALANCED", "2026-05-09", "2026-05-14"),
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
    internal_by_common = {
        "Portfolio Code": "PORTFOLIO_CODE",
        "Security Symbol": "SECURITY_ID",
        "Holding Date": "HOLDING_DATE",
        "From Date": "FROM_DATE",
        "Thru Date": "THRU_DATE",
        "Transaction Date": "TRANSACTION_DATE",
        "Transaction Code": "TRAN",
        "Quantity": "QTY",
        "Price": "PRICE",
        "Beginning Market Value": "BEGIN_MV",
        "Ending Market Value": "END_MV",
        "Beginning Weight": "BEGIN_WEIGHT",
        "Security Return": "SEC_RETURN",
        "Contribution": "CONTRIBUTION",
        "Market Value": "MKT_VAL",
        "Base Market Value": "BASE_MKT_VAL",
        "Amount": "AMOUNT",
        "Commission": "COMMISSION",
        "Net Flow": "FLOW",
        "Income": "INCOME",
        "Portfolio Return": "PORT_RETURN",
    }
    with path.open(encoding=util.ENCODING, newline="") as file:
        rows: dict[tuple[str, ...], dict[str, str]] = {}
        for source_row in csv.DictReader(file):
            row = dict(source_row)
            for common_column, internal_column in internal_by_common.items():
                if common_column in source_row:
                    row[internal_column] = source_row[common_column]
            if "Portfolio Code" in source_row:
                row["PORT"] = source_row["Portfolio Code"]
            if "Security Symbol" in source_row:
                row["SEC"] = source_row["Security Symbol"]
            rows[tuple(row[column] for column in key_columns)] = row
        return rows


def _load_yaml(path: Path) -> dict[str, object]:
    """Return a YAML mapping loaded from a repository file."""
    with path.open(encoding=util.ENCODING) as file:
        loaded = yaml.safe_load(file)
    if not isinstance(loaded, dict):
        raise AssertionError(f"Expected YAML mapping in {path}.")
    return loaded


def _yaml_mapping(value: object, *, label: str) -> dict[str, object]:
    """Return a nested YAML mapping with a useful assertion message."""
    if not isinstance(value, dict):
        raise AssertionError(f"Expected YAML mapping for {label}.")
    return cast(dict[str, object], value)


def _yaml_mapping_rows(value: object, *, label: str) -> dict[str, dict[str, object]]:
    """Return a YAML mapping whose values are row mappings."""
    rows = _yaml_mapping(value, label=label)
    return {
        str(key): _yaml_mapping(row, label=f"{label}.{key}")
        for key, row in rows.items()
    }


def _yaml_string_list(value: object, *, label: str) -> list[str]:
    """Return a YAML list of strings with a useful assertion message."""
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise AssertionError(f"Expected YAML string list for {label}.")
    return cast(list[str], value)


def _package_data_patterns(pyproject: dict[str, object]) -> list[str]:
    """Return package-data patterns as paths relative to the repository root."""
    setuptools_config = _yaml_mapping(pyproject["tool"], label="tool")
    setuptools_config = _yaml_mapping(
        setuptools_config["setuptools"],
        label="tool.setuptools",
    )
    package_data = _yaml_mapping(
        setuptools_config["package-data"],
        label="tool.setuptools.package-data",
    )
    patterns: list[str] = []
    for package_name, package_patterns in package_data.items():
        package_path = str(package_name).replace(".", "/")
        for pattern in _yaml_string_list(
            package_patterns,
            label=f"tool.setuptools.package-data.{package_name}",
        ):
            patterns.append(f"{package_path}/{pattern}")
    return patterns


def _transaction_codes_in_csv(path: Path) -> set[str]:
    """Return lowercase transaction codes from a fixture CSV."""
    with path.open(encoding=util.ENCODING, newline="") as file:
        rows = list(csv.DictReader(file))
    code_column = "Transaction Code" if rows and "Transaction Code" in rows[0] else "TRAN"
    return {
        row[code_column].strip().lower()
        for row in rows
        if row[code_column].strip()
    }


def _demo_transactions_by_natural_key(
    snapshot_path: Path,
) -> dict[tuple[str, str, str, str], dict[str, str]]:
    """Return packaged Axys transactions keyed by visible row attributes."""
    return cast(
        dict[tuple[str, str, str, str], dict[str, str]],
        _csv_rows_by_key(
            snapshot_path / "transactions.csv",
            ("PORT", "TRANSACTION_DATE", "SEC", "TRAN"),
        ),
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


def _normalized_requirement_name(requirement: str) -> str:
    """Return one PEP 503-style package name from a requirement or constraint."""
    name = re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0].strip()
    return re.sub(r"[-_.]+", "-", name).lower()


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
        analytics_dependencies = {
            dependency.split(">=", maxsplit=1)[0].lower()
            for dependency in optional_dependencies["analytics"]
        }

        self.assertEqual(pyproject["project"]["version"], "0.1.5")
        self.assertEqual(ppar.__version__, pyproject["project"]["version"])
        self.assertNotIn("great_tables", pyproject_dependencies)
        self.assertIn("pyyaml", pyproject_dependencies)
        self.assertIn("openpyxl", pyproject_dependencies)
        self.assertNotIn("matplotlib", pyproject_dependencies)
        self.assertNotIn("numpy", pyproject_dependencies)
        self.assertNotIn("pandas", pyproject_dependencies)
        self.assertNotIn("pyarrow", pyproject_dependencies)
        self.assertNotIn("seaborn", pyproject_dependencies)
        self.assertNotIn("scipy", pyproject_dependencies)
        self.assertEqual(
            analytics_dependencies,
            {
                "lxml",
                "matplotlib",
                "numpy",
                "pandas",
                "pyarrow",
                "seaborn",
            },
        )
        self.assertIn("polars>=1.24.0", pyproject["project"]["dependencies"])
        self.assertNotIn("charts", optional_dependencies)
        self.assertNotIn("excel", optional_dependencies)
        self.assertIn("pytest", dev_dependencies)
        self.assertNotIn("scipy-stubs", dev_dependencies)

    def test_ci_constraints_cover_every_direct_requirement(self) -> None:
        """The reproducible CI environment pins every declared requirement."""
        with open("pyproject.toml", "rb") as file:
            pyproject = tomllib.load(file)
        project = pyproject["project"]
        direct_requirements = [
            *pyproject["build-system"]["requires"],
            *project["dependencies"],
            *(
                requirement
                for group in project["optional-dependencies"].values()
                for requirement in group
            ),
        ]
        constraint_lines = Path("constraints/ci.txt").read_text(
            encoding=util.ENCODING
        ).splitlines()
        pinned_constraints = {
            _normalized_requirement_name(line)
            for line in constraint_lines
            if line and not line.startswith("#")
        }

        self.assertTrue(
            {
                _normalized_requirement_name(requirement)
                for requirement in direct_requirements
            }.issubset(pinned_constraints)
        )
        self.assertTrue(
            all(
                "==" in line
                for line in constraint_lines
                if line and not line.startswith("#")
            )
        )

    def test_ci_runs_the_maintained_release_candidate_gate(self) -> None:
        """CI delegates to the deterministic runner that owns the 500x gate."""
        workflow = Path(".github/workflows/release-candidate.yml").read_text(
            encoding=util.ENCODING
        )

        self.assertIn("constraints/ci.txt", workflow)
        self.assertIn("--editable \".[analytics,dev]\"", workflow)
        self.assertIn("scripts/check_release_candidate.py", workflow)
        self.assertIn("scripts/check_project.py", workflow)
        self.assertIn("--build", workflow)
        self.assertNotIn("--skip-project-check", workflow)

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
            Path("ppar/audit"),
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
        demo_guide = Path("docs/audit/packaged_demo.md").read_text(
            encoding=util.ENCODING
        )
        normalized_demo_guide = " ".join(demo_guide.split())

        self.assertIn("portfolio_audit.xlsx", root_readme)
        self.assertNotIn("Open `portfolio_audit.xlsx` when present", root_readme)
        self.assertNotIn("CSV artifacts support audit traceability", root_readme)
        self.assertIn("html audit for browser review", demo_guide.lower())
        self.assertIn("CSV artifacts", demo_guide)
        self.assertIn("supporting detail and traceability", demo_guide)
        self.assertNotIn("same review model in a browser", root_readme)
        self.assertNotIn("same review model in a browser", demo_guide)
        self.assertIn("Open `portfolio_audit.xlsx`", normalized_demo_guide)

    def test_manifest_keeps_source_distribution_resources(self) -> None:
        """The source distribution manifest includes checkout scripts and demo data."""
        manifest = Path("MANIFEST.in").read_text(encoding=util.ENCODING)

        self.assertIn("include scripts/*.py", manifest)
        self.assertIn("recursive-include ppar/setup_templates *.csv *.yaml *.md *.py", manifest)
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
        package_data_patterns = _package_data_patterns(pyproject)

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

    def test_demo_maintainer_modules_stay_out_of_runtime_package(self) -> None:
        """Demo construction helpers live with checkout scripts, not in the wheel."""
        for retired_path in (
            Path("ppar/_demo_market_data.py"),
            Path("ppar/_demo_operational_holdings.py"),
        ):
            with self.subTest(retired_path=retired_path.as_posix()):
                self.assertFalse(retired_path.exists())

        for maintainer_path in (
            Path("scripts/demo_support/market_data.py"),
            Path("scripts/operational_demo_data/holdings.py"),
        ):
            with self.subTest(maintainer_path=maintainer_path.as_posix()):
                self.assertTrue(maintainer_path.is_file())

    def test_package_data_patterns_cover_demo_resources(self) -> None:
        """Every packaged demo resource is covered by explicit package-data globs."""
        with open("pyproject.toml", "rb") as file:
            pyproject = tomllib.load(file)
        package_data_patterns = _package_data_patterns(pyproject)
        demo_resource_paths = [
            path.as_posix()
            for path in Path("ppar/setup_templates").rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
            and path.name != "__init__.py"
        ]

        self.assertGreater(len(demo_resource_paths), 0)
        for resource_path in demo_resource_paths:
            with self.subTest(resource_path=resource_path):
                self.assertTrue(
                    any(fnmatch(resource_path, pattern) for pattern in package_data_patterns),
                    f"{resource_path} is not covered by package-data patterns.",
                )

    def test_packaged_axys_demo_files_are_product_inputs(self) -> None:
        """Packaged Axys/APX demo files stay limited to user-facing setup assets."""
        allowed_suffixes = {".csv", ".md", ".py", ".yaml"}
        axys_demo_files = [
            path
            for path in Path("ppar/setup_templates/axys_apx_audit").rglob("*")
            if path.is_file()
        ]

        self.assertGreater(len(axys_demo_files), 0)
        for path in axys_demo_files:
            with self.subTest(path=path.as_posix()):
                self.assertIn(path.suffix, allowed_suffixes)
                self.assertNotIn("_demo_output", path.as_posix())
                self.assertNotIn("operational_demo_data", path.as_posix())

    def test_packaged_axys_yaml_documents_onboarding_boundaries(self) -> None:
        """The packaged Axys/APX YAML names user-facing onboarding boundaries."""
        yaml_text = Path("ppar/setup_templates/axys_apx_audit/axys_apx_audit.yaml").read_text(
            encoding=util.ENCODING
        )

        for expected_text in [
            "The default command to run Audit is",
            "To customize with your own data",
            "Ambiguous transaction codes must not become external flows",
            "If your snapshot directory names differ, edit snapshots.*.path below",
            "If your CSV filenames or column names differ, edit files.*.path or",
            "rules and impact methods below unchanged",
            "Every setting has a\n  # documented default and may be omitted",
            'Default: "Portfolio Audit Report" and "Security Audit Report"',
            "One-run modes that omit XLSX: --html-only or --csv-only",
            "Usually the original/older source-data extract",
            "Portfolio performance CSV; required",
            "Required columns: Portfolio Code, From Date, Thru Date",
            "Optional columns: Base Currency",
            "Holdings CSV; required",
            "Holding Date, Market Value",
            "Transactions CSV; required",
            "Transaction Date, Transaction Code, Amount",
            "# Data Issues",
            "# Portfolio Performance Calculation",
            "# Security Performance Calculation",
            "# Transaction Rules",
            "Default: no internal code meanings",
            "code-only ti remains unsupported",
            "# Transaction Impact Methods",
            "# Security Return Impact Methods",
            "# Holding Impact Methods",
            "# Price Impact Methods",
            "# Comparison Tolerances",
            "Minimum tolerances used by Audit",
            "# Appendix — Additional Supported Parameters",
            "# Parameter: audit.reconstruction_diagnostics",
            "# Parameter: data_issues.deliver_in_original_cost_incomplete",
            "# Parameter: suppressions",
            "All values are required",
        ]:
            with self.subTest(expected_text=expected_text):
                self.assertIn(expected_text, yaml_text)

        section_order = [
            "# Audit Runtime Parameters",
            "# Audit Snapshots",
            "# Audit Source CSV Files",
            "# Data Issues",
            "# Comparison Tolerances",
            "# Transaction Rules",
            "# Portfolio Performance Calculation",
            "# Security Performance Calculation",
            "# Transaction Impact Methods",
            "# Appendix — Additional Supported Parameters",
        ]
        self.assertEqual(
            [yaml_text.index(section) for section in section_order],
            sorted(yaml_text.index(section) for section in section_order),
        )

        self.assertNotIn(
            "Command-line override: ppar audit",
            yaml_text,
        )
        self.assertNotIn("comparison:", yaml_text)
        configuration = _load_yaml(
            Path("ppar/setup_templates/axys_apx_audit/axys_apx_audit.yaml")
        )
        audit_settings = _yaml_mapping(configuration["audit"], label="audit")
        appendix_parameters = set(
            re.findall(r"^# Parameter: ([^\n]+)$", yaml_text, re.M)
        )
        runtime_appendix_settings = {
            parameter.removeprefix("audit.")
            for parameter in appendix_parameters
            if parameter.startswith("audit.")
        }
        self.assertEqual(audit_settings, {"output_directory": "output"})
        self.assertEqual(
            runtime_appendix_settings,
            {
                "exclude_suppressed",
                "html_output",
                "reconstruction_diagnostics",
                "require_causal_attribution",
                "title",
                "xlsx_output",
            },
        )
        self.assertEqual(
            set(audit_settings) | runtime_appendix_settings,
            {
                "output_directory",
                "title",
                "xlsx_output",
                "html_output",
                "exclude_suppressed",
                "reconstruction_diagnostics",
                "require_causal_attribution",
            },
        )
        self.assertEqual(
            appendix_parameters - {f"audit.{name}" for name in runtime_appendix_settings},
            {
                "data_issues.<issue_type>.exclude",
                "data_issues.deliver_in_original_cost_incomplete",
                "data_issues.large_price_variation.rules[].enabled",
                "extract_contract.enforce_ambiguous_axys_flows",
                "extract_contract.path",
                "files.<optional_dataset>.required",
                "snapshots.<snapshot>.schema",
                "suppressions",
            },
        )
        data_issues = _yaml_mapping(
            configuration["data_issues"],
            label="data_issues",
        )
        self.assertNotIn("deliver_in_original_cost_incomplete", data_issues)
        self.assertEqual(
            list(data_issues),
            [
                "enabled",
                "dividend_rate",
                "duplicate_transactions",
                "holdings_accrued_rate",
                "holdings_nonpositive_price",
                "holdings_price_range",
                "holdings_stale_price",
                "large_price_variation",
                "missing_dividend",
                "pa_sa_rate",
                "transactions_nonpositive_price",
                "transactions_price_range",
                "transaction_security_type_mismatch",
            ],
        )
        tolerances = _yaml_mapping(
            configuration["tolerances"],
            label="tolerances",
        )
        self.assertEqual(list(tolerances), sorted(tolerances))
        transaction_rules = _yaml_mapping(
            configuration["transaction_rules"],
            label="transaction_rules",
        )
        self.assertEqual(list(transaction_rules), sorted(transaction_rules))
        for documented_default in (
            "#   title: null",
            "#   xlsx_output: true",
            "#   html_output: true",
            "#   exclude_suppressed: false",
            "#   reconstruction_diagnostics: false",
            "#   require_causal_attribution: false",
        ):
            with self.subTest(documented_default=documented_default):
                self.assertIn(documented_default, yaml_text)
        for line_number, line in enumerate(yaml_text.splitlines(), start=1):
            if line.lstrip().startswith("#"):
                self.assertLessEqual(
                    len(line),
                    80,
                    f"Audit YAML comment line {line_number} exceeds 80 characters.",
                )
        file_positions = [
            yaml_text.index("  portfolio_performance:"),
            yaml_text.index("  security_performance:"),
            yaml_text.index("  holdings:"),
            yaml_text.index("  transactions:"),
            yaml_text.index("  security_master:"),
            yaml_text.index("  splits:"),
        ]
        self.assertEqual(file_positions, sorted(file_positions))
        self.assertLess(
            yaml_text.index("  portfolio_performance:"),
            yaml_text.index("  security_performance:"),
        )

    def test_packaged_axys_analytics_yaml_documents_onboarding_boundaries(self) -> None:
        """The packaged Axys/APX analytics YAML names first-pass setup choices."""
        yaml_text = Path(
            "ppar/setup_templates/axys_apx_analytics/axys_apx_analytics.yaml"
        ).read_text(encoding=util.ENCODING)

        for expected_text in [
            "PPAR Axys/APX Analytics Configuration",
            "To customize with your own data",
            "your own IMEX/REP",
            "edit files.*.path or",
            "files.*.columns below",
            "Portfolio-level performance CSV file.",
            "Security-level performance CSV file.",
            "Default: portperf.csv",
            "Default: secperf.csv",
            "Default: secmast.csv",
            "documented defaults may be omitted.",
            "Default: Use the performance periods as supplied",
            "Risk Statistics require a fixed frequency and are omitted",
            "frequency: quarterly",
            "holidays: holidays.csv",
            "Optional headerless file containing one YYYY-MM-DD holiday per line.",
            "Command-line override: --holidays ./client_holidays.csv",
            "Command-line override: --annual-risk-free-rate 0.04",
            "Command-line override: --annual-minimum-acceptable-return 0.01",
            "Command-line override: --output-directory ./review_output",
            "Analytics Runtime Parameters",
            "Analytics Source CSV Files",
            "Reporting Classification Mappings",
            "Appendix — Additional Supported Parameters",
            "Parameter: analytics.annual_minimum_acceptable_return",
            "Parameter: analytics.confidence_level",
            "Parameter: analytics.portfolio_value",
            "Parameter: analytics.currency_symbol",
            "#   annual_minimum_acceptable_return: 0.00",
            "#   confidence_level: 0.95",
            "#   portfolio_value: 100000",
            '#   currency_symbol: "$"',
            'security_name: "Security Name"',
            "mappings",
            "classification_column",
            "display_name_column",
        ]:
            with self.subTest(expected_text=expected_text):
                self.assertIn(expected_text, yaml_text)

        self.assertNotIn(
            "Command-line override: ppar analytics",
            yaml_text,
        )
        self.assertEqual(yaml_text.count("  # Default:"), 12)
        self.assertEqual(yaml_text.count("# Default:"), 16)
        configuration = _load_yaml(
            Path("ppar/setup_templates/axys_apx_analytics/axys_apx_analytics.yaml")
        )
        analytics = _yaml_mapping(configuration["analytics"], label="analytics")
        appendix_settings = {
            parameter.removeprefix("analytics.")
            for parameter in re.findall(r"^# Parameter: (analytics\.[a-z_]+)$", yaml_text, re.M)
        }
        self.assertEqual(
            appendix_settings,
            {
                "annual_minimum_acceptable_return",
                "confidence_level",
                "currency_symbol",
                "portfolio_value",
            },
        )
        self.assertEqual(
            set(analytics) | appendix_settings,
            {
                "portfolio",
                "benchmark",
                "frequency",
                "holidays",
                "output_directory",
                "from_date",
                "thru_date",
                "classification",
                "annual_minimum_acceptable_return",
                "annual_risk_free_rate",
                "confidence_level",
                "portfolio_value",
                "currency_symbol",
            },
        )
        self.assertNotIn("annual_minimum_acceptable_return", analytics)
        self.assertNotIn("confidence_level", analytics)
        self.assertNotIn("portfolio_value", analytics)
        self.assertNotIn("currency_symbol", analytics)
        source_layout_positions = [
            yaml_text.index("  portfolio_performance:"),
            yaml_text.index("    path: portperf.csv"),
            yaml_text.index("  security_performance:"),
            yaml_text.index("    path: secperf.csv"),
            yaml_text.index("  security_master:"),
            yaml_text.index("    path: secmast.csv"),
            yaml_text.index("mappings:"),
        ]
        self.assertEqual(source_layout_positions, sorted(source_layout_positions))

    def test_packaged_audit_mappings_omit_analytics_classifications(self) -> None:
        """The Audit starter does not advertise unused analytics groupings."""
        configuration = _load_yaml(
            Path("ppar/setup_templates/axys_apx_audit/axys_apx_audit.yaml")
        )
        file_definitions = configuration["files"]
        self.assertIsInstance(file_definitions, dict)
        assert isinstance(file_definitions, dict)
        self.assertEqual(
            set(file_definitions),
            {
                "portfolio_performance",
                "security_performance",
                "security_master",
                "holdings",
                "transactions",
                "splits",
            },
        )
        for definition in file_definitions.values():
            self.assertEqual(set(definition), {"path", "columns"})
        self.assertNotIn("security_id", configuration)
        self.assertNotIn("defaults", configuration)
        self.assertNotIn("classification", configuration)
        self.assertNotIn("mappings", configuration)
        self.assertNotIn("extract_contract", configuration)

        data_issues = configuration["data_issues"]
        assert isinstance(data_issues, dict)
        targeted_security_rules = {
            issue_type
            for issue_type, settings in data_issues.items()
            if isinstance(settings, dict)
            and isinstance(settings.get("only"), dict)
            and "security_id" in settings["only"]
        }
        self.assertEqual(targeted_security_rules, set())
        reference_filtered_rules = {
            issue_type
            for issue_type, settings in data_issues.items()
            if isinstance(settings, dict)
            and isinstance(settings.get("only"), dict)
            and any(
                str(field).startswith("security_master.")
                for field in settings["only"]
            )
        }
        self.assertTrue(
            {"dividend_rate", "missing_dividend", "holdings_accrued_rate"}
            <= reference_filtered_rules
        )

        for snapshot_name in ("snapshot_a", "snapshot_b"):
            snapshot = Path("ppar/setup_templates/axys_apx_audit") / snapshot_name
            for definition in file_definitions.values():
                path = definition["path"]
                columns = definition["columns"]
                assert isinstance(path, str)
                assert isinstance(columns, dict)
                with (snapshot / path).open(
                    encoding=util.ENCODING,
                    newline="",
                ) as file:
                    headings = next(csv.reader(file))
                self.assertEqual(list(columns.values()), headings)

    def test_user_facing_setup_docs_avoid_retired_command_language(self) -> None:
        """Installed-user docs stay aligned with the current setup/report commands."""
        docs_to_check = [
            Path("README.md"),
            Path("ppar/setup_templates/README.md"),
            Path("ppar/setup_templates/axys_apx_audit/README.md"),
            Path(
                "ppar/setup_templates/axys_apx_audit/"
                "axys_apx_audit.yaml"
            ),
            Path("ppar/setup_templates/axys_apx_analytics/axys_apx_analytics.yaml"),
        ]
        retired_patterns = {
            "old roadmap filename": r"performance_comparison_roadmap",
            "quickstart naming": r"\bquickstart\b|QUICK_START|quick start",
            "old report command": r"\bppar report\b|ppar-report",
            "old full-spec folders": r"axys_full_spec",
            "old performance-comparison demo command": r"ppar-performance-comparison",
            "old generic analytics console script": r"ppar-generic-analytics-demo",
        }

        for doc_path in docs_to_check:
            doc_text = doc_path.read_text(encoding=util.ENCODING)
            for label, pattern in retired_patterns.items():
                with self.subTest(doc_path=doc_path.as_posix(), retired=label):
                    self.assertIsNone(re.search(pattern, doc_text, flags=re.IGNORECASE))

    def test_packaged_axys_setup_documents_onboarding_path(self) -> None:
        """The packaged Axys setup stays action-oriented for site onboarding."""
        readme = Path("ppar/setup_templates/axys_apx_audit/README.md").read_text(
            encoding=util.ENCODING
        )

        for expected_text in [
            "# PPAR Audit Workspace",
            "ppar audit .",
            "output/portfolio",
            "output/security",
            "README.md",
            "run_audit.py",
            "Customizing With Your Own Data",
            "Fully Explained",
        ]:
            with self.subTest(expected_text=expected_text):
                self.assertIn(expected_text, readme)

        self.assertNotIn("SETUP.md", readme)
        self.assertNotIn("ppar setup --guide", readme)
        self.assertNotIn("ppar analytics", readme)
        self.assertNotIn("Performance Analytics", readme)
        self.assertTrue(Path("ppar/setup_templates/axys_apx_analytics/run_analytics.py").exists())
        self.assertTrue(
            Path(
                "ppar/setup_templates/axys_apx_audit/"
                "run_audit.py"
            ).exists()
        )

    def test_eventual_axys_integration_reference_is_documented(self) -> None:
        """The Axys/APX index parks advanced reference publishing explicitly."""
        reference_index = " ".join(
            Path("docs/axys_apx/README.md")
            .read_text(encoding=util.ENCODING)
            .split()
        )

        self.assertIn(
            "Publishing an advanced Axys/APX integration reference remains "
            "a candidate",
            reference_index,
        )
        self.assertIn("should not copy the full research archive", reference_index)
        self.assertIn("available to installed-package users", reference_index)

    def test_common_core_export_reference_avoids_unverified_imex_recipes(self) -> None:
        """The export-planning note does not present invented native profiles."""
        reference = Path("docs/axys_apx/axys_apx_common_core_export.md").read_text(
            encoding=util.ENCODING
        )

        self.assertIn("not an official", reference)
        self.assertIn("## Extraction Planning Worksheet", reference)
        self.assertIn("PPAR-normalized filenames", reference)
        self.assertIn("REP performance report preferred", reference)
        self.assertIn("## Field and Contract Ownership", reference)
        self.assertIn("Chapter_15_Data_Dictionary.md", reference)
        self.assertIn("demo_extract_availability.md", reference)
        self.assertIn("transaction_semantics_matrix.yaml", reference)
        self.assertNotIn("PORTPERF_COMMON", reference)
        self.assertNotIn("SECPERF_COMMON", reference)
        self.assertNotIn("Axys/APX Native Dataset", reference)

    def test_user_installed_analytics_paths_do_not_depend_on_demos(self) -> None:
        """Installed analytics entrypoints avoid maintainer demo helper modules."""
        paths = [
            Path("ppar/analytics/cli.py"),
            Path("ppar/setup_templates/axys_apx_analytics/run_analytics.py"),
            Path("ppar/setup_templates/generic_analytics/run_generic_analytics.py"),
        ]

        for path in paths:
            with self.subTest(path=path.as_posix()):
                text = path.read_text(encoding=util.ENCODING)
                self.assertNotIn("ppar.demos", text)
                self.assertNotIn("analytics_demo_outputs", text)

    def test_repository_readme_is_audit_focused(self) -> None:
        """The top-level README presents one Audit product and workflow."""
        readme = Path("README.md").read_text(encoding=util.ENCODING)
        analytics_readme = Path("docs/analytics/README.md").read_text(
            encoding=util.ENCODING
        )

        self.assertIn("# PPAR Audit", readme)
        self.assertIn("Explain why reported portfolio performance changed.", readme)
        self.assertIn("## What PPAR Audit Answers", readme)
        self.assertIn("**Performance Comparison:**", readme)
        self.assertIn("**Data Issues:**", readme)
        self.assertIn("docs/images/readme/PerformanceAuditPortfolio.jpg", readme)
        self.assertIn("alt=\"PPAR Audit portfolio report\"", readme)
        self.assertNotIn("PerformanceComparisonPortfolio.jpg", readme)
        self.assertNotIn("PerformanceComparisonSecurity.jpg", readme)
        self.assertNotIn("DataIssues.jpg", readme)
        self.assertIn("## Setup", readme)
        self.assertNotIn("## Quick Setup", readme)
        self.assertIn("ppar setup ./my_ppar_audit", readme)
        self.assertIn("ppar audit ./my_ppar_audit", readme)
        self.assertNotIn("ppar setup --guide", readme)
        self.assertNotIn("ppar analytics ./", readme)
        self.assertIn("ppar.yaml", readme)
        self.assertIn("Customizing", readme)
        self.assertNotIn("## For Maintainers", readme)
        self.assertIn("## Additional Repository Capability", readme)
        self.assertIn("docs/analytics/README.md", readme)
        self.assertIn("# PPAR Analytics", analytics_readme)
        self.assertIn("**Performance Attribution:**", analytics_readme)
        self.assertIn("**Ex-Post Risk:**", analytics_readme)
        self.assertIn(
            "ppar setup ./my_ppar_analytics --analytics",
            analytics_readme,
        )
        self.assertIn("ppar analytics ./my_ppar_analytics", analytics_readme)
        self.assertIn(
            "ppar setup ./my_ppar_generic_analytics --generic-analytics",
            analytics_readme,
        )
        self.assertIn('pip install "ppar[analytics]"', analytics_readme)

    def test_generic_analytics_is_documented_as_an_optional_workspace(self) -> None:
        """Generic Analytics is user-facing without displacing Audit onboarding."""
        demo_data_readme = Path("ppar/setup_templates/README.md").read_text(
            encoding=util.ENCODING
        )
        refresh_guide = Path("docs/analytics/analytics_demo_refresh.md").read_text(
            encoding=util.ENCODING
        )
        maintainer_guide = Path("docs/maintainer_guide.md").read_text(
            encoding=util.ENCODING
        )
        normalized_demo_data_readme = " ".join(demo_data_readme.split())
        normalized_refresh_guide = " ".join(refresh_guide.split())
        normalized_maintainer_guide = " ".join(maintainer_guide.split())

        self.assertIn("The public onboarding path starts", demo_data_readme)
        self.assertIn(
            "ppar setup ./my_ppar_generic_analytics --generic-analytics",
            normalized_demo_data_readme,
        )
        self.assertIn(
            "optional, vendor-neutral Generic Analytics workspace",
            normalized_refresh_guide,
        )
        self.assertIn(
            "Installed setup users do not need this refresh path",
            normalized_refresh_guide,
        )
        self.assertIn(
            "ppar setup --generic-analytics",
            normalized_maintainer_guide,
        )
        self.assertIn(
            "Audit remains the default onboarding path",
            normalized_maintainer_guide,
        )

    def test_architecture_doc_maps_current_project_boundaries(self) -> None:
        """The compact architecture map stays tied to current public boundaries."""
        architecture = Path("docs/architecture.md").read_text(encoding=util.ENCODING)
        normalized_architecture = " ".join(architecture.split())
        maintainer_guide = Path("docs/maintainer_guide.md").read_text(
            encoding=util.ENCODING
        )

        self.assertIn("# Maintainer Guide", maintainer_guide)
        self.assertIn("[architecture map](architecture.md)", maintainer_guide)
        for expected_text in [
            "# PPAR Architecture",
            "The public installed command is:",
            "PPAR Audit is the current market-facing product",
            "ppar setup ./my_ppar_audit",
            "ppar audit ./my_ppar_audit",
            "ppar setup ./my_ppar_analytics --analytics",
            "ppar analytics ./my_ppar_analytics",
            "`ppar.analytics`",
            "`ppar.axys_apx`",
            "`ppar.audit`",
            "`ppar.setup_templates`",
            (
                "The Performance Comparison sub-feature does not try to rebuild "
                "a full accounting ledger."
            ),
            "Setup Data Versus Maintainer Data",
            "The YAML files are the main configuration and onboarding surface.",
            "`Performance Differences`",
            "`Performance Difference Causes`",
            "`source_detail.csv`",
            "`audit_support.zip`",
            "Keep new docs rare.",
        ]:
            with self.subTest(expected_text=expected_text):
                self.assertIn(expected_text, normalized_architecture)

    def test_documentation_is_partitioned_by_product(self) -> None:
        """Product documentation has durable Audit and Analytics homes."""
        documentation_index = Path("docs/README.md").read_text(
            encoding=util.ENCODING
        )
        expected_paths = (
            Path("docs/audit/performance_comparison_design.md"),
            Path("docs/audit/safety_invariants.md"),
            Path("docs/audit/demo_source_contract.md"),
            Path("docs/audit/README.md"),
            Path("docs/audit/product_constitution.md"),
            Path("docs/audit/roadmap.md"),
            Path("docs/audit/mvp_plan.md"),
            Path("docs/audit/product_specifications_index.md"),
            Path("docs/analytics/README.md"),
            Path("docs/analytics/analytics_demo_refresh.md"),
            Path("docs/analytics/roadmap.md"),
            Path("docs/maintainer_guide.md"),
            Path("docs/axys_apx/axys_apx_common_core_export.md"),
            Path("docs/archive/roadmap_through_v0.1.5.md"),
            Path(
                "docs/audit/archive/"
                "PPAR_Audit_Foundational_Product_Design_v0.10.md"
            ),
            Path("docs/audit/archive/PPAR_Audit_Work_App_Handoff_Prompt.md"),
        )
        retired_root_paths = (
            Path("docs/performance_comparison_design.md"),
            Path("docs/safety_invariants.md"),
            Path("docs/demo_source_contract.md"),
            Path("docs/analytics_demo_refresh.md"),
            Path("docs/roadmap.md"),
            Path("docs/audit/PPAR_Audit_Foundational_Product_Design_v0.10.md"),
            Path("docs/audit/PPAR_Audit_Work_App_Handoff_Prompt.md"),
            Path("docs/repository_guide.md"),
        )

        self.assertIn("PPAR Audit", documentation_index)
        self.assertIn("PPAR Analytics", documentation_index)
        for documented_path in (
            "audit/README.md",
            "audit/roadmap.md",
            "analytics/README.md",
            "analytics/roadmap.md",
            "archive/roadmap_through_v0.1.5.md",
            "maintainer_guide.md",
        ):
            self.assertIn(documented_path, documentation_index)
        self.assertTrue(all(path.is_file() for path in expected_paths))
        self.assertFalse(any(path.exists() for path in retired_root_paths))

    def test_local_documentation_links_resolve(self) -> None:
        """Checked-in documentation does not contain broken local file links."""
        markdown_paths = [Path("README.md"), *sorted(Path("docs").rglob("*.md"))]
        markdown_link = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
        html_link = re.compile(
            r'<(?:img|a)\b[^>]*(?:src|href)=["\']([^"\']+)["\']'
        )

        for markdown_path in markdown_paths:
            contents = markdown_path.read_text(encoding=util.ENCODING)
            targets = markdown_link.findall(contents) + html_link.findall(contents)
            for target in targets:
                normalized = target.strip().split()[0].strip("<>")
                if normalized.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                local_path = normalized.split("#", 1)[0]
                if not local_path:
                    continue
                with self.subTest(
                    markdown_path=markdown_path.as_posix(),
                    target=target,
                ):
                    self.assertTrue((markdown_path.parent / local_path).exists())

    def test_product_readme_image_references_exist(self) -> None:
        """Audit and Analytics product pages embed checked-in image artifacts."""
        readme_paths = (
            Path("README.md"),
            Path("docs/analytics/README.md"),
        )

        for readme_path in readme_paths:
            readme = readme_path.read_text(encoding=util.ENCODING)
            image_paths = re.findall(r'src="([^"]*images/readme/[^"]+)"', readme)
            self.assertGreater(len(image_paths), 0)
            for image_path in image_paths:
                with self.subTest(
                    readme=readme_path.as_posix(),
                    image_path=image_path,
                ):
                    self.assertTrue((readme_path.parent / image_path).exists())

    def test_site_extract_contract_template_is_documented(self) -> None:
        """The site extract-contract starter template remains linked from docs."""
        template_path = Path("docs/axys_apx/contracts/templates/site_extract_contract.yaml")
        source_contract = Path(
            "docs/audit/demo_source_contract.md"
        )
        demo_guide = Path("docs/audit/packaged_demo.md")

        self.assertTrue(template_path.exists())
        self.assertIn(
            template_path.as_posix(),
            demo_guide.read_text(encoding=util.ENCODING),
        )
        self.assertIn(
            "axys_apx/contracts/templates/site_extract_contract.yaml",
            source_contract.read_text(encoding=util.ENCODING),
        )

    def test_transaction_semantics_matrix_is_documented(self) -> None:
        """The Axys transaction semantics matrix remains linked from workflow docs."""
        matrix_path = Path(
            "docs/axys_apx/contracts/transaction_semantics_matrix.md"
        )
        matrix_yaml_path = Path(
            "docs/axys_apx/contracts/transaction_semantics_matrix.yaml"
        )
        evidence_pack_review_path = Path(
            "docs/audit/archive/performance_comparison_evidence_pack_review.md"
        )
        source_contract = Path(
            "docs/audit/demo_source_contract.md"
        )
        audit_index = Path("docs/audit/README.md")

        self.assertTrue(matrix_path.exists())
        self.assertTrue(matrix_yaml_path.exists())
        self.assertTrue(evidence_pack_review_path.exists())
        self.assertIn(
            "axys_apx/contracts/transaction_semantics_matrix.md",
            source_contract.read_text(encoding=util.ENCODING),
        )
        self.assertIn(
            "archive/performance_comparison_evidence_pack_review.md",
            source_contract.read_text(encoding=util.ENCODING),
        )
        self.assertIn(
            "axys_apx/contracts/transaction_semantics_matrix.yaml",
            source_contract.read_text(encoding=util.ENCODING),
        )
        self.assertIn(
            "axys_apx/contracts/transaction_semantics_matrix.yaml",
            audit_index.read_text(encoding=util.ENCODING),
        )

    def test_fixed_income_transaction_boundary_is_current(self) -> None:
        """Current contracts keep fixed-income treatment context-gated."""
        source_contract = Path(
            "docs/audit/demo_source_contract.md"
        ).read_text(encoding=util.ENCODING)
        matrix_yaml = _load_yaml(
            Path("docs/axys_apx/contracts/transaction_semantics_matrix.yaml")
        )
        rows = _yaml_mapping_rows(matrix_yaml["rows"], label="rows")

        self.assertIn("four proved fixed-income Modified Dietz input", source_contract)
        self.assertIn("amortization/accretion engine", source_contract)
        self.assertIn("bond principal schedule", source_contract)
        for code in ("pa", "sa", "pd"):
            with self.subTest(code=code):
                fixtures = _yaml_string_list(
                    rows[code]["fixtures"],
                    label=f"rows.{code}.fixtures",
                )
                self.assertIn("packaged_demo", fixtures)
                self.assertIn(
                    "code-only treatment remains unknown",
                    str(rows[code]["coverage_notes"]).lower(),
                )

    def test_return_capital_and_short_boundaries_are_current(self) -> None:
        """High-risk capital-return and short codes remain context-gated."""
        matrix_yaml = _load_yaml(
            Path("docs/axys_apx/contracts/transaction_semantics_matrix.yaml")
        )
        rows = _yaml_mapping_rows(matrix_yaml["rows"], label="rows")
        expected_fixtures = {
            "rc": ["packaged_demo", "site_variants/rc_return_of_capital"],
            "pd": ["packaged_demo", "site_variants/pd_principal_paydown"],
            "ss": ["packaged_demo", "site_variants/short_side_trades"],
            "cs": ["packaged_demo", "site_variants/short_side_trades"],
        }

        for code, fixtures in expected_fixtures.items():
            with self.subTest(code=code):
                self.assertEqual(rows[code]["coverage_status"], "partial")
                self.assertEqual(rows[code]["fixtures"], fixtures)
                self.assertIn(
                    "Code-only treatment remains unknown",
                    str(rows[code]["coverage_notes"]),
                )

    def test_candidate_override_profiles_are_documented_as_test_only(self) -> None:
        """Candidate override profiles remain explicit onboarding examples."""
        checklist = Path("docs/audit/site_extract_readiness_checklist.md").read_text(
            encoding=util.ENCODING
        )
        fixture_readme = Path("tests/data/axys/site_variants/README.md").read_text(
            encoding=util.ENCODING
        )
        matrix_yaml = _load_yaml(
            Path("docs/axys_apx/contracts/transaction_semantics_matrix.yaml")
        )
        expected_profiles = {
            "ai": "site_variants/ai_margin_interest",
            "pa": "site_variants/fixed_income_accruals",
            "sa": "site_variants/fixed_income_accruals",
            "rc": "site_variants/rc_return_of_capital",
            "pd": "site_variants/pd_principal_paydown",
            "ss": "site_variants/short_side_trades",
            "cs": "site_variants/short_side_trades",
        }

        self.assertIn("These profiles are not universal Axys/APX rules", checklist)
        self.assertIn("Code-only rows still stay\n`unknown`", checklist)
        self.assertIn("not native Axys schemas", fixture_readme)
        self.assertIn("not promote the transaction code", fixture_readme)
        self.assertIn("best-efforts demo-construction context", fixture_readme)

        for code, fixture in expected_profiles.items():
            profile_name = fixture.removeprefix("site_variants/")
            with self.subTest(code=code):
                rows = _yaml_mapping_rows(matrix_yaml["rows"], label="rows")
                row = rows[code]
                self.assertEqual(row["coverage_status"], "partial")
                self.assertIn(
                    fixture,
                    _yaml_string_list(row["fixtures"], label=f"rows.{code}.fixtures"),
                )
                self.assertIn(profile_name, checklist)
                self.assertIn(profile_name, fixture_readme)

        epus = _yaml_mapping_rows(matrix_yaml["rows"], label="rows")["epus"]
        self.assertEqual(epus["coverage_status"], "covered_context_only")
        self.assertEqual(
            _yaml_string_list(epus["fixtures"], label="rows.epus.fixtures"),
            ["site_variants/alternate_fee_context"],
        )
        self.assertIn("alternate_fee_context", checklist)
        self.assertIn("alternate_fee_context", fixture_readme)

    def test_transaction_matrix_has_one_current_human_view(self) -> None:
        """The human matrix is generated from the machine-readable authority."""
        matrix_yaml = _load_yaml(
            Path("docs/axys_apx/contracts/transaction_semantics_matrix.yaml")
        )
        matrix_contract = Path(
            "docs/axys_apx/contracts/transaction_semantics_matrix.md"
        ).read_text(encoding=util.ENCODING)
        rows = _yaml_mapping_rows(matrix_yaml["rows"], label="rows")

        required_codes = _yaml_string_list(
            matrix_yaml["required_matrix_codes"],
            label="required_matrix_codes",
        )
        self.assertEqual(set(rows), set(required_codes))
        self.assertIn("BEGIN GENERATED TRANSACTION ROWS", matrix_contract)
        self.assertIn("Core Observed Code Matrix", matrix_contract)
        self.assertIn("External-Flow Decision Rules", matrix_contract)
        self.assertIn("Coverage Backlog", matrix_contract)

    def test_product_roadmaps_separate_current_direction_from_history(self) -> None:
        """Each product owns current direction separately from shared history."""
        audit_roadmap = Path("docs/audit/roadmap.md").read_text(
            encoding=util.ENCODING
        )
        analytics_roadmap = Path("docs/analytics/roadmap.md").read_text(
            encoding=util.ENCODING
        )
        archived_roadmap = Path(
            "docs/archive/roadmap_through_v0.1.5.md"
        ).read_text(encoding=util.ENCODING)

        self.assertFalse(Path("docs/roadmap.md").exists())
        self.assertIn("# PPAR Audit Roadmap", audit_roadmap)
        self.assertIn("# PPAR Analytics Roadmap", analytics_roadmap)
        for expected_path in (
            "product_constitution.md",
            "mvp_plan.md",
        ):
            self.assertIn(expected_path, audit_roadmap)
        for heading in (
            "## Roadmap Doctrine",
            "## Active Phase — MVP Completion",
            "## Immediate Priorities",
            "## Material Open Questions",
            "## Maintenance Rule",
        ):
            self.assertIn(heading, audit_roadmap)
        self.assertNotIn("## Proposed Implementation Slices", audit_roadmap)
        self.assertNotIn("## Workstream", audit_roadmap)
        self.assertIn("**Archived implementation journal.**", archived_roadmap)

    def test_audit_mvp_authorities_agree_on_scope_and_sequence(self) -> None:
        """Audit authorities retain one material MVP boundary and current gate."""
        constitution = " ".join(
            Path("docs/audit/product_constitution.md")
            .read_text(encoding=util.ENCODING)
            .split()
        )
        plan = " ".join(
            Path("docs/audit/mvp_plan.md")
            .read_text(encoding=util.ENCODING)
            .split()
        )
        audit_index = " ".join(
            Path("docs/audit/README.md").read_text(encoding=util.ENCODING).split()
        )
        audit_roadmap = " ".join(
            Path("docs/audit/roadmap.md")
            .read_text(encoding=util.ENCODING)
            .split()
        )

        self.assertIn("four bounded MVP capabilities", constitution)
        self.assertIn("four additional product capabilities", plan)
        self.assertIn("Axys/APX transaction semantics and demo coverage", constitution)
        self.assertIn("Workstream D — Axys/APX Transaction Semantics", plan)
        self.assertIn(
            "Founder MVP assessment after the completed Slice 6 technical "
            "release audit",
            audit_roadmap,
        )
        self.assertIn("Slice 6 technical release audit complete", plan)
        self.assertIn("founder MVP assessment next", plan)
        self.assertIn(
            "Slice 5 — Exact case and optional missing-cost work — Complete",
            plan,
        )
        self.assertIn("optional, non-MVP-blocking capability", plan)
        for required_outcome in (
            "YAML-driven transaction semantics",
            "coherent Axys/APX scenarios",
            "exact-case behavior",
            "unknown-code fail-closed behavior",
            "protection against unsupported transaction meanings",
        ):
            self.assertIn(required_outcome, constitution)
            self.assertIn(required_outcome, audit_roadmap)
        self.assertIn("APX Custodial Integrator Mark-to-Market", plan)
        self.assertIn("Deferred; no MVP implementation", plan)
        self.assertNotIn("Slice 5C", plan)
        self.assertNotIn("Slice 5D", plan)
        self.assertIn(
            "Slice 2 — Executive Summary shared model — Complete",
            plan,
        )
        self.assertIn(
            "Slice 3 — Additional Data Issues issue types — Complete",
            plan,
        )
        self.assertIn("same commit whenever the number or identity", audit_index)
        self.assertIn("four bounded MVP capabilities", audit_roadmap)

    def test_trade_blotter_boundary_is_a_fail_closed_regression(self) -> None:
        """The matrix defers source-stage support and records its safety test."""
        matrix_yaml = _load_yaml(
            Path("docs/axys_apx/contracts/transaction_semantics_matrix.yaml")
        )
        pair_patterns = _yaml_mapping_rows(
            matrix_yaml["pair_patterns"],
            label="pair_patterns",
        )
        cancellation = pair_patterns["uppercase_trade_blotter_cancellation"]

        self.assertEqual(
            cancellation["ppar_treatment"],
            "unknown_pending_review_without_explicit_source_stage",
        )
        self.assertEqual(
            cancellation["coverage_status"],
            "covered_regression_test",
        )
        notes = str(cancellation["coverage_notes"])
        self.assertIn("cannot inherit lowercase semantics", notes)
        self.assertIn("product support remains deferred", notes)

    def test_axys_apx_reference_documents_blockers(self) -> None:
        """The Axys/APX reference keeps a single blocker summary discoverable."""
        overview = Path("docs/axys_apx/reference/Chapter_01_Overview.md").read_text(
            encoding=util.ENCODING
        )

        for expected_text in [
            '<a id="axys_apx-blockers"></a>',
            "## 7. Current Cross-Cutting Blockers",
            "AXAPX-B01 | No verified performance extract dictionary",
            "AXAPX-B02 | Stored-versus-recalculated performance is Unknown",
            "AXAPX-B03 | Security-performance footing is Unknown",
            "AXAPX-B04 | Transaction-code coverage is incomplete and context-dependent",
            "AXAPX-B05 | IMEX/APXIX object and field lists are not authoritative",
            "AXAPX-B06 | REP, SSRS, and report definitions are unavailable",
            "AXAPX-B07 | APX SQL/public-view/API contracts are under-evidenced",
            (
                "AXAPX-B08 | Multi-currency, fixed-income, and corporate-action "
                "mechanics remain incomplete"
            ),
            "configurable source-data snapshots",
            "native Axys/APX extraction or methodology",
        ]:
            with self.subTest(expected_text=expected_text):
                self.assertIn(expected_text, overview)

    def test_site_extract_readiness_checklist_is_documented(self) -> None:
        """The site extract readiness checklist remains linked from setup docs."""
        checklist = Path("docs/audit/site_extract_readiness_checklist.md")
        source_contract = Path(
            "docs/audit/demo_source_contract.md"
        )
        template = Path("docs/axys_apx/contracts/templates/site_extract_contract.yaml")

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
        source_contract = Path(
            "docs/audit/demo_source_contract.md"
        )
        local_opt_out = Path(
            "tests/data/axys/site_variants/local_opt_out/ppar_audit.yaml"
        )

        self.assertTrue(local_opt_out.exists())
        self.assertIn("local_opt_out", readme)
        self.assertIn("enforce_ambiguous_axys_flows", readme)
        self.assertIn(
            "local_opt_out",
            source_contract.read_text(encoding=util.ENCODING),
        )

    def test_transaction_semantics_matrix_yaml_matches_contract_codes(self) -> None:
        """The machine-readable transaction matrix stays aligned with the contract."""
        matrix_yaml = _load_yaml(
            Path("docs/axys_apx/contracts/transaction_semantics_matrix.yaml")
        )
        contract_document = Path(
            "docs/axys_apx/contracts/transaction_semantics_matrix.md"
        ).read_text(encoding=util.ENCODING)

        rows = _yaml_mapping_rows(matrix_yaml["rows"], label="rows")
        required_codes = _yaml_string_list(
            matrix_yaml["required_matrix_codes"],
            label="required_matrix_codes",
        )
        self.assertEqual(set(rows), set(required_codes))

        for code, metadata in rows.items():
            self.assertIn(f"`{code}`", contract_document)
            self.assertIn("observed_meaning", metadata)
            self.assertIn("ppar_categories", metadata)
            self.assertIn("coverage_status", metadata)
            self.assertIn("fixtures", metadata)
            self.assertIn("coverage_notes", metadata)

        self.assertEqual(
            set(
                _yaml_string_list(
                    matrix_yaml["ambiguous_external_flow_codes"],
                    label="ambiguous_external_flow_codes",
                )
            ),
            {"li", "lo", "ti", "dp", "wd"},
        )

    def test_transaction_semantics_matrix_rows_have_coverage_rationale(self) -> None:
        """Each transaction matrix row explains its coverage or backlog status."""
        matrix_yaml = _load_yaml(
            Path("docs/axys_apx/contracts/transaction_semantics_matrix.yaml")
        )
        coverage_statuses = set(
            _yaml_mapping(
                matrix_yaml["coverage_statuses"],
                label="coverage_statuses",
            )
        )

        for section_name in ("rows", "pair_patterns"):
            section = _yaml_mapping_rows(matrix_yaml[section_name], label=section_name)
            for code, metadata in section.items():
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

    def test_maintainer_transaction_evidence_matches_matrix_codes(self) -> None:
        """Maintainer evidence groups cover documented fixture families."""
        matrix_yaml = _load_yaml(
            Path("docs/axys_apx/contracts/transaction_semantics_matrix.yaml")
        )
        matrix_codes = set(
            _yaml_string_list(
                matrix_yaml["required_matrix_codes"],
                label="required_matrix_codes",
            )
        )
        non_runtime_codes = set(
            _yaml_string_list(
                matrix_yaml["non_runtime_matrix_codes"],
                label="non_runtime_matrix_codes",
            )
        )

        registered_codes = registered_transaction_evidence_codes()
        self.assertLessEqual(matrix_codes - non_runtime_codes, registered_codes)
        self.assertTrue(non_runtime_codes.isdisjoint(registered_codes))
        self.assertEqual(
            transaction_evidence_groups("in"),
            ("packaged_formula", "fixed_income_safe"),
        )
        self.assertIn(
            "ambiguous_context_required",
            transaction_evidence_groups("wd"),
        )
        self.assertIn(
            "fixed_income_accrued_interest",
            transaction_evidence_groups("pa"),
        )
        self.assertIn(
            "fixed_income_accrued_interest",
            transaction_evidence_groups("sa"),
        )
        self.assertEqual(transaction_evidence_groups(";"), ())
        self.assertIn("context_only", transaction_evidence_groups("exus"))
        self.assertIn("standalone_backlog", transaction_evidence_groups("epus"))
        self.assertIn("fixed_income_backlog", transaction_evidence_groups("pd"))
        self.assertIn("capital_return_backlog", transaction_evidence_groups("pd"))
        self.assertIn("short_side_backlog", transaction_evidence_groups("ss"))
        self.assertEqual(transaction_evidence_groups("not-a-real-code"), ())

        registered_groups = set(TRANSACTION_EVIDENCE_GROUPS)
        self.assertEqual(
            registered_groups,
            {
                "packaged_formula",
                "contextual_packaged",
                "fixed_income_safe",
                "fixed_income_accrued_interest",
                "ambiguous_context_required",
                "context_only",
                "fixed_income_backlog",
                "capital_return_backlog",
                "short_side_backlog",
                "standalone_backlog",
            },
        )
        self.assertEqual(
            AMBIGUOUS_FLOW_TRANSACTION_CODES,
            set(
                _yaml_string_list(
                    matrix_yaml["ambiguous_external_flow_codes"],
                    label="ambiguous_external_flow_codes",
                )
            ),
        )

    def test_demo_transaction_rules_are_known_to_semantics_matrix(self) -> None:
        """Packaged demo transaction rules cannot introduce undocumented codes."""
        matrix_yaml = _load_yaml(
            Path("docs/axys_apx/contracts/transaction_semantics_matrix.yaml")
        )
        demo_yaml = _load_yaml(
            Path(
                "ppar/setup_templates/axys_apx_audit/"
                "axys_apx_audit.yaml"
            )
        )

        matrix_codes = set(_yaml_mapping_rows(matrix_yaml["rows"], label="rows"))
        demo_codes = set(
            _yaml_mapping(
                demo_yaml["transaction_rules"],
                label="transaction_rules",
            )
        )

        self.assertLessEqual(demo_codes, matrix_codes)

    def test_transaction_semantics_coverage_claims_have_fixtures(self) -> None:
        """Coverage statuses in the machine-readable matrix point to real fixtures."""
        matrix_yaml = _load_yaml(
            Path("docs/axys_apx/contracts/transaction_semantics_matrix.yaml")
        )
        packaged_demo_codes = set(
            _yaml_mapping(
                _load_yaml(
                    Path(
                        "ppar/setup_templates/axys_apx_audit/"
                        "axys_apx_audit.yaml"
                    )
                )["transaction_rules"],
                label="transaction_rules",
            )
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

        rows = _yaml_mapping_rows(matrix_yaml["rows"], label="rows")
        for code, metadata in rows.items():
            fixtures = _yaml_string_list(metadata["fixtures"], label=f"rows.{code}.fixtures")
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

        ambiguous_codes = set(
            _yaml_string_list(
                matrix_yaml["ambiguous_external_flow_codes"],
                label="ambiguous_external_flow_codes",
            )
        )
        self.assertLessEqual(
            ambiguous_codes - {"ti"},
            imex_context_codes & rep_semantics_codes & code_only_codes,
        )
        self.assertIn("ti", packaged_demo_codes)

    def test_packaged_demo_transaction_coverage_matches_current_contract(self) -> None:
        """Packaged transaction data agrees with current matrix coverage."""
        matrix_yaml = _load_yaml(
            Path("docs/axys_apx/contracts/transaction_semantics_matrix.yaml")
        )
        rows = _yaml_mapping_rows(matrix_yaml["rows"], label="rows")
        packaged_demo_rule_codes = set(
            _yaml_mapping(
                _load_yaml(
                    Path(
                        "ppar/setup_templates/axys_apx_audit/"
                        "axys_apx_audit.yaml"
                    )
                )["transaction_rules"],
                label="transaction_rules",
            )
        )
        packaged_demo_data_codes = (
            _transaction_codes_in_csv(
                Path(
                    "ppar/setup_templates/axys_apx_audit/"
                    "snapshot_a/transactions.csv"
                )
            )
            | _transaction_codes_in_csv(
                Path(
                    "ppar/setup_templates/axys_apx_audit/"
                    "snapshot_b/transactions.csv"
                )
            )
        )

        self.assertLessEqual(packaged_demo_data_codes, packaged_demo_rule_codes)
        for code in ("ai", "li", "lo", "ti", "wd", "dp", "pa", "sa"):
            with self.subTest(code=code):
                self.assertIn(code, packaged_demo_data_codes)
                fixtures = _yaml_string_list(
                    rows[code]["fixtures"],
                    label=f"rows.{code}.fixtures",
                )
                self.assertIn("packaged_demo", fixtures)
        self.assertEqual(rows["li"]["coverage_status"], "covered_packaged_demo")
        self.assertEqual(rows["lo"]["coverage_status"], "covered_packaged_demo")
        self.assertEqual(rows["pa"]["coverage_status"], "partial")
        self.assertEqual(rows["sa"]["coverage_status"], "partial")

    def test_review_only_action_fixture_is_documented(self) -> None:
        """Synthetic review-only action rows remain a test-only quarantine."""
        readme = Path("tests/data/axys/site_variants/README.md").read_text(
            encoding=util.ENCODING
        )
        matrix_yaml = _load_yaml(
            Path("docs/axys_apx/contracts/transaction_semantics_matrix.yaml")
        )
        fixture = Path(
            "tests/data/axys/site_variants/review_only_actions/"
            "ppar_audit.yaml"
        )

        self.assertTrue(fixture.exists())
        self.assertIn("review_only_actions", readme)
        self.assertIn("neutral review evidence", readme)
        self.assertEqual(
            _yaml_string_list(
                _yaml_mapping_rows(matrix_yaml["rows"], label="rows")[";"]["fixtures"],
                label="rows.;.fixtures",
            ),
            ["site_variants/review_only_actions"],
        )
        self.assertIn(
            "synthetic corporate-action markers",
            str(
                _yaml_mapping_rows(matrix_yaml["rows"], label="rows")[";"][
                    "coverage_notes"
                ]
            ),
        )

    def test_transaction_semantics_matrix_matches_ambiguous_fixture_outputs(self) -> None:
        """Documented ambiguous-code semantics include IMEX-context fixture behavior."""
        matrix_yaml = _load_yaml(
            Path("docs/axys_apx/contracts/transaction_semantics_matrix.yaml")
        )
        specification = AuditSpecification(
            Path(
                "tests/data/axys/site_variants/imex_context/"
                "ppar_audit.yaml"
            )
        )
        frame = TransactionsLoader(specification).load("a")
        assert frame is not None

        rows_by_code = _yaml_mapping_rows(matrix_yaml["rows"], label="rows")
        for code in _yaml_string_list(
            matrix_yaml["ambiguous_external_flow_codes"],
            label="ambiguous_external_flow_codes",
        ):
            rows = frame.filter(pl.col(schema.TRANSACTION_CODE) == code)
            documented = rows_by_code[code]

            self.assertLessEqual(
                set(rows.get_column(schema.TRANSACTION_CATEGORY).to_list()),
                set(
                    _yaml_string_list(
                        documented["ppar_categories"],
                        label=f"rows.{code}.ppar_categories",
                    )
                ),
                code,
            )
            self.assertLessEqual(
                set(rows.get_column(schema.CASH_FLOW_SIGN).to_list()),
                set(
                    _yaml_string_list(
                        documented["cash_flow_signs"],
                        label=f"rows.{code}.cash_flow_signs",
                    )
                ),
                code,
            )
            self.assertLessEqual(
                set(rows.get_column(schema.PERFORMANCE_FLOW_SIGN).to_list()),
                set(
                    _yaml_string_list(
                        documented["performance_flow_signs"],
                        label=f"rows.{code}.performance_flow_signs",
                    )
                ),
                code,
            )

    def test_axys_package_is_included(self) -> None:
        """The Axys subpackage is included in distribution metadata."""
        with open("pyproject.toml", "rb") as file:
            pyproject = tomllib.load(file)

        self.assertIn("ppar.axys_apx", pyproject["tool"]["setuptools"]["packages"])
        self.assertIn("ppar.analytics", pyproject["tool"]["setuptools"]["packages"])
        self.assertIn("ppar.setup_templates", pyproject["tool"]["setuptools"]["packages"])
        self.assertIn(
            "ppar.setup_templates.axys_apx_analytics",
            pyproject["tool"]["setuptools"]["packages"],
        )
        self.assertIn(
            "ppar.setup_templates.axys_apx_audit",
            pyproject["tool"]["setuptools"]["packages"],
        )
        self.assertIn(
            "ppar.setup_templates.generic_analytics",
            pyproject["tool"]["setuptools"]["packages"],
        )
        self.assertNotIn("ppar.demos", pyproject["tool"]["setuptools"]["packages"])
        self.assertIn(
            "ppar.audit",
            pyproject["tool"]["setuptools"]["packages"],
        )
        self.assertIn(
            "ppar.audit.cli",
            pyproject["tool"]["setuptools"]["packages"],
        )

    def test_console_scripts_are_explicit(self) -> None:
        """Installed commands point to packaged modules."""
        with open("pyproject.toml", "rb") as file:
            pyproject = tomllib.load(file)

        self.assertEqual(
            pyproject["project"]["scripts"],
            {
                "ppar": "ppar.cli:main",
            },
        )

    def test_maintainer_guide_documents_demo_onboarding_commands(self) -> None:
        """The maintainer guide keeps the Axys/APX demo smoke path discoverable."""
        guide = Path("docs/maintainer_guide.md").read_text(encoding=util.ENCODING)

        for expected_text in [
            "Run Setup-Generated Smoke Scripts",
            "Run Release-Candidate Checks",
            "scripts/check_release_candidate.py",
            "scripts/check_release_candidate.py --build",
            "--include-generic-data-generation",
            "--write-packaged-assets",
            "--verbose",
            "500x Analytics/Audit scale regression",
            "`--build` regenerates `PPAR.pdf`",
            "`--refresh-images` regenerates the README PNG/JPG assets",
            "final Twine-validated wheel and source distribution",
            "single command used to create a distributable release candidate",
            "Yahoo-dependent generic analytics",
            "ppar.cli setup /tmp/my_ppar_audit",
            "/tmp/my_ppar_audit/run_audit.py",
            "/tmp/my_ppar_analytics/run_analytics.py",
            "/tmp/my_ppar_generic_analytics/run_generic_analytics.py",
            "--generic-analytics",
            "ppar.audit.cli.validate_bundle",
            "ppar.audit.cli.validate_config",
            "scripts/validate_demo_matrix.py",
            "scripts/check_audit_demo_health.py",
            "/tmp/my_ppar_audit/output/portfolio",
            "/tmp/my_ppar_audit/output/security",
            "portfolio_audit.xlsx",
            "report_bundle_contract()",
            "prefer the package-root workflow helpers",
            "direct-submodule imports",
            "package-root exports",
            "`Performance Differences`",
            "`Performance Difference Causes`",
            "`source_detail.csv`",
            "`audit_support.zip`",
            "`Reconstruction Summary`",
            "`Return Reconstruction Checks`",
            "`Security Return Checks`",
            "copied by `ppar setup`",
        ]:
            with self.subTest(expected_text=expected_text):
                self.assertIn(expected_text, guide)
        self.assertNotIn("Other Data Differences", guide)
        self.assertNotIn("Residual Evidence", guide)
        self.assertNotIn("Return Reconstruction Summary", guide)

    def test_maintainer_guide_documents_packaged_demo_inventory(self) -> None:
        """The maintainer guide keeps demo source ownership and scenarios visible."""
        guide = Path("docs/maintainer_guide.md").read_text(encoding=util.ENCODING)

        for expected_text in [
            "Packaged Axys/APX Audit Demo Maintenance",
            "Treat the packaged Axys/APX Audit demo as a small accounting",
            "scripts/demo_support/market_data.py",
            "derive_operational_demo_data.py",
            "scripts/operational_demo_data/holdings.py",
            "audit_transaction_scenarios.csv",
            "audit_holding_scenarios.csv",
            "audit_scenario_calendar.csv",
            "two-independent-change review target",
            "rebuild_audit_demo_data.py",
            "Use `--write` only when you intend to rewrite tracked packaged CSV assets.",
            "Current packaged scenario inventory",
            "`CVNA` split row",
            "`TSLA` `ss`/`cs` rows",
            "`36225MBS1` `pd` row",
            "`91282Y2Y1` `in` row",
            "`91282Y5Y1` `by`/`pa` and `sl`/`sa` rows",
            "`91282Y5Y1` cost-only row",
            "`JPM` `dv` and `rc` rows",
        ]:
            with self.subTest(expected_text=expected_text):
                self.assertIn(expected_text, guide)

    def test_maintainer_guide_documents_release_readiness_checklist(self) -> None:
        """The maintainer guide keeps release-tag readiness checks explicit."""
        guide = Path("docs/maintainer_guide.md").read_text(encoding=util.ENCODING)

        for expected_text in [
            "## Release Readiness",
            "./.venv/bin/python scripts/check_release_candidate.py --build",
            "`pyproject.toml` is the only package-version authority",
            "`ppar.__version__` value is read from installed package metadata",
            "PPAR_RELEASE_VERSION=$(./.venv/bin/python -c",
            'git rev-parse --short "v${PPAR_RELEASE_VERSION}"',
            'git ls-remote --tags origin "v${PPAR_RELEASE_VERSION}"',
            'git tag -f "v${PPAR_RELEASE_VERSION}" HEAD',
            "exactly one",
            "Twine-validated wheel",
            "source distribution",
            "only the `ppar` console script is exposed",
            "ppar setup /tmp/my_ppar_audit",
            "ppar setup /tmp/my_ppar_analytics --analytics",
            "Do not move, create, or push a release tag until the version and release commit",
        ]:
            with self.subTest(expected_text=expected_text):
                self.assertIn(expected_text, guide)

    def test_axys_demo_readme_uses_current_report_sheet_names(self) -> None:
        """The packaged demo guide names the current workbook review path."""
        readme = Path("docs/audit/packaged_demo.md").read_text(
            encoding=util.ENCODING
        )

        for expected_text in [
            "ppar.audit.cli.validate_config",
            "minimum required datasets",
            "required normalized columns",
            "complete YAML treatment",
            "`Performance Differences` sheet",
            "`Performance Difference Causes` sheet",
            "`source_detail.csv`",
            "`audit_support.zip`",
            "`Reconstruction Summary`",
            "`Return Reconstruction Checks`",
            "`Security Return Checks`",
        ]:
            with self.subTest(expected_text=expected_text):
                self.assertIn(expected_text, readme)
        self.assertNotIn("Other Data Differences", readme)
        self.assertNotIn("Residual Evidence", readme)
        self.assertNotIn("Return Reconstruction Summary", readme)

    def test_design_notes_document_both_user_facing_comparison_demos(self) -> None:
        """Design docs keep portfolio and security demo paths in sync."""
        design = Path("docs/audit/performance_comparison_design.md").read_text(
            encoding=util.ENCODING
        )

        for expected_text in [
            "normalized internal datasets",
            "`portfolio_performance`: portfolio identifier",
            "`security_performance`: portfolio/security identifiers",
            "authoritative valuation and flow evidence",
            "including `contribution_impact_methods`, is rejected",
            "Modified Dietz",
            "transaction rules",
        ]:
            with self.subTest(expected_text=expected_text):
                self.assertIn(expected_text, design)

    def test_axys_demo_resources_are_packaged(self) -> None:
        """The Axys/APX demos use packaged resources instead of test fixtures."""
        demo_data = files("ppar.setup_templates")
        axys_apx_analytics_data = demo_data / "axys_apx_analytics"
        axys_demo_data = demo_data / "axys_apx_audit"
        generic_analytics_data = demo_data / "generic_analytics"
        expected_resources = (
            "README.md",
            "axys_apx_audit.yaml",
            "snapshot_a/portperf.csv",
            "snapshot_a/secmast.csv",
            "snapshot_b/secmast.csv",
            "snapshot_b/transactions.csv",
        )

        for resource_path in expected_resources:
            with self.subTest(resource_path=resource_path):
                self.assertTrue((axys_demo_data / resource_path).is_file())
        self.assertFalse(
            (axys_demo_data / "transaction_semantics_policy.yaml").is_file()
        )
        self.assertFalse((axys_demo_data / "column_mappings.yaml").is_file())
        for resource_path in (
            "axys_apx_analytics.yaml",
            "holidays.csv",
            "portperf.csv",
            "secperf.csv",
            "secmast.csv",
        ):
            with self.subTest(resource_path=f"axys_apx_analytics/{resource_path}"):
                self.assertTrue((axys_apx_analytics_data / resource_path).is_file())
        for resource_path in ("README.md", "holidays.csv", "run_generic_analytics.py"):
            with self.subTest(resource_path=f"generic_analytics/{resource_path}"):
                self.assertTrue((generic_analytics_data / resource_path).is_file())

    def test_setup_template_scripts_do_not_depend_on_demo_helpers(self) -> None:
        """Installed setup scripts are self-contained tutorial surfaces."""
        demo_modules = (
            Path("ppar/setup_templates/axys_apx_analytics/run_analytics.py"),
            Path("ppar/setup_templates/generic_analytics/run_generic_analytics.py"),
        )

        for path in demo_modules:
            text = path.read_text(encoding=util.ENCODING)
            with self.subTest(path=path.as_posix()):
                self.assertNotIn("ppar.demos", text)
                self.assertNotIn("analytics_demo_outputs", text)
                self.assertNotIn("tests/data/axys", text)

    def test_axys_portfolio_demo_uses_operational_mega_cap_data(self) -> None:
        """The user-facing comparison demo packages the promoted operational data."""
        axys_demo_data = files("ppar.setup_templates") / "axys_apx_audit"
        holdings_path = Path(
            str(axys_demo_data / "snapshot_a" / "holdings.csv")
        )
        portperf_path = Path(str(axys_demo_data / "snapshot_a" / "portperf.csv"))

        with portperf_path.open(encoding=util.ENCODING, newline="") as file:
            portfolio_codes = {row["Portfolio Code"] for row in csv.DictReader(file)}
        with holdings_path.open(encoding=util.ENCODING, newline="") as file:
            holding_ids = {row["Security Symbol"] for row in csv.DictReader(file)}

        self.assertEqual(
            portfolio_codes,
            {
                "ALPHA",
                "BALANCED",
                "BALANCED_CONTRIBUTION",
                "INCOME",
            },
        )
        self.assertIn("AAPL", holding_ids)
        self.assertIn("NVDA", holding_ids)
        self.assertIn("CASHUSD", holding_ids)
        self.assertIn("912797AA1", holding_ids)
        self.assertIn("91282Y2Y1", holding_ids)
        self.assertIn("91282Y5Y1", holding_ids)

    def test_axys_demo_equities_do_not_use_long_constant_price_runs(self) -> None:
        """Only the founder-approved stale-price demo run may remain constant."""
        axys_demo_data = files("ppar.setup_templates") / "axys_apx_audit"
        expected_failures = [
            (
                "ALPHA",
                "GOOGL",
                date(2025, 12, 31),
                date(2026, 1, 30),
            )
        ]

        for snapshot_name in ("snapshot_a", "snapshot_b"):
            snapshot = Path(str(axys_demo_data / snapshot_name))
            with (snapshot / "secmast.csv").open(
                encoding=util.ENCODING,
                newline="",
            ) as file:
                equity_ids = {
                    row["Security Symbol"]
                    for row in csv.DictReader(file)
                    if row["Asset Class Code"] == "EQ"
                }
            with (snapshot / "holdings.csv").open(
                encoding=util.ENCODING,
                newline="",
            ) as file:
                holding_rows = list(csv.DictReader(file))

            prices_by_holding: dict[tuple[str, str], list[tuple[date, float]]] = {}
            for row in holding_rows:
                if row["Security Symbol"] not in equity_ids:
                    continue
                key = (row["Portfolio Code"], row["Security Symbol"])
                prices_by_holding.setdefault(key, []).append(
                    (date.fromisoformat(row["Holding Date"]), float(row["Price"]))
                )
            failures: list[tuple[str, str, date, date]] = []
            for (portfolio, security), observations in prices_by_holding.items():
                ordered_observations = sorted(observations)
                for previous, current in zip(
                    ordered_observations,
                    ordered_observations[1:],
                    strict=False,
                ):
                    if (current[0] - previous[0]).days < 28:
                        continue
                    if current[1] == previous[1]:
                        failures.append(
                            (portfolio, security, previous[0], current[0])
                        )

            with self.subTest(snapshot=snapshot_name):
                self.assertEqual(failures, expected_failures)

    def test_axys_demo_changed_transactions_have_matching_holdings(self) -> None:
        """Material transaction demo changes have matching month-end holdings evidence."""
        axys_demo_data = files("ppar.setup_templates") / "axys_apx_audit"
        snapshot_a = Path(str(axys_demo_data / "snapshot_a"))
        snapshot_b = Path(str(axys_demo_data / "snapshot_b"))

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
                cash_a = holdings_a[(portfolio, "CASHUSD", holding_date)]
                cash_b = holdings_b[(portfolio, "CASHUSD", holding_date)]

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
        axys_demo_data = files("ppar.setup_templates") / "axys_apx_audit"
        for snapshot_name in ("snapshot_a", "snapshot_b"):
            with self.subTest(snapshot=snapshot_name):
                snapshot = Path(str(axys_demo_data / snapshot_name))
                with (snapshot / "transactions.csv").open(
                    encoding=util.ENCODING,
                    newline="",
                ) as file:
                    for row in csv.DictReader(file):
                        if row["Transaction Code"] != "by":
                            continue
                        expected_amount = -round(
                            float(row["Quantity"]) * float(row["Price"])
                            + float(row["Commission"]),
                            2,
                        )
                        self.assertAlmostEqual(
                            float(row["Amount"]),
                            expected_amount,
                            places=2,
                        )

    def test_axys_demo_fee_transactions_reduce_cash(self) -> None:
        """Packaged fee changes reduce ending cash by their combined amount."""
        axys_demo_data = files("ppar.setup_templates") / "axys_apx_audit"
        snapshot_a = Path(str(axys_demo_data / "snapshot_a"))
        snapshot_b = Path(str(axys_demo_data / "snapshot_b"))
        transaction_key = ("INCOME", "2026-01-20", "CASHUSD", "dp")
        margin_interest_key = ("INCOME", "2026-01-22", "MARGIN", "ai")
        cash_holding_key = ("INCOME", "CASHUSD", "2026-01-30")
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
        amount_delta = _float_delta(
            transactions_a[transaction_key],
            transactions_b[transaction_key],
            "AMOUNT",
        )
        margin_interest_delta = _float_delta(
            transactions_a[margin_interest_key],
            transactions_b[margin_interest_key],
            "AMOUNT",
        )
        combined_expense_delta = amount_delta + margin_interest_delta
        self.assertLess(amount_delta, 0)
        self.assertLess(margin_interest_delta, 0)
        self.assertAlmostEqual(
            _float_delta(holdings_a[cash_holding_key], holdings_b[cash_holding_key], "MKT_VAL"),
            combined_expense_delta,
            places=2,
        )

    def test_axys_demo_omits_synthetic_future_splits(self) -> None:
        """Packaged comparison demo avoids fictional future split transactions."""
        axys_demo_data = files("ppar.setup_templates") / "axys_apx_audit"
        snapshot_a = Path(str(axys_demo_data / "snapshot_a"))
        snapshot_b = Path(str(axys_demo_data / "snapshot_b"))

        transactions_a = _demo_transactions_by_natural_key(snapshot_a)
        transactions_b = _demo_transactions_by_natural_key(snapshot_b)

        for rows in (transactions_a, transactions_b):
            split_rows = [
                row
                for row in rows.values()
                if row["TRAN"] == ";" and row["SEC"] == "TSLA"
            ]
            self.assertEqual(split_rows, [])

    def test_axys_demo_withdrawal_changes_cash_and_reconstructed_return(self) -> None:
        """Packaged withdrawal changes cash and the reconstructed return."""
        axys_demo_data = files("ppar.setup_templates") / "axys_apx_audit"
        snapshot_a = Path(str(axys_demo_data / "snapshot_a"))
        snapshot_b = Path(str(axys_demo_data / "snapshot_b"))
        transaction_key = ("ALPHA", "2026-01-20", "CASHUSD", "wd")
        cash_holding_key = ("ALPHA", "CASHUSD", "2026-01-30")
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
        comparison_path = Path(str(axys_demo_data / "axys_apx_audit.yaml"))
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

    def test_axys_demo_security_performance_has_minimal_source_contract(self) -> None:
        """Security performance demo rows contain only reported-return fields."""
        axys_demo_data = files("ppar.setup_templates") / "axys_apx_audit"
        for snapshot_name in ("snapshot_a", "snapshot_b"):
            with self.subTest(snapshot=snapshot_name):
                snapshot = Path(str(axys_demo_data / snapshot_name))
                holdings = _csv_rows_by_key(
                    snapshot / "holdings.csv",
                    ("PORT", "SEC", "HOLDING_DATE"),
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
                with (snapshot / "secperf.csv").open(
                    encoding=util.ENCODING,
                    newline="",
                ) as file:
                    self.assertEqual(
                        csv.DictReader(file).fieldnames,
                        [
                            "Portfolio Code",
                            "Security Symbol",
                            "Security Type",
                            "From Date",
                            "Thru Date",
                            "Security Return",
                        ],
                    )

    def test_axys_demo_portfolio_performance_matches_reconstruction(
        self,
    ) -> None:
        """Portfolio demo performance rows match configured reconstruction rules."""
        axys_demo_data = files("ppar.setup_templates") / "axys_apx_audit"
        comparison_path = Path(str(axys_demo_data / "axys_apx_audit.yaml"))
        checks = _reconstruction_rows_by_key(
            portfolio_return_reconstruction_checks(comparison_path),
            ("portfolio_id", "from_date", "thru_date"),
        )
        for snapshot_name in ("snapshot_a", "snapshot_b"):
            with self.subTest(snapshot=snapshot_name):
                snapshot = Path(str(axys_demo_data / snapshot_name))
                portperf = _csv_rows_by_key(
                    snapshot / "portperf.csv",
                    ("PORTFOLIO_CODE", "FROM_DATE", "THRU_DATE"),
                )
                with (snapshot / "portperf.csv").open(
                    encoding=util.ENCODING,
                    newline="",
                ) as file:
                    self.assertEqual(
                        csv.DictReader(file).fieldnames,
                        [
                            "Portfolio Code",
                            "From Date",
                            "Thru Date",
                            "Portfolio Return",
                            "Base Currency",
                        ],
                    )
                for raw_key in portperf:
                    key = cast(tuple[str, str, str], raw_key)
                    with self.subTest(snapshot=snapshot_name, key=key):
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
        axys_demo_data = files("ppar.setup_templates") / "axys_apx_audit"
        snapshot_a = Path(str(axys_demo_data / "snapshot_a"))
        snapshot_b = Path(str(axys_demo_data / "snapshot_b"))
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
        comparison_path = Path(str(axys_demo_data / "axys_apx_audit.yaml"))
        checks = _reconstruction_rows_by_key(
            security_return_reconstruction_checks(comparison_path),
            ("portfolio_id", "security_id", "from_date", "thru_date"),
        )

        aapl_periods = {
            "ALPHA": ("2026-05-01", "2026-05-29", "2026-05-29"),
            "BALANCED": ("2026-02-28", "2026-03-31", "2026-03-31"),
            "INCOME": ("2026-05-01", "2026-05-08", "2026-05-08"),
        }
        for portfolio, (from_date, thru_date, holding_date) in aapl_periods.items():
            with self.subTest(portfolio=portfolio, security="AAPL"):
                key = (portfolio, "AAPL", from_date, thru_date)
                check_key = (portfolio, "csusAAPL", from_date, thru_date)
                holding_key = (portfolio, "AAPL", holding_date)
                price_delta = _float_delta(
                    holdings_a[holding_key],
                    holdings_b[holding_key],
                    "PRICE",
                )
                self.assertGreater(price_delta, 0.0)
                self.assertAlmostEqual(
                    _float_delta(secperf_a[key], secperf_b[key], "SEC_RETURN"),
                    _reconstruction_float(
                        checks[check_key],
                        DERIVED_RETURN_DIFFERENCE,
                    ),
                    places=9,
                )
                self.assertEqual(
                    checks[check_key][RECONSTRUCTION_STATUS],
                    RECONSTRUCTION_STATUS_ALIGNED,
                )

        tnote_key = ("INCOME", "91282Y2Y1", "2026-05-15", "2026-05-15")
        tnote_check_key = ("INCOME", "fius91282Y2Y1", "2026-05-15", "2026-05-15")
        tnote_holding_key = ("INCOME", "91282Y2Y1", "2026-05-15")
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
            _reconstruction_float(
                checks[tnote_check_key],
                DERIVED_RETURN_DIFFERENCE,
            ),
            places=9,
        )
        self.assertEqual(
            checks[tnote_check_key][RECONSTRUCTION_STATUS],
            RECONSTRUCTION_STATUS_ALIGNED,
        )

    def test_axys_validation_matrix_documents_problem_scenarios(self) -> None:
        """The Axys validation matrix names the expected review scenarios."""
        matrix = Path("tests/data/axys/README.md").read_text(encoding=util.ENCODING)
        expected_scenarios = {
            "Clean/no issue": "baseline",
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
        """The packaged Axys/APX demo guide describes the field-role model."""
        matrix = Path("docs/audit/packaged_demo.md").read_text(
            encoding=util.ENCODING
        )
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

    def test_axys_demo_source_contract_has_stable_demo_boundaries(self) -> None:
        """The source contract owns demo behavior without historical gates."""
        contract_doc = Path(
            "docs/audit/demo_source_contract.md"
        ).read_text(
            encoding=util.ENCODING
        )

        for expected_text in [
            "## Governing References",
            "## Extraction Requirement Labels",
            "## Minimum Source-Data Contract",
            "## Field Role Contract",
            "## Scenario Preservation Contract",
            "## Cash-Balance Policy",
            "## Date Policy",
            "## Transaction-Code Policy",
            "transaction_semantics_matrix.yaml",
            "## Fixed-Income Transaction Boundary",
            "## Site Extract Contract Setup",
            "## What This Contract Excludes",
        ]:
            with self.subTest(expected_text=expected_text):
                self.assertIn(expected_text, contract_doc)
        self.assertNotIn("## Axys/APX Demo Completion Gate", contract_doc)
        self.assertNotIn("## Axys/APX Demo Freeze Decision Packet", contract_doc)

    def test_minimum_source_data_contract_is_documented(self) -> None:
        """The source-data contract helper and user-facing docs stay aligned."""
        root_readme = Path("README.md").read_text(encoding=util.ENCODING)
        contract_doc = Path(
            "docs/audit/demo_source_contract.md"
        ).read_text(
            encoding=util.ENCODING
        )
        contracts = performance_source_data_contract.source_data_contract()

        self.assertIn(
            "Production inputs may come from reviewed REP, IMEX, custom-report",
            root_readme,
        )
        self.assertIn(
            "PPAR normalizes those files through a customizable `ppar.yaml`",
            root_readme,
        )
        self.assertIn("## Minimum Source-Data Contract", contract_doc)
        self.assertIn("stops before producing a report", contract_doc)
        self.assertIn("performance calculation is configured", contract_doc)
        self.assertIn("required normalized", contract_doc)
        self.assertIn("required normalized column cannot be resolved", contract_doc)
        self.assertIn("required source column is ambiguous", contract_doc)
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
            "holdings: portfolio_id, security_id, holding_date, market_value",
            summary["required_columns"],
        )
        self.assertIn(
            "security_performance: portfolio_id, security_id, from_date, "
            "thru_date, security_return",
            summary["required_columns"],
        )
        self.assertIn(
            "transactions: portfolio_id, security_id, transaction_date, "
            "transaction_code, amount",
            summary["required_columns"],
        )

    def test_performance_comparison_method_constants_use_enum_values(self) -> None:
        """Report and finding constants stay aligned with YAML method enums."""
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

    def test_public_axys_apx_import_contract(self) -> None:
        """The documented Axys/APX package exports remain importable."""
        expected_exports = {
            "AxysClassificationSources",
            "AxysData",
            "AxysPortfolio",
            "AxysSpecification",
        }

        self.assertEqual(set(axys_apx.__all__), expected_exports)
        self.assertIs(
            AxysClassificationSources,
            axys_apx.AxysClassificationSources,
        )
        self.assertIs(AxysData, axys_apx.AxysData)
        self.assertIs(AxysPortfolio, axys_apx.AxysPortfolio)
        self.assertIs(AxysSpecification, axys_apx.AxysSpecification)

    def test_public_audit_import_contract(self) -> None:
        """The Audit root exposes only shared workflow and report APIs."""
        expected_exports = {
            "ComparisonFile": ComparisonFile,
            "ComparisonSnapshot": ComparisonSnapshot,
            "PortfolioPerformanceLoader": PortfolioPerformanceLoader,
            "AuditSpecification": AuditSpecification,
            "HoldingsLoader": HoldingsLoader,
            "REPORT_BUNDLE_REQUIRED_ARTIFACTS": REPORT_BUNDLE_REQUIRED_ARTIFACTS,
            "SecurityPerformanceLoader": SecurityPerformanceLoader,
            "TransactionsLoader": TransactionsLoader,
            "schema": schema,
            "compact_findings_table": compact_findings_table,
            "compare_snapshots": compare_snapshots,
            "report_bundle_contract": report_bundle_contract,
            "report_bundle_validation_issues": report_bundle_validation_issues,
            "summarize_findings": summarize_findings,
            "validate_causal_attribution_ready": validate_causal_attribution_ready,
            "validate_yaml_setup_complete": validate_yaml_setup_complete,
            "write_audit_report_bundle": write_audit_report_bundle,
            "write_audit_review_workbook": write_audit_review_workbook,
        }

        self.assertEqual(set(audit.__all__), set(expected_exports))
        for name, imported_object in expected_exports.items():
            with self.subTest(name=name):
                self.assertIs(imported_object, getattr(audit, name))

    def test_subfeature_import_contracts_are_owned_by_subpackages(self) -> None:
        """Specialized vocabulary is exported by its owning sub-feature."""
        expected_performance_comparison_exports = {
            "CONTEXT",
            "DIRECT_INPUT",
            "EVIDENCE_ROLE",
            "RELATED_OUTPUT",
            "TARGET_OUTPUT",
            "CauseArea",
            "Finding",
            "PerformanceComparison",
            "SuppressionRule",
            "apply_suppressions",
            "findings_to_polars",
            "portfolio_period_cause_summary",
            "portfolio_period_contribution_candidates",
            "portfolio_period_evidence_breakdown",
            "portfolio_period_impact_coverage_summary",
            "portfolio_period_summary",
            "portfolio_period_transaction_cross_checks",
            "rank_portfolio_period_evidence",
            "security_period_evidence_breakdown",
            "security_period_summary",
            "transaction_activity_summary",
            "transaction_matching_diagnostics",
        }
        expected_data_issues_exports = {
            "DATA_ISSUES_CONFIG_KEY",
            "DATA_ISSUE_REGISTRY",
            "DataIssueCategory",
            "DataIssueDefinition",
            "DataIssueType",
            "data_issues_config_summary",
            "validate_data_issues_config",
        }

        self.assertEqual(
            set(performance_comparison.__all__),
            expected_performance_comparison_exports,
        )
        self.assertEqual(set(data_issues.__all__), expected_data_issues_exports)
        self.assertIs(
            PerformanceComparison,
            performance_comparison.PerformanceComparison,
        )
        self.assertIs(DataIssueType, data_issues.DataIssueType)

    def test_runtime_transaction_helpers_exclude_maintainer_registries(self) -> None:
        """Installed Audit helpers omit test-only transaction registries."""
        top_level_exports = set(audit.__all__)
        helper_modules = {
            "source_data_contract": performance_source_data_contract,
            "transaction_summary": performance_transaction_summary,
        }

        self.assertTrue(
            hasattr(performance_transaction_summary, "transaction_semantics_summary")
        )
        self.assertTrue(top_level_exports.isdisjoint(helper_modules))
        for removed_module in (
            "ppar.audit.fixed_income",
            "ppar.audit.transaction_policy",
            "ppar.audit.performance_comparison.backlog_gates",
            "ppar.audit.performance_comparison.transaction_boundary_registry",
        ):
            with self.subTest(removed_module=removed_module):
                self.assertIsNone(importlib.util.find_spec(removed_module))

    def test_public_audit_runner_import_contract(self) -> None:
        """The runner module exposes only the compact workflow helper surface."""
        expected_exports = {
            "compact_findings_table",
            "compare_snapshots",
            "summarize_findings",
            "validate_causal_attribution_ready",
            "validate_yaml_setup_complete",
        }

        self.assertEqual(set(audit_runner.__all__), expected_exports)
        self.assertIs(
            compact_findings_table,
            audit_runner.compact_findings_table,
        )
        self.assertIs(compare_snapshots, audit_runner.compare_snapshots)
        self.assertIs(summarize_findings, audit_runner.summarize_findings)
        self.assertIs(
            validate_causal_attribution_ready,
            audit_runner.validate_causal_attribution_ready,
        )
        self.assertIs(
            validate_yaml_setup_complete,
            audit_runner.validate_yaml_setup_complete,
        )

    def test_public_audit_report_import_contract(self) -> None:
        """The report module exposes only the complete bundle writer."""
        expected_exports = {"write_audit_report_bundle"}

        self.assertEqual(set(audit_report.__all__), expected_exports)
        self.assertIs(
            write_audit_report_bundle,
            audit_report.write_audit_report_bundle,
        )
        self.assertNotIn(
            "report_bundle_validation_issues",
            audit_report.__all__,
        )

    def test_performance_comparison_vocabulary_exports_are_explicit(self) -> None:
        """Vocabulary modules export every declared public schema/code name."""
        module_paths = {
            schema: Path("ppar/audit/schema.py"),
            performance_comparison_findings: Path(
                "ppar/audit/performance_comparison/findings.py"
            ),
            audit_transactions: Path(
                "ppar/audit/transactions.py"
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
            common: Path("ppar/common.py"),
            core_errors: Path("ppar/errors.py"),
            util: Path("ppar/utilities.py"),
        }

        for module, path in module_paths.items():
            with self.subTest(module=module.__name__):
                expected_exports = _declared_public_module_names(path)
                if module is util:
                    expected_exports.update(common.__all__)
                self.assertEqual(
                    set(module.__all__),
                    expected_exports,
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

    def test_audit_import_does_not_load_optional_analytics_dependencies(self) -> None:
        """Audit imports remain independent of Analytics dependencies."""
        command = (
            "import sys; import ppar.audit; "
            "optional = {'lxml', 'matplotlib', 'numpy', 'pandas', 'pyarrow', "
            "'seaborn', 'scipy'}; "
            "raise SystemExit(1 if optional.intersection(sys.modules) else 0)"
        )

        subprocess.run([sys.executable, "-c", command], check=True)

    def test_package_root_does_not_eagerly_load_analytics(self) -> None:
        """Importing the package root keeps Analytics modules out of startup."""
        command = (
            "import sys; import ppar; "
            "raise SystemExit(1 if 'ppar.analytics' in sys.modules else 0)"
        )

        subprocess.run([sys.executable, "-c", command], check=True)

    def test_package_root_lazy_audit_exports_work(self) -> None:
        """Package-root convenience exports match the Audit product surface."""
        command = (
            "from ppar import AuditSpecification, compare_snapshots, "
            "write_audit_report_bundle; "
            "raise SystemExit(0 if all("
            "obj is not None for obj in "
            "(AuditSpecification, compare_snapshots, write_audit_report_bundle)"
            ") else 1)"
        )

        subprocess.run([sys.executable, "-c", command], check=True)

    def test_production_invariants_do_not_use_optimization_sensitive_asserts(self) -> None:
        """Production checks remain active when Python optimization is enabled."""
        violations: list[str] = []
        for path in sorted(Path("ppar").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding=util.ENCODING))
            violations.extend(
                f"{path}:{node.lineno}"
                for node in ast.walk(tree)
                if isinstance(node, ast.Assert)
            )

        self.assertEqual([], violations)


if __name__ == "__main__":
    unittest.main()
