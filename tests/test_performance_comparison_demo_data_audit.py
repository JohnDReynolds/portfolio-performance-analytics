"""Tests for packaged performance-comparison demo data accounting guardrails."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
from typing import Literal, cast
import unittest

import pandas as pd
import polars as pl
import yaml

from ppar.performance_comparison import (
    PerformanceComparisonSpecification,
    TransactionsLoader,
    compare_snapshots,
)
from ppar.performance_comparison import schema as pc_cols
from ppar.performance_comparison.config_validation import validate_config
from ppar.performance_comparison.fixed_income import (
    FIXED_INCOME_BACKLOG_TRANSACTION_CODES,
    fixed_income_transaction_boundary,
)
from ppar.performance_comparison.transactions import (
    TRANSACTION_CATEGORY_EXTERNAL_FLOW,
    TRANSACTION_CATEGORY_FEE_EXPENSE,
    TRANSACTION_CASH_FLOW_SIGN_NEGATIVE,
    TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL,
    TRANSACTION_PERFORMANCE_FLOW_SIGN_PERFORMANCE,
    TRANSACTION_SEMANTICS_SOURCE_YAML_RULE,
)
from ppar.performance_comparison.workbook_tables import (
    _workbook_raw_audit_trail_table,
    _workbook_portfolio_changes_table,
    _workbook_security_changes_table,
    _workbook_underlying_causes_table,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_AUDIT_SCRIPT_PATH = _REPO_ROOT / "scripts" / "audit_performance_comparison_demo_data.py"
_REBUILD_SCRIPT_PATH = (
    _REPO_ROOT
    / "scripts"
    / "operational_demo_data"
    / "rebuild_performance_comparison_demo_data.py"
)
_RENDER_EXTRACT_AVAILABILITY_SCRIPT_PATH = (
    _REPO_ROOT / "scripts" / "render_demo_extract_availability.py"
)
_PACKAGED_COMPARISON_PATH = (
    _REPO_ROOT
    / "ppar"
    / "setup_templates"
    / "axysapx_performance_comparison"
    / "axysapx_performance_comparison.yaml"
)
_DEMO_SOURCE_CONTRACT_PATH = (
    _REPO_ROOT / "docs" / "performance_comparison_demo_source_contract.md"
)
_PACKAGED_AXYS_DIRECTORY = (
    _REPO_ROOT / "ppar" / "setup_templates" / "axysapx_performance_comparison"
)
_PACKAGED_AXYS_README_PATH = _PACKAGED_AXYS_DIRECTORY / "README.md"
_DEMO_EXTRACT_AVAILABILITY_PATH = (
    _PACKAGED_AXYS_DIRECTORY / "demo_extract_availability.yaml"
)
_PACKAGED_DEMO_TRANSACTION_CODES = {
    "by",
    "sl",
    "dv",
    "in",
    "dp",
    "li",
    "lo",
    "pa",
    "pd",
    "rc",
    "sa",
    "ss",
    "cs",
    "wd",
}
_TEST_ONLY_TRANSACTION_CODES: set[str] = set()
_REAL_WORLD_EVIDENCE_REQUIRED_TRANSACTION_CODES = {";"}
_PACKAGED_TRANSACTION_COLUMNS = [
    "PORT",
    "TRANSACTION_DATE",
    "SETTLE_DATE",
    "SEC",
    "TRAN",
    "SEC_TYPE",
    "SRC_DEST_TYPE",
    "SRC_DEST_SYMBOL",
    "SPECIAL_SEC_TYPE",
    "SPECIAL_SEC_SYMBOL",
    "QTY",
    "PRICE",
    "AMOUNT",
    "COMMISSION",
]

_PERFORMANCE_DIFFERENCE_CAUSE_FIELDS = {
    (pc_cols.HOLDINGS, pc_cols.ACCRUED),
    (pc_cols.HOLDINGS, pc_cols.MARKET_VALUE),
    (pc_cols.HOLDINGS, pc_cols.PRICE),
    (pc_cols.HOLDINGS, pc_cols.QUANTITY),
    (pc_cols.SPLITS, pc_cols.SPLIT_FACTOR),
    (pc_cols.TRANSACTIONS, pc_cols.AMOUNT),
    (pc_cols.TRANSACTIONS, pc_cols.COMMISSION),
    (pc_cols.TRANSACTIONS, pc_cols.PRICE),
    (pc_cols.TRANSACTIONS, pc_cols.QUANTITY),
    ("no_underlying_causes_found", None),
}


def _load_audit_module():
    """Load the demo-data audit script as a test module."""
    spec = importlib.util.spec_from_file_location(
        "audit_performance_comparison_demo_data",
        _AUDIT_SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("Could not load performance-comparison demo-data audit.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_rebuild_module():
    """Load the demo-data rebuild script as a test module."""
    spec = importlib.util.spec_from_file_location(
        "rebuild_performance_comparison_demo_data",
        _REBUILD_SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("Could not load performance-comparison demo-data rebuild.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_extract_availability_renderer():
    """Load the extract-availability renderer as a test module."""
    spec = importlib.util.spec_from_file_location(
        "render_demo_extract_availability",
        _RENDER_EXTRACT_AVAILABILITY_SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("Could not load extract-availability renderer.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _transaction_row(
    frame: pl.DataFrame,
    *,
    portfolio: str,
    transaction_date: str,
    security: str,
    transaction_code: str,
) -> dict[str, object]:
    """Return one transaction row by visible packaged-demo attributes."""
    return frame.filter(
        (pl.col(pc_cols.PORTFOLIO_ID) == portfolio)
        & (pl.col(pc_cols.TRANSACTION_DATE).cast(pl.String) == transaction_date)
        & (pl.col(pc_cols.SECURITY_ID) == security)
        & (pl.col(pc_cols.TRANSACTION_CODE) == transaction_code)
    ).row(0, named=True)


def _dataset_fields(frame: pl.DataFrame) -> set[tuple[str, str | None]]:
    """Return normalized dataset/source-column pairs from a workbook table."""
    return {
        (str(row["dataset"]), row["source_column"])
        for row in frame.select(["dataset", "source_column"]).iter_rows(named=True)
    }


def _resolved_transaction_semantics(row: dict[str, object]) -> dict[str, object]:
    """Return the transaction semantics fields that decide flow treatment."""
    return {
        pc_cols.TRANSACTION_CODE: row[pc_cols.TRANSACTION_CODE],
        pc_cols.TRANSACTION_CATEGORY: row[pc_cols.TRANSACTION_CATEGORY],
        pc_cols.CASH_FLOW_SIGN: row[pc_cols.CASH_FLOW_SIGN],
        pc_cols.PERFORMANCE_FLOW_SIGN: row[pc_cols.PERFORMANCE_FLOW_SIGN],
        pc_cols.TRANSACTION_SEMANTICS_SOURCE: row[
            pc_cols.TRANSACTION_SEMANTICS_SOURCE
        ],
    }


class TestPerformanceComparisonDemoDataAudit(unittest.TestCase):
    """Verify packaged demo data remains internally consistent."""

    def test_packaged_demo_source_contract_documents_boundaries(self) -> None:
        """The packaged demo contract preserves key source-data boundaries."""
        text = _DEMO_SOURCE_CONTRACT_PATH.read_text(encoding="utf-8")
        normalized_text = " ".join(text.split())
        normalized_lower_text = normalized_text.lower()

        for expected_text in [
            "normalized demo extracts",
            "not official Axys/APX native schemas",
            "portperf.csv",
            "secperf.csv",
            "CASH_USD",
            "transaction_rules",
        ]:
            self.assertIn(expected_text, normalized_text)

        for expected_text in [
            "mandatory product inputs",
            "realistic packaged-demo fields",
            "optional local-enrichment fields",
            "internal scenario/rebuild fields",
        ]:
            self.assertIn(expected_text, normalized_lower_text)

        for expected_text in [
            "reported performance-extract context",
            "defensible as report-style gain/loss components",
            "must not be described as recomputed tax-lot or accounting-ledger values",
            "review evidence unless a future explicit settlement-date rule",
            "conservative no-id path",
            "does not prove a durable native transaction identifier",
            "not a full accounting-system export",
        ]:
            self.assertIn(expected_text, normalized_lower_text)

        self.assertIn(
            "Review evidence unless a future explicit settlement-date rule",
            normalized_text,
        )

    def test_packaged_demo_transaction_fields_stay_user_facing(self) -> None:
        """Packaged transaction extracts omit internal IDs but keep rule context."""
        scenario_path = (
            _REPO_ROOT
            / "scripts"
            / "operational_demo_data"
            / "performance_comparison_transaction_scenarios.csv"
        )

        for snapshot_directory in ("snapshot_a", "snapshot_b"):
            with self.subTest(snapshot_directory=snapshot_directory):
                transactions = pd.read_csv(
                    _PACKAGED_AXYS_DIRECTORY
                    / snapshot_directory
                    / "transactions.csv",
                    nrows=0,
                )

                self.assertEqual(
                    list(transactions.columns),
                    _PACKAGED_TRANSACTION_COLUMNS,
                )
                self.assertNotIn("TRANSACTION_ID", transactions.columns)
                self.assertIn("SRC_DEST_TYPE", transactions.columns)
                self.assertIn("SPECIAL_SEC_TYPE", transactions.columns)

        scenario_columns = pd.read_csv(scenario_path, nrows=0).columns
        self.assertIn("TRANSACTION_ID", scenario_columns)

    def test_packaged_demo_readme_documents_transaction_coverage_map(self) -> None:
        """The packaged demo README names packaged, test-only, and backlog rows."""
        text = _PACKAGED_AXYS_README_PATH.read_text(encoding="utf-8")

        for expected_text in [
            "Current transaction coverage by home",
            "Packaged demo rows",
            "`by`, `sl`, short-side `ss`/`cs`, `dv`, `in`",
            "fixed-income accrued-interest `pa`/`sa`",
            "equity/security return-of-capital `rc`",
            "MBS principal-paydown `pd`",
            "external-cash `lo`, and external-cash `wd`",
            "YAML rules reserved for runtime guards",
            "Test-only fixtures",
            "`dv` + `by` reinvestment guards",
            "Evidence-blocked backlog",
            "`ai`, uppercase reversal rows",
            "ordinary TNOTE2Y\n  interest uses an `in` transaction row",
            "TNOTE5Y `pa`/`sa` rows are packaged",
        ]:
            self.assertIn(expected_text, text)

    def test_packaged_demo_readme_matches_current_restatement_story(self) -> None:
        """The main packaged README keeps scenario descriptions current."""
        text = _PACKAGED_AXYS_README_PATH.read_text(encoding="utf-8")

        for expected_text in [
            "The controlled restatement includes",
            "`wd` external-withdrawal amount restatement",
            "AAPL `by` transaction amount",
            "MSFT `sl` transaction",
            "inserted `li` row on `CASH_USD`",
            "inserted `lo` row on `CASH_USD`",
            "JPM `dv` dividend amount",
            "JPM `rc` return-of-capital",
            "fee-like `dp` transaction",
            "classified from special-security context",
            "missed/late AAPL `dv` row",
            "real 2026-05-14 payable-date dividend",
            "TNOTE2Y `in` interest",
            "MBSPOOL `pd` principal-paydown",
            "paired TNOTE5Y `by`/`pa` and",
        ]:
            self.assertIn(expected_text, text)

    def test_packaged_snapshot_folders_do_not_include_local_readmes(self) -> None:
        """Setup snapshots stay focused on source CSV files."""
        for snapshot_name in ("snapshot_a", "snapshot_b"):
            with self.subTest(snapshot_name=snapshot_name):
                self.assertFalse(
                    (_PACKAGED_AXYS_DIRECTORY / snapshot_name / "README.md").exists()
                )

    def test_packaged_demo_extract_availability_covers_current_headers(self) -> None:
        """Every packaged Axys/APX demo CSV field has extraction-confidence metadata."""
        availability = yaml.safe_load(
            _DEMO_EXTRACT_AVAILABILITY_PATH.read_text(encoding="utf-8")
        )
        self.assertIsInstance(availability, dict)

        labels = set(availability["confidence_labels"])
        name_labels = set(availability["name_confidence_labels"])
        source_strategy_labels = set(availability["source_strategy_labels"])
        datasets = availability["datasets"]
        expected_files = {
            path.name
            for path in (_PACKAGED_AXYS_DIRECTORY / "snapshot_a").glob("*.csv")
        }
        self.assertEqual(set(datasets), expected_files)

        for snapshot_directory in ("snapshot_a", "snapshot_b"):
            snapshot_path = _PACKAGED_AXYS_DIRECTORY / snapshot_directory
            for file_name in sorted(expected_files):
                header = list(pd.read_csv(snapshot_path / file_name, nrows=0).columns)
                columns = datasets[file_name]["columns"]

                self.assertEqual(list(columns), header)
                for column_name, metadata in columns.items():
                    self.assertIn(metadata["imex_confidence"], labels, column_name)
                    self.assertIn(metadata["rep_confidence"], labels, column_name)
                    self.assertTrue(str(metadata["normalized_meaning"]).strip())
                    self.assertTrue(str(metadata["basis"]).strip())
                    open_questions = metadata["open_questions"]
                    self.assertIsInstance(open_questions, list, column_name)
                    self.assertTrue(open_questions, column_name)
                    for question in open_questions:
                        self.assertTrue(str(question).strip(), column_name)
                    candidate_axys_names = metadata["candidate_axys_names"]
                    candidate_report_labels = metadata["candidate_report_labels"]
                    self.assertIsInstance(candidate_axys_names, list, column_name)
                    self.assertIsInstance(candidate_report_labels, list, column_name)
                    self.assertTrue(candidate_axys_names, column_name)
                    self.assertTrue(candidate_report_labels, column_name)
                    for candidate_name in candidate_axys_names + candidate_report_labels:
                        self.assertTrue(str(candidate_name).strip(), column_name)
                    self.assertIn(metadata["name_confidence"], name_labels, column_name)
                    self.assertTrue(str(metadata["name_notes"]).strip())
                    self.assertIn(
                        metadata["preferred_source"],
                        source_strategy_labels,
                        column_name,
                    )
                    self.assertIn(
                        metadata["fallback_source"],
                        source_strategy_labels,
                        column_name,
                    )
                    self.assertIsInstance(
                        metadata["requires_context_for_semantics"],
                        bool,
                        column_name,
                    )
                    self.assertIsInstance(
                        metadata["blocking_if_missing"],
                        bool,
                        column_name,
                    )
                    self.assertTrue(str(metadata["source_strategy_notes"]).strip())
                    self.assertTrue(str(metadata["comments"]).strip())

        transaction_columns = datasets["transactions.csv"]["columns"]
        self.assertNotIn("TRANSACTION_ID", transaction_columns)
        for context_column in [
            "SEC_TYPE",
            "SRC_DEST_TYPE",
            "SRC_DEST_SYMBOL",
            "SPECIAL_SEC_TYPE",
            "SPECIAL_SEC_SYMBOL",
        ]:
            metadata = transaction_columns[context_column]
            self.assertTrue(metadata["requires_context_for_semantics"])
            self.assertTrue(metadata["blocking_if_missing"])

    def test_packaged_demo_gain_loss_metadata_stays_report_style_context(self) -> None:
        """GAIN_LOSS remains report-style performance context, not a native claim."""
        availability = yaml.safe_load(
            _DEMO_EXTRACT_AVAILABILITY_PATH.read_text(encoding="utf-8")
        )
        datasets = availability["datasets"]

        for file_name in ("portperf.csv", "secperf.csv"):
            with self.subTest(file_name=file_name):
                metadata = datasets[file_name]["columns"]["GAIN_LOSS"]

                self.assertEqual(metadata["name_confidence"], "report_label_inferred")
                self.assertEqual(metadata["preferred_source"], "rep_preferred")
                self.assertEqual(
                    metadata["fallback_source"],
                    "local_discovery_required",
                )
                self.assertIn(
                    "does not prove a native IMEX performance object/field",
                    metadata["basis"],
                )
                self.assertIn("report-dependent", metadata["comments"])
                self.assertIn("Report-style label", metadata["name_notes"])

    def test_packaged_demo_extract_availability_contract_is_current(self) -> None:
        """The human-readable contract is rendered from the YAML contract."""
        renderer = _load_extract_availability_renderer()

        self.assertEqual(renderer.main(["--check"]), 0)

    def test_packaged_demo_transaction_rules_cover_observed_codes(self) -> None:
        """Packaged demo YAML explicitly defines every observed transaction code."""
        summary = validate_config(_PACKAGED_COMPARISON_PATH)

        self.assertEqual(summary["transaction_codes_without_yaml_rules"], "none")

    def test_packaged_demo_transaction_rules_cover_ambiguous_axys_codes(self) -> None:
        """Packaged demo YAML covers observed and ambiguous Axys/APX source codes."""
        observed_codes: set[str] = set()
        for snapshot_directory in ("snapshot_a", "snapshot_b"):
            transactions = pd.read_csv(
                _PACKAGED_AXYS_DIRECTORY / snapshot_directory / "transactions.csv"
            )
            observed_codes.update(
                str(code).strip()
                for code in transactions["TRAN"].dropna()
                if str(code).strip()
            )

        configuration = yaml.safe_load(
            _PACKAGED_COMPARISON_PATH.read_text(encoding="utf-8")
        )
        configured_codes = set(configuration["transaction_rules"].keys())

        self.assertTrue(observed_codes.issubset(configured_codes))
        self.assertTrue({"dp", "li", "lo", "wd"}.issubset(configured_codes))

    def test_packaged_demo_resolves_ambiguous_axys_flow_examples(self) -> None:
        """Packaged ambiguous-code examples resolve only through reviewed context."""
        specification = PerformanceComparisonSpecification(_PACKAGED_COMPARISON_PATH)

        for snapshot_key in ("a", "b"):
            with self.subTest(snapshot_key=snapshot_key):
                frame = TransactionsLoader(specification).load(
                    cast(Literal["a", "b"], snapshot_key)
                )
                assert frame is not None
                resolved_rows = {
                    "ALPHA0203": _transaction_row(
                        frame,
                        portfolio="ALPHA",
                        transaction_date="2026-01-20",
                        security="CASH_USD",
                        transaction_code="wd",
                    ),
                    "INCOME0203": _transaction_row(
                        frame,
                        portfolio="INCOME",
                        transaction_date="2026-01-20",
                        security="CASH_USD",
                        transaction_code="dp",
                    ),
                }
                if snapshot_key == "b":
                    resolved_rows["ALPHA0303"] = _transaction_row(
                        frame,
                        portfolio="ALPHA",
                        transaction_date="2026-02-17",
                        security="CASH_USD",
                        transaction_code="lo",
                    )

                expected_rows = {"ALPHA0203", "INCOME0203"}
                if snapshot_key == "b":
                    expected_rows.add("ALPHA0303")

                self.assertEqual(set(resolved_rows), expected_rows)
                self.assertEqual(
                    _resolved_transaction_semantics(resolved_rows["ALPHA0203"]),
                    {
                        pc_cols.TRANSACTION_CODE: "wd",
                        pc_cols.TRANSACTION_CATEGORY: (
                            TRANSACTION_CATEGORY_EXTERNAL_FLOW
                        ),
                        pc_cols.CASH_FLOW_SIGN: TRANSACTION_CASH_FLOW_SIGN_NEGATIVE,
                        pc_cols.PERFORMANCE_FLOW_SIGN: (
                            TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL
                        ),
                        pc_cols.TRANSACTION_SEMANTICS_SOURCE: (
                            TRANSACTION_SEMANTICS_SOURCE_YAML_RULE
                        ),
                    },
                )
                self.assertEqual(
                    _resolved_transaction_semantics(resolved_rows["INCOME0203"]),
                    {
                        pc_cols.TRANSACTION_CODE: "dp",
                        pc_cols.TRANSACTION_CATEGORY: TRANSACTION_CATEGORY_FEE_EXPENSE,
                        pc_cols.CASH_FLOW_SIGN: TRANSACTION_CASH_FLOW_SIGN_NEGATIVE,
                        pc_cols.PERFORMANCE_FLOW_SIGN: (
                            TRANSACTION_PERFORMANCE_FLOW_SIGN_PERFORMANCE
                        ),
                        pc_cols.TRANSACTION_SEMANTICS_SOURCE: (
                            TRANSACTION_SEMANTICS_SOURCE_YAML_RULE
                        ),
                    },
                )
                if snapshot_key == "b":
                    self.assertEqual(
                        _resolved_transaction_semantics(resolved_rows["ALPHA0303"]),
                        {
                            pc_cols.TRANSACTION_CODE: "lo",
                            pc_cols.TRANSACTION_CATEGORY: (
                                TRANSACTION_CATEGORY_EXTERNAL_FLOW
                            ),
                            pc_cols.CASH_FLOW_SIGN: TRANSACTION_CASH_FLOW_SIGN_NEGATIVE,
                            pc_cols.PERFORMANCE_FLOW_SIGN: (
                                TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL
                            ),
                            pc_cols.TRANSACTION_SEMANTICS_SOURCE: (
                                TRANSACTION_SEMANTICS_SOURCE_YAML_RULE
                            ),
                        },
                    )

    def test_packaged_demo_transaction_codes_stay_reviewer_realistic(self) -> None:
        """Packaged transaction rows avoid synthetic semantic edge cases."""
        for snapshot_directory in ("snapshot_a", "snapshot_b"):
            transactions = pd.read_csv(
                _PACKAGED_AXYS_DIRECTORY / snapshot_directory / "transactions.csv"
            )
            observed_codes = set(transactions["TRAN"].astype(str))

            self.assertLessEqual(observed_codes, _PACKAGED_DEMO_TRANSACTION_CODES)
            self.assertTrue(observed_codes.isdisjoint(_TEST_ONLY_TRANSACTION_CODES))

    def test_packaged_demo_has_no_unevidenced_corporate_action_rows(self) -> None:
        """User-facing corporate actions require real-world evidence first."""
        for snapshot_directory in ("snapshot_a", "snapshot_b"):
            transactions = pd.read_csv(
                _PACKAGED_AXYS_DIRECTORY / snapshot_directory / "transactions.csv"
            )
            observed_codes = set(transactions["TRAN"].astype(str))

            self.assertTrue(
                observed_codes.isdisjoint(
                    _REAL_WORLD_EVIDENCE_REQUIRED_TRANSACTION_CODES
                )
            )

    def test_packaged_demo_fixed_income_boundary_stays_evidenced(self) -> None:
        """Fixed-income demo rows use proved income/accrual inputs only."""
        specification = PerformanceComparisonSpecification(_PACKAGED_COMPARISON_PATH)

        for snapshot_key, snapshot_directory in (
            ("a", "snapshot_a"),
            ("b", "snapshot_b"),
        ):
            with self.subTest(snapshot_key=snapshot_key):
                transactions = pd.read_csv(
                    _PACKAGED_AXYS_DIRECTORY / snapshot_directory / "transactions.csv"
                )
                observed_codes = set(transactions["TRAN"].astype(str))

                disallowed_backlog_codes = (
                    FIXED_INCOME_BACKLOG_TRANSACTION_CODES - {"pd"}
                )
                self.assertTrue(observed_codes.isdisjoint(disallowed_backlog_codes))
                if snapshot_key == "b":
                    self.assertIn("pa", observed_codes)
                    self.assertIn("pd", observed_codes)
                    self.assertIn("sa", observed_codes)

                fixed_income_interest = transactions.loc[
                    (transactions["PORT"] == "INCOME")
                    & (transactions["TRANSACTION_DATE"] == "2026-05-15")
                    & (transactions["SEC"] == "TNOTE2Y")
                    & (transactions["TRAN"] == "in")
                ].iloc[0]
                self.assertEqual(fixed_income_interest["TRAN"], "in")
                self.assertEqual(
                    fixed_income_transaction_boundary(fixed_income_interest["TRAN"]),
                    "safe_income",
                )
                self.assertEqual(fixed_income_interest["SEC"], "TNOTE2Y")
                self.assertEqual(fixed_income_interest["SEC_TYPE"], "fius")
                self.assertGreater(fixed_income_interest["AMOUNT"], 0)

                holdings = pd.read_csv(
                    _PACKAGED_AXYS_DIRECTORY / snapshot_directory / "holdings.csv"
                )
                fixed_income_holdings = holdings.loc[holdings["SEC"] == "TNOTE2Y"]

                self.assertFalse(fixed_income_holdings.empty)
                self.assertGreater(fixed_income_holdings["ACCRUED"].sum(), 0)

                frame = TransactionsLoader(specification).load(
                    cast(Literal["a", "b"], snapshot_key)
                )
                assert frame is not None
                resolved_row = _transaction_row(
                    frame,
                    portfolio="INCOME",
                    transaction_date="2026-05-15",
                    security="TNOTE2Y",
                    transaction_code="in",
                )

                self.assertEqual(resolved_row[pc_cols.TRANSACTION_CODE], "in")
                self.assertEqual(
                    resolved_row[pc_cols.TRANSACTION_CATEGORY],
                    "income",
                )
                self.assertEqual(
                    resolved_row[pc_cols.PERFORMANCE_FLOW_SIGN],
                    "performance",
                )
                self.assertIn(
                    resolved_row[pc_cols.TRANSACTION_SEMANTICS_SOURCE],
                    {"mixed", "yaml_rule"},
                )

                if snapshot_key == "b":
                    frame = TransactionsLoader(specification).load(snapshot_key)
                    assert frame is not None
                    for code, transaction_date, category, cash_sign in (
                        (
                            "pa",
                            "2026-02-10",
                            "fee_expense",
                            TRANSACTION_CASH_FLOW_SIGN_NEGATIVE,
                        ),
                        ("sa", "2026-02-20", "income", "positive"),
                    ):
                        accrued_row = _transaction_row(
                            frame,
                            portfolio="INCOME",
                            transaction_date=transaction_date,
                            security="TNOTE5Y",
                            transaction_code=code,
                        )
                        self.assertEqual(
                            fixed_income_transaction_boundary(code),
                            "accrued_interest_adjunct",
                        )
                        self.assertEqual(
                            accrued_row[pc_cols.TRANSACTION_CATEGORY],
                            category,
                        )
                        self.assertEqual(
                            accrued_row[pc_cols.CASH_FLOW_SIGN],
                            cash_sign,
                        )
                        self.assertEqual(
                            accrued_row[pc_cols.PERFORMANCE_FLOW_SIGN],
                            TRANSACTION_PERFORMANCE_FLOW_SIGN_PERFORMANCE,
                        )

    def test_packaged_demo_wd_uses_contextual_external_flow_rule(self) -> None:
        """Packaged Axys wd rows classify external flow from context, not code alone."""
        specification = PerformanceComparisonSpecification(_PACKAGED_COMPARISON_PATH)
        frame = TransactionsLoader(specification).load("a")
        assert frame is not None

        row = _transaction_row(
            frame,
            portfolio="ALPHA",
            transaction_date="2026-01-20",
            security="CASH_USD",
            transaction_code="wd",
        )

        self.assertEqual(row[pc_cols.TRANSACTION_CODE], "wd")
        self.assertEqual(row[pc_cols.SOURCE_DESTINATION_TYPE], "$pty")
        self.assertEqual(row[pc_cols.SOURCE_DESTINATION_SYMBOL], "$cash")
        self.assertEqual(row[pc_cols.TRANSACTION_CATEGORY], "external_flow")
        self.assertEqual(
            row[pc_cols.TRANSACTION_SEMANTICS_SOURCE],
            "yaml_rule",
        )

    def test_packaged_demo_dp_uses_contextual_fee_rule(self) -> None:
        """Packaged Axys dp fee rows stay performance items, not external flows."""
        specification = PerformanceComparisonSpecification(_PACKAGED_COMPARISON_PATH)
        frame = TransactionsLoader(specification).load("a")
        assert frame is not None

        row = _transaction_row(
            frame,
            portfolio="INCOME",
            transaction_date="2026-01-20",
            security="CASH_USD",
            transaction_code="dp",
        )

        self.assertEqual(row[pc_cols.TRANSACTION_CODE], "dp")
        self.assertEqual(row[pc_cols.SPECIAL_SECURITY_TYPE], "exus")
        self.assertEqual(row[pc_cols.SPECIAL_SECURITY_SYMBOL], "custfee")
        self.assertEqual(row[pc_cols.TRANSACTION_CATEGORY], "fee_expense")
        self.assertEqual(row[pc_cols.PERFORMANCE_FLOW_SIGN], "performance")
        self.assertEqual(
            row[pc_cols.TRANSACTION_SEMANTICS_SOURCE],
            "yaml_rule",
        )

    def test_packaged_demo_cause_fields_match_source_contract(self) -> None:
        """Performance Difference Causes only contains approved demo fields."""
        findings = compare_snapshots(
            _PACKAGED_COMPARISON_PATH,
            require_causal_attribution=True,
        )

        causes = _workbook_underlying_causes_table(findings)
        cause_fields = _dataset_fields(causes)

        self.assertEqual(cause_fields, _PERFORMANCE_DIFFERENCE_CAUSE_FIELDS)
        self.assertNotIn((pc_cols.HOLDINGS, pc_cols.COST), cause_fields)

    def test_packaged_demo_audit_fields_match_source_contract(self) -> None:
        """Cost is not exposed in the packaged Axys/APX demo evidence surface."""
        findings = compare_snapshots(
            _PACKAGED_COMPARISON_PATH,
            require_causal_attribution=True,
        )

        causes = _workbook_underlying_causes_table(findings)
        raw_audit_trail = _workbook_raw_audit_trail_table(findings)
        cause_fields = _dataset_fields(causes)
        raw_fields = _dataset_fields(raw_audit_trail)

        self.assertNotIn((pc_cols.HOLDINGS, pc_cols.COST), raw_fields)
        self.assertNotIn((pc_cols.HOLDINGS, pc_cols.COST), cause_fields)

    def test_packaged_demo_accrual_changes_are_performance_causes(self) -> None:
        """Configured accrual amount changes remain performance-cause rows."""
        findings = compare_snapshots(
            _PACKAGED_COMPARISON_PATH,
            require_causal_attribution=True,
        )

        causes = _workbook_underlying_causes_table(findings)
        accrued_causes = causes.filter(
            (pl.col("dataset") == pc_cols.HOLDINGS)
            & (pl.col("source_column") == pc_cols.ACCRUED)
        )

        self.assertGreater(accrued_causes.height, 0)

    def test_packaged_demo_split_factor_supports_holding_cause(self) -> None:
        """Split-factor evidence supports the CVNA holding-value correction."""
        findings = compare_snapshots(
            _PACKAGED_COMPARISON_PATH,
            require_causal_attribution=True,
        )

        causes = _workbook_underlying_causes_table(findings)
        split_causes = causes.filter(
            (pl.col("dataset") == pc_cols.SPLITS)
            & (pl.col("source_column") == pc_cols.SPLIT_FACTOR)
            & (pl.col("security_id") == "CVNA")
        )

        self.assertEqual(split_causes.height, 1)
        self.assertEqual(
            split_causes["review_guidance"][0],
            (
                "split: Caused CVNA holdings.quantity and related "
                "holdings.market_value to increase using a 5.0 split factor."
            ),
        )

    def test_packaged_demo_rc_row_explains_return_of_capital_cash_effect(self) -> None:
        """Return-of-capital row has explicit reviewer-facing cash wording."""
        findings = compare_snapshots(
            _PACKAGED_COMPARISON_PATH,
            require_causal_attribution=True,
        )

        causes = _workbook_underlying_causes_table(findings)
        rc_causes = causes.filter(
            (pl.col("dataset") == pc_cols.TRANSACTIONS)
            & (pl.col("source_column") == pc_cols.AMOUNT)
            & (pl.col("security_id") == "JPM")
            & (pl.col("snapshot_b_value") == 240.0)
            & pl.col("review_guidance").str.starts_with("rc:")
        )

        self.assertEqual(rc_causes.height, 1)
        self.assertEqual(
            rc_causes["review_guidance"][0],
            (
                "rc: Caused cash-balance ending holdings.market_value "
                "to increase by 240.00."
            ),
        )

    def test_packaged_demo_pd_row_explains_principal_paydown_cash_effect(self) -> None:
        """Principal-paydown row has explicit reviewer-facing cash wording."""
        findings = compare_snapshots(
            _PACKAGED_COMPARISON_PATH,
            require_causal_attribution=True,
        )

        causes = _workbook_underlying_causes_table(findings)
        pd_causes = causes.filter(
            (pl.col("dataset") == pc_cols.TRANSACTIONS)
            & (pl.col("source_column") == pc_cols.AMOUNT)
            & (pl.col("security_id") == "MBSPOOL")
            & pl.col("review_guidance").str.starts_with("pd:")
        )

        self.assertEqual(pd_causes.height, 1)
        self.assertEqual(
            pd_causes["review_guidance"][0],
            (
                "pd: Caused cash-balance ending holdings.market_value "
                "to increase by 320.00."
            ),
        )

    def test_packaged_demo_intentional_review_status_examples(self) -> None:
        """Packaged reports preserve intentional partial and unresolved periods."""
        portfolio_findings = compare_snapshots(_PACKAGED_COMPARISON_PATH)
        portfolio_changes = _workbook_portfolio_changes_table(
            portfolio_findings,
            comparison_path=_PACKAGED_COMPARISON_PATH,
        )
        portfolio_statuses = {
            (
                row["portfolio_id"],
                row["from_date"].isoformat(),
                row["thru_date"].isoformat(),
            ): row
            for row in portfolio_changes.iter_rows(named=True)
        }

        partly_explained = portfolio_statuses[("BALANCED", "2026-05-09", "2026-05-14")]
        unexplained = portfolio_statuses[("INCOME", "2026-04-01", "2026-04-30")]
        self.assertEqual(partly_explained["review_status"], "Partly Explained")
        self.assertGreater(abs(partly_explained["estimated_cause_total"]), 0.0)
        self.assertGreater(abs(partly_explained["unexplained_change"]), 0.0)
        self.assertEqual(unexplained["review_status"], "Unexplained")
        self.assertEqual(unexplained["estimated_cause_total"], 0.0)
        self.assertGreater(abs(unexplained["unexplained_change"]), 0.0)

        security_findings = compare_snapshots(
            _PACKAGED_COMPARISON_PATH,
            comparison_level="security",
        )
        security_changes = _workbook_security_changes_table(
            security_findings,
            comparison_path=_PACKAGED_COMPARISON_PATH,
            comparison_level="security",
        )
        security_statuses = {
            (
                row["portfolio_id"],
                row["security_id"],
                row["from_date"].isoformat(),
                row["thru_date"].isoformat(),
            ): row
            for row in security_changes.iter_rows(named=True)
        }

        partly_explained = security_statuses[
            ("BALANCED", "MSFT", "2026-05-09", "2026-05-14")
        ]
        unexplained = security_statuses[
            ("INCOME", "TNOTE5Y", "2026-04-01", "2026-04-30")
        ]
        self.assertEqual(partly_explained["review_status"], "Partly Explained")
        self.assertGreater(abs(partly_explained["estimated_cause_total"]), 0.0)
        self.assertGreater(abs(partly_explained["unexplained_change"]), 0.0)
        self.assertEqual(unexplained["review_status"], "Unexplained")
        self.assertEqual(unexplained["estimated_cause_total"], 0.0)
        self.assertGreater(abs(unexplained["unexplained_change"]), 0.0)

    def test_packaged_performance_comparison_demo_data_foots(self) -> None:
        """Packaged demo data has no accidental accounting or residual issues."""
        audit_module = _load_audit_module()

        issues = audit_module.audit_demo_data()

        self.assertEqual(issues, [])

    def test_packaged_holdings_are_derived_from_transaction_and_holding_scenarios(
        self,
    ) -> None:
        """Snapshot B holdings match transaction-derived plus explicit scenarios."""
        rebuild_module = _load_rebuild_module()

        summary = rebuild_module.rebuild_demo_performance_files(
            rebuild_module._DEFAULT_AXYS_DIRECTORY,
            write=False,
        )
        snapshots = {snapshot["snapshot"]: snapshot for snapshot in summary["snapshots"]}

        self.assertEqual(snapshots["snapshot_a"]["max_holdings_numeric_delta"], 0.0)
        self.assertEqual(snapshots["snapshot_b"]["max_transaction_numeric_delta"], 0.0)
        self.assertFalse(snapshots["snapshot_b"]["has_transaction_field_drift"])
        self.assertEqual(snapshots["snapshot_b"]["max_holdings_numeric_delta"], 0.0)
        self.assertEqual(snapshots["snapshot_b"]["transaction_scenario_rows"], 21)
        self.assertEqual(
            snapshots["snapshot_b"]["transaction_scenarios_by_type"],
            {
                "by": 2,
                "cs": 1,
                "dp": 1,
                "dv": 4,
                "in": 1,
                "li": 1,
                "lo": 1,
                "pa": 2,
                "pd": 1,
                "rc": 2,
                "sa": 1,
                "sl": 2,
                "ss": 1,
                "wd": 1,
            },
        )
        self.assertEqual(snapshots["snapshot_b"]["transaction_derived_holding_rows"], 35)
        self.assertEqual(
            snapshots["snapshot_b"]["transaction_derived_holdings_by_type"],
            {
                "by": 6,
                "cs": 1,
                "dp": 1,
                "dv": 7,
                "in": 3,
                "li": 1,
                "lo": 1,
                "pa": 3,
                "pd": 4,
                "rc": 1,
                "sa": 1,
                "sl": 4,
                "ss": 1,
                "wd": 1,
            },
        )
        self.assertEqual(snapshots["snapshot_b"]["holding_scenario_rows"], 9)
        self.assertEqual(
            snapshots["snapshot_b"]["holding_scenarios_by_type"],
            {
                "accrual_correction": 1,
                "cash_balance_correction": 1,
                "cost_only_correction": 1,
                "quantity_valuation_correction": 2,
                "valuation_mark": 3,
                "x_ref_holdings_accrued_rate": 1,
            },
        )
        self.assertFalse(snapshots["snapshot_b"]["has_transaction_drift"])
        self.assertFalse(snapshots["snapshot_b"]["has_holdings_drift"])

    def test_remaining_holding_scenarios_are_classified(self) -> None:
        """Residual holding scenarios have explicit scenario types."""
        rebuild_module = _load_rebuild_module()

        scenarios = rebuild_module._load_holding_scenarios(
            rebuild_module._DEFAULT_HOLDING_SCENARIOS_PATH,
        )
        type_counts = {}
        for adjustment in scenarios.for_snapshot("snapshot_b"):
            type_counts[adjustment.scenario_type] = (
                type_counts.get(adjustment.scenario_type, 0) + 1
            )

        self.assertEqual(
            type_counts,
            {
                "valuation_mark": 3,
                "cash_balance_correction": 1,
                "quantity_valuation_correction": 2,
                "accrual_correction": 1,
                "cost_only_correction": 1,
                "x_ref_holdings_accrued_rate": 1,
            },
        )

    def test_scenario_coverage_audit_detects_lost_examples(self) -> None:
        """Scenario coverage guardrail catches missing demo story examples."""
        rebuild_module = _load_rebuild_module()
        summary = {
            "snapshots": [
                {
                    "snapshot": "snapshot_b",
                    "transaction_scenarios_by_type": {"by": 1},
                    "transaction_derived_holdings_by_type": {},
                    "holding_scenarios_by_type": {},
                }
            ]
        }

        issues = rebuild_module._audit_scenario_coverage(summary)

        self.assertGreaterEqual(len(issues), 1)
        self.assertEqual(issues[0].check, "scenario_coverage")

    def test_scenario_calendar_covers_intentional_demo_rows(self) -> None:
        """The simplification calendar covers every current intentional row."""
        rebuild_module = _load_rebuild_module()
        calendar = rebuild_module._load_scenario_calendar(
            rebuild_module._DEFAULT_SCENARIO_CALENDAR_PATH,
        )
        holding_scenarios = rebuild_module._load_holding_scenarios(
            rebuild_module._DEFAULT_HOLDING_SCENARIOS_PATH,
        )
        transaction_scenarios = rebuild_module._load_transaction_scenarios(
            rebuild_module._DEFAULT_TRANSACTION_SCENARIOS_PATH,
        )

        issues = rebuild_module._audit_scenario_calendar(
            calendar=calendar,
            holding_scenarios=holding_scenarios,
            transaction_scenarios=transaction_scenarios,
            axys_directory=rebuild_module._DEFAULT_AXYS_DIRECTORY,
        )

        self.assertEqual(issues, [])

    def test_scenario_calendar_density_confirms_simplified_periods(self) -> None:
        """The calendar confirms every demo period is within the density target."""
        rebuild_module = _load_rebuild_module()
        calendar = rebuild_module._load_scenario_calendar(
            rebuild_module._DEFAULT_SCENARIO_CALENDAR_PATH,
        )

        density_rows = rebuild_module._scenario_calendar_density(calendar)
        density_by_period = {
            (row["portfolio"], row["from_date"], row["thru_date"]): row
            for row in density_rows
        }

        balanced_may_mark = density_by_period[
            ("BALANCED", "2026-05-01", "2026-05-08")
        ]
        balanced_may_corrections = density_by_period[
            ("BALANCED", "2026-05-09", "2026-05-14")
        ]
        balanced_may_short = density_by_period[
            ("BALANCED", "2026-05-15", "2026-05-29")
        ]
        income_february_buy = density_by_period[
            ("INCOME", "2026-01-31", "2026-02-13")
        ]
        income_february_sell = density_by_period[
            ("INCOME", "2026-02-14", "2026-02-27")
        ]
        income_may_mark = density_by_period[("INCOME", "2026-05-01", "2026-05-08")]
        income_may_dividend_payable = density_by_period[
            ("INCOME", "2026-05-09", "2026-05-14")
        ]
        income_may_income = density_by_period[
            ("INCOME", "2026-05-15", "2026-05-15")
        ]
        income_may_paydown = density_by_period[
            ("INCOME", "2026-05-16", "2026-05-22")
        ]
        income_may_late_dividend = density_by_period[
            ("INCOME", "2026-05-23", "2026-05-29")
        ]
        alpha_may = density_by_period[("ALPHA", "2026-05-01", "2026-05-29")]
        self.assertEqual(balanced_may_mark["current_difference_rows"], 2)
        self.assertFalse(balanced_may_mark["needs_intra_month_split"])
        self.assertEqual(balanced_may_corrections["current_difference_rows"], 2)
        self.assertFalse(balanced_may_corrections["needs_intra_month_split"])
        self.assertEqual(balanced_may_short["current_difference_rows"], 2)
        self.assertFalse(balanced_may_short["needs_intra_month_split"])
        self.assertEqual(income_february_buy["current_difference_rows"], 2)
        self.assertFalse(income_february_buy["needs_intra_month_split"])
        self.assertEqual(income_february_sell["current_difference_rows"], 2)
        self.assertFalse(income_february_sell["needs_intra_month_split"])
        self.assertEqual(income_may_mark["current_difference_rows"], 1)
        self.assertFalse(income_may_mark["needs_intra_month_split"])
        self.assertEqual(income_may_dividend_payable["current_difference_rows"], 1)
        self.assertFalse(income_may_dividend_payable["needs_intra_month_split"])
        self.assertEqual(income_may_income["current_difference_rows"], 2)
        self.assertFalse(income_may_income["needs_intra_month_split"])
        self.assertEqual(income_may_paydown["current_difference_rows"], 1)
        self.assertFalse(income_may_paydown["needs_intra_month_split"])
        self.assertEqual(income_may_late_dividend["current_difference_rows"], 1)
        self.assertFalse(income_may_late_dividend["needs_intra_month_split"])
        self.assertEqual(alpha_may["current_difference_rows"], 2)
        self.assertFalse(alpha_may["needs_intra_month_split"])
        self.assertFalse(any(row["needs_intra_month_split"] for row in density_rows))

    def test_scenario_readability_matrix_names_each_period_story(self) -> None:
        """The scenario matrix keeps the demo stories reviewable by period."""
        rebuild_module = _load_rebuild_module()
        calendar = rebuild_module._load_scenario_calendar(
            rebuild_module._DEFAULT_SCENARIO_CALENDAR_PATH,
        )

        matrix_rows = rebuild_module._scenario_readability_matrix(calendar)
        matrix_by_period = {
            (row["portfolio"], row["from_date"], row["thru_date"]): row
            for row in matrix_rows
        }

        for row in matrix_rows:
            with self.subTest(
                portfolio=row["portfolio"],
                from_date=row["from_date"],
                thru_date=row["thru_date"],
            ):
                self.assertTrue(row["within_target"])
                self.assertLessEqual(row["expected_difference_rows"], 2)
                self.assertTrue(row["scenario_families"])
                self.assertTrue(row["primary_securities"])
                self.assertTrue(row["scenario_keys"])
                self.assertTrue(row["scenario_notes"])

        balanced_short = matrix_by_period[("BALANCED", "2026-05-15", "2026-05-29")]
        self.assertEqual(balanced_short["scenario_families"], ["short_side_trade"])
        self.assertEqual(balanced_short["primary_securities"], ["TSLA"])
        self.assertEqual(balanced_short["expected_difference_rows"], 2)

        income_paydown = matrix_by_period[("INCOME", "2026-05-16", "2026-05-22")]
        self.assertEqual(income_paydown["scenario_families"], ["principal_paydown"])
        self.assertEqual(income_paydown["primary_securities"], ["MBSPOOL"])
        self.assertEqual(income_paydown["expected_difference_rows"], 1)
        income_dividend_payable = matrix_by_period[
            ("INCOME", "2026-05-09", "2026-05-14")
        ]
        self.assertEqual(
            income_dividend_payable["scenario_families"],
            ["missed_late_dividend"],
        )
        self.assertEqual(income_dividend_payable["primary_securities"], ["AAPL"])
        self.assertEqual(income_dividend_payable["expected_difference_rows"], 1)
        income_late_dividend = matrix_by_period[
            ("INCOME", "2026-05-23", "2026-05-29")
        ]
        self.assertEqual(
            income_late_dividend["scenario_families"],
            ["missed_late_dividend"],
        )
        self.assertEqual(income_late_dividend["primary_securities"], ["AAPL"])
        self.assertEqual(income_late_dividend["expected_difference_rows"], 1)

    def test_period_split_plan_has_no_remaining_crowded_periods(self) -> None:
        """The split backlog is empty because every demo period is in target."""
        rebuild_module = _load_rebuild_module()
        calendar = rebuild_module._load_scenario_calendar(
            rebuild_module._DEFAULT_SCENARIO_CALENDAR_PATH,
        )
        plan = rebuild_module._load_period_split_plan(
            rebuild_module._DEFAULT_PERIOD_SPLIT_PLAN_PATH,
        )

        issues = rebuild_module._audit_period_split_plan(
            plan=plan,
            calendar=calendar,
        )

        self.assertEqual(issues, [])
        self.assertEqual(len(plan), 0)
        self.assertEqual(rebuild_module._crowded_scenario_calendar_keys(calendar), set())
        self.assertEqual(rebuild_module._scenario_period_split_plan_summary(plan), [])

    def test_transaction_scenarios_create_expected_holding_impacts(self) -> None:
        """Transaction changes create the expected cash and security adjustments."""
        rebuild_module = _load_rebuild_module()
        axys_directory = rebuild_module._DEFAULT_AXYS_DIRECTORY
        base_holdings = pd.read_csv(axys_directory / "snapshot_a" / "holdings.csv")
        base_transactions = rebuild_module._read_packaged_transactions(
            axys_directory / "snapshot_a" / "transactions.csv"
        )
        current_transactions = rebuild_module._read_packaged_transactions(
            axys_directory / "snapshot_b" / "transactions.csv"
        )
        scenarios = rebuild_module._load_transaction_scenarios(
            rebuild_module._DEFAULT_TRANSACTION_SCENARIOS_PATH,
        )
        rebuilt_transactions = rebuild_module._rebuild_transactions(
            "snapshot_b",
            current_transactions=current_transactions,
            base_transactions=base_transactions,
            transaction_scenarios=scenarios,
        )
        periods = pd.read_csv(axys_directory / "snapshot_b" / "portperf.csv")

        adjustments = rebuild_module._transaction_derived_holding_adjustments(
            "snapshot_b",
            base_holdings=base_holdings,
            base_transactions=base_transactions,
            current_transactions=rebuilt_transactions,
            periods=periods,
        )
        by_scenario = {adjustment.scenario: adjustment for adjustment in adjustments}

        self.assertEqual(len(adjustments), 35)
        self.assertNotIn(
            "BALANCED0503 ; transaction changes cash balance.",
            by_scenario,
        )
        self.assertNotIn(
            "BALANCED0503 ; transaction changes ending holding.",
            by_scenario,
        )
        self._assert_adjustment(
            by_scenario["ALPHA0203 wd transaction changes cash balance."],
            portfolio="ALPHA",
            security="CASH_USD",
            holding_date="2026-01-30",
            deltas={"QTY": -1500.0, "MKT_VAL": -1500.0, "COST": -1500.0},
        )
        self._assert_adjustment(
            by_scenario["BALANCED0603 ss transaction changes cash balance."],
            portfolio="BALANCED",
            security="CASH_USD",
            holding_date="2026-05-29",
            deltas={"QTY": 21112.0, "MKT_VAL": 21112.0, "COST": 21112.0},
        )
        self._assert_adjustment(
            by_scenario["BALANCED0604 cs transaction changes cash balance."],
            portfolio="BALANCED",
            security="CASH_USD",
            holding_date="2026-05-29",
            deltas={"QTY": -21789.5, "MKT_VAL": -21789.5, "COST": -21789.5},
        )
        self._assert_adjustment(
            by_scenario["INCOME0203 dp transaction changes cash balance."],
            portfolio="INCOME",
            security="CASH_USD",
            holding_date="2026-01-30",
            deltas={"QTY": -50.0, "MKT_VAL": -50.0, "COST": -50.0},
        )
        self._assert_adjustment(
            by_scenario["ALPHA0304 pa transaction changes cash balance."],
            portfolio="ALPHA",
            security="CASH_USD",
            holding_date="2026-02-27",
            deltas={"QTY": -8.0, "MKT_VAL": -8.0, "COST": -8.0},
        )
        self._assert_adjustment(
            by_scenario["BALANCED0502 dv transaction changes cash balance."],
            portfolio="BALANCED",
            security="CASH_USD",
            holding_date="2026-04-30",
            deltas={"QTY": 117.07, "MKT_VAL": 117.07, "COST": 117.07},
        )
        self._assert_adjustment(
            by_scenario["BALANCED0503 rc transaction changes cash balance."],
            portfolio="BALANCED",
            security="CASH_USD",
            holding_date="2026-04-30",
            deltas={"QTY": 240.0, "MKT_VAL": 240.0, "COST": 240.0},
        )
        self._assert_adjustment(
            by_scenario["INCOME0603 in transaction changes cash balance."],
            portfolio="INCOME",
            security="CASH_USD",
            holding_date="2026-05-29",
            deltas={"QTY": 80.0, "MKT_VAL": 80.0, "COST": 80.0},
        )
        self._assert_adjustment(
            by_scenario["INCOME0605 pd transaction changes ending holding."],
            portfolio="INCOME",
            security="MBSPOOL",
            holding_date="2026-05-29",
            deltas={"MKT_VAL": -320.0, "COST": -320.0},
        )
        self._assert_adjustment(
            by_scenario["INCOME0605 pd transaction changes cash balance."],
            portfolio="INCOME",
            security="CASH_USD",
            holding_date="2026-05-29",
            deltas={"QTY": 320.0, "MKT_VAL": 320.0, "COST": 320.0},
        )
        dividend_payable_adjustment = next(
            adjustment
            for adjustment in adjustments
            if adjustment.scenario == "INCOME0606 dv transaction changes cash balance."
            and adjustment.holding_date == "2026-05-14"
        )
        self._assert_adjustment(
            dividend_payable_adjustment,
            portfolio="INCOME",
            security="CASH_USD",
            holding_date="2026-05-14",
            deltas={"QTY": 220.97, "MKT_VAL": 220.97, "COST": 220.97},
        )
        late_dividend_adjustment = next(
            adjustment
            for adjustment in adjustments
            if adjustment.scenario == "INCOME0604 dv transaction changes cash balance."
            and adjustment.holding_date == "2026-05-29"
        )
        self._assert_adjustment(
            late_dividend_adjustment,
            portfolio="INCOME",
            security="CASH_USD",
            holding_date="2026-05-29",
            deltas={"QTY": -220.97, "MKT_VAL": -220.97, "COST": -220.97},
        )
        self._assert_adjustment(
            by_scenario["INCOME0303 by transaction changes ending holding."],
            portfolio="INCOME",
            security="TNOTE5Y",
            holding_date="2026-02-27",
            deltas={
                "QTY": 5.0,
                "MKT_VAL": 494.0,
                "COST": 494.0,
                "ACCRUED": 5.0 * 4.56 / 485.83,
            },
        )
        self._assert_adjustment(
            by_scenario["INCOME0303 by transaction changes cash balance."],
            portfolio="INCOME",
            security="CASH_USD",
            holding_date="2026-02-27",
            deltas={"QTY": -494.0, "MKT_VAL": -494.0, "COST": -494.0},
        )
        self._assert_adjustment(
            by_scenario["INCOME0304 pa transaction changes cash balance."],
            portfolio="INCOME",
            security="CASH_USD",
            holding_date="2026-02-27",
            deltas={"QTY": -42.5, "MKT_VAL": -42.5, "COST": -42.5},
        )
        self._assert_adjustment(
            by_scenario["INCOME0305 sl transaction changes ending holding."],
            portfolio="INCOME",
            security="TNOTE5Y",
            holding_date="2026-02-27",
            deltas={
                "QTY": -3.0,
                "MKT_VAL": -296.4,
                "COST": -296.3999753000021,
                "ACCRUED": -3.0 * 4.56 / 485.83,
            },
        )
        self._assert_adjustment(
            by_scenario["INCOME0305 sl transaction changes cash balance."],
            portfolio="INCOME",
            security="CASH_USD",
            holding_date="2026-02-27",
            deltas={"QTY": 296.4, "MKT_VAL": 296.4, "COST": 296.4},
        )
        self._assert_adjustment(
            by_scenario["INCOME0306 sa transaction changes cash balance."],
            portfolio="INCOME",
            security="CASH_USD",
            holding_date="2026-02-27",
            deltas={"QTY": 37.25, "MKT_VAL": 37.25, "COST": 37.25},
        )
        self._assert_adjustment(
            by_scenario["ALPHA0401 by transaction changes ending holding."],
            portfolio="ALPHA",
            security="AAPL",
            holding_date="2026-03-31",
            deltas={"QTY": 1.1372, "MKT_VAL": 183.0892, "COST": 183.0892},
        )
        self._assert_adjustment(
            by_scenario["ALPHA0401 by transaction changes cash balance."],
            portfolio="ALPHA",
            security="CASH_USD",
            holding_date="2026-03-31",
            deltas={"QTY": -196.98, "MKT_VAL": -196.98, "COST": -196.98},
        )
        self._assert_adjustment(
            by_scenario["ALPHA0503 dv transaction changes cash balance."],
            portfolio="ALPHA",
            security="CASH_USD",
            holding_date="2026-04-30",
            deltas={"QTY": 537.01, "MKT_VAL": 537.01, "COST": 537.01},
        )
        self._assert_adjustment(
            by_scenario["BALANCED0203 sl transaction changes ending holding."],
            portfolio="BALANCED",
            security="MSFT",
            holding_date="2026-01-30",
            deltas={"QTY": -2.0, "MKT_VAL": -228.0, "COST": -227.9999989303458},
        )
        self._assert_adjustment(
            by_scenario["BALANCED0203 sl transaction changes cash balance."],
            portfolio="BALANCED",
            security="CASH_USD",
            holding_date="2026-01-30",
            deltas={"QTY": 226.0, "MKT_VAL": 226.0, "COST": 226.0},
        )
        self._assert_adjustment(
            by_scenario["BALANCED0403 li transaction changes cash balance."],
            portfolio="BALANCED",
            security="CASH_USD",
            holding_date="2026-03-31",
            deltas={"QTY": 2500.0, "MKT_VAL": 2500.0, "COST": 2500.0},
        )
        self._assert_adjustment(
            by_scenario["ALPHA0303 lo transaction changes cash balance."],
            portfolio="ALPHA",
            security="CASH_USD",
            holding_date="2026-02-27",
            deltas={"QTY": -2000.0, "MKT_VAL": -2000.0, "COST": -2000.0},
        )

    def test_holding_scenario_file_requires_exact_columns(self) -> None:
        """Scenario CSV shape errors fail before any demo files are rebuilt."""
        rebuild_module = _load_rebuild_module()

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bad_scenarios.csv"
            path.write_text("snapshot,PORT\nsnapshot_b,ALPHA\n")

            with self.assertRaisesRegex(ValueError, "columns must exactly match"):
                rebuild_module._load_holding_scenarios(path)

    def test_holding_scenario_file_rejects_base_snapshot_adjustments(self) -> None:
        """Scenario rows must target derived snapshots, not the base snapshot."""
        rebuild_module = _load_rebuild_module()
        columns = ",".join(rebuild_module._HOLDING_SCENARIO_COLUMNS)
        row = (
            "snapshot_a,cash_balance_correction,ALPHA,CASH_USD,2026-01-30,"
            "1,0,1,1,0,Invalid base adjustment\n"
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bad_scenarios.csv"
            path.write_text(f"{columns}\n{row}")

            with self.assertRaisesRegex(ValueError, "derived snapshots"):
                rebuild_module._load_holding_scenarios(path)

    def test_holding_scenario_file_validates_type_delta_patterns(self) -> None:
        """Scenario type controls which holding fields may change."""
        rebuild_module = _load_rebuild_module()
        columns = ",".join(rebuild_module._HOLDING_SCENARIO_COLUMNS)
        row = (
            "snapshot_b,valuation_mark,ALPHA,AAPL,2026-05-29,"
            "1,0,1,0,0,Invalid valuation mark quantity change\n"
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bad_scenarios.csv"
            path.write_text(f"{columns}\n{row}")

            with self.assertRaisesRegex(ValueError, "changed fields"):
                rebuild_module._load_holding_scenarios(path)

    def test_transaction_scenario_file_requires_exact_columns(self) -> None:
        """Transaction scenario CSV shape errors fail before rebuilds."""
        rebuild_module = _load_rebuild_module()

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bad_transaction_scenarios.csv"
            path.write_text("snapshot,TRANSACTION_ID\nsnapshot_b,ALPHA0203\n")

            with self.assertRaisesRegex(ValueError, "columns must exactly match"):
                rebuild_module._load_transaction_scenarios(path)

    def test_transaction_scenario_file_rejects_base_snapshot_adjustments(self) -> None:
        """Transaction scenario rows must target derived snapshots."""
        rebuild_module = _load_rebuild_module()
        row: dict[str, object] = {
            column: ""
            for column in rebuild_module._TRANSACTION_SCENARIO_COLUMNS
        }
        row.update(
            {
                "snapshot": "snapshot_a",
                "action": "adjust",
                "TRANSACTION_ID": "ALPHA0203",
                "QTY_delta": 0,
                "PRICE_delta": 0,
                "AMOUNT_delta": -1,
                "COMMISSION_delta": 0,
                "scenario": "Invalid base adjustment",
            }
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bad_transaction_scenarios.csv"
            pd.DataFrame([row], columns=rebuild_module._TRANSACTION_SCENARIO_COLUMNS).to_csv(
                path,
                index=False,
            )

            with self.assertRaisesRegex(ValueError, "derived snapshots"):
                rebuild_module._load_transaction_scenarios(path)

    def test_transaction_scenarios_can_insert_external_cash_flows(self) -> None:
        """Explicit inserted li/lo rows can drive cash-flow scenarios."""
        rebuild_module = _load_rebuild_module()
        axys_directory = rebuild_module._DEFAULT_AXYS_DIRECTORY
        base_holdings = pd.read_csv(axys_directory / "snapshot_a" / "holdings.csv")
        base_transactions = rebuild_module._read_packaged_transactions(
            axys_directory / "snapshot_a" / "transactions.csv"
        )
        periods = pd.read_csv(axys_directory / "snapshot_b" / "portperf.csv")
        scenario_rows: list[dict[str, object]] = []
        for transaction_id, transaction_code, amount, scenario in (
            ("BALANCED0403", "li", 2500, "Test-only contribution insertion."),
            ("BALANCED0404", "lo", -900, "Test-only withdrawal insertion."),
        ):
            row: dict[str, object] = {
                column: ""
                for column in rebuild_module._TRANSACTION_SCENARIO_COLUMNS
            }
            row.update(
                {
                    "snapshot": "snapshot_b",
                    "action": "insert",
                    "TRANSACTION_ID": transaction_id,
                    "PORT": "BALANCED",
                    "TRANSACTION_DATE": "2026-03-20",
                    "SETTLE_DATE": "2026-03-20",
                    "SEC": "CASH_USD",
                    "TRAN": transaction_code,
                    "SEC_TYPE": "caus",
                    "SRC_DEST_TYPE": "$pty",
                    "SRC_DEST_SYMBOL": "$cash",
                    "QTY": 0,
                    "PRICE": 0,
                    "AMOUNT": amount,
                    "COMMISSION": 0,
                    "QTY_delta": 0,
                    "PRICE_delta": 0,
                    "AMOUNT_delta": 0,
                    "COMMISSION_delta": 0,
                    "scenario": scenario,
                }
            )
            scenario_rows.append(row)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "transaction_scenarios.csv"
            pd.DataFrame(
                scenario_rows,
                columns=rebuild_module._TRANSACTION_SCENARIO_COLUMNS,
            ).to_csv(
                path,
                index=False,
            )
            scenarios = rebuild_module._load_transaction_scenarios(path)

        rebuilt_transactions = rebuild_module._rebuild_transactions(
            "snapshot_b",
            current_transactions=base_transactions,
            base_transactions=base_transactions,
            transaction_scenarios=scenarios,
        )
        inserted = rebuilt_transactions[
            rebuilt_transactions["TRANSACTION_ID"].isin(["BALANCED0403", "BALANCED0404"])
        ].sort_values("TRANSACTION_ID")
        self.assertEqual(inserted["TRAN"].to_list(), ["li", "lo"])
        self.assertEqual(inserted["AMOUNT"].astype(float).to_list(), [2500.0, -900.0])

        adjustments = rebuild_module._transaction_derived_holding_adjustments(
            "snapshot_b",
            base_holdings=base_holdings,
            base_transactions=base_transactions,
            current_transactions=rebuilt_transactions,
            periods=periods,
        )
        by_scenario = {adjustment.scenario: adjustment for adjustment in adjustments}
        self._assert_adjustment(
            by_scenario["BALANCED0403 li transaction changes cash balance."],
            portfolio="BALANCED",
            security="CASH_USD",
            holding_date="2026-03-31",
            deltas={"QTY": 2500.0, "MKT_VAL": 2500.0, "COST": 2500.0},
        )
        self._assert_adjustment(
            by_scenario["BALANCED0404 lo transaction changes cash balance."],
            portfolio="BALANCED",
            security="CASH_USD",
            holding_date="2026-03-31",
            deltas={"QTY": -900.0, "MKT_VAL": -900.0, "COST": -900.0},
        )

    def test_transaction_scenarios_can_insert_accrued_interest_adjuncts(self) -> None:
        """Explicit pa/sa rows can drive cash settlement scenarios."""
        rebuild_module = _load_rebuild_module()
        axys_directory = rebuild_module._DEFAULT_AXYS_DIRECTORY
        base_holdings = pd.read_csv(axys_directory / "snapshot_a" / "holdings.csv")
        base_transactions = rebuild_module._read_packaged_transactions(
            axys_directory / "snapshot_a" / "transactions.csv"
        )
        periods = pd.read_csv(axys_directory / "snapshot_b" / "portperf.csv")
        scenario_rows: list[dict[str, object]] = []
        for transaction_id, transaction_code, amount, scenario in (
            ("TESTPA", "pa", -42.5, "Test-only purchase accrued interest."),
            ("TESTSA", "sa", 37.25, "Test-only sale accrued interest."),
        ):
            row: dict[str, object] = {
                column: ""
                for column in rebuild_module._TRANSACTION_SCENARIO_COLUMNS
            }
            row.update(
                {
                    "snapshot": "snapshot_b",
                    "action": "insert",
                    "TRANSACTION_ID": transaction_id,
                    "PORT": "INCOME",
                    "TRANSACTION_DATE": "2026-05-15",
                    "SETTLE_DATE": "2026-05-15",
                    "SEC": "TNOTE5Y",
                    "TRAN": transaction_code,
                    "SEC_TYPE": "fius",
                    "SRC_DEST_TYPE": "$income",
                    "SRC_DEST_SYMBOL": "$cash",
                    "QTY": 0,
                    "PRICE": 0,
                    "AMOUNT": amount,
                    "COMMISSION": 0,
                    "QTY_delta": 0,
                    "PRICE_delta": 0,
                    "AMOUNT_delta": 0,
                    "COMMISSION_delta": 0,
                    "scenario": scenario,
                }
            )
            scenario_rows.append(row)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "transaction_scenarios.csv"
            pd.DataFrame(
                scenario_rows,
                columns=rebuild_module._TRANSACTION_SCENARIO_COLUMNS,
            ).to_csv(
                path,
                index=False,
            )
            scenarios = rebuild_module._load_transaction_scenarios(path)

        rebuilt_transactions = rebuild_module._rebuild_transactions(
            "snapshot_b",
            current_transactions=base_transactions,
            base_transactions=base_transactions,
            transaction_scenarios=scenarios,
        )
        inserted = rebuilt_transactions[
            rebuilt_transactions["TRANSACTION_ID"].isin(["TESTPA", "TESTSA"])
        ].sort_values("TRANSACTION_ID")
        self.assertEqual(inserted["TRAN"].to_list(), ["pa", "sa"])
        self.assertEqual(inserted["AMOUNT"].astype(float).to_list(), [-42.5, 37.25])

        adjustments = rebuild_module._transaction_derived_holding_adjustments(
            "snapshot_b",
            base_holdings=base_holdings,
            base_transactions=base_transactions,
            current_transactions=rebuilt_transactions,
            periods=periods,
        )
        by_scenario = {adjustment.scenario: adjustment for adjustment in adjustments}
        self._assert_adjustment(
            by_scenario["TESTPA pa transaction changes cash balance."],
            portfolio="INCOME",
            security="CASH_USD",
            holding_date="2026-05-29",
            deltas={"QTY": -42.5, "MKT_VAL": -42.5, "COST": -42.5},
        )
        self._assert_adjustment(
            by_scenario["TESTSA sa transaction changes cash balance."],
            portfolio="INCOME",
            security="CASH_USD",
            holding_date="2026-05-29",
            deltas={"QTY": 37.25, "MKT_VAL": 37.25, "COST": 37.25},
        )

    def _assert_adjustment(
        self,
        adjustment,
        *,
        portfolio: str,
        security: str,
        holding_date: str,
        deltas: dict[str, float],
    ) -> None:
        """Assert one derived holding adjustment has the expected accounting impact."""
        self.assertEqual(adjustment.portfolio, portfolio)
        self.assertEqual(adjustment.security, security)
        self.assertEqual(adjustment.holding_date, holding_date)
        for column, expected_delta in deltas.items():
            self.assertAlmostEqual(adjustment.deltas[column], expected_delta, places=6)
        for column in {"QTY", "PRICE", "MKT_VAL", "COST", "ACCRUED"} - set(deltas):
            self.assertEqual(adjustment.deltas[column], 0.0)


if __name__ == "__main__":
    unittest.main()
