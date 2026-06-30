"""Tests for packaged performance-comparison demo data accounting guardrails."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
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
from ppar.performance_comparison.transactions import (
    TRANSACTION_CATEGORY_EXTERNAL_FLOW,
    TRANSACTION_CATEGORY_FEE_EXPENSE,
    TRANSACTION_CASH_FLOW_SIGN_NEGATIVE,
    TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL,
    TRANSACTION_PERFORMANCE_FLOW_SIGN_PERFORMANCE,
    TRANSACTION_SEMANTICS_SOURCE_YAML_RULE,
)
from ppar.performance_comparison.workbook_tables import (
    _workbook_context_table,
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
    / "demos"
    / "data"
    / "axys"
    / "ppar_performance_comparison.yaml"
)
_DEMO_SOURCE_CONTRACT_PATH = (
    _REPO_ROOT / "docs" / "performance_comparison_demo_source_contract.md"
)
_PACKAGED_AXYS_DIRECTORY = _REPO_ROOT / "ppar" / "demos" / "data" / "axys"
_DEMO_EXTRACT_AVAILABILITY_PATH = (
    _PACKAGED_AXYS_DIRECTORY / "demo_extract_availability.yaml"
)
_PACKAGED_DEMO_TRANSACTION_CODES = {"by", "sl", "dv", "in", "dp", "li", "wd"}
_TEST_ONLY_TRANSACTION_CODES = {"lo"}
_REAL_WORLD_EVIDENCE_REQUIRED_TRANSACTION_CODES = {";"}

_PERFORMANCE_DIFFERENCE_CAUSE_FIELDS = {
    (pc_cols.HOLDINGS, pc_cols.ACCRUED),
    (pc_cols.HOLDINGS, pc_cols.MARKET_VALUE),
    (pc_cols.HOLDINGS, pc_cols.PRICE),
    (pc_cols.HOLDINGS, pc_cols.QUANTITY),
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

        for expected_text in [
            "normalized demo extracts",
            "not official Axys/APX native schemas",
            "portperf.csv",
            "secperf.csv",
            "CASH_USD",
            "transaction_rules",
        ]:
            self.assertIn(expected_text, normalized_text)

    def test_packaged_demo_extract_availability_covers_current_headers(self) -> None:
        """Every packaged Axys demo CSV field has extraction-confidence metadata."""
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
            for path in (_PACKAGED_AXYS_DIRECTORY / "axys_full_spec_a").glob("*.csv")
        }
        self.assertEqual(set(datasets), expected_files)

        for snapshot_directory in ("axys_full_spec_a", "axys_full_spec_b"):
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

    def test_packaged_demo_extract_availability_appendix_is_current(self) -> None:
        """The human-readable appendix is rendered from the YAML contract."""
        renderer = _load_extract_availability_renderer()

        self.assertEqual(renderer.main(["--check"]), 0)

    def test_packaged_demo_transaction_rules_cover_observed_codes(self) -> None:
        """Packaged demo YAML explicitly defines every observed transaction code."""
        summary = validate_config(_PACKAGED_COMPARISON_PATH)

        self.assertEqual(summary["transaction_codes_without_yaml_rules"], "none")

    def test_packaged_demo_transaction_rules_cover_ambiguous_axys_codes(self) -> None:
        """Packaged demo YAML covers observed and ambiguous Axys source codes."""
        observed_codes: set[str] = set()
        for snapshot_directory in ("axys_full_spec_a", "axys_full_spec_b"):
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
                frame = TransactionsLoader(specification).load(snapshot_key)
                assert frame is not None
                resolved_rows = {
                    str(row[pc_cols.TRANSACTION_ID]): row
                    for row in frame.filter(
                        pl.col(pc_cols.TRANSACTION_ID).is_in(
                            ["ALPHA0203", "INCOME0203"]
                        )
                    ).iter_rows(named=True)
                }

                self.assertEqual(set(resolved_rows), {"ALPHA0203", "INCOME0203"})
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

    def test_packaged_demo_transaction_codes_stay_reviewer_realistic(self) -> None:
        """Packaged transaction rows avoid synthetic semantic edge cases."""
        for snapshot_directory in ("axys_full_spec_a", "axys_full_spec_b"):
            transactions = pd.read_csv(
                _PACKAGED_AXYS_DIRECTORY / snapshot_directory / "transactions.csv"
            )
            observed_codes = set(transactions["TRAN"].astype(str))

            self.assertLessEqual(observed_codes, _PACKAGED_DEMO_TRANSACTION_CODES)
            self.assertTrue(observed_codes.isdisjoint(_TEST_ONLY_TRANSACTION_CODES))

    def test_packaged_demo_has_no_unevidenced_corporate_action_rows(self) -> None:
        """User-facing corporate actions require real-world evidence first."""
        for snapshot_directory in ("axys_full_spec_a", "axys_full_spec_b"):
            transactions = pd.read_csv(
                _PACKAGED_AXYS_DIRECTORY / snapshot_directory / "transactions.csv"
            )
            observed_codes = set(transactions["TRAN"].astype(str))

            self.assertTrue(
                observed_codes.isdisjoint(
                    _REAL_WORLD_EVIDENCE_REQUIRED_TRANSACTION_CODES
                )
            )

    def test_packaged_demo_wd_uses_contextual_external_flow_rule(self) -> None:
        """Packaged Axys wd rows classify external flow from context, not code alone."""
        specification = PerformanceComparisonSpecification(_PACKAGED_COMPARISON_PATH)
        frame = TransactionsLoader(specification).load("a")
        assert frame is not None

        row = frame.filter(pl.col(pc_cols.TRANSACTION_ID) == "ALPHA0203").row(
            0,
            named=True,
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

        row = frame.filter(pl.col(pc_cols.TRANSACTION_ID) == "INCOME0203").row(
            0,
            named=True,
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
        self.assertFalse(
            any(dataset == pc_cols.SECURITY_MASTER for dataset, _field in cause_fields)
        )

    def test_packaged_demo_other_data_differences_match_source_contract(self) -> None:
        """Cost and non-performance evidence stays in Other Data Differences."""
        findings = compare_snapshots(
            _PACKAGED_COMPARISON_PATH,
            require_causal_attribution=True,
        )

        causes = _workbook_underlying_causes_table(findings)
        context = _workbook_context_table(findings)
        cause_fields = _dataset_fields(causes)
        context_fields = _dataset_fields(context)

        self.assertIn((pc_cols.HOLDINGS, pc_cols.COST), context_fields)
        self.assertNotIn((pc_cols.HOLDINGS, pc_cols.COST), cause_fields)
        self.assertFalse(
            any(dataset == pc_cols.SECURITY_MASTER for dataset, _field in context_fields)
        )

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
        self.assertNotIn(
            (pc_cols.HOLDINGS, pc_cols.ACCRUED),
            _dataset_fields(_workbook_context_table(findings)),
        )

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

        self.assertEqual(snapshots["axys_full_spec_a"]["max_holdings_numeric_delta"], 0.0)
        self.assertEqual(snapshots["axys_full_spec_b"]["max_transaction_numeric_delta"], 0.0)
        self.assertFalse(snapshots["axys_full_spec_b"]["has_transaction_field_drift"])
        self.assertEqual(snapshots["axys_full_spec_b"]["max_holdings_numeric_delta"], 0.0)
        self.assertEqual(snapshots["axys_full_spec_b"]["transaction_scenario_rows"], 7)
        self.assertEqual(
            snapshots["axys_full_spec_b"]["transaction_scenarios_by_type"],
            {
                "by": 1,
                "dp": 1,
                "dv": 1,
                "in": 1,
                "li": 1,
                "sl": 1,
                "wd": 1,
            },
        )
        self.assertEqual(snapshots["axys_full_spec_b"]["transaction_derived_holding_rows"], 9)
        self.assertEqual(
            snapshots["axys_full_spec_b"]["transaction_derived_holdings_by_type"],
            {
                "by": 2,
                "dp": 1,
                "dv": 1,
                "in": 1,
                "li": 1,
                "sl": 2,
                "wd": 1,
            },
        )
        self.assertEqual(snapshots["axys_full_spec_b"]["holding_scenario_rows"], 7)
        self.assertEqual(
            snapshots["axys_full_spec_b"]["holding_scenarios_by_type"],
            {
                "accrual_correction": 1,
                "cash_balance_correction": 1,
                "cost_only_correction": 1,
                "quantity_valuation_correction": 1,
                "valuation_mark": 3,
            },
        )
        self.assertFalse(snapshots["axys_full_spec_b"]["has_transaction_drift"])
        self.assertFalse(snapshots["axys_full_spec_b"]["has_holdings_drift"])

    def test_remaining_holding_scenarios_are_classified(self) -> None:
        """Residual holding scenarios have explicit scenario types."""
        rebuild_module = _load_rebuild_module()

        scenarios = rebuild_module._load_holding_scenarios(
            rebuild_module._DEFAULT_HOLDING_SCENARIOS_PATH,
        )
        type_counts = {}
        for adjustment in scenarios.for_snapshot("axys_full_spec_b"):
            type_counts[adjustment.scenario_type] = (
                type_counts.get(adjustment.scenario_type, 0) + 1
            )

        self.assertEqual(
            type_counts,
            {
                "valuation_mark": 3,
                "cash_balance_correction": 1,
                "quantity_valuation_correction": 1,
                "accrual_correction": 1,
                "cost_only_correction": 1,
            },
        )

    def test_scenario_coverage_audit_detects_lost_examples(self) -> None:
        """Scenario coverage guardrail catches missing demo story examples."""
        rebuild_module = _load_rebuild_module()
        summary = {
            "snapshots": [
                {
                    "snapshot": "axys_full_spec_b",
                    "transaction_scenarios_by_type": {"by": 1},
                    "transaction_derived_holdings_by_type": {},
                    "holding_scenarios_by_type": {},
                }
            ]
        }

        issues = rebuild_module._audit_scenario_coverage(summary)

        self.assertGreaterEqual(len(issues), 1)
        self.assertEqual(issues[0].check, "scenario_coverage")

    def test_transaction_scenarios_create_expected_holding_impacts(self) -> None:
        """Transaction changes create the expected cash and security adjustments."""
        rebuild_module = _load_rebuild_module()
        axys_directory = rebuild_module._DEFAULT_AXYS_DIRECTORY
        base_holdings = pd.read_csv(axys_directory / "axys_full_spec_a" / "holdings.csv")
        base_transactions = pd.read_csv(
            axys_directory / "axys_full_spec_a" / "transactions.csv"
        )
        rebuilt_transactions = pd.read_csv(
            axys_directory / "axys_full_spec_b" / "transactions.csv"
        )
        periods = pd.read_csv(axys_directory / "axys_full_spec_b" / "portperf.csv")

        adjustments = rebuild_module._transaction_derived_holding_adjustments(
            "axys_full_spec_b",
            base_holdings=base_holdings,
            base_transactions=base_transactions,
            current_transactions=rebuilt_transactions,
            periods=periods,
        )
        by_scenario = {adjustment.scenario: adjustment for adjustment in adjustments}

        self.assertEqual(len(adjustments), 9)
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
            by_scenario["INCOME0203 dp transaction changes cash balance."],
            portfolio="INCOME",
            security="CASH_USD",
            holding_date="2026-01-30",
            deltas={"QTY": -50.0, "MKT_VAL": -50.0, "COST": -50.0},
        )
        self._assert_adjustment(
            by_scenario["BALANCED0502 dv transaction changes cash balance."],
            portfolio="BALANCED",
            security="CASH_USD",
            holding_date="2026-04-30",
            deltas={"QTY": 117.07, "MKT_VAL": 117.07, "COST": 117.07},
        )
        self._assert_adjustment(
            by_scenario["INCOME0603 in transaction changes cash balance."],
            portfolio="INCOME",
            security="CASH_USD",
            holding_date="2026-05-29",
            deltas={"QTY": 80.0, "MKT_VAL": 80.0, "COST": 80.0},
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
            by_scenario["BALANCED0203 sl transaction changes ending holding."],
            portfolio="BALANCED",
            security="MSFT",
            holding_date="2026-01-30",
            deltas={"QTY": -2.0, "MKT_VAL": -228.0, "COST": -224.58001579710972},
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

    def test_holding_scenario_file_requires_exact_columns(self) -> None:
        """Scenario CSV shape errors fail before any demo files are rebuilt."""
        rebuild_module = _load_rebuild_module()

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bad_scenarios.csv"
            path.write_text("snapshot,PORT\naxys_full_spec_b,ALPHA\n")

            with self.assertRaisesRegex(ValueError, "columns must exactly match"):
                rebuild_module._load_holding_scenarios(path)

    def test_holding_scenario_file_rejects_base_snapshot_adjustments(self) -> None:
        """Scenario rows must target derived snapshots, not the base snapshot."""
        rebuild_module = _load_rebuild_module()
        columns = ",".join(rebuild_module._HOLDING_SCENARIO_COLUMNS)
        row = (
            "axys_full_spec_a,cash_balance_correction,ALPHA,CASH_USD,2026-01-30,"
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
            "axys_full_spec_b,valuation_mark,ALPHA,AAPL,2026-05-29,"
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
            path.write_text("snapshot,TRANSACTION_ID\naxys_full_spec_b,ALPHA0203\n")

            with self.assertRaisesRegex(ValueError, "columns must exactly match"):
                rebuild_module._load_transaction_scenarios(path)

    def test_transaction_scenario_file_rejects_base_snapshot_adjustments(self) -> None:
        """Transaction scenario rows must target derived snapshots."""
        rebuild_module = _load_rebuild_module()
        row = {
            column: ""
            for column in rebuild_module._TRANSACTION_SCENARIO_COLUMNS
        }
        row.update(
            {
                "snapshot": "axys_full_spec_a",
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
        base_holdings = pd.read_csv(axys_directory / "axys_full_spec_a" / "holdings.csv")
        base_transactions = pd.read_csv(
            axys_directory / "axys_full_spec_a" / "transactions.csv"
        )
        periods = pd.read_csv(axys_directory / "axys_full_spec_b" / "portperf.csv")
        scenario_rows = []
        for transaction_id, transaction_code, amount, scenario in (
            ("BALANCED0403", "li", 2500, "Test-only contribution insertion."),
            ("BALANCED0404", "lo", -900, "Test-only withdrawal insertion."),
        ):
            row = {
                column: ""
                for column in rebuild_module._TRANSACTION_SCENARIO_COLUMNS
            }
            row.update(
                {
                    "snapshot": "axys_full_spec_b",
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
            "axys_full_spec_b",
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
            "axys_full_spec_b",
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
        self.assertEqual(adjustment.deltas["PRICE"], 0.0)
        self.assertEqual(adjustment.deltas["ACCRUED"], 0.0)


if __name__ == "__main__":
    unittest.main()
