"""Tests for loading normalized transaction comparison sources."""

# Python imports
import datetime as dt
from pathlib import Path
import tempfile
import unittest

# Third-party imports
import polars as pl
import yaml

# Project imports
from ppar.errors import PpaError
from ppar.audit import (
    AuditSpecification,
    TransactionsLoader,
)
from ppar.audit import schema as pc_cols
from ppar.audit.performance_comparison.backlog_gates import (
    CAPITAL_RETURN_BACKLOG_TRANSACTION_CODES,
    CAPITAL_RETURN_POSSIBLE_ROLES,
    CAPITAL_RETURN_REQUIRED_EVIDENCE,
    SHORT_SIDE_BACKLOG_TRANSACTION_CODES,
    SHORT_SIDE_REQUIRED_EVIDENCE,
    transaction_backlog_gate,
)
from ppar.audit.config_validation import validate_config
from ppar.audit.extract_contract import validate_extract_contract
from ppar.audit.fixed_income import (
    FIXED_INCOME_ACCRUED_INTEREST_TRANSACTION_CODES,
    FIXED_INCOME_BACKLOG_TRANSACTION_CODES,
    FIXED_INCOME_FORMULA_INPUTS,
    FIXED_INCOME_OUT_OF_SCOPE,
    fixed_income_transaction_boundary,
)
from ppar.audit.transactions import (
    TRANSACTION_CASH_FLOW_SIGN_NEGATIVE,
    TRANSACTION_CASH_FLOW_SIGN_NONE,
    TRANSACTION_CASH_FLOW_SIGN_POSITIVE,
    TRANSACTION_CASH_FLOW_SIGN_UNKNOWN,
    TRANSACTION_CATEGORY_BUY,
    TRANSACTION_CATEGORY_CORPORATE_ACTION,
    TRANSACTION_CATEGORY_EXTERNAL_FLOW,
    TRANSACTION_CATEGORY_FEE_EXPENSE,
    TRANSACTION_CATEGORY_INCOME,
    TRANSACTION_CATEGORY_SELL,
    TRANSACTION_CATEGORY_TRANSFER,
    TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL,
    TRANSACTION_PERFORMANCE_FLOW_SIGN_NEUTRAL,
    TRANSACTION_PERFORMANCE_FLOW_SIGN_PERFORMANCE,
    TRANSACTION_PERFORMANCE_FLOW_SIGN_UNKNOWN,
    TRANSACTION_SEMANTICS_SOURCE_MIXED,
    TRANSACTION_SEMANTICS_SOURCE_SOURCE,
    TRANSACTION_SEMANTICS_SOURCE_UNKNOWN,
    TRANSACTION_SEMANTICS_SOURCE_YAML_RULE,
    TransactionCashFlowSign,
    TransactionCategory,
    TransactionPerformanceFlowSign,
    TransactionSemanticsSource,
    normalize_transaction_cash_flow_sign,
    normalize_transaction_category,
    normalize_transaction_performance_flow_sign,
    transaction_category_from_code,
    transaction_impact_semantics_available,
)

_BASELINE_COMPARISON_PATH = Path("tests/data/axys/validation/ppar_audit.yaml")
_SITE_EXTRACT_CONTRACT_TEMPLATE_PATH = Path(
    "docs/axys_apx/contracts/templates/site_extract_contract.yaml"
)
_SITE_EXTRACT_CONTRACT_IMEX_TEMPLATE_PATH = Path(
    "docs/axys_apx/contracts/templates/site_extract_contract_imex_context.yaml"
)
_SITE_EXTRACT_CONTRACT_REP_TEMPLATE_PATH = Path(
    "docs/axys_apx/contracts/templates/site_extract_contract_rep_semantics.yaml"
)
_SITE_VARIANT_FIXTURES_PATH = Path("tests/data/axys/site_variants")


def _write_yaml(directory: Path, contents: object) -> Path:
    """Write comparison YAML contents and return the path."""
    path = directory / "ppar_audit.yaml"
    path.write_text(yaml.safe_dump(contents), encoding="utf-8")
    return path


def _write_extract_contract(
    directory: Path,
    required_columns: list[str],
    *,
    version: int | None = None,
) -> Path:
    """Write a minimal site extract contract and return the path."""
    contract_path = directory / "site_extract_contract.yaml"
    contract: dict[str, object] = {
        "datasets": {
            "transactions.csv": {
                "columns": {
                    column: {
                        "requires_context_for_semantics": True,
                        "blocking_if_missing": True,
                    }
                    for column in required_columns
                }
            }
        }
    }
    if version is not None:
        contract["version"] = version
    contract_path.write_text(yaml.safe_dump(contract), encoding="utf-8")
    return contract_path


def _write_raw_extract_contract(directory: Path, contents: object) -> Path:
    """Write raw extract-contract YAML contents and return the path."""
    contract_path = directory / "site_extract_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contents), encoding="utf-8")
    return contract_path


def _minimal_specification(directory: Path) -> dict[str, object]:
    """Return a minimal valid comparison specification with portfolio files."""
    for snapshot_name in ("snapshot_a", "snapshot_b"):
        snapshot_path = directory / snapshot_name
        snapshot_path.mkdir()
        pl.DataFrame(
            {
                "PORTFOLIO_CODE": ["P1"],
                "FROM_DATE": ["2025-01-01"],
                "THRU_DATE": ["2025-01-31"],
                "PORT_RETURN": [0.01],
            }
        ).write_csv(snapshot_path / "portperf.csv")
    return {
        "snapshots": {
            "a": {"path": "snapshot_a"},
            "b": {"path": "snapshot_b"},
        },
        "files": {"portfolio_performance": "portperf.csv"},
    }


class TestTransactionsLoader(unittest.TestCase):
    """Verify normalized transaction loading for snapshots."""

    def test_site_extract_contract_template_is_valid(self) -> None:
        """The documented site extract-contract starter remains valid."""
        contract = yaml.safe_load(
            _SITE_EXTRACT_CONTRACT_TEMPLATE_PATH.read_text(encoding="utf-8")
        )

        validate_extract_contract(
            contract,
            contract_label=str(_SITE_EXTRACT_CONTRACT_TEMPLATE_PATH),
        )

    def test_site_extract_contract_profile_templates_are_valid(self) -> None:
        """Documented IMEX and REP onboarding contract profiles remain valid."""
        for template_path in (
            _SITE_EXTRACT_CONTRACT_IMEX_TEMPLATE_PATH,
            _SITE_EXTRACT_CONTRACT_REP_TEMPLATE_PATH,
        ):
            contract = yaml.safe_load(template_path.read_text(encoding="utf-8"))

            validate_extract_contract(contract, contract_label=str(template_path))

    def test_site_extract_contract_templates_explain_safe_evidence_paths(self) -> None:
        """Template comments document IMEX-context and REP-semantics options."""
        starter_text = _SITE_EXTRACT_CONTRACT_TEMPLATE_PATH.read_text(encoding="utf-8")
        imex_text = _SITE_EXTRACT_CONTRACT_IMEX_TEMPLATE_PATH.read_text(
            encoding="utf-8"
        )
        rep_text = _SITE_EXTRACT_CONTRACT_REP_TEMPLATE_PATH.read_text(encoding="utf-8")

        self.assertIn("IMEX context fields", starter_text)
        self.assertIn("REP/report/custom-report fields", starter_text)
        self.assertIn("conditional transaction_rules", imex_text)
        self.assertIn("REP/report-semantics profile", imex_text)
        self.assertIn("reviewed category/sign semantics", rep_text)
        self.assertIn("unknown pending review", rep_text)

    def test_site_variant_contracts_match_documented_profiles(self) -> None:
        """Fixture contracts stay aligned with the documented onboarding profiles."""
        expected_pairs = {
            "imex_context": _SITE_EXTRACT_CONTRACT_IMEX_TEMPLATE_PATH,
            "rep_semantics": _SITE_EXTRACT_CONTRACT_REP_TEMPLATE_PATH,
        }
        for fixture_name, template_path in expected_pairs.items():
            fixture_path = (
                _SITE_VARIANT_FIXTURES_PATH
                / fixture_name
                / "site_extract_contract.yaml"
            )

            self.assertEqual(
                yaml.safe_load(fixture_path.read_text(encoding="utf-8")),
                yaml.safe_load(template_path.read_text(encoding="utf-8")),
            )

    def test_public_transaction_constants_match_enum_values(self) -> None:
        """Legacy public string constants stay aligned with transaction enums."""
        self.assertEqual(
            TRANSACTION_CATEGORY_EXTERNAL_FLOW,
            TransactionCategory.EXTERNAL_FLOW.value,
        )
        self.assertEqual(TRANSACTION_CATEGORY_BUY, TransactionCategory.BUY.value)
        self.assertEqual(TRANSACTION_CATEGORY_INCOME, TransactionCategory.INCOME.value)
        self.assertEqual(
            TRANSACTION_CASH_FLOW_SIGN_POSITIVE,
            TransactionCashFlowSign.POSITIVE.value,
        )
        self.assertEqual(
            TRANSACTION_CASH_FLOW_SIGN_NEGATIVE,
            TransactionCashFlowSign.NEGATIVE.value,
        )
        self.assertEqual(TRANSACTION_CASH_FLOW_SIGN_NONE, TransactionCashFlowSign.NONE.value)
        self.assertEqual(
            TRANSACTION_CASH_FLOW_SIGN_UNKNOWN,
            TransactionCashFlowSign.UNKNOWN.value,
        )
        self.assertEqual(
            TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL,
            TransactionPerformanceFlowSign.EXTERNAL.value,
        )
        self.assertEqual(
            TRANSACTION_PERFORMANCE_FLOW_SIGN_PERFORMANCE,
            TransactionPerformanceFlowSign.PERFORMANCE.value,
        )
        self.assertEqual(
            TRANSACTION_PERFORMANCE_FLOW_SIGN_NEUTRAL,
            TransactionPerformanceFlowSign.NEUTRAL.value,
        )
        self.assertEqual(
            TRANSACTION_PERFORMANCE_FLOW_SIGN_UNKNOWN,
            TransactionPerformanceFlowSign.UNKNOWN.value,
        )
        self.assertEqual(
            TRANSACTION_SEMANTICS_SOURCE_SOURCE,
            TransactionSemanticsSource.SOURCE.value,
        )
        self.assertEqual(
            TRANSACTION_SEMANTICS_SOURCE_YAML_RULE,
            TransactionSemanticsSource.YAML_RULE.value,
        )
        self.assertEqual(
            TRANSACTION_SEMANTICS_SOURCE_MIXED,
            TransactionSemanticsSource.MIXED.value,
        )
        self.assertEqual(
            TRANSACTION_SEMANTICS_SOURCE_UNKNOWN,
            TransactionSemanticsSource.UNKNOWN.value,
        )

    def test_load_baseline_snapshot_a_transactions(self) -> None:
        """Transaction rows load with normalized internal columns."""
        specification = AuditSpecification(_BASELINE_COMPARISON_PATH)
        frame = TransactionsLoader(specification).load("a")
        assert frame is not None

        self.assertTrue(
            set(pc_cols.TRANSACTIONS_REQUIRED_COLUMNS).issubset(frame.columns)
        )
        self.assertIn(pc_cols.SETTLEMENT_DATE, frame.columns)
        self.assertIn(pc_cols.TRANSACTION_CODE, frame.columns)
        self.assertIn(pc_cols.TRANSACTION_CATEGORY, frame.columns)
        self.assertIn(pc_cols.BROKER, frame.columns)
        self.assertEqual(frame.schema[pc_cols.TRANSACTION_DATE], pl.Date)
        self.assertEqual(frame.schema[pc_cols.SETTLEMENT_DATE], pl.Date)

        target_row = frame.filter(
            (pl.col(pc_cols.PORTFOLIO_ID) == "PORT_A")
            & (pl.col(pc_cols.SECURITY_ID) == "AAPL")
            & (pl.col(pc_cols.TRANSACTION_DATE) == pl.date(2025, 5, 1))
        ).row(0, named=True)
        self.assertEqual(target_row[pc_cols.TRANSACTION_CODE], "BUY")
        self.assertEqual(target_row[pc_cols.TRANSACTION_CATEGORY], TRANSACTION_CATEGORY_BUY)
        self.assertEqual(target_row[pc_cols.QUANTITY], 200.0)
        self.assertEqual(target_row[pc_cols.BROKER], "INIT")

    def test_original_cost_aliases_load_as_typed_optional_evidence(self) -> None:
        """Original-cost aliases normalize to numeric amount and date fields."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "transactions": "transactions.csv",
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(
                    {
                        "PORT": ["P1"],
                        "SEC": ["ABC"],
                        "TRANSACTION_DATE": ["2025-01-15"],
                        "TRAN": ["by"],
                        "ORIG_COST_DATE": ["2020-04-03"],
                        "ORIG_COST": [0.0],
                    }
                ).write_csv(directory / snapshot_name / "transactions.csv")
            path = _write_yaml(directory, configuration)

            frame = TransactionsLoader(AuditSpecification(path)).load("a")
            assert frame is not None
            row = frame.row(0, named=True)

        self.assertEqual(row[pc_cols.ORIGINAL_COST_DATE], dt.date(2020, 4, 3))
        self.assertEqual(row[pc_cols.ORIGINAL_COST], 0.0)

    def test_conflicting_transaction_base_currency_raises_error_504(self) -> None:
        """Transaction currency cannot contradict its portfolio currency."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "transactions": "transactions.csv",
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                snapshot_path = directory / snapshot_name
                pl.DataFrame(
                    {
                        "PORTFOLIO_CODE": ["P1"],
                        "FROM_DATE": ["2025-01-01"],
                        "THRU_DATE": ["2025-01-31"],
                        "PORT_RETURN": [0.01],
                        "BASE_CURRENCY": ["USD"],
                    }
                ).write_csv(snapshot_path / "portperf.csv")
                pl.DataFrame(
                    {
                        "PORT": ["P1"],
                        "TRANSACTION_DATE": ["2025-01-15"],
                        "SEC": ["S1"],
                        "TRAN": ["by"],
                        "BASE_CURRENCY": ["EUR"],
                        "AMOUNT": [-100.0],
                    }
                ).write_csv(snapshot_path / "transactions.csv")
            path = _write_yaml(directory, configuration)

            with self.assertRaises(PpaError) as context:
                TransactionsLoader(AuditSpecification(path)).load("a")

        message = str(context.exception)
        self.assertTrue(message.startswith("Error 504"))
        self.assertIn("authoritative portfolio_performance", message)

    def test_transaction_category_is_inferred_from_transaction_code(self) -> None:
        """Transaction codes are labeled with conservative normalized categories."""
        specification = AuditSpecification(_BASELINE_COMPARISON_PATH)
        frame = TransactionsLoader(specification).load("a")
        assert frame is not None

        dividend_row = frame.filter(pl.col(pc_cols.TRANSACTION_CODE) == "DIV").row(
            0,
            named=True,
        )
        interest_row = frame.filter(pl.col(pc_cols.TRANSACTION_CODE) == "INT").row(
            0,
            named=True,
        )

        self.assertEqual(dividend_row[pc_cols.TRANSACTION_CATEGORY], TRANSACTION_CATEGORY_INCOME)
        self.assertEqual(interest_row[pc_cols.TRANSACTION_CATEGORY], TRANSACTION_CATEGORY_INCOME)

    def test_normalize_transaction_category_handles_known_labels(self) -> None:
        """Source category labels normalize to the documented category vocabulary."""
        self.assertEqual(normalize_transaction_category("Cash Deposit"), "external_flow")
        self.assertEqual(normalize_transaction_category("fee-expense"), "fee_expense")
        self.assertEqual(normalize_transaction_category(""), "unknown")
        self.assertEqual(transaction_category_from_code("by"), "buy")
        self.assertEqual(transaction_category_from_code("sl"), "sell")
        self.assertEqual(transaction_category_from_code("dv"), "income")
        self.assertEqual(transaction_category_from_code("in"), "income")
        self.assertEqual(transaction_category_from_code("wd"), "unknown")
        self.assertEqual(transaction_category_from_code("li"), "unknown")
        self.assertEqual(transaction_category_from_code("lo"), "unknown")
        self.assertEqual(transaction_category_from_code("dp"), "unknown")
        self.assertEqual(transaction_category_from_code("ai"), "unknown")
        self.assertEqual(transaction_category_from_code("ti"), "unknown")
        self.assertEqual(transaction_category_from_code("pa"), "unknown")
        self.assertEqual(transaction_category_from_code("sa"), "unknown")
        self.assertEqual(transaction_category_from_code("pd"), "unknown")
        self.assertEqual(transaction_category_from_code("rc"), "unknown")
        self.assertEqual(transaction_category_from_code("ss"), "unknown")
        self.assertEqual(transaction_category_from_code("cs"), "unknown")
        self.assertEqual(transaction_category_from_code("SELL"), "sell")
        self.assertEqual(transaction_category_from_code("not-a-real-code"), "unknown")
        self.assertEqual(
            transaction_category_from_code("BY", exact_case=True),
            "unknown",
        )

    def test_fixed_income_transaction_boundary_is_modified_dietz_scoped(self) -> None:
        """Fixed-income helper names safe formula inputs and blocked backlog codes."""
        self.assertEqual(fixed_income_transaction_boundary("in"), "safe_income")
        for code in FIXED_INCOME_ACCRUED_INTEREST_TRANSACTION_CODES:
            with self.subTest(code=code):
                self.assertEqual(
                    fixed_income_transaction_boundary(code),
                    "accrued_interest_adjunct",
                )
                self.assertEqual(transaction_category_from_code(code), "unknown")
        for code in FIXED_INCOME_BACKLOG_TRANSACTION_CODES:
            with self.subTest(code=code):
                self.assertEqual(fixed_income_transaction_boundary(code), "backlog")
                self.assertEqual(transaction_category_from_code(code), "unknown")

        self.assertIn(
            "ordinary interest transaction amounts",
            FIXED_INCOME_FORMULA_INPUTS,
        )
        self.assertIn(
            "configured holdings.accrued changes",
            FIXED_INCOME_FORMULA_INPUTS,
        )
        self.assertIn("amortization/accretion engine", FIXED_INCOME_OUT_OF_SCOPE)
        self.assertIn("yield calculation", FIXED_INCOME_OUT_OF_SCOPE)
        self.assertEqual(
            fixed_income_transaction_boundary("IN", exact_case=True),
            "not_fixed_income_boundary",
        )

    def test_high_risk_backlog_gates_are_code_only_boundaries(self) -> None:
        """Capital-return and short-side codes stay gated without context."""
        for code in CAPITAL_RETURN_BACKLOG_TRANSACTION_CODES:
            with self.subTest(code=code):
                self.assertEqual(transaction_backlog_gate(code), "capital_return_policy")
                self.assertEqual(transaction_category_from_code(code), "unknown")

        for code in SHORT_SIDE_BACKLOG_TRANSACTION_CODES:
            with self.subTest(code=code):
                self.assertEqual(transaction_backlog_gate(code), "short_side_evidence")
                self.assertEqual(transaction_category_from_code(code), "unknown")

        self.assertIn("performance income", CAPITAL_RETURN_POSSIBLE_ROLES)
        self.assertIn("corporate-action evidence", CAPITAL_RETURN_POSSIBLE_ROLES)
        self.assertIn("review-only evidence", CAPITAL_RETURN_POSSIBLE_ROLES)
        self.assertIn("cost-basis or principal context", CAPITAL_RETURN_REQUIRED_EVIDENCE)
        self.assertIn("short security type", SHORT_SIDE_REQUIRED_EVIDENCE)
        self.assertIn("amount and quantity signs", SHORT_SIDE_REQUIRED_EVIDENCE)
        self.assertEqual(
            transaction_backlog_gate("RC", exact_case=True),
            "not_backlog_gate",
        )

    def test_transaction_sign_semantics_use_documented_vocabulary(self) -> None:
        """Transaction sign semantics normalize only recognized source labels."""
        self.assertEqual(
            normalize_transaction_cash_flow_sign("cash in"),
            TRANSACTION_CASH_FLOW_SIGN_POSITIVE,
        )
        self.assertEqual(
            normalize_transaction_cash_flow_sign("withdrawal"),
            TRANSACTION_CASH_FLOW_SIGN_NEGATIVE,
        )
        self.assertEqual(
            normalize_transaction_cash_flow_sign("neutral"),
            TRANSACTION_CASH_FLOW_SIGN_NONE,
        )
        self.assertEqual(
            normalize_transaction_cash_flow_sign("not-a-real-sign"),
            TRANSACTION_CASH_FLOW_SIGN_UNKNOWN,
        )
        self.assertEqual(
            normalize_transaction_performance_flow_sign("external flow"),
            TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL,
        )
        self.assertEqual(
            normalize_transaction_performance_flow_sign("included"),
            TRANSACTION_PERFORMANCE_FLOW_SIGN_PERFORMANCE,
        )
        self.assertEqual(
            normalize_transaction_performance_flow_sign("none"),
            TRANSACTION_PERFORMANCE_FLOW_SIGN_NEUTRAL,
        )
        self.assertEqual(
            normalize_transaction_performance_flow_sign("not-a-real-treatment"),
            TRANSACTION_PERFORMANCE_FLOW_SIGN_UNKNOWN,
        )

    def test_transaction_impact_semantics_available_requires_both_signs(self) -> None:
        """Future transaction impact estimates require both semantic fields."""
        self.assertTrue(
            transaction_impact_semantics_available(
                {
                    pc_cols.CASH_FLOW_SIGN: TRANSACTION_CASH_FLOW_SIGN_POSITIVE,
                    pc_cols.PERFORMANCE_FLOW_SIGN: (
                        TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL
                    ),
                }
            )
        )
        self.assertFalse(
            transaction_impact_semantics_available(
                {
                    pc_cols.CASH_FLOW_SIGN: TRANSACTION_CASH_FLOW_SIGN_POSITIVE,
                    pc_cols.PERFORMANCE_FLOW_SIGN: (
                        TRANSACTION_PERFORMANCE_FLOW_SIGN_UNKNOWN
                    ),
                }
            )
        )
        self.assertFalse(
            transaction_impact_semantics_available(
                {pc_cols.CASH_FLOW_SIGN: TRANSACTION_CASH_FLOW_SIGN_POSITIVE}
            )
        )

    def test_source_transaction_category_and_sign_columns_are_loaded(self) -> None:
        """Optional source category and sign columns load without inferred signs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "transactions": "transactions.csv",
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(
                    {
                        "PORT": ["P1"],
                        "SEC": ["S1"],
                        "TRANSACTION_DATE": ["2025-01-31"],
                        "ACTIVITY_CATEGORY": ["cash deposit"],
                        "CASH_FLOW_SIGN": ["positive"],
                        "PERFORMANCE_FLOW_SIGN": ["external"],
                    }
                ).write_csv(directory / snapshot_name / "transactions.csv")
            path = _write_yaml(directory, configuration)
            specification = AuditSpecification(path)

            frame = TransactionsLoader(specification).load("a")
            assert frame is not None
            row = frame.row(0, named=True)

            self.assertEqual(
                row[pc_cols.TRANSACTION_CATEGORY],
                TRANSACTION_CATEGORY_EXTERNAL_FLOW,
            )
            self.assertEqual(row[pc_cols.CASH_FLOW_SIGN], "positive")
            self.assertEqual(row[pc_cols.PERFORMANCE_FLOW_SIGN], "external")
            self.assertEqual(
                row[pc_cols.TRANSACTION_SEMANTICS_SOURCE],
                TRANSACTION_SEMANTICS_SOURCE_SOURCE,
            )
            self.assertTrue(transaction_impact_semantics_available(row))

    def test_unknown_transaction_sign_columns_remain_unknown(self) -> None:
        """Unrecognized sign semantics remain unavailable for impact estimates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "transactions": "transactions.csv",
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(
                    {
                        "PORT": ["P1"],
                        "SEC": ["S1"],
                        "TRANSACTION_DATE": ["2025-01-31"],
                        "ACTIVITY_CATEGORY": ["cash deposit"],
                        "CASH_FLOW_SIGN": ["vendor-only"],
                        "PERFORMANCE_FLOW_SIGN": ["mystery"],
                    }
                ).write_csv(directory / snapshot_name / "transactions.csv")
            path = _write_yaml(directory, configuration)
            specification = AuditSpecification(path)

            frame = TransactionsLoader(specification).load("a")
            assert frame is not None
            row = frame.row(0, named=True)

            self.assertEqual(
                row[pc_cols.CASH_FLOW_SIGN],
                TRANSACTION_CASH_FLOW_SIGN_UNKNOWN,
            )
            self.assertEqual(
                row[pc_cols.PERFORMANCE_FLOW_SIGN],
                TRANSACTION_PERFORMANCE_FLOW_SIGN_UNKNOWN,
            )
            self.assertEqual(
                row[pc_cols.TRANSACTION_SEMANTICS_SOURCE],
                TRANSACTION_SEMANTICS_SOURCE_UNKNOWN,
            )
            self.assertFalse(transaction_impact_semantics_available(row))

    def test_unknown_transaction_code_without_rule_raises_error(self) -> None:
        """Unknown transaction codes require YAML rules or source semantics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "transactions": "transactions.csv",
            }
            configuration["transaction_rules"] = {
                "BUY": {
                    "transaction_category": "buy",
                    "cash_flow_sign": "negative",
                    "performance_flow_sign": "performance",
                }
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(
                    {
                        "PORT": ["P1"],
                        "SEC": ["S1"],
                        "TRANSACTION_DATE": ["2025-01-31"],
                        "TRAN": ["MYSTERY"],
                    }
                ).write_csv(directory / snapshot_name / "transactions.csv")
            path = _write_yaml(directory, configuration)
            specification = AuditSpecification(path)

            with self.assertRaises(PpaError) as context:
                TransactionsLoader(specification).load("a")

            message = str(context.exception)
            self.assertTrue(message.startswith("Error 504"))
            self.assertIn("unknown transaction codes or categories", message)
            self.assertIn("transaction_code=MYSTERY", message)

    def test_yaml_transaction_rules_define_semantics(self) -> None:
        """YAML transaction rules define category and sign semantics by code."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "transactions": "transactions.csv",
            }
            configuration["transaction_rules"] = {
                "DEP": {
                    "transaction_category": "external_flow",
                    "cash_flow_sign": "positive",
                    "performance_flow_sign": "external",
                }
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(
                    {
                        "PORT": ["P1"],
                        "SEC": ["S1"],
                        "TRANSACTION_DATE": ["2025-01-31"],
                        "TRAN": ["DEP"],
                    }
                ).write_csv(directory / snapshot_name / "transactions.csv")
            path = _write_yaml(directory, configuration)
            specification = AuditSpecification(path)

            frame = TransactionsLoader(specification).load("a")
            assert frame is not None
            row = frame.row(0, named=True)

            self.assertEqual(
                row[pc_cols.TRANSACTION_CATEGORY],
                TRANSACTION_CATEGORY_EXTERNAL_FLOW,
            )
            self.assertEqual(row[pc_cols.CASH_FLOW_SIGN], "positive")
            self.assertEqual(row[pc_cols.PERFORMANCE_FLOW_SIGN], "external")
            self.assertEqual(
                row[pc_cols.TRANSACTION_SEMANTICS_SOURCE],
                TRANSACTION_SEMANTICS_SOURCE_YAML_RULE,
            )
            self.assertTrue(transaction_impact_semantics_available(row))

    def test_yaml_transaction_rules_override_source_semantics(self) -> None:
        """Matching YAML semantics are authoritative over source labels."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "transactions": "transactions.csv",
            }
            configuration["transaction_rules"] = {
                "BUY": {
                    "transaction_category": "external_flow",
                    "cash_flow_sign": "positive",
                    "performance_flow_sign": "external",
                }
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(
                    {
                        "PORT": ["P1"],
                        "SEC": ["S1"],
                        "TRANSACTION_DATE": ["2025-01-31"],
                        "TRAN": ["BUY"],
                        "CASH_FLOW_SIGN": ["negative"],
                        "PERFORMANCE_FLOW_SIGN": ["performance"],
                    }
                ).write_csv(directory / snapshot_name / "transactions.csv")
            path = _write_yaml(directory, configuration)
            specification = AuditSpecification(path)

            frame = TransactionsLoader(specification).load("a")
            assert frame is not None
            row = frame.row(0, named=True)

            self.assertEqual(
                row[pc_cols.TRANSACTION_CATEGORY],
                TRANSACTION_CATEGORY_EXTERNAL_FLOW,
            )
            self.assertEqual(row[pc_cols.CASH_FLOW_SIGN], "positive")
            self.assertEqual(row[pc_cols.PERFORMANCE_FLOW_SIGN], "external")
            self.assertEqual(
                row[pc_cols.TRANSACTION_SEMANTICS_SOURCE],
                TRANSACTION_SEMANTICS_SOURCE_YAML_RULE,
            )

    def test_yaml_transaction_rules_mark_rule_only_semantics(self) -> None:
        """YAML rules are tagged as the sole source when no source semantics exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "transactions": "transactions.csv",
            }
            configuration["transaction_rules"] = {
                "CUSTOM": {
                    "transaction_category": "external_flow",
                    "cash_flow_sign": "positive",
                    "performance_flow_sign": "external",
                }
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(
                    {
                        "PORT": ["P1"],
                        "SEC": ["S1"],
                        "TRANSACTION_DATE": ["2025-01-31"],
                        "TRAN": ["CUSTOM"],
                    }
                ).write_csv(directory / snapshot_name / "transactions.csv")
            path = _write_yaml(directory, configuration)
            specification = AuditSpecification(path)

            frame = TransactionsLoader(specification).load("a")
            assert frame is not None
            row = frame.row(0, named=True)

            self.assertEqual(
                row[pc_cols.TRANSACTION_SEMANTICS_SOURCE],
                TRANSACTION_SEMANTICS_SOURCE_YAML_RULE,
            )

    def test_complete_yaml_transaction_rule_is_authoritative(self) -> None:
        """A complete matching YAML rule is the sole semantic authority."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "transactions": "transactions.csv",
            }
            configuration["transaction_rules"] = {
                "BUY": {
                    "transaction_category": "buy",
                    "cash_flow_sign": "negative",
                    "performance_flow_sign": "performance",
                }
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(
                    {
                        "PORT": ["P1"],
                        "SEC": ["S1"],
                        "TRANSACTION_DATE": ["2025-01-31"],
                        "TRAN": ["BUY"],
                        "CASH_FLOW_SIGN": ["negative"],
                    }
                ).write_csv(directory / snapshot_name / "transactions.csv")
            path = _write_yaml(directory, configuration)
            specification = AuditSpecification(path)

            frame = TransactionsLoader(specification).load("a")
            assert frame is not None
            row = frame.row(0, named=True)

            self.assertEqual(row[pc_cols.TRANSACTION_CATEGORY], TRANSACTION_CATEGORY_BUY)
            self.assertEqual(row[pc_cols.CASH_FLOW_SIGN], "negative")
            self.assertEqual(row[pc_cols.PERFORMANCE_FLOW_SIGN], "performance")
            self.assertEqual(
                row[pc_cols.TRANSACTION_SEMANTICS_SOURCE],
                TRANSACTION_SEMANTICS_SOURCE_YAML_RULE,
            )

    def test_partial_yaml_transaction_rule_overrides_only_defined_fields(self) -> None:
        """A partial matching YAML rule combines with remaining source fields."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "transactions": "transactions.csv",
            }
            configuration["transaction_rules"] = {
                "BUY": {"cash_flow_sign": "positive"}
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(
                    {
                        "PORT": ["P1"],
                        "SEC": ["S1"],
                        "TRANSACTION_DATE": ["2025-01-31"],
                        "TRAN": ["BUY"],
                        "CASH_FLOW_SIGN": ["negative"],
                        "PERFORMANCE_FLOW_SIGN": ["performance"],
                    }
                ).write_csv(directory / snapshot_name / "transactions.csv")
            path = _write_yaml(directory, configuration)
            specification = AuditSpecification(path)

            frame = TransactionsLoader(specification).load("a")
            assert frame is not None
            row = frame.row(0, named=True)

            self.assertEqual(row[pc_cols.TRANSACTION_CATEGORY], TRANSACTION_CATEGORY_BUY)
            self.assertEqual(row[pc_cols.CASH_FLOW_SIGN], "positive")
            self.assertEqual(row[pc_cols.PERFORMANCE_FLOW_SIGN], "performance")
            self.assertEqual(
                row[pc_cols.TRANSACTION_SEMANTICS_SOURCE],
                TRANSACTION_SEMANTICS_SOURCE_MIXED,
            )

    def test_conditional_yaml_rules_classify_axys_external_flow_codes(self) -> None:
        """Conditional rules distinguish ambiguous Axys codes using IMEX context."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "transactions": "transactions.csv",
            }
            configuration["transaction_rules"] = {
                "wd": [
                    {
                        "when": {
                            "security_id": "CASHUSD",
                            "source_destination_type": "$pty",
                            "source_destination_symbol": "$cash",
                        },
                        "transaction_category": "external_flow",
                        "cash_flow_sign": "negative",
                        "performance_flow_sign": "external",
                    },
                    {
                        "when": {"source_destination_type": "$sweep"},
                        "transaction_category": "transfer",
                        "cash_flow_sign": "none",
                        "performance_flow_sign": "neutral",
                    },
                ],
                "li": [
                    {
                        "when": {"source_destination_type": "$pty"},
                        "transaction_category": "external_flow",
                        "cash_flow_sign": "positive",
                        "performance_flow_sign": "external",
                    }
                ],
                "lo": [
                    {
                        "when": {"source_destination_type": "$pty"},
                        "transaction_category": "external_flow",
                        "cash_flow_sign": "negative",
                        "performance_flow_sign": "external",
                    }
                ],
                "dp": [
                    {
                        "when": {
                            "special_security_type": "exus",
                            "special_security_symbol": "custfee",
                        },
                        "transaction_category": "fee_expense",
                        "cash_flow_sign": "negative",
                        "performance_flow_sign": "performance",
                    }
                ],
            }
            rows = {
                "PORT": ["P1", "P1", "P1", "P1", "P1"],
                "SEC": ["CASHUSD", "CASHUSD", "AAPL", "AAPL", "CASHUSD"],
                "TRANSACTION_DATE": ["2025-01-31"] * 5,
                "TRAN": ["wd", "wd", "li", "lo", "dp"],
                "SEC_TYPE": ["cash", "cash", "eq", "eq", "cash"],
                "SRC_DEST_TYPE": ["$pty", "$sweep", "$pty", "$pty", "$pty"],
                "SRC_DEST_SYMBOL": ["$cash", "CASHUSD", "$cash", "$cash", "$cash"],
                "SPECIAL_SEC_TYPE": ["", "", "", "", "exus"],
                "SPECIAL_SEC_SYMBOL": ["", "", "", "", "custfee"],
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(rows).write_csv(directory / snapshot_name / "transactions.csv")
            path = _write_yaml(directory, configuration)
            specification = AuditSpecification(path)

            frame = TransactionsLoader(specification).load("a")
            assert frame is not None
            categories = frame.select(
                pc_cols.TRANSACTION_CODE,
                pc_cols.TRANSACTION_CATEGORY,
                pc_cols.CASH_FLOW_SIGN,
                pc_cols.PERFORMANCE_FLOW_SIGN,
                pc_cols.TRANSACTION_SEMANTICS_SOURCE,
            ).to_dicts()

            self.assertEqual(
                [row[pc_cols.TRANSACTION_CATEGORY] for row in categories],
                [
                    TRANSACTION_CATEGORY_EXTERNAL_FLOW,
                    TRANSACTION_CATEGORY_TRANSFER,
                    TRANSACTION_CATEGORY_EXTERNAL_FLOW,
                    TRANSACTION_CATEGORY_EXTERNAL_FLOW,
                    TRANSACTION_CATEGORY_FEE_EXPENSE,
                ],
            )
            self.assertEqual(
                [row[pc_cols.PERFORMANCE_FLOW_SIGN] for row in categories],
                [
                    TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL,
                    TRANSACTION_PERFORMANCE_FLOW_SIGN_NEUTRAL,
                    TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL,
                    TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL,
                    TRANSACTION_PERFORMANCE_FLOW_SIGN_PERFORMANCE,
                ],
            )
            self.assertTrue(
                all(
                    row[pc_cols.TRANSACTION_SEMANTICS_SOURCE]
                    == TRANSACTION_SEMANTICS_SOURCE_YAML_RULE
                    for row in categories
                )
            )

    def test_site_variant_imex_context_classifies_ambiguous_axys_codes(self) -> None:
        """Fixture-backed IMEX context rules classify every ambiguous Axys code."""
        specification = AuditSpecification(
            _SITE_VARIANT_FIXTURES_PATH
            / "imex_context"
            / "ppar_audit.yaml"
        )

        frame = TransactionsLoader(specification).load("a")
        assert frame is not None

        categories_by_code = {
            code[0]: sorted(
                set(rows.get_column(pc_cols.TRANSACTION_CATEGORY).to_list())
            )
            for code, rows in frame.group_by(pc_cols.TRANSACTION_CODE)
        }
        flow_signs_by_code = {
            code[0]: sorted(
                set(rows.get_column(pc_cols.PERFORMANCE_FLOW_SIGN).to_list())
            )
            for code, rows in frame.group_by(pc_cols.TRANSACTION_CODE)
        }

        self.assertEqual(
            categories_by_code,
            {
                "dp": [TRANSACTION_CATEGORY_FEE_EXPENSE, TRANSACTION_CATEGORY_TRANSFER],
                "li": [
                    TRANSACTION_CATEGORY_EXTERNAL_FLOW,
                    TRANSACTION_CATEGORY_TRANSFER,
                ],
                "lo": [
                    TRANSACTION_CATEGORY_EXTERNAL_FLOW,
                    TRANSACTION_CATEGORY_TRANSFER,
                ],
                "wd": [
                    TRANSACTION_CATEGORY_EXTERNAL_FLOW,
                    TRANSACTION_CATEGORY_TRANSFER,
                ],
            },
        )
        self.assertEqual(
            flow_signs_by_code,
            {
                "dp": [
                    TRANSACTION_PERFORMANCE_FLOW_SIGN_NEUTRAL,
                    TRANSACTION_PERFORMANCE_FLOW_SIGN_PERFORMANCE,
                ],
                "li": [
                    TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL,
                    TRANSACTION_PERFORMANCE_FLOW_SIGN_NEUTRAL,
                ],
                "lo": [
                    TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL,
                    TRANSACTION_PERFORMANCE_FLOW_SIGN_NEUTRAL,
                ],
                "wd": [
                    TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL,
                    TRANSACTION_PERFORMANCE_FLOW_SIGN_NEUTRAL,
                ],
            },
        )
        self.assertEqual(
            set(frame.get_column(pc_cols.TRANSACTION_SEMANTICS_SOURCE).to_list()),
            {TRANSACTION_SEMANTICS_SOURCE_YAML_RULE},
        )
        expected_rows = {
            ("li", "$pty", "$cash", "", ""): (
                TRANSACTION_CATEGORY_EXTERNAL_FLOW,
                TRANSACTION_CASH_FLOW_SIGN_POSITIVE,
                TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL,
            ),
            ("li", "$acct", "INTERNAL_ACCT", "", ""): (
                TRANSACTION_CATEGORY_TRANSFER,
                TRANSACTION_CASH_FLOW_SIGN_NONE,
                TRANSACTION_PERFORMANCE_FLOW_SIGN_NEUTRAL,
            ),
            ("lo", "$pty", "$cash", "", ""): (
                TRANSACTION_CATEGORY_EXTERNAL_FLOW,
                TRANSACTION_CASH_FLOW_SIGN_NEGATIVE,
                TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL,
            ),
            ("lo", "$acct", "INTERNAL_ACCT", "", ""): (
                TRANSACTION_CATEGORY_TRANSFER,
                TRANSACTION_CASH_FLOW_SIGN_NONE,
                TRANSACTION_PERFORMANCE_FLOW_SIGN_NEUTRAL,
            ),
            ("dp", "$pty", "$cash", "exus", "custfee"): (
                TRANSACTION_CATEGORY_FEE_EXPENSE,
                TRANSACTION_CASH_FLOW_SIGN_NEGATIVE,
                TRANSACTION_PERFORMANCE_FLOW_SIGN_PERFORMANCE,
            ),
            ("dp", "$sweep", "MMF", "", ""): (
                TRANSACTION_CATEGORY_TRANSFER,
                TRANSACTION_CASH_FLOW_SIGN_NONE,
                TRANSACTION_PERFORMANCE_FLOW_SIGN_NEUTRAL,
            ),
            ("wd", "$pty", "$cash", "", ""): (
                TRANSACTION_CATEGORY_EXTERNAL_FLOW,
                TRANSACTION_CASH_FLOW_SIGN_NEGATIVE,
                TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL,
            ),
            ("wd", "$sweep", "MMF", "", ""): (
                TRANSACTION_CATEGORY_TRANSFER,
                TRANSACTION_CASH_FLOW_SIGN_NONE,
                TRANSACTION_PERFORMANCE_FLOW_SIGN_NEUTRAL,
            ),
        }
        actual_rows = {}
        for row in frame.iter_rows(named=True):
            key = (
                row[pc_cols.TRANSACTION_CODE],
                row[pc_cols.SOURCE_DESTINATION_TYPE] or "",
                row[pc_cols.SOURCE_DESTINATION_SYMBOL] or "",
                row[pc_cols.SPECIAL_SECURITY_TYPE] or "",
                row[pc_cols.SPECIAL_SECURITY_SYMBOL] or "",
            )
            actual_rows[key] = (
                row[pc_cols.TRANSACTION_CATEGORY],
                row[pc_cols.CASH_FLOW_SIGN],
                row[pc_cols.PERFORMANCE_FLOW_SIGN],
            )
        self.assertEqual(actual_rows, expected_rows)

    def test_site_variant_exact_case_rules_keep_codes_distinct(self) -> None:
        """The focused site fixture proves exact code and context matching."""
        path = _SITE_VARIANT_FIXTURES_PATH / "exact_case_rules" / "ppar_audit.yaml"
        frame = TransactionsLoader(AuditSpecification(path)).load("a")
        assert frame is not None

        rows = frame.sort(pc_cols.SECURITY_ID).to_dicts()
        self.assertEqual(
            [row[pc_cols.TRANSACTION_CODE] for row in rows],
            ["by", "BY"],
        )
        self.assertEqual(
            [row[pc_cols.TRANSACTION_CATEGORY] for row in rows],
            [TRANSACTION_CATEGORY_BUY, TRANSACTION_CATEGORY_SELL],
        )
        self.assertEqual(
            validate_config(path, require_complete_yaml_setup=False)[
                "transaction_codes_without_yaml_rules"
            ],
            "none",
        )

    def test_site_variant_fixed_income_accruals_use_explicit_rules(self) -> None:
        """Fixed-income accrued-interest codes stay YAML-scoped, not built-in."""
        specification = AuditSpecification(
            _SITE_VARIANT_FIXTURES_PATH
            / "fixed_income_accruals"
            / "ppar_audit.yaml"
        )

        frame = TransactionsLoader(specification).load("a")
        assert frame is not None

        self.assertEqual(transaction_category_from_code("pa"), "unknown")
        self.assertEqual(transaction_category_from_code("sa"), "unknown")
        self.assertEqual(
            frame.sort(pc_cols.TRANSACTION_CODE)
            .select(
                pc_cols.TRANSACTION_CODE,
                pc_cols.SECURITY_TYPE,
                pc_cols.TRANSACTION_CATEGORY,
                pc_cols.CASH_FLOW_SIGN,
                pc_cols.PERFORMANCE_FLOW_SIGN,
                pc_cols.TRANSACTION_SEMANTICS_SOURCE,
            )
            .to_dicts(),
            [
                {
                    pc_cols.TRANSACTION_CODE: "pa",
                    pc_cols.SECURITY_TYPE: "bond",
                    pc_cols.TRANSACTION_CATEGORY: TRANSACTION_CATEGORY_FEE_EXPENSE,
                    pc_cols.CASH_FLOW_SIGN: TRANSACTION_CASH_FLOW_SIGN_NEGATIVE,
                    pc_cols.PERFORMANCE_FLOW_SIGN: (
                        TRANSACTION_PERFORMANCE_FLOW_SIGN_PERFORMANCE
                    ),
                    pc_cols.TRANSACTION_SEMANTICS_SOURCE: (
                        TRANSACTION_SEMANTICS_SOURCE_YAML_RULE
                    ),
                },
                {
                    pc_cols.TRANSACTION_CODE: "sa",
                    pc_cols.SECURITY_TYPE: "bond",
                    pc_cols.TRANSACTION_CATEGORY: TRANSACTION_CATEGORY_INCOME,
                    pc_cols.CASH_FLOW_SIGN: TRANSACTION_CASH_FLOW_SIGN_POSITIVE,
                    pc_cols.PERFORMANCE_FLOW_SIGN: (
                        TRANSACTION_PERFORMANCE_FLOW_SIGN_PERFORMANCE
                    ),
                    pc_cols.TRANSACTION_SEMANTICS_SOURCE: (
                        TRANSACTION_SEMANTICS_SOURCE_YAML_RULE
                    ),
                },
            ],
        )

    def test_site_variant_ai_margin_interest_uses_explicit_rules(self) -> None:
        """Margin-style ``ai`` rows stay YAML-scoped, not built-in."""
        specification = AuditSpecification(
            _SITE_VARIANT_FIXTURES_PATH
            / "ai_margin_interest"
            / "ppar_audit.yaml"
        )

        frame = TransactionsLoader(specification).load("a")
        assert frame is not None

        self.assertEqual(transaction_category_from_code("ai"), "unknown")
        self.assertEqual(
            frame.select(
                pc_cols.TRANSACTION_CODE,
                pc_cols.SECURITY_TYPE,
                pc_cols.TRANSACTION_CATEGORY,
                pc_cols.CASH_FLOW_SIGN,
                pc_cols.PERFORMANCE_FLOW_SIGN,
                pc_cols.TRANSACTION_SEMANTICS_SOURCE,
            ).to_dicts(),
            [
                {
                    pc_cols.TRANSACTION_CODE: "ai",
                    pc_cols.SECURITY_TYPE: "margin",
                    pc_cols.TRANSACTION_CATEGORY: TRANSACTION_CATEGORY_FEE_EXPENSE,
                    pc_cols.CASH_FLOW_SIGN: TRANSACTION_CASH_FLOW_SIGN_NEGATIVE,
                    pc_cols.PERFORMANCE_FLOW_SIGN: (
                        TRANSACTION_PERFORMANCE_FLOW_SIGN_PERFORMANCE
                    ),
                    pc_cols.TRANSACTION_SEMANTICS_SOURCE: (
                        TRANSACTION_SEMANTICS_SOURCE_YAML_RULE
                    ),
                },
            ],
        )

    def test_site_variant_epus_fee_context_uses_explicit_dp_rule(self) -> None:
        """The observed ``epus expense`` tokens remain contextual to ``dp``."""
        specification = AuditSpecification(
            _SITE_VARIANT_FIXTURES_PATH
            / "alternate_fee_context"
            / "ppar_audit.yaml"
        )

        frame = TransactionsLoader(specification).load("a")
        assert frame is not None

        self.assertEqual(transaction_category_from_code("epus"), "unknown")
        self.assertEqual(
            frame.select(
                pc_cols.TRANSACTION_CODE,
                pc_cols.SPECIAL_SECURITY_TYPE,
                pc_cols.SPECIAL_SECURITY_SYMBOL,
                pc_cols.TRANSACTION_CATEGORY,
                pc_cols.CASH_FLOW_SIGN,
                pc_cols.PERFORMANCE_FLOW_SIGN,
                pc_cols.TRANSACTION_SEMANTICS_SOURCE,
            ).to_dicts(),
            [
                {
                    pc_cols.TRANSACTION_CODE: "dp",
                    pc_cols.SPECIAL_SECURITY_TYPE: "epus",
                    pc_cols.SPECIAL_SECURITY_SYMBOL: "expense",
                    pc_cols.TRANSACTION_CATEGORY: TRANSACTION_CATEGORY_FEE_EXPENSE,
                    pc_cols.CASH_FLOW_SIGN: TRANSACTION_CASH_FLOW_SIGN_NEGATIVE,
                    pc_cols.PERFORMANCE_FLOW_SIGN: (
                        TRANSACTION_PERFORMANCE_FLOW_SIGN_PERFORMANCE
                    ),
                    pc_cols.TRANSACTION_SEMANTICS_SOURCE: (
                        TRANSACTION_SEMANTICS_SOURCE_YAML_RULE
                    ),
                }
            ],
        )

    def test_site_variant_rc_return_of_capital_uses_explicit_rules(self) -> None:
        """Return-of-capital rows stay YAML-scoped for Modified Dietz treatment."""
        specification = AuditSpecification(
            _SITE_VARIANT_FIXTURES_PATH
            / "rc_return_of_capital"
            / "ppar_audit.yaml"
        )

        frame = TransactionsLoader(specification).load("a")
        assert frame is not None

        self.assertEqual(transaction_category_from_code("rc"), "unknown")
        self.assertEqual(
            frame.select(
                pc_cols.TRANSACTION_CODE,
                pc_cols.SECURITY_TYPE,
                pc_cols.SPECIAL_SECURITY_TYPE,
                pc_cols.TRANSACTION_CATEGORY,
                pc_cols.CASH_FLOW_SIGN,
                pc_cols.PERFORMANCE_FLOW_SIGN,
                pc_cols.TRANSACTION_SEMANTICS_SOURCE,
            ).to_dicts(),
            [
                {
                    pc_cols.TRANSACTION_CODE: "rc",
                    pc_cols.SECURITY_TYPE: "equity",
                    pc_cols.SPECIAL_SECURITY_TYPE: "return_of_capital",
                    pc_cols.TRANSACTION_CATEGORY: TRANSACTION_CATEGORY_INCOME,
                    pc_cols.CASH_FLOW_SIGN: TRANSACTION_CASH_FLOW_SIGN_POSITIVE,
                    pc_cols.PERFORMANCE_FLOW_SIGN: (
                        TRANSACTION_PERFORMANCE_FLOW_SIGN_PERFORMANCE
                    ),
                    pc_cols.TRANSACTION_SEMANTICS_SOURCE: (
                        TRANSACTION_SEMANTICS_SOURCE_YAML_RULE
                    ),
                },
            ],
        )

    def test_site_variant_pd_principal_paydown_uses_explicit_rules(self) -> None:
        """Principal-paydown rows stay YAML-scoped for Modified Dietz treatment."""
        specification = AuditSpecification(
            _SITE_VARIANT_FIXTURES_PATH
            / "pd_principal_paydown"
            / "ppar_audit.yaml"
        )

        frame = TransactionsLoader(specification).load("a")
        assert frame is not None

        self.assertEqual(transaction_category_from_code("pd"), "unknown")
        self.assertEqual(
            frame.select(
                pc_cols.TRANSACTION_CODE,
                pc_cols.SECURITY_TYPE,
                pc_cols.SPECIAL_SECURITY_TYPE,
                pc_cols.TRANSACTION_CATEGORY,
                pc_cols.CASH_FLOW_SIGN,
                pc_cols.PERFORMANCE_FLOW_SIGN,
                pc_cols.TRANSACTION_SEMANTICS_SOURCE,
            ).to_dicts(),
            [
                {
                    pc_cols.TRANSACTION_CODE: "pd",
                    pc_cols.SECURITY_TYPE: "bond",
                    pc_cols.SPECIAL_SECURITY_TYPE: "principal_paydown",
                    pc_cols.TRANSACTION_CATEGORY: TRANSACTION_CATEGORY_INCOME,
                    pc_cols.CASH_FLOW_SIGN: TRANSACTION_CASH_FLOW_SIGN_POSITIVE,
                    pc_cols.PERFORMANCE_FLOW_SIGN: (
                        TRANSACTION_PERFORMANCE_FLOW_SIGN_PERFORMANCE
                    ),
                    pc_cols.TRANSACTION_SEMANTICS_SOURCE: (
                        TRANSACTION_SEMANTICS_SOURCE_YAML_RULE
                    ),
                },
            ],
        )

    def test_site_variant_short_side_trades_use_explicit_rules(self) -> None:
        """Short-side rows stay YAML-scoped and lowercase-code specific."""
        specification = AuditSpecification(
            _SITE_VARIANT_FIXTURES_PATH
            / "short_side_trades"
            / "ppar_audit.yaml"
        )

        frame = TransactionsLoader(specification).load("a")
        assert frame is not None

        self.assertEqual(transaction_category_from_code("ss"), "unknown")
        self.assertEqual(transaction_category_from_code("cs"), "unknown")
        self.assertEqual(
            frame.sort(pc_cols.TRANSACTION_CODE, descending=True)
            .select(
                pc_cols.TRANSACTION_CODE,
                pc_cols.SECURITY_TYPE,
                pc_cols.TRANSACTION_CATEGORY,
                pc_cols.CASH_FLOW_SIGN,
                pc_cols.PERFORMANCE_FLOW_SIGN,
                pc_cols.TRANSACTION_SEMANTICS_SOURCE,
            )
            .to_dicts(),
            [
                {
                    pc_cols.TRANSACTION_CODE: "ss",
                    pc_cols.SECURITY_TYPE: "short",
                    pc_cols.TRANSACTION_CATEGORY: TRANSACTION_CATEGORY_SELL,
                    pc_cols.CASH_FLOW_SIGN: TRANSACTION_CASH_FLOW_SIGN_POSITIVE,
                    pc_cols.PERFORMANCE_FLOW_SIGN: (
                        TRANSACTION_PERFORMANCE_FLOW_SIGN_PERFORMANCE
                    ),
                    pc_cols.TRANSACTION_SEMANTICS_SOURCE: (
                        TRANSACTION_SEMANTICS_SOURCE_YAML_RULE
                    ),
                },
                {
                    pc_cols.TRANSACTION_CODE: "cs",
                    pc_cols.SECURITY_TYPE: "short",
                    pc_cols.TRANSACTION_CATEGORY: TRANSACTION_CATEGORY_BUY,
                    pc_cols.CASH_FLOW_SIGN: TRANSACTION_CASH_FLOW_SIGN_NEGATIVE,
                    pc_cols.PERFORMANCE_FLOW_SIGN: (
                        TRANSACTION_PERFORMANCE_FLOW_SIGN_PERFORMANCE
                    ),
                    pc_cols.TRANSACTION_SEMANTICS_SOURCE: (
                        TRANSACTION_SEMANTICS_SOURCE_YAML_RULE
                    ),
                },
            ],
        )

    def test_site_variant_rep_semantics_can_supply_ambiguous_flow_context(self) -> None:
        """REP/report semantics can be the reviewed context for ambiguous codes."""
        specification = AuditSpecification(
            _SITE_VARIANT_FIXTURES_PATH
            / "rep_semantics"
            / "ppar_audit.yaml"
        )

        frame = TransactionsLoader(specification).load("a")
        assert frame is not None

        self.assertEqual(
            frame.select(
                pc_cols.TRANSACTION_CODE,
                pc_cols.TRANSACTION_CATEGORY,
                pc_cols.CASH_FLOW_SIGN,
                pc_cols.PERFORMANCE_FLOW_SIGN,
                pc_cols.TRANSACTION_SEMANTICS_SOURCE,
            ).to_dicts(),
            [
                {
                    pc_cols.TRANSACTION_CODE: "li",
                    pc_cols.TRANSACTION_CATEGORY: TRANSACTION_CATEGORY_EXTERNAL_FLOW,
                    pc_cols.CASH_FLOW_SIGN: TRANSACTION_CASH_FLOW_SIGN_POSITIVE,
                    pc_cols.PERFORMANCE_FLOW_SIGN: (
                        TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL
                    ),
                    pc_cols.TRANSACTION_SEMANTICS_SOURCE: (
                        TRANSACTION_SEMANTICS_SOURCE_SOURCE
                    ),
                },
                {
                    pc_cols.TRANSACTION_CODE: "lo",
                    pc_cols.TRANSACTION_CATEGORY: TRANSACTION_CATEGORY_EXTERNAL_FLOW,
                    pc_cols.CASH_FLOW_SIGN: TRANSACTION_CASH_FLOW_SIGN_NEGATIVE,
                    pc_cols.PERFORMANCE_FLOW_SIGN: (
                        TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL
                    ),
                    pc_cols.TRANSACTION_SEMANTICS_SOURCE: (
                        TRANSACTION_SEMANTICS_SOURCE_SOURCE
                    ),
                },
                {
                    pc_cols.TRANSACTION_CODE: "dp",
                    pc_cols.TRANSACTION_CATEGORY: TRANSACTION_CATEGORY_FEE_EXPENSE,
                    pc_cols.CASH_FLOW_SIGN: TRANSACTION_CASH_FLOW_SIGN_NEGATIVE,
                    pc_cols.PERFORMANCE_FLOW_SIGN: (
                        TRANSACTION_PERFORMANCE_FLOW_SIGN_PERFORMANCE
                    ),
                    pc_cols.TRANSACTION_SEMANTICS_SOURCE: (
                        TRANSACTION_SEMANTICS_SOURCE_SOURCE
                    ),
                },
                {
                    pc_cols.TRANSACTION_CODE: "wd",
                    pc_cols.TRANSACTION_CATEGORY: TRANSACTION_CATEGORY_EXTERNAL_FLOW,
                    pc_cols.CASH_FLOW_SIGN: TRANSACTION_CASH_FLOW_SIGN_NEGATIVE,
                    pc_cols.PERFORMANCE_FLOW_SIGN: (
                        TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL
                    ),
                    pc_cols.TRANSACTION_SEMANTICS_SOURCE: (
                        TRANSACTION_SEMANTICS_SOURCE_SOURCE
                    ),
                },
            ],
        )

    def test_site_variant_review_only_actions_stay_neutral(self) -> None:
        """Correction and synthetic corporate-action rows stay review-only."""
        specification = AuditSpecification(
            _SITE_VARIANT_FIXTURES_PATH
            / "review_only_actions"
            / "ppar_audit.yaml"
        )

        frame = TransactionsLoader(specification).load("a")
        assert frame is not None

        self.assertEqual(
            frame.select(
                pc_cols.TRANSACTION_CODE,
                pc_cols.TRANSACTION_CATEGORY,
                pc_cols.CASH_FLOW_SIGN,
                pc_cols.PERFORMANCE_FLOW_SIGN,
                pc_cols.TRANSACTION_SEMANTICS_SOURCE,
            ).to_dicts(),
            [
                {
                    pc_cols.TRANSACTION_CODE: "CXL",
                    pc_cols.TRANSACTION_CATEGORY: TRANSACTION_CATEGORY_TRANSFER,
                    pc_cols.CASH_FLOW_SIGN: TRANSACTION_CASH_FLOW_SIGN_NONE,
                    pc_cols.PERFORMANCE_FLOW_SIGN: (
                        TRANSACTION_PERFORMANCE_FLOW_SIGN_NEUTRAL
                    ),
                    pc_cols.TRANSACTION_SEMANTICS_SOURCE: (
                        TRANSACTION_SEMANTICS_SOURCE_SOURCE
                    ),
                },
                {
                    pc_cols.TRANSACTION_CODE: "REV",
                    pc_cols.TRANSACTION_CATEGORY: TRANSACTION_CATEGORY_TRANSFER,
                    pc_cols.CASH_FLOW_SIGN: TRANSACTION_CASH_FLOW_SIGN_NONE,
                    pc_cols.PERFORMANCE_FLOW_SIGN: (
                        TRANSACTION_PERFORMANCE_FLOW_SIGN_NEUTRAL
                    ),
                    pc_cols.TRANSACTION_SEMANTICS_SOURCE: (
                        TRANSACTION_SEMANTICS_SOURCE_SOURCE
                    ),
                },
                {
                    pc_cols.TRANSACTION_CODE: ";",
                    pc_cols.TRANSACTION_CATEGORY: (
                        TRANSACTION_CATEGORY_CORPORATE_ACTION
                    ),
                    pc_cols.CASH_FLOW_SIGN: TRANSACTION_CASH_FLOW_SIGN_NONE,
                    pc_cols.PERFORMANCE_FLOW_SIGN: (
                        TRANSACTION_PERFORMANCE_FLOW_SIGN_NEUTRAL
                    ),
                    pc_cols.TRANSACTION_SEMANTICS_SOURCE: (
                        TRANSACTION_SEMANTICS_SOURCE_SOURCE
                    ),
                },
            ],
        )

    def test_source_semantics_can_classify_ambiguous_codes_as_neutral(self) -> None:
        """REP/report semantics can mark ambiguous Axys codes as non-external."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            _write_extract_contract(
                directory,
                [
                    "TRANSACTION_CATEGORY",
                    "CASH_FLOW_SIGN",
                    "PERFORMANCE_FLOW_SIGN",
                ],
            )
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "transactions": "transactions.csv",
            }
            configuration["extract_contract"] = {
                "path": "site_extract_contract.yaml",
                "enforce_ambiguous_axys_flows": True,
            }
            rows = {
                "PORT": ["P1", "P1", "P1", "P1"],
                "SEC": ["CASHUSD", "CASHUSD", "CASHUSD", "CASHUSD"],
                "TRANSACTION_DATE": ["2025-01-31"] * 4,
                "TRAN": ["li", "lo", "dp", "wd"],
                "TRANSACTION_CATEGORY": [
                    "transfer",
                    "transfer",
                    "fee_expense",
                    "transfer",
                ],
                "CASH_FLOW_SIGN": ["none", "none", "negative", "none"],
                "PERFORMANCE_FLOW_SIGN": [
                    "neutral",
                    "neutral",
                    "performance",
                    "neutral",
                ],
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(rows).write_csv(directory / snapshot_name / "transactions.csv")
            path = _write_yaml(directory, configuration)
            specification = AuditSpecification(path)

            frame = TransactionsLoader(specification).load("a")
            assert frame is not None

            self.assertEqual(
                frame.select(
                    pc_cols.TRANSACTION_CODE,
                    pc_cols.TRANSACTION_CATEGORY,
                    pc_cols.CASH_FLOW_SIGN,
                    pc_cols.PERFORMANCE_FLOW_SIGN,
                    pc_cols.TRANSACTION_SEMANTICS_SOURCE,
                ).to_dicts(),
                [
                    {
                        pc_cols.TRANSACTION_CODE: "li",
                        pc_cols.TRANSACTION_CATEGORY: TRANSACTION_CATEGORY_TRANSFER,
                        pc_cols.CASH_FLOW_SIGN: TRANSACTION_CASH_FLOW_SIGN_NONE,
                        pc_cols.PERFORMANCE_FLOW_SIGN: (
                            TRANSACTION_PERFORMANCE_FLOW_SIGN_NEUTRAL
                        ),
                        pc_cols.TRANSACTION_SEMANTICS_SOURCE: (
                            TRANSACTION_SEMANTICS_SOURCE_SOURCE
                        ),
                    },
                    {
                        pc_cols.TRANSACTION_CODE: "lo",
                        pc_cols.TRANSACTION_CATEGORY: TRANSACTION_CATEGORY_TRANSFER,
                        pc_cols.CASH_FLOW_SIGN: TRANSACTION_CASH_FLOW_SIGN_NONE,
                        pc_cols.PERFORMANCE_FLOW_SIGN: (
                            TRANSACTION_PERFORMANCE_FLOW_SIGN_NEUTRAL
                        ),
                        pc_cols.TRANSACTION_SEMANTICS_SOURCE: (
                            TRANSACTION_SEMANTICS_SOURCE_SOURCE
                        ),
                    },
                    {
                        pc_cols.TRANSACTION_CODE: "dp",
                        pc_cols.TRANSACTION_CATEGORY: TRANSACTION_CATEGORY_FEE_EXPENSE,
                        pc_cols.CASH_FLOW_SIGN: TRANSACTION_CASH_FLOW_SIGN_NEGATIVE,
                        pc_cols.PERFORMANCE_FLOW_SIGN: (
                            TRANSACTION_PERFORMANCE_FLOW_SIGN_PERFORMANCE
                        ),
                        pc_cols.TRANSACTION_SEMANTICS_SOURCE: (
                            TRANSACTION_SEMANTICS_SOURCE_SOURCE
                        ),
                    },
                    {
                        pc_cols.TRANSACTION_CODE: "wd",
                        pc_cols.TRANSACTION_CATEGORY: TRANSACTION_CATEGORY_TRANSFER,
                        pc_cols.CASH_FLOW_SIGN: TRANSACTION_CASH_FLOW_SIGN_NONE,
                        pc_cols.PERFORMANCE_FLOW_SIGN: (
                            TRANSACTION_PERFORMANCE_FLOW_SIGN_NEUTRAL
                        ),
                        pc_cols.TRANSACTION_SEMANTICS_SOURCE: (
                            TRANSACTION_SEMANTICS_SOURCE_SOURCE
                        ),
                    },
                ],
            )

    def test_site_variant_code_only_imex_ambiguous_codes_fail_fast(self) -> None:
        """Code-only IMEX fixtures cannot classify ambiguous external flows."""
        specification = AuditSpecification(
            _SITE_VARIANT_FIXTURES_PATH
            / "imex_code_only"
            / "ppar_audit.yaml"
        )

        with self.assertRaises(PpaError) as context:
            TransactionsLoader(specification).load("a")

        message = str(context.exception)
        self.assertIn("ambiguous Axys/APX transaction codes DP, LI, LO, TI, WD", message)
        self.assertIn("IMEX transaction code alone is not enough", message)
        self.assertIn("REP/report extract", message)

    def test_site_variant_local_opt_out_classifies_code_only_rows(self) -> None:
        """Reviewed local opt-out allows code-only ambiguous rows by design."""
        specification = AuditSpecification(
            _SITE_VARIANT_FIXTURES_PATH
            / "local_opt_out"
            / "ppar_audit.yaml"
        )

        frame = TransactionsLoader(specification).load("a")
        assert frame is not None

        self.assertEqual(
            frame.select(
                pc_cols.TRANSACTION_CODE,
                pc_cols.TRANSACTION_CATEGORY,
                pc_cols.CASH_FLOW_SIGN,
                pc_cols.PERFORMANCE_FLOW_SIGN,
                pc_cols.TRANSACTION_SEMANTICS_SOURCE,
            ).to_dicts(),
            [
                {
                    pc_cols.TRANSACTION_CODE: "li",
                    pc_cols.TRANSACTION_CATEGORY: TRANSACTION_CATEGORY_EXTERNAL_FLOW,
                    pc_cols.CASH_FLOW_SIGN: TRANSACTION_CASH_FLOW_SIGN_POSITIVE,
                    pc_cols.PERFORMANCE_FLOW_SIGN: (
                        TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL
                    ),
                    pc_cols.TRANSACTION_SEMANTICS_SOURCE: (
                        TRANSACTION_SEMANTICS_SOURCE_YAML_RULE
                    ),
                },
                {
                    pc_cols.TRANSACTION_CODE: "lo",
                    pc_cols.TRANSACTION_CATEGORY: TRANSACTION_CATEGORY_EXTERNAL_FLOW,
                    pc_cols.CASH_FLOW_SIGN: TRANSACTION_CASH_FLOW_SIGN_NEGATIVE,
                    pc_cols.PERFORMANCE_FLOW_SIGN: (
                        TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL
                    ),
                    pc_cols.TRANSACTION_SEMANTICS_SOURCE: (
                        TRANSACTION_SEMANTICS_SOURCE_YAML_RULE
                    ),
                },
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
                {
                    pc_cols.TRANSACTION_CODE: "wd",
                    pc_cols.TRANSACTION_CATEGORY: TRANSACTION_CATEGORY_EXTERNAL_FLOW,
                    pc_cols.CASH_FLOW_SIGN: TRANSACTION_CASH_FLOW_SIGN_NEGATIVE,
                    pc_cols.PERFORMANCE_FLOW_SIGN: (
                        TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL
                    ),
                    pc_cols.TRANSACTION_SEMANTICS_SOURCE: (
                        TRANSACTION_SEMANTICS_SOURCE_YAML_RULE
                    ),
                },
            ],
        )
        self.assertFalse(
            validate_config(
                _SITE_VARIANT_FIXTURES_PATH
                / "local_opt_out"
                / "ppar_audit.yaml",
                require_complete_yaml_setup=False,
            )["enforce_ambiguous_axys_flows"]
        )

    def test_ambiguous_axys_code_without_matching_context_raises_error(self) -> None:
        """Ambiguous Axys codes fail when IMEX fields cannot match a YAML rule."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "transactions": "transactions.csv",
            }
            configuration["transaction_rules"] = {
                "wd": [
                    {
                        "when": {
                            "security_id": "CASHUSD",
                            "source_destination_type": "$pty",
                            "source_destination_symbol": "$cash",
                        },
                        "transaction_category": "external_flow",
                        "cash_flow_sign": "negative",
                        "performance_flow_sign": "external",
                    }
                ]
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(
                    {
                        "PORT": ["P1"],
                        "SEC": ["CASHUSD"],
                        "TRANSACTION_DATE": ["2025-01-31"],
                        "TRAN": ["wd"],
                    }
                ).write_csv(directory / snapshot_name / "transactions.csv")
            path = _write_yaml(directory, configuration)
            specification = AuditSpecification(path)

            with self.assertRaises(PpaError) as context:
                TransactionsLoader(specification).load("a")

            message = str(context.exception)
            self.assertIn("transaction semantics/context fields", message)
            self.assertIn("REP/report extract", message)

    def test_ambiguous_axys_code_with_nonmatching_context_raises_error(self) -> None:
        """Context columns must match a reviewed rule before semantics are usable."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            _write_extract_contract(
                directory,
                [
                    "SRC_DEST_TYPE",
                    "SRC_DEST_SYMBOL",
                    "SPECIAL_SEC_TYPE",
                    "SPECIAL_SEC_SYMBOL",
                ],
            )
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "transactions": "transactions.csv",
            }
            configuration["extract_contract"] = {
                "path": "site_extract_contract.yaml",
                "enforce_ambiguous_axys_flows": True,
            }
            configuration["transaction_rules"] = {
                "dp": [
                    {
                        "when": {
                            "special_security_type": "exus",
                            "special_security_symbol": "custfee",
                        },
                        "transaction_category": "fee_expense",
                        "cash_flow_sign": "negative",
                        "performance_flow_sign": "performance",
                    }
                ],
                "wd": [
                    {
                        "when": {
                            "security_id": "CASHUSD",
                            "source_destination_type": "$pty",
                            "source_destination_symbol": "$cash",
                        },
                        "transaction_category": "external_flow",
                        "cash_flow_sign": "negative",
                        "performance_flow_sign": "external",
                    }
                ],
            }
            rows = {
                "PORT": ["P1", "P1"],
                "SEC": ["CASHUSD", "CASHUSD"],
                "TRANSACTION_DATE": ["2025-01-31", "2025-01-31"],
                "TRAN": ["dp", "wd"],
                "SRC_DEST_TYPE": ["$pty", "$vendor"],
                "SRC_DEST_SYMBOL": ["$cash", "UNKNOWN"],
                "SPECIAL_SEC_TYPE": ["exus", ""],
                "SPECIAL_SEC_SYMBOL": ["notcustfee", ""],
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(rows).write_csv(directory / snapshot_name / "transactions.csv")
            path = _write_yaml(directory, configuration)
            specification = AuditSpecification(path)

            with self.assertRaises(PpaError) as context:
                TransactionsLoader(specification).load("a")

            message = str(context.exception)
            self.assertIn("unknown transaction codes or categories", message)
            self.assertIn("transaction_code=dp", message)
            self.assertIn("source_destination_type=$vendor", message)
            self.assertIn("special_security_symbol=notcustfee", message)

    def test_ambiguous_axys_code_with_unconditional_rule_requires_context(self) -> None:
        """Ambiguous Axys codes cannot be classified by broad rules without context."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "transactions": "transactions.csv",
            }
            configuration["transaction_rules"] = {
                "wd": {
                    "transaction_category": "external_flow",
                    "cash_flow_sign": "negative",
                    "performance_flow_sign": "external",
                }
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(
                    {
                        "PORT": ["P1"],
                        "SEC": ["CASHUSD"],
                        "TRANSACTION_DATE": ["2025-01-31"],
                        "TRAN": ["wd"],
                    }
                ).write_csv(directory / snapshot_name / "transactions.csv")
            path = _write_yaml(directory, configuration)
            specification = AuditSpecification(path)

            with self.assertRaises(PpaError) as context:
                TransactionsLoader(specification).load("a")

            message = str(context.exception)
            self.assertIn("ambiguous Axys/APX transaction codes WD", message)
            self.assertIn("IMEX transaction code alone is not enough", message)
            self.assertIn("REP/report extract", message)

    def test_site_extract_contract_can_define_required_context_fields(self) -> None:
        """A local extract contract can define the context fields a site exposes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            _write_extract_contract(directory, ["SRC_DEST_TYPE"])
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "transactions": "transactions.csv",
            }
            configuration["extract_contract"] = {
                "path": "site_extract_contract.yaml",
                "enforce_ambiguous_axys_flows": True,
            }
            configuration["transaction_rules"] = {
                "wd": {
                    "transaction_category": "external_flow",
                    "cash_flow_sign": "negative",
                    "performance_flow_sign": "external",
                }
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(
                    {
                        "PORT": ["P1"],
                        "SEC": ["CASHUSD"],
                        "TRANSACTION_DATE": ["2025-01-31"],
                        "TRAN": ["wd"],
                        "SRC_DEST_TYPE": ["$pty"],
                    }
                ).write_csv(directory / snapshot_name / "transactions.csv")
            path = _write_yaml(directory, configuration)
            specification = AuditSpecification(path)

            frame = TransactionsLoader(specification).load("a")
            assert frame is not None

            row = frame.row(0, named=True)
            self.assertEqual(
                row[pc_cols.TRANSACTION_CATEGORY],
                TRANSACTION_CATEGORY_EXTERNAL_FLOW,
            )
            self.assertEqual(
                row[pc_cols.TRANSACTION_SEMANTICS_SOURCE],
                TRANSACTION_SEMANTICS_SOURCE_YAML_RULE,
            )

    def test_context_rule_is_authoritative_over_source_semantics(self) -> None:
        """A complete matching context rule is the semantic authority."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            _write_extract_contract(directory, ["SRC_DEST_TYPE"])
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "transactions": "transactions.csv",
            }
            configuration["extract_contract"] = {
                "path": "site_extract_contract.yaml",
                "enforce_ambiguous_axys_flows": True,
            }
            configuration["transaction_rules"] = {
                "li": {
                    "when": {"source_destination_type": "$pty"},
                    "transaction_category": "external_flow",
                    "cash_flow_sign": "positive",
                    "performance_flow_sign": "external",
                }
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(
                    {
                        "PORT": ["P1"],
                        "SEC": ["CASHUSD"],
                        "TRANSACTION_DATE": ["2025-01-31"],
                        "TRAN": ["li"],
                        "SRC_DEST_TYPE": [" $PTY "],
                        "TRANSACTION_CATEGORY": ["cash deposit"],
                        "CASH_FLOW_SIGN": ["cash in"],
                    }
                ).write_csv(directory / snapshot_name / "transactions.csv")
            path = _write_yaml(directory, configuration)
            specification = AuditSpecification(path)

            frame = TransactionsLoader(specification).load("a")
            assert frame is not None

            row = frame.row(0, named=True)
            self.assertEqual(
                row[pc_cols.TRANSACTION_CATEGORY],
                TRANSACTION_CATEGORY_EXTERNAL_FLOW,
            )
            self.assertEqual(row[pc_cols.CASH_FLOW_SIGN], TRANSACTION_CASH_FLOW_SIGN_POSITIVE)
            self.assertEqual(
                row[pc_cols.PERFORMANCE_FLOW_SIGN],
                TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL,
            )
            self.assertEqual(
                row[pc_cols.TRANSACTION_SEMANTICS_SOURCE],
                TRANSACTION_SEMANTICS_SOURCE_YAML_RULE,
            )

    def test_legacy_transaction_rules_remain_case_insensitive(self) -> None:
        """Omitted case policy preserves existing rule-key and context matching."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "transactions": "transactions.csv",
            }
            configuration["transaction_rules"] = {
                "by": {
                    "when": {"security_type": "csus"},
                    "transaction_category": "fee_expense",
                    "cash_flow_sign": "negative",
                    "performance_flow_sign": "performance",
                }
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(
                    {
                        "PORT": ["P1"],
                        "SEC": ["SEC1"],
                        "TRANSACTION_DATE": ["2025-01-31"],
                        "TRAN": ["BY"],
                        "SEC_TYPE": ["CSUS"],
                    }
                ).write_csv(directory / snapshot_name / "transactions.csv")
            path = _write_yaml(directory, configuration)
            frame = TransactionsLoader(AuditSpecification(path)).load("a")
            assert frame is not None

            row = frame.row(0, named=True)
            self.assertEqual(
                row[pc_cols.TRANSACTION_CATEGORY],
                TRANSACTION_CATEGORY_FEE_EXPENSE,
            )
            self.assertEqual(
                row[pc_cols.TRANSACTION_SEMANTICS_SOURCE],
                TRANSACTION_SEMANTICS_SOURCE_YAML_RULE,
            )

    def test_exact_case_rules_distinguish_codes_and_context_values(self) -> None:
        """A versioned contract may give lowercase and uppercase codes distinct roles."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            _write_extract_contract(directory, ["SEC_TYPE"], version=1)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "transactions": "transactions.csv",
            }
            configuration["extract_contract"] = {
                "path": "site_extract_contract.yaml",
                "transaction_semantics_case": "exact",
            }
            configuration["transaction_rules"] = {
                "by": {
                    "when": {"security_type": "csus"},
                    "transaction_category": "buy",
                    "cash_flow_sign": "negative",
                    "performance_flow_sign": "performance",
                },
                "BY": {
                    "when": {"security_type": "CSUS"},
                    "transaction_category": "sell",
                    "cash_flow_sign": "positive",
                    "performance_flow_sign": "performance",
                },
            }
            transaction_rows = {
                "PORT": ["P1", "P1"],
                "SEC": ["LOWER", "UPPER"],
                "TRANSACTION_DATE": ["2025-01-30", "2025-01-31"],
                "TRAN": ["by", "BY"],
                "SEC_TYPE": ["csus", "CSUS"],
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(transaction_rows).write_csv(
                    directory / snapshot_name / "transactions.csv"
                )
            path = _write_yaml(directory, configuration)
            frame = TransactionsLoader(AuditSpecification(path)).load("a")
            assert frame is not None

            rows = frame.sort(pc_cols.SECURITY_ID).to_dicts()
            self.assertEqual(
                [row[pc_cols.TRANSACTION_CODE] for row in rows],
                ["by", "BY"],
            )
            self.assertEqual(
                [row[pc_cols.TRANSACTION_CATEGORY] for row in rows],
                [TRANSACTION_CATEGORY_BUY, TRANSACTION_CATEGORY_SELL],
            )
            self.assertEqual(
                validate_config(path, require_complete_yaml_setup=False)[
                    "transaction_codes_without_yaml_rules"
                ],
                "none",
            )

    def test_unsupported_uppercase_code_cannot_become_performance_cause(self) -> None:
        """An unsupported uppercase code fails before cause generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            _write_extract_contract(directory, ["SEC_TYPE"], version=1)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "transactions": "transactions.csv",
            }
            configuration["extract_contract"] = {
                "path": "site_extract_contract.yaml",
                "transaction_semantics_case": "exact",
            }
            configuration["transaction_rules"] = {
                "by": {
                    "when": {"security_type": "csus"},
                    "transaction_category": "buy",
                    "cash_flow_sign": "negative",
                    "performance_flow_sign": "performance",
                }
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(
                    {
                        "PORT": ["P1"],
                        "SEC": ["SEC1"],
                        "TRANSACTION_DATE": ["2025-01-31"],
                        "TRAN": ["BY"],
                        "SEC_TYPE": ["CSUS"],
                    }
                ).write_csv(directory / snapshot_name / "transactions.csv")
            path = _write_yaml(directory, configuration)

            with self.assertRaises(PpaError) as context:
                TransactionsLoader(AuditSpecification(path)).load("a")

            message = str(context.exception)
            self.assertIn("unknown transaction codes or categories", message)
            self.assertIn("transaction_code=BY", message)

    def test_exact_case_requires_matching_context_case(self) -> None:
        """Exact mode does not fold native transaction context identifiers."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            _write_extract_contract(directory, ["SEC_TYPE"], version=1)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "transactions": "transactions.csv",
            }
            configuration["extract_contract"] = {
                "path": "site_extract_contract.yaml",
                "transaction_semantics_case": "exact",
            }
            configuration["transaction_rules"] = {
                "by": {
                    "when": {"security_type": "csus"},
                    "transaction_category": "buy",
                    "cash_flow_sign": "negative",
                    "performance_flow_sign": "performance",
                }
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(
                    {
                        "PORT": ["P1"],
                        "SEC": ["SEC1"],
                        "TRANSACTION_DATE": ["2025-01-31"],
                        "TRAN": ["by"],
                        "SEC_TYPE": ["CSUS"],
                    }
                ).write_csv(directory / snapshot_name / "transactions.csv")
            path = _write_yaml(directory, configuration)

            with self.assertRaisesRegex(PpaError, "security_type=CSUS"):
                TransactionsLoader(AuditSpecification(path)).load("a")

    def test_exact_case_requires_versioned_extract_contract(self) -> None:
        """Exact transaction semantics fail closed for an unversioned contract."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            _write_extract_contract(directory, ["SEC_TYPE"])
            configuration = _minimal_specification(directory)
            configuration["extract_contract"] = {
                "path": "site_extract_contract.yaml",
                "transaction_semantics_case": "exact",
            }
            path = _write_yaml(directory, configuration)

            with self.assertRaisesRegex(
                PpaError,
                "exact transaction semantics require a positive integer contract version",
            ):
                validate_config(path, require_complete_yaml_setup=False)

    def test_invalid_transaction_semantics_case_fails_closed(self) -> None:
        """Unknown extract-contract case policy is rejected."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["extract_contract"] = {
                "transaction_semantics_case": "sometimes",
            }
            path = _write_yaml(directory, configuration)

            with self.assertRaisesRegex(
                PpaError,
                "transaction_semantics_case must be one of",
            ):
                validate_config(path, require_complete_yaml_setup=False)

    def test_validate_config_rejects_extract_contract_without_transactions(self) -> None:
        """A local extract contract must define transaction columns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            _write_raw_extract_contract(directory, {"datasets": {}})
            configuration = _minimal_specification(directory)
            configuration["extract_contract"] = {"path": "site_extract_contract.yaml"}
            path = _write_yaml(directory, configuration)

            with self.assertRaises(PpaError) as context:
                validate_config(path)

            self.assertIn("transactions.csv must be a mapping", str(context.exception))

    def test_validate_config_rejects_extract_contract_unknown_column(self) -> None:
        """A local extract contract cannot name unsupported transaction columns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            _write_extract_contract(directory, ["NOT_AN_AXYS_FIELD"])
            configuration = _minimal_specification(directory)
            configuration["extract_contract"] = {"path": "site_extract_contract.yaml"}
            path = _write_yaml(directory, configuration)

            with self.assertRaises(PpaError) as context:
                validate_config(path)

            self.assertIn(
                "Unsupported contract transaction column 'NOT_AN_AXYS_FIELD'",
                str(context.exception),
            )

    def test_validate_config_rejects_extract_contract_non_boolean_flags(self) -> None:
        """A local extract contract must use boolean guard flags."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            _write_raw_extract_contract(
                directory,
                {
                    "datasets": {
                        "transactions.csv": {
                            "columns": {
                                "SRC_DEST_TYPE": {
                                    "requires_context_for_semantics": "yes",
                                    "blocking_if_missing": True,
                                }
                            }
                        }
                    }
                },
            )
            configuration = _minimal_specification(directory)
            configuration["extract_contract"] = {"path": "site_extract_contract.yaml"}
            path = _write_yaml(directory, configuration)

            with self.assertRaises(PpaError) as context:
                validate_config(path)

            self.assertIn(
                "must define boolean requires_context_for_semantics",
                str(context.exception),
            )

    def test_extract_contract_can_disable_ambiguous_flow_enforcement(self) -> None:
        """A config can intentionally disable ambiguous Axys/APX flow enforcement."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "transactions": "transactions.csv",
            }
            configuration["extract_contract"] = {
                "enforce_ambiguous_axys_flows": False,
            }
            configuration["transaction_rules"] = {
                "wd": {
                    "transaction_category": "external_flow",
                    "cash_flow_sign": "negative",
                    "performance_flow_sign": "external",
                }
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(
                    {
                        "PORT": ["P1"],
                        "SEC": ["CASHUSD"],
                        "TRANSACTION_DATE": ["2025-01-31"],
                        "TRAN": ["wd"],
                    }
                ).write_csv(directory / snapshot_name / "transactions.csv")
            path = _write_yaml(directory, configuration)
            specification = AuditSpecification(path)

            frame = TransactionsLoader(specification).load("a")
            assert frame is not None

            self.assertEqual(
                frame.row(0, named=True)[pc_cols.TRANSACTION_SEMANTICS_SOURCE],
                TRANSACTION_SEMANTICS_SOURCE_YAML_RULE,
            )

    def test_invalid_yaml_transaction_rules_raise_error(self) -> None:
        """Transaction rules must be a YAML mapping."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "transactions": "transactions.csv",
            }
            configuration["transaction_rules"] = ["BUY"]
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(
                    {
                        "PORT": ["P1"],
                        "SEC": ["S1"],
                        "TRANSACTION_DATE": ["2025-01-31"],
                        "TRAN": ["BUY"],
                    }
                ).write_csv(directory / snapshot_name / "transactions.csv")
            path = _write_yaml(directory, configuration)
            specification = AuditSpecification(path)

            with self.assertRaises(PpaError) as context:
                TransactionsLoader(specification).load("a")

            self.assertIn("transaction_rules must be a mapping", str(context.exception))

    def test_omitted_transactions_returns_none(self) -> None:
        """Transactions are optional when omitted from YAML."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            path = _write_yaml(directory, _minimal_specification(directory))
            specification = AuditSpecification(path)

            self.assertIsNone(TransactionsLoader(specification).load("a"))

    def test_missing_optional_transactions_returns_none(self) -> None:
        """Missing optional transaction files do not block loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "transactions": "missing_transactions.csv",
            }
            path = _write_yaml(directory, configuration)
            specification = AuditSpecification(path)

            self.assertIsNone(TransactionsLoader(specification).load("a"))

    def test_missing_required_column_raises_error_502(self) -> None:
        """Existing transaction files must contain portfolio, security, and date."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "transactions": "transactions.csv",
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(
                    {
                        "PORT": ["P1"],
                        "SEC": ["S1"],
                        "AMOUNT": [10.0],
                    }
                ).write_csv(directory / snapshot_name / "transactions.csv")
            path = _write_yaml(directory, configuration)
            specification = AuditSpecification(path)

            with self.assertRaises(PpaError) as context:
                TransactionsLoader(specification).load("a")

            self.assertTrue(str(context.exception).startswith("Error 502"))
            self.assertIn("transaction_date", str(context.exception))

    def test_ambiguous_required_column_raises_error_502(self) -> None:
        """Transaction identifier columns must not match multiple aliases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "transactions": "transactions.csv",
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(
                    {
                        "PORT": ["P1"],
                        "PORTFOLIO_ID": ["P1"],
                        "SEC": ["S1"],
                        "TRANSACTION_DATE": ["2025-01-31"],
                    }
                ).write_csv(directory / snapshot_name / "transactions.csv")
            path = _write_yaml(directory, configuration)
            specification = AuditSpecification(path)

            with self.assertRaises(PpaError) as context:
                TransactionsLoader(specification).load("a")

            self.assertTrue(str(context.exception).startswith("Error 502"))
            self.assertIn("Ambiguous transactions", str(context.exception))

    def test_nonnumeric_transaction_amount_raises_error_502(self) -> None:
        """Malformed transaction numeric values fail with field-level context."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            configuration = _minimal_specification(directory)
            configuration["files"] = {
                "portfolio_performance": "portperf.csv",
                "transactions": "transactions.csv",
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                (directory / snapshot_name / "transactions.csv").write_text(
                    "PORT,SEC,TRANSACTION_DATE,AMOUNT\n"
                    "P1,S1,2025-01-31,--\n",
                    encoding="utf-8",
                )
            path = _write_yaml(directory, configuration)
            specification = AuditSpecification(path)

            with self.assertRaises(PpaError) as context:
                TransactionsLoader(specification).load("a")

            message = str(context.exception)
            self.assertTrue(message.startswith("Error 502"))
            self.assertIn("transactions", message)
            self.assertIn("amount", message)
            self.assertIn("--", message)


if __name__ == "__main__":
    unittest.main()
