"""Tests for loading normalized transaction comparison sources."""

# Python imports
from pathlib import Path
import tempfile
import unittest

# Third-party imports
import polars as pl
import yaml

# Project imports
from ppar.errors import PpaError
from ppar.performance_comparison import (
    PerformanceComparisonSpecification,
    TransactionsLoader,
)
from ppar.performance_comparison import schema as pc_cols
from ppar.performance_comparison.config_validation import validate_config
from ppar.performance_comparison.extract_contract import validate_extract_contract
from ppar.performance_comparison.transactions import (
    TRANSACTION_CASH_FLOW_SIGN_NEGATIVE,
    TRANSACTION_CASH_FLOW_SIGN_NONE,
    TRANSACTION_CASH_FLOW_SIGN_POSITIVE,
    TRANSACTION_CASH_FLOW_SIGN_UNKNOWN,
    TRANSACTION_CATEGORY_BUY,
    TRANSACTION_CATEGORY_EXTERNAL_FLOW,
    TRANSACTION_CATEGORY_FEE_EXPENSE,
    TRANSACTION_CATEGORY_INCOME,
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
    transaction_impact_semantics_available,
    transaction_category_from_code,
)

_BASELINE_COMPARISON_PATH = Path("tests/data/axys/validation/ppar_performance_comparison.yaml")
_SITE_EXTRACT_CONTRACT_TEMPLATE_PATH = Path(
    "docs/axys-apx-reference/templates/site_extract_contract.yaml"
)
_SITE_VARIANT_FIXTURES_PATH = Path("tests/data/axys/site_variants")


def _write_yaml(directory: Path, contents: object) -> Path:
    """Write comparison YAML contents and return the path."""
    path = directory / "ppar_performance_comparison.yaml"
    path.write_text(yaml.safe_dump(contents), encoding="utf-8")
    return path


def _write_extract_contract(directory: Path, required_columns: list[str]) -> Path:
    """Write a minimal site extract contract and return the path."""
    contract_path = directory / "site_extract_contract.yaml"
    contract = {
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
        specification = PerformanceComparisonSpecification(_BASELINE_COMPARISON_PATH)
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

    def test_transaction_category_is_inferred_from_transaction_code(self) -> None:
        """Transaction codes are labeled with conservative normalized categories."""
        specification = PerformanceComparisonSpecification(_BASELINE_COMPARISON_PATH)
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
        self.assertEqual(transaction_category_from_code("SELL"), "sell")
        self.assertEqual(transaction_category_from_code("not-a-real-code"), "unknown")

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
            specification = PerformanceComparisonSpecification(path)

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
            specification = PerformanceComparisonSpecification(path)

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
            specification = PerformanceComparisonSpecification(path)

            with self.assertRaises(PpaError) as context:
                TransactionsLoader(specification).load("a")

            message = str(context.exception)
            self.assertTrue(message.startswith("Error 504"))
            self.assertIn("unknown transaction codes or categories", message)
            self.assertIn("transaction_code=MYSTERY", message)

    def test_yaml_transaction_rules_fill_missing_semantics(self) -> None:
        """YAML transaction rules fill category and sign semantics by code."""
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
            specification = PerformanceComparisonSpecification(path)

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
                TRANSACTION_SEMANTICS_SOURCE_MIXED,
            )
            self.assertTrue(transaction_impact_semantics_available(row))

    def test_yaml_transaction_rules_do_not_override_source_semantics(self) -> None:
        """Recognized source semantics remain authoritative over YAML rules."""
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
            specification = PerformanceComparisonSpecification(path)

            frame = TransactionsLoader(specification).load("a")
            assert frame is not None
            row = frame.row(0, named=True)

            self.assertEqual(row[pc_cols.TRANSACTION_CATEGORY], TRANSACTION_CATEGORY_BUY)
            self.assertEqual(row[pc_cols.CASH_FLOW_SIGN], "negative")
            self.assertEqual(row[pc_cols.PERFORMANCE_FLOW_SIGN], "performance")
            self.assertEqual(
                row[pc_cols.TRANSACTION_SEMANTICS_SOURCE],
                TRANSACTION_SEMANTICS_SOURCE_SOURCE,
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
            specification = PerformanceComparisonSpecification(path)

            frame = TransactionsLoader(specification).load("a")
            assert frame is not None
            row = frame.row(0, named=True)

            self.assertEqual(
                row[pc_cols.TRANSACTION_SEMANTICS_SOURCE],
                TRANSACTION_SEMANTICS_SOURCE_YAML_RULE,
            )

    def test_yaml_transaction_rules_mark_mixed_semantics(self) -> None:
        """YAML-filled fields are tagged mixed when source fields are retained."""
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
            specification = PerformanceComparisonSpecification(path)

            frame = TransactionsLoader(specification).load("a")
            assert frame is not None
            row = frame.row(0, named=True)

            self.assertEqual(row[pc_cols.TRANSACTION_CATEGORY], TRANSACTION_CATEGORY_BUY)
            self.assertEqual(row[pc_cols.CASH_FLOW_SIGN], "negative")
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
                            "security_id": "CASH_USD",
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
                "SEC": ["CASH_USD", "CASH_USD", "AAPL", "AAPL", "CASH_USD"],
                "TRANSACTION_DATE": ["2025-01-31"] * 5,
                "TRAN": ["wd", "wd", "li", "lo", "dp"],
                "SEC_TYPE": ["cash", "cash", "eq", "eq", "cash"],
                "SRC_DEST_TYPE": ["$pty", "$sweep", "$pty", "$pty", "$pty"],
                "SRC_DEST_SYMBOL": ["$cash", "CASH_USD", "$cash", "$cash", "$cash"],
                "SPECIAL_SEC_TYPE": ["", "", "", "", "exus"],
                "SPECIAL_SEC_SYMBOL": ["", "", "", "", "custfee"],
            }
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                pl.DataFrame(rows).write_csv(directory / snapshot_name / "transactions.csv")
            path = _write_yaml(directory, configuration)
            specification = PerformanceComparisonSpecification(path)

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
        specification = PerformanceComparisonSpecification(
            _SITE_VARIANT_FIXTURES_PATH
            / "imex_context"
            / "ppar_performance_comparison.yaml"
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

    def test_site_variant_rep_semantics_can_supply_ambiguous_flow_context(self) -> None:
        """REP/report semantics can be the reviewed context for ambiguous codes."""
        specification = PerformanceComparisonSpecification(
            _SITE_VARIANT_FIXTURES_PATH
            / "rep_semantics"
            / "ppar_performance_comparison.yaml"
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

    def test_site_variant_code_only_imex_ambiguous_codes_fail_fast(self) -> None:
        """Code-only IMEX fixtures cannot classify ambiguous external flows."""
        specification = PerformanceComparisonSpecification(
            _SITE_VARIANT_FIXTURES_PATH
            / "imex_code_only"
            / "ppar_performance_comparison.yaml"
        )

        with self.assertRaises(PpaError) as context:
            TransactionsLoader(specification).load("a")

        message = str(context.exception)
        self.assertIn("ambiguous Axys transaction codes DP, LI, LO, WD", message)
        self.assertIn("IMEX transaction code alone is not enough", message)
        self.assertIn("REP/report extract", message)

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
                            "security_id": "CASH_USD",
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
                        "SEC": ["CASH_USD"],
                        "TRANSACTION_DATE": ["2025-01-31"],
                        "TRAN": ["wd"],
                    }
                ).write_csv(directory / snapshot_name / "transactions.csv")
            path = _write_yaml(directory, configuration)
            specification = PerformanceComparisonSpecification(path)

            with self.assertRaises(PpaError) as context:
                TransactionsLoader(specification).load("a")

            message = str(context.exception)
            self.assertIn("IMEX context fields", message)
            self.assertIn("REP/report extract", message)

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
                        "SEC": ["CASH_USD"],
                        "TRANSACTION_DATE": ["2025-01-31"],
                        "TRAN": ["wd"],
                    }
                ).write_csv(directory / snapshot_name / "transactions.csv")
            path = _write_yaml(directory, configuration)
            specification = PerformanceComparisonSpecification(path)

            with self.assertRaises(PpaError) as context:
                TransactionsLoader(specification).load("a")

            message = str(context.exception)
            self.assertIn("ambiguous Axys transaction codes WD", message)
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
                        "SEC": ["CASH_USD"],
                        "TRANSACTION_DATE": ["2025-01-31"],
                        "TRAN": ["wd"],
                        "SRC_DEST_TYPE": ["$pty"],
                    }
                ).write_csv(directory / snapshot_name / "transactions.csv")
            path = _write_yaml(directory, configuration)
            specification = PerformanceComparisonSpecification(path)

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
        """A config can intentionally disable ambiguous Axys flow enforcement."""
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
                        "SEC": ["CASH_USD"],
                        "TRANSACTION_DATE": ["2025-01-31"],
                        "TRAN": ["wd"],
                    }
                ).write_csv(directory / snapshot_name / "transactions.csv")
            path = _write_yaml(directory, configuration)
            specification = PerformanceComparisonSpecification(path)

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
            specification = PerformanceComparisonSpecification(path)

            with self.assertRaises(PpaError) as context:
                TransactionsLoader(specification).load("a")

            self.assertIn("transaction_rules must be a mapping", str(context.exception))

    def test_omitted_transactions_returns_none(self) -> None:
        """Transactions are optional when omitted from YAML."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            path = _write_yaml(directory, _minimal_specification(directory))
            specification = PerformanceComparisonSpecification(path)

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
            specification = PerformanceComparisonSpecification(path)

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
            specification = PerformanceComparisonSpecification(path)

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
            specification = PerformanceComparisonSpecification(path)

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
            specification = PerformanceComparisonSpecification(path)

            with self.assertRaises(PpaError) as context:
                TransactionsLoader(specification).load("a")

            message = str(context.exception)
            self.assertTrue(message.startswith("Error 502"))
            self.assertIn("transactions", message)
            self.assertIn("amount", message)
            self.assertIn("--", message)


if __name__ == "__main__":
    unittest.main()
