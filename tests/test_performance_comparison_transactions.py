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
from ppar.performance_comparison import columns as pc_cols
from ppar.performance_comparison.transactions import (
    TRANSACTION_CASH_FLOW_SIGN_NEGATIVE,
    TRANSACTION_CASH_FLOW_SIGN_NONE,
    TRANSACTION_CASH_FLOW_SIGN_POSITIVE,
    TRANSACTION_CASH_FLOW_SIGN_UNKNOWN,
    TRANSACTION_CATEGORY_BUY,
    TRANSACTION_CATEGORY_EXTERNAL_FLOW,
    TRANSACTION_CATEGORY_INCOME,
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

_BASELINE_COMPARISON_PATH = Path("tests/data/axys/ppar_performance_comparison.yaml")


def _write_yaml(directory: Path, contents: object) -> Path:
    """Write comparison YAML contents and return the path."""
    path = directory / "ppar_performance_comparison.yaml"
    path.write_text(yaml.safe_dump(contents), encoding="utf-8")
    return path


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
