"""Tests for deterministic Audit workbook reviewer guidance."""

from __future__ import annotations

# Python imports
import datetime as dt
import unittest

# Project imports
from ppar.audit import schema as audit_schema
from ppar.audit import workbook_guidance
from ppar.audit import workbook_rows
from ppar.audit import workbook_source_allocation
from ppar.audit.performance_comparison import findings


class TestWorkbookGuidance(unittest.TestCase):
    """Verify direct reviewer-guidance wording contracts."""

    def test_possible_transaction_cause_names_configuration_action(self) -> None:
        """Possible transaction causes explain the change and required action."""
        row = {
            findings.DATASET: audit_schema.TRANSACTIONS,
            findings.SOURCE_COLUMN: audit_schema.AMOUNT,
            findings.SECURITY_ID: "AAPL",
            findings.TRANSACTION_CODE: "BUY",
            findings.DELTA_B_MINUS_A: 100.0,
            workbook_rows.POSSIBLE_CAUSE_ROW: True,
        }

        guidance = workbook_guidance.review_guidance(
            row,
            None,
            comparison_path=None,
            impact_status=workbook_guidance.IMPACT_STATUS_MISSING_METHOD,
            row_kind=workbook_rows.ROW_KIND_UNDERLYING_CAUSE,
        )

        self.assertEqual(
            guidance,
            (
                "BUY: AAPL transactions.amount increased by 100.00. "
                "Add YAML configuration to count it as explained."
            ),
        )

    def test_beginning_holding_guidance_is_concise(self) -> None:
        """Beginning holdings state the changed input without policy narration."""
        row = {
            findings.DATASET: audit_schema.HOLDINGS,
            findings.SOURCE_COLUMN: audit_schema.MARKET_VALUE,
            findings.SECURITY_ID: "AAPL",
            findings.INPUT_DATE: dt.date(2025, 4, 30),
            findings.FROM_DATE: dt.date(2025, 5, 1),
            findings.THRU_DATE: dt.date(2025, 5, 31),
            findings.DELTA_B_MINUS_A: 10.0,
        }

        guidance = workbook_guidance.review_guidance(
            row,
            0.01,
            comparison_path=None,
            impact_status=workbook_guidance.IMPACT_STATUS_ESTIMATED,
            row_kind=workbook_rows.ROW_KIND_UNDERLYING_CAUSE,
        )

        self.assertEqual(
            guidance,
            "AAPL beginning holdings.market_value increased by 10.00.",
        )

    def test_review_only_holding_quantity_names_affected_input(self) -> None:
        """Review-only holding quantities name their market-value relationship."""
        row = {
            findings.DATASET: audit_schema.HOLDINGS,
            findings.SOURCE_COLUMN: audit_schema.QUANTITY,
            findings.SECURITY_ID: "AAPL",
            findings.DELTA_B_MINUS_A: 10.0,
        }

        guidance = workbook_guidance.review_guidance(
            row,
            None,
            comparison_path=None,
            impact_status=workbook_guidance.IMPACT_STATUS_REVIEW_ONLY,
            row_kind=workbook_rows.ROW_KIND_UNDERLYING_CAUSE,
        )
        note = workbook_guidance.review_note(
            row,
            None,
            workbook_rows.USE_EXPLAINS_CHANGE,
            workbook_guidance.IMPACT_STATUS_REVIEW_ONLY,
        )

        expected = (
            "AAPL holdings.quantity increased by 10.00. This affects the performance "
            "calculation through holdings.market_value."
        )
        self.assertEqual(guidance, expected)
        self.assertEqual(note, expected)

    def test_review_only_holding_price_names_affected_input(self) -> None:
        """Review-only holding prices name their market-value relationship."""
        row = {
            findings.DATASET: audit_schema.HOLDINGS,
            findings.SOURCE_COLUMN: audit_schema.PRICE,
            findings.SECURITY_ID: "AAPL",
            findings.DELTA_B_MINUS_A: 3.12,
            workbook_rows.UNSELECTED_RELATED_ESTIMATE: True,
        }

        guidance = workbook_guidance.review_guidance(
            row,
            None,
            comparison_path=None,
            impact_status=workbook_guidance.IMPACT_STATUS_REVIEW_ONLY,
            row_kind=workbook_rows.ROW_KIND_UNDERLYING_CAUSE,
        )
        note = workbook_guidance.review_note(
            row,
            None,
            workbook_rows.USE_EXPLAINS_CHANGE,
            workbook_guidance.IMPACT_STATUS_REVIEW_ONLY,
        )

        expected = (
            "AAPL holdings.price increased by 3.12. This affects the performance "
            "calculation through holdings.market_value."
        )
        self.assertEqual(guidance, expected)
        self.assertEqual(note, expected)

    def test_transaction_cash_balance_names_source_currency_field(self) -> None:
        """Transaction cash effects name the value field matching their source basis."""
        cases = (
            (
                audit_schema.AMOUNT,
                "dv: transactions.amount increased by 32.40. This affects the "
                "performance calculation through cash-balance ending "
                "holdings.market_value.",
            ),
            (
                audit_schema.BASE_AMOUNT,
                "dv: transactions.base_amount increased by 32.40. This affects the "
                "performance calculation through cash-balance ending "
                "holdings.base_market_value.",
            ),
        )

        for source_column, expected in cases:
            with self.subTest(source_column=source_column):
                row = {
                    findings.DATASET: audit_schema.TRANSACTIONS,
                    findings.SOURCE_COLUMN: source_column,
                    findings.TRANSACTION_CODE: "dv",
                    findings.CASH_FLOW_SIGN: "positive",
                    findings.DELTA_B_MINUS_A: 32.4,
                    workbook_rows.NON_ADDITIVE_PORTFOLIO_TRANSACTION: True,
                }

                guidance = workbook_guidance.review_guidance(
                    row,
                    None,
                    comparison_path=None,
                    impact_status=workbook_guidance.IMPACT_STATUS_REVIEW_ONLY,
                    row_kind=workbook_rows.ROW_KIND_CONTEXT,
                )

                self.assertEqual(guidance, expected)

    def test_foreign_holding_components_name_base_currency_input(self) -> None:
        """Foreign holding components name the counted base-currency value."""
        for source_column, change in (
            (audit_schema.PRICE, 0.25),
            (audit_schema.QUANTITY, 30.0),
        ):
            with self.subTest(source_column=source_column):
                row = {
                    findings.DATASET: audit_schema.HOLDINGS,
                    findings.SOURCE_COLUMN: source_column,
                    findings.SECURITY_ID: "CASHEUR",
                    findings.DELTA_B_MINUS_A: change,
                    findings.IMPACT_POLICY: (
                        f"{findings.IMPACT_POLICY_EVIDENCE_ONLY_PREFIX}"
                        f"holdings.{source_column}_row_currency"
                    ),
                    workbook_rows.UNSELECTED_RELATED_ESTIMATE: True,
                }

                guidance = workbook_guidance.review_guidance(
                    row,
                    None,
                    comparison_path=None,
                    impact_status=workbook_guidance.IMPACT_STATUS_REVIEW_ONLY,
                    row_kind=workbook_rows.ROW_KIND_CONTEXT,
                )

                self.assertTrue(
                    guidance.endswith(
                        "This affects the performance calculation through "
                        "holdings.base_market_value."
                    )
                )

    def test_review_only_base_market_value_names_reflected_input(self) -> None:
        """Redundant base values identify the counted market-value representation."""
        row = {
            findings.DATASET: audit_schema.HOLDINGS,
            findings.SOURCE_COLUMN: audit_schema.BASE_MARKET_VALUE,
            findings.SECURITY_ID: "AAPL",
            findings.DELTA_B_MINUS_A: 10.0,
            findings.IMPACT_POLICY: "evidence_only:holdings.base_market_value_redundant",
        }

        guidance = workbook_guidance.review_guidance(
            row,
            None,
            comparison_path=None,
            impact_status=workbook_guidance.IMPACT_STATUS_REVIEW_ONLY,
            row_kind=workbook_rows.ROW_KIND_CONTEXT,
        )
        note = workbook_guidance.review_note(
            row,
            None,
            workbook_rows.USE_EXPLAINS_CHANGE,
            workbook_guidance.IMPACT_STATUS_REVIEW_ONLY,
        )

        expected = (
            "AAPL holdings.base_market_value increased by 10.00. This change is "
            "also reflected in the performance calculation through holdings.market_value."
        )
        self.assertEqual(guidance, expected)
        self.assertEqual(note, expected)

    def test_transaction_components_name_affected_performance_inputs(self) -> None:
        """Transaction components state their change and downstream input roles."""
        cases = (
            (
                audit_schema.QUANTITY,
                "AAPL",
                "by",
                "buy",
                1.1372,
                (
                    "by: AAPL transactions.quantity increased by 1.1372. This affects "
                    "the performance calculation through transactions.amount and "
                    "holdings.market_value."
                ),
            ),
            (
                audit_schema.QUANTITY,
                "MSFT",
                "sl",
                "sell",
                2.0,
                (
                    "sl: MSFT transactions.quantity increased by 2.00. This affects the "
                    "performance calculation through transactions.amount and "
                    "holdings.market_value."
                ),
            ),
            (
                audit_schema.QUANTITY,
                "TSLA",
                "ss",
                "sell",
                50.0,
                (
                    "ss: TSLA transactions.quantity increased by 50.00. This affects the "
                    "performance calculation through transactions.amount."
                ),
            ),
            (
                audit_schema.QUANTITY,
                "TSLA",
                "cs",
                "buy",
                50.0,
                (
                    "cs: TSLA transactions.quantity increased by 50.00. This affects the "
                    "performance calculation through transactions.amount."
                ),
            ),
            (
                audit_schema.PRICE,
                "AAPL",
                "by",
                "buy",
                0.15,
                (
                    "by: AAPL transactions.price increased by 0.15. This affects the "
                    "performance calculation through transactions.amount."
                ),
            ),
            (
                audit_schema.COMMISSION,
                "AAPL",
                "by",
                "buy",
                10.0,
                (
                    "by: AAPL transactions.commission increased by 10.00. This affects "
                    "the performance calculation through transactions.amount."
                ),
            ),
        )

        for source_column, security_id, code, category, change, expected in cases:
            with self.subTest(source_column=source_column, code=code):
                row = {
                    findings.DATASET: audit_schema.TRANSACTIONS,
                    findings.SOURCE_COLUMN: source_column,
                    findings.SECURITY_ID: security_id,
                    findings.TRANSACTION_CODE: code,
                    findings.TRANSACTION_CATEGORY: category,
                    findings.DELTA_B_MINUS_A: change,
                }
                guidance = workbook_guidance.review_guidance(
                    row,
                    None,
                    comparison_path=None,
                    impact_status=workbook_guidance.IMPACT_STATUS_REVIEW_ONLY,
                    row_kind=workbook_rows.ROW_KIND_CONTEXT,
                )

                self.assertEqual(guidance, expected)

    def test_fx_support_guidance_names_rate_and_counted_field(self) -> None:
        """FX support guidance connects the rate to the counted base value."""
        row = {
            findings.DATASET: audit_schema.FX_RATES,
            findings.SOURCE_COLUMN: audit_schema.FX_RATE,
            findings.SECURITY_ID: "SAP",
            findings.FROM_CURRENCY: "EUR",
            findings.TO_CURRENCY: "USD",
            findings.SNAPSHOT_A_VALUE: 1.1,
            findings.SNAPSHOT_B_VALUE: 1.2,
            workbook_source_allocation.FX_RATE_TARGET_FIELD: "holdings.base_market_value",
            workbook_source_allocation.FX_RATE_BASE_VALUE_CHANGE: 25.0,
            workbook_source_allocation.FX_RATE_SUPPORTS_BASE_INPUT: True,
        }

        guidance = workbook_guidance.review_guidance(
            row,
            None,
            comparison_path=None,
            impact_status=workbook_guidance.IMPACT_STATUS_REVIEW_ONLY,
            row_kind=workbook_rows.ROW_KIND_CONTEXT,
        )

        self.assertEqual(
            guidance,
            (
                "EUR-to-USD fx_rates.fx_rate increased by 0.10, from 1.1 to 1.2 "
                "USD per EUR. This affects the performance calculation through "
                "holdings.base_market_value; the counted USD effect for SAP is 25.00."
            ),
        )

    def test_explanation_contract_rejects_wrong_source_and_currency(self) -> None:
        """The semantic contract flags omitted source facts and wrong value basis."""
        row = {
            findings.DATASET: audit_schema.TRANSACTIONS,
            findings.SOURCE_COLUMN: audit_schema.BASE_AMOUNT,
            findings.SECURITY_ID: "SAP",
            findings.TRANSACTION_CODE: "dv",
            findings.DELTA_B_MINUS_A: 32.4,
            workbook_rows.NON_ADDITIVE_PORTFOLIO_TRANSACTION: True,
        }

        issues = workbook_guidance.explanation_contract_issues(
            row,
            (
                "dv: Caused cash-balance ending holdings.market_value to increase "
                "by 32.40."
            ),
            impact_status=workbook_guidance.IMPACT_STATUS_REVIEW_ONLY,
            comparison_path=None,
        )

        self.assertTrue(any("must begin" in issue for issue in issues))
        self.assertTrue(any("holdings.base_market_value" in issue for issue in issues))
        self.assertTrue(any("local-currency language" in issue for issue in issues))

    def test_explanation_contract_accepts_canonical_transaction(self) -> None:
        """Canonical transaction explanations satisfy every structured check."""
        row = {
            findings.DATASET: audit_schema.TRANSACTIONS,
            findings.SOURCE_COLUMN: audit_schema.QUANTITY,
            findings.SECURITY_ID: "AAPL",
            findings.TRANSACTION_CODE: "by",
            findings.TRANSACTION_CATEGORY: "buy",
            findings.DELTA_B_MINUS_A: 1.1372,
        }
        explanation = workbook_guidance.review_guidance(
            row,
            None,
            comparison_path=None,
            impact_status=workbook_guidance.IMPACT_STATUS_REVIEW_ONLY,
            row_kind=workbook_rows.ROW_KIND_CONTEXT,
        )

        self.assertEqual(
            workbook_guidance.explanation_contract_issues(
                row,
                explanation,
                impact_status=workbook_guidance.IMPACT_STATUS_REVIEW_ONLY,
                comparison_path=None,
            ),
            (),
        )

    def test_possible_cause_summary_preserves_configuration_instruction(self) -> None:
        """Possible-cause summaries retain the configuration instruction."""
        self.assertEqual(
            workbook_guidance.possible_cause_summary(["holdings.market_value changed."]),
            (
                "Possible cause: holdings.market_value changed. "
                "Add YAML configuration to count it as explained."
            ),
        )


if __name__ == "__main__":
    unittest.main()
