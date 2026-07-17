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

    def test_beginning_holding_guidance_names_carry_forward(self) -> None:
        """Beginning holdings identify their prior-period origin."""
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
            (
                "Inherited beginning-value difference from the preceding period: "
                "AAPL beginning holdings.market_value increased by 10.00. "
                "This value is retained because it is an input to Modified Dietz."
            ),
        )

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
                "EUR-to-USD FX rate changed from 1.1 to 1.2 USD per EUR; "
                "SAP holdings.base_market_value shows the counted USD effect of 25.00."
            ),
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
