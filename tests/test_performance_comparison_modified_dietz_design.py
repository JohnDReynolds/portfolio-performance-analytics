"""Reference examples for Modified Dietz transaction impact diagnostics."""

# Python imports
from datetime import date
import unittest

# Project imports
from ppar.performance_comparison.modified_dietz import (
    modified_dietz_external_flow_impact,
    modified_dietz_flow_weight,
)


class TestModifiedDietzDesign(unittest.TestCase):
    """Document expected examples for Modified Dietz cross-check diagnostics."""

    def test_beginning_of_day_flow_uses_inclusive_remaining_days(self) -> None:
        """A mid-period flow uses inclusive remaining days by default."""
        impact = modified_dietz_external_flow_impact(
            flow_delta=300.0,
            denominator=10000.0,
            from_date=date(2025, 1, 1),
            thru_date=date(2025, 1, 30),
            flow_date=date(2025, 1, 11),
            inclusion_rule="beginning_of_day",
        )

        self.assertAlmostEqual(impact, 0.02)

    def test_zero_denominator_is_rejected(self) -> None:
        """The skeleton helper does not permit divide-by-zero estimates."""
        with self.assertRaises(ValueError):
            modified_dietz_external_flow_impact(
                flow_delta=300.0,
                denominator=0.0,
                from_date=date(2025, 1, 1),
                thru_date=date(2025, 1, 30),
                flow_date=date(2025, 1, 11),
                inclusion_rule="beginning_of_day",
            )

    def test_out_of_period_flow_date_is_rejected(self) -> None:
        """The skeleton helper requires flow dates inside the target period."""
        with self.assertRaises(ValueError):
            modified_dietz_flow_weight(
                from_date=date(2025, 1, 1),
                thru_date=date(2025, 1, 30),
                flow_date=date(2025, 1, 31),
                inclusion_rule="beginning_of_day",
            )

    def test_end_of_day_flow_excludes_the_flow_date(self) -> None:
        """End-of-day treatment reduces the same flow by one day of weight."""
        impact = modified_dietz_external_flow_impact(
            flow_delta=300.0,
            denominator=10000.0,
            from_date=date(2025, 1, 1),
            thru_date=date(2025, 1, 30),
            flow_date=date(2025, 1, 11),
            inclusion_rule="end_of_day",
        )

        self.assertAlmostEqual(impact, 0.019)

    def test_sign_flows_through_the_weighted_delta(self) -> None:
        """Withdrawals produce negative impacts under the same weighting rule."""
        impact = modified_dietz_external_flow_impact(
            flow_delta=-300.0,
            denominator=10000.0,
            from_date=date(2025, 1, 1),
            thru_date=date(2025, 1, 30),
            flow_date=date(2025, 1, 11),
            inclusion_rule="beginning_of_day",
        )

        self.assertAlmostEqual(impact, -0.02)

    def test_boundary_flows_match_inclusion_rule(self) -> None:
        """Boundary-date examples define beginning and ending period behavior."""
        from_date = date(2025, 1, 1)
        thru_date = date(2025, 1, 30)

        self.assertAlmostEqual(
            modified_dietz_flow_weight(
                from_date=from_date,
                thru_date=thru_date,
                flow_date=from_date,
                inclusion_rule="beginning_of_day",
            ),
            1.0,
        )
        self.assertAlmostEqual(
            modified_dietz_flow_weight(
                from_date=from_date,
                thru_date=thru_date,
                flow_date=thru_date,
                inclusion_rule="end_of_day",
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
