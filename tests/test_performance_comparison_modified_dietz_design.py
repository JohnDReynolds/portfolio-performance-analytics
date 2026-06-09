"""Design-only examples for future Modified Dietz transaction impact support."""

# Python imports
from datetime import date
import unittest


def _modified_dietz_flow_weight(
    *,
    from_date: date,
    thru_date: date,
    flow_date: date,
    inclusion_rule: str,
) -> float:
    """Return the design reference flow weight for a Modified Dietz example."""
    period_days = (thru_date - from_date).days + 1
    if period_days <= 0:
        raise ValueError("period must include at least one day")
    if not from_date <= flow_date <= thru_date:
        raise ValueError("flow_date must be inside the period")

    remaining_days = (thru_date - flow_date).days
    if inclusion_rule == "beginning_of_day":
        remaining_days += 1
    elif inclusion_rule != "end_of_day":
        raise ValueError("inclusion_rule must be beginning_of_day or end_of_day")
    return remaining_days / period_days


def _modified_dietz_external_flow_impact(
    *,
    flow_delta: float,
    denominator: float,
    from_date: date,
    thru_date: date,
    flow_date: date,
    inclusion_rule: str,
) -> float:
    """Return the design reference impact for one external-flow delta."""
    if denominator == 0:
        raise ValueError("denominator must be nonzero")
    weight = _modified_dietz_flow_weight(
        from_date=from_date,
        thru_date=thru_date,
        flow_date=flow_date,
        inclusion_rule=inclusion_rule,
    )
    return flow_delta * weight / denominator


class TestModifiedDietzDesign(unittest.TestCase):
    """Document expected examples for future Modified Dietz implementation."""

    def test_beginning_of_day_flow_uses_inclusive_remaining_days(self) -> None:
        """A mid-period flow uses inclusive remaining days by default."""
        impact = _modified_dietz_external_flow_impact(
            flow_delta=300.0,
            denominator=10000.0,
            from_date=date(2025, 1, 1),
            thru_date=date(2025, 1, 30),
            flow_date=date(2025, 1, 11),
            inclusion_rule="beginning_of_day",
        )

        self.assertAlmostEqual(impact, 0.02)

    def test_end_of_day_flow_excludes_the_flow_date(self) -> None:
        """End-of-day treatment reduces the same flow by one day of weight."""
        impact = _modified_dietz_external_flow_impact(
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
        impact = _modified_dietz_external_flow_impact(
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
            _modified_dietz_flow_weight(
                from_date=from_date,
                thru_date=thru_date,
                flow_date=from_date,
                inclusion_rule="beginning_of_day",
            ),
            1.0,
        )
        self.assertAlmostEqual(
            _modified_dietz_flow_weight(
                from_date=from_date,
                thru_date=thru_date,
                flow_date=thru_date,
                inclusion_rule="end_of_day",
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
