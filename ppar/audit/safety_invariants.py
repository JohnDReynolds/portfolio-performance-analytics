"""Catalog the intended Audit safety invariants.

This module is an executable design catalog, not an enforcement layer. It gives
the completed safety-net program stable identifiers, failure classifications,
scoped guarantees, and links to the controls that enforce them.
"""

from __future__ import annotations

# Python imports
from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class InvariantFailureClass(StrEnum):
    """Describe how a failed safety invariant must surface."""

    INTERNAL_LOGIC_ERROR = "internal_logic_error"
    SOURCE_CONTRACT_ERROR = "source_contract_error"
    VISIBLE_REVIEW_FINDING = "visible_review_finding"
    DEMO_MAINTENANCE_ERROR = "demo_maintenance_error"


class InvariantCoverage(StrEnum):
    """Describe the current implementation coverage for an invariant."""

    ENFORCED = "enforced"
    PARTIAL = "partial"
    NOT_ENFORCED = "not_enforced"
    RETIRED = "retired"


class DifferenceDisposition(StrEnum):
    """Describe the permitted visible treatment of a reportable difference."""

    COUNTED_CAUSE = "counted_cause"
    REVIEW_EVIDENCE = "review_evidence"


@dataclass(frozen=True)
class SafetyInvariant:
    """Describe one stable safety-net objective and its audited baseline.

    Attributes:
        identifier: Stable identifier used by documentation and future tests.
        name: Short reviewer-oriented name.
        guarantee: State that must hold when the invariant is fully enforced.
        failure_class: Primary way a violation must surface.
        coverage: Current enforcement coverage.
        existing_controls: Current code or test controls that enforce it.
        control_tests: Pytest node IDs that exercise the stated controls.
        known_gaps: Specific behavior still required for full enforcement.
        implementation_phase: Implementation phase in the safety-net program.
    """

    identifier: str
    name: str
    guarantee: str
    failure_class: InvariantFailureClass
    coverage: InvariantCoverage
    existing_controls: tuple[str, ...]
    control_tests: tuple[str, ...]
    known_gaps: tuple[str, ...]
    implementation_phase: int


MATERIAL_DIFFERENCE_DEFINITION: Final[str] = (
    "A reportable source difference is any Snapshot A versus Snapshot B change "
    "emitted as a finding after normalization, record matching, and the applicable "
    "field tolerance. It includes row additions, row removals, numeric changes, and "
    "nonnumeric changes. This safety-net meaning is independent of finding severity "
    "and financial-statement materiality."
)

DIFFERENCE_DISPOSITION_RULE: Final[str] = (
    "Every reportable source difference must remain visibly represented as either "
    "a counted cause or review evidence. Suppression metadata may change review "
    "priority, but suppression is not a permitted disposition and must not erase the "
    "difference from the complete audit trail."
)


SAFETY_INVARIANTS: Final[tuple[SafetyInvariant, ...]] = (
    SafetyInvariant(
        identifier="SN-01",
        name="No lost differences",
        guarantee=(
            "Every reportable source difference has a visible disposition and no "
            "difference silently disappears between comparison and report output."
        ),
        failure_class=InvariantFailureClass.INTERNAL_LOGIC_ERROR,
        coverage=InvariantCoverage.ENFORCED,
        existing_controls=(
            "findings.csv retains every suppressed and unsuppressed source finding.",
            "Every persisted finding has a sequence, fingerprint, and disposition.",
            "Every cause row is preserved while its disposition is assigned.",
            "Bundle validation rejects missing or invalid disposition metadata.",
        ),
        control_tests=(
            "tests/test_audit_conservation.py::TestAuditConservation::"
            "test_finding_audit_trail_is_lossless_and_includes_suppressed_rows",
        ),
        known_gaps=(),
        implementation_phase=2,
    ),
    SafetyInvariant(
        identifier="SN-02",
        name="No double counting",
        guarantee=(
            "One economic effect contributes to explained performance at most once, "
            "even when several source rows describe it."
        ),
        failure_class=InvariantFailureClass.INTERNAL_LOGIC_ERROR,
        coverage=InvariantCoverage.ENFORCED,
        existing_controls=(
            "Field roles separate performance inputs from input components.",
            "Evidence-only and cross-check-only policies exclude support from totals.",
            "Every cause has an economic-effect ID and at most one counted owner.",
            "Support-only representations are prohibited from owning an explanation.",
            "Cash holdings count value once while retaining quantity as evidence.",
        ),
        control_tests=(
            "tests/test_audit_conservation.py::TestAuditConservation::"
            "test_no_double_counting_rejects_two_owners_for_one_effect",
        ),
        known_gaps=(),
        implementation_phase=2,
    ),
    SafetyInvariant(
        identifier="SN-03",
        name="Fully Explained arithmetic",
        guarantee=(
            "A Fully Explained result has complete numeric causes whose total equals "
            "Explained Difference and Performance Difference within declared precision."
        ),
        failure_class=InvariantFailureClass.INTERNAL_LOGIC_ERROR,
        coverage=InvariantCoverage.ENFORCED,
        existing_controls=(
            "Portfolio and security cause totals reconcile before construction.",
            "Displayed six-decimal values reconcile before serialization.",
            "Serialized portfolio and security workbook cells reconcile.",
            "Modified Dietz formula components must remain visible in causes.",
        ),
        control_tests=(
            "tests/test_audit_workbook_contract.py::TestAuditWorkbookContract::"
            "test_portfolio_explanation_invariant_rejects_arithmetic_mismatch",
        ),
        known_gaps=(),
        implementation_phase=2,
    ),
    SafetyInvariant(
        identifier="SN-04",
        name="Retired performance-file continuity",
        guarantee=(
            "The former performance-file beginning/end market-value continuity "
            "contract is retired and cannot silently reappear as an active finding."
        ),
        failure_class=InvariantFailureClass.VISIBLE_REVIEW_FINDING,
        coverage=InvariantCoverage.RETIRED,
        existing_controls=(
            "Optional performance-file market values are not loaded as Audit inputs.",
            "Retired continuity issue types are rejected by the strict Data Issues schema.",
            "Holdings and transactions remain the authoritative valuation and flow evidence.",
        ),
        control_tests=(
            "tests/test_audit_financial_integrity.py::TestAuditFinancialIntegrity::"
            "test_retired_performance_values_are_ignored",
        ),
        known_gaps=(),
        implementation_phase=3,
    ),
    SafetyInvariant(
        identifier="SN-05",
        name="Bidirectional source lineage",
        guarantee=(
            "Every report cause traces to source evidence and every reportable source "
            "difference traces forward to its report disposition."
        ),
        failure_class=InvariantFailureClass.INTERNAL_LOGIC_ERROR,
        coverage=InvariantCoverage.ENFORCED,
        existing_controls=(
            "Every finding has a stable logical source-record locator.",
            "Source-backed causes retain source-finding fingerprints.",
            "Derived formula and no-cause disposition rows have explicit lineage types.",
            "Bundle validation checks persisted findings and cause-lineage artifacts.",
        ),
        control_tests=(
            "tests/test_audit_lineage.py::TestAuditLineage::"
            "test_generated_findings_and_causes_have_bidirectional_lineage",
        ),
        known_gaps=(),
        implementation_phase=4,
    ),
    SafetyInvariant(
        identifier="SN-06",
        name="Currency and unit consistency",
        guarantee=(
            "Every counted monetary value has a proven currency basis and values with "
            "incompatible units are never added."
        ),
        failure_class=InvariantFailureClass.SOURCE_CONTRACT_ERROR,
        coverage=InvariantCoverage.ENFORCED,
        existing_controls=(
            "Portfolio performance supplies authoritative base currency.",
            "Foreign Modified Dietz inputs require explicit base-currency values.",
            "All supplied currency codes are normalized and shape-validated.",
            "Foreign countable values require explicit base-currency counterparts.",
            "Same-currency local/base values must agree.",
        ),
        control_tests=(
            "tests/test_audit_financial_integrity.py::TestAuditFinancialIntegrity::"
            "test_foreign_countable_value_requires_explicit_base_value",
        ),
        known_gaps=(),
        implementation_phase=3,
    ),
    SafetyInvariant(
        identifier="SN-07",
        name="Period-boundary safety",
        guarantee=(
            "Every dated input is assigned according to an explicit period and timing "
            "rule, and out-of-period evidence cannot be silently counted."
        ),
        failure_class=InvariantFailureClass.SOURCE_CONTRACT_ERROR,
        coverage=InvariantCoverage.ENFORCED,
        existing_controls=(
            "Modified Dietz validates beginning-of-day and end-of-day flow rules.",
            "Comparison keys and normalized dates are validated by source loaders.",
            "Reversed and overlapping performance periods fail the source contract.",
            "Changed dated evidence is audited across every supported detailed dataset.",
            "Multiply assigned evidence fails and unassigned evidence cannot own impact.",
            "Prior-day holdings and FX values are the only counted beginning boundary.",
        ),
        control_tests=(
            "tests/test_audit_financial_integrity.py::TestAuditFinancialIntegrity::"
            "test_overlapping_performance_periods_fail_source_contract",
        ),
        known_gaps=(),
        implementation_phase=3,
    ),
    SafetyInvariant(
        identifier="SN-08",
        name="Demo scenario preservation",
        guarantee=(
            "Every protected demo scenario retains its intended economic meaning, "
            "portfolio, period, source change, and reviewer-visible result."
        ),
        failure_class=InvariantFailureClass.DEMO_MAINTENANCE_ERROR,
        coverage=InvariantCoverage.ENFORCED,
        existing_controls=(
            "Validated inventory protects economic meaning and calendar semantics.",
            "Source-input dates must remain in each scenario's declared source period.",
            "Actual report status and disposition must match the scenario contract.",
            "Scenario-specific tests protect important transaction examples.",
        ),
        control_tests=(
            "tests/test_audit_demo_data.py::TestAuditDemoData::"
            "test_protected_inventory_detects_removed_or_unregistered_scenarios",
        ),
        known_gaps=(),
        implementation_phase=5,
    ),
    SafetyInvariant(
        identifier="SN-09",
        name="Demo fixture isolation",
        guarantee=(
            "Each demo period stays within its declared independent-change budget, while "
            "carry-forward effects remain visible and are not treated as new changes."
        ),
        failure_class=InvariantFailureClass.DEMO_MAINTENANCE_ERROR,
        coverage=InvariantCoverage.ENFORCED,
        existing_controls=(
            "Fixture metadata assigns scenario rows to independent economic-change IDs.",
            "Source-period counts fail when they exceed the two-change budget.",
            "Paired accounting legs share one economic identity.",
            "Carry-forward effects are explicit and must remain visible in later causes.",
        ),
        control_tests=(
            "tests/test_audit_demo_data.py::TestAuditDemoData::"
            "test_scenario_independent_change_contract_fails_closed",
        ),
        known_gaps=(),
        implementation_phase=5,
    ),
    SafetyInvariant(
        identifier="SN-10",
        name="Report-format parity",
        guarantee=(
            "HTML, XLSX, CSV, and internal tables agree on statuses, totals, causes, and "
            "visible review evidence."
        ),
        failure_class=InvariantFailureClass.INTERNAL_LOGIC_ERROR,
        coverage=InvariantCoverage.ENFORCED,
        existing_controls=(
            "Every review sheet has a canonical internal display fingerprint.",
            "Every persisted review table has a typed semantic CSV fingerprint.",
            "Bundle validation compares HTML and XLSX with canonical review content.",
            "Content mutation tests cover CSV, HTML, XLSX, and manifest drift.",
        ),
        control_tests=(
            "tests/test_audit_report.py::TestAuditReport::"
            "test_report_bundle_validation_catches_cross_format_content_drift",
        ),
        known_gaps=(),
        implementation_phase=6,
    ),
    SafetyInvariant(
        identifier="SN-11",
        name="Deterministic output",
        guarantee=(
            "The same normalized inputs and configuration produce identical findings, "
            "ordering, statuses, and financial values after volatile metadata is removed."
        ),
        failure_class=InvariantFailureClass.INTERNAL_LOGIC_ERROR,
        coverage=InvariantCoverage.ENFORCED,
        existing_controls=(
            "The manifest records ordered typed table and display fingerprints.",
            "The public contract declares the exact volatile metadata exclusions.",
            "A normalized bundle fingerprint covers all nonvolatile manifest semantics.",
            "Repeat-run tests compare normalized manifests and deterministic artifacts.",
        ),
        control_tests=(
            "tests/test_audit_report.py::TestAuditReport::"
            "test_report_bundle_generation_is_semantically_repeatable",
        ),
        known_gaps=(),
        implementation_phase=6,
    ),
    SafetyInvariant(
        identifier="SN-12",
        name="Fail-closed policy coverage",
        guarantee=(
            "A new performance-affecting field cannot become a counted explanation until "
            "its unit, sign, timing, role, and attribution policy are explicit."
        ),
        failure_class=InvariantFailureClass.SOURCE_CONTRACT_ERROR,
        coverage=InvariantCoverage.ENFORCED,
        existing_controls=(
            "Every comparison-surface field must have an explicit accounting role.",
            "Impact-policy requirements derive from performance-input roles.",
            "Unknown changed fields fail even when suppressed by YAML.",
            "Ambiguous transaction semantics and unsafe currency inputs fail closed.",
        ),
        control_tests=(
            "tests/test_audit_lineage.py::TestAuditLineage::"
            "test_policy_requirement_is_derived_from_field_role",
        ),
        known_gaps=(),
        implementation_phase=4,
    ),
)


def safety_invariant(identifier: str) -> SafetyInvariant:
    """Return one catalog entry by stable identifier.

    Args:
        identifier: Stable safety invariant identifier such as ``SN-03``.

    Returns:
        Matching invariant catalog entry.

    Raises:
        KeyError: If the identifier is not present in the catalog.
    """
    for invariant in SAFETY_INVARIANTS:
        if invariant.identifier == identifier:
            return invariant
    raise KeyError(identifier)
