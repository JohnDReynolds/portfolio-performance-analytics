"""Catalog the intended Performance Auditing safety invariants.

This module is an executable design catalog, not an enforcement layer. It gives
later safety-net phases stable identifiers, failure classifications, and scoped
guarantees without changing comparison or report behavior in Phase 1.
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
        coverage: Current enforcement coverage after the Phase 1 audit.
        existing_controls: Current code or test controls that partially enforce it.
        known_gaps: Specific behavior still required for full enforcement.
        implementation_phase: Planned phase in the safety-net program.
    """

    identifier: str
    name: str
    guarantee: str
    failure_class: InvariantFailureClass
    coverage: InvariantCoverage
    existing_controls: tuple[str, ...]
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
        coverage=InvariantCoverage.PARTIAL,
        existing_controls=(
            "PerformanceComparison.compare emits normalized findings.",
            "findings.csv retains suppressed and unsuppressed findings.",
            "Unknown normalized fields default to review context.",
        ),
        known_gaps=(
            "No end-to-end assertion assigns every finding a visible disposition.",
            "source_detail.csv and primary reports exclude suppressed findings.",
            "Tolerance exclusions have no explicit audit record.",
        ),
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
        coverage=InvariantCoverage.PARTIAL,
        existing_controls=(
            "Field roles separate performance inputs from input components.",
            "Evidence-only and cross-check-only policies exclude support from totals.",
            "Workbook cause promotion selects counted and supporting rows separately.",
        ),
        known_gaps=(
            "No stable economic-effect identity or counted-cause ownership record.",
            "No universal assertion rejects two counted representations of one effect.",
        ),
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
        coverage=InvariantCoverage.PARTIAL,
        existing_controls=(
            "Portfolio cause totals reconcile before workbook construction.",
            "Displayed six-decimal values reconcile before serialization.",
            "Serialized portfolio workbook cells reconcile after construction.",
            "Modified Dietz formula components must remain visible in causes.",
        ),
        known_gaps=(
            "Equivalent security-grain assertions are not centralized.",
            "Semantic parity across HTML, XLSX, and CSV is not yet asserted.",
        ),
        implementation_phase=2,
    ),
    SafetyInvariant(
        identifier="SN-04",
        name="Beginning and ending continuity",
        guarantee=(
            "A prior ending value and the next beginning value either reconcile or "
            "produce a visible source-data review finding."
        ),
        failure_class=InvariantFailureClass.VISIBLE_REVIEW_FINDING,
        coverage=InvariantCoverage.PARTIAL,
        existing_controls=(
            "Carry-forward beginning-value differences remain visible in causes.",
            "Return reconstruction reports missing beginning or ending inputs.",
        ),
        known_gaps=(
            "No general within-snapshot continuity comparison exists across periods.",
            "Cash, accrued value, and market value do not share one continuity rule.",
        ),
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
        coverage=InvariantCoverage.PARTIAL,
        existing_controls=(
            "Findings retain dataset, source file, source column, keys, and dates.",
            "Cause rows retain reconstruction-component and policy metadata.",
        ),
        known_gaps=(
            "Findings lack a universal stable source-record locator.",
            "No reverse-lineage assertion covers every reportable finding.",
        ),
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
        coverage=InvariantCoverage.PARTIAL,
        existing_controls=(
            "Portfolio performance supplies authoritative base currency.",
            "Foreign Modified Dietz inputs require explicit base-currency values.",
            "FX rates retain explicit from-currency and to-currency direction.",
        ),
        known_gaps=(
            "Currency codes are normalized but not comprehensively validated.",
            "Local value, FX rate, and supplied base value are not reconciled.",
            "Unit metadata is not attached to every counted numeric value.",
        ),
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
        coverage=InvariantCoverage.PARTIAL,
        existing_controls=(
            "Modified Dietz validates beginning-of-day and end-of-day flow rules.",
            "Comparison keys and normalized dates are validated by source loaders.",
            "Demo transactions must map to exactly one inclusive performance period.",
        ),
        known_gaps=(
            "Production comparison lacks one exhaustive dated-evidence assignment audit.",
            "Unassigned and multiply assigned evidence lack a common visible treatment.",
        ),
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
        coverage=InvariantCoverage.PARTIAL,
        existing_controls=(
            "Independent scenario inventory rejects removed or unregistered keys.",
            "Scenario calendar and rebuild audits protect generated rows and residuals.",
            "Scenario-specific tests protect important transaction examples.",
        ),
        known_gaps=(
            "The inventory protects identity and reason, not a complete semantic contract.",
            "Expected report disposition is distributed across scripts and tests.",
        ),
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
        coverage=InvariantCoverage.PARTIAL,
        existing_controls=(
            "Scenario calendar records expected difference-row density by period.",
            "Period-split audit rejects periods above the current readability target.",
            "Generated causal-story coverage detects unregistered causal securities.",
        ),
        known_gaps=(
            "The current target permits one or two differences rather than a declared "
            "independent economic-change count.",
            "Carry-forward classification is explanatory text rather than fixture metadata.",
        ),
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
        coverage=InvariantCoverage.PARTIAL,
        existing_controls=(
            "Bundle validation checks required artifacts, headers, and CSV row counts.",
            "Workbook construction rechecks portfolio explanation arithmetic.",
            "Manifest and review-summary metadata are cross-validated.",
        ),
        known_gaps=(
            "No normalized semantic comparison spans every report format.",
            "Artifact validation checks shape more thoroughly than content parity.",
        ),
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
        coverage=InvariantCoverage.PARTIAL,
        existing_controls=(
            "Many reviewer tables apply explicit stable sort keys.",
            "JSON output uses sorted keys and schemas use fixed column order.",
        ),
        known_gaps=(
            "No repeat-run equivalence test covers normalized bundle content.",
            "Volatile timestamps and XLSX package metadata lack a canonical exclusion rule.",
        ),
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
        coverage=InvariantCoverage.PARTIAL,
        existing_controls=(
            "Unknown fields default to nonadditive context.",
            "Strict bundle generation rejects known fields without explicit YAML policy.",
            "Ambiguous transaction semantics and unsafe currency inputs fail closed.",
        ),
        known_gaps=(
            "Policy-completeness checks enumerate fields instead of deriving from roles.",
            "New fields can default to context without an explicit classification decision.",
        ),
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
