# Audit Safety Invariants

This is a maintainer-facing contract for the safety-net implementation program.
It defines what the protections mean, audits the current enforcement baseline,
and prevents later phases from changing the meaning of a safety net while
implementing it. It is not another user onboarding document.

The executable catalog is
`ppar.audit.safety_invariants.SAFETY_INVARIANTS`. This document
explains the design decisions behind that catalog.

## Current Guarantee Summary

All twelve cataloged invariants are enforced. Together they provide these
end-to-end guarantees:

- every reportable source difference remains in the complete audit trail as a
  counted cause or visible review evidence;
- one economic effect cannot own more than one explained amount, and portfolio
  and security explanation arithmetic must reconcile internally, at displayed
  precision, and after workbook serialization;
- continuity, currency, unit, and period-boundary controls prevent unsafe
  financial inputs from being counted silently;
- stable source locators and persisted cause lineage support forward and reverse
  traceability;
- every compared field has an explicit accounting role and policy treatment;
- protected demo scenarios, paired accounting legs, isolation limits, and
  carry-forward effects remain executable maintenance contracts; and
- typed artifact fingerprints, cross-format parity, normalized bundle
  fingerprints, and repeat-run checks protect output semantics and determinism.

The executable catalog and its focused tests own detailed control mappings. The
definitions and catalog below explain the stable maintainer contract without
repeating the completed implementation chronology.

## Definitions

### Reportable source difference

A **reportable source difference** is any Snapshot A versus Snapshot B change
emitted as a finding after normalization, record matching, and application of
the relevant field tolerance. It includes:

- row additions and removals;
- numeric field changes; and
- nonnumeric field changes.

This is the safety-net meaning of "material difference." It is independent of
the finding's `severity` and does not claim accounting or financial-statement
materiality. A source field that was never supplied cannot be protected as a
detected difference. A change discarded below comparison tolerance is outside
the current reportable-difference boundary; later implementation must decide
how tolerance exclusions are summarized for audit.

### Permitted dispositions

Every reportable source difference must remain visibly represented as one of:

1. `counted_cause`: contributes exactly once to explained performance; or
2. `review_evidence`: remains visible but contributes no explained amount.

Suppression is metadata, not a third disposition. It may affect review priority,
but it must never erase a difference from the complete audit trail. A source
contract error can stop processing before dispositions are produced when the
system cannot safely interpret the input.

### Counted economic effect

A counted economic effect is the financial change represented by one or more
source rows. Several rows may describe the same effect—such as transaction
amount and resulting cash, local value and base value, or quantity/price and
market value. Those rows may all remain visible, but exactly one designated
representation may own the explained amount.

### Complete audit trail

The complete audit trail is the lossless finding-level artifact from which
reviewer-facing summaries are derived. `findings.csv` implements this definition
because it includes suppressed and unsuppressed findings. Its sequence,
fingerprint, and disposition controls prove that downstream filtering cannot
remove the last visible representation of a reportable difference.

## Failure Classifications

- `internal_logic_error`: the program produced mutually inconsistent results or
  lost or counted evidence incorrectly. Stop generation with an unexpected-logic
  error; never downgrade it to a reviewer warning.
- `source_contract_error`: supplied data or configuration cannot be interpreted
  safely. Stop the affected workflow with an actionable input/configuration
  error.
- `visible_review_finding`: the source may be valid but has a suspicious
  relationship requiring judgment. Generate the report and display the issue;
  do not count it without a separate explicit rule.
- `demo_maintenance_error`: checked-in demo intent, isolation, or generated data
  drifted. Fail source-checkout maintenance/tests; do not produce a
  production-client finding.

These classifications are intentionally separate. For example, a discontinuity
between two source periods is normally a visible review finding; failure to
display that already-detected discontinuity is an internal logic error.

## Audited Invariant Catalog

| ID | Invariant | Coverage | Failure class | Phase |
| --- | --- | --- | --- | --- |
| `SN-01` | No lost differences | Enforced | `internal_logic_error` | 2 |
| `SN-02` | No double counting | Enforced | `internal_logic_error` | 2 |
| `SN-03` | Fully Explained arithmetic | Enforced | `internal_logic_error` | 2 |
| `SN-04` | Beginning and ending continuity | Enforced | `visible_review_finding` | 3 |
| `SN-05` | Bidirectional source lineage | Enforced | `internal_logic_error` | 4 |
| `SN-06` | Currency and unit consistency | Enforced | `source_contract_error` | 3 |
| `SN-07` | Period-boundary safety | Enforced | `source_contract_error` | 3 |
| `SN-08` | Demo scenario preservation | Enforced | `demo_maintenance_error` | 5 |
| `SN-09` | Demo fixture isolation | Enforced | `demo_maintenance_error` | 5 |
| `SN-10` | Report-format parity | Enforced | `internal_logic_error` | 6 |
| `SN-11` | Deterministic output | Enforced | `internal_logic_error` | 6 |
| `SN-12` | Fail-closed policy coverage | Enforced | `source_contract_error` | 4 |

The executable catalog records the audited foundations and gaps for every row.

## Existing Enforcement Map

The main current controls are:

- `workbook_tables._assert_portfolio_explanation_invariants`: internal cause
  totals, Fully Explained equality, and Modified Dietz component visibility;
- `workbook_tables._assert_displayed_portfolio_explanation_reconciliation`:
  six-decimal displayed arithmetic;
- `workbook._assert_written_portfolio_explanations_reconcile`: serialized XLSX
  arithmetic;
- `runner.validate_yaml_setup_complete` and
  `runner.validate_causal_attribution_ready`: strict configuration and impact
  coverage boundaries;
- `field_roles.assert_comparison_fields_classified` and
  `field_roles.requires_explicit_impact_policy`: explicit role registry and
  role-derived fail-closed policy coverage;
- `lineage.assert_finding_source_lineage` and
  `lineage.assert_bidirectional_report_lineage`: stable source identity and
  forward/reverse report lineage;
- `base_currency.with_authoritative_base_currency` and
  `currency_basis.base_currency_monetary_value`: authoritative currency and
  unsafe foreign-value fallback protection;
- `bundle.report_bundle_validation_issues` and `output_integrity`: artifact
  presence, shape, typed CSV semantics, cross-format display parity, and
  normalized bundle fingerprints; and
- `rebuild_audit_demo_data.audit_demo_data`: generated-data
  drift, residual, inventory, calendar, density, and scenario coverage guards.

The catalog and focused tests identify which combination of controls proves each
broader invariant.

## Change-Control Rule

Any later change to an invariant's guarantee, material-difference boundary,
permitted dispositions, or failure classification requires an explicit design
decision and corresponding catalog/test update. An implementation phase may
improve `coverage`; it must not silently weaken the guarantee.
