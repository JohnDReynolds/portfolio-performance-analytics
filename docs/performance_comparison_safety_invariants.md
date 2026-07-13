# Performance Auditing Safety Invariants

This is a maintainer-facing contract for the safety-net implementation program.
It defines what the protections mean, audits the current enforcement baseline,
and prevents later phases from changing the meaning of a safety net while
implementing it. It is not another user onboarding document.

The executable catalog is
`ppar.performance_comparison.safety_invariants.SAFETY_INVARIANTS`. This document
explains the design decisions behind that catalog.

## Phase 1 Outcome

Phase 1 defines and audits the invariants without changing comparison or report
behavior. The audit found substantial existing foundations, but every broad
invariant is currently partial because each has at least one end-to-end gap.

The strongest existing protection is portfolio explanation arithmetic:

- cause totals reconcile before workbook construction;
- displayed six-decimal totals reconcile before serialization;
- serialized workbook cells reconcile after construction; and
- Modified Dietz components with estimated impact must remain visible in the
  causes table.

The most important missing protection is a universal disposition assertion:
the system does not yet prove that every detected source difference reaches a
counted cause or visible review-evidence destination.

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
reviewer-facing summaries are derived. Today, `findings.csv` is closest to this
definition because it includes suppressed and unsuppressed findings. Phase 2
must make the contract explicit and prove that downstream filtering cannot
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

| ID | Invariant | Failure class | Phase |
| --- | --- | --- | --- |
| `SN-01` | No lost differences | `internal_logic_error` | 2 |
| `SN-02` | No double counting | `internal_logic_error` | 2 |
| `SN-03` | Fully Explained arithmetic | `internal_logic_error` | 2 |
| `SN-04` | Beginning and ending continuity | `visible_review_finding` | 3 |
| `SN-05` | Bidirectional source lineage | `internal_logic_error` | 4 |
| `SN-06` | Currency and unit consistency | `source_contract_error` | 3 |
| `SN-07` | Period-boundary safety | `source_contract_error` | 3 |
| `SN-08` | Demo scenario preservation | `demo_maintenance_error` | 5 |
| `SN-09` | Demo fixture isolation | `demo_maintenance_error` | 5 |
| `SN-10` | Report-format parity | `internal_logic_error` | 6 |
| `SN-11` | Deterministic output | `internal_logic_error` | 6 |
| `SN-12` | Fail-closed policy coverage | `source_contract_error` | 4 |

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
- `field_roles.field_role`: nonadditive default for unknown fields;
- `base_currency.with_authoritative_base_currency` and
  `currency_basis.base_currency_monetary_value`: authoritative currency and
  unsafe foreign-value fallback protection;
- `bundle.report_bundle_validation_issues`: artifact presence, shape, metadata,
  headers, and row counts; and
- `rebuild_performance_comparison_demo_data.audit_demo_data`: generated-data
  drift, residual, inventory, calendar, density, and scenario coverage guards.

These controls are foundations, not proof that the broader invariant is already
complete.

## Implementation Sequence After Phase 1

### Phase 2 — Conservation of explanations

Implement `SN-01`, `SN-02`, and the remaining scope of `SN-03`. Introduce an
explicit finding disposition and economic-effect ownership model. Fail report
generation if any reportable difference is undisposed, becomes invisible, or
owns an explained amount more than once.

### Phase 3 — Financial input integrity

Implement `SN-04`, `SN-06`, and `SN-07`. Continuity anomalies should normally
be visible review findings. Unsafe currency, unit, or timing inputs should fail
the source contract before they can affect performance.

### Phase 4 — Lineage and policy coverage

Implement `SN-05` and `SN-12`. Add stable source locators, forward/reverse
lineage checks, and role-derived policy completeness so new fields cannot enter
or bypass attribution accidentally.

### Phase 5 — Demo protection

Implement `SN-08` and `SN-09`. Move the complete scenario meaning, expected
disposition, independent-change count, and carry-forward status into validated
fixture metadata.

### Phase 6 — Output integrity

Implement `SN-10` and `SN-11`. Canonicalize volatile metadata, compare normalized
content across formats, and prove repeat-run equivalence.

## Change-Control Rule

Any later change to an invariant's guarantee, material-difference boundary,
permitted dispositions, or failure classification requires an explicit design
decision and corresponding catalog/test update. An implementation phase may
improve `coverage`; it must not silently weaken the guarantee.
