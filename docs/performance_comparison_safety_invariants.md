# Performance Auditing Safety Invariants

This is a maintainer-facing contract for the safety-net implementation program.
It defines what the protections mean, audits the current enforcement baseline,
and prevents later phases from changing the meaning of a safety net while
implementing it. It is not another user onboarding document.

The executable catalog is
`ppar.performance_comparison.safety_invariants.SAFETY_INVARIANTS`. This document
explains the design decisions behind that catalog.

## Phase 1 Outcome

Phase 1 defined and audited the invariants without changing comparison or report
behavior. The audit found substantial existing foundations and identified the
end-to-end gaps addressed by later phases.

The strongest existing protection is portfolio explanation arithmetic:

- cause totals reconcile before workbook construction;
- displayed six-decimal totals reconcile before serialization;
- serialized workbook cells reconcile after construction; and
- Modified Dietz components with estimated impact must remain visible in the
  causes table.

Phase 2 enforces the first three catalog entries. Phase 3 enforces
financial-input integrity through `SN-04`, `SN-06`, and `SN-07`. Phase 4
enforces lineage and policy coverage through `SN-05` and `SN-12`. Phase 5
enforces demo preservation and isolation through `SN-08` and `SN-09`. Phase 6
enforces output-format parity and repeat-run determinism through `SN-10` and
`SN-11`.

## Phase 2 Outcome

Phase 2 implements conservation of explanations:

- `findings.csv` remains the complete source-finding audit trail and includes
  suppressed rows;
- every finding has a contiguous sequence, unique fingerprint, and explicit
  `review_evidence` disposition;
- every cause row has a `counted_cause` or `review_evidence` disposition, an
  economic-effect identifier, and exactly one owner when counted;
- support-only transaction components, FX rates, and split factors cannot own
  an explained amount;
- cash is represented only through holdings, with market value counted and
  quantity retained as visible review evidence; and
- portfolio and security explanation arithmetic is checked internally, at
  displayed precision, and after workbook construction.

A mismatch is an internal logic error and stops report generation.

The Phase 2 cleanup also retired the standalone `cash` dataset. Cash balances
now have one normalized representation—holdings such as `CASHUSD`, `CASHEUR`,
or `CASHGBP`—so separate cash and holding rows cannot claim the same valuation
effect.

## Phase 3 Outcome

Phase 3 enforces financial-input integrity:

- prior ending and next beginning market values are compared within each
  snapshot at portfolio and security grain;
- continuity mismatches are mandatory visible `Data Audit Issues` and remain
  visible even when optional data-audit checks are disabled;
- supplied currency codes are normalized and validated as three-letter codes;
- foreign holdings and transactions cannot use local monetary values as
  counted base-currency inputs without explicit `base_` counterparts;
- same-currency local/base values must agree, and portfolio-specific FX quotes
  must target the portfolio base currency;
- reversed or overlapping performance periods stop processing;
- changed holdings, transactions, FX rates, and split evidence is audited for
  deterministic period assignment; and
- historical evidence outside an assigned formula boundary may remain visible
  for carry-forward review, but it cannot own an explained amount. A prior-day
  holding or FX value explicitly assigned as the beginning boundary remains a
  Modified Dietz input and may own its explained amount.

Continuity anomalies are reviewer findings. Unsafe currency, unit, or timing
inputs are source-contract errors; attempts to count out-of-period evidence are
internal logic errors.

## Phase 4 Outcome

Phase 4 enforces source lineage and fail-closed policy coverage:

- every finding has a stable locator based on its normalized logical key rather
  than a physical CSV row number;
- source-backed cause rows retain the fingerprints of the findings from which
  they were built;
- formula-derived rows and explicit no-cause dispositions have distinct lineage
  types instead of pretending to be source records;
- bundles persist both `findings.csv` and `cause_lineage.csv`, and validation
  rejects missing or invalid lineage metadata;
- every field on a comparison surface must have an explicit accounting role;
- fields with `performance_input` or `input_component` roles derive their YAML
  policy requirement from that role; and
- an unknown changed field stops processing even if a suppression rule matches
  it, so suppression cannot substitute for a classification decision.

Lineage failures are internal logic errors. Unclassified fields and incomplete
impact treatment are source-contract errors.

## Phase 5 Outcome

Phase 5 makes the demo's intended accounting stories executable contracts:

- every scenario records its economic meaning, source and story periods,
  portfolio, family, primary security, expected report disposition, and
  expected period status;
- the protected contract must agree exactly with the operational scenario
  calendar and the actual source-input date;
- paired transaction legs and paired holding/transaction corrections share one
  independent economic-change identifier instead of being counted as unrelated
  physical rows;
- each source period is limited to at most two independent economic changes;
- carry-forward effects are identified explicitly, do not consume a new-change
  slot in the later period, and must remain visible as beginning-period causes;
- expected `counted_cause`, `review_evidence`, `data_audit_issue`, and
  `fixture_only_context` outcomes are checked against generated report data; and
- an isolation matrix makes source-change counts and visible carry-forward
  stories inspectable in the rebuild audit output.

These failures are demo-maintenance errors. They stop checked-in fixture
maintenance without weakening or hiding any production-client evidence.

## Phase 6 Outcome

Phase 6 makes the generated bundle a fail-closed semantic contract:

- every internal table persisted as CSV has an ordered column list, coarse
  column types, row count, and normalized semantic SHA-256 fingerprint;
- every HTML/XLSX review sheet has a canonical display fingerprint derived from
  its shared internal table, visible columns, labels, values, and row order;
- the three normal review sheets also have exact supporting CSV counterparts:
  `performance_differences.csv`, `performance_difference_causes.csv`, and
  `x_ref_issues.csv`;
- post-write validation compares CSV content to its typed internal fingerprint
  and HTML/XLSX content to its canonical display fingerprint;
- manifest version 3 records a normalized bundle fingerprint and explicitly
  excludes only `manifest.created_at`, XLSX core created/modified timestamps,
  and XLSX ZIP-entry timestamps; and
- mutation tests prove same-row-count content drift is detected, while a
  repeat-run test proves normalized manifests and deterministic artifacts are
  identical for identical inputs and configuration.

Parity or determinism failures are internal logic errors and stop bundle
generation. The volatile exclusions affect reproducibility checks only; they do
not permit any financial value, status, cause, evidence row, ordering, or report
label to vary.

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
- `rebuild_performance_comparison_demo_data.audit_demo_data`: generated-data
  drift, residual, inventory, calendar, density, and scenario coverage guards.

The catalog and focused tests identify which combination of controls proves each
broader invariant.

## Implementation Sequence After Phase 1

### Phase 2 — Conservation of explanations

Complete. `SN-01`, `SN-02`, and `SN-03` now enforce explicit dispositions,
single-owner economic effects, lossless persisted findings, and portfolio plus
security arithmetic.

### Phase 3 — Financial input integrity

Complete. `SN-04`, `SN-06`, and `SN-07` now enforce mandatory continuity
findings, currency/base-unit contracts, unambiguous periods, exhaustive changed
dated-evidence assignment checks, and counted-impact boundary rules.

### Phase 4 — Lineage and policy coverage

Complete. `SN-05` and `SN-12` now enforce stable source locators,
forward/reverse lineage, persisted cause-lineage validation, explicit field
roles, and role-derived policy completeness.

### Phase 5 — Demo protection

Complete. `SN-08` and `SN-09` now protect complete scenario meaning, actual
source periods, expected report disposition/status, independent economic-change
counts, paired accounting legs, and visible carry-forward effects.

### Phase 6 — Output integrity

Complete. `SN-10` and `SN-11` now canonicalize volatile metadata, compare typed
and display-normalized content across CSV/HTML/XLSX/internal tables, reject
content drift, and prove repeat-run equivalence.

## Change-Control Rule

Any later change to an invariant's guarantee, material-difference boundary,
permitted dispositions, or failure classification requires an explicit design
decision and corresponding catalog/test update. An implementation phase may
improve `coverage`; it must not silently weaken the guarantee.
