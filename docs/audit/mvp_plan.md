# PPAR Audit MVP Completion

## Concrete Definition and Implementation Plan

| Document field | Value |
|---|---|
| Status | Active MVP implementation plan — Slice 6 technical release audit complete; founder MVP assessment next |
| Version | 1.9 |
| Date | 2026-07-21 |
| Governing document | [`product_constitution.md`](product_constitution.md) |
| Product roadmap | [`roadmap.md`](roadmap.md) |
| Historical specifications index | [`product_specifications_index.md`](product_specifications_index.md) |
| Scope | Four founder-defined reporting, Data Issues, codification, and Axys/APX transaction-semantics capabilities required before MVP |
| Excluded | History, dashboards, Operational Intelligence, additional platforms, managed workflow, and broad speculative rules catalog |

---

# 1. MVP Definition

The current product is a substantial technical foundation, but the founder does
not yet consider it an MVP. MVP completion requires the current audit engine,
evidence pack, and safety controls plus four additional product capabilities:

1. an **Executive Summary** worksheet before `Performance Differences`, with an
   equivalent first HTML section;
2. additional enumerated **Issue Type** values and checks for `Data Issues`;
3. stronger codification of causes and issues, with safe, understandable user
   control in YAML; and
4. required Axys/APX YAML-driven transaction semantics, coherent scenarios,
   exact-case behavior, unknown-code failure, and protection against unsupported
   transaction meanings defined in Section 7.

The MVP is complete only when these capabilities are implemented, tested,
demonstrated in the packaged Axys/APX scope, documented for users, and preserve
all eleven active safety invariants plus the `SN-04` retirement guard.

This is an implementation plan, not a new visionary product-design phase.

---

# 2. Verified Current Baseline

## 2.1 Review entrypoint

The current ordinary workbook order is exactly:

1. `Executive Summary`
2. `Performance Differences`
3. `Performance Difference Causes`
4. `Data Issues`

The equivalent HTML report begins with the same `Executive Summary`. Both
formats and the canonical CSV derive from the same bounded summary table and
are covered by bundle parity, determinism, and missing/stale-content checks.
The summary adds presentation only; the three analytical sheets
retain their prior relative order, schemas, and financial semantics.

## 2.2 Current Data Issues vocabulary

The current code has thirteen optional issue-type strings:

- `deliver_in_original_cost_incomplete`
- `duplicate_transactions`
- `dividend_rate`
- `holdings_accrued_rate`
- `holdings_nonpositive_price`
- `holdings_price_range`
- `holdings_stale_price`
- `large_price_variation`
- `missing_dividend`
- `pa_sa_rate`
- `transaction_security_type_mismatch`
- `transactions_nonpositive_price`
- `transactions_price_range`

These values now serialize through public `DataIssueType` members without
changing their strings. `DATA_ISSUE_REGISTRY` carries the category,
mandatory/default enablement, required datasets, tolerance applicability, and
reviewer meaning for every implemented issue type. The conservative
`deliver_in_original_cost_incomplete`, `holdings_nonpositive_price`,
`holdings_stale_price`, `large_price_variation`,
`transaction_security_type_mismatch`, and `transactions_nonpositive_price`
checks are off by default. The row-level checks
require a nonempty `only` population when explicitly enabled. The stale-price
check also requires an explicit positive
`minimum_calendar_days` and reviewed `security_master.security_type`
population. The transaction-price check requires a transaction-code filter and
a reviewed security-master type or asset-class qualifier. The mismatch
check requires the reviewed `security_master.security_type` population it
compares against. `large_price_variation` instead requires a nonempty list of
uniquely identified rules when enabled; each rule may define its own filters,
and requires a decimal minimum variation tolerance.

## 2.3 Current cause vocabulary

The product already has:

- stable finding codes such as `PC-PORT-RET`, `PC-HOLD-MV`, and `PC-TXN-AMT`;
- typed evidence roles, severities, confidence, transaction categories, and
  impact methods; and
- coarse root-cause-area strings:
  `security_return_or_contribution`, `market_value_or_holding`,
  `transaction_activity`, `fx_rate`, `portfolio_performance_input`,
  `classification_or_reference`, and `unexplained`.

Root-cause areas remain derived primarily from dataset identity and cannot be
redefined by users. Their existing strings now serialize through the public
`CauseArea` vocabulary.

## 2.4 Current YAML control

Users can already control each optional Data Issues check through:

- `enabled`;
- exact-match `only` filters;
- exact-match `exclude` filters;
- `absolute_tolerance`; and
- `percent_tolerance`.

The issue-specific `large_price_variation` block adds uniquely identified
named rules with rule-level filters and a decimal `minimum_tolerance`. It does
not make named rules available to unrelated checks.

YAML also controls transaction semantics, impact methods, evidence-only fields,
tolerances, and suppressions.

Financially consequential configuration no longer falls back to internal values.
Every Audit YAML explicitly names the comparison level, all six comparison
tolerances, and both extract-contract safety choices. A configured transaction,
holding, or FX dataset also requires its complete applicable transaction,
holding/price, or FX impact-policy block. The maintained configurations spell out
their former effective values, preserving executable behavior while making those
choices reviewable. Each `large_price_variation` rule likewise explicitly names
`minimum_tolerance`.

Data Issues configuration is now a strict, enumerated, fail-closed product
contract. Malformed sections/checks/filters, unknown issue types or fields,
unsupported per-check controls, non-Boolean enablement, and nonfinite, negative,
or nonnumeric tolerances fail during comparison-specification construction with
an actionable YAML path. `validate_config` reports the effective optional
checks and master-switch policy.

---

# 3. Implementation Order

Although the Executive Summary is the most visible feature, the recommended
implementation order is:

1. **Vocabulary and YAML contract foundation — complete.** Slice 1 implemented
   the behavior-preserving codification and strict configuration contract.
2. **Executive Summary — complete.** The founder accepted the revised two-table
   quantity presentation after workbook review.
3. **Additional Data Issues checks — founder review active** — add an approved
   issue type one at a time through the new registry and YAML contract.
4. **Axys/APX transaction semantics and demo coverage** — implement all Section
   7 requirements, keeping posted transactions, staging controls, and audit
   evidence distinct.
5. **Integrated release validation** — refresh documentation/demo artifacts and
   run all cross-format, invariant, determinism, and scale gates.

This order reduces rework. It does not imply that codification is more important
to the user than the summary.

---

# 4. Workstream A — Vocabulary and YAML Contract

## 4.1 Target

Create a small, explicit vocabulary layer while keeping serialized values and
current findings unchanged.

Recommended types:

- `DataIssueType(StrEnum)` for stable issue IDs;
- `DataIssueCategory(StrEnum)` for reviewer grouping; and
- `CauseArea(StrEnum)` for the current coarse cause areas.

A small issue registry should bind each `DataIssueType` to product facts
needed by execution and presentation, such as:

- category;
- mandatory versus optional;
- default-enabled behavior;
- required normalized dataset(s);
- whether absolute/percent tolerances apply; and
- concise reviewer meaning.

Do not create a generic plugin framework or user-authored rule language for the
MVP.

## 4.2 Initial issue categories

Recommended stable categories are:

- `continuity`
- `duplicate`
- `price`
- `income`
- `accrued_interest`
- `position_value`
- `corporate_action`

Only categories backed by an implemented issue type need to appear in output.
Categories organize information; they do not change whether a row is a
performance cause.

## 4.3 YAML boundary

The MVP YAML contract should:

- reject a non-mapping `data_issues` section;
- reject unknown issue-type names;
- reject unsupported keys under a check;
- require actual Boolean values for `enabled`;
- require finite, nonnegative numbers for tolerances;
- validate `only` and `exclude` as mappings with supported filter fields and
  scalar/list values;
- report actionable paths such as
  `data_issues.holdings_price_range.percent_tolerance`.

Existing valid YAML must retain the same findings and reviewer output after the
codification change.

## 4.4 User control

Approved MVP control:

- enable or disable optional checks;
- limit a check to approved populations;
- exclude known false-positive populations; and
- set type-specific tolerances.

Executive Summary display behavior should remain fixed and version-controlled.
Per-client priority labels, ordering policy, and similar presentation controls
are deferred until usage evidence demonstrates that YAML control is valuable.
This avoids speculating about how different users will process the information.

User YAML must not:

- invent new executable issue types;
- redefine a finding code or cause area;
- convert a Data Issues issue into an additive performance cause;
- override Fully/Partly/Unexplained arithmetic; or
- disable mandatory safety findings.

## 4.5 Acceptance

- All current issue and cause string values serialize unchanged.
- Existing packaged-demo findings remain identical apart from explicitly added
  metadata columns approved for output.
- Every supported issue type has registry metadata and tests.
- Invalid and unknown configuration fails before report generation.
- `validate_config` reports enabled checks and relevant policy.
- No safety invariant, parity contract, or test gate is relaxed.

---

# 5. Workstream B — Executive Summary

## 5.1 Target

Add `Executive Summary` as:

- the first ordinary XLSX worksheet;
- the first ordinary HTML report section; and
- a shared canonical review artifact covered by parity and determinism checks.

It is a presentation layer over existing validated results. It
must not introduce a second financial calculation.

Founder workbook review replaced the initial narrative/priority-list design
with two quantity-only tables. The Executive Summary has no YAML display
configuration.

## 5.2 Minimum content

The first table is `Performance Differences`. Its rows are `Portfolios` and
`Portfolio Periods` for a portfolio report, or `Portfolios` and
`Security Periods` for a security report. Its columns are:

- `Total Quantity`;
- `No Performance Differences`;
- `Fully Explained Differences`;
- `Partly Explained Differences`;
- `Unexplained Differences`; and
- `Setup Incomplete`.

Total quantities use the union of primary performance keys evaluated in
Snapshot A and Snapshot B. Period rows use the existing review status directly;
unchanged evaluated units use `No Performance Differences`. Each portfolio is
counted once using the deterministic worst-status precedence:

`Setup Incomplete > Unexplained > Partly Explained > Fully Explained > No Performance Differences`.

Every row must reconcile exactly: the five status quantities sum to total.

The second table is `Data Issues`. It contains stable issue type and quantity
only. Zero-count issue types are omitted. Rows sort by descending quantity with
issue type ascending as the deterministic tie-breaker.

## 5.3 Presentation requirements

- Render exactly two visibly separate quantity sections.
- Do not include links, narrative explanations, questions, recommendations,
  cause summaries, priority lists, snapshot context, or method prose.
- Stack each XLSX Performance Differences header word on its own line, leave
  the row-label header blank, and size the quantity columns to the widest word.
- Give both tables centered 14-point title bands matching their header color,
  a bottom divider, compact content-based row heights, and three blank rows of
  visual separation.
- Bottom-align all XLSX quantity values.
- Do not introduce a composite score, grade, confidence score, or “passed” badge.
- Do not introduce an explanation-completeness percentage.
- Use the same underlying summary model for XLSX and HTML.
- Keep an empty Data Issues table when no issue rows exist.
- Keep current detailed sheet names and semantics unchanged.

## 5.4 Acceptance

- Ordinary workbook order begins with `Executive Summary`.
- Ordinary HTML begins with the equivalent section.
- Current three analytical sheets remain in their existing relative order.
- Every performance row reconciles to its total and evaluated source scope.
- Data Issues quantities reconcile to canonical Data Issues detail and sort
  descending.
- HTML/XLSX/machine semantics pass parity and deterministic-repeat tests.
- Bundle validation fails on missing, stale, or inconsistent summary content.
- Summary size is bounded by two performance rows plus the finite issue-type
  vocabulary without truncating retained evidence.

---

# 6. Workstream C — Additional Data Issues Issue Types

## 6.1 Rule admission standard

Every new issue type must define:

- stable ID and category;
- operational rationale;
- required normalized data and population;
- transparent detection logic;
- default enablement;
- YAML filters/tolerances;
- expected false positives and exclusions;
- reviewer explanation and next action;
- fixture examples in Snapshot A, Snapshot B, or both as appropriate; and
- tests proving that the issue remains independent of performance explanation
  arithmetic.

An issue type should not enter the MVP merely because it is easy to name.

## 6.2 Recommended first candidate set

The following are the strongest candidates that appear supportable from current
normalized data, subject to founder approval and fixture review:

1. **`holdings_nonpositive_price` — founder accepted.** A nonzero
   holding quantity has a zero or negative price within an explicitly
   configured population. It is opt-in, requires a nonempty `only` filter,
   accepts no tolerance, and reports the observed price without treating the
   issue as performance explanation arithmetic.
2. **`transactions_nonpositive_price` — founder accepted.** A
   nonzero-quantity transaction has a zero or negative price inside an explicit
   transaction-code and reviewed security-master population. The check is
   opt-in, accepts no tolerance, does not infer transaction meaning, and ignores
   missing prices and zero-quantity rows.
3. **`holdings_quantity_value_mismatch`** — quantity and market value have a
   suspicious zero/nonzero relationship within an explicitly configured
   population.

These should initially require narrow, visible YAML populations or exclusions.
Worthless securities, cash conventions, accrued-only rows, shorts, derivatives,
and vendor-specific valuation representations are known false-positive risks.

## 6.3 Revisited price-observation checks and remaining deferred candidates

The founder approved two previously deferred price-observation checks after the
packaged demo adopted dated real-market prices, dividends, and splits:

1. **`holdings_stale_price` — implemented for Slice 3D review.** Within an
   explicitly configured population, the same positive `holdings.price` recurs
   across supplied observations for at least `minimum_calendar_days`. The rule
   states that PPAR did not observe every intervening day and does not conclude
   that an unchanged price is necessarily wrong.
2. **`large_price_variation` — implemented for Slice 3E review.** For each
   snapshot, portfolio, established performance period, security, and named
   rule, union the linked previous-period ending `holdings.price`, current
   period-ending `holdings.price`, and eligible `transactions.price` rows whose
   trade dates fall on either inclusive period boundary or between them. Prices
   are normalized to the period-ending share basis using same-snapshot split
   factors. The rule reports one maximum variation when
   `(maximum - minimum) / minimum` strictly exceeds `minimum_tolerance`.
   Every established period with at least two comparable positive observations
   is eligible regardless of calendar length. The packaged common-stock rule
   uses a decimal tolerance of `0.30`. A raw discontinuity remains visible when
   split evidence is missing.

The following remain deferred without stronger source evidence:

- amount-versus-price-times-quantity transaction arithmetic across mixed asset
  types;
- missing corporate actions or spin-offs;
- position roll-forward reconstruction;
- missing FX conclusions without a stronger currency/reference contract; or
- broad security-master validation.

These may become valuable later, but their false-positive and source-contract
surface remains too large for the current bounded implementation.

## 6.4 Acceptance

- Each approved issue type is an enum/registry member.
- Each has a packaged fixture or focused validation fixture with expected rows.
- YAML enable/disable, filter, tolerance, and issue-specific threshold behavior
  is fail closed and tested.
- Named large-price-variation rules retain unique provenance and emit one row
  per matching rule rather than applying hidden precedence or de-duplication.
- Reviewer output includes stable issue type, category, observed values,
  tolerance, explanation, and review key.
- No new issue changes `Performance Difference Explained`, residual, or
  analytical status.
- Demo coverage and fixture isolation checks protect the intended story.

## 6.5 Security-master-enabled scenarios

The optional `security_master` dataset adds reviewed classification context;
it does not create transaction semantics. In the current runtime,
`security_master.*` values are available only as exact-match `only` and
`exclude` qualifiers for Data Issues checks. A reference row must therefore not
cause PPAR to assign a transaction category, cash-flow sign, or performance
treatment by itself.

The following additional demo scenarios are defensible with that boundary:

1. **Reference-scoped priced transactions — founder accepted.**
   `transactions_nonpositive_price` demonstrates a nonpositive buy price inside
   an explicit reviewed price-bearing security population. It combines the
   transaction-code population with a
   `security_master.asset_class_code` or
   `security_master.security_type` qualifier so cash, unpriced corporate
   actions, and other site-specific conventions remain outside the rule.
2. **Reference-scoped fixed-income rate review — implemented.** The packaged
   `holdings_accrued_rate` and `pa_sa_rate` configurations include a reviewed
   fixed-income asset-class qualifier without changing either rate calculation
   or transaction meaning.
3. **Transaction-versus-reference security-type mismatch — founder accepted.**
   The opt-in `transaction_security_type_mismatch` issue
   compares the transaction row's exact-case `transaction_security_type` with
   the snapshot security master for the same exact-case security identifier.
   This is not
   expressible as a field-to-field comparison with today's filters and
   therefore requires a new issue type. It requires a nonempty population,
   reports both observed values in its explanation, distinguishes case-only
   differences, fails closed on ambiguous reference rows, and makes no claim
   that either classification is universally correct.

A later, separately approved extension could allow an explicitly named
`security_master.*` field in `transaction_rules.*.when`. That would support
site-scoped variants when an otherwise usable transaction extract lacks its own
security-type column. It could corroborate the already evidenced contextual
handling of `ai`, `pa`/`sa`, `pd`, `li`/`lo`, and `dp`/`wd`; it would not prove a
new transaction code or allow classification from asset class alone. The rule
must explicitly request the reference field, preserve exact source case, and
fail closed when the required reference row or field is absent. PPAR must not
silently substitute reference data for missing transaction evidence.

Do not initially add currency-mismatch, universal security-type-to-asset-class
consistency, or inferred-corporate-action checks. A reference currency may mean
trading, pricing, or income currency, and type/class dictionaries remain
version- and site-specific. Snapshot-to-snapshot classification-change evidence
may be useful later, but belongs with the deferred broad security-master work
until its reviewer meaning and source contract are defined.

The founder accepted the priced-transaction output and continued Slice 3 with
the security-type mismatch. None adds a blocking requirement to Workstream D,
and the reference-aware transaction-rule extension is not required for the
current-capability scenarios in Section 7.2.

---

# 7. Workstream D — Axys/APX Transaction Semantics and Demo Coverage

## 7.1 Target and scope boundary

The material requirements identified in Sections 7.1 through 7.3 and 7.5 are
blocking for MVP completion. Section 7.4 explicitly classifies optional and
deferred evidence that is not blocking. The work must remain evidence-scoped
and must not turn integration-, version-, or site-specific behavior into
universal Axys/APX semantics.

Workstream D's material MVP completion is limited to:

- driving transaction categories, signs, and performance treatment from
  executable YAML policy rather than transaction-code rules hidden in Python;
- demonstrating coherent Axys/APX transaction, holdings, cash, and reported-
  performance scenarios;
- preserving and matching transaction codes and native context using exact case
  when a versioned source contract requires it;
- failing closed when a transaction code or required policy remains unknown;
  and
- preventing unsupported or merely observed transaction meanings from becoming
  performance causes.

Evidence about an external integration does not itself create an MVP field,
dataset, or product feature. Such work remains deferred until PPAR receives that
input from a validation partner and its material reviewer value is established.

Keep detailed transaction meanings, confidence boundaries, and fixture coverage
in the canonical Axys/APX sources:

- [`transaction_semantics_matrix.yaml`](../axys_apx/contracts/transaction_semantics_matrix.yaml)
  is machine-readable research and fixture evidence, not runtime policy;
- [`transaction_semantics_matrix.md`](../axys_apx/contracts/transaction_semantics_matrix.md)
  is its maintainer-facing companion;
- [`Chapter_05_Transactions.md`](../axys_apx/reference/Chapter_05_Transactions.md)
  owns reader-facing semantics and cautions; and
- [`Research_05_Transactions.md`](../axys_apx/evidence/Research_05_Transactions.md)
  and [`Public_Web_Research_2026-07-17.md`](../axys_apx/evidence/Public_Web_Research_2026-07-17.md)
  preserve granular provenance.

This plan states the required reviewer story, prerequisites, and completion gate
without becoming another transaction-code dictionary.

## 7.2 Required scenarios using current normalized capabilities

The following scenarios can be expressed with the current transaction columns,
normalized categories, impact methods, and conditional YAML rules. Each must be
implemented, tested, and demonstrated before MVP completion. Its source scope
must remain visible; none is asserted as universal Axys/APX behavior.

| Required scenario | Required demo treatment | Required boundary |
|---|---|---|
| Contextual `ai` margin interest | Promote the existing test-only margin-interest pattern into a coherent packaged scenario: negative amount, explicit margin/security context, `fee_expense`, negative cash sign, and performance treatment under the demo's net basis. | Do not classify code-only `ai`; holdings, cash, and reported performance must tell a consistent financing-cost story. |
| `dv` + `by` dividend reinvestment | Add matched income and purchase legs, preferably with `dvwash` context, and demonstrate that neither leg becomes an investor external flow or creates double-counted income. | Require aligned dates, security, amounts, wash context, and coherent holdings/cash effects. |
| Scoped `ti` or `si` deliver-in | Add a site-scoped deliver-in example using explicit source/destination and security context. Public Axys report guidance identifies `li`, `ti`, and `si` as deliver-in cases in that workflow. | Update `transaction_rules` and the research evidence before completing the demo; do not infer external capital versus internal transfer from the code alone. |
| Gross dividend plus separate withholding expense | Demonstrate one observed integration representation using a dividend income row and contextual withholding-expense row. | State that the scenario assumes after-tax/net performance; do not claim one standard withholding code or representation. |
| Alternate contextual fee mapping | Add a site-variant `dp` example using observed `epus expense`-style context alongside the existing `exus custfee` pattern. | Treat `epus` as a configurable token role, not a proven universal standalone transaction code. |

The first four scenarios must be visible in the packaged Axys/APX demo. The
alternate contextual fee mapping must be covered by a site-variant fixture and
the transaction-semantics coverage contract. Keep each example focused on its
distinct reviewer lesson.

## 7.3 Exact-case handling and cancellation boundary

PPAR must be capable of preserving and matching transaction codes and native
transaction-context identifiers by exact case when a source contract requires
it. This includes security type, source/destination symbol, and special-security
values used in conditional YAML rules. Exact-case capability must not create a
global rule that uppercase codes are cancellations: a site may legitimately use
uppercase economic transaction codes.

The reviewed uppercase-cancellation evidence belongs to an AIA Trade Blotter
workflow. A tool creates a cancellation blotter from historical transactions and
uppercases an original code, such as `by` to `BY`, as a cancellation instruction.
The evidence does not establish that the uppercase instruction survives as a
posted transaction or is available in ordinary REP, IMEX, SQL, or report
extracts.

Therefore:

- packaged `snapshot_a/transactions.csv` and `snapshot_b/transactions.csv`
  must not present uppercase cancellation instructions as extracted posted
  transactions;
- cancellation instructions must not contribute an additive performance impact;
- PPAR does not need a Trade Blotter fixture while its supported input remains
  posted-transaction snapshots;
- cancellation interpretation requires an explicit source-stage or extract
  contract, plus linkage to the original transaction; and
- an uppercase row in an ordinary posted-transaction extract remains unknown
  unless local evidence establishes its role.

Before Slice 5A, snapshot transaction matching and reviewer metadata preserved
native case, but semantic classification still normalized case in built-in
category inference, YAML rule keys and conditions, coverage summaries, and
several audit helpers. Slice 5A closed that gap without changing currency-domain
normalization or weakening existing exact-case portfolio/security matching. Its
focused safety regression proves that an uppercase
`BY` in an ordinary exact-case posted extract does not inherit lowercase `by`
semantics: source loading stops with an unknown-transaction error before the row
can become a performance cause. A source-stage cancellation product contract is
therefore not required for the current MVP.

## 7.4 Optional and deferred source-contract evidence

| Evidence item | MVP posture | Reason |
|---|---|---|
| Missing-cost deliver-in | Implemented optional capability; not MVP-blocking. | The narrowly scoped completeness check can reveal missing original-cost evidence in a cited deliver-in report workflow, but original cost does not drive current time-weighted performance and the primary demo correctly omits the optional columns. |
| APX Custodial Integrator Mark-to-Market | Deferred; no MVP implementation. | The evidence establishes version-specific requiredness for one external integration, not the field's accounting meaning, availability in PPAR snapshots, or material value to Performance Comparison. |
| Trade Blotter cancellation control | Product feature deferred; current MVP uses a focused safety regression. | PPAR consumes posted-transaction snapshots, while the reviewed cancellation evidence concerns pre-posting instructions. Exact-case unknown-code failure already prevents unsupported uppercase meanings from becoming performance causes. |

None of these evidence items adds another MVP-blocking field, dataset, fixture,
or source-stage feature. The underlying research remains available for a future
validation partner whose actual workflow demonstrates the need.

Do not add speculative ACA merger, spin-off, or reorganization transaction codes.
Public evidence verifies an APX ACA-to-Reorg-Utility-to-Trade-Blotter workflow but
does not disclose enough transaction fields or codes for a defensible packaged
transaction fixture.

## 7.5 Completion standard

Before this workstream can be marked complete:

- identify the exact source/workflow and preserve its confidence boundary;
- keep transaction categories, signs, and performance treatment driven by
  complete executable YAML policy;
- construct coherent transaction, holdings, cash, and reported-performance
  effects for the implemented Axys/APX scenarios;
- preserve exact case when the versioned source contract requires it;
- fail closed for unknown codes or incomplete required transaction policy;
- prevent unsupported meanings from becoming performance causes;
- update the transaction research matrix and generated companion;
- add validation, reconstruction, reviewer-output, and false-positive tests;
- preserve all financial, conservation, explanation-reconciliation, and output
  invariants; and
- keep external integration and source-stage evidence deferred until actual
  validation-partner inputs justify product support.

---

# 8. Cross-Cutting Release Gates

MVP completion is unacceptable unless:

- all eleven active safety invariants remain enforced and retired `SN-04`
  cannot silently reactivate;
- current valid YAML remains behaviorally compatible unless an intentional
  contract change is documented;
- malformed or unknown new configuration fails closed;
- current financial and report-reconciliation tests pass;
- workbook, HTML, CSV, and machine-readable outputs agree semantically;
- deterministic-repeat tests cover the new summary and metadata;
- bundle manifest/contract versions change deliberately if required;
- setup templates and setup-installed README explain the new behavior;
- portfolio and security demo bundles demonstrate the first-sheet experience;
- no source detail or suppressed finding is lost; and
- the maintained release-candidate and 500x scale checks pass after the
  cross-cutting report/configuration changes.

Established gates must not be raised, relaxed, disabled, or bypassed merely
because MVP work causes a failure.

---

# 9. Proposed Implementation Slices

## Slice 1 — Current vocabulary and strict Data Issues YAML — Complete

Implemented on 2026-07-17 as a behavior-preserving code change:

- add issue/cause enums and registry;
- migrate current constants and groupings to those types;
- add strict YAML validation;
- expose enabled-check summary through config validation; and
- prove current report artifacts remain semantically unchanged.

## Slice 2 — Executive Summary shared model — Complete

Implemented as a bounded presentation change:

- added one canonical summary table shared by CSV, XLSX, and HTML;
- added the first ordinary worksheet and first ordinary HTML section;
- added mutually exclusive portfolio/period quantities over the evaluated source
  universe and a separate setup-incomplete bucket;
- added Data Issues type quantities in descending order;
- removed narrative, links, priority lists, and internal vocabulary after
  founder usability review; and
- extended manifest, bundle validation, parity, determinism, and documentation
  contracts without adding financial calculations, issue types, or YAML controls.

The founder accepted the two-table quantity presentation on 2026-07-18.

## Slice 3 — Additional Data Issues issue types — Complete

- **Slice 3A — founder accepted:** implemented `holdings_nonpositive_price` end
  to end with strict opt-in policy, two-snapshot fixtures, category output, and
  false-positive controls.
- **Slice 3B — founder accepted:** added
  `transactions_nonpositive_price`, requiring transaction-code and reviewed
  security-master populations, and reference-scoped the existing fixed-
  income rate checks.
- **Slice 3C — founder accepted:** added the exact-case
  `transaction_security_type_mismatch` issue defined in Section 6.5 with an
  isolated case-only demo row in each snapshot.
- **Slice 3D — founder accepted:** added the opt-in
  `holdings_stale_price` observed-date rule with a strict calendar-day threshold
  and an isolated two-snapshot GOOGL source-price anomaly.
- **Slice 3E — founder accepted:** added split-normalized,
  period-level `large_price_variation` with uniquely identified overlapping
  rules, scalar-or-list source filters, inclusive period boundaries, strict decimal
  thresholds, deterministic evidence selection, and real-market demo findings.

## Slice 4 — Required current-capability Axys/APX scenarios — Implemented for founder review

- added packaged context-gated `ai`, matched `dv` + `by`, site-scoped `ti`, and
  gross-dividend-plus-withholding scenarios;
- added the alternate contextual `dp` plus `epus expense` site fixture without
  promoting `epus` to a standalone code meaning;
- updated the transaction research matrix and generated companion;
- derived coherent holdings, cash, performance, and reviewer output through the
  maintained scenario generator; and
- preserved the SN-03 Fully Explained invariant across leap-year history copies
  by reconciling only sub-precision six-decimal display residuals while retaining
  raw cause lineage; and
- retained code-only `ai`/`ti`, original-cost, exact-case, and cancellation
  boundaries for their separately scoped work.

Before Slice 5A, the transaction-policy boundary was tightened without adding a
new transaction meaning: complete matching site YAML rules now override source
semantic labels; compatibility categories, safety groups, Data Issues code
groups, and demo accounting/reconstruction effects are loaded from executable
YAML instead of Python transaction-code sets. `Performance Difference Causes`
retains its established output schema; every transaction-associated row starts
its `Explanation` with the native transaction code, including the existing `ai`
and `ti` scenarios.

## Slice 5 — Exact case and optional missing-cost work — Complete

- **Slice 5A — complete:** exact transaction-rule keys and native
  context-condition values are now invariant. The retired case-insensitive
  compatibility option and code-only default meanings have been removed.
  Exact matching keeps
  lowercase and uppercase rules distinct, and leaves an unmatched uppercase
  posted-transaction code unknown rather than treating it as a cancellation.
  A focused site fixture gives `by` and `BY` separate explicit economic meanings;
  it is not a Trade Blotter cancellation fixture.
- **Slice 5B — complete as an optional, non-MVP-blocking capability:** added
  optional normalized
  `transactions.original_cost` and `transactions.original_cost_date` inputs and
  the opt-in `deliver_in_original_cost_incomplete` Data Issues check. The check
  requires an explicit YAML population containing transaction code, security
  type, and source/destination type and symbol; Python assigns no universal
  meaning to `li`, `ti`, or `si`. Enabled checks fail when either source column
  is absent, report one row when either value is blank, accept zero original
  cost as present, and do not calculate cost basis or conclude that Axys used
  its documented trade-date market-value fallback. After founder review, the
  primary demo omits the optional source columns so they are not mistaken for
  Performance Comparison requirements; its YAML retains a disabled, fully
  scoped example while focused tests demonstrate the enabled finding. Lot
  context remains deferred because its source semantics are incomplete and it
  is not needed for this bounded check.
- The exact-case unknown-code safety regression proves that an unsupported
  uppercase transaction cannot inherit lowercase semantics or become a posted-
  transaction performance cause. APX Mark-to-Market and a Trade Blotter
  source-stage product contract are deferred under Section 7.4.

After Slice 5, the founder approved a bounded configuration-hardening pass before
the release audit. It removed internal fallbacks for the consequential settings
listed in Section 2.4, migrated every maintained YAML file with the exact former
effective values, and added fail-closed omission/malformed-value regressions. It
did not add fields, report columns, calculations, issue types, or transaction
meanings. Presentation-only labels, suppressions, optional-file requiredness, and
other non-financial conveniences retain their established defaults.

The founder then approved a bounded output-contract simplification before the
release audit. The standard `ppar audit` command now always creates portfolio
and security reports when their configured inputs are available, skips security
only when `files.security_performance` is unavailable, and no longer exposes a
`--report` selector. `source_detail.csv` now always resides at each report root
and is never duplicated in `supporting_files/` or `audit_support.zip`, including
expanded-support runs. Lower-level comparison-level APIs remain available, and
this packaging change does not alter report columns, calculations, findings, or
financial semantics.

The founder also approved a bounded starter-configuration cleanup before the
release audit. The Audit file mappings now describe only headings present in
both supplied snapshots, and the checked-in CSVs use the same required-first,
optional-alphabetical order shown in YAML. Most demo Data Issues rules now use
business populations such as security type, asset class, and transaction code.
The dividend, missing-dividend, and accrued-rate rules were broadened after the
demo baseline stopped inventing AMZN dividends and adopted consistent
fixed-income accrued income per unit; only the named JPM, AAPL, and 91282Y5Y1
review scenarios remain visible. Axys/APX layouts that map both `security_type`
and `security_symbol` now default to the existing compact composite identity.
The starter also omits `extract_contract` because the packaged contract,
ambiguous-flow enforcement, and exact-case matching are the fail-closed defaults.
Explicit identity overrides, local contract paths, safety opt-outs, and legacy
case matching remain supported. No report column, transaction meaning,
financial method, or tolerance changed.

## Slice 6 — MVP release audit

- do not add another field, issue type, or integration contract unless the
  founder separately approves material value and scope;
- run the full relevant suite and release-candidate checks;
- generate final representative portfolio/security bundles; and
- assess whether the founder considers the resulting product an MVP.

The technical release audit completed on 2026-07-21. A clean baseline passed
before a bounded stabilization review examined installed-package behavior,
output schemas, configuration/CLI consistency, starter coherence, transaction
safety, and explanation contracts. Existing coverage was already strong in the
latter five areas. The review added a built-wheel installation and end-to-end
smoke gate, froze the ordered columns of every persisted default or diagnostic
Audit table, and corrected the declared Polars minimum from 1.16 to 1.24 because
PPAR uses the `nulls_equal` API introduced by that rename. The strengthened
release-candidate sequence then passed 931 tests, static checks, wheel/sdist
builds, installed Analytics and Audit execution, both installed bundle
validators, representative portfolio/security bundles, determinism/parity
validation, and the unchanged 500x scale gate. The founder MVP assessment
remains outstanding.

Post-audit explanation hardening now treats reviewer wording as a structured
runtime contract. Each source-data explanation begins with the exact
`dataset.field`, direction, and absolute change; transaction explanations begin
with the source transaction code. Supporting rows name the selected downstream
Modified Dietz field, and local- versus base-currency wording is derived from the
same impact policy used by the calculation. Report generation fails the SN-03
explanation invariant when these facts disagree. The checks add no report
columns and do not change financial methods, tolerances, or impact ownership.

---

# 10. Remaining Founder Decisions Before Further Application Changes

The Executive Summary boundary and two-table quantity design in Section 5, and
Workstream D's MVP-blocking status, are founder-approved. Workstream D remains
sequenced after Slices 2 and 3; blocking MVP completion does not make it a Slice
2 prerequisite.

1. **YAML priority:** confirm that `review_priority: high | normal | low` and
   comparable workflow/presentation policy are deferred. Recommended: defer
   until real usage establishes a need.

No open product decision blocks the Slice 6 release audit. The founder retains
the final decision about whether the reviewed result is an MVP.

No further visionary product-design phase is required to make these decisions.

---

# 11. MVP Completion Gate

The founder determines that PPAR Audit has reached MVP only after:

- the four required capabilities are implemented and reviewed;
- the generated portfolio and security outputs present the right information in
  a useful first view;
- additional issue types are defensible and controllable;
- cause/issue vocabulary and YAML behavior are stable and fail closed;
- every material Section 7 YAML-semantic, coherent-scenario, exact-case,
  unknown-code, and unsupported-meaning requirement satisfies its evidence,
  fixture, and safety gates;
- all financial, safety, output, determinism, demo, and scale gates pass; and
- remaining gaps are acceptable for a controlled validation partner.

Validation-partner planning begins after this gate, not before it.
