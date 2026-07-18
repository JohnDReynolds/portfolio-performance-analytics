# PPAR Audit MVP Completion

## Concrete Definition and Implementation Plan

| Document field | Value |
|---|---|
| Status | Active MVP implementation plan — Slice 3E implemented; founder output review active |
| Version | 1.4 |
| Date | 2026-07-18 |
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
4. required Axys/APX transaction-semantics, exact-case, source-contract, and
   demo coverage defined in Section 7.

The MVP is complete only when these capabilities are implemented, tested,
demonstrated in the packaged Axys/APX scope, documented for users, and preserve
all twelve safety invariants.

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

The current code has twelve optional issue-type strings:

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

It also has two mandatory continuity types:

- `portfolio_market_value_continuity`
- `security_market_value_continuity`

These values now serialize through public `DataIssueType` members without
changing their strings. `DATA_ISSUE_REGISTRY` carries the category,
mandatory/default enablement, required datasets, tolerance applicability, and
reviewer meaning for every implemented issue type. The conservative
`holdings_nonpositive_price`, `holdings_stale_price`,
`large_price_variation`, `transactions_nonpositive_price`, and
`transaction_security_type_mismatch` are off by default. The row-level checks
require a nonempty `only` population when explicitly enabled. The stale-price
check also requires an explicit positive
`minimum_calendar_days` and reviewed `security_reference.security_type`
population. The transaction-price check requires a transaction-code filter and
a reviewed security-reference type or asset-class qualifier. The mismatch
check requires the reviewed `security_reference.security_type` population it
compares against. `large_price_variation` instead requires a nonempty list of
uniquely identified rules when enabled; each rule may define its own filters,
inclusive-period minimum days, and decimal minimum variation tolerance.

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
named rules with rule-level filters, inclusive `minimum_calendar_days`, and a
decimal `minimum_tolerance`. It does not make named rules available to unrelated
checks.

YAML also controls transaction semantics, impact methods, evidence-only fields,
tolerances, and suppressions.

Data Issues configuration is now a strict, enumerated, fail-closed product
contract. Malformed sections/checks/filters, unknown issue types or fields,
unsupported per-check controls, non-Boolean enablement, and nonfinite, negative,
or nonnumeric tolerances fail during comparison-specification construction with
an actionable YAML path. `validate_config` reports effective optional checks
and the mandatory continuity policy.

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
- keep mandatory continuity checks active even if optional checks are disabled;
  and
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
   transaction-code and reviewed security-reference population. The check is
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
   `minimum_calendar_days` uses inclusive day count
   `(thru_date - from_date) + 1`. Defaults are one calendar day and decimal
   tolerance `0.20`. The packaged rule reports real AVGO movements, including
   the 20.04 percent BALANCED observation supported by the current dated source
   values. A raw discontinuity remains visible when split evidence is missing.

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

## 6.5 Security-reference-enabled scenarios

The optional `security_reference` dataset adds reviewed classification context;
it does not create transaction semantics. In the current runtime,
`security_reference.*` values are available only as exact-match `only` and
`exclude` qualifiers for Data Issues checks. A reference row must therefore not
cause PPAR to assign a transaction category, cash-flow sign, or performance
treatment by itself.

The following additional demo scenarios are defensible with that boundary:

1. **Reference-scoped priced transactions — founder accepted.**
   `transactions_nonpositive_price` demonstrates a nonpositive buy price inside
   an explicit reviewed price-bearing security population. It combines the
   transaction-code population with a
   `security_reference.asset_class_code` or
   `security_reference.security_type` qualifier so cash, unpriced corporate
   actions, and other site-specific conventions remain outside the rule.
2. **Reference-scoped fixed-income rate review — implemented.** The packaged
   `holdings_accrued_rate` and `pa_sa_rate` configurations include a reviewed
   fixed-income asset-class qualifier without changing either rate calculation
   or transaction meaning.
3. **Transaction-versus-reference security-type mismatch — founder accepted.**
   The opt-in `transaction_security_type_mismatch` issue
   compares the transaction row's exact-case `security_type` with the snapshot
   security reference for the same exact-case security identifier. This is not
   expressible as a field-to-field comparison with today's filters and
   therefore requires a new issue type. It requires a nonempty population,
   reports both observed values in its explanation, distinguishes case-only
   differences, fails closed on ambiguous reference rows, and makes no claim
   that either classification is universally correct.

A later, separately approved extension could allow an explicitly named
`security_reference.*` field in `transaction_rules.*.when`. That would support
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

Every requirement in this section is blocking for MVP completion. The work must
remain evidence-scoped and must not turn integration-, version-, or
site-specific behavior into universal Axys/APX semantics.

Keep detailed transaction meanings, confidence boundaries, and fixture coverage
in the canonical Axys/APX sources:

- [`transaction_semantics_matrix.yaml`](../axys_apx/contracts/transaction_semantics_matrix.yaml)
  is the machine-readable implementation contract;
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
| Scoped `ti` or `si` deliver-in | Add a site-scoped deliver-in example using explicit source/destination and security context. Public Axys report guidance identifies `li`, `ti`, and `si` as deliver-in cases in that workflow. | Update the transaction-semantics contract before completing the demo; do not infer external capital versus internal transfer from the code alone. |
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
- any future example belongs in a separate test-only Trade Blotter,
  import-control, or audit-lifecycle fixture;
- cancellation interpretation requires an explicit source-stage or extract
  contract, plus linkage to the original transaction; and
- an uppercase row in an ordinary posted-transaction extract remains unknown
  unless local evidence establishes its role.

Snapshot transaction matching and reviewer metadata already preserve native
case. The remaining implementation gap is semantic classification: current
built-in category inference, YAML rule keys and conditions, coverage summaries,
and several audit helpers still normalize case. Slice 5 must close that gap
without changing currency-domain normalization or weakening existing exact-case
portfolio/security matching.

## 7.4 Required normalized-contract and product work

| Required capability | Required implementation | Intended reviewer value |
|---|---|---|
| Missing-cost deliver-in | Add normalized original-cost amount/date and, if justified, lot context; then implement a narrowly scoped completeness check. | Detect deliver-ins whose apparently reasonable report value may rely on a trade-date market-value fallback rather than supplied original cost. |
| APX foreign-currency Mark-to-Market | Add an optional normalized context field and version/site-aware validation. | Surface the documented APX CI v1-v3 versus v4 requiredness difference without treating the field as an additive performance cause. |
| Trade Blotter cancellation control | Add a distinct staging/control fixture or source-stage contract after exact-case matching exists. | Prove that cancellation instructions are quarantined from posted transaction and performance semantics. |

All three capabilities are required before MVP completion. Missing-cost and
Mark-to-Market behavior must have user-visible packaged or focused validation
examples appropriate to their data contracts. The Trade Blotter cancellation
control must remain a separate staging/control fixture rather than appearing in
packaged posted-transaction snapshots.

Do not add speculative ACA merger, spin-off, or reorganization transaction codes.
Public evidence verifies an APX ACA-to-Reorg-Utility-to-Trade-Blotter workflow but
does not disclose enough transaction fields or codes for a defensible packaged
transaction fixture.

## 7.5 Completion standard

Before this workstream can be marked complete:

- identify the exact source/workflow and preserve its confidence boundary;
- define complete YAML semantics and the normalized fields needed to support
  them;
- construct coherent transaction, holdings, cash, and reported-performance
  effects where those datasets are applicable;
- place each example in the packaged demo, site-variant fixture, or
  staging/control fixture required by Sections 7.2 through 7.4;
- update the canonical transaction-semantics contract and generated companion;
- add validation, reconstruction, reviewer-output, and false-positive tests;
- preserve all financial, conservation, explanation-reconciliation, and output
  invariants; and
- demonstrate that ordinary posted-transaction extracts remain separate from
  Trade Blotter cancellation instructions.

---

# 8. Cross-Cutting Release Gates

MVP completion is unacceptable unless:

- all twelve safety invariants remain enforced;
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

## Slice 3 — Additional Data Issues issue types — Active

- **Slice 3A — founder accepted:** implemented `holdings_nonpositive_price` end
  to end with strict opt-in policy, two-snapshot fixtures, category output, and
  false-positive controls.
- **Slice 3B — founder accepted:** added
  `transactions_nonpositive_price`, requiring transaction-code and reviewed
  security-reference populations, and reference-scoped the existing fixed-
  income rate checks.
- **Slice 3C — founder accepted:** added the exact-case
  `transaction_security_type_mismatch` issue defined in Section 6.5 with an
  isolated case-only demo row in each snapshot.
- **Slice 3D — implemented:** added the opt-in
  `holdings_stale_price` observed-date rule with a strict calendar-day threshold
  and an isolated two-snapshot GOOGL source-price anomaly.
- **Slice 3E — implemented for founder review:** added split-normalized,
  period-level `large_price_variation` with uniquely identified overlapping
  rules, scalar-or-list source filters, inclusive period days, strict decimal
  thresholds, deterministic evidence selection, and real AVGO demo findings.

## Slice 4 — Required current-capability Axys/APX scenarios

- add the packaged `ai`, `dv` + `by`, `ti`/`si`, and withholding scenarios;
- add the alternate contextual fee site-variant fixture;
- update the canonical transaction-semantics contract and generated companion;
- verify coherent holdings, cash, performance, reconstruction, and reviewer
  output; and
- run all affected demo, financial, and coverage gates.

## Slice 5 — Exact case and required transaction source-contract work

- implement exact-case transaction rule and context-condition capability without
  globally assigning cancellation meaning to uppercase codes;
- add original-cost/date support and the bounded deliver-in completeness check;
- add APX Mark-to-Market context and version/site-aware validation;
- add the separate Trade Blotter cancellation-control fixture; and
- prove that cancellation instructions cannot become posted-transaction
  performance causes.

## Slice 6 — Remaining approved issue types and MVP release audit

- repeat the end-to-end Data Issues rule process for any remaining approved
  candidates;
- run the full relevant suite and release-candidate checks;
- generate final representative portfolio/security bundles; and
- assess whether the founder considers the resulting product an MVP.

---

# 10. Remaining Founder Decisions Before Further Application Changes

The Executive Summary boundary and two-table quantity design in Section 5, and
Workstream D's MVP-blocking status, are founder-approved. Workstream D remains
sequenced after Slices 2 and 3; blocking MVP completion does not make it a Slice
2 prerequisite.

1. **Slice 3 output:** approve or revise the implemented
   `holdings_nonpositive_price` reviewer presentation before Slice 4 begins;
   approve the remaining candidates separately before their detection logic is
   implemented.
2. **YAML priority:** confirm that `review_priority: high | normal | low` and
   comparable workflow/presentation policy are deferred. Recommended: defer
   until real usage establishes a need.

No further visionary product-design phase is required to make these decisions.

---

# 11. MVP Completion Gate

The founder determines that PPAR Audit has reached MVP only after:

- the four required capabilities are implemented and reviewed;
- the generated portfolio and security outputs present the right information in
  a useful first view;
- additional issue types are defensible and controllable;
- cause/issue vocabulary and YAML behavior are stable and fail closed;
- every Section 7 transaction-semantic, exact-case, source-contract, and demo
  requirement satisfies its evidence, fixture, and safety gates;
- all financial, safety, output, determinism, demo, and scale gates pass; and
- remaining gaps are acceptable for a controlled validation partner.

Validation-partner planning begins after this gate, not before it.
