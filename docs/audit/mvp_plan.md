# PPAR Audit MVP Completion

## Concrete Definition and Implementation Plan

| Document field | Value |
|---|---|
| Status | Active MVP implementation plan — Slice 1 complete; Slice 2 is next |
| Version | 0.3 |
| Date | 2026-07-17 |
| Governing document | [`product_constitution.md`](product_constitution.md) |
| Historical specifications index | [`product_specifications_index.md`](product_specifications_index.md) |
| Scope | Three founder-defined reporting, Data Issues, and codification gaps required before MVP |
| Excluded | History, dashboards, Operational Intelligence, additional platforms, managed workflow, and broad speculative rules catalog |

---

# 1. MVP Definition

The current product is a substantial technical foundation, but the founder does
not yet consider it an MVP. MVP completion requires the current audit engine,
evidence pack, and safety controls plus three additional product capabilities:

1. an **Executive Summary** worksheet before `Performance Differences`, with an
   equivalent first HTML section;
2. additional enumerated **Issue Type** values and checks for `Data Issues
   Issues`; and
3. stronger codification of causes and issues, with safe, understandable user
   control in YAML.

The MVP is complete only when these capabilities are implemented, tested,
demonstrated in the packaged Axys/APX scope, documented for users, and preserve
all twelve safety invariants.

This is an implementation plan, not a new visionary product-design phase.

---

# 2. Verified Current Baseline

## 2.1 Review entrypoint

The current ordinary workbook order is exactly:

1. `Performance Differences`
2. `Performance Difference Causes`
3. `Data Issues`

There is no `Executive Summary` worksheet or equivalent first HTML section.
Both XLSX and HTML derive from the same ordered `ReviewWorkbookSheet` sequence,
which provides a strong shared-model foundation for the new entrypoint.

## 2.2 Current Data Issues vocabulary

The current code has seven optional issue-type strings:

- `duplicate_transactions`
- `dividend_rate`
- `holdings_accrued_rate`
- `holdings_price_range`
- `missing_dividend`
- `pa_sa_rate`
- `transactions_price_range`

It also has two mandatory continuity types:

- `portfolio_market_value_continuity`
- `security_market_value_continuity`

These values now serialize through public `DataIssueType` members without
changing their strings. `DATA_ISSUE_REGISTRY` carries the category,
mandatory/default enablement, required datasets, tolerance applicability, and
reviewer meaning for every implemented issue type.

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
2. **Executive Summary — next.** Build the visible entrypoint on stable product
   vocabulary and shared validated tables after the remaining scope decisions.
3. **Additional Data Issues checks** — add selected issue types one at a time
   through the new registry and YAML contract.
4. **Integrated release validation** — refresh documentation/demo artifacts and
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

Executive Summary display limits should initially be fixed, version-controlled
product behavior. Per-client priority labels, ordering policy, and similar
presentation controls are deferred until usage evidence demonstrates that YAML
control is valuable. This avoids speculating about how different users will
process the information.

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

It is a presentation and navigation layer over existing validated results. It
must not introduce a second financial calculation.

## 5.2 Minimum content

The first MVP version should show:

### Scope and interpretation

- portfolio or security report level;
- evaluated entity/period scope available from the run;
- snapshot labels/context available from the configuration;
- Modified Dietz review basis; and
- prominent limitation that PPAR explains supported changes but does not
  certify official performance correctness.

### Performance-change overview

- count of changed review units;
- counts of Fully Explained, Partly Explained, and Unexplained review units;
- count of residual-withheld or method-limited units where applicable; and
- a bounded list of priority changed review units with exact performance,
  explained, and unexplained values.

### Cause overview

- stable cause areas present in the report;
- count of affected review units by cause area; and
- evidence links or review keys to `Performance Difference Causes`.

Do not total or average return impacts across incompatible periods, portfolios,
securities, currencies, or return containers.

### Data Issues overview

- total affected rows;
- affected-row counts by stable issue type and category;
- mandatory continuity attention kept distinct; and
- evidence links or review keys to `Data Issues`.

Counts must be labeled as rows/entities where applicable, not unvalidated
“incident” counts.

### Next review step

- deterministic cues for Partly Explained/Unexplained results;
- deterministic cues for mandatory continuity and other high-priority issues;
- direct guidance to the existing detailed sheets and complete source evidence.

## 5.3 Presentation requirements

- Use progressive disclosure: concise first view, complete evidence retained.
- Preserve separate performance and Data Issues attention statements.
- Do not introduce a composite score, grade, confidence score, or “passed” badge.
- Do not introduce an explanation-completeness percentage.
- Use the same underlying summary model for XLSX and HTML.
- Generate an honest empty state when there are no performance differences,
  Data Issues rows, or both.
- Keep current detailed sheet names and semantics unchanged.
- Ensure evidence references remain stable and valid.

## 5.4 Acceptance

- Ordinary workbook order begins with `Executive Summary`.
- Ordinary HTML begins with the equivalent section.
- Current three analytical sheets remain in their existing relative order.
- Every summary count and displayed value reconciles to canonical detail.
- HTML/XLSX/machine semantics pass parity and deterministic-repeat tests.
- Bundle validation fails on missing, stale, or inconsistent summary content.
- Large-result first views are bounded without truncating calculation or
  retained evidence.

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

1. **`holdings_nonpositive_price`** — a nonzero holding quantity has a zero or
   negative price within an explicitly configured population.
2. **`transactions_nonpositive_price`** — a buy/sell-like transaction expected
   to carry a price has a zero or negative price; transaction-category policy is
   required so dividends, cash journals, and corporate actions are not treated
   as priced trades.
3. **`holdings_quantity_value_mismatch`** — quantity and market value have a
   suspicious zero/nonzero relationship within an explicitly configured
   population.

These should initially require narrow, visible YAML populations or exclusions.
Worthless securities, cash conventions, accrued-only rows, shorts, derivatives,
and vendor-specific valuation representations are known false-positive risks.

## 6.3 Deferred candidates

Do not include in the first MVP batch without stronger source evidence:

- stale-price conclusions;
- extreme price-change conclusions without split/corporate-action handling;
- amount-versus-price-times-quantity transaction arithmetic across mixed asset
  types;
- missing corporate actions or spin-offs;
- position roll-forward reconstruction;
- missing FX conclusions without a stronger currency/reference contract; or
- broad security-master validation.

These may become valuable later, but their false-positive and source-contract
surface is too large for an initial bounded implementation.

## 6.4 Acceptance

- Each approved issue type is an enum/registry member.
- Each has a packaged fixture or focused validation fixture with expected rows.
- YAML enable/disable, filter, and tolerance behavior is fail closed and tested.
- Reviewer output includes stable issue type, category, observed values,
  tolerance, explanation, and review key.
- No new issue changes `Performance Difference Explained`, residual, or
  analytical status.
- Demo coverage and fixture isolation checks protect the intended story.

---

# 7. Cross-Cutting Release Gates

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

# 8. Proposed Implementation Slices

## Slice 1 — Current vocabulary and strict Data Issues YAML — Complete

Implemented on 2026-07-17 as a behavior-preserving code change:

- add issue/cause enums and registry;
- migrate current constants and groupings to those types;
- add strict YAML validation;
- expose enabled-check summary through config validation; and
- prove current report artifacts remain semantically unchanged.

## Slice 2 — Executive Summary shared model

- define the minimum canonical summary table/object;
- add first-sheet/first-section rendering;
- add evidence links and empty states;
- extend bundle/parity/determinism contracts; and
- update demo and user documentation.

## Slice 3 — First approved additional issue type

- implement one rule end to end;
- add YAML, fixtures, reviewer wording, and false-positive cases;
- verify independent Data Issues treatment; and
- review output before adding the next type.

## Slice 4 — Remaining approved issue types and MVP release audit

- repeat the end-to-end rule process;
- run the full relevant suite and release-candidate checks;
- generate final representative portfolio/security bundles; and
- assess whether the founder considers the resulting product an MVP.

---

# 9. Remaining Founder Decisions Before Further Application Changes

1. **Executive Summary scope:** approve the minimum content and presentation
   boundaries in Section 5. Recommended: approve.
2. **First new issue types:** approve, revise, or replace the three candidates in
   Section 6.2 before their detection logic is implemented.
3. **YAML priority:** confirm that `review_priority: high | normal | low` and
   comparable workflow/presentation policy are deferred. Recommended: defer
   until real usage establishes a need.
4. **Summary display limits:** decide whether bounded row limits should be fixed
   product behavior initially or configurable in YAML. Recommended: fixed,
   version-controlled defaults for the MVP; add YAML only if real use requires
   it.

No further visionary product-design phase is required to make these decisions.

---

# 10. MVP Completion Gate

The founder determines that PPAR Audit has reached MVP only after:

- the three required capabilities are implemented and reviewed;
- the generated portfolio and security outputs present the right information in
  a useful first view;
- additional issue types are defensible and controllable;
- cause/issue vocabulary and YAML behavior are stable and fail closed;
- all financial, safety, output, determinism, demo, and scale gates pass; and
- remaining gaps are acceptable for a controlled validation partner.

Validation-partner planning begins after this gate, not before it.
