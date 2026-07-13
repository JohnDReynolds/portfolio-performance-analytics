# PPAR Roadmap

This is the central roadmap for PPAR. It covers Axys/APX-focused analytics,
performance auditing, onboarding, user-facing documentation, report evolution,
data auditing, and demo-data guardrails.

Detailed design reference remains in
[`performance_comparison_design.md`](performance_comparison_design.md).
The packaged-demo source boundary is defined in
[`performance_comparison_demo_source_contract.md`](performance_comparison_demo_source_contract.md).
Historical demo-generation notes remain in
[`operational_demo_data_notes.md`](operational_demo_data_notes.md) and
[`../scripts/generic_analytics_demo_data/GENERATION_NOTES.md`](../scripts/generic_analytics_demo_data/GENERATION_NOTES.md).

[axys-apx-blockers]: axys-apx-reference/reference/Chapter_01_Overview.md#axysapx-blockers

## How To Read This Roadmap

Start here when deciding what remains to do:

- **Current Open Items** is the active project backlog across analytics,
  Performance Auditing, onboarding, and public-facing documentation.
- **Axys/APX Extract Contract Review Map** shows the source-contract path that must
  stay aligned as demo fields or site-extract rules change.
- **Axys/APX blockers** in [Chapter 01][axys-apx-blockers]
  is the canonical summary of evidence gaps that block broader Axys/APX-native
  automation.
- **Implementation Phases** is historical implementation detail. Most phase
  notes are complete, superseded, or preserved as rationale rather than active
  work.
- **Transaction-Type Backlog** is the policy boundary for future transaction
  coverage.

## Current Status

The current Performance Auditing workflow is release-candidate quality for the
packaged Axys/APX demo scope. The portfolio and security demos generate
`portfolio_audit.*` and `security_audit.*`, review-summary metadata, manifests,
and CSV audit artifacts. The generated reports focus on Modified Dietz
evidence, source-data differences, Data Audit Issues, review-only context, and
conservative transaction row identity.

Completed guardrails now cover:

- the PyPI wheel package-data surface includes packaged demo CSV/YAML/README
  inputs but excludes source-checkout generation internals;
- the minimum source-data contract is documented and `validate_config` prints
  required datasets and required normalized columns;
- default YAML validation hard-stops before report generation when changed
  fields lack additive, evidence-only, or suppression treatment;
- packaged Axys/APX transaction extracts omit `TRANSACTION_ID`;
- no-ID transaction matching is conservative and case-sensitive;
- ambiguous Axys/APX-style `dp`, `li`, `lo`, and `wd` codes require source/destination
  or special-security context before classification;
- cost, settlement-date, and unsupported corporate-action rows remain review
  evidence unless an explicit future rule changes their treatment;
- `portperf.gain_loss` and `secperf.gain_loss` are documented as report-style
  performance-extract context, not native IMEX object claims or recomputed
  accounting-ledger values;
- report-bundle artifacts are classified as first-stop review surfaces,
  reviewer handoff metadata, audit/export backbone, supplementary diagnostics,
  or opt-in reconstruction diagnostics;
- default report/workbook review order starts with `Performance Differences`,
  `Performance Difference Causes`, and `Source Detail`; optional
  reconstruction diagnostics stay opt-in and secondary;
- durable reader-path docs use current sheet names and reject stale
  `Other Data Differences` / `Residual Evidence` language;
- the package-root `ppar.performance_comparison` API boundary is documented:
  workflow/report helpers stay root-exported while specialized policy and
  evidence-pack helpers remain direct-submodule imports;
- production performance-comparison code has no Pandas dependency in its core
  path; large source-data reads, joins, grouping, and validation remain in
  Polars, with row iteration concentrated in presentation and diagnostic
  assembly;
- report and workbook assembly reuse per-build return-reconstruction diagnostics
  instead of recomputing security checks for each sheet;
- the packaged portfolio and security reports intentionally include Fully
  Explained, intentional `Partly Explained`, and intentional `Unexplained`
  review examples;
- the packaged Axys/APX scenario calendar now keeps every demo period at or
  below the two-difference target, and the intra-month split-plan backlog is
  empty after splitting the crowded BALANCED and INCOME periods into focused
  review windows;
- portfolio and security report explanations intentionally use different
  transaction wording when the review level asks a different question;
- the Axys/APX Demo Freeze Decision Packet is mapped to concrete packaged-data,
  YAML, source-contract, test, and generated-bundle evidence;
- the packaged Axys/APX demo is accepted as the future `vendor: axys` preset seed,
  while preset implementation remains future work;
- `ppar setup` installs optional Python runner scripts beside the workflow data
  they use, so users can run Analytics and Performance Auditing without the
  CLI when they need scheduler, notebook, or automation entrypoints;
- `docs/architecture.md` provides a compact system map for commands, package
  boundaries, data flow, setup data, configuration boundaries, and report
  boundaries;
- `ppar setup` has been dry-run end to end through Analytics and Performance
  Auditing, including current-directory behavior and generated output clutter
  checks;
- the built wheel exposes only the `ppar` console script, includes the Axys/APX
  starter data needed by setup, excludes generated/source-checkout internals,
  and runs the installed setup, analytics, and Performance Auditing smoke
  path;
- release-readiness checks are documented in the repository guide, with
  `pyproject.toml` as the single package-version authority; the maintainer
  made an explicit version decision and selected `0.1.5` as the next release
  candidate version before tagging;
- `ppar.__version__` is exposed from installed package metadata, keeping the
  public version API tied to the same `pyproject.toml` release authority;
- the `0.1.5` distribution artifacts passed `twine check`, metadata and
  package-content inspection, and installed-wheel command smoke with the
  declared runtime dependency set available;
- local release notes now summarize `v0.1.5`, and the local release tag is
  intended to point at the final audited release-record commit before any push
  or publish;
- remote-release readiness was checked without publishing: local `main` is
  ahead of `origin/main`, the remote `v0.1.5` tag does not exist yet, and the
  remote branch remains an ancestor of local `main`;
- the `generic_analytics` dataset is retained as maintainer/demo
  infrastructure for README images, analytics regression tests, and operational
  demo-data derivation, not as the first-user onboarding path;
- generated bundle vocabulary, package-resource entrypoints, distribution
  package-data boundaries, retired setup/command terminology, Performance Auditing
  README screenshots, and first-run onboarding language are validated.
- portfolio and security review bundles use level-specific
  `portfolio_audit.*` and `security_audit.*` entrypoint names; manifests,
  generated READMEs, CLI output, and the public bundle contract share that
  naming boundary.

## Current Open Items

The remaining work is backlog expansion and targeted release hardening, not
core Modified Dietz report cleanup. Performance Auditing is the user-facing
workflow name; Performance Comparison remains the attribution subfeature inside
that workflow. Vendor-preset infrastructure is deliberately parked in Eventual
Deliverables even though the Axys/APX seed is accepted.

### Near-Term Release Hardening

| Area | Deliverable | Exit criteria |
| --- | --- | --- |
| Documentation freshness | Keep durable docs aligned with Performance Auditing terminology, current sheet names, artifact taxonomy, package-root API boundary, optional diagnostics flow, and the canonical Axys/APX blocker summary. | Metadata tests reject stale reader-path terms, preserve the normal review order, and keep blocker navigation discoverable. |
| First-run UX | Keep the first five minutes simple: run setup, run Analytics, run Performance Auditing, then replace CSVs. | Main README, setup-installed README, setup console output, and README screenshots tell the same minimal story before detailed configuration guidance. |

Standing maintenance criteria:

- Keep the project-root `README.md` marketing story aligned with generated
  Analytics and Performance Auditing outputs. Metadata tests now assert that
  embedded README image references resolve.
- Remove clearly unused compatibility shims or simplify report/workbook helpers
  only when a small, behavior-preserving change is obvious and covered by tests.
- Profile again only after meaningful data-size growth, report-shape changes,
  or new bottleneck evidence. New speed work should start from measured timings.

### Transaction And Policy Backlog

| Home | Open item | Exit criteria |
| --- | --- | --- |
| Packaged demo candidate | Additional external-flow variants and corporate actions only when they add a distinct reviewer story beyond the existing packaged `li`, `lo`, `wd`, `dp`, `rc`, `pd`, `ss`/`cs`, and CVNA split-processing cases. | Scenario intent, transactions, holdings, performance rows, YAML rules, report explanations, and source-contract language all align without implying universal Axys/APX behavior. |
| Test-only fixture | More `li` / `lo` external and neutral variants; additional `dp` / `wd` fee, sweep, and external-flow cases; tested candidate override profiles for `ai` and local `pa`/`sa` variants; uppercase reversal/cancellation; synthetic corporate-action rows. | Fixture proves expected semantics, failure mode, or review-only treatment without making the packaged demo less realistic. Candidate override profiles remain copy/adapt onboarding examples when local site evidence differs from conservative packaged defaults. |
| Evidence-blocked backlog | Mergers, spin-offs, ticker changes. | IMEX context, REP/report semantics, or real source samples identify required fields and ppar treatment well enough to avoid code-only classification. |
| Policy expansion | Fee/expense return-basis handling beyond the current scoped examples; settlement-date impact rules; fixed-income principal/accrual cases beyond ordinary interest and configured accrued value. | Explicit YAML policy, source evidence, Modified Dietz role, report wording, and tests are all present. |

### Transaction Coverage Expansion

Do not add "all transaction types" directly to the packaged Axys/APX demo. Expand
coverage in this order:

1. Add or update the transaction semantics matrix with the evidence needed for
   the transaction family.
2. Add test-only fixtures for code-only failure, context-required behavior, and
   review-only behavior.
3. Promote a row to the packaged demo only when it is realistic, internally
   consistent, and useful to the reviewer.

Packaged demo rows should stay narrow enough to tell a coherent Modified Dietz
story. Synthetic edge cases belong in test-only fixtures.

Current triage: the packaged Axys/APX demo already covers normal buys, sells,
ordinary income, fixed-income income, fixed-income `pa`/`sa` accrued-interest
adjuncts with paired trade context, contextual `li`, contextual `lo`, external
`wd`, fee-like `dp`, return of capital `rc`, principal paydown `pd`, short
sale/cover short `ss`/`cs`, and a CVNA split-processing correction. Do not add
more packaged rows merely for symmetry. Additional rows should wait until they
show a nonredundant situation that changes the reviewer conversation more
clearly than the existing packaged and site-variant fixtures.

### Data Auditing Direction

Data Auditing is the user-facing product name for the source-data consistency
checks currently configured under `data_audit_checks`. The near-term surface is a
`Data Audit Issues` worksheet included in Performance Auditing workbooks. The
longer-term direction is to let this capability stand on its own when it is
useful before, after, or apart from a specific performance restatement review.

| Area | Deliverable | Exit criteria |
| --- | --- | --- |
| Data Audit Issues worksheet | Keep the former `X-Ref Issues` worksheet as `Data Audit Issues` inside Performance Auditing workbooks. | The worksheet remains separate from `Performance Difference Causes` and `Source Detail`, runs on the union of Snapshot A and Snapshot B, and flags reviewer-oriented consistency issues without treating them as additive Modified Dietz causes. |
| Data Auditing marketing | Present Data Auditing as a distinct capability included with Performance Auditing rather than as a technical cross-reference implementation detail. | README/setup docs explain that Performance Auditing includes Data Audit Issues for suspicious source-data relationships such as prices, dividend rates, accrued-interest rates, and missing dividends. |
| Data Audit Summary | Add a `Data Audit Summary` worksheet before `Data Audit Issues`. | Review output shows simple issue counts by issue type, snapshot, portfolio, and security before reviewers inspect row-level detail. Keep the first version count-only and avoid scoring or prioritization until reviewer feedback proves it is useful. |
| Standalone Data Auditing workflow | Consider an optional standalone command and workbook, such as `ppar data_audit <site_or_yaml>`, that writes `data_audit.xlsx` and `data_audit.html` without requiring a performance-difference review. | The standalone workflow reuses the same YAML checks and source snapshots, produces a focused audit package, and does not duplicate Performance Comparison attribution logic. |
| Data Audit issue taxonomy | Keep issue types compact and stable. | Issue types use filter-friendly codes such as `duplicate_transactions`, `transactions_price_range`, and `missing_dividend`; row explanations carry the reviewer-facing narrative. |
| Data Audit profiles | Consider simple audit profiles such as `strict`, `normal`, and `quiet`, or documented tolerance presets if profiles feel too heavy. | Profiles or presets adjust tolerances without hiding the resolved YAML behavior from reviewers. |
| Broader Data Audit checks | Expand checks only when they are high-signal and explainable from available source-data. | Candidate checks include stale prices, missing beginning/end holdings, negative quantity surprises, cash/security flow mismatches, split factor versus quantity-jump plausibility, cash roll-forward checks, and richer accrued-interest or paydown checks when required reference data exists. |

### Longer-Term Deliverables

| Area | Deliverable | Exit criteria |
| --- | --- | --- |
| Richer APX demo | Create a second APX-oriented demo that starts from the packaged Axys/APX demo story but adds richer fields from the Axys/APX research only when they materially change validation, logic, Modified Dietz treatment, or user-facing reports. | The APX demo has its own source contract, data/YAML, generated reports, and tests. Added fields such as source/destination context or multi-currency data must affect comparison behavior or reviewer output. |
| Multi-currency expansion | Extend the implemented USD-base EUR/GBP demo only after validating additional quote conventions, portfolio currencies, and source mappings. | Reports continue to separate FX/rate effects from cash-flow and valuation effects without implying unevidenced vendor methodology. |
| Broader extract discovery | Review `docs/axys-apx-reference` for Axys/APX fields that justify new comparison behavior. | Candidate fields are accepted only with confidence notes, source-contract metadata, and a clear effect on validation or report output. |

### Eventual Deliverables

| Area | Deliverable | Exit criteria |
| --- | --- | --- |
| Windows setup validation | Try the full setup on a Windows machine. | From a clean Windows environment, install the package, run `ppar setup`, then successfully run the generated Analytics and Performance Auditing demos using the installed starter workspace. Record and address any Windows-specific path, shell, rendering, or dependency issues. |
| Historical classifications | Support effective-dated security names and classifications even though Axys/APX does not provide a native historical-classification model. | Define an explicit effective-date source contract, deterministic handling for gaps and overlaps, period-correct joins, validation and reporting behavior, and tests proving that attribution uses the classification effective for each performance period. |
| Vendor YAML presets | Add an explicit vendor preset keyword, such as `vendor: axys`, that expands to the accepted packaged Axys/APX demo YAML semantics behind the scenes while still allowing site YAML to override, suppress, or extend preset rules. | Preset expansion is documented, inspectable, and test-covered. The resolved effective YAML can be printed or exported for audit. Overrides have deterministic precedence. The preset does not imply universal Axys/APX behavior; it is versioned, tied to the accepted packaged Axys/APX demo/source contract, and still fails hard when required site-specific context is missing. |
| Axys/APX Python package boundary | Consider renaming the importable `ppar.axys` package to `ppar.axysapx` so the public Python API matches the Axys/APX product positioning. Treat this as a deliberate API rename, not a mechanical docs cleanup. | Imports, pyproject package metadata, tests, docs, and examples all use the chosen package boundary. Decide explicitly whether `ppar.axys` remains as a short-lived compatibility shim or is removed before first public PyPI release. Class names may remain `Axys...` initially if that keeps the change focused. Keep this separate from the future `vendor: axys` YAML keyword, which can remain unchanged until preset semantics are implemented. |
| Generic analytics removal | Eventually remove the `generic_analytics` packaged dataset and optional setup script if the Axys/APX analytics setup fully replaces its remaining maintainer roles. Do not delete it casually: it currently feeds README marketing images, analytics regression tests, optional-value tests, and operational Axys/APX demo-data derivation. | Removal is safe only after README image rendering uses Axys/APX analytics inputs, tests no longer depend on generic demo data, operational derivation scripts use another accepted source universe, package-data globs are simplified, and docs no longer reference generic analytics as active infrastructure. Until then, keep it as maintainer-only infrastructure rather than user-facing onboarding. |
| Report packaging | Maybe add optional report packaging that combines multiple HTML reports and image files into a single PDF handoff package. | PDF generation is opt-in, preserves the existing HTML/XLSX/PNG artifacts as first-class outputs, documents its rendering dependencies, and produces stable page ordering suitable for client review packets. |
| Analytics demo tuning | Revisit the Mega-Cap Alpha vs Mega-Cap Benchmark analytics demo to make the marketing story even stronger. The current demo is compelling, but it may be better if allocation effect is more visible and overall outperformance is modestly higher. | Updated demo data still feels realistic, preserves clear benchmark-relative attribution, and produces refreshed README images. Allocation effect becomes easier to see without making the story look contrived. |
| Tested site override profiles | Create user-friendly, tested candidate override profiles for Axys/APX transaction families that are plausible but not safe enough for the core `vendor: axys` preset. Candidate profiles may include fixed-income accrued interest, margin or negative interest, principal paydown, return of capital, short sale/cover short, and cash-journal patterns. | Each profile has sample source rows, YAML rules, expected normalized semantics, confidence labels, required evidence, and tests. Documentation clearly separates conservative preset defaults from copy/adapt override examples, warns that candidate profiles require site confirmation, and helps onboarding users choose profiles when unmapped transaction codes appear. |
| Ledger double-entry demo | Consider a separate demo for a ledger-style double-entry debit/credit system. Keep it outside the current Axys/APX Modified Dietz demo unless it clearly improves source-data validation, accounting-system integration, or reviewer understanding. | The demo has its own source contract, fixture data, YAML, and tests. It demonstrates balanced debit/credit postings, account-level footing, and mapping from ledger activity into the performance-comparison inputs without implying that ppar reconstructs a full accounting ledger by default. |
| Commercial licensing | Design a commercial PyPI licensing model for PPAR or a future Axys/APX audit product. Prefer a local-execution package with license activation, encrypted local activation token, periodic online validation, and a reasonable offline grace period. Keep calculations local so investment-firm portfolio data does not leave the client environment. | Licensing plan documents activation UX, evaluation licenses, organization-based tiers, offline activation, license-server architecture, subscription/revocation support, optional floating/network licenses, payment integration, machine-fingerprint tradeoffs, security limits, and sample activation code. The plan explicitly recognizes that Python licensing cannot fully prevent piracy; the goal is to make legitimate licensing easy and unauthorized use inconvenient. |

## Axys/APX Extract Contract Review Map

The Axys/APX extract guardrails now have a single review path:

```text
ppar/setup_templates/axysapx_performance_comparison/demo_extract_availability.yaml
  -> scripts/render_demo_extract_availability.py
  -> docs/axys-apx-reference/contracts/demo_extract_availability.md
  -> ppar/performance_comparison/extract_contract.py
  -> ppar/performance_comparison/transactions.py
```

The YAML contract records IMEX/REP availability confidence, candidate source
names, source strategy, and blocking context requirements. The renderer keeps
the human-readable contract current. Runtime validation uses the same contract, or a
site-specific `extract_contract.path`, to prevent ambiguous Axys/APX `dp`, `li`,
`lo`, and `wd` transaction codes from being classified from transaction code
alone when required context is absent.

## Core Idea

The portfolio and each security can be viewed as parallel return containers.

At the portfolio level, external contributions and withdrawals are flows into or
out of the portfolio.

At the security level, buys and sells are flows into or out of the security.

```text
Portfolio return:
  portfolio value change adjusted for contributions and withdrawals

Security return:
  security value change adjusted for buys and sells into or out of the security
```

This symmetry is the central design principle.

```text
Portfolio level:
  contribution = inflow
  withdrawal = outflow

Security level:
  buy = inflow to security
  sell = outflow from security
```

A buy of AAPL is not an external flow for the whole portfolio because cash moved
inside the portfolio from cash to AAPL. But from AAPL's own return perspective,
that buy behaves like a capital inflow.

## Conceptual Formula

The general return model is:

```text
Return = economic gain / adjusted capital base
```

For a Modified Dietz-style model:

```text
Return = (EMV - BMV - net flows + income/accrual components)
         / (BMV + weighted flows)
```

The exact numerator convention must be defined carefully by transaction category,
sign convention, and vendor behavior.

## Why This Matters

The current report can show changed holdings, changed transactions, changed
income/accrual fields, and changed reported performance. But without a full
return-reconstruction layer, some rows should remain evidence rather than
counted explanations to avoid double counting.

Example:

```text
AAPL holdings.market_value changed
AAPL transactions.amount changed
AAPL transactions.quantity/price/commission changed
```

Those rows are related, but not independent. If the report counts both the
changed ending market value and the changed transaction flow without a formal
return model, it can explain the same economic change twice.

A reconstruction layer would allow the report to say:

```text
The reported security return changed because:
  EMV changed
  dated security-level flows changed
  income/accrual changed
  denominator/timing changed
```

That is cleaner and more defensible.

## Portfolio-Level Reconstruction

At the portfolio level, return inputs should be reconstructed from source-data
where possible:

```text
BMV = sum beginning holdings.market_value + holdings.accrued
EMV = sum ending holdings.market_value + holdings.accrued
External flows = transactions categorized as contributions/withdrawals
Weighted flows = external flows weighted by date convention
```

Example YAML shape:

```yaml
portfolio_return_reconstruction:
  method: modified_dietz
  beginning_value_source: holdings
  ending_value_source: holdings
  flow_source: transactions
  flow_timing: transaction_date
  day_count: actual_days
  inclusion_rule: beginning_of_day
  flow_categories:
    - external_flow
```

The comparison can then derive:

```text
derived_portfolio_return_a
derived_portfolio_return_b
derived_return_difference
reported_return_difference
reconstruction_difference
```

The derived return should probably not be too prominent at first. It can appear
as a check or comment, especially when it differs materially from the
vendor-reported return.

## Return Basis Policy

Fee and expense transactions need an explicit return-basis policy. A fee is
always a real cash movement, but whether it reduces reported performance depends
on whether the return is net of fees or gross of fees.

For net-of-fees performance:

```yaml
FEE:
  transaction_category: fee_expense
  cash_flow_sign: negative
  performance_flow_sign: performance
  return_basis: net
```

For gross-of-fees performance:

```yaml
FEE:
  transaction_category: fee_expense
  cash_flow_sign: negative
  performance_flow_sign: external_flow
  return_basis: gross
```

`return_basis` should become a required YAML item wherever fee or expense
transactions are eligible for return reconstruction. Without it, a workbook can
look numerically precise while using the wrong interpretation of fees.

## Security-Level Reconstruction

At the security level, the same idea applies:

```text
BMV = beginning security holdings.market_value + accrued
EMV = ending security holdings.market_value + accrued
Security-level flows = buys/sells/transfers for that security
Income = dividends/interest/fees/accrual changes for that security
```

Example YAML shape:

```yaml
security_return_reconstruction:
  method: modified_dietz
  beginning_value_source: holdings
  ending_value_source: holdings
  flow_source: transactions
  flow_timing: transaction_date
  day_count: actual_days
  inclusion_rule: beginning_of_day
  flow_categories:
    - buy
    - sell
    - transfer
  income_categories:
    - income
    - fee_expense
```

A buy is a security-level inflow. A sell is a security-level outflow.

## Avoiding Double Counting

Once reconstruction exists, the report should distinguish formula inputs from
supporting evidence.

Formula-level inputs might include:

```text
BMV
EMV
weighted flows
income
accrual
```

Supporting inputs might include:

```text
transaction.quantity
transaction.price
transaction.commission
holding.quantity
holding.price
holding.cost
```

Example hierarchy:

```text
Security return changed
  EMV changed
    holdings.market_value changed
      holdings.quantity changed
      holdings.price changed

  Weighted flow changed
    transactions.amount changed
      transactions.quantity changed
      transactions.price changed
      transactions.commission changed
```

Only actual source-data rows should normally receive a counted `Performance
Difference Explained`. Reconstruction formula math may be used internally, but
the default user-facing workbook should attribute the explanation back to rows
the user recognizes from holdings and transactions.

## Possible Report Structure

A future workbook could include:

```text
Performance Differences
Performance Difference Causes
Return Reconstruction Checks
Source Detail
```

Current status: the normal user-facing workbook now uses `Performance
Differences`, `Performance Difference Causes`, and `Source Detail`.
Transaction matching diagnostics remain available in the
bundle CSVs and source detail rather than as a standalone default sheet.
Reconstruction diagnostic sheets remain opt-in so the default workbook stays
focused on reviewable performance differences and their source-data causes.

`Performance Difference Causes` should contain source-data rows:

```text
holdings.market_value
holdings.accrued
transactions.amount
```

Supporting rows such as `holdings.quantity`, `holdings.price`,
`transactions.quantity`, `transactions.price`, and `transactions.commission`
can appear beneath those source rows when they help explain how the counted
source row changed.

A useful future column might be:

```text
Formula Role
```

Possible values:

```text
Beginning Value
Ending Value
Weighted Flow
Income
Accrual
Supporting Input
Context
```

## YAML Design Principle

The YAML must be explicit and required.

No silent defaults.

If the user asks for portfolio or security return reconstruction and omits
required rules, the system should fail before report generation.

Required concepts for every return-reconstruction method:

```yaml
method
beginning_value_source
ending_value_source
flow_source
flow_categories
income_categories
return_basis
sign_convention
```

Additional concepts required only for `method: modified_dietz`:

```yaml
flow_timing
day_count
inclusion_rule
```

Supported return-reconstruction methods:

- `simple_dietz`: denominator is beginning value; flow timing fields are not
  valid because flows are not weighted.
- `modified_simple_dietz`: denominator is beginning value plus 50% of net flows;
  flow timing fields are not valid because every flow receives the same
  mid-period weight.
- `modified_dietz`: denominator is beginning value plus date-weighted flows;
  `flow_timing`, `day_count`, and `inclusion_rule` are required.

The YAML validator should fail hard when a method is missing fields it needs,
includes fields that method cannot use, repeats keys, or includes unknown keys.

Potential future additions:

```yaml
large_flow_policy
subperiod_linking
settlement_date_handling
cash_security_identifier
accrual_treatment
corporate_action_treatment
```

## Implementation Phases

The phase notes below are an implementation journal. Treat a phase as active
only when its status says partial, backlog, future, or superseded with a named
follow-up. Completed phases are retained to explain why the current bundle,
tests, and source contracts look the way they do.

### Phase 1: Security-Level Flow Model

- Implement explicit security-level Modified Dietz reconstruction.
- Treat buys and sells as security-level flows.
- Treat dividends and interest as income inputs.
- Compare derived security return to reported security return.
- Keep outputs conservative at first.

Initial implementation note: performance comparison now supports an optional
security-level `Security Return Checks` worksheet and CSV artifact when YAML
includes `security_return_reconstruction`. The check derives Modified Dietz
security returns from holdings, buy/sell transaction flows, and income
transactions. It compares derived security-return differences with reported
`SEC_RETURN` differences and remains diagnostic-only.

### Phase 2: Portfolio-Level Flow Model

- Implement explicit portfolio-level Modified Dietz reconstruction.
- Treat contributions and withdrawals as portfolio-level external flows.
- Do not treat buys and sells as portfolio-level external flows.
- Compare derived portfolio return to reported portfolio return.

Initial implementation note: performance comparison keeps portfolio external-flow
Modified Dietz estimates as review-only cross-checks. The dedicated return
reconstruction check is the place to compare reported returns with values derived
from holdings, income, and external-flow transactions.

### Phase 3: Return Reconstruction Checks

- Add a worksheet or report section showing:
  - reported return
  - derived return
  - difference
  - materiality status
- Use this as a diagnostic before making reconstruction the primary user-facing
  explanation.

Initial implementation note: performance comparison now supports an optional
portfolio-level `Return Reconstruction Checks` worksheet and CSV artifact when
YAML includes `portfolio_return_reconstruction`. The check derives Modified
Dietz portfolio returns from holdings plus external-flow transactions and
compares them with reported `PORT_RETURN` differences. The worksheet also shows
the derived numerator, derived denominator, beginning value, ending value, net
flow, weighted flow, and changed formula components so a reviewer can see why a
check is marked `Different`.

Diagnostic QA note: report bundles can explicitly include a `Reconstruction
Summary` worksheet/CSV artifact when portfolio or security reconstruction is
enabled. The normal user-facing bundle excludes these diagnostics by default.
When opted in, the summary counts each reconstruction check by status and
diagnostic category, using plain categories such as `Aligned`, `Missing Inputs`,
`Source Inputs Changed`, and `Formula Difference`. The detailed check sheets
include the same category and clearer comments so reviewers can quickly
distinguish source-input changes from missing inputs or model/vendor
differences.

### Phase 4: Attribute Formula Impacts To Source Rows

- Once reconstruction is trusted, use reconstructed formula inputs internally as
  the calculation basis.
- Allocate formula-level impacts back to source rows such as
  `holdings.market_value`, `holdings.accrued`, and `transactions.amount`.
- Show quantity, price, commission, cost, and reference-data rows as supporting
  evidence unless they directly explain a counted source row.

Status note: source-row attribution is implemented for both security and
portfolio checks where `reconstruction_category` is `Source Inputs Changed`.
The workbook uses formula roles such as beginning value, ending value, net flow,
weighted flow, and income internally, then allocates their calculated impact
back to recognizable rows such as `holdings.market_value` and
`transactions.amount` in `Performance Difference Causes`. `Return Reconstruction Checks`
and `Security Return Checks` remain opt-in interim audit trails for the raw
numerator, denominator, and source component inputs.

### Phase 5: Deterministic User-Facing Explanations

- Add a structured explanation layer for `Performance Differences` comments and
  row-level `Explanation`.
- Generate comments from cause/residual patterns, not from free-form inference.
- Prefer specific worksheet and field references over generic instructions.
- Keep comments short enough for the workbook, with detailed evidence remaining
  in `Performance Difference Causes` and `Source Detail`.
- Make guidance action-oriented and understandable when formula rows and
  supporting rows overlap.

The original pain point was crowded periods such as the former BALANCED
`2026-05-01` to `2026-05-29` period, where:

- `holdings.ending_market_value` explains the positive calculated difference;
- `holdings.beginning_market_value` explains a negative denominator effect;
- individual AAPL/MSFT `holdings.market_value` rows foot to the ending-value
  effect but are supporting detail rather than separate additive causes; and
- the `Explanation` text was technically accurate but not helpful
  enough for a normal reviewer.

Future guidance should make those relationships explicit. For example:

```text
Ending holdings value increased; AAPL and MSFT holding rows below show the
security-level source changes that make up this ending-value effect.
```

```text
Beginning holdings value increased, which reduces the calculated return. This
denominator effect accounts for the Unexplained Difference in the Performance
Differences sheet.
```

```text
Supporting detail for the ending holdings value effect; included in the related
performance difference, not counted separately.
```

Example residual classifications:

```text
fully_explained_by_performance_difference_causes
residual_matches_beginning_value_effect
residual_matches_ending_value_effect
residual_matches_transaction_flow_effect
residual_has_identifiable_evidence_but_no_supported_estimate
no_identifiable_evidence_found
```

Example deterministic comment:

```text
The Unexplained Difference matches the beginning holdings market value effect
shown in `Performance Difference Causes`. Reported performance appears to reflect the
ending value change but not the beginning-value denominator effect.
```

Implementation guidance:

- Do not speculate about vendor behavior beyond what the source rows show.
- Use careful language such as "appears to" when the workbook can identify a
  pattern but cannot prove the vendor's calculation method.
- Avoid generic instructions such as "review other sheets" when the workbook
  can name the exact field or row family involved.
- Use consistent terms for formula rows, source rows, and supporting detail so
  reviewers do not have to infer whether a row is counted or merely related.
- Add tests for each explanation type:
  - given a known cause/residual pattern
  - expect a specific comment
  - expect specific row-level `Explanation`
  - avoid untested prose drift
- Keep supporting-row explanations plain enough that reviewers do not need a
  second performance-number column beside `Performance Difference Explained`.

### Phase 6: Generate Holdings From Scenario Inputs

Current packaged demo rebuilding now derives snapshot B `transactions.csv` from
snapshot A transactions plus validated explicit transaction scenarios in
`scripts/operational_demo_data/performance_comparison_transaction_scenarios.csv`.

It then derives snapshot B `holdings.csv` from snapshot A holdings plus two
deterministic sources:

1. transaction-derived holding impacts from changed `by`, `sl`, `wd`, `dv`,
   `in`, and fee-like `dp` rows;
2. validated explicit residual holding scenarios in
   `scripts/operational_demo_data/performance_comparison_holding_scenarios.csv`.

It then derives `secperf.csv` and `portperf.csv` from those holdings and
transactions under the configured YAML reconstruction rules. This removes one
major source of fixture drift: transaction changes and common related
cash/security holding changes now come from scenario intent and rules rather
than independent hand-maintained rows.

The scenario layers are intentionally strict:

- columns must exactly match the expected schema;
- scenario rows can target only derived snapshots, not the base snapshot;
- duplicate scenario keys are rejected;
- each row must change at least one holding value;
- each transaction scenario row must match exactly one base transaction row;
- each holding scenario row must match exactly one packaged holding row;
- transaction scenarios currently change only existing transaction rows and only
  numeric fields (`QTY`, `PRICE`, `AMOUNT`, and `COMMISSION`).

Residual holding scenarios now carry explicit `scenario_type` values:

- `valuation_mark`
- `cash_balance_correction`
- `quantity_valuation_correction`
- `accrual_correction`
- `cost_only_correction`

Each type is validated against the holding fields it is allowed to change. This
keeps remaining manual holding adjustments from becoming generic numeric
patches.

The rebuild/audit summary now exposes the scenario story directly:

- transaction scenarios grouped by transaction code;
- transaction-derived holding impacts grouped by transaction code;
- residual holding scenarios grouped by scenario type.

This gives a quick check that the demo still contains the intended examples
without requiring manual CSV inspection.

The same scenario coverage is now part of the audit guardrail. If an example
type disappears accidentally, the packaged demo-data audit fails before the
reports are regenerated.

Status: complete for the current packaged performance-comparison demo fixture.

### Phase 7: Performance Explanation Scenario Engine

Phase 7 should remain focused on answering one product question:

```text
Why did reported performance change from Snapshot A to Snapshot B?
```

It should not become a general accounting engine. The goal is to generate demo
fixtures that consistently explain performance differences from recognizable
source-data changes, while leaving non-performance accounting differences as
raw audit evidence.

Useful Axys/APX transaction references:

- [`Chapter_05_Transactions.md`](axys-apx-reference/reference/Chapter_05_Transactions.md):
  draft transaction reference with evidence boundaries and confidence levels.
- [`Research_05_Transactions.md`](axys-apx-reference/evidence/Research_05_Transactions.md):
  consolidated research reference for transaction workflows, code evidence,
  dependencies, audit rules, contradictions, and known unknowns.

These references should inform transaction-code interpretation, sign/cash-flow
assumptions, and evidence caveats. They should not be treated as instructions to
reconstruct tax lots, cost methods, settlement accounting, or vendor books.

#### Phase 7A: Performance Explanation Engine Contract

Phase 7A is a design contract, not an implementation phase. It defines what the
performance-comparison engine is allowed to explain and what must stay as
review-only evidence.

Contract question:

```text
What source-data changes are allowed to explain reported performance changes?
```

Performance-explaining inputs:

- beginning holdings market value plus accrued amount;
- ending holdings market value plus accrued amount;
- dated external cash-flow transaction amounts;
- dated security-level buy/sell transaction amounts;
- dividend, interest, fee, and other income/expense amounts when configured;
- reported accrual amounts.

Supporting evidence:

- quantity;
- price;
- commission;
- transaction type;
- transaction date;
- cash-balance movement;
- security identifiers and portfolio identifiers needed to connect evidence to
  the affected performance input.

Raw audit evidence:

- cost and cost-basis fields;
- broker;
- settlement date, unless a future supported rule makes it performance-relevant;
- security reference data;
- any changed field that is not part of the supported performance explanation
  model.

Hard-fail rules:

- required YAML is missing for transaction classification;
- an unknown transaction code appears in performance-relevant data;
- a performance-relevant source field changes without a configured
  interpretation;
- ambiguous Axys/APX `dp`, `li`, `lo`, or `wd` transaction codes appear in an
  extract that lacks the transaction context fields required by the packaged
  Axys/APX extract contract;
- generated transactions, holdings, `secperf.csv`, or `portperf.csv` drift from
  their scenario-derived values;
- expected scenario coverage disappears unexpectedly;
- an unsupported field attempts to explain performance.

Implemented guardrail:

- transaction loading now fails up front when a transaction row cannot resolve a
  known transaction category from a source category, known transaction code, or
  `transaction_rules` entry. This guard is intentionally narrower than
  sign/flow impact-policy validation: known transaction codes with incomplete
  return-impact policy still flow through the existing review workflow instead
  of changing report behavior.
- Axys/APX IMEX transaction codes `li`, `lo`, `dp`, and `wd` are not safe
  external-flow indicators by code alone. The packaged demo now uses
  conditional YAML `transaction_rules` with normalized IMEX context fields such
  as `security_type`, `source_destination_type`,
  `source_destination_symbol`, `special_security_type`, and
  `special_security_symbol`. A `wd` row is classified as an external flow only
  when its cash security and source/destination context match the external
  party cash rule; `dp` is classified as fee expense only when special-security
  context confirms the fee case; `li`/`lo` rules distinguish external party
  flows from internal transfer cases. If an IMEX export cannot provide the
  context fields marked as required by
  `ppar/setup_templates/axysapx_performance_comparison/demo_extract_availability.yaml`, the transaction loader
  fails before YAML rules classify the rows. The next design step for that site
  is to consider a REP/report extract, custom report, or other local-discovery
  source that carries enough classification evidence.
- comparison YAML can now override the default packaged extract contract with:

  ```yaml
  extract_contract:
    path: site_extract_contract.yaml
    enforce_ambiguous_axys_flows: true
  ```

  `enforce_ambiguous_axys_flows` defaults to `true`; setting it to `false` is
  an explicit local opt-out.
- local extract contracts are validated by `validate_config`: the contract must
  define `datasets.transactions.csv.columns`, use supported transaction column
  aliases, and provide boolean `requires_context_for_semantics` and
  `blocking_if_missing` flags.
- `docs/axys-apx-reference/contracts/templates/site_extract_contract.yaml` is the starter
  template for site-specific contracts.
- packaged demo-data audit tests now assert that the user-facing `wd` withdrawal
  rows resolve as external flows and the fee-like `dp` rows resolve as
  fee-expense performance rows in both packaged snapshots.
- `tests/data/axys/site_variants/` contains small site-shape fixtures for the
  next hardening layer: IMEX rows with context fields, REP/report rows with
  reviewed semantics, and code-only IMEX rows that must fail for ambiguous Axys
  flow codes.

#### Phase 7B: Transaction Rule Coverage Validation

Status: complete for the current packaged performance-comparison demo fixture.

The configuration validator now reports:

- observed transaction codes across configured transaction files;
- observed transaction codes that do not have explicit YAML `transaction_rules`.

This summary is informational for ordinary comparison configurations so older
review-policy fixtures can still demonstrate incomplete YAML behavior. The
packaged performance-comparison demo has a stricter guardrail test: every
observed transaction code must be explicitly defined in its YAML
`transaction_rules`.

Ongoing coverage requirement:

- Tests and packaged demo data must continue expanding toward complete Axys/APX
  transaction-type coverage, not just the currently performance-relevant rows.
  Each observed or documented Axys/APX transaction type should have an explicit
  expected classification, even when the expected outcome is `transfer`,
  `corporate_action`, `unknown pending review`, or review-only evidence.
- [`axys-apx-reference/contracts/transaction_semantics_matrix.md`](axys-apx-reference/contracts/transaction_semantics_matrix.md)
  is the implementation-facing seed matrix for that coverage. Future fixtures
  should either satisfy a row in that matrix or update the matrix with the new
  evidence and expected treatment.
- `docs/axys-apx-reference/contracts/transaction_semantics_matrix.yaml` is the
  machine-readable coverage contract. Coverage is complete only when every row
  has a non-backlog fixture or an explicit documented reason to remain
  review-only/unknown.
- [Boundary snapshot](performance_comparison_transaction_boundary_snapshot.md)
  is the compact reviewer-facing view of the same coverage boundary.
- Matrix rows and pair patterns now include machine-checked `coverage_notes`, so
  each covered, context-only, partial, or backlog entry carries a rationale
  instead of relying only on table prose.
- External-flow coverage must include both positive and negative capital-flow
  cases, internal transfers, cash sweeps, fee/expense rows, income rows,
  corporate-action rows, and correction/cancellation/reversal-like rows when
  representative source evidence is available.
- Ambiguous `li`, `lo`, `dp`, and `wd` cases require multiple examples for each
  code: one or more true external-flow examples and one or more non-external
  examples whose context proves transfer, sweep, fee/expense, or another
  non-capital-flow treatment. Code-only examples must remain hard-fail fixtures.
- If IMEX cannot expose the context needed for a documented transaction type,
  tests should model the REP/report or custom-report fallback instead of
  weakening the IMEX guardrail.

#### Phase 7C: Packaged Demo Source Contract Audit

Status: complete for the current packaged performance-comparison demo fixture.

The packaged demo source contract is documented in
[`performance_comparison_demo_source_contract.md`](performance_comparison_demo_source_contract.md).
It states that the packaged CSVs are normalized demo extracts, not official
Axys/APX native schemas, and defines the narrow source-data fields ppar is
allowed to use when explaining performance differences.

The packaged demo audit now verifies the user-facing report tables follow the
source contract:

- `Performance Difference Causes` only contains approved demo source fields;
- holdings `cost` stays in `Source Detail`;
- configured holdings `accrued` changes remain performance-cause rows.

The Axys/APX transaction references should be used to choose careful terminology
and conservative interpretation rules. The package should adopt only the rules
needed to explain performance changes. It should not adopt a broader accounting
model merely because the reference material describes one.

A future foundation pass should move from generic scenario-adjustment files to a
performance-explanation scenario engine that generates both holdings snapshots
from explicit starting performance inputs, transactions, valuation marks,
reported accrual amounts, and cash-balance rules. The remaining explicit holding
scenario rows should shrink as more performance-relevant transaction,
valuation, and accrual amount rules become deterministic.

The current guardrail tests pin the accounting impact of each supported simple
transaction type:

- `wd` and fee-like `dp` reduce ending `CASHUSD` holdings;
- `dv` and `in` increase ending `CASHUSD` holdings;
- `by` increases the traded security holding and reduces cash;
- `sl` reduces the traded security holding and increases cash;
- the packaged CVNA split-processing correction demonstrates a realistic
  corporate-action miss within the demo period: Snapshot A carries stale
  quantity at the split-adjusted price, while Snapshot B corrects quantity and
  ending market value.

Suggested source-of-truth inputs:

```text
starting holdings
intentional scenario changes
transactions
valuation marks / prices
reported accrual amounts
cash-account rules
security reference data
```

The generator should then derive:

```text
holdings.csv
secperf.csv
portperf.csv
```

This would make the demo stack flow in one direction:

```text
scenario intent
  -> transactions
  -> transaction-derived holding impacts and valuation assumptions
  -> holdings
  -> security performance
  -> portfolio performance
  -> reports
```

Required performance-explanation rules before this phase should be considered
complete:

- starting positions, cash, market value, and accrued amounts
- trade sign conventions
- trade-date versus settlement-date treatment
- cash offsets for buys, sells, dividends, interest, fees, contributions, and
  withdrawals
- market value formula, including whether accrued is separate from market value
- reported fixed-income accrual amount treatment
- corporate-action treatment for splits and future action types
- real-world corporate-action fixture policy: user-facing demo splits must use
  actual historical split dates/securities, not fictional future splits; purely
  synthetic corporate-action cases belong in clearly labeled test-only fixtures.
- cash security identifier policy, including multi-currency cash accounts
- rounding policy for quantities, prices, market values, cash, and performance

The generator and auditor should stay coupled. A generated dataset should fail
before writing if:

- holdings no longer foot to the generated transactions and valuation marks;
- cash does not move correctly for transaction types that affect cash;
- `secperf.csv` does not reconstruct from holdings and security-level flows;
- `portperf.csv` does not reconstruct from holdings and portfolio-level flows;
- any intentional partly explained or unexplained scenario is not named in an
  allowlist with a reviewer-facing reason.

This phase is intentionally larger than the current scenario guardrails, but its
scope should stay narrow. It should explain performance-input changes, not
rebuild portfolio accounting. Cost differences should remain visible as
raw audit evidence unless they become direct inputs to a supported
performance explanation.

### Phase 8: Improve Demo Data

Current packaged demo coverage includes realistic examples of:

- portfolio contribution
- portfolio withdrawal
- security buy
- security sell
- dividend
- interest
- fee
- accrual change

Next expansion should be ordered by realism and by the evidence needed to
classify Axys/APX transaction semantics safely.

| Priority | Scenario family | Permanent home | Required evidence before implementation | Notes |
| --- | --- | --- | --- | --- |
| 1 | Additional contribution or withdrawal variants | Test-only first, packaged demo only when nonredundant | Cash security, external-party source/destination context, amount sign, Modified Dietz flow-weighting policy. | The packaged demo now includes one ordinary `li` contribution, one contextual `lo` deliver-out, and one `wd` withdrawal. Add more only when the scenario teaches a new review behavior. |
| 2 | `li` / `lo` external-flow and transfer examples | Test-only first, packaged demo only when realistic | Source/destination type and symbol, security type, amount/quantity signs, or reviewed REP semantics. | Site-variant fixtures and the packaged `li`/`lo` external-flow rows prove the classification patterns. Add transfer examples to the packaged demo only if the business story is realistic and not redundant with withdrawal/contribution. |
| 3 | Additional `dp` / `wd` variants | Test-only first | Special-security context, sweep/cash symbols, source/destination context, and explicit fee/transfer/external-flow treatment. | Keep proving code-only `dp` and `wd` are unsafe. Packaged demo should stay focused on one fee-like `dp` and one external `wd` until a stronger story is needed. |
| 4 | Fixed-income accrued-interest, maturity, and principal-paydown cases (`pa`, `sa`, `ai`, `pd`) | Test-only first, then packaged demo when accounting rules are deterministic | Bond or amortizing security type, accrued-interest treatment, cash offset, principal movement, gain/loss or income treatment, local mapping or REP evidence. | `pa`/`sa` now have a narrow packaged paired-trade story plus test-only local variants. `pd` now has a narrow packaged principal-paydown story with MBS/amortizing-security, principal-paydown, and portfolio-cash destination context. Keep `ai` out of the packaged demo until holdings, accrual, cash, `secperf.csv`, and `portperf.csv` all derive from one coherent scenario intent. |
| 5 | Return of capital (`rc`) and principal paydown (`pd`) | Test-only first, packaged demo only with explicit context | Security, amount sign, local mapping or REP/report treatment, and whether return is Modified Dietz performance income or review-only corporate-action evidence. | The packaged demo now includes context-gated `rc` and `pd` rows, while test-only site variants continue to prove local override profiles. Cost, principal, and amortization handling are best-efforts demo-construction context, not the driver of Modified Dietz classification. |
| 6 | Real-world split / corporate action evidence | Packaged demo for one coherent story; test-only for broader variants | Actual historical date/security, split ratio, quantity and price treatment, report evidence, and a policy for whether it is explanatory or review-only. | The packaged demo includes a CVNA 5-for-1 split-processing correction in May 2026. Additional user-facing split examples should add a new reviewer lesson; synthetic corporate-action fixtures belong in clearly labeled test-only data. |
| 7 | Short sale / cover short (`ss`, `cs`) | Packaged synthetic demo plus test-only variants | Lowercase code, short/security type or resulting negative exposure, cash/margin/short symbols, source/destination context, amount/quantity signs, and reviewed local treatment. | Public APX integration evidence now strongly supports `ss` as short sale and `cs` as cover short. The packaged demo includes one disclosed same-period TSLA short sale / cover short lifecycle using real May 2026 prices; production use still needs local short-account evidence so ppar does not imply universal cash/proceeds or holdings conventions. |
| 8 | Correction/cancellation/reversal-like uppercase rows | Test-only first | Link to original transaction or enough matching fields to identify the reversal target. | Demonstrate review-only or correction behavior without treating an unlinked uppercase row as a new economic event. |

#### Phase 8A: Realistic Transaction Expansion Gate

The packaged-demo contribution scenario passes the same end-to-end standard as
the withdrawal example. Before adding any additional external-flow variant to
`snapshot_a` or `snapshot_b`, document and verify:

- scenario intent: which portfolio, period, cash security, and reviewer-facing
  business story the contribution represents;
- source evidence: the Axys/APX-style transaction code, amount sign,
  source/destination type, source/destination symbol, and any REP/report
  semantic fields needed to prove it is an external capital inflow;
- YAML semantics: the `transaction_rules` entry classifies the row from
  context, not from an ambiguous code alone, and preserves the correct
  `external_flow` cash/performance signs;
- cash and holdings: the transaction-derived cash movement reconciles to the
  generated ending `CASHUSD` holding for the affected period;
- return reconstruction: `portperf.csv` and the portfolio Modified Dietz
  reconstruction agree on net external flow, weighted flow, and return impact;
- report behavior: workbook and CSV artifacts explain the changed contribution
  without double-counting security-level buy/sell, income, fee, or holdings
  effects;
- docs and tests: this roadmap, the demo source contract, the packaged demo
  README, the semantics matrix, and audit tests all name the expected treatment.

Implemented contribution recipe:

- add the contribution as an inserted transaction scenario, not by mutating an
  unrelated base transaction row;
- use an Axys/APX-style `li` row on `CASHUSD` with `SRC_DEST_TYPE=$pty`,
  `SRC_DEST_SYMBOL=$cash`, positive `AMOUNT`, zero quantity/price/commission,
  and same-day settlement unless site evidence says otherwise;
- let the rebuild script derive snapshot B `transactions.csv`, ending cash
  holdings, `portperf.csv`, and reconstruction diagnostics from the scenario;
- keep future external-flow rows out of packaged CSVs until the generated
  workbook has no unintended partly explained or unexplained period.

Rebuild guards now mirror this for both directions: inserted `li` cash rows
prove external contributions increase cash, and inserted `lo` cash rows prove
external deliver-out/withdrawal-style rows decrease cash when external-party
context is present. The packaged demo includes one realistic `li` contribution,
one realistic `lo` deliver-out, and one `wd` withdrawal example; further
variants should remain test-only until a separate user story adds reviewer
value.

`li`/`lo`, additional `dp`/`wd`, and synthetic corporate-action scenarios should
remain test-only until they meet their own version of this gate. A real-world
split can move into the packaged demo only when the demo period and security
support an actual historical split date and the row is clearly tied to
review-only or implemented corporate-action behavior.

#### Phase 8B: Reinvestment Pair Feasibility Gate

The next useful test-only pair family is a reinvested dividend represented by a
`dv` income leg plus a related `by` purchase leg. For Modified Dietz, this does
not require accounting-style pair matching before the rows can be classified.
The required formula boundary is narrower: `dv` is income, and `by` is a
security-level flow that must not become a portfolio external contribution.

Status: partial test-only coverage. Return-reconstruction tests now prove the
`by` leg is not treated as a portfolio external flow and the `dv` leg is not
counted twice as security income. Lightweight pair detection remains optional
reviewer polish for grouping related evidence, not a prerequisite for Modified
Dietz support.

Before adding a reinvestment fixture, require:

- formula-role evidence: same portfolio, security or reinvestment target, date
  or settlement window, and enough source context to show the dividend income
  and security-level buy are not portfolio-level external flows;
- YAML semantics: `dv` remains performance income and `by` remains a
  security-level flow, with no portfolio-level external-flow treatment;
- double-count guard: portfolio return reconstruction must not count the buy
  leg as an external contribution, and security reconstruction must not count
  the dividend income twice;
- report behavior: workbook rows may show likely related income and buy
  evidence together, while making clear which formula input each row supports;
- fixture home: synthetic reinvestment examples belong in test-only data until
  a realistic packaged period and security story makes the example useful to a
  reviewer.

#### Phase 8C: Fixed-Income Transaction Boundary Gate

Status: complete for current packaged fixed-income guardrails. The packaged
demo supports ordinary `in` interest rows, `holdings.accrued` as proved
fixed-income performance inputs, paired-trade `pa`/`sa` accrued-interest
adjuncts, and a narrow `pd` principal-paydown story. Test-only site variants
still prove explicit YAML classification for margin-style `ai` and local
`pa`/`sa`/`pd` variants where site evidence differs from the packaged defaults.

Before adding any new accrued-interest or principal-paydown transaction
examples, require:

- source evidence: bond or accrual security type, cash offset, amount sign,
  accrued-interest or principal-paydown context, and local mapping or
  REP/report semantics;
- accounting evidence: holdings, accrued interest, cash, quantity or principal
  exposure, `secperf.csv`, and `portperf.csv` all derive from one coherent
  scenario intent;
- YAML semantics: `ai`, `pa`, and `sa` distinguish income from fee/expense or
  unknown treatment from context, not from code alone;
- principal boundary: broad `pd` treatment still proves cash movement,
  principal-paydown context, and principal exposure changes before it is
  treated as performance income, corporate-action evidence, or review-only
  evidence;
- report behavior: workbook rows should make clear whether the transaction is
  a direct formula input, supporting accounting evidence, or a blocked
  classification that needs REP/report context.

The first implementation should be test-only unless the scenario is realistic
enough that formula inputs, holdings/accrued, cash, security performance, and
portfolio performance all derive together.

For any additional examples:

- Make all accounting internally consistent.
- Ensure the report demonstrates fully explained, partly explained, and
  unexplained cases.
- Keep all packaged demo accounting internally consistent.
- Run the packaged demo-data rebuild/audit before accepting fixture changes:

  ```bash
  ./.venv/bin/python scripts/operational_demo_data/rebuild_performance_comparison_demo_data.py
  ```

- Use `--write` on the same command to refresh derived `secperf.csv` and
  `portperf.csv`; the command audits after rebuilding so generated demo data is
  not refreshed without a consistency check.
- Any intentional partly explained or unexplained period must be named in the
  audit allowlist with a reviewer-facing reason.

Demo-data notes:

- Current operational demo generation history lives in
  [`operational_demo_data_notes.md`](operational_demo_data_notes.md).
- Current analytics demo data generation history lives in
  [`../scripts/generic_analytics_demo_data/GENERATION_NOTES.md`](../scripts/generic_analytics_demo_data/GENERATION_NOTES.md).
- Those files are historical/process notes; this file is the active roadmap.

### Phase 9: Evidence-Pack Hardening And Reviewer Readiness

Phase 9 turns the current comparison engine and transaction-semantics guardrails
into a portable evidence pack that a reviewer can inspect, validate, and hand
off without knowing the fixture harness.

This phase should not add a new accounting surface. It should make existing
Modified Dietz formula inputs, supporting evidence, extract-contract context,
and diagnostics easier to find and reproduce.

#### Phase 9A: Bundle Navigation Manifest

Status: complete for the current report-bundle evidence pack.

Report-bundle manifests now include reviewer entrypoints:

- primary review artifact;
- period triage table;
- formula-input cause summary;
- supporting context summary;
- transaction diagnostic artifacts;
- finding-level audit trail.

The manifest also records the comparison YAML path when the bundle writer knows
it. This keeps generated bundles closer to a review evidence pack: the README is
for humans, while `manifest.json` gives automation and handoff tooling a stable
artifact map.

Next hardening steps:

- validate that manifest entrypoints point to declared artifacts;
- include extract-contract summary metadata when a comparison YAML has
  `extract_contract.path`;
- include observed transaction-code and unresolved-semantics summary counts;
- keep entrypoints stable even when optional workbook or reconstruction
  diagnostics are included.

#### Phase 9B: Site Extract Readiness

Status: complete for the current ambiguous-flow onboarding surface.

Site-specific extract contracts should become easier to adopt without weakening
the Axys/APX ambiguous-flow guardrail.

Near-term work:

- improve `validate_config` messages for missing transaction context columns;
- add a concise operator-facing checklist for IMEX context, REP/report
  semantics, and code-only failure modes;
- keep `li`, `lo`, `dp`, and `wd` hard-fail behavior tied to the extract
  contract instead of to transaction code alone;
- prefer REP/report or local-discovery examples when IMEX cannot expose enough
  context.

#### Phase 9C: Test-Only Semantics Expansion

Status: complete for the current ambiguous-flow fixture expansion.

The next scenario expansion should stay test-only unless a case adds a clear
reviewer story to the packaged demo.

Priority examples:

- more `li`/`lo` external and neutral variants;
- additional `dp`/`wd` fee, sweep, transfer, and external-flow variants;
- uppercase correction, cancellation, and reversal-like rows;
- synthetic corporate-action examples that prove expected review-only or
  blocked-classification behavior.

Evidence-blocked families such as `ai`, `pd`, `ss`, `cs`, `rc`, mergers,
spin-offs, and ticker changes should remain backlog until source context
identifies the Modified Dietz formula role or confirms that the row is
review-only evidence. Local `pa`/`sa` variants still need site confirmation,
but the packaged demo now includes one narrow paired fixed-income example.

#### Phase 9D: Manifest Validation And Extract Context Summary

Status: complete for the current manifest contract.

Report-bundle validation now checks that `review_entrypoints` values point to
declared bundle artifacts. This keeps the machine-readable navigation contract
from drifting away from the files in the handoff bundle.

Report-bundle manifests now also include:

- source context with the comparison YAML path;
- extract-contract path, ambiguous-flow enforcement status, and required
  transaction context columns when a comparison YAML is available;
- transaction-semantics summary fields for observed transaction codes, unknown
  category counts, and ambiguous-context blocked counts.

Next hardening steps:

- include missing blocking context columns when validation has that information;
- keep transaction-semantics summary counts aligned with future blocked-run or
  preflight diagnostics;
- expose the same extract-contract summary in `validate_config` output when it
  helps operators fix source files faster.

#### Phase 9E: Bundle Validation Completion

Status: complete for the current manifest schema.

Bundle validation now checks the shape of:

- `source_context`;
- `source_context.extract_contract`;
- `transaction_semantics`;
- `review_entrypoints`, including optional list-valued entrypoints.

This keeps generated evidence packs machine-checkable after handoff, not merely
present at write time.

#### Phase 9F: Extract Context Operator Readiness

Status: complete for the current CLI and checklist surface.

`validate_config` now prints the required transaction context columns from the
resolved extract contract. This gives operators the same context-column signal
that appears in bundle manifests, before they write a report bundle.

Next hardening steps:

- make missing-column error messages group the missing fields by operator task;
- add a local-contract example with ambiguous-flow enforcement disabled only as
  an explicit reviewed opt-out fixture;
- keep the CLI output compact enough for smoke-test and onboarding use.

#### Phase 9G: Shared Transaction Semantics Summary

Status: complete for the current validation and bundle paths.

Transaction semantics summaries now use shared helper logic for bundle manifests
and `validate_config`. The shared shape covers observed transaction codes,
codes without YAML rules, unknown category counts, semantics-source counts, and
ambiguous-context blocked counts.

The bundle manifest uses full transaction files when the comparison YAML path is
available, and falls back to finding rows only when the writer has no source
configuration path.

#### Phase 9H: Operator Checklist Docs

Status: complete for the current site-extract onboarding docs.

[`site_extract_readiness_checklist.md`](site_extract_readiness_checklist.md)
now gives operators a short setup checklist for:

- IMEX extracts with context fields;
- REP/report semantic fallbacks;
- expected code-only failure mode;
- bundle-manifest handoff evidence.

The checklist is linked from the packaged demo source contract and the starter
site extract-contract template.

#### Phase 9I: Test-Only Ambiguous Flow Matrix

Status: complete for the current `li`/`lo`/`dp`/`wd` matrix.

The test-only `imex_context` site variant now explicitly proves the ambiguous
Axys/APX flow matrix for `li`, `lo`, `dp`, and `wd`:

- `li` external contribution and neutral transfer;
- `lo` external withdrawal/deliver-out and neutral transfer;
- `dp` fee-like performance row and neutral transfer/sweep;
- `wd` external withdrawal and neutral transfer/sweep.

The demo matrix validator reports this as `Ambiguous flow context variants` so
coverage is visible from the CLI, not only from unit-test internals.

#### Phase 9J: Code-Only Failure Fixtures

Status: complete for the current code-only failure guard.

The `imex_code_only` fixture remains the explicit failure case for ambiguous
Axys/APX flow codes without context columns. The demo matrix validator reports
this as `Code-only failure guard` and expects the loader to fail before broad
YAML classification can treat `li`, `lo`, `dp`, or `wd` as performance inputs.

#### Phase 9K: Local Opt-Out Boundary

Status: complete for the current reviewed local opt-out fixture.

The `local_opt_out` site variant documents the reviewed local-risk path:

```yaml
extract_contract:
  enforce_ambiguous_axys_flows: false
```

This fixture proves the opt-out is supported without making it the normal
workflow. The demo matrix validator reports it as `Reviewed local opt-out`.

#### Phase 9L: Demo Matrix Reporting Polish

Status: complete for the current demo matrix output.

The demo matrix validator now reports the semantic coverage families created by
this phase train:

- ambiguous flow context variants;
- code-only failure guard;
- reviewed local opt-out.

This keeps the evidence-pack summary readable as the test-only scenario surface
expands.

#### Phase 9M: Evidence-Pack Golden Bundle Fixture

Status: complete for the current compact manifest contract.

Report-bundle tests now pin the compact evidence-pack manifest surface:

- top-level manifest keys;
- `manifest_version`;
- `source_context`;
- `source_context.extract_contract`;
- `transaction_semantics`;
- `review_entrypoints`;
- required CSV artifacts and row-count tables.

This is intentionally a contract test, not a large static golden file.

#### Phase 9N: README / CLI Review Flow Tightening

Status: complete for the current review flow.

The bundle README now tells reviewers that `manifest.json` records the artifact
map, source context, transaction semantics summary, and row-count metadata.
`validate_config` also names the source-context/transaction-semantics handoff
surface before report generation, and `validate_demo_matrix` names the
ambiguous-flow coverage families in its success output.

#### Phase 9O: Bundle Manifest Regression Contract

Status: complete for the current manifest schema.

Bundle validation now checks required top-level manifest keys, manifest version,
source context shape, transaction semantics shape, review entrypoint references,
artifact presence, CSV row counts, and optional workbook structure.

Future manifest extensions should either preserve this schema or intentionally
advance `manifest_version` with matching validation and tests.

#### Phase 9P: Final Phase-9 Consolidation

Status: complete.

Phase 9 is now a release-ready evidence-pack baseline:

- bundle manifests are navigable, validated, and source-aware;
- site extract readiness is documented for IMEX, REP/report fallback, code-only
  failure, and reviewed local opt-out paths;
- ambiguous Axys/APX `li`, `lo`, `dp`, and `wd` semantics have test-only context,
  failure, and opt-out fixtures;
- `validate_config` and `validate_demo_matrix` expose the evidence-pack story
  in CLI output.

Next useful trains should move to a new phase family rather than continuing to
grow Phase 9 indefinitely.

### Phase 10: Fixed-Income Modified Dietz Boundary

Phase 10 keeps fixed-income work attached to Modified Dietz formula inputs
instead of drifting into bond accounting.

#### Phase 10A: Fixed-Income Formula Boundary

Status: complete for the current ordinary-interest, accrued-value, and
accrued-interest adjunct boundary.

The supported fixed-income formula inputs are intentionally narrow:

- ordinary `in` interest transaction amounts that are classified as
  performance income;
- configured `holdings.accrued` changes that feed beginning/end market value
  or security-level performance evidence; and
- paired purchase/sale accrued-interest adjunct amounts when fixed-income
  context and paired trade rows support the `pa`/`sa` treatment.

The following remain outside the return-reconstruction layer:

- amortization/accretion engines;
- bond principal schedule reconstruction;
- yield calculation;
- tax-lot accounting.

Those topics may provide supporting evidence in a source extract or reviewer
report, but they are not prerequisites for calculating Modified Dietz once the
formula inputs are available.

#### Phase 10B: Test-Only Ordinary Interest + Accrued Audit

Status: complete for packaged demo guardrails.

The packaged demo audit now pins the proved fixed-income story:

- `INCOME0603` remains an ordinary `in` transaction on `91282Y2Y1`;
- the row resolves as performance income;
- the packaged snapshots include positive `91282Y2Y1` `holdings.accrued` values;
- paired `pa`/`sa` rows appear only as 91282Y5Y1 accrued-interest adjuncts with
  fixed-income trade context; and
- `ai` does not appear in packaged transaction files; and
- `pd` appears only as an MBS principal-paydown row with portfolio-cash
  destination context.

This proves the current public fixture without adding new synthetic bond
accounting behavior.

#### Phase 10C: Principal Paydown / Accrued-Interest Backlog Contract

Status: complete for the current code-level boundary.

The transaction boundary helper treats:

- `in` as safe ordinary interest;
- `pa`/`sa` as accrued-interest adjuncts when fixed-income context supports
  them;
- `ai` as a backlog code;
- broad/code-only `pd` as a still-gated principal-paydown code;
- all other codes as outside this fixed-income boundary helper.

The backlog codes remain `unknown` by code alone. They need source context,
local mapping, REP/report semantics, and coherent holdings/cash/performance
evidence before ppar can assign a Modified Dietz role.

#### Phase 10D: Fixed-Income Reviewer Reporting

Status: complete for the current report surface.

Reviewer output should continue to explain fixed-income rows in formula terms:

- ordinary interest is a transaction amount classified as performance income;
- `holdings.accrued` is valuation/performance evidence when configured;
- under-evidenced principal-paydown or accrued-interest transaction rows are
  blocked backlog items, not silently inferred income or flows.

Future reporting polish can group these rows more clearly, but it should not
change the formula boundary without new source evidence.

### Phase 11: Test-Only Transaction Semantics Expansion

Phase 11 expands the test-only transaction-semantics surface without adding new
packaged demo accounting stories.

#### Phase 11A: Reversal / Cancellation Boundary

Status: complete for the current review-only fixture.

The `review_only_actions` site variant carries uppercase correction-like rows
as source-reviewed neutral evidence:

- `CXL` stays transfer-neutral;
- `REV` stays transfer-neutral;
- both rows use source semantics, not broad transaction-code inference.

This proves correction/cancellation/reversal-like rows can be loaded for review
without treating an unlinked uppercase row as a new economic event.

#### Phase 11B: Expanded `dp` / `wd` Context Matrix

Status: complete for the current site-variant matrix.

The `imex_context` site variant remains the test-only home for `dp` and `wd`
context expansion:

- fee-like `dp` is performance-impacting only when special-security context
  proves the fee role;
- sweep-like `dp` stays neutral transfer evidence;
- external `wd` requires cash plus external-party context;
- sweep-like `wd` stays neutral transfer evidence.

The demo matrix validator reports these rows as ambiguous-flow context variants.

#### Phase 11C: Synthetic Corporate-Action Quarantine

Status: complete for the current review-only fixture.

The same `review_only_actions` fixture includes a synthetic `;` marker as a
neutral corporate-action row. This is deliberately test-only. It proves ppar can
carry corporate-action evidence without turning it into a Modified Dietz formula
input or claiming packaged split support.

Real-world splits, mergers, spin-offs, ticker changes, and return-of-capital
examples remain backlog until source evidence identifies their formula role or
confirms they are review-only evidence.

#### Phase 11D: Demo Matrix + Roadmap Reporting

Status: complete for the current validator/reporting surface.

The demo matrix validator now reports review-only action quarantine alongside
ambiguous-flow context variants, the code-only failure guard, and reviewed
local opt-out coverage. This keeps test-only semantic expansion visible without
changing the packaged demo promise.

### Phase 12: Return-of-Capital And Short-Side Backlog Gates

Phase 12 sharpens high-risk backlog boundaries without implementing return of
capital, principal-paydown, short-sale, or cover-short accounting.

#### Phase 12A: Return-of-Capital Policy Boundary

Status: complete for the current backlog contract.

`rc` remains a capital-return policy gate. A site must provide security
identity, amount sign, cost-basis or report treatment, and local mapping or
REP/report semantics before ppar can choose among:

- performance income;
- corporate-action evidence;
- review-only evidence.

Code-only `rc` rows stay `unknown`; they are not silently treated as income,
external flows, or corporate actions.

#### Phase 12B: Principal / Capital Return Vocabulary Alignment

Status: complete for the current matrix and helper vocabulary.

`pd` remains aligned with the capital-return gate even though the packaged demo
now includes one context-gated MBS principal-paydown example. Broad
principal paydown still needs bond, MBS, or ABS security type,
principal-paydown context, cash movement, principal exposure or quantity
change, and local mapping or REP/report semantics before it can be classified
from site data.

This keeps "return of capital," "principal paydown," and "corporate action" as
separate reviewer decisions rather than interchangeable labels.

#### Phase 12C: Short Sale / Cover Short Evidence Gate

Status: complete for the current backlog contract.

`ss` and `cs` remain short-side evidence gates. Public APX integration evidence
now strongly supports `ss` as short sale and `cs` as cover short. The packaged
demo now includes one disclosed same-period TSLA short sale / cover short
lifecycle using real May 2026 prices, separate proceeds and cover cash movement,
no external-flow treatment, and realized loss on cover. Production rows still
need short security type or resulting negative exposure, cash, margin, or
short-account symbols, source/destination context, amount and quantity signs,
and local mapping or REP/report semantics before ppar can classify them.

Code-only short-side rows stay `unknown`; the project does not infer universal
short-account, short-proceeds, or holdings treatment from the Axys/APX code
alone.

#### Phase 12D: Matrix + Validator Reporting

Status: complete for the current validator surface.

The demo matrix validator now reports `Capital-return and short-side backlog
gates`. The check enforces that `rc` and `pd` are packaged only with explicit
context, while `ss` and `cs` are packaged only in a disclosed synthetic
same-period lifecycle and remain gated for code-only production treatment.

### Phase 13: Matrix Consolidation And Release Readiness

Phase 13 consolidates the transaction-boundary story created by the previous
phase trains.

#### Phase 13A: Transaction Boundary Registry

Status: complete for the current registry surface.

`ppar.performance_comparison.transaction_boundary_registry` now groups
transaction codes into reviewer-facing boundary families:

- packaged formula rows;
- fixed-income safe rows;
- ambiguous context-required rows;
- review-only test rows;
- context-only rows;
- fixed-income backlog;
- capital-return backlog;
- short-side backlog;
- standalone backlog.

The registry is intentionally descriptive. It does not classify transactions at
runtime or replace the transaction semantics matrix.

#### Phase 13B: Demo Matrix Validator Cleanup

Status: complete for the current validator organization.

`validate_demo_matrix` groups scenario checks by review purpose: baseline and
attribution checks, site-variant checks, review-only quarantine checks, and
backlog-gate checks. This keeps the validator readable as it becomes a coverage
contract rather than only a demo smoke test.

#### Phase 13C: Roadmap / Matrix Consistency Audit

Status: complete for the current docs and matrix tests.

Tests now cross-check the boundary registry against
`contracts/transaction_semantics_matrix.yaml`, the roadmap phase claims, and the
site-variant fixtures. The goal is to prevent docs from claiming coverage that
the matrix or fixtures do not support.

#### Phase 13D: Pre-Commit Release Snapshot

Status: complete for the current release-readiness note.

[Boundary snapshot](performance_comparison_transaction_boundary_snapshot.md)
summarizes the current covered, context-required, review-only, context-only,
and backlog-gated transaction families. It is a reviewer aid; the YAML matrix
remains the implementation contract.

### Phase 14: Final Review Pack And Commit Preparation

Phase 14 turns the accumulated evidence-pack work into a commit-ready review
surface.

#### Phase 14A: Change Inventory

Status: complete for the current review pack.

[Evidence-pack review](performance_comparison_evidence_pack_review.md)
summarizes the work by theme: evidence-pack manifests, site extract readiness,
transaction boundaries, test-only fixtures, validator coverage, and reviewer
docs.

#### Phase 14B: Public API / Package Surface Check

Status: complete for the current helper modules.

The new boundary helpers are intentionally direct submodule imports, not
top-level `ppar.performance_comparison` exports. This keeps the public workflow
API focused while still making the review helpers importable for validators and
tests.

#### Phase 14C: Diff Hygiene Pass

Status: complete for the current roadmap and docs checks.

The roadmap has one Phase 9-14 section each, followed by the guiding principle
and transaction backlog. Docs tests pin the review-pack links and public surface
expectations so stale or duplicated release-prep language is easier to catch.

#### Phase 14D: Commit-Ready Validation

Status: complete for the current validation pass.

Commit preparation should include the demo matrix validator and the full test
suite. Do not commit generated review work until both pass.

### Phase 15: Reviewer Handoff Summary

Phase 15 keeps the review pack focused on Modified Dietz evidence while making
bundle handoff easier for a second reviewer or automation.

#### Phase 15A: Compact Bundle Summary Artifact

Status: complete for the current report-bundle contract.

Generated report bundles now include `review_summary.json`. The file repeats the
small subset of manifest data a reviewer needs first:

- Modified Dietz review vocabulary;
- review entrypoints;
- source context;
- finding counts;
- transaction-semantics summary;
- artifact map.

The summary intentionally distinguishes formula inputs from supporting evidence,
context-only rows, review-only rows, and backlog gates. This keeps reinvested
dividends, transaction diagnostics, and other evidence families from being
treated as automatic return inputs unless policy and source evidence justify
that treatment.

#### Phase 15B: Handoff Validation

Status: complete for the current summary schema.

Bundle validation now checks that `review_summary.json` is present, has the
expected summary schema, and matches the manifest fields it repeats. This makes
the compact handoff artifact useful without creating a second source of truth.

### Phase 16: Bundle Contract Hardening

Phase 16 tightens the report-bundle handoff contract without adding transaction
classification scope.

#### Phase 16A: Required Entrypoint Drift Guards

Status: complete for the current report-bundle contract.

Bundle validation now requires the standard review-entrypoint names:

- primary review;
- period triage;
- formula-input causes;
- supporting context;
- transaction diagnostics;
- audit trail;
- review handoff.

This keeps reviewer and automation entrypoints stable even when future bundle
artifacts are added.

#### Phase 16B: Summary Schema Drift Guards

Status: complete for the current compact summary schema.

Report tests now pin the top-level `review_summary.json` keys, and validation
rejects missing summary keys. The summary remains a compact mirror of selected
manifest fields, with one internal Modified Dietz review-basis constant and one
internal vocabulary contract.

### Phase 17: Centralized Bundle Contract

Phase 17 closes the report-bundle contract loop by making the handoff shape
available from one helper.

#### Phase 17A: Public Contract Helper

Status: complete for the current report-bundle contract.

`report_bundle_contract()` returns the stable generated-bundle handoff contract:

- portfolio and security audit filenames;
- required artifact keys;
- manifest version and required manifest keys;
- required review entrypoints;
- review-summary version and required summary keys;
- Modified Dietz review basis and vocabulary keys.

The helper keeps validation, tests, docs, and reviewer automation aligned
without broadening transaction classification behavior.

#### Phase 17B: Contract Snapshot And Handoff Consistency

Status: complete for the current generated-bundle contract.

Report tests now pin the public contract helper and check that
`README.md`, `manifest.json`, and `review_summary.json` agree on reviewer
first-stop artifacts. This keeps the bundle readable for humans while preserving
one machine-readable contract for automation.

### Phase 18: Public Contract Discoverability

Phase 18 makes the generated-bundle contract easier to find without expanding
the report-bundle behavior.

#### Phase 18A: Design / Repository Guide Touchpoint

Status: complete for the current public contract helper.

The design reference and repository guide now name `report_bundle_contract()` as
the public helper for inspecting the generated report-bundle handoff surface.
The docs describe it as a contract helper for required artifacts, manifest keys,
review entrypoints, review-summary keys, Modified Dietz review basis, and
vocabulary keys.

#### Phase 18B: Discoverability Guard

Status: complete for the current docs surface.

Package metadata tests now keep the design and repository-guide references in
place. This keeps the helper discoverable without turning it into a transaction
classification or accounting layer.

### Phase 19: Generated Demo Bundle Smoke

Phase 19 verifies the generated reviewer surface after the bundle-contract and
wording changes.

#### Phase 19A: Portfolio / Security Bundle Regeneration

Status: complete for the current generated demo bundles.

The packaged portfolio and security comparison demos were regenerated under
`_demo_output/performance_comparison_portfolio` and
`_demo_output/performance_comparison_security`. The generated output remains
ignored by Git; the smoke is a reviewer-surface check, not a source artifact to
commit.

#### Phase 19B: Generated Bundle Validation

Status: complete for the current generated demo bundles.

Both generated bundles passed `ppar.performance_comparison.cli.validate_bundle`.
The spot check confirmed:

- `README.md` starts with the recommended review order;
- `manifest.json` and `review_summary.json` agree on review entrypoints;
- transaction component wording is security-prefixed;
- price and quantity component explanations do not repeat raw component deltas;
- commission explanations retain the commission delta.

### Phase 20: Reviewer Wording Guards

Phase 20 pins the reviewer-facing wording that Phase 19 verified.

#### Phase 20A: README Wording Contract

Status: complete for the current generated-bundle README.

Report and workbook contract tests now reject the removed README phrases:

- `## Primary Review Artifact`;
- `Open report.xlsx first`;
- `same review model in a browser`.

#### Phase 20B: Transaction Explanation Contract

Status: complete for the current workbook/report wording.

Report and workbook contract tests now pin transaction component wording:

- price and quantity rows say the security's `transactions.amount` increased or
  decreased, without repeating the raw price or quantity delta in prose;
- commission rows include the security and retain the commission delta;
- the old `Helped explain the changed transactions.amount` wording is rejected.

### Phase 21: Native-Code And Identifier Guardrails

Phase 21 keeps reviewer metadata faithful to source extracts without weakening
case-sensitive identifiers.

#### Phase 21A: Native Transaction-Code Handoff

Status: complete for the current review summary and manifest contract.

`manifest.json` and `review_summary.json` now preserve native observed
transaction-code strings in `observed_codes`. YAML coverage still normalizes
transaction codes for rule-key lookup, but the review handoff no longer converts
native lower-case demo codes such as `by` and `sl` to upper case. This
normalization is scoped to semantic classification and rule coverage; it is not
a broad case-insensitive equality policy for source identifiers.

#### Phase 21B: Source Dataset Report Label

Status: complete for the current workbook/report wording.

The visible Source Detail dataset column now uses `Source Dataset`, while
prose continues to use the standardized `source-data` term and JSON/API keys
continue to use `source_data`.

#### Phase 21C: Case-Sensitive Identifier Audit

Status: complete for the current comparison and workbook paths.

Regression coverage now pins security identifiers as exact comparison keys:
`AAPL` and `aapl` remain separate securities and produce add/drop findings, not
an in-place value change. Transaction-code normalization remains limited to
explicit semantic rule/category lookups; reviewer metadata retains native code
case.

### Phase 22: Transaction Matching Diagnostics Depth

Status: complete for first-pass transaction match outcome vocabulary.

Transaction matching diagnostics now distinguish stable ID matches, snapshot-B
adds, snapshot-B missing rows, and ambiguous strict-fallback keys. Stable
`transaction_id` matches remain the only high-confidence edit pairing. Strict
fallback keys still prevent amount restatements from being inferred as edits,
and duplicate fallback keys are reported as ambiguity diagnostics because two
identical same-day transactions can be legitimate. This improves auditability
of the existing Modified Dietz evidence path without adding tax-lot,
cost-basis, or accounting-ledger reconstruction.

### Phase 23: No-ID Transaction Timing Evidence

Status: complete for first-pass cross-period timing diagnostics.

Transaction presence findings now use the full transaction source row rather
than only the comparison key. When no stable `transaction_id` exists and a
transaction date changes, including a move from one performance period to the
next, the comparison reports an old-date `missing_from_snapshot_b` row and a
new-date `added_in_snapshot_b` row. These rows preserve portfolio/security/date
context and can carry review-only Modified Dietz cross-check estimates when
explicit external-flow semantics and policy inputs are available. The behavior
keeps the Modified Dietz timing effect visible without inferring that the two
no-ID rows are the same transaction.

### Phase 24: Conservative Singleton Fallback Matching

Status: complete for exact one-to-one no-ID transaction matching.

No-ID transaction rows may now be paired only when there is exactly one row in
Snapshot A and one row in Snapshot B for the same portfolio, trade date,
security identifier, and native transaction code. The match is case-sensitive
and does not use amount, quantity, price, nearest-date, or fuzzy matching.
Changed fields from this path are labeled `matched_by_singleton_fallback`,
which is intentionally weaker than `matched_by_id`. Duplicate candidate groups
remain ambiguity diagnostics, and date moves across periods remain add/drop
timing evidence.

### Phase 25: Transaction Match Review Language

Status: complete for reviewer-facing match confidence diagnostics.

Transaction matching diagnostics now include explicit confidence and
interpretation columns. The report distinguishes strong transaction-ID matches,
conservative exact singleton fallback matches, unpaired add/drop rows, and
withheld ambiguous fallback groups without adding fuzzy matching or changing
Modified Dietz impact calculations. The extra labels are presentation guidance
only; transaction identity remains conservative and case-sensitive.

### Phase 26: Transaction Match Diagnostics Report Visibility

Status: superseded by Phase 28.

This phase previously promoted `Transaction Match Diagnostics` to a shared
workbook/report section backed by the existing
`transaction_matching_diagnostics.csv` table. It made match status counts,
confidence, interpretation, and review notes visible in `report.xlsx` and
`report.html` without changing transaction matching behavior or Modified Dietz
calculations. Phase 28 later demoted that standalone section back to audit
support.

### Phase 27: No-ID Packaged Axys/APX Demo

Status: complete for packaged demo source realism.

The user-facing Axys/APX demo transaction CSVs now omit `TRANSACTION_ID` because
the local Axys/APX research corpus does not prove a durable native transaction
identifier as typical REP/IMEX output. Stable transaction IDs remain supported
and test-covered when a local extract provides them, but the packaged demo now
exercises the conservative no-ID matching path by default. Source/destination
and special-security context remains in the packaged transaction rows where it
is needed to classify ambiguous Axys/APX-style codes safely.

### Phase 28: Demote Transaction Match Diagnostics

Status: complete for reviewer-facing report focus.

The standalone `Transaction Match Diagnostics` section has been removed from
the normal `report.xlsx` and `report.html` review flow. Transaction match
status remains visible on raw audit rows, and the audit-support
`transaction_matching_diagnostics.csv` artifact remains in the report bundle
and manifest. This keeps row-identity evidence available without making an
internal matching diagnostic a first-stop reviewer section.

### Phase 29: Report Focus And Audit Placement Polish

Status: complete for generated bundle handoff wording.

The generated bundle README now makes the review/audit split explicit: start in
`report.xlsx` when present, use `report.html` for browser review, use
`Performance Difference Causes` for additive source-data explanations, and use
`Source Detail` for audit and troubleshooting. Supplementary transaction and
external-flow diagnostics remain CSV artifacts, while
`transaction_matching_diagnostics.csv` is positioned specifically as
transaction row-identity audit support rather than a main review stop.

### Phase 30: Demo Realism And No-ID Matching Audit

Status: complete for packaged demo field-boundary guardrails.

The packaged Axys/APX demo now has an explicit field-boundary taxonomy separating
mandatory product inputs, realistic packaged-demo fields, optional
local-enrichment fields, and internal scenario/rebuild fields. Regression
coverage pins the user-facing transaction header, confirms packaged
transactions omit `TRANSACTION_ID`, confirms rule-disambiguation context fields
remain packaged, and allows deterministic `TRANSACTION_ID` only in internal
scenario tooling. Existing no-ID matching coverage already proves exact
singleton fallback behavior, duplicate-candidate quarantine, case-sensitive
transaction/security keys, and date-move add/drop treatment with Modified Dietz
timing estimates.

### Phase 31: Final Demo Contract Pass

Status: complete for review vocabulary and metadata guardrails.

The demo contract now explicitly says the packaged comparison demo is
formula-focused and is not a full accounting-system export. Contract tests pin
cost and settlement fields as review evidence unless an explicit future rule
changes that treatment, preserve the conservative no-ID transaction path, and
verify report-bundle metadata keeps standardized review vocabulary and
entrypoints aligned after the transaction-match diagnostics demotion.

### Phase 32: Release Readiness Sweep

Status: complete for public handoff consistency.

The public README and packaged Axys/APX README now use the same handoff language as
generated bundles: open `report.xlsx` when present, use `report.html` for
browser review, and keep CSV artifacts for diagnostics and audit traceability.
Regression coverage pins that wording so the older "same review model in a
browser" phrasing does not drift back into public demo guidance.

### Phase 33: Transaction Evidence Realism And Conservative Matching Audit

Status: complete for reviewer-facing transaction matching boundaries.

Transaction matching diagnostic notes now name the confidence boundary directly:
stable `transaction_id` matches are strongest, exact singleton fallback matches
require one row in each snapshot for the same portfolio, trade date, security
identifier, and native transaction code, and unmatched strict-fallback rows are
not inferred from fuzzy, nearest-date, amount, quantity, or price similarity.
Generated bundle README guidance also frames
`transaction_matching_diagnostics.csv` as conservative row-identity evidence
rather than a user-facing transaction-linkage claim.

### Phase 34: Demo Source-Contract Final Hardening

Status: complete for packaged Axys/APX field defensibility.

The demo source contract now explicitly treats `portperf.gain_loss` and
`secperf.gain_loss` as report-style performance-extract context, not native
Axys/APX IMEX object claims or recomputed accounting-ledger values. Audit
coverage pins that boundary in `demo_extract_availability.yaml`, confirms
packaged transaction extracts still omit `TRANSACTION_ID`, and keeps
source/destination and special-security context fields required for ambiguous
Axys/APX-style transaction classification.

### Phase 35: Final Release Candidate Audit

Status: complete for generated handoff wording consistency.

The release-candidate audit regenerated both performance-comparison demo
bundles and checked report HTML, README, manifest, review summary, transaction
matching diagnostics, transaction activity, and flow reconciliation artifacts.
The HTML report header now describes `report.html` as the browser review
surface for the bundle instead of reusing the older "same review model"
phrasing. The visible `Source Dataset` report label remains unchanged, while
its tooltip now uses standardized `source-data` prose.

### Phase 36: Reader-Facing Language Final Polish

Status: complete for report-header plain language.

The generated HTML report header now says "Browser view for reviewing this
performance-comparison bundle." This keeps the browser/report.xlsx distinction
clear without exposing the more internal "review surface" wording to first-time
reviewers. Tests continue to reject both the old "same review model" phrase and
the intermediate "Browser review surface" phrase.

### Phase 37: Roadmap Readability Refactor

Status: complete for active-backlog discoverability.

The roadmap now starts with `How To Read This Roadmap`, `Current Status`, and
`Current Open Items` sections so maintainers can see what remains before
reading the implementation journal. The open work is summarized as backlog
expansion, with packaged-demo candidates, test-only fixtures, evidence-blocked
transaction families, and future policy expansion separated from completed
release-candidate cleanup.

### Phase 38: Packaging Surface And Product Boundary Audit

Status: complete for package/product boundary guardrails.

The packaged Axys/APX demo boundary is now test-covered as a wheel/package-data
surface: demo resources under `ppar/setup_templates/axysapx_performance_comparison` stay limited to CSV, YAML,
and Markdown product inputs/notes, and package-data globs do not expose
generated report output, source-checkout scripts, test fixtures, or
demo-generation internals. Source distributions may still include maintainer
scripts, but the repository guide and packaged Axys/APX README now label those as
source-checkout maintenance workflows rather than installed-package demo
entrypoints.

### Phase 39: Minimum Source-Data Contract And Hard-Stop Validation

Status: complete for core source-data contract guardrails.

The minimum source-data contract is now documented in the demo source contract
and backed by `ppar.performance_comparison.source_data_contract`. The contract
names the required normalized columns for portfolio performance, security
performance, holdings, transactions, cash, and FX-rate
datasets, plus the workflow condition that makes each dataset mandatory.
`validate_config` now prints the resolved minimum required datasets and required
source-data columns for the selected comparison. Return reconstruction also
hard-stops earlier: when portfolio or security return reconstruction is
configured, `holdings` and `transactions` are required source files in both
snapshots and cannot be marked optional.

### Phase 40: YAML Strictness And Misleading-Report Prevention

Status: complete for config-validation report-readiness defaults.

`validate_config` now uses the same complete-YAML setup guardrail as normal
report-bundle generation. A comparison YAML that would create a misleading
default report because changed source-data fields lack additive, evidence-only,
or suppression YAML now fails config validation before report generation. The
diagnostic escape hatch is explicit and shared with bundle generation:
`--allow-incomplete-yaml` on the CLI or
`require_complete_yaml_setup=False` in the API. The packaged demo remains
strict-valid; intentionally incomplete validation fixtures opt into diagnostic
validation when they need to inspect malformed or partial setup.

### Phase 41: Artifact Taxonomy And Polars Recon

Status: complete for report-bundle artifact classification and first-pass
execution audit.

The generated bundle artifacts are now grouped into first-stop review surfaces,
reviewer handoff metadata, audit/export backbone tables, supplementary
diagnostics, and opt-in reconstruction diagnostics. The audit did not identify
a safe obsolete CSV to remove: each required artifact is still produced,
manifested, validated, or used as a reviewer/automation handoff. CSV diagnostics
therefore remain in the bundle, while `report.xlsx` and `report.html` stay the
first-stop review surfaces.

The first-pass execution audit found no Pandas import or `to_pandas` conversion
inside `ppar.performance_comparison`. Source loading, joins, grouping, duplicate
checks, and bundle validation are Polars-based. Remaining row iteration is
mostly presentation assembly, reviewer guidance, transaction diagnostics, and
edge-case comparison output where row-shaped prose or audit records are being
created. Quick demo timing in the local environment put portfolio bundle
generation at about four seconds and security bundle generation at about nine
seconds. A `cProfile` pass on the slower security demo showed the largest
cumulative cost in repeated security return reconstruction checks during
workbook/report sheet assembly, especially repeated Polars `filter().collect()`
calls inside reconstruction input lookup. Future speed work should benchmark
and cache or batch those reconstruction lookups before attempting broader
vectorized rewrites.

### Phase 42: Reconstruction Diagnostics Cache

Status: complete for the first measured speed bottleneck.

Report and workbook generation now share a per-build reconstruction diagnostics
cache. The cache is scoped to one comparison path and one report/workbook build,
so long-running processes do not reuse stale source files across separate
builds. Direct helper calls still create a one-off cache, preserving the
existing test/debug ergonomics.

The targeted local timing improved materially:

- security demo generation changed from about 9.2 seconds to about 3.6 seconds;
- portfolio demo generation changed from about 4.0 seconds to about 2.9 seconds.

The generated portfolio and security bundles still validate, and the generated
`report.html` content remained unchanged. Broader speed work remains a future
benchmarking task, but the known repeated security reconstruction bottleneck is
addressed.

### Phase 43: Cleanup And Artifact Retention Sweep

Status: complete for a first conservative dead-code and artifact-boundary pass.

The cleanup pass removed a stale workbook helper for the standalone
`Transaction Match Diagnostics` worksheet, which Phase 28 had already demoted
from the normal workbook/report flow. Transaction match status labels and
tooltips remain because those fields still appear on Source Detail rows and
in the audit-support CSV.

The artifact review did not identify a safe CSV deletion. Required bundle CSVs
remain manifested, validated, documented, or used as reviewer/automation
handoff artifacts. In particular, transaction and flow diagnostics are still
supplementary CSVs, and `transaction_matching_diagnostics.csv` remains
transaction row-identity audit support rather than a first-stop review surface.

Broader simplification remains open. Future cleanup should target high-payoff
report/workbook helper simplifications and obsolete compatibility shims only
when tests prove the public report, manifest, and packaged-demo boundaries are
unchanged.

### Phase 44: Public API And Compatibility Shim Audit

Status: complete for a first public-surface boundary audit.

The package-root `ppar.performance_comparison` API remains focused on workflow
entrypoints, source-data loaders, core finding objects, and report-bundle
handoff helpers. The root export list is now grouped by intent in code so
maintainers can see which names are workflow surface, report handoff, and
compatibility explanation helpers.

No public root exports were removed in this pass. Existing tests and demos still
import several explanation and diagnostic helpers from the package root, so
removing them would be an API-breaking cleanup rather than a safe dead-code
change. Boundary and evidence-pack helpers such as fixed-income rules, backlog
gates, source-data contracts, transaction-boundary registry data, and
transaction summaries remain direct-submodule imports and are intentionally not
added to the package root.

The repository guide now tells Python integrations to prefer package-root
workflow helpers for normal comparison/report use, while treating specialized
policy and evidence-pack helpers as direct-submodule imports.

### Phase 45: Documentation Freshness And Reader Path Sweep

Status: complete for current durable documentation and packaged-demo README
wording.

The reader-path sweep removed stale durable-doc references to the former
`Other Data Differences` sheet and aligned the normal workbook/report flow to
the current default review path: `Performance Differences`, `Performance
Difference Causes`, and `Source Detail`. Supplementary CSV diagnostics remain
documented as audit and troubleshooting artifacts rather than first-stop review
surfaces.

The packaged Axys/APX README and repository guide now use the canonical opt-in
reconstruction diagnostic worksheet names: `Reconstruction Summary`, `Return
Reconstruction Checks`, and `Security Return Checks`. The corresponding CSV
artifact names remain `reconstruction_summary.csv`,
`return_reconstruction_checks.csv`, and
`security_return_reconstruction_checks.csv`.

Metadata tests now reject the stale `Other Data Differences`, `Residual
Evidence`, and `Return Reconstruction Summary` phrases in the durable reader
path docs where those names would mislead new users.

### Phase 46: Optional Reconstruction Diagnostics UX Audit

Status: complete for opt-in reconstruction diagnostics ordering and help text.

An opt-in bundle generated with `--include-reconstruction-diagnostics` and
`--include-workbook` now keeps the default reviewer path first in both
`report.xlsx` and `report.html`: `Performance Differences`, `Performance
Difference Causes`, and `Source Detail`. Optional diagnostics follow that
normal path as `Reconstruction Summary`, `Return Reconstruction Checks`, and
`Security Return Checks`.

The CLI help for `--include-reconstruction-diagnostics` now names those
optional worksheet/report sections and states that matching CSV artifacts are
included. Manifest review entrypoints continue to expose the CSV artifact names:
`reconstruction_summary.csv`, `return_reconstruction_checks.csv`, and
`security_return_reconstruction_checks.csv`.

The temporary opt-in bundle validated successfully, and tests now pin the
workbook sheet order, HTML section order, manifest reconstruction entrypoints,
and CLI help wording.

### Phase 47: Roadmap Pruning And Release Backlog Reconciliation

Status: complete for current open-item reconciliation after Phases 41-46.

The top-level roadmap now treats artifact taxonomy, the first measured speed
bottleneck, API-boundary clarification, reader-path freshness, and optional
reconstruction diagnostics UX as completed release guardrails rather than active
blockers. The active near-term section is now a release-hardening watchlist:
small behavior-preserving code simplifications, future profiling only when new
evidence justifies it, and documentation freshness guardrails.

Transaction coverage, policy expansion, the richer APX demo, multi-currency
modeling, broader extract discovery, and commercial licensing remain visible as
the real forward backlog.

### Phase 48: Packaged Demo Transaction Coverage Triage

Status: complete for packaged-demo promotion triage; later superseded in part
by packaged `lo` and `pa`/`sa` promotions.

The packaged Axys/APX transaction set was reviewed against
`contracts/transaction_semantics_matrix.yaml`, the site-variant fixtures, and the current
transaction backlog. No new transaction row was promoted into the packaged demo
in this pass; later phases promoted `lo` and a narrow paired `pa`/`sa` story
after stronger scenario support existed.

That is intentional. The packaged demo already includes a contextual external
cash `li`, an external cash `wd`, a fee-like `dp`, ordinary `by` / `sl`
trading rows, dividend income, and fixed-income income. Site-variant fixtures
already cover the broader ambiguous-code behavior for `li`, `lo`, `dp`, and
`wd`, including external-party and internal-transfer distinctions. At this
point, a packaged `lo` row was useful only if it added a distinct reviewer
story; a later phase added one contextual deliver-out example with that
standard.

The active backlog now names that standard directly: promote additional
transaction families to packaged data only when they are realistic,
nonredundant, internally consistent across transactions, holdings, performance
rows, YAML rules, report wording, and source-contract language. Synthetic or
surgical coverage remains test-only.

### Phase 49: Portfolio/Security Explanation Consistency Audit

Status: complete for current portfolio-vs-security explanation guardrails.

The portfolio and security `Performance Difference Causes` sheets were reviewed
for the transaction wording split that appears after semantic transaction labels
were added. The difference is intentional: portfolio reports explain
transaction rows by their portfolio-return role, while security reports explain
transaction rows by the affected security return container.

Portfolio report examples stay portfolio-input oriented: `dp`, `dv`, and `in`
transaction rows may say they caused cash-balance ending
`holdings.market_value` to move, and portfolio `wd` rows may show external-flow
and weighted external-flow wording. Security report examples stay
security-container oriented: `li` / `wd` rows use `external flow`, `dp` uses
`fee/expense`, and `dv` / `in` use `income`.

The design doc and workbook contract tests now pin this split so future wording
cleanup does not flatten the two report families into one misleading sentence
template.

### Phase 50: Release-Candidate Drift Sweep

Status: complete for current terminology drift and report-output stability.

The release-candidate drift sweep checked generated bundles, durable docs, and
reviewer-facing tests for stale report names, old browser-review wording,
generic transaction amount explanations, and split source-data terminology.
Only historical roadmap notes and negative regression assertions retained the
old phrases. A small prose-only cleanup standardized remaining package comments
and docstrings on `source-data` wording without changing report output.

### Phase 51: Minimum Source Contract Release Pass

Status: complete for pre-report source-contract guidance and hard-stop clarity.

The packaged Axys/APX README now tells users to run `validate_config` before local
report-bundle generation. Required-file hard stops now name the normalized
dataset and snapshot side that failed, so missing required inputs point to
`files.<dataset>` and `snapshot a` / `snapshot b` instead of only a raw path.
Regression coverage pins both the packaged README guidance and the clearer
missing-file diagnostics.

### Phase 52: Packaged Demo Distribution Audit

Status: complete for installed-package demo-resource access.

The wheel and source distribution were built and inspected. The wheel includes
the packaged Axys/APX CSV/YAML/README inputs and excludes `_demo_output`, tests,
docs, scripts, and generation internals. The source distribution retains
maintainer scripts as intended. A regression test now pins the public demo
entrypoints to `importlib.resources` so installed demos continue reading
bundled resources rather than checkout-relative paths or test data.

### Phase 53: Release Readiness Final Sanity Sweep

Status: complete for current release-candidate sanity checks.

The public smoke path passed: `validate_config`, both packaged demo commands,
and both generated-bundle validators. Generated portfolio and security
`report.html` files were diffed against the phase baseline with no content
changes. A stale-vocabulary sweep found only historical roadmap references and
negative tests that intentionally reject old wording. Build residue from the
distribution audit was removed from the checkout.

### Phase 54: Axys/APX Blocker Alignment

Status: complete for current blocker navigation.

The Axys/APX reference now has a single `Axys/APX blockers` section in
`docs/axys-apx-reference/reference/Chapter_01_Overview.md`. The roadmap points readers to
that section as the canonical summary of evidence gaps that block broader
Axys/APX-native automation, including native performance extract dictionaries,
stored-versus-recalculated performance, security-performance footing,
transaction-code coverage, IMEX object authority, REP/report definitions, and
richer multi-currency, fixed-income, and corporate-action behavior.

Metadata tests now keep that blocker section and the roadmap link discoverable.

### Phase 55: Release Candidate Reality Check

Status: complete for current release-candidate verification.

The full test suite passed with 605 tests. The package build succeeded for both
wheel and source distribution. The wheel and source distribution were inspected
for packaged Axys/APX demo resources; required CSV/YAML/README inputs were present,
and `_demo_output` plus tests were absent from the source distribution.

Portfolio and security `report.html` hashes were captured before and after the
verification pass. They did not change, so this phase did not alter generated
report content. Build residue was removed after inspection.

### Phase 56: Vendor Preset Design Spike

Status: complete for design-only vendor preset framing.

The design notes now define vendor presets as a future convenience layer that
can support multiple vendors, with Axys/APX as the first likely preset. A future
keyword such as `vendor: axys` would expand to versioned preset semantics, then
allow deterministic site YAML overrides. Preset resolution is documented as
`engine defaults < vendor preset < site YAML overrides`.

The design intentionally blocks implementation until the relevant packaged demo
is stable enough to become a preset seed. Presets must remain inspectable,
versioned, source-contract tied, and unable to bypass complete-YAML validation
or ambiguous transaction-code safeguards.

### Phase 57: Axys/APX Demo Finalization Gate

Status: complete for defining the preset prerequisite gate.

The demo source contract now defines an Axys/APX demo completion gate. This gate
states when the packaged Axys/APX CSV/YAML/report story is stable enough to seed a
future `vendor: axys` preset. The gate covers realistic packaged fields, no
leaked internal rebuild identifiers, stable YAML semantics, complete-YAML
treatment, intentional Fully/Partly/Unexplained report examples, ambiguous-code
context safeguards, package-resource validation, and report-content change
discipline.

Vendor preset implementation remains blocked until that gate passes. The gate
does not change runtime behavior; it defines the standard for deciding when the
packaged Axys/APX demo can become reusable default policy.

### Phase 58: Axys/APX Demo YAML Seed Readiness Audit

Status: complete for the current YAML seed-readiness pass.

The packaged Axys/APX comparison YAML was reviewed as a future preset seed. The
audit did not identify a data or runtime-rule change. The YAML now states that
it is a candidate seed for a future `vendor: axys` preset, but not a hidden
preset today. Comments also call out the portfolio/security Modified Dietz
boundaries and separate current packaged transaction rows from reserved
guardrail semantics for `lo` and `;`.

The phase keeps vendor-preset implementation blocked. It only improves the
readability and auditability of the existing explicit YAML policy.

### Phase 59: Axys/APX Demo Completion Gate Audit

Status: complete for the current gate audit. Gate status: near-pass.

The Axys/APX demo completion gate was checked against current packaged files,
generated review bundles, validation commands, and focused tests. The current
state satisfies the auditable mechanical checks: packaged transaction CSVs do
not expose fixture transaction identifiers, the explicit YAML validates with
ambiguous-flow context safeguards, generated bundles validate, demo matrix
coverage passes, and focused package/demo health tests pass.

The gate is not declared fully passed until the Axys/APX demo is intentionally
frozen as a preset seed. Until then, `vendor: axys` remains design-only even
though the current explicit YAML is close enough to serve as the candidate seed.

The health check regenerated both report bundles during this audit. The
portfolio and security `report.html` hashes were unchanged, so this phase did
not change user-facing report content.

### Phase 60: Axys/APX Demo Freeze Readiness Sweep

Status: complete for current freeze-readiness framing.

The packaged Axys/APX README, comparison YAML, source contract, roadmap, and
generated report-story boundary were reviewed as a potential future
`vendor: axys` preset seed. No runtime rule or packaged data change was needed:
the current material already separates packaged rows from guardrail-only YAML
rules, keeps ambiguous Axys/APX-style transaction codes context-gated, preserves
the fee/expense return-basis caveat, and keeps the richer APX and broader
transaction backlog outside the packaged Axys/APX demo.

The active backlog now names the remaining freeze decision directly. The
mechanical checks are green, but the remaining step is a maintainer/product
decision: explicitly accept the current packaged CSV/YAML/report story as the
versioned Axys/APX preset seed. Until that decision is made, the correct state
remains near-pass, not preset implementation.

This phase did not require report regeneration or report-story edits. Portfolio
and security `report.html` hashes were checked and remained unchanged.

### Phase 61: Freeze Decision Prep And Release Backlog Slimming

Status: complete for freeze-decision prep.

The demo source contract now has an Axys/APX Demo Freeze Decision Packet. It turns
the freeze choice into a concise acceptance checklist covering packaged
transaction families, then-current guardrail-only `lo` and `;` rules,
context-gated ambiguous codes, conservative no-ID matching, the net-of-fees
`dp` assumption, intentional Fully/Partly/Unexplained report examples,
review-evidence fields, and the boundary that `vendor: axys` means ppar's
versioned Axys/APX preset semantics rather than universal Axys/APX behavior.

The active roadmap now points the freeze-readiness row to that packet instead
of repeating the checklist inline. The remaining near-term decision is therefore
clean: either accept the packet and freeze the demo as the preset seed later, or
change the packaged demo/YAML/report story before implementing `vendor: axys`.

This phase was documentation-only. It did not require report regeneration or
report-story edits, and the portfolio/security `report.html` hashes remained
unchanged.

### Phase 62: Freeze Packet Acceptance Audit

Status: complete for current freeze-packet auditability.

The Axys/APX Demo Freeze Decision Packet now includes an evidence map. Each accepted
boundary points to concrete support in packaged transaction CSVs, the packaged
README, comparison YAML comments, the field-role/source contract, extract
contract validation, demo matrix tests, and generated report-bundle health
checks.

At that time, the spot audit confirmed that packaged transaction CSV headers
omitted `TRANSACTION_ID`, retained the context columns needed for ambiguous
Axys/APX-style codes, and included only the then-current packaged transaction-code
families. Snapshot A contained `by`, `dp`, `dv`, `in`, `sl`, and `wd`; Snapshot
B contained those plus `li`. Neither packaged snapshot contained `lo` or `;`
in that audit state. This is historical context only: later phases promoted
packaged `lo` and fixed-income `pa`/`sa` rows, while `;` remains guardrail-only.

This phase did not freeze the demo or implement `vendor: axys`; it made the
freeze packet auditable. The remaining decision is still whether to accept that
packet as the seed.

The packaged demo health script regenerated both report bundles during this
audit. Portfolio and security `report.html` hashes remained unchanged, so this
phase did not change user-facing report content.

### Phase 63: Release Candidate Confidence Sweep

Status: complete for current release-candidate confidence.

The broad local release smoke passed after the freeze-packet audit work. The
full test suite passed with 608 tests. The package build succeeded for both
wheel and source distribution, and the build output confirmed packaged Axys/APX
demo CSV/YAML/README resources are included in the wheel.

The packaged performance-comparison demo health script passed after rebuilding
the portfolio and security report bundles, validating both bundles, checking
the extract-availability contract, and validating the packaged demo matrix. The
comparison YAML also passed `validate_config` with ambiguous-flow enforcement
enabled and observed transaction codes then limited to `by`, `dp`, `dv`, `in`,
`li`, `sl`, and `wd`. Later phases expanded the packaged observed-code set to
include `lo`, `pa`, and `sa`.

A stale-term sweep found only intentional historical roadmap references,
negative regression assertions, and boundary language that rejects universal
Axys/APX assumptions. No runtime or user-facing wording change was needed.

Portfolio and security `report.html` hashes were captured before and after the
smoke. They remained unchanged, so this phase did not change user-facing report
content.

### Phase 64: Axys/APX Demo Preset Seed Acceptance

Status: complete for docs-only preset-seed acceptance.

The Axys/APX Demo Freeze Decision Packet is accepted as the versioned seed for a
future `vendor: axys` preset. This changes the release posture from near-pass
to accepted seed for the packaged Axys/APX demo scope. It does not implement
`vendor: axys`, does not add hidden runtime policy, and does not change the
comparison YAML policy or packaged CSV data.

The active release-hardening backlog no longer carries the freeze-readiness
decision as an open item. The remaining vendor-preset work is now implementation
work in Eventual Deliverables: expand the accepted seed into inspectable
resolved YAML, preserve deterministic site overrides, and keep ambiguous-flow
and complete-YAML safeguards in force.

This phase was documentation-only. It did not require report regeneration or
report-story edits, and the portfolio/security `report.html` hashes remained
unchanged.

### Phase 65: Post-Freeze Backlog Reorientation

Status: complete for docs-only post-freeze reorientation.

The roadmap and design notes now separate two ideas that should not be blurred:
the packaged Axys/APX demo is accepted as the future `vendor: axys` preset seed,
but vendor-preset infrastructure is deliberately parked in Eventual
Deliverables until the project explicitly chooses that product lane.

The current open work should not drift back into `vendor: axys` implementation
by accident. The practical next lanes are non-preset backlog work: richer APX
demo design, transaction-policy expansion, cautious code simplification, or
documentation freshness. Preset implementation remains future work even though
the seed is accepted.

This phase was documentation-only. It did not require report regeneration or
report-story edits, and the portfolio/security `report.html` hashes remained
unchanged.

### Phase 66: Packaged `lo` Deliver-Out Scenario

Status: complete for one packaged external-cash `lo` promotion.

The packaged Axys/APX demo now includes a single Snapshot B inserted `lo` row for
`ALPHA` February 2026. The row uses `CASHUSD`, `$pty/$cash`
source/destination context, a negative amount, zero quantity/price/commission,
and same-day settlement. The scenario is intentionally framed as an external
cash deliver-out, not as a generic withdrawal and not as evidence that `lo` is
safe by code alone.

The rebuild pipeline derives the matching lower `CASHUSD` ending holding,
portfolio performance row, and reconstruction diagnostics from the scenario.
The YAML rule remains conditional: `lo` is an external flow only when reviewed
external-party context is present, while internal-transfer branches stay
available for site/test fixtures.

This phase changes the generated portfolio and security report content because
the new fully explained ALPHA February period appears in the user-facing review
surface. The packaged report bundles were regenerated and the new
`report.html` hashes were captured during validation.

### Phase 67: Axys/APX Demo Candidate Promotion Scout

Status: complete for docs-only promotion decision; superseded by the Phase 71
`pa`/`sa` packaged-demo promotion.

The first post-freeze candidate-promotion scout reviewed the new test-only site
variants for `pa`/`sa`, `ai`, `rc`, `pd`, and `ss`/`cs`. The result is
deliberately conservative: no additional transaction type should be promoted
into the packaged Axys/APX demo yet.

At the time, `pa` and `sa` were the strongest future candidates because a bond
accrued-interest purchase/sale story was realistic and directly relevant to
Modified Dietz performance review. Phase 71 later promoted them only after the
packaged demo could derive the transaction rows, cash movement,
quantity-driven holding/accrual rows, `secperf.csv`, `portperf.csv`, and report
comments from one coherent scenario intent.

`ai` and `ss`/`cs` remain onboarding override examples rather than packaged
defaults. `rc` and `pd` now have narrow context-gated packaged stories, while
their broader local variants still prove tested YAML shapes for sites that can
supply local evidence. Cost basis, principal, factor, amortization, margin, and
short-account details remain best-efforts demo-construction context unless a
future return formula explicitly uses those fields. The Modified Dietz product
boundary remains beginning value, ending value, dated external flows,
and configured income/fee performance rows.

This phase was documentation-only. It did not require report regeneration or
report-story edits, and the portfolio/security `report.html` hashes remained
unchanged.

### Phase 68: `pa` / `sa` Demo Promotion Design Packet

Status: complete for docs-only implementation blueprint; implemented by Phase
71.

The `pa`/`sa` accrued-interest research was strong enough to design a
packaged-demo promotion, but not to implement it casually. The promotion needed
to be a coherent fixed-income scenario, not a pair of copied transaction rows.

Recommended scenario:

- Portfolio/security/month: use the existing `INCOME` portfolio and `91282Y5Y1`
  or `91282Y2Y1` in one late-period month, preferably May 2026, where the demo
  already has fixed-income activity and report readers can compare it with the
  ordinary `in` interest row.
- Snapshot shape: Snapshot A should represent the control case without the
  accrued-interest trade adjunct; Snapshot B should add or change a paired bond
  trade plus matching accrued-interest rows.
- Transaction rows: include a principal `by`/`sl` row paired with `pa`/`sa`
  on the same trade/settlement date. Do not add standalone `pa`/`sa` rows
  without a principal trade context.
- YAML rules: classify `pa` and `sa` only through explicit bond/fixed-income
  context. They must not become code-only defaults and must not become
  portfolio external flows.
- Demo-construction rows: derive cash movement, holdings quantity/market value,
  holdings accrued-interest, `secperf.csv`, and `portperf.csv` from one
  scenario intent. Cost basis, amortization, and tax-lot details stay
  best-efforts demo-construction context unless a future formula surface uses
  them.
- Report story: explain that the difference comes from fixed-income
  accrued-interest settlement associated with the bond trade. The reviewer
  should see that the rows affect performance/income timing or security-level
  return reconstruction, but are not Modified Dietz external contributions or
  withdrawals.

Acceptance checklist before implementation:

- The scenario generator or source rows can produce transactions, holdings,
  accrued-interest values, security performance, and portfolio performance
  consistently.
- Portfolio-level Modified Dietz excludes `pa`/`sa` from external-flow
  weighting.
- Security-level Modified Dietz either treats the rows as configured
  income/fee performance rows or leaves them as review evidence, according to
  explicit YAML.
- Report explanations use accrued-interest language, not generic
  `transactions.amount` language, when the row is classified as an
  accrued-interest settlement.
- Raw audit trail and performance-cause rows make the paired principal and
  accrued-interest relationship visible enough that a reviewer does not need
  bond-accounting knowledge to start.
- Tests prove code-only `pa`/`sa` remain unknown or blocked, while explicit
  bond-context YAML classifies the packaged scenario deterministically.

Do not promote `pa`/`sa` in the packaged Axys/APX demo until the above can be met
without weakening the current "do not rebuild an accounting system" boundary.

This phase was documentation-only. It did not require report regeneration or
report-story edits, and the portfolio/security `report.html` hashes remained
unchanged.

### Phase 69: `pa` / `sa` Implementation Feasibility Probe

Status: complete for docs-only feasibility decision.

The packaged demo rebuild path can support a `pa`/`sa` promotion, but not as a
manual CSV edit. The current machinery is intentionally scenario-table driven:

- `performance_comparison_transaction_scenarios.csv` derives Snapshot B
  `transactions.csv` from Snapshot A plus explicit transaction scenarios.
- `performance_comparison_holding_scenarios.csv` supplies explicit holding
  adjustments that are not safely inferable from simple transaction deltas.
- `rebuild_performance_comparison_demo_data.py` derives `holdings.csv`,
  `secperf.csv`, and `portperf.csv` and audits the checked-in packaged files
  against the scenario sources.

That is the right home for `pa`/`sa`, but the current rules do not yet have a
fixed-income accrued-interest adjunct concept. They know:

- `by`/`sl` as security-level trade flows;
- `dv`/`in`/fee-like `dp` as income/expense rows;
- `li`/`lo`/`wd` as contextual external-flow/cash-balance cases; and
- explicit holding scenarios such as accrual corrections and cost-only
  corrections.

Promoting `pa`/`sa` cleanly therefore needs a small generator-side extension:

- add a fixed-income accrued-interest scenario type or transaction-effect
  branch for `pa`/`sa`;
- keep `pa`/`sa` out of portfolio external-flow weighting;
- include `pa`/`sa` in security/portfolio income or review evidence only
  through explicit YAML semantics;
- derive any cash-balance and `ACCRUED` holding movement from scenario intent;
- update scenario coverage expectations and audit tests so the new rows are
  intentional, not accidental backlog leakage; and
- keep code-only `pa`/`sa` test fixtures blocked or unknown.

Feasibility decision: implementation is reasonable after this small fixed-income
adjunct abstraction exists. Do not promote `pa`/`sa` by hand-editing packaged
CSV files. The next implementation phase should first teach the rebuild/audit
path how to express accrued-interest adjuncts, then add the paired bond trade
story.

This phase was documentation-only. It did not require report regeneration or
report-story edits, and the portfolio/security `report.html` hashes remained
unchanged.

### Phase 70: `pa` / `sa` Rebuild Abstraction Slice

Status: complete for tested rebuild-path support; superseded by the packaged
demo promotion in Phase 71.

The operational rebuild path now has an explicit fixed-income accrued-interest
adjunct mechanism for `pa` and `sa`. The support is intentionally narrow:

- `pa`/`sa` are included in the rebuild script's income-code set for derived
  security/portfolio performance when the comparison YAML classifies them as
  performance income/expense.
- `pa`/`sa` transaction scenarios can derive cash-balance adjustments.
- `pa`/`sa` transaction scenarios do not derive `holdings.accrued` adjustments;
  accrued holdings are quantity/terms driven in the demo rebuild path.
- Portfolio external-flow weighting still ignores `pa`/`sa`.
- The packaged Axys/APX demo contains one narrow paired 91282Y5Y1 `pa`/`sa` story.

A focused audit test builds temporary inserted `pa` and `sa` scenario rows and
verifies the derived cash impacts without treating accrued-interest settlement
amounts as direct holding-accrual changes.

Phase 71 used this abstraction to add the coherent paired bond trade story,
update scenario coverage expectations, update YAML rules, rebuild the packaged
data, and regenerate portfolio/security reports. Phase 70 itself changed code
and tests but did not change packaged demo data or regenerate reports; the
portfolio/security `report.html` hashes remained unchanged until Phase 71.

### Phase 71: `pa` / `sa` Packaged Demo Promotion

Status: complete for one narrow fixed-income accrued-interest packaged story.

The packaged Axys/APX demo now includes paired 91282Y5Y1 `by`/`pa` and `sl`/`sa`
rows in Snapshot B. The rows are deliberately scoped:

- `pa` is purchase accrued interest and resolves as fee/expense only with
  fixed-income context;
- `sa` is sale accrued interest and resolves as income only with fixed-income
  context;
- both rows stay outside portfolio external-flow weighting;
- the rebuild path derives matching cash-balance, security-performance, and
  portfolio-performance effects while leaving accrued holdings to quantity/terms
  logic; and
- code-only `pa`/`sa` remains unsafe outside the packaged paired-trade story or
  a reviewed site override profile.

The source contract, transaction semantics matrix, packaged README, readiness
checklist, Axys/APX reference notes, fixed-income boundary registry, and tests
now reflect this boundary. Report wording uses the same Modified Dietz
cash-balance input language as other performance cash transactions.

This phase changed generated report content. The final portfolio/security
`report.html` hashes are captured in the phase summary.

### Phase 72: Post-`pa` / `sa` Report Clarity Sweep

Status: complete for the fixed-income report-reader boundary after the packaged
`pa`/`sa` promotion.

The report artifacts were reviewed for the INCOME/91282Y5Y1 February period. The
important reader boundary is now pinned in both docs and tests:

- paired `by`/`pa` and `sl`/`sa` rows explain cash/performance input movement;
- `transactions.quantity`, `transactions.price`, and `holdings.quantity` remain
  supporting rows for the paired trade story;
- `holdings.accrued` remains a separate holding input driven by quantity/terms,
  not by direct `pa`/`sa` causality; and
- cost remains raw audit context, not a Modified Dietz explanation target.

The packaged Axys/APX README now describes the fixed-income story in those terms,
and the workbook contract test asserts that `pa`/`sa` guidance does not leak
onto the `holdings.accrued` row.

### Phase 73: Axys/APX Demo Freeze Re-Audit

Status: complete for post-`lo` and post-`pa`/`sa` freeze-packet alignment.

The Axys/APX Demo Freeze Decision Packet was rechecked after the packaged `lo` and
fixed-income `pa`/`sa` promotions. The source contract already reflected the
current accepted future `vendor: axys` seed: packaged families are `by`, `sl`,
`dv`, `in`, fixed-income `pa`/`sa`, fee-like `dp`, external-cash `li`,
external-cash `lo`, and external-cash `wd`; `;` remains guardrail-only.

The cleanup was limited to historical roadmap notes from earlier freeze audits.
Those notes still correctly describe what was observed at that time, but now
explicitly point forward to the later `lo` and `pa`/`sa` promotions so readers
do not mistake old spot-audit code lists for the current packaged demo surface.

### Phase 74: Axys/APX Packaged Demo Release Candidate Sweep

Status: complete for the post-promotion release-candidate smoke.

The full local test suite passed with 615 tests and 515 subtests after the
packaged `lo` and fixed-income `pa`/`sa` promotions. The package build
completed for both source distribution and wheel.

Archive inspection confirmed that the wheel and source distribution include the
packaged Axys/APX README, comparison YAML, Snapshot A/B transaction CSVs, and
Snapshot B restatement notes. The same inspection confirmed that `_demo_output`
and `tests/data` are absent from both archives.

The packaged performance-comparison demo health script passed after checking
demo rebuild drift, extract availability rendering, portfolio/security bundle
generation, bundle validation, and demo-matrix validation. The phase did not
change generated report content.

### Phase 75: Low-Risk Cleanup And Dead-Code Scout

Status: complete for one safe private-helper cleanup.

The cleanup scout removed an unused private security-period helper from the
comparison engine after confirming the active report-linking path no longer
called it. Focused period-linking and report tests passed, demo health passed,
and generated report content did not change.

### Phase 76: Axys/APX Packaged Demo Naming And Onboarding Polish

Status: complete for the packaged Axys/APX YAML rename and reader-path polish.

The packaged user-facing Axys/APX comparison YAML is now named
`axysapx_performance_comparison.yaml`. Test-only validation and site-variant
fixtures keep their generic `ppar_performance_comparison.yaml` names because
they are fixture specifications, not packaged onboarding artifacts.

The packaged demo README now states the two intended purposes explicitly:
marketing review of the generated workbook/HTML output, and onboarding support
for configuring a new Axys/APX-style site. Demo entrypoints, rebuild tooling,
package-resource checks, repository docs, and validation instructions reference
the Axys/APX-specific packaged filename. The future `vendor: axys` preset remains
design-only and parked in Eventual Deliverables.

### Phase 77: Axys/APX Setup Starter Kit

Status: complete for the first-client-segment onboarding path.

The packaged Axys/APX demo now keeps one-time setup guidance in the template
README for Axys/APX performance-comparison onboarding. The setup guidance is
intentionally procedural: run `ppar setup`, confirm copied source files, validate
before report generation, keep core transaction types conservative, generate
portfolio and security report bundles, review the workbook in order, and iterate
local overrides one at a time.

The packaged demo README presents that setup guidance before the longer demo
reference material. The sample YAML now includes practical comments about
the first site edits: change snapshot paths and schema mapping, keep filenames
separate from column mapping, run `validate_config`, and avoid guessing at
ambiguous Axys/APX transaction codes without site evidence. This phase did not
change report logic, transaction semantics, or the future `vendor: axys`
implementation boundary.

### Phase 78: Setup Literal Dry Run And Starter Layout Fix

Status: complete for setup path validation.

The setup guide was dry-run in a temporary onboarding directory. Copying only
`axysapx_performance_comparison.yaml` failed as expected because the YAML uses
relative paths to the schema and Snapshot A/B folders. Copying the starter files
together passed `validate_config` and generated both portfolio and security
report bundles with their level-specific audit XLSX/HTML files, `manifest.json`,
and support CSVs.

The setup guide now makes the starter layout explicit: keep the YAML, column
mapping, extract guardrail file, and Snapshot A/B folders together, or edit the
YAML paths before validation. The security-report path now uses a copied YAML
with `comparison.level: security`, matching the real `report_bundle` CLI.

### Phase 79: Axys/APX Onboarding Kit Consolidation

Status: complete for the current starter-kit reader path.

The Axys/APX setup guide now has a first-run edit map that tells onboarding
users which copied files to edit first and which sections to leave alone for
the first report. It reinforces that the starter kit is a working copy, that
Snapshot A/B paths and column mappings are the first safe edits, and that native
transaction codes and security identifiers remain case-sensitive.

The packaged Axys/APX YAML now carries the same boundary in comments: edit snapshot
paths, filenames, and column mappings before changing transaction rules or
impact methods. It also names the marketing-demo versus onboarding-base split:
the common operational story covers core transaction families, while richer
examples such as fixed-income `pa`/`sa` should stay unchanged until local bond
context and site evidence justify overrides.

### Phase 80: Axys/APX Setup Simplification

Status: complete for the current onboarding front door.

The Axys/APX setup guide was compressed into a shorter setup path: copy the
starter kit, replace demo extracts, validate, generate portfolio and security
reports, review in order, and iterate carefully. The simplified guide keeps the
same working commands and starter layout, but removes the repeated purpose and
success-criterion prose so new users see the operational path first.

The top-level README now points Axys/APX performance-comparison users to the
setup guide before the longer packaged-demo reference. This phase did not
change demo data, transaction rules, report logic, or the future `vendor: axys`
boundary.

### Phase 81: One-Folder Setup And Production Report Commands

Status: superseded by the Axys/APX workspace setup path.

The onboarding path now separates one-time setup from repeatable report
generation:

```bash
ppar setup ./my_ppar_data
ppar analytics ./my_ppar_data/analytics
ppar audit ./my_ppar_data/audit
```

Setup creates separate `analytics` and `audit` folders, each
with one user-facing `ppar.yaml`. The production performance-comparison command
writes portfolio and security reports by default when required source files are
available, and supports `--report portfolio`, `--report security`, or
`--report both` for explicit selection.

### Phase 82: Setup Python Runner Scripts

Status: refreshed for the current setup site.

`ppar setup` now installs optional Python runner scripts beside the data and
YAML they execute:

- `analytics/run_analytics.py`
- `audit/run_audit.py`

The setup README mentions those scripts only as optional examples. The main
onboarding path remains the `ppar` command plus heavily documented nearby YAML.
The scripts use setup-site CSV files and paths relative to their own directory,
so users can copy them into schedulers or notebooks without learning
repository-only demo modules.

`ppar setup --include-generic-analytics` can also copy the maintainer-facing
generic analytics sample and its Python runner script, but this remains hidden
from the public Axys/APX onboarding README so it does not distract first-target
users.

### Phase 83: Generic Analytics Disposition

Status: complete for the current public onboarding boundary.

The `generic_analytics` dataset and optional setup script are retained, but
their role is now explicitly maintainer-facing:

- they feed README marketing images;
- they support analytics regression and optional-value tests;
- they provide the source universe for selected operational demo-data
  derivation scripts; and
- they remain useful for source-checkout smoke runs.

They are not the primary installed-user onboarding path. New users should start
with `ppar setup`, which copies the Axys/APX `analytics` and
`performance_comparison` starter folders. The optional Python runner scripts
also use those Axys/APX setup folders rather than the generic demo module.

The generic refresh guide and demo-data README now state this boundary directly.
No generic data or setup scripts were deleted because they still serve the
README, tests, and maintainer refresh workflow.

### Phase 84: Open-Item Reconciliation And Release Backlog Slimming

Status: complete for the current release-hardening backlog.

The near-term release-hardening table now contains only active work with a
concrete next action. Completed housekeeping trains, such as the root README
image/artifact audit, moved into completed status and guardrail tests instead
of remaining as open backlog.

Ongoing reminders that do not currently justify their own implementation train
were moved into standing maintenance criteria:

- keep the marketing README aligned with generated artifacts;
- simplify code only when a small, behavior-preserving cleanup is obvious; and
- profile only when data growth, report-shape changes, or measured timings show
  a real bottleneck.

### Phase 85: Documentation Freshness Guardrail Sweep

Status: complete for the current setup/onboarding vocabulary.

User-facing setup docs now have an explicit metadata guardrail against retired
front-door language:

- `quickstart` / `QUICK_START`;
- `ppar report` / `ppar-report`;
- the former `performance_comparison_roadmap.md` filename;
- former `axys_full_spec` snapshot folders; and
- removed public demo console-script names.

The remaining stale public-demo console-script references in maintainer
generation notes were rewritten as generic smoke-test and Axys/APX starter-data
references. The check intentionally targets installed-user docs and onboarding
surfaces, not historical phase notes or source-checkout-only maintenance
scripts.

### Phase 86: Compact Architecture Map

Status: complete for the current documentation set.

The project now has one small architecture document:
`docs/architecture.md`. It maps the installed `ppar` command surface, package
boundaries, analytics and performance-comparison data flow, setup data versus
maintainer data, YAML configuration boundaries, and report boundaries.

This phase intentionally did not add a decision log, separate onboarding
workflow document, or another performance-comparison design document. The
documentation rule is now: keep new docs rare, and prefer updating the roadmap,
architecture map, repository guide, setup README, or heavily commented YAML
before creating another durable document.

### Phase 87: Setup Site Dry Run And User Surface Audit

Status: complete for the current installed-style setup surface.

The installed-style flow was dry-run into a temporary site directory:

```bash
ppar setup /tmp/ppar_phase87_site
ppar analytics /tmp/ppar_phase87_site/analytics
ppar audit /tmp/ppar_phase87_site/audit
```

The console output stayed concise: setup prints the two run commands and the
customization pointer, analytics lists reviewable analytics files, and
performance comparison lists only the generated workbook paths. Running
`ppar analytics` or `ppar audit` from the setup root still
fails with a useful message, while running those commands from the specific
workflow folders works.

The dry run found one user-output clutter issue: the top-level `ppar` command
was priming Matplotlib cache paths inside `analytics/output`. Analytics cache
setup now happens inside `ppar.analytics.cli` with a temporary cache directory,
and tests assert that `analytics/output` does not receive `.matplotlib` or
`.cache` folders.

### Phase 88: Release Package Smoke Audit

Status: complete for the current wheel/sdist surface.

The package was rebuilt with:

```bash
python -m build --no-isolation
```

The wheel audit confirmed:

- `entry_points.txt` exposes only `ppar = ppar.cli:main`;
- Axys/APX analytics starter files are included;
- Axys/APX performance-comparison starter files and Snapshot A/B CSVs are
  included;
- `_demo_output`, `scripts`, `tests`, `docs`, former `axys` demo parent paths,
  and former `axys_full_spec` paths are not present in the wheel; and
- checkout-maintenance scripts remain in the sdist, not the wheel package.

The built wheel was installed into a temporary environment and the installed
`ppar` command successfully ran:

```bash
ppar --help
ppar setup /tmp/ppar_phase88_site
ppar analytics /tmp/ppar_phase88_site/analytics
ppar audit /tmp/ppar_phase88_site/audit
```

The no-deps temporary wheel environment initially lacked `openpyxl`, so workbook
export failed until that declared runtime dependency was supplied locally. This
confirmed that `openpyxl` must remain a core runtime dependency, not an optional
extra, while Performance Comparison writes `.xlsx` workbooks by default.

### Phase 89: Release Candidate Tagging And Version Readiness Audit

Status: complete for pre-tag readiness audit; no tag was created.

The audit confirmed at that time:

- the working tree was clean before the audit;
- `pyproject.toml` was the only package-version authority and declared `0.1.4`;
- no package-level `__version__` exists to drift;
- the only existing tag is `workbook-review-checkpoint`, not a semantic release
  tag;
- build artifacts such as `dist/`, `build/`, and `*.egg-info` remain ignored;
  and
- Phase 88 already verified the built wheel and installed `ppar` smoke path.

The repository guide now carries the compact release-readiness checklist. A
release tag should wait until the maintainer explicitly chooses whether the next
public artifact remains `0.1.4` or becomes a new version such as `0.1.5` or
`0.2.0`.

### Phase 90: Version Decision And Release Tag Prep

Status: complete for the `0.1.5` release-candidate decision.

The maintainer selected `0.1.5` as the next public version rather than `0.2.0`.
The package-version authority remains `pyproject.toml`, and no package-level
`__version__` was added. The version bump is intentionally small: this remains a
pre-1.0 release, but it captures the Axys/APX onboarding, setup, command-line,
package-surface, README, and release-readiness hardening that accumulated after
`0.1.4`.

Release-tag readiness for `v0.1.5` requires:

- focused package metadata tests passing;
- fresh wheel/sdist build succeeding;
- built metadata reporting `Version: 0.1.5`; and
- a clean working tree before creating the local tag.

### Release Notes Stub

- `v0.1.5`: Axys/APX setup, analytics and performance-comparison onboarding,
  single `ppar` CLI, packaged starter data, streamlined report commands,
  level-specific `portfolio_audit.*` and `security_audit.*` review artifacts,
  marketing README and product-overview refresh, and release artifact audit.

### Phase 91: Publish Dry Run And Final Artifact Audit

Status: complete for the local `0.1.5` publish dry run.

The local `dist/ppar-0.1.5.tar.gz` and `dist/ppar-0.1.5-py3-none-any.whl`
artifacts passed `twine check`. Wheel metadata reported `Name: ppar`,
`Version: 0.1.5`, and `Requires-Python: >=3.12.1`; `entry_points.txt` exposed
only `ppar = ppar.cli:main`.

Package-content inspection confirmed:

- Axys/APX analytics starter data is present in the wheel;
- Axys/APX performance-comparison Snapshot A/B starter data is present in the
  wheel;
- `_demo_output`, `tests`, `docs`, former `axys` demo parent paths, and removed
  `RESTATEMENT_NOTES.md` files are absent from the wheel;
- the source distribution contains `pyproject.toml`; and
- checkout-maintenance scripts remain source-distribution material, not wheel
  package data.

The built wheel was installed into a temporary environment from `dist/` with
`--no-deps`. `ppar --help`, `ppar setup`, and `ppar analytics` passed. As
expected for a no-deps install, `ppar audit` reported that
`openpyxl` is required. Rerunning with the already-installed local dependency
set available completed both portfolio and security workbook generation.

No artifact was uploaded and no tag was pushed.

### Phase 92: Tag Placement Decision And Release Notes Stub

Status: complete for local tag-placement preparation; no artifact was uploaded
and no tag was pushed.

The release notes remain intentionally small and live in this roadmap rather
than a separate changelog. The release record now treats the publish dry-run
audit as part of the tagged source snapshot, so the local `v0.1.5` tag should
point at the final audited release-record commit before any remote push or PyPI
publish.

### Phase 93: Remote Release Readiness Check

Status: complete for read-only remote readiness; no artifact was uploaded and no
tag or branch was pushed.

The local repository was clean before the check. `main` was ahead of
`origin/main`, and `origin/main` was an ancestor of local `main`, so the local
release branch can be pushed without a merge conflict based on the currently
known remote state. A read-only remote query confirmed:

- `origin/main` exists at `ee5d56e`; and
- remote tag `v0.1.5` does not exist yet.

The local `v0.1.5` tag should remain aligned with the final release-record
commit before any push. Do not publish to PyPI until the maintainer explicitly
chooses to upload the already-audited `0.1.5` artifacts.

### Phase 94: Local Release Pause Checkpoint

Status: complete for local release-pause documentation; no artifact was uploaded
and no tag or branch was pushed.

The release-candidate and package-content checks passed again after the final
documentation freshness work. Fresh local `dist/ppar-0.1.5*` artifacts were
rebuilt from current `HEAD` and passed `twine check`, metadata inspection, and
package-content inspection.

The local `v0.1.5` tag is intentionally not trusted as the release pointer yet:
it currently points at an older local commit, while current `HEAD` contains later
release-readiness fixes. A read-only remote check confirmed that `origin` does
not yet have a `v0.1.5` tag. Before any push or publish, rerun the pre-publish
checklist in `docs/repository_guide.md`, retag local `v0.1.5` to final `HEAD`,
rebuild `dist/`, and run `twine check`.

### Phase 95: Public Version API And Fresh RC Pass

Status: complete for package-surface polish; no artifact was uploaded and no tag
or branch was pushed.

The package now exposes `ppar.__version__`, sourced from installed package
metadata, so Python callers can inspect the installed release without parsing
`pyproject.toml`. The metadata test ties this public version value back to the
single package-version authority.

Fresh release-candidate checks passed from the updated commit:

```bash
./.venv/bin/python scripts/check_release_candidate.py
./.venv/bin/python scripts/check_release_candidate.py --build
```

The build pass refreshed local ignored `dist/ppar-0.1.5*` artifacts from the
current source tree. The local `v0.1.5` tag was not moved, no remote tag was
created, and nothing was published.

### Phase 96: Axys/APX Demo Simplification

Status: complete for the current packaged-demo review cadence.

The packaged Axys/APX performance-comparison demo now uses smaller review
periods where the original monthly periods had too many intentional
differences. The scenario calendar keeps each packaged demo period at or below
the two-difference target, and the period split backlog is now empty.

The simplification kept the demo economics and reviewer stories intact while
making each period easier to inspect:

- INCOME February separates the 91282Y5Y1 purchase/accrued-interest pair from
  the sale/accrued-interest pair;
- INCOME May separates the AAPL valuation mark, 91282Y2Y1 income/accrual story,
  and 36225MBS1 principal paydown;
- BALANCED May separates the AAPL valuation/CVNA split story, the MSFT holding
  correction, and the TSLA short-sale/cover-short pair; and
- the roadmap now treats `rc`, `pd`, `ss`/`cs`, and the CVNA split-processing
  correction as scoped packaged-demo examples rather than merely future
  candidates.

The rebuild audit reports no packaged-data drift and no audit issues. Focused
tests assert that no scenario period needs another intra-month split and that
the split-plan backlog remains empty. The supporting research notes still warn
against broad, universal Axys/APX claims for these transaction codes; the
packaged rows are plausible/defensible examples, not vendor-wide semantics.

### Phase 97: Named Audit Artifacts And Release Requalification

Status: complete for the `0.1.5` release-candidate artifact boundary.

Portfolio and security review entrypoints now identify their review level:

- portfolio bundles write `portfolio_audit.html` and, by default,
  `portfolio_audit.xlsx`;
- security bundles write `security_audit.html` and, by default,
  `security_audit.xlsx`; and
- rebuilding a bundle removes stale generic `report.*` files and wrong-level
  audit files from the reused output directory.

The filename mapping is centralized in `review_model.py` and exposed through
`report_bundle_contract()`. Generated manifests, review summaries, bundle
READMEs, CLI and setup-script output, validators, public documentation,
screenshots, and the product-overview PDF use the same names. The old generic
filenames are not retained as duplicate aliases because they would leave two
apparently authoritative review entrypoints in one bundle; the `v0.1.5` release
record calls out the intentional pre-1.0 output-contract change.

The packaged portfolio and security bundles were regenerated and passed bundle
validation. The setup-installed Python runners produced matching named audit
artifacts, the scenario matrix passed, and the Analytics/Performance Auditing
10x scale smoke passed. README screenshots and `PPAR.pdf` were refreshed, then
the full project check completed with wheel and source-distribution build
validation. No tag was moved and no artifact was published as part of this
phase.

### Phase 98: Release Record Consolidation And Tag Alignment

Status: complete for the local `v0.1.5` release boundary.

The Phase 97 implementation is isolated in commit `e11466f` (`Name performance
audit artifacts`). A build-qualified release-candidate pass from that committed
source completed the packaged demo-data audit, extract-availability check,
portfolio/security bundle generation and validation, setup-installed runner
smoke, scenario-matrix validation, 10x scale smoke, full project checks, and
wheel/source-distribution build validation.

This release-record commit is the final local `v0.1.5` source boundary. The
working tree is clean, and the local tag points to this commit rather than the
older pre-Phase-97 checkpoint. A read-only remote check confirms that no remote
`v0.1.5` tag exists. No branch or tag was pushed and no distribution artifact
was published as part of this phase.

## Guiding Principle

Do not casually rebuild an accounting system.

The goal is not to reproduce every vendor methodology perfectly. The goal is to
create a transparent, auditable, configurable return-reconstruction layer that
can explain most real-world differences and clearly identify when vendor
methodology or missing data prevents full explanation.

The future standard should be:

```text
Count formula inputs.
Show supporting accounting evidence.
Do not double count.
Fail hard when required YAML rules are missing.
```

## Transaction-Type Backlog

Use Phase 8 as the implementation order. This backlog is the policy boundary:

- Packaged demo additions should be realistic, internally consistent, and
  understandable to a reviewer without knowing the fixture harness.
- Test-only fixtures should absorb synthetic edge cases, code-only failure
  cases, local-policy gaps, and ambiguous Axys/APX semantics that need surgical
  proof.
- Evidence-blocked cases should stay backlog until IMEX context, REP/report
  semantics, or real source samples justify a classification.

Near-term backlog by home:

| Home | Transaction families | Exit criteria |
| --- | --- | --- |
| Packaged demo candidate | nonredundant external-flow variants and additional corporate actions only when they add a new reviewer lesson beyond the packaged `rc`, `pd`, `ss`/`cs`, and CVNA split-processing examples | Scenario intent, transactions, holdings, security performance, portfolio performance, YAML rules, workbook explanations, and README/source-contract language all align. |
| Test-only fixture | `li`/`lo` external and neutral variants; more `dp`/`wd` fee/sweep/external cases; tested candidate override profiles for `ai` and local `pa`/`sa` variants; uppercase reversal/cancellation; synthetic corporate actions | Fixture proves expected semantics or failure mode without implying the packaged demo is a realistic client story. Candidate override profiles remain copy/adapt onboarding examples when local site evidence differs from conservative packaged defaults. |
| Evidence-blocked backlog | Mergers, spin-offs, ticker changes | Axys/APX REP/report evidence identifies required fields and ppar treatment well enough to avoid code-only classification. |
