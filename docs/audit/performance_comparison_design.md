# Performance Comparison Design Notes

Performance Comparison is the attribution sub-feature inside the broader
Audit workflow. The public workflow asks whether reported
performance changed and whether source-data relationships look suspicious. This
design note focuses on the internal comparison engine that explains changed
reported performance between two source-data snapshots.

## Purpose

The performance comparison feature explains why calculated performance for the
same portfolio and period changed between two source-data extraction dates.

This document is the deep design/reference note. Active Audit implementation is
tracked in the
[`PPAR Audit MVP Completion Plan`](mvp_plan.md), and
broader Audit direction is governed by the
[`PPAR Audit Product Constitution`](product_constitution.md) and
[`PPAR Audit Roadmap`](roadmap.md).
The maintainer-facing safety guarantees and their audited enforcement baseline
are defined in
[`Audit Safety Invariants`](safety_invariants.md).

The core question is:

> Why did my performance for period A change from when I ran it on date 1 to
> when I ran it on date 2?

The feature will compare two snapshot directories. Each snapshot contains
vendor exports such as portfolio performance, security performance, FX rates,
transactions, and holdings.

This should not be treated as an Axys/APX-only feature. The comparison engine
should operate on normalized internal datasets. Vendor-specific behavior should
live in small normalization adapters, with Axys/APX as the first likely adapter.

The first implementation should stay intentionally narrow: compare normalized
portfolio performance, security performance, holdings, FX rates, and
transactions rows, report material changes, and produce a
clear finding model. Deeper causal inference can be added after the finding
model is stable.

## Document Boundary

This document owns stable comparison-engine and reviewer-model design. It does
not own implementation status or current priorities. Use the
[`MVP Completion Plan`](mvp_plan.md) for those questions and executable behavior,
tests, generated artifacts, and machine-readable contracts for implementation
truth.

## Supported Vocabulary

YAML files and report artifacts use plain string values so they remain easy to
read, diff, and edit. The package code centralizes the same values in
`StrEnum` classes to reduce drift and give internal APIs stricter annotations
where practical. Existing public constants remain as compatibility aliases for
the enum members.

Public YAML impact method values are centralized in:

- `TransactionImpactMethod`: `evidence_only`, `modified_dietz`, and
  `transaction_amount_delta_over_return_denominator`.
- `HoldingImpactMethod`: `evidence_only`,
  `quantity_delta_times_snapshot_a_unit_market_value_over_return_denominator`,
  `market_value_delta_over_return_denominator`, and
  `accrued_delta_over_return_denominator`.
- `PriceImpactMethod`: `price_delta_over_snapshot_a_price_times_weight`.
- `FxRateImpactMethod`: `evidence_only`.

The retired `contribution_impact_methods` configuration is rejected. Audit no
longer derives explanation inputs from optional portfolio- or
security-performance output columns.
Transaction sign/flow semantics are centralized in:

- `TransactionCategory`: `external_flow`, `income`, `fee_expense`, `buy`,
  `sell`, `transfer`, `corporate_action`, and `unknown`.
- `TransactionCashFlowSign`: `positive`, `negative`, `none`, and `unknown`.
- `TransactionPerformanceFlowSign`: `external`, `performance`, `neutral`,
  and `unknown`.
- `TransactionSemanticsSource`: `source`, `yaml_rule`, `mixed`, and
  `unknown`.

Modified Dietz policy options are centralized in:

- `ModifiedDietzFlowTiming`: `trade_date` and `settlement_date`.
- `ModifiedDietzDayCount`: `actual_days`.
- `ModifiedDietzInclusionRule`: `beginning_of_day` and `end_of_day`.
- `ModifiedDietzDoubleCountPolicy`: `cross_check_only`.

Finding and review classification values are centralized in:

- `FindingSeverity`: `informational` and `material`.
- `FindingConfidence`: `high`.
- `EvidenceRole`: `target_output`, `direct_input`, `related_output`, and
  `context`.
- `TransactionMatchStatus`: `matched_by_id`,
  `matched_by_singleton_fallback`, `added_in_snapshot_b`,
  `missing_from_snapshot_b`, `ambiguous_fallback_match`,
  `transaction_id_unmatched`, and `strict_fallback_unmatched`.
- `CauseArea`: the seven current coarse performance-cause areas.

Data Issues vocabulary is specified separately in
[`Data Issues Design`](data_issues_design.md).

When a value crosses into Polars tables, CSVs, HTML, XLSX, or YAML, it
should be serialized as its plain string value. Enum members are primarily for
construction, validation, and package-internal type clarity.

## Data Issues Sibling Sub-Feature

Data Issues is a sibling of Performance Comparison within Audit. Its report,
configuration, and issue vocabulary are specified in
[`Data Issues Design`](data_issues_design.md). Performance Comparison may
surface Data Issues context, but Data Issues are not additive performance
causes.

## Non-Goals For The First Pass

- Recalculate full portfolio performance from transactions and holdings.
- Replace existing `ppar.axys_apx` loading behavior.
- Produce final audit conclusions with false certainty.
- Handle every possible source-system schema variant.
- Make the comparison YAML a vendor-specific rule language.
- Build a GUI.

## Design Direction

The feature should have three layers:

1. Raw snapshot inputs: Vendor files in two snapshot directories.
2. Normalization: Vendor-specific adapters normalize files into standard
   internal datasets.
3. Comparison: A source-agnostic engine compares normalized datasets and emits
   findings.

The comparison engine should not know whether the source system was Axys/APX,
FactSet, Bloomberg PORT, custodian files, or another vendor. It should compare
standard datasets with standard column names and let adapters handle source
schema details.

## Current Package Shape

```text
ppar/audit/
  __init__.py
  aliases.py
  schema.py
  specification.py
  source_loader.py
  compare.py
  findings.py
  explain.py
  _transaction_diagnostics.py
  fx_rates.py
  period_linking.py
  holdings.py
  runner.py
  security_performance.py
  transactions.py
```

Current responsibilities:

- `specification.py`: Read and validate comparison YAML.
- `source_loader.py`: Load one snapshot directory, resolve optional files, and
  normalize configured columns.
- `compare.py`: Compare normalized snapshot A/snapshot B data sets.
- Dataset modules such as `fx_rates.py`, `transactions.py`, and
  `holdings.py`: dataset-specific loading, comparison keys,
  changed-column rules, and default aliases.
- `period_linking.py`: Link dated evidence to containing portfolio periods
  where the linkage is conservative.
- `findings.py`: Define finding records, roles, suppressions, and codes.
- `explain.py`: Build portfolio/security period summaries, evidence
  breakdowns, rankings, contribution candidates, cause summaries, and
  transaction summary tables.
- `_transaction_diagnostics.py`: Centralize transaction diagnostic labels,
  provenance-count formatting, match-status review notes, and business sort
  ordering used by explanation tables.
- `runner.py`: Public execution helpers, compact output tables, and summary
  tables.
- `report.py`: HTML, XLSX, and bundle report rendering over stable helper
  tables.

Report rendering is now inside the package, but remains a presentation layer
over stable Polars helper tables. The report module should avoid introducing
separate analytics logic.

## Core Concepts

### Snapshot

A snapshot is a directory of files extracted at one point in time. The snapshot
label should be user-controlled because extraction dates are often business
labels rather than filesystem dates.

Snapshots should be identified neutrally as `a` and `b`, not `old` and `new`.
This avoids implying that one snapshot is newer, better, or authoritative.
Unless otherwise stated, numeric deltas are calculated as snapshot B minus
snapshot A.

Each snapshot may have its own schema. The common case should be one shared
schema, but the configuration should allow snapshot-specific overrides when a
vendor export changes between extraction dates.

### Normalized Dataset

A normalized dataset is the source-agnostic form consumed by the comparison
engine. Initial normalized datasets include:

- `portfolio_performance`
- `security_performance`
- `fx_rates`
- `transactions`
- `holdings`

Normalization should be conservative: preserve useful source columns when they
help explain differences, but present required comparison fields with standard
internal names.

Unless otherwise stated, dataset names and column names in this document refer
to normalized internal comparison datasets, not raw source-file column names.

### Dataset Availability And Minimal Columns

Only `portfolio_performance` should be required. It is the top-level evidence
that performance changed and is enough to produce a useful portfolio-period
comparison.

All other datasets should be optional and may exist in any combination. Missing
optional datasets should reduce explanation depth, not fail the comparison,
unless the user explicitly marks a file as required for preflight existence
checking.

Each normalized dataset has a deliberately small supported column set. Extra
unmapped source columns are ignored unless a later product change gives them an
explicit evidence contract.

`portfolio_performance` required columns:

- `portfolio_id`
- `from_date`
- `thru_date`
- `portfolio_return`

`portfolio_performance` may also provide `base_currency` as portfolio metadata.
It does not supply separate valuation, flow, income, gain/loss, or contribution
evidence.

`security_performance` required columns:

- `portfolio_id`
- `security_id`
- `from_date`
- `thru_date`
- `security_return`

`security_performance` has no supported optional calculated-output columns.
Holdings and transactions provide valuation, weight, flow, and income evidence.

`fx_rates` required columns:

- `from_currency`
- `to_currency`
- `rate_date`
- `fx_rate`

`fx_rates` useful optional columns:

- `rate_source`
- `rate_type`

`fx_rates` represents exchange rates, not currency holdings or cash
balances. Currency exposure belongs in holdings, transactions, or valuation
datasets.

Required FX values must be complete, and `fx_rate` must be finite and strictly
positive. A normalized row is unique by currency pair, rate date, and any
available rate-source/rate-type provenance. These are source-integrity rules;
they do not establish an Axys-native quote direction, reciprocal convention, or
rate-selection method.

`transactions` required columns:

- `portfolio_id`
- `security_id`
- `transaction_date`

`transactions` useful optional columns:

- `transaction_id`
- `original_cost_date`
- `settlement_date`
- `transaction_code`
- `transaction_category`
- `cash_flow_sign`
- `performance_flow_sign`
- `quantity`
- `price`
- `amount`
- `commission`
- `original_cost`
- `currency`
- `broker`

`original_cost` and `original_cost_date` are normalized source evidence. They
remain optional for existing configurations and do not enter performance
comparison calculations. When the opt-in
`deliver_in_original_cost_incomplete` Data Issues check is enabled, both fields
must be available in each snapshot so absence can be distinguished from an
extract that never supplied the columns. The check treats numeric zero as a
supplied cost, reports incomplete evidence through the established Data Issues
schema, and does not infer cost basis or a source-system fallback.

Changed `commission` values are useful review evidence for fee, net amount, and
accounting-treatment differences. By default they remain context evidence. When
YAML sets `transaction_impact_methods.commission.method: evidence_only`, they
move to underlying-cause review rows without receiving return-impact estimates.

Future transaction enhancements should consider optional fixed-income and
income detail fields when real source files provide them:

- `accrued_interest`
- `interest`
- `principal`
- `gross_amount`
- `net_amount`
- `tax_withheld`

These fields should remain part of the `transactions` dataset rather than
forcing a separate income/accrual dataset by default. Holding-level `accrued`
represents an accrued balance at a holding date; transaction-level accrued
interest represents bought/sold accrued interest, posted interest, or other
activity detail. Transaction-level comparisons should still require stable
transaction matching, preferably by `transaction_id`.

`transaction_category` should use a small normalized vocabulary:

- `external_flow`
- `income`
- `fee_expense`
- `buy`
- `sell`
- `transfer`
- `corporate_action`
- `unknown`

The comparison loader can infer this category from common transaction codes
when the source does not provide a category. For Axys/APX-shaped demo data, the
source `transaction_code` values remain native lower-case codes such as `by`,
`sl`, `dv`, `in`, `dp`, `wd`, `rc`, and `pd`. YAML rules map those source codes
to normalized categories such as `buy`, `sell`, `income`, `external_flow`, and
`corporate_action`. The normalized category is an explanation label and a rule
selector; it does not replace the source transaction code in the audit trail.
Transaction-code normalization is limited to semantic classification and
rule-coverage checks for legacy configurations. By default, transaction-rule
keys and native `when` values match stripped source text by exact case,
compatibility inference from code alone is disabled, and unmatched uppercase
codes remain unknown. A maintained site may explicitly request
`legacy_case_insensitive` behavior. Neither mode assigns cancellation meaning to
uppercase codes. Reviewer handoff artifacts preserve native transaction-code
case.

`cash_flow_sign` and `performance_flow_sign` are optional source-supplied
semantics. They should not be inferred from transaction code in the first pass.
Directional sign rules are vendor-specific and accounting-specific enough that
guessing them would make `transaction_activity` look more precise than it is.

`cash_flow_sign` normalizes to:

- `positive`: Cash moves into the portfolio.
- `negative`: Cash moves out of the portfolio.
- `none`: No cash movement.
- `unknown`: Source value is blank, missing, or not recognized.

`performance_flow_sign` normalizes to:

- `external`: Treat as external flow for performance purposes.
- `performance`: Treat as performance-affecting activity.
- `neutral`: No performance-flow effect.
- `unknown`: Source value is blank, missing, or not recognized.

Comparison YAML may define explicit transaction rules keyed by source
transaction code. A matching complete rule is the site-reviewed authority for
the normalized category and both sign fields, even when the extract also
contains recognized semantic labels:

```yaml
transaction_rules:
  by:
    transaction_category: buy
    cash_flow_sign: negative
    performance_flow_sign: performance
  wd:
    transaction_category: external_flow
    cash_flow_sign: negative
    performance_flow_sign: external
```

Transaction impact treatment must also be explicit. External-flow rows can be
marked evidence-only or cross-check-only, and performance-treated amount rows
can opt into the only currently modeled amount-delta estimate:

```yaml
transaction_impact_methods:
  external_flow:
    method: evidence_only
  performance:
    method: transaction_amount_delta_over_return_denominator
    denominator_source: begin_market_value
  quantity:
    method: evidence_only
  price:
    method: evidence_only
  commission:
    method: evidence_only
```

The supported `transaction_impact_methods` contract is intentionally narrow:

- Top-level value must be a mapping.
- Supported keys: `external_flow`, `performance`, `quantity`, `price`, and
  `commission`.
- `external_flow` must be a mapping.
- Supported `external_flow.method` values: `evidence_only` and
  `modified_dietz`.
- `performance` must be a mapping.
- Supported `performance.method` value:
  `transaction_amount_delta_over_return_denominator`.
- Supported `performance.denominator_source` value: `begin_market_value`.
- `quantity`, `price`, and `commission` must be mappings.
- Supported `quantity.method`, `price.method`, and `commission.method` value:
  `evidence_only`.

This policy documents that transaction differences should remain review
evidence unless YAML explicitly selects a supported estimate or diagnostic
cross-check. Missing or unsupported method names are rejected so the comparison
never silently chooses a return convention. `modified_dietz` is supported only
as a cross-check diagnostic: its estimate is reported beside transaction
evidence and is excluded from regular contribution totals.

Security-level comparisons also support opt-in return reconstruction because
buys and sells are security-level capital flows. The diagnostic contract is:

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
  income_categories:
    - income
    - fee_expense
  return_basis: net
  sign_convention: signed_amount
```

When this section is present, missing or malformed configuration fails before
report generation. The current diagnostic implementation uses the normalized
transaction date, actual calendar days, and beginning-of-day flow inclusion.
Buy and sell transaction amounts are inverted from the cash perspective into
security-level flows: buys are inflows to the security, and sells are outflows
from the security. Income transactions, such as dividends and interest, are
included in the security numerator. Normal report bundles use reconstruction
internally, then allocate formula-level impacts back to recognizable source rows
in `Performance Difference Causes`; the detailed security diagnostic worksheet/CSV is
available only when reconstruction diagnostics are explicitly included.

Portfolio return reconstruction is available as an internal explanation source.
When YAML includes `portfolio_return_reconstruction`, normal report bundles can
allocate formula-level impacts back to source rows in `Performance Difference Causes`.
The detailed `Return Reconstruction Checks` worksheet/CSV comparing reported
`PORT_RETURN` differences with Modified Dietz returns derived from holdings and
external-flow transactions is available only when reconstruction diagnostics are
explicitly included.

Transaction impact output separates configured policy from review diagnostics:

- `transaction_impact_policy`: The YAML-selected policy label that applies to
  the row, such as `external_flow:evidence_only` or
  `performance:transaction_amount_delta_over_return_denominator`.
- `transaction_impact_diagnostic`: A review-only explanation of why a
  transaction row is not estimated or why a cross-check-only method has a
  separate diagnostic estimate.
- `transaction_impact_diagnostic_estimate`: A review-only return-impact
  estimate that is shown beside transaction evidence but excluded from
  contribution totals.

Diagnostics must not be interpreted as an aggregation instruction. They are
intended to make reviewer output auditable while full impact formulas remain
gated. Current diagnostic messages include:

- `external-flow evidence-only policy`: YAML explicitly says external-flow
  transaction rows remain review evidence.
- `external-flow impact method missing`: The row has external-flow semantics,
  but no supported external-flow impact method is configured.
- `modified_dietz cross-check estimate`: Internal checks have all required
  Modified Dietz inputs and the row has a diagnostic estimate excluded from
  contribution totals.
- `modified_dietz missing inputs: ...`: Internal checks found missing or
  disqualifying Modified Dietz inputs such as `flow date`,
  `portfolio period`, `nonzero begin_market_value denominator`, or
  `in-period flow date`.

Planned external-flow method names remain reserved and rejected until their
formulas and YAML inputs are implemented, except for the narrow `modified_dietz`
method:

- `modified_dietz`: Requires flow timing convention, day-count convention,
  beginning/end inclusion rule, denominator source, and
  `double_count_policy`.
- `subperiod_linked`: Requires subperiod boundary rule, linking formula, and
  large-flow threshold or explicit breakpoints.
- `unweighted_flow_delta`: Requires explicit reviewer acknowledgement that no
  day-weighting applies and a denominator source.

Each future method must define whether transaction deltas are independent of
portfolio-level `flow` deltas or only explanatory cross-checks. This prevents
double counting when portfolio performance rows already include the external
flow effect.

The `modified_dietz` YAML contract is:

```yaml
transaction_impact_methods:
  external_flow:
    method: modified_dietz
    flow_timing: trade_date
    day_count: actual_days
    inclusion_rule: beginning_of_day
    denominator_source: begin_market_value
    double_count_policy: cross_check_only
```

Supported `modified_dietz` fields:

- `flow_timing`: Which transaction date anchors the flow weight, such as
  `trade_date` or `settlement_date`. Allowed values: `trade_date`,
  `settlement_date`.
- `day_count`: How calendar distance is measured, starting with `actual_days`.
  Allowed value: `actual_days`.
- `inclusion_rule`: Whether a dated flow is treated as beginning-of-day or
  end-of-day for weighting. Allowed values: `beginning_of_day`, `end_of_day`.
- `denominator_source`: Which normalized field supplies the return denominator,
  such as `begin_market_value`. Allowed value: `begin_market_value`.
- `double_count_policy`: Whether transaction-derived impacts are only
  cross-check evidence when portfolio `flow` deltas are present. Allowed value:
  `cross_check_only`.

With `double_count_policy: cross_check_only`, eligible external-flow transaction
amount rows receive `transaction_impact_diagnostic_estimate`; they do not
receive `estimated_return_impact`, and they are not summed into impact coverage
or cause-summary totals.

Design reference examples for `modified_dietz` use:

```text
weighted_flow_impact = flow_delta * flow_weight / denominator
```

For `actual_days`, the period day count is inclusive. For a January 1 through
January 30 period, the period has 30 days. A January 11 flow has 20 remaining
days under `beginning_of_day` treatment and 19 remaining days under
`end_of_day` treatment. With `flow_delta = 300` and
`denominator = 10000`, the design reference impacts are:

- `beginning_of_day`: `300 * (20 / 30) / 10000 = 0.02`
- `end_of_day`: `300 * (19 / 30) / 10000 = 0.019`

These examples define the active cross-check-only calculation and must remain
aligned with the guardrail tests. The result is diagnostic evidence only until
impact selection, reporting, and double-counting behavior explicitly allow an
aggregate treatment.

The staged activation path for external-flow estimates is:

1. Validate YAML strictly so every method variability is explicit.
2. Carry the YAML method as a typed internal policy.
3. Surface eligibility diagnostics in review output without estimating.
4. Enable a cross-check-only Modified Dietz diagnostic estimate, with no
   aggregation into root-cause totals while double-counting risk remains.
5. Add any aggregate treatment only after linkage to portfolio-level flow
   deltas is explicitly modeled and tested.

When no YAML rule matches, recognized source category/sign semantics remain
usable. A partial matching YAML rule overrides the fields it defines and leaves
the other recognized source fields in place. Packaged code aliases provide only
the longstanding compatibility category fallback; their membership is loaded
from packaged YAML rather than defined in Python.

Loaded transaction rows also carry `transaction_semantics_source` so reviewers
can audit how the normalized category/sign/flow treatment was obtained:

- `source`: No matching YAML rule supplied semantics; usable sign/flow semantics
  came from recognized source fields, with any category supplied by source-data
  or the packaged compatibility fallback.
- `yaml_rule`: A complete matching `transaction_rules` entry supplied the
  normalized category and both sign fields.
- `mixed`: A partial matching YAML rule supplied one or more fields while other
  recognized semantic fields remained source-supplied.
- `unknown`: The row still lacks usable sign/flow semantics.

Transaction Activity and Impact Coverage summaries expose
`transaction_semantics_sources` as compact evidence-row counts such as
`source: 3`, `mixed: 1`, or `yaml_rule: 2`. These counts are review aids, not
distinct transaction counts, because the finding table is currently organized
by changed evidence field.

Both semantic fields must be present and non-`unknown` before a transaction row
has the sign/flow inputs required for a return-impact estimate. Until those
sign rules are explicitly source-supplied or configured in YAML and normalized,
transaction differences should remain evidence-only and should not receive an
estimated return impact.

Transaction activity should become eligible for a return-impact estimate only
when the changed evidence has an affected portfolio, security, linked portfolio
period, normalized transaction category, return denominator, and modeled
transaction sign and flow semantics. External-flow and neutral transaction
treatments require separate methods because their effect depends on the
portfolio performance formula's flow handling. Until an applicable method is
available, transaction summaries should expose missing impact inputs instead of
implying an estimate.

Workbook and HTML reports should include compact transaction activity review
rows so reviewers can see changed fields, deltas, and missing impact inputs
without digging through lower-level evidence tables.

`holdings` required columns:

- `portfolio_id`
- `security_id`
- `holding_date`

`holdings` useful optional columns:

- `quantity`
- `market_value`
- `cost`
- `accrued`
- `price`
- `currency`

Changed `cost` / cost-basis values are useful review evidence for tax,
accounting, and downstream unrealized gain/loss questions. By default they
remain context evidence. When YAML sets
`holding_impact_methods.cost.method: evidence_only`, they move to
underlying-cause review rows without receiving return-impact estimates.

Missing required columns should prevent that specific dataset from loading. If
the dataset is optional, the comparison should continue with a finding or report
note that explanation depth is limited. Missing optional evidence datasets may
produce a less detailed explanation without invalidating the reported returns.

### File Presence Requirements

`portfolio_performance` is always required and should not have a configurable
`required` flag. If it is missing from either snapshot, the comparison cannot
start.

All other files are optional by default. For those files, `required: true` means
only that file existence should be validated during preflight. It should not
change downstream comparison behavior.

File presence semantics:

- `portfolio_performance` is always required.
- An optional file omitted from the YAML is ignored.
- An optional file listed as a simple path is loaded if present; if missing, the
  comparison continues with a finding or report note.
- An optional file listed with `required: true` must exist in both snapshots, or
  the comparison fails up front with a clear error.
- Once an optional file exists and is loaded, downstream behavior is the same
  regardless of whether `required` was true or false.

### Comparison Key

The primary comparison keys should be configurable but have sensible defaults:

- Portfolio performance: portfolio code, from date, thru date.
- Security performance: portfolio code, security identifier, from date,
  thru date.
- Prices: security identifier, price date, currency/source where available.
- Transactions: transaction id when available; otherwise a conservative
  composite fallback.
- Holdings: portfolio code, security identifier, holding date.

When transactions have a stable `transaction_id` in both snapshots, matching
rows can report changed amounts as `PC-TXN-AMT` with `matched_by_id`. When no
transaction id is available, an exact singleton fallback may pair rows only
when there is exactly one transaction in each snapshot for the same portfolio,
trade date, security identifier, and native transaction code. This fallback is
case-sensitive and never uses fuzzy date, amount, quantity, or price matching.
Changed fields from this path are labeled `matched_by_singleton_fallback` so
reviewers can distinguish them from transaction-ID matches.

Rows outside that exact singleton fallback use the stricter row-presence key,
which includes portfolio, security, trade date, settlement date if present,
transaction code, quantity, price, and amount. This avoids guessing that two
similar rows are the same transaction. Transaction findings expose
`transaction_match_status` so reviewers can distinguish `matched_by_id`,
`matched_by_singleton_fallback`, `added_in_snapshot_b`,
`missing_from_snapshot_b`, `ambiguous_fallback_match`,
`transaction_id_unmatched`, and `strict_fallback_unmatched` evidence.

Duplicate comparison keys should fail loudly before row-presence checks or
value comparisons run. Silent duplicate handling can collapse rows into a set
or multiply rows during joins, both of which can produce misleading findings.
The default policy is therefore to raise an error for duplicate keys in either
snapshot.

The exception is transaction strict-fallback matching. Two identical same-day
transactions can legitimately occur when no stable `transaction_id` is
available, so duplicate fallback keys should be reported as
`ambiguous_fallback_match` diagnostics instead of paired as edits or rejected
as invalid source-data.

Transaction add/drop findings are built from the full source row, not only the
comparison key. When a no-ID transaction date changes, including a move from
one performance period to the next, the report should show one
`missing_from_snapshot_b` row at the old date/period and one
`added_in_snapshot_b` row at the new date/period. If the rows have explicit
external-flow semantics and a Modified Dietz policy, each add/drop row may
carry a review-only `modified_dietz cross-check estimate` using that row's own
period and flow date. The row is still not treated as an in-place transaction
edit.

### Finding

A finding is one observed difference or one explanation of a difference.
Findings should be data, not prose-only output.

Candidate fields:

- `code`: Stable mnemonic code, such as `PC-PORT-RET`.
- `severity`: Informational, warning, material, or error.
- `confidence`: High, medium, low.
- `snapshot_a_value`: Snapshot A value.
- `snapshot_b_value`: Snapshot B value.
- `delta_b_minus_a`: Numeric difference where applicable, calculated as
  snapshot B minus snapshot A.
- `portfolio_id`: Optional portfolio context.
- `security_id`: Optional security context.
- `from_date`: Optional period start.
- `thru_date`: Optional period end.
- `source_file`: Configured source file associated with the finding.
- `source_record_locator`: Stable identifier derived from the normalized
  logical record key. It does not depend on the record's physical CSV row
  number.
- `source_column`: Source column associated with the finding.
- `message`: Human-readable explanation.
- `suppressed`: Whether a suppression rule hid the finding from normal output.

## Explanation Model

The comparison has two levels of performance delta that may need explanation:

- `portfolio_period`: Portfolio performance for one portfolio and period. This
  is the required top-level explanation target because `portfolio_performance`
  is the only required dataset.
- `security_period`: Security performance for one portfolio, security, and
  period. This is an optional secondary explanation target when
  `security_performance` is available.

Security performance deltas should not be treated as root causes of portfolio
performance deltas. They are related output deltas: useful context, but not the
underlying input change that caused portfolio performance to move.

Root-cause evidence should come from input/source-like datasets such as
holdings, FX rates, transactions, market values, accruals, income, and
other source fields. Security master and classification changes can provide
useful context, but are often not numeric causes by themselves.

The first explanation layer should use the term `related evidence`, not
`root cause`, until the system can calculate cause contribution amounts. A
future contribution-ranking model can estimate how much each root-cause
evidence item explains of the portfolio-period return delta.

The first bridge object is `portfolio_period_summary`: one row per portfolio
return delta with counts of related findings, flags for suppressed findings,
and numeric deltas where they are already meaningful. This bridges raw findings
to future reports without pretending to perform full return attribution.

The current implementation also includes conservative explanation helpers for
evidence breakdowns, review-priority rankings, contribution candidates, coarse
cause summaries, transaction activity summaries, and optional security-period
evidence views. These helpers organize evidence; they still avoid claiming
full causal attribution.

### Evidence Role

Findings distinguish their role in the explanation model using the
`evidence_role` column. This prevents contextual evidence from being mistaken
for a direct time-weighted return driver.

Evidence roles:

- `target_output`: The performance delta being explained, such as a portfolio
  return change.
- `related_output`: Calculated output deltas that help locate the change, such
  as security return changes.
- `direct_input`: Source/input changes that can plausibly drive time-weighted
  return, such as holding price values, FX rates, flows, market values, holdings,
  accruals, transaction amounts, and cash balances.
- `context`: Reference, classification, schema, or accounting context that aids
  investigation but is not a direct performance driver by itself.

The portfolio-period summary and evidence breakdown count roles from
`evidence_role`, while still reporting dataset-level counts for familiar
review workflows.

The stored `evidence_role` is global and portfolio-period oriented. Because
portfolio performance is the only required top-level target, security
performance deltas are stored as `related_output` in the findings table. In the
local `security_period_evidence_breakdown()` helper, a `PC-SEC-RET` finding is
displayed as the security-period `target_output`. This is a presentation choice
for the local security-period view, not a change to the underlying finding
record.

### Dataset Roles

The first-pass role model should remain intentionally small:

- `portfolio_performance`: `target_output` for portfolio return changes;
  `base_currency` is metadata rather than a separately compared value.
- `security_performance`: `related_output` in the global portfolio-period
  model. In a local security-period view, the security return change is the
  local `target_output`.
- `fx_rates`: `input_component` because a rate change explains a translated
  base value; the changed base market value or base transaction amount is the
  direct input counted in performance attribution.
- `transactions`: `direct_input` because activity, cash flow, quantity, price,
  and amount changes can drive performance inputs.
- `holdings`: `direct_input` because quantity, market value, price, and
  accrued-balance changes can drive performance inputs.

## Finding Codes

Initial code family:

```text
PC-SCHEMA       Schema changed between snapshots
PC-FILE-MISS    Expected file missing from one snapshot
PC-ROW-ADD      Row exists only in snapshot B
PC-ROW-DROP     Row exists only in snapshot A
PC-PORT-RET     Portfolio return changed
PC-PORT-MV      Portfolio market value changed
PC-PORT-FLOW    Portfolio flow/income/gain-loss changed
PC-SEC-RET      Security return changed
PC-SEC-WGT      Security weight changed
PC-SEC-CONTR    Security contribution changed
PC-SEC-ADD      Security appears only in snapshot B
PC-SEC-DROP     Security appears only in snapshot A
PC-PRICE        Price changed
PC-FX-RATE      FX rate changed
PC-TXN-ADD      Transaction appears only in snapshot B
PC-TXN-DROP     Transaction appears only in snapshot A
PC-TXN-AMT      Transaction amount changed
PC-TXN-QTY      Transaction quantity changed
PC-TXN-PRICE    Transaction price changed
PC-HOLD-QTY      Holding quantity changed
PC-HOLD-MV       Holding market value changed
PC-HOLD-ACCR     Holding accrued amount changed
PC-REF-ID       Security identifier/reference field changed
PC-REF-CLASS    Security classification changed
PC-RESIDUAL     Unexplained residual remains
```

Codes should be stable once public. New detail can be added through fields
rather than by renaming codes. Canonical codes should be stored and displayed
in uppercase, such as `PC-PORT-RET`. Configuration sections, including
suppressions, can accept case-insensitive code input by normalizing configured
values to uppercase at the boundary.

## Configuration

The user-facing Audit starter keeps its shared source-column mappings and Audit
run settings together in `ppar.yaml`. One file points at both snapshot
directories, maps their common CSV layout, and defines the comparison policies.
This follows the same discoverable configuration pattern as Analytics and avoids
making ordinary users coordinate a second mapping file.

An advanced comparison whose snapshots use different source layouts can still
reference a separate schema YAML from each `snapshots.*.schema` path. A
referenced snapshot schema is a complete override for that snapshot rather than
a partial overlay on the common mappings in `ppar.yaml`.

Audit retains its own exact normalization layer and is not Axys/APX-only.
Mappings can describe any supported source layout, including FX rates,
transactions, holdings, and splits.

Composite Axys/APX security identity uses normalized mapping keys, not literal
CSV captions. A direct `files.*.columns.security_id` mapping takes precedence
over automatic inference. When there is no direct mapping and a dataset layout
maps both `security_type` and `security_symbol`, PPAR constructs the compact
identity from those fields in that order with no separator. An advanced
`security_id` section may override
the components or separator. The component columns are temporary normalization
inputs: PPAR constructs the existing `security_id` and does not add the
components to reports or supporting CSV schemas. In
`files.transactions.columns`, identity `security_type` remains distinct from
transaction-context
`transaction_security_type`, which is the field available to conditional
transaction rules and native Data Issues filters.

The comparison YAML should keep vendor-specific parameters minimal. Prefer
shared, source-agnostic sections for files, tolerances, materiality, and
suppressions. Use vendor-specific schema sections only when inference is
insufficient or when the two snapshots have different schemas.

Financially meaningful execution choices are explicit rather than inherited
from Python. The user-facing `ppar audit` orchestrator explicitly selects
`portfolio` and then `security` when the required security-performance files are
available; the starter YAML therefore does not choose one primary level.
Lower-level single-view calls must supply a comparison level directly or retain
an explicit `comparison.level` in their specialized YAML. Extract-contract
omission uses the packaged contract, ambiguous-flow enforcement, and exact-case
matching. All six comparison tolerances remain mandatory. When
`transactions`, `holdings`, or `fx_rates` is configured, the corresponding
transaction, holding/price, or FX impact-policy block must be complete.

Snapshot mappings support `label`, `path`, and an optional external `schema`
path. The former snapshot `vendor` setting was removed because it selected no
adapter or behavior. Source-column interpretation comes from the shared mapping
sections or an explicit external snapshot schema, while financial meaning comes
from the extract contract and policy sections. Inline `snapshots.*.schema`
mappings are rejected; they were never an executable mapping surface.

Standard Audit filenames are defaults for datasets required by the explicitly
selected report or configured feature. They retain normal missing-file
validation when their `files.*` key is omitted. Genuinely optional evidence
remains explicitly configured: merely placing a standard-named file in both
snapshots must not expand findings or accounting-policy requirements. Explicit
`files.*` paths always take precedence.

See [Axys/APX Common-Core Export Reference](../axys_apx/axys_apx_common_core_export.md) for an
operational Axys/APX export template and starter field-reference tables. Those
tables are guidance only; explicit local schema mappings remain authoritative.

### YAML Locations And Path Resolution

Configuration files should not be required to live inside snapshot
directories. Snapshot directories are data captures; YAML files are reusable
configuration and may be stored beside scripts, in a `comparisons/` directory,
or anywhere else convenient.

Path resolution should be predictable:

1. Absolute paths are accepted as-is.
2. Relative paths in `ppar.yaml` resolve relative to that Audit YAML file.
3. Snapshot data files resolve relative to the configured snapshot directory.
4. Relative paths inside a referenced external schema resolve relative to that
   schema YAML file.

A typical project layout is:

```text
audit/
  ppar.yaml
  snapshot_a/
    portperf.csv
    secperf.csv
  snapshot_b/
    portperf.csv
    secperf.csv
```

Example:

```yaml
comparison:
  name: May restatement review
  level: portfolio

snapshots:
  a:
    label: run_2026_05_01
    path: snapshots/2026-05-01

  b:
    label: run_2026_05_15
    path: snapshots/2026-05-15

files:
  portfolio_performance:
    path: portperf.csv
    columns:
      portfolio_code: Portfolio Code
      from_date: From Date
      thru_date: Thru Date
      portfolio_return: Portfolio Return
  security_performance:
    path: secperf.csv
    columns:
      portfolio_code: Portfolio Code
      security_symbol: Security Symbol
      security_type: Security Type
      from_date: From Date
      thru_date: Thru Date
      security_return: Security Return
  fx_rates:
    path: fx_rates.csv
  transactions:
    path: transactions.csv
    required: true
  holdings:
    path: holdings.csv

extract_contract:
  enforce_ambiguous_axys_flows: true
  transaction_semantics_case: legacy_case_insensitive

transaction_impact_methods:
  external_flow:
    method: evidence_only
  performance:
    method: transaction_amount_delta_over_return_denominator
    denominator_source: begin_market_value
  quantity:
    method: evidence_only
  price:
    method: evidence_only
  commission:
    method: evidence_only

holding_impact_methods:
  market_value:
    method: market_value_delta_over_return_denominator
    denominator_source: begin_market_value
  accrued:
    method: accrued_delta_over_return_denominator
    denominator_source: begin_market_value
  quantity:
    method: quantity_delta_times_snapshot_a_unit_market_value_over_return_denominator
    denominator_source: begin_market_value
  cost:
    method: evidence_only

price_impact_methods:
  price:
    method: price_delta_over_snapshot_a_price_times_weight
    weight_source: snapshot_a_weight

fx_rate_impact_methods:
  fx_rate:
    method: evidence_only

tolerances:
  return: 0.000001
  market_value: 0.01
  quantity: 0.000001
  price: 0.000001
  split_factor: 0.00000001
  fx_rate: 0.00000001

materiality:
  minimum_return_delta: 0.000001
  minimum_market_value_delta: 0.01

suppressions:
  - code: pc-sec-ret
    portfolio_id: PORT_SMALL
    security_id: CASHUSD
    thru_date: 2024-12-31
    reason: Known cash restatement below audit scope.
```

If each snapshot has its own layout, use separate external schema files:

```yaml
snapshots:
  a:
    label: run_2026_05_01
    path: snapshots/2026-05-01
    schema: schemas/snapshot_a.yaml

  b:
    label: run_2026_05_15
    path: snapshots/2026-05-15
    schema: schemas/snapshot_b.yaml
```

Each referenced file uses the same nested `files.<dataset>.columns` mappings as
the common Audit configuration.

### Vendor Preset Design

Vendor presets are a future convenience layer, not a replacement for explicit
comparison YAML. The first likely preset is Axys/APX, but the design should support
multiple vendors and possibly site-specific presets over time.

A preset keyword such as `vendor: axys` should mean "start with ppar's
versioned Axys/APX preset semantics" rather than "assume all Axys/APX installations
behave identically." The Axys/APX preset seed is now the accepted packaged Axys/APX
demo YAML semantics, but preset implementation is deliberately parked until the
project chooses that lane.

Suggested shape:

```yaml
comparison:
  name: May restatement review

vendor: axys

vendor_preset:
  name: axys
  version: packaged-demo-2026-07

files:
  portfolio_performance: portperf.csv
  security_performance: secperf.csv
  transactions: transactions.csv

transaction_rules:
  # Site YAML can override or add rules after the preset is expanded.
```

Preset resolution should be deterministic:

```text
engine defaults < vendor preset < site YAML overrides
```

The resolved configuration must be inspectable. `validate_config` or a future
CLI option such as `--print-resolved-config` should be able to show the
effective YAML after preset expansion, including which rules came from the
engine, the vendor preset, and the site YAML. This keeps the feature auditable
and avoids hidden policy behavior.

Preset design guardrails:

- Presets are design-only until explicit implementation work is approved.
- Presets must be versioned and tied to a documented source contract.
- Presets must support multiple vendors without hard-coding Axys/APX assumptions
  into source-agnostic comparison logic.
- Site YAML must be able to override, suppress, or extend preset rules with
  deterministic precedence.
- Presets must not bypass complete-YAML validation. Changed fields still need
  additive, evidence-only, or suppression treatment after expansion.
- Presets must not weaken ambiguous transaction-code safeguards. For Axys/APX-style
  `dp`, `li`, `lo`, and `wd` rows, required source/destination or
  special-security context must still be present unless a site explicitly uses
  a documented local extract contract.
- The report bundle should record the preset name/version and whether site
  overrides were applied, so reviewers know which policy layer produced the
  resolved rules.

This layer is intentionally later than the current Axys/APX demo hardening. The
Axys/APX demo is accepted as the preset seed, but implementation should remain
parked until the project deliberately chooses vendor-preset infrastructure as
the next product lane.

### Column Mapping Defaults

Source-column mappings are explicit whenever a vendor or site heading differs
from PPAR's normalized field name. Omitting a mapping accepts only the exact
normalized name, including case; PPAR does not guess vendor aliases.

Column mappings should resolve in this order:

1. Complete external schema referenced by that snapshot, when configured.
2. Common `files.<dataset>.columns` mapping in the Audit `ppar.yaml` when no
   external snapshot schema is configured.
3. Exact normalized field name when the effective mapping omits that field.
4. Error when a required column is missing.

This makes a source contract reviewable and prevents generic headings such as
`DATE`, `ID`, `TYPE`, or `VALUE` from being assigned a meaning implicitly.

The current implementation honors both common Audit mappings and explicitly
referenced external schemas for every supported dataset. A mapped source caption
is authoritative. Exact normalized names are the only fallback for fields not
listed in the effective mapping.

## Suppression And Filtering

Suppression rules are explicit exact-match rules. They do not delete findings;
they mark findings as `suppressed=True` so a full audit output can still show
what was hidden from active output.

First-pass suppression fields:

- `code`: Required finding code. Configured values are normalized to uppercase.
- `dataset`: Optional normalized dataset name.
- `portfolio_id`: Optional exact portfolio identifier.
- `security_id`: Optional exact security identifier.
- `from_date`: Optional exact period start date.
- `thru_date`: Optional exact period end date.
- `source_column`: Optional normalized source column name.
- `reason`: Optional informational explanation for the suppression.

Unsupported suppression keys should fail validation instead of being silently
ignored. This keeps configuration mistakes visible.

The public runner keeps the audit trail by default:

```python
compare_snapshots(path)  # Includes suppressed findings.
compare_snapshots(path, include_suppressed=False)  # Active findings only.
```

Finding summaries should include suppression-aware counts:

- finding count by code
- finding count by dataset
- finding count by suppression state
- finding count by code and suppression state

## Public API And Output Layers

The current public runner layer exposes small helpers:

```python
findings = compare_snapshots(path)
summaries = summarize_findings(findings)
compact = compact_findings_table(findings)
periods = portfolio_period_summary(findings)
security_periods = security_period_summary(findings)
evidence = portfolio_period_evidence_breakdown(findings)
ranked_evidence = rank_portfolio_period_evidence(findings)
contribution_candidates = portfolio_period_contribution_candidates(findings)
cause_summary = portfolio_period_cause_summary(findings)
transaction_summary = transaction_activity_summary(findings)
security_evidence = security_period_evidence_breakdown(findings)
```

Current module ownership:

- `runner.py` owns execution-facing helpers: snapshot comparison, compact
  findings, and finding-count summaries.
- `explain.py` owns explanation-facing table helpers: portfolio/security
  period summaries, evidence breakdowns, evidence rankings, contribution
  candidates, cause summaries, and transaction summary tables.
- `_transaction_diagnostics.py` owns transaction-diagnostic presentation helpers
  used by `explain.py`; it does not build Polars output tables.
- `__init__.py` re-exports the stable public helpers so callers do not need to
  care which internal module owns the implementation.
- `ppar.audit` exposes selected workflow and report entrypoints; explanation
  helpers are owned by the Performance Comparison sub-feature.

Current output layers:

- Full audit findings: The complete findings table returned by
  `compare_snapshots()`. It includes all finding columns and includes
  suppressed findings by default.
- Compact active findings: A report-friendly subset returned by
  `compact_findings_table()`. It excludes suppressed findings by default and
  keeps the most useful review columns: code, dataset, evidence role,
  portfolio/security context, period dates, source file, source column, delta,
  and message.
- Summaries: Count tables returned by `summarize_findings()` for code, dataset,
  evidence role, suppression state, and code plus suppression state.
- Portfolio-period summary: A lightweight explanation bridge returned by
  `portfolio_period_summary()`. It groups existing findings around portfolio
  return deltas and reports related evidence counts without claiming causal
  contribution yet. It includes role counts for direct input findings, related
  output findings, and contextual findings.
- Security-period summary: A lightweight optional explanation bridge returned
  by `security_period_summary()`. It groups existing findings around security
  return deltas when `security_performance` is available, and returns a stable
  empty table when it is not.
- Portfolio-period evidence breakdown: A long-form explanation bridge returned
  by `portfolio_period_evidence_breakdown()`. It reports role total rows and
  nonzero dataset rows for each portfolio-period return delta, making the
  evidence mix easier to inspect than a single wide row.
- Portfolio-period evidence ranking: A review-priority helper returned by
  `rank_portfolio_period_evidence()`. It ranks related non-target findings for
  each portfolio-period return delta using a transparent heuristic based on
  evidence role, dataset, and whether a numeric delta exists. The score is not
  a contribution amount or explained return.
- Portfolio-period contribution candidates: A conservative helper returned by
  `portfolio_period_contribution_candidates()`. It adds stable impact columns
  to ranked evidence rows and may return `no_estimate` when a defensible impact
  estimate is not available. Supported estimates use explicit YAML methods and
  holdings-, transaction-, FX-, or reconstruction-derived inputs.
- Portfolio-period cause summary: A coarse explanation-bucket helper returned
  by `portfolio_period_cause_summary()`. It rolls contribution candidates up
  to portfolio period plus root-cause area, preserving finding counts, top
  codes, aggregate confidence, and any currently supported impact estimate.
  Transaction activity remains evidence-only until transaction-type sign and
  flow semantics are explicitly modeled.
- Portfolio-period impact coverage summary: A transparency helper returned by
  `portfolio_period_impact_coverage_summary()`. It counts, by changed
  portfolio period, how many cause areas currently have return-impact
  estimates, how many remain evidence-only, and which missing-input themes are
  blocking the evidence-only areas. It also emits a reviewer-facing
  `impact_coverage_status` and `impact_coverage_review_note`. The coverage
  total sums already-selected cause-area estimates for review context; it is
  not a residual calculation or a complete attribution statement.
- Transaction activity summary: An evidence-only helper returned by
  `transaction_activity_summary()`. It groups changed transaction fields by
  portfolio, security, period, and normalized transaction category, and reports
  summed amount, quantity, and price deltas where present. It does not estimate
  return impact.
- Security-period evidence breakdown: A long-form optional explanation bridge
  returned by `security_period_evidence_breakdown()`. It reports role total
  rows and nonzero dataset rows for each security-period return delta. In this
  local security-period view, the security return finding is treated as the
  `target_output`.

Portfolio performance is the authoritative source of portfolio base currency.
When it is available, holdings and transactions inherit a missing row-level
base currency from the portfolio and fail validation if they contradict it.

Normalized monetary names use one currency-basis rule:

- unqualified monetary fields in holdings and transactions use the row
  `currency`;
- `base_` monetary fields in those detailed datasets use portfolio
  `base_currency`;
- `fx_rates.fx_rate` is `to_currency` units per one `from_currency` unit.

PPAR does not add `local_` duplicates because row currency is already the
detailed-data default. A detailed unqualified value may enter Modified Dietz
without translation only when row and base currencies are equal (or when a
legacy single-currency extract omits both currency fields). A nonzero foreign
holding market value, accrued balance, or transaction amount must provide
`base_market_value`, `base_accrued`, or `base_amount`. Missing translation
fails reconstruction rather than silently
treating local currency as base currency.

This is intentionally not a final report format. It gives callers stable
building blocks for CSV, HTML, XLSX, or portfolio-period bridge reports without
committing the project to presentation details too early.

## Evidence Linking

Some source-input findings can be linked to a portfolio period before any
return-impact estimate is available. This linking improves review grouping; it
does not imply that the system has calculated causal contribution.

Implemented period-linking rules:

- `transactions` and `holdings` findings link directly to portfolio periods by
  `portfolio_id` plus `transaction_date` or `holding_date`, respectively.
- When more than one configured portfolio period contains the source date, the
  finding links to the narrowest containing period for that portfolio.
- Unmatched dated evidence keeps null period fields.
- Portfolio-specific `fx_rates` rows link by `portfolio_id` and `rate_date`.
  They support an estimate only when both snapshots provide the same explicit
  `local_exposure`; a rate row without that exposure remains review evidence.

These rules are intentionally asymmetric. Holding price rows use reconstructed
beginning holdings to derive a portfolio weight. FX rates need an explicit
portfolio identifier and local exposure; PPAR does not infer either from a
performance file.

## Current Limits

The current evidence model is useful, but it should not be overstated.

- Evidence counts are not contribution amounts. Supported YAML methods may
  calculate an estimate, but unestimated evidence must remain visibly distinct.
- Portfolio-period evidence rankings are review-priority heuristics. They help
  sort the audit trail but do not quantify causal contribution.
- Price estimates require holdings-derived beginning weights. FX rates without
  explicit portfolio and exposure context remain unlinked and unestimated.
- Transaction matching depends on stable keys. With `transaction_id`, changed
  amounts can be reported as changed transactions. Without it, conservative
  fallback matching may report one drop and one add rather than guessing two
  similar rows are the same transaction.
- Changed transaction and holding findings are linked to the narrowest
  configured portfolio performance period for the same portfolio when their
  source date falls inside that period. Unmatched dated evidence findings keep
  null period fields.
- Security master changes are context unless future logic ties them to a
  grouping, reporting, or identifier-resolution effect.
- Security-period summaries are optional. The portfolio-period explanation path
  must continue to work when `security_performance` is absent.
- The implementation reconstructs Modified Dietz inputs from holdings and
  transactions. It does not treat optional performance-file output components
  as independent evidence.

## Explanation Estimates And Performance-File Boundary

Audit treats the performance files as reported results, not as a second source
of holdings, flows, income, gain/loss, contribution, or valuation evidence.

The normalized performance-file contract is intentionally narrow:

- `portfolio_performance`: portfolio identifier, period dates, reported
  portfolio return, and optional base-currency metadata;
- `security_performance`: portfolio/security identifiers, period dates, and
  reported security return.

Unmapped extra CSV columns are ignored. Audit does not compare or separately
interpret beginning value, ending value, net flow, income, gain/loss, beginning
weight, or contribution fields from these files. This keeps the user-facing
extract requirement clear and leaves room for a future evidence field only
after that field receives an explicit product contract.

`rank_portfolio_period_evidence()` remains a review-priority sort rather than
a contribution model. The explanation layer may attach a conservative
`estimated_return_impact` only when YAML selects a supported method and all
required reconstructed inputs are present. Otherwise the row remains visible
with a null estimate and explicit missing-input or review-only guidance.

Current estimate inputs are derived as follows:

1. Portfolio denominators use `derived_denominator_a` from the configured
   portfolio return reconstruction.
2. Security denominators use `derived_denominator_a` from the configured
   security return reconstruction.
3. Portfolio-level security weights use the security beginning value divided by
   the portfolio beginning value from those same reconstruction checks.
4. Holding market-value, accrued, and quantity estimates use their configured
   holding impact methods.
5. Holding price estimates use the configured price method and the
   holdings-derived beginning weight.
6. Performance-treated transaction amounts, eligible security transaction
   flows, and configured FX-rate exposure estimates use their explicit YAML
   methods and reconstructed denominators.

Missing reconstruction setup, missing boundary holdings, zero denominators, or
missing transaction semantics fail closed: Audit does not fall back to optional
performance-file columns and does not invent an estimate. Review-only Modified
Dietz external-flow cross-checks remain separate from additive explanation
totals.

The retired `contribution_impact_methods` key is rejected so an old or
misspelled setup cannot silently restore performance-file-derived estimates.
The retired portfolio/security market-value continuity issue types are also not
part of the Data Issues vocabulary; holdings and transaction reconstruction are
the authoritative valuation and flow evidence.

## Report Bundle Contract

The public bundle starts with the level-specific XLSX or HTML report and keeps
the canonical CSV counterparts for parity and automation. The core supporting
artifacts include findings, source detail, performance differences, performance
difference causes, Data Issues, cause lineage, impact coverage, context
evidence, transaction activity, transaction cross-checks, transaction matching
diagnostics, residual status, and top evidence.

`source_detail.csv` is always written at the report root. Other supporting
artifacts are archived by default and are expanded under `supporting_files/`
when requested. The bundle manifest records ordered columns, row counts,
semantic fingerprints, source context, transaction-semantics summaries, and
review entrypoints. Reconstruction diagnostics remain opt-in report artifacts,
while the inexpensive financial and explanation invariants remain active in
normal production runs.

## Long-Term Dataset Watchlist

The current normalized surface—portfolio performance, security performance,
holdings, transactions, and FX rates—remains the implementation boundary.
Additional datasets require real source evidence and an approved product need;
this document does not maintain a speculative dataset catalog.

## Open Design Issues

Current MVP decisions belong in the MVP plan. Other technical questions should
be recorded only when repository evidence or active implementation makes them
actionable. Historical question lists remain available in Git history and must
not be interpreted as current commitments.

## Current Roadmap Location

Treat the current comparison engine, evidence layer, report bundle, return
reconstruction checks, and workbook model as the baseline. Current Audit work
belongs in the
[`PPAR Audit MVP Completion Plan`](mvp_plan.md), and
product direction belongs in the
[`PPAR Audit Product Constitution`](product_constitution.md) and
[`PPAR Audit Roadmap`](roadmap.md),
not here as a competing roadmap.
