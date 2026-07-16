# Performance Comparison Design Notes

Performance Comparison is the attribution sub-feature inside the broader
Performance Auditing workflow. The public workflow asks whether reported
performance changed and whether source-data relationships look suspicious. This
design note focuses on the internal comparison engine that explains changed
reported performance between two source-data snapshots.

## Purpose

The performance comparison feature explains why calculated performance for the
same portfolio and period changed between two source-data extraction dates.

This document is the deep design/reference note. Active forward-looking work is
tracked in the central
[`PPAR Roadmap`](roadmap.md).
The maintainer-facing safety guarantees and their audited enforcement baseline
are defined in
[`Performance Auditing Safety Invariants`](performance_comparison_safety_invariants.md).

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

## Current Checkpoint

The current implementation has crossed from pure design into a usable
comparison, explanation, and report checkpoint. It can load two snapshot
directories, compare the first set of normalized datasets, emit stable finding
records, apply explicit suppressions, and produce reviewer-oriented tables,
HTML, XLSX, CSV, and handoff bundles.

Implemented normalized comparison datasets:

- `portfolio_performance`
- `security_performance`
- `fx_rates`
- `transactions`
- `holdings`

Implemented output helpers:

- full audit findings
- compact active findings
- finding summaries by code, dataset, evidence role, and suppression state
- portfolio-period summaries
- portfolio-period evidence breakdowns
- portfolio-period evidence rankings
- portfolio-period contribution candidates
- portfolio-period cause summaries
- transaction activity summaries
- context evidence summaries and context evidence detail
- security-period summaries
- security-period evidence breakdowns
- HTML and XLSX review reports
- reproducible report bundles with manifest and validation helpers

User-facing entry point:

- `ppar setup`: creates an Axys/APX starter workspace.
- `ppar audit`: writes portfolio and security comparison report
  bundles from a configured workspace.

Developer/internal helper commands:

- `scripts/check_performance_comparison_demo_health.py`: source-checkout smoke
  command that runs `ppar setup`, executes the setup-generated portfolio and
  security scripts, and validates the generated bundles.
- `ppar.performance_comparison.cli.report_bundle`: source-checkout command
  for writing a report bundle from a comparison YAML file.

Current workbook field roles:

- `performance_input`: source fields that directly feed return calculation,
  such as `holdings.market_value`, `holdings.base_market_value`,
  `holdings.accrued`, and
  `transactions.amount`. These can receive `Performance Difference Explained`
  values when the required denominator or weight inputs are available.
- `input_component`: fields that explain or reconcile a performance input, such
  as holding quantity/price, FX rates, and transaction
  quantity/price/commission. Holding quantity can appear beside a related
  holding market-value input. Transaction quantity, price, and commission may
  appear beside a related `transactions.amount` row, while non-promoted support
  remains in `supporting_files/source_detail.csv`, so transaction arithmetic is not
  double-counted or treated as a rebuilt accounting-system formula.
- `reported_performance_component`: portfolio/security performance output
  fields such as return, income, gain/loss, contribution, weight, and market
  value. These remain reporting diagnostics and are not treated as underlying
  causes.
- `context`: explicitly classified review-only supporting fields such as
  holding cost and security reference fields. These remain in the
  `supporting_files/source_detail.csv` unless they are direct inputs to a supported performance
  explanation.

Unknown compared fields do not default to context. They stop processing until
their accounting role is explicitly classified; suppression cannot bypass that
decision. YAML impact-policy requirements are derived from these roles rather
than maintained in a second field-name list.

Cash has one normalized representation: a holding such as `CASHUSD`, `CASHEUR`,
or `CASHGBP`. A source-specific adapter may convert a cash-ledger export into
holding rows, but `cash` is not a separate comparison dataset. This prevents
two source files from claiming the same beginning or ending valuation effect.

Financial-input integrity is fail-closed. Currency codes are normalized and
shape-validated; foreign countable monetary values require explicit base-value
counterparts; same-currency local/base values must agree; portfolio FX quotes
must target the portfolio base currency; and performance periods may not be
reversed or overlap. Changed dated evidence is audited for unambiguous period
assignment. Historical/carry-forward evidence may stay visible, but only
in-period transaction/split rows and explicit prior-day beginning holdings/FX
rows may own explained performance.

Current report vocabulary:

- `Performance Differences`: the main review sheet. It compares reported
  performance between Snapshot A and Snapshot B, shows the explained portion,
  and leaves `Unexplained Difference` blank when a row is fully explained.
- `Performance Difference Causes`: source-data rows that are counted in
  `Explained Difference`.
- `supporting_files/findings.csv`: complete lossless finding audit trail,
  including suppressed rows, stable logical source-record locators, and
  explicit safety dispositions.
- `supporting_files/cause_lineage.csv`: internal cause rows with backward
  finding fingerprints, stable locators, lineage type, economic-effect ID, and
  counted-owner disposition. It supports integration and invariant validation;
  it is not another reviewer-facing workbook sheet.
- `supporting_files/source_detail.csv`: reviewer-friendly active finding rows
  used for audit and troubleshooting.
- `transaction_matching_diagnostics.csv`: supplementary transaction row-identity
  counts, confidence, interpretation, and review notes for audit use.
- `Transaction Code`: the source transaction code from the input file, such as
  an Axys/APX IMEX code.
- `Transaction Category`: ppar's normalized interpretation of a transaction code
  for comparison logic and reviewer explanations.

Report construction enforces this vocabulary as an arithmetic invariant. At
portfolio and security grain, the sum of `Performance Difference Explained` in
the causes table must equal `Explained Difference` in the main table. A `Fully
Explained` result must also have equal performance and explained differences
and zero unexplained difference. The checks run before workbook construction
and again on the serialized six-decimal workbook cells; a mismatch stops report
generation as an unexpected logic error.

Future workbook vocabulary:

- `Data Audit Issues`: consistency issues found inside the union of Snapshot A and
  Snapshot B. These rows are not additive Modified Dietz causes and are not
  `source_detail.csv` findings. They answer a different reviewer question:
  "Which source-data relationships look internally inconsistent?"

- `ppar.performance_comparison.cli.validate_bundle`: source-checkout command
  for validating an existing report bundle.
- `ppar.performance_comparison.cli.validate_config`: source-checkout command
  for validating a comparison YAML file and its default report-readiness
  guardrails.
- `ppar.performance_comparison.cli.validate_demo_matrix`: source-checkout
  command for validating packaged scenario coverage.
- [Axys/APX Common-Core Export Reference](axysapx_common_core_export.md): starter
  export shape for Axys/APX-oriented source-data.

This checkpoint is still a comparison and evidence organization layer. It is
not yet a causal attribution engine or a full return calculator. The report
layer is intentionally conservative: it presents evidence, review cues, and
documented estimates without claiming more precision than the current model
supports.

## Supported Vocabulary

YAML files and report artifacts use plain string values so they remain easy to
read, diff, and edit. The package code centralizes the same values in
`StrEnum` classes to reduce drift and give internal APIs stricter annotations
where practical. Existing public constants remain as compatibility aliases for
the enum members.

Public YAML impact method values are centralized in:

- `ContributionImpactMethod`: `source_field_delta_over_begin_market_value`,
  `vendor_contribution_delta`, and `security_return_delta_times_weight`.
- `TransactionImpactMethod`: `evidence_only`, `modified_dietz`, and
  `transaction_amount_delta_over_return_denominator`.
- `HoldingImpactMethod`: `evidence_only`,
  `quantity_delta_times_snapshot_a_unit_market_value_over_return_denominator`,
  `market_value_delta_over_return_denominator`, and
  `accrued_delta_over_return_denominator`.
- `PriceImpactMethod`: `price_delta_over_snapshot_a_price_times_weight`.
- `FxRateImpactMethod`: `evidence_only`.
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

When a value crosses into Polars tables, CSVs, HTML, XLSX, or YAML, it
should be serialized as its plain string value. Enum members are primarily for
construction, validation, and package-internal type clarity.

## Data Audit Issues Worksheet Design

The `Data Audit Issues` worksheet surfaces source-data consistency problems that
are useful to reviewers but are not themselves performance attribution rows.
It is the user-facing Data Auditing surface inside Performance Comparison. The
internal YAML key remains `data_audit_checks` because these checks cross-reference
related source-data fields.

Purpose:

> Flag source-data relationships that should usually agree, reconcile, or move
> together, without treating those issues as additive Modified Dietz causes.

This worksheet is deliberately separate from both primary evidence sheets:

- `Performance Difference Causes` explains changed performance.
- `source_detail.csv` records row-level A-versus-B differences.
- `Data Audit Issues` reports consistency checks across source-data relationships,
  whether or not those checks explain a performance difference.

Beginning/ending market-value continuity is a mandatory financial-integrity
check at portfolio and security grain. A mismatch remains visible in this
worksheet even when optional Data Audit checks are disabled because both values
participate directly in return calculations.

Checks should run on the union of Snapshot A and Snapshot B. A reviewer should
be able to see an issue that appears only in Snapshot A, only in Snapshot B, or
in both snapshots. Include a `Snapshot` column so the report can distinguish
`Snapshot A`, `Snapshot B`, and, when useful, `A and B`.

Worksheet columns:

```text
Snapshot
Portfolio
As Of Date
Dataset Field
Security
Issue Type
Reference Value
Observed Value
Difference
Tolerance
Explanation
Review Key
```

### YAML Configuration

Consistency checks need YAML tolerances to avoid noisy output. Price checks are
the first place where this matters. Transaction prices can vary widely intraday,
while holdings prices are usually as-of or closing prices and should normally
have much narrower tolerance.

YAML shape:

```yaml
data_audit_checks:
  enabled: true

  dividend_rate:
    enabled: true
    only:
      security_type: stock
    exclude:
      portfolio_id:
        - TEST_PORTFOLIO
    absolute_tolerance: 0.01
    percent_tolerance: 0.50
```

Interpretation:

- `data_audit_checks.enabled`: master switch for optional consistency checks;
  mandatory beginning/ending continuity findings remain active.
- Each issue type is enabled by default when the worksheet is enabled. Set
  `enabled: false` under one issue type to opt out of that check.
- `only`: optional exact-match include filters. A row must match every listed
  field to enter the check.
- `exclude`: optional exact-match exclude filters. A row is dropped when it
  matches any listed field.
- Filter values can be scalars or lists. Field names may be common normalized
  names such as `security_id`, `security_type`, and `portfolio_id`, or
  dataset-qualified names such as `holdings.security_type` and
  `transactions.transaction_code`.
- Tolerances stay per issue type because noisy fields need different limits.

### Initial Checks

Implemented checks should stay high signal, available from the current
normalized datasets, and easy to explain:

1. `holdings_price_range`: for each snapshot, security, and holding date,
   compare same-day same-security `holdings.price` values across portfolios.
2. `transactions_price_range`: for each snapshot, security, and transaction
   date, compare same-day same-security `transactions.price` values across
   portfolios.
3. `duplicate_transactions`: for each snapshot, flag exact duplicate
   transaction rows with the same portfolio, date, security, code, amount,
   quantity, and price.
4. `dividend_rate`: for each snapshot, security, and dividend date, compare
   same-day same-security dividend rates across portfolios.
5. `missing_dividend`: for each snapshot, security, and dividend date where at
   least one portfolio has a dividend, flag other portfolios that
   conservatively appear eligible for that dividend. A portfolio qualifies only
   when it has positive beginning-period quantity, or positive buy activity
   before the dividend date, and has no pre-dividend transaction activity other
   than buys.
6. `pa_sa_rate`: for each snapshot, security, and transaction date, compare
   same-day same-security purchase-accrued and sale-accrued rates across
   portfolios.
7. `holdings_accrued_rate`: for each snapshot, security, and holding date,
   compare same-day same-security `holdings.accrued` per unit across
   portfolios.

Defer checks that require extra reference data or more source-system evidence:

- paydown amount/rate checks;
- bond accrued-interest expectation checks;
- split factor versus quantity-jump plausibility;
- cash roll-forward checks.

Those deferred checks are valuable, but they should not be first because they
can easily imply security-master, dividend-rate, pool-factor, coupon/accrual,
or cash-ledger data that is not yet part of the normalized source contract.

### Report Semantics

Data Audit issue types use compact codes such as `duplicate_transactions` and
`transactions_price_range`. The worksheet intentionally avoids a severity column;
the reviewer-facing explanation, values, and tolerance carry the useful context
without implying false precision.

Do not make Data Audit rows blocking by default. A future YAML option can promote a
specific check to blocking once the project has enough real-world evidence that
the check is stable for a site's source-data.

Data Audit issues should not change:

- `Performance Difference Explained`;
- `Unexplained Difference`;
- `Performance Difference Causes`;
- `source_detail.csv` row counts.

They may be referenced from README/handoff text as a separate review surface.

## Non-Goals For The First Pass

- Recalculate full portfolio performance from transactions and holdings.
- Replace existing `ppar.axys` loading behavior.
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
ppar/performance_comparison/
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

Each normalized dataset should have a small required column set. All other
columns should be optional and preserved when useful.

`portfolio_performance` required columns:

- `portfolio_id`
- `from_date`
- `thru_date`
- `portfolio_return`

`portfolio_performance` useful optional columns:

- `portfolio_name`
- `begin_market_value`
- `end_market_value`
- `flow`
- `income`
- `gain_loss`
- `period_id`
- `currency`

`security_performance` required columns:

- `portfolio_id`
- `security_id`
- `from_date`
- `thru_date`
- `security_return`

`security_performance` useful optional columns:

- `security_name`
- `weight`
- `contribution`
- `begin_market_value`
- `end_market_value`
- `income`
- `gain_loss`
- `ticker`
- `currency`

`weight` means the security weight used to explain or reconcile security
contribution. It may come from beginning weight, average capital weight,
modified-Dietz adjusted weight, or another vendor-provided effective weight.
Method-specific source fields can still be preserved as optional explanatory
columns when useful.

- additional classification code/name pairs

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
- `settlement_date`
- `transaction_code`
- `transaction_category`
- `cash_flow_sign`
- `performance_flow_sign`
- `quantity`
- `price`
- `amount`
- `commission`
- `currency`
- `broker`

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
`sl`, `dv`, `in`, `dp`, `wd`, and `;`. YAML rules map those source codes to
normalized categories such as `buy`, `sell`, `income`, `external_flow`, and
`corporate_action`. The normalized category is an explanation label and a rule
selector; it does not replace the source transaction code in the audit trail.
Transaction-code normalization is limited to semantic classification and
rule-coverage checks. It is not a broad case-insensitive comparison policy for
source identifiers; reviewer handoff artifacts preserve native transaction-code
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

When a source does not provide usable sign semantics, comparison YAML may define
explicit transaction rules keyed by source transaction code:

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

Source-supplied recognized category/sign semantics remain authoritative. YAML
rules fill only missing or `unknown` category, `cash_flow_sign`, and
`performance_flow_sign` values.

Loaded transaction rows also carry `transaction_semantics_source` so reviewers
can audit how the normalized category/sign/flow treatment was obtained:

- `source`: Usable sign/flow semantics came from recognized source fields, with
  any category supplied by source-data or transaction-code inference.
- `yaml_rule`: Usable sign/flow semantics came entirely from `transaction_rules`.
- `mixed`: Some recognized semantics came from the source, while YAML rules
  filled other missing or `unknown` fields.
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
note that explanation depth is limited. Missing optional columns should produce
a less detailed explanation, not a failed comparison.

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
  as security return, weight, or contribution changes.
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
displayed as the security-period `target_output`, while security weight and
contribution changes remain `related_output`. This is a presentation choice for
the local security-period view, not a change to the underlying finding record.

### Dataset Roles

The first-pass role model should remain intentionally small:

- `portfolio_performance`: `target_output` for portfolio return changes, and
  `direct_input` for source fields such as market value, flow, income, and
  gain/loss changes.
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

The existing Axys/APX column mapping configuration and the new comparison
configuration serve different purposes and should have distinct names:

- `axysapx_column_mappings.yaml`: Describes how Axys/APX source columns map to
  normalized internal column names for reusable Axys/APX datasets.
- `performance_comparison.yaml`: Describes which snapshots and files to
  compare, plus comparison tolerances, materiality, and suppressions.

A comparison probably needs one YAML file for the comparison run, not a separate
YAML file inside each snapshot. The comparison YAML can point at both snapshot
directories, define shared rules, and optionally reference vendor schema files
such as `axysapx_column_mappings.yaml`.

The performance comparison feature has its own normalization/default alias
layer. Referencing `axysapx_column_mappings.yaml` is a reuse mechanism for shared
Axys/APX datasets, not a requirement that performance comparison become
Axys/APX-only.
Comparison-only datasets such as FX rates, transactions, holdings,
and cash can use performance-comparison mappings even when the referenced Axys/APX
mapping file does not define them.

The comparison YAML should keep vendor-specific parameters minimal. Prefer
shared, source-agnostic sections for files, tolerances, materiality, and
suppressions. Use vendor-specific schema sections only when inference is
insufficient or when the two snapshots have different schemas.

See [Axys/APX Common-Core Export Reference](axysapx_common_core_export.md) for an
operational Axys/APX export template and starter field-reference tables. Those
tables are guidance only; explicit local schema mappings remain authoritative.

### YAML Locations And Path Resolution

Configuration files should not be required to live inside snapshot
directories. Snapshot directories are data captures; YAML files are reusable
configuration and may be stored beside scripts, in a `comparisons/` directory,
or anywhere else convenient.

Path resolution should be predictable:

1. Absolute paths are accepted as-is.
2. Relative paths in `performance_comparison.yaml` resolve relative to
   that comparison YAML file.
3. Snapshot data files resolve relative to the configured snapshot directory.
4. Relative paths inside a referenced schema YAML, such as
   `axysapx_column_mappings.yaml`, resolve relative to that schema YAML file.

A suggested project layout is:

```text
comparisons/
  performance_comparison.yaml
  axysapx_column_mappings.yaml

snapshots/
  2026-05-01/
    portperf.csv
    secperf.csv

  2026-05-15/
    portperf.csv
    secperf.csv
```

Example:

```yaml
comparison:
  name: May restatement review

snapshots:
  a:
    label: run_2026_05_01
    path: snapshots/2026-05-01
    vendor: axys
    schema: axysapx_column_mappings.yaml

  b:
    label: run_2026_05_15
    path: snapshots/2026-05-15
    vendor: axys
    schema: axysapx_column_mappings.yaml

files:
  portfolio_performance: portperf.csv
  security_performance: secperf.csv
  fx_rates: fx_rates.csv
  transactions:
    path: transactions.csv
    required: true
  holdings: holdings.csv

tolerances:
  return: 0.000001
  contribution: 0.000001
  weight: 0.000001
  market_value: 0.01
  price: 0.000001
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

If each snapshot has its own schema, the comparison YAML should allow
snapshot-specific mappings:

```yaml
snapshots:
  a:
    label: run_2026_05_01
    path: snapshots/2026-05-01
    vendor: axys
    schema:
      portfolio_performance_columns:
        portfolio_code: PORT
        portfolio_return: RETURN

  b:
    label: run_2026_05_15
    path: snapshots/2026-05-15
    vendor: axys
    schema:
      portfolio_performance_columns:
        portfolio_code: PORTFOLIO_CODE
        portfolio_return: PORT_RETURN
```

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

The comparison YAML should use the same defaulting method for column mappings
that the existing Axys/APX YAML uses. Users should not need to specify obvious
column names.

Column mappings should resolve in this order:

1. Snapshot-specific mapping in `performance_comparison.yaml`.
2. Shared comparison-level mapping in `performance_comparison.yaml`.
3. Referenced vendor schema file, such as `axysapx_column_mappings.yaml`.
4. Built-in default aliases.
5. Error when the column is missing or ambiguous.

Built-in aliases should be conservative and dataset-scoped. Generic names such
as `DATE`, `ID`, `TYPE`, and undifferentiated `VALUE` are too ambiguous for
defaults unless a specific schema mapping says what they mean. If a source file
contains two aliases for the same normalized column, loading should fail with a
clear error instead of choosing one by priority.

The current implementation honors explicit mappings from referenced schema YAML
files for `portfolio_performance_columns` and `security_performance_columns`.
For mapped columns, the explicit schema mapping is authoritative. Built-in
aliases remain the fallback for columns not mapped in the schema file.

Comparison-only datasets such as FX rates, transactions, and holdings currently
use the performance-comparison alias/default layer. They do
not require entries in `axysapx_column_mappings.yaml`.

Inline snapshot-specific schema mappings remain a future step. The current
test fixtures use one referenced Axys/APX column-mapping file plus
performance-comparison defaults.

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
- `runner.py` also re-exports explanation helpers and constants as a
  compatibility bridge for earlier imports.

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
  estimate is not available. The first supported estimate uses vendor security
  contribution deltas as related-output impact estimates.
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
  `target_output`, while other security performance findings remain
  `related_output`.

Portfolio performance is the authoritative source of portfolio base currency.
When it is available, holdings and transactions inherit a missing row-level
base currency from the portfolio and fail validation if they contradict it.
Security-performance beginning/ending market value, income, and gain/loss are
compared as related reported-output diagnostics alongside return, weight, and
contribution.

Normalized monetary names use one currency-basis rule:

- unqualified monetary fields in holdings and transactions use the row
  `currency`;
- `base_` monetary fields in those detailed datasets use portfolio
  `base_currency`;
- monetary fields in portfolio/security performance datasets are inherently
  in portfolio base currency and remain unprefixed; and
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

These rules are intentionally asymmetric. Price rows can be linked through
security-period output because they share a security identifier and date. FX
  rates need an explicit portfolio identifier and local exposure; ppar does not
  infer either from `security_performance`.

## Current Limits

The current evidence model is useful, but it should not be overstated.

- Evidence counts are not contribution amounts. A portfolio-period summary can
  say that related price, transaction, holding, or security-output
  findings exist; it does not yet calculate how much each item explains of the
  portfolio return delta.
- Portfolio-period evidence rankings are review-priority heuristics. They help
  sort the audit trail but do not quantify causal contribution.
- Prices often lack portfolio identifiers, but they can be linked through
  security-performance periods when `security_performance` is available. FX
  rates without explicit portfolio and exposure context remain unlinked and
  unestimated.
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
- The implementation compares source evidence. It does not recalculate TWR from
  raw transactions or holdings.

## Contribution Ranking Direction

The existing `rank_portfolio_period_evidence()` helper is a review-priority
sort. It helps decide which findings to inspect first, but it is not a
contribution model and should not be labeled as explaining a portion of return.

The contribution-ranking layer is optional, conservative, and explicit about
its basis. The output should be allowed to say "no estimate" when a finding
lacks the denominator, linkage, or YAML-selected methodology needed for a
defensible return-impact estimate.

Implemented contribution-candidate fields:

- `estimated_return_impact`: Optional numeric estimate of return impact.
- `impact_basis`: Short basis label, such as `portfolio_source_field`,
  `security_contribution`, `linked_holding_price`, or `no_estimate`.
- `impact_confidence`: High, medium, or low confidence in the estimate.
- `impact_method`: The formula or rule used to estimate impact.
- `impact_message`: Human-readable explanation of the estimate or why no
  estimate was produced.

The contribution-candidate implementation preserves all ranked evidence rows
and populates stable impact columns. It estimates only where the YAML explicitly
selects a supported `contribution_impact_methods`, `holding_impact_methods`,
or `transaction_impact_methods` policy and the current evidence carries enough
denominator, weight, or vendor output context to state the method clearly.
When a changed source field is known review evidence but should not receive an
additive estimate, YAML can instead mark it with
`evidence_only_impact_methods`.

```yaml
contribution_impact_methods:
  portfolio_source_field:
    method: source_field_delta_over_begin_market_value
    denominator_source: begin_market_value
    source_fields:
      - income
      - gain_loss
  security_contribution:
    method: vendor_contribution_delta
  security_return:
    method: security_return_delta_times_weight
    weight_source: snapshot_a_weight
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
evidence_only_impact_methods:
  fx_rates:
    method: evidence_only
    source_fields:
      - fx_rate
  holdings:
    method: evidence_only
    source_fields:
      - cost
```

Current supported impact estimates:

1. Vendor security contribution delta:
   - `impact_basis = security_contribution`
   - `impact_method = vendor_contribution_delta`
   - `impact_confidence = medium`
   - Applies only when YAML explicitly configures
     `contribution_impact_methods.security_contribution.method` as
     `vendor_contribution_delta`.
   - Uses the vendor-provided contribution delta as a related-output impact
     estimate. This is preferred for the `security_return_or_contribution`
     cause-area aggregate when available.
2. Weighted security return delta:
   - `impact_basis = security_return_weighted`
   - `impact_method = security_return_delta_times_weight`
   - `impact_confidence = low`
   - Applies only when YAML explicitly configures
     `contribution_impact_methods.security_return.method` as
     `security_return_delta_times_weight` and `weight_source` as
     `snapshot_a_weight`.
   - Formula: `security_return_delta * snapshot_a_weight`.
   - Used as a candidate-level review cross-check or fallback. It is not summed
     with vendor contribution in the same security cause bucket because that
     would double-count two estimates of related security-level performance.
3. Portfolio source-field delta:
   - `impact_basis = portfolio_source_field`
   - `impact_method = source_field_delta_over_begin_market_value`
   - `impact_confidence = low`
   - Applies only when YAML explicitly configures
     `contribution_impact_methods.portfolio_source_field.method` as
     `source_field_delta_over_begin_market_value`, `denominator_source` as
     `begin_market_value`, and the source field in `source_fields`.
   - Formula: `source_field_delta / beginning_market_value`.
   - Currently applies only to return-bearing portfolio source fields such as
     `income` and `gain_loss`. It does not apply to control/output fields such
     as `end_market_value`.
4. Performance-treated transaction amount delta:
   - `impact_basis = transaction_performance_amount`
   - `impact_method = transaction_amount_delta_over_return_denominator`
   - `impact_confidence = low`
   - Formula: `source_signed_transaction_amount_delta / return_denominator`.
   - Applies only when YAML explicitly configures
     `transaction_impact_methods.performance.method` as
     `transaction_amount_delta_over_return_denominator`.
   - Also requires changed transaction `amount` fields whose source-supplied or
     YAML-rule semantics mark them as performance-affecting and whose cash-flow
     sign is positive or negative. Missing/zero denominators, out-of-period
     transaction dates, external-flow treatment, neutral treatment, unknown
     semantics, and cash-flow `none` remain evidence-only until separate
     methods are modeled. Missing-input summaries name these unsupported
     treatments as `transaction impact method`, `external-flow impact method`,
     `neutral-flow impact method`, or `no-cash transaction impact method` so
     reviewers can distinguish semantics gaps from method gaps.
   - If YAML sets `transaction_impact_methods.external_flow.method` to
     `evidence_only`, external-flow transaction rows still receive no estimate,
     but review summaries identify the explicit evidence-only policy rather
     than implying a missing formula.
   - External-flow rows may carry `transaction_impact_diagnostic` values that
     distinguish evidence-only policy, missing method configuration, and
     Modified Dietz cross-check estimates. Eligible Modified Dietz rows may
     also carry `transaction_impact_diagnostic_estimate`. These diagnostics are
     reviewer-facing only; they do not populate `estimated_return_impact`.
5. Holding market value delta:
   - `impact_basis = holding_market_value`
   - `impact_method = market_value_delta_over_return_denominator`
   - `impact_confidence = low`
   - Formula: `holding_market_value_delta / return_denominator`.
   - Applies only when YAML explicitly configures
     `holding_impact_methods.market_value.method` as
     `market_value_delta_over_return_denominator` and `denominator_source` as
     `begin_market_value`.
   - This is intentionally a low-confidence screening estimate because a
     holding market value delta may reflect price, quantity, FX,
     accrued-interest, or booking changes.
6. Weighted price delta:
   - `impact_basis = price_weighted`
   - `impact_method = price_delta_over_snapshot_a_price_times_weight`
   - `impact_confidence = low`
   - Formula: `(price_delta / snapshot_a_price) * snapshot_a_weight`.
   - Applies only when YAML explicitly configures
     `price_impact_methods.price.method` as
     `price_delta_over_snapshot_a_price_times_weight` and `weight_source` as
     `snapshot_a_weight`.
   - Price findings link through security-performance periods, so one changed
     security price may produce one explanation row per affected portfolio.
7. Holding accrued delta:
   - `impact_basis = holding_accrued`
   - `impact_method = accrued_delta_over_return_denominator`
   - `impact_confidence = low`
   - Formula: `holding_accrued_delta / return_denominator`.
   - Applies only when YAML explicitly configures
     `holding_impact_methods.accrued.method` as
     `accrued_delta_over_return_denominator` and `denominator_source` as
     `begin_market_value`.
   - This is intentionally a low-confidence screening estimate because accrued
     balances depend on source income accrual and pricing conventions.
8. Holding quantity delta:
   - `impact_basis = holding_quantity_unit_market_value`
   - `impact_method =
     quantity_delta_times_snapshot_a_unit_market_value_over_return_denominator`
   - `impact_confidence = low`
   - Formula:
     `(holding_quantity_delta * snapshot_a_holding_unit_market_value) /
     return_denominator`.
   - Applies only when YAML explicitly configures
     `holding_impact_methods.quantity.method` as
     `quantity_delta_times_snapshot_a_unit_market_value_over_return_denominator`
     and `denominator_source` as `begin_market_value`.
   - Snapshot A unit market value is calculated from the same holding row as
     `snapshot_a_market_value / snapshot_a_quantity`.
   - This is intentionally a low-confidence screening estimate because unit
     market value can embed price, FX, accrued-interest, or booking effects.
   - `holding_impact_methods.quantity.method: evidence_only` remains
     available when a changed quantity should be visible but not additive.
10. Holding cost evidence:
   - `holding_impact_methods.cost.method: evidence_only` marks changed
     holding cost as intentional review evidence.
   - It does not create `estimated_return_impact`; cost-basis changes are not
     direct period-return attribution in the current model.
11. Transaction quantity, price, and commission evidence:
   - `transaction_impact_methods.quantity.method: evidence_only` and
     `transaction_impact_methods.price.method: evidence_only` mark changed
     transaction units and price values as intentional review evidence.
   - `transaction_impact_methods.commission.method: evidence_only` marks
     changed transaction commission as intentional review evidence.
   - They do not create `estimated_return_impact`; transaction amount is the
     supported additive transaction field so quantity, price, commission, and
     amount are not double-counted.
12. FX rate evidence:
   - The FX-rate row does not receive `estimated_return_impact` because the
     rate is not a separate Modified Dietz input.
   - When `rate delta * unchanged local exposure` matches one changed
     `holdings.base_market_value` or `transactions.base_amount` row, the report
     links the FX rate to that security as supporting evidence. The base-value
     row receives the counted impact.
All other rows use `impact_basis = no_estimate` until a defensible method,
denominator, and linkage are available.

`evidence_only_impact_methods` is the explicit unsupported-but-known escape
hatch. It does not create `estimated_return_impact`; instead, workbook rows are
marked review-only and `Explanation` says the row is configured as
evidence-only. This keeps intentionally review-only changes from looking like
missing setup. Supported dataset keys are `fx_rates`, `holdings`, and
`transactions`; each dataset may list only fields that the comparison engine
already compares.

First contribution estimates should start only where the math is defensible:

- Portfolio performance source fields: Changes in return-bearing fields such
  as `income` and `gain_loss` may support rough return-impact estimates when
  beginning market value is present. These estimates should clearly state the
  denominator and formula. Control/output fields such as `end_market_value`
  should remain evidence-only until a defensible interpretation is modeled.
- Security contribution deltas: If a vendor supplies contribution, changed
  contribution can be ranked by contribution delta. This remains `related_output`
  evidence, not root cause, because contribution is already calculated output.
- Security return deltas: If a security return finding has a usable portfolio
  weight, a low-confidence weighted estimate can be calculated as
  `security_return_delta * snapshot_a_weight`. Prefer vendor contribution
  deltas in the aggregate cause summary when both are present.
- Holding, price, and transaction evidence: These should receive an estimated
  return impact only when the finding can be linked to an affected portfolio,
  period, security, and denominator. Transaction amount evidence also requires
  a normalized transaction category plus modeled transaction sign and flow
  semantics, and currently estimates only `performance` flow treatment.
  Transaction evidence uses the linked portfolio period's snapshot A beginning
  market value as the return denominator when the transaction date maps to a
  configured period.
  Otherwise these rows should remain ranked review evidence with `impact_basis`
  set to `no_estimate`.
- FX evidence: FX rate changes do not receive a separate portfolio-period
  return impact. Configure them as evidence-only and use an explicit portfolio
  and unchanged local exposure to link them to the counted base-currency value.

Contribution ranking should not require every finding to receive an estimate.
A mixed output is acceptable: some rows may have `estimated_return_impact`, and
others may remain review-priority evidence only.

Residual findings should wait until the system has a credible contribution
model. Emitting residuals before enough impact estimates exist would imply a
precision the comparison does not yet have.

Reports may still include a residual status section for portfolio-period return
changes. This section should use a section-level caveat plus compact per-period
reason labels and review notes, rather than calculating an unexplained amount
from partial or mixed-confidence estimates.

Current residual review statuses are withheld labels only:

- `withheld_no_estimates`: No regular contribution estimates are available for
  the changed portfolio period.
- `withheld_partial_estimates`: Some regular contribution estimates exist, but
  they are partial, low/medium confidence, or potentially overlapping.
- `withheld_cross_checks_only`: Review-only transaction cross-check estimates
  exist, but no regular contribution estimates are available.

These statuses keep the unexplained amount visible as a review concern without
emitting a numeric residual. Report helper tables include a
`residual_review_note` field that explains why each withheld status should not
be interpreted as a calculated residual.

Report bundles can be written with `write_performance_comparison_report_bundle()`.
The public bundle API retains HTML output by default for compatibility. The
user-facing `ppar audit` command defaults to XLSX plus HTML. `--no-xlsx-output`
selects HTML-only output, `--no-html-output` selects XLSX-only output, and both
options select promoted CSV-only review output. Every bundle therefore contains
primary review artifacts, raw findings, current
report helper tables as CSV files, a short `README.md`, a compact
`review_summary.json`, and a JSON
manifest with options, counts, artifact names, typed semantic fingerprints, and
canonical review-sheet display fingerprints. This makes reviewer handoffs
reproducible without coupling the comparison engine to a future
Axys/APX-specific presentation layer.

Use `report_bundle_contract()` when code or review automation needs to inspect
the generated-bundle handoff shape. The helper returns the portfolio/security
audit filenames, required artifact keys, manifest keys, review entrypoints,
review-summary keys, normalization version, declared volatile metadata,
Modified Dietz review basis, and review vocabulary keys. It is a contract
surface for the bundle. It remains
not a new transaction-classification or accounting layer.

The intended bundle review order starts in the generated report files:

1. The level-specific XLSX audit, when present, or its HTML counterpart for
   browser review. Start with
   `Performance Differences`, then use `Performance Difference Causes` to
   understand what explains each portfolio-period difference.
2. `source_detail.csv` / `findings.csv`: complete finding-level audit output for
   troubleshooting and traceability.

The generated artifacts fall into a small taxonomy:

- first-stop review surfaces: `portfolio_audit.*` or `security_audit.*`;
- reviewer handoff metadata: `README.md`, `manifest.json`, and
  `review_summary.json`;
- audit/export backbone: `findings.csv`, `performance_differences.csv`,
  `performance_difference_causes.csv`, `x_ref_issues.csv`,
  `needs_review_summary.csv`,
  `portfolio_period_summary.csv`, `cause_summary.csv`, `impact_estimates.csv`,
  `impact_coverage.csv`, `top_evidence.csv`, and `residual_status.csv`;
- supplementary diagnostics: `context_evidence_summary.csv`,
  `context_evidence.csv`, `transaction_activity.csv`,
  `transaction_cross_checks.csv`, `flow_cross_check_reconciliation.csv`, and
  `transaction_matching_diagnostics.csv`;
- opt-in reconstruction diagnostics: `reconstruction_summary.csv`,
  `return_reconstruction_checks.csv`, and
  `security_return_reconstruction_checks.csv`.

Generated bundle `README.md` files explicitly point reviewers to the transaction
activity, transaction cross-check, and flow-reconciliation CSVs as supplementary
transaction and external-flow diagnostics. `transaction_matching_diagnostics.csv`
is audit-support for transaction row-identity evidence rather than a main review
section. The CSV artifacts remain in the bundle because they make handoffs,
automation, validation, and troubleshooting reproducible; they are not meant to
replace the first-stop workbook/report review flow.

User-facing commands package those artifacts in `audit_support.zip` by default
and promote `source_detail.csv` to the report root. The
`--expand-all-supporting-files` option retains the equivalent individual files
under `supporting_files/` for integrations and detailed troubleshooting.

The `ppar.performance_comparison.cli.report_bundle` package CLI module
exposes the same bundle workflow for comparison YAML files.
Existing bundles can be checked with
`ppar.performance_comparison.cli.validate_bundle`, which verifies required
artifacts, manifest metadata, typed CSV content, canonical HTML/XLSX review
content, empty-table headers, and whichever HTML/XLSX primary artifacts the
manifest includes. Manifest version 4 records the selected output modes and
excludes only its generation timestamp and
XLSX creation/package timestamps from normalized repeatability; statuses,
financial values, labels, causes, evidence rows, and ordering remain covered.
Packaged demo scenario coverage can be checked with
`ppar.performance_comparison.cli.validate_demo_matrix`, which verifies that
the current YAML fixtures still produce the reviewer-facing scenarios named in
the demo matrix.

The HTML report starts with a first-screen `Problems` grid instead of a stack of
evidence tables. It uses the same period-level triage data to show one compact
row per actionable issue, with severity, portfolio, period, return delta,
problem, action required, why it matters, and an optional evidence link. Links
use stable period-specific row anchors when the target section carries
portfolio-period fields, so a reviewer can audit the conclusion without using
raw tables as the primary workflow. Rows are sorted to keep needs-review periods
first, then missing-impact-input periods, then larger absolute return deltas,
then portfolio/date. The grid is deliberately static for now: it guides the
reviewer through existing evidence without adding new calculation logic. Its
lightweight browser filters search rendered row text, review status, and
missing-input flags only; they do not change report data or require a server.
The default HTML presentation should stay short at the top: Problems first,
with backing tables inside an Evidence Appendix.

The XLSX workbook is the primary reviewer presentation over the same review
tables used by the HTML/CSV bundle artifacts. Lower-level bundle calls generate
it when requested with `include_workbook=True` or `--include-workbook`; the
packaged demo writes it by default. The workbook starts with the `Performance
Differences` sheet. In the portfolio demo, it has one row per changed portfolio
period, showing the decimal return difference, explained difference, and any
unexplained remainder. In the security demo, it shows security-level return
differences when security-performance rows changed, and it adds explicit
no-difference rows for changed portfolio periods with no security-level return
difference. The
`Performance Difference Causes` sheet lists input rows such as holdings,
transactions, and FX rates; its `B - A Difference` values are raw input-value
differences, and its `Performance Difference Explained` values appear only when
ppar has a defensible input-level explanation. User-facing bundle generation
now requires every changed source-data field that ppar knows how to classify to
be explicitly configured as additive, evidence-only, or suppressed in YAML
before any report artifacts are written. Unresolved residuals are summarized in
the `Performance Differences` comments and the full underlying finding detail
remains in `source_detail.csv`; there is no default residual-evidence sheet unless
a future diagnostic can identify a real reviewable mechanism.
Changed periods without any visible cause or promoted evidence row get a
`no_underlying_causes_found` diagnostic row. The `source_detail.csv` artifact
preserves the full finding-level detail, including context rows such as cost and
reported-performance diagnostics that confirm reporting differences but are not
root causes.
The Explanation wording is intentionally report-level aware. Portfolio
workbooks explain transaction rows by their portfolio-return role, so a `dp`,
`dv`, or `in` transaction can be described as causing the cash-balance ending
`holdings.market_value` row to move, while a portfolio `wd` row can include
weighted external-flow language. Security workbooks explain transaction rows by
the affected security return container, so the same transaction category family
uses semantic labels such as `external flow`, `fee/expense`, or `income`.
Those wording differences are not separate calculations; they reflect the
different review question asked by the portfolio and security report families.
Workbook-specific behavior is limited to spreadsheet
ergonomics such as sheet names, frozen headers, filters, column widths, Excel
number formats, and header comments that explain column meaning.

YAML setup completeness is the default config-validation and report-bundle
guardrail. `validate_config` and the report-bundle API option
`require_complete_yaml_setup=True` fail before writing artifacts if a changed
source field lacks additive, evidence-only, or suppression YAML. The shared CLI
flag `--allow-incomplete-yaml` exists only for diagnostic config checks,
diagnostic bundles, and tests. Strict supported-attribution setup remains an
additional opt-in guardrail through `require_causal_attribution=True`,
`--require-causal-attribution`, or `--require-supported-attribution-setup`.
Strict causal attribution does not require every performance difference to be
fully explained; it only rejects missing setup for supported attribution methods.

The `needs_review_summary.csv` bundle artifact remains a derived period-level
triage export. It highlights changed portfolio periods with evidence-only
areas, missing impact inputs, low-confidence estimates, transaction
cross-checks, or withheld residuals. It does not add new calculation rules; it
only summarizes existing report helper tables into reviewer cues and suggested
next steps. Period-level bundle tables carry a stable `review_key` where
possible, and `needs_review_summary.csv` includes `review_detail_artifacts` to
name the CSVs most relevant to each changed period.

The packaged Axys/APX fixtures intentionally separate user-facing setup
templates from validation fixtures. `ppar setup` installs the portfolio and
security comparison starter file, `run_audit.py`. That setup-generated
script writes `portfolio_audit.*` and `security_audit.*`, plus CSV artifacts and
a manifest under `output/portfolio` and `output/security`.

The remaining YAML files are scenario-coverage fixtures for tests and
validators:

- `performance_comparison.yaml`: Clean baseline.
- `ppar_performance_comparison_restatement.yaml`: Controlled single
  restatement with missing transaction setup.
- `ppar_performance_comparison_restatement_transaction_rules.yaml`: Same data
  with explicit transaction rules and transaction impact setup.
- `ppar_performance_comparison_multi_restatement.yaml`: Multiple portfolios,
  multiple periods, context rows, and residual/coverage behavior.
- `ppar_performance_comparison_policy_gap_demo.yaml`: Missing-YAML actions such
  as selecting `contribution_impact_methods`, configuring
  `transaction_impact_methods`, setting `denominator_source`, and defining
  transaction sign/flow semantics.
- `ppar_performance_comparison_modified_dietz.yaml`: Cross-check-only Modified
  Dietz external-flow diagnostics.
- `ppar_performance_comparison_suppressed.yaml`: Active-vs-suppressed finding
  behavior and audit visibility.

The compact demo scenario matrix lives in `ppar/setup_templates/axysapx_performance_comparison/README.md`.
It lists which YAML fixture covers each reviewer-facing problem type and which
scenarios are intentionally planned rather than covered. It also tracks the
goal that every supported public YAML impact method should have at least one
packaged demo scenario and validator assertion.

Context evidence is summarized separately from impact estimates. The
`context_evidence.csv` bundle artifact contains rows whose evidence role is
`context`, a reviewer-facing `context_use`, and a
`return_impact_treatment` value that states the row is not included in
return-impact estimates. This keeps useful audit context visible without
quietly relaxing the rule that every modeled impact method must be explicit.
Context evidence linked to changed portfolio periods is also surfaced as a
high-priority cue in `needs_review_summary.csv`.
Top-of-report reviewer triage counts include high-priority context groups so
reviewers can see immediately when context evidence needs early attention.
Row-level `context_evidence.csv` detail carries the same priority labels and
reasons as the grouped summary so reviewers do not need to infer priority by
joining artifacts manually.
In the workbook, plausible evidence-only input rows for unresolved periods may
be shown on the `Performance Difference Causes` sheet so the reviewer can see
likely explanations and calculated explanations together. Transaction component
rows such as `transactions.quantity`,
`transactions.price`, and `transactions.commission` also appear on the
`Performance Difference Causes` sheet when they support a changed `transactions.amount`;
their explained-difference columns remain blank because they are inputs for the
changed transaction amount, not separate return-impact estimates. Cost basis
changes remain supporting evidence unless a later model gives them a defensible
return-impact interpretation.

Transaction cross-checks are summarized separately from impact estimates. The
`portfolio_period_transaction_cross_checks()` helper and report section group
rows with `transaction_impact_diagnostic_estimate` by portfolio period and
impact policy. The grouped totals are review aids only and remain excluded from
`estimated_return_impact`, impact coverage totals, and cause-summary totals.
Report bundles include this table as `transaction_cross_checks.csv`.

Flow cross-check reconciliation compares those transaction cross-check totals
with review-only portfolio `flow` delta estimates, calculated as
`flow_delta / return_denominator` when a usable denominator is present. The
`portfolio_period_flow_cross_check_reconciliation()` helper and report section
label each period as `aligned`, `different`, `missing_portfolio_flow_delta`, or
`missing_transaction_cross_check`. These labels are double-counting review
signals only; they do not change contribution totals. Report bundles include
this table as `flow_cross_check_reconciliation.csv`.

Transaction matching diagnostics are summarized separately from transaction
activity. The `transaction_matching_diagnostics()` helper and bundle CSV count
transaction matching labels such as `matched_by_id`,
`matched_by_singleton_fallback`, `added_in_snapshot_b`,
`missing_from_snapshot_b`, `ambiguous_fallback_match`,
`transaction_id_unmatched`, and `strict_fallback_unmatched`, with reviewer
confidence and interpretation columns distinguishing strong transaction-ID
matches, conservative singleton fallback matches, unpaired add/drop rows, and
withheld ambiguous fallback groups. Reviewer notes explain whether rows were
paired by stable transaction ID, paired by exact singleton fallback, appeared
in only one snapshot, or were left unpaired because fallback keys were
ambiguous. Report bundles include this table as
`transaction_matching_diagnostics.csv`; row-level match status remains visible
in the source detail, but the normal workbook and HTML review flow do not
surface a standalone transaction-match section.

Report bundles can include a level-specific HTML audit as the browser view of
the same review model used by its XLSX counterpart. The HTML artifact is
intentionally conservative:
it uses the same workbook table model, section ordering, column labels, and
column tooltips as the workbook, with lightweight CSS and accessible table
captions for browser review rather than separate HTML-specific analytics logic.
The bundle writer is the only report path; standalone HTML
rendering helpers remain internal implementation details.

## Historical Near-Term Roadmap

This section records the earlier near-term plan that led to the current
implementation checkpoint. For current next work, use the central
[`PPAR Roadmap`](roadmap.md).

The design work should continue to move slowly and favor reviewer clarity over
broad new machinery.

1. Tighten YAML specification documentation and examples.
   Make it easier to decide when a changed field should be additive,
   review-only, diagnostic, or unsupported. Keep examples concrete enough that
   a larger real-data review can start from known policy choices.
2. Add tests around real-world edge cases before adding new attribution logic.
   Prefer small fixture rows that capture ambiguous or risky cases: missing
   denominators, overlapping explanations, unmatched transactions, no
   underlying causes, evidence-only fields, and strict attribution failures.
   Initial coverage protects workbook wording for configured methods that
   still cannot estimate because required source inputs, such as a usable
   return denominator, are missing.
3. Expand and validate against more realistic performance comparison data.
   Use larger multi-portfolio, multi-period inputs once the expected workbook
   behavior is clear from smaller fixtures.
   Initial coverage validates that the packaged multi-restatement fixture keeps
   a large clean background portfolio with many periods while still surfacing
   the intended smaller restatement issues.
4. Revisit `explain.py` only when a new feature makes a natural split obvious.
   Avoid splitting it merely because it is large; split only around stable
   responsibilities such as impact estimates, transaction diagnostics, or
   summary tables when active work creates that boundary. Initial transaction
   diagnostic presentation helpers now live in `_transaction_diagnostics.py`;
   keep additional extraction similarly narrow and behavior-preserving.

Guardrails for all four phases:

- Strengthen the contribution-candidate helper only where the math is
  defensible.
- Keep external-flow diagnostics visible while Modified Dietz is limited to
  cross-check-only estimates. Diagnostics should name missing inputs and
  inactive methods without implying that an estimate has been accepted into
  contribution totals.
- Add a residual concept only after there are enough credible contribution
  estimates. A residual emitted too early would imply precision the system does
  not have.
- Keep report/export formats separate from comparison logic. CSV, HTML, and
  XLSX outputs should remain presentation layers over stable helper
  tables.
- Avoid adding new datasets unless real source files expose evidence that is
  not adequately represented by the current normalized datasets.

## Implementation Status

### Milestone 1: Portfolio Performance Difference Engine - Implemented

The core difference engine is implemented for the current normalized dataset
surface.

- Read comparison YAML.
- Load two snapshot directories.
- Normalize required `portfolio_performance` and any available optional
  `security_performance` source using built-in inference where practical.
- Compare rows by configured keys.
- Emit findings for added/dropped rows and changed portfolio/security returns,
  weights, and contributions.
- Apply tolerances and suppressions.
- Return findings as a Polars DataFrame.
- Provide compact findings and summary helper tables.

### Milestone 2: Human Report - Implemented

The reviewer-facing report layer is implemented.

- Add HTML and XLSX output using the current findings helpers.
- Group by portfolio and period.
- Rank largest return and contribution deltas.
- Include impact summaries, cross-check summaries, evidence sections, and
  audit appendices.
- Surface context-only evidence separately from modeled impact estimates.
- Summarize context evidence by dataset, source column, context use, affected
  identifiers, and reviewer priority.
- Include residual withheld statuses and residual review notes without
  calculating numeric residuals from incomplete estimates.
- Include accessible table captions while keeping HTML presentation separate
  from analytics logic.
- Write reproducible report bundles with manifest and validation helpers.
  Current bundles include `context_evidence_summary.csv` and
  `context_evidence.csv` alongside impact, transaction, residual, and
  top-evidence tables.

### Milestone 3: Supporting-File Explanations - Implemented Evidence Layer

Supporting-file comparisons are implemented at the evidence-linking level for
the current datasets. Causal attribution and contribution-ranking estimates
remain intentionally conservative.

- Compare holding price values for securities with changed returns.
- Compare transactions for affected portfolio/security/period rows.
- Compare holdings and cash balances.
- Compare FX rates when present.
- Add confidence, residual, and needs-review findings where the evidence is
  incomplete or method-dependent.

### Milestone 4: Public API And Demo - Implemented

The public command and demo surface is implemented for the current checkpoint.

- Add stable public entry points.
- Add sample comparison fixture directories.
- Add public setup/report commands and source-checkout report/bundle commands.
- Document configuration and finding codes.

## Long-Term Dataset Watchlist

The current normalized dataset set already covers the first useful comparison
surface: portfolio performance, security performance, holdings, transactions,
and FX rates. Additional
datasets should be added only when real source files expose evidence that is
not adequately represented by those existing datasets.

Potential long-term datasets to keep in mind:

- `market_values` or `valuations`: Portfolio/security valuation totals when a
  vendor provides them separately from holdings.
- `tax_lots`: Lot-level cost, realized gain/loss, acquisition-date, or
  tax-basis differences.
- `corporate_actions`: Splits, mergers, spinoffs, symbol changes,
  return-of-capital events, and similar security events.
- `benchmark_performance`: Benchmark-relative performance deltas if benchmark
  comparison becomes part of the explanation question.
- `fees_expenses`: Management fees, custody fees, advisory fees, expense
  accruals, or other charges when they are not represented as transactions.
- `realized_gain_loss`: Realized profit/loss supplied as a separate accounting
  output rather than derivable from transactions or lots.
- `extract_manifest`: Snapshot timestamps, vendor version, accounting basis,
  source system, extraction parameters, and other run metadata.

These are open design items, not near-term implementation targets. Before
adding any of them, prefer strengthening comparisons for useful columns already
present in existing datasets, such as holding cost/accrued values or
transaction quantity, price, and commission.

## Open Design Issues

1. What is the minimum set of fields required to match transactions reliably
   when `transaction_id` is unavailable?
2. Should suppression reasons remain optional, or should production-style
   workflows require them for audit discipline?
3. Should row matching allow fuzzy keys, such as ticker fallback when security
   id changes?
4. Should row-level tolerances stay simple, or should more finding types support
   relative/materiality-aware tolerances?
5. Which contribution-ranking methods, if any, should be added next after the
   initial YAML-gated portfolio/security/transaction methods?
6. When contribution ranking exists, when should an unexplained residual be
   emitted?
7. How much of the existing `ppar.axys` inference code should be shared with
   future vendor adapters, and how much should remain Axys/APX-specific?
8. Which additional supporting-file columns, if any, provide enough explanatory
   value to justify expanding the current normalized comparison surface?

## Current Roadmap Location

Treat the current comparison engine, evidence layer, report bundle, return
reconstruction checks, and workbook model as the baseline. Current next work
should be tracked in
[`roadmap.md`](roadmap.md), not
added here as a competing roadmap.
