# Performance Comparison Design Notes

## Purpose

The performance comparison feature explains why calculated performance for the
same portfolio and period changed between two source-data extraction dates.

The core question is:

> Why did my performance for period A change from when I ran it on date 1 to
> when I ran it on date 2?

The feature will compare two snapshot directories. Each snapshot contains
vendor exports such as portfolio performance, security performance, prices, FX
rates, transactions, positions, cash, and security master/reference files.

This should not be treated as an Axys-only feature. The comparison engine
should operate on normalized internal datasets. Vendor-specific behavior should
live in small normalization adapters, with Axys as the first likely adapter.

The first implementation should stay intentionally narrow: compare normalized
portfolio performance, security performance, security master, positions, cash,
prices, FX rates, and transactions rows, report material changes, and produce a
clear finding model. Deeper causal inference can be added after the finding
model is stable.

## Current Checkpoint

The current implementation has crossed from pure design into a usable
data-loading and comparison checkpoint. It can load two snapshot directories,
compare the first set of normalized datasets, emit stable finding records, apply
explicit suppressions, and produce small summary tables for review.

Implemented normalized comparison datasets:

- `portfolio_performance`
- `security_performance`
- `security_master`
- `prices`
- `fx_rates`
- `transactions`
- `positions`
- `cash`

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
- security-period summaries
- security-period evidence breakdowns

This checkpoint is still a comparison and evidence organization layer. It is
not yet a causal attribution engine, a return calculator, or a final report
writer.

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

The comparison engine should not know whether the source system was Axys,
FactSet, Bloomberg PORT, custodian files, or another vendor. It should compare
standard datasets with standard column names and let adapters handle source
schema details.

## Current Package Shape

```text
ppar/performance_comparison/
  __init__.py
  aliases.py
  cash.py
  columns.py
  specification.py
  source_loader.py
  compare.py
  findings.py
  explain.py
  fx_rates.py
  period_linking.py
  positions.py
  prices.py
  runner.py
  security_master.py
  security_performance.py
  transactions.py
```

Current responsibilities:

- `specification.py`: Read and validate comparison YAML.
- `source_loader.py`: Load one snapshot directory, resolve optional files, and
  normalize configured columns.
- `compare.py`: Compare normalized snapshot A/snapshot B data sets.
- Dataset modules such as `prices.py`, `fx_rates.py`, `transactions.py`,
  `positions.py`, and `cash.py`: dataset-specific loading, comparison keys,
  changed-column rules, and default aliases.
- `period_linking.py`: Link dated evidence to containing portfolio periods
  where the linkage is conservative.
- `findings.py`: Define finding records, roles, suppressions, and codes.
- `explain.py`: Build portfolio/security period summaries, evidence
  breakdowns, rankings, contribution candidates, and cause summaries.
- `runner.py`: Public execution helpers, compact output tables, and summary
  tables.

Report rendering remains deliberately outside the package for now. The current
helpers return stable Polars tables that can later feed CSV, Markdown, HTML, or
other report formats.

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
- `security_master`
- `prices`
- `fx_rates`
- `transactions`
- `positions`
- `cash`

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

`security_master` required columns:

- `security_id`

`security_master` useful optional columns:

- `security_name`
- `ticker`
- `cusip`
- `isin`
- `currency`
- `country`
- `sector`
- `industry`
- `asset_class`
- additional classification code/name pairs

`prices` required columns:

- `security_id`
- `price_date`
- `price`

`prices` useful optional columns:

- `currency`
- `price_source`
- `price_type`

`fx_rates` required columns:

- `from_currency`
- `to_currency`
- `rate_date`
- `fx_rate`

`fx_rates` useful optional columns:

- `rate_source`
- `rate_type`

`fx_rates` represents exchange rates, not currency positions or cash
balances. Currency exposure belongs in positions, cash, transactions, or
valuation datasets.

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

Future transaction enhancements should consider optional fixed-income and
income detail fields when real source files provide them:

- `accrued_interest`
- `interest`
- `principal`
- `gross_amount`
- `net_amount`
- `tax_withheld`

These fields should remain part of the `transactions` dataset rather than
forcing a separate income/accrual dataset by default. Position-level `accrued`
represents an accrued balance at a position date; transaction-level accrued
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
when the source does not provide a category. For example, `BUY` maps to `buy`,
`SELL` maps to `sell`, and `DIV` or `INT` maps to `income`. This category is an
explanation label only.

`cash_flow_sign` and `performance_flow_sign` are optional source-supplied
semantics. They should not be inferred from transaction code in the first pass.
Directional sign rules are vendor-specific and accounting-specific enough that
guessing them would make `transaction_activity` look more precise than it is.
Until those sign rules are explicitly modeled, transaction differences should
remain evidence-only and should not receive an estimated return impact.

Transaction activity should become eligible for a future return-impact estimate
only when the changed evidence has an affected portfolio, security, linked
portfolio period, normalized transaction category, return denominator, and
modeled transaction sign and flow semantics. Until then, transaction summaries
should expose missing impact inputs instead of calculating an estimate.

Markdown reports should include a compact Transaction Activity section so
reviewers can see changed fields, deltas, and missing impact inputs without
digging through lower-level evidence tables.

`positions` required columns:

- `portfolio_id`
- `security_id`
- `position_date`

`positions` useful optional columns:

- `quantity`
- `market_value`
- `cost`
- `accrued`
- `price`
- `currency`

`cash` required columns:

- `portfolio_id`
- `cash_date`

`cash` useful optional columns:

- `currency`
- `cash_balance`
- `market_value`

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
- Positions: portfolio code, security identifier, position date.

When transactions have a stable `transaction_id` in both snapshots, matching
rows can report changed amounts as `PC-TXN-AMT`. When no transaction id is
available, the fallback key includes portfolio, security, trade date, settlement
date if present, transaction code, quantity, price, and amount. This avoids
guessing that two similar rows are the same transaction. In that fallback mode,
an amount restatement is expected to appear as one `PC-TXN-DROP` and one
`PC-TXN-ADD`, not as `PC-TXN-AMT`.

Duplicate comparison keys should fail loudly before row-presence checks or
value comparisons run. Silent duplicate handling can collapse rows into a set
or multiply rows during joins, both of which can produce misleading findings.
The default policy is therefore to raise an error for duplicate keys in either
snapshot.

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

Root-cause evidence should come from input/source-like datasets such as prices,
FX rates, transactions, positions, cash, market values, accruals, income, and
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
  return, such as prices, FX rates, flows, market values, positions, accruals,
  transaction amounts, and cash balances.
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
- `prices`: `direct_input` because price changes can drive valuation and return
  changes.
- `fx_rates`: `direct_input` because exchange-rate changes can drive translated
  values and returns.
- `transactions`: `direct_input` because activity, cash flow, quantity, price,
  and amount changes can drive performance inputs.
- `positions`: `direct_input` because quantity, market value, price, and
  accrued-balance changes can drive performance inputs.
- `cash`: `direct_input` because cash balance and cash market value changes can
  drive portfolio-level valuation and return inputs.
- `security_master`: `context` because reference and classification changes
  usually explain how data is grouped or identified, not a numeric TWR driver
  by themselves.

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
PC-POS-QTY      Position quantity changed
PC-POS-MV       Position market value changed
PC-POS-ACCR     Position accrued amount changed
PC-CASH-MV      Cash balance or cash market value changed
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

The existing Axys column mapping configuration and the new comparison
configuration serve different purposes and should have distinct names:

- `axys_column_mappings.yaml`: Describes how Axys source columns map to
  normalized internal column names for reusable Axys datasets.
- `ppar_performance_comparison.yaml`: Describes which snapshots and files to
  compare, plus comparison tolerances, materiality, and suppressions.

A comparison probably needs one YAML file for the comparison run, not a separate
YAML file inside each snapshot. The comparison YAML can point at both snapshot
directories, define shared rules, and optionally reference vendor schema files
such as `axys_column_mappings.yaml`.

The performance comparison feature has its own normalization/default alias
layer. Referencing `axys_column_mappings.yaml` is a reuse mechanism for shared
Axys datasets, not a requirement that performance comparison become Axys-only.
Comparison-only datasets such as prices, FX rates, transactions, positions,
and cash can use performance-comparison mappings even when the referenced Axys
mapping file does not define them.

The comparison YAML should keep vendor-specific parameters minimal. Prefer
shared, source-agnostic sections for files, tolerances, materiality, and
suppressions. Use vendor-specific schema sections only when inference is
insufficient or when the two snapshots have different schemas.

### YAML Locations And Path Resolution

Configuration files should not be required to live inside snapshot
directories. Snapshot directories are data captures; YAML files are reusable
configuration and may be stored beside scripts, in a `comparisons/` directory,
or anywhere else convenient.

Path resolution should be predictable:

1. Absolute paths are accepted as-is.
2. Relative paths in `ppar_performance_comparison.yaml` resolve relative to
   that comparison YAML file.
3. Snapshot data files resolve relative to the configured snapshot directory.
4. Relative paths inside a referenced schema YAML, such as
   `axys_column_mappings.yaml`, resolve relative to that schema YAML file.

A suggested project layout is:

```text
comparisons/
  ppar_performance_comparison.yaml
  axys_column_mappings.yaml

snapshots/
  2026-05-01/
    portperf.csv
    secperf.csv
    sec_ref.csv

  2026-05-15/
    portperf.csv
    secperf.csv
    sec_ref.csv
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
    schema: axys_column_mappings.yaml

  b:
    label: run_2026_05_15
    path: snapshots/2026-05-15
    vendor: axys
    schema: axys_column_mappings.yaml

files:
  portfolio_performance: portperf.csv
  security_performance: secperf.csv
  security_master: sec_ref.csv
  prices: prices.csv
  fx_rates: fx_rates.csv
  transactions:
    path: transactions.csv
    required: true
  positions: positions_holdings.csv
  cash: cash.csv

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
    security_id: CASH_USD
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

### Column Mapping Defaults

The comparison YAML should use the same defaulting method for column mappings
that the existing Axys YAML uses. Users should not need to specify obvious
column names.

Column mappings should resolve in this order:

1. Snapshot-specific mapping in `ppar_performance_comparison.yaml`.
2. Shared comparison-level mapping in `ppar_performance_comparison.yaml`.
3. Referenced vendor schema file, such as `axys_column_mappings.yaml`.
4. Built-in default aliases.
5. Error when the column is missing or ambiguous.

Built-in aliases should be conservative and dataset-scoped. Generic names such
as `DATE`, `ID`, `TYPE`, and undifferentiated `VALUE` are too ambiguous for
defaults unless a specific schema mapping says what they mean. If a source file
contains two aliases for the same normalized column, loading should fail with a
clear error instead of choosing one by priority.

The current implementation honors explicit mappings from referenced schema YAML
files for `portfolio_performance_columns`, `security_performance_columns`, and
`security_master_columns`. For mapped columns, the explicit schema mapping is
authoritative. Built-in aliases remain the fallback for columns not mapped in
the schema file.

Comparison-only datasets such as prices, FX rates, transactions, positions,
and cash currently use the performance-comparison alias/default layer. They do
not require entries in `axys_column_mappings.yaml`.

Inline snapshot-specific schema mappings remain a future step. The current
test fixtures use one referenced Axys column-mapping file plus
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
  period summaries, evidence breakdowns, and evidence rankings.
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
  blocking the evidence-only areas. The coverage total sums already-selected
  cause-area estimates for review context; it is not a residual calculation or
  a complete attribution statement.
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

This is intentionally not a final report format. It gives callers stable
building blocks for a future CSV, Markdown, HTML, or portfolio-period bridge
report without committing the project to one presentation too early.

## Evidence Linking

Some source-input findings can be linked to a portfolio period before any
return-impact estimate is available. This linking improves review grouping; it
does not imply that the system has calculated causal contribution.

Implemented period-linking rules:

- `transactions`, `positions`, and `cash` findings link directly to portfolio
  periods by `portfolio_id` plus their source date. The source date is
  `transaction_date`, `position_date`, or `cash_date`, respectively.
- When more than one configured portfolio period contains the source date, the
  finding links to the narrowest containing period for that portfolio.
- `prices` findings link through `security_performance` when available. A
  price finding for `security_id` and `price_date` links to every
  portfolio-security period containing that date.
- If the same security appears in multiple portfolios for the same price date,
  the price comparison can emit one linked finding per matching
  portfolio-period. This avoids hiding affected portfolios behind an arbitrary
  single match.
- Unmatched dated evidence keeps null period fields.
- `fx_rates` findings intentionally remain unlinked for now. FX linkage needs
  currency exposure context from positions, cash, transactions, portfolio
  currency, or valuation data. A rate row alone is not enough to identify the
  affected portfolio period conservatively.

These rules are intentionally asymmetric. Price rows can be linked through
security-period output because they share a security identifier and date. FX
rates need currency exposure context that is not guaranteed by
`security_performance`.

## Current Limits

The current evidence model is useful, but it should not be overstated.

- Evidence counts are not contribution amounts. A portfolio-period summary can
  say that related price, transaction, position, cash, or security-output
  findings exist; it does not yet calculate how much each item explains of the
  portfolio return delta.
- Portfolio-period evidence rankings are review-priority heuristics. They help
  sort the audit trail but do not quantify causal contribution.
- Prices often lack portfolio identifiers, but they can be linked through
  security-performance periods when `security_performance` is available. FX
  rates often lack both portfolio identifiers and exposure context, so they
  remain unlinked until a conservative exposure linker exists.
- Transaction matching depends on stable keys. With `transaction_id`, changed
  amounts can be reported as changed transactions. Without it, conservative
  fallback matching may report one drop and one add rather than guessing two
  similar rows are the same transaction.
- Changed transaction, position, and cash findings are linked to the narrowest
  configured portfolio performance period for the same portfolio when their
  source date falls inside that period. Unmatched dated evidence findings keep
  null period fields.
- Security master changes are context unless future logic ties them to a
  grouping, reporting, or identifier-resolution effect.
- Security-period summaries are optional. The portfolio-period explanation path
  must continue to work when `security_performance` is absent.
- The implementation compares source evidence. It does not recalculate TWR from
  raw transactions, positions, prices, or cash.

## Contribution Ranking Direction

The existing `rank_portfolio_period_evidence()` helper is a review-priority
sort. It helps decide which findings to inspect first, but it is not a
contribution model and should not be labeled as explaining a portion of return.

A future contribution-ranking layer should be optional, conservative, and
explicit about its basis. The output should be allowed to say "no estimate"
when a finding lacks the denominator, linkage, or methodology needed for a
defensible return-impact estimate.

Implemented contribution-candidate fields:

- `estimated_return_impact`: Optional numeric estimate of return impact.
- `impact_basis`: Short basis label, such as `portfolio_source_field`,
  `security_contribution`, `linked_position_price`, or `no_estimate`.
- `impact_confidence`: High, medium, or low confidence in the estimate.
- `impact_method`: The formula or rule used to estimate impact.
- `impact_message`: Human-readable explanation of the estimate or why no
  estimate was produced.

The contribution-candidate implementation preserves all ranked evidence rows
and populates stable impact columns. It estimates only where the current
evidence carries enough denominator, weight, or vendor output context to state
the method clearly.

Current supported impact estimates:

1. Vendor security contribution delta:
   - `impact_basis = security_contribution`
   - `impact_method = vendor_contribution_delta`
   - `impact_confidence = medium`
   - Uses the vendor-provided contribution delta as a related-output impact
     estimate. This is preferred for the `security_return_or_contribution`
     cause-area aggregate when available.
2. Weighted security return delta:
   - `impact_basis = security_return_weighted`
   - `impact_method = security_return_delta_times_weight`
   - `impact_confidence = low`
   - Formula: `security_return_delta * snapshot_a_weight`.
   - Used as a candidate-level review cross-check or fallback. It is not summed
     with vendor contribution in the same security cause bucket because that
     would double-count two estimates of related security-level performance.
3. Portfolio source-field delta:
   - `impact_basis = portfolio_source_field`
   - `impact_method = source_field_delta_over_begin_market_value`
   - `impact_confidence = low`
   - Formula: `source_field_delta / beginning_market_value`.
   - Currently applies only to return-bearing portfolio source fields such as
     `income` and `gain_loss`. It does not apply to control/output fields such
     as `end_market_value`.

All other rows use `impact_basis = no_estimate` until a defensible method,
denominator, and linkage are available.

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
- Position, price, and transaction evidence: These should receive an estimated
  return impact only when the finding can be linked to an affected portfolio,
  period, security, and denominator. Transaction evidence also requires a
  normalized transaction category plus modeled transaction sign and flow
  semantics. Otherwise these rows should remain ranked review evidence with
  `impact_basis` set to `no_estimate`.
- FX evidence: FX rate changes should not receive a portfolio-period return
  impact unless they can be linked to affected currency exposure, valuation, or
  transactions.

Contribution ranking should not require every finding to receive an estimate.
A mixed output is acceptable: some rows may have `estimated_return_impact`, and
others may remain review-priority evidence only.

Residual findings should wait until the system has a credible contribution
model. Emitting residuals before enough impact estimates exist would imply a
precision the comparison does not yet have.

Reports may still include a residual status section for portfolio-period return
changes. This section should use a section-level caveat plus compact per-period
reason labels, rather than calculating an unexplained amount from partial or
mixed-confidence estimates.

## Near-Term Roadmap

The next design work should focus on explanation quality before adding broad
new datasets.

- Strengthen the contribution-candidate helper only where the math is
  defensible.
- Add a residual concept only after there are enough credible contribution
  estimates. A residual emitted too early would imply precision the system does
  not have.
- Keep report/export formats separate from comparison logic. CSV, Markdown, or
  HTML outputs can be built from the current helper tables later.
- Consider a dataset comparison registry if more datasets are added. The
  current explicit comparison functions are readable; a registry becomes useful
  only when repeated dataset boilerplate starts to obscure the rules.
- Avoid adding new datasets unless real source files expose evidence that is
  not adequately represented by the current normalized datasets.

## Suggested Milestones

### Milestone 1: Portfolio Performance Difference Engine

Status: mostly implemented for the current fixture set.

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

### Milestone 2: Human Report

Status: intentionally deferred.

- Add CSV, Markdown, or HTML output using the current findings helpers.
- Group by portfolio and period.
- Rank largest return and contribution deltas.
- Include suppressed findings appendix.

### Milestone 3: Supporting-File Explanations

Status: implemented at the comparison/evidence-linking level for current
datasets; causal impact estimates remain limited.

- Compare prices for securities with changed returns.
- Compare transactions for affected portfolio/security/period rows.
- Compare positions and cash balances.
- Compare security master fields and classifications.
- Add confidence levels and residual findings.

### Milestone 4: Public API And Demo

Status: partially implemented.

- Add stable public entry points.
- Add sample comparison fixture directories.
- Add script or demo command.
- Document configuration and finding codes.

## Long-Term Dataset Watchlist

The current normalized dataset set already covers the first useful comparison
surface: portfolio performance, security performance, positions, cash,
transactions, prices, FX rates, and security master/reference data. Additional
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
present in existing datasets, such as position cost/accrued values or
transaction quantity, price, and commission.

## Open Design Issues

1. What is the minimum set of fields required to match transactions reliably
   when `transaction_id` is unavailable?
2. Should suppressions require a `reason` field, or remain optional for terse
   operational configs?
3. Should row matching allow fuzzy keys, such as ticker fallback when security
   id changes?
4. Should numeric tolerances be absolute only at first, or support relative
   tolerances too?
5. What is the simplest defensible contribution-ranking model for
   portfolio-period deltas?
6. When contribution ranking exists, when should an unexplained residual be
   emitted?
7. Should this package reuse `PpaError` codes or introduce comparison-specific
   finding codes only?
8. How much of the existing `ppar.axys` inference code should be shared with
   future vendor adapters, and how much should remain Axys-specific?

## Recommended Starting Point

Start with Milestone 1 and keep it boring. The first useful feature is a
trustworthy diff of required `portfolio_performance` rows, enriched by
`security_performance` when it is present, with stable finding codes,
tolerances, and suppressions. Once that foundation is solid, each supporting
file can become an explanation plugin rather than a one-off branch in a large
comparison function.
