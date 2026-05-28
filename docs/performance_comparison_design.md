# Performance Comparison Design Notes

## Purpose

The performance comparison feature explains why calculated performance for the
same portfolio and period changed between two source-data extraction dates.

The core question is:

> Why did my performance for period A change from when I ran it on date 1 to
> when I ran it on date 2?

The feature will compare two snapshot directories. Each snapshot contains
vendor exports such as portfolio performance, security performance, prices,
transactions, positions, cash, and security master/reference files.

This should not be treated as an Axys-only feature. The comparison engine
should operate on normalized internal datasets. Vendor-specific behavior should
live in small normalization adapters, with Axys as the first likely adapter.

The first implementation should stay intentionally narrow: compare portfolio
and security performance rows, report material return changes, and produce a
clear finding model. Explanations from prices, transactions, holdings, cash, and
security master files can be added after the finding model is stable.

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

## Proposed Package Layout

```text
ppar/performance_comparison/
  __init__.py
  specification.py
  snapshot.py
  adapters.py
  compare.py
  findings.py
  rules.py
  explain.py
  report.py
```

Possible responsibilities:

- `specification.py`: Read and validate comparison YAML.
- `snapshot.py`: Load one snapshot directory and normalize file paths.
- `adapters.py`: Define source-system adapter interfaces.
- `compare.py`: Compare normalized snapshot A/snapshot B data sets.
- `findings.py`: Define finding records, severity, confidence, and codes.
- `rules.py`: Apply tolerances, materiality, filters, and suppressions.
- `explain.py`: Add likely-cause explanations from supporting files.
- `report.py`: Render DataFrame, CSV, JSON, or HTML outputs.

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
engine. Initial normalized datasets should include:

- `portfolio_performance`
- `security_performance`
- `security_master`

Later datasets can include:

- `prices`
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

`transactions` required columns:

- `portfolio_id`
- `security_id`
- `transaction_date`

`transactions` useful optional columns:

- `transaction_id`
- `settlement_date`
- `transaction_code`
- `quantity`
- `price`
- `amount`
- `commission`
- `currency`
- `broker`

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
- Transactions: ideally a transaction id; otherwise a composite fallback.
- Positions: portfolio code, security identifier, position date.

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
- `source_file`: Source file associated with the finding.
- `source_column`: Source column associated with the finding.
- `message`: Human-readable explanation.
- `suppressed`: Whether a suppression rule hid the finding from normal output.

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
PC-PRICE-BEG    Beginning price changed
PC-PRICE-END    Ending price changed
PC-TXN-ADD      Transaction appears only in snapshot B
PC-TXN-DROP     Transaction appears only in snapshot A
PC-TXN-AMT      Transaction amount changed
PC-POS-QTY      Position quantity changed
PC-POS-MV       Position market value changed
PC-CASH-MV      Cash balance or cash market value changed
PC-REF-ID       Security identifier/reference field changed
PC-REF-CLASS    Security classification changed
PC-RESIDUAL     Unexplained residual remains
```

Codes should be stable once public. New detail can be added through fields
rather than by renaming codes.

## Configuration

The existing Axys loader configuration and the new comparison configuration
serve different purposes and should have distinct names:

- `ppar_axys.yaml`: Describes how to load one Axys source set.
- `ppar_performance_comparison.yaml`: Describes how to compare two snapshots.

A comparison probably needs one YAML file for the comparison run, not a separate
YAML file inside each snapshot. The comparison YAML can point at both snapshot
directories, define shared rules, and optionally reference vendor schema files
such as `ppar_axys.yaml`.

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
4. Relative paths inside a referenced schema YAML, such as `ppar_axys.yaml`,
   resolve relative to that schema YAML file.

A suggested project layout is:

```text
comparisons/
  ppar_performance_comparison.yaml
  ppar_axys.yaml

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
    schema: ppar_axys.yaml

  b:
    label: run_2026_05_15
    path: snapshots/2026-05-15
    vendor: axys
    schema: ppar_axys.yaml

files:
  portfolio_performance: portperf.csv
  security_performance: secperf.csv
  security_master: sec_ref.csv
  prices: prices.csv
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

materiality:
  minimum_return_delta: 0.000001
  minimum_market_value_delta: 0.01

suppressions:
  - code: PC-SEC-RET
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
3. Referenced vendor schema file, such as `ppar_axys.yaml`.
4. Built-in default aliases.
5. Error when the column is missing or ambiguous.

The first implementation can support the simple shared-schema case first, but
the configuration shape should not block snapshot-specific schemas later.

## Suppression And Filtering

Suppression rules should be explicit and auditable. They should not delete
findings; they should mark findings as suppressed so a full audit appendix can
still show what was hidden.

Potential filter/suppression fields:

- `code`
- `portfolio_id`
- `security_id`
- `from_date`
- `thru_date`
- `source_file`
- `source_column`
- numeric threshold overrides
- regular-expression matching for portfolio/security identifiers
- required `reason`

Open question: should suppressions be exact-match only at first? Exact matching
is easier to audit and less surprising.

## Suggested Milestones

### Milestone 1: Portfolio Performance Difference Engine

- Read comparison YAML.
- Load two snapshot directories.
- Normalize required `portfolio_performance` and any available optional
  `security_performance` source using built-in inference where practical.
- Compare rows by configured keys.
- Emit findings for added/dropped rows and changed portfolio/security returns,
  weights, and contributions.
- Apply tolerances and suppressions.
- Return findings as a Polars DataFrame.
- Add CSV or JSON output.

### Milestone 2: Human Report

- Add compact text/HTML summary.
- Group by portfolio and period.
- Rank largest return and contribution deltas.
- Include suppressed findings appendix.

### Milestone 3: Supporting-File Explanations

- Compare prices for securities with changed returns.
- Compare transactions for affected portfolio/security/period rows.
- Compare positions and cash balances.
- Compare security master fields and classifications.
- Add confidence levels and residual findings.

### Milestone 4: Public API And Demo

- Add stable public entry point.
- Add sample comparison fixture directories.
- Add script or demo command.
- Document configuration and finding codes.

## Open Design Issues

1. Should output be organized around findings, around portfolio-period bridges,
   or both?
2. What is the minimum set of fields required to match transactions reliably?
3. How should duplicate rows with the same comparison key be handled?
4. Should comparison tolerate missing supporting files, or treat them as
   blocking errors?
5. Should suppressions require a `reason` field?
6. Should row matching allow fuzzy keys, such as ticker fallback when security
   id changes?
7. Should numeric tolerances be absolute only at first, or support relative
   tolerances too?
8. Should an unexplained residual always be emitted when explanations do not
   account for a return delta?
9. Should this package reuse `PpaError` codes or introduce comparison-specific
    finding codes only?
10. How much of the existing `ppar.axys` inference code should be shared with
    future vendor adapters, and how much should remain Axys-specific?

## Recommended Starting Point

Start with Milestone 1 and keep it boring. The first useful feature is a
trustworthy diff of required `portfolio_performance` rows, enriched by
`security_performance` when it is present, with stable finding codes,
tolerances, and suppressions. Once that foundation is solid, each supporting
file can become an explanation plugin rather than a one-off branch in a large
comparison function.
