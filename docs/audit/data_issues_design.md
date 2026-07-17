# Data Issues Design

The `Data Issues` worksheet surfaces source-data consistency problems that
are useful to reviewers but are not themselves performance attribution rows.
Data Issues is a sibling of Performance Comparison within Audit. Its canonical
YAML key is `data_issues` because these checks cross-reference related
source-data fields.

The canonical implementation lives under `ppar.audit.data_issues`:

- `DataIssueType` defines the stable machine-readable issue identifiers;
- `DataIssueCategory` defines reviewer groupings;
- `DataIssueDefinition` captures the contract for one check; and
- `DATA_ISSUE_REGISTRY` binds each implemented issue to its category,
  enablement, required datasets, tolerances, and reviewer meaning.

Purpose:

> Flag source-data relationships that should usually agree, reconcile, or move
> together, without treating those issues as additive Modified Dietz causes.

This worksheet is deliberately separate from both primary evidence sheets:

- `Performance Difference Causes` explains changed performance.
- `source_detail.csv` records row-level A-versus-B differences.
- `Data Issues` reports consistency checks across source-data relationships,
  whether or not those checks explain a performance difference.

Beginning/ending market-value continuity is a mandatory financial-integrity
check at portfolio and security grain. A mismatch remains visible in this
worksheet even when optional Data Issues checks are disabled because both values
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
data_issues:
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

- `data_issues.enabled`: master switch for optional consistency checks;
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

This section is a strict fail-closed contract. The comparison specification is
rejected before source loading or report generation when:

- `data_issues` or a configured issue type is not a mapping;
- an issue type or per-check key is unknown;
- `enabled` is not an actual YAML Boolean;
- a tolerance is Boolean, nonnumeric, nonfinite, or negative; or
- `only`/`exclude` is not a mapping, names an unsupported normalized field, or
  contains an empty or nonscalar value.

Supported filter fields are `snapshot`, `portfolio`/`portfolio_id`,
`security`/`security_id`, `security_type`, `asset_class`, and
`transaction_code`. Dataset-qualified forms continue to resolve by the name
after the final dot. Scalar and list values are compared as case-insensitive
strings.

The seven optional checks support `enabled`, `only`, and `exclude`. Numeric
range/rate checks additionally support both tolerance keys. Duplicate and
missing-dividend checks do not accept unused tolerance keys. The two mandatory
continuity blocks accept only absolute and percent tolerances; they cannot be
disabled or filtered. `validate_config` prints the effective optional checks,
mandatory continuity checks, and master-switch policy.

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

Data Issues issue types use compact codes such as `duplicate_transactions` and
`transactions_price_range`. The worksheet intentionally avoids a severity column;
the reviewer-facing explanation, values, and tolerance carry the useful context
without implying false precision.

Data Issues rows are nonblocking review evidence. Any future blocking policy
requires real-world evidence and an explicit product decision; it is not an
implied extension of the current YAML contract.

Data Issues findings should not change:

- `Performance Difference Explained`;
- `Unexplained Difference`;
- `Performance Difference Causes`;
- `source_detail.csv` row counts.

They may be referenced from README/handoff text as a separate review surface.
