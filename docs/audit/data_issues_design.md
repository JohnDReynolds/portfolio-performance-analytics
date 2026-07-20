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
Category
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
      transactions.security_type: csus
      security_reference.asset_class_code: EQ
    exclude:
      portfolio_id:
        - TEST_PORTFOLIO
    absolute_tolerance: 0.01
    percent_tolerance: 0.50

  large_price_variation:
    enabled: true
    rules:
      - rule_id: common_stock_20_percent
        only:
          transactions.transaction_code: [by, sl]
          security_reference.security_type: csus
        minimum_calendar_days: 1
        minimum_tolerance: 0.20

  deliver_in_original_cost_incomplete:
    enabled: true
    only:
      transaction_code: ti
      security_type: csus
      source_destination_type: $pty
      source_destination_symbol: external_delivery
```

Interpretation:

- `data_issues.enabled`: master switch for optional consistency checks;
  mandatory beginning/ending continuity findings remain active.
- The seven established optional issue types are enabled by default when the
  worksheet is enabled. Set `enabled: false` under one issue type to opt out.
- Conservative opt-in issue types declare that policy in the registry.
  `holdings_nonpositive_price` requires `enabled: true` and a nonempty `only`
  population. `transactions_nonpositive_price` additionally requires an exact
  transaction-code population and either `security_reference.security_type` or
  `security_reference.asset_class_code` in `only`.
  `transaction_security_type_mismatch` requires the exact
  `security_reference.security_type` population it compares against.
  `holdings_stale_price` also requires that reference population plus an
  explicit positive integer `minimum_calendar_days` threshold.
  `deliver_in_original_cost_incomplete` requires transaction code, security
  type, source/destination type, and source/destination symbol in `only`, plus
  both original-cost source columns in each snapshot. Its native filters honor
  an exact-case transaction source contract.
- `large_price_variation` is off by default and uses an issue-specific nonempty
  `rules` list. Every rule has a unique lowercase snake-case `rule_id`; optional
  rule-level `enabled`, `only`, `exclude`, `minimum_calendar_days`, and
  `minimum_tolerance` settings do not become controls for unrelated checks.
- `only`: optional exact-match include filters. A row must match every listed
  field to enter the check.
- `exclude`: optional exact-match exclude filters. A row is dropped when it
  matches any listed field.
- Filter values can be scalars or lists. Field names may be common normalized
  names such as `security_id`, `security_type`, and `portfolio_id`, or
  dataset-qualified names such as `holdings.security_type` and
  `transactions.transaction_code`. Optional security-master qualifiers use the
  explicit `security_reference.*` namespace.
- Within one `large_price_variation` rule, values in a filter list are OR,
  different `only` fields are AND, and any matching `exclude` field removes an
  observation. `holdings.*` and `transactions.*` filters apply only to that
  observation source, so a transaction-code population does not discard the
  beginning and ending holdings prices. Common entity and
  `security_reference.*` filters apply to both sources.
- Tolerances stay per issue type because noisy fields need different limits.

This section is a strict fail-closed contract. The comparison specification is
rejected before source loading or report generation when:

- `data_issues` or a configured issue type is not a mapping;
- an issue type or per-check key is unknown;
- `enabled` is not an actual YAML Boolean;
- a tolerance is Boolean, nonnumeric, nonfinite, or negative; or
- `only`/`exclude` is not a mapping, names an unsupported normalized field, or
  contains an empty or nonscalar value; or
- a conservative opt-in check is enabled without its required nonempty `only`
  population; or
- `large_price_variation` has missing, duplicate, or malformed rule IDs, a
  malformed/nonempty-rules violation, an unknown rule key or dataset namespace,
  a nonpositive calendar-day minimum, or an invalid decimal tolerance.

Supported native filter fields are `snapshot`, `portfolio`/`portfolio_id`,
`security`/`security_id`, `security_type`, `asset_class`, `transaction_code`,
`source_destination_type`, and `source_destination_symbol`. Dataset-qualified
forms continue to resolve by the name after the final dot. Scalar and list
values retain the established case-insensitive comparison behavior except when
the enabled original-cost check runs under an exact-case transaction source
contract; that check then compares all native population fields by exact case.

When `files.security_reference` is configured, a Data Issues filter may also
use `security_reference.security_name`, `ticker`, `cusip`, `isin`,
`security_type`, `asset_class_code`, `asset_class_name`, `sector_code`,
`sector`, `country_code`, `country`, or `currency`. These fields are a separate
reference namespace: `transactions.security_type` means the value carried by a
transaction row, while `security_reference.security_type` means the value
joined from `secmast.csv` for the same snapshot.

Reference joins and reference-filter values preserve exact source case. This is
deliberately stricter than the compatibility behavior for established native
row filters. A configured reference filter fails closed when either snapshot
lacks `files.security_reference`, the referenced column is absent, a relevant
source security has no exact-case reference row, a required reference value is
blank, or an exact-case security identifier is duplicated. The reference data
only qualifies Data Issues populations; it does not enter Modified Dietz,
change performance findings, or create source-detail explanations.

The normalized `security_id` is the join key and must be unique within each
reference snapshot. This is a PPAR input contract, not a claim that symbol alone
is the native Axys/APX key. Axys/APX YAML can declare exact source components;
the starter uses `Security Type` followed by `Security Symbol` and constructs
compact values such as `csusAAPL`. Sites may configure an optional separator
such as `_`. PPAR rejects blank or whitespace-padded components and ambiguous
constructed values rather than guessing between duplicate symbols. Axys/APX
types are typically four characters, but PPAR preserves the source value instead
of enforcing an unverified fixed length.

Twelve row-level optional checks support `enabled`, `only`, and `exclude`. Numeric
range/rate checks additionally support both tolerance keys. Duplicate,
missing-dividend, nonpositive-price, and classification-mismatch checks do not
accept unused tolerance keys. Only `holdings_stale_price` accepts
`minimum_calendar_days`; it must be a positive, non-Boolean integer. The two
mandatory continuity blocks accept only absolute and percent tolerances; they
cannot be disabled or filtered. `validate_config` prints the effective optional
checks, mandatory continuity checks, and master-switch policy.

The thirteenth optional check, `large_price_variation`, accepts only top-level
`enabled` and `rules`. Rule IDs are output provenance and part of the review
key. Rule defaults are one inclusive calendar day and decimal tolerance `0.20`.
Rule order is canonicalized by ID so reordering YAML does not change output.

### Initial Checks

Implemented checks should stay high signal, available from the current
normalized datasets, and easy to explain:

1. `holdings_nonpositive_price`: for each snapshot row in an explicitly
   configured `only` population, flag a finite `holdings.price` less than or
   equal to zero when finite holding quantity is nonzero. The check is off by
   default, uses the fixed condition `price > 0`, and does not treat a missing
   price or zero quantity as this issue.
2. `transactions_nonpositive_price`: for each snapshot row in an explicitly
   configured transaction-code and reviewed reference population, flag a finite
   `transactions.price` less than or equal to zero when finite transaction
   quantity is nonzero. The rule does not infer that a configured transaction
   code is universally price-bearing.
3. `transaction_security_type_mismatch`: for each transaction row in an
   explicitly configured `security_reference.security_type` population,
   compare the transaction and snapshot-reference types using exact source
   case. The issue reports both text values in its explanation, distinguishes
   case-only differences, and does not choose which classification is correct.
4. `holdings_stale_price`: within each snapshot, portfolio, and security in an
   explicitly configured reference population, track an uninterrupted run of
   supplied nonzero-quantity observations with the same positive price. Flag a
   current observation once the run spans at least `minimum_calendar_days`.
   The explanation names the first and current supplied dates and states that
   PPAR did not observe every intervening day.
5. `large_price_variation`: within each snapshot, portfolio, established
   performance period, security, and enabled named rule, combine the ending
   holding price from the immediately linked prior period, the current period's
   ending holding price, and positive eligible transaction prices whose
   normalized trade date is within the current inclusive boundaries. Missing
   boundary holdings are allowed, but at least two comparable positive
   observations are required. Period length is
   `(thru_date - from_date).days + 1`; periods shorter than the rule minimum are
   discarded. Split factors with `observation_date < split_date <= thru_date`
   divide the earlier raw price so all observations use the period-ending share
   basis. A split-date transaction is treated as post-split. Conflicting
   same-date factors and nonpositive factors fail closed. The maximum variation
   is `(maximum adjusted price - minimum adjusted price) / minimum adjusted
   price`; equality with the tolerance does not report. Conflicting nonblank
   price currencies are not compared. Multiple matching rules each emit one
   independently identified row.
6. `deliver_in_original_cost_incomplete`: for each transaction in the explicit
   transaction-code, security-type, and source/destination population, report
   one review row when `transactions.original_cost`,
   `transactions.original_cost_date`, or both are absent. Zero original cost is
   present rather than missing. Enabling the check requires both source columns
   in both snapshots. The check does not calculate cost basis, classify a code
   by itself, or conclude that a source-system fallback occurred.
7. `holdings_price_range`: for each snapshot, security, and holding date,
   compare same-day same-security `holdings.price` values across portfolios.
8. `transactions_price_range`: for each snapshot, security, and transaction
   date, compare same-day same-security `transactions.price` values across
   portfolios.
9. `duplicate_transactions`: for each snapshot, flag exact duplicate
   transaction rows with the same portfolio, date, security, code, amount,
   quantity, and price.
10. `dividend_rate`: for each snapshot, security, and dividend date, compare
   same-day same-security dividend rates across portfolios.
11. `missing_dividend`: for each snapshot, security, and dividend date where at
   least one portfolio has a dividend, flag other portfolios that
   conservatively appear eligible for that dividend. A portfolio qualifies only
   when it has positive beginning-period quantity, or positive buy activity
   before the dividend date, and has no pre-dividend transaction activity other
   than buys.
12. `pa_sa_rate`: for each snapshot, security, and transaction date, compare
   same-day same-security purchase-accrued and sale-accrued rates across
   portfolios.
13. `holdings_accrued_rate`: for each snapshot, security, and holding date,
   compare same-day same-security `holdings.accrued` per unit across
   portfolios.

The nonpositive-price checks are intentionally conservative. Worthless
securities, cash conventions, accrued-only rows, shorts, derivatives, and
vendor-specific valuation representations can legitimately carry zero or
negative prices. Sites should include only a population where positive price is
an understood source-data requirement, and should use `exclude` for known
exceptions. A finding means “review this valuation or population,” not “the
security or transaction is conclusively mispriced.” Transaction checks must
also name the site-reviewed transaction-code population; the security reference
qualifies that population but does not assign transaction semantics.

The stale-price check is also a review signal rather than a market-data
conclusion. It sees only supplied holding observations. It does not establish
that the price remained unchanged on every intervening market day, and the same
price may legitimately recur. Missing, nonpositive, and zero-quantity
observations break an unchanged run rather than being silently skipped.

The large-variation check is likewise a review signal. Missing split evidence
can leave a mechanical raw-price discontinuity visible, but the finding does
not conclude that a split is missing or that a legitimate market move is bad.
The explanation identifies the named rule, inclusive period, selected minimum
and maximum source/date evidence, adjusted prices, and any cumulative split
factor. Ties select the earliest date, then holdings before transactions, then
original source order. The absolute `Difference` remains `Observed Value -
Reference Value`; the explanation carries the percentage variation.

The original-cost completeness check is a source-evidence review, not a
valuation conclusion. The normalized original-cost fields are optional inputs
unless this check is enabled. They do not enter performance calculations or add
columns to any user-facing report; findings use the existing Data Issues
schema.

The optional security-reference dataset makes carefully scoped classification
qualifiers available now. Continue to defer checks that require additional
economic attributes, event histories, or ledger evidence:

- paydown amount/rate checks;
- bond accrued-interest expectation checks;
- split factor versus quantity-jump plausibility;
- cash roll-forward checks.

Those deferred checks are valuable, but they need dividend-rate, pool-factor,
coupon/accrual, event-history, or cash-ledger data that is not yet part of the
normalized source contract. A present-day classification value alone is not
enough to infer those economics.

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
