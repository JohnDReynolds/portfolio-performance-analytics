# Axys Demo Data

The packaged Axys demo data contains only user-facing demo inputs. Test-only
performance comparison scenarios live under `tests/data/axys`.

## Comparison YAML

There is one packaged comparison YAML file. The portfolio and security demo
commands use the same operational Mega-Cap source snapshots and choose the
primary review level at runtime.

The packaged CSV files follow the
[Performance Comparison Demo Source Contract](../../../../docs/performance_comparison_demo_source_contract.md).
They are normalized demo extracts, not official Axys/APX native schemas.

| Role | YAML |
| --- | --- |
| Workbook demos | `ppar_performance_comparison.yaml` |

## Recommended User-Facing Demo

Run the packaged portfolio demo when you want the portfolio-period reviewer-facing
example:

```bash
./.venv/bin/python -m ppar.demos.performance_comparison_portfolio_demo
```

Output:

- `_demo_output/performance_comparison_portfolio/report.xlsx`
- `_demo_output/performance_comparison_portfolio/report.html`
- `_demo_output/performance_comparison_portfolio/manifest.json`
- `_demo_output/performance_comparison_portfolio/*.csv`

Run the packaged security demo when you want the security-period reviewer-facing
example:

```bash
./.venv/bin/python -m ppar.demos.performance_comparison_security_demo
```

Output:

- `_demo_output/performance_comparison_security/report.xlsx`
- `_demo_output/performance_comparison_security/report.html`
- `_demo_output/performance_comparison_security/manifest.json`
- `_demo_output/performance_comparison_security/*.csv`

Start review in `report.xlsx`. Use `report.html` when you want the same review
model in a browser. The report is designed for review, not for raw data export.
It separates performance differences from identifiable input differences and
other evidence:

- `Performance Differences` sheet: one row per portfolio period with a performance
  difference in the portfolio demo.
- `Performance Differences` sheet: one row per security-period return difference in
  the security demo. Review keys include `Portfolio`, `From Date`, `Thru Date`,
  and `Security`.
- `Performance Difference Causes` sheet: input rows such as holdings, transactions,
  and FX rates. `B - A Difference` shows the raw input-value difference, and
  `Performance Difference Explained` appears only when ppar can calculate a
  defensible performance explanation.
- Optional reconstruction diagnostics can add `Return Reconstruction Checks`,
  `Security Return Checks`, and `Reconstruction Summary` sheets for
  implementation review, but normal demo output excludes them by default.
- Evidence-only input rows normally appear in the `Other Data Differences` sheet. If a
  portfolio or security period still has an unexplained difference, plausible
  evidence-only input rows for that same review key are promoted into the
  `Performance Difference Causes` sheet with blank `Performance Difference Explained`
  values so the likely explanation is visible in one place.
- `Other Data Differences` sheet: review-only supporting rows that are not used to
  explain return differences and are not needed to review an unresolved period.
  Transaction quantity, price, and commission rows usually live here as
  supporting evidence for changed `transactions.amount`.
- `Raw Audit Trail` sheet: the underlying finding rows used to build the workbook.

Data used:

- Snapshot A: `axys_full_spec_a`
- Snapshot B: `axys_full_spec_b`
- Files: Axys-style portfolio performance, security performance,
  transactions, holdings, and security reference data.
- Scope: three operational portfolios (`ALPHA`, `BALANCED`, and `INCOME`), six
  monthly periods, ten mega-cap equities, `CASH_USD`, `TBILL13W`, `TNOTE2Y`, and
  `TNOTE5Y`. `ALPHA` is the closest match to the Mega-Cap Alpha analytics
  portfolio; `BALANCED` and `INCOME` reuse the same securities with larger
  cash/fixed-income sleeves.
- YAML: includes transaction semantics; standard field roles supply the common
  performance-input, input-component, and context treatment.
- YAML: maps source transaction codes (`by`, `sl`, `dv`, `in`, `dp`, `wd`, and
  `;`) to normalized categories such as `buy`, `sell`, `income`, `fee_expense`,
  `external_flow`, and `corporate_action`. Reviewer-facing explanations preserve
  the source code rather than uppercasing or replacing it with the category.
- YAML: includes explicit `portfolio_return_reconstruction` settings for
  Modified Dietz diagnostic checks.
- YAML: includes explicit `security_return_reconstruction` settings for
  security-level Modified Dietz diagnostic checks.
- YAML: treats fee-like `dp` transactions as performance-impacting because this packaged
  fixture assumes the reported returns are net of fees. For gross-of-fees
  performance, fees would need a different return-basis policy.
- YAML: strict causal-attribution mode is intentionally not enabled for the
  portfolio demo because one period is meant to show a review-only unexplained
  case.

Expected workbook:

- Changed ALPHA, BALANCED, and INCOME periods in the portfolio demo
  `Performance Differences` sheet.
- Changed security-period returns in the security demo `Performance Differences`
  sheet. These include clean explainable rows and explicit unresolved rows when
  a security-performance difference has no additive input cause.
- `Performance Difference Causes` sheet should show understandable additive transaction
  amount, holding market value, holding accrued, and weighted price examples.
  It should also show plausible input-component rows for unresolved periods
  without assigning them `Performance Difference Explained` values.
- Portfolio-period differences should include fully explained and intentionally
  unexplained statuses.
- Optional reconstruction diagnostics should show where source-derived Modified
  Dietz returns agree with reported return differences and where they do not.
- The controlled restatement includes:
  - a fully explained ALPHA period with AAPL price/security-return changes and
    `CASH_USD` holding changes;
  - a partly explained ALPHA period with a changed buy transaction amount plus
    changed transaction quantity, price, and commission support rows in
    `Other Data Differences`;
  - a fully explained BALANCED period with a dividend transaction amount change;
  - a fully explained INCOME period with a larger advisory-fee expense and
    matching lower `CASH_USD` ending value;
  - a fully explained BALANCED period with both an AAPL price correction and a
    standalone MSFT holding market-value correction;
  - a fully explained INCOME period with the same AAPL price correction plus
    TNOTE2Y market-value and accrued-interest changes, related TNOTE2Y quantity
    evidence, and TNOTE2Y cost in the `Other Data Differences` sheet;
  - an unexplained INCOME period with no identifiable cause found;
  - an ALPHA external-withdrawal restatement visible in the return
    reconstruction check.

Why: this is the most focused workbook for understanding the causal-attribution
model. It keeps the data small and transaction semantics explicit while still
distinguishing identifiable input causes from reported-performance diagnostics.

After generating the workbook demo bundle, validate it with:

```bash
./.venv/bin/python -m ppar.performance_comparison.cli.validate_bundle \
  _demo_output/performance_comparison_portfolio
./.venv/bin/python -m ppar.performance_comparison.cli.validate_bundle \
  _demo_output/performance_comparison_security
```

## YAML Policy Decision Guide

Use the YAML policy blocks to state what ppar is allowed to treat as an
explanation. The values are intentionally explicit because different vendors
can use the same-looking fields with different sign, timing, denominator, or
accounting conventions.

The workbook uses a small field-role model:

| Role | Typical fields | Workbook treatment |
| --- | --- | --- |
| `performance_input` | `holdings.market_value`, `holdings.accrued`, `transactions.amount` | Additive rows on the `Performance Difference Causes` sheet when enough inputs are available. |
| `input_component` | `holdings.quantity`, `holdings.price`, transaction quantity/price/commission | Shown beside related performance inputs when useful, or kept in `Other Data Differences` as support for the related performance input. |
| `reported_performance_component` | portfolio/security performance return, income, gain/loss, contribution, weight, market value | Kept as reporting diagnostics in the audit trail; not treated as root-cause input differences. |
| `context` | holding cost, FX rates, security reference data, unsupported fields | Shown on the `Other Data Differences` sheet unless promoted for review of an unresolved period. |

Missing transaction semantics are still a hard stop for user-facing bundle
generation because transaction amount attribution depends on transaction-code
classification. Blank `Performance Difference Explained` cells mean the row is
review-only or not currently additively estimated, not that YAML setup is
missing.

## Method Coverage Goal

Each supported public YAML impact method should have at least one user-facing
demo example or test-only validation fixture, plus one validator assertion.
Tests can still cover narrow edge cases, but the packaged demos should stay
focused on reviewer-facing workflows.

The supported string vocabulary is summarized in
`docs/performance_comparison_design.md`. The package code backs those strings
with enums, but YAML examples intentionally show the plain string values users
edit.

The portfolio fixture is intentionally action-oriented. It contains ALPHA,
BALANCED, and INCOME operational portfolios with changed portfolio/security
periods so reviewers can see the most understandable causal-attribution bases:

- transaction amount over beginning market value
- holding market value over beginning market value
- holding accrued over beginning market value
- price delta over snapshot A price, weighted by snapshot A security weight

Most portfolio-period differences are fully explained. Some periods are partly
explained when visible source-data changes account for only part of the
Performance Difference. One INCOME period is intentionally unexplained and gets
the `no_underlying_causes_found` diagnostic row so the reviewer has an explicit
place to continue review.

The derived `secperf.csv` and `portperf.csv` files are kept aligned with:

```bash
./.venv/bin/python scripts/operational_demo_data/rebuild_performance_comparison_demo_data.py
```

The default mode audits the checked-in files without writing. Add `--write`
after intentional fixture edits to recompute security beginning weights,
security contributions, and portfolio performance rows from security
performance rows.

Current public YAML targets are intentionally narrow:

- `transaction_rules`: classifies transaction codes for amount attribution.
- `transaction_amount_delta_over_return_denominator`: default amount-impact
  method used after `transaction_rules` mark a transaction code as
  performance-affecting.
- `transaction_impact_methods.external_flow`: optional `modified_dietz`
  cross-checks for external-flow transactions.
- suppression rules: remove known, intentionally ignored differences from the
  active review while retaining them in the raw audit trail.
