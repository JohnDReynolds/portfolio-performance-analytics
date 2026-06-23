# Axys Demo Data

The packaged Axys demo data contains only user-facing demo inputs. Test-only
performance comparison scenarios live under `tests/data/axys`.

## Comparison YAML Files By Role

There are two packaged comparison YAML files. They use the same operational
Mega-Cap source snapshots but review different primary result levels.

| Role | Short name | YAML |
| --- | --- | --- |
| Workbook demo | `portfolio_full_spec` | `ppar_performance_comparison_full_spec.yaml` |
| Workbook demo | `security_full_spec` | `ppar_performance_comparison_security_full_spec.yaml` |

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
It separates performance differences from underlying input differences and
reported checks:

- `Portfolio Differences` sheet: one row per portfolio period with a performance
  difference in the portfolio demo.
- `Security Differences` sheet: one row per security-period return difference in
  the security demo. Review keys include `Portfolio`, `From Date`, `Thru Date`,
  and `Security`.
- `Underlying Causes` sheet: input rows such as positions, transactions, cash, prices,
  and FX rates. `B - A Difference` shows the raw input-value difference, and
  `Performance Difference Explained` appears only when ppar can calculate a
  defensible performance explanation. `Required YAML Setup` is `None` when the
  row is already explainable; otherwise it names the YAML fields or unsupported
  impact method blocking attribution. If a portfolio period has no matching
  input differences, the sheet adds a `no_underlying_cause_found`
  diagnostic row.
- `Reported Performance Checks` sheet: portfolio-performance and
  security-performance rows that confirm reporting differences but are not
  treated as root causes.
- `Context` sheet: review-only supporting rows that are not used to explain return
  differences.
- `Raw Audit Trail` sheet: the underlying finding rows used to build the workbook.

Data used:

- Snapshot A: `axys_full_spec_a`
- Snapshot B: `axys_full_spec_b`
- Files: Axys-style portfolio performance, security performance,
  transactions, positions, prices, cash, and security reference data.
- Scope: one Mega-Cap Alpha operational portfolio, six monthly periods, ten
  mega-cap equities, `CASHBAL`, `TBILL13W`, `TNOTE2Y`, and `TNOTE5Y`.
- YAML: includes every currently supported causal-attribution policy.
- YAML: strict mode is enabled with `--require-causal-attribution`.

Expected workbook:

- Changed portfolio periods in the portfolio demo `Portfolio Differences` sheet.
- Changed AAPL and TNOTE2Y security-period returns in the security demo
  `Security Differences` sheet.
- `Underlying Causes` sheet should show additive transaction amount, position market
  value, position accrued, position quantity, weighted price, and cash examples.
- `Underlying Causes` sheet should also show an explicit evidence-only position
  cost row with `Required YAML Setup` set to `None; configured as evidence-only
  in comparison YAML.`
- `Reported Performance Checks` sheet should show configured
  performance estimates such as security-return weighting and portfolio
  source-field delta. These confirm
  performance-output differences but are not labeled as root-cause input
  differences.
- The controlled restatement includes AAPL price/security-return changes, NVDA
  quantity/market-value/cost changes, TNOTE2Y accrued/security-return changes,
  CASHBAL cash changes, and one AMZN dividend transaction amount change.
- If a future portfolio period has no matching input differences, the
  `Underlying Causes` sheet should add a `no_underlying_cause_found`
  diagnostic row.

Why: this is the most focused workbook for understanding the causal-attribution
model. It keeps the data small and the YAML complete while still distinguishing
underlying input causes from derived performance checks.

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

Start with these decisions:

| Changed data | YAML policy to consider | When to use it | Workbook result |
| --- | --- | --- | --- |
| Portfolio return source fields such as `income` or `gain_loss` | `contribution_impact_methods.portfolio_source_field` | Use when the source field is return-bearing and the beginning market value denominator is appropriate. | Adds `Performance Difference Explained` rows in the `Underlying Causes` sheet. |
| Security contribution | `contribution_impact_methods.security_contribution` | Use when vendor contribution is trusted as the preferred security-level explanation. | Uses vendor contribution delta as the preferred security explanation. |
| Security return plus weight | `contribution_impact_methods.security_return` | Use as a fallback/check when contribution is unavailable and snapshot A weight is the intended weight. | Explains with `security_return_delta_times_weight`. |
| Position market value, accrued, or quantity | `position_impact_methods` | Use when a position field is a reasonable screening explanation and the configured denominator is valid. | Adds low-confidence but additive position explanation rows. |
| Position cost, transaction commission, FX rates, or security master fields | Dataset-specific `evidence_only` or `evidence_only_impact_methods` | Use when the change should be visible to reviewers but should not receive an additive estimate. | Shows the change with `Required YAML Setup` set to no additional setup. |
| Price | `price_impact_methods.price` | Use when price delta over snapshot A price times snapshot A weight is a reasonable screening estimate. | Adds price-driven explanation rows for affected portfolio/security periods. |
| Cash balance or cash market value | `cash_impact_methods` | Use when cash field deltas over the return denominator are a useful screening estimate. | Adds cash explanation rows, usually low confidence. |
| Performance-treated transactions | `transaction_impact_methods.performance` plus `transaction_rules` | Use only when transaction code rules define performance-flow and cash-flow semantics clearly. | Adds transaction amount explanation rows. |
| External-flow transactions | `transaction_impact_methods.external_flow` | Use `evidence_only` for review-only visibility or `modified_dietz` for cross-check diagnostics. | Does not add to explained difference unless a future additive method explicitly supports it. |

If a row in the `Underlying Causes` sheet has blank `Performance Difference
Explained`, the `Required YAML Setup` column should either name the missing YAML
fields or state that no supported additive method exists yet. Do not add a YAML
method just to make the cell non-blank; add it only when the formula matches how
the vendor data should be interpreted.

## Method Coverage Goal

Each supported public YAML impact method should have at least one user-facing
demo example or test-only validation fixture, plus one validator assertion.
Tests can still cover narrow edge cases, but the packaged demos should stay
focused on reviewer-facing workflows.

The supported string vocabulary is summarized in
`docs/performance_comparison_design.md`. The package code backs those strings
with enums, but YAML examples intentionally show the plain string values users
edit.

The `full_spec` fixture is intentionally action-oriented. It contains a
Mega-Cap Alpha operational portfolio with changed portfolio/security periods so
reviewers can see one occurrence of each currently supported
causal-attribution basis:

- security return weighted by beginning weight
- transaction amount over beginning market value
- vendor contribution delta
- portfolio income/gain-loss source-field delta over beginning market value
- position market value over beginning market value
- position accrued over beginning market value
- position quantity times snapshot A unit market value over beginning market value
- price delta over snapshot A price, weighted by snapshot A security weight
- cash balance/market value over beginning market value
- explicit evidence-only treatment for known position cost changes

Current public YAML method targets:

- `contribution_impact_methods.portfolio_source_field`:
  `source_field_delta_over_begin_market_value` (covered)
- `contribution_impact_methods.security_contribution`:
  `vendor_contribution_delta` (covered)
- `contribution_impact_methods.security_return`:
  `security_return_delta_times_weight` (covered)
- `position_impact_methods.market_value`:
  `market_value_delta_over_return_denominator` (covered)
- `position_impact_methods.accrued`:
  `accrued_delta_over_return_denominator` (covered)
- `position_impact_methods.quantity`:
  `quantity_delta_times_snapshot_a_unit_market_value_over_return_denominator`
  (covered)
- `position_impact_methods.quantity`: `evidence_only` (covered)
- `position_impact_methods.cost`: `evidence_only` (covered)
- `price_impact_methods.price`:
  `price_delta_over_snapshot_a_price_times_weight` (covered)
- `cash_impact_methods.cash_balance` and `cash_impact_methods.market_value`:
  `cash_delta_over_return_denominator` (covered)
- `fx_rate_impact_methods.fx_rate`: `evidence_only` (covered)
- `security_master_impact_methods.<reference_or_classification_field>`:
  `evidence_only` for known review-only security reference or classification
  changes (covered)
- `evidence_only_impact_methods.<dataset>`:
  `evidence_only` with explicit `source_fields` for known review-only fields
  without a dataset-specific method target (covered)
- `transaction_impact_methods.external_flow`: `evidence_only` (covered)
- `transaction_impact_methods.external_flow`: `modified_dietz` (covered)
- `transaction_impact_methods.performance`:
  `transaction_amount_delta_over_return_denominator` (covered)
- `transaction_impact_methods.quantity`, `transaction_impact_methods.price`, and
  `transaction_impact_methods.commission`: `evidence_only` (covered)
