# Axys Test Data

This directory contains synthetic Axys-style fixture data used by both the
existing Axys loader tests and the newer performance comparison tests.

The files are intentionally small and controlled. They are designed to exercise
loader, normalization, comparison, suppression, and explanation behavior. They
should not be treated as complete or realistic investment accounting examples.

## Snapshot Directories

- `axys_a`: Baseline snapshot A.
- `axys_b`: Baseline snapshot B. This is intentionally identical to `axys_a`
  for no-difference comparison tests.
- `axys_b_restatement`: Restated snapshot B. This intentionally differs from
  `axys_a` for performance comparison tests.

Each snapshot contains the same basic file set:

- `portperf.csv`: Portfolio performance.
- `secperf.csv`: Security performance.
- `sec_ref.csv`: Security master/reference data.
- `classification_lookup.csv`: Classification lookup data.
- `prices.csv`: Security prices.
- `fx_rates.csv`: FX rate data.
- `transactions.csv`: Transaction activity.
- `positions_holdings.csv`: Position and holding balances.
- `cash.csv`: Cash balances.
- `unreachable_target_secperf.csv`: Security performance data used for
  specific validation behavior.

`fx_currency.csv` is retained for Axys fixture compatibility. The performance
comparison feature currently uses `fx_rates.csv` for FX rate comparisons.

## YAML Files

- `axys_column_mappings.yaml`: Shared Axys column mapping configuration. This
  describes how Axys source columns map to normalized internal column names.
- `ppar_performance_comparison.yaml`: Baseline performance comparison config.
  It compares `axys_a` to `axys_b` and should produce no findings.
- `ppar_performance_comparison_restatement.yaml`: Restatement comparison
  config. It compares `axys_a` to `axys_b_restatement`.
- `ppar_performance_comparison_restatement_transaction_rules.yaml`:
  Restatement comparison config with YAML `transaction_rules` that supply
  transaction sign/flow semantics missing from the Axys transaction CSVs, plus
  an explicit external-flow evidence-only impact policy.
- `ppar_performance_comparison_suppressed.yaml`: Restatement comparison config
  with a suppression rule applied. It should still preserve the full audit
  trail while excluding suppressed findings from active-output helpers.

Relative paths in these YAML files resolve relative to the YAML file location.
Snapshot file paths then resolve relative to each configured snapshot directory.

## Intentional Restatement Themes

`axys_b_restatement` intentionally changes multiple evidence families so the
comparison layer can exercise the current finding model:

- portfolio return and portfolio performance source fields
- security return, weight, and contribution
- security add/drop behavior
- position quantity, market value, and accrued amount
- cash balance and cash market value
- security price
- FX rate
- transaction quantity, price, and amount
- security master reference and classification fields

See `axys_b_restatement/RESTATEMENT_NOTES.md` for the controlled row-level
changes.

## Useful Commands

Run the focused performance comparison runner tests:

```bash
python -m unittest tests.test_performance_comparison_runner
```

Run the full test suite:

```bash
python -m unittest discover -s tests
```

Run the performance comparison demo:

```bash
python -m ppar.demos.performance_comparison_demo
```
