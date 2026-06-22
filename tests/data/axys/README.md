# Axys Test Configuration Fixtures

This directory contains test-only performance comparison YAML files. The Axys
CSV snapshots and shared column mapping schema live in the packaged demo data at
`ppar/demos/data/axys`.

Keeping this directory small avoids maintaining duplicate source-data snapshots
while preserving focused YAML scenarios that tests can mutate or reference
without changing the user-facing demo files.

## Files

- `ppar_performance_comparison.yaml`: Baseline performance comparison config.
  It compares packaged `axys_a` to packaged `axys_b` and should produce no
  findings.
- `ppar_performance_comparison_restatement.yaml`: Controlled restatement config.
  It compares packaged `axys_a` to packaged `axys_b_restatement`.
- `ppar_performance_comparison_restatement_transaction_rules.yaml`: Restatement
  config with explicit transaction rules and impact methods.
- `ppar_performance_comparison_suppressed.yaml`: Restatement config with a
  suppression rule applied for suppression-specific tests.

The YAML paths intentionally point back to `ppar/demos/data/axys` so broad tests
and demos share the same Axys snapshots.
