# Axys Test Fixtures

This directory contains test-only Axys CSV snapshots, a test-only Axys column
mapping schema, and performance comparison YAML files. User-facing demo data
lives under `ppar/demos/data/axys_performance_comparison`.

Keeping the synthetic fixtures here avoids mixing regression scenarios with
the user-facing demo files.

## Shared Test Files

- `axys_column_mappings.yaml`: Test schema for the synthetic Axys snapshots.
- `snapshots/axys_a`: Clean baseline snapshot.
- `snapshots/axys_b`: Clean comparison snapshot with no expected findings.
- `snapshots/axys_b_restatement`: Controlled single-restatement snapshot.
- `snapshots/axys_b_multi_restatement`: Multi-portfolio restatement snapshot
  with wider scenario coverage.
- `snapshots/axys_modified_dietz_a`: Modified Dietz baseline snapshot.
- `snapshots/axys_modified_dietz_b`: Modified Dietz comparison snapshot.

## Comparison YAML Files

The `validation/` directory is the single home for test-only performance
comparison YAML scenarios. These files are used by targeted unit tests and by
`ppar.performance_comparison.cli.validate_demo_matrix`. They are not
user-facing demos; they exist to keep specific edge cases covered without
asking reviewers to inspect extra workbooks.

- `validation/ppar_performance_comparison.yaml`: Tests the clean/no-issue
  control case and proves the comparison
  can run without producing false positives.
- `validation/ppar_performance_comparison_restatement.yaml`: Tests missing
  transaction setup guidance on a controlled restatement.
- `validation/ppar_performance_comparison_restatement_transaction_rules.yaml`:
  Tests that transaction amount rows become explainable when YAML supplies
  transaction rules and impact methods.
- `validation/ppar_performance_comparison_security_restatement.yaml`: Tests
  security-level result comparison with security review keys.
- `validation/ppar_performance_comparison_multi_restatement.yaml`: Stress-tests
  multiple portfolios, multiple periods, context rows, residual/coverage
  behavior, workbook accounting invariants, and a large clean multi-period
  background portfolio that should not create false positives.
- `validation/ppar_performance_comparison_modified_dietz.yaml`: Tests Modified
  Dietz external-flow cross-check diagnostics.
- `validation/ppar_performance_comparison_policy_gap_demo.yaml`: Tests
  missing-YAML setup guidance for omitted contribution, transaction-rule, and
  transaction-impact specifications.
- `validation/ppar_performance_comparison_suppressed.yaml`: Tests
  active-vs-suppressed finding behavior and audit visibility.

Validate the scenario matrix with:

```bash
./.venv/bin/python -m ppar.performance_comparison.cli.validate_demo_matrix
```

## Scenario Matrix

| Scenario | YAML | Expected reviewer action | Status |
| --- | --- | --- | --- |
| Clean/no issue | `baseline` | No Problems-grid row. | Covered |
| Missing contribution policy | `policy_gap` | Select `contribution_impact_methods`. | Covered |
| Missing transaction method | `policy_gap` | Configure `transaction_impact_methods`. | Covered |
| Missing denominator | `policy_gap` | Set `denominator_source`. | Covered |
| Missing transaction sign/flow semantics | `policy_gap` | Define sign/flow semantics. | Covered |
| Low-confidence estimate | `multi` | Decide whether the estimate is acceptable. | Covered |
| Context-only evidence | `multi` | Review context without treating it as impact. | Covered |
| Modified Dietz cross-check | `modified_dietz` | Review cross-check. | Covered |
| Portfolio YAML specifications | `portfolio` | Run strict causal attribution with all supported policies configured. | Covered |
| Security YAML specifications | `security` | Review security-period differences with security review keys. | Covered |
| Suppressed finding | `suppressed` | Exclude from active review; keep audit-visible. | Covered |
| Residual withheld | `multi` | Resolve partial or missing estimates first. | Covered |
| Large clean background | `multi` | Confirm unchanged periods do not create false positives. | Covered |
| Large issue scale | Future generated fixture | Test hundreds of problem rows. | Planned |
