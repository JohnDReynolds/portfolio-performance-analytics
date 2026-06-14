# Axys Performance Comparison Demo Matrix

The packaged Axys demo data is intentionally separate from the unit-test
fixtures. Demo files are shaped for reviewer workflow examples; tests may still
smoke-test them so the examples do not drift.

From a source checkout, validate the covered scenarios with:

```bash
./.venv/bin/python scripts/performance_comparison_validate_demo_matrix.py
```

## Comparison YAML Files

- `baseline`: `ppar_performance_comparison.yaml`
- `single`: `ppar_performance_comparison_restatement.yaml`
- `transaction_rules`:
  `ppar_performance_comparison_restatement_transaction_rules.yaml`
- `multi`: `ppar_performance_comparison_multi_restatement.yaml`
- `policy_gap`: `ppar_performance_comparison_policy_gap_demo.yaml`
- `suppressed`: `ppar_performance_comparison_suppressed.yaml`

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
| Suppressed finding | `suppressed` | Exclude from active review; keep audit-visible. | Covered |
| Residual withheld | `multi` | Resolve partial or missing estimates first. | Covered |
| Large multi-period scale | Future generated fixture | Test hundreds of problem rows. | Planned |

The current matrix favors a small set of reusable CSV snapshots over many
near-duplicate directories. Add new CSV snapshots only when a scenario cannot be
expressed clearly through YAML policy changes against the existing data.

## Method Coverage Goal

Each supported public YAML impact method should have at least one packaged demo
scenario and one validator assertion. Tests can still cover narrow edge cases,
but the demos should prove that each method is understandable from reviewer
outputs.

Current public YAML method targets:

- `contribution_impact_methods.portfolio_source_field`:
  `source_field_delta_over_begin_market_value` (covered)
- `contribution_impact_methods.security_contribution`:
  `vendor_contribution_delta` (covered)
- `contribution_impact_methods.security_return`:
  `security_return_delta_times_weight` (covered)
- `transaction_impact_methods.external_flow`: `evidence_only` (covered)
- `transaction_impact_methods.external_flow`: `modified_dietz` (planned)
- `transaction_impact_methods.performance`:
  `transaction_amount_delta_over_return_denominator` (covered)
