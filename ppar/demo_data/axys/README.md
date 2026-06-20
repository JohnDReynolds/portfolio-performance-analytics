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
- `full_spec`: `ppar_performance_comparison_full_spec.yaml`
- `modified_dietz`: `ppar_performance_comparison_modified_dietz.yaml`
- `policy_gap`: `ppar_performance_comparison_policy_gap_demo.yaml`
- `suppressed`: `ppar_performance_comparison_suppressed.yaml`

## XLSX Workbook Demo Commands

Run these commands from the repository root after installing the optional Excel
dependency with `./.venv/bin/python -m pip install -e ".[excel]"`. Each command
writes a report bundle and an Excel workbook at
`_demo_output/workbooks/<demo_name>/review_workbook.xlsx`.

For these XLSX demos, start review in `review_workbook.xlsx`. Use `report.html`
only when you want a browser-friendly narrative view. The workbook is designed
for review, not for raw data export. It separates portfolio/security
performance differences from underlying input differences and reported checks:

- `Portfolio Differences` sheet: one row per portfolio period with a performance
  difference.
- `Security Differences` sheet: one row per security-period return difference, when
  security performance data exists. Portfolio periods with no security-level
  return differences get an explicit no-differences row.
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

### 1. Baseline / Clean Comparison

```bash
./.venv/bin/python scripts/performance_comparison_report_bundle.py \
  ppar/demo_data/axys/ppar_performance_comparison.yaml \
  _demo_output/workbooks/baseline \
  --include-workbook
```

Data used:

- Snapshot A: `axys_a`
- Snapshot B: `axys_b`
- Files: portfolio performance, security performance, security master, prices,
  FX rates, transactions, positions, and cash.
- YAML: intentionally has no causal-attribution policy blocks because the data
  is expected to compare cleanly.

Expected workbook:

- `Portfolio Differences` sheet: one message row indicating that no portfolio
  performance differences were found.
- `Security Differences` sheet: empty
- `Underlying Causes` sheet: empty
- `Reported Performance Checks` sheet: empty
- `Context` sheet: empty
- `Raw Audit Trail` sheet: empty

Why: this is the clean/no-issue control fixture. It proves the comparison can
run without producing false positives. An empty workbook is the expected result.

### 2. Single Restatement

```bash
./.venv/bin/python scripts/performance_comparison_report_bundle.py \
  ppar/demo_data/axys/ppar_performance_comparison_restatement.yaml \
  _demo_output/workbooks/single_restatement \
  --include-workbook
```

Data used:

- Snapshot A: `axys_a`
- Snapshot B: `axys_b_restatement`
- Files: full Axys-shaped data set: portfolio performance, security
  performance, security master, prices, FX rates, transactions, positions, and
  cash.
- YAML: includes contribution policies for portfolio source fields, vendor
  contribution, and security-return weighting.
- YAML gap: does not include transaction rules or transaction impact methods.

Expected workbook:

- A small review workbook with one changed portfolio period and one changed
  security period.
- `Underlying Causes` sheet should show restated source values such as positions,
  transactions, cash, and prices.
- `Reported Performance Checks` sheet should show portfolio/security performance
  rows such as gain/loss and contribution changes without treating them as root
  causes.
- `Portfolio Differences` sheet should indicate that some setup is missing because
  transaction semantics are not fully specified.

Why: this fixture shows a controlled restatement, but intentionally leaves
transaction interpretation incomplete so the reviewer can see the setup action
that would be needed for more complete attribution.

### 3. Single Restatement With Transaction Rules

```bash
./.venv/bin/python scripts/performance_comparison_report_bundle.py \
  ppar/demo_data/axys/ppar_performance_comparison_restatement_transaction_rules.yaml \
  _demo_output/workbooks/transaction_rules \
  --include-workbook
```

Data used:

- Snapshot A: `axys_a`
- Snapshot B: `axys_b_restatement`
- Files: same full Axys-shaped data set as the single restatement demo.
- YAML: adds transaction rules for `BUY`, `SELL`, `DIV`, and `INT`.
- YAML: adds transaction impact methods for performance transactions and
  evidence-only external flows.

Expected workbook:

- Similar changed portfolio/security rows as the single restatement demo.
- Transaction amount rows should have `Performance Difference Explained` because
  the YAML supplies sign/flow semantics and a transaction amount impact method.
- Transaction quantity and price rows remain visible as input differences, but
  they are not directly modeled as performance explanations.
- This fixture isolates transaction-related setup. Other non-transaction
  input differences may still require separate YAML decisions.

Why: this fixture shows how the same data becomes more actionable when YAML
defines transaction behavior instead of asking the reviewer to infer it.

### 4. Full YAML Specifications / Strict Attribution

```bash
./.venv/bin/python scripts/performance_comparison_report_bundle.py \
  ppar/demo_data/axys/ppar_performance_comparison_full_spec.yaml \
  _demo_output/workbooks/full_spec \
  --include-workbook \
  --require-causal-attribution
```

Data used:

- Snapshot A: `axys_full_spec_a`
- Snapshot B: `axys_full_spec_b`
- Files: compact set of portfolio performance, security performance,
  transactions, positions, and prices.
- YAML: includes every currently supported causal-attribution policy.
- YAML: strict mode is enabled with `--require-causal-attribution`.

Expected workbook:

- Eight changed portfolio periods in the `Portfolio Differences` sheet.
- One changed security-period return in the `Security Differences` sheet.
- `Underlying Causes` sheet should show additive transaction amount, position market
  value, position accrued, and weighted price examples.
- `Underlying Causes` sheet should also show one explicit evidence-only position
  quantity row with `Required YAML Setup` set to `None; configured as
  evidence-only in comparison YAML.`
- `Reported Performance Checks` sheet should show configured
  portfolio/security performance estimates such as security-return weighting,
  vendor contribution delta, and portfolio source-field delta. These confirm
  performance-output differences but are not labeled as root-cause input
  differences.
- Some portfolio changes may still be unexplained when the changed data falls
  outside currently supported attribution methods; strict mode verifies setup
  completeness for supported methods, not perfect explanatory coverage. These
  periods appear in the `Underlying Causes` sheet as `no_underlying_cause_found`
  diagnostic rows.

Why: this is the most focused workbook for understanding the causal-attribution
model. It keeps the data small and the YAML complete while still distinguishing
underlying input causes from derived performance checks.

After generating any workbook demo bundle, validate it with:

```bash
./.venv/bin/python scripts/performance_comparison_validate_bundle.py \
  _demo_output/workbooks/<demo_name>
```

## Validation Fixtures

The remaining comparison YAML files are primarily scenario-coverage fixtures,
not recommended XLSX workbook demos. They are exercised by
`performance_comparison_validate_demo_matrix.py` and targeted unit tests.

- `multi`: Stress-tests multiple portfolios, multiple periods, context rows,
  residual/coverage behavior, and workbook accounting invariants.
- `modified_dietz`: Tests Modified Dietz external-flow cross-check diagnostics.
  The relevant outputs are `transaction_cross_checks.csv`, transaction summary
  tables, and HTML transaction sections rather than workbook review sheets.
- `policy_gap`: Tests Problems-grid and missing-YAML setup guidance for omitted
  contribution, transaction-rule, and transaction-impact specifications.
- `suppressed`: Tests active-vs-suppressed finding behavior and audit
  visibility.

Validate the full packaged scenario matrix, including these fixtures, with:

```bash
./.venv/bin/python scripts/performance_comparison_validate_demo_matrix.py
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
| Full YAML specifications | `full_spec` | Run strict causal attribution with all supported policies configured. | Covered |
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

The supported string vocabulary is summarized in
`docs/performance_comparison_design.md`. The package code backs those strings
with enums, but YAML examples intentionally show the plain string values users
edit.

The `full_spec` fixture is intentionally compact and action-oriented. It
contains eight changed portfolio periods so reviewers can see one occurrence of
each currently supported causal-attribution basis:

- security return weighted by beginning weight
- transaction amount over beginning market value
- vendor contribution delta
- portfolio income/gain-loss source-field delta over beginning market value
- position market value over beginning market value
- position accrued over beginning market value
- price delta over snapshot A price, weighted by snapshot A security weight
- explicit evidence-only treatment for a known position quantity change

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
- `price_impact_methods.price`:
  `price_delta_over_snapshot_a_price_times_weight` (covered)
- `evidence_only_impact_methods.<dataset>`:
  `evidence_only` with explicit `source_fields` (covered)
- `transaction_impact_methods.external_flow`: `evidence_only` (covered)
- `transaction_impact_methods.external_flow`: `modified_dietz` (covered)
- `transaction_impact_methods.performance`:
  `transaction_amount_delta_over_return_denominator` (covered)
