# Axys Demo Data

The packaged Axys demo data contains only user-facing demo inputs. Test-only
performance comparison scenarios live under `tests/data/axys`.

Start with [QUICK_START.md](QUICK_START.md) when onboarding an Axys/APX
performance-comparison site. That file gives the ordered setup path: confirm
source files, copy the vanilla Axys YAML, validate, generate portfolio and
security reports, then iterate local overrides.

## Comparison YAML

There is one packaged Axys comparison YAML file. The portfolio and security demo
commands use the same operational Mega-Cap source snapshots and choose the
primary review level at runtime. The demo is meant to serve two user-facing
purposes: a concise marketing example of the review output, and an onboarding
example for configuring a new Axys-style site.

The packaged CSV files follow the
[Performance Comparison Demo Source Contract](../../../../docs/performance_comparison_demo_source_contract.md).
They are normalized demo extracts, not official Axys/APX native schemas.
The source contract separates mandatory product inputs, realistic packaged-demo
fields, optional local-enrichment fields, and internal scenario/rebuild fields.
For field-by-field IMEX and REP availability confidence, see
[Demo Extract Availability](../../../../docs/axys-apx-reference/contracts/demo_extract_availability.md).

Before generating a local report bundle, run the comparison YAML validator. It
checks the minimum required datasets, required normalized columns, transaction
semantics, extract-contract guardrails, and complete YAML treatment for changed
source-data fields:

```bash
./.venv/bin/python -m ppar.performance_comparison.cli.validate_config \
  ppar/demos/data/axys/axys_performance_comparison.yaml
```

| Role | YAML |
| --- | --- |
| Workbook demos | `axys_performance_comparison.yaml` |

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

Open `report.xlsx` when present. Use `report.html` for browser review, and keep
the CSV artifacts for supplementary diagnostics and audit traceability. The
report is designed for review, not for raw data export. It separates performance
differences from identifiable input differences and other evidence:

- `Performance Differences` sheet: one row per portfolio period with a performance
  difference in the portfolio demo.
- `Performance Differences` sheet: one row per security-period return difference in
  the security demo. Review keys include `Portfolio`, `From Date`, `Thru Date`,
  and `Security`.
- `Performance Difference Causes` sheet: input rows such as holdings, transactions,
  and FX rates. `B - A Difference` shows the raw input-value difference, and
  `Performance Difference Explained` appears only when ppar can calculate a
  defensible performance explanation.
- Optional reconstruction diagnostics can add `Reconstruction Summary`,
  `Return Reconstruction Checks`, and `Security Return Checks` sheets for
  implementation review, but normal demo output excludes them by default.
- Review-only supporting rows remain in the `Raw Audit Trail`. Transaction
  quantity, price, and commission rows may also appear on
  `Performance Difference Causes` when they support a changed
  `transactions.amount`.
- `Raw Audit Trail` sheet: the underlying finding rows used to build the workbook.
  Transaction match status appears here for audit and troubleshooting; the
  separate `transaction_matching_diagnostics.csv` artifact is row-identity audit
  support rather than a main review sheet.

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
- YAML: maps source transaction codes (`by`, `sl`, `dv`, `in`, `pa`, `sa`,
  `dp`, `li`, `lo`, `wd`, and `;`) to normalized categories such as `buy`,
  `sell`, `income`, `fee_expense`, `external_flow`, and `corporate_action`.
  Reviewer-facing explanations preserve the source code rather than uppercasing
  or replacing it with the category.
- Packaged transaction rows intentionally use only the small user-facing set
  `by`, `sl`, `dv`, `in`, `pa`, `sa`, `dp`, `li`, `lo`, and `wd`. The packaged
  `pa` and `sa` rows appear only as fixed-income accrued-interest adjuncts
  paired with TNOTE5Y buy/sell rows. The packaged `li` row is a plain external
  cash contribution with external-party context, and the packaged `lo` row is an
  external cash deliver-out with the same context standard. More ambiguous
  `li`/`lo` transfer cases and synthetic corporate-action rows live in test-only
  fixtures until a realistic packaged story and evidence trail justify adding
  them here.
- Packaged transaction rows omit `TRANSACTION_ID`. ppar supports stable
  transaction IDs when a local extract provides them, but the packaged Axys
  demo uses the more realistic conservative no-ID path by default. Internal
  scenario/rebuild files may still use deterministic transaction IDs as fixture
  handles; those IDs are not packaged as user-facing Axys transaction fields.
- Current transaction coverage by home:

  | Home | Transaction families |
  | --- | --- |
  | Packaged demo rows | `by`, `sl`, `dv`, `in`, fixed-income accrued-interest `pa`/`sa`, fee-like `dp`, external-cash `li`, external-cash `lo`, and external-cash `wd`. |
  | YAML rules reserved for runtime guards | `;` corporate-action rows and non-packaged conditional branches for ambiguous flow codes. |
  | Test-only fixtures | internal-transfer `li`/`lo` site variants, `dp`/`wd` site variants, `pa`/`sa` local-override examples, and `dv` + `by` reinvestment guards. |
  | Evidence-blocked backlog | `ai`, `pd`, `ss`, `cs`, `rc`, uppercase reversal rows, and real-world corporate actions until source evidence and accounting policy are strong enough. |

  The packaged fixed-income story is intentionally narrow: ordinary TNOTE2Y
  interest uses an `in` transaction row, accrued-interest restatement uses
  `holdings.accrued`, and TNOTE5Y `pa`/`sa` rows are packaged only with paired
  fixed-income trade context. The packaged demo does not infer accrued-interest,
  margin-interest, or principal-paydown treatment from code alone.
- Real site extracts should keep ambiguous-flow enforcement enabled. IMEX is
  sufficient only when transaction rows include source/destination and
  special-security context for `dp`, `li`, `lo`, and `wd`; otherwise use a REP,
  custom report, or reviewed source that supplies transaction category and
  cash/performance sign semantics.
- YAML: includes explicit `portfolio_return_reconstruction` settings for
  Modified Dietz diagnostic checks.
- YAML: includes explicit `security_return_reconstruction` settings for
  security-level Modified Dietz diagnostic checks.
- YAML: treats fee-like `dp` transactions as performance-impacting because this packaged
  fixture assumes the reported returns are net of fees. For gross-of-fees
  performance, fees would need a different return-basis policy.

Expected workbook:

- Changed ALPHA, BALANCED, and INCOME periods in the portfolio demo
  `Performance Differences` sheet.
- Changed security-period returns in the security demo `Performance Differences`
  sheet.
- `Performance Difference Causes` sheet should show understandable additive transaction
  amount, holding market value, holding accrued, and weighted price examples.
  It should also show supporting input-component rows without assigning them
  `Performance Difference Explained` values when those rows help explain a
  counted source-data difference.
- The packaged demo intentionally includes fully explained, partly explained,
  and unexplained periods so reviewers can see each status in both report
  families.
- Optional reconstruction diagnostics should show where source-derived Modified
  Dietz returns agree with reported return differences and where they do not.
- The controlled restatement includes:
  - a fully explained ALPHA period with AAPL price/security-return changes and
    `CASH_USD` holding changes;
  - a fully explained ALPHA period with a changed buy transaction amount,
    changed AAPL holding quantity/market value, and related transaction
    quantity, price, and commission support rows;
  - a fully explained ALPHA period with an external cash deliver-out;
  - a fully explained BALANCED period with a dividend transaction amount change;
  - a fully explained BALANCED period with an external cash contribution;
  - a fully explained INCOME period with a larger advisory-fee expense and
    matching lower `CASH_USD` ending value;
  - a partly explained BALANCED period where the same AAPL price correction and
    standalone MSFT holding market-value correction explain part of the
    reported portfolio and MSFT security return differences, leaving an
    intentional residual for reviewer triage;
  - a fully explained INCOME period with the same AAPL price correction plus
    TNOTE2Y market-value and accrued-interest changes, related TNOTE2Y quantity
    evidence, and TNOTE2Y cost in the `Raw Audit Trail`;
  - a partly explained INCOME period where paired TNOTE5Y `by`/`pa` and
    `sl`/`sa` fixed-income trade/accrued-interest settlement rows affect the
    cash/performance inputs, while separate quantity-driven holding value and
    accrued-value rows remain visible as holding inputs. TNOTE5Y cost-only audit
    evidence stays in the raw trail, and incomplete or overlapping estimates
    still require reviewer triage;
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

For a full packaged-demo health pass from a source checkout, maintainers can use
the consolidated maintenance script:

```bash
./.venv/bin/python scripts/check_performance_comparison_demo_health.py
```

That script is not part of the installed-package demo workflow. It runs the
rebuild drift audit, extract-availability contract check, portfolio/security
bundle generation, bundle validation, and packaged scenario matrix validation.

## YAML Policy Decision Guide

Use the YAML policy blocks to state what ppar is allowed to treat as an
explanation. The values are intentionally explicit because different vendors
can use the same-looking fields with different sign, timing, denominator, or
accounting conventions.

The workbook uses a small field-role model:

| Role | Typical fields | Workbook treatment |
| --- | --- | --- |
| `performance_input` | `holdings.market_value`, `holdings.accrued`, `transactions.amount` | Additive rows on the `Performance Difference Causes` sheet when enough inputs are available. |
| `input_component` | `holdings.quantity`, `holdings.price`, transaction quantity/price/commission | Shown beside related performance inputs when useful, or kept in `Raw Audit Trail` as support for the related performance input. |
| `reported_performance_component` | portfolio/security performance return, income, gain/loss, contribution, weight, market value | Kept as reporting diagnostics in the audit trail; not treated as root-cause input differences. |
| `context` | holding cost, FX rates, security reference data, unsupported fields | Kept in `Raw Audit Trail` unless it is a direct input to a supported performance explanation. |

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

The packaged demo is expected to be internally consistent: visible portfolio
and security performance differences should be fully explained by source-data
differences in the workbook. Test-only fixtures can still cover unresolved or
policy-gap scenarios, but the user-facing demo should not depend on accidental
internal inconsistency.

From a source checkout, maintainers keep the derived `secperf.csv` and
`portperf.csv` files aligned with:

```bash
./.venv/bin/python scripts/operational_demo_data/rebuild_performance_comparison_demo_data.py
```

The default mode audits the checked-in files without writing. Add `--write`
after intentional fixture edits to recompute security beginning weights,
security contributions, and portfolio performance rows from security
performance rows.

Current public YAML targets are intentionally narrow:

- `extract_contract`: selects the packaged or site-specific extract contract
  used by runtime guards. The default packaged contract enforces context-field
  presence before ambiguous Axys `li`, `lo`, `dp`, or `wd` rows can be
  classified by YAML rules. Use
  `docs/axys-apx-reference/contracts/templates/site_extract_contract.yaml` as a starter
  when a real site needs a local contract.
- `transaction_rules`: classifies transaction codes for amount attribution.
  Ambiguous Axys-style `li`, `lo`, `dp`, and `wd` examples require matching
  transaction-context fields before they are treated as external flows or
  fee/expense rows. Fixed-income `pa`/`sa` accrued-interest adjuncts require
  fixed-income context and paired-trade support in the packaged demo. Remaining
  fixed-income backlog codes such as `ai` and `pd` require test-only fixture
  proof plus local mapping or REP/report evidence before they should become
  user-facing demo transactions.
- `transaction_amount_delta_over_return_denominator`: default amount-impact
  method used after `transaction_rules` mark a transaction code as
  performance-affecting.
- `transaction_impact_methods.external_flow`: optional `modified_dietz`
  cross-checks for external-flow transactions.
- suppression rules: remove known, intentionally ignored differences from the
  active review while retaining them in the raw audit trail.
