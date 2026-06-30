# Performance Comparison Roadmap

This is the central roadmap for performance-comparison work. It covers return
reconstruction, user-facing explanations, report evolution, and demo-data
guardrails.

Detailed design reference remains in
[`performance_comparison_design.md`](performance_comparison_design.md).
The packaged-demo source boundary is defined in
[`performance_comparison_demo_source_contract.md`](performance_comparison_demo_source_contract.md).
Historical demo-generation notes remain in
[`operational_demo_data_notes.md`](operational_demo_data_notes.md) and
[`../scripts/analytics_demo_data/GENERATION_NOTES.md`](../scripts/analytics_demo_data/GENERATION_NOTES.md).

## Axys Extract Contract Review Map

The Axys extract guardrails now have a single review path:

```text
ppar/demos/data/axys/demo_extract_availability.yaml
  -> scripts/render_demo_extract_availability.py
  -> docs/axys-apx-reference/Appendix_Demo_Extract_Availability.md
  -> ppar/performance_comparison/extract_contract.py
  -> ppar/performance_comparison/transactions.py
```

The YAML contract records IMEX/REP availability confidence, candidate source
names, source strategy, and blocking context requirements. The renderer keeps
the human appendix current. Runtime validation uses the same contract, or a
site-specific `extract_contract.path`, to prevent ambiguous Axys `dp`, `li`,
`lo`, and `wd` transaction codes from being classified from transaction code
alone when required context is absent.

## Core Idea

The portfolio and each security can be viewed as parallel return containers.

At the portfolio level, external contributions and withdrawals are flows into or
out of the portfolio.

At the security level, buys and sells are flows into or out of the security.

```text
Portfolio return:
  portfolio value change adjusted for contributions and withdrawals

Security return:
  security value change adjusted for buys and sells into or out of the security
```

This symmetry is the central design principle.

```text
Portfolio level:
  contribution = inflow
  withdrawal = outflow

Security level:
  buy = inflow to security
  sell = outflow from security
```

A buy of AAPL is not an external flow for the whole portfolio because cash moved
inside the portfolio from cash to AAPL. But from AAPL's own return perspective,
that buy behaves like a capital inflow.

## Conceptual Formula

The general return model is:

```text
Return = economic gain / adjusted capital base
```

For a Modified Dietz-style model:

```text
Return = (EMV - BMV - net flows + income/accrual components)
         / (BMV + weighted flows)
```

The exact numerator convention must be defined carefully by transaction category,
sign convention, and vendor behavior.

## Why This Matters

The current report can show changed holdings, changed transactions, changed
income/accrual fields, and changed reported performance. But without a full
return-reconstruction layer, some rows should remain evidence rather than
counted explanations to avoid double counting.

Example:

```text
AAPL holdings.market_value changed
AAPL transactions.amount changed
AAPL transactions.quantity/price/commission changed
```

Those rows are related, but not independent. If the report counts both the
changed ending market value and the changed transaction flow without a formal
return model, it can explain the same economic change twice.

A reconstruction layer would allow the report to say:

```text
The reported security return changed because:
  EMV changed
  dated security-level flows changed
  income/accrual changed
  denominator/timing changed
```

That is cleaner and more defensible.

## Portfolio-Level Reconstruction

At the portfolio level, return inputs should be reconstructed from source data
where possible:

```text
BMV = sum beginning holdings.market_value + holdings.accrued
EMV = sum ending holdings.market_value + holdings.accrued
External flows = transactions categorized as contributions/withdrawals
Weighted flows = external flows weighted by date convention
```

Example YAML shape:

```yaml
portfolio_return_reconstruction:
  method: modified_dietz
  beginning_value_source: holdings
  ending_value_source: holdings
  flow_source: transactions
  flow_timing: transaction_date
  day_count: actual_days
  inclusion_rule: beginning_of_day
  flow_categories:
    - external_flow
```

The comparison can then derive:

```text
derived_portfolio_return_a
derived_portfolio_return_b
derived_return_difference
reported_return_difference
reconstruction_difference
```

The derived return should probably not be too prominent at first. It can appear
as a check or comment, especially when it differs materially from the
vendor-reported return.

## Return Basis Policy

Fee and expense transactions need an explicit return-basis policy. A fee is
always a real cash movement, but whether it reduces reported performance depends
on whether the return is net of fees or gross of fees.

For net-of-fees performance:

```yaml
FEE:
  transaction_category: fee_expense
  cash_flow_sign: negative
  performance_flow_sign: performance
  return_basis: net
```

For gross-of-fees performance:

```yaml
FEE:
  transaction_category: fee_expense
  cash_flow_sign: negative
  performance_flow_sign: external_flow
  return_basis: gross
```

`return_basis` should become a required YAML item wherever fee or expense
transactions are eligible for return reconstruction. Without it, a workbook can
look numerically precise while using the wrong interpretation of fees.

## Security-Level Reconstruction

At the security level, the same idea applies:

```text
BMV = beginning security holdings.market_value + accrued
EMV = ending security holdings.market_value + accrued
Security-level flows = buys/sells/transfers for that security
Income = dividends/interest/fees/accrual changes for that security
```

Example YAML shape:

```yaml
security_return_reconstruction:
  method: modified_dietz
  beginning_value_source: holdings
  ending_value_source: holdings
  flow_source: transactions
  flow_timing: transaction_date
  day_count: actual_days
  inclusion_rule: beginning_of_day
  flow_categories:
    - buy
    - sell
    - transfer
  income_categories:
    - income
    - fee_expense
```

A buy is a security-level inflow. A sell is a security-level outflow.

## Avoiding Double Counting

Once reconstruction exists, the report should distinguish formula inputs from
supporting evidence.

Formula-level inputs might include:

```text
BMV
EMV
weighted flows
income
accrual
```

Supporting inputs might include:

```text
transaction.quantity
transaction.price
transaction.commission
holding.quantity
holding.price
holding.cost
```

Example hierarchy:

```text
Security return changed
  EMV changed
    holdings.market_value changed
      holdings.quantity changed
      holdings.price changed

  Weighted flow changed
    transactions.amount changed
      transactions.quantity changed
      transactions.price changed
      transactions.commission changed
```

Only actual source-data rows should normally receive a counted `Performance
Difference Explained`. Reconstruction formula math may be used internally, but
the default user-facing workbook should attribute the explanation back to rows
the user recognizes from holdings and transactions.

## Possible Report Structure

A future workbook could include:

```text
Performance Differences
Performance Difference Causes
Other Data Differences
Return Reconstruction Checks
Raw Audit Trail
```

Current status: the normal user-facing workbook now uses `Performance
Differences`, `Performance Difference Causes`, `Other Data Differences`, and
`Raw Audit Trail`. Reconstruction diagnostic sheets remain opt-in so the
default workbook stays focused on reviewable performance differences and their
source-data causes.

`Performance Difference Causes` should contain source-data rows:

```text
holdings.market_value
holdings.accrued
transactions.amount
```

Supporting rows such as `holdings.quantity`, `holdings.price`,
`transactions.quantity`, `transactions.price`, and `transactions.commission`
can appear beneath those source rows when they help explain how the counted
source row changed.

A useful future column might be:

```text
Formula Role
```

Possible values:

```text
Beginning Value
Ending Value
Weighted Flow
Income
Accrual
Supporting Input
Context
```

## YAML Design Principle

The YAML must be explicit and required.

No silent defaults.

If the user asks for portfolio or security return reconstruction and omits
required rules, the system should fail before report generation.

Required concepts for every return-reconstruction method:

```yaml
method
beginning_value_source
ending_value_source
flow_source
flow_categories
income_categories
return_basis
sign_convention
```

Additional concepts required only for `method: modified_dietz`:

```yaml
flow_timing
day_count
inclusion_rule
```

Supported return-reconstruction methods:

- `simple_dietz`: denominator is beginning value; flow timing fields are not
  valid because flows are not weighted.
- `modified_simple_dietz`: denominator is beginning value plus 50% of net flows;
  flow timing fields are not valid because every flow receives the same
  mid-period weight.
- `modified_dietz`: denominator is beginning value plus date-weighted flows;
  `flow_timing`, `day_count`, and `inclusion_rule` are required.

The YAML validator should fail hard when a method is missing fields it needs,
includes fields that method cannot use, repeats keys, or includes unknown keys.

Potential future additions:

```yaml
large_flow_policy
subperiod_linking
settlement_date_handling
cash_security_identifier
accrual_treatment
corporate_action_treatment
```

## Implementation Phases

### Phase 1: Security-Level Flow Model

- Implement explicit security-level Modified Dietz reconstruction.
- Treat buys and sells as security-level flows.
- Treat dividends and interest as income inputs.
- Compare derived security return to reported security return.
- Keep outputs conservative at first.

Initial implementation note: performance comparison now supports an optional
security-level `Security Return Checks` worksheet and CSV artifact when YAML
includes `security_return_reconstruction`. The check derives Modified Dietz
security returns from holdings, buy/sell transaction flows, and income
transactions. It compares derived security-return differences with reported
`SEC_RETURN` differences and remains diagnostic-only.

### Phase 2: Portfolio-Level Flow Model

- Implement explicit portfolio-level Modified Dietz reconstruction.
- Treat contributions and withdrawals as portfolio-level external flows.
- Do not treat buys and sells as portfolio-level external flows.
- Compare derived portfolio return to reported portfolio return.

Initial implementation note: performance comparison keeps portfolio external-flow
Modified Dietz estimates as review-only cross-checks. The dedicated return
reconstruction check is the place to compare reported returns with values derived
from holdings, income, and external-flow transactions.

### Phase 3: Return Reconstruction Checks

- Add a worksheet or report section showing:
  - reported return
  - derived return
  - difference
  - materiality status
- Use this as a diagnostic before making reconstruction the primary user-facing
  explanation.

Initial implementation note: performance comparison now supports an optional
portfolio-level `Return Reconstruction Checks` worksheet and CSV artifact when
YAML includes `portfolio_return_reconstruction`. The check derives Modified
Dietz portfolio returns from holdings plus external-flow transactions and
compares them with reported `PORT_RETURN` differences. The worksheet also shows
the derived numerator, derived denominator, beginning value, ending value, net
flow, weighted flow, and changed formula components so a reviewer can see why a
check is marked `Different`.

Diagnostic QA note: report bundles can explicitly include a `Reconstruction
Summary` worksheet/CSV artifact when portfolio or security reconstruction is
enabled. The normal user-facing bundle excludes these diagnostics by default.
When opted in, the summary counts each reconstruction check by status and
diagnostic category, using plain categories such as `Aligned`, `Missing Inputs`,
`Source Inputs Changed`, and `Formula Difference`. The detailed check sheets
include the same category and clearer comments so reviewers can quickly
distinguish source-input changes from missing inputs or model/vendor
differences.

### Phase 4: Attribute Formula Impacts To Source Rows

- Once reconstruction is trusted, use reconstructed formula inputs internally as
  the calculation basis.
- Allocate formula-level impacts back to source rows such as
  `holdings.market_value`, `holdings.accrued`, and `transactions.amount`.
- Show quantity, price, commission, cost, and reference-data rows as supporting
  evidence unless they directly explain a counted source row.

Status note: source-row attribution is implemented for both security and
portfolio checks where `reconstruction_category` is `Source Inputs Changed`.
The workbook uses formula roles such as beginning value, ending value, net flow,
weighted flow, and income internally, then allocates their calculated impact
back to recognizable rows such as `holdings.market_value` and
`transactions.amount` in `Performance Difference Causes`. `Return Reconstruction Checks`
and `Security Return Checks` remain opt-in interim audit trails for the raw
numerator, denominator, and source component inputs.

### Phase 5: Deterministic User-Facing Explanations

- Add a structured explanation layer for `Performance Differences` comments and
  row-level `Explanation`.
- Generate comments from cause/residual patterns, not from free-form inference.
- Prefer specific worksheet and field references over generic instructions.
- Keep comments short enough for the workbook, with detailed evidence remaining
  in `Performance Difference Causes`, `Other Data Differences`, and `Raw Audit Trail`.
- Make guidance action-oriented and understandable when formula rows and
  supporting rows overlap.

The current pain point is periods like BALANCED `2026-05-01` to `2026-05-29`,
where:

- `holdings.ending_market_value` explains the positive calculated difference;
- `holdings.beginning_market_value` explains a negative denominator effect;
- individual AAPL/MSFT `holdings.market_value` rows foot to the ending-value
  effect but are supporting detail rather than separate additive causes; and
- the current `Explanation` text is technically accurate but not helpful
  enough for a normal reviewer.

Future guidance should make those relationships explicit. For example:

```text
Ending holdings value increased; AAPL and MSFT holding rows below show the
security-level source changes that make up this ending-value effect.
```

```text
Beginning holdings value increased, which reduces the calculated return. This
denominator effect accounts for the Unexplained Difference in the Performance
Differences sheet.
```

```text
Supporting detail for the ending holdings value effect; included in the related
performance difference, not counted separately.
```

Example residual classifications:

```text
fully_explained_by_performance_difference_causes
residual_matches_beginning_value_effect
residual_matches_ending_value_effect
residual_matches_transaction_flow_effect
residual_has_identifiable_evidence_but_no_supported_estimate
no_identifiable_evidence_found
```

Example deterministic comment:

```text
The Unexplained Difference matches the beginning holdings market value effect
shown in `Performance Difference Causes`. Reported performance appears to reflect the
ending value change but not the beginning-value denominator effect.
```

Implementation guidance:

- Do not speculate about vendor behavior beyond what the source rows show.
- Use careful language such as "appears to" when the workbook can identify a
  pattern but cannot prove the vendor's calculation method.
- Avoid generic instructions such as "review other sheets" when the workbook
  can name the exact field or row family involved.
- Use consistent terms for formula rows, source rows, and supporting detail so
  reviewers do not have to infer whether a row is counted or merely related.
- Add tests for each explanation type:
  - given a known cause/residual pattern
  - expect a specific comment
  - expect specific row-level `Explanation`
  - avoid untested prose drift
- Keep supporting-row explanations plain enough that reviewers do not need a
  second performance-number column beside `Performance Difference Explained`.

### Phase 6: Generate Holdings From Scenario Inputs

Current packaged demo rebuilding now derives snapshot B `transactions.csv` from
snapshot A transactions plus validated explicit transaction scenarios in
`scripts/operational_demo_data/performance_comparison_transaction_scenarios.csv`.

It then derives snapshot B `holdings.csv` from snapshot A holdings plus two
deterministic sources:

1. transaction-derived holding impacts from changed `by`, `sl`, `wd`, `dv`,
   `in`, and fee-like `dp` rows;
2. validated explicit residual holding scenarios in
   `scripts/operational_demo_data/performance_comparison_holding_scenarios.csv`.

It then derives `secperf.csv` and `portperf.csv` from those holdings and
transactions under the configured YAML reconstruction rules. This removes one
major source of fixture drift: transaction changes and common related
cash/security holding changes now come from scenario intent and rules rather
than independent hand-maintained rows.

The scenario layers are intentionally strict:

- columns must exactly match the expected schema;
- scenario rows can target only derived snapshots, not the base snapshot;
- duplicate scenario keys are rejected;
- each row must change at least one holding value;
- each transaction scenario row must match exactly one base transaction row;
- each holding scenario row must match exactly one packaged holding row;
- transaction scenarios currently change only existing transaction rows and only
  numeric fields (`QTY`, `PRICE`, `AMOUNT`, and `COMMISSION`).

Residual holding scenarios now carry explicit `scenario_type` values:

- `valuation_mark`
- `cash_balance_correction`
- `quantity_valuation_correction`
- `accrual_correction`
- `cost_only_correction`

Each type is validated against the holding fields it is allowed to change. This
keeps remaining manual holding adjustments from becoming generic numeric
patches.

The rebuild/audit summary now exposes the scenario story directly:

- transaction scenarios grouped by transaction code;
- transaction-derived holding impacts grouped by transaction code;
- residual holding scenarios grouped by scenario type.

This gives a quick check that the demo still contains the intended examples
without requiring manual CSV inspection.

The same scenario coverage is now part of the audit guardrail. If an example
type disappears accidentally, the packaged demo-data audit fails before the
reports are regenerated.

Status: complete for the current packaged performance-comparison demo fixture.

### Phase 7: Performance Explanation Scenario Engine

Phase 7 should remain focused on answering one product question:

```text
Why did reported performance change from Snapshot A to Snapshot B?
```

It should not become a general accounting engine. The goal is to generate demo
fixtures that consistently explain performance differences from recognizable
source-data changes, while leaving non-performance accounting differences as
`Other Data Differences`.

Useful Axys/APX transaction references:

- [`Chapter_05_Transactions.md`](axys-apx-reference/Chapter_05_Transactions.md):
  draft transaction reference with evidence boundaries and confidence levels.
- [`Research_05_Transactions.md`](axys-apx-reference/Research_05_Transactions.md):
  consolidated research reference for transaction workflows, code evidence,
  dependencies, audit rules, contradictions, and known unknowns.

These references should inform transaction-code interpretation, sign/cash-flow
assumptions, and evidence caveats. They should not be treated as instructions to
reconstruct tax lots, cost methods, settlement accounting, or vendor books.

#### Phase 7A: Performance Explanation Engine Contract

Phase 7A is a design contract, not an implementation phase. It defines what the
performance-comparison engine is allowed to explain and what must stay as
review-only evidence.

Contract question:

```text
What source-data changes are allowed to explain reported performance changes?
```

Performance-explaining inputs:

- beginning holdings market value plus accrued amount;
- ending holdings market value plus accrued amount;
- dated external cash-flow transaction amounts;
- dated security-level buy/sell transaction amounts;
- dividend, interest, fee, and other income/expense amounts when configured;
- reported accrual amounts.

Supporting evidence:

- quantity;
- price;
- commission;
- transaction type;
- transaction date;
- cash-balance movement;
- security identifiers and portfolio identifiers needed to connect evidence to
  the affected performance input.

`Other Data Differences`:

- cost and cost-basis fields;
- broker;
- settlement date, unless a future supported rule makes it performance-relevant;
- security reference data;
- any changed field that is not part of the supported performance explanation
  model.

Hard-fail rules:

- required YAML is missing for transaction classification;
- an unknown transaction code appears in performance-relevant data;
- a performance-relevant source field changes without a configured
  interpretation;
- ambiguous Axys `dp`, `li`, `lo`, or `wd` transaction codes appear in an
  extract that lacks the transaction context fields required by the packaged
  Axys extract contract;
- generated transactions, holdings, `secperf.csv`, or `portperf.csv` drift from
  their scenario-derived values;
- expected scenario coverage disappears unexpectedly;
- an unsupported field attempts to explain performance.

Implemented guardrail:

- transaction loading now fails up front when a transaction row cannot resolve a
  known transaction category from a source category, known transaction code, or
  `transaction_rules` entry. This guard is intentionally narrower than
  sign/flow impact-policy validation: known transaction codes with incomplete
  return-impact policy still flow through the existing review workflow instead
  of changing report behavior.
- Axys IMEX transaction codes `li`, `lo`, `dp`, and `wd` are not safe
  external-flow indicators by code alone. The packaged demo now uses
  conditional YAML `transaction_rules` with normalized IMEX context fields such
  as `security_type`, `source_destination_type`,
  `source_destination_symbol`, `special_security_type`, and
  `special_security_symbol`. A `wd` row is classified as an external flow only
  when its cash security and source/destination context match the external
  party cash rule; `dp` is classified as fee expense only when special-security
  context confirms the fee case; `li`/`lo` rules distinguish external party
  flows from internal transfer cases. If an IMEX export cannot provide the
  context fields marked as required by
  `ppar/demos/data/axys/demo_extract_availability.yaml`, the transaction loader
  fails before YAML rules classify the rows. The next design step for that site
  is to consider a REP/report extract, custom report, or other local-discovery
  source that carries enough classification evidence.
- comparison YAML can now override the default packaged extract contract with:

  ```yaml
  extract_contract:
    path: site_extract_contract.yaml
    enforce_ambiguous_axys_flows: true
  ```

  `enforce_ambiguous_axys_flows` defaults to `true`; setting it to `false` is
  an explicit local opt-out.
- local extract contracts are validated by `validate_config`: the contract must
  define `datasets.transactions.csv.columns`, use supported transaction column
  aliases, and provide boolean `requires_context_for_semantics` and
  `blocking_if_missing` flags.
- `docs/axys-apx-reference/templates/site_extract_contract.yaml` is the starter
  template for site-specific contracts.
- packaged demo-data audit tests now assert that the user-facing `wd` withdrawal
  rows resolve as external flows and the fee-like `dp` rows resolve as
  fee-expense performance rows in both packaged snapshots.
- `tests/data/axys/site_variants/` contains small site-shape fixtures for the
  next hardening layer: IMEX rows with context fields, REP/report rows with
  reviewed semantics, and code-only IMEX rows that must fail for ambiguous Axys
  flow codes.

#### Phase 7B: Transaction Rule Coverage Validation

Status: complete for the current packaged performance-comparison demo fixture.

The configuration validator now reports:

- observed transaction codes across configured transaction files;
- observed transaction codes that do not have explicit YAML `transaction_rules`.

This summary is informational for ordinary comparison configurations so older
review-policy fixtures can still demonstrate incomplete YAML behavior. The
packaged performance-comparison demo has a stricter guardrail test: every
observed transaction code must be explicitly defined in its YAML
`transaction_rules`.

Ongoing coverage requirement:

- Tests and packaged demo data must continue expanding toward complete Axys
  transaction-type coverage, not just the currently performance-relevant rows.
  Each observed or documented Axys transaction type should have an explicit
  expected classification, even when the expected outcome is `transfer`,
  `corporate_action`, `unknown pending review`, or review-only evidence.
- [`axys-apx-reference/Appendix_Transaction_Semantics_Matrix.md`](axys-apx-reference/Appendix_Transaction_Semantics_Matrix.md)
  is the implementation-facing seed matrix for that coverage. Future fixtures
  should either satisfy a row in that matrix or update the matrix with the new
  evidence and expected treatment.
- `docs/axys-apx-reference/transaction_semantics_matrix.yaml` is the
  machine-readable coverage contract. Coverage is complete only when every row
  has a non-backlog fixture or an explicit documented reason to remain
  review-only/unknown.
- Matrix rows and pair patterns now include machine-checked `coverage_notes`, so
  each covered, context-only, partial, or backlog entry carries a rationale
  instead of relying only on table prose.
- External-flow coverage must include both positive and negative capital-flow
  cases, internal transfers, cash sweeps, fee/expense rows, income rows,
  corporate-action rows, and correction/cancellation/reversal-like rows when
  representative source evidence is available.
- Ambiguous `li`, `lo`, `dp`, and `wd` cases require multiple examples for each
  code: one or more true external-flow examples and one or more non-external
  examples whose context proves transfer, sweep, fee/expense, or another
  non-capital-flow treatment. Code-only examples must remain hard-fail fixtures.
- If IMEX cannot expose the context needed for a documented transaction type,
  tests should model the REP/report or custom-report fallback instead of
  weakening the IMEX guardrail.

#### Phase 7C: Packaged Demo Source Contract Audit

Status: complete for the current packaged performance-comparison demo fixture.

The packaged demo source contract is documented in
[`performance_comparison_demo_source_contract.md`](performance_comparison_demo_source_contract.md).
It states that the packaged CSVs are normalized demo extracts, not official
Axys/APX native schemas, and defines the narrow source-data fields ppar is
allowed to use when explaining performance differences.

The packaged demo audit now verifies the user-facing report tables follow the
source contract:

- `Performance Difference Causes` only contains approved demo source fields;
- holdings `cost` stays in `Other Data Differences`;
- configured holdings `accrued` changes remain performance-cause rows;
- security-reference fields do not appear as performance causes or demo context
  evidence.

The Axys/APX transaction references should be used to choose careful terminology
and conservative interpretation rules. The package should adopt only the rules
needed to explain performance changes. It should not adopt a broader accounting
model merely because the reference material describes one.

A future foundation pass should move from generic scenario-adjustment files to a
performance-explanation scenario engine that generates both holdings snapshots
from explicit starting performance inputs, transactions, valuation marks,
reported accrual amounts, and cash-balance rules. The remaining explicit holding
scenario rows should shrink as more performance-relevant transaction,
valuation, and accrual amount rules become deterministic.

The current guardrail tests pin the accounting impact of each supported simple
transaction type:

- `wd` and fee-like `dp` reduce ending `CASH_USD` holdings;
- `dv` and `in` increase ending `CASH_USD` holdings;
- `by` increases the traded security holding and reduces cash;
- `sl` reduces the traded security holding and increases cash;
- split/corporate-action evidence is intentionally absent from the current
  user-facing full-spec demo until it can use a real-world split in a period
  designed for that corporate action.

Suggested source-of-truth inputs:

```text
starting holdings
intentional scenario changes
transactions
valuation marks / prices
reported accrual amounts
cash-account rules
security reference data
```

The generator should then derive:

```text
holdings.csv
secperf.csv
portperf.csv
```

This would make the demo stack flow in one direction:

```text
scenario intent
  -> transactions
  -> transaction-derived holding impacts and valuation assumptions
  -> holdings
  -> security performance
  -> portfolio performance
  -> reports
```

Required performance-explanation rules before this phase should be considered
complete:

- starting positions, cash, market value, and accrued amounts
- trade sign conventions
- trade-date versus settlement-date treatment
- cash offsets for buys, sells, dividends, interest, fees, contributions, and
  withdrawals
- market value formula, including whether accrued is separate from market value
- reported fixed-income accrual amount treatment
- corporate-action treatment for splits and future action types
- real-world corporate-action fixture policy: user-facing demo splits must use
  actual historical split dates/securities, not fictional future splits; purely
  synthetic corporate-action cases belong in clearly labeled test-only fixtures.
- cash security identifier policy, including multi-currency cash accounts
- rounding policy for quantities, prices, market values, cash, and performance

The generator and auditor should stay coupled. A generated dataset should fail
before writing if:

- holdings no longer foot to the generated transactions and valuation marks;
- cash does not move correctly for transaction types that affect cash;
- `secperf.csv` does not reconstruct from holdings and security-level flows;
- `portperf.csv` does not reconstruct from holdings and portfolio-level flows;
- any intentional partly explained or unexplained scenario is not named in an
  allowlist with a reviewer-facing reason.

This phase is intentionally larger than the current scenario guardrails, but its
scope should stay narrow. It should explain performance-input changes, not
rebuild portfolio accounting. Cost differences should remain visible as
`Other Data Differences` unless they become direct inputs to a supported
performance explanation.

### Phase 8: Improve Demo Data

Current packaged demo coverage includes realistic examples of:

- portfolio contribution
- portfolio withdrawal
- security buy
- security sell
- dividend
- interest
- fee
- accrual change

Next expansion should be ordered by realism and by the evidence needed to
classify Axys transaction semantics safely.

| Priority | Scenario family | Permanent home | Required evidence before implementation | Notes |
| --- | --- | --- | --- | --- |
| 1 | Additional contribution or withdrawal variants | Test-only first, packaged demo only when nonredundant | Cash security, external-party source/destination context, amount sign, Modified Dietz flow-weighting policy. | The packaged demo now includes one ordinary `li` contribution and one `wd` withdrawal. Add more only when the scenario teaches a new review behavior. |
| 2 | `li` / `lo` external-flow and transfer examples | Test-only first, packaged demo only when realistic | Source/destination type and symbol, security type, amount/quantity signs, or reviewed REP semantics. | Site-variant fixtures and the packaged `li` contribution prove the classification patterns. Add transfer examples to the packaged demo only if the business story is realistic and not redundant with withdrawal/contribution. |
| 3 | Additional `dp` / `wd` variants | Test-only first | Special-security context, sweep/cash symbols, source/destination context, and explicit fee/transfer/external-flow treatment. | Keep proving code-only `dp` and `wd` are unsafe. Packaged demo should stay focused on one fee-like `dp` and one external `wd` until a stronger story is needed. |
| 4 | Fixed-income accrued-interest, maturity, and principal-paydown cases (`pa`, `sa`, `ai`, `pd`) | Test-only first, then packaged demo when accounting rules are deterministic | Bond security type, accrued-interest treatment, cash offset, principal movement, gain/loss or income treatment, local mapping or REP evidence. | Do not add a user-facing bond maturity until holdings, accrual, cash, `secperf.csv`, and `portperf.csv` all derive from one coherent scenario intent. |
| 5 | Return of capital (`rc`) | Test-only first | Security, amount sign, cost-basis/report treatment, whether return is performance income or review-only corporate-action evidence. | Needs explicit policy before it can explain performance. |
| 6 | Real-world split / corporate action evidence | Packaged demo only when the demo period/security supports a real historical event | Actual historical date/security, split ratio, quantity and price treatment, security master/report evidence, and a policy for whether it is explanatory or review-only. | User-facing demo splits must not be fictional future events. Synthetic corporate-action fixtures belong in clearly labeled test-only data. |
| 7 | Short sale / cover short (`ss`, `cs`) | Test-only first | Short/security type, cash/margin/short symbols, amount/quantity signs, and reviewed local treatment. | Keep as backlog until the project has enough short-account evidence to avoid implying a universal Axys convention. |
| 8 | Correction/cancellation/reversal-like uppercase rows | Test-only first | Link to original transaction or enough matching fields to identify the reversal target. | Demonstrate review-only or correction behavior without treating an unlinked uppercase row as a new economic event. |

#### Phase 8A: Realistic Transaction Expansion Gate

The packaged-demo contribution scenario passes the same end-to-end standard as
the withdrawal example. Before adding any additional external-flow variant to
`axys_full_spec_a` or `axys_full_spec_b`, document and verify:

- scenario intent: which portfolio, period, cash security, and reviewer-facing
  business story the contribution represents;
- source evidence: the Axys-style transaction code, amount sign,
  source/destination type, source/destination symbol, and any REP/report
  semantic fields needed to prove it is an external capital inflow;
- YAML semantics: the `transaction_rules` entry classifies the row from
  context, not from an ambiguous code alone, and preserves the correct
  `external_flow` cash/performance signs;
- cash and holdings: the transaction-derived cash movement reconciles to the
  generated ending `CASH_USD` holding for the affected period;
- return reconstruction: `portperf.csv` and the portfolio Modified Dietz
  reconstruction agree on net external flow, weighted flow, and return impact;
- report behavior: workbook and CSV artifacts explain the changed contribution
  without double-counting security-level buy/sell, income, fee, or holdings
  effects;
- docs and tests: this roadmap, the demo source contract, the packaged demo
  README, the semantics matrix, and audit tests all name the expected treatment.

Implemented contribution recipe:

- add the contribution as an inserted transaction scenario, not by mutating an
  unrelated base transaction row;
- use an Axys-style `li` row on `CASH_USD` with `SRC_DEST_TYPE=$pty`,
  `SRC_DEST_SYMBOL=$cash`, positive `AMOUNT`, zero quantity/price/commission,
  and same-day settlement unless site evidence says otherwise;
- let the rebuild script derive snapshot B `transactions.csv`, ending cash
  holdings, `portperf.csv`, and reconstruction diagnostics from the scenario;
- keep future external-flow rows out of packaged CSVs until the generated
  workbook has no unintended partly explained or unexplained period.

Test-only rebuild guards now mirror this for both directions: inserted `li`
cash rows prove external contributions increase cash, and inserted `lo` cash
rows prove external deliver-out/withdrawal-style rows decrease cash when
external-party context is present. The packaged demo keeps only the realistic
`li` contribution and `wd` withdrawal examples until a separate `lo` user story
adds reviewer value.

`li`/`lo`, additional `dp`/`wd`, and synthetic corporate-action scenarios should
remain test-only until they meet their own version of this gate. A real-world
split can move into the packaged demo only when the demo period and security
support an actual historical split date and the row is clearly tied to
review-only or implemented corporate-action behavior.

#### Phase 8B: Reinvestment Pair Feasibility Gate

The next useful test-only pair family is a reinvested dividend represented by a
`dv` income leg plus a paired `by` purchase leg. This should not enter packaged
demo data until the comparison engine can prove the pair is one economic
reinvestment, not two unrelated effects.

Status: partial test-only coverage. Return-reconstruction tests now prove the
`by` leg is not treated as a portfolio external flow and the `dv` leg is not
counted twice as security income. Full pair matching remains future work.

Before adding a reinvestment fixture, require:

- pairing evidence: same portfolio, security or reinvestment target, date or
  settlement window, dividend-wash/source symbol, and offsetting amount
  relationship;
- YAML semantics: `dv` remains performance income and `by` remains a
  security-level flow, with no portfolio-level external-flow treatment;
- double-count guard: portfolio return reconstruction must not count the buy
  leg as an external contribution, and security reconstruction must not count
  the dividend income twice;
- report behavior: workbook rows should show the income and buy evidence as
  related, while making clear which formula input each row supports;
- fixture home: synthetic reinvestment examples belong in test-only data until
  a realistic packaged period and security story makes the example useful to a
  reviewer.

For any additional examples:

- Make all accounting internally consistent.
- Ensure the report demonstrates fully explained, partly explained, and
  unexplained cases.
- Keep all packaged demo accounting internally consistent.
- Run the packaged demo-data rebuild/audit before accepting fixture changes:

  ```bash
  ./.venv/bin/python scripts/operational_demo_data/rebuild_performance_comparison_demo_data.py
  ```

- Use `--write` on the same command to refresh derived `secperf.csv` and
  `portperf.csv`; the command audits after rebuilding so generated demo data is
  not refreshed without a consistency check.
- Any intentional partly explained or unexplained period must be named in the
  audit allowlist with a reviewer-facing reason.

Demo-data notes:

- Current operational demo generation history lives in
  [`operational_demo_data_notes.md`](operational_demo_data_notes.md).
- Current analytics demo data generation history lives in
  [`../scripts/analytics_demo_data/GENERATION_NOTES.md`](../scripts/analytics_demo_data/GENERATION_NOTES.md).
- Those files are historical/process notes; this file is the active roadmap.

## Guiding Principle

Do not casually rebuild an accounting system.

The goal is not to reproduce every vendor methodology perfectly. The goal is to
create a transparent, auditable, configurable return-reconstruction layer that
can explain most real-world differences and clearly identify when vendor
methodology or missing data prevents full explanation.

The future standard should be:

```text
Count formula inputs.
Show supporting accounting evidence.
Do not double count.
Fail hard when required YAML rules are missing.
```

## Transaction-Type Backlog

Use Phase 8 as the implementation order. This backlog is the policy boundary:

- Packaged demo additions should be realistic, internally consistent, and
  understandable to a reviewer without knowing the fixture harness.
- Test-only fixtures should absorb synthetic edge cases, code-only failure
  cases, local-policy gaps, and ambiguous Axys semantics that need surgical
  proof.
- Evidence-blocked cases should stay backlog until IMEX context, REP/report
  semantics, or real source samples justify a classification.

Near-term backlog by home:

| Home | Transaction families | Exit criteria |
| --- | --- | --- |
| Packaged demo candidate | nonredundant external-flow variants; possibly realistic `li`/`lo` transfer examples; real historical split when the demo period supports it | Scenario intent, transactions, holdings, security performance, portfolio performance, YAML rules, workbook explanations, and README/source-contract language all align. |
| Test-only fixture | `li`/`lo` external and neutral variants; more `dp`/`wd` fee/sweep/external cases; uppercase reversal/cancellation; synthetic corporate actions | Fixture proves expected semantics or failure mode without implying the packaged demo is a realistic client story. |
| Evidence-blocked backlog | `ss`, `cs`, `ai`, `pa`, `sa`, `rc`, `pd`, mergers, spin-offs, ticker changes | Axys/REP/report evidence identifies required fields and ppar treatment well enough to avoid code-only classification. |
