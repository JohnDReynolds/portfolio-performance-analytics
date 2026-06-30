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
- `;` split evidence remains outside transaction-derived holding impacts for now.

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

- portfolio withdrawal
- security buy
- security sell
- dividend
- interest
- fee
- split / corporate action evidence
- accrual change

Remaining high-value examples to add:

- portfolio contribution
- transfer in / transfer out
- bond maturity / principal paydown
- corporate actions beyond splits

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

The next transaction-type categories should be added in this order:

1. **Contribution / withdrawal**

   These are true portfolio-level external flows. They change the capital base
   and require Modified Dietz or another explicit flow-weighting convention
   before they can be treated as return-reconstruction inputs.

2. **Transfer in / transfer out**

   These resemble contributions and withdrawals, but the system must know
   whether the transfer is external to the reviewed portfolio or internal to the
   vendor/account structure. They should use the same flow-weighting machinery
   once their external/internal treatment is explicit.

3. **Bond maturity / principal paydown**

   These are useful for fixed-income review because they reduce a bond holding
   and increase cash. They may also interact with accrued interest, amortization,
   realized gain/loss, and vendor-specific income treatment.

4. **Corporate actions beyond splits**

   Mergers, spin-offs, and ticker changes are valuable audit evidence, but they
   often require security-identifier mapping and vendor-specific rules before
   they can be explained cleanly.
