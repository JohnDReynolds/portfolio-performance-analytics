# Return Reconstruction Roadmap

This note captures the long-term design idea for reconstructing and explaining
performance differences at both the portfolio level and the security level.

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

Only formula-level inputs should receive a counted `Performance Difference
Explained`. Supporting rows should be visible, but not additive.

## Possible Report Structure

A future workbook could include:

```text
Performance Differences
Identifiable Causes
Other Evidence
Return Reconstruction Checks
Raw Audit Trail
```

`Identifiable Causes` would contain formula-level inputs:

```text
holdings.begin_market_value
holdings.end_market_value
holdings.accrued
transactions.weighted_flow
transactions.income
```

Supporting rows could appear beneath those rows in the same sheet or in
`Other Evidence`.

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

Required concepts:

```yaml
method
beginning_value_source
ending_value_source
flow_source
flow_timing
day_count
inclusion_rule
flow_categories
income_categories
return_basis
sign_convention
```

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

### Phase 2: Portfolio-Level Flow Model

- Implement explicit portfolio-level Modified Dietz reconstruction.
- Treat contributions and withdrawals as portfolio-level external flows.
- Do not treat buys and sells as portfolio-level external flows.
- Compare derived portfolio return to reported portfolio return.

Initial implementation note: performance comparison now supports counted
portfolio external-flow Modified Dietz estimates when YAML specifies
`transaction_impact_methods.external_flow.double_count_policy:
count_as_explanation`. Full portfolio-return reconstruction and security-level
external-flow reconstruction remain future phases.

### Phase 3: Return Reconstruction Checks

- Add a worksheet or report section showing:
  - reported return
  - derived return
  - difference
  - materiality status
- Use this as a diagnostic before making reconstruction the primary user-facing
  explanation.

### Phase 4: Promote Formula Inputs

- Once reconstruction is trusted, use reconstructed formula inputs as the main
  explanation basis.
- Promote BMV, EMV, weighted flows, income, and accrual changes to
  `Identifiable Causes`.
- Demote quantity, price, commission, cost, and reference-data rows to
  supporting evidence unless they directly explain formula inputs.

### Phase 5: Improve Demo Data

- Add realistic examples:
  - portfolio contribution
  - portfolio withdrawal
  - security buy
  - security sell
  - dividend
  - interest
  - fee
  - accrual change
- Make all accounting internally consistent.
- Ensure the report demonstrates fully explained, partly explained, and
  unexplained cases.

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
