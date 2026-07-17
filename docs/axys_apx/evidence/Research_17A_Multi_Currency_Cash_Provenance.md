# Multi-Currency Cash-Provenance Evidence Ledger

> Focused provenance for the purchase cash-bucket boundary in
> [`../reference/Chapter_17_Multi_Currency.md`](../reference/Chapter_17_Multi_Currency.md).
> This ledger does not define a native Axys/APX cash account, IMEX schema, or FX
> linkage rule.

## Ownership Boundary

- General currency, FX, valuation, and performance evidence belongs in
  [`Research_17_Multi_Currency.md`](Research_17_Multi_Currency.md).
- Cash tokens and sweep/journal evidence belong in
  [`Research_07_Cash.md`](Research_07_Cash.md).
- Transaction code meanings and cancellation boundaries belong in
  [`Research_05_Transactions.md`](Research_05_Transactions.md).
- This file owns only the evidence ladder for deciding whether a purchase's
  funding cash bucket is proven, inferred, or Unknown.

This file replaces the AI-assisted intake `_temp_deep-research-report.md`,
merged on 2026-07-13. Its `turn...` citation tokens were internal to the
originating research session and were not durable. They have been removed;
material claims now point to stable repository sources. Git history retains the
superseded synthesis.

## Source Register

| ID | Source | Use in this ledger |
|---|---|---|
| MCP-S01 | [Transaction Evidence Ledger](Research_05_Transactions.md) | Observed `by` row shape, source/destination context, code/case boundaries, and code-only caution. |
| MCP-S02 | [Cash Evidence Ledger](Research_07_Cash.md) | Cash-like tokens, sweep/journal behavior, and native cash-object gaps. |
| MCP-S03 | [Holdings Evidence Ledger](Research_06_Holdings.md) | Cash-as-position, holdings-delta, currency, and extraction boundaries. |
| MCP-S04 | [Multi-Currency Evidence Ledger](Research_17_Multi_Currency.md) | Official capability versus native field/mechanic boundary and FX-link requirements. |
| MCP-S05 | [Demo Source Contract](../../audit/demo_source_contract.md) | Guardrail against naming a specific cash security when source evidence does not prove it. |
| MCP-S06 | [Official Axys product page](https://www.advent.com/solutions/axys/) | Capability context for security currencies, multicurrency reporting, and FX interfaces. |

## Cash-Provenance Claims

| Claim | Evidence | Confidence | Boundary |
|---|---|---:|---|
| MCP-C001 | Public integration examples show `by` purchase rows with security, quantity, amount, and source/destination type and symbol, including a cash-like `caus,cash` leg. | Medium; MCP-S01 | Partial integration row, not a complete native transaction layout. |
| MCP-C002 | Source/destination type and symbol are the strongest observed transaction-level evidence for the funding cash leg when the site validates them. | Medium-High for observed workflow | They may identify cash context without proving a bank, custody account, or canonical posted cash field. |
| MCP-C003 | APX integration evidence exposes analogous transaction source/destination type and symbol concepts. | Medium; MCP-S01 | Staging/translation evidence, not a native posted-account schema. |
| MCP-C004 | No reviewed public source establishes universal native fields such as `source_cash_account`, `destination_cash_account`, `settlement_account`, or a bank-account identifier. | Unknown native schema | Exact account-level provenance cannot be guaranteed publicly. |
| MCP-C005 | Cash may be represented as holdings/security rows in normalized or site workflows, but public evidence does not establish a vendor-standard cash-security naming convention. | Medium as implementation pattern; MCP-S02-S03 | Names such as `CASHEUR`, `CASH_EUR`, or `EUR` require site mapping. |
| MCP-C006 | `CASH`, `MMF`, `MARGIN`, `SHORT`, `$cash`, `$income`, and `CAUS` are observed cash-context tokens. | Medium-High for integration examples; MCP-S02 | They are not interchangeable or universal native cash buckets. |
| MCP-C007 | If explicit cash-leg fields are absent and exactly one currency-consistent cash holding reconciles the purchase, the bucket can be strongly inferred. | High as conservative inference rule | This is audit reasoning, not a published Axys/APX formula. |
| MCP-C008 | Security trading currency can narrow likely funding currency, but cannot by itself prove the cash security, FX conversion path, or settlement account. | High caution; MCP-S04 | Cross-currency purchases require separate FX and cash evidence. |
| MCP-C009 | If multiple cash buckets, sweeps, journals, margin/short balances, or contradictory movements remain, the exact source must stay weakly inferred or Unknown. | High as safety boundary | False precision is worse than an explicit evidence gap. |
| MCP-C010 | A purchase is investment trade activity, not an external contribution merely because cash decreases. | High conceptually; MCP-S01 | External-flow classification remains separate from funding provenance. |
| MCP-C011 | Uppercase `BY` can be cancellation/delete evidence in observed integrations and must be resolved before funding inference. | Medium-High for workflow; MCP-S01 | Uppercase universality remains Unknown. |
| MCP-C012 | Cash provenance may be strongest in translation or blotter data and weaker after posting or in summarized reports. | Medium as lifecycle inference | Preserve the earliest reliable source row and lineage. |

## Evidence Ladder

| State | Minimum evidence | Permitted conclusion |
|---|---|---|
| Proven | Site-validated source/destination or cash-leg field uniquely identifies the cash bucket. | Name the evidenced cash bucket and retain the source row. |
| Strongly inferred | No explicit leg, but exactly one currency-consistent cash holding reconciles after related FX, sweep, journal, and reinvestment rows. | Name the inferred bucket and disclose the inference basis. |
| Weakly inferred | Funding currency is plausible, but multiple cash buckets or incomplete linkage prevent uniqueness. | State likely currency exposure; do not name an exact cash security. |
| Unknown | Missing or contradictory cash, holdings, currency, FX, or transaction evidence. | Report that funding provenance is not established. |

## Required Context for an Inference

| Evidence group | Fields or observations to preserve |
|---|---|
| Transaction | Portfolio, trade/settlement dates, native code and case, security, quantity, amount/net amount, transaction currency, source/destination fields, and source row ID. |
| Security | Identifier/type plus explicit trading or settlement currency when supplied. |
| Cash holdings | Candidate cash-like securities/buckets, currency, beginning/ending quantity or value, and extraction source. |
| Related activity | FX conversions, sweeps, journals, fees, reinvestment legs, margin/short movements, and reversals. |
| Provenance result | Confidence state, inferred currency, inferred bucket only when permitted, evidence links, and unresolved alternatives. |

## Contradictions and Interpretation Risks

| ID | Tension | Resolution |
|---|---|---|
| MCP-X001 | A source/destination symbol may say cash without identifying a unique account or security. | Distinguish cash-context proof from exact-bucket proof. |
| MCP-X002 | Security trading currency suggests a funding currency but a separate FX trade or settlement currency may intervene. | Require FX/cash evidence before promoting beyond inference. |
| MCP-X003 | Cash-like holdings may include unrestricted cash, money markets, margin, short proceeds, income, or sweep vehicles. | Use site classification and reconcile all related movements. |
| MCP-X004 | Normalized demo identifiers are explicit but are not native Axys/APX names. | Treat them as demo contract values only. |
| MCP-X005 | A `by` example contains useful fields but is not a complete native row specification. | Preserve observed shape without inventing missing columns or universality. |

## Evidence Needed to Resolve the Boundary

| Need | Evidence that would resolve or materially narrow it |
|---|---|
| MCP-U001 Posted cash linkage | Native transaction/audit export showing a purchase and its unique posted cash leg or account identifier. |
| MCP-U002 Cash-security model | Site security master and holdings extracts defining every currency cash, sweep, margin, income, and short-proceeds bucket. |
| MCP-U003 Cross-currency purchase | One complete purchase with trading/settlement currencies, FX rows, cash movements, holdings, and report output. |
| MCP-U004 Lifecycle persistence | Translation, blotter, posted transaction, audit, holdings, and report rows for the same purchase. |
| MCP-U005 APX/Axys differences | Matched native examples proving whether source/destination and account linkage persist differently in each system. |

## Maintenance Rule

Add evidence only when it changes the provenance ladder, proves a native cash
link, or reveals a contradiction. Reader guidance stays in Chapter 17; general
cash and transaction semantics stay in their owning ledgers.
