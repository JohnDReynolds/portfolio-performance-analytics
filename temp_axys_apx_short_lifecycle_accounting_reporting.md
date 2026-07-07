# Advent Axys/APX Short Position Accounting and Reporting Behavior (`ss` / `cs` Lifecycle)

_Date: July 7, 2026_

## Purpose

Research Advent Axys/APX short-position accounting and reporting behavior for a complete short-sale / cover-short lifecycle, with emphasis on evidence needed to build a defensible synthetic demo for performance comparison and Modified Dietz treatment.

The user already has evidence that lowercase `ss` maps to short sale / sell short and lowercase `cs` maps to cover short / buy to cover in APX integration mappings. This note focuses on accounting/reporting mechanics: open short positions, short proceeds, cover-short effects, performance reporting, and demo-safe rules.

---

## Executive conclusion

A synthetic Axys/APX short-sale demo is **defensible**, but only if it is clearly labeled as a **controlled synthetic scenario** rather than a claim about all real customer data.

The best-supported facts are:

- `ss` is the Axys/APX transaction code produced by integration mappings for **SELL / SHORT**.
- `cs` is the Axys/APX transaction code produced by integration mappings for **BUY / COVER SHORT**.
- `cs` maps to `$pty / $cash`.
- `ss` maps to `awus / none`, not ordinary `$pty / $cash`, which strongly suggests that short-sale proceeds are not necessarily ordinary unrestricted cash in the same way as a normal sale.
- Uppercase transaction codes such as `SS` or `CS` can represent deletion/reversal/cancellation behavior in APX/Axys integration contexts and should not be treated as economic short-sale or cover-short transactions without review.
- Consultant documentation for Axys/APX integrations refers to `SHORT`, `MARGIN`, and related short/margin cash sweep symbols, supporting the idea that short-related cash/proceeds may be handled through special cash buckets or sweep conventions.

The least-supported areas are:

- exact holdings-report signs for short positions;
- actual public Axys/APX Portfolio Appraisal rows showing short positions;
- exact APX Public View fields for short market value/cost;
- official SS&C Modified Dietz examples showing `ss`/`cs`.

Therefore, the safest packaged demo recommendation is:

> **Safe for packaged synthetic demo**, provided the demo states assumptions explicitly and does not claim that all production Axys/APX sites represent short positions identically.

---

## Source classification

| Source | Type | Relevance |
|---|---|---|
| ByAllAccounts / Morningstar Custodial Integrator User Guide for APX | Integration documentation / near-primary | Maps WebPortfolio transactions into APX transaction codes, source/destination types and symbols, and reversal behavior. |
| ByAllAccounts / Morningstar Custodial Integrator User Guide for Axys | Integration documentation / near-primary | Same mapping pattern for Axys, useful because it shows `ss`, `cs`, `awus`, `$pty`, `$cash`, and uppercase reversal behavior in Axys context. |
| WealthTechs AIA User Manual for APX Users | Consultant / integration documentation | Shows APX/Axys transaction-row style, Axys vs APX column-set notes, short/margin sweep symbols, and cancellation/uppercase examples. |
| WealthTechs AIA User Manual for Axys Users | Consultant / integration documentation | Shows similar short/margin cash sweep treatment for Axys. |
| SS&C Advent APX report brochures | Official SS&C marketing/report documentation | Confirms report families such as Portfolio Appraisal, holdings, cost, market value, transactions, realized/unrealized gain/loss, but does not document `ss/cs` mechanics. |
| Accounting/performance logic in this note | Inference | Used only where Axys/APX-specific evidence is unavailable. |

---

# 1. Open short position representation

## 1.1 What evidence was found?

### Verified / near-primary

The ByAllAccounts APX guide states that Custodial Integrator generates files for APX including transaction, position, and price files, and imports them using APX Import/Export. It also says transactions are delivered into the APX Trade Blotter for review and posting. This confirms that APX receives both transaction and position information through the integration workflow.

Source: ByAllAccounts / Morningstar, *Custodial Integrator User Guide for APX*, overview and data-translation sections.  
Relevant source note: the guide says Custodial Integrator produces “input for APX in the form of a transaction (Trade Blotter) file, position file, and price file,” then imports them using APX Import/Export.

### Verified code mapping

The APX transaction translation table maps:

```text
SELL / SHORT -> ss
Src/Dest Type = awus
Src/Dest Symbol = none
```

and:

```text
BUY / COVER SHORT -> cs
Src/Dest Type = $pty
Src/Dest Symbol = $cash
```

Source: ByAllAccounts / Morningstar, *Custodial Integrator User Guide for APX*, Table 1, Default Transaction Translation.  
URL: https://www.byallaccounts.net/Manuals/Custodial_Integrator/apx/CI_User_Guide.pdf

The Axys guide shows the same pattern for Axys.

Source: ByAllAccounts / Morningstar, *Custodial Integrator User Guide for Axys*, Table 1, Default Transaction Translation.  
URL: https://www.byallaccounts.net/Manuals/Custodial_Integrator/axys/CI_User_Guide.pdf

## 1.2 Negative quantity?

### Finding

No public official Advent/Axys/APX holdings row was found that proves open short positions are displayed with negative quantity in a standard Portfolio Appraisal or holdings report.

### Inference

For a synthetic demo, representing the open short as a **negative quantity** is the most conventional and defensible representation, but it must be labeled as an implementation assumption.

Example synthetic holding after short sale:

```text
Account  Security  PositionSide  Quantity  Price  MarketValue
DEMO     XYZ       Short         -100      50.00  -5,000.00
```

### Confidence

Medium as general accounting convention; Low as Axys/APX-specific public evidence.

---

## 1.3 Negative market value?

### Finding

No public Axys/APX sample row was found showing a short position with negative market value.

### Inference

A short position is an obligation to deliver a security. For performance and exposure purposes, negative market value / short liability treatment is the cleanest synthetic representation.

### Confidence

Medium as investment-accounting logic; Low as directly documented Axys/APX behavior.

---

## 1.4 Separate short security type?

### Finding

No public evidence was found that Axys/APX requires a distinct “short security type” for the same stock when shorted.

The mapping uses the transaction code `ss` and unusual source/destination fields rather than showing a separate short security type.

### Inference

Shortness is likely represented by transaction/position side, quantity/sign, account/sleeve, or short-cash accounting mechanics rather than by creating a different security master record.

### Confidence

Low to Medium.

---

## 1.5 Separate short account or separate short cash/proceeds bucket?

### Evidence

The WealthTechs AIA APX manual explicitly discusses removing short sweeps and gives the example:

```text
DP,CAUS,CASH,CAUS,SHORT
```

It also discusses margin sweeps:

```text
DP,CAUS,CASH,CAUS,MARGIN
```

Source: WealthTechs, *AIA User Manual for APX Users*, section on Remove Cash Sweeps / Remove Margin Sweeps / Remove Short Sweeps.  
URL: https://www.wealthtechs.com/files/AIADocumentation/APX%20Guide%20-%20AIA%20User%20Manual%20For%20APX%20Users.pdf

The Axys AIA manual similarly references short cash sweep transactions and excludes source/destination symbols such as `margin`, `short`, `dvwash`, `dvshrt`, `dvlong`, `cashrt`, `calong`, and `income` from ordinary cash sweep removal logic.

Source: WealthTechs, *AIA User Manual for Axys Users*.  
URL: https://www.wealthtechs.com/files/AIADocumentation/Axys%20Guide%20-%20AIA%20User%20Manual%20For%20Axys%20Users.pdf

### Interpretation

This supports the idea that some Axys/APX installations use distinct source/destination symbols for short cash, margin cash, or related sweep balances.

It does **not** prove that every Axys/APX implementation posts short-sale proceeds into a `SHORT` symbol.

### Confidence

Medium for the existence of short/margin sweep conventions; Low to Medium for exact proceeds accounting.

---

## 1.6 Can long and short positions in the same security coexist?

### Finding

No public Axys/APX evidence was found answering this directly.

### Inference

Operationally, coexistence may depend on:

- account structure,
- sleeve/subportfolio setup,
- tax-lot accounting,
- custodian feed treatment,
- security-location or long/short side indicators,
- whether the system nets positions at the report level.

A safe demo should avoid simultaneous long and short in the same security unless the demo explicitly uses separate accounts or sleeves.

### Confidence

Unknown.

---

# 2. Short-sale proceeds representation

## 2.1 Ordinary portfolio cash?

### Evidence

Normal sells in the ByAllAccounts APX mapping table are mapped to:

```text
SELL -> sl
Src/Dest Type = $pty
Src/Dest Symbol = $cash
```

But short sells are mapped to:

```text
SELL / SHORT -> ss
Src/Dest Type = awus
Src/Dest Symbol = none
```

Source: ByAllAccounts / Morningstar, *Custodial Integrator User Guide for APX*, Table 1.

### Interpretation

Because `ss` does **not** use the same `$pty / $cash` mapping as ordinary sells, it is not safe to assume short-sale proceeds are ordinary unrestricted portfolio cash.

### Confidence

High for the mapping; Medium for the accounting interpretation.

---

## 2.2 Short cash, restricted cash, margin cash, liability/collateral account?

### Evidence

WealthTechs APX documentation recognizes:

```text
CASH
MARGIN
SHORT
```

as source/destination symbols in sweep logic, including explicit short-sweep examples.

Source: WealthTechs, *AIA User Manual for APX Users*, Remove Short Sweeps.  

### Interpretation

Short-related cash/proceeds may be represented through short cash, margin cash, or other site-specific cash-bucket mechanics.

### Confidence

Medium.

---

## 2.3 Source/destination type and symbol examples

| Scenario | Code / source-destination pattern | Evidence | Confidence |
|---|---|---:|---:|
| Normal sell | `sl / $pty / $cash` | ByAllAccounts APX/Axys mapping | High |
| Short sell | `ss / awus / none` | ByAllAccounts APX/Axys mapping | High |
| Cover short | `cs / $pty / $cash` | ByAllAccounts APX/Axys mapping | High |
| Margin interest | `ai / $pth / $cash / caus margin` | ByAllAccounts APX mapping | High |
| Margin sweep example | `DP,CAUS,CASH,CAUS,MARGIN` | WealthTechs APX AIA | Medium |
| Short sweep example | `DP,CAUS,CASH,CAUS,SHORT` | WealthTechs APX AIA | Medium |

---

# 3. Cover-short transaction representation

## 3.1 Cash effect

### Evidence

The APX and Axys mapping tables map:

```text
BUY / COVER SHORT -> cs
Src/Dest Type = $pty
Src/Dest Symbol = $cash
```

### Interpretation

Cover short uses portfolio cash mechanics, but exact sign convention is not publicly documented.

A synthetic demo should show cash decreasing when the cover is executed.

### Confidence

High for mapping; Medium for sign convention.

---

## 3.2 Quantity effect

### Evidence

No public row-level Axys/APX example was found.

### Inference

A cover short reduces or closes short exposure. If the open short is represented as negative quantity, then covering 100 shares would move quantity from `-100` to `0`.

### Confidence

Medium as accounting logic; Low as direct Axys/APX sample evidence.

---

## 3.3 Market value effect

### Inference

A cover short reduces or eliminates the negative market value / short liability. If fully covered, the short position should disappear from holdings or show zero quantity/value.

### Confidence

Medium as accounting logic; Low as direct Axys/APX evidence.

---

## 3.4 Realized gain/loss

### Inference

Gain/loss is realized when the short is covered:

- If shorted at 50 and covered at 40, gain = 10 per share.
- If shorted at 50 and covered at 60, loss = 10 per share.

This is performance-impacting and should not be treated as an external cash flow.

### Confidence

High as general accounting logic; Low as specific public Axys/APX report evidence.

---

# 4. Sanitized examples found

## 4.1 Actual `ss` / `cs` rows

No public sanitized row was found showing an actual Axys/APX transaction row containing lowercase `ss` or lowercase `cs`.

### Confidence

High that no such row was found in this research pass; not proof that none exists.

---

## 4.2 Near examples: APX/Axys transaction-row format

WealthTechs AIA documentation shows APX/Axys-style transaction rows such as:

```text
ACCTX,010117,LI,100,100,CAUS,MMF,CAUS,CASH
ACCTX,010117,LO,100,100,CAUS,MMF,CAUS,CASH
ACCTX,010117,DP,100,100,CAUS,MMF,CAUS,CASH
ACCTX,010117,WD,100,100,CAUS,MMF,CAUS,CASH
```

It also shows a buy transaction example:

```text
ACCTX, BY, BUY APPLE INC. NASDAQ @100.00
,01012016,0101012016, CSUS,AAPL,100,CAUS,CASH,10000
```

and a cancellation example:

```text
acct123,010101,010101,by,csus,appl,100,caus,cash,10000
acct123,010101,010101,BY,csus,appl,100,caus,cash,10000
```

Source: WealthTechs, *AIA User Manual for APX Users*.

These are not `ss`/`cs` examples, but they are useful because they show the style of transaction fields that appear in source TRN/APX import rows.

---

## 4.3 Synthetic lifecycle example for demo

The following example is **inferred / synthetic**, not copied from an Axys/APX file.

### Step 1: Sell short 100 XYZ at $50

```text
Date        Code  SecType  Symbol  Qty    Price   Amount   SrcDestType  SrcDestSymbol
2026-01-05  ss    CSUS     XYZ     100    50.00   5000.00  awus         none
```

Possible resulting holding:

```text
Date        Symbol  PositionSide  Quantity  Price   MarketValue
2026-01-05  XYZ     Short         -100      50.00   -5000.00
```

Possible cash/proceeds representation:

```text
Date        CashBucket  Amount
2026-01-05  SHORT       5000.00
```

### Step 2: Mark to market at $45

```text
Date        Symbol  PositionSide  Quantity  Price   MarketValue  UnrealizedGainLoss
2026-01-31  XYZ     Short         -100      45.00   -4500.00     +500.00
```

### Step 3: Cover 100 XYZ at $40

```text
Date        Code  SecType  Symbol  Qty    Price   Amount   SrcDestType  SrcDestSymbol
2026-02-10  cs    CSUS     XYZ     100    40.00   4000.00  $pty         $cash
```

Result:

```text
Date        Symbol  Quantity  MarketValue  RealizedGainLoss
2026-02-10  XYZ     0         0.00         +1000.00
```

### Important caveat

The above rows are suitable as **synthetic demo rows only**. They should not be presented as actual Axys/APX export rows.

---

# 5. Performance reporting / Modified Dietz treatment

## 5.1 Beginning and ending market value sign

### Evidence

No public Axys/APX performance report example showing short positions was located.

### Inference

For a defensible synthetic demo, use negative market value for open short exposure.

Example:

```text
BMV = -5,000
EMV = -4,500
Security return effect = positive because liability declined
```

### Confidence

Medium as accounting/performance logic; Low as Axys/APX-specific public evidence.

---

## 5.2 Security-level return

### Inference

Short security-level performance is directionally opposite the underlying price move:

- price down -> short gains
- price up -> short loses

The mechanics depend on the performance report’s treatment of short market value, proceeds, collateral, and cash buckets. For a synthetic demo, report both:

1. short security liability exposure, and
2. cash/proceeds bucket,

so the portfolio-level reconciliation is transparent.

### Confidence

Medium.

---

## 5.3 Portfolio Modified Dietz

### Recommended treatment

`ss` and `cs` should not be investor external flows.

They are internal trades:

- `ss` creates short liability and proceeds/collateral/cash mechanics.
- `cs` uses cash/proceeds to reduce or close short liability.
- realized/unrealized gain/loss belongs in performance.

### Reasoning

An investor did not contribute capital merely because the portfolio sold a security short, and did not withdraw capital merely because the portfolio covered the short.

### Confidence

Medium. The performance conclusion is inference from transaction semantics and investment accounting, not a public SS&C performance manual example.

---

## 5.4 Realized gain/loss

### Inference

Realized gain/loss should be reflected when a cover short closes or reduces the short exposure.

### Confidence

Medium.

---

# 6. Uppercase `SS` / `CS` warnings

## Evidence

ByAllAccounts APX documentation states that reversal transactions are translated by converting the original transaction type code to uppercase. It gives the example that `by` becomes `BY`, and says uppercase is APX’s representation for a transaction to be deleted.

Source: ByAllAccounts / Morningstar, *Custodial Integrator User Guide for APX*, Reversal Transactions section.

WealthTechs AIA documentation also shows cancellation by uppercasing a prior code:

```text
by -> BY
```

It also refers to transaction translation logic that can delete transactions from the Trade Blotter including codes such as:

```text
BY SL SS CS
```

Source: WealthTechs, *AIA User Manual for APX Users*.

## Recommendation

Do not treat uppercase `SS` or `CS` as economic short sale / cover short by default.

Classify uppercase rows as:

```text
review_only_reversal_or_delete_possible
```

### Confidence

High.

---

# 7. Confidence table

| Topic | Finding | Confidence | Source basis |
|---|---|---:|---|
| `ss` means sell short | Yes | High | ByAllAccounts APX and Axys transaction mapping |
| `cs` means cover short | Yes | High | ByAllAccounts APX and Axys transaction mapping |
| `ss` maps to `awus / none` | Yes | High | ByAllAccounts APX and Axys mapping |
| `cs` maps to `$pty / $cash` | Yes | High | ByAllAccounts APX and Axys mapping |
| Normal sell maps differently from short sell | Yes: normal `sl / $pty / $cash`; short `ss / awus / none` | High | ByAllAccounts mapping |
| Short/margin sweep symbols exist in integration docs | Yes: `SHORT`, `MARGIN` examples | Medium | WealthTechs AIA APX/Axys documentation |
| Open short shown as negative quantity | Likely, but not directly proven by public Axys/APX row | Low/Medium | Inference |
| Open short shown as negative market value | Likely, but not directly proven by public Axys/APX row | Low/Medium | Inference |
| Short proceeds are ordinary unrestricted cash | Not safe to assume | Medium | Difference between `ss` and normal sell mapping |
| `cs` reduces/closes negative exposure | Yes as accounting logic | Medium | Inference from code meaning |
| Cover creates realized gain/loss | Yes as accounting logic | Medium | Inference |
| `ss/cs` are external Modified Dietz flows | No; should be internal trades | Medium | Inference from transaction semantics |
| Actual public `ss/cs` row found | No | High | Research result |
| Uppercase `SS/CS` may be reversal/delete/cancel | Yes | High | ByAllAccounts and WealthTechs |

---

# 8. Minimum fields required for a safe demo scenario

For a packaged synthetic demo:

```yaml
minimum_fields:
  account_or_portfolio_id: required
  transaction_date: required
  settlement_date: preferred
  transaction_code: required
  transaction_code_case: required
  security_type: required
  security_id_or_symbol: required
  quantity: required
  price: required
  amount: required
  source_destination_type: required
  source_destination_symbol: required
  cash_bucket_or_cash_security: strongly_preferred
  position_side: strongly_preferred
  beginning_quantity: required_for_lifecycle_demo
  ending_quantity: required_for_lifecycle_demo
  beginning_market_value: required_for_performance_demo
  ending_market_value: required_for_performance_demo
  realized_gain_loss: preferred
  unrealized_gain_loss: preferred
```

For production/customer-data classification:

```yaml
production_safeguards:
  - require_lowercase_ss_or_cs
  - require_security_identifier
  - require_quantity
  - require_amount
  - require_source_destination_fields
  - require_short_context:
      any_of:
        - prior_negative_position
        - resulting_negative_position
        - explicit_short_cash_symbol
        - explicit_margin_or_short_sweep_symbol
        - source_destination_pattern_matches_known_short_mapping
  - if_uppercase_code: review_only
  - if_missing_short_context: review_only
  - if_site_custom_mapping_known: require_site_override
```

---

# 9. Recommended synthetic-demo lifecycle

## Demo assumption set

```yaml
demo_assumptions:
  short_position_quantity_sign: negative
  short_position_market_value_sign: negative
  short_sale_code: ss
  cover_short_code: cs
  short_sale_src_dest_type: awus
  short_sale_src_dest_symbol: none
  cover_short_src_dest_type: "$pty"
  cover_short_src_dest_symbol: "$cash"
  short_proceeds_bucket: SHORT
  portfolio_modified_dietz_external_flow: false
  performance_effect: realized_and_unrealized_gain_loss
```

## Demo lifecycle

```yaml
short_lifecycle:
  - step: sell_short
    transaction_code: ss
    quantity_input: 100
    resulting_position_quantity: -100
    price: 50
    short_market_value: -5000
    proceeds_bucket: SHORT
    external_flow: false

  - step: mark_to_market
    price: 45
    resulting_short_market_value: -4500
    unrealized_gain_loss: 500
    external_flow: false

  - step: cover_short
    transaction_code: cs
    quantity_input: 100
    price: 40
    cash_used: 4000
    resulting_position_quantity: 0
    realized_gain_loss: 1000
    external_flow: false
```

---

# 10. Final recommendation

## Recommendation

**Safe for packaged synthetic demo.**

But only if the package explicitly states:

1. `ss` and `cs` code meanings are supported by Axys/APX integration mapping evidence.
2. The exact open-position representation is synthetic because no public Axys/APX holdings row was found.
3. Short-sale proceeds should not be assumed to be ordinary unrestricted cash.
4. The demo uses `SHORT` / margin-style cash bucket assumptions only as a clearly labeled synthetic model.
5. Uppercase `SS` and `CS` are excluded from economic treatment because they may indicate reversal/delete/cancel behavior.
6. In Modified Dietz, `ss` and `cs` are modeled as internal trades, not investor external flows.

## Recommended label

```text
safe_for_packaged_synthetic_demo_with_disclosed_assumptions
```

## Not recommended

Do **not** ship a blanket production rule that treats every `ss` and `cs` row as a clean short sale / cover short without validating:

- code case,
- security,
- quantity,
- amount,
- source/destination fields,
- short-position context,
- and site-specific transaction mapping.

---

# Source appendix

## ByAllAccounts / Morningstar Custodial Integrator User Guide for APX

Type: Integration documentation / near-primary.

Key source facts:

- Custodial Integrator generates APX transaction, position, and price files and imports them via APX Import/Export.
- Transactions are delivered into an APX Trade Blotter for review and posting.
- `BUY / COVER SHORT -> cs`
- `SELL / SHORT -> ss`
- `cs -> $pty / $cash`
- `ss -> awus / none`
- Reversal transactions are represented by uppercase transaction type codes.

URL: https://www.byallaccounts.net/Manuals/Custodial_Integrator/apx/CI_User_Guide.pdf

## ByAllAccounts / Morningstar Custodial Integrator User Guide for Axys

Type: Integration documentation / near-primary.

Key source facts:

- Same transaction translation pattern as APX.
- `BUY / COVER SHORT -> cs`
- `SELL / SHORT -> ss`
- `cs -> $pty / $cash`
- `ss -> awus / none`
- Uppercase reversal/delete behavior.

URL: https://www.byallaccounts.net/Manuals/Custodial_Integrator/axys/CI_User_Guide.pdf

## WealthTechs AIA User Manual for APX Users

Type: Consultant / integration documentation.

Key source facts:

- Shows APX/Axys-style transaction row examples.
- Discusses Axys and APX column sets, with APX similar to Axys but with additional fields.
- Discusses cash sweeps, margin sweeps, and short sweeps.
- Gives examples:
  - `DP,CAUS,CASH,CAUS,MARGIN`
  - `DP,CAUS,CASH,CAUS,SHORT`
- Shows uppercase cancellation behavior.

URL: https://www.wealthtechs.com/files/AIADocumentation/APX%20Guide%20-%20AIA%20User%20Manual%20For%20APX%20Users.pdf

## WealthTechs AIA User Manual for Axys Users

Type: Consultant / integration documentation.

Key source facts:

- Discusses short cash sweep transactions for Axys.
- Excludes source/destination symbols such as `margin`, `short`, `dvwash`, `dvshrt`, `dvlong`, `cashrt`, `calong`, and `income` from ordinary cash sweep removal logic.

URL: https://www.wealthtechs.com/files/AIADocumentation/Axys%20Guide%20-%20AIA%20User%20Manual%20For%20Axys%20Users.pdf

## SS&C Advent APX report materials

Type: Official SS&C Advent marketing/report documentation.

Key source facts:

- Confirms standard report families such as Portfolio Appraisal, holdings, quantity, cost, market value, realized/unrealized gains and losses, and transaction reporting.
- Does not provide enough detail to prove `ss/cs` mechanics.

Representative URL: https://cdn.advent.com/cms/pdfs/reports/REP_APX.pdf
