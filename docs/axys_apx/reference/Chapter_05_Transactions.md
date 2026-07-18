# Chapter 05 — Transactions

> **Repository chapter:** `docs/axys_apx/reference/Chapter_05_Transactions.md`
> **Status:** Technical reference chapter based on repository research and
> public evidence reviewed through 2026-07-17.
> **Evidence standard:** Facts are marked as Verified, High Confidence,
> Medium Confidence, Low Confidence, or Unknown.
> **Important limitation:** The supplied material does not include
> official complete Axys or APX transaction-code manuals, official IMEX
> transaction object definitions, complete native Trade Blotter layouts,
> native Axys/APX transaction storage schemas, or complete REP
> specifications. Unsupported information is marked **Unknown**.

------------------------------------------------------------------------

## Related chapters

- [Chapter_01_Overview.md](Chapter_01_Overview.md) — provides the repository-wide map and evidence conventions.
- [Chapter_04_Security_Master.md](Chapter_04_Security_Master.md) — transaction interpretation depends on security identity.
- [Chapter_06_Holdings.md](Chapter_06_Holdings.md) — transactions drive holdings, lots, and cost-basis updates.
- [Chapter_10_Performance.md](Chapter_10_Performance.md) — transactions feed performance and attribution inputs.

## Canonical Use of This Chapter

This chapter is the canonical reader-facing reference for Axys/APX transaction
semantics in this repository. Other chapters should describe their local effects
on holdings, cash, corporate actions, performance, or reporting, but should not
maintain separate transaction-code dictionaries. When a transaction code appears
in another chapter, interpret it through this chapter and the implementation
contract in
[`../contracts/transaction_semantics_matrix.yaml`](../contracts/transaction_semantics_matrix.yaml).

The evidence ledger in
[`../evidence/Research_05_Transactions.md`](../evidence/Research_05_Transactions.md)
preserves granular source claims, confidence boundaries, contradictions, and
missing-evidence requirements. If this chapter, the evidence ledger, and the
implementation contract appear to disagree, treat that as a documentation
cleanup issue rather than as three independent sources of truth.

## 1. Overview

Transactions are the central accounting events in Axys/APX-style
portfolio accounting workflows. They connect economic activity to
holdings, cash, tax lots, cost basis, realized gain/loss, income,
performance, reports, IMEX, REP, reconciliation, and audit workflows.
**Confidence:** High for general accounting role; Medium for native
Axys/APX mechanics because public source material is largely third-party
integration or migration evidence.

A practical transaction lifecycle supported by the supplied research is:

``` text
Economic event
    ↓
External source-data
    ↓
Normalization and translation
    ↓
Validation
    ↓
Trade Blotter or staging area
    ↓
Review / exception handling
    ↓
Posting
    ↓
Accounting records updated
    ↓
Holdings, cash, lots, cost basis, income, and realized gain/loss updated
    ↓
Performance, reports, IMEX, REP, reconciliation, and audit
```

### 1.1 Evidence Boundaries

| Area | Status | Confidence | Notes |
| --- | --- | --- | --- |
| Transactions affect holdings, cash, income, cost basis, realized gain/loss, and performance inputs | Supported conceptually and by supplied research | High | General accounting role is strong; native mechanics vary. |
| Trade Blotter workflows exist in Axys/APX integration evidence | Supported | Medium | Evidence is third-party integration documentation. |
| Observed transaction codes such as `by`, `sl`, `li`, `lo`, `dv`, `in`, `dp`, `wd` | Supported as observed codes | Medium | Not a complete official code matrix. |
| Uppercase cancellation instruction, e.g. `by` → `BY` | Supported in reviewed Trade Blotter staging/control workflows | Medium-High | Not established as an exportable posted transaction or a universal native convention. |
| Complete native Axys transaction-code matrix | Unknown | Unknown | Not supplied. |
| Complete native APX transaction-code matrix | Unknown | Unknown | Not supplied. |
| Official IMEX transaction object names | Unknown | Unknown | Not supplied. |
| Native Axys transaction storage model | Unknown | Unknown | Not supplied. |
| Native APX database transaction schema | Unknown | Unknown | Not supplied. |
| Native audit trail and posting-status model | Unknown | Unknown | Not supplied. |

### 1.2 Practical Classification Guidance

For performance, audit, and reconciliation work, it is useful to classify
transactions into practical buckets such as external cash flows,
trading activity, income, fees or expenses, corporate actions, and
reversals or cancellations. External cash flows are especially important
because deposits, withdrawals, transfers in or out, and cash sweeps can
materially change cash balances and performance cash-flow interpretation.

Observed examples in the supplied research include `by` and `sl` for
security trades, `dv`, `in`, and `ai` for income-related activity,
`li` and `lo` for transfer/external-flow candidates, and `dp` and `wd`
for cash-related movement. The meaning of any one code should be
determined with the surrounding context: sign, quantity, amount, security
type, source/destination type, source/destination symbol, special
security markers, performance/cash-flow flag if available, and
firm-specific translation rules.

For performance work, `li` and `lo` are the most important external-flow
candidates, but they are not sufficient by themselves. A cash
contribution pattern may use `li` with cash-like source/destination
fields such as `$pty` and `$cash`; a security-in-kind transfer may also
use `li` with a non-cash security and quantity. Similarly, `lo` can
represent a client withdrawal or an outgoing security transfer, but it
can also appear in debit, correction, or internal movement contexts.
Therefore, external-flow treatment should be assigned only after
checking code, security type, source/destination fields, signs, special
symbols, and firm-specific mapping.

Recommended classification order for audit products:

1. Establish the source stage before interpreting a reversal or cancellation.
   An uppercase-code cancellation instruction is supported only for the reviewed
   Trade Blotter staging/control workflow, not for an ordinary posted extract.
2. Apply firm-specific transaction translation overrides, because public
   integration tables explicitly allow special cases and
   financial-institution-specific customization.
3. Interpret security type and symbol, including cash, fee, income,
   margin, short, and dividend-wash symbols.
4. Interpret the code family: trade, transfer, income/accrual, fee/cash,
   corporate action, principal event, or journal/other.
5. Use amount and quantity signs to determine direction.
6. Assign performance treatment: external flow, internal return event,
   correction/reversal, non-performance event, or unknown.

### 1.3 Interpretation Rule

Transaction code alone is not sufficient to determine accounting
meaning. Interpretation may depend on:

| Context Item | Why It Matters | Confidence |
| --- | --- | --- |
| Transaction code | Primary event indicator. | High |
| Quantity sign | May determine inflow versus outflow. | Medium |
| Amount sign | May determine cash direction. | Medium |
| Security type | May distinguish cash, security, bond, fee, margin, sweep, or short activity. | Medium |
| Source/destination type | May define the offsetting side of the accounting entry. | Medium |
| Source/destination symbol | May identify cash, margin, short, or wash symbols. | Medium |
| Special security type/symbol | Used in observed fee and expense handling. | Medium |
| Portfolio/account configuration | May affect interpretation, including deliver-in/out behavior. | Medium |
| Custodian or interface translation | Integration-specific mappings may alter native codes. | Medium |
| Reversal/cancellation context | Explicit Trade Blotter source stage plus uppercase instruction may identify deletion/cancellation in the observed workflow. | Medium |

### 1.4 Transaction Matching Boundary

Transaction matching for audit and performance-comparison work should be
conservative. A source transaction identifier, when present and trustworthy, is
the preferred linkage between snapshots. Without that identifier, fallback
matching should use only strict one-to-one evidence such as a single transaction
on the same portfolio, trade date, security identifier, and transaction code in
both snapshots. If more than one candidate exists on either side, or if the
trade date, identifier, or code changed, the records should remain unmatched
unless site-specific evidence proves the linkage.

This boundary is intentional. Two identical transactions can legitimately occur
on the same day, transactions can be added or deleted between snapshots, and a
date change can alter weighted cash-flow treatment. Fuzzy matching may create a
plausible story that is wrong. When a transaction cannot be matched, reports
should show the Snapshot A and Snapshot B rows as unmatched while still
explaining the Modified Dietz effect of the observed source-data changes.

Observed AIA transaction-translation conditions are evaluated
case-insensitively inside that integration tool. Separately, the APX AIA guide
describes APX as case-sensitive for selected identifiers. These statements are
not contradictory: evaluator behavior does not authorize normalization of
native transaction codes, security identifiers, or account codes. PPAR should
preserve their exact source case and perform case-insensitive comparison only
under an explicit, versioned site contract.

------------------------------------------------------------------------

## 2. Axys

### 2.1 Axys Transaction Role

| Statement | Confidence | Notes |
| --- | --- | --- |
| Axys supports portfolio accounting workflows involving transactions, positions, settlement/trade information, tax-lot or average-cost accounting, reporting, performance measurement, and reconciliation. | High for broad capability | Source material identifies this from SS&C/Advent product-level evidence, but not detailed mechanics. |
| Axys transaction import workflows can route transactions through a Trade Blotter for review and posting. | Medium | Supported by ByAllAccounts and WealthTechs integration evidence. |
| Axys native transaction file structure is fully known from the supplied material. | Unknown | Not supplied. |
| Axys native transaction-code matrix is fully known from the supplied material. | Unknown | Not supplied. |

### 2.2 Axys Trade Blotter and IMEX Workflow

The supplied research describes an Axys-oriented third-party integration
workflow:

``` text
External financial institution data
    ↓
Aggregation / normalization layer
    ↓
Security and portfolio translation
    ↓
Transaction Trade Blotter file
    ↓
Axys IMEX import
    ↓
Trade Blotter review
    ↓
Post to Axys
```

**Confidence:** Medium. This is observed third-party integration
behavior, not proof of exclusive native Axys behavior.

### 2.3 Axys Files, Folder Labels, and Utilities Observed in Integration Evidence

| Item | Observed Role | Confidence | Caveat |
| --- | --- | --- | --- |
| `topost.trn` | Trade Blotter file receiving transaction imports. | Medium | Third-party integration evidence. |
| `$pathtrn` | Axys user folder label for Trade Blotter location. | Medium | Integration workflow evidence. |
| `imex32.exe` | Axys Import/Export utility referenced by Custodial Integrator. | Medium | Exact native behavior and version coverage Unknown. |
| IMEX logs | Logs generated during import. | Medium | Exact log fields and messages Unknown. |
| `$pathcli` | Axys portfolio/client files; `*.cli`; used to create portfolio-code list in one workflow. | Medium | Integration workflow evidence. |
| `$pathinf` | Contains `sec.inf` and `type.inf`; exported by integration software to generate transactions and positions. | Medium | Integration workflow evidence. |
| `$pathpri` | Axys price-file folder; `*.pri`. | Medium | Integration workflow evidence. |
| `$pathlog` | Folder where Axys Import/Export logs are written. | Medium | Integration workflow evidence. |
| `*.cli` | Client/portfolio files referenced in conversion and integration evidence. | Medium | Native full layout Unknown. |

### 2.4 Axys `.cli` and Conversion Evidence

| Topic | Axys Evidence | Confidence | Notes |
| --- | --- | --- | --- |
| Per-share cost basis | Morningstar conversion evidence states per-share cost-basis data is converted only if provided in exported Advent `.cli` file. | Medium | This is conversion evidence, not a full native `.cli` spec. |
| Deliver-in / deliver-out interpretation | `li` and `lo` may be interpreted differently depending on a transaction-setting code in the Advent client file. | Medium | Code-only interpretation is unsafe. |
| 53rd-character setting | Setting code `Y` maps `li`/`lo` to Deliver-In/Deliver-Out in Morningstar conversion; setting code `N` maps them to Credit/Debit of Security. | Medium | Specific to supplied conversion evidence. |
| `none` or `client` securities | Transactions linked to securities labeled `none` or `client` may be converted as out-of-pocket cash. | Medium | Conversion-layer behavior. |
| Principal paydowns | Principal paydowns from Axys may create conversion complications, including zero-quantity cases. | Medium | Native Axys mechanics Unknown. |
| Transaction and historical prices | Transaction prices and historical security prices may be converted if present in Axys conversion inputs. | Medium | Exact native field names Unknown. |

### 2.5 Axys Reinvestment Evidence

| Statement | Confidence | Notes |
| --- | --- | --- |
| Axys distribution reinvestment activity may appear as Buy plus Distribution transaction pairs in conversion data. | Medium | Based on Morningstar Axys conversion evidence. |
| Reinvestment representation can affect downstream realized and unrealized gain/loss reporting. | Medium | Conversion observation. |
| Native Axys reinvestment representation is fully defined by supplied material. | Unknown | Not supplied. |

### 2.6 Axys Fee Evidence

| Item | Observed Meaning in Supplied Material | Confidence | Caveat |
| --- | --- | --- | --- |
| `epus` | Associated with Management Fee conversion in Morningstar Axys conversion evidence. | Medium | May be a transaction code, label, security type, or conversion-layer term; official definition Unknown. |
| `exus` | Associated with Expense conversion in Morningstar Axys conversion evidence. | Medium | May be a transaction code, label, security type, or conversion-layer term; official definition Unknown. |

### 2.7 Axys Cancellation / Reversal Evidence

| Statement | Confidence | Notes |
| --- | --- | --- |
| WealthTechs APX integration evidence documents a tool that creates a cancellation Trade Blotter by uppercasing the original transaction code, e.g. `by` → `BY`. | High Confidence for that staging/control workflow | The row is an instruction prepared from historical transaction files before review/posting. |
| The uppercase cancellation instruction is available as a posted transaction through REP, IMEX/APXIX, SQL, REST, or ordinary reports. | Unknown | The reviewed evidence does not establish this. |
| Uppercase cancellation behavior is universal across all Axys versions, transaction types, and import methods. | Unknown | Not supported by supplied material. |

### 2.8 Missing-Cost Operational Evidence

Public Axys report guidance provides a useful, bounded field-level view of
deliver-in cost review. It is operational/report evidence, not an official native
transaction schema.

| Observed report field | Practical meaning in the cited workflow |
|---|---|
| `Portfolio_Code_No_Path` | Portfolio identifier without a path. |
| `Tran_Code` | Transaction code. |
| `Security_Type_Code` | Security-type context. |
| `Security_Symbol_No_Type` | Security symbol without type. |
| `Security_Name_Or_Split_Description` | Security or split description. |
| `Trade_Date` | Trade date. |
| `Original_Cost_Date` | Original acquisition/cost date. |
| `Quantity` | Transaction quantity. |
| `Trade_Amount_Local` | Local-currency trade amount. |
| `Original_Cost` | Original cost. |
| `Source_Dest_Type_Code` | Source/destination type. |
| `Source_Dest_Symbol_No_Type` | Source/destination symbol without type. |
| `Lot` | Lot identifier/context. |

The same guidance identifies `li`, `ti`, and `si` as deliver-in cases in that
report. If original cost date and amount are absent, standard reports may use
market value on the trade date rather than visibly flagging missing cost. That
fallback can conceal a data-quality problem, so cost-date/cost-amount completeness
should be reviewed independently of apparently reasonable report values.

------------------------------------------------------------------------

## 3. APX

### 3.1 APX Transaction Role

| Statement | Confidence | Notes |
| --- | --- | --- |
| APX workflows include transaction import, blotter review/posting, reporting, reconciliation, and database/reporting alternatives. | Medium | Supported mainly by third-party integration and consultant evidence. |
| APX users may use SQL/database reporting/export alternatives in addition to IMEX. | Medium | Supported by consultant evidence. |
| Native APX database transaction schema is fully known from supplied material. | Unknown | Not supplied. |
| Native APX transaction-code matrix is fully known from supplied material. | Unknown | Not supplied. |

### 3.2 APX Blotter Types Observed

| Blotter | Observed Purpose | Confidence | Caveat |
| --- | --- | --- | --- |
| Trade Blotter | AIA imports transactions into this blotter; can be consolidated or created per custodian. | Medium | Integration workflow evidence. |
| Statement Blotter | Used to post custodian statement transactions; can support reconciliation against OMS or third-party data using REX. | Medium | Integration workflow evidence. |
| Tax Lot Blotter | Used for tax-lot-level reconciliation of APX-calculated lots versus custodian lots. | Medium | Integration workflow evidence. |
| Position Blotter | Used for importing positions into APX. | Medium | Integration workflow evidence. |
| Account Blotter | Used for importing account information. | Medium | Integration workflow evidence. |
| Initial Transaction Blotter | Used to import positions as deliver-in transactions for accounts without transactions, when configured. | Medium | AIA setting; native APX behavior Unknown. |

### 3.3 APX Trade Blotter Organization Options

| Option | Meaning | Confidence |
| --- | --- | --- |
| Consolidate Into One Blotter | Aggregate all transactions from all custodians into one trade blotter. | Medium |
| Create One Blotter Per Custodian | Distribute transactions into one blotter per custodian. | Medium |
| No Trade Blotter | No trade blotter is created by AIA. | Medium |

### 3.4 APX Transaction Translation Model

The supplied research describes a third-party APX integration model in
which source transactions are normalized before APX transaction
generation.

| Statement | Confidence | Notes |
| --- | --- | --- |
| WebPortfolio interprets financial-institution transaction data and assigns a normalized transaction type. | Medium | ByAllAccounts APX guide evidence. |
| Custodial Integrator translates normalized transaction types into APX transactions. | Medium | ByAllAccounts APX guide evidence. |
| Some APX translations depend on the sign of amount or units. | Medium | Examples include positive/negative transfer behavior. |
| Positive-unit transfer maps to APX `li` in the default translation table. | Medium | Integration default, not complete native documentation. |
| Negative-unit transfer maps to APX `lo` in the default translation table. | Medium | Integration default, not complete native documentation. |
| Translation tables may be customized by financial institution. | Medium | Integration behavior. |

### 3.5 APX Observed Transaction/Blotter Fields

| Field | Description | Axys | APX | IMEX | REP | Confidence |
| --- | --- | --- | --- | --- | --- | --- |
| APX Transaction Type | Transaction code/type used in APX translation table. | Unknown | Observed | Unknown | Unknown | Medium |
| APX Transaction Src/Dest Type | Source/destination security or cash type. | Unknown | Observed | Unknown | Unknown | Medium |
| APX Transaction Src/Dest Symbol | Source/destination symbol, e.g. cash-like symbols. | Unknown | Observed | Unknown | Unknown | Medium |
| APX Transaction Special Security Type | Special security type used in fee/expense examples. | Unknown | Observed | Unknown | Unknown | Medium |
| APX Transaction Special Security Symbol | Special security symbol used in fee/expense examples. | Unknown | Observed | Unknown | Unknown | Medium |
| Broker Representative Field | Field that can receive `$brok` in AIA workflow. | Unknown | Observed | Unknown | Unknown | Medium |
| Lot Location | Axys-era/APX workflow concept integrated into lot accounting. | Observed as Axys-era concept | Observed | Unknown | Unknown | Medium |
| Comment | Transaction import comment or standalone comment. | Unknown | Observed | Unknown | Unknown | Medium |

### 3.6 APX Initial Deliver-In Transactions

| Statement | Confidence | Notes |
| --- | --- | --- |
| AIA can create initial deliver-in transactions from positions for accounts with no transactions. | Medium | AIA/APX workflow evidence. |
| If transactions are received on the same day as initial positions in that scenario, the transactions may be ignored and not written to the blotter. | Medium | AIA workflow evidence. |
| Tax lots may be relevant to initial deliver-in construction. | Low to Medium | Details incomplete. |
| Native APX initial deliver-in behavior independent of AIA is fully known. | Unknown | Not supplied. |

### 3.7 APX Statement Transactions and Reconciliation

| Statement | Confidence | Notes |
| --- | --- | --- |
| APX workflows may distinguish posted portfolio transactions from statement transactions. | Medium | WealthTechs APX evidence. |
| Statement transactions may support reconciliation against custodian or OMS data. | Medium | WealthTechs APX evidence. |
| APX may expose separate UI tabs named `Transactions` and `Statement Transactions` in this workflow. | Medium | Workflow evidence. |

### 3.8 APX Comments and Broker Field

| Topic | Statement | Confidence |
| --- | --- | --- |
| Transaction comments | Rules in Transaction Translation may apply only to transaction comments in certain cases, while standalone comments always post to the import transaction file in the observed workflow. | Medium |
| Broker representative | A `Use $brok` setting can write `$brok` to the broker representative field in the transaction blotter. | Medium |
| `.cli` reference | `$brok` is described as typically defined in the `.cli` file for each APX portfolio. | Medium |
| Broker translations | Broker translations can map broker representative values to APX-specific codes. | Medium |

### 3.9 APX Cash Sweeps, Margin Sweeps, Short Sweeps, and Merge Logic

| Feature | Observed Behavior | Confidence | Caveat |
| --- | --- | --- | --- |
| Cash sweep removal | AIA includes logic to remove cash sweep transactions from source transaction files. | Medium | AIA behavior. |
| Margin and short sweep removal | AIA has similar removal logic for margin and short sweeps. | Medium | AIA behavior. |
| Example sweep patterns | Examples include `DP,CAUS,CASH,CAUS,MMF`, `DP,CAUS,CASH,CAUS,MARGIN`, and `DP,CAUS,CASH,CAUS,SHORT`. | Medium | Source examples only. |
| FX merge | AIA has options to merge FX transactions. | Medium | AIA behavior; native APX FX workflow Unknown. |
| Accrued-interest merge | AIA has options to merge accrued-interest transactions. | Medium | AIA behavior. |
| Dividend/interest merge | AIA has options to merge dividend and interest entries. | Medium | AIA behavior. |

------------------------------------------------------------------------

## 4. IMEX

### 4.1 IMEX Role

| Statement | Axys | APX | Confidence | Notes |
| --- | --- | --- | --- | --- |
| IMEX is an import/export mechanism used in Axys/APX workflows. | Supported | Supported | Medium | Consultant and third-party integration evidence. |
| IMEX supports CSV, tab, and fixed-format import/export in Axys according to consultant documentation. | Supported | Unknown | Medium | Axys-focused consultant evidence. |
| APX maintained IMEX functionality from v1.x to v4.x, but fixed-format file generation was eliminated according to consultant documentation. | Not applicable | Supported | Medium | Version-specific consultant evidence. |
| IMEX plus transaction/label import through Trade Blotter can move fundamental data in and out of Axys/APX. | Supported | Supported | Medium | Consultant evidence. |
| Official IMEX transaction object names are known. | Unknown | Unknown | Unknown | Not supplied. |
| Complete IMEX transaction field list is known. | Unknown | Unknown | Unknown | Not supplied. |

### 4.2 Axys IMEX Details Observed

| Detail | Observed Value | Confidence | Caveat |
| --- | --- | --- | --- |
| Utility | `imex32.exe` | Medium | Third-party integration evidence. |
| Import target | Trade Blotter / `topost.trn` in observed workflow. | Medium | Workflow-specific. |
| Logs | IMEX logs written to `$pathlog` in observed workflow. | Medium | Exact format Unknown. |
| Input support | CSV, tab, fixed-format according to consultant evidence. | Medium | Version coverage Unknown. |

### 4.3 APX IMEX Details Observed

| Detail | Observed Value | Confidence | Caveat |
| --- | --- | --- | --- |
| IMEX availability | APX maintained IMEX functionality in versions referenced by consultant source. | Medium | Exact version behavior Unknown. |
| Fixed-format generation | Eliminated in APX according to consultant documentation. | Medium | Needs official confirmation. |
| Alternative access | SQL/database reporting/export tools may be available. | Medium | APX-specific consultant evidence. |
| Official transaction import/export object names | Unknown | Unknown | Not supplied. |

### 4.4 Candidate IMEX Transaction Fields

The following fields are expected from accounting practice and supplied
research, but official IMEX names are not supplied. Therefore the IMEX
column names should be treated as **Unknown** until official
documentation or production exports are obtained.

| Field | Description | Axys | APX | IMEX | REP | Confidence |
| --- | --- | --- | --- | --- | --- | --- |
| Portfolio | Portfolio/account identifier. | Expected | Expected | Unknown name | Unknown | Unknown |
| Security | Security identifier or symbol. | Expected | Expected | Unknown name | Unknown | Unknown |
| Trade Date | Economic or execution date. | Expected | Expected | Unknown name | Unknown | Unknown |
| Settlement Date | Cash settlement date. | Expected | Expected | Unknown name | Unknown | Unknown |
| Transaction Code | Accounting event code. | Expected | Expected | Unknown name | Unknown | Unknown |
| Quantity | Units affected. | Expected | Expected | Unknown name | Unknown | Unknown |
| Price | Execution price. | Expected | Expected | Unknown name | Unknown | Unknown |
| Amount | Cash or transaction amount. | Expected | Expected | Unknown name | Unknown | Unknown |
| Broker | Broker or representative. | Unknown | Observed in blotter workflow | Unknown name | Unknown | Medium for APX blotter; Unknown for IMEX |
| Currency | Transaction or settlement currency. | Expected | Expected | Unknown name | Unknown | Unknown |
| FX Rate | Currency conversion rate. | Expected for multi-currency | Expected for multi-currency | Unknown name | Unknown | Unknown |
| Comment | Free-form note. | Unknown | Observed in import workflow | Unknown name | Unknown | Medium for APX workflow; Unknown for IMEX |

### 4.5 Transaction Extract Discovery and Classification Context

The available IMEX evidence strengthens the implementation guidance for
transaction extracts without closing the official-schema unknowns.

| Topic | Chapter treatment | Confidence |
|---|---|---:|
| `topost.trn` | Axys Trade Blotter file in `$pathtrn`; generated CI transactions are appended and existing transactions are left unchanged. | Verified for CI |
| File creation | Axys Import/Export can create `topost.trn` if it does not exist in the configured user folder. | Verified for CI |
| Comment boundaries | CI may create beginning and ending comment transactions around generated imports. | Verified for CI |
| Candidate fields | Portfolio, transaction code/subtype, dates, symbol/type, quantity, price, amounts, commission, fees, accrued interest, withholding, source/destination context, cost fields, Perf/CW, Mark to Market, currency, FX, comments, and external IDs should be inspected in live Axys. | Discovery guidance |
| External-flow classification | Source/destination and special-security context can be necessary for `li`, `lo`, `dp`, and `wd`, but official IMEX availability of those fields remains Unknown. | Medium Confidence for integration evidence; official IMEX availability Unknown |
| REP fallback | If IMEX cannot expose enough context to classify external flows, use REP/report/custom extraction as a candidate source of classification evidence. | Design guidance |

Transaction code alone should not be used as a universal external-flow rule.

------------------------------------------------------------------------

## 5. REP and Reports

### 5.1 Report and REP Evidence

| Report / Interface | System | Description | Confidence | Notes |
| --- | --- | --- | --- | --- |
| Transaction Summary Report | APX / Advent reports | Displays account transactions maintained by Advent; sample evidence includes dates, quantity, symbol, security, unit price, and amount. | Medium | Report sample and public report-guide evidence. |
| REP transaction reports | Axys | Unknown | Unknown | Exact report names, parameters, and fields not supplied. |
| REP transaction reports | APX | Unknown beyond Transaction Summary Report evidence | Unknown to Medium | Transaction Summary Report exists, but exact REP implementation Unknown. |
| Replang reports | Axys/APX | Consultant source lists Replang as a report/export alternative. | Medium | Exact transaction report code Unknown. |
| Report Writer Pro / Excel export / ETL | Axys/APX | Consultant source lists these as alternatives. | Medium | Exact transaction fields Unknown. |
| APX SQL/database access | APX | Consultant source lists SQL/database access as an APX reporting/export alternative. | Medium | Native schema Unknown. |

### 5.2 Transaction Summary Report --- Observed Columns

The supplied research includes sample column groups for an APX/Advent
Transaction Summary Report.

| Section | Observed Columns | Confidence |
| --- | --- | --- |
| Dividends | Ex-Date, Pay-Date, Symbol, Security, Amount | Medium |
| Contributions | Trade Date, Settle Date, Quantity, Symbol, Security, Unit Price, Amount | Medium |
| Withdrawals | Trade Date, Settle Date, Quantity, Symbol, Security, Unit Price, Amount | Medium |

### 5.3 Unknown REP Details

| Question | Status |
| --- | --- |
| Which REP reports expose transactions in Axys? | Unknown |
| Which REP reports expose transactions in APX beyond the Transaction Summary Report evidence? | Unknown |
| What are the official APX Transaction Summary Report parameters? | Unknown |
| What are the official APX Transaction Summary Report field names? | Unknown |
| Do REP reports read stored posted records, recalculated values, staged blotter values, or a mixture? | Unknown |
| How do REP outputs reconcile to IMEX exports and native accounting records? | Unknown |

------------------------------------------------------------------------

## 6. Data Model

### 6.1 Conceptual Model

``` text
Portfolio Master
Security Master
Currencies / FX
Pricing
Corporate Actions
Configuration / Translation Tables
        ↓
Transactions
        ↓
Holdings
Cash
Tax Lots
Cost Basis
Income
Realized Gain/Loss
Performance Inputs
Reports / IMEX / REP / Reconciliation / Audit
```

### 6.2 Upstream Dependencies

| Dependency | Role | Failure Mode | Confidence |
| --- | --- | --- | --- |
| Portfolio Master | Maps transaction to account/portfolio. | Unknown portfolio, inactive account, duplicate mapping. | High |
| Security Master | Identifies asset and security type. | Unknown security, duplicate security, ambiguous identifier. | High conceptually; Medium for integration evidence |
| Currency / FX | Supports multi-currency transaction and base-currency reporting. | Missing FX rate, invalid currency, settlement mismatch. | Medium |
| Pricing | Supports buy/sell valuation, cost basis, and reconciliation. | Missing price, price inconsistent with market close. | High conceptually; Medium for native field behavior |
| Corporate Actions | May generate or alter transaction interpretation. | Missing split, return of capital, paydown, reorg event. | Medium |
| Translation Configuration | Maps custodian/source records to Axys/APX accounting form. | Wrong transaction type, wrong security, wrong portfolio. | Medium |
| `.cli` / client settings | May affect interpretation, e.g. `li`/`lo` behavior in Axys conversion evidence. | Misclassified deliver-in/out versus credit/debit. | Medium |

### 6.3 Downstream Dependencies

| Downstream Area | Transaction Impact | Confidence |
| --- | --- | --- |
| Holdings | Buys, sells, transfers, splits, reinvestments, and paydowns change units/exposure. | High |
| Cash | Deposits, withdrawals, buys, sells, fees, dividends, interest, and settlements affect cash. | High |
| Tax Lots | Purchases, sales, transfers, initial deliver-ins, and corporate actions may create/consume/modify lots. | Medium |
| Cost Basis | Purchases, sales, transfers, return of capital, reinvestments, and fees may affect basis. | High conceptually; native mechanics Unknown |
| Income | Dividends, interest, withholding, reinvestment legs, and some bond events affect income. | High conceptually |
| Realized Gain/Loss | Sales, covers, transfers, and lot selection can create realized gain/loss. | High conceptually; native mechanics Unknown |
| Performance | Transactions affect capital flows, holdings, income, prices, and historical restatements. | High conceptually |
| Reports / IMEX / REP | Posted records are exposed through reports and interfaces. | Medium |
| Audit / Reconciliation | Transactions are primary evidence for accounting differences. | High |

### 6.4 Transaction Processing Pipeline

| Stage | Purpose | Typical Failure | Confidence |
| --- | --- | --- | --- |
| Acquire Source-Data | Obtain transactions from custodian, broker, OMS, manual entry, provider, or conversion file. | Missing file, stale file, incomplete batch. | Medium |
| Normalize Records | Convert source records into common representation. | Bad dates, bad signs, malformed identifiers. | Medium |
| Portfolio Translation | Map external account to internal portfolio. | Unknown or duplicate mapping. | Medium |
| Security Translation | Map external security to internal security. | Unknown or ambiguous security. | Medium |
| Transaction Translation | Map external type to accounting code/type. | Unsupported transaction, wrong direction, missing paired leg. | Medium |
| Special Processing | Apply sweeps, FX merge, accrued-interest merge, fee translation, tax logic, cancellation handling. | Suppressed records, bad merge, wrong fee classification. | Medium |
| Validation | Check required fields and plausibility. | Missing quantity, price, FX, dates, or invalid settlement sequence. | Medium |
| Staging / Blotter | Hold records for review. | Exception, cancellation mismatch, pending record. | Medium |
| Posting | Commit transaction to accounting records. | Posting failure, partial batch, unresolved dependency. | Medium |
| Downstream Updates | Update holdings, cash, lots, basis, income, gain/loss. | Position/cash/lot inconsistency. | High conceptually |
| Reporting / Export | Expose records through reports, IMEX, REP, SQL, or other tools. | Interface/report mismatch. | Medium |

### 6.5 AIA APX Processing Order Observed

This table documents AIA/APX integration behavior, not confirmed native
APX processing order.

| Order | Step | Applies To | Confidence |
| --- | --- | --- | --- |
| 3 | Remove Pending Records | All files | Medium |
| 4 | Remove Intra-Account Journals | Transactions | Medium |
| 5 | Remove Cash Sweeps | Transactions | Medium |
| 6 | Withholding Tax Logic | Transactions | Medium |
| 7 | Merge FX Transactions | Transactions | Medium |
| 8 | Merge Accrued Interest Transactions | Transactions | Medium |
| 9 | Transaction Translations | Transactions | Medium |
| 12 | Broker Translations | Transactions | Medium |
| 15 | Adjust Cancel Transactions to Upper Case | Transactions | Medium |
| 16 | Apply Transaction Comment Logic | Transactions | Medium |
| 17 | Merge Dividends and Interest | Transactions | Medium |
| 19 | Post Translations Transaction Translations | Transactions | Medium |
| 23 | Add Interface Comments | All files | Medium |

------------------------------------------------------------------------

## 7. Common Fields

### 7.1 Core Transaction Field Dictionary

| Field | Description | Axys | APX | IMEX | REP | Confidence |
| --- | --- | --- | --- | --- | --- | --- |
| Portfolio ID | Portfolio/account identifier. | Expected | Expected | Unknown name | Unknown | High conceptually |
| Transaction Code | Accounting event code/type. | Observed in examples | Observed in examples | Unknown name | Unknown | Medium for codes; Unknown for official matrix |
| Security Identifier | Security involved in transaction. | Expected | Expected | Unknown name | Symbol/Security observed in report sample | High conceptually |
| Trade Date | Execution/economic date. | Expected | Expected | Unknown name | Observed in report sample | High conceptually |
| Settlement Date | Settlement/cash date. | Expected | Expected | Unknown name | Observed as Settle Date in report sample | High conceptually |
| Entry Date | Date entered into system. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Posting Date | Date posted to accounting records; useful for explaining historical changes. | Unknown | Unknown | Unknown | Unknown | High-priority Unknown for historical review. |
| Quantity | Units traded or affected. | Expected | Expected | Unknown name | Observed in report sample | High conceptually |
| Price | Unit price. | Expected | Expected | Unknown name | Observed as Unit Price in report sample | High conceptually |
| Gross Amount | Transaction value before adjustments. | Expected | Expected | Unknown name | Unknown | Medium conceptually |
| Net Amount | Final cash amount. | Expected | Expected | Unknown name | Amount observed in report sample | Medium conceptually |
| Commission | Trading commission. | Expected optional | Expected optional | Unknown name | Unknown | High conceptually |
| Fees | Trading or administrative fees. | Expected optional | Expected optional | Unknown name | Unknown | High conceptually |
| FX Rate | Currency conversion rate. | Expected when multi-currency | Expected when multi-currency | Unknown name | Unknown | Medium |
| Broker | Broker or representative. | Unknown | Broker representative field observed | Unknown name | Unknown | Medium for APX workflow |
| Batch ID | Import batch identifier. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Source ID | External transaction identifier. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Cash Impact | Signed cash movement normalized for audit and performance use. | Unknown | Unknown | Unknown | Unknown | Medium as recommended field |
| Position Impact | Signed quantity or principal movement normalized for audit use. | Unknown | Unknown | Unknown | Unknown | Medium as recommended field |
| Performance Cash-Flow Flag | Explicit source or derived flag for external-flow treatment. | Unknown | Unknown | Unknown | Unknown | Medium as recommended field |
| Classification | Derived transaction family, such as external flow, trade, income, fee, corporate action, correction, or unknown. | Unknown | Unknown | Unknown | Unknown | Medium as recommended field |
| Classification Confidence | Confidence assigned by mapping/audit logic. | Unknown | Unknown | Unknown | Unknown | Medium as recommended field |
| Raw Line | Preserved source record for audit traceability. | Unknown | Unknown | Unknown | Unknown | Medium as recommended field |
| Comment | Free-form note. | Unknown | Observed in import workflow | Unknown name | Unknown | Medium for APX workflow |

### 7.2 Public Example Transaction Row

The supplied research includes this public third-party example row:

``` text
acct123,010101,010101,by,csus,appl,100,caus,cash,10000
```

A Trade Blotter cancellation example uppercases the transaction code:

``` text
acct123,010101,010101,BY,csus,appl,100,caus,cash,10000
```

Tentative interpretation:

| Position | Observed Value | Tentative Meaning | Confidence |
| --- | --- | --- | --- |
| 1 | `acct123` | Account / portfolio code. | Medium |
| 2 | `010101` | Date field 1. | Unknown |
| 3 | `010101` | Date field 2. | Unknown |
| 4 | `by` / `BY` | Transaction code / cancellation code. | Medium |
| 5 | `csus` | Security type. | Low to Medium |
| 6 | `appl` | Security symbol. | Low to Medium |
| 7 | `100` | Quantity. | Low to Medium |
| 8 | `caus` | Source/destination type. | Low to Medium |
| 9 | `cash` | Source/destination symbol. | Low to Medium |
| 10 | `10000` | Cash amount / net amount / trade amount. | Unknown |

This row is not a complete Axys/APX import layout.

------------------------------------------------------------------------

## 8. Transaction Codes

### 8.1 Warning

The following matrix is an observed-code catalog from supplied research.
It is **not** an official Axys or APX transaction-code reference. Codes
may be native, integration-layer mappings, version-specific,
configuration-dependent, context-dependent, or incomplete.

The strongest public transaction-code evidence in the supplied research
comes from ByAllAccounts Custodial Integrator default translation tables
and Morningstar Axys conversion notes. Those sources are useful for audit
design, but they do not establish a complete native Axys/APX code manual.
The July 2026 transaction-code addenda strengthened several practical
descriptions, especially for fixed-income accrued-interest and principal
paydown codes, while preserving the same evidence boundary.

### 8.2 Observed Transaction Code Matrix

| Code | Observed Meaning | Axys | APX | Confidence | Notes |
| --- | --- | --- | --- | --- | --- |
| `by` | Buy | Observed in examples; official status Unknown | Observed | Medium | Public integration documentation only. |
| `BY` | Cancellation Trade Blotter instruction derived from `by` in the reviewed staging workflow | Observed | Observed | Medium | Posted-export availability and native universality Unknown. |
| `sl` | Sell | Unknown | Observed | Medium | Requires vendor confirmation. |
| `ss` | Short sale | Unknown | Observed | Medium | Requires vendor confirmation. |
| `cs` | Cover short | Unknown | Observed | Medium | Observed in APX integration evidence. |
| `li` | Deliver in / transfer in / credit / deposit / positive movement | Observed | Observed | Medium | External-flow candidate; context determines final treatment. |
| `lo` | Deliver out / transfer out / debit / withdrawal / negative movement | Observed | Observed | Medium | External-flow candidate; context determines final treatment. |
| `dv` | Dividend / income / reinvestment leg | Unknown | Observed | Medium | Often relevant to reinvestment. |
| `in` | Income / interest | Unknown | Observed | Medium | Requires context. |
| `rc` | Return of capital | Observed | Observed | Medium-High | Confirmed in ByAllAccounts Axys/APX translation evidence; native performance and cost-basis treatment still needs site proof. |
| `pd` | Principal paydown / bond return-of-capital case | Observed | Observed | Medium-High | Confirmed in ByAllAccounts bond-security return-of-capital mapping and Morningstar paydown conversion caveats. |
| `ai` | Negative interest or margin interest | Unknown | Observed | Medium | Narrower than generic accrued interest; context- dependent. |
| `sa` | Sell-side accrued interest | Unknown | Observed | Medium | Fixed-income trade adjunct; integration-level evidence. |
| `pa` | Purchase accrued interest / buy-side accrued interest | Unknown | Observed | Medium | Fixed-income trade adjunct; integration-level evidence. |
| `dp` | Debit / fee-related / tax / service charge / cash-security case | Unknown | Observed | Medium | Multiple meanings; usually not an external flow without additional evidence. |
| `wd` | Cash-security sell / withdrawal-like cash-security movement | Unknown | Observed | Medium | Context-dependent; do not infer client withdrawal from the code name. |
| `;` | Journal / comment / other / split in integration table | Unknown | Observed | Medium | Treat as observed integration behavior only. |

### 8.3 Observed APX Translation Patterns

| Source Transaction Concept | Observed APX Translation Pattern | Confidence | Notes |
| --- | --- | --- | --- |
| ATM positive | `li` | Medium | Inflow-like. |
| ATM negative | `lo` | Medium | Outflow-like. |
| Buy | `by` | Medium | Default table evidence. |
| Cash security buy | `dp` | Medium | Special cash-security case. |
| Cover short | `cs` | Medium | Default table evidence. |
| Check | `lo` | Medium | Withdrawal-like. |
| Closure positive | `sl` | Medium | Positive closure maps to sell in observed table. |
| Closure negative | `cs` | Medium | Negative closure maps to cover short in observed table. |
| Credit | `li` | Medium | Inflow-like. |
| Debit non-cash security | `lo` | Medium | Outflow-like. |
| Tax | `dp` with special type/symbol | Medium | Examples include `epus` and withholding-related symbols. |
| Deposit cash | `li` | Medium | Inflow-like. |
| Deposit non-cash security | `li` and `by` pair | Medium | Two-transaction case in source. |
| Direct debit | `lo` | Medium | Outflow-like. |
| Direct deposit | `li` | Medium | Inflow-like. |
| Dividend | `dv` | Medium | Income-related. |
| Reinvested dividend | `dv` and/or paired buy behavior | Medium | Exact native behavior Unknown. |
| Fee | `dp` with special security type/symbol such as `exus custfee` | Medium | Configurable. |
| Recordkeeping fee | `dp` with `epus expense` | Medium | Source-table example. |
| Income bond security positive/negative | `li` / `lo` | Medium | Direction depends on sign. |
| Income cash security | `in` | Medium | Income-like. |
| Income dividend-paying security | `dv` | Medium | Dividend-like. |
| Interest positive | `in` | Medium | Income-like. |
| Interest negative | `ai` | Medium | Margin-interest-like special case. |
| Investment expense | `dp` with `exus custfee` | Medium | Fee-like. |
| Journal | `;` | Medium | Comment/journal-like. |
| Margin interest | `ai` | Medium | Uses margin cash symbol in source. |
| Other | `;` | Medium | Generic/other. |
| Payment | `lo` | Medium | Outflow-like. |
| Point of sale positive/negative | `li` / `lo` | Medium | Direction depends on sign. |
| Reinvestment | `dv` and `by` pair | Medium | Source shows paired APX translation. |
| Repeat payment | `lo` | Medium | Outflow-like. |
| Return of capital | `rc`; bond security may map to `pd` | Medium-High | `rc` is confirmed in public translation mapping evidence; bond-security return of capital maps to `pd`. |
| Sell | `sl` | Medium | Normal sell. |
| Sell cash security | `wd` | Medium | Cash-security special case. |
| Buy accrued interest | `pa` | Medium | Fixed-income trade adjunct; not external flow. |
| Sell accrued interest | `sa` | Medium | Fixed-income trade adjunct; not external flow. |
| Short | `ss` | Medium-High | Short sale; exact signs and short-cash mechanics remain site-specific. |
| Service charge | `dp` with `exus custfee` | Medium | Fee-like. |
| Split | central split factor data | High | Prefer `split.inf` / split-factor evidence; `;` is only a local or conversion marker. |
| Transfer positive/negative | `li` / `lo` | Medium | Direction depends on sign. |
| Withdrawal | `lo` | Medium | Outflow-like. |

### 8.4 Practical Classification Matrix

The following matrix is a practical audit/performance aid derived from
observed research. It is not an official Axys/APX code dictionary.

| Code or marker | Practical group | External flow? | Performance / audit implication |
|---|---|---:|---|
| `li` | Transfer / external-flow candidate | Often, not always | Candidate contribution or security-in-kind transfer; verify cash/security fields and firm mapping. |
| `lo` | Transfer / external-flow candidate | Often, not always | Candidate withdrawal or outgoing security transfer; separate fees, corrections, and internal journals. |
| `by`, `sl`, `ss`, `cs` | Trading activity | No | Affects holdings, cash, realized gain/loss, and exposure; validate price, quantity, commission, settlement cash, and short-account context for `ss`/`cs`. |
| `dv`, `in` | Income | No | Affects income and cash unless reinvested or netted; validate expected dividend/coupon and pay-date treatment. |
| `pa`, `sa` | Fixed-income trade adjunct | No | Validate accrued interest, settlement date, coupon schedule, day count, and total trade settlement. |
| `ai` | Negative interest / margin interest / financing adjustment | No | Validate margin or negative-interest context, amount sign, margin cash/security markers, and financing-rate support. |
| `dp`, `wd` | Cash, fee, expense, external-flow, or cash-security movement | Context-dependent | Treat as context-dependent; verify whether the record is a fee, expense, cash-security buy/sell, tax, internal movement, or true external movement. |
| `rc`, `pd` | Corporate action / principal event | Usually no | Confirmed in public Axys/APX translation mapping evidence as cash-producing security events; validate return-of-capital, paydown, cost-basis, factor, amortization, and performance-report treatment. |
| `;` | Journal, other, placeholder, or locally materialized split marker | Usually no | Prefer central split-factor evidence such as `split.inf` for normal split processing; require firm mapping before treating `;` as split evidence. |
| Uppercase code, e.g. `BY` | Unknown unless source-stage evidence identifies a Trade Blotter cancellation instruction | Unknown | Preserve exact case. Quarantine a proven staging/control instruction and link it to the original; do not infer cancellation from an ordinary posted extract. |
| `epus`, `exus` | Fee/expense security type or marker | No | Preserve as security/type context, not as a standalone transaction-code conclusion. |
| `dvwash` | Reinvestment wash symbol | No | Link dividend and reinvestment buy legs; avoid treating wash cash as external cash flow. |
| `caus margin` | Margin cash symbol/context | No | Supports margin-interest classification; separate financing expense from external flow. |

### 8.5 Minimal Firm-Specific Mapping to Collect

Because public research does not provide complete official native code
matrices, each implementation should collect a client-specific mapping
for ambiguous transaction families.

| Raw code | Security type | Security symbol | Source/destination type | Source/destination symbol | Firm meaning | External flow? | Notes |
|---|---|---|---|---|---|---:|---|
| `li` | Cash or non-cash | Client-specific | `$pty` or equivalent | `$cash` or equivalent | Contribution, security transfer in, correction, or other | Confirm | Identify cash versus in-kind transfer. |
| `lo` | Cash or non-cash | Client-specific | `$pty` or equivalent | `$cash` or equivalent | Withdrawal, security transfer out, correction, or other | Confirm | Separate client withdrawal from fees and journals. |
| `dp` | `epus`, `exus`, cash, or other | Fee/tax/cash symbol | Client-specific | Client-specific | Fee, expense, tax, cash-security buy, or other | Usually no | Confirm gross/net performance treatment. |
| `wd` | Cash or cash-like | Cash-security symbol | Client-specific | Client-specific | Cash-security sell, sweep, withdrawal-like event, or other | Confirm | Do not infer client withdrawal from name alone; require context before treating it as an external flow. |
| `;` | Any | Any | Any | Any | Journal, other, placeholder, or locally materialized split marker | Usually no | Route high-value items to manual review; prefer split-factor extracts for normal split evidence. |
| `dv` / `by` | Real security | Real symbol or `dvwash` | `$ity` or equivalent | `$income` or wash symbol | Dividend reinvestment | No | Link income and buy legs to avoid double counting. |
| `in` / `ai` | Bond, cash, or margin | Real symbol or margin symbol | Client-specific | Client-specific | Interest, negative interest, or margin interest | No | Validate coupon, accrual, or margin-rate support. |
| `pa` / `sa` | Bond or fixed-income security | Real symbol | Client-specific | Client-specific | Buy-side or sell-side accrued interest | No | Validate accrued interest, day count, and settlement economics; do not classify as external flow. |
| `rc` / `pd` | Real security or bond/MBS | Real symbol | Usually `$pty` in ByAllAccounts mapping evidence | Usually `$cash` in ByAllAccounts mapping evidence | Return of capital or principal paydown | No | Confirm context before classification. The packaged demo uses narrow context-gated examples only. |
| `ss` / `cs` | Shortable security or short-side account context | Real symbol | `ss` mapping evidence uses `awus`; `cs` mapping evidence uses `$pty` | `ss` mapping evidence uses no symbol; `cs` mapping evidence uses `$cash` | Short sale or cover short | No | Require exact lowercase code, short-position context, quantity, amount, and site cash/proceeds treatment. Uppercase `SS`/`CS` are separate staging/control evidence only when source stage proves that role. |

### 8.6 Observed Public Mapping Examples

The supplied research now includes a public WebPortfolio-to-Axys mapping
contract. The examples below are representative mappings from that
contract and are not universal Axys transaction-code rules.

| Normalized source event | Sign/context | Observed Axys code(s) | Classification caution |
|---|---|---|---|
| Deposit / direct deposit / credit | Cash in | `li` | External inflow candidate; confirm outside-party source and cash fields. |
| Withdrawal / check / payment | Cash out | `lo` | External outflow candidate; separate client withdrawal from fees and journals. |
| Transfer | Positive / negative | `li` / `lo` | May be cash transfer, in-kind transfer, or internal movement. |
| Buy / sell | Security | `by` / `sl` | Trade activity, not external flow. |
| Short / cover short | Security | `ss` / `cs` | Trade activity; validate short exposure, source/destination fields, cash/proceeds treatment, and quantity/amount signs. |
| Dividend / reinvestment | Income plus buy | `dv`, `by`, `dvwash` | Link paired legs and avoid double-counting income or wash cash. |
| Interest / negative interest | Positive / negative | `in` / `ai` | Separate income from margin or financing expense. |
| Accrued interest on buy/sell | Fixed-income trade adjunct | `pa` / `sa` | Part of bond trade settlement economics; not client external cash flow. |
| Fee / service charge / expense | Fee context | `dp` with `epus`/`exus` symbols | Fee or expense, not client withdrawal unless firm policy proves otherwise. |
| Cash-security buy/sell | Cash security | `dp` / `wd` | Cash-security handling; do not infer external cash flow from the code name. |
| Return of capital / paydown | Normal / bond | `rc` / `pd` | Issuer/principal event producing portfolio cash in public translation mapping evidence; not client capital flow. |
| Split / journal / other | Special marker | `;` | Requires firm mapping or manual review. |
| Trade Blotter cancellation control | Original code uppercased in the reviewed staging workflow | e.g. `BY` | Quarantine from posted economics, require source-stage evidence and original linkage, and flag unmatched instructions. |

Ambiguous `li`, `lo`, `dp`, `wd`, `;`, `epus`, and `exus` cases should
fall into a "requires review" bucket until firm mapping or supporting
source evidence confirms the audit classification.

### 8.7 Fixed-Income Accrued-Interest and Principal Codes

The July 2026 addendum strengthened but did not close the evidence for
`pa`, `sa`, `ai`, and `pd`. ByAllAccounts Axys/APX default translation
tables support the following practical interpretations:

| Code | Better-supported meaning | Practical treatment | External flow? |
|---|---|---|---:|
| `pa` | Purchase accrued interest / buy-side accrued interest. | Fixed-income trade adjunct; validate accrued days, coupon, day count, settlement date, and total settlement. | No |
| `sa` | Sale accrued interest / sell-side accrued interest. | Fixed-income trade adjunct; validate accrued days, coupon, day count, settlement date, and sale proceeds. | No |
| `ai` | Negative interest or margin interest. | Financing, negative-interest, or income-adjustment context; require margin/security context and amount sign. | No |
| `pd` | Principal paydown / bond-security return-of-capital transaction. | Principal event; validate cash received, principal/factor reduction, cost-basis/amortization treatment, and zero-quantity cases. | No |

These are observed integration mappings, not official native code
definitions. `pa` and `sa` should not be used as generic income codes
without bond/accrual context. `ai` should not be described as generic
accrued interest; the strongest public evidence supports negative
interest and margin interest. `pd` should remain a principal-paydown or
bond-return-of-capital candidate until security type, factor/principal
movement, and local methodology are known.

### 8.8 `pa` / `sa` Accrued-Interest Promotion Boundary

The 2026-07-03 accrued-interest research strengthens the practical
interpretation of `pa` and `sa`:

| Code | Strongest supported meaning | Product boundary |
|---|---|---|
| `pa` | BUY accrued interest / purchase accrued interest. | Fixed-income trade-settlement adjunct; not an investor external flow. |
| `sa` | SELL accrued interest / sale accrued interest. | Fixed-income trade-settlement adjunct; not an investor external flow. |

The evidence is still integration-level rather than a complete official native
Axys/APX code manual. It supports a narrow packaged demonstration when paired
trade, cash, holdings/accrued, and performance rows all move together, but it
does not support classifying `pa` or `sa` from code alone. The same evidence
path supports configurable transaction translation, so site-specific YAML or a
future vendor preset must still require fixed-income context and auditable
source-data support.

Safe `pa`/`sa` treatment should verify:

- fixed-income security type or equivalent bond context;
- paired principal transaction, such as `by`/`pa` or `sl`/`sa`;
- trade date and settlement date alignment;
- amount sign and settlement economics;
- accrued-interest, income, cash, or source/destination markers when available;
- holdings/accrued-interest movement; and
- security-performance and portfolio-performance treatment.

The packaged Axys/APX demo promotes `pa`/`sa` only in that coherent paired
fixed-income scenario: accrued-interest transaction rows, cash movement,
holdings/accrued-interest rows, `secperf.csv`, `portperf.csv`, and
reviewer-facing report comments are derived together. That is a demo-quality
gate, not a claim that `pa`/`sa` are universal code-only Axys defaults.

### 8.9 Cancellation and Reversal

| Statement | Axys | APX | Confidence | Notes |
| --- | --- | --- | --- | --- |
| A tool may uppercase a historical transaction code to create a cancellation Trade Blotter instruction, e.g. `by` → `BY`. | Observed | Observed | Medium | Third-party staging/control workflows. |
| Cancellation transaction fields must sufficiently match the original transaction or blotter error may occur. | Unknown | Observed | Medium | APX integration evidence. |
| Cancellation blotters may be created from historical transaction files. | Observed | Observed | Medium | WealthTechs evidence. |
| Cancellation workflows should be treated as high-risk and backed up/reviewed. | Supported recommendation | Supported recommendation | Medium | Based on source warnings. |
| An uppercase instruction survives as a posted transaction or appears in ordinary REP, IMEX/APXIX, SQL, REST, or report extracts. | Unknown | Unknown | Unknown | Not established by reviewed evidence. |
| Uppercase cancellation is universal native behavior across all versions and import methods. | Unknown | Unknown | Unknown | Not supplied. |

### 8.10 Context-Gated Backlog Transaction Boundaries

The 2026-07-07 backlog transaction research strengthens the practical
interpretation of `rc`, `pd`, `ss`, `cs`, and split/journal marker `;`. A later
2026-07-07 `rc`/`pd` research pass further confirms the public integration
mapping evidence for return of capital and bond principal paydown. This does
not convert the codes into safe code-only packaged-demo defaults, because the
strongest evidence is still integration and conversion documentation rather
than an official complete Axys/APX transaction-code manual or native
performance-report methodology.

| Code or marker | Strongest supported meaning | Product boundary |
|---|---|---|
| `rc` | Return of capital. ByAllAccounts Axys/APX default translation evidence maps return-of-capital activity to `rc`, with `$pty` and `$cash` source/destination context. | Policy-gated. Public evidence confirms the translation mapping and cash destination, but not native tax-lot, cost-basis, or performance-report treatment. Do not classify from code alone. |
| `pd` | Principal paydown / bond-security return-of-capital event. ByAllAccounts translation evidence maps bond-security return-of-capital activity to `pd`, again with portfolio-cash destination context; Morningstar Axys conversion evidence discusses principal paydown rows and zero share quantity. | Fixed-income principal event, not ordinary interest income or client external flow. The packaged demo includes only a narrow context-gated MBS/amortizing-security example with cash receipt, principal/holding evidence, and performance-report treatment. |
| `ss` | Short sale / sell short. ByAllAccounts APX mapping evidence maps `SELL / SHORT` to lowercase `ss`. | Requires exact lowercase code, short security type or resulting negative exposure, cash/margin/short-account context, source/destination evidence, and verified amount/quantity sign conventions. Uppercase `SS` requires separate source-stage handling. |
| `cs` | Cover short / buy-cover-short. ByAllAccounts APX mapping evidence maps `BUY / COVER SHORT` to lowercase `cs`. | Requires exact lowercase code, prior or resulting short exposure plus cash/margin/short-account context. Uppercase `CS` requires separate source-stage handling. |
| `;` | Journal, Other, or locally materialized split marker. Public integration evidence maps split/journal/other concepts to `;`, but newer split-file evidence indicates normal Axys split support is central `split.inf` factor data rather than ordinary account-level transactions. | Marker/comment/corporate-action evidence unless local mapping proves a specific economic role. Prefer split-factor extracts for packaged or audited split scenarios. |

#### 8.10.1 `pd` Principal Paydown Boundary

`pd` is now represented in the packaged Audit demo, but only as
a coherent amortizing-security principal-paydown scenario. Safe `pd` treatment should
verify:

- fixed-income security type or equivalent MBS/ABS/amortizing-security context;
- principal-paydown or return-of-principal context;
- cash movement, typically `$pty` / `$cash` portfolio-cash context in the public
  translation evidence;
- holding principal, factor, quantity, or market-value movement consistent with
  the event;
- amount and sign convention, trade/effective date, and local translation
  support;
- reported security-performance and portfolio-performance treatment; and
- reviewer wording that distinguishes principal return from coupon interest,
  accrued interest, amortization/accretion, and client external flow.

Morningstar conversion evidence warns that Advent Axys principal paydown
activity may involve zero share quantity and original-principal adjustment
mechanics. Therefore, quantity alone is not enough to audit a paydown; principal
or factor context is needed where available. Cost-basis, amortization, and tax
lot mechanics remain best-efforts demo-construction context, not requirements
for Modified Dietz calculation.

For Modified Dietz performance-comparison work, `pd` should be treated as a
security-level principal return unless site evidence proves otherwise. It
increases cash and reduces outstanding principal, but it should not be
classified as an investor contribution or withdrawal. At portfolio level, it is
best modeled as an internal movement from security principal exposure to cash
when the MBS/ABS/amortizing-security paydown context is coherent. At security
level, it is a principal return by the security. Public sources do not
establish a universal native Axys/APX field layout for principal amount,
factor, or amortized-cost movement.

Rows lacking fixed-income context, cash destination, or local mapping support
should remain review-only. Reversal/delete rows and custom site-translation rows
should be classified separately before any performance treatment is inferred.

#### 8.10.2 `rc` Return-of-Capital Boundary

`rc` is supported as return of capital in public integration mapping evidence,
and the strengthened research specifically confirms a portfolio-cash
destination in the ByAllAccounts mapping. Its performance role is still
policy-sensitive. A site may reasonably configure return of capital as
performance income, security-level capital-return evidence,
corporate-action/cost-basis evidence, or review-only evidence depending on the
report methodology. Public evidence does not verify native Axys/APX cost-basis
algorithm or exact performance-report treatment.

For this reason, `rc` should remain a site-confirmed rule rather than a packaged
default. Any future packaged example should explicitly state the configured
methodology and should not imply universal Axys/APX treatment.

For Modified Dietz, `rc` should not be modeled as an external client flow when
it represents issuer return of capital. It is a security-level distribution or
capital-return event that may affect cash, return, and cost-basis context, while
tax-basis handling remains best-efforts outside the performance calculation.

#### 8.10.3 `ss` / `cs` Short-Side Boundary

Public integration evidence supports `ss` as short sale and `cs` as cover short,
but a safe performance interpretation needs more than the code:

- security type or position context proving the holding is short;
- quantity sign convention for trade rows and holdings rows;
- amount and cash/proceeds sign convention;
- cash, restricted cash, margin, or short-proceeds account mapping;
- realized gain/loss or lot-closure treatment when a short is covered; and
- reported performance treatment for the short exposure.

The 2026-07-07 short-lifecycle research supports a controlled synthetic demo
when the assumptions are disclosed: negative short quantity, negative short
market value, separate short-proceeds or margin/short-cash handling, no client
external-flow treatment, and realized gain/loss on cover. That is a defensible
demo model, not a universal production rule.

The packaged Axys/APX performance-comparison demo uses that boundary for one
same-period TSLA `ss` / `cs` lifecycle with real May 2026 prices and explicit
source/destination context. It is included to show reviewer behavior, not to
claim universal native Axys/APX short-account mechanics.

For production rows, `ss` and `cs` remain better suited to tested site-variant
YAML and onboarding override examples unless the extract proves the local
short-position and cash/proceeds mechanics.

#### 8.10.4 Concrete Example Gap

The 2026-07-07 research found useful translation-table rows and row-like
comment/import examples. It did **not** find public sanitized examples of:

- account-level Axys/APX transaction report rows for `rc`, `pd`, `ss`, or `cs`;
- IMEX `.cli` rows for those target types;
- REP/Replang report-output rows for those target types;
- APX Public View or SQL result rows for those target types; or
- before/after holdings plus before/after performance rows for the same real
  account event.

Those examples remain the highest-value evidence needed before broad packaged
demo promotion.

The short-lifecycle research reduces the demo-design gap for `ss` and `cs`, but
it does not remove the need for native customer or vendor rows before promoting
code-only production classification.

------------------------------------------------------------------------

## 9. Examples

### 9.1 Buy Example from Public Integration Evidence

``` text
acct123,010101,010101,by,csus,appl,100,caus,cash,10000
```

| Interpretation Item | Value | Confidence |
| --- | --- | --- |
| Account / portfolio | `acct123` | Medium |
| Transaction code | `by` | Medium |
| Security type | `csus` | Low to Medium |
| Security symbol | `appl` | Low to Medium |
| Quantity | `100` | Low to Medium |
| Source/destination type | `caus` | Low to Medium |
| Source/destination symbol | `cash` | Low to Medium |
| Amount | `10000` | Unknown |

### 9.2 Cancellation Example

``` text
acct123,010101,010101,BY,csus,appl,100,caus,cash,10000
```

| Interpretation Item | Value | Confidence |
| --- | --- | --- |
| Staging/control instruction | `BY`, uppercase version of `by` in the cited Trade Blotter workflow | Medium |
| Native universality | Unknown | Unknown |
| Required match fields | Unknown | Unknown for Axys; Medium for APX integration evidence that mismatch can produce blotter error |
| Availability in posted exports | Unknown | Not established for REP, IMEX/APXIX, SQL, REST, or ordinary reports |

### 9.3 Reinvestment Pattern

| System | Observed Pattern | Confidence | Caveat |
| --- | --- | --- | --- |
| Axys | Reinvestment may appear as Buy plus Distribution transaction pairs in conversion data. | Medium | Conversion evidence only. |
| APX | Reinvestment may translate as `dv` and `by` pair in ByAllAccounts integration evidence, often with `dvwash`. | Medium | Integration evidence only. |

### 9.4 Fee Pattern

| System | Observed Pattern | Confidence | Caveat |
| --- | --- | --- | --- |
| Axys | `epus` associated with Management Fee conversion; `exus` associated with Expense conversion. | Medium | Official meaning Unknown. |
| APX | Fee transactions may use `dp` plus special security type/symbol such as `exus custfee` or `epus expense`. | Medium | Integration evidence only. |

### 9.5 Withholding-Tax and Negative-Income Patterns

Observed integration mappings allow several possible treatments for
withholding taxes or income reductions. These are classification
patterns, not verified native Axys/APX field rules.

| Pattern | Meaning | Audit implication |
|---|---|---|
| Separate expense line such as `dp` with `exus` or withholding symbol | Gross dividend plus separate tax/fee | Link income and tax lines for gross/net review. |
| Negative `dv` or negative `in` | Tax or adjustment reduces income directly | Easier net cash view, but gross income can be obscured. |
| Withholding-tax field on income transaction | Single income record carries tax detail | Prefer this when preserved because it keeps gross/net detail together. |

------------------------------------------------------------------------

## 10. Known Issues / Quirks

| Issue / Quirk | Axys | APX | Confidence | Notes |
| --- | --- | --- | --- | --- |
| Code-only interpretation is unsafe. | Supported | Supported | High as design rule; Medium source evidence | Use code, sign, security type, source/destination fields, symbols, and configuration. |
| Direct file access is risky because file formats can change between versions. | Supported | Unknown; less directly applicable | Medium | Consultant evidence cites Axys file-format changes between versions. |
| APX SQL/database access may exist as an alternative export path. | Not applicable | Supported | Medium | Native schema Unknown. |
| `li`/`lo` interpretation may depend on `.cli` setting. | Supported | Unknown | Medium | Morningstar Axys conversion evidence. |
| Reinvestments may appear as paired transactions. | Supported | Supported | Medium | Axys conversion and APX integration evidence. |
| Fees may depend on special security type/symbol and description translation. | Supported | Supported | Medium | Terminology differs across sources. |
| Principal paydowns may produce downstream conversion/reconciliation complications. | Supported | Unknown | Medium | Axys conversion evidence. |
| Uppercase cancellation instructions are observed in Trade Blotter staging/control workflows. | Supported | Supported | Medium | Posted-export availability and native universality Unknown. |
| AIA/APX import may remove pending records, sweeps, intra-account journals, or merge FX/accrued-interest/dividend-interest records. | Not applicable | Supported in integration workflow | Medium | AIA behavior, not confirmed native APX order. |
| Initial deliver-ins may be generated from positions for accounts with no transactions in AIA/APX workflow. | Unknown | Supported in integration workflow | Medium | Native behavior Unknown. |
| Statement transactions and posted transactions may be distinguished in APX workflows. | Unknown | Supported | Medium | Workflow evidence. |
| `;` may represent journal/comment/other/split in APX integration table. | Unknown | Observed | Medium | Treat only as integration evidence. |

------------------------------------------------------------------------

## 11. Audit Rules

These rules are candidate transaction audit controls. They are not
confirmed native Axys/APX validation behavior unless explicitly noted.

### 11.1 Validation Rules

| Rule | Severity | Description | Required Inputs | Confidence |
| --- | --- | --- | --- | --- |
| TR-001 Missing Portfolio | Critical | Transaction references a portfolio that does not exist. | Portfolio ID | High |
| TR-002 Missing Security | Critical | Security transaction references an unknown security. | Security Identifier, Transaction Code | High |
| TR-003 Missing Trade Date | High | Trade-based transaction lacks trade date. | Trade Date | High |
| TR-004 Settlement Before Trade | High | Settlement date precedes trade date. | Trade Date, Settlement Date | High |
| TR-005 Missing Quantity | High | Security transaction lacks required quantity. | Quantity, Transaction Code | High |
| TR-006 Missing Price | Medium | Price-required transaction has no execution price. | Price, Transaction Code | High |
| TR-007 Invalid FX Rate | Medium | Foreign-currency transaction has missing or invalid FX rate. | Currency, FX Rate | Medium |

### 11.2 Translation and Blotter Rules

| Rule | Severity | Description | Confidence |
| --- | --- | --- | --- |
| TR-008 Portfolio Translation Failure | Critical | External portfolio/account cannot be translated. | Medium |
| TR-009 Security Translation Failure | Critical | External security cannot be translated. | Medium |
| TR-010 Unsupported Transaction Type | High | External transaction type has no mapping. | Medium |
| TR-011 Trade Blotter Exception | Medium | Transaction remains in exception state. | Medium |
| TR-012 Cancellation Mismatch | High | A proven staging/control cancellation instruction does not sufficiently match its original transaction. | Medium |
| TR-013 Cancellation Control | High | Cancellation blotters require source-stage identification, segregation from posted economics, review, backup, and operational controls. | Medium |

### 11.3 Accounting Rules

| Rule | Severity | Description | Confidence |
| --- | --- | --- | --- |
| TR-014 Holdings Not Updated | Critical | Posted transaction not reflected in holdings. | High |
| TR-015 Cash Not Updated | Critical | Posted transaction not reflected in cash. | High |
| TR-016 Cost Basis Inconsistency | High | Cost basis inconsistent with transaction history. | Medium |
| TR-017 Tax Lot Inconsistency | High | Tax lots inconsistent with transaction history. | Medium |
| TR-018 Dividend Without Position | Medium | Dividend received without supporting position. | Medium |
| TR-019 Coupon Inconsistent With Bond | Medium | Coupon payment inconsistent with bond characteristics. | Medium |
| TR-020 Return of Capital Without Eligible Security | Medium | Return of capital appears for security not expected to support it. | Medium |
| TR-021 Split Without Quantity Adjustment | High | Split detected without expected holding adjustment. | Medium |
| TR-022 Split Without Price Adjustment | Medium | Historical prices inconsistent with split. | Medium |
| TR-023 Principal Paydown Inconsistency | Medium | Principal paydown inconsistent with expected reduction. | Medium |

### 11.4 Reconciliation and Historical Change Rules

| Rule | Severity | Description | Confidence |
| --- | --- | --- | --- |
| TR-024 Custodian Difference | High | Custodian transactions differ from accounting records. | High |
| TR-025 IMEX Difference | Medium | IMEX export differs from expected accounting records. | Medium |
| TR-026 REP Difference | Medium | REP report differs from accounting records. | Unknown |
| TR-027 Historical Transaction Modified | High | Historical transaction edited. | High |
| TR-028 Historical Transaction Deleted | High | Historical transaction deleted. | High |
| TR-029 Performance Restatement Candidate | High | Historical transaction change may require performance review. | High |
| TR-030 Duplicate Transaction | High | Potential duplicate transaction. | High |
| TR-031 Duplicate External Identifier | Medium | Duplicate external transaction identifier. | Medium |
| TR-032 Stale Pending Transaction | Medium | Pending transaction exceeds operational threshold. | Medium |
| TR-033 Batch Partially Processed | Medium | Import batch incomplete. | Medium |
| TR-034 Stale Account / Stale Price Detection | Medium | Import should identify stale accounts and stale prices before export/posting. | Medium |
| TR-035 External Flow Misclassified | High | Candidate external-flow transaction is classified using code alone without security, source/destination, sign, and firm mapping context. | High |
| TR-036 Fee / Expense as Flow | High | Fee, expense, tax, cash-security movement, or sweep is treated as a client contribution or withdrawal. | Medium |
| TR-037 Orphan Cancellation Control | High | A source-stage-verified cancellation instruction cannot be linked to its original transaction. | Medium |

------------------------------------------------------------------------

## 12. Version Differences

| Topic | Axys | APX | Confidence | Notes |
| --- | --- | --- | --- | --- |
| Axys v2.x binary files and IMEX | Consultant evidence says Axys v2.x introduced binary file formats and IMEX allowed CSV, tab, and fixed formats. | Not applicable | Medium | Needs official confirmation. |
| Axys v3.7 to v3.8 file conversion | Consultant evidence says upgrading from Axys v3.7 to v3.8 required file conversion and produced some files with different formats. | Not applicable | Medium | Supports caution against direct file access. |
| APX v1.x to v4.x IMEX | Not applicable | Consultant evidence says APX maintained IMEX functionality but eliminated fixed-format file generation. | Medium | Needs official confirmation. |
| Native transaction code changes by version | Unknown | Unknown | Unknown | Not supplied. |
| Native Trade Blotter behavior changes by version | Unknown | Unknown | Unknown | Not supplied. |
| Native REP report changes by version | Unknown | Unknown | Unknown | Not supplied. |
| APX v1-v3 versus v4 Mark-to-Market input | Not applicable | ByAllAccounts release notes say Mark-to-Market was required only for foreign currency in APX v1-v3, while APX v4 requires an explicit `y`/`n`. | High Confidence for that integration | Do not generalize the integration field into a native accounting formula. |

------------------------------------------------------------------------

## 13. References

The supplied research identifies the following source categories and
specific references. Confidence varies by source type.

| ID | Source | Type | System | Topics | Confidence |
| --- | --- | --- | --- | --- | --- |
| SRC-001 | SS&C Advent Axys Product Page | Vendor product page | Axys | Portfolio accounting, reporting, performance, reconciliation, transactions, positions, settlement/trade information, tax-lot or average-cost accounting, trade-date or settlement-date accounting. | High for capabilities; Low for mechanics |
| SRC-002 | AdventGuru --- Getting Data In and Out of Advent APX and Axys | Consultant article | Axys/APX | IMEX, Trade Blotter, import/export, Replang, reports, direct-file-access risks, APX SQL/database options. | Medium |
| SRC-003 | WealthTechs AIA User Manual --- APX Users | Third-party integration manual | APX | Trade/Statement/Tax Lot/Position/Account blotters, transaction translation, cancellation, comments, broker fields, processing order. | Medium |
| SRC-004 | WealthTechs AIA User Manual --- Axys Users | Third-party integration manual | Axys | Transaction cancellation, IMEX workflow, blotters, cancellation example. | Medium |
| SRC-005 | ByAllAccounts Custodial Integrator --- APX User Guide | Third-party integration manual | APX | Translation tables, reversals, fees, imports, sign-dependent translation, source/destination fields, special security fields. | Medium |
| SRC-006 | ByAllAccounts Custodial Integrator --- Axys User Guide | Third-party integration manual | Axys | Trade Blotter workflow, IMEX import, `topost.trn`, `imex32.exe`, folder labels, IMEX logs, security/reference files. | Medium |
| SRC-007 | Morningstar Office Advent Axys Conversion Guide | Third-party migration guide | Axys | Reinvestment, deliver-in/out interpretation, `.cli`, cost basis, fees, paydowns, transaction prices, historical prices, conversion caveats. | Medium |
| SRC-008 | Advent Portfolio Exchange Reports Guide | Vendor report guide / public PDF reference | APX | Transaction Summary Report existence. | Low to Medium |
| SRC-009 | Wealth Management Reports / Advent report sample | Vendor/report sample | APX / SSRS | Transaction Summary Report purpose and sample columns. | Medium |
| SRC-010 | AdventGuru --- APX to Axys Conversion | Consultant article | Axys/APX | APX-exported CLI files mapped into Axys `topost.trn`; transaction mappings and tax lots. | Medium |
| SRC-011 | [CSSI missing-cost guidance](https://cssisolutions.com/downloads/how-to-identify-missing-cost-information) | Operational report guidance | Axys | Deliver-in codes, observed report fields, and market-value fallback when original cost is absent. | High for cited workflow |
| SRC-012 | [WealthTechs APX guide](https://wealthtechs.com/AIADocumentation/APX%20Guide%20-%20AIA%20User%20Manual%20For%20APX%20Users.pdf) | Third-party integration manual | APX | Ordered transformation pipeline, cancellation Trade Blotter creation, identifier/rule-evaluator case distinction, and current/historical holdings. | High for cited workflow |
| SRC-013 | [ByAllAccounts APX release notes](https://www.byallaccounts.net/Manuals/Custodial_Integrator/apx/CI_releasenotes.pdf) | Third-party integration release notes | APX | Version-specific Mark-to-Market requiredness. | High for cited integration |

------------------------------------------------------------------------

## 14. Unknowns

### 14.1 Transaction Codes

| ID | Unknown | Priority |
| --- | --- | --- |
| TU-001 | Complete official Axys transaction-code matrix. | High |
| TU-002 | Complete official APX transaction-code matrix. | High |
| TU-003 | Whether Axys/APX transaction codes are identical, overlapping, divergent, version-specific, or configuration-dependent. | High |
| TU-004 | Which observed codes are native versus integration-layer mappings. | High |
| TU-005 | Deprecated or version-specific transaction codes. | Medium |

### 14.2 IMEX

| ID | Unknown | Priority |
| --- | --- | --- |
| TU-006 | Official Axys IMEX transaction export object names. | High |
| TU-007 | Official Axys IMEX transaction import object names. | High |
| TU-008 | Official APX IMEX transaction export/import object names. | High |
| TU-009 | Complete IMEX transaction field list. | High |
| TU-010 | Official Trade Blotter import layout. | High |
| TU-011 | IMEX log fields and validation messages. | Medium |
| TU-012 | Native IMEX object dependency sequence. | Medium |

### 14.3 REP and Reports

| ID | Unknown | Priority |
| --- | --- | --- |
| TU-013 | Which REP reports expose transaction information. | High |
| TU-014 | Official APX Transaction Summary Report parameters and fields. | High |
| TU-015 | Whether REP transaction values are stored, recalculated, or mixed. | Medium |
| TU-016 | Axys transaction report names and fields. | High |
| TU-017 | APX transaction reports beyond Transaction Summary Report. | Medium |
| TU-018 | How REP report values reconcile to IMEX exports and posted accounting records. | Medium |

### 14.4 Internal Data Model and Posting

| ID | Unknown | Priority |
| --- | --- | --- |
| TU-019 | How transactions are physically stored in Axys. | High |
| TU-020 | How transactions are stored in APX. | High |
| TU-021 | Internal identifiers that uniquely identify transactions. | High |
| TU-022 | Native posting status values. | Medium |
| TU-023 | Native Trade Blotter state transitions. | High |
| TU-024 | Native error states and rejection codes. | Medium |
| TU-025 | Native warning messages. | Medium |
| TU-026 | Batch rollback/restart/recovery logic. | Medium |
| TU-027 | Native idempotency or duplicate-detection logic. | Medium |

### 14.5 Historical Changes, Lots, Cost Basis, and Audit

| ID | Unknown | Priority |
| --- | --- | --- |
| TU-028 | How reversals are represented internally. | High |
| TU-029 | Whether uppercase transaction codes universally mean delete/reversal. | High |
| TU-030 | How historical edits are represented. | High |
| TU-031 | Whether deleted transactions are retained for audit. | High |
| TU-032 | How corrections are distinguished from reversals. | Medium |
| TU-033 | How transaction edits propagate into holdings. | Medium |
| TU-034 | How transaction edits propagate into cash. | Medium |
| TU-035 | How transaction edits propagate into performance. | High |
| TU-036 | Whether historical transactions can be reconstructed completely. | High |
| TU-037 | How tax lots are linked to transactions. | High |
| TU-038 | How partial lot disposals are represented. | Medium |
| TU-039 | How per-share cost basis is represented in `.cli` exports. | High |
| TU-040 | How transfer lots preserve acquisition date and basis. | Medium |
| TU-041 | How lot locations are stored and used natively. | Medium |

### 14.6 Multi-Currency and Performance

| ID | Unknown | Priority |
| --- | --- | --- |
| TU-042 | How FX rates are stored. | Medium |
| TU-043 | How cross-currency settlements are represented. | Medium |
| TU-044 | How FX transactions are merged or paired in native workflows. | Medium |
| TU-045 | How base-currency values are stored versus calculated. | Medium |
| TU-046 | Which transaction types affect stored performance. | High |
| TU-047 | Which transaction changes trigger performance restatement. | High |
| TU-048 | How performance restatements are detected or audited. | High |
| TU-049 | Whether edited/deleted historical transactions are visible to performance recalculation workflows. | High |

------------------------------------------------------------------------

## 15. Minimum Additional Material Needed to Promote Unknowns

To convert this chapter from observed/integration-level evidence into a
more authoritative native Axys/APX transaction reference, the following
supplied material would be needed:

| Needed Material | Would Resolve |
| --- | --- |
| Official Axys transaction-code manual or sanitized production code list. | Axys native code matrix. |
| Official APX transaction-code manual or sanitized production code list. | APX native code matrix. |
| Official Axys/APX IMEX manual with transaction objects. | IMEX object names and field layouts. |
| Sample Axys IMEX transaction export/import files. | Axys transaction field names and formats. |
| Sample APX IMEX transaction export/import files. | APX transaction field names and formats. |
| Official Trade Blotter layout documentation. | Native blotter fields, required fields, validation rules. |
| Sample REP transaction reports and report definitions. | REP fields, parameters, report behavior. |
| Sanitized APX database schema or query extracts. | Native APX transaction storage model. |
| Sanitized Axys file/export documentation. | Native Axys transaction storage and file behavior. |
| Audit/log examples for posted, canceled, corrected, and rejected transactions. | Native audit trail, state transitions, and historical reconstruction. |

## Research Provenance

The 2026-06-30 transaction-extract conclusions are incorporated into Section
4.5. The 2026-07-03 accrued-interest conclusions are incorporated into Section
8.8, and the 2026-07-07 context-gated backlog conclusions are incorporated
into Section 8.10. Their granular supporting claims and confidence boundaries
remain in `../evidence/Research_05_Transactions.md`.
