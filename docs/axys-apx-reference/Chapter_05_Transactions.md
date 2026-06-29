# 05 — Transactions

> **Repository chapter:** `docs/05-Transactions.md`  
> **Status:** Draft technical reference based only on supplied research material.  
> **Evidence standard:** Facts are marked as Verified, High Confidence, Medium Confidence, Low Confidence, or Unknown.  
> **Important limitation:** The supplied material does not include official complete Axys or APX transaction-code manuals, official IMEX transaction object definitions, complete native Trade Blotter layouts, native Axys/APX transaction storage schemas, or complete REP specifications. Unsupported information is marked **Unknown**.

---

## 1. Overview

Transactions are the central accounting events in Axys/APX-style portfolio accounting workflows. They connect economic activity to holdings, cash, tax lots, cost basis, realized gain/loss, income, performance, reports, IMEX, REP, reconciliation, and audit workflows.  
**Confidence:** High for general accounting role; Medium for native Axys/APX mechanics because public source material is largely third-party integration or migration evidence.

A practical transaction lifecycle supported by the supplied research is:

```text
Economic event
    ↓
External source data
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
|---|---|---:|---|
| Transactions affect holdings, cash, income, cost basis, realized gain/loss, and performance inputs | Supported conceptually and by supplied research | High | General accounting role is strong; native mechanics vary. |
| Trade Blotter workflows exist in Axys/APX integration evidence | Supported | Medium | Evidence is third-party integration documentation. |
| Observed transaction codes such as `by`, `sl`, `li`, `lo`, `dv`, `in`, `dp`, `wd` | Supported as observed codes | Medium | Not a complete official code matrix. |
| Uppercase cancellation behavior, e.g. `by` → `BY` | Supported in third-party Axys/APX workflows | Medium | Universality across versions and import methods is Unknown. |
| Complete native Axys transaction-code matrix | Unknown | Unknown | Not supplied. |
| Complete native APX transaction-code matrix | Unknown | Unknown | Not supplied. |
| Official IMEX transaction object names | Unknown | Unknown | Not supplied. |
| Native Axys transaction storage model | Unknown | Unknown | Not supplied. |
| Native APX database transaction schema | Unknown | Unknown | Not supplied. |
| Native audit trail and posting-status model | Unknown | Unknown | Not supplied. |

### 1.2 Interpretation Rule

Transaction code alone is not sufficient to determine accounting meaning. Interpretation may depend on:

| Context Item | Why It Matters | Confidence |
|---|---|---:|
| Transaction code | Primary event indicator. | High |
| Quantity sign | May determine inflow versus outflow. | Medium |
| Amount sign | May determine cash direction. | Medium |
| Security type | May distinguish cash, security, bond, fee, margin, sweep, or short activity. | Medium |
| Source/destination type | May define the offsetting side of the accounting entry. | Medium |
| Source/destination symbol | May identify cash, margin, short, or wash symbols. | Medium |
| Special security type/symbol | Used in observed fee and expense handling. | Medium |
| Portfolio/account configuration | May affect interpretation, including deliver-in/out behavior. | Medium |
| Custodian or interface translation | Integration-specific mappings may alter native codes. | Medium |
| Reversal/cancellation context | Uppercase code may represent deletion/cancellation in observed workflows. | Medium |

---

## 2. Axys

### 2.1 Axys Transaction Role

| Statement | Confidence | Notes |
|---|---:|---|
| Axys supports portfolio accounting workflows involving transactions, positions, settlement/trade information, tax-lot or average-cost accounting, reporting, performance measurement, and reconciliation. | High for broad capability | Source material identifies this from SS&C/Advent product-level evidence, but not detailed mechanics. |
| Axys transaction import workflows can route transactions through a Trade Blotter for review and posting. | Medium | Supported by ByAllAccounts and WealthTechs integration evidence. |
| Axys native transaction file structure is fully known from the supplied material. | Unknown | Not supplied. |
| Axys native transaction-code matrix is fully known from the supplied material. | Unknown | Not supplied. |

### 2.2 Axys Trade Blotter and IMEX Workflow

The supplied research describes an Axys-oriented third-party integration workflow:

```text
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

**Confidence:** Medium. This is observed third-party integration behavior, not proof of exclusive native Axys behavior.

### 2.3 Axys Files, Folder Labels, and Utilities Observed in Integration Evidence

| Item | Observed Role | Confidence | Caveat |
|---|---|---:|---|
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
|---|---|---:|---|
| Per-share cost basis | Morningstar conversion evidence states per-share cost-basis data is converted only if provided in exported Advent `.cli` file. | Medium | This is conversion evidence, not a full native `.cli` spec. |
| Deliver-in / deliver-out interpretation | `li` and `lo` may be interpreted differently depending on a transaction-setting code in the Advent client file. | Medium | Code-only interpretation is unsafe. |
| 53rd-character setting | Setting code `Y` maps `li`/`lo` to Deliver-In/Deliver-Out in Morningstar conversion; setting code `N` maps them to Credit/Debit of Security. | Medium | Specific to supplied conversion evidence. |
| `none` or `client` securities | Transactions linked to securities labeled `none` or `client` may be converted as out-of-pocket cash. | Medium | Conversion-layer behavior. |
| Principal paydowns | Principal paydowns from Axys may create conversion complications, including zero-quantity cases. | Medium | Native Axys mechanics Unknown. |
| Transaction and historical prices | Transaction prices and historical security prices may be converted if present in Axys conversion inputs. | Medium | Exact native field names Unknown. |

### 2.5 Axys Reinvestment Evidence

| Statement | Confidence | Notes |
|---|---:|---|
| Axys distribution reinvestment activity may appear as Buy plus Distribution transaction pairs in conversion data. | Medium | Based on Morningstar Axys conversion evidence. |
| Reinvestment representation can affect downstream realized and unrealized gain/loss reporting. | Medium | Conversion observation. |
| Native Axys reinvestment representation is fully defined by supplied material. | Unknown | Not supplied. |

### 2.6 Axys Fee Evidence

| Item | Observed Meaning in Supplied Material | Confidence | Caveat |
|---|---|---:|---|
| `epus` | Associated with Management Fee conversion in Morningstar Axys conversion evidence. | Medium | May be a transaction code, label, security type, or conversion-layer term; official definition Unknown. |
| `exus` | Associated with Expense conversion in Morningstar Axys conversion evidence. | Medium | May be a transaction code, label, security type, or conversion-layer term; official definition Unknown. |

### 2.7 Axys Cancellation / Reversal Evidence

| Statement | Confidence | Notes |
|---|---:|---|
| WealthTechs Axys evidence documents cancellation behavior using uppercase transaction code, e.g. `by` → `BY`. | Medium | Third-party workflow evidence. |
| Uppercase cancellation behavior is universal across all Axys versions, transaction types, and import methods. | Unknown | Not supported by supplied material. |

---

## 3. APX

### 3.1 APX Transaction Role

| Statement | Confidence | Notes |
|---|---:|---|
| APX workflows include transaction import, blotter review/posting, reporting, reconciliation, and database/reporting alternatives. | Medium | Supported mainly by third-party integration and consultant evidence. |
| APX users may use SQL/database reporting/export alternatives in addition to IMEX. | Medium | Supported by consultant evidence. |
| Native APX database transaction schema is fully known from supplied material. | Unknown | Not supplied. |
| Native APX transaction-code matrix is fully known from supplied material. | Unknown | Not supplied. |

### 3.2 APX Blotter Types Observed

| Blotter | Observed Purpose | Confidence | Caveat |
|---|---|---:|---|
| Trade Blotter | AIA imports transactions into this blotter; can be consolidated or created per custodian. | Medium | Integration workflow evidence. |
| Statement Blotter | Used to post custodian statement transactions; can support reconciliation against OMS or third-party data using REX. | Medium | Integration workflow evidence. |
| Tax Lot Blotter | Used for tax-lot-level reconciliation of APX-calculated lots versus custodian lots. | Medium | Integration workflow evidence. |
| Position Blotter | Used for importing positions into APX. | Medium | Integration workflow evidence. |
| Account Blotter | Used for importing account information. | Medium | Integration workflow evidence. |
| Initial Transaction Blotter | Used to import positions as deliver-in transactions for accounts without transactions, when configured. | Medium | AIA setting; native APX behavior Unknown. |

### 3.3 APX Trade Blotter Organization Options

| Option | Meaning | Confidence |
|---|---|---:|
| Consolidate Into One Blotter | Aggregate all transactions from all custodians into one trade blotter. | Medium |
| Create One Blotter Per Custodian | Distribute transactions into one blotter per custodian. | Medium |
| No Trade Blotter | No trade blotter is created by AIA. | Medium |

### 3.4 APX Transaction Translation Model

The supplied research describes a third-party APX integration model in which source transactions are normalized before APX transaction generation.

| Statement | Confidence | Notes |
|---|---:|---|
| WebPortfolio interprets financial-institution transaction data and assigns a normalized transaction type. | Medium | ByAllAccounts APX guide evidence. |
| Custodial Integrator translates normalized transaction types into APX transactions. | Medium | ByAllAccounts APX guide evidence. |
| Some APX translations depend on the sign of amount or units. | Medium | Examples include positive/negative transfer behavior. |
| Positive-unit transfer maps to APX `li` in the default translation table. | Medium | Integration default, not complete native documentation. |
| Negative-unit transfer maps to APX `lo` in the default translation table. | Medium | Integration default, not complete native documentation. |
| Translation tables may be customized by financial institution. | Medium | Integration behavior. |

### 3.5 APX Observed Transaction/Blotter Fields

| Field | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---|---|---|---|---:|
| APX Transaction Type | Transaction code/type used in APX translation table. | Unknown | Observed | Unknown | Unknown | Medium |
| APX Transaction Src/Dest Type | Source/destination security or cash type. | Unknown | Observed | Unknown | Unknown | Medium |
| APX Transaction Src/Dest Symbol | Source/destination symbol, e.g. cash-like symbols. | Unknown | Observed | Unknown | Unknown | Medium |
| APX Transaction Special Security Type | Special security type used in fee/expense examples. | Unknown | Observed | Unknown | Unknown | Medium |
| APX Transaction Special Security Symbol | Special security symbol used in fee/expense examples. | Unknown | Observed | Unknown | Unknown | Medium |
| Broker Representative Field | Field that can receive `$brok` in AIA workflow. | Unknown | Observed | Unknown | Unknown | Medium |
| Lot Location | Axys-era/APX workflow concept integrated into lot accounting. | Observed as Axys-era concept | Observed | Unknown | Unknown | Medium |
| Comment | Transaction import comment or standalone comment. | Unknown | Observed | Unknown | Unknown | Medium |

### 3.6 APX Initial Deliver-In Transactions

| Statement | Confidence | Notes |
|---|---:|---|
| AIA can create initial deliver-in transactions from positions for accounts with no transactions. | Medium | AIA/APX workflow evidence. |
| If transactions are received on the same day as initial positions in that scenario, the transactions may be ignored and not written to the blotter. | Medium | AIA workflow evidence. |
| Tax lots may be relevant to initial deliver-in construction. | Low to Medium | Details incomplete. |
| Native APX initial deliver-in behavior independent of AIA is fully known. | Unknown | Not supplied. |

### 3.7 APX Statement Transactions and Reconciliation

| Statement | Confidence | Notes |
|---|---:|---|
| APX workflows may distinguish posted portfolio transactions from statement transactions. | Medium | WealthTechs APX evidence. |
| Statement transactions may support reconciliation against custodian or OMS data. | Medium | WealthTechs APX evidence. |
| APX may expose separate UI tabs named `Transactions` and `Statement Transactions` in this workflow. | Medium | Workflow evidence. |

### 3.8 APX Comments and Broker Field

| Topic | Statement | Confidence |
|---|---|---:|
| Transaction comments | Rules in Transaction Translation may apply only to transaction comments in certain cases, while standalone comments always post to the import transaction file in the observed workflow. | Medium |
| Broker representative | A `Use $brok` setting can write `$brok` to the broker representative field in the transaction blotter. | Medium |
| `.cli` reference | `$brok` is described as typically defined in the `.cli` file for each APX portfolio. | Medium |
| Broker translations | Broker translations can map broker representative values to APX-specific codes. | Medium |

### 3.9 APX Cash Sweeps, Margin Sweeps, Short Sweeps, and Merge Logic

| Feature | Observed Behavior | Confidence | Caveat |
|---|---|---:|---|
| Cash sweep removal | AIA includes logic to remove cash sweep transactions from source transaction files. | Medium | AIA behavior. |
| Margin and short sweep removal | AIA has similar removal logic for margin and short sweeps. | Medium | AIA behavior. |
| Example sweep patterns | Examples include `DP,CAUS,CASH,CAUS,MMF`, `DP,CAUS,CASH,CAUS,MARGIN`, and `DP,CAUS,CASH,CAUS,SHORT`. | Medium | Source examples only. |
| FX merge | AIA has options to merge FX transactions. | Medium | AIA behavior; native APX FX workflow Unknown. |
| Accrued-interest merge | AIA has options to merge accrued-interest transactions. | Medium | AIA behavior. |
| Dividend/interest merge | AIA has options to merge dividend and interest entries. | Medium | AIA behavior. |

---

## 4. IMEX

### 4.1 IMEX Role

| Statement | Axys | APX | Confidence | Notes |
|---|---|---|---:|---|
| IMEX is an import/export mechanism used in Axys/APX workflows. | Supported | Supported | Medium | Consultant and third-party integration evidence. |
| IMEX supports CSV, tab, and fixed-format import/export in Axys according to consultant documentation. | Supported | Unknown | Medium | Axys-focused consultant evidence. |
| APX maintained IMEX functionality from v1.x to v4.x, but fixed-format file generation was eliminated according to consultant documentation. | Not applicable | Supported | Medium | Version-specific consultant evidence. |
| IMEX plus transaction/label import through Trade Blotter can move fundamental data in and out of Axys/APX. | Supported | Supported | Medium | Consultant evidence. |
| Official IMEX transaction object names are known. | Unknown | Unknown | Unknown | Not supplied. |
| Complete IMEX transaction field list is known. | Unknown | Unknown | Unknown | Not supplied. |

### 4.2 Axys IMEX Details Observed

| Detail | Observed Value | Confidence | Caveat |
|---|---|---:|---|
| Utility | `imex32.exe` | Medium | Third-party integration evidence. |
| Import target | Trade Blotter / `topost.trn` in observed workflow. | Medium | Workflow-specific. |
| Logs | IMEX logs written to `$pathlog` in observed workflow. | Medium | Exact format Unknown. |
| Input support | CSV, tab, fixed-format according to consultant evidence. | Medium | Version coverage Unknown. |

### 4.3 APX IMEX Details Observed

| Detail | Observed Value | Confidence | Caveat |
|---|---|---:|---|
| IMEX availability | APX maintained IMEX functionality in versions referenced by consultant source. | Medium | Exact version behavior Unknown. |
| Fixed-format generation | Eliminated in APX according to consultant documentation. | Medium | Needs official confirmation. |
| Alternative access | SQL/database reporting/export tools may be available. | Medium | APX-specific consultant evidence. |
| Official transaction import/export object names | Unknown | Unknown | Not supplied. |

### 4.4 Candidate IMEX Transaction Fields

The following fields are expected from accounting practice and supplied research, but official IMEX names are not supplied. Therefore the IMEX column names should be treated as **Unknown** until official documentation or production exports are obtained.

| Field | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---|---|---|---|---:|
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

---

## 5. REP and Reports

### 5.1 Report and REP Evidence

| Report / Interface | System | Description | Confidence | Notes |
|---|---|---|---:|---|
| Transaction Summary Report | APX / Advent reports | Displays account transactions maintained by Advent; sample evidence includes dates, quantity, symbol, security, unit price, and amount. | Medium | Report sample and public report-guide evidence. |
| REP transaction reports | Axys | Unknown | Unknown | Exact report names, parameters, and fields not supplied. |
| REP transaction reports | APX | Unknown beyond Transaction Summary Report evidence | Unknown to Medium | Transaction Summary Report exists, but exact REP implementation Unknown. |
| Replang reports | Axys/APX | Consultant source lists Replang as a report/export alternative. | Medium | Exact transaction report code Unknown. |
| Report Writer Pro / Excel export / ETL | Axys/APX | Consultant source lists these as alternatives. | Medium | Exact transaction fields Unknown. |
| APX SQL/database access | APX | Consultant source lists SQL/database access as an APX reporting/export alternative. | Medium | Native schema Unknown. |

### 5.2 Transaction Summary Report — Observed Columns

The supplied research includes sample column groups for an APX/Advent Transaction Summary Report.

| Section | Observed Columns | Confidence |
|---|---|---:|
| Dividends | Ex-Date, Pay-Date, Symbol, Security, Amount | Medium |
| Contributions | Trade Date, Settle Date, Quantity, Symbol, Security, Unit Price, Amount | Medium |
| Withdrawals | Trade Date, Settle Date, Quantity, Symbol, Security, Unit Price, Amount | Medium |

### 5.3 Unknown REP Details

| Question | Status |
|---|---|
| Which REP reports expose transactions in Axys? | Unknown |
| Which REP reports expose transactions in APX beyond the Transaction Summary Report evidence? | Unknown |
| What are the official APX Transaction Summary Report parameters? | Unknown |
| What are the official APX Transaction Summary Report field names? | Unknown |
| Do REP reports read stored posted records, recalculated values, staged blotter values, or a mixture? | Unknown |
| How do REP outputs reconcile to IMEX exports and native accounting records? | Unknown |

---

## 6. Data Model

### 6.1 Conceptual Model

```text
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
|---|---|---|---:|
| Portfolio Master | Maps transaction to account/portfolio. | Unknown portfolio, inactive account, duplicate mapping. | High |
| Security Master | Identifies asset and security type. | Unknown security, duplicate security, ambiguous identifier. | High conceptually; Medium for integration evidence |
| Currency / FX | Supports multi-currency transaction and base-currency reporting. | Missing FX rate, invalid currency, settlement mismatch. | Medium |
| Pricing | Supports buy/sell valuation, cost basis, and reconciliation. | Missing price, price inconsistent with market close. | High conceptually; Medium for native field behavior |
| Corporate Actions | May generate or alter transaction interpretation. | Missing split, return of capital, paydown, reorg event. | Medium |
| Translation Configuration | Maps custodian/source records to Axys/APX accounting form. | Wrong transaction type, wrong security, wrong portfolio. | Medium |
| `.cli` / client settings | May affect interpretation, e.g. `li`/`lo` behavior in Axys conversion evidence. | Misclassified deliver-in/out versus credit/debit. | Medium |

### 6.3 Downstream Dependencies

| Downstream Area | Transaction Impact | Confidence |
|---|---|---:|
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
|---|---|---|---:|
| Acquire Source Data | Obtain transactions from custodian, broker, OMS, manual entry, provider, or conversion file. | Missing file, stale file, incomplete batch. | Medium |
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

This table documents AIA/APX integration behavior, not confirmed native APX processing order.

| Order | Step | Applies To | Confidence |
|---:|---|---|---:|
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

---

## 7. Common Fields

### 7.1 Core Transaction Field Dictionary

| Field | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---|---|---|---|---:|
| Portfolio ID | Portfolio/account identifier. | Expected | Expected | Unknown name | Unknown | High conceptually |
| Transaction Code | Accounting event code/type. | Observed in examples | Observed in examples | Unknown name | Unknown | Medium for codes; Unknown for official matrix |
| Security Identifier | Security involved in transaction. | Expected | Expected | Unknown name | Symbol/Security observed in report sample | High conceptually |
| Trade Date | Execution/economic date. | Expected | Expected | Unknown name | Observed in report sample | High conceptually |
| Settlement Date | Settlement/cash date. | Expected | Expected | Unknown name | Observed as Settle Date in report sample | High conceptually |
| Entry Date | Date entered into system. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Posting Date | Date posted to accounting records. | Unknown | Unknown | Unknown | Unknown | Unknown |
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
| Comment | Free-form note. | Unknown | Observed in import workflow | Unknown name | Unknown | Medium for APX workflow |

### 7.2 Public Example Transaction Row

The supplied research includes this public third-party example row:

```text
acct123,010101,010101,by,csus,appl,100,caus,cash,10000
```

A cancellation example uppercases the transaction code:

```text
acct123,010101,010101,BY,csus,appl,100,caus,cash,10000
```

Tentative interpretation:

| Position | Observed Value | Tentative Meaning | Confidence |
|---:|---|---|---:|
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

---

## 8. Transaction Codes

### 8.1 Warning

The following matrix is an observed-code catalog from supplied research. It is **not** an official Axys or APX transaction-code reference. Codes may be native, integration-layer mappings, version-specific, configuration-dependent, context-dependent, or incomplete.

### 8.2 Observed Transaction Code Matrix

| Code | Observed Meaning | Axys | APX | Confidence | Notes |
|---|---|---|---|---:|---|
| `by` | Buy | Observed in examples; official status Unknown | Observed | Medium | Public integration documentation only. |
| `BY` | Cancellation/deletion of Buy | Observed | Observed | Medium | Uppercase cancellation observed; universality Unknown. |
| `sl` | Sell | Unknown | Observed | Medium | Requires vendor confirmation. |
| `ss` | Short sale | Unknown | Observed | Medium | Requires vendor confirmation. |
| `cs` | Cover short | Unknown | Observed | Medium | Observed in APX integration evidence. |
| `li` | Deliver in / transfer in / credit / deposit / positive movement | Observed | Observed | Medium | Meaning may depend on sign/configuration. |
| `lo` | Deliver out / transfer out / debit / withdrawal / negative movement | Observed | Observed | Medium | Meaning may depend on sign/configuration. |
| `dv` | Dividend / income / reinvestment leg | Unknown | Observed | Medium | Often relevant to reinvestment. |
| `in` | Income / interest | Unknown | Observed | Medium | Requires context. |
| `rc` | Return of capital | Unknown | Observed | Medium | Requires vendor confirmation. |
| `pd` | Principal paydown / bond return-of-capital case | Unknown | Observed | Medium | Bond-related. |
| `ai` | Accrued interest or margin interest | Unknown | Observed | Medium | Context-dependent. |
| `sa` | Sell accrued interest | Unknown | Observed | Medium | Requires vendor confirmation. |
| `pa` | Reinvested dividend / accrued-interest-related buy-like case | Unknown | Observed | Low to Medium | Meaning requires further verification. |
| `dp` | Debit / fee-related / tax / service charge / cash-security case | Unknown | Observed | Medium | Multiple meanings depending on context. |
| `wd` | Withdrawal / cash-security sell case | Unknown | Observed | Medium | Context-dependent. |
| `;` | Journal / comment / other / split in integration table | Unknown | Observed | Medium | Treat as observed integration behavior only. |

### 8.3 Observed APX Translation Patterns

| Source Transaction Concept | Observed APX Translation Pattern | Confidence | Notes |
|---|---|---:|---|
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
| Return of capital | `rc`; bond security may map to `pd` | Medium | Bond-specific behavior requires verification. |
| Sell | `sl` | Medium | Normal sell. |
| Sell cash security | `wd` | Medium | Cash-security special case. |
| Short | `ss` | Medium | Short sale. |
| Accrued interest on sell | `sa` | Medium | Source table maps accrued interest to `sa`. |
| Service charge | `dp` with `exus custfee` | Medium | Fee-like. |
| Split | `;` | Medium | Source maps split to semicolon/comment-like type. |
| Transfer positive/negative | `li` / `lo` | Medium | Direction depends on sign. |
| Withdrawal | `lo` | Medium | Outflow-like. |

### 8.4 Cancellation and Reversal

| Statement | Axys | APX | Confidence | Notes |
|---|---|---|---:|---|
| Lowercase transaction code may be uppercased to represent cancellation/deletion, e.g. `by` → `BY`. | Observed | Observed | Medium | Third-party workflows. |
| Cancellation transaction fields must sufficiently match the original transaction or blotter error may occur. | Unknown | Observed | Medium | APX integration evidence. |
| Cancellation blotters may be created from historical transaction files. | Observed | Observed | Medium | WealthTechs evidence. |
| Cancellation workflows should be treated as high-risk and backed up/reviewed. | Supported recommendation | Supported recommendation | Medium | Based on source warnings. |
| Uppercase cancellation is universal native behavior across all versions and import methods. | Unknown | Unknown | Unknown | Not supplied. |

---

## 9. Examples

### 9.1 Buy Example from Public Integration Evidence

```text
acct123,010101,010101,by,csus,appl,100,caus,cash,10000
```

| Interpretation Item | Value | Confidence |
|---|---|---:|
| Account / portfolio | `acct123` | Medium |
| Transaction code | `by` | Medium |
| Security type | `csus` | Low to Medium |
| Security symbol | `appl` | Low to Medium |
| Quantity | `100` | Low to Medium |
| Source/destination type | `caus` | Low to Medium |
| Source/destination symbol | `cash` | Low to Medium |
| Amount | `10000` | Unknown |

### 9.2 Cancellation Example

```text
acct123,010101,010101,BY,csus,appl,100,caus,cash,10000
```

| Interpretation Item | Value | Confidence |
|---|---|---:|
| Cancellation indicator | `BY`, uppercase version of `by` | Medium |
| Native universality | Unknown | Unknown |
| Required match fields | Unknown | Unknown for Axys; Medium for APX integration evidence that mismatch can produce blotter error |

### 9.3 Reinvestment Pattern

| System | Observed Pattern | Confidence | Caveat |
|---|---|---:|---|
| Axys | Reinvestment may appear as Buy plus Distribution transaction pairs in conversion data. | Medium | Conversion evidence only. |
| APX | Reinvestment may translate as `dv` and `by` pair in ByAllAccounts integration evidence. | Medium | Integration evidence only. |

### 9.4 Fee Pattern

| System | Observed Pattern | Confidence | Caveat |
|---|---|---:|---|
| Axys | `epus` associated with Management Fee conversion; `exus` associated with Expense conversion. | Medium | Official meaning Unknown. |
| APX | Fee transactions may use `dp` plus special security type/symbol such as `exus custfee` or `epus expense`. | Medium | Integration evidence only. |

---

## 10. Known Issues / Quirks

| Issue / Quirk | Axys | APX | Confidence | Notes |
|---|---|---|---:|---|
| Code-only interpretation is unsafe. | Supported | Supported | High as design rule; Medium source evidence | Use code, sign, security type, source/destination fields, symbols, and configuration. |
| Direct file access is risky because file formats can change between versions. | Supported | Unknown / less applicable | Medium | Consultant evidence cites Axys file-format changes between versions. |
| APX SQL/database access may exist as an alternative export path. | Not applicable | Supported | Medium | Native schema Unknown. |
| `li`/`lo` interpretation may depend on `.cli` setting. | Supported | Unknown | Medium | Morningstar Axys conversion evidence. |
| Reinvestments may appear as paired transactions. | Supported | Supported | Medium | Axys conversion and APX integration evidence. |
| Fees may depend on special security type/symbol and description translation. | Supported | Supported | Medium | Terminology differs across sources. |
| Principal paydowns may produce downstream conversion/reconciliation complications. | Supported | Unknown | Medium | Axys conversion evidence. |
| Uppercase cancellation codes are observed. | Supported | Supported | Medium | Universality Unknown. |
| AIA/APX import may remove pending records, sweeps, intra-account journals, or merge FX/accrued-interest/dividend-interest records. | Not applicable | Supported in integration workflow | Medium | AIA behavior, not confirmed native APX order. |
| Initial deliver-ins may be generated from positions for accounts with no transactions in AIA/APX workflow. | Unknown | Supported in integration workflow | Medium | Native behavior Unknown. |
| Statement transactions and posted transactions may be distinguished in APX workflows. | Unknown | Supported | Medium | Workflow evidence. |
| `;` may represent journal/comment/other/split in APX integration table. | Unknown | Observed | Medium | Treat only as integration evidence. |

---

## 11. Audit Rules

These rules are candidate transaction audit controls. They are not confirmed native Axys/APX validation behavior unless explicitly noted.

### 11.1 Validation Rules

| Rule | Severity | Description | Required Inputs | Confidence |
|---|---|---|---|---:|
| TR-001 Missing Portfolio | Critical | Transaction references a portfolio that does not exist. | Portfolio ID | High |
| TR-002 Missing Security | Critical | Security transaction references an unknown security. | Security Identifier, Transaction Code | High |
| TR-003 Missing Trade Date | High | Trade-based transaction lacks trade date. | Trade Date | High |
| TR-004 Settlement Before Trade | High | Settlement date precedes trade date. | Trade Date, Settlement Date | High |
| TR-005 Missing Quantity | High | Security transaction lacks required quantity. | Quantity, Transaction Code | High |
| TR-006 Missing Price | Medium | Price-required transaction has no execution price. | Price, Transaction Code | High |
| TR-007 Invalid FX Rate | Medium | Foreign-currency transaction has missing or invalid FX rate. | Currency, FX Rate | Medium |

### 11.2 Translation and Blotter Rules

| Rule | Severity | Description | Confidence |
|---|---|---|---:|
| TR-008 Portfolio Translation Failure | Critical | External portfolio/account cannot be translated. | Medium |
| TR-009 Security Translation Failure | Critical | External security cannot be translated. | Medium |
| TR-010 Unsupported Transaction Type | High | External transaction type has no mapping. | Medium |
| TR-011 Trade Blotter Exception | Medium | Transaction remains in exception state. | Medium |
| TR-012 Cancellation Mismatch | High | Cancellation transaction does not sufficiently match original transaction. | Medium |
| TR-013 Cancellation Control | High | Cancellation blotters require review, backup, and operational controls. | Medium |

### 11.3 Accounting Rules

| Rule | Severity | Description | Confidence |
|---|---|---|---:|
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
|---|---|---|---:|
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

---

## 12. Version Differences

| Topic | Axys | APX | Confidence | Notes |
|---|---|---|---:|---|
| Axys v2.x binary files and IMEX | Consultant evidence says Axys v2.x introduced binary file formats and IMEX allowed CSV, tab, and fixed formats. | Not applicable | Medium | Needs official confirmation. |
| Axys v3.7 to v3.8 file conversion | Consultant evidence says upgrading from Axys v3.7 to v3.8 required file conversion and produced some files with different formats. | Not applicable | Medium | Supports caution against direct file access. |
| APX v1.x to v4.x IMEX | Not applicable | Consultant evidence says APX maintained IMEX functionality but eliminated fixed-format file generation. | Medium | Needs official confirmation. |
| Native transaction code changes by version | Unknown | Unknown | Unknown | Not supplied. |
| Native Trade Blotter behavior changes by version | Unknown | Unknown | Unknown | Not supplied. |
| Native REP report changes by version | Unknown | Unknown | Unknown | Not supplied. |

---

## 13. References

The supplied research identifies the following source categories and specific references. Confidence varies by source type.

| ID | Source | Type | System | Topics | Confidence |
|---:|---|---|---|---|---:|
| SRC-001 | SS&C Advent Axys Product Page | Vendor product page | Axys | Portfolio accounting, reporting, performance, reconciliation, transactions, positions, settlement/trade information, tax-lot or average-cost accounting, trade-date or settlement-date accounting. | High for capabilities; Low for mechanics |
| SRC-002 | AdventGuru — Getting Data In and Out of Advent APX and Axys | Consultant article | Axys/APX | IMEX, Trade Blotter, import/export, Replang, reports, direct-file-access risks, APX SQL/database options. | Medium |
| SRC-003 | WealthTechs AIA User Manual — APX Users | Third-party integration manual | APX | Trade/Statement/Tax Lot/Position/Account blotters, transaction translation, cancellation, comments, broker fields, processing order. | Medium |
| SRC-004 | WealthTechs AIA User Manual — Axys Users | Third-party integration manual | Axys | Transaction cancellation, IMEX workflow, blotters, cancellation example. | Medium |
| SRC-005 | ByAllAccounts Custodial Integrator — APX User Guide | Third-party integration manual | APX | Translation tables, reversals, fees, imports, sign-dependent translation, source/destination fields, special security fields. | Medium |
| SRC-006 | ByAllAccounts Custodial Integrator — Axys User Guide | Third-party integration manual | Axys | Trade Blotter workflow, IMEX import, `topost.trn`, `imex32.exe`, folder labels, IMEX logs, security/reference files. | Medium |
| SRC-007 | Morningstar Office Advent Axys Conversion Guide | Third-party migration guide | Axys | Reinvestment, deliver-in/out interpretation, `.cli`, cost basis, fees, paydowns, transaction prices, historical prices, conversion caveats. | Medium |
| SRC-008 | Advent Portfolio Exchange Reports Guide | Vendor report guide / public PDF reference | APX | Transaction Summary Report existence. | Low to Medium |
| SRC-009 | Wealth Management Reports / Advent report sample | Vendor/report sample | APX / SSRS | Transaction Summary Report purpose and sample columns. | Medium |
| SRC-010 | AdventGuru — APX to Axys Conversion | Consultant article | Axys/APX | APX-exported CLI files mapped into Axys `topost.trn`; transaction mappings and tax lots. | Medium |

---

## 14. Unknowns

### 14.1 Transaction Codes

| ID | Unknown | Priority |
|---|---|---:|
| TU-001 | Complete official Axys transaction-code matrix. | High |
| TU-002 | Complete official APX transaction-code matrix. | High |
| TU-003 | Whether Axys and APX transaction codes are identical, overlapping, divergent, version-specific, or configuration-dependent. | High |
| TU-004 | Which observed codes are native versus integration-layer mappings. | High |
| TU-005 | Deprecated or version-specific transaction codes. | Medium |

### 14.2 IMEX

| ID | Unknown | Priority |
|---|---|---:|
| TU-006 | Official Axys IMEX transaction export object names. | High |
| TU-007 | Official Axys IMEX transaction import object names. | High |
| TU-008 | Official APX IMEX transaction export/import object names. | High |
| TU-009 | Complete IMEX transaction field list. | High |
| TU-010 | Official Trade Blotter import layout. | High |
| TU-011 | IMEX log fields and validation messages. | Medium |
| TU-012 | Native IMEX object dependency sequence. | Medium |

### 14.3 REP and Reports

| ID | Unknown | Priority |
|---|---|---:|
| TU-013 | Which REP reports expose transaction information. | High |
| TU-014 | Official APX Transaction Summary Report parameters and fields. | High |
| TU-015 | Whether REP transaction values are stored, recalculated, or mixed. | Medium |
| TU-016 | Axys transaction report names and fields. | High |
| TU-017 | APX transaction reports beyond Transaction Summary Report. | Medium |
| TU-018 | How REP report values reconcile to IMEX exports and posted accounting records. | Medium |

### 14.4 Internal Data Model and Posting

| ID | Unknown | Priority |
|---|---|---:|
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
|---|---|---:|
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
|---|---|---:|
| TU-042 | How FX rates are stored. | Medium |
| TU-043 | How cross-currency settlements are represented. | Medium |
| TU-044 | How FX transactions are merged or paired in native workflows. | Medium |
| TU-045 | How base-currency values are stored versus calculated. | Medium |
| TU-046 | Which transaction types affect stored performance. | High |
| TU-047 | Which transaction changes trigger performance restatement. | High |
| TU-048 | How performance restatements are detected or audited. | High |
| TU-049 | Whether edited/deleted historical transactions are visible to performance recalculation workflows. | High |

---

## 15. Minimum Additional Material Needed to Promote Unknowns

To convert this chapter from observed/integration-level evidence into a more authoritative native Axys/APX transaction reference, the following supplied material would be needed:

| Needed Material | Would Resolve |
|---|---|
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
