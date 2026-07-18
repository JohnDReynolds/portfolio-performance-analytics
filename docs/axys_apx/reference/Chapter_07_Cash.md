# Chapter 07 — Cash

**Repository:** AXYS / APX Reference Repository
**Chapter:** `docs/axys_apx/reference/Chapter_07_Cash.md`
**Status:** Technical reference chapter based on repository research and cited public evidence
**Prepared:** 2026-06-29
**Public evidence reviewed:** 2026-07-17

---

## Related chapters

- [Chapter_01_Overview.md](Chapter_01_Overview.md) — provides the repository-wide map and evidence conventions.
- [Chapter_05_Transactions.md](Chapter_05_Transactions.md) — cash movement is often transaction-driven.
- [Chapter_06_Holdings.md](Chapter_06_Holdings.md) — cash and holdings are closely linked in portfolio accounting.
- [Chapter_10_Performance.md](Chapter_10_Performance.md) — cash and flow treatment matter for performance calculations.

## Transaction-Code Boundary

This chapter explains cash effects and cash-evidence surfaces. It should not
maintain a separate transaction-code dictionary for codes such as `li`, `lo`,
`dp`, `wd`, `rc`, or `pd`. Interpret transaction-code semantics through
[Chapter_05_Transactions.md](Chapter_05_Transactions.md), then use this chapter
for cash-specific implications such as source/destination fields, cash-like
symbols, sweep behavior, and external-flow classification evidence.

## 1. Overview

Cash in Axys/APX documentation should be treated as a portfolio-accounting topic with several distinct layers:

Cash activity is best interpreted as a transaction-classification problem rather than a code-only problem. A code such as `dp`, `wd`, `li`, or `lo` can represent cash movement, income, fees, or other accounting activity depending on sign, security type, source/destination fields, special symbols, and translation rules. The supplied research also shows that sweeps and intra-account journals may be removed or netted by integration logic, which should not be assumed to be native Axys or APX behavior without separate evidence.

| Layer | Description | Confidence |
|---|---|---:|
| Cash activity | Deposits, withdrawals, income, fees, sweeps, cash-security buys/sells, margin/short cash activity, and other transactions that affect cash or cash-like accounts. | High Confidence |
| Cash transaction representation | Cash activity appears in inspected integration examples as transaction codes plus security/source/destination type and symbol fields. | High Confidence |
| Cash balances | Ending/current cash by account, date, currency, or cash bucket. | Unknown |
| Cash-like securities | Money market funds, cash sweep vehicles, margin cash, short cash, income cash, dividend-wash symbols, and other cash-like placeholders. | High Confidence as observed tokens; native definitions mostly Unknown |
| IMEX cash interface | Exact Axys/APX IMEX object names and field names for cash balances or cash activity. | Unknown |
| REP/report cash interface | Exact Axys/APX REP or standard report names for cash balances, cash activity, cash reconciliation, or multi-currency cash. | Unknown |

The supplied research supports cash-related transaction and integration behavior
more strongly than native cash-balance storage behavior. Cash is usually observed
through transaction, position, and price workflows rather than through a verified
standalone cash-balance object. Therefore, this chapter documents observed
transaction/interface behavior and preserves cash-balance storage, report, and
IMEX object details as Unknown unless directly supported.

---

## 2. Axys

### 2.1 Product Scope

| Statement | Confidence | Notes |
|---|---:|---|
| Axys is portfolio reporting and portfolio accounting software. | Verified | Supported by supplied research citing SS&C public Axys product material. |
| Axys automates portfolio reporting and accounting workflows. | Verified | Product-level statement only. |
| The supplied public Axys product material does not document the Axys cash data model, IMEX cash objects, REP cash reports, or cash-balance file layouts. | Verified | Absence from inspected source material. |

### 2.2 Axys Cash Transaction Representation

| Statement | Confidence | Notes |
|---|---:|---|
| Axys-oriented transaction translation examples include cash-impacting codes such as `li`, `lo`, `dp`, `wd`, `by`, `sl`, `dv`, `in`, `ai`, `rc`, `pd`, `sa`, `cs`, and `ss`. | High Confidence | Observed in supplied research from ByAllAccounts Custodial Integrator Axys material. |
| Axys cash movements can be represented with source/destination type and source/destination symbol fields. | High Confidence | Observed tokens include `$pty`, `$ity`, `$pth`, `$cash`, `$income`, `CAUS`, `CASH`, `MMF`, `MARGIN`, and `SHORT`. |
| Axys cash-related transaction examples commonly use `$cash` as a source/destination symbol. | High Confidence | Observed in default transaction translation examples. |
| Axys income-related transaction examples may use `$income` as a source/destination symbol. | High Confidence | Observed for dividends, interest, income, and accrued-interest cases. |
| Axys system currency can be configured in integration tooling. | High Confidence | Observed parameter: `axyscur`. |
| Older Axys versions may use a cash asset-class letter other than `c`. | High Confidence | Observed parameter: `axysaccash`; exact version boundaries not supplied. |
| Non-system-currency transactions may require a Mark to Market value in an Axys `topost.trn` integration context. | High Confidence | Observed parameter: `defmarkmarket`. |
| A `Perf/CW` column exists in the Axys `topost.trn` integration context. | High Confidence | Observed parameter: `defperfcw`. |
| The complete native Axys transaction file layout is not established by the supplied material. | Unknown | Do not infer full `topost.trn` layout from integration examples. |

### 2.2.1 Cash Holdings Report Derivation Evidence

Dated Axys Report Writer guidance shows a Cash Holdings report derived from an
Assets by Type report. The example uses report-visible concepts including
`Asset_Class_Code`, `Security_Type_Code`, `Security_Type_Name`, and
`Security_Type_Display_Order`; it uses asset-class code `e` in that particular
configuration. This verifies that cash reporting can be defined through report
classification logic, but it does not establish a universal cash code or native
cash-balance object.

### 2.3 Axys Cash Sweeps

| Statement | Confidence | Notes |
|---|---:|---|
| In a WealthTechs AIA Axys workflow, a cash sweep is a single transaction moving money from one cash account to another, typically between cash and a money market fund. | High Confidence | Third-party integration behavior. |
| In that workflow, cash sweeps may be removed from source `TRN` files when transaction code is `WD` or `DP`, source/destination fields are cash-like, and source/destination symbols exclude specified special symbols. | High Confidence | Third-party integration behavior. |
| Margin and short cash sweeps may be handled as separate optional removal cases. | High Confidence | Observed symbols include `MARGIN` and `SHORT`. |
| The supplied material does not prove that native Axys automatically removes cash sweeps. | Unknown | Treat sweep removal as integration-tool behavior unless verified from Axys documentation or production evidence. |
| Whether standard Axys reports include, exclude, or reclassify sweep transactions is not established. | Unknown | Requires REP/report samples or vendor documentation. |

### 2.4 Axys Intra-Account Cash Journals

| Statement | Confidence | Notes |
|---|---:|---|
| In a WealthTechs AIA Axys workflow, an intra-account cash journal is a pair of opposite transactions that wipe each other out. | High Confidence | Third-party integration behavior. |
| Observed opposite transaction-code pairings include `dp`/`wd`, `li`/`lo`, `ti`/`to`, `si`/`so`, and `tr`/`ts`. | High Confidence | Observed in supplied research. |
| Documented matching criteria include trade date, account, amount, quantity, transaction-code pair, type/security criteria, and symbol exclusions. | High Confidence | Third-party integration behavior. |
| The supplied material does not prove that native Axys automatically nets or removes intra-account cash journals. | Unknown | Treat as integration-tool behavior unless separately verified. |

### 2.5 Axys Multi-Currency Cash

| Statement | Confidence | Notes |
|---|---:|---|
| Axys integration tooling references a configured system currency. | High Confidence | Observed parameter: `axyscur`. |
| Non-system-currency transactions may require a Mark to Market value in an Axys `topost.trn` context. | High Confidence | Observed parameter: `defmarkmarket`. |
| Native Axys cash balance presentation by currency is not established. | Unknown | Need reports, IMEX exports, or vendor documentation. |
| Exchange-rate source, valuation date, local cash fields, and base cash fields are not established. | Unknown | Need multi-currency examples. |

---

## 3. APX

### 3.1 Product Scope

| Statement | Confidence | Notes |
|---|---:|---|
| APX is positioned by SS&C Advent as an integrated portfolio management and client relationship management solution. | Verified | Product-level statement only. |
| APX tracks holdings, transactions, and performance. | Verified | Product-level statement only. |
| The supplied public APX product material does not document the APX cash data model, database tables, IMEX cash objects, REP cash reports, or cash-balance calculations. | Verified | Absence from inspected material. |

### 3.2 APX Cash Transaction Representation

| Statement | Confidence | Notes |
|---|---:|---|
| WealthTechs APX examples use transaction rows beginning with `ACCTX`. | High Confidence | Observed in AIA examples; native status Unknown. |
| WealthTechs APX examples show cash/accounting fields using tokens such as `CAUS`, `CASH`, `MMF`, `MARGIN`, and `SHORT`. | High Confidence | Observed in AIA examples. |
| WealthTechs APX examples use cash-related transaction codes including `DP`, `WD`, `LI`, and `LO`. | High Confidence | Observed in AIA examples. |
| APX broker representative field values may be defined in the `.cli` file for each portfolio in at least one AIA workflow. | High Confidence | Third-party integration behavior. |
| The supplied material does not establish whether `.cli` files are authoritative native APX transaction storage, an import/configuration artifact, or an interface artifact. | Unknown | Needs APX vendor documentation or production evidence. |

### 3.3 APX Cash Sweeps

| Statement | Confidence | Notes |
|---|---:|---|
| In a WealthTechs AIA APX workflow, a cash sweep is a single transaction moving money from one cash account to another, typically between cash and a money market fund. | High Confidence | Third-party integration behavior. |
| Cash sweeps may be removed from source `TRN` records when transaction code is `WD` or `DP`, source/destination type fields are cash-like, and symbols exclude specified special placeholders. | High Confidence | Third-party integration behavior. |
| APX AIA examples include separate margin and short sweep handling using `MARGIN` and `SHORT`. | High Confidence | Third-party integration behavior. |
| The supplied material does not prove that native APX automatically removes or suppresses cash sweeps. | Unknown | Treat as integration-tool behavior unless separately verified. |

### 3.4 APX Intra-Account Cash Journals

| Statement | Confidence | Notes |
|---|---:|---|
| In a WealthTechs AIA APX workflow, intra-account cash journals are opposite transaction pairs that wipe each other out. | High Confidence | Third-party integration behavior. |
| APX AIA uses pair logic similar to Axys AIA for `dp`/`wd`, `li`/`lo`, `ti`/`to`, `si`/`so`, and `tr`/`ts`. | High Confidence | Third-party integration behavior. |
| The supplied material does not prove native APX automatically nets these cash journals. | Unknown | Treat as integration-tool behavior unless separately verified. |

### 3.5 APX Multi-Currency Cash

| Statement | Confidence | Notes |
|---|---:|---|
| APX product material supports APX as a multi-asset, portfolio-accounting/reporting platform. | Verified | Product-level only. |
| Native APX cash balance presentation by currency is not established. | Unknown | Need APX reports, IMEX exports, SQL/public-view documentation, or vendor documentation. |
| APX equivalents for Axys `topost.trn`, `Perf/CW`, and Mark to Market fields are not established. | Unknown | Requires APX-specific import/export documentation. |

---

## 4. Cash Transactions

### 4.1 Observed Cash-Impacting Codes

The following table catalogs observed codes from supplied research. These are not a complete official Axys/APX transaction-code dictionary.

| Code | Observed Context | Likely Cash Impact in Examples | Axys | APX | Confidence |
|---|---|---|---:|---:|---:|
| `li` | Deposits, credits, direct deposits, transfers in, ATM positive, income positive. | Inflow / value moves in. | Observed | Observed in AIA examples | High Confidence as observed |
| `lo` | Withdrawals, checks, debits, payments, transfers out, ATM negative. | Outflow / value moves out. | Observed | Observed in AIA examples | High Confidence as observed |
| `dp` | Cash-security buy, fees, service charges, investment expense, non-cash security debit/tax. | Debit/payment-style cash impact. | Observed | Observed in AIA examples | High Confidence as observed |
| `wd` | Cash-security sell, sweep/withdrawal examples. | Withdrawal/sell-style cash impact. | Observed | Observed in AIA examples | High Confidence as observed |
| `by` | Buy; reinvested-dividend buy leg. | Security purchase funded by cash. | Observed | Observed in transaction research | High Confidence as observed |
| `sl` | Sell; positive closure leg. | Security sale producing cash. | Observed | Observed in transaction research | High Confidence as observed |
| `dv` | Dividend and reinvested-dividend leg. | Income/cash-related. | Observed | Observed in transaction research | High Confidence as observed |
| `in` | Income, interest, dividends on cash securities. | Income/cash-related. | Observed | Observed in transaction research | High Confidence as observed |
| `ai` | Negative interest; margin interest. | Accrued/margin-interest cash effect. | Observed | Observed in transaction research | High Confidence as observed |
| `rc` | Return of capital. | Maps to portfolio cash context (`$pty` / `$cash`) in ByAllAccounts Axys/APX translation evidence. | Observed | Observed in transaction research | Confirmed in integration mapping evidence |
| `pd` | Return of capital for bond security / principal paydown. | Maps to `$pty` / `$cash` portfolio-cash context in ByAllAccounts Axys/APX translation evidence; classify only with MBS/ABS/amortizing-security paydown context. | Observed | Observed in transaction research | Confirmed in integration mapping evidence |
| `sa` | Sale accrued interest / sell-side accrued interest. | Income/cash-related fixed-income trade adjunct. | Observed | Observed in transaction research | High Confidence as observed |
| `cs` | Cover short / negative closure leg. | Cash/security cash-impacting; APX integration mapping points to `$pty / $cash`, but exact sign mechanics remain site-specific. | Observed | Observed in transaction research | Medium-High for code meaning |
| `ss` | Short sale. | Cash/security cash-impacting; APX integration mapping points to `awus / none`, so unrestricted cash treatment is not safe to assume. | Observed | Observed in transaction research | Medium-High for code meaning |

`li` and `lo` are best treated as external-flow candidates rather than
automatic contribution/withdrawal conclusions. A true client cash
contribution or withdrawal should be separated from security-in-kind
transfers, cash-security activity, fees, sweeps, corrections, and
same-day internal journals by checking security type, source/destination
type, source/destination symbol, amount and quantity signs, and
firm-specific mapping.

### 4.2 Transaction-Code Caveats

| Caveat | Confidence | Notes |
|---|---:|---|
| An integration tool may uppercase a historical transaction code while creating a cancellation Trade Blotter instruction. | High Confidence for the cited workflow | Source-stage evidence is required; this does not prove a posted cash transaction. |
| It is unknown whether uppercase cancellation instructions survive in ordinary posted extracts or represent a universal native convention. | Unknown | Requires vendor or installed-system evidence by extraction path and version. |
| Transaction-code semantics may be site-configured, version-sensitive, or dependent on source/destination fields. | Medium Confidence | Do not interpret cash activity from transaction code alone. |
| Cash classification depends on code, sign, quantity, amount, security type, source/destination type, source/destination symbol, and configuration. | High Confidence as implementation guidance | Strongly supported by integration examples. |

---

## 5. Cash Balances

The supplied material does not establish authoritative native Axys or APX cash-balance storage.

| Topic | Axys | APX | Confidence |
|---|---|---|---:|
| Daily stored cash balances | Unknown | Unknown | Unknown |
| Cash derived from transaction history | Plausible, not verified | Plausible, not verified | Unknown |
| Cash as a holdings/security row | Unknown | Unknown | Unknown |
| Cash as account-level balance | Unknown | Unknown | Unknown |
| Cash as currency-level balance | Unknown | Unknown | Unknown |
| Settled cash versus trade-date cash | Unknown | Unknown | Unknown |
| Income cash versus principal cash | Inferred only from `$income` / special symbols | Inferred only from AIA special symbols | Medium Confidence for integration evidence; native status Unknown |
| Margin cash and short cash as native buckets | Observed symbols only | Observed symbols only | High Confidence as observed tokens; Unknown native status |

Implementation note: because cash balances may be derived, reported, stored, or represented as cash-like holdings depending on workflow, integrations should record the extraction source and not assume that a cash balance from one report equals a cash balance from another report or export.

---

## 6. Cash-Like Securities and Sweep Vehicles

### 6.1 Observed Cash-Like Tokens

| Concept / Token | Meaning Supported by Source | Axys | APX | Confidence |
|---|---|---:|---:|---:|
| `$cash` | Special source/destination symbol used for cash in Axys CI transaction translations. | Yes | Unknown from supplied APX sources | High Confidence for Axys CI |
| `$income` | Special source/destination symbol used for income-related Axys CI transaction translations. | Yes | Unknown from supplied APX sources | High Confidence for Axys CI |
| `$pty` | Source/destination type in many cash-impacting Axys CI translations; exact native meaning not expanded. | Yes | Unknown | High Confidence as observed; Unknown meaning |
| `$ity` | Source/destination type in income/accrued-interest Axys CI translations; exact native meaning not expanded. | Yes | Unknown | High Confidence as observed; Unknown meaning |
| `$pth` | Source/destination type used for margin interest; exact native meaning not expanded. | Yes | Unknown | High Confidence as observed; Unknown meaning |
| `CAUS` | Cash/security type token in WealthTechs examples; exact expansion not confirmed. | Yes | Yes | High Confidence as observed; Unknown expansion |
| `CASH` | Cash symbol/token in examples. | Yes | Yes | High Confidence |
| `MMF` | Money market fund / sweep vehicle token in examples. | Yes | Yes | High Confidence |
| `MARGIN` | Margin cash/sweep symbol in examples. | Yes | Yes | High Confidence |
| `caus margin` | Margin/cash context used in public negative-interest or margin-interest mappings. | Yes | Unknown | High Confidence as observed; native definition Unknown |
| `SHORT` | Short cash/sweep symbol in examples. | Yes | Yes | High Confidence |
| `dvwash` | Dividend wash special symbol excluded from sweep-removal logic. | Yes | Yes in AIA docs | High Confidence as observed; Unknown native definition |
| `dvshrt` | Special symbol excluded from sweep-removal logic. | Yes | Yes | High Confidence as observed; Unknown native definition |
| `dvlong` | Special symbol excluded from sweep-removal logic. | Yes | Yes | High Confidence as observed; Unknown native definition |
| `cashrt` | Special symbol excluded from sweep-removal logic. | Yes | Yes | High Confidence as observed; Unknown native definition |
| `calong` | Special symbol excluded from sweep-removal logic. | Yes | Yes | High Confidence as observed; Unknown native definition |
| `income` | Special symbol excluded from sweep-removal logic. | Yes | Yes | High Confidence as observed; Unknown native definition |
| Cash asset-class code `c` | Default Axys CI cash asset-class letter; older Axys versions may differ. | Yes | Not established | High Confidence for Axys CI |

### 6.2 Practical Implications

| Issue | Practical Implication | Confidence |
|---|---|---:|
| Money market sweep vehicles may look like holdings or cash depending on extract/report path. | Downstream tools should explicitly decide whether to treat sweep vehicles as cash, securities, or both. | High Confidence |
| Margin and short cash require separate handling. | Do not assume all cash-like symbols represent unrestricted long cash. | High Confidence |
| Short-sale proceeds require site-specific cash evidence. | Public APX integration evidence maps `ss` differently from ordinary sells, so do not assume short-sale proceeds post to unrestricted portfolio cash without source/destination or report support. A synthetic demo may disclose a `SHORT`, margin, collateral, or short-proceeds bucket. | Medium-High for mapping evidence; Unknown native mechanics |
| Dividend-wash and income-like symbols are excluded from sweep-removal logic in AIA workflows. | Treat `dvwash`, `dvshrt`, `dvlong`, `cashrt`, `calong`, and `income` cautiously. | High Confidence |
| Margin-interest mappings may use margin cash context such as `caus margin`. | Separate financing expense from external cash-flow treatment. | High Confidence as observed mapping guidance |
| Native definitions of many symbols are not supplied. | Do not promote observed tokens into authoritative native field definitions. | Unknown |

---

## 7. IMEX

### 7.1 Supported Findings

| Statement | Axys | APX | Confidence | Notes |
|---|---:|---:|---:|---|
| IMEX is relevant to moving data in and out of Axys/APX. | Yes | Yes | High Confidence | Supported by supplied IMEX research and AdventGuru-derived research. |
| Cash-related data needed for external products may be obtainable through transaction, holding/position, and cash-activity feeds. | Yes | Yes | Medium Confidence | Inference from custodian data and integration manuals; not an IMEX manual. |
| Axys CI cash-adjacent artifacts include `topost.trn`, `ptopost.trn`, `.pos`, `$pathcli`, `$pathinf`, `$pathpri`, `$pathlog`, `imex32.exe`, `pospos32.exe`, and `sipos30`. | Yes | No | Verified for CI workflow | These are integration/reconciliation artifacts, not proof of a native cash-balance object. |
| A standard IMEX cash-balance object name is not verified. | Unknown | Unknown | Unknown | No verified object named `cash`, `cashbal`, `cash_balance`, or equivalent in supplied material. |
| Exact cash-related IMEX column names are not verified. | Unknown | Unknown | Unknown | Need IMEX documentation or sample exports. |
| Whether beginning cash, ending cash, settled cash, trade-date cash, or multi-currency cash balances are distinct IMEX objects is not verified. | Unknown | Unknown | Unknown | Need IMEX documentation or sample exports. |

### 7.2 IMEX Questions to Preserve

| Question | Current Status |
|---|---:|
| Does Axys IMEX provide a transaction export object containing cash-impacting transaction fields? | Unknown |
| Does APX IMEX provide a transaction export object containing cash-impacting transaction fields? | Unknown |
| Does Axys IMEX provide a holdings/positions export where cash appears as a security or asset-class row? | Unknown |
| Does APX IMEX provide a holdings/positions export where cash appears as a security or asset-class row? | Unknown |
| Does IMEX distinguish custodian cash, cash equivalents, money market sweep vehicles, margin cash, short cash, income cash, dividend wash, and currency cash? | Unknown |
| Are cash balances exportable by portfolio, date, currency, custodian, lot location, account, sleeve, strategy, or group? | Unknown |
| Are cash records stored or calculated at export time? | Unknown |
| Are cash records trade-date, settlement-date, or configurable? | Unknown |

---

## 8. REP / Reports

### 8.1 Supported Findings

| Statement | Axys | APX | Confidence | Notes |
|---|---:|---:|---:|---|
| Axys has an extensive library of predefined reports and supports report customization. | Yes | N/A | Verified | Product-level evidence only. |
| APX has extensive reporting capabilities. | N/A | Yes | Verified for general reporting | Cash-specific report names Unknown. |
| Cash-specific REP report names are not verified. | Unknown | Unknown | Unknown | No supplied REP report library, report definitions, or production cash reports. |
| Whether standard reports present cash as a security holding row, separate subtotal, account-level balance, or currency-level balance is not verified. | Unknown | Unknown | Unknown | Need report output or report source. |

### 8.2 REP Questions to Preserve

| Question | Current Status |
|---|---:|
| Which Axys standard REP reports display cash balances? | Unknown |
| Which APX standard reports display cash balances? | Unknown |
| Which reports show cash activity versus cash balance? | Unknown |
| Do cash reports include unsettled trades? | Unknown |
| Do cash reports support settlement-date versus trade-date cash? | Unknown |
| Do cash reports separate principal cash, income cash, margin cash, short cash, and sweep vehicles? | Unknown |
| Are cash totals report-date values, period averages, stored balances, or derived values? | Unknown |
| Can reports be parameterized to include/exclude money market sweep vehicles from cash? | Unknown |
| Which report fields are available for cash currency, exchange rate, local cash, and base cash? | Unknown |

---

## 9. Common Fields and Tokens

This table is a conservative field/token dictionary. It is not a canonical Axys/APX data dictionary.

| Field / Token | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---:|
| Transaction code | Code such as `li`, `lo`, `dp`, `wd`, `by`, `sl`, `dv`, `in`, `ai`, `rc`, `pd`, `sa`, `cs`, `ss`. | Observed | Observed for some codes | Unknown | Unknown | High Confidence |
| `ACCTX` | Transaction-row prefix/object marker shown in WealthTechs AIA examples. | Observed | Observed | Unknown | N/A | High Confidence as observed; Unknown native status |
| Trade date | Used as matching criterion for cash journal removal. | Observed | Observed | Unknown | Unknown | High Confidence |
| Account | Used as matching criterion for cash journal removal. | Observed | Observed | Unknown | Unknown | High Confidence |
| Amount | Used in examples and as matching criterion. | Observed | Observed | Unknown | Unknown | High Confidence |
| Quantity | Used in examples and as matching criterion. | Observed | Observed | Unknown | Unknown | High Confidence |
| Source/Destination Type | Cash/income/security type field in integration examples. | Observed | Observed | Unknown | Unknown | High Confidence |
| Source/Destination Symbol | Cash/income/security symbol field in integration examples. | Observed | Observed | Unknown | Unknown | High Confidence |
| Security Type | Used in sweep-removal criteria and examples. | Observed | Observed | Unknown | Unknown | High Confidence |
| Security Symbol | Used in sweep-removal criteria and examples. | Observed | Observed | Unknown | Unknown | High Confidence |
| Mark to Market | Field required for non-system-currency transactions in ByAllAccounts Axys integration context. | Observed | Unknown | Unknown | Unknown | High Confidence for Axys CI |
| `Perf/CW` | Column in Axys `topost.trn` in ByAllAccounts CI context. | Observed | Unknown | Unknown | Unknown | High Confidence for Axys CI |
| Broker rep field | Field populated by `$brok` in AIA manuals; APX manual says `.cli` typically defines it per portfolio. | Observed | Observed | Unknown | Unknown | High Confidence |
| Cash balance | Ending/current cash amount by account/currency/date. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Settled cash | Cash available after settlement. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Trade-date cash | Cash including trade-date activity regardless of settlement. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Income cash | Separate bucket for income-related cash. | Inferred from `$income` | Inferred from AIA symbol list | Unknown | Unknown | Medium Confidence |
| Margin cash | Separate cash-like bucket or symbol. | Observed token | Observed token | Unknown | Unknown | High Confidence as observed token |
| Short cash | Separate cash-like bucket or symbol. | Observed token | Observed token | Unknown | Unknown | High Confidence as observed token |
| Sweep vehicle | Money market/cash-equivalent vehicle used to move funds to/from cash. | Observed | Observed | Unknown | Unknown | High Confidence |

---

## 10. Examples

### 10.1 Cash Sweep Example

Observed in WealthTechs manuals:

```text
DP,CAUS,CASH,CAUS,MMF
```

| Element | Possible Meaning | Confidence |
|---|---|---:|
| `DP` | Deposit/payment-style transaction code used in sweep-removal examples. | High Confidence as observed |
| `CAUS` | Cash/security type token; exact expansion not confirmed. | High Confidence as observed; Unknown expansion |
| `CASH` | Cash symbol/token. | High Confidence |
| `MMF` | Money market fund / sweep vehicle token. | High Confidence |

Do not infer the full native Axys/APX file layout from this line.

### 10.2 AIA Transaction Row Examples

Observed examples include:

```text
ACCTX,010117,LI,100,100,CAUS,MMF,CAUS,CASH
ACCTX,010117,LO,100,100,CAUS,MMF,CAUS,CASH
ACCTX,010117,DP,100,100,CAUS,MMF,CAUS,CASH
ACCTX,010117,WD,100,100,CAUS,MMF,CAUS,CASH
```

| Segment | Meaning Status |
|---|---|
| `ACCTX` | Observed transaction-row prefix/object marker in AIA examples; native status Unknown. |
| `010117` | Date-like token in `MMDDYY` format in examples; exact date field definition Unknown. |
| `LI`, `LO`, `DP`, `WD` | Observed transaction codes. |
| `100`, `100` | Amount/quantity fields implied by AIA text; exact order should be verified before implementation. |
| `CAUS,MMF,CAUS,CASH` | Type/symbol pairs in examples. |

### 10.3 Intra-Account Cash Journal Pair Example

Observed in WealthTechs AIA material:

```text
ACCTX,010117,DP,100,100,CAUS,MMF,CAUS,CASH
ACCTX,010117,WD,100,100,CAUS,MMF,CAUS,CASH
```

| Item | Notes | Confidence |
|---|---|---:|
| Pair logic | AIA can treat certain pairs as offsetting cash journals if matching criteria are met. | High Confidence for AIA workflow |
| Native Axys behavior | Not established. | Unknown |
| Native APX behavior | Not established. | Unknown |
| Report impact | Not established. | Unknown |

---

## 11. Known Issues / Quirks

| Quirk | Axys | APX | Confidence | Practical Implication |
|---|---:|---:|---:|---|
| Cash sweeps may be represented as transactions moving money between cash and money market funds. | Observed | Observed | High Confidence | Decide whether sweep vehicles are cash, holdings, or both. |
| Third-party tooling may remove cash sweeps. | Observed | Observed | High Confidence | Imported cash balances may differ depending on whether sweeps are retained. |
| Margin and short sweeps may need separate handling. | Observed | Observed | High Confidence | Do not collapse all cash-like symbols into unrestricted cash. |
| Dividend-wash and income-like symbols may be excluded from sweep-removal logic. | Observed | Observed | High Confidence | Treat `dvwash`, `dvshrt`, `dvlong`, `cashrt`, `calong`, and `income` cautiously. |
| Older Axys versions may use a cash asset-class code other than `c`. | Observed | Unknown | High Confidence for Axys CI | Do not hard-code cash asset class without client validation. |
| Non-system-currency Axys transactions may require Mark to Market values in import files. | Observed | Unknown | High Confidence for Axys CI | Multi-currency cash imports require special validation. |
| Uppercase cancellation instructions can occur in a Trade Blotter staging/control context. | Observed | Observed in reviewed integration evidence | Medium-High | Extract/replay tools must use source-stage evidence and keep controls separate from posted cash activity. |
| Adjustment transactions used for reconciliation/conversion can affect downstream performance in target systems. | Medium Confidence | Medium Confidence | Medium Confidence | Cash-balancing entries can alter contribution/withdrawal treatment if misclassified. |
| Cash balance may be derived rather than stored. | Unknown | Unknown | Unknown | Need client sample data or vendor documentation. |
| Settled versus trade-date cash treatment is not established. | Unknown | Unknown | Unknown | High-priority accounting/reporting gap. |

---

## 12. Version Differences

| Version Difference | Axys | APX | Confidence | Notes |
|---|---:|---:|---:|---|
| Cash asset-class letter may differ in older Axys versions. | Yes | Unknown | High Confidence | ByAllAccounts CI research says older Axys versions may use a cash asset-class letter different from `c`. |
| Bond asset-class letter may differ in older Axys versions. | Yes | Unknown | High Confidence | Relevant to transaction translation generally, not cash-specific. |
| APX-specific version differences for cash handling are not documented in supplied material. | N/A | Unknown | Unknown | Need APX release notes/manuals. |
| Axys/APX IMEX version differences for cash objects or fields are not documented in supplied material. | Unknown | Unknown | Unknown | Need IMEX manuals by version. |
| REP version differences for cash report fields are not documented in supplied material. | Unknown | Unknown | Unknown | Need report-definition files or manuals. |

---

## 13. References

The chapter uses repository research plus cited public evidence.

| Reference | Use in this chapter |
|---|---|
| `axys_apx_reference_blueprint.md`, Version 2.0 | Editorial rules, confidence labels, chapter standards. |
| `../evidence/Research_07_Cash.md` | Granular cash evidence ledger for this chapter. |
| `../evidence/Research_05_Transactions.md` | Supporting transaction-code, source/destination field, and reversal/cancellation context. |
| `../evidence/Research_06_Holdings.md` | Holdings/cash-as-position evidence and extraction boundaries. |
| `../evidence/Research_08_Pricing.md` | Price/cash-equivalent valuation evidence and boundaries. |
| `../evidence/Research_12_IMEX.md` | Supporting IMEX/interface cautions and file/log context. |
| `../evidence/Research_13_REP.md` | Supporting REP/reporting cautions. |
| SS&C Advent Axys public product material | Broad Axys product scope only. |
| SS&C Advent APX public product material | Broad APX product scope only. |
| WealthTechs AIA User Manuals for Axys/APX | Cash sweep, intra-account journal, AIA transaction examples, special cash-like symbols. |
| ByAllAccounts Custodial Integrator User Guide for Axys | Axys transaction translation, `$cash`, `$income`, system currency, cash asset-class setting, Mark to Market, `Perf/CW`, reversal behavior. |
| Morningstar Advent Axys conversion guide | Migration/reconciliation caveats only. |
| SS&C Advent Custodial Data product brief | Broad statement that custodian data workflows include positions, transactions, and cash activity. |
| [CSSI equity assets and Cash Holdings guidance](https://cssisolutions.com/downloads/creating-an-equity-assets-by-type-report-and-a-cash-hold) | Observed Report Writer classification fields and Cash Holdings report derivation. |
| [Official Axys product page](https://www.advent.com/solutions/axys/) | Current report-currency, withholding-tax, and currency-capability context. |

---

## 14. Unknowns

| ID | Unknown | Priority |
|---:|---|---:|
| CASH-U001 | Exact Axys IMEX object/file name for cash balances. | High |
| CASH-U002 | Exact APX IMEX object/file name for cash balances. | High |
| CASH-U003 | Whether cash appears as a holding/security row in standard holdings exports. | High |
| CASH-U004 | Whether Axys stores daily cash balances or derives them from transactions. | High |
| CASH-U005 | Whether APX stores daily cash balances or derives them from transactions. | High |
| CASH-U006 | Standard Axys REP report names for cash balances/activity. | High |
| CASH-U007 | Standard APX report names for cash balances/activity. | High |
| CASH-U008 | Settled-date versus trade-date cash behavior. | High |
| CASH-U009 | Multi-currency cash fields and FX valuation logic. | High |
| CASH-U010 | Whether margin cash, short cash, income cash, and sweep cash are separate native cash buckets. | High |
| CASH-U011 | Exact definitions of `CAUS`, `$pty`, `$ity`, `$pth`, `cashrt`, and `calong`. | Medium |
| CASH-U012 | Whether APX has direct equivalents for Axys `topost.trn`, `Perf/CW`, and Mark to Market fields. | Medium |
| CASH-U013 | Whether native Axys/APX reports automatically exclude or include cash sweep transactions. | High |
| CASH-U014 | Whether transaction-code behavior varies by client configuration beyond asset-class code mapping. | Medium |
| CASH-U015 | Version-by-version changes in cash handling. | Medium |
| CASH-U016 | Exact native Axys cash balance report calculations. | High |
| CASH-U017 | Exact native APX cash balance report calculations. | High |
| CASH-U018 | Whether cash balances are stored, calculated at report time, calculated by IMEX, or calculated by report-specific logic. | High |
| CASH-U019 | Whether standard reports separate cash activity from cash balances. | High |
| CASH-U020 | Whether report cash totals include or exclude sweep vehicles and cash equivalents by default. | High |

---

## 15. Evidence Needed to Upgrade Unknowns

| Needed Material | Purpose |
|---|---|
| Axys IMEX manual or sample IMEX exports containing transactions, holdings, and cash balances. | Verify object names, field names, field order, data types, and cash representation. |
| APX IMEX manual or sample IMEX exports containing transactions, holdings, and cash balances. | Same as Axys for APX. |
| Axys REP report definitions or sample reports showing cash. | Verify report names, fields, filters, subtotals, and cash presentation. |
| APX report definitions or sample reports showing cash. | Verify APX report behavior and Axys/APX differences. |
| Security master examples containing cash securities, money market funds, margin cash, and short cash. | Document how cash-like instruments are classified. |
| Portfolio transaction samples with deposits, withdrawals, dividends, interest, fees, FX, sweeps, margin, shorts, and reversals. | Verify transaction-code behavior. |
| Multi-currency portfolio examples. | Document local/base cash, FX rates, Mark to Market, and currency behavior. |
| Client implementation notes or consultant documentation. | Document production quirks and site-specific configuration. |
| Axys/APX version-specific release notes. | Document version differences. |
