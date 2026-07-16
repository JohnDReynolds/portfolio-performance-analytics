# Research Notes: Cash

**Repository:** AXYS / APX Reference Repository  
**Research file:** `docs/axys_apx/evidence/Research_07_Cash.md`
**Target chapter:** `docs/axys_apx/reference/Chapter_07_Cash.md`
**Prepared:** 2026-06-29  
**Governing specification:** `AXYS_APX_REFERENCE_BLUEPRINT.md`, Version 2.0  
**Scope:** Cash behavior, cash-related transaction handling, cash fields, import/export considerations, reporting gaps, and known unknowns for SS&C Axys and SS&C APX.

---

## 1. Research Standard Applied

The attached blueprint requires the repository to document how Axys and APX actually work, to separate Axys and APX behavior when they differ, and to classify important technical statements as `Verified`, `High Confidence`, `Medium Confidence`, or `Unknown`.

This research file follows that standard.

### Confidence Definitions Used Here

| Classification | Meaning in this research file |
|---|---|
| Verified | Supported directly by a cited source inspected for this research. |
| High Confidence | Supported by a specific source, but source is third-party or not full vendor product documentation. |
| Medium Confidence | Plausible and consistent with source material, but not directly confirmed for Axys/APX internal behavior. |
| Unknown | Not supported by inspected material. Do not treat as fact. |

### Source Quality Notes

| Source Type | Examples Used | Evidence Weight |
|---|---|---|
| Repository governing source | AXYS/APX Reference Blueprint v2.0 | Verified for editorial rules only. |
| SS&C official public pages | Axys/APX product pages and briefs | Verified for broad product scope only; limited technical detail. |
| Third-party implementation manuals | WealthTechs AIA manuals for Axys/APX; ByAllAccounts Custodial Integrator Axys guide | High Confidence for interface behavior shown in the documents; not equivalent to SS&C vendor manuals. |
| Conversion/reconciliation guides | Morningstar Axys conversion guide | Medium Confidence for operational implications; not an Axys technical manual. |
| Direct Axys/APX IMEX/REP manuals | Not supplied | Unknown for many chapter-level details. |

---

## 2. Executive Summary

Publicly available source material confirms that Axys and APX are portfolio accounting/reporting systems and that both are used in workflows involving holdings, transactions, and performance. Publicly available source material also confirms several practical cash-adjacent implementation details:

1. Cash activity is part of custodian/portfolio data flows into Advent products.
2. Cash activity is commonly represented through transaction activity rather than only as a standalone cash-balance object.
3. Axys transaction import workflows use cash-related transaction codes and source/destination type/symbol fields.
4. Third-party integration manuals use pseudo-symbols or special symbols such as `$cash`, `$income`, `dvwash`, `cashrt`, `calong`, `margin`, and `short` in Axys/APX-oriented transaction translation examples.
5. Cash sweeps and intra-account cash journals are known integration quirks because they may create offsetting movements that some integrators deliberately remove.
6. Public sources do not provide enough evidence to document authoritative Axys or APX IMEX object names, REP report names, database tables, stored cash-balance tables, or exact standard report behavior.

The research file therefore provides a usable factual base for
`../reference/Chapter_07_Cash.md`, but many implementation details remain
`Unknown` until supported by Axys/APX documentation, IMEX exports, REP report
definitions, or production examples.

---

## 3. Axys Cash Research

### 3.1 Axys Product Scope

| Statement | Classification | Evidence / Notes |
|---|---:|---|
| Axys is positioned by SS&C Advent as portfolio reporting and portfolio accounting software. | Verified | SS&C public Axys product material describes Axys as portfolio reporting and accounting software. |
| Axys automates portfolio reporting and accounting workflows. | Verified | SS&C and other public product pages describe Axys as automating portfolio reporting/accounting. |
| The inspected public Axys product pages do not document the Axys cash data model, IMEX cash objects, REP cash reports, or cash balance file layouts. | Verified | Public pages are product/marketing-level, not technical manuals. |

### 3.2 Axys Cash Transaction Representation

| Statement | Classification | Evidence / Notes |
|---|---:|---|
| Axys transaction translation examples use cash-related transaction codes such as `li`, `lo`, `dp`, `wd`, `by`, `sl`, `dv`, `in`, `ai`, `rc`, and `pd`. | High Confidence | ByAllAccounts Custodial Integrator Axys guide contains default transaction translation tables mapping custodian events to Axys transaction codes. |
| Axys cash movements can be represented with source/destination type and source/destination symbol fields. | High Confidence | ByAllAccounts and WealthTechs examples show Axys/APX-oriented transaction rows with type/symbol fields such as `$pty`, `$ity`, `$pth`, `$cash`, `$income`, `CAUS`, `CASH`, `MMF`, `MARGIN`, and `SHORT`. |
| Axys cash-related transaction examples commonly use `$cash` as a source/destination symbol. | High Confidence | ByAllAccounts default transaction translation table uses `$cash` repeatedly for deposits, withdrawals, buys, sells, fees, transfers, and other cash-related activity. |
| Axys income-related cash postings may use `$income` as a source/destination symbol. | High Confidence | ByAllAccounts translation table maps dividends, interest, income, and accrued-interest cases to `$income`. |
| Axys system currency can be configured in integration tooling. | High Confidence | ByAllAccounts CI parameter `axyscur` is described as the currency code defined as the Axys system currency. |
| Older versions of Axys may use a cash asset-class letter other than `c`. | High Confidence | ByAllAccounts CI parameter `axysaccash` says the default cash asset-class code is `c`, but older Axys versions may use a different letter. |
| Transactions against a currency other than the Axys system currency may require a Mark to Market field value in at least one integration context. | High Confidence | ByAllAccounts CI parameter `defmarkmarket` states that non-system-currency transactions require a Mark to Market value. |
| A `Perf/CW` column exists in the Axys `topost.trn` file in at least one integration context. | High Confidence | ByAllAccounts CI parameter `defperfcw` describes the value to use in the `Perf/CW` column of the Axys `topost.trn` file. |
| The exact native Axys transaction file layout for all versions is not established by the inspected material. | Unknown | The inspected documents show third-party source/target examples, not a complete Axys vendor data dictionary. |

### 3.3 Axys Cash Sweeps

| Statement | Classification | Evidence / Notes |
|---|---:|---|
| In a WealthTechs Axys integration workflow, a cash sweep is defined as a single transaction to move money from one cash account to another, typically to or from cash and a money market fund. | High Confidence | WealthTechs AIA Axys user manual. |
| In that workflow, cash sweeps may be removed from source `TRN` files when transaction code is `WD` or `DP`, source/destination type fields are cash, and source/destination symbols exclude certain special cash-like symbols. | High Confidence | WealthTechs AIA Axys user manual. |
| The inspected WealthTechs Axys manual treats `margin` and `short` cash sweeps as separate optional removal cases. | High Confidence | WealthTechs AIA Axys user manual. |
| The inspected source does not prove that Axys itself automatically removes cash sweeps. | Verified Unknown | The source describes behavior of WealthTechs AIA import tooling, not native Axys accounting behavior. |
| Whether standard Axys reports include or exclude sweep transactions depends on report configuration and is not established by inspected material. | Unknown | Need REP/report examples or vendor documentation. |

### 3.4 Axys Intra-Account Cash Journals

| Statement | Classification | Evidence / Notes |
|---|---:|---|
| In a WealthTechs Axys workflow, an intra-account cash journal is described as a pair of opposite transactions that wipe each other out. | High Confidence | WealthTechs AIA Axys user manual. |
| In that workflow, opposite cash journal pairs include transaction-code pairings such as `dp`/`wd`, `li`/`lo`, `ti`/`to`, `si`/`so`, and `tr`/`ts`. | High Confidence | WealthTechs AIA Axys user manual. |
| The documented removal criteria include matching trade date, account, amount, and quantity, plus cash/security-type criteria and exclusion of special symbols such as `margin`, `short`, `dvwash`, `dvshrt`, `dvlong`, `cashrt`, `calong`, and `income`. | High Confidence | WealthTechs AIA Axys user manual. |
| The inspected source does not prove that Axys itself automatically nets or removes intra-account cash journal pairs. | Verified Unknown | The source describes a third-party import option. |

### 3.5 Axys Cash Reconciliation / Conversion Implications

| Statement | Classification | Evidence / Notes |
|---|---:|---|
| Custodian import and reconciliation workflows can include cash activity. | Verified | SS&C Advent Custodial Data product brief states available data includes positions, transactions, and cash activity. |
| Pre-existing reconciliation issues in a prior portfolio management system or custodian files can affect conversion and reporting workflows. | Medium Confidence | Morningstar Axys conversion guide describes conversion issues and adjustment transactions in a migration context. |
| Adjustment transactions entered during conversion can affect performance reporting for affected accounts in the target system. | Medium Confidence | Morningstar guide discusses effect of adjustment transactions on Morningstar Office performance after conversion, not Axys calculations directly. |
| The inspected Morningstar guide does not provide native Axys cash-balance calculations or Axys reconciliation report names. | Verified Unknown | Conversion-focused source only. |
| Public Axys/APX integration evidence maps bond-security return-of-capital / principal-paydown activity to `pd` with `$pty` / `$cash` destination context. | Medium-High Confidence | The 2026-07-07 `pd` Modified Dietz research reinforces that `pd` should normally increase portfolio cash only when MBS/ABS/amortizing-security paydown context and cash destination evidence are present. |
| Public APX integration evidence maps `SELL / SHORT` to `ss` with `awus / none`, while `BUY / COVER SHORT` maps to `cs` with `$pty / $cash`. | Medium-High for code/cash-context evidence | The 2026-07-07 `ss`/`cs` lifecycle research suggests short-sale proceeds may not behave like unrestricted ordinary sale cash. A synthetic demo may use a disclosed `SHORT`, margin, collateral, or short-proceeds bucket; production treatment remains site-specific until source/destination and holdings evidence prove the cash mechanics. |

---

## 4. APX Cash Research

### 4.1 APX Product Scope

| Statement | Classification | Evidence / Notes |
|---|---:|---|
| APX is positioned by SS&C Advent as an integrated portfolio management and client relationship management solution. | Verified | SS&C APX public product material. |
| APX tracks holdings, transactions, and performance. | Verified | SS&C APX product brief/product page language. |
| The inspected public APX product pages do not document the APX cash data model, database tables, IMEX cash objects, REP cash reports, or cash-balance report calculations. | Verified | Public APX pages are product/marketing-level, not technical manuals. |

### 4.2 APX Cash Transaction Representation

| Statement | Classification | Evidence / Notes |
|---|---:|---|
| WealthTechs APX examples use transaction rows beginning with `ACCTX`. | High Confidence | WealthTechs AIA APX user manual examples. |
| WealthTechs APX examples show cash/accounting fields using patterns such as `CAUS`, `CASH`, `MMF`, `MARGIN`, and `SHORT`. | High Confidence | WealthTechs AIA APX user manual examples. |
| WealthTechs APX examples use transaction codes including `DP`, `WD`, `LI`, and `LO` in cash/journal examples. | High Confidence | WealthTechs AIA APX user manual. |
| APX broker rep field values may be defined in the `.cli` file for each portfolio in at least one AIA workflow. | High Confidence | WealthTechs APX manual states `$brok` is defined in the `.cli` file in APX for each portfolio typically. |
| The inspected material does not establish whether `.cli` files are the authoritative native APX transaction storage mechanism or only an import/configuration/interface artifact. | Unknown | Need APX vendor documentation or production examples. |

### 4.3 APX Cash Sweeps

| Statement | Classification | Evidence / Notes |
|---|---:|---|
| In a WealthTechs APX integration workflow, a cash sweep is defined as a single transaction to move money from one cash account to another, typically to or from cash and a money market fund. | High Confidence | WealthTechs AIA APX user manual. |
| In that workflow, cash sweeps may be removed when source `TRN` records have `WD` or `DP` transaction code, cash source/destination type fields, and symbols excluding special cash-like/margin/short/income placeholders. | High Confidence | WealthTechs AIA APX user manual. |
| The APX manual includes separate examples for margin and short sweeps using `MARGIN` and `SHORT`. | High Confidence | WealthTechs AIA APX user manual. |
| The inspected source does not prove that native APX automatically removes or suppresses cash sweeps. | Verified Unknown | Source is third-party integration manual. |

### 4.4 APX Intra-Account Cash Journals

| Statement | Classification | Evidence / Notes |
|---|---:|---|
| In a WealthTechs APX workflow, intra-account cash journals are opposite transaction pairs that wipe each other out. | High Confidence | WealthTechs AIA APX manual. |
| The APX AIA workflow uses pair logic similar to the Axys workflow for `dp`/`wd`, `li`/`lo`, `ti`/`to`, `si`/`so`, and `tr`/`ts` pairs. | High Confidence | WealthTechs AIA APX manual. |
| The inspected source does not prove native APX automatically nets these cash journals. | Verified Unknown | Source describes third-party import-cleanup behavior. |

---

## 5. IMEX Research

### 5.1 What Is Supported by Inspected Sources

| Statement | Classification | Evidence / Notes |
|---|---:|---|
| IMEX is relevant to moving data in and out of Axys/APX. | High Confidence | Public AdventGuru article describes IMEX as a tool for moving data in and out of Axys and references Axys/APX data import/export workflows. |
| Cash-related data needed for external products may be obtainable through transaction, holding/position, and cash activity feeds, but exact IMEX objects are not confirmed by inspected material. | Medium Confidence | Inference from custodian data and integration manuals; not an IMEX manual. |
| The inspected material does not verify an IMEX object named `cash`, `cashbal`, `cash_balance`, or equivalent. | Unknown | Need IMEX spec/export samples. |
| The inspected material does not verify exact cash-related IMEX column names for Axys or APX. | Unknown | Need IMEX spec/export samples. |
| The inspected material does not verify whether Axys/APX IMEX can export beginning cash, ending cash, settled cash, trade-date cash, or multi-currency cash balances as distinct objects. | Unknown | Need IMEX documentation or sample exports. |

### 5.2 Candidate IMEX Research Questions

These are not facts. They are targeted questions to resolve with IMEX documentation or sample exports.

| Question | Current Status |
|---|---|
| Does Axys IMEX provide a standard transaction export object containing cash-impacting transaction fields? | Unknown |
| Does APX IMEX provide a standard transaction export object containing cash-impacting transaction fields? | Unknown |
| Does Axys IMEX provide a standard positions/holdings export where cash appears as a security/asset class row? | Unknown |
| Does APX IMEX provide a standard positions/holdings export where cash appears as a security/asset class row? | Unknown |
| Does IMEX distinguish custodian cash, cash equivalents, money market sweep vehicles, margin cash, short cash, income cash, dividend wash, and currency cash? | Unknown |
| Are cash balances exportable by portfolio, date, currency, custodian, lot location, account, sleeve, strategy, or portfolio group? | Unknown |
| Which fields identify cash security versus cash account versus currency? | Unknown |
| Are cash records stored or calculated from transactions at export time? | Unknown |
| Are cash records trade-date, settlement-date, or configurable? | Unknown |

---

## 6. REP Research

### 6.1 What Is Supported by Inspected Sources

| Statement | Classification | Evidence / Notes |
|---|---:|---|
| Axys has an extensive library of pre-defined reports and supports report customization. | Verified | Public Axys product material. |
| APX has extensive reporting capabilities, but the inspected public material does not identify cash-specific report names. | Verified for general reporting; Unknown for cash report names | Public APX pages discuss reporting broadly. |
| The inspected material does not verify REP report names for cash balances, cash reconciliation, cash transactions, cash projections, income cash, or multi-currency cash. | Unknown | Need REP report library, report definitions, or production report samples. |
| The inspected material does not verify whether standard REP reports present cash as a security holding row, separate subtotal, account-level balance, or currency-level balance. | Unknown | Need actual report output or report definitions. |

### 6.2 Candidate REP Research Questions

| Question | Current Status |
|---|---|
| Which Axys standard REP reports display cash balances? | Unknown |
| Which APX standard reports display cash balances? | Unknown |
| Which REP reports show cash activity versus cash balance? | Unknown |
| Do REP cash reports include unsettled trades? | Unknown |
| Do REP cash reports support settlement-date versus trade-date cash? | Unknown |
| Do REP cash reports separate principal cash, income cash, margin cash, short cash, and sweep vehicles? | Unknown |
| Are cash totals report-date values, period averages, or derived from holdings snapshots? | Unknown |
| Can REP reports be parameterized to include/exclude money market sweep vehicles from cash? | Unknown |
| Which report fields are available for cash currency, exchange rate, local cash, and base cash? | Unknown |

---

## 7. Cash Data Model Research

### 7.1 Observed Cash-Like Concepts

The following concepts appear in inspected sources. They should be verified against native Axys/APX documentation before being used as canonical repository terminology.

| Concept / Token | Appears In | Meaning Supported by Source | Axys | APX | Confidence |
|---|---|---|---|---|---:|
| `$cash` | ByAllAccounts Axys translation guide | Special source/destination symbol used for cash in transaction translations. | Yes | Not directly from this source | High Confidence for Axys CI |
| `$income` | ByAllAccounts Axys translation guide | Special source/destination symbol used for income-related transaction translations. | Yes | Not directly from this source | High Confidence for Axys CI |
| `$pty` | ByAllAccounts Axys translation guide | Source/destination type in many cash-impacting transaction translations. Exact native meaning not expanded in source. | Yes | Not directly from this source | High Confidence as observed token; Unknown meaning |
| `$ity` | ByAllAccounts Axys translation guide | Source/destination type in income/accrued-interest translations. Exact native meaning not expanded in source. | Yes | Not directly from this source | High Confidence as observed token; Unknown meaning |
| `$pth` | ByAllAccounts Axys translation guide | Source/destination type used for margin interest. Exact native meaning not expanded in source. | Yes | Not directly from this source | High Confidence as observed token; Unknown meaning |
| `CAUS` | WealthTechs Axys/APX AIA examples | Cash/security type token in examples. Exact expansion not confirmed. | Yes | Yes | High Confidence as observed token; Unknown expansion |
| `CASH` | WealthTechs Axys/APX AIA examples | Symbol/token used in cash examples. | Yes | Yes | High Confidence |
| `MMF` | WealthTechs Axys/APX AIA examples | Money market fund/sweep vehicle symbol in examples. | Yes | Yes | High Confidence |
| `MARGIN` | WealthTechs Axys/APX AIA examples | Margin cash/sweep symbol in examples. | Yes | Yes | High Confidence |
| `SHORT` | WealthTechs Axys/APX AIA examples | Short cash/sweep symbol in examples. | Yes | Yes | High Confidence |
| `dvwash` | ByAllAccounts / WealthTechs examples | Dividend wash special symbol excluded from sweep-removal logic. | Yes | Yes in AIA docs | High Confidence as observed token; Unknown native definition |
| `dvshrt` | WealthTechs examples | Special symbol excluded from sweep-removal logic. | Yes | Yes | High Confidence as observed token; Unknown native definition |
| `dvlong` | WealthTechs examples | Special symbol excluded from sweep-removal logic. | Yes | Yes | High Confidence as observed token; Unknown native definition |
| `cashrt` | WealthTechs examples | Special symbol excluded from sweep-removal logic. | Yes | Yes | High Confidence as observed token; Unknown native definition |
| `calong` | WealthTechs examples | Special symbol excluded from sweep-removal logic. | Yes | Yes | High Confidence as observed token; Unknown native definition |
| `income` | WealthTechs examples | Special symbol excluded from sweep-removal logic. | Yes | Yes | High Confidence as observed token; Unknown native definition |
| Cash asset-class code `c` | ByAllAccounts Axys guide | Default cash asset-class letter used by CI for Axys, configurable because older versions may differ. | Yes | Not established | High Confidence for Axys CI |
| System currency | ByAllAccounts Axys guide | Axys system currency configured in integration parameter `axyscur`. | Yes | Not established | High Confidence for Axys CI |

### 7.2 Cash Field Dictionary Draft

This table is a research-stage field/token dictionary. It is not a canonical Axys/APX data dictionary.

| Field / Token | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---:|
| Transaction code | Code such as `li`, `lo`, `dp`, `wd`, `by`, `sl`, `dv`, `in`, `ai`, `rc`, `pd` used in cash-impacting translations. | Observed | Observed for some codes in AIA examples | Unknown | Unknown | High Confidence |
| `ACCTX` | Prefix/object marker shown in WealthTechs AIA transaction examples. | Observed | Observed | Unknown | N/A | High Confidence as observed; Unknown native status |
| Trade date | Used as matching criterion for removal of intra-account cash journals. | Observed | Observed | Unknown | Unknown | High Confidence |
| Account | Used as matching criterion for removal of intra-account cash journals. | Observed | Observed | Unknown | Unknown | High Confidence |
| Amount | Used in transaction examples and as matching criterion for journal removal. | Observed | Observed | Unknown | Unknown | High Confidence |
| Quantity | Used in transaction examples and as matching criterion for journal removal. | Observed | Observed | Unknown | Unknown | High Confidence |
| Source/Destination Type | Cash/income/security type field in integration examples. | Observed | Observed | Unknown | Unknown | High Confidence |
| Source/Destination Symbol | Cash/income/security symbol field in integration examples. | Observed | Observed | Unknown | Unknown | High Confidence |
| Security Type | Used in sweep-removal criteria in WealthTechs manuals. | Observed | Observed | Unknown | Unknown | High Confidence |
| Security Symbol | Used in sweep-removal criteria and examples. | Observed | Observed | Unknown | Unknown | High Confidence |
| Mark to Market | Field required for non-system-currency transactions in ByAllAccounts Axys integration context. | Observed | Unknown | Unknown | Unknown | High Confidence for Axys CI |
| Perf/CW | Column in Axys `topost.trn` file according to ByAllAccounts CI. | Observed | Unknown | Unknown | Unknown | High Confidence for Axys CI |
| Broker rep field | Field populated by `$brok` in AIA manuals; APX manual says `.cli` defines it per portfolio typically. | Observed | Observed | Unknown | Unknown | High Confidence |
| Cash balance | Ending/current cash amount by account/currency/date. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Settled cash | Cash available after settlement. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Trade-date cash | Cash including trade-date activity regardless of settlement. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Income cash | Separate bucket for income-related cash. | Inferred from `$income` | Inferred from AIA special symbol list | Unknown | Unknown | Medium Confidence |
| Margin cash | Separate cash-like bucket or symbol. | Observed token | Observed token | Unknown | Unknown | High Confidence as observed token |
| Short cash | Separate cash-like bucket or symbol. | Observed token | Observed token | Unknown | Unknown | High Confidence as observed token |
| Sweep vehicle | Money market/cash-equivalent vehicle used to move funds to/from cash. | Observed in examples | Observed in examples | Unknown | Unknown | High Confidence |

---

## 8. Cash Transaction Code Research

### 8.1 Codes Observed in Axys-Oriented Translation Examples

The following codes appear in the inspected ByAllAccounts Axys transaction translation table. Meanings are based only on the transaction rows shown in that table and should be verified against Axys transaction-code documentation.

| Code | Observed Context | Cash Impact | Confidence |
|---|---|---:|---:|
| `li` | Deposits, credits, direct deposits, transfers in, ATM positive, income positive. | Likely increases cash or moves value in. | High Confidence as observed |
| `lo` | Withdrawals, checks, debits, payments, transfers out, ATM negative. | Likely decreases cash or moves value out. | High Confidence as observed |
| `dp` | Cash security buy, fees, service charges, investment expense, non-cash security debit/tax. | Cash-impacting debit/payment-style code in examples. | High Confidence as observed |
| `wd` | Cash-security sell and withdrawal-like cash-security movement examples. | Cash-impacting cash-security/sell-style code in examples; not automatically client withdrawal. | High Confidence as observed |
| `by` | Buy; reinvested dividend buy leg. | Security purchase funded by cash. | High Confidence as observed |
| `sl` | Sell; closure positive leg. | Security sale producing cash. | High Confidence as observed |
| `dv` | Dividend and reinvested dividend leg. | Income/cash-related. | High Confidence as observed |
| `in` | Income, interest, dividends on cash securities. | Income/cash-related. | High Confidence as observed |
| `ai` | Interest negative; margin interest. | Negative-interest or margin-interest cash effect. | High Confidence as observed |
| `rc` | Return of capital. | Maps to portfolio cash context (`$pty` / `$cash`) in ByAllAccounts Axys/APX translation evidence. | Confirmed in integration mapping evidence |
| `pd` | Return of capital for bond security / principal paydown. | Maps to portfolio cash context in ByAllAccounts Axys/APX translation evidence. | Confirmed in integration mapping evidence |
| `sa` | Sale accrued interest / sell-side accrued interest. | Income/cash-related fixed-income trade adjunct. | High Confidence as observed |
| `cs` | Cover short / closure negative leg. | Cash/security cash-impacting; integration mapping points to `$pty / $cash`, but exact sign mechanics remain site-specific. | Medium-High for code meaning |
| `ss` | Short sale. | Cash/security cash-impacting; integration mapping points to `awus / none`, so unrestricted cash treatment is not safe to assume. | Medium-High for code meaning |

### 8.2 Transaction-Code Caveats

| Caveat | Classification | Notes |
|---|---:|---|
| Uppercase transaction codes may represent reversal/deletion transactions in an Axys import context. | High Confidence | ByAllAccounts CI guide states reversal transactions are translated by converting original transaction type code to uppercase; examples include `by` to `BY`. |
| It is unknown whether every uppercase code is always a delete/reversal marker in all Axys/APX contexts. | Unknown | Need Axys/APX vendor transaction import documentation. |
| Code semantics may be site-configured or version-sensitive. | Medium Confidence | ByAllAccounts notes configurable asset-class letters and version differences; transaction codes themselves require vendor confirmation. |

---

## 9. Processing Behavior Research

### 9.1 Cash Sweeps and Sweep Vehicles

| Behavior | Axys | APX | Confidence |
|---|---|---|---:|
| Cash sweeps may be represented as `DP` or `WD` transactions between cash and a money-market/sweep vehicle. | Observed in AIA Axys examples | Observed in AIA APX examples | High Confidence |
| Third-party import tooling may remove cash sweep transactions to avoid tracking separate cash and money-market balances in the portfolio accounting system. | Observed in AIA Axys manual | Observed in AIA APX manual | High Confidence |
| Native Axys/APX sweep behavior, including whether standard reports classify sweep vehicles as cash equivalents, is not established. | Unknown | Unknown | Unknown |

### 9.2 Journal Netting / Removal

| Behavior | Axys | APX | Confidence |
|---|---|---|---:|
| Third-party tooling can identify and remove offsetting intra-account cash journal pairs. | Observed | Observed | High Confidence |
| Matching criteria include trade date, account, amount, quantity, transaction-code pair, type/security criteria, and symbol exclusions. | Observed | Observed | High Confidence |
| Native Axys/APX automated netting is not established. | Unknown | Unknown | Unknown |

### 9.3 Multi-Currency Cash

| Behavior | Axys | APX | Confidence |
|---|---|---|---:|
| Axys integration tooling references a configured system currency. | Observed | Not observed | High Confidence for Axys CI |
| Non-system-currency transactions may require a Mark to Market field value in an Axys `topost.trn` context. | Observed | Not observed | High Confidence for Axys CI |
| Native Axys/APX cash balance presentation by currency is not established. | Unknown | Unknown | Unknown |
| Exchange-rate source, valuation date, and local/base cash fields are not established. | Unknown | Unknown | Unknown |

### 9.4 Performance and Cash

| Behavior | Axys | APX | Confidence |
|---|---|---|---:|
| A `Perf/CW` column appears in the Axys `topost.trn` integration context. | Observed | Not observed | High Confidence for Axys CI |
| Whether cash transactions are treated as contributions/withdrawals, income, expenses, or internal movements depends on transaction coding. | Medium Confidence | Medium Confidence | Medium Confidence |
| Exact Axys/APX performance treatment of each cash code is not established by inspected sources. | Unknown | Unknown | Unknown |

---

## 10. Examples

### 10.1 Cash Sweep Example

Observed in WealthTechs manuals:

```text
DP,CAUS,CASH,CAUS,MMF
```

Research interpretation:

| Element | Possible Meaning | Confidence |
|---|---|---:|
| `DP` | Deposit/payment-style transaction code used in sweep-removal examples. | High Confidence as observed |
| `CAUS` | Cash/security type token. Exact expansion unknown. | High Confidence as observed; Unknown expansion |
| `CASH` | Cash symbol/token. | High Confidence |
| `MMF` | Money market fund / sweep vehicle token in example. | High Confidence |

Do not infer full native Axys/APX file layout from this line alone.

### 10.2 APX/Axys AIA Transaction Row Examples

Observed examples include:

```text
ACCTX,010117,LI,100,100,CAUS,MMF,CAUS,CASH
ACCTX,010117,LO,100,100,CAUS,MMF,CAUS,CASH
ACCTX,010117,DP,100,100,CAUS,MMF,CAUS,CASH
ACCTX,010117,WD,100,100,CAUS,MMF,CAUS,CASH
```

Research interpretation:

| Segment | Meaning Status |
|---|---|
| `ACCTX` | Observed transaction-row prefix/object marker in AIA examples. Native status unknown. |
| `010117` | Date-like token in MMDDYY format in examples. Exact date field definition unknown. |
| `LI`, `LO`, `DP`, `WD` | Transaction codes in examples. |
| `100`, `100` | Amount/quantity fields are implied by AIA text, but exact order should be verified. |
| `CAUS,MMF,CAUS,CASH` | Type/symbol pairs in examples. |

### 10.3 Intra-Account Cash Journal Pair Example

Observed in WealthTechs AIA Axys manual:

```text
ACCTX,010117,DP,100,100,CAUS,MMF,CAUS,CASH
ACCTX,010117,WD,100,100,CAUS,MMF,CAUS,CASH
```

Research interpretation:

| Item | Notes |
|---|---|
| Pair logic | Third-party AIA option treats certain pairs as offsetting cash journals if criteria match. |
| Native Axys/APX behavior | Unknown. |
| Report impact | Unknown without report samples. |

---

## 11. Known Issues / Quirks

| Quirk | Axys | APX | Classification | Practical Implication |
|---|---|---|---:|---|
| Cash sweeps may be represented as single transactions moving money between cash and money-market funds. | Observed | Observed | High Confidence | External audit/extract tools should decide whether sweep vehicles are cash, holdings, or both. |
| Third-party tooling may remove cash sweeps to avoid tracking separate cash/MMF balances. | Observed | Observed | High Confidence | Imported cash balances may differ depending on whether sweeps are retained. |
| Margin and short sweeps may need separate handling. | Observed | Observed | High Confidence | Do not assume all cash-like symbols belong to unrestricted long cash. |
| Dividend-wash and income-like symbols are excluded from sweep-removal logic. | Observed | Observed | High Confidence | External tools should treat `dvwash`, `dvshrt`, `dvlong`, `cashrt`, `calong`, and `income` cautiously. |
| Older Axys versions may use a cash asset-class code other than `c`. | Observed | Unknown | High Confidence for Axys CI | Integrations should not hard-code cash asset class without client validation. |
| Non-system-currency Axys transactions may require Mark to Market values in import files. | Observed | Unknown | High Confidence for Axys CI | Multi-currency cash imports require special validation. |
| Uppercase transaction codes may be used as reversal/delete indicators in Axys import context. | Observed | Unknown | High Confidence for Axys CI | Extract/replay tools must distinguish correction records from new activity. |
| Adjustment transactions used for reconciliation/conversion can affect downstream performance in target systems. | Medium Confidence | Medium Confidence | Medium Confidence | Cash-balancing entries can alter contribution/withdrawal treatment if not classified correctly. |
| Cash balance may be derived rather than stored, depending on report/export path. | Unknown | Unknown | Unknown | Need client sample data or vendor documentation. |
| Settled versus trade-date cash treatment is not established. | Unknown | Unknown | Unknown | This is a high-priority gap for accounting/reporting. |

---

## 12. Version Differences

| Version Difference | Axys | APX | Classification | Evidence / Notes |
|---|---|---|---:|---|
| Cash asset-class letter may differ in older Axys versions. | Yes | Unknown | High Confidence | ByAllAccounts CI guide explicitly warns that older Axys versions may use a cash asset-class letter different from `c`. |
| Bond asset-class letter may differ in older Axys versions. | Yes | Unknown | High Confidence | Relevant because transaction translation depends on asset-class configuration, but not cash-specific. |
| APX-specific version differences for cash handling are not documented in inspected material. | N/A | Unknown | Unknown | Need APX release notes/manuals. |
| Axys/APX IMEX version differences for cash objects/fields are not documented in inspected material. | Unknown | Unknown | Unknown | Need IMEX manuals by version. |
| REP version differences for cash report fields are not documented in inspected material. | Unknown | Unknown | Unknown | Need report-definition files or manuals. |

---

## 13. Minimum Additional Source Material Needed

To convert this research into a strong technical chapter, the following materials would materially improve completeness:

| Needed Material | Why Needed |
|---|---|
| Axys IMEX manual or sample IMEX exports containing transactions, holdings, and cash balances. | To verify object names, file names, field names, field order, data types, and cash representation. |
| APX IMEX manual or sample IMEX exports containing transactions, holdings, and cash balances. | Same as above for APX. |
| Axys REP report definitions or sample reports showing cash. | To document report names, field names, filters, and cash subtotals. |
| APX report definitions or sample reports showing cash. | To document APX report behavior and differences from Axys. |
| Security master examples containing cash securities, money market funds, margin cash, and short cash. | To document how cash-like instruments are classified. |
| Portfolio transaction samples with cash deposits, withdrawals, dividends, interest, fees, FX, sweeps, margin, shorts, and reversals. | To verify cash transaction code behavior. |
| Multi-currency portfolio examples. | To document local/base cash, FX rates, and Mark to Market behavior. |
| Client implementation notes or consultant documentation. | To document production quirks and site-specific configuration. |
| Axys/APX version-specific release notes. | To document version differences. |

---

## 14. Chapter Framing Implications

The reader-facing cash chapter should be structured around:

1. Overview
2. Axys Cash
3. APX Cash
4. Cash Transactions
5. Cash Balances
6. Cash-Like Securities and Sweep Vehicles
7. Multi-Currency Cash
8. IMEX
9. REP
10. Field Dictionary
11. Examples
12. Known Issues / Quirks
13. Unknowns
14. References

The chapter should not claim:

- exact IMEX cash object names,
- exact REP cash report names,
- native cash balance table names,
- full transaction-code semantics,
- settled/trade-date behavior,
- native sweep-netting behavior,
- or APX database-field names

until those are supported by direct source material.

---

## 15. References

### Governing Repository Reference

1. `AXYS_APX_REFERENCE_BLUEPRINT.md`, Version 2.0. Defines editorial standards, confidence labels, repository structure, and chapter template.

### Public / External References Inspected

2. SS&C Advent Axys public product page. Used only for broad product-scope statements: portfolio reporting/accounting and reporting capabilities.
3. SS&C Advent Portfolio Exchange public product page/product brief. Used only for broad product-scope statements: integrated portfolio/relationship management, holdings, transactions, performance, reporting.
4. WealthTechs, **AIA User Manual for Axys Users**, PDF. Used for Axys-oriented transaction examples, cash sweep and intra-account cash journal logic, `.cli`/broker rep references, and special cash-like symbols.
5. WealthTechs, **AIA User Manual for APX Users**, PDF. Used for APX-oriented transaction examples, cash sweep and intra-account cash journal logic, `.cli`/broker rep references, and special cash-like symbols.
6. ByAllAccounts, **Custodial Integrator User Guide for Axys**, PDF. Used for Axys transaction translation table, `$cash`, `$income`, transaction codes, reversal behavior, system currency, cash asset-class configuration, Mark to Market, and `Perf/CW` references.
7. Morningstar, **Converting Your Advent Axys Database into Morningstar Office**, PDF. Used only for migration/reconciliation caveats; not treated as an Axys technical data dictionary.
8. SS&C Advent Custodial Data product brief. Used only for the broad statement that custodian data workflows include positions, transactions, and cash activity.

---

## 16. Open Unknowns Register

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

---

## 17. Research Status

This research is suitable as a starting file for
`docs/axys_apx/evidence/Research_07_Cash.md`.

It is not sufficient by itself to write a fully authoritative
`../reference/Chapter_07_Cash.md` chapter with complete IMEX and REP field
dictionaries. The strongest next evidence would be real Axys/APX IMEX exports,
REP report samples, and vendor/manual excerpts specific to cash.

## Deep Research Update Incorporated 2026-07-02

The July 2026 addendum strengthens cash-relevant integration evidence but does
not identify a native cash-balance schema. ByAllAccounts CI supports an Axys
data flow where WebPortfolio data is merged with Axys security information and
produces transaction, position, and price files imported through Axys
Import/Export; cash activity is therefore best treated as part of
transaction/position/price workflows unless a separate cash-balance object is
provided.

Additional verified integration artifacts include `topost.trn`, `ptopost.trn`,
`.pos` replacement files, `$pathcli`, `$pathinf`, `$pathpri`, `$pathlog`,
`imex32.exe`, `pospos32.exe`, and the Axys `sipos30` position-reconciliation
report lead. WealthTechs AIA reinforces cash-sweep and intra-account cash
journal cleanup as integration-tool behavior, including pair rules such as
`dp>wd`, `li>lo`, `ti>to`, `si>so`, and `tr>ts`, excluded symbols such as
`margin`, `short`, `dvwash`, `dvshrt`, `dvlong`, `cashrt`, `calong`, and
`income`, and APX Account Skip Logic that can bypass global cleanup rules for
specific accounts.

APX multi-currency support remains product-level evidence only. The APX
`Income Projection` report is a cash-inflow report lead, not a verified
cash-balance report. Native cash-balance storage, cash IMEX objects, cash report
field dictionaries, settled/trade-date cash behavior, multi-currency cash
fields, and native definitions of `CAUS`, `caxx`, `awxx`, `$pty`, `$ity`,
`$pth`, `cashrt`, and `calong` remain Unknown.
