# Research Notes: Performance

**Repository chapter target:** `../reference/Chapter_10_Performance.md`  
**Research file:** `docs/axys-apx-reference/evidence/Research_10_Performance.md`  
**Prepared under:** `AXYS_APX_REFERENCE_BLUEPRINT.md`, Version 2.0  
**Purpose:** Collect factual, implementation-oriented research for documenting Axys and APX performance behavior.

---

## 1. Scope and Evidence Standard

This file is research material for the reader-facing performance chapter. It is not a performance-measurement textbook and does not attempt to explain portfolio performance theory except where needed to identify system behavior.

The governing repository standard requires factual, implementation-oriented knowledge about Axys and APX, including architecture, accounting data, IMEX, REP, reports, file layouts, data fields, processing behavior, version differences, and quirks. It also requires important technical statements to be classified as **Verified**, **High Confidence**, **Medium Confidence**, or **Unknown**.

### Confidence meanings used in this file

| Confidence | Meaning in this research file |
|---|---|
| Verified | Supported by vendor/public documentation, supplied repository source material, direct sample exports/reports, or explicit production evidence. |
| High Confidence | Strongly consistent with public vendor/product material or common Axys/APX implementation practice, but exact local behavior still requires validation. |
| Medium Confidence | Plausible and operationally useful, but not enough evidence to treat as system fact. |
| Unknown | Not verified. Must not be used as a Chapter 10 fact without additional source material. |

### Primary source limitations

Only the blueprint was supplied as local source material. Public web research found high-level vendor statements about Axys and APX, but not enough public detail to verify internal performance tables, exact IMEX object names, exact REP source files, exact field names, or calculation storage/recalculation behavior.

Therefore, this research file intentionally preserves many Unknowns.

---

## 2. Source Inventory

| Source | Type | Relevant content | Use in this research | Confidence |
|---|---|---|---|---|
| `AXYS_APX_REFERENCE_BLUEPRINT.md`, Version 2.0 | Supplied repository specification | Defines repository purpose, chapter/research structure, standards, confidence labels, and preference for field dictionaries, IMEX exports, REP reports, examples, version differences, and quirks. | Governing specification for this research file. | Verified |
| SS&C Advent Axys product page | Public vendor/product page | Axys is positioned as portfolio reporting/accounting software with predefined reports, report customization, and portfolio performance visibility. | Supports only high-level Axys capability statements. | Verified for public marketing claims; not verified for implementation details |
| SS&C Advent APX product page / product brief | Public vendor/product page/brief | APX is positioned as integrated portfolio management, accounting, reporting, CRM, and performance analytics. | Supports only high-level APX capability statements. | Verified for public marketing claims; not verified for implementation details |
| SS&C/APX index data product brief | Public vendor PDF snippet | States APX has integrated performance analytics and benchmark/index data support for performance analytics. | Supports high-level APX performance analytics and benchmark dependency. | Verified for public marketing claims; not verified for implementation details |
| SS&C APX reports guide search result snippet | Public vendor PDF search snippet | Mentions an APX report that compares portfolio performance to benchmark over a time period. | Useful lead only; full PDF could not be inspected during this research pass. | Medium Confidence |
| AdventGuru IMEX/reporting article snippets | Consultant/public blog snippets | Notes report export options, Report Writer Pro, Replang modifications, CSV/text outputs, and third-party ETL such as xPort. | Useful lead for REP/report extraction options; not vendor-confirmed in this file. | Medium Confidence |
| User-supplied IMEX/REP samples | Not supplied for this chapter | Would be needed to verify performance object names, fields, and local export behavior. | Not available. | Unknown |

---

## 3. Executive Findings

| Finding | Axys | APX | Confidence | Notes |
|---|---:|---:|---|---|
| Both Axys and APX are used for portfolio accounting/reporting workflows that include performance reporting. | Yes | Yes | Verified for high-level capability | Public vendor material supports performance/reporting capability but not internal calculation details. |
| Axys has predefined reports and customizable reporting. | Yes | Not Axys-specific | Verified for high-level capability | Vendor material supports an extensive library of predefined reports and customization for Axys. |
| APX includes performance analytics as part of its integrated platform. | Not APX-specific | Yes | Verified for high-level capability | Public APX product material supports this. |
| APX performance analytics can use benchmarks/index data. | Not APX-specific | Yes | Verified for high-level capability | Public APX index data material supports benchmark data for APX performance analytics. |
| Exact Axys performance storage files/tables are not verified from supplied material. | Unknown | N/A | Unknown | Requires Axys sample files, IMEX definitions, report source, or vendor docs. |
| Exact APX SQL tables/views for stored or calculated performance are not verified from supplied material. | N/A | Unknown | Unknown | Requires APX schema docs, SQL view definitions, or exports. |
| Exact IMEX object names for portfolio performance and security performance are not verified from supplied material. | Unknown | Unknown | Unknown | Candidate terms such as `portperf` and `secperf` require source validation. |
| Exact REP report file names for portfolio/security performance are not verified from supplied material. | Unknown | Unknown | Unknown | Requires REP folder/report inventory or vendor report guide. |
| Whether reports use stored monthly returns or recalculate over requested date ranges is not verified. | Unknown | Unknown | Unknown | This is a critical open question for Chapter 10. |

---

## 4. Axys Performance Research

### 4.1 Verified / high-level Axys behavior

| Topic | Statement | Confidence | Evidence / Notes |
|---|---|---|---|
| Product category | Axys is portfolio reporting and portfolio accounting software. | Verified | Public SS&C Advent Axys product material. |
| Reporting | Axys provides an extensive library of predefined reports and supports report customization. | Verified | Public SS&C Advent Axys product material. |
| Performance visibility | Axys marketing material states that it provides a clear picture of portfolios and their performance. | Verified for marketing claim | Does not verify calculation method or storage model. |
| User base | Axys is used by asset managers, wealth managers, and family offices. | Verified for marketing claim | Public vendor material. |
| Database architecture | Axys is described publicly as a turnkey solution and, in third-party material, as on-premise/proprietary-database oriented. | Medium Confidence | Requires Axys architecture chapter/source confirmation before Chapter 10 use. |

### 4.2 Axys performance processing areas to verify

| Area | Research question | Current status | Evidence needed |
|---|---|---|---|
| Portfolio return calculation | Does Axys calculate portfolio-level TWR, IRR, modified Dietz, or multiple return types depending on report/settings? | Unknown | Vendor performance manual, report documentation, REP source, or controlled examples. |
| Stored vs calculated returns | Does Axys store period performance values, calculate on demand, or both? | Unknown | IMEX performance exports, Axys data files, performance report source, or support documentation. |
| Monthly linking | When a report spans multiple months, does Axys geometrically link stored monthly returns or recalculate from beginning/end market values and flows? | Unknown | Test portfolio, monthly performance files, report output comparison. |
| Security contribution | Does Axys security performance foot to portfolio performance by `weight * return`, by contribution fields, or by report-specific logic? | Unknown | Security performance report samples and IMEX exports. |
| Classification performance | Does Axys calculate classification-level performance directly, aggregate security performance, or use report-level classification groupings? | Unknown | Classification report output and REP source. |
| Composite performance | Does Axys composite reporting use stored account-level returns, recalculated account returns, asset-weighted aggregation, or a separate composite module? | Unknown | Composite report documentation and GIPS/composite samples. |
| Cash treatment | How does Axys include cash/security cash flows in performance? | Unknown | Cash/performance test examples, reports, and transaction coding. |
| Accrued income | Does performance use trade-date or settlement-date positions; does it include accrued income in beginning/end market values? | Unknown | Report options, income/accrual documentation, sample outputs. |
| Multi-currency | How are local/base returns, FX effects, and currency contribution represented? | Unknown | Multi-currency performance reports, settings, and sample exports. |
| Benchmarks | How are benchmarks stored and selected in Axys reports? | Unknown | Benchmark chapter/source material and performance reports. |

### 4.3 Candidate Axys performance data sources — unverified

The following are candidate data sources or terms that may be relevant to Axys performance research. They are **not verified** by the supplied blueprint or public product material and must be confirmed before use in Chapter 10.

| Candidate object / file / report term | Possible meaning | Confidence | Required validation |
|---|---|---|---|
| `portperf` | Candidate IMEX/export object for portfolio performance. | Unknown | Actual IMEX catalog/export sample. |
| `secperf` | Candidate IMEX/export object for security performance. | Unknown | Actual IMEX catalog/export sample. |
| Performance Appraisal report | Candidate Axys report family for period performance. | Unknown | REP/report inventory and output sample. |
| Security Performance report | Candidate report family for security-level return/contribution. | Unknown | REP/report inventory and output sample. |
| Portfolio Performance report | Candidate report family for account-level return. | Unknown | REP/report inventory and output sample. |
| Composite Performance report | Candidate report family for composite/GIPS outputs. | Unknown | Report inventory and composite module source. |
| Classification Performance report | Candidate classification-level report. | Unknown | Report inventory and sample output. |

---

## 5. APX Performance Research

### 5.1 Verified / high-level APX behavior

| Topic | Statement | Confidence | Evidence / Notes |
|---|---|---|---|
| Product category | APX is an integrated portfolio management, accounting, reporting, and client relationship management platform. | Verified for public marketing claim | Public SS&C Advent APX product material. |
| Performance analytics | APX public product material states that APX includes performance analytics. | Verified for public marketing claim | Public APX product material. |
| Performance analytics purpose | Public APX index data material states performance analytics help users understand sources of portfolio performance at sector or security level. | Verified for public marketing claim | Does not verify method, field names, or report implementation. |
| Benchmark support | Public APX index data material states that benchmark/index data can be obtained and maintained for use in APX performance analytics. | Verified for public marketing claim | Does not verify benchmark table/schema. |
| Reporting framework | APX public product brief states the APX reporting framework is built around Microsoft Reporting Services. | Verified for public marketing claim | Useful for REP/reporting chapter cross-reference. |
| Data architecture | APX public/client material describes APX as SQL-based/open database architecture in at least one client case study. | Medium Confidence | Needs vendor technical documentation or APX architecture chapter confirmation before Chapter 10 use. |
| Performance measurement app | SS&C 2020 product update describes APX as a core accounting, reporting, and performance measurement application. | Verified for public marketing claim | Does not verify internal processing details. |

### 5.2 APX performance processing areas to verify

| Area | Research question | Current status | Evidence needed |
|---|---|---|---|
| Storage model | Does APX persist performance results in SQL tables, calculate them dynamically, cache them, or use a mix? | Unknown | APX schema docs, stored procedure/view inventory, controlled report tests. |
| Performance engine | Which APX services/procedures calculate returns and contribution? | Unknown | Technical documentation or database/procedure names from production instance. |
| Portfolio vs security performance | Are account-level and security-level results stored in separate objects/tables/views? | Unknown | APX database dictionary or export samples. |
| Sector/classification performance | Does APX store sector-level performance or derive it from security-level performance at report time? | Unknown | APX performance analytics documentation and report definitions. |
| Attribution | Which attribution methodologies are available in APX, and in which versions/modules? | Unknown | APX performance analytics/attribution guide. Public material verifies analytics but not specific methods. |
| Benchmarks | Which benchmark objects/tables are used by APX performance analytics? | Unknown | APX index data interface documentation and SQL schema. |
| Composite/GIPS | Does APX include composite management and GIPS reporting; how are composite returns calculated and stored? | Unknown / Medium Confidence | Third-party product material mentions GIPS support, but vendor implementation detail is needed. |
| SSRS reports | Which APX SSRS reports contain portfolio/security/classification performance? | Unknown | APX report catalog and RDL files. |
| Multi-currency attribution | Does APX separate local return, base return, and currency effect? | Unknown | APX analytics documentation and sample report. |
| After-tax performance | Whether APX natively supports after-tax performance in the relevant installed versions. | Unknown | Vendor docs or version-specific release notes. |

### 5.3 Candidate APX report leads — unverified until full guide is reviewed

A public search result for an APX reports guide mentions a report that highlights investment decisions and compares portfolio performance to a benchmark over a time period. The full guide was not available for inspection in this pass, so no exact report name should be treated as Verified from that snippet alone.

| Candidate report area | Possible purpose | Confidence | Required validation |
|---|---|---|---|
| Portfolio vs benchmark performance report | Compares portfolio performance to benchmark over a selected period. | Medium Confidence | Full APX Reports Guide or installed SSRS report catalog. |
| Sector/security performance analytics report | Explains performance sources at sector/security level. | Medium Confidence | APX performance analytics documentation and sample outputs. |
| Attribution report | Breaks excess return into effects. | Unknown | APX attribution guide/report catalog. |
| Composite performance report | Shows composite-level performance. | Unknown | APX composite/GIPS module documentation. |

---

## 6. IMEX Research for Performance

### 6.1 What is known from current evidence

No supplied IMEX catalog, IMEX export sample, or vendor IMEX reference was provided with this request. Public web material found in this research pass did not verify exact performance-related IMEX object names, field names, file layouts, or calculation semantics.

### 6.2 Candidate IMEX performance objects and extracts

| Candidate IMEX object / extract | Intended content | Axys | APX | Confidence | Required validation |
|---|---|---:|---:|---|---|
| `portperf` | Portfolio/account-level performance by period. | Unknown | Unknown | Unknown | IMEX object catalog and sample export. |
| `secperf` | Security-level performance by account/security/period. | Unknown | Unknown | Unknown | IMEX object catalog and sample export. |
| Security master / security reference extract | Security identifiers and classifications needed to interpret security performance. | Unknown | Unknown | Medium Confidence as dependency | Actual export object name and fields must be verified. |
| Portfolio/account master extract | Account identifiers, account names, base currency, inception/close dates. | Unknown | Unknown | Medium Confidence as dependency | Actual object name and field list must be verified. |
| Price extract | Market prices used in performance valuation. | Unknown | Unknown | Medium Confidence as dependency | Needed to audit performance inputs; not necessarily needed for stored performance export. |
| Transaction extract | Cash flows, income, trades, fees, transfers affecting performance. | Unknown | Unknown | Medium Confidence as dependency | Needed to audit recalculation causes; object/fields must be verified. |
| Classification extract | Industry/sector/country/style groupings used in classification performance. | Unknown | Unknown | Medium Confidence as dependency | Object/fields must be verified. |
| Benchmark/index return extract | Benchmark returns used in performance/relative reports. | Unknown | Unknown | Medium Confidence as dependency | Object/fields must be verified. |

### 6.3 Minimal IMEX information needed for a useful performance chapter

Chapter 10 should not assert exact object names unless these are supplied. A performance chapter still needs the following categories of IMEX evidence.

| Category | Minimum evidence needed | Why it matters |
|---|---|---|
| Portfolio performance export | Object name, field list, date grain, return fields, market value fields, flow/income fields. | Determines whether portfolio returns are stored/exportable. |
| Security performance export | Object name, field list, date grain, security id fields, weights, returns, contribution fields. | Determines whether security-level performance can be reconciled to portfolio performance. |
| Benchmark performance export | Object name, benchmark id, period return fields, date grain. | Needed for relative performance and attribution. |
| Report parameters | Date range, account/group, benchmark, currency, gross/net, accrual, trade/settlement, classification level. | Performance output often changes with report options. |
| Export examples | At least one single-period and one multi-period example. | Needed to identify linking/recalculation behavior. |
| Changed-performance example | Same period run twice with different results. | Needed to document quirks and audit workflow. |

### 6.4 Candidate field dictionary for performance exports

The following dictionary is a **candidate research checklist**, not a verified Axys/APX field list. Exact field names should be replaced only after sample exports or official object documentation are available.

| Field | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---|
| `PortfolioCode` / account id | Portfolio/account identifier. | Unknown | Unknown | Unknown | Unknown | Unknown |
| `PortfolioName` | Portfolio/account display name. | Unknown | Unknown | Unknown | Unknown | Unknown |
| `StartDate` | Period start date. | Unknown | Unknown | Unknown | Unknown | Unknown |
| `EndDate` | Period end date. | Unknown | Unknown | Unknown | Unknown | Unknown |
| `CurrencyCode` | Base/reporting/local currency code. | Unknown | Unknown | Unknown | Unknown | Unknown |
| `BeginningMarketValue` | Beginning market value used for performance. | Unknown | Unknown | Unknown | Unknown | Unknown |
| `EndingMarketValue` | Ending market value used for performance. | Unknown | Unknown | Unknown | Unknown | Unknown |
| `NetContributions` | External net flows during the period. | Unknown | Unknown | Unknown | Unknown | Unknown |
| `Income` | Income included in performance period. | Unknown | Unknown | Unknown | Unknown | Unknown |
| `Fees` | Fees included/excluded depending on gross/net option. | Unknown | Unknown | Unknown | Unknown | Unknown |
| `PortfolioReturn` | Portfolio-level return for the period. | Unknown | Unknown | Unknown | Unknown | Unknown |
| `GrossReturn` | Gross-of-fee return, if supported. | Unknown | Unknown | Unknown | Unknown | Unknown |
| `NetReturn` | Net-of-fee return, if supported. | Unknown | Unknown | Unknown | Unknown | Unknown |
| `BenchmarkCode` | Benchmark identifier. | Unknown | Unknown | Unknown | Unknown | Unknown |
| `BenchmarkReturn` | Benchmark return for the same period. | Unknown | Unknown | Unknown | Unknown | Unknown |
| `SecurityID` | Security identifier used in performance row. | Unknown | Unknown | Unknown | Unknown | Unknown |
| `Symbol` / ticker | Ticker or display symbol. | Unknown | Unknown | Unknown | Unknown | Unknown |
| `SecurityName` | Security display name. | Unknown | Unknown | Unknown | Unknown | Unknown |
| `AssetClass` | Asset class classification. | Unknown | Unknown | Unknown | Unknown | Unknown |
| `Sector` / `Industry` | Sector/industry classification. | Unknown | Unknown | Unknown | Unknown | Unknown |
| `Weight` | Security or classification weight used in performance/contribution. | Unknown | Unknown | Unknown | Unknown | Unknown |
| `SecurityReturn` | Security-level return. | Unknown | Unknown | Unknown | Unknown | Unknown |
| `Contribution` | Security/classification contribution to portfolio return. | Unknown | Unknown | Unknown | Unknown | Unknown |
| `LocalReturn` | Local-currency return, if supported. | Unknown | Unknown | Unknown | Unknown | Unknown |
| `CurrencyEffect` | FX/currency contribution, if supported. | Unknown | Unknown | Unknown | Unknown | Unknown |
| `AccruedIncome` | Accrued income included in valuation/performance. | Unknown | Unknown | Unknown | Unknown | Unknown |
| `RunDate` | Date/time performance report/export was generated. | Unknown | Unknown | Unknown | Unknown | Unknown |

---

## 7. REP / Report Research for Performance

### 7.1 What is known from current evidence

| Topic | Statement | Confidence | Notes |
|---|---|---|---|
| Axys report customization | Axys public material supports predefined reports and customization. | Verified for marketing claim | Exact REP report names unknown. |
| APX reporting framework | APX product brief states the APX reporting framework is built around Microsoft Reporting Services. | Verified for marketing claim | Important APX/Axys distinction. |
| Replang / Report Writer Pro | Public consultant material states Axys/APX users can create reports via Report Writer Pro or modify Replang source directly. | Medium Confidence | Needs confirmation in repository REP chapter/source. |
| Direct report export | Public consultant material states Axys/APX reports can be exported to Excel/CSV/text through several approaches. | Medium Confidence | Needs local validation. |

### 7.2 Report inventory needed

For a useful final performance chapter, collect a report inventory with these fields.

| Report file/name | System | Level | Purpose | Stored vs recalculated? | Source technology | Confidence |
|---|---|---|---|---|---|---|
| Unknown | Axys | Portfolio | Account-level performance. | Unknown | REP/Replang? | Unknown |
| Unknown | Axys | Security | Security-level performance. | Unknown | REP/Replang? | Unknown |
| Unknown | Axys | Classification | Sector/industry/country performance. | Unknown | REP/Replang? | Unknown |
| Unknown | Axys | Composite | Composite/GIPS performance. | Unknown | REP/Replang? | Unknown |
| Unknown | APX | Portfolio | Account-level performance. | Unknown | SSRS/RDL? | Unknown |
| Unknown | APX | Security | Security-level performance. | Unknown | SSRS/RDL? | Unknown |
| Unknown | APX | Classification | Sector/security analytics. | Unknown | SSRS/RDL? | Unknown |
| Unknown | APX | Benchmark/relative | Portfolio vs benchmark performance. | Unknown | SSRS/RDL? | Medium Confidence as report category only |
| Unknown | APX | Attribution | Performance attribution. | Unknown | SSRS/RDL? | Unknown |

### 7.3 REP/report source questions

| Question | Axys | APX | Current answer |
|---|---:|---:|---|
| Are Axys performance reports implemented in Replang `.rep` files? | Unknown | N/A | Unknown |
| Are APX performance reports implemented as SSRS reports rather than Replang? | N/A | Medium Confidence | Public APX material says reporting framework is built around Microsoft Reporting Services. |
| Can APX also use Replang/REP for some report workflows? | N/A | Unknown | Needs APX implementation evidence. |
| Which reports expose stored performance values rather than calculated values? | Unknown | Unknown | Critical unknown. |
| Which reports support CSV/text output without manual Excel export? | Unknown | Unknown | Needs report source/macro/export examples. |

---

## 8. Performance Data Model Research

### 8.1 Conceptual entities likely required

These entities are conceptual and should not be treated as Axys/APX table names.

| Entity | Purpose | Axys | APX | Confidence |
|---|---|---:|---:|---|
| Portfolio/account | Account whose performance is calculated. | High Confidence | High Confidence | Product category requires accounts/portfolios. Exact fields unknown. |
| Security | Investment/security whose return/contribution may be reported. | High Confidence | High Confidence | Product category requires holdings/security master. Exact fields unknown. |
| Period | Date interval for performance calculation. | High Confidence | High Confidence | Performance reporting requires date ranges. Exact date grain unknown. |
| Valuation | Beginning/ending market values and possibly accrued income. | Medium Confidence | Medium Confidence | Needed for many performance calculations; exact stored model unknown. |
| Transaction / cash flow | External flows and income affecting performance. | Medium Confidence | Medium Confidence | Needed for return calculations/audit; exact treatment unknown. |
| Performance result | Portfolio/security/classification return and contribution values. | Unknown | Unknown | Need to verify stored vs calculated. |
| Benchmark/index | Benchmark returns and constituent classifications. | Unknown | Medium Confidence | APX public material confirms benchmark data for performance analytics; exact model unknown. |
| Classification | Grouping used for sector/industry/country reporting. | Medium Confidence | Medium Confidence | Needed for sector/security analytics; exact source unknown. |
| Composite | Group of portfolios for composite/GIPS reporting. | Unknown | Unknown | Public/product material not enough to verify implementation. |

### 8.2 Stored-vs-recalculated matrix

This is the central implementation question for Chapter 10.

| Output | Axys stored? | Axys recalculated? | APX stored? | APX recalculated? | Current status |
|---|---:|---:|---:|---:|---|
| Daily portfolio return | Unknown | Unknown | Unknown | Unknown | Unknown |
| Monthly portfolio return | Unknown | Unknown | Unknown | Unknown | Unknown |
| Multi-period linked portfolio return | Unknown | Unknown | Unknown | Unknown | Unknown |
| Security return | Unknown | Unknown | Unknown | Unknown | Unknown |
| Security contribution | Unknown | Unknown | Unknown | Unknown | Unknown |
| Classification return | Unknown | Unknown | Unknown | Unknown | Unknown |
| Benchmark return | Unknown | Unknown | Unknown | Unknown | Unknown |
| Composite return | Unknown | Unknown | Unknown | Unknown | Unknown |
| Attribution effects | Unknown | Unknown | Unknown | Unknown | Unknown |

### 8.3 Gross/net and calculation basis options

| Option | Axys | APX | Current status | Evidence needed |
|---|---:|---:|---|---|
| Gross-of-fee performance | Unknown | Unknown | Unknown | Report options and output examples. |
| Net-of-fee performance | Unknown | Unknown | Unknown | Report options and fee transaction treatment. |
| Trade-date performance | Unknown | Unknown | Unknown | Report options and controlled examples. |
| Settlement-date performance | Unknown | Unknown | Unknown | Report options and controlled examples. |
| Accrual-inclusive performance | Unknown | Unknown | Unknown | Report options and valuation examples. |
| Local-currency returns | Unknown | Unknown | Unknown | Multi-currency report examples. |
| Base-currency returns | Unknown | Unknown | Unknown | Multi-currency report examples. |
| Currency effect | Unknown | Unknown | Unknown | Attribution/multi-currency analytics docs. |

---

## 9. Processing Behavior Questions and Test Designs

### 9.1 Stored monthly returns vs recalculation test

| Test | Procedure | Expected evidence |
|---|---|---|
| Single-month portfolio report | Run portfolio performance for one closed month; export report and any performance IMEX object. | Identifies one-period value and field labels. |
| Twelve-month report | Run same portfolio for twelve-month range. | Compare report return to geometric linking of twelve monthly exported returns. |
| Post-period price correction | Change or identify a price correction after period close; rerun same closed period. | Determines whether historical performance changes and why. |
| Post-period transaction correction | Add/backdate a transaction into a closed period; rerun performance. | Determines whether performance is dynamic and what inputs affect it. |
| Stored value comparison | Compare report output to stored performance export, if available. | Determines whether report reads stored values or recalculates. |

### 9.2 Security performance footing test

| Test | Procedure | Interpretation |
|---|---|---|
| Sum of contributions | For one period, sum security contribution field and compare to portfolio return. | If it reconciles, contribution may be preferred over `weight * return`. |
| Sum of `weight * return` | Calculate security weights times security returns and compare to portfolio return. | If it does not foot, identify weight definition, flow timing, cash, fees, or rounding differences. |
| Cash included/excluded | Repeat with and without cash rows, if available. | Determines whether cash is a security-like row or separate component. |
| Classification rollup | Sum security contributions by classification and compare to classification report. | Determines whether classification performance is derived or separately calculated. |

### 9.3 Benchmark / relative performance test

| Test | Procedure | Interpretation |
|---|---|---|
| Portfolio vs benchmark single period | Export portfolio and benchmark returns for same dates. | Identifies benchmark source and date alignment. |
| Benchmark revision | Change benchmark return input or identify vendor revision; rerun report. | Determines sensitivity to benchmark changes. |
| Classification benchmark | Compare portfolio sector performance to benchmark sector return/weight. | Determines attribution/classification dependency. |

### 9.4 Multi-currency test

| Test | Procedure | Interpretation |
|---|---|---|
| Local-only security | Security held in non-base currency with no trade flows. | Separates local security return from FX effect if report supports both. |
| FX rate correction | Change historical FX rate; rerun performance. | Identifies whether base return changes and where FX source is used. |
| Income in local currency | Dividend/coupon in local currency. | Tests income, withholding tax, FX translation, and accrual behavior. |

---

## 10. Known Issues / Quirks — Research Hypotheses

The following are not final facts. They are operational hypotheses commonly relevant to portfolio accounting performance systems and should be validated against Axys/APX outputs before becoming chapter content.

| Quirk / issue | Axys | APX | Confidence | Validation method |
|---|---:|---:|---|---|
| Historical performance can change after a closed period if prices, transactions, income, fees, FX rates, classifications, or security master data are corrected. | Unknown | Unknown | Medium Confidence as general system risk | Rerun closed-period reports after controlled input change. |
| Portfolio-level return may not equal simple sum of security `weight * return` because of cash flows, weight definition, cash treatment, fees, accruals, rounding, and report method. | Unknown | Unknown | Medium Confidence as general performance-system risk | Compare report contribution fields to independent calculations. |
| Security-level report run for a year may differ from linked monthly security-level results if the report recalculates over the full year rather than linking stored monthlies. | Unknown | Unknown | Medium Confidence as general performance-system risk | Monthly vs annual report comparison. |
| Classification-level performance may depend on classification as of run date, period end date, or security history. | Unknown | Unknown | Medium Confidence as general data-model risk | Change classification history and rerun old period. |
| Benchmarks/index returns may be maintained separately from portfolio accounting and can create relative-performance changes when revised. | Unknown | Medium Confidence for APX benchmark dependency | APX public index data material confirms benchmark data support; exact behavior unknown. |
| Rounding in reports may prevent visible values from footing exactly. | Unknown | Unknown | Medium Confidence as general report risk | Export raw precision if available. |
| Different reports may use different options/defaults for gross/net, accrual, cash inclusion, trade/settlement date, or benchmark. | Unknown | Unknown | Medium Confidence as general report risk | Compare report parameter sets and report source. |

---

## 11. Version Difference Research

| Version / product area | Axys | APX | Current status | Evidence needed |
|---|---:|---:|---|---|
| Legacy Axys vs current Axys | Unknown | N/A | Unknown | Axys release notes and installed version documentation. |
| APX pre/post performance analytics module changes | N/A | Unknown | Unknown | APX release notes/performance analytics documentation. |
| APX SSRS report changes across versions | N/A | Unknown | Unknown | APX report guide versioned PDFs/RDL inventory. |
| IMEX object differences between Axys and APX | Unknown | Unknown | Unknown | IMEX catalogs from both systems. |
| Gross/net or fee treatment changes | Unknown | Unknown | Unknown | Versioned documentation and regression examples. |
| Multi-currency calculation changes | Unknown | Unknown | Unknown | Versioned documentation and multi-currency examples. |
| Fixed income/accrual behavior changes | Unknown | Unknown | Unknown | Release notes and examples. |

Public SS&C release material from 2020 describes APX as a core accounting, reporting, and performance measurement application and mentions fixed-income improvements, but this does not establish detailed performance calculation changes.

---

## 12. Examples to Collect

### 12.1 Single-period portfolio performance example

Needed fields:

| Field | Example value | Source needed |
|---|---:|---|
| Portfolio/account code | Unknown | Report/export sample |
| Period start date | Unknown | Report/export sample |
| Period end date | Unknown | Report/export sample |
| Beginning market value | Unknown | Report/export sample |
| Ending market value | Unknown | Report/export sample |
| Contributions/withdrawals | Unknown | Report/export sample |
| Income | Unknown | Report/export sample |
| Fees | Unknown | Report/export sample |
| Gross return | Unknown | Report/export sample |
| Net return | Unknown | Report/export sample |
| Benchmark return | Unknown | Report/export sample |

### 12.2 Security-level performance example

Needed fields:

| Field | Example value | Source needed |
|---|---:|---|
| Portfolio/account code | Unknown | Report/export sample |
| Security id | Unknown | Report/export sample |
| Security name | Unknown | Report/export sample |
| Classification | Unknown | Security master/classification export |
| Beginning weight | Unknown | Report/export sample |
| Average weight | Unknown | Report/export sample |
| Ending weight | Unknown | Report/export sample |
| Security return | Unknown | Report/export sample |
| Contribution | Unknown | Report/export sample |
| Local return | Unknown | Multi-currency report/export sample |
| Currency effect | Unknown | Multi-currency report/export sample |

### 12.3 Changed historical performance example

Needed evidence:

| Evidence | Why needed |
|---|---|
| Original report output | Establish baseline. |
| New report output | Establish changed result. |
| Run dates | Prove the same performance period was run at different times. |
| Input differences | Identify price, transaction, security master, classification, benchmark, or FX change. |
| Report options | Ensure change is not caused by report parameter differences. |
| IMEX extracts before/after | Allow reproducible diagnosis. |

---

## 13. Unknowns Register

These Unknowns should be preserved in Chapter 10 until verified.

| ID | Unknown | Why it matters | Evidence required |
|---|---|---|---|
| PERF-U-001 | Exact Axys performance file/table/object names. | Needed for implementation reference. | Axys data dictionary, IMEX catalog, or sample files. |
| PERF-U-002 | Exact APX performance SQL tables/views/procedures. | Needed for implementation reference. | APX schema documentation or production database inspection. |
| PERF-U-003 | Exact IMEX objects for portfolio performance. | Needed for export guidance. | IMEX catalog and sample export. |
| PERF-U-004 | Exact IMEX objects for security performance. | Needed for security-level analytics/export. | IMEX catalog and sample export. |
| PERF-U-005 | Exact REP/Replang report files for Axys performance reports. | Needed for REP chapter cross-reference. | Axys REP folder inventory and report source. |
| PERF-U-006 | Exact APX SSRS/RDL reports for performance. | Needed for APX report documentation. | APX report catalog/RDL files. |
| PERF-U-007 | Whether portfolio returns are stored, recalculated, or both. | Core behavior question. | Stored values plus rerun tests. |
| PERF-U-008 | Whether security returns/contributions are stored or report-calculated. | Core reconciliation question. | Export/report comparison. |
| PERF-U-009 | Whether multi-period reports link stored periods or recalculate over full range. | Explains why old returns may change. | Monthly vs annual tests. |
| PERF-U-010 | Which return methodology is used by each report. | Prevents incorrect interpretation. | Report documentation and controlled examples. |
| PERF-U-011 | Gross vs net fee treatment. | Required for client/composite reporting. | Report options and fee transaction examples. |
| PERF-U-012 | Accrued income treatment. | Important for fixed income and income-heavy portfolios. | Report options and test portfolio. |
| PERF-U-013 | Cash treatment in security performance. | Needed for contribution footing. | Security report examples with cash rows. |
| PERF-U-014 | Classification timing/historical treatment. | Needed for sector/country performance. | Classification history tests. |
| PERF-U-015 | Benchmark data storage and revision behavior. | Needed for relative performance. | Benchmark export and revision test. |
| PERF-U-016 | Composite/GIPS support details. | Needed for composite performance section. | Composite report docs and examples. |

---

## 14. Recommended Source Material to Request Next

The blueprint instructs that unsupported information should remain Unknown rather than invented. To upgrade this research into a high-confidence chapter, request the following material.

| Priority | Requested material | Why it is needed |
|---:|---|---|
| 1 | Axys IMEX object list or screenshots showing performance-related objects. | Verify `portperf`, `secperf`, and exact export names. |
| 2 | Sample Axys portfolio performance IMEX export. | Verify fields, period grain, stored values. |
| 3 | Sample Axys security performance IMEX export. | Verify security return/weight/contribution behavior. |
| 4 | Axys REP folder/report inventory for performance reports. | Verify report names and source files. |
| 5 | Sample Axys performance reports: account, security, classification, benchmark, composite if applicable. | Verify report names and output fields. |
| 6 | APX report catalog or Reports Guide PDF. | Verify APX report names and SSRS report areas. |
| 7 | APX sample SSRS/RDL files or report output for performance analytics. | Verify APX report fields and parameters. |
| 8 | APX IMEX/export samples for portfolio/security performance, if available. | Verify APX export path and fields. |
| 9 | APX database dictionary or SQL view list for performance-related objects. | Verify storage/recalculation model. |
| 10 | Controlled before/after rerun example where old performance changed. | Document implementation quirks with evidence. |

---

## 15. Draft Reference Notes for Future Chapter

### 15.1 Statements safe to use now

| Statement | Confidence | Suggested chapter placement |
|---|---|---|
| Axys is used for portfolio accounting/reporting and includes reporting that presents portfolio performance. | Verified for public marketing claim | Axys overview; avoid internal details. |
| Axys has predefined and customizable reporting. | Verified for public marketing claim | Axys / REP sections. |
| APX integrates portfolio management, accounting, reporting, and performance analytics. | Verified for public marketing claim | APX overview. |
| APX performance analytics are intended to help explain portfolio performance at sector/security level. | Verified for public marketing claim | APX performance analytics overview. |
| APX can use benchmark/index data in performance analytics. | Verified for public marketing claim | APX benchmark section. |
| APX reporting framework is built around Microsoft Reporting Services. | Verified for public marketing claim | REP/APX reporting section. |
| Exact fields, object names, storage behavior, and report source names remain Unknown without IMEX/REP samples or vendor technical docs. | Verified as limitation | Unknowns section. |

### 15.2 Statements not safe to use yet

| Statement | Why not safe |
|---|---|
| Axys stores TWR in a specific file/table. | No source supplied. |
| APX stores performance in a specific SQL table/view. | No source supplied. |
| `portperf` is the exact IMEX object for portfolio performance. | Not verified by supplied material. |
| `secperf` is the exact IMEX object for security performance. | Not verified by supplied material. |
| A named REP report contains security performance. | No report inventory supplied. |
| Any specific return formula is used by Axys or APX. | No calculation documentation or controlled example supplied. |
| Security performance must foot to portfolio performance by `weight * return`. | No system evidence supplied; often report-specific. |
| APX supports a specific attribution method such as Brinson-Fachler in the current installed version. | Public material confirms analytics generally, not specific methodology/version. |

---

## 16. External Reference URLs

These are research leads, not exhaustive citations.

| Reference | URL | Notes |
|---|---|---|
| SS&C Advent Axys product page | `https://www.advent.com/solutions/axys/` | High-level Axys reporting/accounting/performance visibility. |
| SS&C Advent Portfolio Exchange product page | `https://www.advent.com/solutions/advent-portfolio-exchange/` | High-level APX platform description. |
| SS&C APX product brief | `https://cdn.advent.com/cms/pdfs/briefs/PB_APX.pdf` | High-level APX reporting framework and performance analytics statements. |
| SS&C APX index data brief | `https://cdn.advent.com/cms/pdfs/briefs/PB_INDATA.pdf` | High-level benchmark/index data support for APX performance analytics. |
| SS&C 1H2020 Advent product updates | `https://www.advent.com/news-and-insights/press-releases/ssc-announces-1h2020-ssc-advent-product-updates/` | Mentions APX as accounting, reporting, and performance measurement application. |
| APX Reports Guide lead | `https://cdn.advent.com/cms/pdfs/reports/REP_APX.pdf` | Search-result lead only; full PDF should be collected/reviewed. |
| AdventGuru IMEX/reporting lead | `https://adventguru.com/tag/imex/` | Consultant lead on report export, Report Writer Pro, Replang, xPort. |
| AdventGuru Axys lead | `https://adventguru.com/tag/axys/` | Consultant lead on Axys/APX report customization. |

---

## 17. Appendix — Research-to-Chapter Boundary Guidance

When using this research to maintain `Chapter_10_Performance.md`:

1. Keep Axys and APX sections separate.
2. Do not promote any candidate field, report, or object name to fact without evidence.
3. Use tables for:
   - report inventory,
   - IMEX object inventory,
   - field dictionary,
   - stored-vs-recalculated behavior,
   - known quirks,
   - version differences,
   - Unknowns.
4. Include examples only from supplied report/export samples.
5. Clearly distinguish:
   - portfolio performance,
   - security performance,
   - classification performance,
   - benchmark/relative performance,
   - attribution,
   - composite/GIPS performance.
6. Treat `Unknown` as a valid result, not a gap to fill with speculation.

## 18. Deep IMEX Addendum Incorporated 2026-06-30

Source: `axys_imex_deep_research.md`.

Additional performance-related points:

| Topic | Addendum | Confidence |
|---|---|---:|
| Performance history | Public conversion evidence mentions performance history as an IMEX/migration concern, but clean object names and field lists remain Unknown. | Medium / Unknown |
| Candidate performance fields | Live discovery should inspect portfolio/security/classification/composite identifiers, period start/end dates, beginning/ending market value, return, contribution, external flow, income, fees, average/modified-Dietz weights, benchmark, and currency fields. | Discovery guidance |
| `portperf` / `secperf` status | The deep research reinforces that `portperf` and `secperf` should remain normalized/product-local names unless a live IMEX object, report output, or vendor manual confirms native names. | Unknown |
| REP/report preference | Performance values often require user-visible tie-out; REP/Replang/custom reports may be more appropriate than IMEX when the exact reported value matters. | Design guidance |
| Catalog requirement | Any product should record extraction mechanism, source report or IMEX object, source row, calculation/stored-value confidence, and version. | Design guidance |

## Deep Research Update Incorporated 2026-07-02

The July 2026 addendum upgrades several performance capabilities to verified
product-level evidence while preserving implementation Unknowns. Axys product
material supports time-weighted and internal rates of return, before/after
management fees; blended benchmarks and component-index history; synthetic
index comparison; performance display by portfolio, asset class, sector,
country, or region; composite-management reporting; GIPS-related support; and
multi-currency return components attributable to market prices versus currency
fluctuations. Exact report formulas, files, storage, and IMEX objects remain
Unknown.

Morningstar conversion evidence provides Axys reconciliation report leads:
Performance Summary, Performance by Account, Performance by Security, Portfolio
Cash Flow, Portfolio Current Value, Unrealized Gain/Loss, and Realized
Gain/Loss. Zacks evidence adds third-party return-setting clues: Daily
Calculation Combined Return as a TWR-style setting based on the Advent Axys
Asset Reconciliation Report, and Whole Period Return as a cash-flow-weighted /
IRR-style setting based on the Advent Axys Performance By Security Report.
These are test-design clues, not official Axys formulas.

APX index-data evidence supports loading benchmark/index data into APX for
performance analysis/reporting, including sector-level and security-level index
data, historical index data, and an Attribution by Industry Sector example. The
APX Performance Overview report is an official report-guide lead for
multi-period portfolio performance and benchmark comparison. APX performance
tables/views/procedures, benchmark fields, attribution formulas, and
stored-versus-recalculated behavior remain Unknown.

## Short Lifecycle Performance Addendum Incorporated 2026-07-07

The `temp_axys_apx_short_lifecycle_accounting_reporting.md` research supports a
controlled synthetic performance scenario for lowercase `ss` and `cs` when
assumptions are disclosed. In that model:

- `ss` creates or increases short exposure and related proceeds/collateral
  mechanics;
- `cs` reduces or closes the short exposure;
- open short exposure may be modeled with negative quantity and negative market
  value;
- realized gain/loss occurs when the short is covered; and
- neither `ss` nor `cs` is treated as a client contribution or withdrawal for
  Modified Dietz.

This is a defensible demo convention, not public proof of universal native
Axys/APX report behavior. Production interpretation still needs local evidence
for holdings signs, cash/proceeds buckets, source/destination fields, and
reported performance treatment.
