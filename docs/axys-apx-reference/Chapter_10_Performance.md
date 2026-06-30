# Chapter 10 — Performance

**Repository:** AXYS / APX Reference Repository
**Chapter:** `Chapter_10_Performance.md`
**Prepared from supplied research:** `Research_10_Performance.md` plus supplied supporting research for IMEX, REP, transactions, holdings, cash, pricing, corporate actions, security master, and classifications.
**Governing specification:** `AXYS_APX_REFERENCE_BLUEPRINT.md`, Version 2.0

---

## Related chapters
- [Chapter_01_Overview.md](Chapter_01_Overview.md) — provides the repository-wide map and evidence conventions, plus the shared safe implementation rules.
- [Chapter_05_Transactions.md](Chapter_05_Transactions.md) — transactions are important performance inputs.
- [Chapter_06_Holdings.md](Chapter_06_Holdings.md) — position and valuation changes feed performance calculations.
- [Chapter_14_Reports.md](Chapter_14_Reports.md) — performance is often surfaced through report families and report labels.

## 1. Overview

This chapter documents the currently supported facts and open questions about performance behavior in SS&C Advent Axys and SS&C Advent Portfolio Exchange (APX).

The available supplied research supports broad performance/reporting capability statements for Axys and APX, but it does **not** provide enough evidence to document exact native performance storage files, APX SQL tables, APX stored procedures, Axys `.REP` performance report files, official IMEX performance object names, or report-specific return calculation methods.

Therefore, this chapter is intentionally conservative. Unsupported implementation details are marked **Unknown**.

Performance inputs are downstream of transactions, holdings, cash, prices, and classifications. The supplied evidence does not support treating a transaction code or a report label as sufficient proof of performance semantics; for implementation and audit work, performance analysis should preserve the raw event context and classify external flows, trading activity, income, fees, and corporate actions before interpreting returns and cash-flow effects.

In particular, `li` and `lo` should be handled as external-flow
candidates, not automatic Modified Dietz contribution/withdrawal flags.
The classification should use security type, security symbol,
source/destination type and symbol, amount and quantity signs,
reversal/cancellation state, and firm-specific mapping so that fees,
sweeps, corrections, cash-security activity, and in-kind transfers are
not misread as client cash flows.

Performance review workflows should preserve a "requires review" bucket
for ambiguous `li`, `lo`, `dp`, `wd`, `;`, `epus`, and `exus` cases. In
observed public mappings, dividend reinvestment may use `dv`, `by`, and
`dvwash`, while margin-interest handling may use `ai` with margin cash
context; neither pattern should be treated as an external cash flow.

### 1.1 Confidence labels

| Label | Meaning in this chapter |
|---|---|
| Verified | Directly supported by supplied research, vendor/public material summarized in the research, or identified report/integration documentation. |
| High Confidence | Strongly supported by supplied research and consistent implementation evidence, but still not a complete vendor specification. |
| Medium Confidence | Plausible and operationally useful, but supported only indirectly, by product/consultant/integration material, or by general portfolio-accounting behavior. |
| Unknown | Not established by supplied source material. Do not implement or document as fact without additional evidence. |

### 1.2 Scope

Performance in this chapter includes:

| Area | Included? | Notes |
|---|---:|---|
| Portfolio/account performance | Yes | Return, valuation, cash flow, gross/net, benchmark comparison where evidence exists. |
| Security-level performance | Yes | Security return, weight, contribution, and footing behavior remain mostly Unknown. |
| Classification performance | Yes | Sector/asset-class/country/region/custom classification performance is included where supported by product/reporting evidence. |
| Benchmark-relative performance | Yes | APX benchmark/index support is vendor-supported at a high level; detailed schema remains Unknown. |
| Composite/GIPS performance | Yes | Broad product claims exist; detailed calculation/storage remains Unknown. |
| Performance attribution | Limited | APX performance analytics are vendor-supported at a high level, but exact attribution methods are Unknown from supplied material. |
| General performance theory | No | Included only when required to define system behavior or test design. |

---

## 2. Executive Summary

| Topic | Axys | APX | Confidence | Notes |
|---|---|---|---:|---|
| Product performance capability | Axys is portfolio reporting/accounting software with performance measurement/reporting capabilities. | APX is an integrated portfolio management/accounting/reporting platform with performance analytics. | Verified for product capability | Public vendor material summarized in supplied research supports high-level capability only. |
| Reporting customization | Axys has predefined reports and customization, including Report Writer Pro. | APX has standard reports, custom reporting, dashboards, and SSRS-oriented reporting paths. | Verified for high-level capability | Exact performance report files and fields are mostly Unknown. |
| REP / report extraction | Axys reports use RepLang `.REP` files in supported examples; REP32-based extraction is used by at least one Axys/APX connector. | APX can be accessed by report/macro extraction in at least one connector and also has SQL/SSRS-style reporting paths. | Verified for cited workflows; Medium for generalization | Report-specific performance behavior remains Unknown. |
| IMEX performance objects | Candidate names such as `portperf` and `secperf` are not verified. | Same. | Unknown | Do not document as official object names without actual IMEX catalog or exports. |
| Stored vs recalculated returns | Unknown. | Unknown. | Unknown | This is the central unresolved behavior for Chapter 10. |
| Security performance footing | Unknown whether security rows foot to portfolio return by `weight * return`, contribution fields, report-specific logic, or another method. | Same. | Unknown | Requires security performance exports/reports. |
| Classification performance | Axys public material supports performance display by asset class, sector, country, and region. | APX report-guide snippet supports custom classification, industry group, and sector reporting. | Verified at reporting category level | Storage/history behavior remains Unknown. |
| Benchmark/index support | Unknown detailed Axys model. | APX public material supports benchmark/index data for performance analytics. | APX Verified for high-level capability | Detailed benchmark tables/export fields are Unknown. |
| Historical performance changes | Possible when underlying accounting/reference data changes, but exact Axys/APX recalculation behavior is Unknown. | Same. | Medium Confidence as operational risk; Unknown as product behavior | Requires controlled before/after report tests. |

---

## 3. Axys

### 3.1 Verified / supported Axys behavior

| Topic | Statement | Confidence |
|---|---|---:|
| Product category | Axys is portfolio reporting and accounting software. | Verified |
| Reporting | Axys provides predefined reports and report customization. | Verified |
| Report Writer Pro | Axys supports Axys Report Writer Pro. | Verified |
| Performance visibility | Axys product material states that Axys provides visibility into portfolio performance. | Verified for marketing/product claim |
| REP / Replang | Supplied REP research supports that Axys reports are written in RepLang and that `.REP` files such as `AMAN.REP` can be copied and edited in a text editor. | Verified for cited report examples |
| Report execution / extraction | A third-party connector uses Advent standard reports/macros, `REP32.exe`, the REP32 engine, RepLang scripting, and macros for Axys/APX extraction. | Verified for connector |

### 3.2 Axys performance behavior not yet verified

| Area | Current Status | Needed evidence |
|---|---|---|
| Portfolio return method | Unknown | Vendor performance manual, report guide, REP source, or controlled report output. |
| Stored vs calculated returns | Unknown | Performance storage/export sample and report rerun tests. |
| Daily/monthly return storage | Unknown | Axys files, IMEX object catalog, or production export. |
| Multi-period linking | Unknown | Monthly vs multi-period report comparison. |
| Security-level returns | Unknown | Security performance report/export. |
| Security contribution | Unknown | Report/export showing contribution and portfolio return. |
| Classification performance | Unknown below high-level reporting capability | Classification performance report, REP source, or controlled classification tests. |
| Gross vs net fee treatment | Unknown | Report options, fee transactions, and paired gross/net samples. |
| Accrued income treatment | Unknown | Fixed-income report examples and report options. |
| Cash treatment | Unknown | Performance reports with cash rows and cash transaction coding. |
| Multi-currency performance | Unknown | Multi-currency performance report/export and FX test case. |
| Composite/GIPS calculation details | Unknown | Composite module/report documentation and examples. |

### 3.3 Candidate Axys performance terms — not verified

The following terms appear as candidate or user-recalled performance extract concepts, but the supplied research does not verify them as official Axys object names.

| Candidate term | Possible meaning | Status | Treatment |
|---|---|---:|---|
| `portperf` | Portfolio/account-level performance extract. | Unknown | Do not use as official IMEX object name without sample/export definition. |
| `secperf` | Security-level performance extract. | Unknown | Do not use as official IMEX object name without sample/export definition. |
| Portfolio Performance report | Account-level performance report. | Unknown | Need report inventory. |
| Security Performance report | Security-level return/contribution report. | Unknown | Need report inventory. |
| Classification Performance report | Classification/sector/asset-class performance report. | Unknown | Need report inventory. |
| Composite Performance report | Composite/GIPS performance report. | Unknown | Need report inventory. |

---

## 4. APX

### 4.1 Verified / supported APX behavior

| Topic | Statement | Confidence |
|---|---|---:|
| Product category | APX is an integrated portfolio management, accounting, reporting, and client-management platform. | Verified for product capability |
| Performance analytics | APX product material states that APX includes performance analytics. | Verified for product capability |
| Benchmark/index support | APX index-data material supports benchmark/index data for APX performance analytics. | Verified for product capability |
| Reporting framework | APX public material and supplied REP research support APX standard reports, custom reporting, dashboards, and SSRS/reporting paths. | Verified / High Confidence depending path |
| Accounting/reporting/performance scope | SS&C product update material described APX as a core accounting, reporting, and performance measurement application. | Verified for product claim |
| SQL/reporting access | Supplied REP/IMEX research records consultant evidence that APX users may use SQL Server, Public Views, Stored Accounting Functions, SSRS, REST API, and related tools. | Medium Confidence |

### 4.2 APX performance behavior not yet verified

| Area | Current Status | Needed evidence |
|---|---|---|
| APX performance storage tables | Unknown | APX schema, public view list, stored accounting function documentation, or SQL extracts. |
| Performance engine/procedures | Unknown | APX technical documentation or observed stored procedures/functions. |
| Stored vs dynamic performance | Unknown | Stored values plus report rerun tests. |
| Portfolio vs security performance objects | Unknown | APX export/schema/report samples. |
| Sector/classification performance storage | Unknown | APX performance analytics documentation or report definitions. |
| Attribution methodology | Unknown | APX attribution/performance analytics manual. |
| Benchmark storage and revision behavior | Unknown | Benchmark/index interface documentation and sample exports. |
| Composite/GIPS calculation details | Unknown | APX composite/GIPS documentation and reports. |
| SSRS performance report names and datasets | Unknown | APX Reports Guide, SSRS catalog, RDL files, or report samples. |
| Multi-currency attribution | Unknown | APX multi-currency performance sample. |
| After-tax performance | Unknown | Version-specific APX documentation. |

### 4.3 Candidate APX report areas — partially supported but not fully verified

| Candidate area | Possible purpose | Confidence | Required validation |
|---|---|---:|---|
| Portfolio vs benchmark report | Compares account performance to benchmark over a time period. | Medium Confidence | Full APX Reports Guide or SSRS report catalog. |
| Sector/security performance analytics | Explains sources of performance at sector/security level. | Medium Confidence | APX performance analytics documentation and sample output. |
| Attribution report | Breaks relative return into attribution effects. | Unknown | APX attribution report documentation. |
| Composite performance report | Shows composite/GIPS performance. | Unknown | APX composite report documentation. |

---

## 5. IMEX

### 5.1 Current evidence status

No supplied IMEX catalog, IMEX object dictionary, performance export sample, or vendor IMEX performance manual was provided. The supplied IMEX research verifies that IMEX is a significant Axys/APX interface area, but it does not verify exact performance object names or field layouts.

| IMEX topic | Axys | APX | Confidence |
|---|---|---|---:|
| IMEX exists / import-export relevance | Supported by supplied IMEX research. | Supported by supplied IMEX research and AIA/connector evidence. | Verified / High Confidence depending workflow |
| Performance object names | Unknown. | Unknown. | Unknown |
| Portfolio performance export fields | Unknown. | Unknown. | Unknown |
| Security performance export fields | Unknown. | Unknown. | Unknown |
| Benchmark performance export fields | Unknown. | Unknown. | Unknown |
| Stored vs calculated export semantics | Unknown. | Unknown. | Unknown |

### 5.2 Candidate IMEX extracts for performance work

These are extract categories, not verified object names.

| Extract category | Purpose | Axys | APX | Confidence |
|---|---|---:|---:|---:|
| Portfolio performance | Account-level return, valuation, flows, income, fees, benchmark. | Unknown object | Unknown object | Unknown |
| Security performance | Security return, weight, contribution, classification. | Unknown object | Unknown object | Unknown |
| Portfolio/account master | Account identifiers, names, base currency, inception/close dates. | Dependency | Dependency | Medium Confidence as dependency |
| Security master/reference | Security identifiers, names, types, currencies, classifications. | Dependency | Dependency | Medium Confidence as dependency |
| Holdings/positions | Beginning/end holdings, market value, quantity. | Dependency | Dependency | Medium Confidence as dependency |
| Transactions | Cash flows, income, fees, trades, corrections. | Dependency | Dependency | Medium Confidence as dependency |
| Prices | Beginning/end valuation prices and restatement analysis. | Dependency | Dependency | Medium Confidence as dependency |
| FX rates | Base/local conversion and currency effect analysis. | Unknown object | Unknown object | Unknown |
| Classifications | Sector, asset class, country, region, custom groups. | Dependency | Dependency | Medium Confidence as dependency |
| Benchmarks/index returns | Relative performance and attribution. | Unknown object | Unknown object | Unknown |

### 5.3 Candidate performance field dictionary

This table is a research checklist. It is **not** a verified Axys/APX IMEX field dictionary.

| Field | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---:|
| Portfolio code/account id | Portfolio/account identifier. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Portfolio name | Display name. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Start date | Performance period start. | Unknown | Unknown | Unknown | Unknown | Unknown |
| End date | Performance period end. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Currency code | Base/reporting/local currency. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Beginning market value | Starting valuation used in return calculation. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Ending market value | Ending valuation used in return calculation. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Net contributions | External net flows during period. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Income | Income included in period performance. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Fees | Fees included/excluded depending on gross/net options. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Portfolio return | Account-level return. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Gross return | Gross-of-fee return, if supported. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Net return | Net-of-fee return, if supported. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Benchmark code | Benchmark identifier. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Benchmark return | Benchmark return for same period. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Security identifier | Product security identifier. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Security type | Product security type used with symbol in integration contexts. | Medium Confidence as security dependency | Medium Confidence as security dependency | Unknown | Unknown | Medium Confidence |
| Security name | Security display name. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Asset class | Classification used in performance/reporting. | High Confidence as reporting category | Medium Confidence | Unknown | Likely report output | Medium Confidence |
| Sector / industry | Classification used in performance/reporting. | High Confidence as reporting category | High Confidence as report category | Unknown | Likely report output | Medium Confidence |
| Country / region | Classification used in Axys reporting. | Verified as reporting category | Unknown | Unknown | Likely report output | Medium Confidence |
| Weight | Security/classification weight used in contribution. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Security return | Security-level return. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Contribution | Security/classification contribution to portfolio return. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Local return | Local-currency return. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Currency effect | FX contribution/effect. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Accrued income | Accrual component included in performance valuation. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Run date | Report/export generation date. | Unknown | Unknown | Unknown | Unknown | Unknown |

### 5.4 IMEX cautions

| Caution | Confidence |
|---|---:|
| Do not assume `portperf` or `secperf` are official object names. | Unknown until verified |
| Do not assume Axys and APX use identical performance IMEX objects or fields. | Unknown |
| Do not assume an IMEX performance export contains stored values rather than report-calculated values. | Unknown |
| Do not assume security-level rows foot to portfolio returns without explicit contribution fields or a report reconciliation. | Unknown |
| Do not treat third-party output fields as native IMEX field names unless confirmed by IMEX documentation or actual exports. | High Confidence caution |

---

## 6. REP / Reports

### 6.1 Axys REP/report evidence

| Topic | Statement | Confidence |
|---|---|---:|
| RepLang | Axys reports are written in Advent RepLang in supplied REP research examples. | Verified |
| `.REP` files | `AMAN.REP` is identified as an Axys Assets Under Management report file in a supplied REP research example. | Verified for example |
| Custom reports | Axys standard reports can be copied and edited in a text editor in the cited example. | Verified for example |
| REP32 | A connector uses `REP32.exe`, standard reports, macros, RepLang scripting, and the REP32 engine to extract Axys/APX data. | Verified for connector |
| Performance report names | Exact Axys performance `.REP` files are not identified in supplied material. | Unknown |
| Stored vs recalculated | Whether Axys performance reports read stored values or recalculate values is Unknown. | Unknown |

### 6.2 APX report evidence

| Topic | Statement | Confidence |
|---|---|---:|
| Standard reports | APX has a large standard report library at product level. | Verified |
| SSRS/custom reporting | APX supports flexible custom reporting; supplied REP research records SSRS/SQL reporting evidence. | Verified / Medium Confidence depending source |
| REP32 connector extraction | A connector uses REP32/RepLang/macros for APX extraction in supported versions. | Verified for connector |
| APX Reports Guide | A public APX Reports Guide was identified in research; full performance report content was not fully inspected. | Medium Confidence lead |
| Performance report names | Exact APX performance report names, RDL names, and datasets are Unknown. | Unknown |
| Stored vs recalculated | Whether APX performance reports use stored values, stored accounting functions, public views, SQL procedures, or report-specific calculations is Unknown. | Unknown |

### 6.3 Report inventory to collect

| Report / file | System | Level | Needed to answer |
|---|---|---|---|
| Portfolio performance report | Axys | Portfolio | Return method, date options, gross/net, stored/recalculated behavior. |
| Security performance report | Axys | Security | Security returns, weights, contribution, cash handling, footing. |
| Classification performance report | Axys | Classification | Sector/asset-class/country/region grouping and historical classification behavior. |
| Composite/GIPS report | Axys | Composite | Composite return source and aggregation method. |
| Portfolio vs benchmark report | APX | Portfolio/benchmark | Benchmark selection, benchmark return source, relative performance. |
| Performance analytics / attribution report | APX | Security/classification | Source of sector/security performance and attribution effects. |
| Composite/GIPS report | APX | Composite | Composite support and calculation/storage behavior. |
| SSRS/RDL performance reports | APX | Multiple | Dataset/stored procedure/view dependencies. |

---

## 7. Data Model

### 7.1 Conceptual entities

These entities organize performance research. They are not asserted as Axys/APX table names.

| Entity | Purpose | Axys | APX | Confidence |
|---|---|---:|---:|---:|
| Portfolio/account | Account whose performance is calculated. | Required concept | Required concept | High Confidence |
| Security | Investment whose return/contribution may be reported. | Required concept | Required concept | High Confidence |
| Period | Performance date interval. | Required concept | Required concept | High Confidence |
| Valuation | Beginning/ending market values and possibly accruals. | Required for many return methods | Required for many return methods | Medium Confidence |
| Transactions/cash flows | External flows, income, fees, trades, corrections. | Performance dependency | Performance dependency | Medium Confidence |
| Prices | Valuation input and restatement source. | Performance dependency | Performance dependency | Medium Confidence |
| FX rates | Multi-currency performance dependency. | Unknown details | Unknown details | Unknown |
| Performance result | Portfolio/security/classification return and contribution. | Storage Unknown | Storage Unknown | Unknown |
| Benchmark/index | Benchmark returns and relative performance. | Unknown details | Product-level support | APX Verified at capability level; details Unknown |
| Classification | Asset class, sector, country, region, industry, custom grouping. | Reporting category supported | Reporting category supported | Medium / High depending category |
| Composite | Group of accounts used for composite/GIPS reporting. | Product-level capability | Product-level capability | Medium Confidence |

### 7.2 Stored vs recalculated matrix

| Output | Axys stored? | Axys recalculated? | APX stored? | APX recalculated? | Status |
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

### 7.3 Dependencies that can affect reported performance

| Dependency | How it can affect performance | Confidence |
|---|---|---:|
| Transactions | Backdated trades, cash flows, income, fees, transfers, reversals, and corrections can change valuations and cash-flow treatment. | Medium Confidence as operational risk; exact Axys/APX behavior Unknown |
| Holdings | Beginning/end holdings drive valuation and security-level exposure. | Medium Confidence |
| Cash | Cash flows and cash classification affect return calculation, contribution, and gross/net reporting. | Medium Confidence |
| Prices | Historical price changes can alter beginning/end market values and security returns. | Medium Confidence |
| Corporate actions | Splits, dividends, reinvestments, reorganizations, and price/quantity corrections can affect holdings, prices, income, and returns. | Medium Confidence |
| Security master | Security type, identifiers, currency, and asset-class information may affect report grouping and calculations. | Medium Confidence |
| Classifications | Classification changes may affect historical grouping, but current-vs-historical classification behavior is Unknown. | Unknown for Axys/APX behavior |
| Benchmarks | Benchmark revisions can affect relative performance and attribution. | APX benchmark dependency supported at high level; detailed behavior Unknown |
| FX rates | Historical FX rate changes can affect base-currency returns. | Unknown details |
| Report parameters | Gross/net, accrual, currency, consolidation, date basis, benchmark, and classification options may change output. | Medium Confidence as reporting risk; exact options Unknown |

---

## 8. Common Fields

This table follows the repository field dictionary standard. It includes observed dependencies and candidate performance fields. Most field names remain Unknown because no performance export or report field dictionary was supplied.

| Field | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---:|
| Portfolio/account identifier | Identifies account/portfolio whose performance is reported; research checklist aliases include `PortfolioCode` / account id. | Unknown exact field | Unknown exact field | Unknown | Unknown | Unknown |
| Portfolio name | Portfolio/account display name; research checklist alias `PortfolioName`. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Portfolio code | Portfolio code appears in Axys reporting examples outside performance; exact performance field name Unknown. | Yes, report context | Unknown | Unknown | Yes, in REP examples | Verified outside performance |
| Security symbol | Product security identifier; symbol/type pairing is important in integration contexts. | Yes | Yes | Unknown | Unknown | Medium Confidence |
| Security type | Product security type; not the same as asset class/classification. | Yes | Yes | Unknown | Unknown | Medium Confidence |
| Start date | Beginning date of performance period; research checklist alias `StartDate`. | Unknown | Unknown | Unknown | Unknown | Unknown |
| End date | Ending date of performance period; research checklist alias `EndDate`. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Currency code | Base, reporting, or local currency code; research checklist alias `CurrencyCode`. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Beginning market value | Valuation at period start; research checklist alias `BeginningMarketValue`. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Ending market value | Valuation at period end; research checklist alias `EndingMarketValue`. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Contributions/withdrawals | External-flow treatment derived from classified cash/security movements; research checklist alias `NetContributions`. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Income | Income included in performance. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Fees | Fees included/excluded depending on gross/net return. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Portfolio return | Account-level return; research checklist alias `PortfolioReturn`. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Gross return | Gross-of-fee return; research checklist alias `GrossReturn`. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Net return | Net-of-fee return; research checklist alias `NetReturn`. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Security identifier | Security identifier used in a performance row; research checklist alias `SecurityID`. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Security name | Security display name; research checklist alias `SecurityName`. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Security return | Security-level return; research checklist alias `SecurityReturn`. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Weight | Security or classification weight. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Contribution | Contribution to portfolio return. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Benchmark code | Benchmark identifier; research checklist alias `BenchmarkCode`. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Benchmark return | Benchmark period return; research checklist alias `BenchmarkReturn`. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Asset class | Reporting/performance grouping category in Axys product material; research checklist alias `AssetClass`. | Yes, category | Likely | Unknown | Likely report output | Medium Confidence |
| Sector | Reporting/performance grouping category. | Yes, category | Yes, report category | Unknown | Likely report output | Medium Confidence |
| Country | Axys performance display category. | Yes, category | Unknown | Unknown | Likely report output | Medium Confidence |
| Region | Axys performance display category. | Yes, category | Unknown | Unknown | Likely report output | Medium Confidence |
| Industry group | APX report-guide snippet category. | Unknown | Yes, report category | Unknown | Likely report output | Medium Confidence |
| Custom classification | APX report-guide snippet category. | Unknown | Yes, report category | Unknown | Likely report output | Medium Confidence |
| Local return | Local-currency return; research checklist alias `LocalReturn`. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Currency effect | Currency/FX effect; research checklist alias `CurrencyEffect`. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Accrued income | Accrued income included in valuation/performance; research checklist alias `AccruedIncome`. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Run date | Report/export run timestamp; research checklist alias `RunDate`. | Unknown | Unknown | Unknown | Unknown | Unknown |

---

## 9. Examples and Test Designs

Because no actual performance exports or reports were supplied, this section provides controlled test designs rather than factual examples of Axys/APX output.

### 9.1 Stored monthly returns vs recalculation test

| Step | Procedure | Evidence produced |
|---:|---|---|
| 1 | Run one-month portfolio performance report for a closed month. | Baseline one-period return and report options. |
| 2 | Run the same portfolio for each month in a 12-month range. | Monthly return series. |
| 3 | Run one 12-month performance report for the same account and date range. | Multi-period reported return. |
| 4 | Compare the 12-month report to geometric linking of monthly returns. | Evidence of linking vs full-period recalculation, subject to rounding/report options. |
| 5 | If a performance export exists, compare report values to exported/stored values. | Evidence of stored vs report-calculated behavior. |

**Status:** Test design only. Axys/APX behavior remains Unknown until executed.

### 9.2 Security performance footing test

| Step | Procedure | Evidence produced |
|---:|---|---|
| 1 | Export or run security-level performance for one account/period. | Security rows, weights, returns, contributions if available. |
| 2 | Sum any report-provided contribution field. | Contribution footing to portfolio return. |
| 3 | Independently calculate `weight * security return` for each row. | Evidence whether simple weight-return multiplication matches report return. |
| 4 | Repeat with and without cash rows if report options allow. | Cash treatment evidence. |
| 5 | Compare classification rollups to security rows grouped by classification. | Evidence whether classification performance is derived or separately calculated. |

**Status:** Test design only. Do not assume security rows foot by `weight * return`.

### 9.3 Historical performance change test

| Step | Procedure | Evidence produced |
|---:|---|---|
| 1 | Save original report/export for a closed historical period. | Baseline performance evidence. |
| 2 | Identify a specific input change: price, transaction, FX, classification, benchmark, or security master. | Controlled cause. |
| 3 | Rerun same report with identical parameters. | New performance evidence. |
| 4 | Compare old and new output. | Determines whether historical report changed. |
| 5 | Export supporting inputs before/after where possible. | Root-cause evidence. |

**Status:** Test design only. Exact Axys/APX behavior is Unknown.

### 9.4 Benchmark-relative test

| Step | Procedure | Evidence produced |
|---:|---|---|
| 1 | Run portfolio-vs-benchmark report for one period. | Portfolio return, benchmark return, relative return if available. |
| 2 | Export benchmark return source if available. | Benchmark source values. |
| 3 | Change or identify a benchmark revision. | Revision event. |
| 4 | Rerun report. | Evidence of benchmark revision impact. |

**Status:** Test design only. APX has high-level benchmark support, but detailed behavior remains Unknown.

---

## 10. Known Issues / Quirks

The following are documented as cautions or research hypotheses, not proven Axys/APX behavior unless confidence indicates otherwise.

| Issue / Quirk | Axys | APX | Confidence | Notes |
|---|---:|---:|---:|---|
| Historical performance may change after prior-period prices, transactions, income, fees, FX rates, classifications, benchmarks, or security master data are corrected. | Unknown | Unknown | Medium Confidence as general performance-system risk | Requires controlled Axys/APX rerun tests. |
| Portfolio return may not equal simple sum of security `weight * return`. | Unknown | Unknown | Medium Confidence as general performance-system risk | Security contribution fields, cash, flows, weights, fees, accruals, and rounding may matter. |
| Security annual report may differ from linked monthly security returns. | Unknown | Unknown | Medium Confidence as general performance-system risk | Requires monthly-vs-annual report comparison. |
| Classification performance may use current classifications, historical classifications, report-date classifications, or stored snapshots. | Unknown | Unknown | Unknown | High-priority test area. |
| Benchmark revisions can change relative performance. | Unknown | Unknown | Medium Confidence; APX benchmark support verified at product level | Need benchmark storage/export evidence. |
| Rounding can prevent visible report values from footing exactly. | Unknown | Unknown | Medium Confidence as report risk | Need raw precision export. |
| Different reports may use different gross/net, accrual, cash, currency, trade/settlement, consolidation, or benchmark defaults. | Unknown | Unknown | Medium Confidence as report risk | Need report parameter inventory. |
| REP/report extraction may be report-layout-sensitive. | Yes | Yes | Medium Confidence | Supplied REP research supports REP as report-driven extraction; layout/version control is important. |
| Direct Axys file access is version-sensitive. | Yes | N/A | Medium Confidence | Supplied IMEX/classification research records consultant warnings about file-format changes. |
| APX has multiple reporting/data-access paths. | N/A | Yes | Medium Confidence | SQL/SSRS/Public Views/Stored Accounting Functions/REST/API references require local validation. |

---

## 11. Version Differences

| Version / product area | Axys | APX | Status | Evidence needed |
|---|---|---|---|---|
| Legacy vs current Axys performance behavior | Unknown | N/A | Unknown | Axys release notes, performance manuals, installed report files. |
| Axys 3.8.6 / 3.8.7 report changes | Connector/release evidence exists outside performance specifics. | N/A | Partial | Report files and release notes. |
| APX performance analytics module changes | N/A | Unknown | Unknown | APX release notes/performance analytics manuals. |
| APX SSRS report changes across versions | N/A | Unknown | Unknown | APX Reports Guide by version and RDL catalog. |
| Axys vs APX IMEX performance objects | Unknown | Unknown | Unknown | Paired IMEX catalogs/exports. |
| Gross/net or fee treatment changes | Unknown | Unknown | Unknown | Versioned performance documentation. |
| Multi-currency performance changes | Unknown | Unknown | Unknown | Versioned multi-currency examples and release notes. |
| Fixed-income/accrual performance behavior | Unknown | Unknown | Unknown | Fixed-income report examples, pricing/accrual documentation. |

---

## 12. References

This chapter is based only on supplied repository research and source summaries. The following supplied files were used:

| Supplied file | Use in this chapter |
|---|---|
| `AXYS_APX_REFERENCE_BLUEPRINT.md` | Governing chapter structure, confidence labels, and facts-first standard. |
| `Research_10_Performance.md` | Primary performance research. |
| `Research_12_IMEX.md` | IMEX, REP32, file/interface, direct-file-access, and import/export cautions. |
| `Research_13_REP.md` | REP, RepLang, Report Writer Pro, REP32, SSRS/reporting-path evidence. |
| `Research_04_Security_Master.md` | Security identity, symbol/type, security master dependencies. |
| `Research_05_Transactions.md` | Transaction dependencies, performance restatement risk, transaction correction implications. |
| `Research_06_Holdings.md` | Holdings/report valuation dependencies. |
| `Research_07_Cash.md` | Cash, cash-flow, and performance/cash treatment unknowns. |
| `Research_08_Pricing.md` | Price files, stale/missing/calculated prices, price restatement implications. |
| `Research_09_Corporate_Actions.md` | Corporate-action effects on holdings/prices/transactions/performance. |
| `Research_11_Classifications.md` | Classification reporting categories and historical-classification unknowns. |

External source names and URLs are preserved in the supplied research files rather than repeated as authoritative direct citations here.

---

## 13. Unknowns

These Unknowns should remain open until supported by vendor documentation, production examples, REP source files, IMEX exports, APX schema/public views, or controlled tests.

| ID | Unknown | Why it matters | Evidence required |
|---|---|---|---|
| PERF-U-001 | Exact Axys performance file/table/object names. | Needed for implementation reference. | Axys data dictionary, IMEX catalog, or sample files. |
| PERF-U-002 | Exact APX performance SQL tables/views/procedures. | Needed for implementation reference. | APX schema/public views/stored accounting functions. |
| PERF-U-003 | Exact IMEX object for portfolio performance. | Needed for export guidance. | IMEX catalog and sample export. |
| PERF-U-004 | Exact IMEX object for security performance. | Needed for security-level analytics/export. | IMEX catalog and sample export. |
| PERF-U-005 | Exact Axys `.REP` files for performance reports. | Needed for REP chapter cross-reference and report behavior. | Axys REP folder inventory and report source. |
| PERF-U-006 | Exact APX SSRS/RDL reports for performance. | Needed for APX report documentation. | APX report catalog/RDL files. |
| PERF-U-007 | Whether portfolio returns are stored, recalculated, or both. | Core behavior question. | Stored values plus rerun tests. |
| PERF-U-008 | Whether security returns/contributions are stored or report-calculated. | Core reconciliation question. | Export/report comparison. |
| PERF-U-009 | Whether multi-period reports link stored periods or recalculate over the full range. | Explains historical changes and month-vs-year differences. | Monthly-vs-annual tests. |
| PERF-U-010 | Which return methodology each report uses. | Prevents incorrect interpretation. | Report documentation and controlled examples. |
| PERF-U-011 | Gross vs net fee treatment. | Required for client/composite reporting. | Report options and fee transaction examples. |
| PERF-U-012 | Accrued income treatment. | Important for fixed income and income-heavy portfolios. | Report options and test portfolio. |
| PERF-U-013 | Cash treatment in security performance. | Needed for contribution footing. | Security report examples with cash rows. |
| PERF-U-014 | Classification timing/historical treatment. | Needed for sector/country performance and attribution. | Classification history tests. |
| PERF-U-015 | Benchmark data storage and revision behavior. | Needed for relative performance. | Benchmark export and revision tests. |
| PERF-U-016 | Composite/GIPS support details. | Needed for composite performance section. | Composite report docs and examples. |
| PERF-U-017 | APX attribution methodologies and fields. | Needed for attribution documentation. | APX performance analytics/attribution guide. |
| PERF-U-018 | Multi-currency local/base/currency-effect fields. | Needed for global portfolios. | Multi-currency report/export samples. |
| PERF-U-019 | Fixed-income performance treatment, including accrued income and calculated prices. | Needed for bonds and income portfolios. | Fixed-income report examples and pricing/accrual docs. |
| PERF-U-020 | Whether report outputs reconcile to IMEX outputs. | Needed for integration design and audit. | Paired report and IMEX extracts. |

---

## 14. Minimum Additional Evidence Needed

To upgrade this chapter from conservative reference to implementation manual, collect the following:

| Evidence | Purpose |
|---|---|
| Axys performance report samples | Confirm report names, fields, parameters, and output semantics. |
| Axys `.REP` source for performance reports | Confirm RepLang fields, calculations, stored vs recalculated behavior. |
| APX Reports Guide / APX report catalog | Identify APX performance report names and parameters. |
| APX SSRS/RDL files or dataset definitions | Identify report datasets, stored procedures, public views, and fields. |
| IMEX object catalog for Axys and APX | Confirm portfolio/security performance object names and field lists. |
| Sample IMEX performance exports | Confirm actual field names, grain, identifiers, and stored/exported values. |
| Same-period rerun examples | Determine whether historical performance changes after data corrections. |
| Monthly vs annual report examples | Determine linking vs full-period recalculation. |
| Security performance report with contribution | Determine security-row footing behavior. |
| Classification performance examples | Determine classification source and historical/current treatment. |
| Benchmark/index export or report | Determine benchmark data source and revision behavior. |
| Multi-currency performance examples | Determine local/base/currency-effect support. |
| Composite/GIPS report examples | Determine composite aggregation/storage/report behavior. |

## 15. Deep IMEX Update

The deep IMEX research confirms that performance remains a report/extract
boundary area rather than a solved IMEX object dictionary.

| Topic | Chapter treatment | Confidence |
|---|---|---:|
| Performance history | Public conversion evidence mentions performance history as an IMEX/migration concern, but object names and fields remain Unknown. | Medium / Unknown |
| `portperf` / `secperf` | Treat as normalized/local names unless a live IMEX object, report output, or vendor manual confirms native names. | Unknown |
| Candidate live-discovery fields | Portfolio/security/classification/composite IDs, period dates, beginning/ending market value, return, contribution, external flow, income, fees, weights, benchmark, and currency fields. | Discovery guidance |
| REP/report preference | Use REP/Replang/custom reports when reported performance must tie to user-visible Axys/APX output. | Design guidance |
| Extraction metadata | Record source object/report, parameters, version, row lineage, and stored-vs-recalculated confidence. | Design guidance |
