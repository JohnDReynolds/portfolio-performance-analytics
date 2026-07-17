# Research Notes: Reports

**Repository chapter target:** `../reference/Chapter_14_Reports.md`  
**Research file:** `docs/axys_apx/evidence/Research_14_Reports.md`
**Prepared:** 2026-06-29  
**Governing specification:** `axys_apx_reference_blueprint.md`, Version 2.0

---

## 1. Research Scope

This research file collects factual and implementation-oriented information for the repository chapter on **Reports** in SS&C Axys and SS&C APX.

The governing repository standard requires that important technical statements be classified as:

- **Verified**
- **High Confidence**
- **Medium Confidence**
- **Unknown**

This research file intentionally preserves Unknowns where report behavior, field names, report internals, or version-specific differences are not supported by available source material.

---

## 2. Source Basis and Evidence Standard

| Source | Type | Applies To | Use in This Research | Confidence Impact |
|---|---:|---|---|---|
| `axys_apx_reference_blueprint.md`, Version 2.0 | Repository specification | Axys/APX repository | Governing editorial standard, required repository structure, required treatment of Unknowns | Verified |
| SS&C / Advent Axys public product page | Vendor public product page | Axys | Confirms Axys positioning as portfolio accounting/reporting software with reporting capabilities | Verified for public product claims only |
| SS&C / Advent APX public product page | Vendor public product page | APX | Confirms APX positioning as integrated portfolio/accounting/reporting solution | Verified for public product claims only |
| Advent Portfolio Exchange Reports Guide, public PDF/search-indexed copy | Vendor report guide | APX | Confirms a library of APX report examples and several report names/visible output columns | Verified for visible report-guide content only |
| Consultant articles / pages discussing Axys/APX reporting and REP | Consultant documentation | Axys/APX | Supports existence of custom reporting, Report Writer Pro, and Replang/REP customization practices | High Confidence unless directly quoting vendor material |
| Public secondary software listings | Third-party summaries | Axys/APX | Used only as supporting context, not as sole evidence for implementation details | Medium Confidence |
| Production REP source, IMEX exports, report outputs | Not supplied | Axys/APX | Needed for exact field dictionaries and report internals | Unknown until supplied |

---

## 3. Executive Findings

| Finding | Axys | APX | Confidence | Notes |
|---|---|---|---|---|
| Reports are a core functional area of both products. | Yes | Yes | Verified | Supported by vendor/public product descriptions and repository blueprint. |
| Axys supports predefined reports and custom report work. | Yes | Unknown exact catalog | High Confidence | Public material says Axys has predefined reports and easy/custom report options; exact report names require Axys report documentation or installed report library. |
| APX has a documented investment-management report guide with named report examples. | Not applicable | Yes | Verified | APX report guide content lists named reports and visible output columns. |
| APX reports in the public guide are built on Microsoft SQL Server Reporting Services (SSRS). | Unknown for Axys | Yes for guide-covered APX reports | Verified | Public APX guide states this for investment management reports. |
| Axys/APX users can create/customize reports using Report Writer Pro or by modifying Replang source code. | Yes | Yes | High Confidence | Consultant material supports this; exact product/version boundaries require vendor docs. |
| REP is relevant to report customization and/or report execution in Axys/APX. | Yes | Yes | High Confidence | Blueprint includes REP as a repository topic; consultant sources discuss Replang/REP. Exact syntax and report object behavior belong in Chapter 13 unless report-specific. |
| IMEX is not itself a report engine, but report research must distinguish IMEX exports from human/client-facing reports. | Yes | Yes | High Confidence | Blueprint separates IMEX, REP, and Reports. Exact overlap with report datasets is Unknown without samples. |
| Exact Axys report file names, REP source file names, report parameters, and output fields are not verified from supplied materials. | Unknown | Unknown | Unknown | Requires installed report library, REP source, screenshots, or generated report samples. |
| Exact APX report stored procedures, SSRS datasets, RDL names, parameters, and database dependencies are not verified. | Not applicable | Unknown | Unknown | Requires APX report deployment or report project files. |

---

## 4. Terminology

| Term | Description | Axys | APX | Confidence |
|---|---|---:|---:|---|
| Report | A human-readable or client-facing output generated from portfolio/accounting/performance data. | Yes | Yes | High Confidence |
| REP | Repository term for Advent report-language/report-related artifacts. Often associated with Replang in consultant usage. | Yes | Yes | High Confidence |
| Replang | Advent report language/source-code layer referenced by consultants for Axys/APX report customization. | Yes | Yes | High Confidence |
| Report Writer Pro | Custom reporting tool referenced in consultant material for Axys/APX. | Yes | Yes | High Confidence |
| SSRS | Microsoft SQL Server Reporting Services. Public APX report guide says APX investment management reports are built on SSRS. | Unknown | Yes | Verified for APX guide-covered reports |
| IMEX | Import/export facility covered separately in the repository. Distinct from client-facing reports. | Yes | Yes | Verified as repository topic; detailed behavior Unknown here |
| Report package | A bundled set of reports, commonly used for client reporting. | Yes | Yes | Medium Confidence |
| Compound report | A multi-part report package/report object referenced in public custom-report examples. | Yes | Possibly | Medium Confidence |
| Client reporting | Reports intended for presentation to clients, including overview, allocation, performance, risk, fixed income, and disclosure-style outputs. | Yes | Yes | High Confidence |
| Business intelligence report | Internal/management report category shown in APX report guide. | Unknown | Yes | Verified for APX guide |
| Performance analytics report | Report category shown in APX guide, including attribution, contribution, and risk reports. | Unknown | Yes | Verified for APX guide |

---

## 5. Axys Report Research

### 5.1 Axys Report Behavior

| Statement | Classification | Evidence / Notes |
|---|---:|---|
| Axys is positioned publicly as portfolio management/accounting software with portfolio reporting capabilities. | Verified | Vendor/public product material. |
| Axys includes an extensive library of predefined reports and supports report customization. | Verified for public claim; exact implementation Unknown | Vendor/public and consultant material state this generally. |
| Axys custom reports may be produced by modifying existing reports or creating custom report output. | High Confidence | Supported by consultant material describing Axys custom reports. |
| Axys custom report work may involve Report Writer Pro. | High Confidence | Consultant material references Report Writer Pro in Axys/APX contexts. |
| Axys custom report work may involve direct changes to Replang source code. | High Confidence | Consultant material references Replang source-code changes for Axys/APX reporting. |
| Axys report behavior may depend on report parameters, report source, installed version, local customizations, performance settings, and data state. | Medium Confidence | Common implementation reality; not verified by supplied Axys-specific report source. |
| Exact Axys report names are not verified from supplied material. | Unknown | Need Axys standard report catalog, report menu export, REP source tree, or sample reports. |
| Exact Axys report output fields are not verified from supplied material. | Unknown | Need sample reports or REP source. |
| Whether a given Axys report uses stored performance values or recalculates values at run time is report-specific and not verified here. | Unknown | Need report source and processing documentation. |
| Whether Axys reports read directly from `.cli`, IMEX objects, performance stores, price files, security master, or intermediate report tables is report-specific and not verified here. | Unknown | Need report source and implementation documentation. |

### 5.2 Axys Report Customization

| Topic | Research Finding | Confidence | Required Follow-up Evidence |
|---|---|---:|---|
| Add portfolio code to reports | Public consultant material discusses adding portfolio code to an Axys report such as an AUM report. | High Confidence | The actual article/report source is needed to verify exact field names and steps. |
| Custom reports | Axys custom report development is a known consultant service area. | High Confidence | Need report source, output sample, and parameter list for repository-grade documentation. |
| Report Writer Pro | Referenced as a custom reporting tool for Axys. | High Confidence | Need vendor/user guide for exact capabilities, supported versions, and syntax. |
| REP/Replang source edits | Referenced by consultant material as an Axys/APX reporting path. | High Confidence | Need sample `.rep`/Replang files or report source tree. |
| Report packaging | Axys report packaging exists in public custom-report examples. | Medium Confidence | Need installed package configuration or screenshots to document accurately. |

### 5.3 Axys Report Categories

The following categories are reasonable research buckets for Axys reports, but the exact standard report names are **Unknown** until the Axys report catalog or installed report library is supplied.

| Category | Example Report Purpose | Axys Standard Report Names | Confidence |
|---|---|---:|---|
| Holdings / positions | Position listing, holdings by asset class, holdings by security | Unknown | Unknown |
| Transactions / activity | Transaction register, realized gain/loss, cash activity | Unknown | Unknown |
| Performance | Portfolio performance, performance summary, period returns | Unknown | Unknown |
| Security performance | Security-level returns/contribution | Unknown | Unknown |
| Classification / allocation | Allocation by asset class, industry, country, sector | Unknown | Unknown |
| Income | Income received/accrued, dividends, interest | Unknown | Unknown |
| Cash | Cash balances, cash activity | Unknown | Unknown |
| Pricing | Price listings, missing prices, stale prices | Unknown | Unknown |
| Corporate actions | Split/dividend/corporate-action reporting | Unknown | Unknown |
| Management / AUM | Assets under management, account/portfolio summary | Unknown | Medium Confidence category only |
| Client presentation | Client package reports | Unknown | Medium Confidence category only |

### 5.4 Axys Fields Commonly Expected in Reports

The following table is a **research checklist**, not a verified Axys field dictionary. Names shown are generic business labels, not verified Axys REP/IMEX/database field names.

| Business Concept | Possible Report Label | Exact Axys Field Name | Confidence |
|---|---|---:|---|
| Portfolio identifier | Portfolio / Account / Portfolio Code | Unknown | Unknown |
| Portfolio name | Account name / Portfolio name | Unknown | Unknown |
| Report start date | From date / Beginning date | Unknown | Unknown |
| Report end date | To date / Ending date | Unknown | Unknown |
| Security identifier | Symbol / Cusip / Security ID | Unknown | Unknown |
| Security name | Description / Security name | Unknown | Unknown |
| Quantity | Quantity / Shares / Par | Unknown | Unknown |
| Market value | Market Value | Unknown | Unknown |
| Cost | Cost / Book value / Tax cost | Unknown | Unknown |
| Price | Price / Market price | Unknown | Unknown |
| Weight | Weight / Percent of portfolio | Unknown | Unknown |
| Return | Return | Unknown | Unknown |
| Contribution | Contribution | Unknown | Unknown |
| Income | Income / Dividends / Interest | Unknown | Unknown |
| Classification | Asset class / Industry / Sector / Country | Unknown | Unknown |

---

## 6. APX Report Research

### 6.1 APX Report Behavior

| Statement | Classification | Evidence / Notes |
|---|---:|---|
| APX is publicly positioned as an integrated CRM and portfolio management solution connecting front/middle/back-office functions. | Verified | Vendor/public APX product material. |
| APX includes portfolio accounting and reporting capabilities. | Verified | Vendor/public APX product material. |
| APX investment-management reports in the public Advent Portfolio Exchange Reports Guide are built on Microsoft SQL Server Reporting Services (SSRS). | Verified | Public APX report guide. |
| APX report guide examples include internal/business intelligence reports, portfolio-manager analytics, and client-reporting reports. | Verified | Public APX report guide table of contents. |
| APX reports may include charts, graphs, drill-down/expand behavior, and branded presentation output. | Verified for guide examples | Public APX guide describes and shows these features. |
| Exact APX SSRS report filenames, RDL names, stored procedures, query datasets, and parameter names are not verified. | Unknown | Need APX report deployment files or SSRS catalog extract. |
| Exact APX database tables/views used by reports are not verified. | Unknown | Need report SQL, database documentation, or query capture. |
| Whether APX reports use stored performance values, calculated performance tables, report datasets, or live calculation services is report-specific and not verified here. | Unknown | Need report source and APX performance-processing documentation. |

### 6.2 APX Report Categories and Publicly Visible Report Names

The public APX report guide provides examples grouped broadly as business intelligence, portfolio-manager analytics, and client reporting. The following report names are visible in public guide material.

| Category | Report Name | Report Purpose / Visible Behavior | Confidence |
|---|---|---|---:|
| Business intelligence | Account Distribution | Segments business by AUM, revenue contribution, and client tenure. | Verified |
| Business intelligence | Account Characteristics | Account/client-characteristic reporting. Exact fields Unknown. | Verified name; details Unknown |
| Business intelligence | Account Characteristics (By Custodian) | Account characteristics grouped by custodian. Exact fields Unknown. | Verified name; details Unknown |
| Business intelligence | Asset Flows | Asset-flow reporting. Exact fields Unknown. | Verified name; details Unknown |
| Business intelligence | Business Summary Dashboard | Dashboard-style business summary. Exact fields Unknown. | Verified name; details Unknown |
| Portfolio analytics | Activity Profile | Portfolio activity profile. Exact fields Unknown. | Verified name; details Unknown |
| Portfolio analytics | Attribution by Classification | Shows attribution effects by classification and compares portfolio to benchmark. | Verified |
| Portfolio analytics | Attribution Summary | Shows portfolio attribution overview including portfolio return, benchmark return, active return, allocation effect, selection effect, and total effect. | Verified |
| Portfolio analytics | Attribution by Selected Groupings | Shows attribution effects for reporting segments and supports expansion/drill-down to lower levels where data is available. | Verified |
| Portfolio analytics | Contribution by Classification | Compares contribution by portfolio segments. | Verified |
| Portfolio analytics | Contribution Summary | Shows segment contribution to total portfolio performance and may be run with or without a benchmark. | Verified |
| Portfolio analytics | Contribution Detail | Shows detailed flattened contribution output rather than nested grouping output. | Verified |
| Portfolio analytics | Risk Statistics | Risk-statistics report. Exact metrics Unknown. | Verified name; details Unknown |
| Client reporting | Cover Page | Client package cover page. Exact fields Unknown. | Verified name; details Unknown |
| Client reporting | Household Overview | Household-level overview. Exact fields Unknown. | Verified name; details Unknown |
| Client reporting | Portfolio Overview | Portfolio overview. Exact fields Unknown. | Verified name; details Unknown |
| Client reporting | Performance Overview | Performance overview. Exact fields Unknown. | Verified name; details Unknown |
| Client reporting | Risk Overview | Risk overview. Exact fields Unknown. | Verified name; details Unknown |
| Client reporting | Policy Overview | Investment policy overview. Exact fields Unknown. | Verified name; details Unknown |
| Client reporting | Historical Policy Overview | Historical policy overview. Exact fields Unknown. | Verified name; details Unknown |
| Client reporting | Allocation Summary | Allocation summary. Exact fields Unknown. | Verified name; details Unknown |
| Client reporting | Equity Overview | Equity overview. Exact fields Unknown. | Verified name; details Unknown |
| Client reporting | Fixed Income Distribution | Fixed-income distribution. Exact fields Unknown. | Verified name; details Unknown |
| Client reporting | Disclaimer and Terms | Disclosure/disclaimer section. Exact wording Unknown. | Verified name; details Unknown |

### 6.3 APX Attribution / Contribution Visible Output Fields

The following fields or column labels are visible in the public APX report-guide examples. These are **report labels**, not necessarily database, IMEX, or REP field names.

| Report Label / Field | Report Context | Meaning | APX | Axys | Confidence |
|---|---|---|---:|---:|---:|
| Portfolio Return | Attribution Summary / Contribution Summary | Total portfolio return for report period | Visible in APX example | Unknown | Verified |
| Benchmark Return | Attribution Summary / Contribution Summary | Benchmark return for report period | Visible in APX example | Unknown | Verified |
| Active Return | Attribution Summary / Contribution Summary | Difference between portfolio and benchmark return | Visible in APX example | Unknown | Verified |
| Allocation Effect | Attribution Summary / attribution reports | Attribution allocation component | Visible in APX example | Unknown | Verified |
| Selection Effect | Attribution Summary / attribution reports | Attribution selection component | Visible in APX example | Unknown | Verified |
| Total Effect | Attribution Summary / attribution reports | Total attribution effect | Visible in APX example | Unknown | Verified |
| Industry Sector | Attribution/contribution grouping | Classification/grouping label | Visible in APX example | Unknown | Verified |
| Security | Attribution/contribution detail | Security-level grouping/detail | Visible in APX example | Unknown | Verified |
| Avg Wgt | Attribution/contribution detail | Average weight | Visible in APX example | Unknown | Verified |
| Return | Attribution/contribution detail | Return for portfolio, benchmark, or segment | Visible in APX example | Unknown | Verified |
| Contrib | Attribution/contribution detail | Contribution | Visible in APX example | Unknown | Verified |
| Portfolio | Report section | Portfolio-side columns | Visible in APX example | Unknown | Verified |
| Benchmark | Report section | Benchmark-side columns | Visible in APX example | Unknown | Verified |
| Difference | Report section | Portfolio minus benchmark comparison columns | Visible in APX example | Unknown | Verified |
| Top Contributors | Contribution/attribution summary | Ranking section for positive contribution | Visible in APX example | Unknown | Verified |
| Bottom Contributors | Contribution/attribution summary | Ranking section for negative contribution | Visible in APX example | Unknown | Verified |
| Top Attribution Effects | Attribution summary | Ranking section by attribution effect | Visible in APX example | Unknown | Verified |
| Bottom Attribution Effects | Attribution summary | Ranking section by attribution effect | Visible in APX example | Unknown | Verified |
| Largest Weights | Attribution/contribution summary | Ranking section by average weight | Visible in APX example | Unknown | Verified |

### 6.4 APX Report Parameters and Processing

| Topic | Research Finding | Confidence |
|---|---|---:|
| Date range | APX examples show reports run over a from/to period. Exact parameter names Unknown. | Verified visible behavior; parameter names Unknown |
| Portfolio and benchmark | APX examples show portfolio-vs-benchmark reporting. Exact object IDs and parameter names Unknown. | Verified visible behavior; parameter names Unknown |
| Classification grouping | APX examples show industry-sector/classification grouping. Exact classification source Unknown. | Verified visible behavior; data source Unknown |
| Drill-down / expansion | APX selected-groupings example shows expandable groupings down to lower levels where data exists. | Verified |
| Security-level output | APX selected-groupings and summary examples show security-level rows/sections. | Verified |
| Benchmark detail | APX examples show benchmark weights, returns, and contribution. Exact benchmark data source Unknown. | Verified visible behavior; source Unknown |
| Report engine | APX public guide says guide-covered investment reports use SSRS. | Verified |
| Query / dataset layer | Exact SQL, stored procedures, APX views, or data services used by reports are Unknown. | Unknown |
| Performance calculation source | Whether values are pulled from stored performance, calculated dynamically, or sourced from performance-analytics stores is Unknown. | Unknown |
| Currency handling | Multi-currency reporting behavior is not verified by guide snippets. | Unknown |
| Composite / household behavior | Household Overview exists; exact aggregation behavior Unknown. | Verified report existence; behavior Unknown |

---

## 7. IMEX and Reports

### 7.1 IMEX vs Reports

| Statement | Classification | Notes |
|---|---:|---|
| IMEX and Reports are distinct repository topics. | Verified | Blueprint lists separate chapters for IMEX, REP, and Reports. |
| IMEX exports are better treated as machine-readable data extracts, while reports are better treated as human/client/internal presentation outputs. | High Confidence | Repository structure and product usage imply distinct purposes. |
| Some report outputs may contain values that can also be exported or approximated through IMEX. | Medium Confidence | Common practical overlap, but exact objects/fields not verified. |
| IMEX should not be assumed to reproduce a report exactly without validating calculation method, parameters, and source fields. | High Confidence | Important repository caution; exact behavior is report-specific. |
| Exact IMEX objects feeding or matching each report are Unknown. | Unknown | Need IMEX catalog, report source, and sample outputs. |

### 7.2 Report-to-IMEX Reconciliation Checklist

For each report documented in the chapter, the research should eventually collect:

| Item | Why It Matters | Current Status |
|---|---|---:|
| Report name | Allows unambiguous reference | Known for selected APX guide reports; Unknown for Axys catalog |
| Report engine | REP/Replang, Report Writer Pro, SSRS, other | APX guide: SSRS for guide-covered reports; Axys Unknown |
| Report source file | Needed for reproducibility and field mapping | Unknown |
| Parameter list | Dates, portfolio, benchmark, grouping, currency, fees, etc. | Unknown |
| Source-data | Database tables, IMEX objects, performance stores, price/security master | Unknown |
| Output fields | Labels and definitions | Partially known for APX attribution/contribution examples |
| Calculation method | Stored vs recalculated, time-weighted vs other, linked vs point-to-point | Unknown |
| Version | Axys/APX version affects behavior | Unknown |
| Local customizations | Reports are commonly customized | Unknown |
| Reconciliation method | How to compare report to IMEX/exported values | Unknown |

---

## 8. REP and Reports

### 8.1 REP / Replang Role

| Statement | Axys | APX | Confidence | Notes |
|---|---:|---:|---:|---|
| REP/Replang is relevant to reporting. | Yes | Yes | High Confidence | Blueprint includes REP; consultant material references Replang reporting. |
| Report Writer Pro can be used for report customization. | Yes | Yes | High Confidence | Consultant material supports this generally. |
| Direct Replang source-code edits can be used for Axys/APX reports. | Yes | Yes | High Confidence | Consultant material supports this generally. |
| REP syntax is documented in this research file. | No | No | Unknown | Belongs primarily in Chapter 13; no source supplied here. |
| Exact REP variables/field names for Chapter 14 are known. | No | No | Unknown | Need REP source samples. |

### 8.2 REP Documentation Needs for Chapter 14

To produce a complete technical chapter, collect at least one representative REP/custom report source file for each major report family:

| Report Family | Needed Evidence | Status |
|---|---|---:|
| Holdings | REP source + output sample + parameter screen | Unknown |
| Transactions/activity | REP source + output sample + parameter screen | Unknown |
| Performance | REP source + output sample + parameter screen | Unknown |
| Security performance | REP source + output sample + parameter screen | Unknown |
| Attribution/contribution | REP/SSRS source + output sample + parameter screen | Unknown |
| Classification/allocation | REP source + output sample + parameter screen | Unknown |
| AUM/management | REP source + output sample + parameter screen | Unknown |
| Client reporting package | Package definition + component reports + sample PDF | Unknown |

---

## 9. Report Data Model Research

### 9.1 Generic Report Metadata Model

This table is a proposed repository research structure. It is not an Axys/APX physical schema.

| Field | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---:|
| `system` | `Axys` or `APX` | Yes | Yes | No | No | Repository convention |
| `report_name` | User-facing report name | Unknown exact catalog | Partially known | No | Maybe | Medium |
| `report_family` | Holdings, performance, attribution, client reporting, etc. | Unknown | Partially known | No | Maybe | Medium |
| `report_engine` | REP/Replang, Report Writer Pro, SSRS, other | Unknown | SSRS for guide reports | No | Yes if REP | Medium |
| `source_artifact` | REP file, RDL file, report definition, package component | Unknown | Unknown | No | Yes if REP | Unknown |
| `parameters` | Report runtime inputs | Unknown | Unknown | No | Yes/SSRS | Unknown |
| `portfolio_scope` | Portfolio, group, household, composite, benchmark | Unknown | Partially visible | Possible | Possible | Medium |
| `date_scope` | As-of date or from/to period | Unknown | Visible in examples | Possible | Possible | Medium |
| `classification_scope` | Asset class, sector, country, industry, custom group | Unknown | Industry sector visible | Possible | Possible | Medium |
| `output_fields` | Labels/columns emitted by report | Unknown | Partially visible | No | Yes/SSRS | Medium |
| `calculation_method` | Stored/recalculated/linked/point-to-point | Unknown | Unknown | No | Maybe | Unknown |
| `version_notes` | Version-specific behavior | Unknown | Unknown | No | Maybe | Unknown |
| `quirks` | Known behavior, exceptions, reconciliation issues | Unknown | Unknown | No | Maybe | Unknown |

### 9.2 Report Field Dictionary Standard

The repository blueprint requires this field dictionary format:

| Field | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---|---|---|---|---|
| Unknown | Exact report fields must be collected from samples/source before being documented as system fields. | Unknown | Unknown | Unknown | Unknown | Unknown |

---

## 10. Report Families to Document in Chapter 14

### 10.1 Holdings / Position Reports

| Research Question | Axys | APX | Confidence |
|---|---:|---:|---:|
| Standard report names | Unknown | Unknown from APX guide snippets except portfolio/equity overview categories | Unknown |
| Standard columns | Unknown | Unknown | Unknown |
| As-of vs period behavior | Unknown | Unknown | Unknown |
| Tax-lot visibility | Unknown | Unknown | Unknown |
| Cash treatment | Unknown | Unknown | Unknown |
| Price/date source | Unknown | Unknown | Unknown |
| Security classification source | Unknown | Unknown | Unknown |

### 10.2 Transaction / Activity Reports

| Research Question | Axys | APX | Confidence |
|---|---:|---:|---:|
| Standard report names | Unknown | Activity Profile visible for APX guide | APX name Verified; details Unknown |
| Transaction-code fields | Unknown | Unknown | Unknown |
| Trade date vs settle date behavior | Unknown | Unknown | Unknown |
| Cash effect presentation | Unknown | Unknown | Unknown |
| Corporate-action presentation | Unknown | Unknown | Unknown |
| Cancel/reversal handling | Unknown | Unknown | Unknown |

### 10.3 Performance Reports

| Research Question | Axys | APX | Confidence |
|---|---:|---:|---:|
| Portfolio performance report names | Unknown | Performance Overview visible in APX guide | APX name Verified; details Unknown |
| Security-level performance report names | Unknown | Security-level rows visible in APX contribution/attribution examples | Medium |
| Stored vs recalculated values | Unknown | Unknown | Unknown |
| Linking method | Unknown | Unknown | Unknown |
| Periodicity | Unknown | Unknown | Unknown |
| Fee gross/net handling | Unknown | Unknown | Unknown |
| Currency/local/base handling | Unknown | Unknown | Unknown |
| Composite behavior | Unknown | Unknown | Unknown |

### 10.4 Attribution Reports

| Research Question | Axys | APX | Confidence |
|---|---:|---:|---:|
| Standard attribution report names | Unknown | Attribution by Classification; Attribution Summary; Attribution by Selected Groupings | APX Verified |
| Report-level portfolio return | Unknown | Visible | APX Verified |
| Report-level benchmark return | Unknown | Visible | APX Verified |
| Allocation effect | Unknown | Visible | APX Verified |
| Selection effect | Unknown | Visible | APX Verified |
| Total effect | Unknown | Visible | APX Verified |
| Classification grouping | Unknown | Industry Sector visible | APX Verified |
| Drill-down to security level | Unknown | Visible in selected grouping example where data exists | APX Verified |
| Attribution formula | Unknown | Unknown | Unknown |
| Arithmetic vs geometric attribution | Unknown | Unknown | Unknown |
| Multi-period attribution method | Unknown | Unknown | Unknown |

### 10.5 Contribution Reports

| Research Question | Axys | APX | Confidence |
|---|---:|---:|---:|
| Standard contribution report names | Unknown | Contribution by Classification; Contribution Summary; Contribution Detail | APX Verified |
| Can run without benchmark | Unknown | Public guide says Contribution Summary can be run without benchmark | APX Verified |
| Columns | Unknown | Avg Wgt, Return, Contrib visible in examples | APX Verified for labels |
| Nested vs flattened views | Unknown | Contribution Detail described as flattened view | APX Verified |
| Classification source | Unknown | Unknown | Unknown |
| Contribution formula | Unknown | Unknown | Unknown |

### 10.6 Risk Reports

| Research Question | Axys | APX | Confidence |
|---|---:|---:|---:|
| Standard risk report names | Unknown | Risk Statistics; Risk Overview | APX names Verified |
| Metrics included | Unknown | Unknown | Unknown |
| Benchmark support | Unknown | Unknown | Unknown |
| Periodicity | Unknown | Unknown | Unknown |
| Source returns | Unknown | Unknown | Unknown |

### 10.7 Client Reports / Packages

| Research Question | Axys | APX | Confidence |
|---|---:|---:|---:|
| Cover pages | Unknown | Cover Page visible in APX guide | APX Verified |
| Household reporting | Unknown | Household Overview visible in APX guide | APX Verified |
| Portfolio overview | Unknown | Portfolio Overview visible in APX guide | APX Verified |
| Allocation summary | Unknown | Allocation Summary visible in APX guide | APX Verified |
| Equity overview | Unknown | Equity Overview visible in APX guide | APX Verified |
| Fixed-income distribution | Unknown | Fixed Income Distribution visible in APX guide | APX Verified |
| Disclaimers/terms | Unknown | Disclaimer and Terms visible in APX guide | APX Verified |
| Branding/logos | Unknown | APX guide says reports can be branded | APX Verified |
| Package definition mechanics | Unknown | Unknown | Unknown |

---

## 11. Version Differences

| Topic | Axys | APX | Confidence |
|---|---|---|---:|
| Report engine | Unknown; likely REP/Replang/Report Writer Pro in many environments | APX guide-covered reports use SSRS; Replang/Report Writer Pro also referenced by consultants | Medium |
| Standard report catalog | Unknown | Partially known from public APX guide | Medium |
| Customization layer | Report Writer Pro/Replang referenced | SSRS for guide reports; Report Writer Pro/Replang referenced | Medium |
| Database-backed reports | Unknown | Likely for SSRS reports, but exact datasets Unknown | Medium |
| Web/browser report execution | Unknown | APX public guide references access through APX report page | Verified for guide claim |
| Report packaging | Medium Confidence based on custom-report examples | Unknown | Medium |
| Performance analytics integration | Unknown | APX public material references performance analytics; guide includes attribution/contribution/risk reports | High Confidence |

---

## 12. Known Issues / Quirks to Validate

These are not verified defects. They are research targets that should be confirmed from production observations, report source, or test cases.

| Potential Issue / Quirk | Why It Matters | Axys | APX | Current Confidence |
|---|---|---:|---:|---:|
| Report output may not reconcile to IMEX extracts without matching parameters and calculation method. | Prevents false reconciliation errors. | Unknown | Unknown | High Confidence as caution |
| Stored monthly performance may differ from report-calculated longer-period performance. | Important for performance reports and audit tools. | Unknown | Unknown | Unknown |
| Classification reports may use classifications as-of a report date, transaction date, or current master state. | Affects historical attribution/allocation. | Unknown | Unknown | Unknown |
| Security-level contribution may not sum exactly to portfolio return because of rounding, cash, fees, residuals, derivatives, or calculation method. | Affects reconciliation. | Unknown | Unknown | Medium Confidence as generic risk; not Axys/APX verified |
| Benchmarks may use different data sources than portfolio data. | Affects attribution and contribution reports. | Unknown | Unknown | Medium Confidence |
| Custom reports may differ materially from standard reports even if names are similar. | Avoids overgeneralization. | Unknown | Unknown | High Confidence |
| Report package output may hide component report parameters. | Important for reproducibility. | Unknown | Unknown | Medium Confidence |
| APX SSRS report labels may not match database field names. | Prevents wrong field dictionary entries. | Not applicable | Unknown but likely | Medium Confidence |
| Axys Replang labels may not match IMEX field names. | Prevents wrong field dictionary entries. | Unknown | Unknown | Medium Confidence |
| Upgrades may overwrite or break customized reports. | Operational risk. | Unknown | Unknown | Medium Confidence; needs vendor/consultant confirmation |

---

## 13. Examples

### 13.1 APX Attribution Summary Example — Research Abstraction

**Source basis:** Public APX report guide example.

| Section | Visible Items | Confidence |
|---|---|---:|
| Report heading | Portfolio vs benchmark over a date range | Verified |
| Summary | Portfolio Return, Benchmark Return, Active Return, Allocation Effect, Selection Effect, Total Effect | Verified |
| Largest weights | Security, Avg Wgt, Return, Contrib | Verified |
| Industry sector attribution | Top/bottom attribution effects with Allocation, Selection, Total | Verified |
| Industry sector contribution | Top/bottom contributors with Avg Wgt, Return, Contrib | Verified |
| Security attribution | Top/bottom attribution effects by security | Verified |
| Security contribution | Top/bottom contributors by security | Verified |
| Formula used | Unknown | Unknown |
| Source tables/views | Unknown | Unknown |
| Exact SSRS dataset names | Unknown | Unknown |

### 13.2 APX Contribution Summary Example — Research Abstraction

**Source basis:** Public APX report guide example.

| Section | Visible Items | Confidence |
|---|---|---:|
| Performance summary | Portfolio Return, Benchmark Return, Active Return | Verified |
| Largest weights | Security, Avg Wgt, Return, Contrib | Verified |
| Classification contribution | Top/bottom performers and contributors | Verified |
| Security contribution | Top/bottom performers and contributors | Verified |
| Benchmark optionality | Report may be run without benchmark for absolute performance focus | Verified |
| Formula used | Unknown | Unknown |
| Data source | Unknown | Unknown |

### 13.3 Axys Custom AUM Report Example — Research Abstraction

**Source basis:** Public consultant article title/snippet about adding portfolio code to an Axys AUM report.

| Item | Research Finding | Confidence |
|---|---|---:|
| Standard/custom report area | Assets Under Management report customization exists as a public consultant example | High Confidence |
| Customization objective | Add portfolio code to report output | High Confidence |
| Exact field name for portfolio code | Unknown | Unknown |
| Exact report source file | Unknown | Unknown |
| Exact implementation steps | Unknown | Unknown |
| Whether this applies to all Axys versions | Unknown | Unknown |

---

## 14. Required Source Material for a Stronger Chapter

The current research is useful for a skeleton and for APX report-guide-derived report names, but it is not enough for a complete technical reference chapter with exact field dictionaries and processing rules.

To strengthen `Chapter_14_Reports.md`, collect the following:

| Priority | Needed Material | Why Needed |
|---:|---|---|
| 1 | Axys standard report catalog or screenshots of report menu | Verify Axys report names and categories |
| 1 | APX report catalog from installed environment | Verify report names beyond public APX guide |
| 1 | Sample Axys report outputs for holdings, transactions, performance, contribution, AUM, allocation | Build field dictionary and behavior notes |
| 1 | Sample APX report outputs for same categories | Build APX field dictionary and behavior notes |
| 1 | REP/Replang source files for representative Axys reports | Identify exact fields, parameters, source-data, and logic |
| 1 | APX SSRS RDL files or SSRS catalog export | Identify exact datasets, parameters, labels, grouping, and queries |
| 1 | IMEX exports corresponding to same report periods/portfolios | Document report-to-IMEX reconciliation |
| 2 | Vendor user guides for Axys Report Writer Pro / Replang | Document syntax and supported customization |
| 2 | Vendor APX reporting/admin guide | Document APX SSRS deployment/version behavior |
| 2 | Known customized report examples | Document local customization risks |
| 3 | Production observations of report runtimes and reconciliation exceptions | Document known quirks |

---

## 15. Repository Chapter Boundary Recommendation

Chapter 14 should remain organized around:

1. Overview
2. Reporting Concepts
3. Axys Reports
4. APX Reports
5. Report Engines
   - Axys REP/Replang / Report Writer Pro
   - APX SSRS reports
   - APX/Axys REP overlap where verified
6. Report Families
   - Holdings
   - Transactions
   - Performance
   - Attribution
   - Contribution
   - Risk
   - Allocation / Classification
   - AUM / Business Intelligence
   - Client Reporting Packages
7. IMEX vs Reports
8. REP vs Reports
9. Field Dictionary
10. Examples
11. Known Issues / Quirks
12. Version Differences
13. References
14. Unknowns

---

## 16. Unknowns Register

| Unknown | Axys | APX | How to Resolve |
|---|---:|---:|---|
| Complete standard report list | Unknown | Partially known | Provide installed report catalog / vendor guide |
| Exact report file names | Unknown | Unknown | Provide REP/RDL/report source tree |
| Report parameter names | Unknown | Unknown | Provide report definitions or screenshots |
| Report source tables/views/files | Unknown | Unknown | Provide report source/query definitions |
| Stored vs recalculated performance behavior | Unknown | Unknown | Provide performance report source and vendor docs |
| Report-to-IMEX equivalence | Unknown | Unknown | Provide matching report output and IMEX extracts |
| Gross/net-of-fee handling | Unknown | Unknown | Provide report parameters/source and samples |
| Currency handling | Unknown | Unknown | Provide multicurrency sample reports |
| Classification as-of behavior | Unknown | Unknown | Provide classification history examples and report source |
| Benchmark data source and matching logic | Unknown | Unknown | Provide benchmark master/source and attribution report definitions |
| Rounding and residual treatment | Unknown | Unknown | Provide detailed output and calculation source |
| Composite/household aggregation rules | Unknown | Unknown | Provide composite/household report definitions |
| Upgrade/customization behavior | Unknown | Unknown | Provide vendor admin guide or production notes |

---

## 17. References

### 17.1 Governing Repository Source

1. `axys_apx_reference_blueprint.md`, Version 2.0. Defines repository purpose, editorial standards, confidence labels, standard chapter template, field dictionary format, and required repository structure.

### 17.2 Public Vendor / Product Sources

2. SS&C Advent Axys public product page. Publicly describes Axys as portfolio reporting/accounting software with predefined and customizable reporting capabilities.  
   URL: `https://www.advent.com/solutions/axys/`

3. SS&C Advent Portfolio Exchange public product page. Publicly describes APX as an integrated client relationship management and portfolio management solution with portfolio accounting and reporting capabilities.  
   URL: `https://www.advent.com/solutions/advent-portfolio-exchange/`

4. Advent Portfolio Exchange Reports Guide. Public PDF/search-indexed report guide showing APX report categories, report names, SSRS basis for guide-covered reports, and example report output labels.  
   URL: `https://cdn.advent.com/cms/pdfs/reports/REP_APX.pdf`

5. Index Data for Advent Portfolio Exchange product brief. Public product brief referencing APX reports and benchmark/model-portfolio-relative reporting.  
   URL: `https://cdn.advent.com/cms/pdfs/briefs/PB_INDATA.pdf`

### 17.3 Consultant / Secondary Sources

6. AdventGuru category/tag pages for REP, Axys, and APX. Consultant-authored material referencing Report Writer Pro, Replang source code, and APX API/reporting contexts.  
   URLs:  
   - `https://adventguru.com/tag/rep/`  
   - `https://adventguru.com/category/portfolio-management-systems/axys/`  
   - `https://adventguru.com/category/portfolio-management-systems/apx/`

7. Consultant article PDF: “How to Add Portfolio Code to Axys Reports.” Public consultant material showing an example customization request for an Axys report.  
   URL: `https://assets.ctfassets.net/xhy36q2d1lqu/77QC4aNbyhPo9FfmjRYNzc/d00a0d6601214601543e30e34f203626/PortfolioCodetoAxys.pdf`

8. Hedgeweek article: “Axys Report Writer Pro - Creating reports to your exact specifications.” Public article referencing Axys Report Writer Pro.  
   URL: `https://www.hedgeweek.com/axys-report-writer-pro-creating-reports-your-exact-specifications/`

9. Salentica / Elements Data Broker pages for SS&C Advent APX & Axys. Public third-party integration pages referencing Axys/APX reporting capabilities.  
   URLs:  
   - `https://elements.salentica.com/kb/article/252-data-broker-ss-c-advent-apx-axys/`  
   - `https://engage.salentica.com/kb/article/247-data-broker-ss-c-advent-apx-axys/`

---

## 18. Research Conclusion

The available material supports a useful but incomplete Chapter 14 research base.

**Verified material is strongest for APX public report examples**, especially report names in the public APX report guide and visible labels in attribution/contribution reports.

**Axys report details remain largely Unknown** beyond the existence of predefined/customizable reporting and public consultant examples of custom report work.

A complete technical reference chapter requires installed-system evidence: Axys report catalogs, APX SSRS/RDL definitions, REP/Replang source, report parameter screenshots, sample outputs, and matching IMEX extracts.

## Deep Research Update Incorporated 2026-07-02

The July 2026 addendum strengthens Axys report examples and APX report-label
coverage. CSSI evidence verifies an Axys Portfolio Appraisal Report Writer
workflow with `Portfolio Code`, `Management Mode`, and `$askport`; an Axys
Transaction Summary customization lead using `$:tfile`; and an AUM by Sector
customization that copies `aman.rep` to `_aumsect.rep`, uses sector values that
match the sector file, and uses `$firmg` as an Other catch-all. These are
report/customization examples, not native schemas or complete report catalogs.

APX indexed Reports Guide text strengthens business-intelligence report labels
and dimensions, including Account Distribution, AUM Distribution, Revenue
Distribution, Effective Rate, Strategy, Product Line, Account Manager,
Salesperson, Consultant, Custodian, Location, and Tax Status. Attribution and
contribution evidence adds report labels/sections such as Portfolio Return,
Benchmark Return, Active Return, Allocation Effect, Selection Effect, Total
Effect, Largest Weights, Top/Bottom Attribution Effects, Top/Bottom
Contributors, Portfolio/Benchmark/Difference groups, Avg Wgt, Return, Contrib,
Alloc, Select, Industry Sector, and Security.

Contribution Detail is described as a flattened most-detailed-level view.
APX Portfolio Appraisal is strengthened as a report behavior lead for holdings
by individual tax lot or position. CSSI evidence adds APX SSRS package drift,
delivery exceptions, pre-cycle validation, hosted/cloud package updates, and
portal delivery as operational/report-production cautions. APX RDL names,
datasets, stored procedures, report formulas, report-to-IMEX equivalence, and
stored-versus-recalculated behavior remain Unknown.

## Market Value vs Accrued Interest Report Addendum Incorporated 2026-07-08

Source: `temp_Axys_APX_Market_Value_vs_Accrued_Interest_Summary.md`.

The July 2026 market-value/accrued-interest research adds a specific Portfolio
Appraisal report interpretation: public SS&C samples and a public APX Portfolio
Appraisal example show fixed-income Market Value and Accrued Interest as
separate report concepts. The layout strongly implies that Portfolio Appraisal
total/subtotal presentation combines market value and accrued interest, while
Market Value itself remains a clean-value concept.

This is strong report-output evidence, but it is not yet an official IMEX,
database, public-view, or stored-field specification. Follow-up evidence should
collect official report formulas, IMEX fields for accrued interest, and any
site-specific report definitions that distinguish clean market value, accrued
interest, and total position value.
