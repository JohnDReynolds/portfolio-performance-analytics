# Chapter 14 — Reports

**Repository:** AXYS / APX Reference Repository
**Chapter file:** `Chapter_14_Reports.md`
**Prepared:** 2026-06-29
**Governing specification:** `AXYS_APX_REFERENCE_BLUEPRINT.md`, Version 2.0
**Source basis:** Supplied research files only, especially `Research_14_Reports.md`, with cross-reference to supplied architecture, holdings, transactions, performance, classifications, IMEX, and REP research.

---

## Related chapters
- [Chapter_01_Overview.md](Chapter_01_Overview.md) — provides the repository-wide map, evidence conventions, and shared safe implementation rules.
- [Chapter_10_Performance.md](Chapter_10_Performance.md) — many performance outputs are surfaced through reports.
- [Chapter_11_Classifications.md](Chapter_11_Classifications.md) — report families often use classification labels.
- [Chapter_13_Rep.md](Chapter_13_Rep.md) — reports often depend on REP or report-source definitions.

## 1. Overview

Reports are a core functional area of both Axys and APX. In this repository, a **report** means a human-readable, client-facing, management-facing, or analyst-facing output generated from portfolio accounting, holdings, transaction, performance, classification, benchmark, or relationship data.

This chapter intentionally separates:

Reports are downstream views of the accounting data model. They may summarize transactions, cash, holdings, performance, and classifications, but a report label such as "Asset Flows" or "Activity Profile" is not proof of the underlying transaction semantics. For implementation and audit work, report outputs should be traced back to the originating transactions and normalized flow classifications rather than treated as a self-describing source of truth.

| Topic | Description | Primary Chapter |
|---|---|---|
| Reports | User-facing and analyst-facing report outputs, report families, report labels, report behavior, and report reconciliation cautions. | Chapter 14 |
| REP / Replang | Advent report source, `.REP` files, RepLang expressions, REP32, Report Writer Pro, macros, report automation mechanics. | Chapter 13 |
| IMEX | Import/export utility behavior and machine-readable structured data movement. | Chapter 12 |
| APX SSRS | APX report framework where reports are implemented through Microsoft SQL Server Reporting Services. | Chapter 14, with details in APX architecture chapters |

### 1.1 Confidence Standard

| Classification | Meaning |
|---|---|
| Verified | Directly supported by supplied research based on vendor material, report guide content, specific sample/report evidence, or directly supplied repository specification. |
| High Confidence | Strongly supported by consultant documentation, integration documentation, or multiple consistent research sources, but not fully proven from official vendor technical manuals. |
| Medium Confidence | Plausible and supported by partial evidence; requires confirmation from vendor manuals, installed report libraries, REP/RDL source, or sample outputs. |
| Unknown | Not established by the supplied material. Do not implement or document as fact without more evidence. |

### 1.2 High-Level Findings

| Finding | Axys | APX | Confidence |
|---|---:|---:|---:|
| Reports are a major product capability. | Yes | Yes | Verified |
| Axys has predefined reports and supports report customization. | Yes | Not Axys-specific | Verified at product level |
| APX has a documented public investment-management report guide with named report examples. | Not applicable | Yes | Verified |
| APX guide-covered investment reports are built on Microsoft SQL Server Reporting Services. | Unknown | Yes | Verified |
| Report Writer Pro / Replang / REP are relevant to Axys reporting and to some APX extraction/custom reporting workflows. | Yes | Yes | High Confidence |
| REP-based extraction may be used by connectors for both Axys and APX. | Yes | Yes | Verified for connector workflows |
| IMEX and Reports are distinct interfaces. | Yes | Yes | Verified by repository structure |
| Exact Axys standard report catalog is not supplied. | Unknown | Not applicable | Unknown |
| Exact APX SSRS RDL names, datasets, stored procedures, and database dependencies are not supplied. | Not applicable | Unknown | Unknown |
| Whether any specific report uses stored values or recalculates values is report-specific and not established unless stated below. | Unknown | Unknown | Unknown |

---

## 2. Reporting Concepts

### 2.1 Report Purposes

| Report Purpose | Description | Axys | APX | Confidence |
|---|---|---:|---:|---:|
| Client reporting | Presentation reports for clients, households, portfolios, allocations, performance, risk, fixed income, and disclosure sections. | Yes | Yes | High Confidence |
| Management reporting | Internal reporting such as assets under management, account distribution, business summary, revenue, account characteristics, and asset flows. | Yes | Yes | Medium / Verified for named APX examples |
| Operations reporting | Transaction, holdings, reconciliation, cash, pricing, blotter, and exception outputs. | Yes | Yes | High Confidence |
| Performance analytics | Portfolio, benchmark, contribution, attribution, and risk reporting. | Yes | Yes | Verified at product level; APX names Verified |
| Data extraction | Report output used as an integration/export surface. | Yes | Yes | High Confidence |

### 2.2 Report Output Is Not Necessarily Source-data

Report labels are not automatically native field names. A report column such as `Market Value`, `Contrib`, `Avg Wgt`, or `Portfolio Return` may be:

| Possibility | Status |
|---|---:|
| A native database or file field | Unknown unless source documentation confirms it |
| A calculated value produced by report logic | Unknown unless report source confirms it |
| A label renamed for presentation | High Confidence as a common reporting possibility |
| A value read from stored performance or accounting results | Unknown unless report source or vendor documentation confirms it |
| A value derived from IMEX-exportable data | Unknown unless reconciled to an IMEX export |

**Repository rule:** Do not map report labels directly to Axys files, APX tables, IMEX fields, or REP variables unless the supporting source material explicitly proves that mapping.

---

## 3. Axys Reports

### 3.1 Axys Report Architecture Summary

| Area | Axys Behavior | Confidence | Notes |
|---|---|---:|---|
| Product reporting | Axys is publicly positioned as portfolio accounting/reporting software. | Verified | Product-level claim only. |
| Predefined reports | Axys has predefined reports. | Verified | Exact full catalog not supplied. |
| Custom reports | Axys supports custom report work. | Verified / High Confidence | Report Writer Pro and Replang evidence support custom work. |
| Report Writer Pro | Axys supports Report Writer Pro. | Verified | Product and REP research. |
| REP/Replang | Axys reports are written in RepLang in the supplied REP research examples. | Verified for cited examples | Full grammar and field dictionary not supplied. |
| `.REP` files | Axys report files such as `AMAN.REP` are supported by supplied REP research. | Verified for examples | Do not infer full report catalog. |
| Report source paths | Example paths such as `e:\axys34\rep` or `\axys3\rep` appear in supplied consultant examples. | Verified for examples only | Installation path is site-specific. |
| Report calculations | Whether values are stored or recalculated is report-specific and Unknown. | Unknown | Requires report source, vendor docs, or controlled tests. |

### 3.2 Verified / Supported Axys Report Names and Files

The supplied research does not include a full Axys report catalog. The following are the report names or files supported by supplied material.

| Report / File | System | Description | Confidence | Caveat |
|---|---|---|---:|---|
| `Portfolio Appraisal` | Axys | Holdings/assets point-in-time report; can be generated in Report Writer. | Verified | Exact standard report file name not supplied. |
| `AMAN.REP` | Axys | Assets Under Management report file in consultant customization example. | Verified for example | Do not assume every installation uses identical file/path/customization. |
| `AMAN_XX.REP` | Axys | Example copied/customized version of `AMAN.REP`. | Verified for example | Example name only. |
| `aman.rep` | Axys | Lowercase spelling of AUM report file in research examples. | Verified for example | Case handling by filesystem/version Unknown. |
| `_aumsect.rep` | Axys | User-created copy of `aman.rep` in an AUM-by-sector example. | Verified for example | Not a standard vendor report name. |
| `CDIhold.rep` | Axys | WealthTechs-provided historical holdings calculation report in AIA/NBIN workflow. | Verified for workflow | Third-party workflow artifact. |
| `Transaction Summary` customization lead | Axys | CSSI evidence includes a transaction summary customization lead using `$:tfile`. | Verified for example | Exact standard file/report name Unknown. |
| `Reconciliation report` | Axys | Used in a Morningstar conversion process to compare Axys to custodian records as of last transaction date. | Verified for conversion workflow | Exact report name/file Unknown. |
| `Position Reconciliation report` | Axys | Enhanced in Axys 3.8.7 according to supplied REP research. | Verified at named-report level | File name and fields Unknown. |
| `Multicurrency reports` | Axys | Axys 3.8.7 included additional/improved multicurrency reports. | Verified at product release level | Exact report names Unknown. |

### 3.3 Axys Portfolio Appraisal

| Behavior / Field | Description | Confidence |
|---|---|---:|
| Report existence | Axys has a `Portfolio Appraisal` report. | Verified |
| Holdings purpose | Used as a holdings/assets point-in-time report. | Verified / High Confidence |
| Consolidated group behavior | A consolidated group produces one Portfolio Appraisal showing assets for the entire group in the supplied CSSI example. | Verified for example |
| Unconsolidated group behavior | An unconsolidated group normally produces multiple portfolio appraisals, one per member, in the supplied CSSI example. | Verified for example |
| Management Mode | In the supplied CSSI example, Management Mode can produce a single combined appraisal for an unconsolidated group. | Verified for example |
| Owner portfolio code | `Portfolio Code` can be added as a Portfolio Appraisal column in the supplied Report Writer example. | Verified |
| Sample labels | `Quantity`, `Security`, `Price`, `Market Value`, `Pct Assets`, `Yield`, `Portfolio Code`. | Verified for sample |

### 3.4 Axys AUM / Management Reports

| Item | Description | Confidence |
|---|---|---:|
| `AMAN.REP` | Assets Under Management report source file in supplied consultant example. | Verified for example |
| Portfolio code customization | The supplied REP research supports adding portfolio code to the Axys AUM report using Replang customization. | Verified for example |
| `$:fileo` | Replang variable used in the AUM example to display portfolio code. | Verified for example |
| `#~8portmv` | Replang expression used in the AUM example to print portfolio market value. | Verified for example |
| AUM by sector customization | CSSI evidence copies `aman.rep` to `_aumsect.rep`, uses sector-file values, and uses `$firmg` as an Other catch-all. | Verified for example |
| Report menu path | Research mentions Axys Reports / Mgmt / Assets Under Management in an example. | Verified for example; version-specific |

### 3.5 Axys Report Categories — Current Evidence Status

| Report Family | Evidence Status | Known Axys Report Names / Files | Confidence |
|---|---|---|---:|
| Holdings / positions | Supported by `Portfolio Appraisal` and `CDIhold.rep` examples. | `Portfolio Appraisal`, `CDIhold.rep` | Verified for examples |
| Transactions / activity | Axys transaction reports likely exist, but exact standard names are not supplied. | Unknown | Unknown |
| Cash | Cash-specific report names not supplied. | Unknown | Unknown |
| Pricing | Price/missing-price/stale-price report names not supplied. | Unknown | Unknown |
| Performance | Product-level performance reporting is supported; exact report names not supplied. | Unknown | Unknown |
| Security performance | Not verified in supplied report research. | Unknown | Unknown |
| Classification / allocation | Product-level reporting by asset class/sector/country/region is supported; exact reports not supplied. | Unknown | Medium / Unknown |
| AUM / management | AUM report examples supplied. | `AMAN.REP`, Assets Under Management | Verified for example |
| Reconciliation | Conversion/reconciliation report examples supplied. | `Reconciliation report`, `Position Reconciliation report` | Verified name-level / workflow-level |
| Multicurrency | Release research mentions multicurrency reports. | Unknown | Verified as release category; names Unknown |
| Client package reports | Likely supported by product reporting, but exact names not supplied. | Unknown | Unknown |

---

## 4. APX Reports

### 4.1 APX Report Architecture Summary

| Area | APX Behavior | Confidence | Notes |
|---|---|---:|---|
| Product reporting | APX is publicly positioned as an integrated portfolio/accounting/reporting/client-management platform. | Verified | Product-level claim. |
| Standard report library | APX has a large standard report library. | Verified | Public product material and report guide. |
| APX investment-management report guide | Public guide lists multiple report names and examples. | Verified | Applies to guide-covered reports. |
| Report engine | Guide-covered APX investment-management reports are built on Microsoft SQL Server Reporting Services. | Verified | Exact RDL/dataset names not supplied. |
| Custom reporting | APX supports flexible custom reporting, report packaging, and dashboards. | Verified | Product-level claim. |
| Report operations | APX SSRS package drift, delivery exceptions, pre-cycle validation, hosted/cloud package updates, and portal delivery are operational cautions. | Medium Confidence | Operational/admin context; RDLs, datasets, formulas, and source equivalence remain Unknown. |
| REP/Replang overlap | Connector and consultant evidence shows REP32/Replang can be used in some APX extraction/reporting workflows. | Verified for connector; Medium/High generally | Exact version boundaries Unknown. |
| SQL/database access | APX users may have SQL/reporting access paths in consultant/integration evidence. | Medium Confidence | Exact public views, stored procedures, datasets Unknown. |

### 4.2 APX Report Categories and Names

The following APX report names are supported by the supplied `Research_14_Reports.md` and APX architecture/report research. These are report names or guide-visible names; they are not database table names.

| Category | Report Name | Description / Visible Purpose | Confidence |
|---|---|---|---:|
| Business intelligence | Account Distribution | Segments business by AUM, revenue contribution, and client tenure. | Verified |
| Business intelligence | AUM Distribution | Business-intelligence label/dimension from indexed APX Reports Guide evidence. | Verified label |
| Business intelligence | Revenue Distribution | Business-intelligence label/dimension from indexed APX Reports Guide evidence. | Verified label |
| Business intelligence | Effective Rate | Business-intelligence label/dimension from indexed APX Reports Guide evidence. | Verified label |
| Business intelligence dimensions | Strategy / Product Line / Account Manager / Salesperson / Consultant / Custodian / Location / Tax Status | Business-intelligence dimensions from indexed APX Reports Guide evidence. | Verified labels |
| Business intelligence | Account Characteristics | Account/client-characteristic reporting. Exact fields Unknown. | Verified name; details Unknown |
| Business intelligence | Account Characteristics (By Custodian) | Account characteristics grouped by custodian. Exact fields Unknown. | Verified name; details Unknown |
| Business intelligence | Asset Flows | Asset-flow reporting. Exact fields Unknown. | Verified name; details Unknown |
| Business intelligence | Business Summary Dashboard | Dashboard-style business summary. Exact fields Unknown. | Verified name; details Unknown |
| Portfolio analytics | Activity Profile | Portfolio activity profile. Exact fields Unknown. | Verified name; details Unknown |
| Portfolio analytics | Attribution by Classification | Attribution effects by classification; compares portfolio and benchmark. | Verified |
| Portfolio analytics | Attribution Summary | Portfolio attribution overview including return/effect summary. | Verified |
| Portfolio analytics | Attribution by Selected Groupings | Attribution by selected reporting segments with expansion/drill-down where data exists. | Verified |
| Portfolio analytics | Contribution by Classification | Contribution by portfolio segments/classifications. | Verified |
| Portfolio analytics | Contribution Summary | Segment contribution to total portfolio performance; can be run with or without benchmark. | Verified |
| Portfolio analytics | Contribution Detail | Detailed flattened contribution output. | Verified |
| Portfolio analytics | Risk Statistics | Risk-statistics report. Exact metrics Unknown. | Verified name; details Unknown |
| Client reporting | Cover Page | Client package cover page. | Verified name; details Unknown |
| Client reporting | Household Overview | Household-level overview. | Verified name; details Unknown |
| Client reporting | Portfolio Overview | Portfolio overview. | Verified name; details Unknown |
| Client reporting | Performance Overview | Performance overview. | Verified name; details Unknown |
| Client reporting | Risk Overview | Risk overview. | Verified name; details Unknown |
| Client reporting | Policy Overview | Policy overview. | Verified name; details Unknown |
| Client reporting | Historical Policy Overview | Historical policy overview. | Verified name; details Unknown |
| Client reporting | Allocation Summary | Allocation summary. | Verified name; details Unknown |
| Client reporting | Equity Overview | Equity overview. | Verified name; details Unknown |
| Client reporting | Fixed Income Distribution | Fixed-income distribution. | Verified name; details Unknown |
| Client reporting | Fixed Income Overview | Fixed-income overview. | Verified name; details Unknown |
| Client reporting | Income Projection | Income projection. | Verified name; details Unknown |
| Client reporting / holdings | Portfolio Appraisal | Portfolio holdings by tax lot or position according to supplied APX holdings research. | Medium Confidence |
| Client reporting / tax | Realized Gains and Losses | Realized gain/loss report. | Verified name |
| Client reporting / activity | Transaction Summary | Account transaction reporting. | Verified name |
| Client reporting | Disclaimer and Terms | Disclosure/disclaimer section. | Verified name; details Unknown |

### 4.3 APX Attribution Reports

| Report | Supported Behavior / Labels | Confidence | Unknowns |
|---|---|---:|---|
| Attribution Summary | Displays portfolio-vs-benchmark attribution overview; visible labels include `Portfolio Return`, `Benchmark Return`, `Active Return`, `Allocation Effect`, `Selection Effect`, and `Total Effect`. | Verified | Attribution formula, arithmetic/geometric method, source tables, stored-vs-calculated behavior. |
| Attribution by Classification | Shows attribution effects by classification. | Verified | Classification source, classification-as-of behavior, detailed formula. |
| Attribution by Selected Groupings | Shows attribution by selected report segments and can expand/drill down to lower levels where data exists. | Verified | Exact grouping parameter names, drilldown hierarchy source, security-level source. |

### 4.4 APX Contribution Reports

| Report | Supported Behavior / Labels | Confidence | Unknowns |
|---|---|---:|---|
| Contribution Summary | Shows contribution to performance; may be run with or without benchmark. Visible labels include `Avg Wgt`, `Return`, and `Contrib`. | Verified | Contribution formula, averaging method, rounding/residual treatment. |
| Contribution by Classification | Compares contribution by portfolio segments/classifications. | Verified | Classification source, date behavior, grouping hierarchy. |
| Contribution Detail | Presents flattened contribution output rather than nested grouping output. | Verified | Exact columns and export format. |

### 4.5 APX Transaction and Holdings Reports

| Report | Supported Behavior / Labels | Confidence | Unknowns |
|---|---|---:|---|
| Transaction Summary | Displays account transactions maintained by Advent; supplied research observes columns/sections such as `Trade Date`, `Settle Date`, `Quantity`, `Symbol`, `Security`, `Unit Price`, and `Amount` in examples. | Verified / Medium depending field | Exact report parameters, transaction code visibility, posted-vs-blotter inclusion. |
| Portfolio Appraisal | APX report guide snippet says it shows holdings by individual tax lot or position. | Medium Confidence | Exact guide page, fields, parameters, tax-lot behavior. |
| Realized Gains and Losses | Report name verified in APX guide research. | Verified name | Exact fields and lot-selection behavior. |

### 4.6 APX Report Labels Observed in Supplied Research

These are report labels, not proven database fields.

| Label | Report Context | Description | Axys | APX | Confidence |
|---|---|---|---:|---:|---:|
| Portfolio Return | Attribution / Contribution examples | Portfolio return over report period. | Unknown | Yes | Verified |
| Benchmark Return | Attribution / Contribution examples | Benchmark return over report period. | Unknown | Yes | Verified |
| Active Return | Attribution / Contribution examples | Portfolio minus benchmark return. | Unknown | Yes | Verified |
| Allocation Effect | Attribution examples | Attribution allocation component. | Unknown | Yes | Verified |
| Selection Effect | Attribution examples | Attribution selection component. | Unknown | Yes | Verified |
| Total Effect | Attribution examples | Total attribution effect. | Unknown | Yes | Verified |
| Industry Sector | Attribution / contribution grouping | Classification/grouping label. | Unknown | Yes | Verified |
| Security | Attribution/contribution/transaction/holding examples | Security-level row or security description. | Unknown | Yes | Verified as label |
| Avg Wgt | Attribution/contribution examples | Average weight. | Unknown | Yes | Verified |
| Return | Attribution/contribution examples | Segment/security/portfolio/benchmark return. | Unknown | Yes | Verified |
| Contrib | Contribution examples | Contribution value. | Unknown | Yes | Verified |
| Portfolio | Report section | Portfolio-side columns. | Unknown | Yes | Verified |
| Benchmark | Report section | Benchmark-side columns. | Unknown | Yes | Verified |
| Difference | Report section | Portfolio minus benchmark comparison. | Unknown | Yes | Verified |
| Top Contributors | Attribution/contribution summaries | Positive contribution ranking section. | Unknown | Yes | Verified |
| Bottom Contributors | Attribution/contribution summaries | Negative contribution ranking section. | Unknown | Yes | Verified |
| Top Attribution Effects | Attribution summary | Positive attribution effect ranking. | Unknown | Yes | Verified |
| Bottom Attribution Effects | Attribution summary | Negative attribution effect ranking. | Unknown | Yes | Verified |
| Largest Weights | Attribution/contribution summary | Largest average-weight ranking section. | Unknown | Yes | Verified |
| Trade Date | Transaction Summary example | Transaction trade date label. | Unknown | Yes | Verified as report label |
| Settle Date | Transaction Summary example | Transaction settlement date label. | Unknown | Yes | Verified as report label |
| Quantity | Transaction / Portfolio Appraisal examples | Quantity label. | Yes in Axys sample | Yes | Verified as report label |
| Unit Price | Transaction Summary example | Transaction unit price label. | Unknown | Yes | Verified as report label |
| Amount | Transaction Summary example | Transaction amount label. | Unknown | Yes | Verified as report label |
| Market Value | Holdings / business examples | Valuation measure. | Yes in Axys sample | Yes | Verified as label |

---

## 5. IMEX and Reports

### 5.1 Distinction

| Topic | IMEX | Reports | Confidence |
|---|---|---|---:|
| Primary purpose | Machine-readable data import/export. | Human/client/internal presentation and analysis output. | High Confidence |
| Configuration | IMEX object/field definitions, import/export utility, logs. | REP source, Report Writer Pro definitions, SSRS/RDL definitions, macros, report packages. | High Confidence |
| Output shape | Object-oriented structured records. | Report-shaped tables, sections, headings, charts, subtotals, packages. | High Confidence |
| Calculation behavior | May export stored or generated objects depending IMEX object; exact behavior often Unknown. | May read stored values or calculate at run time; exact behavior report-specific. | High Confidence as caution |
| Reconciliation | Must compare only after aligning portfolio scope, date range, currency, classifications, benchmark, gross/net options, and calculation method. | Same. | High Confidence |

### 5.2 Report-to-IMEX Reconciliation Checklist

For each report that must reconcile to IMEX or downstream tooling, capture the following metadata:

| Item | Why It Matters | Current Status |
|---|---|---:|
| Report name | Unambiguous reference. | Partially known for APX; limited for Axys. |
| Report family | Holdings, performance, attribution, transaction, risk, etc. | Partially known. |
| Report engine | REP/Replang, Report Writer Pro, SSRS, other. | APX guide: SSRS; Axys examples: REP/Replang. |
| Report source artifact | `.REP`, `.RPW`, macro, RDL, package definition. | Mostly Unknown. |
| Runtime parameters | Portfolio/group, date range, benchmark, classification, currency, fee basis. | Mostly Unknown. |
| Output fields | Column labels, sections, hierarchy, totals. | Partially known for APX examples and Axys Portfolio Appraisal sample. |
| Source-data | Native files, APX SQL, public views, stored accounting functions, performance stores, IMEX objects. | Unknown. |
| Calculation method | Stored vs recalculated; linked vs point-to-point; gross/net; benchmark alignment. | Unknown. |
| Version | Report behavior and available reports may differ by version. | Mostly Unknown. |
| Local customization | Custom reports may diverge from vendor standards. | Unknown for installed sites. |

### 5.3 Report Output Should Not Be Treated as IMEX

| Caution | Confidence |
|---|---:|
| A report export to Excel/CSV is still a report output unless it is explicitly an IMEX object export. | High Confidence |
| Report-export column names may be presentation labels rather than data dictionary fields. | High Confidence |
| Report totals may include grouping, sorting, rounding, residuals, or formatting not present in IMEX extracts. | Medium Confidence |
| Report parameters may hide assumptions that are explicit in IMEX object fields or not represented at all. | Medium Confidence |

---

## 6. REP and Reports

### 6.1 Axys REP / Replang Relationship

| Statement | Confidence | Notes |
|---|---:|---|
| Axys reports can be represented as `.REP` files written in RepLang. | Verified for supplied examples | Full report catalog not supplied. |
| Axys standard reports should be copied before modification in supplied consultant examples. | Verified for example | Good operational practice; not necessarily vendor policy. |
| Axys custom reports can be run through Custom / Any Report in supplied example. | Verified for example | Exact menu may vary by version. |
| Replang variables and expressions such as `$:fileo` and `#~8portmv` are supported only as sample evidence. | Verified for sample | Do not infer full grammar. |

### 6.2 APX REP / Replang Relationship

| Statement | Confidence | Notes |
|---|---:|---|
| APX guide-covered investment reports use SSRS. | Verified | Applies to those guide-covered reports. |
| REP32/Replang may be used in APX extraction workflows by third-party connectors. | Verified for connector | Not proof that all APX reports are REP. |
| Consultant research says Replang remains part of APX reporting architecture. | Medium Confidence | Exact version boundaries Unknown. |
| APX may also expose SQL Server Reporting Services, public views, stored accounting functions, REST API, and SQL-based reporting paths. | Medium Confidence | Exact report source dependencies Unknown. |

### 6.3 REP Fields and Expressions Observed in Supplied Research

| Field / Expression | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---:|
| `AMAN.REP` | Assets Under Management report source in example. | Yes | Unknown | No | Yes | Verified for sample |
| `#~8portmv` | Prints portfolio market value in AUM example. | Yes | Unknown | No | Yes | Verified for sample |
| `$:fileo` | Displays portfolio code in AUM example. | Yes | Unknown | No | Yes | Verified for sample |
| `$askport` | Header variable used in Portfolio Appraisal example to show CLI code entered at runtime. | Yes | Unknown | No | Yes | Verified for sample |
| `$:tfile` | Used in supplied research as transaction-summary analog to show CLI file containing a transaction. | Yes | Unknown | No | Yes | Verified as CSSI statement |
| `$firmg` | Used as an “Other” sector catch-all in an AUM sector example. | Yes | Unknown | No | Yes | Verified for sample |
| `\n` | Line break / carriage return marker in Replang example. | Yes | Unknown | No | Yes | Verified for sample |
| `.` prefix | Print command marker in Replang example. | Yes | Unknown | No | Yes | Verified for sample |
| `REP32.exe` | Advent reporting application/engine used by connector. | Yes | Yes | No | Yes | Verified for connector |

---

## 7. Data Model for Documenting Reports

This section defines a repository documentation model for reports. It is not an Axys file schema or APX database schema.

### 7.1 Report Metadata Model

| Field | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---:|
| `system` | `Axys` or `APX`. | Yes | Yes | No | No | Repository convention |
| `report_name` | User-facing report name. | Partially known | Partially known | No | Maybe | Medium |
| `report_family` | Holdings, transactions, performance, attribution, contribution, risk, AUM, client package. | Partially known | Partially known | No | Maybe | Medium |
| `report_engine` | REP/Replang, Report Writer Pro, SSRS, report package, other. | REP examples | SSRS for guide reports | No | Yes if REP | Medium |
| `source_artifact` | `.REP`, `.RPW`, macro, RDL, package definition, custom report file. | Mostly Unknown | Mostly Unknown | No | Yes if REP | Unknown |
| `parameters` | Runtime inputs. | Unknown | Unknown | No | Yes / SSRS | Unknown |
| `portfolio_scope` | Portfolio, group, household, composite, benchmark, account. | Unknown | Partially visible | Possible | Possible | Medium |
| `date_scope` | As-of date, from/to period, prior business day, report date. | Unknown | Partially visible | Possible | Possible | Medium |
| `classification_scope` | Asset class, sector, industry, country, region, custom group. | Unknown | Partially visible | Possible | Possible | Medium |
| `output_fields` | Labels and columns emitted by the report. | Partially known | Partially known | No | Yes / SSRS | Medium |
| `calculation_method` | Stored, recalculated, linked, point-to-point, report-specific. | Unknown | Unknown | No | Maybe | Unknown |
| `version_notes` | Known version differences. | Limited | Limited | No | Maybe | Unknown |
| `quirks` | Known issues, reconciliation warnings, local customizations. | Limited | Limited | No | Maybe | Medium |

### 7.2 Report Field Dictionary

| Field | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---:|
| `Portfolio Code` | Owner portfolio code; available as Axys Portfolio Appraisal column in supplied Report Writer example. | Yes | Unknown | Unknown | Yes | Verified |
| `Quantity` | Holding quantity label in Axys Portfolio Appraisal sample and transaction/holding contexts. | Yes | Yes as label | Unknown | Yes / SSRS | Verified as label |
| `Security` | Security description/name label in holdings, transactions, attribution/contribution examples. | Yes as label | Yes as label | Unknown | Yes / SSRS | Verified as label |
| `Price` | Price label in Axys holdings sample and APX transaction report examples. | Yes as label | Yes as label | Unknown | Yes / SSRS | Verified as label |
| `Market Value` | Market value output measure. | Yes as label | Yes as label | Unknown | Yes / SSRS | Verified as label |
| `Pct Assets` | Percent-of-assets label in Axys Portfolio Appraisal sample. | Yes | Unknown | Unknown | Yes | Verified for sample |
| `Yield` | Yield label in Axys Portfolio Appraisal sample. | Yes | Unknown | Unknown | Yes | Verified for sample |
| `Portfolio Return` | APX attribution/contribution report label. | Unknown | Yes | Unknown | SSRS | Verified as APX label |
| `Benchmark Return` | APX attribution/contribution report label. | Unknown | Yes | Unknown | SSRS | Verified as APX label |
| `Active Return` | APX attribution/contribution report label. | Unknown | Yes | Unknown | SSRS | Verified as APX label |
| `Allocation Effect` | APX attribution report label. | Unknown | Yes | Unknown | SSRS | Verified as APX label |
| `Selection Effect` | APX attribution report label. | Unknown | Yes | Unknown | SSRS | Verified as APX label |
| `Total Effect` | APX attribution report label. | Unknown | Yes | Unknown | SSRS | Verified as APX label |
| `Avg Wgt` | APX attribution/contribution average weight label. | Unknown | Yes | Unknown | SSRS | Verified as APX label |
| `Return` | APX attribution/contribution report label. | Unknown | Yes | Unknown | SSRS | Verified as APX label |
| `Contrib` | APX contribution report label. | Unknown | Yes | Unknown | SSRS | Verified as APX label |
| `Trade Date` | APX Transaction Summary report label. | Unknown | Yes | Unknown | SSRS / report | Verified as APX label |
| `Settle Date` | APX Transaction Summary report label. | Unknown | Yes | Unknown | SSRS / report | Verified as APX label |
| `Unit Price` | APX Transaction Summary report label. | Unknown | Yes | Unknown | SSRS / report | Verified as APX label |
| `Amount` | APX Transaction Summary report label. | Unknown | Yes | Unknown | SSRS / report | Verified as APX label |
| `Industry Sector` | APX attribution/contribution grouping label. | Unknown | Yes | Unknown | SSRS | Verified as APX label |

---

## 8. Report Families

### 8.1 Holdings / Position Reports

| System | Known Reports | Supported Behavior | Confidence | Unknowns |
|---|---|---|---:|---|
| Axys | `Portfolio Appraisal`; `CDIhold.rep` in AIA workflow | Holdings/assets point-in-time reporting; owner portfolio code can be added in supplied example. | Verified for examples | Full field list, source files, tax-lot behavior, report calculations. |
| APX | `Portfolio Appraisal` | Shows holdings by tax lot or position according to supplied APX holdings research. | Verified for report concept | Full guide text, exact fields, report engine details, source tables. |

### 8.2 Transaction / Activity Reports

| System | Known Reports | Supported Behavior | Confidence | Unknowns |
|---|---|---|---:|---|
| Axys | Unknown standard names | Transaction reports likely exist but are not verified by supplied report catalog. | Unknown | Report names, fields, transaction-code display, posted/blotter distinction. |
| APX | `Transaction Summary`, `Activity Profile` | Transaction Summary displays account transactions; labels include trade/settle dates, quantity, security, unit price, amount in supplied research. | Verified name; field labels from examples | Transaction-code fields, canceled/reversed transaction handling, source tables. |

### 8.3 Performance Reports

| System | Known Reports | Supported Behavior | Confidence | Unknowns |
|---|---|---|---:|---|
| Axys | Unknown standard names | Product-level performance reporting capability is supported. | Verified at product level | Exact report names, stored vs recalculated behavior, linking, gross/net, currency. |
| APX | `Performance Overview`; attribution/contribution reports | APX guide includes performance and analytics reports. | Verified names | Stored vs recalculated behavior, formula, currency, fees, composite behavior. |

### 8.4 Attribution Reports

| System | Known Reports | Supported Behavior | Confidence | Unknowns |
|---|---|---|---:|---|
| Axys | Unknown | No supplied Axys attribution report catalog. | Unknown | Whether standard Axys has attribution reports in supplied materials. |
| APX | `Attribution by Classification`, `Attribution Summary`, `Attribution by Selected Groupings` | Portfolio vs benchmark attribution; allocation/selection/total effects; classification and drilldown examples. | Verified | Attribution methodology, arithmetic/geometric basis, multi-period behavior. |

### 8.5 Contribution Reports

| System | Known Reports | Supported Behavior | Confidence | Unknowns |
|---|---|---|---:|---|
| Axys | Unknown | No supplied Axys contribution report catalog. | Unknown | Names, fields, contribution formula. |
| APX | `Contribution by Classification`, `Contribution Summary`, `Contribution Detail` | Segment/security contribution; `Contribution Summary` may run with or without benchmark; `Contribution Detail` flattened. | Verified | Formula, rounding, residuals, benchmark alignment. |

### 8.6 Risk Reports

| System | Known Reports | Supported Behavior | Confidence | Unknowns |
|---|---|---|---:|---|
| Axys | Unknown | Risk report names not supplied. | Unknown | Metrics, source returns, benchmark handling. |
| APX | `Risk Statistics`, `Risk Overview` | Report names verified. | Verified name; details Unknown | Metrics, formulas, period definitions, data source. |

### 8.7 Allocation / Classification Reports

| System | Known Reports | Supported Behavior | Confidence | Unknowns |
|---|---|---|---:|---|
| Axys | AUM-by-sector custom example; product-level allocation/classification reporting | Axys can display performance by asset class, sector, country, or region and group portfolios by categories. | Verified at product level; examples Verified | Exact standard report names, classification source, historical classification behavior. |
| APX | `Allocation Summary`, `Equity Overview`, attribution/contribution by classification | APX guide supports classification/grouping output including Industry Sector. | Verified | Classification source and as-of behavior. |

### 8.8 Client Reporting Packages

| System | Known Reports / Components | Supported Behavior | Confidence | Unknowns |
|---|---|---|---:|---|
| Axys | Unknown component names | Product/reporting capability supports client reporting. | Medium Confidence | Package definitions, component reports, branding, output workflow. |
| APX | `Cover Page`, `Household Overview`, `Portfolio Overview`, `Performance Overview`, `Risk Overview`, `Policy Overview`, `Historical Policy Overview`, `Allocation Summary`, `Equity Overview`, `Fixed Income Distribution`, `Fixed Income Overview`, `Income Projection`, `Disclaimer and Terms` | APX guide includes client-reporting report components. | Verified names | Package configuration, output order, parameter inheritance. |

---

## 9. Examples

### 9.1 Axys: Portfolio Appraisal with Owner Portfolio Code

**Source basis:** Supplied holdings and REP research.

| Step | Behavior | Confidence |
|---:|---|---:|
| 1 | Create or modify an Axys Portfolio Appraisal in Report Writer. | Verified for example |
| 2 | Add `Portfolio Code` as a column. | Verified |
| 3 | Use `Management Mode` in the supplied example for combined group output. | Verified for example |
| 4 | Run for an unconsolidated group to show holdings with owner portfolio code. | Verified for example |
| 5 | Output labels in the example include `Quantity`, `Security`, `Price`, `Market Value`, `Pct Assets`, `Yield`, and `Portfolio Code`. | Verified for sample |

**Implementation caution:** This proves behavior for the supplied example, not every Axys version, report variation, or local customization.

### 9.2 Axys: AUM Report Customization

**Source basis:** Supplied REP research.

| Item | Supported Detail | Confidence |
|---|---|---:|
| Standard file | `AMAN.REP` | Verified for example |
| Safe customization pattern | Copy standard report before editing. | Verified for example |
| Example copied file | `AMAN_XX.REP` | Verified for example |
| Market value expression | `#~8portmv` | Verified for example |
| Portfolio-code variable | `$:fileo` | Verified for example |
| Output line break | `\n` | Verified for example |

### 9.3 APX: Attribution Summary

**Source basis:** Supplied APX Reports research.

| Section / Label | Supported Detail | Confidence |
|---|---|---:|
| Summary | `Portfolio Return`, `Benchmark Return`, `Active Return`, `Allocation Effect`, `Selection Effect`, `Total Effect`. | Verified |
| Rankings | `Top Contributors`, `Bottom Contributors`, `Top Attribution Effects`, `Bottom Attribution Effects`, `Largest Weights`. | Verified |
| Detail labels | `Security`, `Avg Wgt`, `Return`, `Contrib`. | Verified |
| Grouping | `Industry Sector` appears as grouping label in examples. | Verified |
| Formula | Unknown. | Unknown |
| Source-data sets | Unknown. | Unknown |

### 9.4 APX: Contribution Summary

**Source basis:** Supplied APX Reports research.

| Section / Label | Supported Detail | Confidence |
|---|---|---:|
| Performance summary | Shows portfolio, benchmark, and active return labels in examples. | Verified |
| Benchmark optionality | May be run without benchmark for absolute performance focus. | Verified |
| Detail labels | `Avg Wgt`, `Return`, `Contrib`. | Verified |
| Formula | Unknown. | Unknown |
| Source-data sets | Unknown. | Unknown |

---

## 10. Version Differences

| Topic | Axys | APX | Confidence |
|---|---|---|---:|
| Report engine | REP/Replang and Report Writer Pro are supported in supplied examples. | SSRS for guide-covered APX reports; REP32/Replang in connector/custom contexts. | Medium / Verified by context |
| Standard report catalog | Full catalog not supplied. | Partially known from public APX Reports Guide. | Medium |
| Customization layer | Report Writer Pro and Replang examples. | SSRS/custom reporting plus Report Writer Pro/Replang evidence. | Medium |
| Database-backed reports | Not verified for Axys. | Likely for SSRS reports; exact datasets Unknown. | Medium |
| Axys 3.8.7 | Enhanced Position Reconciliation report, expanded generic date framework, additional/improved multicurrency reports. | Not applicable. | Verified from supplied REP research |
| Connector-supported versions | Axys 3.8.6 appears as minimum supported version for a Data Broker connector. | APX 15.2 / 16.1 / 16.2 / 17.1 appear as supported connector versions. | Verified for connector only |
| APX report guide | Not applicable. | Guide-covered reports use SSRS. | Verified |
| APX Replang keywords | Not applicable. | Consultant research says APX has more Replang keywords than Axys. | Medium Confidence |

---

## 11. Known Issues / Quirks

| Issue / Quirk | Axys | APX | Confidence | Practical Implication |
|---|---:|---:|---:|---|
| Report labels may not equal native field names. | Yes | Yes | High Confidence | Do not build data dictionaries from report headings alone. |
| Report output may not reconcile to IMEX without matching parameters. | Yes | Yes | High Confidence | Align dates, portfolios, currency, benchmark, classification, and fee basis. |
| Custom reports may differ from standard reports with similar names. | Yes | Yes | High Confidence | Record source artifact and local modifications. |
| REP-based extraction can be layout-sensitive. | Yes | Yes where REP used | Medium Confidence | Column/heading changes can break parsers. |
| APX SSRS labels may not match database field names. | Not applicable | Yes | Medium Confidence | Inspect RDL/query definitions before mapping fields. |
| Axys Replang labels may not match IMEX fields. | Yes | Unknown | Medium Confidence | Inspect REP source and compare to IMEX. |
| Stored monthly performance may differ from longer-period report output. | Unknown | Unknown | Unknown | Requires controlled performance tests. |
| Classification reports may use current or historical classifications. | Unknown | Unknown | Unknown | Requires report source and before/after classification tests. |
| Security-level contribution may not sum exactly to portfolio return. | Unknown | Unknown | Medium Confidence as generic caution | Check report contribution fields and residuals. |
| Benchmark data may use a different source path from portfolio data. | Unknown | Unknown | Medium Confidence | Critical for attribution/contribution reconciliation. |
| Upgrades may overwrite or break customized reports. | Unknown | Unknown | Medium Confidence | Preserve report source and test after upgrades. |
| APX report package output may hide component report parameters. | Unknown | Unknown | Medium Confidence | Capture package definitions and component parameters. |

---

## 12. References

### 12.1 Supplied Repository Sources

| Source | Use |
|---|---|
| `AXYS_APX_REFERENCE_BLUEPRINT.md`, Version 2.0 | Governing editorial standard, chapter structure, field dictionary format, confidence labels, and Unknown handling. |
| `Research_14_Reports.md` | Primary source for report names, APX report labels, report families, IMEX/report cautions, version differences, and Unknowns. |
| `Research_13_REP.md` | REP/Replang mechanics, `.REP` examples, `AMAN.REP`, `REP32.exe`, Report Writer Pro, and custom report cautions. |
| `Research_12_IMEX.md` | IMEX/report distinction, connector extraction, IMEX logs, file artifacts, and report-based extraction cautions. |
| `Research_06_Holdings.md` | `Portfolio Appraisal`, holdings report behavior, `CDIhold.rep`, report labels, group behavior, and holdings extract workflows. |
| `Research_05_Transactions.md` | `Transaction Summary`, transaction-report labels, blotter/report distinctions, and transaction-output cautions. |
| `Research_10_Performance.md` | Performance report unknowns, stored-vs-recalculated risk, candidate report families, and performance testing requirements. |
| `Research_11_Classifications.md` | Classification report categories, APX custom classification snippets, and classification-as-of Unknowns. |
| `Research_03_APX_Architecture.md` | APX SSRS reporting framework, report-guide names, APX report customization, and APX data-access context. |
| `Research_02_Axys_Architecture.md` | Axys reporting architecture, Report Writer Pro, Replang/REP, and file-oriented report/customization context. |

### 12.2 External Sources Identified in Supplied Research

The chapter relies on the supplied research summaries rather than independently adding new source material. External source types identified in those research files include:

| Source Type | Examples / Relevance |
|---|---|
| SS&C Advent product pages | Axys and APX product-level reporting capabilities. |
| Advent Portfolio Exchange Reports Guide | APX report categories, report names, visible labels, SSRS basis. |
| Consultant documentation | Axys/APX Report Writer Pro, Replang, `.REP`, report customization examples. |
| Integration documentation | Data Broker / REP32 extraction, AIA/CI report and IMEX workflows. |
| Conversion documentation | Axys Reconciliation report usage and report/export caveats. |

---

## 13. Unknowns

The following items remain Unknown and should not be documented as facts until supported by vendor documentation, report source, report output, IMEX exports, or production observations.

| Unknown | Axys | APX | How to Resolve |
|---|---:|---:|---|
| Complete standard report catalog | Unknown | Partially known | Provide installed report catalog or vendor report guide. |
| Exact report file names | Unknown | Unknown | Provide REP/RDL/report source tree. |
| Report parameter names | Unknown | Unknown | Provide report definitions, screenshots, or RDL/REP source. |
| Report source-data | Unknown | Unknown | Inspect REP source, APX RDL datasets, APX SQL, public views, or stored accounting functions. |
| Stored vs recalculated performance behavior | Unknown | Unknown | Compare controlled report outputs with stored exports and reruns after input changes. |
| Report-to-IMEX equivalence | Unknown | Unknown | Provide matched report output and IMEX extracts. |
| Gross/net-of-fee handling | Unknown | Unknown | Provide report options/source and fee examples. |
| Currency handling | Unknown | Unknown | Provide multicurrency report samples and source definitions. |
| Classification as-of behavior | Unknown | Unknown | Run before/after classification tests and inspect report source. |
| Benchmark data source | Unknown | Unknown | Provide benchmark master/source and attribution report definitions. |
| Rounding and residual treatment | Unknown | Unknown | Provide detailed report output and calculation source. |
| Composite aggregation rules | Unknown | Unknown | Provide composite report definitions and test examples. |
| Household aggregation rules | Unknown | Unknown | Provide household report definitions and examples. |
| Report-package mechanics | Unknown | Unknown | Provide package definitions and component parameters. |
| Upgrade impact on custom reports | Unknown | Unknown | Provide vendor admin guide, release notes, or production upgrade notes. |
| APX SSRS RDL names and datasets | Not applicable | Unknown | Provide APX report deployment files or SSRS catalog export. |
| APX report stored procedures/public views | Not applicable | Unknown | Provide APX reporting database documentation or RDL SQL. |
| Axys REP field dictionary | Unknown | Not applicable | Provide RepLang Programmer's Guide and report source. |
| Axys transaction/cash/performance report names | Unknown | Not applicable | Provide Axys standard report catalog and sample outputs. |

---

## 14. Chapter Summary

Reports are a first-class operational interface in both Axys and APX, but the evidence differs by product.

For **Axys**, the supplied material supports predefined reports, Report Writer Pro, REP/Replang customization, specific examples such as `Portfolio Appraisal`, `AMAN.REP`, and `CDIhold.rep`, and selected report labels from holdings and AUM examples. The supplied material does not provide a complete Axys standard report catalog or a full field dictionary.

For **APX**, the supplied material provides stronger report-name evidence through the public APX report guide. Guide-covered APX investment reports use SSRS and include business intelligence, portfolio analytics, attribution, contribution, risk, holdings, transaction, and client-reporting report names. The supplied material does not provide APX RDL source, SQL datasets, stored procedures, exact parameter names, or database field mappings.

For both systems, report outputs must be treated carefully. A report label is not automatically an IMEX field, REP variable, APX database column, or stored accounting value. Report-to-IMEX or report-to-database reconciliation requires matching report parameters, calculation method, date scope, portfolio scope, benchmark, currency, gross/net settings, and classification logic.
