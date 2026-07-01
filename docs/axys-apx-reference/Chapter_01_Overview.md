# Chapter 01 — Overview

Repository: AXYS / APX Reference Repository
Chapter: `Chapter_01_Overview.md`
Prepared: 2026-06-29
Governing specification: `AXYS_APX_REFERENCE_BLUEPRINT.md`, Version 2.0
Source basis: supplied repository chapters `Chapter_02_Axys_Architecture.md` through `Chapter_16_Glossary.md` only

---

## Cross-chapter entity flow
This repository is easiest to read as one linked data lifecycle:

- Portfolio / account -> Security master -> Transactions -> Holdings / positions -> Cash -> Pricing -> Corporate actions -> Performance -> Reports
- IMEX and REP are the main exchange paths that move data between these stages, while the glossary and data dictionary provide the terminology backbone.

## Safe implementation rules
These rules should guide any implementation or further documentation work in this repository:

- Do not treat report labels as native fields.
- Do not assume IMEX object names or REP field names without evidence.
- Do not equate security type with classification, asset class, sector, country, or region.
- Do not assume performance is stored or recalculated without explicit evidence.
- Do not normalize or merge data across Axys and APX without first validating the source context and confidence level.

## Chapter cross-reference index
Use this index to jump to the chapters that most directly connect to a topic:

- Overview — entry point for the repository map, evidence conventions, and guardrails.
- Architecture — Axys and APX architecture and platform role: [Chapter_02_Axys_Architecture.md](Chapter_02_Axys_Architecture.md) and [Chapter_03_APX_Architecture.md](Chapter_03_APX_Architecture.md).
- Security master — identity and matching context: [Chapter_04_Security_Master.md](Chapter_04_Security_Master.md).
- Transactions — transaction meaning, coding, and posting context: [Chapter_05_Transactions.md](Chapter_05_Transactions.md).
- Holdings — positions and valuation context: [Chapter_06_Holdings.md](Chapter_06_Holdings.md).
- Cash — cash movement and cash-like activity: [Chapter_07_Cash.md](Chapter_07_Cash.md).
- Pricing — price import and valuation context: [Chapter_08_Pricing.md](Chapter_08_Pricing.md).
- Corporate actions — splits, reorgs, and related account changes: [Chapter_09_Corporate_Actions.md](Chapter_09_Corporate_Actions.md).
- Performance — performance inputs, outputs, and evidence boundaries: [Chapter_10_Performance.md](Chapter_10_Performance.md).
- Classifications — grouping and classification-like reporting context: [Chapter_11_Classifications.md](Chapter_11_Classifications.md).
- IMEX — import/export workflows and exchange paths: [Chapter_12_Imex.md](Chapter_12_Imex.md).
- REP — report extraction and report-source context: [Chapter_13_Rep.md](Chapter_13_Rep.md).
- Reports — report families and report-label caution: [Chapter_14_Reports.md](Chapter_14_Reports.md).
- Data dictionary and glossary — field-level vocabulary and terminology: [Chapter_15_Data_Dictionary.md](Chapter_15_Data_Dictionary.md) and [Chapter_16_Glossary.md](Chapter_16_Glossary.md).

## Chapter dependency map
A compact view of the main evidence flow:

Security master → Transactions → Holdings / positions → Cash → Pricing → Corporate actions → Performance → Reports

IMEX / REP → data movement, extraction, and report-source context → Reports

The glossary and data dictionary support the terminology and field vocabulary used throughout the repository.

## How to use this reference
Choose the path that fits your immediate question:

- If you are new to the topic, start with [Chapter_02_Axys_Architecture.md](Chapter_02_Axys_Architecture.md), [Chapter_03_APX_Architecture.md](Chapter_03_APX_Architecture.md), and [Chapter_12_Imex.md](Chapter_12_Imex.md) for the big picture.
- If you need to understand transactions, start with [Chapter_04_Security_Master.md](Chapter_04_Security_Master.md), [Chapter_05_Transactions.md](Chapter_05_Transactions.md), [Chapter_06_Holdings.md](Chapter_06_Holdings.md), and [Chapter_07_Cash.md](Chapter_07_Cash.md).
- If you are focused on performance or reporting, start with [Chapter_08_Pricing.md](Chapter_08_Pricing.md), [Chapter_10_Performance.md](Chapter_10_Performance.md), [Chapter_11_Classifications.md](Chapter_11_Classifications.md), [Chapter_13_Rep.md](Chapter_13_Rep.md), and [Chapter_14_Reports.md](Chapter_14_Reports.md).
- If you are looking up a term or field, start with [Chapter_15_Data_Dictionary.md](Chapter_15_Data_Dictionary.md) and [Chapter_16_Glossary.md](Chapter_16_Glossary.md).

## How to read a chapter
Each chapter is intended to be read in the same compact pattern:

- Start with the chapter’s scope and what it covers.
- Check the evidence and confidence notes before treating anything as settled.
- Look for implementation cautions, Unknowns, and any cross-references to adjacent chapters.
- Use the chapter as a starting point, then move to the linked chapters for the next step.

## 1. Overview

This chapter introduces the AXYS / APX Reference Repository and summarizes the factual scope of the supplied chapters. It is a repository orientation chapter, not a replacement for the detailed subject chapters.

The repository documents supported facts about SS&C Advent Axys and SS&C Advent Portfolio Exchange (APX), with emphasis on implementation-oriented behavior: architecture, accounting data, IMEX, REP, reports, file artifacts, data fields, processing behavior, version differences, and known quirks.

This chapter follows the repository blueprint rule that unsupported behavior must be marked **Unknown** rather than inferred.

### 1.0 Scope and non-goals

This repository is a reference set for documented Axys / APX behavior, evidence, and terminology. It is not intended to be:

- a full vendor specification,
- a software implementation guide for the underlying systems,
- a product roadmap or sales document, or
- a substitute for source-system data dictionaries, vendor manuals, or production exports.

The material here is deliberately scoped to what the supplied chapters can support with explicit evidence and confidence labels.

### 1.1 Repository Purpose

| Purpose | Repository Treatment | Confidence |
|---|---|---:|
| Preserve factual knowledge about Axys and APX | Supported by the governing blueprint and implemented across supplied chapters. | Verified |
| Serve developers, consultants, investment operations, performance analysts, data engineers, and AI coding assistants | Stated audience in blueprint. | Verified |
| Document implementation-oriented behavior rather than portfolio-accounting theory | Repository standard. | Verified |
| Separate Axys and APX behavior whenever behavior differs | Required editorial principle. | Verified |
| Preserve Unknowns | Required editorial principle. | Verified |

### 1.2 Confidence Labels

| Label | Meaning in this repository |
|---|---|
| Verified | Directly supported by supplied source material, supplied research, cited vendor/public material inside the research, observed examples recorded in the research, or the governing blueprint. |
| High Confidence | Strongly supported by supplied research and consistent implementation evidence, but not proven as a complete vendor specification. |
| Medium Confidence | Plausible and supported by partial, third-party, consultant, conversion, integration, or workflow-specific evidence. |
| Unknown | Not established from supplied material. Do not implement or document as fact without additional evidence. |

### 1.2a Evidence ladder

Use the confidence labels as an evidence ladder, not as a simple yes/no flag:

- Verified is the strongest level and should be treated as the best-supported statement in this repository.
- High Confidence is strong but still narrower than a full vendor specification or complete source-system schema.
- Medium Confidence is useful for workflow patterns, practitioner observations, and plausible interpretations that need validation.
- Unknown means the repository does not currently support the statement as a fact and the statement should not be treated as implementation guidance.

### 1.3 Evidence Boundary

This overview uses only the supplied completed repository chapters. No external research is added here.

| Evidence Area | Included | Notes |
|---|---:|---|
| Governing blueprint | Yes | Defines editorial rules, structure, and success criteria. |
| Axys architecture chapter | Yes | Supplies Axys architecture, file artifacts, IMEX, REP, version cautions, and Unknowns. |
| APX architecture chapter | Yes | Supplies APX platform, SSRS/reporting, IMEX, blotter, and SQL/access-path evidence. |
| Subject chapters 04–14 | Yes | Supply domain-specific facts for security master, transactions, holdings, cash, pricing, corporate actions, performance, classifications, IMEX, REP, and reports. |
| Data dictionary and glossary chapters | Yes | Supply consolidated field/token/report terminology. |
| Unsupplied vendor manuals, source exports, or production data | No | Needed to resolve many Unknowns, but not supplied here. |

---

## 2. Repository Structure

The repository is organized into subject chapters and matching research files. The supplied source material for this overview includes chapters 02 through 16.

| Chapter | Subject | Primary Purpose | Current Evidence Status |
|---|---|---|---|
| `Chapter_01_Overview.md` | Overview | Repository orientation and cross-chapter summary. | This chapter. |
| `Chapter_02_Axys_Architecture.md` | Axys architecture | Axys architecture, file-oriented character, IMEX, REP, data-domain dependencies, version cautions. | Supplied. |
| `Chapter_03_APX_Architecture.md` | APX architecture | APX platform role, SQL/SSRS/reporting architecture, IMEX, blotter concepts, APX-specific Unknowns. | Supplied. |
| `Chapter_04_Security_Master.md` | Security master | Security identity, symbol/type matching, `sec.inf`, `type.inf`, security translations. | Supplied. |
| `Chapter_05_Transactions.md` | Transactions | Transaction role, blotters, observed transaction codes, external-flow classification, translation patterns, IMEX and report status. | Supplied. |
| `Chapter_06_Holdings.md` | Holdings | Portfolio Appraisal, positions, `.pos`, holdings extraction, reconciliation, stored/calculated Unknowns. | Supplied. |
| `Chapter_07_Cash.md` | Cash | Cash activity, sweep/journal logic, cash-like symbols, cash-balance Unknowns. | Supplied. |
| `Chapter_08_Pricing.md` | Pricing | `.pri` files, price import, missing/stale/calculated price behavior, APX price-set workflow evidence. | Supplied. |
| `Chapter_09_Corporate_Actions.md` | Corporate actions | Axys `split.inf`, APX ACA/Reorg Utility workflow, dividends/reorg Unknowns. | Supplied. |
| `Chapter_10_Performance.md` | Performance | Performance capability, report labels, APX attribution/contribution evidence, storage/recalculation Unknowns. | Supplied. |
| `Chapter_11_Classifications.md` | Classifications | Security type versus classification, asset class/sector/country/region/custom classification evidence. | Supplied. |
| `Chapter_12_Imex.md` | IMEX | Import/export utilities, workflow artifacts, logs, object/field dictionary status. | Supplied. |
| `Chapter_13_Rep.md` | REP | `.REP`, RepLang, Report Writer Pro, REP32, report-based extraction. | Supplied. |
| `Chapter_14_Reports.md` | Reports | Axys and APX report families, APX named reports, report-label cautions. | Supplied. |
| `Chapter_15_Data_Dictionary.md` | Data dictionary | Cross-repository field, token, file, utility, report-label, and artifact index. | Supplied. |
| `Chapter_16_Glossary.md` | Glossary | Repository term definitions and ambiguity notes. | Supplied. |

---

## 3. Axys Summary

Axys is documented in the supplied chapters as a portfolio accounting, reporting, performance measurement, and reconciliation platform with a file-oriented operational character, a Report Writer / REP / RepLang reporting layer, and an import/export layer commonly referred to as IMEX.

### 3.1 Axys Product and Architecture Summary

| Area | Axys Finding | Confidence | Notes |
|---|---|---:|---|
| Product role | Portfolio accounting, portfolio reporting, performance measurement, reconciliation, and related investment operations workflows. | Verified / High Confidence | Supported at product-capability and chapter-summary level. |
| Data architecture | Commonly treated in supplied practitioner/integration evidence as proprietary/file-oriented rather than SQL-centered. | Medium Confidence | Exact physical storage internals remain Unknown. |
| IMEX | Axys Import/Export utility is observed as `imex32.exe` in Custodial Integrator evidence. | Verified for CI workflow | Full IMEX object dictionary Unknown. |
| REP / reporting | Axys reports are written in RepLang in supplied examples; `.REP` files and Report Writer Pro are supported. | Verified for examples | Full RepLang grammar and report catalog Unknown. |
| Direct file access | Possible in some practitioner contexts but version-sensitive and high-risk. | Medium / High Confidence as caution | Prefer IMEX, REP, or controlled exports where possible. |
| Version risk | Consultant evidence reports Axys file format changes and file conversion between versions such as 3.7 and 3.8. | Medium Confidence | Requires version-specific validation. |

### 3.2 Axys File and Utility Artifacts

The following artifacts are observed in supplied chapters. They are not a complete native Axys file specification.

| Artifact | Domain | Description | Confidence |
|---|---|---|---:|
| `imex32.exe` | IMEX | Axys Import/Export utility executable in CI workflow. | Verified for CI workflow |
| `pospos32.exe` | Positions | Axys Post Positions utility in CI workflow. | Verified for CI workflow |
| `REP32.exe` | Reports / REP | Report engine/client tool used by connector workflows. | Verified for connector |
| `*.cli` / `.cli` | Portfolio/client/account | Axys client/portfolio files in conversion and integration evidence. | Verified for Axys context |
| `topost.trn` | Transactions | Axys Trade Blotter file receiving imported transaction rows in CI workflow. | Verified for CI workflow |
| `ptopost.trn` | Positions | CI-generated position import file, reported as CSV and possibly lot-specific when enabled. | Verified for CI workflow |
| `.pos` | Positions | Position files created/replaced in Position Post / CI workflow. | Verified for CI workflow |
| `sec.inf` | Security master | Security information file/artifact in integration and conversion evidence. | Verified |
| `type.inf` | Security type | Security type information file/artifact in integration and conversion evidence. | Verified |
| `split.inf` / `SPLIT.INF` | Corporate actions | Axys securities splits file in conversion/corporate-action research. | High Confidence |
| `*.pri` | Pricing | Axys price files in `$pathpri` context. | Verified for CI workflow |
| `imexPrices.log` | IMEX logs | Price-import log/tab in CI workflow. | Verified for CI workflow |
| `imexPositions.log` | IMEX logs | Position-import log/tab in CI workflow. | Verified for CI workflow |
| `imexPositionLots.log` | IMEX logs | Position-lots log/tab used when position lots are imported. | Verified for CI workflow |
| `AMAN.REP` | REP / reports | Assets Under Management report file in Axys customization example. | Verified for example |
| `CDIhold.rep` | REP / holdings | WealthTechs-provided historical holdings extract report in AIA workflow. | Verified for workflow |
| `sipos30` | Reconciliation | Custom reconciliation report cited in CI context. | Verified for CI context |

### 3.3 Axys Data Domains

| Data Domain | Axys Evidence | Native Storage Status | Confidence |
|---|---|---|---:|
| Portfolios/accounts | `.cli`, portfolio codes, portfolio groups, and report output evidence. | Exact schema Unknown. | High Confidence |
| Security master | `sec.inf`, `type.inf`, symbol/type matching, security translations. | Complete field layout Unknown. | Verified in integration context |
| Transactions | `topost.trn`, `.cli`, observed transaction codes, Trade Blotter import workflow. | Complete code matrix and storage model Unknown. | High Confidence for workflow; Unknown for full native spec |
| Holdings/positions | Portfolio Appraisal, `.pos`, `ptopost.trn`, `CDIhold.rep`, reconciliation workflows. | Stored-versus-calculated model Unknown. | Verified for reports/workflows |
| Cash | Cash activity through transactions and cash-like tokens. | Cash-balance storage Unknown. | High Confidence for transaction tokens; native balances Unknown |
| Prices | `*.pri`, price import logs, calculated/missing/stale price workflow evidence. | Full `.pri` layout and native price model Unknown. | Verified for workflow artifacts |
| Corporate actions | `split.inf`; reinvestment and paydown conversion evidence. | Broader dividend/reorg storage Unknown. | High Confidence for split artifact |
| Performance | Product-level capability and report capability. | Stored-vs-recalculated behavior Unknown. | Verified capability; implementation Unknown |
| Classifications | Asset class/sector/country/region reporting capability and export labels. | Classification storage and history Unknown. | Verified at reporting/category level |

---

## 4. APX Summary

APX is documented in the supplied chapters as an integrated portfolio management, accounting, reporting, performance analytics, and client relationship platform. The supplied APX evidence supports a more centralized, SQL/reporting-oriented architecture than Axys, but does not provide native APX database schemas, table names, stored procedures, or full API details.

### 4.1 APX Product and Architecture Summary

| Area | APX Finding | Confidence | Notes |
|---|---|---:|---|
| Product role | Integrated portfolio management, accounting, reporting, performance, and client relationship platform. | Verified | Product-level evidence. |
| Architecture | Centralized, enterprise-oriented platform; SQL/reporting access options appear in supplied research. | Verified / High Confidence at architecture level | Exact schema Unknown. |
| Reporting | APX guide-covered investment reports are described as built on Microsoft SQL Server Reporting Services. | Verified for guide-covered reports | Exact RDL/dataset/stored procedure names Unknown. |
| REP / REP32 | REP32/RepLang/macros appear in APX connector workflows and practitioner evidence. | Verified for connector; Medium Confidence generally | Not proof that all APX reports are REP-based. |
| IMEX | APX Import/Export appears as `APXIX.exe`, `apxix.exe`, and `ApxIx` in supplied integration research. | Verified in contexts; naming relationship Unknown | Full object/field dictionary Unknown. |
| SQL / Public Views / Stored Accounting Functions / REST | Practitioner research identifies these as APX access paths. | Medium Confidence | View/function/API names and coverage Unknown. |
| Blotters | Trade, position, lot/tax lot, statement, account, initial transaction, pending, and dividend adjustment blotters appear in AIA/CI evidence. | Verified for workflows | Native complete blotter taxonomy Unknown. |

### 4.2 APX Utility, Report, and Workflow Artifacts

| Artifact / Term | Domain | Description | Confidence |
|---|---|---|---:|
| `APXIX.exe` / `apxix.exe` | IMEX | APX import/export executable/function in AIA/CI context. | Verified in context |
| `ApxIx` | IMEX | APX Import/Export terminology in CI context. | Verified in context; relationship to `APXIX.exe` Unknown |
| `REP32.exe` | REP / reports | Connector report extraction engine. | Verified for connector |
| `Advent IMEX Log` | IMEX logs | APX AIA log for most recent IMEX import. | Verified for AIA workflow |
| `Advent IMEX History Log` | IMEX logs | APX AIA history log for IMEX imports. | Verified for AIA workflow |
| `Trade Blotter` | Transactions | APX transaction staging/review/import blotter. | Verified in workflows |
| `Position Blotter` | Positions | APX position import/reconciliation blotter. | Verified in CI workflow |
| `Lot Blotter` / `Tax Lot Blotter` | Lots | APX lot/tax-lot import/reconciliation context. | Verified in workflows |
| `Statement Blotter` | Statement transactions | Statement transaction/reconciliation workflow. | Verified in AIA workflow |
| `Account Blotter` | Account data | Account demographic import workflow. | Verified in AIA workflow |
| `Initial Transaction Blotter` | Initial positions | AIA-created deliver-in transactions from positions when configured. | Verified in AIA workflow |
| `APX Reorg Utility` | Corporate actions | Utility run after ACA actions are downloaded; generated transactions post to APX Trade Blotter. | Verified |
| `SourceId` | Pricing | Price source label observed in AIA APX price context. | Verified for AIA context only |
| `CDIhold.rep` | Holdings | AIA historical holdings extract report for APX workflow. | Verified for workflow |

### 4.3 APX Report Families Identified

The supplied report chapter identifies APX report names. Report names are not database tables or field names.

| Report Family | Examples | Confidence |
|---|---|---:|
| Business intelligence | Account Distribution, Account Characteristics, Account Characteristics (By Custodian), Asset Flows, Business Summary Dashboard. | Verified names |
| Portfolio analytics | Activity Profile, Attribution by Classification, Attribution Summary, Attribution by Selected Groupings, Contribution by Classification, Contribution Summary, Contribution Detail, Risk Statistics. | Verified names |
| Client reporting | Cover Page, Household Overview, Portfolio Overview, Performance Overview, Risk Overview, Policy Overview, Historical Policy Overview, Allocation Summary, Equity Overview, Fixed Income Distribution, Fixed Income Overview, Income Projection, Disclaimer and Terms. | Verified names |
| Holdings / activity / tax | Portfolio Appraisal, Transaction Summary, Realized Gains and Losses. | Verified or Medium Confidence depending report evidence |

### 4.4 APX Data Domains

| Data Domain | APX Evidence | Native Storage Status | Confidence |
|---|---|---|---:|
| Portfolios/accounts | APX Portfolio Code, account blotter fields, account reports. | Exact key/schema Unknown. | Verified in workflows |
| Security master | APX Symbol, APX Security Type, `sec.inf`, `type.inf`, security translation workflows. | SQL table/view names Unknown. | Verified in integration context |
| Transactions | Trade Blotter, Statement Blotter, transaction reports, observed translation fields and codes. | Full code matrix and storage model Unknown. | High Confidence for workflows |
| Holdings/positions | Portfolio Appraisal, Position Blotter, current-date APX SQL extraction in AIA workflow, `CDIhold.rep`. | Native table/view names Unknown. | Verified for workflows |
| Cash | Cash-like tokens and sweep/journal integration logic. | Cash-balance model Unknown. | High Confidence for workflow tokens; native balances Unknown |
| Prices | AIA price-file update logic, price-set logic, custodian-specific pricing, `SourceId`. | Native price schema Unknown. | Verified for AIA workflow |
| Corporate actions | ACA Server, APX Reorg Utility, Trade Blotter postings. | Reorg output fields and final lifecycle Unknown. | Verified workflow; field-level Unknown |
| Performance | Product-level performance analytics; APX attribution/contribution report labels. | Stored-vs-recalculated behavior Unknown. | Verified capability and report labels |
| Classifications | APX report evidence for custom classification, industry group, sector; attribution/contribution reports. | Storage/history Unknown. | Verified at report/category level |

---

## 5. Axys vs APX Comparison

| Dimension | Axys | APX | Confidence |
|---|---|---|---:|
| Product orientation | Portfolio accounting/reporting/performance/reconciliation platform. | Integrated portfolio/accounting/reporting/performance/client-management platform. | Verified |
| Architecture character | Proprietary/file-oriented operational character in supplied evidence. | Centralized / SQL-reporting-oriented platform in supplied evidence. | Axys Medium; APX High |
| Primary import/export layer | IMEX / Axys Import/Export utility; `imex32.exe` in CI context. | APX Import/Export / `APXIX.exe` / `ApxIx`; exact naming relationship Unknown. | Verified in contexts |
| Primary report customization evidence | `.REP`, RepLang, Report Writer Pro, `AMAN.REP` example. | SSRS reports plus REP32/RepLang connector evidence and practitioner evidence. | Axys Verified; APX Mixed |
| Native files observed | `.cli`, `sec.inf`, `type.inf`, `split.inf`, `.pri`, `.pos`, `.REP`. | APX may generate/use `sec.inf`, `type.inf`, `.pri`, `.cli`, `.REP`-style artifacts in workflows, but native SQL schema remains Unknown. | High for observed contexts |
| SQL access | Not verified as Axys architecture. | Public Views, Stored Accounting Functions, SQL/SSRS, and REST mentioned in practitioner evidence. | APX Medium Confidence |
| Blotters | `topost.trn` Trade Blotter evidence; position workflows via `ptopost.trn` / Position Post. | Trade, position, lot/tax lot, statement, account, initial transaction, and other blotter concepts. | Verified in workflows |
| Corporate actions | `split.inf` is the strongest Axys-specific corporate-action artifact. | ACA/APX workflow with ACA Server, APX Reorg Utility, APX Trade Blotter postings. | Verified / High Confidence |
| Performance implementation | Product-level capability; stored/recalculated behavior Unknown. | Product/report-level capability; stored/recalculated behavior Unknown. | Capability Verified; implementation Unknown |
| Classification history | Unknown. | Unknown. | Unknown |

---

## 6. IMEX Overview

IMEX is documented as an import/export mechanism used in Axys and APX workflows. The supplied chapters verify multiple IMEX-adjacent artifacts and workflows but do not provide a complete official IMEX object dictionary.

### 6.1 IMEX Findings

| Finding | Axys | APX | Confidence |
|---|---:|---:|---:|
| IMEX exists as an import/export mechanism. | Yes | Yes | Verified / High Confidence |
| Axys utility name `imex32.exe` is observed. | Yes | No | Verified for CI workflow |
| APX import/export names `APXIX.exe`, `apxix.exe`, and `ApxIx` are observed. | No | Yes | Verified in contexts; relationship Unknown |
| Security data can be exported/used as `sec.inf` in CI context. | Yes | Yes | Verified in CI context |
| Security type data can be exported/used as `type.inf` in CI context. | Yes | Yes | Verified in CI context |
| Transactions can be imported through Trade Blotter workflows. | Yes | Yes | Verified in workflows |
| Positions/prices can be imported in integration workflows. | Yes | Yes | Verified in workflows |
| Official native IMEX object names are known. | Unknown | Unknown | Unknown |
| Complete native IMEX field dictionaries are known. | Unknown | Unknown | Unknown |

### 6.2 IMEX Object Status

| Data Area | Axys IMEX Object Name | APX IMEX Object Name | Status |
|---|---|---|---|
| Transactions | Unknown | Unknown | Trade Blotter workflows verified; official object names not supplied. |
| Positions / holdings | Unknown | Unknown | Position files and blotters verified in workflows; official object names not supplied. |
| Position lots | Unknown | Unknown | APX lot import workflow verified; Axys lot status Unknown. |
| Prices | Unknown | Unknown | Price files/logs verified in workflows; official object names not supplied. |
| Security master | Unknown | Unknown | `sec.inf` observed; object name and layout Unknown. |
| Security types | Unknown | Unknown | `type.inf` observed; object name and layout Unknown. |
| Portfolio/account master | Unknown | Unknown | `.cli` and account blotter evidence exists; object names Unknown. |
| Cash balances | Unknown | Unknown | Native cash balance objects not supplied. |
| Corporate actions | Unknown | Unknown | Axys `split.inf` exists; APX ACA workflow exists; IMEX objects Unknown. |
| Performance / `portperf` | Unknown | Unknown | Candidate/user-recalled term only; not verified. |
| Security performance / `secperf` | Unknown | Unknown | Candidate/user-recalled term only; not verified. |
| Classifications | Unknown | Unknown | Storage/export mechanism not established. |

---

## 7. REP and Reports Overview

REP, RepLang, Report Writer Pro, and REP32 are documented as report and report-extraction mechanisms. Reports are distinct from IMEX exports unless an export is explicitly identified as an IMEX object.

### 7.1 REP Findings

| Finding | Axys | APX | Confidence |
|---|---:|---:|---:|
| `.REP` report files are supported in examples. | Yes | Possible / not fully verified | Verified for Axys examples |
| RepLang is Advent's proprietary report-writing language. | Yes | Medium Confidence | Verified for Axys; Medium for APX |
| Report Writer Pro is supported. | Yes | Medium Confidence | Verified for Axys; practitioner evidence for APX |
| REP32 is used by at least one connector. | Yes | Yes | Verified for connector |
| Report output can be used for extraction. | Yes | Yes | High Confidence |
| Full RepLang grammar is supplied. | No | No | Unknown |
| REP32 command-line syntax is supplied. | No | No | Unknown |

### 7.2 Reports and Report Labels

| Rule | Description | Confidence |
|---|---|---:|
| Report output is not automatically source-data | Report labels may be presentation labels, calculated values, renamed fields, or stored values. | High Confidence |
| Report names are not database table names | APX report names do not prove APX SQL object names. | Verified caution |
| Report values may be stored or recalculated | Exact behavior is report-specific and mostly Unknown. | Unknown / High Confidence as caution |
| Report parsing is layout-sensitive | Columns, headings, sections, totals, and formatting can change. | Medium / High Confidence |
| Custom report behavior is installation-specific | Custom `.REP`, `.RPW`, macro, or SSRS reports may differ from vendor standards. | High Confidence |

### 7.3 Report Artifacts Identified

| Report / Artifact | System | Description | Confidence |
|---|---|---|---:|
| `Portfolio Appraisal` | Axys | Holdings/assets point-in-time report; Report Writer customization example exists. | Verified |
| `Portfolio Appraisal` | APX | APX holdings report evidence; shows holdings by tax lot or position according to supplied report research. | Medium / Verified depending source capture |
| `AMAN.REP` | Axys | Assets Under Management report file in example. | Verified for example |
| `CDIhold.rep` | Axys/APX workflow | WealthTechs historical holdings extract report. | Verified for AIA workflow |
| `Transaction Summary` | APX | Transaction report with labels such as Trade Date, Settle Date, Quantity, Symbol, Security, Unit Price, Amount. | Verified / Medium depending label |
| `Attribution Summary` | APX | Performance attribution report with portfolio/benchmark/active/allocation/selection/total effect labels. | Verified |
| `Contribution Summary` | APX | Contribution report with labels such as Avg Wgt, Return, Contrib. | Verified |
| `Risk Statistics` | APX | APX analytics/risk report name. | Verified name |
| `Realized Gains and Losses` | APX | Realized gain/loss report name. | Verified name |

---

## 8. Cross-Repository Conceptual Data Model

The supplied chapters support a conceptual data model. The following entities are not asserted as native Axys files, APX tables, or IMEX objects unless explicitly marked.

```text
Portfolio / Account / Client
        |
        +-- Security Master / Security Type
        |
        +-- Transactions / Blotters / Posted Activity
        |
        +-- Cash Activity / Cash-like Securities
        |
        +-- Holdings / Positions / Lots
        |
        +-- Prices / Price Sources / Price Sets
        |
        +-- Corporate Actions / Splits / Reorg Workflow
        |
        +-- Classifications / Groups / Report Dimensions
        |
        +-- Performance / Benchmarks / Composites
        |
        +-- Reports / IMEX / REP / SQL / Integrations
```

### 8.1 Entity Status

| Entity | Axys Status | APX Status | Confidence |
|---|---|---|---:|
| Portfolio/account | Supported; `.cli`, Portfolio Code, groups, reports. | Supported; APX Portfolio Code, account blotters, account reports. | Verified generally; field-level mixed |
| Client/relationship/household | Client reporting supported; detailed model Unknown. | APX CRM/client/household reporting supported; schema Unknown. | Verified generally; schema Unknown |
| Security master | `sec.inf`, `type.inf`, symbol/type matching. | APX security information, APX Symbol, APX Security Type, `sec.inf`, `type.inf`. | Verified in workflows |
| Transactions | Trade Blotter, `.cli`, observed codes. | Trade Blotter, Statement Blotter, Transaction Summary, observed codes. | Verified generally; full code matrix Unknown |
| Holdings/positions | Portfolio Appraisal, `.pos`, `CDIhold.rep`, Position Post. | Portfolio Appraisal, Position Blotter, current-date SQL extraction in AIA workflow, `CDIhold.rep`. | Verified in reports/workflows; storage Unknown |
| Cash | Transaction/cash-like token evidence. | Transaction/cash-like token evidence. | High Confidence for tokens; cash-balance storage Unknown |
| Prices | `*.pri`, `imexPrices.log`, calculated/missing/stale price workflow. | AIA price update logic, price sets, custodian pricing, `SourceId`. | Verified for workflows; native schema Unknown |
| Corporate actions | `split.inf`; conversion evidence for reinvestments/paydowns. | ACA Server, APX Reorg Utility, Trade Blotter postings. | Axys split High; APX ACA Verified |
| Performance | Product-level capability; exact objects Unknown. | Product/report capability; attribution/contribution report labels. | Capability Verified; implementation Unknown |
| Classifications | Asset class/sector/country/region reporting categories and export labels. | Custom classification/industry group/sector report categories. | Report-level Verified; storage Unknown |
| Reports | REP/Replang examples and predefined/custom reports. | SSRS-based guide reports plus REP32 connector evidence. | Verified / High Confidence |

### 8.2 Source System and Lineage Requirements

Any downstream extract or interface documented from this repository should preserve source lineage.

| Metadata | Reason | Confidence |
|---|---|---:|
| Source system: Axys or APX | Behaviors and interfaces differ. | Verified repository rule |
| Source path: IMEX, REP, report export, SQL/public view, API, direct file, integration tool | Field meanings and calculation timing may differ by path. | High Confidence |
| Source artifact name | Needed to distinguish `sec.inf`, `type.inf`, `.pri`, `.REP`, report names, blotter files, SQL views, or custom exports. | High Confidence |
| System version | Version differences and file/report behavior may matter. | High Confidence |
| Run date and as-of date | Report generation date and economic/reporting date can differ. | High Confidence |
| Portfolio/account scope | Reports and exports may operate on portfolios, groups, households, composites, or accounts. | High Confidence |
| Currency and date basis | Trade/settlement date, local/base currency, and multi-currency behavior may affect values. | Medium / High Confidence |
| Report parameters | Gross/net, benchmark, classification, consolidation, date range, and grouping options can alter output. | High Confidence |

---

## 9. Common Fields, Labels, Tokens, and Artifacts

This overview lists high-value cross-repository entries. The full data dictionary remains `Chapter_15_Data_Dictionary.md`.

### 9.1 Security and Portfolio Identifiers

| Field / Token | Description | Axys | APX | IMEX | REP / Reports | Confidence |
|---|---|---:|---:|---:|---:|---:|
| `Portfolio Code` | Portfolio/account code in Axys reports/exports. | Yes | Unknown / possible analog | Unknown | Yes | Verified for Axys contexts |
| `APX Portfolio Code` | APX portfolio identifier in CI/AIA workflows. | No | Yes | Related | No | Verified for APX workflow |
| `Portfolio Name` | Axys asset export label. | Yes | Unknown | Unknown | Likely report/export | Verified for Axys export workflow |
| `Account Number` | APX account demographic field in AIA Account Blotter. | Unknown | Yes | Related | No | Verified in AIA context |
| `Custodian Account Number` | Custodian-side account identifier distinct from APX Portfolio Code in AIA caution. | Unknown | Yes | Related | No | Verified in AIA context |
| `Symbol` | Product security symbol in matching/report contexts. | Yes | Yes | Related | Likely report label | Verified in integration contexts |
| `Type` | Security type paired with symbol. | Yes | Yes | Related | Unknown | Verified in integration contexts |
| `Axys Symbol` | Axys target security symbol in CI translation. | Yes | No | Related | No | Verified for CI context |
| `APX Symbol` | APX target security symbol in CI translation. | No | Yes | Related | No | Verified for CI context |
| `Security Type` | Security type label/concept; not asset class. | Yes | Yes | Related | Report/export label in some contexts | Verified in context |
| `CUSIP` | External identifier used in matching/translations. | Matching context | Matching context | Unknown | Unknown | Verified in integration context |
| `Ticker` | External identifier used in matching/translations. | Matching context | Matching context | Unknown | Unknown | Verified in integration context |

### 9.2 Observed Security Type Codes and Prefixes

These are examples only, not a full official dictionary.

| Code / Prefix | Observed Context | Axys | APX | Confidence |
|---|---|---:|---:|---:|
| `csus` / `CSUS` | Security type examples in CI/transaction examples. | Yes | Yes | Verified as observed example |
| `efus` | Security translation example. | Yes | Yes | Verified as observed example |
| `tfus` | Axys duplicate security type example. | Yes | Unknown | Verified as observed example |
| `oaus` | Axys duplicate security type example. | Yes | Unknown | Verified as observed example |
| `adus` | APX duplicate security type example. | Unknown | Yes | Verified as observed example |
| `epus` / `exus` | Fee/expense/special security contexts; exact native meaning Unknown. | Yes in conversion context | Yes in APX context | Medium Confidence |
| `CAUS` / `caus` | Cash/security type token in transaction examples. | Yes | Yes | Verified as observed token; expansion Unknown |
| `aw`, `br`, `ex`, `ep`, `pi`, `rs` | Prefixes excluded from CI security matching. | Yes in CI context | Yes in CI context | Verified for CI only |

### 9.3 Observed Transaction Codes

This table is an observed-code catalog, not an official transaction-code matrix.

| Code | Observed Meaning / Context | Axys | APX | Confidence |
|---|---|---:|---:|---:|
| `by` | Buy. | Observed | Observed | Medium Confidence |
| `BY` | Uppercase cancellation/deletion/reversal example. | Observed | Observed | Medium Confidence |
| `sl` / `SL` | Sell / uppercase cancellation example. | Observed in contexts | Observed | Medium Confidence |
| `ss` / `SS` | Short sale / uppercase cancellation example. | Observed in contexts | Observed | Medium Confidence |
| `cs` / `CS` | Cover short / uppercase cancellation example. | Observed in contexts | Observed | Medium Confidence |
| `li` | Deliver in / transfer in / credit / deposit depending context. | Observed | Observed | Medium Confidence |
| `lo` | Deliver out / transfer out / debit / withdrawal depending context. | Observed | Observed | Medium Confidence |
| `dv` | Dividend / income / reinvestment leg in examples. | Observed in contexts | Observed | Medium Confidence |
| `in` | Income / interest in examples. | Observed in contexts | Observed | Medium Confidence |
| `rc` | Return of capital in examples. | Observed in contexts | Observed | Medium Confidence |
| `pd` | Principal paydown / bond return-of-capital case in examples. | Observed in contexts | Observed | Medium Confidence |
| `ai` | Accrued interest or margin interest depending context. | Observed in contexts | Observed | Medium Confidence |
| `sa` | Sell accrued interest in examples. | Observed in contexts | Observed | Medium Confidence |
| `dp` | Debit / fee / tax / service charge / cash-security case. | Observed | Observed | Medium Confidence |
| `wd` | Withdrawal / cash-security sell case. | Observed | Observed | Medium Confidence |
| `ti` / `to`, `si` / `so`, `tr` / `ts` | Opposite transaction pairs in intra-account cash journal removal logic. | Observed in AIA context | Observed in AIA context | High Confidence as workflow tokens |

### 9.4 Cash-Like Tokens

| Token | Description | Axys | APX | Confidence |
|---|---|---:|---:|---:|
| `$cash` | Axys CI source/destination symbol for cash. | Yes | Unknown | High Confidence for Axys CI |
| `$income` | Axys CI source/destination symbol for income-related activity. | Yes | Unknown | High Confidence for Axys CI |
| `$pty`, `$ity`, `$pth` | Source/destination type tokens in Axys CI translations; expansions Unknown. | Yes | Unknown | High Confidence as observed tokens; meaning Unknown |
| `CASH` / `cash` | Cash token in examples. | Yes | Yes | High Confidence as observed token |
| `MMF` | Money-market/sweep vehicle token. | Yes | Yes | High Confidence as observed token |
| `MARGIN` / `margin` | Margin cash/sweep token. | Yes | Yes | High Confidence as observed token |
| `caus margin` | Margin/cash context used in observed negative-interest or margin-interest mappings. | Yes | Unknown | High Confidence as observed token; native definition Unknown |
| `SHORT` / `short` | Short cash/sweep token. | Yes | Yes | High Confidence as observed token |
| `dvwash`, `dvshrt`, `dvlong`, `cashrt`, `calong`, `income` | Special symbols excluded from sweep-removal logic in AIA workflows; `dvwash` also appears in reinvestment-linking evidence. | Yes | Yes | High Confidence as observed tokens; native definitions Unknown |

### 9.5 REP Expressions and Report Labels

| Field / Expression | Description | Axys | APX | IMEX | REP / Reports | Confidence |
|---|---|---:|---:|---:|---:|---:|
| `AMAN.REP` | Axys AUM report file in example. | Yes | Unknown | No | Yes | Verified for example |
| `$:fileo` | Replang token displaying portfolio code in Axys AUM example. | Yes | Unknown | No | Yes | Verified for example |
| `$askport` | Replang variable used in Portfolio Appraisal header example. | Yes | Unknown | No | Yes | Verified for example |
| `$:tfile` | Replang token described as showing CLI file containing a transaction. | Yes | Unknown | No | Yes | Verified as consultant statement |
| `$firmg` | Replang variable used as “Other” sector catch-all in AUM sector example. | Yes | Unknown | No | Yes | Verified for example |
| `#~8portmv` | Prints portfolio market value in `AMAN.REP` example. | Yes | Unknown | No | Yes | Verified for example |
| `\n` | Line break / carriage return marker in Replang example. | Yes | Unknown | No | Yes | Verified for example |
| `.` prefix | Print command marker in Replang example. | Yes | Unknown | No | Yes | Verified for example |
| `#width`, `#cnt` | Layout expressions observed in report example; full semantics Unknown. | Yes | Unknown | No | Yes | Medium Confidence |

---

## 10. Examples

### 10.1 Axys IMEX / Trade Blotter Workflow Example

Supported in CI workflow evidence only.

```text
External financial institution data
    ↓
Custodial Integrator download / translation
    ↓
Axys Security Information and Security Type Information (`sec.inf`, `type.inf`)
    ↓
Generated transaction / position / price files
    ↓
Axys Import/Export utility (`imex32.exe` in CI context)
    ↓
Trade Blotter / position / price import
    ↓
IMEX log review
    ↓
Acceptance step in integration workflow
```

| Step | Evidence Status |
|---|---:|
| Security/type information used for mapping | Verified for CI workflow |
| Transactions delivered to `topost.trn` | Verified for CI workflow |
| Prices imported with `imexPrices.log` review | Verified for CI workflow |
| Position lots may use `imexPositionLots.log` | Verified for CI workflow |
| Native object names and full field layouts | Unknown |

### 10.2 APX AIA / Blotter Workflow Example

Supported in APX AIA/CI workflow evidence only.

```text
Custodian / external source files
    ↓
AIA / integration processing
    ↓
APX security/account/transaction translation
    ↓
APX IMEX / APXIX / blotter import workflow
    ↓
Trade Blotter, Position Blotter, Tax Lot Blotter, Statement Blotter, or Account Blotter
    ↓
Review / reconciliation / posting workflow
    ↓
Reports, logs, and downstream controls
```

| Step | Evidence Status |
|---|---:|
| Trade Blotter exists in APX workflows | Verified |
| Position Blotter exists in CI workflow | Verified |
| Tax Lot / Lot Blotter exists when enabled in workflow | Verified |
| Account Blotter used for account demographics in AIA workflow | Verified |
| Native APX blotter schemas and states | Unknown |

### 10.3 Security Translation Example Pattern

The supplied security master and IMEX chapters support a practical integration identity model:

```text
External identifier(s)
    WP Name
    WP Ticker
    WP Cusip
    Institution
    WP Account #
        ↓
Product security identity
    Axys Symbol + Type
    APX Symbol + Type
```

| Rule | Confidence |
|---|---:|
| Preserve symbol and type together. | Verified in CI context |
| Do not treat symbol alone as unique. | High Confidence |
| Do not treat symbol + type as the formal native primary key without more evidence. | Unknown |

### 10.4 Transaction Cancellation Example

The supplied transaction, IMEX, and glossary chapters include an uppercase-code cancellation example.

```csv
acct123,010101,010101,by,csus,appl,100,caus,cash,10000
acct123,010101,010101,BY,csus,appl,100,caus,cash,10000
```

| Interpretation | Confidence |
|---|---:|
| Lowercase `by` appears as buy in the example. | Medium Confidence |
| Uppercase `BY` appears as a cancellation/deletion/reversal example in the workflow. | Medium Confidence |
| Column meanings and universal native semantics are fully known. | Unknown |

### 10.5 Axys REP Customization Example

The supplied REP/report chapters include an Axys `AMAN.REP` example.

```replang
.#~8portmv\n
.#~8portmv $:fileo\n
#width #cnt 16* 25+ 16+
#width #cnt 16* 35+ 16+
```

| Element | Evidence Status |
|---|---:|
| `AMAN.REP` is an Axys AUM report in the example. | Verified for example |
| `$:fileo` displays portfolio code in the example. | Verified for example |
| `#~8portmv` prints portfolio market value in the example. | Verified for example |
| Full RepLang grammar and cross-version behavior | Unknown |

---

## 11. Known Issues / Quirks

| Issue / Quirk | Axys | APX | Confidence | Practical Effect |
|---|---:|---:|---:|---|
| Direct Axys file access is version-sensitive. | Yes | N/A | Medium / High Confidence | Prefer IMEX, REP, or controlled exports unless file format is verified for the specific version. |
| Axys file conversion/version changes can alter formats. | Yes | N/A | Medium Confidence | Version-specific validation is required. |
| Report labels are not native fields. | Yes | Yes | High Confidence | Do not map report labels directly to IMEX fields or APX tables without evidence. |
| Report output parsing is layout-sensitive. | Yes | Yes | Medium / High Confidence | Downstream parsers need controlled report source and parameters. |
| Symbol alone can be ambiguous. | Yes | Yes | Verified in CI examples | Preserve symbol + type and source context. |
| Security type is not asset class/sector/classification. | Yes | Yes | High Confidence | Do not conflate identity/security-type fields with classification fields. |
| Uppercase transaction-code cancellation is observed but not universal. | Yes | Yes | Medium Confidence | Treat as workflow evidence, not complete native rule. |
| Code-only external-flow classification is unsafe. | Yes | Yes | High Confidence as design rule | Treat `li`/`lo` as external-flow candidates, and route ambiguous `li`, `lo`, `dp`, `wd`, `;`, `epus`, and `exus` cases to review until context confirms treatment. |
| Cash sweep and intra-account journal removal are integration-tool behaviors. | Yes | Yes | High Confidence for workflow; native behavior Unknown | Do not assume Axys/APX natively suppresses or removes these transactions. |
| Price import can fail when a target Axys price file is open/in use. | Yes | No evidence | Verified for CI workflow | Review `imexPrices.log`. |
| APX report names are not data sources by themselves. | N/A | Yes | Verified caution | Report names do not prove table names, views, or stored procedures. |
| APX public views may be limited. | N/A | Yes | Medium Confidence | SQL extract coverage must be validated. |
| APX cloud/hosted environments may restrict direct access or execution. | Unknown | Unknown | Unknown | Requires environment-specific evidence. |

---

## 12. Version Differences and Environment Notes

| Version / Environment Topic | Axys | APX | Confidence |
|---|---|---|---:|
| Axys v3.x IMEX era | Consultant evidence says IMEX was introduced in Axys v3.x era. | N/A | Medium / High Confidence |
| Axys 3.7 to 3.8 file conversion | Consultant evidence says conversion was required and some file formats changed. | N/A | Medium Confidence |
| Axys 3.8.6 | Connector documentation lists Axys 3.8.6 as supported minimum for one connector. | N/A | Verified for connector only |
| Axys 3.8.7 | Supplied REP/report research notes enhanced Position Reconciliation, generic date framework expansion, and additional/improved multicurrency reports. | N/A | Verified as release-note evidence |
| APX v1.x–v4.x IMEX | Consultant evidence says APX retained IMEX but eliminated fixed-format file generation and could export Axys v3 format. | Yes | Medium Confidence |
| APX 15.2, 16.1, 16.2, 17.1 | Salentica connector documentation lists these APX versions as supported for that connector. | Yes | Verified for connector only |
| APX 3.0 reporting | Supplied APX architecture research says APX 3.0 introduced SSRS-based reporting framework. | Yes | High Confidence |
| Current APX REST/API behavior | Practitioner evidence mentions REST API in recent APX contexts. | Yes | Medium Confidence; endpoint details Unknown |

---

## 13. Repository Use Guidance

### 13.1 Safe Documentation Practices

| Practice | Reason | Confidence |
|---|---|---:|
| Cite the specific chapter/source basis when extending a claim. | Prevents unsupported generalization. | High Confidence |
| Separate Axys and APX behavior. | Architecture and interface paths differ. | Verified repository rule |
| Preserve Unknowns explicitly. | Unknown is preferable to invented certainty. | Verified repository rule |
| Treat integration behavior as integration behavior. | AIA/CI/Data Broker evidence may not be native platform behavior. | High Confidence |
| Record version, environment, and extraction path. | File/report/IMEX behavior may vary. | High Confidence |
| Prefer tables, examples, field dictionaries, and known quirks. | Repository standard. | Verified |

### 13.2 Unsafe Assumptions

| Unsupported Assumption | Status |
|---|---:|
| Axys has a SQL schema that can be queried like APX. | Unknown / unsupported |
| APX table names can be inferred from report names. | Unsupported |
| `portperf` and `secperf` are official IMEX object names. | Unknown |
| Symbol alone uniquely identifies securities. | Unsafe |
| Security type equals asset class or sector. | Unsafe |
| Cash balances are available through a known native IMEX object. | Unknown |
| Report labels are native fields. | Unsafe |
| Uppercase transaction codes are universal cancellation codes in every context. | Unknown |
| Axys and APX use identical IMEX objects and fields. | Unknown |
| Performance reports always read stored values or always recalculate. | Unknown |
| Historical classification reports use historical classifications. | Unknown |

---

## 14. Highest-Priority Unknowns

The following Unknowns recur across chapters and should be carried forward until resolved by vendor documentation, actual exports, report source, schema evidence, or controlled tests.

| Unknown | Why It Matters | Evidence Needed |
|---|---|---|
| Official Axys IMEX object names | Required for implementable Axys interfaces. | Axys IMEX manual, screenshots, object list, sample exports/imports. |
| Official APX IMEX object names | Required for implementable APX interfaces. | APX IMEX manual, screenshots, object list, sample exports/imports. |
| Complete transaction-code matrix | Required for safe transaction interpretation. | Vendor transaction-code documentation or production examples with reconciliation. |
| Native Axys file layouts | Required before direct file parsing. | Sanitized files plus version-specific documentation. |
| Native APX table/view/function names | Required before APX SQL integration. | APX public view/schema documentation or controlled SQL extracts. |
| Security-master field layouts for `sec.inf` and `type.inf` | Required for reliable security import/export. | Sanitized files or vendor field dictionary. |
| Price file layouts and price-source rules | Required for valuation audit and price import/export. | `.pri` samples, price import docs, APX price-set docs. |
| Cash-balance storage/extract method | Required for cash reconciliation. | Cash reports, IMEX exports, APX SQL/public views, or vendor docs. |
| Holdings stored-versus-calculated behavior | Required for repeatability and audit. | Current and historical holdings tests, report source, SQL/file evidence. |
| Performance stored-versus-recalculated behavior | Required for performance audit and historical-change analysis. | Controlled rerun tests, performance exports, report source, APX stored function docs. |
| Security/classification history behavior | Required for historical attribution and classification reporting. | Classification extracts across dates, report tests, schema/docs. |
| REP field dictionary and grammar | Required for robust custom report coding. | RepLang documentation or report source library. |
| APX SSRS report datasets and stored procedures | Required for APX report-to-data reconciliation. | RDL files, SSRS catalog, stored procedure/view documentation. |
| APX ACA/Reorg Utility output fields | Required for corporate-action audit. | ACA/APX download samples, Trade Blotter exports, Reorg Utility docs. |

---

## 15. References

This chapter is based on the following supplied repository material only.

| Source | Description |
|---|---|
| `AXYS_APX_REFERENCE_BLUEPRINT.md`, Version 2.0 | Governing specification for editorial standards, structure, confidence labels, and repository success criteria. |
| `Chapter_02_Axys_Architecture.md` | Axys architecture, file artifacts, IMEX, REP, data-domain dependencies, version cautions. |
| `Chapter_03_APX_Architecture.md` | APX architecture, platform role, SSRS/reporting, IMEX, SQL/access paths, blotter concepts. |
| `Chapter_04_Security_Master.md` | Security identity, symbol/type matching, `sec.inf`, `type.inf`, translations, field evidence. |
| `Chapter_05_Transactions.md` | Transaction lifecycle, blotters, observed transaction codes, external-flow classification, translation and cancellation examples. |
| `Chapter_06_Holdings.md` | Holdings/positions, Portfolio Appraisal, `.pos`, `CDIhold.rep`, reconciliation behavior. |
| `Chapter_07_Cash.md` | Cash activity, cash-like tokens, sweeps, journals, cash-balance Unknowns. |
| `Chapter_08_Pricing.md` | Pricing, `.pri`, missing/stale/calculated price evidence, APX price-set workflow. |
| `Chapter_09_Corporate_Actions.md` | Axys `split.inf`, APX ACA/Reorg Utility workflow, corporate-action Unknowns. |
| `Chapter_10_Performance.md` | Performance capability, APX attribution/contribution labels, stored/recalculated Unknowns. |
| `Chapter_11_Classifications.md` | Classification concepts, security type separation, asset class/sector/country/region/custom classification evidence. |
| `Chapter_12_Imex.md` | IMEX utilities, logs, workflow artifacts, object/field dictionary status. |
| `Chapter_13_Rep.md` | REP, RepLang, Report Writer Pro, REP32, report extraction risks and examples. |
| `Chapter_14_Reports.md` | Report families, APX named reports, report-label cautions. |
| `Chapter_15_Data_Dictionary.md` | Cross-repository field, token, artifact, report, and dictionary consolidation. |
| `Chapter_16_Glossary.md` | Repository term definitions, ambiguities, and implementation notes. |

---

## 16. Chapter Summary

The supplied repository chapters are sufficient to create an overview chapter. They support a conservative technical framing:

- Axys is documented primarily through file-oriented, IMEX, REP, report, and integration evidence.
- APX is documented as a centralized platform with SSRS/reporting, IMEX, blotter, SQL/public view/stored accounting function, and connector evidence.
- Security identity is best treated as symbol plus security type in integration contexts, but formal native primary keys remain Unknown.
- Transactions, holdings, cash, prices, corporate actions, performance, and classifications are all documented at the workflow/report/artifact level, but many native schemas and official IMEX object names remain Unknown.
- REP and reports are powerful extraction paths, but report labels must not be treated as native fields.
- Unknowns are not defects in the repository; they are explicit research targets.
