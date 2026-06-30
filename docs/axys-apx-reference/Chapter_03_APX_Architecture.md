# Chapter 03 — APX Architecture

**Repository:** AXYS / APX Reference Repository
**Chapter:** `Chapter_03_APX_Architecture.md`
**Status:** Draft technical reference chapter
**Prepared:** 2026-06-29
**Governing specification:** `AXYS_APX_REFERENCE_BLUEPRINT.md`, Version 2.0

---

## Related chapters
- [Chapter_01_Overview.md](Chapter_01_Overview.md) — provides the repository-wide map and evidence conventions.
- [Chapter_02_Axys_Architecture.md](Chapter_02_Axys_Architecture.md) — provides the Axys counterpart for architecture and workflow terms.
- [Chapter_12_Imex.md](Chapter_12_Imex.md) — covers the APX import/export context.
- [Chapter_14_Reports.md](Chapter_14_Reports.md) — ties architecture to the report families and report-label evidence.

## 1. Overview

This chapter documents the architecture of **SS&C Advent Portfolio Exchange (APX)** using only the supplied repository research and source material.

The chapter is intentionally conservative. Where the supplied material supports a product-level statement but does not support implementation-level details, the implementation detail is marked **Unknown**.

### 1.1 Confidence Labels

| Classification | Meaning in this chapter |
|---|---|
| Verified | Directly supported by supplied research that cites vendor material, report guide text, third-party implementation documentation, or the repository blueprint. |
| High Confidence | Strongly supported by one or more supplied sources, but not enough to assert low-level implementation details such as table names, stored procedures, or exact field layouts. |
| Medium Confidence | Plausible and partially supported, but requires vendor documentation, sample files, APX exports, screenshots, or production confirmation. |
| Unknown | Not supported by the supplied material. Do not treat as fact. |

### 1.2 Scope

This chapter covers APX architecture at the level supported by the supplied material:

- product and platform role;
- APX versus Axys architectural differences;
- reporting architecture, including SSRS and REP32/RepLang evidence;
- IMEX / Import Export Utility evidence;
- known APX report names;
- architecture-level field and label evidence;
- version references;
- known quirks and implementation cautions;
- important Unknowns.

This chapter does **not** document unsupported APX database tables, stored procedures, internal services, calculation engines, or native schema names.

---

## 2. APX Product and Platform Role

| Statement | Classification | Notes |
|---|---:|---|
| APX is an integrated portfolio and client management solution. | Verified | Supported by supplied APX architecture research based on SS&C product material. |
| APX includes portfolio accounting and reporting capabilities. | Verified | Supported by supplied APX architecture research and APX product material. |
| APX is positioned as connecting front, middle, and back offices on a single platform. | Verified | Product-level vendor statement captured in the supplied research. |
| APX is part of the broader SS&C Advent solution ecosystem. | Verified | Supported by supplied APX architecture research. |
| APX supports client reporting and communications through reports and portal-oriented output. | Verified | Supported by supplied APX product brief research. |
| APX is described in current ecosystem research as connected with Advent Genesis and its accounting engine. | High Confidence | Based on industry-source research; useful as product direction, not schema evidence. |
| APX can be locally deployed or cloud-delivered according to supplied product-level material. | Verified / Medium Confidence depending source | Product pages support delivery positioning; implementation topology remains Unknown. |

### 2.1 Architectural Implication

APX should be documented as a **centralized, integrated platform** rather than as a classic Axys-style file-oriented application. However, the supplied material does not provide enough evidence to document APX server topology, database schemas, table names, stored procedures, service names, scheduling components, or calculation-engine internals.

| Architecture Area | Supported Treatment | Classification |
|---|---|---:|
| Application scope | Portfolio/accounting/reporting/client-management platform. | Verified |
| Reporting | SSRS-based report framework plus evidence of REP32/RepLang use in some integration workflows. | Verified / High Confidence |
| Integration | IMEX, standard reports/macros, REP32/RepLang, SQL/reporting tools, and possibly REST API in recent versions. | High / Medium Confidence |
| Internal persistence | Exact APX database schema and storage behavior are not verified. | Unknown |
| Calculation engine | Stored-versus-recalculated behavior for performance, holdings, or other values is not verified. | Unknown |

---

## 3. APX Versus Axys Architecture

| Topic | Axys | APX | Classification | Notes |
|---|---|---|---:|---|
| Product orientation | Portfolio accounting/reporting system. | Integrated portfolio and client-management platform with accounting/reporting/client-management positioning. | Verified | Vendor product material supports high-level distinction. |
| Platform orientation | Supplied research characterizes Axys as file-oriented/proprietary in public practitioner sources. | APX is positioned as centralized and enterprise-oriented. | High Confidence | Do not infer unverified APX table names. |
| Data architecture | Axys direct file access appears in practitioner and integration research; file formats are version-sensitive. | APX has SQL/database/reporting access options in practitioner research, but exact schema is Unknown. | Medium / High Confidence | APX SQL access is an architectural distinction, but supportability and schema details are Unknown. |
| Reporting engine | Axys reporting often involves `.REP`, Replang, Report Writer Pro, and REP32. | APX reports include SSRS-based reporting; REP32/RepLang may still appear in integration workflows. | High Confidence | Relationship between SSRS and REP32 in APX is not fully documented. |
| IMEX | Axys IMEX is evidenced as the Axys Import/Export utility in third-party guides. | APX IMEX / Import Export Utility is evidenced in APX setup and integration workflows. | High Confidence | Exact APX object names Unknown. |
| Trade/blotter workflow | Axys transaction imports can flow through `topost.trn` / Trade Blotter in integration evidence. | APX has Trade Blotter, Position Blotter, Tax Lot Blotter, Statement Blotter, and other blotter concepts in integration evidence. | High Confidence for cited workflows | Native full lifecycle needs vendor docs. |
| REST API | Not established for Axys in supplied material. | Practitioner source says recent APX versions may include a RESTful API option. | Medium Confidence | Version/licensing/coverage Unknown. |
| APX-to-Axys conversion | Axys may be a target format. | Practitioner guidance says APX data can be exported to Axys 3 format through IMEX for selected categories. | Medium Confidence | Exact object support and version behavior Unknown. |

---

## 4. Conceptual APX Architecture

The following diagram is a **conceptual map** based on supplied research. It must not be read as a verified internal APX deployment topology.

```text
                              +-----------------------------+
                              | SS&C Advent APX Application |
                              | Portfolio / CRM / Reporting |
                              +--------------+--------------+
                                             |
                 +---------------------------+---------------------------+
                 |                           |                           |
        +--------v---------+        +--------v---------+        +--------v---------+
        | APX SSRS Reports |        | Advent Client    |        | IMEX / Import    |
        | Standard/custom  |        | Tools / REP32    |        | Export Utility   |
        +--------+---------+        +--------+---------+        +--------+---------+
                 |                           |                           |
        +--------v---------+        +--------v---------+        +--------v---------+
        | Client reports   |        | Macros / RepLang |        | Advent-format    |
        | Portals / output |        | Extracts         |        | imports/exports  |
        +------------------+        +------------------+        +------------------+
                                             |
                                      +------v-------+
                                      | Integrations |
                                      | CRM / ETL    |
                                      +--------------+
```

| Layer | Description | Classification |
|---|---|---:|
| APX application layer | User-facing portfolio, accounting, client-management, and reporting application. | Verified at product level |
| APX reporting layer | Standard/custom APX reporting, including SSRS-based reports. | Verified / High Confidence |
| Advent client tools layer | REP32, macros, and RepLang used by at least one APX/Axys connector. | High Confidence |
| IMEX layer | Import Export Utility used in APX setup/import/export workflows. | High Confidence |
| Data storage layer | Internal APX persistence layer. Exact tables/views/stored procedures are not supplied. | Unknown |
| API layer | Recent APX versions may include REST API access. | Medium Confidence |

---

## 5. APX Reporting Architecture

### 5.1 SSRS-Based APX Reports

| Statement | Classification | Notes |
|---|---:|---|
| APX investment-management reports are described as built on Microsoft SQL Server Reporting Services. | Verified | Supported by supplied APX architecture research citing APX Reports Guide text. |
| APX 3.0 introduced a reporting framework using Microsoft SQL Server Reporting Services. | High Confidence | Based on supplied industry-release research. |
| APX reports can include charts, graphs, customized branding, and customized data elements. | Verified | Supported by supplied APX product/reporting research. |
| APX reports can include data from Advent suite components and external sources. | Verified | Supported by supplied APX product brief research. |
| APX report output can support client portal delivery. | Verified | Supported by supplied APX product brief research. |
| APX SSRS report dataset names, stored procedures, report server paths, and RDL internals are not documented in supplied material. | Unknown | Requires APX reporting/administrator documentation or production report definitions. |

### 5.2 REP32 / RepLang / Macro-Based Extraction

| Statement | Classification | Notes |
|---|---:|---|
| A third-party connector for Axys/APX requires Advent client tools, including `REP32.exe`, installed on a client-side machine. | High Confidence | Connector-specific evidence. |
| That connector uses standard Advent reports and macros to generate extracts. | High Confidence | Connector-specific evidence. |
| That connector uses the REP32 engine plus RepLang scripting and macros. | High Confidence | Connector-specific evidence. |
| The connector host is described as a Windows machine and ideally always powered on for scheduled unattended extraction. | High Confidence | Connector-specific evidence. |
| At least one connector is a 32-bit Windows application. | High Confidence | Connector-specific evidence. |
| REP32 is required for all APX reports. | Unknown | Supplied evidence supports specific connector workflows only. |
| The exact relationship between APX SSRS reports and legacy REP/REP32 reports is not documented in supplied material. | Unknown | Requires APX reporting architecture guide. |

### 5.3 Report Technology Boundary

| Reporting Path | APX Evidence | Appropriate Repository Treatment | Classification |
|---|---|---|---:|
| SSRS standard/custom reports | APX Reports Guide and APX 3.0 reporting research. | First-class APX reporting architecture. | Verified / High Confidence |
| REP32 / RepLang / macros | Third-party connector uses these tools for APX/Axys extraction. | Supported integration/report extraction path; do not treat as universal APX report engine. | High Confidence |
| Report Writer Pro / Replang direct editing | Practitioner sources say APX can still use these approaches. | Mention cautiously; exact version support Unknown. | Medium Confidence |
| Report packaging / portal delivery | Vendor product brief supports report packaging and portal/client reporting. | Product-level capability; implementation details Unknown. | Verified at product level |
| Direct SQL / public views / stored accounting functions | Practitioner research supports APX database/reporting access options. | APX architectural distinction; exact schema and supportability Unknown. | Medium Confidence |
| REST API | Practitioner research says recent APX versions may include REST API. | Mention as possible in recent versions only. | Medium Confidence |

---

## 6. Known APX Report Names

The following report names are supported by the supplied APX report-guide research. Treat them as **report names**, not as database tables or APX schema fields.

### 6.1 Business Intelligence / Account Segmentation

| Report Name | Context | Classification |
|---|---|---:|
| Account Distribution | Business intelligence / account segmentation | Verified |
| Account Characteristics | Business intelligence | Verified |
| Account Characteristics (By Custodian) | Business intelligence | Verified |
| Asset Flows | Business intelligence | Verified |
| Business Summary Dashboard | Business intelligence | Verified |

### 6.2 Analytics / Performance / Risk

| Report Name | Context | Classification |
|---|---|---:|
| Activity Profile | Analytics for portfolio managers | Verified |
| Attribution by Classification | Performance analytics | Verified |
| Attribution Summary | Performance analytics | Verified |
| Attribution by Selected Groupings | Performance analytics | Verified |
| Contribution by Classification | Performance analytics | Verified |
| Contribution Summary | Performance analytics | Verified |
| Contribution Detail | Performance analytics | Verified |
| Risk Statistics | Performance / risk analytics | Verified |

### 6.3 Client Reporting / Holdings / Transactions

| Report Name | Context | Classification |
|---|---|---:|
| Cover Page | Client reporting | Verified |
| Household Overview | Client reporting | Verified |
| Portfolio Overview | Client reporting | Verified |
| Performance Overview | Client reporting | Verified |
| Risk Overview | Client reporting | Verified |
| Policy Overview | Client reporting | Verified |
| Historical Policy Overview | Client reporting | Verified |
| Allocation Summary | Client reporting | Verified |
| Equity Overview | Client reporting | Verified |
| Fixed Income Distribution | Client reporting | Verified |
| Fixed Income Overview | Client reporting | Verified |
| Income Projection | Client reporting | Verified |
| Portfolio Appraisal | Holdings / portfolio appraisal | Verified / Medium Confidence depending source capture |
| Realized Gains and Losses | Realized gain/loss / tax lots | Verified |
| Transaction Summary | Transaction listing | Verified |
| Disclaimer and Terms | Client reporting | Verified |

### 6.4 Report Names Are Not Data Sources by Themselves

| Caution | Classification | Reason |
|---|---:|---|
| APX report names do not prove APX table names. | Verified caution | Report names and labels are presentation/report artifacts. |
| APX report labels do not prove database column names. | Verified caution | Supplied research explicitly warns not to map report labels to tables/columns without schema evidence. |
| A report may show calculated values, stored values, or values retrieved through stored accounting functions; the supplied material does not identify which. | Unknown | Requires report definitions, APX schema, or controlled tests. |

---

## 7. IMEX / Import Export Utility

### 7.1 Supported APX IMEX Facts

| Statement | Classification | Notes |
|---|---:|---|
| IMEX / Import Export Utility is used in APX-related integration workflows. | High Confidence | Supported by supplied APX architecture and IMEX research. |
| Third-party APX setup instructions describe launching the Import Export Utility / IMEX from the Advent folder. | High Confidence | Connector/setup-specific evidence. |
| Third-party setup instructions describe importing `.mac` and `.scr` files into APX using an `Import Advent format` action in IMEX. | High Confidence | Integration setup evidence. |
| APX AIA documentation includes Advent IMEX Log and Advent IMEX History Log tools. | Verified for AIA workflow | Useful for troubleshooting importing and blotter issues in that workflow. |
| Practitioner APX-to-Axys conversion guidance says to use IMEX to export APX data to Axys 3 format. | Medium Confidence | Conversion-practitioner evidence; object coverage and versions Unknown. |
| Exact APX IMEX object names are not supplied. | Unknown | Requires APX IMEX manual, screenshots, export definitions, or sample output. |
| Exact APX IMEX field layouts are not supplied. | Unknown | Requires sample exports/imports or vendor documentation. |

### 7.2 APX-to-Axys Conversion Items Mentioned in Research

A practitioner APX-to-Axys conversion source lists these items as export candidates from APX to Axys 3 format through IMEX. These should be treated as **Medium Confidence conversion leads**, not official object names.

| Item Mentioned | Repository Topic | Classification | Notes |
|---|---|---:|---|
| Prices | Pricing | Medium Confidence | Exact IMEX object name Unknown. |
| Portfolios | Accounts / portfolios | Medium Confidence | Exact structure Unknown. |
| Splits | Corporate actions | Medium Confidence | Exact fields Unknown. |
| Security information | Security master | Medium Confidence | Exact fields Unknown. |
| Sectors | Classifications | Medium Confidence | Exact fields Unknown. |
| Industries | Classifications | Medium Confidence | Exact fields Unknown. |
| Asset classes | Classifications | Medium Confidence | Exact fields Unknown. |
| Indexes | Benchmarks / indexes | Medium Confidence | Exact fields Unknown. |
| Composites | Performance / GIPS / composites | Medium Confidence | Exact fields Unknown. |
| Performance history | Performance | Medium Confidence for difficulty; Unknown for object/fields | Research warns export/import may be version-dependent and frustrating. |

### 7.3 APX IMEX Unknowns

| Unknown | Why It Matters | Needed Evidence |
|---|---|---|
| Exact APX IMEX object names. | Required for implementable interface documentation. | APX IMEX manual, screenshots, sample `.mac`/`.scr`, export definitions. |
| Whether APX IMEX object names match Axys IMEX. | Required for cross-system interface design. | Side-by-side Axys/APX IMEX documentation or exports. |
| Which APX IMEX objects export transactions, holdings, prices, securities, classifications, performance, composites. | Required for repository success criteria. | Vendor IMEX guide or actual APX exports. |
| Whether APX IMEX files are fixed-width, CSV, tab-delimited, Advent-format, or object-specific. | Required for parsing/import. | Sample exports/imports. |
| Whether APX IMEX exports stored performance or recalculates at export time. | Required for performance audit/reproducibility. | Vendor documentation or controlled test. |
| Whether APX IMEX formats are stable across APX 15.x, 16.x, 17.x, and later. | Required for version-safe interfaces. | Versioned docs or regression samples. |

---

## 8. APX Data Model — What Is Known and Unknown

### 8.1 Architecture-Level Entities

These entities are supported at the product/workflow level. Their native APX table names and internal keys are **Unknown** unless otherwise stated.

| Entity | APX Evidence | Classification | Native Storage Status |
|---|---|---:|---|
| Portfolio / account | APX is a portfolio/accounting platform; APX Portfolio Code appears in integration evidence. | Verified / High Confidence | Table/key Unknown |
| Client / relationship / household | APX is positioned as integrated client/relationship platform. | Verified | Table/key Unknown |
| Security master | APX security matching uses APX Symbol and APX Security Type in integration evidence. | Verified for integration context | Table/key Unknown |
| Transactions | APX tracks transactions; Trade Blotter and Transaction Summary evidence exists. | Verified / High Confidence | Table/key Unknown |
| Holdings / positions | APX tracks holdings; Position Blotter and Portfolio Appraisal evidence exists. | Verified / High Confidence | Table/key Unknown |
| Prices | APX price-file and price-source evidence exists in AIA/pricing research. | High Confidence for workflow | Table/key Unknown |
| Cash | Cash-related transactions and symbols appear in APX integration examples. | High Confidence for workflow | Native cash model Unknown |
| Performance | APX includes performance analytics and reports. | Verified at product/report level | Stored vs recalculated Unknown |
| Classifications | APX reports include custom classification, industry group, sector, attribution/contribution reports. | Verified at report level | Storage/history Unknown |
| Benchmarks / indexes | APX performance analytics can use benchmark/index data in supplied performance research. | Verified at product level | Table/key Unknown |
| Composites | APX supports composite management for GIPS compliance at product level. | Verified | Implementation Unknown |
| Blotters | Trade/position/tax-lot/statement/account blotters appear in APX integration research. | Verified for workflows | Native states/transitions Unknown |

### 8.2 APX Security Identity Evidence

| Field / Concept | Description | APX | Classification | Caution |
|---|---|---:|---:|---|
| `APX Symbol` | APX security symbol used by Custodial Integrator security translations; example value `524659208`. | Yes | Verified for CI context | Not proven as database column name. |
| `APX Security Type` / `APX Type` | APX security type used with symbol; example `efus`. | Yes | Verified for CI context | Not proven as database column name. |
| Symbol + type pairing | Security matching uses both APX symbol and security type in CI workflows. | Yes | Verified for CI context | Formal APX primary key Unknown. |
| Duplicate security conditions | Same symbol with different security types, or ticker/CUSIP duplicate definitions, can cause matching ambiguity. | Yes | Verified for CI context | Integration behavior; native constraints Unknown. |

### 8.3 Report Output Labels Observed in APX Research

These are **report output labels**, not verified APX database fields.

| Field / Label | Description | Axys | APX | IMEX | REP / Reports | Classification |
|---|---|---:|---:|---:|---:|---:|
| Account # | Account identifier displayed in APX sample reports. | Unknown | Report output label | Unknown | Portfolio Appraisal / Transaction Summary | Verified as label only |
| Market Value | Report output measure. | Unknown | Observed | Unknown | Account Distribution / Portfolio Appraisal | Verified as label only |
| Revenue | Report output measure. | Unknown | Observed | Unknown | Account Distribution | Verified as label only |
| Effective Rate / Eff. Rate | Report output measure. | Unknown | Observed | Unknown | Account Distribution | Verified as label only |
| AUM | Assets under management measure. | Unknown | Observed | Unknown | Account Distribution | Verified as label only |
| Trade Date | Transaction date label. | Unknown | Observed | Unknown | Transaction Summary | Verified as label only |
| Settle Date | Settlement date label. | Unknown | Observed | Unknown | Transaction Summary | Verified as label only |
| Quantity | Holding/transaction quantity label. | Unknown | Observed | Unknown | Portfolio Appraisal / Transaction Summary | Verified as label only |
| Security | Security description label. | Unknown | Observed | Unknown | Portfolio Appraisal / Transaction Summary | Verified as label only |
| Cost | Cost label. | Unknown | Observed | Unknown | Transaction Summary | Verified as label only |
| Total Cost | Cost label. | Unknown | Observed | Unknown | Transaction Summary | Verified as label only |
| Unit Cost | Unit cost label. | Unknown | Observed | Unknown | Transaction Summary | Verified as label only |
| Price | Price label. | Unknown | Observed | Unknown | Transaction Summary | Verified as label only |
| Proceeds | Sale proceeds label. | Unknown | Observed | Unknown | Transaction Summary / Realized Gains and Losses | Verified as label only |
| Gain/Loss | Gain/loss label. | Unknown | Observed | Unknown | Transaction Summary / Realized Gains and Losses | Verified as label only |
| Cost Basis | Cost-basis label. | Unknown | Observed | Unknown | Realized Gains and Losses | Verified as label only |
| Open Date | Lot open-date label. | Unknown | Observed | Unknown | Realized Gains and Losses | Verified as label only |
| Close Date | Lot close-date label. | Unknown | Observed | Unknown | Realized Gains and Losses | Verified as label only |
| Short Term | Realized gain/loss term bucket. | Unknown | Observed | Unknown | Realized Gains and Losses | Verified as label only |
| Long Term | Realized gain/loss term bucket. | Unknown | Observed | Unknown | Realized Gains and Losses | Verified as label only |
| Percent of Portfolio | Portfolio Appraisal measure. | Unknown | Observed | Unknown | Portfolio Appraisal | Verified as label only |
| Yield | Portfolio Appraisal measure. | Unknown | Observed | Unknown | Portfolio Appraisal | Verified as label only |
| Unrealized Gain and Loss | Portfolio Appraisal measure. | Unknown | Observed | Unknown | Portfolio Appraisal | Verified as label only |

### 8.4 Architecture-Level Field Dictionary

| Field | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---:|
| `APX Portfolio Code` | Portfolio identifier referenced in APX integration/reconciliation workflows. | No | Yes | Related in CI/AIA workflows | Unknown | Verified for workflow |
| Custodian Account Number | Custodian-side account identifier; distinct from APX Portfolio Code in AIA guide cautions. | Unknown | Yes | Related | Unknown | Verified for workflow |
| Account Number | Account demographic field example in APX Account Blotter workflow. | Unknown | Yes | Related | Unknown | Verified for AIA workflow |
| Account Name | Account demographic field example in APX Account Blotter workflow. | Unknown | Yes | Related | Unknown | Verified for AIA workflow |
| Account Type | Account demographic field example in APX Account Blotter workflow. | Unknown | Yes | Related | Unknown | Verified for AIA workflow |
| `APX Symbol` | Security symbol used in APX security translation examples. | No | Yes | Related | Unknown | Verified for CI workflow |
| `APX Security Type` | Security type used with APX Symbol. | No | Yes | Related | Unknown | Verified for CI workflow |
| `sec.inf` | Security information file used in APX CI context. | Unknown | Yes in CI context | Source/import context | Unknown | Verified for CI context |
| `type.inf` | Security type information file used in APX CI context. | Unknown | Yes in CI context | Source/import context | Unknown | Verified for CI context |
| `APXIX.exe` / `apxix.exe` | APX import/export function or utility referenced in APX integration research. | No | Yes | Utility | No | Verified for integration contexts; naming/version details Unknown |
| `ApxIx` | APX Import/Export utility terminology in APX CI research. | No | Yes | Utility | No | Verified in CI context; relationship to `APXIX.exe` Unknown |
| Trade Blotter name | APX blotter into which transactions are imported. | Unknown | Yes | Related | Unknown | Verified for CI/AIA workflows |
| Position Blotter name | APX blotter into which positions are imported for reconciliation. | Unknown | Yes | Related | Unknown | Verified for CI workflow |
| Lot Blotter name | APX blotter into which position lots are imported when enabled. | Unknown | Yes | Related | Unknown | Verified for CI workflow |
| `SourceId` | Price source field shown in APX AIA price context. | No | Yes | Price import context | Unknown | Verified for AIA context only |
| `.mac` | Macro file imported into APX using IMEX in setup workflow. | Unknown | Yes | Import setup | Macro/report-related | High Confidence |
| `.scr` | Script file imported into APX using IMEX in setup workflow. | Unknown | Yes | Import setup | Script/report-related | High Confidence |

---

## 9. APX Processing Behavior

### 9.1 General Processing Areas

| Process | APX Behavior Supported by Research | Classification | Notes |
|---|---|---:|---|
| Portfolio accounting | APX tracks holdings, transactions, performance, positions, cash, and multiple asset types at product level. | Verified | Exact processing sequence Unknown. |
| Reporting | APX provides standard/custom reports and client reporting. | Verified | SSRS evidence strong; report internals Unknown. |
| Performance analytics | APX provides performance analytics and performance-related reports. | Verified | Stored/recalculated behavior Unknown. |
| Transaction import/review | APX integration evidence uses Trade Blotter and other blotters. | High Confidence | Workflow-specific evidence; native state model Unknown. |
| Position reconciliation | APX CI/AIA research references Position Blotter and position reconciliation workflows. | High Confidence | Exact APX native reconciliation model Unknown. |
| Security matching | APX CI requires APX symbol/type resolution and may require translation. | Verified for CI | Native APX security constraints Unknown. |
| Corporate actions | ACA for APX sends APX holdings to ACA, cross-references actions, runs APX Reorg Utility, and posts transactions to APX Trade Blotter. | Verified for ACA workflow | Final transaction fields/codes Unknown. |
| Price import/update | APX AIA pricing settings include update/add/replace, price set logic, and custodian-specific pricing options. | Verified for AIA workflow | Native price schema Unknown. |

### 9.2 Blotter Concepts in APX Integration Evidence

| Blotter | Purpose in Supplied Research | Classification | Caution |
|---|---|---:|---|
| Trade Blotter | Transaction import/review destination in APX workflows. | Verified for workflows | Full native state model Unknown. |
| Statement Blotter | Used to post or reconcile custodian statement transactions in AIA workflow. | Verified for AIA workflow | Native relationship to posted transactions Unknown. |
| Tax Lot Blotter | Used for lot-level reconciliation in AIA workflow. | Verified for AIA workflow | Available only where lots enabled/configured. |
| Position Blotter | Used for imported positions/reconciliation. | Verified for CI/AIA workflows | Full schema Unknown. |
| Account Blotter | Used for account demographic import in AIA workflow. | Verified for AIA workflow | Full schema Unknown. |
| Initial Transaction Blotter | Used to create initial deliver-in transactions from positions in AIA workflow. | Verified for AIA workflow | AIA behavior unless vendor docs confirm native generality. |
| Dividend Adjustment Blotter | Mentioned in APX AIA research. | Verified for AIA workflow | Full behavior Unknown. |

### 9.3 Security Matching and Translation

| Behavior | Classification | Notes |
|---|---:|---|
| CI must determine APX security type and APX security symbol for positions and transactions imported into APX. | Verified | APX CI context. |
| Security matching may fail if the security is missing, insufficient identifier information is available, or more than one APX security matches. | Verified | APX CI context. |
| Security translations take precedence over other security matches in the CI workflow. | Verified | APX CI context. |
| Same APX symbol can exist with different security types in duplicate examples. | Verified | APX CI context. |
| A security may be defined once by ticker and once by CUSIP, creating ambiguity. | Verified | APX CI context. |
| APX security types with prefixes `aw`, `br`, `ex`, `ep`, `pi`, and `rs` are excluded from CI matching. | Verified for CI context | Do not generalize to APX platform-wide security type behavior without vendor documentation. |
| CI does not modify APX Security Type or Security Information as part of security translation. | Verified | APX CI context. |

---

## 10. Version References and Version Differences

| Version / Release | Statement | Classification | Notes |
|---|---|---:|---|
| APX 3.0 | APX 3.0 introduced a new reporting framework using Microsoft SQL Server Reporting Services, expanded data access, and enhanced CRM features. | High Confidence | Industry-release research; vendor release notes would strengthen. |
| APX 15.2 / 16.1 / 16.2 / 17.1 | A third-party connector was tested/supported on these APX versions. | High Confidence for connector | Connector support does not equal APX vendor lifecycle. |
| Recent APX versions | Practitioner source says RESTful API became available in recent APX versions. | Medium Confidence | Needs official APX API documentation and version/license confirmation. |
| APX v1.x to v4.x | Practitioner source says APX maintained IMEX functionality but eliminated fixed-format generation. | Medium Confidence | Needs versioned IMEX docs. |
| APX with Genesis | 2024 industry research states APX and its accounting engine are part of SS&C Advent Genesis. | High Confidence for platform direction | Does not document schema or deployment topology. |
| Cloud/local delivery | Product material says APX can be local or cloud-delivered. | Verified at product level | REP/IMEX/client-tool implications Unknown. |

---

## 11. Known Issues / Quirks

| Quirk / Issue | System | Classification | Practical Impact |
|---|---|---:|---|
| APX extraction may still depend on installed Advent client tools such as REP32 for some third-party integrations. | APX / Axys | High Confidence | Integration hosts may need Windows, Advent client tools, credentials, and scheduling controls. |
| At least one connector is a 32-bit Windows application. | APX / Axys connector context | High Confidence | May require a persistent Windows host. |
| Standard reports, macros, and RepLang scripts are used by at least one integration path. | APX / Axys | High Confidence | Report/script changes can break integrations. |
| `.mac` and `.scr` files may be imported into APX through IMEX for integration setup. | APX | High Confidence | Integration packages may add or modify APX macro/script artifacts. |
| APX-to-Axys conversion may be straightforward for some reference/static data but difficult for performance history. | APX | Medium Confidence | Performance migration needs special validation. |
| Newer APX environments may have REST API options in addition to older IMEX/report-based approaches. | APX | Medium Confidence | Do not present IMEX/REP as the only APX integration option for all versions. |
| Public material lists report names and output labels, but not database fields. | APX | Verified | Do not map report labels directly to APX tables/columns. |
| Security symbol alone can be ambiguous. | APX / Axys CI context | Verified | Use symbol + security type when available. |
| Blotter locks/open blotters can block imports in at least one APX integration workflow. | APX | Verified for AIA workflow | Operational process should ensure blotters are closed/unlocked before import. |
| Security master size can affect integration runtime. | APX CI context | Verified | Larger security masters and higher transaction/position volume may slow translation. |
| Price-file update/replace settings can materially alter APX valuation inputs in AIA workflow. | APX | Verified for AIA workflow | Replace Entire File is high-risk and needs control. |
| APX public views are useful but may be limited. | APX | Medium Confidence | Do not assume public views expose all data needed for extracts. |

---

## 12. Examples

### 12.1 APX Security Translation Example

Source research gives an APX security translation example:

| External Field | Example Value |
|---|---|
| WP Ticker | `LMNVX` |
| WP Name | `LEGG MASON VLE TR INSTL` |
| APX Symbol | `524659208` |
| APX Type | `efus` |

| Technical Point | Classification |
|---|---:|
| APX matching/translation can map an external ticker to an APX symbol that is not the ticker. | Verified |
| APX security identity in this workflow uses both symbol and type. | Verified |
| `efus` is an example APX security type code in the integration source. | Verified |
| Whether `efus` is universal across APX versions/sites is Unknown. | Unknown |

### 12.2 APX Duplicate Security Cases

| Duplicate Condition | Example / Explanation | Classification |
|---|---|---:|
| Same symbol, different security types | Example research includes same symbol under different security types such as `ktc csus` and `ktc adus`. | Verified for CI context |
| Same security defined by ticker and by CUSIP | Guide states a security may be defined twice, once with ticker as symbol and once with CUSIP as symbol. | Verified for CI context |
| Multiple overlapping translations | More than one CI translation can match the same security. | Verified for CI context |

### 12.3 APX Cancel Transaction Example

An APX AIA example shows a previously posted transaction:

```csv
acct123,010101,010101,by,csus,appl,100,caus,cash,10000
```

becoming:

```csv
acct123,010101,010101,BY,csus,appl,100,caus,cash,10000
```

| Technical Point | Classification | Caution |
|---|---:|---|
| The AIA workflow uses uppercase transaction code in a cancellation example. | Verified for AIA example | Do not generalize to all APX transaction workflows. |
| Lowercase-to-uppercase behavior appears in multiple third-party transaction workflow sources. | Medium Confidence | Native universality Unknown. |
| Full APX cancellation/reversal rule set is not supplied. | Unknown | Requires vendor transaction/blotter docs or production examples. |

### 12.4 APX ACA Corporate Action Workflow

| Workflow Step | Classification | Notes |
|---|---:|---|
| APX sends updated security holdings to ACA Server. | Verified | Advent Corporate Actions for APX research. |
| ACA cross-references APX securities to an action database. | Verified | Product brief research. |
| Users can review ACA transactions or allow simpler transactions to download automatically. | Verified | Product brief research. |
| Downloaded reviewed actions cause the APX Reorg Utility to run. | Verified | Product brief research. |
| APX Reorg Utility-generated transactions post to APX Trade Blotter. | Verified | Product brief research. |
| Final APX transaction codes and field mappings for ACA-generated entries are supplied. | Unknown | Need ACA/APX sample output or vendor guide. |

---

## 13. Implementation Guidance

### 13.1 Preferred Documentation Discipline

| Guidance | Classification | Rationale |
|---|---:|---|
| Distinguish APX report labels from APX database fields. | Verified caution | Supplied research only supports report labels, not schema names. |
| Distinguish IMEX exports/imports from REP/report output. | High Confidence | They are different integration surfaces. |
| Distinguish APX SSRS reporting from REP32/RepLang extraction. | High Confidence | Both appear in research, but relationship is not fully documented. |
| Treat APX SQL/public view access as a possible architecture path, not as a complete data dictionary. | Medium Confidence | Practitioner evidence supports access options; schema Unknown. |
| Capture version, deployment model, source mechanism, report/IMEX object, parameters, date range, and output schema for any APX extract. | High Confidence | Required for reproducibility. |
| Do not assume APX and Axys IMEX outputs are identical. | Medium Confidence | APX-to-Axys format export exists as a conversion topic, not equality proof. |
| Do not document native APX table names unless supplied by APX schema or production evidence. | Verified caution | No table names are verified. |

### 13.2 Extract Design Implications

| Extract Need | Safer First Evidence Source | Notes |
|---|---|---|
| Client report values | APX standard/custom report output | Use report when the report output itself is the business-approved value. |
| Raw transactions | APX IMEX / Trade Blotter / report / SQL evidence, depending site | Exact native object Unknown. |
| Holdings/positions | APX Portfolio Appraisal, Position Blotter, APX SQL/current holdings in AIA workflow, or IMEX if documented | Date semantics must be captured. |
| Prices | APX price file/import workflow, price sets, `SourceId` if exposed | Preserve price source/price set where available. |
| Security master | APX security master export/public view/IMEX, with APX Symbol and APX Security Type | Avoid ticker-only joins. |
| Performance | APX performance reports / IMEX / SQL only after verifying stored-vs-recalculated behavior | Critical Unknown. |
| Classifications | APX reports or security/classification exports | Historical/current classification behavior Unknown. |

---

## 14. References

This chapter was prepared from the supplied repository files only.

| Reference | Supplied File | Use in This Chapter |
|---|---|---|
| Blueprint | `AXYS_APX_REFERENCE_BLUEPRINT.md` | Governing specification, chapter template, confidence discipline. |
| APX architecture research | `Research_03_APX_Architecture.md` | Primary source for APX architecture, reports, IMEX, REP, version references, quirks, Unknowns. |
| Axys architecture research | `Research_02_Axys_Architecture.md` | Axys/APX contrast, file-oriented Axys context, reporting and IMEX contrast. |
| Security master research | `Research_04_Security_Master.md` | APX Symbol, APX Security Type, `sec.inf`, `type.inf`, security matching quirks. |
| Transactions research | `Research_05_Transactions.md` | Trade Blotter, transaction translation, blotter examples, cancellation example. |
| Holdings research | `Research_06_Holdings.md` | Portfolio Appraisal, Position Blotter, holdings extract workflow. |
| Cash research | `Research_07_Cash.md` | Cash transaction representation and cash-like symbols in APX workflows. |
| Pricing research | `Research_08_Pricing.md` | APX pricing settings, price sets, `SourceId`, price-file update behavior. |
| Corporate actions research | `Research_09_Corporate_Actions.md` | ACA/APX workflow, Reorg Utility, Trade Blotter corporate-action posting. |
| Performance research | `Research_10_Performance.md` | APX performance analytics, stored-versus-recalculated Unknowns. |
| Classifications research | `Research_11_Classifications.md` | APX classification report capabilities and classification Unknowns. |
| IMEX research | `Research_12_IMEX.md` | IMEX/Import Export Utility, logs, APX/Axys differences, file examples, Unknowns. |
| REP research | `Research_13_REP.md` | REP32, RepLang, Report Writer Pro, APX SSRS/REP relationship, report quirks. |

---

## 15. Unknowns

The following items must remain **Unknown** until supported by vendor documentation, sample exports/imports, report definitions, screenshots, or production observations.

### 15.1 Architecture and Deployment Unknowns

| Unknown | Why It Matters | Needed Evidence |
|---|---|---|
| APX server topology and service components. | Required for implementation, monitoring, backups, and deployment documentation. | APX technical architecture/admin guide. |
| APX database names, table names, view names, and stored procedures. | Required before documenting SQL extraction or schema. | APX schema docs or sanitized production schema. |
| APX security/permissions model for reports, IMEX, SQL, and API. | Required for operational implementation. | APX admin/security documentation. |
| Differences among local, hosted, cloud-delivered, and Genesis-era APX deployments. | Required for version/deployment-specific guidance. | Versioned deployment docs and practitioner evidence. |

### 15.2 IMEX Unknowns

| Unknown | Why It Matters | Needed Evidence |
|---|---|---|
| Exact APX IMEX object names. | Needed for implementable interface documentation. | IMEX manual, screenshots, sample exports. |
| APX IMEX transaction object and field list. | Needed for transaction import/export chapters. | Transaction IMEX export/import samples. |
| APX IMEX security master object and field list. | Needed for security master and classification chapters. | Security IMEX samples. |
| APX IMEX performance export behavior. | Needed to answer stored-vs-recalculated performance questions. | Performance export samples and controlled tests. |
| APX IMEX log formats and error codes. | Needed for troubleshooting and audit. | IMEX logs and documentation. |
| Whether `ApxIx` and `APXIX.exe` are the same utility, different labels, or context/version-specific names. | Needed for precise executable documentation. | APX installation documentation or executable inventory. |

### 15.3 Reporting / REP / SSRS Unknowns

| Unknown | Why It Matters | Needed Evidence |
|---|---|---|
| Exact relationship between APX SSRS reports and REP32/RepLang reports. | Needed to describe APX reporting architecture accurately. | APX reporting architecture guide. |
| APX SSRS report datasets, stored procedures, and public views. | Needed for report reproduction and SQL extract design. | RDLs, report server definitions, schema docs. |
| Full APX standard report inventory. | Needed for Chapter 14 and report cross-references. | APX Reports Guide / installed report catalog. |
| REP32 command-line syntax for APX. | Needed for unattended automation. | REP32 docs or working macros/scripts. |
| Macro/script file formats for `.mac` and `.scr`. | Needed for implementation documentation. | Sample files and APX import guide. |
| Which APX reports use stored values versus recalculated values. | Critical for audit and performance reproducibility. | Report docs and controlled tests. |

### 15.4 Data Model Unknowns

| Unknown | Why It Matters | Needed Evidence |
|---|---|---|
| Formal APX security master primary key. | Needed for reliable joins and imports. | APX schema docs or vendor data dictionary. |
| Whether APX Symbol + APX Security Type is the formal native key. | CI evidence uses symbol/type, but formal key Unknown. | APX security master docs or production tests. |
| APX transaction table/field names. | Needed for transaction chapter and audit tooling. | APX schema or transaction export. |
| APX holdings/positions storage model. | Needed for holdings chapter and reconciliation. | Schema docs, position exports, controlled tests. |
| APX price key and price-source/price-set model. | Needed for pricing and valuation audits. | Price docs, price file samples, public views. |
| APX classification storage and historical behavior. | Needed for classification and performance reporting. | Classification exports, APX report definitions, controlled tests. |
| APX performance storage/recalculation model. | Needed for performance audit. | Performance schema/docs and rerun tests. |

### 15.5 Version and Supportability Unknowns

| Unknown | Why It Matters | Needed Evidence |
|---|---|---|
| Version-specific APX IMEX differences across 15.x, 16.x, 17.x, and later. | Needed for stable interfaces. | Versioned IMEX docs and samples. |
| Version-specific APX report differences. | Needed for report-based extraction stability. | Versioned APX report catalogs/RDLs. |
| REST API availability by APX version and license. | Needed before documenting API as an integration path. | Official API documentation and client-version evidence. |
| Supportability of direct SQL extraction. | Needed for client implementation risk. | SS&C support statement or contract-specific guidance. |
| Supportability of custom RepLang/REP in APX cloud-delivered environments. | Needed for integration planning. | APX cloud/admin documentation. |

---

## 16. Chapter Summary

APX is supported by the supplied material as an integrated portfolio/accounting/reporting/client-management platform with a strong reporting architecture, including SSRS-based reports and evidence of REP32/RepLang use in some integration workflows.

The strongest supported APX architecture facts are:

| Fact | Classification |
|---|---:|
| APX is an integrated portfolio and client management solution. | Verified |
| APX includes accounting, reporting, performance analytics, and client-reporting capabilities. | Verified |
| APX investment-management reports are described as built on Microsoft SQL Server Reporting Services. | Verified |
| APX standard report names include Account Distribution, Attribution by Classification, Contribution reports, Risk Statistics, Portfolio Appraisal, Realized Gains and Losses, and Transaction Summary. | Verified |
| APX integrations may use Advent client tools such as REP32, standard reports, macros, and RepLang. | High Confidence |
| APX IMEX / Import Export Utility appears in setup/import/export workflows. | High Confidence |
| APX security matching in integration workflows uses APX Symbol and APX Security Type. | Verified for CI context |
| APX has multiple integration surfaces: reports, macros/RepLang, IMEX, SQL/reporting tools, and possibly REST API in recent versions. | Medium / High Confidence depending surface |

The most important unresolved APX architecture facts are:

| Unknown | Why It Matters |
|---|---|
| Native APX database tables and columns. | Needed for direct SQL interfaces and data dictionary. |
| Exact APX IMEX object names and fields. | Needed for interface implementation. |
| Stored-versus-recalculated performance behavior. | Needed for performance audit and reproducibility. |
| Exact relationship among SSRS, REP32, Replang, macros, and report packaging. | Needed for reporting architecture. |
| Version-specific behavior across APX releases and deployment models. | Needed for safe implementation guidance. |

## 17. Deep IMEX Update

The Axys IMEX deep research adds APX contrast points that should remain
architecture-level, not field-level claims.

| Topic | Chapter treatment | Confidence |
|---|---|---:|
| APX IMEX continuity | Practitioner evidence says APX v1.x through v4.x retained IMEX functionality while fixed-format generation was eliminated. | Medium |
| APX-to-Axys exports | Conversion evidence mentions APX exporting Axys v3-format reference data such as sectors, industries, asset classes, indexes, composites, and performance history. | Medium |
| Performance history | Performance-history export through IMEX is mentioned as difficult; exact APX object and fields remain Unknown. | Medium / Unknown |
| Alternative extraction surfaces | REP/RepLang, SSRS, SQL/public views/stored accounting functions, REST, and third-party ETL may be relevant depending on deployment. | Medium / High Confidence by surface |

Do not assume APX IMEX object names, fields, command syntax, or fixed-format
behavior match Axys without APX-specific documentation or samples.
