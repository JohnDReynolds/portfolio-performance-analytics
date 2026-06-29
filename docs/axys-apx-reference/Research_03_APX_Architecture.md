# Research_03_APX_Architecture.md

**Repository:** AXYS / APX Reference Repository  
**Chapter target:** `Chapter_03_APX_Architecture.md`  
**Research file:** `Research_03_APX_Architecture.md`  
**Prepared:** 2026-06-29  
**Governing specification:** `AXYS_APX_REFERENCE_BLUEPRINT.md`, Version 2.0

---

## 1. Research Scope

This research file collects factual, implementation-oriented information relevant to APX architecture, with Axys comparison where the available evidence supports it.

The chapter should emphasize:

- APX architecture and platform behavior
- Differences between APX and Axys where documented
- IMEX usage and limitations
- REP / reporting architecture
- known report names
- known export/import mechanisms
- version differences
- implementation quirks
- areas where available evidence is insufficient

Per the repository blueprint, every important technical statement is classified as one of:

| Classification | Meaning in this file |
|---|---|
| Verified | Directly supported by cited source material or governing blueprint. |
| High Confidence | Strongly supported by one or more sources, but not enough to assert low-level implementation details. |
| Medium Confidence | Plausible and partially supported, but requires vendor documentation, sample files, or production confirmation. |
| Unknown | Not supported by available evidence. Do not use as fact in the chapter. |

---

## 2. Source Inventory

| Source ID | Source | Type | Notes | Confidence Use |
|---|---|---|---|---|
| S1 | `AXYS_APX_REFERENCE_BLUEPRINT.md`, Version 2.0 | User-supplied governing specification | Defines editorial rules, repository structure, confidence labels, and required standards. | Verified for repository policy only. |
| S2 | SS&C Advent APX product page, `Advent Portfolio Exchange®` | Vendor product page | Describes APX as integrated portfolio and client management, portfolio accounting/reporting, and an enterprise platform for front/middle/back offices. | Verified for product-positioning statements. Not sufficient for database internals. |
| S3 | SS&C product brief for Advent Portfolio Exchange | Vendor brief | Describes reporting framework, report customization, external-source inclusion, portals, and client reporting. | Verified for high-level reporting capabilities. |
| S4 | Advent Portfolio Exchange Reports Guide / `REP_APX.pdf` search-accessible text | Vendor report guide / report brochure | Lists APX report names and states APX reports are built on Microsoft SQL Server Reporting Services. Search-accessible text includes selected report descriptions and sample field labels. | Verified for listed report names and high-level report framework. |
| S5 | Salentica Engage Data Broker article for SS&C Advent APX & Axys | Third-party integration documentation | Describes a connector requiring Advent client tools including REP32, using standard reports, macros, RepLang scripting, and tested APX/Axys versions. | High Confidence for integration behavior; source is third-party, not SS&C. |
| S6 | Salentica Data Broker setup article | Third-party integration documentation | Describes importing `.mac` and `.scr` files into APX using the Import Export Utility / IMEX. | High Confidence for practical IMEX setup behavior. |
| S7 | AdventGuru IMEX / integration articles | Consultant / practitioner documentation | Describes IMEX as a data-in/data-out mechanism for Axys/APX, trade blotter import for transactions and labels, APX REST API option in recent versions, and APX-to-Axys conversion via IMEX exports. | High Confidence / Medium Confidence depending on detail. |
| S8 | WatersTechnology BST Awards 2024 article | Industry article | States APX and its accounting engine are now part of SS&C Advent Genesis platform. | High Confidence for product/platform positioning; not low-level architecture. |
| S9 | Finance / industry news APX 3.0 articles | Industry articles | State APX 3.0 introduced a Microsoft SQL Server Reporting Services reporting framework, expanded data access, and enhanced CRM features. | High Confidence for release-level direction; not sufficient for schema details. |

---

## 3. Executive Research Summary

| Topic | Research Finding | Classification | Evidence |
|---|---|---|---|
| APX product role | APX is positioned as an integrated portfolio and client management solution combining portfolio accounting/reporting with relationship/client-management capabilities. | Verified | S2, S3 |
| APX platform scope | APX is described as connecting front, middle, and back offices on a single platform. | Verified | S2 |
| APX reporting framework | APX investment management reports are described as built on Microsoft SQL Server Reporting Services (SSRS). | Verified | S4, S9 |
| APX standard reports | Public APX report-guide text lists reports including Account Distribution, Attribution by Classification, Attribution Summary, Contribution reports, Risk Statistics, Portfolio Appraisal, Realized Gains and Losses, and Transaction Summary. | Verified | S4 |
| APX report customization | APX reporting materials describe customizable reports, branding, charts/graphs, customized data elements, and inclusion of data from Advent suite components or external sources. | Verified | S3, S4 |
| REP32 integration dependency | A third-party Advent connector requires Advent client tools, specifically REP32, and uses standard reports, macros, and RepLang scripting for extraction. | High Confidence | S5 |
| IMEX availability | Practitioner and integration documentation identify IMEX / Import Export Utility as a mechanism used with Axys and APX. | High Confidence | S6, S7 |
| IMEX importing `.mac` / `.scr` | Third-party setup documentation describes importing `.mac` and `.scr` files into APX through IMEX. | High Confidence | S6 |
| APX REST API | Practitioner documentation states recent APX versions include a RESTful API option not available when earlier IMEX-focused articles were written. | Medium Confidence | S7 |
| APX-to-Axys export | Practitioner documentation says APX data can be exported via IMEX to Axys 3 format for items such as prices, portfolios, splits, security information, sectors, industries, asset classes, indexes, and composites. | Medium Confidence | S7 |
| APX performance history export | Practitioner documentation says exporting APX performance history via IMEX for Axys import can be frustrating and version-dependent. | Medium Confidence | S7 |
| APX database table names | No verified table names were available in the supplied blueprint or accessible public sources. | Unknown | Gap |
| APX exact IMEX object names | No authoritative IMEX object list was available in the supplied blueprint or accessible public sources. | Unknown | Gap |
| APX internal accounting engine storage model | Public material identifies APX as a portfolio accounting/reporting product, but does not document internal storage tables or recalculation logic. | Unknown | Gap |

---

## 4. APX Architecture — Verified and Supported Facts

### 4.1 Product and Platform Role

| Statement | Classification | Notes / Evidence |
|---|---|---|
| APX is an integrated portfolio and client management solution. | Verified | Vendor product page / brief. |
| APX is positioned for portfolio accounting and reporting. | Verified | Vendor product page / brief. |
| APX is described as connecting front, middle, and back offices on a single platform. | Verified | Vendor product page. |
| APX is part of the broader Advent Investment Suite / SS&C Advent solution ecosystem. | Verified | Vendor product page and third-party industry descriptions. |
| APX supports client reporting and communications through reports and portal integrations. | Verified | Vendor product brief. |
| APX can be deployed locally or cloud-delivered, according to third-party marketplace text. | Medium Confidence | Third-party source; needs vendor contract/deployment documentation before using as implementation fact. |
| APX and its accounting engine are described in a 2024 industry article as part of SS&C Advent Genesis. | High Confidence | Industry article; useful as recent product direction, not as schema evidence. |

### 4.2 Architectural Implications

| Area | Supported implication | Classification | Use in chapter? |
|---|---|---|---|
| Enterprise platform | APX should be documented as more centralized and integrated than classic file-oriented Axys, but only at a high level unless supplied docs prove internals. | High Confidence | Yes, with caution. |
| Reporting | APX reporting architecture should discuss SSRS in addition to legacy REP/REP32/RepLang integration. | Verified / High Confidence | Yes. |
| Data access | APX exposes or supports multiple data-access paths: standard reports, macros/RepLang, IMEX, and possibly REST API in recent versions. | Medium Confidence | Yes, but split by confidence and source. |
| Internal storage | APX likely uses server/database-backed architecture, but exact table and stored-procedure internals are not verified here. | Unknown | Do not assert without source. |

---

## 5. APX vs Axys — Architecture-Level Comparison

| Topic | Axys | APX | Confidence | Notes |
|---|---|---|---|---|
| Product positioning | Portfolio accounting and reporting product. | Integrated portfolio and client management solution with portfolio accounting/reporting and CRM/client management positioning. | Verified | Vendor pages support high-level distinction. |
| Platform orientation | Public material and legacy practitioner references indicate Axys is commonly treated as a workstation/reporting/report-file ecosystem. | APX is marketed as a centralized enterprise platform spanning front, middle, and back offices. | High Confidence | Do not infer storage details beyond evidence. |
| Reporting engine | Axys reporting often involves REP32 / `.rep` reports in practitioner documentation. | APX reports include SSRS-based reporting; REP32 may still be used by integrations using Advent client tools. | High Confidence | Need vendor docs to define precise coexistence. |
| IMEX | IMEX is described as moving data in and out of Axys. | IMEX / Import Export Utility is described in third-party docs as used with APX, including importing macro/script files. | High Confidence | Exact object names Unknown. |
| Trade blotter import | Practitioner source says transactions and label data can be imported through trade blotter for Axys/APX. | Same statement applies to APX in practitioner source. | Medium Confidence | Need vendor import guide for object-level detail. |
| REST API | Not established for Axys in available sources. | Practitioner source says recent APX versions have a RESTful API option. | Medium Confidence | Needs APX API docs. |
| APX-to-Axys conversion | Axys is a target format in practitioner APX-to-Axys conversion discussion. | Practitioner source recommends exporting APX data to Axys 3 format via IMEX for conversion. | Medium Confidence | Requires validation per APX version. |
| Performance history portability | Unknown from authoritative source. | Practitioner source says performance history export/import can be difficult and APX-version-dependent. | Medium Confidence | Must not generalize to all APX versions without evidence. |

---

## 6. APX Reporting Architecture

### 6.1 SSRS-Based APX Reports

| Statement | Classification | Evidence / Notes |
|---|---|---|
| APX investment management reports are described as built on Microsoft SQL Server Reporting Services. | Verified | APX Reports Guide and APX 3.0 release coverage. |
| APX 3.0 introduced a new reporting framework using Microsoft SQL Server Reporting Services. | High Confidence | Industry-release coverage. |
| APX reports can include charts, graphs, customized branding, and customized data elements. | Verified | Vendor brief / report guide. |
| APX reports can include data from other SS&C Advent solution components and external sources. | Verified | Vendor brief. |
| APX report output can support client portal delivery. | Verified | Vendor brief. |

### 6.2 REP32 / RepLang / Macro-Based Extraction

| Statement | Classification | Evidence / Notes |
|---|---|---|
| A third-party connector for Axys/APX requires Advent client tools, including REP32, installed on a client-side machine. | High Confidence | Salentica integration documentation. |
| That connector uses standard Advent reports and macros to generate extracts. | High Confidence | Salentica integration documentation. |
| That connector also uses RepLang scripting and macros. | High Confidence | Salentica integration documentation. |
| The host machine for the connector is described as a Windows machine, ideally always powered on for scheduled unattended extraction. | High Confidence | Salentica integration documentation. |
| The connector is a 32-bit Windows application. | High Confidence | Salentica integration documentation. |
| The exact APX report definitions used by the connector are not listed in available source text. | Unknown | Need connector package or report/macro files. |

### 6.3 Known APX Report Names from Public Report Guide Text

The following report names are supported by the APX Reports Guide text available in public search snippets. Use these names as report names, not as proof of internal data tables.

| Report Name | Category / Context | Evidence | Classification |
|---|---|---|---|
| Account Distribution | Business intelligence / account segmentation | APX Reports Guide table of contents and snippet | Verified |
| Account Characteristics | Business intelligence | APX Reports Guide | Verified |
| Account Characteristics (By Custodian) | Business intelligence | APX Reports Guide | Verified |
| Asset Flows | Business intelligence | APX Reports Guide | Verified |
| Business Summary Dashboard | Business intelligence | APX Reports Guide | Verified |
| Activity Profile | Analytics for portfolio managers | APX Reports Guide | Verified |
| Attribution by Classification | Performance analytics | APX Reports Guide | Verified |
| Attribution Summary | Performance analytics | APX Reports Guide | Verified |
| Attribution by Selected Groupings | Performance analytics | APX Reports Guide | Verified |
| Contribution by Classification | Performance analytics | APX Reports Guide | Verified |
| Contribution Summary | Performance analytics | APX Reports Guide | Verified |
| Contribution Detail | Performance analytics | APX Reports Guide | Verified |
| Risk Statistics | Performance / risk analytics | APX Reports Guide | Verified |
| Cover Page | Client reporting | APX Reports Guide | Verified |
| Household Overview | Client reporting | APX Reports Guide | Verified |
| Portfolio Overview | Client reporting | APX Reports Guide | Verified |
| Performance Overview | Client reporting | APX Reports Guide | Verified |
| Risk Overview | Client reporting | APX Reports Guide | Verified |
| Policy Overview | Client reporting | APX Reports Guide | Verified |
| Historical Policy Overview | Client reporting | APX Reports Guide | Verified |
| Allocation Summary | Client reporting | APX Reports Guide | Verified |
| Equity Overview | Client reporting | APX Reports Guide | Verified |
| Fixed Income Distribution | Client reporting | APX Reports Guide | Verified |
| Fixed Income Overview | Client reporting | APX Reports Guide | Verified |
| Income Projection | Client reporting | APX Reports Guide | Verified |
| Portfolio Appraisal | Client reporting / holdings | APX Reports Guide | Verified |
| Realized Gains and Losses | Client reporting / tax lots / realized gain/loss | APX Reports Guide | Verified |
| Transaction Summary | Client reporting / transaction listing | APX Reports Guide | Verified |
| Disclaimer and Terms | Client reporting | APX Reports Guide | Verified |

### 6.4 Report-Level Field Labels Seen in Public Report Text

These are report output labels observed in the publicly available APX report-guide text. They should not be treated as APX database field names.

| Label | Observed report context | Meaning / Use | Classification |
|---|---|---|---|
| Market Value | Account Distribution / Portfolio Appraisal examples | Report output measure. | Verified |
| Revenue | Account Distribution examples | Report output measure for period revenue. | Verified |
| Effective Rate / Eff. Rate | Account Distribution examples | Report output measure. | Verified |
| Count | Account Distribution examples | Report output measure. | Verified |
| AUM | Account Distribution examples | Report output measure. | Verified |
| Account # | Portfolio Appraisal / Transaction Summary examples | Report display identifier. | Verified |
| Trade Date | Transaction Summary example | Report output transaction date label. | Verified |
| Settle Date | Transaction Summary example | Report output settlement date label. | Verified |
| Quantity | Transaction Summary / Portfolio Appraisal examples | Report output quantity label. | Verified |
| Security | Transaction Summary / Portfolio Appraisal examples | Report output security description label. | Verified |
| Cost | Transaction Summary / Portfolio Appraisal examples | Report output cost label. | Verified |
| Total Cost | Transaction Summary example | Report output cost label. | Verified |
| Unit Cost | Transaction Summary example | Report output cost label. | Verified |
| Price | Transaction Summary example | Report output price label. | Verified |
| Proceeds | Transaction Summary / Realized Gains and Losses examples | Report output proceeds label. | Verified |
| Gain/Loss | Transaction Summary / Realized Gains and Losses examples | Report output gain/loss label. | Verified |
| Cost Basis | Realized Gains and Losses example | Report output cost-basis label. | Verified |
| Open Date | Realized Gains and Losses example | Report output lot open-date label. | Verified |
| Close Date | Realized Gains and Losses example | Report output lot close-date label. | Verified |
| Short Term | Realized Gains and Losses example | Report output realized-gain classification label. | Verified |
| Long Term | Realized Gains and Losses example | Report output realized-gain classification label. | Verified |
| Percent of Portfolio | Portfolio Appraisal description | Report output measure. | Verified |
| Yield | Portfolio Appraisal description | Report output measure. | Verified |
| Unrealized Gain and Loss | Portfolio Appraisal description | Report output measure. | Verified |

---

## 7. IMEX / Import Export Utility Research

### 7.1 Supported Facts

| Statement | Classification | Evidence / Notes |
|---|---|---|
| IMEX is described by practitioner sources as a tool that facilitates moving data in and out of Axys. | High Confidence | AdventGuru. |
| Practitioner sources extend the Axys/APX data-in/data-out discussion to APX as well as Axys. | High Confidence | AdventGuru. |
| Transactions and label data can be imported through the trade blotter according to practitioner documentation. | Medium Confidence | Needs vendor import guide for exact transaction codes, field names, and limits. |
| Third-party APX setup instructions describe launching the Import Export Utility / IMEX from the Advent folder. | High Confidence | Salentica setup documentation. |
| Third-party setup instructions describe importing `.mac` and `.scr` files into APX using an `Import Advent format` action in IMEX. | High Confidence | Salentica setup documentation. |
| Practitioner APX-to-Axys conversion guidance says to use IMEX to export APX data to Axys 3 format. | Medium Confidence | AdventGuru. |

### 7.2 APX-to-Axys Conversion Items Mentioned in Practitioner Source

A practitioner APX-to-Axys conversion article lists the following as items to export from APX to Axys 3 format via IMEX. This is useful for research but should be validated against actual IMEX screens/documentation before converting into authoritative chapter instructions.

| Item Mentioned | Possible Repository Topic | Classification | Notes |
|---|---|---|---|
| Prices | Pricing | Medium Confidence | Exact IMEX object name Unknown. |
| Portfolios | Portfolios / Accounts | Medium Confidence | Exact structure Unknown. |
| Splits | Corporate Actions | Medium Confidence | Exact fields Unknown. |
| Security information | Security Master | Medium Confidence | Exact fields Unknown. |
| Sectors | Classifications | Medium Confidence | Exact fields Unknown. |
| Industries | Classifications | Medium Confidence | Exact fields Unknown. |
| Asset classes | Classifications | Medium Confidence | Exact fields Unknown. |
| Indexes | Benchmarks / Indexes | Medium Confidence | Exact fields Unknown. |
| Composites | Performance / GIPS / Composites | Medium Confidence | Exact fields Unknown. |
| Performance history | Performance | Medium Confidence for difficulty; Unknown for exact export object and fields. | Practitioner source warns performance-history export/import can be version-dependent/frustrating. |

### 7.3 IMEX Unknowns That Should Be Preserved

| Question | Current Status | Required Evidence |
|---|---|---|
| What are the exact APX IMEX object names? | Unknown | APX IMEX documentation, screenshots, sample `.mac` / `.scr`, or production exports. |
| Does APX IMEX use the same object names as Axys IMEX? | Unknown | Comparative APX/Axys IMEX documentation or side-by-side exports. |
| Which APX IMEX objects export transactions, holdings, prices, security master, classifications, performance, composites? | Unknown | Vendor IMEX guide or actual export menus/files. |
| Are APX IMEX files fixed-width, CSV, tab-delimited, Advent-format, or object-specific? | Unknown | Sample exports. |
| Does APX IMEX export stored performance or recalculate at export time? | Unknown | Vendor documentation or controlled production test. |
| Are APX IMEX export formats stable across APX 15.x, 16.x, 17.x, and later? | Unknown | Versioned documentation or regression sample files. |

---

## 8. REP / Reporting Research

### 8.1 REP, REP32, RepLang, and APX

| Statement | Classification | Evidence / Notes |
|---|---|---|
| REP32 exists as an Advent client reporting tool used by third-party connectors for Axys/APX data extraction. | High Confidence | Salentica Data Broker article. |
| RepLang scripting and macros can be used by at least one connector to automate report-based extraction. | High Confidence | Salentica Data Broker article. |
| The connector uses standard Advent reports and macros to generate extracts. | High Confidence | Salentica Data Broker article. |
| The exact relationship between APX SSRS reports and legacy REP/REP32 reports is not fully documented in available sources. | Unknown | Need vendor APX reporting architecture guide. |
| The extension `.rep` is associated with Advent reports in practitioner sources, but exact APX usage needs validation. | Medium Confidence | AdventGuru snippets mention `.rep` reports; not enough for APX chapter details. |

### 8.2 Reporting Architecture Model for Chapter Drafting

Use this as a cautiously worded model, not as a low-level implementation claim:

| Layer | Description | Confidence |
|---|---|---|
| APX application layer | User-facing portfolio/accounting/client-management application. | Verified at product level. |
| APX SSRS reporting layer | APX report framework built on Microsoft SQL Server Reporting Services for standard/custom reports. | Verified / High Confidence. |
| Advent client tools layer | REP32, macros, and RepLang may be installed and used by integrations for extraction. | High Confidence for integration pattern. |
| IMEX layer | Import Export Utility used for Advent-format imports/exports and setup/import of `.mac` / `.scr` files. | High Confidence for usage, Unknown for object internals. |
| Data storage layer | Internal APX persistence layer. Exact schema, object model, and storage/recalculation behavior are not verified. | Unknown. |

---

## 9. Version and Deployment Research

### 9.1 Version References Found

| Version / Release | Statement | Classification | Evidence / Notes |
|---|---|---|---|
| APX 3.0 | APX 3.0 introduced a new reporting framework using Microsoft SQL Server Reporting Services, expanded data access, and enhanced CRM features. | High Confidence | Industry release coverage. |
| APX 15.2 / 16.1 / 16.2 / 17.1 | A third-party connector was tested on and supported these APX versions. | High Confidence | Salentica integration documentation. This is connector support, not APX vendor lifecycle. |
| Recent APX versions | Practitioner source says RESTful API became available in recent APX versions. | Medium Confidence | Need official APX API documentation. |
| APX with Genesis | 2024 industry source states APX and its accounting engine are now part of SS&C Advent Genesis. | High Confidence | Product direction; not enough for implementation details. |

### 9.2 Version Differences to Preserve

| Topic | Known Difference / Concern | Classification | Required Evidence |
|---|---|---|---|
| APX 3.0 reporting | SSRS reporting introduced in APX 3.0 according to release coverage. | High Confidence | Vendor release notes would upgrade to Verified. |
| APX performance history export | Practitioner source says performance-history export for Axys import is version-dependent/frustrating. | Medium Confidence | Actual APX versioned IMEX docs and sample exports. |
| APX REST API | Practitioner source says recent versions include REST API option. | Medium Confidence | APX API manual. |
| APX cloud / Genesis behavior | APX is described as cloud-deliverable and connected with Genesis in current product ecosystem. | Medium / High Confidence | Need SS&C deployment docs for technical chapter. |

---

## 10. Known Implementation Quirks / Practical Observations

| Quirk / Observation | System | Classification | Practical implication |
|---|---|---|---|
| APX extraction may still depend on installed Advent client tools such as REP32 for some third-party integrations. | APX / Axys | High Confidence | Integration servers/workstations may need Advent client software, Windows environment, and credentials. |
| At least one connector is a 32-bit Windows application. | APX / Axys connector context | High Confidence | Scheduling/unattended processing may require a persistent Windows host. |
| Standard reports, macros, and RepLang scripts are used by at least one integration path. | APX / Axys | High Confidence | Changes to reports/macros/scripts can affect integrations. |
| `.mac` and `.scr` files may be imported into APX through IMEX for integration setup. | APX | High Confidence | Integration packages may modify or add APX report/macro/script artifacts. |
| APX-to-Axys conversion through IMEX may be straightforward for some reference/static data but problematic for performance history. | APX | Medium Confidence | Performance history migration requires special validation and should not be assumed portable. |
| Newer APX environments may have REST API options in addition to older IMEX/report-based approaches. | APX | Medium Confidence | Chapter should not present IMEX/REP as the only possible APX integration path for all versions. |
| Public materials list report names and report output labels, but not database columns. | APX | Verified | Do not map report labels directly to APX tables/columns without documentation. |

---

## 11. Field Dictionary — Architecture-Level Only

No verified APX database field dictionary was available from the supplied blueprint or public sources reviewed. The following table records only architecture-level identifiers and report-output labels observed in source text. These should not be treated as database fields.

| Field / Label | Description | Axys | APX | IMEX | REP / Reports | Confidence |
|---|---|---|---|---|---|---|
| Account # | Account identifier displayed in APX sample reports. | Unknown | Report output label observed. | Unknown | Seen in Portfolio Appraisal / Transaction Summary examples. | Verified as label only. |
| Market Value | Report output measure. | Unknown | Observed in APX report-guide text. | Unknown | Seen in Account Distribution / Portfolio Appraisal examples. | Verified as label only. |
| Revenue | Report output measure. | Unknown | Observed in APX Account Distribution text. | Unknown | APX report-guide text. | Verified as label only. |
| Effective Rate / Eff. Rate | Report output measure. | Unknown | Observed in APX Account Distribution text. | Unknown | APX report-guide text. | Verified as label only. |
| AUM | Report output measure. | Unknown | Observed in APX Account Distribution text. | Unknown | APX report-guide text. | Verified as label only. |
| Trade Date | Transaction date label in Transaction Summary report. | Unknown | Observed in APX report-guide text. | Unknown | Transaction Summary. | Verified as label only. |
| Settle Date | Settlement date label in Transaction Summary report. | Unknown | Observed in APX report-guide text. | Unknown | Transaction Summary. | Verified as label only. |
| Quantity | Holding/transaction quantity label. | Unknown | Observed in APX report-guide text. | Unknown | Portfolio Appraisal / Transaction Summary. | Verified as label only. |
| Security | Security description label. | Unknown | Observed in APX report-guide text. | Unknown | Portfolio Appraisal / Transaction Summary. | Verified as label only. |
| Cost | Cost label. | Unknown | Observed in APX report-guide text. | Unknown | Transaction Summary. | Verified as label only. |
| Total Cost | Cost label. | Unknown | Observed in APX report-guide text. | Unknown | Transaction Summary. | Verified as label only. |
| Unit Cost | Cost label. | Unknown | Observed in APX report-guide text. | Unknown | Transaction Summary. | Verified as label only. |
| Price | Price label. | Unknown | Observed in APX report-guide text. | Unknown | Transaction Summary. | Verified as label only. |
| Proceeds | Sale proceeds label. | Unknown | Observed in APX report-guide text. | Unknown | Transaction Summary / Realized Gains and Losses. | Verified as label only. |
| Gain/Loss | Gain/loss output label. | Unknown | Observed in APX report-guide text. | Unknown | Transaction Summary / Realized Gains and Losses. | Verified as label only. |
| Cost Basis | Cost-basis output label. | Unknown | Observed in APX report-guide text. | Unknown | Realized Gains and Losses. | Verified as label only. |
| Open Date | Lot open-date output label. | Unknown | Observed in APX report-guide text. | Unknown | Realized Gains and Losses. | Verified as label only. |
| Close Date | Lot close-date output label. | Unknown | Observed in APX report-guide text. | Unknown | Realized Gains and Losses. | Verified as label only. |
| Short Term | Realized gain/loss term bucket. | Unknown | Observed in APX report-guide text. | Unknown | Realized Gains and Losses. | Verified as label only. |
| Long Term | Realized gain/loss term bucket. | Unknown | Observed in APX report-guide text. | Unknown | Realized Gains and Losses. | Verified as label only. |

---

## 12. Candidate Architecture Diagram for Chapter Use

This diagram is a research-level conceptual map. It should be labeled conceptual unless validated against APX technical documentation.

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

Unknown: the internal APX data storage layer, table names, stored procedures, and calculation engines are not documented by the available source set.

---

## 13. Research Questions for Chapter Completion

The following items should be requested from the user or collected from vendor/production material before writing a highly detailed APX architecture chapter.

### 13.1 Highest-Priority Missing Evidence

| Need | Why It Matters |
|---|---|
| APX technical architecture guide or administrator guide | Needed to verify server components, database dependencies, service topology, permissions, and deployment model. |
| APX IMEX user guide or screenshots | Needed to identify exact IMEX object names, export/import types, file formats, and version differences. |
| Sample APX IMEX exports | Needed to create field dictionaries and examples without inventing fields. |
| Sample APX REP / SSRS report definitions | Needed to document report parameters, datasets, report names, and extraction logic. |
| APX database schema documentation | Needed before documenting database tables, keys, stored procedures, or direct SQL extraction. |
| APX REST API documentation for the client’s version | Needed to document modern API access accurately. |
| Version-specific APX release notes | Needed to separate APX 3.0, 15.x, 16.x, 17.x, Genesis-era behavior, and API availability. |
| Production observations from APX environments | Needed to document quirks, scheduling, permissions, report execution behavior, performance runtime, and integration fragility. |

### 13.2 Questions to Ask a Practitioner

| Question | Target Chapter Section |
|---|---|
| Which APX version is in use? | Version differences |
| Is APX deployed locally, hosted, cloud-delivered, or through Genesis? | Architecture / deployment |
| Are reports run through APX UI, SSRS, REP32, or all of these? | Reporting |
| Are integrations based on IMEX, REP/RepLang, direct SQL, SSRS subscriptions, REST API, or vendor connectors? | Integration architecture |
| What exact IMEX objects are used for security master, portfolios, transactions, prices, holdings, performance, classifications, and composites? | IMEX |
| Are APX exports scheduled through Windows Task Scheduler, APX scheduler, SSRS subscriptions, connector software, or another scheduler? | Processing behavior |
| Does the firm permit direct SQL access to APX databases? | Data model / integration |
| Does the firm rely on stored performance, report-calculated performance, or both? | Performance architecture |
| Which report outputs are treated as books-and-records vs analytical/client-facing output? | Reports / governance |
| What breaks after APX upgrades? | Known quirks |

---

## 14. Unsupported / Unknown Information

The following statements must not be asserted in the chapter unless future source material verifies them.

| Unsupported Claim | Status | Comment |
|---|---|---|
| APX stores transactions in a specific named SQL table. | Unknown | No table names verified. |
| APX stores security master fields in specific named SQL columns. | Unknown | No schema verified. |
| APX IMEX transaction object is named a specific value. | Unknown | Object list not available. |
| APX performance reports always use stored monthly performance. | Unknown | No evidence. |
| APX performance reports always recalculate dynamically. | Unknown | No evidence. |
| APX and Axys IMEX outputs are identical. | Unknown | Conversion source suggests export to Axys 3 format is possible for many items, but not identity. |
| APX REST API exists in all current installations. | Unknown | Practitioner source indicates recent versions; licensing/configuration/version unknown. |
| REP32 is required for all APX reporting. | Unknown | Evidence only supports specific third-party extraction workflows; APX also has SSRS reports. |
| SSRS reports expose all accounting data needed for third-party analytics. | Unknown | Public report guide lists reports but not datasets or completeness. |
| APX report labels are database fields. | Unknown / False as stated | Report labels should not be treated as schema fields without evidence. |
| APX direct SQL querying is supported by SS&C for client integrations. | Unknown | Need vendor support statement or contract terms. |
| APX Genesis-era architecture changes database layout. | Unknown | Public article identifies product/platform positioning only. |

---

## 15. Recommended Use in `Chapter_03_APX_Architecture.md`

### Use as Verified / High Confidence

- APX is an integrated portfolio and client management platform.
- APX includes portfolio accounting and reporting capabilities.
- APX has SSRS-based investment management reports.
- APX standard report names include the report list captured in Section 6.3.
- Third-party APX/Axys integrations may use REP32, standard reports, macros, and RepLang.
- IMEX / Import Export Utility is used in APX-related integration setup and data movement workflows.
- Some known third-party connector support references APX versions 15.2, 16.1, 16.2, and 17.1.

### Use only as Medium Confidence

- Recent APX versions may support a RESTful API.
- APX-to-Axys conversion can export many data categories to Axys 3 format through IMEX.
- APX performance-history export/import may be version-dependent and difficult.

### Do Not Use Except as Unknown

- Specific APX database table names.
- Specific APX database column names.
- Exact APX IMEX object names.
- APX calculation-engine internals.
- Stored-vs-recalculated performance behavior.
- Direct SQL supportability.
- Exact relationship among APX SSRS, REP32, `.rep`, `.mac`, and `.scr` artifacts beyond cited integration examples.

---

## 16. References

1. `AXYS_APX_REFERENCE_BLUEPRINT.md`, Version 2.0. User-supplied governing specification.
2. SS&C Advent, `Advent Portfolio Exchange®` product page. Public vendor page accessed via web search on 2026-06-29.
3. SS&C Advent, `Product Brief: Advent Portfolio Exchange`. Public vendor brief accessed via web search on 2026-06-29.
4. SS&C Advent, `Advent Portfolio Exchange Reports Guide` / `REP_APX.pdf`. Public vendor report guide text available via search snippets and mirrored text.
5. Salentica Engage, `Data Broker - SS&C|Advent APX & Axys`. Public third-party connector documentation accessed 2026-06-29.
6. Salentica Data Broker, `Setup the Salentica Advent Connector for Data Broker`. Public third-party setup documentation accessed 2026-06-29.
7. AdventGuru, IMEX / integration articles including `Getting Data In and Out of Advent APX and Axys` related tag pages and APX integration posts. Practitioner/consultant source accessed 2026-06-29.
8. AdventGuru, `There and Back Again: APX to Axys Conversion`, 2019-05-22. Practitioner/consultant source accessed 2026-06-29.
9. WatersTechnology, `BST Awards 2024: Best portfolio accounting platform — SS&C Advent`, 2024-11-01. Industry article accessed 2026-06-29.
10. Finextra / WealthBriefing / related APX 3.0 release coverage. Industry release coverage accessed 2026-06-29.

---

## 17. Appendix — Evidence Notes by Source

### S1 — Governing Blueprint

The blueprint requires facts-first documentation, separation of Axys and APX, preservation of Unknowns, use of field dictionaries, examples, version differences, known quirks, and confidence labels. It defines this research file as part of the repository structure.

### S2 / S3 — Vendor APX Product Material

Vendor material supports APX product-level statements: integrated portfolio and client management, portfolio accounting/reporting, front/middle/back-office platform positioning, reporting and communications, customizable report presentation, and use with client portals.

### S4 — APX Reports Guide

The APX Reports Guide supports the SSRS reporting-framework statement and the report-name inventory. It also provides visible report output labels, but those labels are not database field names.

### S5 / S6 — Salentica Data Broker Documentation

The Salentica material supports a practical integration pattern: installed Advent client tools, REP32, standard reports, macros, RepLang scripting, Windows host, and APX version references for connector support. It also supports importing `.mac` and `.scr` artifacts into APX via IMEX for connector setup.

### S7 / S8 — Practitioner / Industry Sources

Practitioner sources are valuable for implementation quirks, especially IMEX, conversion, and REST API observations. They should be treated as High Confidence only when they describe direct practical procedures and Medium Confidence when extrapolated to broader APX behavior.

