# Research Notes: REP

Repository: AXYS / APX Reference Repository  
Target chapter: `docs/axys-apx-reference/reference/Chapter_13_Rep.md`  
Research file: `docs/axys-apx-reference/evidence/Research_13_REP.md`  
Prepared: 2026-06-29  
Governing specification: `AXYS_APX_REFERENCE_BLUEPRINT.md`, Version 2.0

---

## 0. Research Discipline

This research follows the repository blueprint requirements:

- Document factual, implementation-oriented knowledge.
- Separate Axys and APX whenever behavior differs.
- Prefer evidence from vendor documentation, reports, examples, consultant documentation, and production observations.
- Classify each technical statement as **Verified**, **High Confidence**, **Medium Confidence**, or **Unknown**.
- Do not invent field names, report behavior, transaction codes, report internals, or implementation details.

This file is intentionally conservative. Where the available evidence does not identify exact fields, command syntax, report layouts, or processing behavior, the research marks those items as **Unknown**.

---

## 1. Scope of This Research File

This research covers the **REP / RepLang / report execution layer** for Axys and APX.

The term **REP** is used here to refer to Advent report source files and report execution artifacts commonly associated with:

- `.REP` report files
- Advent **RepLang** report-writing language
- Axys Reports / Custom / Any Report workflow
- Report Writer Pro output or editable report source
- REP32 / reporting engine execution where supported
- APX reporting paths that continue to use RepLang or Report Writer Pro
- APX reporting paths that differ from REP, especially SSRS/custom reporting

This research does **not** attempt to document every Advent standard report. That belongs primarily in `14-Reports.md`. This file focuses on the REP mechanism and what is known about using REP as a technical interface.

---

## 2. Evidence Register

| ID | Source | Type | Key Evidence | Reliability |
|---|---|---|---|---|
| SRC-001 | AXYS/APX Reference Blueprint v2.0 | Repository specification | Requires facts-first documentation, confidence labels, Axys/APX separation, Unknowns rather than invention | Governing |
| SRC-002 | SS&C Advent Axys product page | Vendor product page | Axys has extensive predefined reports, easy customization, Report Writer Pro, multiple reports/graphs/objects on one page, performance reports by portfolio and categories | High |
| SRC-003 | SS&C Advent APX product page | Vendor product page | APX has vast standard report library, automated report packaging, flexible custom reporting, dashboards, performance analytics, composite management | High |
| SRC-004 | SS&C Advent Axys 3.8.7 blog post | Vendor blog | Axys 3.8.7 included enhanced Position Reconciliation report, expanded generic date framework, additional/improved multicurrency reports | High |
| SRC-005 | CSSI / Client Server Specialists PDF, “How to Add Portfolio Code to Axys Reports” | Consultant technical PDF | Axys reports are written in RepLang; AMAN.REP is Assets Under Management report; reports can be copied and edited in text editor; example fields/expressions `#~8portmv`, `$:fileo`; report path example `e:\axys34\rep` | Medium-High |
| SRC-006 | Salentica/Black Diamond Data Broker Advent Connector article | Integration documentation | Connector for Axys/APX uses Advent standard reports and macros; requires Advent Client Tools / REP32.exe; uses REP32 engine, RepLang scripting, and macros; tested minimum Axys/APX versions listed | Medium-High |
| SRC-007 | AdventGuru, “Using Visual Studio Code to Modify Advent Replang Reports in Axys and APX” | Consultant blog | Axys and APX can create reports with Report Writer Pro or direct Replang source edits; Replang remains part of Axys/APX reporting architecture; APX adds more keywords than Axys; APX also has SQL Server integration options | Medium |
| SRC-008 | AdventGuru Axys/APX category page excerpt | Consultant blog index/excerpt | APX data access options include Stored Accounting Functions, Public Views, SSRS, REST API; Report Writer Pro/Replang remain available | Medium |
| SRC-009 | Microsoft SSRS documentation | Vendor documentation for SSRS, not APX-specific | SSRS is an on-premises platform for creating, deploying, and managing paginated reports | High for SSRS, not evidence of APX-specific behavior |
| SRC-010 | APX Reports Guide search result from Advent CDN | Vendor-looking PDF search result | Indicates APX reports use Microsoft SQL Server Reporting Services platform and can be customized/branded | Medium until full guide is obtained |

Reference URLs:

- SRC-002: https://www.advent.com/solutions/axys/
- SRC-003: https://www.advent.com/solutions/advent-portfolio-exchange/
- SRC-004: https://www.advent.com/news-and-insights/blog/a-new-version-of-axys-just-in-time-for-upgrade-season/
- SRC-005: https://assets.ctfassets.net/xhy36q2d1lqu/77QC4aNbyhPo9FfmjRYNzc/d00a0d6601214601543e30e34f203626/PortfolioCodetoAxys.pdf
- SRC-006: https://engage.salentica.com/kb/article/247-data-broker-ss-c-advent-apx-axys/
- SRC-007: https://adventguru.com/2024/09/09/using-visual-studio-code-to-modify-advent-replang-reports-in-axys-and-apx/
- SRC-009: https://learn.microsoft.com/en-us/sql/reporting-services/create-deploy-and-manage-mobile-and-paginated-reports

---

## 3. Executive Summary

| Topic | Research Finding | Confidence |
|---|---|---|
| Axys REP foundation | Axys reports are written in **RepLang**, Advent's proprietary report-writing language, and report files use `.REP` filenames such as `AMAN.REP`. | Verified |
| Axys report editing | A standard Axys report can be copied, renamed, opened in a plain text editor, modified, saved, and run through **Custom / Any Report**. | Verified |
| Axys Report Writer Pro | Axys supports predefined reports and custom reports through **Axys Report Writer Pro**. | Verified |
| APX REP/Replang availability | Consultant evidence says APX users can still create reports using Report Writer Pro or by editing Replang source directly. | Medium Confidence |
| APX alternate reporting | Vendor and consultant evidence show APX has flexible custom reporting, dashboards, automated report packaging, and SQL Server/SSRS-oriented reporting options. | High Confidence |
| REP32 | Integration documentation states an Advent Connector for Axys/APX uses **REP32.exe**, the REP32 engine, standard reports, macros, and RepLang scripting to extract data. | Verified for that connector; Medium-High as general architecture evidence |
| Exact RepLang grammar | The full RepLang grammar, keyword dictionary, built-in variables, and report runtime semantics were not available in the supplied/public evidence. | Unknown |
| Exact report-field dictionary | Only a few field/expression examples are supported by the public evidence: `#~8portmv` and `$:fileo`. | Verified only for those examples |
| Vendor support for custom RepLang | Consultant PDF states Advent publishes a RepLang Programmer's Guide on request but support will not answer/debug custom RepLang report questions. | Medium-High |

---

## 4. REP Concepts and Terminology

| Term | Description | Axys | APX | Confidence |
|---|---|---|---|---|
| `.REP` file | Report source file used by Axys reports. Example: `AMAN.REP`. | Verified | Unknown / possible for Replang-based APX reports | Verified for Axys |
| REP | Short form commonly used for Advent report files/reporting layer. | High Confidence | Medium Confidence | Medium Confidence |
| RepLang | Advent proprietary Report Writing Language used to write Axys reports. | Verified | Medium Confidence for continued APX use | Verified for Axys; Medium for APX |
| Report Writer Pro | Advent reporting tool for creating/customizing reports. | Verified | Medium Confidence | High for Axys, Medium for APX |
| REP32.exe | Advent client-side reporting application/engine referenced by integration docs. | Medium-High | Medium-High | Medium-High |
| Standard report | Vendor-supplied report accessible from reporting menus. | Verified | High Confidence | High Confidence |
| Custom / Any Report | Axys menu path for running a copied/custom `.REP` report by name. | Verified | Unknown | Verified for Axys |
| Macro | Automation artifact used with standard reports by the Advent Connector. | Medium-High | Medium-High | Medium-High |
| SSRS | Microsoft SQL Server Reporting Services; APX custom reporting path according to vendor/consultant sources. | Not native Axys path in current evidence | High Confidence | High Confidence for APX reporting option |

---

## 5. Axys REP Research

### 5.1 Axys Reporting Capabilities

| Statement | Classification | Evidence / Notes |
|---|---|---|
| Axys automates portfolio reporting and accounting. | Verified | Vendor product page. |
| Axys includes an extensive library of predefined reports. | Verified | Vendor product page. |
| Axys supports report customization. | Verified | Vendor product page. |
| Axys supports Report Writer Pro for custom reports. | Verified | Vendor page says users can choose from hundreds of predefined reports or create their own with Axys Report Writer Pro. |
| Axys can place multiple reports, graphs, and objects on one page for repeated automated use across portfolio groups. | Verified | Vendor product page. |
| Axys report output can include high-impact graphics such as pie charts, line graphs, and bar charts. | Verified | Vendor product page. |
| Axys can display performance by portfolios, asset classes, sectors, countries, or regions. | Verified | Vendor product page. |
| Axys has reporting related to composites and GIPS performance measurement standards. | Verified | Vendor product page. |

### 5.2 Axys `.REP` Files and RepLang

| Statement | Classification | Evidence / Notes |
|---|---|---|
| Axys reports are written in RepLang. | Verified | CSSI PDF states: “Axys reports are written in RepLang, Advent's proprietary Report Writing Language.” |
| RepLang is Advent's proprietary report-writing language. | Verified | CSSI PDF. |
| Axys reports can be opened and edited in a plain text editor. | Verified | CSSI PDF says to open `AMAN_XX.REP` in Notepad or another text editor. |
| Word processors should not be used to edit RepLang files. | Verified | CSSI PDF says not to use Word or WordPerfect; use a text editor such as Notepad. |
| Axys displays the path and file name of the report file in the lower-left corner of the Reports window when a report dialog is displayed. | Verified | CSSI PDF. |
| Example Axys report directory: `e:\axys34\rep`. | Verified as one consultant example; not a universal path | CSSI PDF explicitly says “on my system.” |
| `AMAN.REP` is the Assets Under Management report in the CSSI example. | Verified | CSSI PDF. |
| A safe customization pattern is to copy a standard `.REP` file, rename it, and modify the copy rather than the original. | Verified | CSSI PDF describes copying `AMAN.REP` to `AMAN_XX.REP`. |
| A copied custom Axys report can be run through Axys Reports / Custom / Any Report by entering the report name without necessarily typing the `.REP` extension. | Verified | CSSI PDF instructs entering `AMAN_XX` in Custom / Any Report. |
| RepLang uses a period (`.`) to tell Axys to print. | Verified | CSSI PDF statement. |
| `\n` signals carriage return / end of printed line in the CSSI example. | Verified | CSSI PDF statement. |
| The expression `.#~8portmv\n` prints portfolio market value followed by a line break in the CSSI example. | Verified | CSSI PDF statement. |
| `$:fileo` displays portfolio code in the CSSI example. | Verified | CSSI PDF says to add `$:fileo` after `portmv` to display portfolio code. |
| `#width #cnt 16* 25+ 16+` is a line in the sample report controlling page width/spacing in the example; changing `25` to `35` provides more room. | Verified only for the sample report | CSSI PDF. |

### 5.3 Axys REP Execution and Automation

| Statement | Classification | Evidence / Notes |
|---|---|---|
| A third-party Advent Connector can use Advent standard reports and macros to generate a data extract from Axys. | Verified for that connector | Salentica/Black Diamond documentation. |
| The connector requires the Advent Client Tools, specifically REP32.exe, installed on the machine where the connector runs. | Verified for that connector | Salentica/Black Diamond documentation. |
| The connector uses the REP32 engine to extract data. | Verified for that connector | Salentica/Black Diamond documentation. |
| The connector uses RepLang scripting and macros. | Verified for that connector | Salentica/Black Diamond documentation. |
| The connector can be scheduled to run unattended or at predefined daily frequencies. | Verified for that connector | Salentica/Black Diamond documentation. |
| REP32.exe is the general Axys command-line/report execution engine. | Unknown | Public evidence shows use by a connector, but not complete invocation semantics or command-line syntax. |
| Exact macro file format used by REP32 automation. | Unknown | Not available in current evidence. |
| Exact unattended report scheduling syntax. | Unknown | Not available in current evidence. |

### 5.4 Axys Version Notes

| Version / Area | Statement | Classification | Evidence / Notes |
|---|---|---|---|
| Axys 3.8.6 | Advent Connector documentation lists Axys 3.8.6 as a tested/supported minimum version. | Verified for that connector | Salentica/Black Diamond documentation. |
| Axys 3.8.7 | SS&C Advent blog says Axys 3.8.7 added an enhanced Position Reconciliation report, expanded generic date framework, and additional/improved multicurrency reports. | Verified | Vendor blog. |
| Axys generic date framework | Expanded generic date framework gives additional flexibility in reporting and automation. | Verified | Vendor blog. |
| Report differences between Axys 3.8.6 and 3.8.7 | Unknown beyond vendor bullet points | Need release notes or report files. |
| REP language changes by Axys version | Unknown | Need RepLang Programmer's Guide or keyword list by version. |

---

## 6. APX REP and Reporting Research

### 6.1 APX Reporting Capabilities

| Statement | Classification | Evidence / Notes |
|---|---|---|
| APX is an integrated portfolio and client management solution. | Verified | Vendor product page. |
| APX includes a vast library of standard reports. | Verified | Vendor product page. |
| APX supports automated report packaging. | Verified | Vendor product page. |
| APX supports flexible custom reporting. | Verified | Vendor product page. |
| APX supports customizable dashboards. | Verified | Vendor product page. |
| APX includes performance analytics. | Verified | Vendor product page. |
| APX supports composite management for GIPS compliance. | Verified | Vendor product page. |
| APX can be deployed locally or cloud-delivered with or without outsourcing services. | Verified | Vendor product page. |

### 6.2 APX RepLang / Report Writer Pro

| Statement | Classification | Evidence / Notes |
|---|---|---|
| APX users can create reports using Report Writer Pro or by updating Replang source directly. | Medium Confidence | AdventGuru consultant source; not directly vendor-confirmed in accessible docs. |
| Replang remains part of the reporting architecture of both Axys and APX. | Medium Confidence | AdventGuru consultant source. |
| APX has more Replang keywords than Axys in current versions. | Medium Confidence | AdventGuru states Replang for Axys has roughly 100 keywords and current APX adds another 100+ keywords; exact keyword list not supplied. |
| Exact APX Replang keyword set. | Unknown | Not available in current evidence. |
| Exact APX `.REP` file locations. | Unknown | Not available in current evidence. |
| Whether every Axys `.REP` report can run unchanged in APX. | Unknown | No evidence. |
| Whether APX stores Report Writer Pro reports as `.REP` source files, database objects, or another structure in all versions. | Unknown | Not available in current evidence. |

### 6.3 APX SSRS / SQL Reporting Path

| Statement | Classification | Evidence / Notes |
|---|---|---|
| APX supports custom reporting beyond REP/Replang, including dashboards and flexible reporting. | Verified | Vendor product page. |
| APX has SQL Server-based reporting/data access options, including SSRS, according to consultant evidence. | Medium Confidence | AdventGuru. |
| APX data access/reporting options cited by consultant source include Stored Accounting Functions, Public Views, SSRS, REST API, and reporting tools that can use that infrastructure. | Medium Confidence | AdventGuru. |
| Microsoft SSRS is an on-premises platform for creating, deploying, and managing paginated reports. | Verified for SSRS generally | Microsoft documentation. |
| APX reports guide search result suggests APX reports use Microsoft SQL Server Reporting Services and can be customized/branded with firm name, logo, and graphics. | Medium Confidence | Search result from Advent CDN; obtain full PDF before treating as Verified. |
| Exact APX SSRS report dataset names, stored procedures, public views, report server paths, or deployment process. | Unknown | Need APX Reports Guide / APX Reporting Guide / production sample. |

### 6.4 APX REP Execution and Automation

| Statement | Classification | Evidence / Notes |
|---|---|---|
| A third-party Advent Connector can use Advent standard reports and macros to generate a data extract from APX. | Verified for that connector | Salentica/Black Diamond documentation. |
| The connector requires Advent Client Tools / REP32.exe for APX use. | Verified for that connector | Salentica/Black Diamond documentation. |
| The connector uses the REP32 engine, RepLang scripting, and macros for APX extraction. | Verified for that connector | Salentica/Black Diamond documentation. |
| The Advent Connector documentation lists APX versions 15.2, 16.1, 16.2, and 17.1 as tested/supported minimum platform versions for that integration. | Verified for that connector | Salentica/Black Diamond documentation. |
| Whether APX always requires REP32.exe for all REP/Replang reports. | Unknown | Evidence only covers one connector/integration. |
| Whether APX cloud-delivered environments permit direct custom REP execution by clients. | Unknown | Need vendor/admin documentation. |

---

## 7. REP vs IMEX Research Notes

The chapter `13-REP.md` should distinguish REP from IMEX without documenting IMEX in full.

| Comparison Point | REP | IMEX | Confidence |
|---|---|---|---|
| Primary purpose | Report generation and report-driven extraction/output. | Import/export interface for structured data objects. | High Confidence for REP purpose; IMEX details belong to Chapter 12 |
| Source artifact | `.REP` source files, Report Writer Pro definitions, macros, report dialogs, report engine. | IMEX object definitions and export/import formats. | Medium-High |
| Output shape | Report-shaped output; may be presentation-oriented or extract-oriented depending on report. | Data-object-shaped import/export files. | High Confidence |
| Customization | Edit RepLang, use Report Writer Pro, customize report packaging/macros. | Change object, field selection, export/import configuration. | Medium-High |
| Automation evidence | Standard reports and macros can be run by connector using REP32 engine. | Unknown in this REP research file. | Verified for connector |
| Data extraction suitability | Good when a required value exists only or most conveniently in a report, but can be brittle if report layout changes. | Good when stable structured object export exists. | Medium Confidence; general implementation observation, not vendor-verified |
| Audit use | Report output can validate/compare accounting results or expose calculated values. | IMEX can provide source-data and stored object values. | Medium Confidence |

Implementation note: Use REP as an integration surface only when the report result itself is the intended data product, or when IMEX/API/database access cannot provide the needed calculated value. REP extraction should be treated as report-version-sensitive unless field names, headings, output format, and macro parameters are controlled.

---

## 8. Report Names Identified in Sources

| Report Name | Report File / Tool Name | System | Description | Confidence |
|---|---|---|---|---|
| Assets Under Management | `AMAN.REP` | Axys | Standard Axys report used in CSSI customization example; displays market values of all portfolios in selected group as of selected date; values sorted by asset class in screenshot/example. | Verified for example |
| Position Reconciliation report | Unknown file name | Axys | Enhanced in Axys 3.8.7 with improved handling of pending/unsettled trades and management insight. | Verified as named report; file unknown |
| Multicurrency reports | Unknown | Axys | Axys 3.8.7 included additional and improved multicurrency reports. | Verified as vendor statement; report names unknown |
| Standard management and client reports | Unknown | APX | APX has vast standard report library. | Verified generally; names unknown |
| SSRS reports | Unknown | APX | APX can have SSRS reports, according to consultant/vendor-adjacent sources. | Medium Confidence |

---

## 9. Field / Expression Dictionary from Available Evidence

Only fields or expressions explicitly supported by available evidence are included.

| Field / Expression | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---|---|---|---|---|
| `#~8portmv` | Prints portfolio market value in the CSSI `AMAN.REP` example. | Yes | Unknown | No evidence | Yes | Verified for Axys sample |
| `$:fileo` | Displays portfolio code in the CSSI `AMAN.REP` example. | Yes | Unknown | No evidence | Yes | Verified for Axys sample |
| `\n` | Carriage return / end-of-line marker in printed output in the CSSI example. | Yes | Unknown | No evidence | Yes | Verified for Axys sample |
| `.` prefix | RepLang print command marker according to CSSI example. | Yes | Unknown | No evidence | Yes | Verified for Axys sample |
| `#width` | Appears in sample report line controlling page width/spacing; exact full semantics unknown. | Yes | Unknown | No evidence | Yes | Medium Confidence |
| `#cnt` | Appears in sample report line; exact full semantics unknown. | Yes | Unknown | No evidence | Yes | Medium Confidence |
| `AMAN.REP` | Assets Under Management report source file in example environment. | Yes | Unknown | No evidence | Yes | Verified for Axys sample |
| `REP32.exe` | Advent reporting application/engine used by Advent Connector. | Yes | Yes | No evidence | Yes | Verified for connector |

Do **not** infer a full RepLang variable dictionary from these examples. The expressions above are sample-supported only.

---

## 10. Processing Behavior

### 10.1 Axys Processing Behavior

| Behavior | Statement | Confidence | Evidence / Notes |
|---|---|---|---|
| Report source execution | Axys reads `.REP` files to produce report output. | Verified for sample | CSSI PDF shows report file path displayed and source edited. |
| Print command | A period (`.`) tells Axys to print. | Verified | CSSI PDF. |
| Line break | `\n` signals end of printed line / carriage return. | Verified | CSSI PDF. |
| Report path display | Axys Reports window displays report path/file name in lower-left corner when a report is selected. | Verified | CSSI PDF. |
| Custom report execution | A copied report can be run from Custom / Any Report by entering the custom report name. | Verified | CSSI PDF. |
| Report layout modification | Width/spacing values may need to be changed when adding output fields to a report. | Verified for sample | CSSI PDF changes `25` to `35`. |
| Report automation | REP32 can be used by an integration connector to extract data through standard reports/macros. | Verified for connector | Salentica/Black Diamond documentation. |
| Report calculation timing | Whether Axys REP recalculates all values at run time or reads stored performance/history depending on report. | Unknown | Needs specific report documentation and examples. |
| Report parameter handling | Exact syntax for dates, portfolio/group, reporting currency, graph flags, consolidation, and settings in `.REP` source. | Unknown | Screenshot/dialog exists, but source semantics unavailable. |

### 10.2 APX Processing Behavior

| Behavior | Statement | Confidence | Evidence / Notes |
|---|---|---|---|
| Standard reports and macros | APX can be accessed by a connector using Advent standard reports and macros. | Verified for connector | Salentica/Black Diamond documentation. |
| REP32 extraction | Connector uses REP32 engine for APX data extraction. | Verified for connector | Salentica/Black Diamond documentation. |
| Replang direct edits | APX reports can be created/modified with direct Replang source edits according to consultant source. | Medium Confidence | AdventGuru. |
| Report Writer Pro | APX users can use Report Writer Pro according to consultant source. | Medium Confidence | AdventGuru. |
| SSRS reports | APX supports or commonly uses SSRS-based reporting according to consultant and vendor-adjacent search evidence. | Medium-High | AdventGuru plus APX Reports Guide search result. |
| APX calculation source | Whether APX REP reports calculate directly from database, stored accounting functions, public views, stored performance tables, or report-specific procedures. | Unknown | Need APX report technical documentation. |
| APX report packaging internals | Exact relationship between REP, SSRS, and automated report packaging. | Unknown | Need APX reporting/packaging guide. |

---

## 11. Known Issues / Quirks

| Quirk / Issue | Axys | APX | Confidence | Notes |
|---|---|---|---|---|
| Custom RepLang support may be limited | Yes | Unknown | Medium-High | CSSI PDF says Advent publishes a RepLang Programmer's Guide on request but support will not answer/debug custom RepLang report questions. |
| Custom report development is trial-and-error-oriented | Yes | Unknown | Medium-High | CSSI PDF warns custom RepLang development is best suited for users comfortable with trial and error. |
| Edit report copies, not originals | Yes | Likely | Verified for Axys; Medium for APX | CSSI example copies `AMAN.REP` before editing. This is a general safe practice, but APX evidence not specific. |
| Use plain text editors for RepLang | Yes | Likely | Verified for Axys; Medium for APX | CSSI warns against Word/WordPerfect. Applies logically to source code, but APX-specific source handling is not verified. |
| Line-number-aware editor is helpful | Yes | Yes | Medium | AdventGuru says Notepad lacks line numbers and VSCode/Notepad++ help. |
| Replang language support in modern editors is limited | Yes | Yes | Medium | AdventGuru says Replang was not a built-in VSCode supported language. |
| APX has more reporting paths than Axys | Unknown/Not applicable | Yes | Medium | APX has SQL Server/SSRS/REST/Public Views options according to consultant source; exact boundary between reporting interfaces requires vendor docs. |
| Report extraction can depend on workstation/client tools | Yes | Yes | Verified for connector | Connector requires REP32.exe and client tools on a Windows machine. |
| REP-based integration can be report-layout-sensitive | Yes | Yes | Medium | General implementation risk; not explicitly vendor-documented in sources. |

---

## 12. Version Differences

| Area | Axys | APX | Confidence |
|---|---|---|---|
| Minimum version cited by Advent Connector | Axys 3.8.6 | APX 15.2 / 16.1 / 16.2 / 17.1 | Verified for connector only |
| Recent Axys release information | Axys 3.8.7 enhanced Position Reconciliation, generic date framework, multicurrency reports | Not applicable | Verified |
| REP language keyword count | Axys roughly 100 keywords according to consultant source | Current APX adds another 100+ keywords according to consultant source | Medium Confidence |
| Report storage architecture | Proprietary database / report files in Axys environment; exact storage not fully documented here | SQL Server-based APX platform plus reporting options; exact REP/SSRS storage boundary unknown | Medium for broad distinction, Unknown for implementation details |
| Cloud delivery | Axys generally positioned as turnkey/light IT footprint; exact cloud/client tool model unknown | APX vendor says local or cloud-delivered | Verified for APX delivery option; Unknown for REP implications |

---

## 13. Example: Axys `AMAN.REP` Customization Pattern

This example is supported by the CSSI PDF and should be treated as an **example**, not a general RepLang tutorial.

### Goal

Add portfolio code to the Axys Assets Under Management report.

### Evidence-supported steps

| Step | Action | Confidence |
|---|---|---|
| 1 | Open Axys Reports / Mgmt / Assets Under Management to identify the standard report and see its file path in the lower-left corner. | Verified |
| 2 | Identify the report file as `AMAN.REP`. | Verified for the example |
| 3 | Copy `AMAN.REP` to a custom filename such as `AMAN_XX.REP`. | Verified |
| 4 | Run the copied report via Axys Reports / Custom / Any Report, entering `AMAN_XX`. | Verified |
| 5 | Open `AMAN_XX.REP` in a text editor such as Notepad. | Verified |
| 6 | Find the line containing `.#~8portmv\n`. | Verified for sample |
| 7 | Change it to `.#~8portmv $:fileo\n` to add portfolio code after market value. | Verified for sample |
| 8 | Adjust a width/spacing line from `#width #cnt 16* 25+ 16+` to `#width #cnt 16* 35+ 16+` to create more room. | Verified for sample |
| 9 | Save and run the copied report. | Verified |

### Supported code fragment

```replang
.#~8portmv\n
.#~8portmv $:fileo\n
#width #cnt 16* 25+ 16+
#width #cnt 16* 35+ 16+
```

### Important limitations

- This example verifies only the specific report and expressions shown.
- It does not establish a complete RepLang grammar.
- It does not prove that `$:fileo` is available in all reports, all contexts, all Axys versions, or APX.
- It does not prove equivalent behavior for APX.

---

## 14. REP as a Data Extraction Interface

### 14.1 Evidence-Supported Observations

| Statement | Classification | Notes |
|---|---|---|
| REP reports can be used to generate data extracts in at least one integration product. | Verified for connector | Salentica/Black Diamond connector uses standard reports/macros and uploads extracted data. |
| A connector can require an always-on Windows workstation/server with Advent Client Tools installed. | Verified for connector | The connector is a 32-bit Windows application and should be hosted on a powered-on machine. |
| REP extraction may rely on existing credentials and mapping information downstream. | Verified for connector | Data Broker uses credentials/mapping to load data into CRM. |
| REP extraction can coexist with IMEX/API/database approaches. | Medium Confidence | AdventGuru describes blended approaches for data gathering/sharing. |

### 14.2 Recommended Treatment in Repository

For repository documentation, REP extraction should be documented as:

- **Report-driven**, not object-export-driven.
- Potentially useful where a firm already relies on Advent standard reports.
- Potentially useful for calculated/report-only values.
- Potentially brittle unless report file, parameters, layout, output format, and version are controlled.
- Distinct from IMEX, APX database/public view access, REST API, and SSRS.

These are implementation guidance notes, not vendor guarantees.

---

## 15. Open Questions / Unknowns

The following are important unknowns for the eventual `13-REP.md` chapter.

| Unknown | Why It Matters | Needed Evidence |
|---|---|---|
| Full RepLang Programmer's Guide contents | Required for reliable syntax, variables, functions, control flow, date handling, totals, report parameters | Advent RepLang Programmer's Guide |
| Full list of Axys standard `.REP` files | Needed for report catalog and report-to-file mapping | Axys `rep` directory listing from production/test install |
| Full list of APX Replang reports | Needed to separate APX REP behavior from APX SSRS/report package behavior | APX report directory/export/sample environment |
| Axys report parameter syntax | Needed for automation and reproducible execution | `.REP` files, macros, report docs |
| REP32.exe command-line options | Needed for unattended report execution | Advent technical docs or working examples |
| Macro file syntax and examples | Needed for automation documentation | Sample macros and Advent docs |
| Report output formats | Needed for using REP as extract interface | Standard/custom report examples in text/CSV/PDF/fixed-width formats |
| Which reports use stored performance vs recalculation | Critical for performance audit chapters | Report documentation plus controlled test cases |
| APX relationship among REP, SSRS, Report Writer Pro, report packaging | Needed to describe APX accurately | APX Reports Guide, APX admin/report packaging docs |
| APX public views/stored accounting functions used by standard reports | Needed for developers building alternatives to REP output parsing | APX schema docs/public view docs |
| Version-specific Replang keyword differences | Needed to document Axys/APX and version differences | Keyword lists by version |
| Vendor support boundaries for custom reports in current SS&C policy | Needed for implementation risk | Current SS&C support policy / client community docs |

---

## 16. Candidate Materials to Request from User

The current public evidence is enough to create a conservative research file, but not enough to write a comprehensive technical manual chapter with full field dictionaries and processing semantics.

To improve `13-REP.md`, request any of the following:

1. Axys `rep` folder listing, including standard `.REP` filenames.
2. Sample Axys `.REP` files such as `AMAN.REP`, performance reports, holdings reports, and transaction reports.
3. Sample Report Writer Pro report definitions or generated `.REP` source.
4. Axys macro files used to run reports unattended.
5. REP32.exe command-line examples.
6. RepLang Programmer's Guide.
7. APX Reports Guide / APX Reporting Guide.
8. APX SSRS report catalog and sample `.rdl` files.
9. APX Public Views / Stored Accounting Functions documentation.
10. Sample REP output files used as data extracts.
11. Known production examples where report output differs from IMEX output.
12. Versioned report files from Axys 3.8.6 vs 3.8.7 or APX 15.x/16.x/17.x.

---

## 17. Chapter Outline Suggested by Research

Chapter 13 can use this structure:

1. Overview
2. REP, RepLang, Report Writer Pro, and REP32 Terminology
3. Axys REP Architecture
4. APX REP and Reporting Architecture
5. REP vs IMEX vs APX SQL/SSRS/API Reporting
6. Report File Locations and Naming
7. Report Execution and Automation
8. Report Source Editing Guidelines
9. Field / Expression Dictionary
10. Known Standard Reports and File Names
11. Examples
12. Version Differences
13. Known Issues / Quirks
14. References
15. Unknowns

---

## 18. Fact Table for Direct Transfer to Chapter

| Statement | Axys | APX | Confidence | Source |
|---|---|---|---|---|
| Axys has an extensive library of predefined reports and supports easy report customization. | Yes | N/A | Verified | SRC-002 |
| Axys reports are written in RepLang, Advent's proprietary Report Writing Language. | Yes | Unknown | Verified | SRC-005 |
| Axys Report Writer Pro can be used to create custom reports. | Yes | Unknown | Verified | SRC-002 |
| `AMAN.REP` is the Assets Under Management report in the documented Axys example. | Yes | Unknown | Verified for example | SRC-005 |
| Standard reports should be copied before editing to avoid modifying originals. | Yes | Likely | Verified for Axys | SRC-005 |
| RepLang source should be edited in a text editor, not a word processor. | Yes | Likely | Verified for Axys | SRC-005 |
| `.#~8portmv\n` prints portfolio market value followed by a new line in the sample report. | Yes | Unknown | Verified for sample | SRC-005 |
| `$:fileo` displays portfolio code in the sample report. | Yes | Unknown | Verified for sample | SRC-005 |
| APX has a vast library of standard reports and automated report packaging. | N/A | Yes | Verified | SRC-003 |
| APX supports flexible custom reporting. | N/A | Yes | Verified | SRC-003 |
| APX has performance analytics and composite management support for GIPS compliance. | N/A | Yes | Verified | SRC-003 |
| APX users can create reports using Report Writer Pro or Replang source edits. | N/A | Yes | Medium Confidence | SRC-007 |
| Replang remains part of Axys and APX reporting architecture. | Yes | Yes | Medium Confidence | SRC-007 |
| APX users have SQL Server/database reporting options including Stored Accounting Functions, Public Views, SSRS, and REST API. | N/A | Yes | Medium Confidence | SRC-007/SRC-008 |
| Advent Connector for Axys/APX uses Advent standard reports and macros to generate data extracts. | Yes | Yes | Verified for connector | SRC-006 |
| Advent Connector requires REP32.exe / Advent Client Tools and uses REP32 engine to extract data. | Yes | Yes | Verified for connector | SRC-006 |
| Advent Connector tested minimum versions include Axys 3.8.6 and APX 15.2/16.1/16.2/17.1. | Yes | Yes | Verified for connector | SRC-006 |
| Axys 3.8.7 added an enhanced Position Reconciliation report. | Yes | N/A | Verified | SRC-004 |
| Axys 3.8.7 expanded the generic date framework for reporting/automation flexibility. | Yes | N/A | Verified | SRC-004 |
| Axys 3.8.7 added/improved multicurrency reports. | Yes | N/A | Verified | SRC-004 |

---

## 19. References

1. SS&C Advent. “Axys.” https://www.advent.com/solutions/axys/
2. SS&C Advent. “Advent Portfolio Exchange.” https://www.advent.com/solutions/advent-portfolio-exchange/
3. SS&C Advent. “A New Version of Axys – Just in Time for Upgrade Season.” https://www.advent.com/news-and-insights/blog/a-new-version-of-axys-just-in-time-for-upgrade-season/
4. Client Server Specialists / CSSI. “How to Add Portfolio Code to Axys Reports.” https://assets.ctfassets.net/xhy36q2d1lqu/77QC4aNbyhPo9FfmjRYNzc/d00a0d6601214601543e30e34f203626/PortfolioCodetoAxys.pdf
5. Salentica / Black Diamond. “Data Broker - SS&C|Advent APX & Axys.” https://engage.salentica.com/kb/article/247-data-broker-ss-c-advent-apx-axys/
6. AdventGuru. “Using Visual Studio Code to Modify Advent Replang Reports in Axys and APX.” https://adventguru.com/2024/09/09/using-visual-studio-code-to-modify-advent-replang-reports-in-axys-and-apx/
7. Microsoft. “SQL Server Reporting Services.” https://learn.microsoft.com/en-us/sql/reporting-services/create-deploy-and-manage-mobile-and-paginated-reports

---

## 20. Research Completeness Assessment

| Area | Completeness | Notes |
|---|---|---|
| Axys high-level reporting | Good | Vendor and consultant evidence sufficient for broad statements. |
| Axys `.REP` mechanics | Partial | One concrete example exists; full RepLang grammar is missing. |
| Axys report file dictionary | Low | Only `AMAN.REP` is supported by current evidence. |
| Axys automation / REP32 | Partial | Connector evidence confirms use, but command syntax is missing. |
| APX high-level reporting | Good | Vendor evidence supports high-level capabilities. |
| APX Replang/Report Writer Pro | Partial | Consultant evidence supports existence, but vendor docs needed. |
| APX SSRS/custom reporting | Partial-Good | Vendor high-level and consultant/Advent CDN search evidence exists; full APX Reports Guide needed. |
| Field dictionary | Low | Only sample expressions from the CSSI Axys example are supported. |
| Processing behavior | Low-Partial | Some specific execution/editing behaviors verified; calculation internals unknown. |
| Version differences | Low-Partial | Some Axys 3.8.7 and connector-supported versions known; report language changes unknown. |

## 21. Deep IMEX Addendum Incorporated 2026-06-30

Source: `axys_imex_deep_research.md`.

Additional REP/IMEX boundary points:

| Topic | Addendum | Confidence |
|---|---|---:|
| REP as alternative extraction | Deep IMEX research reinforces that REP/Replang/custom reports are an alternative extraction path when IMEX object schemas are incomplete or when values must tie to user-visible reports. | High Confidence |
| Connector evidence | Salentica Data Broker uses Advent standard reports, macros, REP32, RepLang scripting, and installed Advent Client Tools for Axys/APX extraction. | Verified for connector |
| Best-fit split | IMEX is the better starting point for security reference, security types, prices, transactions, and positions where object/export support exists; REP is often better for performance values, classification performance, and report-specific tie-outs. | Design guidance |
| Required metadata | REP-derived extracts should record report file/name, version, parameters, layout, row lineage, and whether values are stored or recalculated. | Design guidance |
| Boundary | A report label or REP variable does not by itself establish underlying transaction, performance, or IMEX field semantics. | Medium / boundary |

## Deep Research Update Incorporated 2026-07-02

The July 2026 addendum strengthens current practitioner evidence that Replang
remains relevant for both Axys and APX. AdventGuru evidence says Axys/APX users
can still create reports with Report Writer Pro or direct Replang source edits;
Axys Replang has roughly 100 keywords and current APX adds 100+ more. The
exact keyword lists, compatibility rules, and report catalogs remain Unknown.

APX reporting should remain explicitly multi-path: Replang/compound reports,
SQL Server, Stored Accounting Functions, Public Views, SSRS, REST API,
dashboards, and report packaging can all be relevant depending on deployment
and support boundaries. CSSI and other practitioner evidence reinforces that
APX SSRS/report-package operations can be complex and operationally fragile,
but this does not expose RDL datasets, stored procedures, or schema.

REP32 automation evidence adds environment-specific Axys failure modes:
`rep32.exe` can fail under elevation/permission constraints, scripted PDF
output can fail if PrimoPDF is missing, and Axys 3.8.7.7 support notes mention
PrimoAPI as a default PDF-printer lead. AdventGuru also provides a
performance-history report lead involving Net of Fees `(PRF)` and Gross of
Fees `(PBF)` data; report filename, storage mechanics, and calculation behavior
remain Unknown.
