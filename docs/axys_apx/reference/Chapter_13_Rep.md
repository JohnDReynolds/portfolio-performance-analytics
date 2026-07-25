# Chapter 13 — REP

Repository: AXYS / APX Reference Repository
Chapter: `docs/axys_apx/reference/Chapter_13_Rep.md`
Prepared: 2026-06-29
Status: Technical reference chapter based on repository research and cited public evidence.
Public evidence reviewed: 2026-07-17

---

## Related chapters

- [Chapter_01_Overview.md](Chapter_01_Overview.md) — provides the repository-wide map, evidence conventions, and shared safe implementation rules.
- [Chapter_02_Axys_Architecture.md](Chapter_02_Axys_Architecture.md) — connects REP to the broader Axys architecture narrative.
- [Chapter_12_Imex.md](Chapter_12_Imex.md) — distinguishes REP/report extraction from IMEX import/export.
- [Chapter_14_Reports.md](Chapter_14_Reports.md) — ties REP and report source content to the report families.

## 1. Overview

REP is the Advent reporting layer associated with report files, report source, report execution, report customization, and report-driven data extraction in Axys/APX environments.

This chapter documents REP as a technical interface. It does **not** attempt to catalog every Axys or APX standard report. Standard report content belongs primarily in `Chapter_14_Reports.md`. This chapter focuses on:

- `.REP` report files
- RepLang
- Report Writer Pro
- REP32 / report-engine execution evidence
- macros and report-driven extraction
- Axys report customization behavior
- APX reporting paths that overlap with or differ from REP
- the boundary between REP, IMEX, APX SQL/Public Views/SSRS/REST, and direct file access

All unsupported behavior is marked **Unknown**.

REP should be treated as a reporting and extraction interface rather than a source of canonical transaction semantics. A report can expose holdings, cash, performance, or classification outputs derived from underlying transactions, but the report label or REP variable does not by itself establish the underlying transaction classification. In practice, report extraction should be paired with transaction-level evidence when interpreting flows or audit outcomes.

### 1.1 Confidence

This chapter uses the repository confidence labels defined in
[Chapter 01](Chapter_01_Overview.md#2-confidence-and-evidence-boundary).
Confidence remains scoped to the cited product, version, report, connector,
or workflow.

### 1.2 Source discipline

This chapter follows the repository blueprint:

- facts first
- separate Axys/APX when behavior differs
- prefer evidence over assumptions
- preserve Unknowns
- do not invent field names, report behavior, command syntax, transaction codes, report internals, or implementation details

---

## 2. REP, RepLang, Report Writer Pro, and REP32 Terminology

| Term | Description | Axys | APX | Confidence |
|---|---|---:|---:|---|
| REP | Common shorthand for Advent report source/reporting artifacts. | Yes | Yes, where Replang/REP reports are used | High Confidence |
| `.REP` file | RepLang report source. Examples include `AMAN.REP` and historical `PERHSUM.REP`. | Verified | Possible in report/integration contexts; native coverage Unknown | Verified for Axys examples |
| `.RPW` file | Report Writer Pro report type whose underlying format is still RepLang in historical guidance. | Verified historically | Practitioner evidence | Version-sensitive |
| RepLang | Advent proprietary Report Writing Language. | Verified | Medium Confidence | Verified for Axys; Medium Confidence for APX |
| Report Writer Pro | Advent reporting tool for creating/customizing reports. | Verified | Medium Confidence | Verified for Axys; Medium Confidence for APX |
| REP32.exe | Advent client-side report engine/application referenced by connector documentation. | Verified for connector | Verified for connector | Verified for connector |
| Standard report | Vendor-supplied report available from report menus/report library. | Verified | Verified | Verified |
| Custom / Any Report | Axys menu path used to run a copied/custom report by report name. | Verified | Unknown | Verified for Axys |
| Macro | Automation artifact used with standard reports by an Advent connector. | Verified for connector | Verified for connector | Verified for connector |
| SSRS | Microsoft SQL Server Reporting Services; APX reporting path in supplied evidence. | Not established for Axys | Medium Confidence | Medium Confidence for APX |

---

## 3. REP vs IMEX vs APX SQL / SSRS / REST

REP is not IMEX. REP is report-driven; IMEX is object/file import/export-oriented.

| Mechanism | Primary role | Axys | APX | Confidence |
|---|---|---:|---:|---|
| REP / RepLang | Generate report output; can also be used for report-driven extraction. | Yes | Yes, where RepLang remains available | High Confidence for Axys; Medium Confidence for APX |
| Report Writer Pro | Build or customize reports. | Yes | Medium Confidence | Verified for Axys; Medium Confidence for APX |
| REP32.exe | Execute/extract through standard reports/macros in connector workflows. | Yes | Yes | Verified for connector |
| IMEX | Import/export structured data files/objects. | Yes | Yes, but exact APX behavior is incomplete in supplied evidence | Verified for Axys CI context; Medium Confidence for APX |
| Direct Axys file access | Read/write Axys data files directly. | Possible but risky | Not applicable in same way | Medium Confidence |
| APX SQL / Public Views / Stored Accounting Functions | SQL/reporting/data-access alternatives. | No | Yes | Medium Confidence |
| APX SSRS | SQL Server Reporting Services reporting path. | No | Yes | Medium Confidence |
| APX REST API | Officially documented, expanding API path in APX 21.1+ releases. | No | Yes | Verified at release-capability level |

### 3.1 Implementation distinction

| Interface | Best used when | Caution |
|---|---|---|
| REP | The required output is already report-shaped, or a calculated value is available through a standard/custom report. | Report layout, report parameters, and report version must be controlled. |
| IMEX | A stable structured import/export object exists for the desired data. | Native object names and full field dictionaries remain Unknown for many objects in supplied evidence. |
| APX SQL/Public Views/Stored Accounting Functions | APX installation exposes supported database/reporting views or functions. | Exact view names, function names, support boundaries, and security-master/performance coverage are Unknown. |
| APX SSRS | APX report output is implemented as SSRS reports. | Exact report definitions, datasets, stored procedures, and deployment details are Unknown. |
| Direct Axys files | Only when no safer interface exists and file format/version risk is controlled. | Consultant evidence warns Axys file formats can change across versions. |

---

## 4. Axys REP

### 4.1 Axys reporting capabilities

| Statement | Confidence |
|---|---|
| Axys automates portfolio reporting and accounting. | Verified |
| Axys includes an extensive library of predefined reports. | Verified |
| Axys supports report customization. | Verified |
| Axys supports Axys Report Writer Pro. | Verified |
| Axys can place multiple reports, graphs, and objects on one page for repeated automated use across portfolio groups. | Verified |
| Axys can output graphics such as pie charts, line graphs, and bar charts. | Verified |
| Axys can display performance by portfolios, asset classes, sectors, countries, or regions. | Verified |
| Axys includes reporting related to composites and GIPS performance measurement standards. | Verified |

### 4.2 Axys `.REP` files and RepLang

| Statement | Confidence | Notes |
|---|---|---|
| Axys reports are written in RepLang. | Verified | Supported by the supplied CSSI technical PDF. |
| RepLang is Advent's proprietary report-writing language. | Verified | Supported by the supplied CSSI technical PDF. |
| Axys reports can be opened and edited in a plain text editor. | Verified | CSSI example uses Notepad or another text editor. |
| Word processors should not be used to edit RepLang files. | Verified | CSSI warns not to use Word or WordPerfect. |
| The Axys Reports window displays the report path and file name in the lower-left corner when a report dialog is displayed. | Verified | CSSI example. |
| `AMAN.REP` is the Assets Under Management report in the documented Axys example. | Verified for example | Do not infer all installations use the same path. |
| Example report directory `e:\axys34\rep` is a sample path only. | Verified for example | Not a universal installation path. |
| A standard `.REP` file can be copied, renamed, edited, saved, and run as a custom report. | Verified | CSSI `AMAN.REP` example. |
| A copied custom report can be run from Axys Reports / Custom / Any Report by entering the report name. | Verified | CSSI example. |
| A period (`.`) tells Axys to print in the CSSI RepLang example. | Verified for example | Full RepLang print semantics remain Unknown. |
| `\n` signals carriage return / end of printed line in the CSSI example. | Verified for example | Full RepLang formatting grammar remains Unknown. |
| `.#~8portmv\n` prints portfolio market value followed by a line break in the CSSI example. | Verified for example | Do not generalize beyond supported context. |
| `$:fileo` displays portfolio code in the CSSI example. | Verified for example | Availability in all reports/versions/APX is Unknown. |

### 4.3 Axys REP execution and automation

| Statement | Confidence | Notes |
|---|---|---|
| A third-party Advent connector can use Advent standard reports and macros to generate a data extract from Axys. | Verified for connector | Salentica / Black Diamond connector evidence. |
| The connector requires Advent Client Tools, specifically `REP32.exe`, on the machine where the connector runs. | Verified for connector | Connector-specific evidence. |
| The connector uses the REP32 engine to extract data. | Verified for connector | Connector-specific evidence. |
| The connector uses RepLang scripting and macros. | Verified for connector | Connector-specific evidence. |
| The connector can be scheduled to run unattended or at predefined daily frequencies. | Verified for connector | Connector-specific evidence. |
| A working REP32 command pattern is known from a dated Axys example. | Verified for example | Full grammar and complete option set remain Unknown. |
| Macro file syntax is known. | Unknown | Not supplied. |
| Exact unattended scheduling syntax is known. | Unknown | Not supplied. |

Observed public pattern, reproduced as operational evidence rather than a complete
command specification:

```text
Rep32.exe -m macroname -p portcode "-b date1 date2"
```

The same source demonstrates switches including `-J`, `-x`, `-su`, and `-z`; their
complete semantics and version coverage still require vendor documentation or tests.

### 4.4 Axys report file handling pattern

Recommended documented pattern from the supplied Axys example:

| Step | Action | Confidence |
|---:|---|---|
| 1 | Select the standard report in Axys and note its file path/name in the Reports window. | Verified |
| 2 | Copy the standard `.REP` file before editing. | Verified |
| 3 | Rename the copy using a distinct custom report name. | Verified |
| 4 | Run the copied report through Custom / Any Report before modifying it. | Verified |
| 5 | Edit the copied `.REP` file with a plain text editor. | Verified |
| 6 | Save and rerun the copied report. | Verified |
| 7 | Preserve the original vendor-supplied `.REP` file. | Verified for Axys example / High Confidence as practice |

### 4.5 Axys version notes

| Version / Area | Statement | Confidence |
|---|---|---|
| Axys 3.8.6 | Listed as a tested/supported minimum version for one Advent Data Broker connector. | Verified for connector only |
| Axys 3.8.7 | Added an enhanced Position Reconciliation report. | Verified |
| Axys 3.8.7 | Expanded the generic date framework for reporting/automation flexibility. | Verified |
| Axys 3.8.7 | Added or improved multicurrency reports. | Verified |
| Axys 3.8.7.7 | PrimoAPI PDF printer is a lead for scripted PDF output workflows. | Medium Confidence |
| Axys 3.7 to 3.8 | Consultant evidence says file conversion was required and some file formats changed. | Medium Confidence |
| Axys RepLang keyword changes by version | Unknown. | Unknown |
| Report differences between Axys 3.8.6 and 3.8.7 beyond vendor bullets | Unknown. | Unknown |

---

## 5. APX REP and Reporting

### 5.1 APX reporting capabilities

| Statement | Confidence |
|---|---|
| APX is an integrated portfolio and client management solution. | Verified |
| APX includes a vast library of standard reports. | Verified |
| APX supports automated report packaging. | Verified |
| APX supports flexible custom reporting. | Verified |
| APX supports customizable dashboards. | Verified |
| APX includes performance analytics. | Verified |
| APX supports composite management for GIPS compliance. | Verified |
| APX can be deployed locally or cloud-delivered with or without outsourcing services. | Verified |

### 5.2 APX RepLang / Report Writer Pro

| Statement | Confidence | Notes |
|---|---|---|
| APX users can create reports using Report Writer Pro or by updating Replang source directly. | Medium Confidence | Consultant evidence only. |
| Replang remains part of the reporting architecture of both Axys/APX. | Medium Confidence | Consultant evidence only. |
| APX has more Replang keywords than Axys in current versions. | Medium Confidence | Consultant source states current APX adds 100+ keywords beyond Axys; exact list not supplied. |
| Exact APX Replang keyword set is known. | Unknown | Not supplied. |
| Exact APX `.REP` file locations are known. | Unknown | Not supplied. |
| Every Axys `.REP` report can run unchanged in APX. | Unknown | No supporting evidence. |
| APX Report Writer Pro storage format is known for all versions/environments. | Unknown | Not supplied. |

### 5.3 APX SQL / SSRS reporting path

| Statement | Confidence | Notes |
|---|---|---|
| APX supports custom reporting beyond REP/Replang. | Verified | Vendor product capability evidence. |
| APX has dashboards and flexible custom reporting. | Verified | Vendor product capability evidence. |
| APX users can use SQL Server-based reporting/data access options, including SSRS, according to consultant evidence. | Medium Confidence | Consultant evidence. |
| APX data access/reporting options include Stored Accounting Functions, Public Views, SSRS, REST API, and related tools. | Mixed: REST capability Verified; other access paths Medium Confidence | Exact support boundaries remain deployment-specific. |
| APX reporting is multi-path: Replang/compound reports, SQL Server, Stored Accounting Functions, Public Views, SSRS, REST API, dashboards, and report packaging may be relevant depending on deployment. | High Confidence for categories | Exact source dependencies vary by report/environment and remain Unknown. |
| Microsoft SSRS is an on-premises platform for creating, deploying, and managing paginated reports. | Verified for SSRS generally | Not APX-specific implementation evidence. |
| The public APX Reports Guide identifies 29 SSRS-based investment-management reports and describes customization/branding. | Verified | Installed/current inventory and RDL internals remain Unknown. |
| Exact APX SSRS report dataset names are known. | Unknown | Not supplied. |
| Exact APX report stored procedures/public views are known. | Unknown | Not supplied. |
| Exact APX report-server paths/deployment process are known. | Unknown | Not supplied. |

### 5.4 APX REP execution and automation

| Statement | Confidence | Notes |
|---|---|---|
| A third-party Advent connector can use Advent standard reports and macros to generate a data extract from APX. | Verified for connector | Connector-specific evidence. |
| The connector requires Advent Client Tools / `REP32.exe` for APX extraction. | Verified for connector | Connector-specific evidence. |
| The connector uses the REP32 engine, RepLang scripting, and macros for APX extraction. | Verified for connector | Connector-specific evidence. |
| Connector-tested APX versions include 15.2, 16.1, 16.2, and 17.1. | Verified for connector only | Not a general APX support matrix. |
| APX always requires REP32.exe for all REP/Replang reports. | Unknown | Evidence covers one connector. |
| APX cloud-delivered environments permit direct custom REP execution by clients. | Unknown | Requires vendor/admin documentation. |

---

## 6. REP Artifacts and Named Examples

| Report name | File / tool name | System | Description | Confidence |
|---|---|---:|---|---|
| Assets Under Management | `AMAN.REP` | Axys | Standard Axys report in CSSI example; displays market values of portfolios in selected group as of selected date, sorted by asset class in the example. | Verified for example |
| Performance History for Selected Time Periods | `PERHSUM.REP` | Axys 3.6 | Historical standard report identified in public modification guidance. | Verified historical evidence |
| Position Reconciliation report | Unknown file name | Axys | Enhanced in Axys 3.8.7 with improved handling of pending/unsettled trades and management insight. | Verified as named report; file name Unknown |
| Multicurrency reports | Unknown | Axys | Axys 3.8.7 included additional/improved multicurrency reports. | Verified as vendor statement; individual names Unknown |
| Transaction Summary Report | Unknown REP/SSRS file name | APX / Advent reports | Report sample evidence shows transaction sections and fields. | Medium Confidence |
| `CDIhold.rep` | `CDIhold.rep` | Axys | Custom WealthTechs AIA holdings extract report. | Verified for AIA workflow |
| `sipos30` | Unknown extension | Axys | Custom reconciliation report cited by CI for calculated positions vs downloaded custodian positions. | Verified for CI context |

### 6.1 Artifact caveat

`AMAN.REP` and historical `PERHSUM.REP` are verified Axys filenames in the
reviewed public evidence. The complete APX report-name inventory is maintained
in [Chapter 14](Chapter_14_Reports.md). Other names or tools here are either:

- named reports with unknown file names,
- connector/custom report artifacts,
- report samples whose REP/SSRS implementation is Unknown,
- or vendor/product capability references.

Do not infer file names for standard reports that are not supplied.

---

## 7. Field / Expression Dictionary

This table includes only field names, expressions, executable names, and labels directly supported by supplied material. It is not a complete RepLang dictionary.

| Field / Expression / Name | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---|
| `#~8portmv` | Prints portfolio market value in the CSSI `AMAN.REP` example. | Yes | Unknown | No evidence | Yes | Verified for Axys sample |
| `$:fileo` | Displays portfolio code in the CSSI `AMAN.REP` example. | Yes | Unknown | No evidence | Yes | Verified for Axys sample |
| `\n` | Carriage return / end-of-line marker in the CSSI example. | Yes | Unknown | No evidence | Yes | Verified for Axys sample |
| `.` prefix | Print command marker according to CSSI example. | Yes | Unknown | No evidence | Yes | Verified for Axys sample |
| `#width` | Appears in sample report line controlling page width/spacing; full semantics Unknown. | Yes | Unknown | No evidence | Yes | Medium Confidence |
| `#cnt` | Appears in sample report line; full semantics Unknown. | Yes | Unknown | No evidence | Yes | Medium Confidence |
| `AMAN.REP` | Assets Under Management report source file in sample Axys environment. | Yes | Unknown | No evidence | Yes | Verified for Axys sample |
| `REP32.exe` | Advent reporting application/engine used by an Advent connector. | Yes | Yes | No evidence | Yes | Verified for connector |
| `.RPW` | Report Writer Pro-generated report extension according to practitioner source. | Yes | Yes | No evidence | Related | Medium Confidence |
| `.REP` | Manually coded / report source file extension according to supplied material. | Yes | Medium Confidence | No evidence | Yes | Verified for Axys; Medium for APX |
| `CDIhold.rep` | Custom AIA holdings extract report. | Yes | Unknown | No evidence | Yes | Verified for AIA |
| `sipos30` | Reconciliation report cited by CI. | Yes | Unknown | No evidence | Yes | Verified for CI context |

### 7.1 Unsupported field dictionary items

The following remain **Unknown**:

| Unknown | Needed evidence |
|---|---|
| Full RepLang keyword list. | RepLang Programmer's Guide or vendor documentation. |
| Full RepLang grammar. | RepLang Programmer's Guide. |
| Full Axys report variable dictionary. | Vendor docs or report source library. |
| Full APX Replang variable/keyword dictionary. | APX Replang documentation or report source library. |
| Complete REP32 command-line parameter set and version differences. | Vendor technical documentation; one public working pattern is known. |
| Macro syntax. | Sample macros or vendor documentation. |
| Report output field list for each standard report. | Report guide, `.REP` source, or generated report examples. |
| Whether `$:fileo` is valid in all reports. | Production report tests or RepLang documentation. |
| Whether `#~8portmv` has identical behavior across versions. | RepLang documentation or versioned report examples. |

---

## 8. Processing Behavior

### 8.1 Axys processing behavior

| Behavior | Statement | Confidence |
|---|---|---|
| Report source execution | Axys reads `.REP` report source files to produce report output. | Verified for sample |
| Print marker | A period (`.`) tells Axys to print in the CSSI example. | Verified for sample |
| Line break | `\n` ends a printed line / carriage return in the CSSI example. | Verified for sample |
| Report path display | Axys Reports window displays report path/file name in the lower-left corner. | Verified |
| Custom report execution | A copied `.REP` report can be run from Custom / Any Report by name. | Verified |
| Report layout modification | Width/spacing values may need adjustment when adding output fields. | Verified for sample |
| REP32 automation | REP32 can be used by a connector to extract data through standard reports/macros. | Verified for connector |
| Report calculation timing | Whether Axys REP recalculates values at run time or reads stored values depends on the specific report; exact behavior is Unknown. | Unknown |
| Report parameter handling | Exact syntax for dates, portfolios/groups, reporting currency, consolidation, and report options is Unknown. | Unknown |
| Report output format control | Exact text/CSV/fixed/PDF output behavior by report is Unknown. | Unknown |

### 8.2 APX processing behavior

| Behavior | Statement | Confidence |
|---|---|---|
| Standard reports/macros | APX can be accessed by a connector using Advent standard reports and macros. | Verified for connector |
| REP32 extraction | A connector uses REP32 engine for APX extraction. | Verified for connector |
| Replang direct edits | APX reports can be created/modified with direct Replang source edits according to consultant evidence. | Medium Confidence |
| Report Writer Pro | APX users can use Report Writer Pro according to consultant evidence. | Medium Confidence |
| SSRS reports | APX supports or commonly uses SSRS-based reporting according to consultant/vendor-adjacent evidence. | Medium Confidence |
| Calculation source | Whether APX REP reports calculate directly from database tables, stored accounting functions, public views, stored performance data, or report-specific procedures is Unknown. | Unknown |
| Report packaging internals | Exact relationship among REP, SSRS, dashboards, and automated report packaging is Unknown. | Unknown |

---

## 9. REP as a Data Extraction Interface

### 9.1 Evidence-supported extraction use

| Statement | Confidence |
|---|---|
| REP reports can be used to generate data extracts in at least one third-party connector. | Verified for connector |
| An Advent connector can use standard reports and macros to generate extracts for Axys/APX. | Verified for connector |
| The connector can require a powered-on Windows workstation/server with Advent Client Tools installed. | Verified for connector |
| REP extraction can be scheduled in that connector context. | Verified for connector |
| REP extraction can coexist with IMEX, database, API, or SSRS approaches. | Medium Confidence |

### 9.2 Practical use cases

| Use case | REP suitability | Confidence |
|---|---|---|
| Extract report-ready values already calculated by an Advent report. | Good candidate. | Medium Confidence |
| Extract formatted client-report output. | Good candidate. | High Confidence |
| Extract structured object data where stable IMEX object exists. | IMEX may be preferable. | Medium Confidence |
| Extract APX data exposed by Public Views or Stored Accounting Functions. | SQL/view path may be preferable. | Medium Confidence |
| Automate repeated standard report output. | Supported in connector evidence. | Verified for connector |
| Build robust machine interface from presentation reports. | Risky unless output layout and version are controlled. | Medium Confidence |

### 9.3 REP extraction risks

| Risk | Description | Confidence |
|---|---|---|
| Layout sensitivity | If parsing report output, changes to columns, headings, spacing, sections, or totals can break downstream processing. | Medium Confidence |
| Parameter sensitivity | Different date/portfolio/report options can change output shape and values. | Medium Confidence |
| Version sensitivity | Report source and output may differ across versions or customizations. | Medium Confidence |
| Hidden calculation source | A report may calculate values, read stored values, use database functions, or combine sources; exact behavior is report-specific and often Unknown. | High Confidence as a caution; exact behavior Unknown |
| Custom support limits | Consultant evidence says vendor support may not debug custom RepLang reports. | Medium-High |
| Client-tool dependency | Connector evidence requires Advent Client Tools / REP32 on a client machine. | Verified for connector |

### 9.4 REP/IMEX Extraction Boundary

The available IMEX evidence clarifies the REP/IMEX split for extraction
design.

| Topic | Chapter treatment | Confidence |
|---|---|---:|
| REP as fallback | REP/Replang/custom reports are appropriate when IMEX schemas are incomplete or values must match user-visible reports. | High Confidence |
| Connector evidence | Salentica Data Broker uses Advent standard reports, macros, REP32, RepLang scripting, and installed Advent Client Tools for Axys/APX extraction. | Verified for connector |
| Best-fit split | Use IMEX first for security master, security types, transactions, prices, and positions where stable objects exist; use REP for performance, classification performance, and report-specific tie-outs. | Design guidance |
| Required metadata | REP-derived extracts should record report file/name, version, parameters, layout, row lineage, and stored-vs-recalculated confidence. | Design guidance |

A report label or REP variable does not by itself establish underlying IMEX,
database, transaction, or performance semantics.

---

## 10. Examples

### 10.1 Axys `AMAN.REP` customization pattern

This example is supported by the supplied CSSI technical PDF. It is an example only.

#### Goal

Add portfolio code to the Axys Assets Under Management report.

#### Evidence-supported steps

| Step | Action | Confidence |
|---:|---|---|
| 1 | Open Axys Reports / Mgmt / Assets Under Management to identify the standard report and see its file path in the lower-left corner. | Verified |
| 2 | Identify the report file as `AMAN.REP` in the example. | Verified for example |
| 3 | Copy `AMAN.REP` to a custom filename such as `AMAN_XX.REP`. | Verified |
| 4 | Run the copied report via Axys Reports / Custom / Any Report, entering `AMAN_XX`. | Verified |
| 5 | Open `AMAN_XX.REP` in a text editor. | Verified |
| 6 | Find the line containing `.#~8portmv\n`. | Verified for sample |
| 7 | Change it to `.#~8portmv $:fileo\n` to add portfolio code after market value. | Verified for sample |
| 8 | Adjust width/spacing from `#width #cnt 16* 25+ 16+` to `#width #cnt 16* 35+ 16+` to create more room. | Verified for sample |
| 9 | Save and run the copied report. | Verified |

#### Supported code fragment

```replang
.#~8portmv\n
.#~8portmv $:fileo\n
#width #cnt 16* 25+ 16+
#width #cnt 16* 35+ 16+
```

#### Limitations

| Limitation | Confidence |
|---|---|
| This example verifies only the specific report and expressions shown. | Verified |
| It does not establish a complete RepLang grammar. | Verified caveat |
| It does not prove `$:fileo` is available in every report, context, Axys version, or APX. | Unknown |
| It does not prove equivalent behavior for APX. | Unknown |

### 10.2 Connector-style REP extraction model

The supplied connector evidence supports the following model for at least one third-party connector:

```text
Advent Axys/APX environment
        ↓
Advent Client Tools installed on Windows machine
        ↓
REP32.exe / REP32 engine
        ↓
Standard Advent reports + macros
        ↓
RepLang scripting/macros
        ↓
Generated data extract
        ↓
Connector uploads/processes extracted data
```

| Statement | Confidence |
|---|---|
| This model is verified for the cited connector. | Verified for connector |
| This model should not be assumed to define all REP automation. | Verified caveat |
| A working REP32 pattern is verified; the complete syntax remains Unknown. | Verified for the example; complete specification Unknown |

---

## 11. Known Issues / Quirks

| Quirk / Issue | Axys | APX | Confidence | Notes |
|---|---:|---:|---|---|
| Custom RepLang support may be limited. | Yes | Unknown | Medium-High | Consultant PDF says Advent publishes a RepLang Programmer's Guide on request but support will not answer/debug custom RepLang report questions. |
| Custom report development can be trial-and-error-oriented. | Yes | Unknown | Medium-High | Consultant PDF warns custom RepLang is best suited for users comfortable with trial and error. |
| Edit report copies, not originals. | Yes | Likely | Verified for Axys; Medium for APX | CSSI example copies `AMAN.REP` before editing. |
| Do not modify the mission-critical performance update report in place. | Yes | Unknown | High Confidence operational guidance | Preserve the vendor report and test any copy against known results. |
| Use plain text editors for RepLang. | Yes | Likely | Verified for Axys; Medium for APX | CSSI warns against Word/WordPerfect. |
| Line-number-aware editor is useful. | Yes | Yes | Medium Confidence | AdventGuru notes Notepad lacks line numbers; modern editors help. |
| Replang language support in modern editors may be limited. | Yes | Yes | Medium Confidence | AdventGuru notes Replang is not built into VS Code by default. |
| REP32 automation can fail for environment-specific reasons such as elevation/permission issues or missing PrimoPDF for scripted PDF output. | Yes | Maybe | Medium Confidence | Treat as operational evidence, not a universal REP32 rule. |
| Report Writer Pro generated reports may use `.RPW`; manually edited reports may use `.REP`. | Yes | Yes | Medium Confidence | Practitioner evidence. |
| Manual edits can interfere with future Report Writer Pro editing because of checksum behavior. | Yes | Yes | Medium Confidence | Practitioner evidence from IMEX research. |
| APX has more reporting paths than Axys. | N/A | Yes | Medium Confidence | APX SQL Server/Public Views/SSRS/REST options according to consultant evidence. |
| REP-based extraction can depend on Advent client tools and a Windows machine. | Yes | Yes | Verified for connector | Connector evidence. |
| REP output parsing can be brittle if report layout changes. | Yes | Yes | Medium Confidence | Implementation caution, not vendor guarantee. |
| Axys direct file access is risky across versions. | Yes | N/A | Medium Confidence | Relevant because REP/IMEX are safer alternatives when sufficient. |

---

## 12. Version Differences

| Area | Axys | APX | Confidence |
|---|---|---|---|
| Connector minimum version evidence | Axys 3.8.6 | APX 15.2 / 16.1 / 16.2 / 17.1 | Verified for connector only |
| Recent Axys release evidence | Axys 3.8.7 enhanced Position Reconciliation, expanded generic date framework, and added/improved multicurrency reports. | Not applicable | Verified |
| Replang keyword count | Consultant evidence says Axys has roughly 100 keywords. | Consultant evidence says current APX adds 100+ more. | Medium Confidence |
| Report storage architecture | Axys uses report files in the Axys environment; exact complete architecture Unknown. | APX has SQL Server/reporting options; exact REP/SSRS/report package boundary Unknown. | Medium for broad distinction; Unknown for internals |
| Cloud delivery | Exact Axys client-tool implications Unknown. | APX can be locally installed or cloud-delivered. | Verified for APX deployment; Unknown for REP implications |
| Fixed-format IMEX note | Axys supported fixed-format IMEX historically per consultant evidence. | Consultant evidence says APX v1.x-v4.x maintained IMEX but eliminated fixed-format generation. | Medium Confidence |
| Axys file format version risk | Consultant evidence says Axys 3.7 to 3.8 conversion changed some file formats. | Not established in same way. | Medium Confidence |

---

## 13. Cross-Chapter Integration Notes

### 13.1 Relationship to `Chapter_12_Imex.md`

| Topic | REP chapter treatment | IMEX chapter treatment |
|---|---|---|
| Structured import/export | Mention only as contrast. | Primary topic. |
| `REP32.exe` connector extraction | Primary topic. | Mention as report-based export alternative. |
| IMEX object names | Unknown here. | Unknown unless supported by IMEX evidence. |
| `topost.trn`, `sec.inf`, `type.inf`, `.pri`, `.cli` | Mention only where relevant to REP/IMEX boundary. | Primary IMEX/file-interface context. |

### 13.2 Relationship to `Chapter_14_Reports.md`

| Topic | REP chapter treatment | Reports chapter treatment |
|---|---|---|
| Report mechanism | Primary topic. | Supporting context. |
| Standard report catalog | Only report names identified in supplied evidence. | Primary topic. |
| Report output fields | Only sample-supported examples. | Full catalog if evidence exists. |
| Report purpose and interpretation | Limited. | Primary topic. |

### 13.3 Relationship to `Chapter_10_Performance.md`

| Topic | REP chapter treatment | Performance chapter treatment |
|---|---|---|
| Performance report availability | Axys/APX reporting capabilities noted. | Calculation/storage/recalculation behavior. |
| Stored vs recalculated values | Mark Unknown unless report-specific evidence exists. | Primary research topic. |
| Report values as audit evidence | REP can expose values; exact source Unknown. | Compare against performance data model. |

### 13.4 Relationship to `Chapter_05_Transactions.md`

| Topic | REP chapter treatment | Transactions chapter treatment |
|---|---|---|
| Transaction Summary Report | Mention as report evidence if supported. | Transaction report fields and transaction interpretation. |
| `didpost.aud` audit trail | Not a REP artifact; mention only as cross-chapter context if needed. | Transaction/audit trail artifact. |
| Transaction report fields | Only report-output evidence. | Transaction report/data model chapter. |

---

## 14. References

This chapter uses the repository research plus public evidence reviewed through
2026-07-17.

### 14.1 Governing specification

| Ref | Source |
|---|---|
| BP-001 | `axys_apx_reference_blueprint.md`, Version 2.0 |

### 14.2 REP evidence sources

| Ref | Source | Use in chapter |
|---|---|---|
| REP-001 | `../evidence/Research_13_REP.md` | Granular REP/RepLang, Report Writer Pro, and REP32 evidence ledger. |
| REP-002 | SS&C Advent Axys product page, as summarized in supplied research. | Axys reporting capability statements. |
| REP-003 | SS&C Advent APX product page, as summarized in supplied research. | APX reporting capability statements. |
| REP-004 | SS&C Advent Axys 3.8.7 blog post, as summarized in supplied research. | Axys version/reporting enhancements. |
| REP-005 | CSSI / Client Server Specialists PDF, “How to Add Portfolio Code to Axys Reports,” as summarized in supplied research. | Axys `.REP` file, RepLang, `AMAN.REP`, sample expressions. |
| REP-006 | Salentica / Black Diamond Advent Connector article, as summarized in supplied research. | REP32, standard reports/macros, connector-supported versions. |
| REP-007 | AdventGuru Replang / Axys / APX reporting articles, as summarized in supplied research. | APX Replang, Report Writer Pro, SQL/SSRS/Public Views/REST context, editor quirks. |
| REP-008 | Microsoft SSRS documentation, as summarized in supplied research. | General SSRS definition only, not APX-specific behavior. |
| REP-009 | [CSSI REP32 hyperlink guidance](https://cssisolutions.com/downloads/how-to-add-hyperlinks-to-reports) | Working REP32 command pattern and observed switches. |
| REP-010 | [Historical PERHSUM guidance](https://static1.1.sqspcdn.com/static/f/425065/4721492/1257913571447/Modifying-PERSHUM-Report.pdf) | `PERHSUM.REP`, Axys 3.6, and `.REP`/`.RPW` report-type evidence. |
| REP-011 | [APX Reports Guide](https://cdn.advent.com/cms/pdfs/reports/REP_APX.pdf) | SSRS framework and 29-report inventory. |

### 14.3 Supporting supplied research

| Ref | Source | Use in chapter |
|---|---|---|
| IMEX-001 | `../evidence/Research_12_IMEX.md` | REP vs IMEX distinction, connector extraction, `.RPW`/`.REP`, direct file access caution, version notes. |
| SEC-001 | `../evidence/Research_04_Security_Master.md` | REP32 and report/macro extraction context; APX public view limitations. |
| TRN-001 | `../evidence/Research_05_Transactions.md` | Transaction Summary Report/report-output cross-reference; `didpost.aud` and transaction report boundary. |

---

## 15. Unknowns

The following Unknowns should remain in the repository until supported by vendor documentation, report source files, sample outputs, production examples, or controlled tests.

### 15.1 RepLang and source syntax

| Unknown | Needed evidence |
|---|---|
| Full RepLang Programmer's Guide contents. | Advent RepLang Programmer's Guide. |
| Full RepLang grammar, including syntax, control flow, functions, date handling, grouping, totals, formatting, variables, and output options. | Programmer's Guide and report source examples. |
| Complete Axys keyword list by version. | Axys RepLang documentation by version. |
| Complete APX keyword list by version. | APX Replang documentation by version. |
| Whether APX Replang is backward-compatible with Axys Replang. | Vendor docs or paired report tests. |

### 15.2 Report libraries and file locations

| Unknown | Needed evidence |
|---|---|
| Full list of Axys standard `.REP` files. | Axys `rep` folder listing from a production/test install. |
| Full list of APX Replang reports. | APX report directory/export/sample environment. |
| APX `.REP` storage location and format. | APX admin/reporting documentation. |
| Relationship between APX Replang reports and APX SSRS reports. | APX Reports Guide / APX reporting admin documentation. |
| Report file naming conventions across versions. | Versioned report directories or release notes. |

### 15.3 Report execution and automation

| Unknown | Needed evidence |
|---|---|
| Complete REP32.exe option set and version differences. | Vendor technical docs; one working public example is documented above. |
| Macro file syntax. | Sample macros and vendor documentation. |
| How dates, portfolio groups, currencies, composites, report options, and output destinations are passed to REP32. | Macro samples / command-line examples. |
| Whether REP32 behavior differs materially between Axys/APX. | Vendor documentation or connector configuration samples. |
| Whether APX cloud environments permit client-run custom REP execution. | SS&C APX cloud/admin policy documentation. |

### 15.4 Report values and processing

| Unknown | Needed evidence |
|---|---|
| Which reports use stored values vs recalculation at run time. | Report source, vendor docs, controlled tests. |
| Which Axys reports read stored performance vs calculate performance dynamically. | Performance report source and tests. |
| Which APX reports read stored accounting functions, public views, stored performance tables, or report-specific SQL. | APX report definitions and database documentation. |
| Whether report output can be reconciled exactly to IMEX exports for each object. | Paired IMEX exports and report outputs. |
| Whether report output values change after historical transaction edits. | Controlled before/after tests. |

### 15.5 Vendor support and customization boundaries

| Unknown | Needed evidence |
|---|---|
| Current SS&C support policy for custom RepLang. | Current vendor support policy or client documentation. |
| Current SS&C support policy for Report Writer Pro modifications. | Current vendor support policy. |
| Current SS&C support policy for REP-based third-party integrations. | Vendor integration documentation. |
| Current SS&C support policy for APX public views / SQL / SSRS / REST as integration interfaces. | APX technical/admin documentation. |

---

## 16. Minimum Additional Evidence Needed for a Stronger REP Chapter

The current supplied material is sufficient for a conservative REP chapter, but not for a complete technical manual of RepLang or report automation. The following materials would materially improve this chapter:

| Needed material | Would resolve |
|---|---|
| Axys `rep` folder listing. | Standard `.REP` report catalog and file names. |
| Sample Axys `.REP` files. | Report structure, variables, parameters, output behavior. |
| RepLang Programmer's Guide. | Grammar, keyword dictionary, variables, functions, supported syntax. |
| Sample Report Writer Pro `.RPW` files and generated `.REP` source. | Report Writer Pro / RepLang relationship. |
| REP32 command-line examples. | Automation, unattended execution, parameter handling. |
| Sample macros. | Macro syntax and scheduling behavior. |
| APX Reports Guide / Reporting Guide. | APX REP vs SSRS/report package architecture. |
| APX SSRS report catalog and sample `.rdl` files. | SSRS dataset/procedure/view usage. |
| APX Public Views / Stored Accounting Functions documentation. | APX reporting/data-access source behavior. |
| Sample REP output used as data extracts. | Output parsing rules and field layout. |
| Known production examples where REP output differs from IMEX output. | Interface reconciliation and audit behavior. |
| Versioned report files from Axys 3.8.6 vs 3.8.7 or APX 15.x/16.x/17.x. | Version-difference documentation. |

## Research Provenance

The deep IMEX REP/IMEX boundary conclusions are incorporated into Section 9.4.
Their granular supporting claims and confidence boundaries remain in
`../evidence/Research_13_REP.md`.
