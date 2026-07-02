# Research_02_Axys_Architecture.md

Research status: Draft research notes  
Repository chapter target: `Chapter_02_Axys_Architecture.md`  
Prepared: 2026-06-29  
Governing specification: `AXYS_APX_REFERENCE_BLUEPRINT.md`, Version 2.0

---

## 1. Scope

This file collects factual research for the Axys architecture chapter of the Axys/APX Reference Repository.

The emphasis is Axys architecture, with APX included where it clarifies version differences, migration behavior, shared reporting infrastructure, IMEX behavior, or architectural contrast.

This file is research material, not a finished repository chapter.

---

## 2. Confidence Classification

| Classification | Meaning in this research file |
|---|---|
| Verified | Supported directly by cited vendor material, regulatory filing, available public documentation, or directly supplied repository specification. |
| High Confidence | Supported by credible consultant documentation, public product pages, multiple consistent secondary sources, or long-standing product behavior, but not fully verified from vendor technical manuals. |
| Medium Confidence | Plausible and supported by partial evidence or practitioner material, but requires confirmation from vendor manuals, installed systems, sample exports, or production observations. |
| Unknown | Not sufficiently supported by the supplied material or public evidence reviewed. Do not treat as fact. |

---

## 3. Source Register

| Source ID | Source | Type | Relevance | Notes |
|---|---|---|---|---|
| S1 | `AXYS_APX_REFERENCE_BLUEPRINT.md`, Version 2.0 | User-supplied repository specification | Governing editorial rules | Requires facts first, Axys/APX separation, confidence labels, Unknowns, tables, IMEX/REP/report detail where supported. |
| S2 | SS&C Advent Axys product page, https://www.advent.com/solutions/axys/ | Vendor product page | Axys feature-level behavior and storage positioning | Supports Axys portfolio accounting, reporting, performance measurement, Report Writer Pro, security types, accounting options, grouping, GIPS-related statements, and proprietary-database product positioning. |
| S3 | SS&C Advent APX product page, https://www.advent.com/solutions/advent-portfolio-exchange/ | Vendor product page | APX contrast | Supports APX as integrated client relationship and portfolio management solution, single platform, centralized/scalable, standard reports, report packaging, cloud/local deployment statements. |
| S4 | Advent Software 2007 SEC filing, https://www.sec.gov/Archives/edgar/data/1002225/000110465907025400/a07-7653_110ka.htm | Regulatory filing | Historical product descriptions | Supports historical descriptions of APX as SQL database/browser UI and Axys as portfolio management/reporting system with Report Writer Pro and multi-currency capabilities. |
| S5 | Advent client story PDF, River Road / APX, https://cdn.advent.com/cms/pdfs/clients/CS_RR.pdf | Vendor client story | APX architecture and migration contrast | Supports APX SQL-based database platform, integrated accounting/reporting/CRM, Packager, migration from Axys. |
| S6 | AdventGuru IMEX article, https://adventguru.com/tag/imex/ | Consultant/practitioner article | IMEX and version behavior | Supports Axys v1 open text, Axys v2 binary, IMEX CSV/tab/fixed import/export, Axys 3.7 to 3.8 file conversion caution, APX fixed-format limitation. |
| S7 | AdventGuru Replang / Axys article, https://adventguru.com/tag/axys/ | Consultant/practitioner article | REP/Replang/reporting architecture | Supports Replang as part of Axys/APX reporting architecture and use of Report Writer Pro/Replang source modification. |
| S8 | AdventGuru Report Writer article, https://adventguru.com/2019/06/26/updating-report-writer-reports-without-the-app/ | Consultant/practitioner article | RPW/REP/Replang mechanics | Supports RPW extension, REP distinction, Replang underlying format, checksum/report writer modification quirk. |
| S9 | Advent User Group PDF: “How to Add Portfolio Code to Axys Reports,” https://assets.ctfassets.net/xhy36q2d1lqu/77QC4aNbyhPo9FfmjRYNzc/d00a0d6601214601543e30e34f203626/PortfolioCodetoAxys.pdf | User-group technical note | Axys REP example | Supports Axys reports written in RepLang, AMAN.REP example, report directory example, `$:fileo` as portfolio code in a specific report-editing example. |
| S10 | Salentica Data Broker KB, https://engage.salentica.com/kb/article/247-data-broker-ss-c-advent-apx-axys/ | Integration vendor documentation | Deployment and integration context | Supports Axys/APX as on-prem/local-machine/VM/server systems in that integration context and PMS-to-CRM data extraction. |
| S11 | Financial Advisor Magazine article, https://www.fa-mag.com/news/the-best-software-you-might-never-own-1513.html?issue=73 | Industry article | Historical Axys architecture commentary | Supports historical claim that Axys launched in 1993, used proprietary/flat-file database technology. Treat as secondary source. |
| S12 | FinFolio Advent conversion page, https://www.finfolio.com/advent-apx-moxy-apx-conversions | Migration vendor page | File-name clues | Mentions CLI, PRI, INF, PRF, GRP files in Advent conversions. Treat as Medium Confidence unless confirmed against Axys manuals or production files. |
| S13 | ByAllAccounts Custodial Integrator Axys User Guide, https://www.byallaccounts.net/Manuals/Custodial_Integrator/axys/CI_User_Guide.pdf | Integration vendor documentation | Axys workflow architecture | Supports a local intranet CI workflow that downloads WebPortfolio data, merges it with Axys security information, creates transaction/position/price files, and imports through the Axys Import/Export utility. |

---

## 4. Executive Research Summary

| Topic | Axys | APX | Confidence |
|---|---|---|---|
| Product type | Portfolio accounting, portfolio management, performance measurement, and reporting system. | Integrated portfolio accounting, reporting, performance, and client relationship/prospect management platform. | Verified |
| Primary architecture character | Public sources support Axys as a proprietary-database portfolio accounting/reporting platform that is file-oriented in practical integration and operations evidence. Vendor pages do not expose complete storage internals. | Public vendor and SEC sources describe APX as a single SQL-based database platform with browser-based or enterprise access characteristics. | Axys: High Confidence for combined characterization; APX: Verified |
| Reporting architecture | Uses Report Writer Pro and Replang/REP-style reports. | Also uses Report Writer Pro/Replang/REP in at least some reporting workflows, while APX also supports broader reporting/packaging and SQL-based reporting options. | High Confidence |
| IMEX role | Import/export mechanism used to move data in and out of Axys; supports CSV, tab, and fixed formats according to practitioner source. | IMEX functionality maintained in APX v1.x to v4.x per practitioner source, but fixed-format generation was reportedly eliminated. | High Confidence |
| Direct file access | Practitioner source states Axys users may read/write data files directly if they know the format, but this is not best practice because formats changed across versions. | APX users may query SQL database or use SQL-based reporting/extraction tools. | Axys: Medium/High Confidence; APX: High Confidence |
| Chapter implication | The Axys architecture chapter should describe Axys as proprietary-database and file-oriented in practice, with Report Writer/Replang, IMEX, version-specific file conversion risk, and local/on-prem connector patterns. | APX should be treated mainly as architectural contrast or cross-reference to Chapter 03, not as the main subject of Chapter 02. | Verified as repository-structure implication from S1 |

---

## 5. Axys Architecture: Verified and High-Confidence Facts

### 5.1 Product Role

| Statement | Classification | Evidence / Notes |
|---|---:|---|
| Axys is a portfolio management and reporting solution for investment management organizations. | Verified | Vendor product page and Advent SEC filing. |
| Axys provides portfolio accounting functionality, performance measurement, and flexible reporting. | Verified | Vendor product page and SEC filing describe these capabilities. |
| Axys supports reporting on holdings, asset allocation, realized/unrealized gains and losses, income, performance, and related portfolio information. | Verified | SEC filing describes these categories. |
| Axys supports security types including cash, equities, fixed income, money market/cash types, municipal bonds, corporate/government bonds, mortgage-backed securities, and step-up bonds. | Verified | Vendor product page. |
| Axys supports tax-lot or average-cost accounting and trade-date or settlement-date accounting as selectable accounting treatments. | Verified | Vendor product page. |
| Axys supports portfolio grouping/reporting by manager, asset class, investment objective, and custom categories. | Verified | Vendor product page. |
| Axys has integrated multi-currency capabilities, including reporting restatement in any currency and separation of return components attributable to market prices versus currency-rate fluctuations. | Verified | Advent SEC filing. |
| Axys can calculate time-weighted and internal rates of return, before or after management fees. | Verified | Vendor product page. |
| Axys provides Report Writer Pro for custom reports. | Verified | Vendor product page and SEC filing. |

### 5.2 Architectural Character

| Statement | Classification | Evidence / Notes |
|---|---:|---|
| Axys should be described as proprietary-database and file-oriented in practice, not merely as a flat-file system. | High Confidence | SS&C product positioning supports proprietary-database wording; practitioner/integration evidence supports file-oriented operational handling. Complete physical internals remain Unknown. |
| Axys v1.x reportedly maintained an open text file structure similar to earlier Advent Professional Portfolio behavior. | Medium Confidence | Practitioner source. Needs vendor manual or installed version confirmation. |
| Axys v2.x reportedly introduced a binary file format. | Medium Confidence | Practitioner source. Needs vendor manual or installed version confirmation. |
| Most firms reportedly moved eventually to Axys v3.x, and IMEX reduced concern about binary file formats by allowing supported import/export. | Medium Confidence | Practitioner source. Needs corroboration. |
| A direct-read/direct-write approach against Axys data files is risky because file formats can change by version. | High Confidence | Practitioner source specifically cites version-to-version conversion concerns; operationally consistent with proprietary/binary storage. |
| Upgrading from Axys 3.7 to Axys 3.8 reportedly requires a file conversion process, and some resulting 3.8 files reportedly have different file formats. | Medium Confidence | Practitioner source. Requires vendor release documentation or production observation for verification. |

### 5.3 Deployment / Runtime Context

| Statement | Classification | Evidence / Notes |
|---|---:|---|
| Axys is marketed as not disrupting existing infrastructure. | Verified | Vendor product page. |
| Axys/APX integrations may be installed on a local machine, virtual machine, or server on the firm's network in some integration-vendor deployment contexts. | High Confidence | Salentica KB describes Axys/APX as on-prem and not publicly accessible for that integration. |
| Whether all current SS&C-hosted Axys deployments are strictly on-prem, hosted, cloud-delivered, or hybrid is Unknown from the reviewed material. | Unknown | Requires vendor deployment documentation or client environment evidence. |
| Axys executable names, service names, scheduler components, directory conventions, and network share requirements are Unknown from the reviewed material. | Unknown | Needs installed-system evidence or vendor technical docs. |

---

## 6. Axys Storage and File Architecture

### 6.1 Storage-Level Research Findings

| Item | Finding | Classification | Notes |
|---|---|---:|---|
| Overall storage architecture | Axys is best described as proprietary-database and file-oriented in practical integration/operations evidence rather than as a simple flat-file system or a documented SQL database. | High Confidence for characterization; Unknown for internals | Do not write a chapter sentence claiming exact physical layout without verification. |
| Axys v1.x | Reported to use an open text file structure. | Medium Confidence | Practitioner source only. |
| Axys v2.x | Reported to introduce binary file format. | Medium Confidence | Practitioner source only. |
| Axys v3.x | Reported to use IMEX to export/import CSV, tab, and fixed formats. | High Confidence | Practitioner source. |
| Axys 3.7 to 3.8 | Reported file conversion and some changed file formats. | Medium Confidence | Practitioner source. |
| SQL database | No reviewed source verifies Axys as SQL-based. | Unknown / Negative Evidence | APX is the product clearly described as SQL-based. |
| Direct file modification | Possible for knowledgeable users per practitioner source, but not best practice. | Medium Confidence | Best chapter treatment: “unsupported/high risk unless verified.” |

### 6.2 Candidate Axys File Names Mentioned by Public Sources

These file names should not be treated as a complete Axys file dictionary. They are included only as research leads.

| File / Pattern | Possible Meaning | Source | Confidence | Chapter Handling |
|---|---|---|---:|---|
| `CLI` | Client/account/transaction-related files in migration context. | Migration vendor page. | Medium Confidence | Do not define exact layout without sample files or manual. |
| `PRI` | Prices/factors in migration context. | Migration vendor page. | Medium Confidence | Verify with Axys file documentation or sample exports. |
| `INF` | Security master / type / split information in migration context. | Migration vendor page. | Medium Confidence | Verify exact file names and contents. |
| `SECURITY.INF` | Security master in migration context. | Migration vendor page. | Medium Confidence | Candidate for Chapter 04 cross-reference. |
| `TYPE.INF` | Security type information in migration context. | Migration vendor page. | Medium Confidence | Candidate for Chapter 04/15 cross-reference. |
| `SPLIT.INF` | Split information in migration context. | Migration vendor page. | Medium Confidence | Candidate for Chapter 09 cross-reference. |
| `PRF` | Performance returns in migration context. | Migration vendor page. | Medium Confidence | Candidate for Chapter 10 cross-reference. |
| `GRP` | Groups/color groups in migration context. | Migration vendor page. | Medium Confidence | Candidate for classifications/groups cross-reference. |
| `.REP` | Replang report source/report files. | AdventGuru and AUG technical note. | High Confidence | Safe to discuss in REP/reporting architecture. |
| `.RPW` | Report Writer-created report file extension; underlying format described as Replang by practitioner source. | AdventGuru Report Writer article. | High Confidence | Safe with source qualification. |

### 6.3 Unknown File Details

| Unknown | Why it matters |
|---|---|
| Exact Axys physical data directory structure | Needed to document backup, migration, file inventory, and source-control practices. |
| Exact file-level ownership of portfolios, transactions, prices, securities, classifications, groups, and performance | Needed for architecture and data dictionary chapters. |
| Whether file names differ by Axys version, install options, or module | Needed for safe implementation guidance. |
| Whether `.CLI`, `.PRI`, `.INF`, `.PRF`, `.GRP` are always native files, export files, or migration shorthand in each context | Prevents false file dictionary entries. |
| File locking / multi-user concurrency model | Architecture-critical; not verified in reviewed sources. |
| Backup consistency requirements | Architecture-critical; not verified in reviewed sources. |
| Character encoding and delimiter rules for native files and IMEX exports | Required for reliable tooling; not verified in reviewed sources. |

---

## 7. Reporting Architecture: REP, Replang, Report Writer Pro

### 7.1 Supported Findings

| Statement | Axys | APX | Confidence |
|---|---|---|---:|
| Report Writer Pro is used to create custom reports. | Verified | High Confidence | Axys vendor page verifies; APX use is supported by practitioner sources. |
| Replang is part of the reporting architecture. | High Confidence | High Confidence | Practitioner sources state Replang remains part of Axys/APX reporting architecture. |
| Axys reports are written in RepLang, Advent's proprietary Report Writing Language. | High Confidence | Unknown for all APX reports | AUG technical note specifically says Axys reports are written in RepLang. |
| `.REP` files are associated with Replang reports. | High Confidence | High Confidence | Practitioner/AUG sources. |
| `.RPW` files are Report Writer-created files whose underlying format is Replang. | High Confidence | High Confidence | Practitioner source. |
| Report Writer-created `.RPW` files contain a checksum; manual code changes can make future editing in Report Writer problematic. | High Confidence | High Confidence | Practitioner source. |
| If a Report Writer file is manually modified into unsupported Replang, a copy/backup workflow is recommended. | High Confidence | High Confidence | Practitioner source. |
| Plain text editors such as Notepad/Notepad++/VS Code can be used to modify Replang source files. | Medium Confidence | Medium Confidence | Practitioner source; supportability depends on firm/vendor policy. |
| Advent Support does not answer RepLang questions according to an AUG technical note. | Medium Confidence | Unknown | Historical/user-group source; verify current support policy. |

### 7.2 Report Names / Report File Examples

| Report / File | System | Description | Confidence | Notes |
|---|---|---|---:|---|
| `AMAN.REP` | Axys | Assets Under Management report file in an AUG example. | High Confidence | Technical note shows how to copy and edit this report. |
| `AMAN_XX.REP` | Axys | Example copy of `AMAN.REP` for customization. | High Confidence | Example name, not standard production report. |
| “Assets Under Management Report” | Axys | Standard report accessed from Axys Reports → Mgmt → Assets Under Management in AUG example. | High Confidence | Menu path may be version-specific; verify. |
| “Performance History Report” | Axys | Mentioned in migration/consulting download context. | Medium Confidence | Needs direct report catalog verification. |
| “Report Writer Pro reports” | Axys/APX | Custom reports built through Report Writer Pro. | High Confidence | Exact catalog unknown. |
| “Advent Packager” / report packaging | APX | APX incorporated Advent Packager according to client story; APX product page also mentions automated report packaging. | Verified | This belongs primarily in APX architecture but useful as contrast. |

### 7.3 Replang Example Tokens / Fields Observed in Public Material

These are not a general Replang reference. They are only tokens observed in the AUG example.

| Token / Field | Context | System | Meaning from source | Confidence |
|---|---|---|---|---:|
| `.#~8portmv` | AUG example editing `AMAN.REP` | Axys | Prints portfolio market value in the example. | Medium Confidence |
| `$:fileo` | AUG example editing `AMAN.REP` | Axys | Portfolio code in the example. | Medium Confidence |
| `#width #cnt 16* 25+ 16+` | AUG example editing `AMAN.REP` | Axys | Width/layout expression modified to allow more room. | Medium Confidence |
| `\n` | AUG example | Axys | Carriage return/end-of-line in printed output. | Medium Confidence |

Chapter warning: these tokens should be included only as examples from a specific report-editing source, not as a universal REP field dictionary.

---

## 8. IMEX Research

### 8.1 Role of IMEX

| Statement | Axys | APX | Confidence |
|---|---|---|---:|
| IMEX provides a supported mechanism to move data into and out of Advent portfolio systems. | High Confidence | High Confidence | Practitioner source; aligns with migration/integration behavior. |
| IMEX allows Axys users to import/export files in CSV, tab, and fixed formats. | High Confidence | Not fully supported for fixed output in APX per source | Practitioner source. |
| IMEX reduced the need to directly read/write proprietary Axys data files after binary formats were introduced. | Medium Confidence | Not applicable | Inference from practitioner source; verify in manuals. |
| IMEX functionality exists in APX v1.x through v4.x. | Medium Confidence | Medium Confidence | Practitioner source; version-specific confirmation needed. |
| APX v1.x to v4.x reportedly eliminated fixed-format file generation while maintaining IMEX functionality. | Medium Confidence | Medium Confidence | Practitioner source only. |
| APX can export data to Axys v3 format according to practitioner source. | Medium Confidence | Medium Confidence | Needs APX IMEX manual verification. |

### 8.2 IMEX Architecture Implications

| Implication | Confidence | Notes |
|---|---:|---|
| For Axys integrations, IMEX should generally be preferred over native file reads/writes when practical. | High Confidence | Supported by file-format-change risk. |
| Custom exports may also be built by modifying REP/Replang reports or Report Writer Pro reports to produce CSV/text. | High Confidence | Supported by practitioner source. |
| IMEX should be documented separately from report exports because IMEX is an import/export subsystem, while REP/Report Writer exports are report-output mechanisms. | High Confidence | Repository structure separates IMEX and REP; public sources support both as distinct mechanisms. |
| Exact IMEX object names, field names, delimiters, and import/export command syntax are Unknown from reviewed material. | Unknown | Requires IMEX manual or sample exports. |

### 8.3 IMEX Unknowns Needed for Architecture Chapter

| Unknown | Needed Evidence |
|---|---|
| IMEX executable/module name and invocation syntax | Vendor manual or installed-system examples. |
| Whether IMEX is GUI-only, command-line capable, macro-driven, or batch schedulable in specific versions | Vendor manual or production scripts. |
| IMEX object list for portfolios, transactions, securities, prices, holdings, groups, performance | IMEX object catalog or sample `*.imx` configuration. |
| Whether Axys and APX use identical IMEX object names | Side-by-side IMEX documentation or exports. |
| Error/log file behavior | IMEX manual or production logs. |
| Import validation behavior and transaction rollback behavior | IMEX manual or production tests. |
| REP versus IMEX export equivalence for common extracts | Sample reports/exports. |

---

## 9. APX Architecture Contrast

APX belongs primarily in `Research_03_APX_Architecture.md` and `Chapter_03_APX_Architecture.md`, but the following points are relevant to the Axys architecture chapter because APX is the main architectural successor/contrast.

### 9.1 Verified APX Facts

| Statement | Classification | Evidence / Notes |
|---|---:|---|
| APX is an integrated client relationship management and portfolio management solution. | Verified | Vendor page. |
| APX connects front, middle, and back offices on a single platform. | Verified | Vendor page. |
| APX is described as centralized and scalable for portfolio, relationship, and prospect data. | Verified | Vendor page. |
| APX supports a library of standard reports, automated report packaging, performance analytics, flexible custom reporting, composite management support for GIPS compliance, multi-currency/multi-asset coverage, and front-to-back suite integration. | Verified | Vendor page. |
| APX can be deployed locally or cloud-delivered with or without outsourcing services, according to the vendor product page. | Verified | Vendor page. |
| Historical Advent SEC filing describes APX as leveraging a single SQL database and delivering client/prospect information through a browser-based user interface. | Verified as historical | 2007 SEC filing. |
| Historical Advent client story describes APX as integrating portfolio accounting, reporting, and client relationship management on a single SQL-based database platform. | Verified as vendor client-story claim | Client story PDF. |
| APX includes built-in performance measurement and analytics for performance attribution, according to the client story. | Verified as vendor client-story claim | Client story PDF. |
| APX incorporated Advent Packager in the cited client story. | Verified as vendor client-story claim | Client story PDF. |
| APX was described in 2007 as a migration path for Axys clients. | Verified as historical | SEC filing. |

### 9.2 Architectural Contrast Table

| Dimension | Axys | APX | Confidence |
|---|---|---|---:|
| Data architecture | Proprietary/file-oriented according to secondary/practitioner sources; exact internals unverified. | SQL-based database platform according to vendor/SEC sources. | Axys Medium; APX Verified |
| User interface | Unknown from reviewed architecture sources. | Historical SEC filing describes browser-based UI. | Axys Unknown; APX Verified historical |
| Reporting | Report Writer Pro and Replang/REP are core. | Report Writer/Replang may remain available; APX also supports automated report packaging and SQL-oriented reporting options. | High Confidence |
| Integration | IMEX, report exports, custom REP, third-party ETL, possible direct file access with risk. | SQL queries, Excel/SQL tools, SSRS/Crystal reports, IMEX, report exports. | High Confidence |
| Migration relationship | Axys predecessor/installed base. | APX described as migration path for Axys clients. | Verified historical |
| Scalability positioning | Vendor markets Axys for asset managers, wealth managers, family offices; public source says hundreds of firms. | Vendor and client story position APX as centralized/scalable enterprise solution. | Verified |

---

## 10. Processing Behavior

### 10.1 Verified / High-Confidence Processing Behavior

| Processing Area | Axys | APX | Confidence | Notes |
|---|---|---|---:|---|
| Portfolio accounting | Records/accounts for equities, fixed income, mutual funds, cash, and other instruments. | Tracks holdings, transactions, and performance. | Verified | From vendor/SEC sources. |
| Accounting basis | Supports tax-lot or average-cost accounting; supports trade-date or settlement-date accounting. | Unknown from reviewed APX technical material. | Axys Verified; APX Unknown | APX likely supports related accounting functions, but do not claim without source. |
| Performance calculation | Calculates time-weighted and internal rates of return before/after fees. | Supports performance analytics; client story says built-in performance measurement and attribution. | Axys Verified; APX Verified at feature level | Exact stored/recalculated behavior Unknown. |
| Grouped reporting | Reports portfolios grouped by manager, asset class, investment objective, or custom category. | Supports centralized relationship/prospect/portfolio data and standard/custom reporting. | Axys Verified; APX Verified at feature level | Exact group data model Unknown. |
| Reconciliation | Axys vendor page states automated reconciliation of trade information, settlement data, transactions, and positions. | Unknown from reviewed technical material. | Axys Verified; APX Unknown | APX likely has reconciliation workflows but not asserted here. |
| Fixed income analytics | Axys tracks yield method, amortization/accretion, duration, odd coupon dates, ratings, tax status, etc. | Unknown from reviewed technical material. | Axys Verified; APX Unknown | Needs APX source. |

### 10.2 Important Unknown Processing Questions

| Question | Why it belongs in architecture research |
|---|---|
| Does Axys store calculated performance values, recalculate on demand, or both? | Affects runtime architecture, reproducibility, and audit behavior. |
| Which Axys reports use stored performance versus recalculated performance? | Required for report architecture and performance chapters. |
| Where does Axys maintain portfolio groups, classifications, benchmarks, and composites? | Needed for architecture/data model cross-reference. |
| How does Axys lock files during posting, pricing, reporting, IMEX, or backup? | Critical for integrations and operational reliability. |
| How does Axys handle concurrent users? | Critical for architecture chapter; not verified. |
| What are the batch scheduling and automation mechanisms? | Needed for production processing architecture. |
| What logs are created by Axys, IMEX, report runs, imports, and conversions? | Needed for auditability and support. |
| Are there supported APIs beyond IMEX/REP/report exports? | Unknown from reviewed material. |

---

## 11. Data Model Research

### 11.1 Common Architectural Entities

These entities are supported at the feature level, but their exact physical file/table names are mostly Unknown.

| Entity | Axys Support | APX Support | Confidence | Notes |
|---|---|---|---:|---|
| Portfolio / account | Supported. | Supported. | Verified | Exact keys/field names Unknown. |
| Client / relationship | Supported in reporting/account context; exact client model Unknown. | Client relationship and prospect data central to APX. | Axys Medium; APX Verified | APX has CRM-style integration. |
| Security master | Supported indirectly by security-type and instrument support. | Supported indirectly by holdings/transactions/security coverage. | High Confidence | Exact fields Unknown. |
| Transactions | Supported. | Supported. | Verified | Exact transaction file/table fields Unknown. |
| Holdings / positions | Supported. | Supported. | Verified | Exact fields Unknown. |
| Prices | Supported indirectly by performance/market-price return components. | Supported indirectly. | Medium Confidence | Exact price file/table Unknown. |
| Cash | Supported as security/instrument type and accounting category. | Supported indirectly by portfolio accounting. | Verified / High Confidence | Exact cash files Unknown. |
| Performance | Supported. | Supported. | Verified | Exact storage/calculation model Unknown. |
| Benchmarks / indices | Axys supports blended benchmarks and component index history. | Supports performance analytics; exact benchmark model Unknown. | Axys Verified; APX Unknown/High Confidence | APX requires APX docs. |
| Groups / classifications | Axys can group/report by manager, asset class, objective, custom categories; can display performance by asset class, sector, country, region. | APX supports centralized portfolio/relationship/prospect data and standard/custom reports. | Axys Verified; APX Unknown | Exact classification stores Unknown. |
| Composites | Axys supports GIPS-related performance measurement/reporting. | APX has composite management support for GIPS compliance. | Verified | Exact composite model Unknown. |

### 11.2 Field Dictionary: Architecture-Level Fields Mentioned in Sources

This table is not a full Axys/APX field dictionary. It lists only fields/tokens observed or directly implied by reviewed source material.

| Field / Token | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---:|
| `portfolio code` | Portfolio identifier displayed in an AUG Replang customization example. | Yes | Unknown | Unknown | Yes | Medium Confidence |
| `$:fileo` | Replang token used in the AUG example to display portfolio code. | Yes | Unknown | Unknown | Yes | Medium Confidence |
| `portmv` / `.#~8portmv` | Replang/example token for portfolio market value in `AMAN.REP` editing example. | Yes | Unknown | Unknown | Yes | Medium Confidence |
| account / portfolio number | PMS account number is distinguished from custodian account number in integration-vendor documentation. | Yes | Yes | Unknown | Unknown | High Confidence |
| custodian account number | Custodian-side account identifier distinct from PMS account number. | Yes | Yes | Unknown | Unknown | High Confidence |
| market value | Reported/calculated portfolio amount; token observed in Replang example. | Yes | Unknown | Unknown | Yes | Medium Confidence |
| security type | Security master/type concept inferred from vendor security-type support and migration file `TYPE.INF`. | Yes | Unknown | Unknown | Unknown | Medium Confidence |
| performance return | Performance returns mentioned in vendor pages and migration file `PRF`. | Yes | Yes | Unknown | Unknown | Medium Confidence |
| group | Portfolio/account group concept supported by vendor pages and migration file `GRP`. | Yes | Unknown | Unknown | Unknown | Medium Confidence |

### 11.3 Data Model Unknowns

| Area | Unknowns |
|---|---|
| Portfolio identifiers | Exact Axys portfolio code field name, length, case sensitivity, uniqueness rules, and relationship to file names. |
| Security identifiers | Primary security identifier, ticker/CUSIP/SEDOL/ISIN handling, internal security ID behavior. |
| Transaction schema | Native transaction fields, transaction-code dictionary, date fields, settlement/trade-date handling. |
| Holdings schema | Position fields, cost fields, lot fields, accrual fields, cash treatment. |
| Pricing schema | Price-date keys, currency fields, price factors, split factors, stale-price behavior. |
| Performance schema | Stored performance file/table names, period keys, gross/net fields, classification-level fields. |
| Group/classification schema | Storage of manager/asset-class/objective/custom categories, color groups, composite membership. |
| Benchmark schema | Index definitions, blended benchmark history, synthetic indices. |
| Multi-currency schema | Currency code fields, FX rates, local/base return decomposition. |

---

## 12. Integration Architecture

### 12.1 Supported Integration Mechanisms

| Mechanism | Axys | APX | Confidence | Notes |
|---|---|---|---:|---|
| IMEX | Supported. | Supported per practitioner source. | High Confidence | Exact object list Unknown. |
| Report exports to Excel/XLS | Supported. | Supported. | High Confidence | Practitioner source says Axys/APX reports can be exported directly to Excel and macros can store outputs in XLS/other formats. |
| Report Writer Pro CSV/text reports | Supported. | Supported. | High Confidence | Practitioner source. |
| Replang custom reports | Supported. | Supported. | High Confidence | Practitioner source. |
| Direct native file reads/writes | Possible for knowledgeable Axys users but risky/not best practice. | Not typical; APX is SQL-based. | Medium Confidence | Do not recommend without caution. |
| SQL queries | Not verified for Axys. | Supported. | Axys Unknown; APX High/Verified | APX SQL basis is verified. |
| Excel query against database | Not verified for Axys. | Supported per practitioner source. | APX High Confidence | Need APX security/permissions documentation. |
| SSRS / Crystal reports | Not verified for Axys. | Supported per practitioner source. | APX High Confidence | Needs firm-specific deployment verification. |
| Third-party ETL | Supported in practice. | Supported in practice. | High Confidence | Practitioner source mentions xPort/ETL-style products. |
| PMS-to-CRM Data Broker extraction | Supported in Salentica integration context. | Supported in Salentica integration context. | High Confidence | Field mapping performed during implementation. |

### 12.2 Architecture Guidance for Future Chapter

| Guidance | Confidence | Rationale |
|---|---:|---|
| Treat IMEX and REP/report-export integrations as safer than native Axys file manipulation. | High Confidence | File-format changes and version conversion risk. |
| Treat APX SQL access as an APX architectural distinction, not an Axys capability. | Verified | APX SQL is supported by vendor/SEC sources; Axys SQL is not. |
| Do not promise that report export, IMEX, and native data contain equivalent values. | Unknown | Needs side-by-side sample comparison. |
| Distinguish “data extraction for integration” from “client reporting.” | High Confidence | IMEX/ETL versus REP/Report Writer serve overlapping but distinct use cases. |
| For any implemented extract, capture source mechanism, report/IMEX object, version, parameters, date range, and output schema. | High Confidence | Required for reproducibility and auditability. |

---

## 13. Version Differences

| Version / Era | Reported Difference | Confidence | Evidence / Notes |
|---|---|---:|---|
| Professional Portfolio / Proport | Earlier Advent system reportedly stored files in open text format. | Medium Confidence | Practitioner source. |
| Axys v1.x | Reportedly maintained similar open file structure. | Medium Confidence | Practitioner source. |
| Axys v2.x | Reportedly first version to implement binary file format. | Medium Confidence | Practitioner source. |
| Axys v3.x | IMEX allowed CSV/tab/fixed import/export and reduced concerns about binary storage. | High Confidence | Practitioner source. |
| Axys 3.7 to 3.8 | Reportedly requires file conversion; some resulting files have different formats. | Medium Confidence | Practitioner source. |
| APX v1.x to v4.x | Reportedly maintains IMEX functionality but eliminates fixed-format generation. | Medium Confidence | Practitioner source. |
| APX 2007-era | SEC filing describes APX as single SQL database and browser-based UI. | Verified historical | May not fully describe current APX versions. |
| Current APX | Vendor page describes local or cloud-delivered deployment with/without outsourcing services. | Verified | Current public product page. |

---

## 14. Known Issues / Quirks

| Quirk / Issue | System | Confidence | Practical Impact |
|---|---|---:|---|
| Native Axys file formats may change across versions. | Axys | High Confidence | Direct file integrations can break after upgrades; IMEX/report exports preferred. |
| Axys 3.7 to 3.8 conversion reportedly changed some file formats. | Axys | Medium Confidence | Upgrade testing should include all custom integrations. |
| Report Writer-created `.RPW` files have checksum behavior; manual edits can prevent future editing in Report Writer. | Axys/APX reporting | High Confidence | Preserve original `.RPW`, copy to `.REP`, document manual changes. |
| Replang source generated by Report Writer may be abstract and harder to modify manually than hand-written Replang. | Axys/APX reporting | High Confidence | Maintenance risk; requires experienced report developer. |
| REP/Replang development is usually text-editor-based in practitioner workflows; modern editor support may be limited. | Axys/APX reporting | Medium Confidence | Source-control and editor setup are firm-specific. |
| APX SQL architecture does not automatically make APX “open” for all integration purposes. | APX | Medium Confidence | SQL access may still require vendor schema knowledge, permissions, support boundaries, and version compatibility. |
| PMS account numbers and custodian account numbers may differ and should not be conflated. | Axys/APX integrations | High Confidence | Important for CRM, custodian, reconciliation, and reporting integrations. |
| Public sources do not provide reliable field dictionaries. | Axys/APX | Verified by absence | Exact field names must come from vendor docs, IMEX exports, REP source, or production samples. |

---

## 15. Examples

### 15.1 Example: Safer Axys Integration Pattern

| Step | Mechanism | Confidence | Notes |
|---|---|---:|---|
| 1 | Identify needed data domain: portfolios, transactions, securities, prices, holdings, performance, groups. | High Confidence | Domain list supported at feature level. |
| 2 | Prefer IMEX export or REP/report export over native file reads. | High Confidence | Avoids file-format risk. |
| 3 | Save exact export/report name, parameters, Axys version, run date, date range, and output schema. | High Confidence | Required for reproducibility. |
| 4 | Validate totals against standard Axys reports. | Medium Confidence | Specific report names unknown. |
| 5 | Treat custom Replang reports as source code; preserve original vendor files. | High Confidence | Supported by REP/RPW checksum behavior. |
| 6 | Regression-test after Axys upgrades. | High Confidence | Version conversion/file-format risk. |

### 15.2 Example: REP Customization Workflow from AUG Source

| Step | Action | Confidence | Notes |
|---|---|---:|---|
| 1 | From Axys Reports menu, choose standard Assets Under Management report. | High Confidence for example | Menu path may be version-specific. |
| 2 | Observe report path/file name in Axys Reports UI. | High Confidence for example | AUG source says Axys displays path/file in lower-left corner. |
| 3 | Copy `AMAN.REP` to a new file such as `AMAN_XX.REP`. | High Confidence for example | Avoids modifying original. |
| 4 | Run custom report through Custom → Any Report. | High Confidence for example | Version/menu path may vary. |
| 5 | Edit `.REP` in a text editor, not a word processor. | High Confidence for example | Source recommends text editor. |
| 6 | Add `$:fileo` after market value token to display portfolio code. | Medium Confidence | Specific to example. |
| 7 | Adjust width expression to allow display space. | Medium Confidence | Specific to example. |

### 15.3 Example: Axys-to-APX Architectural Migration Contrast

| Area | Axys | APX | Confidence |
|---|---|---|---:|
| Data access | File/IMEX/report oriented. | SQL database/report/IMEX oriented. | Axys Medium; APX Verified/High |
| Reporting | Report Writer Pro/Replang. | Standard reports, custom reports, Packager, Report Writer/Replang, SQL-based reporting options. | High |
| Operations | Portfolio accounting/reporting. | Integrated accounting/reporting/CRM/front-to-back platform. | Verified |
| Migration | Source system in many client stories. | Migration path/successor platform in historical vendor descriptions. | Verified historical |

---

## 16. Chapter-Writing Guidance

### 16.1 What Can Be Stated Firmly in `Chapter_02_Axys_Architecture.md`

| Statement Type | Safe Treatment |
|---|---|
| Axys product role | State as Verified: portfolio accounting, reporting, performance measurement system. |
| Axys reporting | State as Verified/High Confidence: Report Writer Pro and Replang/REP are central/custom reporting mechanisms. |
| Axys IMEX | State as High Confidence: IMEX is a supported import/export path and should be preferred over direct file manipulation where possible. |
| Axys storage | State cautiously: vendor material supports proprietary-database positioning, while public/practitioner sources support file-oriented operational handling; exact physical file architecture requires vendor or production evidence. |
| Version risk | State as High/Medium Confidence: direct native-file integrations are risky because file formats have changed across versions. |
| APX contrast | State as Verified: APX is SQL-based, centralized, integrated platform in vendor/SEC historical sources. |

### 16.2 Source-Surface and Lineage Rules

Every architecture-based extract should identify the source surface that produced
the value. Do not assume that an IMEX export, REP report, SSRS report, SQL view,
REST API response, connector output, or direct native-file read is equivalent
until reconciled.

| Claim type | Evidence required |
|---|---|
| Report displays value | Report output or report guide. |
| REP token computes value | REP source or Replang documentation. |
| IMEX exports value | IMEX object/export sample. |
| APX SQL stores value | Schema/table/view evidence. |
| Axys native file stores value | File layout, sample, or vendor evidence. |
| Value is books-and-records | Firm policy, vendor documentation, or controlled reconciliation. |
| Value is calculated at runtime | Report/function documentation or controlled test. |

Recommended extraction lineage fields include `system_name`, `system_version`,
`deployment_type`, `source_surface`, `source_object_or_report`,
`source_parameters`, `run_datetime`, `posting_state`, `price_source`,
`classification_source`, `calculation_mode`, `lineage_confidence`,
`raw_output_hash`, and `parser_version`.

### 16.3 What Should Be Marked Unknown in the Chapter

| Area | Required to Verify |
|---|---|
| Native Axys file layouts | Vendor technical manuals or sample system files. |
| Complete file dictionary | Installed system file inventory and Axys manuals. |
| Complete IMEX object list | IMEX manual or sample configurations. |
| Complete REP field dictionary | Replang Programmer’s Guide, report source inventory, production examples. |
| Locking/concurrency | Vendor admin documentation or production observation. |
| Stored versus calculated performance behavior | Vendor performance documentation and report tests. |
| Batch automation commands | Admin manuals or production scripts. |
| Error/log behavior | IMEX/report/admin documentation and logs. |
| Backup/restore procedure | Vendor admin documentation. |

---

## 17. Research Gaps / Requested Additional Material

The current material is enough to draft a cautious architecture chapter, but not enough to produce a complete technical architecture reference. The following would materially improve accuracy:

| Needed Material | Why Needed |
|---|---|
| Axys installation/admin guide | Directory structure, deployment model, backup, multi-user setup, file locking. |
| Axys IMEX manual or object catalog | Object names, field names, delimiters, import/export syntax, error handling. |
| Axys Report Writer Pro / Replang Programmer’s Guide | REP syntax, field names, report object behavior, supported functions. |
| Sample Axys `rep` directory | Real report names and REP source patterns. |
| Sample IMEX exports | Field dictionaries for portfolios, transactions, securities, prices, holdings, performance, groups. |
| Version-specific release notes, especially Axys 3.7/3.8 | Confirm file conversion and changed file formats. |
| Axys data directory listing from a non-sensitive test/demo environment | Verify file names and module-owned storage. |
| APX architecture/admin documentation | Properly separate APX-specific SQL architecture details into Chapter 03. |
| Production observations from consultants/users | File locking, run-time behavior, batch workflows, common failure modes. |

---

## 18. Proposed Reference Architecture Diagram for Future Chapter

```text
                         +----------------------+
                         |      Axys Users      |
                         |  Ops / Perf / Reports|
                         +----------+-----------+
                                    |
                                    v
+-------------------+      +--------------------+      +------------------+
| Custodian / Market| ---> | Axys Application   | ---> | Standard Reports |
| Data Inputs       |      | Portfolio Accounting|     | Report Writer Pro|
+-------------------+      | Performance/Reports |     | Replang .REP     |
                           +---------+----------+      +--------+---------+
                                     |                          |
                                     v                          v
                           +--------------------+      +------------------+
                           | Proprietary /      |      | Excel / CSV /    |
                           | File-Oriented Data |      | Printed Reports  |
                           | Store              |      +------------------+
                           +---------+----------+
                                     |
                                     v
                           +--------------------+
                           | IMEX Import/Export |
                           | CSV / Tab / Fixed* |
                           +--------------------+

* Fixed-format support is reported for Axys IMEX. APX fixed-format generation is reported eliminated in APX v1.x-v4.x by a practitioner source; verify before documenting as universal behavior.
```

Confidence: Medium. The diagram is a research synthesis, not a vendor architecture diagram.

---

## 19. Reference Notes by Repository Chapter Cross-Link

| Repository Chapter | Relevant Findings from This Research |
|---|---|
| `Chapter_02_Axys_Architecture.md` | Axys file-oriented/proprietary architecture, Report Writer/Replang, IMEX, version conversion risk. |
| `Chapter_03_APX_Architecture.md` | SQL-based APX, centralized platform, browser UI historical statement, local/cloud delivery, Packager. |
| `Chapter_04_Security_Master.md` | `SECURITY.INF`, `TYPE.INF` are candidate research leads only. |
| `Chapter_05_Transactions.md` | `CLI` is a candidate transaction/client file lead only; exact transaction schema Unknown. |
| `Chapter_08_Pricing.md` | `PRI` is a candidate price/factor file lead only. |
| `Chapter_10_Performance.md` | `PRF` is a candidate performance-return file lead only. |
| `Chapter_11_Classifications.md` | `GRP` is a candidate group/color-group file lead only. |
| `Chapter_12_Imex.md` | IMEX is the safer supported integration path; object/field catalog Unknown. |
| `Chapter_13_Rep.md` | `.REP`, `.RPW`, Replang, Report Writer checksum/manual editing quirks. |
| `Chapter_14_Reports.md` | `AMAN.REP`, Assets Under Management, Report Writer Pro, Packager/APX. |
| `Chapter_15_Data_Dictionary.md` | Only a few tokens/fields supported; most field names Unknown. |

---

## 20. Unknowns Register

| ID | Unknown | Priority |
|---|---|---:|
| U-AXYS-ARCH-001 | Exact Axys physical file layout by version. | High |
| U-AXYS-ARCH-002 | Exact Axys data directory structure and naming conventions. | High |
| U-AXYS-ARCH-003 | File locking/concurrency behavior. | High |
| U-AXYS-ARCH-004 | Supported backup/restore procedure and whether hot backup is safe. | High |
| U-AXYS-ARCH-005 | IMEX object names and exact field dictionaries. | High |
| U-AXYS-ARCH-006 | REP/Replang complete field/function dictionary. | High |
| U-AXYS-ARCH-007 | Standard report catalog by Axys version. | High |
| U-AXYS-ARCH-008 | Stored-versus-recalculated performance behavior. | High |
| U-AXYS-ARCH-009 | Batch scheduling, command-line, macro automation details. | Medium |
| U-AXYS-ARCH-010 | Log locations and error-file naming for IMEX/report runs. | Medium |
| U-AXYS-ARCH-011 | Direct native-file write support boundary and vendor support policy. | High |
| U-AXYS-ARCH-012 | Whether current SS&C-hosted Axys configurations differ materially from legacy on-prem installs. | Medium |
| U-AXYS-ARCH-013 | Exact relationship among portfolio code, account number, custodian account number, and file name. | High |
| U-AXYS-ARCH-014 | Version-specific differences before/after Axys 3.8. | Medium |
| U-AXYS-ARCH-015 | Whether `CLI`, `PRI`, `INF`, `PRF`, `GRP` are exact native files, export files, or migration shorthand in each version/context. | High |
| U-AXYS-ARCH-016 | Whether current SS&C-hosted or managed Axys environments differ operationally from legacy on-prem Axys. | High |
| U-AXYS-ARCH-017 | Whether `imex32.exe`, `pospos32.exe`, and `REP32.exe` can be run unattended in each client environment. | High |
| U-AXYS-ARCH-018 | Whether IMEX imports are atomic by row, file, blotter, or batch. | High |
| U-AXYS-ARCH-019 | Whether Axys has firm-level scheduler/batch mechanisms beyond Windows scheduling and third-party tooling. | Medium |
| U-AXYS-ARCH-020 | Which log files prove successful completion of transaction, position, price, report, and performance processing. | High |
| U-AXYS-ARCH-021 | Whether standard Axys reports and custom REP reports use the same calculation path for the same values. | High |
| U-AXYS-ARCH-022 | Vendor support policy for direct native-file reads/writes in current Axys environments. | High |

---

## 21. Research Conclusion

The available evidence supports a cautious architecture chapter centered on these facts:

1. Axys is a portfolio accounting, performance measurement, and reporting system with Report Writer Pro and Replang/REP-based reporting.
2. Axys should be treated as proprietary-database in product positioning and file-oriented in practical integration/operations evidence unless better vendor documentation proves a more precise architecture.
3. IMEX and report exports are the best-supported integration mechanisms identified from public evidence.
4. Direct Axys native-file access is a known practitioner possibility but should be documented as risky and version-sensitive, not as a recommended default.
5. APX is architecturally distinct: public vendor and SEC sources support that APX uses a SQL-based, centralized platform and broader integrated CRM/reporting/packaging architecture.
6. Low-level Axys architecture details — file layout, locking, scheduling, IMEX object schema, REP field dictionary, and stored/recalculated performance rules — remain Unknown and require vendor manuals, installed-system samples, or production observations.

## 22. Deep IMEX Addendum Incorporated 2026-06-30

Source: `axys_imex_deep_research.md`.

Architecture-level additions:

| Topic | Addendum | Confidence |
|---|---|---:|
| `imex32.exe` | Axys Import/Export utility and primary public-facing import/export interface in CI evidence. | Verified for CI |
| `pospos32.exe` | Axys Post Positions utility used with position-post workflows. | Verified for CI |
| Folder labels | `$pathexe`, `$pathtrn`, `$pathcli`, `$pathinf`, `$pathpri`, and `$pathlog` are observed CI folder labels for executables, Trade Blotter, client files, information files, prices, and logs. | Verified for CI |
| IMEX boundary | IMEX is safer than direct native-file access, but public evidence does not expose one universal object/field catalog. | High Confidence / Unknown |
| Discovery requirement | Real integrations should discover object names, fields, templates, formats, and logs from the licensed installation and record Axys version. | Design guidance |
| REP boundary | REP/Replang/custom reports remain an architectural extraction path for report-shaped or performance tie-out values. | High Confidence |
