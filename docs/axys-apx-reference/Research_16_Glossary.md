# Research_16_Glossary.md

Research notes for `Chapter_16_Glossary.md`.

This file follows `AXYS_APX_REFERENCE_BLUEPRINT.md` Version 2.0. It is intended as research support for Chapter 16, not as the chapter text.

## Scope

This glossary research collects terms that are likely to appear across the Axys/APX reference repository. It emphasizes implementation-oriented definitions for:

- Axys behavior
- APX behavior
- IMEX
- REP / RepLang
- reports
- data model terminology
- performance terminology
- security master terminology
- holdings, transactions, pricing, cash, classifications, and corporate actions
- version differences and known quirks

## Confidence Key

| Classification | Meaning |
|---|---|
| Verified | Directly supported by supplied repository instructions, public vendor material, public practitioner material, or named documentation. |
| High Confidence | Strongly supported by public material and common Axys/APX usage, but not yet verified against vendor manuals or production exports. |
| Medium Confidence | Plausible and consistent with observed Axys/APX terminology, but should be confirmed with vendor documentation, sample IMEX exports, REP source, or production data. |
| Unknown | Not verified. Do not use as a factual statement in Chapter 16 without additional evidence. |

## Source Base Used

| Source | Type | Notes |
|---|---|---|
| AXYS_APX_REFERENCE_BLUEPRINT.md Version 2.0 | Supplied governing specification | Defines repository purpose, required confidence labels, structure, and prohibition against invented behavior. |
| SS&C Advent Axys product page | Public vendor source | Describes Axys as portfolio reporting/accounting software with performance measurement, reporting, multicurrency, fixed income, corporate actions, and Report Writer Pro capabilities. |
| SS&C Advent Portfolio Exchange product page | Public vendor source | Describes APX as an integrated portfolio and client management solution connecting front, middle, and back office on one platform. |
| SS&C Advent APX product brief page | Public vendor source | Describes APX as centralized book of record for portfolio management, performance measurement, accounting, reporting, composite management, performance analytics, and CRM. |
| Salentica / Black Diamond Data Broker article for Advent APX & Axys | Public integration documentation | Describes REP32, Advent standard reports, macros, data extraction, Axys/APX minimum versions for that connector, and use of PMS account number. |
| AdventGuru practitioner material | Public practitioner source | Describes Axys/APX reporting options, Replang, Report Writer Pro, APX SQL Server access options, IMEX, Stored Accounting Functions, Public Views, SSRS, REST API, and ETL patterns. |
| Public third-party APX/Axys service/provider pages | Public practitioner/market sources | Useful only for broad confirmation that Axys/APX operations, reconciliation, reporting, and performance workflows are supported by third parties. |

## Important Research Constraint

Field names, transaction codes, report names, and file layouts should not be finalized from this glossary research unless they are later verified against one of the following:

- vendor documentation
- IMEX object definitions
- actual IMEX export headers
- REP source files
- production report output
- client configuration exports
- consultant documentation with exact examples

---

# 1. Core Product Terms

| Term | Working Definition | Axys | APX | IMEX | REP | Confidence | Evidence / Notes |
|---|---|---|---|---|---|---|---|
| Advent | Legacy vendor name historically associated with Axys and APX. SS&C Advent branding is used in current product materials. | Applicable | Applicable | N/A | N/A | Verified | Current vendor pages use SS&C Advent branding for Axys and Advent Portfolio Exchange. |
| SS&C Advent | Current product/vendor branding for Axys and APX in public materials. | Applicable | Applicable | N/A | N/A | Verified | Public vendor pages identify the products under SS&C Advent. |
| Axys | SS&C Advent portfolio management, portfolio accounting, performance measurement, and reporting solution. | Primary product | Related product only | Supports integration/extract workflows, but exact object set requires verification. | Uses REP/Report Writer Pro reporting architecture; exact details require verification. | Verified | Vendor material states Axys automates portfolio reporting and accounting, includes performance measurement, reporting, multicurrency, and fixed income support. |
| APX | Advent Portfolio Exchange; integrated portfolio management, accounting, performance measurement, reporting, and client relationship management solution. | Related product only | Primary product | Supports IMEX and other APX integration options; exact object set requires verification. | Uses reporting architecture including REP/RepLang; exact details require verification. | Verified | Vendor material describes APX as an integrated portfolio and client management solution and centralized book of record. |
| Advent Portfolio Exchange | Full product name for APX. | N/A | Full product name | N/A | N/A | Verified | Public vendor source. |
| Advent Investment Suite | Product suite context in which Axys/APX are marketed. | Part of suite per vendor material. | Part of suite per vendor material. | N/A | N/A | Verified | Vendor pages describe Axys/APX as part of Advent Investment Suite. |
| Portfolio Accounting System / PMS | System of record for portfolio accounting data such as portfolios, holdings, transactions, cash, performance, and reporting. | Axys can serve this role. | APX can serve this role. | Extracts may use PMS account numbers. | Reports may be generated through REP32 in integration workflows. | High Confidence | Vendor and integration material use portfolio accounting / PMS context. |
| Book of Record | Authoritative system record for portfolio, accounting, performance, and related business data. | Public Axys material implies but does not use this term in the reviewed source. | APX is explicitly described as a centralized book of record. | N/A | N/A | Verified for APX; Medium Confidence for Axys | APX vendor product brief. |
| Front Office | Investment-facing personnel/workflows such as portfolio management and client-facing use. | Unknown specific Axys implementation. | APX is marketed as connecting front, middle, and back offices. | N/A | Reports/dashboards may serve front office users. | Verified for APX; Unknown for Axys | Public APX vendor material. |
| Middle Office | Operations/control workflows between investment decision-making and back-office accounting. | Unknown specific Axys implementation. | APX is marketed as connecting front, middle, and back offices. | N/A | N/A | Verified for APX; Unknown for Axys | Public APX vendor material. |
| Back Office | Accounting, reconciliation, reporting, and operations workflows. | Axys is used for accounting and reporting. | APX is used for accounting and reporting. | IMEX/extracts may support back-office integrations. | REP reports often support extraction and reporting. | High Confidence | Public vendor and integration material. |

---

# 2. Architecture and Platform Terms

| Term | Working Definition | Axys | APX | IMEX | REP | Confidence | Evidence / Notes |
|---|---|---|---|---|---|---|---|
| Proprietary Database | Vendor/private data store rather than a general-purpose relational database exposed to users. | Vendor material says Axys is used by family offices that prefer a proprietary database. | APX public material emphasizes centralized platform/book of record; underlying SQL Server access is referenced by practitioner material. | Unknown object mapping. | Reports access data through Advent reporting architecture. | Verified for Axys phrase; High Confidence for contrast with APX | Public Axys vendor page and practitioner material. |
| SQL Server Database | Microsoft SQL Server database used as underlying APX data infrastructure. | Not applicable based on public sources reviewed. | APX practitioner material states users can tap APX's underlying SQL Server database. | May coexist with IMEX. | May coexist with REP/SSRS. | High Confidence | AdventGuru practitioner source. |
| Genesis Platform | Platform/infrastructure referenced by SS&C in connection with Data Lens for APX. | Not applicable in reviewed sources. | APX Data Lens described as built on Genesis platform. | N/A | N/A | Verified | APX vendor page. |
| Data Lens for APX | Data integration, aggregation, visualization, dashboard/reporting capability associated with APX. | N/A | APX-related capability. | N/A | N/A | Verified | Public APX vendor page describes Data Lens for APX. |
| Client Tools | Locally installed Advent software components used by integrations or reporting. | Data Broker article requires Advent client tools for Axys/APX connector. | Same. | May be used for extraction workflows. | Includes REP32 reporting application per integration documentation. | Verified | Data Broker integration article. |
| REP32.exe | Advent reporting application/engine used by at least one connector to run reports/macros. | Used in Axys/APX extraction workflow in Data Broker article. | Same. | Could generate extract files indirectly through reports. | REP32 is explicitly tied to reporting and RepLang/macros in integration documentation. | Verified | Data Broker article. |
| 32-bit Windows Application | Software architecture attribute of the specific Data Broker Connector, not necessarily Axys/APX itself. | Connector can be installed for Axys environments. | Connector can be installed for APX environments. | Connector uses reports/macros for extracts. | Requires REP32 installed. | Verified | Data Broker article. |
| Local Deployment | On-premise/local deployment model. | Public connector article supports on-premise or AOS-hosted Axys/APX users. | Public vendor source states APX can be deployed locally or cloud-delivered. | N/A | N/A | Verified | Vendor and integration material. |
| Cloud-Delivered | Hosted/cloud delivery option. | AOS-hosted users referenced for connector; details Unknown. | APX public vendor source states cloud-delivered option. | N/A | N/A | Verified for APX; High Confidence for Axys AOS context | Vendor and integration material. |
| AOS-hosted | Advent Outsourcing Services or hosted Advent environment; exact expansion not verified in supplied material. | Mentioned in connector context. | Mentioned in connector context. | N/A | REP32/report execution may be hosted. | Medium Confidence | Data Broker article uses term but does not define expansion. |
| REST API | Modern APX integration option referenced in practitioner material. | Not identified for Axys in reviewed sources. | APX practitioner material references REST API as an option. | Alternative/complement to IMEX. | Alternative/complement to reporting extracts. | High Confidence | AdventGuru practitioner source. |
| Public Views | APX database/reporting access layer referenced in practitioner material. | Not applicable in reviewed sources. | Referenced as APX integration/reporting option. | Alternative/complement to IMEX. | Alternative/complement to REP. | High Confidence | AdventGuru practitioner source. |
| Stored Accounting Functions | APX data access/reporting functions referenced in practitioner material. | Not identified for Axys. | APX option referenced by practitioner source. | Alternative/complement to IMEX. | Alternative/complement to REP. | High Confidence | AdventGuru practitioner source. |
| SSRS | SQL Server Reporting Services; reporting option referenced for APX. | Not identified for Axys in reviewed sources. | APX option referenced by practitioner source. | Alternative/complement to IMEX. | Alternative/complement to REP. | High Confidence | AdventGuru practitioner source. |
| ETL | Extract, Transform, Load tools/processes used to populate a data warehouse from Axys/APX or related systems. | Used by Axys firms per practitioner source. | Used by APX firms per practitioner source. | IMEX may supply extracts for ETL. | REP reports may supply extracts for ETL. | High Confidence | AdventGuru practitioner material mentions ETL tools like xPort. |
| Data Warehouse | Firm-managed repository populated from Axys/APX extracts for analytics/reporting. | Practitioner source describes Axys/APX users populating warehouses. | Same. | Common extract source. | Common extract source. | High Confidence | AdventGuru practitioner source. |
| xPort | Third-party ETL/extract product referenced by practitioner material. | Used by some Axys users. | Used by some APX users. | May overlap with IMEX/export workflows. | May overlap with REP output. | High Confidence | AdventGuru practitioner source. |

---

# 3. Reporting Terms

| Term | Working Definition | Axys | APX | IMEX | REP | Confidence | Evidence / Notes |
|---|---|---|---|---|---|---|---|
| Report | Generated output from Axys/APX reporting system, usually for portfolio/accounting/performance/client reporting. | Axys has predefined and customizable reports. | APX has standard reports and automated report packaging. | Reports may be exported or used as extraction source. | Core output of REP/RepLang/report engine. | Verified | Public vendor and integration material. |
| Standard Report | Vendor-provided/predefined report. | Axys has hundreds of predefined reports per vendor material. | APX has vast library of standard reports per vendor material. | May be used for extracts. | REP32 can run standard reports in connector workflow. | Verified | Vendor and Data Broker material. |
| Custom Report | Report modified or built to meet firm-specific requirements. | Axys supports custom reports and Report Writer Pro. | APX supports flexible custom reporting. | May be used to produce custom extracts. | Often implemented via Report Writer Pro or RepLang. | Verified | Vendor and practitioner material. |
| Report Writer Pro | Advent report authoring/customization tool. | Public Axys vendor material says users can create reports with Axys Report Writer Pro. | Practitioner material says Axys and APX users can create reports using Report Writer Pro. | N/A | Related to report creation; exact internal format Unknown. | Verified for Axys; High Confidence for APX | Vendor and practitioner material. |
| REP | Common shorthand for Advent report file/report architecture. Exact formal definition should be verified. | Used in Axys reporting. | Used in APX reporting. | REP output can be an alternative extract source. | Primary glossary term for repository Chapter 13. | Medium Confidence | Public integration/practitioner material references reports, RepLang, and REP32; exact `.rep` file semantics require vendor/manual verification. |
| RepLang / Replang | Advent report scripting/language used to modify/create reports. | Practitioner source states Replang is part of Axys reporting architecture. | Practitioner source states Replang is part of APX reporting architecture; APX may add more keywords. | Can be used to produce text/CSV extracts through reports. | Core report programming language. | High Confidence | AdventGuru practitioner material. |
| REP32 | Reporting engine/application. | Used with Axys client tools in integration workflow. | Used with APX client tools in integration workflow. | May generate extract files from reports/macros. | Executes reports/macros per integration documentation. | Verified | Data Broker article. |
| Macro | Automated sequence used to run reports/extracts. | Used with Axys/APX connector according to Data Broker article. | Same. | May automate extract generation. | Used with REP32/RepLang reporting workflow. | Verified | Data Broker article. |
| Report Packaging | Process of combining/automating multiple reports into a client or management reporting package. | Axys can place multiple reports/graphs/objects on one page and automate repeated use. | APX vendor source references automated report packaging. | N/A | Often uses report architecture. | Verified | Public vendor sources. |
| Compound Report | Report package containing multiple report objects; exact Axys/APX implementation should be verified. | Likely supported by Axys based on public custom report examples and vendor report packaging. | Likely supported by APX. | N/A | REP/reporting architecture. | Medium Confidence | Public examples mention compound reports but not authoritative vendor docs. |
| Report Object | Component within a report package; exact technical definition Unknown. | Unknown. | Unknown. | N/A | Unknown. | Unknown | Requires sample REP source or vendor documentation. |
| Report Menu | User/menu configuration for reports. | Known from practitioner and integration contexts, but exact configuration storage Unknown. | Unknown. | N/A | Used to expose reports to users. | Medium Confidence | Public AIA snippets mention custom report menu, but exact details need documentation. |
| Output to Excel | Export of report results to Microsoft Excel. | Practitioner material says Axys/APX users can export reports directly to Excel. | Same. | Alternative to IMEX. | Report output method. | High Confidence | AdventGuru practitioner source. |
| CSV Report Output | Text/CSV output generated from reports. | Practitioner material says Report Writer Pro can be changed to CSV format. | Same. | Alternative/complement to IMEX. | Report output method. | High Confidence | AdventGuru practitioner source. |
| Drill-Down | Ability to access detailed data from reports. | Axys vendor material says drill-down access to detailed data directly from reports. | Not verified in reviewed APX source. | N/A | Report interface behavior. | Verified for Axys; Unknown for APX | Axys vendor page. |
| Graphics | Charts/visual report elements such as pie charts, line graphs, bar charts. | Axys supports high-impact graphics in reports. | Unknown in reviewed APX source. | N/A | Report output. | Verified for Axys; Unknown for APX | Axys vendor page. |

---

# 4. IMEX and Integration Terms

| Term | Working Definition | Axys | APX | IMEX | REP | Confidence | Evidence / Notes |
|---|---|---|---|---|---|---|---|
| IMEX | Import/export mechanism used in Advent environments. Exact object list, command syntax, and file schemas require verification. | Used by Axys users; exact behavior Unknown. | Practitioner material says APX import/export methods like IMEX remain efficient/reliable for certain data elements. | Core term. | Alternative/complement to REP extracts. | High Confidence for existence; Unknown for details | AdventGuru practitioner source. |
| Import | Loading data into Axys/APX from external source. | Supported generally by IMEX/import workflows; details Unknown. | Supported generally by IMEX/import workflows; details Unknown. | Core operation. | N/A | Medium Confidence | Requires vendor IMEX docs or sample workflows. |
| Export | Extracting data from Axys/APX to external files/systems. | Supported via reports, IMEX, ETL. | Supported via reports, IMEX, ETL/API. | Core operation. | Reports may export data. | High Confidence | Public integration/practitioner sources. |
| Extract | Generated file or dataset output from Axys/APX. | Can be produced via REP reports/macros or IMEX. | Same; APX also has database/API options. | Core output concept. | Common report output use. | High Confidence | Data Broker and AdventGuru material. |
| Data Feed | Recurring transfer of financial/account data from Axys/APX to another platform. | Data Broker connector pushes daily feed for Axys/APX users. | Same. | May use report-generated files or IMEX. | REP32 generates extract for connector workflow. | Verified | Data Broker article. |
| Daily Feed | Scheduled daily financial data extraction. | Connector supports daily feed. | Connector supports daily feed. | Unknown if IMEX-specific. | REP32/report macros in connector. | Verified | Data Broker article. |
| PMS Account Number | Account identifier received from the Portfolio Management System integration. | Relevant to Axys integrations. | Relevant to APX integrations. | Likely field in integration outputs; exact IMEX field Unknown. | May appear in report output. | Verified as integration concept; Unknown as exact field | Data Broker article distinguishes PMS account number from custodian account number. |
| Custodian Account Number | Account identifier from custodian integration. | May differ from PMS account number. | Same. | May be mapped in integrations. | May appear in reports. | Verified as integration concept; Unknown as Axys/APX native field | Data Broker article. |
| Mapping | Translation or association between source fields/accounts and target CRM/integration fields. | Required by integration workflows. | Same. | Extract mappings may be required. | Report output may be mapped. | High Confidence | Data Broker article references credentials and mapping information. |
| Translation File | File used by integration tooling to map/normalize values. Exact Advent-native status Unknown. | Unknown. | Unknown. | Could be external ETL construct. | N/A | Medium Confidence | Public AIA snippet references translation files, but not Advent-native semantics. |
| Source-data Folder | Folder where normalized custodian/source files are stored in AIA workflow; not necessarily Advent-native. | External integration concept. | External integration concept. | N/A | N/A | Medium Confidence | Public AIA snippet; should not be treated as Axys/APX native without verification. |
| Extract Folder | File system folder used for report or integration outputs. | Used in external connector/report workflows. | Used in external connector/report workflows. | IMEX outputs may be written to folders; exact rules Unknown. | REP outputs may be written to folders. | Medium Confidence | Integration snippets mention folders; exact Advent behavior requires verification. |
| Unattended Run | Scheduled/automated execution without user interaction. | Connector can be scheduled to run unattended. | Same. | Possible for extracts. | REP32/macros can be used in scheduled workflow. | Verified for connector; Medium Confidence generally | Data Broker article. |
| API | Programmatic interface. | Not verified for Axys. | APX REST API referenced by practitioner material. | Alternative to IMEX. | Alternative to REP. | High Confidence for APX; Unknown for Axys | AdventGuru practitioner source. |
| Financial Data Extract | Dataset from portfolio/accounting system used by downstream platforms. | Data Broker connector uses standard reports/macros to generate extract. | Same. | Could be IMEX or report output. | Verified via REP32/report workflow. | Verified | Data Broker article. |

---

# 5. Portfolio, Account, and Entity Terms

| Term | Working Definition | Axys | APX | IMEX | REP | Confidence | Evidence / Notes |
|---|---|---|---|---|---|---|---|
| Portfolio | Accounting/reporting entity containing holdings, transactions, cash, and performance. | Axys manages and reports portfolios. | APX tracks portfolios/strategies and portfolio data. | Likely exported by portfolio/account objects; exact object names Unknown. | Appears in portfolio reports. | Verified generally; Unknown for exact fields | Vendor material. |
| Account | Often used interchangeably with portfolio in integrations; exact distinction varies by implementation. | Unknown native semantics. | Unknown native semantics. | Account identifiers may appear in extracts. | Account identifiers may appear in reports. | Medium Confidence | Data Broker distinguishes PMS/custodian account number; exact Axys/APX entity model needs verification. |
| Client | Person/entity served by investment firm. | Axys supports client reporting; exact CRM model Unknown. | APX includes client relationship management. | May be exported. | May appear in reports. | Verified generally; Unknown for exact fields | Vendor material. |
| Relationship | Client/household/business relationship construct. | Unknown. | APX is a client relationship management solution; exact relationship table Unknown. | Unknown. | Unknown. | Medium Confidence | Vendor APX page supports CRM concept but not data structure. |
| Prospect | Potential client. | Unknown. | APX public page references prospect data. | Unknown. | Unknown. | Verified for APX concept; Unknown for implementation | APX vendor page. |
| Household | Grouping of accounts/clients, often used in wealth management. | Unknown. | Unknown. | Unknown. | Unknown. | Unknown | Common industry term but not verified in supplied/public sources. |
| Portfolio Group | Grouping of portfolios for reporting/composite/management. | Axys can manage/report portfolios grouped by manager, asset class, objective, or any category. | APX likely supports grouping, but exact behavior Unknown. | May be used in extracts. | Appears in grouped reports. | Verified for Axys; Medium Confidence for APX | Axys vendor page. |
| Composite | Group of portfolios used for composite performance reporting, often GIPS-related. | Axys has composite management features and reporting to track portfolio entry/exit dates. | APX has composite management support for GIPS compliance. | Unknown. | Composite reports likely exist; exact names Unknown. | Verified | Vendor material. |
| Portfolio Entry Date | Date a portfolio enters a composite. | Axys vendor material says composite management tracks entry/exit dates. | Likely relevant to APX composite support; exact fields Unknown. | Unknown. | Unknown. | Verified for Axys concept; Unknown exact fields | Axys vendor page. |
| Portfolio Exit Date | Date a portfolio exits a composite. | Axys vendor material says composite management tracks entry/exit dates. | Likely relevant to APX composite support; exact fields Unknown. | Unknown. | Unknown. | Verified for Axys concept; Unknown exact fields | Axys vendor page. |
| Manager | Grouping/reporting attribute referenced by Axys public material. | Axys can group portfolios by manager. | Unknown exact APX behavior. | May be a portfolio attribute; exact field Unknown. | May appear in reports. | Verified for Axys concept; Unknown field | Axys vendor page. |
| Investment Objective | Portfolio grouping/category attribute referenced by Axys public material. | Axys can group by investment objective. | Unknown exact APX behavior. | Unknown. | Unknown. | Verified for Axys concept; Unknown field | Axys vendor page. |

---

# 6. Security Master and Instrument Terms

| Term | Working Definition | Axys | APX | IMEX | REP | Confidence | Evidence / Notes |
|---|---|---|---|---|---|---|---|
| Security Master | Reference data for securities/instruments. Exact Axys/APX file/table and field names require verification. | Axys supports broad security types and fixed income characteristics. | APX supports broad asset types. | Security master exports likely exist, but object names/fields Unknown. | Security reference data appears in reports. | High Confidence generally; Unknown for fields | Vendor material supports security/instrument management; field dictionary requires samples/docs. |
| Security | Tradable/reportable instrument. | Axys supports cash, equities, fixed income, and other types. | APX supports equities, fixed income, mutual funds, FX, derivatives, alternatives. | Likely central extract entity. | Report entity. | Verified generally | Vendor material. |
| Instrument | Alternative term for security/asset type. | Axys product page uses security types. | APX product brief references instrument/asset coverage. | Unknown. | Unknown. | Verified generally | Vendor material. |
| Cash Type | Cash or money-market-like security/category. | Axys supports money market and other cash types. | APX supports cash data and broad instruments. | Unknown. | Unknown. | Verified for Axys; High Confidence for APX | Vendor material. |
| Equity | Common stock or equity instrument. | Axys supports equities. | APX supports equities. | Unknown. | Unknown. | Verified | Vendor material. |
| Fixed Income | Bonds and related income instruments. | Axys has substantial fixed income capabilities. | APX supports fixed income. | Unknown. | Unknown. | Verified | Vendor material. |
| Municipal Bond | Fixed income security type. | Axys supports variable rate municipal bonds. | Not specifically verified. | Unknown. | Unknown. | Verified for Axys; Unknown for APX specific subtype | Axys vendor page. |
| Corporate Bond | Fixed income security type. | Axys supports corporate bonds. | APX supports fixed income generally. | Unknown. | Unknown. | Verified for Axys; High Confidence for APX | Vendor material. |
| Government Bond | Fixed income security type. | Axys supports government bonds. | APX supports fixed income generally. | Unknown. | Unknown. | Verified for Axys; High Confidence for APX | Vendor material. |
| Mortgage-Backed Security / MBS | Fixed income security backed by mortgage collateral. | Axys supports MBS. | APX support not specifically verified. | Unknown. | Unknown. | Verified for Axys; Unknown for APX subtype | Axys vendor page. |
| Step-Up Bond | Bond whose coupon rate changes according to terms. | Axys supports step-up bonds. | APX support not specifically verified. | Unknown. | Unknown. | Verified for Axys; Unknown for APX subtype | Axys vendor page. |
| Mutual Fund | Pooled investment vehicle. | Not specifically verified in reviewed Axys material. | APX supports mutual funds. | Unknown. | Unknown. | Verified for APX; Unknown for Axys in reviewed source | APX vendor product brief. |
| FX | Foreign exchange/currency instrument or transaction context. | Axys has multicurrency capabilities. | APX supports FX and settlement in any currency. | Unknown. | Unknown. | Verified for APX; High Confidence for Axys multicurrency context | Vendor material. |
| Derivative | Instrument whose value derives from underlying assets. | Not specifically verified in reviewed Axys material. | APX supports derivatives. | Unknown. | Unknown. | Verified for APX; Unknown for Axys in reviewed source | APX vendor product brief. |
| Alternative Investment | Non-traditional asset such as private equity. | Not specifically verified in reviewed Axys material. | APX supports alternative investments including private equity. | Unknown. | Unknown. | Verified for APX; Unknown for Axys | APX vendor product brief. |
| Private Equity | Alternative investment type. | Unknown. | APX product brief names private equity. | Unknown. | Unknown. | Verified for APX | APX vendor product brief. |
| Security Identifier | Identifier used to link transactions, holdings, prices, and performance to a security. Exact fields Unknown. | Unknown field names. | Unknown field names. | Unknown object/header names. | Unknown report fields. | Unknown | Requires security master export/report samples. |
| Ticker | Market symbol. | Unknown. | Unknown. | Unknown. | Unknown. | Unknown | Common industry field, but not verified in supplied/public Axys/APX sources. |
| CUSIP | North American security identifier. | Unknown. | Unknown. | Unknown. | Unknown. | Unknown | Common industry field, but not verified here. |
| ISIN | International security identifier. | Unknown. | Unknown. | Unknown. | Unknown. | Unknown | Common industry field, but not verified here. |
| SEDOL | UK/foreign security identifier. | Unknown. | Unknown. | Unknown. | Unknown. | Unknown | Common industry field, but not verified here. |

---

# 7. Transaction, Holdings, Cash, and Pricing Terms

| Term | Working Definition | Axys | APX | IMEX | REP | Confidence | Evidence / Notes |
|---|---|---|---|---|---|---|---|
| Transaction | Accounting event such as trade, income, cash movement, or corporate action. | Axys tracks transactions according to vendor material. | APX tracks transactions according to vendor material. | Transaction exports likely exist; exact object/name/fields Unknown. | Transaction reports likely exist; exact names Unknown. | Verified generally; Unknown for transaction codes | Vendor material. |
| Trade | Buy/sell or market transaction. | Axys reconciles trade information. | APX tracks transactions; exact trade behavior Unknown. | Unknown. | Unknown. | Verified for Axys trade information; High Confidence for APX | Vendor/integration material. |
| Buy | Purchase transaction type. | Unknown transaction code/field. | Unknown transaction code/field. | Unknown. | Unknown. | Unknown | Do not invent Axys/APX transaction codes. |
| Sell | Sale transaction type. | Unknown transaction code/field. | Unknown transaction code/field. | Unknown. | Unknown. | Unknown | Do not invent Axys/APX transaction codes. |
| Settlement Data | Data describing settlement of trades/cash. | Axys vendor material mentions automated reconciliation of settlement data. | APX public material references settlement in any currency. | Unknown. | Unknown. | Verified generally | Vendor material. |
| Trade Date Accounting | Accounting treatment recognizing trades on trade date. | Axys can select trade date or settlement date accounting. | Unknown in reviewed APX sources. | Unknown. | Unknown. | Verified for Axys; Unknown for APX | Axys vendor page. |
| Settlement Date Accounting | Accounting treatment recognizing trades on settlement date. | Axys can select trade date or settlement date accounting. | Unknown in reviewed APX sources. | Unknown. | Unknown. | Verified for Axys; Unknown for APX | Axys vendor page. |
| Tax Lot Accounting | Cost/accounting method tracking lots. | Axys can select tax lot accounting. | Unknown in reviewed APX sources. | Unknown. | Unknown. | Verified for Axys; Unknown for APX | Axys vendor page. |
| Average Cost Accounting | Cost method based on average cost. | Axys can select average cost accounting. | Unknown in reviewed APX sources. | Unknown. | Unknown. | Verified for Axys; Unknown for APX | Axys vendor page. |
| Position | Holding/quantity/value of a security in a portfolio. | Axys reconciles positions and reports holdings/performance. | APX gives insight into positions. | Position/holding extracts likely; exact object names Unknown. | Position reports likely; exact names Unknown. | Verified generally; Unknown for fields | Vendor material. |
| Holding | Security/cash position as of a date or period. | Axys reports portfolios and securities; exact holdings files Unknown. | APX tracks holdings. | Holdings extracts likely; exact object names Unknown. | Holdings reports likely; exact names Unknown. | Verified generally; Unknown for fields | Vendor material. |
| Cash | Cash balance/instrument/category. | Axys supports cash types and multicurrency. | APX provides insight into cash and supports settlement in any currency. | Cash extracts likely; exact object names Unknown. | Cash reports likely; exact names Unknown. | Verified generally; Unknown for fields | Vendor material. |
| Price | Valuation price for a security. | Unknown storage/field definitions. | Unknown storage/field definitions. | Price extracts likely; exact object names Unknown. | Price reports likely; exact names Unknown. | Medium Confidence generally; Unknown for fields | Pricing is implicit in accounting/performance, but specific Axys/APX price model not verified. |
| Closing Price | Market close price. | Unknown. | Unknown. | Unknown. | Unknown. | Unknown | Requires price export/report samples. |
| Market Value | Quantity times price plus valuation adjustments; exact Axys/APX definition Unknown. | Unknown exact calculation. | Unknown exact calculation. | Unknown. | Unknown. | Medium Confidence | Common accounting concept, but exact product calculation requires verification. |
| Accrued Interest | Interest earned but not yet paid on fixed income. | Axys tracks fixed income characteristics and amortization/accretion; exact accrual behavior Unknown. | Unknown exact behavior. | Unknown. | Unknown. | Medium Confidence | Common fixed income accounting concept; not explicitly verified in public source. |
| Amortization | Fixed income accounting adjustment over time. | Axys tracks amortization. | Unknown exact APX behavior. | Unknown. | Unknown. | Verified for Axys concept | Axys vendor page. |
| Accretion | Fixed income accounting adjustment over time. | Axys tracks accretion. | Unknown exact APX behavior. | Unknown. | Unknown. | Verified for Axys concept | Axys vendor page. |
| Coupon | Bond interest payment/term. | Axys tracks odd coupon dates and fixed income characteristics. | Unknown exact APX behavior. | Unknown. | Unknown. | High Confidence | Axys vendor page mentions odd coupon dates; exact coupon processing requires docs. |
| Odd Coupon Date | Non-standard coupon date. | Axys tracks odd coupon dates. | Unknown. | Unknown. | Unknown. | Verified for Axys | Axys vendor page. |

---

# 8. Performance, Attribution, Benchmark, and Composite Terms

| Term | Working Definition | Axys | APX | IMEX | REP | Confidence | Evidence / Notes |
|---|---|---|---|---|---|---|---|
| Performance Measurement | Calculation/reporting of investment returns. | Axys supports comprehensive performance measurement. | APX includes performance measurement/performance analytics. | Performance extracts may exist; exact object names Unknown. | Performance reports exist but exact names Unknown. | Verified generally | Vendor material. |
| Time-Weighted Return / TWR | Return measure that reduces impact of external cash flows. | Axys can calculate time-weighted returns. | APX performance measurement likely includes TWR, but not directly verified in reviewed sources. | Unknown. | Unknown. | Verified for Axys; Medium Confidence for APX | Axys vendor page explicitly names time-weighted returns. |
| Internal Rate of Return / IRR | Money-weighted return measure. | Axys can calculate internal rates of return. | Unknown in reviewed APX sources. | Unknown. | Unknown. | Verified for Axys; Unknown for APX | Axys vendor page. |
| Before Fees | Return calculated before management fees. | Axys can calculate returns before management fees. | Unknown. | Unknown. | Unknown. | Verified for Axys | Axys vendor page. |
| After Fees | Return calculated after management fees. | Axys can calculate returns after management fees. | Unknown. | Unknown. | Unknown. | Verified for Axys | Axys vendor page. |
| Performance History | Stored or maintained historical performance records; exact storage/recalculation rules Unknown. | Axys vendor material says users can update performance history for significant contributions/withdrawals. | Unknown exact APX storage/recalculation rules. | Performance history exports may exist; exact object names Unknown. | Performance reports may use stored or calculated values; behavior Unknown. | Verified concept for Axys; Unknown implementation | Axys vendor page. |
| Significant Contribution / Withdrawal | External cash flow large enough to affect performance calculation treatment. | Axys vendor material references updating performance history for significant contributions/withdrawals. | Unknown. | Unknown. | Unknown. | Verified for Axys concept; Unknown threshold/rules | Axys vendor page. |
| Benchmark | Index or blended benchmark used for performance comparison. | Axys can create blended benchmarks and compare against indices/synthetic indices. | APX performance analytics likely supports benchmarks, but not verified in reviewed sources. | Unknown. | Unknown. | Verified for Axys; Medium Confidence for APX | Axys vendor page. |
| Blended Benchmark | Benchmark composed of multiple indices/components. | Axys can create blended benchmarks and track components historically. | Unknown. | Unknown. | Unknown. | Verified for Axys | Axys vendor page. |
| Synthetic Index | User/system-created index used for comparison. | Axys can compare performance against indices including synthetic indices. | Unknown. | Unknown. | Unknown. | Verified for Axys | Axys vendor page. |
| Index | External or internal benchmark series. | Axys can compare performance against indices. | Unknown. | Unknown. | Unknown. | Verified for Axys; Medium Confidence for APX | Axys vendor page. |
| Composite Management | Managing composites and portfolio membership for reporting/GIPS. | Axys has composite management features. | APX has composite management support for GIPS compliance. | Unknown. | Composite reports likely; exact names Unknown. | Verified | Vendor material. |
| GIPS | Global Investment Performance Standards. | Axys facilitates GIPS compliance. | APX has composite management support for GIPS compliance. | Unknown. | GIPS/composite reports likely; exact names Unknown. | Verified | Vendor material. |
| Performance Analytics | Analytics around performance measurement and attribution. | Axys performance measurement; exact analytics module Unknown. | APX public material names performance analytics. | Unknown. | Unknown. | Verified for APX; High Confidence for Axys broad concept | Vendor material. |
| Attribution | Explanation of performance sources. | Not verified in reviewed Axys source. | APX page navigation includes Performance, Reporting & Attribution; exact APX attribution functionality Unknown. | Unknown. | Unknown. | Medium Confidence | Public navigation/source material not sufficient for detailed behavior. |
| Contribution | Performance contribution by security/classification. | Unknown. | Unknown. | Unknown. | Unknown. | Unknown | Requires performance report samples or docs. |
| Classification-Level Performance | Performance displayed by asset class, sector, country, region, etc. | Axys can display performance by asset classes, sectors, countries, or regions. | Unknown exact APX behavior. | Unknown. | Likely reports exist; exact names Unknown. | Verified for Axys; Unknown for APX | Axys vendor page. |
| Security Performance | Performance by individual security. | Unknown exact Axys report/export behavior. | Unknown exact APX report/export behavior. | Unknown. | Unknown. | Unknown | Requires REP/IMEX samples. |
| Portfolio Performance | Performance at portfolio level. | Axys calculates/display portfolio performance. | APX tracks performance. | Unknown. | Unknown. | Verified generally | Vendor material. |

---

# 9. Classification and Reporting Dimension Terms

| Term | Working Definition | Axys | APX | IMEX | REP | Confidence | Evidence / Notes |
|---|---|---|---|---|---|---|---|
| Classification | Dimension used to group securities/portfolios for reporting/performance. | Axys can report by asset class, sector, country, region, manager, objective, and categories. | APX likely supports classification/dimensions, but exact behavior Unknown. | Classification exports likely; exact object names Unknown. | Classification reports likely; exact names Unknown. | Verified for Axys concept; Unknown for fields | Axys vendor page. |
| Asset Class | Classification/dimension such as equity, fixed income, cash. | Axys can group/report by asset class. | APX supports multi-asset-class coverage. | Unknown. | Unknown. | Verified | Vendor material. |
| Sector | Security classification such as economic sector. | Axys can display performance by sectors. | Unknown. | Unknown. | Unknown. | Verified for Axys | Axys vendor page. |
| Country | Geographic/security classification. | Axys can display performance by countries. | Unknown. | Unknown. | Unknown. | Verified for Axys | Axys vendor page. |
| Region | Geographic classification. | Axys can display performance by regions. | Unknown. | Unknown. | Unknown. | Verified for Axys | Axys vendor page. |
| Category | Flexible grouping attribute. | Axys can group portfolios by “any category you wish.” | Unknown. | Unknown. | Unknown. | Verified for Axys; Unknown exact configuration | Axys vendor page. |
| Rating | Fixed income credit/quality characteristic. | Axys tracks ratings. | Unknown exact APX behavior. | Unknown. | Unknown. | Verified for Axys | Axys vendor page. |
| Tax Status | Fixed income/security tax attribute. | Axys tracks tax status. | Unknown. | Unknown. | Unknown. | Verified for Axys | Axys vendor page. |
| Yield Method | Fixed income calculation attribute. | Axys tracks yield method. | Unknown. | Unknown. | Unknown. | Verified for Axys | Axys vendor page. |
| Duration | Fixed income risk/interest-rate sensitivity measure. | Axys tracks duration. | Unknown. | Unknown. | Unknown. | Verified for Axys | Axys vendor page. |

---

# 10. Corporate Actions, Income, and Reconciliation Terms

| Term | Working Definition | Axys | APX | IMEX | REP | Confidence | Evidence / Notes |
|---|---|---|---|---|---|---|---|
| Corporate Action | Event affecting a security, such as split, dividend, merger, or similar. | Axys has integrated corporate actions processing. | APX likely supports corporate actions, but not verified in reviewed sources. | Unknown. | Unknown. | Verified for Axys; Unknown for APX | Axys vendor page. |
| Dividend | Income distribution from a security. | Unknown exact storage/transaction/report behavior. | Unknown exact behavior. | Unknown. | Unknown. | Unknown | Requires transaction/corporate action documentation. |
| Split | Corporate action changing shares outstanding/security quantity basis. | Unknown exact storage/transaction/report behavior. | Unknown exact behavior. | Unknown. | Unknown. | Unknown | Requires corporate action documentation. |
| Interest | Fixed income/cash income. | Axys fixed income support implies interest handling, but exact behavior Unknown. | Unknown exact behavior. | Unknown. | Unknown. | Medium Confidence | Vendor source supports fixed income but not transaction specifics. |
| Withholding Tax | Tax withheld on income, especially international. | Axys can automatically calculate international withholding tax. | Unknown. | Unknown. | Unknown. | Verified for Axys | Axys vendor page. |
| Reconciliation | Process of comparing/matching internal accounting data to external trade, settlement, transaction, or position data. | Axys vendor material references automated reconciliation of trade information, settlement data, transactions, and positions. | APX likely supports reconciliation workflows, but not verified in reviewed source. | Extracts may support reconciliation. | Reports/macros may support reconciliation extracts. | Verified for Axys; Medium Confidence for APX | Axys vendor page and third-party service material. |
| Trade Information Reconciliation | Reconciliation of trade data. | Axys supports automated reconciliation of trade information. | Unknown exact APX behavior. | Unknown. | Unknown. | Verified for Axys | Axys vendor page. |
| Position Reconciliation | Reconciliation of positions/holdings. | Axys supports automated reconciliation of positions. | Unknown exact APX behavior. | Unknown. | Unknown. | Verified for Axys | Axys vendor page. |
| Transaction Reconciliation | Reconciliation of transactions. | Axys supports automated reconciliation of transactions. | Unknown exact APX behavior. | Unknown. | Unknown. | Verified for Axys | Axys vendor page. |

---

# 11. Common Field Name Research Targets

The following are glossary candidates, not verified Axys/APX field names. They should be used as research targets when reviewing IMEX exports, REP output, and data dictionaries.

| Candidate Term | Generic Meaning | Axys | APX | IMEX | REP | Confidence | Research Needed |
|---|---|---|---|---|---|---|---|
| Portfolio ID | Unique identifier for a portfolio/account. | Unknown exact field name. | Unknown exact field name. | Unknown. | Unknown. | Unknown | Verify export/report headers. |
| Portfolio Code | Short/account code for a portfolio. | Unknown. | Unknown. | Unknown. | Unknown. | Unknown | Verify export/report headers. |
| Account Number | Account number; may refer to PMS or custodian number depending on integration. | Integration concept verified. | Integration concept verified. | Unknown exact field. | Unknown exact field. | Medium Confidence | Compare PMS account number vs custodian account number in samples. |
| Security ID | Internal security identifier. | Unknown. | Unknown. | Unknown. | Unknown. | Unknown | Verify security master exports. |
| Symbol | Ticker-like display identifier. | Unknown. | Unknown. | Unknown. | Unknown. | Unknown | Verify security master/report samples. |
| CUSIP | External identifier. | Unknown. | Unknown. | Unknown. | Unknown. | Unknown | Verify security master samples. |
| ISIN | External identifier. | Unknown. | Unknown. | Unknown. | Unknown. | Unknown | Verify security master samples. |
| Description | Security/account text description. | Unknown. | Unknown. | Unknown. | Unknown. | Unknown | Verify headers. |
| Transaction Code | Code identifying transaction type. | Unknown. | Unknown. | Unknown. | Unknown. | Unknown | Requires transaction code table/vendor docs. |
| Trade Date | Date transaction was executed. | Axys supports trade-date accounting concept. | Unknown. | Unknown. | Unknown. | Medium Confidence | Verify transaction exports. |
| Settlement Date | Date transaction settles. | Axys supports settlement-date accounting concept. | APX supports settlement in any currency, but field Unknown. | Unknown. | Unknown. | Medium Confidence | Verify transaction exports. |
| Quantity | Number of units/shares/par. | Unknown. | Unknown. | Unknown. | Unknown. | Unknown | Verify holdings/transaction exports. |
| Price | Security transaction/valuation price. | Unknown. | Unknown. | Unknown. | Unknown. | Unknown | Verify pricing/transaction exports. |
| Market Value | Valuation amount. | Unknown. | Unknown. | Unknown. | Unknown. | Unknown | Verify holdings/performance reports. |
| Cost | Accounting cost basis. | Axys supports tax lot/average cost accounting. | Unknown. | Unknown. | Unknown. | Medium Confidence | Verify holdings/transactions. |
| Accrued Interest | Fixed income accrued interest. | Axys fixed income context supports concept; field Unknown. | Unknown. | Unknown. | Unknown. | Medium Confidence | Verify fixed income reports/exports. |
| Cash Amount | Cash movement amount. | Unknown. | Unknown. | Unknown. | Unknown. | Unknown | Verify transaction/cash exports. |
| Local Currency | Currency of security/transaction. | Axys/APX have multicurrency capabilities. | APX settlement in any currency. | Unknown. | Unknown. | Medium Confidence | Verify currency fields. |
| Base Currency | Reporting/accounting base currency. | Axys can restate reports in any currency. | APX multi-currency coverage. | Unknown. | Unknown. | Medium Confidence | Verify reports/exports. |
| Return | Performance return. | Axys calculates TWR/IRR. | APX performance measurement. | Unknown. | Unknown. | Medium Confidence | Verify performance export/report headers. |
| Weight | Portfolio/security/classification weight. | Unknown. | Unknown. | Unknown. | Unknown. | Unknown | Verify performance/holdings reports. |
| Benchmark Return | Benchmark performance. | Axys benchmark comparisons verified. | Unknown. | Unknown. | Unknown. | Medium Confidence | Verify performance reports. |

---

# 12. Report Name Research Targets

The following are possible report categories, not verified report names. Do not convert these into exact report names without supporting REP/report menu evidence.

| Candidate Report Category | Working Meaning | Axys | APX | Confidence | Evidence / Research Needed |
|---|---|---|---|---|
| Holdings Report | Report showing positions/holdings as of a date. | Likely exists. | Likely exists. | Medium Confidence | Vendor material supports holdings/positions; exact report names Unknown. |
| Transaction Report | Report listing transactions. | Likely exists. | Likely exists. | Medium Confidence | Vendor material supports transactions; exact report names Unknown. |
| Performance Report | Report showing portfolio/classification/security performance. | Verified category for Axys; likely for APX. | Verified broad category for APX. | High Confidence | Vendor material supports performance reporting; exact names Unknown. |
| Composite Report | Report showing composite performance/membership. | Likely exists. | Likely exists. | Medium Confidence | Composite management support verified; report names Unknown. |
| Security Master Report | Report/extract of security reference data. | Unknown. | Unknown. | Unknown | Requires sample report or IMEX object list. |
| Pricing Report | Report/extract of security prices. | Unknown. | Unknown. | Unknown | Requires sample report or export. |
| Cash Report | Report showing cash balances/activity. | Likely exists. | Likely exists. | Medium Confidence | Vendor material supports cash; exact names Unknown. |
| Reconciliation Report | Report supporting reconciliation of trades/positions/transactions. | High Confidence for Axys. | Medium Confidence for APX. | Medium Confidence | Axys reconciliation capability verified; exact names Unknown. |
| GIPS Report | Report supporting GIPS/composite compliance. | Likely exists. | Likely exists. | Medium Confidence | GIPS/composite support verified; exact names Unknown. |
| Client Statement / Client Report | Client-facing report package. | Likely exists. | Likely exists. | Medium Confidence | Reporting/product positioning supports client reporting; exact names Unknown. |

---

# 13. Version and Environment Terms

| Term | Working Definition | Axys | APX | IMEX | REP | Confidence | Evidence / Notes |
|---|---|---|---|---|---|---|---|
| Axys 3.8.6 | Minimum supported Axys version for one public Data Broker connector. | Specific connector support version. | N/A | Not necessarily IMEX minimum. | REP32/client tools needed. | Verified for connector only | Data Broker article lists Advent Axys 3.8.6 as minimum supported version for that connector. |
| APX 15.2 / 16.1 / 16.2 / 17.1 | APX versions tested/supported by one public Data Broker connector. | N/A | Specific connector support versions. | Not necessarily APX product minimum. | REP32/client tools needed. | Verified for connector only | Data Broker article lists these versions. |
| Current Version | Current vendor-supported product version. | Unknown. | Unknown. | Unknown. | Unknown. | Unknown | Requires SS&C release notes/customer portal. |
| Version Difference | Product behavior difference across releases. | Unknown unless documented. | Unknown unless documented. | Unknown. | Unknown. | Unknown | Requires release notes or verified production comparisons. |
| Replang Keyword Set | Available language keywords/functions. | Practitioner source says Axys has roughly 100 Replang keywords. | Practitioner source says current APX versions add another 100-plus keywords. | N/A | Core language detail. | Medium Confidence | Practitioner source; requires vendor language reference for exact keyword lists. |
| Windows 7 / Windows 10 | OS versions recommended for one Data Broker connector machine. | Connector environment only. | Connector environment only. | N/A | REP32 installed on machine. | Verified for connector only | Data Broker article. |

---

# 14. Known Quirks and Implementation Notes

| Topic | Research Finding | Axys | APX | Confidence | Notes |
|---|---|---|---|---|---|
| Do not assume identical behavior between Axys and APX | Repository standard requires separate Axys and APX behavior whenever they differ. | Applies. | Applies. | Verified | From supplied blueprint. |
| Use Unknown rather than inventing field names | Field names, transaction codes, report behavior, and implementation details must not be invented. | Applies. | Applies. | Verified | From supplied blueprint. |
| APX has more modern data access options than Axys | Practitioner material states APX users can access underlying SQL Server database via Stored Accounting Functions, Public Views, SSRS, REST API, etc., and that APX has capabilities Axys users do not. | Contrast only. | Applies. | High Confidence | Practitioner source; verify exact availability by APX version. |
| IMEX remains relevant even where APIs exist | Practitioner material states APX import/export methods like IMEX may still be efficient and reliable for certain data elements. | Likely relevant. | Applies. | High Confidence | Practitioner source. |
| REP/RepLang remains part of reporting architecture | Practitioner material states Replang is still part of Axys/APX reporting architecture. | Applies. | Applies. | High Confidence | Practitioner source. |
| Report-based extracts are common integration pattern | Data Broker connector uses Advent standard reports and macros via REP32 to generate extracts. | Applies for connector. | Applies for connector. | Verified | Data Broker article. |
| Account identifiers can be ambiguous | Integration documentation distinguishes custodian account number from PMS account number and says they may be the same but are treated differently. | Applies to integrations. | Applies to integrations. | Verified | Data Broker article. |
| Public marketing pages are insufficient for field dictionaries | Vendor pages verify broad capabilities but not field names, transaction code lists, IMEX objects, REP report internals, or processing rules. | Applies. | Applies. | Verified | Research conclusion from source limitations. |
| Third-party connector version support is not product lifecycle support | Axys/APX versions listed by Data Broker are connector support constraints only. | Applies. | Applies. | Verified | Avoid representing connector minimum versions as vendor product requirements. |
| External integration terms may not be Advent-native | Terms from AIA/Data Broker such as source folder, translation files, connector mappings may be integration-layer constructs rather than Axys/APX native concepts. | Applies. | Applies. | Medium Confidence | Requires distinction in final glossary. |

---

# 15. Unknowns to Carry Forward

These items should be explicitly marked Unknown in the chapter unless additional source material is supplied.

| Unknown | Why It Matters | Needed Evidence |
|---|---|---|
| Exact IMEX object names for transactions, holdings, performance, prices, security master, classifications, and cash. | Needed for implementation-ready glossary and data dictionary. | IMEX manual, object list, export headers, or production extract samples. |
| Exact Axys transaction codes. | Required for transaction glossary and audit use cases. | Vendor transaction code table or transaction export/report examples. |
| Exact APX transaction codes. | Required for APX-specific glossary. | APX documentation or transaction export samples. |
| Exact security master field names. | Needed for data dictionary and glossary cross-reference. | Security master IMEX export or REP output. |
| Whether Axys/APX use the same field names for security identifiers. | Avoids false equivalence between products. | Side-by-side Axys/APX extracts. |
| Performance storage vs recalculation rules. | Important for performance glossary terms such as stored return, recalculated return, linked return, and performance history. | Vendor performance documentation, REP source, or controlled production test. |
| Exact report names for holdings, transaction, performance, security master, pricing, composite, and reconciliation reports. | Needed for a final glossary and report chapter cross-reference. | Report menu export, REP folder, report source files, or screenshots. |
| REP file structure and syntax. | Needed for accurate glossary of REP, RepLang, report object, macro, and report packaging. | REP source examples and language reference. |
| APX database schema names and public views. | Needed for APX glossary/data model. | APX data dictionary, public view reference, or SQL metadata. |
| API endpoint names and APX REST API version behavior. | Needed for modern APX integration glossary. | APX API documentation. |
| Corporate action storage and processing. | Needed for dividend/split/corporate action glossary. | Corporate action module documentation or export samples. |
| Classification storage and hierarchy behavior. | Needed for asset class/sector/country/region glossary. | Security master/classification export and report samples. |
| Multicurrency calculation rules. | Needed for base/local currency and FX return terms. | Multicurrency accounting/performance documentation. |
| Fixed income accrual/amortization/accretion details. | Needed for fixed income glossary. | Fixed income accounting documentation and report samples. |

---

# 16. Proposed Glossary Structure for Chapter_16_Glossary.md

Recommended chapter organization:

1. Overview and confidence legend
2. Core product terms
3. Architecture and platform terms
4. Reporting and REP terms
5. IMEX and integration terms
6. Portfolio/account/client terms
7. Security master and instrument terms
8. Transaction/holding/cash/pricing terms
9. Performance/benchmark/composite terms
10. Classification terms
11. Corporate action and reconciliation terms
12. Field-name glossary / research targets
13. Report-name glossary / research targets
14. Version/environment glossary
15. Known ambiguities
16. Unknowns

---

# 17. Reference URLs

- SS&C Advent Axys product page: https://www.advent.com/solutions/axys/
- SS&C Advent Portfolio Exchange product page: https://www.advent.com/solutions/advent-portfolio-exchange/
- SS&C Advent APX product brief page: https://www.advent.com/resources/all-resources/brief-advent-portfolio-exchange/
- Salentica / Black Diamond Data Broker article for SS&C Advent APX & Axys: https://engage.salentica.com/kb/article/247-data-broker-ss-c-advent-apx-axys/
- AdventGuru Axys/APX reporting and Replang material: https://adventguru.com/category/portfolio-management-systems/axys/
- AdventGuru IMEX tag page: https://adventguru.com/tag/imex/
- SS&C portfolio management solutions page: https://www.ssctech.com/solutions/portfolio-management

---

# 18. Research Notes for Future Expansion

The final glossary will become much more useful if the following are supplied before chapter drafting:

| Priority | Material | Use |
|---|---|---|
| 1 | Sample IMEX exports for security master, transactions, holdings, prices, portperf, secperf, classifications, and cash. | Convert candidate terms into verified field dictionary entries. |
| 2 | REP source files for common Axys/APX reports. | Verify report names, RepLang terms, field references, and output behavior. |
| 3 | Report output samples. | Verify business names, columns, groupings, and calculation labels. |
| 4 | Axys/APX transaction code lists. | Build accurate transaction glossary. |
| 5 | APX SQL/public view documentation. | Build APX-specific data access glossary. |
| 6 | Vendor performance documentation. | Resolve stored/recalculated return terminology. |
| 7 | Corporate action samples. | Resolve dividend, split, merger, coupon, and related glossary entries. |

---

# 19. Summary

This research file supports a useful glossary chapter but intentionally avoids false precision. The most reliable verified terms are product-level, reporting-level, and broad capability terms. The least verified areas are exact field names, transaction codes, IMEX object names, REP internals, report names, and processing behavior. Those should remain Unknown until supported by vendor documentation, REP source, IMEX export samples, or production observations.

---

# 20. Deep IMEX Addendum Incorporated 2026-06-30

Source: `axys_imex_deep_research.md`.

Additional glossary candidates:

| Term | Definition / usage | Confidence |
|---|---|---:|
| `imex32.exe` | Axys Import/Export utility executable in CI evidence. | Verified for CI |
| `pospos32.exe` | Axys Post Positions utility executable in CI evidence. | Verified for CI |
| `$pathexe` | Axys executable folder label in CI configuration. | Verified for CI |
| `$pathtrn` | Axys user/Trade Blotter folder label in CI configuration. | Verified for CI |
| `$pathcli` | Axys client/portfolio file folder label in CI configuration. | Verified for CI |
| `$pathinf` | Axys information folder label containing `sec.inf` and `type.inf` in CI context. | Verified for CI |
| `$pathpri` | Axys price folder label containing `*.pri` files in CI context. | Verified for CI |
| `$pathlog` | Axys IMEX log folder label in CI context. | Verified for CI |
| `ptopost.trn` | CI position file, CSV format, optionally containing lots when enabled/available. | Verified for CI |
| `MISSINGPRICES_yyyymmdd.csv` | CI optional diagnostic file for unresolved/missing prices. | Verified for CI |
| `SECTRANSLATIONS_yyyymmdd.csv` | CI optional security translation diagnostic/output file. | Verified for CI |
| `imex_catalog` | Proposed product-owned catalog of observed IMEX objects, fields, formats, directionality, examples, source, and confidence by version/installation. | Design guidance |

These terms should be labeled as CI/integration or product-design terms unless
additional vendor documentation confirms native Axys/APX status.
