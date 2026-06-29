# Chapter_16_Glossary.md

Repository: AXYS / APX Reference Repository  
Chapter: `Chapter_16_Glossary.md`  
Prepared: 2026-06-29  
Governing specification: `AXYS_APX_REFERENCE_BLUEPRINT.md`, Version 2.0  
Source basis: supplied research files `Research_02_Axys_Architecture.md` through `Research_16_Glossary.md`

---

## 1. Overview

This chapter defines commonly used Axys/APX repository terms. It is a technical glossary, not a replacement for the detailed subject chapters.

The glossary follows the repository rule that unsupported Axys or APX behavior must not be invented. Where exact field names, transaction codes, report names, IMEX object names, REP variables, or processing behavior are not supported by the supplied source material, the entry is marked `Unknown`.

### Confidence Legend

| Confidence | Meaning |
|---|---|
| Verified | Directly supported by the supplied blueprint, supplied research, named vendor material, named report evidence, named integration documentation, or cited practitioner documentation summarized in the research. |
| High Confidence | Strongly supported by supplied research and consistent implementation evidence, but not fully verified from vendor manuals, sample exports, or production observations. |
| Medium Confidence | Plausible and supported by partial, third-party, conversion, integration, or practitioner evidence. Requires confirmation before implementation use. |
| Unknown | Not verified by supplied material. Do not treat as fact. |

### Glossary Columns

| Column | Meaning |
|---|---|
| Term | Repository glossary term. |
| Definition | Implementation-oriented definition. |
| Axys | Axys-specific status or caveat. |
| APX | APX-specific status or caveat. |
| Interface / Report Context | Whether the term is relevant to IMEX, REP, reports, SQL, API, or integrations. |
| Confidence | Confidence classification. |

---

## 2. Core Product Terms

| Term | Definition | Axys | APX | Interface / Report Context | Confidence |
|---|---|---|---|---|---|
| Advent | Legacy vendor name historically associated with Axys and APX. Current materials use SS&C Advent branding. | Applicable. | Applicable. | N/A. | Verified |
| SS&C Advent | Current product/vendor branding used in supplied research for Axys and APX. | Applicable. | Applicable. | N/A. | Verified |
| Axys | SS&C Advent portfolio accounting, portfolio management, performance measurement, and reporting solution. | Primary product. Public material supports portfolio accounting, reporting, performance measurement, reconciliation, multicurrency, fixed income, and Report Writer Pro capabilities. | Related product only. | IMEX and REP/reporting are relevant integration/reporting mechanisms. | Verified |
| APX | Advent Portfolio Exchange; integrated portfolio management, accounting, reporting, performance, and client relationship management platform. | Related product only. | Primary product. Public material supports centralized platform/book-of-record positioning and front/middle/back office integration. | IMEX, REP/REP32, SSRS, SQL/public views, and possibly REST API may be relevant depending on version/environment. | Verified for product role; lower-level interfaces vary by confidence. |
| Advent Portfolio Exchange | Full product name for APX. | N/A. | Full product name. | N/A. | Verified |
| Advent Investment Suite | Product suite context in which Axys/APX appear in vendor/product material. | Part of suite context. | Part of suite context. | N/A. | Verified |
| Portfolio Accounting System / PMS | System of record for portfolio accounting data such as portfolios, holdings, transactions, cash, prices, performance, and reports. | Axys can serve this role. | APX can serve this role. | Integration documentation may use PMS account number. | High Confidence |
| Book of Record | Authoritative record for portfolio/accounting/performance/client-management data. | Public Axys material supports accounting/reporting use, but “centralized book of record” was not established as the reviewed Axys wording. | APX is described in supplied research as a centralized book of record. | Relevant to data access and reporting architecture. | Verified for APX; Medium Confidence for Axys concept. |
| Front Office | Portfolio management/client-facing workflow layer. | Exact Axys front-office implementation Unknown. | APX is marketed as connecting front, middle, and back offices. | Reports/dashboards may support this audience. | Verified for APX; Unknown for Axys specifics. |
| Middle Office | Operations, controls, reconciliation, and processing layer between front and back office. | Exact Axys middle-office implementation Unknown. | APX is marketed as connecting front, middle, and back offices. | IMEX/REP/integration workflows may support this layer. | Verified for APX; Unknown for Axys specifics. |
| Back Office | Accounting, reconciliation, reporting, settlement, and operational processing layer. | Axys accounting/reporting and reconciliation capabilities are supported by research. | APX accounting/reporting capabilities are supported by research. | IMEX, REP, reports, and SQL/API extracts may support back-office workflows. | High Confidence |

---

## 3. Architecture and Platform Terms

| Term | Definition | Axys | APX | Interface / Report Context | Confidence |
|---|---|---|---|---|---|
| Proprietary / File-Oriented Architecture | Architecture where data is held in proprietary or file-oriented structures rather than exposed relational tables. | Public/practitioner research characterizes Axys as proprietary/file-oriented; exact internals remain partly unverified. | APX is not primarily characterized this way in supplied research. | Direct file access is risky; IMEX/REP are preferred where available. | Medium Confidence for Axys. |
| SQL-Based Platform | Platform backed by SQL Server or SQL-oriented data access. | Not verified for Axys. | APX is described in vendor/practitioner research as SQL-based or SQL-accessible. Exact schema remains Unknown. | Public views, Stored Accounting Functions, SSRS, SQL tools, and possibly REST API may be relevant. | High Confidence for APX access model; Unknown for table names. |
| Local Deployment | Locally installed/on-premise deployment. | Integration research supports local/on-prem or hosted deployment contexts for Axys. | APX can be deployed locally according to vendor/product research. | May require installed client tools such as REP32 for some integrations. | Verified / High Confidence depending source. |
| Cloud-Delivered | Hosted/cloud delivery option. | AOS-hosted Axys contexts are referenced by connector research; exact hosting model Unknown. | APX public product material supports cloud-delivered option. | Client-tool and custom-report access may vary by hosting model. | Verified for APX; Medium Confidence for Axys hosted context. |
| AOS-hosted | Hosted Advent environment referenced in integration material. Exact expansion and technical model are not established by supplied source material. | Mentioned in connector context. | Mentioned in connector context. | May affect where REP32/report execution occurs. | Medium Confidence |
| Client Tools | Locally installed Advent components needed by some integrations. | Data Broker connector requires Advent client tools in Axys/APX contexts. | Same. | Includes REP32 in the connector workflow. | Verified for connector context. |
| REP32.exe | Advent report engine/application used by at least one connector to run standard reports/macros and RepLang scripts. | Used by connector for Axys extraction. | Used by connector for APX extraction. | REP/report extraction mechanism; not IMEX itself. | Verified for connector context. |
| 32-bit Windows Application | Architecture characteristic of a specific third-party Data Broker connector, not necessarily Axys/APX themselves. | Connector can be used with Axys. | Connector can be used with APX. | Requires Windows environment and installed Advent client tools in that connector context. | Verified for connector context only. |
| Genesis Platform | SS&C platform referenced in supplied APX research in connection with APX/Data Lens direction. | N/A. | APX-related platform context. | Not enough evidence to document implementation internals. | Verified as product-direction reference; implementation Unknown. |
| Data Lens for APX | APX-related data integration, aggregation, visualization, dashboard/reporting capability referenced by supplied research. | N/A. | APX-related capability. | Distinct from traditional IMEX/REP unless source material proves overlap. | Verified at product-capability level. |
| Direct File Access | Reading/writing Axys data files directly rather than through supported utilities or reports. | Practitioner research says possible for knowledgeable users but not best practice because file formats can change. | Not the main APX access model in supplied research. | Prefer IMEX/REP/exported reports unless file formats are verified. | Medium / High Confidence as caution. |
| File Conversion | Conversion of files between versions or systems. | Axys v3.7 to v3.8 conversion risk is described in practitioner research. | APX-to-Axys conversion through IMEX is described as possible but version-sensitive. | Migration/conversion workflows require validation. | Medium Confidence |

---

## 4. IMEX and Integration Terms

| Term | Definition | Axys | APX | Interface / Report Context | Confidence |
|---|---|---|---|---|---|
| IMEX | Advent import/export utility or mechanism. | ByAllAccounts research explicitly defines IMEX as the Axys Import/Export utility in that guide. | APX integration research references Advent IMEX logs and APX Import/Export workflows; technical identity with Axys IMEX is Unknown. | Core import/export chapter topic. | Verified for Axys terminology; High Confidence for APX existence; exact APX details Unknown. |
| Import | Loading data into Axys/APX. | Verified in third-party workflows for transactions, positions, and prices through Axys Import/Export. | Verified in APX workflows through APX Import/Export, blotters, and AIA/CI integrations. | IMEX and blotters may both participate. | High Confidence |
| Export | Extracting data from Axys/APX. | Can be performed through reports, IMEX, and integration tooling. | Can be performed through reports, IMEX, SQL/public views, SSRS, and possibly REST API depending on environment. | Distinguish machine-readable exports from presentation reports. | High Confidence |
| Extract | Generated file or dataset used outside Axys/APX. | May be created through IMEX, REP reports, macros, or third-party tools. | Same, with additional SQL/API options in some APX versions. | Integration/data warehouse context. | High Confidence |
| Data Feed | Recurring data transfer from Axys/APX to another system. | Data Broker connector supports feeds from Axys. | Data Broker connector supports feeds from APX. | Connector uses reports/macros rather than proving native IMEX object names. | Verified for connector context. |
| Daily Feed | Scheduled daily extract. | Supported in connector context. | Supported in connector context. | REP32/report macros may generate it. | Verified for connector context. |
| Mapping | Association between source and target identifiers or fields. | Relevant to portfolio, security, transaction, broker, fee, and account translations. | Same. | Often performed by integration tooling rather than native IMEX. | High Confidence |
| Translation | Mapping/normalization of external values to Axys/APX values. | Axys integrations use security/account/transaction translation. | APX integrations use security/account/transaction/broker/fee translation. | Must not be assumed native unless documented. | High Confidence |
| Trade Blotter | Staging area/file/workflow for transaction review/posting. | Axys CI workflow writes transactions to `topost.trn` Trade Blotter file. | APX workflows use Trade Blotter; ACA-generated reorg transactions post to APX Trade Blotter in vendor brief. | Transaction import/review context. | Verified for cited workflows. |
| Position Blotter | APX staging area for position imports/reconciliation in some workflows. | Axys equivalent not established as a named blotter from supplied material. | CI/APX research identifies Position Blotter. | Position reconciliation/import context. | Verified for APX CI workflow. |
| Lot Blotter / Tax Lot Blotter | APX staging/reconciliation area for tax-lot or lot-level position data in some workflows. | Unknown. | Identified in APX CI/AIA research. | Lot reconciliation/import context. | Verified for workflow; native generality Unknown. |
| Statement Blotter | APX workflow object for statement transactions used in reconciliation contexts. | Unknown. | Identified in APX AIA research. | Transaction/reconciliation context. | Verified for workflow. |
| Account Blotter | APX workflow object for importing account demographic data in AIA research. | Unknown. | Identified in APX AIA research. | Account import context. | Verified for workflow. |
| Initial Transaction Blotter | APX workflow object for creating initial deliver-in transactions from positions when configured. | Unknown. | Identified in APX AIA research. | Initial position/transaction conversion context. | Verified for workflow. |
| APXIX.exe / ApxIx | APX Import/Export utility naming seen in APX research. | N/A. | APX import/export executable/function name in specific integration/manual contexts. Whether names are version/context variants is Unknown. | APX IMEX. | Verified in specific sources; broader equivalence Unknown. |
| imex32.exe | Axys Import/Export utility executable name in ByAllAccounts CI research. | Axys CI looks for `imex32.exe`. | N/A. | Axys IMEX. | Verified for CI workflow. |
| IMEX Log | Log from IMEX import/export process. | CI research references IMEX logs such as `imexPrices.log`, `imexPositions.log`, and `imexPositionLots.log`. | APX AIA research references Advent IMEX Log and History Log. | Troubleshooting/audit context. | Verified for workflows. |
| Source Data Folder | Folder used by third-party integration tooling to hold input files. | External integration concept unless verified as Axys-native. | External integration concept unless verified as APX-native. | Integration-specific. | Medium Confidence |
| Extract Folder | Folder used for generated extracts/reports/files. | Used in integration/report workflows. | Used in integration/report workflows. | May contain REP/IMEX outputs. | Medium Confidence |
| Unattended Run | Scheduled execution without user interaction. | Connector supports scheduled unattended extraction. | Same. | REP32/macros used by connector. | Verified for connector context. |
| REST API | Programmatic APX integration option referenced in practitioner research. | Not verified for Axys. | Practitioner research says newer APX versions may include REST API options. | Alternative/complement to IMEX/REP. | Medium / High Confidence for APX existence; endpoint details Unknown. |
| Public Views | APX database views referenced as access path. | Not applicable to Axys in supplied research. | Available but limited according to practitioner research. | SQL/reporting extraction path. | Medium / High Confidence; field lists Unknown. |
| Stored Accounting Functions | APX database/reporting functions referenced by practitioner material. | Not verified. | Referenced as APX access path. | SQL/reporting extraction path. | Medium Confidence |
| SSRS | Microsoft SQL Server Reporting Services. | Not established as Axys report engine. | APX reports guide/research identifies SSRS-based APX investment reports. | APX reporting engine/context. | Verified for APX guide-covered reports. |
| ETL | Extract, Transform, Load process/tooling used to populate downstream data warehouses. | Practitioner material references ETL patterns for Axys. | Same for APX. | IMEX/REP/API/SQL may be sources. | High Confidence |
| Data Warehouse | Downstream repository populated from Axys/APX data. | Practitioner research references warehouse patterns. | Same. | Integration/reporting context. | High Confidence |

---

## 5. REP, Replang, and Reporting Terms

| Term | Definition | Axys | APX | Interface / Report Context | Confidence |
|---|---|---|---|---|---|
| Report | Generated output from accounting/performance/client/reporting data. | Axys has predefined and customizable reports. | APX has standard reports, client reporting, analytics reports, dashboards, and packaging. | REP/SSRS/report output; distinct from IMEX unless configured as extract. | Verified |
| Standard Report | Vendor-supplied/predefined report. | Axys has hundreds/predefined reports in vendor material. | APX has standard report library in vendor material and report guide. | May be run by REP32/macros in integrations. | Verified |
| Custom Report | Firm-specific or modified report. | Axys supports Report Writer Pro and Replang customization. | APX supports flexible custom reporting; Replang/REP also appears in practitioner/integration evidence. | Custom output may be used for data extraction. | High Confidence |
| Report Writer Pro | Advent report authoring/customization tool. | Verified for Axys from vendor material. | Practitioner evidence supports APX usage, but exact version boundaries are not fully verified. | REP/report customization. | Verified for Axys; Medium / High Confidence for APX. |
| REP | Advent report file/reporting layer, commonly associated with `.REP`, REP32, and Replang. | Axys `.REP` examples are verified. | APX RepLang/REP usage is supported by practitioner/connector evidence but exact artifact locations are Unknown. | Chapter 13 topic. | Verified for Axys examples; Medium / High Confidence for APX. |
| `.REP` | Report source/file extension used in Axys examples. | `AMAN.REP` and `CDIhold.rep` appear in supplied research. | APX `.REP` usage possible but exact locations/coverage Unknown. | REP source artifact. | Verified for Axys examples; Unknown/Medium for APX. |
| `.RPW` | Report Writer-created report extension referenced by practitioner research. | Relevant to Axys custom reports. | Relevant to APX custom reports per practitioner research. | Report Writer / Replang relationship. | Medium Confidence |
| Replang / RepLang | Advent proprietary report writing language. | Axys reports are written in Replang in CSSI example. | Practitioner source says Replang remains part of APX architecture and APX has additional keywords. | REP/report source language. | Verified for Axys; Medium / High Confidence for APX. |
| REP32 | Report execution engine/application used by connector. | Used for Axys extraction in connector. | Used for APX extraction in connector. | Reports/macros/extract generation. | Verified for connector context. |
| Macro | Automation artifact used to run standard reports/extracts. | Connector uses standard reports and macros. | Same. | REP32/report automation. | Verified for connector context. |
| Report Packaging | Process of bundling/automating multiple reports. | Axys can combine multiple reports/graphs/objects on one page. | APX supports automated report packaging. | Client reporting context. | Verified |
| Compound Report | Multi-part report package/report object; exact Axys/APX implementation not verified. | Likely supported by report packaging/custom report examples. | Likely supported by APX reporting. | Report package context. | Medium Confidence |
| Report Object | Component within a report package or page. | Exact technical definition Unknown. | Exact technical definition Unknown. | Report packaging context. | Unknown |
| Report Menu | User/menu configuration exposing reports. | Axys examples reference report menu/custom report menu. | APX examples/reference workflows include custom report menu in AIA contexts. | REP/report administration. | Medium Confidence |
| Output to Excel | Report export to Excel. | Practitioner research says Axys/APX users can export reports to Excel. | Same. | Report output, not necessarily IMEX. | High Confidence |
| CSV Report Output | Report output changed or designed to produce CSV/text. | Practitioner research says Report Writer/Replang can produce CSV/text exports. | Same. | Report-driven extraction. | High Confidence |
| Drill-Down | Ability to access underlying detail from a report. | Axys vendor material supports drill-down from reports. | APX report guide has expansion/drill-down behavior in some reports; exact term may differ. | Report UI/output. | Verified for Axys; High Confidence for APX examples. |
| Graphics | Chart/visual report output. | Axys supports graphics in reports. | APX reports include charts/graphs in report guide/product material. | Report output. | Verified |
| SSRS Report | APX report built on Microsoft SQL Server Reporting Services. | Not established for Axys. | APX report guide-covered investment reports are SSRS-based. | APX reporting source artifacts may be RDL/SSRS, but exact files Unknown. | Verified for APX guide-covered reports. |

---

## 6. Portfolio, Account, Client, and Composite Terms

| Term | Definition | Axys | APX | Interface / Report Context | Confidence |
|---|---|---|---|---|---|
| Portfolio | Accounting/reporting entity containing holdings, transactions, cash, and performance. | Axys manages and reports portfolios. | APX tracks portfolio data. | Portfolio identifiers appear in reports/integrations; exact native field names vary/Unknown. | Verified generally. |
| Account | Often used interchangeably with portfolio in integration/report contexts, but exact distinction is environment-specific. | Native Axys account/portfolio distinction Unknown from supplied material. | Native APX account/portfolio distinction Unknown from supplied material. | Integration may distinguish PMS account number and custodian account number. | Medium Confidence |
| Client | Person/entity served by the firm. | Axys supports client reporting; exact client data model Unknown. | APX includes client relationship management. | Reports and integrations may include client identifiers. | Verified generally; exact fields Unknown. |
| Relationship | Client/household/business relationship construct. | Unknown. | APX CRM positioning supports relationship concept; schema Unknown. | CRM/reporting context. | Medium Confidence |
| Prospect | Potential client. | Unknown. | APX product material references prospect data. | CRM/reporting context. | Verified for APX concept; implementation Unknown. |
| Household | Grouping of related clients/accounts, common in wealth workflows. | Not verified in supplied Axys research. | APX report guide includes Household Overview. | Report/client-reporting context. | Verified as APX report name; Axys Unknown. |
| Portfolio Group | Grouping of portfolios for reporting or management. | Axys can group/report portfolios by manager, asset class, objective, or custom categories. | APX grouping likely, but exact data model Unknown. | Reports/IMEX may use group scope; exact fields Unknown. | Verified for Axys; Medium Confidence for APX. |
| Composite | Group of portfolios for composite performance reporting, often GIPS-related. | Axys supports composite management concepts. | APX supports composite management for GIPS compliance. | Composite reports/extracts likely; exact names/objects Unknown. | Verified |
| GIPS | Global Investment Performance Standards. | Axys supports GIPS-related performance measurement/reporting at product level. | APX supports composite management for GIPS compliance at product level. | Composite/performance reporting context. | Verified |
| Portfolio Entry Date | Date a portfolio enters a composite. | Axys vendor material references tracking entry/exit dates. | APX composite support implies similar need, but exact field Unknown. | Composite reporting. | Verified for Axys concept; Unknown for APX field. |
| Portfolio Exit Date | Date a portfolio exits a composite. | Axys vendor material references tracking entry/exit dates. | APX composite support implies similar need, but exact field Unknown. | Composite reporting. | Verified for Axys concept; Unknown for APX field. |
| Manager | Portfolio grouping/reporting attribute. | Axys can group portfolios by manager. | APX exact behavior Unknown from supplied material. | Portfolio/group report dimension. | Verified for Axys; Unknown for APX. |
| Investment Objective | Portfolio grouping/reporting attribute. | Axys can group portfolios by investment objective. | APX exact behavior Unknown. | Portfolio/group report dimension. | Verified for Axys; Unknown for APX. |
| PMS Account Number | Account identifier from portfolio management system integration. | Relevant to Axys integrations. | Relevant to APX integrations. | Data Broker distinguishes it from custodian account number. | Verified as integration concept; exact native field Unknown. |
| Custodian Account Number | Account identifier from custodian. | May differ from PMS account number. | Same. | Used in mapping/reconciliation integrations. | Verified as integration concept; exact native field Unknown. |
| APX Portfolio Code | Portfolio identifier label observed in APX integration research. | N/A. | Used in Custodial Integrator/APX workflow. | APX integration field. | Verified for CI workflow. |
| Portfolio Code | Portfolio/account code label observed in Axys Report Writer/report examples and integrations. | Observed in Axys report/export contexts. | APX equivalent may exist but exact field usage varies. | REP/report/integration context. | Verified for observed Axys contexts; APX Unknown/Medium. |

---

## 7. Security Master and Instrument Terms

| Term | Definition | Axys | APX | Interface / Report Context | Confidence |
|---|---|---|---|---|---|
| Security Master | Reference data for securities/instruments. | Axys security information appears in `sec.inf` / `type.inf` workflows; exact complete schema Unknown. | APX security information appears in `sec.inf` / `type.inf` integration workflows and APX Security Master references; exact SQL schema Unknown. | Security master IMEX object names Unknown. Reports may expose security labels. | Verified concept; field/schema details Unknown. |
| Security | Tradable/reportable instrument. | Axys supports cash, equities, fixed income, money market/cash types, municipal/corporate/government bonds, MBS, step-up bonds. | APX supports equities, fixed income, mutual funds, FX, derivatives, alternatives/private equity at product level. | Security identifiers appear across transactions, holdings, prices, performance, reports. | Verified generally. |
| Instrument | Alternative term for security/asset. | Axys product material uses security types. | APX product material supports multiple instrument/asset types. | Data model/reporting context. | Verified generally. |
| Security Symbol | Product security symbol used in matching. May be ticker, CUSIP-like value, or internal symbol depending setup. | Axys CI uses Axys Symbol. | APX CI uses APX Symbol. | Appears in integration/security translation context. | Verified for CI context. |
| Security Type | Product security type code used with symbol to identify securities. Not the same as classification such as sector or asset class. | Axys CI uses Axys Security Type examples such as `csus`, `tfus`, `oaus`. | APX CI uses APX Security Type examples such as `efus`, `adus`, `epus`. | Security matching/import context. | Verified for examples; full type dictionary Unknown. |
| `sec.inf` | Security information file/artifact referenced in Axys/APX integration and conversion research. | Axys security file/information context verified by multiple research files. | APX CI/AIA uses `sec.inf`-layout/security info in integration context; native APX SQL storage remains Unknown. | Security information/import/export context. | Verified for integration/conversion contexts. |
| `type.inf` | Security type information file/artifact referenced in integration and conversion research. | Axys security type file context verified. | APX CI uses `type.inf` in security matching context. | Security type/import/export context. | Verified for integration/conversion contexts. |
| `split.inf` | Axys securities splits file in conversion/corporate action research. | High-confidence Axys file evidence. | APX equivalent/native split storage Unknown. | Corporate action/conversion context. | High Confidence for Axys; Unknown for APX. |
| `.pri` / `*.pri` | Price file extension/artifact in Axys/APX price-file workflows. | Axys price files appear in `$pathpri` in CI research. | APX AIA price-file examples include `.pri` naming such as `mmddyy_CDI.pri`. | Price import/merge context. | Verified for integration contexts; native layout Unknown. |
| `.cli` / `*.cli` | Client/portfolio/account files referenced in Axys conversion and integration research; APX AIA research also references `.cli` in certain contexts. | Axys client files; may contain transaction/cost-basis data in conversion research. | APX `.cli` references appear in integration contexts; native role Unknown. | Conversion/import/config context. | Verified for Axys conversion/integration; Medium for APX integration artifact. |
| `topost.trn` | Axys Trade Blotter file in CI workflow. | CI appends transactions to `topost.trn`. | N/A. | Transaction import/staging. | Verified for CI workflow. |
| Cash Type | Cash or money-market-like security/category. | Axys supports money market and other cash types. | APX supports cash/settlement/multi-asset coverage generally. | Cash/holding/transaction context. | Verified for Axys; High Confidence for APX. |
| Equity | Equity security/instrument. | Supported. | Supported. | Security master/holdings/transactions/performance context. | Verified |
| Fixed Income | Bond and related debt security category. | Axys has detailed fixed income capabilities. | APX supports fixed income generally. | Security master/pricing/accrual/income context. | Verified |
| Municipal Bond | Fixed income subtype. | Axys supports variable-rate municipal bonds. | APX subtype support not specifically verified from supplied material. | Security master/fixed income context. | Verified for Axys; Unknown for APX subtype. |
| Corporate Bond | Fixed income subtype. | Axys supports corporate bonds. | APX supports fixed income generally. | Security master/fixed income context. | Verified for Axys; High Confidence for APX. |
| Government Bond | Fixed income subtype. | Axys supports government bonds. | APX supports fixed income generally. | Security master/fixed income context. | Verified for Axys; High Confidence for APX. |
| Mortgage-Backed Security / MBS | Mortgage-backed fixed income instrument. | Axys supports MBS. | APX subtype support not specifically verified. | Security master/pricing/paydown context. | Verified for Axys; Unknown for APX subtype. |
| Step-Up Bond | Bond whose coupon rate changes by terms. | Axys supports step-up bonds. | APX subtype support not specifically verified. | Fixed income security master context. | Verified for Axys; Unknown for APX subtype. |
| Mutual Fund | Pooled investment vehicle. | Not specifically verified in supplied Axys material, though likely in practice. | APX product brief supports mutual funds. | Security/holding/transaction context. | Verified for APX; Unknown for Axys in supplied material. |
| FX | Foreign exchange/currency instrument or currency context. | Axys has multicurrency capabilities. | APX supports FX and settlement in any currency. | Transactions/pricing/performance context. | Verified generally. |
| Derivative | Instrument whose value derives from underlying assets. | Not specifically verified for Axys in supplied material. | APX supports derivatives. | Security/holding context. | Verified for APX; Unknown for Axys. |
| Alternative Investment | Non-traditional investment such as private equity. | Not specifically verified for Axys. | APX supports alternatives including private equity. | Security/holding/reporting context. | Verified for APX; Unknown for Axys. |
| Ticker | Market ticker identifier. | Used in security matching examples, but native field name Unknown. | Same. | Matching/report labels; not necessarily native field. | Verified as integration identifier; native field Unknown. |
| CUSIP | North American security identifier. | Used in security matching/conversion examples. | Same. | Matching/report labels; not necessarily native field. | Verified as integration identifier; native field Unknown. |
| ISIN | International security identifier. | Not verified in supplied material. | Not verified in supplied material. | Unknown. | Unknown |
| SEDOL | Security identifier common in non-US markets. | Not verified in supplied material. | Not verified in supplied material. | Unknown. | Unknown |
| Reserved Security Type Prefix | Prefix excluded from security matching by ByAllAccounts CI. | CI excludes prefixes `aw`, `br`, `ex`, `ep`, `pi`, `rs` from Axys matching. | Same prefixes excluded from APX CI matching. | Integration matching rule only; not necessarily platform-wide. | Verified for CI context. |

---

## 8. Transactions, Holdings, Cash, and Pricing Terms

| Term | Definition | Axys | APX | Interface / Report Context | Confidence |
|---|---|---|---|---|---|
| Transaction | Accounting event such as trade, income, cash movement, fee, transfer, corporate action, or correction. | Axys tracks transactions; `.cli`/Trade Blotter workflows appear in research. | APX tracks transactions; blotters and Transaction Summary report appear in research. | Transaction IMEX object names Unknown; reports/REP may expose transactions. | Verified generally; exact code matrix Unknown. |
| Transaction Code | Code identifying transaction type. | Observed codes exist in integration research, but official complete Axys code list Unknown. | Observed codes exist in integration research, but official complete APX code list Unknown. | Do not implement from glossary alone. | Medium Confidence for observed examples; Unknown as official matrix. |
| Buy | Purchase transaction type. | Observed `by` in integration examples but official code status requires confirmation. | Observed `by` in APX integration examples. | Transaction import/report context. | Medium Confidence as observed code; official behavior Unknown. |
| Sell | Sale transaction type. | Observed in integration/conversion contexts. | Observed `sl` in APX integration examples. | Transaction import/report context. | Medium Confidence as observed code; official behavior Unknown. |
| Short Sale | Transaction creating/increasing short exposure. | Exact native Axys code behavior Unknown. | Observed `ss` in APX integration examples. | Transaction import/report context. | Medium Confidence for observed examples. |
| Cover Short | Transaction reducing short exposure. | Exact native Axys code behavior Unknown. | Observed `cs` in APX integration examples. | Transaction import/report context. | Medium Confidence for observed examples. |
| Deliver In / Transfer In | Movement of security/cash into a portfolio without ordinary purchase economics. | `li` appears in Axys conversion/integration contexts; interpretation can depend on `.cli` setting in conversion evidence. | `li` appears in APX integration examples. | Transaction import/conversion context. | Medium Confidence; code-only interpretation unsafe. |
| Deliver Out / Transfer Out | Movement of security/cash out of a portfolio without ordinary sale economics. | `lo` appears in Axys conversion/integration contexts; interpretation can depend on `.cli` setting. | `lo` appears in APX integration examples. | Transaction import/conversion context. | Medium Confidence; code-only interpretation unsafe. |
| Reversal / Cancellation | Transaction or import record intended to cancel/delete/reverse a prior transaction. | Uppercase code pattern such as `by` → `BY` appears in Axys workflow evidence. | Same pattern appears in APX workflow evidence. | Trade Blotter/import context. | Medium Confidence; universal native rule Unknown. |
| Reinvestment | Income distribution reinvested into additional shares/units. | Morningstar conversion research says Axys distribution reinvestments may appear as Buy + Distribution pairs. | ByAllAccounts APX research shows `dv`/`by` paired examples. | Transaction/conversion context. | Medium Confidence |
| Dividend | Income distribution from a security. | Exact storage/transaction/report behavior Unknown. | Exact behavior Unknown; report examples show dividends in Transaction Summary context. | Transaction/report/corporate-action context. | Unknown / Medium depending report label. |
| Interest | Fixed income or cash income. | Axys fixed-income capability implies handling, but exact transaction behavior Unknown. | Exact behavior Unknown; integration examples include income/interest logic. | Transaction/cash/income context. | Medium Confidence |
| Return of Capital | Distribution reducing cost basis and/or returning capital. | Exact Axys behavior Unknown. | `rc` and bond-related `pd` appear in APX integration examples. | Transaction/corporate action context. | Medium Confidence for observed examples; official behavior Unknown. |
| Principal Paydown | Fixed income/MBS principal reduction transaction/event. | Morningstar conversion evidence notes Axys paydown conversion issues. | Exact APX behavior Unknown. | Fixed income/transaction/corporate action context. | Medium Confidence for Axys conversion issue. |
| Fee | Transaction reducing portfolio cash/value for charges. | Fee coding examples include `epus`/`exus` in conversion research but exact native semantics Unknown. | APX fee translation examples use special security type/symbol fields. | Transaction/cash/performance context. | Medium Confidence for observed examples. |
| Trade | Buy/sell/market transaction. | Axys reconciles trade information. | APX tracks transactions; exact trade model Unknown. | Transaction/report/reconciliation context. | Verified for Axys capability; High Confidence for APX. |
| Trade Date | Economic/execution date. | Axys supports trade-date accounting option. | Exact APX support Unknown from supplied material. | Transaction/report/performance context. | Verified for Axys concept; Unknown for APX field. |
| Settlement Date | Date cash/security settles. | Axys supports settlement-date accounting option. | APX supports settlement in any currency at product level; field Unknown. | Transaction/cash/reconciliation context. | Verified for Axys concept; Medium Confidence for APX concept. |
| Trade-Date Accounting | Accounting treatment recognizing activity on trade date. | Axys supports trade-date accounting. | Unknown in supplied APX material. | Accounting/report setting. | Verified for Axys; Unknown for APX. |
| Settlement-Date Accounting | Accounting treatment recognizing activity on settlement date. | Axys supports settlement-date accounting. | Unknown in supplied APX material. | Accounting/report setting. | Verified for Axys; Unknown for APX. |
| Tax-Lot Accounting | Lot-specific accounting method. | Axys supports tax-lot accounting. | APX lot/tax-lot blotter workflows exist in integration research, but product-level accounting basis detail Unknown. | Holdings/transactions/realized gain context. | Verified for Axys; Medium Confidence for APX workflow. |
| Average-Cost Accounting | Accounting method using average cost. | Axys supports average-cost accounting. | Unknown from supplied APX material. | Accounting/report setting. | Verified for Axys; Unknown for APX. |
| Position | Quantity/value of a security in a portfolio. | Axys supports position/holding reporting and reconciliation. | APX tracks holdings/positions and supports position blotters in workflow evidence. | Holdings/position extracts; exact objects Unknown. | Verified generally. |
| Holding | Security/cash position as of a date. | Axys Portfolio Appraisal supports holdings reporting. | APX Portfolio Appraisal report exists in report research. | Holdings reports/position imports. | Verified for report concepts; storage model Unknown. |
| Portfolio Appraisal | Holdings/assets report. | Axys Portfolio Appraisal is verified; can show holdings and, in examples, Portfolio Code. | APX Portfolio Appraisal appears in APX report guide research. | Report output; not necessarily IMEX. | Verified for Axys; Medium/Verified for APX depending report-guide evidence. |
| Cash | Cash balance/instrument/category. | Axys supports cash types and multicurrency. | APX supports cash/settlement/multi-asset context. | Cash may appear in transactions/holdings; exact cash IMEX object Unknown. | Verified generally; exact fields Unknown. |
| Cash Sweep | Movement between cash and money market/cash-like vehicles; may be removed by integration tooling. | WealthTechs Axys workflow defines/removes cash sweeps under specific conditions. | Similar APX AIA workflow evidence. | Integration cleanup, not verified native behavior. | High Confidence for workflow; native behavior Unknown. |
| Intra-Account Cash Journal | Offset pair of cash journal transactions within same account; may be removed by integration tooling. | WealthTechs Axys workflow identifies removal logic. | WealthTechs APX workflow identifies removal logic. | Integration cleanup. | High Confidence for workflow; native behavior Unknown. |
| Price | Valuation price for a security. | Axys `.pri` file/import workflows verified; exact price schema Unknown. | APX AIA price-file/update logic verified; exact APX schema Unknown. | Pricing IMEX/file/report context. | Verified for workflow artifacts; exact model Unknown. |
| Closing Price | Market close price. | Not specifically verified as Axys field/concept. | Not specifically verified as APX field/concept. | Requires price samples/docs. | Unknown |
| Price File | File containing prices for import/update. | Axys `*.pri` in `$pathpri` context. | APX AIA `.pri` price-file examples. | Pricing/import context. | Verified for integration contexts. |
| Missing Price | Security/position without usable price. | Axys CI Missing Price file behavior described in pricing research. | Native APX missing-price behavior Unknown. | Pricing exception context. | Verified for Axys CI workflow; APX Unknown. |
| Stale Price | Price that is available but not current enough for processing. | Axys CI release notes discuss stale price states. | Unknown. | Pricing exception context. | Medium Confidence for Axys CI workflow. |
| Calculated Price | Price computed from units and market value when external price unavailable in CI workflow. | Axys CI release notes support calculated price behavior. | Unknown. | Pricing/import context. | Verified for Axys CI release behavior. |
| Price Set | Group/source set of prices. | Unknown in supplied Axys material. | APX AIA supports Price Set Logic and custodian-specific price behavior. | APX pricing workflow. | Verified for AIA workflow; native schema Unknown. |
| SourceId | Price-source field/label observed in APX AIA price context. | N/A. | Observed in APX price-source context, not security master. | Price import/source context. | Verified for AIA context; native field status Unknown. |
| Market Value | Valuation amount. Exact formula may depend on report/security type/accrual/currency. | Report labels show market value in Axys examples. | APX report examples include Market Value labels. | Holdings/reporting context. | Verified as report label; exact calculation Unknown. |
| Accrued Interest | Interest earned but not paid, common in fixed income. | Axys fixed-income support and conversion research indicate relevance; exact fields Unknown. | Exact APX behavior Unknown. | Fixed-income pricing/income context. | Medium Confidence |
| Amortization | Fixed-income accounting adjustment over time. | Axys tracks amortization. | APX exact behavior Unknown. | Fixed income/security master/report context. | Verified for Axys. |
| Accretion | Fixed-income accounting adjustment over time. | Axys tracks accretion. | APX exact behavior Unknown. | Fixed income/security master/report context. | Verified for Axys. |
| Coupon | Bond interest term/payment. | Axys tracks fixed-income characteristics including odd coupon dates. | APX exact behavior Unknown. | Fixed income/security master/income context. | High Confidence for Axys. |
| Odd Coupon Date | Non-standard fixed-income coupon date. | Axys tracks odd coupon dates. | APX exact behavior Unknown. | Fixed income/security master context. | Verified for Axys. |

---

## 9. Performance, Attribution, Benchmark, and Composite Terms

| Term | Definition | Axys | APX | Interface / Report Context | Confidence |
|---|---|---|---|---|---|
| Performance Measurement | Calculation/reporting of investment returns. | Axys supports performance measurement. | APX includes performance measurement/analytics. | Performance reports/extracts; exact IMEX objects Unknown. | Verified generally. |
| Time-Weighted Return / TWR | Return measure intended to reduce external cash-flow impact. | Axys can calculate time-weighted returns. | APX likely supports performance returns but exact TWR support not verified in supplied material. | Performance report/export context. | Verified for Axys; Unknown/Medium for APX. |
| Internal Rate of Return / IRR | Money-weighted return measure. | Axys can calculate internal rates of return. | Unknown in supplied APX material. | Performance report/export context. | Verified for Axys; Unknown for APX. |
| Before Fees / Gross of Fees | Return before management fees. | Axys can calculate returns before fees. | APX exact behavior Unknown. | Performance option/report context. | Verified for Axys; Unknown for APX. |
| After Fees / Net of Fees | Return after fees. | Axys can calculate returns after fees. | APX exact behavior Unknown. | Performance option/report context. | Verified for Axys; Unknown for APX. |
| Performance History | Historical performance values or records. Exact storage/recalculation rules are not verified. | Axys product material references updating performance history for significant contributions/withdrawals. | APX performance history storage/recalculation Unknown. | Performance IMEX/REP/report context. | Verified concept for Axys; implementation Unknown. |
| Significant Contribution / Withdrawal | External flow large enough to affect performance treatment. | Axys product material references it. | Unknown in supplied APX material. | Performance history/report setting. | Verified for Axys concept; thresholds Unknown. |
| Benchmark | Index or benchmark used for comparison. | Axys can compare performance to indices/synthetic indices and create blended benchmarks. | APX performance reports use benchmarks in APX report-guide examples. | Performance/attribution/report context. | Verified |
| Blended Benchmark | Benchmark composed of multiple components. | Axys supports blended benchmarks and component index history. | APX exact support Unknown in supplied material. | Benchmark/performance context. | Verified for Axys; Unknown for APX. |
| Synthetic Index | User/system-created index for comparison. | Axys can compare to synthetic indices. | Unknown. | Benchmark/performance context. | Verified for Axys. |
| Index | Benchmark return series. | Axys supports index comparisons. | APX benchmark examples exist in reports. | Benchmark/performance context. | Verified generally. |
| Performance Analytics | Analytics around performance, contribution, attribution, and risk. | Axys supports performance measurement; specific analytics module details Unknown. | APX product material and reports support performance analytics. | Reports/SSRS/REP context. | Verified for APX; High Confidence for Axys broad use. |
| Attribution | Explanation of excess return by factors such as allocation/selection/classification. | Axys attribution behavior not verified in supplied material. | APX report guide includes Attribution Summary, Attribution by Classification, and Attribution by Selected Groupings. | Performance analytics reports. | Verified for APX report names/labels; Axys Unknown. |
| Contribution | Contribution to total return by security/classification/segment. | Axys exact contribution reports Unknown. | APX report guide includes Contribution Summary, Detail, and by Classification. | Performance analytics reports. | Verified for APX report names/labels; Axys Unknown. |
| Allocation Effect | Attribution component visible in APX report examples. | Unknown. | Visible in APX Attribution Summary examples. | APX report label; not proven database field. | Verified as APX report label. |
| Selection Effect | Attribution component visible in APX report examples. | Unknown. | Visible in APX Attribution Summary examples. | APX report label; not proven database field. | Verified as APX report label. |
| Total Effect | Total attribution effect visible in APX report examples. | Unknown. | Visible in APX Attribution Summary examples. | APX report label; not proven database field. | Verified as APX report label. |
| Active Return | Portfolio return minus benchmark return in report context. | Unknown exact Axys report label. | Visible in APX Attribution Summary examples. | APX report label. | Verified for APX report label. |
| Classification-Level Performance | Performance by asset class, sector, country, region, industry, or custom grouping. | Axys can display performance by asset classes, sectors, countries, or regions. | APX report guide includes classification-based attribution/contribution. | Performance/classification report context. | Verified generally. |
| Security Performance | Security-level return/contribution. | Exact Axys report/export behavior Unknown. | APX selected grouping/detail reports show security-level rows in research. | Performance report/export context. | Medium Confidence for APX report output; Unknown for exact fields. |
| Portfolio Performance | Portfolio-level return. | Axys calculates/displays portfolio performance. | APX tracks performance and reports portfolio return in APX examples. | Performance reports/exports. | Verified generally. |
| Stored Return | Return stored in data rather than calculated at report run. | Unknown. | Unknown. | Critical performance Unknown. | Unknown |
| Recalculated Return | Return calculated dynamically at report/export time. | Unknown. | Unknown. | Critical performance Unknown. | Unknown |
| Linked Return | Multi-period return produced by linking subperiod returns. | Exact Axys/APX behavior Unknown. | Exact Axys/APX behavior Unknown. | Performance chapter test target. | Unknown |

---

## 10. Classification and Reporting Dimension Terms

| Term | Definition | Axys | APX | Interface / Report Context | Confidence |
|---|---|---|---|---|---|
| Classification | Dimension used to group securities, holdings, portfolios, or performance. | Axys reports/grouping support asset class, sector, country, region, manager, objective, and custom categories at product level. | APX report guide includes custom classification/industry/sector and attribution by classification. | Classification fields may appear in reports/exports; exact storage Unknown. | Verified generally; storage Unknown. |
| Asset Class | Broad investment class such as equity, fixed income, cash. | Axys can group/report by asset class; `Asset Class` appears in an Axys export example. | APX supports multi-asset-class coverage and likely asset class reporting. | Report/export label; native field Unknown. | Verified for Axys concept/label; High Confidence for APX concept. |
| Sector | Security classification such as economic sector. | Axys can display performance by sectors. | APX report guide references sector/industry group/classification reporting. | Report dimension. | Verified generally. |
| Industry | Security classification below/related to sector. | Exact Axys field Unknown. | APX report guide/research references industry/industry sector/group in reports. | Report dimension; storage Unknown. | Medium / High Confidence depending APX report label. |
| Industry Group | Classification group in APX report guide research. | Unknown. | APX report guide snippet references industry group. | Report dimension. | Verified for APX report-guide snippet; storage Unknown. |
| Country | Geographic/security classification. | Axys can display performance by countries. | APX exact behavior Unknown from supplied material. | Report dimension. | Verified for Axys; Unknown for APX. |
| Region | Geographic classification. | Axys can display performance by regions. | APX exact behavior Unknown. | Report dimension. | Verified for Axys; Unknown for APX. |
| Custom Classification | User-defined classification scheme. | Axys custom categories supported for portfolio grouping; security-level custom classification Unknown. | APX report guide references custom classification. | Report dimension; storage Unknown. | Verified for APX report concept; Medium for Axys grouping concept. |
| Category | Flexible grouping attribute. | Axys can group portfolios by any category. | Unknown. | Portfolio/group reports. | Verified for Axys; Unknown for APX. |
| Rating | Fixed income credit/quality characteristic. | Axys tracks ratings. | APX exact behavior Unknown. | Security master/fixed income reports. | Verified for Axys. |
| Tax Status | Tax attribute. | Axys tracks tax status. | Unknown. | Security master/fixed income reports. | Verified for Axys. |
| Yield Method | Fixed income calculation attribute. | Axys tracks yield method. | Unknown. | Fixed income/security master. | Verified for Axys. |
| Duration | Fixed income interest-rate sensitivity measure. | Axys tracks duration. | Unknown. | Fixed income/security master/report context. | Verified for Axys. |
| Label | Term used in practitioner research with transaction/label imports. Relationship to classifications is not established. | Unknown relationship. | Unknown relationship. | IMEX/trade blotter context. | Medium Confidence term; classification relationship Unknown. |

---

## 11. Corporate Actions, Income, and Reconciliation Terms

| Term | Definition | Axys | APX | Interface / Report Context | Confidence |
|---|---|---|---|---|---|
| Corporate Action | Event affecting a security such as split, dividend, merger, return of capital, or reorganization. | Axys has integrated corporate actions processing at product level; `split.inf` is supported in conversion research. | APX has Advent Corporate Actions for APX workflow evidence, including ACA, Reorg Utility, and Trade Blotter. | Corporate action transactions/reports/IMEX objects Unknown. | Verified at product/workflow level; exact fields Unknown. |
| Advent Corporate Actions / ACA | SS&C Advent corporate-actions solution referenced in APX research. | Not verified for Axys integration in supplied material. | ACA for APX workflow sends holdings, cross-references securities, reviews/downloads actions, runs APX Reorg Utility, and posts transactions to APX Trade Blotter. | APX corporate-action workflow. | Verified for APX workflow. |
| Reorg Utility | APX utility run during ACA workflow. | N/A. | Runs after reviewed ACA actions are downloaded to APX, per vendor brief research. | Corporate action/Trade Blotter context. | Verified for APX ACA workflow. |
| Split | Corporate action changing share quantity basis. | Axys `split.inf` supports split evidence; exact fields Unknown. | APX split storage Unknown; ACA/reorg workflow may handle corporate actions generally. | Corporate-action/pricing/holdings context. | High Confidence for Axys file; Unknown for exact APX storage. |
| Reverse Split | Split reducing shares. | Likely represented through split factor conventions if supported, but exact evidence Unknown. | Unknown. | Corporate-action/pricing context. | Unknown / Medium as concept only. |
| Stock Dividend | Share distribution. | Exact behavior Unknown. | Exact behavior Unknown. | Corporate-action/transaction context. | Unknown |
| Cash Dividend | Cash income distribution. | Likely transaction-driven but exact codes Unknown. | Report/integration examples support dividend concepts; exact native behavior Unknown. | Income/transaction/report context. | Medium Confidence |
| Return of Capital | Distribution treated as capital return/cost-basis reduction. | Exact behavior Unknown. | Observed in APX integration code examples; official behavior Unknown. | Transaction/corporate action context. | Medium Confidence |
| Merger / Reorganization | Corporate action replacing/exchanging securities/cash. | Exact Axys process Unknown. | ACA/APX workflow supports reorg activities generally; exact postings Unknown. | Reorg Utility/Trade Blotter context. | Verified for APX workflow existence; details Unknown. |
| Spin-off | Corporate action distributing new security. | Unknown. | ACA workflow may handle complex events but exact behavior Unknown. | Corporate-action context. | Unknown / Medium as concept only. |
| Cash-in-Lieu | Cash paid for fractional shares or reorganization remainder. | Unknown. | Unknown. | Corporate-action transaction/report context. | Unknown |
| Withholding Tax | Tax withheld on income. | Axys can automatically calculate international withholding tax. | APX exact behavior Unknown. | Income/cash transaction context. | Verified for Axys; Unknown for APX. |
| Reconciliation | Comparing internal accounting records against external trades, settlement, positions, transactions, or custodian records. | Axys supports automated reconciliation of trade information, settlement data, transactions, and positions. | APX reconciliation workflows appear in AIA/position/statement blotter research, but product-level details vary. | Reports/IMEX/REP extracts may support reconciliation. | Verified for Axys; High Confidence for APX workflows. |
| Trade Information Reconciliation | Reconciliation of trade data. | Axys supports automated reconciliation of trade information. | Exact APX behavior Unknown. | Transaction/reconciliation reports. | Verified for Axys. |
| Position Reconciliation | Reconciliation of holdings/positions. | Axys supports position reconciliation; Axys 3.8.7 enhanced Position Reconciliation report. | APX AIA/CI workflows support position reconciliation and Position Blotter. | Report/import context. | Verified for Axys; Verified for APX workflows. |
| Transaction Reconciliation | Reconciliation of transactions. | Axys supports transaction reconciliation at product level. | APX statement/transaction blotter workflows support reconciliation contexts. | Reports/imports. | Verified for Axys; High Confidence for APX workflows. |
| Reconciliation Report | Report comparing internal and external values. | Axys Reconciliation report appears in conversion research; Position Reconciliation report named in version research. | APX reconciliation report names mostly Unknown. | Report/REP context. | Verified for Axys examples; APX Unknown. |

---

## 12. Field Name Glossary / Research Targets

The entries in this section are candidate field-name concepts. They are not authoritative Axys/APX native field names unless explicitly marked as observed. Use them as a research checklist when reviewing IMEX exports, REP source, report output, APX SQL/public views, or production samples.

| Candidate Field / Label | Definition | Axys | APX | Interface / Report Context | Confidence |
|---|---|---|---|---|---|
| Portfolio ID | Identifier for portfolio/account. | Exact native field Unknown. | Exact native field Unknown. | IMEX/REP/report target. | Unknown |
| Portfolio Code | Short portfolio/account code. | Observed in Axys report/export examples and CI workflow. | APX Portfolio Code observed in CI workflow. | Report/integration label. | Verified for observed contexts; exact native semantics Unknown. |
| Account Number | Account identifier; may mean PMS or custodian number depending context. | Integration concept. | Integration concept. | Mapping/reconciliation. | Medium Confidence |
| PMS Account Number | Account number from portfolio management system. | Integration concept. | Integration concept. | Data Broker mapping. | Verified as integration concept. |
| Custodian Account Number | Account number from custodian. | Integration concept. | Integration concept. | Data Broker/AIA mapping. | Verified as integration concept. |
| Security ID | Internal security identifier. | Exact field Unknown. | Exact field Unknown. | Security master/transactions/holdings. | Unknown |
| Security Symbol / Symbol | Product security symbol. | Observed as Axys Symbol in CI context. | Observed as APX Symbol in CI context. | Security matching/export/report context. | Verified for CI contexts. |
| Security Type / Type | Product security type code. | Observed in CI contexts. | Observed in CI contexts. | Security matching/export/report context. | Verified for CI contexts. |
| Security | Security name/description report/export label. | Observed in Axys asset/portfolio appraisal examples. | Observed in APX report labels. | Report/export label; native field Unknown. | Verified as label. |
| Sec Type Code | Axys export label from AdvisorEngine research. | Observed. | Unknown. | Export/report label. | Verified for that export workflow. |
| APX Security Type / APX Type | APX security type in CI translation example. | N/A. | Observed. | Integration/security matching. | Verified for CI context. |
| CUSIP | External security identifier. | Observed in matching/conversion examples. | Observed in matching examples. | Security matching; native field Unknown. | Verified as integration identifier. |
| Ticker | External market/security identifier. | Observed in matching examples. | Observed in matching examples. | Security matching; native field Unknown. | Verified as integration identifier. |
| Trade Date | Transaction execution date. | Axys supports trade-date accounting; exact export field Unknown. | APX report examples include Trade Date labels. | Transaction reports/exports. | Verified as concept/label; exact field Unknown. |
| Settle Date / Settlement Date | Transaction settlement date. | Axys supports settlement-date accounting; exact field Unknown. | APX report examples include Settle Date labels. | Transaction reports/exports. | Verified as concept/label; exact field Unknown. |
| Quantity | Units/shares/par. | Observed in Axys and APX report examples. | Observed in APX report examples. | Holdings/transaction report label. | Verified as label; exact native field Unknown. |
| Price | Security price or transaction price. | Observed in Axys holdings/price contexts. | Observed in APX Transaction Summary report labels. | Pricing/transaction/report label. | Verified as label; exact native field Unknown. |
| Unit Price | Transaction unit price label. | Unknown from Axys report samples in supplied research. | Observed in APX Transaction Summary examples. | Report label. | Verified for APX report label. |
| Market Value | Valuation amount label. | Observed in Axys Portfolio Appraisal/AUM contexts. | Observed in APX report examples. | Report label. | Verified as label; calculation Unknown. |
| Pct Assets / Percent of Portfolio | Portfolio weight/allocation label. | Axys Portfolio Appraisal sample includes Pct Assets. | APX Portfolio Appraisal description includes Percent of Portfolio. | Holdings report label. | Verified as report label. |
| Cost / Total Cost / Unit Cost | Cost basis or transaction cost labels. | Exact Axys labels vary/Unknown. | Observed in APX transaction/report examples. | Transaction/realized gain/holdings reports. | Verified as APX labels; exact native fields Unknown. |
| Cost Basis | Basis label in realized gain/loss reports. | Exact Axys output Unknown. | Observed in APX Realized Gains and Losses examples. | Report label. | Verified for APX report label. |
| Proceeds | Sale proceeds label. | Unknown Axys exact label. | Observed in APX reports. | Transaction/realized gain report label. | Verified for APX report label. |
| Gain/Loss | Realized/unrealized gain/loss label. | Axys reports likely include gain/loss but exact label Unknown. | Observed in APX reports. | Report label. | Verified for APX report label; Axys Unknown. |
| Ex-Date | Dividend ex-date label. | Unknown Axys exact label. | Observed in APX Transaction Summary dividend section research. | Report label. | Medium / Verified depending report sample. |
| Pay-Date | Dividend payment date label. | Unknown Axys exact label. | Observed in APX Transaction Summary dividend section research. | Report label. | Medium / Verified depending report sample. |
| Open Date | Lot opening date. | Unknown. | Observed in APX Realized Gains and Losses examples. | Report label. | Verified for APX report label. |
| Close Date | Lot closing/disposal date. | Unknown. | Observed in APX Realized Gains and Losses examples. | Report label. | Verified for APX report label. |
| Return | Performance return label. | Axys calculates returns; exact labels Unknown. | APX report examples include Return. | Performance report label. | Verified as report label; exact calculation Unknown. |
| Portfolio Return | Total portfolio return label. | Axys concept verified; exact label Unknown. | Observed in APX attribution/contribution examples. | Report label. | Verified for APX report label. |
| Benchmark Return | Benchmark return label. | Axys benchmark concept verified; exact label Unknown. | Observed in APX examples. | Report label. | Verified for APX report label. |
| Active Return | Portfolio-minus-benchmark return label. | Unknown. | Observed in APX examples. | Report label. | Verified for APX report label. |
| Avg Wgt | Average weight label. | Unknown. | Observed in APX attribution/contribution examples. | Report label. | Verified for APX report label. |
| Contrib | Contribution label. | Unknown. | Observed in APX attribution/contribution examples. | Report label. | Verified for APX report label. |
| Asset Class | Classification label. | Observed in Axys export/reports/product material. | Likely APX but exact label/source Unknown. | Classification/report label. | Verified for Axys label; APX Medium. |
| Industry Sector | Classification label in APX reports. | Unknown. | Observed in APX attribution/contribution examples. | Report label. | Verified for APX report label. |
| SourceId | Price source label observed in APX AIA price context. | N/A. | Observed. | Price import/source context. | Verified for AIA context only. |
| Perf/CW | Column in Axys `topost.trn` per ByAllAccounts CI research. | Observed. | Unknown. | Trade Blotter/import field. | Verified for CI context; meaning requires source documentation. |
| Mark to Market | Field/value required for non-system-currency transactions in ByAllAccounts Axys CI context. | Observed. | Unknown. | Transaction import/multicurrency context. | Verified for CI context. |
| Broker Representative Field | APX/AIA field populated from `$brok` in research. | May appear in Axys/APX integration contexts. | Observed in APX AIA workflow. | Transaction blotter field. | Medium / High Confidence for workflow. |
| Lot Location | Axys-era/APX workflow concept used for lot accounting/custodian tracking. | Described as old Axys carryover in APX AIA research. | Used in APX AIA workflow. | Lot/transaction context. | Medium Confidence |
| $:fileo | Replang token used in Axys `AMAN.REP` example to display portfolio code. | Observed. | Unknown. | REP field/token. | Verified for example only. |
| #~8portmv | Replang expression in Axys `AMAN.REP` example for portfolio market value. | Observed. | Unknown. | REP expression. | Verified for example only. |
| $askport | Axys REP variable used in CSSI Portfolio Appraisal header example. | Observed. | Unknown. | REP variable. | Verified for example only. |
| $:tfile | Axys REP variable described as transaction-source CLI file in report example. | Observed. | Unknown. | REP variable. | Verified for example only. |
| $firmg | Axys REP variable used as “Other” classification catch-all in AUM sector example. | Observed. | Unknown. | REP variable. | Verified for example only. |

---

## 13. Report Name Glossary / Research Targets

| Report / Report Family | Definition | Axys | APX | Interface / Report Context | Confidence |
|---|---|---|---|---|---|
| Assets Under Management | Management/AUM report family. | `AMAN.REP` is an Axys Assets Under Management report file in CSSI example. | APX equivalent Unknown. | REP/report. | Verified for Axys example. |
| `AMAN.REP` | Axys report file for Assets Under Management in CSSI example. | Observed. | N/A. | REP file. | Verified for example. |
| Portfolio Appraisal | Holdings/assets report. | Verified Axys report; Report Writer can add Portfolio Code column in example. | APX report guide includes Portfolio Appraisal. | Holdings report. | Verified for Axys; Verified/Medium for APX depending report-guide source. |
| Position Reconciliation Report | Report for position reconciliation. | Axys 3.8.7 enhanced this report. | APX equivalent Unknown. | Reconciliation report. | Verified for Axys named report. |
| Transaction Summary | Transaction report. | Axys equivalent/report name Unknown. | APX report guide lists Transaction Summary; sample columns include trade/settle dates, quantity, security, price, proceeds, gain/loss depending context. | APX report. | Verified for APX report name. |
| Realized Gains and Losses | Report for realized gain/loss. | Axys exact report name Unknown. | APX report guide lists Realized Gains and Losses. | APX report. | Verified for APX report name. |
| Account Distribution | Business intelligence report. | N/A. | APX report guide lists Account Distribution. | APX SSRS/report guide. | Verified |
| Account Characteristics | Business intelligence report. | N/A. | APX report guide lists Account Characteristics. | APX report. | Verified for name; fields Unknown. |
| Account Characteristics (By Custodian) | Business intelligence report. | N/A. | APX report guide lists this report. | APX report. | Verified for name; fields Unknown. |
| Asset Flows | Business intelligence report. | Unknown Axys equivalent. | APX report guide lists Asset Flows. | APX report. | Verified for name; fields Unknown. |
| Business Summary Dashboard | Dashboard-style business report. | Unknown. | APX report guide lists this report. | APX report/dashboard. | Verified for name; fields Unknown. |
| Activity Profile | Portfolio activity/analytics report. | Unknown. | APX report guide lists Activity Profile. | APX report. | Verified for name; fields Unknown. |
| Attribution by Classification | Attribution report by classification. | Unknown. | APX report guide lists this report. | APX performance analytics report. | Verified |
| Attribution Summary | Summary attribution report. | Unknown. | APX report guide lists this report and examples show Portfolio Return, Benchmark Return, Active Return, Allocation Effect, Selection Effect, Total Effect. | APX performance analytics report. | Verified |
| Attribution by Selected Groupings | Attribution report with expandable groupings. | Unknown. | APX report guide lists this report. | APX performance analytics report. | Verified |
| Contribution by Classification | Contribution report by classification. | Unknown. | APX report guide lists this report. | APX performance analytics report. | Verified |
| Contribution Summary | Summary contribution report. | Unknown. | APX report guide lists this report. | APX performance analytics report. | Verified |
| Contribution Detail | Flattened/detailed contribution report. | Unknown. | APX report guide lists this report. | APX performance analytics report. | Verified |
| Risk Statistics | Risk statistics report. | Unknown Axys equivalent. | APX report guide lists Risk Statistics. | APX performance/risk report. | Verified for name; metrics Unknown. |
| Cover Page | Client package cover page. | Unknown Axys equivalent. | APX report guide lists Cover Page. | APX client reporting. | Verified for name. |
| Household Overview | Household/client report. | Unknown. | APX report guide lists Household Overview. | APX client reporting. | Verified for name; behavior Unknown. |
| Portfolio Overview | Portfolio overview report. | Unknown Axys exact name. | APX report guide lists Portfolio Overview. | Client report. | Verified for APX name. |
| Performance Overview | Performance client report. | Unknown Axys exact name. | APX report guide lists Performance Overview. | Client report. | Verified for APX name. |
| Risk Overview | Risk client report. | Unknown. | APX report guide lists Risk Overview. | Client report. | Verified for APX name. |
| Policy Overview | Investment policy client report. | Unknown. | APX report guide lists Policy Overview. | Client report. | Verified for APX name. |
| Historical Policy Overview | Historical investment policy report. | Unknown. | APX report guide lists Historical Policy Overview. | Client report. | Verified for APX name. |
| Allocation Summary | Allocation client report. | Unknown Axys exact name. | APX report guide lists Allocation Summary. | Client report. | Verified for APX name. |
| Equity Overview | Equity client report. | Unknown. | APX report guide lists Equity Overview. | Client report. | Verified for APX name. |
| Fixed Income Distribution | Fixed-income distribution report. | Unknown. | APX report guide lists Fixed Income Distribution. | Client report. | Verified for APX name. |
| Fixed Income Overview | Fixed-income client report. | Unknown. | APX report guide lists Fixed Income Overview in APX architecture research. | Client report. | Verified for APX name. |
| Income Projection | Income projection report. | Unknown. | APX report guide lists Income Projection in APX architecture research. | Client report. | Verified for APX name. |
| Disclaimer and Terms | Disclosure/disclaimer client report. | Unknown. | APX report guide lists Disclaimer and Terms. | Client report. | Verified for APX name. |
| `CDIhold.rep` | WealthTechs-provided holdings extract report for historical holdings calculation in AIA/NBIN duplicate-handling workflow. | Used in Axys workflow. | Used in APX workflow per research. | Custom REP/report. | Verified for workflow; not standard Advent report. |

---

## 14. Version and Environment Terms

| Term | Definition | Axys | APX | Interface / Report Context | Confidence |
|---|---|---|---|---|---|
| Axys v1.x | Early Axys version line. Practitioner research says open text file structure. | Applies. | N/A. | File/IMEX architecture context. | Medium Confidence |
| Axys v2.x | Axys version line. Practitioner research says binary file format introduced. | Applies. | N/A. | Direct-file-access caution. | Medium Confidence |
| Axys v3.x | Axys version line. Practitioner research says IMEX reduced need for direct file access and supported import/export formats. | Applies. | N/A. | IMEX/version context. | Medium Confidence |
| Axys 3.7 to 3.8 Conversion | Version upgrade/file conversion described by practitioner research. | Applies. | N/A. | File-format risk. | Medium Confidence |
| Axys 3.8.6 | Minimum Axys version supported by one Data Broker connector. | Connector support only. | N/A. | REP32/client tools connector context. | Verified for connector only. |
| Axys 3.8.7 | Axys version referenced in vendor blog; enhanced Position Reconciliation report, generic date framework, and multicurrency reports. | Applies. | N/A. | Report/version context. | Verified |
| APX 3.0 | APX release reported to introduce SSRS reporting framework in APX architecture research. | N/A. | Applies historically. | APX reporting/SSRS context. | High Confidence |
| APX v1.x through v4.x | Practitioner research says APX maintained IMEX functionality but eliminated fixed-format file generation. | N/A. | Applies to those versions per practitioner source. | APX IMEX/version context. | Medium Confidence |
| APX 15.2 / 16.1 / 16.2 / 17.1 | APX versions listed by Data Broker connector as supported/tested. | N/A. | Connector support only. | REP32/client tools connector context. | Verified for connector only. |
| Current Version | Current vendor-supported Axys/APX release. | Unknown. | Unknown. | Requires SS&C release notes/customer portal. | Unknown |
| Version Difference | Behavior difference across releases. | Must be documented only when sourced. | Must be documented only when sourced. | IMEX/REP/report schemas may vary by version. | Unknown unless supported. |
| Replang Keyword Set | Available RepLang keywords/functions. | Practitioner source says Axys has roughly 100 keywords. | Practitioner source says APX adds 100+ more keywords in current versions. | REP language context. | Medium Confidence; exact list Unknown. |
| Windows 7 / Windows 10 | OS versions recommended for one Data Broker connector machine. | Connector environment only. | Connector environment only. | REP32/client tools connector context. | Verified for connector only. |

---

## 15. Known Ambiguities and Implementation Notes

| Ambiguity / Note | Axys | APX | Confidence | Repository Handling |
|---|---|---|---|---|
| Do not assume identical Axys/APX behavior. | Applies. | Applies. | Verified | Separate Axys and APX whenever evidence differs. |
| Public marketing pages are not field dictionaries. | Applies. | Applies. | Verified | Use product pages only for broad capability statements. |
| Third-party integration terms may not be Advent-native. | Applies. | Applies. | High Confidence | Label AIA/CI/Data Broker terms as workflow-specific unless vendor docs confirm native behavior. |
| Connector version support is not product lifecycle support. | Applies. | Applies. | Verified | Do not treat Data Broker minimum versions as SS&C product minimums. |
| Report labels are not necessarily database fields. | Applies. | Applies. | Verified | Treat report output labels separately from IMEX fields and database columns. |
| Security Type is not the same as Sector/Industry/Asset Class. | Applies. | Applies. | High Confidence | Security Type identifies instrument/type; classifications are reporting/grouping dimensions. |
| Code-only transaction interpretation is unsafe. | Applies. | Applies. | High Confidence | Interpret transaction codes with sign, security type, source/destination fields, context, and configuration. |
| Uppercase cancellation code pattern is workflow-supported but not universal. | Applies. | Applies. | Medium Confidence | Document examples but do not generalize without official transaction documentation. |
| Direct Axys file parsing is brittle. | Applies. | N/A. | High Confidence as caution | Prefer IMEX/REP/report exports unless file formats are verified for the installed version. |
| APX SQL/public view access does not imply complete schema visibility. | N/A. | Applies. | High Confidence | Public views may be limited; exact table/view fields require APX documentation or SQL metadata. |
| Performance stored-vs-recalculated behavior is unresolved. | Applies. | Applies. | Unknown | Keep performance storage/recalculation terms Unknown until tested or documented. |
| IMEX object names remain mostly unresolved. | Applies. | Applies. | Unknown | Do not invent object names such as transaction/security/performance objects without samples/docs. |
| REP internals remain incomplete. | Applies. | Applies. | Unknown | Only documented sample tokens should be treated as verified. |
| APX SSRS reports and legacy REP/REP32 can coexist in some environments/workflows. | Unknown exact boundaries. | Applies. | Medium / High Confidence | Document report engine per report, not globally. |

---

## 16. Unknowns to Carry Forward

| Unknown | Why It Matters | Needed Evidence |
|---|---|---|
| Exact Axys IMEX object names for transactions, holdings, security master, prices, cash, classifications, performance, composites, and corporate actions. | Required for implementation-ready integration documentation. | Axys IMEX manual, object list, screenshots, export definitions, or sample exports. |
| Exact APX IMEX object names for the same areas. | Required for APX-specific integration documentation. | APX IMEX manual, APXIX/ApxIx definitions, sample exports/imports. |
| Complete Axys transaction code matrix. | Needed for transaction glossary, audit rules, and import/export validation. | Vendor transaction code documentation, production exports, REP reports, or IMEX specs. |
| Complete APX transaction code matrix. | Needed for APX transaction/audit documentation. | APX vendor documentation or production exports. |
| Native Axys security master field dictionary. | Required for authoritative security master glossary. | Sanitized `sec.inf`, `type.inf`, IMEX export, or vendor data dictionary. |
| Native APX security master schema/public views. | Required for APX data dictionary/glossary. | APX SQL schema, public view documentation, or export samples. |
| Exact price-file layouts. | Required for pricing audit/import tooling. | Sanitized `.pri` files, APX price-file samples, IMEX price specs. |
| Exact corporate-action fields and workflow details. | Needed for split/dividend/reorg glossary and audit rules. | ACA/APX docs, `split.inf` sample, corporate-action report/export samples. |
| Performance storage versus recalculation rules. | Critical for interpreting performance history, stored returns, multi-period reports, and restatements. | Vendor performance docs, controlled tests, IMEX performance exports, report source. |
| Exact performance IMEX fields such as portfolio/security return, contribution, weights, benchmark return. | Required for Chapter 10 and glossary precision. | `portperf`, `secperf`, or equivalent verified exports if they exist. |
| Exact report source artifacts for major Axys reports. | Required to document REP/report fields and calculations. | `.REP` files, Report Writer definitions, parameter screens, output samples. |
| Exact APX SSRS report datasets/RDLs/stored procedures. | Required to document APX report internals. | APX report catalog, RDLs, SQL datasets, vendor reporting documentation. |
| Classification storage and effective-date behavior. | Needed to explain historical classification reporting. | Security/classification exports, APX public views, controlled before/after report tests. |
| Multicurrency calculation rules. | Needed for local/base currency, FX effect, and performance glossary terms. | Multicurrency reports, settings, exports, and vendor docs. |
| Fixed-income accrual/amortization/accretion formulas and fields. | Needed for fixed income glossary precision. | Fixed-income security exports, reports, and vendor accounting documentation. |
| API endpoint names and version behavior. | Needed for modern APX integration glossary. | APX REST API documentation and version notes. |

---

## 17. References

This chapter was prepared from the supplied repository blueprint and research files. The research files summarize vendor, integration, consultant, report-guide, and conversion evidence. The following supplied files were used as source material:

| Supplied File | Role in Chapter 16 |
|---|---|
| `AXYS_APX_REFERENCE_BLUEPRINT(46).md` | Governing editorial specification, confidence labels, chapter structure, and field dictionary standard. |
| `Research_02_Axys_Architecture(5).md` | Axys architecture, file orientation, IMEX, REP, version differences. |
| `Research_03_APX_Architecture(4).md` | APX architecture, SSRS, IMEX, REP32, APX report names, SQL/API options. |
| `Research_04_Security_Master(19).md` | Security master terms, `sec.inf`, `type.inf`, symbols, security types, duplicate/security translation quirks. |
| `Research_05_Transactions(18).md` | Transaction lifecycle, Trade Blotter, codes observed in integrations, reversals, fees, reinvestments. |
| `Research_06_Holdings(10).md` | Holdings, Portfolio Appraisal, position blotters, `CDIhold.rep`, group behavior. |
| `Research_07_Cash(6).md` | Cash, sweeps, intra-account journals, cash-like symbols, cash transaction contexts. |
| `Research_08_Pricing(10).md` | Pricing, `.pri`, price sets, calculated/missing/stale prices, price-source terminology. |
| `Research_09_Corporate_Actions(8).md` | Corporate actions, `split.inf`, ACA for APX, Reorg Utility, Trade Blotter. |
| `Research_10_Performance(6).md` | Performance measurement, benchmark, stored/recalculated unknowns, performance data model terms. |
| `Research_11_Classifications(7).md` | Classification terms, asset class, sector, industry, country, custom classification, storage unknowns. |
| `Research_12_IMEX(12).md` | IMEX terms, files/folders, logs, Axys/APX import/export behavior, direct-file cautions. |
| `Research_13_REP(11).md` | REP, RepLang, Report Writer Pro, REP32, `.REP`, report examples and tokens. |
| `Research_14_Reports(3).md` | APX report guide names, report families, APX report labels, report/IMEX/REP distinctions. |
| `Research_15_Data_Dictionary(2).md` | Data dictionary conventions, field family organization, source precedence, Unknown handling. |
| `Research_16_Glossary(1).md` | Glossary research base and initial term inventory. |

---

## 18. Summary

This glossary intentionally favors supported, implementation-oriented definitions over false precision. Product-level and report-level terms are the most reliable. Exact field names, IMEX object names, transaction codes, REP internals, report source names, APX SQL objects, and performance storage/recalculation rules remain Unknown unless explicitly supported by the supplied research.
