# Research Notes: IMEX

**Repository:** AXYS / APX Reference Repository  
**Research file:** `docs/axys_apx/evidence/Research_12_IMEX.md`
**Target chapter:** `docs/axys_apx/reference/Chapter_12_Imex.md`
**Prepared:** 2026-06-29  
**Governing specification:** `AXYS_APX_REFERENCE_BLUEPRINT.md`, Version 2.0  

---

## 0. Scope and Method

This file is research material for the reader-facing IMEX chapter. It is not the chapter itself. It collects factual information, examples, implementation observations, source citations, and unknowns relating to Axys/APX IMEX and related extraction/import mechanisms.

The governing editorial rule is: **do not invent Axys or APX behavior**. Every technical statement below is classified as one of:

| Classification | Meaning |
|---|---|
| Verified | Directly supported by a cited source reviewed for this research. |
| High Confidence | Strongly supported by multiple sources, established product architecture, or consistent third-party implementation evidence, but not vendor-primary for the exact statement. |
| Medium Confidence | Plausible and consistent with available evidence, but source support is indirect, partial, version-specific, or consultant-derived. |
| Unknown | Not established from available sources; requires vendor documentation, production examples, or sample exports. |

---

## 1. Executive Summary

| Topic | Research Finding | Axys | APX | Confidence | Source |
|---|---|---:|---:|---|---|
| IMEX definition | IMEX is explicitly defined in a ByAllAccounts / Custodial Integrator Axys guide as the **Axys Import/Export utility**. | Yes | Not from this source | Verified | [S5] |
| Axys IMEX import categories | A third-party Axys integration guide states that the Axys Import/Export utility is used to import requested **Transaction, Position, and Price** files into Axys. | Yes | Not from this source | Verified | [S6] |
| IMEX logs | The same Axys guide states that IMEX execution logs are retained and can be reviewed; tabs are shown for imported data types. | Yes | Not from this source | Verified | [S6] |
| APX IMEX logs | A WealthTechs APX guide includes **Advent IMEX Log** and **Advent IMEX History Log** tools and says these logs are useful for troubleshooting importing and blotter issues. | Not this source | Yes | Verified | [S9] |
| Axys source files/folders used by integrations | A CI Axys guide identifies Axys Trade Blotter `topost.trn`, client files `*.cli`, information files `sec.inf` and `type.inf`, price files `*.pri`, and IMEX log folder behavior. | Yes | No | Verified | [S7] |
| Axys direct file access risk | AdventGuru warns that Axys users may read/write underlying files but that this is not best practice because file formats can change between versions; it gives Axys 3.7 to 3.8 conversion as an example. | Yes | No | Medium Confidence | [S12] |
| APX database access | AdventGuru states APX users may query the APX database and use SQL-based tools; SS&C describes APX as centralized/scalable and connected across front/middle/back office. | No | Yes | Medium Confidence | [S2], [S12], [S14] |
| REP32 extraction | Salentica documentation for an Advent Data Broker connector says the connector uses Advent standard reports/macros, requires Advent Client Tools including `REP32.exe`, and uses the REP32 engine plus RepLang scripting/macros to extract data. | Yes | Yes | Verified for that connector | [S10], [S11] |
| Report language | Consultant documentation states Axys reports are written in Advent’s proprietary Report Writing Language, Replang; AdventGuru says Replang remains part of Axys/APX reporting architecture. | Yes | Yes | High Confidence | [S15], [S14] |
| Report file extensions | AdventGuru states Report Writer-created reports have `.RPW` extension and many manually coded reports use `.REP`. | Yes | Yes | Medium Confidence | [S13] |
| Fixed-format APX IMEX | AdventGuru states APX v1.x to v4.x maintained IMEX functionality, but fixed-format file generation was eliminated; it also says APX can export Axys v3 format. | Not applicable | Yes | Medium Confidence | [S12] |
| Standard report examples | Evidence includes Axys `AMAN.REP` Assets Under Management and a custom `CDIhold.rep` holdings extract report. | Yes | Unknown | Verified for cited examples | [S15], [S8] |

---

## 2. Source Register

| ID | Source | Type | Relevance | Confidence Notes |
|---|---|---|---|---|
| S1 | SS&C Advent Axys product page, `advent.com/solutions/axys/` | Vendor product page | Confirms Axys scope: portfolio reporting/accounting, performance measurement, reporting, hundreds of pre-defined reports, Report Writer Pro. | Verified for marketing/product capability, not low-level IMEX behavior. |
| S2 | SS&C Advent Portfolio Exchange product page, `advent.com/solutions/advent-portfolio-exchange/` | Vendor product page | Confirms APX integrated platform, standard reports, performance analytics, flexible custom reporting, GIPS composite support. | Verified for product positioning, not low-level IMEX behavior. |
| S3 | SS&C Axys product brief PDF, `PB_AXYS.pdf` | Vendor product brief | Confirms Axys reporting/customization claims similar to product page. | Verified for high-level reporting only. |
| S4 | APX reports guide PDF, `REP_APX.pdf` | Vendor PDF | Search result confirms APX report guide exists; open failed through browsing tool. | Unknown until obtained/read directly. |
| S5 | ByAllAccounts Custodial Integrator Axys User Guide PDF | Third-party integration guide | Defines CI and IMEX; states IMEX = Axys Import/Export utility. | Verified for that guide’s terminology. |
| S6 | ByAllAccounts Custodial Integrator Axys User Guide PDF | Third-party integration guide | Describes Axys IMEX import of Transaction, Position, and Price files, IMEX logs, and error checking. | Verified for CI’s Axys workflow. |
| S7 | ByAllAccounts Custodial Integrator Axys User Guide PDF | Third-party integration guide | Lists Axys folders/files used by CI: Trade Blotter, `*.cli`, `sec.inf`, `type.inf`, `*.pri`, log folder. | Verified for CI’s Axys configuration. |
| S8 | WealthTechs AIA User Manual for Axys Users PDF | Third-party integration guide | Mentions custom `CDIhold.rep`, Axys custom label `$pathCDI`, holdings extract folder, position files. | Verified for AIA workflow; not necessarily native Axys standard. |
| S9 | WealthTechs AIA User Manual for APX Users PDF | Third-party integration guide | Mentions APX blotters, transaction translation, APX IMEX log/history log, tools for positions and cancel transactions. | Verified for AIA workflow. |
| S10 | Salentica Elements Data Broker: SS&C Advent APX & Axys | Third-party connector documentation | States Axys/APX send data to Data Broker; connector uses standard reports/macros and REP32 installed on client machine. | Verified for Data Broker connector. |
| S11 | Salentica Engage Data Broker: SS&C Advent APX & Axys | Third-party connector documentation | Lists minimum supported versions; says connector uses REP32 engine, RepLang scripting and macros. | Verified for Data Broker connector. |
| S12 | AdventGuru IMEX article/tag page | Consultant / expert blog | Discusses Axys direct files vs IMEX/report/export options, APX IMEX/version behavior. | Medium confidence; consultant-derived. |
| S13 | AdventGuru REP article/tag page | Consultant / expert blog | Discusses Report Writer Pro, `.RPW`, `.REP`, checksum limitation, Replang modifications. | Medium confidence; consultant-derived. |
| S14 | AdventGuru APX reporting article/category page | Consultant / expert blog | Discusses APX SQL Server database, public views, SSRS, REST API, Replang in Axys/APX. | Medium confidence; consultant-derived, partly current as of 2024. |
| S15 | “How to Add Portfolio Code to Axys Reports” PDF | Consultant tutorial | States Axys reports are written in Replang; shows AMAN.REP path/name and report menu context. | Medium confidence; consultant-derived, direct example. |
| S16 | WealthTechs APX AIA guide | Third-party integration guide | Shows APX Transaction Translation and blotter behavior examples. | Verified for AIA workflow only. |

---

## 3. IMEX Overview

### 3.1 Definition

| Statement | Axys | APX | Confidence | Evidence |
|---|---:|---:|---|---|
| IMEX is an Advent/Axys import/export mechanism; one third-party guide explicitly expands IMEX as “Axys Import/Export utility.” | Yes | Unknown | Verified for Axys terminology | ByAllAccounts guide terminology section: “IMEX – acronym for the Axys Import/Export utility.” [S5] |
| The name “Advent IMEX” is used in APX-related integration tooling, including “Advent IMEX Log” and “Advent IMEX History Log.” | Unknown | Yes | Verified for APX AIA tooling | WealthTechs APX guide. [S9] |
| Whether APX’s IMEX implementation is technically identical to Axys IMEX is not established from the reviewed sources. | Unknown | Unknown | Unknown | Requires SS&C APX IMEX documentation or sample APX IMEX `.inf`/log files. |

### 3.2 IMEX vs REP vs Direct File Access

| Mechanism | Axys | APX | Description | Confidence | Evidence |
|---|---:|---:|---|---|---|
| IMEX | Yes | Yes, but behavior/version details incomplete | Import/export utility or tool used in integration workflows. | Verified for Axys; Medium for APX | [S5], [S6], [S9], [S12] |
| REP / REP32 / Replang | Yes | Yes | Report engine and report language used for extraction, standard reports, macros, and custom reports. | High Confidence | [S10], [S11], [S13], [S14], [S15] |
| Report Writer Pro | Yes | Yes likely, but exact APX support/version scope not fully verified | GUI/custom reporting tool that generates report logic; source may be modifiable as Replang with caveats. | Verified for Axys high-level; Medium for APX | [S1], [S13], [S14] |
| Direct Axys file read/write | Yes | No | Possible for knowledgeable users but not best practice because file formats may change across versions. | Medium Confidence | [S12] |
| SQL/database querying | No for Axys native files | Yes | APX users may query APX database and use SQL-based tools according to consultant source. | Medium Confidence | [S12], [S14] |

---

## 4. Axys Findings

### 4.1 Axys Product and Reporting Context

| Statement | Confidence | Evidence |
|---|---|---|
| Axys is marketed by SS&C Advent as portfolio reporting and accounting software with an extensive library of pre-defined reports and report customization. | Verified | SS&C Axys page says Axys automates portfolio reporting/accounting and has an extensive library of pre-defined reports plus customization. [S1] |
| SS&C lists Axys capabilities including GIPS-compliant performance measurement, performance measurement, reporting, automated reconciliation, and support for cash/equities/fixed income. | Verified | SS&C Axys page. [S1] |
| SS&C states Axys includes hundreds of pre-defined reports and Axys Report Writer Pro. | Verified | SS&C Axys page/product brief. [S1], [S3] |

### 4.2 Axys IMEX Import Behavior From CI Evidence

The best reviewed low-level Axys IMEX evidence comes from the ByAllAccounts Custodial Integrator guide. The guide describes a third-party workflow, not a general SS&C IMEX manual. Therefore, statements below are **Verified for that workflow** but should not be generalized to every Axys IMEX use without additional vendor documentation.

| Statement | Confidence | Evidence |
|---|---|---|
| CI uses the Axys Import/Export utility to import requested Transaction, Position, and Price files into Axys. | Verified | [S6] |
| CI retains the log of the Axys Import/Export utility execution and makes it available through a View IMEX Logs dialog. | Verified | [S6] |
| CI’s View IMEX Logs dialog contains one tab for each imported data type. | Verified | [S6] |
| If position lots are used in CI, the IMEX log tab is `imexPositionLots.log` instead of `imexPositions.log`. | Verified | [S6] |
| If prices are requested for more than the prior business day, CI shows one `imexPrices` tab per historical day delivered. | Verified | [S6] |
| If a target Axys file is open/in use during import, the import may fail; the CI guide gives the example of a price file open in Axys causing an error in `imexPrices.log`. | Verified | [S6] |
| IMEX import should be followed by review of logs/errors before accepting exported data in the CI workflow. | Verified | [S6] |

### 4.3 Axys Files/Folders Relevant to IMEX and Integrations

| File / Folder / Label | Description | Confidence | Evidence / Notes |
|---|---|---|---|
| `topost.trn` | Axys Trade Blotter file in the user folder; CI appends transactions to this file, leaving existing transactions unchanged. | Verified for CI | [S7] |
| `$pathtrn` | User folder used by CI for Axys Trade Blotter output. | Verified for CI | [S7] |
| `$pathcli` | Client folder where Axys portfolio `*.cli` files are stored; CI traverses this folder and subfolders to create portfolio code list. | Verified for CI | [S7] |
| `*.cli` | Axys portfolio files. | Verified for CI; also supported by other third-party PA source | [S7] |
| `$pathinf` | Information folder where Axys `sec.inf` and `type.inf` files are stored. | Verified for CI | [S7] |
| `sec.inf` | Axys Security Information file; CI exports it to generate transaction/position files for import and uses symbol/security type information and other security record data. | Verified for CI | [S7] |
| `type.inf` | Axys Security Type Information file; CI exports it to generate transaction/position files for import. | Verified for CI | [S7] |
| `$pathpri` | Price folder where Axys `*.pri` files are stored; CI exports price files for merge purposes and imports generated price files to this folder. | Verified for CI | [S7] |
| `*.pri` | Axys price files. | Verified for CI | [S7] |
| `$pathlog` | Folder where Axys Import/Export log files are written; if custom label not defined in Axys, CI guide says it defaults to CI working directory. | Verified for CI | [S7] |
| `ptopost.trn` | Position file written by CI to `\CI\exported\ptopost.trn`; guide says it is CSV format and may contain lot-specific data when enabled/available. | Verified for CI | [S7] |
| `.pos` files | Axys replacement position files created by Position Post when Post Positions to Axys is checked. | Verified for CI | [S6] |
| `CDIhold.rep` | Custom Axys report supplied by WealthTechs to calculate holdings for AIA; added to custom report menu. | Verified for AIA | [S8] |
| `$pathCDI` | Custom Axys label used by AIA to map a network path for holdings extract. | Verified for AIA | [S8] |

### 4.4 Axys Transaction/Position/Price Processing From CI Evidence

| Area | Statement | Confidence | Evidence |
|---|---|---|---|
| Transactions | CI can export transactions to Axys Trade Blotter in the User Folder `$pathtrn`. | Verified for CI | [S7] |
| Transactions | CI can include source transaction information as a comment in Trade Blotter; these comments do not post to Axys client files, according to the guide. | Verified for CI | [S7] |
| Transactions | CI’s Accept step updates its internal transaction counter so the next run only downloads transactions not already downloaded. If Accept is not completed, the same transactions plus new ones may download next time. | Verified for CI | [S6] |
| Positions | CI can retrieve positions for prior business day and write them to `\CI\exported\ptopost.trn` in CSV format. | Verified for CI | [S7] |
| Positions | If lots are enabled and selected, `ptopost.trn` can contain lot-specific data where lots are available. | Verified for CI | [S7] |
| Positions | CI can import positions to Axys `.pos` files, replacing the `.pos` file for mapped portfolios if Post Positions to Axys is checked. | Verified for CI | [S7] |
| Positions | Position Post creates a replacement `.pos` file in the Axys position folder for each configured portfolio. | Verified for CI | [S6] |
| Prices | CI exports price files from `$pathpri` for merge purposes and imports generated price files back to `$pathpri`. | Verified for CI | [S7] |
| Prices | If an Axys price file is open when CI imports, import can fail and the error appears in `imexPrices.log`. | Verified for CI | [S6] |
| Security master | CI uses `sec.inf` and `type.inf` but states it does **not** modify Axys Security `sec.inf` or Security Type `type.inf` information. | Verified for CI | [S6], [S7] |

### 4.5 Axys Security Translation / Matching Quirks

| Statement | Confidence | Evidence |
|---|---|---|
| CI will not import data into Axys unless all securities resolve to Axys securities. | Verified for CI | [S5] |
| CI identifies untranslated securities where no corresponding Axys security is found. | Verified for CI | [S5] |
| CI identifies duplicated securities where more than one Axys security is found. | Verified for CI | [S5] |
| Duplicate/security ambiguity examples include the same symbol used more than once with different security types, an Axys security defined once with ticker and again with CUSIP, or overlapping CI security translations. | Verified for CI | [S5] |
| In one guide example, Axys symbols and security types shown include `ktc csus`, `ktc adus`, `tfus`, and `oaus`; these should be treated as examples only, not a complete list of valid types. | Verified for CI example only | [S5] |
| CI supports global and account-specific security translations; account-specific translations are for rare cases where securities with the same identifying information are different across accounts. | Verified for CI | [S5] |
| If an account-specific security translation is created for a security, the guide says the user will be required to establish account-specific translations for each account containing that security and cannot establish a global security translation for that security. | Verified for CI | [S5] |

### 4.6 Axys Report / REP Evidence

| Statement | Confidence | Evidence |
|---|---|---|
| A consultant tutorial states Axys reports are written in Replang, Advent’s proprietary Report Writing Language. | Medium Confidence | [S15] |
| The same tutorial shows the standard Assets Under Management report file name as `AMAN.REP`. | Medium Confidence | [S15] |
| The tutorial shows an example Axys reports directory `e:\axys34\rep`; this is an example path, not a universal installation path. | Medium Confidence | [S15] |
| WealthTechs AIA guide describes adding a custom report `CDIhold.rep` to the custom report menu in Axys for holdings extraction. | Verified for AIA | [S8] |
| Salentica connector documentation states extraction can use Advent standard reports and macros, with Advent client tools including `REP32.exe` installed on the client machine. | Verified for that connector | [S10], [S11] |

### 4.7 Axys Version Differences and File Format Risks

| Statement | Confidence | Evidence |
|---|---|---|
| AdventGuru states direct reading/writing of Axys files is not best practice because file formats can change between versions. | Medium Confidence | [S12] |
| AdventGuru gives an example that upgrading from Axys 3.7 to 3.8 requires file conversion and some resulting Axys 3.8 files have different formats. | Medium Confidence | [S12] |
| Salentica Data Broker documentation lists Advent Axys 3.8.6 as the minimum supported Axys version for that connector. | Verified for that connector | [S11] |
| Whether IMEX file schemas differ among Axys 3.7, 3.8, and later versions is Unknown from reviewed sources. | Unknown | Requires SS&C release notes or IMEX manuals. |

---

## 5. APX Findings

### 5.1 APX Product and Reporting Context

| Statement | Confidence | Evidence |
|---|---|---|
| SS&C describes APX as an integrated portfolio and client management platform spanning front, middle, and back office. | Verified | [S2] |
| SS&C lists APX features including a large library of standard reports, automated report packaging, performance analytics, flexible custom reporting, composite management support for GIPS compliance, and multi-currency/multi-asset coverage. | Verified | [S2] |
| SS&C describes APX as simplifying operations with a centralized, scalable platform for portfolio, relationship, and prospect data. | Verified | [S2] |

### 5.2 APX IMEX / Importing Evidence

| Statement | Confidence | Evidence |
|---|---|---|
| WealthTechs AIA for APX has an **Advent IMEX Log** that shows the log for the last importing done through the IMEX tool and is useful for troubleshooting importing issues. | Verified for AIA | [S9] |
| WealthTechs AIA for APX has an **Advent IMEX History Log** that shows logs for all importing done through the IMEX tool and is useful for troubleshooting importing and blotter issues. | Verified for AIA | [S9] |
| The reviewed APX AIA guide uses APX blotters for transactions/positions/accounts/tax lots/etc. | Verified for AIA | [S9], [S16] |
| Whether APX’s IMEX command-line interface, `.inf` control files, supported objects, or field lists match Axys IMEX is Unknown from reviewed sources. | Unknown | Requires APX IMEX manual or samples. |

### 5.3 APX Blotters and Translation Workflow Evidence

| Statement | Confidence | Evidence |
|---|---|---|
| WealthTechs APX guide lists or discusses AIA/APX blotters including Account Blotter, Initial Transaction Blotter, Trade Blotter, Position Blotter, Statement Blotter, Tax Lot Blotter, Pending Blotters, and Dividend Adjustment Blotter. | Verified for AIA | [S9], [S16] |
| AIA Account Blotter imports account demographic data from custodians such as Account Number, Account Name, Account Type and may update existing account data and/or add new accounts to APX. | Verified for AIA | [S9] |
| AIA Initial Transaction Blotter can be driven by a process setting to create initial deliver-in transactions from positions when the APX account has no transactions. | Verified for AIA | [S9] |
| AIA Transaction Translation supports IF/THEN style logic assigned to source-data transaction rows and is not case sensitive, according to the guide. Treat that as AIA translation-rule behavior, not as a general ppar rule for case-insensitive transaction-code or security-identifier matching. | Verified for AIA | [S16] |
| AIA Transaction Translation can use special assignment tokens such as `[*-1]`, `[TradeDate]`, and `[SettleDate]`. | Verified for AIA | [S16] |
| An AIA example says transaction codes `BY`, `SL`, `SS`, and `CS` can be deleted from the Trade Blotter by a Transaction Translation rule. Treat these as example transaction codes in AIA/APX import context, not a complete APX transaction code dictionary. | Verified for AIA example | [S16] |
| AIA Vehicle Filter can eliminate transactions, prices, and positions for specific vehicles from appearing in any APX blotter or import output. | Verified for AIA | [S16] |
| AIA Tools Menu includes a Delete all Positions tool, and the guide says it should be done before posting current position files in APX. | Verified for AIA | [S9] |
| AIA Cancel Transactions can create a trade blotter to cancel previously posted transactions from historical transaction files in its archive folder; the guide warns to use with caution and back up accounts. | Verified for AIA | [S9] |

### 5.4 APX Database / Reporting / Extraction Evidence

| Statement | Confidence | Evidence |
|---|---|---|
| AdventGuru states APX users may query the APX database via Excel and other programs, write SSRS or Crystal reports, and use SQL-based tools to export/import selected data. | Medium Confidence | [S12] |
| AdventGuru states APX users can tap APX’s SQL Server database using Stored Accounting Functions, Public Views, SSRS, REST API, and other tools that use that infrastructure. | Medium Confidence | [S14] |
| Salentica Data Broker connector documentation says the connector supports Advent APX versions 15.2, 16.1, 16.2, and 17.1 for that connector. | Verified for that connector | [S11] |
| Salentica documentation says the connector uses Advent Client Tools, `REP32.exe`, REP32 engine, RepLang scripting, and macros to extract data. | Verified for that connector | [S10], [S11] |
| Whether APX stores or recalculates each specific performance measure during IMEX/REP extraction is Unknown from reviewed sources. | Unknown | Requires APX data dictionary, stored accounting function docs, or sample report behavior. |

### 5.5 APX Version Differences

| Statement | Confidence | Evidence |
|---|---|---|
| AdventGuru states APX v1.x through v4.x maintained IMEX functionality, but fixed-format file generation was eliminated. | Medium Confidence | [S12] |
| AdventGuru states APX can export data to Axys v3 format. | Medium Confidence | [S12] |
| Salentica lists APX 15.2/16.1/16.2/17.1 as minimum supported versions for its Data Broker connector. | Verified for that connector | [S11] |
| The mapping between APX v1.x–v4.x and later APX 15.x–17.x version behavior is Unknown from reviewed sources. | Unknown | Requires SS&C APX release/version lineage documentation. |

---

## 6. IMEX Data Model and Field Names

### 6.1 Verified Axys Field Names From Reviewed Sources

The reviewed public sources do **not** provide a full SS&C IMEX data dictionary. The table below lists field names and file names directly found in sources.

| Field / Name | Description | Axys | APX | IMEX | REP | Confidence | Evidence |
|---|---|---:|---:|---:|---:|---|---|
| Symbol | Symbol used to identify a security; missing prices file says it may be any Axys security symbol defined in security master. | Yes | Unknown | Related | No | Verified for CI missing-prices output | [S5] |
| Type | Security type associated with Symbol; missing-prices file says it may be any Axys security type defined in security master. | Yes | Unknown | Related | No | Verified for CI missing-prices/security translation output | [S5] |
| Name | Name of security/position with no price. | Yes | Unknown | Related | No | Verified for CI missing-prices output | [S5] |
| WP Account | WebPortfolio account name/nickname in CI missing-prices output. | Integration-specific | No | No | No | Verified for CI | [S5] |
| Institution | Financial institution where security is held. | Integration-specific | Unknown | No | No | Verified for CI | [S5] |
| WP Name | Security name as it appears in WebPortfolio. | Integration-specific | No | No | No | Verified for CI security translation file | [S5] |
| WP Ticker | WebPortfolio ticker symbol. | Integration-specific | No | No | No | Verified for CI security translation file | [S5] |
| WP Cusip | WebPortfolio CUSIP. | Integration-specific | No | No | No | Verified for CI security translation file | [S5] |
| WP Account # | Account number as it appears in WebPortfolio if translation is account-specific. | Integration-specific | No | No | No | Verified for CI security translation file | [S5] |
| Axys Symbol | Output-side Axys security symbol. | Yes | No | Related | No | Verified for CI security translation file | [S5] |
| Account Number | APX account demographic field example in AIA Account Blotter. | Unknown | Yes | Related | No | Verified for AIA APX | [S9] |
| Account Name | APX account demographic field example in AIA Account Blotter. | Unknown | Yes | Related | No | Verified for AIA APX | [S9] |
| Account Type | APX account demographic field example in AIA Account Blotter. | Unknown | Yes | Related | No | Verified for AIA APX | [S9] |
| Custodian Account Number | AIA/APX account filter field; guide warns to use custodian account number, not APX Portfolio Code, when filtering a specific account. | Unknown | Yes | Related | No | Verified for AIA APX | [S9] |
| APX Portfolio Code | APX portfolio identifier referenced by AIA guide as distinct from custodian account number. | Unknown | Yes | Related | No | Verified for AIA APX | [S9] |
| Portfolio Code | Axys portfolio/account code list created by CI from `*.cli`; also translation errors exist in CI. | Yes | Unknown | Related | Yes possibly | Verified for CI | [S6], [S7] |

### 6.2 Missing Prices File — CI / Axys

This is **not** a native SS&C IMEX export definition. It is a CI-generated optional output file useful for identifying securities missing prices.

| Col # | Column Header | Required | Data Type | Description | Confidence | Evidence |
|---:|---|---:|---|---|---|---|
| 1 | Symbol | Yes | CHAR512 | Symbol used to identify this security; may be any Axys security symbol defined in security master. | Verified for CI | [S5] |
| 2 | Type | Yes | CHAR6 | Type of symbol in Symbol field; may be any Axys security type defined in security master. | Verified for CI | [S5] |
| 3 | Name | No | CHAR128 | Name of security/position that has no price. | Verified for CI | [S5] |
| 4 | WP Account | Yes | CHAR64 | WebPortfolio account nickname/name for reference. | Verified for CI | [S5] |
| 5 | Institution | No | CHAR128 | Institution where security is held. | Verified for CI | [S5] |

### 6.3 Security Translations File — CI / Axys

This is **not** a native SS&C IMEX file definition. It is a CI-generated file listing security translations maintained in CI.

| Col # | Column Header | Required | Data Type | Description | Confidence | Evidence |
|---:|---|---:|---|---|---|---|
| 1 | WP Name | Required/asterisk in guide | CHAR128 | Name of security as it appears in WebPortfolio. | Verified for CI | [S5] |
| 2 | WP Ticker | Required/asterisk in guide | CHAR6 | WebPortfolio ticker if available. | Verified for CI | [S5] |
| 3 | WP Cusip | Required/asterisk in guide | CHAR9 | WebPortfolio CUSIP if available. | Verified for CI | [S5] |
| 4 | Institution | No | CHAR128 | Institution where security is held. | Verified for CI | [S5] |
| 5 | WP Account # | No | CHAR128 | WebPortfolio account number if translation is account-specific. | Verified for CI | [S5] |
| 6 | Axys Symbol | Yes | CHAR512 | Symbol used to identify this security; product-specific. | Verified for CI | [S5] |
| 7 | Type | Yes | CHAR6 | Type of symbol in Symbol field. | Verified for CI | [S5] |

### 6.4 Native IMEX Field Dictionary Status

| Question | Status | Confidence | Needed Evidence |
|---|---|---|---|
| Which IMEX object exports transactions? | Unknown from reviewed public sources. CI imports transactions through Axys Trade Blotter / IMEX workflow, but native IMEX object name is not verified. | Unknown | Axys/APX IMEX manual; `.inf` control files; sample logs. |
| Which IMEX object exports holdings/positions? | Unknown. CI uses Position Post and imports position files, but native object names are not verified. | Unknown | IMEX manual; sample position import/export files. |
| Which IMEX object exports security master? | Unknown. CI exports Axys `sec.inf` and `type.inf` for its own use; native IMEX object names are not verified. | Unknown | IMEX manual; sample `sec.inf` export control file. |
| Which fields are required by native IMEX transaction import? | Unknown. | Unknown | Vendor IMEX data dictionary and sample valid/invalid imports. |
| Does IMEX import prices by `.pri` file date/security type/symbol? | Partially observed for CI; native field requirements unknown. | Unknown | IMEX price import examples and logs. |
| Are Axys and APX IMEX field names identical? | Unknown. | Unknown | Axys and APX IMEX manuals or paired sample exports. |

---

## 7. REP / Report-Based Extraction Findings Related to IMEX Chapter

REP is technically a separate chapter (`13-REP.md`), but it is important in `12-IMEX.md` because many practical “export” workflows use reports/macros rather than IMEX.

### 7.1 REP32 and Macros

| Statement | Axys | APX | Confidence | Evidence |
|---|---:|---:|---|---|
| A Data Broker Advent connector uses Advent standard reports and macros to generate a data extract. | Yes | Yes | Verified for connector | [S10] |
| The connector requires Advent client tools, specifically `REP32.exe`, installed on the machine. | Yes | Yes | Verified for connector | [S10], [S11] |
| The connector uses the REP32 engine to extract data and uses RepLang scripting and macros. | Yes | Yes | Verified for connector | [S11] |
| REP/report extraction can be scheduled/unattended for daily feeds in that connector context. | Yes | Yes | Verified for connector | [S11] |

### 7.2 Replang / Report Writer Pro

| Statement | Axys | APX | Confidence | Evidence |
|---|---:|---:|---|---|
| Consultant sources describe Replang as Advent’s proprietary report writing language. | Yes | Yes | High Confidence | [S15], [S14] |
| Report Writer Pro-generated reports have `.RPW` extension; underlying format is Replang according to AdventGuru. | Yes | Yes | Medium Confidence | [S13] |
| Manually modifying Report Writer-generated code can prevent future modification in Report Writer because the Report Writer checksum no longer matches. | Yes | Yes | Medium Confidence | [S13] |
| AdventGuru advises backing up the `.RPW`, saving modified code as `.REP`, and reapplying manual changes after future Report Writer edits if needed. | Yes | Yes | Medium Confidence | [S13] |
| Replang can be used to generate CSV, text formats, and Advent file formats according to AdventGuru. | Yes | Yes | Medium Confidence | [S12] |

### 7.3 Report Examples

| Report / File | System | Purpose in Source | Confidence | Evidence |
|---|---|---|---|---|
| `AMAN.REP` | Axys | Assets Under Management report; tutorial copies it before modifying to add portfolio code. | Medium Confidence | [S15] |
| `CDIhold.rep` | Axys | Custom WealthTechs AIA holdings extract report. | Verified for AIA | [S8] |
| `sipos30` | Axys | Custom report cited by CI as a reconciliation report comparing calculated positions in Axys versus downloaded positions from custodian. | Verified for CI | [S7] |
| APX standard reports | APX | SS&C says APX includes large library of standard reports; specific names not verified from opened sources. | Verified high-level; report names Unknown | [S2] |

---

## 8. Processing Behavior and Workflow Examples

### 8.1 Axys CI Daily Workflow

**Classification:** Verified for ByAllAccounts Custodial Integrator workflow.

1. Review account status to identify accounts failing to update or lacking prior-day data. [S6]
2. Import/download data from WebPortfolio. [S6]
3. Resolve portfolio code translation errors. [S6]
4. Resolve stale account data issues. [S6]
5. Review price preview. [S6]
6. Resolve untranslated securities by entering the security in Axys and reloading Axys security data into CI, or by creating a CI security translation. [S6]
7. Resolve duplicate securities by creating a security translation. [S6]
8. Export/format/deliver data into Axys. [S6]
9. Review IMEX logs and associated export information. [S6]
10. Accept exported data, which updates CI’s transaction counter for later downloads. [S6]

### 8.2 Axys Import Log Review Example

| Condition | Expected Behavior in Source | Confidence | Evidence |
|---|---|---|---|
| Import requested for transactions, positions, and prices | IMEX logs have tabs for imported data types. | Verified for CI | [S6] |
| Position lots enabled | `imexPositionLots.log` tab used instead of `imexPositions.log`. | Verified for CI | [S6] |
| Multiple historical price dates requested | One `imexPrices` tab per historical price day. | Verified for CI | [S6] |
| Axys price file open/in use | Price import can fail; error appears in `imexPrices.log`. | Verified for CI | [S6] |

### 8.3 APX AIA Import/Blotter Troubleshooting Example

| Tool / Feature | Behavior in Source | Confidence | Evidence |
|---|---|---|---|
| Advent IMEX Log | Shows log for last importing done through IMEX tool; useful for troubleshooting importing issues. | Verified for AIA | [S9] |
| Advent IMEX History Log | Shows logs for all importing done through IMEX tool; useful for troubleshooting importing and blotter issues. | Verified for AIA | [S9] |
| Cancel Transactions | Creates a trade blotter to cancel previously posted transactions from historical transaction files; use caution and back up accounts. | Verified for AIA | [S9] |

### 8.4 APX AIA Cancel Transaction Example

The APX AIA guide gives an example where a previously posted transaction:

```csv
acct123,010101,010101,by,csus,appl,100,caus,cash,10000
```

becomes:

```csv
acct123,010101,010101,BY,csus,appl,100,caus,cash,10000
```

**Interpretation:** The source appears to distinguish lowercase and uppercase transaction codes in the cancellation workflow. The exact APX semantic rule behind that case change is not established from the source alone. Treat this as **Verified example syntax**, but **Unknown native APX rule** until validated against APX documentation or test data. [S9]

---

## 9. Implementation Quirks and Operational Cautions

| Quirk / Caution | Axys | APX | Confidence | Evidence / Notes |
|---|---:|---:|---|---|
| File-in-use can break imports | Yes | Unknown | Verified for CI/Axys | CI guide says import may fail if file is in use; price file open in Axys is example. [S6] |
| Direct Axys file access is risky across versions | Yes | No | Medium Confidence | AdventGuru warns against reading/writing raw files because formats can change, e.g., 3.7 to 3.8 conversion. [S12] |
| Security master ambiguity blocks import | Yes | Unknown | Verified for CI/Axys | CI will not import unless securities resolve; duplicate symbols/security types can cause ambiguity. [S5] |
| Comments in CI Trade Blotter do not post to client files | Yes | No | Verified for CI/Axys | CI guide. [S7] |
| Position import can replace existing `.pos` files | Yes | No | Verified for CI/Axys | CI guide says mapped portfolio `.pos` file is replaced when Post Positions to Axys is checked. [S7] |
| Accept/export workflow matters for duplicate transaction downloads | Yes | No | Verified for CI/Axys | Without Accept, same transactions plus new ones may download next time. [S6] |
| Report Writer manual code changes can break future GUI modification | Yes | Yes | Medium Confidence | AdventGuru checksum discussion. [S13] |
| REP32/RepLang extraction requires installed Advent client tools in connector workflows | Yes | Yes | Verified for connector | Salentica connector docs. [S10], [S11] |
| APX has more database/reporting options than Axys | No | Yes | Medium Confidence | AdventGuru states APX has SQL Server, Stored Accounting Functions, Public Views, SSRS, REST API. [S14] |
| APX fixed-format IMEX export eliminated in APX v1.x–v4.x | No | Yes | Medium Confidence | AdventGuru. [S12] |

---

## 10. Version Differences and Support Matrix Evidence

| System | Version / Range | Statement | Confidence | Evidence |
|---|---|---|---|---|
| Axys | 3.7 to 3.8 | Upgrade requires file conversion; some 3.8 files have different file formats, according to AdventGuru. | Medium Confidence | [S12] |
| Axys | 3.8.6 | Minimum supported version for Salentica Data Broker connector. | Verified for that connector | [S11] |
| APX | v1.x to v4.x | Maintained IMEX functionality, but fixed-format file generation eliminated; can export Axys v3 format. | Medium Confidence | [S12] |
| APX | 15.2 / 16.1 / 16.2 / 17.1 | Supported/tested versions for Salentica Data Broker connector. | Verified for that connector | [S11] |
| APX | Current / later versions | Exact IMEX behavior and fields are Unknown. | Unknown | Need SS&C docs/samples. |

---

## 11. Examples for Future Chapter

### 11.1 Axys IMEX-Adjacent File/Folder Example

```text
$pathtrn      User folder / trade blotter location
    topost.trn       Axys Trade Blotter file used by CI

$pathcli      Client portfolio folder
    *.cli            Axys portfolio files

$pathinf      Information folder
    sec.inf          Security information
    type.inf         Security type information

$pathpri      Price folder
    *.pri            Price files

$pathlog      IMEX log folder
    imexTransactions.log      Example pattern inferred from CI tab naming; exact native name Unknown
    imexPositions.log         Named in CI guide
    imexPositionLots.log      Named in CI guide
    imexPrices.log            Named in CI guide
```

**Classification:** File/folder names are Verified for CI sources except `imexTransactions.log`, which is listed as an inferred example pattern and remains **Unknown** until an actual log sample is obtained.

### 11.2 Axys CI Security Translation Example Concepts

```text
Incoming custodian/WebPortfolio security identifiers:
    WP Name
    WP Ticker
    WP Cusip
    Institution
    WP Account #    # only for account-specific translation

Output Axys identifiers:
    Axys Symbol
    Type
```

**Classification:** Verified for CI security translation file, not native IMEX. [S5]

### 11.3 APX AIA Transaction Translation Example Concepts

```text
Transaction Translation:
    IF source-data transaction rows meet conditions
    THEN assign/delete/transform target blotter behavior

Special assignment examples:
    [*-1]
    [TradeDate]
    [SettleDate]
```

**Classification:** Verified for AIA/APX workflow. [S16]

---

## 12. Questions the Future Chapter Should Answer, With Current Evidence Status

| Blueprint Success Question | Current Evidence Status | Confidence | Research Gap |
|---|---|---|---|
| Which IMEX object exports transactions? | Not answered by reviewed sources. CI imports transactions into Axys Trade Blotter and uses IMEX, but native object name unknown. | Unknown | Need IMEX manual or sample `.inf` files. |
| Which REP report contains security performance? | Not answered by reviewed sources. | Unknown | Need Axys/APX standard report list or examples. |
| Where are classifications stored? | Not answered in IMEX-specific sources. | Unknown | Need security master/classification docs and IMEX exports. |
| Does APX store or recalculate performance? | Not answered in reviewed sources. | Unknown | Need APX performance/reporting docs or Stored Accounting Function documentation. |
| Which fields identify a security? | For Axys CI, Symbol + Type identify Axys security; CUSIP/ticker/name can be incoming identifiers. Native IMEX security key unknown. | Verified for CI; Unknown native | Need native Axys/APX field dictionaries. |
| Which fields are required? | CI output file required fields known for missing-prices/security-translation reports only. Native IMEX required fields unknown. | Unknown native | Need IMEX data dictionary. |
| Which reports use stored values? | Not answered. | Unknown | Need report docs and production tests. |
| What are known quirks? | Several integration quirks documented: file locks, position replacement, security ambiguity, Accept step, Report Writer checksum, version file-format differences. | Verified/Medium depending on item | Need production examples and vendor docs. |

---

## 13. Unknowns To Preserve

The following should remain explicitly marked **Unknown** in
`../reference/Chapter_12_Imex.md` until better evidence is obtained.

1. Native Axys IMEX object names for transactions, positions, prices, security master, classifications, performance, and portfolio master.
2. Native APX IMEX object names and whether they match Axys names.
3. Native IMEX field dictionaries and required fields for each object.
4. Whether APX IMEX supports the same import/export modes as Axys IMEX in modern APX versions.
5. Whether APX IMEX writes to blotters first, database tables directly, or uses separate staging depending on object type/version.
6. Exact IMEX control file syntax, including `.inf` file structure, command-line switches, source/target file options, delimiter options, and fixed-vs-delimited behavior.
7. Whether Axys IMEX export can export performance values from stored performance files or only import/export accounting/security/price data.
8. Whether `portperf`, `secperf`, security master, and classifications are native IMEX objects, report outputs, stored files, or firm-specific report conventions.
9. Whether Axys and APX standard report names for performance/security performance are stable across versions.
10. Exact REP report names for portfolio performance, security performance, holdings, transactions, classifications, and prices.
11. Whether APX public views / Stored Accounting Functions are official supported data-access layers for third-party integrations or only internal/reporting tools.
12. Whether direct SQL querying of APX database is vendor-supported, tolerated, or discouraged.
13. Whether Axys IMEX log file names are standardized outside the CI workflow.
14. Whether APX IMEX log file names and locations are standardized.
15. Whether uppercase/lowercase transaction code behavior in the APX AIA cancel example reflects APX-native cancellation semantics or AIA-generated blotter convention.

---

## 14. Recommended Additional Evidence To Request From User / Repository

To turn this research into a stronger chapter, request any of the following if available:

| Desired Artifact | Why It Matters |
|---|---|
| Axys IMEX user manual or help pages | Primary source for object names, fields, control files, command behavior. |
| APX IMEX user manual/help pages | Primary source for APX-specific IMEX behavior and version differences. |
| Sample Axys IMEX `.inf` files | Reveals object names, file formats, delimiter/fixed settings, field names. |
| Sample APX IMEX `.inf` files | Same as above for APX. |
| Sample IMEX import/export logs | Confirms log names, errors, processing sequence, object labels. |
| Sample Axys `sec.inf`, `type.inf`, `.pri`, `.cli`, `.pos`, `topost.trn` files | Confirms file layout and fields; sensitive data can be anonymized. |
| Sample REP reports: performance, holdings, security master, classifications | Reveals report names, RepLang variables, output fields. |
| APX public view / Stored Accounting Function documentation | Needed for APX database/report extraction chapter overlap. |
| Screenshots of IMEX dialogs from Axys and APX | Confirms UI labels, selectable objects, export/import options. |
| SS&C release notes for Axys 3.7/3.8 and APX versions | Needed for version-difference table. |

---

## 15. Chapter Structure Suggested From Research

```text
## 12. IMEX

1. Overview
   - Define IMEX narrowly.
   - Distinguish IMEX from REP/report exports and direct file access.
   - Explicit confidence legend.

2. Axys
   - IMEX as Axys Import/Export utility.
   - Transaction/position/price import evidence.
   - Axys files and folders relevant to IMEX.
   - Logs and failure modes.

3. APX
   - APX IMEX evidence is thinner.
   - APX AIA IMEX log/history log.
   - APX blotter-based workflows.
   - APX database/report alternatives.

4. IMEX Data Model
   - Known file names.
   - Known CI output fields.
   - Native IMEX fields marked Unknown pending docs.

5. REP / Report-Based Exports
   - Why REP is often used instead of IMEX.
   - REP32, macros, Replang, Report Writer Pro.
   - Standard/custom report examples.

6. Processing Behavior
   - Axys CI workflow.
   - APX AIA workflow.
   - Logs, Accept, file locks, position replacement.

7. Version Differences
   - Axys 3.7/3.8 file conversion caution.
   - Connector support versions.
   - APX v1.x-v4.x IMEX/fixed-format note.

8. Known Issues / Quirks
   - File locks.
   - Security ambiguity.
   - Account vs portfolio code.
   - Direct file access risk.
   - Report Writer checksum.

9. Examples
   - Missing prices fields.
   - Security translation fields.
   - Cancel transaction example.

10. References

11. Unknowns
```

---

## 16. References

[S1] SS&C Advent, **Axys® | Portfolio Reporting and Portfolio Accounting Software**. Key reviewed lines: Axys automates portfolio reporting/accounting; extensive pre-defined reports; Report Writer Pro; performance measurement; reconciliation.  
URL: `https://www.advent.com/solutions/axys/`

[S2] SS&C Advent, **Advent Portfolio Exchange® | Portfolio Management Solution**. Key reviewed lines: APX integrated portfolio/client management platform; standard reports; performance analytics; flexible custom reporting; GIPS composite support.  
URL: `https://www.advent.com/solutions/advent-portfolio-exchange/`

[S3] SS&C Advent, **Industry standard portfolio management and reporting / PB_AXYS.pdf**. Key reviewed lines: hundreds of predefined reports and Report Writer Pro.  
URL: `https://cdn.advent.com/cms/pdfs/briefs/PB_AXYS.pdf`

[S4] SS&C Advent, **Advent Portfolio Exchange Reports Guide / REP_APX.pdf**. Browsing tool search result found the PDF, but opening failed. Must be acquired separately for use.  
URL: `https://cdn.advent.com/cms/pdfs/reports/REP_APX.pdf`

[S5] ByAllAccounts, **Custodial Integrator Axys User Guide**. Key reviewed lines include terminology defining IMEX as Axys Import/Export utility; security translation; missing prices fields; security translation fields; resolving untranslated/duplicated securities.  
URL: `https://www.byallaccounts.net/Manuals/Custodial_Integrator/axys/CI_User_Guide.pdf`

[S6] ByAllAccounts, **Custodial Integrator Axys User Guide**. Key reviewed lines include data translation workflow; Export step; View IMEX Logs; transaction/position/price import; log tabs; file-in-use errors; Accept step.  
URL: `https://www.byallaccounts.net/Manuals/Custodial_Integrator/axys/CI_User_Guide.pdf`

[S7] ByAllAccounts, **Custodial Integrator Axys User Guide**. Key reviewed lines include Axys folders/files: `topost.trn`, `$pathcli`, `*.cli`, `$pathinf`, `sec.inf`, `type.inf`, `$pathpri`, `*.pri`, `$pathlog`, `ptopost.trn`, position posting.  
URL: `https://www.byallaccounts.net/Manuals/Custodial_Integrator/axys/CI_User_Guide.pdf`

[S8] WealthTechs, **AIA User Manual for Axys Users**. Key reviewed lines include `CDIhold.rep`, Axys custom report menu, `$pathCDI`, holdings extract folder, AIA additional use cases.  
URL: `https://www.wealthtechs.com/files/AIADocumentation/Axys%20Guide%20-%20AIA%20User%20Manual%20For%20Axys%20Users.pdf`

[S9] WealthTechs, **AIA User Manual for APX Users**. Key reviewed lines include APX account/initial transaction blotters; Advent IMEX Log; Advent IMEX History Log; Delete all Positions; Cancel Transactions example.  
URL: `https://www.wealthtechs.com/files/AIADocumentation/APX%20Guide%20-%20AIA%20User%20Manual%20For%20APX%20Users.pdf`

[S10] Salentica Elements, **Data Broker: SS&C Advent APX & Axys**. Key reviewed lines: Axys/APX on-prem/AOS data extraction; connector uses standard reports/macros and client machine with Advent client tools including REP32.  
URL: `https://elements.salentica.com/kb/article/252-data-broker-ss-c-advent-apx-axys/`

[S11] Salentica Engage, **Data Broker - SS&C Advent APX & Axys**. Key reviewed lines: minimum supported versions; connector is 32-bit Windows application; requires Advent Client Tools / REP32.exe; uses REP32 engine, RepLang scripting, and macros.  
URL: `https://engage.salentica.com/kb/article/247-data-broker-ss-c-advent-apx-axys/`

[S12] AdventGuru, **IMEX**. Key reviewed lines: Axys direct data-file access not best practice due to file-format changes; Axys 3.7 to 3.8 conversion example; APX database/SQL extraction options; Axys/APX reports/Report Writer/Replang options; APX v1.x-v4.x IMEX and fixed-format statement.  
URL: `https://adventguru.com/tag/imex/`

[S13] AdventGuru, **REP**. Key reviewed lines: Report Writer Pro; `.RPW` vs `.REP`; Replang source; checksum caveat; backup and manual modification workflow.  
URL: `https://adventguru.com/tag/rep/`

[S14] AdventGuru, **APX category / State of Reporting Development for Axys and APX Users**. Key reviewed lines: APX SQL Server database; Stored Accounting Functions; Public Views; SSRS; REST API; Replang remains part of Axys/APX reporting architecture.  
URL: `https://adventguru.com/category/portfolio-management-systems/apx/`

[S15] Consultant tutorial, **How to Add Portfolio Code to Axys Reports**. Key reviewed lines: Axys reports written in Replang; AMAN.REP; example path `e:\axys34\rep`; copying reports before modification.  
URL: `https://assets.ctfassets.net/xhy36q2d1lqu/77QC4aNbyhPo9FfmjRYNzc/d00a0d6601214601543e30e34f203626/PortfolioCodetoAxys.pdf`

[S16] WealthTechs, **AIA User Manual for APX Users**. Key reviewed lines: Transaction Translation IF/THEN logic; not case sensitive within that translation-rule context; special assignment values; example deleting transaction codes; Vehicle Filter.
URL: `https://www.wealthtechs.com/files/AIADocumentation/APX%20Guide%20-%20AIA%20User%20Manual%20For%20APX%20Users.pdf`

---

## 17. Citation Crosswalk to Browsing Evidence

This section preserves the exact browsing citations used during research so a later editor can re-open the evidence trail.

- Blueprint loaded from attached file: `AXYS_APX_REFERENCE_BLUEPRINT(19).md`.
- SS&C Axys page: `turn489413view0`, especially lines 137–145, 152–164, 187–198.
- SS&C APX page: `turn489413view1`, especially lines 131–141, 146–171, 180–184.
- ByAllAccounts CI Axys guide terminology: `turn917709view0`, lines 59–62.
- ByAllAccounts CI Axys guide workflow: `turn932341view0`, lines 1209–1231.
- ByAllAccounts CI Axys guide Axys folders/files: `turn932341view2`, lines 770–809.
- ByAllAccounts CI Axys guide IMEX logs: `turn932341view1`, lines 1377–1423.
- ByAllAccounts CI Axys guide security translation: `turn597765view3`, lines 1044–1175 and 1493–1535.
- WealthTechs Axys AIA guide holdings extract: `turn917709view2`, lines 1156–1160; screenshot `turn604883view3`.
- WealthTechs APX AIA guide blotters/logs/cancel transactions: `turn917709view1`, `turn932341view3`, `turn932341view4`, screenshot `turn604883view1`, `turn604883view2`.
- Salentica Elements connector: `turn489413view2`, lines 60–77.
- Salentica Engage connector: `turn489413view3`, lines 69–97.
- AdventGuru IMEX: `turn489413view4`, lines 30–44.
- AdventGuru REP: `turn489413view5`, lines 25–32.
- AdventGuru APX reporting: `turn489413view6`, lines 125–134.
- PortfolioCodeToAxys PDF: `turn917709view3`, lines 20–23; screenshot `turn604883view4`.

---

## 18. Deep Research Addendum: Axys IMEX Inventory and Discovery Strategy

Source incorporated: `axys_imex_deep_research.md`, prepared 2026-06-30.

### 18.1 Central limitation

The deep research did **not** find a public, authoritative Axys IMEX data
dictionary listing every object and selectable field. The complete inventory
appears to require proprietary SS&C/Advent documentation or discovery from a
licensed Axys installation through `imex32.exe`, Axys Help, saved templates,
macros, logs, and exported layouts.

This limitation should control all chapter language:

- confirmed integration artifacts are useful implementation evidence;
- candidate field families are discovery targets, not official schemas;
- no chapter should imply that a normalized repository field is generally
  available in every Axys IMEX extract unless a source proves that availability.

### 18.2 Publicly supported Axys IMEX inventory

| Area | Public evidence | Direction / role | Confidence | Caveat |
|---|---|---|---:|---|
| Security information | `sec.inf` used/exported by CI. | Axys reference data used to map securities and generate valid imports. | Verified for CI | Full field layout Unknown. |
| Security type information | `type.inf` used/exported by CI. | Security type reference used with security master. | Verified for CI | Full type dictionary Unknown. |
| Transactions / Trade Blotter | `topost.trn` in `$pathtrn`. | Import/appended into Trade Blotter for review/posting. | Verified for CI | Complete Trade Blotter layout Unknown. |
| Positions | `ptopost.trn`; Position Post creates `.pos` files. | Position import/post workflow. | Verified for CI | Holdings vs native position storage remains Unknown. |
| Position lots | `ptopost.trn` can contain lot data; `imexPositionLots.log`. | Optional import when lots are enabled and available. | Verified for CI | Lot schema Unknown. |
| Prices | `*.pri` files in `$pathpri`. | Price import/append workflow. | Verified for CI | Price-file layout and naming conventions Unknown. |
| Missing prices diagnostic | `MISSINGPRICES_yyyymmdd.csv`. | CI optional diagnostic output. | Verified for CI | Not necessarily native IMEX object. |
| Security translations diagnostic | `SECTRANSLATIONS_yyyymmdd.csv`. | CI optional diagnostic/output file. | Verified for CI | Not native security master. |
| Portfolio/client files | `*.cli` in `$pathcli`. | CI traverses files to build portfolio-code lists. | Verified for CI | Direct file layout Unknown. |
| Classifications/reference lists | sectors, industries, asset classes, indexes, composites. | Export mentioned in conversion context. | Medium | Exact object names and fields Unknown. |
| Performance history | performance history export concerns. | Possible IMEX/migration concern. | Medium | Public evidence says this can be frustrating; clean field catalog Unknown. |
| Reports/macros alternative | REP32, standard reports, macros, Replang. | Extraction alternative when IMEX is incomplete or report-shaped values are needed. | Verified for connector | Not IMEX. |

### 18.3 Confirmed folder labels and executable names to preserve

| Artifact | Meaning in public evidence | Confidence |
|---|---|---:|
| `imex32.exe` | Axys Import/Export utility. | Verified for CI |
| `pospos32.exe` | Axys Post Positions utility. | Verified for CI |
| `$pathexe` | Axys executable folder. | Verified for CI |
| `$pathtrn` | User folder / Trade Blotter location. | Verified for CI |
| `$pathcli` | Client folder with `*.cli` portfolio files. | Verified for CI |
| `$pathinf` | Information folder containing `sec.inf` and `type.inf`. | Verified for CI |
| `$pathpri` | Price folder containing `*.pri`. | Verified for CI |
| `$pathlog` | IMEX log folder. | Verified for CI |

### 18.4 Candidate field families for live discovery

These are discovery checklists, not confirmed universal IMEX schemas.

| Object family | Candidate field families to inspect |
|---|---|
| Security master / `sec.inf` | Symbol, type, name, ticker, CUSIP, ISIN, currency, pricing method, multiplier, classifications, country/region, coupon, maturity, factor, issue date, user-defined fields. |
| Security types / `type.inf` | Type code, description, asset/accounting behavior, pricing units, quote type, multiplier, cash/security behavior, accrual/performance/reporting behavior. |
| Transactions / `topost.trn` | Portfolio, transaction code/subtype, trade/settle/post/effective dates, symbol/type, quantity, price, amount, commission, fees, accrued interest, withholding, source/destination type and symbol, cost date/amount, Perf/CW, Mark to Market, currency, FX, comments/source IDs. |
| Positions / `ptopost.trn`, `.pos` | Portfolio, as-of date, symbol/type, quantity/units, price, market value, accrued income, cost, local/base currency, stale flags, custodian/account context. |
| Position lots | Lot identifier, open/acquisition date, quantity, cost, market value, currency, lot-level tax-cost fields. |
| Prices / `.pri` | Symbol/type, price date, price, source, currency, factor, quote multiplier. |
| Classifications / labels | Asset class, sector, industry, country, region, code/name, sort order, parent/child relationships, security or portfolio assignments, effective dates. |
| Performance-related exports | Portfolio/security/classification/composite identifiers, period dates, beginning/ending values, returns, contributions, weights, external flows, income, fees, benchmark fields. |

### 18.5 Live Axys IMEX catalog procedure

If a licensed Axys environment is available, build an empirical catalog:

1. Run `imex32.exe` interactively on a test workstation.
2. For each exportable/importable object, record the object name exactly as
   displayed.
3. Capture selectable fields, field order, labels, internal tokens if shown,
   data type/width if shown, direction, and required/optional status.
4. Save every export/import template or layout definition.
5. Export CSV, tab, and fixed formats where available.
6. Repeat with a rich test portfolio containing cash, equities, funds, fixed
   income, options if applicable, income, fees, contributions/withdrawals,
   corporate actions, multicurrency examples, classifications, and tax lots.
7. Collect IMEX logs and record record counts, field counts, errors, rejected
   rows, and line/column diagnostics.
8. Compare IMEX output with REP/report output for performance, holdings,
   classifications, and other values that need user-visible tie-out.

Suggested search terms for existing contracts/templates/logs in Axys folders:
`*.imx`, `*.iex`, `*.exp`, `*.fmt`, `*.ini`, `*.log`, `*imex*`,
`*export*`, `*import*`, `*.csv`, `*.tab`, `*.txt`.

### 18.6 Recommended internal `imex_catalog` structure

| Column | Purpose |
|---|---|
| `axys_version` | Version that produced the observation. |
| `object_name` | IMEX object/dataset name exactly as displayed. |
| `direction` | Import, export, or both. |
| `field_token` | Internal field name if known. |
| `field_label` | Display/header label. |
| `data_type` / `width` | Type, precision, length, or fixed-width metadata if shown. |
| `required_import` | Whether required for import. |
| `export_available` / `import_available` | Directional availability. |
| `example_value` | Sanitized example. |
| `source` | Public document, live IMEX, log, user supplied, inferred. |
| `confidence` | Confirmed, probable, inferred, or Unknown. |

### 18.7 Product-design bottom line

The public record supports a resilient audit architecture:

1. Use IMEX where it is strong: security reference, security types, prices,
   transactions, positions, and other operational/reference files.
2. Use REP/Replang/custom reports where user-visible performance, classification
   performance, or report-shaped tie-outs are needed.
3. Maintain a versioned schema catalog per client/installation.
4. Avoid hard-coding one universal Axys IMEX schema.
5. Normalize into product-owned schemas while preserving original source file,
   source row, original field names, extraction mechanism, and confidence.

## Deep Research Update Incorporated 2026-07-02

The July 2026 addendum materially strengthens APX IMEX workflow evidence.
ByAllAccounts APX CI evidence verifies `apxix.exe` exporting `sec.inf` and
`type.inf`, extracting APX group portfolio codes, and importing transactions,
prices, positions, and position lots into APX blotters or price records. APX
BackOffice Utilities and APX DataPort are CI prerequisites for relevant
position workflows, and APX authentication can block `apxix.exe`; Apxix logs
may be required for diagnosis.

WealthTechs AIA verifies `APXIX.exe` writing to `imexhist.log`, APX SQL
connection settings such as Data Base Server, Data Base Name, user/password,
Windows Authentication, and Test Connection, `SourceId` display behavior in
price imports, and `.veh` to `sec.inf`-layout security import/update behavior.
These are environment/workflow facts, not APX schema definitions.

APX Market Data Manager / Interactive Data RemotePlus is an adjacent non-IMEX
reference-data path for prices/evaluations, factors, security setup,
fixed-income terms, index values, dividends/splits, exchange rates, and
trade-blotter posting from dividend downloads. FinFolio conversion evidence
adds migration leads for `CLI`, `PRI`, `INF`, `SPLIT.INF`, `PRF`, and `GRP`,
but those remain migration file-family leads rather than official object
names or complete layouts.
