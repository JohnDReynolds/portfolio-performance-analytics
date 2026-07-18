# Chapter 12 — IMEX

**Repository:** AXYS / APX Reference Repository
**Chapter:** `docs/axys_apx/reference/Chapter_12_Imex.md`
**Status:** Technical reference chapter
**Prepared:** 2026-06-29
**Source basis:** Repository research plus public evidence reviewed through 2026-07-17.

---

## Related chapters

- [Chapter_01_Overview.md](Chapter_01_Overview.md) — provides the repository-wide map, evidence conventions, and shared safe implementation rules.
- [Chapter_02_Axys_Architecture.md](Chapter_02_Axys_Architecture.md) — frames the architectural place of IMEX in Axys.
- [Chapter_03_APX_Architecture.md](Chapter_03_APX_Architecture.md) — frames the APX-side import/export context.
- [Chapter_13_Rep.md](Chapter_13_Rep.md) — distinguishes IMEX from REP/report extraction.

## 1. Overview

IMEX is the import/export layer referenced in the supplied research for moving data into and out of Axys/APX environments. The available source material supports strong statements about specific integration workflows, utilities, file names, logs, and report-based extraction alternatives. It does **not** provide a complete official SS&C IMEX object dictionary, native IMEX field dictionary, or complete native APX/Axys command syntax.

This chapter therefore distinguishes:

The supplied IMEX evidence also shows that transaction meaning is shaped by import and review stages before posting. Transactions arrive through staging layers such as Trade Blotter files and may be adjusted by translation rules, special security markers, and reversal handling before they affect accounting state. For downstream systems, preserving the original transaction code, translation mapping, source/destination context, and sign is more reliable than assuming the code alone carries the accounting meaning.

| Layer | Description | Axys | APX | Confidence |
|---|---|---:|---:|---|
| IMEX / Import-Export utility | Import/export mechanism used in integration workflows. | Yes | Yes, but APX details thinner | Verified for Axys CI; Verified for APX AIA logs; Medium Confidence for broader APX behavior |
| Trade Blotter / blotter import | Staging/review path used by transaction import workflows. | Yes | Yes | Verified for cited integration workflows |
| REP / REP32 / Replang | Report engine/report language used for extraction, reports, macros, and custom exports. | Yes | Yes | Verified for connector workflows; High Confidence generally |
| Report Writer Pro | Report customization mechanism; `.RPW` and `.REP` behavior appears in consultant evidence. | Yes | Yes, based on research evidence | Verified for Axys product capability; Medium Confidence for detailed behavior |
| Direct file access | Reading/writing proprietary Axys files directly. | Possible but discouraged by consultant evidence | Not applicable in same way | Medium Confidence |
| SQL/public views/Stored Accounting Functions/REST | APX database/reporting access alternatives. REST capability is officially documented for APX 21.1+ releases. | No | Yes | REST capability Verified by release; exact schemas and other paths vary |

### 1.1 Confidence

This chapter uses the repository confidence labels defined in
[Chapter 01](Chapter_01_Overview.md#2-confidence-and-evidence-boundary).
Confidence remains scoped to the cited product, version, report, connector,
or workflow.

### 1.2 Scope boundaries

This chapter documents what the supplied material supports about IMEX and IMEX-adjacent extraction/import paths. It does not attempt to invent:

- official IMEX object names,
- full import/export schemas,
- command-line syntax,
- `.inf` control-file syntax,
- native APX database table names,
- native Axys proprietary file layouts,
- complete REP report catalogs,
- complete transaction/security/performance object dictionaries.

Where the supplied material is incomplete, this chapter marks the matter **Unknown**.

---

## 2. Axys

### 2.1 Axys IMEX definition

| Statement | Confidence |
|---|---:|
| A ByAllAccounts Custodial Integrator Axys guide explicitly defines IMEX as the **Axys Import/Export utility**. | Verified |
| The Axys Import/Export utility is used in the CI workflow to import requested Transaction, Position, and Price files into Axys. | Verified for CI workflow |
| The exact native Axys IMEX object names used by SS&C documentation are not supplied. | Unknown |
| The exact native field list for each Axys IMEX object is not supplied. | Unknown |

### 2.2 Axys IMEX executable and related utilities

| Artifact | Description | Confidence | Notes |
|---|---|---:|---|
| `imex32.exe` | Axys Import/Export utility executable referenced by ByAllAccounts CI research. | Verified for CI workflow | The research identifies it as the executable CI looks for when using Axys Import/Export. |
| `pospos32.exe` | Axys Post Positions utility referenced by ByAllAccounts CI research. | Verified for CI workflow | Position-import context, not transaction-object documentation. |
| `REP32.exe` | Report engine/client tool used by Salentica Data Broker connector. | Verified for connector | REP/report extraction path, not IMEX. |

### 2.3 Axys files and folders used by integration workflows

The following file and folder labels are supported by the supplied research. They are documented here as **observed integration artifacts**, not as a complete native Axys file-layout specification.

| File / Folder / Label | Description | Confidence | Caveat |
|---|---|---:|---|
| `$pathexe` | Axys executable folder label used in CI configuration. | Verified for CI workflow | Example label, not necessarily visible to all users. |
| `$pathtrn` | Axys user folder / Trade Blotter location used by CI. | Verified for CI workflow | Used for `topost.trn`. |
| `topost.trn` | Axys Trade Blotter file receiving transaction imports. CI appends generated transactions to this file and leaves existing transactions unchanged. | Verified for CI workflow | Full native layout Unknown. |
| `$pathcli` | Axys client folder where portfolio/client `*.cli` files are stored. CI traverses this folder to build a portfolio-code list. | Verified for CI workflow | Full `.cli` layout Unknown. |
| `*.cli` | Axys portfolio/client files. | Verified for CI/conversion context | Used by CI and conversion material; not a complete file spec. |
| `$pathinf` | Axys information folder containing `sec.inf` and `type.inf` in the CI workflow. | Verified for CI workflow | Full field layouts Unknown. |
| `sec.inf` | Axys Security Information file exported/used by CI to support generation of transaction and position files. | Verified for CI workflow | Complete native layout Unknown. |
| `type.inf` | Axys Security Type Information file exported/used by CI. | Verified for CI workflow | Complete native layout Unknown. |
| `$pathpri` | Axys price folder containing `*.pri` files. | Verified for CI workflow | Price-file layout Unknown. |
| `*.pri` | Axys price files used by CI for merge/import behavior. | Verified for CI workflow | Full file layout Unknown. |
| `$pathlog` | Folder where Axys Import/Export logs are written in the CI workflow. | Verified for CI workflow | Standardization outside CI Unknown. |
| `ptopost.trn` | Position file written by CI to `\CI\exported\ptopost.trn`; research says it is CSV format and may contain lot-specific data when enabled/available. | Verified for CI workflow | Position context, not native IMEX object spec. |
| `.pos` files | Axys replacement position files created by Position Post when configured. | Verified for CI workflow | Position storage details Unknown. |
| `didpost.aud` | Audit Trail file for posted transactions, based on consultant evidence in supplied transaction research. | Medium Confidence | Layout, retention behavior, and universality Unknown. |

### 2.4 Axys import categories supported by supplied evidence

| Category | Evidence-supported behavior | Confidence |
|---|---|---:|
| Transactions | CI can deliver transactions into the Axys Trade Blotter file `topost.trn` for review and posting. | Verified for CI workflow |
| Positions | CI can write prior-business-day positions to `ptopost.trn` and import positions to Axys `.pos` files when configured. | Verified for CI workflow |
| Position lots | CI may use a position-lots log tab when position lots are enabled. | Verified for CI workflow |
| Prices | CI exports price files from `$pathpri` for merge purposes and imports generated price files back to `$pathpri`. | Verified for CI workflow |
| Security information | CI exports/uses `sec.inf` and `type.inf`, but the research says CI does not modify Axys Security or Security Type information as part of its security translation workflow. | Verified for CI workflow |

### 2.5 Axys IMEX logs and error review

| Log / Behavior | Description | Confidence |
|---|---|---:|
| View IMEX Logs | CI retains Axys Import/Export utility execution logs and exposes them through a View IMEX Logs dialog. | Verified for CI workflow |
| One tab per imported data type | CI View IMEX Logs has one tab for each imported data type. | Verified for CI workflow |
| `imexPositions.log` | Referenced in the supplied research as an IMEX positions log tab. | Verified for CI workflow |
| `imexPositionLots.log` | Used instead of `imexPositions.log` when position lots are used. | Verified for CI workflow |
| `imexPrices.log` | Price-import log tab; multiple historical price days may produce one price log tab per day. | Verified for CI workflow |
| `imexTransactions.log` | Inferred pattern only in the research example. | Unknown |
| File-in-use failure | If an Axys file targeted by import is open/in use, import may fail; the cited example is an Axys price file open during price import with an error in `imexPrices.log`. | Verified for CI workflow |
| Accept step | In the CI workflow, accepting exported data updates CI’s internal transaction counter. If not accepted, the same transactions plus new transactions may download again. | Verified for CI workflow |

### 2.6 Axys security-resolution behavior relevant to IMEX

| Behavior | Description | Confidence |
|---|---|---:|
| Import blocked by unresolved securities | CI will not import data into Axys unless all securities resolve to Axys securities. | Verified for CI workflow |
| Untranslated securities | CI identifies securities where no corresponding Axys security is found. | Verified for CI workflow |
| Duplicate securities | CI identifies securities where more than one Axys security is found. | Verified for CI workflow |
| Duplicate examples | Duplicate/security ambiguity examples include same symbol with different security types, such as `ktc csus` and `ktc adus`, ticker and CUSIP entered as separate securities, examples such as `tfus` and `oaus`, and overlapping CI security translations. | Verified for CI examples |
| Security identity in integration context | CI uses Axys Symbol + Type as the target security identifier pair in security translation files. | Verified for CI workflow |
| Native primary key | Whether Symbol + Type is the formal native Axys primary key is not established. | Unknown |

### 2.7 Axys transaction import and Trade Blotter behavior

| Item | Evidence-supported statement | Confidence |
|---|---|---:|
| Trade Blotter file | Transactions are delivered to `topost.trn` in the Axys user folder. | Verified for CI workflow |
| Append behavior | Generated CI transactions are appended to the end of `topost.trn`; existing transactions are left unchanged. | Verified for CI workflow |
| File creation | If no Trade Blotter file exists, the Axys Import/Export utility can create one in the configured user folder. | Verified for CI workflow |
| Comment boundaries | Generated transaction blocks can be bounded by beginning and ending comment transactions containing the generation date. | Verified for CI workflow |
| Source comments | CI can include source transaction information as a comment in Trade Blotter; the comments do not post to Axys client files, according to the guide. | Verified for CI workflow |
| All Axys transaction imports must use Trade Blotter | Not established as a universal rule. | Unknown |

### 2.8 Axys observed transaction-import fields and parameters

These are observed integration fields/parameters. They are not a complete native Axys transaction import layout.

| Field / Parameter | Description | Axys | IMEX | Confidence | Caveat |
|---|---|---:|---:|---:|---|
| Axys Transaction Type | Target transaction code in ByAllAccounts translation table. | Yes | Related | Medium Confidence | Integration field, not official native IMEX field name. |
| Axys Transaction Src/Dest Type | Source/destination type field in ByAllAccounts translation table. | Yes | Related | Medium Confidence | Integration field. |
| Axys Transaction Src/Dest Symbol | Source/destination symbol in ByAllAccounts translation table. | Yes | Related | Medium Confidence | Integration field. |
| Axys Transaction Special Security Type / Symbol | Special security type/symbol used for certain transactions such as fees. | Yes | Related | Medium Confidence | Integration field. |
| Broker column | Can be populated by CI parameter, including when Commission is populated. | Yes | Related | Medium Confidence | Full layout Unknown. |
| Commission column | Can be populated or suppressed by CI configuration. | Yes | Related | Medium Confidence | Full layout Unknown. |
| Lot location column | Populated through CI parameter; example default `253` appears in transaction research. | Yes | Related | Medium Confidence | Native lot-location model Unknown. |
| Quantity column | CI parameter `defdivquan` can provide Quantity value for Dividend transactions lacking reported Quantity. | Yes | Related | Medium Confidence | Integration behavior. |
| Mark to Market field | CI parameter can define value for non-system-currency transactions requiring Mark to Market. | Yes | Related | Medium Confidence | Native multicurrency mechanics Unknown. |
| Perf/CW column | CI parameter `defperfcw` defines value for Perf/CW column of Axys `topost.trn`. | Yes | Related | Medium Confidence | Meaning Unknown. |
| Currency code | CI parameter `axyscur` defines system currency for translation. | Yes | Related | Medium Confidence | Configuration, not necessarily file column. |

### 2.9 Axys version and file-format cautions

| Statement | Confidence | Notes |
|---|---:|---|
| Direct Axys file access is not best practice because proprietary file formats can change between versions. | Medium Confidence | Consultant evidence in supplied research. |
| A cited example says upgrading from Axys 3.7 to 3.8 required file conversion and produced some files with different formats. | Medium Confidence | Consultant evidence. |
| Salentica Data Broker documentation lists Axys 3.8.6 as a minimum supported version for that connector. | Verified for connector | |
| Whether Axys IMEX file schemas differ among Axys versions is not established. | Unknown | |
| Consultant evidence reports large Axys 3.x Audit Trail IMEX exports may fail in some environments. | Medium Confidence | Do not generalize without production testing. |

---

## 3. APX

### 3.1 APX IMEX evidence level

The APX evidence strongly supports the utility, log, and blotter workflow through two
independent integration guides, but it still does not provide a complete official APX
IMEX object dictionary.

| Statement | Confidence |
|---|---:|
| APX-related integration tooling uses “Advent IMEX Log” and “Advent IMEX History Log” tools. | Verified for AIA workflow |
| The AIA APX guide says these logs are useful for troubleshooting importing and blotter issues. | Verified for AIA workflow |
| ByAllAccounts APX installation research says `apxix.exe` exports Security (`sec.inf`) and Security Type (`type.inf`) data from APX in CI context. | Verified for CI context |
| WealthTechs APX AIA research identifies `APXIX.exe` as the import/export function of APX. | Verified for AIA context |
| `ApxIx`, `apxix.exe`, and `APXIX.exe` are capitalization/label variants for the APX Import/Export utility in the reviewed guides. | High Confidence; installed filename and version remain site-specific |
| Native APX IMEX object names and field dictionaries are not supplied. | Unknown |

### 3.2 APX import/export executables and names

| Artifact / Name | Description | Confidence | Caveat |
|---|---|---:|---|
| `APXIX.exe` / `apxix.exe` / `ApxIx` | Capitalization and label variants used for the APX Import/Export utility. | High Confidence across AIA/CI guides | Confirm the installed filename/version before automation. |
| `REP32.exe` | Used by Salentica Data Broker connector to extract Axys/APX data through reports/macros. | Verified for connector | REP path, not IMEX object documentation. |

### 3.3 APX IMEX logs

| Tool / Log | Description | Confidence |
|---|---|---:|
| Advent IMEX Log | Shows the log for the last importing done through the IMEX tool; useful for troubleshooting importing issues. | Verified for AIA workflow |
| Advent IMEX History Log | Shows logs for all importing done through the IMEX tool; useful for troubleshooting importing and blotter issues. | Verified for AIA workflow |
| `imexhist.log` | APX Import/Export history log named in the WealthTechs AIA guide. | Verified for that workflow; location and version coverage Unknown |
| APX IMEX log schema / error codes | Not established in supplied material. | Unknown |

### 3.4 APX blotters relevant to import workflows

The APX AIA research identifies multiple blotters. These are documented as AIA/APX workflow evidence, not necessarily a complete APX-native blotter taxonomy.

| Blotter / Tool | Description in supplied research | Confidence |
|---|---|---:|
| Account Blotter | Imports account demographic data from custodians, such as Account Number, Account Name, and Account Type; may update or add APX account data. | Verified for AIA workflow |
| Initial Transaction Blotter | Can be driven by an AIA setting to create initial deliver-in transactions from positions when an APX account has no transactions. | Verified for AIA workflow |
| Trade Blotter | Receives transaction imports; AIA may consolidate into one blotter, create one per custodian, or create no trade blotter. | Verified for AIA workflow |
| Position Blotter | Used for positions. | Verified for AIA workflow |
| Statement Blotter | Used to post statement transactions and support reconciliation workflows. | Verified for AIA workflow |
| Tax Lot Blotter | Used for tax-lot-level reconciliation/import workflows. | Verified for AIA workflow |
| Pending Blotters | Mentioned in AIA workflow research. | Verified for AIA workflow |
| Dividend Adjustment Blotter | Mentioned in AIA workflow research. | Verified for AIA workflow |
| Cancel Transactions | AIA tool creates a trade blotter to cancel previously posted transactions from historical transaction files; guide warns to use caution and back up accounts. | Verified for AIA workflow |

### 3.5 APX security/type export evidence

| Statement | Confidence | Caveat |
|---|---:|---|
| ByAllAccounts CI research states APX Import/Export exports Security (`sec.inf`) and Security Type (`type.inf`) data from APX. | Verified for CI context | Complete file layout Unknown. |
| Exported APX security/type information enables CI to produce transactions, positions, and prices using security symbols and security types defined in APX. | Verified for CI context | Integration workflow only. |
| APX AIA Vehicle Import Settings can translate a `.veh` file to the layout of `sec.inf` in the AIA Archive folder. | Verified for AIA context | `.veh` is AIA-specific. |
| Complete APX native `sec.inf` and `type.inf` layouts are not supplied. | Unknown | Requires sample exports or vendor documentation. |

### 3.6 APX transaction translation and cancellation example

The APX AIA guide gives a cancellation example:

```csv
acct123,010101,010101,by,csus,appl,100,caus,cash,10000
```

becomes:

```csv
acct123,010101,010101,BY,csus,appl,100,caus,cash,10000
```

| Interpretation | Confidence |
|---|---:|
| The example demonstrates uppercase transformation of a historical transaction code while creating an AIA/APX cancellation Trade Blotter. | Verified example syntax for the staging/control workflow |
| The uppercase instruction is proven to survive as a posted transaction or be exportable through ordinary IMEX/APXIX, REP, SQL, REST, or report paths. | Unknown |
| Whether uppercase transaction codes are universal APX-native cancellation semantics across all versions and import paths is not established. | Unknown |
| Exact column meanings in the example are not fully documented by supplied material. | Unknown for full layout |

### 3.7 APX observed transaction/import fields

These field labels are observed in integration or report evidence. They are not official APX database or IMEX field names unless separately verified.

| Field / Label | Description | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|
| APX Transaction Type | Target transaction code in ByAllAccounts APX translation table. | Yes | Related | Unknown | Medium Confidence |
| APX Transaction Src/Dest Type | Source/destination type in ByAllAccounts APX translation table. | Yes | Related | Unknown | Medium Confidence |
| APX Transaction Src/Dest Symbol | Source/destination symbol in ByAllAccounts APX translation table. | Yes | Related | Unknown | Medium Confidence |
| APX Transaction Special Security Type / Symbol | Special security type/symbol for certain transaction handling, including fees. | Yes | Related | Unknown | Medium Confidence |
| Account Number | Account demographic field example in AIA Account Blotter. | Yes | Related | Unknown | Verified for AIA workflow |
| Account Name | Account demographic field example in AIA Account Blotter. | Yes | Related | Unknown | Verified for AIA workflow |
| Account Type | Account demographic field example in AIA Account Blotter. | Yes | Related | Unknown | Verified for AIA workflow |
| Custodian Account Number | AIA/APX account filter field; guide warns to use custodian account number, not APX Portfolio Code, when filtering a specific account. | Yes | Related | Unknown | Verified for AIA workflow |
| APX Portfolio Code | APX portfolio identifier referenced as distinct from custodian account number. | Yes | Related | Unknown | Verified for AIA workflow |
| SourceId | Price source field shown in APX AIA price context, not a security-master or transaction field. | Yes | Price context | Unknown | Verified, non-transaction context |
| `imexhist.log` | AIA/APX import-export history log reference. | Yes | Related | Unknown | Verified for AIA context |

### 3.8 APX database/reporting alternatives

| Mechanism | Description | Confidence | Caveat |
|---|---|---:|---|
| SQL/database query | AdventGuru states APX users may query APX database data using Excel and SQL-based tools. | Medium Confidence | Vendor support status Unknown. |
| Stored Accounting Functions | AdventGuru states APX data can be accessed through Stored Accounting Functions. | Medium Confidence | Names and fields Unknown. |
| Public Views | AdventGuru states APX public views are available but limited and do not expose all desired data. | Medium Confidence | View names/fields Unknown. |
| SSRS / Crystal / SQL tools | APX reporting alternatives cited in supplied research. | Medium Confidence | Implementation details Unknown. |
| REST API | Official APX 21.1+ release material documents expanding market, reference, accounting, entity, report, performance, gain/loss, audit, analytics, cost, and fee-related API access. | Verified at release-capability level | Exact endpoints, schemas, entitlements, and IMEX equivalence Unknown. |

### 3.9 APX version notes

| Version / Range | Statement | Confidence |
|---|---|---:|
| APX v1.x–v4.x | AdventGuru states APX maintained IMEX functionality but eliminated fixed-format file generation; APX could export Axys v3 format. | Medium Confidence |
| APX 15.2 / 16.1 / 16.2 / 17.1 | Salentica connector documentation lists these as supported versions for that connector. | Verified for connector |
| Current/later APX | Exact IMEX objects, fields, logs, and version-specific behavior are not supplied. | Unknown |
| APX CI prerequisites | APX BackOffice Utilities and APX DataPort are prerequisites in relevant CI workflows. | Verified for CI workflow |
| APX authentication / logs | APX authentication can block `apxix.exe`; Apxix logs may be needed for diagnosis. | Verified for CI workflow |

---

## 4. IMEX Data Model Status

### 4.1 Native object names

The supplied research does not establish official native IMEX object names.

| Data Area | Axys IMEX object name | APX IMEX object name | Current Status |
|---|---|---|---|
| Transactions | Unknown | Unknown | CI imports transactions through Trade Blotter workflows, but official object names not supplied. |
| Positions / holdings | Unknown | Unknown | Position files and Position Post are observed, but object names not supplied. |
| Position lots | Unknown | Unknown | Log behavior observed in CI; object names not supplied. |
| Prices | Unknown | Unknown | Price files/logs observed; native object names not supplied. |
| Security master | Unknown | Unknown | `sec.inf` observed/exported/used, but official object names and complete layout not supplied. |
| Security types | Unknown | Unknown | `type.inf` observed/exported/used, but official object names and complete layout not supplied. |
| Portfolio/account master | Unknown | Unknown | `.cli` and Account Blotter evidence exist, but object names not supplied. |
| Performance / `portperf` | Unknown | Unknown | Not established as native IMEX object, report output, stored file, or firm-specific convention. |
| Security performance / `secperf` | Unknown | Unknown | Not established as native IMEX object, report output, stored file, or firm-specific convention. |
| Classifications | Unknown | Unknown | Storage/export mechanism not established in IMEX-specific material. |

### 4.2 Native field dictionaries

| Question | Status |
|---|---|
| Complete Axys native IMEX transaction field list | Unknown |
| Complete APX native IMEX transaction field list | Unknown |
| Complete Axys native IMEX security-master field list | Unknown |
| Complete APX native IMEX security-master field list | Unknown |
| Complete Axys native IMEX price field list | Unknown |
| Complete APX native IMEX price field list | Unknown |
| Required/optional flags for native IMEX fields | Unknown |
| Native field data types, lengths, precision, and date formats | Unknown |
| Native import validation rules and error codes | Unknown |
| Whether Axys/APX field names are identical | Unknown |

### 4.3 Known field-label evidence

The following field labels are supported by the research. They are intentionally labeled by context.

| Field / Name | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---:|
| Symbol | Security symbol used in CI missing-prices output; may be any Axys/APX security symbol defined in security master depending on workflow. | Yes | Yes, in APX CI research | Related | No | Verified for CI |
| Type | Security type paired with Symbol. | Yes | Yes | Related | No | Verified for CI |
| Name | Security/position name in CI missing-prices output. | Yes | Yes | Related | No | Verified for CI |
| WP Account | WebPortfolio account nickname/name in CI output. | Integration-specific | Integration-specific | No | No | Verified for CI |
| Institution | Financial institution where security is held. | Integration-specific | Integration-specific | No | No | Verified for CI |
| WP Name | Security name as it appears in WebPortfolio. | Integration-specific | Integration-specific | No | No | Verified for CI |
| WP Ticker | WebPortfolio ticker. | Integration-specific | Integration-specific | No | No | Verified for CI |
| WP Cusip | WebPortfolio CUSIP. | Integration-specific | Integration-specific | No | No | Verified for CI |
| WP Account # | WebPortfolio account number for account-specific translations. | Integration-specific | Integration-specific | No | No | Verified for CI |
| Axys Symbol | Target Axys security symbol in CI translation file. | Yes | No | Related | No | Verified for CI |
| APX Symbol | Target APX security symbol in CI translation file. | No | Yes | Related | No | Verified for CI |
| Account Number | APX account demographic field in AIA Account Blotter. | Unknown | Yes | Related | Unknown | Verified for AIA |
| Account Name | APX account demographic field in AIA Account Blotter. | Unknown | Yes | Related | Unknown | Verified for AIA |
| Account Type | APX account demographic field in AIA Account Blotter. | Unknown | Yes | Related | Unknown | Verified for AIA |
| Portfolio Code | Axys portfolio/account code list built from `*.cli`; APX portfolio identifier also appears as APX Portfolio Code. | Yes | Yes | Related | Possibly | Verified for integration contexts |
| Trade Date | Visible in transaction/report examples and AIA assignment logic. | Observed | Observed | Unknown native | Observed report label | Medium Confidence |
| Settle Date / Settlement Date | Visible in transaction/report examples and AIA assignment logic. | Observed | Observed | Unknown native | Observed report label | Medium Confidence |
| Quantity | Visible in transaction/report examples. | Observed | Observed | Unknown native | Observed report label | Medium Confidence |
| Unit Price | Visible in Transaction Summary Report sample sections. | Unknown | Observed | Unknown | Observed | Medium Confidence |
| Amount | Visible in Transaction Summary Report sample sections. | Unknown | Observed | Unknown | Observed | Medium Confidence |
| Unit Cost | Visible in Transaction Summary Report Sales section. | Unknown | Observed | Unknown | Observed | Medium Confidence |
| Total Cost | Visible in Transaction Summary Report Sales section. | Unknown | Observed | Unknown | Observed | Medium Confidence |
| Proceeds | Visible in Transaction Summary Report Sales section. | Unknown | Observed | Unknown | Observed | Medium Confidence |
| Gain/Loss | Visible in Transaction Summary Report Sales section. | Unknown | Observed | Unknown | Observed | Medium Confidence |
| Ex-Date | Visible in Transaction Summary Report Dividends section. | Unknown | Observed | Unknown | Observed | Medium Confidence |
| Pay-Date | Visible in Transaction Summary Report Dividends section. | Unknown | Observed | Unknown | Observed | Medium Confidence |

### 4.4 Publicly Supported Axys IMEX-Adjacent Inventory

Public sources do **not** provide a complete official Axys IMEX object and
field dictionary. The strongest evidence remains integration-level evidence
from Custodial Integrator, AIA, conversion, and connector workflows.

| Area | Public evidence | Role | Confidence | Boundary |
|---|---|---|---:|---|
| Security information | `sec.inf` | Security reference used to map/generate valid imports. | Verified for CI | Full layout Unknown. |
| Security types | `type.inf` | Type reference used with security master. | Verified for CI | Full type dictionary Unknown. |
| Transactions | `topost.trn` | Trade Blotter import/append file. | Verified for CI | Full Trade Blotter layout Unknown. |
| Positions | `ptopost.trn`, `.pos` | Position Post input/replacement position files. | Verified for CI | Native holdings mechanics Unknown. |
| Position lots | `imexPositionLots.log` | Optional lot import/log path. | Verified for CI | Lot schema Unknown. |
| Prices | `*.pri`, `imexPrices.log` | Price import/append workflow. | Verified for CI | Price-file layout Unknown. |
| Missing prices | `MISSINGPRICES_yyyymmdd.csv` | CI diagnostic output. | Verified for CI | Not a native IMEX object. |
| Security translations | `SECTRANSLATIONS_yyyymmdd.csv` | CI diagnostic/output file. | Verified for CI | Not native security master. |
| Portfolios | `*.cli` traversal | Portfolio-code discovery in CI. | Verified for CI | Direct file layout Unknown. |
| Classifications, indexes, composites | Conversion/export mentions. | Reference/migration candidates. | Medium | Object names/fields Unknown. |
| Performance history | Migration/export concern. | Possible IMEX concern. | Medium | Clean field catalog Unknown. |

### 4.5 Live Installation Discovery Pattern

For real implementation, build a versioned IMEX catalog from a licensed Axys
environment instead of hard-coding a universal schema:

1. Run `imex32.exe` interactively.
2. Record every displayed object name exactly.
3. Capture selectable fields, labels, internal tokens if shown, order, type,
   width, required/importable/exportable status, and
   contracts/templates/layout files.
4. Export all available formats: CSV, tab, and fixed where supported.
5. Repeat with portfolios containing cash, equities, funds, fixed income,
   corporate actions, fees, income, external flows, multicurrency, security
   classifications, and tax lots.
6. Collect IMEX logs and record counts, field counts, warnings, rejected rows,
   and line/column diagnostics.
7. Compare IMEX output with REP/report output for performance,
   classifications, holdings, and any value that must tie to a user-visible
   report.

### 4.6 Product-Design Guidance

| Design rule | Reason |
|---|---|
| Use IMEX for operational/reference data where a stable object/export exists. | Strongest evidence is for security reference, security types, transactions, prices, and positions. |
| Use REP/Replang/custom reports for user-visible report values. | Performance and classification tie-outs may be report-shaped or recalculated. |
| Preserve original source field labels and row lineage. | Client installs, versions, and templates may differ. |
| Maintain a versioned `imex_catalog`. | Public evidence does not prove one universal IMEX schema. |
| Keep normalized product schemas separate from vendor schemas. | Normalized audit fields are product design, not proof of IMEX availability. |

---

## 5. CI Output File Dictionaries

The following dictionaries are included because the research supplies field-level details. They are **not native SS&C IMEX file definitions**.

### 5.1 Missing prices file — Axys/APX CI context

| Column | Required | Data Type | Description | Axys | APX | Confidence |
|---|---:|---|---|---:|---:|---:|
| Symbol | Yes | `CHAR512` | Security symbol defined in the relevant security master. | Yes | Yes, in CI context | Verified for CI |
| Type | Yes | `CHAR6` | Security type associated with Symbol. | Yes | Yes, in CI context | Verified for CI |
| Name | No | `CHAR128` | Name of security or position with no price. | Yes | Yes, in CI context | Verified for CI |
| WP Account | Yes | `CHAR64` | WebPortfolio account nickname/name for reference. | Integration-specific | Integration-specific | Verified for CI |
| Institution | No | `CHAR128` | Institution where security is held. | Integration-specific | Integration-specific | Verified for CI |

### 5.2 Security translations file — Axys CI context

| Column | Required | Data Type | Description | Confidence |
|---|---:|---|---|---:|
| WP Name | At least one starred source field required | `CHAR128` | Name of security as it appears in WebPortfolio. | Verified for CI |
| WP Ticker | At least one starred source field required | `CHAR6` | WebPortfolio ticker if available. | Verified for CI |
| WP Cusip | At least one starred source field required | `CHAR9` | WebPortfolio CUSIP if available. | Verified for CI |
| Institution | No | `CHAR128` | Institution where security is held. | Verified for CI |
| WP Account # | No | `CHAR128` | WebPortfolio account number if translation is account-specific. | Verified for CI |
| Axys Symbol | Yes | `CHAR512` | Product-specific Axys symbol used to identify the security. | Verified for CI |
| Type | Yes | `CHAR6` | Security type defined in the Axys security master. | Verified for CI |
| Created | Yes | `DATE` | Date translation was first created, `YYYYMMDD`. | Verified for CI |
| Last Modified | Yes | `DATE` | Date translation was last modified, `YYYYMMDD`. | Verified for CI |

### 5.3 Security translations file — APX CI context

| Column | Required | Data Type | Description | Confidence |
|---|---:|---|---|---:|
| WP Name | At least one starred source field required | `CHAR128` | Name of security as it appears in WebPortfolio. | Verified for CI |
| WP Ticker | At least one starred source field required | `CHAR6` | WebPortfolio ticker if available. | Verified for CI |
| WP Cusip | At least one starred source field required | `CHAR9` | WebPortfolio CUSIP if available. | Verified for CI |
| Institution | No | `CHAR128` | Institution where security is held. | Verified for CI |
| WP Account # | No | `CHAR128` | WebPortfolio account number if translation is account-specific. | Verified for CI |
| APX Symbol | Yes | `CHAR512` | Product-specific APX symbol used to identify the security. | Verified for CI |
| Type | Yes | `CHAR6` | Security type defined in the APX security master. | Verified for CI |
| Created | Yes | `DATE` | Date translation was first created, `YYYYMMDD`. | Verified for CI |
| Last Modified | Yes | `DATE` | Date translation was last modified, `YYYYMMDD`. | Verified for CI |

---

## 6. REP / Report-Based Extraction

REP belongs in its own chapter, but it is material to IMEX because practical exports may use report extraction rather than native IMEX objects.

### 6.1 REP32 and connector extraction

| Statement | Axys | APX | Confidence |
|---|---:|---:|---:|
| A Salentica Data Broker Advent connector uses Advent standard reports and macros to generate extracts. | Yes | Yes | Verified for connector |
| The connector requires Advent client tools, including `REP32.exe`, installed on the machine. | Yes | Yes | Verified for connector |
| The connector uses the REP32 engine plus RepLang scripting and macros. | Yes | Yes | Verified for connector |
| The connector can be scheduled/unattended for daily feeds in its context. | Yes | Yes | Verified for connector |

### 6.2 Replang and Report Writer Pro

| Statement | Axys | APX | Confidence |
|---|---:|---:|---:|
| Consultant material describes Replang as Advent’s proprietary report writing language. | Yes | Yes | High Confidence |
| Report Writer Pro-generated reports have `.RPW` extension, according to AdventGuru. | Yes | Yes | Medium Confidence |
| Manually coded reports commonly use `.REP`, according to AdventGuru. | Yes | Yes | Medium Confidence |
| Manually editing Report Writer-generated code can prevent future modification in Report Writer because the Report Writer checksum no longer matches. | Yes | Yes | Medium Confidence |
| AdventGuru recommends backing up `.RPW`, saving modified code as `.REP`, and reapplying manual changes after future Report Writer edits if needed. | Yes | Yes | Medium Confidence |
| Replang can generate CSV, text formats, and Advent file formats, according to AdventGuru. | Yes | Yes | Medium Confidence |

### 6.3 Report examples supported by supplied research

| Report / File | System | Purpose / Evidence | Confidence |
|---|---|---|---:|
| `AMAN.REP` | Axys | Assets Under Management report example in consultant tutorial. | Medium Confidence |
| `e:\axys34\rep` | Axys | Example reports-directory path in consultant tutorial; not a universal installation path. | Medium Confidence |
| `CDIhold.rep` | Axys | Custom WealthTechs AIA holdings extract report added to Axys custom report menu. | Verified for AIA workflow |
| `sipos30` | Axys | Custom CI reconciliation report comparing calculated Axys positions against downloaded custodian positions. | Verified for CI workflow |
| Transaction Summary Report | APX / Advent reports | Report sample displays account transactions maintained by Advent; visible sections include Purchases, Sales, Dividends, Contributions, and Withdrawals. | Medium Confidence |
| APX standard reports | APX | SS&C says APX has a large standard report library. | Verified product-level; names Unknown |

### 6.4 Transaction Summary Report visible fields

The Transaction Summary Report evidence is report-output evidence only. Do not treat these as IMEX field names or database column names.

| Section | Visible Fields | Confidence |
|---|---|---:|
| Purchases | Trade Date, Settle Date, Quantity, Symbol, Security, Unit Price, Amount | Medium Confidence |
| Sales | Trade Date, Settle Date, Quantity, Symbol, Security, Unit Cost, Total Cost, Unit Price, Proceeds, Gain/Loss | Medium Confidence |
| Dividends | Ex-Date, Pay-Date, Symbol, Security, Amount | Medium Confidence |
| Contributions | Trade Date, Settle Date, Quantity, Symbol, Security, Unit Price, Amount | Medium Confidence |
| Withdrawals | Trade Date, Settle Date, Quantity, Symbol, Security, Unit Price, Amount | Medium Confidence |

---

## 7. Processing Behavior

### 7.1 Axys CI daily workflow

This workflow is documented as a ByAllAccounts Custodial Integrator workflow. It is not asserted as the only native Axys workflow.

| Step | Behavior | Confidence |
|---:|---|---:|
| 1 | Review account status to identify failed updates or missing prior-day data. | Verified for CI |
| 2 | Import/download data from WebPortfolio. | Verified for CI |
| 3 | Resolve portfolio-code translation errors. | Verified for CI |
| 4 | Resolve stale account data issues. | Verified for CI |
| 5 | Review price preview. | Verified for CI |
| 6 | Resolve untranslated securities by adding the security in Axys and reloading Axys security data into CI, or by creating a CI security translation. | Verified for CI |
| 7 | Resolve duplicate securities by creating a security translation. | Verified for CI |
| 8 | Export/format/deliver data into Axys. | Verified for CI |
| 9 | Review IMEX logs and export information. | Verified for CI |
| 10 | Accept exported data, updating CI’s transaction counter for later downloads. | Verified for CI |

### 7.2 Axys import behavior by object type in CI workflow

| Object / Data Type | Processing behavior | Confidence |
|---|---|---:|
| Transactions | Delivered to the designated Trade Blotter file for review/posting. | Verified for CI |
| Positions | Written to `ptopost.trn`; may include lot-specific data when enabled/available. | Verified for CI |
| Position lots | Reflected through `imexPositionLots.log` when position lots are enabled. | Verified for CI |
| Prices | Price files are exported for merge and imported to the price folder; file-in-use can fail. | Verified for CI |
| Securities/types | `sec.inf` and `type.inf` are exported/used for matching and generation, but CI does not modify them. | Verified for CI |

### 7.3 APX AIA import and blotter workflow

This workflow is documented as a WealthTechs AIA/APX workflow. It is not asserted as all native APX processing.

| Area | Processing behavior | Confidence |
|---|---|---:|
| Import logs | Advent IMEX Log and Advent IMEX History Log support troubleshooting imports and blotter issues. | Verified for AIA |
| Account data | Account Blotter can add/update account demographic data. | Verified for AIA |
| Initial transactions | AIA can create initial deliver-in transactions from positions when an account has no transactions. | Verified for AIA |
| APX identifier case | The guide describes APX as case-sensitive when configuring vehicle symbols, account codes, and the management-fee symbol. Preserve exact source case; broader field rules remain Unknown. | Verified for the cited AIA workflow |
| Transaction translation | IF/THEN-style logic can transform source-data transaction rows; rules are not case-sensitive in the AIA guide. This is an AIA translation-rule behavior, not a general instruction to compare native transaction codes or security identifiers case-insensitively. | Verified for AIA |
| Special assignment tokens | Examples include `[*-1]`, `[TradeDate]`, and `[SettleDate]`. | Verified for AIA |
| Vehicle filter | Can eliminate transactions, prices, and positions for specific vehicles from appearing in any APX blotter or import output. | Verified for AIA |
| Delete all Positions | AIA Tools Menu includes Delete all Positions; guide says it should be done before posting current position files in APX. | Verified for AIA |
| Cancel Transactions | Creates cancellation trade blotter from historical transaction files; guide warns to use caution and back up accounts. | Verified for AIA |

### 7.4 Transaction-code context note

Supplied transaction research documents observed transaction codes and uppercase cancellation examples. For this IMEX chapter, the key implementation point is:

| Rule | Confidence |
|---|---:|
| Do not interpret transaction import rows from code alone. Sign, security type, source/destination type, source/destination symbol, special security fields, fee mappings, and integration configuration may affect meaning. | High Confidence as design rule; Medium Confidence source evidence |
| Uppercase transaction-code cancellation behavior is observed in Trade Blotter staging/control workflows, but is not proven as a posted-export representation or universal native behavior. | Medium Confidence for integration workflows; posted-export availability and native universality Unknown |

---

## 8. Version Differences and Environment Differences

| System | Version / Environment | Statement | Confidence |
|---|---|---|---:|
| Axys | Axys 3.7 to 3.8 | Consultant evidence says an upgrade required file conversion and some 3.8 files had different formats. | Medium Confidence |
| Axys | 3.8.6 | Salentica Data Broker lists this as minimum supported Axys version for that connector. | Verified for connector |
| Axys | Axys 3.x audit trail | Consultant evidence reports large Audit Trail exports through IMEX may fail in some environments. | Medium Confidence |
| APX | v1.x–v4.x | Consultant evidence says APX retained IMEX functionality but eliminated fixed-format file generation. | Medium Confidence |
| APX | v1.x–v4.x | Consultant evidence says APX can export Axys v3 format. | Medium Confidence |
| APX | 15.2 / 16.1 / 16.2 / 17.1 | Salentica Data Broker lists these as supported versions for that connector. | Verified for connector |
| APX | Later/current versions | Exact IMEX object behavior, field names, logs, and differences are not supplied. | Unknown |
| Axys/APX | Connector workflows | REP32/RepLang/macros can be used by third-party connectors for data extraction. | Verified for connector |

---

## 9. Known Issues / Quirks

| Issue / Quirk | Axys | APX | Confidence | Practical implication |
|---|---:|---:|---:|---|
| File-in-use import failure | Yes | Unknown | Verified for Axys CI | Close target files before import; review IMEX logs. |
| `topost.trn` append behavior | Yes | No | Verified for Axys CI | Existing blotter content may remain; imports may append rather than replace. |
| Generated comment boundaries | Yes | Unknown | Verified for Axys CI | Comment rows may identify generated import blocks; do not assume they post to client files. |
| CI Accept step affects duplicate future downloads | Yes | Unknown | Verified for Axys CI | Failure to Accept can cause already downloaded transactions to appear again. |
| Position import can replace `.pos` files | Yes | No | Verified for Axys CI | Position imports can overwrite replacement position files. |
| Security ambiguity blocks import | Yes | Yes in security research context | Verified for CI | Resolve duplicate/missing securities before import. |
| Symbol alone is not a safe security key | Yes | Yes | High Confidence | Use Symbol + Type in CI context; native primary key still Unknown. |
| Direct Axys file access is version-risky | Yes | No | Medium Confidence | Prefer IMEX/REP/export interfaces unless version-controlled. |
| Report Writer checksum issue | Yes | Yes | Medium Confidence | Manual Replang edits may break later GUI editing in Report Writer Pro. |
| APX public views are limited | No | Yes | Medium Confidence | Validate view coverage before relying on public views for full extracts. |
| APX fixed-format generation eliminated in early APX range | No | Yes | Medium Confidence | Version-specific import/export behavior must be checked. |
| Uppercase cancellation instructions | Yes | Yes | Verified Trade Blotter staging/control examples; posted-export representation and native universality Unknown | Require explicit source-stage evidence and segregate from posted economics. |
| Large Audit Trail IMEX export reliability | Yes | Unknown | Medium Confidence | Audit exports may need operational validation and backup strategy. |

---

## 10. Examples

### 10.1 Axys IMEX-adjacent folder layout

```text
$pathtrn
    topost.trn        # Axys Trade Blotter file used by CI

$pathcli
    *.cli             # Axys portfolio/client files

$pathinf
    sec.inf           # Security information
    type.inf          # Security type information

$pathpri
    *.pri             # Price files

$pathlog
    imexPositions.log
    imexPositionLots.log
    imexPrices.log
```

| Note | Confidence |
|---|---:|
| File/folder names listed above are supported in CI workflow evidence. | Verified for CI |
| `imexTransactions.log` is not included as a verified file name because the supplied research treats it as an inferred pattern rather than confirmed. | Unknown |

### 10.2 Axys CI security translation concept

```text
Incoming source identifiers:
    WP Name
    WP Ticker
    WP Cusip
    Institution
    WP Account #       # account-specific only

Target Axys identifiers:
    Axys Symbol
    Type
```

| Note | Confidence |
|---|---:|
| This structure is verified for CI security translation output. | Verified for CI |
| It is not a native Axys IMEX security-master field dictionary. | Verified caveat |

### 10.3 APX CI security translation concept

```text
Incoming source identifiers:
    WP Name
    WP Ticker
    WP Cusip
    Institution
    WP Account #       # account-specific only

Target APX identifiers:
    APX Symbol
    Type
```

| Note | Confidence |
|---|---:|
| This structure is verified for CI security translation output. | Verified for CI |
| It is not a native APX database, public view, or IMEX field dictionary. | Verified caveat |

### 10.4 APX AIA transaction cancellation example

```csv
acct123,010101,010101,by,csus,appl,100,caus,cash,10000
acct123,010101,010101,BY,csus,appl,100,caus,cash,10000
```

| Interpretation | Confidence |
|---|---:|
| The example demonstrates lowercase `by` transformed to uppercase `BY` while creating an AIA/APX cancellation Trade Blotter. | Verified example syntax for staging/control |
| Full column meanings are not established by supplied material. | Unknown |
| Availability of `BY` as a posted transaction in ordinary extraction paths is established. | No; Unknown |
| Universal native APX cancellation semantics are not established. | Unknown |

### 10.5 REP/report extraction example

```text
REP32.exe
    Advent standard reports
    macros
    RepLang scripts
    scheduled connector extraction
```

| Note | Confidence |
|---|---:|
| This extraction pattern is verified for the Salentica Data Broker connector. | Verified for connector |
| It does not prove that every desired field is available through REP. | Unknown |

---

## 11. Field Dictionary

This table intentionally combines known field labels and known unknowns. It is not a complete IMEX dictionary.

| Field / Artifact | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---:|
| `imex32.exe` | Axys Import/Export executable. | Yes | No | Utility | No | Verified for CI |
| `APXIX.exe` / `apxix.exe` / `ApxIx` | APX Import/Export utility; capitalization/label varies by guide. | No | Yes | Utility | No | High Confidence across AIA/CI guides |
| APX BackOffice Utilities | APX CI prerequisite in supplied workflow evidence. | No | Yes | Utility / prerequisite | No | Verified for CI workflow |
| APX DataPort | APX CI prerequisite in supplied workflow evidence. | No | Yes | Utility / prerequisite | No | Verified for CI workflow |
| Market Data Manager / Interactive Data | Adjacent APX reference-data path for prices, factors, security setup, fixed-income terms, index values, dividends/splits, exchange rates, and dividend-driven blotter postings. | No | Yes | Adjacent non-IMEX path | Maybe | Medium Confidence |
| FinFolio `CLI`, `PRI`, `INF`, `SPLIT.INF`, `PRF`, and `GRP` | Migration-file leads from conversion evidence. | Yes | Maybe | Maybe | Maybe | Medium Confidence; not official object dictionary |
| `REP32.exe` | Report extraction engine/client tool used by connector. | Yes | Yes | No | Yes | Verified for connector |
| `topost.trn` | Axys Trade Blotter file. | Yes | No | Transaction import workflow | Unknown | Verified for CI |
| `ptopost.trn` | Position file written by CI. | Yes | No | Position workflow | No | Verified for CI |
| `didpost.aud` | Audit Trail file for posted transactions based on consultant evidence. | Yes | Yes in consultant evidence | Audit/export related | Unknown | Medium Confidence |
| `sec.inf` | Security information file. | Yes | Yes in APX export context | Security workflow | Unknown | Verified for CI context |
| `type.inf` | Security type information file. | Yes | Yes in APX export context | Security workflow | Unknown | Verified for CI context |
| `*.cli` | Portfolio/client files. | Yes | Conversion context | Related | Unknown | Verified for CI/conversion context |
| `*.pri` | Price files. | Yes | Unknown | Price workflow | Unknown | Verified for CI |
| `*.pos` | Axys position files. | Yes | No | Position workflow | Unknown | Verified for CI |
| Symbol | Security symbol in CI output/missing-prices context. | Yes | Yes | Related | Unknown | Verified for CI |
| Type | Security type paired with Symbol. | Yes | Yes | Related | Unknown | Verified for CI |
| Axys Symbol | Target Axys security identifier in CI translation file. | Yes | No | Related | No | Verified for CI |
| APX Symbol | Target APX security identifier in CI translation file. | No | Yes | Related | No | Verified for CI |
| WP Name | WebPortfolio security name in CI file. | Integration | Integration | No | No | Verified for CI |
| WP Ticker | WebPortfolio ticker in CI file. | Integration | Integration | No | No | Verified for CI |
| WP Cusip | WebPortfolio CUSIP in CI file. | Integration | Integration | No | No | Verified for CI |
| Institution | Financial institution in CI files. | Integration | Integration | No | No | Verified for CI |
| WP Account / WP Account # | WebPortfolio account identifier in CI files. | Integration | Integration | No | No | Verified for CI |
| Portfolio Code | Portfolio identifier in Axys/APX integration contexts. | Yes | Yes | Related | Possibly | Verified for integration context |
| Account Number | APX Account Blotter field example. | Unknown | Yes | Related | Unknown | Verified for AIA |
| Account Name | APX Account Blotter field example. | Unknown | Yes | Related | Unknown | Verified for AIA |
| Account Type | APX Account Blotter field example. | Unknown | Yes | Related | Unknown | Verified for AIA |
| Transaction Type | Observed Axys/APX integration translation field. | Yes | Yes | Related | Unknown | Medium Confidence |
| Transaction Src/Dest Type | Observed Axys/APX integration translation field. | Yes | Yes | Related | Unknown | Medium Confidence |
| Transaction Src/Dest Symbol | Observed Axys/APX integration translation field. | Yes | Yes | Related | Unknown | Medium Confidence |
| Transaction Special Security Type / Symbol | Observed Axys/APX integration translation field. | Yes | Yes | Related | Unknown | Medium Confidence |
| Trade Date | Transaction/report date field. | Observed | Observed | Unknown native | Observed | Medium Confidence |
| Settle Date | Transaction/report date field. | Observed | Observed | Unknown native | Observed | Medium Confidence |
| Quantity | Transaction/report quantity field. | Observed | Observed | Unknown native | Observed | Medium Confidence |
| Unit Price | Report-output field. | Unknown | Observed | Unknown | Observed | Medium Confidence |
| Amount | Report-output field. | Unknown | Observed | Unknown | Observed | Medium Confidence |
| Unit Cost | Report-output field in Sales section. | Unknown | Observed | Unknown | Observed | Medium Confidence |
| Total Cost | Report-output field in Sales section. | Unknown | Observed | Unknown | Observed | Medium Confidence |
| Proceeds | Report-output field in Sales section. | Unknown | Observed | Unknown | Observed | Medium Confidence |
| Gain/Loss | Report-output field in Sales section. | Unknown | Observed | Unknown | Observed | Medium Confidence |
| Ex-Date | Report-output field in Dividends section. | Unknown | Observed | Unknown | Observed | Medium Confidence |
| Pay-Date | Report-output field in Dividends section. | Unknown | Observed | Unknown | Observed | Medium Confidence |

---

## 12. References

The following references are the underlying sources summarized by
`../evidence/Research_12_IMEX.md` and related topic ledgers. The IDs below are
local to this chapter's retained source list.

| ID | Source | Use in this chapter |
|---|---|---|
| S1 | SS&C Advent Axys product page | Axys product/reporting context. |
| S2 | SS&C Advent Portfolio Exchange product page | APX product/reporting context. |
| S3 | SS&C Axys product brief PDF | Axys reporting and Report Writer Pro product-level context. |
| S4 | APX Reports Guide PDF | Identified in research but not opened; not used for specific field claims. |
| S5 | ByAllAccounts Custodial Integrator Axys User Guide | IMEX definition, security translation, missing prices fields, security resolution. |
| S6 | ByAllAccounts Custodial Integrator Axys User Guide | Axys import workflow, IMEX logs, transaction/position/price behavior, Accept step. |
| S7 | ByAllAccounts Custodial Integrator Axys User Guide | Axys folders/files: `topost.trn`, `*.cli`, `sec.inf`, `type.inf`, `*.pri`, `$pathlog`, `ptopost.trn`. |
| S8 | WealthTechs AIA User Manual for Axys Users | `CDIhold.rep`, `$pathCDI`, holdings extract/report context. |
| S9 | WealthTechs AIA User Manual for APX Users | APX blotters, Advent IMEX Log, Advent IMEX History Log, Cancel Transactions. |
| S10 | Salentica Elements Data Broker: SS&C Advent APX & Axys | REP32, standard reports/macros, connector extraction. |
| S11 | Salentica Engage Data Broker: SS&C Advent APX & Axys | Connector versions, REP32 engine, RepLang scripting/macros. |
| S12 | AdventGuru IMEX / integration material | Direct-file-access caution, Axys 3.7/3.8, APX IMEX/fixed-format note, APX SQL options. |
| S13 | AdventGuru REP material | `.RPW`, `.REP`, checksum caveat, Replang modification workflow. |
| S14 | AdventGuru APX reporting material | APX SQL Server, Stored Accounting Functions, Public Views, SSRS, REST API, Replang context. |
| S15 | Consultant tutorial, “How to Add Portfolio Code to Axys Reports” | Replang, `AMAN.REP`, example report directory. |
| S16 | WealthTechs AIA User Manual for APX Users | APX Transaction Translation IF/THEN logic, special assignment tokens, Vehicle Filter. |
| S17 | [ByAllAccounts APX CI guide](https://www.byallaccounts.net/Manuals/Custodial_Integrator/apx/CI_User_Guide.pdf) | `apxix.exe`, APX import/export workflow, and compatibility artifacts. |
| S18 | [WealthTechs AIA APX guide](https://wealthtechs.com/AIADocumentation/APX%20Guide%20-%20AIA%20User%20Manual%20For%20APX%20Users.pdf) | `APXIX.exe`, `imexhist.log`, transformation order, cancellation, and current/historical holdings paths. |
| S19 | [APX 21.1 release](https://www.advent.com/news-and-insights/blog/advent-investment-suite-release-2021-efficiency-trading-compliance/) | Official REST API capability introduction. |
| Research 04 | Supplied Security Master research | `sec.inf`, `type.inf`, security identity, security translation, APX/Axys security-master evidence. |
| Research 05 | Supplied Transactions research | `topost.trn`, transaction fields, APX Transaction Summary Report, cancellation examples, `didpost.aud`. |

---

## 13. Unknowns

The following items remain **Unknown** and should not be promoted to fact without additional source material.

### 13.1 IMEX objects and schemas

| Unknown | Needed evidence |
|---|---|
| Native Axys IMEX object names for transactions, positions, prices, security master, security types, portfolios, classifications, performance, and reports. | Axys IMEX manual, help pages, screenshots, `.inf` control files, sample exports/logs. |
| Native APX IMEX object names for the same data areas. | APX IMEX manual, help pages, sample exports/logs. |
| Complete field dictionary for each native IMEX object. | Vendor data dictionary or sanitized production exports/import specs. |
| Required vs optional fields for native IMEX import. | Vendor import specs or validated sample imports with error logs. |
| Field data types, lengths, precision, date formats, and delimiter/fixed-width rules. | Vendor IMEX manuals and sample files. |
| Native import validation rules and error/error-code schema. | IMEX manuals and logs. |
| Whether Axys/APX IMEX field names are identical. | Paired Axys/APX manuals or sample exports from both systems. |

### 13.2 Control files, command behavior, and logs

| Unknown | Needed evidence |
|---|---|
| Exact IMEX control-file syntax, including `.inf` file structure. | IMEX manual or sample `.inf` files. |
| Command-line switches for `imex32.exe` and APX import/export utility. | Utility documentation or executable help output. |
| Exact installed filename/version for the APX Import/Export utility. Public guides use `ApxIx`, `apxix.exe`, and `APXIX.exe` for the same function. | APX installation documentation or executable listing. |
| Standard Axys IMEX log names outside the CI workflow. | Sample logs from native IMEX runs. |
| APX IMEX log names beyond the verified `imexhist.log`, plus all locations and formats. | Sample logs or APX documentation. |
| Whether IMEX logs are machine-readable enough for automated audit. | Sample logs. |

### 13.3 Performance and reports

| Unknown | Needed evidence |
|---|---|
| Whether `portperf` is a native IMEX object, report output, stored file, or firm-specific convention. | IMEX object list, sample export, or report source. |
| Whether `secperf` is a native IMEX object, report output, stored file, or firm-specific convention. | IMEX object list, sample export, or report source. |
| Whether Axys IMEX can export stored performance values. | Axys IMEX/performance documentation and samples. |
| Whether APX stores or recalculates specific performance values during IMEX/REP extraction. | APX performance documentation, Stored Accounting Function docs, report behavior tests. |
| Exact REP report names for portfolio performance, security performance, holdings, transactions, classifications, and prices. | REP report catalog or `.REP`/`.RPW` source files. |
| Which reports use stored values vs recalculated values. | Report documentation and production tests. |

### 13.4 Direct file/database access

| Unknown | Needed evidence |
|---|---|
| Native Axys proprietary file layouts for `.cli`, `.pri`, `.pos`, `topost.trn`, `didpost.aud`, `sec.inf`, and `type.inf`. | Vendor file-layout documentation or sanitized production files. |
| Whether direct SQL querying of APX is vendor-supported, tolerated, or discouraged. | SS&C support policy or APX technical documentation. |
| APX public view names and fields for securities, transactions, holdings, prices, classifications, and performance. | Public-view list/schema documentation. |
| Stored Accounting Function names, parameters, and fields. | APX documentation. |
| Exact APX REST endpoint/schema coverage and equivalence to each IMEX object. | APX OpenAPI documentation, entitlement matrix, and paired extracts. |

### 13.5 Version differences

| Unknown | Needed evidence |
|---|---|
| IMEX changes between Axys 3.7, 3.8, 3.8.6, and later versions. | Release notes and sample exports. |
| IMEX changes between APX v1.x–v4.x and APX 15.x–17.x. | APX release/version documentation. |
| Whether fixed-format elimination applies beyond the cited APX v1.x–v4.x range. | APX documentation. |
| Whether large Audit Trail IMEX export issues persist in current Axys/APX versions. | Production tests or vendor notes. |

## Research Provenance

The 2026-06-30 Axys IMEX inventory and field-discovery conclusions are
incorporated into Sections 4.4 through 4.6. Their granular supporting claims and
confidence boundaries remain in `../evidence/Research_12_IMEX.md`.
