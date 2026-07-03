# Chapter 02 — Axys Architecture

Repository: AXYS / APX Reference Repository
Chapter: `Chapter_02_Axys_Architecture.md`
Prepared: 2026-06-29
Governing specification: `AXYS_APX_REFERENCE_BLUEPRINT.md`, Version 2.0
Source basis: supplied research files only

---

## Related chapters
- [Chapter_01_Overview.md](Chapter_01_Overview.md) — provides the repository-wide map and evidence conventions.
- [Chapter_03_APX_Architecture.md](Chapter_03_APX_Architecture.md) — contrasts Axys and APX architecture.
- [Chapter_12_Imex.md](Chapter_12_Imex.md) — covers the import/export pathways that move Axys data.
- [Chapter_13_Rep.md](Chapter_13_Rep.md) — covers REP/reporting workflows that sit alongside IMEX.

## 1. Overview

This chapter documents the architecture of **SS&C Advent Axys** as supported by the supplied research material. APX is included only where it clarifies architectural contrast, migration behavior, IMEX behavior, REP behavior, or Axys/APX version differences.

Axys should be treated as a portfolio accounting, performance, and reporting platform with proprietary-database product positioning, file-oriented operational evidence, a report-writing layer based on REP/Replang, and an import/export layer commonly referred to as IMEX.

The strongest supplied evidence supports the following architectural framing.

| Area | Axys finding | APX contrast | Confidence |
|---|---|---|---:|
| Product role | Portfolio accounting, portfolio management, performance measurement, reconciliation, and reporting system. | Integrated portfolio management, accounting, reporting, performance analytics, and client relationship platform. | Verified at product-capability level |
| Data architecture | Vendor material positions Axys as using a proprietary database, while public/practitioner sources show file-oriented operational handling. Exact physical storage internals are not fully documented in supplied material. | Vendor/public sources describe APX as SQL-based or centralized database-oriented. | Axys: High Confidence for combined characterization; APX: Verified / High Confidence |
| Integration | IMEX, REP/report exports, Report Writer Pro, Replang reports, third-party connector workflows, and cautious direct-file handling. | SQL/public views/stored accounting functions/SSRS/REST may also exist in APX environments, in addition to IMEX and reports. | High Confidence for integration categories; details vary |
| Reporting | REP/Replang and Report Writer Pro are central to Axys reporting. | APX has broader reporting architecture, including SSRS-style reporting paths and report packaging, while some Replang/REP usage remains supported by supplied practitioner sources. | Axys: Verified / High Confidence; APX: Medium / High Confidence |
| Version risk | Direct Axys file integrations are risky because file formats may differ by version; Axys 3.7 to 3.8 file conversion is cited in supplied research. | APX IMEX retained functionality in cited APX v1.x through v4.x source, but fixed-format generation was reportedly removed. | Medium Confidence |
| Unknowns | Native file layouts, locking, backup consistency model, scheduler, service names, executable inventory, and full data dictionary are not established. | APX table/view names and exact SQL schemas are outside this chapter and mostly Unknown in supplied material. | Unknown |

### 1.1 Chapter Scope

This chapter covers:

- Axys architecture and subsystem boundaries.
- Axys file-oriented storage evidence.
- Axys folder/file artifacts observed in supplied research.
- IMEX as an Axys import/export mechanism.
- REP/Replang/Report Writer Pro as the reporting layer.
- Trade Blotter and import staging concepts where they affect architecture.
- Security, transaction, holdings, price, cash, corporate action, performance, and classification dependencies at an architecture level.
- APX architectural contrast where supported.
- Known quirks and open unknowns.

This chapter does **not** define a complete native Axys file layout, transaction-code dictionary, IMEX object dictionary, REP language reference, or APX database schema. Those are Unknown unless explicitly supported by supplied source material.

---

## 2. Confidence Labels

| Label | Meaning in this chapter |
|---|---|
| Verified | Directly supported by supplied research material, cited vendor/product source summaries, inspected third-party documentation summarized in the research, or the repository blueprint. |
| High Confidence | Strongly supported by multiple supplied research files or consistent implementation evidence, but not a full vendor technical specification. |
| Medium Confidence | Supported by practitioner, integration, migration, or conversion evidence; useful but not sufficient as a universal product claim. |
| Unknown | Not established by the supplied material. Do not implement or document as fact without additional evidence. |

---

## 3. Axys Architectural Summary

### 3.1 Axys Product Role

| Statement | Confidence | Notes |
|---|---:|---|
| Axys is a portfolio reporting and accounting system used by investment management organizations. | Verified | Supported by supplied architecture and product research. |
| Axys supports portfolio accounting, performance measurement, reconciliation, and flexible reporting. | Verified | Product-level capability evidence. |
| Axys supports holdings, transactions, positions, securities, prices, cash, performance, groups/classifications, and reports as functional domains. | High Confidence | Supported across supplied research files. Exact native storage for each domain is not fully established. |
| Axys supports tax-lot or average-cost accounting, and trade-date or settlement-date accounting. | Verified at product-capability level | Exact storage and posting mechanics are Unknown. |
| Axys supports multi-currency reporting/capabilities at product level. | Verified at product-capability level | Exact FX file/table and calculation mechanics are Unknown. |
| Axys supports Report Writer Pro and predefined reports. | Verified | REP/Replang details remain incomplete. |

### 3.2 Architecture Layers

The supplied research supports the following architecture model for documentation purposes.

```text
External sources / users
    ├── Custodian files
    ├── Broker / OMS / provider files
    ├── Manual entry
    ├── Third-party integration tools
    └── Conversion / migration files

Axys interface layer
    ├── IMEX / Axys Import-Export utility
    ├── Trade Blotter import workflows
    ├── REP / Replang / Report Writer Pro
    ├── Report exports / macros / REP32 connector workflows
    └── Direct native file access (possible in some contexts, high-risk)

Axys accounting/reference layer
    ├── Portfolio/client files
    ├── Security information
    ├── Security type information
    ├── Transactions / blotters / posted activity
    ├── Positions / holdings
    ├── Prices
    ├── Cash-related activity
    ├── Corporate actions / split information
    ├── Performance inputs/results
    └── Groups / classifications

Axys reporting/output layer
    ├── Standard reports
    ├── Custom REP reports
    ├── Report Writer Pro reports
    ├── Excel/text/CSV-style exports
    └── Third-party connector extracts
```

| Layer | Axys | Confidence | Notes |
|---|---|---:|---|
| External source layer | Custodian, broker, aggregation, migration, and user-entered data can feed Axys workflows. | High Confidence | Strongest evidence is third-party integration documentation. |
| IMEX/import-export layer | Axys Import/Export utility is used in third-party workflows to import transactions, positions, and prices. | Verified for those workflows | Native complete object catalog is Unknown. |
| Trade Blotter layer | Transaction imports can route to a Trade Blotter such as `topost.trn` in supplied integration evidence. | Verified for CI workflow; Medium Confidence as broader Axys concept | Exact native blotter schema is Unknown. |
| File-oriented data layer | Axys uses files such as `.cli`, `sec.inf`, `type.inf`, `.pri`, and `split.inf` in supplied conversion/integration evidence; `PRF` and `GRP` also appear as migration leads. | High Confidence for file existence in cited contexts; Medium Confidence for migration leads | Complete native file layout is Unknown. |
| REP/report layer | Axys reports are written in RepLang; `.REP` files such as `AMAN.REP` appear in supplied examples. | Verified for examples | Full RepLang grammar is Unknown. |
| Direct-file layer | Knowledgeable users may read/write direct files, but this is discouraged due to version/file-format risk. | Medium Confidence | Consultant evidence; should be treated as high-risk. |

---

## 4. Axys Data Architecture

### 4.1 Proprietary-Database and File-Oriented Character

The supplied research does not provide a full Axys vendor storage manual. It does support a conservative statement that Axys is proprietary-database in product positioning and file-oriented in practical integration and operations evidence. "File-oriented" is therefore an integration/operations characterization, not a complete physical-storage specification.

| Statement | Confidence | Implementation treatment |
|---|---:|---|
| Axys is positioned by vendor material as using a proprietary database. | Verified | Do not infer relational tables or public schema from this phrase. |
| Axys is characterized in supplied practitioner/industry research as proprietary and file-oriented in practical workflows. | High Confidence | Use as architecture framing, not as a full physical schema claim. |
| Axys should not be reduced to "simple flat files" without qualification. | High Confidence | Public integration evidence shows files, but not a complete open flat-file database specification. |
| Axys is not verified by the supplied material as SQL-based. | Unknown / negative evidence | Do not design Axys extraction assuming SQL tables unless a specific client environment proves otherwise. |
| Axys direct file reads/writes are possible in some practitioner contexts. | Medium Confidence | Treat as unsupported/high-risk unless verified for version and file. |
| Axys file formats may change by version. | Medium / High Confidence | Prefer IMEX, REP, or vendor-supported exports over direct native file parsing. |

### 4.1.1 Artifact-Class Boundary

Do not collapse native data files, IMEX staging files, report source files,
and conversion shorthand into one universal Axys file dictionary. Each artifact
should carry its source context.

| Artifact class | Examples | Treatment |
|---|---|---|
| Native / operational file evidence | `.cli`, `sec.inf`, `type.inf`, `.pri`, `split.inf` in conversion/integration contexts | High Confidence for presence in cited contexts; exact native layout Unknown. |
| Interface / staging files | `topost.trn`, `ptopost.trn`, IMEX logs | Verified for specific integration workflows; not necessarily universal architecture. |
| Report source/output files | `.REP`, `.RPW`, `AMAN.REP`, `CDIhold.rep` | Verified for report workflows; not database schemas. |
| Migration shorthand / conversion leads | `CLI`, `PRI`, `INF`, `PRF`, `GRP` | Medium Confidence unless sample/source docs verify exact meaning. |

### 4.2 File and Folder Artifacts Observed in Supplied Research

The following artifacts are supported by the supplied research. They are not a complete file dictionary.

| File / Folder / Label | Domain | Description | Axys | APX | Confidence | Caveat |
|---|---|---|---:|---:|---:|---|
| `.cli` | Portfolio/client / transactions / account files | Client/account files used in Axys conversion and integration contexts. | Yes | Mentioned in some APX-oriented workflows as an artifact, but native APX meaning is not fully established. | High Confidence for Axys | Exact layout Unknown. |
| `*.cli` | Portfolio/client folder contents | CI traverses client folder/subfolders to build portfolio code lists. | Yes | Unknown | Verified for CI workflow | Not a complete portfolio schema. |
| `$pathcli` | Folder label | Axys client folder label in CI evidence. | Yes | No | Verified for CI workflow | Installation-specific label. |
| `topost.trn` | Trade Blotter | Axys Trade Blotter file used by CI; generated transactions are appended. | Yes | No | Verified for CI workflow | Complete field layout Unknown. |
| `$pathtrn` | Folder label | Axys user folder / Trade Blotter location in CI evidence. | Yes | No | Verified for CI workflow | Installation-specific label. |
| `sec.inf` | Security information | Security information file referenced in Axys integration/conversion research. | Yes | Also exported in APX CI context as compatibility/interface data. | Verified in integration/conversion contexts | Complete field layout Unknown. |
| `SECURITY.INF` | Security master | Uppercase file-name form referenced as a migration/conversion lead. | Yes | Unknown | Medium Confidence | Treat as case/style variant or conversion shorthand until sample files confirm. |
| `type.inf` | Security type information | Security type information file referenced in Axys integration/conversion research. | Yes | Also exported in APX CI context as compatibility/interface data. | Verified in integration/conversion contexts | Complete field layout Unknown. |
| `TYPE.INF` | Security type information | Uppercase file-name form referenced as a migration/conversion lead. | Yes | Unknown | Medium Confidence | Treat as case/style variant or conversion shorthand until sample files confirm. |
| `$pathinf` | Folder label | Axys information folder containing `sec.inf` and `type.inf` in CI evidence. | Yes | No | Verified for CI workflow | Example paths may vary. |
| `.pri` / `*.pri` | Prices | Price files referenced in Axys conversion/integration workflows. | Yes | APX AIA examples include `.pri` files. | Verified for integration contexts | Complete price file layout Unknown. |
| `$pathpri` | Folder label | Axys price folder label in CI evidence. | Yes | No | Verified for CI workflow | Installation-specific label. |
| `split.inf` / `SPLIT.INF` | Corporate actions / splits | Axys securities splits file in conversion research. | Yes | Unknown as native APX file | High Confidence | Complete split layout Unknown. |
| `PRF` / `.PRF` | Performance returns | Migration/conversion source mentions PRF files as performance-return leads. | Yes | Unknown | Medium Confidence | Candidate artifact only; exact native/export status Unknown. |
| `GRP` / `.GRP` | Groups / classifications | Migration/conversion source mentions GRP files as group/color-group leads. | Yes | Unknown | Medium Confidence | Candidate artifact only; exact native/export status Unknown. |
| `.pos` | Positions | Replacement position files created by Position Post in CI evidence. | Yes | Unknown | Verified for CI workflow | Not proven as complete native holdings store. |
| `ptopost.trn` | Position import staging | CI position file written in CSV format; may include lot-specific data where enabled. | Yes | Unknown | Verified for CI workflow | CI artifact, not native spec. |
| `$pathlog` | Logs | Folder where Axys Import/Export logs are written in CI workflow. | Yes | No | Verified for CI workflow | Exact log schema Unknown. |
| `imexPrices.log` | IMEX log | Axys price-import log referenced in CI evidence. | Yes | No | Verified for CI workflow | Log format Unknown. |
| `imexPositions.log` | IMEX log | Axys position-import log referenced in CI evidence. | Yes | No | Verified for CI workflow | Log format Unknown. |
| `imexPositionLots.log` | IMEX log | Used instead of `imexPositions.log` when position lots are imported in CI workflow. | Yes | No | Verified for CI workflow | Log format Unknown. |
| `CDIhold.rep` | REP/report | WealthTechs-provided holdings extract report for AIA workflow. | Yes | Yes in AIA/APX workflow | Verified for AIA workflow | Not a standard Axys vendor report unless separately confirmed. |
| `$pathCDI` | Custom label/path | Custom label mapped to network path for AIA holdings extract workflow. | Yes | Yes in AIA/APX workflow | Verified for AIA workflow | Workflow-specific. |
| `AMAN.REP` | REP/report | Assets Under Management report in Axys customization example. | Yes | Unknown | Verified for example | Not a full report catalog. |
| `.REP` | Report source | Replang report files. | Yes | Possible for APX Replang workflows | Verified for Axys; Medium Confidence for APX | Full rules Unknown. |
| `.RPW` | Report Writer output | Report Writer-created report file extension per practitioner research. | Yes | Yes per practitioner research | Medium / High Confidence | Manual editing checksum behavior reported. |
| `REP32.exe` | Report engine/client tool | Used by third-party connector with standard reports/macros and RepLang. | Yes | Yes | Verified for connector | Command-line syntax Unknown. |

### 4.3 Candidate Native Data Domains

The supplied research supports these architectural domains at the product or integration level, but not their complete native file layout.

| Domain | Axys support | Evidence type | Confidence | Native storage status |
|---|---|---|---:|---|
| Portfolios/accounts | Supported. `.cli` files and portfolio codes are observed. | Product, integration, conversion | High Confidence | Exact schema Unknown. |
| Security master | Supported. `sec.inf`, `type.inf`, symbol/type matching are observed. | Integration, conversion | High Confidence | Exact field dictionary Unknown. |
| Transactions | Supported. Trade Blotter and transaction codes are observed in integration material. | Product, integration, conversion | High Confidence | Official transaction-code matrix and native field layout Unknown. |
| Holdings/positions | Supported. Portfolio Appraisal, POS files, position blotters, `.pos` workflow are observed. | Reports, integration | High Confidence | Stored-vs-calculated model Unknown. |
| Prices | Supported. `.pri`, price imports, price logs are observed. | Integration, conversion | High Confidence | Complete price file layout Unknown. |
| Cash | Supported through transactions, cash/security tokens, sweep examples. | Integration, product | High Confidence for cash activity; Unknown for cash-balance store | Exact cash balance objects Unknown. |
| Corporate actions | Splits supported through `split.inf`; ACA-for-Axys workflow also exists at vendor workflow level. | Conversion, vendor ACA brief | Axys split evidence High Confidence; Axys ACA workflow Verified; fields Unknown | Exact processing model Unknown. |
| Performance | Supported at product level. | Product, reports | Verified at capability level | Stored-vs-recalculated behavior Unknown. |
| Classifications/groups | Supported in reports/grouping. | Product, reports, integration | Verified at capability level | Exact storage/historical behavior Unknown. |
| REP reports | Supported. | Report examples | Verified for Axys | Full report catalog Unknown. |
| IMEX objects | Supported as import/export utility. | Integration docs | Verified for CI workflow | Full object catalog Unknown. |

---

## 5. Axys Executables, Utilities, and Runtime Components

The supplied research identifies only a small subset of executable or utility names. Do not infer a complete Axys executable inventory from this table.

| Executable / Utility | System | Description | Confidence | Caveat |
|---|---|---|---:|---|
| `imex32.exe` | Axys | Axys Import/Export utility referenced by ByAllAccounts Custodial Integrator research. | Verified in CI context | Native invocation syntax and full object catalog Unknown. |
| `pospos32.exe` | Axys | Axys Post Positions utility referenced in security/IMEX research. | Verified in CI context | Full behavior Unknown. |
| `REP32.exe` | Axys/APX | Advent report engine/client tool used by a third-party connector with standard reports/macros, RepLang scripting, and macros. | Verified for connector | Full command syntax Unknown. |
| `APXIX.exe` / `apxix.exe` | APX | APX Import/Export utility/function referenced in supplied APX integration research. | Verified in APX integration context | Whether `ApxIx` and `APXIX.exe` are identical or context-specific remains Unknown. |

### 5.1 Runtime Unknowns

| Unknown | Why it matters |
|---|---|
| Complete Axys executable inventory | Needed for operations, automation, monitoring, and deployment documentation. |
| Axys service names | Needed for support/runbook documentation. |
| Axys scheduler or batch automation mechanisms | Needed for production processing architecture. |
| Axys locking model | Needed for safe imports, reports, backups, and direct file handling. |
| Axys backup consistency requirements | Needed before documenting safe backup/restore procedures. |
| Axys multi-user concurrency model | Needed for integrations running during user activity. |
| Axys log inventory | Needed for auditability and failure diagnosis. |

---

## 6. Axys IMEX Architecture

### 6.1 IMEX Role

IMEX is an import/export mechanism used to move data into and out of Axys/APX-style environments. In supplied Axys integration material, IMEX is explicitly described as the **Axys Import/Export utility**.

| Statement | Axys | APX | Confidence | Notes |
|---|---:|---:|---:|---|
| IMEX is an import/export mechanism. | Yes | Yes in supplied APX tooling context | High Confidence | Exact APX equivalence to Axys IMEX is Unknown. |
| In Axys CI evidence, IMEX imports Transaction, Position, and Price files. | Yes | No | Verified for CI workflow | Native object names Unknown. |
| Axys IMEX logs are retained and reviewable in CI workflow. | Yes | No | Verified for CI workflow | Log schemas Unknown. |
| APX tooling has Advent IMEX Log and Advent IMEX History Log tools. | No | Yes | Verified for AIA workflow | APX native IMEX details Unknown. |
| IMEX is safer than direct native Axys file manipulation for many integrations. | Yes | Not same issue | High Confidence | Based on file-format-change risk. |

### 6.2 Axys IMEX Workflow Example

The following is supported for the ByAllAccounts Custodial Integrator workflow. It should not be generalized into a complete native Axys IMEX specification.

```text
External account / custodian / WebPortfolio data
    ↓
Custodial Integrator download and translation
    ↓
Axys security/type information loaded from Axys (`sec.inf`, `type.inf`)
    ↓
Generated transaction, position, and price files
    ↓
Axys Import/Export utility (`imex32.exe` in CI context)
    ↓
Trade Blotter / position / price import
    ↓
IMEX log review
    ↓
Acceptance step in the integration workflow
```

| Step | Evidence-supported behavior | Confidence |
|---|---|---:|
| Source download | CI downloads external data. | Verified for CI workflow |
| Security matching | CI uses Axys Security Information and Security Type Information. | Verified for CI workflow |
| Transaction generation | CI can export transactions to Axys Trade Blotter in `$pathtrn`. | Verified for CI workflow |
| Position generation | CI can write positions to `ptopost.trn` and use Position Post to create replacement `.pos` files. | Verified for CI workflow |
| Price generation | CI can export/merge/import price files in `$pathpri`. | Verified for CI workflow |
| Logs | CI exposes IMEX logs by imported data type. | Verified for CI workflow |
| Accept step | Accepting export updates CI transaction counter; skipping acceptance can cause prior transactions to re-download next run. | Verified for CI workflow |

### 6.3 IMEX Log Behavior

| Log / UI behavior | Axys | Confidence | Notes |
|---|---:|---:|---|
| One IMEX log tab per imported data type. | Yes | Verified for CI workflow | Exact tab names beyond supplied examples are Unknown. |
| `imexPositionLots.log` used instead of `imexPositions.log` when position lots are imported. | Yes | Verified for CI workflow | CI-specific behavior. |
| One `imexPrices` tab per historical price day when prices are requested for more than the prior business day. | Yes | Verified for CI workflow | CI-specific behavior. |
| Open/in-use target price file can cause import failure; error appears in `imexPrices.log`. | Yes | Verified for CI workflow | Important operational quirk. |

### 6.4 IMEX Object and Field Unknowns

| Question | Status |
|---|---:|
| Official Axys IMEX object name for transactions | Unknown |
| Official Axys IMEX object name for positions/holdings | Unknown |
| Official Axys IMEX object name for security master | Unknown |
| Official Axys IMEX object name for prices | Unknown |
| Official Axys IMEX object name for performance | Unknown |
| Official Axys IMEX object name for classifications/groups | Unknown |
| Whether Axys and APX use identical IMEX object names | Unknown |
| Complete IMEX field lists, data types, required fields, error codes, and rollback behavior | Unknown |
| Whether IMEX can be safely automated by command line in all versions | Unknown |
| Whether `*.imx` control/configuration files exist in a given environment and define object behavior | Unknown |

---

## 7. REP, Replang, and Report Writer Architecture

### 7.1 Axys REP Layer

| Statement | Confidence | Notes |
|---|---:|---|
| Axys reports are written in RepLang, Advent's proprietary report-writing language. | Verified | Supported by supplied REP research. |
| `.REP` files are Axys report source files in supplied examples. | Verified for examples | `AMAN.REP` is the best-supported example. |
| Axys Report Writer Pro is supported for custom reports. | Verified | Product-level evidence. |
| Reports can be copied and modified in text editors in supplied examples. | Verified for examples | Word processors should not be used for source editing. |
| Axys Reports window can display report path/file in a supplied example. | Verified for example | Installation path is not universal. |
| A copied custom report can be run from Reports / Custom / Any Report in supplied example. | Verified for example | Menu path may be version-specific. |

### 7.2 APX Reporting Contrast

| Statement | Confidence | Notes |
|---|---:|---|
| APX has a large standard report library, automated report packaging, dashboards, and flexible custom reporting. | Verified at product level | APX reporting is broader than Axys REP alone. |
| Consultant evidence says APX can still use Report Writer Pro/Replang source edits. | Medium Confidence | Requires APX-specific vendor confirmation for exact versions/environments. |
| APX reporting options may include SQL Server, public views, stored accounting functions, SSRS, REST API, and other database/report tooling. | Medium Confidence | Consultant evidence; exact schema and access permissions Unknown. |
| APX cloud-delivered environments may constrain direct report execution or file access. | Unknown | Vendor deployment-specific evidence needed. |

### 7.3 REP Examples

| Report / File | System | Description | Confidence |
|---|---|---|---:|
| `AMAN.REP` | Axys | Assets Under Management report in supplied CSSI example. | Verified for example |
| `AMAN_XX.REP` | Axys | Example copied/customized report name. | Verified for example |
| `CDIhold.rep` | Axys/APX AIA workflow | WealthTechs-provided holdings extraction report for historical holdings in AIA workflow. | Verified for workflow |
| `sipos30` | Axys CI workflow | Custom reconciliation report comparing calculated positions in Axys versus downloaded custodian positions. | Verified for CI workflow |
| Position Reconciliation report | Axys | Enhanced in Axys 3.8.7 according to supplied REP research. | Verified as named report; file name Unknown |

### 7.4 REP Expression Tokens Observed

These tokens are examples only. They are not a full RepLang dictionary.

| Token / Expression | Meaning in supplied example | Axys | APX | Confidence |
|---|---|---:|---:|---:|
| `#~8portmv` | Prints portfolio market value in `AMAN.REP` example. | Yes | Unknown | Verified for example |
| `$:fileo` | Displays portfolio code in `AMAN.REP` example. | Yes | Unknown | Verified for example |
| `\n` | Carriage return / line break in printed output in example. | Yes | Unknown | Verified for example |
| `.` prefix | Print command marker in example. | Yes | Unknown | Verified for example |
| `#width` | Appears in report width/layout expression; full semantics Unknown. | Yes | Unknown | Medium Confidence |
| `#cnt` | Appears in report width/layout expression; full semantics Unknown. | Yes | Unknown | Medium Confidence |
| `$askport` | Used in a Portfolio Appraisal header example to show entered CLI code. | Yes | Unknown | Verified for example |
| `$:tfile` | Used by CSSI as transaction-summary analog to show CLI file containing a transaction. | Yes | Unknown | Verified as example statement |
| `$firmg` | Used as “Other” sector catch-all in an AUM sector example. | Yes | Unknown | Verified for example |

### 7.5 REP / Report Writer Quirks

| Quirk | System | Confidence | Practical effect |
|---|---|---:|---|
| Edit copies, not original reports. | Axys; likely general | Verified for Axys example | Preserve vendor reports and reduce upgrade risk. |
| Use plain text editors, not word processors. | Axys; likely general for source files | Verified for Axys example | Prevent source corruption. |
| Report Writer-created `.RPW` files may contain checksum behavior; manual edits can impair later GUI editing. | Axys/APX reporting | Medium / High Confidence | Preserve original `.RPW`, copy to `.REP`, document manual changes. |
| Report extraction can be layout-sensitive. | Axys/APX | Medium Confidence | Stable report output requires controlled report source, parameters, and version. |
| Full RepLang grammar is not supplied. | Axys/APX | Unknown | Do not invent syntax or variables. |

---

## 8. Axys Data Domain Dependencies

### 8.1 Portfolio / Client Files

| Statement | Confidence | Notes |
|---|---:|---|
| Axys portfolio/client files appear as `.cli` files in supplied integration and conversion material. | High Confidence | Exact layout Unknown. |
| CI uses the Axys client folder and `*.cli` files to build portfolio code lists. | Verified for CI workflow | `$pathcli` is a CI folder label. |
| `.cli` files may contain client/account and transaction-related data in conversion contexts. | Medium / High Confidence | Exact native meaning by version Unknown. |
| Per-share cost-basis data may be converted only if present in exported `.cli` files in Morningstar conversion evidence. | Medium Confidence | Conversion-specific evidence. |

### 8.2 Security Master and Security Type

| Statement | Axys | APX | Confidence |
|---|---:|---:|---:|
| Security matching uses product security symbol and security type in supplied CI workflows. | Yes | Yes | Verified for CI context |
| `sec.inf` and `type.inf` are referenced as security information and security type information. | Yes | Yes in APX CI export context | Verified in integration contexts |
| Symbol alone may be ambiguous; same symbol can exist under different security types, and ticker/CUSIP variants can duplicate the same security. | Yes | Yes | Verified for CI context |
| Security translations may be required before imports can proceed. | Yes | Yes | Verified for CI context |
| Security type should not be confused with asset class, sector, industry, country, or other classification. | Yes | Yes | High Confidence |

### 8.3 Transactions

| Statement | Confidence | Notes |
|---|---:|---|
| Transactions are central accounting events that affect holdings, cash, lots, cost basis, income, realized gain/loss, performance, reports, IMEX, REP, reconciliation, and audit workflows. | High Confidence | Supported by transaction research and general accounting role. |
| Axys transaction imports can route through `topost.trn` Trade Blotter in supplied CI workflow. | Verified for CI workflow | Complete field layout Unknown. |
| Observed transaction codes include `by`, `sl`, `li`, `lo`, `dv`, `in`, `dp`, `wd`, `ss`, `cs`, `rc`, `pd`, `ai`, `sa`, and uppercase cancellation examples. | Medium Confidence | Observed in third-party integration material; not official code matrix. |
| Transaction meaning can depend on code, sign, security type, source/destination fields, special security fields, configuration, and context. | High Confidence as implementation rule | Do not interpret code alone. |
| Uppercase transaction codes may represent cancellation/deletion in supplied workflows. | Medium Confidence | Do not generalize to all native workflows without vendor documentation. |

### 8.4 Holdings / Positions

| Statement | Confidence | Notes |
|---|---:|---|
| Axys has a Portfolio Appraisal report that displays holdings/assets. | Verified | Supported by supplied holdings research. |
| Consolidated and unconsolidated group settings affect Portfolio Appraisal output. | Verified | Axys Report Writer example. |
| `Portfolio Code` can be added as a Portfolio Appraisal column in supplied Axys Report Writer example. | Verified | Useful for owner portfolio code in group holdings output. |
| AIA workflows can use `CDIhold.rep` for historical holdings calculation. | Verified for AIA workflow | Current-date data may be read differently from historical holdings. |
| Exact native Axys holdings storage separate from reports/transactions/prices is Unknown. | Unknown | Do not claim a persistent holdings table/file without evidence. |

### 8.5 Prices

| Statement | Confidence | Notes |
|---|---:|---|
| Axys price files are observed as `*.pri` in `$pathpri` in supplied CI research. | Verified for CI workflow | Complete layout Unknown. |
| Price imports can fail when target price files are open/in use. | Verified for CI workflow | Check `imexPrices.log`. |
| `mergepri` is cited by practitioner research as a price-file merge command with primary-source precedence. | Medium Confidence | Exact syntax and support status Unknown. |
| Price files can be source/custodian-specific in APX AIA workflows. | Verified for APX AIA workflow | APX price-set schema Unknown. |
| Exact price key, fields, source hierarchy, stale thresholds, split adjustment mechanics, and currency handling are Unknown. | Unknown | Requires price file samples or vendor docs. |

### 8.6 Cash

| Statement | Confidence | Notes |
|---|---:|---|
| Cash activity appears primarily through transaction workflows in supplied evidence. | High Confidence | Exact cash-balance object Unknown. |
| Axys cash-related transaction examples use tokens such as `$cash`, `$income`, `CAUS`, `CASH`, `MMF`, `MARGIN`, and `SHORT`. | High Confidence as observed tokens | Native meanings and universality Unknown. |
| Cash sweeps and intra-account journals may be removed by third-party import tooling. | High Confidence for AIA workflows | Not proven as native Axys behavior. |
| Axys system currency and cash asset-class code can be configured in CI evidence; older Axys versions may use different cash asset-class letters. | High Confidence for CI context | Native settings documentation Unknown. |
| Exact cash balance storage, settled/trade-date cash distinction, and cash REP/IMEX objects are Unknown. | Unknown | Requires samples/docs. |

### 8.7 Corporate Actions

| Statement | Axys | APX | Confidence |
|---|---:|---:|---:|
| `split.inf` is identified as Axys securities splits file in conversion research. | Yes | Unknown | High Confidence |
| Axys conversion packages commonly include `.cli`, `sec.inf`, `split.inf`, `.pri`, and `type.inf`. | Yes | Unknown | High Confidence |
| Distribution reinvestment may appear as Buy + Distribution pairs in conversion evidence. | Yes | Unknown | Verified for conversion behavior only |
| APX has an Advent Corporate Actions workflow involving holdings sent to ACA, cross-reference to action database, review/download, APX Reorg Utility, and Trade Blotter postings. | No | Yes | Verified at APX product/workflow level |
| Exact Axys split file fields, dividend/reorg files, corporate action transaction codes, and APX Reorg Utility field mapping are Unknown. | Yes | Yes | Unknown |

### 8.8 Performance

| Statement | Confidence | Notes |
|---|---:|---|
| Axys supports performance measurement at product level. | Verified | Exact implementation Unknown. |
| Candidate IMEX object names such as `portperf` and `secperf` are not verified by supplied material. | Unknown | Do not use as fact. |
| Whether Axys stores monthly performance, recalculates on demand, or uses both behaviors is Unknown. | Unknown | Critical research gap. |
| Whether security performance contributions foot to portfolio returns by contribution fields or `weight * return` is Unknown. | Unknown | Requires samples. |
| Which reports use stored values versus recalculation is Unknown. | Unknown | Requires report tests/source. |

### 8.9 Groups and Classifications

| Statement | Confidence | Notes |
|---|---:|---|
| Axys supports grouping portfolios by manager, asset class, objective, or firm-defined categories at product level. | Verified | Storage details Unknown. |
| Axys can display performance by asset class, sector, country, or region at product level. | Verified | Historical classification behavior Unknown. |
| Asset Class appears as an Axys export/report field in supplied AdvisorEngine workflow. | Verified for that workflow | Not necessarily native field name. |
| Symbol + security type should be preserved when joining classification data. | High Confidence | Avoid duplicate/ambiguous security matches. |
| Exact classification lookup storage, IMEX object names, effective dating, and historical report behavior are Unknown. | Unknown | Requires source exports/report tests. |

---

## 9. APX Architectural Contrast

This section is included to separate Axys from APX where the supplied research shows architectural differences.

| Dimension | Axys | APX | Confidence |
|---|---|---|---:|
| Core data architecture | Proprietary-database product positioning plus file-oriented practical integration evidence; exact internals unverified. | SQL-based / centralized database platform in vendor/public/historical research. | Axys: High for characterization; APX: Verified / High Confidence |
| Primary integration surfaces | IMEX, Trade Blotter, REP/Replang, Report Writer Pro, report exports, direct file access with caution. | IMEX/APXIX, blotters, SQL/database access, public views, stored accounting functions, SSRS, REST API, report packaging, REP/Replang in some workflows. | High Confidence for categories; field details Unknown |
| Reporting | REP/Replang is central. | SSRS/custom reporting and report packaging appear more prominent; Replang may still exist. | Axys: Verified; APX: Medium / High |
| Files | `.cli`, `sec.inf`, `type.inf`, `.pri`, `split.inf`, `.REP`, `.RPW` appear in supplied Axys evidence. | APX may generate or consume Axys-compatible/interchange files in integration contexts; native SQL tables/views not identified. | High for Axys file evidence; APX native Unknown |
| Deployment | Local/on-prem/server-style integration contexts are supported by supplied connector docs. | Vendor material supports local or cloud-delivered deployment. | Verified for APX product claim; Axys deployment variants Unknown |
| Version behavior | Axys v1 open text, v2 binary, v3 IMEX, v3.7-to-v3.8 conversion cited by practitioner research. | APX v1.x-v4.x IMEX retained but fixed-format generation removed; APX can export Axys v3 format per practitioner research. | Medium Confidence |

### 9.1 APX Content That Belongs in Chapter 03

The following APX topics should be documented in `Chapter_03_APX_Architecture.md`, not here, unless used for contrast:

- APX SQL schema, public views, stored accounting functions.
- APX SSRS report catalog and report server architecture.
- APX REST API coverage.
- APX cloud/local deployment models.
- APX Reorg Utility internals.
- APX performance analytics / attribution internals.
- APX database security and audit trail model.

---

## 10. Integration Architecture Guidance

### 10.1 Preferred Extraction / Import Order

Where multiple options exist, the supplied research supports the following conservative preference for Axys integrations.

| Preference | Mechanism | Rationale | Confidence |
|---:|---|---|---:|
| 1 | Vendor-supported IMEX export/import, where object and fields are known | More stable than direct file parsing; structured import/export path. | High Confidence |
| 2 | REP/Replang custom report with controlled output | Useful when the desired value is report-derived or not exposed through known IMEX objects. | High Confidence |
| 3 | Standard report export with documented parameters and layout | Useful for operational reports, but layout/version sensitive. | Medium / High Confidence |
| 4 | Third-party connector using REP32/macros or approved client tools | Supported for connector workflows. | Verified for connector; environment-specific |
| 5 | Direct native file read | High risk because file formats can change and exact layouts are mostly Unknown. | Medium Confidence as caution |
| 6 | Direct native file write | Highest risk; do not recommend without vendor documentation, backup, version testing, and production controls. | High Confidence as caution |

### 10.2 Source-Surface Model

Every extracted field should record the surface it came from. Do not assume two
surfaces are equivalent until reconciled by portfolio, security, date, currency,
posting status, and report parameters.

```text
Business domain value
    ↓
Possible source surfaces
    ├── Native files / proprietary database
    ├── IMEX export/import
    ├── Trade Blotter / blotters
    ├── REP/Replang report source/output
    ├── APX SSRS reports
    ├── APX SQL/public views/stored accounting functions
    ├── APX REST API
    ├── Connector/macros/scripts
    └── Conversion files
```

### 10.3 Required Metadata for Any Axys Extract

Any Axys extract used for integration, audit, or research should record the following metadata.

| Metadata | Why it matters | Confidence |
|---|---|---:|
| Axys version | File formats and report behavior may vary by version. | High Confidence |
| Source mechanism | Distinguishes IMEX, REP, Report Writer, connector, direct file, or manual export. | High Confidence |
| Source report / object / file name | Needed for reproducibility. | High Confidence |
| Report/IMEX parameters | Dates, portfolio/group, currency, gross/net, consolidation, price source, etc. change output. | High Confidence |
| Run date/time | Important for historical reproducibility and changed-performance audits. | High Confidence |
| Portfolio/account group | Output depends on scope. | High Confidence |
| Output schema and delimiter | Required for stable parsing. | High Confidence |
| Field definitions | Export headers may not equal native field names. | High Confidence |
| Error/log files | Required to prove successful import/export. | High Confidence |
| Local customizations | Custom REP, labels, macros, translations, price logic, and security mappings can change output. | High Confidence |
| Source surface | Distinguishes IMEX, REP, SSRS, SQL, API, native file, connector, or conversion file. | High Confidence |
| Deployment type | Local, hosted, cloud-delivered, Genesis-era, or unknown deployment may change available surfaces. | High Confidence as caution |
| Posting state | Pending/blotter/posted/final status affects reconciliation. | High Confidence |
| Calculation mode | Stored, calculated, report-derived, or unknown values should not be conflated. | High Confidence |
| Raw output hash and parser version | Needed to detect changed extracts and reproduce ingestion. | High Confidence |

### 10.4 Claim-Type Evidence Boundary

| Claim type | Evidence required |
|---|---|
| Report displays value | Report output or report guide. |
| REP token computes value | REP source or Replang documentation. |
| IMEX exports value | IMEX object/export sample. |
| APX SQL stores value | Schema/table/view evidence. |
| Axys native file stores value | File layout/sample/vendor evidence. |
| Value is books-and-records | Firm policy, vendor documentation, or controlled reconciliation. |
| Value is calculated at runtime | Report/function documentation or controlled test. |

### 10.5 Axys vs APX Integration Boundary

| Rule | Confidence | Notes |
|---|---:|---|
| Do not assume Axys has APX SQL access. | Verified by absence / High Confidence | APX SQL evidence does not transfer to Axys. |
| Do not assume APX native storage is Axys file storage. | High Confidence | APX can use compatibility/export artifacts, but native APX architecture is SQL-centered in supplied research. |
| Do not assume Axys and APX IMEX field names are identical. | Unknown | Requires paired object dictionaries or exports. |
| Do not assume report outputs equal IMEX outputs. | High Confidence | Reports can calculate, aggregate, format, or filter values. |
| Do not assume direct files, IMEX, REP, and SQL/public views contain identical values at the same time. | High Confidence | Timing, report options, posting state, price source, and calculation paths may differ. |

---

## 11. Processing Behavior

### 11.1 Supported Processing Behaviors

| Process | Axys behavior | APX contrast | Confidence |
|---|---|---|---:|
| Accounting basis | Axys supports tax-lot or average-cost accounting and trade-date or settlement-date accounting at product level. | APX likely supports comparable accounting workflows, but exact supplied APX evidence is not detailed here. | Axys Verified; APX Unknown in this chapter |
| Reconciliation | Axys product material supports automated reconciliation of trade information, settlement data, transactions, and positions. | APX reconciliation workflows are observed in AIA/position blotter contexts. | Axys Verified at capability level; APX workflow-specific |
| Transaction import | CI can generate Trade Blotter files and use IMEX to import transactions to Axys. | APX integration workflows use APX Trade Blotters and APXIX/APX import/export. | Verified for workflows |
| Position import | CI can generate position files and use Position Post in Axys. | APX CI imports positions into Position Blotter and lots into Lot Blotter when enabled. | Verified for workflows |
| Price import | CI imports price files to Axys and logs errors. | APX AIA has price file update logic and price set logic. | Verified for workflows |
| Historical holdings | AIA historical holdings load can require `CDIhold.rep`; current-date extraction may differ. | APX AIA current-date data can be read from SQL; historical load requires report. | Verified for AIA workflow |
| Corporate actions | Axys split file `split.inf` exists in conversion evidence; ACA-for-Axys can process simple/mandatory events to Trade Blotter. | APX ACA workflow posts reorg transactions to Trade Blotter via Reorg Utility. | Axys High Confidence for split file; Axys/APX ACA Verified workflow |
| Performance calculation | Axys supports performance reporting at product level. | APX supports performance analytics at product level. | Verified at capability level; mechanics Unknown |

### 11.2 Critical Unknown Processing Behaviors

| Unknown | Why it matters |
|---|---|
| Does Axys store, cache, or recalculate performance results for different reports? | Required for performance audit and repeatability. |
| Which reports use stored monthly values versus recalculated date-range values? | Required for explaining changed historical performance. |
| What is the exact Axys file-locking and concurrency model? | Required for safe import/export and backup scheduling. |
| How does Axys recover from partial IMEX imports? | Required for production support and audit. |
| Are transaction imports atomic by file, row, blotter, or batch? | Required for reliable integration. |
| Which logs prove that an import completed successfully? | Required for operations. |
| How are canceled/deleted/reversed transactions represented internally? | Required for audit and historical reconstruction. |
| How do classification changes affect historical reports? | Required for performance and holdings history. |
| How does Axys treat cash sweeps, income cash, margin cash, and short cash natively? | Required for cash reconciliation and performance. |
| How do split corrections affect historical prices, holdings, and performance? | Required for corporate action audits. |
| Do current SS&C-hosted or managed Axys environments differ operationally from legacy on-prem Axys? | Required for file access, REP, IMEX, backups, and automation design. |
| Can `imex32.exe`, `pospos32.exe`, and `REP32.exe` run unattended in each client environment? | Required for production scheduling and monitoring. |
| Are IMEX imports atomic by row, file, blotter, or batch? | Required for rollback, recovery, and audit. |
| Does Axys have firm-level scheduler/batch mechanisms beyond Windows scheduling and third-party tooling? | Required for production runbooks. |
| Do standard Axys reports and custom REP reports use the same calculation path for the same values? | Required for report reconciliation. |
| What is the vendor support boundary for direct native-file reads/writes in current Axys environments? | Required before recommending direct-file integration. |

---

## 12. Version Differences

| Version / Era | Axys finding | APX contrast | Confidence | Notes |
|---|---|---|---:|---|
| Advent Professional Portfolio / Proport era | Earlier Advent system reportedly used open text file structure. | N/A | Medium Confidence | Practitioner evidence only. |
| Axys v1.x | Reportedly maintained similar open file structure. | N/A | Medium Confidence | Needs vendor confirmation. |
| Axys v2.x | Reportedly introduced binary file format. | N/A | Medium Confidence | Needs vendor confirmation. |
| Axys v3.x | IMEX introduced / used for CSV, tab, and fixed-format import/export in practitioner evidence. | APX later maintains IMEX functionality. | Medium / High Confidence | Exact object catalog Unknown. |
| Axys 3.7 to 3.8 | File conversion reportedly required; some resulting 3.8 files had different formats. | N/A | Medium Confidence | Important direct-file-access warning. |
| Axys 3.8.6 | Minimum Axys version listed by Salentica Data Broker connector. | N/A | Verified for that connector | Connector-specific. |
| Axys 3.8.7 | Vendor blog says enhanced Position Reconciliation report, expanded generic date framework, and additional/improved multicurrency reports. | N/A | Verified as vendor release statement | Report file details Unknown. |
| APX v1.x-v4.x | N/A | IMEX functionality reportedly maintained, fixed-format generation eliminated. | Medium Confidence | Practitioner evidence. |
| APX 15.2 / 16.1 / 16.2 / 17.1 | N/A | Listed as supported/tested by Salentica Data Broker connector. | Verified for that connector | Connector-specific. |
| Current APX | N/A | Vendor material supports local or cloud-delivered deployment. | Verified at product level | Deployment-specific REP/IMEX access Unknown. |

---

## 13. Known Issues / Quirks

| Quirk / Issue | System | Confidence | Practical impact |
|---|---|---:|---|
| Direct Axys file access can break across versions. | Axys | Medium / High Confidence | Prefer IMEX/REP; test after upgrades. |
| Axys 3.7 to 3.8 conversion reportedly changed some file formats. | Axys | Medium Confidence | Version-specific direct-file readers need regression testing. |
| `CLI`, `PRI`, `INF`, `PRF`, and `GRP` may be native files, export files, or migration shorthand depending on version/context. | Axys | Medium Confidence | Do not promote these names into a complete file dictionary without sample files or vendor documentation. |
| Open/in-use price files can cause Axys price import failure. | Axys CI workflow | Verified | Review `imexPrices.log`; avoid imports while files are open. |
| CI Accept step affects whether transactions re-download. | Axys CI workflow | Verified | Skipping acceptance can duplicate downloads in later runs. |
| Security ambiguity blocks imports in CI workflows. | Axys/APX CI | Verified | Resolve symbol/type ambiguity and translations before import. |
| Symbol alone is not reliable for security joins. | Axys/APX CI | Verified | Preserve security type with symbol. |
| Transaction code alone is not enough to infer accounting behavior. | Axys/APX integration | High Confidence | Use sign, amount, source/destination, security type, special symbols, and configuration. |
| Uppercase transaction-code cancellation examples are workflow-specific. | Axys/APX integration | Medium Confidence | Do not treat as universal native rule without documentation. |
| REP/Report Writer manual edits can create supportability issues. | Axys/APX reporting | Medium / High Confidence | Preserve originals and document customizations. |
| APX fixed-format IMEX generation reportedly removed in cited versions. | APX | Medium Confidence | Do not design APX extracts assuming fixed-format output. |
| Historical holdings extraction may require report calculation. | Axys/APX AIA | Verified for AIA workflow | Current-date and historical extraction paths can differ. |
| Product pages do not provide field dictionaries. | Axys/APX | Verified by absence | Field names must come from exports, reports, manuals, or production samples. |

---

## 14. Architecture-Level Field and Artifact Dictionary

This dictionary includes only fields, labels, tokens, and artifacts supported in supplied material. It is not a complete Axys/APX data dictionary.

| Field / Artifact | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---:|
| `Portfolio Code` | Portfolio identifier shown in report/export examples. | Yes | Unknown | Unknown | Yes | Verified for Axys report/export examples |
| `APX Portfolio Code` | APX portfolio identifier in integration context. | No | Yes | Related | No | Verified for APX integration workflow |
| `Custodian Account Number` | Custodian-side account number distinct from APX Portfolio Code in AIA context. | Unknown | Yes | Related | No | Verified for AIA/APX workflow |
| `Security` | Security name/description in Axys export/report examples. | Yes | Unknown | Unknown | Yes likely | Verified for one Axys export workflow |
| `Security Symbol` / `Symbol` | Product/external security symbol. | Yes | Yes | Related | Unknown | Verified for CI contexts |
| `Axys Symbol` | Target Axys security symbol in CI security translation. | Yes | No | Related | No | Verified for CI workflow |
| `APX Symbol` | Target APX security symbol in CI security translation. | No | Yes | Related | No | Verified for CI workflow |
| `Security Type` / `Type` | Security type used with symbol for matching. | Yes | Yes | Related | Unknown | Verified for CI contexts |
| `Sec Type Code` | Axys export field in AdvisorEngine workflow. | Yes | Unknown | Unknown | Likely report/export | Verified for one export workflow |
| `Asset Class` | Classification/allocation field in Axys export and product reporting context. | Yes | Likely | Unknown | Likely | High Confidence for Axys |
| `Sector` | Reporting classification category. | Yes | Yes | Unknown | Likely | Verified at product/report level |
| `Industry Group` | APX report classification category in supplied snippet. | Unknown | Yes | Unknown | Likely | Verified only from snippet |
| `Country` | Axys reporting classification category. | Yes | Likely | Unknown | Likely | Verified for Axys product level |
| `Region` | Axys reporting classification category. | Yes | Likely | Unknown | Likely | Verified for Axys product level |
| `Quantity` | Holding quantity in reports/imports. | Yes | Yes | Related | Yes | Verified in report/integration examples |
| `Price` | Holding/report price label; `.pri` price files observed. | Yes | Yes in APX AIA context | Related | Yes likely | Verified for examples |
| `Market Value` | Valuation field in reports. | Yes | Likely | Unknown | Yes | Verified for examples |
| `Pct Assets` | Percent of assets in Portfolio Appraisal example. | Yes | Unknown | Unknown | Yes | Verified for Axys report example |
| `Yield` | Yield in Portfolio Appraisal example. | Yes | Unknown | Unknown | Yes | Verified for Axys report example |
| `APX Transaction Type` | APX transaction type field in ByAllAccounts translation table. | No | Yes | Related | No | Verified for integration field |
| `Axys Transaction Type` | Axys transaction type field in ByAllAccounts translation table. | Yes | No | Related | No | Verified for integration field |
| `Transaction Src/Dest Type` | Source/destination type in transaction translation. | Yes | Yes | Related | No | Verified for integration field |
| `Transaction Src/Dest Symbol` | Source/destination symbol in transaction translation. | Yes | Yes | Related | No | Verified for integration field |
| `Transaction Special Security Type` | Special security type used for some transaction mappings such as fees. | Yes | Yes | Related | No | Verified for integration field |
| `Transaction Special Security Symbol` | Special security symbol used for some transaction mappings such as fees. | Yes | Yes | Related | No | Verified for integration field |
| `$cash` | Cash source/destination symbol in Axys CI translation. | Yes | Unknown | Related | No | High Confidence for Axys CI |
| `$income` | Income source/destination symbol in Axys CI translation. | Yes | Unknown | Related | No | High Confidence for Axys CI |
| `CAUS` | Cash/security type token in AIA examples. | Yes | Yes | Related | No | High Confidence as observed token; expansion Unknown |
| `CASH` | Cash symbol/token in examples. | Yes | Yes | Related | No | High Confidence as observed token |
| `MMF` | Money-market/sweep symbol in examples. | Yes | Yes | Related | No | High Confidence as observed token |
| `MARGIN` | Margin cash/sweep symbol in examples. | Yes | Yes | Related | No | High Confidence as observed token |
| `SHORT` | Short cash/sweep symbol in examples. | Yes | Yes | Related | No | High Confidence as observed token |
| `Perf/CW` | Column in Axys `topost.trn` per CI cash research. | Yes | Unknown | Related | No | High Confidence for CI context |
| `Mark to Market` | Value required for non-system-currency transactions in Axys CI context. | Yes | Unknown | Related | No | High Confidence for CI context |
| `SourceId` | APX price source field shown in AIA price context. | No | Yes | Related | No | Verified for AIA/APX context |
| `#~8portmv` | REP expression printing portfolio market value in sample. | Yes | Unknown | No | Yes | Verified for Axys sample |
| `$:fileo` | REP token displaying portfolio code in sample. | Yes | Unknown | No | Yes | Verified for Axys sample |
| `REP32.exe` | Report engine/client tool used by connector. | Yes | Yes | No | Yes | Verified for connector |
| `imex32.exe` | Axys Import/Export utility. | Yes | No | Yes | No | Verified for CI context |
| `APXIX.exe` / `apxix.exe` | APX Import/Export utility/function. | No | Yes | Yes | No | Verified for APX integration context |
| `pospos32.exe` | Axys Post Positions utility. | Yes | No | Related | No | Verified for CI context |

---

## 15. Examples

### 15.1 Axys CI Import Architecture Example

```text
External source-data
    ↓
Custodial Integrator
    ├── Reads Axys Security Information (`sec.inf`)
    ├── Reads Axys Security Type Information (`type.inf`)
    ├── Resolves portfolio-code translations
    ├── Resolves security translations
    ├── Generates transaction file for Trade Blotter
    ├── Generates position file (`ptopost.trn` in CI workflow)
    └── Generates price files
    ↓
Axys Import/Export utility (`imex32.exe`)
    ├── Imports transactions
    ├── Imports positions / position lots where enabled
    └── Imports prices
    ↓
Log review
    ├── `imexPositions.log` or `imexPositionLots.log`
    └── `imexPrices.log`
```

Classification: Verified for the supplied Custodial Integrator workflow. Native object names and full field layouts remain Unknown.

### 15.2 Axys File/Folders in CI Context

```text
$pathtrn      Axys user / Trade Blotter folder
    topost.trn

$pathcli      Axys client / portfolio folder
    *.cli

$pathinf      Axys information folder
    sec.inf
    type.inf

$pathpri      Axys price folder
    *.pri

$pathlog      Axys Import/Export log folder
    imexPositions.log
    imexPositionLots.log
    imexPrices.log
```

Classification: Verified for the supplied CI workflow. Paths and labels may be configuration-specific.

### 15.3 Axys REP Customization Example

The supplied REP research supports the following `AMAN.REP` customization example.

```replang
.#~8portmv\n
.#~8portmv $:fileo\n
#width #cnt 16* 25+ 16+
#width #cnt 16* 35+ 16+
```

| Item | Meaning in supplied example | Confidence |
|---|---|---:|
| `AMAN.REP` | Assets Under Management report file. | Verified for example |
| `#~8portmv` | Portfolio market value. | Verified for example |
| `$:fileo` | Portfolio code. | Verified for example |
| `\n` | End-of-line / carriage return. | Verified for example |
| `#width #cnt ...` | Width/layout adjustment. | Verified for example; full semantics Unknown |

### 15.4 Security Identity Example

Supplied security master and classification research provides examples where external ticker can map to product symbol and security type.

```text
External ticker:  LMNVX
External name:    LEGG MASON VLE TR INSTL
Product symbol:   524659208
Product type:     efus
```

Interpretation:

| Observation | Confidence |
|---|---:|
| Product symbol may be a CUSIP-like value rather than ticker. | Verified for example |
| Security identity in CI context uses symbol plus security type. | Verified |
| Classification joins should not rely on ticker alone. | High Confidence |
| Whether `efus` is universal across all versions/sites is Unknown. | Unknown |

### 15.5 Transaction Row Example

Supplied transaction and IMEX research includes a public example row.

```csv
acct123,010101,010101,by,csus,appl,100,caus,cash,10000
```

A cancellation example changes `by` to `BY`:

```csv
acct123,010101,010101,BY,csus,appl,100,caus,cash,10000
```

| Element | Interpretation status |
|---|---|
| `acct123` | Account/portfolio code, Medium Confidence. |
| `010101`, `010101` | Date fields, exact meanings Unknown. |
| `by` | Buy transaction code in observed integration example, Medium Confidence. |
| `BY` | Cancellation/deletion form in observed integration example, Medium Confidence. |
| `csus`, `appl`, `caus`, `cash` | Security/source/destination type-symbol fields in observed example, exact native dictionary Unknown. |
| `100`, `10000` | Quantity/amount-like fields, exact field order Unknown. |

Do not treat this example as a complete Axys transaction import layout.

---

## 15.1 Deep IMEX Architecture Update

The 2026-06-30 IMEX deep research reinforces the architectural position of
IMEX as an integration utility rather than a public data model.

| Artifact / concept | Architectural role | Confidence |
|---|---|---:|
| `imex32.exe` | Axys Import/Export utility used by CI workflows. | Verified for CI |
| `pospos32.exe` | Axys Post Positions utility. | Verified for CI |
| `$pathexe`, `$pathtrn`, `$pathcli`, `$pathinf`, `$pathpri`, `$pathlog` | CI-observed folder labels for executables, Trade Blotter, portfolio/client files, security/type information, prices, and logs. | Verified for CI |
| IMEX object catalog | Must be discovered from live Axys, vendor docs, templates, layouts, and logs; not publicly complete. | Unknown |
| REP/report extraction | Architectural alternative when report-shaped values or performance tie-outs are needed. | High Confidence |

Implementation guidance: capture object names, field labels/tokens, templates,
formats, logs, Axys version, source row lineage, and confidence per client
installation. Do not treat a normalized audit schema as an Axys native schema.

---

## 16. Unsupported or Unknown Information

The following items remain unsupported by the supplied material and must be documented as Unknown until additional evidence is provided.

| Area | Unknown |
|---|---|
| Axys physical storage | Full directory structure; native file ownership by domain; complete file layouts; record delimiters; encoding; locking. |
| Axys transaction schema | Complete official transaction-code matrix; `topost.trn` field layout; posted transaction storage; audit trail schema. |
| Axys security master | Complete `sec.inf` and `type.inf` layouts; security primary key; fixed-income fields; classification fields. |
| Axys holdings | Whether holdings are stored, derived, or both; lot-level holdings availability through IMEX/REP. |
| Axys pricing | Complete `.pri` layout; price key; source/price set; stale/missing price reports; split adjustment behavior. |
| Axys cash | Cash balance object; settled versus trade-date cash; income/margin/short cash storage; cash REP/IMEX objects. |
| Axys corporate actions | Exact `split.inf` layout; dividend/reorg files; corporate action transaction codes; split processing formulas. |
| Axys performance | Stored versus recalculated behavior; performance file/table names; `portperf`/`secperf` object names; report footing logic. |
| Axys classifications | Native classification tables/files; effective dating; historical classification behavior; IMEX object names. |
| IMEX | Complete object catalog, field dictionary, command syntax, logs, validation, rollback, scheduling, and version differences. |
| REP/Replang | Full grammar, variables, macros, command-line options, output formats, report catalog. |
| Deployment | Axys services, scheduler, network share requirements, backup model, user permissions, hosting variants. |
| APX contrast | Full APX SQL schema, public views, stored accounting functions, SSRS datasets, API coverage. |

---

## 17. References

This chapter is based on the supplied research and source material only.

| Source file | Used for |
|---|---|
| `AXYS_APX_REFERENCE_BLUEPRINT.md` | Governing editorial specification, confidence-label discipline, chapter structure. |
| `../evidence/Research_02_Axys_Architecture.md` | Core Axys/APX architecture findings, version differences, integration and reporting architecture. |
| `../evidence/Research_04_Security_Master.md` | Security master, `sec.inf`, `type.inf`, `imex32.exe`, `APXIX.exe`, security matching, symbol/type identity. |
| `../evidence/Research_05_Transactions.md` | Transaction lifecycle, Trade Blotter, `topost.trn`, observed transaction fields/codes, cancellation examples. |
| `../evidence/Research_06_Holdings.md` | Portfolio Appraisal, holdings reports, `.cli`, `.pos`, `CDIhold.rep`, position workflows. |
| `../evidence/Research_07_Cash.md` | Cash transaction representation, sweep handling, cash-like tokens, cash data model unknowns. |
| `../evidence/Research_08_Pricing.md` | `.pri`, `$pathpri`, price logs, missing/stale/calculated price behavior, APX price set contrast. |
| `../evidence/Research_09_Corporate_Actions.md` | `split.inf`, Axys conversion files, APX ACA / Reorg Utility / Trade Blotter contrast. |
| `../evidence/Research_10_Performance.md` | Performance capability, stored/recalculated unknowns, candidate performance interfaces. |
| `../evidence/Research_11_Classifications.md` | Asset class/sector/country/region/custom classification research, symbol/type join cautions, version notes. |
| `../evidence/Research_12_IMEX.md` | IMEX definition, files/folders, logs, REP32 connector contrast, object unknowns. |
| `../evidence/Research_13_REP.md` | REP/Replang architecture, Report Writer Pro, `AMAN.REP`, `REP32.exe`, report examples and quirks. |

---

## 18. Chapter Maintenance Notes

Before upgrading any `Unknown` or `Medium Confidence` item in this chapter, obtain one or more of the following:

1. Vendor Axys technical manuals.
2. Vendor IMEX object dictionaries.
3. Vendor RepLang Programmer's Guide.
4. Sanitized Axys file samples: `.cli`, `sec.inf`, `type.inf`, `.pri`, `split.inf`, `topost.trn`, `.pos`.
5. Sanitized IMEX exports/import definitions.
6. Sanitized REP source files and output examples.
7. Production screenshots of IMEX setup, report parameter dialogs, logs, and folder labels.
8. Controlled before/after test cases for transactions, prices, splits, classifications, and performance recalculation.
9. Axys release notes around 3.7, 3.8, 3.8.6, and 3.8.7.
10. APX architecture materials for the separate APX architecture chapter.

Do not convert observed integration behavior into universal native Axys behavior without explicit support.
