# Chapter 15 — Data Dictionary

**Repository:** AXYS / APX Reference Repository  
**Chapter:** `Chapter_15_Data_Dictionary.md`  
**Governing specification:** `AXYS_APX_REFERENCE_BLUEPRINT.md`, Version 2.0  
**Prepared:** 2026-06-29  
**Status:** Draft technical reference chapter based only on supplied research and source material

---

## Related chapters
- [Chapter_01_Overview.md](Chapter_01_Overview.md) — provides the repository-wide map and evidence conventions.
- [Chapter_04_Security_Master.md](Chapter_04_Security_Master.md) — contributes many identity-related field names.
- [Chapter_12_Imex.md](Chapter_12_Imex.md) — contributes the IMEX-related field and token inventory.
- [Chapter_16_Glossary.md](Chapter_16_Glossary.md) — complements the data dictionary with canonical terminology.

## 1. Overview

This chapter is a cross-repository data dictionary for the Axys/APX Reference Repository. It consolidates field names, report labels, file names, utility names, observed tokens, and data concepts that appear in the supplied research material.

This chapter does **not** replace the subject chapters for security master, transactions, holdings, cash, pricing, corporate actions, performance, classifications, IMEX, REP, or reports. Those chapters remain the primary place to document processing behavior in detail. This chapter provides a conservative field-level index and identifies where each field or token is supported, unsupported, or still unknown.

Important distinction:

- A **native field name** is a field proven to exist in Axys/APX internal files, APX database objects, or vendor-defined IMEX/REP objects.
- A **report label** is a label shown in a report or report guide. It must not be assumed to be a database field.
- An **integration field** is a field shown in a third-party integration workflow, conversion guide, or connector file. It must not be assumed to be a native Axys/APX field unless separately verified.
- A **token** is a REP/Replang expression, file name, executable name, path label, transaction code, security type code, or source/destination symbol observed in supplied research.

Unless otherwise stated, all field names in this chapter are **observed labels or tokens**, not complete vendor data dictionary entries.

---

## 2. Confidence Labels

| Confidence | Meaning in this chapter |
|---|---|
| Verified | Directly supported by supplied source material, supplied research, cited vendor/public material inside the research, or observed examples recorded in the research. |
| High Confidence | Strongly supported by the supplied research, but not proven as a complete vendor field definition. |
| Medium Confidence | Plausible and supported by partial, third-party, conversion, consultant, or workflow-specific evidence. |
| Unknown | Not established from the supplied material. Do not implement as fact without additional evidence. |

The repository standard requires that unsupported behavior be marked **Unknown** rather than invented.

---

## 3. Source Precedence for Data Dictionary Entries

When field names or meanings conflict, use the following evidence order.

| Rank | Source Type | Use in Data Dictionary | Confidence Impact |
|---:|---|---|---|
| 1 | Actual IMEX export/import samples from a known Axys/APX version | Verifies literal object names, field names, order, values, and delimiters | Verified for that environment/version |
| 2 | Actual REP/SSRS report output or REP/RDL source from a known version | Verifies report labels, report fields, parameters, and calculated output | Verified for that report/version |
| 3 | Vendor documentation | Authoritative if versioned and specific | Verified if directly applicable |
| 4 | Existing repository research with source traceability | Consolidated evidence | Verified, High Confidence, or Medium Confidence depending source |
| 5 | Consultant or integration documentation | Practical implementation evidence | High Confidence or Medium Confidence |
| 6 | Production observations | Valuable for quirks and behavior | High Confidence when repeatable and documented |
| 7 | General portfolio-accounting concepts | Useful for organizing field families only | Not sufficient for field names |

---

## 4. Data Dictionary Conventions

### 4.1 Standard Field Dictionary Format

The governing blueprint defines the base format:

| Field | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---|
| Example | Example row only | Unknown | Unknown | Unknown | Unknown | Unknown |

### 4.2 Expanded Format Used in This Chapter

Some tables add columns such as **Context**, **Field Family**, **Native vs Label**, or **Notes** where needed.

| Column | Meaning |
|---|---|
| Field / Token | Literal field name, report label, file name, executable, REP expression, transaction code, or observed token. |
| Description | Meaning supported by the supplied research. |
| Context | Where the field/token was observed: Axys, APX, IMEX, REP, report, integration, conversion, or example. |
| Axys | Whether the supplied research supports Axys use/exposure. |
| APX | Whether the supplied research supports APX use/exposure. |
| IMEX | Whether the supplied research supports IMEX/import-export use. |
| REP | Whether the supplied research supports REP/report use. |
| Confidence | Confidence label. |

---

## 5. High-Level Data Model

The supplied research supports the following high-level entities. Exact native files, APX tables, IMEX objects, and report source fields remain incomplete unless explicitly named below.

| Entity / Data Area | Axys | APX | Evidence Status | Confidence |
|---|---|---|---|---|
| Portfolio / account | Supported as a core accounting/reporting entity. | Supported as a core accounting/reporting entity. | Exact native key fields are only partially observed. | Verified at product/entity level; field-level Unknown |
| Security master | Supported; `sec.inf` and `type.inf` appear in integration/conversion evidence. | Supported; security information and security type appear in integration evidence; APX native table names are not supplied. | Symbol + security type appears repeatedly in matching workflows. | Verified in integration context |
| Transactions | Supported; `.cli`, `topost.trn`, and transaction-code examples appear in research. | Supported; Trade Blotter, transaction translation, and Transaction Summary report labels appear in research. | Official complete transaction-code matrix is not supplied. | Verified at entity level; code matrix Medium/Unknown |
| Holdings / positions | Supported through Portfolio Appraisal, position files, `.pos`, and reports. | Supported through Portfolio Appraisal, Position Blotter, SQL/current-date extraction in AIA workflow, and reports. | Native storage model remains Unknown. | Verified at entity level |
| Cash | Represented through transaction activity and cash-like symbols/tokens in integrations. | Represented through transaction activity and cash-like symbols/tokens in integrations. | Native cash balance objects and report names remain Unknown. | High Confidence for integration tokens; native fields Unknown |
| Prices | `*.pri`, `$pathpri`, `imexPrices.log`, calculated price behavior appear in research. | AIA price-file update logic, price set logic, and `SourceId` appear in research. | Native price schema remains Unknown. | Verified for workflow artifacts; native fields Unknown |
| Corporate actions | `split.inf` is supported for Axys split evidence; dividends/reorgs likely transaction-driven but not fully verified. | ACA for APX sends holdings to ACA, runs APX Reorg Utility, posts to APX Trade Blotter. | Exact corporate-action field dictionary Unknown. | Axys split file High Confidence; APX ACA workflow Verified |
| Performance | Product-level capability supported; APX reports expose portfolio/benchmark/attribution labels. | Product-level capability supported; APX reports expose performance/attribution/contribution labels. | Exact IMEX objects such as `portperf`/`secperf` remain unverified. | Product-level Verified; field-level mixed |
| Classifications | Axys supports grouping/reporting by asset class, sector, country, region, etc. | APX report guide supports custom classification, industry group, sector in report context. | Native classification storage remains Unknown. | Report/category support Verified; native fields Unknown |
| IMEX | Axys Import/Export utility observed as `imex32.exe`; imports transaction, position, and price files in CI workflow. | APX import/export observed as `APXIX.exe` / `ApxIx` / IMEX logs in integration contexts. | Exact object list and field dictionary Unknown. | Verified for utilities/workflows; object fields Unknown |
| REP / reports | Axys `.REP`, Replang, `AMAN.REP`, Report Writer Pro, `REP32.exe` are supported. | APX supports SSRS reports and also has REP32/Replang extraction evidence in connector context. | Exact report source dictionaries Unknown. | Verified for examples; full syntax Unknown |

---

## 6. Axys Data Dictionary Entries

### 6.1 Axys Files, Folders, and Utilities

| Field / Token | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---|
| `imex32.exe` | Axys Import/Export utility executable referenced by ByAllAccounts Custodial Integrator research. | Yes | No | Yes | No | Verified in CI context |
| `pospos32.exe` | Axys Post Positions utility referenced by security-master research addendum. | Yes | No | Utility-adjacent | No | Verified in CI context |
| `topost.trn` | Axys Trade Blotter file receiving transaction imports in CI workflow. | Yes | No | Yes, transaction import workflow | Unknown | Verified in CI context |
| `*.cli` | Axys client/portfolio files referenced in conversion and integration research. | Yes | Unknown / APX `.cli` appears in some APX integration notes but native status Unknown | Related | Unknown | Verified for Axys conversion/integration context |
| `.cli` | Advent/Axys client file; Morningstar conversion evidence says cost-basis data may be present if exported. | Yes | Unknown | Related | Unknown | Medium Confidence for field content |
| `sec.inf` | Securities/security information file in Axys conversion and integration evidence. | Yes | APX CI context also references `sec.inf` | Related | Unknown | Verified in conversion/integration context |
| `type.inf` | Security type information file in Axys conversion and integration evidence. | Yes | APX CI context also references `type.inf` | Related | Unknown | Verified in conversion/integration context |
| `split.inf` / `SPLIT.INF` | Securities splits file in Axys conversion/corporate-action research. | Yes | Unknown native APX status | Unknown | Unknown | High Confidence for Axys |
| `*.pri` | Security price files in Axys price folder/integration evidence. | Yes | APX AIA examples use `.pri` file names in price workflow | Related | No | Verified for integration context |
| `*.pos` | Position files created/replaced in Axys Position Post / CI workflow. | Yes | Unknown | Related | No | Verified for CI context |
| `ptopost.trn` | Position file written by CI to `\CI\exported\ptopost.trn`; may contain lot-specific data when enabled/available. | Yes | Unknown | Related | No | Verified for CI context |
| `$pathexe` | Axys executable folder label in CI configuration. | Yes | No | Configuration | No | Verified in CI context |
| `$pathtrn` | Axys user folder / Trade Blotter output folder label in CI configuration. | Yes | No | Configuration | Unknown | Verified in CI context |
| `$pathcli` | Axys client folder label; CI traverses this folder for `*.cli` portfolio files. | Yes | No | Configuration | Unknown | Verified in CI context |
| `$pathinf` | Axys information folder label; research example includes `C:\axys\inf\`. | Yes | No | Configuration | Unknown | Verified in CI context |
| `$pathpri` | Axys price folder label. | Yes | No | Configuration | No | Verified in CI context |
| `$pathlog` | Axys IMEX log folder label. | Yes | No | Configuration | No | Verified in CI context |
| `imexPrices.log` | Axys IMEX price-import log name/tab observed in CI evidence. | Yes | No | Yes | No | Verified in CI context |
| `imexPositions.log` | Position IMEX log tab implied by CI research. | Yes | No | Yes | No | Verified for CI context |
| `imexPositionLots.log` | IMEX log tab used instead of `imexPositions.log` if position lots are used in CI workflow. | Yes | No | Yes | No | Verified for CI context |
| `e:\axys34\rep` | Example Axys report directory path from consultant report-editing example. | Yes, example only | No | No | Yes | Verified as example only |
| `\axys3\rep` | Example report directory used in consultant holdings/report research. | Yes, example only | No | No | Yes | Verified as example only |

### 6.2 Axys REP / Replang Tokens

| Field / Token | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---|
| `AMAN.REP` | Assets Under Management report file in Axys consultant example. | Yes | Unknown | No | Yes | Verified for example |
| `AMAN_XX.REP` | Example copy of `AMAN.REP` for customization. | Yes | Unknown | No | Yes | Verified for example |
| `_aumsect.rep` | User-created copy of `aman.rep` in AUM-by-sector example. | Yes | Unknown | No | Yes | Verified for example |
| `CDIhold.rep` | WealthTechs-provided report for historical holdings calculation in AIA workflow. | Yes | Yes in APX AIA workflow | No | Yes | Verified for AIA workflow |
| `sipos30` | Custom reconciliation report cited by CI research comparing calculated positions to custodian positions. | Yes | Unknown | No | Yes | Verified for CI context |
| `$askport` | Report variable used in consultant Portfolio Appraisal header example to display entered CLI code. | Yes | Unknown | No | Yes | Verified for example |
| `$:fileo` | Replang token used to display portfolio code in Axys report example. | Yes | Unknown | No | Yes | Verified for example |
| `$:tfile` | Replang token described as showing the CLI file containing a transaction in transaction-summary context. | Yes | Unknown | No | Yes | Verified as consultant statement |
| `$firmg` | Replang variable used as “Other” sector catch-all in AUM sector example. | Yes | Unknown | No | Yes | Verified for example |
| `#~8portmv` | Prints portfolio market value in `AMAN.REP` example. | Yes | Unknown | No | Yes | Verified for example |
| `.#~8portmv\n` | Print expression for portfolio market value followed by line break in consultant example. | Yes | Unknown | No | Yes | Verified for example |
| `.` prefix | Replang print command marker in consultant example. | Yes | Unknown | No | Yes | Verified for example |
| `\n` | Carriage return / end-of-line marker in consultant example. | Yes | Unknown | No | Yes | Verified for example |
| `#width` | Appears in sample report layout expression; full semantics not supplied. | Yes | Unknown | No | Yes | Medium Confidence |
| `#cnt` | Appears in sample report layout expression; full semantics not supplied. | Yes | Unknown | No | Yes | Medium Confidence |
| `#width #cnt 16* 25+ 16+` | Sample width/layout expression modified in consultant report example. | Yes | Unknown | No | Yes | Verified for example; semantics Medium Confidence |

### 6.3 Axys Report / Export Labels

| Field / Label | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---|
| `Portfolio Name` | Required column in AdvisorEngine Axys XLS asset import workflow. | Yes | Unknown | Unknown | Likely report/export | Verified for export workflow |
| `Portfolio Code` | Portfolio identifier column in AdvisorEngine Axys asset export and Portfolio Appraisal customization. | Yes | Unknown | Unknown | Yes | Verified for report/export workflows |
| `Security` | Security name/description column in Axys asset export / Portfolio Appraisal sample. | Yes | Unknown | Unknown | Yes | Verified for report/export workflow |
| `Sec Type Code` | Security type code column in AdvisorEngine Axys asset export. | Yes | Unknown | Unknown | Likely report/export | Verified for export workflow |
| `Security Symbol` | Security symbol column in AdvisorEngine Axys asset export. | Yes | Unknown | Unknown | Likely report/export | Verified for export workflow |
| `Security Type` | Security type column/label in Axys asset export and integration matching. | Yes | Unknown | Unknown | Likely report/export | Verified in export/integration context |
| `Market Value` | Valuation amount in Axys Portfolio Appraisal and asset export examples. | Yes | Unknown | Unknown | Yes | Verified for report/export workflow |
| `Quantity` | Holding quantity in Portfolio Appraisal and asset export examples. | Yes | Unknown | Unknown | Yes | Verified for report/export workflow |
| `Asset Class` | Classification field in AdvisorEngine Axys asset export. | Yes | Unknown | Unknown | Likely report/export | Verified for export workflow |
| `Price` | Holding price displayed in Axys Portfolio Appraisal sample. | Yes | Unknown | Unknown | Yes | Verified for sample |
| `Pct Assets` | Percent-of-assets column in Portfolio Appraisal sample. | Yes | Unknown | Unknown | Yes | Verified for sample |
| `Yield` | Yield column in Portfolio Appraisal sample. | Yes | Unknown | Unknown | Yes | Verified for sample |

---

## 7. APX Data Dictionary Entries

### 7.1 APX Utilities, Import/Export, and Blotter Names

| Field / Token | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---|
| `APXIX.exe` | APX import/export function executable referenced in AIA APX research. | No | Yes | Yes | No | Verified in AIA context |
| `ApxIx` | APX Import/Export Utility name in ByAllAccounts APX terminology. | No | Yes | Yes | No | Verified in CI context |
| `Advent IMEX Log` | APX AIA log showing last importing done through IMEX tool. | No | Yes | Yes | No | Verified in AIA context |
| `Advent IMEX History Log` | APX AIA history log showing all importing through IMEX tool. | No | Yes | Yes | No | Verified in AIA context |
| `Trade Blotter` | APX transaction staging/review/import blotter. | Unknown | Yes | Yes | Unknown | Verified in APX workflows |
| `Position Blotter` | APX blotter for imported positions in Custodial Integrator workflow. | Unknown | Yes | Yes | No | Verified in CI workflow |
| `Lot Blotter` | APX blotter for position lots when lots enabled. | Unknown | Yes | Yes | No | Verified in CI workflow |
| `Tax Lot Blotter` | APX AIA blotter used for tax-lot reconciliation workflow. | Unknown | Yes | Yes | No | Verified in AIA workflow |
| `Statement Blotter` | APX AIA blotter for statement transactions and reconciliation. | Unknown | Yes | Yes | No | Verified in AIA workflow |
| `Account Blotter` | APX AIA blotter for account demographic imports. | Unknown | Yes | Yes | No | Verified in AIA workflow |
| `Initial Transaction Blotter` | APX AIA blotter for initial deliver-in transactions generated from positions. | Unknown | Yes | Yes | No | Verified in AIA workflow |
| `Pending Blotters` | APX AIA pending blotter concept. | Unknown | Yes | Yes | No | Verified in AIA workflow |
| `Dividend Adjustment Blotter` | APX AIA blotter name/concept recorded in IMEX research. | Unknown | Yes | Yes | No | Verified in AIA workflow |
| `Statement Transactions` | APX portfolio tab for statement transactions in AIA workflow. | No | Yes | No | UI/report context | Verified in AIA workflow |
| `Transactions` | APX portfolio tab for posted trade-blotter transactions in AIA workflow. | Unknown | Yes | No | UI/report context | Verified in AIA workflow |
| `APX Reorg Utility` | APX utility run in Advent Corporate Actions workflow before transactions post to Trade Blotter. | No | Yes | Unknown | No | Verified for ACA/APX workflow |
| `Automation Results` | Email summary after scripted ACA processing. | No | Yes | Unknown | No | Verified for ACA/APX workflow |
| `$pathCDI` | Custom label mapped to a network path for AIA holdings extract workflow. | Yes | Yes | Workflow setting | Report output destination | Verified for AIA workflow |
| `Holdings Extract Folder (h.CDI)` | AIA setting mapped to holdings extract output folder. | Yes | Yes | Workflow setting | Report output destination | Verified for AIA workflow |
| `cdirecon` | Typical holdings extract group in AIA workflow. | Unknown | Yes | Workflow setting | REP/report | Verified as typical, not mandatory |

### 7.2 APX Security and Account Fields / Labels

| Field / Label | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---|
| `APX Symbol` | APX security symbol in security translation examples. | Unknown | Yes | Import/export context | Unknown | Verified in CI context |
| `APX Security Type` | APX security type in security translation examples. | Unknown | Yes | Import/export context | Unknown | Verified in CI context |
| `APX Type` | Alternate label for APX security type in translation examples. | Unknown | Yes | Import/export context | Unknown | Verified in CI context |
| `APX Portfolio Code` | APX portfolio identifier used in Custodial Integrator delivery/translation. | No | Yes | Yes in CI workflow | No | Verified in CI workflow |
| `Account #` | Account identifier displayed in APX sample reports. | Unknown | Report output label | Unknown | Reports | Verified as report label only |
| `Account Number` | APX account demographic field example in AIA Account Blotter. | Unknown | Yes | Related | No | Verified in AIA context |
| `Account Name` | APX account demographic field example in AIA Account Blotter. | Unknown | Yes | Related | No | Verified in AIA context |
| `Account Type` | APX account demographic field example in AIA Account Blotter. | Unknown | Yes | Related | No | Verified in AIA context |
| `Custodian Account Number` | AIA/APX account filter field; distinct from APX Portfolio Code in AIA guide. | Unknown | Yes | Related | No | Verified in AIA context |
| `SourceId` | Price source field shown in AIA APX price context; not a security-master field. | No | Yes | Price import context | Unknown | Verified in AIA/APX context only |

### 7.3 APX Report Labels

These are report labels from APX report-guide research. They are not verified database field names.

| Field / Label | Description | Axys | APX | IMEX | REP / Reports | Confidence |
|---|---|---:|---:|---:|---:|---|
| `Market Value` | Report output measure in APX reports. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Revenue` | Account Distribution report output measure. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Effective Rate` / `Eff. Rate` | Account Distribution report output measure. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Count` | Account Distribution report output measure. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `AUM` | Account Distribution report output measure. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Trade Date` | Transaction Summary report date label. | Unknown | Yes | Unknown | APX Transaction Summary | Verified as report label |
| `Settle Date` | Transaction Summary report settlement date label. | Unknown | Yes | Unknown | APX Transaction Summary | Verified as report label |
| `Ex-Date` | Dividend section report label in Transaction Summary sample. | Unknown | Yes | Unknown | APX Transaction Summary | Verified as report label |
| `Pay-Date` | Dividend section report label in Transaction Summary sample. | Unknown | Yes | Unknown | APX Transaction Summary | Verified as report label |
| `Quantity` | Holding/transaction quantity label. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Security` | Security description label. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Symbol` | Security symbol label in transaction/report examples. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Unit Price` | Transaction Summary sample column label. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Price` | Transaction Summary sample column label. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Cost` | Transaction Summary / Portfolio Appraisal report label. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Total Cost` | Transaction Summary report label. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Unit Cost` | Transaction Summary report label. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Proceeds` | Transaction Summary / Realized Gains and Losses report label. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Gain/Loss` | Transaction Summary / Realized Gains and Losses report label. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Cost Basis` | Realized Gains and Losses report label. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Open Date` | Realized Gains and Losses lot label. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Close Date` | Realized Gains and Losses lot label. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Short Term` | Realized gain/loss classification label. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Long Term` | Realized gain/loss classification label. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Percent of Portfolio` | Portfolio Appraisal report description label. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Yield` | Portfolio Appraisal report description label. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Unrealized Gain and Loss` | Portfolio Appraisal report description label. | Unknown | Yes | Unknown | APX reports | Verified as report label |

### 7.4 APX Performance / Attribution / Contribution Labels

These labels are visible in APX report-guide examples and must not be treated as APX SQL field names without report source or schema documentation.

| Field / Label | Description | Axys | APX | IMEX | REP / Reports | Confidence |
|---|---|---:|---:|---:|---:|---|
| `Portfolio Return` | Total portfolio return for report period. | Unknown | Yes | Unknown | APX Attribution / Contribution reports | Verified as report label |
| `Benchmark Return` | Benchmark return for report period. | Unknown | Yes | Unknown | APX Attribution / Contribution reports | Verified as report label |
| `Active Return` | Portfolio return less benchmark return in report context. | Unknown | Yes | Unknown | APX Attribution reports | Verified as report label |
| `Allocation Effect` | Attribution allocation component. | Unknown | Yes | Unknown | APX Attribution reports | Verified as report label |
| `Selection Effect` | Attribution selection component. | Unknown | Yes | Unknown | APX Attribution reports | Verified as report label |
| `Total Effect` | Total attribution effect. | Unknown | Yes | Unknown | APX Attribution reports | Verified as report label |
| `Industry Sector` | Classification/grouping label in APX attribution/contribution examples. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Avg Wgt` | Average weight label in APX attribution/contribution detail. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Return` | Return label for portfolio, benchmark, segment, or security. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Contrib` | Contribution label. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Portfolio` | Portfolio-side report section label. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Benchmark` | Benchmark-side report section label. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Difference` | Portfolio-minus-benchmark comparison section. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Top Contributors` | Ranking section for positive contribution. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Bottom Contributors` | Ranking section for negative contribution. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Top Attribution Effects` | Ranking section by attribution effect. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Bottom Attribution Effects` | Ranking section by attribution effect. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Largest Weights` | Ranking section by average weight. | Unknown | Yes | Unknown | APX reports | Verified as report label |

---

## 8. Shared or Cross-System Field Families

### 8.1 Security Identifier Fields and Tokens

| Field / Token | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---|
| `Symbol` | Security symbol used in matching, price, missing-price, and report contexts. | Yes | Yes | Related | Likely report label | Verified in integration/report contexts |
| `Type` | Security type associated with symbol in integration/export examples. | Yes | Yes | Related | Unknown | Verified in integration contexts |
| `Axys Symbol` | Axys target security symbol in security translation. | Yes | No | Related | No | Verified in CI context |
| `Axys Security Type` | Axys target security type in security translation. | Yes | No | Related | No | Verified in CI context |
| `APX Symbol` | APX target security symbol in security translation. | No | Yes | Related | No | Verified in CI context |
| `APX Security Type` | APX target security type in security translation. | No | Yes | Related | No | Verified in CI context |
| `CUSIP` | External/security identifier used in security matching/translations. | Matching context | Matching context | Unknown | Unknown | Verified in integration context |
| `Ticker` | External identifier used in security matching/translations. | Matching context | Matching context | Unknown | Unknown | Verified in integration context |
| `WP Ticker` | WebPortfolio ticker in CI security translation file. | Integration-specific | Integration-specific | No | No | Verified in CI context |
| `WP Cusip` | WebPortfolio CUSIP in CI security translation file. | Integration-specific | Integration-specific | No | No | Verified in CI context |
| `WP Name` | WebPortfolio security name in CI security translation file. | Integration-specific | Integration-specific | No | No | Verified in CI context |
| `Financial Institution` | Used with security name in custom WebPortfolio translations. | Integration context | Integration context | Unknown | Unknown | Verified in CI context |
| `Institution` | Institution where security is held in missing-prices output. | Integration context | Integration context | No | No | Verified in CI context |
| `Name` | Name of security/position with no price in missing-prices output. | Yes in CI output | Unknown | Related | Unknown | Verified in CI context |
| `Account #` | Account-specific security translation key in CI examples. | Matching context | Matching context | Unknown | Unknown | Verified in CI context |
| `WP Account #` | Account number in WebPortfolio if translation is account-specific. | Integration-specific | Integration-specific | No | No | Verified in CI context |

### 8.2 Security Type Codes and Prefixes

These are observed examples, not a complete security type dictionary.

| Code / Prefix | Description / Observed Context | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---|
| `csus` / `CSUS` | Security type examples in CI / transaction examples. | Yes | Yes | Related | Unknown | Verified as observed example |
| `efus` | Security type example in security translation. | Yes | Yes | Related | Unknown | Verified as observed example |
| `tfus` | Axys duplicate security type example. | Yes | Unknown | Related | Unknown | Verified as observed example |
| `oaus` | Axys duplicate security type example. | Yes | Unknown | Related | Unknown | Verified as observed example |
| `adus` | APX duplicate security type example. | Unknown | Yes | Related | Unknown | Verified as observed example |
| `epus` | Observed in fee/special security contexts; source terminology conflicts between transaction code/security type/expense label. | Yes in conversion context | Yes in APX fee context | Related | Unknown | Medium Confidence; exact native meaning Unknown |
| `exus` | Observed in fee/special security contexts; source terminology conflicts between expense/security type labels. | Yes in conversion context | Yes in APX fee context | Related | Unknown | Medium Confidence; exact native meaning Unknown |
| `CAUS` / `caus` | Cash/security type token in transaction examples. | Yes | Yes | Related | Unknown | Verified as observed token; expansion Unknown |
| `aw`, `br`, `ex`, `ep`, `pi`, `rs` | Prefixes treated as reserved and excluded from CI security matching. | Yes in CI context | Yes in CI context | Matching behavior | No | Verified for CI context only |

### 8.3 Portfolio / Account Identifier Fields

| Field / Label | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---|
| `Portfolio Code` | Axys portfolio/account code in reports/exports; also general portfolio identifier concept. | Yes | Unknown | Related | Yes | Verified for Axys report/export context |
| `APX Portfolio Code` | APX portfolio identifier in APX CI/AIA workflows. | No | Yes | Related | No | Verified for APX CI/AIA context |
| `Portfolio Name` | Axys asset export label. | Yes | Unknown | Unknown | Likely report/export | Verified for export context |
| `Account Number` | APX account field in Account Blotter example. | Unknown | Yes | Related | No | Verified in APX AIA context |
| `Account Name` | APX account field in Account Blotter example. | Unknown | Yes | Related | No | Verified in APX AIA context |
| `Account Type` | APX account field in Account Blotter example. | Unknown | Yes | Related | No | Verified in APX AIA context |
| `Custodian Account Number` | APX AIA field distinct from APX Portfolio Code. | Unknown | Yes | Related | No | Verified in APX AIA context |
| PMS account number | Integration-vendor documentation distinguishes PMS account number from custodian account number. | Yes | Yes | Unknown | Unknown | High Confidence |
| Custodian account number | Custodian-side account identifier distinct from PMS account number. | Yes | Yes | Unknown | Unknown | High Confidence |
| `WP Account` | WebPortfolio account nickname/name in CI missing-prices output. | Integration-specific | Integration-specific | No | No | Verified in CI context |

### 8.4 Date Fields and Labels

| Field / Label | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---|
| `Trade Date` | Transaction execution/economic date label in APX report and transaction concepts. | Expected / observed in examples | Yes | Unknown | APX reports | Verified as APX report label; Axys conceptual High Confidence |
| `Settle Date` | Transaction settlement date label in APX Transaction Summary. | Expected / observed in examples | Yes | Unknown | APX reports | Verified as APX report label; Axys conceptual High Confidence |
| `Ex-Date` | Dividend report section label in APX Transaction Summary sample. | Unknown | Yes | Unknown | APX report | Verified as report label |
| `Pay-Date` | Dividend report section label in APX Transaction Summary sample. | Unknown | Yes | Unknown | APX report | Verified as report label |
| `Open Date` | Realized Gains and Losses lot label. | Unknown | Yes | Unknown | APX report | Verified as report label |
| `Close Date` | Realized Gains and Losses lot label. | Unknown | Yes | Unknown | APX report | Verified as report label |
| Price date | Date for which a security price applies. Exact field name not supplied. | Conceptual | Conceptual | Unknown | Unknown | High Confidence as concept; native field Unknown |
| Valuation / report as-of date | Report date used for holdings/valuation reports. Exact field name not supplied. | Conceptual | Conceptual | Unknown | Report parameter likely | High Confidence as concept; native field Unknown |
| Performance start/end date | Period dates for performance reports. Exact field names not supplied. | Conceptual | APX visible behavior | Unknown | Reports | Medium Confidence; field names Unknown |
| Effective date | Corporate action/classification/security-master effective date concept. Exact field name not supplied. | Unknown | Unknown | Unknown | Unknown | Unknown |

### 8.5 Transaction Fields and Codes

#### 8.5.1 Transaction Field Labels / Parameters

| Field / Label | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---|
| Transaction code | Code identifying transaction type. | Observed | Observed | Related | Unknown | High Confidence as concept; official matrix Unknown |
| `APX Transaction Type` | ByAllAccounts APX translation table field. | No | Yes | Import/export context | No | Medium Confidence; integration field |
| `APX Transaction Src/Dest Type` | APX transaction translation field. | No | Yes | Import/export context | No | Medium Confidence; integration field |
| `APX Transaction Src/Dest Symbol` | APX transaction translation field. | No | Yes | Import/export context | No | Medium Confidence; integration field |
| `APX Transaction Special Security Type` | APX transaction translation field for special fee/security logic. | No | Yes | Import/export context | No | Medium Confidence; integration field |
| `APX Transaction Special Security Symbol` | APX transaction translation field for special fee/security logic. | No | Yes | Import/export context | No | Medium Confidence; integration field |
| Source/Destination Type | Cash/security type field in integration examples. | Observed | Observed | Unknown | Unknown | High Confidence in integration context |
| Source/Destination Symbol | Cash/security symbol field in integration examples. | Observed | Observed | Unknown | Unknown | High Confidence in integration context |
| Broker Representative Field | Transaction blotter field populated by `$brok` in AIA workflow. | Observed | Observed | Unknown | Unknown | High Confidence in AIA context |
| `$brok` | Value written to broker representative field in AIA workflow; APX manual says typically defined in `.cli` file per portfolio. | Observed | Observed | Unknown | Unknown | High Confidence in AIA context |
| Lot Location | Axys-era concept used in APX/AIA workflow for lot accounting/custodian tracking. | Yes, described as Axys carryover | Yes in AIA workflow | Unknown | Unknown | Medium Confidence |
| `250` | Default source-file lot-location value in WealthTechs/APX context. | Unknown | Yes | Unknown | Unknown | Medium Confidence; workflow-specific |
| Comment | Transaction import/comment logic field. | Observed | Observed | Unknown | Unknown | Medium Confidence |
| `Perf/CW` | Column in Axys `topost.trn` file according to ByAllAccounts CI. | Yes | Unknown | Related | Unknown | High Confidence for Axys CI context |
| Mark to Market | Field/value required for non-system-currency transactions in ByAllAccounts Axys integration context. | Yes | Unknown | Related | Unknown | High Confidence for Axys CI context |

#### 8.5.2 Observed Transaction Codes

This table is an observed-code catalog only. It is not a complete official Axys/APX transaction-code matrix.

| Code | Observed Meaning / Context | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---|
| `by` | Buy. | Observed | Observed | Trade Blotter/import context | Unknown | Medium Confidence |
| `BY` | Cancellation/deletion/reversal of buy in uppercase-code workflows. | Observed | Observed | Trade Blotter/import context | Unknown | Medium Confidence |
| `sl` | Sell. | Unknown / likely observed in examples | Observed | Trade Blotter/import context | Unknown | Medium Confidence |
| `SL` | Uppercase sell code appears in AIA delete/translation examples. | Unknown | Observed | Trade Blotter/import context | Unknown | Medium Confidence |
| `ss` | Short sale. | Unknown | Observed | Trade Blotter/import context | Unknown | Medium Confidence |
| `SS` | Uppercase short-sale code appears in AIA delete/translation examples. | Unknown | Observed | Trade Blotter/import context | Unknown | Medium Confidence |
| `cs` | Cover short. | Unknown | Observed | Trade Blotter/import context | Unknown | Medium Confidence |
| `CS` | Uppercase cover-short code appears in AIA delete/translation examples. | Unknown | Observed | Trade Blotter/import context | Unknown | Medium Confidence |
| `li` | Deliver in / transfer in / credit / deposit / positive direction depending context. | Observed | Observed | Trade Blotter/import context | Unknown | Medium Confidence |
| `lo` | Deliver out / transfer out / debit / withdrawal / negative direction depending context. | Observed | Observed | Trade Blotter/import context | Unknown | Medium Confidence |
| `dv` | Dividend / income / reinvestment leg. | Unknown | Observed | Trade Blotter/import context | Unknown | Medium Confidence |
| `in` | Income / interest. | Unknown | Observed | Trade Blotter/import context | Unknown | Medium Confidence |
| `rc` | Return of capital. | Unknown | Observed | Trade Blotter/import context | Unknown | Medium Confidence |
| `pd` | Principal paydown / bond return-of-capital case. | Unknown | Observed | Trade Blotter/import context | Unknown | Medium Confidence |
| `ai` | Accrued interest or margin interest depending context. | Unknown | Observed | Trade Blotter/import context | Unknown | Medium Confidence |
| `sa` | Sell accrued interest. | Unknown | Observed | Trade Blotter/import context | Unknown | Medium Confidence |
| `pa` | Reinvested dividend / accrued-interest-related buy-like case; meaning requires further verification. | Unknown | Observed | Trade Blotter/import context | Unknown | Low to Medium Confidence |
| `dp` | Debit / fee-related / tax / service charge / cash-security case. | Observed | Observed | Trade Blotter/import context | Unknown | Medium Confidence |
| `wd` | Withdrawal / cash-security sell case. | Observed | Observed | Trade Blotter/import context | Unknown | Medium Confidence |
| `;` | Journal / comment / other / split in integration table. | Unknown | Observed | Trade Blotter/import context | Unknown | Medium Confidence |
| `ti` / `to` | Opposite transaction pair examples for intra-account cash journal removal. | Observed in AIA context | Observed in AIA context | Integration workflow | Unknown | High Confidence as observed tokens |
| `si` / `so` | Opposite transaction pair examples for intra-account cash journal removal. | Observed in AIA context | Observed in AIA context | Integration workflow | Unknown | High Confidence as observed tokens |
| `tr` / `ts` | Opposite transaction pair examples for intra-account cash journal removal. | Observed in AIA context | Observed in AIA context | Integration workflow | Unknown | High Confidence as observed tokens |

### 8.6 Cash Tokens and Cash-Like Symbols

These tokens are observed in integration workflows. Native cash-balance schema remains Unknown.

| Field / Token | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---|
| `$cash` | Special source/destination symbol used for cash in ByAllAccounts Axys translations. | Yes | Unknown | Related | No | High Confidence for Axys CI |
| `$income` | Special source/destination symbol used for income-related translations. | Yes | Unknown | Related | No | High Confidence for Axys CI |
| `$pty` | Source/destination type token in cash-impacting Axys translations; exact native meaning not expanded. | Yes | Unknown | Related | No | High Confidence as observed token; meaning Unknown |
| `$ity` | Source/destination type token in income/accrued-interest translations; exact native meaning not expanded. | Yes | Unknown | Related | No | High Confidence as observed token; meaning Unknown |
| `$pth` | Source/destination type token used for margin interest; exact native meaning not expanded. | Yes | Unknown | Related | No | High Confidence as observed token; meaning Unknown |
| `CASH` / `cash` | Cash symbol/token in integration examples. | Yes | Yes | Related | Unknown | High Confidence as observed token |
| `MMF` | Money-market/sweep vehicle symbol in examples. | Yes | Yes | Related | Unknown | High Confidence as observed token |
| `MARGIN` / `margin` | Margin cash/sweep symbol in examples. | Yes | Yes | Related | Unknown | High Confidence as observed token |
| `SHORT` / `short` | Short cash/sweep symbol in examples. | Yes | Yes | Related | Unknown | High Confidence as observed token |
| `dvwash` | Dividend wash special symbol excluded from sweep-removal logic. | Yes | Yes in AIA docs | Related | Unknown | High Confidence as observed token; native definition Unknown |
| `dvshrt` | Special symbol excluded from sweep-removal logic. | Yes | Yes | Related | Unknown | High Confidence as observed token; native definition Unknown |
| `dvlong` | Special symbol excluded from sweep-removal logic. | Yes | Yes | Related | Unknown | High Confidence as observed token; native definition Unknown |
| `cashrt` | Special symbol excluded from sweep-removal logic. | Yes | Yes | Related | Unknown | High Confidence as observed token; native definition Unknown |
| `calong` | Special symbol excluded from sweep-removal logic. | Yes | Yes | Related | Unknown | High Confidence as observed token; native definition Unknown |
| `income` | Special symbol excluded from sweep-removal logic. | Yes | Yes | Related | Unknown | High Confidence as observed token; native definition Unknown |
| Cash asset-class code `c` | Default cash asset-class letter used by CI for Axys; older versions may use a different letter. | Yes | Unknown | Related | Unknown | High Confidence for Axys CI |
| `axyscur` | CI parameter described as the currency code defined as Axys system currency. | Yes | No | Configuration | No | High Confidence for Axys CI |
| `defmarkmarket` | CI parameter for Mark to Market value on non-system-currency transactions. | Yes | No | Configuration | No | High Confidence for Axys CI |
| `defperfcw` | CI parameter for `Perf/CW` column in Axys `topost.trn`. | Yes | No | Configuration | No | High Confidence for Axys CI |

### 8.7 Holdings / Position Fields

| Field / Label | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---|
| `Quantity` | Holding/position quantity. | Yes | Yes as report label | Unknown | Yes | Verified in report/export contexts |
| `Security` | Security name/description in holding/report output. | Yes | Yes | Unknown | Yes | Verified in report/export contexts |
| `Price` | Holding price. | Yes | Yes as report label | Unknown | Yes | Verified in report contexts |
| `Market Value` | Holding/account valuation amount. | Yes | Yes | Unknown | Yes | Verified in report contexts |
| `Pct Assets` | Percent-of-assets label in Axys Portfolio Appraisal sample. | Yes | Unknown | Unknown | Yes | Verified for Axys sample |
| `Percent of Portfolio` | Percent-of-portfolio label in APX Portfolio Appraisal description. | Unknown | Yes | Unknown | APX report | Verified as APX report label |
| `Yield` | Yield field in Portfolio Appraisal examples/description. | Yes | Yes | Unknown | Reports | Verified as report label |
| `Portfolio Code` | Owner portfolio code in Axys Portfolio Appraisal custom report. | Yes | Unknown | Unknown | Yes | Verified |
| `Position Blotter name` | APX object receiving imported positions for reconciliation. | Unknown | Yes | Yes | No | Verified in APX CI context |
| `Lot Blotter name` | APX object receiving lots when lots enabled. | Unknown | Yes | Yes | No | Verified in APX CI context |
| `Trade Blotter name` | APX object receiving imported transactions. | Unknown | Yes | Yes | No | Verified in APX CI context |
| `Name` | Missing-prices output field for name of security/position with no price. | Yes | Unknown | Related | No | Verified in CI context |

### 8.8 Pricing Fields and Artifacts

| Field / Token | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---|
| `*.pri` | Price file extension in Axys/APX price-file workflows. | Yes | Yes in AIA APX example | Related | No | Verified for integration contexts |
| `$pathpri` | Axys price-folder label. | Yes | No | Related | No | Verified for CI Axys |
| `imexPrices.log` | Axys price-import log. | Yes | No | Yes | No | Verified for CI Axys |
| `Price File` | Output/import target for prices. | Yes | Yes | Related | Unknown | Verified for integration contexts |
| `Missing Price file` | CI output/category for securities without usable price unless calculated price is available/current. | Yes | Unknown | Related | Unknown | Verified for CI Axys |
| `third party security price` | Price source/state mentioned in CI release notes. | Yes in CI context | Unknown | Unknown | Unknown | Verified for CI Axys |
| `holding price` | Price source/state mentioned in CI release notes. | Yes in CI context | Unknown | Unknown | Unknown | Verified for CI Axys |
| `calculated price` | Price derived by CI from units and market value where custodian/IDC price unavailable. | Yes in CI context | Unknown | Unknown | Unknown | Verified for CI Axys |
| `IDC price` | Pricing-source reference in CI release notes. | Yes in CI context | Unknown | Unknown | Unknown | Verified for CI Axys |
| `SourceId` | Price source label in AIA/APX price context. | No | Yes | Unknown | Unknown | Verified for AIA/APX context only |
| `Price File Update Logic` | AIA/APX setting controlling update/add/replace behavior. | No | Yes | Related | No | Verified for AIA/APX |
| `Price Set Logic` | AIA/APX setting for price sets and custodian-specific price behavior. | No | Yes | Related | No | Verified for AIA/APX |
| `Clean Price File` | AIA/APX setting removing prices for securities held only in filtered accounts. | No | Yes | Related | No | Verified for AIA/APX |
| `Update Existing & Add New` | APX AIA price import option. | No | Yes | Related | No | Verified for AIA/APX |
| `Add New` | APX AIA price import option. | No | Yes | Related | No | Verified for AIA/APX |
| `Replace Entire File` | APX AIA price import option. | No | Yes | Related | No | Verified for AIA/APX |
| `mmddyy_CDI.pri` | Example custom APX/AIA price file name. | No | Yes | Related | No | Verified as example only |
| `mergepri` | Advent script command mentioned by consultant source for merging price files with primary-source precedence. | Yes | Yes | Price-file workflow | No | Medium Confidence |
| Price date | Date of price. Exact native field name not supplied. | Unknown | Unknown | Unknown | Unknown | High Confidence as concept; native Unknown |
| Price value | Numeric security price. Exact native field name not supplied. | Unknown | Unknown | Unknown | Unknown | High Confidence as concept; native Unknown |
| Price source | Vendor/custodian/source of price. | CI mentions sources | `SourceId` observed in AIA context | Unknown | Unknown | Medium Confidence; native Unknown |
| Price set | APX price-set grouping concept. | Unknown | AIA APX setting | Unknown | Unknown | Medium Confidence for APX AIA; native Unknown |
| Factor | Fixed-income/security factor concept. Exact field name not supplied. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Price multiplier | Pricing multiplier concept. Exact field name not supplied. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Accrued interest | Fixed-income/accrual concept; exact price/security field names not supplied. | Unknown | Unknown | Unknown | Unknown | Medium Confidence as concept; native Unknown |

### 8.9 Corporate Action Fields and Artifacts

| Field / Token | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---|
| `split.inf` / `SPLIT.INF` | Axys securities splits file. | Yes | Unknown | Unknown | Unknown | High Confidence for Axys |
| `.veh` | AIA vehicle file transformed to `sec.inf` layout in Axys/APX security import workflows. | Yes in AIA context | Yes in AIA context | Import staging | Unknown | Verified in AIA context |
| Security identifier | Identifier tying corporate action to security. Exact field name not supplied. | Required conceptually | Required conceptually | Unknown | Unknown | Unknown as field |
| Split date / effective date | Date on which split applies. Exact field name not supplied. | Likely in `split.inf` but unverified | Unknown | Unknown | Unknown | Unknown |
| Split ratio / factor | Split ratio or factor. Exact field name not supplied. | Likely in `split.inf` but unverified | Unknown | Unknown | Unknown | Unknown |
| Old security identifier | Security being reorganized. Exact field name not supplied. | Unknown | Unknown | Unknown | Unknown | Unknown |
| New security identifier | Replacement/distributed security. Exact field name not supplied. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Cash-in-lieu amount | Cash from fractional shares. Exact field name not supplied. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Announcement date | Corporate action announcement date. Exact field name not supplied. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Record date | Holder-of-record date. Exact field name not supplied. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Ex-date | Market entitlement date; APX report label observed for dividends. | Unknown | Yes as report label | Unknown | Report | Verified as APX report label only |
| Pay date / Pay-Date | Payment date; APX report label observed. | Unknown | Yes as report label | Unknown | Report | Verified as APX report label only |

### 8.10 Classification Fields

| Field / Label | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---|
| `Asset Class` | Classification/grouping field; Axys product/report/export evidence. | Yes | Likely | Unknown | Likely | High Confidence for Axys; Medium for APX |
| `Sector` | Classification/reporting group. | Yes | Yes | Unknown | Likely | Verified as reporting category |
| `Industry Group` | APX report classification category. | Unknown | Yes | Unknown | Likely | Verified limited to APX report-guide snippet |
| `Industry` | Common classification term; exact Axys/APX field not verified. | Unknown | Unknown | Unknown | Unknown | Unknown |
| `Country` | Classification/reporting group in Axys product material. | Yes | Likely | Unknown | Likely | Verified for Axys reporting; APX Unknown |
| `Region` | Classification/reporting group in Axys product material. | Yes | Likely | Unknown | Likely | Verified for Axys reporting; APX Unknown |
| `Custom Classification` | User-defined classification scheme in APX report-guide snippet. | Unknown | Yes | Unknown | Likely | Verified for APX report snippet; Axys Unknown |
| Manager | Portfolio grouping category in Axys product material. | Yes | Unknown | Unknown | Reports | Verified as Axys grouping category |
| Investment objective | Portfolio grouping category in Axys product material. | Yes | Unknown | Unknown | Reports | Verified as Axys grouping category |
| Label | AdventGuru mentions importing transaction and label data through trade blotter; relationship to classifications is not established. | Yes | Yes | Related | Unknown | Verified term exists; classification relationship Unknown |
| Classification Code | Possible lookup code for classification value. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Classification Name | Possible lookup name for classification value. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Effective Date | Possible effective date of classification assignment. | Unknown | Unknown | Unknown | Unknown | Unknown |

### 8.11 Performance Fields

The supplied research verifies product-level performance capability and selected APX report labels. Exact Axys/APX performance storage fields and IMEX object names remain Unknown.

| Field / Label | Description | Axys | APX | IMEX | REP / Reports | Confidence |
|---|---|---:|---:|---:|---:|---|
| `Portfolio Return` | Portfolio-level return label in APX performance/attribution reports. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Benchmark Return` | Benchmark return label in APX reports. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Active Return` | Portfolio less benchmark return label in APX reports. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Allocation Effect` | Attribution allocation component. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Selection Effect` | Attribution selection component. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Total Effect` | Total attribution effect. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Avg Wgt` | Average weight. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Return` | Return label for report segment/security/portfolio/benchmark. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Contrib` | Contribution label. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| Beginning market value | Performance input concept. Exact field name not supplied. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Ending market value | Performance input concept. Exact field name not supplied. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Net contributions | Performance input concept. Exact field name not supplied. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Gross return | Gross-of-fee return concept. Exact field name not supplied. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Net return | Net-of-fee return concept. Exact field name not supplied. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Security return | Security-level return concept. Exact field name not supplied. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Contribution | Contribution concept. Exact native field name not supplied. | Unknown | APX `Contrib` report label | Unknown | APX reports | Verified only for APX label |
| Local return | Local-currency return concept. Exact field name not supplied. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Currency effect | FX/currency contribution concept. Exact field name not supplied. | Unknown | Unknown | Unknown | Unknown | Unknown |
| `portperf` | Candidate performance IMEX/export object mentioned as unverified in research. | Unknown | Unknown | Unknown | No | Unknown |
| `secperf` | Candidate security performance IMEX/export object mentioned as unverified in research. | Unknown | Unknown | Unknown | No | Unknown |

---

## 9. IMEX Data Dictionary Status

### 9.1 Verified IMEX-Related Items

| Item | Description | Axys | APX | Confidence |
|---|---|---:|---:|---|
| IMEX / Import-Export utility | Import/export mechanism used in Advent workflows. | Yes | Yes | Verified for Axys CI; APX Verified in AIA/CI terminology |
| `imex32.exe` | Axys Import/Export executable. | Yes | No | Verified in CI context |
| `APXIX.exe` | APX import/export executable/function. | No | Yes | Verified in AIA context |
| `ApxIx` | APX Import/Export Utility terminology. | No | Yes | Verified in CI context |
| Transaction files | Imported through Axys IMEX in CI workflow. | Yes | Yes via blotter workflows | Verified for Axys CI; APX workflows Verified, exact object names Unknown |
| Position files | Imported through Axys IMEX / APX Position Blotter workflows. | Yes | Yes | Verified in integration contexts |
| Price files | Imported through Axys IMEX / APX AIA price workflow. | Yes | Yes | Verified in integration contexts |
| IMEX logs | Import logs are retained/viewable in integration workflows. | Yes | Yes | Verified in integration contexts |

### 9.2 Unknown IMEX Object Dictionary

The supplied material does **not** provide a complete vendor IMEX object dictionary. The following remain Unknown.

| IMEX Object / Field Area | Axys | APX | Confidence |
|---|---:|---:|---|
| Security master object name | Unknown | Unknown | Unknown |
| Security type object name | Unknown | Unknown | Unknown |
| Transaction export object name | Unknown | Unknown | Unknown |
| Transaction import object name | Unknown | Unknown | Unknown |
| Holdings/current positions object name | Unknown | Unknown | Unknown |
| Lot-level holdings object name | Unknown | Unknown | Unknown |
| Cash balance object name | Unknown | Unknown | Unknown |
| Price object name | Unknown | Unknown | Unknown |
| Split/corporate-action object name | Unknown | Unknown | Unknown |
| Classification lookup object name | Unknown | Unknown | Unknown |
| Classification assignment object name | Unknown | Unknown | Unknown |
| Portfolio performance object name | Unknown | Unknown | Unknown |
| Security performance object name | Unknown | Unknown | Unknown |
| Benchmark/index return object name | Unknown | Unknown | Unknown |
| IMEX field order | Unknown | Unknown | Unknown |
| IMEX date formats | Unknown | Unknown | Unknown |
| IMEX numeric formats | Unknown | Unknown | Unknown |
| IMEX validation/error fields | Unknown | Unknown | Unknown |

---

## 10. REP / Report Data Dictionary Status

### 10.1 Verified REP / Report Artifacts

| Artifact | Description | Axys | APX | Confidence |
|---|---|---:|---:|---|
| `.REP` files | Report source files associated with Replang; Axys examples supplied in research. | Yes | Possible / APX Replang evidence from consultant and connector sources | Verified for Axys; Medium Confidence for APX |
| Replang | Advent proprietary report-writing language. | Yes | Medium Confidence for continued APX use | Verified for Axys |
| Report Writer Pro | Custom report tool. | Yes | Medium Confidence for APX use | Verified for Axys product capability; APX consultant-supported |
| `REP32.exe` | Advent reporting engine/tool used by Data Broker connector for Axys/APX extraction. | Yes | Yes | Verified for connector |
| SSRS | Microsoft SQL Server Reporting Services; APX report guide says APX investment-management reports are built on SSRS. | No | Yes | Verified for guide-covered APX reports |
| `AMAN.REP` | Axys AUM report file in example. | Yes | Unknown | Verified for example |
| `CDIhold.rep` | Holdings extraction report in AIA workflow. | Yes | Yes | Verified for AIA workflow |

### 10.2 Unknown REP / Report Field Areas

| Area | Axys | APX | Confidence |
|---|---:|---:|---|
| Complete RepLang keyword dictionary | Unknown | Unknown | Unknown |
| Complete Axys report field dictionary | Unknown | N/A | Unknown |
| Complete APX Replang field dictionary | N/A | Unknown | Unknown |
| APX SSRS RDL names | N/A | Unknown | Unknown |
| APX report dataset names | N/A | Unknown | Unknown |
| APX stored procedure/public view dependencies | N/A | Unknown | Unknown |
| Report parameters by report | Unknown | Unknown | Unknown |
| Stored vs recalculated report values | Unknown | Unknown | Unknown |
| Report-to-IMEX reconciliation mappings | Unknown | Unknown | Unknown |

---

## 11. Report Name Index

### 11.1 Axys Report Names / Files Identified

| Report / File | Description | Confidence |
|---|---|---|
| `AMAN.REP` | Assets Under Management report file in consultant example. | Verified for example |
| `AMAN_XX.REP` | Customized copy of `AMAN.REP` in consultant example. | Verified for example |
| `Portfolio Appraisal` | Axys holdings/assets report; Report Writer can customize columns such as Portfolio Code. | Verified |
| `Assets Under Management` | Axys report accessed in consultant example. | Verified for example |
| `Position Reconciliation report` | Axys report enhanced in Axys 3.8.7 vendor blog research. | Verified as named report; file name Unknown |
| `CDIhold.rep` | Custom holdings extract report in WealthTechs AIA workflow. | Verified for workflow |
| `sipos30` | CI-cited reconciliation report comparing calculated positions to downloaded positions. | Verified for CI context |
| Reconciliation report | Used in Morningstar conversion process as of last transaction date. | Verified for conversion workflow |
| Performance History Report | Mentioned in migration/consulting context. | Medium Confidence; exact report catalog status Unknown |

### 11.2 APX Report Names Identified

| Report Name | Category / Context | Confidence |
|---|---|---|
| Account Distribution | Business intelligence / account segmentation | Verified |
| Account Characteristics | Business intelligence | Verified name; details Unknown |
| Account Characteristics (By Custodian) | Business intelligence | Verified name; details Unknown |
| Asset Flows | Business intelligence | Verified name; details Unknown |
| Business Summary Dashboard | Business intelligence | Verified name; details Unknown |
| Activity Profile | Portfolio analytics/activity | Verified name; details Unknown |
| Attribution by Classification | Performance analytics | Verified |
| Attribution Summary | Performance analytics | Verified |
| Attribution by Selected Groupings | Performance analytics | Verified |
| Contribution by Classification | Performance analytics | Verified |
| Contribution Summary | Performance analytics | Verified |
| Contribution Detail | Performance analytics | Verified |
| Risk Statistics | Performance/risk analytics | Verified name; details Unknown |
| Cover Page | Client reporting | Verified name; details Unknown |
| Household Overview | Client reporting | Verified name; details Unknown |
| Portfolio Overview | Client reporting | Verified name; details Unknown |
| Performance Overview | Client reporting | Verified name; details Unknown |
| Risk Overview | Client reporting | Verified name; details Unknown |
| Policy Overview | Client reporting | Verified name; details Unknown |
| Historical Policy Overview | Client reporting | Verified name; details Unknown |
| Allocation Summary | Client reporting | Verified name; details Unknown |
| Equity Overview | Client reporting | Verified name; details Unknown |
| Fixed Income Distribution | Client reporting | Verified name; details Unknown |
| Fixed Income Overview | Client reporting | Verified name; details Unknown |
| Income Projection | Client reporting | Verified name; details Unknown |
| Portfolio Appraisal | Holdings / client reporting; APX guide says holdings by tax lot or position. | Medium Confidence pending full guide capture; name supported |
| Realized Gains and Losses | Client reporting / tax lots / realized gains | Verified |
| Transaction Summary | Transaction listing | Verified |
| Disclaimer and Terms | Client reporting / disclosure | Verified name; details Unknown |

---

## 12. Alias and Synonym Map

This table maps concepts across observed labels. It does not assert identical semantics unless explicitly stated.

| Concept | Axys Labels / Tokens | APX Labels / Tokens | IMEX / Integration Labels | REP / Report Labels | Same Meaning? | Confidence |
|---|---|---|---|---|---|---|
| Portfolio identifier | `Portfolio Code`, `$:fileo`, `$askport` in example context | `APX Portfolio Code`, `Account #` report label | `Portfolio Code`, custodian/account translation fields | `Portfolio Code`, `$:fileo` | Related, not proven identical across all contexts | Medium Confidence |
| Account identifier | `.cli` portfolio/client file context | `Account Number`, `Account #`, `Custodian Account Number` | PMS account number, custodian account number | Account report labels | Related; distinctions matter | High Confidence |
| Security identifier | `Axys Symbol`, `Security Symbol`, `Symbol`, `CUSIP`, `Ticker` | `APX Symbol`, `Symbol`, `CUSIP`, `Ticker` | `WP Ticker`, `WP Cusip`, `WP Name` | `Security`, `Symbol` | Related, not interchangeable | High Confidence |
| Security type | `Axys Security Type`, `Sec Type Code`, `Security Type`, `Type` | `APX Security Type`, `APX Type`, `Type` | `type.inf` | Report labels may show Security Type | Related; source-specific | High Confidence |
| Holding quantity | `Quantity` | `Quantity` | Quantity in transaction/position examples | `Quantity` | Likely same concept; exact precision/semantics Unknown | High Confidence |
| Price | `Price`, `*.pri`, `calculated price` | `Price`, `Unit Price`, `SourceId`, Price Set Logic | Price files | Report labels | Related; source/date semantics Unknown | Medium Confidence |
| Market value | `Market Value`, `#~8portmv` | `Market Value`, AUM | Unknown | Reports | Related; calculation source Unknown | Medium Confidence |
| Cash | `$cash`, `CASH`, `CAUS`, cash asset-class `c` | `CASH`, `CAUS`, `MMF`, `MARGIN`, `SHORT` | Source/destination type/symbol | Cash report names Unknown | Related; native cash model Unknown | Medium Confidence |
| Income | `$income`, `income`, `dv`, `in` | `income`, dividends/interest report sections | Transaction codes/tokens | Income Projection report | Related; transaction semantics context-dependent | Medium Confidence |
| Asset class | `Asset Class` | Custom classification / asset allocation context | Unknown | Allocation reports | Related; storage Unknown | Medium Confidence |
| Sector / industry | Sector, country, region in Axys product material | `Industry Sector`, Industry Group, sector, custom classification | Unknown | Attribution/classification reports | Related; storage Unknown | Medium Confidence |
| Return | Performance capability; exact labels Unknown | `Portfolio Return`, `Benchmark Return`, `Return` | `portperf` candidate Unknown | APX performance reports | APX report labels verified; Axys field Unknown | Medium Confidence |
| Contribution | Conceptual / field Unknown | `Contrib`, contribution report labels | `secperf` candidate Unknown | APX reports | APX label verified; formula Unknown | Medium Confidence |
| Attribution effect | Unknown | `Allocation Effect`, `Selection Effect`, `Total Effect` | Unknown | APX attribution reports | APX labels verified; formula Unknown | Verified as APX labels |

---

## 13. Example Field Sets

These examples are included because they are explicitly supported by supplied research. They are not general import/export templates unless stated.

### 13.1 Axys Asset Export Field Order Example

AdvisorEngine’s Axys asset import workflow expects this field order from an Axys XLS export:

```text
Portfolio Name
Portfolio Code
Security
Sec Type Code
Security Symbol
Security Type
Market Value
Quantity
Asset Class
```

| Interpretation | Confidence |
|---|---|
| Axys can produce an XLS export/report containing portfolio identity, security identity, valuation, quantity, and asset class fields. | Verified for the documented workflow |
| The column labels are export/report labels, not proven native file/database field names. | Verified caveat |
| The workflow does not prove the IMEX object name or IMEX field names. | Verified caveat |

### 13.2 APX Security Translation Example

Observed APX security translation example:

| WP Ticker | WP Name | APX Symbol | APX Type |
|---|---|---|---|
| `LMNVX` | `LEGG MASON VLE TR INSTL` | `524659208` | `efus` |

| Interpretation | Confidence |
|---|---|
| APX matching/translation can map an external ticker to an APX symbol that is not the ticker. | Verified |
| APX security identity in this workflow uses both symbol and type. | Verified |
| `efus` is an example APX security type code in the integration source. | Verified |
| Whether `efus` is universal across all APX versions/sites is Unknown. | Unknown |

### 13.3 Axys Duplicate Security Example

Research records an Axys duplicate-security case where WebPortfolio provided ticker, CUSIP, and name, and Axys contained both ticker and CUSIP entries.

| Condition | Example / Explanation | Confidence |
|---|---|---|
| WebPortfolio provides ticker, CUSIP, and name; Axys has both ticker and CUSIP entries. | NEW PERSPECTIVE FD CL A with ticker `ANWPX` and CUSIP `648018109`; CI cannot determine which Axys security master entry to use. | Verified in integration context |
| Same CUSIP entered twice with different security types. | Examples include `tfus` and `oaus`. | Verified in integration context |

### 13.4 Public Transaction Row Example

A public transaction example row appears in research:

```text
acct123,010101,010101,by,csus,appl,100,caus,cash,10000
```

Tentative interpretation:

| Position | Observed Value | Tentative Meaning | Confidence |
|---:|---|---|---|
| 1 | `acct123` | Account / portfolio code | Medium |
| 2 | `010101` | Date field 1 | Unknown |
| 3 | `010101` | Date field 2 | Unknown |
| 4 | `by` | Transaction code | Medium |
| 5 | `csus` | Security type | Low to Medium |
| 6 | `appl` | Security symbol | Low to Medium |
| 7 | `100` | Quantity | Low to Medium |
| 8 | `caus` | Source/destination security type | Low to Medium |
| 9 | `cash` | Source/destination symbol | Low to Medium |
| 10 | `10000` | Cash amount / net amount / trade amount | Unknown |

This row must not be treated as a complete Axys/APX import field dictionary.

### 13.5 Uppercase Cancellation Example

Research records the following cancellation pattern:

```text
acct123,010101,010101,by,csus,appl,100,caus,cash,10000
```

becomes:

```text
acct123,010101,010101,BY,csus,appl,100,caus,cash,10000
```

| Interpretation | Confidence |
|---|---|
| Third-party Axys/APX workflows use uppercase transaction codes to represent reversal/deletion/cancellation. | Medium Confidence |
| Field mismatch between reversal and original can cause a Trade Blotter error in APX integration workflow. | Medium Confidence |
| Whether uppercase cancellation is universally valid across all Axys/APX versions, transaction types, import methods, and native workflows is Unknown. | Unknown |

### 13.6 APX Report Labels for Transaction Summary

Observed APX Transaction Summary report labels include:

| Section | Labels |
|---|---|
| Dividends | `Ex-Date`, `Pay-Date`, `Symbol`, `Security`, `Amount` |
| Contributions | `Trade Date`, `Settle Date`, `Quantity`, `Symbol`, `Security`, `Unit Price`, `Amount` |
| Withdrawals | `Trade Date`, `Settle Date`, `Quantity`, `Symbol`, `Security`, `Unit Price`, `Amount` |

These are report labels, not verified APX database field names.

### 13.7 APX Attribution / Contribution Labels

Observed APX report-guide labels include:

```text
Portfolio Return
Benchmark Return
Active Return
Allocation Effect
Selection Effect
Total Effect
Industry Sector
Security
Avg Wgt
Return
Contrib
Portfolio
Benchmark
Difference
Top Contributors
Bottom Contributors
Top Attribution Effects
Bottom Attribution Effects
Largest Weights
```

These labels are valid report-output evidence. They do not establish formula definitions, stored-performance fields, APX database column names, or IMEX field names.

---

## 14. Known Issues and Quirks

| Issue / Quirk | Axys | APX | Confidence | Data Dictionary Implication |
|---|---:|---:|---|---|
| Field labels may be report/export labels rather than native fields. | Yes | Yes | High Confidence | Do not map report labels directly to files/tables without evidence. |
| Security identity commonly requires both symbol and security type in integration workflows. | Yes | Yes | Verified in CI context | Dictionary entries should preserve both fields. |
| Duplicate security matches can occur when ticker and CUSIP entries coexist or same symbol appears with different security types. | Yes | Yes | Verified in CI context | Security identifier mapping must include ambiguity handling. |
| Security translations can be account-specific. | Yes | Yes | Verified in CI context | A global security mapping may not always be valid. |
| Security type / asset class changes can affect historical performance and may require regeneration. | Context not fully separated | Context not fully separated | Verified in research context | Classification/security fields can affect downstream reports. |
| Security master import can fail if referenced industry group/sector records do not exist. | Context not fully separated | Context not fully separated | Verified in research context | Classification lookup dependencies matter. |
| Direct Axys file access is risky because file formats can change between versions. | Yes | No | Medium Confidence | Prefer IMEX/REP/report outputs when possible. |
| Axys v3.7 to 3.8 reportedly required file conversion and some format changes. | Yes | No | Medium Confidence | Version-tag all file-layout observations. |
| APX public views exist but are limited and may not expose all desired data. | No | Yes | Medium Confidence | SQL/public-view field dictionaries require environment validation. |
| REP/Report Writer files may contain checksum behavior; manual changes can break future Report Writer editing. | Yes | Yes | Medium Confidence | REP-derived field dictionaries should record source artifact and customization status. |
| REP-based extraction can be report-layout-sensitive. | Yes | Yes | Medium Confidence | Stable extract reports should be version-controlled. |
| APX extraction may still require installed Advent client tools such as `REP32.exe` in some integrations. | Yes | Yes | Verified for connector | Data dictionaries should record extraction mechanism. |
| APX reporting includes SSRS-based reports in public guide material. | No | Yes | Verified for APX guide | APX report labels may come from SSRS/RDL, not REP. |
| Cash sweeps and intra-account journals may be removed by third-party integration tooling. | Yes | Yes | High Confidence for AIA workflows | Raw source transactions and posted accounting records may differ. |
| `li` / `lo` interpretation can depend on settings in Axys conversion evidence. | Yes | Unknown | Medium Confidence | Transaction code alone is not enough. |
| `epus` and `exus` terminology conflicts across sources. | Yes | Yes | Medium Confidence | Do not classify them definitively without vendor docs. |
| Price source and price set may matter, especially in APX AIA pricing workflows. | Unknown | Yes | Verified for AIA workflow | Price fields should include source/set when available. |
| APX ACA-generated transactions post to APX Trade Blotter, but final accounting post lifecycle is Unknown. | No | Yes | Verified workflow; lifecycle Unknown | Corporate-action fields need blotter status/source fields if available. |

---

## 15. Version Differences Recorded in Supplied Research

| Area | Axys | APX | Confidence |
|---|---|---|---|
| Axys v1.x file format | Reported open text file structure. | N/A | Medium Confidence |
| Axys v2.x file format | Reported binary file format. | N/A | Medium Confidence |
| Axys v3.x IMEX | IMEX allowed CSV, tab, and fixed import/export according to practitioner source. | N/A | High / Medium Confidence depending detail |
| Axys 3.7 to 3.8 | Reported file conversion and some file-format changes. | N/A | Medium Confidence |
| Axys 3.8.6 | Minimum supported version for one Data Broker connector. | N/A | Verified for connector only |
| Axys 3.8.7 | Enhanced Position Reconciliation report, expanded generic date framework, and additional/improved multicurrency reports in vendor blog research. | N/A | Verified |
| APX v1.x to v4.x IMEX | IMEX functionality maintained; fixed-format generation reportedly eliminated. | Yes | Medium Confidence |
| APX 3.0 reporting | SSRS reporting framework introduced according to release coverage. | Yes | High Confidence |
| APX 15.2 / 16.1 / 16.2 / 17.1 | Supported/tested APX versions for one Data Broker connector. | Yes | Verified for connector only |
| Recent APX REST API | Practitioner source says recent APX versions have REST API option. | Yes | Medium Confidence |
| APX / Genesis | 2024 industry source states APX and accounting engine are part of SS&C Advent Genesis. | Yes | High Confidence for product direction, not schema |

---

## 16. Required Field Sets for Common Use Cases

These are recommended minimum **conceptual** field families for downstream use. They are not verified Axys/APX export schemas.

### 16.1 Security Master Extract

| Conceptual Field | Purpose | Native Field Known? | Confidence |
|---|---|---:|---|
| Security symbol | Security identity | Observed as Symbol/Axys Symbol/APX Symbol | Verified in integration contexts |
| Security type | Disambiguates symbol and instrument behavior | Observed as Type/Security Type | Verified in integration contexts |
| Security name / description | Display and matching | Observed as Security/Name/WP Name | Verified in report/integration contexts |
| CUSIP | Matching identifier | Observed in integration contexts | Verified in integration contexts |
| Ticker | Matching identifier | Observed in integration contexts | Verified in integration contexts |
| Asset class | Classification | Observed in Axys export | Verified for export context |
| Sector / industry / country / region | Classification | Report category supported; exact field unknown | Mixed |
| Currency | Pricing/accounting currency | Conceptual; exact field unknown | Unknown |
| Fixed-income terms | Accrual, coupon, maturity, factor, etc. | Not supplied | Unknown |

### 16.2 Transaction Extract

| Conceptual Field | Purpose | Native Field Known? | Confidence |
|---|---|---:|---|
| Portfolio/account identifier | Owner of transaction | Observed labels/tokens | High Confidence |
| Trade date | Economic date | APX report label; conceptual Axys | High Confidence as concept |
| Settlement date | Cash settlement date | APX report label; conceptual Axys | High Confidence as concept |
| Transaction code | Accounting event type | Observed codes | Medium Confidence; complete matrix Unknown |
| Security symbol/type | Security reference | Observed in example rows | Medium Confidence |
| Quantity | Holding impact | Observed | High Confidence |
| Price / unit price | Execution price | Observed as report label | Medium Confidence |
| Amount | Cash/economic amount | Observed in APX report labels | Medium Confidence |
| Source/destination type/symbol | Cash/security movement context | Observed in integration examples | Medium Confidence |
| Broker / broker representative | Operational metadata | Observed in AIA workflow | Medium Confidence |
| Comment | Operational note | Observed in AIA workflow | Medium Confidence |

### 16.3 Holdings / Position Extract

| Conceptual Field | Purpose | Native Field Known? | Confidence |
|---|---|---:|---|
| Portfolio/account identifier | Owner of holding | Observed | High Confidence |
| Security identifier | Security held | Observed | High Confidence |
| Quantity | Position units | Observed | Verified as report/export label |
| Price | Valuation price | Observed | Verified as report label |
| Market value | Valuation amount | Observed | Verified as report/export label |
| Percent / weight | Allocation | Observed as `Pct Assets` / Percent of Portfolio / Avg Wgt | Verified as report labels |
| Cost / cost basis | Tax/accounting basis | Observed in APX report labels | Verified as report labels |
| Unrealized gain/loss | Valuation/cost difference | Observed in APX report description | Verified as report label |
| Yield | Income/valuation field | Observed | Verified as report label |
| Lot fields | Tax-lot detail | APX report labels include Open/Close Date; full lot schema Unknown | Mixed |

### 16.4 Pricing Extract

| Conceptual Field | Purpose | Native Field Known? | Confidence |
|---|---|---:|---|
| Security symbol/type | Price key | Likely from integration context; exact `.pri` layout Unknown | Medium Confidence |
| Price date | Price key | Unknown | Unknown |
| Price value | Valuation | Unknown native field | Unknown |
| Price source / `SourceId` | Provenance/source selection | `SourceId` observed in APX AIA context | Verified only in workflow context |
| Price set | APX price grouping | AIA Price Set Logic observed | Verified for AIA workflow |
| Currency | Price currency | Unknown | Unknown |
| Calculated price flag/source | Missing-price fallback | CI release notes mention calculated price | Verified for CI Axys |

### 16.5 Performance Extract

| Conceptual Field | Purpose | Native Field Known? | Confidence |
|---|---|---:|---|
| Portfolio/account identifier | Performance subject | Unknown native field | Unknown |
| Start/end dates | Performance period | Unknown native field | Unknown |
| Portfolio return | Return result | APX report label observed | Verified as APX report label |
| Benchmark return | Benchmark result | APX report label observed | Verified as APX report label |
| Active return | Relative return | APX report label observed | Verified as APX report label |
| Weight / average weight | Performance weighting | APX `Avg Wgt` report label observed | Verified as report label |
| Security return | Security-level return | Unknown native field | Unknown |
| Contribution | Contribution result | APX `Contrib` report label observed | Verified as report label |
| Classification | Segment/group | APX Industry Sector observed | Verified as report label |
| Attribution effects | Allocation, selection, total effects | APX labels observed | Verified as report labels |
| Gross/net flag | Fee basis | Unknown | Unknown |
| Currency/local/base fields | Multi-currency performance | Unknown | Unknown |

---

## 17. Unknowns

The following gaps should remain Unknown until supported by vendor documentation, sample exports, REP/SSRS source, report outputs, or production observations.

### 17.1 Axys Unknowns

| Unknown | Needed Evidence |
|---|---|
| Exact native Axys security-master file layout | Sanitized `sec.inf`, `type.inf`, vendor field docs |
| Exact Axys transaction file layout | Sanitized `.cli`, `topost.trn`, IMEX transaction import docs |
| Complete official Axys transaction code matrix | Vendor transaction documentation or production code dictionary |
| Exact Axys holdings storage model | Reports, files, or vendor docs showing stored vs calculated holdings |
| Exact Axys cash balance storage/report model | Cash reports, IMEX exports, or file documentation |
| Exact Axys `.pri` file layout | Sanitized price files or vendor pricing docs |
| Exact Axys `split.inf` layout | Sanitized split file or vendor docs |
| Axys performance storage files/fields | IMEX exports, performance reports, vendor docs |
| Axys classification storage model | Security master/classification exports and report tests |
| Axys REP field dictionary / RepLang grammar | RepLang Programmer’s Guide or source dictionaries |
| Whether Axys reports use stored or recalculated values for each report | REP source and controlled report tests |

### 17.2 APX Unknowns

| Unknown | Needed Evidence |
|---|---|
| APX native SQL tables/views for security master | APX schema/public-view docs or sanitized SQL extracts |
| APX security-master field dictionary | APX docs, IMEX exports, public views |
| APX transaction storage tables/fields | APX schema, Trade Blotter exports, vendor docs |
| Complete official APX transaction code matrix | Vendor transaction/import documentation |
| APX IMEX object list and field dictionary | APX IMEX docs or export samples |
| APX price table/source/price set schema | APX schema, price imports, AIA archive examples |
| APX corporate action / ACA generated transaction fields | ACA downloads, Reorg Utility docs, Trade Blotter sample |
| APX performance storage/calculation model | APX performance docs, SQL views/procedures, report source |
| APX classification tables and effective dating | Schema/docs and controlled historical report tests |
| APX SSRS report RDL/dataset/stored procedure names | SSRS catalog/RDL files |
| Relationship between APX REP/Replang reports and SSRS reports | APX reporting architecture documentation |

### 17.3 IMEX Unknowns

| Unknown | Needed Evidence |
|---|---|
| Axys IMEX object names | Axys IMEX manual, export setup screens, sample exports |
| APX IMEX object names | APX IMEX manual, export setup screens, sample exports |
| Field order and delimiters by object | Sample exports/imports |
| Required vs optional fields | Vendor import specifications and validation logs |
| Error and log file schemas | IMEX logs from controlled imports |
| Version differences across Axys/APX releases | Versioned docs and sample exports |
| Whether APX fixed-format output is unavailable in all relevant APX versions | Versioned APX IMEX documentation |

### 17.4 REP / Reports Unknowns

| Unknown | Needed Evidence |
|---|---|
| Complete Axys report catalog | Installed report menu/export, REP folder inventory, vendor docs |
| Complete APX report catalog | APX Reports Guide, SSRS catalog |
| Exact report source file for each standard report | `.REP`, `.RPW`, RDL, report definitions |
| Exact report parameters | Screenshots, macros, report metadata, source files |
| Report output fields and calculation definitions | Sample outputs and source code |
| Stored vs recalculated behavior by report | Controlled tests and vendor docs |
| Report-to-IMEX reconciliation mappings | Side-by-side exports/reports |

---

## 18. References

This chapter was prepared from the supplied repository blueprint and supplied research files only.

| Source Material | Use |
|---|---|
| `AXYS_APX_REFERENCE_BLUEPRINT(44).md` | Governing editorial standard, confidence labels, chapter structure, field dictionary format |
| `Research_02_Axys_Architecture(4).md` | Axys architecture, files, IMEX, REP, version/file-format cautions |
| `Research_03_APX_Architecture(3).md` | APX architecture, SSRS reporting, APX IMEX, REP32, APX reports |
| `Research_04_Security_Master(18).md` | Security master fields/tokens, `sec.inf`, `type.inf`, security matching, symbol/type behavior |
| `Research_05_Transactions(17).md` | Transaction fields, transaction codes, Trade Blotter, source/destination fields, cancellation behavior |
| `Research_06_Holdings(9).md` | Portfolio Appraisal fields, holdings/positions, position/lot blotters, holdings extract reports |
| `Research_07_Cash(5).md` | Cash tokens, cash sweeps, cash-like source/destination symbols, cash transaction behavior |
| `Research_08_Pricing(9).md` | Price files, price logs, price source/sets, calculated prices, APX price-file update logic |
| `Research_09_Corporate_Actions(7).md` | `split.inf`, ACA for APX, Reorg Utility, corporate-action field unknowns |
| `Research_10_Performance(5).md` | Performance field candidates, product-level performance capability, performance Unknowns |
| `Research_11_Classifications(6).md` | Classification fields/categories, asset class/sector/industry group, custom classification |
| `Research_12_IMEX(11).md` | IMEX utilities, logs, file/folder labels, Data Broker/REP32 extraction, native object unknowns |
| `Research_13_REP(10).md` | REP/Replang tokens, report files, Report Writer Pro, REP32, SSRS distinction |
| `Research_14_Reports(2).md` | APX report names and labels, report categories, report/IMEX distinction |
| `Research_15_Data_Dictionary(1).md` | Data dictionary design, source precedence, unknown tracking |

---

## 19. Chapter Maintenance Rules

1. Do not add a field as a native Axys/APX field unless the source proves it is native.
2. Mark report labels as report labels.
3. Mark integration fields as integration fields.
4. Preserve file names and tokens exactly as observed, including case where known.
5. Do not normalize transaction codes or security type codes into uppercase/lowercase unless the source supports that transformation.
6. Do not treat APX report labels as SQL field names.
7. Do not treat Axys report labels as native file fields.
8. When a field appears in multiple contexts, document each context separately.
9. Version-tag fields and layouts whenever the evidence is version-specific.
10. If a field’s meaning is uncertain, keep the meaning Unknown rather than inferring from portfolio-accounting convention.
