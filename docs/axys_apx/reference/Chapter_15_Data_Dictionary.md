# Chapter 15 — Data Dictionary

**Repository:** AXYS / APX Reference Repository
**Chapter:** `Chapter_15_Data_Dictionary.md`
**Governing specification:** `axys_apx_reference_blueprint.md`, Version 2.0
**Prepared:** 2026-06-29
**Status:** Technical reference chapter based on repository research and cited public evidence
**Public evidence reviewed:** 2026-07-17

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

## 2. Conventions and Confidence

| Confidence | Meaning in this chapter |
|---|---|
| Verified | Directly supported by supplied source material, supplied research, cited vendor/public material inside the research, or observed examples recorded in the research. |
| High Confidence | Strongly supported by the supplied research, but not proven as a complete vendor field definition. |
| Medium Confidence | Plausible and supported by partial, third-party, conversion, consultant, or workflow-specific evidence. |
| Unknown | Not established from the supplied material. Do not implement as fact without additional evidence. |

The repository standard requires that unsupported behavior be marked **Unknown** rather than invented.

---

### 2.1 Source Precedence

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

### 2.2 Entry Format

#### Standard Format

Most dictionary tables use this base format:

| Column | Meaning |
|---|---|
| Field | Literal observed field, label, file, executable, REP expression, code, or token. |
| Description | Meaning supported by the supplied research. |
| Axys | Whether the supplied research supports Axys use/exposure. |
| APX | Whether the supplied research supports APX use/exposure. |
| IMEX | Whether the supplied research supports IMEX/import-export use. |
| REP | Whether the supplied research supports REP/report use. |
| Confidence | Evidence strength for the entry in the stated context. |

#### Expanded Format

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

## 3. Axys Literal Entries

### Axys Files, Folders, and Utilities

| Field / Token | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---|
| `imex32.exe` | Axys Import/Export utility executable referenced by ByAllAccounts Custodial Integrator research. | Yes | No | Yes | No | Verified in CI context |
| `pospos32.exe` | Axys Post Positions utility referenced by security-master research addendum. | Yes | No | Utility-adjacent | No | Verified in CI context |
| `topost.trn` | Axys Trade Blotter file receiving transaction imports in CI workflow. | Yes | No | Yes, transaction import workflow | Unknown | Verified in CI context |
| `*.cli` | Axys client/portfolio files referenced in conversion and integration research. | Yes | APX `.cli` appears in integration notes; native status Unknown | Related | Unknown | Verified for Axys conversion/integration context |
| `.cli` | Advent/Axys client file; Morningstar conversion evidence says cost-basis data may be present if exported. | Yes | Unknown | Related | Unknown | Medium Confidence for field content |
| `sec.inf` | Securities/security information file in Axys conversion and integration evidence. | Yes | APX CI context also references `sec.inf` | Related | Unknown | Verified in conversion/integration context |
| `SECURITY.INF` | Uppercase security-master file-name form referenced as a migration/conversion lead. | Yes | Unknown | Unknown | Unknown | Medium Confidence |
| `type.inf` | Security type information file in Axys conversion and integration evidence. | Yes | APX CI context also references `type.inf` | Related | Unknown | Verified in conversion/integration context |
| `TYPE.INF` | Uppercase security-type file-name form referenced as a migration/conversion lead. | Yes | Unknown | Unknown | Unknown | Medium Confidence |
| `split.inf` / `SPLIT.INF` | Securities splits file in Axys conversion/corporate-action research. | Yes | Unknown native APX status | Unknown | Unknown | High Confidence for Axys |
| `*.pri` | Security price files in Axys price folder/integration evidence. | Yes | APX AIA examples use `.pri` file names in price workflow | Related | No | Verified for integration context |
| `.PRF` / `.PBF` | Axys performance-history artifacts in dated operational guidance. | Yes | Unknown | Unknown | Related to performance reports | High Confidence; exact layouts Unknown |
| `.GRP` | Axys portfolio-group artifact associated with reporting/performance history. | Yes | Unknown | Unknown | Related | High Confidence operational evidence |
| `.CPG` | Axys composite artifact with member entry/exit dates in dated guidance. | Yes | Unknown | Unknown | Related | High Confidence operational evidence |
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

### Axys REP / Replang Tokens

| Field / Token | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---|
| `AMAN.REP` | Assets Under Management report file in Axys consultant example. | Yes | Unknown | No | Yes | Verified for example |
| `PERHSUM.REP` | Performance History for Selected Time Periods in historical Axys 3.6 guidance. | Yes | Unknown | No | Yes | Verified historical evidence |
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

### Axys Report / Export Labels

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

## 4. APX Literal Entries

### APX Utilities, Import/Export, and Blotter Names

| Field / Token | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---|
| `APXIX.exe` / `apxix.exe` / `ApxIx` | Capitalization/label variants for the APX Import/Export utility in independent guides. | No | Yes | Yes | No | High Confidence; installed filename/version site-specific |
| `Advent IMEX Log` | APX AIA log showing last importing done through IMEX tool. | No | Yes | Yes | No | Verified in AIA context |
| `imexhist.log` | APX IMEX history-log filename in WealthTechs guidance. | No | Yes | Yes | No | Verified for AIA context; location/version Unknown |
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
| `Axys ACA Trade Blotter processing` | ACA-for-Axys workflow where simple/mandatory events can process to the Axys Trade Blotter. | Yes | N/A | Unknown | Unknown | Verified for workflow; fields Unknown |
| `Automation Results` | Email summary after scripted ACA processing. | Yes | Yes | Unknown | No | Verified for ACA workflow; fields Unknown |
| `$pathCDI` | Custom label mapped to a network path for AIA holdings extract workflow. | Yes | Yes | Workflow setting | Report output destination | Verified for AIA workflow |
| `Holdings Extract Folder (h.CDI)` | AIA setting mapped to holdings extract output folder. | Yes | Yes | Workflow setting | Report output destination | Verified for AIA workflow |
| `cdirecon` | Typical holdings extract group in AIA workflow. | Unknown | Yes | Workflow setting | REP/report | Verified as typical, not mandatory |

### APX Security and Account Fields / Labels

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

### APX Report Labels

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
| `Proceeds` | Transaction Summary / Realized Gains/Losses report label. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Gain/Loss` | Transaction Summary / Realized Gains/Losses report label. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Cost Basis` | Realized Gains/Losses report label. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Open Date` | Realized Gains/Losses lot label. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Close Date` | Realized Gains/Losses lot label. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Short Term` | Realized gain/loss classification label. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Long Term` | Realized gain/loss classification label. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Percent of Portfolio` | Portfolio Appraisal report description label. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Yield` | Portfolio Appraisal report description label. | Unknown | Yes | Unknown | APX reports | Verified as report label |
| `Unrealized Gain and Loss` | Portfolio Appraisal report description label. | Unknown | Yes | Unknown | APX reports | Verified as report label |

### APX Performance / Attribution / Contribution Labels

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

## 5. Shared Literal Families

### Security Identifier Fields and Tokens

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

### Security Type Codes and Prefixes

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

### Portfolio / Account Identifier Fields

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

### Date Fields and Labels

| Field / Label | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---|
| `Trade Date` | Transaction execution/economic date label in APX report and transaction concepts. | Expected / observed in examples | Yes | Unknown | APX reports | Verified as APX report label; Axys conceptual High Confidence |
| `Settle Date` | Transaction settlement date label in APX Transaction Summary. | Expected / observed in examples | Yes | Unknown | APX reports | Verified as APX report label; Axys conceptual High Confidence |
| `Ex-Date` | Dividend report section label in APX Transaction Summary sample. | Unknown | Yes | Unknown | APX report | Verified as report label |
| `Pay-Date` | Dividend report section label in APX Transaction Summary sample. | Unknown | Yes | Unknown | APX report | Verified as report label |
| `Open Date` | Realized Gains/Losses lot label. | Unknown | Yes | Unknown | APX report | Verified as report label |
| `Close Date` | Realized Gains/Losses lot label. | Unknown | Yes | Unknown | APX report | Verified as report label |
| Price date | Date for which a security price applies. Exact field name not supplied. | Conceptual | Conceptual | Unknown | Unknown | High Confidence as concept; native field Unknown |
| Valuation / report as-of date | Report date used for holdings/valuation reports. Exact field name not supplied. | Conceptual | Conceptual | Unknown | Report parameter likely | High Confidence as concept; native field Unknown |
| Performance start/end date | Period dates for performance reports. Exact field names not supplied. | Conceptual | APX visible behavior | Unknown | Reports | Medium Confidence; field names Unknown |
| Effective date | Corporate action/classification/security-master effective date concept. Exact field name not supplied. | Unknown | Unknown | Unknown | Unknown | Unknown |

### Transaction Fields and Codes

#### Transaction Field Labels / Parameters

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

#### Observed Transaction Codes

This table is an observed-code catalog only. It is not a complete official Axys/APX transaction-code matrix.

| Code | Observed Meaning / Context | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---|
| `by` | Buy. | Observed | Observed | Trade Blotter/import context | Unknown | Medium Confidence |
| `BY` | Cancellation instruction derived from `by` in a reviewed Trade Blotter staging/control workflow. | Observed | Observed | Trade Blotter control context | Posted-export availability Unknown | Medium Confidence |
| `sl` | Sell. | Likely observed in examples; native status Unknown | Observed | Trade Blotter/import context | Unknown | Medium Confidence |
| `SL` | Uppercase sell-derived instruction appears in AIA cancellation Trade Blotter examples. | Unknown | Observed | Trade Blotter control context | Posted-export availability Unknown | Medium Confidence |
| `ss` | Short sale; APX integration evidence maps `SELL / SHORT` to exact lowercase `ss`. | Unknown native field layout | Observed | Trade Blotter/import context | Unknown | Medium-High for code meaning; Unknown native mechanics |
| `SS` | Uppercase short-sale-derived instruction appears in AIA cancellation Trade Blotter examples. | Unknown | Observed | Trade Blotter control context | Posted-export availability Unknown | Medium Confidence |
| `cs` | Cover short; APX integration evidence maps `BUY / COVER SHORT` to exact lowercase `cs`. | Unknown native field layout | Observed | Trade Blotter/import context | Unknown | Medium-High for code meaning; Unknown native mechanics |
| `CS` | Uppercase cover-short-derived instruction appears in AIA cancellation Trade Blotter examples. | Unknown | Observed | Trade Blotter control context | Posted-export availability Unknown | Medium Confidence |
| `li` | Deliver in / transfer in / credit / deposit / positive direction depending context. | Observed | Observed | Trade Blotter/import context | Unknown | Medium Confidence |
| `lo` | Deliver out / transfer out / debit / withdrawal / negative direction depending context. | Observed | Observed | Trade Blotter/import context | Unknown | Medium Confidence |
| `dv` | Dividend / income / reinvestment leg. | Unknown | Observed | Trade Blotter/import context | Unknown | Medium Confidence |
| `in` | Income / interest. | Unknown | Observed | Trade Blotter/import context | Unknown | Medium Confidence |
| `rc` | Return of capital; public translation evidence maps it to portfolio-cash context. | Unknown native field layout | Observed | Trade Blotter/import context | Unknown | Medium-High Confidence for integration mapping |
| `pd` | Principal paydown / bond-security return-of-capital case; public translation evidence maps it to portfolio-cash context. Treat as principal-return context, not client external flow, when MBS/ABS/amortizing-security paydown evidence is present. | Unknown native field layout | Observed | Trade Blotter/import context | Unknown | Medium-High Confidence for integration mapping |
| `ai` | Negative interest or margin interest depending context. | Unknown | Observed | Trade Blotter/import context | Unknown | Medium Confidence |
| `sa` | Sale accrued interest / sell-side accrued interest. | Unknown | Observed | Trade Blotter/import context | Unknown | Medium Confidence |
| `pa` | Purchase accrued interest / buy-side accrued interest. | Unknown | Observed | Trade Blotter/import context | Unknown | Medium Confidence |
| `dp` | Debit / fee-related / tax / service charge / cash-security case. | Observed | Observed | Trade Blotter/import context | Unknown | Medium Confidence |
| `wd` | Cash-security sell / withdrawal-like cash-security movement. | Observed | Observed | Trade Blotter/import context | Unknown | Medium Confidence |
| `;` | Journal / comment / other / split in integration table. | Unknown | Observed | Trade Blotter/import context | Unknown | Medium Confidence |
| `ti` / `to` | Opposite transaction pair examples for intra-account cash journal removal. | Observed in AIA context | Observed in AIA context | Integration workflow | Unknown | High Confidence as observed tokens |
| `si` / `so` | Opposite transaction pair examples for intra-account cash journal removal. | Observed in AIA context | Observed in AIA context | Integration workflow | Unknown | High Confidence as observed tokens |
| `tr` / `ts` | Opposite transaction pair examples for intra-account cash journal removal. | Observed in AIA context | Observed in AIA context | Integration workflow | Unknown | High Confidence as observed tokens |

### Cash Tokens and Cash-Like Symbols

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
| `caus margin` | Margin/cash context used in public negative-interest or margin-interest mappings. | Yes | Unknown | Related | Unknown | High Confidence as observed token; native definition Unknown |
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

### Holdings / Position Fields

| Field / Label | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---|
| `Quantity` | Holding/position quantity. | Yes | Yes as report label | Unknown | Yes | Verified in report/export contexts |
| `Security` | Security name/description in holding/report output. | Yes | Yes | Unknown | Yes | Verified in report/export contexts |
| `Price` | Holding price. | Yes | Yes as report label | Unknown | Yes | Verified in report contexts |
| `Market Value` | Holding/account valuation amount; fixed-income appraisal evidence strongly implies clean market value separate from accrued interest. | Yes | Yes | Unknown | Yes | Verified in report contexts; accrued exclusion High / strongly implied |
| `Accrued Interest` | Fixed-income interest earned but not paid; appraisal evidence shows it separately from Market Value. | Yes as report concept | Yes as report concept | Unknown | Reports | High as separate report concept; native field Unknown |
| `Pct Assets` | Percent-of-assets label in Axys Portfolio Appraisal sample. | Yes | Unknown | Unknown | Yes | Verified for Axys sample |
| `Percent of Portfolio` | Percent-of-portfolio label in APX Portfolio Appraisal description. | Unknown | Yes | Unknown | APX report | Verified as APX report label |
| `Yield` | Yield field in Portfolio Appraisal examples/description. | Yes | Yes | Unknown | Reports | Verified as report label |
| `Portfolio Code` | Owner portfolio code in Axys Portfolio Appraisal custom report. | Yes | Unknown | Unknown | Yes | Verified |
| `Position Blotter name` | APX object receiving imported positions for reconciliation. | Unknown | Yes | Yes | No | Verified in APX CI context |
| `Lot Blotter name` | APX object receiving lots when lots enabled. | Unknown | Yes | Yes | No | Verified in APX CI context |
| `Trade Blotter name` | APX object receiving imported transactions. | Unknown | Yes | Yes | No | Verified in APX CI context |
| `Name` | Missing-prices output field for name of security/position with no price. | Yes | Unknown | Related | No | Verified in CI context |

### Pricing Fields and Artifacts

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

### Corporate Action Fields and Artifacts

| Field / Token | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---|
| `split.inf` / `SPLIT.INF` | Axys securities splits file. | Yes | Unknown | Unknown | Unknown | High Confidence for Axys |
| `.veh` | AIA vehicle file transformed to `sec.inf` layout in Axys/APX security import workflows. | Yes in AIA context | Yes in AIA context | Import staging | Unknown | Verified in AIA context |
| Security identifier | Identifier tying corporate action to security. Exact field name not supplied. | Required conceptually | Required conceptually | Unknown | Unknown | Unknown as field |
| Split date / effective date | Date on which split applies. AdventGuru consultant merge code uses logical `SplitDate` after loading exported split files. | Likely in `split.inf`; exact official header Unknown | Unknown | Unknown | Unknown | Medium to High Confidence as logical field |
| Split security / symbol | Security identifier affected by a split. AdventGuru consultant merge code uses logical `SplitSymbol`. | Likely in `split.inf`; exact official header Unknown | Unknown | Unknown | Unknown | Medium to High Confidence as logical field |
| Split ratio / factor | Split ratio or factor. AdventGuru consultant merge code uses logical `SplitFactor`. | Likely in `split.inf`; exact official header/factor convention Unknown | Unknown | Unknown | Unknown | Medium to High Confidence as logical field |
| Old security identifier | Security being reorganized. Exact field name not supplied. | Unknown | Unknown | Unknown | Unknown | Unknown |
| New security identifier | Replacement/distributed security. Exact field name not supplied. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Cash-in-lieu amount | Cash from fractional shares. Exact field name not supplied. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Announcement date | Corporate action announcement date. Exact field name not supplied. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Record date | Holder-of-record date. Exact field name not supplied. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Ex-date | Market entitlement date; APX report label observed for dividends. | Unknown | Yes as report label | Unknown | Report | Verified as APX report label only |
| Pay date / Pay-Date | Payment date; APX report label observed. | Unknown | Yes as report label | Unknown | Report | Verified as APX report label only |

### Classification Fields

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

### Performance Fields

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
| Beginning market value | Performance input concept. For fixed income, may need accrued interest added when reconstructing dirty-value returns. Exact field name not supplied. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Ending market value | Performance input concept. For fixed income, may need accrued interest added when reconstructing dirty-value returns. Exact field name not supplied. | Unknown | Unknown | Unknown | Unknown | Unknown |
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

## 6. Alias and Synonym Map

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

## 7. Interpretation and Ownership Boundaries

This chapter indexes observed literal names and source-specific labels. It does
not define the native schema behind them.

- IMEX/APXIX object discovery, utilities, paths, and logs are owned by
  [Chapter 12](Chapter_12_Imex.md).
- REP and RepLang behavior is owned by
  [Chapter 13](Chapter_13_Rep.md).
- Report names and report families are owned by
  [Chapter 14](Chapter_14_Reports.md).
- Domain meaning and unresolved behavior remain in Chapters 04-11 and 17.
- Report labels are not native fields unless separate evidence establishes the
  mapping.
- Candidate normalized fields are implementation design aids, not proof of
  native Axys/APX names.
- The current evidence does not provide complete native field dictionaries,
  authoritative IMEX/APXIX object catalogs, REP grammars, APX schemas, or API
  contracts.

Topic-specific Unknowns are maintained in the owning subject chapter rather
than repeated here.

### Case-Handling Policy

Observed casing is part of a literal value unless a separate domain or site
contract authorizes normalization. The examples in this dictionary do not
establish a general lowercase or uppercase convention.

| Field family | Default handling | Evidence boundary |
|---|---|---|
| Security/vehicle symbol and ticker-like product identifier | Preserve raw case; exact-case match by default. | APX AIA explicitly identifies vehicle symbols as case-sensitive; broader native rules remain Unknown. |
| Security type and special-security type/symbol | Preserve raw case; do not copy casing from an example into a global rule. | Axys/APX CI material contains mixed-case examples. |
| Portfolio/account code | Preserve raw case; exact-case match by default. | APX AIA explicitly identifies account code as case-sensitive for the cited workflow. |
| Asset-class, sector, industry, and other classification codes | Preserve raw case pending a site-specific dictionary. | No complete native casing contract was recovered. |
| Transaction code | Preserve raw case. Any case-insensitive rule evaluation must be explicit and must not merge posted rows with Trade Blotter cancellation controls. | AIA evaluator behavior and cancellation staging are workflow-specific. |
| Descriptive text and report labels | Preserve the extracted value; presentation casing is not identifier semantics. | Report labels are not native-field contracts. |
| ISO currency code | Normalize only under the separate currency-domain contract. | Independent standardized domain; not an Axys/APX identifier rule. |

If a site authorizes case-insensitive comparison, retain the raw value and use a
separate comparison key. Case-only differences must remain available for audit
and reconciliation.

---

## 8. References

This chapter is based on the supplied repository blueprint and supplied research files only.

| Source Material | Use |
|---|---|
| `../axys_apx_reference_blueprint.md` | Governing editorial standard, confidence labels, chapter structure, field dictionary format |
| `../evidence/Research_02_Axys_Architecture.md` | Axys architecture, files, IMEX, REP, version/file-format cautions |
| `../evidence/Research_03_APX_Architecture.md` | APX architecture, SSRS reporting, APX IMEX, REP32, APX reports |
| `../evidence/Research_04_Security_Master.md` | Security master fields/tokens, `sec.inf`, `type.inf`, security matching, symbol/type behavior |
| `../evidence/Research_05_Transactions.md` | Transaction fields, transaction codes, Trade Blotter, source/destination fields, cancellation behavior |
| `../evidence/Research_06_Holdings.md` | Portfolio Appraisal fields, holdings/positions, position/lot blotters, holdings extract reports |
| `../evidence/Research_07_Cash.md` | Cash tokens, cash sweeps, cash-like source/destination symbols, cash transaction behavior |
| `../evidence/Research_08_Pricing.md` | Price files, price logs, price source/sets, calculated prices, APX price-file update logic |
| `../evidence/Research_09_Corporate_Actions.md` | `split.inf`, ACA for Axys/APX, Reorg Utility, corporate-action field unknowns |
| `../evidence/Research_10_Performance.md` | Performance claims, literal-artifact evidence, and native-field Unknowns |
| `../evidence/Research_11_Classifications.md` | Classification claims, observed labels, and native-field Unknowns |
| `../evidence/Research_12_IMEX.md` | IMEX/APXIX utilities, logs, paths, artifacts, and native-object Unknowns |
| `../evidence/Research_13_REP.md` | REP/Replang tokens, report files, Report Writer Pro, REP32, SSRS distinction |
| `../evidence/Research_14_Reports.md` | APX report names and labels, report categories, report/IMEX distinction |
| `../evidence/Research_15_Data_Dictionary.md` | Evidence ownership boundary for this derivative index |

---

## 9. Maintenance Rules

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
11. Never use a case-insensitive integration rule as proof that native product identifiers are case-insensitive.
