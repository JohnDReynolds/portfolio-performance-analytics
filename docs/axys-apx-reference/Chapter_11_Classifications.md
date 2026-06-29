# 11. Classifications

Repository: AXYS / APX Reference Repository  
Chapter: `docs/11-Classifications.md`  
Prepared from supplied research material: 2026-06-29  
Governing specification: `AXYS_APX_REFERENCE_BLUEPRINT.md`, Version 2.0

---

## 1. Overview

This chapter documents classification-related behavior in SS&C Advent Axys and SS&C Advent Portfolio Exchange (APX), based only on the supplied research and source material.

The term **classification** is used conservatively in this chapter. The supplied research does not establish a single native Axys/APX object named `classification`. Instead, classification-like information appears in several different contexts:

| Context | Examples | Status |
|---|---|---:|
| Security-level classification | Asset class, sector, industry group, country, region, custom classification | Partially supported by report/product evidence; exact storage Unknown |
| Portfolio/account grouping | Manager, asset class, investment objective, firm-defined category | Verified for Axys product capability; APX storage Unknown |
| Report grouping | Holdings or performance grouped by asset class, sector, country, region, industry group, custom classification | Verified at capability/report-snippet level; exact report internals Unknown |
| Security type | Product security type codes such as `csus`, `efus`, `tfus`, `oaus`, `CAUS`, `CSUS` | Verified in integration context; not the same as asset class/sector/industry |
| Integration/export labels | `Asset Class`, `Security Type`, `Sec Type Code`, `Security Symbol` | Verified in specific export/integration contexts; not proven native field names |
| Labels | AdventGuru mentions transaction and label import through Trade Blotter/IMEX context | Relationship to classifications Unknown |

### 1.1 Confidence labels

| Confidence | Meaning |
|---|---|
| Verified | Directly supported by supplied research material derived from vendor, third-party integration, report, or consultant documentation. |
| High Confidence | Strongly supported by the supplied material, but not proven by an official native Axys/APX schema or sample export. |
| Medium Confidence | Plausible and useful, but support is indirect, context-specific, or source-limited. |
| Unknown | Not established by supplied material. Do not implement or document as fact without additional evidence. |

### 1.2 Key rules for this chapter

| Rule | Reason | Confidence |
|---|---|---:|
| Do not treat `Security Type` as synonymous with `Asset Class`, `Sector`, `Industry`, `Country`, `Region`, or `Custom Classification`. | Security Type is used for product security identity and matching; classifications are used in reporting/grouping contexts. | High Confidence |
| Preserve product security symbol and security type when joining classification data. | ByAllAccounts CI evidence shows symbol alone can be ambiguous. | Verified for CI context |
| Do not assume classification values are historical/effective-dated. | Supplied material does not establish historical classification storage. | Unknown |
| Do not assume historical reports use either current or historical classifications. | Supplied material does not establish report lookup timing. | Unknown |
| Do not invent IMEX object names or REP field tokens for classification data. | No supplied source provides official classification IMEX object names or RepLang classification tokens. | Unknown |

---

## 2. Axys

### 2.1 Axys classification-related capabilities

| Statement | Confidence | Notes |
|---|---:|---|
| Axys includes reporting and report customization capabilities. | Verified | SS&C Axys product material in the supplied research describes predefined reports and report customization. |
| Axys supports Axys Report Writer Pro. | Verified | Product material states users can choose predefined reports or create custom reports with Axys Report Writer Pro. |
| Axys can manage and report portfolios grouped by manager, asset class, investment objective, or any category chosen by the firm. | Verified | This is portfolio/account grouping capability, not necessarily security-level classification storage. |
| Axys can display performance by portfolios, asset classes, sectors, countries, or regions. | Verified | This establishes classification-aware reporting capability. |
| Axys classification-like data may be used at both portfolio-group level and security/holding/reporting level. | Medium Confidence | Product material references portfolio grouping and performance display categories, but does not define storage. |
| Axys supports firm-defined grouping categories at least for portfolios. | Verified | Product material states portfolios can be grouped by any category the firm chooses. |
| Whether Axys supports arbitrary firm-defined security-level classification schemes is established. | Unknown | The supplied material does not prove security-level custom classification storage. |

### 2.2 Axys security identity versus classification

Axys classification extraction must keep security identity separate from classification attributes.

| Item | Axys evidence | Classification relevance | Confidence |
|---|---|---|---:|
| Axys Symbol | ByAllAccounts CI uses Axys Symbol in security translation. | Required join key in integration context. | Verified for CI context |
| Axys Security Type / Type | ByAllAccounts CI uses Axys Security Type with symbol. | Required join key in integration context; not the same as asset class. | Verified for CI context |
| `sec.inf` | CI uses Axys Security Information from `sec.inf`; Morningstar conversion research identifies `sec.inf` as Axys securities file. | Possible security-reference source; classification fields inside it are not established. | Verified for file/use; classification fields Unknown |
| `type.inf` | CI uses Axys Security Type Information from `type.inf`; Morningstar identifies `type.inf` as security type file. | Security type metadata source; not a complete asset-class/sector dictionary. | Verified for file/use; classification mapping Unknown |
| Duplicate symbol/type situations | CI evidence describes duplicate/ambiguous matching when same symbol has multiple types or ticker/CUSIP are both used as symbols. | Classification joins based only on ticker/symbol can be wrong. | Verified for CI context |

### 2.3 Axys security type examples

The supplied research contains examples of security type codes in integration documentation. These are examples only, not a complete dictionary.

| Example code | Context | Confidence | Notes |
|---|---|---:|---|
| `csus` | ByAllAccounts CI security matching examples | Verified for example | Do not infer meaning without a type dictionary. |
| `efus` | ByAllAccounts CI security translation example | Verified for example | Example maps ticker `LMNVX` to symbol `524659208` and type `efus`. |
| `tfus` | ByAllAccounts Axys duplicate example | Verified for example | Same CUSIP can appear under different security types. |
| `oaus` | ByAllAccounts Axys duplicate example | Verified for example | Same CUSIP can appear under different security types. |
| `CAUS` | CI security translation file description | Verified for example | Case may vary across examples; do not normalize without evidence. |
| `CSUS` | CI security translation file description | Verified for example | Case may vary across examples; do not normalize without evidence. |

### 2.4 Axys reserved security type prefixes in CI matching

| Prefix | CI behavior | Axys native behavior | Confidence |
|---|---|---|---:|
| `aw` | Excluded from ByAllAccounts CI security matching | Unknown | Verified for CI context only |
| `br` | Excluded from ByAllAccounts CI security matching | Unknown | Verified for CI context only |
| `ex` | Excluded from ByAllAccounts CI security matching | Unknown | Verified for CI context only |
| `ep` | Excluded from ByAllAccounts CI security matching | Unknown | Verified for CI context only |
| `pi` | Excluded from ByAllAccounts CI security matching | Unknown | Verified for CI context only |
| `rs` | Excluded from ByAllAccounts CI security matching | Unknown | Verified for CI context only |

Do not treat the reserved-prefix rule as a universal Axys reporting, IMEX, REP, or database rule unless supported by vendor documentation or production evidence.

### 2.5 Axys asset class evidence

| Statement | Confidence | Notes |
|---|---:|---|
| A third-party Axys asset-import workflow expects an `Asset Class` column in an Axys XLS export. | Verified for that export workflow | AdvisorEngine export column list includes `Asset Class`. |
| `Asset Class` is a supported Axys report/export label in at least one integration workflow. | Verified | The workflow combines portfolio, security, valuation, quantity, and asset class. |
| `Asset Class` is likely a common Axys classification/reporting output. | High Confidence | Supported by Axys product capability language plus third-party export field evidence. |
| The native Axys security-master field name for asset class is `Asset Class`. | Unknown | Export column headers are not necessarily native field names. |
| Whether Axys derives asset class from security type, security master fields, portfolio grouping, report logic, or a separate lookup table. | Unknown | No supplied source establishes derivation. |

### 2.6 Axys holdings/report examples involving classifications

| Report / output | Classification-related evidence | Confidence | Notes |
|---|---|---:|---|
| Portfolio Appraisal | Axys Report Writer can create a Portfolio Appraisal; sample displayed headings such as `EQUITY MUTUAL FUNDS`, `U.S. Equity`, and `Large Cap`. | Verified for example | Do not infer universal grouping hierarchy from the sample. |
| Portfolio Appraisal with `Portfolio Code` | `Portfolio Code` can be added as a column; Management Mode can produce a combined group report with owner portfolio code beside each holding. | Verified for CSSI example | Useful when classification output is generated for groups. |
| Assets Under Management / `AMAN.REP` | CSSI example modifies `AMAN.REP`; report values are sorted by asset class in the example context. | Verified for example | Does not prove classification field tokens beyond sample evidence. |
| `CDIhold.rep` | WealthTechs AIA uses `CDIhold.rep` to calculate historical holdings in certain workflows. | Verified for AIA workflow | Not a standard Axys classification report unless separately verified. |

### 2.7 Axys classification storage status

| Question | Status | Confidence |
|---|---|---:|
| Are sector, industry, country, and region stored directly on Axys security records? | Not established. | Unknown |
| Are classifications stored in separate lookup files? | Not established. | Unknown |
| Are classification assignments stored historically/effective-dated? | Not established. | Unknown |
| Does Axys store classification snapshots in performance files? | Not established. | Unknown |
| Do historical Axys reports use current classification metadata or historical metadata? | Not established. | Unknown |
| Are Axys sector/industry/country/region values vendor-provided, firm-defined, or both? | Not established. | Unknown |
| Is `Asset Class` derived from `Security Type`? | Not established. | Unknown |
| Does Axys expose raw classification assignments through IMEX? | Not established. | Unknown |
| Which REP/Replang field tokens return classifications? | Not established. | Unknown |

### 2.8 Axys processing and implementation cautions

| Caution | Confidence | Notes |
|---|---:|---|
| Direct parsing of Axys files is version-sensitive. | Verified from consultant source | AdventGuru research notes open text files in Axys v1.x, binary files in v2.x, IMEX in v3.x, and file conversion/format changes between Axys v3.7 and v3.8. |
| Prefer IMEX, REP/report output, or controlled exports over direct raw-file parsing. | High Confidence | Based on file-format risk and repository evidence standards. |
| Classification-dependent reports may be affected by metadata changes. | Unknown | Plausible but unverified; must be tested before documenting as behavior. |
| Classification joins should retain symbol, type, security name, and source/export context. | High Confidence | Prevents ambiguity from symbol/type duplicates and ticker/CUSIP variants. |

---

## 3. APX

### 3.1 APX classification-related capabilities

| Statement | Confidence | Notes |
|---|---:|---|
| APX has a report guide/public report documentation. | Verified | Supplied research identifies publicly indexed APX Reports Guide (`REP_APX.pdf`). |
| At least one APX report can display custom classification, industry group, or sector. | Verified, limited to snippet | The supplied research is based on a public search-result snippet; full report text was not available in the research pass. |
| The same APX report graphically displays equity allocations. | Verified, limited to snippet | Report name and full field list remain Unknown. |
| APX supports classification/allocation reporting beyond a single fixed sector hierarchy. | Medium Confidence | The phrase “any custom classification, industry group, or sector” implies flexibility, but details require the full APX Reports Guide. |
| APX supports custom reporting, dashboards, standard reports, and performance analytics at product level. | Verified | Product material in supplied REP research supports this. |
| APX has SQL/reporting options beyond IMEX. | Medium Confidence | AdventGuru research notes APX SQL Server, Public Views, Stored Accounting Functions, SSRS, REST API, and related reporting tools. |

### 3.2 APX security identity versus classification

| Item | APX evidence | Classification relevance | Confidence |
|---|---|---|---:|
| APX Symbol | ByAllAccounts CI uses APX Symbol in security translation. | Required join key in integration context. | Verified for CI context |
| APX Security Type / Type | ByAllAccounts CI uses APX Security Type with symbol. | Required join key in integration context; not the same as custom classification/sector. | Verified for CI context |
| `sec.inf` | CI uses APX Security Information from `sec.inf`; later research says APX CI uses `apxix.exe` to export Security (`sec.inf`) data. | Possible integration/security-reference file; classification fields not established. | Verified for CI context; native classification fields Unknown |
| `type.inf` | CI uses APX Security Type Information from `type.inf`; later research says APX CI uses `apxix.exe` to export Security Type (`type.inf`) data. | Security type metadata source; not a classification dictionary unless proven. | Verified for CI context; classification mapping Unknown |
| Duplicate symbol/type situations | CI evidence describes duplicate/ambiguous APX security matching. | Classification joins based only on ticker/symbol can be wrong. | Verified for CI context |

### 3.3 APX security type examples

The supplied research contains examples of APX security type codes in integration documentation. These are examples only.

| Example code | Context | Confidence | Notes |
|---|---|---:|---|
| `csus` | CI examples | Verified for example | Meaning requires type dictionary. |
| `efus` | CI security translation example | Verified for example | Example maps ticker `LMNVX` to symbol `524659208` and type `efus`. |
| `adus` | CI duplicate example | Verified for example | Same symbol may appear with different types. |
| `epus` | CI examples / fee-related integration context in other research | Verified for example | Do not equate with asset class or sector. |

### 3.4 APX reserved security type prefixes in CI matching

| Prefix | CI behavior | APX native behavior | Confidence |
|---|---|---|---:|
| `aw` | Excluded from ByAllAccounts CI security matching | Unknown | Verified for CI context only |
| `br` | Excluded from ByAllAccounts CI security matching | Unknown | Verified for CI context only |
| `ex` | Excluded from ByAllAccounts CI security matching | Unknown | Verified for CI context only |
| `ep` | Excluded from ByAllAccounts CI security matching | Unknown | Verified for CI context only |
| `pi` | Excluded from ByAllAccounts CI security matching | Unknown | Verified for CI context only |
| `rs` | Excluded from ByAllAccounts CI security matching | Unknown | Verified for CI context only |

Do not generalize this rule into APX SQL, APX reporting, APX public views, or native APX security type rules without additional evidence.

### 3.5 APX classification storage status

| Question | Status | Confidence |
|---|---|---:|
| What APX SQL tables hold classifications? | Not established. | Unknown |
| Are APX classifications stored on security master records, separate classification tables, report metadata, or custom fields? | Not established. | Unknown |
| Are APX custom classifications user-defined lookup values? | Plausible from report snippet, but storage not verified. | Medium Confidence |
| Are APX classification assignments historical/effective-dated? | Not established. | Unknown |
| Do APX historical reports use classification values as of the report date, current classifications, or stored snapshots? | Not established. | Unknown |
| Does APX performance reporting store classification snapshots? | Not established. | Unknown |
| Does APX export Axys v3 format with classification fields? | Not established. | Unknown |
| Are APX public views sufficient to extract all classification data? | Not established. | Unknown |

### 3.6 APX report and extraction implications

| Extraction path | Classification relevance | Confidence | Notes |
|---|---|---:|---|
| APX standard reports | At least one report supports custom classification, industry group, or sector. | Verified, limited to snippet | Report name and full columns Unknown. |
| APX REP/Replang reports | Consultant research says APX users can still use Report Writer Pro/Replang. | Medium Confidence | Exact field tokens Unknown. |
| APX SSRS / SQL reporting | APX has SQL Server/reporting options according to consultant research. | Medium Confidence | Exact tables/views/stored functions Unknown. |
| APX IMEX | APX maintains IMEX functionality according to consultant research. | Verified from consultant source | Classification object names Unknown. |
| APX fixed-format IMEX | Consultant research says APX eliminated fixed-format file generation in APX v1.x through v4.x while retaining IMEX. | Verified from consultant source | Do not design APX classification extraction assuming fixed-format IMEX output. |

---

## 4. IMEX

### 4.1 IMEX facts relevant to classifications

| Statement | Axys | APX | Confidence | Notes |
|---|---:|---:|---:|---|
| IMEX exists as an import/export mechanism. | Yes | Yes | Verified from supplied research | Axys v3.x IMEX introduction and APX v1.x through v4.x IMEX continuity are supported by consultant research. |
| IMEX was introduced in the Axys v3.x era to support CSV, tab, and fixed-format import/export. | Yes | N/A | Verified from consultant source | Axys v1.x/v2.x file-format history is also supported in research. |
| APX maintained IMEX functionality in APX v1.x through v4.x. | N/A | Yes | Verified from consultant source | Exact modern behavior not fully established. |
| APX eliminated fixed-format file generation. | N/A | Yes | Verified from consultant source | Do not assume fixed-format APX IMEX output. |
| IMEX can move transaction and label data through Trade Blotter context. | Yes | Yes | Verified from consultant source | Relationship of “label” to classification is Unknown. |
| Axys CI uses `imex32.exe` to export Security (`sec.inf`) and Security Type (`type.inf`) data. | Yes | N/A | Verified for CI context | Complete field layouts Unknown. |
| APX CI uses `apxix.exe` to export Security (`sec.inf`) and Security Type (`type.inf`) data. | N/A | Yes | Verified for CI context | Complete field layouts Unknown. |
| Exact native IMEX object names for classifications are known. | Unknown | Unknown | Unknown | No supplied material identifies them. |

### 4.2 IMEX object status for classification-related data

| Data need | Axys IMEX status | APX IMEX status | Confidence |
|---|---|---|---:|
| Security master / security reference export | `sec.inf` export/use verified in CI context; native IMEX object name Unknown | `sec.inf` export/use verified in CI context; native object name Unknown | Verified for CI file use; object Unknown |
| Security type export | `type.inf` export/use verified in CI context; native IMEX object name Unknown | `type.inf` export/use verified in CI context; native object name Unknown | Verified for CI file use; object Unknown |
| Asset class export | Not established | Not established | Unknown |
| Sector export | Not established | Not established | Unknown |
| Industry group export | Not established | Not established | Unknown |
| Country / region export | Not established | Not established | Unknown |
| Custom classification export | Not established | Not established | Unknown |
| Portfolio grouping categories | Not established | Not established | Unknown |
| Classification lookup tables | Not established | Not established | Unknown |
| Classification import/update | Not established | Not established | Unknown |

### 4.3 IMEX implementation guidance

| Guidance | Confidence | Rationale |
|---|---:|---|
| If extracting security-level classifications through IMEX, preserve symbol and type in the output. | High Confidence | Symbol alone can be ambiguous in CI evidence. |
| Treat `Asset Class`, `Sector`, `Industry Group`, `Country`, `Region`, and custom classification fields as Unknown until seen in an export definition or sample. | Unknown | No official object/field list was supplied. |
| Store IMEX object name, export definition, file format, run date, and Axys/APX version with any extracted classification dataset. | High Confidence as implementation control | Classification extraction is version/source sensitive. |
| Do not treat `sec.inf` or `type.inf` as a complete classification extract without inspecting field contents. | High Confidence | Sources verify these files in security/type context, not full classification context. |
| If APX classification data is needed, consider SQL/report sources in addition to IMEX. | Medium Confidence | APX database/report extraction options are supported by consultant research; schema Unknown. |

### 4.4 Minimal IMEX evidence needed

| Evidence | Would establish |
|---|---|
| Axys IMEX object list / help / screenshots | Whether classification-related objects exist and what they are called. |
| APX IMEX object list / help / screenshots | Same for APX. |
| Sample Axys security master/security reference IMEX export | Whether asset class, sector, industry, country, region, or custom classification are exported. |
| Sample APX security master/security reference IMEX export | Same for APX. |
| Sample classification lookup export | Whether classification code/name/hierarchy/sort order exists outside security master. |
| Successful classification import sample and logs | Whether classification values can be imported/updated and required fields. |

---

## 5. REP / Replang / Report Output

### 5.1 REP facts relevant to classifications

| Statement | Axys | APX | Confidence | Notes |
|---|---:|---:|---:|---|
| Axys reports are written in RepLang. | Yes | Unknown | Verified for Axys | CSSI research supports Axys RepLang. |
| Axys supports Report Writer Pro. | Yes | N/A | Verified | Product material supports this. |
| APX users can use Report Writer Pro or Replang source edits. | N/A | Yes | Medium Confidence | Consultant source; exact APX source/keyword set Unknown. |
| Axys and APX reports can be used by connector workflows through standard reports, macros, REP32, and RepLang scripting. | Yes | Yes | Verified for connector | Data Broker evidence. |
| REP/report output can be exported/automated and used as an extraction path. | Yes | Yes | High Confidence | Supported by AdventGuru and connector research. |
| Exact REP report names for Axys classifications are known. | Unknown | N/A | Unknown | No supplied report catalog. |
| Exact APX report name for the custom classification/industry group/sector report is known. | N/A | Unknown | Unknown | Search snippet lacked report name. |
| Exact RepLang field tokens for classifications are known. | Unknown | Unknown | Unknown | No supplied report source. |

### 5.2 Known report names and report files relevant to classification context

| Report / file | System | Classification relevance | Confidence |
|---|---|---|---:|
| `Portfolio Appraisal` | Axys | Holdings/assets point-in-time report; sample grouped by broad asset/security categories. | Verified for holdings/reporting context |
| `Portfolio Appraisal` | APX | APX reports guide snippet says Portfolio Appraisal shows holdings by individual tax lot or position. | Medium Confidence |
| `AMAN.REP` | Axys | Assets Under Management report in CSSI example; output sorted by asset class in example. | Verified for example |
| `CDIhold.rep` | Axys | AIA historical holdings calculation report; may support classification-enriched holdings if report logic includes classifications, but not established. | Verified for AIA workflow; classification Unknown |
| `CDIhold.rep` | APX | AIA historical holdings calculation report; same caution. | Verified for AIA workflow; classification Unknown |
| APX custom classification / industry group / sector allocation report | APX | Public APX Reports Guide snippet says a report can display custom classification, industry group, or sector. | Verified, limited to snippet; report name Unknown |

### 5.3 REP extraction strategy for classifications

| Strategy | When to use | Confidence |
|---|---|---:|
| Use standard report output when the report itself is the source of truth for grouped allocation/performance. | When users rely on the report and report logic may include rollups or transformations not present in raw master data. | High Confidence |
| Use custom RepLang/Report Writer output for stable data extracts. | When a stable column layout is needed and field tokens are known locally. | High Confidence as approach; field tokens Unknown |
| Use IMEX/security master export for raw classification assignments if fields are available. | When raw classification assignment, not report rollup, is needed. | Medium Confidence; object/fields Unknown |
| In APX, consider SQL/Public View/SSRS sources when IMEX or REP does not expose needed classification fields. | APX environments with accessible database/reporting infrastructure. | Medium Confidence; schemas Unknown |
| Store report name, report source file, run parameters, version, and as-of date with any report-derived classification extract. | Always. | High Confidence as control |

### 5.4 REP limitations to preserve

| Limitation | Confidence |
|---|---:|
| Report output labels are not necessarily native field names. | High Confidence |
| Report group headings are not necessarily raw stored classification values. | High Confidence |
| Custom report behavior may be installation-specific. | High Confidence |
| RepLang field tokens for asset class, sector, industry, country, region, and custom classification are not established by supplied research. | Unknown |
| Whether reports use stored values or recalculate values at runtime depends on report internals and is Unknown for classification reports. | Unknown |

---

## 6. Data Model

### 6.1 Concepts to keep separate

| Concept | Definition in this chapter | Evidence status |
|---|---|---:|
| Security Symbol | Product security identifier used in matching; may be ticker, CUSIP-like, or firm-specific. | Verified for CI context |
| Security Type | Product security type code used with symbol in security matching. | Verified for CI context |
| Asset Class | Reporting/grouping classification seen in Axys product and export evidence. | High Confidence for Axys; APX Medium Confidence |
| Sector | Classification/reporting group. | Verified as reporting category |
| Industry Group | APX report category in public snippet; security import dependency in security-master research. | Verified in limited contexts |
| Industry Sector | Security-master import dependency in AdventGuru research. | Verified as import dependency; exact field Unknown |
| Country | Axys product material says performance can be displayed by country. | Verified as Axys reporting category |
| Region | Axys product material says performance can be displayed by region. | Verified as Axys reporting category |
| Custom Classification | APX report snippet says a report can display any custom classification. | Verified, limited to snippet |
| Portfolio Grouping Category | Axys can group portfolios by manager, asset class, investment objective, or chosen category. | Verified |
| Label | Term appears in IMEX/Trade Blotter context. | Relationship to classification Unknown |

### 6.2 Candidate logical model

The following model is a documentation and extraction model. It is not asserted as native Axys/APX schema.

| Logical entity | Purpose | Candidate keys | Candidate fields | Confidence |
|---|---|---|---|---:|
| `security_identity` | Preserve system security identity. | system, symbol, type | name, ticker, CUSIP, source | High Confidence as integration model |
| `security_type` | Preserve product security type metadata. | system, type | type description, reserved prefix flag, source | Medium Confidence |
| `classification_scheme` | Identify classification family. | system, scheme name | asset class, sector, industry group, country, region, custom scheme | Medium Confidence |
| `classification_value` | Store lookup value in a scheme. | system, scheme, code/name | display name, parent, sort order | Unknown as native; useful as model |
| `security_classification_assignment` | Assign a security to a classification value. | system, symbol, type, scheme | value, effective date if present, source | Unknown as native; useful as model |
| `portfolio_grouping_assignment` | Assign a portfolio/account to grouping category. | system, portfolio code, scheme/category | manager, objective, asset class, custom category | Medium Confidence for Axys portfolio grouping |
| `report_classification_output` | Capture report-derived classifications/allocations. | report, as-of date, portfolio/group, classification | market value, percent, weight, return, source report | Medium Confidence |

### 6.3 Data lineage requirements

| Requirement | Reason | Confidence |
|---|---|---:|
| Store source system: Axys or APX. | Behaviors and extraction paths differ. | Verified repository rule |
| Store extraction path: IMEX, REP, SQL/public view, third-party export, manual report. | Field meanings and timing may differ by path. | High Confidence |
| Store source artifact: report file, IMEX object, export file, SQL view, integration tool. | Needed to audit report-derived vs raw classification data. | High Confidence |
| Store as-of date and run date separately. | Historical classifications are Unknown; run date may matter. | High Confidence |
| Store both symbol and type for security-level classification joins. | Avoids duplicate/ambiguous matches. | Verified for CI context |
| Preserve display labels exactly as exported. | Report/export labels may not equal native fields. | High Confidence |

---

## 7. Common Fields and Field Dictionary

This table is intentionally conservative. It catalogs observed labels and useful candidate fields. It is not a native Axys/APX data dictionary.

| Field | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---:|
| `Symbol` | Product security symbol. | Yes in CI context | Yes in CI context | Unknown native | Likely visible in reports | Verified for CI context |
| `Security Symbol` | Security symbol label in AdvisorEngine Axys export. | Yes | Unknown | Unknown | Likely report/export | Verified for one Axys export |
| `Axys Symbol` | Target Axys security symbol in CI security translation. | Yes | No | CI-related | No | Verified for CI context |
| `APX Symbol` | Target APX security symbol in CI security translation. | No | Yes | CI-related | No | Verified for CI context |
| `Type` | Security type associated with symbol in CI outputs. | Yes | Yes | Unknown native | Likely report/export | Verified for CI context |
| `Sec Type Code` | Security type code label in AdvisorEngine Axys export. | Yes | Unknown | Unknown | Likely report/export | Verified for one Axys export |
| `Security Type` | Security type label/description or code depending on export context. | Yes | Yes | Unknown | Likely report/export | Verified in CI/export contexts |
| `Security` | Security name/description in AdvisorEngine Axys export and holdings reports. | Yes | Likely | Unknown | Yes | Verified for Axys export/report sample |
| `Asset Class` | Classification/grouping field. | Yes | Likely | Unknown | Likely | High Confidence for Axys; Medium for APX |
| `Sector` | Classification/reporting group. | Yes | Yes | Unknown | Likely | Verified as reporting category |
| `Industry Group` | APX report classification category; security import dependency context. | Unknown | Yes | Unknown | Likely | Verified in limited contexts |
| `Industry Sector` | Referenced as a required dependency for security import in security-master research. | Yes/likely | Yes/likely | Import dependency | Unknown | Verified as dependency; exact field Unknown |
| `Industry` | Common classification concept; exact Axys/APX field not supplied. | Unknown | Unknown | Unknown | Unknown | Unknown |
| `Country` | Axys reporting category. | Yes | Likely | Unknown | Likely | Verified for Axys reporting; APX Unknown |
| `Region` | Axys reporting category. | Yes | Likely | Unknown | Likely | Verified for Axys reporting; APX Unknown |
| `Custom Classification` | APX report category from public snippet. | Unknown | Yes | Unknown | Likely | Verified, limited to APX snippet |
| `Portfolio Name` | Portfolio identity/display field in AdvisorEngine Axys export. | Yes | Unknown | Unknown | Yes | Verified for one Axys export |
| `Portfolio Code` | Portfolio code in AdvisorEngine Axys export and Axys Portfolio Appraisal column. | Yes | Unknown | Unknown | Yes | Verified for Axys examples |
| `APX Portfolio Code` | Portfolio-code translation field identifying target APX portfolio in CI. | No | Yes | CI-related | No | Verified for CI workflow |
| `Market Value` | Holding valuation used in classification/allocation reporting. | Yes | Likely | Unknown | Yes | Verified for Axys export/report sample |
| `Quantity` | Holding quantity used in holdings/classification outputs. | Yes | Likely | Unknown | Yes | Verified for Axys export/report sample |
| `Pct Assets` | Percent-of-assets field in Axys Portfolio Appraisal sample. | Yes | Unknown | Unknown | Yes | Verified for Axys sample |
| `Weight` | Allocation/percentage field concept. | Likely | Likely | Unknown | Likely | Medium Confidence |
| `Classification Code` | Possible classification lookup code. | Unknown | Unknown | Unknown | Unknown | Unknown |
| `Classification Name` | Possible classification lookup name. | Unknown | Unknown | Unknown | Unknown | Unknown |
| `Effective Date` | Possible date for historical/effective-dated classification assignment. | Unknown | Unknown | Unknown | Unknown | Unknown |

---

## 8. Examples

### 8.1 Axys asset export with classification field

A third-party Axys asset import workflow expects an XLS export with the following columns in order:

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
|---|---:|
| `Asset Class` can appear in an Axys report/export workflow. | Verified |
| The export combines portfolio identity, security identity, valuation, quantity, and classification. | Verified for that workflow |
| The export does not prove that `Asset Class` is a native security-master field. | Verified limitation |
| The export does not prove an IMEX field name. | Verified limitation |

### 8.2 Axys security translation and classification join caution

Observed CI translation pattern:

```text
WP Ticker:    LMNVX
WP Name:      LEGG MASON VLE TR INSTL
Axys Symbol:  524659208
Axys Type:    efus
```

| Interpretation | Confidence |
|---|---:|
| The integration maps an external ticker to an Axys symbol that may be CUSIP-like. | Verified |
| Security identity in this workflow uses symbol plus type. | Verified |
| Classification fields are not shown in this translation example. | Verified |
| Downstream classification joins should retain both product symbol and product type. | High Confidence |

### 8.3 APX security translation and classification join caution

Observed CI translation pattern:

```text
WP Ticker:  LMNVX
WP Name:    LEGG MASON VLE TR INSTL
APX Symbol: 524659208
APX Type:   efus
```

| Interpretation | Confidence |
|---|---:|
| The integration maps an external ticker to an APX symbol that may be CUSIP-like. | Verified |
| APX security identity in this workflow uses symbol plus type. | Verified |
| Classification fields are not shown in this translation example. | Verified |
| Downstream classification joins should retain both product symbol and product type. | High Confidence |

### 8.4 Duplicate security matching scenario

ByAllAccounts CI research describes duplicate matching when:

```text
Scenario A: same symbol appears under multiple security types.
Scenario B: same security is defined once using ticker as symbol and once using CUSIP as symbol.
Scenario C: overlapping security translations match the same source security.
```

| Implementation consequence | Confidence |
|---|---:|
| Symbol alone is not safe as a classification join key. | Verified for CI context |
| Ticker-only joins can misclassify holdings when ticker and CUSIP variants coexist. | High Confidence |
| Store product symbol, product type, security name, CUSIP/ticker when available, and source artifact. | High Confidence as implementation guidance |

### 8.5 APX custom classification report

The APX Reports Guide snippet in the supplied research states that a report can display:

```text
any custom classification, industry group, or sector
```

and graphically display equity allocations.

| Interpretation | Confidence |
|---|---:|
| APX has report support for custom classification, industry group, and sector. | Verified, limited to snippet |
| The exact report name is not established. | Unknown |
| The exact output fields are not established. | Unknown |
| It is not known whether the report exports raw classification assignments or only aggregate allocation values. | Unknown |

### 8.6 Historical classification behavior test case

Because historical classification behavior is Unknown, use a controlled test before relying on historical outputs.

| Step | Action |
|---:|---|
| 1 | Select a test security held in a closed historical month. |
| 2 | Record its current asset class, sector, industry, country, region, and any custom classifications. |
| 3 | Run the target historical holdings/performance/classification report. |
| 4 | In a test environment, change one classification value. |
| 5 | Rerun the same report for the same historical period. |
| 6 | Compare whether the old period appears under the old or new classification. |

| Outcome | Interpretation |
|---|---|
| Historical output changes | Report likely uses current classification lookup or current report mapping. |
| Historical output does not change | Report may use stored classification snapshots, stored performance/report output, or another historical mechanism. |
| Different reports behave differently | Classification behavior is report-specific. |

---

## 9. Known Issues / Quirks

| Issue / quirk | Axys | APX | Confidence | Impact |
|---|---:|---:|---:|---|
| Symbol alone may be ambiguous. | Yes | Yes | Verified for CI context | Use symbol + type when joining classifications. |
| Same symbol can exist with different security types. | Yes | Yes | Verified for CI context | Classification joins can misassign values. |
| Same security can be defined by ticker and CUSIP as separate symbols. | Yes | Yes | Verified for CI context | Ticker-only or CUSIP-only joins can produce duplicate/missing classifications. |
| Reserved CI type prefixes `aw`, `br`, `ex`, `ep`, `pi`, `rs` are excluded from CI matching. | Yes | Yes | Verified for CI context | CI extracts/matches may omit securities with those prefixes. |
| Security Type and classification fields can be confused by users. | Yes | Yes | High Confidence | Documentation and data contracts must separate them. |
| Direct Axys file parsing is version-sensitive. | Yes | N/A | Verified from consultant source | Prefer IMEX/REP/report output or controlled exports. |
| Axys file formats changed across versions. | Yes | N/A | Verified from consultant source | Version-specific parsers can break. |
| APX fixed-format IMEX generation was eliminated. | N/A | Yes | Verified from consultant source | Do not assume fixed-format APX classification exports. |
| APX has SQL/reporting extraction options beyond IMEX. | N/A | Yes | Medium Confidence | SQL/public-view/reporting schema still Unknown. |
| Third-party export column names may not equal native field names. | Yes | Yes | High Confidence | Treat headers such as `Asset Class` as export/report labels until native docs confirm. |
| Historical classification behavior is not established. | Unknown | Unknown | Unknown | Critical risk for historical performance, attribution, and audit. |

---

## 10. Version Differences

| Version / system | Statement | Confidence | Notes |
|---|---|---:|---|
| Axys v1.x | Maintained open text file structure. | Verified from consultant source | Relevant to direct-file parsing risk. |
| Axys v2.x | Introduced binary file format. | Verified from consultant source | Relevant to direct-file parsing risk. |
| Axys v3.x | IMEX introduced, supporting CSV, tab, and fixed formats. | Verified from consultant source | Relevant to extraction strategy. |
| Axys v3.7 to v3.8 | Upgrade required file conversion; some v3.8 files had different formats. | Verified from consultant source | Do not assume direct file layout stability. |
| APX v1.x through v4.x | Maintained IMEX functionality. | Verified from consultant source | Exact modern APX IMEX behavior still requires documentation. |
| APX v1.x through v4.x | Fixed-format generation eliminated. | Verified from consultant source | Classification exports should not rely on fixed-format output. |
| APX | Can export data to Axys v3 format. | Verified from consultant source | Whether this includes classification fields is Unknown. |
| Axys/APX classification schemas by version | No version-specific classification schema differences were supplied. | Unknown | Need release notes, export samples, or schema documentation. |

---

## 11. Implementation Guidance

### 11.1 Recommended extraction decision table

| Need | Preferred source if available | Reason | Confidence |
|---|---|---|---:|
| Raw security classification assignments | IMEX/security master export or APX SQL/public view | Raw data is easier to normalize than report rollups. | Medium Confidence; object/schema Unknown |
| Classification allocation report matching Advent output | REP/report output | Report output may include Advent-specific grouping/rollup logic. | High Confidence |
| Portfolio grouping categories | Portfolio/account master export, group report, or Axys/APX configuration export | Portfolio grouping is separate from security classification. | Medium Confidence |
| Historical classification audit | Controlled before/after report test plus export snapshots | Historical behavior is Unknown. | High Confidence as testing requirement |
| APX classification extraction | APX report, SQL/public view, or IMEX export depending site capabilities | APX has multiple access paths. | Medium Confidence |
| Axys classification extraction | REP/report output or IMEX export; avoid raw file parsing unless unavoidable | Direct file parsing is version-sensitive. | High Confidence |

### 11.2 Minimum safe downstream schema

For downstream tools that consume Axys/APX classifications, a minimum safe schema should preserve source identity and lineage.

| Field | Required? | Reason |
|---|---:|---|
| `source_system` | Yes | Axys and APX behavior can differ. |
| `extract_method` | Yes | IMEX, REP, SQL, report export, third-party integration, etc. |
| `source_artifact` | Yes | Report name, file name, IMEX object, SQL view, export file. |
| `run_date` | Yes | Historical behavior may depend on current metadata at run time. |
| `as_of_date` | Yes | Needed for holdings/performance/classification context. |
| `portfolio_code` | When portfolio/account output is present | Needed for portfolio grouping and allocations. |
| `security_symbol` | When security-level output is present | Required join key component. |
| `security_type` | When security-level output is present | Required join key component in integration context. |
| `security_name` | Recommended | Helps identify ticker/CUSIP duplicate issues. |
| `classification_scheme` | Yes for classification values | Examples: Asset Class, Sector, Industry Group, Country, Region, Custom. |
| `classification_value` | Yes for classification values | Preserve exact report/export label. |
| `classification_code` | If available | Do not invent if absent. |
| `effective_date` | If available | Historical/effective-dated classifications are Unknown. |
| `confidence` | Recommended | Preserve evidence level. |

### 11.3 Do not promote these unsupported assumptions

| Assumption | Correct treatment |
|---|---|
| Axys stores sector/industry/country directly in `sec.inf`. | Unknown |
| APX stores classifications in a specific SQL table. | Unknown |
| IMEX object `X` exports classifications. | Unknown until verified from IMEX docs/sample. |
| REP field token `X` returns asset class/sector/industry. | Unknown until verified from `.REP` source or programmer guide. |
| Classifications are historical/effective-dated. | Unknown |
| Historical reports use current classification values. | Unknown |
| Historical reports use report-date classification values. | Unknown |
| Historical reports use stored classification snapshots. | Unknown |
| Asset Class is always derived from Security Type. | Unknown |
| Sector/industry hierarchy follows GICS, BICS, SIC, or any named taxonomy. | Unknown |
| Axys and APX use identical classification fields. | Unknown |
| Advent “labels” are equivalent to classifications. | Unknown |

---

## 12. References

This chapter was drafted from the supplied research files and their referenced source material. The most relevant supplied research files were:

| Supplied research file | Use in this chapter |
|---|---|
| `AXYS_APX_REFERENCE_BLUEPRINT(29).md` | Governing editorial rules, confidence labels, chapter template, and Unknown handling. |
| `Research_11_Classifications.md` | Primary classification research: Axys/APX classification capabilities, field candidates, IMEX/REP unknowns, test plan, quirks. |
| `Research_04_Security_Master(12).md` | Security identity, `sec.inf`, `type.inf`, symbol/type matching, security import dependencies, duplicate security quirks. |
| `Research_06_Holdings(3).md` | Portfolio Appraisal, holdings fields, group/management mode behavior, report examples, `CDIhold.rep`. |
| `Research_12_IMEX(5).md` | IMEX definition, Axys/APX IMEX behavior, file/folder evidence, REP32 extraction context, direct file access cautions. |
| `Research_13_REP(4).md` | RepLang, `.REP`, Report Writer Pro, REP32, report execution, APX reporting paths. |
| `Research_05_Transactions(11).md` | Trade Blotter/label context and transaction-related caution where labels are not proven classifications. |
| `Research_08_Pricing(3).md` | Pricing/reporting and security master dependency cautions relevant to classification-enriched holdings/valuation. |
| `Research_09_Corporate_Actions(1).md` | Security master and corporate-action context; no direct classification field definitions supplied. |

Referenced external source categories from the supplied research include:

| Source category | Supported facts used here |
|---|---|
| SS&C Advent Axys product material | Axys reporting, Report Writer Pro, portfolio grouping categories, performance by asset class/sector/country/region. |
| SS&C Advent APX product material and APX Reports Guide snippet | APX report library, custom reporting, classification/industry/sector allocation report snippet. |
| ByAllAccounts Custodial Integrator guides for Axys/APX | Symbol/type security identity, `sec.inf`, `type.inf`, security translation fields, duplicate matching, reserved CI type prefixes. |
| AdvisorEngine Advent Axys Asset Import KB | Axys export column list including `Asset Class`. |
| CSSI Axys report tutorials | Portfolio Appraisal behavior, `Portfolio Code` column, Management Mode, `AMAN.REP`, RepLang examples. |
| WealthTechs AIA manuals | `CDIhold.rep` and report-mediated holdings extraction workflows. |
| AdventGuru consultant articles | Axys/APX IMEX/version history, direct file access risks, APX SQL/reporting options, Report Writer/Replang context. |
| Salentica/Black Diamond Data Broker documentation | REP32, standard reports, macros, RepLang extraction for Axys/APX connector. |

---

## 13. Unknowns

### 13.1 Highest-priority unknowns

| Priority | Unknown | Axys | APX | Evidence needed |
|---:|---|---:|---:|---|
| 1 | Where are security classifications stored? | Unknown | Unknown | Security master docs, APX SQL dictionary, IMEX export, REP source, sample data. |
| 1 | Exact IMEX object names for classification/security reference export. | Unknown | Unknown | IMEX manuals, object lists, screenshots, `.ini`/definition files. |
| 1 | Exact field names for asset class, sector, industry, country, region, custom classification. | Unknown | Unknown | Vendor field dictionary, IMEX/REP exports, APX public views. |
| 1 | Exact REP/Replang field tokens for classifications. | Unknown | Unknown | `.REP` source, Report Writer docs, RepLang Programmer's Guide. |
| 1 | Whether classification assignments are historical/effective-dated. | Unknown | Unknown | Vendor docs or controlled before/after tests. |
| 1 | Whether historical reports use current classifications or historical snapshots. | Unknown | Unknown | Controlled test evidence and report internals. |
| 2 | Whether classifications are security-level, portfolio-level, or both in native storage. | Unknown | Unknown | Security/portfolio exports and configuration documentation. |
| 2 | Whether a security can belong to multiple custom classification schemes. | Unknown | Unknown | Configuration screenshots/docs and sample exports. |
| 2 | Whether sector/industry/country/region are vendor taxonomy values or firm-defined. | Unknown | Unknown | Site configuration, lookup exports, vendor docs. |
| 2 | Whether APX export to Axys v3 format includes classification fields. | N/A | Unknown | APX export sample. |
| 2 | Whether labels are related to classifications. | Unknown | Unknown | IMEX/Trade Blotter/label documentation. |

### 13.2 Evidence to request before making stronger claims

| Requested artifact | Why it matters |
|---|---|
| Sample Axys IMEX export of security master/security reference data | Confirms whether classification fields appear and what they are called. |
| Sample APX IMEX export of security master/security reference data | Confirms APX field names and export structure. |
| Axys/APX IMEX object list or screenshots | Confirms object names and supported import/export paths. |
| REP/Replang source for holdings by asset class | Confirms report field tokens and grouping logic. |
| REP/Replang source for performance by asset class/sector/country/region | Confirms performance classification reporting behavior. |
| APX Reports Guide full PDF or screenshots around custom classification report | Confirms report name, parameters, and fields. |
| APX SQL public view/data dictionary examples | Confirms classification table/view names and field names. |
| Small anonymized classification export | Confirms symbol/type/name/CUSIP/asset class/sector/industry/country/region/custom fields. |
| Before/after classification edit test | Confirms historical classification behavior. |
| Versioned exports from Axys/APX environments | Confirms version differences and field stability. |
