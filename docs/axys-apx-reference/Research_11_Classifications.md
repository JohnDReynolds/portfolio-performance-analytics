# Research Notes: 11-Classifications.md

Repository: AXYS / APX Reference Repository  
Research file: `research/11-Classifications-research.md`  
Prepared: 2026-06-29  
Governing specification: `AXYS_APX_REFERENCE_BLUEPRINT.md`, Version 2.0

---

## 0. Scope and Confidence Rules

This file collects research for the planned chapter `docs/11-Classifications.md`.

The focus is factual, implementation-oriented information about how classifications appear to behave in SS&C Advent Axys and SS&C Advent APX, especially as they relate to:

- security master / reference data
- portfolio grouping
- performance reporting
- holdings reporting
- IMEX
- REP / Replang reporting
- field names observed in public documentation or third-party integration material
- report names or report capabilities
- implementation quirks
- version differences
- unknowns that require vendor documentation, exported files, or production examples

Every important technical statement is assigned one of:

| Confidence | Meaning |
|---|---|
| Verified | Directly supported by an identified source, sample export, report, or vendor/third-party documentation. |
| High Confidence | Strongly supported by multiple sources or by consistent Axys/APX behavior known from integration/reporting context, but not yet proven by supplied sample files. |
| Medium Confidence | Plausible and consistent with available sources, but needs confirmation from real IMEX output, REP source, vendor docs, or production observation. |
| Unknown | Not sufficiently supported. Must not be implemented as fact without more evidence. |

> Editorial note: This is a research file, not the final chapter. It intentionally preserves uncertainty.

---

## 1. Executive Summary

| Topic | Research Finding | Confidence |
|---|---|---|
| Axys supports classification-based reporting | SS&C's Axys product material states that Axys can display performance by portfolios, asset classes, sectors, countries, or regions. | Verified |
| Axys supports arbitrary portfolio grouping categories | SS&C's Axys product material states that portfolios can be grouped by manager, asset class, investment objective, or any category the firm chooses. | Verified |
| APX reports can use custom classifications | The publicly indexed APX Reports Guide snippet states that a report can be run to display any custom classification, industry group, or sector and graphically display equity allocations. | Verified, limited to snippet |
| Axys/APX share Replang/REP report customization lineage | AdventGuru describes report development options for both Axys and APX, including Report Writer Pro and Replang reports. | Verified |
| IMEX exists in Axys v3.x and APX | AdventGuru states that IMEX was introduced with Axys v3.x and that APX v1.x through v4.x maintains IMEX functionality, with fixed-format generation removed in APX. | Verified |
| Axys v1.x had open text files; Axys v2.x introduced binary files | AdventGuru reports this version-history point. | Verified from consultant source |
| Direct file access is risky across Axys versions | AdventGuru warns that file formats changed across versions, including conversion from Axys v3.7 to v3.8. | Verified from consultant source |
| Axys/APX security matching uses symbol and type | ByAllAccounts Custodial Integrator docs for both Axys and APX refer to Security Information from `sec.inf` and `type.inf` and use product security symbol and security type for matching. | Verified |
| Reserved security type prefixes exist in Axys/APX CI matching | ByAllAccounts docs state that security types with prefixes `aw`, `br`, `ex`, `ep`, `pi`, `rs` are treated as reserved and excluded from security match processing. | Verified for Custodial Integrator matching; not necessarily global Axys/APX behavior |
| Classification field names in native Axys/APX security master | Public sources do not provide a complete authoritative field list for security classifications. | Unknown |
| Exact IMEX object name for classification export/import | Not verified from public sources in this research pass. | Unknown |
| Exact REP report names for Axys classification reports | Not verified from public sources in this research pass, except generic capability statements and APX Reports Guide snippet. | Unknown |
| APX database table names for classifications | Not verified from public sources in this research pass. | Unknown |
| Whether classifications are stored historically | Not verified. Needs vendor docs or sample reports/exports showing effective dates or historical lookup behavior. | Unknown |

---

## 2. Source Inventory

### 2.1 Supplied Repository Specification

| Source | Type | What it supports |
|---|---|---|
| `AXYS_APX_REFERENCE_BLUEPRINT.md`, Version 2.0 | User-supplied governing specification | Editorial rules: separate Axys/APX, classify confidence, preserve unknowns, prefer evidence, produce research suitable for repository. |

### 2.2 Public / Web Sources Used

| Source | Type | Relevant Evidence |
|---|---|---|
| SS&C Advent Axys product page | Vendor product page | Axys reporting, performance measurement, grouping by manager/asset class/investment objective/custom category, performance by asset classes/sectors/countries/regions. |
| AdventGuru, “Integration” / report development discussion | Consultant / practitioner article | Axys version history, IMEX introduction, APX IMEX behavior, report export/custom report/Replang options, direct file access warning. |
| Advent Portfolio Exchange Reports Guide (`REP_APX.pdf`) | Vendor report guide, public PDF | Search-result snippet indicates APX report can display custom classification, industry group, or sector and graphically display equity allocations. Full text was not reliably accessible during this pass. |
| ByAllAccounts Custodial Integrator User Guide for Axys | Third-party integration manual | Axys security master references: `sec.inf`, `type.inf`, Axys Symbol, Axys Security Type; security matching logic; reserved type prefixes. |
| ByAllAccounts Custodial Integrator User Guide for APX | Third-party integration manual | APX security master references: `sec.inf`, `type.inf`, APX Symbol, APX Security Type; security matching logic; reserved type prefixes. |
| AdvisorEngine Advent Axys Asset Import KB | Third-party integration KB | Example Axys export column list includes Portfolio Name, Portfolio Code, Security, Sec Type Code, Security Symbol, Security Type, Market Value, Quantity, Asset Class. |
| Salentica Elements Data Broker Axys/APX KB | Third-party integration KB | Axys and APX are described as on-prem systems that can send data to Data Broker. |

---

## 3. Axys Classification Research

### 3.1 Axys Product-Level Capabilities

| Statement | Evidence | Confidence |
|---|---|---|
| Axys includes reporting and report customization capabilities. | SS&C Axys product page says Axys has an extensive library of pre-defined reports and report customization. | Verified |
| Axys supports Report Writer Pro. | SS&C Axys product page says users can choose from hundreds of pre-defined reports or create their own with Axys Report Writer Pro. | Verified |
| Axys can manage and report on portfolios grouped by manager, asset class, investment objective, or any chosen category. | SS&C Axys product page states this capability. | Verified |
| Axys can display performance by portfolios, asset classes, sectors, countries, or regions. | SS&C Axys product page states this capability. | Verified |
| Axys classifications may be used both at portfolio-group level and security/holding level. | Product page references both grouping portfolios by categories and performance display by asset classes/sectors/countries/regions; the exact data storage layer is not shown. | Medium Confidence |
| Axys supports firm-defined classification categories beyond vendor-provided categories. | Product page says portfolios can be grouped by any category; whether this applies equally to security-level classifications is not directly documented in the source. | Medium Confidence |

### 3.2 Axys Security Master / Security Type Evidence

| Statement | Evidence | Confidence |
|---|---|---|
| Axys security matching in ByAllAccounts Custodial Integrator uses Axys Security Information from `sec.inf` and `type.inf`. | ByAllAccounts Axys CI Guide states that CI uses Security Translations and Axys Security Information from `sec.inf` and `type.inf` files. | Verified for CI integration context |
| The Axys security identifier pair includes Axys Symbol and Axys Security Type. | ByAllAccounts Axys CI Guide lists Axys Symbol and Axys Security Type as the target Axys security information in security translations. | Verified for CI integration context |
| Axys security type examples include `csus`, `efus`, `tfus`, `oaus`, `CAUS`, and `CSUS` in CI examples and output field descriptions. | ByAllAccounts Axys CI Guide gives examples such as `csus`, `efus`, `tfus`, `oaus`; its security translations file description gives `CAUS` and `CSUS`. | Verified for CI examples |
| Axys security master can contain duplicate or ambiguous security definitions when the same ticker/symbol appears with different security types or ticker and CUSIP are both used as symbols. | ByAllAccounts Axys CI Guide describes duplicated securities from multiple Axys securities matching the same WebPortfolio security. | Verified for CI matching context |
| Axys Security Type should not be confused with security classifications such as sector/industry/country/asset class. | Public sources show “security type” as product security typing; classification fields are mentioned separately in reporting contexts. | High Confidence |
| Axys reserved security type prefixes `aw`, `br`, `ex`, `ep`, `pi`, `rs` are excluded from ByAllAccounts CI security matching. | ByAllAccounts Axys CI Guide states these prefixes are treated as Axys reserved types and excluded from the security match process. | Verified for CI matching context |
| The same reserved type prefix behavior applies globally inside Axys reporting or IMEX. | Source only discusses CI security matching. | Unknown |

### 3.3 Axys Asset Class Evidence

| Statement | Evidence | Confidence |
|---|---|---|
| A third-party Axys asset import example expects an `Asset Class` field in an Axys export. | AdvisorEngine KB lists an Axys export with fields including `Asset Class`. | Verified for that integration/export workflow |
| `Asset Class` is a common Axys classification/reporting output field. | SS&C Axys product page references asset class grouping and performance display; AdvisorEngine lists Asset Class as an export field. | High Confidence |
| The exact native Axys field name for asset class in the security master is `Asset Class`. | AdvisorEngine uses an exported column header, not necessarily a native database/IMEX/REP field. | Medium Confidence |
| Asset class can be used for holdings/import workflows and not just performance display. | AdvisorEngine import requires Asset Class along with holdings fields. | Medium Confidence |
| The source of `Asset Class` may be security master, security type mapping, custom classification, report calculation, or another lookup. | Not established by public sources. | Unknown |

### 3.4 Axys Classification Storage

| Question | Current Evidence | Confidence |
|---|---|---|
| Are security classifications stored directly on security records? | No authoritative public Axys security master dictionary was found. | Unknown |
| Are classifications stored in separate lookup files? | No public source found in this pass. | Unknown |
| Are sector/industry/country/region maintained by security or by security type? | Axys product page says reporting can display by those categories but does not define storage. | Unknown |
| Can a security have multiple classification schemes simultaneously? | APX report snippet refers to custom classification; Axys product material suggests flexible grouping but does not prove multiple concurrent security classification schemes. | Unknown |
| Are classification values effective-dated/historical? | No public source found. | Unknown |
| Does Axys store historical classification snapshots in performance files? | No public source found. | Unknown |
| Do REP reports use current security master classifications or classifications as of report date? | No public source found. | Unknown |
| Does IMEX export classification lookup tables separately from security master fields? | No public source found. | Unknown |

### 3.5 Axys Processing Behavior / Quirks

| Statement | Evidence | Confidence |
|---|---|---|
| Direct reading/writing of Axys data files is possible for users who understand underlying file formats, but is not best practice due to version-related file format changes. | AdventGuru states Axys users may read/write data files but warns that it is not best practice because formats change between versions. | Verified from consultant source |
| Axys v1.x used open text file structure; Axys v2.x introduced binary file format. | AdventGuru states this version history. | Verified from consultant source |
| Axys v3.x introduced IMEX to import/export CSV, tab, and fixed formats. | AdventGuru states IMEX was introduced as part of the Axys v3.x era. | Verified from consultant source |
| Upgrading Axys v3.7 to v3.8 requires file conversion and some resulting v3.8 files have different file format. | AdventGuru states this. | Verified from consultant source |
| Classification extraction should generally prefer IMEX or REP/report output over direct file parsing. | Derived from AdventGuru's direct-file warning and repository goal to avoid brittle implementation. | High Confidence |
| Classification-dependent reports may change if current classification metadata changes after a historical performance period. | This is plausible in many systems but not verified for Axys. | Unknown |
| Classification changes can affect historical grouping in reports. | Not verified for Axys. Needs controlled test. | Unknown |

---

## 4. APX Classification Research

### 4.1 APX Product-Level / Report-Level Capabilities

| Statement | Evidence | Confidence |
|---|---|---|
| APX has report capabilities documented in an `Advent Portfolio Exchange Reports Guide`. | Publicly indexed `REP_APX.pdf` exists. | Verified |
| At least one APX report can be run to display any custom classification, industry group, or sector. | Search-result snippet for `REP_APX.pdf` states this. | Verified, limited to snippet |
| The same APX report graphically displays equity allocations. | Search-result snippet for `REP_APX.pdf` states this. | Verified, limited to snippet |
| APX supports sector/industry/custom-classification reporting for allocations. | Based on the APX Reports Guide snippet. | High Confidence |
| APX classification reporting is not limited to a fixed vendor sector/industry hierarchy. | The phrase “any custom classification, industry group, or sector” implies flexibility, but full report docs are needed. | Medium Confidence |
| APX can export reports and custom reporting via SSRS, Crystal, SQL tools, Replang, or Report Writer-style tooling. | AdventGuru describes APX users querying APX database via Excel, writing SSRS/Crystal reports, using SQL tools, exporting reports, and modifying Replang reports. | Verified from consultant source |
| APX's SQL database likely makes classification extraction easier than Axys direct files. | AdventGuru says APX users can query the APX database via Excel and other SQL tools. Exact classification table names remain unknown. | High Confidence for general SQL access; Unknown for classification schema |

### 4.2 APX Security Master / Security Type Evidence

| Statement | Evidence | Confidence |
|---|---|---|
| APX security matching in ByAllAccounts Custodial Integrator uses APX Security information from `sec.inf` and `type.inf`. | ByAllAccounts APX CI Guide states this. | Verified for CI integration context |
| The APX security identifier pair includes APX Symbol and APX Security Type. | ByAllAccounts APX CI Guide lists APX Symbol and APX Security Type as target security information in security translations. | Verified for CI integration context |
| APX security type examples include `csus`, `efus`, `adus`, and `epus`. | ByAllAccounts APX CI Guide includes examples. | Verified for CI examples |
| APX Security Type should not be confused with classification fields such as sector/industry/custom classification. | Sources show “security type” as instrument/security master typing; report snippet separately mentions custom classification/industry/sector. | High Confidence |
| APX reserved security type prefixes `aw`, `br`, `ex`, `ep`, `pi`, `rs` are excluded from ByAllAccounts CI security matching. | ByAllAccounts APX CI Guide states these prefixes are treated as APX reserved types and excluded from the security match process. | Verified for CI matching context |
| The same reserved type prefix behavior applies globally inside APX reporting or SQL. | Source only discusses CI security matching. | Unknown |
| APX security master can contain duplicate or ambiguous security definitions when ticker/symbol and CUSIP overlap or same symbol appears with different security types. | ByAllAccounts APX CI Guide describes duplicated securities from multiple APX entries. | Verified for CI matching context |

### 4.3 APX Classification Storage

| Question | Current Evidence | Confidence |
|---|---|---|
| What database tables hold APX classifications? | Not found in public sources in this pass. | Unknown |
| Are APX classifications in the security master, a classification table, or report metadata? | Not verified. | Unknown |
| Are APX custom classifications user-defined in a lookup table? | Likely, given APX report guide snippet, but the storage model is not verified. | Medium Confidence |
| Are APX classifications effective-dated? | Not verified. | Unknown |
| Are APX classification assignments historical or current-only? | Not verified. | Unknown |
| Does APX performance reporting use stored classification snapshots or current classification lookups? | Not verified. | Unknown |
| Does APX maintain compatibility exports in Axys v3 format that include classification fields? | AdventGuru states APX can export data to Axys v3 format, but classification content is not specified. | Unknown |

### 4.4 APX Processing Behavior / Quirks

| Statement | Evidence | Confidence |
|---|---|---|
| APX v1.x through v4.x maintains IMEX functionality. | AdventGuru states this. | Verified from consultant source |
| APX removed fixed-format file generation from IMEX. | AdventGuru states fixed-format generation was eliminated in APX. | Verified from consultant source |
| APX can export data to Axys v3 format. | AdventGuru states this. | Verified from consultant source |
| APX users may query the APX database through Excel and other software. | AdventGuru states this. | Verified from consultant source |
| APX users may write SSRS or Crystal reports to extract data. | AdventGuru states this. | Verified from consultant source |
| APX users may use SQL-based tools to export/import selected data. | AdventGuru states this. | Verified from consultant source |
| Classification extraction in APX could be implemented via SQL/report queries rather than only IMEX. | Derived from AdventGuru's SQL/report access statements. Exact schema unknown. | High Confidence |
| APX classification reports may be affected by current classification definitions. | Plausible, but not verified. | Unknown |

---

## 5. IMEX Research for Classifications

### 5.1 IMEX General Behavior

| Statement | Evidence | Confidence |
|---|---|---|
| IMEX allows Axys users to import/export files in CSV, tab, and fixed formats. | AdventGuru states IMEX was introduced to allow CSV, tab, and fixed import/export. | Verified from consultant source |
| APX maintains IMEX functionality from v1.x to v4.x. | AdventGuru states this. | Verified from consultant source |
| APX eliminated fixed-format generation from IMEX. | AdventGuru states this. | Verified from consultant source |
| IMEX is a preferred integration path over direct Axys file parsing for version-resilient extraction. | Derived from AdventGuru version/file-format warnings. | High Confidence |
| IMEX plus trade blotter can import transaction and label data. | AdventGuru states IMEX plus trade blotter import can move transaction and label data. | Verified from consultant source |
| IMEX can export/import classification data. | Not directly verified. | Unknown |
| IMEX object names for security master/classification values are known. | Not found in public sources in this pass. | Unknown |
| IMEX has a standard object for security reference data, security master, classifications, labels, or groups. | Plausible, but object names and field names require vendor docs or sample `.imx`/export files. | Unknown |

### 5.2 IMEX Classification Extraction Hypotheses Requiring Verification

These are **not facts**. They are proposed test cases for future verification.

| Hypothesis | How to Verify | Confidence |
|---|---|---|
| Security-level classifications can be exported as columns on a security master/security reference IMEX export. | Run IMEX export of security master / security reference data and inspect available fields. | Unknown |
| Portfolio-level classifications/groupings can be exported through a portfolio/master IMEX object. | Run IMEX export of portfolio master / group/category data. | Unknown |
| Classification lookup tables can be exported separately from assigned classification values. | Look for IMEX objects that list class code/name/description/hierarchy/order. | Unknown |
| Report classifications are generated dynamically from security master fields at report time. | Compare report output before/after changing classification metadata in a controlled test database. | Unknown |
| Labels are related to classifications. | Need precise Advent definition of “label” vs “classification” in Axys/APX. | Unknown |

### 5.3 Minimal IMEX Evidence Needed

To convert unknowns into verified statements, collect:

1. IMEX export setup screen or definition file for:
   - security master / security reference
   - portfolio master
   - account/group/category definitions
   - classification lookup tables, if present
2. Sample IMEX exports containing:
   - security identifier
   - security type
   - asset class
   - sector
   - industry
   - country
   - region
   - custom class/group, if configured
3. Any IMEX documentation listing:
   - object names
   - field names
   - data types
   - required fields
   - import vs export support
   - Axys/APX version differences
4. A controlled before/after classification edit test:
   - export security master before edit
   - run classification report before edit
   - change classification
   - export security master after edit
   - rerun historical report
   - compare outputs

---

## 6. REP / Replang Research for Classifications

### 6.1 REP / Report Development Evidence

| Statement | Evidence | Confidence |
|---|---|---|
| Axys and APX users can export reports directly to Excel and can automate report output in XLS and other file formats. | AdventGuru states this. | Verified from consultant source |
| Axys and APX users can create custom reports via Report Writer Pro and convert output to CSV. | AdventGuru states this. | Verified from consultant source |
| Axys and APX users can modify Replang reports to build CSV, other text formats, and Advent file formats. | AdventGuru states this. | Verified from consultant source |
| REP/Replang is a plausible way to extract classification-enriched holdings/performance data. | Derived from report customization/export capabilities. | High Confidence |
| Exact Axys REP report names for classifications are known. | Not verified. | Unknown |
| Exact APX REP report names for classification reports are known. | Only a public APX Reports Guide snippet indicates report capability; report name not captured. | Unknown |
| REP field tokens for classifications are known. | Not found in public sources in this pass. | Unknown |

### 6.2 APX Report Guide Evidence

| Statement | Evidence | Confidence |
|---|---|---|
| A public APX Reports Guide PDF exists at `cdn.advent.com/cms/pdfs/reports/REP_APX.pdf`. | Search result located the PDF. | Verified |
| The APX Reports Guide includes a report that can display “any custom classification, industry group, or sector.” | Search result snippet says this. | Verified, limited to snippet |
| The report graphically displays equity allocations. | Search result snippet says this. | Verified, limited to snippet |
| The report name corresponding to that snippet is known. | The snippet did not include the report name; full reliable extraction was not available in this pass. | Unknown |
| APX report output can be used as a classification source. | If report output includes classification labels and values, yes; but exact report output fields are not verified. | Medium Confidence |

### 6.3 REP Classification Extraction Strategy

Recommended, evidence-conservative approach:

| Approach | Notes | Confidence |
|---|---|---|
| Use REP/report output to extract current holdings by classification when users already run classification reports. | Report output can be exported/automated per AdventGuru. Exact report/field names must be discovered per installation. | High Confidence |
| Use a custom REP/Replang CSV to force stable column layout. | AdventGuru says Replang reports can be modified to build CSV/text formats. Actual field tokens need local report source. | High Confidence |
| Prefer report output when classification logic is report-specific. | If classifications are transformed or grouped inside report logic, the report is the source of truth. Needs report inspection. | Medium Confidence |
| Prefer IMEX/security master if the goal is raw classification assignments. | IMEX likely extracts raw master data, but classification IMEX object/fields are not verified. | Medium Confidence |
| Maintain both raw and report-derived classifications in downstream audit tooling. | Useful because report logic may differ from raw master data; not an Axys/APX fact. | Medium Confidence as design recommendation |

---

## 7. Data Model Research

### 7.1 Concepts to Keep Separate

| Concept | Description | Evidence | Confidence |
|---|---|---|---|
| Security Symbol | Product security identifier, may be ticker or CUSIP-like value depending on setup. | ByAllAccounts Axys/APX CI Guides. | Verified for CI context |
| Security Type | Product security type such as `csus`, `efus`, etc. Used with symbol in matching. | ByAllAccounts Axys/APX CI Guides. | Verified for CI context |
| Asset Class | Reporting/grouping classification seen in product and third-party export material. | SS&C Axys product page; AdvisorEngine export field. | High Confidence |
| Sector | Classification used for performance display/reporting. | SS&C Axys product page; APX report guide snippet. | Verified as reporting category |
| Industry Group | APX report-guide snippet says report can display industry group. | APX Reports Guide snippet. | Verified, limited to snippet |
| Country | Classification used in Axys performance display. | SS&C Axys product page. | Verified as reporting category |
| Region | Classification used in Axys performance display. | SS&C Axys product page. | Verified as reporting category |
| Custom Classification | APX report-guide snippet mentions any custom classification. | APX Reports Guide snippet. | Verified, limited to snippet |
| Portfolio Grouping Category | Axys can group portfolios by manager, asset class, investment objective, or any category. | SS&C Axys product page. | Verified |
| Label | AdventGuru mentions importing transaction and label data through the trade blotter. Relationship to classifications is not established. | AdventGuru. | Verified term exists; classification relationship Unknown |

### 7.2 Candidate Logical Model

This is **not** asserted as native Axys/APX schema. It is a normalized research model for documenting observed exports and reports.

| Logical Table | Purpose | Candidate Keys | Candidate Fields | Confidence |
|---|---|---|---|---|
| `security` | Security identity | `security_id`, `symbol`, `security_type` | symbol, type, name, CUSIP/ticker if exported | Medium Confidence |
| `security_type` | Instrument/security type metadata | `security_type` | type code, type description, reserved prefix flag | Medium Confidence |
| `classification_scheme` | Defines classification family | `scheme_id` | asset class, sector, industry group, country, region, custom scheme name | Medium Confidence |
| `classification_value` | Lookup value within scheme | `scheme_id`, `class_code` | code, name, parent code, sort order | Unknown |
| `security_classification_assignment` | Assigns security to classification value | `security_id`, `scheme_id`, optional date | class code/name, effective date | Unknown |
| `portfolio_classification_assignment` | Assigns portfolio/account to group/category | portfolio code, scheme/category | manager, objective, custom category | Medium Confidence |
| `report_classification_output` | Output of classification-aware report | portfolio, date, security/group/classification | market value, weight, return, allocation | Medium Confidence |

### 7.3 Common Field Dictionary

| Field | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---|
| `Symbol` / `Security Symbol` / `Axys Symbol` / `APX Symbol` | Product security symbol used to identify security; may be ticker or CUSIP-like. | Yes | Yes | Unknown | Likely | Verified for CI context |
| `Type` / `Security Type` / `Axys Security Type` / `APX Security Type` | Product security type code. | Yes | Yes | Unknown | Likely | Verified for CI context |
| `Security` | Security name/description in AdvisorEngine Axys export column list. | Yes | Unknown | Unknown | Likely | Verified for one third-party Axys export |
| `Sec Type Code` | Security type code column in AdvisorEngine Axys export. | Yes | Unknown | Unknown | Likely | Verified for one third-party Axys export |
| `Security Type` | Security type description or code depending on export. | Yes | Yes | Unknown | Likely | Verified for CI/AdvisorEngine contexts |
| `Asset Class` | Classification/grouping field. | Yes | Likely | Unknown | Likely | High Confidence for Axys; Medium Confidence for APX |
| `Sector` | Classification/reporting group. | Yes | Yes | Unknown | Likely | Verified as reporting category |
| `Industry Group` | APX report classification category. | Unknown | Yes | Unknown | Likely | Verified, limited to APX Reports Guide snippet |
| `Industry` | Common classification term, but exact Axys/APX field not verified. | Unknown | Unknown | Unknown | Unknown | Unknown |
| `Country` | Classification/reporting group. | Yes | Likely | Unknown | Likely | Verified for Axys reporting; APX unknown |
| `Region` | Classification/reporting group. | Yes | Likely | Unknown | Likely | Verified for Axys reporting; APX unknown |
| `Custom Classification` | User-defined classification scheme. | Unknown | Yes | Unknown | Likely | Verified for APX report snippet; Axys unknown |
| `Portfolio Name` | Portfolio identity/display field in AdvisorEngine Axys export. | Yes | Unknown | Unknown | Yes | Verified for one third-party Axys export |
| `Portfolio Code` | Portfolio code in AdvisorEngine Axys export. | Yes | Unknown | Unknown | Yes | Verified for one third-party Axys export |
| `Market Value` | Holding valuation field used in classification/allocation outputs. | Yes | Likely | Unknown | Yes | Verified for one third-party Axys export |
| `Quantity` | Holding quantity field. | Yes | Likely | Unknown | Yes | Verified for one third-party Axys export |
| `Weight` | Allocation/percentage field. | Likely | Likely | Unknown | Likely | Medium Confidence |
| `Classification Code` | Possible lookup code for classification value. | Unknown | Unknown | Unknown | Unknown | Unknown |
| `Classification Name` | Possible lookup name for classification value. | Unknown | Unknown | Unknown | Unknown | Unknown |
| `Effective Date` | Date on which classification assignment becomes effective. | Unknown | Unknown | Unknown | Unknown | Unknown |

---

## 8. Examples

### 8.1 Example: Axys Holding Export with Asset Class

A third-party Axys asset import workflow expects an XLS export with the following fields in this exact order:

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

Research interpretation:

| Observation | Confidence |
|---|---|
| `Asset Class` can appear as an Axys report/export column in at least one integration workflow. | Verified |
| The export combines portfolio identity, security identity, valuation, quantity, and classification. | Verified for the documented workflow |
| The workflow does not prove that `Asset Class` is a native security master field. | Verified limitation |
| The workflow does not prove the IMEX field name. | Verified limitation |

### 8.2 Example: Axys Security Translation

From ByAllAccounts CI Axys guide, a security with ticker `LMNVX` can be translated to Axys symbol `524659208` and type `efus`.

```text
WP Ticker: LMNVX
WP Name:   LEGG MASON VLE TR INSTL
Axys Symbol: 524659208
Axys Type:   efus
```

Research interpretation:

| Observation | Confidence |
|---|---|
| Axys security identity in this integration is symbol + type. | Verified |
| The symbol may be a CUSIP-like value rather than the ticker. | Verified in example |
| Classification fields are not part of this specific translation example. | Verified |
| Classification extracts should retain both symbol and type to avoid ambiguity. | High Confidence |

### 8.3 Example: APX Security Translation

From ByAllAccounts CI APX guide, a security with ticker `LMNVX` can be translated to APX symbol `524659208` and type `efus`.

```text
WP Ticker: LMNVX
WP Name:   LEGG MASON VLE TR INSTL
APX Symbol: 524659208
APX Type:   efus
```

Research interpretation:

| Observation | Confidence |
|---|---|
| APX security identity in this integration is symbol + type. | Verified |
| The symbol may be a CUSIP-like value rather than the ticker. | Verified in example |
| APX shares similar external matching behavior with Axys in this CI integration. | Verified for CI context |
| Classification extracts should retain both symbol and type to avoid ambiguity. | High Confidence |

### 8.4 Example: Duplicate Security Quirk

ByAllAccounts CI guides describe duplicate security matching where:

```text
Example A:
Same ticker/symbol appears in multiple product securities with different security types.

Example B:
A security is defined once with ticker as symbol and again with CUSIP as symbol.
```

Research interpretation:

| Observation | Confidence |
|---|---|
| Symbol alone is not always enough to identify a security in Axys/APX integrations. | Verified for CI context |
| Classification joins should avoid joining only on ticker if symbol/type is available. | High Confidence |
| Classification joins may be wrong when ticker and CUSIP variants coexist. | High Confidence |
| A robust downstream schema should store product symbol, product type, name, CUSIP/ticker if available, and source report/export. | High Confidence as implementation recommendation |

### 8.5 Example: APX Custom Classification Report

The APX Reports Guide search snippet states that a report can display:

```text
any custom classification, industry group, or sector
```

and graphically display equity allocations.

Research interpretation:

| Observation | Confidence |
|---|---|
| APX includes report support for custom classifications, industry group, and sector. | Verified, limited to snippet |
| The exact report name must be identified from the full APX Reports Guide or APX report library. | Unknown |
| The report likely outputs or displays allocation by classification. | High Confidence |
| It is not yet known whether the report can export raw classification assignments or only aggregated allocation totals. | Unknown |

---

## 9. Known Issues / Quirks

| Quirk | Axys | APX | Impact | Confidence |
|---|---|---|---|---|
| Symbol alone may be ambiguous. | Yes in CI examples | Yes in CI examples | Use symbol + security type when joining classification data. | Verified for CI context |
| Ticker and CUSIP may both be used as symbols for the same security. | Yes in CI examples | Yes in CI examples | Duplicate or inconsistent classification joins possible. | Verified for CI context |
| Same symbol may exist under different security types. | Yes in CI examples | Yes in CI examples | Classification export must preserve security type. | Verified for CI context |
| Reserved type prefixes excluded from CI matching: `aw`, `br`, `ex`, `ep`, `pi`, `rs`. | Yes in CI docs | Yes in CI docs | Do not assume CI sees all security types. | Verified for CI context |
| Direct Axys file parsing is version-sensitive. | Yes | Less relevant because APX SQL/database access exists, but APX can export Axys v3 format. | Prefer IMEX/REP/report/SQL exports over direct Axys file reads. | Verified from consultant source |
| Axys file formats changed across versions. | Yes | N/A | Version-specific parsers can break. | Verified from consultant source |
| APX IMEX does not generate fixed-format files. | N/A | Yes | Do not design APX classification extraction assuming fixed-format IMEX output. | Verified from consultant source |
| Classification reports may depend on current metadata rather than historical snapshots. | Unknown | Unknown | Critical for historical audit and performance attribution. | Unknown |
| Security Type and Asset Class may be conflated by users. | Likely | Likely | Documentation should explicitly separate instrument type from reporting classification. | High Confidence |
| Third-party export column names may not equal native field names. | Yes | Yes | Avoid treating exported headers as canonical IMEX/DB fields without source. | High Confidence |

---

## 10. Version Differences

| Version / System | Statement | Evidence | Confidence |
|---|---|---|---|
| Axys v1.x | Maintained open text file structure. | AdventGuru. | Verified from consultant source |
| Axys v2.x | First version to implement binary file format. | AdventGuru. | Verified from consultant source |
| Axys v3.x | IMEX introduced, supporting CSV, tab, and fixed formats. | AdventGuru. | Verified from consultant source |
| Axys v3.7 to v3.8 | Upgrade requires file conversion; some v3.8 files have different file format. | AdventGuru. | Verified from consultant source |
| APX v1.x through v4.x | Maintains IMEX functionality. | AdventGuru. | Verified from consultant source |
| APX v1.x through v4.x | Fixed-format generation eliminated. | AdventGuru. | Verified from consultant source |
| APX | Can export data to Axys v3 format. | AdventGuru. | Verified from consultant source |
| Axys/APX classification schema differences by version | Not found. | None. | Unknown |

---

## 11. Implementation Guidance for the Future Chapter

### 11.1 Suggested Chapter Framing

The final chapter should avoid claiming that classifications are one single native object. Based on available evidence, the safest framing is:

1. Axys/APX classifications appear in multiple contexts:
   - security classifications
   - portfolio grouping categories
   - performance/holdings report groupings
   - custom report classifications
   - possibly labels or firm-specific metadata
2. Security identity must be handled separately from classification:
   - symbol
   - security type
   - name
   - CUSIP/ticker when available
3. “Security Type” is not the same as “Asset Class,” “Sector,” “Industry,” or custom classification.
4. For extraction, the repository should document:
   - raw source data path, if using IMEX
   - report source path, if using REP/Replang
   - APX SQL/report path, if using APX database access
   - exact source of classification values
   - whether values are current or historical

### 11.2 Recommended Chapter Sections

Use the standard chapter template:

1. Overview
2. Axys
3. APX
4. IMEX
5. REP
6. Data Model
7. Common Fields
8. Examples
9. Known Issues / Quirks
10. References
11. Unknowns

### 11.3 Suggested Strong Claims for Chapter 11

These can likely be used in the final chapter:

| Claim | Confidence |
|---|---|
| Axys can report performance by asset classes, sectors, countries, or regions. | Verified |
| Axys can group portfolios by manager, asset class, investment objective, or a user-chosen category. | Verified |
| APX has report support for custom classification, industry group, and sector. | Verified, limited to APX Reports Guide snippet |
| Axys/APX security identification in integration contexts should preserve both symbol and security type. | Verified for CI context |
| Direct Axys file parsing is not recommended for stable classification extraction because file formats vary by version. | Verified from consultant source |
| IMEX and REP/report output are safer extraction points than direct Axys file reads. | High Confidence |
| APX offers SQL/report extraction options beyond IMEX. | Verified from consultant source |
| The exact native classification field names and table/file storage require additional source material. | Verified limitation |

### 11.4 Claims to Avoid or Mark Unknown

Do **not** state any of the following as fact without additional evidence:

| Unsupported Claim | Correct Treatment |
|---|---|
| Axys stores sector/industry/country directly in `sec.inf`. | Unknown |
| APX stores classifications in a specific SQL table. | Unknown |
| IMEX object `X` exports classifications. | Unknown unless vendor docs/sample IMEX provided |
| REP field token `X` returns asset class/sector/industry. | Unknown unless report source provided |
| Axys/APX classifications are historical/effective-dated. | Unknown |
| Historical reports use classification values as of report date. | Unknown |
| Historical reports use current classification values. | Unknown |
| Asset Class is always derived from Security Type. | Unknown |
| Sector/industry hierarchy follows GICS, BICS, SIC, or proprietary taxonomy. | Unknown |
| APX and Axys use identical classification fields. | Unknown |
| “Label” means classification. | Unknown |

---

## 12. Research Questions / Unknowns

### 12.1 Highest Priority Unknowns

| Priority | Question | Why It Matters | Needed Evidence |
|---:|---|---|---|
| 1 | Where are security classifications stored in Axys? | Determines correct extraction and audit strategy. | Axys security master docs, IMEX object docs, sample exports, or REP source. |
| 1 | Where are security classifications stored in APX? | Determines SQL/IMEX/report extraction strategy. | APX database dictionary, APX report source, IMEX docs. |
| 1 | What are the exact IMEX objects and field names for classifications? | Required for implementation. | IMEX documentation or exported setup definitions. |
| 1 | What are the exact REP/Replang field tokens for asset class/sector/industry/country/region/custom classification? | Required for stable report-based extract. | REP source files or report writer docs. |
| 1 | Are classifications current-only or historical/effective-dated? | Critical for historical performance and audit. | Controlled test or vendor documentation. |
| 2 | Does performance reporting use classification snapshot or current classification lookup? | Historical reporting can change if classifications are restated. | Before/after test. |
| 2 | Are classifications security-level, portfolio-level, or both? | Avoids mixing account groupings with security classifications. | Security master and portfolio master exports. |
| 2 | Can a security belong to multiple classification schemes? | Needed for custom classifications. | APX/Axys configuration screenshots/docs. |
| 2 | Are sector/industry/country/region vendor taxonomies or firm-defined? | Needed for downstream semantic interpretation. | Source system configuration/docs. |
| 2 | Does APX export Axys v3 format include classification fields? | Important migration/integration path. | APX export sample. |

### 12.2 Evidence to Request from User / Client

Ask for any of the following, in order of usefulness:

1. Sample Axys IMEX export of security master / security reference data.
2. Sample APX IMEX export of security master / security reference data.
3. Any IMEX `.ini`, definition, mapping, or object selection screenshots.
4. REP/Replang source for:
   - holdings by asset class
   - performance by asset class
   - sector/industry allocation
   - custom classification reports
5. APX Reports Guide full PDF or screenshots around the custom classification report.
6. APX SQL data dictionary or read-only query examples for classification tables.
7. A small anonymized export with:
   - symbol
   - type
   - name
   - CUSIP
   - asset class
   - sector
   - industry
   - country
   - region
   - custom classification
8. Before/after test evidence:
   - classification before edit
   - historical report before edit
   - classification after edit
   - same historical report after edit

---

## 13. Suggested Test Plan

### Test 1: Determine Whether Axys Historical Reports Use Current Classifications

| Step | Action |
|---:|---|
| 1 | Choose a test security held in a closed historical month. |
| 2 | Record its current asset class/sector/industry/country/region. |
| 3 | Run a historical holdings or performance-by-classification report for that month. |
| 4 | Change the security classification in a test copy of Axys. |
| 5 | Rerun the same historical report. |
| 6 | Compare whether the old month appears under the old or new classification. |
| 7 | Restore data or use test environment only. |

Expected outcomes:

| Outcome | Interpretation |
|---|---|
| Historical report changes classification | Report uses current classification lookup. |
| Historical report does not change classification | Classification may be stored historically or in performance/report snapshots. |
| Only some reports change | Classification behavior is report-specific. |

### Test 2: Determine Whether APX Historical Reports Use Current Classifications

Same as Test 1, but perform in APX test environment. If APX SQL tables are accessible, also query classification assignments before and after edit.

### Test 3: Compare IMEX vs REP Output

| Step | Action |
|---:|---|
| 1 | Export security master/classification data through IMEX. |
| 2 | Export the same portfolio/date through a REP/Replang classification report. |
| 3 | Compare classification values by symbol/type. |
| 4 | Identify report-derived overrides or rollups. |

### Test 4: Identify Classification Hierarchy

| Step | Action |
|---:|---|
| 1 | Export or report asset class, sector, industry, country, region. |
| 2 | Determine whether values form parent/child hierarchy or independent attributes. |
| 3 | Check whether custom classifications are flat or hierarchical. |
| 4 | Confirm sort order and display names. |

---

## 14. References

### 14.1 Governing Repository Specification

- `AXYS_APX_REFERENCE_BLUEPRINT.md`, Version 2.0. User-supplied blueprint for repository standards.

### 14.2 Web References

- SS&C Advent Axys product page: `https://www.advent.com/solutions/axys/`
  - Supports: Axys predefined reports, report customization, Report Writer Pro, portfolio grouping, performance display by asset class/sector/country/region.
- AdventGuru Integration article/tag page: `https://adventguru.com/tag/integration/`
  - Supports: Axys version history, IMEX introduction, APX IMEX behavior, APX SQL/report extraction options, Replang/custom report export options, direct-file access warning.
- APX Reports Guide PDF: `https://cdn.advent.com/cms/pdfs/reports/REP_APX.pdf`
  - Supports: APX report capability for custom classification/industry group/sector based on public search snippet. Full reliable text extraction should be obtained for final chapter.
- ByAllAccounts Custodial Integrator User Guide for Axys: `https://www.byallaccounts.net/Manuals/Custodial_Integrator/axys/CI_User_Guide.pdf`
  - Supports: Axys `sec.inf` and `type.inf`, Axys Symbol, Axys Security Type, security matching behavior, duplicate securities, reserved type prefixes, security translation file fields.
- ByAllAccounts Custodial Integrator User Guide for APX: `https://www.byallaccounts.net/Manuals/Custodial_Integrator/apx/CI_User_Guide.pdf`
  - Supports: APX `sec.inf` and `type.inf`, APX Symbol, APX Security Type, security matching behavior, duplicate securities, reserved type prefixes.
- AdvisorEngine Advent Axys Asset Import KB: `https://support.advisorengine.com/portal/en/kb/articles/5019002001`
  - Supports: example Axys export fields including `Asset Class`.
- Salentica Elements Data Broker Axys/APX KB: `https://elements.salentica.com/kb/article/252-data-broker-ss-c-advent-apx-axys/`
  - Supports: Axys/APX as on-prem systems sending data to Data Broker.

---

## 15. Final Notes for Chapter Author

The final `11-Classifications.md` should be conservative.

Recommended tone:

- “Axys supports classification-based reporting” is safe.
- “APX has at least one report supporting custom classification, industry group, or sector” is safe, but cite the APX Reports Guide and ideally inspect the PDF text directly.
- “Security Type is not the same thing as Asset Class/Sector/Industry” should be emphasized.
- “Exact native field names are not yet verified” should be preserved until actual IMEX/REP/database evidence is supplied.
- Avoid creating a fake data dictionary. Use the field dictionary above as a research inventory, not a vendor schema.

