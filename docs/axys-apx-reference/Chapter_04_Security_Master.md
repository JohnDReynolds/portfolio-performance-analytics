# Chapter 04 — Security Master

> Repository chapter expanded from the supplied chapter, research notes, independent research addenda, and repository blueprint. This chapter treats the research files as raw evidence and reorganizes the supported facts into a technical reference. Unsupported items are marked **Unknown**. Integration-specific behavior is identified as such and is not promoted to native Axys/APX behavior unless the supplied research supports that conclusion.

---

## Related chapters
- [Chapter_01_Overview.md](Chapter_01_Overview.md) — provides the repository-wide map and evidence conventions.
- [Chapter_05_Transactions.md](Chapter_05_Transactions.md) — security identity is essential for transaction translation and posting.
- [Chapter_11_Classifications.md](Chapter_11_Classifications.md) — distinguishes security identity from classification and grouping data.
- [Chapter_15_Data_Dictionary.md](Chapter_15_Data_Dictionary.md) — consolidates the security-related fields and tokens.

## 1. Overview

The Security Master is the reference dataset used to identify and describe securities used in Axys and APX workflows. The supplied research verifies that both Axys and APX have a security-master concept and that third-party tools rely on security information for matching, translation, import, export, pricing, positions, transactions, and downstream reporting.

The available material does **not** provide a complete native Axys or APX security-master schema. The strongest verified evidence comes from integration and conversion documentation, especially Custodial Integrator, AIA, AdvisorEngine, Morningstar conversion documentation, Salentica/Elements Data Broker documentation, and AdventGuru practitioner notes.

| Topic | Axys | APX | Confidence | Notes |
|---|---:|---:|---:|---|
| Distinct security master / security information concept | Yes | Yes | Verified | Multiple integration sources refer to Axys/APX security information, security master, or security master file workflows. |
| Security information used for import matching | Yes | Yes | Verified | Custodial Integrator and AIA both rely on security information during import workflows. |
| Symbol + security type used operationally for matching/translation | Yes | Yes | Verified in integration context | CI uses product-specific symbol and type in translation and missing-price workflows. |
| Formal native primary key | Unknown | Unknown | Unknown | Symbol + type is operationally important, but the formal native primary key is not established. |
| Complete native field dictionary | Unknown | Unknown | Unknown | Supplied sources provide integration/export labels, not full vendor schema. |
| Security (`sec.inf`) and Security Type (`type.inf`) exported by import/export utilities | Yes | Yes | Verified in CI context | Axys uses `imex32.exe`; APX uses `apxix.exe` in the CI installation guides. |
| Exact `sec.inf` / `type.inf` field layouts | Unknown | Unknown | Unknown | Public evidence confirms use/export, but not complete layout. |
| REP security-master report name | Unknown | Unknown | Unknown | REP32 extraction is verified for connectors, but no standard security-master report name was supplied. |
| APX public-view security schema | N/A | Unknown | Unknown | APX public views exist, but security-specific view names and field lists were not established. |

### 1.1 Evidence Classification

This chapter uses the confidence labels required by the repository blueprint.

| Label | Meaning in this chapter |
|---|---|
| Verified | Directly supported by the supplied research notes or source excerpts summarized there. |
| High Confidence | Strongly supported by vendor-adjacent, connector, conversion, or practitioner evidence, but not complete official schema documentation. |
| Medium Confidence | Plausible and operationally useful, but source evidence is indirect, practitioner-specific, connector-specific, or not fully separated between Axys and APX. |
| Unknown | Not established by the supplied source material. Do not implement as fact without additional evidence. |

### 1.2 Scope Boundaries

This chapter documents only what the supplied material supports. It does not attempt to reconstruct proprietary Advent/SS&C schemas, hidden database tables, undocumented REP field names, or native IMEX object definitions.

| Item | Treatment |
|---|---|
| Integration files such as CI `SECTRANSLATIONS_yyyymmdd.csv` | Documented as integration artifacts, not native security-master schemas. |
| AIA `.veh` files | Documented as AIA-generated files using `sec.inf` layout, not native field dictionaries. |
| AdvisorEngine XLS export labels | Documented as export/report labels, not native field names. |
| Morningstar conversion guidance | Documented as conversion-context evidence for Axys `sec.inf`, not complete Advent schema documentation. |
| AdventGuru practitioner notes | Used for implementation cautions; not treated as official vendor documentation. |
| Connector version support | Treated as connector-specific, not product-wide compatibility. |

---

## 2. Security Identity Model

The supplied material supports a practical identity model used by integration workflows: securities are often identified by a product-specific symbol and a security type. External matching may also use ticker, CUSIP, security name, financial institution, and account number.

### 2.1 Supported Identifier Concepts

| Identifier / Label | Axys | APX | Description | Confidence | Scope |
|---|---:|---:|---|---:|---|
| Symbol | Yes | Yes | Product-specific security symbol used by CI to identify securities. | Verified | Integration/security matching context. |
| Type / Security Type | Yes | Yes | Product-specific security type used with symbol in CI matching and translation. | Verified | Integration/security matching context. |
| Axys Symbol | Yes | No | CI target symbol for Axys security translation. | Verified | Axys CI translation file. |
| APX Symbol | No | Yes | CI target symbol for APX security translation. | Verified | APX CI translation file. |
| APX Security Type / APX Type | No | Yes | CI target APX security type; required with symbol for imported APX positions and transactions in CI. | Verified | APX CI translation file, examples, and import context. |
| Ticker | Matching context | Matching context | External identifier used for security matching/translations. | Verified | WebPortfolio / CI context. |
| CUSIP | Matching context | Matching context | External/security identifier used for matching and duplicate-resolution workflows. | Verified | WebPortfolio / CI / AIA context. |
| Security Name / WP Name | Matching context | Matching context | External security name used in translation files. | Verified | CI context. |
| Financial Institution / Institution | Matching context | Matching context | Used with external security information in CI translation files. | Verified | CI context. |
| WP Account # / Account # | Matching context | Matching context | Used for account-specific security translations. | Verified | CI context. |

### 2.2 Symbol + Type

In the supplied research, `Symbol` plus `Type` is the clearest recurring operational identifier pair for both Axys and APX integration workflows.

| Statement | Axys | APX | Confidence | Important Limitation |
|---|---:|---:|---:|---|
| CI must determine product security type and product security symbol for imported positions and transactions. | Yes | Yes | Verified | CI behavior; not proof of native database primary key. |
| CI security translations map external identifiers to product symbol and type. | Yes | Yes | Verified | Translation artifact, not native schema. |
| CI missing-prices output uses `Symbol` and `Type`. | Yes | Yes | Verified | Output file from CI, not security master. |
| Same symbol can be associated with different security types. | Yes | Yes | Verified | Duplicate/security matching example context. |
| Symbol alone is not safe as a unique cross-system key. | Yes | Yes | High Confidence | Supported by duplicate-symbol examples. |
| Symbol + type is the formal native primary key. | Unknown | Unknown | Unknown | Not established by the supplied material. |

### 2.3 External Identifier Matching

External matching may use ticker, CUSIP, name, institution, and account number. The research supports this primarily through CI and AIA workflows.

| External Identifier | Use | Axys | APX | Confidence |
|---|---|---:|---:|---:|
| Ticker | Matching external/security records to product security symbol. | Yes | Yes | Verified |
| CUSIP | Matching external/security records to product security symbol or existing security master records. | Yes | Yes | Verified |
| Security Name / WP Name | Used in CI security translation files. | Yes | Yes | Verified |
| Institution | Used in CI security translation files. | Yes | Yes | Verified |
| WP Account # | Used for account-specific translations. | Yes | Yes | Verified |

### 2.4 Primary Key Status

| Question | Axys | APX | Answer |
|---|---:|---:|---|
| Is symbol used in security matching? | Yes | Yes | Verified. |
| Is security type used in security matching? | Yes | Yes | Verified. |
| Is symbol + type operationally important? | Yes | Yes | Verified in integration context. |
| Is symbol + type the formal native primary key? | Unknown | Unknown | Not established. |
| Is there a separate APX SQL identity field? | N/A | Unknown | Not established. |
| Is there a complete Axys native file key definition? | Unknown | N/A | Not established. |

---

## 3. Axys Security Master

### 3.1 Axys Summary

The supplied research verifies that Axys maintains security information used by integrations and reporting/export workflows. Axys CI documentation refers to Axys Security Information and Security Type Information. CI uses Axys symbol and type to generate/import positions, prices, and transactions. AIA documentation indicates that vehicle/security information can be transformed into a `sec.inf` layout and imported into Axys.

| Topic | Axys Status | Confidence | Notes |
|---|---|---:|---|
| Security information exists | Yes | Verified | Integration sources refer to Axys Security Information and Axys securities. |
| Security type information exists | Yes | Verified | CI uses Axys Security Type Information. |
| `sec.inf` used for security information | Yes, in CI/AIA/conversion context | Verified in context | Complete layout Unknown. |
| `type.inf` used for security type information | Yes, in CI context | Verified in context | Complete layout Unknown. |
| Axys Import/Export executable | `imex32.exe` | Verified | CI installation guide context. |
| Axys Post Positions utility | `pospos32.exe` | Verified | CI installation guide context. |
| Native security-master field dictionary | Unknown | Unknown | Not supplied. |
| Native file/storage model | Unknown | Unknown | `sec.inf` is evidenced, but not complete storage model. |

### 3.2 Axys Folder and Utility Evidence

The Axys CI documentation provides verified folder and executable names used in the integration context.

| Item | Description | Confidence | Scope |
|---|---|---:|---|
| `imex32.exe` | Axys Import/Export utility used by CI. | Verified | CI installation/import-export context. |
| `pospos32.exe` | Axys Post Positions utility. | Verified | CI position import context. |
| `$pathexe` | Axys executable folder label in CI. | Verified | CI configuration. |
| `$pathtrn` | Axys user folder label in CI; transactions delivered to `topost.trn`. | Verified | CI configuration. |
| `$pathcli` | Axys client folder label in CI. | Verified | CI configuration. |
| `$pathinf` | Axys information folder label in CI; example path `C:\axys\inf\`. | Verified | CI configuration. |
| `$pathpri` | Axys price folder label in CI. | Verified | CI configuration. |
| `$pathlog` | Axys log folder label in CI. | Verified | CI configuration. |
| `topost.trn` | Axys Trade Blotter file referenced by CI for transactions. | Verified | Transaction import workflow, not security master. |

### 3.3 Axys `sec.inf` and `type.inf`

The research supports a careful statement: Axys security and security type data can be exported by CI using Axys Import/Export, and AIA can generate/import files using `sec.inf` layout. The complete field layout is not supplied.

| File / Layout | Axys Evidence | Confidence | Limitations |
|---|---|---:|---|
| `sec.inf` | CI installation guide states `imex32.exe` exports Security (`sec.inf`) data from Axys. | Verified in CI context | Complete field layout Unknown. |
| `type.inf` | CI installation guide states `imex32.exe` exports Security Type (`type.inf`) data from Axys. | Verified in CI context | Complete field layout Unknown. |
| `.veh` file using `sec.inf` layout | WealthTechs AIA Axys manual states AIA creates a vehicle file with the layout of `sec.inf`. | Verified in AIA context | AIA artifact, not native field dictionary. |
| Axys `sec.inf` in conversion | Morningstar conversion guidance states Axys `sec.inf` can support conversion of user-defined security names and selected accrued-interest fields. | Verified in conversion context | Exact field names Unknown. |

### 3.4 Axys Security Matching Behavior

| Behavior | Axys | Confidence | Notes |
|---|---:|---:|---|
| CI must determine Axys security type and Axys security symbol for imported positions and transactions. | Yes | Verified | CI context. |
| CI can identify securities that it cannot translate. | Yes | Verified | Remediation requires defining the security in Axys or providing a CI security translation. |
| Duplicate matches can occur when more than one Axys security uses the same symbol with different security types. | Yes | Verified | CI duplicate-resolution context. |
| Duplicate matches can occur when one security is defined once by ticker and once by CUSIP. | Yes | Verified | CI duplicate-resolution context. |
| Duplicate matches can occur when multiple CI translations match the same security. | Yes | Verified | CI translation context. |
| Account-specific translations are supported. | Yes | Verified | CI context. |
| Once account-specific translations are used for a security, additional account-specific translations may be required and a global translation cannot be established for that security. | Yes | Verified | CI context. |

### 3.5 Axys Export Label Evidence

AdvisorEngine documentation requires an Advent Axys XLS asset import with these exact columns. These are verified export/report labels, not proven native field names.

| Column Order | Label | Security-Master Relevance | Confidence | Caveat |
|---:|---|---|---:|---|
| 1 | `Portfolio Name` | Portfolio context, not security master. | Verified | Export label only. |
| 2 | `Portfolio Code` | Portfolio context, not security master. | Verified | Export label only. |
| 3 | `Security` | Security label/name in export. | Verified | Not proven native field. |
| 4 | `Sec Type Code` | Security type code in export. | Verified | Not proven native field. |
| 5 | `Security Symbol` | Security symbol in export. | Verified | Not proven native field. |
| 6 | `Security Type` | Security type label in export. | Verified | Not proven native field. |
| 7 | `Market Value` | Position/valuation value, not security master. | Verified | Export context. |
| 8 | `Quantity` | Position/holding value, not security master. | Verified | Export context. |
| 9 | `Asset Class` | Asset-class label in export. | Verified | Not proven native security-master field. |

### 3.6 Axys AIA Vehicle Import Behavior

WealthTechs AIA documentation provides integration-specific import settings. These should be documented as AIA behavior, not native Axys schema.

| AIA Topic | Axys Behavior | Confidence | Scope |
|---|---|---:|---|
| Vehicle/security file creation | AIA creates a vehicle file with the layout of `sec.inf` from custodian information. | Verified | AIA context. |
| Archive file | AIA saves the `.veh` file in the Archive folder. | Verified | AIA context. |
| Import option: `Update Existing & Add New` | Available in AIA vehicle import settings. | Verified | AIA context. |
| Import option: `Add New` | Available in AIA vehicle import settings. | Verified | AIA context. |
| Import option: `Replace Entire File` | Available in AIA vehicle import settings. | Verified | AIA context. |
| Import option: `None` | Available in AIA vehicle import settings; when selected, AIA does not import the file. | Verified | AIA context. |
| Create security information for missing custodian `.veh` records | Supported by AIA according to the supplied research. | Verified | AIA context. |

### 3.7 Axys Conversion Notes

Morningstar conversion documentation provides evidence about Axys `sec.inf` in conversion workflows.

| Conversion Finding | Confidence | Limitation |
|---|---:|---|
| If Advent Axys `sec.inf` is provided, security names for user-defined securities can be converted. | Verified in conversion context | Does not provide field layout. |
| Fields used to calculate accrued interest can be converted if specifically selected/exported through the Axys security file. | Verified in conversion context | Exact fields Unknown. |
| If the Axys `Use Security Type` box is selected, datapoints needed for accrued-interest calculation may not be exported to the Advent Axys security file. | Verified in conversion context | Conversion-specific warning. |

---

## 4. APX Security Master

### 4.1 APX Summary

The supplied research verifies that APX has centralized security information used by import/export utilities and integration products. CI documentation uses APX Security Information and Security Type Information, and maps external identifiers to APX Symbol and Type. AIA documentation states that APX vehicle/security information can be transformed into a `sec.inf` layout. Practitioner research also indicates APX users may access data through Public Views, Stored Accounting Functions, SSRS, REST API, and related tooling, but the supplied material does not establish security-specific APX public view names or field lists.

| Topic | APX Status | Confidence | Notes |
|---|---|---:|---|
| Security information exists | Yes | Verified | CI, AIA, and product-level research. |
| Security type information exists | Yes | Verified | CI uses APX Security Type Information. |
| `sec.inf` used for security information | Yes, in CI/AIA context | Verified in context | Complete layout Unknown. |
| `type.inf` used for security type information | Yes, in CI context | Verified in context | Complete layout Unknown. |
| APX Import/Export executable | `apxix.exe` / `APXIX.exe`; `ApxIx` also appears as terminology | Verified / Unknown naming relationship | Relationship between names is not established. |
| Native SQL table/view names | Unknown | Unknown | Public view names and fields not supplied. |
| Complete native field dictionary | Unknown | Unknown | Not supplied. |
| Public views | Exist generally, but limited | Medium Confidence | Security-specific coverage Unknown. |

### 4.2 APX Import/Export Utility Evidence

| Item | Description | Confidence | Scope |
|---|---|---:|---|
| `apxix.exe` | APX Import/Export utility used by CI to export Security (`sec.inf`) and Security Type (`type.inf`) data from APX. | Verified | CI installation context. |
| `APXIX.exe` | APX import/export function named in WealthTechs AIA manual. | Verified | AIA context. |
| `ApxIx` | APX Import/Export terminology in ByAllAccounts APX guide. | Verified | CI terminology. |
| Relationship between `apxix.exe`, `APXIX.exe`, and `ApxIx` | Unknown | Unknown | The supplied research does not establish whether these are identical labels, version-specific names, case differences, or context-specific terminology. |

### 4.3 APX `sec.inf` and `type.inf`

| File / Layout | APX Evidence | Confidence | Limitations |
|---|---|---:|---|
| `sec.inf` | CI installation guide states `apxix.exe` exports Security (`sec.inf`) data from APX. | Verified in CI context | Complete field layout Unknown. |
| `type.inf` | CI installation guide states `apxix.exe` exports Security Type (`type.inf`) data from APX. | Verified in CI context | Complete field layout Unknown. |
| `.veh` file using `sec.inf` layout | WealthTechs AIA APX manual states AIA translates a `.veh` file to the layout of `sec.inf`. | Verified in AIA context | AIA artifact, not native field dictionary. |
| Fixed-income / accrual-related `sec.inf` fields | Morningstar conversion evidence suggests selected Axys `sec.inf` exports may include such fields. | Medium Confidence | Conversion context only; exact field names and availability remain Unknown. |

### 4.4 APX Security Matching Behavior

| Behavior | APX | Confidence | Notes |
|---|---:|---:|---|
| CI must determine APX security type and APX security symbol for all imported positions and transactions. | Yes | Verified | CI context. |
| CI can fail matching when the security is not in the APX security master. | Yes | Verified | CI context. |
| CI can fail matching when insufficient identifier information is available. | Yes | Verified | CI context. |
| CI can fail matching when more than one APX security matches. | Yes | Verified | CI context. |
| CI Security Translations use external identifiers including ticker, CUSIP, financial institution, and security name. | Yes | Verified | CI context. |
| CI target APX security information includes APX Symbol and APX Security Type. | Yes | Verified | CI context. |
| Security translations take precedence over other security matches. | Yes | Verified | CI context. |
| External ticker or CUSIP may directly match APX symbol when no translation matches first. | Yes | Verified | CI context. |
| CI does not modify APX Security Type or Security Information as part of security translation. | Yes | Verified | CI context. |

### 4.5 APX Reserved Type-Prefix Exclusions in CI

The APX CI guide identifies reserved security-type prefixes that CI excludes from its security match process.

| Prefix | APX CI Treatment | Confidence | Caveat |
|---|---|---:|---|
| `aw` | Excluded from CI security match process. | Verified in CI context | Do not generalize as platform-wide APX rule. |
| `br` | Excluded from CI security match process. | Verified in CI context | Do not generalize as platform-wide APX rule. |
| `ex` | Excluded from CI security match process. | Verified in CI context | Do not generalize as platform-wide APX rule. |
| `ep` | Excluded from CI security match process. | Verified in CI context | Do not generalize as platform-wide APX rule. |
| `pi` | Excluded from CI security match process. | Verified in CI context | Do not generalize as platform-wide APX rule. |
| `rs` | Excluded from CI security match process. | Verified in CI context | Do not generalize as platform-wide APX rule. |

### 4.6 APX AIA Vehicle Import Behavior

| AIA Topic | APX Behavior | Confidence | Scope |
|---|---|---:|---|
| Vehicle/security file creation | AIA gathers custodian security information and translates the `.veh` file to `sec.inf` layout. | Verified | AIA context. |
| Archive file | AIA places the `sec.inf`-layout `.veh` file in the AIA Archive folder for viewing after the Daily Process. | Verified | AIA context. |
| Create security information for missing custodian `.veh` records | Supported by AIA according to the supplied research. | Verified | AIA context. |
| Create security information for existing records | Supported by AIA according to the supplied research. | Verified | AIA context. |
| Import option: `Update Existing & Add New` | Available in AIA vehicle import settings. | Verified | AIA context. |
| Import option: `Add New` | Available in AIA vehicle import settings. | Verified | AIA context. |
| Import option: `Replace Entire File` | Available in AIA vehicle import settings. | Verified | AIA context. |
| Import option: `None` | Available in AIA vehicle import settings. | Verified | AIA context. |

### 4.7 APX Public Views / SQL / REST Access

The supplied research includes practitioner evidence that APX users may access APX data through database/reporting mechanisms, but it does not identify security-master view names or fields.

| Access Mechanism | APX Status | Confidence | Security-Master Coverage |
|---|---|---:|---|
| Public Views | Exist, but limited. | Medium Confidence | Unknown. |
| Stored Accounting Functions | Mentioned by practitioner source. | Medium Confidence | Unknown. |
| SSRS | Mentioned by practitioner source. | Medium Confidence | Unknown. |
| REST API | Mentioned by practitioner source. | Medium Confidence | Unknown. |
| Direct APX SQL Server access | Practitioner source discusses APX SQL/reporting infrastructure. | Medium Confidence | Security-specific schema Unknown. |

### 4.8 APX Authentication / Logging Quirk

| Quirk | Confidence | Scope |
|---|---:|---|
| APX authentication can affect `apxix.exe`; CI may appear to wait when APX refuses access, and diagnosis may require reviewing the Apxix log file. | Verified | CI/APX import-export context. |

---

## 5. IMEX / Import-Export

### 5.1 IMEX Summary

The supplied material verifies that Axys and APX import/export utilities participate in security workflows, including export of Security (`sec.inf`) and Security Type (`type.inf`) data in the CI context. The material does not establish official native IMEX object names, complete security import schemas, or complete field layouts.

| Topic | Axys | APX | Confidence | Notes |
|---|---:|---:|---:|---|
| Import/export utility identified | `imex32.exe` | `apxix.exe` / `APXIX.exe` / `ApxIx` | Verified / naming relationship Unknown | Context differs by source. |
| Security export to `sec.inf` | Yes | Yes | Verified in CI context | Complete layout Unknown. |
| Security Type export to `type.inf` | Yes | Yes | Verified in CI context | Complete layout Unknown. |
| Transaction import through import/export utility | Yes | Yes | Verified in CI context | See [Chapter_05_Transactions.md](Chapter_05_Transactions.md). |
| Price import through import/export utility | Yes | Yes | Verified in CI context | See [Chapter_08_Pricing.md](Chapter_08_Pricing.md). |
| Position import through import/export utility | Yes | Yes | Verified in CI context | See [Chapter_06_Holdings.md](Chapter_06_Holdings.md). |
| Native security IMEX object name | Unknown | Unknown | Unknown | Not supplied. |
| Native security type IMEX object name | Unknown | Unknown | Unknown | Not supplied. |
| Required fields for security import/update | Unknown | Unknown | Unknown | Not supplied. |
| Security type import through IMEX | Medium Confidence caution | Medium Confidence caution | Medium Confidence | AdventGuru states the author does not believe security types can be imported through IMEX; not vendor-confirmed. |

### 5.2 Axys Import/Export Workflow Evidence

| Workflow Step / Artifact | Axys Evidence | Confidence | Scope |
|---|---|---:|---|
| Export Security data | CI uses `imex32.exe` to export Security (`sec.inf`) data from Axys. | Verified | CI installation context. |
| Export Security Type data | CI uses `imex32.exe` to export Security Type (`type.inf`) data from Axys. | Verified | CI installation context. |
| Generate transactions using Axys symbols and types | Exported data enables CI to generate transactions using symbols/types defined in Axys. | Verified | CI context. |
| Generate positions using Axys symbols and types | Exported data enables CI to generate positions using symbols/types defined in Axys. | Verified | CI context. |
| Generate prices using Axys symbols and types | Exported data enables CI to generate prices using symbols/types defined in Axys. | Verified | CI context. |
| Import transactions | CI uses Axys Import/Export to import transactions into a designated Trade Blotter. | Verified | CI context. |
| Import prices | CI uses Axys Import/Export to import prices into Axys `.pri` price file. | Verified | CI context. |
| Import positions | CI uses Axys Import/Export and `pospos32.exe` to import positions into a temporary Trade Blotter and optionally into the Axys Position Blotter. | Verified | CI context. |

### 5.3 APX Import/Export Workflow Evidence

| Workflow Step / Artifact | APX Evidence | Confidence | Scope |
|---|---|---:|---|
| Export Security data | CI uses `apxix.exe` to export Security (`sec.inf`) data from APX. | Verified | CI installation context. |
| Export Security Type data | CI uses `apxix.exe` to export Security Type (`type.inf`) data from APX. | Verified | CI installation context. |
| Generate transactions using APX symbols and types | Exported data enables CI to generate transactions using symbols/types defined in APX. | Verified | CI context. |
| Generate positions using APX symbols and types | Exported data enables CI to generate positions using symbols/types defined in APX. | Verified | CI context. |
| Generate prices using APX symbols and types | Exported data enables CI to generate prices using symbols/types defined in APX. | Verified | CI context. |
| Import transactions | CI uses APX Import/Export to import transactions into a designated Trade Blotter. | Verified | CI context. |
| Import prices | CI uses APX Import/Export to import prices into APX for a specified date. | Verified | CI context. |
| Import positions | CI uses APX Import/Export to import positions into a Position Blotter. | Verified | CI context. |
| Import position lots | CI uses APX Import/Export to import position lots into a Position Lot Blotter if enabled. | Verified | CI context. |

### 5.4 `sec.inf` / `type.inf` IMEX Status

| Question | Axys | APX | Answer |
|---|---:|---:|---|
| Can Security data be exported to `sec.inf` by the product import/export utility in CI context? | Yes | Yes | Verified. |
| Can Security Type data be exported to `type.inf` by the product import/export utility in CI context? | Yes | Yes | Verified. |
| Is `sec.inf` the complete native security-master storage model? | Unknown | Unknown | Not established. |
| Is `type.inf` the complete native security-type storage model? | Unknown | Unknown | Not established. |
| Are the official IMEX object names known? | Unknown | Unknown | Not supplied. |
| Are required import fields known? | Unknown | Unknown | Not supplied. |
| Are the layouts identical between Axys and APX? | Unknown | Unknown | Not established. |
| Are security-translation files, AIA `.veh` files, AdvisorEngine XLS labels, and CRM connector fields native security schemas? | No evidence | No evidence | Treat as integration/report artifacts unless separately verified. |

---

## 6. REP / Report Writer / Replang

### 6.1 REP Summary

REP32, standard reports, macros, and RepLang are verified extraction mechanisms used by at least one Axys/APX connector. The supplied research does not identify a standard security-master report name or prove complete REP field exposure for security-master data.

| Topic | Axys | APX | Confidence | Notes |
|---|---:|---:|---:|---|
| REP32 used by connector | Yes | Yes | Verified for connector | Salentica/Elements Data Broker documentation. |
| Advent standard reports/macros used by connector | Yes | Yes | Verified for connector | Connector-specific. |
| RepLang scripting/macros used by connector | Yes | Yes | Verified for connector | Connector-specific. |
| Report Writer Pro available | Yes | Yes/Unknown from supplied evidence | Verified for Axys; Medium Confidence for APX practitioner context | Axys product materials mention Report Writer Pro; practitioner notes discuss APX reporting. |
| Standard security-master report name | Unknown | Unknown | Unknown | Not found in supplied research. |
| Complete REP field exposure | Unknown | Unknown | Unknown | Not established. |
| APX public views as alternative | N/A | Possible but limited | Medium Confidence | Security-specific fields Unknown. |

### 6.2 REP Extraction Notes

| Note | Confidence | Implementation Meaning |
|---|---:|---|
| A practical extract may be generated through REP32 standard reports/macros rather than direct database/file access. | High Confidence | Supported by connector documentation; validate fields in each site. |
| Report/export labels may not equal native database or file fields. | High Confidence | Treat labels from CRM/report exports as output labels unless verified. |
| Public views in APX may not expose all desired data. | Medium Confidence | Validate APX view list and field coverage in the client environment. |
| Standard security-master REP report name remains Unknown. | Unknown | Do not name a report without source evidence. |

---

## 7. Data Model

### 7.1 Supported Conceptual Model

The available evidence supports a conceptual security-reference model, not a full native schema.

```text
External security identifiers
    Ticker
    CUSIP
    Security name
    Institution
    Account number
        |
        | matching / translation
        v
Product security identifier
    Axys Symbol + Type
    APX Symbol + Type
        |
        | used by import/export workflows
        v
Positions, transactions, prices, reports, performance, classifications
```

### 7.2 Native Schema Status

| Data Area | Axys | APX | Confidence | Notes |
|---|---:|---:|---:|---|
| Security symbol | Verified in integration context | Verified in integration context | Verified | Product-specific symbol used by CI. |
| Security type | Verified in integration context | Verified in integration context | Verified | Product-specific type used by CI. |
| Security name | Verified in export/conversion/translation context | Verified in translation context | Verified in context | Native field name Unknown. |
| Ticker | Verified matching context | Verified matching context | Verified | External/matching identifier. |
| CUSIP | Verified matching context | Verified matching context | Verified | External/matching identifier. |
| Asset class | Verified export/processing context | Context not fully separated | Verified / Medium Confidence | Export label for Axys; performance impact noted in Advent context. |
| Industry group | Import dependency, not field name | Import dependency, not field name | Verified as dependency, exact system split Unknown | Source not fully separated. |
| Industry sector | Import dependency, not field name | Import dependency, not field name | Verified as dependency, exact system split Unknown | Source not fully separated. |
| Fixed-income accrued-interest fields | May be exportable through Axys security file when selected | Unknown | Verified conversion context / Unknown field names | Exact fields Unknown. |
| Complete security type dictionary | Unknown | Unknown | Unknown | Examples only. |
| Native primary key | Unknown | Unknown | Unknown | Not supplied. |

### 7.3 Classification Dependencies

AdventGuru practitioner research states that security import/merge workflows can depend on referenced industry group and sector records already existing. The supplied evidence is not fully separated between Axys and APX.

| Dependency | Axys | APX | Confidence | Notes |
|---|---:|---:|---:|---|
| Industry group must exist before import | Yes/likely | Yes/likely | Verified as Advent import dependency; system split not fully separated | Treat as implementation caution. |
| Industry sector must exist before import | Yes/likely | Yes/likely | Verified as Advent import dependency; system split not fully separated | Treat as implementation caution. |
| Complete classification dictionary | Unknown | Unknown | Unknown | Not supplied. |
| Exact security-master fields for industry group/sector | Unknown | Unknown | Unknown | Not supplied. |

### 7.4 Security Type / Asset Class Processing Impact

| Statement | Axys | APX | Confidence | Notes |
|---|---:|---:|---:|---|
| Security type and asset-class definitions can affect accounting/performance behavior. | Yes/likely | Yes/likely | Medium Confidence | Practitioner source, not fully separated. |
| Reclassing a security type as another asset class can affect historical performance. | Yes/likely | Yes/likely | Verified as AdventGuru statement; system split not fully separated | May require performance regeneration according to source. |
| Performance history may need regeneration after security type / asset-class configuration changes. | Yes/likely | Yes/likely | Verified as AdventGuru statement; system split not fully separated | Treat as implementation caution. |

---

## 8. Field Dictionaries

### 8.1 Common Field / Label Dictionary

This table consolidates field names, labels, executable names, file names, and integration artifacts found in the supplied research. It is intentionally conservative.

| Field / Label / Artifact | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---:|
| `Security` | AdvisorEngine Axys XLS asset export label. | Yes | Unknown | Unknown | Possibly report/export | Verified, export-context only |
| `Sec Type Code` | AdvisorEngine Axys XLS asset export label. | Yes | Unknown | Unknown | Possibly report/export | Verified, export-context only |
| `Security Symbol` | AdvisorEngine Axys XLS asset export label. | Yes | Unknown | Unknown | Possibly report/export | Verified, export-context only |
| `Security Type` | Export label / security type concept. | Yes | Yes | Unknown | Possibly report/export | Verified in context |
| `Asset Class` | AdvisorEngine Axys XLS asset export label; also relevant to processing caveats. | Yes | Context not fully separated | Unknown | Possibly report/export | Verified export-context / Medium Confidence processing context |
| `Axys Symbol` | Product-specific Axys security symbol in CI translations. | Yes | No | Integration context | Unknown | Verified |
| `APX Symbol` | Product-specific APX security symbol in CI translations. | No | Yes | Integration context | Unknown | Verified |
| `APX Security Type` | APX type in CI translation workflow. | No | Yes | Integration context | Unknown | Verified |
| `APX Type` | Example APX type label in translation example. | No | Yes | Integration context | Unknown | Verified |
| `Type` | Security type field in CI security translations and missing-prices files. | Yes | Yes | Integration output | Unknown | Verified |
| `Symbol` | Product security symbol in CI missing-prices files. | Yes | Yes | Integration output | Unknown | Verified |
| `Ticker` | External identifier used in matching/translations. | Matching context | Matching context | Unknown | Unknown | Verified |
| `CUSIP` | External identifier used in matching/translations and AIA validation. | Matching context | Matching context | Unknown | Unknown | Verified |
| `WP Name` | WebPortfolio security name in CI security translations. | Yes | Yes | Integration output | Unknown | Verified |
| `WP Ticker` | WebPortfolio ticker in CI security translations. | Yes | Yes | Integration output | Unknown | Verified |
| `WP Cusip` | WebPortfolio CUSIP in CI security translations. | Yes | Yes | Integration output | Unknown | Verified |
| `Institution` | Institution name in CI security translations and missing-prices files. | Yes | Yes | Integration output | Unknown | Verified |
| `Financial Institution` | External/institution matching concept in CI. | Yes | Yes | Integration context | Unknown | Verified |
| `WP Account #` | WebPortfolio account number in CI security translations. | Yes | Yes | Integration output | Unknown | Verified |
| `WP Account` | WebPortfolio account nickname in CI missing-prices file. | Yes | Yes | Integration output | Unknown | Verified |
| `Created` | Date CI security translation was first created. | Yes | Yes | Integration output | Unknown | Verified |
| `Last Modified` | Date CI security translation was last modified. | Yes | Yes | Integration output | Unknown | Verified |
| `Name` | Name field in CI missing-prices file. | Yes | Yes | Integration output | Unknown | Verified |
| `sec.inf` | Security information file/layout used by CI/AIA/conversion workflows. | Yes | Yes | Export/import context | Unknown | Verified in context |
| `type.inf` | Security type information file used by CI workflows. | Yes | Yes | Export context | Unknown | Verified in CI context |
| `.veh` | AIA vehicle file transformed to `sec.inf` layout. | Yes | Yes | Import staging | Unknown | Verified in AIA context |
| `imex32.exe` | Axys Import/Export utility executable. | Yes | No | Utility | No | Verified |
| `apxix.exe` / `APXIX.exe` | APX Import/Export utility/function. | No | Yes | Utility | No | Verified |
| `ApxIx` | APX Import/Export terminology in CI guide. | No | Yes | Utility terminology | No | Verified |
| `pospos32.exe` | Axys Post Positions utility. | Yes | No | Utility-adjacent | No | Verified |
| `topost.trn` | Axys Trade Blotter file referenced by CI for transactions. | Yes | No | Transaction workflow | Unknown | Verified in CI context |
| `$pathexe` | Axys executable folder label in CI configuration. | Yes | No | Folder config | Unknown | Verified in CI context |
| `$pathtrn` | Axys user folder label in CI configuration. | Yes | No | Folder config | Unknown | Verified in CI context |
| `$pathcli` | Axys client folder label in CI configuration. | Yes | No | Folder config | Unknown | Verified in CI context |
| `$pathinf` | Axys information folder label in CI configuration. | Yes | No | Folder config | Unknown | Verified in CI context |
| `$pathpri` | Axys price folder label in CI configuration. | Yes | No | Folder config | Unknown | Verified in CI context |
| `$pathlog` | Axys log folder label in CI configuration. | Yes | No | Folder config | Unknown | Verified in CI context |
| `SourceId` | APX price source field shown in AIA APX screenshot/example; not a security-master field. | No | Yes | Price import context | Unknown | Verified, non-security-master context |
| `aw`, `br`, `ex`, `ep`, `pi`, `rs` | APX CI reserved type prefixes excluded from CI security matching. | Unknown | Yes | Matching behavior | Unknown | Verified in CI context |
| Industry group | Referenced classification dependency for imports. | Yes/likely | Yes/likely | Import dependency | Unknown | Verified as Advent dependency; exact field Unknown |
| Industry sector | Referenced classification dependency for imports. | Yes/likely | Yes/likely | Import dependency | Unknown | Verified as Advent dependency; exact field Unknown |

### 8.2 Axys CI Security Translations File Dictionary

The Axys CI User Guide includes an optional output file named `SECTRANSLATIONS_yyyymmdd.csv`. This is a Custodial Integrator translation file, not the native Axys security-master schema.

| Column Header | Required | Data Type | Description | Confidence |
|---|---:|---|---|---:|
| `WP Name` | `*` | `CHAR128` | Name of the security as it appears in WebPortfolio. | Verified |
| `WP Ticker` | `*` | `CHAR6` | Ticker symbol from WebPortfolio, if available. | Verified |
| `WP Cusip` | `*` | `CHAR9` | CUSIP from WebPortfolio, if available. | Verified |
| `Institution` | No | `CHAR128` | Name of the institution where the security is held. | Verified |
| `WP Account #` | No | `CHAR128` | WebPortfolio account number if the translation is account-specific. | Verified |
| `Axys Symbol` | Yes | `CHAR512` | Symbol used to identify the security; product-specific. | Verified |
| `Type` | Yes | `CHAR6` | Security type defined in the security master, such as `CAUS` or `CSUS`; product-specific. | Verified |
| `Created` | Yes | `DATE` | Date security translation was first created, `YYYYMMDD`. | Verified |
| `Last Modified` | Yes | `DATE` | Date security translation was last modified, `YYYYMMDD`. | Verified |

The CI guide states that at least one of the `*` fields must be provided.

### 8.3 APX CI Security Translations File Dictionary

The APX CI User Guide includes an optional output file named `SECTRANSLATIONS_yyyymmdd.csv`. This is a Custodial Integrator translation file, not the native APX database schema.

| Column Header | Required | Data Type | Description | Confidence |
|---|---:|---|---|---:|
| `WP Name` | `*` | `CHAR128` | Name of the security as it appears in WebPortfolio. | Verified |
| `WP Ticker` | `*` | `CHAR6` | Ticker symbol from WebPortfolio, if available. | Verified |
| `WP Cusip` | `*` | `CHAR9` | CUSIP from WebPortfolio, if available. | Verified |
| `Institution` | No | `CHAR128` | Name of the institution where the security is held. | Verified |
| `WP Account #` | No | `CHAR128` | WebPortfolio account number if the translation is account-specific. | Verified |
| `APX Symbol` | Yes | `CHAR512` | Symbol used to identify the security; product-specific. | Verified |
| `Type` | Yes | `CHAR6` | Security type defined in the security master, such as `CAUS` or `CSUS`; product-specific. | Verified |
| `Created` | Yes | `DATE` | Date security translation was first created, `YYYYMMDD`. | Verified |
| `Last Modified` | Yes | `DATE` | Date security translation was last modified, `YYYYMMDD`. | Verified |

The CI guide states that at least one of the `*` fields must be provided.

### 8.4 CI Missing Prices File Dictionary

Both Axys and APX CI User Guides include an optional output file named `MISSINGPRICES_yyyymmdd.csv`. This is not a security-master file, but it verifies the security identifier vocabulary used by CI.

| Column Header | Required | Data Type | Axys Description | APX Description | Confidence |
|---|---:|---|---|---|---:|
| `Symbol` | Yes | `CHAR512` | Any Axys security symbol defined in the security master. | Any APX security symbol defined in the security master. | Verified |
| `Type` | Yes | `CHAR6` | Any Axys security type defined in the security master. | Any APX security type defined in the security master. | Verified |
| `Name` | No | `CHAR128` | Name of the security or position with no price. | Name of the security or position with no price. | Verified |
| `WP Account` | Yes | `CHAR64` | WebPortfolio account nickname. | WebPortfolio account nickname. | Verified |
| `Institution` | No | `CHAR128` | Financial institution where the security is held. | Financial institution where the security is held. | Verified |

---

## 9. Examples

### 9.1 APX Security Translation Example

The supplied research includes this APX CI translation example.

| WP Ticker | WP Name | APX Symbol | APX Type | Confidence |
|---|---|---|---|---:|
| `LMNVX` | `LEGG MASON VLE TR INSTL` | `524659208` | `efus` | Verified |

Interpretation:

| Technical Point | Confidence |
|---|---:|
| An external ticker can map to an APX symbol that is not the same as the ticker. | Verified |
| APX CI translation uses APX Symbol and APX Type together. | Verified |
| `efus` is an example APX security type code in the integration source. | Verified |
| Whether `efus` is universal across all APX versions/sites is Unknown. | Unknown |

### 9.2 APX Duplicate Security Examples

| Duplicate Condition | Example / Explanation | Confidence |
|---|---|---:|
| Same symbol, different security types | Research example references `ktc csus` and `ktc adus`. | Verified |
| Same security defined by ticker and CUSIP | A security may be defined twice, once with ticker as symbol and once with CUSIP as symbol. | Verified |
| Multiple overlapping translations | Multiple CI translations can match the same security and produce more than one possible APX translation. | Verified |

### 9.3 Axys Duplicate Security Examples

| Duplicate Condition | Example / Explanation | Confidence |
|---|---|---:|
| WebPortfolio provides ticker, CUSIP, and name; Axys contains both ticker and CUSIP entries | Research example describes NEW PERSPECTIVE FD CL A with ticker `ANWPX` and CUSIP `648018109`; CI cannot determine which Axys security-master entry to use without resolution. | Verified |
| Same CUSIP entered twice with different security types | Research states the same CUSIP had been entered twice in Axys security master with different security types, such as `tfus` and `oaus`. | Verified |
| Same symbol with different types | CI can encounter more than one Axys security using the same symbol but different type. | Verified |

### 9.4 Axys XLS Asset Export Example

AdvisorEngine requires an Advent Axys XLS asset import in this order:

| Order | Column |
|---:|---|
| 1 | `Portfolio Name` |
| 2 | `Portfolio Code` |
| 3 | `Security` |
| 4 | `Sec Type Code` |
| 5 | `Security Symbol` |
| 6 | `Security Type` |
| 7 | `Market Value` |
| 8 | `Quantity` |
| 9 | `Asset Class` |

Interpretation:

| Technical Point | Confidence |
|---|---:|
| Axys can produce an XLS export/report containing security name/symbol/type/asset-class fields alongside position values. | Verified |
| This evidence is insufficient to prove native security-master field names or IMEX field names. | Verified caveat |

---

## 10. Processing Behavior

### 10.1 Security Matching and Translation

| Process | Axys | APX | Confidence | Notes |
|---|---:|---:|---:|---|
| Match external security data to product security master | Yes | Yes | Verified | CI/AIA context. |
| Use ticker in matching | Yes | Yes | Verified | External matching context. |
| Use CUSIP in matching | Yes | Yes | Verified | External matching context. |
| Use security name/institution in translations | Yes | Yes | Verified | CI translation files. |
| Map external security to product symbol/type | Yes | Yes | Verified | CI translation files. |
| Translation may be account-specific | Yes | Yes | Verified | CI context. |
| Security translations take precedence over automatic matching | Unknown | Yes | Verified for APX | Axys precedence not established in supplied research. |
| CI does not modify product security information as part of translation | Unknown | Yes | Verified for APX | Axys equivalent not established. |

### 10.2 Missing Security Remediation

| Condition | Axys | APX | Confidence | Remediation Supported by Research |
|---|---:|---:|---:|---|
| Security not found / cannot translate | Yes | Yes | Verified | Define the security in the product or provide CI security translation. |
| Insufficient identifier information | Yes | Yes | Verified | Supply additional matching/translation information. |
| More than one matching security | Yes | Yes | Verified | Define translations or remove/resolve redundant security records/translations. |
| Security not in APX security master | N/A | Yes | Verified | Define security and rerun import with APX security data import selected. |

### 10.3 Duplicate Resolution Rules and Cautions

| Issue | Axys | APX | Confidence | Implementation Caution |
|---|---:|---:|---:|---|
| Same security appears once by ticker and once by CUSIP | Yes | Yes | Verified | Do not assume ticker and CUSIP records are automatically equivalent. |
| Same symbol appears with different security types | Yes | Yes | Verified | Do not match on symbol alone. |
| Multiple translations match the same external security | Yes | Yes | Verified | Translation table may require cleanup. |
| Account-specific translations constrain later global translations | Yes | Yes | Verified | Once account-specific translations exist for a security, global translation may not be available for that same security. |

### 10.4 Import Dependencies

| Dependency / Behavior | Axys | APX | Confidence | Notes |
|---|---:|---:|---:|---|
| Industry group must exist before security import | Yes/likely | Yes/likely | Verified as Advent import dependency; system split not fully separated | Treat as implementation caution. |
| Industry sector must exist before security import | Yes/likely | Yes/likely | Verified as Advent import dependency; system split not fully separated | Treat as implementation caution. |
| Open/locked blotters can block imports | Unknown | Yes | Verified in AIA APX workflow | APX integration/process quirk. |
| Security-master size affects translation runtime | Unknown | Yes | Verified in CI workflow | APX CI context. |
| Unknown accounts/vehicles may be written to a Pending Data to Process file | Unknown | Yes | Verified in AIA workflow | AIA context. |

### 10.5 Performance Impact of Security-Type / Asset-Class Changes

| Change | Axys | APX | Confidence | Consequence |
|---|---:|---:|---:|---|
| Security type reclassified as another asset class | Yes/likely | Yes/likely | Verified as AdventGuru statement; system split not fully separated | Historic performance may be affected. |
| Security type / asset class configuration changes | Yes/likely | Yes/likely | Verified as AdventGuru statement; system split not fully separated | Performance history may require regeneration. |

---

## 11. Version and Environment Notes

| Statement | Axys | APX | Confidence | Scope |
|---|---:|---:|---:|---|
| Salentica/Elements Data Broker lists support for Axys 3.8.6. | Yes | No | Verified for connector only | Not product-wide compatibility statement. |
| Salentica/Elements Data Broker lists support for APX 15.2, 16.1, 16.2, and 17.1. | No | Yes | Verified for connector only | Not product-wide compatibility statement. |
| APX public views exist but are limited. | N/A | Yes | Medium Confidence | Practitioner source. |
| `.addlabel` script command works in Axys but is not valid in APX. | Yes | Not valid | Verified | Not a security-master command; relevant scripting/platform difference. |
| APX equivalent workflow for the cited `.addlabel` context involves posting through the trade blotter. | N/A | Yes | Verified in supplied research | Not generalized beyond cited context. |
| Exact security-master storage model is the same across Axys and APX. | Unknown | Unknown | Unknown | Not established. |
| Version-by-version security-master schema differences. | Unknown | Unknown | Unknown | Not supplied. |

---

## 12. Known Issues / Quirks

| Issue / Quirk | Axys | APX | Confidence | Notes |
|---|---:|---:|---:|---|
| Duplicate security identification from ticker/CUSIP | Yes | Yes | Verified | Can require manual translation/resolution. |
| Same symbol with different security types | Yes | Yes | Verified | Symbol alone is unsafe as a unique key. |
| Security translations may be required before imports can complete. | Yes | Yes | Verified | CI context. |
| Security translations can be account-specific. | Yes | Yes | Verified | CI context. |
| Account-specific translations may prevent later global translation for the same security. | Yes | Yes | Verified | CI context. |
| `sec.inf` layout is used by third-party integration tooling for vehicle/security data. | Yes | Yes | Verified in AIA context | Not full native schema. |
| `sec.inf` may be important for conversion of user-defined securities and accrued-interest fields from Axys. | Yes | Unknown | Verified in conversion context | Exact field names Unknown. |
| Security master merge/import depends on classification tables. | Context not fully separated | Context not fully separated | Verified as AdventGuru statement | Exact field names Unknown. |
| Changes to security type/asset class can affect performance history. | Context not fully separated | Context not fully separated | Verified as AdventGuru statement | May require regeneration. |
| APX public views are limited. | N/A | Yes | Medium Confidence | Security-specific coverage Unknown. |
| `addlabel` script command works in Axys but not APX. | Yes | Not valid | Verified | Scripting/platform quirk, not security field rule. |
| Open/locked blotters can block imports. | Unknown | Yes | Verified in AIA workflow | APX integration/process quirk. |
| Security-master size affects integration runtime. | Unknown | Yes | Verified in CI workflow | APX CI context. |
| APX CI reserved type prefixes excluded from matching. | N/A | Yes | Verified in CI context | Do not generalize to all APX. |
| Report output labels and connector extraction fields should not be promoted to native field names. | Yes | Yes | High Confidence caution | Applies across this chapter. |

---

## 13. Implementation Guidance

### 13.1 Safe Assumptions

The following assumptions are safe for implementation planning, but only within the stated scope.

| Safe Assumption | Scope | Confidence |
|---|---|---:|
| Axys and APX security matching integrations need product-specific symbol and type. | CI/integration workflows. | Verified |
| `sec.inf` and `type.inf` are important security/security-type exchange files in CI workflows for both Axys and APX. | CI context. | Verified |
| External identifiers such as ticker and CUSIP can create ambiguous matches. | CI/AIA matching context. | Verified |
| Security translations may be required when matching is ambiguous or undesired. | CI context. | Verified |
| Report/export labels are not necessarily native schema fields. | Repository-wide caution. | High Confidence |
| Security type / asset-class changes can have downstream performance consequences. | Advent implementation caution; not fully separated. | Medium Confidence / Verified as practitioner statement |

### 13.2 Unsafe Assumptions

| Do Not Assume | Reason |
|---|---|
| Symbol alone uniquely identifies a security. | Duplicate symbol/different type examples exist. |
| Ticker and CUSIP records automatically represent the same security. | Duplicate ticker/CUSIP entries can require manual translation. |
| CI translation file columns are native Axys/APX field names. | They are integration artifact columns. |
| AdvisorEngine XLS labels are native Axys field names. | They are export/import labels for one workflow. |
| `sec.inf` contains every security-master field. | Complete native storage model Unknown. |
| `type.inf` contains every security-type configuration. | Complete layout Unknown. |
| Security types can be imported through IMEX. | Practitioner source expresses doubt; vendor confirmation absent. |
| APX public views expose all security-master fields. | Public views are described as limited; security coverage Unknown. |
| `aw`, `br`, `ex`, `ep`, `pi`, `rs` are universal APX type rules. | They are only verified as APX CI reserved-prefix exclusions. |
| Connector version support equals product compatibility. | Connector-specific evidence only. |

### 13.3 Suggested Extract Validation Checklist

This checklist is derived from supplied evidence and known unknowns.

| Validation Item | Axys | APX | Rationale |
|---|---:|---:|---|
| Confirm security symbol field/source. | Yes | Yes | Symbol is required in integration matching. |
| Confirm security type field/source. | Yes | Yes | Type is required in integration matching. |
| Confirm whether extract source is native file, IMEX, REP, public view, or integration artifact. | Yes | Yes | Prevents confusing output labels with native fields. |
| Confirm treatment of ticker-vs-CUSIP duplicates. | Yes | Yes | Known duplicate-matching issue. |
| Confirm account-specific translation behavior. | Yes | Yes | Can constrain future translations. |
| Confirm classification dependencies before import. | Yes | Yes | Industry group/sector dependencies may block imports. |
| Confirm whether security type import is supported in the target environment. | Yes | Yes | Not vendor-confirmed in supplied material. |
| Confirm whether performance must be regenerated after type/asset-class changes. | Yes | Yes | Known implementation caution. |
| Confirm APX public-view field coverage before relying on it. | N/A | Yes | Public views may be limited. |
| Confirm `sec.inf` / `type.inf` layouts from actual site evidence. | Yes | Yes | Complete layouts Unknown. |

---

## 14. Unknowns

The following items remain Unknown and should not be documented as fact until supported by vendor documentation, sample exports, REP sources, production examples, or sanitized client files.

| Unknown | Axys | APX | Needed Evidence |
|---|---:|---:|---|
| Exact native security-master file/table/storage model. | Unknown | Unknown | Vendor file layout, APX schema, public view list, or representative export. |
| Exact `sec.inf` field layout. | Unknown | Unknown | Sanitized `sec.inf` or official file layout. |
| Exact `type.inf` field layout. | Unknown | Unknown | Sanitized `type.inf` or official file layout. |
| Native security-master field dictionary. | Unknown | Unknown | Vendor data dictionary or complete IMEX/REP/public-view field list. |
| Native security type field dictionary. | Unknown | Unknown | Vendor data dictionary or `type.inf` layout. |
| Official IMEX object name for Security. | Unknown | Unknown | IMEX object list, sample `.imx`, or vendor documentation. |
| Official IMEX object name for Security Type. | Unknown | Unknown | IMEX object list, sample `.imx`, or vendor documentation. |
| Required fields for creating a security through IMEX/import. | Unknown | Unknown | Successful import sample or vendor import spec. |
| Required fields for updating a security through IMEX/import. | Unknown | Unknown | Successful update sample or vendor import spec. |
| Whether Axys and APX `sec.inf` layouts are identical. | Unknown | Unknown | Comparative sanitized files or documentation. |
| Whether Axys and APX `type.inf` layouts are identical. | Unknown | Unknown | Comparative sanitized files or documentation. |
| Formal security-master primary key. | Unknown | Unknown | Vendor schema or tested behavior. |
| REP security-master report names. | Unknown | Unknown | REP report catalog or `.rep` source. |
| Whether REP exposes all security-master fields. | Unknown | Unknown | Report Writer field dictionary or tested extract. |
| APX public security view names. | N/A | Unknown | APX public-view list or query output. |
| APX public security view field coverage. | N/A | Unknown | APX schema/view documentation. |
| Complete security type dictionary. | Unknown | Unknown | `type.inf`, setup export, or vendor dictionary. |
| Complete asset-class mapping. | Unknown | Unknown | Setup export or vendor dictionary. |
| Exact industry group / industry sector field names. | Unknown | Unknown | Security import docs and classification setup files. |
| Bond/fixed-income security master fields. | Unknown | Unknown | Fixed-income security export or vendor field dictionary. |
| Accrued-interest field names in Axys `sec.inf`. | Unknown | N/A | Sanitized fixed-income `sec.inf` records or vendor layout. |
| Options/derivatives security master fields. | Unknown | Unknown | Security export, public view schema, or vendor documentation. |
| Tax-lot or position-level attributes vs security-master attributes. | Unknown | Unknown | REP/IMEX samples showing attribute ownership. |
| Version-specific security-master schema differences. | Unknown | Unknown | Release notes or multiple version-specific sample exports. |

---

## 15. Recommended Evidence to Improve This Chapter

The following evidence would convert many Unknowns into documented facts.

| Priority | Evidence Needed | Would Resolve |
|---:|---|---|
| 1 | Sanitized Axys `sec.inf` export. | Axys field layout, key fields, fixed-income fields if present. |
| 2 | Sanitized Axys `type.inf` export. | Axys security type layout and type dictionary. |
| 3 | Sanitized APX `sec.inf` export. | APX security layout and comparison to Axys. |
| 4 | Sanitized APX `type.inf` export. | APX type layout and comparison to Axys. |
| 5 | APX public-view list and security-related view field list. | APX SQL/public-view coverage. |
| 6 | Official IMEX object list or sample security `.imx` export. | Security/security type IMEX object names. |
| 7 | Successful security import samples, including error logs. | Required import fields and validation behavior. |
| 8 | REP report source for security-list/security-master reports. | REP report names and field exposure. |
| 9 | Security samples for equities, mutual funds, bonds, options, cash equivalents, currencies. | Asset-class-specific field coverage. |
| 10 | Version-specific Axys/APX documentation. | Version differences. |

---

## 15.1 Deep IMEX Update

The deep IMEX research reinforces that `sec.inf` and `type.inf` are supported
as CI-observed Axys reference files, but not as complete public schemas.

| Item | Chapter treatment | Confidence |
|---|---|---:|
| `sec.inf` | Security Information used by CI to map external securities and generate valid imports. | Verified for CI |
| `type.inf` | Security Type Information used by CI with security master data. | Verified for CI |
| `MISSINGPRICES_yyyymmdd.csv` | CI diagnostic with Symbol, Type, Name, WP Account, and Institution. | Verified for CI output |
| `SECTRANSLATIONS_yyyymmdd.csv` | CI diagnostic/output mapping WebPortfolio identifiers to Axys Symbol and Type. | Verified for CI output |
| Candidate live-discovery fields | Symbol/type, name, ticker, CUSIP, ISIN, currency, multiplier, factor, coupon, maturity, classifications, and user-defined fields. | Discovery guidance |

Do not present the CI diagnostics or candidate live-discovery fields as an
official universal Axys IMEX security-master schema.

---

## 16. References

This chapter is based only on the supplied repository chapter and research material. Source names below are taken from the supplied research notes.

### 16.1 Governing Repository Document

| Source | Use |
|---|---|
| `AXYS_APX_REFERENCE_BLUEPRINT.md`, Version 2.0 | Editorial standards, evidence labels, chapter structure, field dictionary standard, and requirement to preserve Unknowns. |

### 16.2 Supplied Chapter / Research Files

| Source | Use |
|---|---|
| `Chapter_04_Security_Master.md` | Prior chapter draft used as source material, then reorganized and expanded. |
| `Research_04_Security_Master.md` | Research notes, independent research addendum, and independent research update. |

### 16.3 Source Material Summarized in the Research Notes

| Source | Supported Topics | Caution |
|---|---|---|
| SS&C Advent Axys product page / brief | Axys product-level reporting and Report Writer Pro statements. | Product-level only; not field dictionary. |
| SS&C Advent APX product brief | APX product-level statements about centralized book of record, holdings, transactions, performance, asset classes, security layers, and audit trail. | Product-level only; not field dictionary. |
| ByAllAccounts Custodial Integrator User Guide for Axys | Axys security matching, duplicate cases, security translations, account-specific translations, missing-prices file dictionary, security-translations file dictionary. | CI behavior, not complete native schema. |
| ByAllAccounts Custodial Integrator Installation Guide for Axys | `imex32.exe`, export of Security (`sec.inf`) and Security Type (`type.inf`), `pospos32.exe`, import workflows. | CI installation context. |
| ByAllAccounts Custodial Integrator User Guide for APX | APX security matching, APX Symbol/APX Security Type, translation precedence, reserved type-prefix exclusions, missing-prices file dictionary, security-translations file dictionary. | CI behavior, not complete native schema. |
| ByAllAccounts Custodial Integrator Installation Guide for APX | `apxix.exe`, export of Security (`sec.inf`) and Security Type (`type.inf`), APX import workflow, authentication/logging caution. | CI installation context. |
| WealthTechs AIA User Manual for Axys Users | AIA `.veh` file using `sec.inf` layout, Axys import options. | AIA behavior, not native schema. |
| WealthTechs AIA User Manual for APX Users | AIA `.veh` file using `sec.inf` layout, APX import options, APX export folder behavior. | AIA behavior, not native schema. |
| Morningstar Advent Axys conversion guide | Axys `sec.inf` conversion of user-defined security names and accrued-interest-related fields when selected. | Conversion context only. |
| AdvisorEngine Advent Axys Asset Import | Axys XLS export column labels and order. | Export/report labels only. |
| Salentica / Elements / Engage Data Broker documentation | REP32, standard reports/macros, RepLang scripting/macros, connector version support. | Connector-specific. |
| AdventGuru Axys/APX integration and reporting articles | Security type handling cautions, security-master merge dependencies, public view limitations, APX reporting/database access options, Report Writer Pro/Replang usage. | Practitioner source; not official vendor schema. |
| FinFolio Advent conversion page | Advent `INF` files used in conversion workflows to build target security master. | Conversion-vendor clue; Medium Confidence only. |
| Advent / Thomson Reuters DataScope brief | General Advent ecosystem reference-data concepts. | Geneva-focused; not used for Axys/APX security-master behavior unless separately verified. |
