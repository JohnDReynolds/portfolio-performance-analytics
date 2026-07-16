# Research Notes: Security Master

Repository area: `docs/axys-apx-reference/evidence/`  
Prepared: 2026-06-29  
Target chapter: `docs/axys-apx-reference/reference/Chapter_04_Security_Master.md`  
Scope: Axys, APX, IMEX, REP, field names, report names, processing behavior, version differences, implementation quirks, examples, references.

## Governing specification

These notes follow `AXYS_APX_REFERENCE_BLUEPRINT.md` Version 2.0. The blueprint requires factual, implementation-oriented Axys/APX documentation, separation of Axys and APX when behavior differs, and explicit confidence labels: Verified, High Confidence, Medium Confidence, Unknown.

## Confidence labels used here

| Label | Meaning in these notes |
|---|---|
| Verified | Directly supported by cited source text or visible source screenshot. |
| High Confidence | Strongly supported by one or more third-party or vendor-adjacent sources, but not enough to prove complete product behavior. |
| Medium Confidence | Plausible and useful, but the available evidence is partial, context-specific, or indirect. |
| Unknown | Not established by the available source material. Do not promote to fact without additional evidence. |

## Executive summary

| Topic | Research finding | System | Confidence |
|---|---|---:|---|
| Security master exists as a distinct dataset/concept. | Multiple integration/conversion sources refer to an Axys/APX security master or security information that is read, merged, imported, or used for matching. | Axys/APX | Verified |
| Security identity commonly combines symbol and security type. | Integration guides use `APX Symbol` / `APX Security Type` and examples such as `524659208` + `efus`; duplicate examples show same symbol may appear with different security types. | APX; similar Axys behavior indicated | Verified |
| `sec.inf` and `type.inf` are used by APX-oriented Custodial Integrator as APX security and type information. | ByAllAccounts APX guide states that Custodial Integrator uses APX security information from `sec.inf` and `type.inf` files. | APX integration context | Verified |
| Axys security master duplication can involve both ticker and CUSIP entries for the same listed security. | ByAllAccounts Axys guide gives an example where Axys has both ticker and CUSIP entries matching the same security. | Axys | Verified |
| Security master import can fail if referenced industry group/sector records do not exist. | AdventGuru states Advent will not import a security with an invalid industry group or industry sector. | Axys/APX context not fully separated | Verified |
| Security type / asset class changes can affect historical performance. | AdventGuru states reclassing a security type as another asset class impacts historic performance and may require regeneration. | Axys/APX context not fully separated | Verified |
| APX users can access public views, but public views are limited. | AdventGuru states APX has public views but they are limited and do not expose all desired data. | APX | Verified |
| REP32 can be used for Axys/APX extraction via standard reports/macros and RepLang. | Data Broker documentation states its connector uses REP32, standard reports, macros, and RepLang scripting. | Axys/APX | Verified |
| Exact native Axys/APX security-master field dictionary is not publicly established by the available sources. | Available sources show integration field labels and filenames, but not a vendor data dictionary. | Axys/APX | Unknown |

## Axys

### Axys security master behavior

| Statement | Confidence | Evidence |
|---|---:|---|
| Axys has a security master that can contain securities identified by symbol and type. | Verified | The ByAllAccounts Axys guide describes an `Axys securities` table and examples where searches use Axys symbol information, including same CUSIP entered twice with different security types. |
| Axys security matching may produce duplicate matches when both ticker and CUSIP are present as separate security-master entries. | Verified | The ByAllAccounts Axys guide example says WebPortfolio provides ticker, CUSIP, and name for a security and Axys has entries for both ticker and CUSIP; CI cannot determine which Axys security master entry to use without resolution. |
| Axys security translations can be defined when automatic security matching is not desired. | Verified | The Axys Custodial Integrator guide describes custom security translations when CI automatically maps a security to an Axys symbol and the user wants a different translation. |
| Axys account-specific security translations can be used for rare cases where securities with the same identifier information are different securities in different accounts. | Verified | The Axys Custodial Integrator guide says account-specific security translations are for rare cases where securities with the same CUSIP/ticker/name held in different accounts are actually different securities. |
| Axys security master data appears in CRM-oriented asset exports as `Security`, `Sec Type Code`, `Security Symbol`, `Security Type`, and `Asset Class`. | Verified, but export-context only | AdvisorEngine’s Advent Axys Asset Import article requires these fields in an XLS export from Axys. This proves these labels appear in at least one Axys export/report workflow, not necessarily that they are native field names. |
| Axys `INF` files are used in conversion workflows to build a target security master. | Medium Confidence | FinFolio states that Advent `INF` files are used to build FinFolio’s security master. The statement is conversion-vendor specific and does not provide a native Axys data dictionary. |

### Axys field evidence

| Field or label | Meaning / use shown in sources | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---:|
| `Security` | Required column in an Axys XLS asset export for AdvisorEngine import. | Yes | Unknown | Unknown | Possibly report output | Verified, export-context only |
| `Sec Type Code` | Required column in Axys XLS asset export. | Yes | Unknown | Unknown | Possibly report output | Verified, export-context only |
| `Security Symbol` | Required column in Axys XLS asset export. | Yes | Unknown | Unknown | Possibly report output | Verified, export-context only |
| `Security Type` | Required column in Axys XLS asset export. | Yes | Unknown | Unknown | Possibly report output | Verified, export-context only |
| `Asset Class` | Required column in Axys XLS asset export. | Yes | Unknown | Unknown | Possibly report output | Verified, export-context only |
| Axys symbol | Used by CI to search Axys securities table. | Yes | Unknown | Unknown | Unknown | Verified |
| Axys security type | Appears in duplicate examples; same CUSIP can exist with different security types such as `tfus` and `oaus`. | Yes | Unknown | Unknown | Unknown | Verified |
| Ticker | WebPortfolio identifier used for translation/matching. | Integration input | Integration input | Unknown | Unknown | Verified |
| CUSIP | WebPortfolio identifier used for translation/matching; can match Axys security entries. | Integration input / matching key | Integration input / matching key | Unknown | Unknown | Verified |
| Name / Financial Institution | Used for custom WebPortfolio security translation. | Integration input | Integration input | Unknown | Unknown | Verified |

### Axys implementation quirks

| Quirk | Description | Confidence |
|---|---|---:|
| Duplicate security matches | If both ticker and CUSIP identify securities in Axys, CI may find more than one candidate and require manual security translation. | Verified |
| Account-specific translations are mutually constraining | The ByAllAccounts guide says once account-specific translations are used for a security, additional account-specific translations may be required for each account and a global translation cannot be established for that security. | Verified |
| Security type / asset class changes can invalidate downstream historical reporting. | AdventGuru states that reclassing a security type as another asset class impacts historic performance and that performance history may need regeneration after configuration changes. | Verified |
| Industry group and sector dependencies must be handled before security-master import. | AdventGuru states that Advent will not import a security with invalid industry group or sector. | Verified |
| Axys script command difference vs APX | AdventGuru’s example uses `.addlabel` in Axys and notes that this script command is not valid in APX. This is not a security-master command, but it is a version/platform difference relevant to implementation scripting around Axys/APX migrations. | Verified |

## APX

### APX security master behavior

| Statement | Confidence | Evidence |
|---|---:|---|
| APX is described by SS&C as a centralized book of record integrating portfolio management, performance measurement, accounting, and reporting. | Verified | SS&C APX product brief. |
| APX manages holdings, transactions, performance, positions, cash, and multiple asset types. | Verified | SS&C APX product brief says APX tracks holdings, transactions and performance, and supports equities, fixed income, mutual funds, FX, derivatives, and alternatives. |
| APX security matching in third-party integrations uses APX security type and APX symbol. | Verified | ByAllAccounts APX guide states CI must determine APX security type and security symbol for all positions and transactions imported into APX. |
| APX security information may be cached/imported into integration tools from `sec.inf` and `type.inf`. | Verified in CI context | ByAllAccounts APX guide says CI maintains a copy of APX Security Information and Security Type Information and uses APX security information from `sec.inf` and `type.inf`. |
| APX security matching can fail when the security is not in the APX security master, insufficient identifier information is available, or more than one APX security matches. | Verified | ByAllAccounts APX guide lists these cases under Security Matching. |
| APX trade and position blotters are part of import workflow; transactions are reviewed/postable through a Trade Blotter. | Verified | ByAllAccounts APX guide states files are imported using APX Import/Export and transactions are delivered to the designated Trade Blotter for review and posting. |
| APX uses a patented audit trail and multiple security layers. | Verified as product-level claim | SS&C APX product brief states APX protects sensitive data with multiple security layers and a patented audit trail. |
| APX public views exist but are limited. | Verified | AdventGuru states APX features public views, but they do not provide access to all desired data. |

### APX field evidence

| Field or label | Meaning / use shown in sources | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---:|
| `APX Symbol` | APX security symbol used by CI security translations; example value `524659208`. | Unknown | Yes | Import/export context | Unknown | Verified |
| `APX Security Type` / `APX Type` | APX security type used with symbol; example `efus`. | Unknown | Yes | Import/export context | Unknown | Verified |
| `sec.inf` | APX security information file as referenced by ByAllAccounts CI. | Unknown | Yes, in CI context | Source/importable data context | Unknown | Verified |
| `type.inf` | APX security type information file as referenced by ByAllAccounts CI. | Unknown | Yes, in CI context | Source/importable data context | Unknown | Verified |
| `Trade Blotter` file | Transaction import target in APX workflow. | Unknown | Yes | Yes, APX Import/Export utility | Unknown | Verified |
| `position file` | File produced by CI and imported into APX. | Unknown | Yes | Yes, APX Import/Export utility | Unknown | Verified |
| `price file` | File produced by CI and imported into APX. | Unknown | Yes | Yes, APX Import/Export utility | Unknown | Verified |
| `CUSIP` | Identifier used for matching custodial/WebPortfolio security data to APX security master. | Matching context | Matching context | Unknown | Unknown | Verified |
| `Ticker` | Identifier used for matching/translations. | Matching context | Matching context | Unknown | Unknown | Verified |

### APX implementation quirks

| Quirk | Description | Confidence |
|---|---|---:|
| Security match resolution may be required before import. | CI must determine APX security type and symbol; failures require resolving security translation errors. | Verified |
| Same APX symbol can exist with different security types. | ByAllAccounts APX guide says duplicate securities can occur when more than one APX security uses the same symbol but each has a different security type. | Verified |
| Same security may be defined by ticker and by CUSIP. | ByAllAccounts APX guide says duplicated securities can occur when an APX security is defined twice, once with ticker as symbol and once with CUSIP as symbol. | Verified |
| Security translations take precedence over other security matches. | ByAllAccounts APX guide says CI security translations take precedence over all other security matches. | Verified |
| Security master size affects integration runtime. | ByAllAccounts APX guide states the time required for data translation varies depending on the size of the APX security master and the volume of downloaded positions/transactions. | Verified |
| Blotter locking / open blotters can block imports. | WealthTechs AIA manual says open blotters should be closed and not locked for import to work. This is an APX integration/process quirk, not a core security-master field rule. | Verified |
| Unknown accounts/vehicles may be written to a pending data process file. | WealthTechs AIA manual states accounts or vehicles not found during validation are written to a Pending Data to Process file so users can add them later. | Verified |

## IMEX / Import-Export

### Verified import/export facts

| Statement | System | Confidence |
|---|---:|---:|
| APX Import/Export utility is named `ApxIx` in the ByAllAccounts manual terminology section. | APX | Verified |
| Custodial Integrator produces APX input as a transaction / Trade Blotter file, position file, and price file; these are imported into APX using APX Import/Export. | APX | Verified |
| Custodial Integrator can import APX security data into its local copy to support translation. | APX | Verified |
| AIA validates custodian vehicles by reading from the APX Security Master File and matching received CUSIPs. | APX | Verified |
| IMEX optional import of unique records is mentioned as a possible way to merge security records. | Axys/APX context not fully separated | Verified |
| AdventGuru says a full-replace security-master import can be used in a merge workflow. | Axys/APX context not fully separated | Verified |
| Exact IMEX object names for native security-master export/import are not established by available sources. | Axys/APX | Unknown |

### IMEX research gaps

| Unknown | Why it remains unknown |
|---|---|
| Native IMEX object name(s) for Axys security master. | No source in the gathered material provides an official IMEX object list or data dictionary. |
| Native IMEX object name(s) for APX security master. | Sources show APX Import/Export utility usage and `sec.inf`/`type.inf`, but not a formal object name. |
| Required fields for creating/importing securities via IMEX. | Available sources show integration labels and matching fields, not the complete required import schema. |
| Whether Axys and APX IMEX security-master layouts are identical. | Not established. |
| Whether `sec.inf`/`type.inf` are native Axys files, APX exported files, compatibility artifacts, or integration-specific copies. | ByAllAccounts refers to them for APX CI, but the broader APX native storage model is not documented in the sources. |

## REP / Report Writer / Replang

### Verified REP/reporting facts

| Statement | System | Confidence |
|---|---:|---:|
| Axys and APX can use Report Writer Pro and Replang reports for custom reporting. | Axys/APX | Verified |
| A Data Broker Advent Connector uses Advent standard reports and macros to generate extracts. | Axys/APX | Verified |
| The connector requires Advent client tools, specifically `REP32.exe`, and uses the REP32 engine plus some RepLang scripting and macros. | Axys/APX | Verified |
| APX has public views, but a third-party consultant source says they are limited and do not expose all desired data. | APX | Verified |
| Exact standard report names for a security-master report were not found. | Axys/APX | Unknown |

### REP implementation notes

| Topic | Note | Confidence |
|---|---|---:|
| Extract strategy | A practical extract may be generated through REP32 standard reports/macros rather than direct database/file access, depending on the environment. | High Confidence |
| Field names in report output | CRM import examples show user-facing report/export column labels may not equal native database/file fields. Treat such labels as report/export labels unless verified from source-data dictionary. | High Confidence |
| Public views | APX public views may be useful for some extracts, but do not assume full security-master coverage without validating the view list in the client environment. | Verified / implementation caution |

## Processing behavior

| Process | Behavior | System | Confidence |
|---|---|---:|---:|
| Security matching | Integration tooling tries to match external security identifiers to APX/Axys security information using ticker, CUSIP, name/institution, symbol, and type. | Axys/APX | Verified |
| Duplicate resolution | If more than one matching security is found, users may need to define translations or remove redundant security records/translations. | Axys/APX | Verified |
| Missing security | If a security is not defined in APX security master, CI requires remediation such as defining the security and rerunning import with APX security data import selected. | APX | Verified |
| Translation precedence | Security translations take precedence over all other security matches in CI. | APX | Verified |
| Import staging | APX integration can produce blotter/files first, then users review/post transactions. | APX | Verified |
| Import dependency | Open/locked blotters may prevent import in APX AIA workflow. | APX | Verified |
| Merge dependency | Security master merge/import may depend on industry group and sector definitions already being present/standardized. | Axys/APX context | Verified |
| Performance dependency | Security type and asset class changes can require performance-history regeneration. | Axys/APX context | Verified |

## Version differences / environment differences

| Statement | System | Confidence |
|---|---:|---:|
| Data Broker Advent Connector minimum supported versions listed by Salentica/Elements are Axys 3.8.6 and APX 15.2/16.1/16.2/17.1. | Axys/APX | Verified for that connector only |
| APX is described as an integrated platform spanning front, middle, and back office, while Axys is described as portfolio reporting/accounting software with predefined reports and customization. | Axys/APX | Verified as product-positioning language |
| `.addlabel` script command example works in Axys but is not valid in APX; APX equivalent workflow mentioned is posting through the trade blotter. | Axys/APX | Verified |
| The same exact security-master storage model between Axys and APX is not established. | Axys/APX | Unknown |
| APX public views are available but limited, whereas Axys extraction examples often involve REP/report/macros and files. | Axys/APX | Medium Confidence; source evidence is partial |

## Examples

### APX security translation example

Source example:

| WP Ticker | WP Name | APX Symbol | APX Type |
|---|---|---|---|
| `LMNVX` | `LEGG MASON VLE TR INSTL` | `524659208` | `efus` |

Interpretation:

| Technical point | Confidence |
|---|---:|
| APX matching/translation can map an external ticker to an APX symbol that is not the ticker. | Verified |
| APX security identity in this workflow uses both symbol and type. | Verified |
| `efus` is an example APX security type code in the integration source. | Verified |
| Whether `efus` is universal across all APX versions/sites is Unknown. | Unknown |

### APX duplicate security cases

Source examples identify duplicate situations:

| Duplicate condition | Example / explanation | Confidence |
|---|---|---:|
| Same symbol, different security types | Example in guide: `ktc csus` and `ktc adus`. | Verified |
| Same security defined by ticker and CUSIP | Guide states a security may be defined twice, once with ticker as symbol and once with CUSIP as symbol. | Verified |
| Multiple overlapping translations | Multiple Custodial Integrator translations can match the same security and result in more than one possible APX translation. | Verified |

### Axys duplicate security case

Source example:

| Condition | Example / explanation | Confidence |
|---|---|---:|
| WebPortfolio provides ticker, CUSIP, and name; Axys contains both ticker and CUSIP entries. | Guide example describes NEW PERSPECTIVE FD CL A with ticker `ANWPX` and CUSIP `648018109`; CI cannot determine which Axys security-master entry to use. | Verified |
| Same CUSIP entered twice with different security types. | Guide says the same CUSIP had been entered twice in Axys security master with different security types, e.g. `tfus`, `oaus`. | Verified |

### Axys asset export field order example

AdvisorEngine requires this exact column order for an Advent Axys XLS asset import:

1. `Portfolio Name`
2. `Portfolio Code`
3. `Security`
4. `Sec Type Code`
5. `Security Symbol`
6. `Security Type`
7. `Market Value`
8. `Quantity`
9. `Asset Class`

Interpretation:

| Technical point | Confidence |
|---|---:|
| Axys can produce an XLS export/report containing security name/symbol/type/asset-class fields alongside position values. | Verified |
| This is not sufficient to prove the native security-master field names or IMEX field names. | Verified caveat |

## Common field dictionary candidates

This table is intentionally conservative. These are candidate fields/labels observed in sources, not a complete security-master data dictionary.

| Field | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---:|
| `Security` | Export label in AdvisorEngine Axys asset import. | Yes | Unknown | Unknown | Likely report/export | Verified, export-context only |
| `Sec Type Code` | Export label in AdvisorEngine Axys asset import. | Yes | Unknown | Unknown | Likely report/export | Verified, export-context only |
| `Security Symbol` | Export label in AdvisorEngine Axys asset import. | Yes | Unknown | Unknown | Likely report/export | Verified, export-context only |
| `Security Type` | Export label in AdvisorEngine Axys asset import. | Yes | Unknown | Unknown | Likely report/export | Verified, export-context only |
| `Asset Class` | Export label in AdvisorEngine Axys asset import. | Yes | Unknown | Unknown | Likely report/export | Verified, export-context only |
| `APX Symbol` | APX symbol in ByAllAccounts security translation. | Unknown | Yes | Import/export context | Unknown | Verified |
| `APX Security Type` / `APX Type` | APX type in ByAllAccounts security translation. | Unknown | Yes | Import/export context | Unknown | Verified |
| `Ticker` | External identifier used in security matching/translations. | Matching context | Matching context | Unknown | Unknown | Verified |
| `CUSIP` | External/security identifier used in matching/translations. | Matching context | Matching context | Unknown | Unknown | Verified |
| `WP Name` / security name | External WebPortfolio security name. | Matching context | Matching context | Unknown | Unknown | Verified |
| `Financial Institution` | Used with name in custom WebPortfolio security translations. | Matching context | Matching context | Unknown | Unknown | Verified |
| `Account #` | Account-specific translation key. | Matching context | Matching context | Unknown | Unknown | Verified |
| `sec.inf` | Security information file referenced by CI. | Unknown | Yes, CI context | Possible source/import file | Unknown | Verified for APX CI context |
| `type.inf` | Security type information file referenced by CI. | Unknown | Yes, CI context | Possible source/import file | Unknown | Verified for APX CI context |
| Industry group | Required referenced classification when importing securities, per AdventGuru. | Yes/likely | Yes/likely | Import dependency | Unknown | Verified as Advent import dependency, exact field unknown |
| Industry sector | Required referenced classification when importing securities, per AdventGuru. | Yes/likely | Yes/likely | Import dependency | Unknown | Verified as Advent import dependency, exact field unknown |

## Known issues / quirks

| Issue | Axys | APX | Confidence |
|---|---:|---:|---:|
| Duplicate security identification from ticker/CUSIP | Verified | Verified | Verified |
| Same symbol with different security types | Verified examples | Verified examples | Verified |
| Security translations may be required for imports | Verified | Verified | Verified |
| Security translations can be account-specific | Verified | Verified | Verified |
| Security master merge/import depends on classification tables | Context not fully separated | Context not fully separated | Verified |
| Changes to security type/asset class can affect performance history | Context not fully separated | Context not fully separated | Verified |
| APX public views are limited | N/A | Verified | Verified |
| `addlabel` script command works in Axys but not APX | Verified | Verified not supported | Verified |
| Blotter lock/open status can block imports | Unknown | Verified in AIA workflow | Verified |
| Security-master size affects integration runtime | Unknown | Verified in CI workflow | Verified |

## Unknowns requiring additional source material

The following should remain `Unknown` in the chapter until supported by vendor documentation, sample IMEX exports/imports, REP reports, or production examples.

| Unknown | Needed evidence |
|---|---|
| Exact Axys native security-master file name(s), including whether all security master data is in `sec.inf` or related `.inf` files. | Axys file layout documentation or representative profile export. |
| Exact APX native security-master table/view/file names. | APX database schema/public view list or official APX documentation. |
| Exact IMEX object names for security master and security types. | IMEX object list, sample `.imx`, or export screenshots. |
| Required IMEX fields to create/update a security. | Vendor IMEX documentation or successful import sample. |
| Whether security symbol + security type is the formal primary key in Axys/APX. | Vendor schema documentation or tested behavior. |
| REP report names that expose the security master. | REP report catalog or `.rep` source. |
| Whether Report Writer Pro exposes all security-master fields. | Report Writer Pro field dictionary. |
| Whether APX public views expose all fields needed for a security-master chapter. | APX public-view field list from a client site/version. |
| Version-by-version differences in security-master layout or behavior. | Release notes or multiple sample exports from different Axys/APX versions. |
| Complete security type and asset class dictionaries. | `type.inf`, asset class setup export, or vendor field dictionary. |
| Full classification dependencies: industry group vs industry sector vs other classifications. | Security import docs and classification setup files. |
| Bond/fixed-income security master fields. | Fixed income security export, sample `sec.inf`, APX public view schema, or vendor documentation. |
| Options/derivatives/security reference fields. | Same as above. |
| Tax-lot or position-level security attributes vs security-master attributes. | REP/IMEX samples showing which fields live where. |

## Recommended next evidence to request

To turn this research into a stronger chapter, request one or more of the following:

1. A sanitized Axys `sec.inf` / security master export.
2. A sanitized Axys `type.inf` / security type file.
3. A sanitized APX security master IMEX export.
4. A sanitized APX security type IMEX export.
5. REP report source (`.rep`) for any security-master/security-list report.
6. Screenshots or field dictionaries from APX public views related to securities.
7. Vendor documentation for IMEX security/security type object definitions.
8. Examples of successful security imports, including error files/logs.
9. Samples for equities, mutual funds, bonds, options, cash equivalents, and currencies.
10. Any version-specific documentation for Axys 3.x and APX 15.x/16.x/17.x.

## References

### Governing repository document

- Uploaded file: `AXYS_APX_REFERENCE_BLUEPRINT(9).md`.

### Web / document sources consulted

1. SS&C Advent — Axys product page. Supports current product-level positioning for Axys reporting/accounting.
2. SS&C Advent — APX product brief. Supports product-level APX statements about centralized book of record, holdings/transactions/performance, reporting, multiple asset classes, security layers, and audit trail.
3. ByAllAccounts — `Custodial Integrator User Guide` for APX. Supports APX Import/Export utility terminology, APX security matching, APX symbol/type, `sec.inf`, `type.inf`, duplicate handling, and security translations.
4. ByAllAccounts — `Custodial Integrator User Guide` for Axys. Supports Axys security translation behavior, duplicate ticker/CUSIP examples, account-specific translations, and Axys security table references.
5. WealthTechs — `AIA User Manual for APX Users`. Supports APX Security Master File matching by CUSIP, pending data processing, and import/blotter locking notes.
6. AdventGuru — `Demystifying Portfolio Accounting Systems Integration Post-Merger/Acquisition`. Supports merger behavior, industry group/sector import dependencies, security master full-replace/import-unique notes, and performance-history impacts of security type/asset class changes.
7. AdventGuru — IMEX / reporting / Replang articles. Supports custom reporting options using Report Writer Pro, Replang, CSV/text exports, and APX public view limitations.
8. AdvisorEngine — `Advent Axys Asset Import`. Supports Axys XLS export field labels and order for CRM import.
9. Salentica/Elements — `Data Broker: SS&C Advent APX & Axys`. Supports Axys/APX on-premises context, Data Broker extraction using Advent standard reports/macros, REP32, RepLang, and version requirements for that connector.
10. FinFolio — Advent conversion page. Supports conversion-vendor claim that Advent `INF` files can be used to build a target security master; treat as Medium Confidence only.
11. Advent / Thomson Reuters DataScope brief. Supports general Advent ecosystem reference-data concepts, but source is Geneva-focused, so do not use as evidence of Axys/APX security-master behavior.

## Source-specific notes and caution

| Source | Caution |
|---|---|
| SS&C product pages | Useful for product-level claims; not sufficient for field dictionaries. |
| ByAllAccounts CI guides | Strong for integration behavior; not necessarily complete native Axys/APX schema documentation. |
| WealthTechs AIA manual | Strong for AIA-specific APX processing; do not generalize all APX behavior from AIA. |
| AdventGuru | Expert/consultant source, useful for implementation quirks; not vendor documentation. |
| AdvisorEngine | Shows report/export labels for one CRM import workflow only. |
| Salentica/Elements Data Broker | Describes a specific connector; version support and extraction workflow are connector-specific. |
| FinFolio | Conversion-vendor source; useful clue about `INF` files but not primary Axys/APX documentation. |
| Thomson Reuters DataScope brief | Geneva-focused; keep outside Axys/APX claims unless separately verified. |

---

## Independent Research Addendum — 2026-06-29

Purpose: supplement the original `Research_04_Security_Master.md` with
independent public-source research around the remaining evidence gaps identified
for `../reference/Chapter_04_Security_Master.md`.

Scope of this addendum:

1. Axys/APX security-master file evidence, especially `sec.inf` and `type.inf`.
2. Axys/APX security import/export utilities and process clues.
3. REP / Report Writer / Replang extraction clues.
4. APX public view / SQL reporting clues.
5. Any additional security-master fields or behavior supported by public sources.

## Addendum confidence labels

| Label | Meaning |
|---|---|
| Verified | Directly supported by cited public source text or visible PDF screenshot / extracted PDF text. |
| High Confidence | Strong evidence from public vendor, vendor-adjacent, or practitioner source, but not an official native data dictionary. |
| Medium Confidence | Plausible implementation clue, but source is conversion-specific, connector-specific, or otherwise indirect. |
| Unknown | Not established by the independent public-source research. |

## New findings summary

| Finding | System | Confidence | Source |
|---|---:|---:|---|
| Axys Custodial Integrator expects `imex32.exe` in the Axys executable folder and identifies it as Axys' Import/Export utility. | Axys | Verified | ByAllAccounts CI Axys User Guide |
| Axys Custodial Integrator identifies `pospos32.exe` as Axys Post Positions utility. | Axys | Verified | ByAllAccounts CI Axys User Guide |
| Axys folder conventions in the CI guide include executable folder, user folder, client folder, information folder, price folder, and log folder; example paths include `C:\axys\inf\` for information files. | Axys | Verified in CI context | ByAllAccounts CI Axys User Guide |
| Axys Custodial Integrator can import Axys Security Information and Security Type Information into CI to support positions, prices, and transactions generated for Axys. | Axys | Verified in CI context | ByAllAccounts CI Axys User Guide |
| Axys AIA vehicle settings create a vehicle file with the layout of `sec.inf` and can import it to Axys unless the `None` option is selected. | Axys | Verified in AIA context | WealthTechs AIA User Manual for Axys Users |
| APX AIA identifies `APXIX.exe` as the import/export function of APX. | APX | Verified | WealthTechs AIA User Manual for APX Users |
| APX AIA vehicle import settings translate a `.veh` file to the layout of the `sec.inf` file in the AIA Archive folder for viewing at the end of the Daily Process. | APX | Verified in AIA context | WealthTechs AIA User Manual for APX Users |
| APX CI security matching excludes APX security types with prefixes `aw`, `br`, `ex`, `ep`, `pi`, and `rs` from the security match process. | APX | Verified in CI context | ByAllAccounts CI APX User Guide |
| APX CI does not change APX Security Type or Security Information as part of security translation. | APX | Verified in CI context | ByAllAccounts CI APX User Guide |
| AdventGuru states security types cannot be imported through IMEX, based on the author's knowledge, and that security type / asset class definitions are central to accounting behavior. | Axys/APX context not fully separated | Medium Confidence | AdventGuru |
| Morningstar conversion guidance states that if the Advent Axys `sec.inf` file is provided, security names for user-defined securities and data fields used to calculate accrued interest can be converted. | Axys | Verified in conversion context | Morningstar Advent Axys conversion guide |
| The same Morningstar source warns that if the Advent Axys `Use Security Type` box is selected, some datapoints needed for accrued-interest calculation may not be exported to the Advent Axys security file. | Axys | Verified in conversion context | Morningstar Advent Axys conversion guide |
| APX users can access APX data using Stored Accounting Functions, Public Views, SSRS, REST API, and related reporting/database tooling. | APX | Medium Confidence | AdventGuru practitioner article |
| APX public views are described by AdventGuru as useful but limited and not exposing all desired data. | APX | Medium Confidence | AdventGuru practitioner article |
| Data Broker Advent connector uses Advent standard reports/macros, requires `REP32.exe`, and uses the REP32 engine with RepLang scripting and macros. | Axys/APX | Verified for that connector | Salentica/Elements Data Broker documentation |
| Axys product documentation / marketing states Axys has hundreds of predefined reports and Axys Report Writer Pro. | Axys | Verified product-level | SS&C Advent Axys page |

## Axys — new evidence

### Axys Import/Export executable and folder conventions

| Item | Evidence-supported statement | Confidence |
|---|---|---:|
| `imex32.exe` | The Axys CI guide says CI looks for `imex32.exe` in the Axys executable folder and identifies it as Axys' Import/Export utility. | Verified |
| `pospos32.exe` | The Axys CI guide says CI looks for `pospos32.exe` in the Axys executable folder and identifies it as Axys Post Positions utility. | Verified |
| `$pathexe` | CI guide label for Axys executable folder. | Verified in CI context |
| `$pathtrn` | CI guide label for Axys user folder; transactions are delivered to the Axys Trade Blotter file `topost.trn` in this folder. | Verified in CI context |
| `$pathcli` | CI guide label for Axys client folder. | Verified in CI context |
| `$pathinf` | CI guide label for Axys information folder; screenshot example path is `C:\axys\inf\`. | Verified in CI context |
| `$pathpri` | CI guide label for Axys price folder. | Verified in CI context |
| `$pathlog` | CI guide label for Axys log folder. | Verified in CI context |

Interpretation:

- This supports that Axys environments have a recognized information-file folder (`$pathinf`) and that integration tooling expects security/type information to be available from Axys.
- This does **not** prove the complete native security-master schema or that all security-master fields are stored only in `sec.inf`.

### Axys security information import into CI

| Statement | Confidence |
|---|---:|
| CI maintains a copy of Axys Security Information and Security Type Information to use in generating positions, prices, and transactions for Axys import. | Verified in CI context |
| CI can import Axys security data each time the data translation process runs, depending on configuration. | Verified in CI context |

### Axys AIA vehicle / security import clues

| Statement | Confidence |
|---|---:|
| AIA Vehicle Settings create a vehicle file with the layout of `sec.inf` from custodian information. | Verified in AIA context |
| The AIA-created `.veh` file is saved in the Archive folder and imported to Axys unless the `None` option is selected. | Verified in AIA context |
| AIA Axys vehicle import options include `Update Existing & Add New`, `Add New`, `Replace Entire File`, and `None`. | Verified in AIA context |
| AIA can create a security information file using WealthTechs `.veh` records from custodians that do not currently exist in Axys. | Verified in AIA context |

Implementation caution:

- AIA terminology (`vehicle`, `.veh`, Vehicle Importing Translations) is vendor-specific. It is useful evidence that security information can be transformed into `sec.inf`-layout imports, but it is **not** an official Advent field dictionary.

### Axys conversion evidence: `sec.inf` and accrued-interest data

| Statement | Confidence |
|---|---:|
| Morningstar conversion guidance says that, if Advent Axys `sec.inf` is provided, security names for user-defined securities can be converted. | Verified in conversion context |
| The same guidance says fields used to calculate accrued interest can be converted if supplied through the Advent Axys security file. | Verified in conversion context |
| The same guidance says if the Axys `Use Security Type` box is selected, datapoints needed for accrued-interest calculation may not be exported to the Advent Axys security file. | Verified in conversion context |

Interpretation:

- Public conversion guidance supports that `sec.inf` may contain user-defined security names and, when specifically selected/exported, fixed-income/accrual-related fields.
- The exact field names for accrued-interest calculation remain **Unknown** from the available public material.

## APX — new evidence

### APX Import/Export executable

| Item | Evidence-supported statement | Confidence |
|---|---|---:|
| `APXIX.exe` | The WealthTechs APX AIA manual states that `APXIX.exe` is the import/export function of APX. | Verified |

### APX AIA vehicle / security import clues

| Statement | Confidence |
|---|---:|
| APX AIA Vehicle Import Settings gather security information provided by the custodian. | Verified in AIA context |
| APX AIA translates the `.veh` file to the layout of `sec.inf` in the AIA Archive folder for viewing at the end of the Daily Process. | Verified in AIA context |
| APX AIA options include creating security information for custodian `.veh` records that do not currently exist in the APX database, and creating security information for existing records. | Verified in AIA context |

Implementation caution:

- The APX AIA documentation indicates APX has an import/export function and that vendor tooling can transform custodian vehicle records into `sec.inf` layout. It does **not** establish APX's native SQL table names, public view names, or complete field schema.

### APX security matching details from CI

| Statement | Confidence |
|---|---:|
| CI must determine APX security type and security symbol for all positions and transactions imported into APX. | Verified |
| CI may require user intervention when the security is missing from the APX security master, insufficient identifier information is present, or more than one APX security matches. | Verified |
| CI security translation input may include Ticker, CUSIP, Financial Institution, and Security Name. | Verified |
| CI target APX security information includes APX Symbol and APX Security Type. | Verified |
| CI uses APX Security Information from `sec.inf` and `type.inf` files to match securities. | Verified in CI context |
| Security translations take precedence over all other security matches. | Verified |
| APX security types with prefixes `aw`, `br`, `ex`, `ep`, `pi`, and `rs` are treated as reserved by CI and excluded from its security match process. | Verified in CI context |
| CI does not modify APX Security Type or Security Information as part of security translation. | Verified in CI context |

Implementation caution:

- The `aw`, `br`, `ex`, `ep`, `pi`, and `rs` reserved-prefix rule is specifically documented for Custodial Integrator security matching. Do **not** generalize it into an APX platform-wide security type rule without vendor documentation.

## IMEX / Import-Export — new evidence

| Statement | System | Confidence |
|---|---:|---:|
| `imex32.exe` is the Axys Import/Export utility referenced by ByAllAccounts CI. | Axys | Verified |
| `APXIX.exe` is the APX import/export function referenced by WealthTechs AIA. | APX | Verified |
| `APXIX.exe` is distinct from the APX CI guide's `ApxIx` terminology; the available evidence does not establish whether these are the same executable, different names, or version/context differences. | APX | Unknown |
| Native security-master IMEX object names remain Unknown. | Axys/APX | Unknown |
| Required fields for native security import remain Unknown. | Axys/APX | Unknown |
| Security type import through IMEX remains not officially verified; AdventGuru states the author does not believe security types can be imported through IMEX. | Axys/APX | Medium Confidence |

## REP / Report Writer / Replang — new evidence

| Statement | System | Confidence |
|---|---:|---:|
| Data Broker Advent connector is installed on a client-side machine with Advent client tools, specifically `REP32.exe`. | Axys/APX | Verified for connector |
| Data Broker uses Advent standard reports and macros to generate extracts. | Axys/APX | Verified for connector |
| Data Broker uses the REP32 engine plus RepLang scripting and macros. | Axys/APX | Verified for connector |
| AdventGuru states Axys and APX users can export reports to Excel, create macros, use Report Writer Pro, modify RepLang reports, and use third-party ETL tools. | Axys/APX | Medium Confidence |
| SS&C Axys product page says Axys includes hundreds of predefined reports and Axys Report Writer Pro. | Axys | Verified product-level |
| Exact REP report names for security master remain Unknown. | Axys/APX | Unknown |
| Whether REP exposes all security-master fields remains Unknown. | Axys/APX | Unknown |

## APX public views / SQL reporting — new evidence

| Statement | Confidence |
|---|---:|
| AdventGuru states APX users can access APX data using Stored Accounting Functions, Public Views, SSRS, REST API, and other reporting/database tools. | Medium Confidence |
| AdventGuru states APX public views are limited and do not expose all data users may want. | Medium Confidence |
| Public names and field lists for APX security-master views remain Unknown. | Unknown |
| Whether public views expose all fields needed for security-master extraction remains Unknown. | Unknown |

## Additional field dictionary candidates from independent research

These are added as **field/label candidates**, not as a complete data dictionary.

| Field / Label | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---:|
| `imex32.exe` | Axys Import/Export utility executable. | Yes | No | Utility | No | Verified |
| `APXIX.exe` | APX import/export function executable. | No | Yes | Utility | No | Verified |
| `pospos32.exe` | Axys Post Positions utility executable. | Yes | No | Utility-adjacent | No | Verified |
| `topost.trn` | Axys Trade Blotter file referenced by CI for transactions. | Yes | No | Transaction import workflow | Unknown | Verified in CI context |
| `$pathexe` | Axys executable folder label in CI. | Yes | No | Folder config | Unknown | Verified in CI context |
| `$pathtrn` | Axys user folder label in CI. | Yes | No | Folder config | Unknown | Verified in CI context |
| `$pathcli` | Axys client folder label in CI. | Yes | No | Folder config | Unknown | Verified in CI context |
| `$pathinf` | Axys information folder label in CI. | Yes | No | Folder config | Unknown | Verified in CI context |
| `$pathpri` | Axys price folder label in CI. | Yes | No | Folder config | Unknown | Verified in CI context |
| `$pathlog` | Axys log folder label in CI. | Yes | No | Folder config | Unknown | Verified in CI context |
| `.veh` | AIA vehicle file transformed to `sec.inf` layout. | Yes, AIA context | Yes, AIA context | Import staging | Unknown | Verified in AIA context |
| `SourceId` | APX price source field shown in AIA APX screenshot/example, not security-master field. | No | Yes | Price import context | Unknown | Verified, non-security-master context |
| `aw`, `br`, `ex`, `ep`, `pi`, `rs` | APX security-type prefixes treated as reserved and excluded by CI from security match process. | Unknown | Yes, CI context | Matching behavior | Unknown | Verified in CI context |

## Updated known quirks / cautions

| Issue | Axys | APX | Confidence |
|---|---:|---:|---:|
| Import/export executable name differs by platform/context: `imex32.exe` for Axys, `APXIX.exe` in AIA APX manual, `ApxIx` in CI APX terminology. | Verified | Verified / naming conflict Unknown | Verified / Unknown |
| `sec.inf` layout is used by third-party integration tooling for vehicle/security data. | Verified in AIA context | Verified in AIA context | Verified |
| `sec.inf` may be important for conversion of user-defined securities and accrued-interest fields from Axys. | Verified in conversion context | Unknown | Verified for Axys conversion context |
| APX CI reserved type-prefix exclusions should be treated as integration-specific, not system-wide, until vendor documentation confirms otherwise. | N/A | Verified in CI context | Verified caution |
| Report output labels and connector extraction fields should not be promoted to native field names without vendor schema documentation. | Yes | Yes | High Confidence caution |

## Unknowns remaining after independent research

| Unknown | Status after independent research | Needed evidence |
|---|---|---|
| Exact Axys `sec.inf` field layout. | Still Unknown. Public sources confirm use of `sec.inf` but do not provide a complete field dictionary. | Sanitized `sec.inf`, vendor file layout, or REP/IMEX export showing fields. |
| Exact Axys `type.inf` field layout. | Still Unknown. | Sanitized `type.inf` or vendor file layout. |
| Exact APX SQL security-master table/view names. | Still Unknown. | APX public-view list, SQL schema documentation, or sanitized query output. |
| Exact APX security IMEX object name. | Still Unknown. | APX IMEX object documentation or sample export definition. |
| Exact Axys security IMEX object name. | Still Unknown. | Axys IMEX object documentation or sample export definition. |
| Whether `ApxIx` and `APXIX.exe` are the same utility, different labels, or version/context-specific names. | Still Unknown. | APX installation documentation or executable listing. |
| Required fields for native security import/update. | Still Unknown. | Successful import sample, vendor import spec, or error log documentation. |
| Formal primary key for Axys security master. | Still Unknown. | Vendor data dictionary or tested database/file behavior. |
| Formal primary key for APX security master. | Still Unknown. | Vendor data dictionary or APX SQL/public-view schema. |
| REP security-master report names. | Still Unknown. | REP report catalog or `.rep` source. |
| Complete security-type dictionary and asset-class mapping. | Still Unknown. | `type.inf`, configuration export, or vendor dictionary. |
| Bond/security accrual field names in `sec.inf`. | Still Unknown. | Sanitized fixed-income security records or vendor `sec.inf` layout. |

## Additional references from independent research

1. ByAllAccounts — `Custodial Integrator User Guide` for APX. Public PDF. Supports APX security matching, APX Symbol/APX Security Type, `sec.inf`/`type.inf`, reserved type prefixes in CI matching, duplicate handling, and translation precedence. URL: `https://www.byallaccounts.net/Manuals/Custodial_Integrator/apx/CI_User_Guide.pdf`
2. ByAllAccounts — `Custodial Integrator User Guide` for Axys. Public PDF. Supports Axys `imex32.exe`, `pospos32.exe`, Axys folder labels, and CI use of Axys Security Information / Security Type Information. URL: `https://www.byallaccounts.net/Manuals/Custodial_Integrator/axys/CI_User_Guide.pdf`
3. WealthTechs — `AIA User Manual for APX Users`. Public PDF. Supports `APXIX.exe` as APX import/export function and APX `.veh` to `sec.inf` layout behavior. URL: `https://www.wealthtechs.com/files/AIADocumentation/APX%20Guide%20-%20AIA%20User%20Manual%20For%20APX%20Users.pdf`
4. WealthTechs — `AIA User Manual for Axys Users`. Public PDF. Supports Axys AIA export/import folders and vehicle/security import behavior using `sec.inf` layout. URL: `https://www.wealthtechs.com/files/AIADocumentation/Axys%20Guide%20-%20AIA%20User%20Manual%20For%20Axys%20Users.pdf`
5. Morningstar — `Converting Your Advent Axys Database into Morningstar Office`. Public PDF. Supports Axys `sec.inf` conversion behavior for user-defined security names and accrued-interest fields. URL: `https://gladmainnew.morningstar.com/articles/tutorial/30/AdventAxys.pdf`
6. Salentica / Elements — `Data Broker: SS&C Advent APX & Axys`. Public article. Supports REP32, standard reports/macros, RepLang scripting, and client-side connector model. URL: `https://elements.salentica.com/kb/article/252-data-broker-ss-c-advent-apx-axys/`
7. Salentica / Engage — `Data Broker - SS&C|Advent APX & Axys`. Public article. Supports the same REP32 / standard reports / macros / RepLang connector behavior. URL: `https://engage.salentica.com/kb/article/247-data-broker-ss-c-advent-apx-axys/`
8. AdventGuru — Axys / APX / reports / integration articles. Practitioner source. Supports security-type IMEX caution, APX public-view limitations, APX reporting/database access options, Report Writer Pro, RepLang, and CSV/text export approaches. URLs: `https://adventguru.com/category/portfolio-management-systems/axys/`, `https://adventguru.com/category/portfolio-management-systems/apx/`, `https://adventguru.com/2013/04/25/getting-data-in-and-out-of-advent-apx-and-axys/`
9. SS&C Advent — Axys product page. Supports product-level statements about predefined reports and Axys Report Writer Pro. URL: `https://www.advent.com/solutions/axys/`

## Source-specific caution for addendum

| Source | Caution |
|---|---|
| ByAllAccounts CI guides | Strong evidence for CI workflow and integration behavior, but not complete native Axys/APX schema documentation. |
| WealthTechs AIA manuals | Strong evidence for AIA-specific file handling and import behavior, but AIA is third-party tooling and may not represent all native Advent workflows. |
| Morningstar conversion guide | Strong evidence for conversion scope, but conversion-specific and not a native Advent field dictionary. |
| AdventGuru | Practitioner evidence; useful for implementation cautions but should not be treated as official vendor documentation. |
| Salentica Data Broker | Connector-specific extraction behavior. Confirms REP32-based extraction is viable for that connector, not that every desired security-master field is extractable. |
| SS&C product pages | Product-level evidence only; not a field dictionary. |

---

## Independent Research Update — 2026-06-29

Purpose: conduct additional independent public-source research against the
remaining evidence gaps for `../reference/Chapter_04_Security_Master.md`.

Method: public web/document search focused on vendor, vendor-adjacent, connector, conversion, and practitioner sources. Priority was given to PDF manuals and documentation that explicitly mention Axys/APX security master handling, `sec.inf`, `type.inf`, Import/Export utilities, security translation fields, REP extraction, and APX reporting/database access.

## Updated confidence labels

| Label | Meaning |
|---|---|
| Verified | Directly supported by the cited source text or visible PDF page. |
| High Confidence | Strongly supported by vendor-adjacent or practitioner documentation, but not an official SS&C data dictionary. |
| Medium Confidence | Plausible and implementation-useful, but source is connector-specific, conversion-specific, or practitioner commentary. |
| Unknown | Not established by public sources found in this research pass. Do not use as fact in the repository chapter. |

## New evidence gathered

### 1. Axys Import/Export utility exports Security and Security Type data

| Finding | System | Confidence | Evidence |
|---|---:|---:|---|
| ByAllAccounts Custodial Integrator uses the Axys Import/Export utility `imex32.exe` to export Security (`sec.inf`) and Security Type (`type.inf`) data from Axys. | Axys | Verified | ByAllAccounts CI Installation Guide for Axys states that CI uses `imex32.exe` to export Security (`sec.inf`) and Security Type (`type.inf`) data from the Axys installation. |
| The exported Axys security and security type data enables CI to generate transactions, positions, and prices using the security symbols and security types defined in Axys. | Axys | Verified | Same source. |
| CI also uses Axys Import/Export to import transactions into a designated Trade Blotter and prices into the Axys `.pri` price file. | Axys | Verified | Same source. |
| CI uses both Axys Import/Export and the Position Post utility `pospos32.exe` to import positions into a temporary Trade Blotter and optionally into the Axys Position Blotter. | Axys | Verified | Same source. |

Repository impact:

- The earlier “Unknown” about whether Axys `sec.inf` / `type.inf` can be exported through IMEX can be narrowed:
  - **Verified in CI context**: `imex32.exe` exports Security (`sec.inf`) and Security Type (`type.inf`) data for CI.
  - **Still Unknown**: the complete native `sec.inf` / `type.inf` field layouts, official IMEX object names, and whether every Axys installation/version supports identical layouts.

### 2. APX Import/Export utility exports Security and Security Type data

| Finding | System | Confidence | Evidence |
|---|---:|---:|---|
| ByAllAccounts Custodial Integrator uses the APX Import/Export utility `apxix.exe` to export Security (`sec.inf`) and Security Type (`type.inf`) data from APX. | APX | Verified | ByAllAccounts CI Installation Guide for APX states that CI uses `apxix.exe` to export Security (`sec.inf`) and Security Type (`type.inf`) data. |
| The exported APX security and security type data enables CI to produce transactions, positions, and prices using the security symbols and security types defined in APX. | APX | Verified | Same source. |
| CI uses APX Import/Export to import transactions into a designated Trade Blotter, prices into APX for a specified date, positions into a Position Blotter, and position lots into a Position Lot Blotter if enabled. | APX | Verified | Same source. |
| APX authentication can affect `apxix.exe`; CI may appear to wait when APX refuses access, and diagnosis may require reviewing the Apxix log file. | APX | Verified | Same source. |

Repository impact:

- The earlier “Unknown” about APX `sec.inf` / `type.inf` export can be narrowed:
  - **Verified in CI context**: `apxix.exe` exports Security (`sec.inf`) and Security Type (`type.inf`) data from APX.
  - **Still Unknown**: complete APX native database schema, public view/table names, complete IMEX object names, and full file layouts.

### 3. Axys security translation file field dictionary

The ByAllAccounts Axys CI User Guide includes a field dictionary for optional CI output file `SECTRANSLATIONS_yyyymmdd.csv`.

Important limitation: this is a **CI security translations file**, not the native Axys Security Master file layout.

| Column Header | Required | Data Type | Description | System | Confidence |
|---|---:|---|---|---:|---:|
| `WP Name` | `*` | `CHAR128` | Name of the security as it appears in WebPortfolio. | Axys CI | Verified |
| `WP Ticker` | `*` | `CHAR6` | Ticker symbol from WebPortfolio, if available. | Axys CI | Verified |
| `WP Cusip` | `*` | `CHAR9` | CUSIP from WebPortfolio, if available. | Axys CI | Verified |
| `Institution` | No | `CHAR128` | Name of the institution where the security is held. | Axys CI | Verified |
| `WP Account #` | No | `CHAR128` | WebPortfolio account number if the translation is account-specific. | Axys CI | Verified |
| `Axys Symbol` | Yes | `CHAR512` | Symbol used to identify the security. Product-specific. | Axys CI | Verified |
| `Type` | Yes | `CHAR6` | Security type defined in the security master, such as `CAUS` or `CSUS`. Product-specific. | Axys CI | Verified |
| `Created` | Yes | `DATE` | Date security translation was first created, `YYYYMMDD`. | Axys CI | Verified |
| `Last Modified` | Yes | `DATE` | Date security translation was last modified, `YYYYMMDD`. | Axys CI | Verified |

The guide states that at least one of the `*` fields must be provided.

Repository impact:

- This gives a verified **integration field dictionary** for Axys security translations.
- It does **not** prove native Axys security-master field names beyond the existence/use of Axys symbol and security type in CI matching.

### 4. APX security translation file field dictionary

The ByAllAccounts APX CI User Guide includes a field dictionary for optional CI output file `SECTRANSLATIONS_yyyymmdd.csv`.

Important limitation: this is a **CI security translations file**, not the native APX Security Master table or file layout.

| Column Header | Required | Data Type | Description | System | Confidence |
|---|---:|---|---|---:|---:|
| `WP Name` | `*` | `CHAR128` | Name of the security as it appears in WebPortfolio. | APX CI | Verified |
| `WP Ticker` | `*` | `CHAR6` | Ticker symbol from WebPortfolio, if available. | APX CI | Verified |
| `WP Cusip` | `*` | `CHAR9` | CUSIP from WebPortfolio, if available. | APX CI | Verified |
| `Institution` | No | `CHAR128` | Name of the institution where the security is held. | APX CI | Verified |
| `WP Account #` | No | `CHAR128` | WebPortfolio account number if the translation is account-specific. | APX CI | Verified |
| `APX Symbol` | Yes | `CHAR512` | Symbol used to identify the security. Product-specific. | APX CI | Verified |
| `Type` | Yes | `CHAR6` | Security type defined in the security master, such as `CAUS` or `CSUS`. Product-specific. | APX CI | Verified |
| `Created` | Yes | `DATE` | Date security translation was first created, `YYYYMMDD`. | APX CI | Verified |
| `Last Modified` | Yes | `DATE` | Date security translation was last modified, `YYYYMMDD`. | APX CI | Verified |

The guide states that at least one of the `*` fields must be provided.

Repository impact:

- This gives a verified **integration field dictionary** for APX security translations.
- It supports using `APX Symbol` + `Type` in integration/matching workflows.
- It still does **not** prove the formal APX database primary key.

### 5. Axys and APX missing-prices file fields

Both the Axys and APX CI User Guides include a field dictionary for optional output file `MISSINGPRICES_yyyymmdd.csv`.

Important limitation: this is not a security-master file. It is useful because it confirms the same security identifier vocabulary used by CI.

| Column Header | Required | Data Type | Axys description | APX description | Confidence |
|---|---:|---|---|---|---:|
| `Symbol` | Yes | `CHAR512` | Any Axys security symbol defined in the security master. | Any APX security symbol defined in the security master. | Verified |
| `Type` | Yes | `CHAR6` | Any Axys security type defined in the security master. | Any APX security type defined in the security master. | Verified |
| `Name` | No | `CHAR128` | Name of the security or position with no price. | Name of the security or position with no price. | Verified |
| `WP Account` | Yes | `CHAR64` | WebPortfolio account nickname. | WebPortfolio account nickname. | Verified |
| `Institution` | No | `CHAR128` | Financial institution where the security is held. | Financial institution where the security is held. | Verified |

Repository impact:

- Confirms `Symbol` and `Type` as the core identifier pair in CI security-master matching context for both Axys and APX.
- Does not prove formal native primary key.

### 6. APX security matching rules and reserved type-prefix exclusions

| Finding | System | Confidence | Evidence |
|---|---:|---:|---|
| CI must determine APX security type and security symbol for all positions and transactions imported into APX. | APX | Verified | APX CI User Guide. |
| CI can fail security matching if the security is not defined in APX, insufficient identifier information exists, or more than one APX security matches. | APX | Verified | APX CI User Guide. |
| CI Security Translations use external identifiers including ticker, CUSIP, financial institution, and security name. | APX | Verified | APX CI User Guide. |
| Target APX fields in the translation are APX Symbol and APX Security Type. | APX | Verified | APX CI User Guide. |
| Security translations take precedence over other security matches. | APX | Verified | APX CI User Guide. |
| APX securities directly match if the external ticker or CUSIP matches APX symbol, when no translation matches first. | APX | Verified | APX CI User Guide. |
| APX security types with prefixes `aw`, `br`, `ex`, `ep`, `pi`, and `rs` are treated as reserved by CI and excluded from the CI security match process. | APX CI | Verified in CI context | APX CI User Guide. |

Repository impact:

- The reserved-prefix rule should be documented only as a **CI matching rule**, not as a universal APX platform rule.
- The direct-match rule supports that integrations may match external ticker/CUSIP to APX symbol.

### 7. Axys security matching and duplicate resolution

| Finding | System | Confidence | Evidence |
|---|---:|---:|---|
| CI must determine Axys security type and security symbol for all positions and transactions imported into Axys. | Axys | Verified | Axys CI User Guide. |
| CI can identify securities that it cannot translate; remediation is to define the security in Axys or provide a CI security translation. | Axys | Verified | Axys CI User Guide. |
| Duplicate Axys matches can occur when more than one Axys security uses the same symbol but different security type. | Axys | Verified | Axys CI User Guide. |
| Duplicate Axys matches can occur when a security is defined once with ticker as symbol and again with CUSIP as symbol. | Axys | Verified | Axys CI User Guide. |
| Duplicate matches can also occur when multiple CI translations match the same security. | Axys | Verified | Axys CI User Guide. |
| Account-specific translations are supported, but once used for a security, additional account-specific translations may be required and a global translation cannot be established for that same security. | Axys | Verified | Axys CI User Guide. |

Repository impact:

- Supports a dedicated “Duplicate Security Matching” subsection in the chapter.
- Supports warning that symbol alone is not safe as a unique cross-system match key.

### 8. AIA `.veh` to `sec.inf` layout behavior

| Finding | System | Confidence | Evidence |
|---|---:|---:|---|
| WealthTechs AIA Axys Vehicle Settings create a vehicle file with the layout of `sec.inf` from custodian information and save it as a `.veh` file in the Archive folder. | Axys | Verified in AIA context | WealthTechs AIA User Manual for Axys Users. |
| AIA Axys vehicle import options include `Update Existing & Add New`, `Add New`, `Replace Entire File`, and `None`. | Axys | Verified in AIA context | WealthTechs AIA User Manual for Axys Users. |
| AIA APX Vehicle Settings create a vehicle file with the layout of `sec.inf` from custodian information and save it as a `.veh` file in the Archive folder. | APX | Verified in AIA context | WealthTechs AIA User Manual for APX Users. |
| AIA APX vehicle import options include `Update Existing & Add New`, `Add New`, `Replace Entire File`, and `None`. | APX | Verified in AIA context | WealthTechs AIA User Manual for APX Users. |
| The `.veh`/Vehicle terminology is WealthTechs AIA-specific. | Axys/APX | High Confidence caution | The manuals describe AIA behavior, not an SS&C native data dictionary. |

Repository impact:

- Supports saying that third-party import tooling can generate/import files using a `sec.inf` layout.
- Does not establish the complete `sec.inf` field layout.

### 9. Axys conversion evidence around `sec.inf` and accrued-interest data

| Finding | System | Confidence | Evidence |
|---|---:|---:|---|
| Morningstar conversion guidance says that if Advent Axys `sec.inf` is provided, security names for user-defined securities can be converted. | Axys | Verified in conversion context | Morningstar Advent Axys conversion guide. |
| The same guidance says fields used to calculate accrued interest can be converted if specifically selected for export through the Advent Axys security file. | Axys | Verified in conversion context | Morningstar Advent Axys conversion guide. |
| The guide warns that if the Advent Axys `Use Security Type` box is selected, datapoints needed for accrued-interest calculation may not be exported to the Advent Axys security file. | Axys | Verified in conversion context | Morningstar Advent Axys conversion guide. |

Repository impact:

- Supports a cautious note that fixed-income/accrual-related fields may be present/exportable through Axys security file workflows when specifically selected.
- Exact field names remain **Unknown**.

### 10. REP / Report Writer / Replang extraction evidence

| Finding | System | Confidence | Evidence |
|---|---:|---:|---|
| The Salentica/Engage Advent Connector is installed on a client-side machine with Advent client tools, including `REP32.exe`. | Axys/APX | Verified for connector | Salentica/Engage Data Broker documentation. |
| The connector uses Advent standard reports and macros to generate data extracts. | Axys/APX | Verified for connector | Salentica/Engage Data Broker documentation. |
| The connector uses the REP32 engine and some RepLang scripting/macros. | Axys/APX | Verified for connector | Salentica/Engage Data Broker documentation. |
| SS&C/Advent Axys product materials state that Axys has hundreds of predefined reports and Axys Report Writer Pro. | Axys | Verified product-level | SS&C Advent Axys product page / brief. |

Repository impact:

- Supports REP32/Report Writer/RepLang as verified extraction mechanisms used by at least one connector.
- Still Unknown: exact standard REP report names for a complete Security Master extract and whether every security-master field is exposed.

### 11. APX public views / SQL / REST reporting evidence

| Finding | System | Confidence | Evidence |
|---|---:|---:|---|
| AdventGuru states APX users can access APX SQL Server data using Stored Accounting Functions, Public Views, SSRS, REST API, and other tools using that infrastructure. | APX | Medium Confidence | AdventGuru practitioner article/tag page. |
| AdventGuru states APX public views are useful but limited and do not expose all data users may want. | APX | Medium Confidence | AdventGuru practitioner article/tag page. |
| Exact public-view names and fields for APX securities were not found in public sources during this research pass. | APX | Unknown | Negative search result. |

Repository impact:

- APX public views can be mentioned only as a possible APX extraction mechanism, with the limitation that actual security view names/fields remain Unknown.

## Updated status of previously identified unknowns

| Unknown | Updated status | Notes |
|---|---|---|
| Axys `sec.inf` existence/use | Narrowed | Verified in CI context that `imex32.exe` exports Security (`sec.inf`) data from Axys. Complete layout remains Unknown. |
| Axys `type.inf` existence/use | Narrowed | Verified in CI context that `imex32.exe` exports Security Type (`type.inf`) data from Axys. Complete layout remains Unknown. |
| APX `sec.inf` existence/use | Narrowed | Verified in CI context that `apxix.exe` exports Security (`sec.inf`) data from APX. Complete layout remains Unknown. |
| APX `type.inf` existence/use | Narrowed | Verified in CI context that `apxix.exe` exports Security Type (`type.inf`) data from APX. Complete layout remains Unknown. |
| Exact Axys `sec.inf` field layout | Still Unknown | Public sources found confirm use/export but do not provide full native layout. |
| Exact Axys `type.inf` field layout | Still Unknown | Public sources found confirm use/export but do not provide full native layout. |
| Exact APX `sec.inf` field layout | Still Unknown | Public sources found confirm use/export but do not provide full native layout. |
| Exact APX `type.inf` field layout | Still Unknown | Public sources found confirm use/export but do not provide full native layout. |
| Official Axys IMEX object names for security/security type | Still Unknown | Public sources identify exported files and utility executable, not formal object names. |
| Official APX IMEX object names for security/security type | Still Unknown | Public sources identify exported files and utility executable, not formal object names. |
| Required fields for native security import/update | Still Unknown | AIA import options are known, but no full native import schema was found. |
| Formal Axys primary key | Still Unknown | CI behavior supports Symbol + Type as an identifier pair in integration context, but formal key not proven. |
| Formal APX primary key | Still Unknown | CI behavior supports Symbol + Type as an identifier pair in integration context, but formal SQL key not proven. |
| REP security-master report names | Still Unknown | REP32 extraction is verified for connectors, but no public security-master report name was found. |
| APX public-view security table/view names | Still Unknown | Public sources mention public views generally, but not security-specific view names/fields. |
| Complete security type / asset-class dictionary | Still Unknown | Example type codes found, but no complete dictionary found. |
| Fixed-income/bond security-master fields | Still Unknown | Morningstar confirms accrued-interest-related fields may be exported if selected, but field names are not provided. |
| Options/derivatives security-master fields | Still Unknown | No complete native fields found. |

## Evidence-strength summary for chapter writing

| Topic | Best current statement for repository chapter | Confidence |
|---|---|---:|
| Axys Security Master export | Axys CI uses `imex32.exe` to export Security (`sec.inf`) and Security Type (`type.inf`) data. | Verified in CI context |
| APX Security Master export | APX CI uses `apxix.exe` to export Security (`sec.inf`) and Security Type (`type.inf`) data. | Verified in CI context |
| Axys/APX identifier pair | CI uses Symbol + Type as the product-specific target identifier in security translation and missing-price workflows. | Verified in CI context |
| Native field dictionary | Complete native Axys/APX Security Master field dictionaries remain Unknown. | Unknown |
| Formal primary key | Symbol + Type is operationally important, but formal native primary key remains Unknown. | Unknown |
| Security type import | Practitioner source says security types cannot be imported through IMEX; treat as Medium Confidence and do not state as vendor fact. | Medium Confidence |
| REP extraction | REP32/standard reports/macros/RepLang are used by a connector to extract Axys/APX data. | Verified for connector |
| APX public views | APX public views and SQL-related reporting mechanisms exist, but security-specific view names and coverage remain Unknown. | Medium Confidence / Unknown |
| `.veh` files | WealthTechs AIA can create `.veh` files using `sec.inf` layout for Axys/APX import workflows. | Verified in AIA context |

## Recommended chapter updates based on this research

1. Add an “Import/Export Utility Evidence” table:
   - Axys: `imex32.exe`; exports `sec.inf` and `type.inf`; imports transactions, prices, and positions in CI context.
   - APX: `apxix.exe`; exports `sec.inf` and `type.inf`; imports transactions, prices, positions, and position lots in CI context.
2. Add a “CI Security Translation File Dictionary” for Axys and APX, clearly labeled as CI output, not native Security Master schema.
3. Add a “Missing Prices File Identifier Dictionary” showing `Symbol`, `Type`, `Name`, `WP Account`, and `Institution`, clearly labeled as CI output.
4. Update Unknowns:
   - Replace “Does Axys/APX use sec.inf/type.inf?” with “Complete layout and official object names Unknown.”
   - Keep native schema, primary key, and complete security-type dictionary Unknown.
5. Add a caution that AIA `.veh` files and CI translation files are integration artifacts and must not be promoted to native Advent schema.
6. Add APX-specific authentication/logging quirk for `apxix.exe` in CI context.
7. Add Axys/APX import setting options from AIA only under integration-specific behavior.

## Sources added / re-verified in this research pass

1. ByAllAccounts — `Custodial Integrator Installation Guide (Axys)`.
   - URL: `https://www.byallaccounts.net/Manuals/Custodial_Integrator/axys/CI_Installation_Guide.pdf`
   - Supports: `imex32.exe`, export of Security (`sec.inf`) and Security Type (`type.inf`), import workflow, `pospos32.exe`.
2. ByAllAccounts — `Custodial Integrator Installation Guide (APX)`.
   - URL: `https://www.byallaccounts.net/Manuals/Custodial_Integrator/apx/CI_Installation_Guide.pdf`
   - Supports: `apxix.exe`, export of Security (`sec.inf`) and Security Type (`type.inf`), APX import workflow, authentication/logging caution.
3. ByAllAccounts — `Custodial Integrator User Guide (Axys)`.
   - URL: `https://www.byallaccounts.net/Manuals/Custodial_Integrator/axys/CI_User_Guide.pdf`
   - Supports: Axys security matching, duplicate cases, translation resolution, missing-prices file dictionary, security-translations file dictionary.
4. ByAllAccounts — `Custodial Integrator User Guide (APX)`.
   - URL: `https://www.byallaccounts.net/Manuals/Custodial_Integrator/apx/CI_User_Guide.pdf`
   - Supports: APX security matching, APX Symbol/APX Security Type, translation precedence, reserved type-prefix exclusions, missing-prices file dictionary, security-translations file dictionary.
5. WealthTechs — `AIA User Manual for Axys Users`.
   - URL: `https://www.wealthtechs.com/files/AIADocumentation/Axys%20Guide%20-%20AIA%20User%20Manual%20For%20Axys%20Users.pdf`
   - Supports: `.veh` file using `sec.inf` layout, Axys import options.
6. WealthTechs — `AIA User Manual for APX Users`.
   - URL: `https://www.wealthtechs.com/files/AIADocumentation/APX%20Guide%20-%20AIA%20User%20Manual%20For%20APX%20Users.pdf`
   - Supports: `.veh` file using `sec.inf` layout, APX import options, APX export folder behavior.
7. Morningstar — `Converting Your Advent Axys Database into Morningstar Office`.
   - URL: `https://gladmainnew.morningstar.com/articles/tutorial/30/AdventAxys.pdf`
   - Supports: Axys `sec.inf` conversion of user-defined security names and accrued-interest-related fields when selected.
8. Salentica / Engage — `Data Broker - SS&C|Advent APX & Axys`.
   - URL: `https://engage.salentica.com/kb/article/247-data-broker-ss-c-advent-apx-axys/`
   - Supports: REP32, standard reports/macros, RepLang scripting/macros, version support for connector.
9. SS&C Advent — `Axys` product page / brief.
   - URL: `https://www.advent.com/solutions/axys/`
   - Supports: product-level Axys predefined reports and Report Writer Pro.
10. AdventGuru — Axys/APX integration and reporting articles.
   - URLs: `https://adventguru.com/category/portfolio-management-systems/axys/`, `https://adventguru.com/category/portfolio-management-systems/apx/`, `https://adventguru.com/2024/02/`, `https://adventguru.com/tag/apx/`
   - Supports: practitioner commentary on security type handling, security-master merging dependencies, public views/SQL/REST options, and Report Writer Pro/Replang usage.

## Deep IMEX Addendum Incorporated 2026-06-30

Source: `axys_imex_deep_research.md`.

Additional security-master points to carry into Chapter 04:

| Topic | Addendum | Confidence |
|---|---|---:|
| `sec.inf` / `type.inf` role | Public CI evidence supports `sec.inf` and `type.inf` as Axys reference inputs used to map external securities and generate valid transaction, position, and price imports. | Verified for CI workflow |
| Missing prices diagnostic | `MISSINGPRICES_yyyymmdd.csv` exposes `Symbol`, `Type`, `Name`, `WP Account`, and `Institution` in CI context. | Verified for CI output |
| Security translations diagnostic | `SECTRANSLATIONS_yyyymmdd.csv` exposes WebPortfolio identifiers, institution/account context, Axys Symbol, Type, Created, and Last Modified in CI context. | Verified for CI output |
| Candidate live-discovery fields | A live IMEX catalog should inspect symbol, security type, names, ticker, CUSIP, ISIN, SEDOL, currency, price multiplier, factor, coupon, maturity, classifications, and user-defined fields. | Discovery guidance |
| Critical caveat | These diagnostics and candidate fields do not prove an official universal Axys IMEX security-master schema. | Unknown / boundary |

## Deep Research Update Incorporated 2026-07-02

The 2026-07-02 addendum reinforces the existing conservative posture. Axys
`imex32.exe` and APX `apxix.exe` are verified in CI context as exporting
Security (`sec.inf`) and Security Type (`type.inf`) data, and APX CI requires
APX security type plus symbol for imported positions and transactions. Axys CI
security translation can map external identifiers to Axys symbol/type and can
encounter duplicate matches, so symbol-only joins remain unsafe.

WealthTechs AIA `.veh` files and AdvisorEngine XLS labels remain
integration/report artifacts, not native schemas. Morningstar conversion
evidence suggests Axys `sec.inf` can carry user-defined names and
accrued-interest-related fields when selected, but fixed-income field names
remain Unknown. APX security public-view names, formal native primary keys,
complete `sec.inf`/`type.inf` layouts, official Security/Security Type IMEX
object names, and complete security-type dictionaries remain Unknown.
