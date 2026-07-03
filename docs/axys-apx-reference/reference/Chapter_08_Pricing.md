# Chapter 08 — Pricing

Repository: AXYS / APX Reference Repository
Chapter: `docs/axys-apx-reference/reference/Chapter_08_Pricing.md`
Status: Technical reference chapter based on supplied research only
Governing specification: `AXYS_APX_REFERENCE_BLUEPRINT.md`, Version 2.0

---

## Related chapters
- [Chapter_01_Overview.md](Chapter_01_Overview.md) — provides the repository-wide map and evidence conventions.
- [Chapter_06_Holdings.md](Chapter_06_Holdings.md) — prices are used to value holdings and positions.
- [Chapter_10_Performance.md](Chapter_10_Performance.md) — pricing inputs drive performance and valuation results.
- [Chapter_12_Imex.md](Chapter_12_Imex.md) — price imports are one of the most common IMEX workflows.

## 1. Overview

Pricing is a core dependency for portfolio accounting, holdings valuation, reconciliation, performance measurement, and reporting. In Axys and APX environments, pricing evidence appears most clearly in integration workflows, price-file import/export behavior, holdings valuation reports, and operational pricing exceptions.

This chapter distinguishes four layers that should not be conflated:

| Layer | Description | Current evidence status |
|---|---|---:|
| Native Axys/APX pricing model | Internal storage, price keys, source hierarchy, price sets, stale-price rules, and field layout. | Mostly Unknown |
| IMEX / import-export pricing | Price files, import utilities, logs, update modes, and integration-controlled imports. | Partially supported |
| REP / reporting pricing | Report output that displays price, market value, or missing/stale-price information. | Partially supported |
| Integration-layer pricing | Custodial Integrator, AIA, price-source merge logic, calculated prices, missing-price files, and price-set workflows. | Best supported |

The strongest supplied evidence is integration-layer evidence. Complete native Axys/APX price schemas, official IMEX object names, APX public-view names, and standard REP price-report names remain **Unknown**.

### 1.1 Confidence labels

| Label | Meaning |
|---|---|
| Verified | Directly supported by the supplied research/source material. |
| High Confidence | Strongly supported by multiple supplied sources or standard portfolio-accounting practice, but not fully verified as native Axys/APX behavior. |
| Medium Confidence | Supported by credible third-party, consultant, conversion, or integration evidence, but not official native documentation. |
| Unknown | Not established by the supplied material. Do not promote to fact without additional documentation, sample exports, production evidence, or observed system behavior. |

---

## 2. Pricing Concepts

| Concept | Description | Axys | APX | Confidence |
|---|---|---:|---:|---:|
| Price date | Date for which a security price applies. | Implied by `*.pri` files and historical price logs; exact field Unknown. | Implied by AIA APX price-file naming such as `mmddyy_CDI.pri`; exact field Unknown. | Medium |
| Security price | Unit price used to value a security. | Price files are imported in CI workflow. | AIA imports price data into APX price files. | Medium |
| Price file | File containing prices for import/merge/update. | `*.pri` observed in CI workflow. | `mmddyy_CDI.pri` observed in AIA APX workflow. | Verified for integration workflows |
| Price source | Vendor, custodian, calculated, or other source provenance. | IDC, custodian, third-party, holding, and calculated price concepts appear in CI release notes. Native field Unknown. | `SourceId` appears in APX AIA price context. Native field Unknown. | Medium |
| Missing price | Price absent or unusable for processing. | Missing Price file behavior observed in CI release notes. | Native behavior Unknown. | Axys CI: Verified; APX: Unknown |
| Stale price | Price available but not current enough for processing. | Stale third-party and holding prices appear in CI release notes. | Unknown. | Axys CI: Medium |
| Calculated price | Price computed from units and market value when source prices are unavailable. | Documented in CI release notes. | Unknown. | Axys CI: Verified |
| Price set | Set or grouping of prices that may differ by custodian/source. | Unknown. | AIA APX Price Set Logic observed. | APX AIA: Verified |
| Price merge / trumping order | Process of selecting between multiple source prices. | `mergepri` consultant evidence; primary source precedence. | AIA APX custodian trumping order; `mergepri` consultant evidence. | Medium |
| Price precision | Number of decimal digits retained in output price. | CI release notes discuss calculated-price decimal precision. | Unknown. | Axys CI: Verified |
| FX rate | Currency conversion rate for non-base-currency valuation. | Axys has multicurrency capability; exact price/FX schema Unknown. | APX has multicurrency capability; exact price/FX schema Unknown. | Capability: Verified; schema: Unknown |

---

## 3. Axys Pricing

### 3.1 Product capability context

| Statement | Confidence | Notes |
|---|---:|---|
| Axys is marketed as portfolio reporting and accounting software. | Verified | Product-level capability only. |
| Axys supports performance measurement, reconciliation, multicurrency capabilities, and corporate-actions processing at a product-capability level. | Verified | Does not define pricing internals. |
| Native Axys price-file layouts, price keys, stale-price thresholds, and price-source hierarchy are not established by the supplied material. | Unknown | Requires vendor documentation or production samples. |

### 3.2 Axys price files and folders

| Artifact | Description | Confidence | Caveat |
|---|---|---:|---|
| `$pathpri` | Axys price-folder label from Custodial Integrator evidence. | Verified for CI workflow | Not necessarily complete Axys path model. |
| `*.pri` | Axys price files in `$pathpri`. | Verified for CI workflow | Complete layout Unknown. |
| `imexPrices.log` | Axys IMEX price-import log name/tab observed in CI evidence. | Verified for CI workflow | Exact log schema Unknown. |
| `imexPrices` tab(s) | CI View IMEX Logs may show one `imexPrices` tab per historical price day when multiple historical price days are delivered. | Verified for CI workflow | Integration UI behavior. |

### 3.3 Axys CI price import workflow

Custodial Integrator evidence supports the following Axys-oriented workflow:

```text
External source-data
        ↓
Custodial Integrator download / translation
        ↓
Axys security and type information used for mapping
        ↓
Transaction file + position file + price file
        ↓
Axys Import/Export utility / imex32.exe in CI context
        ↓
Price records imported if successful
        ↓
IMEX log review, including imexPrices.log
```

| Step | Evidence-supported statement | Confidence |
|---|---|---:|
| Source translation | CI uses Axys security information to generate output files. | Verified for CI |
| Price file generation | CI produces price files as part of data translation. | Verified for CI |
| IMEX import | Requested Transaction, Position, and Price files are imported into Axys using the Axys Import/Export utility. | Verified for CI |
| Log review | IMEX logs can be reviewed after import. | Verified for CI |
| Error handling | An open/in-use Axys price file can cause import failure; the error appears in `imexPrices.log`. | Verified for CI |

### 3.4 Axys calculated, missing, and stale price behavior

The most specific Axys pricing behavior in the supplied material comes from Custodial Integrator release notes. These statements are **CI release behavior**, not proof of native Axys pricing behavior.

| Statement | Confidence | Notes |
|---|---:|---|
| CI release notes mention `third party security price`, `holding price`, `calculated price`, `Missing Price file`, and `Price file`. | Verified for CI | Indicates multiple price-source states in the CI workflow. |
| A fixed bug involved third-party security price and holding price being unavailable or stale while calculated price was available and current; the security was incorrectly exported to Missing Price file. | Verified for CI | Missing/stale/calculated price routing behavior. |
| Corrected behavior exports the calculated price to the Price file. | Verified for CI | CI behavior. |
| CI 3.4 corrected a bug where calculated bond prices lost the last two decimal digits. | Verified for CI | Fixed-income pricing precision issue. |
| CI 3.4 outputs up to the maximum number of decimal digits set for the firm; default is four; support can configure greater precision. | Verified for CI | CI setting, not proven native Axys precision. |
| When no IDC price and no custodian price exist, CI can calculate output price using position units and market value, if available. | Verified for CI | Formula details and bond scaling convention remain Unknown. |
| Calculated price can be truncated to zero if output precision is too low and price is very small. | Verified for CI | Important audit condition. |

### 3.5 Axys price merge evidence

| Statement | Confidence | Caveat |
|---|---:|---|
| AdventGuru states exported Axys/APX price file formats are simple enough that users could write a merger, but Advent has a `mergepri` script command. | Medium | Consultant evidence. |
| `mergepri` allows specifying a destination and multiple sources. | Medium | Exact syntax Unknown. |
| The first source is primary. | Medium | Consultant evidence. |
| Prices in the first source file are not overwritten by secondary source prices. | Medium | Consultant evidence. |
| Supported versions, exact syntax, and error behavior for `mergepri` are Unknown. | Unknown | Needs script-command documentation or production testing. |

### 3.6 Axys native pricing unknowns

| Unknown | Needed evidence |
|---|---|
| Complete `.pri` file layout. | Sanitized `.pri` file, Axys file-layout documentation, or IMEX manual. |
| Whether one `.pri` file normally represents one date, source, currency, custodian, or another convention. | Production files and naming standards. |
| Native price key. | Vendor documentation or reproducible tests. |
| Whether Axys stores separate bid, ask, close, evaluated, clean, or dirty prices. | Vendor manual or price-file sample. |
| Axys price-source hierarchy. | Vendor documentation or integration configuration. |
| Native stale-price thresholds. | Vendor documentation, settings, or missing/stale price report. |
| Native missing-price report names. | REP/report catalog or screen/report sample. |
| Relationship between transaction price and historical price file price. | Transaction export plus price export for same dates. |
| Split-adjustment behavior for historical prices. | Split scenario with prices and holdings before/after. |

---

## 4. APX Pricing

### 4.1 Product capability context

| Statement | Confidence | Notes |
|---|---:|---|
| APX is marketed as an integrated portfolio and client management platform with accounting, reporting, and performance capabilities. | Verified | Product-level capability only. |
| APX supports multi-currency and multi-asset coverage at a product-capability level. | Verified | Exact pricing schema Unknown. |
| APX provides standard reports and flexible custom reporting at a product-capability level. | Verified | Exact price report catalog Unknown. |

### 4.2 APX AIA pricing settings

The strongest APX pricing evidence is from WealthTechs AIA for APX. These settings are AIA workflow behavior and should not be promoted to native APX internals without additional evidence.

| AIA APX setting / concept | Observed behavior | Confidence | Caveat |
|---|---|---:|---|
| Price File Update Logic | Determines how AIA imports price data while considering existing APX price data. | Verified for AIA workflow | Native APX internals Unknown. |
| Update Existing & Add New | Updates prices for vehicles currently in an APX price file and adds prices for new vehicles. | Verified for AIA workflow | AIA option. |
| Add New | Adds new vehicles to the price file if they do not exist in APX. | Verified for AIA workflow | AIA option. |
| Replace Entire File | Deletes existing prices currently in APX and replaces them with prices in the AIA import file. | Verified for AIA workflow | High-risk operational option; exact scope Unknown. |
| Clean Price File | `Remove Prices for Accounts Filtered` removes prices for securities held only in filtered accounts from the APX price file. | Verified for AIA workflow | AIA option; native APX status Unknown. |
| Price Set Logic | Option used only if client uses price sets in APX. | Verified for AIA workflow | Exact APX price-set model Unknown. |
| Create A Price For Each Custodian | Creates a price file specifically for each custodian; securities can be priced differently depending on custodian. | Verified for AIA workflow | AIA/APX context. |
| Merge Pricing Using Multi Custodian Setting Trumping Order | Default AIA option; assumes one price file per day for all custodians; first custodian has highest priority. | Verified for AIA workflow | Trumping order configured in Multi Custodian Settings. |
| Merge Pricing Using Multi Custodian Setting Trumping Order and Use Custom Price File Name | Uses custodian trumping order and custom price filename; example updates/adds/replaces one APX price file named like `mmddyy_CDI.pri`. | Verified for AIA workflow | Example-specific filename. |

### 4.3 APX price sets and custodian-specific pricing

| Implication | Treatment | Confidence |
|---|---|---:|
| APX environments may support or accommodate price sets. | Observed in AIA APX guide. | Verified for AIA setting; native APX model Unknown |
| Pricing source/custodian can materially affect valuation. | Strongly implied by custodian-specific price-file logic. | Medium |
| A migration/audit tool should preserve price source or price set when available. | Implementation recommendation based on observed workflow. | High Confidence |
| Native APX price-set schema is not established. | Preserve as Unknown. | Unknown |

### 4.4 APX price source field evidence

| Field / Label | Context | Axys | APX | Confidence | Caveat |
|---|---|---:|---:|---:|---|
| `SourceId` | Price-source label shown in AIA APX price context. | No | Yes | Verified in AIA/APX context | Not proven native APX database, IMEX, or REP field name. |

### 4.5 APX native pricing unknowns

| Unknown | Needed evidence |
|---|---|
| Native APX price tables, views, or stored accounting functions. | APX schema/public-view docs or sanitized SQL output. |
| Native APX price key and whether source/price set is part of the key. | Vendor docs or production observation. |
| Native APX price-file layout for `.pri` imports. | AIA archive sample, APX import spec, or vendor docs. |
| Whether APX price files are persisted files, import staging files, compatibility artifacts, or AIA-generated exchange files. | Vendor documentation or installation evidence. |
| Whether `mmddyy_CDI.pri` is AIA-specific, APX-native-compatible, or both. | AIA examples plus APX import documentation. |
| Exact scope of `Replace Entire File`. | AIA/vendor documentation or test system. |
| Native missing-price and stale-price reports. | APX report guide, REP catalog, or sample reports. |
| Native APX field names for price date, price, source, currency, and price set. | Vendor data dictionary, public views, or sample exports. |

---

## 5. Axys / APX Comparison

| Area | Axys | APX | Confidence |
|---|---|---|---:|
| Observed price files | `*.pri` in `$pathpri` folder from CI evidence. | AIA example `mmddyy_CDI.pri`; price files and price sets from AIA APX guide. | Verified for integration contexts |
| Import utility | Axys Import/Export utility; `imex32.exe` in CI context. | APX import/export appears as `APXIX.exe` in related research; AIA pricing imported to APX price files. | Verified for integration contexts; native details Unknown |
| Price import update modes | Native modes Unknown; CI imports price files and logs errors. | AIA documents Update/Add/Replace modes. | APX AIA: Verified; Axys native: Unknown |
| Price source merging | `mergepri` primary-source precedence from consultant evidence. | AIA multi-custodian trumping order and `mergepri` consultant evidence. | Medium |
| Custodian-specific pricing | Unknown from supplied Axys sources. | AIA APX supports creating a price for each custodian. | APX AIA: Verified |
| Missing Price file | CI release notes mention Missing Price file. | Unknown. | Axys CI: Verified; APX: Unknown |
| Calculated prices | CI can calculate price using units and market value when no IDC/custodian price exists. | Unknown. | Axys CI: Verified |
| Price precision | CI release notes discuss default four decimals and configurable calculated-price precision. | Unknown. | Axys CI: Verified |
| Direct database price access | Not applicable in same way; Axys direct files are version-risky. | APX SQL/public-view options mentioned in supplied research; exact price views Unknown. | Medium for APX access option; fields Unknown |
| REP/report price access | Report Writer Pro/Replang available; exact price report names Unknown. | Standard/custom reporting available; exact price report names Unknown. | Medium/Unknown |

---

## 6. IMEX Pricing Coverage

### 6.1 Observed IMEX-adjacent pricing facts

| Statement | Axys | APX | Confidence | Caveat |
|---|---:|---:|---:|---|
| CI uses Axys IMEX to import price files. | Yes | No | Verified for CI | Not a complete IMEX object dictionary. |
| CI retains IMEX price logs. | Yes | No | Verified for CI | Log schema Unknown. |
| IMEX price import can fail if the target price file is open/in use. | Yes | No | Verified for CI | Operational quirk. |
| Multiple historical price days can create multiple `imexPrices` log tabs. | Yes | No | Verified for CI | CI UI behavior. |
| Exact native IMEX price object name is established. | No | No | Unknown | Need official IMEX docs or samples. |
| Exact native IMEX price field list is established. | No | No | Unknown | Need official IMEX docs or samples. |

### 6.2 Candidate IMEX price documentation structure

Until official IMEX material or production samples are supplied, the pricing IMEX
structure remains a documentation target rather than a verified native schema:

| Attribute | Current status | Needed evidence |
|---|---|---|
| IMEX price object name | Unknown | Axys/APX IMEX manual, screenshots, logs, or control files. |
| Direction | Axys CI price import observed; APX AIA price import observed; native export Unknown. | IMEX docs and export samples. |
| Required key fields | Unknown | Sample `.pri` or IMEX field dictionary. |
| Price date field | Unknown | Sample export/import. |
| Security identifier fields | Unknown; symbol/type likely related in security contexts but not proven for price files. | Price-file samples. |
| Price value field | Unknown official name. | Price-file samples. |
| Price source field | Unknown; `SourceId` observed only in AIA/APX context. | APX docs/export. |
| Currency field | Unknown. | Multi-currency price sample. |
| Price set field | Unknown. | APX price-set docs/export. |
| Null/missing behavior | Partially observed in CI release notes. | IMEX manual plus logs. |
| Error logs | `imexPrices.log` observed in CI. | Actual successful and failed log samples. |

### 6.3 Axys IMEX-adjacent workflow example

```text
$pathpri
    *.pri                  # observed Axys price files; exact naming convention Unknown

CI / IMEX workflow
    Source prices
        ↓
    Custodial Integrator translation
        ↓
    Price file generated
        ↓
    Axys Import/Export utility imports price file
        ↓
    imexPrices.log reviewed
```

Classification: `*.pri`, `$pathpri`, and `imexPrices.log` are Verified for the CI workflow. Exact file naming, file layout, and native object name remain Unknown.

---

## 7. REP / Report Pricing Coverage

### 7.1 REP context

Supplied REP research supports that Axys/APX extraction can use REP32, Report Writer Pro, RepLang, standard reports, macros, and custom reports. Pricing-specific REP report names were not established.

| Statement | Axys | APX | Confidence |
|---|---:|---:|---:|
| Axys/APX users can create reports using Report Writer Pro or Replang source. | Yes | Yes | Medium Confidence |
| Data Broker connector uses standard reports/macros and REP32 for Axys/APX extraction. | Yes | Yes | Verified for connector |
| Exact standard price report names are known. | No | No | Unknown |
| Exact REP price fields are known. | No | No | Unknown |
| Whether REP reports expose stored price values or calculated valuation outputs is known. | No | No | Unknown |

### 7.2 Price-related reports or sources to request

| Desired report/source | Why it matters |
|---|---|
| Standard price list report | Confirms report name, parameters, and output fields. |
| Missing price report | Confirms missing-price detection logic and visible fields. |
| Stale price report | Confirms stale threshold and source/date fields. |
| Holdings valuation report | Confirms how price appears with quantity and market value. |
| Fixed-income valuation report | Confirms price, factor, accrued interest, market value, and income fields. |
| APX public-view query for prices | Confirms APX schema/view fields and source/price-set behavior. |
| REP source (`.rep`) for price reports | Confirms RepLang variables and calculation behavior. |

---

## 8. Field Dictionary

This table is a conservative research catalog. It is **not** a verified native Axys/APX price data dictionary.

| Field / Label | Definition / meaning | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---:|
| `*.pri` | Price file extension observed in Axys/APX price-file workflows. | Yes | Yes in AIA APX example | Related | No | Verified for integration contexts |
| `$pathpri` | Axys price-folder label from CI evidence. | Yes | No | Related | No | Verified for CI Axys |
| `imexPrices.log` | Axys IMEX price-import log name/tab observed in CI evidence. | Yes | No | Yes | No | Verified for CI Axys |
| `Price File` | Output/import target for prices. | Yes | Yes | Related | Unknown | Verified for integration contexts |
| `Missing Price file` | CI output/category for securities without usable price, unless calculated price is available/current. | Yes | Unknown | Related | Unknown | Verified for CI Axys |
| `third party security price` | Price source type mentioned in CI release notes. | Yes in CI context | Unknown | Unknown | Unknown | Verified for CI Axys |
| `holding price` | Price source/state mentioned in CI release notes. | Yes in CI context | Unknown | Unknown | Unknown | Verified for CI Axys |
| `calculated price` | Price derived by CI, including from units and market value where custodian/IDC price is unavailable. | Yes in CI context | Unknown | Unknown | Unknown | Verified for CI Axys |
| `IDC price` | Pricing-source reference in CI release notes. | Yes in CI context | Unknown | Unknown | Unknown | Verified for CI Axys |
| `SourceId` | Price-source label observed in prior AIA/APX price context. | No | Yes | Unknown | Unknown | Verified for AIA/APX context only |
| `Price File Update Logic` | AIA/APX setting controlling update/add/replace behavior. | No | Yes | Related | No | Verified for AIA/APX |
| `Price Set Logic` | AIA/APX setting for price sets and custodian-specific price behavior. | No | Yes | Related | No | Verified for AIA/APX |
| `Clean Price File` | AIA/APX setting removing prices for securities held only in filtered accounts. | No | Yes | Related | No | Verified for AIA/APX |
| Price date | Date of price. | Unknown official field | Unknown official field | Unknown | Unknown | High as concept; native Unknown |
| Security identifier | Security whose price is supplied. | Unknown official field | Unknown official field | Unknown | Unknown | High as concept; native Unknown |
| Price value | Numeric price. | Unknown official field | Unknown official field | Unknown | Unknown | High as concept; native Unknown |
| Currency | Currency of price. | Unknown | Unknown | Unknown | Unknown | Medium concept; native Unknown |
| Price source | Source/vendor/custodian/price-set provenance. | CI mentions several sources; native field Unknown. | `SourceId` observed in AIA/APX context; native field Unknown. | Unknown | Unknown | Medium |
| Price set | APX price grouping or source set. | Unknown | AIA APX setting | Unknown | Unknown | Medium for APX AIA; native Unknown |
| Factor | Fixed-income/security factor. | Unknown exact behavior | Unknown | Unknown | Unknown | Conceptual; system-specific Unknown |
| Price multiplier | Multiplier used in valuation. | Unknown | Unknown | Unknown | Unknown | Conceptual; system-specific Unknown |
| Accrued interest | Fixed-income income accrual; relationship to price unclear. | Conversion research suggests `sec.inf` can include accrued-interest-related fields; price role Unknown. | Unknown | Unknown | Unknown | Medium concept; system-specific Unknown |

---

## 9. Processing Behavior

### 9.1 Price import/update patterns

| Pattern | Axys | APX | Confidence | Notes |
|---|---|---|---:|---|
| Append/update existing prices | Native behavior Unknown; CI imports price files. | AIA APX `Update Existing & Add New`. | APX AIA: Verified; Axys: Unknown | Need native Axys docs. |
| Add new only | Unknown. | AIA APX `Add New`. | APX AIA: Verified | Useful migration/audit mode. |
| Replace entire file | Unknown. | AIA APX `Replace Entire File`. | APX AIA: Verified | High-risk option; exact scope Unknown. |
| Merge with source priority | AdventGuru `mergepri`. | AIA APX custodian trumping order; AdventGuru `mergepri`. | Medium | Need script docs. |
| Create per-custodian price | Unknown. | AIA APX option. | APX AIA: Verified | Price-set/source-specific valuation risk. |
| Remove filtered-account prices | Unknown. | AIA APX Clean Price File. | APX AIA: Verified | Can create intentional omissions. |
| Calculate price from units and market value | CI release notes. | Unknown. | Axys CI: Verified | Fixed-income/missing-price fallback. |

### 9.2 Pricing dependency graph

```text
Security Master
    ├── security symbol / type / identifier
    ├── issue currency / pricing currency
    ├── security type / asset class
    ├── fixed-income terms / factor / accrued-interest fields (exact fields Unknown)
    └── price multiplier (exact Axys/APX behavior Unknown)

Transactions
    ├── transaction price
    ├── quantity
    ├── trade / settlement dates
    └── cash effects

Holdings / Positions
    ├── quantity
    ├── market value
    ├── units and market value may support calculated price in CI
    └── stale or missing holding price can affect CI output classification

Price Files
    ├── imported price
    ├── source / custodian / price set
    ├── calculated price fallback
    └── missing/stale price exception handling

Performance / Reports
    ├── beginning and ending market values
    ├── income and cash flows
    ├── valuation-date prices
    └── historical restatement risk if prices change
```

---

## 10. Version Differences and Release-Note Evidence

| System | Version / context | Pricing-related statement | Confidence | Caveat |
|---|---|---|---:|---|
| CI for Axys | V3.19.001 release note | Bond-price calculation correction noted in CI release evidence. | Verified for CI release | Not native Axys release note. |
| CI for Axys | V3.18.001 release note | Internal price-table source column expanded to six characters. | Verified for CI release | CI/internal table behavior; native `.pri` layout not proven. |
| CI for Axys | V3.17.001 release note | `cashDivCalculateQuantity` appears in CI release evidence. | Verified for CI release | Dividend/calculation setting context; pricing impact should be validated before use. |
| CI for Axys | V3.11.001 release note | Corrected case where current calculated price was incorrectly exported to Missing Price file instead of Price file when third-party security price and holding price were unavailable/stale. | Verified for CI release | Not native Axys release note. |
| CI for Axys | V3.4 release note | Bond calculated prices were truncating last two decimals; corrected to output up to configured maximum decimal digits. | Verified for CI release | Not native Axys release note. |
| CI for Axys | V3.4 release note | Default calculated-price decimal precision is four; technical support can configure greater precision. | Verified for CI release | CI setting. |
| CI for Axys | V3.4 release note | When no IDC price and no custodian price exist, output price is calculated from units and market value if available. | Verified for CI release | CI fallback. |
| APX AIA | Public manual context | Price File Update Logic and Price Set Logic documented. | Verified for AIA manual | AIA workflow-specific. |
| APX | APX v1.x–v4.x consultant evidence | APX maintained IMEX functionality but eliminated fixed-format file generation. | Medium | Interface-version evidence; pricing-specific field impact Unknown. |
| Axys | Axys 3.7 to 3.8 consultant evidence | File conversion changed some file formats; direct file access is risky. | Medium | Applies generally to files; exact `.pri` impact Unknown. |

---

## 11. Known Issues / Quirks

| Quirk | System | Description | Confidence | Audit implication |
|---|---|---|---:|---|
| Price file open/in use can fail import | Axys CI | Open Axys price file can cause import failure and `imexPrices.log` error. | Verified for CI | Check logs and rerun after closing file. |
| Current calculated price may be misclassified in older CI versions | Axys CI | Bug fixed where calculated price went to Missing Price file rather than Price file. | Verified for CI release | Version-aware missing-price review. |
| Bond calculated price truncation | Axys CI | Older CI could truncate calculated bond-price decimals. | Verified for CI release | Audit low-price/high-par fixed income carefully. |
| Calculated price can be based on units and market value | Axys CI | Fallback when IDC/custodian price unavailable. | Verified for CI release | Distinguish vendor price from derived price. |
| Price source precedence | Axys/APX | `mergepri` / custodian trumping order can preserve primary source price. | Medium | Preserve source hierarchy in audits. |
| Replace entire price file | APX AIA | AIA option deletes existing prices and replaces them with import file prices. | Verified for AIA | High-risk control; require backup/review. |
| Remove filtered-account prices | APX AIA | Prices for securities held only in filtered accounts may be omitted. | Verified for AIA | Missing prices may be intentional but must be traceable. |
| Same security priced differently by custodian | APX AIA | AIA option permits custodian-specific price files. | Verified for AIA | Valuation/reporting may differ by custodian/source. |
| Interlisted duplicate cleanup | APX AIA | AIA can remove unnecessary duplicate holdings and prices for interlisted-security workflows. | Verified for AIA | Requires holdings/pricing context; not a native APX rule. |
| FX replace risk | APX AIA | FX File Update Logic includes update/add/replace-style modes in AIA evidence. | Verified for AIA | Exact FX-file layout and native APX behavior remain Unknown. |
| Price-source-width risk | Axys CI | Six-character source-column enhancement means source truncation should be version-aware. | Verified for CI release | CI/internal table behavior; native `.pri` source-width Unknown. |
| Direct file access risk | Axys | Consultant evidence warns Axys file formats can change across versions. | Medium | Prefer IMEX/REP; version-control file readers. |

---

## 12. Audit Rules

These are candidate audit rules based on supplied research and general accounting controls. They are not native Axys/APX rules unless explicitly marked as observed.

| Rule ID | Name | Description | Required inputs | Detection logic | Severity | Confidence |
|---|---|---|---|---|---|---:|
| PRICE-001 | Missing Price | Security with nonzero holding lacks a price for valuation date. | holdings, security, price date, price file/report | holding_quantity != 0 and no price for valuation date | High | High concept; native implementation Unknown |
| PRICE-002 | Stale Price | Price date is older than allowed threshold for valuation date. | security, price date, valuation date, threshold | valuation_date - price_date > threshold | Medium/High | High concept; threshold Unknown |
| PRICE-003 | Zero Price | Price is zero or near zero for non-cash/non-defaulted security. | security type, price | price <= configured_minimum | High | High concept |
| PRICE-004 | Calculated Price Flag | Price was derived from units and market value rather than vendor/custodian price. | price source, position units, market value | source == calculated or price = MV / units | Medium | Verified for CI calculated-price concept |
| PRICE-005 | Price Source Precedence Violation | Lower-priority source overwrote higher-priority source. | price source order, price file sequence | secondary source changed primary source price | High | Medium; based on merge/trumping evidence |
| PRICE-006 | Price File Replace Risk | Replace-entire-file option used without control evidence. | import log, mode, backup flag | mode == Replace Entire File and backup/review missing | High | Verified AIA mode; audit rule recommendation |
| PRICE-007 | Custodian-Specific Price Divergence | Same security/date has different prices by custodian/source. | security, date, source/custodian, price | max(price) - min(price) > tolerance | Medium | Verified AIA possibility; tolerance Unknown |
| PRICE-008 | Price File Lock Failure | Price import failed because file was open/in use. | IMEX log | log contains file-in-use or price import failure | Medium | Verified CI example |
| PRICE-009 | Split Price Discontinuity | Large price change coincides with split but quantity/price ratio is inconsistent. | prices, split ratio, holdings | price ratio inconsistent with split ratio | High | Industry practice; native behavior Unknown |
| PRICE-010 | Bond Price Precision Loss | Fixed-income calculated price appears truncated or rounded beyond tolerance. | bond price, precision, units, market value | abs(price - MV/units) > tolerance due to truncation | Medium | Verified CI historical issue |
| PRICE-011 | Filtered Account Price Omission | Price omitted because only filtered accounts hold the security. | filter settings, holdings, price file | security held only in filtered accounts and omitted | Low/Medium | Verified AIA setting |
| PRICE-012 | Transaction Price vs Market Price Outlier | Trade price differs materially from market close/evaluated price. | transaction price, date price, tolerance | abs(trade_price - close_price) / close_price > tolerance | Medium | Recommendation / industry practice |
| PRICE-013 | FX Rate Missing for Foreign Price | Foreign-currency price lacks FX rate for valuation/reporting. | price currency, base currency, FX rate | price_currency != base_currency and FX missing | High | High concept; native fields Unknown |
| PRICE-014 | Negative Price | Price less than zero for security types where negative prices are invalid. | security type, price | price < 0 and not allowed | High | High concept |
| PRICE-015 | Price Multiplier Mismatch | Market value not consistent with quantity, price, multiplier, and factor. | quantity, price, multiplier, factor, market value | MV != quantity × price × multiplier × factor within tolerance | High | High concept; system fields Unknown |

---

## 13. Examples

### 13.1 APX AIA price-file update mode

```text
Process date: 2026-06-29
Mode: Update Existing & Add New
Security: ABC
Existing price: 100.00
Incoming price: 101.25
Result: existing price updated; new securities added if not present
```

Classification: Conceptual example based on AIA APX mode name. Exact APX internal result and price-key scope are Unknown.

### 13.2 APX AIA multi-custodian trumping order

```text
Custodian order:
    0001  highest priority
    0002  lower priority

Incoming prices for same security/date:
    Custodian 0001: 25.10
    Custodian 0002: 25.05

Expected AIA merge concept:
    25.10 selected because 0001 is higher priority.
```

Classification: Medium Confidence based on AIA APX guide's custodian trumping-order description. Exact tie-break and key fields are Unknown.

### 13.3 Axys CI calculated price from position data

```text
No IDC price available.
No custodian price available.
Position units: 100,000
Position market value: 98,750
Calculated output price concept: market value / units = 0.9875
```

Classification: Verified for the CI release-note concept that price can be calculated from units and market value if available. Exact formula, bond scaling convention, factor treatment, and price multiplier behavior are Unknown.

### 13.4 Axys CI price-file lock

```text
User has an Axys price file open.
CI attempts price import through the Axys Import/Export utility.
Import fails.
User reviews imexPrices.log to identify the error.
```

Classification: Verified for ByAllAccounts CI workflow.

---

## 14. Migration and Implementation Guidance

| Guidance | Rationale | Confidence |
|---|---|---:|
| Preserve price date, source, custodian, price set, and currency whenever available. | Source/custodian and price-set logic can affect APX valuation; price source hierarchy can affect Axys/APX merge behavior. | High Confidence |
| Do not assume Axys `*.pri` and APX `*.pri` files have identical layouts. | Evidence shows similar file extensions in integration contexts, not identical native schemas. | High Confidence caution |
| Do not infer native price fields from AIA or CI UI labels without sample files or vendor docs. | Integration labels may not equal native fields. | High Confidence caution |
| Treat calculated prices as distinct from vendor/custodian prices. | CI can calculate prices from units and market value. | Verified for CI concept |
| Treat `Replace Entire File` as a high-risk APX AIA operation. | It can delete existing prices and replace them with import data. | Verified for AIA setting |
| Prefer IMEX/REP/exported reports over direct raw file readers when possible. | Consultant evidence warns Axys file formats can change between versions. | Medium |
| Validate market value using quantity × price × factor × multiplier only when the required fields and conventions are known. | Factor/multiplier behavior is not established from supplied material. | High Confidence caution |

---

## 15. References

### 15.1 Supplied repository materials

| Reference | Use in this chapter |
|---|---|
| `AXYS_APX_REFERENCE_BLUEPRINT.md` | Governing specification, chapter template, confidence-label discipline. |
| `../evidence/Research_08_Pricing.md` | Primary source for pricing-specific Axys/APX, IMEX, REP, field, processing, quirk, and unknown material. |
| `../evidence/Research_12_IMEX.md` | Supporting source for IMEX terminology, Axys folders/files, `*.pri`, `$pathpri`, `imexPrices.log`, IMEX log behavior, direct file access cautions, and REP32/Replang context. |
| `../evidence/Research_13_REP.md` | Supporting source for REP/RepLang, Report Writer Pro, REP32, standard reports/macros, and report-based extraction. |
| `../evidence/Research_04_Security_Master.md` | Supporting source for security master dependencies, `SourceId` in APX price context, `sec.inf`, `type.inf`, and APX public-view cautions. |
| `../evidence/Research_05_Transactions.md` | Supporting source for transaction price dependency, direct-file-access caution, price/holdings/performance dependencies. |
| `../evidence/Research_06_Holdings.md` | Supporting source for holdings valuation dependencies and report-visible fields such as price, quantity, and market value. |

### 15.2 Source material summarized in supplied research

| Source | Pricing relevance |
|---|---|
| ByAllAccounts / Morningstar, `Custodial Integrator User Guide` for Axys | CI price-file import workflow, `$pathpri`, `*.pri`, IMEX logs, price file lock behavior. |
| ByAllAccounts / Morningstar, `Custodial Integrator Release Notes` for Axys | Calculated price, missing/stale price routing, bond calculated-price precision, IDC/custodian price fallback. |
| WealthTechs, `AIA User Manual for APX Users` | APX AIA Price File Update Logic, Clean Price File, Price Set Logic, custodian-specific prices, trumping order, custom `.pri` filename. |
| WealthTechs, `AIA User Manual for Axys Users` | Adjacent Axys AIA processing and import context. |
| AdventGuru Axys/APX price-file and IMEX articles | `mergepri`, direct file access cautions, APX IMEX/version behavior. |
| SS&C Advent Axys product page | Product-level capability context. |
| SS&C Advent Portfolio Exchange product page | Product-level APX capability context. |
| UNAPEN PriceFusion for APX | Ecosystem evidence for APX pricing/reference-data automation, not native APX behavior. |
| SS&C Advent APX Reports Guide (`REP_APX.pdf`) | Report guide discovered in research, but pricing-specific report catalog not established from supplied material. |

---

## 16. Known Unknowns

### 16.1 Highest-priority unknowns

| ID | Unknown | Why it matters | Evidence needed |
|---|---|---|---|
| PU-001 | Complete Axys `.pri` file layout. | Required for importers, validators, migration tools. | Sanitized price file, file layout manual, or IMEX manual. |
| PU-002 | Complete APX price import/export layout. | Required for APX price integration. | AIA/APX archive price files, APX IMEX docs, or public views. |
| PU-003 | Native IMEX price object names for Axys and APX. | Required for Chapter 12 and this chapter's field dictionaries. | Vendor IMEX object catalog or sample control files. |
| PU-004 | Native price key fields. | Needed for dedupe, replacement, and update logic. | Vendor docs or reproducible tests. |
| PU-005 | Native price source / price set model. | Needed to avoid overwriting correct valuations. | APX price-set docs, Axys source docs, sample exports. |
| PU-006 | Standard REP report names for prices, missing prices, stale prices. | Required for REP coverage. | REP catalog, `.rep` source, report output samples. |
| PU-007 | APX public-view/stored accounting function fields for prices. | Needed for APX SQL extraction. | View schema or query output. |
| PU-008 | Primary exchange close handling. | Needed for audit rules around trade-price vs market-price comparison. | Vendor docs or price-source setup screenshots. |
| PU-009 | Fixed-income evaluated price and factor behavior. | Critical for bonds and mortgage-backed securities. | Fixed-income security sample, price sample, valuation report. |
| PU-010 | Price multiplier handling. | Critical for market value calculation. | Security master docs and holdings valuation examples. |
| PU-011 | FX-rate storage and price currency handling. | Needed for multicurrency valuation. | FX rate file/export/report docs. |
| PU-012 | Corporate-action price adjustment behavior. | Needed for split/audit rules. | Split scenario before/after prices and holdings. |
| PU-013 | Stale-price threshold and exception process. | Needed for production audit rules. | Reports, settings, operations procedures. |
| PU-014 | Whether transaction prices are independent from daily price files. | Needed for transaction audit and performance restatement analysis. | Transaction export plus price export for same dates. |
| PU-015 | Whether historical price changes trigger stored performance regeneration. | Needed for Chapter 10 dependency. | Performance docs, test scenario, production observations. |

### 16.2 Additional unknowns

| Unknown | Needed evidence |
|---|---|
| Price-file naming convention by date/source/currency. | Multiple real price files. |
| Whether cash equivalents require explicit prices. | Security/pricing docs and sample holdings. |
| Whether zero prices are allowed for defaulted securities or options. | Vendor docs and exception reports. |
| Whether prices are stored as clean or dirty prices for fixed income. | Fixed-income valuation report and price file. |
| Whether accrued interest is stored with price, security, transaction, or calculated at report time. | Fixed-income docs and report source. |
| Whether APX price sets can coexist for the same security/date/source. | APX docs/test data. |
| Whether Axys supports multiple price sources per date natively. | Vendor docs or sample files. |
| Whether missing-price files are generated by Axys, CI, APX, or third-party workflows. | Logs/manuals. |
| Whether `mergepri` is available in all Axys/APX versions. | Script-command documentation and version notes. |

## 17. Deep IMEX Update

The deep IMEX research strengthens the CI-observed price workflow but keeps
native price schemas Unknown.

| Topic | Chapter treatment | Confidence |
|---|---|---:|
| `.pri` append workflow | CI appends prices to prior-business-day price files in `$pathpri`. | Verified for CI |
| No overwrite behavior | CI does not replace an existing populated price for the same security/day in the price file. | Verified for CI |
| Historical price logs | Multiple historical price days may produce one `imexPrices` tab per day. | Verified for CI |
| Price preview fields | Symbol, Price, Source, Price Date, and Price As-Of Date appear in CI preview evidence. | Verified for CI |
| Candidate live-discovery fields | Symbol, type, price date, price, price source, currency, factor, quote multiplier, and price-set/source provenance. | Discovery guidance |

These points do not prove the official IMEX price object name, `.pri` field
layout, or APX price public-view schema.
