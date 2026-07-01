# Research Notes — 08-Pricing.md

Repository area: `research/`  
Prepared: 2026-06-29  
Target chapter: `docs/08-Pricing.md`  
Scope: Axys, APX, IMEX, REP, field names, report names, processing behavior, version differences, implementation quirks, examples, references.

## Governing specification

This research file follows `AXYS_APX_REFERENCE_BLUEPRINT.md` Version 9 Actionable Research-First Repository Constitution.

The Blueprint requires:

- research output for topics where adequate source material is not yet available for chapter drafting;
- source summaries;
- extracted technical facts;
- explicit confidence levels;
- separate Axys and APX behavior where behavior differs;
- explicit Unknowns rather than invented details;
- IMEX and REP treatment as first-class topics;
- preservation of source caveats and contradictions.

## Confidence labels used here

| Label | Meaning in this research file |
|---|---|
| Verified | Directly supported by reviewed source text, source screenshot, or prior repository research already supported by source material. |
| High Confidence | Strongly supported by multiple sources or by standard portfolio-accounting practice, but not fully verified from official Axys/APX pricing documentation. |
| Medium Confidence | Supported by one or more credible third-party integration, consultant, conversion, or product sources; useful but not definitive native behavior. |
| Unknown | Not established from available sources. Do not promote to repository fact without vendor documentation, sample exports, production evidence, or observed system behavior. |

## Research question

Collect factual information about pricing in Axys and APX for future use in `08-Pricing.md`, focusing on:

- price architecture;
- price sources;
- closing prices;
- primary exchange close;
- evaluated prices;
- fixed-income pricing;
- factor pricing;
- accrued interest;
- price multipliers;
- FX rates;
- missing prices;
- stale prices;
- split-adjusted prices;
- corporate-action price effects;
- security-master dependencies;
- holdings dependencies;
- performance dependencies;
- IMEX price imports/exports;
- REP/report price exposure;
- field names;
- report names;
- processing behavior;
- version differences;
- implementation quirks;
- examples;
- references.

## Executive summary

| Topic | Research finding | Axys | APX | Confidence | Source basis |
|---|---|---:|---:|---|---|
| Pricing is a core accounting dependency. | Portfolio accounting systems use security prices to value holdings, calculate market value, support performance, and detect stale/missing data. | Yes | Yes | High Confidence | Industry practice; supported indirectly by CI workflows importing position and price files. |
| Axys CI imports price files through IMEX. | ByAllAccounts Custodial Integrator for Axys states CI produces transaction, position, and price files and imports them into Axys using the Axys Import/Export utility. | Yes | No | Verified for CI workflow | ByAllAccounts CI Axys User Guide. |
| `*.pri` files are observed Axys price files. | Prior IMEX research and ByAllAccounts Axys evidence identify `$pathpri` as the Axys price folder and `*.pri` as price files. | Yes | No | Verified for CI workflow | Research_12_IMEX; ByAllAccounts CI Axys User Guide. |
| Price file locks can break Axys import. | ByAllAccounts Axys evidence says if an Axys price file is open/in use, price import may fail and the error appears in `imexPrices.log`. | Yes | No | Verified for CI workflow | ByAllAccounts CI Axys User Guide; Research_12_IMEX. |
| Multiple historical price days can produce multiple IMEX price-log tabs. | Prior IMEX research says if prices are requested for more than the prior business day, CI shows one `imexPrices` tab per historical day delivered. | Yes | No | Verified for CI workflow | Research_12_IMEX. |
| APX AIA supports price file update logic. | WealthTechs AIA APX guide documents `Update Existing & Add New`, `Add New`, and `Replace Entire File` options for importing price data while considering existing APX price data. | No | Yes | Verified for AIA workflow | WealthTechs AIA APX User Manual. |
| APX AIA supports price set logic. | WealthTechs AIA APX guide documents custodian-specific price files and custodian trumping-order merge options, including custom price-file names such as `mmddyy_CDI.pri`. | No | Yes | Verified for AIA workflow | WealthTechs AIA APX User Manual. |
| APX AIA can remove prices for filtered accounts. | WealthTechs AIA APX guide documents a Clean Price File option to remove prices for securities held only in filtered accounts. | No | Yes | Verified for AIA workflow | WealthTechs AIA APX User Manual. |
| `SourceId` appears in an APX price import context. | Prior Security Master research notes identify `SourceId` as an APX price source field shown in AIA APX screenshot/example, not a security-master field. | No | Yes | Verified, non-security-master context | Research_04_Security_Master addendum. |
| Price merging exists as an Axys/APX scripting workflow. | AdventGuru states exported price file formats are simple enough to merge and that Advent has a `mergepri` script command; first source is primary and is not overwritten by secondary source prices. | Yes | Yes | Medium Confidence | AdventGuru consultant source. |
| Fixed-income calculated prices can be generated from units and market value. | CI Axys release notes state that where no IDC price or custodian price exists, output price can be calculated using position units and market value if available. | Yes | No direct APX evidence in same source | Verified for CI Axys release behavior | ByAllAccounts CI Axys Release Notes. |
| Calculated-price decimal precision matters. | CI Axys release notes state a bond calculated-price truncation defect was corrected; CI 3.4 outputs up to the maximum number of decimal digits set for the firm, default four, with support able to configure greater precision. | Yes | No direct APX evidence in same source | Verified for CI Axys release behavior | ByAllAccounts CI Axys Release Notes. |
| Calculated price vs missing price classification has version-specific behavior. | CI Axys release notes state a bug was fixed where current calculated price could be incorrectly exported to Missing Price file instead of Price file. | Yes | No direct APX evidence in same source | Verified for CI Axys release behavior | ByAllAccounts CI Axys Release Notes. |
| Native Axys/APX price field dictionary is not established. | Available public sources identify files, settings, logs, and integration options, but not complete official native price fields. | Unknown | Unknown | Unknown | Requires SS&C IMEX or pricing manuals, sample `.pri`, APX public views, or production exports. |
| Exact standard REP price report names are not established. | Available sources do not provide a complete Axys/APX price report catalog. | Unknown | Unknown | Unknown | Requires REP catalog, `.rep` files, APX reports guide, or sample reports. |

## Source register

| ID | Source | Type | System | Relevance | Confidence notes |
|---|---|---|---|---|---|
| S1 | `AXYS_APX_REFERENCE_BLUEPRINT.md` | Repository governing specification | Repository | Governs research-first mode, confidence labels, Chapter 08 scope, IMEX/REP standards, and Unknown handling. | Verified for repository process. |
| S2 | ByAllAccounts, `Custodial Integrator User Guide` for Axys | Third-party integration guide | Axys | Defines IMEX as Axys Import/Export utility; describes transaction, position, and price files imported into Axys; identifies price file/folder behavior through prior repository research. | Verified for CI workflow; not a complete native SS&C pricing manual. |
| S3 | ByAllAccounts, `Custodial Integrator Release Notes` for Axys | Third-party integration release notes | Axys | Provides version-specific pricing bug fixes: calculated prices, stale/missing prices, bond calculated-price truncation, output precision. | Verified for CI release behavior; not native Axys documentation. |
| S4 | WealthTechs, `AIA User Manual for APX Users` | Third-party integration manual | APX | Documents AIA APX pricing settings, price-file update logic, clean price file, price set logic, custodian trumping order, and custom `.pri` file naming. | Verified for AIA workflow; not a complete APX pricing manual. |
| S5 | WealthTechs, `AIA User Manual for Axys Users` | Third-party integration manual | Axys | Provides adjacent evidence about Axys AIA processing and imports; pricing-specific source evidence was weaker than the APX manual in this pass. | Medium; use cautiously. |
| S6 | AdventGuru, Axys/APX price-file and IMEX articles | Consultant / practitioner source | Axys/APX | Mentions `mergepri`, simple exported price-file formats, primary source precedence, direct file access risks, IMEX/report alternatives. | Medium Confidence; consultant-derived. |
| S7 | SS&C Advent Axys product page | Vendor product page | Axys | Confirms broad Axys accounting/reporting, reconciliation, performance, multicurrency, and corporate action capabilities. | Verified for product capability, not field-level pricing mechanics. |
| S8 | SS&C Advent Portfolio Exchange product page | Vendor product page | APX | Confirms APX broad accounting/reporting/performance, multi-currency/multi-asset, and standard/custom reporting capabilities. | Verified for product capability, not field-level pricing mechanics. |
| S9 | UNAPEN PriceFusion for APX | Third-party product page | APX | Shows modern market demand for APX pricing/reference-data automation and LSEG DataScope Select integration; supports price-source ecosystem context only. | Medium for ecosystem; not native APX behavior. |
| S10 | Prior repository research: `Research_12_IMEX.md` | Repository research | Axys/APX | Consolidated evidence for IMEX files/folders, `*.pri`, `$pathpri`, IMEX logs, direct file access, REP32/Replang. | Internal research; source-supported but should still be traceable to underlying sources. |
| S11 | Prior repository research: `Research_04_Security_Master.md` | Repository research | Axys/APX | Supports security symbol/type identity, security master dependencies, `SourceId` price context, `sec.inf`, `type.inf`, and APX public-view cautions. | Internal research; source-supported. |
| S12 | Prior repository research: `Research_05_Transactions.md` | Repository research | Axys/APX | Supports transaction-price dependency, `.pri`, security/transaction relationships, performance dependencies, and direct-file-access cautions. | Internal research; source-supported. |
| S13 | Advent Portfolio Exchange Reports Guide / `REP_APX.pdf` | Vendor report guide discovered by search | APX | Search result confirms report guide exists and includes holdings/valuation-style columns in public snippet, but not enough for exact pricing report catalog. | Low to Medium until the full PDF is acquired and read. |

## Sources searched but not sufficient

| Search / Source | Result | Why insufficient |
|---|---|---|
| Public search for `Axys price IMEX field dictionary` | Found integration references and consultant pages. | Did not locate official Axys IMEX price object field dictionary. |
| Public search for `APX price file IMEX APXIX price fields` | Found WealthTechs AIA APX guide. | AIA workflow is useful but not official native APX pricing schema. |
| Public search for `Advent Axys .pri file format` | Found AdventGuru references and CI evidence. | Did not locate complete `.pri` file layout or official documentation. |
| Public search for `Advent APX price report REP` | Found general APX reports guide and report snippets. | Did not locate exact price report names/parameters/columns. |
| Product pages for Axys/APX | Confirm broad capabilities. | Do not provide low-level pricing fields, file layouts, or calculation rules. |
| PriceFusion for APX | Confirms pricing/reference-data integration market. | Does not document native APX pricing schema or behavior. |

## Pricing concepts and research treatment

### Conceptual pricing model

**Industry Practice — High Confidence:** In a portfolio accounting system, pricing converts positions into market values, drives many holdings and performance calculations, and supports reconciliation, stale-price detection, and exception review.

**System-specific status:** The available public evidence confirms that Axys/APX integration workflows process price files, but it does not establish a complete native Axys/APX price architecture.

| Concept | Definition | Axys support from available evidence | APX support from available evidence | Confidence |
|---|---|---|---|---|
| Price date | Date for which a security price applies. | Implied by `.pri` files and historical-day IMEX price logs; exact field Unknown. | Implied by AIA price-file naming such as `mmddyy_CDI.pri`; exact field Unknown. | Medium |
| Security price | Unit price used to value a security. | Price files imported through CI; calculated prices can be output to Price file. | AIA imports price data into APX price files. | Medium |
| Price file | File containing prices for import/merge/update. | `*.pri` observed in CI research. | `mmddyy_CDI.pri` observed in AIA APX guide. | Verified for integration workflows |
| Price source | Vendor/custodian/calculated/source provenance of price. | Custodian, IDC, third-party security price, and calculated price are mentioned in CI release notes; native source field Unknown. | `SourceId` appears in an APX price-source context in prior research; custodian-specific price logic observed in AIA. | Medium |
| Missing price | Price absent or unusable for required processing. | CI has Missing Price file behavior; current calculated-price bug fixed to export to Price file instead of Missing Price file. | Unknown natively; AIA may filter/merge price files, but missing-price behavior not established. | Medium for Axys CI; Unknown native |
| Stale price | Price available but not current enough. | CI release notes discuss stale third-party/holding prices and current calculated prices. | Unknown. | Medium for Axys CI |
| Calculated price | Price computed from position units and market value when custodian/IDC price not available. | Explicitly documented in CI release notes. | Unknown. | Verified for Axys CI release behavior |
| Price set | A set of prices that can differ by custodian/source/logic. | Unknown. | AIA APX documents Price Set Logic options, including custodian-specific prices. | Verified for AIA/APX workflow |
| Price merge / trumping order | Process of choosing among multiple source prices. | AdventGuru `mergepri`; primary source precedence. | AIA APX multi-custodian trumping order; AdventGuru `mergepri`. | Medium |
| Price precision | Number of decimals retained in output price. | CI release notes discuss default four decimal digits and support-configured precision for calculated bond prices. | Unknown. | Verified for Axys CI release behavior |
| FX rate | Currency conversion rate used for non-base-currency valuation. | Axys multicurrency capability is vendor-confirmed; exact pricing/FX storage Unknown. | APX multicurrency capability is vendor-confirmed; exact pricing/FX storage Unknown. | High for capability; Unknown for schema |

## Axys findings

### Axys product capability context

| Statement | Confidence | Evidence treatment |
|---|---|---|
| Axys is currently marketed by SS&C Advent as portfolio reporting and accounting software. | Verified | Vendor product page. |
| Axys supports performance measurement, automated reconciliation, multicurrency capabilities, and corporate-actions processing at a product-capability level. | Verified for product capability | Vendor product page; does not provide pricing internals. |
| Axys pricing mechanics, price-file layouts, and native price storage are not described in the product page. | Verified absence in reviewed source | Requires vendor manual or production sample. |

### Axys price files and folders

| Artifact | Description | Confidence | Caveat |
|---|---|---:|---|
| `$pathpri` | Axys price folder identified in prior IMEX research based on ByAllAccounts CI evidence. | Verified for CI workflow | Not necessarily complete Axys path model. |
| `*.pri` | Axys price files in the price folder. | Verified for CI workflow | Complete layout Unknown. |
| `imexPrices.log` | IMEX log associated with price import; errors such as open price file can appear here. | Verified for CI workflow | Exact log schema Unknown. |
| `imexPrices` tab(s) | CI View IMEX Logs may show one `imexPrices` tab per historical day when prices are requested for multiple historical days. | Verified for CI workflow | Integration UI behavior. |

### Axys CI price import workflow

**Observed Behavior — Verified for ByAllAccounts CI workflow:** CI downloads external data, merges it with Axys security information, produces transaction, position, and price files, and imports those files into Axys using the Axys Import/Export utility.

```text
External source-data
        ↓
Custodial Integrator download / translation
        ↓
Axys security and type information used for mapping
        ↓
Transaction file + position file + price file
        ↓
Axys Import/Export utility (`imex32.exe` in CI context)
        ↓
Price records available in Axys if import succeeds
        ↓
IMEX log review (`imexPrices.log`)
```

| Step | Evidence-supported statement | Confidence |
|---|---|---:|
| Translate source-data | CI uses Axys security information to generate output files. | Verified for CI |
| Produce price file | CI produces price files as part of data translation. | Verified for CI |
| Import via IMEX | Requested Transaction, Position, and Price files are imported into Axys using Axys Import/Export utility. | Verified for CI |
| Log review | IMEX logs can be reviewed after import. | Verified for CI |
| Error handling | An open/in-use Axys price file can cause import failure; error appears in `imexPrices.log`. | Verified for CI |

### Axys missing and stale price behavior from CI release notes

| Statement | Confidence | Notes |
|---|---:|---|
| CI release notes mention `third party security price`, `holding price`, `calculated price`, `Missing Price file`, and `Price file`. | Verified for CI release notes | Indicates multiple price-source states in CI workflow. |
| A fixed bug involved third-party security price and holding price being unavailable or stale while calculated price was available and current; the security price was incorrectly exported to Missing Price file. | Verified for CI release notes | Strong evidence of missing/stale/calculated price handling in CI. |
| Corrected behavior exports the calculated price to the Price file. | Verified for CI release notes | CI behavior, not necessarily native Axys behavior. |
| CI 3.4 corrected a bug where calculated bond prices lost the last two decimal digits. | Verified for CI release notes | Pricing precision issue. |
| CI 3.4 outputs up to the maximum number of decimal digits set for the firm; default is four, support can configure calculated price decimals greater than one. | Verified for CI release notes | Applies to CI calculated security prices. |
| When no IDC price and no custodian price exist, CI can calculate output price using position units and market value, if available. | Verified for CI release notes | Important fixed-income/derived-price behavior. |
| Calculated price can be truncated to zero if output precision is too low and price is very small. | Verified for CI release notes | Important audit condition. |

### Axys price merge evidence

| Statement | Confidence | Evidence treatment |
|---|---:|---|
| AdventGuru states exported price file formats for Axys and APX are simple enough that users could write a merger, but Advent has an existing `mergepri` script command. | Medium | Consultant evidence; useful implementation clue. |
| `mergepri` allows specifying a destination and multiple sources. | Medium | Consultant evidence. |
| The first source is primary. | Medium | Consultant evidence. |
| Prices in the first source file are not overwritten by prices from secondary source files. | Medium | Consultant evidence. |
| Exact `mergepri` syntax, supported versions, and error behavior are Unknown from available evidence. | Unknown | Requires script command documentation or production tests. |

### Axys native price architecture unknowns

The following remain Unknown for Axys:

| Unknown | Needed evidence |
|---|---|
| Complete `.pri` file layout. | Sanitized price file, Axys file layout documentation, IMEX manual. |
| Whether one `.pri` file normally represents one date, one source, one currency, one custodian, or another convention. | Production files and naming standards. |
| Native price key: security symbol + type + date? source? currency? | Vendor documentation or reproducible tests. |
| Whether Axys stores separate bid/ask/close/evaluated prices. | Vendor manual or price-file sample. |
| Price source hierarchy inside Axys. | Vendor documentation or integration configs. |
| Native stale-price thresholds. | Vendor documentation or system settings. |
| Native missing-price report names. | REP/report catalog or screen/report sample. |
| Exact relationship between transaction price and historical price file price. | Vendor docs or reconciliation test. |
| Whether prices are split-adjusted by file replacement, corporate-action processing, report calculation, or external vendor feed. | Vendor docs and sample split scenario. |

## APX findings

### APX product capability context

| Statement | Confidence | Evidence treatment |
|---|---|---|
| APX is currently marketed by SS&C Advent as an integrated portfolio and client management platform with accounting/reporting/performance capabilities. | Verified | Vendor product page. |
| APX supports multi-currency and multi-asset coverage at product-capability level. | Verified for capability | Vendor product page; exact pricing schema unknown. |
| APX provides standard reports and flexible custom reporting at product-capability level. | Verified for capability | Vendor product page; exact pricing report catalog unknown. |

### APX AIA pricing settings

The strongest APX pricing evidence found in this pass is from WealthTechs AIA for APX. This evidence is specific to an AIA workflow and must not be treated as a complete native APX pricing manual.

| AIA APX setting / concept | Observed behavior | Confidence | Caveat |
|---|---|---:|---|
| Price File Update Logic | Determines how AIA imports price data while factoring in existing APX price data. | Verified for AIA workflow | Native APX internals Unknown. |
| Update Existing & Add New | Updates prices for vehicles currently in an APX price file and adds prices for new vehicles. | Verified for AIA workflow | AIA option. |
| Add New | Adds new vehicles to price file if they do not exist in APX. | Verified for AIA workflow | AIA option. |
| Replace Entire File | Deletes existing prices currently in APX and replaces them with prices in the AIA import file. | Verified for AIA workflow | High-risk operational option. |
| Clean Price File | `Remove Prices for Accounts Filtered` removes prices for securities held only in filtered accounts from the APX price file. | Verified for AIA workflow | AIA option; native APX status Unknown. |
| Price Set Logic | Option used only if client uses price sets in APX. | Verified for AIA workflow | Exact APX price-set model Unknown. |
| Create A Price For Each Custodian | Creates a price file specifically for each custodian; securities can be priced differently depending on custodian. | Verified for AIA workflow | AIA/APX context. |
| Merge Pricing Using Multi Custodian Setting Trumping Order | Default AIA option; assumes one price file per day for all custodians; first custodian has highest priority. | Verified for AIA workflow | Trumping order configured in Multi Custodian Settings. |
| Merge Pricing Using Multi Custodian Setting Trumping Order and Use Custom Price File Name | Uses custodian trumping order and custom price filename; example updates/adds/replaces one APX price file `mmddyy_CDI.pri`. | Verified for AIA workflow | Example-specific filename. |

### APX price sets and custodian-specific pricing

**Observed Behavior — Verified for AIA workflow:** The AIA APX manual explicitly allows creating prices per custodian. It states that securities could be priced differently depending on their specific custodian.

Research implications:

| Implication | Treatment | Confidence |
|---|---|---:|
| APX environments may support or at least accommodate price sets. | Observed in AIA APX guide. | Medium-to-Verified for AIA setting |
| Pricing source/custodian can materially affect valuation. | Strongly implied by custodian-specific price-file logic. | Medium |
| A migration/audit tool should preserve price source or price set when available. | Recommendation based on observed workflow. | High Confidence as recommendation |
| Native APX price-set schema is Unknown. | Explicit Unknown. | Unknown |

### APX price source field evidence

Prior repository Security Master research identified `SourceId` as an APX price source field shown in an AIA APX screenshot/example, not a security-master field.

| Field / Label | Context | Axys | APX | Confidence | Caveat |
|---|---|---:|---:|---|---|
| `SourceId` | Price source field shown in AIA APX price context. | No | Yes | Verified in AIA screenshot/example context | Not proven to be native APX database, IMEX, or REP field name. |

### APX native price architecture unknowns

The following remain Unknown for APX:

| Unknown | Needed evidence |
|---|---|
| Native APX price tables, views, or stored accounting functions. | APX schema/public-view docs or sanitized SQL output. |
| Native APX price key and whether source/price set is part of the key. | Vendor docs or production observation. |
| Native APX price-file layout for `.pri` imports. | AIA archive sample, APX import spec, or vendor docs. |
| Whether APX price files are true persisted files, import staging files, compatibility artifacts, or AIA-generated exchange files. | Vendor documentation or installation evidence. |
| Whether `mmddyy_CDI.pri` is AIA-specific, APX-native-compatible, or both. | AIA examples plus APX import documentation. |
| Exact behavior of `Replace Entire File`: scope by date, source, all securities, all price sets, or file-specific. | AIA/vendor documentation or test system. |
| Native missing-price and stale-price reports. | APX report guide, REP report catalog, or sample reports. |
| Native APX field names for price date, price, source, currency, and price set. | Vendor data dictionary, public views, or sample exports. |

## Axys/APX comparison matrix

| Area | Axys | APX | Confidence |
|---|---|---|---|
| Observed price files | `*.pri` in `$pathpri` folder from CI evidence. | AIA example `mmddyy_CDI.pri`; price files and price sets from AIA APX guide. | Verified for integration contexts |
| Import utility | Axys Import/Export utility; `imex32.exe` in CI evidence. | APX import/export function appears as `APXIX.exe` in prior research; AIA pricing imported to APX price files. | Verified for integration contexts; native details Unknown |
| Price import update modes | Unknown native; CI imports price files and logs errors. | AIA documents Update/Add/Replace modes. | Verified for AIA APX; Axys native Unknown |
| Price source merging | AdventGuru `mergepri` primary-source precedence. | AIA multi-custodian trumping order and AdventGuru `mergepri`. | Medium |
| Custodian-specific pricing | Unknown from reviewed Axys sources. | AIA APX supports creating a price for each custodian. | Verified for AIA APX |
| Missing price file | CI release notes mention Missing Price file. | Unknown. | Verified for CI Axys; APX Unknown |
| Calculated prices | CI can output calculated prices using units and market value when no IDC/custodian price exists. | Unknown. | Verified for CI Axys |
| Price precision | CI release notes discuss default four decimal digits and configurable calculated-price precision. | Unknown. | Verified for CI Axys |
| Direct database price access | Not applicable in same way; Axys direct files are risky. | APX public views/SQL options mentioned by consultant sources; exact price views Unknown. | Medium for access option; Unknown for fields |
| REP/report price access | Report Writer Pro/Replang available, exact price report names Unknown. | Standard/custom reporting available, exact price report names Unknown. | Medium/Unknown |

## IMEX pricing coverage

### Verified / observed IMEX-adjacent pricing facts

| Statement | Axys | APX | Confidence | Caveat |
|---|---:|---:|---|---|
| CI uses Axys IMEX to import price files. | Yes | No | Verified for CI | Not complete IMEX object documentation. |
| CI retains IMEX price logs. | Yes | No | Verified for CI | Log schema Unknown. |
| IMEX price import can fail if target price file is open/in use. | Yes | No | Verified for CI | Operational quirk. |
| Multiple historical price days can create multiple `imexPrices` log tabs. | Yes | No | Verified for CI | CI UI behavior. |
| Exact native IMEX price object name is not established. | Unknown | Unknown | Unknown | Need official IMEX docs/samples. |
| Exact native IMEX price field list is not established. | Unknown | Unknown | Unknown | Need official IMEX docs/samples. |

### Candidate IMEX price object documentation requirements

The future Chapter 08 should not invent native IMEX object names. It should document a placeholder structure like this until source material is obtained:

| Field / Attribute | Current status | Needed evidence |
|---|---|---|
| IMEX price object name | Unknown | Axys/APX IMEX manual, `.inf`, logs, screenshots. |
| Direction | Import observed for Axys CI; APX AIA price import observed; native export Unknown. | IMEX docs and export samples. |
| Required key fields | Unknown | Sample `.pri` or IMEX field dictionary. |
| Price date field | Unknown | Sample export/import. |
| Security identifier field(s) | Unknown; likely symbol/type in observed security contexts but not proven for price file. | Price-file samples. |
| Price value field | Unknown official name. | Price-file samples. |
| Price source field | Unknown; `SourceId` observed in AIA/APX context only. | APX docs/export. |
| Currency field | Unknown. | Multi-currency price sample. |
| Price set field | Unknown. | APX price-set docs/export. |
| Null/missing behavior | Partially observed in CI release notes. | IMEX manual + logs. |
| Error logs | `imexPrices.log` observed in CI. | Actual log samples. |

### Sample IMEX-adjacent Axys workflow

```text
$pathpri
    062926.pri              # example only; exact naming convention Unknown

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

**Classification:** File extension and `$pathpri` are Verified for CI. The example file name is illustrative only; exact naming convention is Unknown.

## REP / reporting coverage

### REP and Replang context

Prior repository research supports that Axys/APX extraction can use REP32, Report Writer Pro, Replang, standard reports, macros, and custom reports. However, pricing-specific REP report names were not established in this research pass.

| Statement | Axys | APX | Confidence |
|---|---:|---:|---|
| Axys/APX users can create reports using Report Writer Pro or Replang source. | Yes | Yes | Medium Confidence from consultant and connector evidence |
| Data Broker connector uses standard reports/macros and REP32 for Axys/APX extraction. | Yes | Yes | Verified for connector |
| Exact standard price report names are Unknown. | Unknown | Unknown | Unknown |
| Exact REP price fields are Unknown. | Unknown | Unknown | Unknown |
| Whether REP reports expose stored price values or calculated valuation outputs is Unknown. | Unknown | Unknown | Unknown |

### Price-related report candidates to request

| Desired report/source | Why it matters |
|---|---|
| Standard price list report | Confirms report name, parameters, output fields. |
| Missing price report | Confirms missing-price detection logic and visible fields. |
| Stale price report | Confirms stale threshold and source/date fields. |
| Holdings valuation report | Confirms how price appears alongside quantity and market value. |
| Fixed-income valuation report | Confirms price, factor, accrued interest, market value, income fields. |
| APX public-view query for prices | Confirms APX schema/view fields and source/price-set behavior. |
| REP source (`.rep`) for price reports | Confirms RepLang variables and calculation behavior. |

## Field dictionary candidates

Important warning: the table below is a conservative research catalog. It is not a verified native Axys/APX price data dictionary.

| Field / Label | Definition / meaning | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---|
| `*.pri` | Price file extension observed in Axys/APX price-file workflows. | Yes | Yes in AIA APX example | Related | No | Verified for integration contexts |
| `$pathpri` | Axys price-folder label from CI evidence. | Yes | No | Related | No | Verified for CI Axys |
| `imexPrices.log` | Axys IMEX price-import log name/tab observed in CI evidence. | Yes | No | Yes | No | Verified for CI Axys |
| `Price File` | Output/import target for prices. | Yes | Yes | Related | Unknown | Verified for integration contexts |
| `Missing Price file` | CI output/category for securities without usable price, unless calculated price is available/current. | Yes | Unknown | Related | Unknown | Verified for CI Axys |
| `third party security price` | Price source type mentioned in CI release notes. | Yes in CI context | Unknown | Unknown | Unknown | Verified for CI Axys |
| `holding price` | Price source/state mentioned in CI release notes. | Yes in CI context | Unknown | Unknown | Unknown | Verified for CI Axys |
| `calculated price` | Price derived by CI, including from units and market value where custodian/IDC price is unavailable. | Yes in CI context | Unknown | Unknown | Unknown | Verified for CI Axys |
| `IDC price` | Pricing-source reference in CI release notes. | Yes in CI context | Unknown | Unknown | Unknown | Verified for CI Axys |
| `SourceId` | Price source label observed in prior AIA/APX price context. | No | Yes | Unknown | Unknown | Verified for AIA/APX context only |
| `Price File Update Logic` | AIA/APX setting controlling update/add/replace behavior. | No | Yes | Related | No | Verified for AIA/APX |
| `Price Set Logic` | AIA/APX setting for price sets and custodian-specific price behavior. | No | Yes | Related | No | Verified for AIA/APX |
| `Clean Price File` | AIA/APX setting removing prices for securities held only in filtered accounts. | No | Yes | Related | No | Verified for AIA/APX |
| Price date | Date of price. | Unknown official field | Unknown official field | Unknown | Unknown | High as concept; native Unknown |
| Security identifier | Security whose price is being supplied. | Unknown official field; symbol/type likely related but not verified | Unknown official field | Unknown | Unknown | High as concept; native Unknown |
| Price value | Numeric price. | Unknown official field | Unknown official field | Unknown | Unknown | High as concept; native Unknown |
| Currency | Currency of price. | Unknown | Unknown | Unknown | Unknown | Medium concept; native Unknown |
| Price source | Source/vendor/custodian/pricing set. | CI mentions several sources; native field Unknown | `SourceId` observed in AIA/APX context; native field Unknown | Unknown | Unknown | Medium |
| Price set | APX price-set grouping. | Unknown | AIA APX setting | Unknown | Unknown | Medium for APX AIA; native Unknown |
| Factor | Fixed-income/security factor. | Chapter 04 scope; pricing relationship Unknown | Unknown | Unknown | Unknown | Conceptual; system-specific Unknown |
| Price multiplier | Security master/pricing multiplier. | Unknown | Unknown | Unknown | Unknown | Conceptual; system-specific Unknown |
| Accrued interest | Fixed-income income accrual. | Conversion research suggests `sec.inf` can include accrued-interest-related fields; price role Unknown. | Unknown | Unknown | Unknown | Medium concept; system-specific Unknown |

## Processing behavior

### Price import/update patterns

| Pattern | Axys | APX | Confidence | Notes |
|---|---|---|---|---|
| Append/update existing prices | Unknown native; CI imports price files. | AIA APX `Update Existing & Add New`. | Verified for AIA APX; Axys Unknown | Need native Axys docs. |
| Add new only | Unknown native. | AIA APX `Add New`. | Verified for AIA APX | Useful migration/audit mode. |
| Replace entire file | Unknown native. | AIA APX `Replace Entire File`. | Verified for AIA APX | High-risk option; exact scope Unknown. |
| Merge with source priority | AdventGuru `mergepri`. | AIA APX custodian trumping order; AdventGuru `mergepri`. | Medium | Need script docs. |
| Create per-custodian price | Unknown. | AIA APX option. | Verified for AIA APX | Price-set/source-specific valuation risk. |
| Remove filtered-account prices | Unknown. | AIA APX Clean Price File. | Verified for AIA APX | Can affect valuation if shared securities. |
| Calculate price from units and market value | CI release notes. | Unknown. | Verified for CI Axys | Important fixed-income/missing-price fallback. |

### Pricing dependency graph

```text
Security Master
    ├── security symbol / type / identifier
    ├── issue currency / pricing currency
    ├── security type / asset class
    ├── fixed-income terms / factor / accrued-interest fields (Unknown exact fields)
    └── price multiplier (Unknown exact Axys/APX behavior)

Transactions
    ├── transaction price
    ├── quantity
    ├── trade/settle dates
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
    ├── valuation date prices
    └── historical restatement risk if prices change
```

## Version differences and release-note evidence

| System | Version / context | Pricing-related statement | Confidence | Caveat |
|---|---|---|---:|---|
| CI for Axys | V3.11.001 release note | Corrected case where current calculated price was incorrectly exported to Missing Price file instead of Price file when third-party security price and holding price were unavailable/stale. | Verified for CI release | Not native Axys release note. |
| CI for Axys | V3.4 release note | Bond calculated prices were truncating last two decimals; corrected to output up to configured maximum decimal digits. | Verified for CI release | Not native Axys release note. |
| CI for Axys | V3.4 release note | Default calculated-price decimal precision is four; technical support can configure greater than one decimal. | Verified for CI release | CI setting, not necessarily Axys native. |
| CI for Axys | V3.4 release note | When no IDC price and no custodian price exist, output price is calculated from units and market value if available. | Verified for CI release | CI fallback, not necessarily native Axys. |
| APX AIA | 2023 public manual | Price File Update Logic and Price Set Logic documented. | Verified for AIA manual | AIA version/workflow-specific. |
| APX | APX v1.x-v4.x consultant evidence | AdventGuru says APX maintained IMEX functionality but eliminated fixed-format file generation. | Medium | Interface-version evidence; pricing-specific field impact Unknown. |
| Axys | Axys 3.7 to 3.8 consultant evidence | AdventGuru says file conversion changed some file formats; direct file access is risky. | Medium | Applies generally to files; exact `.pri` impact Unknown. |

## Implementation quirks and audit implications

### Observed quirks

| Quirk | System | Description | Confidence | Audit implication |
|---|---|---|---:|---|
| Price file open/in use can fail import | Axys CI | Open Axys price file can cause import failure and `imexPrices.log` error. | Verified for CI | Check logs and rerun after closing file. |
| Current calculated price may be misclassified in older CI versions | Axys CI | Bug fixed where calculated price went to Missing Price file rather than Price file. | Verified for CI release | Version-aware missing-price review. |
| Bond calculated price truncation | Axys CI | Older CI could truncate bond calculated-price decimals. | Verified for CI release | Audit low-price/high-par fixed income carefully. |
| Calculated price can be based on units and market value | Axys CI | Fallback when IDC/custodian price unavailable. | Verified for CI release | Distinguish vendor price from derived price. |
| Price source precedence | Axys/APX | `mergepri` / custodian trumping order can preserve primary source price. | Medium | Preserve source hierarchy in audits. |
| Replace entire price file | APX AIA | AIA option deletes existing prices and replaces with import file prices. | Verified for AIA | High-risk control; require backup/review. |
| Remove filtered-account prices | APX AIA | Prices for securities held only in filtered accounts may be omitted. | Verified for AIA | Missing prices may be intentional but must be traceable. |
| Same security priced differently by custodian | APX AIA | AIA option permits custodian-specific price files. | Verified for AIA | Valuation/reporting may differ by custodian/source. |
| Direct file access risk | Axys | Consultant evidence warns Axys file formats can change across versions. | Medium | Prefer IMEX/REP; version-control file readers. |

### Candidate audit rules

These are research candidates for future Chapter 08 / Chapter 14. They are not Axys/APX-native rules unless source evidence is listed.

| Rule ID | Name | Description | Required inputs | Detection logic | Severity | Confidence |
|---|---|---|---|---|---|---|
| PRICE-001 | Missing Price | Security with nonzero holding lacks a price for valuation date. | holdings, security, price date, price file/report | holding_quantity != 0 and no price for valuation date | High | High concept; native implementation Unknown |
| PRICE-002 | Stale Price | Price date is older than allowed threshold for valuation date. | security, price date, valuation date, threshold | valuation_date - price_date > threshold | Medium/High | High concept; threshold Unknown |
| PRICE-003 | Zero Price | Price is zero or near zero for non-cash/non-defaulted security. | security type, price | price <= configured_minimum | High | High concept |
| PRICE-004 | Calculated Price Flag | Price was derived from units and market value rather than vendor/custodian price. | price source, position units, market value | source == calculated or price = MV/units | Medium | Verified for CI calculated-price concept |
| PRICE-005 | Price Source Precedence Violation | Lower-priority source overwrote higher-priority source. | price source order, price file sequence | secondary source changed primary source price | High | Medium; based on merge/trumping evidence |
| PRICE-006 | Price File Replace Risk | Replace-entire-file option used without control evidence. | import log, mode, backup flag | mode == Replace Entire File and backup/review missing | High | Verified AIA mode; audit rule recommendation |
| PRICE-007 | Custodian-Specific Price Divergence | Same security/date has different prices by custodian/source. | security, date, source/custodian, price | max(price)-min(price) > tolerance | Medium | Verified AIA possibility; tolerance Unknown |
| PRICE-008 | Price File Lock Failure | Price import failed because file was open/in use. | IMEX log | log contains file-in-use or price import failure | Medium | Verified CI example |
| PRICE-009 | Split Price Discontinuity | Large price change coincides with split but quantity/price ratio not consistent. | prices, split ratio, holdings | price_ratio inconsistent with split ratio | High | Industry practice; native behavior Unknown |
| PRICE-010 | Bond Price Precision Loss | Fixed-income calculated price appears truncated or rounded beyond tolerance. | bond price, precision, units, MV | abs(price - MV/units) > tolerance due to truncation | Medium | Verified CI historical issue |
| PRICE-011 | Filtered Account Price Omission | Price omitted because only filtered accounts hold the security. | filter settings, holdings, price file | security held only in filtered accounts and omitted | Low/Medium | Verified AIA setting |
| PRICE-012 | Transaction Price vs Market Price Outlier | Trade price differs materially from market close/evaluated price. | transaction price, date price, tolerance | abs(trade_price - close_price)/close_price > tolerance | Medium | Recommendation / industry practice |
| PRICE-013 | FX Rate Missing for Foreign Price | Foreign-currency price lacks FX rate for valuation/reporting. | price currency, base currency, FX rate | price_currency != base_currency and missing FX | High | High concept; native fields Unknown |
| PRICE-014 | Negative Price | Price less than zero for security types where negative prices are invalid. | security type, price | price < 0 and not allowed | High | High concept |
| PRICE-015 | Price Multiplier Mismatch | Market value not consistent with quantity, price, multiplier, and factor. | quantity, price, multiplier, factor, MV | MV != quantity * price * multiplier * factor within tolerance | High | High concept; system fields Unknown |

## Examples

### Example 1 — APX AIA price-file update mode

```text
Process date: 2026-06-29
Mode: Update Existing & Add New
Security: ABC
Existing price: 100.00
Incoming price: 101.25
Result: existing price updated; new securities added if not present
```

**Classification:** Example is conceptual based on AIA APX mode name. Exact APX internal result and price-key scope are Unknown.

### Example 2 — APX AIA multi-custodian trumping order

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

**Classification:** Medium Confidence based on AIA APX guide's custodian trumping-order description. Exact tie-break and key fields are Unknown.

### Example 3 — Axys CI calculated price from position data

```text
No IDC price available.
No custodian price available.
Position units: 100,000
Position market value: 98,750
Calculated output price concept: market value / units = 0.9875
```

**Classification:** Verified for CI release-note concept that price can be calculated from units and market value if available. Exact formula, bond scaling convention, and price multiplier behavior are Unknown.

### Example 4 — Price file lock

```text
User has an Axys price file open.
CI attempts price import through Axys Import/Export utility.
Import fails.
User reviews `imexPrices.log` to identify the error.
```

**Classification:** Verified for ByAllAccounts CI workflow.

## Contradictions and interpretive cautions

| Issue | Evidence | Caution | Confidence |
|---|---|---|---:|
| `.pri` file meaning across systems | Axys CI uses `*.pri`; APX AIA example uses `mmddyy_CDI.pri`. | Do not assume identical layout, path, key, or storage model across Axys/APX. | High caution |
| Price source terminology | CI release notes mention IDC, custodian, third-party, holding, calculated; APX AIA shows `SourceId` and custodian trumping. | Do not normalize all sources into one field without proof. | High caution |
| Calculated price vs official price | CI can calculate output price; native Axys may also have or consume other price types. | Label calculated prices separately from vendor/evaluated prices. | High caution |
| Replace Entire File scope | APX AIA says deletes existing prices currently in APX and replaces them with import file prices. | Exact scope is Unknown: date, source, price set, or all file-represented prices. Treat as high-risk. | High caution |
| Price set vs price file | APX AIA discusses price sets and price files. | Do not infer database model from AIA UI labels. | High caution |
| Primary close vs custodian price | Chapter scope includes primary exchange close, but reviewed sources do not confirm native Axys/APX handling. | Mark as Unknown. | Unknown |
| Evaluated fixed-income price | PriceFusion and CI release notes indicate fixed-income pricing ecosystem, but native APX/Axys evaluated-price behavior is Unknown. | Avoid vendor-specific claims without docs. | Unknown |
| Split-adjusted prices | Chapter scope includes split-adjusted prices; reviewed pricing sources do not define native behavior. | Defer to Corporate Actions research and sample data. | Unknown |

## Known unknowns

### Highest-priority unknowns

| ID | Unknown | Why it matters | Evidence needed |
|---|---|---|---|
| PU-001 | Complete Axys `.pri` file layout. | Required for importers, validators, migration tools. | Sanitized price file, file layout manual, IMEX manual. |
| PU-002 | Complete APX price import/export layout. | Required for APX price integration. | AIA/APX archive price files, APX IMEX docs, public views. |
| PU-003 | Native IMEX price object names for Axys and APX. | Required for Chapter 12 and Chapter 08 field dictionaries. | Vendor IMEX object catalog or sample `.inf`. |
| PU-004 | Native price key fields. | Needed for dedupe, replacement, update logic. | Vendor docs or reproducible tests. |
| PU-005 | Native price source / price set model. | Needed to avoid overwriting correct valuations. | APX price-set docs, Axys source docs, sample exports. |
| PU-006 | Standard REP report names for prices, missing prices, stale prices. | Required for REP coverage. | REP catalog, `.rep` source, report output samples. |
| PU-007 | APX public-view/stored accounting function fields for prices. | Needed for APX SQL extraction. | View schema or query output. |
| PU-008 | Primary exchange close handling. | Chapter scope item; needed for audit rules. | Vendor docs or price source setup screenshots. |
| PU-009 | Fixed-income evaluated price and factor behavior. | Critical for bonds and MBS. | Fixed-income security sample, price sample, valuation report. |
| PU-010 | Price multiplier handling. | Critical for market value calculation. | Security master docs and holdings valuation examples. |
| PU-011 | FX-rate storage and price currency handling. | Needed for multicurrency valuation. | FX rate file/export/report docs. |
| PU-012 | Corporate-action price adjustment behavior. | Needed for split/audit rules. | Split scenario before/after prices and holdings. |
| PU-013 | Stale-price threshold and exception process. | Needed for production audit rules. | Reports, settings, operations procedures. |
| PU-014 | Whether transaction prices are independent from daily price files. | Needed for transaction audit and performance restatement analysis. | Transaction export + price export for same dates. |
| PU-015 | Whether historical price changes trigger stored performance regeneration. | Needed for Chapter 10 dependency. | Performance docs, test scenario, production observations. |

### Additional unknowns

| Unknown | Needed evidence |
|---|---|
| Price file naming convention by date/source/currency. | Multiple real price files. |
| Whether cash equivalents require explicit prices. | Security/pricing docs and sample holdings. |
| Whether zero prices are allowed for defaulted securities or options. | Vendor docs and exception reports. |
| Whether prices are stored as clean or dirty prices for fixed income. | Fixed-income valuation report and price file. |
| Whether accrued interest is stored with price, security, transaction, or calculated at report time. | Fixed-income docs and report source. |
| Whether APX price sets can coexist for the same security/date/source. | APX docs/test data. |
| Whether Axys supports multiple price sources per date natively. | Vendor docs or sample files. |
| Whether missing-price files are generated by Axys, CI, or both. | Logs/manuals. |
| Whether `mergepri` is available in all Axys/APX versions. | Script-command documentation and version notes. |

## Recommended next evidence to request

To convert this research into a stronger Chapter 08, request any of the following:

1. Sanitized Axys `.pri` file samples for several dates.
2. Sanitized APX price import/export files, including any `mmddyy_CDI.pri` examples.
3. Axys IMEX price import/export control files or `.inf` examples.
4. APX IMEX price import/export control files or logs.
5. `imexPrices.log` samples from successful and failed imports.
6. Missing-price output file samples from Axys/CI/APX workflows.
7. REP report source or output for price list, missing prices, stale prices, holdings valuation, and fixed-income valuation.
8. APX public-view schema or query output for prices and price sources.
9. Price setup screenshots: price sources, price sets, stale thresholds, custodian priority/trumping order.
10. Fixed-income examples: bond price, factor, accrued interest, quantity, market value.
11. Corporate-action split example: prices/holdings before and after split.
12. FX-rate files or reports used for valuation.
13. Any official SS&C Axys/APX pricing manual, IMEX manual, or release notes.
14. Production observation notes from an Axys/APX user explaining daily pricing workflow.

## Draft chapter structure suggested from research

```text
# 08. Pricing

1. Overview
   - Pricing role in portfolio accounting.
   - Explicit distinction between native pricing, integration price files, REP reports, and derived valuation outputs.

2. Axys Pricing
   - Product capability context.
   - `*.pri`, `$pathpri`, CI price import.
   - IMEX price logs and file-lock quirk.
   - Calculated price / missing price behavior from CI release notes.
   - Unknown native price schema.

3. APX Pricing
   - Product capability context.
   - AIA Price File Update Logic.
   - Price Set Logic and custodian-specific prices.
   - `SourceId` observed in AIA/APX context.
   - Unknown native APX price schema/public views.

4. Axys/APX Differences
   - File-based Axys evidence vs APX price-set/source evidence.
   - APX SQL/public-view possibility.
   - Direct file access cautions.

5. IMEX Coverage
   - Observed price import.
   - Unknown native object names and fields.
   - Log behavior.

6. REP Coverage
   - Report Writer/REP32/Replang context.
   - Price report names Unknown.
   - Required evidence.

7. Field Dictionary
   - Observed fields/labels.
   - Candidate conceptual fields.
   - Native Unknowns.

8. Processing Rules
   - Import/update/add/replace.
   - Merge/trumping order.
   - Calculated price fallback.
   - Missing/stale price handling.

9. Calculation Dependencies
   - Holdings quantity.
   - Security master.
   - Price source.
   - Currency/FX.
   - Factor/multiplier/accrued interest.
   - Performance.

10. Audit Rules
   - Missing/stale/zero/outlier prices.
   - Source precedence.
   - price-file replace controls.
   - fixed-income precision.
   - split/price discontinuity.

11. Migration Considerations
   - Preserve source, date, price set, precision.
   - Avoid direct raw file readers unless version-controlled.
   - Reconcile valuation from quantity × price × factor × multiplier.

12. Examples
   - AIA price update modes.
   - Multi-custodian trumping.
   - Calculated price from MV/units.
   - File lock failure.

13. References

14. Known Unknowns

15. Future Research
```

## References

### Repository / uploaded research sources

1. `AXYS_APX_REFERENCE_BLUEPRINT.md`, Version 9 Actionable Research-First Repository Constitution. Governing repository specification.
2. `Research_12_IMEX.md`. Prior repository research supporting IMEX definition, `*.pri`, `$pathpri`, price imports, log behavior, direct file access cautions, REP32/Replang context.
3. `Research_04_Security_Master.md`. Prior repository research supporting security master dependencies, `SourceId` price-source context, `sec.inf`/`type.inf`, APX public-view cautions.
4. `Research_05_Transactions.md`. Prior repository research supporting transaction price dependency, direct-file-access caution, price/holdings/performance dependencies.

### Public web / document sources consulted

1. ByAllAccounts / Morningstar, `Custodial Integrator User Guide` for Axys. URL: `https://www.byallaccounts.net/Manuals/Custodial_Integrator/axys/CI_User_Guide.pdf`
2. ByAllAccounts / Morningstar, `Custodial Integrator Release Notes` for Axys. URL: `https://www.byallaccounts.net/Manuals/Custodial_Integrator/axys/CI_releasenotes.pdf`
3. WealthTechs, `AIA User Manual for APX Users`. URL: `https://www.wealthtechs.com/files/AIADocumentation/APX%20Guide%20-%20AIA%20User%20Manual%20For%20APX%20Users.pdf`
4. WealthTechs, `AIA User Manual for Axys Users`. URL: `https://www.wealthtechs.com/files/AIADocumentation/Axys%20Guide%20-%20AIA%20User%20Manual%20For%20Axys%20Users.pdf`
5. AdventGuru, Axys/APX and price-file articles, including `mergepri` evidence. URL: `https://adventguru.com/tag/axys/` and `https://adventguru.com/category/portfolio-management-systems/apx/`
6. AdventGuru, IMEX articles. URL: `https://adventguru.com/tag/imex/`
7. SS&C Advent, Axys product page. URL: `https://www.advent.com/solutions/axys/`
8. SS&C Advent, Advent Portfolio Exchange product page. URL: `https://www.advent.com/solutions/advent-portfolio-exchange/`
9. UNAPEN, PriceFusion Pricing and Reference Data for APX. URL: `https://unapen.com/products/pricefusion-pricing-and-reference-data-for-apx`
10. SS&C Advent, APX Reports Guide discovered in public search. URL: `https://cdn.advent.com/cms/pdfs/reports/REP_APX.pdf`

## Bottom line

The available public evidence is sufficient to produce a strong Research Mode file for Chapter 08, but not sufficient to write a definitive pricing chapter with exact native Axys/APX price schemas.

The strongest supported material is integration-layer evidence:

- Axys CI imports price files through IMEX.
- Axys CI uses `*.pri` files and `imexPrices.log` in the observed workflow.
- Axys CI release notes provide real pricing quirks involving missing/stale prices, calculated prices, fixed-income calculated-price precision, and Price vs Missing Price file routing.
- APX AIA documents price-file update modes, price-set logic, custodian-specific pricing, custodian trumping order, and custom price-file naming.
- AdventGuru provides useful consultant evidence for `mergepri` and direct-file-access caution.

The core unresolved gap remains the official native price field dictionary for Axys and APX. Until `.pri` samples, IMEX object definitions, APX public views, or official pricing manuals are available, future Chapter 08 should preserve exact field names, object names, report names, and native storage behavior as **Unknown**.

## Deep IMEX Addendum Incorporated 2026-06-30

Source: `axys_imex_deep_research.md`.

Additional pricing points:

| Topic | Addendum | Confidence |
|---|---|---:|
| `.pri` workflow | Public CI evidence supports appending prices to prior-business-day price files in `$pathpri`. | Verified for CI workflow |
| No overwrite behavior | If a security already exists in the price file for that day and already has a price, CI does not replace it. | Verified for CI workflow |
| Multiple historical days | If historical price days are delivered, CI may show one `imexPrices` tab per historical day. | Verified for CI workflow |
| Price preview fields | CI preview evidence includes Symbol, Price, Source, Price Date, and Price As-Of Date. | Verified for CI workflow |
| Candidate price fields | Live discovery should inspect symbol, type, price date, price, price source, currency, factor, quote multiplier, and price-set/source provenance where available. | Discovery guidance |
| Boundary | `.pri` workflow evidence does not prove the official native IMEX price object name or full price-file layout. | Unknown / boundary |
