# Evidence Ledger: IMEX and APXIX

This file preserves source-level support for
[`Chapter_12_Imex.md`](../reference/Chapter_12_Imex.md). Discovery procedures,
workflow explanations, normalized schema guidance, and implementation cautions
are canonical in that chapter.

## Source Register

| Source ID | Source | Scope and boundary |
|---|---|---|
| IMEX-S01 | `../axys_apx_reference_blueprint.md` | Editorial rules only. |
| IMEX-S02 | ByAllAccounts Custodial Integrator Axys User Guide | Integration evidence for `imex32.exe`, Axys files/paths, transaction/position/price imports, logs, translations, and operational errors. | CI workflow only. |
| IMEX-S03 | ByAllAccounts Custodial Integrator APX User Guide | Integration evidence for `apxix.exe`, APX groups, security/type exports, transactions, prices, positions, lots, and authentication. | CI workflow only. |
| IMEX-S04 | WealthTechs AIA manuals for Axys and APX | Integration evidence for `CDIhold.rep`, `$pathCDI`, APXIX, `imexhist.log`, blotters, SQL connection settings, and `.veh` security imports. | AIA workflow only. |
| IMEX-S05 | Salentica Data Broker documentation | Connector evidence for reports/macros, `REP32.exe`, client tools, and tested versions. | Connector behavior only. |
| IMEX-S06 | AdventGuru IMEX, REP, and APX reporting material | Practitioner evidence for Axys file-version risk, APX IMEX continuity, fixed-format changes, SQL/reporting access paths, and RepLang alternatives. | Version/site applicability requires validation. |
| IMEX-S07 | Axys product page and brief | Product/reporting context only; no low-level IMEX contract. |
| IMEX-S08 | APX product and release material | Product, SQL-oriented platform, and versioned REST capability context. | REST capability is separate from IMEX/APXIX. |
| IMEX-S09 | APX Market Data Manager / Interactive Data RemotePlus material | Adjacent price/reference-data path; not IMEX. |
| IMEX-S10 | FinFolio and other conversion material | Migration leads for Axys/APX file families and reference exports. | Not an official object catalog. |
| IMEX-S11 | [`Public_Web_Research_2026-07-17.md`](Public_Web_Research_2026-07-17.md) | Cross-topic ledger, especially WEB-20260717-002-007, 013, and WEB-20260718-001. |

## Supported Claims

| Claim ID | Supported claim | Source | Confidence | Reader owner |
|---|---|---|---|---|
| IMEX-C001 | A reviewed integration guide expands IMEX as the Axys Import/Export utility. | IMEX-S02 | Verified for that terminology | Chapter 12 |
| IMEX-C002 | The reviewed Axys CI workflow uses `imex32.exe` for transaction, position, position-lot, and price imports and exposes type-specific logs. | IMEX-S02 | Verified for CI | Chapter 12 |
| IMEX-C003 | Reviewed Axys integration evidence identifies `topost.trn`, `ptopost.trn`, `*.cli`, `.pos`, `sec.inf`, `type.inf`, `*.pri`, `$pathtrn`, `$pathcli`, `$pathinf`, `$pathpri`, and `$pathlog`. | IMEX-S02 | Verified for CI | Chapters 02, 04-08, 12, and 15 |
| IMEX-C004 | Axys CI imports can fail when a target file is open and require log review before acceptance; position posting may replace configured `.pos` files. | IMEX-S02 | Verified for CI | Chapter 12 |
| IMEX-C005 | The reviewed APX workflows use normalized APXIX/ApxIx naming, including `apxix.exe` / `APXIX.exe`, and record IMEX history in `imexhist.log`. | IMEX-S03, IMEX-S04, IMEX-S11 | Verified for reviewed workflows | Chapters 03, 12, and 15 |
| IMEX-C006 | APX integration evidence covers security/type export, portfolio/group codes, transactions, prices, positions, lots, and multiple review blotters. | IMEX-S03, IMEX-S04 | Verified for reviewed workflows | Chapters 03-08 and 12 |
| IMEX-C007 | APXIX authentication, DataPort/BackOffice prerequisites, SQL connection settings, and Apxix logs can be operational dependencies in reviewed workflows. | IMEX-S03, IMEX-S04 | Verified for those workflows | Chapter 12 |
| IMEX-C008 | `MISSINGPRICES_yyyymmdd.csv` and `SECTRANSLATIONS_yyyymmdd.csv` are CI diagnostics, not proven native IMEX objects. | IMEX-S02 | Verified for CI; native status Unknown | Chapters 04, 08, 12, and 15 |
| IMEX-C009 | Axys direct file access is version-sensitive; APX IMEX continuity and fixed-format behavior vary by product era. | IMEX-S06 | Medium Confidence practitioner evidence | Chapters 02, 03, and 12 |
| IMEX-C010 | REP/RepLang, reports/macros, SQL-related paths, REST, and market-data utilities are distinct or adjacent extraction surfaces, not synonyms for IMEX/APXIX. | IMEX-S04-S09 | High Confidence boundary | Chapters 03, 12-14 |
| IMEX-C011 | Official REST releases establish versioned APX API capability but do not prove endpoint equivalence to IMEX objects or site entitlement. | IMEX-S08, IMEX-S11 | Verified at release-capability level | Chapters 03 and 12 |
| IMEX-C012 | APX transaction input requiredness, including Mark-to-Market behavior, can be version/workflow-specific. | IMEX-S11 claim WEB-20260717-013 | Verified for the cited documentation | Chapters 05 and 12 |
| IMEX-C013 | No reviewed public source provides one complete authoritative Axys/APX IMEX object-and-field dictionary. | Absence across IMEX-S02-S11 | Unknown | Chapters 12 and 15 |
| IMEX-C014 | A reliable implementation should maintain a versioned, installation-specific catalog with exact object/field labels, direction, type/width, examples, source, and confidence. | IMEX-S02-S11 | High Confidence design guidance | Chapter 12 |
| IMEX-C015 | The APX AIA guide describes selected APX identifiers as case-sensitive while describing its Transaction Translation evaluator as not case-sensitive; evaluator behavior does not establish native identifier equivalence. | IMEX-S04, IMEX-S11 claim WEB-20260718-001 | Verified for AIA workflow; broader native rules Unknown | Chapters 04, 05, 12, and 15 |

## Evidence Needed to Resolve Current Unknowns

| Gap ID | Missing evidence | Questions resolved |
|---|---|---|
| IMEX-U001 | Live Axys `imex32.exe` and APXIX catalogs by version. | Exact objects, fields, directionality, requiredness, and formats. |
| IMEX-U002 | Saved import/export definitions, macros, layouts, and logs. | Command syntax, field order, validation, errors, and repeatability. |
| IMEX-U003 | Sanitized exports/imports for each domain. | Headers, values, precision, dates, keys, and version drift. |
| IMEX-U004 | APX schema/public-view/API documentation and entitlements. | Relationship among APXIX, SQL, stored functions, and REST. |
| IMEX-U005 | Matching IMEX/APXIX and REP/report output. | Report transformations, stored/calculated values, and reconciliation. |
| IMEX-U006 | Installed utility/version inventory and hosted-access constraints. | Availability, authentication, client tools, and operational support. |

## Maintenance Rule

Add sources and scoped claims here. Put discovery steps, workflow diagrams,
normalized schemas, and implementation recommendations in Chapter 12. Put REP
behavior in Chapter 13, reports in Chapter 14, and literal artifacts in Chapter
15. Git history retains the earlier narrative, citation crosswalk, candidate
field inventories, and proposed live-catalog procedure.
