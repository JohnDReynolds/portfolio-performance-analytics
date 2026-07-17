# Evidence Ledger: APX Architecture

This file preserves source-level support for
[`Chapter_03_APX_Architecture.md`](../reference/Chapter_03_APX_Architecture.md).
Architecture explanations, access-surface comparisons, version tables, and
implementation cautions are canonical in that chapter.

## Source Register

| Source ID | Source | Scope and boundary |
|---|---|---|
| APX-S01 | `../axys_apx_reference_blueprint.md` | Editorial rules only. |
| APX-S02 | [Advent Portfolio Exchange](https://www.advent.com/solutions/advent-portfolio-exchange/) and APX product brief | Vendor evidence for integrated platform, accounting/reporting, client management, reports, and packaging. | No internal schema. |
| APX-S03 | [APX Reports Guide](https://cdn.advent.com/cms/pdfs/reports/REP_APX.pdf) | Vendor evidence for SSRS-based guide reports, report inventory, and visible labels. | No RDLs, datasets, or formulas. |
| APX-S04 | Historical Advent SEC filing and River Road/APX client story | Dated evidence for relational/SQL-based centralized platform positioning and Axys migration contrast. | Historical, not a current schema contract. |
| APX-S05 | Salentica Data Broker and connector setup material | Integration evidence for REP32, reports/macros, client tools, `.mac`/`.scr`, IMEX, and tested versions. | Connector scope only. |
| APX-S06 | AdventGuru APX/IMEX/integration material | Practitioner evidence for IMEX continuity, APX-to-Axys exports, SQL/public views/stored functions, SSRS, REST, and RepLang. | Supportability and version scope require validation. |
| APX-S07 | ByAllAccounts and WealthTechs APX integration guides | Workflow evidence for APXIX, logs, security/type exports, imports, and blotters. | Integration scope only. |
| APX-S08 | APX 3.0 release coverage and later official APX releases | Dated SSRS direction and official REST capability progression from 21.1 through 23.1. | Release capability, not endpoint contract. |
| APX-S09 | 2024 Genesis platform coverage | Current product/platform positioning only; no schema or topology implication. |
| APX-S10 | SS&C ACA/APX material | Vendor workflow evidence for holdings-to-ACA, Reorg Utility, and Trade Blotter. | Corporate-action workflow only. |
| APX-S11 | [`Public_Web_Research_2026-07-17.md`](Public_Web_Research_2026-07-17.md) | Cross-topic ledger, especially WEB-20260717-001-007, 019, and 021. |

## Supported Claims

| Claim ID | Supported claim | Source | Confidence | Reader owner |
|---|---|---|---|---|
| APX-C001 | APX is an integrated portfolio/client-management, accounting, reporting, and performance platform spanning front-, middle-, and back-office positioning. | APX-S02 | Verified at product level | Chapter 03 |
| APX-C002 | Historical sources support relational/SQL-based centralized platform positioning; exact current database topology and schema remain Unknown. | APX-S04, APX-S11 claim WEB-20260717-001 | Verified historically | Chapter 03 |
| APX-C003 | The reviewed APX Reports Guide describes guide-covered investment-management reports as SSRS-based. | APX-S03 | Verified for the guide | Chapters 03 and 14 |
| APX-C004 | Reviewed connector workflows use installed client tools, `REP32.exe`, standard reports, macros, and RepLang; this is not proof that REP32 runs all APX reporting. | APX-S05 | Verified for the connector | Chapters 03 and 13 |
| APX-C005 | Reviewed APX workflows use IMEX/APXIX and can import connector setup artifacts such as `.mac` and `.scr`; exact native objects remain Unknown. | APX-S05-S07 | Verified/High Confidence for workflows | Chapters 03 and 12 |
| APX-C006 | Practitioner conversion evidence supports APX-to-Axys v3-format exports for reference/performance categories, with version-dependent limitations. | APX-S06 | Medium Confidence | Chapters 03 and 12 |
| APX-C007 | APX data access may involve SSRS, REP/RepLang, SQL/public views/stored accounting functions, IMEX/APXIX, REST, or connectors depending on version and deployment. | APX-S03, APX-S05-S08 | Mixed confidence by surface | Chapter 03 |
| APX-C008 | Official releases establish expanding REST capability from APX 21.1 through 23.1, but not endpoints, schemas, licensing, entitlements, or site availability. | APX-S08, APX-S11 claims WEB-20260717-002-005 | Verified at release-capability level | Chapters 03 and 12 |
| APX-C009 | APXIX naming and `imexhist.log` are verified in reviewed integration guides. | APX-S07, APX-S11 claims WEB-20260717-006-007 | Verified for those workflows | Chapters 03, 12, and 15 |
| APX-C010 | Genesis references establish platform positioning only and do not prove schema, API, report-stack, or IMEX changes for an installed site. | APX-S09 | High Confidence boundary | Chapter 03 |
| APX-C011 | The reviewed ACA workflow sends holdings to ACA, uses Reorg Utility, and posts to Trade Blotter. | APX-S10, APX-S11 claim WEB-20260717-021 | Verified for the workflow | Chapters 03 and 09 |
| APX-C012 | Report names and labels do not establish tables, fields, views, stored functions, or API properties. | APX-S03-S08 | High Confidence boundary | Chapters 03, 14, and 15 |
| APX-C013 | Exact topology, schemas, functions, RDL datasets, IMEX objects, API contracts, direct-SQL supportability, hosted access, and stored/calculated behavior remain unavailable. | Absence across APX-S02-S11 | Unknown | Chapter 03 Unknowns |

## Evidence Needed to Resolve Current Unknowns

| Gap ID | Missing evidence | Questions resolved |
|---|---|---|
| APX-U001 | Versioned APX architecture/admin and deployment documentation. | Services, topology, permissions, hosted/cloud differences, and scheduling. |
| APX-U002 | Schema/public-view/stored-function documentation and sanitized metadata. | Tables, views, functions, keys, lineage, and direct-SQL supportability. |
| APX-U003 | APXIX catalogs, definitions, exports, and logs. | Native objects, fields, formats, and version drift. |
| APX-U004 | RDL/report catalog and REP sources. | Dataset/query dependencies, report engines, parameters, and formulas. |
| APX-U005 | Versioned API/OpenAPI and entitlement documentation. | Endpoints, schemas, authentication, roles, and availability. |
| APX-U006 | Controlled cross-surface reconciliations. | SSRS/REP/SQL/APXIX/API equivalence and stored/calculated behavior. |

## Maintenance Rule

Add sources and scoped claims here. Put platform comparisons, diagrams, access
guidance, and version tables in Chapter 03. Put interface behavior in Chapters
12-14 and literal artifacts in Chapter 15. Git history retains the earlier
report catalogs, field-label tables, and chapter-drafting material.
