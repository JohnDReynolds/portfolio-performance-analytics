# Evidence Ledger: Axys Architecture

This file preserves source-level support for
[`Chapter_02_Axys_Architecture.md`](../reference/Chapter_02_Axys_Architecture.md).
Architecture explanations, diagrams, extraction guidance, and implementation
cautions are canonical in that chapter.

## Source Register

| Source ID | Source | Scope and boundary |
|---|---|---|
| AXYS-S01 | `../axys_apx_reference_blueprint.md` | Editorial rules only. |
| AXYS-S02 | [SS&C Advent Axys](https://www.advent.com/solutions/axys/) and Axys product brief | Vendor evidence for product role, proprietary-database positioning, accounting choices, performance, grouping, and Report Writer Pro. | Does not expose physical storage. |
| AXYS-S03 | [Advent 2007 SEC filing](https://www.sec.gov/Archives/edgar/data/1002225/000110465907025400/a07-7653_110ka.htm) | Historical product descriptions and Axys/APX contrast. | Dated evidence. |
| AXYS-S04 | AdventGuru IMEX, Axys, and Report Writer material | Practitioner evidence for Axys version/file history, IMEX formats, direct-file risk, RepLang, `.REP`/`.RPW`, and conversion. | Version/site applicability requires validation. |
| AXYS-S05 | [Portfolio Code in Axys Reports](https://assets.ctfassets.net/xhy36q2d1lqu/77QC4aNbyhPo9FfmjRYNzc/d00a0d6601214601543e30e34f203626/PortfolioCodetoAxys.pdf) | Consultant example for Axys RepLang, `AMAN.REP`, paths, and `$:fileo`. | Example-specific. |
| AXYS-S06 | Salentica Data Broker documentation | Connector evidence for local/VM/server deployment, client tools, reports/macros, and REP32. | Connector scope only. |
| AXYS-S07 | ByAllAccounts Custodial Integrator Axys guides | Integration evidence for local workflow, `imex32.exe`, `pospos32.exe`, Axys files/paths, and imports. | CI workflow only. |
| AXYS-S08 | FinFolio and Morningstar conversion material | Migration evidence for CLI/PRI/INF/PRF/GRP and more specific Axys file families. | Conversion scope; not layouts. |
| AXYS-S09 | Historical industry and Advent presentation material | Dated flat-file/proprietary architecture descriptions. | Historical positioning, not current internals. |
| AXYS-S10 | [`Public_Web_Research_2026-07-17.md`](Public_Web_Research_2026-07-17.md) | Cross-topic ledger, especially WEB-20260717-001 and 008-014. |

## Supported Claims

| Claim ID | Supported claim | Source | Confidence | Reader owner |
|---|---|---|---|---|
| AXYS-C001 | Axys is a portfolio accounting, reporting, performance-measurement, and reconciliation product with configurable accounting/reporting capabilities. | AXYS-S02, AXYS-S03 | Verified at product level | Chapter 02 |
| AXYS-C002 | Current vendor wording supports proprietary-database positioning, while dated and operational evidence is strongly file-oriented; exact current physical internals remain Unknown. | AXYS-S02, AXYS-S04, AXYS-S07-S10 | Verified as scoped descriptions | Chapter 02 |
| AXYS-C003 | Practitioner evidence describes Axys v1 open-text, v2 binary, and v3-era IMEX behavior plus 3.7-to-3.8 conversion risk. | AXYS-S04 | Medium Confidence, version-specific | Chapters 02 and 12 |
| AXYS-C004 | Direct native-file read/write is a practitioner-known but brittle, version-sensitive path and should not be the default integration surface. | AXYS-S04 | High Confidence caution | Chapter 02 |
| AXYS-C005 | Axys reporting evidence supports Report Writer Pro, RepLang, `.REP` files, and historical `.RPW` distinctions. | AXYS-S02, AXYS-S04, AXYS-S05, AXYS-S10 | Verified/High Confidence by artifact | Chapters 02 and 13 |
| AXYS-C006 | Reviewed CI evidence identifies `imex32.exe`, `pospos32.exe`, `$pathexe`, `$pathtrn`, `$pathcli`, `$pathinf`, `$pathpri`, and `$pathlog`. | AXYS-S07 | Verified for CI | Chapters 02, 12, and 15 |
| AXYS-C007 | Reviewed integration evidence uses IMEX for transaction, position, lot, and price workflows and uses REP/report paths for report-shaped extraction. | AXYS-S06, AXYS-S07 | Verified for reviewed workflows | Chapters 02, 12, and 13 |
| AXYS-C008 | Migration and historical evidence identifies `.cli`, `.pri`, `sec.inf`, `type.inf`, `split.inf`, `.PRF`, `.PBF`, `.GRP`, `.CPG`, and named report artifacts, without complete layouts. | AXYS-S08-S10 | Verified or High Confidence for cited contexts; complete layouts Unknown | Chapters 02 and 15 |
| AXYS-C009 | Local machine, VM, server, and connector-host deployment patterns are evidenced, but current hosted/cloud/hybrid topology is not established generally. | AXYS-S06, AXYS-S07 | Verified for cited contexts; general topology Unknown | Chapter 02 |
| AXYS-C010 | Values from native files, IMEX, REP, reports, connectors, or other surfaces are not assumed equivalent until reconciled with version and parameters. | AXYS-S04-S10 | High Confidence lineage boundary | Chapters 01, 02, and 12-14 |
| AXYS-C011 | Complete file layouts, locking, services, scheduling, backup/restore, IMEX schemas, REP grammar, and stored/calculated behavior remain unavailable. | Absence across AXYS-S02-S10 | Unknown | Chapter 02 Unknowns |

## Evidence Needed to Resolve Current Unknowns

| Gap ID | Missing evidence | Questions resolved |
|---|---|---|
| AXYS-U001 | Versioned installation and administration guides. | Topology, directories, services, locking, scheduling, backup, and concurrency. |
| AXYS-U002 | Sanitized native files plus version-specific layouts. | Physical formats, keys, precision, and change history. |
| AXYS-U003 | Live IMEX catalog, definitions, exports, and logs. | Objects, fields, formats, validation, and automation. |
| AXYS-U004 | RepLang guide and representative `.REP`/`.RPW` sources. | Grammar, fields, execution, and report dependencies. |
| AXYS-U005 | Controlled source-surface reconciliation. | Native/IMEX/REP/report equivalence and stored/calculated behavior. |

## Maintenance Rule

Add sources and scoped claims here. Put system diagrams, operational guidance,
and comparison narrative in Chapter 02. Put literal artifacts in Chapter 15 and
interface-specific behavior in Chapters 12-14. Git history retains the earlier
file inventories, proposed diagrams, and chapter-writing guidance.
