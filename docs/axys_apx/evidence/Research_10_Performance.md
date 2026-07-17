# Evidence Ledger: Performance

This file preserves source-level support for
[`Chapter_10_Performance.md`](../reference/Chapter_10_Performance.md). Method
boundaries, examples, reconstruction guidance, and implementation cautions are
canonical in that chapter.

## Source Register

| Source ID | Source | Scope and boundary |
|---|---|---|
| PERF-S01 | `../axys_apx_reference_blueprint.md` | Editorial rules only. |
| PERF-S02 | [SS&C Advent Axys](https://www.advent.com/solutions/axys/) and Axys product material | Vendor capability evidence for return types, fee views, benchmarks, classifications, composites/GIPS context, and multicurrency reporting. | Does not establish report formulas, files, or storage. |
| PERF-S03 | [Advent Portfolio Exchange](https://www.advent.com/solutions/advent-portfolio-exchange/) and APX product brief | Vendor capability evidence for accounting, reporting, performance analytics, and reporting framework. | No native schemas or calculation contracts. |
| PERF-S04 | [Index Data for APX](https://cdn.advent.com/cms/pdfs/briefs/PB_INDATA.pdf) | Vendor evidence for benchmark/index data in APX analytics. | No benchmark tables or revision rules. |
| PERF-S05 | [APX Reports Guide](https://cdn.advent.com/cms/pdfs/reports/REP_APX.pdf) | Vendor guide evidence for performance, attribution, contribution, benchmark, and risk outputs. | Installed catalog, datasets, and formulas remain Unknown. |
| PERF-S06 | Historical Axys performance/report guidance, including `PERHSUM.REP` | Dated evidence for `.PRF`, `.PBF`, report artifacts, and TWR/DCF/IRR distinctions. | Historical, not a current universal contract. |
| PERF-S07 | Morningstar Axys conversion material | Third-party reconciliation/report leads and conversion observations. | Conversion scope only. |
| PERF-S08 | Zacks Advent Axys reconciliation material | Third-party setting/report clues for combined/daily and whole-period returns. | Test-design evidence, not official formulas. |
| PERF-S09 | Public Portfolio Appraisal samples incorporated 2026-07-08 | Report-output evidence separating Market Value and Accrued Interest. | No native fields or stored-value formula. |
| PERF-S10 | Short-lifecycle accounting/reporting research incorporated 2026-07-07 | Controlled demo convention for `ss`/`cs` and short exposure. | Not proof of universal native behavior. |
| PERF-S11 | [`Public_Web_Research_2026-07-17.md`](Public_Web_Research_2026-07-17.md) | Cross-topic ledger, especially WEB-20260717-002-004, 008-009, and 014-019. |

## Supported Claims

| Claim ID | Supported claim | Source | Confidence | Reader owner |
|---|---|---|---|---|
| PERF-C001 | Axys and APX support portfolio performance and reporting at product level. | PERF-S02, PERF-S03 | Verified at capability level | Chapter 10 |
| PERF-C002 | Reviewed Axys material supports time-weighted and internal-rate-of-return reporting, before/after fee views, blended/synthetic benchmarks, classification views, composite/GIPS context, and multicurrency return components. | PERF-S02, PERF-S11 | Verified at stated capability level | Chapter 10 |
| PERF-C003 | Reviewed APX material supports performance analytics, benchmark/index data, and guide-visible performance, attribution, contribution, and risk outputs. | PERF-S03-S05, PERF-S11 | Verified for the reviewed materials | Chapters 10 and 14 |
| PERF-C004 | Historical Axys evidence identifies `.PRF` and `.PBF` performance-history artifacts and `PERHSUM.REP`; exact layouts and current applicability remain Unknown. | PERF-S06, PERF-S11 claims WEB-20260717-008-009 | Verified historically; layouts and current applicability Unknown | Chapters 02, 10, 13-15 |
| PERF-C005 | Historical evidence distinguishes time-weighted Performance History from DCF/IRR-style reporting. | PERF-S06 | Verified historically | Chapter 10 |
| PERF-C006 | Morningstar and Zacks materials provide report/setting leads for reconciliation but do not establish official Axys formulas. | PERF-S07, PERF-S08 | Medium to High Confidence as test leads | Chapter 10 |
| PERF-C007 | `portperf` and `secperf` remain product-local candidate names, not verified native IMEX objects. | Absence across PERF-S02-S11 | Unknown | Chapters 10, 12, and 15 |
| PERF-C008 | Fixed-income report output separates Market Value and Accrued Interest; dirty-value reconstruction should preserve accrued interest unless site evidence proves market value already includes it. | PERF-S09 | High Confidence report interpretation; native formula Unknown | Chapters 06, 10, and 14 |
| PERF-C009 | A controlled demo may model lowercase `ss` as opening/increasing short exposure and `cs` as reducing/closing it; neither is a client external flow by default. | PERF-S10 | Design convention only | Chapters 05 and 10 |
| PERF-C010 | Performance extracts require source mechanism, report/object, parameters, period grain, row lineage, version, and stored-versus-calculated confidence. | PERF-S04-S11 | High Confidence design guidance | Chapters 10 and 12 |
| PERF-C011 | Exact report-specific formulas, storage/recalculation behavior, fee/FX/rounding settings, security-to-portfolio footing, classification timing, API/report datasets, and native fields remain unresolved. | Absence across PERF-S02-S11 | Unknown | Chapter 10 Unknowns |

## Evidence Needed to Resolve Current Unknowns

| Gap ID | Missing evidence | Questions resolved |
|---|---|---|
| PERF-U001 | Versioned IMEX/APXIX catalogs and performance exports. | Native objects, fields, grain, and stored values. |
| PERF-U002 | Axys `.REP` and APX RDL/report definitions with parameters. | Formulas, datasets, return methods, fee settings, and recalculation. |
| PERF-U003 | Controlled monthly and multi-period reruns with stored values. | Stored-versus-recalculated behavior and linking. |
| PERF-U004 | Security/classification reports with portfolio footing. | Weights, contribution, cash/residual treatment, and aggregation. |
| PERF-U005 | Benchmark exports and revision examples. | Benchmark identity, source, timing, and restatement behavior. |
| PERF-U006 | Fixed-income and multicurrency reports/exports. | Accrual, principal, local/base returns, and currency effects. |
| PERF-U007 | Composite/GIPS configuration and reports. | Account weighting, membership, fee basis, and composite calculation. |

## Maintenance Rule

Add sources and scoped claims here. Put performance explanations, reconstruction
rules, examples, and tests in Chapter 10. Put report catalogs in Chapter 14 and
literal labels in Chapter 15. Git history retains the earlier research
narrative and hypothetical object/field inventories.
