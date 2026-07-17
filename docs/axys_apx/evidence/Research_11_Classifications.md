# Evidence Ledger: Classifications

This file preserves source-level support for
[`Chapter_11_Classifications.md`](../reference/Chapter_11_Classifications.md).
Conceptual models, examples, historical-drift tests, and implementation guidance
are canonical in that chapter.

## Source Register

| Source ID | Source | Scope and boundary |
|---|---|---|
| CLASS-S01 | `../axys_apx_reference_blueprint.md` | Editorial rules only. |
| CLASS-S02 | [SS&C Advent Axys](https://www.advent.com/solutions/axys/) | Vendor evidence for portfolio grouping and performance views by asset class, sector, country, and region. | No storage or effective-dating model. |
| CLASS-S03 | [APX Reports Guide](https://cdn.advent.com/cms/pdfs/reports/REP_APX.pdf) | Vendor report evidence for Equity Overview and custom-classification/industry/sector output. | Report output, not native fields. |
| CLASS-S04 | ByAllAccounts Custodial Integrator guides for Axys and APX | Integration evidence for `sec.inf`, `type.inf`, symbol/type matching, duplicates, and reserved prefixes. | CI matching behavior only. |
| CLASS-S05 | [AdvisorEngine Axys Asset Import](https://support.advisorengine.com/portal/en/kb/articles/5019002001) | Third-party export workflow with an `Asset Class` label. | Export label, not proven native field. |
| CLASS-S06 | AdventGuru integration, merger, and reporting material | Practitioner evidence for IMEX/version cautions, reference dependencies, reclassification effects, and APX access paths. | Version/site applicability requires validation. |
| CLASS-S07 | CSSI composite/group material | Dated evidence for `.CPG` composite artifacts and member entry/exit dates. | Artifact-specific; not a full classification model. |
| CLASS-S08 | [`Public_Web_Research_2026-07-17.md`](Public_Web_Research_2026-07-17.md) | Cross-topic ledger, especially WEB-20260717-011, 014, and 019. |

## Supported Claims

| Claim ID | Supported claim | Source | Confidence | Reader owner |
|---|---|---|---|---|
| CLASS-C001 | Axys can group portfolios by manager, asset class, investment objective, or firm-chosen category and can display performance by asset class, sector, country, or region. | CLASS-S02 | Verified at product/reporting level | Chapter 11 |
| CLASS-C002 | APX report evidence supports custom-classification, industry-group, and sector output, including Equity Overview. | CLASS-S03, CLASS-S08 claim WEB-20260717-019 | Verified for the guide | Chapters 11 and 14 |
| CLASS-C003 | In reviewed integration workflows, Axys/APX security identity preserves symbol and security type using `sec.inf` and `type.inf`. | CLASS-S04 | Verified for CI | Chapters 04 and 11 |
| CLASS-C004 | Security type is not equivalent to asset class, sector, industry, country, region, or another reporting classification. | CLASS-S02-S05 | High Confidence boundary | Chapters 04 and 11 |
| CLASS-C005 | CI matching excludes reserved prefixes `aw`, `br`, `ex`, `ep`, `pi`, and `rs`; this is not proven as a global product rule. | CLASS-S04 | Verified for CI only | Chapters 04 and 11 |
| CLASS-C006 | `Asset Class` is observed as an Axys export/report label in one integration workflow, not as a verified native security-master field. | CLASS-S05 | Verified as workflow label | Chapters 11 and 15 |
| CLASS-C007 | Conversion evidence mentions sectors, industries, asset classes, indexes, and composites as export/reference data, but exact objects and fields remain Unknown. | CLASS-S06 | Medium Confidence | Chapters 11 and 12 |
| CLASS-C008 | `.CPG` is a historical composite artifact with member entry/exit-date evidence. | CLASS-S07, CLASS-S08 claim WEB-20260717-011 | Verified historically; current applicability Unknown | Chapters 02, 10, 11, and 15 |
| CLASS-C009 | Practitioner evidence warns that classification/reference changes can affect historical performance and may require regeneration; this is not established as a universal rule. | CLASS-S06 | Medium Confidence implementation caution | Chapters 10 and 11 |
| CLASS-C010 | Classification extraction should distinguish scheme, value, security assignment, portfolio grouping, and report output rather than assume one native object. | CLASS-S02-S07 | High Confidence design boundary | Chapter 11 |
| CLASS-C011 | Native fields/tables, hierarchy, effective dating, historical as-of behavior, report lookup timing, and taxonomy remain unresolved. | Absence across CLASS-S02-S08 | Unknown | Chapter 11 Unknowns |

## Evidence Needed to Resolve Current Unknowns

| Gap ID | Missing evidence | Questions resolved |
|---|---|---|
| CLASS-U001 | Axys/APX security and classification exports with metadata. | Native objects, fields, lookup values, hierarchy, and assignments. |
| CLASS-U002 | REP/RDL definitions for holdings/performance by classification. | Tokens, datasets, rollups, and report-vs-raw transformations. |
| CLASS-U003 | APX schema/public-view documentation. | Tables/views, keys, and effective dating. |
| CLASS-U004 | Controlled before/after classification edits and historical reruns. | Current lookup versus historical snapshot behavior. |
| CLASS-U005 | Portfolio grouping, composite, and custom-classification configuration. | Security-level versus account-level schemes and multiple assignments. |

## Maintenance Rule

Add sources and scoped claims here. Put logical models, test procedures, and
reader guidance in Chapter 11. Put security identity in Chapter 04 and literal
labels/artifacts in Chapter 15. Git history retains the earlier narrative,
hypotheses, and proposed tests.
