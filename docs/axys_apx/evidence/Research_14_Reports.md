# Evidence Ledger: Reports

This file preserves source-level support for
[`Chapter_14_Reports.md`](../reference/Chapter_14_Reports.md). Reader
explanations, report catalogs, examples, and implementation cautions are
canonical in that chapter.

## Source Register

| Source ID | Source | Type and scope | Confidence boundary |
|---|---|---|---|
| REP-S01 | `../axys_apx_reference_blueprint.md` | Repository specification. | Editorial rules only; not product evidence. |
| REP-S02 | [SS&C Advent Axys](https://www.advent.com/solutions/axys/) | Vendor product page supporting broad Axys reporting and customization capability. | Does not provide report files, fields, parameters, or formulas. |
| REP-S03 | [Advent Portfolio Exchange](https://www.advent.com/solutions/advent-portfolio-exchange/) | Vendor product page supporting broad APX reporting capability. | Does not establish report internals. |
| REP-S04 | [Advent Portfolio Exchange Reports Guide](https://cdn.advent.com/cms/pdfs/reports/REP_APX.pdf) | Vendor report guide supporting the guide-listed 29-report catalog, SSRS basis, purposes, visible labels, and listed risk measures. | Does not establish installed catalog drift, RDL names, datasets, queries, or formulas. |
| REP-S05 | [Index Data for APX](https://cdn.advent.com/cms/pdfs/briefs/PB_INDATA.pdf) | Vendor product brief supporting benchmark/model-relative reporting context. | Product capability only. |
| REP-S06 | [AdventGuru REP material](https://adventguru.com/tag/rep/) and Axys/APX category material | Practitioner evidence for Report Writer Pro, RepLang, reporting, and integration patterns. | Version and installation applicability require validation. |
| REP-S07 | [Portfolio Code in Axys Reports](https://assets.ctfassets.net/xhy36q2d1lqu/77QC4aNbyhPo9FfmjRYNzc/d00a0d6601214601543e30e34f203626/PortfolioCodetoAxys.pdf) | Consultant example supporting Axys Portfolio Appraisal customization and `$askport`. | Example-specific; not a complete REP grammar. |
| REP-S08 | [Salentica Data Broker](https://engage.salentica.com/kb/article/247-data-broker-ss-c-advent-apx-axys/) | Integration documentation supporting report/macro extraction through installed client tools and `REP32.exe`. | Connector behavior, not a universal product architecture. |
| REP-S09 | Public Axys/APX Portfolio Appraisal samples incorporated 2026-07-08. | Report-output evidence distinguishing fixed-income Market Value and Accrued Interest. | Does not establish native fields or formulas. |
| REP-S10 | [`Public_Web_Research_2026-07-17.md`](Public_Web_Research_2026-07-17.md) | Cross-topic public-source ledger. | Use the listed claim scope and date. |

## Supported Claims

| Claim ID | Supported claim | Source | Confidence | Reader owner |
|---|---|---|---|---|
| REP-C001 | Reporting is a core supported capability of Axys and APX. | REP-S02, REP-S03 | Verified at product-capability level | Chapter 14 overview |
| REP-C002 | Axys supports predefined and customizable reporting. | REP-S02, REP-S06 | Verified for broad vendor claim; High Confidence for practitioner detail | Chapter 14 Axys reports |
| REP-C003 | The public APX Reports Guide presents an SSRS-based, normalized 29-report inventory with report purposes and visible labels. | REP-S04, REP-S10 claims WEB-20260717-019 | Verified for the guide | Chapter 14 APX reports |
| REP-C004 | Report Writer Pro and RepLang are supported Axys customization paths in the reviewed evidence; APX applicability is practitioner- and version-dependent. | REP-S06, REP-S07 | Verified for Axys examples; Medium to High Confidence for APX contexts | Chapters 13 and 14 |
| REP-C005 | One Axys/APX connector uses standard reports, macros, installed client tools, and `REP32.exe` to generate extracts. | REP-S08 | Verified for that connector | Chapters 13 and 14 |
| REP-C006 | IMEX/APXIX and reports are distinct evidence surfaces; matching labels do not prove matching objects or fields. | REP-S04, REP-S08 | High Confidence boundary | Chapters 12 and 14 |
| REP-C007 | Public Axys customization evidence supports Portfolio Appraisal use of `Portfolio Code`, `Management Mode`, and `$askport`. | REP-S07 | Verified for the example | Chapters 13-15 |
| REP-C008 | Consultant examples support Axys AUM/report customization using named REP files and observed RepLang tokens. | REP-S06 | Verified for cited examples only | Chapters 13-15 |
| REP-C009 | The APX guide exposes attribution, contribution, risk, holdings, transaction, allocation, income, and client-reporting labels and sections. | REP-S04 | Verified as report output | Chapters 10, 14, and 15 |
| REP-C010 | Public Portfolio Appraisal output treats fixed-income Market Value and Accrued Interest as separate concepts; subtotal/total presentation strongly implies their combination while Market Value remains clean-value presentation. | REP-S09 | High Confidence report interpretation | Chapters 06, 10, and 14 |
| REP-C011 | A report name or visible label does not establish a native file field, IMEX object, APX database column, stored function, or API field. | REP-S04, REP-S08 | High Confidence boundary | Chapters 12, 14, and 15 |
| REP-C012 | Exact report formulas, stored-versus-recalculated behavior, report datasets, source objects, and report-to-IMEX equivalence remain unresolved. | Absence across REP-S02-S09 | Unknown | Chapter 14 Unknowns |
| REP-C013 | Historical Axys evidence identifies `PERHSUM.REP` as Performance History for Selected Time Periods in an Axys 3.6 context. | REP-S10 claim WEB-20260717-009 | Verified as historical evidence | Chapters 10 and 14 |

## Evidence Needed to Resolve Current Unknowns

| Gap ID | Missing evidence | Questions it would resolve |
|---|---|---|
| REP-U001 | Installed Axys and APX report catalogs. | Current report names, local additions, and catalog drift. |
| REP-U002 | Representative Axys `.REP` files and Report Writer definitions. | Parameters, fields, source logic, formulas, and RepLang behavior. |
| REP-U003 | APX RDL files, SSRS catalog metadata, datasets, and queries. | Report sources, stored procedures/views, formulas, and deployment differences. |
| REP-U004 | Matching report output and IMEX/APXIX extracts. | Report-to-extract equivalence, transformations, rounding, and omissions. |
| REP-U005 | Controlled stored-versus-rerun performance examples. | Whether specific reports read stored values, recalculate, or combine both. |
| REP-U006 | Multi-currency and fixed-income report definitions and outputs. | Currency basis, accrued-interest treatment, and native total-value formulas. |
| REP-U007 | Upgrade and customization records. | Whether upgrades replace, migrate, or invalidate custom reports. |

## Maintenance Rule

Add new material as a source entry and one or more scoped claims. Put report
catalogs, explanations, worked examples, and implementation recommendations in
Chapter 14. Put literal report labels in Chapter 15 and REP/RepLang behavior in
Chapter 13. Git history retains the earlier research narrative and proposed
chapter structure.
