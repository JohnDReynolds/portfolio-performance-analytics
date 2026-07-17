# Evidence Ledger: REP and RepLang

This file preserves source-level support for
[`Chapter_13_Rep.md`](../reference/Chapter_13_Rep.md). REP explanations,
examples, extraction guidance, and implementation cautions are canonical in
that chapter. Report catalogs belong in Chapter 14.

## Source Register

| Source ID | Source | Type and scope | Confidence boundary |
|---|---|---|---|
| REPX-S01 | `../axys_apx_reference_blueprint.md` | Repository specification. | Editorial rules only. |
| REPX-S02 | [SS&C Advent Axys](https://www.advent.com/solutions/axys/) | Vendor product page supporting predefined/custom reports and Report Writer Pro. | No grammar, file catalog, or runtime contract. |
| REPX-S03 | [Advent Portfolio Exchange](https://www.advent.com/solutions/advent-portfolio-exchange/) | Vendor product page supporting standard/custom reports, packaging, and analytics. | Does not establish REP applicability. |
| REPX-S04 | [Axys 3.8.7 release](https://www.advent.com/news-and-insights/blog/a-new-version-of-axys-just-in-time-for-upgrade-season/) | Vendor release evidence for reporting and multicurrency enhancements. | Release-specific capability only. |
| REPX-S05 | [Portfolio Code in Axys Reports](https://assets.ctfassets.net/xhy36q2d1lqu/77QC4aNbyhPo9FfmjRYNzc/d00a0d6601214601543e30e34f203626/PortfolioCodetoAxys.pdf) | Consultant technical example for Axys `.REP` files, RepLang, copying/editing, `AMAN.REP`, `$:fileo`, and `#~8portmv`. | Example-specific; path and workflow are not universal. |
| REPX-S06 | [Salentica Data Broker](https://engage.salentica.com/kb/article/247-data-broker-ss-c-advent-apx-axys/) | Connector documentation for reports, macros, `REP32.exe`, RepLang, and installed client tools. | Connector behavior only. |
| REPX-S07 | [AdventGuru RepLang editing](https://adventguru.com/2024/09/09/using-visual-studio-code-to-modify-advent-replang-reports-in-axys-and-apx/) and related Axys/APX material | Practitioner evidence for direct source editing, APX keyword expansion, and multiple APX reporting/data-access paths. | Version and support boundaries require validation. |
| REPX-S08 | [Microsoft SSRS](https://learn.microsoft.com/en-us/sql/reporting-services/create-deploy-and-manage-mobile-and-paginated-reports) | Primary documentation for SSRS generally. | Not evidence of APX-specific datasets or deployment. |
| REPX-S09 | [`Public_Web_Research_2026-07-17.md`](Public_Web_Research_2026-07-17.md) | Cross-topic public-source ledger. | Use the listed claim scope and date. |

## Supported Claims

| Claim ID | Supported claim | Source | Confidence | Reader owner |
|---|---|---|---|---|
| REPX-C001 | Axys reports in the reviewed example use RepLang and `.REP` files. | REPX-S05 | Verified for the example | Chapter 13 |
| REPX-C002 | `AMAN.REP` is the Assets Under Management report in the reviewed Axys example. | REPX-S05 | Verified for the example | Chapters 13-15 |
| REPX-C003 | The reviewed Axys workflow copies a standard report, edits the copy in a text editor, and runs it through Custom / Any Report. | REPX-S05 | Verified for the example | Chapter 13 |
| REPX-C004 | In the reviewed report, `$:fileo` displays portfolio code and `#~8portmv` prints portfolio market value. | REPX-S05 | Verified for the example | Chapters 13 and 15 |
| REPX-C005 | Axys supports Report Writer Pro and predefined/custom reporting at product level. | REPX-S02 | Verified at product-capability level | Chapter 13 |
| REPX-C006 | Practitioner evidence supports continued RepLang/Report Writer Pro use in APX, with more APX keywords than Axys. | REPX-S07 | Medium Confidence; exact lists and versions Unknown | Chapter 13 |
| REPX-C007 | APX reporting/data access can involve SSRS, SQL-related access, Stored Accounting Functions, Public Views, REST, dashboards, packaging, and RepLang paths depending on deployment. | REPX-S03, REPX-S07, REPX-S09 | Confidence varies by surface: Verified capability to Medium Confidence practitioner evidence | Chapters 03 and 13 |
| REPX-C008 | One Axys/APX connector uses standard reports, macros, `REP32.exe`, RepLang scripting, and installed client tools for extraction. | REPX-S06 | Verified for that connector | Chapters 13 and 14 |
| REPX-C009 | A working historical REP32 command pattern and switches are publicly evidenced. | REPX-S09 claim WEB-20260717-010 | Verified for the cited pattern; full option set Unknown | Chapter 13 |
| REPX-C010 | Historical Axys evidence distinguishes `.REP` and Report Writer Pro `.RPW` artifacts and identifies `PERHSUM.REP`. | REPX-S09 claim WEB-20260717-009 | Verified historical evidence | Chapters 10, 13, and 14 |
| REPX-C011 | REP-derived extraction is useful when values must tie to visible reports, but labels and variables do not establish underlying IMEX fields or native semantics. | REPX-S05, REPX-S06 | High Confidence boundary | Chapters 12-15 |
| REPX-C012 | REP-derived extracts require report identity, version, parameters, layout, row lineage, and stored-versus-recalculated status for reproducibility. | REPX-S05, REPX-S06 | High Confidence design guidance | Chapter 13 |
| REPX-C013 | Public practitioner evidence records permission/elevation and PDF-printer failure modes in particular REP32 workflows. | REPX-S07 | Medium Confidence; environment-specific | Chapter 13 quirks |
| REPX-C014 | The full RepLang grammar, macro syntax, complete REP32 option set, installed report libraries, and APX REP/SSRS boundary remain unavailable. | Absence across REPX-S02-S09 | Unknown | Chapter 13 Unknowns |

## Evidence Needed to Resolve Current Unknowns

| Gap ID | Missing evidence | Questions it would resolve |
|---|---|---|
| REPX-U001 | Current RepLang Programmer's Guide and keyword lists by product/version. | Grammar, functions, variables, compatibility, and APX extensions. |
| REPX-U002 | Installed Axys/APX report directories and representative `.REP`/`.RPW` files. | File catalogs, locations, source syntax, and local customization. |
| REPX-U003 | Macro files and complete REP32 command documentation. | Unattended execution, parameters, output formats, and error handling. |
| REPX-U004 | APX reporting administration documentation plus RDL/catalog metadata. | REP, SSRS, packaging, hosted, and client-execution boundaries. |
| REPX-U005 | Matching REP output, IMEX/APXIX output, and source reports. | Transformations, field equivalence, row lineage, and reconciliation. |
| REPX-U006 | Controlled performance reports and stored/exported values. | Stored-versus-recalculated performance behavior. |
| REPX-U007 | Current SS&C customization and support policy. | Supported editing, upgrade, and troubleshooting boundaries. |

## Maintenance Rule

Add new material as a source entry and scoped claim. Keep executable names,
literal expressions, and report files exact. Put report catalogs in Chapter 14,
literal tokens in Chapter 15, and reader guidance in Chapter 13. Git history
retains the earlier research narrative, proposed outline, and fact-transfer
tables.
