# Evidence Ledger: Corporate Actions

This file preserves source-level support for
[`Chapter_09_Corporate_Actions.md`](../reference/Chapter_09_Corporate_Actions.md).
Event explanations, examples, audit treatment, and implementation cautions are
canonical in that chapter.

## Source Register

| Source ID | Source | Scope and boundary |
|---|---|---|
| CA-S01 | `../axys_apx_reference_blueprint.md` | Editorial rules only; not product evidence. |
| CA-S02 | SS&C Advent, *Overcome the Challenges of Corporate Actions Processing with Axys* | Vendor workflow evidence for ACA with Axys; does not expose fields or final posting details. |
| CA-S03 | SS&C Advent, *Advent Corporate Actions for APX* | Vendor workflow evidence for APX holdings, ACA review/download, Reorg Utility, and Trade Blotter. |
| CA-S04 | SS&C Advent, *What is SS&C Advent Corporate Actions?* | General ACA product positioning only. |
| CA-S05 | AdventGuru conversion and merger material | Consultant evidence for exported `split.inf` files and logical `SplitDate`, `SplitSymbol`, and `SplitFactor` labels. |
| CA-S06 | FinFolio Advent conversion notes | Converter evidence that split transactions may be derived or “blown out” from `SPLIT.INF`. |
| CA-S07 | Morningstar Office, *Advent Axys Database Conversion* | Conversion evidence identifying `.cli`, `sec.inf`, `split.inf`, `.pri`, and `type.inf`. |
| CA-S08 | ByAllAccounts Axys/APX Custodial Integrator guides | Integration mappings for `SPLIT`, `JOURNAL`, `OTHER`, `rc`, and `pd`; workflow-specific. |
| CA-S09 | WealthTechs AIA manuals | Integration examples including a `;` split-comment row; not a native split schema. |
| CA-S10 | SourceForge Walnut release notes | Third-party tooling evidence that `split.inf` is a readable structured artifact; no field dictionary. |
| CA-S11 | [`Public_Web_Research_2026-07-17.md`](Public_Web_Research_2026-07-17.md) | Cross-topic public-source ledger; use each claim’s stated scope. |

## Supported Claims

| Claim ID | Supported claim | Source | Confidence | Reader owner |
|---|---|---|---|---|
| CA-C001 | ACA is a distinct corporate-action solution with documented Axys and APX integration workflows. | CA-S02-S04 | Verified at product/workflow level | Chapter 09 |
| CA-C002 | The reviewed Axys ACA workflow receives active holdings, provides reports and an Automation Results email, and can process simple/mandatory events to Trade Blotter. | CA-S02 | Verified for the vendor-described workflow | Chapter 09 |
| CA-C003 | The reviewed APX ACA workflow sends holdings to ACA, supports review/download, runs APX Reorg Utility, and posts transactions to APX Trade Blotter. | CA-S03, CA-S11 claim WEB-20260717-021 | Verified for the vendor-described workflow | Chapter 09 |
| CA-C004 | Public conversion evidence identifies `split.inf` / `SPLIT.INF` as the Axys securities-splits file. | CA-S05-S07, CA-S10 | High Confidence | Chapters 09 and 15 |
| CA-C005 | Axys conversion packages may include `.cli`, `sec.inf`, `split.inf`, `.pri`, and `type.inf`. | CA-S07 | High Confidence for that conversion workflow | Chapters 02 and 09 |
| CA-C006 | `SplitDate`, `SplitSymbol`, and `SplitFactor` are logical labels in consultant merge code, not verified official `split.inf` headers. | CA-S05 | Medium to High Confidence as logical fields; official headers Unknown | Chapters 09 and 15 |
| CA-C007 | The strongest Axys evidence treats ordinary split history as security-level factor data; it does not prove ordinary account-level split transactions exist natively. | CA-S05-S07 | High Confidence inference | Chapter 09 |
| CA-C008 | Conversion tools may materialize split transactions from `SPLIT.INF`; this is converter behavior, not proof of native storage. | CA-S06 | Medium Confidence | Chapter 09 |
| CA-C009 | Integration evidence can represent a split as a `;` comment/marker row, but that row is not an authoritative native split record. | CA-S08, CA-S09 | Medium Confidence for the workflows | Chapters 05 and 09 |
| CA-C010 | `split.inf` and ACA are separate evidence surfaces; reviewed sources do not establish that ACA writes `split.inf`. | CA-S02-S07 | High Confidence boundary | Chapter 09 |
| CA-C011 | Integration mappings support `rc` as return of capital and bond-context `pd` as principal paydown/return; native cost-basis and performance mechanics remain Unknown. | CA-S08 | Medium to High Confidence for mappings; native mechanics Unknown | Chapters 05, 09, and 10 |
| CA-C012 | A normal split is conservatively treated as quantity/price-neutral audit evidence rather than a client external flow; cash-in-lieu requires separate review. | CA-S05-S09 | High Confidence audit guidance; native formulas Unknown | Chapter 09 |
| CA-C013 | Exact split schemas, ACA fields/statuses, APX storage, transaction mappings, correction behavior, and final posting lifecycle remain unavailable. | Absence across CA-S02-S11 | Unknown | Chapter 09 Unknowns |

## Evidence Needed to Resolve Current Unknowns

| Gap ID | Missing evidence | Questions resolved |
|---|---|---|
| CA-U001 | Sanitized `split.inf` rows and version information. | Headers, date/symbol/factor conventions, reverse splits, and corrections. |
| CA-U002 | Axys `.cli`, holdings, price, and report rows around real events. | Native transaction materialization, price adjustment, and performance effects. |
| CA-U003 | APX ACA/Reorg Utility documentation and before/after blotter exports. | Action IDs, statuses, mappings, posting, reversals, and correction lifecycle. |
| CA-U004 | Axys/APX IMEX catalogs and corporate-action exports. | Native objects, fields, and import/export coverage. |
| CA-U005 | REP/RDL definitions and outputs for corporate-action review. | Report names, fields, parameters, and source equivalence. |
| CA-U006 | Cash-in-lieu, merger, spin-off, redemption, and paydown examples. | Transaction, holdings, basis, cash, and performance treatment. |

## Maintenance Rule

Add sources and scoped claims here. Put event taxonomies, worked examples,
audit treatment, and implementation recommendations in Chapter 09. Put
transaction-code interpretation in Chapter 05 and literal artifacts in Chapter
15. Git history retains the earlier narrative and proposed chapter structure.
