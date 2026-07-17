# Evidence Ledger: Security Master

This file preserves source-level support for
[`Chapter_04_Security_Master.md`](../reference/Chapter_04_Security_Master.md).
Matching guidance, examples, field interpretation, and implementation cautions
are canonical in that chapter.

## Source Register

| Source ID | Source | Scope and boundary |
|---|---|---|
| SEC-S01 | `../axys_apx_reference_blueprint.md` | Editorial rules only. |
| SEC-S02 | ByAllAccounts Custodial Integrator installation/user guides for Axys and APX | Integration evidence for `imex32.exe`, `apxix.exe`, `sec.inf`, `type.inf`, symbol/type matching, translations, duplicates, diagnostics, and reserved prefixes. | CI behavior only. |
| SEC-S03 | WealthTechs AIA manuals for Axys and APX | Integration evidence for `.veh` files using `sec.inf` layout and import options. | AIA behavior only. |
| SEC-S04 | [AdvisorEngine Axys Asset Import](https://support.advisorengine.com/portal/en/kb/articles/5019002001) | Export workflow with security/type/asset-class labels. | Export labels, not native fields. |
| SEC-S05 | Morningstar Axys conversion guide | Conversion evidence for `sec.inf`, user-defined security names, and selectable accrued-interest-related data. | Does not expose exact fields. |
| SEC-S06 | Salentica Data Broker documentation | Connector evidence for REP32, reports/macros, and RepLang extraction. | Connector scope only. |
| SEC-S07 | AdventGuru security, merger, and APX reporting material | Practitioner evidence for lookup dependencies, reclassification effects, public views, SQL/REST paths, and Report Writer/RepLang. | Version/site applicability requires validation. |
| SEC-S08 | APX and Axys product material | Product/security-master context only; no native dictionary. |
| SEC-S09 | [`Public_Web_Research_2026-07-17.md`](Public_Web_Research_2026-07-17.md) | Cross-topic ledger, especially WEB-20260717-002 and 006-007. |

## Supported Claims

| Claim ID | Supported claim | Source | Confidence | Reader owner |
|---|---|---|---|---|
| SEC-C001 | Reviewed Axys CI workflows use `imex32.exe` to export Security (`sec.inf`) and Security Type (`type.inf`) data. | SEC-S02 | Verified for CI | Chapters 04 and 12 |
| SEC-C002 | Reviewed APX CI workflows use `apxix.exe` to export Security (`sec.inf`) and Security Type (`type.inf`) data. | SEC-S02, SEC-S09 claims WEB-20260717-006-007 | Verified for CI | Chapters 04 and 12 |
| SEC-C003 | In reviewed integration workflows, product security identity preserves symbol plus security type; this does not prove the formal native primary key. | SEC-S02 | Verified for CI; native key Unknown | Chapter 04 |
| SEC-C004 | CI translations can use external ticker, CUSIP, name, institution, and account context and target Axys/APX symbol and type. | SEC-S02 | Verified for CI | Chapters 04 and 15 |
| SEC-C005 | APX CI translation precedence and direct ticker/CUSIP-to-symbol matching are documented for that workflow. | SEC-S02 | Verified for APX CI | Chapter 04 |
| SEC-C006 | Axys CI can encounter duplicate matches from shared symbols with different types, ticker/CUSIP duplicates, or multiple translations; symbol-only joins are unsafe. | SEC-S02 | Verified for Axys CI | Chapter 04 |
| SEC-C007 | Account-specific Axys translations can constrain later global/account mappings in the reviewed workflow. | SEC-S02 | Verified for Axys CI | Chapter 04 |
| SEC-C008 | Reserved prefixes `aw`, `br`, `ex`, `ep`, `pi`, and `rs` are excluded by reviewed CI matching; this is not a universal product rule. | SEC-S02 | Verified for CI only | Chapters 04 and 11 |
| SEC-C009 | `MISSINGPRICES_yyyymmdd.csv` and `SECTRANSLATIONS_yyyymmdd.csv` expose useful integration identifier fields but are not native security-master schemas. | SEC-S02 | Verified for CI | Chapters 04, 08, 12, and 15 |
| SEC-C010 | Reviewed AIA workflows create `.veh` files using a `sec.inf` layout and offer add/update/replace import choices. | SEC-S03 | Verified for AIA | Chapters 04 and 15 |
| SEC-C011 | Axys export/report labels such as `Security`, `Sec Type Code`, `Security Symbol`, `Security Type`, and `Asset Class` are workflow evidence, not proven native fields. | SEC-S04 | Verified as export labels | Chapters 04, 11, and 15 |
| SEC-C012 | Morningstar conversion evidence supports selectable user-defined names and accrued-interest-related security data in Axys exports, without exact field names. | SEC-S05 | Verified for conversion; fields Unknown | Chapters 04 and 10 |
| SEC-C013 | Practitioner evidence reports industry-group/sector lookup dependencies and historical-performance risk from reclassification. | SEC-S07 | Medium Confidence implementation caution | Chapters 04 and 11 |
| SEC-C014 | APX public views/SQL and official security-master REST loading capability are evidenced, but view/endpoint fields, keys, entitlements, and coverage remain Unknown. | SEC-S07, SEC-S09 claim WEB-20260717-002 | Verified for REST capability; Medium Confidence for access surfaces; contracts Unknown | Chapters 03 and 04 |
| SEC-C015 | Complete `sec.inf`/`type.inf` layouts, official IMEX object names, native keys, import requiredness, security-type dictionary, and fixed-income/derivative fields remain unavailable. | Absence across SEC-S02-S09 | Unknown | Chapter 04 Unknowns |

## Evidence Needed to Resolve Current Unknowns

| Gap ID | Missing evidence | Questions resolved |
|---|---|---|
| SEC-U001 | Sanitized Axys/APX security/type exports and version metadata. | Native fields, keys, types, requiredness, and layouts. |
| SEC-U002 | Live IMEX/APXIX object catalogs and definitions. | Official object names, import/update behavior, and validation. |
| SEC-U003 | APX schema/public-view/API documentation. | Native security keys, views, endpoints, roles, and coverage. |
| SEC-U004 | REP/RDL security-master extracts and source definitions. | Report fields, source equivalence, and completeness. |
| SEC-U005 | Fixed-income, derivatives, identifiers, and classification-rich samples. | Coupon/maturity/factor/accrual fields, alternate identifiers, and lookup behavior. |
| SEC-U006 | Controlled duplicate, translation, reclassification, and identifier-change examples. | Matching precedence, history, downstream effects, and correction behavior. |

## Maintenance Rule

Add sources and scoped claims here. Put matching procedures, examples, and
reader guidance in Chapter 04. Put classifications in Chapter 11, interface
behavior in Chapter 12, and literal labels/artifacts in Chapter 15. Git history
retains the earlier field tables, independent-research addenda, and proposed
chapter updates.
