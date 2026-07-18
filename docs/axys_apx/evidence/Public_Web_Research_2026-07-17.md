# Public Web Research Ledger — 2026-07-17

**Purpose:** Preserve compact provenance for the public-web research incorporated
into the reader-facing Axys/APX reference chapters on 2026-07-17.

**Retrieval/review date:** 2026-07-17

**Additional review date:** 2026-07-18

This file is an evidence ledger, not a parallel reference manual. The matching
files under `../reference/` contain the maintained conclusions and implementation
guidance. Each claim below records only the source, bounded conclusion, confidence,
and evidence still needed.

## Evidence classes

| Class | Treatment |
|---|---|
| Official current vendor material | Strong product/release capability evidence; not automatically a field or formula specification. |
| Official historical vendor material | Valid for the stated historical release/era; do not project forward without confirmation. |
| Operational integration guidance | Strong evidence for the cited workflow; not necessarily universal native behavior. |
| Practitioner/report customization guidance | Useful artifact and execution evidence; version and site scope must remain explicit. |
| Legal/administrative record | Secondary evidence of the quoted historical product behavior; not a current vendor contract. |

## Claim register

| Claim ID | Public source | System / version scope | Bounded supported claim | Confidence and limitation |
|---|---|---|---|---|
| WEB-20260717-001 | [Advent investor presentation, March 2008](https://media.corporate-ir.net/media_files/irol/96/96626/presentations/Advent_IR_Presentation_Mar_13_08.pdf) | Axys/APX, 2008 positioning | Advent characterized Axys as a flat-file accounting platform and APX as a `.NET` relational platform. | Verified historical positioning; not a current physical-storage specification. |
| WEB-20260717-002 | [Advent Investment Suite 2021 release](https://www.advent.com/news-and-insights/blog/advent-investment-suite-release-2021-efficiency-trading-compliance/) | APX 21.1 | REST APIs cover market, reference, and accounting data, including pricing/security-master loads and holdings, performance, and gain/loss extraction. | Verified release capability; endpoint schemas, authentication, and entitlements not public in reviewed material. |
| WEB-20260717-003 | [SS&C Advent 2H2021 updates](https://investor.ssctech.com/news-and-events/news-details/2021/SSC-Announces-2H2021-SSC-Advent-Product-Updates-11-10-2021/default.aspx) | APX, late 2021 | Entity and accounting REST APIs were added; Performance Update runtime was improved. | Verified release statement; site adoption/configuration Unknown. |
| WEB-20260717-004 | [APX 22.1 product update](https://www.advent.com/news-and-insights/blog/an-ever-more-powerful-and-trusted-solution-for-investment-managers/) | APX 22.1 | More than 60 APIs and expanded historical-change, attribution/analytics, cost, and session-audit access were announced. | Verified release capability; exact contracts and fields Unknown. |
| WEB-20260717-005 | [SS&C Advent 1H2023 updates](https://www.advent.com/news-and-insights/press-releases/ssc-announces-1h2023-ssc-advent-product-updates/) | APX 23.1 | Report data was expanded and fee/model-fee fields were added. | Verified release statement; exact report/API mappings Unknown. |
| WEB-20260717-006 | [ByAllAccounts APX Custodial Integrator guide](https://www.byallaccounts.net/Manuals/Custodial_Integrator/apx/CI_User_Guide.pdf) | APX CI workflow | `apxix.exe`/`ApxIx` denotes APX Import/Export; security/type exports, translation, blotter, position, and lot workflows are documented. | High Confidence for this integration; native object dictionary still absent. |
| WEB-20260717-007 | [WealthTechs AIA guide for APX](https://wealthtechs.com/AIADocumentation/APX%20Guide%20-%20AIA%20User%20Manual%20For%20APX%20Users.pdf) | APX AIA workflow | `APXIX.exe`, `imexhist.log`, ordered transformation, cancellation Trade Blotter creation using uppercased instructions, current SQL holdings, and historical report-calculated holdings are documented. | High Confidence for this integration; posted-export representation, native universality, and schemas Unknown. |
| WEB-20260717-008 | [CSSI PRF/PBF name guidance](https://cssisolutions.com/downloads/solving-the-cli-prf-pbf-name-problem) | Axys operational workflow | `.PRF`/`.PBF` are performance-history state; names are copied from `.CLI`/`.GRP` on first creation and are not automatically resynchronized after later renames. | High Confidence operational evidence; exact layout/version coverage Unknown. |
| WEB-20260717-009 | [Historical PERHSUM guidance](https://static1.1.sqspcdn.com/static/f/425065/4721492/1257913571447/Modifying-PERSHUM-Report.pdf) | Axys 3.6 | `PERHSUM.REP` is Performance History for Selected Time Periods; `.REP` and `.RPW` are distinct report types and `.RPW` remains RepLang-based. | Verified historical evidence; not a current support statement. |
| WEB-20260717-010 | [CSSI REP32 hyperlink guidance](https://cssisolutions.com/downloads/how-to-add-hyperlinks-to-reports) | Dated Axys workflow | A working `Rep32.exe -m ... -p ... "-b ..."` command pattern and switches `-J`, `-x`, `-su`, and `-z` are demonstrated. | Verified working example; complete syntax and version coverage Unknown. |
| WEB-20260717-011 | [CSSI group-of-composites guidance](https://cssisolutions.com/downloads/how-to-create-a-group-of-composites) | Dated Axys workflow | `.CPG` composite artifacts, `_cpgtree.rep`, `_cpglist.rep`, and composite member entry/exit dates are documented. | High Confidence operational evidence; exact layout/current support Unknown. |
| WEB-20260717-012 | [CSSI missing-cost guidance](https://cssisolutions.com/downloads/how-to-identify-missing-cost-information) | Axys report workflow | A bounded set of transaction/cost report fields is exposed; `li`, `ti`, and `si` are deliver-in cases, and absent original cost/date may fall back to trade-date market value in standard reports. | High Confidence for the cited workflow; not an official native schema or universal rule. |
| WEB-20260717-013 | [ByAllAccounts APX release notes](https://www.byallaccounts.net/Manuals/Custodial_Integrator/apx/CI_releasenotes.pdf) | APX CI v1-v4 behavior | Mark-to-Market was required only for foreign-currency transactions in APX v1-v3 and requires explicit `y`/`n` in APX v4. | High Confidence for this integration; does not define the accounting calculation. |
| WEB-20260717-014 | [Official Axys product page](https://www.advent.com/solutions/axys/) | Current public Axys capability | Axys supports TWR/IRR before/after fees, performance by multiple classifications, blended/synthetic benchmarks, composites with entry/exit dates, report-currency restatement, withholding tax, return bifurcation, and trading/income/risk currency concepts. | Verified capability statements; fields, formulas, and site configuration Unknown. |
| WEB-20260717-015 | [APX and GIPS transparency](https://www.advent.com/news-and-insights/blog/achieving-gips-compliance-transparency-takes-center-stage/) | APX public capability | APX supports time-weighted and money-weighted return calculations. | Verified capability; exact report formulas/settings Unknown. |
| WEB-20260717-016 | [SEC record quoting Axys Help](https://www.sec.gov/litigation/apdocuments/3-14194-event-38.pdf) | Historical Axys behavior cited in 2008 record | Performance History is described as TWR and DCF as IRR; the cited historical approximation divides at valuation boundaries, calculates interval IRRs, and links results. | Verified historical secondary/legal evidence; not a current universal contract. |
| WEB-20260717-017 | [SEC post-hearing reply brief](https://www.sec.gov/file/division-enforcements-post-hearing-reply-brief-13) | Historical Axys report behavior | The cited Average Capital Base report used Modified Dietz. | Verified for the cited historical report; not all Axys reports. |
| WEB-20260717-018 | [Advent performance-attribution paper](https://cdn.advent.com/cms/pdfs/papers/WP_PA.pdf) | General Advent methodology background | The paper explains relative/absolute attribution, BHB formulas, and an option to combine interaction with selection. | Verified methodology background; does not prove a configured APX implementation. |
| WEB-20260717-019 | [Advent Portfolio Exchange Reports Guide](https://cdn.advent.com/cms/pdfs/reports/REP_APX.pdf) | Guide-covered APX reports | The guide identifies 29 SSRS-based reports, report purposes/labels, Portfolio Appraisal and Equity Overview behavior, and Risk Statistics metrics. | Verified for the reviewed guide; installed/current inventory, RDLs, datasets, and formulas Unknown. |
| WEB-20260717-020 | [CSSI equity-assets and Cash Holdings guidance](https://cssisolutions.com/downloads/creating-an-equity-assets-by-type-report-and-a-cash-hold) | Dated Axys Report Writer workflow | Report-visible fields include asset-class/security-type concepts, and a Cash Holdings report is derived through classification logic. | High Confidence for the example; example asset-class code is not universal. |
| WEB-20260717-021 | [SS&C Advent Corporate Actions for APX brief](https://www.advent.com/resources/all-resources/brief-advent-corporate-actions-for-apx/) | APX ACA workflow | APX sends holdings to ACA; actions are cross-referenced/reviewed; APX Reorg Utility runs; generated transactions post to APX Trade Blotter. | Verified workflow capability; exact fields, statuses, and postings Unknown. |
| WEB-20260718-001 | [WealthTechs AIA guide for APX](https://wealthtechs.com/AIADocumentation/APX%20Guide%20-%20AIA%20User%20Manual%20For%20APX%20Users.pdf) | APX AIA workflow | The guide describes APX as case-sensitive for vehicle symbols, account codes, and the management-fee symbol, while separately stating that AIA Transaction Translation rules are not case-sensitive. | High Confidence for the cited workflow; not a complete native APX field-by-field casing contract. |
| WEB-20260718-002 | [ByAllAccounts Axys CI guide](https://www.byallaccounts.net/Manuals/Custodial_Integrator/axys/CI_User_Guide.pdf) and [APX CI guide](https://www.byallaccounts.net/Manuals/Custodial_Integrator/apx/CI_User_Guide.pdf) | Axys/APX CI workflows | Security and classification examples use mixed case, including lowercase security types in mapping examples and uppercase types in translation-file examples; the guides do not establish a global lowercase convention. | High Confidence for observed examples; equivalence of values differing only by case is not established. |

## Topic routing

| Evidence file | Claim IDs |
|---|---|
| `Research_02_Axys_Architecture.md` | 001, 008-011, 014 |
| `Research_03_APX_Architecture.md` | 001-007, 019, 021 |
| `Research_04_Security_Master.md` | 002, 006, 007, 20260718-001-002 |
| `Research_05_Transactions.md` | 007, 012, 013, 20260718-001-002 |
| `Research_06_Holdings.md` | 007, 019 |
| `Research_07_Cash.md` | 014, 020 |
| `Research_08_Pricing.md` | 002, 007 |
| `Research_09_Corporate_Actions.md` | 021 |
| `Research_10_Performance.md` | 002-004, 008-009, 014-019 |
| `Research_11_Classifications.md` | 011, 014, 019 |
| `Research_12_IMEX.md` | 002-007, 013, 20260718-001 |
| `Research_13_REP.md` | 002-005, 009-010, 019 |
| `Research_14_Reports.md` | 009, 019 |
| `Research_17_Multi_Currency.md` | 013-014 |

## Public-evidence ceiling

The public phase did not recover the following implementation contracts:

- complete current Axys native file layouts;
- current official Axys/APX IMEX object and field dictionaries;
- APX OpenAPI/Postman contracts, authentication, or entitlement matrices;
- APX SQL tables, public-view schemas, stored functions, or RDL datasets;
- current site-specific TWR, attribution, FX, fee, cash-flow, and rounding settings;
- proof that IMEX, REP, REST, SQL, and SSRS routes are semantically equivalent; or
- installed-system configuration and custom-report inventories.

These items remain gated by customer-portal, installed-system, or sanitized
client evidence.
