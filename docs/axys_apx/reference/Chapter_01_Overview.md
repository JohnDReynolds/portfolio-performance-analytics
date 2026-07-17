# Chapter 01 — Overview

This chapter explains how to use the Axys/APX reference, summarizes the
system-level distinction, and identifies the cross-cutting evidence gaps that
still constrain implementation. It is an entry point, not a condensed copy of
the subject chapters.

## 1. Scope

The reference documents supported SS&C Advent Axys and Advent Portfolio
Exchange (APX) behavior relevant to source-data extraction, reconciliation,
performance comparison, and implementation decisions.

It is not a complete vendor specification, an official source-system schema,
or evidence that ppar reproduces native Axys/APX methodology. Product behavior
that is not supported by the maintained evidence remains **Unknown**.

Read [`../README.md`](../README.md) for folder roles and
[`README.md`](README.md) for the complete chapter index.

## 2. Confidence and Evidence Boundary

| Label | Meaning |
|---|---|
| Verified | Directly supported by identified source material or a recorded observation. |
| High Confidence | Strongly supported, but narrower than a complete vendor specification. |
| Medium Confidence | Plausible and partially supported; local validation is required. |
| Unknown | Not established by the available evidence and not safe to assume. |

Confidence applies to the scope stated with a claim. A fact verified for a
specific connector, report, release, or client workflow is not automatically a
universal product rule.

The evidence hierarchy and editorial rules are governed by
[`../axys_apx_reference_blueprint.md`](../axys_apx_reference_blueprint.md).
Reader conclusions belong in the subject chapters; sources, locators, dated
research, and contradictions belong in [`../evidence/`](../evidence/).

## 3. Cross-Chapter Data Flow

The repository follows this conceptual lifecycle:

```text
Portfolio / account
    -> Security master
    -> Transactions
    -> Holdings and cash
    -> Prices and corporate actions
    -> Performance and classifications
    -> Reports
```

IMEX and REP are exchange paths across that lifecycle. They are not additional
business domains. Chapter 15 indexes literal artifacts; Chapter 16 defines
conceptual vocabulary.

## 4. Axys and APX at a Glance

| Area | Axys | APX | Boundary |
|---|---|---|---|
| Product role | Portfolio accounting, reporting, performance, and operations platform. | Integrated portfolio and client management platform and centralized book of record. | Capability statements do not establish internal schemas. |
| Architecture evidence | Proprietary-database positioning plus version-sensitive file, utility, IMEX, and REP evidence. | SQL-oriented platform evidence plus reports, IMEX/APXIX, public-view/function, and API capability evidence. | Exact native storage and access contracts remain incomplete. |
| Integration evidence | `imex32.exe`, REP/RepLang, named operational files, and connector workflows. | IMEX/APXIX, blotters, reporting/SSRS, SQL-related access paths, and REST capability from official release evidence. | Connector and practitioner behavior is installation- and version-specific. |
| Reporting | REP/RepLang, Report Writer Pro, and named historical or workflow reports. | Standard and custom reporting, SSRS-era evidence, and named report families. | Report names and labels do not prove native fields or tables. |
| Main implementation risk | Treating observed files or codes as a complete native specification. | Inferring schemas, endpoints, or entitlements from high-level access capability. | Validate version, environment, interface, and source context. |

For details, use [Chapter 02](Chapter_02_Axys_Architecture.md) for Axys and
[Chapter 03](Chapter_03_APX_Architecture.md) for APX.

## 5. Canonical Chapter Ownership

Information should be maintained once and linked elsewhere.

| Information | Canonical chapter |
|---|---|
| Axys architecture, utilities, and system-level artifacts | [Chapter 02](Chapter_02_Axys_Architecture.md) |
| APX architecture, access paths, and versioned API capability | [Chapter 03](Chapter_03_APX_Architecture.md) |
| Security identity and master data | [Chapter 04](Chapter_04_Security_Master.md) |
| Transaction codes, context gates, and flow classification | [Chapter 05](Chapter_05_Transactions.md) |
| Holdings and positions | [Chapter 06](Chapter_06_Holdings.md) |
| Cash and cash-like activity | [Chapter 07](Chapter_07_Cash.md) |
| Prices and valuation inputs | [Chapter 08](Chapter_08_Pricing.md) |
| Corporate actions and split evidence | [Chapter 09](Chapter_09_Corporate_Actions.md) |
| Performance methodology boundaries and artifacts | [Chapter 10](Chapter_10_Performance.md) |
| Classifications and grouping dimensions | [Chapter 11](Chapter_11_Classifications.md) |
| IMEX/APXIX utilities, paths, logs, and discovery | [Chapter 12](Chapter_12_Imex.md) |
| REP, RepLang, and report-based extraction | [Chapter 13](Chapter_13_Rep.md) |
| Report names, labels, and report families | [Chapter 14](Chapter_14_Reports.md) |
| Literal fields, codes, filenames, utilities, labels, and aliases | [Chapter 15](Chapter_15_Data_Dictionary.md) |
| Conceptual definitions and ambiguity notes | [Chapter 16](Chapter_16_Glossary.md) |
| Multi-currency evidence and audit-adapter boundary | [Chapter 17](Chapter_17_Multi_Currency.md) |

## 6. Safe Cross-Cutting Rules

- Do not treat report labels as native fields.
- Do not infer IMEX objects, REP fields, APX tables, public views, stored
  functions, or API endpoints from capability statements or report names.
- Preserve security symbol, type, source system, and source context. Symbol
  alone is not a safe universal identity key.
- Do not equate security type with asset class, sector, country, region, or
  another classification.
- Do not classify external flows from transaction code alone when source,
  destination, security, or portfolio context can change the meaning.
- Do not assume performance values are stored or recalculated without explicit
  report-, export-, or environment-specific evidence.
- Record the product, version, installation, extraction path, report
  parameters, and evidence confidence with source-data.
- Treat AIA, Custodial Integrator, Data Broker, conversion, and consultant
  behavior as workflow evidence unless native product behavior is separately
  established.

<a id="axys_apx-blockers"></a>

## 7. Current Cross-Cutting Blockers

The reference supports configurable source-data snapshots and conservative
Modified Dietz comparison logic. The following gaps prevent broader claims or
turnkey native automation.

| ID | Blocker | What it constrains | Canonical owner | Evidence needed |
|---|---|---|---|---|
| AXAPX-B01 | No verified performance extract dictionary. | Turnkey native performance exports and official export recipes. | Chapters 10, 12, and 15 | Versioned IMEX catalog, performance exports, report output, or vendor manual. |
| AXAPX-B02 | Stored-versus-recalculated performance is Unknown. | Claims about native recalculation and historical-change behavior. | Chapter 10 | Controlled reruns, stored/exported values, report source, or vendor documentation. |
| AXAPX-B03 | Security-performance footing is Unknown. | Automated security-to-portfolio rollups and contribution reconciliation. | Chapters 10 and 11 | Weights, returns, contribution, cash rows, and portfolio footing from the same output. |
| AXAPX-B04 | Transaction-code coverage is incomplete and context-dependent. | Universal classification, external-flow inference, fixed-income automation, and reversals. | Chapter 05 | Official or local code matrix plus source/destination and controlled examples. |
| AXAPX-B05 | IMEX/APXIX object and field lists are not authoritative. | A generic connector prescribing exact objects and columns. | Chapter 12 | Versioned manuals, live catalog, and sanitized exports. |
| AXAPX-B06 | REP, SSRS, and report definitions are unavailable. | Standard report-extraction recipes and precise report-to-source reconciliation. | Chapters 13 and 14 | `.REP` files, RDL definitions, report parameters, and sample outputs. |
| AXAPX-B07 | APX SQL/public-view/API contracts are under-evidenced. | Direct connectors requiring exact schemas, endpoints, authentication, and entitlements. | Chapter 03 | Schema documentation, function signatures, API material, and sanitized metadata. |
| AXAPX-B08 | Multi-currency, fixed-income, and corporate-action mechanics remain incomplete. | FX attribution, accrual/principal automation, and native event treatment. | Chapters 07-10 and 17 | Currency, fixed-income, and corporate-action reports, exports, and controlled examples. |

These blockers do not prevent a configurable comparison workflow that uses
explicit extract contracts and transaction semantics. They matter when a
conclusion is presented as native Axys/APX extraction or methodology.

## 8. Version and Environment Boundary

Public and practitioner evidence spans different product eras and workflows.
The maintained chapters currently distinguish, among other examples:

- Axys 3.x-era IMEX and file-conversion evidence;
- connector-specific Axys and APX minimum versions;
- the APX 3.0 reporting-framework era;
- official APX REST capability beginning with release 21.1 and later
  extensions; and
- current product-positioning pages that do not document legacy internals.

These observations establish dated capability, not universal compatibility.
Use the version tables in Chapters 02, 03, 12, 13, and 14 before applying an
artifact or workflow to an installed environment.

## 9. Using the Reference Safely

1. Start with the subject chapter that owns the question.
2. Check its confidence and version/environment boundary.
3. Follow its evidence link only when provenance or unresolved detail matters.
4. Use Chapter 15 to locate literal names and Chapter 16 to resolve terms.
5. Use `../contracts/` only for ppar implementation behavior; contracts are not
   vendor specifications.
6. Preserve Unknowns until stronger evidence changes the owning chapter.

The highest-value missing materials remain sanitized exports, live IMEX/APXIX
catalogs, report definitions and outputs, versioned vendor manuals, APX schema
or API contracts, and controlled before/after examples.

## 10. References

- [`../README.md`](../README.md) — repository map and file roles.
- [`../axys_apx_reference_blueprint.md`](../axys_apx_reference_blueprint.md) —
  editorial and evidence rules.
- [`README.md`](README.md) — complete chapter index.
- [`../evidence/README.md`](../evidence/README.md) — evidence index and
  maintenance boundary.
- [`../contracts/README.md`](../contracts/README.md) — implementation contract
  index.
