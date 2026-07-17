# Pricing Evidence Ledger

> Compact provenance for
> [`../reference/Chapter_08_Pricing.md`](../reference/Chapter_08_Pricing.md).
> This ledger records source-supported pricing claims, integration controls,
> contradictions, and missing evidence. It is not a native Axys/APX price-file
> specification, price-source hierarchy, or valuation methodology.

## Ownership Boundary

- Reader explanations, examples, audit rules, field guidance, and canonical
  Unknowns belong in Chapter 08.
- Security, holdings, transaction, corporate-action, and performance ledgers own
  their domain-specific dependencies.
- IMEX/REP interface detail belongs in the matching interface evidence ledgers.
- Cross-topic public-web provenance belongs in
  [`Public_Web_Research_2026-07-17.md`](Public_Web_Research_2026-07-17.md).

The former narrative research file was reduced after its durable conclusions
were incorporated into Chapter 08. Git history remains the recovery path for
superseded prose, conceptual examples, audit-rule drafts, and chapter outlines.

## Source Register

| ID | Source | Type and scope | Default confidence |
|---|---|---|---:|
| PRC-S01 | [ByAllAccounts Custodial Integrator Axys guide](https://www.byallaccounts.net/Manuals/Custodial_Integrator/axys/CI_User_Guide.pdf) | Third-party Axys price-file import, folders, IMEX, logs, and file-lock behavior. | Medium-High for workflow |
| PRC-S02 | [ByAllAccounts Custodial Integrator Axys release notes](https://www.byallaccounts.net/Manuals/Custodial_Integrator/axys/CI_releasenotes.pdf) | Version-specific calculated, missing/stale, source-width, and precision behavior. | High for cited CI releases |
| PRC-S03 | [WealthTechs AIA manual for APX](https://www.wealthtechs.com/files/AIADocumentation/APX%20Guide%20-%20AIA%20User%20Manual%20For%20APX%20Users.pdf) | Third-party update modes, clean files, price sets, custodian prices, and trumping order. | Medium-High for workflow |
| PRC-S04 | [WealthTechs AIA manual for Axys](https://www.wealthtechs.com/files/AIADocumentation/Axys%20Guide%20-%20AIA%20User%20Manual%20For%20Axys%20Users.pdf) | Adjacent Axys AIA processing context. | Medium |
| PRC-S05 | [AdventGuru Axys articles](https://adventguru.com/tag/axys/) and [IMEX articles](https://adventguru.com/tag/imex/) | Consultant `mergepri`, precedence, IMEX, and direct-file cautions. | Medium |
| PRC-S06 | [SS&C Advent Axys product page](https://www.advent.com/solutions/axys/) | Vendor accounting, performance, reconciliation, and multicurrency capabilities. | High for capabilities; Low for mechanics |
| PRC-S07 | [SS&C Advent APX product page](https://www.advent.com/solutions/advent-portfolio-exchange/) | Vendor accounting, reporting, performance, multicurrency, and multi-asset capabilities. | High for capabilities; Low for mechanics |
| PRC-S08 | [UNAPEN PriceFusion for APX](https://unapen.com/products/pricefusion-pricing-and-reference-data-for-apx) | Third-party APX pricing/reference-data ecosystem context. | Medium for ecosystem only |
| PRC-S09 | [APX Reports Guide](https://cdn.advent.com/cms/pdfs/reports/REP_APX.pdf) | Vendor report guide; insufficient for a complete pricing report catalog. | High for reviewed descriptions |
| PRC-S10 | [Public Web Research Ledger](Public_Web_Research_2026-07-17.md) | `WEB-20260717-002`: official APX 21.1 REST pricing/security-master load capability. | Per claim |

## Axys Price-Workflow Claims

| Claim | Evidence | Confidence | Boundary or chapter impact |
|---|---|---:|---|
| PRC-C001 | Axys CI uses security/type information to generate transaction, position, and price files and imports requested files through Axys Import/Export. | Medium-High; PRC-S01 | CI workflow, not a complete native pricing architecture. |
| PRC-C002 | `$pathpri` is an observed Axys price folder and `*.pri` are observed price files. | Medium-High; PRC-S01 | Complete layout, key, naming, date/source/currency scope, and storage role remain Unknown. |
| PRC-C003 | The observed CI workflow uses `imex32.exe` and `imexPrices.log`; multiple historical days can produce multiple `imexPrices` tabs. | Medium-High; PRC-S01 | Integration executable/UI/log behavior; exact schema and version coverage unverified. |
| PRC-C004 | If an Axys price file is open or in use, CI price import can fail and log the error in `imexPrices.log`. | Medium-High; PRC-S01 | Operational failure mode, not a universal error message specification. |
| PRC-C005 | CI can append prices to prior-business-day files and avoids replacing a same-day security record that already has a price in the documented workflow. | Medium-High; PRC-S01 | Integration update rule, not native Axys precedence. |
| PRC-C006 | CI price preview evidence exposes `Symbol`, `Price`, `Source`, `Price Date`, and `Price As-Of Date`. | Medium-High; PRC-S01 | Preview labels, not official native `.pri` fields. |

## Calculated, Missing, Stale, and Precision Claims

| Claim | Source-supported observation | Confidence | Safety boundary |
|---|---|---:|---|
| PRC-C020 | CI distinguishes IDC, custodian, third-party security, holding, and calculated price concepts. | High for CI releases; PRC-S02 | Do not normalize them into a native Axys source field without proof. |
| PRC-C021 | When IDC and custodian prices are unavailable, CI can calculate output price from position units and market value when available. | High for CI releases; PRC-S02 | Exact formula, scaling, factor, multiplier, and zero-unit handling remain unverified. |
| PRC-C022 | A corrected CI defect had routed a current calculated price to the Missing Price file when other prices were missing or stale; corrected behavior routes it to the Price file. | High for cited release; PRC-S02 | Version-specific integration behavior, not native Axys missing-price logic. |
| PRC-C023 | CI 3.4 corrected calculated bond-price truncation and uses firm-configured decimal precision, defaulting to four in the cited release context. | High for cited release; PRC-S02 | Precision can still truncate small prices to zero; not a native Axys precision specification. |
| PRC-C024 | Later CI evidence records a V3.19.001 bond-price calculation correction. | High for cited CI release; PRC-S02 | Retain CI version when analyzing calculated-price discrepancies. |
| PRC-C025 | CI V3.18.001 added support for a six-character internal price-table source column. | High for cited CI release; PRC-S02 | Does not establish the native Axys price-source field or maximum width. |
| PRC-C026 | CI V3.17.001 added `cashDivCalculateQuantity` for cash-dividend calculated units. | High for cited CI release; PRC-S02 | Transaction/integration option with pricing dependency; not a price-file schema rule. |
| PRC-C027 | Missing and stale pricing states are evidenced in CI, but native Axys stale thresholds, reports, and exception processes are not. | Mixed: High for CI; Unknown native | Chapter audit thresholds must remain configured policy, not vendor defaults. |

## APX Price-Workflow Claims

| Claim | Source-supported observation | Confidence | Safety boundary |
|---|---|---:|---|
| PRC-C040 | APX AIA Price File Update Logic offers `Update Existing & Add New`, `Add New`, and `Replace Entire File`. | Medium-High; PRC-S03 | AIA labels; exact APX key and replace scope remain Unknown. |
| PRC-C041 | `Replace Entire File` deletes existing prices in the AIA-described scope and replaces them with import-file prices. | Medium-High; PRC-S03 | High-risk control; date/source/set/file scope is not established. |
| PRC-C042 | AIA Clean Price File can remove prices for securities held only in filtered accounts. | Medium-High; PRC-S03 | Can create valuation omissions; integration behavior. |
| PRC-C043 | AIA Price Set Logic is used when APX price sets exist and can create custodian-specific price files. | Medium-High; PRC-S03 | Native price-set schema, persistence, and key semantics remain Unknown. |
| PRC-C044 | AIA can merge pricing by Multi Custodian Settings trumping order, with the first custodian highest priority. | Medium-High; PRC-S03 | Exact tie-break and key fields remain Unknown. |
| PRC-C045 | AIA can combine trumping order with a custom filename such as `mmddyy_CDI.pri`. | Medium-High; PRC-S03 | Example filename; do not infer universal APX naming/layout. |
| PRC-C046 | AIA offers an Axys-style three-column set and an APX column set with additional fields. | Medium-High; PRC-S03 | Strong evidence that similarly named `.pri` artifacts need not share layout. |
| PRC-C047 | `SourceId` appears as a price-source label in an AIA/APX context. | Medium-High for screenshot/example | Not proven as a native database, IMEX, REST, or REP field. |
| PRC-C048 | AIA FX File Update Logic also has Update, Add, and Replace modes. | Medium-High; PRC-S03 | FX-file schema and replace scope remain Unknown. |
| PRC-C049 | APX AIA/NBIN cleanup can remove interlisted duplicate position/price rows so the retained listing/currency matches the holding. | Medium-High; PRC-S03 | Cleanup is integration-specific and valuation-sensitive. |
| PRC-C050 | Official APX 21.1 release material says REST APIs can load pricing and security-master data. | High for release capability; PRC-S10 | Endpoint schemas, keys, source/set semantics, entitlements, and equivalence to file routes remain Unknown. |

## Merge, Source, and Interface Claims

| Claim | Evidence synthesis | Confidence | Safe conclusion |
|---|---|---:|---|
| PRC-C060 | AdventGuru describes a `mergepri` command with a destination and multiple sources; the first source is primary and is not overwritten by secondary sources. | Medium; PRC-S05 | Consultant evidence; syntax, versions, and errors require command documentation or tests. |
| PRC-C061 | Axys CI and APX AIA both expose price-source precedence concepts, but use different labels and workflows. | Medium-High | Preserve original source/custodian/set metadata; do not collapse models prematurely. |
| PRC-C062 | Direct access to Axys files is version-sensitive; APX may offer SQL/public-view or REST alternatives. | Medium for access architecture; PRC-S05, PRC-S10 | Exact supported price views/endpoints and fields remain Unknown. |
| PRC-C063 | Reviewed evidence proves price import routes but not native IMEX price object names, field lists, required keys, export support, or null behavior. | Unknown native details | Do not invent IMEX object or field names. |
| PRC-C064 | Axys/APX reporting capabilities are established, but exact standard price, missing-price, and stale-price report names and REP fields are not. | High for capability; Unknown for catalog | Report outputs may show calculated valuation rather than stored price records. |

## Valuation-Dependency Claims

| Claim | Evidence synthesis | Confidence | Boundary |
|---|---|---:|---|
| PRC-C080 | Pricing is a dependency of holdings valuation, performance, reconciliation, and exception review. | High conceptually | Native calculation order and regeneration behavior remain Unknown. |
| PRC-C081 | A complete audit key may need security identity, price date/as-of date, value, source/custodian, currency, price set, factor, multiplier, and lineage. | High as discovery guidance | Native availability and field names are unverified. |
| PRC-C082 | Calculated fixed-income price may depend on units and market value, while final valuation can also depend on factor, multiplier, and accrued interest. | High conceptually; partial CI evidence | Exact native formulas and clean/dirty price treatment remain Unknown. |
| PRC-C083 | Price-source precedence, update/replace mode, filtered-account cleanup, and precision can materially change valuation without a transaction change. | High as audit conclusion | Preserve configuration and import-log provenance. |
| PRC-C084 | Transaction price and daily/historical price-file price are distinct evidence concepts whose native relationship is unverified. | High caution | Reconcile them; do not overwrite one with the other. |
| PRC-C085 | Corporate-action split adjustment and historical-price restatement behavior are not defined by the pricing sources. | Unknown | Use Chapter 09 evidence and controlled before/after scenarios. |
| PRC-C086 | Vendor product pages establish multicurrency capability but not price-currency or FX-rate storage and valuation mechanics. | High for capability; Unknown for schema | Require local/base amounts, currency, FX source/date, and report settings. |

## Contradictions and Interpretation Risks

| ID | Tension | Resolution |
|---|---|---|
| PRC-X001 | Both systems use `.pri`-named artifacts in reviewed integrations, but Axys and APX layouts and roles may differ. | Treat extension as workflow evidence only; preserve system, route, and version. |
| PRC-X002 | A calculated price can be current and usable while vendor/custodian/holding prices are missing or stale. | Preserve source and calculation status; do not reduce status to present/missing. |
| PRC-X003 | APX update/add/replace labels imply mutation but do not reveal native price keys or scope. | Treat Replace Entire File as high risk until a controlled test proves scope. |
| PRC-X004 | Price set, price file, custodian, `SourceId`, and REST source semantics may represent different layers. | Do not map them one-to-one without paired vendor/site evidence. |
| PRC-X005 | Price files/logs show import behavior, while reports may expose derived valuation. | Keep stored/imported price evidence separate from report-calculated market value. |
| PRC-X006 | Fixed-income market value may exclude accrued interest even when performance uses dirty value. | Preserve price, factor, market value, and accrued interest separately. |
| PRC-X007 | AIA interlisted cleanup and filtered-account cleanup can remove prices intentionally. | Record cleanup settings and reconcile omissions against actual held listings. |
| PRC-X008 | Product and ecosystem sources prove capabilities or demand, not native schemas. | Use them only for capability/ecosystem claims. |

## Evidence Required to Resolve Canonical Unknowns

Chapter 08 owns the complete Unknowns table. This section records the missing
evidence needed to resolve it.

| Need | Evidence that would resolve or materially narrow it |
|---|---|
| PRC-U001 Native layouts and keys | Sanitized Axys/APX price files plus versioned IMEX/import specifications defining fields, key, requiredness, and scope. |
| PRC-U002 APX REST route | OpenAPI documentation, authentication/entitlement rules, schemas, errors, and paired REST/file/site outputs. |
| PRC-U003 Price sources and sets | APX price-set and Axys source configuration, extracts, precedence rules, and reproducible same-security/date tests. |
| PRC-U004 Reports and exceptions | Versioned REP/RDL source and output for price lists, missing/stale prices, and valuation reports. |
| PRC-U005 Fixed income | One bond/MBS package with price, units, factor, multiplier, accrued interest, clean/dirty value, source, and report output. |
| PRC-U006 FX and currency | Price/FX files or APIs with local/base currencies, rate source/date, valuation output, and report-currency settings. |
| PRC-U007 Corporate actions | Split and other action scenarios with prices, holdings, transactions, and performance before and after processing. |
| PRC-U008 Historical regeneration | Controlled historical price correction proving which holdings, gains, and stored/reported performance outputs regenerate. |
| PRC-U009 Version differences | The same import, replace, merge, precision, and exception cases across supported Axys/APX and integration versions. |
| PRC-U010 Operational lineage | Successful/failed `imexPrices.log`, AIA/APX logs, source files, configuration, backups, and resulting records for one load. |

Highest-value next acquisition: a sanitized multi-source pricing package for a
single valuation date containing source files, security master, price/FX
configuration, successful and failed import logs, stored/exported prices,
holdings valuation, and performance output.

## Maintenance Rule

Add a claim only for new provenance, a narrowed boundary, or a contradiction.
Update Chapter 08 when reader guidance or an Unknown changes. Do not append
conceptual examples, audit-rule drafts, or another narrative research pass.
