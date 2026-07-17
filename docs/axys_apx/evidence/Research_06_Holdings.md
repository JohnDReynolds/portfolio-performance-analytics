# Holdings Evidence Ledger

> Compact provenance for
> [`../reference/Chapter_06_Holdings.md`](../reference/Chapter_06_Holdings.md).
> This ledger records source-supported holdings claims, interpretation
> boundaries, contradictions, and missing evidence. It is not a native Axys/APX
> holdings schema or report specification.

## Ownership Boundary

- Reader explanations, examples, field guidance, and canonical Unknowns belong
  in Chapter 06.
- Transaction semantics belong in
  [`Research_05_Transactions.md`](Research_05_Transactions.md) and the
  transaction contract.
- Cross-topic public-web provenance belongs in
  [`Public_Web_Research_2026-07-17.md`](Public_Web_Research_2026-07-17.md).
- This file owns granular holdings claims and the evidence needed to resolve
  their boundaries.

The former narrative research file was reduced after its durable conclusions
were incorporated into Chapter 06. Git history remains the recovery path for
superseded research prose, examples, and drafting metadata.

## Source Register

| ID | Source | Type and scope | Default confidence |
|---|---|---|---:|
| HLD-S01 | SS&C Advent Axys product material | Vendor capability context; no native holdings schema. | High for capabilities; Low for mechanics |
| HLD-S02 | SS&C Advent APX product material and dated product updates | Vendor capability and release context. | High for cited capabilities |
| HLD-S03 | WealthTechs AIA manual for Axys, version 3.1.0 | Third-party POS reconciliation, current/historical holdings extraction, and `CDIhold.rep`. | Medium-High for workflow |
| HLD-S04 | [WealthTechs AIA manual for APX](https://wealthtechs.com/AIADocumentation/APX%20Guide%20-%20AIA%20User%20Manual%20For%20APX%20Users.pdf), version 3.1.0 | Third-party APX SQL/current holdings, historical report extraction, and duplicate cleanup. | Medium-High for workflow |
| HLD-S05 | ByAllAccounts Custodial Integrator APX User Guide | Third-party `apxix.exe`, position/lot blotters, prices, logs, and exclusions. | Medium-High for workflow |
| HLD-S06 | CSSI, *Accessing Cli PortCodes Within Groups* | Consultant Axys Portfolio Appraisal, group, Management Mode, and report-field example. | Medium-High for example |
| HLD-S07 | CSSI, *How To Create an Assets Under Management By Sector Report* | Consultant REP path, file, variable, and copy-before-edit guidance. | Medium for example |
| HLD-S08 | Morningstar, *Converting Your Advent Axys Database into Morningstar Office*, version 2.0 | Conversion files, reconciliation, and principal-paydown limitations. | Medium |
| HLD-S09 | [Advent Portfolio Exchange Reports Guide](https://cdn.advent.com/cms/pdfs/reports/REP_APX.pdf) | Vendor guide for Portfolio Appraisal and report-visible holdings concepts. | High for reviewed report descriptions |
| HLD-S10 | [Public Web Research Ledger](Public_Web_Research_2026-07-17.md) | `WEB-20260717-007` and `019`: current-SQL/historical-report boundary and APX report guide. | Per claim |

## Report and Holdings Claims

| Claim | Evidence | Confidence | Boundary or chapter impact |
|---|---|---:|---|
| HLD-C001 | Axys has a Portfolio Appraisal that can serve as a point-in-time holdings/assets report. | Medium-High; HLD-S06 and cited compliance use | The report is evidence of holdings presentation, not native storage. |
| HLD-C002 | Axys Report Writer exposes `Portfolio Code`, `Management Mode`, and selectable columns for Portfolio Appraisal. | Medium-High; HLD-S06 | Verified for the cited example; defaults and version coverage are unverified. |
| HLD-C003 | The cited Axys sample displays `Quantity`, `Security`, `Price`, `Market Value`, `Pct Assets`, `Yield`, and `Portfolio Code`. | Medium-High; HLD-S06 | Example columns, not a complete standard field list. |
| HLD-C004 | An Axys consolidated group produces one appraisal; an unconsolidated group normally produces appraisals by member. | Medium-High; HLD-S06 | APX group behavior is not established. |
| HLD-C005 | In the cited Axys setup, Management Mode plus `Portfolio Code` produces one combined unconsolidated-group appraisal with owner portfolio beside each holding. | Medium-High; HLD-S06 | Customized behavior, not a default guarantee. |
| HLD-C006 | Axys example REP artifacts include `\axys3\rep`, `aman.rep`, a copied `_aumsect.rep`, Replang, `$askport`, `$:fileo`, `$:tfile`, and `$firmg`. | Medium; HLD-S06-S07 | Example/version-specific; exact variable semantics require REP documentation. |
| HLD-C007 | The reviewed APX Reports Guide describes Portfolio Appraisal as holdings by tax lot or position with value, cost, income, gain/loss, allocation, and related presentation concepts. | High for guide; HLD-S09-S10 | Installed columns, datasets, formulas, and lot mechanics remain site-specific. |
| HLD-C008 | Fixed-income appraisal output can present Market Value separately from Accrued Interest. | High; public SS&C/APX report samples | Strongly implies clean market value plus separate accrued interest; a stored Total Value field is unverified. |
| HLD-C009 | For performance reconstruction, total fixed-income position value may need Market Value plus Accrued Interest when the source report uses that presentation. | High as conservative interpretation | Do not assume a holdings market-value extract already includes accrued interest. |

## Extraction, Reconciliation, and Artifact Claims

| Claim | Source-supported observation | Confidence | Safety boundary |
|---|---|---:|---|
| HLD-C020 | AIA reads current-date Axys files directly but uses `CDIhold.rep` to calculate historical holdings. | Medium-High; HLD-S03 | AIA behavior, not a universal Axys extraction model. |
| HLD-C021 | AIA reads current APX holdings from SQL but uses `CDIhold.rep` for historical holdings. | Medium-High; HLD-S04, HLD-S10 | Does not identify APX tables, views, or stored-versus-calculated rules. |
| HLD-C022 | The AIA historical workflow maps `$pathCDI` and `Holdings Extract Folder (h.CDI)` to a shared output path, commonly with a `cdirecon` group. | Medium-High; HLD-S03-S04 | Workflow-specific configuration. |
| HLD-C023 | AIA custodian master-position data contains current holdings by account with market value and quantity and is used to create POS files for comparison with Advent-calculated positions. | Medium-High; HLD-S03-S04 | Reconciliation source, not native holdings storage. |
| HLD-C024 | NBIN/interlisted cleanup can remove duplicate holdings and prices from `.pos` and `.pri` source files after reading holdings for the relevant date. | Medium-High; HLD-S03-S04 | Integration-specific and potentially valuation-sensitive. |
| HLD-C025 | Custodial Integrator APX uses `apxix.exe`, configured output/log folders, and named Trade, Position, and optional Lot blotters. | Medium-High; HLD-S05 | Full APX import/export object model remains unverified. |
| HLD-C026 | CI can import prior-business-day positions, optionally position lots, and exclude stale or failed accounts. | Medium-High; HLD-S05 | Lots are optional and not imported by default in the documented workflow. |
| HLD-C027 | CI maintains APX security/type information for generating positions, prices, and transactions and exposes an `APX Portfolio Code` delivery field. | Medium-High; HLD-S05 | Integration field names, not native database schema. |
| HLD-C028 | Axys CI writes position CSV to `\CI\exported\ptopost.trn`; Position Post can create replacement `.pos` files for configured portfolios. | Medium-High; HLD-S05 and deep IMEX provenance | Integration behavior; not proof of a canonical holdings store. |
| HLD-C029 | Lot-enabled CI output can use `imexPositionLots.log` rather than `imexPositions.log`; cash, money-market, or no-lot positions may fall back to position-level output. | Medium-High; deep IMEX provenance | Exact native lot/position layouts remain Unknown. |
| HLD-C030 | Morningstar identifies `.cli`, `sec.inf`, `split.inf`, `.pri`, and `type.inf` as Axys conversion inputs. | Medium; HLD-S08 | These dependencies do not establish a separate persistent holdings file. |
| HLD-C031 | The documented APX CI configuration uses executable, output, and log folders and warns that the output-folder name should not contain spaces. | Medium-High; HLD-S05 | Integration requirement; not a native APX holdings rule. |
| HLD-C032 | APX CI displays Import/Export logs after execution and requires named Position/Lot blotters to exist before import. | Medium-High; HLD-S05 | Operational workflow evidence, not complete status/error semantics. |

## Accounting and Interpretation Claims

| Claim | Evidence synthesis | Confidence | Safe conclusion |
|---|---|---:|---|
| HLD-C040 | Holdings depend conceptually on portfolio identity, transactions, security master, prices, splits, and report/as-of settings. | High conceptually | Native materialization and recalculation timing remain Unknown. |
| HLD-C041 | Principal paydown conversion can differ because Axys adjusts original principal while the target conversion uses factors; some paydown rows have zero share quantity. | Medium; HLD-S08 | Quantity alone is insufficient; preserve principal/factor, cash, and performance context. |
| HLD-C042 | Such paydown conversion differences can make converted holdings disagree with Axys performance output. | Medium; HLD-S08 | A conversion limitation, not proof of an Axys defect. |
| HLD-C043 | Public evidence supports `ss`/`cs` code meanings but does not prove native short quantity, market-value, proceeds, or holdings signs. | Medium-High for code; Low for mechanics | Synthetic negative quantity/value must be disclosed; production rules require site extracts. |
| HLD-C044 | APX CI does not overwrite an existing same-business-day price in the documented position workflow. | Medium-High; HLD-S05 | Integration behavior with holdings-valuation implications. |

## Contradictions and Interpretation Risks

| ID | Tension | Resolution |
|---|---|---|
| HLD-X001 | Portfolio Appraisal exposes holdings but does not reveal whether values are stored, report-calculated, or mixed. | Treat report output and native storage as separate evidence questions. |
| HLD-X002 | Current data can be read directly in AIA while historical holdings require report calculation. | Preserve date and extraction route; do not assume one schema covers both. |
| HLD-X003 | APX Portfolio Appraisal supports lot/position presentation while CI has separate Position and Lot blotters. | Do not infer import storage or lot schema from report presentation. |
| HLD-X004 | Axys report examples expose useful columns and variables but are customizable. | Qualify every field as example/report-visible unless an official default is supplied. |
| HLD-X005 | Market Value and Accrued Interest are separate in fixed-income report output, but storage and dirty-value calculation are unverified. | Preserve both fields and document any derived total. |
| HLD-X006 | `.pos`, `ptopost.trn`, SQL, and report extracts are different workflow artifacts. | Do not label any one as the canonical native holdings store without vendor or production proof. |

## Evidence Required to Resolve Canonical Unknowns

Chapter 06 owns the complete Unknowns table. This section records the missing
evidence rather than repeating reader guidance.

| Need | Evidence that would resolve or materially narrow it |
|---|---|
| HLD-U001 Native objects and storage | Versioned Axys holdings IMEX objects/layouts and APX holdings tables, public views, functions, or import/export definitions. |
| HLD-U002 Stored versus calculated behavior | Controlled current/historical runs with source-data, extraction route, report settings, and repeatable outputs. |
| HLD-U003 Date semantics | Report/import documentation and samples proving trade-date, settlement-date, accounting-date, and as-of behavior. |
| HLD-U004 Portfolio Appraisal defaults | Unmodified, versioned Axys/APX report definitions and outputs, including group and lot/position settings. |
| HLD-U005 Lot support | Native lot-level imports/exports and reports with acquisition date, quantity, cost, tax fields, and lineage. |
| HLD-U006 Special holdings | Native cash, short, accrued-income, unsettled, FX, and multicurrency position examples. |
| HLD-U007 Identifier hierarchy | Security-master documentation plus holdings output proving keys, type/symbol roles, and cross-system mapping. |
| HLD-U008 Principal paydowns | One native `pd` event with pre/post principal or factor, quantity, market value, accrued interest, cash, and performance. |
| HLD-U009 REP portability | Axys/APX Replang manuals and source for `CDIhold.rep`, Portfolio Appraisal, and representative custom reports. |

Highest-value next acquisition: one sanitized account package containing a
current and historical holdings extract, the report definition and settings,
transactions, prices, security master, cash, and lot detail for the same as-of
dates.

## Maintenance Rule

Add a claim only for new provenance, a narrowed boundary, or a contradiction.
Update Chapter 06 when reader guidance or an Unknown changes. Do not append
worked examples, draft chapter structures, or another narrative research pass.
