# Chapter 06 — Holdings

Repository: AXYS / APX Reference Repository
Chapter file: `docs/axys-apx-reference/reference/Chapter_06_Holdings.md`
Prepared: 2026-06-29
Governing specification: `AXYS_APX_REFERENCE_BLUEPRINT.md`, Version 2.0

---

## Related chapters
- [Chapter_01_Overview.md](Chapter_01_Overview.md) — provides the repository-wide map and evidence conventions.
- [Chapter_05_Transactions.md](Chapter_05_Transactions.md) — holdings are updated from posted transactions.
- [Chapter_07_Cash.md](Chapter_07_Cash.md) — holdings and cash often move together in reconciliation workflows.
- [Chapter_10_Performance.md](Chapter_10_Performance.md) — holdings provide the base for performance measurement.

## Transaction-Code Boundary

This chapter explains holdings effects and position evidence. It should not
maintain a separate transaction-code dictionary. Interpret transaction codes
through [Chapter_05_Transactions.md](Chapter_05_Transactions.md) and use this
chapter only for the holdings-side implications, such as quantity, market value,
principal, accrued income, lots, and cost-basis context.

## 1. Overview

This chapter documents holdings and position-related behavior in Axys and APX using only the supplied research material.

In this chapter, **holdings** means a portfolio's assets or positions as shown or extracted at a point in time. The available evidence is strongest for report output, reconciliation workflows, third-party integration workflows, and named artifacts. The available evidence is weaker for native storage models, canonical IMEX object names, and internal Axys/APX calculation rules.

A holdings change should be read as the outcome of posted transactions and associated accounting rules, not as a standalone event inferred from a transaction code. For performance and reconciliation work, it is helpful to distinguish external cash flows from trading activity, income, fees, and corporate-action events because those categories affect cash, lots, cost basis, and valuation differently.

### Confidence labels

| Label | Meaning |
|---|---|
| Verified | Directly supported by supplied research and cited source material. |
| High Confidence | Strongly supported by one or more sources, but not a complete vendor specification. |
| Medium Confidence | Plausible and consistent with available sources, but evidence is partial, indirect, source-specific, or integration-specific. |
| Unknown | Not established by supplied material. Do not promote to fact without additional evidence. |

### Evidence boundary

| Area | Current documentation status |
|---|---|
| Axys Portfolio Appraisal report behavior | Strong evidence. |
| Axys group Portfolio Appraisal behavior | Strong evidence from CSSI example. |
| APX Portfolio Appraisal behavior | Partial evidence only; full APX reports guide not supplied. |
| AIA holdings extraction and reconciliation workflows | Strong evidence for those workflows only. |
| APX Custodial Integrator position / lot blotter workflow | Strong evidence for that workflow only. |
| Native Axys holdings storage | Unknown. |
| Native APX holdings table/view names | Unknown. |
| Canonical holdings IMEX object names | Unknown for both Axys and APX. |
| Complete default holdings report field dictionaries | Unknown. |

---

## 2. Axys

### 2.1 Axys holdings report concept

| Statement | Confidence | Notes |
|---|---:|---|
| Axys has a report called `Portfolio Appraisal`. | Verified | CSSI documentation walks through creating a new Portfolio Appraisal in Axys Report Writer. |
| Axys Portfolio Appraisal can be used as a holdings/assets point-in-time report. | High Confidence | The research includes a Code of Ethics example where an Axys Portfolio Appraisal Report can be used in lieu of a separate holdings report for specified accounts. |
| Axys Report Writer can create a Portfolio Appraisal with columns selected through `Define -> Columns`. | Verified | CSSI example. |
| `Portfolio Code` is an available Portfolio Appraisal column in Axys Report Writer. | Verified | CSSI example. |
| Axys Report Writer has a `Management Mode` option relevant to group Portfolio Appraisal output. | Verified | CSSI example. |
| A CSSI sample Portfolio Appraisal displayed `Quantity`, `Security`, `Price`, `Market Value`, `Pct Assets`, `Yield`, and `Portfolio Code`. | Verified for sample | Do not treat this as a complete default field list. |
| The same CSSI sample includes grouping labels such as `EQUITY MUTUAL FUNDS`, `U.S. Equity`, and `Large Cap`. | Verified for sample | Treat these as example report headings, not universal classification values. |

### 2.2 Axys group Portfolio Appraisal behavior

| Group / report condition | Observed behavior | Confidence |
|---|---|---:|
| Consolidated group | Produces a single Portfolio Appraisal showing assets for the entire group. | Verified |
| Unconsolidated group | Produces multiple portfolio appraisals, one for each group member. | Verified |
| Unconsolidated group with custom Report Writer setup | CSSI example shows a single combined appraisal can be produced with owner `Portfolio Code` beside each holding by adding `Portfolio Code` and using `Management Mode`. | Verified for example |

Implementation note: owner-portfolio identification in a combined Axys holdings report should not be assumed to appear by default. In the CSSI example it is explicitly added through Report Writer.

### 2.3 Axys report customization and REP artifacts

| Artifact / label | Description | Confidence |
|---|---|---:|
| `Portfolio Appraisal` | Holdings/assets report created in Axys Report Writer. | Verified |
| `Portfolio Code` | Axys Report Writer column available in Portfolio Appraisal. | Verified |
| `Management Mode` | Axys Report Writer option used in the CSSI group Portfolio Appraisal example. | Verified |
| `$askport` | Header variable used in CSSI example to show the CLI code entered when report is run. | Verified for example |
| `$:fileo` | REP variable used in a CSSI AUM report example to add portfolio code. | Verified as CSSI statement |
| `$:tfile` | REP variable used by CSSI to show the CLI file containing a transaction in a transaction-summary context. | Verified as CSSI statement |
| `aman.rep` | Axys Assets Under Management report file in CSSI example. | Verified for example |
| `_aumsect.rep` | User-created copy of `aman.rep` in CSSI example. | Verified for example |
| `\axys3\rep` | Example Axys report-file directory in CSSI material. | Verified for cited example only |
| `Replang` | Axys report programming language referenced by CSSI. | Verified |

### 2.4 Axys holdings-related files and source artifacts

The supplied material identifies several Axys files that are relevant to holdings, reconciliation, or conversion. This does **not** establish a native Axys holdings-storage table or file.

| File / artifact | Description in supplied evidence | Holdings relevance | Confidence |
|---|---|---|---:|
| `.cli (clients file)` | Client files in Morningstar conversion guide. | Portfolio/client source context; cost-basis and client-file settings may affect converted data. | Verified |
| `sec.inf (securities file)` | Securities file in Morningstar conversion guide. | Security master input for holdings interpretation. | Verified |
| `type.inf (security type file)` | Security type file in Morningstar conversion guide. | Security type/classification input for holdings interpretation. | Verified |
| `split.inf (securities splits file)` | Securities splits file in Morningstar conversion guide. | Split information may affect position quantities. | Verified |
| `.pri (security prices file)` | Security prices file in Morningstar conversion guide. | Pricing input for holdings valuation. | Verified |
| `.pos` | Position source/import files in AIA and Custodial Integrator contexts. | Position files used in reconciliation/import workflows. | Verified for cited workflows |
| `CDIhold.rep` | WealthTechs-provided report used to calculate historical holdings in AIA workflow. | Historical holdings extract. | Verified for AIA workflow |
| `$pathCDI` | Custom label mapped to network path for AIA holdings extract workflow. | Output path for holdings extract workflow. | Verified for workflow |
| `Holdings Extract Folder (h.CDI)` | AIA advanced setting mapped to holdings extract folder. | Output destination for holdings extract workflow. | Verified for workflow |

### 2.5 Axys current-date versus historical holdings extraction in AIA

| Scenario | Observed behavior | Confidence | Caveat |
|---|---|---:|---|
| Current-date Axys extraction in AIA | AIA can read current-date Axys data files directly. | Verified for AIA workflow | Do not generalize to all Axys extraction methods. |
| Historical Axys extraction in AIA | Historical loading requires a report to calculate holdings. WealthTechs provides `CDIhold.rep`. | Verified for AIA workflow | The report is workflow-specific. |
| NBIN interlisted duplicate handling | AIA may read Axys holdings as of the data-file date and remove duplicate holdings and prices from `.pos` and `.pri` source files. | Verified for AIA workflow | Applies to the documented AIA/NBIN workflow. |

### 2.6 Axys reconciliation and conversion behavior

| Topic | Statement | Confidence | Caveat |
|---|---|---:|---|
| Custodian master position file | In AIA, the custodian downloads a master position file listing current holdings by account with market value and quantity. | Verified for AIA workflow | Third-party workflow. |
| POS file creation | The master position file is used to create POS files in Advent. | Verified for AIA workflow | Third-party workflow. |
| Reconciliation report | AIA reconciliation compares calculated positions in Advent to downloaded custodian positions and displays non-equal positions. | Verified for AIA workflow | Third-party workflow. |
| Morningstar conversion reconciliation | A Reconciliation report from Advent Axys as of the last transaction date of backup is used by Morningstar to show out-of-balance items compared to the custodian record. | Verified | Conversion-specific. |
| Principal paydown methodology | Morningstar states Axys adjusts original principal amount to decrease principal balance, while Morningstar uses principal factors in holding price calculation. | Verified as conversion limitation | Do not document as an Axys defect. |
| Zero-quantity principal paydown issue | Some principal paydown transaction types are provided with zero share quantity and cannot be processed by Morningstar Office; affected converted holdings may not match Axys performance reporting results. | Verified as conversion limitation | Conversion-specific; native Axys behavior remains partially unknown. Do not use quantity alone to decide whether a `pd` principal event occurred. |
| Short-side positions | Public integration evidence supports `ss` as short sale and `cs` as cover short, but native holdings representation is not established. | Medium-High for code meaning; Unknown for native holdings representation | A disclosed synthetic demo may model negative quantity and negative market value. For production, do not assume whether the site uses negative quantity, negative market value, separate short position, separate account, or short/margin cash until a holdings or position extract proves it. |

### 2.7 Axys holdings storage model

| Statement | Confidence | Notes |
|---|---:|---|
| The supplied material does not identify a canonical Axys holdings storage file or table separate from reports, transactions, security files, price files, and position/reconciliation artifacts. | Unknown | Do not invent a holdings table/file name. |
| Axys holdings appear to be reportable from portfolio/client, transaction, security, price, split, and position-related data sources. | Medium Confidence | This is an interpretation from the file/report evidence, not a vendor storage specification. |
| Whether Axys stores current positions persistently, calculates them on demand, or uses multiple mechanisms depending on report/workflow is Unknown. | Unknown | Requires vendor docs or production examples. |

---

## 3. APX

### 3.1 APX product and holdings context

| Statement | Confidence | Notes |
|---|---:|---|
| APX is described by SS&C as an integrated portfolio and client management solution. | Verified | Product-level context only. |
| APX 20.1 is described as a core accounting, reporting, and performance measurement application. | Verified | Product-level/release context. |
| SS&C announced APX enhancements in 2015 including improvements to position reconciliation, dividend processing, and cost basis handling. | Verified | Release/update context. |
| APX has a SQL layer from which current-date holdings can be read in the WealthTechs AIA workflow. | Verified for AIA workflow | Do not infer APX table or view names. |

### 3.2 APX Portfolio Appraisal

| Statement | Confidence | Notes |
|---|---:|---|
| APX has a Portfolio Appraisal report. | Verified for report concept | Report output is not proof of database, public-view, or IMEX field names. |
| APX Portfolio Appraisal can show holdings by individual tax lot or position. | Verified for report concept | Report output is not proof of lot-storage mechanics. |
| APX Portfolio Appraisal output can include quantity, cost, market value, percent of portfolio, yield, and unrealized gain/loss concepts. | Verified for report concept | Treat these as report labels until exact datasets and fields are supplied. |
| Public APX sample client reports may include a `PORTFOLIO APPRAISAL` section and report parameters such as `STY:APX`. | Medium Confidence | Downstream client-report evidence only; do not treat as a vendor report specification. |
| Standard APX Portfolio Appraisal default columns are Unknown. | Unknown | No complete APX report output or guide section supplied. |
| APX group behavior for Portfolio Appraisal is Unknown. | Unknown | No APX-specific evidence comparable to the Axys CSSI group example. |

### 3.2.1 Market value and accrued interest in appraisal output

The July 2026 market-value/accrued-interest research strengthens a fixed-income
holdings rule: Portfolio Appraisal-style output should be read as showing market
value and accrued interest separately when both are present. Public SS&C report
samples and a public APX Portfolio Appraisal example strongly imply that Market
Value is clean market value and that accrued interest is a separate line or
field used in total/subtotal presentation.

For holdings extraction and audit work, do not assume `holdings.market_value`
already includes accrued interest. Preserve `holdings.accrued` separately when
available, and derive dirty or total position value as market value plus accrued
interest when the downstream performance method requires it. A dedicated stored
Total Value field remains Unknown.

### 3.3 APX Custodial Integrator position import and reconciliation

The APX Custodial Integrator material provides the strongest APX position-import evidence. This is a third-party integration workflow, not a complete APX native data model.

| Item / setting | Observed behavior | Confidence |
|---|---|---:|
| `apxix.exe` | APX Import/Export utility executable used in Custodial Integrator workflow. | Verified |
| APX executable folder | Configuration field required by Custodial Integrator. | Verified |
| APX output folder | Used to transfer data to and from APX; should not contain spaces. | Verified |
| APX log folder | Used for APX Import/Export utility logs. | Verified |
| Portfolio group name | Contains portfolios referenced in the integration workflow. | Verified |
| Trade Blotter name | Transaction-import target; blotter must be created in APX. | Verified |
| Position Blotter name | Position-import target for reconciliation; blotter must be created in APX. | Verified |
| Lot Blotter name | Position-lot import target when position lots are enabled; blotter must be created in APX. | Verified |
| `Include positions for export` | Retrieves prior-business-day positions and imports them to the named Position Blotter. | Verified |
| `Include position lots` | If lots are enabled, lots are not imported by default; this option includes lots for the next import. | Verified |
| Advanced export options | Position options also apply to position lots when lots are enabled and imported. | Verified |
| Exclude stale accounts | Position export can exclude positions for stale accounts. | Verified |
| Exclude failed accounts | Position export can exclude positions for failed accounts. | Verified |
| Security price behavior | Custodial Integrator can download security prices for positions for the prior business day and will not overwrite existing same-day APX price records that already contain a price. | Verified |
| APX Security Information / Security Type Information | Custodial Integrator maintains a copy for generating positions, prices, and transactions for APX import. | Verified |
| `APX Portfolio Code` | Portfolio-code translation field identifying the portfolio in APX to which data is delivered. | Verified for CI workflow |

### 3.4 APX AIA holdings extraction and reconciliation

| Scenario | Observed behavior | Confidence | Caveat |
|---|---|---:|---|
| Custodian master position file | In the WealthTechs APX AIA workflow, a custodian master position file lists current holdings by account with market value and quantity. | Verified | Third-party workflow. |
| POS file creation | The master position file is used to create POS files in Advent. | Verified for AIA workflow | Third-party workflow. |
| Reconciliation | AIA Reconciliation report compares calculated positions in Advent versus downloaded custodian positions and displays positions not equal between custodian and Advent. | Verified for AIA workflow | Third-party workflow. |
| NBIN interlisted duplicate handling | AIA reads APX holdings as of the data-file date and removes unnecessary duplicate holdings and prices from `.pos` and `.pri` source files. | Verified for AIA workflow | Third-party workflow. |
| Current-date APX extraction | Current-date APX data can be read from APX SQL in the AIA workflow. | Verified for AIA workflow | Do not infer table/view names. |
| Historical APX extraction | Historical loading requires a report to calculate holdings; WealthTechs provides `CDIhold.rep`. | Verified for AIA workflow | Report is workflow-specific. |
| `CDIhold.rep` setup | Added to the custom report menu in APX for this workflow. | Verified for AIA workflow | Third-party workflow. |
| `$pathCDI` setup | Added in `APX > Admin > Global Settings > Configurations` and mapped to a network path. | Verified for AIA workflow | Third-party workflow. |
| `Holdings Extract Folder (h.CDI)` | AIA advanced setting mapped to the same network path. | Verified for AIA workflow | Third-party workflow. |
| Holdings extract group | Typically `cdirecon` in the documented workflow. | Verified as typical in source | Do not treat as mandatory. |

### 3.5 APX holdings storage model

| Statement | Confidence | Notes |
|---|---:|---|
| APX current-date holdings can be read from APX SQL in the AIA workflow. | Verified for AIA workflow | Does not identify SQL objects. |
| Native APX holdings table names, public view names, stored accounting functions, and stored-versus-calculated rules are Unknown. | Unknown | Requires APX schema, public-view documentation, or production SQL extract. |
| Whether APX stores current positions, calculates holdings through stored accounting functions, or uses both depending on workflow is Unknown. | Unknown | Current supplied evidence is insufficient. |

---

## 4. IMEX / Import-Export

### 4.1 Holdings IMEX status

The supplied material does not provide a formal Axys or APX IMEX object dictionary for holdings. It identifies workflows where holdings/positions are imported, exported, or calculated through third-party tools and reports.

| Area | Axys | APX | Confidence |
|---|---|---|---:|
| Formal holdings IMEX object name | Unknown. | Unknown. | Unknown |
| Formal position IMEX object name | Unknown. | Unknown. | Unknown |
| Formal lot-level holdings IMEX object name | Unknown. | Unknown. | Unknown |
| Position import artifact | POS files can be created from custodian master position files in AIA workflows. | POS files can be created from custodian master position files in AIA workflows. | Verified for AIA workflows |
| Position blotter | Unknown from supplied Axys material except POS-file reconciliation workflows. | Custodial Integrator imports positions into APX Position Blotter. | APX Verified for CI workflow |
| Lot blotter | Unknown. | Custodial Integrator imports lots into APX Lot Blotter when position lots are enabled and selected. | APX Verified for CI workflow |
| Current-date extraction | AIA can read current-date data files from Axys. | AIA can read current-date data from APX SQL. | Verified for AIA workflow |
| Historical extraction | AIA historical holdings loading requires report calculation via `CDIhold.rep`. | AIA historical holdings loading requires report calculation via `CDIhold.rep`. | Verified for AIA workflow |

### 4.2 IMEX-related files and folders

| Artifact | Axys | APX | Description | Confidence |
|---|---:|---:|---|---:|
| `.pos` | Yes | Yes in source-file workflow | Position files created or manipulated in AIA / reconciliation workflows. | Verified for workflows |
| `ptopost.trn` | Yes | No | CI writes positions to `\CI\exported\ptopost.trn` in CSV format. | Verified for CI workflow |
| `.pri` | Yes | Yes in source-file workflow | Price files; relevant to valuation and duplicate interlisted security handling. | Verified for workflows |
| `pospos32.exe` | Yes | No | Axys Position Post utility can create replacement `.pos` files for configured portfolios. | Verified for CI workflow |
| `apxix.exe` | No | Yes | APX Import/Export utility executable in Custodial Integrator workflow. | Verified |
| APX output folder | No | Yes | Folder used to transfer data to and from APX in CI workflow; should not contain spaces. | Verified |
| APX log folder | No | Yes | Folder where APX Import/Export logs are used/displayed in CI workflow. | Verified |
| `imexPositionLots.log` | Yes | Yes in CI context | Position-lot import log can appear when lots are enabled and available. | Verified for CI workflow |
| `CDIhold.rep` | Yes | Yes | Report-mediated historical holdings calculation in AIA workflow. | Verified for workflow |

### 4.3 IMEX unknowns

| Unknown | Why it matters |
|---|---|
| Canonical Axys IMEX object name for current positions/holdings. | Needed to document reliable holdings exports. |
| Canonical APX IMEX object name for current positions/holdings. | Needed to document reliable APX holdings exports. |
| Whether holdings are exported from stored snapshots, calculated reports, SQL views, or multiple mechanisms. | Determines reconciliation and repeatability. |
| Whether lot-level holdings are available through Axys IMEX. | Needed for tax-lot holdings documentation. |
| Whether lot-level holdings are available through APX IMEX outside Custodial Integrator. | Needed for APX lot-level extracts. |
| Standard holdings import/export field list. | Needed for implementation. |
| Position date semantics: trade date, settlement date, position date, close date, or report as-of date. | Critical for reconciliation and performance. |

---

## 5. REP / Report Layer

### 5.1 Holdings-related reports and REP artifacts

| Report / file / variable | System | Description | Confidence |
|---|---|---|---:|
| `Portfolio Appraisal` | Axys | Holdings/assets point-in-time report; can be generated in Report Writer. | Verified |
| `Portfolio Appraisal` | APX | APX reports guide search-result evidence says it shows holdings by tax lot or position. | Medium Confidence |
| `CDIhold.rep` | Axys | WealthTechs-provided report for historical holdings calculation in AIA / NBIN duplicate handling workflow. | Verified for workflow |
| `CDIhold.rep` | APX | WealthTechs-provided report for historical holdings calculation in AIA / NBIN duplicate handling workflow. | Verified for workflow |
| `aman.rep` | Axys | Assets Under Management report copied in CSSI AUM-by-sector example. | Verified for example |
| `_aumsect.rep` | Axys | User-created copy of `aman.rep` in CSSI AUM-by-sector example. | Verified for example |
| `Reconciliation report` | Axys | Used by Morningstar conversion process to compare Axys to custodian records as of last transaction date. | Verified |
| WealthTechs Reconciliation reports | Axys/APX | AIA says WealthTechs has two reconciliation reports; one displays on the Advent report screen and one exports directly to Excel. | Verified for workflow |
| `$askport` | Axys | Header variable in CSSI Portfolio Appraisal example. | Verified for example |
| `$:fileo` | Axys | Used in AUM report article to add portfolio code. | Verified as CSSI statement |
| `$:tfile` | Axys | Used to show transaction source CLI file in transaction-summary context. | Verified as CSSI statement |
| `$firmg` | Axys | Used as “Other” catch-all classification variable in CSSI AUM sector example. | Verified for example |
| `$pathCDI` | Axys/APX | Custom label mapped to holdings extract network path in AIA workflow. | Verified for workflow |

### 5.2 REP cautions for holdings

| Caution | Confidence | Notes |
|---|---:|---|
| A report output field label is not automatically a native database, file, or IMEX field name. | High Confidence | Applies to all report-derived fields. |
| `CDIhold.rep` is a WealthTechs-provided report in the AIA workflow and should not be documented as a standard Axys/APX report without additional evidence. | Verified for workflow | Standard/vendor status Unknown. |
| Report customization examples prove only the demonstrated report and context. | High Confidence | Do not infer complete Report Writer or Replang syntax from examples. |
| Historical holdings extraction in AIA requires report calculation even where current-date data can be read directly. | Verified for workflow | Important for repeatability and reconciliation. |

---

## 6. Data Model

### 6.1 Conceptual holdings dependencies

The supplied evidence supports a conservative conceptual dependency model. This is not a native Axys/APX schema.

```text
Portfolio / client account
        ↓
Transactions and/or posted accounting records
        ↓
Security master and security type data
        ↓
Prices and corporate actions / splits
        ↓
Holdings / positions report or extract
        ↓
Reconciliation, valuation, reporting, performance, conversion
```

### 6.2 Holdings-related source categories

| Source category | Axys evidence | APX evidence | Confidence |
|---|---|---|---:|
| Portfolio/account | `.cli` client files; Portfolio Code in Report Writer examples. | `APX Portfolio Code` in CI portfolio-code translation. | Verified for cited contexts |
| Security master | `sec.inf`; `type.inf`; security fields appear in reports. | APX Security Information and Security Type Information used by CI. | Verified for cited contexts |
| Transactions | Holdings conceptually depend on transactions; Morningstar conversion issues link transaction handling to converted holdings. | CI/AIA workflows use blotters and reconciliation; transaction-to-position internals not fully described. | Medium Confidence |
| Prices | `.pri`; price column and market value in report output. | CI downloads prices for positions; will not overwrite same-day APX price records already containing price. | Verified for cited contexts |
| Splits/corporate actions | `split.inf` described as securities splits file. | Not established in supplied APX holdings evidence. | Axys Verified for conversion file; APX Unknown |
| Position files | `.pos` source/import files in AIA workflows. | `.pos` source files and Position Blotter workflow. | Verified for workflows |
| Lot-level data | Unknown from supplied Axys material. | APX Lot Blotter exists when position lots are enabled for the firm in CI workflow. | APX Verified for CI workflow; Axys Unknown |

### 6.3 Stored versus calculated holdings

| Question | Axys | APX | Confidence |
|---|---|---|---:|
| Are current holdings stored in a canonical native table/file? | Unknown. | Unknown. | Unknown |
| Can current holdings be read directly in an AIA workflow? | Yes, current-date Axys data files can be read directly in that workflow. | Yes, current-date APX data can be read from APX SQL in that workflow. | Verified for AIA workflow |
| Can historical holdings require report calculation? | Yes, in AIA workflow via `CDIhold.rep`. | Yes, in AIA workflow via `CDIhold.rep`. | Verified for AIA workflow |
| Are lot-level holdings available? | Unknown from supplied material. | Available in APX CI workflow if position lots are enabled and `Include position lots` is selected. | APX Verified for CI workflow; Axys Unknown |

---

## 7. Common Fields

The table below lists fields and labels observed in supplied source material. It is **not** a complete native holdings field dictionary.

| Field / Label | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---:|
| `Quantity` | Holding quantity displayed in CSSI Portfolio Appraisal sample. | Yes | Unknown | Unknown | Yes | Verified for sample |
| `Security` | Security name/description displayed in CSSI Portfolio Appraisal sample. | Yes | Unknown | Unknown | Yes | Verified for sample |
| `Price` | Price displayed in CSSI Portfolio Appraisal sample. | Yes | Unknown | Unknown | Yes | Verified for sample |
| `Market Value` | Market value displayed in CSSI Portfolio Appraisal sample. | Yes | Unknown | Unknown | Yes | Verified for sample |
| `Pct Assets` | Percent-of-assets displayed in CSSI Portfolio Appraisal sample. | Yes | Unknown | Unknown | Yes | Verified for sample |
| `Yield` | Yield displayed in CSSI Portfolio Appraisal sample. | Yes | Unknown | Unknown | Yes | Verified for sample |
| `Portfolio Code` | Owner portfolio code available as Axys Portfolio Appraisal column in Report Writer. | Yes | Unknown | Unknown | Yes | Verified |
| `APX Portfolio Code` | Portfolio-code translation field identifying portfolio in APX to which CI data is delivered. | No | Yes | Yes, CI workflow | No | Verified for CI workflow |
| `Symbol` | Missing-prices output field; may be any APX security symbol defined in security master. | Unknown | Yes | Yes, CI missing-prices output | No | Verified for CI workflow |
| `Type` | Missing-prices output field; may be any APX security type defined in security master. | Unknown | Yes | Yes, CI missing-prices output | No | Verified for CI workflow |
| `Name` | Missing-prices output field: name of security or position with no price. | Unknown | Yes | Yes, CI missing-prices output | No | Verified for CI workflow |
| `WP Account` | WebPortfolio account nickname in CI output. | No | External-to-APX | Yes, CI output | No | Verified for CI workflow |
| `Institution` | Financial institution name in CI output. | No | External-to-APX | Yes, CI output | No | Verified for CI workflow |
| Position Blotter name | APX object into which CI imports positions for reconciliation. | Unknown | Yes | Yes | No | Verified for CI workflow |
| Lot Blotter name | APX object into which CI imports position lots when lots enabled. | Unknown | Yes | Yes | No | Verified for CI workflow |
| Trade Blotter name | APX object into which CI imports transactions. | Unknown | Yes | Yes | No | Verified for CI workflow |
| `Holdings Extract Folder (h.CDI)` | AIA setting mapped to network folder containing holdings extract output. | Yes | Yes | Workflow setting | Report output destination | Verified for AIA workflow |

### Field cautions

| Caution | Confidence |
|---|---:|
| Axys Portfolio Appraisal sample fields should not be assumed to be complete default columns. | High Confidence |
| APX Portfolio Appraisal fields remain mostly Unknown until the APX Reports Guide section or sample report is supplied. | Unknown |
| CI missing-prices fields are integration output fields, not native APX holdings fields. | Verified for CI context |
| AIA settings and `CDIhold.rep` fields are workflow-specific, not native object names unless separately verified. | Verified for workflow |

---

## 8. Examples

### 8.1 Axys: group Portfolio Appraisal with owner portfolio code

Source-backed Axys behavior:

1. Create a new Axys Report Writer report: `File -> New -> Portfolio Appraisal`.
2. Optionally add `$askport` to the report header to display the CLI code entered in the run dialog.
3. Use `Define -> Columns` and add `Portfolio Code`.
4. Use `Define -> Options` and select `Management Mode`.
5. Run the report for an unconsolidated group.
6. In the CSSI example, the report produces one combined Portfolio Appraisal for the group and the `Portfolio Code` column identifies which CLI file owns each holding.

Classification: **Verified for the CSSI example.** Do not assert that this is the only or default way to identify owner portfolio in every Axys holdings report.

### 8.2 Axys/APX: AIA historical holdings extract for interlisted duplicate handling

Source-backed workflow:

1. NBIN source files may contain duplicate price and position data for interlisted securities.
2. AIA reads holdings as of the data-file date and removes unnecessary duplicate holdings and prices from `.pos` and `.pri` source files.
3. In Axys, current-date data files can be read directly from Axys.
4. In APX, current-date data can be read from APX SQL.
5. Historical data loading in both workflows requires a report to calculate holdings.
6. WealthTechs supplies `CDIhold.rep` for this purpose.
7. The report is added to the custom report menu and `$pathCDI` is mapped to a network path.
8. AIA Advanced settings map `Holdings Extract Folder (h.CDI)` to that path.
9. The holdings extract group is typically `cdirecon` in the documented workflow.

Classification: **Verified for the WealthTechs AIA workflow.**

### 8.3 APX: Custodial Integrator position and lot import for reconciliation

Source-backed workflow:

1. Configure the APX executable folder containing `apxix.exe`.
2. Configure the APX output folder and APX log folder.
3. Configure a Portfolio group name.
4. Configure a Position Blotter name; the blotter must be created in APX.
5. If position lots are enabled, configure a Lot Blotter name; the blotter must be created in APX.
6. Enable `Include positions for export` to retrieve prior-business-day positions and import them into the named Position Blotter.
7. If lots are enabled, lots are not imported by default. Use `Include position lots` for the next import.
8. Use stale/failed-account exclusions where applicable.

Classification: **Verified for Custodial Integrator APX workflow.**

### 8.4 Axys conversion: holdings mismatch after principal paydown conversion

Source-backed conversion issue:

1. Morningstar conversion guidance identifies Axys files including `.cli`, `sec.inf`, `split.inf`, `.pri`, and `type.inf`.
2. Morningstar states that Axys adjusts original principal amount to decrease principal balance.
3. Morningstar uses principal factors in holding price calculation.
4. Some Axys principal paydown transaction types may be provided with zero share quantity and cannot be processed by Morningstar Office.
5. In those cases, converted holdings may not match Axys performance reporting results.

Classification: **Verified as Morningstar conversion limitation.** Do not document as a native Axys error without additional evidence.

The 2026-07-07 `rc`/`pd` and `pd` Modified Dietz research reinforces that `pd`
is best treated as a principal-paydown or bond return-of-capital event, not as
ordinary coupon income. For holdings audits, preserve principal, factor,
amortization, or equivalent principal-balance evidence when available; share
quantity alone may be zero or insufficient for reconciliation.

---

## 9. Known Issues / Quirks

| Quirk | Axys | APX | Confidence | Notes |
|---|---:|---:|---:|---|
| Consolidated versus unconsolidated group settings materially affect Portfolio Appraisal output. | Yes | Unknown | Axys Verified | Axys consolidated group gives one appraisal; unconsolidated group gives multiple appraisals unless customized as in CSSI example. |
| Owner portfolio code in a combined group report may require explicit Report Writer customization. | Yes | Unknown | Verified for Axys example | Add `Portfolio Code` column. |
| Historical holdings extraction may require a report even when current-date data can be read directly. | Yes | Yes | Verified for AIA workflow | Axys current-date files; APX current-date SQL; historical via `CDIhold.rep`. |
| Interlisted securities may create duplicate position and price data in NBIN source files. | Yes | Yes | Verified for AIA workflow | AIA removes unnecessary duplicate holdings/prices from `.pos` and `.pri`. |
| APX output folder in Custodial Integrator should not contain spaces. | No | Yes | Verified | CI APX guide states this. |
| APX position lots are optional / firm-dependent in Custodial Integrator. | No | Yes | Verified | Lot Blotter field appears if position lots are enabled. |
| Custodial Integrator does not import APX lots by default even when lots are enabled. | No | Yes | Verified | User must select `Include position lots` for next import. |
| APX position import can exclude stale or failed accounts. | No | Yes | Verified | CI advanced position export options. |
| Custodial Integrator will not overwrite APX same-day security price records that already contain a price. | No | Yes | Verified | Relevant to position valuation/reconciliation. |
| Principal paydown handling can cause converted holdings not to match Axys performance reporting results. | Yes, conversion context | Unknown | Verified as conversion issue | Morningstar methodology difference and zero-quantity limitation. |
| AIA / CI workflows are not native schema documentation. | Yes | Yes | High Confidence | Treat as integration behavior unless vendor docs confirm native behavior. |

---

## 10. Version Differences and Release Notes

| Version / date | System | Evidence | Holdings relevance | Confidence |
|---|---|---|---|---:|
| 2015 Advent product updates | APX / Axys | SS&C / PRNewswire release. | APX enhancements included position reconciliation, dividend processing, and cost basis handling. Axys received UI, CRM2 support, and portfolio template upgrades. | Verified |
| APX 20.1 / 2020 | APX | SS&C product updates. | APX described as core accounting, reporting, and performance measurement application; fixed-income improvements and new UI noted. | Verified |
| AIA User Manual version 3.1.0 / 2022 | Axys/APX integration | WealthTechs AIA manuals. | Documents current versus historical holdings extraction behavior and `CDIhold.rep` workflow. | Verified for AIA workflow |
| Morningstar conversion guide Version 2.0 | Axys conversion | Morningstar PDF. | Documents files required for conversion and principal paydown limitations. | Verified |

### Version-difference unknowns

| Unknown | Notes |
|---|---|
| Whether Axys Portfolio Appraisal default columns differ across Axys versions. | Requires official reports guide or versioned report outputs. |
| Whether APX Portfolio Appraisal behavior differs across APX versions. | Requires APX Reports Guide and sample outputs. |
| Whether AIA `CDIhold.rep` behavior differs across Axys/APX versions. | Requires report source and workflow documentation. |
| Whether current-date versus historical extraction behavior differs outside AIA. | Requires vendor documentation or production observations. |

---

## 11. References

### Supplied repository and research sources

1. `AXYS_APX_REFERENCE_BLUEPRINT.md`, Version 2.0. Governing repository specification.
2. `../evidence/Research_06_Holdings.md`. Primary research source for this chapter.
3. `../evidence/Research_04_Security_Master.md`. Used only for security-master context where holdings depend on security identity/type.
4. `../evidence/Research_05_Transactions.md`. Used only for transaction-to-holdings dependency context.
5. `../evidence/Research_12_IMEX.md`. Used only for IMEX/interface context and file/artifact separation.
6. `../evidence/Research_13_REP.md`. Used only for REP/report-layer terminology and report caution.

### External source references represented in the supplied research

| Source | Use in this chapter |
|---|---|
| SS&C Advent Axys product page | Product/reporting context only. |
| SS&C Advent APX product page / product brief | Product/reporting context only. |
| SS&C 1H2020 Advent Product Updates | APX 20.1 context. |
| SS&C / PRNewswire 2015 Advent Product Updates | Position reconciliation, dividend processing, and cost-basis improvement context. |
| WealthTechs AIA User Manual for Axys Users, version 3.1.0, 2022 | AIA POS/reconciliation/current-versus-historical holdings extraction and `CDIhold.rep` behavior. |
| WealthTechs AIA User Manual for APX Users, version 3.1.0, 2022 | APX SQL current-date holdings, historical report-calculated holdings, `.pos` / `.pri`, and AIA settings. |
| ByAllAccounts / Morningstar Custodial Integrator User Guide for APX | `apxix.exe`, APX output/log folders, portfolio group, trade/position/lot blotters, position and lot import behavior, stale/failed account exclusions, security price behavior. |
| CSSI, `Accessing Cli PortCodes Within Groups` | Axys Portfolio Appraisal group behavior, Report Writer steps, Portfolio Code column, Management Mode, `$askport`, `$:fileo`, `$:tfile`, and sample columns. |
| CSSI, `How To Create an Assets Under Management By Sector Report` | `\axys3\rep`, `aman.rep`, `_aumsect.rep`, copy-before-modify guidance, `$firmg` example. |
| Morningstar, `Converting Your Advent Axys Database into Morningstar Office`, Version 2.0 | Axys conversion files, Reconciliation report, principal paydown / zero-quantity limitations, conversion issue categories. |
| Chase Investment Counsel Code of Ethics filing | Evidence that Axys Portfolio Appraisal can be used as a holdings report for specified accounts. |
| Advent Portfolio Exchange Reports Guide search result | APX Portfolio Appraisal evidence; still only Medium Confidence until the guide itself is supplied and reviewed. |

---

## 12. Unknowns

The following should remain **Unknown** until supported by vendor documentation, production observations, sample reports, sample IMEX exports, or source report files.

| Unknown | Why it matters | Best evidence to resolve |
|---|---|---|
| Canonical Axys holdings IMEX object name(s). | Needed for precise IMEX chapter and extract design. | Axys IMEX manual, sample holdings export, or IMEX macro/control files. |
| Canonical APX holdings IMEX object name(s). | Needed for precise APX extract design. | APX Import/Export manual, `apxix.exe` documentation, or sample export. |
| Axys native holdings storage mechanics. | Needed to distinguish stored positions from report-calculated holdings. | Vendor technical manual or production file layout examples. |
| APX SQL holdings table/view names. | Needed for APX data model. | APX database schema, official reporting data dictionary, public-view docs, or sample query. |
| Lot-level holdings support in Axys. | Needed for field dictionary and tax-lot examples. | Axys report guide or sample lot-level Portfolio Appraisal. |
| Lot-level holdings support in APX Portfolio Appraisal. | Search result suggests tax lot/position support, but full verification is missing. | APX Reports Guide PDF. |
| Standard Portfolio Appraisal default columns for Axys versions. | Current fields are from a CSSI sample and may be customized. | Official Axys reports guide or unmodified report output. |
| Standard Portfolio Appraisal default columns for APX versions. | Needed to avoid assuming Axys columns match APX. | Official APX reports guide or unmodified report output. |
| Date basis for holdings reports. | Critical for reconciliation and performance. | Vendor report options manual or sample reports showing settings. |
| Treatment of cash, shorts, accruals, unsettled trades, FX, and multicurrency holdings. | Critical for real-world holdings interpretation. | Report guides, report outputs, IMEX samples. |
| Security identifier hierarchy used in holdings reports. | Needed for data dictionary. | Security master docs and report output samples. |
| Whether `CDIhold.rep` is generic or WealthTechs-specific. | Avoids documenting a third-party report as a standard report. | Report source or vendor notes. |
| Whether REP syntax and variables differ between Axys and APX. | Needed for REP cross-reference and APX report extraction. | Axys/APX Replang manuals and sample `.rep` files. |
| Whether holdings report values are stored, recalculated, or mixed depending on date/report/settings. | Critical for audit and repeatability. | Controlled production tests or vendor reporting documentation. |

---

## 13. Recommended future evidence

The supplied research is sufficient for this conservative chapter, but a more complete future revision would benefit from:

1. Official Axys reports guide section for Portfolio Appraisal.
2. Official APX reports guide section for Portfolio Appraisal.
3. One unmodified Axys Portfolio Appraisal report output for a simple portfolio.
4. One unmodified APX Portfolio Appraisal report output for a simple portfolio.
5. One lot-level holdings report sample.
6. Axys/APX IMEX or Import/Export manual pages for positions/holdings.
7. Sample `.rep` files: `CDIhold.rep`, `aman.rep`, and any standard Portfolio Appraisal report definition.
8. Sample current-date and historical holdings extracts from Axys and APX.
9. Client-specific notes documenting how holdings are calculated relative to transactions, prices, splits, and accruals.

## 14. Deep IMEX Update

The deep IMEX research adds position-import detail that belongs with holdings
only as IMEX-adjacent position evidence.

| Topic | Chapter treatment | Confidence |
|---|---|---:|
| `ptopost.trn` | CI writes positions to `\CI\exported\ptopost.trn` in CSV format. | Verified for CI |
| `.pos` files | Position Post can create replacement `.pos` files for configured Axys portfolios. | Verified for CI |
| Position lots | If lots are enabled and available, position output may contain lot data and use `imexPositionLots.log`. | Verified for CI |
| Candidate position fields | Portfolio/account, as-of date, symbol/type/name, quantity, price, market value, accrued income, cost, local/base currency, FX, stale flag, and custodian/account context. | Discovery guidance |
| Candidate lot fields | Lot identifier, open/acquisition dates, quantity, cost, market value, tax-cost fields, currency, and source row lineage. | Discovery guidance |

This evidence does not establish native holdings storage mechanics or standard
Portfolio Appraisal columns.
