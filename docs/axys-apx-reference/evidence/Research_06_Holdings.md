# Research Notes: Holdings

Repository: AXYS / APX Reference Repository  
Research file: `docs/axys-apx-reference/evidence/Research_06_Holdings.md`  
Target chapter: `docs/axys-apx-reference/reference/Chapter_06_Holdings.md`  
Prepared: 2026-06-29  
Governing spec: `AXYS_APX_REFERENCE_BLUEPRINT.md`, Version 2.0

## 1. Scope and Method

This file collects source-backed research for the reader-facing holdings chapter. It is not the chapter itself. It is intended to preserve factual evidence, confidence classifications, useful report names, field names, processing behaviors, implementation quirks, and unresolved unknowns.

The governing blueprint requires facts-first documentation, separate Axys and APX treatment, preference for vendor documentation / exports / REP reports / production observations / consultant documentation, and classification of important technical statements as Verified, High Confidence, Medium Confidence, or Unknown.

### Confidence labels used here

| Label | Meaning in this research file |
|---|---|
| Verified | Directly supported by a cited source in the references section or by the uploaded blueprint. |
| High Confidence | Strongly supported by one or more sources, but still not a complete vendor specification. |
| Medium Confidence | Plausible and consistent with available sources, but source coverage is partial or indirect. |
| Unknown | Not established by the available source material. Do not document as fact in Chapter 06 unless additional evidence is supplied. |

## 2. Executive Summary

| Topic | Axys | APX | Confidence |
|---|---|---|---|
| Holdings report concept | Axys has a Portfolio Appraisal report that displays holdings/assets for a portfolio or group. | APX has a Portfolio Appraisal report; an APX reports guide search result states that Portfolio Appraisal shows holdings by tax lot or position. | Axys: Verified. APX: Medium Confidence until full APX reports guide is obtained. |
| Group behavior | A consolidated group produces a single Portfolio Appraisal showing assets for the entire group; an unconsolidated group produces multiple portfolio appraisals, one per member. | Unknown from available APX-specific evidence. | Axys: Verified. APX: Unknown. |
| Owner portfolio code on group holdings | Axys Report Writer can add a Portfolio Code column to Portfolio Appraisal; in Management Mode an unconsolidated group can produce a single combined appraisal with the owner portfolio code beside each holding. | Unknown from available APX-specific evidence. | Axys: Verified. APX: Unknown. |
| Historical holdings extraction | For a third-party AIA workflow, current-date Axys data files can be read directly, but historical loading requires a report to calculate holdings; WealthTechs provides `CDIhold.rep`. | For the analogous APX workflow, current-date data can be read from APX SQL, but historical loading requires a report to calculate holdings; WealthTechs provides `CDIhold.rep`. | Verified for AIA workflow. Do not generalize beyond that workflow without vendor docs. |
| Custodian position files | In AIA, the custodian downloads a master position file listing current holdings by account with market value and quantity; this is used to create POS files in Advent, then compared with Advent-calculated positions. | Same AIA documentation pattern exists for APX users. | Verified for AIA workflow. |
| Position reconciliation imports | Unknown from available Axys material, except POS-file reconciliation workflow in AIA. | Custodial Integrator APX uses `apxix.exe`; imports positions into a Position Blotter and, if enabled, lots into a Lot Blotter. | APX: Verified for Custodial Integrator workflow. Axys: Unknown. |
| Position lots | Unknown from available Axys material. | APX Custodial Integrator documentation has a Lot Blotter and says this field is present if position lots are enabled for the firm. | APX: Verified for Custodial Integrator workflow. |
| Holdings storage model | Axys appears to derive holdings from client/account files, transaction history, security master, prices, and reports; source material does not fully define internal storage. | APX has SQL access for current holdings in AIA workflow, but source material does not identify schema tables or stored-versus-calculated rules. | Unknown / Medium Confidence depending statement. |

## 3. Axys Holdings Research

### 3.1 Portfolio Appraisal

| Statement | Classification | Evidence / Notes |
|---|---:|---|
| Axys has a report called `Portfolio Appraisal`. | Verified | CSSI consultant article walks through creating a new `Portfolio Appraisal` in Axys Report Writer. |
| A Portfolio Appraisal for a consolidated group returns a single portfolio appraisal showing assets for the entire group. | Verified | CSSI article Q/A explicitly states this. |
| A Portfolio Appraisal for an unconsolidated group returns several portfolio appraisals, one for each member. | Verified | CSSI article Q/A explicitly states this. |
| Axys Report Writer can create a Portfolio Appraisal with columns selected through `Define -> Columns`. | Verified | CSSI article gives report-writer steps. |
| `Portfolio Code` is an available Portfolio Appraisal column in Axys Report Writer. | Verified | CSSI article instructs selecting `Portfolio Code` from available columns and adding it. |
| Axys Report Writer has a `Management Mode` option relevant to group Portfolio Appraisal output. | Verified | CSSI article instructs choosing `Define -> Options` and selecting `Management Mode`. |
| In the CSSI example, a Portfolio Appraisal displayed holdings columns: `Quantity`, `Security`, `Price`, `Market Value`, `Pct Assets`, `Yield`, `Portfolio Code`. | Verified | These column labels are visible in the CSSI PDF screenshot/report example. |
| The sample Portfolio Appraisal report groups holdings by broad asset/security categories such as `EQUITY MUTUAL FUNDS`, `U.S. Equity`, and `Large Cap`. | Verified for example only | The CSSI sample shows these group headings. Do not infer universal classification behavior. |
| The Portfolio Appraisal report is suitable evidence of holdings at a point in time. | High Confidence | SEC adviser Code of Ethics example says a Portfolio Appraisal Report run from Axys can be used in lieu of a separate holdings report for specified accounts. This supports real-world treatment of the report as a holdings report. |

### 3.2 Axys report customization and REP location

| Statement | Classification | Evidence / Notes |
|---|---:|---|
| Axys report files reside in a `\axys3\rep` directory in at least some installations. | Verified for cited example | CSSI AUM-by-sector article instructs users to navigate to `\axys3\rep`. Do not assume every installation path is exactly this. |
| `aman.rep` is the Axys Assets Under Management report file name in the CSSI example. | Verified for cited example | CSSI instructs copying `aman.rep` and saving as `_aumsect.rep`. |
| It is a recommended practice to copy a standard `.rep` report before modifying it, not modify original Axys reports directly. | Verified as consultant guidance | CSSI says never modify original Axys reports; make modifications only to copies. |
| `Replang` is the Axys report programming language. | Verified | CSSI article says “Replang, the Axys report programming language.” |
| `$askport` is a report variable used in a CSSI Portfolio Appraisal header example to show the CLI code entered when the report is run. | Verified for cited example | CSSI article describes adding `$askport` to the report header. |
| `$:fileo` is a variable used in a prior AUM report article to add portfolio code to a report. | Verified as CSSI statement | CSSI article references prior use of `$:fileo`. Full semantics require additional REP documentation. |
| `$:tfile` is used by CSSI as a transaction-summary analog to show the CLI file containing a transaction. | Verified as CSSI statement | Relevant to transaction reports, but included because it distinguishes holding owner field behavior from transaction owner behavior. |
| `$firmg` was used in the CSSI AUM-by-sector report example as an “Other” sector catch-all for holdings with no sector defined. | Verified for example only | CSSI article says `$firmg` is entered as a seventh variable named “Other.” Exact variable semantics require source REP docs. |

### 3.3 Axys holdings data sources and file artifacts

| Statement | Classification | Evidence / Notes |
|---|---:|---|
| Morningstar’s Axys conversion guide lists Axys database files supplied for conversion: `.cli` client files, `sec.inf`, `split.inf`, `.pri`, and `type.inf`. | Verified | Morningstar conversion guide lists these files. |
| `.cli` files are described by Morningstar as client files. | Verified | Morningstar conversion guide says `.cli (clients file)`. |
| `sec.inf` is described by Morningstar as the securities file. | Verified | Morningstar conversion guide says `sec.inf (securities file)`. |
| `split.inf` is described by Morningstar as the securities splits file. | Verified | Morningstar conversion guide says `split.inf (securities splits file)`. |
| `.pri` is described by Morningstar as the security prices file. | Verified | Morningstar conversion guide says `.pri (security prices file)`. |
| `type.inf` is described by Morningstar as the security type file. | Verified | Morningstar conversion guide says `type.inf (security type file)`. |
| A Reconciliation report from Advent Axys as of the last transaction date of backup is used by Morningstar to show out-of-balance items compared to the custodian record. | Verified | Morningstar conversion guide states this. |
| The available source material does not identify a canonical Axys holdings storage file separate from reports and transaction/security/price files. | Unknown | No source supplied or found gives a definitive Axys holdings file/table such as a persistent holdings table. |
| Axys holdings can be calculated by reports for historical dates in at least the AIA workflow. | Verified for AIA workflow | WealthTechs AIA Axys guide says current date data files can be read from Axys; historical data loading requires a report to calculate holdings, using `CDIhold.rep`. |
| AIA can create POS files from a custodian master position file listing current holdings, market value, and quantity. | Verified for AIA workflow | WealthTechs AIA Axys guide states this in the Additional Use Cases / Reconciliation section. |
| AIA reconciliation compares calculated positions in Advent to downloaded custodian positions and displays non-equal positions. | Verified for AIA workflow | WealthTechs AIA Axys guide states this. |
| For interlisted securities in the NBIN workflow, duplicate price and position data in source files may require removing duplicate holdings and prices from `.pos` and `.pri` source files. | Verified for AIA workflow | WealthTechs AIA Axys guide states this. |

### 3.4 Axys processing behavior and quirks

| Statement | Classification | Evidence / Notes |
|---|---:|---|
| Consolidated and unconsolidated group settings materially affect Portfolio Appraisal output. | Verified | CSSI article gives different output behavior. |
| `Management Mode` can be used in Axys Report Writer to produce a single group Portfolio Appraisal with owner portfolio code beside each holding. | Verified for CSSI example | CSSI instructions and test result state this. |
| In an AIA historical-load workflow, historical holdings require report calculation rather than direct current-date file read. | Verified for AIA workflow | WealthTechs AIA Axys guide. |
| Principal paydown methodology can affect converted holdings. Morningstar states Axys adjusts original principal amount to decrease principal balance, while Morningstar uses principal factors in holding price calculation. | Verified as Morningstar conversion limitation | Important for fixed income / mortgage-backed holdings conversion and reconciliation. |
| Some principal paydown transaction types are provided with zero share quantity and cannot be processed by Morningstar Office; affected holdings in converted data will not match Axys performance reporting results. | Verified as Morningstar conversion limitation | This is not necessarily an Axys bug; it is a conversion limitation and methodology difference documented by Morningstar. The 2026-07-07 `rc`/`pd` and `pd` Modified Dietz research reinforces that `pd` should preserve principal or factor context where available because share quantity alone may be insufficient. |
| Short-sale and cover-short holdings representation is not established by public native Axys/APX examples. | Unknown / Medium concept | The 2026-07-07 `ss`/`cs` research confirms the code meanings in APX integration mapping evidence, but did not locate native holdings rows. A synthetic demo may use negative quantity and negative market value if disclosed as assumptions; production treatment remains site-specific until proven by holdings or position extracts. |
| Zero-quantity transactions, missing transaction prices, and missing quarter-end prices are issues to resolve during conversion/reconciliation. | Verified as Morningstar guidance | These issues can affect holdings/report matching but are not specific to an Axys holdings table. |

## 4. APX Holdings Research

### 4.1 APX product context

| Statement | Classification | Evidence / Notes |
|---|---:|---|
| APX is described by SS&C as an integrated portfolio and client management solution. | Verified | SS&C APX product page / product brief. |
| APX 20.1 is described by SS&C as a core accounting, reporting, and performance measurement application. | Verified | SS&C 1H2020 Advent Product Updates press release. |
| In 2015, SS&C announced APX enhancements including improvements to position reconciliation, dividend processing, and cost basis handling. | Verified | SS&C / PRNewswire release. |
| APX has a SQL layer from which current-date holdings can be read in the WealthTechs AIA workflow. | Verified for AIA workflow | WealthTechs APX AIA guide says current date data can be read from APX SQL. Do not infer table names. |

### 4.2 APX Portfolio Appraisal and reports

| Statement | Classification | Evidence / Notes |
|---|---:|---|
| APX has a Portfolio Appraisal report. | Medium Confidence | Search result for the Advent Portfolio Exchange Reports Guide says “The Portfolio Appraisal shows your clients all holdings in an account by individual tax lot or position.” The PDF could not be fully opened through the browser tool; obtain `REP_APX.pdf` for verification before promoting beyond Chapter 06's current caveat. |
| APX Portfolio Appraisal can show holdings by individual tax lot or position. | Medium Confidence | Same APX reports guide search snippet. Needs full source capture for Verified classification. |
| Public APX sample client reports may include a `PORTFOLIO APPRAISAL` section and report parameters including `STY:APX`. | Medium Confidence | Public client-report PDFs appear to show APX-source report output, but they are downstream presentations, not vendor docs. Treat only as examples if cited and inspected. |

### 4.3 APX import/export, positions, and lots

| Statement | Classification | Evidence / Notes |
|---|---:|---|
| APX Import/Export utility executable is `apxix.exe` in the Custodial Integrator APX workflow. | Verified | Custodial Integrator APX User Guide says it looks for `apxix.exe` and identifies it as the APX Import/Export utility. |
| Custodial Integrator APX requires an APX executable folder, APX output folder, and APX log folder. | Verified | Custodial Integrator APX guide lists these configuration fields. |
| The APX output folder is used to transfer data to and from APX and should not contain spaces in its folder name. | Verified | Custodial Integrator APX guide. |
| Custodial Integrator APX displays APX Import/Export utility log files after executing the utility. | Verified | Custodial Integrator APX guide. |
| APX configuration includes a Portfolio group name containing portfolios referenced in the integration workflow. | Verified | Custodial Integrator APX guide. |
| APX configuration includes a Trade Blotter name into which transactions are imported; the blotter must be created in APX. | Verified | Custodial Integrator APX guide. |
| APX configuration includes a Position Blotter name into which positions are imported for reconciliation; the blotter must be created in APX. | Verified | Custodial Integrator APX guide. |
| APX configuration may include a Lot Blotter name into which position lots are imported for reconciliation; this field is present if position lots are enabled for the firm. | Verified | Custodial Integrator APX guide. |
| Custodial Integrator can include positions for export, retrieving positions for the prior business day and importing them to the named Position Blotter. | Verified | Custodial Integrator APX guide. |
| If position lots are enabled, position lots are not imported by default; a user can include lots for the next import by selecting `Include position lots`. | Verified | Custodial Integrator APX guide. |
| Advanced export options for positions also apply to position lots when lots are enabled and imported. | Verified | Custodial Integrator APX guide. |
| Custodial Integrator has options to exclude positions for stale accounts and failed accounts. | Verified | Custodial Integrator APX guide. |
| Custodial Integrator can download security prices for positions for the prior business day; it will not overwrite price records for that business day if records already contain a price. | Verified | Custodial Integrator APX guide. |
| Custodial Integrator maintains a copy of APX Security Information and Security Type Information for generating positions, prices, and transactions for APX import. | Verified | Custodial Integrator APX guide. |
| Portfolio code translation includes an `APX Portfolio Code` column identifying the portfolio in APX to which data is delivered. | Verified | Custodial Integrator APX guide. |
| Missing-prices output file field `Symbol` may be any APX security symbol defined in the security master; `Type` may be any APX security type defined in the security master. | Verified | Custodial Integrator APX guide. |

### 4.4 APX AIA holdings extraction and reconciliation workflow

| Statement | Classification | Evidence / Notes |
|---|---:|---|
| In the WealthTechs APX AIA workflow, a custodian master position file lists current holdings of each account with market value and quantity. | Verified | APX AIA guide Additional Use Cases section. |
| The master position file is used to create POS files in Advent. | Verified for AIA workflow | APX AIA guide. |
| The AIA Reconciliation report compares calculated positions in Advent versus downloaded custodian positions and displays positions not equal between custodian and Advent. | Verified for AIA workflow | APX AIA guide. |
| For NBIN interlisted duplicate handling, AIA must read APX holdings as of the data-file date and remove unnecessary duplicate holdings and prices from `.pos` and `.pri` source files. | Verified for AIA workflow | APX AIA guide. |
| Current-date APX data can be read from APX SQL in the AIA workflow. | Verified for AIA workflow | APX AIA guide. |
| Historical APX loading in the AIA workflow requires a report to calculate holdings; WealthTechs provides `CDIhold.rep`. | Verified for AIA workflow | APX AIA guide. |
| `CDIhold.rep` is added to the custom report menu in APX for this workflow. | Verified for AIA workflow | APX AIA guide. |
| The custom label `$pathCDI` is added in `APX > Admin > Global Settings > Configurations` and mapped to a network path in this workflow. | Verified for AIA workflow | APX AIA guide. |
| The AIA `Holdings Extract Folder (h.CDI)` advanced setting is mapped to the same network path in this workflow. | Verified for AIA workflow | APX AIA guide. |
| The holdings extract group is typically `cdirecon` in this workflow. | Verified as typical in source | APX AIA guide says “typically cdirecon.” Do not treat as mandatory. |

## 5. IMEX / Import-Export Research

The available source material does not provide a formal Axys IMEX object dictionary for holdings. It does, however, identify several import/export and report-mediated flows that should be captured carefully.

| Area | Axys | APX | Confidence |
|---|---|---|---:|
| Formal IMEX holdings object names | Unknown. No reliable object names for holdings were verified. | Unknown. No reliable object names for holdings were verified. | Unknown |
| Import/export executable | Unknown from available Axys material. | `apxix.exe` is used by Custodial Integrator as APX Import/Export utility. | APX Verified |
| Holdings as import artifact | POS files can be created from custodian position files in AIA workflows. | POS files can be created from custodian position files in AIA workflows; APX Position Blotter receives positions for reconciliation in Custodial Integrator. | Verified for cited workflows |
| Price import artifact | `.pri` appears as Axys security price file in Morningstar conversion guide; `.pri` source files appear in AIA duplicate-handling workflow. | `.pri` source files appear in AIA duplicate-handling workflow; Custodial Integrator can export security prices to APX. | Verified for cited workflows |
| Historical holdings export | Historical loading requires a report to calculate holdings in AIA Axys workflow. | Historical loading requires a report to calculate holdings in AIA APX workflow. | Verified for cited workflow |
| Current-date extraction | Current-date data files can be read from Axys in AIA workflow. | Current-date data can be read from APX SQL in AIA workflow. | Verified for cited workflow |

### IMEX Unknowns to Resolve Before Strengthening Chapter 06

| Question | Status |
|---|---:|
| What are the canonical Axys IMEX object names for current positions / holdings, if any? | Unknown |
| What are the canonical APX IMEX object names for current positions / holdings, if any? | Unknown |
| Are holdings exported from stored position snapshots, report calculations, or both? | Unknown |
| Are lot-level holdings available through IMEX in Axys? | Unknown |
| Are lot-level holdings available through APX IMEX outside Custodial Integrator? | Unknown |
| What are the required fields for a standard holdings import/export record? | Unknown |
| What date semantics are used: trade date, settlement date, position date, close date, or report as-of date? | Unknown |

## 6. REP / Report Research

| Report / file / variable | System | Description | Confidence |
|---|---|---|---:|
| `Portfolio Appraisal` | Axys | Holdings/assets point-in-time report; can be generated in Report Writer. | Verified |
| `Portfolio Appraisal` | APX | APX reports guide snippet says it shows holdings by tax lot or position. | Medium Confidence |
| `CDIhold.rep` | Axys | WealthTechs-provided report for historical holdings calculation in AIA / NBIN duplicate handling workflow. | Verified for workflow |
| `CDIhold.rep` | APX | WealthTechs-provided report for historical holdings calculation in AIA / NBIN duplicate handling workflow. | Verified for workflow |
| `aman.rep` | Axys | Assets Under Management report file copied in CSSI AUM-by-sector example. | Verified for example |
| `_aumsect.rep` | Axys | User-created copy of `aman.rep` in CSSI AUM-by-sector example. | Verified for example |
| `Reconciliation report` | Axys | Used by Morningstar conversion process to compare Axys to custodian records as of last transaction date. | Verified |
| WealthTechs Reconciliation reports | Axys/APX | AIA says WealthTechs has two reconciliation reports; one displays on Axys Report screen and one exports directly to Excel. | Verified for APX guide language; likely typo says Axys screen even in APX guide. |
| `$askport` | Axys REP / Report Writer | Header variable used in CSSI example for CLI code entered at runtime. | Verified for example |
| `Portfolio Code` | Axys Report Writer column | Available column in Portfolio Appraisal. | Verified |
| `$:fileo` | Axys REP variable | Used in AUM report article to add portfolio code. | Verified as CSSI statement |
| `$:tfile` | Axys REP variable | Used to show transaction source CLI file in transaction summary. | Verified as CSSI statement |
| `$firmg` | Axys REP variable | Used as “Other” catch-all classification variable in CSSI AUM sector example. | Verified for example |
| `$pathCDI` | Axys/APX configuration label | Custom label mapped to a network path for AIA holdings extract folder workflow. | Verified for workflow |

## 7. Field Dictionary Candidates

These fields and labels are safe candidates to include in Chapter 06 only with appropriate source qualification.

| Field / Label | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---:|
| Quantity | Holding quantity displayed in CSSI Portfolio Appraisal sample. | Yes | Unknown | Unknown | Yes | Verified for sample |
| Security | Security name/description displayed in CSSI Portfolio Appraisal sample. | Yes | Unknown | Unknown | Yes | Verified for sample |
| Price | Price displayed in CSSI Portfolio Appraisal sample. | Yes | Unknown | Unknown | Yes | Verified for sample |
| Market Value | Market value displayed in CSSI Portfolio Appraisal sample. | Yes | Unknown | Unknown | Yes | Verified for sample |
| Pct Assets | Percent-of-assets displayed in CSSI Portfolio Appraisal sample. | Yes | Unknown | Unknown | Yes | Verified for sample |
| Yield | Yield displayed in CSSI Portfolio Appraisal sample. | Yes | Unknown | Unknown | Yes | Verified for sample |
| Portfolio Code | Owner portfolio code available as an Axys Portfolio Appraisal column in Report Writer. | Yes | Unknown | Unknown | Yes | Verified |
| APX Portfolio Code | Portfolio-code translation field identifying the portfolio in APX for Custodial Integrator delivery. | No | Yes | Yes, in CI workflow | No | Verified for CI workflow |
| Symbol | Missing-prices output field; may be an APX security symbol defined in security master. | Unknown | Yes | Yes, in CI missing-prices output | No | Verified for CI workflow |
| Type | Missing-prices output field; may be an APX security type defined in security master. | Unknown | Yes | Yes, in CI missing-prices output | No | Verified for CI workflow |
| Name | Missing-prices output field: name of security or position with no price. | Unknown | Yes | Yes, in CI missing-prices output | No | Verified for CI workflow |
| WP Account | WebPortfolio account nickname in CI output. | No | External-to-APX | Yes, in CI missing-prices output | No | Verified for CI workflow |
| Institution | Financial institution name in CI output. | No | External-to-APX | Yes, in CI missing-prices output | No | Verified for CI workflow |
| Position Blotter name | APX object into which Custodial Integrator imports positions for reconciliation. | Unknown | Yes | Yes | No | Verified for CI workflow |
| Lot Blotter name | APX object into which Custodial Integrator imports position lots when lots enabled. | Unknown | Yes | Yes | No | Verified for CI workflow |
| Trade Blotter name | APX object into which Custodial Integrator imports transactions. | Unknown | Yes | Yes | No | Verified for CI workflow |
| Holdings Extract Folder (h.CDI) | AIA setting mapped to network folder containing holdings extract output. | Yes | Yes | Workflow setting | Report output destination | Verified for AIA workflow |

## 8. Examples

### 8.1 Axys: group Portfolio Appraisal with owner portfolio code

Source-backed behavior:

1. Create a new Axys Report Writer report: `File -> New -> Portfolio Appraisal`.
2. Optionally add `$askport` to the report header to display the CLI code entered in the run dialog.
3. Use `Define -> Columns` and add `Portfolio Code`.
4. Use `Define -> Options` and select `Management Mode`.
5. Run the report for an unconsolidated group.
6. In the CSSI example, the report produces one combined Portfolio Appraisal for the group and the `Portfolio Code` column identifies which CLI file owns each holding.

Classification: Verified for the CSSI example. Do not assert that this is the only or default way to identify owner portfolio in every Axys holdings report.

### 8.2 Axys/APX: AIA historical holdings extract for interlisted duplicate handling

Source-backed behavior:

1. NBIN source files may contain duplicate price and position data for interlisted securities.
2. AIA reads APX holdings as of the data-file date and removes unnecessary duplicate holdings and prices from `.pos` and `.pri` source files.
3. In Axys, current-date data files can be read from Axys; historical data loading requires a report to calculate holdings.
4. In APX, current-date data can be read from APX SQL; historical data loading requires a report to calculate holdings.
5. WealthTechs supplies `CDIhold.rep` for this purpose.
6. The report is added to the custom report menu and `$pathCDI` is mapped to a network path.
7. AIA Advanced settings map `Holdings Extract Folder (h.CDI)` to that path; the holdings extract group is typically `cdirecon`.

Classification: Verified for the WealthTechs AIA workflow.

### 8.3 APX: Custodial Integrator position import for reconciliation

Source-backed behavior:

1. Configure the APX executable folder containing `apxix.exe`.
2. Configure an APX output folder and APX log folder.
3. Configure a Portfolio group name.
4. Configure a Position Blotter name; the blotter must be created in APX.
5. If position lots are enabled, configure a Lot Blotter name; the blotter must be created in APX.
6. Enable `Include positions for export` to retrieve prior-business-day positions and import them into the named Position Blotter.
7. If lots are enabled, lots are not imported by default; use `Include position lots` for the next import.

Classification: Verified for Custodial Integrator APX workflow.

## 9. Known Issues / Quirks

| Quirk | System | Classification | Notes |
|---|---|---:|---|
| Consolidated vs unconsolidated group settings change Portfolio Appraisal output. | Axys | Verified | A consolidated group gives a single appraisal; an unconsolidated group normally gives several appraisals, unless customized Management Mode approach is used. |
| Owner portfolio code in a combined group report is not necessarily shown by default; it can be added through Report Writer. | Axys | Verified for CSSI example | Add `Portfolio Code` column. |
| Historical holdings extraction may require running a report, even if current-date data can be read directly. | Axys/APX | Verified for AIA workflow | Axys: current date files; APX: current date SQL; historical: `CDIhold.rep`. |
| Interlisted securities may create duplicate position and price data in NBIN source files. | Axys/APX | Verified for AIA workflow | AIA handles by reading holdings and removing duplicate holdings/prices from `.pos` and `.pri`. |
| APX output folder in Custodial Integrator should not contain spaces. | APX | Verified | CI APX guide states this. |
| APX position lots are optional/firm-dependent in Custodial Integrator. | APX | Verified | Lot Blotter field present if position lots enabled. |
| Custodial Integrator does not import lots by default even when lots are enabled. | APX | Verified | User must select `Include position lots` for next import. |
| APX position import can exclude stale or failed accounts. | APX | Verified | CI advanced position export options. |
| Custodial Integrator will not overwrite same-day APX security price records that already contain a price. | APX | Verified | CI guide states this. |
| Principal paydown handling can cause converted holdings not to match Axys performance reporting results. | Axys conversion | Verified | Morningstar guide documents methodology differences and zero-quantity transaction limitations. |
| `pd` principal-paydown classification should not rely on holdings quantity alone. | Axys/APX performance comparison | Medium-High for conservative audit treatment | The 2026-07-07 `pd` Modified Dietz research recommends preserving principal amount, factor, amortization, or equivalent principal-balance evidence when available. |
| New uploads/copies of old reports can have recent file dates but old content. | Repository process | High Confidence | Blueprint requires preserving evidence and source freshness; final author should inspect source content, not only metadata. |

## 10. Version Differences and Release Notes

| Version / Date | System | Evidence | Holdings relevance | Confidence |
|---|---|---|---|---:|
| 2015 second major product upgrades | APX / Axys | SS&C / PRNewswire release | APX enhancements included improvements to position reconciliation, dividend processing, and cost basis handling. Axys received UI, CRM2 support, and portfolio template upgrades. | Verified |
| 20.1 release / 2020 | APX | SS&C product updates | APX described as core accounting, reporting, and performance measurement application; fixed income improvements and new UI noted. | Verified |
| AIA User Manual version 3.1.0 / 2022 | Axys/APX integration | WealthTechs AIA manuals | Documents current vs historical holdings extraction behavior and `CDIhold.rep` workflow. | Verified for AIA workflow |
| Morningstar conversion guide Version 2.0 | Axys conversion | Morningstar PDF | Documents files required for conversion and principal paydown limitation. | Verified |

## 11. References

### Source references used

1. AXYS / APX Reference Blueprint, Version 2.0, uploaded by user. Governing repository specification.
2. SS&C Advent Axys product page. Establishes Axys as a portfolio reporting/accounting solution with predefined reports and customization.
3. SS&C Advent APX product page / product brief. Establishes APX as integrated portfolio/client management.
4. SS&C 1H2020 Advent Product Updates. APX described as core accounting, reporting, and performance measurement application; release highlights.
5. SS&C / PRNewswire 2015 Advent Product Updates. APX improvements included position reconciliation, dividend processing, and cost basis handling.
6. WealthTechs, `AIA User Manual for Axys Users`, version 3.1.0, 2022. Used for POS files, reconciliation behavior, current vs historical holdings extraction, `CDIhold.rep`, `.pos`/`.pri`, `$pathCDI`, and `h.CDI` workflow.
7. WealthTechs, `AIA User Manual for APX Users`, version 3.1.0, 2022. Used for APX SQL current-date holdings, historical report-calculated holdings, `CDIhold.rep`, `.pos`/`.pri`, and AIA settings.
8. ByAllAccounts / Morningstar, `Custodial Integrator User Guide` for APX. Used for `apxix.exe`, APX output/log folders, portfolio group, trade/position/lot blotters, position and lot import behavior, stale/failed account exclusions, security price behavior, APX security fields.
9. CSSI, `Accessing Cli PortCodes Within Groups`. Used for Axys Portfolio Appraisal group behavior, Report Writer steps, Portfolio Code column, Management Mode, `$askport`, `$:fileo`, `$:tfile`, and sample columns.
10. CSSI, `How To Create an Assets Under Management By Sector Report`. Used for `\axys3\rep`, `aman.rep`, `_aumsect.rep`, not modifying original reports, and `$firmg` example.
11. Morningstar, `Converting Your Advent Axys Database into Morningstar Office`, Version 2.0. Used for Axys conversion files, Reconciliation report, principal paydown / zero-quantity limitation, and conversion issue categories.
12. Chase Investment Counsel Code of Ethics filing. Used as evidence that an Axys Portfolio Appraisal Report can be used in lieu of a separate holdings report for specified accounts.
13. Advent Portfolio Exchange Reports Guide (`REP_APX.pdf`) search result only. Used only as Medium Confidence until the PDF itself is supplied or opened successfully.

## 12. Unknowns and Requested Source Material

The following should remain Unknown in Chapter 06 unless additional documentation, sample exports, or production observations are supplied.

| Unknown | Why it matters | Best additional source |
|---|---|---|
| Canonical Axys holdings IMEX object name(s). | Needed for precise IMEX section. | Axys IMEX manual / sample holdings export / macro. |
| Canonical APX holdings IMEX object name(s). | Needed for precise IMEX section. | APX Import/Export manual / `apxix.exe` documentation / sample export. |
| Axys holdings storage mechanics. | Needed to distinguish stored positions from report-calculated holdings. | Vendor technical manual or production file layout examples. |
| APX SQL holdings table/view names. | Needed for APX data model. | APX database schema / official reporting data dictionary / sample query. |
| Lot-level holdings support in Axys. | Needed for field dictionary and tax-lot examples. | Axys report guide or sample Portfolio Appraisal tax-lot report. |
| Lot-level holdings support in APX Portfolio Appraisal. | Search result suggests tax lot/position, but needs full source verification. | APX Reports Guide PDF. |
| Standard Portfolio Appraisal default columns for Axys versions. | Sample columns are from a CSSI report example and may be customized. | Official Axys reports guide / unmodified sample report. |
| Standard Portfolio Appraisal default columns for APX versions. | Needed to avoid assuming Axys columns match APX. | Official APX reports guide / unmodified sample report. |
| Date basis for holdings reports: trade-date vs settlement-date vs accounting-date. | Critical for reconciliation and performance. | Vendor report options manual / sample reports with settings. |
| Treatment of cash, short positions, accrued income, unsettled trades, FX, and multicurrency holdings. | Critical for real-world holdings chapter. | Report guides and sample outputs. |
| Security identifier hierarchy used in holdings reports. | Needed for data dictionary. | Security master docs and report output samples. |
| Whether `CDIhold.rep` is generic or WealthTechs-specific. | Avoid documenting third-party workflow as standard Axys/APX. | Report source or vendor notes. |
| Whether REP syntax and variables differ between Axys and APX. | Needed for REP chapter cross-reference. | Axys/APX Replang manuals and sample `.rep` files. |

## 13. Recommended Next Inputs Before Strengthening Chapter 06

To strengthen the reader-facing holdings chapter, collect any of the following:

1. Official Axys reports guide section for Portfolio Appraisal.
2. Official APX reports guide (`REP_APX.pdf`) section for Portfolio Appraisal.
3. One unmodified Axys Portfolio Appraisal report output for a simple portfolio.
4. One unmodified APX Portfolio Appraisal report output for a simple portfolio.
5. One lot-level holdings report sample, if used by either system.
6. Axys/APX IMEX or Import/Export manual pages for positions/holdings.
7. Sample `.rep` files: `CDIhold.rep`, `aman.rep`, and any standard Portfolio Appraisal report definition.
8. Sample current-date and historical holdings extracts from Axys and APX.
9. Any client-specific notes documenting how holdings are calculated relative to transactions, prices, splits, and accruals.

Until those are available, Chapter 06 should heavily distinguish verified report behavior from unknown internal data-storage behavior.

## 14. Deep IMEX Addendum Incorporated 2026-06-30

Source: `axys_imex_deep_research.md`.

Additional holdings/position points:

| Topic | Addendum | Confidence |
|---|---|---:|
| `ptopost.trn` | CI writes positions to `\CI\exported\ptopost.trn` in CSV format before Position Post. | Verified for CI workflow |
| `.pos` files | Position Post can create replacement `.pos` files in the Axys position folder for configured portfolios. | Verified for CI workflow |
| Position lots | When lots are enabled and available, position output may contain lot-specific data and use `imexPositionLots.log`. | Verified for CI workflow |
| Cash / money-market fallback | For cash or money-market positions, or where no lot exists, CI may output position data instead of lot data. | Verified for CI workflow |
| Candidate position fields | Live discovery should inspect portfolio/account, as-of date, symbol/type/name, quantity/units, price, market value, accrued income, cost/book/tax cost, local/base currency, FX rate, stale flag, and custodian/account context. | Discovery guidance |
| Candidate lot fields | Live discovery should inspect lot identifier, open/acquisition dates, quantity, cost, market value, tax-cost fields, currency, and source row lineage. | Discovery guidance |
| Boundary | `ptopost.trn` and `.pos` evidence is integration-level position evidence; it does not establish native holdings storage mechanics or standard holdings report columns. | Unknown / boundary |

## Deep Research Update Incorporated 2026-07-02

The July 2026 addendum strengthens holdings evidence in two areas. First,
APX/Advent Portfolio Appraisal can be treated as a verified report concept for
showing holdings by individual tax lot or position, with report-output concepts
such as quantity, cost, market value, percent of portfolio, yield, and
unrealized gain/loss. These remain report-output labels, not APX database,
public-view, or IMEX field names.

Second, Axys CI position behavior is more specific: CI writes positions to
`\CI\exported\ptopost.trn` in CSV format; Position Post can create replacement
`.pos` files for configured portfolios; lot-enabled output can contain lot data;
cash/money-market or no-lot positions fall back to position data; and
`imexPositionLots.log` may appear instead of `imexPositions.log`. This narrows
one integration workflow but does not prove native Axys holdings storage
mechanics. Canonical Axys/APX holdings IMEX object names, APX SQL/public-view
names, complete Portfolio Appraisal columns, holdings date semantics, and
cash/short/accrual/FX treatment remain Unknown.
