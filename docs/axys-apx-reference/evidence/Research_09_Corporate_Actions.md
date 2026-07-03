# Research Notes: Corporate Actions

**Repository area:** `docs/axys-apx-reference/evidence/`  
**Target chapter:** `docs/axys-apx-reference/reference/Chapter_09_Corporate_Actions.md`  
**Systems:** SS&C Advent Axys; SS&C Advent Portfolio Exchange (APX)  
**Prepared:** 2026-06-29  
**Governing specification:** `AXYS_APX_REFERENCE_BLUEPRINT.md`, Version 2.0

---

## 1. Research Scope and Evidence Policy

This file collects implementation-oriented research for a future technical reference chapter on corporate actions in Axys and APX.

The governing repository standard requires that important statements be classified as:

- **Verified** — directly supported by a cited source, uploaded sample, vendor documentation, or observed file/report evidence.
- **High Confidence** — strongly supported by multiple indirect sources or longstanding implementation practice, but not directly verified in supplied source material.
- **Medium Confidence** — plausible and consistent with public evidence or known Advent usage, but requires confirmation before use in a production reference chapter.
- **Unknown** — not verified from the supplied material or available public evidence.

This research intentionally preserves unknowns. It does not invent transaction codes, IMEX object names, REP report names, table names, file layouts, or processing details.

---

## 2. Source Inventory

| Source | Type | Relevance | Notes | Confidence Use |
|---|---:|---|---|---|
| `AXYS_APX_REFERENCE_BLUEPRINT.md` | Uploaded repository specification | Governs format, confidence labels, factual discipline, Axys/APX separation | Used as editorial authority only | Verified for repository process |
| SS&C Advent, “Overcome the Challenges of Corporate Actions Processing with Axys” | Vendor product brief PDF | ACA integration with Axys, active-holdings script, reports, Automation Results email, Trade Blotter processing, simple/complex event review | Public vendor brief | Verified for ACA/Axys product claims |
| SS&C Advent, “Brief: Advent Corporate Actions for APX” | Vendor web page / product brief | APX integration with Advent Corporate Actions (ACA), processing flow, Trade Blotter, Reorg Utility | Public vendor page | Verified for ACA/APX product claims |
| SS&C Advent, “What is SS&C Advent Corporate Actions?” | Vendor web page | General ACA positioning; dashboard/calendar/reporting improvements | Public vendor page | Verified for ACA product positioning only |
| AdventGuru, “Demystifying Portfolio Accounting Systems Integration Post-Merger/Acquisition” / Data Conversion category | Consultant article | Mentions Axys/APX merge workflows; exported CSV copies of `split.inf`; import into merged Advent environment | Public consultant source, not official vendor documentation | Medium to High Confidence depending on claim |
| FinFolio, “Advent Axys, Moxy & APX Conversions” | Vendor conversion page | Lists Advent source files used in conversion; explicitly mentions split transactions “blown out” from `SPLIT.INF` | Third-party converter | Medium Confidence for conversion observations |
| Morningstar Office, “Advent Axys Database Conversion” PDF | Third-party conversion guide | Lists Axys export files including `.cli`, `sec.inf`, `split.inf`, `.pri`, `type.inf`; discusses export mechanics and transaction conversion limitations | Third-party conversion guide; includes procedural Axys export steps | High Confidence for Axys file existence and conversion requirements |
| AdventGuru, “Building Advent APX Data Pipeline Integration with REST API” | Consultant article | Notes APX REST API may not cover all data; IMEX/Replang/public views/stored procedures/SSRS may still be needed | Public consultant source | Medium Confidence for APX integration strategy |
| SS&C Advent corporate actions processing white paper | Vendor white paper | General corporate action concepts and operational challenges | Not Axys/APX-specific | Background only |

---

## 3. Executive Findings

| Finding | Axys | APX | Classification | Evidence / Notes |
|---|---|---|---:|---|
| Corporate actions are a distinct operational domain for Advent users; SS&C offers Advent Corporate Actions (ACA) as a separate corporate actions solution. | ACA is marketed as integrated with Axys | ACA is marketed specifically for APX | Verified | SS&C product briefs describe ACA workflows for both Axys and APX. |
| ACA for Axys can receive active holdings by daily script, provide reports and Automation Results email, and process simple/mandatory events to the Trade Blotter. | Yes, per vendor brief | N/A | Verified | Vendor-described Axys workflow; exact fields and final posting lifecycle Unknown. |
| ACA for APX can send holdings from APX to the ACA Server, cross-reference securities to an action database, review/download actions, run the APX Reorg Utility, and post transactions to the APX Trade Blotter. | Not stated | Yes, per vendor brief | Verified | Vendor-described APX workflow. |
| Axys has a `split.inf` file used for security splits. | Yes | Unknown whether native APX stores an equivalent physical file | High Confidence | Third-party conversion and consultant sources identify `split.inf` / `SPLIT.INF` as Advent Axys security splits file. |
| Axys conversion/extract packages commonly include `.cli`, `sec.inf`, `split.inf`, `.pri`, and `type.inf`. | Yes | Unknown | High Confidence | Morningstar conversion guide lists these files as Advent Axys database files. |
| Split transactions may be derived or “blown out” from `SPLIT.INF` during conversion. | Yes, in third-party conversion process | Listed by converter for Advent Axys/Moxy/APX conversions, but source-system distinction is unclear | Medium Confidence | FinFolio page states split transactions are “blown out” from `SPLIT.INF`. |
| APX provides a Reorg Utility used in the ACA workflow. | Not identified in public Axys brief | Yes | Verified for APX only | SS&C ACA/APX brief says the Reorg Utility runs and transactions post to APX Trade Blotter. |
| APX Trade Blotter is the destination for ACA-generated reorganization transactions. | Not applicable from public source | Yes | Verified | SS&C ACA/APX brief. |
| Exact APX database tables for corporate actions are public-source Unknown. | Unknown | Unknown | Unknown | No supplied APX schema or vendor technical docs provided. |
| Exact IMEX object names and fields for corporate actions are public-source Unknown. | Unknown | Unknown | Unknown | No IMEX dictionary, export samples, or mapping files supplied. |
| Exact REP reports used for corporate action review/audit are public-source Unknown. | Unknown | Unknown | Unknown | No REP report catalog or sample reports supplied. |

---

## 4. Corporate Action Event Types Relevant to Axys/APX Research

The following event taxonomy is included only as a research checklist. It should not be read as evidence that Axys/APX has one canonical corporate action object covering all event types.

| Event Type | Typical Accounting Impact | Axys Evidence | APX Evidence | Confidence | Notes |
|---|---|---|---|---:|---|
| Cash dividend | Cash income transaction; possible receivable/accrual depending workflow | Transaction treatment likely through `.cli` activity, but exact codes Unknown | Transaction treatment likely through APX transaction/trade blotter workflow, exact codes Unknown | Medium Confidence | Transaction codes must not be invented. |
| Reinvested dividend | Distribution plus purchase/reinvestment transaction pair may appear in conversion | Morningstar notes distribution reinvest transaction types may translate into transaction pairs in conversion | Unknown | Verified for conversion behavior only | Not evidence of all Axys internal behavior. |
| Stock split | Share quantity adjustment; price/factor handling; possible separate split file | `split.inf` identified as securities splits file | ACA/Reorg Utility supports reorg activities generally; exact split storage Unknown | High Confidence for Axys; Unknown for APX storage | `split.inf` is the strongest Axys-specific corporate action evidence found. |
| Reverse split | Share quantity adjustment; price/factor handling | Likely represented in split data if supported by split factor conventions; exact fields Unknown | Unknown | Medium Confidence | Requires sample `split.inf` or IMEX export to confirm. |
| Stock dividend | Similar to split or share distribution depending system workflow | Unknown | Unknown | Unknown | Do not assume treatment. |
| Return of capital | Income/cost basis transaction | Unknown | Unknown | Unknown | Requires transaction code dictionary. |
| Cash exchange | Cash/security exchange event | Verified in ACA-for-Axys brief as simple/mandatory example | ACA/APX workflow likely supports simple events, but exact public example not found | Verified for Axys ACA coverage; Unknown for output | Exact transaction output Unknown. |
| Stock exchange | Security exchange event | Verified in ACA-for-Axys brief as simple/mandatory example | ACA/APX workflow likely supports simple events, but exact public example not found | Verified for Axys ACA coverage; Unknown for output | Exact transaction output Unknown. |
| Merger | Security reorganization; close old/open new security; possible cash-in-lieu | ACA-for-Axys brief covers taxable, non-taxable, combination, and option mergers | ACA/APX workflow handles reorg activities generally | Verified for Axys ACA coverage and APX workflow; Unknown for exact postings | Need Trade Blotter/Reorg Utility documentation and samples. |
| Spin-off | Security reorganization; new security; basis allocation | ACA-for-Axys brief covers spin-offs | ACA/APX workflow likely supports complex events via review, but exact posting Unknown | Verified for Axys ACA coverage; APX Medium | ACA specialists help interpret complex events; specific transaction outputs Unknown. |
| Tender / exchange offer | Voluntary election; quantity/cash/security exchange | Unknown | ACA has alerts/review process for complex events, but exact support Unknown | Medium Confidence | Need ACA/APX user guide. |
| Symbol/ticker/name change | Security master update; possible cross-reference update | ACA-for-Axys brief covers name changes | ACA likely covers identifier events, but exact public APX example not found | Verified for Axys ACA coverage; APX Unknown | Need security master before/after samples. |
| CUSIP change | Security master update; possible security identifier history | Unknown | Unknown | Unknown | Need security master/APX schema. |
| Bond call / redemption | Principal/cash transaction; quantity reduction | Unknown | Unknown | Unknown | Could be transaction-driven rather than corporate-action-file-driven. |
| Bankruptcy | Write-off, reorganization, or security status event | ACA-for-Axys brief covers bankruptcies | ACA likely covers complex events, but exact public APX example not found | Verified for Axys ACA coverage; APX Unknown | Exact accounting output Unknown. |

---

## 5. Axys Research

### 5.1 Axys Storage and Files

| Item | Description | Classification | Evidence / Research Note |
|---|---|---:|---|
| `.cli` files | Client/account files used in Axys conversion; contain transactions in conversion context | High Confidence | Morningstar lists `.cli` among required Advent Axys database files and FinFolio says clients/accounts/transactions are created from CLI files. |
| `sec.inf` | Axys securities file | High Confidence | Morningstar lists `sec.inf` as “securities file.” |
| `split.inf` / `SPLIT.INF` | Axys securities splits file | High Confidence | Morningstar lists `split.inf` as “securities splits file.” AdventGuru describes exported CSV copies of firms’ `split.inf` files. |
| `.pri` files | Axys security prices file(s) | High Confidence | Morningstar lists `.pri` as “security prices file.” |
| `type.inf` | Axys security type file | High Confidence | Morningstar lists `type.inf` as “security type file.” |
| `bond.inf` | Mentioned in legacy file operations; purpose in corporate actions not established | Medium Confidence | Found in an old script snippet copying `SEC.INF` and `BOND.INF`; not enough for chapter claim beyond existence in some environments. |
| Dividend-specific `.inf` file | No verified Axys file found | Unknown | No supplied or public source confirmed an Axys `dividend.inf` or equivalent. |
| Merger/reorg-specific `.inf` file | No verified Axys file found | Unknown | No supplied or public source confirmed a separate Axys reorganization file. |

### 5.2 Axys Corporate Action Processing Model

| Topic | Research Finding | Classification | Notes |
|---|---|---:|---|
| Splits | Axys maintains split data in `split.inf` in at least some environments/versions. | High Confidence | Confirm with exported sample before documenting field layout. |
| ACA for Axys | SS&C Advent marketed ACA as integrated with Axys, receiving active holdings by daily script, providing reports and Automation Results email, and processing simple/mandatory events to the Trade Blotter. | Verified for vendor workflow claim | Exact fields, transaction rows, and final posting lifecycle Unknown. |
| Simple vs complex ACA events | Axys ACA brief says simple/mandatory events such as cash or stock exchanges can automatically process, while non-simple/mandatory option events such as mergers with options require review before processing. | Verified for vendor workflow claim | Downstream status fields Unknown. |
| ACA coverage | Axys ACA brief covers U.S. and non-U.S. equities, fixed income, taxable/non-taxable/combination mergers, name changes, spin-offs, bankruptcies, and many others. | Verified for vendor coverage claim | Exact accounting output Unknown. |
| Split conversion | Third-party conversion workflows may convert `SPLIT.INF` into explicit split transactions. | Medium Confidence | This describes converter behavior, not necessarily native Axys reports. |
| Cash dividends | Likely represented as transactions in `.cli` files rather than a central corporate action master. | Medium Confidence | Requires Axys transaction code dictionary and sample `.cli` export. |
| Reinvested dividends | Third-party Morningstar conversion guide says Advent Axys distribution reinvest transaction types may translate into two transaction pairs, normally a Buy and corresponding Distribution. | Verified for Morningstar conversion behavior | Do not generalize beyond conversion without Axys documentation. |
| Price adjustment after split | Axys has both `.pri` price files and `split.inf` split data, but exact adjustment mechanics are Unknown. | Unknown | Need Axys pricing/split documentation or before/after samples. |
| Performance recalculation impact | Corporate action corrections may affect holdings, prices, income, and performance, but exact Axys recalculation behavior by report is Unknown. | Unknown | Should be covered by Performance chapter only with evidence. |
| Security identifier changes | Unknown whether Axys records identifier history versus overwrites security master fields. | Unknown | Need `sec.inf` samples before/after CUSIP/ticker changes. |
| Manual entry vs automated feed | ACA supports automated workflow paths, but exact manual-entry interaction and final posting behavior remain Unknown. | Unknown | Verified at workflow level only. |

### 5.2.1 Axys ACA, `split.inf`, and Conversion Boundaries

`split.inf` and ACA are separate evidence surfaces. `split.inf` is an
Axys file artifact for split history in conversion and extract evidence.
ACA is a corporate-action workflow/product that can provide action data
and process events to the Axys Trade Blotter. Public sources do not
establish whether ACA writes `split.inf`, writes Trade Blotter
transactions only, writes both, or uses different paths by event type.

Conversion-materialized split transactions are a third surface. FinFolio
conversion evidence says split transactions can be "blown out" from
`SPLIT.INF`; that is target-system conversion behavior and should not be
treated as proof that native Axys stores split transactions in every
account `.cli` file.

### 5.3 Axys Field Dictionary Candidates

These are research placeholders, not confirmed field names.

| Field / Concept | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---|---|---|---|---:|
| Security identifier | Identifier tying split/corporate action to a security | Required conceptually | Required conceptually | Unknown | Unknown | Unknown |
| Split date / effective date | Date on which split applies | Likely present in `split.inf` | Unknown | Unknown | Unknown | Unknown |
| Split ratio / factor | Ratio or factor for old/new shares | Likely present in `split.inf` | Unknown | Unknown | Unknown | Unknown |
| Old security identifier | Security being reorganized | Unknown | Unknown | Unknown | Unknown | Unknown |
| New security identifier | Replacement / distributed security | Unknown | Unknown | Unknown | Unknown | Unknown |
| Cash-in-lieu amount | Cash from fractional shares | Unknown | Unknown | Unknown | Unknown | Unknown |
| Ex-date | Market date for dividend/split entitlement | Unknown | Unknown | Unknown | Unknown | Unknown |
| Record date | Holder-of-record date | Unknown | Unknown | Unknown | Unknown | Unknown |
| Pay date | Payment date | Unknown | Unknown | Unknown | Unknown | Unknown |
| Announcement date | Public action announcement date | Unknown | Unknown | Unknown | Unknown | Unknown |
| Transaction code | Posting code for generated transactions | Known concept in Axys transaction data | Known concept in APX transaction data | Unknown | Unknown | Unknown |

**Important:** The field names above are concepts only. They must not be used as Axys/APX field names until verified from IMEX, REP, exported files, or vendor dictionaries.

---

## 6. APX Research

### 6.1 APX and Advent Corporate Actions (ACA)

| ACA/APX Workflow Step | APX Behavior | Classification | Evidence / Research Note |
|---|---|---:|---|
| Send holdings to ACA | APX can send updated security holdings information to ACA Server. | Verified | SS&C ACA for APX brief. |
| Cross-reference securities | ACA cross-references APX securities to an action database and creates action records. | Verified | SS&C ACA for APX brief. |
| Automation summary | If using a script, ACA sends an Automation Results email summarizing transaction activity after script processing. | Verified | SS&C ACA for APX brief. |
| Review/download actions | Users can review all ACA transactions before download, or allow simple transactions to download automatically while reviewing complex events. | Verified | SS&C ACA for APX brief. |
| Reorg Utility | Downloaded reviewed actions to APX cause the APX Reorg Utility to run. | Verified | SS&C ACA for APX brief. |
| Trade Blotter | APX Reorg Utility-generated transactions post to APX Trade Blotter. | Verified | SS&C ACA for APX brief. |
| Alerts | ACA Alerts notify users of revisions and tips for processing complex APX events; alerts are emailed if the firm holds the highlighted security and can be viewed online. | Verified | SS&C ACA for APX brief. |
| ACA specialists | ACA specialists analyze tax opinions, compile reorg transactions, troubleshoot processing issues, and challenge data vendors. | Verified | SS&C ACA for APX brief. |

### 6.2 APX Native Corporate Action Storage

| Topic | Research Finding | Classification | Notes |
|---|---|---:|---|
| APX corporate action database tables | Not identified in supplied/public sources. | Unknown | Need APX schema, public views, stored procedures, or official technical documentation. |
| APX Reorg Utility input format | Not identified in supplied/public sources. | Unknown | Need ACA/APX user guide or reorg utility documentation. |
| APX Trade Blotter field mapping for ACA transactions | Not identified in supplied/public sources. | Unknown | Need sample ACA download, APX blotter export, or IMEX transaction export. |
| APX split storage | Not identified in supplied/public sources. | Unknown | Third-party pages mention `SPLIT.INF` in Advent conversions, but not enough to document APX native storage. |
| APX import/export via IMEX for corporate actions | IMEX may still be required for some APX data, but no corporate-action-specific object list was found. | Medium Confidence for integration strategy; Unknown for corporate action fields | Consultant source notes some APX API elements may be unavailable and may require IMEX/Replang/views/stored procedures/SSRS. |
| APX REST API as corporate action source | Not verified. | Unknown | Public consultant source says APX API can pull holdings and select performance; corporate action endpoints not verified. |

---

## 7. IMEX Research

### 7.1 Confirmed / Unconfirmed IMEX Objects

No supplied IMEX dictionary, sample IMEX exports, or command scripts were provided for corporate actions. Therefore, exact IMEX object names and field lists are Unknown.

| Candidate Data Need | Likely Source Type | Axys IMEX Status | APX IMEX Status | Confidence | Notes |
|---|---|---|---|---:|---|
| Transactions generated by dividends/reorgs | Transaction export | Unknown object/field names | Unknown object/field names | Unknown | Need IMEX transaction object definition and transaction code list. |
| Security master updates from corporate actions | Security export | Unknown | Unknown | Unknown | Need Security Master chapter and IMEX dictionary. |
| Split file / split factors | Split export/import or file export | Unknown IMEX object | Unknown | Unknown | Axys `split.inf` exists, but IMEX mapping not verified. |
| Price/factor adjustment after split | Price export | Unknown | Unknown | Unknown | Need pricing IMEX chapter linkage. |
| ACA-generated APX transactions | Trade blotter/transaction export | Not applicable | Unknown | Unknown | Need APX ACA sample output. |
| Holdings used for ACA eligibility | Holdings export | Not applicable | Unknown | Unknown | Vendor says APX sends holdings to ACA, but not via IMEX. |

### 7.2 Minimal IMEX Research Questions to Resolve

| Question | Why It Matters | Status |
|---|---|---:|
| Which IMEX object exports transactions in Axys and APX? | Needed to audit dividend/reorg postings. | Unknown |
| Are stock splits exported through transaction IMEX, security master IMEX, a split-specific IMEX object, or only through file export? | Needed to model split corrections. | Unknown |
| Does IMEX expose original ACA action IDs or only resulting APX transactions? | Needed for traceability from external ACA to APX postings. | Unknown |
| Does APX IMEX expose Trade Blotter rows before posting and posted transactions after posting? | Needed to distinguish pending vs final corporate actions. | Unknown |
| Are generated reorg transactions marked with source/system fields? | Needed to separate manual reorg entries from ACA-generated entries. | Unknown |
| What fields identify canceled/reversed/reposted corporate-action transactions? | Needed to audit historical return changes. | Unknown |

---

## 8. REP / Replang Research

No supplied REP source, report catalog, or sample output was provided for corporate actions. Public search did not identify exact corporate-action REP report names or fields.

| Report / REP Need | Axys | APX | Confidence | Notes |
|---|---|---|---:|---|
| Report listing stock splits for date range | Unknown | Unknown | Unknown | Need REP report inventory or sample. |
| Report listing dividend transactions | Unknown | Unknown | Unknown | Could be a transaction report rather than corporate action report. |
| Report listing security master changes | Unknown | Unknown | Unknown | Need REP/security master evidence. |
| Report listing APX Trade Blotter corporate action entries | Not applicable | Unknown | Unknown | Need APX report catalog. |
| ACA reports | Axys ACA brief says ACA provides reports | APX/ACA has “reports” per product positioning | Verified for existence of ACA reporting capability only | Exact report names/fields Unknown. |
| Automation Results email | Axys ACA brief says an email summarizes U.S. equity and ADR transaction activity affecting the firm | APX ACA workflow can send Automation Results email after script processing | Verified for workflow-level evidence | Exact fields Unknown. |
| Reconciliation report | Axys conversion guide requests a Reconciliation report as of last transaction date | Unknown | High Confidence for Axys conversion workflow | Not specific to corporate actions, but relevant for conversion/control. |

### REP Research Questions

| Question | Status |
|---|---:|
| Which standard REP reports display split history? | Unknown |
| Which standard REP reports display dividend/reorg transaction codes? | Unknown |
| Can REP read `split.inf` directly? | Unknown |
| Can REP report APX Trade Blotter transactions before posting? | Unknown |
| Does REP expose transaction source/origin fields for ACA-generated APX entries? | Unknown |
| Are there separate Axys and APX report names for corporate action review? | Unknown |

---

## 9. Data Model Research

### 9.1 Conceptual Entities

The following entities are useful for organizing research. They are not confirmed Axys/APX table names.

| Conceptual Entity | Purpose | Axys Evidence | APX Evidence | Confidence |
|---|---|---|---|---:|
| Security | Master record affected by symbol/CUSIP/name/type changes | `sec.inf` identified as securities file | APX security master exists conceptually, exact table Unknown | High Confidence for Axys file; Unknown for APX table |
| Price | Price records may need adjustment/validation around split dates | `.pri` identified as security prices file | APX price storage Unknown | High Confidence for Axys file; Unknown for APX table |
| Split | Split factor/history | `split.inf` identified as security splits file | APX split storage Unknown | High Confidence for Axys file; Unknown for APX |
| Transaction | Cash dividends, reinvestments, reorg postings, cash-in-lieu | `.cli` files used for transactions in conversion; Axys ACA can process simple/mandatory events to Trade Blotter | APX Trade Blotter and posted transactions likely relevant; exact storage Unknown | Verified for ACA workflow; exact fields Unknown |
| Holding | Used to determine action eligibility | Axys ACA receives active holdings via daily script | APX sends holdings to ACA Server | Verified for ACA workflow; exact interface fields Unknown |
| Corporate action instruction | Vendor/ACA event record and processing instruction | ACA-for-Axys workflow supports action/instruction concept; exact fields Unknown | ACA creates action records and transaction instructions, but exact APX data model Unknown | Verified at workflow level only |

### 9.2 Relationship Hypotheses Requiring Verification

| Hypothesis | Classification | Verification Needed |
|---|---:|---|
| Axys applies `split.inf` records to reports/holdings rather than storing split-generated transactions in each `.cli` file. | Medium Confidence | Compare `.cli` history and split-aware position reports before/after split; inspect Axys docs. |
| Some conversion tools “materialize” Axys split history into explicit transactions because their target systems need transaction-level splits. | Medium Confidence | Confirm with conversion specifications and sample conversion outputs. |
| Cash dividends in Axys/APX are represented as transactions rather than records in a central dividend master. | Medium Confidence | Need transaction code dictionary and sample exports. |
| APX ACA processing creates pending Trade Blotter transactions that must be posted/accepted before affecting accounting. | Medium Confidence | Vendor says actions post to Trade Blotter; exact lifecycle requires APX documentation. |
| ACA maintains event records separate from APX accounting records. | Medium Confidence | Vendor says ACA cross-references securities to an action database; need ACA data model. |
| Axys ACA processing creates Trade Blotter rows that can be distinguished from manual or custodian-imported rows. | Unknown | Axys Trade Blotter export after ACA processing or ACA report/download sample. |
| ACA action IDs or review statuses survive into Axys/APX posted accounting records. | Unknown | ACA reports/downloads plus Trade Blotter and posted transaction exports. |

---

## 10. Processing Behavior Research

### 10.1 Axys Processing Behavior

| Process | Known / Unknown | Classification | Notes |
|---|---|---:|---|
| Split entry/edit | Axys has `split.inf`; exact UI/workflow Unknown. | High Confidence for storage file; Unknown for process | Need Axys user manual or screen documentation. |
| Split impact on positions | Expected to affect share quantities in holdings/reporting; exact timing and calculation Unknown. | Medium Confidence | Do not document formula without sample. |
| Split impact on prices | Expected need to reconcile split factors and price history; exact Axys behavior Unknown. | Medium Confidence | Need pricing/split examples. |
| Dividend posting | Likely transaction-driven; exact transaction codes Unknown. | Medium Confidence | Need `.cli` sample and code dictionary. |
| Distribution reinvestment | Conversion may see reinvestment represented as Buy + Distribution pair. | Verified for Morningstar conversion only | Not enough for native reference. |
| ACA active-holdings script | ACA for Axys receives active holdings through a daily script. | Verified for vendor workflow claim | Exact file/interface and fields Unknown. |
| ACA reports and Automation Results email | ACA for Axys provides reports and an email summarizing U.S. equity and ADR transaction activity. | Verified for vendor workflow claim | Exact report/email fields Unknown. |
| ACA Trade Blotter processing | Simple/mandatory events can process to the Axys Trade Blotter; non-simple/mandatory option events require review before processing. | Verified for vendor workflow claim | Exact transaction rows, codes, and final posting lifecycle Unknown. |
| Correction/backdating | Corporate-action corrections can affect prior holdings/performance in general, but Axys mechanics Unknown. | Unknown | Need production observations or docs. |
| Reorg/merger | Unknown. | Unknown | Need examples. |

### 10.2 APX Processing Behavior

| Process | Known / Unknown | Classification | Notes |
|---|---|---:|---|
| ACA holdings upload | APX sends updated security holdings to ACA Server. | Verified | Vendor ACA/APX brief. |
| ACA action matching | ACA cross-references APX securities to action database. | Verified | Vendor ACA/APX brief. |
| User review | Users review all ACA transactions or auto-download simple transactions while reviewing complex events. | Verified | Vendor ACA/APX brief. |
| Download to APX | Reviewed actions are downloaded to APX. | Verified | Vendor ACA/APX brief. |
| Reorg Utility | APX Reorg Utility runs during ACA workflow. | Verified | Vendor ACA/APX brief. |
| Trade Blotter posting | Transactions automatically post to the APX Trade Blotter. | Verified | Vendor ACA/APX brief. |
| Final accounting post | Whether Trade Blotter records are automatically posted to final transaction history or require user approval is Unknown from public evidence. | Unknown | Need APX Trade Blotter/Reorg Utility docs. |
| Corrections/revisions | ACA Alerts notify of revisions and tips for complex events. | Verified | Exact correction workflow in APX Unknown. |

---

## 11. Examples

### 11.1 Example Research Scenario: Stock Split in Axys

**Scenario:** A portfolio holds 100 shares of XYZ. XYZ has a 2-for-1 split effective 2026-04-15.

| Required Data | Potential Axys Location | Status |
|---|---|---:|
| Security identifier for XYZ | `sec.inf` / transaction security reference | Unknown field name |
| Split effective date | `split.inf` | Unknown field name |
| Split factor / ratio | `split.inf` | Unknown field name |
| Pre/post split prices | `.pri` files | Unknown field name/layout |
| Account holdings before split | Derived from `.cli` transactions or report | Unknown extraction method |
| Reported post-split shares | Holdings/appraisal report | Unknown report name |

**Research conclusion:** Axys split history is evidenced by `split.inf`, but the field layout, calculation formula, and report behavior require sample files or vendor documentation.

### 11.2 Example Research Scenario: APX ACA Reorganization

**Scenario:** ACA identifies a merger affecting a security held in APX.

| Workflow Step | Evidence Status |
|---|---:|
| APX sends updated security holdings to ACA Server | Verified |
| ACA cross-references securities to action database | Verified |
| ACA creates action records / transactions | Verified at workflow level |
| User reviews transactions or auto-downloads simple transactions | Verified |
| Download reviewed actions to APX | Verified |
| APX Reorg Utility runs | Verified |
| Transactions post to APX Trade Blotter | Verified |
| Final transaction codes and fields | Unknown |
| Downstream impact on performance reports | Unknown |

---

## 12. Known Issues / Quirks

| Issue / Quirk | Axys | APX | Classification | Notes |
|---|---|---|---:|---|
| `split.inf` may need special handling during data conversion/merger. | Yes | Unknown | Medium Confidence | AdventGuru provides merge code for exported CSV copies of split files. |
| Split histories may be converted into explicit split transactions by third-party conversion tools. | Yes, in converter context | Unclear | Medium Confidence | FinFolio states split transactions are “blown out” from `SPLIT.INF`. |
| Axys exported files must be exported from Axys; raw Advent format may not be readable by third parties. | Yes | Unknown | High Confidence | Morningstar guide states exported files are needed because raw Advent format cannot be read by their process. |
| Distribution reinvestment conversion may create transaction pairs and affect gain/loss reporting in target system. | Yes, in Morningstar conversion context | Unknown | Verified for conversion behavior only | Do not assume APX behavior. |
| APX API may not cover all data needs; IMEX/Replang/views/stored procedures/SSRS may still be required. | Not applicable | Yes, generally | Medium Confidence | Consultant source; not corporate-action-specific. |
| ACA/APX simple vs complex events have different review workflows. | Not applicable | Yes | Verified | Vendor says simple events may be downloaded automatically while complex events are reviewed. |
| ACA revisions/alerts can change processing instructions. | Not applicable | Yes | Verified | Important for audit trails and historical corrections. |

---

## 13. Version Differences

No version-specific Axys/APX corporate action behavior was verified from supplied material.

| Version Topic | Axys | APX | Status |
|---|---|---|---:|
| Axys versions using `split.inf` | Unknown | Not applicable | Unknown |
| APX versions supporting ACA integration | Not applicable | Unknown | Unknown |
| APX versions containing Reorg Utility | Not applicable | Unknown | Unknown |
| APX versions supporting REST API access to corporate-action-related data | Not applicable | Unknown | Unknown |
| Differences between self-hosted APX, APX Dedicated, and APX Multi-Tenant for ACA workflow | Not applicable | Unknown | Unknown |

---

## 14. Implementation Guidance for Chapter Maintenance

### 14.1 What Can Be Stated Safely Now

Chapter 09 can safely state:

1. Axys data conversions commonly require `.cli`, `sec.inf`, `split.inf`, `.pri`, and `type.inf` files.
2. `split.inf` is identified by third-party conversion documentation as the Axys securities splits file.
3. ACA is verified as an Advent corporate-actions workflow for both Axys and APX at the vendor workflow/product-claim level.
4. In the ACA/Axys workflow, active holdings can be received by daily script, ACA reports and Automation Results email can be produced, simple/mandatory events can process to the Axys Trade Blotter, and complex/option events require review.
5. In the ACA/APX workflow, APX holdings can be sent to ACA, ACA creates action records, users review/download actions, the APX Reorg Utility runs, and resulting transactions post to the APX Trade Blotter.
6. Exact corporate action IMEX objects, REP reports, APX tables, Axys split field layouts, ACA source IDs, final posting lifecycle, and transaction codes remain Unknown unless additional source material is supplied.

### 14.2 What Should Not Be Stated Yet

Do not state any of the following without additional evidence:

- Exact Axys `split.inf` columns.
- Whether ACA-for-Axys writes to `split.inf`, writes only Trade Blotter rows, writes both, or varies by event type.
- Whether Axys ACA-generated Trade Blotter rows are distinguishable from manual or custodian-imported rows.
- Exact Axys or APX transaction codes for dividends, splits, mergers, spin-offs, cash-in-lieu, or return of capital.
- Exact IMEX object names for corporate actions.
- Exact REP report names for corporate actions.
- APX database table names for Reorg Utility, Trade Blotter, or ACA records.
- Whether APX stores a `split.inf` equivalent.
- Whether Axys stores dividends in a separate corporate action file.
- Whether APX final accounting transactions are automatically posted from Trade Blotter or require a manual approval/posting step.
- Whether simple/mandatory ACA events are final-posted after Trade Blotter creation.
- Whether complex-event review status survives into posted transaction records.
- Whether cost-basis/taxability data from ACA specialists is stored in Axys/APX final accounting records.
- Formulae for share/price adjustments in Axys/APX.
- How performance reports use split data or reorg postings.

---

## 15. Evidence Needed to Upgrade Unknowns

| Needed Source | Purpose | Highest-Value Questions Answered |
|---|---|---|
| Axys exported `split.inf` sample | Confirm field layout, date/factor/security keys | What fields are in split history? How are ratios stored? |
| Axys `.cli` sample around cash dividend, reinvestment, split, merger | Confirm transaction codes and generated postings | Which corporate actions are transaction-driven? |
| Axys IMEX dictionary or scripts | Confirm object names and field names | Can IMEX export/import splits? transactions? prices? |
| APX IMEX dictionary or sample exports | Confirm transaction/blotter/security/split objects | How to extract ACA-generated postings? |
| APX ACA/Reorg Utility user guide | Confirm workflow, statuses, posting lifecycle, fields | What does Reorg Utility produce? What fields are retained? |
| APX public views / schema dictionary | Confirm table/view names and supported access paths | Where are Trade Blotter and posted transactions stored? |
| REP report catalog or sample outputs | Identify report names and fields | Which REP reports audit splits/dividends/reorgs? |
| Before/after examples from production | Confirm performance/holding/pricing impacts | How do corrections affect historical reports? |

---

## 16. Chapter Structure Informed by Research

```markdown
## 09-Corporate-Actions

## Overview
- Define scope narrowly: splits, dividends, reorganizations, identifier changes, and ACA/APX workflow.
- State that corporate actions may appear as split records, transactions, security master changes, price/factor adjustments, and/or APX Trade Blotter entries.
- Mark all unsupported details Unknown.

## Axys
- Files: `.cli`, `sec.inf`, `split.inf`, `.pri`, `type.inf`.
- `split.inf` as securities splits file.
- Unknown: field layout, transaction code mapping, report behavior.

## APX
- ACA integration workflow.
- Reorg Utility and Trade Blotter.
- Unknown: tables, fields, status lifecycle, final-posting behavior.

## IMEX
- Unknown exact object names and fields.
- Research checklist for transactions, securities, splits, prices, holdings.

## REP
- Unknown exact report names.
- Need sample reports/catalog.

## Data Model
- Conceptual: Security, Price, Split, Transaction, Holding, ACA Action.
- Do not use conceptual names as actual field/table names.

## Common Fields
- Use only conceptual field placeholders until verified.

## Examples
- Axys stock split example with unknown field names.
- APX ACA reorg workflow example.

## Known Issues / Quirks
- Conversion materialization of split transactions.
- APX simple vs complex event review.
- ACA revision alerts.

## References
- Vendor and third-party source list.

## Unknowns
- Explicit unresolved list.
```

---

## 17. References

1. SS&C Advent. “Overcome the Challenges of Corporate Actions Processing with Axys.” Product brief PDF.
   URL: https://cdn.advent.com/cms/pdfs/briefs/PB_ACAAXYS.pdf

2. SS&C Advent. “Brief: Advent Corporate Actions for APX.” Public product brief page.
   URL: https://www.advent.com/resources/all-resources/brief-advent-corporate-actions-for-apx/

3. SS&C Advent. “What is SS&C Advent Corporate Actions?” Ask Advent page.
   URL: https://www.advent.com/news-and-insights/ask-advent/ask-advent-what-is-ssc-advent-corporate-actions/

4. AdventGuru. “Demystifying Portfolio Accounting Systems Integration Post-Merger/Acquisition” / Data Conversion category.
   URL: https://adventguru.com/category/portfolio-management-systems/data-conversion/

5. AdventGuru. “Building Advent APX Data Pipeline Integration with REST API.”
   URL: https://adventguru.com/2025/10/14/building-advent-apx-data-pipeline-integration-with-rest-api/

6. FinFolio. “Advent Axys®, Moxy® & APX® Conversions.”
   URL: https://www.finfolio.com/advent-apx-moxy-apx-conversions

7. Morningstar Office. “Advent Axys Database Conversion.” PDF.
   URL: https://gladmainnew.morningstar.com/articles/tutorial/30/AdventAxys.pdf

8. SS&C Advent. “Corporate Actions Processing.” White paper.
   URL: https://cdn.advent.com/cms/pdfs/papers/WP_Corporate_Actions_Processing.pdf

---

## 18. Open Unknowns Register

| ID | Unknown | System | Priority |
|---:|---|---|---:|
| CA-U001 | Exact Axys `split.inf` layout | Axys | High |
| CA-U002 | Whether APX has an equivalent split history table/file and its schema | APX | High |
| CA-U003 | Exact transaction codes for dividends, reinvestments, splits, mergers, spin-offs, returns of capital, cash-in-lieu | Axys/APX | High |
| CA-U004 | IMEX object names for transactions, securities, split history, prices, and Trade Blotter | Axys/APX | High |
| CA-U005 | REP reports that expose corporate actions or generated transactions | Axys/APX | High |
| CA-U006 | APX Reorg Utility input/output fields and statuses | APX | High |
| CA-U007 | APX Trade Blotter lifecycle for ACA-generated transactions | APX | High |
| CA-U008 | Whether ACA stores a persistent action ID that survives into APX accounting records | APX/ACA | High |
| CA-U009 | Axys handling of cash dividends separate from transactions | Axys | Medium |
| CA-U010 | Axys handling of stock dividends separate from split records | Axys | Medium |
| CA-U011 | Performance recalculation impact of backdated corporate actions | Axys/APX | Medium |
| CA-U012 | Version-specific differences in ACA/APX integration | APX | Medium |
| CA-U013 | Whether APX REST API exposes corporate-action-related data | APX | Medium |
| CA-U014 | Treatment of fractional shares and cash-in-lieu | Axys/APX | Medium |
| CA-U015 | Identifier-change history versus overwrite behavior | Axys/APX | Medium |
| CA-U016 | Whether ACA-for-Axys writes only Trade Blotter transactions or can also update `split.inf`, `sec.inf`, `.pri`, or other Axys files | Axys | High |
| CA-U017 | Exact fields in the ACA Automation Results email for Axys and APX | Axys/APX | Medium |
| CA-U018 | Whether ACA reports contain stable action IDs that can be reconciled to Trade Blotter or posted transactions | Axys/APX | High |
| CA-U019 | Whether simple/mandatory auto-processed events are flagged differently from reviewed complex events in downstream records | Axys/APX | High |
| CA-U020 | Whether ACA revisions/alerts can be linked to already-posted transactions | Axys/APX | High |
| CA-U021 | Whether Axys Trade Blotter rows created by ACA are distinguishable from manual or custodian-imported rows | Axys | High |
| CA-U022 | Whether APX Reorg Utility rows created from ACA are distinguishable from manual reorg rows | APX | High |
| CA-U023 | Whether cost-basis/taxability data confirmed by ACA specialists survives into Axys/APX accounting records or only appears in ACA reports/instructions | Axys/APX | High |

---

## 19. Research Quality Assessment

This research is adequate to seed a conservative chapter, but not enough to produce a fully detailed technical manual chapter with field dictionaries, IMEX definitions, REP report names, or transaction code mappings.

To produce a stronger chapter, the highest-value additional source material would be:

1. Axys exported `split.inf` and `.cli` examples around real corporate actions.
2. APX ACA/Reorg Utility documentation or screenshots/samples.
3. IMEX dictionaries/scripts for transactions, securities, prices, splits, holdings, and Trade Blotter.
4. REP report catalog or sample outputs for transaction history, splits, reorgs, and reconciliation.

Until those are available, unsupported details should remain marked **Unknown**.
