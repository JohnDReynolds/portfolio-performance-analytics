# Chapter 09 — Corporate Actions

Repository: AXYS / APX Reference Repository
Chapter: `docs/axys-apx-reference/reference/Chapter_09_Corporate_Actions.md`
Prepared: 2026-06-29
Governing specification: `AXYS_APX_REFERENCE_BLUEPRINT.md`, Version 2.0

---

## Related chapters
- [Chapter_01_Overview.md](Chapter_01_Overview.md) — provides the repository-wide map and evidence conventions.
- [Chapter_05_Transactions.md](Chapter_05_Transactions.md) — corporate actions can generate or affect transaction records.
- [Chapter_06_Holdings.md](Chapter_06_Holdings.md) — corporate actions change holdings and cost basis.
- [Chapter_10_Performance.md](Chapter_10_Performance.md) — corporate actions can materially affect performance attribution.

## 1. Overview

Corporate actions in Axys and APX can affect security reference data, holdings, prices, transactions, cash, cost basis, realized gain/loss, income, reporting, reconciliation, and performance.

This chapter documents only behavior supported by the supplied research and source material. It intentionally preserves Unknowns where source material does not establish field names, object names, report names, transaction codes, schemas, or processing rules.

### 1.1 Confidence Labels

| Label | Meaning in this chapter |
|---|---|
| Verified | Directly supported by supplied research, cited vendor material, source screenshots, or observed file/report evidence summarized in the supplied research. |
| High Confidence | Strongly supported by supplied research and multiple indirect sources, but not enough to assert a complete vendor specification. |
| Medium Confidence | Plausible and useful, but source support is conversion-specific, integration-specific, consultant-derived, or otherwise incomplete. |
| Unknown | Not established by the supplied material. Do not implement or document as product fact without additional evidence. |

### 1.2 Scope

This chapter covers:

| Area | Included? | Notes |
|---|---:|---|
| Axys split-related evidence | Yes | `split.inf` is the strongest Axys-specific corporate-action artifact supported by the supplied research. |
| Axys dividend/reinvestment evidence | Limited | Reinvestment evidence comes mainly from conversion behavior and transaction research. |
| Axys Advent Corporate Actions workflow | Yes | Vendor source evidence supports ACA integration with Axys, active holdings by daily script, reports, Automation Results email, and Trade Blotter processing. |
| APX Advent Corporate Actions workflow | Yes | Vendor source evidence supports APX integration with Advent Corporate Actions, APX Reorg Utility, and APX Trade Blotter. |
| IMEX corporate-action object names | Preserved as Unknown | No supplied IMEX dictionary or samples verify object names. |
| REP corporate-action report names | Preserved as Unknown | No supplied REP catalog or report output verifies corporate-action-specific reports. |
| Transaction codes for dividends, splits, mergers, spin-offs, ROC, cash-in-lieu | Preserved as Unknown unless explicitly marked as observed transaction-research examples | Do not infer transaction codes from accounting concepts. |
| Axys/APX internal schemas | Preserved as Unknown | No native schema documentation was supplied. |

### 1.3 Key Findings

| Finding | Axys | APX | Confidence |
|---|---|---|---:|
| Corporate actions may appear as split records, transaction activity, security-master changes, price changes, holdings changes, and/or blotter activity. | Yes, partially supported | Yes, partially supported | Medium Confidence |
| Axys conversion packages commonly include `.cli`, `sec.inf`, `split.inf`, `.pri`, and `type.inf`. | Yes | Unknown | High Confidence |
| `split.inf` is identified in supplied research as the Axys securities splits file. | Yes | Unknown | High Confidence |
| Third-party conversion tools may materialize split history from `SPLIT.INF` into explicit split transactions. | Yes, conversion context | Unclear | Medium Confidence |
| ACA is verified as an Advent corporate-actions workflow for both Axys and APX at the vendor workflow/product-claim level. | Yes | Yes | Verified |
| In the ACA/Axys workflow, active holdings can be received by daily script, ACA reports and Automation Results email can be produced, simple/mandatory events can process to the Trade Blotter, and complex/option events require review. | Yes | N/A | Verified |
| In the ACA/APX workflow, APX sends holdings to ACA, ACA creates action records, actions are reviewed/downloaded, APX Reorg Utility runs, and generated transactions post to APX Trade Blotter. | N/A | Yes | Verified |
| `split.inf` and ACA are separate evidence surfaces; public sources do not prove whether ACA writes `split.inf`, writes Trade Blotter transactions only, writes both, or varies by event type. | Unknown | N/A | Unknown |
| Exact Axys `split.inf` field layout is not established. | Unknown | N/A | Unknown |
| Exact APX Reorg Utility input/output fields and statuses are not established. | N/A | Unknown | Unknown |
| Exact corporate-action IMEX object names are not established. | Unknown | Unknown | Unknown |
| Exact corporate-action REP reports are not established. | Unknown | Unknown | Unknown |

---

## 2. Corporate Action Event Taxonomy

The following table is a technical checklist, not a claim that Axys or APX uses a single canonical corporate-action object for every event type.

| Event Type | Typical Accounting / Operational Impact | Axys Evidence | APX Evidence | Confidence |
|---|---|---|---|---:|
| Cash dividend | Cash income transaction; possible receivable/accrual depending workflow. | Likely transaction-driven, but exact Axys codes Unknown. | Likely transaction or blotter activity, but exact APX codes Unknown. | Medium Confidence |
| Reinvested dividend | Distribution plus purchase/reinvestment activity. | Morningstar conversion research says Axys distribution reinvest transaction types may translate into transaction pairs, normally Buy plus Distribution. | Unknown from supplied corporate-action research. | Verified for conversion behavior only |
| Stock split | Share quantity adjustment; possible price/factor handling. | `split.inf` is identified as securities splits file. | ACA/Reorg Utility supports reorg workflow generally; exact split storage Unknown. | Axys High Confidence; APX Unknown for storage |
| Reverse split | Share quantity adjustment; likely ratio/factor handling. | Likely represented in split data if supported by split conventions, but exact fields Unknown. | Unknown. | Medium Confidence / Unknown |
| Stock dividend | Share distribution or split-like accounting, depending system setup. | Unknown. | Unknown. | Unknown |
| Return of capital | Cash/cost-basis transaction. | Exact transaction codes and processing Unknown. | Exact transaction codes and processing Unknown. | Unknown |
| Principal paydown | Principal reduction; fixed-income/MBS-specific effect. | Transaction research shows principal paydown conversion issues in Axys conversion context; exact corporate-action treatment Unknown. | Unknown. | Medium Confidence for conversion issue; Unknown native behavior |
| Cash exchange | Cash/security exchange event. | Verified in ACA-for-Axys brief as a simple/mandatory example; exact output Unknown. | Likely ACA workflow candidate; exact public APX example not found. | Verified for Axys ACA coverage; Unknown for output |
| Stock exchange | Security exchange event. | Verified in ACA-for-Axys brief as a simple/mandatory example; exact output Unknown. | Likely ACA workflow candidate; exact public APX example not found. | Verified for Axys ACA coverage; Unknown for output |
| Merger | Security reorganization; old/new security mapping; possible cash-in-lieu. | ACA-for-Axys brief covers taxable, non-taxable, combination, and option mergers; exact postings Unknown. | ACA/APX workflow supports reorganization processing generally; exact postings Unknown. | Verified for Axys ACA coverage and APX workflow; Unknown for details |
| Spin-off | New security; cost-basis allocation; possible cash-in-lieu. | ACA-for-Axys brief covers spin-offs; exact postings Unknown. | ACA may support complex event review generally; exact APX output Unknown. | Verified for Axys ACA coverage; APX Medium |
| Tender / exchange offer | Voluntary election; cash/security exchange. | Unknown. | ACA alerts/review process may support complex events generally; exact support Unknown. | Medium Confidence |
| Symbol/ticker/name change | Security-master update and downstream matching impact. | ACA-for-Axys brief covers name changes; exact `sec.inf` behavior Unknown. | Unknown. | Verified for Axys ACA coverage; field behavior Unknown |
| CUSIP change | Security-master update; possible cross-reference challenge. | Unknown. | ACA cross-references APX securities to action database, but APX identifier-history behavior Unknown. | Medium Confidence for ACA cross-reference; Unknown for storage |
| Bond call / redemption | Cash/principal transaction, holding reduction, gain/loss. | Unknown. | Unknown. | Unknown |
| Bankruptcy | Write-off, reorganization, or security status event. | ACA-for-Axys brief covers bankruptcies; exact output Unknown. | ACA complex workflow likely, but exact APX behavior Unknown. | Verified for Axys ACA coverage; APX Unknown |

---

## 3. Axys

### 3.1 Evidence-Supported Axys File Artifacts

The supplied research identifies several Axys file artifacts that can be relevant to corporate actions, security reference data, pricing, holdings, and transaction conversion.

| Artifact | Description | Corporate Action Relevance | Confidence |
|---|---|---|---:|
| `.cli` | Client/account files used in Axys conversion and transaction contexts. | Corporate-action-related transactions may be represented in client/account activity, but exact codes are Unknown. | High Confidence |
| `sec.inf` | Securities file. | Security master attributes may be affected by identifier/name/type changes. Exact field behavior Unknown. | High Confidence |
| `split.inf` / `SPLIT.INF` | Securities splits file. | Strongest Axys-specific evidence for stock split history. Exact layout Unknown. | High Confidence |
| `.pri` | Security price file(s). | Prices may need validation or adjustment around split/reorg dates. Exact split-adjustment mechanics Unknown. | High Confidence |
| `type.inf` | Security type file. | Security type can affect accounting/reporting interpretation. Exact corporate-action dependency Unknown. | High Confidence |
| `bond.inf` | Mentioned in legacy file operations in supplied research. | Corporate-action relevance not established. | Medium Confidence for existence in some contexts; Unknown for this chapter |
| `topost.trn` | Observed Axys Trade Blotter file from transaction/IMEX research. | Corporate-action transactions could theoretically be imported through transaction workflow, but no corporate-action-specific mapping is verified. | Medium Confidence |
| `didpost.aud` | Observed audit-trail file associated with posted transactions in consultant research. | Could be relevant to auditing posted corporate-action transactions, but layout and retention behavior Unknown. | Medium Confidence |

### 3.2 Axys Split Evidence

| Statement | Confidence | Notes |
|---|---:|---|
| Axys has a `split.inf` file identified as a securities splits file. | High Confidence | Supported by Morningstar conversion research and consultant/conversion references summarized in the supplied corporate-action research. |
| `split.inf` is part of common Axys conversion/extract packages along with `.cli`, `sec.inf`, `.pri`, and `type.inf`. | High Confidence | Supported by supplied conversion research. |
| Third-party conversion tools may convert or “blow out” split records from `SPLIT.INF` into explicit split transactions. | Medium Confidence | This describes converter behavior, not native Axys accounting behavior. |
| Exact `split.inf` columns are not established. | Unknown | No sanitized file, vendor layout, or IMEX dictionary was supplied. |
| Exact split ratio/date/security key conventions are not established. | Unknown | Do not invent date/factor fields. |
| Whether Axys uses split records at report time, materializes split transactions internally, or uses another mechanism is not established. | Unknown | Requires Axys documentation or before/after sample files/reports. |

### 3.3 Axys and Advent Corporate Actions

SS&C/Advent's Axys ACA product brief describes Advent Corporate Actions
as integrated with Axys portfolio management software. The brief supports
the following workflow-level claims, but it does not provide field names,
transaction rows, `split.inf` behavior, or final posting mechanics.

| Statement | Confidence | Notes |
|---|---:|---|
| ACA integrates with Axys portfolio management software. | Verified | Vendor product brief. |
| ACA receives active holdings from Axys through a daily script. | Verified | Workflow-level vendor claim; exact interface fields Unknown. |
| ACA provides reports and an Automation Results email summarizing U.S. equity and ADR transaction activity affecting the firm. | Verified | Exact report/email fields Unknown. |
| Simple/mandatory events, such as cash or stock exchanges, can automatically process to the Axys Trade Blotter. | Verified | Exact Trade Blotter rows, transaction codes, and final posting lifecycle Unknown. |
| Non-simple/mandatory events with options, such as mergers with options, require review before processing. | Verified | Exact review-status fields Unknown. |
| ACA specialists analyze and confirm cost basis, taxability, and data elements required by Axys. | Verified for vendor claim | Does not prove that these data elements survive into Axys accounting records. |
| ACA covers U.S. and non-U.S. equities and fixed income, including taxable/non-taxable/combination mergers, name changes, spin-offs, bankruptcies, and other events. | Verified for vendor coverage claim | Exact accounting output remains Unknown. |

`split.inf` and ACA should not be conflated. `split.inf` is an Axys file
artifact for split history in conversion/extract evidence. ACA is a
corporate-action workflow/product that can provide action data and
process events to the Axys Trade Blotter. Public sources do not establish
whether ACA writes `split.inf`, writes Trade Blotter transactions only,
writes both, or uses different paths for different event types.

Conversion-materialized split transactions are a third surface:
third-party conversion evidence says split records may be "blown out"
from `SPLIT.INF` into target-system split transactions. That does not
prove that native Axys stores split transactions in each account `.cli`
file.

### 3.4 Axys Dividends and Reinvestments

| Topic | Evidence-Supported Statement | Confidence | Notes |
|---|---|---:|---|
| Cash dividends | Cash dividends are likely transaction-driven in Axys-style accounting, but exact Axys transaction codes and fields are Unknown. | Medium Confidence | This is not enough to document codes. |
| Reinvested dividends | Morningstar conversion research says Axys distribution reinvest transaction types may translate into transaction pairs, normally a Buy plus corresponding Distribution transaction type. | Verified for Morningstar conversion behavior | Do not generalize to all native Axys behavior without samples. |
| Dividend master file | No supplied source verifies a dedicated Axys dividend master file. | Unknown | Do not invent `dividend.inf` or equivalent. |
| Dividend ex-date / record-date / pay-date fields | Not established for Axys. | Unknown | May be available in reports or transaction exports, but not verified here. |

### 3.5 Axys Reorganizations, Mergers, Spin-Offs, and Identifier Changes

| Topic | Axys Status | Confidence | Needed Evidence |
|---|---|---:|---|
| Merger/reorg transaction generation | Not established. | Unknown | `.cli` samples around merger/reorg, transaction code documentation, REP/IMEX samples. |
| Spin-off handling | Not established. | Unknown | Before/after security, transaction, holding, and price samples. |
| Cash-in-lieu handling | Not established. | Unknown | Transaction code documentation or sample reorg postings. |
| Ticker/name/CUSIP changes | Not established whether Axys stores identifier history or overwrites `sec.inf` fields. | Unknown | `sec.inf` before/after examples or vendor documentation. |
| Security-to-security mapping | Not established. | Unknown | Reorg examples or conversion documentation. |

### 3.6 Axys Processing Behavior

| Process | Known / Unknown | Confidence |
|---|---|---:|
| Split entry/edit workflow | Axys has `split.inf`; exact UI, import, edit, and validation workflow Unknown. | High Confidence for file; Unknown for process |
| Split effect on holdings | Expected to affect share quantities in split-aware holdings/reporting, but exact Axys mechanics Unknown. | Medium Confidence |
| Split effect on prices | Axys has `.pri` price files and `split.inf` split data, but whether price history is adjusted by file update, report calculation, or another mechanism is Unknown. | Unknown |
| Dividend posting | Likely transaction-driven; exact transaction codes and required fields Unknown. | Medium Confidence |
| ACA active-holdings script | ACA receives active holdings from Axys through a daily script. | Verified for workflow; fields Unknown |
| ACA simple/mandatory event processing | Simple/mandatory events can automatically process to the Axys Trade Blotter. | Verified for workflow; rows/codes/final posting Unknown |
| ACA complex/option event review | Non-simple/mandatory events with options require review before processing. | Verified for workflow; review-status fields Unknown |
| Historical corrections | Corporate-action corrections can affect historical holdings/prices/performance conceptually, but exact Axys recalculation behavior is Unknown. | Unknown |
| Performance impact | Unsupported at implementation level in supplied corporate-action research. | Unknown |

### 3.7 Axys Field Dictionary

The supplied research does not provide verified native Axys corporate-action field names. The following table therefore distinguishes verified artifacts from conceptual placeholders.

| Field / Concept | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---:|
| `split.inf` | Axys securities splits file artifact. | Yes | Unknown | Unknown | Unknown | High Confidence |
| `.cli` | Axys client/account files used in transaction/conversion contexts. | Yes | Unknown / conversion context | Unknown | Unknown | High Confidence |
| `sec.inf` | Axys securities file. | Yes | Unknown | Unknown | Unknown | High Confidence |
| `.pri` | Axys security price file(s). | Yes | Unknown | Unknown | Unknown | High Confidence |
| `type.inf` | Axys security type file. | Yes | Unknown | Unknown | Unknown | High Confidence |
| Security identifier | Concept tying action to security. | Conceptual only | Conceptual only | Unknown | Unknown | Unknown |
| Split date / effective date | Conceptual split date. | Likely required conceptually | Unknown | Unknown | Unknown | Unknown |
| Split ratio / factor | Conceptual split ratio/factor. | Likely required conceptually | Unknown | Unknown | Unknown | Unknown |
| Transaction code | Posting code for generated/dividend/reorg activity. | Known concept in transaction data | Known concept in APX transaction data | Unknown | Unknown | Unknown |
| Ex-date | Market entitlement date. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Record date | Holder-of-record date. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Pay date | Payment date. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Cash-in-lieu amount | Fractional-share cash amount. | Unknown | Unknown | Unknown | Unknown | Unknown |

---

## 4. APX

### 4.1 APX and Advent Corporate Actions

The strongest APX-specific evidence in the supplied research is the vendor-described integration between APX and Advent Corporate Actions (ACA).

| ACA/APX Workflow Step | APX Behavior | Confidence |
|---|---|---:|
| Holdings transfer | APX can send updated security holdings information to the ACA Server. | Verified |
| Security cross-reference | ACA cross-references APX securities to an action database and creates action records. | Verified |
| Automation summary | If using a script, ACA sends an Automation Results email summarizing transaction activity after script processing. | Verified |
| Review/download | Users can review all ACA transactions before download, or allow simple transactions to download automatically while reviewing complex events. | Verified |
| Reorg Utility | Downloaded reviewed actions to APX cause the APX Reorg Utility to run. | Verified |
| Trade Blotter | APX Reorg Utility-generated transactions post to APX Trade Blotter. | Verified |
| Alerts | ACA Alerts notify users of revisions and tips for processing complex APX events; alerts may be emailed if the firm holds the highlighted security and may be viewed online. | Verified |
| Specialist support | ACA specialists analyze tax opinions, compile reorganization transactions, troubleshoot processing issues, and challenge data vendors. | Verified |

### 4.2 APX ACA Workflow Model

The supplied research supports the following workflow-level model:

```text
APX security holdings
        ↓
Sent to ACA Server
        ↓
ACA cross-references held securities to action database
        ↓
ACA creates action records / transaction instructions
        ↓
User review or scripted/simple-event automation
        ↓
Reviewed actions downloaded to APX
        ↓
APX Reorg Utility runs
        ↓
Generated transactions post to APX Trade Blotter
        ↓
Final accounting impact: Unknown from supplied evidence
```

| Workflow Boundary | Status | Confidence |
|---|---|---:|
| APX → ACA holdings handoff | Supported by vendor brief. | Verified |
| ACA action creation | Supported at workflow level. | Verified |
| User review vs automatic simple-event download | Supported by vendor brief. | Verified |
| APX Reorg Utility run | Supported by vendor brief. | Verified |
| APX Trade Blotter as destination | Supported by vendor brief. | Verified |
| Whether Trade Blotter records are automatically posted to final transaction history | Not established. | Unknown |
| Transaction codes and fields produced by Reorg Utility | Not established. | Unknown |
| Persistent ACA action ID in APX accounting records | Not established. | Unknown |

### 4.3 APX Native Corporate Action Storage

| Topic | Status | Confidence | Notes |
|---|---|---:|---|
| APX corporate-action database tables | Not identified. | Unknown | Requires APX schema, public views, stored procedures, or official technical documentation. |
| APX Reorg Utility input format | Not identified. | Unknown | Requires ACA/APX user guide, screenshots, or sample imports. |
| APX Reorg Utility output format | Not identified. | Unknown | Requires Trade Blotter export or transaction sample. |
| APX Trade Blotter field mapping for ACA transactions | Not identified. | Unknown | Requires ACA download, APX blotter export, or IMEX transaction export. |
| APX split storage | Not identified. | Unknown | Do not infer `split.inf` equivalent from Axys evidence. |
| APX final posted transaction storage | Not identified in corporate-action research. | Unknown | Transaction chapter has general APX transaction evidence, but not ACA-specific storage. |
| APX REST API exposure for corporate actions | Not verified. | Unknown | Consultant research says REST/API may not cover all data needs, but corporate-action endpoints were not established. |

### 4.4 APX Simple vs Complex Events

| Topic | Evidence-Supported Statement | Confidence |
|---|---|---:|
| Simple actions | ACA/APX workflow can allow simple transactions to download automatically. | Verified |
| Complex actions | Users can review complex events before download. | Verified |
| Revisions | ACA Alerts notify users of revisions and tips for complex APX events. | Verified |
| Operational implication | Corporate-action processing may not be a one-time static event; revisions and complex-event review can affect audit and control design. | High Confidence |
| Exact APX status fields for simple/complex/revised events | Not established. | Unknown |

### 4.5 APX Field Dictionary

The supplied research does not provide verified APX corporate-action table or field names. The following are workflow terms and conceptual placeholders only.

| Field / Concept | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---:|
| ACA Server / ACA workflow | External ACA processing workflow/server context. | Yes, workflow term | Yes, workflow term | Unknown | Unknown | Verified |
| Action database | ACA database cross-referenced to held securities. | Yes, workflow term | Yes, ACA workflow term | Unknown | Unknown | Verified |
| Action records / instructions | ACA-created records/instructions. | Yes, workflow term; fields Unknown | Yes, workflow term; fields Unknown | Unknown | Unknown | Verified at workflow level |
| Automation Results email | Email summarizing transaction activity after script processing. | Yes | Yes | No | No | Verified |
| APX Reorg Utility | APX utility run after reviewed ACA actions are downloaded. | No | Yes | Unknown | Unknown | Verified |
| Axys Trade Blotter | Destination for simple/mandatory ACA-for-Axys events. | Yes | N/A | Related | Unknown | Verified for workflow; rows Unknown |
| APX Trade Blotter | Destination for Reorg Utility-generated transactions. | N/A | Yes | Related | Unknown | Verified |
| ACA Alert | Revision/tip notification for complex events. | Yes, product-level ACA context | Yes | No | Unknown | Verified |
| ACA action ID | Persistent identifier tying ACA event to APX posting. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Reorg status | Status of APX reorganization action. | N/A | Unknown | Unknown | Unknown | Unknown |
| Corporate-action transaction code | Code generated into APX blotter/transaction history. | Unknown | Unknown | Unknown | Unknown | Unknown |

---

## 5. IMEX

### 5.1 Corporate Action IMEX Coverage

No supplied IMEX dictionary, sample IMEX export, import control file, or command script identifies corporate-action-specific IMEX object names. Exact object names and field lists remain Unknown.

| Candidate Data Need | Likely Data Surface | Axys IMEX Status | APX IMEX Status | Confidence |
|---|---|---|---|---:|
| Dividend transactions | Transaction export/import | Object and fields Unknown | Object and fields Unknown | Unknown |
| Reinvestment transactions | Transaction export/import | Object and fields Unknown | Object and fields Unknown | Unknown |
| Split history | Split-specific export/import, security export, or file export | `split.inf` exists, but IMEX object Unknown | Unknown | Unknown |
| Security master updates | Security export/import | Object and fields Unknown | Object and fields Unknown | Unknown |
| Price/factor adjustments around splits | Price export/import | Object and fields Unknown | Object and fields Unknown | Unknown |
| APX ACA-generated transactions | Trade Blotter or transaction export | N/A | Object and fields Unknown | Unknown |
| ACA action identifiers | ACA/APX export or database/API | N/A | Unknown | Unknown |
| Holdings used for ACA eligibility | Holdings export or direct APX-to-ACA process | N/A | Vendor says APX sends holdings to ACA; mechanism not identified as IMEX. | Verified for workflow; Unknown for interface |

### 5.2 IMEX Questions to Preserve

| Question | Why It Matters | Status |
|---|---|---:|
| Which IMEX object exports posted transactions in Axys and APX? | Needed to audit dividend/reorg postings. | Unknown |
| Which IMEX object exports Trade Blotter rows before posting? | Needed to review pending corporate-action activity. | Unknown |
| Are stock splits exported through transaction IMEX, security master IMEX, a split-specific object, or only file export? | Needed to model split corrections. | Unknown |
| Does IMEX expose original ACA action IDs or only resulting APX transactions? | Needed for traceability. | Unknown |
| Are ACA-generated transactions marked with source/origin fields? | Needed to distinguish ACA-generated from manual entries. | Unknown |
| What fields identify canceled/reversed/reposted corporate-action transactions? | Needed for historical audit. | Unknown |
| Does APX IMEX expose Reorg Utility status or only resulting blotter rows? | Needed for workflow monitoring. | Unknown |

### 5.3 IMEX Implementation Guidance

| Guidance | Confidence |
|---|---:|
| Treat IMEX as a possible extraction surface for resulting transactions, securities, prices, holdings, or split data, but do not assume corporate-action-specific objects exist. | High Confidence |
| For Axys splits, confirm whether `split.inf` can be exported/imported through IMEX or must be handled as an Axys information file. | Unknown pending evidence |
| For APX ACA output, confirm whether the usable extract is Trade Blotter rows, posted transactions, ACA action records, or report output. | Unknown pending evidence |
| Do not design an implementation around inferred transaction codes or inferred field names. | High Confidence |

---

## 6. REP / Replang

### 6.1 Corporate-Action-Specific REP Evidence

No supplied REP source, report catalog, or report output identifies corporate-action-specific Axys/APX report names or fields.

| Report / REP Need | Axys | APX | Confidence | Notes |
|---|---|---|---:|---|
| Report listing split history | Unknown | Unknown | Unknown | Need REP report inventory or sample. |
| Report listing dividend transactions | Unknown | Unknown | Unknown | Could be a transaction report rather than a corporate-action report. |
| Report listing reorganization entries | Unknown | Unknown | Unknown | Need sample report or APX report catalog. |
| Report listing security master changes | Unknown | Unknown | Unknown | Need security master/report evidence. |
| Report listing APX Trade Blotter corporate-action entries | N/A | Unknown | Unknown | Need APX report catalog or Trade Blotter report. |
| ACA reports | N/A | ACA has reporting capability in product positioning | Verified for existence of ACA reporting capability only | Exact report names and fields Unknown. |
| Reconciliation report | Axys conversion workflow requests a Reconciliation report as of the last transaction date | Unknown | High Confidence for Axys conversion workflow | Not corporate-action-specific. |

### 6.2 REP Questions to Preserve

| Question | Status |
|---|---:|
| Which standard REP reports display split history? | Unknown |
| Can REP read `split.inf` directly? | Unknown |
| Which standard REP reports display dividend/reorg transactions? | Unknown |
| Does APX have a standard ACA/Reorg Utility report? | Unknown |
| Can REP report APX Trade Blotter rows before final posting? | Unknown |
| Does REP expose transaction source/origin fields for ACA-generated APX entries? | Unknown |
| Are Axys and APX report names different for corporate-action review? | Unknown |
| Which reports use stored values versus recalculated values around split/reorg dates? | Unknown |

### 6.3 REP Implementation Guidance

| Guidance | Confidence |
|---|---:|
| Use REP/report output only after confirming exact report name, parameters, output fields, and whether the report includes posted or pending activity. | High Confidence |
| Treat transaction reports as possible corporate-action evidence, but do not assume they expose original corporate-action source records. | High Confidence |
| Treat APX ACA reports as separate from APX accounting records unless sample output proves the relationship. | Medium Confidence |

---

## 7. Data Model

### 7.1 Conceptual Data Model

The following model is conceptual. These are not verified Axys/APX table names.

```text
Security Master
    ↓
Corporate Action Event
    ↓
ACA Action Record / Instruction
    ↓
Review / Automation / Exception Workflow
    ↓
Split record, security update, transaction, price adjustment, or blotter row
    ↓
Holdings, cash, cost basis, income, realized gain/loss, prices
    ↓
Reports, reconciliation, performance, audit trail
```

### 7.2 Conceptual Entities

| Conceptual Entity | Purpose | Axys Evidence | APX Evidence | Confidence |
|---|---|---|---|---:|
| Security | Security record affected by symbol/CUSIP/name/type/reorg changes. | `sec.inf` identified as securities file. | APX security master exists in integration/security research; exact table Unknown. | Axys High Confidence; APX Unknown for table |
| Price | Price records may need validation around split dates. | `.pri` identified as security prices file. | APX price storage Unknown. | Axys High Confidence; APX Unknown |
| Split | Split factor/history. | `split.inf` identified as securities splits file. | APX split storage Unknown. | Axys High Confidence; APX Unknown |
| Transaction | Dividends, reinvestments, reorg postings, cash-in-lieu, ROC. | `.cli` files used for transactions in conversion context. | APX Trade Blotter receives ACA-generated transactions. | Axys High Confidence for conversion; APX Verified for Trade Blotter workflow |
| Holding | Used to determine event eligibility and impact. | Axys ACA receives active holdings by daily script; exact interface Unknown. | APX sends holdings to ACA Server. | Verified for ACA workflow; fields Unknown |
| ACA action record / instruction | External ACA event/instruction after action matching, normalization, specialist review, or automation. | ACA-for-Axys workflow evidence supports the concept; exact fields Unknown. | ACA creates action records. | Verified at workflow level |
| Trade Blotter row | Pending/imported transaction row. | ACA-for-Axys brief says simple/mandatory events can process to Trade Blotter; `topost.trn` observed in Axys transaction research. | ACA-generated transactions post to APX Trade Blotter. | Verified for workflow; fields Unknown |
| Audit trail | Record of posted transactions or changes. | `didpost.aud` observed in transaction research. | `didpost.aud` described in consultant research for Axys/APX, but ACA-specific use Unknown. | Medium Confidence |

### 7.3 Relationship Hypotheses Requiring Verification

| Hypothesis | Confidence | Verification Needed |
|---|---:|---|
| Axys applies `split.inf` records to holdings/reports rather than storing split-generated transactions in every `.cli` file. | Medium Confidence | Compare `.cli`, `split.inf`, holdings reports, and performance reports before/after split. |
| Conversion tools materialize `SPLIT.INF` into explicit transactions because target systems require transaction-level splits. | Medium Confidence | Conversion specifications and sample converted output. |
| Cash dividends in Axys/APX are represented as transactions rather than central dividend-master records. | Medium Confidence | Transaction code dictionary and sample exports. |
| APX ACA processing creates Trade Blotter transactions that must be reviewed/posted before final accounting effect. | Medium Confidence | Vendor says transactions post to Trade Blotter, but final lifecycle is Unknown. |
| ACA maintains event records separate from APX accounting records. | Medium Confidence | Vendor says ACA uses an action database; exact data model Unknown. |
| APX posted transactions retain a source/origin marker tying them back to ACA. | Unknown | APX transaction export or ACA/Reorg Utility documentation. |
| Axys ACA-generated Trade Blotter rows retain a source/origin marker tying them back to ACA. | Unknown | Axys Trade Blotter export after ACA processing or ACA report/download sample. |

---

## 8. Common Fields

The following table separates verified labels/artifacts from conceptual fields. Conceptual fields must not be treated as native Axys/APX field names.

| Field / Label | Description | Axys | APX | IMEX | REP | Confidence |
|---|---|---:|---:|---:|---:|---:|
| `.cli` | Axys client/account file artifact used in conversion/transaction contexts. | Yes | Conversion context only | Unknown | Unknown | High Confidence |
| `sec.inf` | Axys securities file artifact. | Yes | Unknown | Unknown | Unknown | High Confidence |
| `split.inf` / `SPLIT.INF` | Axys securities splits file artifact. | Yes | Unknown | Unknown | Unknown | High Confidence |
| `.pri` | Axys security price file artifact. | Yes | APX AIA examples may use `.pri` file naming, but native meaning Unknown | Unknown | Unknown | High Confidence for Axys; Medium for APX AIA context |
| `type.inf` | Axys security type file artifact. | Yes | Unknown | Unknown | Unknown | High Confidence |
| `ACA reports` | ACA reporting surface in vendor workflow. | Yes | Yes | No | Unknown | Verified for existence; fields Unknown |
| `Automation Results email` | ACA script/automation processing summary. | Yes | Yes | No | No | Verified for existence; fields Unknown |
| `APX Reorg Utility` | APX utility in ACA workflow. | No | Yes | Unknown | Unknown | Verified |
| `Axys Trade Blotter` | Destination for simple/mandatory ACA-for-Axys events. | Yes | N/A | Related | Unknown | Verified for workflow; rows Unknown |
| `APX Trade Blotter` | Destination for ACA-generated APX transactions. | N/A | Yes | Related | Unknown | Verified |
| `ACA Alerts` | ACA notifications for revisions/complex events. | Yes | Yes | No | Unknown | Verified |
| Security identifier | Conceptual identifier tying an event to a security. | Unknown field | Unknown field | Unknown | Unknown | Unknown |
| Old security identifier | Conceptual reorg field. | Unknown | Unknown | Unknown | Unknown | Unknown |
| New security identifier | Conceptual reorg field. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Split effective date | Conceptual split date. | Unknown field in `split.inf` | Unknown | Unknown | Unknown | Unknown |
| Split ratio / factor | Conceptual split ratio/factor. | Unknown field in `split.inf` | Unknown | Unknown | Unknown | Unknown |
| Ex-date | Conceptual dividend/split entitlement date. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Record date | Conceptual record date. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Pay date | Conceptual payment date. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Cash-in-lieu amount | Conceptual fractional-share cash amount. | Unknown | Unknown | Unknown | Unknown | Unknown |
| Transaction code | Posting code for resulting transaction. | Unknown for corporate-action-specific codes | Unknown for corporate-action-specific codes | Unknown | Unknown | Unknown |
| Source/origin | Marker identifying ACA/manual/import origin. | Unknown | Unknown | Unknown | Unknown | Unknown |

---

## 9. Examples

### 9.1 Axys Stock Split Research Scenario

Scenario: A portfolio holds 100 shares of XYZ. XYZ has a 2-for-1 split effective 2026-04-15.

| Required Data | Potential Axys Location | Status |
|---|---|---:|
| Security identifier for XYZ | `sec.inf` or transaction/security reference | Exact field Unknown |
| Split effective date | `split.inf` | Exact field Unknown |
| Split ratio/factor | `split.inf` | Exact field Unknown |
| Pre-split and post-split prices | `.pri` files | Exact layout and adjustment mechanics Unknown |
| Account holdings before split | `.cli`, position files, or holdings report | Extraction method Unknown |
| Reported post-split shares | Portfolio Appraisal / holdings report | Report behavior Unknown for split mechanics |

Research conclusion: Axys split history is evidenced by `split.inf`, but the field layout, calculation formula, report timing, and price-adjustment behavior require sample files or vendor documentation.

### 9.2 Axys Reinvested Dividend Conversion Scenario

Scenario: A mutual fund pays a dividend that is reinvested.

| Required Data | Potential Axys Location | Status |
|---|---|---:|
| Distribution transaction | `.cli` activity | Exact code Unknown |
| Buy/reinvestment transaction | `.cli` activity | Exact code Unknown |
| Income amount | Transaction activity/report | Exact field Unknown |
| Reinvested quantity | Transaction activity/report | Exact field Unknown |
| Tax/cost-basis impact | Lot/cost basis records/reports | Unknown |

Research conclusion: Morningstar conversion evidence says Axys distribution reinvest activity may translate into transaction pairs, normally Buy plus Distribution, but this remains conversion-context evidence only.

### 9.3 APX ACA Reorganization Scenario

Scenario: ACA identifies a merger affecting a security held in APX.

| Workflow Step | Evidence Status |
|---|---:|
| APX sends updated security holdings to ACA Server. | Verified |
| ACA cross-references APX securities to action database. | Verified |
| ACA creates action records / transaction instructions. | Verified at workflow level |
| User reviews transactions or simple transactions download automatically. | Verified |
| Reviewed actions download to APX. | Verified |
| APX Reorg Utility runs. | Verified |
| Generated transactions post to APX Trade Blotter. | Verified |
| Final posted transaction codes and fields. | Unknown |
| Whether user approval/posting is required after Trade Blotter creation. | Unknown |
| Downstream performance impact. | Unknown |

### 9.4 Corporate Action Audit Trace Example

A conservative audit trace for APX ACA-generated activity should preserve at least the following concepts, even though exact fields are Unknown:

| Audit Concept | Evidence Status |
|---|---:|
| Held security at eligibility date | APX sends holdings to ACA — Verified workflow |
| ACA action record | ACA creates action records — Verified workflow |
| Review or automation path | Review/auto-download behavior — Verified workflow |
| Reorg Utility processing | Verified workflow |
| Trade Blotter row(s) | Verified workflow |
| Final posted transaction(s) | Unknown |
| Link from posted transaction back to ACA event | Unknown |
| Correction/revision alerts | ACA Alerts — Verified workflow |

---

## 10. Known Issues / Quirks

| Issue / Quirk | Axys | APX | Confidence | Notes |
|---|---|---|---:|---|
| `split.inf` may need special handling during data conversion or firm merger. | Yes | Unknown | Medium Confidence | Consultant/conversion research references exported CSV copies of split files and merging. |
| Split histories may be converted into explicit split transactions by third-party conversion tools. | Yes, conversion context | Unclear | Medium Confidence | Do not treat converter output as native Axys transaction storage. |
| Raw Advent-format files may need export/conversion before third-party tools can read them. | Yes | Unknown | High Confidence | Conversion research emphasizes exported files rather than raw proprietary format. |
| Distribution reinvestment conversion may create transaction pairs and affect gain/loss reporting in target systems. | Yes, conversion context | Unknown | Verified for conversion behavior only | Do not generalize without native samples. |
| ACA simple and complex events can follow different review paths. | N/A | Yes | Verified | Simple transactions may download automatically; complex events may require review. |
| ACA revisions/alerts can alter processing instructions. | N/A | Yes | Verified | Important for audit and historical controls. |
| APX API may not cover all needed data; IMEX/Replang/public views/stored procedures/SSRS may still be required. | N/A | Yes | Medium Confidence | General integration strategy evidence, not corporate-action-specific object evidence. |
| Exact corporate-action transaction codes are not verified. | Unknown | Unknown | Unknown | Do not infer codes from observed transaction matrix unless corporate-action-specific sample verifies them. |
| Code-only interpretation is unsafe for transaction-like corporate actions. | Yes | Yes | High Confidence as design rule | Supported by transaction research showing sign, type, source/destination, and configuration matter. |
| Split/price/holding reconciliation requires coordinated data. | Yes | Yes | High Confidence as implementation guidance | Exact mechanics Unknown. |

---

## 11. Version Differences

No version-specific Axys/APX corporate-action behavior was verified from the supplied corporate-action research.

| Version Topic | Axys | APX | Status |
|---|---|---|---:|
| Axys versions using `split.inf` | Unknown | N/A | Unknown |
| Exact `split.inf` layout by Axys version | Unknown | N/A | Unknown |
| APX versions supporting ACA integration | N/A | Unknown | Unknown |
| APX versions containing APX Reorg Utility | N/A | Unknown | Unknown |
| APX versions supporting ACA automation emails/alerts | N/A | Unknown | Unknown |
| APX REST API support for corporate-action data | N/A | Unknown | Unknown |
| Self-hosted APX vs APX Dedicated vs APX Multi-Tenant ACA differences | N/A | Unknown | Unknown |
| IMEX fixed-format vs delimited support for corporate-action-related data | Unknown | Unknown | Unknown |

Related version-context evidence from other supplied research:

| Area | System | Evidence | Confidence |
|---|---|---|---:|
| Axys direct file formats can change across versions | Axys | Consultant IMEX research cites Axys 3.7 to 3.8 file conversion risk. | Medium Confidence |
| APX retained IMEX but eliminated fixed-format generation in early APX v1.x-v4.x, according to consultant source | APX | IMEX research. | Medium Confidence |
| APX has SQL/reporting alternatives | APX | IMEX/REP research. | Medium Confidence |

---

## 12. Implementation Guidance

### 12.1 Safe Statements for Developers

A developer may safely assume only the following from the supplied material:

| Statement | Confidence |
|---|---:|
| Axys conversion/extract packages commonly include `.cli`, `sec.inf`, `split.inf`, `.pri`, and `type.inf`. | High Confidence |
| `split.inf` is identified as an Axys securities splits file. | High Confidence |
| ACA is verified as an Advent corporate-actions workflow for both Axys and APX at the vendor workflow/product-claim level. | Verified |
| In the ACA/Axys workflow, active holdings can be received by daily script, reports and Automation Results email can be produced, simple/mandatory events can process to Trade Blotter, and complex/option events require review. | Verified |
| In the ACA/APX workflow, APX holdings can be sent to ACA, ACA creates action records, users review/download actions, APX Reorg Utility runs, and resulting transactions post to APX Trade Blotter. | Verified |
| Exact field layouts, transaction codes, IMEX objects, REP report names, APX tables, ACA action IDs, and final-posting lifecycle must be treated as Unknown. | Verified by absence in supplied source material |

### 12.2 Do Not Implement Without Additional Evidence

Do not implement any of the following as hard-coded product facts without additional source material:

| Unsupported Assumption | Status |
|---|---:|
| Exact `split.inf` columns. | Unknown |
| Exact Axys split ratio convention. | Unknown |
| Exact Axys/APX transaction codes for dividends, splits, mergers, spin-offs, return of capital, cash-in-lieu, or paydowns. | Unknown |
| Exact IMEX object names for corporate actions or splits. | Unknown |
| Exact REP report names for corporate actions. | Unknown |
| APX database table names for ACA action records, Reorg Utility output, Trade Blotter rows, or posted transactions. | Unknown |
| Whether APX final accounting transactions are automatically posted from Trade Blotter or require manual approval/posting. | Unknown |
| Whether APX stores an equivalent of Axys `split.inf`. | Unknown |
| Whether Axys stores dividends in a separate corporate-action master. | Unknown |
| Whether posted APX transactions retain an ACA action ID. | Unknown |
| Formulae for share, price, factor, or cost-basis adjustments. | Unknown |
| How performance reports use split/reorg/dividend data. | Unknown |

### 12.3 Minimal Data for Future Testing

To test a corporate-action workflow across Axys/APX, a sanitized example should ideally include:

| Data | Purpose |
|---|---|
| Security master before and after action | Identify ticker/CUSIP/name/type changes. |
| Holdings before and after action | Verify quantity and market-value effects. |
| Transaction history before and after action | Identify generated cash/security entries. |
| Price records before and after action | Verify split/reorg price treatment. |
| Split file or action record | Verify event date, factor, and security key. |
| Trade Blotter export | Validate pending/reviewed postings. |
| Posted transaction export | Validate final accounting effect. |
| REP/report output | Compare user-facing reports against source-data. |
| Audit trail/export/logs | Confirm processing source, user/script, date/time, and corrections. |

---

## 13. Audit and Reconciliation Considerations

These are implementation guidance rules derived from supplied research and general accounting dependency logic. They are not verified native Axys/APX system rules unless explicitly noted.

| Rule ID | Rule | Inputs | Confidence |
|---|---|---|---:|
| CA-AUD-001 | Split events should reconcile to post-split holdings quantities. | Pre/post holdings, split factor, action date. | High Confidence as audit design; native behavior Unknown |
| CA-AUD-002 | Split events should reconcile to price history or valuation behavior around the split date. | Pre/post prices, split factor, holdings, market values. | High Confidence as audit design; native behavior Unknown |
| CA-AUD-003 | Dividend and reinvestment activity should be checked for paired income and purchase/reinvestment entries where the source workflow represents reinvestments as pairs. | Transactions, holdings, income, quantity. | Medium Confidence; conversion evidence only |
| CA-AUD-004 | ACA-generated APX reorg activity should be traceable from held security to ACA action to Reorg Utility to Trade Blotter to final posted transactions. | ACA action records, Trade Blotter, posted transactions, audit logs. | High Confidence as audit design; exact fields Unknown |
| CA-AUD-005 | ACA Alerts and revisions should be reviewed before finalizing complex-event audit conclusions. | ACA alerts, action revisions, posting dates. | Verified for alert existence; field details Unknown |
| CA-AUD-006 | Corporate-action transaction codes must be interpreted with security type, source/destination, quantity sign, amount sign, and configuration context. | Transaction export, security master, transaction config. | High Confidence from transaction research |
| CA-AUD-007 | Corporate-action corrections may restate prior holdings, prices, income, and performance; affected reports should be rerun and compared. | Historical reports, transactions, prices, holdings. | High Confidence as design rule; system mechanics Unknown |
| CA-AUD-008 | Conversion-created split transactions should not be assumed to exist in native Axys data unless verified against source files. | `split.inf`, `.cli`, conversion output. | High Confidence caution |
| CA-AUD-009 | ACA-processed events should be traceable from held security to ACA action/instruction to Trade Blotter row to posted accounting result where fields are available. | Holdings, ACA report/email/action export, Trade Blotter export, posted transaction export, audit trail. | High Confidence as audit design; exact fields Unknown |
| CA-AUD-010 | Events auto-processed as simple/mandatory should be distinguishable from complex/reviewed events where workflow output supports it. | ACA workflow output, Automation Results email, Trade Blotter records, review status if available. | Verified workflow concept; fields Unknown |
| CA-AUD-011 | ACA alerts/revisions should trigger review of previously processed or pending events. | ACA alerts, revised action records, prior postings, Trade Blotter, audit trail. | Verified alert/revision existence; exact fields Unknown |
| CA-AUD-012 | Axys split effects should reconcile across `split.inf`, holdings, prices, and any materialized split transactions created by conversion or downstream tools. | `split.inf`, holdings before/after, price records, `.cli` transactions, conversion output. | High Confidence as audit design; exact native mechanics Unknown |
| CA-AUD-013 | Corporate actions with options, such as mergers with options, should retain evidence of option/election treatment. | ACA action records, election data, review evidence, Trade Blotter output. | Verified for Axys ACA workflow concept; exact fields Unknown |
| CA-AUD-014 | Events involving cost basis or taxability should preserve the data elements and assumptions used to process the action where available. | ACA specialist data, tax opinion, cost-basis allocation, transaction output, lots/cost basis. | Vendor Axys ACA brief supports the specialist-review concept; fields Unknown |

---

## 14. References

The chapter is based only on supplied source and research material. The following references are drawn from the supplied research notes.

### 14.1 Governing Repository Specification

1. `AXYS_APX_REFERENCE_BLUEPRINT.md`, Version 2.0. Governs factual discipline, confidence labels, Axys/APX separation, field dictionary format, and Unknown handling.

### 14.2 Corporate Actions Research Sources

1. SS&C Advent — “Brief: Advent Corporate Actions for APX.” Supports the APX/ACA workflow, APX holdings handoff, ACA cross-reference, review/download process, APX Reorg Utility, APX Trade Blotter, alerts, and specialist support.
2. SS&C Advent — “Overcome the Challenges of Corporate Actions Processing with Axys.” Supports the Axys/ACA workflow, active-holdings script, reports, Automation Results email, Trade Blotter processing, simple/complex event distinction, and coverage claims.
3. SS&C Advent — “What is SS&C Advent Corporate Actions?” Supports ACA product positioning, dashboard/calendar/reporting context, and general corporate-action operations.
4. AdventGuru — Data conversion / merger integration articles. Supports consultant evidence around Axys/APX conversion, exported CSV copies of `split.inf`, and integration cautions.
5. FinFolio — Advent Axys, Moxy & APX conversion material. Supports converter-observed handling where split transactions may be “blown out” from `SPLIT.INF`.
6. Morningstar Office — Advent Axys Database Conversion guide. Supports Axys conversion package evidence for `.cli`, `sec.inf`, `split.inf`, `.pri`, and `type.inf`; also supports conversion behavior around distribution reinvestments and conversion limitations.
7. AdventGuru — APX REST/API/data pipeline article. Supports general caution that APX API may not cover all data and IMEX/Replang/public views/stored procedures/SSRS may still be needed.
8. SS&C Advent — Corporate Actions Processing white paper. Used as corporate-actions background only; not Axys/APX-specific implementation evidence.

### 14.3 Cross-Chapter Supplied Research Used

| Research File | Chapter Use |
|---|---|
| `../evidence/Research_04_Security_Master.md` | `sec.inf`, `type.inf`, security matching, symbol/type caution, security-master dependencies. |
| `../evidence/Research_05_Transactions.md` | Transaction context, observed transaction-code caution, Trade Blotter, `topost.trn`, `didpost.aud`, reinvestment conversion, code-only interpretation warning. |
| `../evidence/Research_06_Holdings.md` | Holdings dependency and APX holdings-to-external-workflow context. |
| `../evidence/Research_08_Pricing.md` | `.pri`, price dependency, split/price audit considerations, price-file caution. |
| `../evidence/Research_12_IMEX.md` | IMEX unknowns, Axys file/folder artifacts, import/export/log cautions, APX IMEX/log context. |
| `../evidence/Research_13_REP.md` | REP/Replang/reporting unknowns and report extraction cautions. |

---

## 15. Unknowns Register

| ID | Unknown | System | Priority | Needed Evidence |
|---:|---|---|---:|---|
| CA-U001 | Exact Axys `split.inf` layout. | Axys | High | Sanitized `split.inf`, vendor file layout, or IMEX/export documentation. |
| CA-U002 | Whether APX has an equivalent split history table/file and its schema. | APX | High | APX schema, public views, IMEX export, or APX documentation. |
| CA-U003 | Exact transaction codes for dividends, reinvestments, splits, mergers, spin-offs, returns of capital, principal paydowns, cash-in-lieu. | Axys/APX | High | Official transaction code manuals or sanitized exports. |
| CA-U004 | IMEX object names for transactions, securities, split history, prices, holdings, and Trade Blotter. | Axys/APX | High | Axys/APX IMEX manuals, control files, screenshots, sample exports. |
| CA-U005 | REP reports that expose corporate actions or generated transactions. | Axys/APX | High | REP catalog, `.rep` files, sample report outputs. |
| CA-U006 | APX Reorg Utility input/output fields and statuses. | APX | High | APX Reorg Utility documentation or sample workflow. |
| CA-U007 | APX Trade Blotter lifecycle for ACA-generated transactions. | APX | High | APX Trade Blotter documentation or production sample. |
| CA-U008 | Whether ACA stores a persistent action ID that survives into APX accounting records. | APX/ACA | High | ACA/APX export with action ID through posting. |
| CA-U009 | Axys handling of cash dividends separate from transactions. | Axys | Medium | `.cli` sample, transaction export, report output, vendor docs. |
| CA-U010 | Axys handling of stock dividends separate from split records. | Axys | Medium | `split.inf` sample and stock-dividend example. |
| CA-U011 | Performance recalculation impact of backdated corporate actions. | Axys/APX | Medium | Performance chapter evidence, before/after test cases. |
| CA-U012 | Version-specific differences in ACA/APX integration. | APX | Medium | APX/ACA release notes and versioned documentation. |
| CA-U013 | Whether APX REST API exposes corporate-action-related data. | APX | Medium | APX REST/API documentation or test output. |
| CA-U014 | Treatment of fractional shares and cash-in-lieu. | Axys/APX | Medium | Reorg/split samples and transaction exports. |
| CA-U015 | Identifier-change history versus overwrite behavior. | Axys/APX | Medium | Security master before/after samples and schema docs. |
| CA-U016 | Whether REP can read `split.inf` directly. | Axys | Medium | REP source examples or RepLang documentation. |
| CA-U017 | Whether ACA-generated APX Trade Blotter rows contain source/origin fields. | APX | High | Trade Blotter export after ACA processing. |
| CA-U018 | Whether corporate-action corrections appear in `didpost.aud` or other audit files with sufficient detail. | Axys/APX | High | Audit trail export and documentation. |
| CA-U019 | Whether ACA-for-Axys writes only Trade Blotter transactions or can also update `split.inf`, `sec.inf`, `.pri`, or other Axys files. | Axys | High | Axys ACA workflow sample or vendor technical documentation. |
| CA-U020 | Exact fields in the ACA Automation Results email for Axys and APX. | Axys/APX | Medium | Sanitized Automation Results email. |
| CA-U021 | Whether ACA reports contain stable action IDs that can be reconciled to Trade Blotter or posted transactions. | Axys/APX | High | ACA report/export and accounting output sample. |
| CA-U022 | Whether simple/mandatory auto-processed events are flagged differently from reviewed complex events in downstream records. | Axys/APX | High | ACA output, Trade Blotter export, posted transaction export. |
| CA-U023 | Whether ACA revisions/alerts can be linked to already-posted transactions. | Axys/APX | High | ACA alert history and posting/audit trail sample. |
| CA-U024 | Whether Axys Trade Blotter rows created by ACA are distinguishable from manual or custodian-imported rows. | Axys | High | Axys Trade Blotter export after ACA processing. |
| CA-U025 | Whether APX Reorg Utility rows created from ACA are distinguishable from manual reorg rows. | APX | High | APX Reorg Utility and Trade Blotter samples. |
| CA-U026 | Whether cost-basis/taxability data confirmed by ACA specialists survives into Axys/APX accounting records or only appears in ACA reports/instructions. | Axys/APX | High | ACA action details plus final accounting records. |

---

## 16. Additional Evidence Needed

To upgrade this chapter from conservative technical reference to detailed implementation manual, the highest-value additional material would be:

| Needed Source | Would Resolve |
|---|---|
| Sanitized Axys `split.inf` sample | Field layout, security keys, date/factor conventions. |
| Sanitized Axys `.cli` examples around dividends, reinvestments, splits, mergers, ROC, paydowns | Transaction codes, posting patterns, cash/security effects. |
| Sanitized Axys `.pri` examples around split dates | Price adjustment behavior. |
| Axys IMEX dictionary/scripts | Object names and fields for transactions, securities, splits, prices. |
| APX ACA/Reorg Utility user guide | Workflow statuses, inputs, outputs, review/posting lifecycle. |
| APX ACA sample output | Action records, generated transactions, source IDs. |
| APX Trade Blotter export after ACA processing | Field mapping and final posting workflow. |
| APX posted transaction export after ACA processing | Final accounting effect and traceability. |
| APX public views/schema dictionary | Native table/view names and fields. |
| REP report catalog / `.rep` sources | Report names and fields for split/reorg/dividend review. |
| Before/after production examples | Performance, holdings, price, and reporting effects of corrections. |

---

## 17. Bottom Line

The supplied material is sufficient to write a conservative Chapter 09, but not sufficient to document complete corporate-action mechanics.

The strongest supported material is:

1. Axys has a `split.inf` securities splits file in the supplied conversion evidence.
2. Axys conversion packages commonly include `.cli`, `sec.inf`, `split.inf`, `.pri`, and `type.inf`.
3. Third-party conversion tools may materialize split history into explicit transactions.
4. ACA is verified as an Advent corporate-actions workflow for both Axys and APX.
5. Axys ACA evidence includes active holdings by daily script, reports, Automation Results email, simple/mandatory events processing to Trade Blotter, and complex/option events requiring review.
6. APX ACA evidence includes APX holdings handoff, ACA action records, review/download, APX Reorg Utility, and APX Trade Blotter.
7. Many implementation-critical details remain Unknown: field layouts, transaction codes, IMEX object names, REP report names, ACA action IDs, final posting behavior, cost-basis/taxability storage, and performance impact.

Use this chapter as a disciplined reference boundary: it documents what is known, identifies where corporate actions touch adjacent data areas, and prevents unsupported Axys/APX behavior from being invented.
