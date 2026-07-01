# Chapter 05 Transactions — Consolidated Research Reference

> **Date consolidated:** 2026-06-29  
> **Purpose:** Single research reference for future drafting of `Chapter 05 — Transactions.md`.  
> **Scope:** Axys/APX-style portfolio accounting transactions, transaction interfaces, Trade Blotter workflows, IMEX/REP evidence, transaction codes, transaction fields, dependencies, audit rules, contradictions, and known unknowns.  
> **Status:** Consolidated research document. This is **not** the finished Chapter 05 and should not be treated as a normative vendor reference.

---

## 1. Executive Summary

This document consolidates the supplied Chapter 05 transaction research into one coherent research reference. The source material consisted of preliminary research notes, deep document mining, a source catalog, lifecycle notes, processing-pipeline notes, dependency analysis, field-dictionary drafts, code-matrix drafts, audit-rule drafts, known-unknowns tracking, and a later evidence-hunt addendum.

The central research conclusion is that **transactions are the core accounting events in Axys/APX-style portfolio accounting systems**. They connect external economic events to holdings, cash, tax lots, cost basis, realized gain/loss, income, performance, reporting, reconciliation, IMEX exports/imports, and audit workflows.

Public evidence supports a practical transaction lifecycle involving:

```text
Economic event
        ↓
External source-data
        ↓
Normalization and translation
        ↓
Validation
        ↓
Trade Blotter or staging area
        ↓
Review / exception handling
        ↓
Posting
        ↓
Accounting records updated
        ↓
Holdings, cash, lots, cost basis, income, and realized gain/loss updated
        ↓
Performance, reports, IMEX, REP, reconciliation, and audit
```

The strongest public evidence concerns:

- Axys/APX transaction workflows involving Trade Blotters and IMEX.
- APX third-party transaction translation tables and observed transaction codes.
- Cancellation/reversal behavior using uppercase transaction codes in third-party Axys/APX integration workflows.
- Reinvestment activity represented as paired transactions.
- Fee handling through special security types/symbols and configurable translation.
- Axys `.cli` references, cost-basis caveats, deliver-in/out interpretation caveats, and conversion complications.
- APX Transaction Summary Report existence and sample report columns.
- The need to interpret transactions using code, sign, security type, source/destination fields, symbol, configuration, and context rather than code alone.

The weakest remaining areas are still:

- Complete official Axys transaction-code matrix.
- Complete official APX transaction-code matrix.
- Official IMEX transaction object names.
- Complete official trade blotter field layout.
- Native Axys transaction storage model.
- Native APX database transaction schema.
- Native transaction audit trail.
- Native posting-status model and state transitions.
- Exact REP parameters and official transaction report specifications.

The future Chapter 05 should therefore distinguish carefully between:

1. General accounting principles.
2. Observed Axys/APX behavior.
3. Third-party integration behavior.
4. Migration/conversion behavior.
5. Official vendor-supported behavior.
6. Unknown or unverified behavior.

---

## 2. Research Scope and Evidence Rules

### 2.1 Research Objective

The purpose of this research package is to support a future Chapter 05 on portfolio accounting transactions. It is intended to collect and organize evidence, not to assert final vendor-specific behavior.

The future chapter should be based on this document but should not simply copy it. This document intentionally preserves research caveats, source confidence levels, contradictions, and open questions.

### 2.2 Governing Rules Used in the Research

The source research applied the following rules:

- Do not invent unsupported Axys or APX behavior.
- Treat third-party integration manuals as evidence, but not as official vendor documentation.
- Distinguish direct source statements from interpretation.
- Assign confidence levels.
- Preserve unknowns.
- Do not treat observed transaction-code examples as complete official transaction-code matrices.
- Do not treat integration workflows as necessarily equivalent to native system internals.

### 2.3 Confidence Levels

| Confidence | Meaning |
|---|---|
| Verified | Official vendor documentation or reproducible production evidence. |
| High | Strong agreement across multiple independent sources or strong accounting consensus. |
| Medium | Supported by one or more credible third-party sources. |
| Low | Limited or indirect evidence. |
| Unknown | Not yet verified. |

### 2.4 Source-Type Confidence

| Source Type | Default Confidence | Notes |
|---|---:|---|
| Official SS&C/Advent vendor material | High | Product pages may be high for capabilities but low for mechanics. |
| Vendor report samples/guides | Medium to High | Stronger when full report content is available. |
| Third-party integration manuals | Medium | Useful for operational evidence, but may reflect integration-layer behavior. |
| Third-party migration guides | Medium | Useful for conversion behavior and caveats; not necessarily native semantics. |
| Consultant articles | Medium | Useful for architectural context; should be corroborated. |
| Search-result snippets or partial references | Low | Should not be promoted without full review. |

---

## 3. Source Catalog and Evidence Quality

### 3.1 Sources Reviewed

| ID | Source | Type | System | Topics Covered | Confidence | Notes |
|---:|---|---|---|---|---|---|
| SRC-001 | SS&C Advent Axys Product Page | Vendor product page | Axys | Portfolio accounting, reporting, performance measurement, reconciliation, transactions, positions, settlement/trade information, tax-lot or average-cost accounting, trade-date or settlement-date accounting. | High for product capabilities; Low for detailed mechanics | Confirms broad product capabilities, not implementation details. |
| SRC-002 | AdventGuru — Getting Data In and Out of Advent APX and Axys | Consultant article | Axys/APX | IMEX, Trade Blotter, import/export concepts, Replang, reports, direct-file-access risks, APX SQL/database options. | Medium | Useful architectural context. |
| SRC-003 | WealthTechs AIA User Manual — APX Users | Third-party integration manual | APX | Trade Blotters, Statement Blotters, Tax Lot Blotters, Position Blotters, Account Blotters, transaction translation, cancellation, comments, broker fields, processing order, downloaded data categories. | Medium | Major research source. |
| SRC-004 | WealthTechs AIA User Manual — Axys Users | Third-party integration manual | Axys | Transaction cancellation, IMEX workflow, blotters, cancellation example. | Medium | Complements APX manual. |
| SRC-005 | ByAllAccounts Custodial Integrator — APX User Guide | Third-party integration manual | APX | Transaction translation tables, reversals, fees, imports, sign-dependent translation, source/destination fields, special security fields. | Medium | Strong evidence for integration behavior. |
| SRC-006 | ByAllAccounts Custodial Integrator — Axys User Guide | Third-party integration manual | Axys | Trade Blotter workflow, IMEX import process, `topost.trn`, `imex32.exe`, folder labels, IMEX logs, security/reference file usage. | Medium | Strong operational workflow evidence. |
| SRC-007 | Morningstar Office Advent Axys Conversion Guide | Third-party migration guide | Axys | Reinvestment, deliver-in/out interpretation, `.cli`, cost basis, fees, paydowns, transaction prices, historical prices, conversion caveats. | Medium | Valuable migration observations. |
| SRC-008 | Advent Portfolio Exchange Reports Guide | Vendor report guide / public PDF reference | APX | APX Transaction Summary Report existence. | Low to Medium | Full text access was limited in research. |
| SRC-009 | Wealth Management Reports / Advent report sample | Vendor or Advent report sample | APX / SSRS | Transaction Summary Report purpose and sample columns. | Medium | Supports REP/report section. |
| SRC-010 | AdventGuru — APX to Axys Conversion | Consultant article | Axys/APX | APX-exported CLI files mapped into Axys `topost.trn`; complexity of transaction mappings and tax lots. | Medium | Reinforces mapping complexity. |

### 3.2 Topic Coverage Matrix

| Topic | Primary Sources |
|---|---|
| Trade Blotter | SRC-003, SRC-004, SRC-005, SRC-006, SRC-010 |
| IMEX | SRC-002, SRC-004, SRC-006 |
| Transaction Translation | SRC-003, SRC-005 |
| Cancellation / Reversal | SRC-003, SRC-004, SRC-005 |
| Reinvestment | SRC-005, SRC-007 |
| Fees | SRC-005, SRC-007 |
| Deliver In / Deliver Out | SRC-005, SRC-007 |
| `.cli` References | SRC-003, SRC-006, SRC-007, SRC-010 |
| Processing Pipeline | SRC-003, SRC-005, SRC-006 |
| Reporting / REP | SRC-001, SRC-008, SRC-009 |
| Direct File Access Risks | SRC-002 |
| APX SQL/Database Access | SRC-002 |

### 3.3 Strongest Sources

The strongest transaction-specific public sources found were:

1. WealthTechs AIA User Manual for APX Users.
2. ByAllAccounts Custodial Integrator APX User Guide.
3. ByAllAccounts Custodial Integrator Axys User Guide.
4. Morningstar Office Advent Axys Conversion Guide.

These contain the richest operational and transaction-specific evidence, although they remain third-party integration or migration sources rather than official native Axys/APX manuals.

### 3.4 Weakest Evidence Areas

Public research remains weak for:

- Official Axys transaction codes.
- Official APX transaction codes.
- Official IMEX object definitions.
- Official trade blotter layouts.
- REP report parameters.
- Native Axys file storage model.
- Native APX database transaction schema.
- Native audit trail and transaction identifiers.

### 3.5 Promotion Guidance

| Evidence Type | Promotion Treatment |
|---|---|
| Official SS&C/Advent documentation | Can be promoted to Verified. |
| Multiple independent production observations | Can be promoted to High Confidence. |
| Single integration guide | Keep as Medium Confidence. |
| Consultant opinion only | Keep as Low or Medium depending on corroboration. |
| Search-result snippet only | Keep Low until full source reviewed. |

---

## 4. Conceptual Transaction Model

### 4.1 What a Transaction Represents

A transaction is an accounting event that changes or explains the state of a portfolio. In portfolio accounting, transactions are the bridge between economic events and accounting records.

Examples include:

- Buy.
- Sell.
- Short sale.
- Cover short.
- Deposit.
- Withdrawal.
- Dividend.
- Interest.
- Reinvested dividend.
- Return of capital.
- Principal paydown.
- Fee.
- Transfer in.
- Transfer out.
- Split.
- Journal/comment.
- Cancellation/reversal.

### 4.2 Transaction Centrality

Transactions sit at the center of the portfolio accounting data model.

```text
Security Master
Portfolio Master
Currencies
Pricing
Corporate Actions
        ↓
Transactions
        ↓
Holdings
Cash
Tax Lots
Cost Basis
Income
Realized Gain/Loss
Performance
Reports
IMEX
REP
Audit
```

### 4.3 General Accounting Effects by Conceptual Category

| Category | Typical Accounting Effect | Confidence |
|---|---|---:|
| Buy | Increase holdings, decrease cash. | High |
| Sell | Decrease holdings, increase cash, realize gain/loss. | High |
| Short Sale | Create or increase short exposure and related cash/margin accounting. | Medium |
| Cover Short | Reduce short exposure. | Medium |
| Deposit | Increase cash. | High |
| Withdrawal | Decrease cash. | High |
| Dividend | Increase income and cash. | High |
| Reinvested Dividend | Increase holdings and income; may be represented as paired income and buy transactions. | High conceptually; Medium source evidence for paired representation. |
| Interest | Increase income and cash. | High |
| Fee | Reduce cash and portfolio value. | High |
| Transfer In | Increase holdings and/or cash without normal purchase economics. | High |
| Transfer Out | Decrease holdings and/or cash without normal sale economics. | High |
| Split | Change quantity without changing market value. | High |
| Return of Capital | Reduce cost basis and/or distribute cash. | Medium |
| Principal Paydown | Reduce principal exposure, commonly bond or mortgage-backed security related. | Medium |
| Journal / Comment | Operational notation or non-standard movement depending on system context. | Medium |
| Cancellation / Reversal | Remove, reverse, or cancel a prior transaction. | Medium as observed behavior. |

### 4.4 Key Design Principle: Code Alone Is Not Enough

The research repeatedly supports the principle that transaction codes should not be interpreted in isolation.

Transaction meaning may depend on:

- transaction code,
- sign of quantity,
- sign of amount,
- security type,
- source/destination type,
- source/destination symbol,
- special security type/symbol,
- portfolio/account mapping,
- custodian-specific translation rules,
- integration configuration,
- `.cli` or client-file settings,
- whether the transaction is part of a paired transaction,
- whether the transaction is a cancellation/reversal.

This principle is strongly supported as a design recommendation, even though some details are based on third-party integration evidence.

---

## 5. Transaction Lifecycle

### 5.1 High-Level Lifecycle

The conceptual lifecycle begins when an economic event occurs and ends when that event has been validated, posted, reflected in accounting records, made available for reporting/export, and incorporated into downstream processes such as performance.

```text
Economic Event
        ↓
External Source
        ↓
Data Translation
        ↓
Validation
        ↓
Trade Blotter / Staging
        ↓
Review / Exceptions
        ↓
Posting
        ↓
Accounting Records Updated
        ↓
Holdings Updated
        ↓
Cash Updated
        ↓
Lots / Cost Basis / Income / Realized Gain-Loss Updated
        ↓
Performance Impact
        ↓
Reports / IMEX / REP / Reconciliation / Audit
```

### 5.2 Stage 1 — Economic Event

Economic events occur outside Axys/APX and include purchases, sales, dividends, interest, deposits, withdrawals, fees, transfers, corporate actions, and corrections.

**Confidence:** High.

### 5.3 Stage 2 — External Source

Observed or expected sources include:

- Custodian files.
- Broker files.
- OMS data.
- Manual entry.
- Corporate action providers.
- Aggregation platforms.
- Conversion/migration files.

Public integration guides demonstrate imports from custodians and external systems.

**Confidence:** Medium.

### 5.4 Stage 3 — Normalization and Translation

Translation may include:

- Portfolio translation.
- Security translation.
- Transaction-type translation.
- Broker translation.
- Fee translation.
- Currency translation.
- Cash-symbol translation.
- Custodian-specific mapping.
- Direction determination from sign.
- Creation of paired transactions.

Integration software frequently performs these steps before import.

**Confidence:** Medium.

### 5.5 Stage 4 — Validation

Candidate validation checks include:

- Portfolio exists.
- Security exists.
- Transaction code is valid or mapped.
- Trade date exists where required.
- Settlement date is valid.
- Settlement date does not precede trade date.
- Quantity exists where required.
- Price exists where required.
- Currency and FX rate are valid where required.
- Duplicate transactions are detected.
- External transaction identifiers are unique where available.
- Cancellation/reversal matches the original sufficiently.

Public integration documentation explicitly supports portfolio and security mapping validation.

**Confidence:** Medium for Axys/APX integration evidence; High as general accounting control design.

### 5.6 Stage 5 — Trade Blotter / Staging

Trade Blotters appear in multiple public integration guides. In observed workflows, transactions may be loaded into a blotter before posting.

Typical blotter activities include:

- Review.
- Exception handling.
- Corrections.
- Cancellation.
- Approval.
- Posting preparation.

Exact native Axys and APX workflow details remain unknown.

**Confidence:** Medium.

### 5.7 Stage 6 — Posting

Posting conceptually commits the transaction into accounting records.

Expected downstream effects include:

- Holdings.
- Cash.
- Tax lots.
- Cost basis.
- Income.
- Realized gains/losses.
- Performance inputs.
- Reporting outputs.

Exact posting mechanics remain unknown.

**Confidence:** Medium for native mechanics; High for conceptual accounting role.

### 5.8 Stage 7 — Holdings

Typical holdings effects:

| Transaction | Expected Holdings Impact |
|---|---|
| Buy | Increase quantity. |
| Sell | Decrease quantity. |
| Transfer In | Increase quantity. |
| Transfer Out | Decrease quantity. |
| Split | Change quantity without changing market value. |
| Reinvestment | Increase quantity, usually tied to income. |
| Principal Paydown | May reduce principal/position exposure depending on security type. |

**Confidence:** High conceptually.

### 5.9 Stage 8 — Cash

Typical cash effects:

| Transaction | Expected Cash Impact |
|---|---|
| Buy | Reduce cash. |
| Sell | Increase cash. |
| Deposit | Increase cash. |
| Withdrawal | Decrease cash. |
| Dividend | Increase cash unless reinvested. |
| Interest | Increase cash unless reinvested/accrued differently. |
| Fee | Reduce cash. |
| Return of Capital | Increase cash and may reduce cost basis. |

Settlement timing may differ from trade timing.

**Confidence:** High conceptually.

### 5.10 Stage 9 — Lots, Cost Basis, Income, and Realized Gain/Loss

Certain transactions create, modify, or consume tax lots. Cost basis is affected by purchases, sales, transfers, return of capital, and some corporate actions. Income transactions affect income and may also affect cash. Sales and other disposals may generate realized gain/loss.

Unknowns include native Axys/APX lot algorithms, partial lot disposal representation, and exact internal links between transaction records and tax lots.

**Confidence:** Medium.

### 5.11 Stage 10 — Performance Impact

Transactions influence performance indirectly through:

- beginning capital,
- ending capital,
- external cash flows,
- income,
- holdings,
- prices,
- timing assumptions,
- historical restatements.

Historical transaction edits may restate performance.

**Confidence:** High conceptually; exact Axys/APX implementation belongs in Chapter 10.

### 5.12 Stage 11 — Reporting, IMEX, REP, Reconciliation, and Audit

Observed and expected outputs include:

- Transaction reports.
- Holdings reports.
- Cash reports.
- Performance reports.
- IMEX exports.
- REP reports.
- Reconciliation reports.
- Audit outputs.

Exact report-generation sequence remains unknown.

**Confidence:** Medium.

### 5.13 Historical Corrections

Observed correction concepts include:

- Cancellation.
- Reversal.
- Correction.
- Re-import.
- Uppercase cancellation codes in third-party workflows.
- Historical transaction files used to create cancellation blotters.

Native historical audit models remain unknown.

**Confidence:** Medium.

### 5.14 Performance Restatement Risk

Historical transaction changes may affect:

- Holdings.
- Cash.
- Tax lots.
- Cost basis.
- Realized gain/loss.
- Income.
- Performance.
- Reports.
- Reconciliation.

**Confidence:** High.

---

## 6. Transaction Processing Pipeline

### 6.1 Conceptual Pipeline

The processing pipeline focuses on transformation from raw source-data to posted accounting records.

```text
Acquire Source-data
        ↓
Normalize Records
        ↓
Translate Portfolio
        ↓
Translate Security
        ↓
Translate Transaction Type
        ↓
Apply Special Processing Rules
        ↓
Validate
        ↓
Stage / Trade Blotter
        ↓
Review / Exceptions
        ↓
Post
        ↓
Update Holdings
        ↓
Update Cash
        ↓
Update Lots / Cost Basis / Income / Realized Gain-Loss
        ↓
Performance Impact
        ↓
Reporting / IMEX / REP / Reconciliation / Audit
```

### 6.2 Pipeline Stages and Failure Points

| Stage | Purpose | Typical Failure |
|---|---|---|
| Acquire Source-data | Obtain transactions from custodian, broker, OMS, manual entry, or provider. | Missing file, stale file, incomplete batch. |
| Normalize | Convert source formats into common representation. | Bad dates, inconsistent signs, malformed identifiers. |
| Portfolio Translation | Map external account to internal portfolio. | Unknown portfolio, duplicate mapping, inactive account. |
| Security Translation | Map ticker/CUSIP/ISIN/source security to internal security. | Unknown security, ambiguous security, duplicate mapping. |
| Transaction Translation | Map external transaction type to accounting transaction type/code. | Unsupported transaction, wrong direction, missing paired leg. |
| Special Processing | Apply sweeps, FX merge, accrued-interest merge, fee translation, tax logic, cancellation handling. | Unintended suppression, bad merge, incorrect fee classification. |
| Validation | Verify required fields and accounting plausibility. | Missing quantity, missing price, invalid dates, bad FX. |
| Staging | Hold records in Trade Blotter or equivalent. | Review exception, cancellation mismatch. |
| Posting | Commit to accounting records. | Posting failure, partial batch, unresolved dependency. |
| Downstream Updates | Update holdings, cash, lots, cost basis, income, realized gain/loss. | Holdings mismatch, cash mismatch, lot inconsistency. |
| Reporting / Export | Expose transactions and derived records. | REP/IMEX/report differences, reconciliation issues. |

### 6.3 AIA APX Processing Order Observed in Public Evidence

The WealthTechs APX guide gives an explicit logical order for applying translations and filters to source files. Relevant transaction-related steps include:

| Order | Step | Applies To | Confidence |
|---:|---|---|---:|
| 3 | Remove Pending Records | All files | Medium |
| 4 | Remove Intra-Account Journals | Transactions | Medium |
| 5 | Remove Cash Sweeps | Transactions | Medium |
| 6 | Withholding Tax Logic | Transactions | Medium |
| 7 | Merge FX Transactions | Transactions | Medium |
| 8 | Merge Accrued Interest Transactions | Transactions | Medium |
| 9 | Transaction Translations | Transactions | Medium |
| 12 | Broker Translations | Transactions | Medium |
| 15 | Adjust Cancel Transactions to Upper Case | Transactions | Medium |
| 16 | Apply Transaction Comment Logic | Transactions | Medium |
| 17 | Merge Dividends and Interest | Transactions | Medium |
| 19 | Post Translations Transaction Translations | Transactions | Medium |
| 23 | Add Interface Comments | All files | Medium |

### 6.4 Interpretation of AIA Processing Order

This processing order shows that real-world transaction imports are not simple file loads. They may include:

- pending-record removal,
- journal filtering,
- cash sweep filtering,
- withholding-tax logic,
- FX transaction merging,
- accrued-interest merging,
- transaction translation,
- broker translation,
- cancellation handling,
- comment logic,
- dividend/interest merging,
- post-translation cleanup,
- interface comments.

This should be treated as AIA processing behavior, not confirmed APX native processing order.

### 6.5 Candidate Pipeline Audit Rules

| Rule | Description |
|---|---|
| TP-001 | Every source transaction should reach a terminal state: posted, rejected, canceled, or explicitly ignored. |
| TP-002 | Portfolio must exist before posting. |
| TP-003 | Security must exist before posting. |
| TP-004 | Posting should update holdings consistently. |
| TP-005 | Posting should update cash consistently. |
| TP-006 | Historical corrections should trigger downstream review. |
| TP-007 | Import batches should not be partially processed without explicit exception reporting. |
| TP-008 | Duplicate detection should use portfolio, security, date, quantity, price, amount, and external identifier where available. |

---

## 7. Transaction Dependencies

### 7.1 Upstream Dependencies

Transactions require upstream reference data and configuration.

#### 7.1.1 Portfolio Master

Transactions require a valid portfolio or account.

Typical dependency:

- Portfolio exists.
- Portfolio is active or eligible for posting.
- Base currency is known.
- External account maps to internal portfolio.

Failure examples:

- Unknown portfolio.
- Closed portfolio.
- Missing account translation.
- Duplicate mapping.

**Confidence:** High.

#### 7.1.2 Security Master

Most security transactions require a valid security.

Typical dependency:

- Internal security ID.
- Ticker/CUSIP/ISIN translation.
- Security type.
- Currency.
- Duplicate/ambiguous match handling.

Failure examples:

- Unknown security.
- Duplicate security.
- Ambiguous CUSIP/ticker mapping.
- Missing security type.

**Confidence:** High conceptually; Medium for specific integration evidence.

#### 7.1.3 Currency and FX

Multi-currency transactions may require:

- Transaction currency.
- Portfolio base currency.
- Settlement currency.
- FX rate.
- FX-rate date.

Unknown:

- Exact Axys/APX FX processing sequence.
- How FX rates are stored natively.
- How cross-currency settlements are represented.

**Confidence:** Medium.

#### 7.1.4 Pricing

Some transaction types require execution price.

Typical examples:

- Buy.
- Sell.
- Short sale.
- Cover short.

Other transactions such as deposits and withdrawals generally do not require security price.

Unknown:

- Whether stored transaction price always equals historical pricing database value.
- Native Axys/APX storage precision.

**Confidence:** High conceptually; Medium for native field behavior.

#### 7.1.5 Corporate Actions

Corporate actions may generate transactions or alter interpretation.

Examples:

- Splits.
- Return of capital.
- Bond paydowns.
- Calls.
- Reorganizations.

**Confidence:** Medium.

#### 7.1.6 Configuration and Translation Rules

Integration evidence shows dependencies on:

- portfolio-code translations,
- security translations,
- broker translations,
- transaction translations,
- fee translations,
- special security types/symbols,
- cash symbols,
- custodian-specific translation rules,
- `.cli` or client-file settings.

**Confidence:** Medium.

### 7.2 Downstream Dependencies

#### 7.2.1 Holdings

Transactions are a primary source of position changes.

Examples:

- Buy increases holdings.
- Sell decreases holdings.
- Transfer moves holdings.
- Split changes quantity.
- Paydown may reduce principal.

**Confidence:** High.

#### 7.2.2 Cash

Transactions drive cash movements.

Examples:

- Deposits.
- Withdrawals.
- Buy settlement.
- Sell settlement.
- Fees.
- Income.

Settlement timing may differ from trade timing.

**Confidence:** High.

#### 7.2.3 Tax Lots

Certain transactions create or modify tax lots.

Examples:

- Buy.
- Sell.
- Transfer.
- Corporate action.
- Initial deliver-in.

Unknown:

- Native Axys/APX lot algorithms.
- How tax lots link to transactions.
- How partial lot disposals are represented.

**Confidence:** Medium.

#### 7.2.4 Cost Basis

Cost basis is affected by:

- purchases,
- sales,
- transfer lots,
- return of capital,
- certain corporate actions,
- reinvestments,
- fees depending on treatment.

**Confidence:** High conceptually.

#### 7.2.5 Performance

Performance calculations consume posted accounting data.

Potential dependencies:

- cash flows,
- holdings,
- prices,
- income,
- transaction timing,
- historical transaction changes.

Historical transaction edits may restate performance.

**Confidence:** High.

#### 7.2.6 Reporting, IMEX, REP, and Audit

Transactions feed:

- transaction reports,
- holdings reports,
- cash reports,
- performance reports,
- IMEX exports,
- REP reports,
- reconciliation,
- audit rules.

**Confidence:** Medium.

### 7.3 Cross-Chapter Dependencies

| Chapter | Relationship | Confidence |
|---|---|---:|
| 03 Portfolios | Transactions require portfolios. | High |
| 04 Security Master | Transactions reference securities. | High |
| 06 Holdings | Holdings derived from transactions. | High |
| 07 Cash | Cash ledger updated by transactions. | High |
| 08 Pricing | Certain transactions require prices. | High |
| 09 Corporate Actions | Some corporate actions generate transactions or affect interpretation. | Medium |
| 10 Performance | Transactions influence performance. | High |
| 11 Classifications | Transactions inherit security classifications indirectly. | Medium |
| 12 IMEX | Transaction import/export interface. | Medium |
| 13 REP | Transaction reporting interface. | Medium |
| 14 Audit Rules | Transactions are a primary audit source. | High |

### 7.4 Dependency Risk Matrix

| Dependency | Missing or Bad Result |
|---|---|
| Portfolio | Cannot post transaction or posts to wrong account. |
| Security | Cannot identify asset or misclassifies asset. |
| Price | Valuation, cost basis, and realized gain/loss issues. |
| FX Rate | Currency errors and incorrect base-currency values. |
| Holdings | Position mismatch. |
| Cash | Ledger mismatch. |
| Tax Lots | Realized gain/loss and cost-basis errors. |
| Performance | Return restatement or inaccurate performance. |
| Reports | Reporting inconsistency. |
| Audit Trail | Inability to reconstruct history. |

---

## 8. Transaction Translation, Trade Blotters, IMEX, REP, and Direct Access

### 8.1 Translation Layer

Public integration evidence shows that source transactions may be normalized before being translated into Axys/APX transaction records.

Translation may involve:

- source transaction type,
- APX/Axys transaction code,
- sign of units or amount,
- source/destination type,
- source/destination symbol,
- special security type/symbol,
- fee description text,
- broker representative field,
- security type,
- custodian-specific rules,
- source-file configuration,
- account/portfolio mapping.

### 8.2 Axys Data Translation and Trade Blotter Workflow

The ByAllAccounts Axys guide describes a workflow where external WebPortfolio data is downloaded, merged with Axys security information, and converted into:

- transaction Trade Blotter file,
- position file,
- price file.

These files are imported into Axys using the Axys Import/Export utility. Transactions are delivered to a designated Trade Blotter for review and posting.

Observed Axys workflow:

```text
External financial institution data
        ↓
Aggregation / normalization layer
        ↓
Security and portfolio translation
        ↓
Transaction Trade Blotter file
        ↓
IMEX import
        ↓
Trade Blotter review
        ↓
Post to Axys
```

**Confidence:** Medium. This is a third-party integration workflow, not confirmed as the only native Axys workflow.

### 8.3 Axys Trade Blotter File: `topost.trn`

The later evidence hunt found additional Axys-specific details from the ByAllAccounts Axys guide.

| Item | Observed Value | Role | Confidence | Caveat |
|---|---|---|---:|---|
| Trade Blotter file | `topost.trn` | Receives transaction imports. | Medium | Third-party integration evidence. |
| Folder label | `$pathtrn` | Axys user folder label for Trade Blotter location. | Medium | Third-party integration evidence. |
| Behavior | Generated transactions appended to Trade Blotter | Existing transactions left unchanged. | Medium | Workflow-specific. |

This evidence is strong enough to include in future Chapter 05 as observed Axys-oriented integration behavior, not as verified native storage documentation.

### 8.4 Axys IMEX Executable and Logs

The ByAllAccounts Axys guide says Custodial Integrator looks for `imex32.exe`, described as the Axys Import/Export utility, and references IMEX logs generated during import.

| Item | Observed Value | Confidence | Caveat |
|---|---|---:|---|
| Executable | `imex32.exe` | Medium | Third-party integration documentation. |
| Description | Axys Import/Export utility | Medium | Source-specific wording. |
| Related logs | IMEX logs | Medium | Exact log fields unknown. |

### 8.5 Axys Supporting Files and Folder Labels

The evidence hunt identified several Axys folder labels and files relevant to transaction generation.

| Folder / File | Observed Role | Confidence | Caveat |
|---|---|---:|---|
| `$pathcli` | Axys portfolio/client files; `*.cli`; used to create portfolio-code list. | Medium | Integration workflow evidence. |
| `$pathinf` | Contains `sec.inf` and `type.inf`; exported by CI to generate transactions and positions. | Medium | Integration workflow evidence. |
| `$pathpri` | Axys price-file folder; `*.pri`. | Medium | Integration workflow evidence. |
| `$pathlog` | Folder where Axys Import/Export logs are written. | Medium | Integration workflow evidence. |
| `topost.trn` | Trade Blotter file receiving transaction imports. | Medium | Integration workflow evidence. |

This does not fully define native Axys transaction storage.

### 8.6 APX Transaction Translation Model

The ByAllAccounts APX guide states that WebPortfolio interprets financial-institution transaction data, assigns a normalized WebPortfolio transaction type, and Custodial Integrator translates those normalized transaction types into APX transactions.

The guide also states that some translations depend on the sign of amount or units. For example, a security transfer with negative units is translated as APX `lo`, while positive units translate as APX `li`.

Extracted facts:

| Fact | System | Confidence |
|---|---|---:|
| Source transactions are normalized before APX transaction generation. | APX | Medium |
| Transaction sign may determine direction of cash or security movement. | APX | Medium |
| Positive-unit transfer maps to APX `li` in the ByAllAccounts default translation. | APX | Medium |
| Negative-unit transfer maps to APX `lo` in the ByAllAccounts default translation. | APX | Medium |
| Translation table is general case and may be subject to special cases and customization by financial institution. | APX | Medium |

### 8.7 APX Observed Source/Destination and Special Security Fields

The ByAllAccounts APX guide explicitly names the following fields:

- APX Transaction Type.
- APX Transaction Src/Dest Type.
- APX Transaction Src/Dest Symbol.
- APX Transaction Special Security Type.
- APX Transaction Special Security Symbol.

The same guide states that fee transactions may use special security type `exus` or `epus` with fee symbols such as `custfee` or `expense`.

| Field | Observed? | Confidence | Caveat |
|---|---|---:|---|
| Transaction Type | Yes | Medium | Integration guide field. |
| Src/Dest Type | Yes | Medium | Integration guide field. |
| Src/Dest Symbol | Yes | Medium | Integration guide field. |
| Special Security Type | Yes | Medium | Integration guide field. |
| Special Security Symbol | Yes | Medium | Integration guide field. |
| `exus` / `epus` | Observed special security types or labels | Medium | Need vendor confirmation. |
| `custfee` / `expense` | Observed fee symbols | Medium | Need vendor confirmation. |

### 8.8 APX Blotter Types

The WealthTechs APX manual identifies multiple APX blotter concepts:

| Blotter | Purpose in Source | Confidence |
|---|---|---:|
| Trade Blotter | AIA imports transactions into this blotter; can be consolidated or created per custodian. | Medium |
| Statement Blotter | Used to post statement transactions from custodians; can support reconciliation against OMS or third-party data using REX. | Medium |
| Tax Lot Blotter | Used for tax-lot-level reconciliation of APX calculated lots versus custodian lots. | Medium |
| Position Blotter | Used for importing positions into APX. | Medium |
| Account Blotter | Used for importing account information. | Medium |
| Initial Transaction Blotter | Used to import positions as deliver-in transactions for accounts without transactions, when configured. | Medium |

### 8.9 Initial Deliver-In Transactions

The WealthTechs APX manual says that when `Create Initial Deliver-In Transactions From Positions` is checked, AIA checks APX; if the account has no transactions, positions are written to the initial transaction blotter. If transactions are received on the same day as initial positions, the transactions are ignored and not written to the blotter.

Extracted facts:

| Fact | System | Confidence |
|---|---|---:|
| AIA can create initial deliver-in transactions from positions for accounts with no transactions. | APX | Medium |
| Same-day source transactions may be ignored in that initial-position scenario. | APX | Medium |
| Tax lots may be relevant to initial deliver-in construction. | APX | Low to Medium |

### 8.10 Trade Blotter Organization

The WealthTechs APX guide describes options for Trade Blotter logic:

| Option | Meaning | Confidence |
|---|---|---:|
| Consolidate Into One Blotter | Aggregates all transactions from all custodians into one trade blotter. | Medium |
| Create One Blotter Per Custodian | Distributes transactions into one blotter per custodian. | Medium |
| No Trade Blotter | No trade blotter is created by AIA. | Medium |

### 8.11 Statement Blotter and Reconciliation

The WealthTechs APX guide states that the Statement Blotter can be used to post custodian statement transactions and that APX users may reconcile these against portfolio transactions or OMS transactions using REX.

The guide also states that statement blotter transactions can be viewed in an APX portfolio tab named `Statement Transactions`, while transactions posted through the Trade Blotter can be viewed in the portfolio tab `Transactions`.

Extracted facts:

| Fact | System | Confidence |
|---|---|---:|
| APX workflows may distinguish posted portfolio transactions from statement transactions. | APX | Medium |
| Statement transactions may support reconciliation against custodian or OMS data. | APX | Medium |
| APX may expose separate UI tabs for `Transactions` and `Statement Transactions` in this workflow. | APX | Medium |

### 8.12 Transaction Comments

The WealthTechs APX guide states that rules set up in Transaction Translation apply only to transaction comments like one example in the manual, while standalone comments always post to the import transaction file.

Extracted facts:

| Fact | System | Confidence |
|---|---|---:|
| APX transaction import files may include transaction comments. | APX | Medium |
| Some comments are conditional under translation logic. | APX | Medium |
| Standalone comments always post to the import transaction file in this workflow. | APX | Medium |

### 8.13 Broker Representative Field

The WealthTechs APX guide describes a `Use $brok` setting that writes `$brok` to the broker representative field in the transaction blotter. It states that this is typically defined in the `.cli` file in APX for each portfolio. It also describes broker translations that map the broker representative field to a unique code set up in APX.

Extracted facts:

| Fact | System | Confidence |
|---|---|---:|
| APX transaction blotter may have a broker representative field. | APX | Medium |
| AIA can write `$brok` into the broker representative field. | APX | Medium |
| `$brok` is described as typically defined in the `.cli` file for each APX portfolio. | APX | Medium |
| Broker translations can map broker representative values to APX-specific codes. | APX | Medium |

### 8.14 Lot Location

The WealthTechs APX guide describes lot location as an old Axys carryover that allowed tracking multiple custodians in one account and says it is integrated into lot accounting. It also documents a setting to change lot location, with `250` described as a default source-file value.

Extracted facts:

| Fact | System | Confidence |
|---|---|---:|
| Lot location is described as an Axys-era concept used to track multiple custodians in one account. | Axys/APX | Medium |
| Lot location is relevant to lot accounting in the AIA/APX workflow. | APX | Medium |
| AIA can modify lot location values during processing. | APX | Medium |
| `250` is described as a default source-file value in the WealthTechs context. | APX | Medium |

### 8.15 Cash Sweeps, Margin Sweeps, Short Sweeps, and Double-Entry Notes

The WealthTechs APX guide states that WealthTechs uses double-entry accounting transactions and that users do not need to track cash and money market fund balances in the portfolio accounting system unless they want to.

It describes rules for removing cash sweeps from source transaction files based on `WD` or `DP` transaction codes with cash-related security/source-destination types and symbols.

Extracted facts:

| Fact | System | Confidence |
|---|---|---:|
| AIA includes logic to remove cash sweep transactions from source transaction files. | APX | Medium |
| AIA has similar sweep removal logic for margin and short sweeps. | APX | Medium |
| Source examples include `DP,CAUS,CASH,CAUS,MMF`, `DP,CAUS,CASH,CAUS,MARGIN`, and `DP,CAUS,CASH,CAUS,SHORT`. | APX | Medium |
| AIA has options to merge FX transactions, merge accrued-interest transactions, and merge dividend/interest entries. | APX | Medium |

Design implication: cash-like transactions require special handling. A parser should not infer accounting impact from code alone. Source/destination fields, cash symbols, margin symbols, short symbols, and wash symbols matter.

### 8.16 IMEX

AdventGuru states that Axys v2.x introduced binary file formats and that IMEX allowed users to import and export CSV, tab, and fixed formats. It also states that IMEX combined with transaction and label import through the Trade Blotter provides a way to move fundamental data in and out of Axys and APX.

Extracted facts:

| Fact | System | Confidence |
|---|---|---:|
| IMEX supports CSV, tab, and fixed-format import/export in Axys according to consultant documentation. | Axys | Medium |
| IMEX plus Trade Blotter transaction import is described as a comprehensive way to move fundamental data in and out of Axys/APX. | Axys/APX | Medium |
| APX v1.x to v4.x maintained IMEX functionality, but fixed-format file generation was eliminated according to AdventGuru. | APX | Medium |
| APX users may also query the database and use SQL-based reporting/export tools. | APX | Medium |

### 8.17 Direct File Access

AdventGuru warns that direct Axys file access is not best practice because file formats can change between versions. It gives an example that upgrading from Axys v3.7 to v3.8 required file conversion and produced some files with different formats.

Design implication: Chapter 05 should distinguish:

- native transactions,
- Trade Blotter imports,
- IMEX export/import objects,
- REP/report output,
- direct file access,
- SQL/database access in APX.

### 8.18 REP and Transaction Reports

Evidence supports existence of an APX Transaction Summary Report.

A report sample states that the Transaction Summary Report displays all account transactions maintained by Advent and provides an independent record apart from the custodian.

Sample column groups observed:

| Section | Columns Observed |
|---|---|
| Dividends | Ex-Date, Pay-Date, Symbol, Security, Amount. |
| Contributions | Trade Date, Settle Date, Quantity, Symbol, Security, Unit Price, Amount. |
| Withdrawals | Trade Date, Settle Date, Quantity, Symbol, Security, Unit Price, Amount. |

Research treatment:

| Report | System | Description | Confidence |
|---|---|---|---:|
| Transaction Summary Report | APX / Advent reports | Displays account transactions maintained by Advent; sample columns include dates, quantity, symbol, security, unit price, and amount. | Medium |
| Exact REP parameters / report source code | APX | Unknown. | Unknown |

### 8.19 Report Writer, Replang, and Export Alternatives

AdventGuru lists report/export alternatives including:

- Excel export,
- Report Writer Pro,
- Replang reports,
- ETL products,
- APX SQL/database access.

This supports documenting IMEX as one interface rather than the only reporting/export path.

---

## 9. Transaction Code Matrix

### 9.1 Important Warning

The transaction-code matrix below is a research catalog of observed codes and conceptual meanings. It is **not** an official Axys or APX transaction-code reference.

Many rows are based on third-party integration manuals. Codes may be:

- official native codes,
- integration-layer translations,
- version-specific,
- configuration-dependent,
- context-dependent,
- incomplete.

Rows should be promoted to Verified only when supported by official vendor documentation or reproducible production evidence.

### 9.2 Observed Transaction Codes

| Code | Observed Meaning | Axys | APX | Source Type | Confidence | Notes |
|---|---|---|---|---|---:|---|
| `by` | Buy | Unknown / observed in examples | Observed | ByAllAccounts, WealthTechs | Medium | Public integration documentation only. |
| `BY` | Cancellation/deletion of Buy | Observed | Observed | WealthTechs, ByAllAccounts | Medium | Uppercase cancellation observed; native universality unknown. |
| `sl` | Sell | Unknown | Observed | ByAllAccounts | Medium | Requires vendor confirmation. |
| `ss` | Short sale | Unknown | Observed | ByAllAccounts | Medium | Requires vendor confirmation. |
| `cs` | Cover short | Unknown | Observed | ByAllAccounts | Medium | Added in later evidence hunt. |
| `li` | Deliver in / transfer in / credit / deposit / ATM positive / direct deposit | Observed | Observed | ByAllAccounts, Morningstar | Medium | Meaning may depend on sign/configuration. |
| `lo` | Deliver out / transfer out / debit / withdrawal / payment / ATM negative / direct debit | Observed | Observed | ByAllAccounts, Morningstar | Medium | Meaning may depend on sign/configuration. |
| `dv` | Dividend / income / reinvestment leg | Unknown | Observed | ByAllAccounts | Medium | Often paired with reinvestment. |
| `in` | Income / interest | Unknown | Observed | ByAllAccounts | Medium | Added in deep mining/evidence hunt. |
| `rc` | Return of capital | Unknown | Observed | ByAllAccounts | Medium | Requires vendor confirmation. |
| `pd` | Principal paydown / bond return-of-capital case | Unknown | Observed | ByAllAccounts | Medium | Bond-related. |
| `ai` | Accrued interest or margin interest | Unknown | Observed | ByAllAccounts | Medium | Requires context. |
| `sa` | Sell accrued interest | Unknown | Observed | ByAllAccounts | Medium | Requires vendor confirmation. |
| `pa` | Reinvested dividend / accrued-interest-related buy-like case | Unknown | Observed | ByAllAccounts | Low to Medium | Meaning requires further verification. |
| `dp` | Debit / fee-related / tax / service charge / cash-security case | Unknown | Observed | ByAllAccounts | Medium | Multiple meanings depending on context. |
| `wd` | Withdrawal / cash-security sell case | Unknown | Observed | ByAllAccounts | Medium | Requires context. |
| `;` | Journal / comment / other / split in integration table | Unknown | Observed | ByAllAccounts | Medium | Treat as observed integration behavior. |

### 9.3 APX Default Translation Patterns Observed

| Source Transaction Concept | APX Translation Pattern | Notes | Confidence |
|---|---|---|---:|
| ATM positive | `li` | Moves money/security into account in default table. | Medium |
| ATM negative | `lo` | Moves money/security out of account in default table. | Medium |
| Buy | `by` | Uses APX default cash account fields in table. | Medium |
| Cash security buy | `dp` | Special cash-security case. | Medium |
| Cover short | `cs` | Default table maps cover short to `cs`. | Medium |
| Check | `lo` | Withdrawal-like. | Medium |
| Closure positive | `sl` | Positive closure maps to sell in table. | Medium |
| Closure negative | `cs` | Negative closure maps to cover short in table. | Medium |
| Credit | `li` | Inflow-like. | Medium |
| Debit non-cash security | `lo` | Outflow-like. | Medium |
| Tax | `dp` with special type/symbol | Uses examples such as `epus` and `with`. | Medium |
| Deposit cash | `li` | Inflow-like. | Medium |
| Deposit non-cash security | `li` and `by` pair | Source shows two-transaction case. | Medium |
| Direct debit | `lo` | Outflow-like. | Medium |
| Direct deposit | `li` | Inflow-like. | Medium |
| Dividend | `dv` | Income-related. | Medium |
| Reinvested dividend | `dv` with `dvwash` and/or paired buy behavior | Requires further verification. | Medium |
| Fee | `dp` with `exus custfee` | Fee translation can be customized. | Medium |
| Recordkeeping fee | `dp` with `epus expense` | Source table example. | Medium |
| Income bond security positive/negative | `li` / `lo` | Direction depends on sign. | Medium |
| Income cash security | `in` | Income-like. | Medium |
| Income dividend-paying security | `dv` | Dividend-like. | Medium |
| Interest positive | `in` | Income-like. | Medium |
| Interest negative | `ai` | Margin-interest-like special case. | Medium |
| Investment expense | `dp` with `exus custfee` | Fee-like. | Medium |
| Journal | `;` | Comment/journal-like. | Medium |
| Margin interest | `ai` | Uses margin cash symbol in source. | Medium |
| Other | `;` | Generic/other. | Medium |
| Payment | `lo` | Outflow-like. | Medium |
| Point of sale positive/negative | `li` / `lo` | Direction depends on sign. | Medium |
| Reinvestment | `dv` and `by` pair | Source shows two-line APX translation. | Medium |
| Repeat payment | `lo` | Outflow-like. | Medium |
| Return of capital | `rc`; bond security maps to `pd` | Bond-specific behavior requires verification. | Medium |
| Sell | `sl` | Normal sell. | Medium |
| Sell cash security | `wd` | Cash-security special case. | Medium |
| Short | `ss` | Short sale. | Medium |
| Accrued interest on sell | `sa` | Source table maps accrued interest to `sa`. | Medium |
| Service charge | `dp` with `exus custfee` | Fee-like. | Medium |
| Split | `;` | Source maps split to semicolon/comment-like type. | Medium |
| Transfer positive/negative | `li` / `lo` | Direction depends on sign. | Medium |
| Withdrawal | `lo` | Outflow-like. | Medium |

### 9.4 Reversal / Cancellation Codes

Multiple third-party sources show cancellation/reversal by converting lower-case transaction code to upper-case transaction code, for example:

```text
by → BY
```

Observed example row:

```text
acct123,010101,010101,by,csus,appl,100,caus,cash,10000
```

becomes:

```text
acct123,010101,010101,BY,csus,appl,100,caus,cash,10000
```

Extracted facts:

| Fact | System | Confidence |
|---|---|---:|
| ByAllAccounts APX guide states reversal transactions are represented by uppercasing the original transaction code. | APX | Medium |
| The guide describes uppercase type code as APX representation for a transaction to be deleted. | APX | Medium |
| Field mismatch between reversal and original can cause an APX Trade Blotter error. | APX | Medium |
| WealthTechs APX guide can create a cancellation trade blotter from historical transaction files. | APX | Medium |
| WealthTechs APX cancellation logic uppercases transaction code, e.g., `by` to `BY`. | APX | Medium |
| Historical transaction-file list is derived from archive-folder files in that workflow. | APX | Medium |
| Workflow assumes transactions were previously posted to APX. | APX | Medium |
| Source recommends backing up accounts before cancellation. | APX | Medium |
| WealthTechs Axys guide documents analogous cancellation behavior for Axys users. | Axys | Medium |
| Public Axys-oriented example shows `by` becoming `BY`. | Axys | Medium |

Unknown: whether uppercase cancellation is universally valid across all Axys/APX versions, all transaction types, all import methods, and all native workflows.

### 9.5 Fee-Related Codes and Special Security Types

The ByAllAccounts APX guide states that fee transactions may use special security type `exus` or `epus` along with APX fee symbols such as `custfee` or `expense`.

It defines fee transactions for Custodial Integrator customization as:

1. WebPortfolio transaction type Fee, Investment Expense, or Service Charge.
2. Non-cash security Debit transactions that translate to APX `lo`.
3. `dp` or `wd` transactions generated during processing of other transaction types, such as a sell of a fund to pay a fee, generated as `sl` followed by `dp`.

Fee translation customization parameters observed:

| Parameter | Purpose | Confidence |
|---|---|---:|
| `defFeeType` | Default APX security type for fee transactions; example default `epus`. | Medium |
| `defFeeSymbol` | Default APX security symbol for fee transactions; example default `custfee`. | Medium |
| `xlateFeeDesc<n>` | Matches fee-description text. | Medium |
| `xlateFeeType<n>` | APX security type to use for matching fee transactions. | Medium |
| `xlateFeeSymbol<n>` | APX security symbol to use for matching fee transactions. | Medium |

Morningstar Axys conversion evidence:

| Fact | System | Confidence |
|---|---|---:|
| Multiple Axys fee types containing transaction code or label `epus` are converted as Management Fees in Morningstar Office. | Axys | Medium |
| Transaction codes or labels listed as `exus` are converted as Expenses in Morningstar Office. | Axys | Medium |
| Fee coding accuracy in Axys affects downstream conversion classification. | Axys | Medium |

Unknown: whether `epus` and `exus` are official Axys security types, APX security types, transaction codes, expense security types, or conversion-layer labels.

### 9.6 Reinvestment Representation

Evidence from Morningstar and ByAllAccounts suggests reinvestment activity may be represented by paired transactions.

| Evidence | System | Confidence |
|---|---|---:|
| Axys distribution reinvest transactions may appear as Buy plus Distribution transaction pairs in conversion data. | Axys | Medium |
| APX reinvestment may translate as `dv` and `by` pair in ByAllAccounts integration evidence. | APX | Medium |
| Reinvestment representation may affect realized and unrealized gain/loss reporting in downstream systems. | Axys/APX | Medium |
| Conversion tools may preserve the source representation rather than normalize it. | Axys | Medium |

Audit implication: paired reinvestment transactions should be checked for date, security, quantity/amount relationship, and income/cash linkage.

### 9.7 Deliver-In / Deliver-Out Interpretation

Morningstar Axys conversion evidence states that `li` deliver-in and `lo` deliver-out transactions may be interpreted differently depending on a transaction-setting code in the Advent client file.

| Fact | System | Confidence |
|---|---|---:|
| `li` and `lo` can be interpreted differently depending on a transaction-setting code in the Advent client file. | Axys | Medium |
| Setting code `Y` in the referenced 53rd character position maps `li`/`lo` to Deliver-In/Deliver-Out in Morningstar conversion. | Axys | Medium |
| Setting code `N` in the referenced 53rd character position maps `li`/`lo` to Credit/Debit of Security in Morningstar conversion. | Axys | Medium |
| Code-only transaction interpretation is unsafe for `li`/`lo`. | Axys | High as a design recommendation; Medium source evidence. |

### 9.8 Practical Classification Reference for Performance and Audit

The following section consolidates the standalone Axys Transaction Types Reference into this research chapter. It is a working reference for classifying Advent/SS&C Axys transaction types for performance, audit, reconciliation, and data-quality rules. It should not be treated as an official Axys manual or exhaustive proprietary code list. The most important practical point remains that transaction code alone is usually not sufficient. In practice, classification should use the transaction code together with security type, security symbol, source/destination type, source/destination symbol, quantity/sign, amount/sign, cash-security versus non-cash-security context, performance/cash-flow flag if available, and firm-specific trade-blotter mapping.

Public source notes for the standalone transaction-types reference:

- Custodial Integrator translates normalized WebPortfolio transaction types into Axys transaction codes and fields.
- The ByAllAccounts default translation table includes Axys codes such as `li`, `lo`, `by`, `sl`, `cs`, `ss`, `dv`, `in`, `ai`, `dp`, `wd`, `pa`, `sa`, `rc`, and `pd`.
- The same table warns that special cases and financial-institution-specific customizations can affect translation.
- Morningstar conversion notes discuss `epus`, `exus`, and the special treatment of `li`/`lo` deliver-in/deliver-out versus credit/debit behavior based on a setting in the Axys `.cli` client file.

#### 9.8.1 Executive Classification Table

| Axys code / marker | Common meaning | Group | External cash flow? | Position effect | Cash effect | Performance / audit notes |
|---|---|---|---|---|---|---|
| `li` | Deliver in / credit / deposit / transfer in | External cash flows / transfers | Often yes, but depends | Cash or security increase | Often increase | Core candidate for contribution/inflow. Must distinguish cash contribution from security transfer, internal journal, or correction. |
| `lo` | Deliver out / debit / withdrawal / transfer out | External cash flows / transfers | Often yes, but depends | Cash or security decrease | Often decrease | Core candidate for withdrawal/outflow. Must distinguish client withdrawal from fee, internal movement, or security transfer. |
| `by` | Buy | Trading activity | No | Increase long position | Decrease cash | Match execution price to close; validate commission, accrued interest, and settlement cash. |
| `sl` | Sell | Trading activity | No | Decrease long position | Increase cash | Match execution price to close; validate realized gain/loss, quantity, commission, settlement cash. |
| `cs` | Cover short | Trading activity | No | Decrease short position | Usually decrease cash | Public integration table maps cover-short to `cs`. Also appears in some closure scenarios. |
| `ss` | Short sale | Trading activity | No | Increase short position | Usually increase cash/proceeds | Validate borrow/short-sale treatment, proceeds, and exposure sign. |
| `dv` | Dividend | Income | No | Usually none unless reinvested | Increase cash or wash cash | Common equity dividend code; can be paired with `by` for reinvestment. |
| `in` | Interest / income | Income | No | Usually none | Increase cash | Common interest/bond income code. For cash-security dividend/income, public mappings may use `in`. |
| `ai` | Accrued interest / negative interest / margin interest | Income / trading adjunct | No | Usually none | Increase/decrease depending use | Public table maps negative interest and margin interest to `ai`; accrued-interest sell side maps to `sa`. |
| `pa` | Accrued interest on buy / purchase accrued interest | Trading adjunct / income | No | Usually none | Usually decrease cash | Used in public table for buy accrued interest; not an external cash flow. |
| `sa` | Accrued interest on sell / sale accrued interest | Trading adjunct / income | No | Usually none | Usually increase cash | Used in public table for sell accrued interest; not an external cash flow. |
| `dp` | Disbursement/payment/deposit-like cash transaction; fee/expense in integration mappings | Internal cash movements / expenses | Usually no | Usually none | Usually decrease/increase depending sign/context | Public mappings use `dp` for fees, service charges, investment expenses, cash-security buys, and some taxes. Needs firm-specific interpretation. |
| `wd` | Withdrawal-like cash entry for cash-security sell | Internal cash movements / cash security | Usually no | Usually none | Usually increase/decrease depending context | Public table maps cash-security sell to `wd`; do not assume it is a client withdrawal. |
| `rc` | Return of capital | Corporate actions / income | Usually no | Usually none or cost-basis impact | Increase cash | Validate against corporate action data; affects tax-lot/cost-basis and income classification. |
| `pd` | Principal paydown | Corporate actions / fixed income | Usually no | Decrease principal/quantity or cost basis | Increase cash | Important for MBS/ABS/bonds. Conversion notes warn Axys principal-paydown data may be incomplete for downstream processing. |
| `;` | Split / journal / other marker in public table | Corporate actions / non-performance | Usually no | Depends | Depends | Public table shows `;` for SPLIT, JOURNAL, and OTHER. Treat as placeholder/marker requiring manual interpretation. |
| `epus` | Expense/security type used for expenses/taxes/recordkeeping | Fee/expense security type | No | Usually none | Usually decrease cash | Morningstar notes convert Axys `epus` fee types as management fees; CI uses it as a configurable fee type. |
| `exus` | Expense/security type used for expenses/custody fees | Fee/expense security type | No | Usually none | Usually decrease cash | Morningstar notes convert Axys `exus` as expenses; CI examples use `exus custfee`. |
| `dvwash` | Dividend reinvestment wash security/symbol | Reinvestment support | No | Support marker | Wash cash | Used to pair dividend and buy for reinvested dividends. Not a true economic holding. |
| `caus margin` | Margin/cash symbol used in margin interest mappings | Financing support | No | Usually none | Cash/margin balance | Used in public mapping for negative interest/margin interest. |
| Uppercase transaction code, e.g. `BY` | Reversal/deletion of original transaction | Non-performance-affecting / correction | Depends | Reverses original | Reverses original | CI states reversal transactions convert the original type code to uppercase, Axys representation for transaction deletion. |

#### 9.8.2 Group 1 — External Cash Flows

External cash-flow classification is central for Modified Dietz and performance audit. In Axys, the most likely transaction codes are `li` and `lo`, but the economic meaning depends on the security/cash fields and firm setup.

##### `li` — Deliver in / inflow / credit / deposit / transfer in

| Attribute | Notes |
|---|---|
| Primary interpretation | Incoming cash or security transfer. |
| Typical use | Client contribution, direct deposit, incoming ACAT/security transfer, positive ATM/POS/transfer mapping, credit. |
| External flow? | Usually yes for true client contributions/transfers into the managed account. Not always. |
| Cash contribution pattern | `li` with source/destination type like `$pty` and source/destination symbol like `$cash`. |
| Security transfer pattern | `li` with non-cash security type/symbol and positive quantity. |
| Performance treatment | External flow for Dietz/TWR if it represents capital entering the portfolio from outside. |
| Audit tests | Compare to custodian cash-flow record; detect same-day offsetting `lo`; identify duplicate posting; verify performance-flow flag; verify quantity/price for security-in-kind transfers. |
| Common pitfall | Treating every `li` as a client contribution. Some are corrections, security credits, internal transfers, or data-interface artifacts. |

Practical classification rule:

```text
if code == 'li' and cash security/symbol == cash and source/destination indicates outside party:
    classify as external cash inflow candidate
elif code == 'li' and noncash security:
    classify as security transfer-in candidate
else:
    classify as deliver-in requiring firm mapping
```

##### `lo` — Deliver out / outflow / debit / withdrawal / transfer out

| Attribute | Notes |
|---|---|
| Primary interpretation | Outgoing cash or security transfer. |
| Typical use | Client withdrawal, direct debit, check/payment, outgoing transfer, negative ATM/POS/transfer mapping, debit. |
| External flow? | Usually yes for true client withdrawals/transfers out of the managed account. Not always. |
| Cash withdrawal pattern | `lo` with source/destination type like `$pty` and source/destination symbol like `$cash`. |
| Security transfer pattern | `lo` with non-cash security type/symbol and negative/outgoing quantity. |
| Performance treatment | External flow for Dietz/TWR if it represents capital leaving the portfolio to the client/outside party. |
| Audit tests | Compare to custodian withdrawal records; detect duplicate outflow; separate fees/taxes from true withdrawals; identify same-day transfer across household accounts. |
| Common pitfall | Treating all `lo` as external withdrawals. Some `lo` transactions may be debit/correction/security-out events. |

#### 9.8.3 Group 2 — Internal Cash Movements, Fees, Expenses, and Financing

These transactions affect cash but generally should not be treated as external client cash flows unless a firm's methodology explicitly says so.

##### `dp` — Payment/disbursement/deposit-like cash transaction, often used for fees/expenses in integrations

| Attribute | Notes |
|---|---|
| Primary interpretation | Context-dependent cash posting. Public mappings show `dp` for fee, service charge, investment expense, cash-security buy, and some tax mappings. |
| External flow? | Usually no. |
| Position effect | Usually none, unless paired with a security/fee symbol. |
| Cash effect | Usually decreases cash for fees/expenses, but sign/context matter. |
| Common security types/symbols | `exus custfee`, `epus expense`, `epus with`, and other firm-defined fee symbols. |
| Audit tests | Fee reasonableness; management-fee schedule; custody-fee schedule; withholding-tax classification; detect fees incorrectly marked as withdrawals. |
| Performance treatment | Usually an internal expense affecting return, not an external flow. But gross-of-fee versus net-of-fee reporting matters. |

##### `wd` — Cash-security sell withdrawal-like code

| Attribute | Notes |
|---|---|
| Primary interpretation | Public table maps cash-security SELL to `wd`. |
| External flow? | Usually no by itself. |
| Audit tests | Confirm whether this is cash-security liquidation versus client withdrawal. |
| Common pitfall | Name looks like withdrawal, but mapping may represent sale/redemption of a cash security. |

##### `ai` — Accrued interest / negative interest / margin interest

| Attribute | Notes |
|---|---|
| Primary interpretation | Interest-related adjustment, often negative interest or margin interest in public mappings. |
| External flow? | No. |
| Cash effect | Usually expense/debit for margin interest; sign matters. |
| Audit tests | Validate against margin debit balances and rate schedules; separate from normal bond interest. |

##### `epus` and `exus` — Expense-related Axys security types

These are not transaction codes in the same way as `by` or `sl`; they commonly appear as special security types or fee/expense classifications in Axys interface data.

| Security type | Common use | Audit notes |
|---|---|---|
| `epus` | Expense/payment/security type, management fee, tax/withholding in some mappings | Morningstar conversion notes say Axys `epus` fee codes are converted as management fees. Confirm firm use. |
| `exus` | Expense/security type, custody fee, service charge, generic expense | Morningstar conversion notes say Axys `exus` is converted as expenses. Confirm firm use. |

#### 9.8.4 Group 3 — Trading Activity

Trading transactions normally affect holdings and cash but should not be classified as external flows.

##### `by` — Buy

| Attribute | Notes |
|---|---|
| Primary interpretation | Purchase of a security. |
| External flow? | No. |
| Position effect | Increases long position. |
| Cash effect | Decreases cash. |
| Performance treatment | Internal transaction; affects holdings and future return but not a client flow. |
| Audit tests | Execution price vs market close; commission reasonableness; settlement amount; cash availability; trade-date versus settle-date; duplicate trade detection. |
| Special cases | Reinvested dividends may be represented as paired `dv` + `by`, often using `dvwash`. |

##### `sl` — Sell

| Attribute | Notes |
|---|---|
| Primary interpretation | Sale of a long security. |
| External flow? | No. |
| Position effect | Decreases long position. |
| Cash effect | Increases cash. |
| Audit tests | Execution price vs market close; realized gain/loss; lot relief; settlement amount; commission; wash-sale/tax-lot implications where applicable. |

##### `ss` — Short sale

| Attribute | Notes |
|---|---|
| Primary interpretation | Opening/increasing short position. |
| External flow? | No. |
| Position effect | Increases short exposure. |
| Cash effect | Usually increases cash/proceeds or short-credit balance. |
| Audit tests | Quantity sign; market value sign; borrow fees; margin treatment; performance exposure sign. |

##### `cs` — Cover short

| Attribute | Notes |
|---|---|
| Primary interpretation | Closing/decreasing short position. |
| External flow? | No. |
| Position effect | Decreases short exposure. |
| Cash effect | Usually decreases cash. |
| Audit tests | Quantity sign; gain/loss; short proceeds; closing price validation. |

##### `pa` — Purchase accrued interest / accrued interest on buy

| Attribute | Notes |
|---|---|
| Primary interpretation | Accrued interest paid when buying a bond. |
| External flow? | No. |
| Position effect | Usually none; attached to bond trade economics. |
| Cash effect | Decreases cash as part of settlement. |
| Audit tests | Validate accrued days, coupon, day-count convention, settlement date, ex-coupon rules. |

##### `sa` — Sale accrued interest / accrued interest on sell

| Attribute | Notes |
|---|---|
| Primary interpretation | Accrued interest received when selling a bond. |
| External flow? | No. |
| Position effect | Usually none; attached to bond trade economics. |
| Cash effect | Increases cash as part of settlement. |
| Audit tests | Validate accrued days, coupon, day-count convention, settlement date, ex-coupon rules. |

#### 9.8.5 Group 4 — Income

Income transactions usually affect cash and return, but are not external cash flows.

##### `dv` — Dividend

| Attribute | Notes |
|---|---|
| Primary interpretation | Dividend income from equity or dividend-paying security. |
| External flow? | No. |
| Cash effect | Increases cash unless reinvested or netted with withholding. |
| Position effect | None unless paired with reinvestment buy. |
| Audit tests | Expected dividend amount = shares × dividend rate; compare ex-date, record date, payable date; withholding-tax treatment; foreign tax reclaim; duplicate dividends. |
| Reinvestment pattern | Public mapping shows reinvestment as `dv` plus `by`, often using `dvwash`. |

##### `in` — Interest / income

| Attribute | Notes |
|---|---|
| Primary interpretation | Interest income, bond income, cash-security income, or income on certain security types. |
| External flow? | No. |
| Cash effect | Increases cash. |
| Position effect | Usually none. |
| Audit tests | Validate bond coupon schedule, coupon rate, day count, position quantity, pay date, accrued interest, and cash receipt. |

##### Negative dividend / withholding tax treatments

Some interfaces can represent withholding taxes or related tax/fee adjustments in different ways:

| Pattern | Meaning | Audit implication |
|---|---|---|
| Separate expense line, e.g. `dp` with `exus`/withholding symbol | Gross dividend plus separate tax/fee | Good for gross/net audit but requires linking lines. |
| Negative `dv` or negative `in` | Tax reduces income directly | Easier net cash but can obscure gross income. |
| Withholding tax field on income transaction | Single income transaction with withholding field | Best if downstream system preserves field. |

#### 9.8.6 Group 5 — Corporate Actions and Fixed-Income Principal Events

##### `rc` — Return of capital

| Attribute | Notes |
|---|---|
| Primary interpretation | Return of capital distribution. |
| External flow? | Usually no; it is issuer distribution, not client capital movement. |
| Cash effect | Increases cash. |
| Cost-basis effect | Often reduces cost basis. |
| Audit tests | Compare to corporate action source; verify tax classification; ensure not misclassified as dividend; verify cost-basis adjustment. |

##### `pd` — Principal paydown

| Attribute | Notes |
|---|---|
| Primary interpretation | Principal repayment/paydown, common for mortgage-backed and asset-backed securities. |
| External flow? | No. |
| Cash effect | Increases cash. |
| Position effect | Decreases principal balance or amortized quantity depending system method. |
| Audit tests | Validate principal factor; beginning factor vs ending factor; quantity/principal decrease; cash received; cost-basis and amortization. |
| Known issue | Conversion notes warn that Axys data may not contain complete downstream information for securities with principal paydown transaction types in some conversion contexts. |

##### `;` — Split / journal / other marker

| Attribute | Notes |
|---|---|
| Primary interpretation | Public translation table shows `;` for split, journal, and other. This appears to be a marker/placeholder rather than a conventional economic trade code. |
| External flow? | Usually no. |
| Position effect | For splits: quantity changes, market value should not materially change. For journals/other: depends. |
| Audit tests | For splits: verify split ratio, pre/post quantity, price adjustment, no artificial return. For journals/other: require firm mapping. |

#### 9.8.7 Group 6 — Non-Performance-Affecting Events, Reversals, Corrections, and Placeholders

##### Uppercase transaction codes — reversal / deletion

Public CI documentation states that reversal transactions are translated by converting the original transaction type code to uppercase. For example, if the original code was `by`, the reversal code is `BY`.

| Attribute | Notes |
|---|---|
| Meaning | Reversal/deletion of original transaction. |
| Performance treatment | Should reverse the original transaction's effect; do not count as a new independent economic event. |
| Audit tests | Match uppercase reversal to original lower-case transaction; verify same security, date, quantity, amount; identify orphan reversals. |

##### `;` / OTHER / JOURNAL

| Attribute | Notes |
|---|---|
| Meaning | Placeholder or special handling in public mappings. |
| Performance treatment | Cannot classify safely without firm-specific mapping. |
| Audit tests | Require manual review bucket; high-value journals should be reconciled to operational explanation. |

#### 9.8.8 Suggested Audit-Rule Matrix

| Audit rule | Relevant codes | Test |
|---|---|---|
| External-flow detection | `li`, `lo` | Identify cash/security flows that enter/leave portfolio from outside; exclude fees, trades, income, internal transfers. |
| Performance change explanation | `li`, `lo`, `dv`, `in`, `rc`, `pd`, `by`, `sl`, reversals | Compare prior-run vs current-run transaction sets for same historical period. |
| Trade price reasonableness | `by`, `sl`, `ss`, `cs` | Compare execution price to same-day close, high/low, or VWAP where available. |
| Dividend expected vs actual | `dv`, negative `dv`, withholding lines | Shares × declared dividend rate; match payable date and withholding. |
| Interest/coupon expected vs actual | `in`, `ai`, `pa`, `sa` | Coupon schedule, day count, accrual, settlement dates. |
| Split integrity | `;` split marker, firm split codes if any | Quantity changes by split ratio; price adjusts inversely; market value / MV approximately unchanged. |
| Return of capital classification | `rc` | Verify tax/corporate-action classification; cost basis decreases; not treated as ordinary dividend. |
| Principal paydown | `pd` | Validate principal factor, cash received, principal reduction, amortization. |
| Fee classification | `dp`, `epus`, `exus`, fee symbols | Separate custody fee, management fee, withholding tax, other expense; gross/net performance treatment. |
| Reversal/orphan detection | Uppercase codes | Match reversal to original; flag orphan reversals or mismatched amounts. |
| Reinvestment linkage | `dv` + `by`, `dvwash` | Confirm dividend cash equals reinvestment buy amount; avoid double-counting income/cash. |

#### 9.8.9 Recommended Normalized Transaction Fields for an Audit Product

To classify Axys/APX transactions robustly, normalize raw data into this schema:

| Field | Type | Notes |
|---|---:|---|
| `portfolio_id` | string | Axys portfolio/client code. |
| `transaction_id` | string | Stable ID if available; otherwise hash of raw fields. |
| `trade_date` | date | Economic/trade date. |
| `settle_date` | date nullable | Settlement date. |
| `posted_date` | date nullable | When posted/changed in system. Important for "why did history change?" |
| `transaction_code_raw` | string | Raw Axys code, preserving case. |
| `transaction_code_norm` | string | Lowercase normalized code. |
| `is_reversal` | boolean | True if uppercase reversal/deletion or explicit cancel flag. |
| `security_type` | string nullable | e.g. `csus`, `caus`, `fius`, `epus`, `exus`. |
| `security_id` | string nullable | Axys symbol/CUSIP/cash/fee symbol. |
| `src_dest_type` | string nullable | e.g. `$pty`, `$ity`, `$pth`. |
| `src_dest_symbol` | string nullable | e.g. `$cash`, `$income`. |
| `quantity` | decimal nullable | Signed if available. |
| `price` | decimal nullable | Transaction price. |
| `gross_amount` | decimal nullable | Before fees/taxes if available. |
| `net_amount` | decimal nullable | Net cash impact if available. |
| `commission` | decimal nullable | Trade commission. |
| `withholding_tax` | decimal nullable | Foreign/tax withholding. |
| `cash_impact` | decimal nullable | Signed cash impact in base currency. |
| `position_impact` | decimal nullable | Signed quantity/principal impact. |
| `currency` | string nullable | Transaction currency. |
| `performance_cash_flow_flag` | string/boolean nullable | Use if Axys/firm exports it. |
| `classification` | enum | External flow, trade, income, fee, corp action, correction, unknown. |
| `classification_confidence` | enum | Confirmed, probable, requires mapping, unknown. |
| `raw_line` | string | Preserve original source for audit traceability. |

#### 9.8.10 Practical Classification Hierarchy

Use a hierarchy like this instead of classifying by transaction code alone:

1. **Reversal / cancel detection** — Uppercase code or cancel flag means reversal/delete; link back to original transaction.
2. **Explicit firm mapping override** — Apply client-specific mapping table first. This is essential for `dp`, `wd`, `li`, `lo`, `epus`, `exus`, and `;` records.
3. **Security type and symbol** — Cash security, fee security, income cash, dividend wash, margin cash, real security.
4. **Code family** — `by/sl/ss/cs` = trade family; `dv/in/ai/pa/sa` = income/accrual family; `li/lo` = transfer/external-flow candidate; `rc/pd` = corporate-action/principal family; `dp/wd` = cash/fee/context-dependent family.
5. **Amount and quantity signs** — Positive/negative units and cash can determine in/out direction.
6. **Performance treatment** — Decide whether the event is an external flow, internal return event, non-managed transfer, correction, or non-performance event.

#### 9.8.11 Minimal Firm-Specific Mapping Table to Request from Each Axys Client

Ask each client to complete this table for their implementation:

| Raw code | Security type | Security symbol | Src/Dest type | Src/Dest symbol | Meaning at this firm | External flow? | Gross/net performance? | Notes |
|---|---|---|---|---|---|---|---|---|
| `li` | `caus`/cash | `$cash` | `$pty` | `$cash` | Client contribution | Yes | N/A | Confirm cash symbol. |
| `lo` | `caus`/cash | `$cash` | `$pty` | `$cash` | Client withdrawal | Yes | N/A | Confirm checks vs fees. |
| `dp` | `exus` | `custfee` | `$pty` | `$cash` | Custody fee | No | Net/gross depends | Confirm reporting policy. |
| `dp` | `epus` | `expense` | `$pty` | `$cash` | Management fee | No | Net/gross depends | Confirm if fee should be excluded for gross-of-fee. |
| `dv` | real security | real symbol | `$ity` | `$income` | Dividend | No | Return event | Link to expected dividend feed. |
| `in` | bond/security | real symbol | `$ity` | `$income` | Interest | No | Return event | Link to coupon schedule. |
| `rc` | real security | real symbol | `$pty` | `$cash` | Return of capital | No | Return/cost basis event | Confirm cost-basis handling. |
| `pd` | bond/MBS | real symbol | `$pty` | `$cash` | Principal paydown | No | Principal event | Confirm principal factor handling. |

#### 9.8.12 Publicly Observed WebPortfolio-to-Axys Mappings

The following rows are based on public ByAllAccounts Custodial Integrator default mappings. They are useful because they show how a real Axys integration maps normalized custodian activity into Axys codes, but they are not a universal Axys transaction-code manual.

| Normalized transaction type | Sign/context | Axys code(s) observed | Notes |
|---|---|---|---|
| ATM | `+` | `li` | Cash in. |
| ATM | `-` | `lo` | Cash out. |
| BUY | security | `by` | Standard buy. |
| BUY | cash security | `dp` | Context-dependent cash security handling. |
| COVER SHORT | security | `cs` | Cover short. |
| BUY ACCRUED INTEREST | bond/accrual | `pa` | Purchase accrued interest. |
| BUY Reinvested Div | reinvested dividend | `by` with `dvwash` | Reinvestment support. |
| CHECK | cash out | `lo` | Likely withdrawal/payment. |
| CLOSURE | `+` | `sl` | Public table maps positive closure to sell. |
| CLOSURE | `-` | `cs` | Public table maps negative closure to cover short. |
| CREDIT | cash/security | `li` | Inflow/credit candidate. |
| DEBIT | non-cash security | `lo` | Debit/outflow candidate. |
| DEBIT tax | tax/fee | `dp` with `epus with` | Withholding/tax style mapping. |
| DEPOSIT | cash | `li` | External inflow candidate. |
| DEPOSIT | non-cash security | `li` and `by` | Transfer/deposit plus buy in some contexts. |
| DIRECT DEBIT | cash out | `lo` | External outflow candidate. |
| DIRECT DEPOSIT | cash in | `li` | External inflow candidate. |
| DIVIDEND | dividend-paying security | `dv` | Dividend. |
| DIVIDEND | cash security | `in` | Cash-security income. |
| DIVIDEND | reinvested dividend | `dv` with `dvwash` | Paired with buy. |
| FEE | fee | `dp` with `exus custfee` | Fee/expense, not external flow. |
| RECORDKEEPING | fee | `dp` with `epus expense` | Fee/expense. |
| INCOME | bond security | `in` | Interest/income. |
| INCOME | cash security | `in` | Cash income. |
| INCOME | dividend-paying security | `dv` | Dividend-style income. |
| INTEREST | `+` | `in` | Interest income. |
| INTEREST | `-` | `ai` | Negative interest/accrual/margin-style mapping. |
| INVESTMENT EXPENSE | fee | `dp` with `exus custfee` | Expense. |
| JOURNAL | other | `;` | Requires firm mapping/manual review. |
| MARGIN INTEREST | margin | `ai` with `caus margin` | Financing expense. |
| OTHER | other | `;` | Requires firm mapping/manual review. |
| PAYMENT | cash out | `lo` | Payment/withdrawal candidate. |
| POINT OF SALE | `+` | `li` | Cash in candidate. |
| POINT OF SALE | `-` | `lo` | Cash out candidate. |
| REINVESTMENT | dividend + buy | `dv` and `by` with `dvwash` | Reinvested dividend pair. |
| REPEAT PAYMENT | cash out | `lo` | Payment/outflow candidate. |
| RETURN OF CAPITAL | normal | `rc` | Return of capital. |
| RETURN OF CAPITAL | bond security | `pd` | Principal paydown. |
| SELL | security | `sl` | Standard sell. |
| SELL | cash security | `wd` | Cash security handling. |
| SHORT | security | `ss` | Short sale. |
| SELL ACCRUED INTEREST | bond/accrual | `sa` | Sale accrued interest. |
| SERVICE CHARGE | fee | `dp` with `exus custfee` | Fee/expense. |
| SPLIT | split | `;` | Split marker/special handling. |
| TRANSFER | `+` | `li` | Transfer in candidate. |
| TRANSFER | `-` | `lo` | Transfer out candidate. |
| WITHDRAWAL | cash out | `lo` | External outflow candidate. |

#### 9.8.13 Bottom Line for an Audit Product

For an Axys/APX audit product, treat this reference as a seed mapping. The product should ship with:

1. A default mapping based on observed public Axys integration behavior.
2. A client-specific mapping override table.
3. A confidence level on every transaction classification.
4. A “requires review” bucket for ambiguous `li`, `lo`, `dp`, `wd`, `;`, `epus`, and `exus` cases.
5. A reconciliation report showing how raw codes were converted into audit classifications.

This design avoids the biggest risk: falsely treating every cash-like transaction as a performance external flow.

---

## 10. Transaction Field Dictionary

### 10.1 Important Warning

This field dictionary catalogs transaction fields commonly encountered in portfolio accounting systems and expected to appear in Axys/APX workflows. It is **not** a verified vendor field dictionary.

Fields are classified using public evidence and accounting practice.

### 10.2 Core Transaction Fields

| Field | Definition | Accounting Meaning | Typical Type | Required | Confidence |
|---|---|---|---|---|---:|
| Portfolio ID | Portfolio/account identifier. | Identifies owner of transaction. | String | Yes | High |
| Transaction Code | Type of accounting event. | Determines conceptual transaction. | String | Yes | High |
| Security Identifier | Security involved. | Links to Security Master. | String | Usually | High |
| Trade Date | Execution date. | Economic transaction date. | Date | Usually | High |
| Settlement Date | Settlement date. | Cash movement timing. | Date | Usually | High |
| Entry Date | Date entered into system. | Audit trail. | Date | Unknown | Medium |
| Posting Date | Date transaction posted. | Accounting update date. | Date | Unknown | Medium |
| Quantity | Units traded or affected. | Holding impact. | Decimal | Depends | High |
| Price | Unit price. | Valuation and cost basis. | Decimal | Depends | High |
| Gross Amount | Trade value before adjustments. | Cash calculation. | Decimal | Depends | High |
| Net Amount | Final cash amount. | Cash ledger update. | Decimal | Depends | High |
| Commission | Trading commission. | Cost basis/cash effect. | Decimal | Optional | High |
| Fees | Administrative/trading fees. | Expense or cost adjustment. | Decimal | Optional | High |
| FX Rate | Currency conversion rate. | Multi-currency accounting. | Decimal | Optional | Medium |
| Broker | Executing broker or representative. | Operational metadata. | String | Optional | Medium |
| Batch ID | Import batch identifier. | Import audit. | String | Unknown | Medium |
| Source ID | External transaction identifier. | Reconciliation and duplicate detection. | String | Unknown | Medium |
| Comment | Free-form note. | Operational metadata. | String | Optional | Medium |

### 10.3 Field Relationships

#### Portfolio ID

Purpose: associates every transaction with one portfolio or account.

Downstream dependencies:

- holdings,
- cash,
- performance,
- audit,
- reporting.

**Confidence:** High.

#### Security Identifier

Purpose: links transactions to the Security Master.

Common examples:

- internal security ID,
- ticker,
- CUSIP,
- ISIN.

Unknown: official Axys/APX preferred internal key hierarchy.

**Confidence:** Medium.

#### Trade Date

Purpose: represents when the investment decision or economic execution occurred.

Typical uses:

- performance,
- holdings,
- trade sequencing,
- audit.

Public evidence supports trade-date accounting capability in Axys.

**Confidence:** High.

#### Settlement Date

Purpose: represents when cash legally settles.

Typical uses:

- cash ledger,
- receivables,
- payables,
- reconciliation.

Audit candidate: settlement date should generally not precede trade date.

**Confidence:** High.

#### Quantity

Purpose: number of units affected.

Common cases:

- shares,
- bonds,
- fund units,
- lots.

Unknown: native Axys/APX storage precision.

**Confidence:** High.

#### Price

Purpose: execution price.

Dependencies:

- cost basis,
- realized gain/loss,
- transaction value.

Unknown: whether stored price always equals historical pricing database value.

**Confidence:** Medium.

#### Gross and Net Amount

Purpose: represent total trade value before and after commissions, fees, taxes, or adjustments.

Unknown: official Axys/APX naming conventions.

**Confidence:** Medium.

#### Broker

Observed in third-party integration workflows as a transaction attribute. May be populated through broker translation.

**Confidence:** Medium.

#### Source Identifier

External identifier used for reconciliation and duplicate detection.

Observed concept; official implementation unknown.

**Confidence:** Medium.

### 10.4 Observed APX / Blotter Fields

| Field | Observed Context | Confidence | Caveat |
|---|---|---:|---|
| APX Transaction Type | ByAllAccounts APX translation table. | Medium | Integration guide field. |
| APX Transaction Src/Dest Type | ByAllAccounts APX translation table. | Medium | Integration guide field. |
| APX Transaction Src/Dest Symbol | ByAllAccounts APX translation table. | Medium | Integration guide field. |
| APX Transaction Special Security Type | ByAllAccounts APX translation table. | Medium | Integration guide field. |
| APX Transaction Special Security Symbol | ByAllAccounts APX translation table. | Medium | Integration guide field. |
| Broker Representative Field | WealthTechs APX guide. | Medium | Integration workflow evidence. |
| Lot Location | WealthTechs APX guide. | Medium | Axys-era/APX workflow evidence. |
| Comment | WealthTechs APX guide. | Medium | Import file/comment logic evidence. |

### 10.5 Public Example Transaction Row

Public example rows from WealthTechs show a delimited layout:

```text
acct123,010101,010101,by,csus,appl,100,caus,cash,10000
```

Likely fields:

| Position | Observed Value | Tentative Meaning | Confidence |
|---:|---|---|---:|
| 1 | `acct123` | Account / portfolio code. | Medium |
| 2 | `010101` | Date field 1. | Unknown |
| 3 | `010101` | Date field 2. | Unknown |
| 4 | `by` | Transaction code. | Medium |
| 5 | `csus` | Security type. | Low to Medium |
| 6 | `appl` | Security symbol. | Low to Medium |
| 7 | `100` | Quantity. | Low to Medium |
| 8 | `caus` | Source/destination security type. | Low to Medium |
| 9 | `cash` | Source/destination symbol. | Low to Medium |
| 10 | `10000` | Cash amount / net amount / trade amount. | Unknown |

Important warning: this should not be treated as a complete Axys/APX import field dictionary. It is only a public example row from a third-party integration manual.

### 10.6 Candidate IMEX Fields

Expected but unverified transaction fields in transaction exports:

- Portfolio.
- Security.
- Trade Date.
- Settlement Date.
- Quantity.
- Price.
- Amount.
- Transaction Code.
- Broker.
- Currency.
- FX Rate.
- Comment.

**Confidence:** Unknown until official IMEX documentation or production exports are obtained.

### 10.7 Candidate REP Fields

REP fields remain mostly unknown.

Future research should identify:

- transaction summary report columns,
- posted vs pending transaction visibility,
- sort order,
- export columns,
- report parameters,
- underlying report source.

Observed Transaction Summary Report sample columns include:

- Ex-Date,
- Pay-Date,
- Trade Date,
- Settle Date,
- Quantity,
- Symbol,
- Security,
- Unit Price,
- Amount.

---

## 11. Axys Conversion Evidence from Morningstar

### 11.1 Distribution Reinvestment

The Morningstar Axys conversion guide states that Advent Axys distribution reinvest transaction types may translate into two transaction pairs, normally listed as Buy plus a corresponding Distribution transaction type. It says the data is converted into Morningstar Office as provided and may affect unrealized/realized gain/loss reporting.

Extracted facts:

| Fact | System | Confidence |
|---|---|---:|
| Axys distribution reinvest transactions may appear as Buy plus Distribution transaction pairs in conversion data. | Axys | Medium |
| Reinvestment representation may affect realized and unrealized gain/loss reporting in downstream systems. | Axys | Medium |
| Conversion tools may preserve the source representation rather than normalize it. | Axys | Medium |

### 11.2 Cost Basis in `.cli`

The Morningstar guide states that per-share cost-basis data is only converted if provided within the exported Advent `.cli` file.

Extracted facts:

| Fact | System | Confidence |
|---|---|---:|
| Advent `.cli` files may contain per-share cost-basis data. | Axys | Medium |
| Cost-basis conversion can depend on whether that data is present in the exported `.cli` file. | Axys | Medium |

### 11.3 Deliver-In / Deliver-Out Versus Credit / Debit

The Morningstar guide states that `li` deliver-in and `lo` deliver-out transactions with transaction-setting code `Y` in the 53rd character position inside the Advent client file are converted as Deliver-In or Deliver-Out. The same `li`/`lo` transactions with setting code `N` in that position are converted as Credit of Security or Debit of Security.

Extracted facts:

| Fact | System | Confidence |
|---|---|---:|
| `li` and `lo` can be interpreted differently depending on a transaction-setting code in the Advent client file. | Axys | Medium |
| Setting code `Y` in the referenced position maps `li`/`lo` to Deliver-In/Deliver-Out in Morningstar conversion. | Axys | Medium |
| Setting code `N` in the referenced position maps `li`/`lo` to Credit/Debit of Security in Morningstar conversion. | Axys | Medium |
| Code-only transaction interpretation is unsafe for `li`/`lo`. | Axys | High as recommendation; Medium source evidence. |

### 11.4 Transactions Linked to `none` or `client` Securities

The Morningstar guide states that if Axys database transactions are linked to securities labeled `none` or `client`, they are converted as out-of-pocket cash from the provided database files.

Extracted facts:

| Fact | System | Confidence |
|---|---|---:|
| Some Axys transactions may be linked to securities labeled `none` or `client` in conversion data. | Axys | Medium |
| Conversion tools may map such transactions to out-of-pocket cash. | Axys | Medium |
| Security linkage is important for correctly interpreting transaction accounting. | Axys | Medium |

### 11.5 Principal Paydown / Zero Quantity Caveat

The Morningstar guide discusses principal paydown transactions from Axys in the context of mortgage-backed securities and notes that some converted data may include zero-share quantity issues that cannot be processed by Morningstar Office, leading to reconciliation differences and holdings that do not match Axys performance reporting results.

Extracted facts:

| Fact | System | Confidence |
|---|---|---:|
| Principal paydowns from Axys may create conversion complications. | Axys | Medium |
| Zero-quantity security cases may create reconciliation problems in downstream systems. | Axys | Medium |
| Converted holdings may fail to match Axys performance reporting if source files contain incomplete data. | Axys | Medium |

### 11.6 Fee Codes and Conversion

The Morningstar guide states that multiple Axys fee types containing transaction code or label `epus` are converted as Management Fees, while transaction codes or labels listed as `exus` are converted as Expenses.

Extracted facts:

| Fact | System | Confidence |
|---|---|---:|
| `epus` appears in public Axys conversion documentation as associated with Management Fee conversion. | Axys | Medium |
| `exus` appears in public Axys conversion documentation as associated with Expense conversion. | Axys | Medium |
| Fee coding accuracy affects downstream conversion classification. | Axys | Medium |

### 11.7 Transaction Prices and Historical Prices

The Morningstar conversion guide states that transaction prices and historical security prices are converted if provided.

Extracted fact:

| Fact | System | Confidence |
|---|---|---:|
| Transaction prices and historical security prices may be converted if present in Axys conversion inputs. | Axys | Medium |

---

## 12. Audit Rules

### 12.1 Purpose

This section consolidates candidate audit rules for transaction processing in Axys/APX-style portfolio accounting systems.

These rules are research candidates. They are not yet verified product behavior. Future versions should promote rules to Verified only when supported by vendor documentation or reproducible production evidence.

### 12.2 Severity Levels

| Severity | Meaning |
|---|---|
| Critical | Likely accounting corruption or incorrect financial reporting. |
| High | Strong possibility of incorrect accounting. |
| Medium | Requires investigation. |
| Low | Informational or operational warning. |

### 12.3 Validation Rules

| Rule | Severity | Description | Required Fields / Inputs | Expected Result | Confidence |
|---|---|---|---|---|---:|
| TR-001 Missing Portfolio | Critical | Transaction references a portfolio that does not exist. | Portfolio ID. | Transaction should not post. | High |
| TR-002 Missing Security | Critical | Security transaction references an unknown security. | Security Identifier, Transaction Code. | Transaction should not post. | High |
| TR-003 Missing Trade Date | High | Trade-based transaction lacks trade date. | Trade Date. | Hold/reject depending on workflow. | High |
| TR-004 Settlement Before Trade | High | Settlement date precedes trade date. | Trade Date, Settlement Date. | Investigate or reject. | High |
| TR-005 Missing Quantity | High | Security transaction lacks required quantity. | Quantity, Transaction Code. | Hold/reject unless cash-only exception. | High |
| TR-006 Missing Price | Medium | Price-required transaction has no execution price. | Price, Transaction Code. | Investigate. | High |
| TR-007 Invalid FX Rate | Medium | Foreign-currency transaction has missing or invalid FX rate. | Currency, FX Rate. | Investigate. | Medium |

### 12.4 Translation Rules

| Rule | Severity | Description | Confidence |
|---|---|---|---:|
| TR-008 Portfolio Translation Failure | Critical | Portfolio cannot be translated from external source. | Medium |
| TR-009 Security Translation Failure | Critical | Security cannot be translated. | Medium |
| TR-010 Unsupported Transaction Type | High | External transaction type has no mapping. | Medium |
| TR-CODE-CONTEXT | High | Transaction effect should not be interpreted from transaction code alone. Use code, sign, quantity, amount, security type, source/destination type, source/destination symbol, and configuration. | Medium-to-High as design rule; Medium source evidence |

### 12.5 Blotter Rules

| Rule | Severity | Description | Confidence |
|---|---|---|---:|
| TR-011 Trade Blotter Exception | Medium | Transaction remains in exception state. | Medium |
| TR-012 Cancellation Mismatch | High | Cancellation transaction does not sufficiently match original transaction. | Medium |
| TR-013 Cancellation Requires Backup / Control | High | Cancellation blotters should be treated as high-risk operational workflows requiring backups and review. | Medium |

### 12.6 Accounting Rules

| Rule | Severity | Description | Confidence |
|---|---|---|---:|
| TR-014 Holdings Not Updated | Critical | Posted transaction not reflected in holdings. | High |
| TR-015 Cash Not Updated | Critical | Posted transaction not reflected in cash. | High |
| TR-016 Cost Basis Inconsistency | High | Cost basis inconsistent with transaction history. | Medium |
| TR-017 Tax Lot Inconsistency | High | Tax lots inconsistent with transaction history. | Medium |

### 12.7 Income Rules

| Rule | Severity | Description | Possible Explanations | Confidence |
|---|---|---|---|---:|
| TR-018 Dividend Without Position | Medium | Dividend received without supporting position. | timing, short position, stale holdings, incorrect security. | Medium |
| TR-019 Coupon Inconsistent With Bond | Medium | Coupon payment inconsistent with bond characteristics. | bad security terms, wrong rate, wrong pay date, missing amortization/paydown logic. | Medium |
| TR-020 Return of Capital Without Eligible Security | Medium | Return of capital appears for security not expected to support it. | bad code, wrong security type, missing corporate-action data. | Medium |

### 12.8 Corporate Action Rules

| Rule | Severity | Description | Confidence |
|---|---|---|---:|
| TR-021 Split Without Quantity Adjustment | High | Split detected without expected holding adjustment. | Medium |
| TR-022 Split Without Price Adjustment | Medium | Historical prices inconsistent with split. | Medium |
| TR-023 Principal Paydown Inconsistency | Medium | Observed principal paydown inconsistent with expected reduction. | Medium |

### 12.9 Reconciliation Rules

| Rule | Severity | Description | Confidence |
|---|---|---|---:|
| TR-024 Custodian Difference | High | Custodian transactions differ from accounting system. | High |
| TR-025 IMEX Difference | Medium | IMEX export differs from expected accounting records. | Medium |
| TR-026 REP Difference | Medium | REP report differs from accounting records. | Unknown |

### 12.10 Historical Change Rules

| Rule | Severity | Description | Downstream Impacts | Confidence |
|---|---|---|---|---:|
| TR-027 Historical Transaction Modified | High | Historical transaction edited. | holdings, cash, performance, reports. | High |
| TR-028 Historical Transaction Deleted | High | Deleted historical transaction. | realized gain, holdings, performance. | High |
| TR-029 Performance Restatement Candidate | High | Historical transaction change may require performance review. | performance, composites, reports, audit trail. | High |

### 12.11 Operational Rules

| Rule | Severity | Description | Detection Ideas | Confidence |
|---|---|---|---|---:|
| TR-030 Duplicate Transaction | High | Potential duplicate transaction. | portfolio, security, trade date, quantity, price, amount. | High |
| TR-031 Duplicate External Identifier | Medium | Duplicate external transaction identifier. | source ID, custodian ID, batch ID. | Medium |
| TR-032 Stale Pending Transaction | Medium | Pending transaction exceeds operational threshold. | pending status age, batch date. | Medium |
| TR-033 Batch Partially Processed | Medium | Import batch incomplete. | expected vs posted count, rejects, logs. | Medium |
| TR-034 Stale Account / Stale Price Detection | Medium | Import should identify stale accounts and stale prices before export/posting. | account age, price date. | Medium |

### 12.12 Audit Taxonomy

Public evidence suggests transaction auditing naturally falls into seven groups:

1. Validation.
2. Translation.
3. Blotter.
4. Accounting.
5. Income.
6. Corporate Actions.
7. Reconciliation.

This taxonomy should become the basis for the future Audit Rules chapter or audit section.

---

## 13. Contradictions, Tensions, and Interpretive Warnings

| Issue | Evidence | Interpretation | Confidence |
|---|---|---|---:|
| IMEX vs Trade Blotter | Some sources describe IMEX as import/export utility; transaction workflows also route transactions through Trade Blotter review/posting. | Likely complementary rather than contradictory: IMEX/import tooling may produce or load files, while Trade Blotter is part of transaction review/posting. Needs vendor confirmation. | Medium |
| Exact transaction codes | Third-party sources expose APX/Axys-like codes. | These may be integration-specific, version-specific, or configuration-dependent. Do not treat as complete official matrix. | High for caution; Medium for individual examples. |
| Are `by`, `sl`, `li`, etc. official APX codes? | Third-party sources use them as APX transaction codes. | Likely yes for many, but not verified from official vendor docs. | Medium |
| Uppercase cancellation universality | ByAllAccounts and WealthTechs both show lower-to-upper cancellation examples. | Strong integration evidence, but not confirmed across all versions/workflows. | Medium |
| Reinvestment representation | Public sources suggest reinvestments may be paired transactions. | Important for accounting and migration; exact native behavior requires vendor or production evidence. | Medium |
| Deliver-in/out interpretation | Morningstar notes `li`/`lo` can be interpreted differently depending on `.cli` transaction setting. | Meaning may depend on fields beyond visible code. Warn against code-only interpretation. | Medium |
| `epus` and `exus` terminology | Morningstar calls them transaction codes; ByAllAccounts describes special security types. | Terminology may vary by source or be imprecise. Needs vendor documentation. | Medium |
| Is IMEX the transaction source of truth? | IMEX imports files; Trade Blotter stages transactions; reports may show posted transactions; APX database can be queried. | IMEX is an interface, not necessarily the canonical native store. | Medium |
| Is `;` a transaction code or comment marker? | ByAllAccounts maps Journal, Other, and Split to `;`. | Treat as observed integration behavior only. | Medium |
| Are position-generated initial deliver-ins native APX behavior or AIA behavior? | WealthTechs documents this as an AIA setting. | Treat as AIA behavior unless vendor docs confirm native support. | Medium |
| REP/report source of truth | Report evidence suggests Transaction Summary Report; other sources focus on IMEX/import workflows. | REP/report output and IMEX exports should be documented separately. | Medium |
| Direct file access | AdventGuru warns direct file access is risky due to version changes. | Avoid basing tooling on proprietary files unless necessary and version-controlled. | Medium |

---

## 14. Known Unknowns and Research Backlog

### 14.1 Transaction Codes

| ID | Question | Why Important | Status | Potential Sources | Priority |
|---|---|---|---|---|---:|
| TU-001 | What is the complete official Axys transaction code matrix? | Required for definitive documentation. | Unknown | Vendor manuals, IMEX documentation, production systems. | High |
| TU-002 | What is the complete official APX transaction code matrix? | Required for definitive documentation. | Unknown | Vendor manuals, IMEX documentation, production systems. | High |
| TU-003 | Are Axys and APX transaction codes identical, overlapping, or divergent by version/configuration? | Needed for Axys/APX differences section. | Unknown | Official manuals, production exports. | High |
| TU-004 | Which observed codes are native versus integration-layer mappings? | Avoids false documentation. | Unknown | Vendor docs, production evidence. | High |
| TU-005 | Are deprecated or version-specific transaction codes relevant? | Version coverage. | Unknown | Release notes, manuals. | Medium |

### 14.2 IMEX

| ID | Question | Priority |
|---|---|---:|
| TU-006 | What are the official Axys IMEX transaction export object names? | High |
| TU-007 | What are the official Axys IMEX transaction import object names? | High |
| TU-008 | What are the official APX IMEX transaction export/import object names? | High |
| TU-009 | What is the complete IMEX transaction field list? | High |
| TU-010 | What is the official trade blotter import layout? | High |
| TU-011 | What logs and validation messages does IMEX produce? | Medium |
| TU-012 | What is the native IMEX object dependency sequence? | Medium |

### 14.3 REP and Reports

| ID | Question | Priority |
|---|---|---:|
| TU-013 | Which REP reports expose transaction information? | High |
| TU-014 | What are the official APX Transaction Summary Report parameters and fields? | High |
| TU-015 | Are REP values stored or recalculated? | Medium |
| TU-016 | What transaction reports exist in Axys? | High |
| TU-017 | What transaction reports exist in APX beyond Transaction Summary Report? | Medium |
| TU-018 | How do REP report values reconcile to IMEX exports and posted accounting records? | Medium |

### 14.4 Internal Data Model

| ID | Question | Priority |
|---|---|---:|
| TU-019 | How are transactions physically stored in Axys? | High |
| TU-020 | How are transactions stored in APX? | High |
| TU-021 | What internal identifiers uniquely identify transactions? | High |
| TU-022 | What posting status values exist? | Medium |
| TU-023 | What are the native Trade Blotter state transitions? | High |
| TU-024 | What native error states and rejection codes exist? | Medium |
| TU-025 | What native warning messages exist? | Medium |
| TU-026 | What batch rollback/restart/recovery logic exists? | Medium |
| TU-027 | What idempotency or duplicate-detection logic exists natively? | Medium |

### 14.5 Accounting and Historical Changes

| ID | Question | Priority |
|---|---|---:|
| TU-028 | How are reversals represented internally? | High |
| TU-029 | Whether uppercase transaction codes universally mean delete/reversal across all versions/configurations. | High |
| TU-030 | How are historical edits represented? | High |
| TU-031 | How are deleted transactions retained for audit? | High |
| TU-032 | How are corrections distinguished from reversals? | Medium |
| TU-033 | How do transaction edits propagate into holdings? | Medium |
| TU-034 | How do transaction edits propagate into cash? | Medium |
| TU-035 | How do transaction edits propagate into performance? | High |
| TU-036 | Can historical transactions be reconstructed completely? | High |

### 14.6 Lots, Cost Basis, and Tax

| ID | Question | Priority |
|---|---|---:|
| TU-037 | How are tax lots linked to transactions? | High |
| TU-038 | How are partial lot disposals represented? | Medium |
| TU-039 | How is per-share cost basis represented in `.cli` exports? | High |
| TU-040 | How do transfer lots preserve acquisition date and basis? | Medium |
| TU-041 | How are lot locations stored and used natively? | Medium |

### 14.7 Multi-Currency

| ID | Question | Priority |
|---|---|---:|
| TU-042 | How are FX rates stored? | Medium |
| TU-043 | How are cross-currency settlements represented? | Medium |
| TU-044 | How are FX transactions merged or paired in native workflows? | Medium |
| TU-045 | How are base-currency values stored versus calculated? | Medium |

### 14.8 Performance

| ID | Question | Priority |
|---|---|---:|
| TU-046 | Which transaction types affect stored performance? | High |
| TU-047 | Which transaction changes trigger performance restatement? | High |
| TU-048 | How are performance restatements detected or audited? | High |
| TU-049 | Are edited/deleted historical transactions visible to performance recalculation workflows? | High |

### 14.9 Audit

| ID | Question | Priority |
|---|---|---:|
| TU-050 | What audit trail is retained? | High |
| TU-051 | What user, timestamp, and batch metadata are retained? | High |
| TU-052 | Can deleted transactions be reconstructed completely? | High |
| TU-053 | Are original and corrected transaction versions both retained? | High |
| TU-054 | What evidence is available from logs, IMEX, REP, and database queries? | Medium |

### 14.10 Highest-Value Future Research Targets

The highest-value future research targets are:

1. Official Axys transaction documentation.
2. Official APX transaction documentation.
3. Official IMEX manuals.
4. Official REP manuals.
5. Official Trade Blotter documentation.
6. Sample IMEX transaction exports.
7. Sample REP transaction reports.
8. Production transaction files.
9. Vendor training manuals.
10. Release notes.
11. Public sample `.cli` files or sanitized exports.
12. Historical Advent manuals.

### 14.11 Suggested Search Targets

Future search queries should include:

1. `Advent Axys transaction codes by sl li lo dv`
2. `Advent APX transaction codes by sl li lo dv`
3. `Advent Axys trade blotter file format`
4. `Advent APX trade blotter file format`
5. `Axys IMEX transaction export fields`
6. `APX Import Export Utility transaction blotter fields`
7. `Advent APX Reports Guide Transaction Summary Report`
8. `Axys .cli transaction field positions`
9. `Advent client file transaction setting 53rd character`
10. `Advent Axys transaction code epus exus`
11. `APX transaction code uppercase delete`
12. `Advent Replang transaction report`

### 14.12 Exit Criteria for Research Maturity

This research package can be considered substantially mature when:

- All high-priority questions are answered or explicitly documented as unavailable.
- Medium-priority questions have documented evidence or rationale.
- Remaining low-priority questions are catalogued for future investigation.
- Official or production evidence has been obtained for transaction-code matrices and field layouts.
- Native Axys/APX distinctions are clearly separated from third-party integration behavior.

---

## 15. Research Conclusions

### 15.1 Transactions Are Central to Axys/APX Accounting Workflows

Public vendor and integration sources connect transactions to accounting, reconciliation, trade/settlement handling, positions, prices, performance, tax treatment, and reporting.

**Confidence:** High for general centrality; Low to Medium for detailed mechanics.

### 15.2 Trade Blotter Appears to Be a Key Transaction-Review/Posting Concept

Third-party Axys/APX integration guides repeatedly route transactions into Trade Blotters before posting. APX evidence also distinguishes Trade Blotters from Statement Blotters, Tax Lot Blotters, Position Blotters, Account Blotters, and Initial Transaction Blotters.

**Confidence:** Medium.

### 15.3 IMEX Should Be Treated as a First-Class Transaction Interface, Not the Only Interface

IMEX is documented as a practical import/export mechanism, but reports, Replang, Report Writer Pro, APX SQL/database access, and direct file access also appear in public evidence.

**Confidence:** Medium.

### 15.4 Exact Transaction Codes Should Remain Observed, Not Verified

Several transaction codes are visible in public third-party guides, but the complete official code matrix remains unverified.

**Confidence:** High for caution; Medium for individual examples.

### 15.5 Transaction Context Matters More Than Transaction Code Alone

Sign, security type, source/destination fields, special symbols, configuration, and custodian-specific translation rules can alter transaction interpretation.

**Confidence:** High as a design recommendation; Medium source evidence.

### 15.6 Reinvestment, Fees, Transfers, and Reversals Require Special Attention

The research highlights these as high-risk areas:

- Reinvestment may be paired transactions.
- Fees may depend on special security types/symbols and description translations.
- `li`/`lo` may depend on sign or `.cli` settings.
- Cancellations may require uppercase transaction codes and matching original fields.

**Confidence:** Medium.

### 15.7 Direct Proprietary File Access Should Be Treated Cautiously

Consultant evidence warns that direct Axys file formats may change between versions. Interfaces based on direct files need version awareness and careful validation.

**Confidence:** Medium.

### 15.8 Future Chapter 05 Should Separate Layers

Future Chapter 05 should distinguish these layers:

1. Conceptual accounting transactions.
2. Native Axys/APX behavior.
3. Trade Blotter staging.
4. IMEX import/export behavior.
5. REP/report output.
6. Third-party integration translation behavior.
7. Migration/conversion behavior.
8. Audit and reconciliation behavior.

This distinction is essential because public evidence often describes integration behavior rather than native system internals.

---

## Appendix A — Candidate Files and Repository Artifacts

Candidate files mentioned in the source research:

```text
research/2026-06-29-accounting-transactions-research.md
research/2026-06-29-transactions-deep-document-mining.md
research/2026-06-29-transaction-codes-and-field-layouts.md
research/transaction-source-index.csv
appendices/transaction-code-matrix.csv
appendices/transaction-code-matrix-draft.csv
appendices/transaction-field-dictionary-draft.csv
docs/05-Transactions.md
```

---

## Appendix B — Original Research Package Components Consolidated

This consolidated document integrates the following supplied materials:

| Original File | Integrated Into |
|---|---|
| `2026-06-29-accounting-transactions-research.md` | Executive summary, source catalog, extracted facts, conclusions, research backlog. |
| `2026-06-29-transactions-deep-document-mining.md` | Source evidence, Trade Blotter/IMEX/APX translation, cancellation, processing order, fee translation, conversion evidence, contradictions, unknowns. |
| `transaction-audit-rules.md` | Audit Rules section. |
| `transaction-code-matrix.md` | Transaction Code Matrix section. |
| `transaction-dependencies.md` | Transaction Dependencies section. |
| `transaction-field-dictionary.md` | Transaction Field Dictionary section. |
| `transaction-known-unknowns.md` | Known Unknowns and Research Backlog. |
| `transaction-lifecycle.md` | Transaction Lifecycle section. |
| `transaction-processing-pipeline.md` | Transaction Processing Pipeline section. |
| `transaction-sources.md` | Source Catalog and Evidence Quality. |
| `transactions_evidence_hunt_addendum.md` | Axys `topost.trn`, `imex32.exe`, folder labels, APX observed fields/codes, Transaction Summary Report, remaining unknowns. |

---

## Appendix C — Bottom Line for Future Chapter Drafting

The future Chapter 05 should be written from this research reference as a polished technical chapter. It should not reproduce every research caveat in-line, but it should retain the evidence discipline established here.

The most defensible framing is:

- Treat transactions as the central accounting event.
- Explain conceptual transaction categories first.
- Describe lifecycle and processing pipeline second.
- Separate observed Axys/APX integration evidence from official/native behavior.
- Present transaction code tables as observed/research unless verified.
- Emphasize that transaction interpretation requires context.
- Include audit rules tied to lifecycle stages.
- Preserve a known-unknowns backlog for unresolved vendor-specific mechanics.


---

# Appendix D — Independent Research Addendum

> **Date added:** 2026-06-29  
> **Purpose:** Independent web research pass requested by the user after the initial consolidated research package.  
> **Method:** Targeted searches for public Axys/APX transaction documentation, IMEX transaction details, Trade Blotter layouts, REP/report evidence, and version-specific interface behavior.  
> **Important limitation:** This addendum did not locate official complete Axys/APX transaction-code manuals, official complete IMEX transaction object definitions, or complete native Trade Blotter schemas. Newly discovered or re-confirmed information should therefore be treated as supplemental evidence, not replacement vendor documentation.

## D.1 Independent Research Summary

The independent research pass found additional publicly accessible evidence, but it did **not** resolve the most important native-documentation gaps. The strongest newly confirmed facts are:

1. SS&C's current Axys product page confirms Axys portfolio accounting/reporting capabilities, including automated reconciliation of trade information, settlement data, transactions, and positions; tax-lot or average-cost accounting; trade-date or settlement-date accounting; commission tracking by purpose or broker; integrated multi-currency capabilities; and corporate-actions processing.
2. SS&C's current APX product material confirms APX is positioned as a centralized book of record and integrated portfolio management, performance measurement, accounting, and reporting platform; it tracks holdings, transactions, and performance, and includes compliance/audit-trail language.
3. ByAllAccounts Custodial Integrator user guides for both Axys and APX provide directly named transaction translation fields: Transaction Type, Transaction Src/Dest Type, Transaction Src/Dest Symbol, and Transaction Special Security Type/Symbol.
4. ByAllAccounts explicitly documents that transfer interpretation can depend on sign: positive security units translate as `li`, while negative security units translate as `lo`, for both Axys and APX integration workflows.
5. ByAllAccounts documents special security type/symbol usage for fee transactions, including `exus` or `epus` with fee symbols such as `custfee` or `expense`, for both Axys and APX integration workflows.
6. WealthTechs AIA APX documentation confirms Trade Blotter organization settings: consolidate into one blotter, create one blotter per custodian, or create no trade blotter.
7. Advent/SSRS report evidence confirms that the Transaction Summary Report displays account transactions maintained by Advent, with observable sections/columns for sales, dividends, contributions, and withdrawals.
8. AdventGuru consultant evidence confirms that IMEX and Trade Blotter import are complementary, that direct Axys file access is risky across versions, that APX users may use SQL/reporting alternatives, and that APX v1.x through v4.x maintained IMEX while eliminating fixed-format file generation.
9. AdventGuru APX-to-Axys conversion evidence adds an important Axys audit-trail claim: transactions going into Axys portfolios need to go through the Trade Blotter, and records posted through the Trade Blotter are stored in the `didpost.aud` audit-trail file. This is consultant evidence and should remain **Medium Confidence** until vendor documentation or production observation confirms it.

## D.2 Search Results and Coverage

| Research Target | Result | Evidence Quality | Status |
|---|---|---:|---|
| Official Axys transaction-code manual | Not found in public search results. | Unknown | Still unresolved. |
| Official APX transaction-code manual | Not found in public search results. | Unknown | Still unresolved. |
| Official Axys IMEX transaction object names | Not found in public search results. | Unknown | Still unresolved. |
| Official APX IMEX transaction object names | Not found in public search results. | Unknown | Still unresolved. |
| Native Trade Blotter import layout | Public third-party examples found; official full schema not found. | Medium for examples; Unknown for full native schema | Partially unresolved. |
| REP / Transaction Summary Report evidence | Public Advent report guide/sample evidence found. | Medium | Partially resolved for report existence and visible sample fields only. |
| Axys/APX current product capability confirmation | Public SS&C pages found. | High for broad capabilities; Low for field mechanics | Updated capability evidence. |
| Version/interface notes | AdventGuru evidence found. | Medium | Updated interface/version evidence. |

## D.3 Newly Confirmed or Strengthened Sources

| ID | Source | Type | System | Newly Useful Evidence | Confidence |
|---:|---|---|---|---|---:|
| SRC-011 | SS&C Advent Axys product page | Vendor product page | Axys | Confirms automated reconciliation of trade information, settlement data, transactions, and positions; tax-lot/average-cost accounting; trade-date/settlement-date accounting; commission tracking; multi-currency capabilities; corporate-actions processing. | High for capabilities; Low for mechanics |
| SRC-012 | SS&C Advent Portfolio Exchange product brief/page | Vendor product material | APX | Confirms APX as centralized book of record and integrated portfolio management, performance measurement, accounting, and reporting platform; tracks holdings, transactions, and performance; includes audit-trail/compliance language. | High for capabilities; Low for mechanics |
| SRC-013 | ByAllAccounts Custodial Integrator Axys User Guide | Third-party integration manual | Axys | Confirms named Axys transaction translation fields, sign-dependent transfer translation, and fee special security type/symbol usage. | Medium |
| SRC-014 | ByAllAccounts Custodial Integrator APX User Guide | Third-party integration manual | APX | Confirms named APX transaction translation fields, sign-dependent transfer translation, and fee special security type/symbol usage. | Medium |
| SRC-015 | WealthTechs AIA User Manual for APX Users | Third-party integration manual | APX | Confirms Trade Blotter logic settings and distinction between Trade Blotter and Statement Transactions in APX-oriented workflow. | Medium |
| SRC-016 | WealthTechs AIA User Manual for Axys Users | Third-party integration manual | Axys | Confirms Transaction Translation logic, force-negative and force-date tools, and deletion of blotter transactions matching codes such as `BY`, `SL`, `SS`, `CS` in AIA workflow. | Medium |
| SRC-017 | Advent / SSRS Wealth Management Reports sample | Vendor/report sample | APX / Advent reports | Confirms Transaction Summary Report purpose and visible sample columns/sections. | Medium |
| SRC-018 | AdventGuru — Getting Data In and Out of Advent APX and Axys | Consultant article | Axys/APX | Confirms IMEX/interface/version points and warns against direct Axys file access due to version changes. | Medium |
| SRC-019 | AdventGuru — There and Back Again: APX to Axys Conversion | Consultant article | Axys/APX | Confirms APX-exported CLI conversion into Axys `topost.trn`; states Axys transaction imports must go through Trade Blotter and that Trade Blotter posting stores records in `didpost.aud`. | Medium |

## D.4 Axys Capability Evidence from Current SS&C Material

SS&C's current Axys page supports the following capability statements:

| Statement | Confidence | Notes |
|---|---:|---|
| Axys automates portfolio reporting and accounting. | High for capability | Vendor product material. |
| Axys includes automated reconciliation involving trade information, settlement data, transactions, and positions. | High for capability | Does not define transaction fields or reconciliation algorithms. |
| Axys supports tax-lot or average-cost accounting. | High for capability | Exact lot data model Unknown. |
| Axys supports trade-date or settlement-date accounting. | High for capability | Exact posting mechanics Unknown. |
| Axys can track and report commissions by purpose or broker. | High for capability | Exact transaction fields Unknown. |
| Axys has integrated multicurrency capabilities. | High for capability | FX transaction storage Unknown. |
| Axys has integrated corporate-actions processing. | High for capability | Corporate-action transaction generation Unknown. |

### D.4.1 Research Impact

These vendor capability statements strengthen the chapter's broad Axys coverage, but they do **not** provide:

- native transaction storage schema,
- transaction code matrix,
- IMEX transaction object names,
- Trade Blotter import schema,
- audit-trail schema,
- REP transaction field definitions.

## D.5 APX Capability Evidence from Current SS&C Material

SS&C's current APX product material supports the following capability statements:

| Statement | Confidence | Notes |
|---|---:|---|
| APX is positioned as an integrated portfolio management, accounting, performance measurement, and reporting platform. | High for capability | Vendor product material. |
| APX acts as a centralized book of record for portfolio-management business data. | High for capability | Vendor product material. |
| APX tracks holdings, transactions, and performance. | High for capability | Does not define transaction schema. |
| APX covers portfolios, positions, cash, and performance. | High for capability | Does not define internal data tables. |
| APX includes multiple security layers and a patented audit trail according to SS&C product brief language. | Medium to High for capability | Exact audit trail data model Unknown. |
| APX supports broad asset classes and settlement in any currency. | High for capability | Exact cross-currency transaction representation Unknown. |

### D.5.1 Research Impact

These statements strengthen broad APX coverage and justify treating APX transaction data as part of a broader book-of-record/accounting/reporting platform. They do **not** resolve native database schema or transaction-code unknowns.

## D.6 Axys and APX Transaction Translation Field Evidence

The ByAllAccounts Axys and APX guides provide unusually specific transaction translation field names. These are still integration-guide fields, but they are sufficiently specific to include in Chapter 05 as observed integration fields.

### D.6.1 Axys Observed Fields

| Field Name in Source | Meaning in Source | Confidence | Treatment |
|---|---|---:|---|
| Axys Transaction Type | Axys transaction code to which the WebPortfolio transaction type is translated. | Medium | Observed integration field. |
| Axys Transaction Src/Dest Type | Value of the Axys transaction Src/Dest Type field. | Medium | Observed integration field. |
| Axys Transaction Src/Dest Symbol | Source/destination symbol, including use of default cash/income cash symbols. | Medium | Observed integration field. |
| Axys Transaction Special Security Type / Symbol | Special security type/symbol used instead of a real security for some transactions, such as fees. | Medium | Observed integration field. |

### D.6.2 APX Observed Fields

| Field Name in Source | Meaning in Source | Confidence | Treatment |
|---|---|---:|---|
| APX Transaction Type | APX transaction code to which the WebPortfolio transaction type is translated. | Medium | Observed integration field. |
| APX Transaction Src/Dest Type | Value of the APX transaction Src/Dest Type field. | Medium | Observed integration field. |
| APX Transaction Src/Dest Symbol | Source/destination symbol, including use of default cash/income cash symbols. | Medium | Observed integration field. |
| APX Transaction Special Security Type / Symbol | Special security type/symbol used instead of a real security for some transactions, such as fees. | Medium | Observed integration field. |

### D.6.3 Research Impact

The field dictionary should be updated so these names are no longer listed merely as conceptual candidate fields. They are observed field labels in third-party Axys/APX integration documentation. However, whether they exactly match all native IMEX names, database column names, REP names, or internal storage names remains **Unknown**.

## D.7 Sign-Dependent Transfer Translation

ByAllAccounts documents the same sign-dependent transfer principle for Axys and APX integration workflows:

| System | Sign / Quantity Condition | Observed Translation | Confidence | Notes |
|---|---|---|---:|---|
| Axys | Positive security units | `li` | Medium | Transfer In in the ByAllAccounts default translation model. |
| Axys | Negative security units | `lo` | Medium | Transfer Out in the ByAllAccounts default translation model. |
| APX | Positive security units | `li` | Medium | Transfer In in the ByAllAccounts default translation model. |
| APX | Negative security units | `lo` | Medium | Transfer Out in the ByAllAccounts default translation model. |

### D.7.1 Research Impact

This strengthens the existing interpretation rule: transaction code alone is insufficient. Quantity sign and source/destination context must be retained in any parser or audit design.

## D.8 Special Security Type/Symbol for Fees

ByAllAccounts documents use of special security type/symbol fields for fees in both Axys and APX integration workflows.

| System | Observed Special Types | Observed Symbols | Confidence | Notes |
|---|---|---|---:|---|
| Axys | `exus`, `epus` | `custfee`, `expense` | Medium | Integration-guide terminology. |
| APX | `exus`, `epus` | `custfee`, `expense` | Medium | Integration-guide terminology. |

### D.8.1 Research Impact

This independently supports the earlier Morningstar evidence that `epus` and `exus` are important in fee/expense classification. However, the exact native classification of `epus` and `exus` remains unresolved because sources use different terminology: Morningstar refers to transaction codes or labels, while ByAllAccounts refers to special security types.

## D.9 WealthTechs AIA Trade Blotter Evidence

The WealthTechs APX manual provides explicit Trade Blotter configuration settings in the AIA workflow.

| APX AIA Setting | Meaning | Confidence | Caveat |
|---|---|---:|---|
| Consolidate Into One Blotter | Aggregates transactions from all custodians into one Trade Blotter. | Medium | AIA workflow, not necessarily native APX default. |
| Create One Blotter Per Custodian | Creates one Trade Blotter per custodian. | Medium | AIA workflow. |
| No Trade Blotter | AIA creates no Trade Blotter. | Medium | AIA workflow. |
| Trade Blotter Name | User enters the blotter where transactions should be imported. | Medium | AIA workflow. |

The same APX manual also distinguishes portfolio tabs or views for `Transactions` and `Statement Transactions` in its statement blotter discussion.

### D.9.1 Research Impact

The Trade Blotter section should distinguish:

1. Native APX transaction view/tab behavior — **partially observed** through AIA documentation.
2. AIA-generated blotter files/settings — **Medium Confidence**.
3. Native APX internal blotter state transitions — **Unknown**.

## D.10 WealthTechs AIA Transaction Translation Logic

The WealthTechs Axys manual documents Transaction Translation (`TT`) logic in the AIA workflow. It states that TT allows `IF, THEN` programming logic against source transaction data, is not case-sensitive, and is overridden by other translations such as Vehicle, Account, or Broker. It also documents special assignment behavior such as forcing a number field negative and forcing trade-date or settlement-date field values.

| Feature | System | Confidence | Caveat |
|---|---|---:|---|
| Transaction Translation allows IF/THEN logic. | Axys AIA workflow | Medium | Third-party integration behavior. |
| TT is not case-sensitive. | Axys AIA workflow | Medium | Third-party integration behavior. |
| TT can be overridden by Vehicle, Account, or Broker translations. | Axys AIA workflow | Medium | Third-party integration behavior. |
| `[TradeDate]` and `[SettleDate]` can be used as assignment values. | Axys AIA workflow | Medium | Third-party integration behavior. |
| AIA can delete Trade Blotter transactions that match transaction codes such as `BY`, `SL`, `SS`, `CS`. | Axys/APX AIA workflow | Medium | AIA behavior; native universality Unknown. |

### D.10.1 Research Impact

This adds useful detail to the processing-pipeline and audit sections. It also reinforces that transaction-processing order and transformations can be integration-layer behavior before native posting.

## D.11 REP / Transaction Summary Report Evidence

The Advent/SSRS Wealth Management Reports sample confirms the Transaction Summary Report purpose and visible sample columns.

| Report Section | Visible Columns / Fields | Confidence | Notes |
|---|---|---:|---|
| Sales | Trade Date, Settle Date, Quantity, Symbol, Security, Unit Cost, Total Cost, Unit Price, Proceeds, Gain/Loss | Medium | Visible in report sample. |
| Dividends | Ex-Date, Pay-Date, Symbol, Security, Amount | Medium | Visible in report sample. |
| Contributions | Trade Date, Settle Date, Quantity, Symbol, Security, Unit Price, Amount | Medium | Visible in report sample. |
| Withdrawals | Trade Date, Settle Date, Quantity, Symbol, Security, Unit Price, Amount | Medium | Visible in report sample. |

The sample describes the Transaction Summary Report as displaying all account transactions maintained by Advent, giving an independent record apart from the custodian.

### D.11.1 Research Impact

Chapter 05 can include these report sections and visible fields as observed report evidence. It should **not** infer that these are REP field names, database field names, IMEX field names, or all available report fields.

## D.12 IMEX, Report, and Direct Access Evidence from AdventGuru

AdventGuru provides useful historical/interface context:

| Statement | System | Confidence | Notes |
|---|---|---:|---|
| Axys v2.x introduced binary file formats. | Axys | Medium | Consultant evidence. |
| IMEX allowed import/export files in CSV, tab, and fixed formats. | Axys | Medium | Consultant evidence. |
| IMEX plus Trade Blotter import provides a practical path to move fundamental data in/out of Axys/APX. | Axys/APX | Medium | Consultant evidence. |
| Direct Axys file access is not best practice because file formats can change across versions. | Axys | Medium | Consultant evidence. |
| Upgrading from Axys v3.7 to v3.8 required file conversion and some resulting files had different formats. | Axys | Medium | Consultant evidence. |
| APX users may query the APX database via Excel and use SSRS, Crystal, and other SQL tools. | APX | Medium | Consultant evidence. |
| Axys/APX users can use Excel export, Report Writer Pro, Replang reports, and ETL tools. | Axys/APX | Medium | Consultant evidence. |
| APX v1.x through v4.x maintained IMEX functionality but eliminated fixed-format file generation. | APX | Medium | Consultant evidence. |

### D.12.1 Research Impact

This strengthens the chapter's interface-layer distinction:

1. IMEX is an import/export mechanism.
2. Trade Blotter is a transaction posting/review pathway.
3. REP/Replang/report output is a separate reporting/export pathway.
4. Direct file access is possible in Axys but risky.
5. APX has database/SQL/reporting alternatives that Axys does not have in the same way.

## D.13 APX-to-Axys Conversion Evidence: `topost.trn` and `didpost.aud`

AdventGuru's APX-to-Axys conversion article provides important transaction import and audit-trail evidence:

| Item | Statement | Confidence | Caveat |
|---|---|---:|---|
| APX exported portfolios may be converted into Axys Trade Blotter records. | Consultant evidence | Medium | Conversion-specific. |
| The conversion tool described takes APX-exported portfolio (`CLI`) files and builds an Axys Trade Blotter file named `topost.trn`. | Consultant evidence | Medium | Tool-specific but technically specific. |
| Axys does not let users import portfolio (`CLI`) files directly in the described conversion scenario. | Consultant evidence | Medium | Needs vendor confirmation. |
| Transactions going into an Axys portfolio need to go through the Trade Blotter in the described scenario. | Consultant evidence | Medium | Needs vendor confirmation for universality. |
| Records posted through the Trade Blotter are stored in the `didpost.aud` audit-trail file for recordkeeping. | Consultant evidence | Medium | High-value claim needing confirmation. |
| Reconstructing tax lots to match original APX files can be difficult. | Consultant evidence | Medium | Consistent with earlier known unknowns. |

### D.13.1 Research Impact

This is the most important independent addition. The existing research mentioned `topost.trn`, but this adds `didpost.aud` as an observed Axys audit-trail artifact associated with Trade Blotter posting.

Recommended treatment in Chapter 05:

| Claim | Recommended Status |
|---|---|
| `topost.trn` is an observed Axys Trade Blotter file name. | Medium Confidence. |
| `didpost.aud` is an observed Axys audit-trail file associated with Trade Blotter posting. | Medium Confidence. |
| All Axys transaction imports must always go through Trade Blotter. | Unknown / do not assert universally. |
| `didpost.aud` contains complete transaction audit trail with all fields. | Unknown. |

## D.14 Updated Transaction Field Dictionary Notes

The following field names can be promoted from merely conceptual to **Observed in integration documentation**:

| Field | Axys | APX | IMEX | REP | Confidence | Source Treatment |
|---|---|---|---|---|---:|---|
| Transaction Type | Observed as Axys Transaction Type | Observed as APX Transaction Type | Unknown | Unknown | Medium | ByAllAccounts integration fields. |
| Transaction Src/Dest Type | Observed | Observed | Unknown | Unknown | Medium | ByAllAccounts integration fields. |
| Transaction Src/Dest Symbol | Observed | Observed | Unknown | Unknown | Medium | ByAllAccounts integration fields. |
| Transaction Special Security Type / Symbol | Observed | Observed | Unknown | Unknown | Medium | ByAllAccounts integration fields. |
| Trade Date | Observed in AIA assignment logic and Transaction Summary Report | Observed in AIA/report evidence | Unknown official IMEX name | Observed report label | Medium | Multiple sources. |
| Settle Date / Settlement Date | Observed in AIA assignment logic and Transaction Summary Report | Observed in AIA/report evidence | Unknown official IMEX name | Observed report label | Medium | Multiple sources. |
| Unit Cost | Unknown | Observed in Transaction Summary Report Sales section | Unknown | Observed report label | Medium | Report sample only. |
| Total Cost | Unknown | Observed in Transaction Summary Report Sales section | Unknown | Observed report label | Medium | Report sample only. |
| Proceeds | Unknown | Observed in Transaction Summary Report Sales section | Unknown | Observed report label | Medium | Report sample only. |
| Gain/Loss | Unknown | Observed in Transaction Summary Report Sales section | Unknown | Observed report label | Medium | Report sample only. |

## D.15 Updated Observed Code Matrix Notes

The independent pass strengthened the evidence for the following observed codes:

| Code | Systems | Meaning / Role Observed | Confidence | Notes |
|---|---|---|---:|---|
| `by` | Axys/APX integration | Buy. | Medium | ByAllAccounts translation tables. |
| `li` | Axys/APX integration | Transfer/deposit/inflow-like direction; sign-dependent. | Medium | Positive units transfer in. |
| `lo` | Axys/APX integration | Transfer/withdrawal/outflow-like direction; sign-dependent. | Medium | Negative units transfer out. |
| `dp` | Axys/APX integration | Cash-security buy, fee, tax, service-charge, debit-like cases. | Medium | Context dependent. |
| `cs` | Axys/APX integration | Cover short. | Medium | Appears in ByAllAccounts and AIA examples. |
| `pa` | Axys/APX integration | Accrued-interest or reinvested-dividend-related buy-like case. | Low to Medium | Meaning still ambiguous. |
| `BY`, `SL`, `SS`, `CS` | Axys/APX AIA workflow | Uppercase codes used in deletion/cancellation or blotter-delete logic. | Medium | Do not assert universal native behavior. |

## D.16 Updated Unknowns After Independent Research

The following remain unresolved after independent research:

| ID | Unknown | Status After Independent Research | Priority |
|---|---|---|---:|
| TU-001 | Complete official Axys transaction-code matrix. | Not found publicly. | High |
| TU-002 | Complete official APX transaction-code matrix. | Not found publicly. | High |
| TU-006 | Official Axys IMEX transaction export object names. | Not found publicly. | High |
| TU-007 | Official Axys IMEX transaction import object names. | Not found publicly. | High |
| TU-008 | Official APX IMEX transaction export/import object names. | Not found publicly. | High |
| TU-009 | Complete IMEX transaction field list. | Not found publicly. | High |
| TU-010 | Official native Trade Blotter import layout. | Only public examples/third-party references found. | High |
| TU-014 | Official APX Transaction Summary Report parameters and full fields. | Partial public report sample found; full parameter specification not found. | High |
| TU-019 | Native Axys transaction storage model. | Not found publicly. | High |
| TU-020 | Native APX transaction storage model. | Not found publicly. | High |
| TU-021 | Native transaction identifiers. | Not found publicly. | High |
| TU-028 | Native reversal representation. | Only integration-level uppercase evidence found. | High |
| TU-031 | Native retention model for deleted transactions. | `didpost.aud` claim found for Axys Trade Blotter posting; detailed retention model still unknown. | High |

## D.17 Updated Research Conclusions

### D.17.1 Public Research Can Strengthen, But Not Fully Complete, Chapter 05

Independent research strengthened several areas, especially public product capability support, ByAllAccounts field labels, AIA Trade Blotter settings, Transaction Summary Report columns, and the `didpost.aud` audit-trail artifact. However, public research did not locate the missing official manuals needed to eliminate major unknowns.

### D.17.2 Integration Documentation Is the Best Public Transaction Evidence

The most useful public transaction details continue to come from third-party integration documentation, especially ByAllAccounts and WealthTechs. These sources are sufficiently detailed for implementation-aware documentation but should remain labeled **Medium Confidence**.

### D.17.3 Official Product Pages Are Useful for Capability Boundaries Only

Current SS&C product pages confirm that Axys/APX handle transactions, positions, reconciliation, performance, accounting, reporting, and audit/compliance workflows. They do not provide the implementation details required for field dictionaries, transaction code matrices, IMEX object definitions, or native storage schemas.

### D.17.4 `didpost.aud` Should Be Added to the Axys Known-Artifacts List

The `didpost.aud` file should be added as an observed Axys audit-trail artifact associated with Trade Blotter posting, but with **Medium Confidence** and a clear caveat that full content/layout/retention behavior remains **Unknown**.

## D.18 Source URLs for Independent Research Addendum

| Source ID | URL |
|---|---|
| SRC-011 | https://www.advent.com/solutions/axys/ |
| SRC-012 | https://www.advent.com/resources/all-resources/brief-advent-portfolio-exchange/ |
| SRC-012b | https://www.advent.com/solutions/advent-portfolio-exchange/ |
| SRC-013 | https://www.byallaccounts.net/Manuals/Custodial_Integrator/axys/CI_User_Guide.pdf |
| SRC-014 | https://www.byallaccounts.net/Manuals/Custodial_Integrator/apx/CI_User_Guide.pdf |
| SRC-015 | https://www.wealthtechs.com/files/AIADocumentation/APX%20Guide%20-%20AIA%20User%20Manual%20For%20APX%20Users.pdf |
| SRC-016 | https://www.wealthtechs.com/files/AIADocumentation/Axys%20Guide%20-%20AIA%20User%20Manual%20For%20Axys%20Users.pdf |
| SRC-017 | https://cdn.advent.com/cms/pdfs/reports/REP_SSRS.pdf |
| SRC-018 | https://adventguru.com/2013/04/25/getting-data-in-and-out-of-advent-apx-and-axys/ |
| SRC-019 | https://adventguru.com/2019/05/22/there-and-back-again/ |

## D.19 Recommended Changes to Chapter 05 Based on Addendum

| Chapter Area | Recommended Update |
|---|---|
| Axys Known Files / Artifacts | Add `didpost.aud` as observed audit-trail file associated with Trade Blotter posting; mark Medium Confidence. |
| Field Dictionary | Promote Src/Dest Type, Src/Dest Symbol, Special Security Type/Symbol from conceptual to observed integration fields for Axys and APX. |
| REP Section | Add Sales fields from Transaction Summary Report: Trade Date, Settle Date, Quantity, Symbol, Security, Unit Cost, Total Cost, Unit Price, Proceeds, Gain/Loss. |
| IMEX Section | Keep official object names Unknown; add stronger interface-layer caveat that public research did not locate official object definitions. |
| Transaction Codes | Keep observed matrix; do not promote to official. Strengthen sign-dependent `li`/`lo` treatment for both Axys and APX. |
| Known Issues / Quirks | Add direct-file-access version risk and APX fixed-format elimination as interface/version quirks. |
| Unknowns | Preserve native schemas, code matrices, and IMEX object definitions as high-priority Unknowns. |



---

# Appendix E — Second Independent Research Pass

> **Date added:** 2026-06-29  
> **Purpose:** Targeted public-source research pass to reduce the remaining Unknowns before drafting `Chapter_05_Transactions.md`.  
> **Scope:** Axys/APX transaction codes, Trade Blotter artifacts, IMEX behavior, REP/report evidence, audit trail evidence, transaction-file fields, and integration-layer quirks.  
> **Method:** Targeted searches and review of public SS&C Advent pages, Advent report PDFs, ByAllAccounts Custodial Integrator guides, WealthTechs AIA guides, and AdventGuru consultant articles.  
> **Important limitation:** This pass still did **not** locate official complete Axys/APX transaction-code manuals, official IMEX transaction object definitions, complete native Trade Blotter layouts, native Axys transaction storage schemas, or native APX database transaction schemas. Those items remain **Unknown**.

## E.1 Bottom Line

The second independent research pass found useful additional evidence, but it did **not** eliminate the largest documentation gaps.

The strongest incremental findings are:

1. The ByAllAccounts Axys guide gives more concrete Axys transaction-import details than previously captured, including explicit references to `topost.trn`, append behavior, `imex32.exe`, `pospos32.exe`, IMEX logs, folder labels, and several observed transaction-file columns or settings.
2. The ByAllAccounts Axys guide explicitly states that transactions are delivered into the designated Trade Blotter for review and posting, and that generated transactions are appended to `topost.trn` rather than replacing existing blotter contents.
3. The ByAllAccounts Axys guide gives observed transaction translation fields: `Axys Transaction Type`, `Axys Transaction Src/Dest Type`, and `Axys Transaction Special Security Type / Symbol`.
4. The ByAllAccounts Axys and APX guides both document sign-dependent transfer interpretation: negative units translate as `lo`; positive units translate as `li`.
5. The ByAllAccounts APX guide gives a fuller observed APX translation matrix for `by`, `dp`, `cs`, `pa`, `sl`, `li`, `lo`, `dv`, `in`, `ai`, `rc`, `pd`, `wd`, `ss`, `sa`, and `;`.
6. AdventGuru provides additional evidence that `didpost.aud` is an Audit Trail file used by both Axys and APX to review posted transactions, and that large Axys 3.x Audit Trail exports through IMEX may fail in some environments.
7. The Advent/SSRS Wealth Management Reports PDF gives stronger visible evidence for the Transaction Summary Report sections and fields: Purchases, Sales, Dividends, Contributions, and Withdrawals.
8. The WealthTechs Axys AIA guide adds specific observed handling for withholding tax and original-face quantity options in an Axys integration workflow.

## E.2 Research Targets and Outcomes

| Target | Outcome | Confidence | Status |
|---|---|---:|---|
| Complete official Axys transaction-code matrix | Not found publicly. | Unknown | Still unresolved. |
| Complete official APX transaction-code matrix | Not found publicly. | Unknown | Still unresolved. |
| Official Axys IMEX transaction object names | Not found publicly. | Unknown | Still unresolved. |
| Official APX IMEX transaction object names | Not found publicly. | Unknown | Still unresolved. |
| Native Trade Blotter full field layout | Partial third-party integration evidence found; official full layout not found. | Medium for observed fields; Unknown for native schema | Partially unresolved. |
| Axys `topost.trn` behavior | Stronger third-party evidence found. | Medium | Improved. |
| Axys/APX `didpost.aud` audit trail | Stronger consultant evidence found. | Medium | Improved. |
| IMEX logs | Stronger third-party Axys evidence found. | Medium | Improved. |
| REP Transaction Summary Report fields | Stronger visible report evidence found. | Medium | Improved. |
| APX SQL/native transaction schema | Not found publicly. | Unknown | Still unresolved. |
| Native deleted-transaction retention model | Not found publicly. | Unknown | Still unresolved. |

## E.3 Sources Reviewed in Second Pass

| ID | Source | Type | System | Useful Evidence | Confidence |
|---:|---|---|---|---|---:|
| SRC-020 | SS&C Advent Axys product page | Vendor product page | Axys | Confirms current Axys positioning around portfolio accounting/reporting, automated reconciliation of trades/settlement/transactions/positions, tax-lot or average-cost accounting, trade-date or settlement-date accounting, commissions by purpose/broker, multicurrency, and corporate-actions processing. | High for capabilities; Low for implementation mechanics |
| SRC-021 | SS&C Advent Portfolio Exchange product page | Vendor product page | APX | Confirms APX as integrated portfolio/client management with accounting/reporting, performance analytics, standard reports, custom reporting, multi-currency/multi-asset coverage, and front-to-back integration. | High for capabilities; Low for implementation mechanics |
| SRC-022 | ByAllAccounts Custodial Integrator Axys User Guide | Third-party integration manual | Axys | IMEX definition, Trade Blotter workflow, `topost.trn`, append behavior, folder labels, transaction translation fields, sign-dependent transfers, reversal by uppercase code, IMEX logs, configuration parameters and observed transaction-file fields. | Medium |
| SRC-023 | ByAllAccounts Custodial Integrator APX User Guide | Third-party integration manual | APX | Transaction translation fields, sign-dependent transfers, default transaction code mappings, fee special security type/symbols, reversal by uppercase code, commission/broker/lot-location parameter evidence. | Medium |
| SRC-024 | WealthTechs AIA User Manual for Axys Users | Third-party integration manual | Axys | Withholding-tax handling options, examples of `dv` lines, original-face quantity settings, transaction quantity rounding behavior. | Medium |
| SRC-025 | AdventGuru — Fixing the Audit Trail Export | Consultant article | Axys/APX | `didpost.aud` as Audit Trail file for posted transactions; Axys 3.x large Audit Trail IMEX export reliability issue; direct AUD file handling caveat. | Medium |
| SRC-026 | AdventGuru — There and Back Again: APX to Axys Conversion | Consultant article | Axys/APX | APX-exported CLI mapped to Axys `topost.trn`; transactions going into Axys portfolios through Trade Blotter in that scenario; `didpost.aud` recordkeeping claim; tax-lot reconstruction caution. | Medium |
| SRC-027 | AdventGuru — Getting Data In and Out of Advent APX and Axys | Consultant article | Axys/APX | IMEX history, Axys v2.x binary file context, direct-file-access risk, APX SQL/export alternatives, APX fixed-format generation eliminated while IMEX remained. | Medium |
| SRC-028 | Advent / SSRS Wealth Management Reports PDF | Vendor/report sample | APX / Advent reports | Transaction Summary Report visible sections and fields: Purchases, Sales, Dividends, Contributions, Withdrawals; report purpose statement. | Medium |

## E.4 Source URLs

| ID | URL |
|---:|---|
| SRC-020 | https://www.advent.com/solutions/axys/ |
| SRC-021 | https://www.advent.com/solutions/advent-portfolio-exchange/ |
| SRC-022 | https://www.byallaccounts.net/Manuals/Custodial_Integrator/axys/CI_User_Guide.pdf |
| SRC-023 | https://www.byallaccounts.net/Manuals/Custodial_Integrator/apx/CI_User_Guide.pdf |
| SRC-024 | https://www.wealthtechs.com/files/AIADocumentation/Axys%20Guide%20-%20AIA%20User%20Manual%20For%20Axys%20Users.pdf |
| SRC-025 | https://adventguru.com/2019/09/30/fixing-the-audit-trail-export/ |
| SRC-026 | https://adventguru.com/2019/05/22/there-and-back-again/ |
| SRC-027 | https://adventguru.com/2013/04/25/getting-data-in-and-out-of-advent-apx-and-axys/ |
| SRC-028 | https://cdn.advent.com/cms/pdfs/reports/REP_SSRS.pdf |

## E.5 Axys — Updated Evidence

### E.5.1 Axys Capability Boundary

SS&C's current Axys page supports the following capability statements. These are product capability statements, not field-level documentation.

| Statement | Confidence | Treatment in Chapter 05 |
|---|---:|---|
| Axys automates portfolio reporting and accounting. | High for capability | Can be stated as product capability. |
| Axys supports automated reconciliation of trade information, settlement data, transactions, and positions. | High for capability | Can be stated as product capability; algorithm and fields Unknown. |
| Axys supports tax-lot or average-cost accounting. | High for capability | Can be stated as product capability; lot data model Unknown. |
| Axys supports trade-date or settlement-date accounting. | High for capability | Can be stated as product capability; posting mechanics Unknown. |
| Axys can track/report commissions by purpose or broker. | High for capability | Can be stated as product capability; native transaction fields Unknown. |
| Axys supports integrated multicurrency and corporate-actions processing. | High for capability | Can be stated as product capability; FX/corporate-action transaction generation Unknown. |

### E.5.2 Axys Trade Blotter and `topost.trn`

The ByAllAccounts Axys guide strengthens the evidence for `topost.trn`.

| Item | Observed Evidence | Confidence | Caveat |
|---|---|---:|---|
| `topost.trn` | Transactions are delivered to the Axys Trade Blotter file named `topost.trn` in the Axys user folder. | Medium | Third-party integration workflow. |
| Append behavior | Custodial Integrator appends transactions to the end of `topost.trn` and leaves existing transactions unchanged. | Medium | Workflow-specific. |
| Creation behavior | If no Trade Blotter file exists, the Axys Import/Export utility creates one. | Medium | Third-party integration evidence. |
| Comment boundaries | Generated transaction blocks can be bounded by beginning and ending comment transactions containing the generation date. | Medium | Third-party integration evidence. |
| Review/posting | Transactions are delivered to the designated Trade Blotter for review and posting to Axys. | Medium | Does not prove all native imports must use this path. |

### E.5.3 Axys Executables, Folders, and Files

| Artifact | Observed Role | Confidence | Caveat |
|---|---|---:|---|
| `imex32.exe` | Axys Import/Export utility executable. | Medium | Third-party guide terminology. |
| `pospos32.exe` | Axys Post Positions utility executable. | Medium | Relevant to positions, not transaction chapter except as integration context. |
| `$pathexe` | Root folder where Axys executables are located. | Medium | Integration configuration. |
| `$pathtrn` | Axys user folder where `topost.trn` is found or created. | Medium | Integration configuration. |
| `$pathcli` | Folder where Axys portfolio/client files `*.cli` are stored. | Medium | Used by CI to build portfolio-code list. |
| `$pathinf` | Folder where `sec.inf` and `type.inf` are stored. | Medium | Used by CI to support generation of transaction and position files. |
| `$pathpri` | Axys price-file folder containing `*.pri`. | Medium | Pricing context. |
| `$pathlog` | Folder where Axys Import/Export logs are written. | Medium | Log format Unknown. |

### E.5.4 Axys Observed Transaction Translation Fields

The ByAllAccounts Axys guide explicitly names these fields in its translation-table discussion.

| Field Name in Source | Description in Source | Confidence | Repository Treatment |
|---|---|---:|---|
| Axys Transaction Type | Axys transaction code to which the WebPortfolio transaction type is translated. | Medium | Observed integration field; native IMEX/database/REP name Unknown. |
| Axys Transaction Src/Dest Type | Value of the Axys transaction Src/Dest Type field. | Medium | Observed integration field; native IMEX/database/REP name Unknown. |
| Axys Transaction Src/Dest Symbol | Source/destination symbol; guide references default cash and income cash account fields. | Medium | Observed integration field; native IMEX/database/REP name Unknown. |
| Axys Transaction Special Security Type / Symbol | Special security type/symbol used instead of the real security for some cases, such as fees. | Medium | Observed integration field; native IMEX/database/REP name Unknown. |

### E.5.5 Axys Additional Observed Transaction-File Columns / Parameters

The ByAllAccounts Axys guide describes parameters that populate or affect columns in the Axys `topost.trn` transaction file. These are not complete layout documentation.

| Observed Field / Parameter | Meaning in Source | Confidence | Caveat |
|---|---|---:|---|
| Broker column | A value such as `$brok` can be used when Broker must be populated, including when Commission has a value. | Medium | Column observed via integration parameter; full layout Unknown. |
| Commission column | Can be filled when commission value is available if `axyscommission=y`; can be suppressed with `axyscommission=n`. | Medium | Column observed via integration parameter; full layout Unknown. |
| Lot location column | `axyslotlocation` defines value used for the lot location column; example default `253`. | Medium | Native lot-location model Unknown. |
| Quantity column | `defdivquan` can supply a Quantity value for Dividend transactions with no reported Quantity. | Medium | Integration behavior. |
| Mark to Market field | Non-system-currency transactions may require Mark to Market field; parameter defines value to use. | Medium | Native multicurrency mechanics Unknown. |
| Perf/CW column | `defperfcw` defines value for Perf/CW column of Axys `topost.trn`. | Medium | Exact meaning and native behavior Unknown. |
| Currency code | `axyscur` defines Axys system currency for CI translation. | Medium | Configuration parameter, not necessarily transaction-file column. |

### E.5.6 Axys Sign-Dependent Transfer Translation

| Condition | Observed Translation | Confidence | Caveat |
|---|---|---:|---|
| Transfer of securities with negative units | `lo` | Medium | ByAllAccounts Axys integration evidence. |
| Transfer of securities with positive units | `li` | Medium | ByAllAccounts Axys integration evidence. |

### E.5.7 Axys Reversal / Cancellation

| Statement | Confidence | Caveat |
|---|---:|---|
| Custodial Integrator translates Axys reversal transactions by converting the original transaction type code to uppercase. | Medium | Integration evidence only. |
| Example: original `by` becomes reversal `BY`. | Medium | Integration evidence only. |
| If reversal fields do not match original transaction fields, Axys may generate a Trade Blotter error requiring resolution. | Medium | Integration evidence only. |
| Universal native uppercase-cancellation behavior across all Axys versions/import paths remains Unknown. | Unknown | Requires vendor or production evidence. |

### E.5.8 Axys Withholding Tax and Original Face Evidence

The WealthTechs Axys AIA guide adds integration-specific evidence for fixed-income and income-related transaction handling.

| Feature | Observed Behavior | Confidence | Caveat |
|---|---|---:|---|
| Original face as quantity on TRN | AIA can write original face value to the quantity field on `BY` transactions. | Medium | AIA behavior; native Axys behavior Unknown. |
| Original face as quantity on both files | AIA can write original face to quantity on `BY` transaction and position file. | Medium | AIA behavior. |
| Original face as quantity on POS | AIA can write original face to quantity on positions. | Medium | Position context. |
| Withholding tax: Treat as Expense | AIA option for withholding taxes on income transactions. | Medium | Integration-specific. |
| Withholding tax: Treat as Negative Dividend | AIA option; example uses a `dv` line with negative amount. | Medium | Integration-specific. |
| Withholding tax field in transaction | AIA option adds a withholding tax field and deducts withholding from trade amount. | Medium | Integration-specific. |
| Quantity decimal rounding | AIA default described as six decimals; option can force two decimal-place quantities. | Medium | Integration-specific. |

## E.6 APX — Updated Evidence

### E.6.1 APX Capability Boundary

SS&C's current APX page supports these broad capability statements. These are not field-level specifications.

| Statement | Confidence | Treatment in Chapter 05 |
|---|---:|---|
| APX is an integrated portfolio and client management solution. | High for capability | Can be stated as product positioning. |
| APX supports portfolio accounting/reporting and performance analytics. | High for capability | Can be stated as product capability. |
| APX has a vast standard report library and flexible custom reporting. | High for capability | Can be stated generally; report specs Unknown. |
| APX provides multi-currency and multi-asset class coverage. | High for capability | Can be stated generally; transaction schema Unknown. |
| APX can be deployed locally or cloud-delivered with or without outsourcing services. | High for deployment option | Not directly transaction-specific. |

### E.6.2 APX Observed Transaction Translation Fields

The ByAllAccounts APX guide explicitly names these fields.

| Field Name in Source | Description in Source | Confidence | Repository Treatment |
|---|---|---:|---|
| APX Transaction Type | APX transaction code to which the WebPortfolio transaction type is translated. | Medium | Observed integration field; native IMEX/database/REP name Unknown. |
| APX Transaction Src/Dest Type | Value of the APX transaction's Src/Dest Type field. | Medium | Observed integration field; native IMEX/database/REP name Unknown. |
| APX Transaction Src/Dest Symbol | Source/destination symbol; guide references default cash and income cash account fields. | Medium | Observed integration field; native IMEX/database/REP name Unknown. |
| APX Transaction Special Security Type / Symbol | Special security type/symbol used instead of a real security, for example fee handling. | Medium | Observed integration field; native IMEX/database/REP name Unknown. |

### E.6.3 APX Observed Code Matrix — Stronger Public Evidence

The ByAllAccounts APX default translation table provides the strongest public code evidence found in this pass. The following remains an observed integration translation matrix, not an official native APX code manual.

| Observed Code | Observed Source Concept(s) | Confidence | Caveat |
|---|---|---:|---|
| `by` | Buy; reinvestment paired leg; non-cash deposit paired leg. | Medium | Integration translation table. |
| `dp` | Cash security buy; tax; fee; recordkeeping; investment expense; service charge. | Medium | Context-dependent. |
| `cs` | Cover short; negative closure. | Medium | Integration translation table. |
| `pa` | Accrued interest / reinvested-dividend-related case in Buy row. | Low to Medium | Meaning remains ambiguous. |
| `li` | ATM positive; credit; deposit; direct deposit; positive income bond security; point-of-sale positive; transfer positive. | Medium | Direction/sign dependent. |
| `lo` | ATM negative; check; non-cash debit; direct debit; negative income bond security; payment; point-of-sale negative; transfer negative; withdrawal. | Medium | Direction/sign dependent. |
| `dv` | Dividend; dividend-paying security income; reinvested dividend; reinvestment paired leg. | Medium | Income/reinvestment context. |
| `in` | Cash-security dividend/income; interest positive. | Medium | Integration translation table. |
| `ai` | Interest negative; margin interest. | Medium | Context-dependent. |
| `rc` | Return of capital. | Medium | Integration translation table. |
| `pd` | Bond-security return-of-capital/principal paydown. | Medium | Bond-related special case. |
| `sl` | Sell; positive closure. | Medium | Integration translation table. |
| `wd` | Sell cash security. | Medium | Cash-security special case. |
| `ss` | Short. | Medium | Integration translation table. |
| `sa` | Accrued interest on sell. | Medium | Integration translation table. |
| `;` | Journal, Other, Split. | Medium | Treat as integration behavior; official native meaning Unknown. |

### E.6.4 APX Sign-Dependent Transfer Translation

| Condition | Observed Translation | Confidence | Caveat |
|---|---|---:|---|
| Transfer of securities with negative units | `lo` | Medium | ByAllAccounts APX integration evidence. |
| Transfer of securities with positive units | `li` | Medium | ByAllAccounts APX integration evidence. |

### E.6.5 APX Reversal / Cancellation

| Statement | Confidence | Caveat |
|---|---:|---|
| Custodial Integrator translates APX reversal transactions by converting original transaction type code to uppercase. | Medium | Integration evidence only. |
| Example: original `by` becomes reversal `BY`. | Medium | Integration evidence only. |
| If reversal fields do not match original transaction fields, APX may generate a Trade Blotter error requiring resolution. | Medium | Integration evidence only. |
| Universal native uppercase-cancellation behavior across all APX versions/import paths remains Unknown. | Unknown | Requires vendor or production evidence. |

### E.6.6 APX Additional Observed Transaction Parameters

The ByAllAccounts APX guide also describes parameters that resemble the Axys configuration names but apply to APX in this guide. These are integration configuration parameters, not native APX database fields.

| Observed Field / Parameter | Meaning in Source | Confidence | Caveat |
|---|---|---:|---|
| Broker column | A value such as `$brok` can be used when Broker must be populated. | Medium | Integration behavior; native column name Unknown. |
| Commission column | Can be populated or suppressed by configuration. | Medium | Integration behavior. |
| Lot location column | `axyslotlocation` parameter provides value used for lot location column; example `253`. | Medium | Parameter name appears legacy; native APX model Unknown. |
| Currency code | `axyscur` defines APX system currency in the guide's parameter table. | Medium | Integration configuration. |
| Default account identifier | Can use WebPortfolio account number, internal ID, or account name for untranslated accounts. | Medium | Integration behavior. |
| Dividend quantity default | `defdivquan` can provide a quantity value for dividend transactions with no reported quantity. | Medium | Integration behavior. |

## E.7 IMEX — Updated Evidence

### E.7.1 Confirmed / Strengthened

| Statement | System | Confidence | Caveat |
|---|---|---:|---|
| IMEX is explicitly defined by ByAllAccounts as the Axys Import/Export utility. | Axys | Medium | Third-party guide. |
| `imex32.exe` is the executable CI looks for to use the Axys Import/Export utility. | Axys | Medium | Third-party guide. |
| Axys Import/Export utility can create a `topost.trn` Trade Blotter file if one does not exist in the configured user folder. | Axys | Medium | Third-party integration evidence. |
| Axys Import/Export utility logs are retained and can be reviewed in a View IMEX Logs dialog. | Axys | Medium | Exact log schema Unknown. |
| AdventGuru says IMEX allowed CSV, tab, and fixed-format import/export after Axys v2.x introduced binary files. | Axys | Medium | Consultant evidence. |
| AdventGuru says APX v1.x through v4.x retained IMEX functionality but eliminated fixed-format generation. | APX | Medium | Consultant evidence. |
| AdventGuru says IMEX plus Trade Blotter import provides a practical path to move fundamental data in/out of Axys/APX. | Axys/APX | Medium | Consultant evidence. |

### E.7.2 Still Unknown

| Unknown | Priority | Notes |
|---|---:|---|
| Official Axys IMEX transaction object names | High | Not found publicly. |
| Official APX IMEX transaction object names | High | Not found publicly. |
| Official transaction export field list | High | Not found publicly. |
| Official transaction import field list | High | Not found publicly. |
| Official IMEX log schema / error codes | Medium | Only general log existence found. |
| Whether APX IMEX transaction objects have fields identical to Axys Trade Blotter fields | High | Not verified. |

## E.8 REP / Reports — Updated Evidence

### E.8.1 Transaction Summary Report

The Advent/SSRS Wealth Management Reports PDF provides visible Transaction Summary Report evidence. It should be treated as report-output evidence only, not REP source code, IMEX schema, or database schema.

| Report Section | Visible Fields | Confidence |
|---|---|---:|
| Purchases | Trade Date, Settle Date, Quantity, Symbol, Security, Unit Price, Amount | Medium |
| Sales | Trade Date, Settle Date, Quantity, Symbol, Security, Unit Cost, Total Cost, Unit Price, Proceeds, Gain/Loss | Medium |
| Dividends | Ex-Date, Pay-Date, Symbol, Security, Amount | Medium |
| Contributions | Trade Date, Settle Date, Quantity, Symbol, Security, Unit Price, Amount | Medium |
| Withdrawals | Trade Date, Settle Date, Quantity, Symbol, Security, Unit Price, Amount | Medium |

The same report sample describes the Transaction Summary Report as displaying account transactions maintained by Advent and serving as an independent record apart from the custodian.

### E.8.2 Report Menu / Category Evidence

The visible report sample lists transaction display categories including:

- Call
- Contribution
- Dividends
- Expenses
- Interest
- Maturities
- Purchase
- Puts
- Return of Capital
- Principal Payments
- Sales
- Withdrawals

**Confidence:** Medium. These are visible report-sample categories, not official transaction codes.

### E.8.3 Still Unknown

| Unknown | Priority | Notes |
|---|---:|---|
| Official REP report file name for Transaction Summary Report | High | Not confirmed. |
| Report parameters for Transaction Summary Report | High | Not found. |
| Whether Transaction Summary Report is REP/Replang, SSRS, APX report, or multiple implementations | High | Sample is Advent/SSRS; exact implementation may vary. |
| Whether report reads posted transactions, audit-trail data, calculated views, or database tables | High | Not found. |
| Whether Axys has an equivalent standard transaction summary report and exact columns | Medium | Not confirmed by public evidence. |

## E.9 Audit Trail — Updated Evidence

### E.9.1 `didpost.aud`

AdventGuru provides two separate pieces of consultant evidence about `didpost.aud`.

| Statement | System | Confidence | Caveat |
|---|---|---:|---|
| `didpost.aud` is described as the Audit Trail file. | Axys/APX | Medium | Consultant evidence. |
| The file is described as a critical component of Advent APX and Axys. | Axys/APX | Medium | Consultant evidence. |
| It allows users to review transactions posted to portfolios. | Axys/APX | Medium | Consultant evidence. |
| It may facilitate small- and large-scale removal of transactions posted in error. | Axys/APX | Medium | Operational claim; native controls Unknown. |
| In the APX-to-Axys conversion article, records posted through the Trade Blotter are described as stored in `didpost.aud` for recordkeeping. | Axys | Medium | Conversion-specific article; universality Unknown. |

### E.9.2 Axys 3.x Audit Trail Export Quirk

| Statement | System | Confidence | Caveat |
|---|---|---:|---|
| AdventGuru reports that large Axys 3.x Audit Trail exports through IMEX stopped working reliably in multiple sites reviewed by the consultant. | Axys | Medium | Consultant observation, not vendor confirmation. |
| The issue was characterized as an Axys issue, not necessarily an APX issue; APX export was not recently tested by the author. | Axys/APX | Medium | Do not generalize to APX. |
| The author preferred exported files but resorted to direct `AUD` file read/write only because export was unreliable. | Axys | Medium | Reinforces direct-file-access caution. |
| Proactive management of very large `didpost.aud` files may include backups, exports, or periodic file rotation, but such processes are operationally risky. | Axys/APX | Medium | Consultant evidence; not vendor best-practice documentation. |

### E.9.3 Audit Trail Unknowns

| Unknown | Priority |
|---|---:|
| Full `didpost.aud` native file layout | High |
| Whether `didpost.aud` is identical between Axys and APX | High |
| Whether every posted transaction always appears in `didpost.aud` | High |
| How deleted/reversed transactions are represented in `didpost.aud` | High |
| Whether `didpost.aud` stores original and corrected transaction versions | High |
| Whether `didpost.aud` can be reliably exported in current Axys/APX versions | Medium |

## E.10 Updated Field Dictionary Additions

These fields/labels should be added or promoted in the future Chapter 05 field dictionary.

| Field / Label | Axys | APX | IMEX | REP | Confidence | Notes |
|---|---|---|---|---|---:|---|
| Transaction Type | Observed as `Axys Transaction Type` | Observed as `APX Transaction Type` | Unknown | Unknown | Medium | Integration field label. |
| Transaction Src/Dest Type | Observed | Observed | Unknown | Unknown | Medium | Integration field label. |
| Transaction Src/Dest Symbol | Observed | Observed | Unknown | Unknown | Medium | Integration field label. |
| Transaction Special Security Type / Symbol | Observed | Observed | Unknown | Unknown | Medium | Integration field label. |
| Broker column | Observed through CI parameter | Observed through CI parameter | Unknown | Unknown | Medium | Required when commission value is populated in CI workflow. |
| Commission column | Observed through CI parameter | Observed through CI parameter | Unknown | Unknown | Medium | Can be populated/suppressed in CI workflow. |
| Lot location column | Observed through CI parameter | Observed through CI parameter | Unknown | Unknown | Medium | Native lot-location model Unknown. |
| Mark to Market field | Observed in Axys CI parameter | Unknown | Unknown | Unknown | Medium | Required for non-system-currency transactions in source. |
| Perf/CW column | Observed in Axys `topost.trn` parameter | Unknown | Unknown | Unknown | Medium | Exact meaning Unknown. |
| Withholding Tax field | Observed in WealthTechs Axys AIA example | Unknown | Unknown | Unknown | Medium | Integration-specific. |
| Unit Cost | Unknown | Observed in report output | Unknown | Observed | Medium | Transaction Summary Report Sales. |
| Total Cost | Unknown | Observed in report output | Unknown | Observed | Medium | Transaction Summary Report Sales. |
| Proceeds | Unknown | Observed in report output | Unknown | Observed | Medium | Transaction Summary Report Sales. |
| Gain/Loss | Unknown | Observed in report output | Unknown | Observed | Medium | Transaction Summary Report Sales. |
| Ex-Date | Unknown | Observed in report output | Unknown | Observed | Medium | Transaction Summary Report Dividends. |
| Pay-Date | Unknown | Observed in report output | Unknown | Observed | Medium | Transaction Summary Report Dividends. |

## E.11 Updated Observed Artifacts Matrix

| Artifact | System | Observed Role | Confidence | Caveat |
|---|---|---|---:|---|
| `topost.trn` | Axys | Trade Blotter file receiving transaction imports; CI appends generated transactions. | Medium | Integration evidence. |
| `ptopost.trn` | Axys | Position file generated by CI in CSV format. | Medium | Position context; included for interface context only. |
| `didpost.aud` | Axys/APX | Audit Trail file for posted transactions. | Medium | Consultant evidence; full layout Unknown. |
| `imex32.exe` | Axys | Import/Export utility. | Medium | Third-party guide. |
| `pospos32.exe` | Axys | Post Positions utility. | Medium | Position context. |
| `sec.inf` | Axys | Security information file used by CI for transaction/position generation. | Medium | Full native layout Unknown. |
| `type.inf` | Axys | Security type information file used by CI. | Medium | Full native layout Unknown. |
| `*.cli` | Axys/APX conversion context | Portfolio/client files used in conversion and CI portfolio-code lookup. | Medium | Full native layout Unknown. |
| `*.pri` | Axys | Price files used by CI for price merging/import. | Medium | Pricing context. |

## E.12 Updated Known Quirks

| Quirk | System | Confidence | Notes |
|---|---|---:|---|
| Code-only interpretation is unsafe. | Axys/APX | High as design rule; Medium source evidence | Strengthened by sign-dependent `li`/`lo`, fee special security fields, and source/destination fields. |
| Uppercase reversal is observed but not proven universal. | Axys/APX | Medium | ByAllAccounts documents uppercase conversion; native universality Unknown. |
| `topost.trn` append behavior matters operationally. | Axys | Medium | Generated imports may append to existing blotter content. |
| Beginning/ending comment transactions may bound generated import blocks. | Axys | Medium | CI-specific. |
| Broker may be required when Commission is populated. | Axys/APX | Medium | Integration parameter evidence. |
| Lot location exists as a transaction-related field/column in integration workflows. | Axys/APX | Medium | Native model Unknown. |
| Non-system-currency transactions may require a Mark to Market field in Axys CI workflow. | Axys | Medium | Native multicurrency mechanics Unknown. |
| Axys 3.x large Audit Trail export via IMEX may fail in some environments. | Axys | Medium | Consultant evidence only; not generalized to APX. |
| Direct native-file handling should remain last resort. | Axys/APX | Medium | Reinforced by AdventGuru caution. |
| Withholding tax treatment can be implemented multiple ways in Axys AIA workflow. | Axys | Medium | Integration-specific. |
| Original-face value can be written to quantity in Axys AIA workflow. | Axys | Medium | Integration-specific; especially relevant for fixed income. |

## E.13 Updated Unknowns After Second Pass

### E.13.1 High-Priority Unknowns Still Open

| ID | Unknown | Status |
|---|---|---|
| TU-001 | Complete official Axys transaction-code matrix. | Not found publicly. |
| TU-002 | Complete official APX transaction-code matrix. | Not found publicly. |
| TU-006 | Official Axys IMEX transaction export object names. | Not found publicly. |
| TU-007 | Official Axys IMEX transaction import object names. | Not found publicly. |
| TU-008 | Official APX IMEX transaction export/import object names. | Not found publicly. |
| TU-009 | Complete official IMEX transaction field list. | Not found publicly. |
| TU-010 | Official native Trade Blotter import layout. | Only partial third-party evidence found. |
| TU-014 | Official APX Transaction Summary Report parameters and full implementation details. | Public report-output sample found; official parameter spec not found. |
| TU-019 | Native Axys transaction storage model. | Not found publicly. |
| TU-020 | Native APX transaction storage model. | Not found publicly. |
| TU-021 | Native transaction identifiers. | Not found publicly. |
| TU-028 | Native reversal representation. | Integration-level uppercase evidence found; native universality Unknown. |
| TU-031 | Native retention model for deleted transactions. | `didpost.aud` evidence improved; detailed retention still Unknown. |

### E.13.2 Unknowns That Were Reduced But Not Closed

| Topic | What Improved | What Remains Unknown |
|---|---|---|
| Axys Trade Blotter import file | `topost.trn`, append behavior, comment boundaries, and folder location strengthened. | Complete official layout, required fields, validation rules. |
| Transaction fields | Several observed fields/columns identified through CI parameters. | Official IMEX/database/REP names and definitions. |
| APX transaction codes | Default integration matrix expanded. | Official native code manual and version differences. |
| Audit trail | `didpost.aud` evidence strengthened. | Native layout, retention, deleted/corrected transaction model. |
| REP report fields | Transaction Summary Report visible fields expanded. | REP implementation, report parameters, source of values. |
| IMEX logs | Existence and review workflow strengthened. | Log schema, error codes, machine-readable fields. |

## E.14 Practical Implications for Chapter 05

The future chapter should incorporate the following changes:

1. **Do not describe the transaction code matrix as official.** It should remain an observed integration-code matrix.
2. **Promote observed transaction fields** such as Transaction Type, Src/Dest Type, Src/Dest Symbol, Special Security Type/Symbol, Broker, Commission, Lot Location, Mark to Market, Perf/CW, and Withholding Tax where source evidence supports them.
3. **Add `topost.trn` append behavior** and beginning/ending comment transaction behavior to the Axys Trade Blotter section.
4. **Add `didpost.aud`** as an Axys/APX audit-trail artifact, with Medium Confidence and explicit unknowns around layout/retention.
5. **Add Axys Audit Trail export quirk**: large Axys 3.x Audit Trail exports via IMEX may fail in some environments, per consultant evidence.
6. **Expand Transaction Summary Report section** with Purchases, Sales, Dividends, Contributions, and Withdrawals field tables.
7. **Add withholding-tax options** and original-face quantity handling as Axys AIA integration quirks.
8. **Keep official IMEX object names Unknown.** No public source found official transaction object definitions.
9. **Keep native Axys/APX storage schemas Unknown.** Public sources did not resolve these.
10. **Separate product capabilities from mechanics.** SS&C product pages support broad capabilities, not field-level implementation details.

## E.15 Minimum Additional Material Still Needed

Even after two independent research passes, the following would materially improve Chapter 05:

| Needed Material | Would Resolve |
|---|---|
| Official Axys transaction-code manual | Native Axys transaction code matrix, deprecated codes, version differences. |
| Official APX transaction-code manual | Native APX transaction code matrix, deprecated codes, version differences. |
| Official Axys IMEX manual | Transaction object names, fields, data types, import/export rules. |
| Official APX IMEX manual | Transaction object names, fields, data types, import/export rules. |
| Official Trade Blotter layout documentation | Required/optional fields, validation, import behavior. |
| Sanitized production `topost.trn` files | Field order, examples, variation by transaction type. |
| Sanitized production IMEX transaction exports | Export schema and field naming. |
| Sanitized `didpost.aud` export | Audit trail fields, correction/reversal/deletion visibility. |
| Official REP/Replang report definitions | Transaction report names, parameters, fields, source behavior. |
| APX database schema or sanitized SQL extracts | Native APX transaction storage model. |

## E.16 Recommended Confidence Changes

| Topic | Prior Confidence | Updated Confidence | Reason |
|---|---:|---:|---|
| Axys `topost.trn` file name | Medium | Medium, stronger support | Multiple public integration/consultant sources. |
| Axys `topost.trn` append behavior | Unknown/Medium | Medium | ByAllAccounts Axys guide states append behavior. |
| Axys IMEX log existence | Medium | Medium, stronger support | ByAllAccounts Axys guide references View IMEX Logs workflow. |
| `didpost.aud` as audit trail | Medium | Medium, stronger support | AdventGuru has two relevant articles. |
| Axys/APX uppercase cancellation | Medium | Medium, stronger support | ByAllAccounts Axys and APX guides both document behavior. |
| APX observed code matrix | Medium | Medium, stronger support | ByAllAccounts APX default translation table expanded. |
| Official native code matrix | Unknown | Unknown | Still not found. |
| Official IMEX transaction objects | Unknown | Unknown | Still not found. |
| REP Transaction Summary visible fields | Medium | Medium, stronger support | Advent report sample visibly lists sections/fields. |

## E.17 Final Research Position After Second Pass

The available public evidence is now strong enough to write a useful, implementation-aware Chapter 05 technical reference, provided it maintains strict confidence labels.

The chapter can responsibly document:

- Axys/APX transaction centrality.
- Trade Blotter workflow evidence.
- `topost.trn` as observed Axys Trade Blotter file.
- `didpost.aud` as observed Axys/APX Audit Trail file.
- Observed Axys/APX integration transaction fields.
- Observed APX and Axys transaction-code mappings from ByAllAccounts.
- Uppercase reversal behavior as integration evidence.
- IMEX as an important interface and log-producing utility.
- REP/report evidence for Transaction Summary Report output fields.
- Known integration quirks around fees, transfers, original face, withholding tax, commissions, brokers, lot location, and multicurrency fields.

The chapter cannot responsibly document without Unknown labels:

- complete native transaction code matrices,
- official IMEX transaction object names,
- native Axys storage layout,
- native APX database schema,
- complete Trade Blotter field layout,
- native audit-retention behavior,
- REP/Replang source definitions.

## E.18 Deep IMEX Addendum Incorporated 2026-06-30

Source: `axys_imex_deep_research.md`.

Additional transaction-specific points:

| Topic | Addendum | Confidence |
|---|---|---:|
| `topost.trn` role | Public CI evidence supports `topost.trn` as the Axys Trade Blotter file in `$pathtrn`; generated transactions are appended and existing transactions are left unchanged. | Verified for CI workflow |
| File creation | If the Trade Blotter file does not exist, Axys Import/Export can create one in the configured user folder in the CI workflow. | Verified for CI workflow |
| Comment boundaries | CI can create beginning and ending comment transactions around its generated block. | Verified for CI workflow |
| Transaction comments | CI can include source transaction information as Trade Blotter comments that do not post unless configured. | Verified for CI workflow |
| Candidate transaction fields | A live Axys catalog should inspect portfolio, transaction code/subtype, trade/settle/post/effective dates, symbol/type, quantity, price, gross/net/cash amounts, commission, fees, accrued interest, withholding, source/destination type and symbol, cost fields, Perf/CW, Mark to Market, currency, FX, external source ID, and comments. | Discovery guidance |
| External-flow caution | Source/destination and special-security fields are observed integration labels, but official IMEX availability remains Unknown. Transaction-code-only interpretation remains unsafe for `li`, `lo`, `dp`, and `wd`. | Medium / Unknown boundary |
| REP fallback | When IMEX does not expose enough transaction context to classify flows, report extraction or custom REP output should be considered for classification evidence. | Design guidance |
