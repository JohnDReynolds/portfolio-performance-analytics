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
External source data
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

The processing pipeline focuses on transformation from raw source data to posted accounting records.

```text
Acquire Source Data
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
| Acquire Source Data | Obtain transactions from custodian, broker, OMS, manual entry, or provider. | Missing file, stale file, incomplete batch. |
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
