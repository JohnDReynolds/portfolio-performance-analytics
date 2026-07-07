# Chapter 05 — Transactions

> **Repository chapter:** `docs/axys-apx-reference/reference/Chapter_05_Transactions.md`\
> **Status:** Technical reference chapter based only on supplied research
> material.\
> **Evidence standard:** Facts are marked as Verified, High Confidence,
> Medium Confidence, Low Confidence, or Unknown.\
> **Important limitation:** The supplied material does not include
> official complete Axys or APX transaction-code manuals, official IMEX
> transaction object definitions, complete native Trade Blotter layouts,
> native Axys/APX transaction storage schemas, or complete REP
> specifications. Unsupported information is marked **Unknown**.

------------------------------------------------------------------------

## Related chapters
- [Chapter_01_Overview.md](Chapter_01_Overview.md) — provides the repository-wide map and evidence conventions.
- [Chapter_04_Security_Master.md](Chapter_04_Security_Master.md) — transaction interpretation depends on security identity.
- [Chapter_06_Holdings.md](Chapter_06_Holdings.md) — transactions drive holdings, lots, and cost-basis updates.
- [Chapter_10_Performance.md](Chapter_10_Performance.md) — transactions feed performance and attribution inputs.

## Canonical Use of This Chapter

This chapter is the canonical reader-facing reference for Axys/APX transaction
semantics in this repository. Other chapters should describe their local effects
on holdings, cash, corporate actions, performance, or reporting, but should not
maintain separate transaction-code dictionaries. When a transaction code appears
in another chapter, interpret it through this chapter and the implementation
contract in
[`../contracts/transaction_semantics_matrix.yaml`](../contracts/transaction_semantics_matrix.yaml).

The evidence archive in
[`../evidence/Research_05_Transactions.md`](../evidence/Research_05_Transactions.md)
preserves source-by-source research and may contain older wording retained for
provenance. If this chapter, the research archive, and the implementation
contract appear to disagree, treat that as a documentation cleanup issue rather
than as three independent sources of truth.

## 1. Overview

Transactions are the central accounting events in Axys/APX-style
portfolio accounting workflows. They connect economic activity to
holdings, cash, tax lots, cost basis, realized gain/loss, income,
performance, reports, IMEX, REP, reconciliation, and audit workflows.\
**Confidence:** High for general accounting role; Medium for native
Axys/APX mechanics because public source material is largely third-party
integration or migration evidence.

A practical transaction lifecycle supported by the supplied research is:

``` text
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

### 1.1 Evidence Boundaries

  --------------------------------------------------------------------------
  Area               Status                      Confidence Notes
  ------------------ ---------------- --------------------- ----------------
  Transactions       Supported                         High General
  affect holdings,   conceptually and                       accounting role
  cash, income, cost by supplied                            is strong;
  basis, realized    research                               native mechanics
  gain/loss, and                                            vary.
  performance inputs

  Trade Blotter      Supported                       Medium Evidence is
  workflows exist in                                        third-party
  Axys/APX                                                  integration
  integration                                               documentation.
  evidence

  Observed           Supported as                    Medium Not a complete
  transaction codes  observed codes                         official code
  such as `by`,                                             matrix.
  `sl`, `li`, `lo`,
  `dv`, `in`, `dp`,
  `wd`

  Uppercase          Supported in                    Medium Universality
  cancellation       third-party                            across versions
  behavior,          Axys/APX                               and import
  e.g. `by` → `BY`   workflows                              methods is
                                                            Unknown.

  Complete native    Unknown                        Unknown Not supplied.
  Axys
  transaction-code
  matrix

  Complete native    Unknown                        Unknown Not supplied.
  APX
  transaction-code
  matrix

  Official IMEX      Unknown                        Unknown Not supplied.
  transaction object
  names

  Native Axys        Unknown                        Unknown Not supplied.
  transaction
  storage model

  Native APX         Unknown                        Unknown Not supplied.
  database
  transaction schema

  Native audit trail Unknown                        Unknown Not supplied.
  and posting-status
  model
  --------------------------------------------------------------------------

### 1.2 Practical Classification Guidance

For performance, audit, and reconciliation work, it is useful to classify
transactions into practical buckets such as external cash flows,
trading activity, income, fees or expenses, corporate actions, and
reversals or cancellations. External cash flows are especially important
because deposits, withdrawals, transfers in or out, and cash sweeps can
materially change cash balances and performance cash-flow interpretation.

Observed examples in the supplied research include `by` and `sl` for
security trades, `dv`, `in`, and `ai` for income-related activity,
`li` and `lo` for transfer/external-flow candidates, and `dp` and `wd`
for cash-related movement. The meaning of any one code should be
determined with the surrounding context: sign, quantity, amount, security
type, source/destination type, source/destination symbol, special
security markers, performance/cash-flow flag if available, and
firm-specific translation rules.

For performance work, `li` and `lo` are the most important external-flow
candidates, but they are not sufficient by themselves. A cash
contribution pattern may use `li` with cash-like source/destination
fields such as `$pty` and `$cash`; a security-in-kind transfer may also
use `li` with a non-cash security and quantity. Similarly, `lo` can
represent a client withdrawal or an outgoing security transfer, but it
can also appear in debit, correction, or internal movement contexts.
Therefore, external-flow treatment should be assigned only after
checking code, security type, source/destination fields, signs, special
symbols, and firm-specific mapping.

Recommended classification order for audit products:

1. Detect reversals, cancellations, and uppercase-code deletes before
   treating a record as a new economic event.
2. Apply firm-specific transaction translation overrides, because public
   integration tables explicitly allow special cases and
   financial-institution-specific customization.
3. Interpret security type and symbol, including cash, fee, income,
   margin, short, and dividend-wash symbols.
4. Interpret the code family: trade, transfer, income/accrual, fee/cash,
   corporate action, principal event, or journal/other.
5. Use amount and quantity signs to determine direction.
6. Assign performance treatment: external flow, internal return event,
   correction/reversal, non-performance event, or unknown.

### 1.3 Interpretation Rule

Transaction code alone is not sufficient to determine accounting
meaning. Interpretation may depend on:

  ----------------------------------------------------------------------------
  Context Item            Why It Matters                            Confidence
  ----------------------- ----------------------- ----------------------------
  Transaction code        Primary event                                   High
                          indicator.

  Quantity sign           May determine inflow                          Medium
                          versus outflow.

  Amount sign             May determine cash                            Medium
                          direction.

  Security type           May distinguish cash,                         Medium
                          security, bond, fee,
                          margin, sweep, or short
                          activity.

  Source/destination type May define the                                Medium
                          offsetting side of the
                          accounting entry.

  Source/destination      May identify cash,                            Medium
  symbol                  margin, short, or wash
                          symbols.

  Special security        Used in observed fee                          Medium
  type/symbol             and expense handling.

  Portfolio/account       May affect                                    Medium
  configuration           interpretation,
                          including
                          deliver-in/out
                          behavior.

  Custodian or interface  Integration-specific                          Medium
  translation             mappings may alter
                          native codes.

  Reversal/cancellation   Uppercase code may                            Medium
  context                 represent
                          deletion/cancellation
                          in observed workflows.
  ----------------------------------------------------------------------------

### 1.4 Transaction Matching Boundary

Transaction matching for audit and performance-comparison work should be
conservative. A source transaction identifier, when present and trustworthy, is
the preferred linkage between snapshots. Without that identifier, fallback
matching should use only strict one-to-one evidence such as a single transaction
on the same portfolio, trade date, security identifier, and transaction code in
both snapshots. If more than one candidate exists on either side, or if the
trade date, identifier, or code changed, the records should remain unmatched
unless site-specific evidence proves the linkage.

This boundary is intentional. Two identical transactions can legitimately occur
on the same day, transactions can be added or deleted between snapshots, and a
date change can alter weighted cash-flow treatment. Fuzzy matching may create a
plausible story that is wrong. When a transaction cannot be matched, reports
should show the Snapshot A and Snapshot B rows as unmatched while still
explaining the Modified Dietz effect of the observed source-data changes.

Observed AIA transaction-translation logic may be case-insensitive inside that
integration workflow, but ppar should preserve native transaction-code and
security-identifier case unless a site-specific mapping explicitly says
otherwise.

------------------------------------------------------------------------

## 2. Axys

### 2.1 Axys Transaction Role

  ------------------------------------------------------------------------
  Statement                               Confidence Notes
  --------------------- ---------------------------- ---------------------
  Axys supports            High for broad capability Source material
  portfolio accounting                               identifies this from
  workflows involving                                SS&C/Advent
  transactions,                                      product-level
  positions,                                         evidence, but not
  settlement/trade                                   detailed mechanics.
  information, tax-lot
  or average-cost
  accounting,
  reporting,
  performance
  measurement, and
  reconciliation.

  Axys transaction                            Medium Supported by
  import workflows can                               ByAllAccounts and
  route transactions                                 WealthTechs
  through a Trade                                    integration evidence.
  Blotter for review
  and posting.

  Axys native                                Unknown Not supplied.
  transaction file
  structure is fully
  known from the
  supplied material.

  Axys native                                Unknown Not supplied.
  transaction-code
  matrix is fully known
  from the supplied
  material.
  ------------------------------------------------------------------------

### 2.2 Axys Trade Blotter and IMEX Workflow

The supplied research describes an Axys-oriented third-party integration
workflow:

``` text
External financial institution data
    ↓
Aggregation / normalization layer
    ↓
Security and portfolio translation
    ↓
Transaction Trade Blotter file
    ↓
Axys IMEX import
    ↓
Trade Blotter review
    ↓
Post to Axys
```

**Confidence:** Medium. This is observed third-party integration
behavior, not proof of exclusive native Axys behavior.

### 2.3 Axys Files, Folder Labels, and Utilities Observed in Integration Evidence

  --------------------------------------------------------------------------
  Item             Observed Role                 Confidence Caveat
  ---------------- ------------------ --------------------- ----------------
  `topost.trn`     Trade Blotter file                Medium Third-party
                   receiving                                integration
                   transaction                              evidence.
                   imports.

  `$pathtrn`       Axys user folder                  Medium Integration
                   label for Trade                          workflow
                   Blotter location.                        evidence.

  `imex32.exe`     Axys Import/Export                Medium Exact native
                   utility referenced                       behavior and
                   by Custodial                             version coverage
                   Integrator.                              Unknown.

  IMEX logs        Logs generated                    Medium Exact log fields
                   during import.                           and messages
                                                            Unknown.

  `$pathcli`       Axys                              Medium Integration
                   portfolio/client                         workflow
                   files; `*.cli`;                          evidence.
                   used to create
                   portfolio-code
                   list in one
                   workflow.

  `$pathinf`       Contains `sec.inf`                Medium Integration
                   and `type.inf`;                          workflow
                   exported by                              evidence.
                   integration
                   software to
                   generate
                   transactions and
                   positions.

  `$pathpri`       Axys price-file                   Medium Integration
                   folder; `*.pri`.                         workflow
                                                            evidence.

  `$pathlog`       Folder where Axys                 Medium Integration
                   Import/Export logs                       workflow
                   are written.                             evidence.

  `*.cli`          Client/portfolio                  Medium Native full
                   files referenced                         layout Unknown.
                   in conversion and
                   integration
                   evidence.
  --------------------------------------------------------------------------

### 2.4 Axys `.cli` and Conversion Evidence

  ----------------------------------------------------------------------------------
  Topic            Axys Evidence                       Confidence Notes
  ---------------- ------------------------ --------------------- ------------------
  Per-share cost   Morningstar conversion                  Medium This is conversion
  basis            evidence states                                evidence, not a
                   per-share cost-basis                           full native `.cli`
                   data is converted only                         spec.
                   if provided in exported
                   Advent `.cli` file.

  Deliver-in /     `li` and `lo` may be                    Medium Code-only
  deliver-out      interpreted differently                        interpretation is
  interpretation   depending on a                                 unsafe.
                   transaction-setting code
                   in the Advent client
                   file.

  53rd-character   Setting code `Y` maps                   Medium Specific to
  setting          `li`/`lo` to                                   supplied
                   Deliver-In/Deliver-Out                         conversion
                   in Morningstar                                 evidence.
                   conversion; setting code
                   `N` maps them to
                   Credit/Debit of
                   Security.

  `none` or        Transactions linked to                  Medium Conversion-layer
  `client`         securities labeled                             behavior.
  securities       `none` or `client` may
                   be converted as
                   out-of-pocket cash.

  Principal        Principal paydowns from                 Medium Native Axys
  paydowns         Axys may create                                mechanics Unknown.
                   conversion
                   complications, including
                   zero-quantity cases.

  Transaction and  Transaction prices and                  Medium Exact native field
  historical       historical security                            names Unknown.
  prices           prices may be converted
                   if present in Axys
                   conversion inputs.
  ----------------------------------------------------------------------------------

### 2.5 Axys Reinvestment Evidence

  ------------------------------------------------------------------------
  Statement                               Confidence Notes
  --------------------- ---------------------------- ---------------------
  Axys distribution                           Medium Based on Morningstar
  reinvestment activity                              Axys conversion
  may appear as Buy                                  evidence.
  plus Distribution
  transaction pairs in
  conversion data.

  Reinvestment                                Medium Conversion
  representation can                                 observation.
  affect downstream
  realized and
  unrealized gain/loss
  reporting.

  Native Axys                                Unknown Not supplied.
  reinvestment
  representation is
  fully defined by
  supplied material.
  ------------------------------------------------------------------------

### 2.6 Axys Fee Evidence

  --------------------------------------------------------------------------
  Item             Observed Meaning            Confidence Caveat
                   in Supplied
                   Material
  ---------------- ---------------- --------------------- ------------------
  `epus`           Associated with                 Medium May be a
                   Management Fee                         transaction code,
                   conversion in                          label, security
                   Morningstar Axys                       type, or
                   conversion                             conversion-layer
                   evidence.                              term; official
                                                          definition
                                                          Unknown.

  `exus`           Associated with                 Medium May be a
                   Expense                                transaction code,
                   conversion in                          label, security
                   Morningstar Axys                       type, or
                   conversion                             conversion-layer
                   evidence.                              term; official
                                                          definition
                                                          Unknown.
  --------------------------------------------------------------------------

### 2.7 Axys Cancellation / Reversal Evidence

  ------------------------------------------------------------------------
  Statement                               Confidence Notes
  --------------------- ---------------------------- ---------------------
  WealthTechs Axys                            Medium Third-party workflow
  evidence documents                                 evidence.
  cancellation behavior
  using uppercase
  transaction code,
  e.g. `by` → `BY`.

  Uppercase                                  Unknown Not supported by
  cancellation behavior                              supplied material.
  is universal across
  all Axys versions,
  transaction types,
  and import methods.
  ------------------------------------------------------------------------

------------------------------------------------------------------------

## 3. APX

### 3.1 APX Transaction Role

  ------------------------------------------------------------------------
  Statement                               Confidence Notes
  --------------------- ---------------------------- ---------------------
  APX workflows include                       Medium Supported mainly by
  transaction import,                                third-party
  blotter                                            integration and
  review/posting,                                    consultant evidence.
  reporting,
  reconciliation, and
  database/reporting
  alternatives.

  APX users may use                           Medium Supported by
  SQL/database                                       consultant evidence.
  reporting/export
  alternatives in
  addition to IMEX.

  Native APX database                        Unknown Not supplied.
  transaction schema is
  fully known from
  supplied material.

  Native APX                                 Unknown Not supplied.
  transaction-code
  matrix is fully known
  from supplied
  material.
  ------------------------------------------------------------------------

### 3.2 APX Blotter Types Observed

  ------------------------------------------------------------------------
  Blotter          Observed Purpose            Confidence Caveat
  ---------------- ---------------- --------------------- ----------------
  Trade Blotter    AIA imports                     Medium Integration
                   transactions                           workflow
                   into this                              evidence.
                   blotter; can be
                   consolidated or
                   created per
                   custodian.

  Statement        Used to post                    Medium Integration
  Blotter          custodian                              workflow
                   statement                              evidence.
                   transactions;
                   can support
                   reconciliation
                   against OMS or
                   third-party data
                   using REX.

  Tax Lot Blotter  Used for                        Medium Integration
                   tax-lot-level                          workflow
                   reconciliation                         evidence.
                   of
                   APX-calculated
                   lots versus
                   custodian lots.

  Position Blotter Used for                        Medium Integration
                   importing                              workflow
                   positions into                         evidence.
                   APX.

  Account Blotter  Used for                        Medium Integration
                   importing                              workflow
                   account                                evidence.
                   information.

  Initial          Used to import                  Medium AIA setting;
  Transaction      positions as                           native APX
  Blotter          deliver-in                             behavior
                   transactions for                       Unknown.
                   accounts without
                   transactions,
                   when configured.
  ------------------------------------------------------------------------

### 3.3 APX Trade Blotter Organization Options

  ------------------------------------------------------------------------
  Option                Meaning                                 Confidence
  --------------------- --------------------- ----------------------------
  Consolidate Into One  Aggregate all                               Medium
  Blotter               transactions from all
                        custodians into one
                        trade blotter.

  Create One Blotter    Distribute                                  Medium
  Per Custodian         transactions into one
                        blotter per
                        custodian.

  No Trade Blotter      No trade blotter is                         Medium
                        created by AIA.
  ------------------------------------------------------------------------

### 3.4 APX Transaction Translation Model

The supplied research describes a third-party APX integration model in
which source transactions are normalized before APX transaction
generation.

  --------------------------------------------------------------------------
  Statement                                 Confidence Notes
  ----------------------- ---------------------------- ---------------------
  WebPortfolio interprets                       Medium ByAllAccounts APX
  financial-institution                                guide evidence.
  transaction data and
  assigns a normalized
  transaction type.

  Custodial Integrator                          Medium ByAllAccounts APX
  translates normalized                                guide evidence.
  transaction types into
  APX transactions.

  Some APX translations                         Medium Examples include
  depend on the sign of                                positive/negative
  amount or units.                                     transfer behavior.

  Positive-unit transfer                        Medium Integration default,
  maps to APX `li` in the                              not complete native
  default translation                                  documentation.
  table.

  Negative-unit transfer                        Medium Integration default,
  maps to APX `lo` in the                              not complete native
  default translation                                  documentation.
  table.

  Translation tables may                        Medium Integration behavior.
  be customized by
  financial institution.
  --------------------------------------------------------------------------

### 3.5 APX Observed Transaction/Blotter Fields

  --------------------------------------------------------------------------------------------
  Field            Description          Axys       APX        IMEX      REP         Confidence
  ---------------- -------------------- ---------- ---------- --------- --------- ------------
  APX Transaction  Transaction          Unknown    Observed   Unknown   Unknown         Medium
  Type             code/type used in
                   APX translation
                   table.

  APX Transaction  Source/destination   Unknown    Observed   Unknown   Unknown         Medium
  Src/Dest Type    security or cash
                   type.

  APX Transaction  Source/destination   Unknown    Observed   Unknown   Unknown         Medium
  Src/Dest Symbol  symbol,
                   e.g. cash-like
                   symbols.

  APX Transaction  Special security     Unknown    Observed   Unknown   Unknown         Medium
  Special Security type used in
  Type             fee/expense
                   examples.

  APX Transaction  Special security     Unknown    Observed   Unknown   Unknown         Medium
  Special Security symbol used in
  Symbol           fee/expense
                   examples.

  Broker           Field that can       Unknown    Observed   Unknown   Unknown         Medium
  Representative   receive `$brok` in
  Field            AIA workflow.

  Lot Location     Axys-era/APX         Observed   Observed   Unknown   Unknown         Medium
                   workflow concept     as
                   integrated into lot  Axys-era
                   accounting.          concept

  Comment          Transaction import   Unknown    Observed   Unknown   Unknown         Medium
                   comment or
                   standalone comment.
  --------------------------------------------------------------------------------------------

### 3.6 APX Initial Deliver-In Transactions

  ------------------------------------------------------------------------
  Statement                               Confidence Notes
  --------------------- ---------------------------- ---------------------
  AIA can create                              Medium AIA/APX workflow
  initial deliver-in                                 evidence.
  transactions from
  positions for
  accounts with no
  transactions.

  If transactions are                         Medium AIA workflow
  received on the same                               evidence.
  day as initial
  positions in that
  scenario, the
  transactions may be
  ignored and not
  written to the
  blotter.

  Tax lots may be                      Low to Medium Details incomplete.
  relevant to initial
  deliver-in
  construction.

  Native APX initial                         Unknown Not supplied.
  deliver-in behavior
  independent of AIA is
  fully known.
  ------------------------------------------------------------------------

### 3.7 APX Statement Transactions and Reconciliation

  -----------------------------------------------------------------------------
  Statement                                    Confidence Notes
  -------------------------- ---------------------------- ---------------------
  APX workflows may                                Medium WealthTechs APX
  distinguish posted                                      evidence.
  portfolio transactions
  from statement
  transactions.

  Statement transactions may                       Medium WealthTechs APX
  support reconciliation                                  evidence.
  against custodian or OMS
  data.

  APX may expose separate UI                       Medium Workflow evidence.
  tabs named `Transactions`
  and
  `Statement Transactions`
  in this workflow.
  -----------------------------------------------------------------------------

### 3.8 APX Comments and Broker Field

  ------------------------------------------------------------------------
  Topic                 Statement                               Confidence
  --------------------- --------------------- ----------------------------
  Transaction comments  Rules in Transaction                        Medium
                        Translation may apply
                        only to transaction
                        comments in certain
                        cases, while
                        standalone comments
                        always post to the
                        import transaction
                        file in the observed
                        workflow.

  Broker representative A `Use $brok` setting                       Medium
                        can write `$brok` to
                        the broker
                        representative field
                        in the transaction
                        blotter.

  `.cli` reference      `$brok` is described                        Medium
                        as typically defined
                        in the `.cli` file
                        for each APX
                        portfolio.

  Broker translations   Broker translations                         Medium
                        can map broker
                        representative values
                        to APX-specific
                        codes.
  ------------------------------------------------------------------------

### 3.9 APX Cash Sweeps, Margin Sweeps, Short Sweeps, and Merge Logic

  ----------------------------------------------------------------------------------------
  Feature             Observed Behavior                        Confidence Caveat
  ------------------- ----------------------------- --------------------- ----------------
  Cash sweep removal  AIA includes logic to remove                 Medium AIA behavior.
                      cash sweep transactions from
                      source transaction files.

  Margin and short    AIA has similar removal logic                Medium AIA behavior.
  sweep removal       for margin and short sweeps.

  Example sweep       Examples include                             Medium Source examples
  patterns            `DP,CAUS,CASH,CAUS,MMF`,                            only.
                      `DP,CAUS,CASH,CAUS,MARGIN`,
                      and
                      `DP,CAUS,CASH,CAUS,SHORT`.

  FX merge            AIA has options to merge FX                  Medium AIA behavior;
                      transactions.                                       native APX FX
                                                                          workflow
                                                                          Unknown.

  Accrued-interest    AIA has options to merge                     Medium AIA behavior.
  merge               accrued-interest
                      transactions.

  Dividend/interest   AIA has options to merge                     Medium AIA behavior.
  merge               dividend and interest
                      entries.
  ----------------------------------------------------------------------------------------

------------------------------------------------------------------------

## 4. IMEX

### 4.1 IMEX Role

  ------------------------------------------------------------------------------------
  Statement           Axys          APX                  Confidence Notes
  ------------------- ------------- ------------- ----------------- ------------------
  IMEX is an          Supported     Supported                Medium Consultant and
  import/export                                                     third-party
  mechanism used in                                                 integration
  Axys/APX workflows.                                               evidence.

  IMEX supports CSV,  Supported     Unknown                  Medium Axys-focused
  tab, and                                                          consultant
  fixed-format                                                      evidence.
  import/export in
  Axys according to
  consultant
  documentation.

  APX maintained IMEX Not           Supported                Medium Version-specific
  functionality from  applicable                                    consultant
  v1.x to v4.x, but                                                 evidence.
  fixed-format file
  generation was
  eliminated
  according to
  consultant
  documentation.

  IMEX plus           Supported     Supported                Medium Consultant
  transaction/label                                                 evidence.
  import through
  Trade Blotter can
  move fundamental
  data in and out of
  Axys/APX.

  Official IMEX       Unknown       Unknown                 Unknown Not supplied.
  transaction object
  names are known.

  Complete IMEX       Unknown       Unknown                 Unknown Not supplied.
  transaction field
  list is known.
  ------------------------------------------------------------------------------------

### 4.2 Axys IMEX Details Observed

  ----------------------------------------------------------------------------
  Detail           Observed Value              Confidence Caveat
  ---------------- ---------------- --------------------- --------------------
  Utility          `imex32.exe`                    Medium Third-party
                                                          integration
                                                          evidence.

  Import target    Trade Blotter /                 Medium Workflow-specific.
                   `topost.trn` in
                   observed
                   workflow.

  Logs             IMEX logs                       Medium Exact format
                   written to                             Unknown.
                   `$pathlog` in
                   observed
                   workflow.

  Input support    CSV, tab,                       Medium Version coverage
                   fixed-format                           Unknown.
                   according to
                   consultant
                   evidence.
  ----------------------------------------------------------------------------

### 4.3 APX IMEX Details Observed

  --------------------------------------------------------------------------
  Detail           Observed Value                Confidence Caveat
  ---------------- ------------------ --------------------- ----------------
  IMEX             APX maintained                    Medium Exact version
  availability     IMEX functionality                       behavior
                   in versions                              Unknown.
                   referenced by
                   consultant source.

  Fixed-format     Eliminated in APX                 Medium Needs official
  generation       according to                             confirmation.
                   consultant
                   documentation.

  Alternative      SQL/database                      Medium APX-specific
  access           reporting/export                         consultant
                   tools may be                             evidence.
                   available.

  Official         Unknown                          Unknown Not supplied.
  transaction
  import/export
  object names
  --------------------------------------------------------------------------

### 4.4 Candidate IMEX Transaction Fields

The following fields are expected from accounting practice and supplied
research, but official IMEX names are not supplied. Therefore the IMEX
column names should be treated as **Unknown** until official
documentation or production exports are obtained.

  ----------------------------------------------------------------------------------------------------
  Field         Description         Axys             APX              IMEX      REP         Confidence
  ------------- ------------------- ---------------- ---------------- --------- --------- ------------
  Portfolio     Portfolio/account   Expected         Expected         Unknown   Unknown        Unknown
                identifier.                                           name

  Security      Security identifier Expected         Expected         Unknown   Unknown        Unknown
                or symbol.                                            name

  Trade Date    Economic or         Expected         Expected         Unknown   Unknown        Unknown
                execution date.                                       name

  Settlement    Cash settlement     Expected         Expected         Unknown   Unknown        Unknown
  Date          date.                                                 name

  Transaction   Accounting event    Expected         Expected         Unknown   Unknown        Unknown
  Code          code.                                                 name

  Quantity      Units affected.     Expected         Expected         Unknown   Unknown        Unknown
                                                                      name

  Price         Execution price.    Expected         Expected         Unknown   Unknown        Unknown
                                                                      name

  Amount        Cash or transaction Expected         Expected         Unknown   Unknown        Unknown
                amount.                                               name

  Broker        Broker or           Unknown          Observed in      Unknown   Unknown     Medium for
                representative.                      blotter workflow name                APX blotter;
                                                                                           Unknown for
                                                                                                  IMEX

  Currency      Transaction or      Expected         Expected         Unknown   Unknown        Unknown
                settlement                                            name
                currency.

  FX Rate       Currency conversion Expected for     Expected for     Unknown   Unknown        Unknown
                rate.               multi-currency   multi-currency   name

  Comment       Free-form note.     Unknown          Observed in      Unknown   Unknown     Medium for
                                                     import workflow  name                         APX
                                                                                             workflow;
                                                                                           Unknown for
                                                                                                  IMEX
  ----------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

## 5. REP and Reports

### 5.1 Report and REP Evidence

  ----------------------------------------------------------------------------------
  Report /       System        Description               Confidence Notes
  Interface
  -------------- ------------- ------------------ ----------------- ----------------
  Transaction    APX / Advent  Displays account              Medium Report sample
  Summary Report reports       transactions                         and public
                               maintained by                        report-guide
                               Advent; sample                       evidence.
                               evidence includes
                               dates, quantity,
                               symbol, security,
                               unit price, and
                               amount.

  REP            Axys          Unknown                      Unknown Exact report
  transaction                                                       names,
  reports                                                           parameters, and
                                                                    fields not
                                                                    supplied.

  REP            APX           Unknown beyond     Unknown to Medium Transaction
  transaction                  Transaction                          Summary Report
  reports                      Summary Report                       exists, but
                               evidence                             exact REP
                                                                    implementation
                                                                    Unknown.

  Replang        Axys/APX      Consultant source             Medium Exact
  reports                      lists Replang as a                   transaction
                               report/export                        report code
                               alternative.                         Unknown.

  Report Writer  Axys/APX      Consultant source             Medium Exact
  Pro / Excel                  lists these as                       transaction
  export / ETL                 alternatives.                        fields Unknown.

  APX            APX           Consultant source             Medium Native schema
  SQL/database                 lists SQL/database                   Unknown.
  access                       access as an APX
                               reporting/export
                               alternative.
  ----------------------------------------------------------------------------------

### 5.2 Transaction Summary Report --- Observed Columns

The supplied research includes sample column groups for an APX/Advent
Transaction Summary Report.

  ------------------------------------------------------------------------
  Section               Observed Columns                        Confidence
  --------------------- --------------------- ----------------------------
  Dividends             Ex-Date, Pay-Date,                          Medium
                        Symbol, Security,
                        Amount

  Contributions         Trade Date, Settle                          Medium
                        Date, Quantity,
                        Symbol, Security,
                        Unit Price, Amount

  Withdrawals           Trade Date, Settle                          Medium
                        Date, Quantity,
                        Symbol, Security,
                        Unit Price, Amount
  ------------------------------------------------------------------------

### 5.3 Unknown REP Details

  -----------------------------------------------------------------------
  Question                            Status
  ----------------------------------- -----------------------------------
  Which REP reports expose            Unknown
  transactions in Axys?

  Which REP reports expose            Unknown
  transactions in APX beyond the
  Transaction Summary Report
  evidence?

  What are the official APX           Unknown
  Transaction Summary Report
  parameters?

  What are the official APX           Unknown
  Transaction Summary Report field
  names?

  Do REP reports read stored posted   Unknown
  records, recalculated values,
  staged blotter values, or a
  mixture?

  How do REP outputs reconcile to     Unknown
  IMEX exports and native accounting
  records?
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 6. Data Model

### 6.1 Conceptual Model

``` text
Portfolio Master
Security Master
Currencies / FX
Pricing
Corporate Actions
Configuration / Translation Tables
        ↓
Transactions
        ↓
Holdings
Cash
Tax Lots
Cost Basis
Income
Realized Gain/Loss
Performance Inputs
Reports / IMEX / REP / Reconciliation / Audit
```

### 6.2 Upstream Dependencies

  ----------------------------------------------------------------------------
  Dependency       Role                 Failure Mode                Confidence
  ---------------- -------------------- ---------------- ---------------------
  Portfolio Master Maps transaction to  Unknown                           High
                   account/portfolio.   portfolio,
                                        inactive
                                        account,
                                        duplicate
                                        mapping.

  Security Master  Identifies asset and Unknown             High conceptually;
                   security type.       security,                   Medium for
                                        duplicate         integration evidence
                                        security,
                                        ambiguous
                                        identifier.

  Currency / FX    Supports             Missing FX rate,                Medium
                   multi-currency       invalid
                   transaction and      currency,
                   base-currency        settlement
                   reporting.           mismatch.

  Pricing          Supports buy/sell    Missing price,      High conceptually;
                   valuation, cost      price                Medium for native
                   basis, and           inconsistent            field behavior
                   reconciliation.      with market
                                        close.

  Corporate        May generate or      Missing split,                  Medium
  Actions          alter transaction    return of
                   interpretation.      capital,
                                        paydown, reorg
                                        event.

  Translation      Maps                 Wrong                           Medium
  Configuration    custodian/source     transaction
                   records to Axys/APX  type, wrong
                   accounting form.     security, wrong
                                        portfolio.

  `.cli` / client  May affect           Misclassified                   Medium
  settings         interpretation,      deliver-in/out
                   e.g. `li`/`lo`       versus
                   behavior in Axys     credit/debit.
                   conversion evidence.
  ----------------------------------------------------------------------------

### 6.3 Downstream Dependencies

  --------------------------------------------------------------------------
  Downstream Area       Transaction Impact                        Confidence
  --------------------- ----------------------- ----------------------------
  Holdings              Buys, sells, transfers,                         High
                        splits, reinvestments,
                        and paydowns change
                        units/exposure.

  Cash                  Deposits, withdrawals,                          High
                        buys, sells, fees,
                        dividends, interest,
                        and settlements affect
                        cash.

  Tax Lots              Purchases, sales,                             Medium
                        transfers, initial
                        deliver-ins, and
                        corporate actions may
                        create/consume/modify
                        lots.

  Cost Basis            Purchases, sales,          High conceptually; native
                        transfers, return of               mechanics Unknown
                        capital, reinvestments,
                        and fees may affect
                        basis.

  Income                Dividends, interest,               High conceptually
                        withholding,
                        reinvestment legs, and
                        some bond events affect
                        income.

  Realized Gain/Loss    Sales, covers,             High conceptually; native
                        transfers, and lot                 mechanics Unknown
                        selection can create
                        realized gain/loss.

  Performance           Transactions affect                High conceptually
                        capital flows,
                        holdings, income,
                        prices, and historical
                        restatements.

  Reports / IMEX / REP  Posted records are                            Medium
                        exposed through reports
                        and interfaces.

  Audit /               Transactions are                                High
  Reconciliation        primary evidence for
                        accounting differences.
  --------------------------------------------------------------------------

### 6.4 Transaction Processing Pipeline

  -----------------------------------------------------------------------------
  Stage            Purpose            Typical Failure                Confidence
  ---------------- ------------------ ------------------- ---------------------
  Acquire Source   Obtain             Missing file, stale                Medium
  Data             transactions from  file, incomplete
                   custodian, broker, batch.
                   OMS, manual entry,
                   provider, or
                   conversion file.

  Normalize        Convert source     Bad dates, bad                     Medium
  Records          records into       signs, malformed
                   common             identifiers.
                   representation.

  Portfolio        Map external       Unknown or                         Medium
  Translation      account to         duplicate mapping.
                   internal
                   portfolio.

  Security         Map external       Unknown or                         Medium
  Translation      security to        ambiguous security.
                   internal security.

  Transaction      Map external type  Unsupported                        Medium
  Translation      to accounting      transaction, wrong
                   code/type.         direction, missing
                                      paired leg.

  Special          Apply sweeps, FX   Suppressed records,                Medium
  Processing       merge,             bad merge, wrong
                   accrued-interest   fee classification.
                   merge, fee
                   translation, tax
                   logic,
                   cancellation
                   handling.

  Validation       Check required     Missing quantity,                  Medium
                   fields and         price, FX, dates,
                   plausibility.      or invalid
                                      settlement
                                      sequence.

  Staging /        Hold records for   Exception,                         Medium
  Blotter          review.            cancellation
                                      mismatch, pending
                                      record.

  Posting          Commit transaction Posting failure,                   Medium
                   to accounting      partial batch,
                   records.           unresolved
                                      dependency.

  Downstream       Update holdings,   Position/cash/lot       High conceptually
  Updates          cash, lots, basis, inconsistency.
                   income, gain/loss.

  Reporting /      Expose records     Interface/report                   Medium
  Export           through reports,   mismatch.
                   IMEX, REP, SQL, or
                   other tools.
  -----------------------------------------------------------------------------

### 6.5 AIA APX Processing Order Observed

This table documents AIA/APX integration behavior, not confirmed native
APX processing order.

  -------------------------------------------------------------------------
                 Order Step            Applies To                Confidence
  -------------------- --------------- --------------- --------------------
                     3 Remove Pending  All files                     Medium
                       Records

                     4 Remove          Transactions                  Medium
                       Intra-Account
                       Journals

                     5 Remove Cash     Transactions                  Medium
                       Sweeps

                     6 Withholding Tax Transactions                  Medium
                       Logic

                     7 Merge FX        Transactions                  Medium
                       Transactions

                     8 Merge Accrued   Transactions                  Medium
                       Interest
                       Transactions

                     9 Transaction     Transactions                  Medium
                       Translations

                    12 Broker          Transactions                  Medium
                       Translations

                    15 Adjust Cancel   Transactions                  Medium
                       Transactions to
                       Upper Case

                    16 Apply           Transactions                  Medium
                       Transaction
                       Comment Logic

                    17 Merge Dividends Transactions                  Medium
                       and Interest

                    19 Post            Transactions                  Medium
                       Translations
                       Transaction
                       Translations

                    23 Add Interface   All files                     Medium
                       Comments
  -------------------------------------------------------------------------

------------------------------------------------------------------------

## 7. Common Fields

### 7.1 Core Transaction Field Dictionary

  ---------------------------------------------------------------------------------------------------------------
  Field         Description          Axys             APX              IMEX      REP                   Confidence
  ------------- -------------------- ---------------- ---------------- --------- ----------------- --------------
  Portfolio ID  Portfolio/account    Expected         Expected         Unknown   Unknown                     High
                identifier.                                            name                          conceptually

  Transaction   Accounting event     Observed in      Observed in      Unknown   Unknown               Medium for
  Code          code/type.           examples         examples         name                        codes; Unknown
                                                                                                     for official
                                                                                                           matrix

  Security      Security involved in Expected         Expected         Unknown   Symbol/Security             High
  Identifier    transaction.                                           name      observed in         conceptually
                                                                                 report sample

  Trade Date    Execution/economic   Expected         Expected         Unknown   Observed in                 High
                date.                                                  name      report sample       conceptually

  Settlement    Settlement/cash      Expected         Expected         Unknown   Observed as                 High
  Date          date.                                                  name      Settle Date in      conceptually
                                                                                 report sample

  Entry Date    Date entered into    Unknown          Unknown          Unknown   Unknown                  Unknown
                system.

  Posting Date  Date posted to       Unknown          Unknown          Unknown   Unknown                  Unknown
                accounting records;                                                               important for
                useful for explaining                                                              historical
                historical changes.                                                                 review.

  Quantity      Units traded or      Expected         Expected         Unknown   Observed in                 High
                affected.                                              name      report sample       conceptually

  Price         Unit price.          Expected         Expected         Unknown   Observed as Unit            High
                                                                       name      Price in report     conceptually
                                                                                 sample

  Gross Amount  Transaction value    Expected         Expected         Unknown   Unknown                   Medium
                before adjustments.                                    name                          conceptually

  Net Amount    Final cash amount.   Expected         Expected         Unknown   Amount observed           Medium
                                                                       name      in report sample    conceptually

  Commission    Trading commission.  Expected         Expected         Unknown   Unknown                     High
                                     optional         optional         name                          conceptually

  Fees          Trading or           Expected         Expected         Unknown   Unknown                     High
                administrative fees. optional         optional         name                          conceptually

  FX Rate       Currency conversion  Expected when    Expected when    Unknown   Unknown                   Medium
                rate.                multi-currency   multi-currency   name

  Broker        Broker or            Unknown          Broker           Unknown   Unknown           Medium for APX
                representative.                       representative   name                              workflow
                                                      field observed

  Batch ID      Import batch         Unknown          Unknown          Unknown   Unknown                  Unknown
                identifier.

  Source ID     External transaction Unknown          Unknown          Unknown   Unknown                  Unknown
                identifier.

  Cash Impact   Signed cash movement Unknown          Unknown          Unknown   Unknown                  Medium
                normalized for audit                                                                as recommended
                and performance use.                                                                   field

  Position      Signed quantity or   Unknown          Unknown          Unknown   Unknown                  Medium
  Impact        principal movement                                                                  as recommended
                normalized for audit                                                                   field
                use.

  Performance   Explicit source or   Unknown          Unknown          Unknown   Unknown                  Medium
  Cash-Flow     derived flag for                                                                    as recommended
  Flag          external-flow                                                                        field
                treatment.

  Classification Derived transaction Unknown          Unknown          Unknown   Unknown                  Medium
                family, such as                                                                     as recommended
                external flow, trade,                                                               field
                income, fee,
                corporate action,
                correction, or
                unknown.

  Classification Confidence assigned Unknown          Unknown          Unknown   Unknown                  Medium
  Confidence    by mapping/audit                                                                     as recommended
                logic.                                                                                 field

  Raw Line      Preserved source      Unknown          Unknown          Unknown   Unknown                  Medium
                record for audit                                                                     as recommended
                traceability.                                                                         field

  Comment       Free-form note.      Unknown          Observed in      Unknown   Unknown           Medium for APX
                                                      import workflow  name                              workflow
  ---------------------------------------------------------------------------------------------------------------

### 7.2 Public Example Transaction Row

The supplied research includes this public third-party example row:

``` text
acct123,010101,010101,by,csus,appl,100,caus,cash,10000
```

A cancellation example uppercases the transaction code:

``` text
acct123,010101,010101,BY,csus,appl,100,caus,cash,10000
```

Tentative interpretation:

    Position Observed Value   Tentative Meaning                               Confidence
  ---------- ---------------- ------------------------------------------ ---------------
           1 `acct123`        Account / portfolio code.                           Medium
           2 `010101`         Date field 1.                                      Unknown
           3 `010101`         Date field 2.                                      Unknown
           4 `by` / `BY`      Transaction code / cancellation code.               Medium
           5 `csus`           Security type.                               Low to Medium
           6 `appl`           Security symbol.                             Low to Medium
           7 `100`            Quantity.                                    Low to Medium
           8 `caus`           Source/destination type.                     Low to Medium
           9 `cash`           Source/destination symbol.                   Low to Medium
          10 `10000`          Cash amount / net amount / trade amount.           Unknown

This row is not a complete Axys/APX import layout.

------------------------------------------------------------------------

## 8. Transaction Codes

### 8.1 Warning

The following matrix is an observed-code catalog from supplied research.
It is **not** an official Axys or APX transaction-code reference. Codes
may be native, integration-layer mappings, version-specific,
configuration-dependent, context-dependent, or incomplete.

The strongest public transaction-code evidence in the supplied research
comes from ByAllAccounts Custodial Integrator default translation tables
and Morningstar Axys conversion notes. Those sources are useful for audit
design, but they do not establish a complete native Axys/APX code manual.
The July 2026 transaction-code addenda strengthened several practical
descriptions, especially for fixed-income accrued-interest and principal
paydown codes, while preserving the same evidence boundary.

### 8.2 Observed Transaction Code Matrix

  ---------------------------------------------------------------------------------------------------
  Code        Observed Meaning           Axys        APX             Confidence Notes
  ----------- -------------------------- ----------- ----------- -------------- ---------------------
  `by`        Buy                        Observed in Observed            Medium Public integration
                                         examples;                              documentation only.
                                         official
                                         status
                                         Unknown

  `BY`        Cancellation/deletion of   Observed    Observed            Medium Uppercase
              Buy                                                               cancellation
                                                                                observed;
                                                                                universality Unknown.

  `sl`        Sell                       Unknown     Observed            Medium Requires vendor
                                                                                confirmation.

  `ss`        Short sale                 Unknown     Observed            Medium Requires vendor
                                                                                confirmation.

  `cs`        Cover short                Unknown     Observed            Medium Observed in APX
                                                                                integration evidence.

  `li`        Deliver in / transfer in / Observed    Observed            Medium External-flow
              credit / deposit /                                                candidate; context
              positive movement                                                 determines final
                                                                                treatment.

  `lo`        Deliver out / transfer out Observed    Observed            Medium External-flow
              / debit / withdrawal /                                            candidate; context
              negative movement                                                 determines final
                                                                                treatment.

  `dv`        Dividend / income /        Unknown     Observed            Medium Often relevant to
              reinvestment leg                                                  reinvestment.

  `in`        Income / interest          Unknown     Observed            Medium Requires context.

  `rc`        Return of capital          Observed    Observed      Medium-High Confirmed in
                                                                                ByAllAccounts
                                                                                Axys/APX
                                                                                translation
                                                                                evidence; native
                                                                                performance and
                                                                                cost-basis
                                                                                treatment still
                                                                                needs site proof.

  `pd`        Principal paydown / bond   Observed    Observed      Medium-High Confirmed in
              return-of-capital case                                            ByAllAccounts
                                                                                bond-security
                                                                                return-of-capital
                                                                                mapping and
                                                                                Morningstar
                                                                                paydown
                                                                                conversion
                                                                                caveats.

  `ai`        Negative interest or       Unknown     Observed            Medium Narrower than
              margin interest                                                   generic accrued
                                                                                interest; context-
                                                                                dependent.

  `sa`        Sell-side accrued          Unknown     Observed            Medium Fixed-income
              interest                                                          trade adjunct;
                                                                                integration-level
                                                                                evidence.

  `pa`        Purchase accrued interest  Unknown     Observed            Medium Fixed-income
              / buy-side accrued                                                trade adjunct;
              interest                                                          integration-level
                                                                                evidence.

  `dp`        Debit / fee-related / tax  Unknown     Observed            Medium Multiple meanings;
              / service charge /                                                usually not an
              cash-security case                                                external flow without
                                                                                additional evidence.

  `wd`        Cash-security sell /       Unknown     Observed            Medium Context-dependent;
              withdrawal-like                                                   do not infer client
              cash-security movement                                            withdrawal from
                                                                                the code name.

  `;`         Journal / comment / other  Unknown     Observed            Medium Treat as observed
              / split in integration                                            integration behavior
              table                                                             only.
  ---------------------------------------------------------------------------------------------------

### 8.3 Observed APX Translation Patterns

  -----------------------------------------------------------------------------------
  Source Transaction  Observed APX                Confidence Notes
  Concept             Translation
                      Pattern
  ------------------- ---------------- --------------------- ------------------------
  ATM positive        `li`                            Medium Inflow-like.

  ATM negative        `lo`                            Medium Outflow-like.

  Buy                 `by`                            Medium Default table evidence.

  Cash security buy   `dp`                            Medium Special cash-security
                                                             case.

  Cover short         `cs`                            Medium Default table evidence.

  Check               `lo`                            Medium Withdrawal-like.

  Closure positive    `sl`                            Medium Positive closure maps to
                                                             sell in observed table.

  Closure negative    `cs`                            Medium Negative closure maps to
                                                             cover short in observed
                                                             table.

  Credit              `li`                            Medium Inflow-like.

  Debit non-cash      `lo`                            Medium Outflow-like.
  security

  Tax                 `dp` with                       Medium Examples include `epus`
                      special                                and withholding-related
                      type/symbol                            symbols.

  Deposit cash        `li`                            Medium Inflow-like.

  Deposit non-cash    `li` and `by`                   Medium Two-transaction case in
  security            pair                                   source.

  Direct debit        `lo`                            Medium Outflow-like.

  Direct deposit      `li`                            Medium Inflow-like.

  Dividend            `dv`                            Medium Income-related.

  Reinvested dividend `dv` and/or                     Medium Exact native behavior
                      paired buy                             Unknown.
                      behavior

  Fee                 `dp` with                       Medium Configurable.
                      special security
                      type/symbol such
                      as
                      `exus custfee`

  Recordkeeping fee   `dp` with                       Medium Source-table example.
                      `epus expense`

  Income bond         `li` / `lo`                     Medium Direction depends on
  security                                                   sign.
  positive/negative

  Income cash         `in`                            Medium Income-like.
  security

  Income              `dv`                            Medium Dividend-like.
  dividend-paying
  security

  Interest positive   `in`                            Medium Income-like.

  Interest negative   `ai`                            Medium Margin-interest-like
                                                             special case.

  Investment expense  `dp` with                       Medium Fee-like.
                      `exus custfee`

  Journal             `;`                             Medium Comment/journal-like.

  Margin interest     `ai`                            Medium Uses margin cash symbol
                                                             in source.

  Other               `;`                             Medium Generic/other.

  Payment             `lo`                            Medium Outflow-like.

  Point of sale       `li` / `lo`                     Medium Direction depends on
  positive/negative                                          sign.

  Reinvestment        `dv` and `by`                   Medium Source shows paired APX
                      pair                                   translation.

  Repeat payment      `lo`                            Medium Outflow-like.

  Return of capital   `rc`; bond                Medium-High `rc` is confirmed
                      security may map                      in public
                      to `pd`                               translation
                                                             mapping evidence;
                                                             bond-security
                                                             return of capital
                                                             maps to `pd`.

  Sell                `sl`                            Medium Normal sell.

  Sell cash security  `wd`                            Medium Cash-security special
                                                             case.

  Buy accrued         `pa`                            Medium Fixed-income
  interest                                                   trade adjunct;
                                                             not external
                                                             flow.

  Sell accrued        `sa`                            Medium Fixed-income
  interest                                                   trade adjunct;
                                                             not external
                                                             flow.

  Short               `ss`                            Medium Short sale.

  Service charge      `dp` with                       Medium Fee-like.
                      `exus custfee`

  Split               central split                   High   Prefer `split.inf`
                      factor data                            / split-factor
                                                             evidence; `;` is
                                                             only a local or
                                                             conversion marker.

  Transfer            `li` / `lo`                     Medium Direction depends on
  positive/negative                                          sign.

  Withdrawal          `lo`                            Medium Outflow-like.
  -----------------------------------------------------------------------------------

### 8.4 Practical Classification Matrix

The following matrix is a practical audit/performance aid derived from
observed research. It is not an official Axys/APX code dictionary.

| Code or marker | Practical group | External flow? | Performance / audit implication |
|---|---|---:|---|
| `li` | Transfer / external-flow candidate | Often, not always | Candidate contribution or security-in-kind transfer; verify cash/security fields and firm mapping. |
| `lo` | Transfer / external-flow candidate | Often, not always | Candidate withdrawal or outgoing security transfer; separate fees, corrections, and internal journals. |
| `by`, `sl`, `ss`, `cs` | Trading activity | No | Affects holdings, cash, realized gain/loss, and exposure; validate price, quantity, commission, and settlement cash. |
| `dv`, `in` | Income | No | Affects income and cash unless reinvested or netted; validate expected dividend/coupon and pay-date treatment. |
| `pa`, `sa` | Fixed-income trade adjunct | No | Validate accrued interest, settlement date, coupon schedule, day count, and total trade settlement. |
| `ai` | Negative interest / margin interest / financing adjustment | No | Validate margin or negative-interest context, amount sign, margin cash/security markers, and financing-rate support. |
| `dp`, `wd` | Cash, fee, expense, external-flow, or cash-security movement | Context-dependent | Treat as context-dependent; verify whether the record is a fee, expense, cash-security buy/sell, tax, internal movement, or true external movement. |
| `rc`, `pd` | Corporate action / principal event | Usually no | Confirmed in public Axys/APX translation mapping evidence as cash-producing security events; validate return-of-capital, paydown, cost-basis, factor, amortization, and performance-report treatment. |
| `;` | Journal, other, placeholder, or locally materialized split marker | Usually no | Prefer central split-factor evidence such as `split.inf` for normal split processing; require firm mapping before treating `;` as split evidence. |
| Uppercase code, e.g. `BY` | Reversal / deletion | Depends on original | Link to the original transaction and reverse original economics rather than treating it as independent activity. |
| `epus`, `exus` | Fee/expense security type or marker | No | Preserve as security/type context, not as a standalone transaction-code conclusion. |
| `dvwash` | Reinvestment wash symbol | No | Link dividend and reinvestment buy legs; avoid treating wash cash as external cash flow. |
| `caus margin` | Margin cash symbol/context | No | Supports margin-interest classification; separate financing expense from external flow. |

### 8.5 Minimal Firm-Specific Mapping to Collect

Because public research does not provide complete official native code
matrices, each implementation should collect a client-specific mapping
for ambiguous transaction families.

| Raw code | Security type | Security symbol | Source/destination type | Source/destination symbol | Firm meaning | External flow? | Notes |
|---|---|---|---|---|---|---:|---|
| `li` | Cash or non-cash | Client-specific | `$pty` or equivalent | `$cash` or equivalent | Contribution, security transfer in, correction, or other | Confirm | Identify cash versus in-kind transfer. |
| `lo` | Cash or non-cash | Client-specific | `$pty` or equivalent | `$cash` or equivalent | Withdrawal, security transfer out, correction, or other | Confirm | Separate client withdrawal from fees and journals. |
| `dp` | `epus`, `exus`, cash, or other | Fee/tax/cash symbol | Client-specific | Client-specific | Fee, expense, tax, cash-security buy, or other | Usually no | Confirm gross/net performance treatment. |
| `wd` | Cash or cash-like | Cash-security symbol | Client-specific | Client-specific | Cash-security sell, sweep, withdrawal-like event, or other | Confirm | Do not infer client withdrawal from name alone; require context before treating it as an external flow. |
| `;` | Any | Any | Any | Any | Journal, other, placeholder, or locally materialized split marker | Usually no | Route high-value items to manual review; prefer split-factor extracts for normal split evidence. |
| `dv` / `by` | Real security | Real symbol or `dvwash` | `$ity` or equivalent | `$income` or wash symbol | Dividend reinvestment | No | Link income and buy legs to avoid double counting. |
| `in` / `ai` | Bond, cash, or margin | Real symbol or margin symbol | Client-specific | Client-specific | Interest, negative interest, or margin interest | No | Validate coupon, accrual, or margin-rate support. |
| `pa` / `sa` | Bond or fixed-income security | Real symbol | Client-specific | Client-specific | Buy-side or sell-side accrued interest | No | Validate accrued interest, day count, and settlement economics; do not classify as external flow. |
| `rc` / `pd` | Real security or bond/MBS | Real symbol | Usually `$pty` in ByAllAccounts mapping evidence | Usually `$cash` in ByAllAccounts mapping evidence | Return of capital or principal paydown | No | Confirm context before classification. The packaged demo uses narrow context-gated examples only. |

### 8.6 Observed Public Mapping Examples

The supplied research now includes a public WebPortfolio-to-Axys mapping
contract. The examples below are representative mappings from that
contract and are not universal Axys transaction-code rules.

| Normalized source event | Sign/context | Observed Axys code(s) | Classification caution |
|---|---|---|---|
| Deposit / direct deposit / credit | Cash in | `li` | External inflow candidate; confirm outside-party source and cash fields. |
| Withdrawal / check / payment | Cash out | `lo` | External outflow candidate; separate client withdrawal from fees and journals. |
| Transfer | Positive / negative | `li` / `lo` | May be cash transfer, in-kind transfer, or internal movement. |
| Buy / sell | Security | `by` / `sl` | Trade activity, not external flow. |
| Short / cover short | Security | `ss` / `cs` | Trade activity; validate short exposure and cash signs. |
| Dividend / reinvestment | Income plus buy | `dv`, `by`, `dvwash` | Link paired legs and avoid double-counting income or wash cash. |
| Interest / negative interest | Positive / negative | `in` / `ai` | Separate income from margin or financing expense. |
| Accrued interest on buy/sell | Fixed-income trade adjunct | `pa` / `sa` | Part of bond trade settlement economics; not client external cash flow. |
| Fee / service charge / expense | Fee context | `dp` with `epus`/`exus` symbols | Fee or expense, not client withdrawal unless firm policy proves otherwise. |
| Cash-security buy/sell | Cash security | `dp` / `wd` | Cash-security handling; do not infer external cash flow from the code name. |
| Return of capital / paydown | Normal / bond | `rc` / `pd` | Issuer/principal event producing portfolio cash in public translation mapping evidence; not client capital flow. |
| Split / journal / other | Special marker | `;` | Requires firm mapping or manual review. |
| Reversal / deletion | Original code uppercased | e.g. `BY` | Reverse original economics; flag orphan reversals. |

Ambiguous `li`, `lo`, `dp`, `wd`, `;`, `epus`, and `exus` cases should
fall into a "requires review" bucket until firm mapping or supporting
source evidence confirms the audit classification.

### 8.7 Fixed-Income Accrued-Interest and Principal Codes

The July 2026 addendum strengthened but did not close the evidence for
`pa`, `sa`, `ai`, and `pd`. ByAllAccounts Axys/APX default translation
tables support the following practical interpretations:

| Code | Better-supported meaning | Practical treatment | External flow? |
|---|---|---|---:|
| `pa` | Purchase accrued interest / buy-side accrued interest. | Fixed-income trade adjunct; validate accrued days, coupon, day count, settlement date, and total settlement. | No |
| `sa` | Sale accrued interest / sell-side accrued interest. | Fixed-income trade adjunct; validate accrued days, coupon, day count, settlement date, and sale proceeds. | No |
| `ai` | Negative interest or margin interest. | Financing, negative-interest, or income-adjustment context; require margin/security context and amount sign. | No |
| `pd` | Principal paydown / bond-security return-of-capital transaction. | Principal event; validate cash received, principal/factor reduction, cost-basis/amortization treatment, and zero-quantity cases. | No |

These are observed integration mappings, not official native code
definitions. `pa` and `sa` should not be used as generic income codes
without bond/accrual context. `ai` should not be described as generic
accrued interest; the strongest public evidence supports negative
interest and margin interest. `pd` should remain a principal-paydown or
bond-return-of-capital candidate until security type, factor/principal
movement, and local methodology are known.

### 8.8 Cancellation and Reversal

  ------------------------------------------------------------------------------------------
  Statement                Axys             APX                     Confidence Notes
  ------------------------ ---------------- ---------------- ----------------- -------------
  Lowercase transaction    Observed         Observed                    Medium Third-party
  code may be uppercased                                                       workflows.
  to represent
  cancellation/deletion,
  e.g. `by` → `BY`.

  Cancellation transaction Unknown          Observed                    Medium APX
  fields must sufficiently                                                     integration
  match the original                                                           evidence.
  transaction or blotter
  error may occur.

  Cancellation blotters    Observed         Observed                    Medium WealthTechs
  may be created from                                                          evidence.
  historical transaction
  files.

  Cancellation workflows   Supported        Supported                   Medium Based on
  should be treated as     recommendation   recommendation                     source
  high-risk and backed                                                         warnings.
  up/reviewed.

  Uppercase cancellation   Unknown          Unknown                    Unknown Not supplied.
  is universal native
  behavior across all
  versions and import
  methods.
  ------------------------------------------------------------------------------------------

------------------------------------------------------------------------

## 9. Examples

### 9.1 Buy Example from Public Integration Evidence

``` text
acct123,010101,010101,by,csus,appl,100,caus,cash,10000
```

  Interpretation Item         Value            Confidence
  --------------------------- ----------- ---------------
  Account / portfolio         `acct123`            Medium
  Transaction code            `by`                 Medium
  Security type               `csus`        Low to Medium
  Security symbol             `appl`        Low to Medium
  Quantity                    `100`         Low to Medium
  Source/destination type     `caus`        Low to Medium
  Source/destination symbol   `cash`        Low to Medium
  Amount                      `10000`             Unknown

### 9.2 Cancellation Example

``` text
acct123,010101,010101,BY,csus,appl,100,caus,cash,10000
```

  ------------------------------------------------------------------------
  Interpretation Item   Value                                   Confidence
  --------------------- --------------------- ----------------------------
  Cancellation          `BY`, uppercase                             Medium
  indicator             version of `by`

  Native universality   Unknown                                    Unknown

  Required match fields Unknown               Unknown for Axys; Medium for
                                                  APX integration evidence
                                                 that mismatch can produce
                                                             blotter error
  ------------------------------------------------------------------------

### 9.3 Reinvestment Pattern

  ------------------------------------------------------------------------
  System           Observed Pattern            Confidence Caveat
  ---------------- ---------------- --------------------- ----------------
  Axys             Reinvestment may                Medium Conversion
                   appear as Buy                          evidence only.
                   plus
                   Distribution
                   transaction
                   pairs in
                   conversion data.

  APX              Reinvestment may                Medium Integration
                   translate as                           evidence only.
                   `dv` and `by`
                   pair in
                   ByAllAccounts
                   integration
                   evidence, often
                   with `dvwash`.
  ------------------------------------------------------------------------

### 9.4 Fee Pattern

  -------------------------------------------------------------------------
  System           Observed Pattern             Confidence Caveat
  ---------------- ----------------- --------------------- ----------------
  Axys             `epus` associated                Medium Official meaning
                   with Management                         Unknown.
                   Fee conversion;
                   `exus` associated
                   with Expense
                   conversion.

  APX              Fee transactions                 Medium Integration
                   may use `dp` plus                       evidence only.
                   special security
                   type/symbol such
                   as `exus custfee`
                   or
                   `epus expense`.
  -------------------------------------------------------------------------

### 9.5 Withholding-Tax and Negative-Income Patterns

Observed integration mappings allow several possible treatments for
withholding taxes or income reductions. These are classification
patterns, not verified native Axys/APX field rules.

| Pattern | Meaning | Audit implication |
|---|---|---|
| Separate expense line such as `dp` with `exus` or withholding symbol | Gross dividend plus separate tax/fee | Link income and tax lines for gross/net review. |
| Negative `dv` or negative `in` | Tax or adjustment reduces income directly | Easier net cash view, but gross income can be obscured. |
| Withholding-tax field on income transaction | Single income record carries tax detail | Prefer this when preserved because it keeps gross/net detail together. |

------------------------------------------------------------------------

## 10. Known Issues / Quirks

  ----------------------------------------------------------------------------------------------------------
  Issue / Quirk                           Axys          APX                  Confidence Notes
  --------------------------------------- ------------- ------------- ----------------- --------------------
  Code-only interpretation is unsafe.     Supported     Supported        High as design Use code, sign,
                                                                           rule; Medium security type,
                                                                        source evidence source/destination
                                                                                        fields, symbols, and
                                                                                        configuration.

  Direct file access is risky because     Supported     Unknown /                Medium Consultant evidence
  file formats can change between                       less                            cites Axys
  versions.                                             applicable                      file-format changes
                                                                                        between versions.

  APX SQL/database access may exist as an Not           Supported                Medium Native schema
  alternative export path.                applicable                                    Unknown.

  `li`/`lo` interpretation may depend on  Supported     Unknown                  Medium Morningstar Axys
  `.cli` setting.                                                                       conversion evidence.

  Reinvestments may appear as paired      Supported     Supported                Medium Axys conversion and
  transactions.                                                                         APX integration
                                                                                        evidence.

  Fees may depend on special security     Supported     Supported                Medium Terminology differs
  type/symbol and description                                                           across sources.
  translation.

  Principal paydowns may produce          Supported     Unknown                  Medium Axys conversion
  downstream conversion/reconciliation                                                  evidence.
  complications.

  Uppercase cancellation codes are        Supported     Supported                Medium Universality
  observed.                                                                             Unknown.

  AIA/APX import may remove pending       Not           Supported in             Medium AIA behavior, not
  records, sweeps, intra-account          applicable    integration                     confirmed native APX
  journals, or merge                                    workflow                        order.
  FX/accrued-interest/dividend-interest
  records.

  Initial deliver-ins may be generated    Unknown       Supported in             Medium Native behavior
  from positions for accounts with no                   integration                     Unknown.
  transactions in AIA/APX workflow.                     workflow

  Statement transactions and posted       Unknown       Supported                Medium Workflow evidence.
  transactions may be distinguished in
  APX workflows.

  `;` may represent                       Unknown       Observed                 Medium Treat only as
  journal/comment/other/split in APX                                                    integration
  integration table.                                                                    evidence.
  ----------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

## 11. Audit Rules

These rules are candidate transaction audit controls. They are not
confirmed native Axys/APX validation behavior unless explicitly noted.

### 11.1 Validation Rules

  ------------------------------------------------------------------------------
  Rule          Severity      Description        Required             Confidence
                                                 Inputs
  ------------- ------------- ------------------ ------------- -----------------
  TR-001        Critical      Transaction        Portfolio ID               High
  Missing                     references a
  Portfolio                   portfolio that
                              does not exist.

  TR-002        Critical      Security           Security                   High
  Missing                     transaction        Identifier,
  Security                    references an      Transaction
                              unknown security.  Code

  TR-003        High          Trade-based        Trade Date                 High
  Missing Trade               transaction lacks
  Date                        trade date.

  TR-004        High          Settlement date    Trade Date,                High
  Settlement                  precedes trade     Settlement
  Before Trade                date.              Date

  TR-005        High          Security           Quantity,                  High
  Missing                     transaction lacks  Transaction
  Quantity                    required quantity. Code

  TR-006        Medium        Price-required     Price,                     High
  Missing Price               transaction has no Transaction
                              execution price.   Code

  TR-007        Medium        Foreign-currency   Currency, FX             Medium
  Invalid FX                  transaction has    Rate
  Rate                        missing or invalid
                              FX rate.
  ------------------------------------------------------------------------------

### 11.2 Translation and Blotter Rules

  ---------------------------------------------------------------------------
  Rule             Severity         Description                    Confidence
  ---------------- ---------------- ------------------- ---------------------
  TR-008 Portfolio Critical         External                           Medium
  Translation                       portfolio/account
  Failure                           cannot be
                                    translated.

  TR-009 Security  Critical         External security                  Medium
  Translation                       cannot be
  Failure                           translated.

  TR-010           High             External                           Medium
  Unsupported                       transaction type
  Transaction Type                  has no mapping.

  TR-011 Trade     Medium           Transaction remains                Medium
  Blotter                           in exception state.
  Exception

  TR-012           High             Cancellation                       Medium
  Cancellation                      transaction does
  Mismatch                          not sufficiently
                                    match original
                                    transaction.

  TR-013           High             Cancellation                       Medium
  Cancellation                      blotters require
  Control                           review, backup, and
                                    operational
                                    controls.
  ---------------------------------------------------------------------------

### 11.3 Accounting Rules

  --------------------------------------------------------------------------
  Rule             Severity         Description                   Confidence
  ---------------- ---------------- ------------------ ---------------------
  TR-014 Holdings  Critical         Posted transaction                  High
  Not Updated                       not reflected in
                                    holdings.

  TR-015 Cash Not  Critical         Posted transaction                  High
  Updated                           not reflected in
                                    cash.

  TR-016 Cost      High             Cost basis                        Medium
  Basis                             inconsistent with
  Inconsistency                     transaction
                                    history.

  TR-017 Tax Lot   High             Tax lots                          Medium
  Inconsistency                     inconsistent with
                                    transaction
                                    history.

  TR-018 Dividend  Medium           Dividend received                 Medium
  Without Position                  without supporting
                                    position.

  TR-019 Coupon    Medium           Coupon payment                    Medium
  Inconsistent                      inconsistent with
  With Bond                         bond
                                    characteristics.

  TR-020 Return of Medium           Return of capital                 Medium
  Capital Without                   appears for
  Eligible                          security not
  Security                          expected to
                                    support it.

  TR-021 Split     High             Split detected                    Medium
  Without Quantity                  without expected
  Adjustment                        holding
                                    adjustment.

  TR-022 Split     Medium           Historical prices                 Medium
  Without Price                     inconsistent with
  Adjustment                        split.

  TR-023 Principal Medium           Principal paydown                 Medium
  Paydown                           inconsistent with
  Inconsistency                     expected
                                    reduction.
  --------------------------------------------------------------------------

### 11.4 Reconciliation and Historical Change Rules

  -------------------------------------------------------------------------
  Rule             Severity         Description                  Confidence
  ---------------- ---------------- ----------------- ---------------------
  TR-024 Custodian High             Custodian                          High
  Difference                        transactions
                                    differ from
                                    accounting
                                    records.

  TR-025 IMEX      Medium           IMEX export                      Medium
  Difference                        differs from
                                    expected
                                    accounting
                                    records.

  TR-026 REP       Medium           REP report                      Unknown
  Difference                        differs from
                                    accounting
                                    records.

  TR-027           High             Historical                         High
  Historical                        transaction
  Transaction                       edited.
  Modified

  TR-028           High             Historical                         High
  Historical                        transaction
  Transaction                       deleted.
  Deleted

  TR-029           High             Historical                         High
  Performance                       transaction
  Restatement                       change may
  Candidate                         require
                                    performance
                                    review.

  TR-030 Duplicate High             Potential                          High
  Transaction                       duplicate
                                    transaction.

  TR-031 Duplicate Medium           Duplicate                        Medium
  External                          external
  Identifier                        transaction
                                    identifier.

  TR-032 Stale     Medium           Pending                          Medium
  Pending                           transaction
  Transaction                       exceeds
                                    operational
                                    threshold.

  TR-033 Batch     Medium           Import batch                     Medium
  Partially                         incomplete.
  Processed

  TR-034 Stale     Medium           Import should                    Medium
  Account / Stale                   identify stale
  Price Detection                   accounts and
                                    stale prices
                                    before
                                    export/posting.

  TR-035 External  High             Candidate                         High
  Flow                              external-flow
  Misclassified                     transaction is
                                    classified using
                                    code alone without
                                    security,
                                    source/destination,
                                    sign, and firm
                                    mapping context.

  TR-036 Fee /     High             Fee, expense,                     Medium
  Expense as                        tax, cash-security
  Flow                              movement, or sweep
                                    is treated as a
                                    client contribution
                                    or withdrawal.

  TR-037 Orphan    High             Uppercase                         Medium
  Reversal                          reversal/cancel
                                    code cannot be
                                    linked to its
                                    original
                                    transaction.
  -------------------------------------------------------------------------

------------------------------------------------------------------------

## 12. Version Differences

  -----------------------------------------------------------------------------
  Topic         Axys          APX                    Confidence Notes
  ------------- ------------- --------------- ----------------- ---------------
  Axys v2.x     Consultant    Not applicable             Medium Needs official
  binary files  evidence says                                   confirmation.
  and IMEX      Axys v2.x
                introduced
                binary file
                formats and
                IMEX allowed
                CSV, tab, and
                fixed
                formats.

  Axys v3.7 to  Consultant    Not applicable             Medium Supports
  v3.8 file     evidence says                                   caution against
  conversion    upgrading                                       direct file
                from Axys                                       access.
                v3.7 to v3.8
                required file
                conversion
                and produced
                some files
                with
                different
                formats.

  APX v1.x to   Not           Consultant                 Medium Needs official
  v4.x IMEX     applicable    evidence says                     confirmation.
                              APX maintained
                              IMEX
                              functionality
                              but eliminated
                              fixed-format
                              file
                              generation.

  Native        Unknown       Unknown                   Unknown Not supplied.
  transaction
  code changes
  by version

  Native Trade  Unknown       Unknown                   Unknown Not supplied.
  Blotter
  behavior
  changes by
  version

  Native REP    Unknown       Unknown                   Unknown Not supplied.
  report
  changes by
  version
  -----------------------------------------------------------------------------

------------------------------------------------------------------------

## 13. References

The supplied research identifies the following source categories and
specific references. Confidence varies by source type.

  ------------------------------------------------------------------------------------------------
              ID Source          Type            System     Topics                      Confidence
  -------------- --------------- --------------- ---------- ---------------------- ---------------
         SRC-001 SS&C Advent     Vendor product  Axys       Portfolio accounting,         High for
                 Axys Product    page                       reporting,               capabilities;
                 Page                                       performance,                   Low for
                                                            reconciliation,              mechanics
                                                            transactions,
                                                            positions,
                                                            settlement/trade
                                                            information, tax-lot
                                                            or average-cost
                                                            accounting, trade-date
                                                            or settlement-date
                                                            accounting.

         SRC-002 AdventGuru ---  Consultant      Axys/APX   IMEX, Trade Blotter,            Medium
                 Getting Data In article                    import/export,
                 and Out of                                 Replang, reports,
                 Advent APX and                             direct-file-access
                 Axys                                       risks, APX
                                                            SQL/database options.

         SRC-003 WealthTechs AIA Third-party     APX        Trade/Statement/Tax             Medium
                 User Manual --- integration                Lot/Position/Account
                 APX Users       manual                     blotters, transaction
                                                            translation,
                                                            cancellation,
                                                            comments, broker
                                                            fields, processing
                                                            order.

         SRC-004 WealthTechs AIA Third-party     Axys       Transaction                     Medium
                 User Manual --- integration                cancellation, IMEX
                 Axys Users      manual                     workflow, blotters,
                                                            cancellation example.

         SRC-005 ByAllAccounts   Third-party     APX        Translation tables,             Medium
                 Custodial       integration                reversals, fees,
                 Integrator ---  manual                     imports,
                 APX User Guide                             sign-dependent
                                                            translation,
                                                            source/destination
                                                            fields, special
                                                            security fields.

         SRC-006 ByAllAccounts   Third-party     Axys       Trade Blotter                   Medium
                 Custodial       integration                workflow, IMEX import,
                 Integrator ---  manual                     `topost.trn`,
                 Axys User Guide                            `imex32.exe`, folder
                                                            labels, IMEX logs,
                                                            security/reference
                                                            files.

         SRC-007 Morningstar     Third-party     Axys       Reinvestment,                   Medium
                 Office Advent   migration guide            deliver-in/out
                 Axys Conversion                            interpretation,
                 Guide                                      `.cli`, cost basis,
                                                            fees, paydowns,
                                                            transaction prices,
                                                            historical prices,
                                                            conversion caveats.

         SRC-008 Advent          Vendor report   APX        Transaction Summary      Low to Medium
                 Portfolio       guide / public             Report existence.
                 Exchange        PDF reference
                 Reports Guide

         SRC-009 Wealth          Vendor/report   APX / SSRS Transaction Summary             Medium
                 Management      sample                     Report purpose and
                 Reports /                                  sample columns.
                 Advent report
                 sample

         SRC-010 AdventGuru ---  Consultant      Axys/APX   APX-exported CLI files          Medium
                 APX to Axys     article                    mapped into Axys
                 Conversion                                 `topost.trn`;
                                                            transaction mappings
                                                            and tax lots.
  ------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

## 14. Unknowns

### 14.1 Transaction Codes

  -----------------------------------------------------------------------------
  ID                    Unknown                                        Priority
  --------------------- -------------------------- ----------------------------
  TU-001                Complete official Axys                             High
                        transaction-code matrix.

  TU-002                Complete official APX                              High
                        transaction-code matrix.

  TU-003                Whether Axys and APX                               High
                        transaction codes are
                        identical, overlapping,
                        divergent,
                        version-specific, or
                        configuration-dependent.

  TU-004                Which observed codes are                           High
                        native versus
                        integration-layer
                        mappings.

  TU-005                Deprecated or                                    Medium
                        version-specific
                        transaction codes.
  -----------------------------------------------------------------------------

### 14.2 IMEX

  ------------------------------------------------------------------------
  ID                    Unknown                                   Priority
  --------------------- --------------------- ----------------------------
  TU-006                Official Axys IMEX                            High
                        transaction export
                        object names.

  TU-007                Official Axys IMEX                            High
                        transaction import
                        object names.

  TU-008                Official APX IMEX                             High
                        transaction
                        export/import object
                        names.

  TU-009                Complete IMEX                                 High
                        transaction field
                        list.

  TU-010                Official Trade                                High
                        Blotter import
                        layout.

  TU-011                IMEX log fields and                         Medium
                        validation messages.

  TU-012                Native IMEX object                          Medium
                        dependency sequence.
  ------------------------------------------------------------------------

### 14.3 REP and Reports

  ------------------------------------------------------------------------
  ID                    Unknown                                   Priority
  --------------------- --------------------- ----------------------------
  TU-013                Which REP reports                             High
                        expose transaction
                        information.

  TU-014                Official APX                                  High
                        Transaction Summary
                        Report parameters and
                        fields.

  TU-015                Whether REP                                 Medium
                        transaction values
                        are stored,
                        recalculated, or
                        mixed.

  TU-016                Axys transaction                              High
                        report names and
                        fields.

  TU-017                APX transaction                             Medium
                        reports beyond
                        Transaction Summary
                        Report.

  TU-018                How REP report values                       Medium
                        reconcile to IMEX
                        exports and posted
                        accounting records.
  ------------------------------------------------------------------------

### 14.4 Internal Data Model and Posting

  ------------------------------------------------------------------------------
  ID                    Unknown                                         Priority
  --------------------- --------------------------- ----------------------------
  TU-019                How transactions are                                High
                        physically stored in Axys.

  TU-020                How transactions are stored                         High
                        in APX.

  TU-021                Internal identifiers that                           High
                        uniquely identify
                        transactions.

  TU-022                Native posting status                             Medium
                        values.

  TU-023                Native Trade Blotter state                          High
                        transitions.

  TU-024                Native error states and                           Medium
                        rejection codes.

  TU-025                Native warning messages.                          Medium

  TU-026                Batch                                             Medium
                        rollback/restart/recovery
                        logic.

  TU-027                Native idempotency or                             Medium
                        duplicate-detection logic.
  ------------------------------------------------------------------------------

### 14.5 Historical Changes, Lots, Cost Basis, and Audit

  ------------------------------------------------------------------------
  ID                    Unknown                                   Priority
  --------------------- --------------------- ----------------------------
  TU-028                How reversals are                             High
                        represented
                        internally.

  TU-029                Whether uppercase                             High
                        transaction codes
                        universally mean
                        delete/reversal.

  TU-030                How historical edits                          High
                        are represented.

  TU-031                Whether deleted                               High
                        transactions are
                        retained for audit.

  TU-032                How corrections are                         Medium
                        distinguished from
                        reversals.

  TU-033                How transaction edits                       Medium
                        propagate into
                        holdings.

  TU-034                How transaction edits                       Medium
                        propagate into cash.

  TU-035                How transaction edits                         High
                        propagate into
                        performance.

  TU-036                Whether historical                            High
                        transactions can be
                        reconstructed
                        completely.

  TU-037                How tax lots are                              High
                        linked to
                        transactions.

  TU-038                How partial lot                             Medium
                        disposals are
                        represented.

  TU-039                How per-share cost                            High
                        basis is represented
                        in `.cli` exports.

  TU-040                How transfer lots                           Medium
                        preserve acquisition
                        date and basis.

  TU-041                How lot locations are                       Medium
                        stored and used
                        natively.
  ------------------------------------------------------------------------

### 14.6 Multi-Currency and Performance

  ------------------------------------------------------------------------
  ID                    Unknown                                   Priority
  --------------------- --------------------- ----------------------------
  TU-042                How FX rates are                            Medium
                        stored.

  TU-043                How cross-currency                          Medium
                        settlements are
                        represented.

  TU-044                How FX transactions                         Medium
                        are merged or paired
                        in native workflows.

  TU-045                How base-currency                           Medium
                        values are stored
                        versus calculated.

  TU-046                Which transaction                             High
                        types affect stored
                        performance.

  TU-047                Which transaction                             High
                        changes trigger
                        performance
                        restatement.

  TU-048                How performance                               High
                        restatements are
                        detected or audited.

  TU-049                Whether                                       High
                        edited/deleted
                        historical
                        transactions are
                        visible to
                        performance
                        recalculation
                        workflows.
  ------------------------------------------------------------------------

------------------------------------------------------------------------

## 15. Minimum Additional Material Needed to Promote Unknowns

To convert this chapter from observed/integration-level evidence into a
more authoritative native Axys/APX transaction reference, the following
supplied material would be needed:

  -----------------------------------------------------------------------
  Needed Material                     Would Resolve
  ----------------------------------- -----------------------------------
  Official Axys transaction-code      Axys native code matrix.
  manual or sanitized production code
  list.

  Official APX transaction-code       APX native code matrix.
  manual or sanitized production code
  list.

  Official Axys/APX IMEX manual with  IMEX object names and field
  transaction objects.                layouts.

  Sample Axys IMEX transaction        Axys transaction field names and
  export/import files.                formats.

  Sample APX IMEX transaction         APX transaction field names and
  export/import files.                formats.

  Official Trade Blotter layout       Native blotter fields, required
  documentation.                      fields, validation rules.

  Sample REP transaction reports and  REP fields, parameters, report
  report definitions.                 behavior.

  Sanitized APX database schema or    Native APX transaction storage
  query extracts.                     model.

  Sanitized Axys file/export          Native Axys transaction storage and
  documentation.                      file behavior.

  Audit/log examples for posted,      Native audit trail, state
  canceled, corrected, and rejected   transitions, and historical
  transactions.                       reconstruction.
  -----------------------------------------------------------------------

## 16. Deep IMEX Update

The 2026-06-30 IMEX deep research strengthens the implementation guidance for
transaction extracts without closing the official-schema unknowns.

| Topic | Chapter treatment | Confidence |
|---|---|---:|
| `topost.trn` | Axys Trade Blotter file in `$pathtrn`; generated CI transactions are appended and existing transactions are left unchanged. | Verified for CI |
| File creation | Axys Import/Export can create `topost.trn` if it does not exist in the configured user folder. | Verified for CI |
| Comment boundaries | CI may create beginning and ending comment transactions around generated imports. | Verified for CI |
| Candidate fields | Portfolio, transaction code/subtype, dates, symbol/type, quantity, price, amounts, commission, fees, accrued interest, withholding, source/destination context, cost fields, Perf/CW, Mark to Market, currency, FX, comments, and external IDs should be inspected in live Axys. | Discovery guidance |
| External-flow classification | Source/destination and special-security context can be necessary for `li`, `lo`, `dp`, and `wd`, but official IMEX availability of those fields remains Unknown. | Medium / Unknown |
| REP fallback | If IMEX cannot expose enough context to classify external flows, use REP/report/custom extraction as a candidate source of classification evidence. | Design guidance |

Transaction code alone should not be used as a universal external-flow rule.

## 17. `pa` / `sa` Accrued-Interest Promotion Boundary

The 2026-07-03 accrued-interest research strengthens the practical
interpretation of `pa` and `sa`:

| Code | Strongest supported meaning | Product boundary |
|---|---|---|
| `pa` | BUY accrued interest / purchase accrued interest. | Fixed-income trade-settlement adjunct; not an investor external flow. |
| `sa` | SELL accrued interest / sale accrued interest. | Fixed-income trade-settlement adjunct; not an investor external flow. |

The evidence is still integration-level rather than a complete official native
Axys/APX code manual. It supports a narrow packaged demonstration when paired
trade, cash, holdings/accrued, and performance rows all move together, but it
does not support classifying `pa` or `sa` from code alone. The same evidence
path supports configurable transaction translation, so site-specific YAML or a
future vendor preset must still require fixed-income context and auditable
source-data support.

Safe `pa`/`sa` treatment should verify:

- fixed-income security type or equivalent bond context;
- paired principal transaction, such as `by`/`pa` or `sl`/`sa`;
- trade date and settlement date alignment;
- amount sign and settlement economics;
- accrued-interest, income, cash, or source/destination markers when available;
- holdings/accrued-interest movement; and
- security-performance and portfolio-performance treatment.

The packaged Axys/APX demo promotes `pa`/`sa` only in that coherent paired
fixed-income scenario: accrued-interest transaction rows, cash movement,
holdings/accrued-interest rows, `secperf.csv`, `portperf.csv`, and
reviewer-facing report comments are derived together. That is a demo-quality
gate, not a claim that `pa`/`sa` are universal code-only Axys defaults.

## 18. Under-Evidenced Backlog Transaction Update

The 2026-07-07 backlog transaction research strengthens the practical
interpretation of `rc`, `pd`, `ss`, `cs`, and split/journal marker `;`. A later
2026-07-07 `rc`/`pd` research pass further confirms the public integration
mapping evidence for return of capital and bond principal paydown. This does
not convert the codes into safe code-only packaged-demo defaults, because the
strongest evidence is still integration and conversion documentation rather
than an official complete Axys/APX transaction-code manual or native
performance-report methodology.

| Code or marker | Strongest supported meaning | Product boundary |
|---|---|---|
| `rc` | Return of capital. ByAllAccounts Axys/APX default translation evidence maps return-of-capital activity to `rc`, with `$pty` and `$cash` source/destination context. | Policy-gated. Public evidence confirms the translation mapping and cash destination, but not native tax-lot, cost-basis, or performance-report treatment. Do not classify from code alone. |
| `pd` | Principal paydown / bond-security return-of-capital event. ByAllAccounts translation evidence maps bond-security return-of-capital activity to `pd`, again with portfolio-cash destination context; Morningstar Axys conversion evidence discusses principal paydown rows and zero share quantity. | Fixed-income principal event, not ordinary interest income or client external flow. The packaged demo includes only a narrow context-gated MBS/amortizing-security example with cash receipt, principal/holding evidence, and performance-report treatment. |
| `ss` | Short sale / sell short candidate. | Requires short security type, cash/margin/short-account context, and verified amount/quantity sign conventions. |
| `cs` | Cover short / buy-cover-short candidate. | Requires the same short-account context as `ss`; keep lowercase economic cover-short treatment separate from uppercase cancellation/delete patterns. |
| `;` | Journal, Other, or locally materialized split marker. Public integration evidence maps split/journal/other concepts to `;`, but newer split-file evidence indicates normal Axys split support is central `split.inf` factor data rather than ordinary account-level transactions. | Marker/comment/corporate-action evidence unless local mapping proves a specific economic role. Prefer split-factor extracts for packaged or audited split scenarios. |

### 18.1 `pd` Principal Paydown Boundary

`pd` is now represented in the packaged performance-comparison demo, but only as
a coherent amortizing-security principal-paydown scenario. Safe `pd` treatment should
verify:

- fixed-income security type or equivalent MBS/ABS/amortizing-security context;
- principal-paydown or return-of-principal context;
- cash movement, typically `$pty` / `$cash` portfolio-cash context in the public
  translation evidence;
- holding principal, factor, quantity, or market-value movement consistent with
  the event;
- amount and sign convention, trade/effective date, and local translation
  support;
- reported security-performance and portfolio-performance treatment; and
- reviewer wording that distinguishes principal return from coupon interest,
  accrued interest, amortization/accretion, and client external flow.

Morningstar conversion evidence warns that Advent Axys principal paydown
activity may involve zero share quantity and original-principal adjustment
mechanics. Therefore, quantity alone is not enough to audit a paydown; principal
or factor context is needed where available. Cost-basis, amortization, and tax
lot mechanics remain best-efforts demo-construction context, not requirements
for Modified Dietz calculation.

For Modified Dietz performance-comparison work, `pd` should be treated as a
security-level principal return unless site evidence proves otherwise. It
increases cash and reduces outstanding principal, but it should not be
classified as an investor contribution or withdrawal. At portfolio level, it is
best modeled as an internal movement from security principal exposure to cash
when the MBS/ABS/amortizing-security paydown context is coherent. At security
level, it is a principal return by the security. Public sources do not
establish a universal native Axys/APX field layout for principal amount,
factor, or amortized-cost movement.

Rows lacking fixed-income context, cash destination, or local mapping support
should remain review-only. Reversal/delete rows and custom site-translation rows
should be classified separately before any performance treatment is inferred.

### 18.2 `rc` Return-of-Capital Boundary

`rc` is supported as return of capital in public integration mapping evidence,
and the strengthened research specifically confirms a portfolio-cash
destination in the ByAllAccounts mapping. Its performance role is still
policy-sensitive. A site may reasonably configure return of capital as
performance income, security-level capital-return evidence,
corporate-action/cost-basis evidence, or review-only evidence depending on the
report methodology. Public evidence does not verify native Axys/APX cost-basis
algorithm or exact performance-report treatment.

For this reason, `rc` should remain a site-confirmed rule rather than a packaged
default. Any future packaged example should explicitly state the configured
methodology and should not imply universal Axys/APX treatment.

For Modified Dietz, `rc` should not be modeled as an external client flow when
it represents issuer return of capital. It is a security-level distribution or
capital-return event that may affect cash, return, and cost-basis context, while
tax-basis handling remains best-efforts outside the performance calculation.

### 18.3 `ss` / `cs` Short-Side Boundary

Public integration evidence supports `ss` as short sale and `cs` as cover short,
but a safe performance interpretation needs more than the code:

- security type or position context proving the holding is short;
- quantity sign convention for trade rows and holdings rows;
- amount and cash/proceeds sign convention;
- cash, restricted cash, margin, or short-proceeds account mapping;
- realized gain/loss or lot-closure treatment when a short is covered; and
- reported performance treatment for the short exposure.

Until those items are available in a coherent source-data scenario, `ss` and
`cs` are better suited to tested site-variant YAML and onboarding override
examples than to the default packaged Axys/APX demo.

### 18.4 Concrete Example Gap

The 2026-07-07 research found useful translation-table rows and row-like
comment/import examples. It did **not** find public sanitized examples of:

- account-level Axys/APX transaction report rows for `rc`, `pd`, `ss`, or `cs`;
- IMEX `.cli` rows for those target types;
- REP/Replang report-output rows for those target types;
- APX Public View or SQL result rows for those target types; or
- before/after holdings plus before/after performance rows for the same real
  account event.

Those examples remain the highest-value evidence needed before broad packaged
demo promotion.
