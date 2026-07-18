# Cash Evidence Ledger

> Compact provenance for
> [`../reference/Chapter_07_Cash.md`](../reference/Chapter_07_Cash.md).
> This ledger records source-supported cash claims, transaction-context
> boundaries, contradictions, and missing evidence. It is not a native Axys/APX
> cash schema or transaction-code dictionary.

## Ownership Boundary

- Reader explanations, examples, cash field guidance, and canonical Unknowns
  belong in Chapter 07.
- Code meanings and matching policy belong in
  [`Research_05_Transactions.md`](Research_05_Transactions.md) and the
  transaction contract; this ledger records only cash-specific implications.
- Holdings and pricing provenance belong in their matching ledgers.
- Cross-topic public-web provenance belongs in
  [`Public_Web_Research_2026-07-17.md`](Public_Web_Research_2026-07-17.md).

The former narrative research file was reduced after its durable conclusions
were incorporated into Chapter 07. Git history remains the recovery path for
superseded prose, examples, field drafts, and chapter outlines.

## Source Register

| ID | Source | Type and scope | Default confidence |
|---|---|---|---:|
| CSH-S01 | [SS&C Advent Axys product page](https://www.advent.com/solutions/axys/) | Vendor product and current currency/withholding capabilities; no cash schema. | High for capabilities; Low for mechanics |
| CSH-S02 | SS&C Advent APX product material | Vendor holdings/transactions/performance and multicurrency capabilities. | High for capabilities; Low for mechanics |
| CSH-S03 | [ByAllAccounts Custodial Integrator Axys guide](https://www.byallaccounts.net/Manuals/Custodial_Integrator/axys/CI_User_Guide.pdf) | Third-party transaction translation, cash tokens, currency, Mark-to-Market, and `Perf/CW`. | Medium-High for workflow |
| CSH-S04 | [WealthTechs AIA manual for Axys](https://www.wealthtechs.com/files/AIADocumentation/Axys%20Guide%20-%20AIA%20User%20Manual%20For%20Axys%20Users.pdf) | Third-party cash sweeps, journal cleanup, symbols, and examples. | Medium-High for workflow |
| CSH-S05 | [WealthTechs AIA manual for APX](https://www.wealthtechs.com/files/AIADocumentation/APX%20Guide%20-%20AIA%20User%20Manual%20For%20APX%20Users.pdf) | Third-party APX sweep, journal, `ACCTX`, margin/short, and skip-logic examples. | Medium-High for workflow |
| CSH-S06 | Morningstar Axys conversion guide | Conversion/reconciliation and adjustment-transaction cautions. | Medium |
| CSH-S07 | SS&C Advent Custodial Data product brief | Vendor capability that custodian flows include positions, transactions, and cash activity. | High for capability |
| CSH-S08 | [CSSI Cash Holdings guidance](https://cssisolutions.com/downloads/creating-an-equity-assets-by-type-report-and-a-cash-hold) | Dated Axys Report Writer example deriving Cash Holdings through classification. | Medium-High for example |
| CSH-S09 | [Public Web Research Ledger](Public_Web_Research_2026-07-17.md) | `WEB-20260717-014` and `020`: Axys currency capabilities and classification-derived Cash Holdings. | Per claim |

## Cash Representation Claims

| Claim | Evidence | Confidence | Boundary or chapter impact |
|---|---|---:|---|
| CSH-C001 | Custodian workflows can deliver positions, transactions, and cash activity into Advent-oriented processing. | High for capability; CSH-S07 | Does not prove a standalone native cash-balance object. |
| CSH-C002 | Reviewed Axys/APX integrations represent much cash activity through transaction codes plus source/destination type and symbol fields. | Medium-High; CSH-S03-S05 | Code-only interpretation is unsafe; transaction semantics remain owned by Chapter 05. |
| CSH-C003 | The documented mappings repeatedly use `$cash` for cash and `$income` for income-related transaction context. | Medium-High; CSH-S03 | Observed integration tokens, not universal native definitions. |
| CSH-C004 | Axys CI exposes a configured system currency (`axyscur`) and a cash asset-class setting (`axysaccash`) whose default is `c` but may differ in older Axys versions. | Medium-High; CSH-S03 | Do not hard-code currency or cash asset-class values. |
| CSH-C005 | In the documented Axys CI context, non-system-currency transactions can require Mark to Market and `topost.trn` has a `Perf/CW` column. | Medium-High; CSH-S03 | Exact field meaning, layout, and APX equivalence remain Unknown. |
| CSH-C006 | Dated Axys Report Writer guidance derives a Cash Holdings report through asset-class/security-type classification logic. | Medium-High; CSH-S08-S09 | The example's classification code and formula are not universal native cash rules. |
| CSH-C007 | Current product evidence supports Axys report-currency and currency-aware capabilities. | High for capability; CSH-S01, CSH-S09 | Native local/base cash fields and FX valuation mechanics remain Unknown. |

## Observed Cash-Like Tokens

| Claim | Token | Observed role | Confidence and boundary |
|---|---|---|---|
| CSH-C020 | `$cash` | Cash source/destination symbol. | Medium-High; CSH-S03; integration context only. |
| CSH-C021 | `$income` | Income-related source/destination symbol. | Medium-High; CSH-S03; integration context only. |
| CSH-C022 | `$pty` | Source/destination type used in many portfolio-cash translations. | Medium-High as token; exact expansion Unknown. |
| CSH-C023 | `$ity` | Source/destination type used in income/accrued-interest translations. | Medium-High as token; exact expansion Unknown. |
| CSH-C024 | `$pth` | Source/destination type used in margin-interest context. | Medium-High as token; exact expansion Unknown. |
| CSH-C025 | `CAUS`, `CASH`, `MMF` | Cash/security type and cash/money-market symbols in AIA examples. | Medium-High as observed values; native expansions/roles unverified. |
| CSH-C026 | `MARGIN`, `SHORT` | Separate margin- and short-sweep contexts in AIA examples. | Medium-High for workflow; do not treat as unrestricted cash. |
| CSH-C027 | `dvwash`, `dvshrt`, `dvlong`, `cashrt`, `calong`, `income` | Special symbols excluded from generic sweep/journal cleanup. | Medium-High for workflow; native definitions mostly Unknown. |
| CSH-C028 | `ACCTX` | Prefix/object marker in WealthTechs transaction-row examples. | Medium-High as observed marker; native APX/Axys status Unknown. |
| CSH-C029 | `caxx`, `awxx` | Cash/security-type leads recorded by later integration research. | Low-Medium; exact definitions and native status Unknown. |

## Sweep and Journal Claims

| Claim | Source-supported observation | Confidence | Safety boundary |
|---|---|---:|---|
| CSH-C040 | AIA defines a cash sweep as movement between cash accounts, commonly cash and a money-market fund. | Medium-High; CSH-S04-S05 | Integration definition, not native system behavior. |
| CSH-C041 | AIA can remove `DP`/`WD` sweep rows when cash type/symbol criteria match, with separate options for margin and short sweeps. | Medium-High; CSH-S04-S05 | Removal changes imported representation; native Axys/APX does not thereby net sweeps. |
| CSH-C042 | AIA defines intra-account cash journals as opposite transactions that wipe each other out. | Medium-High; CSH-S04-S05 | Third-party cleanup concept. |
| CSH-C043 | Documented opposite-code pairs include `dp`/`wd`, `li`/`lo`, `ti`/`to`, `si`/`so`, and `tr`/`ts`. | Medium-High; CSH-S04-S05 | Only `dp`/`wd` and `li`/`lo` have broader evidence in the transaction contract; other pairs remain backlog. |
| CSH-C044 | Journal cleanup matches trade date, account, amount, quantity, code pair, cash/security types, and symbol exclusions. | Medium-High; CSH-S04-S05 | Preserve unmatched rows; do not use fuzzy pairing. |
| CSH-C045 | APX AIA Account Skip Logic can bypass global cleanup for selected accounts. | Medium-High; CSH-S05 | Site/account configuration can change observed cash activity. |
| CSH-C046 | No reviewed source proves that native Axys/APX automatically removes sweeps or nets intra-account journals. | Unknown native behavior | Keep integration preprocessing distinct from accounting/report treatment. |

## Transaction-Family Cash Implications

| Claim | Evidence synthesis | Confidence | Safe conclusion |
|---|---|---:|---|
| CSH-C060 | `li`/`lo` can increase/decrease cash or move securities, but external-flow treatment requires outside-party and source/destination context. | Medium-High; CSH-S03 and transaction ledger | Do not infer contribution/withdrawal from code alone. |
| CSH-C061 | `dp`/`wd` can represent fees, taxes, cash-security activity, sweeps, transfers, or withdrawal-like movement. | Medium-High | Require special-security, cash/security, source/destination, and pair context. |
| CSH-C062 | `by`/`sl`, `dv`/`in`/`ai`, `rc`/`pd`, and `ss`/`cs` can all affect cash while remaining investment/performance events rather than external capital flows. | High conceptually; mapping confidence varies | Preserve the economic family before assigning cash-flow treatment. |
| CSH-C063 | Public mappings place `pd` in `$pty`/`$cash` context when principal-paydown evidence exists. | Medium-High | Cash receipt requires fixed-income/principal evidence; not an investor contribution. |
| CSH-C064 | Public mapping uses `awus / none` for `ss` but `$pty / $cash` for `cs`. | Medium-High for mapping | Short proceeds may be restricted, collateral, margin, or site-specific; ordinary cash treatment is unsafe. |
| CSH-C065 | Reviewed integration tools can create uppercase cancellation Trade Blotter instructions from historical transaction codes. | Medium-High for staging/control | Require explicit source stage and keep the instruction out of posted cash economics; posted-export availability and universality remain Unknown. |

## Interface and Report Claims

| Claim | Observed evidence | Confidence | Boundary |
|---|---|---:|---|
| CSH-C080 | Axys integration flow can combine source-data with security information and create transaction, position, and price artifacts imported through `imex32.exe`. | Medium-High; CSH-S03 | Cash may be reconstructed across artifacts; this does not prove no cash object exists. |
| CSH-C081 | Related observed artifacts include `topost.trn`, `ptopost.trn`, `.pos`, `$pathcli`, `$pathinf`, `$pathpri`, `$pathlog`, `pospos32.exe`, and a `sipos30` report lead. | Medium-High for integration | Artifact roles and version coverage vary; Chapter 12 owns interface detail. |
| CSH-C082 | Reviewed sources do not identify native cash-balance IMEX object names, fields, or distinct beginning/ending/settled/trade-date objects. | Unknown | Do not invent `cash`, `cashbal`, or equivalent objects. |
| CSH-C083 | Reviewed sources do not establish standard Axys/APX cash report names or whether reports present cash as a security row, subtotal, account balance, or currency balance. | Unknown | Cash Holdings is a dated derived-report example, not a universal standard report. |
| CSH-C084 | APX Income Projection is a cash-inflow report lead, not evidence of a cash-balance report. | Low-Medium | Keep activity/projection separate from balance evidence. |

## Contradictions and Interpretation Risks

| ID | Tension | Resolution |
|---|---|---|
| CSH-X001 | Cash activity is clearly present in integration feeds, but a standalone native cash-balance object is not documented. | Separate transaction/position evidence from balance-object claims. |
| CSH-X002 | Sweep vehicles can be retained as holdings or removed as cash-cleanup activity. | Preserve preprocessing configuration and reconcile cash plus money-market exposure together. |
| CSH-X003 | Margin, short, income, and dividend-wash symbols are cash-like but not equivalent to unrestricted cash. | Keep their buckets/context distinct until site evidence proves aggregation. |
| CSH-X004 | A report can derive Cash Holdings through classification, while storage and report formulas remain unknown. | Treat classification-derived reports as presentation evidence only. |
| CSH-X005 | Mark to Market and `Perf/CW` are observed Axys import columns, but native meanings and APX equivalents are unverified. | Retain native values and require local documentation before interpreting them. |
| CSH-X006 | AIA removes matched pairs, but native reports may include or classify them differently. | Record preprocessing lineage and never infer native netting from cleaned input. |

## Evidence Required to Resolve Canonical Unknowns

Chapter 07 owns the complete Unknowns table. This section records the missing
evidence needed to resolve it.

| Need | Evidence that would resolve or materially narrow it |
|---|---|
| CSH-U001 Cash objects and storage | Axys/APX IMEX definitions, database/public-view documentation, or production extracts for cash balances. |
| CSH-U002 Balance derivation | Controlled transactions, holdings, cash, and report outputs proving stored versus calculated balances. |
| CSH-U003 Date basis | Paired trade-date and settlement-date examples with unsettled activity and report settings. |
| CSH-U004 Report behavior | Versioned REP/RDL definitions and outputs for cash balance, activity, reconciliation, and projection reports. |
| CSH-U005 Cash-like classification | Security-master and holdings examples for cash, money markets, margin, short proceeds, income cash, and sweep vehicles. |
| CSH-U006 Token meanings | Vendor or site translation documentation for `CAUS`, `$pty`, `$ity`, `$pth`, `caxx`, `awxx`, `cashrt`, and `calong`. |
| CSH-U007 Multicurrency | Local/base cash, currency, FX rate/source/date, Mark-to-Market, and report-currency examples. |
| CSH-U008 Performance treatment | Native performance reports paired with deposits, withdrawals, fees, income, sweeps, principal returns, shorts, and reversals. |
| CSH-U009 Version/configuration differences | Versioned manuals, site mappings, asset-class codes, and account skip/cleanup settings. |

Highest-value next acquisition: one sanitized portfolio package with beginning
and ending cash by currency, all intervening transactions, holdings including
cash-like securities, settlement status, source/destination fields, and the
native cash/performance reports for the same period.

## Maintenance Rule

Add a claim only for new provenance, a narrowed boundary, or a contradiction.
Update Chapter 07 when reader guidance or an Unknown changes. Do not duplicate
the Chapter 05 code matrix or append another narrative research pass.
