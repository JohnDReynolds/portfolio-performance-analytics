# Transaction Evidence Ledger

> Compact provenance for
> [`../reference/Chapter_05_Transactions.md`](../reference/Chapter_05_Transactions.md).
> This ledger records what the reviewed sources support, where they conflict,
> and what evidence is still missing. It is not an official Axys/APX
> transaction-code dictionary or an implementation contract.

## Ownership Boundary

- Reader explanations, examples, audit guidance, and canonical Unknowns belong
  in Chapter 05.
- Current ppar behavior and fixture coverage belong in
  [`../contracts/transaction_semantics_matrix.yaml`](../contracts/transaction_semantics_matrix.yaml).
- Human-readable contract guidance belongs in
  [`../contracts/transaction_semantics_matrix.md`](../contracts/transaction_semantics_matrix.md).
- Cross-topic public-web provenance belongs in
  [`Public_Web_Research_2026-07-17.md`](Public_Web_Research_2026-07-17.md).
- This file owns source identity, granular transaction claims, contradictions,
  confidence boundaries, and the evidence needed to resolve open questions.

The former consolidated research narrative and its dated addenda were reduced
to this ledger after their durable conclusions were incorporated into Chapter
05 and the transaction contract. Git history remains the recovery path for the
superseded research prose.

## Source Register

| ID | Source | Type and scope | Default confidence |
|---|---|---|---:|
| TRN-S01 | [SS&C Advent Axys product page](https://www.advent.com/solutions/axys/) | Vendor product capabilities; no native field or code definitions. | High for capabilities; Low for mechanics |
| TRN-S02 | [SS&C Advent Portfolio Exchange product page](https://www.advent.com/solutions/advent-portfolio-exchange/) | Vendor product capabilities; no native field or code definitions. | High for capabilities; Low for mechanics |
| TRN-S03 | [ByAllAccounts Custodial Integrator APX User Guide](https://www.byallaccounts.net/Manuals/Custodial_Integrator/apx/CI_User_Guide.pdf) | Third-party APX translation tables, reversals, fees, and import parameters. | Medium |
| TRN-S04 | [ByAllAccounts Custodial Integrator Axys User Guide](https://www.byallaccounts.net/Manuals/Custodial_Integrator/axys/CI_User_Guide.pdf) | Third-party Axys Trade Blotter, IMEX, fields, paths, and translation behavior. | Medium |
| TRN-S05 | [WealthTechs AIA manual for APX](https://www.wealthtechs.com/files/AIADocumentation/APX%20Guide%20-%20AIA%20User%20Manual%20For%20APX%20Users.pdf) | Third-party APX blotters, ordered translation, cancellation, and reconciliation workflows. | Medium |
| TRN-S06 | [WealthTechs AIA manual for Axys](https://www.wealthtechs.com/files/AIADocumentation/Axys%20Guide%20-%20AIA%20User%20Manual%20For%20Axys%20Users.pdf) | Third-party Axys import, cancellation, withholding, and original-face options. | Medium |
| TRN-S07 | Morningstar Office Advent Axys Conversion Guide | Third-party conversion evidence for `.cli`, reinvestment, `li`/`lo`, fees, cost, and principal paydowns. No durable public URL was captured. | Medium |
| TRN-S08 | [Advent/SSRS Wealth Management Reports](https://cdn.advent.com/cms/pdfs/reports/REP_SSRS.pdf) | Vendor report sample showing Transaction Summary sections and visible columns. | Medium |
| TRN-S09 | [AdventGuru: Getting Data In and Out](https://adventguru.com/2013/04/25/getting-data-in-and-out-of-advent-apx-and-axys/) | Consultant evidence for IMEX history, reports, APX alternatives, and direct-file risk. | Medium |
| TRN-S10 | [AdventGuru: APX to Axys Conversion](https://adventguru.com/2019/05/22/there-and-back-again/) | Consultant conversion evidence for APX-exported `.cli`, `topost.trn`, Trade Blotter posting, and tax-lot complexity. | Medium |
| TRN-S11 | [AdventGuru: Fixing the Audit Trail Export](https://adventguru.com/2019/09/30/fixing-the-audit-trail-export/) | Consultant evidence for `didpost.aud` and an Axys 3.x large-export issue. | Medium |
| TRN-S12 | [Public Web Research Ledger](Public_Web_Research_2026-07-17.md) | Dated provenance for ordered APX transformation/cancellation, Axys missing-cost review, and APX Mark-to-Market requiredness. | Per claim |

The transaction-specific sources are predominantly integration, conversion, or
consultant material. They can establish observed behavior, but not universal
native behavior across Axys/APX versions, sites, and import paths.

## General and Lifecycle Claims

| Claim | Evidence | Confidence | Boundary or chapter impact |
|---|---|---:|---|
| TRN-C001 | Transactions feed holdings, cash, tax lots, cost basis, realized gain/loss, income, performance, reports, reconciliation, and audit workflows. | High conceptually | Product pages support broad capabilities; native mechanics remain version- and site-dependent. |
| TRN-C002 | Reviewed integration workflows normalize and translate source activity before staging it in a Trade Blotter for review and posting. | Medium; TRN-S03-S06 | An observed integration lifecycle, not proof of the only native lifecycle. |
| TRN-C003 | APX AIA workflows expose Trade, Statement, Tax Lot, Position, Account, and Initial Transaction blotter concepts. | Medium; TRN-S05 | AIA behavior; do not generalize every blotter or setting to native APX. |
| TRN-C004 | APX translation can be customized by financial institution and may depend on signs and ordered rules. | Medium; TRN-S03, TRN-S05, TRN-S12 | Code-only interpretation is unsafe. |
| TRN-C005 | Axys integration evidence routes generated transactions to `topost.trn` in `$pathtrn`, where the integration appends rather than replaces existing rows. | Medium; TRN-S04 | Specific to the documented integration workflow. |
| TRN-C006 | The Axys workflow references `imex32.exe`, IMEX logs, `$pathcli`, `$pathinf`, `$pathpri`, and `$pathlog`. | Medium; TRN-S04 | Exact native version coverage, layouts, and messages are unverified. |
| TRN-C007 | The integration can create beginning/end comment transactions and include source comments around its generated block. | Medium; TRN-S04 | Comments may be non-posting unless configured; `;` remains context-dependent. |
| TRN-C008 | `didpost.aud` is described as an Axys/APX Audit Trail artifact associated with posted transactions. | Medium; TRN-S10-S11 | Layout, completeness, retention, correction visibility, and version equivalence are Unknown. |
| TRN-C009 | Large Axys 3.x Audit Trail exports through IMEX reportedly failed in some environments, motivating risky direct-file workarounds. | Medium; TRN-S11 | Historical consultant observation, not current vendor guidance. |
| TRN-C010 | APX can offer SQL/database or reporting alternatives in addition to IMEX; old direct access to Axys binary files is version-sensitive. | Medium; TRN-S09 | Interface options do not establish canonical storage or supported current access. |

## Observed Transaction Codes and Tokens

These are source-supported observations, not a complete official matrix. The
contract may apply narrower behavior because it also reflects ppar tests,
fixtures, and safety policy.

| Claim | Code or token | Observed meaning | Evidence and confidence | Required boundary |
|---|---|---|---|---|
| TRN-C100 | `by` | Buy; also a reinvestment or non-cash-deposit buy leg. | Medium; TRN-S03-S04, TRN-S07 | Internal trade, not an external flow; inspect pair context. |
| TRN-C101 | `sl` | Sell or long-security sale. | Medium; TRN-S03-S04 | Internal trade, not an external flow. |
| TRN-C102 | `ss` | Short sale; default mapping includes `SELL / SHORT -> ss`. | Medium-High for code; TRN-S03 | Cash, quantity, collateral, and holdings mechanics remain site-specific. |
| TRN-C103 | `cs` | Cover short; default mapping includes `BUY / COVER SHORT -> cs`. | Medium-High for code; TRN-S03 | Requires prior/resulting short exposure and local cash mechanics. |
| TRN-C104 | `dv` | Dividend or income distribution; may be the income leg of reinvestment. | Medium; TRN-S03-S04, TRN-S07 | Performance income, not client capital; link paired legs. |
| TRN-C105 | `in` | Income or positive interest, including cash-security income in observed mappings. | Medium; TRN-S03-S04 | Require security and amount context. |
| TRN-C106 | `ai` | Negative interest or margin interest. | Medium; TRN-S03-S04 | Not generic accrued interest; require financing/margin and sign context. |
| TRN-C107 | `pa` | Purchase or buy-side accrued interest. | Medium; TRN-S03-S04 | Fixed-income settlement adjunct; not an external flow; not safe from code alone. |
| TRN-C108 | `sa` | Sale or sell-side accrued interest. | Medium; TRN-S03-S04 | Fixed-income settlement adjunct; not an external flow; not safe from code alone. |
| TRN-C109 | `rc` | Return of capital with portfolio-cash context in observed mappings. | Medium-High for mapping; TRN-S03-S04 | Issuer/corporate-action event, not client capital; performance and basis policy remain unverified. |
| TRN-C110 | `pd` | Principal paydown or bond-security return-of-capital event. | Medium-High for mapping; TRN-S03-S04, TRN-S07 | Principal event, not coupon or client flow; quantity may be zero. |
| TRN-C111 | `li` | Deliver in, transfer in, deposit, credit, or positive movement. | Medium; TRN-S03-S04, TRN-S07 | External-flow candidate only when outside-party context proves it. |
| TRN-C112 | `lo` | Deliver out, transfer out, withdrawal, debit, or negative movement. | Medium; TRN-S03-S04, TRN-S07 | External-flow candidate only when outside-party context proves it. |
| TRN-C113 | `dp` | Fee, tax, expense, service charge, cash-security buy, or debit-like posting. | Medium; TRN-S03-S04, TRN-S07 | Meaning depends on special-security and source/destination context. |
| TRN-C114 | `wd` | Cash-security sell or withdrawal-like cash-security movement. | Medium; TRN-S03 | Do not infer a client withdrawal from the mnemonic alone. |
| TRN-C115 | `;` | Journal, Other, Split, comment, or control marker in observed integrations. | Low-Medium; TRN-S03-S04 | Require local mapping or corporate-action/report evidence; may not be economic. |
| TRN-C116 | `epus` | Fee/expense token; associated with management fee in conversion evidence and expense-like use in integration evidence. | Medium but conflicting; TRN-S03, TRN-S07 | Role may be security type, symbol, label, or code; no standalone code rule. |
| TRN-C117 | `exus` | Expense/fee token; associated with expense in conversion evidence and fee special-security use in integration evidence. | Medium but conflicting; TRN-S03, TRN-S07 | Role may be security type, symbol, label, or code; no standalone code rule. |
| TRN-C118 | `dvwash` | Dividend-reinvestment wash symbol. | Medium; TRN-S03 | Pairing context only; not an external flow or ordinary holding. |
| TRN-C119 | `caus margin` | Margin cash/security context used with margin interest. | Medium; TRN-S03 | Supports contextual `ai` classification only. |

## Translation and Context Claims

| Claim | Source-supported observation | Confidence | Safety boundary |
|---|---|---:|---|
| TRN-C140 | Positive-unit transfers map to `li`; negative-unit transfers map to `lo` in documented Axys/APX integration defaults. | Medium; TRN-S03-S04 | Sign-dependent integration mapping, not a universal native rule. |
| TRN-C141 | ATM, point-of-sale, direct-deposit/direct-debit, credit/debit, payment, withdrawal, and transfer concepts can map into `li`/`lo` according to direction. | Medium; TRN-S03 | Preserve source concept and signs; code alone cannot prove external flow. |
| TRN-C142 | Morningstar conversion evidence says a `.cli` transaction setting at character 53 can make `li`/`lo` mean Deliver In/Out (`Y`) or Credit/Debit of Security (`N`). | Medium; TRN-S07 | Conversion-specific and strong evidence that code-only treatment is unsafe. |
| TRN-C143 | A non-cash deposit may produce `li` plus `by`; reinvestment may produce `dv` plus `by`, often with `dvwash`. | Medium; TRN-S03, TRN-S07 | Match date, security, amount, income, cash, and wash context to avoid double counting. |
| TRN-C144 | Cash-security buy may map to `dp`; cash-security sell may map to `wd`. | Medium; TRN-S03 | These codes are not reliable external-flow mnemonics. |
| TRN-C145 | Fees may use `dp` with special type/symbol combinations such as `exus custfee` or `epus expense`. | Medium; TRN-S03 | Preserve local fee parameters and token roles. |
| TRN-C146 | Fee translation parameters include `defFeeType`, `defFeeSymbol`, and numbered description/type/symbol overrides. | Medium; TRN-S03 | Parameter names prove configurability, not universal fee semantics. |
| TRN-C147 | Margin interest maps to `ai` with margin context; positive interest maps to `in`. | Medium; TRN-S03 | Require sign and security/cash context. |
| TRN-C148 | Return of capital maps to `rc`; bond-security return of capital maps to `pd`, with `$pty`/`$cash` context in the documented table. | Medium-High for translation; TRN-S03-S04 | Native basis, principal, and performance algorithms remain Unknown. |
| TRN-C149 | `SELL / SHORT` maps to `ss` with `awus / none`; `BUY / COVER SHORT` maps to `cs` with `$pty / $cash`. | Medium-High for translation; TRN-S03 | These fields do not prove universal short-proceeds or collateral mechanics. |
| TRN-C150 | Axys AIA offers multiple withholding treatments, including expense treatment or a withholding field deducted from trade amount. | Medium; TRN-S06 | Integration options; audit gross/net dividend and separate tax lines. |
| TRN-C151 | Axys AIA has original-face quantity and quantity-rounding options. | Medium; TRN-S06 | Integration-specific; retain original face/principal evidence for fixed income. |
| TRN-C152 | AIA can create initial deliver-ins from positions for accounts with no transactions and may ignore same-day transactions in that configured scenario. | Medium; TRN-S05 | AIA workflow only; not native APX behavior. |

## Reversal, Matching, and Special-Family Claims

| Claim | Evidence synthesis | Confidence | Safe conclusion |
|---|---|---:|---|
| TRN-C160 | Reviewed Axys/APX integration tools can create cancellation Trade Blotter instructions by uppercasing historical transaction codes, such as `by -> BY`; examples also reference `SL`, `SS`, and `CS`. | Medium-High; TRN-S03-S06, TRN-S12 | Treat this as staging/control evidence. Posted-export availability and native universality are Unknown. |
| TRN-C161 | An APX cancellation must sufficiently match original transaction fields or the Trade Blotter can reject it. | Medium; TRN-S03, TRN-S05 | Prefer source transaction ID; otherwise use strict, unique matching and preserve unmatched rows. |
| TRN-C162 | Cancellation workflows may derive records from historical transaction files and carry backup/review warnings. | Medium; TRN-S05-S06 | Do not infer native deletion retention or idempotency. |
| TRN-C163 | `pa`/`sa` are usable only with fixed-income context, a paired principal trade, aligned trade/settlement dates, accrued-interest amount, and coherent settlement economics. | Medium for source meaning; High as safety boundary | They are not investor contributions or withdrawals. Native holdings/performance rows remain unverified. |
| TRN-C164 | `pd` can increase cash while reducing outstanding principal, and conversion evidence reports zero-share paydown rows and original-principal adjustments. | Medium; TRN-S07 | Do not require quantity movement; seek bond/MBS type, factor/principal, cash, and local report treatment. |
| TRN-C165 | `rc` represents issuer return of capital rather than a client contribution, but public evidence does not establish native basis reduction or performance-report treatment. | Medium-High for mapping; Low for mechanics | Keep performance classification policy- or site-gated. |
| TRN-C166 | `ss` and `cs` are internal security trades, not external capital flows. | High conceptually; Medium-High for code meaning | Production use requires short exposure, signs, and short-cash/margin/source-destination evidence. |
| TRN-C167 | A synthetic short lifecycle may assume negative position and market value, separate proceeds/collateral, and gain/loss on cover. | General accounting inference, not native evidence | Suitable only when explicitly disclosed; never proof of universal Axys/APX mechanics. |
| TRN-C168 | A split can appear as `;` in integration mappings, while normal Axys split evidence may instead reside in central split-factor data such as `split.inf`. | Low-Medium; integration evidence plus Chapter 09 provenance | Prefer split-factor evidence over requiring account-level transaction rows. |
| TRN-C169 | The APX AIA guide distinguishes selected case-sensitive APX identifiers from its case-insensitive Transaction Translation evaluator. | High for cited workflow; TRN-S05, TRN-S12 claim WEB-20260718-001 | Preserve native transaction and context-field case; reproduce case-insensitive evaluation only under an explicit workflow/site contract. |

## Fields, Reports, and Interface Evidence

| Claim | Observed evidence | Confidence | Boundary |
|---|---|---:|---|
| TRN-C180 | Translation guides expose transaction type, source/destination type and symbol, special-security type and symbol, broker, lot location, and comments. | Medium; TRN-S03-S06 | Observed integration fields, not official IMEX object definitions. |
| TRN-C181 | Candidate Axys Trade Blotter settings/columns include transaction type, source/destination fields, special-security fields, commission, broker, lot, Perf/CW, and Mark to Market. | Medium; TRN-S04 | The full `topost.trn` layout and native meanings are Unknown. |
| TRN-C182 | The visible Transaction Summary sample has Purchases, Sales, Dividends, Contributions, and Withdrawals sections. | Medium; TRN-S08 | A report presentation, not a native transaction schema. |
| TRN-C183 | Visible Sales columns include Trade Date, Settle Date, Quantity, Symbol, Security, Unit Cost, Total Cost, Unit Price, Proceeds, and Gain/Loss. | Medium; TRN-S08 | Report fields may omit classification context required for audit. |
| TRN-C184 | Public Axys missing-cost guidance identifies portfolio, code, security type/symbol/name, trade/original-cost dates, quantity, trade amount, original cost, source/destination, and lot fields. | Per WEB-20260717-012 in TRN-S12 | Report guidance, not a complete native transaction layout. |
| TRN-C185 | That guidance warns standard reports can substitute trade-date market value when original cost is missing. | Per WEB-20260717-012 in TRN-S12 | Independently test original-cost date/amount completeness. |
| TRN-C186 | APX Mark-to-Market requiredness can be version-specific in documented import workflows. | Per WEB-20260717-013 in TRN-S12 | Do not generalize one version's field requirement or calculation. |
| TRN-C187 | Per-share cost basis is converted only when available in exported `.cli` data in the reviewed conversion path. | Medium; TRN-S07 | Conversion behavior; native storage and lot mechanics remain Unknown. |
| TRN-C188 | IMEX and REP/report extraction are complementary: IMEX is an interface, while reports may provide classification or reconciliation context absent from an extract. | Medium; TRN-S08-S09 | Neither is established as the canonical native transaction store. |

## Contradictions and Interpretation Risks

| ID | Tension | Resolution |
|---|---|---|
| TRN-X001 | Sources use `epus` and `exus` inconsistently for management fees, expenses, taxes, labels, or special securities. | Preserve token role and local mapping; do not make either a universal standalone code. |
| TRN-X002 | `li`/`lo` look like external-flow mnemonics but also represent security transfers, credits/debits, and internal movement. | Require source/destination, security type, signs, and local settings. |
| TRN-X003 | `dp`/`wd` look cash-directional but can be fee, tax, cash-security trade, sweep, transfer, or external activity. | Require special-security and source/destination context. |
| TRN-X004 | Uppercase cancellation is documented for Trade Blotter staging/control, but not as a universal native convention or an ordinary posted-export representation. | Preserve case. Require source-stage evidence and original linkage; an uppercase posted row remains unknown from case alone. |
| TRN-X005 | `;` can be a journal/other/split transaction marker or a comment/control boundary. | Require the file, field, and local mapping context. |
| TRN-X006 | Short code meaning is stronger than evidence for quantity, market value, proceeds, collateral, and performance signs. | Separate code recognition from accounting classification. |
| TRN-X007 | Translation tables prove observed defaults while simultaneously documenting site customization. | Treat defaults as provenance and require overrides/site evidence in production. |
| TRN-X008 | IMEX, Trade Blotter, audit files, reports, and APX database access expose different lifecycle views. | Do not equate an interface artifact with canonical storage or complete audit history. |
| TRN-X009 | Product pages establish broad capabilities but do not define transaction mechanics. | Use them only for capability claims. |

## Evidence Required to Resolve Canonical Unknowns

Chapter 05 owns the full Unknowns table. The items below state the missing
evidence rather than duplicating that table.

| Need | Evidence that would resolve or materially narrow it |
|---|---|
| TRN-U001 Official code coverage | Versioned SS&C Axys and APX transaction-code dictionaries, including reserved codes, case behavior, and configurable overrides. |
| TRN-U002 Native transaction layouts | Versioned Axys Trade Blotter/IMEX layouts and APX import, Public View, or database schemas with requiredness and data types. |
| TRN-U003 IMEX definitions | Official transaction object names, field lists, dependencies, ordering, validation, logs, and error semantics. |
| TRN-U004 REP/report definitions | Official report names, parameters, sections, field definitions, and reconciliations to transaction extracts. |
| TRN-U005 Posting lifecycle | Vendor documentation or sanitized production evidence for staging, posting states, stable IDs, errors, retries, duplicate detection, and idempotency. |
| TRN-U006 Audit and correction history | Sanitized `didpost.aud`/audit exports showing originals, corrections, reversals, deletions, rejected rows, timestamps, and actor/source identifiers. |
| TRN-U007 Holdings/cash propagation | Paired pre/post transaction, holdings, and cash extracts for representative codes and corrections. |
| TRN-U008 Cost and lots | Paired transaction, lot, cost-basis, realized-gain, and tax outputs for buys, sells, transfers, return of capital, and paydowns. |
| TRN-U009 Performance treatment | Paired transaction and native portfolio/security performance reports proving external-flow, income, expense, principal, and correction treatment. |
| TRN-U010 Fixed-income families | Native `pa`, `sa`, and `pd` rows with accrued-interest, coupon/day-count, settlement, original face, factor/principal, holdings, cash, and performance evidence. |
| TRN-U011 Return of capital | Native `rc` rows with cash, cost basis, lots, holdings, and performance-report treatment. |
| TRN-U012 Short lifecycle | Native `ss`/`cs` transaction, position, proceeds/collateral, margin, realized-gain, and performance rows from one complete lifecycle. |
| TRN-U013 Multi-currency | Transaction examples with trade/settlement/portfolio currencies, FX rates, local/base amounts, fees, income, and native performance output. |
| TRN-U014 Version differences | The same representative import/report cases across supported Axys/APX versions and site configurations. |

Highest-value next acquisition: one sanitized, internally consistent account
package containing raw import rows, posted transaction output, audit trail,
holdings, cash, lots/cost, and portfolio/security performance before and after
the same events. Without customer-portal or installed-system access, public
research can refine provenance but is unlikely to close the native-layout and
behavior gaps.

## Maintenance Rule

Add a new claim only when a source contributes new provenance, narrows a
boundary, or exposes a contradiction. Update Chapter 05 when reader guidance or
an Unknown changes. Update the YAML contract only when ppar behavior, fixture
coverage, or validation policy changes. Do not append another narrative
research pass to this file.
