# Multi-Currency Evidence Ledger

> Compact provenance for
> [`../reference/Chapter_17_Multi_Currency.md`](../reference/Chapter_17_Multi_Currency.md).
> This ledger records supported capability and connector claims, native-mechanic
> boundaries, contradictions, and missing evidence. It is not an Axys/APX FX
> schema or currency-performance methodology.

## Ownership Boundary

- Reader explanations, normalized adapter guidance, and canonical Unknowns
  belong in Chapter 17.
- Purchase cash-bucket inference has a deliberately narrow evidence ledger in
  [`Research_17A_Multi_Currency_Cash_Provenance.md`](Research_17A_Multi_Currency_Cash_Provenance.md).
- Security, transaction, cash, pricing, and performance claims remain owned by
  their subject ledgers; this file records their multi-currency intersection.
- Cross-topic public-web provenance belongs in
  [`Public_Web_Research_2026-07-17.md`](Public_Web_Research_2026-07-17.md).

This file replaces the AI-assisted intake `_temp_deep-research-report-2.md`,
merged on 2026-07-13. Its internal bracketed references pointed back to older
repository chapters rather than independent sources and have been replaced by
durable local links and claim IDs. Git history remains the recovery path for
the superseded narrative and implementation handoff.

## Source Register

| ID | Source | Type and scope | Default confidence |
|---|---|---|---:|
| MC-S01 | [Official Axys product page](https://www.advent.com/solutions/axys/) | Current vendor capability claims for report currency, withholding, return components, and security-currency concepts. | High for capabilities; Low for mechanics |
| MC-S02 | [Official APX product page](https://www.advent.com/solutions/advent-portfolio-exchange/) | Vendor centralized-book, multicurrency/multi-asset, reporting, and performance capabilities. | High for capabilities; Low for mechanics |
| MC-S03 | [ByAllAccounts Axys CI guide](https://www.byallaccounts.net/Manuals/Custodial_Integrator/axys/CI_User_Guide.pdf) | Third-party `axyscur`, Mark-to-Market, `Perf/CW`, source/destination, and import context. | Medium-High for workflow |
| MC-S04 | [ByAllAccounts APX CI release notes](https://www.byallaccounts.net/Manuals/Custodial_Integrator/apx/CI_releasenotes.pdf) | Versioned APX v1-v4 Mark-to-Market requiredness. | High for cited integration releases |
| MC-S05 | [Security Master Evidence Ledger](Research_04_Security_Master.md) | Currency-field and native security-layout boundaries. | Per claim |
| MC-S06 | [Transaction Evidence Ledger](Research_05_Transactions.md) | Transaction/settlement currency, FX-link, and source/destination boundaries. | Per claim |
| MC-S07 | [Cash Evidence Ledger](Research_07_Cash.md) | System currency, cash tokens, multicurrency cash, and Mark-to-Market/`Perf/CW` boundaries. | Per claim |
| MC-S08 | [Pricing Evidence Ledger](Research_08_Pricing.md) | `$pathpri`, `*.pri`, logs, sources, FX-file modes, and native FX-schema gaps. | Per claim |
| MC-S09 | [Performance Evidence Ledger](Research_10_Performance.md) | Market-versus-currency capability and stored/recalculated performance gaps. | Per claim |
| MC-S10 | [Public Web Research Ledger](Public_Web_Research_2026-07-17.md) | `WEB-20260717-013` and `014`: APX requiredness and current Axys capabilities. | Per claim |

## Product-Capability Claims

| Claim | Evidence | Confidence | Boundary or chapter impact |
|---|---|---:|---|
| MC-C001 | Current Axys material supports report-currency restatement, withholding-tax handling, market-price versus currency-rate return components, and trading/income/risk currency concepts. | High for capability; MC-S01, MC-S10 | Does not establish native fields, formulas, or site configuration. |
| MC-C002 | Reviewed Axys material also describes money-market/other cash types, trade- or settlement-date accounting, and data-interface use of domestic/international prices and spot/forward exchange rates. | High for capability; MC-S01 | Does not establish FX record layouts, keys, quote direction, or import objects. |
| MC-C003 | APX public material supports centralized accounting/reporting/performance, multicurrency and multi-asset coverage, and settlement/cash capabilities. | High for capability; MC-S02 | Does not establish cash-ledger, FX, or performance-decomposition schemas. |
| MC-C004 | Historical Axys 3.8.7 release evidence identifies additional or improved multicurrency reports. | Medium-High for cited release | Dated report capability; names, fields, and current support are unverified. |

## Currency-Configuration Claims

| Claim | Source-supported observation | Confidence | Safety boundary |
|---|---|---:|---|
| MC-C020 | `axyscur` is a configured Axys system-currency value in the documented ByAllAccounts connector. | Medium-High; MC-S03, MC-S07 | Connector setting; native global/database/portfolio scope is Unknown. |
| MC-C021 | Official Axys language distinguishes trading, income, and risk currency concepts for securities. | High for capability; MC-S01 | Native field names, storage, export layout, and interaction are Unknown. |
| MC-C022 | A generic security currency is a reasonable discovery field, but public evidence does not prove a native currency triad in `sec.inf` or another export. | Medium as discovery guidance; MC-S05 | Do not collapse or invent fields without site evidence. |
| MC-C023 | Transaction currency and FX rate are expected audit concepts; public evidence does not establish native Axys/APX field names or layouts. | Medium concept; Unknown schema; MC-S06 | Require explicit client mapping. |
| MC-C024 | Public evidence does not prove a distinct native settlement-currency field or its relationship to transaction currency. | Unknown; MC-S06 | Preserve separately only when supplied. |
| MC-C025 | Public evidence does not establish whether report currency is stored separately from portfolio or system currency. | Unknown | Require report headers, parameters, or site mapping. |

## FX Artifact and Rate Claims

| Claim | Evidence synthesis | Confidence | Safe conclusion |
|---|---|---:|---|
| MC-C040 | `$pathpri`, `*.pri`, and `imexPrices.log` are observed Axys-oriented pricing artifacts. | Medium-High for connector; MC-S08 | Recognize as pricing artifacts, not a proven FX schema. |
| MC-C041 | Official capability material says interfaces can populate spot and forward rates. | High for capability; MC-S01 | Separate objects/layouts and operational semantics remain Unknown. |
| MC-C042 | No reviewed source proves that `.pri` universally stores FX, how currency pairs are keyed, or whether rates are modeled as securities, pairs, or another object. | Unknown; MC-S08 | A universal `.pri` FX loader is unsupported. |
| MC-C043 | `mergepri` is consultant-documented price-file merging with first-source precedence. | Medium; MC-S08 | Apply to FX only after site confirmation of file content and supported version. |
| MC-C044 | APX AIA exposes FX-file Update, Add, and Replace modes. | Medium-High for workflow; MC-S08 | Native FX-file schema, key, and replace scope remain Unknown. |
| MC-C045 | Public evidence does not establish pair identifier, effective-date field, spot/forward subtype, bid/ask/close/average distinction, direct/inverse quote convention, or reciprocal generation. | Unknown | Require normalized client-supplied fields and document policy. |
| MC-C046 | Price missing/stale/calculated evidence cannot be projected automatically onto native FX-rate handling. | High caution; MC-S08 | Treat FX exception policy as explicit audit configuration. |

## Transaction, Cash, and Interface Claims

| Claim | Source-supported observation | Confidence | Safety boundary |
|---|---|---:|---|
| MC-C060 | Source/destination type and symbol appear in partner transaction mappings alongside `$cash`, `$income`, `CASH`, `MMF`, `MARGIN`, and `SHORT`. | Medium-High for workflows; MC-S06-S07 | Useful context; not universal native cash-account fields. |
| MC-C061 | ByAllAccounts Axys CI exposes Mark to Market through `defmarkmarket` and `Perf/CW` through `defperfcw` in `topost.trn` context. | Medium-High for workflow; MC-S03, MC-S07 | Field existence/requiredness is supported; accounting meaning is Unknown. |
| MC-C062 | APX CI v1-v3 required Mark to Market only for foreign-currency transactions, while v4 requires explicit `y` or `n`. | High for cited releases; MC-S04, MC-S10 | Versioned interface requiredness, not a calculation definition or Axys rule. |
| MC-C063 | Public evidence does not establish native cross-currency trade, FX conversion, settlement, and cash-leg linkage. | Unknown; MC-S06-S07 | Do not infer a conversion chain from transaction code or security currency alone. |
| MC-C064 | Public evidence does not establish native cash-balance presentation by currency or local/base cash fields. | Unknown; MC-S07 | Require explicit cash currency and extraction source. |

## Valuation and Performance Claims

| Claim | Evidence synthesis | Confidence | Boundary |
|---|---|---:|---|
| MC-C080 | Axys publicly claims return components attributable to market prices and currency-rate fluctuations. | High for capability; MC-S01, MC-S09 | Consume report-supplied components; do not reproduce an undocumented formula. |
| MC-C081 | No reviewed source establishes local-to-base/report valuation formula or which FX date applies to holdings, trades, settlements, income, fees, gains, or flows. | Unknown | Require report/source evidence or explicit audit policy. |
| MC-C082 | No reviewed source establishes whether cash and securities use the same conversion rule. | Unknown | Keep asset/cash conversion evidence separate. |
| MC-C083 | Arithmetic/geometric decomposition, interaction allocation, flow treatment, and definitions of local/currency/base return are not publicly established. | Unknown; MC-S09 | Conventional formulas must not be labeled as Axys formulas. |
| MC-C084 | Stored-versus-report-time calculation and historical FX correction/restatement behavior are not established. | Unknown; MC-S09 | Require controlled before/after reruns and stored/exported outputs. |
| MC-C085 | Report-supplied local/base values or currency components may be ingested as evidence when report context and currency are explicit. | High as conservative audit boundary | Preserve source values; compare rather than reconstruct. |

## Contradictions and Interpretation Risks

| ID | Tension | Resolution |
|---|---|---|
| MC-X001 | Official sources describe rich currency capabilities while public schemas remain absent. | Separate capability confidence from field/mechanic confidence. |
| MC-X002 | `axyscur` sounds like a native base currency but is evidenced only as connector configuration. | Keep connector scope; require native/site proof before mapping portfolio base currency. |
| MC-X003 | Spot/forward import is a supported capability, but `.pri` is only proven as a pricing artifact. | Do not equate capability with file format. |
| MC-X004 | Mark to Market and `Perf/CW` exist in import context but their economic semantics are not documented. | Preserve raw values and requiredness; avoid formula interpretation. |
| MC-X005 | Trading/income/risk currencies are official concepts but not verified exported fields. | Accept explicit client mappings without assigning native names. |
| MC-X006 | Market/currency return bifurcation is official capability, but formula and interaction treatment are absent. | Compare native report output only; label independent calculations separately. |
| MC-X007 | Price-source and missing/stale controls may resemble FX controls. | Do not inherit pricing rules for FX without site evidence. |

## Evidence Required to Resolve Canonical Unknowns

Chapter 17 owns the full Unknowns and minimum adapter contract. This section
records the missing evidence needed to resolve its native-mechanic boundaries.

| Need | Evidence that would resolve or materially narrow it |
|---|---|
| MC-U001 Currency fields and scope | Official portfolio/security/transaction dictionaries plus sanitized exports showing system, portfolio, report, trading, income, risk, transaction, and settlement currencies. |
| MC-U002 FX layouts and quotation | Native import/export/API samples with pair, rate date, rate, type, source, direct/inverse convention, reciprocal behavior, and spot/forward distinction. |
| MC-U003 Valuation rules | Worked vendor/site examples for holdings, cash, transactions, income, fees, gains, and flows with local/base/report values and rate dates. |
| MC-U004 Performance decomposition | Report definitions or worked examples reconciling local, currency, interaction, and base return components. |
| MC-U005 Mark to Market and `Perf/CW` | Original field documentation plus paired imports and accounting/report outputs proving meaning. |
| MC-U006 Cross-currency lifecycle | One trade with transaction, FX, settlement, cash, holdings, and performance evidence from the same account. |
| MC-U007 Historical restatement | Controlled before/after rerun changing only a historical FX rate, with stored and reported outputs. |
| MC-U008 Version/site differences | The same representative cases across supported Axys/APX versions, connectors, and site configurations. |

Highest-value next acquisition: a sanitized multi-currency account package with
security/portfolio currency fields, dated FX rows, transactions, cash and
holdings by currency, comparable report-currency runs, and performance
components for one period.

## Maintenance Rule

Add a claim only for new provenance, a narrowed boundary, or a contradiction.
Update Chapter 17 when reader guidance or an Unknown changes. Put detailed
purchase cash-provenance claims in Research 17A, and do not append another
narrative research or implementation-handoff section here.
