# Chapter 17 — Multi-Currency

**Repository:** AXYS / APX Reference Repository  
**Chapter:** `docs/axys-apx-reference/reference/Chapter_17_Multi_Currency.md`  
**Prepared:** 2026-07-13  
**Evidence:** `../evidence/Research_17_Multi_Currency.md` and
`../evidence/Research_17A_Multi_Currency_Cash_Provenance.md`

---

## Related chapters

- [Chapter_04_Security_Master.md](Chapter_04_Security_Master.md) — candidate
  security-currency fields and the missing native field dictionary.
- [Chapter_05_Transactions.md](Chapter_05_Transactions.md) — transaction and
  source/destination context.
- [Chapter_07_Cash.md](Chapter_07_Cash.md) — cash-like securities, cash legs,
  sweeps, and system-currency connector evidence.
- [Chapter_08_Pricing.md](Chapter_08_Pricing.md) — `.pri` files, price imports,
  and the unresolved FX-rate schema.
- [Chapter_10_Performance.md](Chapter_10_Performance.md) — report and
  performance-methodology boundaries.

## 1. Evidence conclusion

The supplied evidence supports a narrow, export-and-report-based Axys
multi-currency audit adapter. It does not support reconstruction of Axys's
native currency valuation or performance engine.

The July 2026 research intake deliberately searched for native currency fields,
FX file layouts, valuation rules, and return-bifurcation formulas. It recovered
no durable public source establishing those mechanics. The strongest findings
remain third-party connector artifacts already represented elsewhere in this
reference. The negative result is useful: unsupported mechanics must remain
`Unknown`, while explicit client-supplied currency and FX evidence can be
normalized and audited without claiming Axys formula parity.

| Evidence level | Supported conclusion |
|---|---|
| Verified product capability | Axys product material supports multi-currency reporting and market-price versus currency-rate return components at a capability level. |
| High Confidence connector evidence | ByAllAccounts-oriented Axys workflows expose `axyscur`, `defmarkmarket`, `defperfcw`, `topost.trn`, and source/destination context. |
| Partially supported pricing workflow | `$pathpri`, `*.pri`, and `imexPrices.log` are observed Axys-oriented connector artifacts. |
| Unknown native mechanics | Native currency fields, FX layouts, quote conventions, rate-date rules, conversion formulas, interaction treatment, and historical-restatement behavior are not established. |

Connector evidence describes the observed connector workflow. It is not proof
that every Axys site exposes the same artifacts or that a connector parameter
maps one-to-one to a native Axys field.

## 2. Currency configuration boundary

| Question | Current conclusion | Confidence |
|---|---|---|
| Does an Axys-oriented connector have a configured system currency? | Yes; `axyscur` is observed in ByAllAccounts-oriented evidence. | High Confidence for that connector context |
| Is `axyscur` a native portfolio base-currency field? | Unknown. Its native scope and storage are not established. | Unknown |
| Is report currency stored separately from system or portfolio currency? | Unknown. | Unknown |
| Are trading, income, and risk currencies stored separately for each security? | Axys capability material describes those concepts, but the native field names and export layout are not established. | Verified capability; Unknown schema |
| Are transaction and settlement currency separate native fields? | Unknown. | Unknown |

An implementation may accept explicit client-mapped values for these concepts.
It must not assign Axys-native field names or scope without site evidence.

## 3. FX-rate and pricing boundary

The evidence establishes price-workflow artifacts, not an Axys FX schema.

| Artifact or behavior | Supported treatment |
|---|---|
| `$pathpri`, `*.pri`, `imexPrices.log` | Recognize as Axys-oriented connector pricing artifacts. |
| `.pri` as a universal FX-rate file | Do not assume. FX content, keys, and layout are Unknown. |
| `mergepri` | Treat as consultant-documented price-file merge behavior; use for FX only after site confirmation. |
| Spot versus forward layouts | Unknown. |
| Currency-pair identifier | Unknown. |
| Effective-date field and rate-date selection | Unknown. |
| Direct versus inverse quote convention | Unknown. |
| Reciprocal-rate generation | Unknown. |
| Missing, stale, duplicate, and corrected FX handling | Unknown as native Axys behavior. |

The safe adapter boundary is a normalized, client-supplied rate record with:

- from currency,
- to currency,
- effective date,
- rate,
- optional rate type, and
- optional source provenance.

These are audit-contract concepts, not claimed Axys field names.

## 4. Transactions and cash provenance

Source/destination type and symbol fields are the strongest observed
transaction-level cash context. Cash-like tokens such as `$cash`, `$income`,
`CASH`, `MMF`, `MARGIN`, and `SHORT` occur in partner integration material, but
they are not a universal native naming standard.

For a security purchase, cash provenance should use four states:

| State | Use when |
|---|---|
| Proven | An exported, site-validated source/destination cash leg identifies the cash security or bucket. |
| Strongly inferred | Exactly one currency-consistent cash holding has the reconciling movement. |
| Weakly inferred | The currency exposure is plausible, but multiple cash buckets or sweep activity prevent a unique match. |
| Unknown | The available transaction, holdings, and FX evidence does not prove the source. |

Do not hard-code a cash-security name such as `CASHEUR` or `CASH_EUR`. Require a
site mapping or classify cash-like holdings from client-confirmed security
master, type, description, holdings, and currency evidence.

Cross-currency purchases require separate FX evidence. A transaction code or
security trading currency alone does not prove the conversion path, cash
security, bank account, or settlement linkage.

## 5. Valuation and performance boundary

The current evidence does not establish:

- which FX date Axys uses for a valuation, transaction, income item, fee,
  realized gain, or external flow;
- whether cash and securities use the same conversion rule;
- the accounting meaning or calculation of Mark to Market in the observed
  connector import;
- the semantic meaning of `Perf/CW`;
- the exact market-price versus currency-rate return formula;
- whether the decomposition is arithmetic or geometric;
- how an interaction term is allocated;
- how flows enter the decomposition; or
- whether corrected historical FX rates alter stored values or only rerun
  reports.

If a site exports local value, base value, local return, currency effect, or
base return, an audit adapter may ingest those values as report evidence. It
must not label a separately calculated conventional formula as the Axys method.

## 6. Minimum evidence contract

A source package capable of supporting the first audit phase should provide:

| Audit need | Minimum source evidence | Fallback |
|---|---|---|
| Holdings exposure | Portfolio, as-of date, security, market value, and explicit or mapped currency. | Mark the exposure currency unavailable. |
| Cash exposure | Client-confirmed cash row or extract plus currency. | Do not assign a specific cash security. |
| FX rates | Normalized pair, effective date, and rate; source/type when available. | Report missing FX evidence. |
| Transactions | Portfolio, relevant dates, security, amount, and explicit or mapped transaction currency. | Limit review to holdings/report evidence. |
| Cash-leg context | Source/destination type and symbol when exported. | Preserve provenance as inferred or Unknown. |
| Currency performance | Report-supplied local, currency-effect, and base values with report context. | Do not reconstruct Axys bifurcation. |
| Report restatement | Comparable report runs with explicit report currency and period. | Treat restatement mechanics as unavailable. |

## 7. Safe first-phase audit behavior

The evidence supports the following conservative behavior:

- validate currency completeness on exposed holdings, cash, transactions, and
  report rows;
- validate dated FX coverage for currency exposure;
- identify missing, changed, stale, duplicate, and reciprocal-inconsistent
  normalized rates under explicit audit policy;
- flag cross-currency transactions that lack an evidenced FX link;
- compare report-supplied local, base, and currency-effect values without
  reconstructing the Axys formula;
- classify cash provenance with explicit confidence; and
- preserve source values, mappings, assumptions, and Unknowns in findings.

These checks describe an audit product's behavior. They do not assert native
Axys accounting behavior.

## 8. Explicitly blocked behavior

Do not implement the following as Axys-native behavior without new evidence:

- automatic discovery of portfolio base currency from `axyscur`;
- separate native trading, income, and risk currency fields;
- a universal `.pri` FX loader;
- an assumed FX quote direction or reciprocal rule;
- an assumed rate-date or missing-rate policy;
- automatic cross-currency trade/FX/cash linking from transaction code alone;
- Axys-style market/currency return bifurcation; or
- historical FX restatement semantics.

## 9. Evidence needed to advance

The most valuable next evidence is:

1. An official Axys security or portfolio field dictionary.
2. A sanitized native FX import/export file with its layout.
3. A multi-currency holdings and transaction extract from one Axys site.
4. The same Axys report run in two reporting currencies.
5. A report showing local, currency, interaction, and base return components.
6. A controlled before/after rerun in which only an historical FX rate changes.
7. The original connector manual pages defining `axyscur`, Mark to Market, and
   `Perf/CW`.

Until that evidence is available, the normalized evidence-only adapter is the
strongest defensible implementation boundary.

## 10. Evidence provenance

- [Research_17_Multi_Currency.md](../evidence/Research_17_Multi_Currency.md)
  preserves the focused follow-up and implementation handoff.
- [Research_17A_Multi_Currency_Cash_Provenance.md](../evidence/Research_17A_Multi_Currency_Cash_Provenance.md)
  preserves the earlier cash-provenance synthesis.

Both source reports were AI-assisted research outputs. Their internal ChatGPT
citation tokens are not durable citations. Their durable contribution is the
conservative synthesis and explicit inventory of Unknowns, not new primary
vendor evidence.
