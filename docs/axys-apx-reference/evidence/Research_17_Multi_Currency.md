# Axys Multi-Currency Audit Support Follow-Up

**Evidence file:** `Research_17_Multi_Currency.md`  
**Original intake file:** `_temp_deep-research-report-2.md`  
**Merged:** 2026-07-13  
**Target chapter:** `../reference/Chapter_17_Multi_Currency.md`  
**Status:** AI-assisted research synthesis; not vendor documentation

> **Citation boundary:** Bracketed references such as `[B1]` point to the
> uploaded repository files listed in this report. They do not add independent
> primary evidence. The report recovered no durable public URL supporting an
> Axys multi-currency mechanic. Its one public URL concerns brand context only
> and is not evidence for Axys behavior. Treat the implementation handoff as a
> conservative synthesis of the existing evidence archive.

## Executive answer

A **narrow, defensible Axys multi-currency audit model is implementable now**, but only if it is framed as an **export-and-report evidence model**, not as a reconstruction of Axys’s native accounting or performance engine.

What can be implemented now with defensible boundaries is this:

- Accept **explicit client-supplied currency fields** on holdings, cash, transactions, and performance extracts when available.
- Accept **client-supplied dated FX rates** in a normalized rate file, even if that file is produced outside native Axys, because the public evidence does **not** establish a complete Axys-native FX record layout.
- Use Axys-oriented integration artifacts such as **`axyscur`**, **`defmarkmarket`**, and **`defperfcw`** only as **connector-context evidence**, not as proven native Axys field semantics. [B1]
- Treat **multi-currency report restatements** and any **local/base/currency-effect outputs** as **report evidence to ingest and compare**, not as formulas to reproduce, because the exact Axys bifurcation formula is **not publicly established**. [B3]
- Use **evidence-only FX checks**: missing rates, changed rates, stale-rate exposure, non-system-currency transaction flags, and restated-report differences linked to exposure. [B1] [B2] [B3]

What **cannot** be implemented defensibly today from public evidence is this:

- A proven native Axys rule for **portfolio base currency** versus **system currency** versus **report currency**.
- Native field names or file layouts for separate **trading currency**, **income currency**, and **risk currency**.
- A proven Axys-native FX storage model for **spot versus forward rates**, **pair identification**, **rate date selection**, **direct versus inverse quotation**, or **reciprocal generation**.
- The exact Axys method for converting local values into base/reporting currency for valuation or performance.
- The exact Axys **market-versus-currency return bifurcation formula**, including any interaction-term treatment.
- Native Axys mechanics for **cross-currency trade plus FX plus cash-leg linkage**.
- Whether historical FX corrections **recalculate stored history** or only affect rerun reports. [B1] [B2] [B3] [B4] [B5]

The practical conclusion is straightforward: **Phase 1 should be a normalized, evidence-only Axys audit adapter** that requires explicit exported currency and FX fields when they matter. Anything more ambitious is blocked pending client data, report samples, or official documentation.

## Currency and FX evidence

The strongest evidence available in this session came from the uploaded Axys reference chapters. Those chapters preserve concrete field names and artifact names in a few places, but they also document major gaps. In several cases, the best answer is not “yes” or “no,” but **“not publicly established.”**

### Axys currency-field and artifact dictionary

| Item | What can be said defensibly | Best-known Axys artifact or field | Implementation status | Evidence note |
|---|---|---|---|---|
| Portfolio base currency | **Not publicly established.** No uploaded source proves a native portfolio-level base-currency field or how it is represented in an export. | Unknown | **Blocked pending client data or documentation** | Uploaded file `Chapter_07_Cash(3).md`, lines 96-99 says only that integration tooling references a configured system currency and that native multi-currency cash presentation is not established. Underlying source named as ByAllAccounts CI guide. Tier 2. Confidence: High for the connector parameter; Unknown for native Axys portfolio storage. [B1] |
| System currency | In Axys-oriented integration evidence, a configured **system currency** exists. | `axyscur` | **Safe to implement as configurable inference** in connector-specific workflows only | Uploaded file `Chapter_07_Cash(3).md`, lines 67 and 96: “Axys system currency can be configured in integration tooling” and “Axys integration tooling references a configured system currency.” Underlying source named as ByAllAccounts `Custodial Integrator User Guide for Axys`. Tier 2. Confidence: High. Native Axys scope remains unproven. [B1] |
| `axyscur` | Best current reading: a **connector configuration value** representing the Axys system currency in the ByAllAccounts CI workflow. Public evidence does **not** prove whether it is global, database-level, or portfolio-level in native Axys. | `axyscur` | **Safe to implement as configurable inference** only | Same evidence as above. The uploaded chapter does not preserve the direct public URL for the ByAllAccounts guide, so the safest formulation is connector-scoped only. Tier 2. Confidence: High. [B1] |
| Report currency distinct from portfolio currency | **Not publicly established** from a native Axys report parameter dictionary in the available evidence set. | Unknown | **Blocked pending client report samples** | No uploaded chapter provides a parameter list or report header proving the term or field. Performance chapter explicitly says report-specific return methods and multi-currency fields remain unknown. Tier 1/2 evidence gap. Confidence: Unknown. [B3] |
| Security trading currency | **Not publicly established.** | Unknown | **Blocked pending client data or documentation** | Uploaded `Chapter_04_Security_Master(10).md` does not provide a field layout for `sec.inf`; the only currency-related item is a generic “currency” candidate live-discovery field, not a separate trading-currency field. Tier 2 evidence gap. Confidence: Unknown. [B6] |
| Security income currency | **Not publicly established.** | Unknown | **Blocked pending client data or documentation** | Same. No public `sec.inf` field list in the uploaded evidence proves a distinct income-currency field. [B6] |
| Security risk currency | **Not publicly established.** | Unknown | **Blocked pending client data or documentation** | Same. No public field list proves a distinct risk-currency field. [B6] |
| Security “currency” singular | A generic security-level **currency** concept is plausible, but only as a **candidate live-discovery field**, not as a validated Axys field name or separate trading/income/risk triad. | “currency” candidate field | **Evidence-only audit check** until actual export is seen | Uploaded `Chapter_04_Security_Master(10).md`, line 817 lists “currency” only as a candidate discovery field alongside multiplier, factor, coupon, maturity, and classifications. Tier 2/3 synthesis. Confidence: Medium. [B6] |
| Transaction currency | Public evidence treats this as an expected concept, but the field name and layout are **unknown**. | Expected concept only | **Blocked pending client data or documentation** | Uploaded `Chapter_05_Transactions(9).md`, lines 941-947: “Currency” and “FX Rate” are expected in multi-currency contexts, but IMEX/REP/native names are unknown. Tier 2 synthesis with explicit unknown. Confidence: Unknown/Medium. [B4] |
| Settlement currency | Public evidence does **not** prove separate Axys representation from transaction currency. | Unknown | **Blocked pending client data or documentation** | Same evidence. The chapter uses “transaction or settlement currency” conceptually, but does not prove distinct Axys columns. [B4] |

### FX-rate artifact and record-layout table

| Artifact or question | What is actually supported | Best-known file/interface | Implementation status | Evidence note |
|---|---|---|---|---|
| Axys price/rate import folder | Axys-oriented CI evidence shows a **price folder**. | `$pathpri` | **Safe to implement as Axys connector behavior** | Uploaded `Chapter_08_Pricing(3).md`, lines 74-77. Tier 2. Confidence: Verified for CI workflow. [B2] |
| Axys price files | Axys-oriented CI evidence shows **`*.pri` price files**. | `*.pri` | **Safe to implement as Axys connector behavior** | Uploaded `Chapter_08_Pricing(3).md`, line 75. Tier 2. Confidence: Verified for CI workflow. [B2] |
| IMEX price logs | Axys-oriented CI evidence shows **`imexPrices.log`**. | `imexPrices.log` | **Safe to implement as Axys connector behavior** | Uploaded `Chapter_08_Pricing(3).md`, line 76. Tier 2. Confidence: Verified for CI workflow. [B2] |
| `.pri` in an FX context | **Not publicly established.** Public evidence shows `.pri` as a price-file artifact, but does **not** prove that FX rates are stored there, how they are keyed, or whether currencies are represented as securities, pairs, or another record type. | `*.pri` | **Blocked pending client files or docs** | Uploaded `Chapter_08_Pricing(3).md`, lines 131-140 explicitly say complete `.pri` layout, native price key, and separate price/rate conventions are unknown. Tier 2 synthesis. Confidence: Unknown. [B2] |
| Spot vs forward rates | Axys capability to use exchange rates is acknowledged in the broader research set, but the public evidence available here does **not** identify separate spot and forward record layouts or file names. | Unknown | **Blocked pending client data or documentation** | No uploaded chapter exposes such a layout. Pricing chapter says “exact price/FX schema Unknown.” Tier 1/2 evidence gap. Confidence: Unknown. [B2] |
| Currency pair identifier | **Not publicly established.** | Unknown | **Blocked pending client data or documentation** | No field dictionary or sample FX row found. [B2] [B4] |
| Effective date of a rate | **Not publicly established** for Axys FX specifically. `*.pri` and historical price-day imports exist, but no public FX-date layout was found. | Unknown | **Blocked pending client data or documentation** | Pricing chapter proves price-day imports in CI, not FX-specific dating. Tier 2. Confidence: Unknown. [B2] |
| Bid / ask / close / average / spot / forward type distinction | **Not publicly established** for Axys currencies. | Unknown | **Blocked pending client data or documentation** | Uploaded `Chapter_08_Pricing(3).md`, line 138 says whether Axys stores separate bid/ask/close/evaluated/clean/dirty prices is unknown. No FX-specific subtype evidence extends that. [B2] |
| Direct vs inverse quotation | **Not publicly established.** | Unknown | **Blocked pending client data or documentation** | No file layout or manual found. [B2] |
| Reciprocal-rate generation | **Not publicly established.** | Unknown | **Blocked pending client data or documentation** | No file layout or manual found. [B2] |
| Missing/stale/corrected rates | The public evidence supports **missing/stale/calculated price handling** in CI release notes, but that evidence is for prices. It does **not** prove native Axys FX-rate handling. | Missing Price file, calculated price routing | **Evidence-only audit check** unless site confirms FX uses same machinery | Uploaded `Chapter_08_Pricing(3).md`, lines 113-119. Tier 2. Confidence: Verified for CI price workflow only. [B2] |
| `mergepri` | Consultant evidence says `mergepri` is a script command that merges price files and preserves the first source as primary. It is **not** proven to be FX-specific. | `mergepri` | **Safe to implement as configurable inference** only if the client confirms FX is staged as price-file content | Uploaded `Chapter_08_Pricing(3).md`, lines 125-129. Underlying source named as AdventGuru. Tier 3. Confidence: Medium. [B2] |

### Transaction and cash-field table

| Item | What is supported | Best-known field / token / artifact | Implementation status | Evidence note |
|---|---|---|---|---|
| Source/destination context exists in Axys-oriented transaction translation examples | Yes, in partner integration evidence. | Source/destination type and symbol fields | **Safe to implement as configurable inference** when client exports them | Uploaded `Chapter_07_Cash(3).md`, lines 63-66: source/destination type and symbol fields are explicitly mentioned with tokens `$pty`, `$ity`, `$pth`, `$cash`, `$income`, `CAUS`, `CASH`, `MMF`, `MARGIN`, `SHORT`. Tier 2. Confidence: High. [B1] |
| Cash indicator tokens | Observed in partner integration material; not validated as native universal Axys codes. | `$cash`, `$income`, `CASH`, `MMF`, `MARGIN`, `SHORT` | **Evidence-only audit check** | Same evidence. Tier 2. Confidence: High for observation; Unknown for universality. [B1] |
| Mark to Market column exists in Axys import context | Yes, in ByAllAccounts CI context. | “Mark to Market”, `defmarkmarket`, `topost.trn` | **Safe to implement as connector-specific audit check** | Uploaded `Chapter_07_Cash(3).md`, lines 69 and 320, and line 398: this field is required for non-system-currency transactions in the CI workflow. Tier 2. Confidence: High. Its accounting meaning is not established. [B1] |
| `Perf/CW` column exists in Axys import context | Yes, in ByAllAccounts CI context. | `Perf/CW`, `defperfcw`, `topost.trn` | **Safe to implement as connector-specific audit check** | Uploaded `Chapter_07_Cash(3).md`, lines 70 and 321. Tier 2. Confidence: High. Its accounting meaning is not established. [B1] |
| Currency and FX fields in transaction extracts | Expected conceptually, but field names are unknown. | “Currency”, “FX Rate” as candidate fields | **Blocked pending client data or documentation** | Uploaded `Chapter_05_Transactions(9).md`, lines 941-947 and 1364-1365. Tier 2 synthesis. Confidence: Unknown/Medium. [B4] |
| Candidate live Axys inspection list | Currency, FX, Mark to Market, Perf/CW, source/destination context should be inspected if the client can expose them. | Candidate fields only | **Evidence-only audit check** | Uploaded `Chapter_05_Transactions(9).md`, line 2630. Tier 2 synthesis. Confidence: Medium. [B4] |
| Native cross-currency trade / FX / cash linkage | **Not publicly established.** | Unknown | **Blocked pending client data or documentation** | Uploaded `Chapter_05_Transactions(9).md`, lines 2537-2546 explicitly keep FX storage, cross-currency settlements, and FX pairing as unknowns. Tier 2 synthesis. Confidence: Unknown. [B4] |

The net result is that the public evidence supports a **connector-era transaction context vocabulary**, but not a native Axys multi-currency transaction schema.

## Valuation and performance behavior

The central implementation issue is not whether Axys is “multi-currency capable.” The central issue is whether public evidence is good enough to reproduce Axys behavior. For valuation and currency-return bifurcation, the answer is largely **no**.

### Valuation-rules table

| Question | What can be said defensibly | Implementation status | Evidence note |
|---|---|---|---|
| How Axys converts local market value to portfolio/report currency | **Not publicly established.** No uploaded source provides an Axys-specific formula or report calculation note. | **Blocked pending client data or documentation** | Performance chapter keeps multi-currency details unresolved; pricing chapter says exact FX schema is unknown. [B2] [B3] |
| Which date’s FX rate is used in reports | **Not publicly established** for valuation-date, trade-date, prior-date, or settlement-date selection. | **Blocked pending client data or documentation** | No public field dictionary or report-source example found. [B2] [B3] |
| Whether cash and securities use the same conversion rule | **Not publicly established.** | **Blocked pending client data or documentation** | Cash chapter says native cash presentation by currency and local/base cash fields are not established. [B1] |
| Whether income, fees, realized gains, and flows translate on transaction date or period-end date | **Not publicly established.** | **Blocked pending client data or documentation** | No public Axys rule found. Transaction chapter preserves this area as unknown. [B4] |
| “Mark to Market” meaning in non-system-currency transaction imports | Public evidence proves only that a Mark-to-Market field is present and may be required in the ByAllAccounts CI workflow for non-system-currency transactions. Its accounting meaning is **not publicly established**. | **Safe only as connector-specific evidence check** | Uploaded `Chapter_07_Cash(3).md`, lines 69, 97, 320, 398. Tier 2. Confidence: High for field existence; Unknown for meaning. [B1] |
| `defmarkmarket` meaning | Public evidence supports only that it is an integration parameter tied to the Mark-to-Market field. The meaning of the default and how Axys uses it are **not publicly established**. | **Safe only as connector-specific evidence check** | Same evidence. [B1] |
| `Perf/CW` purpose | Public evidence supports only that a `Perf/CW` column exists in the same CI import context. Its semantic meaning is **not publicly established**. | **Safe only as connector-specific evidence check** | Uploaded `Chapter_07_Cash(3).md`, lines 70 and 321. [B1] |
| `defperfcw` meaning | Public evidence supports only that it is an integration parameter associated with that column. Native Axys meaning is **not publicly established**. | **Safe only as connector-specific evidence check** | Same evidence. [B1] |
| Historical FX corrections recalculate prior values or returns | **Not publicly established.** Stored-versus-recalculated performance remains unresolved. | **Blocked pending client rerun tests** | Uploaded `Chapter_10_Performance(3).md`, lines 103-105 and 540-573. Tier 1/2 synthesis. Confidence: Unknown. [B3] |

No fully sourced public **Axys** numerical example was found for any of those rules. I did **not** substitute a conventional multi-currency accounting formula and present it as Axys behavior.

### Performance-bifurcation table

Axys public capability material, as preserved in the uploaded performance chapter, says Axys can separate multi-currency return components attributable to **market prices** versus **currency-rate fluctuations**. That is the strongest positive finding. Everything more detailed remains unresolved. [B3]

| Question | Finding | Implementation status | Evidence note |
|---|---|---|---|
| Does Axys publicly claim market-versus-currency return bifurcation? | **Yes, at capability level.** | **Safe to implement as report-consumed capability only** | Uploaded `Chapter_10_Performance(3).md`, lines 97-99. Underlying source named as Axys product material. Tier 1. Confidence: Verified for capability only. [B3] |
| Exact formula | **Not publicly established.** | **Blocked pending official docs or client report source** | Performance chapter explicitly stops at capability level and marks multi-currency fields/examples as unresolved. [B3] |
| Arithmetic vs geometric decomposition | **Not publicly established.** | **Blocked pending official docs or client report source** | No public formula found. [B3] |
| Definitions of local return, currency return, base return, reporting-currency return | **Not publicly established.** | **Blocked pending client report samples** | No public report field dictionary found. [B3] |
| Interaction term treatment | **Not publicly established.** | **Blocked pending official docs or client report source** | No public formula found. [B3] |
| Cash-flow treatment within decomposition | **Not publicly established.** | **Blocked pending official docs or client report source** | No public report-source evidence found. [B3] |
| Multiple trading/income/risk currencies inside the same decomposition | **Not publicly established.** | **Blocked pending security-master and report examples** | Security-currency triad itself is not public in the evidence set. [B6] |
| Stored vs calculated at report runtime | **Not publicly established.** | **Blocked pending rerun tests or report source** | Uploaded `Chapter_10_Performance(3).md`, lines 103-105 and 540-573. [B3] |
| Reports that may be relevant | Only **conversion-context leads**, not proven bifurcation reports: `Performance Summary`, `Performance by Account`, `Performance by Security`, `Portfolio Cash Flow`, `Portfolio Current Value`, `Unrealized Gain/Loss`, `Realized Gain/Loss`. | **Evidence-only review leads** | Uploaded `Chapter_10_Performance(3).md`, lines 292-299. Underlying source named as Morningstar conversion-context material. Tier 2. Confidence: Medium as report leads, not as formula proof. [B3] |
| Public screenshots/sample output proving currency split | **Not located.** | **Blocked pending client report output or archival manuals** | No screenshot or sample output in the uploaded evidence set. [B3] |

A fully sourced **public numerical example of Axys currency-return bifurcation was not located**. Therefore the correct status is **“not publicly established.”**

## Export feasibility and implementation choices

The safest first implementation is a **minimum viable Axys export specification** that asks the client for exactly the fields that public evidence fails to standardize. That turns the public-evidence gap into an explicit adapter contract instead of an undocumented assumption.

### Minimum viable Axys export specification for an audit product

| Audit need | Best-known Axys source artifact | Known field/column names | Directly observed, derived, or unavailable | Public evidence sufficient? | Safest fallback |
|---|---|---|---|---|---|
| Local and base market values | Holdings or valuation report export; possibly report restatement pair | No validated Axys-native local/base field names found | Mostly **unavailable** from public evidence | **No** | Require client export with explicit `market_value_local`, `market_value_base`, `as_of_date`, `portfolio_id`, `security_id` |
| Security currency | Security master export or holding/report column | No validated Axys-native field name; only generic candidate “currency” exists | **Unavailable** from public evidence | **No** | Require explicit `security_currency` in exported holdings/security reference |
| Cash currency | Cash holding rows or separate cash extract | No validated Axys-native field name | **Unavailable** from public evidence | **No** | Require explicit `cash_currency` or infer only from client-confirmed cash-holding rows |
| FX rates by date | Client-provided file, possibly derived from Axys price/rate workflow | No validated Axys-native pair/date field names | Best treated as **client-supplied normalized input** | **No** for native Axys layout | Accept normalized `from_currency`, `to_currency`, `rate_date`, `fx_rate`, plus optional `rate_type` and `rate_source` |
| Cross-currency transactions | Transaction extract / blotter / custom REP export | Candidate fields only: currency, FX, Mark to Market, Perf/CW, source/destination context | Mostly **unavailable** from public evidence | **No** | Require explicit `transaction_currency`, optional `settlement_currency`, optional `fx_rate`, and source/destination context if used |
| Local return, currency return, base return | Performance report or report-export | No validated Axys-native field names | **Unavailable** from public evidence | **No** | If client can export them, ingest as **report-supplied values only**; otherwise do not attempt Axys-style bifurcation |
| Restated reports | Same report run in multiple currencies | Parameter names not recovered publicly | **Derived report evidence** | **No** for parameter mechanics | Accept duplicated report runs with explicit report currency in the file header or loader config |

### Implementation decision table

| Candidate behavior | Classification | Why |
|---|---|---|
| Recognize `$pathpri`, `*.pri`, and `imexPrices.log` as real Axys-oriented connector artifacts | **Safe to implement as Axys behavior** | Verified in Axys-oriented CI workflow evidence. [B2] |
| Flag `axyscur`, `defmarkmarket`, and `defperfcw` in CI-sourced Axys import files or configs | **Safe to implement as Axys behavior** in the **connector-context sense** | These names are concretely preserved in ByAllAccounts Axys integration evidence. [B1] |
| Treat `axyscur` as native per-portfolio base currency | **Safe to implement as configurable inference** only, and usually not recommended | Public evidence proves only a connector-level “system currency” parameter, not native portfolio storage. [B1] |
| Treat `.pri` as a universal Axys FX-rate file | **Blocked pending client data or documentation** | Public evidence proves `.pri` is a price-file artifact, not an FX record layout. [B2] |
| Use `mergepri` to merge client-confirmed Axys FX files with first-source priority | **Safe to implement as configurable inference** | `mergepri` is only consultant-documented, and only as a price-file merge tool. [B2] |
| Require normalized FX rates from the client when multi-currency auditing is enabled | **Safe to implement as configurable inference** | This is the cleanest way around the missing public Axys FX dictionary. |
| Infer separate trading/income/risk currencies from a single security currency field | **Blocked pending client data or documentation** | No public evidence establishes those native Axys fields. [B6] |
| Use report-supplied local/base/currency-effect values if the client can export them | **Safe to implement as evidence-only audit check** | Axys capability is public, but formula and storage are not. [B3] |
| Reproduce Axys market-versus-currency bifurcation formula | **Blocked pending client data or documentation** | Exact formula is not publicly established. [B3] |
| Treat missing or changed FX rows as evidence only when linked to exposure | **Safe to implement as evidence-only audit check** | This is defensible and does not claim Axys-native formula parity. |
| Infer cross-currency trade/cash/FX linkage automatically from transaction code alone | **Blocked pending client data or documentation** | Public evidence explicitly warns that transaction code alone is insufficient, even for cash-flow classification. [B1] [B4] |
| Use source/destination type and symbol as context for ambiguous transactions | **Safe to implement as configurable inference** when the client exports them | Strong partner evidence supports their usefulness, but not universal Axys-native availability. [B1] [B4] |

The smallest realistic contract for Phase 1 is therefore:

- **required**: normalized holdings with explicit currency,
- **required for multi-currency mode**: normalized FX rates,
- **required if transaction review is enabled**: normalized transactions with explicit or client-mapped transaction currency,
- **optional but valuable**: Mark-to-Market and `Perf/CW` fields when the client uses the ByAllAccounts Axys connector,
- **optional display-only**: report-exported local/base/currency-effect outputs.

## Unresolved questions and bibliography

### Unresolved questions

| Unresolved question | Current status | Exact evidence that would resolve it |
|---|---|---|
| Is Axys base currency native, global, or portfolio-level? | **Unknown** | Official Axys portfolio master dictionary, actual portfolio export showing currency field, or report parameter documentation |
| What exactly does `axyscur` map to in native Axys? | **Strongly supported as connector config only** | Original ByAllAccounts manual page that defines the parameter, plus a matching Axys field or screen |
| Are trading, income, and risk currencies stored separately for a security? | **Unknown** | Official `sec.inf` layout, security export sample, or REP field catalog showing those fields |
| Where are spot and forward rates stored? | **Unknown** | Official FX interface guide, actual FX import/export sample, or manual describing rate object type |
| Does Axys store FX as securities, currency pairs, prices, or another record type? | **Unknown** | File layout or sample rows with currency pair and date |
| How are direct/inverse quotes handled? | **Unknown** | FX manual or sample file with pair convention explicitly documented |
| What does “Mark to Market” mean in Axys non-system-currency imports? | **Unknown beyond field existence** | Original ByAllAccounts manual page or Axys import spec with column definition |
| What does `Perf/CW` mean? | **Unknown beyond field existence** | Original manual page or import spec with explicit semantic definition |
| How does Axys select the FX date for valuation and performance? | **Unknown** | Performance manual, REP source, or same-period rerun test changing only FX date input |
| What is the Axys market-versus-currency return formula? | **Not publicly established** | Official report guide, REP source, or worked vendor example with reconcilable columns |
| Are historical FX corrections stored or only reflected on rerun? | **Unknown** | Controlled before/after report reruns plus stored/exported performance evidence |
| Can transaction exports link trade, FX conversion, and cash movement? | **Unknown** | Real Axys transaction export or custom REP extract containing all three relationships |
| Does Axys use separate cash securities by currency? | **Unknown** | Holdings/report sample naming multiple currencies in the same portfolio, or official cash-model documentation |

### Bibliography

The sessions’ strongest implementation evidence came from the uploaded reference chapters. Those chapters preserve source titles and evidence summaries, but in several cases they do **not** preserve a durable public URL. I therefore separate the bibliography into **uploaded-file evidence** and the one **public URL** recovered during this session that is relevant only to SS&C Advent branding, not to Axys mechanics.

#### Uploaded-file evidence

| ID | Source title | Publisher / provenance | Date | URL | Pages / lines used | Supporting quote or precise paraphrase | Source tier | Confidence |
|---|---|---|---|---|---|---|---|---|
| B1 | `Chapter_07_Cash(3).md` | AXYS / APX Reference Repository uploaded by user | Prepared 2026-06-29 | Uploaded file only; direct public URL not preserved in file | Lines 63-70, 96-99, 320-321, 398, 431-436 | Preserves that Axys-oriented integration evidence includes `axyscur`, `defmarkmarket`, `defperfcw`, source/destination cash-context tokens, and that native cash-currency presentation is not established | Tier 2 for underlying ByAllAccounts guide; Tier 1 only for any cited SS&C product material; mixed chapter synthesis | High for connector facts; Unknown for native Axys semantics |
| B2 | `Chapter_08_Pricing(3).md` | AXYS / APX Reference Repository uploaded by user | Repository chapter, no separate published date in excerpt | Uploaded file only; direct public URLs not preserved in file | Lines 74-77, 101-105, 113-129, 131-140, 533-539 | Preserves that Axys CI uses `$pathpri`, `*.pri`, and `imexPrices.log`; that missing/stale/calculated price behavior is documented in CI release notes; and that `mergepri` is consultant-documented only | Tier 2 for ByAllAccounts CI materials; Tier 3 for AdventGuru; mixed chapter synthesis | Verified for CI workflow artifacts; Medium for `mergepri`; Unknown for native FX layout |
| B3 | `Chapter_10_Performance(3).md` | AXYS / APX Reference Repository uploaded by user | Repository chapter, no separate published date in excerpt | Uploaded file only; direct public URLs not preserved in file | Lines 94-100, 103-105, 292-299, 540-573 | Preserves that Axys product material claims market-versus-currency return components; but stored-vs-calculated behavior, exact formulas, multi-currency fields, and report mechanics remain unresolved | Tier 1 for underlying SS&C product material where cited; Tier 2 for Morningstar conversion report leads; mixed chapter synthesis | Verified for capability only; Unknown for formula and storage |
| B4 | `Chapter_05_Transactions(9).md` | AXYS / APX Reference Repository uploaded by user | Repository chapter, no separate published date in excerpt | Uploaded file only; direct public URLs not preserved in file | Lines 941-947, 1364-1365, 2537-2546, 2630-2632 | Preserves that transaction currency and FX rate are expected concepts, but native names are unknown; and that FX storage, cross-currency settlement, and FX pairing remain unknown | Tier 2 synthesis backed by partner/conversion sources | Unknown/Medium |
| B5 | `Chapter_01_Overview(1).md` | AXYS / APX Reference Repository uploaded by user | Prepared 2026-06-29 | Uploaded file only; direct public URLs not preserved in file | Lines 715-718 | Preserves a release-note finding that Axys 3.8.7 had “additional/improved multicurrency reports” | Tier 1 if underlying release note is official; URL not preserved | Verified as release-note evidence within the chapter |
| B6 | `Chapter_04_Security_Master(10).md` | AXYS / APX Reference Repository uploaded by user | Repository chapter, no separate published date in excerpt | Uploaded file only; direct public URLs not preserved in file | Line 817 | Preserves only a generic candidate “currency” field, not separate trading/income/risk currencies | Tier 2 synthesis | Medium for generic currency discovery; Unknown for separate Axys currency triad |

#### Public URL recovered in this session

| ID | Source title | Publisher | Date | URL | Section used | Supporting quote or precise paraphrase | Source tier | Confidence |
|---|---|---|---|---|---|---|---|---|
| W1 | “SS&C Advent sues software company Advent AI for trademark infringement” | Reuters | 2026-06-23 | `https://www.reuters.com/legal/legalindustry/ssc-advent-sues-software-company-advent-ai-trademark-infringement-2026-06-23/` | Body text | Used only for current-brand context: SS&C Advent is the current business name descended from Advent Software; it does **not** support any Axys mechanic | Tier 3 for brand context only | High for brand context; irrelevant to mechanics |

## IMPLEMENTATION HANDOFF

### A. Capabilities proven by evidence

The evidence supports these narrow statements:

- Axys-oriented integration workflows use **`axyscur`**, **`defmarkmarket`**, and **`defperfcw`**. [B1]
- Axys-oriented pricing workflows use **`$pathpri`**, **`*.pri`**, and **`imexPrices.log`**. [B2]
- Axys public capability material supports a **market-price versus currency-rate fluctuation** split at the product-claim level. [B3]
- Axys 3.8.7 release-note evidence says there were **additional/improved multicurrency reports**. [B5]

### B. Data fields and artifacts proven by evidence

The evidence proves or strongly supports these artifacts and names:

- `axyscur`
- `defmarkmarket`
- `defperfcw`
- `topost.trn`
- `$pathpri`
- `*.pri`
- `imexPrices.log`
- source/destination type and symbol context in partner workflows
- observed cash-context tokens such as `$cash`, `$income`, `CASH`, `MMF`, `MARGIN`, `SHORT`
- candidate transaction concepts for `Currency`, `FX Rate`, `Mark to Market`, and `Perf/CW`, but **not** their native Axys field dictionary. [B1] [B2] [B4]

### C. Behavior that may be implemented only as configurable inference

These are reasonable to support, but only with explicit client mapping or configuration:

- Treat `axyscur` as the **site/system currency** in a ByAllAccounts Axys connector context.
- Accept client-provided normalized FX rates as the authoritative audit input for Axys multi-currency mode.
- If the client confirms that FX is staged through `.pri`-style files, allow a configurable loader and source-precedence policy informed by `mergepri`.
- Use source/destination type and symbol context to classify ambiguous cash-like transactions when those fields are exported.
- Consume report-supplied local/base/currency-effect values as display and comparison evidence without claiming formula parity. [B1] [B2] [B3] [B4]

### D. Behavior that must remain unknown

These must remain unknown until the client supplies direct evidence:

- native Axys base-currency storage and scope
- native Axys security trading/income/risk currency fields
- Axys-native FX storage model and record layout
- spot versus forward operational distinction in Axys files
- pair quotation convention and reciprocal handling
- valuation-date FX selection rules
- cash-versus-security conversion rule parity
- exact meaning of Mark to Market and `Perf/CW`
- exact market-versus-currency bifurcation formula
- interaction-term treatment
- cross-currency trade / cash / FX linkage
- historical FX correction restatement behavior [B1] [B2] [B3] [B4] [B6]

### E. The smallest realistic first development phase

The smallest realistic first phase is:

1. Require explicit **client-mapped currency fields** on holdings, transactions, and any cash extract.
2. Require a normalized **FX rate file** with `from_currency`, `to_currency`, `rate_date`, and `fx_rate`.
3. Treat Axys-specific multi-currency mechanics as **evidence-only** unless the client provides direct export or report proof.
4. Add connector-specific checks for **`axyscur`**, **Mark to Market**, and **`Perf/CW`** when the client uses a ByAllAccounts-style Axys workflow.
5. Accept report-supplied **local/base/currency-effect** outputs when available, but do not reconstruct them.
6. Block any attempt to claim **Axys-native currency-return formula compatibility** until client report source, archival manuals, or controlled rerun evidence is obtained.
