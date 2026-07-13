# Axys/APX Common-Core Export Reference

This note sketches a PPAR-normalized Axys/APX extract shape for analytics and
performance auditing. It is an extraction-planning aid, not an official
Axys/APX schema, executable export recipe, or generic implementation contract.

See [Performance Comparison Design Notes](performance_comparison_design.md) for
the broader comparison model, implemented checkpoint, YAML semantics, and report
bundle workflow that can consume these normalized export shapes.

Axys/APX installations vary by site. The current evidence does not establish a
universal IMEX object catalog, performance object names, field mnemonics,
profile names, command syntax, or REP layouts. In particular, `portperf.csv`
and `secperf.csv` are PPAR-normalized filenames, not verified native object or
profile names.

## Extraction Planning Worksheet

Use this table when speaking with an Axys/APX administrator or report writer.
Do not turn it into an IMEX command script until the local installation proves
the exact object/profile names, fields, parameters, and date/currency basis.

| PPAR dataset | Practical first source | Local questions to resolve |
| --- | --- | --- |
| `portperf.csv` | REP performance report preferred. | Which report reproduces the reported portfolio return? Is the value stored or report-calculated? What are its date, currency, and gross/net bases? |
| `secperf.csv` | REP security-performance or attribution report preferred. | Does it provide security return and portfolio/security keys? Do weights and contributions foot to the portfolio report? |
| `holdings.csv` | IMEX positions/holdings export or REP appraisal report. | Are values local or portfolio-base? Is accrued income included in market value or stated separately? Can both beginning and ending dates be produced? |
| `transactions.csv` | IMEX transaction export first; REP/custom report fallback. | Are transaction code, amount, security, and economic date present? For ambiguous codes, are source/destination and special-security fields available? |
| `fx_rates.csv` | Validated REP, FX/price, or other controlled local source. | What is the quote convention, effective date, source, rate type, portfolio base currency, and linked local exposure? |
| `splits.csv` | `split.inf` or an equivalent local split-factor export. | Is the factor a multiplier or inverse? Which date is represented? |
| `sec_ref.csv` | IMEX security-information export or security-master report. | Which identifier is stable, and which classification/currency fields are current rather than historical? |

## Starter Field Reference

These tables list candidate source or report labels that a local export may use
for values normalized by PPAR. They do not establish official Axys/APX field
names. Confidence describes availability of the underlying value, not confidence
that the candidate label is exact. The local extract and PPAR column mapping
remain authoritative.

### Portfolio Performance

| PPAR dataset | Candidate source/report label | Canonical meaning | Other candidate labels | Value availability confidence |
| --- | --- | --- | --- | --- |
| `portperf` | `PORT` | `portfolio_id` | `ACCOUNT`, `ACCT`, `PORTFOLIO` | High |
| `portperf` | `DATE` | `period_end_date` | `AS_OF_DATE`, `PERIOD_END` | High |
| `portperf` | `BEG_MV` | `beginning_market_value` | `BEGIN_MV`, `BMV`, `BEGIN_VALUE` | Medium |
| `portperf` | `END_MV` | `ending_market_value` | `EMV`, `ENDING_VALUE`, `MARKET_VALUE` | Medium |
| `portperf` | `RETURN` | `portfolio_return` | `RET`, `PERF`, `PERFORMANCE` | Medium |
| `portperf` | `FLOW` | `net_external_flow` | `NET_FLOW`, `CONTRIB_WITHDRAW`, `CASH_FLOW` | Medium |
| `portperf` | `INCOME` | `income` | `INC`, `DIV_INT`, `INV_INCOME` | Medium |
| `portperf` | `GAIN_LOSS` | `gain_loss` | `GL`, `GAIN`, `REAL_UNREAL_GL` | Low/Medium |

### Security Performance

| PPAR dataset | Candidate source/report label | Canonical meaning | Other candidate labels | Value availability confidence |
| --- | --- | --- | --- | --- |
| `secperf` | `PORT` | `portfolio_id` | `ACCOUNT`, `ACCT`, `PORTFOLIO` | High |
| `secperf` | `SEC` | `security_id` | `SECURITY`, `SEC_ID`, `SECNO` | High |
| `secperf` | `DATE` | `period_end_date` | `AS_OF_DATE`, `PERIOD_END` | High |
| `secperf` | `BEG_MV` | `beginning_market_value` | `BEGIN_MV`, `BMV`, `BEGIN_VALUE` | Medium |
| `secperf` | `END_MV` | `ending_market_value` | `EMV`, `ENDING_VALUE`, `MARKET_VALUE` | Medium |
| `secperf` | `WEIGHT` | `portfolio_weight` | `WGT`, `PCT_ASSETS`, `PERCENT_ASSETS` | Medium |
| `secperf` | `RETURN` | `security_return` | `RET`, `PERF`, `PERFORMANCE` | Medium |
| `secperf` | `CONTRIB` | `return_contribution` | `CONTRIBUTION`, `CTR`, `RET_CONTRIB` | Medium |
| `secperf` | `INCOME` | `income` | `INC`, `DIV_INT`, `INV_INCOME` | Medium |
| `secperf` | `GAIN_LOSS` | `gain_loss` | `GL`, `GAIN`, `REAL_UNREAL_GL` | Low/Medium |

### Security Master

| PPAR dataset | Candidate source/report label | Canonical meaning | Other candidate labels | Value availability confidence |
| --- | --- | --- | --- | --- |
| `sec-ref` | `SEC` | `security_id` | `SECURITY`, `SEC_ID`, `SECNO` | High |
| `sec-ref` | `DESC` | `security_description` | `DESCRIPTION`, `NAME`, `SEC_DESC` | High |
| `sec-ref` | `CUSIP` | `cusip` | `CUSIP_NO`, `CUSIP_NUMBER` | High |
| `sec-ref` | `SYMBOL` | `ticker_symbol` | `TICKER`, `TICKER_SYMBOL` | Medium |
| `sec-ref` | `TYPE` | `security_type` | `SEC_TYPE`, `ASSET_TYPE`, `INV_TYPE` | Medium |
| `sec-ref` | `CURRENCY` | `currency_code` | `CURR`, `CCY`, `LOCAL_CCY` | Medium |
| `sec-ref` | `COUNTRY` | `country` | `CNTRY`, `ISSUE_COUNTRY` | Medium |
| `sec-ref` | `INDUSTRY` | `industry` | `IND`, `INDUSTRY_CODE`, `SECTOR` | Medium |
| `sec-ref` | `MATURITY` | `maturity_date` | `MAT_DATE`, `MATURITY_DATE` | Medium |
| `sec-ref` | `COUPON` | `coupon_rate` | `CPN`, `COUPON_RATE` | Medium |

### Transactions

| PPAR dataset | Candidate source/report label | Canonical meaning | Other candidate labels | Value availability confidence |
| --- | --- | --- | --- | --- |
| transactions | `PORT` | `portfolio_id` | `ACCOUNT`, `ACCT`, `PORTFOLIO` | High |
| transactions | `DATE` | `trade_date` | `TRADE_DATE`, `TRD_DATE` | High |
| transactions | `SETTLE_DATE` | `settlement_date` | `SETTLE`, `SET_DATE`, `STL_DATE` | Medium |
| transactions | `SEC` | `security_id` | `SECURITY`, `SEC_ID`, `SECNO` | High |
| transactions | `TRAN` | `transaction_code` | `TRAN_CODE`, `TRANS_CODE`, `ACTIVITY` | Medium |
| transactions | `QTY` | `quantity` | `QUANTITY`, `SHARES`, `UNITS` | High |
| transactions | `PRICE` | `transaction_price` | `PX`, `TRADE_PRICE` | High |
| transactions | `AMOUNT` | `net_amount` | `AMT`, `NET_AMOUNT`, `NET_AMT` | High |
| transactions | `COMMISSION` | `commission` | `COMM`, `COMMISH` | Medium |
| transactions | `BROKER` | `broker` | `BRKR`, `BROKER_CODE` | Low/Medium |

### Holdings

| PPAR dataset | Candidate source/report label | Canonical meaning | Other candidate labels | Value availability confidence |
| --- | --- | --- | --- | --- |
| holdings | `PORT` | `portfolio_id` | `ACCOUNT`, `ACCT`, `PORTFOLIO` | High |
| holdings | `DATE` | `as_of_date` | `AS_OF_DATE`, `HOLDING_DATE` | High |
| holdings | `SEC` | `security_id` | `SECURITY`, `SEC_ID`, `SECNO` | High |
| holdings | `QTY` | `quantity` | `QUANTITY`, `SHARES`, `UNITS` | High |
| holdings | `PRICE` | `price` | `PX`, `MARKET_PRICE` | High |
| holdings | `MV` | `market_value` | `MKT_VAL`, `MARKET_VALUE`, `VALUE` | High |
| holdings | `COST` | `cost_basis` | `BOOK_COST`, `TAX_COST`, `ORIG_COST` | Medium |
| holdings | `ACCRUED` | `accrued_income` | `ACCRUED_INT`, `ACCRUAL` | Medium |
| holdings | `CURRENCY` | `currency_code` | `CURR`, `CCY` | Medium |

### FX Rates

| PPAR dataset | Candidate source/report label | Canonical meaning | Other candidate labels | Value availability confidence |
| --- | --- | --- | --- | --- |
| FX / currency | `DATE` | `fx_rate_date` | `AS_OF_DATE`, `RATE_DATE` | Unknown pending local discovery |
| FX / currency | `CURRENCY` | `currency_code` | `CURR`, `CCY`, `FROM_CCY` | Unknown pending local discovery |
| FX / currency | `BASE_CURRENCY` | `base_currency_code` | `BASE_CURR`, `BASE_CCY`, `TO_CCY` | Unknown pending local discovery |
| FX / currency | `RATE` | `fx_rate` | `FX_RATE`, `EXCH_RATE`, `EXCHANGE_RATE` | Unknown pending local discovery |

### Cash Holdings

PPAR does not define a separate cash dataset. Normalize cash balances as
holdings such as `CASHUSD`, `CASHEUR`, or `CASHGBP`. Use holding `quantity` for
the currency units, `price` (normally `1.0`) for local unit value,
`market_value` for row-currency value, and `base_market_value` for translated
portfolio-base value. A cash-ledger export requires an adapter into that single
holdings representation.
