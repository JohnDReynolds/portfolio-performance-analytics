# Axys Common-Core Export Reference

This note sketches a common-core Axys export shape for analytics and
performance-comparison demos. It is a starter reference, not a generic Axys
implementation contract.

See [Performance Comparison Design Notes](performance_comparison_design.md) for
the broader comparison model, implemented checkpoint, YAML semantics, and report
bundle workflow that can consume these normalized export shapes.

Axys installations vary by site. Treat the IMEX profile names, dataset names,
field mnemonics, date syntax, and portfolio-list syntax below as placeholders
that must be adapted to the local installation. Some firms may export these
datasets through REP/Replang reports instead of IMEX profiles.

## IMEX Template

```bat
REM ============================================================
REM Axys common-core IMEX export script - TEMPLATE ONLY
REM Not guaranteed generic across Axys installations.
REM Variances:
REM   1. IMEX profile names are site-specific.
REM   2. Dataset/file aliases may differ.
REM   3. Field mnemonics may differ.
REM   4. Date/portfolio parameter syntax may differ.
REM   5. Some firms export via REP/replang instead of IMEX profiles.
REM ============================================================

SET AXYS_IMEX=C:\Axys3\IMEX.EXE
SET OUTDIR=C:\AxysExports\CommonCore
SET PORTLIST=C:\AxysExports\portfolios.txt
SET START_DATE=2026-02-01
SET END_DATE=2026-02-28

REM ---- Portfolio-level performance: official control total
REM Expected dataset/profile: portperf
"%AXYS_IMEX%" EXPORT PROFILE=PORTPERF_COMMON PORTLIST="%PORTLIST%" START=%START_DATE% END=%END_DATE% OUT="%OUTDIR%\portperf.csv"

REM ---- Security-level performance: explains portfolio return
REM Expected dataset/profile: secperf
"%AXYS_IMEX%" EXPORT PROFILE=SECPERF_COMMON PORTLIST="%PORTLIST%" START=%START_DATE% END=%END_DATE% OUT="%OUTDIR%\secperf.csv"

REM ---- Security master / sec-ref
REM Site variance: may be called sec, sec-ref, security, security master.
"%AXYS_IMEX%" EXPORT PROFILE=SECREF_COMMON OUT="%OUTDIR%\sec_ref.csv"

REM ---- Transactions
REM Site variance: transaction source may be posted transactions,
REM trade blotter, or a REP/replang transaction ledger.
"%AXYS_IMEX%" EXPORT PROFILE=TRANSACTIONS_COMMON PORTLIST="%PORTLIST%" START=%START_DATE% END=%END_DATE% OUT="%OUTDIR%\transactions.csv"

REM ---- Holdings / holdings
REM Site variance: may export holding file, appraisal report, or holdings report.
"%AXYS_IMEX%" EXPORT PROFILE=HOLDINGS_COMMON PORTLIST="%PORTLIST%" DATE=%END_DATE% OUT="%OUTDIR%\holdings.csv"

REM ---- Prices
REM Site variance: historical price files may be by date/security type.
"%AXYS_IMEX%" EXPORT PROFILE=PRICES_COMMON START=%START_DATE% END=%END_DATE% OUT="%OUTDIR%\prices.csv"

REM ---- FX rates
REM Optional for single-currency firms.
"%AXYS_IMEX%" EXPORT PROFILE=FX_COMMON START=%START_DATE% END=%END_DATE% OUT="%OUTDIR%\fx_rates.csv"

REM ---- Cash balances
REM Often derivable from holdings if cash is represented as security rows.
"%AXYS_IMEX%" EXPORT PROFILE=CASH_COMMON PORTLIST="%PORTLIST%" DATE=%END_DATE% OUT="%OUTDIR%\cash.csv"
```

## Starter Field Reference

These tables describe likely Axys source fields and common aliases for the
normalized datasets used by ppar. They are intentionally conservative reference
notes. A local `axys_column_mappings.yaml` file remains authoritative when a
site uses different field names.

### Portfolio Performance

| Axys Native Dataset | Most Common Native Field | Canonical Meaning | Alias Native Field Names | Confidence |
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

| Axys Native Dataset | Most Common Native Field | Canonical Meaning | Alias Native Field Names | Confidence |
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

| Axys Native Dataset | Most Common Native Field | Canonical Meaning | Alias Native Field Names | Confidence |
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

| Axys Native Dataset | Most Common Native Field | Canonical Meaning | Alias Native Field Names | Confidence |
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

### Holdings And Holdings

| Axys Native Dataset | Most Common Native Field | Canonical Meaning | Alias Native Field Names | Confidence |
| --- | --- | --- | --- | --- |
| holdings / holdings | `PORT` | `portfolio_id` | `ACCOUNT`, `ACCT`, `PORTFOLIO` | High |
| holdings / holdings | `DATE` | `as_of_date` | `AS_OF_DATE`, `HOLDING_DATE` | High |
| holdings / holdings | `SEC` | `security_id` | `SECURITY`, `SEC_ID`, `SECNO` | High |
| holdings / holdings | `QTY` | `quantity` | `QUANTITY`, `SHARES`, `UNITS` | High |
| holdings / holdings | `PRICE` | `price` | `PX`, `MARKET_PRICE` | High |
| holdings / holdings | `MV` | `market_value` | `MKT_VAL`, `MARKET_VALUE`, `VALUE` | High |
| holdings / holdings | `COST` | `cost_basis` | `BOOK_COST`, `TAX_COST`, `ORIG_COST` | Medium |
| holdings / holdings | `ACCRUED` | `accrued_income` | `ACCRUED_INT`, `ACCRUAL` | Medium |
| holdings / holdings | `CURRENCY` | `currency_code` | `CURR`, `CCY` | Medium |

### Prices

| Axys Native Dataset | Most Common Native Field | Canonical Meaning | Alias Native Field Names | Confidence |
| --- | --- | --- | --- | --- |
| prices | `SEC` | `security_id` | `SECURITY`, `SEC_ID`, `SECNO` | High |
| prices | `DATE` | `price_date` | `PRICE_DATE`, `AS_OF_DATE` | High |
| prices | `PRICE` | `price` | `PX`, `CLOSE_PRICE`, `MARKET_PRICE` | High |
| prices | `CURRENCY` | `currency_code` | `CURR`, `CCY` | Medium |
| prices | `SOURCE` | `price_source` | `PRICE_SOURCE`, `SRC`, `VENDOR` | Low/Medium |

### FX Rates

| Axys Native Dataset | Most Common Native Field | Canonical Meaning | Alias Native Field Names | Confidence |
| --- | --- | --- | --- | --- |
| FX / currency | `DATE` | `fx_rate_date` | `AS_OF_DATE`, `RATE_DATE` | High |
| FX / currency | `CURRENCY` | `currency_code` | `CURR`, `CCY`, `FROM_CCY` | High |
| FX / currency | `BASE_CURRENCY` | `base_currency_code` | `BASE_CURR`, `BASE_CCY`, `TO_CCY` | Medium |
| FX / currency | `RATE` | `fx_rate` | `FX_RATE`, `EXCH_RATE`, `EXCHANGE_RATE` | High |

### Cash

| Axys Native Dataset | Most Common Native Field | Canonical Meaning | Alias Native Field Names | Confidence |
| --- | --- | --- | --- | --- |
| cash | `PORT` | `portfolio_id` | `ACCOUNT`, `ACCT`, `PORTFOLIO` | High |
| cash | `DATE` | `as_of_date` | `AS_OF_DATE`, `BALANCE_DATE` | High |
| cash | `CURRENCY` | `currency_code` | `CURR`, `CCY` | High |
| cash | `CASH` | `cash_balance` | `CASH_BAL`, `BALANCE`, `CASH_BALANCE` | Medium |
| cash | `MV` | `market_value_base` | `MARKET_VALUE`, `BASE_VALUE`, `VALUE` | Medium |
