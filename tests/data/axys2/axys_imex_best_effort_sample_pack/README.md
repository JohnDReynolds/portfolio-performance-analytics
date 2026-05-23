# Axys IMEX Best-Effort Sample Data Pack

Generated: 2026-05-23
Coverage: 2025-05-01 through 2026-05-22 (277 business days; 13 month-end-ish periods)
Portfolios: PORT_A, PORT_B, PORT_C
Securities: 25 US equity tickers, 7 US Treasury CUSIPs, 1 USD cash security

IMPORTANT:
- This is synthetic sample data in a best-effort Axys/IMEX-like layout.
- It is not an official SS&C Axys export and not an official Axys data dictionary.
- Equity tickers/CUSIPs and Treasury CUSIPs are intended to be realistic identifiers; prices, transactions, holdings, and performance are synthetic.
- Native field names are intentionally simple/common-field headers from the previously defined common-core mapping.
- Performance figures are internally approximate, not production-grade performance accounting calculations.
- The goal is importer development, testing, and product-design realism.

Files:
- portperf.csv: portfolio-level monthly performance/control totals
- secperf.csv: security-level monthly performance/contribution rows
- sec_ref.csv: security master/reference data
- transactions.csv: posted transaction-style activity, >=100 rows per portfolio
- positions_holdings.csv: month-end position snapshots, including CASH_USD
- prices.csv: daily price history for every security including Treasuries and CASH_USD
- fx_currency.csv: daily USD/USD FX rates
- cash.csv: month-end cash balances by portfolio/currency

Intentional event coverage:
- BUY and SELL transactions
- DIV dividend transactions
- INT Treasury interest transactions
- SPLIT sample split transaction
- CASH_USD cash rows
