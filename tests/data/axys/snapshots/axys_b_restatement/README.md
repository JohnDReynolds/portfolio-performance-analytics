# Axys IMEX Best-Effort Sample Data Pack

Generated: 2026-05-23
Coverage: 2025-05-01 through 2026-05-22 (277 business days; 13 month-end-ish periods)
Portfolios: PORT_A, PORT_B, PORT_C
Securities: 25 US equity tickers, 7 US Treasury CUSIPs, 1 USD cash security

IMPORTANT:
- This is synthetic sample data in a best-effort Axys/IMEX-like layout.
- It is not an official SS&C Axys export and not an official Axys data dictionary.
- Equity tickers/CUSIPs and Treasury CUSIPs are intended to be realistic identifiers; transactions, holdings, and performance are synthetic.
- Native field names are intentionally simple/common-field headers from the previously defined common-core mapping.
- Performance figures are internally approximate, not production-grade performance accounting calculations.
- The goal is importer development, testing, and product-design realism.

Files:
- portperf.csv: portfolio-level monthly performance/control totals, merged with
  validation scenarios and compatibility columns
- secperf.csv: security-level monthly performance/contribution rows, merged
  with validation scenarios and compatibility columns
- sec_ref.csv: security master/reference data, merged with validation
  classification columns
- classification_lookup.csv: lookup-table classification source
- unreachable_target_secperf.csv: intentionally unreconcilable validation rows
- transactions.csv: posted transaction-style activity, >=100 rows per portfolio
- holdings.csv: month-end holding snapshots, including CASH_USD
- fx_rates.csv: daily FX rates
- cash.csv: month-end cash balances by portfolio/currency

Shared fixture configuration lives one directory above this snapshot:

- axys_column_mappings.yaml: Axys column mapping configuration
- ppar_performance_comparison.yaml: comparison configuration for axys_a and axys_b

Intentional event coverage:
- BUY and SELL transactions
- DIV dividend transactions
- INT Treasury interest transactions
- SPLIT sample split transaction
- CASH_USD cash rows
