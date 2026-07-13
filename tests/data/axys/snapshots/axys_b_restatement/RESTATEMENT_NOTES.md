# Axys Restatement Fixture Notes

This snapshot intentionally differs from `axys_a` for future performance
comparison tests.

Controlled changes:

- `portperf.csv`: Restated `PORT_A` for `2025-05-30`.
- `secperf.csv`: Restated the `PORT_A`/`AAPL` row for `2025-05-30`.
- `secperf.csv`: Added `RESTATED_SEC` for `PORT_A` on `2025-05-30`.
- `secperf.csv`: Removed `PFE` for `PORT_A` on `2025-05-30`.
- `holdings.csv`: Restated the `PORT_A`/`AAPL` quantity, market
  value, cost, and accrued amount, plus the `CASHUSD` quantity and market value,
  for `2025-05-30`.
- `fx_rates.csv`: Restated the `EUR` to `USD` spot rate for `2025-05-30`.
- `transactions.csv`: Restated the `TXN000001` AAPL transaction quantity,
  price, and amount.
- `sec_ref.csv`: Restated the `AAPL` name and sector fields.
- `sec_ref.csv`: Added `RESTATED_SEC`.

The existing `axys_a` and `axys_b` snapshots remain identical for a baseline
no-difference comparison.
