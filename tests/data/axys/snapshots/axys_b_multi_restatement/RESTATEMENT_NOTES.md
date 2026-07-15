# Axys Multi-Portfolio Restatement Fixture Notes

This snapshot intentionally differs from `axys_a` across multiple portfolios.
It is meant for demo/report review, where one portfolio is not enough to judge
triage, sorting, filters, and drilldown flow.

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
- `secref.csv`: Restated the `AAPL` name and sector fields.
- `secref.csv`: Added `RESTATED_SEC`.
- `portperf.csv`: Restated `PORT_B` for `2025-05-30` with a larger gain/loss
  and return change.
- `secperf.csv`: Restated the `PORT_B`/`META` gain/loss, return, and
  contribution fields for `2025-05-30`.
- `holdings.csv`: Restated the `PORT_B`/`META` price, market value,
  cost, and accrued amount for `2025-05-30`.
- `transactions.csv`: Restated the `PORT_B`/`META` transaction `TXN000078`
  quantity, amount, and commission.
- `portperf.csv`: Restated `PORT_C` for `2025-05-30` with a return and flow
  change.
- `holdings.csv`: Restated the `PORT_C`/`NVDA` quantity, price,
  market value, cost, and accrued amount for `2025-05-30`.
- `transactions.csv`: Restated `PORT_C` transaction `TXN000044` amount and
  commission. The multi-restatement YAML supplies transaction semantics so
  these rows can exercise transaction impact and context review.

The existing `axys_a` and `axys_b` snapshots remain identical for a baseline
no-difference comparison.

Related YAML demos:

- `ppar_performance_comparison_multi_restatement.yaml` supplies explicit
  contribution and transaction policies and is the default report-bundle demo.
- `ppar_performance_comparison_policy_gap_demo.yaml` reuses this snapshot but
  intentionally leaves selected policy inputs unresolved so Problems-grid
  action wording can be reviewed without duplicating CSV data.
