# Full-Spec Restatement Notes

Snapshot B intentionally restates selected rows from the operational snapshot:

- `ALPHA`, latest period: AAPL price and security return increase; the
  `CASH_USD` position increases. Together these fully explain the portfolio-period
  performance difference.
- `ALPHA`, earlier period: one dividend transaction amount increases. This
  fully explains another portfolio-period performance difference.
- `BALANCED`, middle period: one dividend transaction amount increases. This
  fully explains the portfolio-period performance difference.
- `BALANCED`, earlier period: one buy transaction has changed quantity, price,
  and commission fields. Those fields are intentionally review-only, so the
  period is unexplained by additive rows but still has visible `Underlying
  Causes` sheet evidence.
- `INCOME`, latest period: TNOTE2Y accrued income and reported security return
  increase; the same AAPL price change is also visible as an input difference.
  TNOTE2Y quantity and market value are visible as related input differences,
  and cost remains context. This period is partly explained.
- `INCOME`, middle period: the `CASH_USD` position increases. This fully
  explains the portfolio-period performance difference.

The corresponding YAML files define transaction semantics. Standard field roles
distinguish performance inputs, related input components, reported performance
checks, and context rows.
