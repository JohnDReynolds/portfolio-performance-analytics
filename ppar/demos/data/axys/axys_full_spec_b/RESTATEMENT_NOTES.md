# Full-Spec Restatement Notes

Snapshot B intentionally restates selected rows from the operational snapshot:

- `ALPHA`, January 2026: a `wd` external-withdrawal amount increases, and the
  ending `CASH_USD` holding decreases by the same amount. The transaction row is
  treated as external-flow evidence because cash security and external-party
  source/destination context match the packaged YAML rule.
- `ALPHA`, March 2026: one AAPL `by` transaction amount changes, and the same
  transaction also has changed quantity, price, and commission fields. The
  amount is the modeled transaction input; the changed components remain
  supporting evidence.
- `ALPHA`, May 2026: AAPL price and security-return changes plus a `CASH_USD`
  holding correction fully explain the portfolio-period performance difference.
- `BALANCED`, January 2026: one MSFT `sl` transaction changes amount, quantity,
  price, and commission. The amount is modeled as security sell activity, while
  quantity, price, and commission remain supporting evidence.
- `BALANCED`, March 2026: an inserted `li` row on `CASH_USD` adds an external
  cash contribution with external-party context, and the generated ending cash
  holding, portfolio performance, and reconstruction diagnostics stay aligned.
- `BALANCED`, April 2026: one JPM `dv` dividend amount increases. This
  performance-income transaction amount is mirrored by higher `CASH_USD`.
- `BALANCED`, May 2026: the same AAPL price correction applies globally, and an
  MSFT holding market value is corrected. Together these explain part of the
  portfolio-period and MSFT security-period performance differences. Snapshot B
  also includes a small intentional reported-return residual so the packaged
  reports show a realistic Partly Explained review case.
- `INCOME`, January 2026: one fee-like `dp` transaction increases the advisory
  fee expense and lowers `CASH_USD`. The row is classified from special-security
  context, not from the `dp` code alone.
- `INCOME`, May 2026: TNOTE2Y `in` interest increases, TNOTE2Y accrued interest
  and market value change, and the same AAPL price correction is also visible.
  TNOTE2Y market value and accrued interest are additive, TNOTE2Y quantity is
  related evidence, and cost remains context.
- `INCOME`, April 2026: a TNOTE5Y cost-only correction remains review evidence
  and does not explain performance by itself. Snapshot B also includes an
  intentional reported-return residual so the packaged reports show a realistic
  Unexplained review case with plausible but non-additive evidence.

The corresponding YAML files define transaction semantics. Standard field roles
distinguish performance inputs, related input components, reported performance
checks, and context rows.
