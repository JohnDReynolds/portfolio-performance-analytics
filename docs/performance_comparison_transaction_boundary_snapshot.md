# Performance Comparison Transaction Boundary Snapshot

This snapshot summarizes the current transaction-semantics coverage boundary.
It is a review aid for the machine-readable matrix in
[`axys-apx-reference/contracts/transaction_semantics_matrix.yaml`](axys-apx-reference/contracts/transaction_semantics_matrix.yaml);
the matrix remains the implementation contract.

## Covered Formula Inputs

| Family | Codes | Boundary |
| --- | --- | --- |
| Packaged formula rows | `by`, `sl`, `dv`, `in` | Covered by packaged demo data and tests. |
| Fixed-income safe row | `in` | Ordinary interest can be performance income when configured and evidenced. |

## Context-Required Rows

| Family | Codes | Boundary |
| --- | --- | --- |
| Ambiguous Axys flows | `li`, `lo`, `dp`, `wd` | Classification requires IMEX context, REP/report semantics, or an explicit reviewed local opt-out. |

## Review-Only And Context-Only Rows

| Family | Codes | Boundary |
| --- | --- | --- |
| Review-only test rows | `;` | Covered only as neutral corporate-action evidence in the test-only `review_only_actions` fixture. |
| Context-only token | `exus` | Covered as special-security context for fee-like `dp`; standalone transaction-code treatment is unproven. |

## Backlog Gates

| Family | Codes | Boundary |
| --- | --- | --- |
| Fixed-income backlog | `ai`, `pa`, `sa`, `pd` | Needs bond/accrual context, cash offset, amount sign, principal or quantity movement where applicable, and local mapping or REP/report semantics. |
| Capital-return backlog | `rc`, `pd` | Needs policy evidence before choosing performance income, corporate-action evidence, or review-only evidence. |
| Short-side backlog | `ss`, `cs` | Needs short security type, cash/margin/short-account symbols, amount and quantity signs, and local mapping or REP/report semantics. |
| Standalone fee-token backlog | `epus` | Needs evidence that the token appears as a standalone transaction code rather than label/context. |

## Release-Readiness Notes

- The packaged demo should remain focused on realistic, internally consistent
  formula-input examples.
- Test-only fixtures carry ambiguous, local-policy, review-only, and blocked
  behavior until a real reviewer story justifies packaged demo coverage.
- The demo matrix validator checks fixture coverage and the high-risk backlog
  gates before release.
