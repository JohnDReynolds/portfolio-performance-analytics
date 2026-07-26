# Transaction Policy Ownership

PPAR resolves transaction meaning once. Client configuration owns local
economic interpretation; downstream Audit features consume the resulting
normalized semantics instead of maintaining their own transaction-code
dictionaries.

## Authorities

### Client transaction rules

`transaction_rules` in `ppar.yaml` is the client-facing authority for local
transaction meaning. A matching rule can assign:

- `transaction_category`;
- `cash_flow_sign`; and
- `performance_flow_sign`.

Transaction-code keys and `when` conditions preserve exact source case. A
complete matching rule is authoritative over source-supplied semantic fields.
Unknown codes, unmatched contextual rules, and incomplete meanings remain
fail-closed.

### Normalized transaction rows

`TransactionsLoader` is the single runtime resolution boundary. It applies
source mappings and `transaction_rules`, validates the result, and records the
semantic provenance. Performance Comparison, reconstruction, reviewer
guidance, and Data Issues consume these normalized rows.

Data Issues may use:

- normalized semantics when the check concerns economic meaning; or
- an explicit `only` population when the check concerns a narrower local
  transaction family, such as dividends or purchase/sale accrued interest.

An explicit population selects rows; it does not create transaction meaning.
The selected rows must still resolve through `transaction_rules` or recognized
source semantics.

### Axys/APX safety policy

Product code retains only the small Axys/APX fail-closed boundary that cannot
be expressed as local economic meaning: ambiguous flow-like codes require the
configured source context before a YAML rule may classify them. Disabling this
guard remains an explicit local extract-contract decision.

The safety policy does not define buy, sell, income, fee, transfer, or external
flow meanings.

## Research and Test Evidence

The transaction-semantics matrix under `docs/axys_apx/contracts/` records
research, fixture coverage, confidence, and unresolved cases. It is evidence
for maintainers and tests, not a runtime dependency and not a transaction-code
dictionary installed clients must trust.

Maintainer-only fixture gates and code-family registries belong under
`scripts/` or `tests/`. They may validate demonstration coverage, but product
modules must not import them.

## Protected Behavior

Changes to transaction policy must preserve:

- exact-case transaction-rule matching;
- conditional context matching;
- ambiguous-flow fail-closed behavior;
- unknown-code failure;
- normalized category and sign validation;
- Modified Dietz flow and income treatment;
- transaction-semantics provenance;
- Data Issues independence from additive performance explanations; and
- existing output schemas.

The packaged Audit demonstration is the parity reference for financial results,
finding classifications, and output schemas. Research coverage may grow without
expanding runtime authority.
