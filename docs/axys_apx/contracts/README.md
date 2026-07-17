# Implementation Contracts

This folder contains implementation-facing contracts for ppar demo, validation,
and test behavior.

Use these files when you need structured behavior rules or generated
availability summaries. Use the `../reference/Chapter_*.md` files for normal
reader-facing explanation and the `../evidence/` folder for provenance.

The contracts should stay conservative. They may operationalize the local
Axys/APX reference, but they should not imply official vendor methodology or a
complete vendor schema.

## Contract Index

| Contract | Use it when |
|---|---|
| [transaction_semantics_matrix.md](transaction_semantics_matrix.md) | You need the narrative rules behind local transaction-code classification. |
| [transaction_semantics_matrix.yaml](transaction_semantics_matrix.yaml) | You need machine-readable transaction semantics for validation or tests. |
| [demo_extract_availability.md](demo_extract_availability.md) | You need PPAR's generated extract requirements, likely source paths, and packaged-demo column availability. |
| [templates/site_extract_contract.yaml](templates/site_extract_contract.yaml) | You need a broad site-level source-data contract starting point. |
| [templates/site_extract_contract_imex_context.yaml](templates/site_extract_contract_imex_context.yaml) | You need an IMEX-focused site-contract example. |
| [templates/site_extract_contract_rep_semantics.yaml](templates/site_extract_contract_rep_semantics.yaml) | You need a REP/report-focused site-contract example. |

## Update Rules

- Update contracts only when a reference-chapter conclusion has a demo,
  validation, test, or user-facing extract implication.
- Keep generated files generated. Update the source-data input or renderer first,
  then regenerate.
- After changing `transaction_semantics_matrix.yaml`, run
  `./.venv/bin/python scripts/render_transaction_semantics_matrix.py` and commit
  the regenerated row table.
- Keep site-contract templates vendor-aware but not vendor-guaranteed; they are
  examples for local source-data negotiation.
