# PPAR Audit Documentation

PPAR Audit documentation is organized by authority and maintenance lifecycle.
Start with the product constitution for stable product authority, the roadmap
for forward-looking direction, and the MVP plan for the work currently in
scope. Historical design material is retained under [`archive/`](archive/) and
is not current implementation authority.

## Start Here

| Need | Document | Authority |
| --- | --- | --- |
| Understand product identity, doctrine, boundaries, claims, and founder decisions | [Product Constitution](product_constitution.md) | Current Audit product and capability-scope authority |
| Understand product stages, evidence gates, priorities, and open questions | [Audit Roadmap](roadmap.md) | Current Audit product-direction authority within the constitution's boundaries |
| Understand active MVP sequence, slice status, and acceptance gates | [MVP Completion Plan](mvp_plan.md) | Current Audit implementation-plan authority within the constitution's scope |
| Locate approved historical requirements | [Product Specifications Index](product_specifications_index.md) | Index to retained founder-approved detail; not executable truth |
| Understand the comparison engine | [Performance Comparison Design](performance_comparison_design.md) | Current technical reference, subordinate to executable behavior |
| Understand source-data integrity checks | [Data Issues Design](data_issues_design.md) | Current Data Issues technical reference |
| Understand non-negotiable safety behavior | [Audit Safety Invariants](safety_invariants.md) | Maintainer-facing safety contract |

## Implementation and Operator References

| Document | Purpose |
| --- | --- |
| [Demo Source Contract](demo_source_contract.md) | Defines the packaged Audit demo source and fixture boundary. |
| [Site Extract Readiness Checklist](site_extract_readiness_checklist.md) | Guides site-specific Axys/APX source-contract review before report generation. |

The machine-readable transaction authority is
[`../axys_apx/contracts/transaction_semantics_matrix.yaml`](../axys_apx/contracts/transaction_semantics_matrix.yaml).
Its maintained human-readable companion is
[`../axys_apx/contracts/transaction_semantics_matrix.md`](../axys_apx/contracts/transaction_semantics_matrix.md).

## Historical Material

The [`archive/`](archive/) directory contains migration snapshots, completed
working prompts, checkpoint notes, and process records. Archived documents may
explain why a decision was made, but they must not be used to infer current
product behavior or current priorities.

The long historical product-design corpus is retained once in
[`archive/PPAR_Audit_Foundational_Product_Design_v0.10.md`](archive/PPAR_Audit_Foundational_Product_Design_v0.10.md).
The top-level Product Specifications Index points to the relevant sections
without copying that corpus.

## Authority Order

When documents disagree, use this order:

1. Current executable behavior, tests, generated artifacts, and machine-readable
   contracts
2. Current safety-invariant catalog and maintainer contract
3. Current setup and user documentation
4. Audit Product Constitution for product scope, boundaries, and founder
   decisions
5. Audit roadmap for stages, evidence gates, priorities, and open questions
6. Active MVP plan for implementation sequence, detailed acceptance, and status
7. Current technical reference sections
8. Product Specifications Index and linked approved historical requirements
9. Archived snapshots, checkpoints, prompts, and brainstorming material

Product direction does not turn a proposed capability into implemented behavior,
and historical implementation text does not override current contracts.

## Maintenance Rules

- Add a new document only when it has a distinct audience, authority, and
  maintenance lifecycle.
- Link to product doctrine and contracts instead of copying them.
- Put product doctrine and founder decisions in the constitution, forward-looking
  direction in the roadmap, and active work in the MVP plan.
- Update the constitution, roadmap, and active plan in the same commit whenever
  the number or identity of MVP capabilities changes; neither the roadmap nor
  the plan may expand the constitution's capability boundary by itself.
- Put executable policy in code or machine-readable contracts and explain it in
  the nearest owning technical document.
- Move completed prompts, checkpoints, and superseded plans to `archive/`.
- Keep archived documents frozen apart from an archival banner or repaired link.
- Replace the MVP plan when its gate is complete; do not let it become a
  cumulative implementation journal.
- Check local links whenever files are moved or renamed.
