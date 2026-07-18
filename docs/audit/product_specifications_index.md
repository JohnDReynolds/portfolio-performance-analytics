# PPAR Audit Product Specifications Index

| Document field | Value |
| --- | --- |
| Status | Current index to retained founder-approved product-design detail |
| Version | 2.1 |
| Date | 2026-07-18 |
| Governing product document | [PPAR Audit Product Constitution](product_constitution.md) |
| Product roadmap | [PPAR Audit Roadmap](roadmap.md) |
| Active implementation plan | [PPAR Audit MVP Completion Plan](mvp_plan.md) |
| Retained detailed source | [Foundational Product Design v0.10](archive/PPAR_Audit_Foundational_Product_Design_v0.10.md) |
| Purpose | Locate approved historical requirements without duplicating the archived 9,000-line design corpus |

## How to Use This Index

The retained v0.10 document records the founder-approved Phase 2 and Phase 3
product-design exercise, requirement identifiers, evidence intake, and design
rationale. It is preserved for provenance and detailed research. It is not a
description of current executable behavior and is not the active roadmap.

Use this index when a current decision or implementation task requires the
approved historical detail behind a product boundary. Read only the relevant
linked section. Do not treat adjacent historical plans, speculative feature
detail, or old phase gates as current work.

When sources disagree, use:

1. current executable behavior, tests, generated artifacts, and machine-readable
   contracts;
2. the current safety-invariant catalog and maintainer contract;
3. current user/setup documentation;
4. the Product Constitution, product roadmap, and active MVP plan;
5. current sections of the technical design reference; and
6. the approved historical detail linked here.

The archive remains intentionally unchanged. Current corrections belong in the
owning current document rather than in the historical snapshot.

## Approved Detail Map

| Area | Retained detail | Current interpretation and owner |
| --- | --- | --- |
| Users, roles, decision rights, and conceptual workflow | [Phase 2](archive/PPAR_Audit_Foundational_Product_Design_v0.10.md#17-phase-2--users-workflows-and-conceptual-product-architecture) | Approved product hypotheses and human-authority boundaries. Real-client workflow remains unvalidated, and managed workflow is not an active feature. Current product stance belongs in the constitution. |
| Performance Change Investigation | [Phase 3A](archive/PPAR_Audit_Foundational_Product_Design_v0.10.md#18-phase-3a--performance-change-investigation) | Requirements `PCI-001` through `PCI-067`. Current arithmetic, evidence, lineage, and report behavior are governed by code, tests, safety invariants, and the technical design. |
| Performance Data Quality Audit | [Phase 3B](archive/PPAR_Audit_Foundational_Product_Design_v0.10.md#19-phase-3b--performance-data-quality-audit) | Requirements `PDQ-001` through `PDQ-070`. Current checks and YAML behavior are governed by executable contracts. Additional MVP issue types are governed by the active MVP plan. |
| Audit Readiness | [Phase 3C](archive/PPAR_Audit_Foundational_Product_Design_v0.10.md#20-phase-3c--audit-readiness) | Requirements `ARD-001` through `ARD-092`. Technical readiness foundations exist, but a broader user-facing readiness product requires evidence and prioritization. |
| Executive Investigation Summary | [Phase 3D](archive/PPAR_Audit_Foundational_Product_Design_v0.10.md#21-phase-3d--executive-investigation-summary) | Requirements `EIS-001` through `EIS-085`. The bounded Executive Summary is an active MVP gap; the MVP plan governs its implementation scope. |
| Audit Health Dashboard and Operational Intelligence | [Phase 3E](archive/PPAR_Audit_Foundational_Product_Design_v0.10.md#22-phase-3e--audit-health-dashboard-and-operational-intelligence) | Requirements `AHD-001` through `AHD-076`. Direction only, evidence-gated, and excluded from MVP completion. No history/dashboard implementation is authorized by the retained design. |
| Human Review and Disposition | [Phase 3F](archive/PPAR_Audit_Foundational_Product_Design_v0.10.md#23-phase-3f--human-review-and-disposition-boundary) | Boundary only. Workflow schemas, assignments, state, permissions, storage, notifications, and integrations remain intentionally unspecified and outside the active MVP. |

## Supporting Historical Material

| Material | Retained section | Use |
| --- | --- | --- |
| Source classification and authority notes | [Appendix A](archive/PPAR_Audit_Foundational_Product_Design_v0.10.md#appendix-a--source-register) | Understand which repository and external sources informed the product-design exercise. Recheck current files before relying on an old observation. |
| Representative workbook observations | [Appendix B](archive/PPAR_Audit_Foundational_Product_Design_v0.10.md#appendix-b--representative-workbook-observations) | Historical evidence about the reviewed workbook checkpoint, not the current workbook contract. |
| Product-expansion inventory | [Appendix C](archive/PPAR_Audit_Foundational_Product_Design_v0.10.md#appendix-c--incorporated-product-expansion-inventory) | Idea provenance only. It does not authorize implementation. |
| Former phase outline | [Appendix D](archive/PPAR_Audit_Foundational_Product_Design_v0.10.md#appendix-d--planned-foundational-design-sections) | Historical planning record. It is not the current roadmap. |
| Implementation-document intake | [Appendix E](archive/PPAR_Audit_Foundational_Product_Design_v0.10.md#appendix-e--additional-implementation-document-intake) | Historical discrepancy and evidence notes. Verify all current behavior independently. |
| External evidence | [Appendix F](archive/PPAR_Audit_Foundational_Product_Design_v0.10.md#appendix-f--external-evidence) | Context for roles, governance, methodology, and market framing; not evidence of a PPAR capability. |

## Current Owning Documents

| Subject | Current owner |
| --- | --- |
| Product identity, principles, boundaries, claims, and founder decisions | [Product Constitution](product_constitution.md) |
| Product stages, evidence gates, priorities, and open questions | [Audit Roadmap](roadmap.md) |
| Active MVP scope, sequence, and acceptance gates | [MVP Completion Plan](mvp_plan.md) |
| Comparison-engine contracts and implementation rationale | [Performance Comparison Design](performance_comparison_design.md) |
| Data Issues contracts and configuration | [Data Issues Design](data_issues_design.md) |
| Safety guarantees and failure classifications | [Audit Safety Invariants](safety_invariants.md) |
| Packaged-demo source boundary | [Demo Source Contract](demo_source_contract.md) |
| Site-specific extract review | [Site Extract Readiness Checklist](site_extract_readiness_checklist.md) |
| Transaction semantics | [Machine-readable matrix](../axys_apx/contracts/transaction_semantics_matrix.yaml) and [rendered contract](../axys_apx/contracts/transaction_semantics_matrix.md) |
| Historical documentation | [Audit archive](archive/README.md) |

## Maintenance Rule

Keep this file as an index. Do not copy detailed specifications back into it.
Add or revise an entry only when the location, current interpretation, or owning
document for an approved requirement changes.

If future evidence changes a product requirement, record the decision in the
current owning document and link it here. Do not rewrite the archived v0.10
source to make history look current.
