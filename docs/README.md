# PPAR Documentation

PPAR has two market-facing products that share one codebase and common integration
infrastructure.

| Area | Location | Purpose |
| --- | --- | --- |
| PPAR Audit | [`audit/`](audit/) | Audit product planning, source-data contracts, safety invariants, evidence review, and operational demo maintenance. |
| PPAR Analytics | [`analytics/`](analytics/) | Analytics product maintenance and demo-refresh guidance. |
| Axys/APX integration | [`axys_apx/`](axys_apx/) | Shared Axys/APX research, evidence, contracts, reference material, and common-core export guidance. |
| Shared platform | This directory | Architecture, repository orientation, and the portfolio-level roadmap. |
| Documentation assets | [`images/`](images/) | Generated images and product-overview maintenance guidance. |

Start with:

- [`architecture.md`](architecture.md) for the shared system boundary.
- [`repository_guide.md`](repository_guide.md) for maintainer workflows.
- [`roadmap.md`](roadmap.md) for cross-product direction and implementation history.
- [`audit/PPAR_Audit_Foundational_Product_Design.md`](audit/PPAR_Audit_Foundational_Product_Design.md)
  for the developing Audit product plan.
- [`analytics/analytics_demo_refresh.md`](analytics/analytics_demo_refresh.md)
  for Analytics demo and README-asset maintenance.

New product-specific documentation should go directly into the corresponding
product directory. Shared architecture and integration material should remain outside
the product directories. Historical working notes belong under the relevant
product's `archive/` directory.
