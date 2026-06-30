# Axys Site Variant Fixtures

These fixtures model small site-specific extract shapes used by contract tests.
They are not native Axys schemas. Each variant focuses on ambiguous Axys
transaction codes whose performance treatment cannot be trusted from code alone.

| Variant | Purpose |
| --- | --- |
| `imex_context` | IMEX-style transaction rows include source/destination and special-security context, so conditional YAML can classify `li`, `lo`, `dp`, and `wd`. |
| `rep_semantics` | REP/report-style rows carry already-reviewed category and sign semantics, so a site contract can use those fields as the blocking classification context. |
| `imex_code_only` | Code-only IMEX-style rows intentionally omit the required context and must fail before YAML can classify ambiguous external flows. |

The `imex_context` and `rep_semantics` contract files intentionally match the
documented onboarding profiles in
`docs/axys-apx-reference/templates/site_extract_contract_imex_context.yaml` and
`docs/axys-apx-reference/templates/site_extract_contract_rep_semantics.yaml`.
