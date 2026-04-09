Synthetic SS&C Axys IMEX-style demo files
========================================

These are synthetic CSV files designed to be IMEX-like.
They are not guaranteed to match the exact column names or export options in a specific Axys installation.

Files:
- imex_secperf_port_small.csv
- imex_portperf_port_small.csv
- imex_secperf_port_large.csv
- imex_portperf_port_large.csv
- imex_security_master.csv
- imex_classification_lookup.csv
- imex_classification_hierarchy.csv

Coverage:
- 24 contiguous months spanning 2024-01 through 2025-12
- Some months are one monthly period; some are split into 2 or 3 contiguous subperiods
- For each portfolio, secperf and portperf have identical time periods
- Some periods have exact equality between SUM(BEGIN_WEIGHT * SEC_RETURN) and PORT_RETURN
- Other periods have a slight difference
