# Performance Test Fixtures

These CSV files are test-only analytics fixtures. They intentionally stay under
`tests/data/performance` instead of `ppar/demos/data/performance` because they
anchor regression values, frequency-consolidation behavior, validation errors,
or oversized-output checks.

Use packaged demo performance data in `ppar/demos/data/performance` for
user-facing examples. Add files here only when the test needs a narrow fixture
that would be distracting or misleading as demo data.

## Fixture Roles

- `Big 2.csv` and `big2_daily.csv`: Monthly/daily two-security data for
  frequency-consolidation and data-source format tests.
- `Large-Cap Portfolio.csv`, `Mega-Cap Portfolio.csv`, `Magnificent 7.csv`,
  `mag7_daily.csv`, and `economic_sector_daily.csv`: Larger regression and
  audit fixtures with expected output baselines.
- `abcde_*.csv`: Small exact-value attribution regression fixtures.
- `case_*.csv`: Focused frequency, date-window, and short-position edge cases.
