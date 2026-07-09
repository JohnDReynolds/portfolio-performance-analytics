# Documentation Images

This directory contains durable image assets used by repository documentation.

| Path | Contents | Maintenance |
| --- | --- | --- |
| `readme/` | Generated PNG/JPG assets embedded in the root `README.md`. | Regenerate with `./.venv/bin/python scripts/render_readme_images.py`. |
| `performance_comparison/` | Performance Auditing workflow and report-preview SVGs. | Update manually when the documented report workflow changes. |

Generated demo/report outputs belong under `_demo_output/`, not here.
