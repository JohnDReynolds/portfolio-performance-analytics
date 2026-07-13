# Documentation Images

This directory contains durable image assets used by repository documentation.

| Path | Contents | Maintenance |
| --- | --- | --- |
| `readme/` | Generated PNG/JPG assets embedded in the root `README.md`. | Regenerate with `./.venv/bin/python scripts/render_readme_images.py`. |

Regenerate the root `PPAR.pdf` from the current root `README.md` with:

```bash
./.venv/bin/python scripts/render_readme_pdf.py
```

The release-candidate workflow can refresh and validate both sets of assets with
`./.venv/bin/python scripts/check_release_candidate.py --refresh-images`.

Generated demo/report outputs belong under `_demo_output/`, not here.
