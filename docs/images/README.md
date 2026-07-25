# Documentation Images

This directory contains durable image assets used by repository documentation.

| Path | Contents | Maintenance |
| --- | --- | --- |
| `readme/` | Generated Audit and Analytics PNG/JPG assets embedded in the root and product READMEs. | Regenerate with `./.venv/bin/python scripts/render_readme_images.py`. |

Regenerate the root `PPAR.pdf` from the current root `README.md` with:

```bash
./.venv/bin/python scripts/render_readme_pdf.py
```

The release-preparation command automatically regenerates `PPAR.pdf` before it
builds distributable artifacts:

```bash
./.venv/bin/python scripts/check_release_candidate.py --build
```

Use `--refresh-images` when the README PNG/JPG assets also need regeneration. That
option refreshes the images and then rebuilds `PPAR.pdf` from the current README.

Generated demo/report outputs belong under `_demo_output/`, not here.
