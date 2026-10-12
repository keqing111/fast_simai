# Report data snapshot

These JSON files are sanitized copies of the v3 campaign aggregates used by
`REPORT.md` and `tools/generate_report_figures.py`. The machine-local `attempt`
path is intentionally removed. Raw simulator attempts remain under the ignored
`results/` tree because they occupy roughly 698 MiB.

Regenerate the snapshot and figures from a completed local campaign with:

```bash
PYTHONPATH=src python3 tools/export_report_data.py
PYTHONPATH=src python3 tools/generate_report_figures.py
```
