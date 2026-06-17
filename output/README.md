# Output Directory

This folder is organized by purpose:

- `final/`: final V2.9 homework HTML/Markdown report, execution rules, charts, metrics, NAV, and attribution tables.
- `research_history/`: readable Markdown reports from earlier strategy iterations.
- `intermediate/`: generated CSV evidence used during research. Most large intermediate CSV files are ignored by Git; two compact summary files are kept because the final report generator uses them for comparison and stress-test tables.
- `intermediate/large/`: bulky NAV and overlay-weight files kept locally for audit, not intended for normal commits.

To regenerate the final deliverables, including the HTML report and charts:

```bash
python scripts/final_v29_report.py
```
