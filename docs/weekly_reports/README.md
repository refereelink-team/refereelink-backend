# Weekly Development Report Artifact

This directory contains the generated weekly report Artifact for the Soccer
Analysis project. The report is a self-contained, print-friendly HTML document
with a restrained broadsheet editorial layout, inline SVG diagrams, Git-derived
metrics, commit timeline, working-tree notes, and visual evidence.

Generate a report locally:

```bash
python tools/generate_weekly_report.py
```

The command writes both a dated snapshot (`report-YYYY-MM-DD.html`) and the
stable Artifact entry point (`latest.html`). The scheduled Saturday run uses
the same command so the report remains reproducible from the repository.

The generator deliberately excludes `docs/weekly_reports/` from the working
tree section. That keeps the report's own output from appearing as a new
development change in the next run.
