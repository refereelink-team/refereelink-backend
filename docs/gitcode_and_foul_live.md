# Git remotes & live foul alerts

## Remotes

- **Primary (only push target):** `origin` → `https://gitcode.com/linshengyin/SC.git`
- Do **not** push to GitHub for this workflow.
- Development branch: `dev` (created from gitcode `main`).

If `git push origin dev` returns HTTP 403 with *“image repository”*, the gitcode
project is currently mirror/read-only for pushes. Keep committing on local
`dev` and retry push once write access is enabled on gitcode.

Clone / sync:

```bash
git clone https://gitcode.com/linshengyin/SC.git
cd SC
git checkout main
git pull origin main
git checkout -b dev   # or: git checkout dev && git pull
```

## Live foul alerts (wide-angle)

Enable from the referee console **Foul Detect** toggle or:

```bash
uv run python -m app.server.main --enable_foul_detection ...
```

Pipeline behaviour:

1. **Geometry contact** (main thread): opposite-team proximity + closing speed →
   `foul_candidate` with identity `Home T{id}` / `Away T{id}`.
2. **Async MVFoul worker** (optional): classifies a short ROI clip off the critical
   path. Requires `assets/weights/mvfoul.pth.tar` and `fouls_far`/`offside` on
   `PYTHONPATH`. Without them, geometry candidates still appear (`evidence.source=geometry`).

Dashboard: **EVENTS & ALERTS** shows a readable summary; **2D PITCH** highlights
involved track IDs; status cards show foul queue / classify latency.
