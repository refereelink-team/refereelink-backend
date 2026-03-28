# Repository Guidelines

## Project Structure & Module Organization
`sports/` contains the reusable Python package. Keep domain logic in focused modules: `sports/annotators/` for drawing and visualization helpers, `sports/common/` for shared tracking and classification utilities, and `sports/configs/` for sport-specific configuration objects. `examples/soccer/` is the runnable demo, with `main.py` as the CLI entry point, `setup.sh` for downloading weights and sample videos, and `notebooks/` for model-training workflows. There is no committed `tests/` directory yet; add new tests under `tests/`, mirroring package paths.

## Build, Test, and Development Commands
Install the package in editable mode from the repository root:

```bash
python -m pip install -e .
python -m pip install -e ".[tests]"
```

Run the current test suite with:

```bash
pytest
```

Set up the soccer demo assets with:

```bash
pip install -r examples/soccer/requirements.txt
cd examples/soccer && ./setup.sh
```

Run the demo locally, for example:

```bash
python examples/soccer/main.py --source_video_path examples/soccer/data/2e57b9_0.mp4 --target_video_path out.mp4 --device cpu --mode PLAYER_DETECTION
```

## Coding Style & Naming Conventions
Target Python 3.8+. Use 4-space indentation, `snake_case` for modules, functions, and variables, `PascalCase` for classes, and `UPPER_CASE` for constants such as model paths and class IDs. Follow the existing style: explicit imports, type hints on reusable helpers, and short docstrings for non-trivial functions. No formatter or linter config is checked in, so keep changes consistent with surrounding code and avoid unrelated reformatting.

## Testing Guidelines
Use `pytest` for new tests. Name files `test_*.py` and keep fixtures close to the behavior under test. Prioritize tests for geometry transforms, tracking/classification utilities, and config invariants; these are more stable than video-output snapshots. If you add CLI behavior, cover argument handling and failure paths.

## Commit & Pull Request Guidelines
Recent commits use short, direct subjects such as `ball annotator added` and `ready for tests`. Keep commit messages concise, imperative, and specific to one change. PRs should include a summary, validation steps, linked issues when relevant, and screenshots or short clips for annotation/video-output changes.

## Data & Asset Handling
`examples/soccer/setup.sh` downloads weights and videos into `examples/soccer/data/`. Do not commit downloaded binaries, large media, or machine-specific paths. Keep secrets and API keys out of the repository.
