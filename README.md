# camera-taught-arm

Webcam hand tracking (MediaPipe) drives an SO-101 robot arm (MuJoCo sim, then real hardware via LeRobot). PyTorch class project.

## Setup
```
uv sync
uv run python scripts/download_models.py
uv run hf auth login
```
Data and models live in `CTA_DATA_DIR` (default `C:\Users\trayd\dev\cta-data`), synced with `uv run python scripts/sync_data.py push|pull`.

## Test
```
uv run pytest -v
```

## Docs
- Specs: [docs/specs](docs/specs)
- Plans: [docs/plans](docs/plans)
