"""Sync sessions/models with private Hugging Face Hub repos.

Usage: python scripts/sync_data.py push|pull [--what sessions|models|config|all]
`config` = calibration.json + sim_real_map.json (data_dir) <-> cta-models repo under config/.
Requires `uv run hf auth login` once per machine.
"""
import argparse
import shutil
import tempfile
from pathlib import Path

from cta.paths import data_dir, models_dir

# what -> (repo suffix, repo_type, local dir getter)
TARGETS = {
    "sessions": ("cta-sessions", "dataset", lambda: data_dir() / "sessions"),
    "models": ("cta-models", "model", models_dir),
    "config": ("cta-models", "model", data_dir),  # only CONFIG_FILES, under config/ in the repo
}
CONFIG_FILES = ["calibration.json", "sim_real_map.json"]


def _sync_config(api, direction: str, repo_id: str, repo_type: str, path: Path) -> None:
    from huggingface_hub import snapshot_download

    if direction == "push":
        api.create_repo(repo_id, repo_type=repo_type, private=True, exist_ok=True)
        api.upload_folder(repo_id=repo_id, repo_type=repo_type, folder_path=str(path),
                          path_in_repo="config", allow_patterns=CONFIG_FILES)
        return
    with tempfile.TemporaryDirectory() as tmp:
        snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=tmp,
                          allow_patterns=[f"config/{f}" for f in CONFIG_FILES])
        for f in CONFIG_FILES:
            src = Path(tmp) / "config" / f
            if src.exists():
                shutil.copy2(src, path / f)
                print(f"  pulled {f} -> {path / f}")


def sync(direction: str, what: str) -> None:
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    user = api.whoami()["name"]
    names = list(TARGETS) if what == "all" else [what]
    for name in names:
        suffix, repo_type, local = TARGETS[name]
        repo_id = f"{user}/{suffix}"
        path = local()
        path.mkdir(parents=True, exist_ok=True)
        if name == "config":
            _sync_config(api, direction, repo_id, repo_type, path)
        elif direction == "push":
            api.create_repo(repo_id, repo_type=repo_type, private=True, exist_ok=True)
            api.upload_folder(repo_id=repo_id, repo_type=repo_type, folder_path=str(path))
        else:
            snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=str(path))
        print(f"{direction} {name}: {repo_id} <-> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("direction", choices=["push", "pull"])
    ap.add_argument("--what", choices=[*TARGETS, "all"], default="all")
    args = ap.parse_args()
    sync(args.direction, args.what)


if __name__ == "__main__":
    main()
