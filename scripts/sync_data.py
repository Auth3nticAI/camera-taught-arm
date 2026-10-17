"""Sync sessions/models with private Hugging Face Hub repos.

Usage: python scripts/sync_data.py push|pull [--what sessions|models|all]
Requires `uv run hf auth login` once per machine.
"""
import argparse

from cta.paths import data_dir, models_dir

# what -> (repo suffix, repo_type, local dir getter)
TARGETS = {
    "sessions": ("cta-sessions", "dataset", lambda: data_dir() / "sessions"),
    "models": ("cta-models", "model", models_dir),
}


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
        if direction == "push":
            api.create_repo(repo_id, repo_type=repo_type, private=True, exist_ok=True)
            api.upload_folder(repo_id=repo_id, repo_type=repo_type, folder_path=str(path))
        else:
            snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=str(path))
        print(f"{direction} {name}: {repo_id} <-> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("direction", choices=["push", "pull"])
    ap.add_argument("--what", choices=["sessions", "models", "all"], default="all")
    args = ap.parse_args()
    sync(args.direction, args.what)


if __name__ == "__main__":
    main()
