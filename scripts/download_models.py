"""Download the MediaPipe hand and pose landmarker models into models_dir()."""
import urllib.request

from cta.paths import models_dir

MODELS = {
    "hand_landmarker.task": ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                             "hand_landmarker/float16/latest/hand_landmarker.task"),
    "pose_landmarker_full.task": ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                                  "pose_landmarker_full/float16/latest/pose_landmarker_full.task"),
}


def main() -> None:
    for name, url in MODELS.items():
        dest = models_dir() / name
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"already present: {dest}")
            continue
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, dest)
        print(f"saved {dest} ({dest.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
