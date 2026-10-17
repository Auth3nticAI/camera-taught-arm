"""Download the MediaPipe hand landmarker model into models_dir()."""
import urllib.request

from cta.paths import models_dir

URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
       "hand_landmarker/float16/latest/hand_landmarker.task")


def main() -> None:
    dest = models_dir() / "hand_landmarker.task"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"already present: {dest}")
        return
    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, dest)
    print(f"saved {dest} ({dest.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
