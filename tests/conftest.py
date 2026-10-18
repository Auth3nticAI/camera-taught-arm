"""Isolate every test from the real data folder (calibration.json, sim_real_map.json, sessions).

Without this, tests that construct FollowerArm/SimRealMap read the user's real sim_real_map.json,
so a real alignment file changed test results (seen 2026-10-17). Models stay at the real location.
"""
import pytest

from cta import paths

_REAL_MODELS = None


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    global _REAL_MODELS
    if _REAL_MODELS is None:
        _REAL_MODELS = str(paths.models_dir())
    monkeypatch.setenv("CTA_MODELS_DIR", _REAL_MODELS)
    monkeypatch.setenv("CTA_DATA_DIR", str(tmp_path / "cta-data"))
    yield
