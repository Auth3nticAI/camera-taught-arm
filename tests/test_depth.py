"""Depth Anything V2 wrapper (cta.depth): robust patch sampling + the real model (skipped if it can't load)."""
import numpy as np
import pytest

from cta.depth import DEPTH_MODEL_ID, DepthEstimator, sample_patch


# ---- sample_patch: median over a small patch, robust to NaN / edges ---------------------------------

def test_patch_median_ignores_outliers():
    d = np.full((100, 200), 2.0)
    d[50, 100] = 50.0  # one wild pixel in the patch
    assert sample_patch(d, (0.5, 0.5), radius_frac=0.03) == pytest.approx(2.0)


def test_patch_samples_the_right_place():
    d = np.zeros((100, 200))
    d[:, 100:] = 3.0  # right half farther
    assert sample_patch(d, (0.8, 0.5)) == pytest.approx(3.0)
    assert sample_patch(d, (0.2, 0.5)) == pytest.approx(0.0)


def test_patch_ignores_nan():
    d = np.full((100, 200), 1.5)
    d[45:55, 95:105] = np.nan
    d[50, 107] = 1.5
    assert sample_patch(d, (0.5, 0.5), radius_frac=0.06) == pytest.approx(1.5)
    d[:] = np.nan
    assert np.isnan(sample_patch(d, (0.5, 0.5)))


@pytest.mark.parametrize("uv", [(0.0, 0.0), (1.0, 1.0), (0.999, 0.001), (0.0, 0.5)])
def test_patch_at_the_edges_is_clipped_not_crashing(uv):
    d = np.full((100, 200), 2.5)
    assert sample_patch(d, uv) == pytest.approx(2.5)


@pytest.mark.parametrize("uv", [(-0.2, 0.5), (0.5, 1.3), (np.nan, 0.5), None])
def test_patch_outside_the_image_is_nan(uv):
    assert np.isnan(sample_patch(np.full((10, 10), 1.0), uv))


def test_patch_none_map_is_nan():
    assert np.isnan(sample_patch(None, (0.5, 0.5)))


# ---- DepthEstimator: lazy, clear fallback -------------------------------------------------------------

def test_estimator_is_lazy_and_reports_a_failed_load():
    est = DepthEstimator(model_id="this-org-does-not-exist/nope-model", local_only=True)
    assert est._model is None  # nothing loaded at construction
    assert est.load() is False and est.error  # a clear reason, no exception
    assert est.predict(np.zeros((72, 128, 3), np.uint8)) is None


@pytest.fixture(scope="module")
def real():
    est = DepthEstimator()
    if not est.load():
        pytest.skip(f"Depth Anything V2 not loadable here: {est.error}")
    return est


def test_real_model_returns_metric_depth(real):
    assert real.model_id == DEPTH_MODEL_ID
    rng = np.random.default_rng(0)
    rgb = (rng.uniform(0, 255, (360, 640, 3))).astype(np.uint8)
    d = real.predict(rgb)
    assert d.ndim == 2 and d.dtype == np.float32
    assert abs(d.shape[1] / d.shape[0] - 640 / 360) < 0.1  # aspect kept
    assert np.all(np.isfinite(d)) and 0.0 <= float(d.min()) and float(d.max()) <= 25.0  # metres (indoor max 20)
