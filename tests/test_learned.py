"""--learned: the trained network drives the arm's 5 joints; the pinch rule keeps the gripper (D-V)."""
import json

import numpy as np
import pytest
import torch

from cta.learned import LearnedDriver, blend_gripper
from cta.learning import train_head


def _run(tmp_path, input_mode='hand', dim=4):
    rng = np.random.default_rng(0)
    takes = {f't{i}': {'features': rng.normal(size=(16, dim)).astype('float32'),
                       'targets': rng.normal(size=(16, 6)).astype('float32')} for i in range(4)}
    m = train_head(takes, tmp_path / 'run', epochs=2)
    m['config']['input'] = input_mode
    (tmp_path / 'run' / 'metrics.json').write_text(json.dumps(m))
    return tmp_path / 'run'


class FakeEncoder:
    def __init__(self, dim=4):
        self.dim, self.calls = dim, []

    def __call__(self, images):
        self.calls.append([img.copy() for img in images])
        return np.stack([np.full(self.dim, float(img.mean()) / 255, np.float32) for img in images])


def _lm():
    lm = np.zeros((21, 3))
    lm[:, 0], lm[:, 1] = np.linspace(0.4, 0.6, 21), np.linspace(0.4, 0.6, 21)
    return lm


def test_driver_runs_the_trained_head_on_the_hand_crop(tmp_path):
    enc = FakeEncoder()
    d = LearnedDriver(_run(tmp_path), encode=enc, smooth=False)
    rgb = np.zeros((90, 160, 3), np.uint8)
    rgb[40:50, 75:85] = 255
    q = d.predict(rgb, _lm(), 0.0)
    assert q.shape == (6,) and np.isfinite(q).all()
    assert len(enc.calls[-1]) == 1 and enc.calls[-1][0].shape == (224, 224, 3)  # one view: the hand crop
    expect = d.head(torch.from_numpy(enc(enc.calls[-1])).float())[0].detach().numpy()
    assert np.allclose(q, expect, atol=1e-6)


def test_both_input_concatenates_full_then_hand_like_training(tmp_path):
    enc = FakeEncoder(dim=4)
    d = LearnedDriver(_run(tmp_path, 'both', dim=8), encode=enc, smooth=False)
    d.predict(np.zeros((90, 160, 3), np.uint8), _lm(), 0.0)
    assert len(enc.calls[-1]) == 2  # full frame, then hand crop (same order as select_features)


def test_smoothing_damps_a_one_frame_spike_and_resets(tmp_path):
    d = LearnedDriver(_run(tmp_path), encode=FakeEncoder())
    rgb = np.zeros((90, 160, 3), np.uint8)
    base = d.predict(rgb, _lm(), 0.0)
    spike = d.predict(np.full((90, 160, 3), 255, np.uint8), _lm(), 1 / 60)
    raw = LearnedDriver(_run(tmp_path / 'b'), encode=FakeEncoder(), smooth=False)
    raw_spike = raw.predict(np.full((90, 160, 3), 255, np.uint8), _lm(), 0.0)
    raw_base = raw.predict(rgb, _lm(), 1 / 60)
    assert np.abs(spike - base).max() < 0.5 * np.abs(raw_spike - raw_base).max() + 1e-9
    d.reset()
    assert np.allclose(d.predict(rgb, _lm(), 2.0), raw.head_predict(rgb, _lm()), atol=1e-5)


def test_missing_landmarks_are_refused(tmp_path):
    d = LearnedDriver(_run(tmp_path), encode=FakeEncoder())
    with pytest.raises(ValueError, match='landmarks'):
        d.predict(np.zeros((90, 160, 3), np.uint8), np.full((21, 3), np.nan), 0.0)


def test_blend_keeps_the_pinch_gripper():
    q = blend_gripper(np.arange(6, dtype=float), np.full(6, 9.0))
    assert q.tolist() == [0, 1, 2, 3, 4, 9]


class _Obs:
    lm_img = _lm()


def test_learned_command_gates_on_pipeline_status(tmp_path):
    from cta.learned import learned_command
    d = LearnedDriver(_run(tmp_path), encode=FakeEncoder(), smooth=False)
    rgb, pipe_q = np.zeros((90, 160, 3), np.uint8), np.full(6, 0.7)
    q, net = learned_command(d, 'TRACKING', _Obs(), rgb, 0.0, pipe_q, None)
    assert net is not None and q[5] == 0.7 and np.allclose(q[:5], net[:5])
    q2, net2 = learned_command(d, 'COAST', None, rgb, 0.01, pipe_q, net)  # brief dropout: hold the network pose
    assert np.allclose(q2[:5], net[:5]) and net2 is net
    q3, net3 = learned_command(d, 'NO HAND', None, rgb, 0.02, pipe_q, net)
    assert net3 is None and np.allclose(q3, pipe_q)
    q4, _ = learned_command(d, 'COAST', None, rgb, 0.03, pipe_q, None)  # COAST without a network pose yet
    assert np.allclose(q4, pipe_q)
