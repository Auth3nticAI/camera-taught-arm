import json

import numpy as np
import pytest
import torch

from cta.learning import JointHead, load_head, main, prepare_image, train_head, validate_setups


def test_head_learns_and_checkpoint_reloads(tmp_path):
    rng = np.random.default_rng(12)
    takes = {}
    for i in range(6):
        x = rng.normal(size=(32, 4)).astype('float32')
        y = np.tile((x[:, :1] * 0.2 + 0.1), (1, 6))
        takes[f'take_{i}'] = {'features': x, 'targets': y}
    result = train_head(takes, tmp_path / 'run', epochs=70, seed=42)
    assert result['test_mae_deg'] < result['mean_baseline_test_mae_deg'] * 0.5
    model = load_head(tmp_path / 'run' / 'best.pt')
    assert model(torch.from_numpy(takes['take_0']['features'])).shape == (32, 6)
    assert torch.isfinite(model(torch.zeros(1, 4))).all()
    assert json.loads((tmp_path / 'run' / 'metrics.json').read_text())['target'] == 'q_baseline'
    with pytest.raises(FileExistsError):
        train_head(takes, tmp_path / 'run', epochs=1)


def test_training_mean_and_normalization_ignore_test_and_validation(tmp_path):
    from cta.learning_data import split_takes
    split = split_takes(['a', 'b', 'c'])
    takes = {}
    for part, names in split.items():
        for name in names:
            value = 1.0 if part == 'train' else 100.0
            takes[name] = {'features': np.full((4, 2), value, dtype='float32'),
                           'targets': np.full((4, 6), value, dtype='float32')}
    train_head(takes, tmp_path / 'run', epochs=1)
    model = load_head(tmp_path / 'run' / 'best.pt')
    assert model.feature_mean.tolist() == [1.0, 1.0]
    checkpoint = torch.load(tmp_path / 'run' / 'best.pt', weights_only=True)
    assert checkpoint['mean_target'] == [1.0] * 6


def test_prepare_image_preserves_full_view_without_center_crop():
    image = np.zeros((40, 80, 3), np.uint8)
    image[:, :10, 0] = 255
    square = prepare_image(image)
    assert square.shape == (224, 224, 3)
    assert square[112, 0, 0] == 255
    assert square[0].max() == 0


def test_setup_mismatch_is_rejected_before_extraction():
    base = {'hand': 'left', 'mirrored': True, 'calibration': {'roll0': 0.1}}
    validate_setups([base, dict(base)])
    with pytest.raises(ValueError, match='calibration'):
        validate_setups([base, {**base, 'calibration': {'roll0': 1.0}}])
    with pytest.raises(ValueError, match='mirrored'):
        validate_setups([base, {**base, 'mirrored': False}])


def test_empty_data_cli_returns_actionable_failure_without_model_download(tmp_path, capsys):
    assert main(['audit', '--sessions', str(tmp_path)]) == 1
    assert '"usable_frames": 0' in capsys.readouterr().out
    assert main(['extract', '--sessions', str(tmp_path), '--output', str(tmp_path / 'cache')]) == 1
    assert 'at least 3' in capsys.readouterr().out
    assert not (tmp_path / 'cache').exists()


def test_validate_setups_ignores_measured_fps_jitter():
    """Same camera, same mode: the measured fps differs slightly every run (59.41 vs 59.47) - not a new setup."""
    from cta.learning import validate_setups
    base = {'hand': 'left', 'mirrored': False, 'control': 'reach'}
    a = {**base, 'camera_mode': {'width': 1920, 'height': 1080, 'fps': 59.411, 'fourcc': 'MJPG'}}
    b = {**base, 'camera_mode': {'width': 1920, 'height': 1080, 'fps': 59.471, 'fourcc': 'MJPG'}}
    validate_setups([a, b])
    c = {**base, 'camera_mode': {'width': 1280, 'height': 720, 'fps': 59.4, 'fourcc': 'MJPG'}}
    with pytest.raises(ValueError, match='camera_mode'):
        validate_setups([a, c])


# ---- regularization experiment (ch. 12): dropout, feature noise, image augmentation ---------------------

def _toy_takes(aug=False):
    rng = np.random.default_rng(3)
    takes = {}
    for i in range(6):
        x = rng.normal(size=(32, 4)).astype('float32')
        t = {'features': x, 'targets': np.tile(x[:, :1] * 0.2 + 0.1, (1, 6)).astype('float32')}
        if aug:
            t['features_aug'] = (x[:, None, :] + rng.normal(scale=0.01, size=(32, 2, 4))).astype('float32')
        takes[f'take_{i}'] = t
    return takes


def test_regularized_head_records_config_and_reloads(tmp_path):
    m = train_head(_toy_takes(aug=True), tmp_path / 'r', epochs=40, seed=42, dropout=0.3, weight_decay=0.05,
                   feature_noise=0.1, use_aug=True)
    assert m['config'] == {'dropout': 0.3, 'weight_decay': 0.05, 'feature_noise': 0.1, 'use_aug': True,
                           'hidden': 128}
    head = load_head(tmp_path / 'r' / 'best.pt')
    assert any(isinstance(l, torch.nn.Dropout) and l.p == 0.3 for l in head.net)
    assert m['frames']['train_aug'] == len(m['split']['train']) * 32 * 2  # augmented copies only for TRAIN takes
    assert m['frames']['train'] == len(m['split']['train']) * 32  # originals counted separately


def test_augmented_copies_never_used_for_val_or_test(tmp_path):
    takes = _toy_takes(aug=True)
    for t in takes.values():
        t['features_aug'][:] = 1e6  # poison: if used for val/test the metrics explode
    m = train_head(takes, tmp_path / 'r', epochs=5, seed=42, use_aug=False)
    assert np.isfinite(m['test_mae_deg']) and 'train_aug' not in m['frames']


def test_augment_image_keeps_left_right_and_shape():
    """No horizontal flip: a flip would swap the user's left/right and corrupt the targets."""
    from cta.learning import augment_image
    img = np.zeros((224, 224, 3), np.uint8)
    img[:, :60] = 255  # bright LEFT band
    rng = np.random.default_rng(0)
    for _ in range(20):
        out = augment_image(img, rng)
        assert out.shape == img.shape and out.dtype == np.uint8
        assert out[:, :80].mean() > out[:, -80:].mean()  # still brighter on the left
