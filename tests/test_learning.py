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


# ---- hand-crop input (D-U): a close-up of the hand next to the full frame ---------------------------------

def _lm_box(u0, v0, u1, v1):
    lm = np.zeros((21, 3))
    lm[:, 0] = np.linspace(u0, u1, 21)
    lm[:, 1] = np.linspace(v0, v1, 21)
    return lm


def test_hand_crop_centres_the_hand_and_pads_outside_the_image():
    from cta.learning import hand_crop
    img = np.zeros((100, 200, 3), np.uint8)
    img[40:60, 140:160] = 255  # the "hand"
    out = hand_crop(img, _lm_box(0.70, 0.40, 0.80, 0.60))
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8
    assert out[112, 112].min() == 255  # hand at the centre of the crop
    assert out[100:124, 100:124].mean() > 200
    edge = hand_crop(img, _lm_box(0.95, 0.90, 1.05, 1.10))  # partly off-image: zero padding, no crash
    assert edge.shape == (224, 224, 3) and edge[-1, -1].max() == 0


def test_hand_crop_without_landmarks_is_black():
    from cta.learning import hand_crop
    out = hand_crop(np.full((50, 50, 3), 200, np.uint8), np.full((21, 3), np.nan))
    assert out.shape == (224, 224, 3) and out.max() == 0


def test_select_features_concatenates_full_and_hand_including_aug():
    from cta.learning import select_features
    z = {'features': np.ones((5, 4)), 'features_hand': np.full((5, 4), 2.0), 'targets': np.zeros((5, 6)),
         'features_aug': np.ones((5, 2, 4)), 'features_hand_aug': np.full((5, 2, 4), 2.0)}
    both = select_features(z, 'both')
    assert both['features'].shape == (5, 8) and both['features_aug'].shape == (5, 2, 8)
    assert both['features'][0].tolist() == [1, 1, 1, 1, 2, 2, 2, 2]
    assert select_features(z, 'hand')['features'][0].tolist() == [2, 2, 2, 2]
    assert select_features(z, 'full')['features'].shape == (5, 4)
    with pytest.raises(ValueError, match='features_hand'):
        select_features({'features': np.ones((5, 4)), 'targets': np.zeros((5, 6))}, 'both')


def test_explicit_split_is_used_and_must_be_disjoint(tmp_path):
    takes = _toy_takes()
    split = {'train': ['take_0', 'take_1', 'take_2', 'take_3'], 'val': ['take_4'], 'test': ['take_5']}
    m = train_head(takes, tmp_path / 'r', epochs=2, split=split)
    assert m['split'] == split
    with pytest.raises(ValueError, match='disjoint'):
        train_head(takes, tmp_path / 'r2', epochs=1, split={**split, 'val': ['take_0']})
