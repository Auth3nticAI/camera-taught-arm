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
