import json

import cv2
import numpy as np
import pytest

from cta.learning_data import audit_sessions, iter_rgb, read_take, split_takes


def make_take(root, name='take', status=(2, 0, 2, 3, 2), video_frames=None, **meta):
    path = root / name
    path.mkdir(parents=True)
    n = len(status)
    q = np.tile(np.arange(n)[:, None], (1, 6)).astype(float)
    q[-1, 0] = np.nan
    np.savez(path / 'frames.npz', t=np.arange(n) / 60, status=np.array(status), q_baseline=q)
    (path / 'meta.json').write_text(json.dumps({'mode': 'free', 'control': 'reach', **meta}))
    writer = cv2.VideoWriter(str(path / 'human.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 60, (64, 48))
    assert writer.isOpened()
    for i in range(n if video_frames is None else video_frames):
        frame = np.zeros((48, 64, 3), np.uint8)
        frame[:, :, 2] = 40 * i  # BGR red; decoder must return RGB
        writer.write(frame)
    writer.release()
    return path


def test_read_take_masks_bad_labels_and_preserves_video_indices(tmp_path):
    path = make_take(tmp_path)
    _, rows = read_take(path, stride=1)
    assert rows.tolist() == [0, 2]
    frames = list(iter_rgb(path, rows))
    assert len(frames) == 2
    assert frames[1][0, 0, 0] > 60 and frames[1][0, 0, 2] < 10


@pytest.mark.parametrize('meta', [{'control': 'arm'}, {'mode': 'paired'}])
def test_rejects_incompatible_controller_and_paired_takes(tmp_path, meta):
    path = make_take(tmp_path, **meta)
    with pytest.raises(ValueError):
        read_take(path)


def test_audit_reports_rejected_and_empty_sessions(tmp_path):
    assert audit_sessions(tmp_path)['accepted'] == []
    make_take(tmp_path, 'good')
    make_take(tmp_path, 'short', video_frames=2)
    report = audit_sessions(tmp_path, stride=1)
    assert report['accepted'][0]['name'] == 'good'
    assert report['usable_frames'] == 2
    assert report['rejected'][0]['name'] == 'short'


def test_split_requires_three_takes_and_never_splits_frames():
    names = [f'take_{i}' for i in range(12)]
    split = split_takes(names)
    assert split == split_takes(list(reversed(names)))
    assert set(split['train']).isdisjoint(split['val'])
    assert set(split['test']).isdisjoint(split['train'] + split['val'])
    assert sorted(sum(split.values(), [])) == sorted(names)
    with pytest.raises(ValueError):
        split_takes(['one', 'two'])


def test_decoder_detects_truncation_even_if_container_count_is_wrong(tmp_path):
    path = make_take(tmp_path, video_frames=2)
    with pytest.raises(ValueError):
        list(iter_rgb(path, np.array([0, 3])))


def test_corrupt_archive_is_reported_without_losing_other_takes(tmp_path):
    path = make_take(tmp_path)
    (path / 'frames.npz').write_bytes(b'PK\x03\x04not-a-zip')
    report = audit_sessions(tmp_path)
    assert len(report['rejected']) == 1
    assert report['accepted'] == []
