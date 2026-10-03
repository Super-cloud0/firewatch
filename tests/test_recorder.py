import cv2
import numpy as np

from ecowatch.recorder import ClipRecorder


def frame(v):
    return np.full((48, 64, 3), v, np.uint8)


def count_frames(path):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def test_clip_has_preroll_and_postroll(tmp_path):
    rec = ClipRecorder(tmp_path, fps=10, pre_seconds=1, post_seconds=0.55)
    t = 0.0
    for _ in range(30):  # 3 с без тревоги — в клип попадёт только последняя секунда
        assert rec.push(frame(0), False, t) is None
        t += 0.1
    for _ in range(5):
        rec.push(frame(255), True, t)
        t += 0.1
    assert rec.recording
    path = None
    while path is None:
        path = rec.push(frame(0), False, t)
        t += 0.1
    assert path.exists() and not rec.recording
    assert count_frames(path) == 10 + 5 + 6


def test_close_without_recording(tmp_path):
    assert ClipRecorder(tmp_path).close() is None
