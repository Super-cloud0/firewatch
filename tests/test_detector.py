from types import SimpleNamespace

import numpy as np

from ecowatch.detector import parse_result


def test_parse_classify_result():
    r = SimpleNamespace(
        names={0: "fire", 1: "normal", 2: "Smoke"}, probs=SimpleNamespace(data=np.array([0.7, 0.2, 0.1]))
    )
    res = parse_result(r)
    assert res.scores == {"fire": 0.7, "normal": 0.2, "smoke": 0.1}
    assert res.top() == ("fire", 0.7)
    assert res.boxes == []


def test_parse_detect_result_filters_boxes_and_derives_normal():
    boxes = SimpleNamespace(
        conf=np.array([0.8, 0.2, 0.5]),
        cls=np.array([0, 0, 1]),
        xyxy=np.array([[1, 2, 30, 40], [0, 0, 5, 5], [10, 10, 50, 60]], dtype=float),
    )
    boxes.__len__ = lambda: 3  # noqa: E731
    r = SimpleNamespace(names={0: "fire", 1: "smoke"}, probs=None, boxes=_Sized(boxes, 3))
    res = parse_result(r, box_conf=0.3)
    assert len(res.boxes) == 2
    assert res.scores["fire"] == 0.8 and res.scores["smoke"] == 0.5
    assert abs(res.scores["normal"] - 0.2) < 1e-9


def test_parse_detect_result_empty():
    r = SimpleNamespace(names={0: "fire"}, probs=None, boxes=None)
    res = parse_result(r)
    assert res.scores == {"fire": 0.0, "normal": 1.0}
    assert res.top()[0] == "normal"


class _Sized:
    def __init__(self, inner, n):
        self._inner, self._n = inner, n

    def __len__(self):
        return self._n

    def __getattr__(self, item):
        return getattr(self._inner, item)
