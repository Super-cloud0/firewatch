from ecowatch.config import DetectionParams
from ecowatch.detector import FrameResult
from ecowatch.tracker import ThreatTracker, classify_frame, evaluate_single

FIRE = FrameResult({"fire": 0.9, "normal": 0.05, "smoke": 0.05})
CLEAN = FrameResult({"fire": 0.02, "normal": 0.95, "smoke": 0.03})
PARAMS = DetectionParams(conf_threshold=0.6, margin=0.15, ema_alpha=1.0, confirm_frames=3, clear_frames=2)


def test_classify_frame_requires_threshold_and_margin():
    p = DetectionParams(conf_threshold=0.6, margin=0.2)
    assert classify_frame({"fire": 0.9, "normal": 0.1}, p)[2]
    assert not classify_frame({"fire": 0.55, "normal": 0.45}, p)[2]  # ниже порога
    assert not classify_frame({"fire": 0.62, "smoke": 0.5}, p)[2]  # мал отрыв
    assert not classify_frame({"normal": 0.9, "fire": 0.1}, p)[2]
    assert not classify_frame({}, p)[2]


def test_onset_after_confirm_frames_and_clear_after_clear_frames():
    t = ThreatTracker(PARAMS)
    decisions = [t.update(FIRE) for _ in range(3)]
    assert [d.active for d in decisions] == [False, False, True]
    assert decisions[0].frame_danger and decisions[0].label == "normal"
    assert decisions[2].onset and decisions[2].label == "fire"
    assert sum(d.onset for d in decisions + [t.update(FIRE) for _ in range(5)]) == 1

    d1 = t.update(CLEAN)
    assert d1.active and not d1.cleared  # одного чистого кадра мало
    d2 = t.update(CLEAN)
    assert d2.cleared and not d2.active and d2.label == "normal"


def test_single_noisy_frame_does_not_trigger():
    t = ThreatTracker(PARAMS)
    seq = [CLEAN, FIRE, CLEAN, FIRE, FIRE, CLEAN]
    assert not any(t.update(r).active for r in seq)


def test_ema_suppresses_spike():
    t = ThreatTracker(DetectionParams(ema_alpha=0.3, confirm_frames=1))
    for _ in range(10):
        t.update(CLEAN)
    assert not t.update(FIRE).frame_danger  # один всплеск не перебивает историю


def test_escalation_from_smoke_to_fire():
    t = ThreatTracker(PARAMS)
    smoke = FrameResult({"smoke": 0.9, "normal": 0.05, "fire": 0.05})
    for _ in range(3):
        d = t.update(smoke)
    assert d.label == "smoke"
    d = t.update(FIRE)
    assert d.active and d.label == "fire" and not d.onset


def test_evaluate_single():
    assert evaluate_single(FIRE).label == "fire"
    d = evaluate_single(CLEAN)
    assert d.label == "normal" and d.confidence == 0.95 and not d.active
