"""Временное сглаживание и гистерезис: превращает поток кадров в инциденты.

Вероятности сглаживаются экспоненциальным средним, а инцидент начинается
только после `confirm_frames` подряд опасных кадров и завершается после
`clear_frames` подряд чистых. Это убирает «мигание» тревоги и ложные
срабатывания на единичных кадрах.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import DetectionParams
from .detector import FrameResult
from .hazards import get_hazard


@dataclass
class Decision:
    label: str  # активный класс угрозы или "normal"
    confidence: float  # сглаженная вероятность label
    active: bool  # идёт ли сейчас инцидент
    frame_danger: bool  # опасен ли текущий кадр (до гистерезиса)
    scores: dict[str, float] = field(default_factory=dict)
    onset: bool = False  # инцидент начался на этом кадре
    cleared: bool = False  # инцидент завершился на этом кадре

    @property
    def ambiguous(self) -> bool:
        ranked = sorted(self.scores.values(), reverse=True)
        return len(ranked) >= 2 and ranked[0] - ranked[1] < 0.12


def classify_frame(scores: dict[str, float], params: DetectionParams) -> tuple[str, float, bool]:
    """Решение по одному набору вероятностей: (кандидат, вероятность, опасно ли)."""
    if not scores:
        return "normal", 0.0, False
    ranked = sorted(scores.items(), key=lambda kv: (kv[1], get_hazard(kv[0]).severity), reverse=True)
    top_label, top_conf = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    danger = get_hazard(top_label).danger and top_conf >= params.conf_threshold and top_conf - second >= params.margin
    return top_label, top_conf, danger


class ThreatTracker:
    def __init__(self, params: DetectionParams | None = None) -> None:
        self.params = params or DetectionParams()
        self.reset()

    def reset(self) -> None:
        self._ema: dict[str, float] = {}
        self._danger_streak = 0
        self._clear_streak = 0
        self.active = False
        self.label = "normal"

    def _smooth(self, scores: dict[str, float]) -> dict[str, float]:
        a = self.params.ema_alpha
        if not self._ema:
            self._ema = dict(scores)
        else:
            for k in set(self._ema) | set(scores):
                self._ema[k] = a * scores.get(k, 0.0) + (1 - a) * self._ema.get(k, 0.0)
        return dict(self._ema)

    def update(self, result: FrameResult) -> Decision:
        p = self.params
        smoothed = self._smooth(result.scores)
        candidate, cand_conf, frame_danger = classify_frame(smoothed, p)
        onset = cleared = False

        if frame_danger:
            self._danger_streak += 1
            self._clear_streak = 0
        else:
            self._clear_streak += 1
            self._danger_streak = 0

        if not self.active:
            if self._danger_streak >= p.confirm_frames:
                self.active, onset, self.label = True, True, candidate
        else:
            if frame_danger and get_hazard(candidate).severity >= get_hazard(self.label).severity:
                self.label = candidate  # эскалация: дым → пожар
            if self._clear_streak >= p.clear_frames:
                self.active, cleared = False, True

        if self.active:
            label, conf = self.label, smoothed.get(self.label, 0.0)
        else:
            # Пока угроза не подтверждена, статус — «норма» (frame_danger покажет, что идёт проверка).
            label = "normal"
            conf = smoothed.get("normal", 1.0 - cand_conf if get_hazard(candidate).danger else cand_conf)
        return Decision(label, conf, self.active, frame_danger, smoothed, onset, cleared)


def evaluate_single(result: FrameResult, params: DetectionParams | None = None) -> Decision:
    """Решение для отдельного изображения — без сглаживания и гистерезиса."""
    params = params or DetectionParams()
    candidate, conf, danger = classify_frame(result.scores, params)
    label = candidate if danger else "normal"
    if not danger:
        conf = result.scores.get("normal", conf)
    return Decision(label, conf, danger, danger, dict(result.scores), onset=danger)
