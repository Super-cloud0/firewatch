"""Каталог классов, которые умеет различать модель."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Hazard:
    key: str
    title: str
    icon: str
    color: str  # HEX для интерфейса
    description: str
    danger: bool
    severity: int  # чем больше, тем опаснее; влияет на выбор класса при равенстве

    @property
    def bgr(self) -> tuple[int, int, int]:
        h = self.color.lstrip("#")
        r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
        return b, g, r


HAZARDS: dict[str, Hazard] = {
    h.key: h
    for h in (
        Hazard("fire", "Пожар", "🔥", "#ef4444", "Открытое пламя · критическая опасность", True, 3),
        Hazard("trash", "Сжигание мусора", "🗑️", "#eab308", "Горение бытовых отходов, пластика", True, 2),
        Hazard("smoke", "Дым", "💨", "#94a3b8", "Обнаружен дым · требуется проверка", True, 1),
        Hazard("factory", "Производственный дым", "🏭", "#3b82f6", "Промышленный выброс", False, 0),
        Hazard("normal", "Норма", "✅", "#22c55e", "Угроз не обнаружено", False, 0),
    )
}

_ALIASES = {"flame": "fire", "fires": "fire", "garbage": "trash", "background": "normal", "none": "normal"}


def normalize(name: str) -> str:
    key = str(name).strip().lower()
    return _ALIASES.get(key, key)


def get_hazard(name: str) -> Hazard:
    key = normalize(name)
    if key in HAZARDS:
        return HAZARDS[key]
    return Hazard(key, key.capitalize(), "❔", "#8b8fa3", "Неизвестный класс модели", False, 0)


def is_danger(name: str) -> bool:
    return get_hazard(name).danger
