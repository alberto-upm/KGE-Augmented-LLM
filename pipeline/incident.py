"""Estado de la incidencia en construcción y orden de propiedades."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PropertySpec:
    uri: str
    label: str
    active: bool


@dataclass
class Incident:
    """Estado parcial de una incidencia.

    `props[uri] = valor | None`. `sources[uri] = "RULE" | "CBR" | "KGE" | "USER"`.
    `skipped` registra propiedades que el usuario marcó como skip explícitamente.
    """

    props: dict[str, str | None] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    skipped: set[str] = field(default_factory=set)

    def known(self) -> dict[str, str]:
        return {k: v for k, v in self.props.items() if v is not None}

    def update(self, prop: str, value: str, source: str) -> None:
        self.props[prop] = value
        self.sources[prop] = source

    def skip(self, prop: str) -> None:
        self.props[prop] = None
        self.skipped.add(prop)

    def to_dict(self) -> dict[str, Any]:
        return {"props": dict(self.props), "sources": dict(self.sources), "skipped": sorted(self.skipped)}


def load_property_specs(cfg: dict) -> list[PropertySpec]:
    return [
        PropertySpec(uri=item["uri"], label=item["label"], active=bool(item.get("active", True)))
        for item in cfg["graph"].get("interaction_order", [])
    ]


def active_specs(specs: list[PropertySpec]) -> list[PropertySpec]:
    return [s for s in specs if s.active]


def is_complete(incident: Incident, specs: list[PropertySpec]) -> bool:
    """Una incidencia está completa cuando todas las props activas tienen valor o se marcaron skip."""
    for spec in active_specs(specs):
        if incident.props.get(spec.uri) is None and spec.uri not in incident.skipped:
            return False
    return True
