"""Sugerencia única con trazabilidad de fuente."""
from __future__ import annotations

from dataclasses import dataclass, field

# Fuentes válidas. CBR+KGE indica que el candidato apareció en ambas listas.
RULE = "RULE"
CBR = "CBR"
KGE = "KGE"
CBR_KGE = "CBR+KGE"
USER = "USER"
VALID_SOURCES = {RULE, CBR, KGE, CBR_KGE, USER}


@dataclass
class Suggestion:
    value: str
    source: str
    confidence: float = 0.0
    rule_id: str | None = None
    kge_score: float | None = None
    cbr_score: float | None = None
    rank: int | None = None
    extras: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source not in VALID_SOURCES:
            raise ValueError(f"source inválido: {self.source}. Esperado {VALID_SOURCES}")

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "source": self.source,
            "confidence": self.confidence,
            "rule_id": self.rule_id,
            "kge_score": self.kge_score,
            "cbr_score": self.cbr_score,
            "rank": self.rank,
        }
