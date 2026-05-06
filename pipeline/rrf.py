"""Weighted Reciprocal Rank Fusion.

Modo alternativo de fusión cuando `fusion.mode = weighted_rrf` en el YAML.
El flujo principal usa CBR-preferente con KGE como ordenador (ver
`inference.py`); este módulo está disponible para experimentación.

Fórmula:
    score(d) = w_kge · 1/(k + rank_kge(d)) + w_cbr · 1/(k + rank_cbr(d))
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RankedItem:
    value: str
    rank_kge: int | None
    rank_cbr: int | None
    score_kge: float | None
    score_cbr: float | None
    rrf_score: float


def weighted_rrf(
    rank_kge: list[tuple[str, float]],
    rank_cbr: list[tuple[str, float]],
    w_kge: float,
    w_cbr: float,
    k: int = 60,
) -> list[RankedItem]:
    """Fusiona dos rankings (lista ordenada de (valor, score)) con pesos."""
    pos_kge = {v: (i + 1, s) for i, (v, s) in enumerate(rank_kge)}
    pos_cbr = {v: (i + 1, s) for i, (v, s) in enumerate(rank_cbr)}

    candidates = set(pos_kge) | set(pos_cbr)
    items: list[RankedItem] = []
    for v in candidates:
        rk = pos_kge.get(v)
        rc = pos_cbr.get(v)
        score = 0.0
        if rk is not None:
            score += w_kge * 1.0 / (k + rk[0])
        if rc is not None:
            score += w_cbr * 1.0 / (k + rc[0])
        items.append(
            RankedItem(
                value=v,
                rank_kge=rk[0] if rk else None,
                rank_cbr=rc[0] if rc else None,
                score_kge=rk[1] if rk else None,
                score_cbr=rc[1] if rc else None,
                rrf_score=score,
            )
        )
    items.sort(key=lambda x: x.rrf_score, reverse=True)
    return items
