"""Recomendación por propiedad (cascada).

Paso 1 — Reglas (PyClause): si una regla aplica con conf ≥ threshold,
        devuelve solo esa sugerencia y termina.
Paso 2 — CBR: configuraciones similares aportan candidatos preferentes.
Paso 3 — KGE: ordena/puntúa cualquier candidato (sea CBR o no) por
        plausibilidad sobre el estado parcial.
Paso 4 — Composición top-K: CBR-preferentes primero (ordenados por KGE),
        rellena huecos con KGE-only.

`fusion.mode = weighted_rrf` en el YAML cambia al modo alternativo de
`pipeline.rrf` (CBR y KGE pesados con RRF).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .cbr import CBRIndex, recommend as cbr_recommend, vocabulary_for_property
from .rrf import weighted_rrf
from .traceability import CBR, CBR_KGE, KGE, RULE, Suggestion


class KGEScorer(Protocol):
    def score_property(
        self,
        known_props: dict[str, str],
        target_prop: str,
        candidates: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Devuelve [(valor, score)] ordenado por score descendente."""


class RuleEngine(Protocol):
    def query(
        self,
        known_props: dict[str, str],
        target_prop: str,
        confidence_threshold: float,
    ) -> list[tuple[str, float, str]]:
        """Devuelve [(valor, confianza, rule_id)] o []."""


@dataclass
class InferenceResources:
    rule_engine: RuleEngine | None
    kge: KGEScorer
    cbr: CBRIndex
    cfg: dict


def recommend_property(
    known_props: dict[str, str],
    target_prop: str,
    res: InferenceResources,
) -> list[Suggestion]:
    cfg = res.cfg
    top_k = cfg["fusion"].get("top_k_final", 5)
    mode = cfg["fusion"].get("mode", "cbr_preferred")
    conf_threshold = cfg["rules"].get("confidence_threshold", 0.75)

    # Paso 1 — Reglas
    if res.rule_engine is not None:
        try:
            rule_hits = res.rule_engine.query(known_props, target_prop, conf_threshold)
        except Exception as e:  # PyClause puede fallar por reglas inaplicables
            print(f"[inference] Aviso: motor de reglas falló ({e}); siguiendo a CBR/KGE")
            rule_hits = []
        rule_hits = [(v, c, rid) for v, c, rid in rule_hits if c >= conf_threshold]
        if rule_hits:
            value, conf, rid = rule_hits[0]
            return [Suggestion(value=value, source=RULE, confidence=conf, rule_id=rid, rank=1)]

    # Paso 2 — CBR
    cbr_top = cbr_recommend(res.cbr, known_props, target_prop, top_k=cfg["cbr"].get("top_k", 10))
    cbr_values = {v for v, _ in cbr_top}

    # Paso 3 — KGE: puntúa todos los candidatos del vocabulario para esa prop
    vocab = list(vocabulary_for_property(res.cbr, target_prop))
    kge_scored = res.kge.score_property(known_props, target_prop, candidates=vocab)

    if mode == "weighted_rrf":
        return _compose_weighted_rrf(cbr_top, kge_scored, cfg, top_k)

    return _compose_cbr_preferred(cbr_top, cbr_values, kge_scored, top_k)


def _compose_cbr_preferred(
    cbr_top: list[tuple[str, float]],
    cbr_values: set[str],
    kge_scored: list[tuple[str, float]],
    top_k: int,
) -> list[Suggestion]:
    kge_score_by_val = {v: s for v, s in kge_scored}

    # CBR ordenado internamente por KGE (desempate). Si el KGE no conoce el
    # valor, le damos score 0 para mantenerlo abajo dentro del bloque CBR.
    cbr_sorted = sorted(
        cbr_top,
        key=lambda x: kge_score_by_val.get(x[0], 0.0),
        reverse=True,
    )

    suggestions: list[Suggestion] = []
    rank = 1
    for value, cbr_score in cbr_sorted:
        kge_score = kge_score_by_val.get(value)
        source = CBR_KGE if kge_score is not None and kge_score > 0 else CBR
        suggestions.append(
            Suggestion(
                value=value,
                source=source,
                confidence=cbr_score,
                cbr_score=cbr_score,
                kge_score=kge_score,
                rank=rank,
            )
        )
        rank += 1
        if rank > top_k:
            break

    # Rellena con KGE-only los huecos restantes
    for value, kge_score in kge_scored:
        if rank > top_k:
            break
        if value in cbr_values:
            continue
        suggestions.append(
            Suggestion(
                value=value,
                source=KGE,
                confidence=kge_score,
                kge_score=kge_score,
                rank=rank,
            )
        )
        rank += 1

    return suggestions


def _compose_weighted_rrf(
    cbr_top: list[tuple[str, float]],
    kge_scored: list[tuple[str, float]],
    cfg: dict,
    top_k: int,
) -> list[Suggestion]:
    fcfg = cfg["fusion"]
    fused = weighted_rrf(
        rank_kge=kge_scored,
        rank_cbr=cbr_top,
        w_kge=fcfg.get("w_kge", 0.7),
        w_cbr=fcfg.get("w_cbr", 0.3),
        k=fcfg.get("k_rrf", 60),
    )
    suggestions: list[Suggestion] = []
    for rank, item in enumerate(fused[:top_k], start=1):
        if item.rank_cbr is not None and item.rank_kge is not None:
            source = CBR_KGE
        elif item.rank_cbr is not None:
            source = CBR
        else:
            source = KGE
        suggestions.append(
            Suggestion(
                value=item.value,
                source=source,
                confidence=item.rrf_score,
                kge_score=item.score_kge,
                cbr_score=item.score_cbr,
                rank=rank,
            )
        )
    return suggestions
