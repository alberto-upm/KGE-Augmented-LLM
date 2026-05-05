"""Case-Based Reasoning: índice de bloques históricos y recuperación.

Cada bloque (incidencia entera) se reduce a un dict prop→valor. Para una
configuración parcial, se buscan los bloques más similares por Jaccard sobre
los pares prop-valor coincidentes y se extraen los valores que esos bloques
tienen para la propiedad objetivo.
"""
from __future__ import annotations

import json
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .io_graph import Triple


def block_to_props(triples: list[Triple]) -> dict[str, str]:
    """Reduce un bloque a dict prop→valor.

    Si una propiedad aparece varias veces se queda con la primera ocurrencia
    (suficiente para los grafos del proyecto donde la mayoría de props son
    funcionales).
    """
    props: dict[str, str] = {}
    for _s, p, o in triples:
        if p not in props:
            props[p] = o
    return props


@dataclass
class CBRIndex:
    cases: list[dict[str, str]]    # cada caso = dict prop→valor

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self.cases, f)

    @classmethod
    def load(cls, path: Path) -> "CBRIndex":
        with path.open("rb") as f:
            cases = pickle.load(f)
        return cls(cases=cases)


def build_index(blocks: dict[str, list[Triple]]) -> CBRIndex:
    cases = [block_to_props(triples) for triples in blocks.values()]
    return CBRIndex(cases=cases)


def _jaccard(known: dict[str, str], case: dict[str, str]) -> float:
    if not known:
        return 0.0
    known_pairs = set(known.items())
    case_pairs = set(case.items())
    inter = len(known_pairs & case_pairs)
    union = len(known_pairs | case_pairs)
    return inter / union if union else 0.0


def recommend(
    index: CBRIndex,
    known_props: dict[str, str],
    target_prop: str,
    top_k: int,
) -> list[tuple[str, float]]:
    """Devuelve [(valor_objetivo, score)] ordenado por score descendente.

    El score combina (similitud Jaccard del caso con el estado parcial) y la
    frecuencia con la que ese valor aparece entre los casos similares.
    """
    if not known_props:
        # sin contexto, devolvemos los valores más frecuentes para esa prop
        counts: Counter[str] = Counter(c[target_prop] for c in index.cases if target_prop in c)
        if not counts:
            return []
        max_count = counts.most_common(1)[0][1]
        return [(v, c / max_count) for v, c in counts.most_common(top_k)]

    sims_per_value: dict[str, list[float]] = {}
    for case in index.cases:
        if target_prop not in case:
            continue
        sim = _jaccard(known_props, case)
        if sim <= 0:
            continue
        sims_per_value.setdefault(case[target_prop], []).append(sim)

    scored: list[tuple[str, float]] = []
    for value, sims in sims_per_value.items():
        # similitud agregada: media ponderada por número de apariciones (a más
        # casos similares que coinciden, más confianza en ese valor).
        score = sum(sims) / len(sims) * (1 + min(len(sims), 10) / 20)
        scored.append((value, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def vocabulary_for_property(index: CBRIndex, target_prop: str) -> set[str]:
    return {c[target_prop] for c in index.cases if target_prop in c}


def run_build_index(cfg: dict, blocks: dict[str, list[Triple]]) -> Path:
    index = build_index(blocks)
    out_path = Path(cfg["output"]["out_dir"]) / "cbr_index.pkl"
    index.save(out_path)
    print(f"[CBR] Índice con {len(index.cases):,} casos guardado en {out_path}")

    summary = {
        "n_cases": len(index.cases),
        "props_observed": sorted({p for c in index.cases for p in c.keys()}),
    }
    (out_path.parent / "cbr_summary.json").write_text(json.dumps(summary, indent=2))
    return out_path
