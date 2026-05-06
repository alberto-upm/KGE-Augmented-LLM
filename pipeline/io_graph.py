"""Carga de grafos y agrupación por bloques."""
from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Iterable

Triple = tuple[str, str, str]


def _expand_curie(curie: str, prefixes: dict[str, str]) -> str:
    if ":" in curie and not curie.startswith("http"):
        prefix, local = curie.split(":", 1)
        if prefix in prefixes:
            return prefixes[prefix] + local
    return curie


def _shorten(uri: str, prefixes: dict[str, str]) -> str:
    for prefix, expansion in prefixes.items():
        if uri.startswith(expansion):
            return f"{prefix}:{uri[len(expansion):]}"
    return uri


def load_graph(cfg: dict) -> list[Triple]:
    graph_cfg = cfg["graph"]
    fmt = graph_cfg["format"].lower()
    path = Path(graph_cfg["path"])
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra el grafo en {path}")

    if fmt == "ttl":
        return _load_ttl(path, graph_cfg.get("prefixes", {}))
    if fmt == "pkl":
        return _load_pkl(path, graph_cfg.get("prefixes", {}))
    if fmt in ("tsv", "txt"):
        return _load_tsv(path, graph_cfg.get("prefixes", {}), graph_cfg.get("has_header", True))
    raise ValueError(f"Formato no soportado: {fmt}")


def _load_ttl(path: Path, prefixes: dict[str, str]) -> list[Triple]:
    from rdflib import Graph

    g = Graph()
    g.parse(str(path), format="turtle")
    triples: list[Triple] = []
    for s, p, o in g:
        triples.append((_shorten(str(s), prefixes), _shorten(str(p), prefixes), _shorten(str(o), prefixes)))
    return triples


def _load_tsv(path: Path, prefixes: dict[str, str], has_header: bool) -> list[Triple]:
    triples: list[Triple] = []
    with path.open("r", encoding="utf-8") as f:
        first = True
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if first and has_header:
                first = False
                continue
            first = False
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            s, p, o = parts[0], parts[1], parts[2]
            triples.append((_shorten(s, prefixes), _shorten(p, prefixes), _shorten(o, prefixes)))
    return triples


def _load_pkl(path: Path, prefixes: dict[str, str]) -> list[Triple]:
    with path.open("rb") as f:
        data = pickle.load(f)

    if isinstance(data, list) and data and isinstance(data[0], (tuple, list)) and len(data[0]) == 3:
        return [(_shorten(str(s), prefixes), _shorten(str(p), prefixes), _shorten(str(o), prefixes)) for s, p, o in data]

    if isinstance(data, dict):
        triples: list[Triple] = []
        for subj, props in data.items():
            s = _shorten(str(subj), prefixes)
            if isinstance(props, dict):
                for prop, value in props.items():
                    p = _shorten(str(prop), prefixes)
                    if isinstance(value, (list, tuple, set)):
                        for v in value:
                            triples.append((s, p, _shorten(str(v), prefixes)))
                    else:
                        triples.append((s, p, _shorten(str(value), prefixes)))
        return triples

    raise ValueError(
        "Formato de pickle no reconocido. Esperado: lista de tripletas o dict {sujeto: {prop: valor}}."
    )


def group_by_block(
    triples: Iterable[Triple],
    block_predicate: str,
    block_value: str,
    prefixes: dict[str, str] | None = None,
) -> dict[str, list[Triple]]:
    """Agrupa tripletas por sujeto-bloque.

    - Si `block_predicate` y `block_value` están vacíos: cada sujeto es un bloque.
    - Si solo `block_value` está vacío: bloques son sujetos que tengan el predicado.
    - Si ambos están definidos: bloques son sujetos que cumplan
      `(s, block_predicate, block_value)`.
    """
    prefixes = prefixes or {}
    bp = _expand_curie(block_predicate, prefixes) if block_predicate else ""
    bv = _expand_curie(block_value, prefixes) if block_value else ""

    by_subject: dict[str, list[Triple]] = defaultdict(list)
    block_subjects: set[str] = set()

    no_filter = not block_predicate and not block_value

    for s, p, o in triples:
        by_subject[s].append((s, p, o))
        if no_filter:
            block_subjects.add(s)
            continue
        p_full = _expand_curie(p, prefixes)
        o_full = _expand_curie(o, prefixes)
        if (p == block_predicate or p_full == bp) and (not bv or o == block_value or o_full == bv):
            block_subjects.add(s)

    return {s: by_subject[s] for s in block_subjects}
