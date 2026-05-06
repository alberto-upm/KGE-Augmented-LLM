"""Extracción de tripletas y división por bloques de incidencia.

Para grafos con bloques identificables (incidencias):
  - Lee el TTL con parser manual (fallback si rdflib falla)
  - Agrupa tripletas por incidencia (sujeto-bloque)
  - Divide 90 % train / 10 % test a nivel de incidencia completa
  - Escribe tres ficheros en data/<grafo>/raw/:
      train_triples.tsv   — 3 col (s, p, o)        para KGE y AnyBURL
      train_blocks.tsv    — 4 col (block, s, p, o)  para CBR
      test_system.tsv     — 4 col (block, s, p, o)  para evaluación

Para grafos planos (DBpedia):
  - Normaliza el TSV existente a triples.tsv sin cabecera

Uso:
    python run.py --config configs/incidents.yaml --mode convert
    python run.py --config configs/dbpedia.yaml   --mode convert
"""
from __future__ import annotations

import pickle
import random
import re
from collections import defaultdict
from pathlib import Path

from .io_graph import Triple, _shorten


# ── Parser TTL manual ─────────────────────────────────────────────────────────

def _parse_prefixes(text: str) -> dict[str, str]:
    prefixes: dict[str, str] = {}
    for m in re.finditer(r"@prefix\s+(\w*):\s*<([^>]+)>\s*\.", text):
        prefixes[m.group(1)] = m.group(2)
    return prefixes


def _resolve(token: str, declared: dict[str, str], cfg_prefixes: dict[str, str]) -> str:
    token = token.strip().rstrip(";,. ")
    if token.startswith("<") and token.endswith(">"):
        return _shorten(token[1:-1], cfg_prefixes)
    if ":" in token and not token.startswith("_:"):
        pfx, local = token.split(":", 1)
        if pfx in declared:
            return _shorten(declared[pfx] + local, cfg_prefixes)
    return token


def _parse_ttl_blocks(path: Path, cfg_prefixes: dict[str, str]) -> dict[str, list[Triple]]:
    """Lee el TTL y devuelve dict {block_id: [tripletas]}."""
    text = path.read_text(encoding="utf-8", errors="replace")
    declared = _parse_prefixes(text)

    blocks: dict[str, list[Triple]] = defaultdict(list)
    pending: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("@") or line.startswith("#"):
            continue
        pending.append(line)
        if not line.endswith("."):
            continue

        block = " ".join(pending).rstrip(". ").strip()
        pending = []

        tokens = block.split()
        if len(tokens) < 3:
            continue

        subj = _resolve(tokens[0], declared, cfg_prefixes)
        rest = " ".join(tokens[1:])
        for pair in rest.split(";"):
            parts = pair.split()
            if len(parts) >= 2:
                pred = _resolve(parts[0], declared, cfg_prefixes)
                obj  = _resolve(parts[1], declared, cfg_prefixes)
                blocks[subj].append((subj, pred, obj))

    return dict(blocks)


def _parse_ttl_blocks_rdflib(path: Path, cfg_prefixes: dict[str, str]) -> dict[str, list[Triple]]:
    from rdflib import Graph  # type: ignore

    g = Graph()
    g.parse(str(path), format="turtle")
    blocks: dict[str, list[Triple]] = defaultdict(list)
    for s, p, o in g:
        subj = _shorten(str(s), cfg_prefixes)
        blocks[subj].append((
            subj,
            _shorten(str(p), cfg_prefixes),
            _shorten(str(o), cfg_prefixes),
        ))
    return dict(blocks)


def _load_blocks_ttl(path: Path, cfg_prefixes: dict[str, str]) -> dict[str, list[Triple]]:
    try:
        blocks = _parse_ttl_blocks_rdflib(path, cfg_prefixes)
        print(f"[convert] rdflib OK: {len(blocks):,} bloques leídos.")
        return blocks
    except Exception as e:
        print(f"[convert] rdflib falló ({e}); usando parser manual.")
        blocks = _parse_ttl_blocks(path, cfg_prefixes)
        print(f"[convert] Parser manual: {len(blocks):,} bloques leídos.")
        return blocks


# ── Loaders planos (sin bloques) ──────────────────────────────────────────────

def _load_pkl_flat(path: Path, cfg_prefixes: dict[str, str]) -> list[Triple]:
    with path.open("rb") as f:
        data = pickle.load(f)
    triples: list[Triple] = []
    if isinstance(data, list) and data and isinstance(data[0], (tuple, list)) and len(data[0]) == 3:
        for s, p, o in data:
            triples.append((_shorten(str(s), cfg_prefixes), _shorten(str(p), cfg_prefixes), _shorten(str(o), cfg_prefixes)))
        return triples
    if isinstance(data, dict):
        for subj, props in data.items():
            s = _shorten(str(subj), cfg_prefixes)
            if isinstance(props, dict):
                for prop, value in props.items():
                    p = _shorten(str(prop), cfg_prefixes)
                    for v in (value if isinstance(value, (list, tuple, set)) else [value]):
                        triples.append((s, p, _shorten(str(v), cfg_prefixes)))
        return triples
    raise ValueError("Formato de pickle no reconocido.")


def _load_tsv_flat(path: Path, has_header: bool) -> list[Triple]:
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
            if len(parts) >= 3:
                triples.append((parts[0], parts[1], parts[2]))
    return triples


# ── Escritura de ficheros ──────────────────────────────────────────────────────

def _write_3col(triples: list[Triple], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s, p, o in triples:
            f.write(f"{s}\t{p}\t{o}\n")


def _write_4col(blocks: dict[str, list[Triple]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for block_id, triples in blocks.items():
            for s, p, o in triples:
                f.write(f"{block_id}\t{s}\t{p}\t{o}\n")


# ── Conversión con split 90/10 por bloques (incidencias) ─────────────────────

def _filter_blocks_by_predicate(
    blocks: dict[str, list[Triple]],
    block_predicate: str,
    block_value: str,
) -> dict[str, list[Triple]]:
    """Filtra bloques que contengan (s, block_predicate, block_value)."""
    if not block_predicate and not block_value:
        return blocks
    out: dict[str, list[Triple]] = {}
    for bid, triples in blocks.items():
        for s, p, o in triples:
            if (not block_predicate or p == block_predicate) and \
               (not block_value or o == block_value):
                out[bid] = triples
                break
    return out


def convert_incidents(cfg: dict) -> dict[str, Path]:
    """Convierte el TTL de incidencias con split 90/10 por bloque.

    Escribe en data/incidents/raw/:
      train_triples.tsv   — tripletas de train (3 col)
      train_blocks.tsv    — bloques de train con block_id (4 col), para CBR
      test_system.tsv     — bloques de test  con block_id (4 col), para eval
    """
    graph_cfg  = cfg["graph"]
    src_path   = Path(graph_cfg["path"])
    cfg_pfx    = graph_cfg.get("prefixes", {})
    split_cfg  = cfg["split"]
    seed       = split_cfg.get("seed", 42)
    test_ratio = split_cfg.get("test_system_ratio", 0.10)
    out_dir    = src_path.parent

    train_triples_path = out_dir / "train_triples.tsv"
    train_blocks_path  = out_dir / "train_blocks.tsv"
    test_system_path   = out_dir / "test_system.tsv"

    if train_triples_path.exists() and test_system_path.exists():
        print(f"[convert] Ya existen los ficheros de split en {out_dir}. Bórralos para regenerar.")
        return {
            "train_triples": train_triples_path,
            "train_blocks":  train_blocks_path,
            "test_system":   test_system_path,
        }

    print(f"[convert] Leyendo {src_path} y agrupando por bloques…")
    all_blocks = _load_blocks_ttl(src_path, cfg_pfx)

    # Filtrar bloques que sean incidencias reales
    block_pred  = graph_cfg.get("block_predicate", "")
    block_val   = graph_cfg.get("block_value", "")
    incident_blocks = _filter_blocks_by_predicate(all_blocks, block_pred, block_val)
    print(f"[convert] Bloques (incidencias) identificados: {len(incident_blocks):,}")

    # División 90 / 10 por bloque completo
    block_ids = list(incident_blocks.keys())
    random.Random(seed).shuffle(block_ids)
    n_test = max(1, int(len(block_ids) * test_ratio))
    test_ids  = set(block_ids[:n_test])
    train_ids = set(block_ids[n_test:])

    train_blocks = {bid: incident_blocks[bid] for bid in train_ids}
    test_blocks  = {bid: incident_blocks[bid] for bid in test_ids}

    train_triples_flat = [t for ts in train_blocks.values() for t in ts]

    print(f"[convert] Train: {len(train_ids):,} incidencias, {len(train_triples_flat):,} tripletas")
    print(f"[convert] Test:  {len(test_ids):,}  incidencias")

    _write_3col(train_triples_flat, train_triples_path)
    _write_4col(train_blocks, train_blocks_path)
    _write_4col(test_blocks, test_system_path)

    print(f"\n[convert] Ficheros generados:")
    print(f"  {train_triples_path}  ← KGE y AnyBURL")
    print(f"  {train_blocks_path}   ← índice CBR")
    print(f"  {test_system_path}    ← evaluación del sistema")
    print(f"\n[convert] Actualiza configs/incidents.yaml:")
    print(f"  format: tsv")
    print(f"  path:   {train_triples_path}")
    print(f"  has_header: false")

    return {
        "train_triples": train_triples_path,
        "train_blocks":  train_blocks_path,
        "test_system":   test_system_path,
    }


# ── Conversión simple para grafos planos (DBpedia) ────────────────────────────

def convert_flat(cfg: dict) -> Path:
    """Normaliza un grafo plano (TSV/PKL) a triples.tsv sin cabecera."""
    graph_cfg  = cfg["graph"]
    fmt        = graph_cfg["format"].lower()
    src_path   = Path(graph_cfg["path"])
    cfg_pfx    = graph_cfg.get("prefixes", {})
    has_header = graph_cfg.get("has_header", True)
    out_path   = src_path.parent / "triples.tsv"

    if out_path.exists():
        print(f"[convert] Ya existe {out_path}. Bórralo para regenerar.")
        return out_path

    print(f"[convert] Leyendo {src_path} (formato {fmt})…")
    if fmt == "pkl":
        triples = _load_pkl_flat(src_path, cfg_pfx)
    elif fmt in ("tsv", "txt"):
        triples = _load_tsv_flat(src_path, has_header)
    else:
        raise ValueError(f"Formato no soportado para conversión plana: {fmt}")

    print(f"[convert] {len(triples):,} tripletas. Guardando en {out_path}…")
    _write_3col(triples, out_path)

    print(f"\n[convert] Actualiza el YAML:")
    print(f"  format: tsv")
    print(f"  path:   {out_path}")
    print(f"  has_header: false")
    return out_path


# ── Punto de entrada (elegido por config) ─────────────────────────────────────

def convert(cfg: dict):
    """Decide si usar split por bloques o conversión plana según el config."""
    fmt = cfg["graph"]["format"].lower()
    block_pred = cfg["graph"].get("block_predicate", "")
    block_val  = cfg["graph"].get("block_value", "")

    # TTL con bloques definidos → split 90/10 por incidencia
    if fmt == "ttl" and (block_pred or block_val):
        return convert_incidents(cfg)

    # Resto (TSV plano, PKL) → normalización simple
    return convert_flat(cfg)
