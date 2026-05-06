"""Extracción de tripletas y división por bloques de incidencia.

Para grafos con bloques identificables (incidencias):
  - Lee el TTL con rdflib
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
from collections import defaultdict
from pathlib import Path

from .io_graph import Triple, _shorten


# ── Utilidades TTL ────────────────────────────────────────────────────────────

def _expand_curie(curie: str, prefixes: dict[str, str]) -> str:
    if ":" in curie and not curie.startswith("http"):
        prefix, local = curie.split(":", 1)
        if prefix in prefixes:
            return prefixes[prefix] + local
    return curie


def _extract_label(node, cfg_prefixes: dict[str, str]) -> str:
    s = str(node)
    if "#" in s:
        return s.split("#")[-1]
    return s.split("/")[-1]


def _load_ttl_graph(path: Path):
    from rdflib import Graph  # type: ignore

    g = Graph()
    g.parse(str(path), format="turtle")
    return g


def _split_ttl_into_chunks(path: Path, blocks_per_chunk: int = 1000) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    prefix_lines: list[str] = []
    blocks: list[str] = []
    current: list[str] = []
    seen_data = False

    for raw_line in lines:
        stripped = raw_line.strip()
        if not seen_data and (not stripped or stripped.startswith("@prefix") or stripped.startswith("#")):
            if stripped.startswith("@prefix"):
                prefix_lines.append(stripped)
            continue

        if not stripped:
            if current:
                blocks.append("\n".join(current))
                current = []
                seen_data = True
            continue

        seen_data = True
        current.append(raw_line)

    if current:
        blocks.append("\n".join(current))

    prefix_blob = "\n".join(prefix_lines)
    chunks: list[str] = []
    for i in range(0, len(blocks), blocks_per_chunk):
        chunk_blocks = blocks[i : i + blocks_per_chunk]
        chunk = prefix_blob + "\n\n" + "\n\n".join(chunk_blocks) + "\n"
        chunks.append(chunk)
    return chunks


def extract_all_triples(g, cfg_prefixes: dict[str, str], *, verbose: bool = True) -> list[Triple]:
    """Devuelve todas las tripletas útiles del grafo, omitiendo rdf:type."""
    from rdflib.namespace import RDF  # type: ignore

    triples: list[Triple] = []
    skipped = 0
    for s, p, o in g:
        if p == RDF.type:
            skipped += 1
            continue
        head = _extract_label(s, cfg_prefixes)
        relation = _extract_label(p, cfg_prefixes)
        tail = _extract_label(o, cfg_prefixes)
        triples.append((head, relation, tail))
    if verbose:
        print(f"      {len(triples):,} tripletas extraídas  ({skipped:,} rdf:type omitidas)")
    return triples


def _incident_subjects_from_graph(g, block_predicate: str, block_value: str, cfg_prefixes: dict[str, str]) -> set[str]:
    from rdflib import URIRef  # type: ignore
    from rdflib.namespace import RDF  # type: ignore

    subjects: set[str] = set()
    pred_uri = _expand_curie(block_predicate, cfg_prefixes) if block_predicate else ""
    value_uri = _expand_curie(block_value, cfg_prefixes) if block_value else ""

    for s, p, o in g:
        if block_predicate:
            if pred_uri:
                if p != URIRef(pred_uri):
                    continue
            elif _extract_label(p, cfg_prefixes) != block_predicate:
                continue
        if block_value:
            if value_uri:
                if str(o) != value_uri:
                    continue
            elif _extract_label(o, cfg_prefixes) != block_value:
                continue
        subjects.add(_extract_label(s, cfg_prefixes))

    return subjects


def _group_triples_by_subject(triples: list[Triple]) -> dict[str, list[Triple]]:
    blocks: dict[str, list[Triple]] = defaultdict(list)
    for s, p, o in triples:
        blocks[s].append((s, p, o))
    return dict(blocks)


def _load_incident_data_ttl(
    path: Path,
    cfg_prefixes: dict[str, str],
    block_predicate: str,
    block_value: str,
) -> tuple[list[Triple], set[str]]:
    from rdflib import Graph  # type: ignore

    try:
        chunks = _split_ttl_into_chunks(path)
    except Exception as e:
        raise RuntimeError(f"No se pudo preparar el TTL por bloques desde {path}. Error: {e}") from e

    all_triples: list[Triple] = []
    incident_subjects: set[str] = set()
    total_loaded = 0
    total_extracted = 0
    total_skipped_type = 0

    try:
        for idx, chunk in enumerate(chunks, start=1):
            g = Graph()
            try:
                g.parse(data=chunk, format="turtle")
            except Exception as e:
                raise RuntimeError(
                    f"rdflib no pudo parsear el bloque/chunk {idx}/{len(chunks)} de {path}. "
                    f"Error original: {e}"
                ) from e
            total_loaded += len(g)
            chunk_triples = extract_all_triples(g, cfg_prefixes, verbose=False)
            all_triples.extend(chunk_triples)
            total_extracted += len(chunk_triples)
            total_skipped_type += len(g) - len(chunk_triples)
            incident_subjects.update(
                _incident_subjects_from_graph(g, block_predicate, block_value, cfg_prefixes)
            )
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "No se pudo leer el TTL porque `rdflib` no está instalado. "
            "Instálalo en el entorno activo para ejecutar la conversión."
        ) from e

    print(f"[convert] {total_loaded:,} tripletas cargadas.")
    print(f"      {total_extracted:,} tripletas extraídas  ({total_skipped_type:,} rdf:type omitidas)")
    print(f"[convert] Incidencias únicas: {len(incident_subjects):,}")
    return all_triples, incident_subjects


def _split_by_incident_ids(
    triples: list[Triple],
    incident_ids: set[str],
    train_ratio: float,
    seed: int,
) -> tuple[list[Triple], list[Triple], dict[str, list[Triple]], dict[str, list[Triple]]]:
    rng = random.Random(seed)

    ordered_incidents = sorted(incident_ids)
    rng.shuffle(ordered_incidents)

    n_total = len(ordered_incidents)
    n_train = int(n_total * train_ratio)
    train_ids = set(ordered_incidents[:n_train])
    test_ids = set(ordered_incidents[n_train:])

    print(
        f"[convert] Incidencias  →  train: {len(train_ids):,}  "
        f"test: {len(test_ids):,}"
    )

    train_triples: list[Triple] = []
    test_triples: list[Triple] = []
    non_incident = 0

    for triple in triples:
        head = triple[0]
        if head in train_ids:
            train_triples.append(triple)
        elif head in test_ids:
            test_triples.append(triple)
        else:
            train_triples.append(triple)
            non_incident += 1

    if non_incident:
        print(f"[convert] Tripletas auxiliares (no-incidencia) añadidas a train: {non_incident:,}")

    print(
        f"[convert] Tripletas  →  train: {len(train_triples):,}  "
        f"test: {len(test_triples):,}"
    )

    train_blocks_all = _group_triples_by_subject(train_triples)
    test_blocks_all = _group_triples_by_subject(test_triples)
    train_blocks = {bid: train_blocks_all[bid] for bid in train_ids if bid in train_blocks_all}
    test_blocks = {bid: test_blocks_all[bid] for bid in test_ids if bid in test_blocks_all}

    return train_triples, test_triples, train_blocks, test_blocks


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
    block_pred  = graph_cfg.get("block_predicate", "")
    block_val   = graph_cfg.get("block_value", "")
    all_triples, incident_ids = _load_incident_data_ttl(src_path, cfg_pfx, block_pred, block_val)
    train_ratio = 1.0 - test_ratio
    train_triples, _test_triples, train_blocks, test_blocks = _split_by_incident_ids(
        all_triples,
        incident_ids,
        train_ratio=train_ratio,
        seed=seed,
    )

    _write_3col(train_triples, train_triples_path)
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
