"""Evaluación del sistema completo sobre `test_system`.

Para cada bloque de test, oculta cada propiedad activa una a una y mide:
- accuracy del top-1
- coverage del top-K (¿está la verdadera entre las K?)
- desglose por `source`: qué fuente acertó (RULE/CBR/KGE/CBR+KGE).

El test_system se carga desde el TSV multi-columna que escribe `split.py`:
    block_id  s  p  o
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from pipeline.cbr import CBRIndex, block_to_props
from pipeline.incident import active_specs, load_property_specs
from pipeline.inference import InferenceResources, recommend_property


def load_test_system_blocks(path: Path) -> dict[str, list[tuple[str, str, str]]]:
    blocks: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            bid, s, p, o = parts
            blocks[bid].append((s, p, o))
    return dict(blocks)


def evaluate(cfg: dict, res: InferenceResources) -> dict:
    splits_dir = Path(cfg["split"]["out_dir"])
    test_path = splits_dir / "test_system.tsv"
    if not test_path.exists():
        raise FileNotFoundError(f"No existe {test_path}. Ejecuta el split primero.")

    test_blocks = load_test_system_blocks(test_path)
    specs = load_property_specs(cfg)
    actives = active_specs(specs)

    top_k_final = cfg["fusion"].get("top_k_final", 5)

    per_prop: dict[str, dict] = {
        s.uri: {
            "n": 0,
            "top1_correct": 0,
            "topk_correct": 0,
            "by_source": Counter(),
            "by_source_correct": Counter(),
        }
        for s in actives
    }
    overall_top1 = 0
    overall_topk = 0
    overall_n = 0

    for bid, triples in test_blocks.items():
        full_props = block_to_props(triples)
        for spec in actives:
            true_value = full_props.get(spec.uri)
            if true_value is None:
                continue
            known = {p: v for p, v in full_props.items() if p != spec.uri}

            suggestions = recommend_property(known, spec.uri, res)
            if not suggestions:
                continue

            stats = per_prop[spec.uri]
            stats["n"] += 1
            overall_n += 1

            top1 = suggestions[0]
            stats["by_source"][top1.source] += 1
            if top1.value == true_value:
                stats["top1_correct"] += 1
                stats["by_source_correct"][top1.source] += 1
                overall_top1 += 1

            topk_values = {s.value for s in suggestions[:top_k_final]}
            if true_value in topk_values:
                stats["topk_correct"] += 1
                overall_topk += 1

    report = {
        "n_blocks": len(test_blocks),
        "top_k": top_k_final,
        "overall": {
            "n": overall_n,
            "top1_accuracy": overall_top1 / overall_n if overall_n else 0.0,
            "topk_coverage": overall_topk / overall_n if overall_n else 0.0,
        },
        "by_property": {
            uri: {
                "n": s["n"],
                "top1_accuracy": s["top1_correct"] / s["n"] if s["n"] else 0.0,
                "topk_coverage": s["topk_correct"] / s["n"] if s["n"] else 0.0,
                "by_source_count": dict(s["by_source"]),
                "by_source_correct": dict(s["by_source_correct"]),
            }
            for uri, s in per_prop.items()
        },
    }

    out_dir = Path(cfg["output"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_system.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    _print_report(report)
    print(f"\nReporte completo en {out_path}")
    return report


def _print_report(report: dict) -> None:
    print("\n=== Evaluación del sistema completo ===")
    o = report["overall"]
    print(f"Bloques de test: {report['n_blocks']}   intentos: {o['n']}")
    print(f"Top-1 accuracy:  {o['top1_accuracy']:.4f}")
    print(f"Top-{report['top_k']} coverage: {o['topk_coverage']:.4f}\n")

    print(f"{'Propiedad':50s}  {'n':>5}  {'top1':>6}  {'topK':>6}  fuente_top1")
    print("-" * 100)
    for uri, s in report["by_property"].items():
        sources = ", ".join(f"{k}:{v}" for k, v in s["by_source_count"].items())
        print(
            f"{uri[:50]:50s}  {s['n']:>5}  "
            f"{s['top1_accuracy']:>6.4f}  {s['topk_coverage']:>6.4f}  {sources}"
        )
