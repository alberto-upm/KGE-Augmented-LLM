"""Evaluación del KGE: MRR y Hits@k. Lee la tabla comparativa y la imprime."""
from __future__ import annotations

import json
from pathlib import Path


def print_comparison(cfg: dict) -> dict:
    path = Path(cfg["kge"]["comparison_path"])
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta primero `run.py --mode train` para generar la tabla."
        )
    data = json.loads(path.read_text())
    metric = data["selection_metric"]
    winner = data["winner"]

    rows = data["candidates"]
    keys = ["mrr", "hits_at_1", "hits_at_3", "hits_at_10"]

    name_w = max(len("Modelo"), max(len(r["name"]) for r in rows))
    header = f"{'Modelo'.ljust(name_w)}  " + "  ".join(k.rjust(10) for k in keys)
    print(header)
    print("-" * len(header))
    for r in rows:
        marker = "  ← ganador" if r["name"] == winner else ""
        line = f"{r['name'].ljust(name_w)}  " + "  ".join(f"{r['metrics'][k]:10.4f}" for k in keys)
        print(line + marker)
    print(f"\nMétrica de selección: {metric}")
    return data


def evaluate_from_results(results, cfg: dict) -> dict:
    """Atajo cuando se llama justo después de `train_all_candidates` sin reentrenar."""
    return {
        "selection_metric": cfg["kge"].get("selection_metric", "mrr"),
        "candidates": [{"name": r.name, "metrics": r.metrics} for r in results],
    }
