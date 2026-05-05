"""Punto de entrada único del pipeline.

Uso:
    python run.py --config configs/incidents.yaml --mode train
    python run.py --config configs/incidents.yaml --mode eval
    python run.py --config configs/incidents.yaml --mode train --only split

Modos:
  - train: io_graph → split → (kge_trainer.train_all_candidates → select_best)
  - eval:  imprime comparison.json
  - serve: bucle interactivo (Fase C, aún no implementado)

Flags:
  --only {split,kge}        ejecuta solo esa etapa del modo train
  --skip-rules              salta el minado de reglas (Fase B)
  --no-llm                  fuerza fallback de menú numerado
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from pipeline.io_graph import group_by_block, load_graph
from pipeline.split import run_split


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cmd_train(cfg: dict, args: argparse.Namespace) -> None:
    only = args.only

    print(f"[1/3] Cargando grafo: {cfg['graph']['path']}")
    triples = load_graph(cfg)
    print(f"      Tripletas: {len(triples):,}")

    print("[2/3] Agrupando por bloques")
    blocks = group_by_block(
        triples,
        block_predicate=cfg["graph"]["block_predicate"],
        block_value=cfg["graph"]["block_value"],
        prefixes=cfg["graph"].get("prefixes", {}),
    )
    print(f"      Bloques detectados: {len(blocks):,}")

    block_subjects = set(blocks.keys())
    block_triples_set = set()
    for ts in blocks.values():
        block_triples_set.update(ts)
    extra = [t for t in triples if t[0] not in block_subjects and t not in block_triples_set]

    print("[3/3] Split por bloques")
    splits = run_split(cfg, blocks, extra)
    print(
        f"      train={len(splits.train):,}  val={len(splits.val):,}  "
        f"test_kge={len(splits.test_kge):,}  test_system_blocks={len(splits.test_system):,}"
    )

    if only == "split":
        return

    if only in (None, "kge"):
        print("[KGE] Entrenando candidatos")
        from pipeline.kge_trainer import select_best, train_all_candidates

        results = train_all_candidates(cfg)
        select_best(results, cfg)
        from evaluation.eval_kge import print_comparison

        print_comparison(cfg)


def cmd_eval(cfg: dict, args: argparse.Namespace) -> None:
    from evaluation.eval_kge import print_comparison

    print_comparison(cfg)


def cmd_serve(cfg: dict, args: argparse.Namespace) -> None:
    print("Modo `serve` aún no implementado (llega en Fase C).", file=sys.stderr)
    sys.exit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description="KGE-Augmented LLM pipeline")
    parser.add_argument("--config", required=True, help="Ruta al YAML de configuración")
    parser.add_argument("--mode", choices=["train", "eval", "serve"], default="train")
    parser.add_argument("--only", choices=["split", "kge"], default=None)
    parser.add_argument("--skip-rules", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    if args.no_llm:
        cfg.setdefault("llm", {})["use_llm"] = False

    {
        "train": cmd_train,
        "eval": cmd_eval,
        "serve": cmd_serve,
    }[args.mode](cfg, args)


if __name__ == "__main__":
    main()
