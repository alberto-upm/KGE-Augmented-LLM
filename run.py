"""Punto de entrada único del pipeline.

Uso:
    python run.py --config configs/incidents.yaml --mode train
    python run.py --config configs/incidents.yaml --mode eval
    python run.py --config configs/incidents.yaml --mode train --only split

Modos:
  - train: io_graph → split → rule_mining → KGE candidatos → select_best → CBR
  - eval:  imprime comparison.json del KGE y, si existe, eval_system.json
  - serve: bucle interactivo (Fase C, aún no implementado)

Flags:
  --only {split,rules,kge,cbr}  ejecuta solo esa etapa del modo train
  --skip-rules                  salta el minado de reglas
  --no-llm                      fuerza fallback de menú numerado
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


def _load_blocks(cfg: dict):
    triples = load_graph(cfg)
    blocks = group_by_block(
        triples,
        block_predicate=cfg["graph"]["block_predicate"],
        block_value=cfg["graph"]["block_value"],
        prefixes=cfg["graph"].get("prefixes", {}),
    )
    block_subjects = set(blocks.keys())
    block_triples_set = set()
    for ts in blocks.values():
        block_triples_set.update(ts)
    extra = [t for t in triples if t[0] not in block_subjects and t not in block_triples_set]
    return triples, blocks, extra


def cmd_train(cfg: dict, args: argparse.Namespace) -> None:
    only = args.only

    print(f"[1/4] Cargando grafo: {cfg['graph']['path']}")
    _, blocks, extra = _load_blocks(cfg)
    print(f"      Bloques detectados: {len(blocks):,}")

    print("[2/4] Split por bloques")
    splits = run_split(cfg, blocks, extra)
    print(
        f"      train={len(splits.train):,}  val={len(splits.val):,}  "
        f"test_kge={len(splits.test_kge):,}  test_system_blocks={len(splits.test_system):,}"
    )
    if only == "split":
        return

    if only in (None, "rules") and not args.skip_rules:
        print("[3/4] Minado de reglas con AnyBURL")
        try:
            from pipeline.rule_mining import run_rule_mining

            run_rule_mining(cfg)
        except FileNotFoundError as e:
            print(f"      Aviso: {e}", file=sys.stderr)
            print("      Continuando sin reglas (cascada caerá a CBR/KGE).", file=sys.stderr)
    if only == "rules":
        return

    if only in (None, "kge"):
        print("[4/4] Entrenando candidatos KGE")
        from pipeline.kge_trainer import select_best, train_all_candidates

        results = train_all_candidates(cfg)
        select_best(results, cfg)
        from evaluation.eval_kge import print_comparison

        print_comparison(cfg)
    if only == "kge":
        return

    if only in (None, "cbr"):
        print("[CBR] Construyendo índice de casos")
        from pipeline.cbr import run_build_index

        # Reconstruimos `blocks` solo con los bloques que NO están en test_system,
        # para no contaminar el índice con casos que se usan en evaluación.
        train_blocks = {bid: triples for bid, triples in blocks.items() if bid not in splits.test_system}
        run_build_index(cfg, train_blocks)


def cmd_eval(cfg: dict, args: argparse.Namespace) -> None:
    from evaluation.eval_kge import print_comparison

    print_comparison(cfg)

    eval_system_path = Path(cfg["output"]["out_dir"]) / "eval_system.json"
    if eval_system_path.exists():
        import json

        report = json.loads(eval_system_path.read_text())
        from evaluation.eval_system import _print_report

        _print_report(report)
    else:
        print(
            f"\nNo existe {eval_system_path}. Para generarlo, ejecuta el sistema "
            "completo (Fase C) o llama a evaluation.eval_system.evaluate desde un script."
        )


def cmd_convert(cfg: dict, args: argparse.Namespace) -> None:
    """Extrae tripletas del grafo de entrada y las guarda como TSV."""
    from pipeline.extract_triples import convert

    out = convert(cfg)
    print(f"\nPróximo paso: actualiza el YAML con:")
    print(f"  format: tsv")
    print(f"  path:   {out}")
    print(f"  has_header: false")


def cmd_serve(cfg: dict, args: argparse.Namespace) -> None:
    """Bucle interactivo: carga el ganador KGE + CBR + reglas y crea una incidencia."""
    import json

    from pipeline.cbr import CBRIndex
    from pipeline.incident import Incident, load_property_specs
    from pipeline.inference import InferenceResources
    from pipeline.interaction import run_interaction_loop
    from pipeline.kge_trainer import PyKEENScorer, load_best_for_inference
    from pipeline.verbalizer import (
        LLMClient,
        collect_vocabulary,
        extract_from_free_text,
        finish,
        make_llm_ask,
        verbalize_props,
    )

    out_dir = Path(cfg["output"]["out_dir"])

    cbr_path = out_dir / "cbr_index.pkl"
    if not cbr_path.exists():
        print(f"No existe {cbr_path}. Ejecuta `--mode train` primero.", file=sys.stderr)
        sys.exit(2)
    cbr = CBRIndex.load(cbr_path)

    print("[serve] Cargando ganador KGE…")
    pipeline_result = load_best_for_inference(cfg)
    kge = PyKEENScorer(pipeline_result)

    rule_engine = None
    if not args.skip_rules:
        rules_path = Path(cfg["rules"]["out_dir"]) / "filtered.txt"
        if rules_path.exists():
            try:
                from pipeline.rule_mining import (
                    PyClauseRuleEngine,
                    _parse_anyburl_line,
                    load_into_pyclause,
                )

                with rules_path.open("r", encoding="utf-8") as f:
                    rules = [r for r in (_parse_anyburl_line(l) for l in f) if r is not None]
                loader, qa_handler = load_into_pyclause(rules, cfg)
                rule_engine = PyClauseRuleEngine(loader, qa_handler, rules)
            except Exception as e:
                print(f"[serve] Aviso: PyClause no disponible ({e}); cascada solo CBR/KGE.", file=sys.stderr)
        else:
            print(f"[serve] {rules_path} no existe; cascada solo CBR/KGE.")

    res = InferenceResources(rule_engine=rule_engine, kge=kge, cbr=cbr, cfg=cfg)
    specs = load_property_specs(cfg)
    vocab_by_prop = collect_vocabulary(cbr.cases, specs)

    llm = LLMClient(cfg)
    incident = Incident()

    print("\n¿Quieres pegar primero un texto libre describiendo la incidencia? (Enter para saltar)")
    raw_text = input("> ").strip()
    if raw_text:
        prefilled = extract_from_free_text(raw_text, vocab_by_prop)
        for prop, value in prefilled.items():
            from pipeline.traceability import USER

            incident.update(prop, value, USER)
            print(f"  · pre-rellenado {prop} = {value}")

    llm_ask = make_llm_ask(llm) if llm.use_llm else None
    run_interaction_loop(incident, res, specs, vocab_by_prop, llm_ask=llm_ask)

    summary = verbalize_props(incident, llm)
    print("\n" + "=" * 60)
    print(summary)
    print("=" * 60)

    confirm = input("\n¿Guardar la incidencia? [s/n]: ").strip().lower()
    if confirm in ("", "s", "si", "sí", "y", "yes"):
        with open(Path(cfg["kge"]["best_model_path"]) / "selected.json", "r") as f:
            kge_name = json.load(f).get("model", "?")
        finish(incident, kge_name, summary, Path(cfg["output"]["jsonl"]))
        print(f"Guardada en {cfg['output']['jsonl']}")
    else:
        print("No se guardó.")


def main() -> None:
    parser = argparse.ArgumentParser(description="KGE-Augmented LLM pipeline")
    parser.add_argument("--config", required=True, help="Ruta al YAML de configuración")
    parser.add_argument("--mode", choices=["train", "eval", "serve", "convert"], default="train")
    parser.add_argument("--only", choices=["split", "rules", "kge", "cbr"], default=None)
    parser.add_argument("--skip-rules", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    if args.no_llm:
        cfg.setdefault("llm", {})["use_llm"] = False

    {
        "train":   cmd_train,
        "eval":    cmd_eval,
        "serve":   cmd_serve,
        "convert": cmd_convert,
    }[args.mode](cfg, args)


if __name__ == "__main__":
    main()
