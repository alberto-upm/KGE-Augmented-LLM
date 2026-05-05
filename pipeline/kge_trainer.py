"""Entrenamiento de modelos KGE con PyKEEN y selección del mejor.

Entrena cada candidato declarado en el YAML, mide MRR/Hits y persiste:
  - <best_model_path>/<model_name>/   (modelo ganador completo)
  - <comparison_path>                  (JSON con la tabla comparativa)
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainResult:
    name: str
    metrics: dict[str, float]
    pipeline_dir: Path
    pipeline_result: Any  # pykeen PipelineResult


def _train_one(
    model_name: str,
    model_kwargs: dict,
    shared_cfg: dict,
    train_tsv: Path,
    val_tsv: Path,
    test_tsv: Path,
    out_dir: Path,
) -> TrainResult:
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory

    train = TriplesFactory.from_path(str(train_tsv), create_inverse_triples=False)
    val = TriplesFactory.from_path(
        str(val_tsv),
        entity_to_id=train.entity_to_id,
        relation_to_id=train.relation_to_id,
        create_inverse_triples=False,
    )
    test = TriplesFactory.from_path(
        str(test_tsv),
        entity_to_id=train.entity_to_id,
        relation_to_id=train.relation_to_id,
        create_inverse_triples=False,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    result = pipeline(
        training=train,
        validation=val,
        testing=test,
        model=model_name,
        model_kwargs={"embedding_dim": model_kwargs["embedding_dim"]},
        training_kwargs={
            "num_epochs": model_kwargs["epochs"],
            "batch_size": shared_cfg.get("batch_size", 1024),
        },
        optimizer=shared_cfg.get("optimizer", "adam"),
        optimizer_kwargs={"lr": shared_cfg.get("learning_rate", 0.001)},
        loss=shared_cfg.get("loss", "nssa"),
        random_seed=shared_cfg.get("random_seed", 42),
    )
    result.save_to_directory(str(out_dir))

    metrics = {
        "mrr": float(result.metric_results.get_metric("both.realistic.inverse_harmonic_mean_rank")),
        "hits_at_1": float(result.metric_results.get_metric("both.realistic.hits_at_1")),
        "hits_at_3": float(result.metric_results.get_metric("both.realistic.hits_at_3")),
        "hits_at_10": float(result.metric_results.get_metric("both.realistic.hits_at_10")),
    }
    return TrainResult(name=model_name, metrics=metrics, pipeline_dir=out_dir, pipeline_result=result)


def train_all_candidates(cfg: dict) -> list[TrainResult]:
    splits_dir = Path(cfg["split"]["out_dir"])
    train_tsv = splits_dir / "train.tsv"
    val_tsv = splits_dir / "val.tsv"
    test_tsv = splits_dir / "test_kge.tsv"

    kge_cfg = cfg["kge"]
    shared = kge_cfg.get("shared", {})
    base_out = Path(kge_cfg["best_model_path"]).parent / "candidates"

    results: list[TrainResult] = []
    for cand in kge_cfg["candidates"]:
        name = cand["model"]
        out_dir = base_out / name
        print(f"[KGE] Entrenando {name} → {out_dir}")
        res = _train_one(
            model_name=name,
            model_kwargs={"embedding_dim": cand["embedding_dim"], "epochs": cand["epochs"]},
            shared_cfg=shared,
            train_tsv=train_tsv,
            val_tsv=val_tsv,
            test_tsv=test_tsv,
            out_dir=out_dir,
        )
        results.append(res)
    return results


_METRIC_KEYS = {"mrr", "hits_at_1", "hits_at_3", "hits_at_10"}


def select_best(results: list[TrainResult], cfg: dict) -> TrainResult:
    metric = cfg["kge"].get("selection_metric", "mrr")
    if metric not in _METRIC_KEYS:
        raise ValueError(f"selection_metric debe ser uno de {_METRIC_KEYS}, recibido {metric}")

    best = max(results, key=lambda r: r.metrics[metric])

    best_path = Path(cfg["kge"]["best_model_path"])
    if best_path.exists():
        shutil.rmtree(best_path)
    shutil.copytree(best.pipeline_dir, best_path)
    (best_path / "selected.json").write_text(
        json.dumps({"model": best.name, "metric": metric, "metrics": best.metrics}, indent=2)
    )

    comparison_path = Path(cfg["kge"]["comparison_path"])
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(
        json.dumps(
            {
                "selection_metric": metric,
                "winner": best.name,
                "candidates": [{"name": r.name, "metrics": r.metrics} for r in results],
            },
            indent=2,
        )
    )
    print(f"[KGE] Ganador por {metric}: {best.name} (valor {best.metrics[metric]:.4f})")
    return best


def load_best_for_inference(cfg: dict):
    """Carga el modelo ganador (PipelineResult restaurado) para inferencia."""
    from pykeen.pipeline import PipelineResult

    best_path = Path(cfg["kge"]["best_model_path"])
    return PipelineResult.from_directory(str(best_path))
