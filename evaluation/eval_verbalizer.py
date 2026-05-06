"""Evaluación del verbalizador.

Dos métricas:

1. **Acierto de extracción**: simulando respuestas del usuario en lenguaje
   natural ("la opción 2", "sí", "el primero"), comprobamos que `_llm_extract`
   las mapea al ID correcto del top-K. Si falla, la cascada anti-alucinación
   debe rechazar y volver a preguntar; aquí medimos el ratio que el LLM
   resuelve sin fricciones.

2. **BERTScore del resumen final**: contra resúmenes humanos de referencia
   (ground-truth opcional). Si no hay GT, basta el reporte de extracción.
"""
from __future__ import annotations

import json
from pathlib import Path

from pipeline.interaction import parse_user_input
from pipeline.traceability import KGE, Suggestion
from pipeline.verbalizer import LLMClient, verbalize_props


# ---------- Acierto de extracción ----------

_SYNTHETIC_RESPONSES = [
    # (respuesta del usuario, índice esperado del top-K, kind esperado)
    ("1", 0, "pick"),
    ("3", 2, "pick"),
    ("si", 0, "pick"),       # acepta el highlighted
    ("yes", 0, "pick"),
    ("no", None, "next"),
    ("skip", None, "skip"),
    ("texto-fuera-vocabulario-xyz", None, "invalid"),
]


def evaluate_extraction(top_k_size: int = 5) -> dict:
    fake_top_k = [
        Suggestion(value=f"option_{i}", source=KGE, kge_score=1.0 - i * 0.1, rank=i + 1)
        for i in range(top_k_size)
    ]
    vocab = {s.value for s in fake_top_k}

    n_ok = 0
    failures = []
    for raw, exp_idx, exp_kind in _SYNTHETIC_RESPONSES:
        decision = parse_user_input(raw, fake_top_k, highlighted=0, vocab=vocab)
        ok = decision.kind == exp_kind and (exp_idx is None or decision.index == exp_idx)
        if ok:
            n_ok += 1
        else:
            failures.append(
                {
                    "input": raw,
                    "expected_kind": exp_kind,
                    "expected_index": exp_idx,
                    "actual_kind": decision.kind,
                    "actual_index": decision.index,
                }
            )
    return {
        "n": len(_SYNTHETIC_RESPONSES),
        "correct": n_ok,
        "accuracy": n_ok / len(_SYNTHETIC_RESPONSES),
        "failures": failures,
    }


# ---------- BERTScore del resumen ----------

def evaluate_summary_quality(predictions: list[str], references: list[str]) -> dict:
    """Mide BERTScore F1 medio entre resúmenes generados y referencias humanas."""
    if not predictions or not references:
        return {"n": 0, "bertscore_f1": None}
    if len(predictions) != len(references):
        raise ValueError("predictions y references deben tener la misma longitud")

    try:
        from bert_score import score  # type: ignore
    except ImportError as e:
        raise ImportError("Instala bert-score para esta métrica.") from e

    _, _, f1 = score(predictions, references, lang="es", verbose=False)
    return {"n": len(predictions), "bertscore_f1": float(f1.mean().item())}


def run(cfg: dict, gt_path: Path | None = None) -> dict:
    """Punto de entrada del modo eval-verbalizer.

    Si `gt_path` apunta a un JSONL con `{"props": ..., "resumen_humano": ...}`,
    carga referencias y mide BERTScore. Si no, solo evalúa extracción.
    """
    out_dir = Path(cfg["output"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    extraction = evaluate_extraction()
    report: dict = {"extraction": extraction}

    if gt_path and gt_path.exists():
        from pipeline.incident import Incident

        llm = LLMClient(cfg)
        preds, refs = [], []
        with gt_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if "resumen_humano" not in row or "props" not in row:
                    continue
                inc = Incident(props=row["props"])
                preds.append(verbalize_props(inc, llm))
                refs.append(row["resumen_humano"])
        report["summary"] = evaluate_summary_quality(preds, refs)

    out_path = out_dir / "eval_verbalizer.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReporte completo en {out_path}")
    return report
