"""Minado de reglas con AnyBURL y carga en PyClause.

AnyBURL es una herramienta Java externa: se invoca por subprocess con su JAR.
El binario debe estar en `cfg.rules.anyburl_jar`. PyClause envuelve la
inferencia de reglas en Python (`pip install pyclause`).
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Rule:
    """Regla en formato AnyBURL.

    `head` y `body` se mantienen como strings literales (en el formato que
    devuelve AnyBURL) para alimentar PyClause sin reformatear.
    """

    text: str            # línea original "support\tconfidence\thead <= body"
    support: int
    confidence: float
    head: str
    body: str

    @property
    def length(self) -> int:
        return self.body.count(",") + 1 if self.body.strip() else 0


_LINE_RE = re.compile(r"^(\d+)\s+(\d+)\s+([0-9.]+)\s+(.+?)\s+<=\s+(.*)$")


def _parse_anyburl_line(line: str) -> Rule | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    m = _LINE_RE.match(line)
    if not m:
        return None
    support_pred, support_total, conf, head, body = m.groups()
    return Rule(
        text=line,
        support=int(support_pred),
        confidence=float(conf),
        head=head.strip(),
        body=body.strip(),
    )


def _write_anyburl_config(train_tsv: Path, out_dir: Path, max_rule_length: int) -> Path:
    """Escribe el `config-learn.properties` que AnyBURL espera.

    Mantenemos los tiempos de ejecución bajos por defecto; el usuario puede
    ajustarlos editando el fichero generado.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "config-learn.properties"
    cfg_path.write_text(
        "\n".join(
            [
                f"PATH_TRAINING       = {train_tsv}",
                f"PATH_OUTPUT         = {out_dir}/rules",
                f"SNAPSHOTS_AT        = 10,50,100",
                f"MAX_LENGTH_CYCLIC   = {max_rule_length}",
                f"MAX_LENGTH_ACYCLIC  = {max_rule_length}",
                f"WORKER_THREADS      = 4",
                "",
            ]
        )
    )
    return cfg_path


def mine_rules(cfg: dict) -> list[Rule]:
    rules_cfg = cfg["rules"]
    splits_dir = Path(cfg["split"]["out_dir"])
    train_tsv = splits_dir / "train.tsv"
    if not train_tsv.exists():
        raise FileNotFoundError(f"No existe {train_tsv}. Ejecuta el split primero.")

    out_dir = Path(rules_cfg["out_dir"])
    jar_path = Path(rules_cfg["anyburl_jar"])
    if not jar_path.exists():
        raise FileNotFoundError(
            f"No se encuentra AnyBURL.jar en {jar_path}. Descárgalo y colócalo ahí.\n"
            "Más info: https://web.informatik.uni-mannheim.de/AnyBURL/"
        )

    cfg_path = _write_anyburl_config(train_tsv, out_dir, rules_cfg.get("max_rule_length", 3))

    print(f"[rules] Lanzando AnyBURL: {jar_path}")
    subprocess.run(
        ["java", "-Xmx12G", "-cp", str(jar_path), "de.unima.ki.anyburl.LearnReinforced", str(cfg_path)],
        check=True,
    )

    raw_files = sorted(out_dir.glob("rules-*"))
    if not raw_files:
        raw_files = sorted(out_dir.glob("rules*"))
    if not raw_files:
        raise FileNotFoundError(f"AnyBURL no produjo ficheros de reglas en {out_dir}")
    raw_path = raw_files[-1]
    print(f"[rules] Leyendo reglas de {raw_path}")

    rules: list[Rule] = []
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            r = _parse_anyburl_line(line)
            if r is not None:
                rules.append(r)
    print(f"[rules] AnyBURL devolvió {len(rules):,} reglas en bruto")
    return rules


def filter_rules(rules: list[Rule], min_support: int, confidence_threshold: float) -> list[Rule]:
    out = [r for r in rules if r.support >= min_support and r.confidence >= confidence_threshold]
    print(
        f"[rules] Tras filtro support≥{min_support} y conf≥{confidence_threshold}: "
        f"{len(out):,} reglas"
    )
    return out


def save_filtered(rules: list[Rule], cfg: dict) -> Path:
    out_dir = Path(cfg["rules"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "filtered.txt"
    with path.open("w", encoding="utf-8") as f:
        for r in rules:
            f.write(r.text + "\n")
    return path


def load_into_pyclause(filtered_rules: list["Rule"], cfg: dict):
    """Carga las reglas filtradas en PyClause para inferencia.

    PyClause espera:
      - data: lista de tripletas (s, p, o) o ruta a fichero TSV
      - rules: lista de strings "head <= body"
      - stats: lista de [num_preds, support] por regla

    Devuelve una tupla (loader, qa_handler) lista para responder consultas.
    """
    try:
        from c_clause import Loader, QAHandler  # type: ignore
        from clause import Options  # type: ignore
    except ImportError as e:
        raise ImportError(
            "PyClause no está instalado. Instala con:\n"
            "  pip install git+https://github.com/symbolic-kg/PyClause.git"
        ) from e

    splits_dir = Path(cfg["split"]["out_dir"])
    train_tsv = str(splits_dir / "train.tsv")

    # Formateamos reglas y estadísticas para PyClause:
    # stats[i] = [total_groundings, correct_predictions]
    # confidence = correct_predictions / total_groundings
    rule_strings = [f"{r.head} <= {r.body}" if r.body else r.head for r in filtered_rules]
    stats = [
        [r.support, max(1, int(r.support * r.confidence))]
        for r in filtered_rules
    ]

    opts = Options()
    opts.set("qa_handler.aggregation_function", "noisyor")

    loader = Loader(options=opts.get("loader"))
    loader.load_data(data=train_tsv)
    loader.load_rules(rules=rule_strings, stats=stats)

    qa = QAHandler(options=opts.get("qa_handler"))
    return loader, qa


def run_rule_mining(cfg: dict) -> Path:
    rules = mine_rules(cfg)
    filtered = filter_rules(
        rules,
        min_support=cfg["rules"]["min_support"],
        confidence_threshold=cfg["rules"]["confidence_threshold"],
    )
    return save_filtered(filtered, cfg)


class PyClauseRuleEngine:
    """Adaptador para el protocolo `RuleEngine` de inference.py.

    Usa QAHandler de PyClause con dirección "tail": dada la consulta
    (known_value, target_prop, ?), devuelve [(valor, confianza, rule_id)].
    """

    def __init__(self, loader, qa_handler, rules: list[Rule]):
        self.loader = loader
        self.qa = qa_handler
        # Mapa relación → regla de mayor confianza, para poder devolver rule_id.
        self._top_rule_by_rel: dict[str, Rule] = {}
        for r in rules:
            prev = self._top_rule_by_rel.get(r.head)
            if prev is None or r.confidence > prev.confidence:
                self._top_rule_by_rel[r.head] = r

    def query(
        self,
        known_props: dict[str, str],
        target_prop: str,
        confidence_threshold: float,
    ) -> list[tuple[str, float, str]]:
        """Devuelve [(valor, confianza, rule_id)] o []."""
        # Construimos consultas (head_value, target_prop) para cada valor conocido.
        queries = [(v, target_prop) for v in known_props.values()]
        if not queries:
            return []

        try:
            self.qa.calculate_answers(queries=queries, loader=self.loader, direction="tail")
            raw = self.qa.get_answers(as_string=True)
        except Exception:
            return []

        best_rule = self._top_rule_by_rel.get(target_prop)
        rid = best_rule.text if best_rule else "unknown"

        seen: dict[str, float] = {}
        for answers_for_query in raw:
            for value, confidence in answers_for_query:
                conf = float(confidence)
                if conf >= confidence_threshold:
                    if conf > seen.get(value, -1.0):
                        seen[value] = conf

        results = [(v, c, rid) for v, c in seen.items()]
        results.sort(key=lambda x: x[1], reverse=True)
        return results
