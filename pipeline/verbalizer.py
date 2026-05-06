"""Capa LLM: pregunta natural, extracción anti-alucinación, verbalización final.

El LLM:
  - solo formula preguntas (envuelve el menú top-K),
  - solo extrae IDs del top-K (nunca inventa valores),
  - genera el resumen final con plantilla.

Cuando `use_llm=false`, todas las funciones caen al menú numerado de
`interaction.render_menu` y al parser estructurado.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .incident import Incident, PropertySpec
from .interaction import render_menu
from .traceability import Suggestion


# ---------- Cliente LLM (vLLM/OpenAI compatible) ----------

class LLMClient:
    """Wrapper minimal sobre el endpoint OpenAI-compatible de vLLM."""

    def __init__(self, cfg: dict):
        self.cfg = cfg["llm"]
        self.use_llm = bool(self.cfg.get("use_llm", True))
        self.model = self.cfg.get("model", "")
        self.endpoint = self.cfg.get("endpoint", "http://localhost:8000/v1")
        self.temperature = float(self.cfg.get("temperature", 0.0))

    def chat(self, system: str, user: str) -> str:
        if not self.use_llm:
            return ""
        # Importación perezosa para no exigir openai si el usuario no lo usa.
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Para hablar con vLLM hace falta `pip install openai`."
            ) from e

        client = OpenAI(base_url=self.endpoint, api_key="EMPTY")
        resp = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


# ---------- Pre-llenado opcional desde texto libre ----------

def extract_from_free_text(
    text: str,
    vocab_by_prop: dict[str, set[str]],
    label_index: dict[str, str] | None = None,
) -> dict[str, str]:
    """Pre-rellena propiedades si encuentra valores del vocabulario en `text`.

    Lookup directo: para cada propiedad, si algún valor del vocabulario aparece
    como substring (case-insensitive) en el texto libre, lo asigna. Si más
    adelante quieres usar GLiNER2 o spaCy, este es el sitio para sustituir
    la heurística.
    """
    found: dict[str, str] = {}
    text_low = text.lower()
    for prop, vocab in vocab_by_prop.items():
        for value in vocab:
            label = (label_index or {}).get(value, value).lower()
            needle = label.split(":")[-1].lower()
            if needle and needle in text_low:
                found[prop] = value
                break
    return found


# ---------- Pregunta natural envolviendo el menú top-K ----------

_ASK_SYSTEM = (
    "Eres un asistente que ayuda al usuario a completar una nueva incidencia. "
    "Recibes una propiedad pendiente y un menú con candidatos numerados. "
    "Tu trabajo es formular UNA pregunta natural breve para el usuario, "
    "explicando qué se le pide y mencionando 2-3 candidatos como referencia. "
    "NO inventes valores nuevos: el usuario debe responder con un número del "
    "menú, 'si'/'no'/'skip' o un valor del vocabulario."
)


def _llm_ask(spec: PropertySpec, suggestions: list[Suggestion], incident: Incident, highlighted: int, llm: LLMClient) -> str:
    menu = render_menu(spec.label, suggestions, highlighted)
    if not llm.use_llm:
        return menu

    known = ", ".join(f"{k.split(':')[-1]}={v.split(':')[-1]}" for k, v in incident.known().items()) or "ninguno"
    user = (
        f"Estado parcial conocido: {known}.\n"
        f"Propiedad pendiente: {spec.label} ({spec.uri}).\n\n"
        f"Menú a presentar:\n{menu}\n\n"
        f"Genera la pregunta natural breve y al final pega el menú tal cual."
    )
    natural = llm.chat(_ASK_SYSTEM, user).strip()
    if not natural:
        return menu
    return f"{natural}\n\n{menu}"


# ---------- Verbalización final ----------

_DEFAULT_TEMPLATE = (
    "Se va a crear una nueva incidencia para el cliente {cliente} "
    "con el usuario {usuario} de tipo «{tipo}» donde el origen de la incidencia "
    "será «{origen}», el grupo de soporte «{grupo}» y el técnico asignado «{tecnico}»."
)

_TEMPLATE_SLOTS = {
    "cliente": "repcon:int_hasCustomer",
    "usuario": "repcon:hasUser",
    "tipo": "repcon:hasTypeInc",
    "origen": "repcon:incident_hasOrigin",
    "grupo": "repcon:hasSupportGroup",
    "tecnico": "repcon:hasTechnician",
    "categoria": "repcon:hasSupportCategory",
}


def _short(uri: str | None) -> str:
    if not uri:
        return "sin asignar"
    return uri.split(":")[-1]


def _generic_template(incident: Incident) -> str:
    pairs = [
        f"{prop.split(':')[-1]} = {_short(value)}"
        for prop, value in incident.known().items()
    ]
    return "Nueva entidad con los siguientes valores: " + "; ".join(pairs) + "."


def verbalize_props(incident: Incident, llm: LLMClient | None = None) -> str:
    has_incidents_slots = any(uri in incident.props for uri in _TEMPLATE_SLOTS.values())
    if has_incidents_slots:
        slots = {key: _short(incident.props.get(uri)) for key, uri in _TEMPLATE_SLOTS.items()}
        plantilla = _DEFAULT_TEMPLATE.format(**slots)
    else:
        plantilla = _generic_template(incident)

    if llm is None or not llm.use_llm:
        return plantilla

    system = (
        "Reformula la siguiente frase en español natural y breve, "
        "manteniendo todos los datos exactos y sin añadir nada nuevo."
    )
    natural = llm.chat(system, plantilla).strip()
    return natural or plantilla


def finish(incident: Incident, kge_model_name: str, summary: str, jsonl_path: Path) -> dict:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kge_model": kge_model_name,
        "props": dict(incident.props),
        "sources": dict(incident.sources),
        "skipped": sorted(incident.skipped),
        "resumen": summary,
    }
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def make_llm_ask(llm: LLMClient):
    """Devuelve una función `llm_ask(spec, suggestions, incident, highlighted)`."""
    def _ask(spec, suggestions, incident, highlighted):
        return _llm_ask(spec, suggestions, incident, highlighted, llm)
    return _ask


def collect_vocabulary(cases: Iterable[dict[str, str]], specs: list[PropertySpec]) -> dict[str, set[str]]:
    """Vocabulario por propiedad a partir de los casos del CBR."""
    vocab: dict[str, set[str]] = {s.uri: set() for s in specs}
    for case in cases:
        for spec in specs:
            v = case.get(spec.uri)
            if v is not None:
                vocab[spec.uri].add(v)
    return vocab
