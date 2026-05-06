"""Bucle interactivo y parsing del input del usuario.

Inputs aceptados:
  - "1".."5"       → selecciona ese candidato del top-K
  - "si|s|yes|y"   → acepta el destacado actual (top-1 por defecto)
  - "no|n"         → descarta el destacado y muestra el siguiente
  - "skip"         → deja la propiedad como None
  - texto libre    → se valida contra el vocabulario del grafo
"""
from __future__ import annotations

from dataclasses import dataclass

from .incident import Incident, PropertySpec, active_specs
from .inference import InferenceResources, recommend_property
from .traceability import USER, Suggestion


YES = {"si", "sí", "s", "yes", "y", "ok"}
NO = {"no", "n"}
SKIP = {"skip", "saltar"}


@dataclass
class Decision:
    kind: str  # "pick" | "next" | "skip" | "custom" | "invalid"
    index: int | None = None
    value: str | None = None
    source: str | None = None


def parse_user_input(
    raw: str,
    top_k: list[Suggestion],
    highlighted: int,
    vocab: set[str],
) -> Decision:
    if raw is None:
        return Decision("invalid")
    text = raw.strip()
    if not text:
        return Decision("invalid")
    low = text.lower()

    if low in SKIP:
        return Decision("skip")
    if low in YES:
        if highlighted >= len(top_k):
            return Decision("invalid")
        s = top_k[highlighted]
        return Decision("pick", index=highlighted, value=s.value, source=s.source)
    if low in NO:
        return Decision("next")
    if low.isdigit():
        idx = int(low) - 1
        if 0 <= idx < len(top_k):
            s = top_k[idx]
            return Decision("pick", index=idx, value=s.value, source=s.source)
        return Decision("invalid")

    if text in vocab:
        return Decision("custom", value=text, source=USER)
    return Decision("invalid")


def render_menu(prop_label: str, top_k: list[Suggestion], highlighted: int) -> str:
    lines = [f"Para «{prop_label}» propongo:"]
    for i, s in enumerate(top_k):
        marker = "▶" if i == highlighted else " "
        kge = f"{s.kge_score:.2f}" if s.kge_score is not None else "—"
        cbr = f"{s.cbr_score:.2f}" if s.cbr_score is not None else "—"
        lines.append(
            f" {marker} {i + 1}) {s.value:50s} ({s.source:8s} kge={kge} cbr={cbr})"
        )
    lines.append("\nResponde con: 1-5 / si / no / skip / texto libre")
    return "\n".join(lines)


def run_interaction_loop(
    incident: Incident,
    res: InferenceResources,
    specs: list[PropertySpec],
    vocab_by_prop: dict[str, set[str]],
    llm_ask=None,
    input_fn=input,
    print_fn=print,
) -> Incident:
    """Recorre las propiedades activas en orden y llena el incidente."""
    for spec in active_specs(specs):
        if incident.props.get(spec.uri) is not None or spec.uri in incident.skipped:
            continue

        suggestions = recommend_property(incident.known(), spec.uri, res)
        if not suggestions:
            print_fn(f"\n[{spec.label}] sin sugerencias del sistema.")
            raw = input_fn("Introduce un valor o «skip»: ")
            decision = parse_user_input(raw, [], 0, vocab_by_prop.get(spec.uri, set()))
            if decision.kind == "skip":
                incident.skip(spec.uri)
            elif decision.kind == "custom":
                incident.update(spec.uri, decision.value, decision.source)
            continue

        highlighted = 0
        while True:
            if llm_ask is not None:
                question = llm_ask(spec, suggestions, incident, highlighted)
            else:
                question = render_menu(spec.label, suggestions, highlighted)
            print_fn("\n" + question)
            raw = input_fn("> ")

            decision = parse_user_input(
                raw, suggestions, highlighted, vocab_by_prop.get(spec.uri, set())
            )
            if decision.kind == "pick":
                incident.update(spec.uri, decision.value, decision.source)
                break
            if decision.kind == "custom":
                incident.update(spec.uri, decision.value, decision.source)
                break
            if decision.kind == "skip":
                incident.skip(spec.uri)
                break
            if decision.kind == "next":
                if highlighted + 1 < len(suggestions):
                    highlighted += 1
                else:
                    print_fn("(no hay más candidatos; introduce un número, texto libre o skip)")
                continue
            print_fn("Entrada no reconocida.")

    return incident
