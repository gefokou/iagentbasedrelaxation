"""Normalisation des réponses d'une requête SPARQL en un ensemble de chaînes comparables.

Règles (identiques pour la référence et pour la requête générée) :
  - ASK : {"true"} ou {"false"} ;
  - toutes les colonnes sont aplaties, sauf les colonnes de libellé (?xLabel, ?xDescription,
    ?xAltLabel) quand la colonne de base ?x est aussi projetée ;
  - URI d'entité Wikidata -> identifiant (Q42, P19) ; autre URI -> telle quelle ;
  - nombre -> forme décimale canonique (5, 5.0, "+5" -> "5") ;
  - date xsd:dateTime à minuit -> AAAA-MM-JJ ;
  - autre chaîne -> minuscules, espaces réduits, langue ignorée.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

_ENTITY = re.compile(r"^https?://www\.wikidata\.org/(?:entity|prop/direct)/([QP]\d+)$")
_NUMERIC = {"decimal", "integer", "double", "float", "int", "long", "nonNegativeInteger",
            "positiveInteger", "negativeInteger", "short"}
_LABEL_SUFFIXES = ("AltLabel", "Label", "Description")


def _number(text: str) -> str | None:
    try:
        d = Decimal(text.strip().lstrip("+"))
    except InvalidOperation:
        return None
    if not d.is_finite():
        return None
    s = format(d.normalize(), "f")
    return "0" if s in ("-0", "") else s


def normalise_value(cell: dict) -> str:
    kind, value = cell.get("type"), cell.get("value", "")
    if kind == "uri":
        m = _ENTITY.match(value)
        return m.group(1) if m else value
    if kind == "bnode":
        return "_:" + value
    dt = (cell.get("datatype") or "").rsplit("#", 1)[-1]
    if dt in _NUMERIC:
        n = _number(value)
        if n is not None:
            return n
    if dt == "dateTime":
        m = re.fullmatch(r"([+-]?\d{4,})-(\d\d)-(\d\d)T00:00:00Z?", value.strip())
        if m:
            return f"{m.group(1).lstrip('+')}-{m.group(2)}-{m.group(3)}"
        return value.strip()
    return " ".join(value.casefold().split())


def _kept_vars(variables: list[str]) -> list[str]:
    keep = []
    for v in variables:
        base = next((v[: -len(suf)] for suf in _LABEL_SUFFIXES
                     if v.endswith(suf) and len(v) > len(suf)), None)
        if base is not None and base in variables:
            continue
        keep.append(v)
    return keep


def answers_from_select(variables: list[str], bindings: list[dict]) -> set[str]:
    cols = _kept_vars(variables)
    out: set[str] = set()
    for row in bindings:
        for v in cols:
            if v in row:
                out.add(normalise_value(row[v]))
    return out


def answers_from_ask(boolean: bool) -> set[str]:
    return {"true" if boolean else "false"}
