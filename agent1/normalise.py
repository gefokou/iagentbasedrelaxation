"""Normalisation d'une requête en forme canonique : triplets (QID, PID, sens).

Deux requêtes écrites différemment (noms de variables, ordre des triplets, DISTINCT,
espaces) donnent la même forme ; une inversion sujet/objet (triple-flip) donne une forme
différente.
"""
from __future__ import annotations

import re

from rdflib import Variable
from rdflib.plugins.sparql.algebra import CompValue, translateQuery, traverse
from rdflib.plugins.sparql.parser import parseQuery

_PROP_URI = re.compile(r"https?://www\.wikidata\.org/prop/(direct|statement|qualifier)?/?([PQ]\d+)")
_NS = {"direct": "", "statement": "ps:", "qualifier": "pq:", None: "p:"}


def _prop_uri(m: re.Match) -> str:
    return _NS[m.group(1)] + m.group(2)


def _term(t) -> str:
    if isinstance(t, Variable):
        return "?"
    s = str(t)
    if s.startswith("Path("):  # chemin de propriétés : wdt:P279*, p:P166/ps:P166...
        return "PATH:" + _PROP_URI.sub(_prop_uri, s[5:-1])
    m = _PROP_URI.fullmatch(s)
    if m:
        return _prop_uri(m)
    m = re.search(r"/entity/([QP]\d+)$", s)
    return m.group(1) if m else f"LIT:{s}"


def canonical_form(query: str) -> tuple:
    """Forme canonique d'une requête valide. Lève une exception si elle ne s'analyse pas."""
    alg = translateQuery(parseQuery(query))
    kind = alg.algebra.name
    triples: list[tuple] = []
    state = {"agg": False, "pv": []}

    def visit(node):
        if isinstance(node, CompValue):
            if node.name == "BGP":
                triples.extend(node.triples)
            elif node.name.startswith("Aggregate_"):
                state["agg"] = True
            elif node.name == "Project":
                state["pv"] = list(node.PV)   # post-ordre : la dernière est la plus externe
        return None

    traverse(alg.algebra, visitPost=visit)

    # classement indépendant du nom des variables, puis renommage par ordre d'apparition
    def sig(t):
        return tuple(_term(x) for x in t)

    ordered = sorted(triples, key=lambda t: (sig(t), tuple(str(x) for x in t)))
    names: dict[str, str] = {}

    def vname(v: Variable) -> str:
        return names.setdefault(str(v), f"?v{len(names)}")

    out = []
    for s, p, o in ordered:
        out.append(tuple(vname(x) if isinstance(x, Variable) else _term(x) for x in (s, p, o)))
    proj = tuple(sorted(vname(v) for v in state["pv"] if str(v) in names))
    return (kind, state["agg"], tuple(sorted(out)), proj)


def triples_of(form: tuple) -> set[tuple]:
    return set(form[2])


def jaccard(a: tuple, b: tuple) -> float:
    ta, tb = triples_of(a), triples_of(b)
    return len(ta & tb) / len(ta | tb) if (ta or tb) else 1.0
