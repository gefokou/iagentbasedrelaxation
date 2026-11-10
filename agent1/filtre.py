"""Filtre : syntaxe et existence des QID/PID. Aucune exécution de la requête."""
from __future__ import annotations

import re

from rdflib.plugins.sparql.parser import parseQuery

from .normalise import canonical_form
from .types import Checked, Generated
from .wikidata import WikidataClient

STD_PREFIXES = {
    "wd": "http://www.wikidata.org/entity/",
    "wdt": "http://www.wikidata.org/prop/direct/",
    "p": "http://www.wikidata.org/prop/",
    "ps": "http://www.wikidata.org/prop/statement/",
    "pq": "http://www.wikidata.org/prop/qualifier/",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "wikibase": "http://wikiba.se/ontology#",
    "bd": "http://www.bigdata.com/rdf#",
    "schema": "http://schema.org/",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}

_ENT = re.compile(r"\bwd:(Q\d+)\b")
_PROP = re.compile(r"\b(?:wdt|p|ps|pq):(P\d+)\b")


def add_prefixes(query: str) -> str:
    """Déclare les préfixes Wikidata utilisés mais absents (comme le fait le service WDQS)."""
    declared = set(re.findall(r"(?i)\bPREFIX\s+(\w*):", query))
    used = set(re.findall(r"(?<![\w<\"'/#])(\w+):[\w(]", query)) & set(STD_PREFIXES)
    missing = [k for k in sorted(used) if k not in declared]
    head = "".join(f"PREFIX {k}: <{STD_PREFIXES[k]}>\n" for k in missing)
    return head + query


def ids_in(query: str) -> tuple[list[str], list[str]]:
    return (list(dict.fromkeys(_ENT.findall(query))),
            list(dict.fromkeys(_PROP.findall(query))))


def check(gen: Generated, wd: WikidataClient | None, check_ids: bool = True,
          strict: bool = False) -> Checked:
    res = Checked(technique=gen.technique, query=None, ok=False)
    res.warnings.extend(gen.notes)
    if not gen.query:
        res.reasons.append(gen.error or "pas de requête")
        return res
    q = add_prefixes(gen.query.strip())
    # 1. syntaxe
    try:
        parseQuery(q)
    except Exception as e:
        res.reasons.append(f"syntaxe : {str(e).splitlines()[0][:160]}")
        return res
    if not re.search(r"(?i)\b(SELECT|ASK)\b", q):
        res.reasons.append("la requête n'est ni SELECT ni ASK")
        return res
    # 2. forme normalisée (doit aussi pouvoir se construire)
    try:
        res.form = canonical_form(q)
    except Exception as e:
        res.reasons.append(f"normalisation : {type(e).__name__}: {str(e)[:120]}")
        return res
    qids, pids = ids_in(q)
    # 3. existence des identifiants
    if check_ids and wd is not None and (qids or pids):
        try:
            info = wd.info(qids + pids)
            missing = [i for i in qids + pids if i in info and not info[i]["exists"]]
            if missing:
                res.reasons.append("identifiants inexistants : " + ", ".join(missing))
                res.form = None
                return res
        except Exception as e:
            if strict:   # en mode strict, on ne laisse pas passer une requête non vérifiée
                res.reasons.append(f"existence non vérifiable ({type(e).__name__}: {e})")
                res.form = None
                return res
            res.warnings.append(f"existence non vérifiée ({type(e).__name__})")
    # 4. provenance : identifiants absents du prompt (risque d'hallucination), non bloquant
    if gen.allowed_ids:
        foreign = [i for i in qids + pids if i not in gen.allowed_ids]
        if foreign:
            res.warnings.append("identifiants hors prompt : " + ", ".join(foreign))
    res.query, res.ok = q, True
    return res
