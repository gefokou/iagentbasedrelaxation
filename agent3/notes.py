"""Notes d'intégration par entité, par correspondance de motifs (DeBellis et Neches, ONTOLLM 2026).

Méthode de l'article, adaptée à Wikidata :
  - chaque triplet est transformé en phrase anglaise par un motif déterministe
    (« The {label de la propriété} of {sujet} is {objet}. ») ; aucun LLM ne paraphrase ;
  - les phrases d'une même entité sont REGROUPÉES en une seule note (l'article montre que les
    notes consolidées par sujet sont retrouvées bien mieux que les phrases isolées) ;
  - les phrases sont classées par similarité avec la question (même encodeur que T3, comme
    l'article qui réutilise le modèle d'embedding du RAG) ; sans encodeur, ordre par
    nombre de valeurs (P31 d'abord) ;
  - chaque note garde l'identifiant de l'entité (rattachement au graphe).
Sont exclus : identifiants externes, images/fichiers Commons, formes géographiques, URL. Une seule requête SPARQL pour toutes les entités.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from agent1.wikidata import WikidataClient

_QID = re.compile(r"^Q[1-9]\d{0,9}$")
_PREPS = {"of", "by", "in", "at", "to", "from", "for", "on", "with", "after", "as"}


def join_values(values: list[str]) -> str:
    if len(values) <= 1:
        return "".join(values)
    return ", ".join(values[:-1]) + " and " + values[-1]


def verbalise(subject: str, pid: str, label: str, values: list[str]) -> str:
    """Motif déterministe triplet(s) -> phrase. Les motifs sont listés ici, rien d'autre."""
    vals = join_values(values)
    lab = (label or pid).strip()
    if pid == "P31":
        return f"{subject} is an instance of {vals}."
    if pid == "P279":
        return f"{subject} is a subclass of {vals}."
    if lab.split()[-1].lower() in _PREPS:          # « member of », « educated at », « named after »
        return f"{subject} is {lab} {vals}."
    verb = "are" if len(values) > 1 else "is"
    return f"The {lab} of {subject} {verb} {vals}."


@dataclass
class Note:
    tag: str                       # N1, N2...
    id: str
    label: str = ""
    description: str = ""
    sentences: list[str] = field(default_factory=list)

    def render(self) -> str:
        head = f"[{self.tag}] {self.label or self.id} ({self.id})"
        if self.description:
            head += f" — {self.description}"
        return head + "\n" + " ".join(self.sentences) if self.sentences else head


def _value(o: dict, label: str) -> str:
    if o["type"] == "uri" and "/entity/Q" in o["value"]:
        qid = o["value"].rsplit("/", 1)[-1]
        return f"{label} ({qid})" if label else qid
    return re.sub(r"T00:00:00Z$", "", o["value"])


class NoteBuilder:
    def __init__(self, wd: WikidataClient, encoder=None, max_props: int = 15,
                 max_values: int = 3):
        self.wd, self.encoder = wd, encoder
        self.max_props, self.max_values = max_props, max_values

    def build(self, ids: list[str], question: str | None = None) -> list[Note]:
        ids = [i for i in dict.fromkeys(ids) if _QID.match(i)]
        if not ids:
            return []
        info = self.wd.info(ids)
        vals = " ".join(f"wd:{i}" for i in ids)
        rows = self.wd.sparql(f"""
SELECT ?e ?p ?pl ?o ?ol WHERE {{
  VALUES ?e {{ {vals} }}
  ?e ?p ?o .
  ?prop wikibase:directClaim ?p ; wikibase:propertyType ?t .
  FILTER(?t NOT IN (wikibase:ExternalId, wikibase:CommonsMedia, wikibase:GeoShape,
                    wikibase:TabularData, wikibase:Url))
  OPTIONAL {{ ?prop rdfs:label ?pl . FILTER(LANG(?pl) = "en") }}
  OPTIONAL {{ ?o rdfs:label ?ol . FILTER(LANG(?ol) = "en") }}
}} LIMIT 4000""")
        per: dict[str, dict[str, dict]] = {i: {} for i in ids}
        for r in rows:
            eid = r["e"]["value"].rsplit("/", 1)[-1]
            pid = r["p"]["value"].rsplit("/", 1)[-1]
            d = per.setdefault(eid, {}).setdefault(pid, {"label": "", "values": []})
            d["label"] = d["label"] or r.get("pl", {}).get("value", "") or pid
            v = _value(r["o"], r.get("ol", {}).get("value", ""))
            if v not in d["values"]:
                d["values"].append(v)
        notes = []
        for n, eid in enumerate(ids, 1):
            subject = info.get(eid, {}).get("label", "") or eid
            props = per.get(eid, {})
            sents = {pid: verbalise(subject, pid, d["label"], d["values"][: self.max_values])
                     for pid, d in props.items()}
            notes.append(Note(
                tag=f"N{n}", id=eid, label=info.get(eid, {}).get("label", ""),
                description=info.get(eid, {}).get("description", ""),
                sentences=self._select(sents, props, question)))
        return notes

    def _select(self, sents: dict[str, str], props: dict, question: str | None) -> list[str]:
        if not sents:
            return []
        if self.encoder is not None and question:
            pids = list(sents)
            m = self.encoder.encode([question] + [sents[p] for p in pids])
            sims = m[1:] @ m[0]
            order = sorted(range(len(pids)), key=lambda i: (pids[i] != "P31", -float(sims[i])))
        else:   # sans encodeur : P31, puis propriétés à peu de valeurs (naissance, pays...)
            pids = sorted(sents, key=lambda k: (k != "P31", len(props[k]["values"]), k))
            order = list(range(len(pids)))
        return [sents[pids[i]] for i in order[: self.max_props]]
