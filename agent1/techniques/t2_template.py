"""T2 : Template-based SPARQL Generation (Patrons prédéfinis).

Génération d'une requête SPARQL par sélection d'un patron structurel standardisé
(1-hop direct, 1-hop inverse, count, ASK, 2-hop) et instanciation avec les entités
et relations issues du linking.
"""
from __future__ import annotations

import re
import time
from ..config import Config
from ..types import Generated, Linking

TEMPLATE_1HOP_FORWARD = "SELECT DISTINCT ?obj WHERE {{ wd:{ent} wdt:{prop} ?obj . }}"
TEMPLATE_1HOP_INVERSE = "SELECT DISTINCT ?sbj WHERE {{ ?sbj wdt:{prop} wd:{ent} . }}"
TEMPLATE_ASK = "ASK WHERE {{ wd:{ent1} wdt:{prop} wd:{ent2} . }}"
TEMPLATE_COUNT = "SELECT (COUNT(DISTINCT ?obj) AS ?count) WHERE {{ wd:{ent} wdt:{prop} ?obj . }}"
TEMPLATE_2HOP = "SELECT DISTINCT ?obj WHERE {{ wd:{ent} wdt:{prop1} ?mid . ?mid wdt:{prop2} ?obj . }}"


def _detect_template_type(question: str) -> str:
    q = question.lower().strip()
    if re.search(r"^(is|are|was|were|did|do|does)\b", q):
        return "ask"
    if re.search(r"\b(how many|count|number of)\b", q):
        return "count"
    if re.search(r"\b(who|which|what)\b.*\b(in|at|by|from|of)\s+\w+\s*\??$", q) and not re.search(r"^(what is|what was|what are)\b", q):
        return "1hop_inverse"
    return "1hop_forward"


class T2Template:
    name = "T2"
    needs_link = True

    def __init__(self, cfg: Config, llm=None):
        self.cfg = cfg
        self.llm = llm

    def generate(self, question: str, link: Linking) -> Generated:
        t0 = time.time()
        g = Generated(technique=self.name)
        try:
            ents = link.top_entities()
            props = link.top_properties()

            if not ents or not props:
                g.error = "T2: entité ou propriété manquante pour instancier un patron"
                g.seconds = time.time() - t0
                return g

            tpl_type = _detect_template_type(question)

            if tpl_type == "ask" and len(ents) >= 2:
                g.query = TEMPLATE_ASK.format(ent1=ents[0].id, prop=props[0].id, ent2=ents[1].id)
            elif tpl_type == "count":
                g.query = TEMPLATE_COUNT.format(ent=ents[0].id, prop=props[0].id)
            elif tpl_type == "1hop_inverse":
                g.query = TEMPLATE_1HOP_INVERSE.format(prop=props[0].id, ent=ents[0].id)
            elif len(props) >= 2 and len(ents) == 1 and (" of the " in question.lower() or " whose " in question.lower()):
                g.query = TEMPLATE_2HOP.format(ent=ents[0].id, prop1=props[0].id, prop2=props[1].id)
            else:
                g.query = TEMPLATE_1HOP_FORWARD.format(ent=ents[0].id, prop=props[0].id)

            g.raw = f"Template applied ({tpl_type}): {g.query}"
            g.allowed_ids = {c.id for c in ents + props}
        except Exception as e:
            g.error = f"{type(e).__name__}: {e}"
        g.seconds = time.time() - t0
        return g
