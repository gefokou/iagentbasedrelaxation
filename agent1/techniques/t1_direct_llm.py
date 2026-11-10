"""T1 : Direct LLM SPARQL Generation (LangChain-based).

Génération directe d'une requête SPARQL par prompt guidé au modèle de langage,
en lui fournissant la question en langage naturel et les entités (QIDs) et
propriétés (PIDs) identifiées par Falcon 2.0 ou le linker amont.
"""
from __future__ import annotations

import re
import time
from ..config import Config
from ..types import Generated, Linking

PROMPT_T1 = """You are a SPARQL expert for Wikidata.
Generate a valid Wikidata SPARQL query to answer the question.
Use only standard Wikidata prefixes (wd:, wdt:, p:, ps:, pq:).

Candidate Entities:
{entities}

Candidate Properties:
{properties}

Question: {question}

Return ONLY the SPARQL query wrapped in <SPARQL> and </SPARQL> tags.
"""


def _extract_sparql(text: str) -> str | None:
    m = re.search(r"<SPARQL>(.*?)</SPARQL>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"```sparql\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m2:
        return m2.group(1).strip()
    m3 = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if m3:
        content = m3.group(1).strip()
        if "SELECT" in content.upper() or "ASK" in content.upper():
            return content
    if "SELECT" in text.upper() or "ASK" in text.upper():
        lines = [l for l in text.splitlines() if not l.strip().startswith("#")]
        query = "\n".join(lines).strip()
        if "SELECT" in query.upper() or "ASK" in query.upper():
            return query
    return None


class T1DirectLLM:
    name = "T1"
    needs_link = True

    def __init__(self, cfg: Config, llm):
        self.cfg = cfg
        self.llm = llm

    def generate(self, question: str, link: Linking) -> Generated:
        t0 = time.time()
        g = Generated(technique=self.name)
        try:
            top_ents = link.top_entities()
            top_props = link.top_properties()
            ents_desc = "\n".join(f"- {c.id}: {c.label or 'unknown'}" for c in top_ents) or "- None"
            props_desc = "\n".join(f"- {c.id}: {c.label or 'unknown'}" for c in top_props) or "- None"

            prompt = PROMPT_T1.format(
                entities=ents_desc,
                properties=props_desc,
                question=question
            )

            raw = self.llm.chat("You are a SPARQL generator for Wikidata.", prompt)
            g.raw = raw
            extracted = _extract_sparql(raw)
            g.query = extracted
            if not g.query:
                g.error = "aucune requête SPARQL dans la réponse T1"

            allowed = {c.id for c in top_ents + top_props}
            g.allowed_ids = allowed
        except Exception as e:
            g.error = f"{type(e).__name__}: {e}"
        g.seconds = time.time() - t0
        return g
