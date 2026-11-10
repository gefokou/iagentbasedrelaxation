"""Liaison d'entités et de propriétés (commune à T3 et T4).

Le LLM repère les mentions dans la question ; Wikidata (wbsearchentities) fournit les
candidats, comme dans mKGQAgent. Le choix final parmi les candidats est laissé au
prompt de génération (ex. « Paris » ville ou personne).
"""
from __future__ import annotations

import json
import re

from .config import Config
from .llm import LLMClient
from .types import Candidate, Linking
from .wikidata import WikidataClient, WikidataError

_SYSTEM = (
    "You analyse questions that will be answered with Wikidata. "
    "Given a question, list (1) the named entities or classes it mentions and (2) the "
    "relations (Wikidata properties) it asks about. For each, give the text as written in "
    "the question and its English form. Also give the question language as an ISO code. "
    'Answer with JSON only: {"language":"fr","entities":[{"text":"...","en":"..."}],'
    '"properties":[{"text":"...","en":"..."}]}'
)

_FR_WORDS = {"le", "la", "les", "un", "une", "des", "du", "de", "où", "qui", "quel", "quelle",
             "quels", "quelles", "est", "sont", "été", "né", "née", "combien", "quand",
             "pays", "capitale", "et", "dans", "par", "pour", "avec", "a", "ont", "fait"}


def guess_language(question: str) -> str:
    words = re.findall(r"\w+", question.lower())
    hits = sum(w in _FR_WORDS for w in words)
    return "fr" if hits >= 2 or re.search(r"[éèêàùçôî]", question.lower()) else "en"


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("pas de JSON")
    return json.loads(m.group(0))


def _fallback_mentions(question: str) -> dict:
    """Si le LLM échoue : suites de mots capitalisés (hors premier mot)."""
    toks = question.rstrip("?! .").split()
    ents, cur = [], []
    for i, t in enumerate(toks):
        w = t.strip(",;:()\"'")
        if w[:1].isupper() and i > 0:
            cur.append(w)
        elif cur:
            ents.append(" ".join(cur))
            cur = []
    if cur:
        ents.append(" ".join(cur))
    return {"language": guess_language(question),
            "entities": [{"text": e, "en": e} for e in ents], "properties": []}


class Linker:
    def __init__(self, cfg: Config, llm: LLMClient, wd: WikidataClient):
        self.cfg, self.llm, self.wd = cfg, llm, wd

    def _search(self, forms: list[str], langs: list[str], kind: str, n: int) -> list[Candidate]:
        seen: dict[str, Candidate] = {}
        for form in forms:
            if not form:
                continue
            for lg in langs:
                try:
                    hits = self.wd.search(form, lg, kind, limit=n)
                except WikidataError:
                    if self.cfg.strict:
                        raise
                    continue
                for h in hits:
                    seen.setdefault(h.id, h)
            if len(seen) >= n:
                break
        return list(seen.values())[:n]

    def link(self, question: str) -> Linking:
        try:
            mentions = _parse_json(self.llm.chat(_SYSTEM, question))
        except Exception:
            if self.cfg.strict:
                raise
            mentions = _fallback_mentions(question)
        lang = str(mentions.get("language") or guess_language(question))[:2].lower()
        langs = ["en"] if lang == "en" else ["en", lang]
        out = Linking(language=lang)
        for kind, key, n in (("item", "entities", self.cfg.entity_candidates),
                             ("property", "properties", self.cfg.property_candidates)):
            for m in mentions.get(key, []):
                text = (m.get("text") or "").strip()
                en = (m.get("en") or "").strip()
                if not text and not en:
                    continue
                cands = self._search([en, text], langs, kind, n)
                if cands:
                    target = out.entities if kind == "item" else out.properties
                    target[text or en] = cands
        return out
