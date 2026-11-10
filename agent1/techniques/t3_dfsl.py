"""T3 : DFSL (D'Abramo, Zugarini, Torroni, 2025) avec le code officiel des auteurs.

https://github.com/jacopodabramo/DFSL  (copie non modifiée dans external/dfsl)

Repris tel quel : encodage question + entités + relations (all-mpnet-base-v2), similarité
cosinus, k = 5 exemples du plus proche au plus éloigné, prompt, extraction <SPARQL>.
Remplacé : leur serveur vLLM par une API compatible OpenAI (voir dfsl_official/runner.py).

Dans la pipeline réelle, les entités et relations « gold » de l'article sont remplacées par
celles de la liaison (module EL/RL amont, comme dans l'architecture KGQA de l'article).
"""
from __future__ import annotations

import re
import time
from pathlib import Path

from ..config import Config
from ..filtre import ids_in
from ..types import Generated, Linking

NO_QUERY = "aucune requête SPARQL dans la réponse"


def _norm(q: str) -> str:
    return re.sub(r"\W+", " ", q.lower()).strip()


class T3Dfsl:
    name = "T3"
    needs_link = True

    def __init__(self, cfg: Config, llm, examples: list[dict], encode=None,
                 cache_dir: str | None = None):
        from dfsl_official.runner import (ENT_URI, REL_URI, _cached_embeddings, _dfsl,
                                          default_encoder, to_dfsl_format)
        if not examples:
            raise ValueError("stock d'exemples vide")
        self.cfg, self.llm = cfg, llm
        self.ENT, self.REL = ENT_URI, REL_URI
        (self._similarity, self._embed_text, self._prompt_for, self._extract,
         self._prompt) = _dfsl()
        self.encode = encode or default_encoder()
        self.train, _ = to_dfsl_format(examples, [], {})
        texts = [self._embed_text(d, "all") for d in self.train.values()]
        self.emb = (_cached_embeddings(self.encode, texts, Path(cache_dir)) if cache_dir
                    else self.encode(texts))

    def _labelled(self, uri: str, cand) -> list | str:
        return [uri + cand.id, cand.label] if cand.label else uri + cand.id

    def generate(self, question: str, link: Linking) -> Generated:
        t0 = time.time()
        g = Generated(technique=self.name)
        try:
            ents = [self._labelled(self.ENT, c) for c in link.top_entities()]
            rels = [self._labelled(self.REL, c) for c in link.top_properties()]
            q = {"question": question, "entities": ents, "relations": rels}
            tst = {"q": q}
            k, extra = self.cfg.k_examples, int(self.cfg.exclude_exact_question)
            sims = self._similarity(self.emb, self.encode([self._embed_text(q, "all")]),
                                    self.train, tst, k + extra)
            ids = [i for i in sims["q"] if not (extra and _norm(self.train[i]["question"]) == _norm(question))]
            sims = {"q": ids[:k]}
            text = self._prompt_for(self._prompt, q, "all", sims, self.train, tst, "q")["text"]
            g.raw = self.llm.chat("", text)        # prompt entier en un message, comme les auteurs
            found = self._extract(g.raw, self._prompt["open_tag"], self._prompt["close_tag"])
            g.query = found[0].strip() if found else None
            if g.query is None:
                g.error = NO_QUERY
            shown = {c.id for c in link.top_entities() + link.top_properties()}
            for i in sims["q"]:
                shown.update(x for part in ids_in(self.train[i]["query"]) for x in part)
            g.allowed_ids = shown
        except Exception as e:
            g.error = f"{type(e).__name__}: {e}"
        g.seconds = time.time() - t0
        return g
