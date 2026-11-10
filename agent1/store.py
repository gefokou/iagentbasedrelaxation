"""Encodeur SBERT (classement des notes de l'Agent 3) et lecture de LC-QuAD 2.0.

La recherche d'exemples de T3 est faite par le code officiel de DFSL (external/dfsl).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

_ENT = re.compile(r"wd:(Q\d+)")
_REL = re.compile(r"(?:wdt|p|ps|pq):(P\d+)")


class SentenceTransformerEncoder:
    """Encodeur SBERT ; import paresseux pour ne pas exiger le paquet aux tests."""

    def __init__(self, name: str):
        from sentence_transformers import SentenceTransformer
        self.name = name
        self.model = SentenceTransformer(name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True,
                                            show_progress_bar=False), dtype=np.float32)


# ------------------------------------------------------------- construction
def ids_from_sparql(sparql: str) -> tuple[list[str], list[str]]:
    return (list(dict.fromkeys(_ENT.findall(sparql))),
            list(dict.fromkeys(_REL.findall(sparql))))


def load_lcquad2(path: str) -> list[dict]:
    """Lit train.json de LC-QuAD 2.0 -> [{'question','sparql','ids_e','ids_r'}] (sans libellés)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for it in data:
        q = it.get("question")
        if not q or q.strip().lower() in {"n/a", "[]"} or len(q) < 5:
            nn = it.get("NNQT_question") or ""
            q = re.sub(r"[{}]", "", nn).strip()
        sp = it.get("sparql_wikidata")
        if not q or not sp:
            continue
        e, r = ids_from_sparql(sp)
        out.append({"question": q.strip(), "sparql": sp.strip(), "ids_e": e, "ids_r": r})
    return out
