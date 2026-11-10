"""Vérifie les trois points non testés dans le conteneur : Wikidata, LLM, SBERT.

    python smoke_test.py            # depuis le dossier baseline, avec GROQ_API_KEY défini
"""
from __future__ import annotations

import sys
import traceback

from agent1.config import Config
from agent1.llm import LLMClient
from agent1.store import SentenceTransformerEncoder
from agent1.wikidata import WikidataClient
from agent2.executor import Executor

cfg = Config()
results: list[tuple[str, bool, str]] = []


def step(name, fn):
    try:
        results.append((name, True, str(fn())))
    except Exception as e:
        traceback.print_exc()
        results.append((name, False, f"{type(e).__name__}: {e}"))


wd = WikidataClient(cfg)


def wd_search():
    c = wd.search("Albert Einstein", "en")[0]
    assert c.id == "Q937", c
    return f"{c.id} {c.label}"


def wd_info():
    i = wd.info(["Q937", "P19", "Q999999999", "Q99999999999"])   # 3e : absent ; 4e : forme invalide
    assert i["Q937"]["exists"] and i["P19"]["exists"] and not i["Q999999999"]["exists"] and not i["Q99999999999"]["exists"], i
    return f"Q937={i['Q937']['label']}, P19={i['P19']['label']}, QID absent détecté"


def llm_call():
    out = LLMClient(cfg).chat("Réponds par un seul mot.", "Capitale de la France ?")
    assert "paris" in out.lower(), out
    return f"{cfg.llm_model} -> {out!r}"


def sbert():
    enc = SentenceTransformerEncoder(cfg.embed_model)
    m = enc.encode(["Quelle est la capitale de la France ?",
                    "What is the capital of Germany?",
                    "Combien de films a tourné DiCaprio ?"])
    s = m @ m.T
    assert s[0, 1] > s[0, 2], s
    return f"{cfg.embed_model}, dimension {m.shape[1]}, capitale~capitale {s[0,1]:.2f} > capitale~films {s[0,2]:.2f}"


step("Wikidata : wbsearchentities", wd_search)
step("Wikidata : wbgetentities (existence)", wd_info)
def wdqs_exec():
    r = Executor(cfg).run("SELECT ?u WHERE { wd:Q937 wdt:P19 ?u }")
    assert r.status == "ok" and r.answers == ["Q3012"], r
    k = Executor(cfg).run("ASK { wd:Q937 wdt:P19 wd:Q3012 }")
    assert k.status == "ok" and k.answers == ["true"], k
    bad = Executor(cfg).run("SELECT ?u WHERE { wd:Q937 wdt:P19 ?u ")
    assert bad.status == "syntaxe", bad
    return f"Agent 2 : Einstein -> {r.answers} en {r.seconds:.1f} s ; ASK vrai ; erreur de syntaxe détectée"


step("LLM : appel chat", llm_call)
step("Wikidata : exécution WDQS (Agent 2)", wdqs_exec)
step("SBERT : encodage multilingue", sbert)

print()
for name, ok, msg in results:
    print(("OK  " if ok else "ÉCHEC"), name, "-", msg)
sys.exit(0 if all(ok for _, ok, _ in results) else 1)
