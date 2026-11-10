"""Tests de l'Agent 1 avec de faux clients (aucun réseau)."""
import json
import zlib
import re
from pathlib import Path

import numpy as np
import pytest

from agent1 import Agent1, Config
from agent1.filtre import add_prefixes, check
from agent1.normalise import canonical_form
import torch
from agent1.types import Candidate, Checked, Generated
from agent1.vote import vote

SEED = Path(__file__).parent.parent / "data" / "seed_examples.jsonl"
SEED_EXAMPLES = [json.loads(l) for l in SEED.read_text(encoding="utf-8").splitlines() if l.strip()]
P = ("PREFIX wd: <http://www.wikidata.org/entity/> "
     "PREFIX wdt: <http://www.wikidata.org/prop/direct/> ")


# --------------------------------------------------------------- faux clients
def fake_encode(texts):
    """Sac de mots haché (tenseurs, comme l'encodeur des auteurs de DFSL)."""
    m = torch.zeros(len(texts), 1024)
    for i, t in enumerate(texts):
        for w in re.findall(r"\w+", t.lower()):
            m[i, zlib.crc32(w.encode()) % 1024] += 1
    return m


KNOWN = {"Q937": "Albert Einstein", "Q3012": "Ulm", "Q183": "Germany", "P19": "place of birth",
         "P36": "capital", "P166": "award received", "Q38104": "Nobel Prize in Physics",
         "Q6256": "country", "P31": "instance of", "Q1726": "Munich", "P159": "x",
         "Q4830453": "business", "P279": "subclass of"}


class FakeWD:
    def __init__(self):
        self.calls = []

    def search(self, text, lang="en", kind="item", limit=5):
        self.calls.append(("search", text, kind))
        t = text.lower()
        if "einstein" in t:
            return [Candidate("Q937", "Albert Einstein", "physicist")]
        if "birth" in t or "born" in t or "naissance" in t:
            return [Candidate("P19", "place of birth", "")]
        return []

    def info(self, ids, langs=()):
        self.calls.append(("info", tuple(ids)))
        return {i: {"exists": i in KNOWN, "label": KNOWN.get(i, ""), "description": ""}
                for i in ids}

    def label(self, i):
        return KNOWN.get(i, "")



class FakeLLM:
    """Répond selon le contenu du prompt ; garde un journal des prompts reçus."""

    def __init__(self, t3=None):
        self.t3, self.prompts = t3, []
        self.cfg = Config()

    def chat(self, system, user, temperature=None, max_tokens=None):
        self.prompts.append((system, user))
        if "You analyse questions" in system:
            return json.dumps({"language": "en",
                               "entities": [{"text": "Albert Einstein", "en": "Albert Einstein"}],
                               "properties": [{"text": "born", "en": "place of birth"}]})
        return self.t3


GOOD = P + "SELECT ?uri WHERE { wd:Q937 wdt:P19 ?uri . }"
GOOD_B = P + "SELECT DISTINCT ?answer WHERE { wd:Q937 wdt:P19 ?answer }"   # même forme
FLIPPED = P + "SELECT ?uri WHERE { ?uri wdt:P19 wd:Q937 . }"
FAKE_ID = P + "SELECT ?uri WHERE { wd:Q99999999 wdt:P19 ?uri . }"


def make_agent(t3, t4, **cfg):
    """T3 = DFSL officiel (faux modèle, faux encodeur) ; T4 = GRASP (faux exécutable)."""
    llm = FakeLLM(f"<SPARQL>{t3}</SPARQL>")
    out = {"output": {"type": "answer", "sparql": t4}}
    base_cfg = dict(techniques=("T3", "T4"), priority=("T3", "T4"))
    base_cfg.update(cfg)
    agent = Agent1(Config(**base_cfg), llm=llm, wd=FakeWD(), store=SEED_EXAMPLES, encode=fake_encode,
                   grasp_runner=lambda q: out)
    return agent, llm


# ---------------------------------------------------------------- normalisation
def test_same_form_despite_names_and_distinct():
    assert canonical_form(GOOD) == canonical_form(GOOD_B)


def test_triple_flip_changes_form():
    assert canonical_form(GOOD) != canonical_form(FLIPPED)


def test_ask_differs_from_select_and_paths_supported():
    ask = P + "ASK { wd:Q937 wdt:P19 wd:Q3012 }"
    assert canonical_form(ask)[0] == "AskQuery"
    path = P + "SELECT ?u WHERE { ?t wdt:P279* wd:Q4830453 . ?u wdt:P31 ?t }"
    assert any(t[1].startswith("PATH:P279") for t in canonical_form(path)[2])
    count = P + "SELECT (COUNT(?u) AS ?c) WHERE { ?u wdt:P31 wd:Q6256 }"
    assert canonical_form(count)[1] is True


# ------------------------------------------------------------------------ filtre
def gen(q, tech="T3", allowed=()):
    return Generated(technique=tech, query=q, allowed_ids=set(allowed))


def test_filter_accepts_valid_and_injects_prefixes():
    r = check(gen("SELECT ?u WHERE { wd:Q937 wdt:P19 ?u }"), FakeWD())
    assert r.ok and r.query.startswith("PREFIX wd:")


def test_filter_rejects_syntax_and_missing_ids_and_empty():
    wd = FakeWD()
    assert "syntaxe" in check(gen(P + "SELECT ?u WHERE { wd:Q937 wdt:P19 "), wd).reasons[0]
    assert "inexistants" in check(gen(FAKE_ID), wd).reasons[0]
    assert not check(Generated(technique="T3", error="boom"), wd).ok


def test_filter_warns_on_ids_outside_prompt_but_keeps():
    r = check(gen(GOOD, allowed={"Q937"}), FakeWD())
    assert r.ok and any("hors prompt" in w and "P19" in w for w in r.warnings)


def test_filter_network_failure_does_not_reject():
    class Down(FakeWD):
        def info(self, *a, **k):
            raise RuntimeError("réseau")
    r = check(gen(GOOD), Down())
    assert r.ok and any("non vérifiée" in w for w in r.warnings)


def test_add_prefixes_leaves_declared_ones():
    q = "PREFIX wd: <http://www.wikidata.org/entity/>\nSELECT ?u WHERE { wd:Q1 wdt:P2 ?u }"
    out = add_prefixes(q)
    assert out.count("PREFIX wd:") == 1 and "PREFIX wdt:" in out


# -------------------------------------------------------------------------- vote
def mk(tech, q):
    c = check(gen(q, tech), FakeWD())
    assert c.ok, c.reasons
    return c


def test_vote_agreement_keeps_priority_technique():
    d = vote([mk("T4", GOOD_B), mk("T3", GOOD)], ("T3", "T4"))
    assert d.rule == "accord" and d.technique == "T3" and d.query.endswith("?uri . }")


def test_vote_disagreement_uses_priority_order():
    d = vote([mk("T4", FLIPPED), mk("T3", GOOD)], ("T3", "T4"))
    assert d.rule == "priorite" and d.technique == "T3"
    d = vote([mk("T4", FLIPPED), mk("T3", GOOD)], ("T4", "T3"))
    assert d.technique == "T4"


def test_vote_single_survivor_and_none():
    bad = Checked("T3", None, False, ["syntaxe"])
    d = vote([bad, mk("T4", GOOD)], ("T3", "T4"))
    assert d.rule == "unique" and d.technique == "T4"
    d = vote([bad, Checked("T4", None, False, ["pas de requête"])], ("T3", "T4"))
    assert d.rule == "aucun" and d.query is None


def test_vote_three_techniques_majority_beats_priority():
    c = [mk("T3", FLIPPED), mk("T4", GOOD), mk("T2", GOOD_B)]
    d = vote(c, ("T3", "T4", "T2"))
    assert d.rule == "accord" and d.technique == "T4"


# ------------------------------------------------------------------ bout en bout
def test_end_to_end_agreement():
    agent, llm = make_agent(GOOD, GOOD_B)
    r = agent.run("Where was Albert Einstein born?")
    assert r.rule == "accord" and r.technique == "T3"
    assert "Q937" in r.query and [g.technique for g in r.generated] == ["T3", "T4"]
    t3p = next(u for s, u in llm.prompts if "Utilize all provided golden entities" in u)
    # prompt officiel de DFSL : instruction, exemples séparés par ###, entités et relations de la liaison
    assert t3p.count("###") >= 5 and "Gold Entities:" in t3p
    assert "http://www.wikidata.org/entity/Q937 (Albert Einstein)" in t3p
    assert "http://www.wikidata.org/prop/direct/P19 (place of birth)" in t3p
    assert t3p.rstrip().endswith("Query:")


def test_t3_examples_nearest_first_and_exact_question_excluded():
    agent, llm = make_agent(GOOD, GOOD)
    agent.run("What is the capital of Germany?")          # présente dans le stock : exclue
    t3p = next(u for s, u in llm.prompts if "Utilize all provided golden entities" in u)
    assert t3p.split("Examples:")[1].count("Question: What is the capital of Germany?") == 1  # seulement la question posée


def test_grasp_alone_does_not_need_the_linker_or_the_llm():
    agent, llm = make_agent(GOOD, GOOD, techniques=("T4",))
    r = agent.run("Where was Albert Einstein born?")
    assert r.technique == "T4" and r.rule == "unique" and llm.prompts == []


def test_end_to_end_t3_hallucination_falls_back_to_t4():
    agent, _ = make_agent(FAKE_ID, GOOD)
    r = agent.run("Where was Albert Einstein born?")
    assert r.rule == "unique" and r.technique == "T4"
    assert any("inexistants" in x for c in r.checked for x in c.reasons)


def test_end_to_end_both_fail_and_one_technique_crash_is_isolated():
    agent, _ = make_agent("SELECT broken {{", "pas de sparql ici")
    r = agent.run("Where was Albert Einstein born?")
    assert r.query is None and r.rule == "aucun"
    json.dumps(r.to_dict())  # sérialisable


def test_no_execution_of_generated_query():
    agent, _ = make_agent(GOOD, FLIPPED)
    agent.run("Where was Albert Einstein born?")
    assert not hasattr(agent.wd, "executed")           # le faux client n'a pas de SPARQL
    assert all(c[0] in {"search", "info"} for c in agent.wd.calls)


def test_single_technique_configurations():
    agent, _ = make_agent(GOOD, FLIPPED, techniques=("T3",))
    r = agent.run("Where was Albert Einstein born?")
    assert r.technique == "T3" and r.rule == "unique"
    agent, _ = make_agent(GOOD, FLIPPED, techniques=("T4",))
    assert agent.run("Where was Albert Einstein born?").technique == "T4"


# ------------------------------------------------------------------ mode strict
def test_strict_rejects_unverifiable_ids_and_lenient_keeps_them():
    class Down(FakeWD):
        def info(self, *a, **k):
            raise RuntimeError("réseau")
    assert not check(gen(GOOD), Down(), strict=True).ok
    assert "non vérifiable" in check(gen(GOOD), Down(), strict=True).reasons[0]
    assert check(gen(GOOD), Down(), strict=False).ok


def test_strict_linker_fails_loudly_instead_of_heuristic_fallback():
    from agent1.linking import Linker

    class Boom:
        def chat(self, *a, **k):
            raise RuntimeError("LLM hors service")
    with pytest.raises(RuntimeError):
        Linker(Config(strict=True), Boom(), FakeWD()).link("Où est né Albert Einstein ?")
    # hors mode strict : repli heuristique (ici aucune mention capitalisée en dehors du 1er mot)
    assert Linker(Config(strict=False), Boom(), FakeWD()).link("Où est né Albert Einstein ?")


def test_add_prefixes_does_not_redeclare_inline_prefixes():
    q = add_prefixes(P + "SELECT ?u WHERE { wd:Q1 wdt:P2 ?u }")
    assert q.count("PREFIX wd:") == 1 and q.count("PREFIX wdt:") == 1
