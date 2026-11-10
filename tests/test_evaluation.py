import json

import pytest

from agent1.evaluation import (check_store_size, classify, remove_leaks, run_evaluation,
                               summarize)
from agent1.types import Checked, Generated

P = ("PREFIX wd: <http://www.wikidata.org/entity/> "
     "PREFIX wdt: <http://www.wikidata.org/prop/direct/> ")


def ex(q, s="SELECT ?u WHERE { wd:Q1 wdt:P2 ?u }"):
    return {"question": q, "sparql": P + s, "entities": [], "relations": []}


def test_small_store_is_refused():
    with pytest.raises(ValueError, match="trop petit"):
        check_store_size([ex("a b c")] * 7, 1000)
    check_store_size([ex("a b c")] * 1000, 1000)


def test_leaks_exact_neighbour_and_gold_query_are_removed():
    test = [{"id": "q1", "question": "Where was Albert Einstein born?",
             "gold_sparql": P + "SELECT ?u WHERE { wd:Q937 wdt:P19 ?u }"}]
    store = [
        ex("where was albert einstein BORN"),                       # identique (casse, ponctuation)
        ex("Where was Albert Einstein born in Germany?"),           # voisine (Jaccard 6/8 = 0,75)
        ex("Who is the place of birth of Einstein Albert?"),        # lointaine mais même requête
        ex("Capital of Germany?", "SELECT ?u WHERE { wd:Q183 wdt:P36 ?u }"),   # à garder
    ]
    store[2]["sparql"] = P + "SELECT DISTINCT ?x WHERE { wd:Q937 wdt:P19 ?x }"
    kept, removed = remove_leaks(store, test, near=0.7)
    assert [e["question"] for e in kept] == ["Capital of Germany?"]
    assert {r["reason"].split(" (")[0] for r in removed} == {
        "question identique", "question voisine", "même requête que la référence du test"}


def gen(tech, query=None, error=None):
    return Generated(technique=tech, query=query, error=error)


class R:
    def __init__(self, generated, checked, query=None):
        self.generated, self.checked, self.query = generated, checked, query
        self.technique, self.rule, self.detail = "T3", "unique", ""


def test_infra_error_is_not_counted_even_if_other_technique_answered():
    r = R([gen("T3", error="LLMError: HTTP 429"), gen("T4", "q")],
          [Checked("T3", None, False, ["pas de requête"]), Checked("T4", "q", True)], query="q")
    assert classify(r)[0] == "infra"


def test_model_failure_is_counted_as_failure_not_infra():
    r = R([gen("T3", error="aucune requête SPARQL dans la réponse"), gen("T4")],
          [Checked("T3", None, False, ["x"]), Checked("T4", None, False, ["syntaxe"])])
    assert classify(r)[0] == "echec_modele"


def test_unverifiable_ids_is_infra():
    r = R([gen("T3", "q")], [Checked("T3", None, False, ["existence non vérifiable (X)"])])
    assert classify(r)[0] == "infra"


class FakeAgent:
    """Échoue (panne) la première fois sur q2, puis réussit."""
    def __init__(self):
        self.calls = []

    def run(self, q):
        self.calls.append(q)
        if q == "q2" and self.calls.count("q2") == 1:
            raise RuntimeError("Groq indisponible")
        res = R([gen("T3", "SELECT ?u WHERE { wd:Q937 wdt:P19 ?u }")],
                [Checked("T3", "SELECT ?u WHERE { wd:Q937 wdt:P19 ?u }", True)],
                query=P + "SELECT ?u WHERE { wd:Q937 wdt:P19 ?u }")
        return res


def test_run_resume_and_retry_only_infra(tmp_path):
    test = [{"id": "a", "question": "q1", "gold_sparql": P + "SELECT ?x WHERE { wd:Q937 wdt:P19 ?x }"},
            {"id": "b", "question": "q2"}]
    out = tmp_path / "results.jsonl"
    agent = FakeAgent()
    s1 = run_evaluation(agent, test, out)
    assert s1["comptees"] == 1 and s1["exclues_infra_a_rejouer"] == ["b"]
    s2 = run_evaluation(agent, test, out)                      # reprise : ne rejoue que « b »
    assert agent.calls == ["q1", "q2", "q2"]
    assert s2["comptees"] == 2 and s2["exclues_infra_a_rejouer"] == []
    assert s2["forme_identique_a_la_reference"] == {"n": 1, "taux": 1.0}
    assert len(out.read_text().splitlines()) == 2
    assert summarize([json.loads(l) for l in out.read_text().splitlines()])["questions"] == 2


def test_leak_detection_works_on_lcquad_style_queries_without_prefixes():
    """Format réel de LC-QuAD 2.0 : espace en tête, aucun PREFIX."""
    gold = " select distinct ?obj where { wd:Q188920 wdt:P2813 ?obj . ?obj wdt:P31 wd:Q1002697 } "
    test = [{"id": "t", "question": "Which magazine is the mouthpiece of Delta?", "gold_sparql": gold}]
    store = [{"question": "Totally different wording here", "entities": [], "relations": [],
              "sparql": "select distinct ?x where { wd:Q188920 wdt:P2813 ?x . ?x wdt:P31 wd:Q1002697 }"},
             {"question": "Capital of Germany", "entities": [], "relations": [],
              "sparql": "select ?u where { wd:Q183 wdt:P36 ?u }"}]
    kept, removed = remove_leaks(store, test)
    assert len(kept) == 1 and removed[0]["reason"].startswith("même requête")
