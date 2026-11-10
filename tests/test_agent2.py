"""Tests de l'Agent 2 avec de fausses sessions (formats réels des réponses WDQS)."""
import json

import pytest
import requests

from agent1.config import Config
from agent2.answers import answers_from_select, normalise_value
from agent2.executor import Executor
from agent2.metrics import bootstrap_ci, compute_metrics, f1, paired_diff
from agent2.run import run_execution

E = "http://www.wikidata.org/entity/"


def uri(x):
    return {"type": "uri", "value": E + x}


def lit(v, dt=None, lang=None):
    d = {"type": "literal", "value": v}
    if dt:
        d["datatype"] = "http://www.w3.org/2001/XMLSchema#" + dt
    if lang:
        d["xml:lang"] = lang
    return d


class Resp:
    def __init__(self, code=200, data=None, text="", headers=None):
        self.status_code, self._d, self.text, self.headers = code, data, text, headers or {}

    def json(self):
        return self._d


class Sess:
    def __init__(self, responses):
        self.responses, self.calls, self.headers = list(responses), [], {}

    def post(self, url, **kw):
        self.calls.append((url, kw))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def select(vars_, rows):
    return Resp(200, {"head": {"vars": vars_}, "results": {"bindings": rows}})


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)


# ---------------------------------------------------------------- normalisation
def test_normalise_entities_numbers_dates_strings():
    assert normalise_value(uri("Q3012")) == "Q3012"
    assert normalise_value({"type": "uri", "value": "http://www.wikidata.org/prop/direct/P19"}) == "P19"
    assert normalise_value(lit("+5", "decimal")) == normalise_value(lit("5.0", "integer")) == "5"
    assert normalise_value(lit("1879-03-14T00:00:00Z", "dateTime")) == "1879-03-14"
    assert normalise_value(lit("  Ulm  ", lang="de")) == normalise_value(lit("ULM", lang="en")) == "ulm"


def test_label_columns_dropped_only_when_base_column_present():
    rows = [{"x": uri("Q1"), "xLabel": lit("One")}]
    assert answers_from_select(["x", "xLabel"], rows) == {"Q1"}
    assert answers_from_select(["xLabel"], rows) == {"one"}


# --------------------------------------------------------------------- exécuteur
def test_select_ok_empty_and_ask():
    s = Sess([select(["u"], [{"u": uri("Q3012")}]), select(["u"], []),
              Resp(200, {"head": {}, "boolean": True})])
    ex = Executor(Config(), s)
    a = ex.run("SELECT ?u WHERE { wd:Q937 wdt:P19 ?u }")
    assert a.status == "ok" and a.answers == ["Q3012"] and a.executable
    assert ex.run("SELECT ?u WHERE { wd:Q1 wdt:P1 ?u }").status == "vide"
    k = ex.run("ASK { wd:Q937 wdt:P19 wd:Q3012 }")
    assert k.status == "ok" and k.ask and k.answers == ["true"]
    assert s.calls[0][1]["data"]["query"].startswith("SELECT")


def test_syntax_error_timeout_and_service_statuses():
    s = Sess([Resp(400, text="MalformedQueryException: Encountered ..."),
              Resp(500, text="java.util.concurrent.TimeoutException"),
              requests.Timeout("lent"),
              Resp(503), Resp(503), Resp(503), Resp(503), Resp(503)])
    ex = Executor(Config(exec_retries=5), s)
    r1 = ex.run("SELECT broken")
    assert r1.status == "syntaxe" and "Malformed" in r1.error
    assert ex.run("SELECT ?a WHERE {?a ?b ?c}").status == "timeout"
    assert ex.run("SELECT ?x WHERE {?x ?y ?z}").status == "timeout"
    r4 = ex.run("SELECT ?q WHERE {?q ?r ?s}")
    assert r4.status == "service" and not r4.is_final and len(s.calls) == 8


def test_retry_after_then_success_and_cache(tmp_path):
    s = Sess([Resp(429, headers={"retry-after": "3"}), select(["u"], [{"u": uri("Q1")}])])
    cache = tmp_path / "c.jsonl"
    ex = Executor(Config(), s, cache_path=cache)
    assert ex.run("SELECT ?u WHERE {?u ?p ?o}").status == "ok" and len(s.calls) == 2
    ex.run("SELECT  ?u WHERE {?u ?p ?o}")                    # espaces différents : même clé
    assert len(s.calls) == 2
    ex2 = Executor(Config(), Sess([]), cache_path=cache)      # relu depuis le disque
    assert ex2.run("SELECT ?u WHERE {?u ?p ?o}").answers == ["Q1"]


def test_service_failures_are_never_cached(tmp_path):
    s = Sess([Resp(503)] * 5 + [select(["u"], [])])
    ex = Executor(Config(exec_retries=5), s, cache_path=tmp_path / "c.jsonl")
    assert ex.run("SELECT ?u WHERE {?u ?p ?o}").status == "service"
    assert ex.run("SELECT ?u WHERE {?u ?p ?o}").status == "vide"   # rejoué, pas mémorisé


# ----------------------------------------------------------------------- métriques
def test_f1_conventions():
    assert f1(set(), set()) == 1.0 and f1(set(), {"a"}) == 0.0 and f1({"a"}, set()) == 0.0
    assert f1({"a", "b"}, {"a", "b"}) == 1.0 and f1({"a"}, None) == 0.0
    assert f1({"a", "b"}, {"a", "c"}) == pytest.approx(0.5)


def test_bootstrap_and_paired_diff_are_deterministic():
    v = [1.0, 0.0, 1.0, 1.0, 0.5, 0.0]
    assert bootstrap_ci(v) == bootstrap_ci(v)
    lo, hi = bootstrap_ci(v)
    assert lo <= sum(v) / len(v) <= hi
    d = paired_diff([1, 1, 0], [0, 1, 1])
    assert (d["gagne"], d["perd"], d["egal"]) == (1, 1, 1) and d["diff"] == 0.0


# --------------------------------------------------------------- exécution complète
GOLD = "SELECT ?u WHERE { wd:Q937 wdt:P19 ?u }"
FLIP = "SELECT ?u WHERE { ?u wdt:P19 wd:Q937 }"


def test_run_execution_end_to_end_with_exclusions(tmp_path):
    items = [{"id": "a", "question": "qa", "gold_sparql": GOLD, "template_id": "1"},
             {"id": "b", "question": "qb", "gold_sparql": GOLD, "template_id": "2"},
             {"id": "c", "question": "qc", "gold_sparql": GOLD, "template_id": "2"}]
    recs = [
        {"id": "a", "status": "ok", "query": GOLD,
         "candidates": {"T3": {"query": GOLD, "ok": True}, "T4": {"query": FLIP, "ok": True},
                        "T3_GOLD": {"query": None, "ok": False}}},
        {"id": "b", "status": "echec_modele", "query": None, "candidates": {}},
        {"id": "c", "status": "infra", "query": None},
    ]
    res = tmp_path / "results.jsonl"
    res.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    ok = select(["u"], [{"u": uri("Q3012")}])
    # a: référence, VOTE (=GOLD, en cache ensuite), T4 (retourné) ; b: référence en cache
    s = Sess([ok, select(["u"], []), ])
    out = tmp_path / "out"
    m = run_execution(res, items, out, Executor(Config(), s))
    assert m["questions_comptees"] == 2 and "c" in m["exclues"] and "infrastructure" in m["exclues"]["c"]
    sysm = m["systemes"]
    assert sysm["VOTE"]["f1"] == 0.5 and sysm["VOTE"]["sans_requete"] == 1   # a : 1.0, b : 0
    assert sysm["T4"]["n"] == 1 and sysm["T4"]["vide"] == 1.0 and sysm["T4"]["f1"] == 0.0
    assert sysm["T3_GOLD"]["sans_requete"] == 1
    assert (out / "metrics.json").exists() and len((out / "exec_results.jsonl").read_text().splitlines()) == 3


def test_service_outage_on_a_system_query_excludes_the_question(tmp_path):
    items = [{"id": "a", "question": "qa", "gold_sparql": GOLD}]
    res = tmp_path / "r.jsonl"
    res.write_text(json.dumps({"id": "a", "status": "ok", "query": FLIP, "candidates": {}}) + "\n")
    s = Sess([select(["u"], [{"u": uri("Q3012")}])] + [Resp(502)] * 5)
    m = run_execution(res, items, tmp_path / "o", Executor(Config(exec_retries=5), s))
    assert m["questions_comptees"] == 0 and "à rejouer" in m["exclues"]["a"]


def test_empty_gold_is_flagged_and_reported_separately(tmp_path):
    items = [{"id": "a", "question": "qa", "gold_sparql": GOLD}]
    res = tmp_path / "r.jsonl"
    res.write_text(json.dumps({"id": "a", "status": "ok", "query": GOLD, "candidates": {}}) + "\n")
    m = run_execution(res, items, tmp_path / "o", Executor(Config(), Sess([select(["u"], [])])))
    assert m["gold_vide"] == 1 and m["systemes"]["VOTE"]["f1"] == 1.0
    assert m["systemes"]["VOTE"]["f1_gold_non_vide"] == {"n": 0, "f1": None}
