"""Chaînage des trois agents (pipeline.run_one) avec de faux services."""
from agent1.config import Config
from agent2.executor import Executor
from agent3.answerer import Agent3
from agent3.notes import NoteBuilder
from pipeline import run_one
from test_agent1 import GOOD, make_agent
from test_agent2 import E, Resp, Sess, select, uri
from test_agent3 import FakeLLM, FakeWD


def test_three_agents_are_chained_end_to_end():
    a1, _ = make_agent(GOOD, GOOD)
    sess = Sess([select(["uri"], [{"uri": uri("Q3012")}])])
    a2 = Executor(Config(), sess)
    llm3 = FakeLLM("Albert Einstein est né à Ulm [N1].")
    a3 = Agent3(llm3, NoteBuilder(FakeWD()))
    out = run_one("Where was Albert Einstein born?", a1, a2, a3)
    # Agent 1 -> Agent 2 : la requête retenue est celle qui a été exécutée
    assert sess.calls[0][1]["data"]["query"] == out["query"] and "Q937" in out["query"]
    # Agent 2 -> Agent 3 : le résultat exécuté et les notes arrivent dans le prompt de l'Agent 3
    prompt = llm3.prompts[0][1]
    assert "Query result:" in prompt and "uri=Q3012" in prompt and "[N1] Ulm (Q3012)" in prompt
    assert out["execution"]["status"] == "ok" and out["answer"]["status"] == "repondu"
    assert out["answer"]["fidelity"]["supported"]


def test_failed_query_flows_through_without_calling_the_answer_llm():
    a1, _ = make_agent(GOOD, GOOD)
    a2 = Executor(Config(), Sess([Resp(400, text="MalformedQueryException: x")]))
    llm3 = FakeLLM("ne doit pas servir")
    out = run_one("Where was Albert Einstein born?", a1, a2, Agent3(llm3, NoteBuilder(FakeWD())))
    assert out["execution"]["status"] == "syntaxe" and out["answer"]["status"] == "sans_reponse"
    assert llm3.prompts == []


def test_no_query_from_agent1_means_no_execution():
    a1, _ = make_agent("SELECT broken {{", "rien")
    sess = Sess([])
    out = run_one("Where was Albert Einstein born?", a1, Executor(Config(), sess),
                  Agent3(FakeLLM("x"), NoteBuilder(FakeWD())))
    assert out["query"] is None and out["execution"] is None and sess.calls == []
    assert out["answer"]["exec_status"] == "sans_requete"


def test_batch_isolates_failures_and_resumes(tmp_path, monkeypatch):
    import json
    import pipeline
    calls = []

    def fake_run_one(q, a1, a2, a3):
        calls.append(q)
        if q == "b" and calls.count("b") == 1:
            raise ConnectionError("coupure réseau")
        return {"question": q, "answer": {"status": "repondu", "answer": "ok " + q}}
    monkeypatch.setattr(pipeline, "run_one", fake_run_one)
    items = [{"id": "1", "question": "a"}, {"id": "2", "question": "b"}, {"id": "3", "question": "c"}]
    path = tmp_path / "out" / "p.jsonl"
    assert pipeline.run_batch(items, None, None, None, path) == 1        # « b » en erreur, « c » traité
    recs = {json.loads(l)["id"]: json.loads(l) for l in path.read_text().splitlines()}
    assert "pipeline_error" in recs["2"] and recs["3"]["answer"]["answer"] == "ok c"
    assert pipeline.run_batch(items, None, None, None, path) == 0        # reprise : seule « b » rejouée
    assert calls == ["a", "b", "c", "b"]
    assert all("pipeline_error" not in json.loads(l) for l in path.read_text().splitlines())


def test_fidelity_accepts_fullwidth_brackets_zero_width_and_result_tag():
    from agent3.answerer import fidelity
    f = fidelity("Ulm【N1】 et [​N2] [Result]", "[N1] a\n[N2] b")
    assert f["unknown_citations"] == [] and f["cites_a_note"] and f["cites_result"]
