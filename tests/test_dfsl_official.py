"""DFSL officiel : code des auteurs (external/dfsl) + appel modèle compatible OpenAI."""
import json
import zlib

import pytest
import torch

from agent1.llm import LLMError
from dfsl_official import StopForQuota, run_dfsl, to_dfsl_format

P = "PREFIX wd: <http://www.wikidata.org/entity/>\nPREFIX wdt: <http://www.wikidata.org/prop/direct/>\n"


def fake_encode(texts):
    out = torch.zeros(len(texts), 1024)
    for i, t in enumerate(texts):
        for w in t.lower().replace("?", " ").replace("(", " ").replace(")", " ").split():
            out[i, zlib.crc32(w.encode()) % 1024] += 1
    return out


def ex(q, s, e=(), r=()):
    return {"question": q, "sparql": s,
            "entities": [{"id": i, "label": l} for i, l in e],
            "relations": [{"id": i, "label": l} for i, l in r]}


STORE = [
    ex("Where was Johann Sebastian Bach born?", "SELECT ?u WHERE { wd:Q1339 wdt:P19 ?u }",
       [("Q1339", "Johann Sebastian Bach")], [("P19", "place of birth")]),
    ex("Which is the capital of France?", "SELECT ?u WHERE { wd:Q142 wdt:P36 ?u }",
       [("Q142", "France")], [("P36", "capital")]),
    ex("Who is the mother of Marie Curie?", "SELECT ?u WHERE { wd:Q7186 wdt:P25 ?u }",
       [("Q7186", "Marie Curie")], [("P25", "mother")]),
]
TEST = [{"id": "q1", "question": "Where was Ludwig van Beethoven born?",
         "gold_sparql": "SELECT ?x WHERE { wd:Q255 wdt:P19 ?x }"}]
LABELS = {"Q255": "Ludwig van Beethoven", "P19": "place of birth"}


class FakeLLM:
    def __init__(self, reply="<SPARQL>\nSELECT ?x WHERE { wd:Q255 wdt:P19 ?x }\n</SPARQL>", fail_after=None):
        self.prompts, self.reply, self.fail_after = [], reply, fail_after

    def chat(self, system, user, **kw):
        assert system == ""
        if self.fail_after is not None and len(self.prompts) >= self.fail_after:
            raise LLMError("HTTP 429: quota")
        self.prompts.append(user)
        return self.reply


def run(tmp_path, llm, test=TEST, **kw):
    return run_dfsl(STORE, test, tmp_path, llm, LABELS, k=2, encode=fake_encode, min_store=1,
                    log=lambda *_: None, **kw)


def test_format_follows_the_paper_figure_uris_labels_and_prefixes():
    train, tst = to_dfsl_format(STORE, TEST, LABELS)
    assert train["s0"]["entities"] == [["http://www.wikidata.org/entity/Q1339", "Johann Sebastian Bach"]]
    assert train["s0"]["relations"] == [["http://www.wikidata.org/prop/direct/P19", "place of birth"]]
    assert train["s0"]["query"].startswith("PREFIX wd:") and "PREFIX wdt:" in train["s0"]["query"]
    assert tst["q1"]["entities"] == [["http://www.wikidata.org/entity/Q255", "Ludwig van Beethoven"]]


def test_prompt_is_the_authors_prompt_nearest_example_first_with_gold_entities(tmp_path):
    llm = FakeLLM()
    s = run(tmp_path, llm)
    p = llm.prompts[0]
    assert s["nouvelles"] == 1 and "Utilize all provided golden entities" in p
    near = json.loads((tmp_path / "dfsl_similar.json").read_text())["q1"]
    assert near[0] == STORE[0]["question"]                                # Bach est le plus proche
    assert p.index(near[0]) < p.index(near[1])                           # le plus proche d'abord
    assert p.rstrip().endswith("Query:")
    assert "Question: Where was Ludwig van Beethoven born?" in p
    assert "http://www.wikidata.org/entity/Q255 (Ludwig van Beethoven)" in p   # entités gold
    rec = json.loads((tmp_path / "dfsl_results.jsonl").read_text().splitlines()[0])
    assert rec["query"].strip() == "SELECT ?x WHERE { wd:Q255 wdt:P19 ?x }"
    assert json.loads((tmp_path / "dfsl_similar.json").read_text())["q1"][0] == STORE[0]["question"]


def test_resume_does_not_redo_finished_questions_and_max_questions(tmp_path):
    test = TEST + [{"id": "q2", "question": "Capital of Germany?", "gold_sparql": "SELECT ?x WHERE { wd:Q183 wdt:P36 ?x }"}]
    llm = FakeLLM()
    assert run(tmp_path, llm, test, max_questions=1)["nouvelles"] == 1
    assert run(tmp_path, llm, test)["nouvelles"] == 1 and len(llm.prompts) == 2
    assert run(tmp_path, llm, test)["nouvelles"] == 0 and len(llm.prompts) == 2


def test_quota_failure_stops_cleanly_and_nothing_is_recorded_for_that_question(tmp_path):
    test = TEST + [{"id": "q2", "question": "Capital of Germany?", "gold_sparql": "SELECT ?x WHERE { wd:Q183 wdt:P36 ?x }"}]
    with pytest.raises(StopForQuota, match="reprendre"):
        run(tmp_path, FakeLLM(fail_after=1), test)
    ids = [json.loads(l)["id"] for l in (tmp_path / "dfsl_results.jsonl").read_text().splitlines()]
    assert ids == ["q1"]


def test_reply_without_tags_is_recorded_as_no_query(tmp_path):
    s = run(tmp_path, FakeLLM(reply="Je ne sais pas."))
    rec = json.loads((tmp_path / "dfsl_results.jsonl").read_text().splitlines()[0])
    assert rec["query"] is None and "aucune requête" in rec["error"] and s["sans_requete"] == 1


def test_test_question_itself_is_removed_from_the_store(tmp_path):
    leaky = STORE + [ex("where was ludwig van beethoven BORN", "SELECT ?x WHERE { wd:Q255 wdt:P19 ?x }")]
    s = run_dfsl(leaky, TEST, tmp_path, FakeLLM(), LABELS, k=2, encode=fake_encode, min_store=1,
                 log=lambda *_: None)
    assert s["fuites_retirees"] == 1 and s["stock_utilise"] == 3


def test_extras_only_execution_counts_only_processed_questions(tmp_path):
    from agent2.run import run_execution
    from agent2.executor import Execution

    class Ex:
        def run(self, q):
            return Execution(status="ok", answers=["Q1"], variables=["x"], n_rows=1) if "wd:Q1" in q \
                else Execution(status="ok", answers=["Q2"], variables=["x"], n_rows=1)
    test = [{"id": "a", "question": "x", "gold_sparql": "SELECT ?x WHERE { wd:Q1 wdt:P1 ?x }"},
            {"id": "b", "question": "y", "gold_sparql": "SELECT ?x WHERE { wd:Q1 wdt:P1 ?x }"}]
    f = tmp_path / "d.jsonl"
    f.write_text(json.dumps({"id": "a", "query": "SELECT ?x WHERE { wd:Q1 wdt:P1 ?x }"}) + "\n")
    m = run_execution(None, test, tmp_path / "o", Ex(), {"T3_GOLD": f})
    assert m["questions_comptees"] == 1 and "T3_GOLD" in m["systemes"] and "VOTE" not in m["systemes"]
    assert m["sans_resultat_agent1"] == ["b"]


def test_entities_without_label_do_not_crash_the_authors_encoder():
    train, tst = to_dfsl_format(STORE, [{"id": "t", "question": "q", "gold_sparql": "SELECT ?x WHERE { wd:Q999 wdt:P1 ?x }"}], {})
    assert tst["t"]["entities"] == ["http://www.wikidata.org/entity/Q999"]
