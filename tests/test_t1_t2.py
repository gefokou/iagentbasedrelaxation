import pytest
from unittest.mock import patch
from agent1.config import Config
from agent1.falcon_linker import FalconLinker
from agent1.langgraph_workflow import create_kbqa_graph
from agent1.techniques import T1DirectLLM, T2Template
from agent1.types import Candidate, Linking


class FakeLLM:
    def __init__(self, answer="<SPARQL>SELECT ?x WHERE { wd:Q937 wdt:P19 ?x }</SPARQL>"):
        self.answer = answer
        self.calls = []

    def chat(self, system, user):
        self.calls.append((system, user))
        return self.answer


def test_t1_direct_llm():
    cfg = Config()
    llm = FakeLLM()
    t1 = T1DirectLLM(cfg, llm)
    link = Linking(entities={"Einstein": [Candidate("Q937", "Albert Einstein")]},
                   properties={"born": [Candidate("P19", "place of birth")]})
    res = t1.generate("Where was Albert Einstein born?", link)
    assert res.technique == "T1"
    assert res.query == "SELECT ?x WHERE { wd:Q937 wdt:P19 ?x }"
    assert res.error is None
    assert "Q937" in res.allowed_ids


def test_t2_template_forward():
    cfg = Config()
    t2 = T2Template(cfg)
    link = Linking(entities={"Einstein": [Candidate("Q937", "Albert Einstein")]},
                   properties={"born": [Candidate("P19", "place of birth")]})
    res = t2.generate("Where was Albert Einstein born?", link)
    assert res.technique == "T2"
    assert "wd:Q937 wdt:P19 ?obj" in res.query


def test_t2_template_ask():
    cfg = Config()
    t2 = T2Template(cfg)
    link = Linking(entities={"Einstein": [Candidate("Q937", "Albert Einstein")], "Ulm": [Candidate("Q3012", "Ulm")]},
                   properties={"born": [Candidate("P19", "place of birth")]})
    res = t2.generate("Was Albert Einstein born in Ulm?", link)
    assert res.technique == "T2"
    assert res.query.startswith("ASK WHERE { wd:Q937 wdt:P19 wd:Q3012")


def test_t2_template_count():
    cfg = Config()
    t2 = T2Template(cfg)
    link = Linking(entities={"Einstein": [Candidate("Q937", "Albert Einstein")]},
                   properties={"children": [Candidate("P40", "child")]})
    res = t2.generate("How many children did Albert Einstein have?", link)
    assert res.technique == "T2"
    assert "COUNT(DISTINCT ?obj)" in res.query


def test_falcon_linker_mock():
    fake_response = {
        "entities_wikidata": [{"URI": "http://www.wikidata.org/entity/Q937", "surface form": "Albert Einstein"}],
        "relations_wikidata": [{"URI": "http://www.wikidata.org/entity/P19", "surface form": "born"}]
    }
    with patch("urllib.request.urlopen") as mock_url:
        mock_resp = mock_url.return_value.__enter__.return_value
        import json
        mock_resp.read.return_value = json.dumps(fake_response).encode("utf-8")
        linker = FalconLinker()
        res = linker.link("Where was Albert Einstein born?")
        assert res.top_entities()[0].id == "Q937"
        assert res.top_properties()[0].id == "P19"


def test_langgraph_workflow_compilation():
    from agent1.agent import Agent1
    from agent2.executor import Executor
    from agent3.answerer import Agent3
    from agent3.notes import NoteBuilder
    from tests.test_agent1 import FakeWD, KNOWN
    cfg = Config(techniques=("T2",))
    wd = FakeWD()
    a1 = Agent1(cfg, wd=wd)
    ex = Executor(cfg)
    a3 = Agent3(FakeLLM("ans"), NoteBuilder(wd))
    graph = create_kbqa_graph(a1, ex, a3)
    assert graph is not None
