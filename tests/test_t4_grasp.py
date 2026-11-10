"""T4 = GRASP (code officiel) : on teste l'enveloppe, avec un faux exécutable `grasp`."""
import json
import stat
import sys

import pytest

from agent1.config import Config
from agent1.evaluation import classify
from agent1.techniques.t4_grasp import Cache, GraspError, SubprocessRunner, T4Grasp, grasp_env, sparql_of
from agent1.types import Linking

Q = "SELECT ?x WHERE { wd:Q937 wdt:P19 ?x }"
OUT = {"id": "unknown", "output": {"type": "answer", "sparql": Q, "answer": "Ulm"}}


def cfg_for(tmp_path, script):
    return Config(grasp_cmd=f"{sys.executable} {script}", grasp_cache=str(tmp_path / "cache.jsonl"),
                  llm_model="m1", llm_base_url="http://x/v1", llm_api_key_env="FAKE_KEY_ENV")


def test_sparql_of_reads_output_field_and_cancel_best_attempt():
    assert sparql_of(OUT) == Q
    assert sparql_of({"output": {"type": "cancel", "sparql": None}}) is None
    assert sparql_of({"output": "texte"}) is None


def test_env_uses_grasp_variables_and_requires_the_key(monkeypatch):
    cfg = Config(llm_model="m1", llm_base_url="http://x/v1", llm_api_key_env="FAKE_KEY_ENV", reasoning_effort="low")
    monkeypatch.delenv("FAKE_KEY_ENV", raising=False)
    with pytest.raises(GraspError, match="FAKE_KEY_ENV"):
        grasp_env(cfg, {})
    monkeypatch.setenv("FAKE_KEY_ENV", "secret")
    env = grasp_env(cfg, {})
    assert env["MODEL"] == "m1" and env["MODEL_ENDPOINT"] == "http://x/v1"
    assert env["MODEL_PROVIDER"] == "openai/completions" and env["REASONING_EFFORT"] == "low"
    assert env["MODEL_API_KEY"] == "secret" and env["KG"] == "wikidata"


def test_subprocess_runner_passes_question_and_env_not_the_key_on_the_command_line(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEY_ENV", "secret")
    script = tmp_path / "fakegrasp.py"
    script.write_text(
        "import json, os, sys\n"
        f"json.dump(dict(os.environ), open(r'{tmp_path}/env.json','w'))\n"
        f"json.dump(sys.argv[1:], open(r'{tmp_path}/argv.json','w'))\n"
        "print('journal'); print(json.dumps({'output': {'sparql': 'SELECT 1'}}))\n")
    cfg = cfg_for(tmp_path, script)
    out = SubprocessRunner(cfg)("Where was Einstein born?")
    assert out["output"]["sparql"] == "SELECT 1"
    argv = json.loads((tmp_path / "argv.json").read_text())
    assert argv == ["run", "configs/run.yaml", "--input", "Where was Einstein born?"]
    assert "secret" not in " ".join(argv)
    assert json.loads((tmp_path / "env.json").read_text())["MODEL_API_KEY"] == "secret"


def test_generate_uses_cache_so_a_question_costs_one_run(tmp_path):
    calls = []
    cache = Cache(str(tmp_path / "c.jsonl"))
    t4 = T4Grasp(Config(llm_model="m1"), runner=lambda q: calls.append(q) or OUT, cache=cache)
    assert t4.generate("q", Linking()).query == Q
    assert t4.generate("q", Linking()).query == Q
    assert calls == ["q"]
    again = T4Grasp(Config(llm_model="m1"), runner=lambda q: pytest.fail("rejoué"),
                    cache=Cache(str(tmp_path / "c.jsonl")))
    assert again.generate("q", Linking()).query == Q              # relu depuis le fichier
    other = T4Grasp(Config(llm_model="m2"), runner=lambda q: calls.append(q) or OUT, cache=cache)
    other.generate("q", Linking())
    assert calls == ["q", "q"]                                    # autre modèle : autre entrée


def test_no_query_is_a_model_failure_but_a_crash_is_infra():
    from types import SimpleNamespace as NS
    from agent1.types import Checked
    empty = T4Grasp(Config(), runner=lambda q: {"output": {"type": "cancel", "sparql": None}}, cache=Cache(None))
    g = empty.generate("q", Linking())
    assert g.query is None and classify(NS(generated=[g], checked=[Checked("T4", None, False, ["x"])],
                                          query=None))[0] == "echec_modele"
    def boom(q):
        raise GraspError("GRASP a dépassé 900 s")
    g = T4Grasp(Config(), runner=boom, cache=Cache(None)).generate("q", Linking())
    assert "GraspError" in g.error and classify(NS(generated=[g], checked=[Checked("T4", None, False, ["x"])],
                                                   query=None))[0] == "infra"


def test_agent1_votes_between_dfsl_official_and_grasp():
    from tests.test_agent1 import make_agent
    agent, _ = make_agent(Q, Q)                       # T3 = DFSL officiel, T4 = faux GRASP
    r = agent.run("Where was Albert Einstein born?")
    assert [g.technique for g in r.generated] == ["T3", "T4"] and r.rule == "accord"
