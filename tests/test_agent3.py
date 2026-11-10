"""Tests de l'Agent 3 et du pipeline avec de faux clients."""
from agent1.types import Candidate, Linking
from agent2.executor import Execution
from agent3.answerer import Agent3, fidelity
from agent3.notes import NoteBuilder

E = "http://www.wikidata.org/entity/"
P = "http://www.wikidata.org/prop/direct/"


def uri(x):
    return {"type": "uri", "value": x}


class FakeWD:
    def __init__(self):
        self.queries = []

    def info(self, ids, langs=()):
        names = {"Q3012": ("Ulm", "city in Germany"), "Q937": ("Albert Einstein", "physicist")}
        return {i: {"exists": True, "label": names.get(i, ("", ""))[0],
                    "description": names.get(i, ("", ""))[1]} for i in ids}

    def sparql(self, q):
        self.queries.append(q)
        return [
            {"e": uri(E + "Q3012"), "p": uri(P + "P17"), "pl": {"type": "literal", "value": "country"},
             "o": uri(E + "Q183"), "ol": {"type": "literal", "value": "Germany"}},
            {"e": uri(E + "Q3012"), "p": uri(P + "P1082"), "pl": {"type": "literal", "value": "population"},
             "o": {"type": "literal", "value": "126000"}},
            {"e": uri(E + "Q937"), "p": uri(P + "P569"), "pl": {"type": "literal", "value": "date of birth"},
             "o": {"type": "literal", "value": "1879-03-14T00:00:00Z"}},
        ]


class FakeLLM:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def chat(self, system, user, temperature=None, max_tokens=None):
        self.prompts.append((system, user))
        return self.reply


def ok_exec():
    return Execution("ok", answers=["Q3012"], n_rows=1, variables=["u"], preview=[{"u": "Q3012"}])


def link():
    return Linking(entities={"Einstein": [Candidate("Q937", "Albert Einstein")]})


def test_notes_are_built_in_one_query_with_labels_and_clean_dates():
    wd = FakeWD()
    notes = NoteBuilder(wd).build(["Q3012", "Q937", "bad", "Q3012"])
    assert len(wd.queries) == 1 and [n.id for n in notes] == ["Q3012", "Q937"]
    assert "The country of Ulm is Germany (Q183)." in notes[0].render()
    assert "[N1] Ulm (Q3012) — city in Germany" in notes[0].render()
    assert "The date of birth of Albert Einstein is 1879-03-14." in notes[1].render()
    assert "T00:00" not in notes[1].render()
    assert NoteBuilder(wd).build([]) == [] and len(wd.queries) == 1


def test_answer_is_grounded_and_prompt_contains_only_result_and_notes():
    llm = FakeLLM("Albert Einstein est né à Ulm [N1].")
    a3 = Agent3(llm, NoteBuilder(FakeWD()))
    r = a3.run("Où est né Albert Einstein ?", ok_exec(), link())
    assert r.status == "repondu" and r.fidelity["supported"] and r.fidelity["cites_a_note"]
    system, user = llm.prompts[0]
    assert "ONLY" in system and "Query result:" in user and "[N1] Ulm (Q3012)" in user
    assert "[N2] Albert Einstein (Q937)" in user


def test_unsupported_number_is_flagged():
    a3 = Agent3(FakeLLM("Ulm compte 500000 habitants [N1]."), NoteBuilder(FakeWD()))
    r = a3.run("Population d'Ulm ?", ok_exec(), link())
    assert r.status == "repondu" and not r.fidelity["supported"] and "500000" in r.fidelity["unsupported"]
    assert fidelity("Ulm compte 126000 habitants [N1]", "population: 126000\n[N1] Ulm")["supported"]
    assert fidelity("Voir [N9]", "[N1] Ulm")["unknown_citations"] == ["N9"]


def test_insufficient_and_no_answer_paths_do_not_invent():
    a3 = Agent3(FakeLLM("INSUFFISANT"), NoteBuilder(FakeWD()))
    assert a3.run("q", ok_exec(), link()).status == "insuffisant"
    llm = FakeLLM("ne doit pas être appelé")
    a3 = Agent3(llm, NoteBuilder(FakeWD()))
    for st in ("vide", "syntaxe", "timeout"):
        r = a3.run("q", Execution(st), link())
        assert r.status == "sans_reponse" and r.exec_status == st and "Aucune réponse" in r.answer
    assert a3.run("q", None, link()).exec_status == "sans_requete" and llm.prompts == []


def test_ask_result_and_infra_error_are_explicit():
    llm = FakeLLM("Oui [N1].")
    a3 = Agent3(llm, NoteBuilder(FakeWD()))
    a3.run("Einstein est-il né à Ulm ?", Execution("ok", answers=["true"], ask=True, n_rows=1), link())
    assert "yes/no question): yes" in llm.prompts[0][1]

    class Down(FakeWD):
        def sparql(self, q):
            raise RuntimeError("Wikidata HS")
    r = Agent3(FakeLLM("x"), NoteBuilder(Down())).run("q", ok_exec(), link())
    assert r.status == "erreur" and "Wikidata HS" in r.error and r.answer == ""


def test_pattern_sentences_are_deterministic_and_grammatical():
    from agent3.notes import verbalise
    assert verbalise("Einstein", "P31", "instance of", ["human"]) == "Einstein is an instance of human."
    assert verbalise("Einstein", "P19", "place of birth", ["Ulm"]) == "The place of birth of Einstein is Ulm."
    assert verbalise("Einstein", "P69", "educated at", ["ETH", "UZH"]) == "Einstein is educated at ETH and UZH."
    assert verbalise("Einstein", "P40", "child", ["A", "B", "C"]) == "The child of Einstein are A, B and C."
    assert verbalise("Q1", "P1", "", ["x"]) == "The P1 of Q1 is x."


def test_notes_are_consolidated_per_subject_and_ranked_by_question_similarity():
    import numpy as np

    class Enc:     # la phrase qui parle de « population » est la plus proche de la question
        def encode(self, texts):
            return np.array([[1.0 if "population" in t else 0.0, 1.0 if "country" in t else 0.2]
                             for t in texts], dtype=np.float32)
    wd = FakeWD()
    notes = NoteBuilder(wd, encoder=Enc(), max_props=1).build(["Q3012"], "Quelle est la population ?")
    assert len(notes) == 1 and len(notes[0].sentences) == 1           # une note par sujet
    assert "population" in notes[0].sentences[0]
    both = NoteBuilder(wd).build(["Q3012"])[0]
    assert len(both.sentences) == 2 and both.render().count("[N1]") == 1


def test_spaced_citations_are_recognised_and_media_properties_are_excluded():
    f = fidelity("Né à Ulm [ N2 ]. [ N1 ]", "[N1] Ulm\n[N2] Einstein")
    assert f["supported"] and f["cites_a_note"] and f["unknown_citations"] == []
    wd = FakeWD()
    NoteBuilder(wd).build(["Q3012"])
    q = wd.queries[0]
    assert "wikibase:CommonsMedia" in q and "wikibase:ExternalId" in q and "wikibase:Url" in q


def test_prompt_carries_query_language_and_count_rule():
    llm = FakeLLM("Il y en a 1 [N1].")
    lk = link()
    lk.language = "fr"
    q = "SELECT (COUNT(?x) AS ?v) WHERE { ?x wdt:P974 wd:Q131792 }"
    Agent3(llm, NoteBuilder(FakeWD())).run("Combien ?", ok_exec(), lk, q)
    system, user = llm.prompts[0]
    assert "Answer language: fr" in user and q in user
    assert "never recount" in system and "direction" in system


def test_number_words_and_token_level_matching_catch_miscounts():
    from agent3.answerer import expected_value, fidelity
    # q07 réel : le résultat est 5, le LLM a écrit « three » d'après les notes
    f = fidelity("Ska has three countries of origin [N1].", "Query result:\nvalue=5\n[N1] Ska", "5")
    assert not f["supported"] and f["result_value_missing"] == "5" and "3" in f["unsupported"]
    assert fidelity("Ska has 5 countries [N1].", "Query result:\nvalue=5\n[N1] Ska", "5")["supported"]
    assert fidelity("Il y en a cinq [N1].", "value=5\n[N1] x", "5")["supported"]
    assert "3" in fidelity("3 pays [N1]", "année 1829 [N1]")["unsupported"]    # « 3 » n'est pas dans « 1829 »
    assert fidelity("Né en 1879 [N1]", "date of birth 1879-03-14 [N1]")["supported"]
    assert expected_value(Execution("ok", preview=[{"value": "5"}])) == "5"
    assert expected_value(Execution("ok", preview=[{"u": "Q1"}])) is None
    assert expected_value(Execution("ok", preview=[{"a": "1"}, {"a": "2"}])) is None


def test_count_mismatch_triggers_one_corrective_retry():
    class Seq(FakeLLM):
        def chat(self, system, user, temperature=None, max_tokens=None):
            self.prompts.append((system, user))
            return ["Ska has three countries [N1].", "Ska has 5 countries of origin [N1]."][len(self.prompts) - 1]
    llm = Seq("")
    ex = Execution("ok", answers=["5"], n_rows=1, variables=["value"], preview=[{"value": "5"}])
    r = Agent3(llm, NoteBuilder(FakeWD())).run("How many?", ex, link())
    assert len(llm.prompts) == 2 and "the query result is 5" in llm.prompts[1][1]
    assert r.answer.startswith("Ska has 5") and r.fidelity["supported"] and r.fidelity["retried"]
