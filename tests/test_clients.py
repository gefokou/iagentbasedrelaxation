"""Tests des clients HTTP avec une fausse session (réponses au format réel des API)."""
import pytest

from agent1.config import Config
from agent1.llm import LLMClient, LLMError
from agent1.wikidata import WikidataClient


class Resp:
    def __init__(self, code=200, data=None, headers=None):
        self.status_code, self._d, self.headers, self.text = code, data, headers or {}, str(data)

    def json(self):
        return self._d


class Sess:
    def __init__(self, responses):
        self.responses, self.calls, self.headers = list(responses), [], {}

    def _next(self, url, **kw):
        self.calls.append((url, kw))
        r = self.responses.pop(0)
        return r(url, kw) if callable(r) else r

    get = post = lambda self, url, **kw: self._next(url, **kw)


def uri(x):
    return {"type": "uri", "value": x}


def lit(x, lang="en"):
    return {"type": "literal", "value": x, "xml:lang": lang}


# ------------------------------------------------------------------------- LLM
def test_llm_retries_on_429_strips_think_and_sends_auth(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setenv("GROQ_API_KEY", "k")
    ok = Resp(200, {"choices": [{"message": {"content": "<think>x</think>réponse"}}]})
    s = Sess([Resp(429, "rate", {"retry-after": "0"}), ok])
    out = LLMClient(Config(), s).chat("sys", "user")
    assert out == "réponse" and len(s.calls) == 2
    url, kw = s.calls[-1]
    assert url.endswith("/chat/completions") and kw["headers"]["Authorization"] == "Bearer k"
    assert kw["json"]["temperature"] == 0.0 and kw["json"]["model"] == "llama-3.3-70b-versatile"


def test_llm_errors(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        LLMClient(Config(), Sess([])).chat("s", "u")
    monkeypatch.setenv("GROQ_API_KEY", "k")
    with pytest.raises(LLMError, match="401"):
        LLMClient(Config(), Sess([Resp(401, "no")])).chat("s", "u")


# -------------------------------------------------------------------- Wikidata
def test_search_and_info_parse_real_formats():
    s = Sess([
        Resp(200, {"search": [{"id": "Q937", "label": "Albert Einstein",
                               "description": "physicist"}]}),
        Resp(200, {"entities": {
            "Q937": {"labels": {"en": {"value": "Albert Einstein"}}, "descriptions": {}},
            "Q99999999": {"id": "Q99999999", "missing": ""}}}),
    ])
    wd = WikidataClient(Config(), s)
    c = wd.search("Einstein")[0]
    assert (c.id, c.label, c.description) == ("Q937", "Albert Einstein", "physicist")
    wd.search("Einstein")                      # mis en cache
    info = wd.info(["Q937", "Q99999999", "bad id"])
    assert info["Q937"]["exists"] and info["Q937"]["label"] == "Albert Einstein"
    assert info["Q99999999"]["exists"] is False and "bad id" not in info
    assert len(s.calls) == 2
    assert "User-Agent" in wd.s.headers


def test_wikidata_honours_retry_after_and_throttles(monkeypatch):
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    s = Sess([Resp(429, "slow", {"retry-after": "7"}),
              Resp(200, {"search": [{"id": "Q1", "label": "x"}]})])
    wd = WikidataClient(Config(), s)
    assert wd.search("x")[0].id == "Q1"
    assert 7.0 in sleeps and len(s.calls) == 2


def test_wikidata_gives_up_after_configured_retries(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    cfg = Config(wd_retries=3)
    s = Sess([Resp(429, "x")] * 3)
    from agent1.wikidata import WikidataError
    with pytest.raises(WikidataError, match="429"):
        WikidataClient(cfg, s).search("x")
    assert len(s.calls) == 3


def test_info_never_reads_an_api_error_as_missing():
    from agent1.wikidata import WikidataError
    s = Sess([Resp(200, {"error": {"code": "no-such-entity", "info": "x"}})])
    with pytest.raises(WikidataError, match="no-such-entity"):
        WikidataClient(Config(), s).info(["Q937"])


def test_info_invalid_id_shape_is_absent_without_calling_api():
    s = Sess([Resp(200, {"entities": {"Q937": {"labels": {"en": {"value": "Einstein"}}}}})])
    info = WikidataClient(Config(), s).info(["Q937", "Q99999999999"])   # 11 chiffres
    assert info["Q937"]["exists"] and info["Q99999999999"]["exists"] is False
    assert "Q99999999999" not in s.calls[0][1]["params"]["ids"]


def test_model_can_be_overridden_by_argument_or_env(monkeypatch):
    monkeypatch.delenv("AGENT1_MODEL", raising=False)
    assert Config().with_preset("groq_llama").llm_model == "llama-3.3-70b-versatile"
    assert Config().with_preset("groq_llama", "openai/gpt-oss-120b").llm_model == "openai/gpt-oss-120b"
    monkeypatch.setenv("AGENT1_MODEL", "autre-modele")
    c = Config().with_preset("groq_llama")
    assert c.llm_model == "autre-modele" and c.llm_base_url.endswith("groq.com/openai/v1")
    assert Config().with_preset("openai_gpt41", "x").llm_base_url.startswith("https://api.openai.com")


def test_llm_empty_reply_is_an_error_and_reasoning_effort_is_sent(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    empty = Resp(200, {"choices": [{"message": {"content": "", "reasoning": "..."}}]})
    cfg = Config(reasoning_effort="low")
    s = Sess([empty])
    with pytest.raises(LLMError, match="réponse vide"):
        LLMClient(cfg, s).chat("s", "u")
    assert s.calls[0][1]["json"]["reasoning_effort"] == "low"
    s2 = Sess([Resp(200, {"choices": [{"message": {"content": "ok"}}]})])
    LLMClient(Config(reasoning_effort=None), s2).chat("s", "u")
    assert "reasoning_effort" not in s2.calls[0][1]["json"]
