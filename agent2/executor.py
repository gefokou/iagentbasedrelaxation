"""Agent 2 : exécute UNE FOIS la requête sur le vrai service Wikidata (WDQS).

Statuts : ok (réponse non vide) | vide | syntaxe (rejet 400) | timeout | service.
`service` (429, 5xx autres que timeout, réseau coupé après toutes les reprises) n'est PAS un
verdict sur la requête : il est signalé « à rejouer » et jamais mis en cache ni compté.
Aucune réparation, aucune seconde tentative de la requête elle-même.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests

from agent1.config import Config

from .answers import answers_from_ask, answers_from_select

STATUSES = ("ok", "vide", "syntaxe", "timeout", "service")


@dataclass
class Execution:
    status: str
    answers: list[str] = field(default_factory=list)   # ensemble normalisé, trié
    ask: bool = False
    n_rows: int = 0
    variables: list[str] = field(default_factory=list)
    error: str = ""
    seconds: float = 0.0
    # aperçu brut (30 premières lignes, variable -> valeur ; entités réduites à leur QID),
    # lu par l'Agent 3 pour verbaliser ; l'ensemble normalisé `answers` sert au F1
    preview: list[dict] = field(default_factory=list)

    @property
    def executable(self) -> bool:     # la requête a été acceptée et a tourné
        return self.status in ("ok", "vide")

    @property
    def is_final(self) -> bool:       # verdict sur la requête (≠ panne du service)
        return self.status != "service"

    def answer_set(self) -> set[str]:
        return set(self.answers)

    def to_dict(self) -> dict:
        return asdict(self)


def query_key(query: str) -> str:
    return hashlib.sha1(" ".join(query.split()).encode("utf-8")).hexdigest()


class Executor:
    def __init__(self, cfg: Config | None = None, session: requests.Session | None = None,
                 cache_path: str | Path | None = None):
        self.cfg = cfg or Config()
        self.s = session or requests.Session()
        self.s.headers.update({"User-Agent": self.cfg.user_agent})
        self._last = 0.0
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, dict] = {}
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._cache[rec["key"]] = rec["execution"]

    # ------------------------------------------------------------------ cache
    def _remember(self, key: str, ex: Execution) -> None:
        self._cache[key] = ex.to_dict()
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "execution": ex.to_dict()},
                                   ensure_ascii=False) + "\n")

    def _throttle(self) -> None:
        wait = self.cfg.wd_min_interval_s - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    # -------------------------------------------------------------- exécution
    def run(self, query: str) -> Execution:
        q = (query or "").strip()
        if not q:
            return Execution("syntaxe", error="requête vide")
        key = query_key(q)
        if key in self._cache:
            return Execution(**self._cache[key])
        ex = self._execute(q)
        if ex.is_final:
            self._remember(key, ex)
        return ex

    def _execute(self, q: str) -> Execution:
        t0 = time.time()
        last = ""
        for attempt in range(self.cfg.exec_retries):
            self._throttle()
            try:
                r = self.s.post(self.cfg.wd_sparql, data={"query": q},
                                headers={"Accept": "application/sparql-results+json"},
                                timeout=self.cfg.exec_timeout_s)
            except requests.Timeout:
                return Execution("timeout", error="délai client dépassé",
                                 seconds=time.time() - t0)
            except requests.RequestException as e:
                last = f"réseau : {type(e).__name__}: {e}"
                time.sleep(min(2 ** attempt, 30))
                continue
            code, body = r.status_code, (getattr(r, "text", "") or "")
            if code == 200:
                try:
                    return self._parse(r.json(), time.time() - t0)
                except (ValueError, KeyError) as e:
                    return Execution("service", error=f"réponse illisible : {e}",
                                     seconds=time.time() - t0)
            if code == 400 or "MalformedQueryException" in body:
                return Execution("syntaxe", error=_first_line(body) or "HTTP 400",
                                 seconds=time.time() - t0)
            if "TimeoutException" in body or "timeout" in body.lower()[:400]:
                return Execution("timeout", error=_first_line(body)[:200],
                                 seconds=time.time() - t0)
            last = f"HTTP {code}"
            if code in (429, 500, 502, 503, 504):
                ra = str(r.headers.get("retry-after", "")).strip()
                delay = float(ra) if ra.replace(".", "", 1).isdigit() else 2 ** attempt * 2
                time.sleep(min(delay, 120))
                continue
            break
        return Execution("service", error=last, seconds=time.time() - t0)

    @staticmethod
    def _parse(data: dict, seconds: float) -> Execution:
        if "boolean" in data:
            return Execution("ok", answers=sorted(answers_from_ask(bool(data["boolean"]))),
                             ask=True, n_rows=1, seconds=seconds)
        variables = data["head"]["vars"]
        bindings = data["results"]["bindings"]
        answers = sorted(answers_from_select(variables, bindings))
        preview = [{v: _raw(row[v]) for v in variables if v in row} for row in bindings[:30]]
        return Execution("ok" if answers else "vide", answers=answers, n_rows=len(bindings),
                         variables=variables, seconds=seconds, preview=preview)


def _raw(cell: dict) -> str:
    v = cell.get("value", "")
    if cell.get("type") == "uri" and "wikidata.org/entity/" in v:
        return v.rsplit("/", 1)[-1]
    return v


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()[:300]
    return ""
