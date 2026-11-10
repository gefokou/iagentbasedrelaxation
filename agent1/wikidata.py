"""Accès à Wikidata : recherche d'entités/propriétés, existence, libellés.

Utilisé par la liaison (T3) et le filtre. Le point d'accès SPARQL sert à l'Agent 3 (notes).
"""
from __future__ import annotations

import re
import time

import requests

from .config import Config
from .types import Candidate

_ID = re.compile(r"^[QP][1-9]\d{0,9}$")   # format Wikibase : 10 chiffres au plus


class WikidataError(RuntimeError):
    pass


class WikidataClient:
    def __init__(self, cfg: Config, session: requests.Session | None = None):
        self.cfg = cfg
        self.s = session or requests.Session()
        self.s.headers.update({"User-Agent": cfg.user_agent})
        self._labels: dict[str, dict] = {}
        self._last = 0.0
        self._search_cache: dict[tuple, list[Candidate]] = {}

    # ------------------------------------------------------------------ HTTP
    def _throttle(self) -> None:
        wait = self.cfg.wd_min_interval_s - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def _get(self, url: str, params: dict, headers: dict | None = None) -> dict:
        """GET avec cadence minimale, respect de Retry-After et reprises (429, 5xx)."""
        last = ""
        for attempt in range(self.cfg.wd_retries):
            self._throttle()
            try:
                r = self.s.get(url, params=params, headers=headers,
                               timeout=self.cfg.wd_timeout_s)
            except requests.RequestException as e:
                last = str(e)
                time.sleep(min(2 ** attempt, 60))
                continue
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}"
            if r.status_code in (429, 500, 502, 503, 504):
                ra = str(r.headers.get("retry-after", "")).strip()
                delay = float(ra) if ra.replace(".", "", 1).isdigit() else 2 ** attempt * 2
                time.sleep(min(delay, 120))
                continue
            break
        raise WikidataError(last)

    # ------------------------------------------------------- recherche (NEL)
    def search(self, text: str, lang: str = "en", kind: str = "item",
               limit: int = 5) -> list[Candidate]:
        """wbsearchentities ; kind = 'item' ou 'property'."""
        key = (text.lower(), lang, kind, limit)
        if key in self._search_cache:
            return self._search_cache[key]
        data = self._get(self.cfg.wd_api, {
            "action": "wbsearchentities", "search": text, "language": lang,
            "uselang": lang, "type": kind, "limit": limit, "format": "json"})
        out = [Candidate(id=h["id"], label=h.get("label", ""),
                         description=h.get("description", ""))
               for h in data.get("search", [])]
        self._search_cache[key] = out
        return out

    # ------------------------------------------------ libellés et existence
    def info(self, ids: list[str], langs: tuple[str, ...] = ("en", "fr")) -> dict[str, dict]:
        """id -> {'exists': bool, 'label': str, 'description': str}. Mis en cache."""
        for i in dict.fromkeys(ids):   # forme invalide : n'existe pas, sans appeler l'API
            if not _ID.match(i) and re.match(r"^[QP]\d+$", i):
                self._labels[i] = {"exists": False, "label": "", "description": ""}
        todo = [i for i in dict.fromkeys(ids) if _ID.match(i) and i not in self._labels]
        for i in range(0, len(todo), 50):
            chunk = todo[i:i + 50]
            data = self._get(self.cfg.wd_api, {
                "action": "wbgetentities", "ids": "|".join(chunk),
                "props": "labels|descriptions", "languages": "|".join(langs),
                "format": "json"})
            if "entities" not in data:   # erreur d'API : ne jamais la lire comme « absent »
                raise WikidataError(f"wbgetentities : {data.get('error', data)}")
            ents = data["entities"]
            for eid in chunk:
                e = ents.get(eid, {"missing": ""})
                if "missing" in e:
                    self._labels[eid] = {"exists": False, "label": "", "description": ""}
                    continue
                lab = ""
                desc = ""
                for lg in langs:
                    lab = lab or e.get("labels", {}).get(lg, {}).get("value", "")
                    desc = desc or e.get("descriptions", {}).get(lg, {}).get("value", "")
                self._labels[eid] = {"exists": True, "label": lab, "description": desc}
        return {i: self._labels[i] for i in ids if i in self._labels}

    def label(self, eid: str) -> str:
        return self.info([eid]).get(eid, {}).get("label", "")

    # ---------------------------------------------------------------- SPARQL
    def sparql(self, query: str) -> list[dict]:
        data = self._get(self.cfg.wd_sparql, {"query": query, "format": "json"},
                         headers={"Accept": "application/sparql-results+json"})
        return data["results"]["bindings"]
