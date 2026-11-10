"""Falcon 2.0 Client for Entity & Relation Linking on Wikidata.

Official service: https://labs.tib.eu/falcon/falcon2/api
"""
from __future__ import annotations

import json
import re
import urllib.request
from .types import Candidate, Linking


class FalconLinker:
    ENDPOINT = "https://labs.tib.eu/falcon/falcon2/api?mode=long"

    def __init__(self, timeout_s: float = 15.0):
        self.timeout_s = timeout_s

    def link(self, question: str) -> Linking:
        req = urllib.request.Request(
            self.ENDPOINT,
            data=json.dumps({"text": question}).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "KBQA-Falcon/1.0"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return Linking()

        ents: dict[str, list[Candidate]] = {}
        for item in data.get("entities_wikidata", []):
            uri = item.get("URI", "")
            surface = item.get("surface form", "")
            m = re.search(r"Q\d+", uri)
            if m:
                qid = m.group(0)
                ents.setdefault(surface, []).append(Candidate(id=qid, label=surface))

        props: dict[str, list[Candidate]] = {}
        for item in data.get("relations_wikidata", []):
            uri = item.get("URI", "")
            surface = item.get("surface form", "")
            m = re.search(r"P\d+", uri)
            if m:
                pid = m.group(0)
                props.setdefault(surface, []).append(Candidate(id=pid, label=surface))

        return Linking(entities=ents, properties=props)
