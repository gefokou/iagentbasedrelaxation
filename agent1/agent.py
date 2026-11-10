"""Agent 1 : générer (T1 = Direct LLM, T2 = Templates, T3 = DFSL), filtrer, exécuter, voter, transmettre."""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor

from . import filtre
from .config import Config
from .falcon_linker import FalconLinker
from .linking import Linker
from .llm import LLMClient
from .techniques import T1DirectLLM, T2Template, T3Dfsl, T4Grasp
from .techniques.t4_grasp import Cache
from .types import Agent1Result, Generated, Linking
from .vote import vote
from .wikidata import WikidataClient
from agent2.executor import Executor


class Agent1:
    def __init__(self, cfg: Config | None = None, *, llm=None, wd=None, store=None,
                 store_path: str | None = None, encode=None, cache_dir: str | None = None,
                 grasp_runner=None, executor=None):
        """`store` : liste d'exemples (ou `store_path`, JSONL) pour T3 ; `encode` : encodeur
        des auteurs de DFSL (par défaut all-mpnet-base-v2) ; `grasp_runner` : remplace
        l'appel à la commande `grasp` (tests) ; `executor` : exécuteur pour valider les requêtes avant vote."""
        self.cfg = cfg or Config()
        self.llm = llm or LLMClient(self.cfg)
        self.wd = wd or WikidataClient(self.cfg)
        self.linker = Linker(self.cfg, self.llm, self.wd)
        self.falcon = FalconLinker()
        self.executor = executor or Executor(self.cfg)
        self.techniques = []
        for name in self.cfg.techniques:
            if name == "T1":
                self.techniques.append(T1DirectLLM(self.cfg, self.llm))
            elif name == "T2":
                self.techniques.append(T2Template(self.cfg, self.llm))
            elif name == "T3":
                if store is None:
                    if not store_path:
                        raise ValueError("T3 a besoin d'un stock d'exemples (store_path)")
                    with open(store_path, encoding="utf-8") as f:
                        store = [json.loads(l) for l in f if l.strip()]
                self.techniques.append(T3Dfsl(self.cfg, self.llm, store, encode, cache_dir))
            elif name == "T4":
                self.techniques.append(T4Grasp(self.cfg, runner=grasp_runner,
                                               cache=Cache(None) if grasp_runner else None))
            else:
                raise ValueError(f"technique inconnue : {name} (T1, T2, T3 ou T4)")

    def run(self, question: str) -> Agent1Result:
        t0 = time.time()
        # Linking Falcon 2.0 avec repli sur le linker Wikidata
        if any(t.needs_link for t in self.techniques):
            if getattr(self.cfg, "use_falcon", False):
                link = self.falcon.link(question)
                if not link.entities and not link.properties:
                    link = self.linker.link(question)
            else:
                link = self.linker.link(question)
        else:
            link = Linking()

        with ThreadPoolExecutor(max_workers=len(self.techniques)) as ex:   # 1. générer
            futures = [ex.submit(t.generate, question, link) for t in self.techniques]
            generated: list[Generated] = [f.result() for f in futures]

        # 2-3. filtrer syntaxe et identifiants
        checked = [filtre.check(g, self.wd, self.cfg.check_ids_exist, self.cfg.strict)
                   for g in generated]

        # 4. exécuter avant le vote pour éliminer les requêtes vides (MFS) et voter
        d = vote(checked, self.cfg.priority, executor=self.executor)

        return Agent1Result(                                      # 5. transmettre
            question=question, query=d.query, technique=d.technique,
            rule=d.rule, detail=d.detail, linking=link, generated=generated,
            checked=checked, seconds=time.time() - t0)
