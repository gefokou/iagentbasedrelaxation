"""T4 : GRASP (Walter et Bast, 2025), exécuté avec le code officiel de ses auteurs.

https://github.com/ad-freiburg/grasp  (arXiv 2507.08107)

Ce module n'implémente RIEN de GRASP : il lance sa commande `grasp run` sur la question et
lit la requête SPARQL de sa sortie JSON. Seul le modèle de langage est remplacé, par les
variables d'environnement que prévoit GRASP lui-même (MODEL, MODEL_PROVIDER, MODEL_ENDPOINT,
MODEL_API_KEY). La clé reste dans l'environnement : elle n'est jamais écrite ni journalisée.

Contrairement à T3, GRASP est un agent : il appelle des outils et exécute des requêtes
pendant sa génération : c'est la technique publiée, utilisée telle quelle.

Les résultats sont mis en cache (question + modèle) : une question déjà traitée ne coûte
plus aucun appel au modèle, ce qui permet de rejouer le vote hors ligne.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from ..config import Config
from ..types import Generated, Linking

NO_QUERY = "aucune requête SPARQL dans la réponse"   # échec du modèle, pas de l'infrastructure


class GraspError(RuntimeError):
    """Échec d'infrastructure : GRASP absent, délai dépassé, sortie illisible."""


def grasp_env(cfg: Config, base: dict | None = None) -> dict:
    """Variables d'environnement lues par configs/run.yaml de GRASP."""
    env = dict(os.environ if base is None else base)
    key = os.getenv(cfg.llm_api_key_env)
    if not key:
        raise GraspError(f"variable {cfg.llm_api_key_env} absente de l'environnement")
    env.update({
        "MODEL": cfg.llm_model,
        "MODEL_PROVIDER": "openai/completions",   # API chat compatible OpenAI (Groq, etc.)
        "MODEL_ENDPOINT": cfg.llm_base_url,
        "MODEL_API_KEY": key,
        "KG": "wikidata",
    })
    if cfg.reasoning_effort:
        env["REASONING_EFFORT"] = cfg.reasoning_effort
    return env


def sparql_of(output: dict) -> str | None:
    """Requête SPARQL de la sortie JSON de `grasp run` (champ output.sparql)."""
    inner = output.get("output", output)
    if isinstance(inner, dict):
        sp = inner.get("sparql")
        return sp.strip() if isinstance(sp, str) and sp.strip() else None
    return None


class SubprocessRunner:
    """Lance `grasp run <config> --input <question>` et renvoie son JSON."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __call__(self, question: str) -> dict:
        c = self.cfg
        cmd = [*c.grasp_cmd.split(), "run", c.grasp_config, "--input", question]
        try:
            p = subprocess.run(cmd, cwd=c.grasp_dir or None, env=grasp_env(c),
                               capture_output=True, text=True, timeout=c.grasp_timeout_s)
        except FileNotFoundError as e:
            raise GraspError(f"commande GRASP introuvable ({c.grasp_cmd}) : {e}") from e
        except subprocess.TimeoutExpired as e:
            raise GraspError(f"GRASP a dépassé {c.grasp_timeout_s:.0f} s") from e
        if p.returncode != 0:
            raise GraspError(f"GRASP code {p.returncode} : {p.stderr.strip()[-300:]}")
        for line in reversed(p.stdout.strip().splitlines()):   # la sortie JSON est la dernière ligne
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        raise GraspError("sortie de GRASP sans JSON")


class Cache:
    """JSONL : une ligne {question, model, output} par question traitée."""

    def __init__(self, path: str | None):
        self.path = Path(path) if path else None
        self.data: dict[tuple[str, str], dict] = {}
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.data[(r["question"], r["model"])] = r["output"]

    def get(self, question: str, model: str) -> dict | None:
        return self.data.get((question, model))

    def put(self, question: str, model: str, output: dict) -> None:
        self.data[(question, model)] = output
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"question": question, "model": model, "output": output},
                                   ensure_ascii=False) + "\n")


class T4Grasp:
    name = "T4"
    needs_link = False      # GRASP fait sa propre liaison avec ses outils

    def __init__(self, cfg: Config, runner=None, cache: Cache | None = None):
        self.cfg = cfg
        self.runner = runner or SubprocessRunner(cfg)
        self.cache = cache if cache is not None else Cache(cfg.grasp_cache)

    def generate(self, question: str, link: Linking) -> Generated:   # `link` : non utilisé
        t0 = time.time()
        g = Generated(technique=self.name)
        try:
            out = self.cache.get(question, self.cfg.llm_model)
            if out is None:
                out = self.runner(question)
                self.cache.put(question, self.cfg.llm_model, out)
            g.raw = json.dumps(out, ensure_ascii=False)[:4000]
            g.query = sparql_of(out)
            if g.query is None:
                g.error = NO_QUERY
        except Exception as e:
            g.error = f"{type(e).__name__}: {e}"
        g.seconds = time.time() - t0
        return g
