"""Pipeline de base : question -> Agent 1 (SPARQL) -> Agent 2 (exécution) -> Agent 3 (réponse).

    python pipeline.py "Où est né Albert Einstein ?" --store data/lcquad2_store.jsonl
    python pipeline.py --questions data/test40.jsonl --store ... --out runs/pipeline

Une ligne JSON par question dans <out>/pipeline.jsonl (requête, statut d'exécution, réponse,
notes, fidélité). Aucune réparation : une requête qui échoue donne « aucune réponse ».
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent1.agent import Agent1
from agent1.config import PRESETS, Config
from agent1.llm import LLMClient
from agent1.store import SentenceTransformerEncoder
from agent1.wikidata import WikidataClient
from agent2.executor import Executor
from agent3.answerer import Agent3
from agent3.notes import NoteBuilder


def run_one(question: str, a1: Agent1, a2: Executor, a3: Agent3) -> dict:
    r1 = a1.run(question)
    ex = a2.run(r1.query) if r1.query else None
    r3 = a3.run(question, ex, r1.linking, r1.query)
    return {"question": question, "query": r1.query, "technique": r1.technique, "rule": r1.rule,
            "execution": ({k: v for k, v in ex.to_dict().items() if k != "answers"} if ex else None),
            "answer": r3.to_dict()}


def run_batch(items: list[dict], a1, a2, a3, path: Path) -> int:
    """Traite chaque question isolément ; reprend là où l'on s'est arrêté."""
    for i, it in enumerate(items):
        it.setdefault("id", f"q{i + 1:03d}")
    path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if path.exists():       # reprise : on garde ce qui a abouti, on rejoue les pannes
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if not r.get("pipeline_error"):
                    done[r["id"]] = r
    failed = 0
    for it in items:
        if it["id"] in done:
            continue
        try:
            rec = {"id": it["id"], **run_one(it["question"], a1, a2, a3)}
            print(it["id"], "|", rec["answer"]["status"], "|", rec["answer"]["answer"][:100])
        except Exception as e:   # panne réseau/LLM : on note, on continue avec la question suivante
            failed += 1
            rec = {"id": it["id"], "question": it["question"],
                   "pipeline_error": f"{type(e).__name__}: {e}"}
            print(it["id"], "| ERREUR |", rec["pipeline_error"][:100])
        done[it["id"]] = rec
        path.write_text("".join(json.dumps(done[i["id"]], ensure_ascii=False) + "\n"
                                for i in items if i["id"] in done), encoding="utf-8")
    if failed:
        print(f"{failed} question(s) en erreur : relancer la même commande les rejoue.")
    return 1 if failed else 0


def build(cfg: Config, store_path: str):
    wd = WikidataClient(cfg)
    llm = LLMClient(cfg)
    enc = SentenceTransformerEncoder(cfg.embed_model)     # un seul chargement, partagé
    a1 = Agent1(cfg, llm=llm, wd=wd, store_path=store_path, cache_dir=".cache")   # embeddings du stock mis en cache
    return a1, Executor(cfg), Agent3(llm, NoteBuilder(wd, encoder=enc))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--questions", help="JSONL {id, question}")
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", default="runs/pipeline")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="groq_llama")
    ap.add_argument("--model")
    a = ap.parse_args()
    cfg = Config().with_preset(a.preset, a.model)
    a1, a2, a3 = build(cfg, a.store)
    if a.question:
        print(json.dumps(run_one(a.question, a1, a2, a3), ensure_ascii=False, indent=2))
        return 0
    items = [json.loads(l) for l in Path(a.questions).read_text(encoding="utf-8").splitlines() if l.strip()]
    return run_batch(items, a1, a2, a3, Path(a.out) / "pipeline.jsonl")


if __name__ == "__main__":
    raise SystemExit(main())
