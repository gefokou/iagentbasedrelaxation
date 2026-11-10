"""python -m agent1 "Où est né Albert Einstein ?" --store data/seed_examples.jsonl"""
from __future__ import annotations

import argparse
import json
import sys

from .agent import Agent1
from .config import PRESETS, Config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="agent1")
    ap.add_argument("question", nargs="?", help="question ; sinon lue sur l'entrée standard")
    ap.add_argument("--store", required=True, help="stock d'exemples JSONL (voir build_store.py)")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="groq_llama")
    ap.add_argument("--model", help="identifiant du modèle (sinon variable AGENT1_MODEL ou préréglage)")
    ap.add_argument("--techniques", default="T3,T4", help="T3 = DFSL, T4 = GRASP (codes officiels)")
    ap.add_argument("--cache", default=".cache")
    ap.add_argument("--lenient", action="store_true",
                    help="autorise les replis silencieux (déconseillé)")
    ap.add_argument("--full", action="store_true", help="sortie détaillée")
    a = ap.parse_args(argv)
    q = a.question or sys.stdin.read().strip()
    cfg = Config(techniques=tuple(a.techniques.split(",")),
                 strict=not a.lenient).with_preset(a.preset, a.model)
    res = Agent1(cfg, store_path=a.store, cache_dir=a.cache).run(q)
    out = res.to_dict() if a.full else {
        "question": res.question, "query": res.query, "technique": res.technique,
        "rule": res.rule, "detail": res.detail}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if res.query else 1


if __name__ == "__main__":
    raise SystemExit(main())
