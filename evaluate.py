"""Évalue l'Agent 1 sur un jeu de questions, sans fuite.

    python evaluate.py --store data/lcquad2_store.jsonl --test data/test40.jsonl --out runs/v1

Le fichier de test est un JSONL : {"id": "q01", "question": "...", "gold_sparql": "..."}
(`gold_sparql` est facultatif). Le dossier de sortie reçoit store_filtered.jsonl,
leaks_removed.json, results.jsonl et summary.json. Relancer la même commande reprend là où
l'on s'est arrêté et rejoue les questions en erreur d'infrastructure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent1.agent import Agent1
from agent1.config import PRESETS, Config
from agent1.evaluation import check_store_size, remove_leaks, run_evaluation


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="groq_llama")
    ap.add_argument("--model", help="identifiant du modèle (sinon AGENT1_MODEL ou préréglage)")
    ap.add_argument("--techniques", default="T1,T2,T3", help="T1 = Direct LLM, T2 = Templates, T3 = DFSL")
    ap.add_argument("--min-store", type=int, default=1)
    ap.add_argument("--falcon", action="store_true", help="utilise Falcon 2.0 pour le linking")
    ap.add_argument("--near", type=float, default=0.8, help="seuil de questions voisines")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    test = read_jsonl(a.test)
    for i, t in enumerate(test):
        t.setdefault("id", f"q{i + 1:03d}")
    examples = read_jsonl(a.store)
    check_store_size(examples, a.min_store)                       # garde-fou 1
    kept, removed = remove_leaks(examples, test, a.near)          # garde-fou 2
    (out / "leaks_removed.json").write_text(json.dumps(removed, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
    (out / "store_filtered.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in kept), encoding="utf-8")
    print(f"stock : {len(examples)} exemples, {len(removed)} retirés (fuites), {len(kept)} gardés")

    cfg = Config(techniques=tuple(a.techniques.split(",")), use_falcon=a.falcon).with_preset(a.preset, a.model)  # strict
    agent = Agent1(cfg, store=kept if "T3" in cfg.techniques else None,
                   cache_dir=str(out / "cache"))
    summary = run_evaluation(agent, test, out / "results.jsonl")  # garde-fou 3
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["exclues_infra_a_rejouer"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
