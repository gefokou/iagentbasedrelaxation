"""DFSL officiel (code des auteurs) sur LC-QuAD 2.0 avec un modèle compatible OpenAI.

    python run_dfsl.py --store data/lcquad2_store.jsonl --test data/test40.jsonl --out runs/dfsl_gptoss
    (variables d'environnement du modèle comme pour evaluate.py ; --max-questions N pour borner)

Reprend où il s'est arrêté (quota du jour, coupure). Puis, pour mesurer le F1 avec le même
évaluateur que les autres systèmes :
    python execute_eval.py --test data/test40.jsonl --out runs/dfsl_gptoss \\
        --extra T3_GOLD=runs/dfsl_gptoss/dfsl_results.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent1.config import PRESETS, Config
from agent1.filtre import ids_in
from agent1.llm import LLMClient
from agent1.wikidata import WikidataClient
from dfsl_official import StopForQuota, run_dfsl


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def fetch_labels(test: list[dict], cfg: Config, cache: Path) -> dict[str, str]:
    """Libellés des entités et relations de référence des questions de test (lus sur Wikidata)."""
    ids = sorted({i for t in test for part in ids_in(t["gold_sparql"]) for i in part})
    have = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    todo = [i for i in ids if i not in have]
    if todo:
        info = WikidataClient(cfg).info(todo)     # lève une erreur si Wikidata ne répond pas
        have.update({i: info.get(i, {}).get("label", "") for i in todo})
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(have, ensure_ascii=False), encoding="utf-8")
    return have


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="groq_llama")
    ap.add_argument("--model")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-questions", type=int, default=0)
    ap.add_argument("--min-store", type=int, default=1000)
    ap.add_argument("--near", type=float, default=0.8)
    a = ap.parse_args()
    cfg = Config().with_preset(a.preset, a.model)
    test = read_jsonl(a.test)
    for i, t in enumerate(test):
        t.setdefault("id", f"q{i + 1:03d}")
    out = Path(a.out)
    labels = fetch_labels(test, cfg, out / "labels.json")
    try:
        summary = run_dfsl(read_jsonl(a.store), test, out, LLMClient(cfg), labels, k=a.k,
                           near=a.near, min_store=a.min_store, max_questions=a.max_questions)
    except StopForQuota as e:
        print(f"ARRÊT : {e}")
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
