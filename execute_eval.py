"""Agent 2 sur les résultats de l'Agent 1 : exécution réelle sur Wikidata, F1, intervalles.

    python evaluate.py --store data/lcquad2_store.jsonl --test data/test40.jsonl --out runs/v1
    python execute_eval.py --results runs/v1/results.jsonl --test data/test40.jsonl --out runs/v1

Chaque requête (référence, vote, T3, T4) est exécutée une fois sur query.wikidata.org.
Les verdicts définitifs sont mémorisés dans exec_cache.jsonl ; relancer rejoue seulement les
pannes du service.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent1.config import Config
from agent2.executor import Executor
from agent2.run import run_execution


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", help="results.jsonl de evaluate.py (sans lui : pas de VOTE)")
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", action="append", default=[], metavar="NOM=fichier.jsonl",
                    help="système externe (ex. T3_GOLD=runs/.../dfsl_results.jsonl), répétable")
    a = ap.parse_args()
    out = Path(a.out)
    test = [json.loads(l) for l in Path(a.test).read_text(encoding="utf-8").splitlines() if l.strip()]
    for i, t in enumerate(test):
        t.setdefault("id", f"q{i + 1:03d}")
    extras = {k: Path(v) for k, v in (x.split("=", 1) for x in a.extra)}
    metrics = run_execution(Path(a.results) if a.results else None, test, out,
                            Executor(Config(), cache_path=out / "exec_cache.jsonl"), extras)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 1 if any("à rejouer" in v for v in metrics["exclues"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
