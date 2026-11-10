"""Construit le stock d'exemples de T3 à partir de LC-QuAD 2.0 (train.json).

    python build_store.py train.json data/lcquad2_store.jsonl [--max 25000]

Les libellés des entités et propriétés sont lus sur Wikidata (par lots de 50, mis en
cache) : il faut donc un accès réseau à wikidata.org. Pour éviter toute fuite, retirer
du stock les questions de test (ou garder exclude_exact_question = True).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent1.config import Config
from agent1.store import load_lcquad2
from agent1.wikidata import WikidataClient


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("train_json")
    ap.add_argument("out_jsonl")
    ap.add_argument("--max", type=int, default=0)
    a = ap.parse_args()
    items = load_lcquad2(a.train_json)
    if a.max:
        items = items[: a.max]
    wd = WikidataClient(Config())
    ids = sorted({i for it in items for i in it["ids_e"] + it["ids_r"]})
    print(f"{len(items)} exemples, {len(ids)} identifiants à libeller")
    # cache disque : en cas d'arrêt (429, coupure), la relance reprend où elle s'était arrêtée
    cache_path = Path(a.out_jsonl).with_suffix(".labels.json")
    if cache_path.exists():
        wd._labels.update(json.loads(cache_path.read_text(encoding="utf-8")))
    todo = [i for i in ids if i not in wd._labels]
    print(f"{len(ids) - len(todo)} libellés déjà en cache, {len(todo)} à lire")
    for k in range(0, len(todo), 50):
        wd.info(todo[k:k + 50])
        cache_path.write_text(json.dumps(wd._labels, ensure_ascii=False), encoding="utf-8")
        print(f"  {min(k + 50, len(todo))}/{len(todo)}", end="\r", flush=True)
    print()
    info = wd.info(ids)
    absent = sum(1 for i in ids if not info.get(i, {}).get("exists"))
    if ids and absent / len(ids) > 0.10:   # signe d'une panne d'API, pas de vrais identifiants supprimés
        cache_path.unlink(missing_ok=True)
        raise SystemExit(f"{absent}/{len(ids)} identifiants « inexistants » : lecture Wikidata "
                         "défaillante. Cache supprimé ; relancer. Stock non écrit.")
    with open(a.out_jsonl, "w", encoding="utf-8") as f:
        for it in items:
            rec = {
                "question": it["question"], "sparql": it["sparql"],
                "entities": [{"id": i, "label": info.get(i, {}).get("label", "")}
                             for i in it["ids_e"]],
                "relations": [{"id": i, "label": info.get(i, {}).get("label", "")}
                              for i in it["ids_r"]],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
