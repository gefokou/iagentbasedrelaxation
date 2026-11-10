"""Tire 40 questions du jeu de TEST officiel de LC-QuAD 2.0 (6046 questions), de façon
reproductible et variée.

    python make_test40.py test.json data/test40.jsonl [--train train.json] [--n 40] [--seed 42]

Méthode : on garde les questions propres (texte présent, sans accolades de gabarit, requête
de référence analysable) ; on tire au hasard (graine fixe) en tournant sur les 13 types de
gabarits (`template_id`) pour que chaque type de question soit représenté. Avec --train, on
écarte aussi les questions dont le texte exact figure dans le jeu d'entraînement.
Chaque ligne : {"id","question","gold_sparql","template_id","uid"}.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

from agent1.filtre import add_prefixes
from agent1.normalise import canonical_form


def norm(q: str) -> str:
    return " ".join(re.findall(r"\w+", q.lower()))


def clean(item: dict) -> bool:
    q = item.get("question")
    sp = item.get("sparql_wikidata")
    if not isinstance(q, str) or not sp or len(q.strip()) < 15:
        return False
    if "{" in q or "}" in q or q.strip().lower() == "n/a":
        return False
    try:
        canonical_form(add_prefixes(sp.strip()))
    except Exception:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("test_json")
    ap.add_argument("out_jsonl")
    ap.add_argument("--train")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude", help="JSONL déjà utilisé (ex. test40) : ses questions sont écartées")
    ap.add_argument("--prefix", default="q", help="préfixe des identifiants (ex. d pour dev)")
    a = ap.parse_args()

    test = json.loads(Path(a.test_json).read_text(encoding="utf-8"))
    in_train: set[str] = set()
    if a.train:
        in_train = {norm(x["question"]) for x in json.loads(Path(a.train).read_text(encoding="utf-8"))
                    if isinstance(x.get("question"), str)}
    used_uid: set = set()
    used_q: set[str] = set()
    if a.exclude:
        for line in Path(a.exclude).read_text(encoding="utf-8").splitlines():
            if line.strip():
                o = json.loads(line)
                used_uid.add(o.get("uid"))
                used_q.add(norm(o["question"]))
    pool = [x for x in test if clean(x) and norm(x["question"]) not in in_train
            and x.get("uid") not in used_uid and norm(x["question"]) not in used_q]
    print(f"{len(pool)} questions utilisables sur {len(test)}")

    rng = random.Random(a.seed)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for x in pool:
        by_type[str(x.get("template_id"))].append(x)
    for v in by_type.values():
        rng.shuffle(v)
    order = sorted(by_type, key=lambda k: -len(by_type[k]))   # types fréquents d'abord
    picked: list[dict] = []
    for round_ in zip_longest(*(by_type[k] for k in order)):
        for x in round_:
            if x is not None and len(picked) < a.n:
                picked.append(x)
        if len(picked) >= a.n:
            break
    rng.shuffle(picked)

    out = [{"id": f"{a.prefix}{i + 1:02d}", "question": re.sub(r"\s+", " ", x["question"]).strip(),
            "gold_sparql": x["sparql_wikidata"].strip(), "template_id": str(x.get("template_id")),
            "uid": x.get("uid")} for i, x in enumerate(picked)]
    Path(a.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_jsonl).write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out),
                                 encoding="utf-8")
    types = defaultdict(int)
    for o in out:
        types[o["template_id"]] += 1
    print(f"{len(out)} questions écrites dans {a.out_jsonl}; par type : {dict(sorted(types.items()))}")


if __name__ == "__main__":
    main()
