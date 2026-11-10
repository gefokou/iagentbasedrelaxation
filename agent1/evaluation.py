"""Évaluation de l'Agent 1 sans fuite et sans comptage erroné.

Trois garde-fous :
  1. le stock d'exemples doit être un vrai stock (taille minimale), pas l'amorce ;
  2. les questions de test (et leurs voisines) sont retirées du stock avant l'évaluation ;
  3. une réponse n'est comptée que si aucune technique n'a eu d'erreur d'infrastructure.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from .filtre import add_prefixes
from .normalise import canonical_form

MODEL_FAILURE = "aucune requête SPARQL dans la réponse"


# ------------------------------------------------------------------ 1. stock
def check_store_size(examples: list[dict], minimum: int) -> None:
    if len(examples) < minimum:
        raise ValueError(
            f"stock d'exemples trop petit ({len(examples)} < {minimum}) : l'amorce "
            "seed_examples.jsonl ne doit pas servir à évaluer. Construire le stock avec "
            "build_store.py à partir de LC-QuAD 2.0.")


# ------------------------------------------------------------------ 2. fuites
def _tokens(q: str) -> set[str]:
    return set(re.findall(r"\w+", q.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def _form(sparql: str):
    try:   # LC-QuAD 2.0 écrit wd:/wdt: sans déclarer les préfixes
        return canonical_form(add_prefixes(sparql.strip()))
    except Exception:
        return None


def remove_leaks(examples: list[dict], test_items: list[dict],
                 near: float = 0.8) -> tuple[list[dict], list[dict]]:
    """Retire du stock tout exemple qui révèle une question de test.

    Retiré si : même question (après normalisation) ; question voisine (Jaccard sur les
    mots >= `near`) ; requête de même forme canonique que la requête de référence du test.
    Retourne (stock conservé, [{'question','reason','test_id'}...]).
    """
    tq = [(t.get("id"), t["question"], _tokens(t["question"]),
           _form(t["gold_sparql"]) if t.get("gold_sparql") else None) for t in test_items]
    exact = {" ".join(sorted(re.findall(r"\w+", q.lower()))): tid for tid, q, _, _ in tq}
    gold_forms = {f: tid for tid, _, _, f in tq if f is not None}
    kept, removed = [], []
    for ex in examples:
        toks = _tokens(ex["question"])
        key = " ".join(sorted(re.findall(r"\w+", ex["question"].lower())))
        reason, tid = None, None
        if key in exact:
            reason, tid = "question identique", exact[key]
        else:
            for t_id, _, ttoks, _ in tq:
                if _jaccard(toks, ttoks) >= near:
                    reason, tid = f"question voisine (Jaccard >= {near})", t_id
                    break
        if reason is None and gold_forms:
            sp = ex.get("sparql") or ex.get("gold_sparql") or ""
            f = _form(sp) if sp else None
            if f is not None and f in gold_forms:
                reason, tid = "même requête que la référence du test", gold_forms[f]
        if reason:
            removed.append({"question": ex["question"], "reason": reason, "test_id": tid})
        else:
            kept.append(ex)
    return kept, removed


# ------------------------------------------------------------------ 3. comptage
def is_infra_error(err: str | None) -> bool:
    return bool(err) and not err.startswith(MODEL_FAILURE)


def classify(result) -> tuple[str, list[str]]:
    """'ok' | 'echec_modele' | 'infra' et la liste des erreurs d'infrastructure."""
    infra = [f"{g.technique}: {g.error}" for g in result.generated
             if is_infra_error(g.error)]
    infra += [f"{c.technique}: {r}" for c in result.checked
              for r in c.reasons if "non vérifiable" in r]
    if infra:
        return "infra", infra
    return ("ok" if result.query else "echec_modele"), []


def record_of(item: dict, result=None, exc: Exception | None = None) -> dict:
    base = {"id": item.get("id"), "question": item["question"]}
    if exc is not None:
        return {**base, "status": "infra", "infra_errors": [f"{type(exc).__name__}: {exc}"],
                "query": None, "technique": None, "rule": None}
    status, infra = classify(result)
    rec = {**base, "status": status, "infra_errors": infra, "query": result.query,
           "technique": result.technique, "rule": result.rule, "detail": result.detail,
           # requête de chaque technique/témoin : `query` seulement si elle a passé le filtre
           "candidates": {c.technique: {"query": c.query if c.ok else None, "ok": c.ok}
                          for c in result.checked},
           "errors": {g.technique: g.error for g in result.generated if g.error},
           "rejections": {c.technique: c.reasons for c in result.checked if c.reasons},
           "warnings": {c.technique: c.warnings for c in result.checked if c.warnings}}
    gold = item.get("gold_sparql")
    if gold and result.query and status == "ok":
        gf, qf = _form(gold), _form(result.query)
        rec["same_form_as_gold"] = bool(gf is not None and qf == gf)
    return rec


def summarize(records: list[dict]) -> dict:
    counted = [r for r in records if r["status"] in ("ok", "echec_modele")]
    ok = [r for r in counted if r["status"] == "ok"]
    gold = [r for r in ok if "same_form_as_gold" in r]
    return {
        "questions": len(records),
        "comptees": len(counted),
        "exclues_infra_a_rejouer": [r["id"] for r in records if r["status"] == "infra"],
        "requete_valide": len(ok),
        "echec_modele": len(counted) - len(ok),
        "taux_requete_valide": round(len(ok) / len(counted), 3) if counted else None,
        "regles": dict(Counter(r["rule"] for r in ok)),
        "techniques_retenues": dict(Counter(r["technique"] for r in ok)),
        "forme_identique_a_la_reference": (
            {"n": len(gold), "taux": round(sum(r["same_form_as_gold"] for r in gold) / len(gold), 3)}
            if gold else None),
        "note": "L'exécution et le F1 relèvent de l'Agent 2 ; 'forme identique' est un indicateur "
                "sur les triplets, pas un F1.",
    }


def load_records(path: Path) -> dict:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r
    return out


def save_records(path: Path, records: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records.values()),
                   encoding="utf-8")
    tmp.replace(path)


def run_evaluation(agent, test_items: list[dict], out_path: Path,
                   retry_infra: bool = True) -> dict:
    """Exécute l'agent sur chaque question ; reprend là où l'on s'est arrêté."""
    records = load_records(out_path)
    for item in test_items:
        old = records.get(item["id"])
        if old and (old["status"] != "infra" or not retry_infra):
            continue
        try:
            records[item["id"]] = record_of(item, result=agent.run(item["question"]))
        except Exception as e:   # panne LLM / Wikidata en mode strict : question non comptée
            records[item["id"]] = record_of(item, exc=e)
        save_records(out_path, records)
    return summarize([records[t["id"]] for t in test_items if t["id"] in records])
