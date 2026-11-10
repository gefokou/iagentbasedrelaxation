"""Exécute les requêtes de l'Agent 1 et de la référence, puis calcule les métriques.

Entrée : results.jsonl de evaluate.py (une ligne par question, avec `query` retenue par le
vote et `candidates` par technique) et le jeu de test (gold_sparql).
Sorties : exec_results.jsonl (une ligne par question) et metrics.json.
Une question est EXCLUE (listée « à rejouer » ou « écartée ») si :
  - l'Agent 1 a eu une erreur d'infrastructure sur elle ;
  - la référence ne s'exécute pas (invalide, timeout) ou le service Wikidata a failli ;
  - une requête d'un système a rencontré une panne du service (statut `service`).
Relancer la commande rejoue les pannes : les verdicts définitifs sont en cache.
"""
from __future__ import annotations

import json
from pathlib import Path

from .executor import Executor
from .metrics import compute_metrics, f1

SYSTEMS = ("VOTE", "T3", "T4", "T3_GOLD")   # T3_GOLD : DFSL officiel avec entités gold (run_dfsl.py)


def _queries(rec: dict) -> dict[str, str | None]:
    q = {"VOTE": rec.get("query")} if rec.get("vote", True) else {}
    for tech, c in (rec.get("candidates") or {}).items():
        q[tech] = c.get("query") if c.get("ok") else None
    return {k: v for k, v in q.items() if k in SYSTEMS}


def evaluate_question(rec: dict, item: dict, ex: Executor) -> dict:
    row = {"id": item["id"], "question": item["question"],
           "template_id": item.get("template_id"), "excluded": None, "systems": {},
           "gold_empty": False}
    if rec["status"] == "infra":
        row["excluded"] = "erreur d'infrastructure de l'Agent 1 : à rejouer (evaluate.py)"
        return row
    gold_q = item.get("gold_sparql")
    if not gold_q:
        row["excluded"] = "pas de requête de référence"
        return row
    g = ex.run(gold_q)
    row["gold_status"] = g.status
    if g.status == "service":
        row["excluded"] = f"service Wikidata indisponible pour la référence : à rejouer ({g.error})"
        return row
    if not g.executable:
        row["excluded"] = f"référence inexploitable ({g.status} : {g.error})"
        return row
    gold = g.answer_set()
    row["gold_empty"] = not gold
    row["gold_answers"] = len(gold)
    for name, query in _queries(rec).items():
        if not query:
            row["systems"][name] = {"query": None, "status": "sans_requete", "f1": 0.0,
                                    "n_answers": 0, "error": ""}
            continue
        e = ex.run(query)
        if e.status == "service":
            row["excluded"] = f"service Wikidata indisponible ({name}) : à rejouer ({e.error})"
            row["systems"] = {}
            return row
        row["systems"][name] = {"query": query, "status": e.status,
                                "f1": f1(gold, e.answer_set()), "n_answers": len(e.answers),
                                "error": e.error}
    return row


def run_execution(results_path: Path | None, test_items: list[dict], out_dir: Path,
                  executor: Executor, extras: dict[str, Path] | None = None) -> dict:
    """`extras` : {nom du système: JSONL de {id, query}} (ex. DFSL officiel, GRASP) fusionnés
    dans les candidats. Sans `results_path`, seuls les systèmes extras sont mesurés (pas de VOTE)."""
    records = {}
    if results_path is not None:
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                records[r["id"]] = r
    else:
        records = {t["id"]: {"id": t["id"], "status": "ok", "query": None, "vote": False,
                             "candidates": {}} for t in test_items}
    for name, path in (extras or {}).items():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if e["id"] in records:
                records[e["id"]].setdefault("candidates", {})[name] = {
                    "query": e.get("query"), "ok": bool(e.get("query"))}
    rows, missing = [], []
    for item in test_items:
        if item["id"] not in records:
            missing.append(item["id"])
            continue
        rec = records[item["id"]]
        if not rec.get("vote", True) and not rec["candidates"]:
            missing.append(item["id"])     # mode extras seuls : question pas encore traitée
            continue
        rows.append(evaluate_question(rec, item, executor))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "exec_results.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    counted = [r for r in rows if not r["excluded"]]
    metrics = compute_metrics(counted)
    metrics["questions_total"] = len(test_items)
    metrics["sans_resultat_agent1"] = missing
    metrics["exclues"] = {r["id"]: r["excluded"] for r in rows if r["excluded"]}
    metrics["note"] = ("F1 = F1 des ensembles de réponses (D'Abramo et Zugarini 2025) ; deux "
                       "ensembles vides = 1. 'f1_gold_non_vide' écarte ce cas. Les questions "
                       "exclues ne sont pas comptées ; voir 'exclues'.")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    return metrics
