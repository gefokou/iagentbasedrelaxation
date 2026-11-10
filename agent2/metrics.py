"""Métriques de l'Agent 2 : F1 sur les ensembles de réponses, intervalles bootstrap.

F1 comme D'Abramo et Zugarini (2025) : F1 entre l'ensemble de réponses de la requête de
référence et celui de la requête générée ; deux ensembles vides valent 1, un seul vide vaut 0.
Pas de requête transmise : F1 = 0 (la question reste comptée).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def f1(gold: set[str], pred: set[str] | None) -> float:
    if pred is None:
        return 0.0
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    inter = len(gold & pred)
    if inter == 0:
        return 0.0
    p, r = inter / len(pred), inter / len(gold)
    return 2 * p * r / (p + r)


def bootstrap_ci(values: list[float], iters: int = 10000, seed: int = 0) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    v = np.asarray(values, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(v), size=(iters, len(v)))
    means = v[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_diff(a: list[float], b: list[float], iters: int = 10000, seed: int = 0) -> dict:
    """Différence moyenne a − b sur les mêmes questions, avec IC 95 % bootstrap apparié."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    lo, hi = bootstrap_ci(d.tolist(), iters, seed)
    return {"n": len(d), "diff": round(float(d.mean()), 4) if len(d) else None,
            "ic95": [round(lo, 4), round(hi, 4)],
            "gagne": int((d > 0).sum()), "perd": int((d < 0).sum()), "egal": int((d == 0).sum())}


def _r(x: float) -> float:
    return round(float(x), 4)


def system_summary(rows: list[dict], name: str) -> dict:
    sel = [r for r in rows if name in r["systems"]]
    cells = [r["systems"][name] for r in sel]
    n = len(cells)
    if not n:
        return {"n": 0}
    f = [c["f1"] for c in cells]
    lo, hi = bootstrap_ci(f)
    nonempty = [r["systems"][name]["f1"] for r in sel if not r["gold_empty"]]
    count = lambda st: sum(c["status"] == st for c in cells)   # noqa: E731
    return {
        "n": n,
        "f1": _r(np.mean(f)), "f1_ic95": [_r(lo), _r(hi)],
        "f1_gold_non_vide": {"n": len(nonempty), "f1": _r(np.mean(nonempty)) if nonempty else None},
        "reponse_exacte": _r(np.mean([x == 1.0 for x in f])),
        "executable": _r(sum(c["status"] in ("ok", "vide") for c in cells) / n),
        "non_vide": _r(count("ok") / n),
        "vide": _r(count("vide") / n),
        "syntaxe": count("syntaxe"), "timeout": count("timeout"),
        "sans_requete": count("sans_requete"),
    }


COMPARISONS = (("VOTE", "T3"), ("VOTE", "T4"), ("T4", "T3"), ("T3_GOLD", "T3"), ("VOTE", "T3_GOLD"))


def compute_metrics(rows: list[dict]) -> dict:
    """`rows` : questions comptées (aucune exclue)."""
    names = [n for n in ("VOTE", "T3", "T4", "T3_GOLD") if any(n in r["systems"] for r in rows)]
    out = {"questions_comptees": len(rows), "gold_vide": sum(r["gold_empty"] for r in rows),
           "systemes": {n: system_summary(rows, n) for n in names}, "differences": {}}
    for a, b in COMPARISONS:
        both = [r for r in rows if a in r["systems"] and b in r["systems"]]
        if both:
            out["differences"][f"{a} - {b}"] = paired_diff(
                [r["systems"][a]["f1"] for r in both], [r["systems"][b]["f1"] for r in both])
    # limites hautes d'un choix parfait entre deux techniques (connaît la référence : non réalisable)
    for a, b in (("T3", "T4"),):
        both = [r for r in rows if a in r["systems"] and b in r["systems"]]
        if both:
            out[f"limite_haute_meilleur_de_{a}_{b}"] = _r(np.mean(
                [max(r["systems"][a]["f1"], r["systems"][b]["f1"]) for r in both]))
    by_t: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if "VOTE" in r["systems"]:
            by_t[str(r.get("template_id"))].append(r["systems"]["VOTE"]["f1"])
    out["par_type_VOTE"] = {t: {"n": len(v), "f1": _r(np.mean(v))} for t, v in sorted(by_t.items())}
    return out
