"""Sélection après pré-exécution Wikidata :
1. Les requêtes candidates valides sont exécutées sur Wikidata.
2. Les requêtes renvoyant un statut 'ok' (non vide) sont privilégiées par rapport à 'vide'.
3. Le vote départage par forme commune et consensus, puis par priorité configurée.
"""
from __future__ import annotations

from collections import defaultdict
from .types import Checked, Decision


def vote(checked: list[Checked], priority: tuple[str, ...], executor=None) -> Decision:
    valid = [c for c in checked if c.ok and c.form is not None]
    if not valid:
        why = "; ".join(f"{c.technique}: {', '.join(c.reasons)}" for c in checked)
        return Decision(None, None, "aucun", why)

    rank = {t: i for i, t in enumerate(priority)}

    # Si un exécuteur est fourni, exécuter d'abord pour vérifier les résultats non vides (évite les MFS/vides)
    if executor is not None:
        exec_map = {}
        for c in valid:
            if c.query:
                try:
                    ex = executor.run(c.query)
                    exec_map[c.technique] = ex.status
                except Exception:
                    exec_map[c.technique] = "erreur"
            else:
                exec_map[c.technique] = "aucun"

        # Prioriser les candidats avec statut 'ok' (non vides)
        ok_candidates = [c for c in valid if exec_map.get(c.technique) == "ok"]
        if ok_candidates:
            valid = ok_candidates

    by_priority = sorted(valid, key=lambda c: rank.get(c.technique, len(rank)))
    if len(valid) == 1:
        c = valid[0]
        return Decision(c.query, c.technique, "unique",
                        "seul candidat ayant passé le filtre et l'exécution")

    groups: dict[tuple, list[Checked]] = defaultdict(list)
    for c in valid:
        groups[c.form].append(c)

    best = max(groups.values(),
               key=lambda g: (len(g), -min(rank.get(c.technique, len(rank)) for c in g)))
    if len(best) >= 2:
        pick = sorted(best, key=lambda c: rank.get(c.technique, len(rank)))[0]
        return Decision(pick.query, pick.technique, "accord",
                        "forme commune après exécution : " + ", ".join(c.technique for c in best))

    pick = by_priority[0]
    return Decision(pick.query, pick.technique, "priorite",
                    "désaccord entre " + ", ".join(c.technique for c in valid)
                    + f" ; priorité {' > '.join(priority)}")
