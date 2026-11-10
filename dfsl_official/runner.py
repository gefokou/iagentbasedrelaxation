"""DFSL avec le code officiel des auteurs ; seul le serveur de modèle est remplacé.

Repris tel quel du dépôt des auteurs (external/dfsl, non modifié) :
  - le texte encodé pour la recherche (question + entités + relations, `create_input_to_embedd`),
  - l'encodeur all-mpnet-base-v2 et la similarité cosinus (`compute_similarity`, k plus proches),
  - le prompt, les exemples du plus proche au plus éloigné, et l'extraction <SPARQL>
    (`dynamic_few_shots_wikidata`, `create_prompt_dfsl`, `extraction_pipe`).
Remplacé : `Dfsl.call_api` (POST vers un serveur vLLM maison) par un appel chat à une API
compatible OpenAI (Groq...). La suite (exécution et F1) est faite par notre évaluateur, le même
pour tous les systèmes.

Conditions de l'article reproduites : entités et relations gold fournies dans le prompt et dans
l'encodage, k = 5, une seule requête (pas de beam search : Groq ne renvoie pas plusieurs
hypothèses ; DFSL-MQ n'est donc pas reproduit). Les entités gold des questions de test sont
celles de la requête de référence ; leurs libellés sont lus sur Wikidata.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from agent1.evaluation import check_store_size, remove_leaks
from agent1.filtre import add_prefixes, ids_in

EXT = Path(__file__).resolve().parent.parent / "external" / "dfsl"
ENT_URI = "http://www.wikidata.org/entity/"
REL_URI = "http://www.wikidata.org/prop/direct/"   # forme de la Figure 1 de l'article
NO_QUERY = "aucune requête SPARQL dans la réponse"


class StopForQuota(RuntimeError):
    """Panne du modèle (quota du jour atteint, réseau) : on s'arrête, la reprise continue."""


def _dfsl():
    """Importe les fonctions des auteurs depuis external/dfsl (sans modifier leur code)."""
    if str(EXT) not in sys.path:
        sys.path.insert(0, str(EXT))
    from modules.utilis.prompting_utils import create_prompt_dfsl, extraction_pipe
    from modules.utilis.utilities import compute_similarity, create_input_to_embedd
    from prompt.prompts import dynamic_few_shots_wikidata
    return (compute_similarity, create_input_to_embedd, create_prompt_dfsl, extraction_pipe,
            dynamic_few_shots_wikidata)


def _ent(i: str, labels: dict) -> list:
    lab = labels.get(i, "")
    return [ENT_URI + i, lab] if lab else ENT_URI + i      # sans libellé : texte simple (format des auteurs)


def _rel(i: str, labels: dict) -> list:
    lab = labels.get(i, "")
    return [REL_URI + i, lab] if lab else REL_URI + i


def with_prefixes(sparql: str) -> str:
    """Requêtes avec PREFIX wd/wdt déclarés, comme dans la Figure 1 de l'article."""
    return add_prefixes(sparql.strip()).strip()


def to_dfsl_format(store: list[dict], test: list[dict], labels: dict[str, str]):
    """Stock et test au format JSON des auteurs ({id: {question, query, entities, relations}})."""
    train = {}
    for n, ex in enumerate(store):
        lab = {**labels, **{e["id"]: e.get("label", "") for e in ex.get("entities", [])},
               **{r["id"]: r.get("label", "") for r in ex.get("relations", [])}}
        sp = ex.get("sparql") or ex.get("gold_sparql") or ""
        q, p = ids_in(sp)
        train[f"s{n}"] = {"question": ex["question"], "query": with_prefixes(sp),
                          "entities": [_ent(i, lab) for i in q],
                          "relations": [_rel(i, lab) for i in p]}
    tst = {}
    for t in test:
        q, p = ids_in(t["gold_sparql"])
        tst[t["id"]] = {"question": t["question"], "query": with_prefixes(t["gold_sparql"]),
                        "entities": [_ent(i, labels) for i in q],
                        "relations": [_rel(i, labels) for i in p]}
    return train, tst


def default_encoder():
    """Encodeur des auteurs : all-mpnet-base-v2, sorties en tenseurs."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-mpnet-base-v2")
    return lambda texts: model.encode(texts, convert_to_tensor=True, show_progress_bar=len(texts) > 500)


def _cached_embeddings(encode, texts: list[str], cache_dir: Path):
    """Embeddings mis en cache dans `cache_dir`, clé = contenu des textes (jamais périmé)."""
    import hashlib

    import torch
    key = hashlib.sha1("\n".join(texts).encode("utf-8")).hexdigest()[:16]
    path = Path(cache_dir) / f"emb_{key}.pt"
    if path.exists():
        return torch.load(path)
    emb = encode(texts)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb, path)
    return emb


def run_dfsl(store: list[dict], test: list[dict], out_dir: Path, llm, labels: dict[str, str],
             *, k: int = 5, encode=None, near: float = 0.8, min_store: int = 1000,
             max_questions: int = 0, log=print) -> dict:
    """Traite les questions de `test` ; reprend là où `dfsl_results.jsonl` s'est arrêté.

    `llm.chat(system, user)` : client à API compatible OpenAI (agent1.llm.LLMClient).
    Retourne un résumé ; lève StopForQuota si le modèle tombe en panne (la reprise continue).
    """
    check_store_size(store, min_store)
    kept, removed = remove_leaks(store, test, near=near)
    log(f"stock : {len(store)} exemples, {len(removed)} retirés (fuites), {len(kept)} gardés")
    train, tst = to_dfsl_format(kept, test, labels)
    compute_similarity, embed_text, create_prompt, extract, prompt = _dfsl()
    encode = encode or default_encoder()
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_tr = _cached_embeddings(encode, [embed_text(d, "all") for d in train.values()], out_dir)
    emb_te = encode([embed_text(d, "all") for d in tst.values()])
    sims = compute_similarity(emb_tr, emb_te, train, tst, k)
    (out_dir / "dfsl_test.json").write_text(json.dumps(tst, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "dfsl_similar.json").write_text(json.dumps(
        {i: [train[s]["question"] for s in v] for i, v in sims.items()}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    path = out_dir / "dfsl_results.jsonl"
    done = {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()} \
        if path.exists() else set()
    n_new = n_none = 0
    for t in test:
        if t["id"] in done:
            continue
        if max_questions and n_new >= max_questions:
            log(f"limite de {max_questions} questions atteinte : relancer pour continuer")
            break
        text = create_prompt(prompt, tst[t["id"]], "all", sims, train, tst, t["id"])["text"]
        try:
            raw = llm.chat("", text)               # le prompt entier en un message, comme les auteurs
        except Exception as e:
            raise StopForQuota(f"{type(e).__name__}: {e} ; {len(done) + n_new} questions faites, "
                               "relancer la même commande pour reprendre") from e
        found = extract(raw, prompt["open_tag"], prompt["close_tag"])
        rec = {"id": t["id"], "question": t["question"], "query": found[0] if found else None,
               "n_queries": len(found), "error": "" if found else NO_QUERY, "raw": raw}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_new += 1
        n_none += not found
        log(f"  {t['id']} : {'requête' if found else 'AUCUNE requête'}")
    return {"nouvelles": n_new, "deja_faites": len(done), "sans_requete": n_none,
            "fuites_retirees": len(removed), "stock_utilise": len(kept), "fichier": str(path)}
