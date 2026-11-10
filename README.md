# Baseline : question → SPARQL → réponse (T3 = DFSL, T4 = GRASP, vote)

Question en langage naturel → requête SPARQL sur Wikidata → exécution → réponse rédigée.
La pipeline n'utilise que **deux techniques publiées, avec le code officiel de leurs auteurs**,
et un **vote** qui les combine (l'apport du projet).

| Nom | Technique | Code |
|---|---|---|
| T3 | DFSL (D'Abramo, Zugarini, Torroni 2025) | `external/dfsl` (copie non modifiée de github.com/jacopodabramo/DFSL, MIT) |
| T4 | GRASP (Walter, Bast 2025) | github.com/ad-freiburg/grasp, lancé par `grasp run` |
| VOTE | sélection entre T3 et T4, **sans exécuter** | `agent1/vote.py` (projet) |

## Agent 1 : générer, filtrer, voter
1. **Liaison** des entités et propriétés de la question (wbsearchentities). Elle fournit à T3 les
   entités et relations que l'article suppose données (module EL/RL amont). GRASP lie lui-même.
2. **T3** : k = 5 exemples les plus proches (all-mpnet-base-v2 ; question + entités + relations),
   prompt des auteurs, une requête. **T4** : agent GRASP complet.
3. **Filtre** : syntaxe (rdflib) et existence des QID/PID. Rien n'est exécuté.
4. **Vote** : les requêtes sont ramenées à une forme canonique (triplets avec variables
   renommées). Même forme → accord ; sinon priorité fixe T3 > T4 ; un seul survivant → lui.

## Agent 2 : exécution et métriques
Une seule exécution de la requête retenue sur Wikidata. Statuts : ok / vide / syntaxe / timeout /
service (panne : jamais mise en cache). F1 par ensemble de réponses (D'Abramo : deux ensembles
vides = 1), intervalle de confiance bootstrap, différences appariées, `f1_gold_non_vide`.

## Agent 3 : réponse en langage naturel
Voir plus bas.

## Installation
    pip install -r requirements.txt
    pip install git+https://github.com/ad-freiburg/grasp.git@main      # T4, Python >= 3.12
    # index Wikidata de GRASP : voir son README (GRASP_INDEX_DIR, wikidata.tar.gz)
    set AGENT1_GRASP_DIR=C:\chemin\vers\grasp      # dossier du dépôt GRASP (configs/run.yaml)

Modèle : API compatible OpenAI, par variables d'environnement (la clé n'est jamais écrite) :
`GROQ_API_KEY`, `AGENT1_MODEL`, `AGENT1_REASONING` (ex. `low`). Pour GRASP, ces valeurs sont
transmises à ses variables `MODEL`, `MODEL_PROVIDER=openai/completions`, `MODEL_ENDPOINT`,
`MODEL_API_KEY`. Résultats de GRASP mis en cache par (question, modèle) dans
`runs/grasp_cache.jsonl` : le vote se rejoue ensuite hors ligne.

## Utilisation
    python build_store.py train.json data/lcquad2_store.jsonl        # stock : LC-QuAD 2.0 train
    python pipeline.py "Where was Albert Einstein born?" --store data/lcquad2_store.jsonl
    python pipeline.py --questions data/test40.jsonl --store data/lcquad2_store.jsonl --out runs/pipeline

## Étude comparative
    # 1. T3 et T4, puis vote (stock sans fuite : questions de test et quasi-doublons retirés)
    python evaluate.py --store data/lcquad2_store.jsonl --test data/test40.jsonl --out runs/v1
    # 2. exécution réelle, F1, intervalles : T3, T4, VOTE
    python execute_eval.py --results runs/v1/results.jsonl --test data/test40.jsonl --out runs/v1
    # 3. DFSL officiel seul, conditions de l'article (entités et relations gold fournies) : T3_GOLD
    python run_dfsl.py --store data/lcquad2_store.jsonl --test data/test40.jsonl --out runs/dfsl_gold
    python execute_eval.py --test data/test40.jsonl --out runs/dfsl_gold --extra T3_GOLD=runs/dfsl_gold/dfsl_results.jsonl

- `T3_GOLD` (conditions de l'article, comparable à leurs 81,93) se distingue de `T3` (pipeline
  réelle, entités issues de la liaison) : l'écart mesure le coût de la liaison.
- `evaluate.py` s'arrête sur une panne et reprend à la même commande ; les pannes ne sont pas
  comptées comme des échecs. `run_dfsl.py --max-questions N` borne une session (quota du jour).
- Mode strict (par défaut) : aucune dégradation silencieuse. `--lenient` restaure les replis.
- Jeu de test : `data/test40.jsonl` (LC-QuAD 2.0, jeu de test officiel, `make_test40.py`).

## Écarts avec l'article DFSL (à écrire dans le rapport)
- Une seule requête par question : pas de beam search (les API gratuites ne renvoient pas
  plusieurs hypothèses), donc pas de DFSL-MQ.
- Modèles non fine-tunés disponibles gratuitement, différents de Llama-3 70B / CodeLlama 70B /
  Mixtral 8x7B.
- Dans la pipeline réelle, entités et relations viennent de la liaison, non de la référence.
- Le serveur vLLM des auteurs est remplacé par une API chat compatible OpenAI ; le prompt, la
  recherche d'exemples et l'extraction sont les leurs.

## Tests
    python -m pytest tests

## Agent 3 : réponse en langage naturel (GraphRAG) et pipeline complet
`pipeline.py` enchaîne Agent 1 -> Agent 2 -> Agent 3 :
```
python pipeline.py "Où est né Albert Einstein ?" --store data\lcquad2_store.jsonl
python pipeline.py --questions data\test40.jsonl --store data\lcquad2_store.jsonl --out runs\pipeline
```
Agent 3 : (1) requête sans réponse (vide, syntaxe, timeout, aucune requête) -> message
explicite, **sans appeler le LLM** ; (2) sinon, notes à 1 saut pour les entités de la réponse
(8 au plus) et les entités principales de la question (3), une seule requête SPARQL ;
(3) le LLM répond uniquement d'après ces notes et le résultat, avec citations [N1] ;
« INSUFFISANT » s'il ne peut pas ; (4) contrôle de fidélité sans LLM : nombres et identifiants Q
de la réponse absents du contexte -> `fidelity.supported = false`. Une panne Wikidata/LLM
donne `status: "erreur"`, jamais un texte de remplacement.
Limite : le contrôle de fidélité ne voit pas les affirmations textuelles sans chiffre ni Q ;
un échantillon doit être relu à la main.

### Notes par motifs (DeBellis et Neches, ONTOLLM 2026)
Les notes de l'Agent 3 suivent la méthode de cet article : triplets -> phrases anglaises par
**motifs déterministes** (sans LLM : « The place of birth of X is Y. », « X is an instance of
Y. », « X is educated at Y. »), phrases **regroupées en une note par entité** (l'article montre
que les notes consolidées sont mieux retrouvées que les phrases isolées), classées par
similarité avec la question avec l'encodeur SBERT déjà utilisé par T3. Écarts avec l'article :
Wikidata n'a pas d'étiquettes d'intégration par prédicat, on utilise l'étiquette anglaise de la
propriété ; pas de base vectorielle ni de seuil minScore : le classement sert à choisir les 15
phrases les plus proches ; les entités viennent de la requête, pas d'une recherche vectorielle.
