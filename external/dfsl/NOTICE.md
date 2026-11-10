# Code officiel de DFSL (copie non modifiée)

Source : https://github.com/jacopodabramo/DFSL (commit 3dd7f23, 2024-09-01), licence MIT.
D'Abramo, Zugarini, Torroni, « Dynamic Few-Shot Learning for Knowledge Graph Question
Answering » (arXiv 2407.01409).

Les fichiers `modules/`, `prompt/` et `main.py` sont ceux des auteurs, sans aucune modification.
Le projet n'utilise que : `compute_similarity`, `create_input_to_embedd`, `create_prompt_dfsl`,
`extraction_pipe` et le prompt `dynamic_few_shots_wikidata`. Le serveur vLLM personnalisé des
auteurs (`Dfsl.call_api`) est remplacé par un appel à une API compatible OpenAI : voir
`dfsl_official/runner.py`.
