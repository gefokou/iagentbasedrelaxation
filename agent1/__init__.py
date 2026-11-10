"""Agent 1 de la baseline : question en langage naturel -> requête SPARQL (Wikidata).

Deux techniques publiées, avec le code officiel de leurs auteurs : T3 (DFSL) et T4 (GRASP).
Sélection par vote, sans exécution de requête.
"""
from .agent import Agent1
from .config import Config, PRESETS
from .types import Agent1Result

__all__ = ["Agent1", "Config", "PRESETS", "Agent1Result"]
