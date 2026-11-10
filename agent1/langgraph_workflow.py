"""Pipeline Multi-Agents avec LangGraph.

Graphe d'état exécutant le flux complet :
1. Linking d'entités et propriétés (Falcon 2.0 / Wikidata)
2. Génération parallèle (T1 Direct LLM, T2 Templates, T3 DFSL)
3. Filtrage statique et vote sans exécution
4. Exécution SPARQL sur Wikidata (Agent 2)
5. GraphRAG et synthèse en langage naturel avec citations (Agent 3)
"""
from __future__ import annotations

from typing import Any, Dict, Optional, TypedDict
from langgraph.graph import StateGraph, END

from agent1.agent import Agent1
from agent1.types import Agent1Result, Linking
from agent2.executor import Execution, Executor
from agent3.answerer import Agent3


class KBQAState(TypedDict):
    question: str
    linking: Optional[Linking]
    agent1_result: Optional[Dict[str, Any]]
    execution_result: Optional[Dict[str, Any]]
    final_answer: Optional[Dict[str, Any]]


def create_kbqa_graph(agent1: Agent1, executor: Executor, agent3: Agent3):
    workflow = StateGraph(KBQAState)

    def node_agent1(state: KBQAState) -> Dict[str, Any]:
        res1: Agent1Result = agent1.run(state["question"])
        return {
            "linking": res1.linking,
            "agent1_result": res1.to_dict()
        }

    def node_agent2(state: KBQAState) -> Dict[str, Any]:
        res1_dict = state.get("agent1_result") or {}
        query = res1_dict.get("query")
        if query:
            ex = executor.run(query)
            return {"execution_result": ex.to_dict()}
        return {"execution_result": None}

    def node_agent3(state: KBQAState) -> Dict[str, Any]:
        res1_dict = state.get("agent1_result") or {}
        query = res1_dict.get("query")
        ex_dict = state.get("execution_result")
        link = state.get("linking") or Linking()

        ex_obj = None
        if ex_dict:
            ex_obj = Execution(
                query=ex_dict.get("query", ""),
                status=ex_dict.get("status", "none"),
                answers=ex_dict.get("answers", []),
                seconds=ex_dict.get("seconds", 0.0),
                raw_error=ex_dict.get("raw_error")
            )

        ans3 = agent3.run(state["question"], ex_obj, link, query)
        return {"final_answer": ans3.to_dict()}

    workflow.add_node("agent1_generation_vote", node_agent1)
    workflow.add_node("agent2_sparql_execution", node_agent2)
    workflow.add_node("agent3_graphrag_synthesis", node_agent3)

    workflow.set_entry_point("agent1_generation_vote")
    workflow.add_edge("agent1_generation_vote", "agent2_sparql_execution")
    workflow.add_edge("agent2_sparql_execution", "agent3_graphrag_synthesis")
    workflow.add_edge("agent3_graphrag_synthesis", END)

    return workflow.compile()
