"""Configuration de l'Agent 1."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

PRESETS: dict[str, dict] = {
    "groq_gpt_oss": dict(
        llm_model="openai/gpt-oss-120b",
        llm_base_url="https://api.groq.com/openai/v1",
        llm_api_key_env="GROQ_API_KEY",
    ),
    "groq_qwen": dict(
        llm_model="qwen/qwen3.8-27b",
        llm_base_url="https://api.groq.com/openai/v1",
        llm_api_key_env="GROQ_API_KEY",
    ),
    "groq_llama": dict(
        llm_model="llama-3.3-70b-versatile",
        llm_base_url="https://api.groq.com/openai/v1",
        llm_api_key_env="GROQ_API_KEY",
    ),
    "openai_gpt41": dict(
        llm_model="gpt-4.1",
        llm_base_url="https://api.openai.com/v1",
        llm_api_key_env="OPENAI_API_KEY",
    ),
}


@dataclass
class Config:
    llm_model: str = os.getenv("AGENT1_MODEL", "llama-3.3-70b-versatile")
    llm_base_url: str = os.getenv("AGENT1_BASE_URL", "https://api.groq.com/openai/v1")
    llm_api_key_env: str = os.getenv("AGENT1_API_KEY_ENV", "GROQ_API_KEY")
    temperature: float = 0.0
    max_tokens: int = 2048
    reasoning_effort: str | None = os.getenv("AGENT1_REASONING") or None
    llm_retries: int = 6
    timeout_s: float = 60.0

    embed_model: str = os.getenv("AGENT1_EMBED", "paraphrase-multilingual-mpnet-base-v2")
    k_examples: int = 5
    exclude_exact_question: bool = True

    entity_candidates: int = 3
    property_candidates: int = 3
    use_falcon: bool = False

    grasp_cmd: str = os.getenv("AGENT1_GRASP_CMD", "grasp")
    grasp_dir: str = os.getenv("AGENT1_GRASP_DIR", "")
    grasp_config: str = os.getenv("AGENT1_GRASP_CONFIG", "configs/run.yaml")
    grasp_timeout_s: float = 900.0
    grasp_cache: str = os.getenv("AGENT1_GRASP_CACHE", "runs/grasp_cache.jsonl")

    wd_api: str = "https://www.wikidata.org/w/api.php"
    wd_sparql: str = "https://query.wikidata.org/sparql"
    wd_timeout_s: float = 30.0
    wd_min_interval_s: float = 0.5
    wd_retries: int = 6
    exec_timeout_s: float = 70.0
    exec_retries: int = 5
    user_agent: str = os.getenv(
        "AGENT1_UA", "baseline-agent1/0.1 (projet de Master, Universite de Dschang)"
    )

    techniques: tuple[str, ...] = ("T1", "T2", "T3")
    priority: tuple[str, ...] = ("T3", "T1", "T2")
    check_ids_exist: bool = True
    strict: bool = True

    def with_preset(self, name: str, model: str | None = None) -> "Config":
        cfg = replace(self, **PRESETS[name])
        chosen = model or os.getenv("AGENT1_MODEL")
        return replace(cfg, llm_model=chosen) if chosen else cfg

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.llm_api_key_env)
