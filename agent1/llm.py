"""Client LLM minimal pour toute API compatible OpenAI (Groq, OpenAI, vLLM...)."""
from __future__ import annotations

import re
import time

import requests

from .config import Config


class LLMError(RuntimeError):
    pass


_THINK = re.compile(r"<think>.*?</think>", re.S)


class LLMClient:
    def __init__(self, cfg: Config, session: requests.Session | None = None):
        self.cfg = cfg
        self.session = session or requests.Session()

    def chat(self, system: str, user: str, temperature: float | None = None,
             max_tokens: int | None = None) -> str:
        key = self.cfg.api_key
        if not key:
            raise LLMError(f"clé API absente : définir la variable {self.cfg.llm_api_key_env}")
        payload = {
            "model": self.cfg.llm_model,
            "messages": ([{"role": "system", "content": system}] if system else [])
                        + [{"role": "user", "content": user}],
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.cfg.max_tokens,
        }
        if self.cfg.reasoning_effort:
            payload["reasoning_effort"] = self.cfg.reasoning_effort
        url = self.cfg.llm_base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {key}"}
        last = ""
        for attempt in range(self.cfg.llm_retries):
            try:
                r = self.session.post(url, json=payload, headers=headers,
                                      timeout=self.cfg.timeout_s)
            except requests.RequestException as e:
                last = str(e)
                time.sleep(min(5 * 2 ** attempt, 60))   # coupure réseau : attendre plus longtemps
                continue
            if r.status_code == 200:
                text = r.json()["choices"][0]["message"]["content"] or ""
                text = _THINK.sub("", text).strip()   # modèles à raisonnement (Qwen3...)
                if not text:   # jamais de réponse vide silencieuse
                    raise LLMError("réponse vide : budget de jetons épuisé par le raisonnement ? "
                                   "augmenter max_tokens ou baisser reasoning_effort")
                return text
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code in (429, 500, 502, 503, 504):
                wait = r.headers.get("retry-after")
                time.sleep(float(wait) if wait and wait.replace(".", "").isdigit()
                           else 2 ** attempt)
                continue
            break
        raise LLMError(last)
