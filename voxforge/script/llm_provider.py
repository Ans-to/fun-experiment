"""Zero-API-Key Local LLM Router (Ollama + llama-cpp-python GGUF).

Provides optional neural dialogue polishing and chapter generation when local
Ollama or GGUF weights are available, with deterministic timeout/schema guards.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx


class ZeroKeyLLMClient:
    """Unified interface over local Ollama or llama-cpp-python (zero API keys)."""

    def __init__(self, backend: str, model_id_or_path: str) -> None:
        self.backend = backend
        self.model_id_or_path = model_id_or_path
        self._llama_instance: Any = None

    @property
    def is_neural_llm(self) -> bool:
        return self.backend in ("ollama", "llamacpp")

    def _get_llama(self) -> Any:
        if self._llama_instance is None and self.backend == "llamacpp":
            from llama_cpp import Llama

            self._llama_instance = Llama(
                model_path=self.model_id_or_path,
                n_ctx=4096,
                n_threads=4,
                verbose=False,
            )
        return self._llama_instance

    def generate_json_turns(
        self,
        chapter_title: str,
        facts_block: str,
        target_words: int,
    ) -> list[dict[str, str]] | None:
        """Ask local LLM to write grounded two-host exchanges, returning validated list or None."""
        if not self.is_neural_llm:
            return None

        system_prompt = (
            "You are an Emmy-winning BBC/Netflix documentary film writer composing dual-voice narration for "
            "HOST_A (Alex Vance, cultural historian) and HOST_B (Dr. Maya Lin, structural archaeologist). "
            "Write pure cinematic documentary voiceover — NEVER mention podcasts, studios, microphones, or hosts sitting at a table. "
            "Use ONLY the provided verified research facts; do not invent numbers, motives, dates, causal claims, or quotations. "
            "Write for the ear: one clear idea per sentence, mostly 8–20 words, active verbs, concrete nouns, natural contractions, "
            "and varied sentence length. Keep each speaker turn to 1–3 sentences and at most 45 words. "
            "Explain unavoidable specialist terms in plain language. Cut generic praise, repeated summaries, filler transitions, "
            "and lines that merely describe what a viewer can already see. Give important facts room to land; do not cram lists. "
            "Never use clichés like 'delve', 'tapestry', 'testament to', or 'buckle up'. "
            "Output ONLY a JSON array of objects: [{\"speaker\": \"HOST_A\", \"text\": \"...\"}, ...]."
        )
        user_prompt = (
            f"Documentary Act: {chapter_title}\n"
            f"Target total words: {target_words}\n"
            f"Verified Research Facts:\n{facts_block}\n\n"
            "Write vivid, authoritative documentary narration alternating between historical scope and engineering forensics as JSON:"
        )

        raw_text = ""
        try:
            if self.backend == "ollama":
                resp = httpx.post(
                    "http://127.0.0.1:11434/api/chat",
                    json={
                        "model": self.model_id_or_path,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "stream": False,
                        "options": {"temperature": 0.65, "num_predict": min(2800, target_words * 3)},
                    },
                    timeout=60.0,
                )
                if resp.status_code == 200:
                    raw_text = resp.json().get("message", {}).get("content", "")
            elif self.backend == "llamacpp":
                llm = self._get_llama()
                out = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.65,
                    max_tokens=min(2400, target_words * 2),
                )
                raw_text = out["choices"][0]["message"]["content"]
        except Exception:
            return None

        if not raw_text:
            return None

        m = re.search(r"\[.*\]", raw_text, flags=re.DOTALL)
        if not m:
            return None
        try:
            parsed = json.loads(m.group(0))
            valid: list[dict[str, str]] = []
            for item in parsed:
                if isinstance(item, dict) and item.get("speaker") in ("HOST_A", "HOST_B") and item.get("text"):
                    txt = str(item["text"]).strip()
                    if len(txt.split()) >= 8:
                        valid.append({"speaker": str(item["speaker"]), "text": txt})
            return valid if len(valid) >= 4 else None
        except Exception:
            return None
