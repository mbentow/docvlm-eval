"""Ollama runner.

Two settings do most of the work and are on by default here:

``format`` = the JSON Schema
    Ollama supports constrained decoding **including on vision models**. Asking
    a model to "reply in JSON" and then regex-ing the result measures your regex
    as much as the model.

``temperature`` = 0
    Sampling noise between two runs of the same config is indistinguishable from
    a real difference, which quietly ruins every A/B you run afterwards.

Multiple hosts can be given; requests are handed out round-robin, which is how
you use a two-machine pool without a load balancer in front.
"""

from __future__ import annotations

import asyncio
import base64
import itertools
import time
from typing import Any

import httpx

from docvlm_eval.runners.base import Runner, RunnerOutput

DEFAULT_HOST = "http://localhost:11434"


class OllamaRunner(Runner):
    """Talks to one or more Ollama servers."""

    name = "ollama"

    def __init__(
        self,
        model: str,
        params: dict[str, Any] | None = None,
        *,
        host: str | list[str] = DEFAULT_HOST,
        timeout: float = 300.0,
    ) -> None:
        super().__init__(model, params)
        hosts = [host] if isinstance(host, str) else list(host)
        self.hosts = [h.rstrip("/") for h in hosts if h]
        if not self.hosts:
            raise ValueError("OllamaRunner needs at least one host")
        self._cycle = itertools.cycle(self.hosts)
        self._lock = asyncio.Lock()
        self._client = httpx.AsyncClient(timeout=timeout)

    async def _next_host(self) -> str:
        async with self._lock:
            return next(self._cycle)

    async def extract(
        self,
        image: bytes,
        prompt: str,
        json_schema: dict[str, Any],
        case_id: str = "",
    ) -> RunnerOutput:
        host = await self._next_host()
        options = {
            "temperature": self.params.get("temperature", 0),
            **{
                k: v
                for k, v in self.params.items()
                if k in {"num_ctx", "num_predict", "top_p", "top_k", "seed", "repeat_penalty"}
            },
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "images": [base64.b64encode(image).decode()],
            "stream": False,
            "options": options,
            "keep_alive": self.params.get("keep_alive", "5m"),
        }
        if self.params.get("format", "schema") == "schema":
            payload["format"] = json_schema
        elif self.params.get("format") == "json":
            payload["format"] = "json"

        started = time.perf_counter()
        try:
            response = await self._client.post(f"{host}/api/generate", json=payload)
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as exc:
            return RunnerOutput(
                None,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=f"HTTP {exc.response.status_code}: {exc.response.text[:200]}",
                meta={"host": host},
            )
        except (httpx.HTTPError, ValueError) as exc:
            return RunnerOutput(
                None,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=f"{type(exc).__name__}: {exc}",
                meta={"host": host},
            )

        latency_ms = (time.perf_counter() - started) * 1000
        raw = body.get("response", "") or ""
        data = self.parse_json(raw)
        return RunnerOutput(
            data=data,
            raw=raw,
            latency_ms=latency_ms,
            tokens_in=int(body.get("prompt_eval_count") or 0),
            tokens_out=int(body.get("eval_count") or 0),
            cost_usd=0.0,  # local inference; energy is tracked elsewhere
            error="" if data is not None else "output was not valid JSON",
            meta={
                "host": host,
                "eval_duration_ms": (body.get("eval_duration") or 0) / 1e6,
                "prompt_eval_duration_ms": (body.get("prompt_eval_duration") or 0) / 1e6,
                "load_duration_ms": (body.get("load_duration") or 0) / 1e6,
            },
        )

    async def describe(self) -> dict[str, Any]:
        """Record the model digest, not just its tag.

        Tags are mutable — ``qwen3-vl:30b`` today is not necessarily the blob you
        benchmarked last month. The digest is what makes the run reproducible.
        """
        info: dict[str, Any] = {
            "runner": self.name,
            "model": self.model,
            "hosts": self.hosts,
        }
        try:
            response = await self._client.post(
                f"{self.hosts[0]}/api/show", json={"model": self.model}, timeout=30.0
            )
            response.raise_for_status()
            body = response.json()
            details = body.get("details", {}) or {}
            info["quantization"] = details.get("quantization_level")
            info["parameter_size"] = details.get("parameter_size")
            info["family"] = details.get("family")
        except (httpx.HTTPError, ValueError):
            pass
        try:
            tags = await self._client.get(f"{self.hosts[0]}/api/tags", timeout=30.0)
            tags.raise_for_status()
            for entry in tags.json().get("models", []):
                if entry.get("name") == self.model:
                    info["model_digest"] = (entry.get("digest") or "")[:16]
                    info["model_size_bytes"] = entry.get("size")
                    break
        except (httpx.HTTPError, ValueError):
            pass
        try:
            version = await self._client.get(f"{self.hosts[0]}/api/version", timeout=10.0)
            version.raise_for_status()
            info["server_version"] = version.json().get("version")
        except (httpx.HTTPError, ValueError):
            pass
        return info

    async def aclose(self) -> None:
        await self._client.aclose()
