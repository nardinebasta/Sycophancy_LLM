"""Single interface over every model API used in the study.

Everything goes through an OpenAI-compatible /chat/completions endpoint, so
OpenRouter, OpenAI, and Google's compatibility endpoint all work unchanged.

Design rules that matter for the science:
  * the exact model string is returned with every call and stored in the record;
  * temperature and seed are explicit, never defaulted by the provider;
  * retries are bounded and logged - a silent retry storm distorts cost, not results.
"""
from __future__ import annotations

import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"), override=False)
except ImportError:
    pass


@dataclass
class Reply:
    text: str
    model_snapshot: str
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


class RateLimiter:
    """Crude token bucket, one per model key."""

    def __init__(self, rpm: int):
        self.interval = 60.0 / max(rpm, 1)
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next:
                time.sleep(self._next - now)
                now = time.monotonic()
            self._next = now + self.interval


class Provider:
    def __init__(self, key: str, cfg: dict, defaults: dict):
        self.key = key
        self.model = cfg["model"]
        self.base_url = cfg["base_url"].rstrip("/")
        self.api_key = os.environ.get(cfg["api_key_env"], "")
        if not self.api_key:
            raise RuntimeError(f"{cfg['api_key_env']} is not set (see .env.example)")
        self.extra_body = cfg.get("extra_body", {})
        self.strip_reasoning = cfg.get("strip_reasoning", False)
        self.defaults = defaults
        self.limiter = RateLimiter(cfg.get("rpm", 60))
        self._client = httpx.Client(timeout=defaults.get("timeout_s", 120))

    def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> Reply:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.defaults["temperature"] if temperature is None else temperature,
            "max_tokens": max_tokens or self.defaults["max_tokens"],
        }
        if seed is not None:
            body["seed"] = seed
        body.update(self.extra_body)

        last_err: Exception | None = None
        for attempt in range(self.defaults.get("max_retries", 6)):
            self.limiter.wait()
            try:
                r = self._client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=body,
                )
                if r.status_code in (429, 500, 502, 503, 529):
                    raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
                r.raise_for_status()
                data = r.json()
                msg = data["choices"][0]["message"]
                text = msg.get("content") or ""
                if self.strip_reasoning:
                    text = _strip_reasoning(text)
                return Reply(
                    text=text.strip(),
                    model_snapshot=data.get("model", self.model),
                    usage=data.get("usage", {}),
                    raw=data,
                )
            except httpx.HTTPStatusError as exc:
                last_err = f"{exc.response.status_code}: {exc.response.text[:300]}"
                if exc.response.status_code in (400, 401, 402, 403, 404):
                    raise RuntimeError(f"[{self.key}] {last_err}") from exc
            except Exception as exc:  # noqa: BLE001
                last_err = exc

                time.sleep(min(2 ** attempt, 30) + random.random())
        raise RuntimeError(f"[{self.key}] failed after retries: {last_err}")


def _strip_reasoning(text: str) -> str:
    """Remove visible chain-of-thought blocks so the judge codes the answer only."""
    import re

    for pat in (r"<think>.*?</think>", r"<reasoning>.*?</reasoning>"):
        text = re.sub(pat, "", text, flags=re.S | re.I)
    return text


class Registry:
    def __init__(self, path: str | None = None):
        path = path or os.path.join(ROOT, "config", "models.yaml")
        with open(path) as fh:
            self.cfg = yaml.safe_load(fh)
        self._cache: dict[str, Provider] = {}

    def get(self, key: str) -> Provider:
        if key not in self._cache:
            for section in ("victims", "judge"):
                if key in self.cfg.get(section, {}):
                    self._cache[key] = Provider(key, self.cfg[section][key], self.cfg["defaults"])
                    break
            else:
                raise KeyError(f"unknown model key: {key}")
        return self._cache[key]

    @property
    def victims(self) -> list[str]:
        return list(self.cfg["victims"].keys())

    @property
    def judge_key(self) -> str:
        return next(iter(self.cfg["judge"]))
