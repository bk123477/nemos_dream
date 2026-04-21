"""ABC bases for NVIDIA-hosted clients (NIM chat, judge, safety, reward, embeddings).

Design goal: keep the *shared* bits (base URL, API key loading, sensible
timeouts/retries) in one place, and let each stage subclass freely for its
own ``.call(...)`` shape without touching this file.

Typical usage — subclass in your stage module::

    from nemos_dream.nvidia_clients import NvidiaSyncClient

    class DecomposeClient(NvidiaSyncClient):
        model_env = "NEMOTRON_MODEL"

        def call(self, text: str, *, schema: dict) -> str:
            resp = self.openai.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": text}],
                temperature=0.2,
                extra_body={"nvext": {"guided_json": schema}},
            )
            return resp.choices[0].message.content

Async variant: inherit from :class:`NvidiaAsyncClient` and make ``call`` a
coroutine.

**Rule of thumb for edits here:** only add things that *every* stage would
benefit from. Per-stage helpers belong in the stage subpackage, not in this
base. Renames/removals must be additive — downstream subclasses depend on
``base_url``, ``api_key_env``, ``model_env``, ``openai``.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any

from openai import AsyncOpenAI, OpenAI

_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"


class NvidiaClient(ABC):
    """Common base: URL + API key + retry/timeout defaults.

    Subclasses override ``model_env`` (and optionally ``base_url`` /
    ``api_key_env``) and must implement :meth:`call`.
    """

    base_url: str = _DEFAULT_BASE_URL
    api_key_env: str = "NVIDIA_API_KEY"
    model_env: str | None = None  # override in subclass, e.g. "NEMOTRON_MODEL"

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        **openai_kwargs: Any,
    ) -> None:
        if base_url is not None:
            self.base_url = base_url
        if api_key_env is not None:
            self.api_key_env = api_key_env
        self.model = model or self._resolve_model_from_env()
        self._openai_kwargs = {"timeout": timeout, "max_retries": max_retries, **openai_kwargs}

    def _resolve_model_from_env(self) -> str | None:
        env = self.model_env
        if env is None:
            return None
        value = os.environ.get(env) or os.environ.get("NVIDIA_API_BASE_DEFAULT_MODEL")
        return value

    def _api_key(self) -> str:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(
                f"Environment variable {self.api_key_env!r} is not set — "
                "copy .env.example to .env and fill in NVIDIA_API_KEY."
            )
        return key

    def _connection_kwargs(self) -> dict[str, Any]:
        return {"base_url": self.base_url, "api_key": self._api_key(), **self._openai_kwargs}

    @abstractmethod
    def call(self, *args: Any, **kwargs: Any) -> Any:
        """Stage-specific entrypoint — subclass defines the signature."""


class NvidiaSyncClient(NvidiaClient):
    """Sync base — gives subclasses ``self.openai: OpenAI`` lazily."""

    @cached_property
    def openai(self) -> OpenAI:
        return OpenAI(**self._connection_kwargs())


class NvidiaAsyncClient(NvidiaClient):
    """Async base — gives subclasses ``self.openai: AsyncOpenAI`` lazily."""

    @cached_property
    def openai(self) -> AsyncOpenAI:
        return AsyncOpenAI(**self._connection_kwargs())
