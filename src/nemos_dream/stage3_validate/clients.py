"""Stage-3 NIM client subclasses.

Each class subclasses ``nvidia_clients.NvidiaAsyncClient`` with a stage-
specific ``call(...)`` coroutine. When ``NVIDIA_API_KEY`` is not set, the
stage runner skips constructing these and falls back to the offline
deterministic stubs in each phase module — that way ``uv run pytest`` and
``scripts/run_stage3_example.py`` work with zero credentials while live
runs exercise the same code paths the hackathon submission is judged on.

Model defaults (override via env or explicit ctor arg):

- ``SAFETY_MODEL``   — ``nvidia/llama-3.1-nemoguard-8b-content-safety``
- ``JUDGE_MODEL``    — ``nvidia/nemotron-3-super-120b-a12b``
- ``REWARD_MODEL``   — ``nvidia/nemotron-4-340b-reward``
- ``EMBED_MODEL``    — ``nvidia/llama-3.2-nv-embedqa-1b-v2``

The embed client uses ``langchain_nvidia_ai_endpoints.NVIDIAEmbeddings``
(Curator's reference embedder) instead of raw OpenAI — NIM's embedding
endpoint is accessed through a slightly different surface than chat.
"""

from __future__ import annotations

import json
import os
from typing import Any

from nemos_dream.nvidia_clients import NvidiaAsyncClient, NvidiaSyncClient

_JUDGE_PROMPT = """You are a senior Korean-localisation reviewer for a SODA-style dialogue dataset.
Score the Korean rewrite below on 5 axes (integers 1-5). Return ONLY a JSON object.

[English source]
{en}

[Korean rewrite]
{ko}

[Speaker register/emotion meta]
register={register}  emotion={emotion}(intensity {intensity}/5)  speech_acts={speech_acts}

[Cultural refs mapped]
{refs}

Axes (1 = broken, 5 = flawless):
- property_preservation: speech-act/register/emotion intensity preserved across the rewrite
- naturalness: reads like natural Korean for this platform/age bracket
- cultural_appropriateness: cultural substitutions are idiomatic (not hallucinated) in KR
- register_consistency: honorific level stays consistent across turns
- persona_style_consistency: each speaker's line matches the assigned KR persona

Reply with:
{{"property_preservation": X,
  "naturalness": X,
  "cultural_appropriateness": X,
  "register_consistency": X,
  "persona_style_consistency": X,
  "reasoning": {{"property_preservation": "...", "naturalness": "...",
                 "cultural_appropriateness": "...", "register_consistency": "...",
                 "persona_style_consistency": "..."}}}}
"""


_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "property_preservation": {"type": "integer", "minimum": 1, "maximum": 5},
        "naturalness": {"type": "integer", "minimum": 1, "maximum": 5},
        "cultural_appropriateness": {"type": "integer", "minimum": 1, "maximum": 5},
        "register_consistency": {"type": "integer", "minimum": 1, "maximum": 5},
        "persona_style_consistency": {"type": "integer", "minimum": 1, "maximum": 5},
        "reasoning": {"type": "object", "additionalProperties": {"type": "string"}},
    },
    "required": [
        "property_preservation",
        "naturalness",
        "cultural_appropriateness",
        "register_consistency",
        "persona_style_consistency",
    ],
}


_PER_MODEL_KEYS = ("NEMO_GUARD", "NEMO_3_SUPER", "NEMO_REWARD", "NEMO_EMBED")


def nvidia_api_key_available() -> bool:
    """True when either the blanket key or all four per-model keys are present."""
    if os.environ.get("NVIDIA_API_KEY"):
        return True
    return all(os.environ.get(k) for k in _PER_MODEL_KEYS)


class SafetyClient(NvidiaAsyncClient):
    """NeMoGuard content-safety screen.

    The hosted NIM endpoint returns a JSON envelope with categorical flags
    (``S1``–``S14`` etc. per MLCommons taxonomy). ``call`` returns ``True``
    when the payload is clean, ``False`` when any category fires.
    """

    model_env = "SAFETY_MODEL"
    api_key_env = "NEMO_GUARD"

    def __init__(self, *, model: str | None = None, **kwargs: Any) -> None:
        super().__init__(
            model=model or "nvidia/llama-3.1-nemoguard-8b-content-safety",
            **kwargs,
        )

    async def call(self, text: str) -> bool:  # type: ignore[override]
        resp = await self.openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": text}],
            temperature=0.0,
        )
        raw = resp.choices[0].message.content or ""
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            # NIM sometimes wraps the JSON in a code fence when refused
            return False
        for k, v in payload.items():
            if k.startswith("S") and str(v).lower() == "yes":
                return False
        return True


class JudgeClient(NvidiaAsyncClient):
    """Nemotron judge — 5 axes + per-axis reasoning via ``guided_json``."""

    model_env = "JUDGE_MODEL"
    api_key_env = "NEMO_3_SUPER"

    def __init__(self, *, model: str | None = None, **kwargs: Any) -> None:
        super().__init__(
            model=model or "nvidia/nemotron-3-super-120b-a12b",
            **kwargs,
        )

    async def call(  # type: ignore[override]
        self,
        *,
        en: str,
        ko: str,
        register: str,
        emotion: str,
        intensity: int,
        speech_acts: list[str],
        refs: str,
    ) -> dict[str, Any]:
        prompt = _JUDGE_PROMPT.format(
            en=en,
            ko=ko,
            register=register,
            emotion=emotion,
            intensity=intensity,
            speech_acts=",".join(speech_acts),
            refs=refs,
        )
        resp = await self.openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            extra_body={"nvext": {"guided_json": _JUDGE_SCHEMA}},
        )
        return json.loads(resp.choices[0].message.content or "{}")


_REWARD_PROMPT = """You are scoring an EN→KR rewrite as a reward model. Return ONLY a JSON
object with two integer fields on a 1-5 scale.

[English source]
{en}

[Korean rewrite]
{ko}

Axes (1 = terrible, 5 = excellent):
- correctness: Does the Korean rewrite convey the same meaning as the source without losing or adding facts?
- coherence: Does the Korean dialogue read as a single coherent exchange (turn flow, reference, tone)?

Reply with:
{{"correctness": X, "coherence": X}}
"""


_REWARD_SCHEMA = {
    "type": "object",
    "properties": {
        "correctness": {"type": "integer", "minimum": 1, "maximum": 5},
        "coherence": {"type": "integer", "minimum": 1, "maximum": 5},
    },
    "required": ["correctness", "coherence"],
}


class RewardClient(NvidiaAsyncClient):
    """Nemotron-3-Super-120B reward-style scorer.

    ``nvidia/nemotron-4-340b-reward`` reached EOL in NIM's managed catalog
    (the sibling 70B-reward retired 2026-04-15); the active replacement is
    to run a short judge-style prompt on ``nvidia/nemotron-3-super-120b-a12b``
    and capture ``correctness`` / ``coherence`` on a 1-5 scale. The output
    shape (``dict[str, float]``) matches what consumers expected from the
    old logprob-based reward endpoint.
    """

    model_env = "REWARD_MODEL"
    api_key_env = "NEMO_REWARD"

    def __init__(self, *, model: str | None = None, **kwargs: Any) -> None:
        super().__init__(
            model=model or "nvidia/nemotron-3-super-120b-a12b",
            **kwargs,
        )

    async def call(self, *, en: str, ko: str) -> dict[str, float]:  # type: ignore[override]
        prompt = _REWARD_PROMPT.format(en=en, ko=ko)
        resp = await self.openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            extra_body={"nvext": {"guided_json": _REWARD_SCHEMA}},
        )
        payload = json.loads(resp.choices[0].message.content or "{}")
        scores: dict[str, float] = {}
        for k in ("correctness", "coherence"):
            v = payload.get(k)
            if isinstance(v, int):
                scores[k] = float(v)
        return scores


class EmbedClient(NvidiaSyncClient):
    """NV-Embed wrapper using ``langchain_nvidia_ai_endpoints.NVIDIAEmbeddings``.

    Sync on purpose — Curator's embedder call patterns are sync and phase
    4 (intra-KR coherence) batches per-row, so the async surface would only
    add overhead. Raises at import-time only when ``embed`` is invoked
    without the NV AI endpoints package installed.
    """

    model_env = "EMBED_MODEL"
    api_key_env = "NEMO_EMBED"

    def __init__(self, *, model: str | None = None, **kwargs: Any) -> None:
        super().__init__(
            model=model or "nvidia/llama-3.2-nv-embedqa-1b-v2",
            **kwargs,
        )
        self._lc_embed: Any = None

    @property
    def langchain(self) -> Any:
        if self._lc_embed is None:
            from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

            self._lc_embed = NVIDIAEmbeddings(
                model=self.model,
                api_key=self._api_key(),
                base_url=self.base_url,
            )
        return self._lc_embed

    def call(self, texts: list[str]) -> list[list[float]]:  # type: ignore[override]
        return [list(v) for v in self.langchain.embed_documents(texts)]

    def embed_fn(self):
        """Return a plain ``Callable[[list[str]], list[list[float]]]``.

        Each phase module expects a naked callable, not a client instance,
        so they can stay decoupled from NVIDIA plumbing and unit-test with a
        lambda.
        """

        return lambda texts: self.call(texts)


def build_default_clients() -> dict[str, Any]:
    """Lazy-construct every stage-3 NIM client when ``NVIDIA_API_KEY`` is set.

    Returns a dict suitable for splatting into ``runner.run_async``::

        runner.run_async(input_path, out_dir, **build_default_clients())

    When the key is missing, returns ``{}`` so the runner's offline stubs
    kick in transparently.
    """

    if not nvidia_api_key_available():
        return {}

    safety = SafetyClient()
    judge = JudgeClient()
    reward = RewardClient()
    embed = EmbedClient()

    async def safety_fn(text: str) -> bool:
        return await safety.call(text)

    async def judge_fn(**kw: Any) -> dict[str, Any]:
        return await judge.call(**kw)

    async def reward_fn(**kw: Any) -> dict[str, float]:
        return await reward.call(**kw)

    return {
        "safety_fn": safety_fn,
        "judge_fn": judge_fn,
        "reward_fn": reward_fn,
        "embed_fn": embed.embed_fn(),
    }
