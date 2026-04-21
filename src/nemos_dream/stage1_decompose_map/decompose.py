"""Stage 1a — sociolinguistic decomposition of a SODA-style dialogue.

**Primary path — NeMo Data Designer** (``_decompose_data_designer``):
   Batched ``llm-structured`` column on NVIDIA NIM. Uses the private
   :class:`_FullDecomposition` wrapper (speakers + scene + dialogue_decomposed)
   as ``output_format`` so DD's built-in ``PydanticResponseRecipe`` parses and
   validates the response. Matches the working recipe from the source repo.

**Fallback — unified NIM one-shot** (``_decompose_sequential``):
   Single ``guided_json`` chat completion per row via the team's
   :class:`NvidiaSyncClient` factory. Triggers on ``ImportError`` or any DD
   runtime failure.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel

from nemos_dream.nvidia_clients import NvidiaSyncClient
from nemos_dream.schemas import (
    CulturalRef,
    DialogueDecomposed,
    Emotion,
    RawInput,
    Scene,
    Speaker,
    Turn,
)

from .prompts import SYSTEM_PROMPT, USER_TEMPLATE, format_dialogue_block


class _StrictSpeaker(BaseModel):
    """Strict speaker schema used only as DD's ``output_format``.

    Uses ``Literal`` enums (like the source repo's ``SpeakerProfile``) so DD's
    auto-generated prompt schema tightly constrains LLM output. Values are
    converted to the team's looser :class:`Speaker` (whose ``role_in_scene``
    and ``gender_hint`` are plain ``str``) in :func:`_pack`.
    """

    name_en: str
    role_in_scene: Literal[
        "parent", "child", "sibling", "spouse", "partner", "friend",
        "coworker", "boss", "subordinate", "teacher", "student", "stranger",
        "service_staff", "other",
    ]
    gender_hint: Literal["male", "female", "unknown"]
    age_group_hint: Literal["teen", "20s", "30s", "40plus", "unknown"]
    register: Literal["intimate", "casual", "formal", "public"]
    dominant_emotion: Emotion
    personality_traits: list[str] = []
    interests_hints: list[str] = []
    occupation_hint: str = ""
    speech_style_notes: str = ""


class _StrictScene(BaseModel):
    """Strict scene schema (Literal enums) — see :class:`_StrictSpeaker`."""

    narrative_en: str
    setting: Literal[
        "home", "school", "workplace", "restaurant", "phone_call",
        "text_message", "public_space", "vehicle", "online", "other",
    ]
    relationship_type: Literal[
        "family", "romantic", "friendship", "professional",
        "acquaintance", "stranger", "other",
    ]
    topics: list[str] = []


class _FullDecomposition(BaseModel):
    """Combined shape the LLM is asked to return — mirrors the source repo's
    ``DecomposedDialogue`` so DD's pydantic recipe has a concrete schema to
    drive both prompt injection and response parsing."""

    speakers: list[_StrictSpeaker]
    scene: _StrictScene
    dialogue_decomposed: DialogueDecomposed

_ALLOWED_REF_TYPES = {
    "holiday", "brand", "service", "event", "food", "pop_culture", "slang",
    "person", "place", "meme", "other",
}
_ALLOWED_SPEECH_ACTS = {
    "complaint", "brag", "question", "empathy_seeking", "sarcasm",
    "joke", "statement", "greeting", "request", "announce", "advice", "other",
}
_SPEECH_ACT_ALIASES = {
    "expressive": "statement", "exclamation": "statement", "assertive": "statement",
    "directive": "request", "commissive": "statement", "declaration": "statement",
    "inform": "statement", "informative": "statement",
    "thanks": "greeting", "farewell": "greeting",
    "empathy": "empathy_seeking", "sympathy_seeking": "empathy_seeking",
    "praise": "brag", "boast": "brag",
}
_ALLOWED_EMOTIONS = {"joy", "anger", "sadness", "fear", "surprise", "disgust", "neutral"}
_EMOTION_ALIASES = {
    "happiness": "joy", "love": "joy", "excitement": "joy",
    "frustration": "anger", "annoyance": "anger", "rage": "anger",
    "grief": "sadness", "melancholy": "sadness",
    "anxiety": "fear", "worry": "fear", "nervousness": "fear",
    "amazement": "surprise", "shock": "surprise",
}
_ALLOWED_REGISTERS = {"intimate", "casual", "formal", "public"}
_ALLOWED_AGE = {"teen", "20s", "30s", "40plus", "unknown"}

_AGE_PATTERNS = [
    ("teen", ["teen", "13", "14", "15", "16", "17", "18", "19", "youth"]),
    ("20s", ["20", "college", "young adult", "twenties"]),
    ("30s", ["30", "thirties"]),
    ("40plus", ["40", "50", "60", "senior", "elder", "middle-aged"]),
]


class DecomposeResult(NamedTuple):
    turns: list[Turn]
    speakers: list[Speaker]
    scene: Scene
    dialogue_decomposed: DialogueDecomposed


# --------------------------------------------------------------------------- #
# normalization helpers
# --------------------------------------------------------------------------- #

def _clean_emotion(em: Any) -> dict:
    if isinstance(em, str):
        em = {"type": em, "intensity": 3}
    if not isinstance(em, dict):
        em = {}
    et = str(em.get("type", "")).lower()
    et = _EMOTION_ALIASES.get(et, et)
    etype = et if et in _ALLOWED_EMOTIONS else "neutral"
    try:
        inten = int(round(float(em.get("intensity", 3))))
    except (TypeError, ValueError):
        inten = 3
    inten = max(1, min(5, inten))
    return {"type": etype, "intensity": inten}


def _clean_register(val: Any) -> str:
    reg = str(val or "casual").lower()
    return reg if reg in _ALLOWED_REGISTERS else "casual"


def _clean_age(val: Any) -> str:
    raw = str(val or "").lower()
    if raw in _ALLOWED_AGE:
        return raw
    for target, markers in _AGE_PATTERNS:
        if any(m in raw for m in markers):
            return target
    return "unknown"


def _clean_gender(val: Any) -> str:
    g = str(val or "unknown").lower()
    if g in {"male", "female", "unknown"}:
        return g
    if g in {"m", "man", "boy"}:
        return "male"
    if g in {"f", "woman", "girl"}:
        return "female"
    return "unknown"


def _clean_str_list(val: Any, *, max_len: int = 5) -> list[str]:
    if isinstance(val, str):
        items = [v.strip() for v in val.replace("/", ",").split(",") if v.strip()]
    elif isinstance(val, list):
        items = [str(v).strip() for v in val if str(v).strip()]
    else:
        return []
    seen: list[str] = []
    for item in (x.lower() for x in items):
        if item and item not in seen:
            seen.append(item)
        if len(seen) >= max_len:
            break
    return seen


def _clean_refs(refs: Any, *, dialogue_text: str) -> list[dict]:
    """Coerce cultural_refs into valid CulturalRef dicts.

    Enforces the team invariant that ``term`` must appear verbatim in the
    dialogue text by dropping entries that don't substring-match.
    """
    if not isinstance(refs, list):
        return []
    haystack = dialogue_text.lower()
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for r in refs:
        if isinstance(r, str):
            entry = {"type": "other", "term": r}
        elif isinstance(r, dict) and "term" in r:
            entry = {"type": r.get("type", "other"), "term": str(r["term"])}
        else:
            continue
        entry["term"] = entry["term"].lower().strip()
        if not entry["term"] or entry["term"] not in haystack:
            continue
        if entry["type"] not in _ALLOWED_REF_TYPES:
            entry["type"] = "other"
        key = (entry["type"], entry["term"])
        if key in seen:
            continue
        seen.add(key)
        out.append(entry)
    return out


def _clean_speech_acts(acts: Any) -> list[str]:
    if not isinstance(acts, list):
        return []
    seen: list[str] = []
    for a in acts:
        sa = str(a or "").lower().replace(" ", "_")
        sa = _SPEECH_ACT_ALIASES.get(sa, sa)
        if sa in _ALLOWED_SPEECH_ACTS and sa not in seen:
            seen.append(sa)
    return seen


def _unique_speakers(speakers: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in speakers:
        s = str(s).strip()
        if s and s not in seen:
            out.append(s)
            seen.add(s)
    return out


def _normalize_speakers_scene(
    data: dict,
    *,
    speakers_in: list[str],
    narrative_in: str,
) -> tuple[list[dict], dict]:
    """Coerce an LLM payload's ``speakers`` + ``scene`` fields."""
    raw_speakers = data.get("speakers") or []
    cleaned: dict[str, dict] = {}
    for sp in raw_speakers:
        if not isinstance(sp, dict):
            continue
        name = str(sp.get("name_en", "")).strip()
        if name not in speakers_in or name in cleaned:
            continue
        cleaned[name] = {
            "name_en": name,
            "role_in_scene": str(sp.get("role_in_scene") or "other"),
            "gender_hint": _clean_gender(sp.get("gender_hint")),
            "age_group_hint": _clean_age(sp.get("age_group_hint")),
            "register": _clean_register(sp.get("register")),
            "dominant_emotion": _clean_emotion(sp.get("dominant_emotion")),
            "personality_traits": _clean_str_list(sp.get("personality_traits")),
            "interests_hints": _clean_str_list(sp.get("interests_hints")),
            "occupation_hint": str(sp.get("occupation_hint") or "").strip(),
            "speech_style_notes": str(sp.get("speech_style_notes") or "").strip(),
        }
    for name in speakers_in:
        if name not in cleaned:
            cleaned[name] = {
                "name_en": name,
                "role_in_scene": "other",
                "gender_hint": "unknown",
                "age_group_hint": "unknown",
                "register": "casual",
                "dominant_emotion": {"type": "neutral", "intensity": 3},
                "personality_traits": [],
                "interests_hints": [],
                "occupation_hint": "",
                "speech_style_notes": "",
            }
    speakers_list = [cleaned[n] for n in speakers_in]

    scene = data.get("scene") or {}
    if not isinstance(scene, dict):
        scene = {}
    # Force narrative verbatim — the only SODA sidecar we must preserve.
    scene["narrative_en"] = narrative_in
    scene["setting"] = str(scene.get("setting") or "other")
    scene["relationship_type"] = str(scene.get("relationship_type") or "other")
    scene["topics"] = _clean_str_list(scene.get("topics"))
    return speakers_list, scene


def _normalize_dialogue_decomposed(dd: dict, *, dialogue_text: str) -> dict:
    """Coerce an LLM payload's ``dialogue_decomposed`` field."""
    if not isinstance(dd, dict):
        dd = {}
    dd["overall_register"] = _clean_register(dd.get("overall_register"))
    dd["overall_emotion"] = _clean_emotion(dd.get("overall_emotion"))
    dd["speech_acts"] = _clean_speech_acts(dd.get("speech_acts"))
    dd["cultural_refs"] = _clean_refs(dd.get("cultural_refs"), dialogue_text=dialogue_text)
    return dd


def _normalize(
    data: dict,
    *,
    speakers_in: list[str],
    narrative_in: str,
    dialogue_text: str,
) -> dict:
    """Coerce a unified LLM payload (speakers + scene + dialogue_decomposed)."""
    speakers_list, scene = _normalize_speakers_scene(
        data, speakers_in=speakers_in, narrative_in=narrative_in,
    )
    data["speakers"] = speakers_list
    data["scene"] = scene
    data["dialogue_decomposed"] = _normalize_dialogue_decomposed(
        data.get("dialogue_decomposed") or {}, dialogue_text=dialogue_text,
    )
    return data


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays in a DD payload to plain Python.

    DD's ``DataFrame.iterrows()`` can hand back pydantic payloads wrapped in
    ``numpy.ndarray``; downstream dict-walking then trips ``ValueError: The
    truth value of an array...``. This flattens everything to JSON-native
    types so :func:`_normalize` and :func:`_pack` work unchanged.
    """
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore[assignment]
    if np is not None and isinstance(obj, np.ndarray):
        return [_to_jsonable(x) for x in obj.tolist()]
    if np is not None and isinstance(obj, np.generic):
        return obj.item()
    if hasattr(obj, "model_dump"):
        return _to_jsonable(obj.model_dump())
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return obj


def _pack(row: RawInput, data: dict) -> DecomposeResult:
    turns = [
        Turn(index=i, speaker=sp, text=utt)
        for i, (sp, utt) in enumerate(zip(row.speakers, row.dialogue, strict=False))
    ]
    speakers = [Speaker.model_validate(s) for s in data["speakers"]]
    scene = Scene.model_validate(data["scene"])
    dd_raw = data["dialogue_decomposed"]
    dd = DialogueDecomposed(
        overall_register=dd_raw["overall_register"],
        overall_emotion=Emotion.model_validate(dd_raw["overall_emotion"]),
        speech_acts=dd_raw["speech_acts"],
        cultural_refs=[CulturalRef.model_validate(r) for r in dd_raw["cultural_refs"]],
    )
    return DecomposeResult(turns=turns, speakers=speakers, scene=scene, dialogue_decomposed=dd)


# --------------------------------------------------------------------------- #
# NIM fallback path (sequential)
# --------------------------------------------------------------------------- #

class _DecomposeClient(NvidiaSyncClient):
    """One-shot full-decomposition NIM client used by the unified fallback."""

    model_env = "NEMOTRON_MODEL"

    def call(self, user_prompt: str, *, schema: dict, temperature: float = 0.2) -> str:
        model = self.model or "nvidia/nemotron-3-nano-30b-a3b"
        resp = self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            extra_body={"nvext": {"guided_json": schema}},
            temperature=temperature,
        )
        return resp.choices[0].message.content or "{}"


def _decompose_sequential(rows: list[RawInput]) -> list[DecomposeResult]:
    base_url = os.environ.get("NVIDIA_API_BASE")
    client = _DecomposeClient(base_url=base_url) if base_url else _DecomposeClient()

    # Expected shape — mirrors ``DialogueDecomposed`` plus speakers and scene.
    schema = {
        "type": "object",
        "properties": {
            "speakers": {"type": "array"},
            "scene": {"type": "object"},
            "dialogue_decomposed": {"type": "object"},
        },
        "required": ["speakers", "scene", "dialogue_decomposed"],
    }

    out: list[DecomposeResult] = []
    for row in rows:
        uniq = _unique_speakers(row.speakers)
        user_prompt = USER_TEMPLATE.format(
            narrative=row.narrative,
            speakers=json.dumps(uniq, ensure_ascii=False),
            dialogue_block=format_dialogue_block(row.dialogue, row.speakers),
        )
        raw = client.call(user_prompt, schema=schema)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}
        data = _normalize(
            data,
            speakers_in=uniq,
            narrative_in=row.narrative,
            dialogue_text=" ".join(row.dialogue).lower(),
        )
        out.append(_pack(row, data))
    return out


# --------------------------------------------------------------------------- #
# Data Designer path — batched full decomposition (primary)
#
# Mirrors the recipe the source repo (`data_pipeline_phase1`) used before the
# port: pass a concrete pydantic class as ``output_format`` (so DD drives both
# prompt schema injection and response parsing), embed the full SYSTEM_PROMPT
# in the column prompt, leave concurrency/timeout at sensible defaults, and do
# NOT add ``extra_body.nvext.guided_json`` — it breaks DD's internal parser on
# build.nvidia.com's Nemotron-Nano endpoint and causes silent per-request hangs.
# --------------------------------------------------------------------------- #

_DD_PROMPT = (
    SYSTEM_PROMPT
    + '\n\nNarrative: """{{ narrative }}"""\n\n'
    + "Speakers: {{ speakers_json }}\n\n"
    + "Dialogue:\n{{ dialogue_block }}\n\n"
    + "Return the JSON object."
)


def _decompose_data_designer(rows: list[RawInput]) -> list[DecomposeResult] | None:
    """Batched DD run — one ``_FullDecomposition`` per row. Returns ``None`` on
    any failure so the caller falls back to :func:`_decompose_sequential`."""
    # DD builds ``httpx.HTTPTransport`` directly, which bypasses httpx's usual
    # env-var proxy discovery. On networks where an HTTP(S)_PROXY is required,
    # that means DD silently cannot reach external NIM endpoints and every
    # request times out at ~80 s. Patch before DD imports wire up their
    # transports.
    from nemos_dream._proxy_patch import apply_proxy_patches
    apply_proxy_patches()

    try:
        import pandas as pd
        from data_designer.config.config_builder import DataDesignerConfigBuilder
        from data_designer.config.models import (
            ChatCompletionInferenceParams,
            ModelConfig,
            ModelProvider,
        )
        from data_designer.config.seed_source import LocalFileSeedSource
        from data_designer.interface import DataDesigner
    except ImportError as exc:
        print(f"[decompose] data-designer import failed ({exc}); using NIM fallback.")
        return None

    try:
        nim = ModelProvider(
            name="nvidia-nim",
            endpoint=os.environ.get("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1"),
            provider_type="openai",
            api_key=os.environ["NVIDIA_API_KEY"],
        )
        model = ModelConfig(
            alias="nemotron-nano",
            model=os.environ.get("NEMOTRON_MODEL", "nvidia/nemotron-3-nano-30b-a3b"),
            provider="nvidia-nim",
            skip_health_check=True,
            inference_parameters=ChatCompletionInferenceParams(
                temperature=0.2,
                max_parallel_requests=16,
                timeout=90,
            ),
        )

        artifact_path = Path(".artifacts") / "data_designer_stage1"
        artifact_path.mkdir(parents=True, exist_ok=True)
        seed_path = artifact_path / "seed.jsonl"

        seed_records = []
        for r in rows:
            uniq = _unique_speakers(r.speakers)
            seed_records.append({
                "narrative": r.narrative,
                "speakers_json": json.dumps(uniq, ensure_ascii=False),
                "dialogue_block": format_dialogue_block(r.dialogue, r.speakers),
            })
        pd.DataFrame(seed_records).to_json(
            seed_path, orient="records", lines=True, force_ascii=False,
        )

        builder = DataDesignerConfigBuilder(model_configs=[model])
        builder.with_seed_dataset(LocalFileSeedSource(path=str(seed_path)))
        builder.add_column(
            name="decomposed",
            column_type="llm-structured",
            prompt=_DD_PROMPT,
            model_alias="nemotron-nano",
            output_format=_FullDecomposition,
        )

        designer = DataDesigner(artifact_path=str(artifact_path), model_providers=[nim])
        result = designer.create(
            config_builder=builder,
            num_records=len(seed_records),
            dataset_name="decomposed",
        )
        df = result.load_dataset() if hasattr(result, "load_dataset") else result

        if len(df) != len(rows):
            print(
                f"[decompose] data-designer produced {len(df)}/{len(rows)} rows; "
                "falling back to sequential NIM for full coverage.",
            )
            return None

        out: list[DecomposeResult] = []
        for row, (_, dd_row) in zip(rows, df.iterrows(), strict=True):
            payload = dd_row["decomposed"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            payload = _to_jsonable(payload)
            if not isinstance(payload, dict):
                payload = {}
            normalized = _normalize(
                payload,
                speakers_in=_unique_speakers(row.speakers),
                narrative_in=row.narrative,
                dialogue_text=" ".join(row.dialogue).lower(),
            )
            out.append(_pack(row, normalized))
        print(f"[decompose] Data Designer produced {len(out)} rows ✓")
        return out
    except Exception as exc:  # noqa: BLE001 — fall back on any DD runtime error
        print(
            f"[decompose] data-designer runtime error "
            f"({type(exc).__name__}: {exc}); falling back to sequential NIM.",
        )
        return None


# --------------------------------------------------------------------------- #
# public entrypoint
# --------------------------------------------------------------------------- #

def decompose(rows: Iterable[RawInput]) -> list[DecomposeResult]:
    """Return one :class:`DecomposeResult` per row, in input order.

    Primary path: NeMo Data Designer batch. Fallback: unified NIM per row via
    :class:`NvidiaSyncClient`. Both paths emit the same shape.
    """
    rows_list = list(rows)
    if not rows_list:
        return []
    via_dd = _decompose_data_designer(rows_list)
    return via_dd if via_dd is not None else _decompose_sequential(rows_list)
