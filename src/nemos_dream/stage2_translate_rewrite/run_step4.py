"""Internal stage-2 step 4: polish ``step3_korean_dialogue`` into ``final_dialogue``."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field
import pandas as pd

from nemos_dream.io_utils import read_jsonl
from nemos_dream.schemas import Stage2Output
from nemos_dream.stage2_translate_rewrite.pipeline_modes import (
    DEFAULT_PIPELINE_MODE,
    PIPELINE_MODES,
    default_stage2_output_path,
    normalize_pipeline_mode,
    requires_step4,
)
from nemos_dream.stage2_translate_rewrite.persona_retriever import ensure_list, format_persona_prompt_context
from nemos_dream.stage2_translate_rewrite.run_step3 import (
    DEFAULT_ENDPOINT,
    DEFAULT_ENV_FILE,
    DEFAULT_MAX_PARALLEL_REQUESTS,
    DEFAULT_MODEL,
    DEFAULT_MODEL_TIMEOUT,
    DataDesigner,
    RetryableGenerationError,
    append_jsonl,
    build_invalid_output_path,
    build_provider,
    build_retry_errors_output_path,
    collect_row_ids,
    count_jsonl_rows,
    dd,
    load_jsonl_dicts,
    load_environment,
    replace_queue_file,
    resolve_retry_input_path,
    safe_dataset_token,
    to_json_safe,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = REPO_ROOT / "data" / "stage2" / "stage2_rewrite.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "stage2" / "out.jsonl"
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "data" / "stage2" / "artifacts" / "step4"


class KoreanDialogueTurn(BaseModel):
    index: int
    speaker: str = Field(min_length=1)
    text: str = Field(min_length=1)


class RefinedDialoguePayload(BaseModel):
    dialogue: list[KoreanDialogueTurn] = Field(min_length=1)


def clean_text(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def contains_ascii_letters(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text))


def extract_relationship_type(row: dict[str, Any]) -> str:
    scene = row.get("scene") or {}
    return str(scene.get("relationship_type") or "").strip().lower()


def extract_speaker_role_map(row: dict[str, Any]) -> dict[str, str]:
    role_map: dict[str, str] = {}
    for speaker in ensure_list(row.get("speakers")):
        name_en = clean_text(speaker.get("name_en")).lower()
        role_in_scene = clean_text(speaker.get("role_in_scene")).lower()
        if name_en and role_in_scene:
            role_map[name_en] = role_in_scene
    return role_map


def safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def extract_persona_entries(row: dict[str, Any]) -> list[dict[str, Any]]:
    persona_entries: list[dict[str, Any]] = []
    for item in ensure_list(row.get("persona")):
        retrieved = item.get("retrieved_persona") or {}
        source_profile = item.get("source_speaker_profile") or {}
        persona_entries.append(
            {
                "speaker_name_en": clean_text(item.get("speaker_name_en")).lower(),
                "persona_name": clean_text(retrieved.get("name")),
                "age": safe_int(retrieved.get("age")),
                "sex": clean_text(retrieved.get("sex")),
                "role_in_scene": clean_text(source_profile.get("role_in_scene")).lower(),
            }
        )
    return persona_entries


def is_intimate_relationship(relationship_type: str) -> bool:
    return relationship_type in {"friendship", "family", "romantic"}


def is_formal_relationship(relationship_type: str) -> bool:
    return relationship_type in {"professional", "acquaintance"}


def detect_speech_level(text: str) -> str:
    cleaned = clean_text(text)
    if not cleaned:
        return "unknown"

    polite_markers = [
        "습니다",
        "습니까",
        "세요",
        "해요",
        "해요.",
        "이에요",
        "예요",
        "드릴게요",
        "주시겠어요",
        "주실 수",
        "괜찮으세요",
        "하시나요",
        "감사합니다",
        "죄송합니다",
        "알겠습니다",
    ]
    casual_markers = [
        "야",
        "지?",
        "네.",
        "거야",
        "했어",
        "했지",
        "할게",
        "고마워",
        "미안해",
        "괜찮아",
        "몰라",
        "좋아",
        "싫어",
        "맞아",
        "있어?",
        "없어?",
        "거든",
        "해봐",
        "가자",
    ]

    polite_score = sum(marker in cleaned for marker in polite_markers)
    casual_score = sum(marker in cleaned for marker in casual_markers)
    if polite_score and not casual_score:
        return "polite"
    if casual_score and not polite_score:
        return "casual"
    if polite_score and casual_score:
        return "mixed"
    return "unknown"


def contains_excessive_formality_for_intimate(text: str) -> bool:
    cleaned = clean_text(text)
    if not cleaned:
        return False
    markers = [
        "씨",
        "고객님",
        "말씀",
        "감사드립니다",
        "드리겠습니다",
        "드시겠어요",
        "주시겠어요",
        "괜찮으세요",
        "하셨습니다",
        "하실까요",
        "하시나요",
        "해 주셔서 감사합니다",
    ]
    return any(marker in cleaned for marker in markers)


def contains_excessive_casualness_for_formal(text: str) -> bool:
    cleaned = clean_text(text)
    if not cleaned:
        return False
    markers = [
        "야",
        "너는",
        "너가",
        "네가",
        "니가",
        "가자",
        "좋아?",
        "뭐해",
        "왜 그래",
        "괜찮아?",
    ]
    return any(marker in cleaned for marker in markers)


def has_scene_role_persona_conflict(row: dict[str, Any]) -> bool:
    role_map = extract_speaker_role_map(row)
    persona_entries = extract_persona_entries(row)
    if len(persona_entries) < 2:
        return False

    ages_by_role: dict[str, list[int]] = {}
    for entry in persona_entries:
        age = entry.get("age")
        speaker_name_en = entry.get("speaker_name_en", "")
        role = entry.get("role_in_scene") or role_map.get(speaker_name_en, "")
        if role and age is not None:
            ages_by_role.setdefault(role, []).append(age)

    parents = ages_by_role.get("parent", [])
    children = ages_by_role.get("child", [])
    teachers = ages_by_role.get("teacher", [])
    students = ages_by_role.get("student", [])
    bosses = ages_by_role.get("boss", [])
    subordinates = ages_by_role.get("subordinate", [])

    if parents and children and min(parents) <= max(children):
        return True
    if teachers and students and min(teachers) <= max(students):
        return True
    if bosses and subordinates and min(bosses) + 5 < max(subordinates):
        return True
    return False


def dialogue_has_mixed_speech_levels(dialogue: list[dict[str, Any]]) -> bool:
    levels_by_speaker: dict[str, set[str]] = {}
    for turn in dialogue:
        speaker = clean_text(turn.get("speaker"))
        level = detect_speech_level(turn.get("text"))
        if not speaker or level in {"unknown", "mixed"}:
            continue
        levels_by_speaker.setdefault(speaker, set()).add(level)
    return any(len(levels) >= 2 for levels in levels_by_speaker.values())


def should_fallback_to_step3(
    row: dict[str, Any],
    refined_dialogue: list[dict[str, Any]],
) -> bool:
    relationship_type = extract_relationship_type(row)
    if dialogue_has_mixed_speech_levels(refined_dialogue):
        return True

    if has_scene_role_persona_conflict(row):
        return True

    for turn in refined_dialogue:
        text = clean_text(turn.get("text"))
        if is_intimate_relationship(relationship_type) and contains_excessive_formality_for_intimate(text):
            return True
        if is_formal_relationship(relationship_type) and contains_excessive_casualness_for_formal(text):
            return True
    return False


def format_source_dialogue_prompt(source_dialogue: list[dict[str, Any]]) -> str:
    if not source_dialogue:
        return "source_dialogue 없음"
    blocks: list[str] = []
    for turn in source_dialogue:
        blocks.append(
            "\n".join(
                [
                    f"[Turn {turn.get('index')}]",
                    f"- speaker: {turn.get('speaker')}",
                    f"- text: {clean_text(turn.get('text'))}",
                ]
            )
        )
    return "\n\n".join(blocks)


def format_step3_dialogue_prompt(step3_dialogue: list[dict[str, Any]]) -> str:
    if not step3_dialogue:
        return "step3_korean_dialogue 없음"
    blocks: list[str] = []
    for turn in step3_dialogue:
        blocks.append(
            "\n".join(
                [
                    f"[Turn {turn.get('index')}]",
                    f"- speaker: {turn.get('speaker')}",
                    f"- text: {clean_text(turn.get('text'))}",
                ]
            )
        )
    return "\n\n".join(blocks)


def format_mapped_refs_prompt(mapped_refs: list[dict[str, Any]]) -> str:
    if not mapped_refs:
        return "mapped_refs 없음"
    lines: list[str] = []
    for item in mapped_refs:
        lines.append(
            (
                f"- {item.get('term', '')} -> {item.get('ko', '')} "
                f"(type={item.get('type', '')}, notes={item.get('notes', '')})"
            )
        )
    return "\n".join(lines)


def format_scene_prompt(row: dict[str, Any]) -> str:
    scene = row.get("scene") or {}
    decomposed = row.get("dialogue_decomposed") or {}
    overall_emotion = decomposed.get("overall_emotion") or {}
    return "\n".join(
        [
            f"- narrative_en: {scene.get('narrative_en', '')}",
            f"- setting: {scene.get('setting', '')}",
            f"- relationship_type: {scene.get('relationship_type', '')}",
            f"- topics: {scene.get('topics', [])}",
            f"- overall_register: {decomposed.get('overall_register', '')}",
            f"- overall_emotion: {overall_emotion.get('type', '')} ({overall_emotion.get('intensity', '')})",
            f"- speech_acts: {decomposed.get('speech_acts', [])}",
        ]
    )


def format_speaker_profile_prompt(
    speakers: list[dict[str, Any]],
    personas: list[dict[str, Any]],
) -> str:
    if not speakers and not personas:
        return "speaker metadata 없음"

    persona_by_index: dict[int, dict[str, Any]] = {}
    for item in ensure_list(personas):
        speaker_index = safe_int(item.get("speaker_index"))
        if speaker_index is not None:
            persona_by_index[speaker_index] = item

    blocks: list[str] = []
    total = max(len(ensure_list(speakers)), len(ensure_list(personas)))
    for index in range(total):
        speaker = speakers[index] if index < len(speakers) else {}
        persona_item = persona_by_index.get(index, {})
        retrieved = persona_item.get("retrieved_persona") or {}
        emotion = speaker.get("dominant_emotion") or {}
        blocks.append(
            "\n".join(
                [
                    f"[Speaker {index}] {speaker.get('name_en', persona_item.get('speaker_name_en', ''))}",
                    f"- role_in_scene: {speaker.get('role_in_scene', '')}",
                    f"- register: {speaker.get('register', '')}",
                    f"- dominant_emotion: {emotion.get('type', '')} ({emotion.get('intensity', '')})",
                    f"- speech_style_notes: {speaker.get('speech_style_notes', '')}",
                    f"- personality_traits: {speaker.get('personality_traits', [])}",
                    f"- interests_hints: {speaker.get('interests_hints', [])}",
                    f"- gender_hint: {speaker.get('gender_hint', '') or 'unknown'}",
                    f"- age_group_hint: {speaker.get('age_group_hint', '') or 'unknown'}",
                    f"- occupation_hint: {speaker.get('occupation_hint', '') or '없음'}",
                    (
                        f"- selected persona: {retrieved.get('name', '')} / {retrieved.get('sex', '')} / "
                        f"{retrieved.get('age', '')}세 / {retrieved.get('occupation', '') or '직업정보없음'} / "
                        f"{retrieved.get('normalized_location', '')}"
                    ),
                    f"- persona_summary: {retrieved.get('persona', '')}",
                    f"- cultural_background: {retrieved.get('cultural_background', '')}",
                    f"- skills_and_expertise: {retrieved.get('skills_and_expertise', '')}",
                    f"- hobbies_and_interests: {retrieved.get('hobbies_and_interests', '')}",
                ]
            )
        )
    return "\n\n".join(blocks)


def format_quality_prompt_context(step3_dialogue: list[dict[str, Any]]) -> str:
    notes: list[str] = []
    for turn in step3_dialogue:
        index = turn.get("index")
        speaker = turn.get("speaker")
        raw_text = str(turn.get("text") or "")
        cleaned = clean_text(raw_text)
        turn_notes: list[str] = []
        if re.search(r"[A-Za-z]", cleaned):
            turn_notes.append("영문 잔존 여부 확인 필요")
        if any(quote in raw_text for quote in ['"', "'", "“", "”"]):
            turn_notes.append("따옴표/직역 흔적 점검")
        if len(cleaned) >= 120:
            turn_notes.append("한 턴이 길어 구어체 호흡 점검")
        if re.search(r"\s{2,}", raw_text):
            turn_notes.append("공백 정리 필요")
        if cleaned.endswith((":", ";", ",")):
            turn_notes.append("문장 종결 표현 점검")
        if turn_notes:
            notes.append(f"- Turn {index} / {speaker}: {', '.join(turn_notes)}")

    if not notes:
        return (
            "표면적으로 큰 오류는 적어 보입니다. 그래도 직역투, 어색한 조사/어미, "
            "호칭, 높임법, 구어체 리듬을 다시 점검하세요."
        )
    return "\n".join(notes)


def format_relationship_guidance(row: dict[str, Any]) -> str:
    scene = row.get("scene") or {}
    relationship_type = str(scene.get("relationship_type") or "").strip().lower()
    overall_register = str((row.get("dialogue_decomposed") or {}).get("overall_register") or "").strip().lower()
    personas = ensure_list(row.get("persona"))

    guidance: list[str] = []
    if overall_register == "formal":
        guidance.append("전체적으로 예의를 갖춘 존댓말을 기본으로 두세요.")

    if relationship_type == "acquaintance":
        guidance.append("아직 충분히 가까운 사이가 아니므로 과도한 친밀감보다 자연스러운 거리감을 유지하세요.")
    elif relationship_type == "professional":
        guidance.append("직업적 역할 차이와 장면의 위계를 반영해 호칭과 높임 표현을 안정적으로 맞추세요.")
    elif relationship_type == "friendship":
        guidance.append("친한 사이면 불필요한 존댓말, 과도한 격식, 딱딱한 호칭을 피하고 자연스러운 반말/해요체/친근한 호칭을 선택하세요.")
    else:
        guidance.append("관계가 모호하면 지나치게 튀는 말투보다 무난하고 자연스러운 한국어를 우선하세요.")

    ages: list[int] = []
    for item in personas:
        retrieved = item.get("retrieved_persona") or {}
        age_value = safe_int(retrieved.get("age"))
        if age_value is not None:
            ages.append(age_value)
    if len(ages) >= 2 and max(ages) - min(ages) >= 12:
        guidance.append("화자 간 나이 차가 큰 편이면 한국어 화용론에 맞는 호칭과 높임 차이를 자연스럽게 반영하세요.")

    if not guidance:
        guidance.append("관계, register, 나이 차, 직업적 역할을 함께 보고 적절한 높임법을 선택하세요.")
    return "\n".join(f"- {item}" for item in guidance)


def format_consistency_prompt_context(row: dict[str, Any]) -> str:
    scene = row.get("scene") or {}
    relationship_type = str(scene.get("relationship_type") or "").strip().lower()
    personas = ensure_list(row.get("persona"))

    lines: list[str] = [
        "- 가장 중요한 목표는 말투 일관성입니다. 한 번 정한 호칭, 높임 수준, 말끝을 턴마다 흔들지 마세요.",
        "- 각 화자의 persona상 나이, 성별, 관계를 먼저 해석한 뒤 그에 맞는 한국어 화법을 정하고 끝까지 유지하세요.",
        "- 친구/연인/가족처럼 친밀한 관계인데 '씨', '고객님', 과한 존댓말, 과한 설명체가 섞이지 않게 하세요.",
        "- 직장/서비스/면접처럼 격식이 필요한 관계인데 반말이나 지나치게 사적인 호칭이 섞이지 않게 하세요.",
        "- 한 화자가 상대를 부르는 호칭은 대화 전체에서 가능한 한 하나의 체계로 유지하세요.",
        "- 문장마다 번역투 표현을 새로 덧입히지 말고, 실제 한국인이 바로 말할 법한 짧고 자연스러운 구어체 리듬을 우선하세요.",
        "- persona는 배경 장식이 아니라 말투 선택의 핵심 근거입니다. 세대감, 성별, 생활 배경이 드러나는 화법을 우선 고려하세요.",
        "- persona와 scene role이 충돌해 보일 때는 persona를 그대로 직역하지 말고, 현재 장면의 관계와 role_in_scene이 먼저 자연스럽게 유지되도록 조정하세요.",
        "- 특히 family, friendship에서는 상대를 남처럼 부르거나 존댓말/반말이 턴마다 뒤집히지 않게 하세요.",
    ]

    if relationship_type == "friendship":
        lines.append("- friendship이면 특별한 근거 없이 서로 지나치게 높이거나 딱딱하게 대하지 마세요.")
    if relationship_type == "family":
        lines.append("- family이면 가족 관계에 맞는 호칭과 자연스러운 친밀도를 유지하고, 서로를 어색하게 높이지 마세요.")
    if relationship_type == "romantic":
        lines.append("- romantic이면 지나치게 사무적이거나 거리감 있는 호칭을 피하고 친밀한 말투를 유지하세요.")
    if relationship_type == "professional":
        lines.append("- professional이면 사회적 거리와 역할을 반영한 존댓말 체계를 유지하세요.")

    if len(personas) >= 2:
        age_descriptions: list[str] = []
        for item in personas:
            retrieved = item.get("retrieved_persona") or {}
            name = retrieved.get("name", "")
            age = retrieved.get("age", "")
            sex = retrieved.get("sex", "")
            if name or age or sex:
                age_descriptions.append(f"{name}({age}세, {sex})")
        if age_descriptions:
            lines.append("- 화자 persona 참고: " + ", ".join(age_descriptions))

    return "\n".join(lines)


def build_seed_dataframe(row: Stage2Output) -> pd.DataFrame:
    payload = row.model_dump()
    source_dialogue = ensure_list(payload.get("source_dialogue"))
    step3_korean_dialogue = ensure_list(payload.get("step3_korean_dialogue"))
    speakers = ensure_list(payload.get("speakers"))
    personas = ensure_list(payload.get("persona"))
    mapped_refs = ensure_list(payload.get("mapped_refs"))

    seed_rows = [
        {
            "source_id": str(payload.get("id") or "row"),
            "source_dialogue": source_dialogue,
            "step3_korean_dialogue": step3_korean_dialogue,
            "speakers": speakers,
            "persona": personas,
            "mapped_refs": mapped_refs,
            "source_turn_count": len(step3_korean_dialogue),
            "source_dialogue_prompt_context": format_source_dialogue_prompt(source_dialogue),
            "step3_dialogue_prompt_context": format_step3_dialogue_prompt(step3_korean_dialogue),
            "speaker_profile_prompt_context": format_speaker_profile_prompt(speakers=speakers, personas=personas),
            "persona_prompt_context": format_persona_prompt_context(personas),
            "mapped_refs_prompt_context": format_mapped_refs_prompt(mapped_refs),
            "scene_prompt_context": format_scene_prompt(payload),
            "relationship_guidance_prompt_context": format_relationship_guidance(payload),
            "consistency_prompt_context": format_consistency_prompt_context(payload),
            "quality_prompt_context": format_quality_prompt_context(step3_korean_dialogue),
        }
    ]
    return pd.DataFrame(seed_rows)


def build_model_configs(model_name: str) -> list[dd.ModelConfig]:
    return [
        dd.ModelConfig(
            alias="step4-dialogue-polisher",
            model=model_name,
            provider="nvidia-cloud",
            inference_parameters=dd.ChatCompletionInferenceParams(
                temperature=0.25,
                top_p=0.9,
                max_tokens=8192,
                timeout=DEFAULT_MODEL_TIMEOUT,
                max_parallel_requests=DEFAULT_MAX_PARALLEL_REQUESTS,
            ),
        )
    ]


def build_config_builder(seed_df: pd.DataFrame, model_name: str) -> dd.DataDesignerConfigBuilder:
    builder = dd.DataDesignerConfigBuilder(model_configs=build_model_configs(model_name))
    builder.with_seed_dataset(dd.DataFrameSeedSource(df=seed_df))
    builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="step4_refined_dialogue",
            model_alias="step4-dialogue-polisher",
            system_prompt=(
                "당신은 한국어 대화 데이터셋 품질 개선 전문가입니다. "
                "이미 한국어로 번역된 멀티턴 대화를, 실제 한국인이 말한 것처럼 더 자연스럽고 일관되게 다듬으세요. "
                "화자별 dominant_emotion, speech_style_notes, register, 관계, 나이, 성별, persona를 반영해 "
                "호칭, 높임법, 어휘, 문장 호흡을 조정하되 대화의 핵심 의미와 턴 구조는 유지하세요. "
                "특히 말투 일관성, 관계에 따른 호칭 일관성, persona 기반 화법 유지가 가장 중요합니다. "
                "친밀한 관계인데 과하게 격식을 차리거나, 격식 있는 관계인데 갑자기 반말이 섞이면 안 됩니다. "
                "persona 정보가 scene role과 충돌해 보이더라도, 대화 안에서는 현재 관계와 role_in_scene이 먼저 자연스럽게 유지되어야 합니다. "
                "번역투보다 자연스러운 한국어 구어체를 우선하세요. "
                "반드시 구조화된 결과만 반환하세요."
            ),
            prompt="""\
현재 한국어 대화를 실제 한국 화자들이 말하는 것처럼 더 자연스럽게 다듬으세요.

[current_step3_korean_dialogue]
{{ step3_dialogue_prompt_context }}

[original_source_dialogue]
{{ source_dialogue_prompt_context }}

[scene]
{{ scene_prompt_context }}

[speaker metadata]
{{ speaker_profile_prompt_context }}

[selected personas]
{{ persona_prompt_context }}

[mapped_refs]
{{ mapped_refs_prompt_context }}

[relationship and style guidance]
{{ relationship_guidance_prompt_context }}

[consistency checklist]
{{ consistency_prompt_context }}

[language quality review]
{{ quality_prompt_context }}

[출력해야 하는 turn 수]
정확히 {{ source_turn_count }}개 turn을 출력하세요.

[반드시 지킬 규칙]
1. step3_korean_dialogue의 턴 개수와 순서를 유지하세요.
2. 각 turn은 index, speaker, text를 모두 포함해야 합니다.
3. speaker 이름은 현재 step3_korean_dialogue의 한국 이름을 그대로 유지하세요.
4. step3_korean_dialogue를 기반으로 자연화하되, 어색하면 문장 전체를 다시 써도 됩니다.
5. dominant_emotion, speech_style_notes, register, relationship_type, age, gender, occupation, persona를 참고해 실제 한국어 화용에 맞는 호칭과 높임 표현을 선택하세요.
6. 화자 간 거리감, 나이 차, 직업적 위계가 느껴지면 그 차이가 말투에 자연스럽게 드러나야 합니다. 친근한 사이인 경우 해당 스타일을 반영하세요.
7. 한번 정한 호칭, 높임 수준, 반말/존댓말 체계, 말끝 패턴은 대화 전체에서 일관되게 유지하세요.
8. friendship, family, romantic처럼 친밀한 관계에서는 특별한 근거 없이 과도하게 격식을 차리지 마세요.
9. professional, acquaintance처럼 거리감이 필요한 관계에서는 갑작스러운 반말, 과한 친근 호칭, 어색한 사적 말투를 쓰지 마세요.
10. persona의 나이와 성별, 생활 배경을 말투 결정의 핵심 근거로 사용하세요. persona와 어긋나는 세대감 없는 화법은 피하세요.
11. persona가 scene role과 충돌하더라도 관계_type과 role_in_scene을 무너뜨리지 마세요. 예를 들어 family인데 서로를 남처럼 부르거나, friendship인데 과한 직장식 존댓말을 쓰지 마세요.
12. 한국어 번역투를 줄이고, 실제 한국인이 바로 말할 법한 자연스러운 구어체로 다듬으세요.
13. 핵심 의미, 정보, 질문-응답 관계, 장면의 흐름은 유지하세요. 근거 없는 새 설정이나 배경 설명을 추가하지 마세요.
14. persona 정보는 말투와 시선, 세대감, 어휘 선택에 은은하게 반영하되, 대화 내용에 없는 개인사나 관계 재설정을 새로 꺼내지 마세요.
15. 한국어 구어체 기준으로 직역투, 번역체, 어색한 조사/어미, 부자연스러운 반복, 문어체 과잉, 어색한 높임법을 고치세요.
16. 비한국어 표현, 영문 잔존, 어색한 구두점, 잘린 문장, 부자연스러운 호흡이 있으면 자연스럽게 수정하세요.
17. 이미 자연스러운 turn은 수정하지 않아도 됩니다.
18. turn을 합치거나 쪼개지 말고 1:1 대응을 유지하세요.
19. 설명문, 주석, 마크다운 코드블록 없이 오직 구조화된 결과만 반환하세요.
20. 대화 내부에서 대화의 흐름과 관계가 일관되게 유지되어야 합니다.
21. 마지막으로 출력 전 turn 수가 정확히 {{ source_turn_count }}개인지 스스로 확인하세요.
""",
            output_format=RefinedDialoguePayload,
        )
    )
    return builder


def run_data_designer(
    api_key: str,
    seed_df: pd.DataFrame,
    model_name: str,
    endpoint: str,
    artifact_dir: str | Path,
    mode: str,
    dataset_name: str,
) -> pd.DataFrame:
    data_designer = DataDesigner(
        model_providers=[build_provider(api_key=api_key, endpoint=endpoint)],
        artifact_path=str(artifact_dir),
    )
    builder = build_config_builder(seed_df=seed_df, model_name=model_name)
    data_designer.validate(builder)
    if mode == "preview":
        preview = data_designer.preview(config_builder=builder, num_records=len(seed_df))
        return preview.dataset.copy()

    creation_results = data_designer.create(
        config_builder=builder,
        num_records=len(seed_df),
        dataset_name=dataset_name,
    )
    return creation_results.load_dataset().copy()


def parse_json_like(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
    return value


def extract_generated_turns(value: Any) -> list[dict[str, Any]]:
    parsed = parse_json_like(value)
    if isinstance(parsed, dict):
        if "dialogue" in parsed and isinstance(parsed["dialogue"], list):
            turns = parsed["dialogue"]
        elif "turns" in parsed and isinstance(parsed["turns"], list):
            turns = parsed["turns"]
        else:
            raise ValueError("Structured dialogue output does not contain 'dialogue' or 'turns'.")
    elif isinstance(parsed, list):
        turns = parsed
    else:
        raise ValueError(f"Unexpected dialogue payload type: {type(parsed)!r}")
    return [to_json_safe(item) for item in turns]


def normalize_refined_dialogue(
    original_row: dict[str, Any],
    step3_korean_dialogue: list[dict[str, Any]],
    generated_value: Any,
) -> list[dict[str, Any]]:
    generated_turns = extract_generated_turns(generated_value)
    if len(generated_turns) != len(step3_korean_dialogue):
        raise ValueError(
            f"Generated turn count mismatch. expected={len(step3_korean_dialogue)} actual={len(generated_turns)}"
        )

    normalized_turns: list[dict[str, Any]] = []
    for source_turn, generated_turn in zip(step3_korean_dialogue, generated_turns, strict=True):
        text = clean_text((generated_turn or {}).get("text"))
        if not text:
            raise ValueError("Generated turn text is empty.")
        if contains_ascii_letters(text):
            fallback_text = clean_text(source_turn.get("text"))
            if fallback_text and not contains_ascii_letters(fallback_text):
                text = fallback_text

        source_speaker = clean_text(source_turn.get("speaker"))
        if not source_speaker:
            source_speaker = clean_text((generated_turn or {}).get("speaker"))
        if not source_speaker:
            raise ValueError("Speaker name is empty after normalization.")

        normalized_turns.append(
            {
                "index": source_turn.get("index"),
                "speaker": source_speaker,
                "text": text,
            }
        )

    if should_fallback_to_step3(original_row, normalized_turns):
        return [
            {
                "index": turn.get("index"),
                "speaker": clean_text(turn.get("speaker")),
                "text": clean_text(turn.get("text")),
            }
            for turn in step3_korean_dialogue
        ]
    return normalized_turns


def build_final_dialogue_payload(refined_dialogue: list[dict[str, Any]]) -> list[dict[str, Any]]:
    final_dialogue = [to_json_safe(turn) for turn in refined_dialogue if turn]
    if not final_dialogue:
        raise ValueError("Final dialogue payload is empty.")
    return final_dialogue


def mirror_step3_to_final(row: Stage2Output, pipeline_mode: str) -> Stage2Output:
    normalized_mode = normalize_pipeline_mode(pipeline_mode)
    step3_dialogue = [
        {
            "index": turn.get("index"),
            "speaker": clean_text(turn.get("speaker")),
            "text": clean_text(turn.get("text")),
        }
        for turn in ensure_list(row.model_dump().get("step3_korean_dialogue"))
        if turn
    ]
    if not step3_dialogue:
        raise ValueError("step3_korean_dialogue is empty; cannot mirror to final_dialogue.")

    payload = row.model_dump()
    payload["final_dialogue"] = step3_dialogue
    translation_meta = dict(payload.get("translation_meta") or {})
    translation_meta["pipeline_mode"] = normalized_mode
    translation_meta["step4_applied"] = False
    payload["translation_meta"] = translation_meta
    return Stage2Output.model_validate(payload)


def run_single_row(
    row: Stage2Output,
    *,
    api_key: str,
    model_name: str,
    endpoint: str,
    artifact_root: str | Path,
    mode: str,
    dataset_name_prefix: str,
    pipeline_mode: str,
) -> Stage2Output:
    normalized_mode = normalize_pipeline_mode(pipeline_mode)
    if not requires_step4(normalized_mode):
        return mirror_step3_to_final(row, normalized_mode)

    seed_df = build_seed_dataframe(row)
    artifact_dir = Path(artifact_root) / safe_dataset_token(row.id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = (
        f"{dataset_name_prefix}-{safe_dataset_token(row.id)}-"
        f"{datetime.now().strftime('%Y%m%d%H%M%S')}"
    )
    try:
        result_df = run_data_designer(
            api_key=api_key,
            seed_df=seed_df,
            model_name=model_name,
            endpoint=endpoint,
            artifact_dir=artifact_dir,
            mode=mode,
            dataset_name=dataset_name,
        )
    except Exception as exc:
        raise RetryableGenerationError(str(exc)) from exc

    if result_df.empty:
        raise RetryableGenerationError("Data Designer returned an empty dataframe.")

    generated_row = to_json_safe(result_df.iloc[0].to_dict())
    refined_dialogue = normalize_refined_dialogue(
        original_row=row.model_dump(),
        step3_korean_dialogue=ensure_list(row.model_dump().get("step3_korean_dialogue")),
        generated_value=generated_row.get("step4_refined_dialogue"),
    )
    final_dialogue = build_final_dialogue_payload(refined_dialogue)
    payload = row.model_dump()
    payload["final_dialogue"] = final_dialogue
    translation_meta = dict(payload.get("translation_meta") or {})
    translation_meta["pipeline_mode"] = normalized_mode
    translation_meta["step4_applied"] = True
    payload["translation_meta"] = translation_meta
    return Stage2Output.model_validate(payload)


def run(
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path | None = None,
    *,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    env_file: str | Path | None = DEFAULT_ENV_FILE,
    model: str = DEFAULT_MODEL,
    endpoint: str = DEFAULT_ENDPOINT,
    mode: str = "create",
    pipeline_mode: str = DEFAULT_PIPELINE_MODE,
    num_records: int | None = None,
    dataset_name: str = "stage2-step4-polish",
    retry_errors_from: str | Path | None = None,
    retry_source: str = "auto",
    resume: bool = True,
) -> Path:
    normalized_mode = normalize_pipeline_mode(pipeline_mode)
    api_key = load_environment(env_file) if requires_step4(normalized_mode) else ""
    in_retry_mode = retry_errors_from is not None
    resolved_output_path = Path(output_path) if output_path else (
        Path(retry_errors_from) if retry_errors_from else default_stage2_output_path(REPO_ROOT / "data" / "stage2", normalized_mode)
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    retry_errors_path = build_retry_errors_output_path(resolved_output_path)
    invalid_path = build_invalid_output_path(resolved_output_path)
    resolved_input_path = (
        resolve_retry_input_path(retry_errors_from, retry_source)
        if retry_errors_from
        else Path(input_path)
    )

    input_rows = list(read_jsonl(resolved_input_path, Stage2Output))
    if num_records is not None:
        input_rows = input_rows[:num_records]

    if not in_retry_mode and not resume:
        for path in [resolved_output_path, retry_errors_path, invalid_path]:
            if path.exists():
                path.unlink()

    processed_ids = collect_row_ids(resolved_output_path)
    if not in_retry_mode and resume:
        processed_ids |= collect_row_ids(retry_errors_path)
        processed_ids |= collect_row_ids(invalid_path)

    next_retry_path = retry_errors_path.with_name(f"{retry_errors_path.name}.next")
    next_invalid_path = invalid_path.with_name(f"{invalid_path.name}.next")
    if in_retry_mode:
        for path in [next_retry_path, next_invalid_path]:
            if path.exists():
                path.unlink()

    saved = 0
    retry_errors = 0
    invalid = 0
    for row in input_rows:
        if row.id in processed_ids:
            continue
        try:
            generated_row = run_single_row(
                row,
                api_key=api_key,
                model_name=model,
                endpoint=endpoint,
                artifact_root=artifact_dir,
                mode=mode,
                dataset_name_prefix=dataset_name,
                pipeline_mode=normalized_mode,
            )
        except RetryableGenerationError as exc:
            error_row = row.model_dump()
            error_row["step4_status"] = "retry_error"
            error_row["step4_error"] = str(exc)
            append_jsonl(next_retry_path if in_retry_mode else retry_errors_path, error_row)
            retry_errors += 1
            continue
        except Exception as exc:
            error_row = row.model_dump()
            error_row["step4_status"] = "invalid_generation"
            error_row["step4_error"] = str(exc)
            append_jsonl(next_invalid_path if in_retry_mode else invalid_path, error_row)
            invalid += 1
            continue

        append_jsonl(resolved_output_path, generated_row)
        saved += 1

    if in_retry_mode:
        replace_queue_file(retry_errors_path, load_jsonl_dicts(next_retry_path))
        replace_queue_file(invalid_path, load_jsonl_dicts(next_invalid_path))
        for path in [next_retry_path, next_invalid_path]:
            if path.exists():
                path.unlink()

    print(
        "Stage2 step4 summary: "
        f"saved={saved} retry_errors={retry_errors} invalid={invalid} "
        f"total_output_rows={count_jsonl_rows(resolved_output_path)}"
    )
    return resolved_output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run stage2 internal step4 (naturalize step3_korean_dialogue into final_dialogue)."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input Stage2 intermediate JSONL path.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output Stage2 JSONL path. Defaults to a mode-specific file under data/stage2/.",
    )
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR), help="Directory for Data Designer artifacts.")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE), help="Path to .env containing NVIDIA_API_KEY.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="NVIDIA model name.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="NVIDIA OpenAI-compatible endpoint.")
    parser.add_argument("--mode", choices=["preview", "create"], default="create", help="Use Data Designer preview or create mode.")
    parser.add_argument(
        "--pipeline-mode",
        choices=PIPELINE_MODES,
        default=DEFAULT_PIPELINE_MODE,
        help="Stage-2 finalization variant: default | direct | naive_persona.",
    )
    parser.add_argument("--num-records", type=int, default=None, help="Optionally limit the number of input rows.")
    parser.add_argument("--dataset-name", default="stage2-step4-polish", help="Dataset name prefix for Data Designer artifacts.")
    parser.add_argument(
        "--retry-errors-from",
        default=None,
        help="Base output JSONL path. If set, rerun from '<base>.retry-errors.jsonl' or '<base>.invalid.jsonl'.",
    )
    parser.add_argument(
        "--retry-source",
        choices=["auto", "retry-errors", "invalid"],
        default="auto",
        help="Which queue to consume when --retry-errors-from is set.",
    )
    parser.add_argument("--resume", dest="resume", action="store_true", help="Resume from existing output and queue files.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Ignore existing output and start a fresh run.")
    parser.set_defaults(resume=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    resolved_output = args.output or str(
        default_stage2_output_path(REPO_ROOT / "data" / "stage2", args.pipeline_mode)
    )
    output_path = run(
        input_path=args.input,
        output_path=resolved_output,
        artifact_dir=args.artifact_dir,
        env_file=args.env_file,
        model=args.model,
        endpoint=args.endpoint,
        mode=args.mode,
        pipeline_mode=args.pipeline_mode,
        num_records=args.num_records,
        dataset_name=args.dataset_name,
        retry_errors_from=args.retry_errors_from,
        retry_source=args.retry_source,
        resume=args.resume,
    )
    print(f"Saved stage2 step4 output to {output_path}")


if __name__ == "__main__":
    main()
