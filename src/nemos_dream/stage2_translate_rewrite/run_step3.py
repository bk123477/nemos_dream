"""Internal stage-2 step 3: persona retrieval + first-pass Korean rewrite."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from nemos_dream.io_utils import read_jsonl
from nemos_dream.schemas import Stage1Output, Stage2Output
from nemos_dream.stage2_translate_rewrite.persona_retriever import (
    PersonaRetriever,
    ensure_list,
    format_persona_prompt_context,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def bootstrap_local_data_designer() -> None:
    packages_root = REPO_ROOT / "DataDesigner" / "packages"
    for relative_path in [
        "data-designer-config/src",
        "data-designer-engine/src",
        "data-designer/src",
    ]:
        candidate = packages_root / relative_path
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


try:
    import data_designer.config as dd
    from data_designer.interface import DataDesigner
except ImportError:  # pragma: no cover - local fallback
    bootstrap_local_data_designer()
    import data_designer.config as dd
    from data_designer.interface import DataDesigner

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - fallback for lightweight environments
    def load_dotenv(dotenv_path: str | os.PathLike[str] | None = None, override: bool = False) -> bool:
        path = Path(dotenv_path) if dotenv_path is not None else REPO_ROOT / ".env"
        if not path.exists():
            return False
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if override or key not in os.environ:
                os.environ[key] = value
        return True


DEFAULT_INPUT = REPO_ROOT / "data" / "stage1" / "out.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "stage2" / "stage2_rewrite.jsonl"
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "data" / "stage2" / "artifacts" / "step3"
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
DEFAULT_PERSONA_DIR = REPO_ROOT / "data" / "persona_age_gender"
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"
DEFAULT_ENDPOINT = "https://integrate.api.nvidia.com/v1"
DEFAULT_SEED = 7
DEFAULT_MODEL_TIMEOUT = 180
DEFAULT_MAX_PARALLEL_REQUESTS = 1


class RetryableGenerationError(RuntimeError):
    """Transient or infrastructure-like failure that belongs in retry-errors."""


class PersonaRetrievalParams(BaseModel):
    persona_dir: str
    base_seed: int = DEFAULT_SEED


class KoreanDialogueTurn(BaseModel):
    index: int
    speaker: str
    text: str = Field(min_length=1)


class KoreanDialoguePayload(BaseModel):
    dialogue: list[KoreanDialogueTurn] = Field(min_length=1)


@lru_cache(maxsize=4)
def get_persona_retriever(persona_dir: str) -> PersonaRetriever:
    return PersonaRetriever(persona_dir=persona_dir)


@dd.custom_column_generator(
    required_columns=["source_id", "speakers"],
    side_effect_columns=["persona_prompt_context"],
)
def retrieve_persona_column(
    row: dict[str, Any],
    generator_params: PersonaRetrievalParams,
) -> dict[str, Any]:
    retriever = get_persona_retriever(generator_params.persona_dir)
    speakers = ensure_list(row.get("speakers"))
    personas = retriever.select_many(
        speakers=speakers,
        record_id=str(row.get("source_id") or ""),
        base_seed=generator_params.base_seed,
    )
    if not personas:
        personas = retriever.sample_random_personas_for_speakers(
            speakers=speakers,
            record_id=str(row.get("source_id") or ""),
            base_seed=generator_params.base_seed,
        )
    row["persona"] = personas
    row["persona_prompt_context"] = format_persona_prompt_context(personas)
    return row


def load_environment(env_file: str | Path | None) -> str:
    candidates: list[Path] = []
    if env_file:
        candidates.append(Path(env_file))
    candidates.append(DEFAULT_ENV_FILE)
    for candidate in candidates:
        if candidate.exists():
            load_dotenv(candidate, override=False)
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        searched = ", ".join(str(path) for path in candidates)
        raise RuntimeError(f"NVIDIA_API_KEY is not set. Checked env files: {searched}")
    return api_key


def to_json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): to_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return to_json_safe(value.model_dump())
    if hasattr(value, "tolist"):
        return to_json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return to_json_safe(value.item())
        except Exception:
            pass
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def append_jsonl(path: str | Path, row: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = row.model_dump_json() if hasattr(row, "model_dump_json") else json.dumps(to_json_safe(row), ensure_ascii=False)
    with output_path.open("a", encoding="utf-8") as file:
        file.write(payload + "\n")


def overwrite_jsonl(path: str | Path, rows: list[Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            payload = row.model_dump_json() if hasattr(row, "model_dump_json") else json.dumps(to_json_safe(row), ensure_ascii=False)
            file.write(payload + "\n")


def load_jsonl_dicts(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def replace_queue_file(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    if rows:
        overwrite_jsonl(target, rows)
    elif target.exists():
        target.unlink()


def count_jsonl_rows(path: str | Path) -> int:
    target = Path(path)
    if not target.exists():
        return 0
    with target.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def collect_row_ids(path: str | Path) -> set[str]:
    target = Path(path)
    if not target.exists():
        return set()
    ids: set[str] = set()
    with target.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_id = str(row.get("id") or "").strip()
            if row_id:
                ids.add(row_id)
    return ids


def safe_dataset_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip())
    return token.strip("-") or "row"


def build_retry_errors_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.retry-errors.jsonl")


def build_invalid_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.invalid.jsonl")


def resolve_retry_input_path(base_output_path: str | Path, retry_source: str = "auto") -> Path:
    base_path = Path(base_output_path)
    retry_path = build_retry_errors_output_path(base_path)
    invalid_path = build_invalid_output_path(base_path)

    if retry_source == "retry-errors":
        if retry_path.exists():
            return retry_path
        raise FileNotFoundError(f"Retry queue not found: {retry_path}")

    if retry_source == "invalid":
        if invalid_path.exists():
            return invalid_path
        raise FileNotFoundError(f"Invalid queue not found: {invalid_path}")

    if retry_path.exists():
        return retry_path
    if invalid_path.exists():
        return invalid_path
    raise FileNotFoundError(
        f"No retry input found. Checked: {retry_path} and {invalid_path}"
    )


def format_source_dialogue_prompt(source_dialogue: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for turn in source_dialogue:
        blocks.append(
            "\n".join(
                [
                    f"[Turn {turn.get('index')}]",
                    f"- speaker: {turn.get('speaker')}",
                    f"- text: {turn.get('text')}",
                ]
            )
        )
    return "\n\n".join(blocks)


def format_speaker_prompt(speakers: list[dict[str, Any]]) -> str:
    if not speakers:
        return "speaker metadata 없음"
    blocks: list[str] = []
    for index, speaker in enumerate(speakers):
        emotion = speaker.get("dominant_emotion") or {}
        blocks.append(
            "\n".join(
                [
                    f"[Speaker {index}] {speaker.get('name_en', '')}",
                    f"- role_in_scene: {speaker.get('role_in_scene', '')}",
                    f"- gender_hint: {speaker.get('gender_hint', '') or 'unknown'}",
                    f"- age_group_hint: {speaker.get('age_group_hint', '') or 'unknown'}",
                    f"- register: {speaker.get('register', '')}",
                    f"- dominant_emotion: {emotion.get('type', '')} ({emotion.get('intensity', '')})",
                    f"- personality_traits: {speaker.get('personality_traits', [])}",
                    f"- interests_hints: {speaker.get('interests_hints', [])}",
                    f"- occupation_hint: {speaker.get('occupation_hint', '') or '없음'}",
                    f"- speech_style_notes: {speaker.get('speech_style_notes', '')}",
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


def format_culture_refs_prompt(row: dict[str, Any]) -> str:
    decomposed = row.get("dialogue_decomposed") or {}
    culture_refs = decomposed.get("cultural_refs") or []
    if not culture_refs:
        return "culture_refs 없음"
    lines: list[str] = []
    for item in culture_refs:
        lines.append(f"- {item.get('term', '')} (type={item.get('type', '')})")
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


def build_seed_dataframe(row: Stage1Output) -> pd.DataFrame:
    payload = row.model_dump()
    source_id = str(payload.get("id") or "row")
    source_dialogue = payload.get("source_dialogue") or []
    speakers = payload.get("speakers") or []
    mapped_refs = payload.get("mapped_refs") or []
    seed_rows = [
        {
            "source_id": source_id,
            "source_dialogue": source_dialogue,
            "speakers": speakers,
            "mapped_refs": mapped_refs,
            "source_turn_count": len(source_dialogue),
            "source_dialogue_prompt_context": format_source_dialogue_prompt(source_dialogue),
            "speaker_prompt_context": format_speaker_prompt(speakers),
            "mapped_refs_prompt_context": format_mapped_refs_prompt(mapped_refs),
            "culture_refs_prompt_context": format_culture_refs_prompt(payload),
            "scene_prompt_context": format_scene_prompt(payload),
        }
    ]
    return pd.DataFrame(seed_rows)


def build_provider(api_key: str, endpoint: str) -> dd.ModelProvider:
    return dd.ModelProvider(
        name="nvidia-cloud",
        endpoint=endpoint,
        provider_type="openai",
        api_key=api_key,
    )


def build_model_configs(model_name: str) -> list[dd.ModelConfig]:
    return [
        dd.ModelConfig(
            alias="step3-dialogue-writer",
            model=model_name,
            provider="nvidia-cloud",
            inference_parameters=dd.ChatCompletionInferenceParams(
                temperature=0.35,
                top_p=0.9,
                max_tokens=8192,
                timeout=DEFAULT_MODEL_TIMEOUT,
                max_parallel_requests=DEFAULT_MAX_PARALLEL_REQUESTS,
            ),
        )
    ]


def build_config_builder(
    seed_df: pd.DataFrame,
    model_name: str,
    persona_dir: str | Path,
    base_seed: int,
) -> dd.DataDesignerConfigBuilder:
    builder = dd.DataDesignerConfigBuilder(model_configs=build_model_configs(model_name))
    builder.with_seed_dataset(dd.DataFrameSeedSource(df=seed_df))
    builder.add_column(
        dd.CustomColumnConfig(
            name="persona",
            generator_function=retrieve_persona_column,
            generator_params=PersonaRetrievalParams(
                persona_dir=str(persona_dir),
                base_seed=base_seed,
            ),
        )
    )
    builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="step3_korean_dialogue",
            model_alias="step3-dialogue-writer",
            system_prompt=(
                "당신은 영어 대화를 한국어 대화 데이터셋으로 현지화하는 전문가입니다. "
                "직역을 피하고, 한국 화자가 실제로 말할 법한 자연스러운 대화로 다시 쓰세요. "
                "출력의 모든 text는 완전한 한국어 문장이어야 하며, 영어, 일본어, 중국어, 스페인어, 프랑스어 등 "
                "한국어가 아닌 표현을 섞지 마세요. "
                "각 source turn에 정확히 하나의 한국어 turn을 대응시키고, 턴을 합치거나 생략하지 마세요. "
                "반드시 구조화된 출력만 반환하세요."
            ),
            prompt="""\
영어 대화를 한국어 대화 데이터셋으로 변환하세요.

[source_dialogue]
{{ source_dialogue_prompt_context }}

[scene]
{{ scene_prompt_context }}

[source speaker hints]
{{ speaker_prompt_context }}

[selected korean personas]
{{ persona_prompt_context }}

[mapped_refs]
{{ mapped_refs_prompt_context }}

[culture_refs]
{{ culture_refs_prompt_context }}

[출력해야 하는 turn 수]
정확히 {{ source_turn_count }}개 turn을 출력하세요.

[반드시 지킬 규칙]
1. source_dialogue의 턴 개수와 순서를 유지하세요.
2. 각 turn은 index, speaker, text를 모두 포함해야 합니다.
3. index는 source_dialogue와 같게 유지하고, speaker는 선택된 한국 persona 이름으로 작성하세요.
4. 가장 중요한 우선순위는 다음과 같습니다.
   1) 선택된 persona의 성별, 연령대, 직업, 지역감, 말투
   2) mapped_refs와 culture_refs의 한국식 표현 반영
   3) source_dialogue의 의미 유지
   4) 자연스러운 한국어 구어체 현지화
5. text는 모두 자연스러운 한국어여야 하며, 직역투를 피하세요.
6. text 안에 영어, 일본어, 중국어, 스페인어, 프랑스어, 로마자 감탄사, 원문 단어를 절대 남기지 마세요.
7. 외래어가 필요하더라도 한국어 표기 또는 한국어 표현으로 바꾸세요. 예: tough -> 힘들다, encouragement -> 격려, absolutely -> 정말/분명히.
8. 선택된 persona를 활용해 각 화자의 말투, 세대감, 존댓말/반말, 관심사와 직업적 시각을 자연스럽게 반영하세요.
9. mapped_refs가 있으면 해당 한국어 표현을 우선 사용하세요. culture_refs가 있으면 한국 문화 맥락에서 가장 자연스러운 호칭, 제도, 장소, 종교/행사 표현으로 반영하세요.
10. source speaker hints의 gender_hint, age_group_hint, occupation_hint와 검색된 persona 정보를 함께 참고하되, 최종 대화는 검색된 persona 설정과 충돌하지 않게 작성하세요.
11. scene의 핵심 정보와 대화의 의미는 유지하되, 한국 문화와 대화 습관에 맞게 표현은 자연스럽게 바꿔도 됩니다.
12. 필요하다면 자기소개, 상대 호칭, 관계 호칭은 선택된 한국 persona 이름과 한국식 대화 관습에 맞게 현지화할 수 있습니다.
13. turn을 요약하거나 합치지 말고, source_dialogue의 각 turn에 대해 1:1로 대응되는 한국어 turn을 작성하세요.
14. 마지막으로 출력하기 전에 turn 개수가 정확히 {{ source_turn_count }}개인지 스스로 확인하세요.
15. 설명문, 주석, 마크다운 코드블록 없이 오직 구조화된 결과만 반환하세요.
""",
            output_format=KoreanDialoguePayload,
        )
    )
    return builder


def run_data_designer(
    api_key: str,
    seed_df: pd.DataFrame,
    model_name: str,
    endpoint: str,
    persona_dir: str | Path,
    artifact_dir: str | Path,
    mode: str,
    base_seed: int,
    dataset_name: str,
) -> pd.DataFrame:
    data_designer = DataDesigner(
        model_providers=[build_provider(api_key=api_key, endpoint=endpoint)],
        artifact_path=str(artifact_dir),
    )
    builder = build_config_builder(
        seed_df=seed_df,
        model_name=model_name,
        persona_dir=persona_dir,
        base_seed=base_seed,
    )
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


def extract_persona_list(value: Any) -> list[dict[str, Any]]:
    parsed = parse_json_like(value)
    if parsed is None:
        return []
    if not isinstance(parsed, list):
        raise ValueError(f"Unexpected persona payload type: {type(parsed)!r}")
    return [to_json_safe(item) for item in parsed]


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


def normalize_korean_dialogue(
    source_dialogue: list[dict[str, Any]],
    generated_value: Any,
    personas: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    generated_turns = extract_generated_turns(generated_value)
    if len(generated_turns) != len(source_dialogue):
        raise ValueError(
            f"Generated turn count mismatch. expected={len(source_dialogue)} actual={len(generated_turns)}"
        )

    speaker_name_map: dict[str, str] = {}
    for persona in personas:
        source_profile = persona.get("source_speaker_profile") or {}
        retrieved_persona = persona.get("retrieved_persona") or {}
        source_name = str(source_profile.get("name_en") or "").strip()
        persona_name = str(retrieved_persona.get("name") or "").strip()
        if source_name and persona_name:
            speaker_name_map[source_name] = persona_name

    normalized_turns: list[dict[str, Any]] = []
    for source_turn, generated_turn in zip(source_dialogue, generated_turns, strict=True):
        text = str((generated_turn or {}).get("text") or "").strip()
        if not text:
            raise ValueError("Generated turn text is empty.")
        source_speaker = str(source_turn.get("speaker") or "").strip()
        normalized_turns.append(
            {
                "index": source_turn.get("index"),
                "speaker": speaker_name_map.get(source_speaker, source_speaker),
                "text": text,
            }
        )
    return normalized_turns


def run_single_row(
    row: Stage1Output,
    *,
    api_key: str,
    model_name: str,
    endpoint: str,
    persona_dir: str | Path,
    artifact_root: str | Path,
    mode: str,
    base_seed: int,
    dataset_name_prefix: str,
) -> Stage2Output:
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
            persona_dir=persona_dir,
            artifact_dir=artifact_dir,
            mode=mode,
            base_seed=base_seed,
            dataset_name=dataset_name,
        )
    except Exception as exc:
        raise RetryableGenerationError(str(exc)) from exc

    if result_df.empty:
        raise RetryableGenerationError("Data Designer returned an empty dataframe.")

    generated_row = to_json_safe(result_df.iloc[0].to_dict())
    personas = extract_persona_list(generated_row.get("persona"))
    korean_dialogue = normalize_korean_dialogue(
        source_dialogue=row.model_dump().get("source_dialogue") or [],
        generated_value=generated_row.get("step3_korean_dialogue"),
        personas=personas,
    )
    stage2_payload = row.model_dump()
    stage2_payload["persona"] = personas
    stage2_payload["step3_korean_dialogue"] = korean_dialogue
    return Stage2Output.model_validate(stage2_payload)


def run(
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path | None = None,
    *,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    env_file: str | Path | None = DEFAULT_ENV_FILE,
    persona_dir: str | Path = DEFAULT_PERSONA_DIR,
    model: str = DEFAULT_MODEL,
    endpoint: str = DEFAULT_ENDPOINT,
    seed: int = DEFAULT_SEED,
    mode: str = "create",
    num_records: int | None = None,
    dataset_name: str = "stage2-step3-rewrite",
    retry_errors_from: str | Path | None = None,
    retry_source: str = "auto",
    resume: bool = True,
) -> Path:
    if not Path(persona_dir).exists():
        raise FileNotFoundError(
            f"Persona directory not found: {persona_dir}. "
            "Run `python -m nemos_dream.stage2_translate_rewrite.persona_downloader` first."
        )

    api_key = load_environment(env_file)
    in_retry_mode = retry_errors_from is not None
    resolved_output_path = Path(output_path) if output_path else (
        Path(retry_errors_from) if retry_errors_from else DEFAULT_OUTPUT
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    retry_errors_path = build_retry_errors_output_path(resolved_output_path)
    invalid_path = build_invalid_output_path(resolved_output_path)
    resolved_input_path = (
        resolve_retry_input_path(retry_errors_from, retry_source)
        if retry_errors_from
        else Path(input_path)
    )

    input_rows = list(read_jsonl(resolved_input_path, Stage1Output))
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
                persona_dir=persona_dir,
                artifact_root=artifact_dir,
                mode=mode,
                base_seed=seed,
                dataset_name_prefix=dataset_name,
            )
        except RetryableGenerationError as exc:
            error_row = row.model_dump()
            error_row["step3_status"] = "retry_error"
            error_row["step3_error"] = str(exc)
            append_jsonl(next_retry_path if in_retry_mode else retry_errors_path, error_row)
            retry_errors += 1
            continue
        except Exception as exc:
            error_row = row.model_dump()
            error_row["step3_status"] = "invalid_generation"
            error_row["step3_error"] = str(exc)
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
        "Stage2 step3 summary: "
        f"saved={saved} retry_errors={retry_errors} invalid={invalid} "
        f"total_output_rows={count_jsonl_rows(resolved_output_path)}"
    )
    return resolved_output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run stage2 internal step3 (persona retrieval + first-pass Korean rewrite)."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input Stage1 JSONL path.")
    parser.add_argument(
        "--output",
        default=None,
        help=f"Output JSONL path. Defaults to {DEFAULT_OUTPUT}.",
    )
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR), help="Directory for Data Designer artifacts.")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE), help="Path to .env containing NVIDIA_API_KEY.")
    parser.add_argument("--persona-dir", default=str(DEFAULT_PERSONA_DIR), help="Directory containing persona JSONL files.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="NVIDIA model name.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="NVIDIA OpenAI-compatible endpoint.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base seed for persona sampling.")
    parser.add_argument("--mode", choices=["preview", "create"], default="create", help="Use Data Designer preview or create mode.")
    parser.add_argument("--num-records", type=int, default=None, help="Optionally limit the number of input rows.")
    parser.add_argument("--dataset-name", default="stage2-step3-rewrite", help="Dataset name prefix for Data Designer artifacts.")
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
    output_path = run(
        input_path=args.input,
        output_path=args.output,
        artifact_dir=args.artifact_dir,
        env_file=args.env_file,
        persona_dir=args.persona_dir,
        model=args.model,
        endpoint=args.endpoint,
        seed=args.seed,
        mode=args.mode,
        num_records=args.num_records,
        dataset_name=args.dataset_name,
        retry_errors_from=args.retry_errors_from,
        retry_source=args.retry_source,
        resume=args.resume,
    )
    print(f"Saved stage2 step3 output to {output_path}")


if __name__ == "__main__":
    main()
