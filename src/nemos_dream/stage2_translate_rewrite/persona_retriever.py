from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ALL_GENDERS = ("male", "female")
ALL_AGE_BUCKETS = ("10s", "20s", "30s", "40s", "50s", "60s", "70s", "80s", "90s")

SEX_TO_HINT = {
    "남자": "male",
    "여자": "female",
}

AGE_BUCKET_TO_FILE_KEY = {
    "10대": "10s",
    "20대": "20s",
    "30대": "30s",
    "40대": "40s",
    "50대": "50s",
    "60대": "60s",
    "70대": "70s",
    "80대": "80s",
    "90대": "90s",
}

GENERAL_KEYWORD_MAP = {
    "student": ["학생", "대학생", "조교", "학교", "교육"],
    "teacher": ["교사", "강사", "교수", "교육", "학교"],
    "priest": ["교회", "성당", "종교", "예배", "미사", "신부", "목사", "수녀"],
    "service_staff": ["서비스", "판매", "상점", "매장", "도우미"],
    "boss": ["대표", "임원", "사장", "경영", "관리"],
    "executive": ["대표", "임원", "경영", "관리"],
    "co-founder": ["대표", "창업", "경영", "사업"],
    "pet sitter": ["반려", "동물", "고양이", "강아지", "수의", "펫"],
    "stranger": [],
    "friend": ["친화", "사교", "공감", "대화"],
    "partner": ["연애", "가족", "배우자", "관계"],
    "religion": ["종교", "교회", "성당", "미사", "예배"],
    "spirituality": ["종교", "성찰", "교회", "성당"],
    "pets": ["반려", "동물", "고양이", "강아지", "수의", "펫"],
    "animals": ["동물", "고양이", "강아지", "수의", "반려"],
    "vacation": ["여행", "드라이브", "휴식"],
    "cat": ["고양이", "반려묘", "반려"],
    "listening": ["공감", "대화", "상담"],
    "friendship": ["친구", "공감", "사교"],
    "communication": ["대화", "소통", "상담"],
}

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PERSONA_DIR = REPO_ROOT / "data" / "persona_age_gender"


def stable_int(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
        return [converted]
    return [value]


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def normalize_gender_hint(value: Any) -> str | None:
    normalized = str(value or "").strip().lower()
    if normalized in {"male", "man", "m", "boy"}:
        return "male"
    if normalized in {"female", "woman", "f", "girl"}:
        return "female"
    return None


def expand_age_bucket_hint(value: Any) -> list[str]:
    normalized = str(value or "").strip().lower()
    mapping = {
        "teen": ["10s"],
        "teens": ["10s"],
        "10s": ["10s"],
        "20s": ["20s"],
        "30s": ["30s"],
        "40s": ["40s"],
        "50s": ["50s"],
        "60s": ["60s"],
        "70s": ["70s"],
        "80s": ["80s"],
        "90s": ["90s"],
        "40plus": ["40s", "50s", "60s", "70s", "80s", "90s"],
        "50plus": ["50s", "60s", "70s", "80s", "90s"],
        "60plus": ["60s", "70s", "80s", "90s"],
        "older": ["50s", "60s", "70s", "80s", "90s"],
    }
    return mapping.get(normalized, [])


def choose_random_age_bucket(speaker: dict[str, Any], rng: random.Random) -> str:
    role = str(speaker.get("role_in_scene") or "").strip().lower()
    occupation = str(speaker.get("occupation_hint") or "").strip().lower()
    if role == "student" or occupation == "student":
        population = ["10s", "20s", "30s", "40s"]
        weights = [0.45, 0.4, 0.1, 0.05]
        return rng.choices(population=population, weights=weights, k=1)[0]
    if role in {"teacher", "boss"} or occupation in {"priest", "executive", "co-founder"}:
        population = ["30s", "40s", "50s", "60s"]
        weights = [0.15, 0.35, 0.3, 0.2]
        return rng.choices(population=population, weights=weights, k=1)[0]
    population = list(ALL_AGE_BUCKETS)
    weights = [0.08, 0.2, 0.18, 0.15, 0.12, 0.1, 0.08, 0.06, 0.03]
    return rng.choices(population=population, weights=weights, k=1)[0]


def build_keyword_hints(speaker: dict[str, Any]) -> list[str]:
    keywords: list[str] = []
    interests_hints = ensure_list(speaker.get("interests_hints"))
    personality_traits = ensure_list(speaker.get("personality_traits"))
    for raw_value in [
        speaker.get("occupation_hint"),
        speaker.get("role_in_scene"),
        *interests_hints,
        *personality_traits,
    ]:
        normalized = str(raw_value or "").strip().lower()
        if not normalized:
            continue
        keywords.extend(GENERAL_KEYWORD_MAP.get(normalized, []))
    return dedupe_preserve_order(keywords)


def compact_persona_record(row: dict[str, Any]) -> dict[str, Any]:
    known_fields = {
        "persona_id",
        "name",
        "sex",
        "age",
        "age_bucket",
        "occupation",
        "normalized_location",
        "persona",
        "cultural_background",
        "skills_and_expertise",
        "hobbies_and_interests",
        "career_goals_and_ambitions",
        "summary_text",
    }
    extra = {key: value for key, value in row.items() if key not in known_fields}
    return {
        "persona_id": row.get("persona_id"),
        "name": row.get("name"),
        "sex": row.get("sex"),
        "age": row.get("age"),
        "age_bucket": row.get("age_bucket"),
        "occupation": row.get("occupation"),
        "normalized_location": row.get("normalized_location"),
        "persona": row.get("persona"),
        "cultural_background": row.get("cultural_background"),
        "skills_and_expertise": row.get("skills_and_expertise"),
        "hobbies_and_interests": row.get("hobbies_and_interests"),
        "career_goals_and_ambitions": row.get("career_goals_and_ambitions"),
        "summary_text": row.get("summary_text"),
        "extra": extra,
    }


def build_fallback_persona_entry(
    row: dict[str, Any],
    speaker_index: int,
    *,
    speaker_name_en: str = "",
    source_speaker_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    compact = compact_persona_record(row)
    return {
        "speaker_index": speaker_index,
        "speaker_name_en": speaker_name_en or f"speaker_{speaker_index}",
        "source_speaker_profile": source_speaker_profile or {},
        "selection_metadata": {
            "selected_random_gender": True,
            "selected_random_age_group": True,
            "matched_by_keywords": False,
            "match_score": 0.0,
            "candidate_gender": None,
            "candidate_age_buckets": [],
            "keyword_hints": [],
            "fallback_mode": "global_random_sample",
        },
        "retrieved_persona": compact,
    }


@dataclass(slots=True)
class PersonaSelectionContext:
    gender_key: str
    age_bucket_keys: list[str]
    selected_random_gender: bool
    selected_random_age: bool
    keyword_hints: list[str]


class PersonaRetriever:
    def __init__(self, persona_dir: str | Path) -> None:
        self.persona_dir = Path(persona_dir)
        if not self.persona_dir.exists():
            raise FileNotFoundError(f"Persona directory not found: {self.persona_dir}")
        self.file_map = self._discover_files()

    def _discover_files(self) -> dict[tuple[str, str], Path]:
        file_map: dict[tuple[str, str], Path] = {}
        for gender in ALL_GENDERS:
            for age_bucket in ALL_AGE_BUCKETS:
                path = self.persona_dir / f"persona_{gender}_{age_bucket}.jsonl"
                if path.exists():
                    file_map[(gender, age_bucket)] = path
        if not file_map:
            raise FileNotFoundError(f"No persona JSONL files found under: {self.persona_dir}")
        return file_map

    def _build_selection_context(
        self,
        speaker: dict[str, Any],
        rng: random.Random,
    ) -> PersonaSelectionContext:
        gender_key = normalize_gender_hint(speaker.get("gender_hint"))
        selected_random_gender = gender_key is None
        if gender_key is None:
            gender_key = rng.choice(list(ALL_GENDERS))

        age_bucket_keys = expand_age_bucket_hint(speaker.get("age_group_hint"))
        selected_random_age = not age_bucket_keys
        if not age_bucket_keys:
            age_bucket_keys = [choose_random_age_bucket(speaker, rng)]

        return PersonaSelectionContext(
            gender_key=gender_key,
            age_bucket_keys=age_bucket_keys,
            selected_random_gender=selected_random_gender,
            selected_random_age=selected_random_age,
            keyword_hints=build_keyword_hints(speaker),
        )

    def _candidate_paths(self, context: PersonaSelectionContext) -> list[Path]:
        paths: list[Path] = []
        for age_bucket in context.age_bucket_keys:
            path = self.file_map.get((context.gender_key, age_bucket))
            if path is not None:
                paths.append(path)
        return paths

    def _score_persona(self, row: dict[str, Any], keyword_hints: list[str], speaker: dict[str, Any]) -> float:
        occupation_text = str(row.get("occupation") or "")
        search_blob = " ".join(
            str(row.get(field) or "")
            for field in [
                "occupation",
                "persona",
                "cultural_background",
                "skills_and_expertise",
                "hobbies_and_interests",
                "summary_text",
            ]
        )
        score = 0.0
        for keyword in keyword_hints:
            if keyword in occupation_text:
                score += 8.0
            elif keyword in search_blob:
                score += 3.0

        register = str(speaker.get("register") or "").strip().lower()
        if register == "formal" and row.get("age_bucket") in {"40대", "50대", "60대", "70대", "80대", "90대"}:
            score += 0.5

        emotion_type = str((speaker.get("dominant_emotion") or {}).get("type") or "").strip().lower()
        if emotion_type in {"neutral", "sadness"} and any(
            token in search_blob for token in ["차분", "신중", "꼼꼼", "정갈", "안정", "침착"]
        ):
            score += 0.5
        return score

    def _iter_candidate_rows(self, candidate_paths: list[Path]) -> Any:
        for path in candidate_paths:
            with path.open("r", encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue
                    yield json.loads(line)

    def _iter_all_rows(self) -> Any:
        candidate_paths = sorted(set(self.file_map.values()))
        yield from self._iter_candidate_rows(candidate_paths)

    def select_one(
        self,
        speaker: dict[str, Any],
        record_id: str,
        speaker_index: int,
        base_seed: int,
        excluded_persona_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        excluded_persona_ids = excluded_persona_ids or set()
        rng = random.Random(base_seed + stable_int(f"{record_id}:{speaker_index}"))
        context = self._build_selection_context(speaker=speaker, rng=rng)
        candidate_paths = self._candidate_paths(context)
        if not candidate_paths:
            raise FileNotFoundError(
                f"No candidate persona files found for gender={context.gender_key}, "
                f"ages={context.age_bucket_keys}"
            )

        scored_candidates: list[tuple[float, int, dict[str, Any]]] = []
        fallback_reservoir: list[dict[str, Any]] = []
        seen = 0
        for row in self._iter_candidate_rows(candidate_paths):
            persona_id = str(row.get("persona_id") or "")
            if persona_id in excluded_persona_ids:
                continue

            seen += 1
            if len(fallback_reservoir) < 64:
                fallback_reservoir.append(row)
            else:
                replace_idx = rng.randint(0, seen - 1)
                if replace_idx < len(fallback_reservoir):
                    fallback_reservoir[replace_idx] = row

            score = self._score_persona(row=row, keyword_hints=context.keyword_hints, speaker=speaker)
            if score <= 0:
                continue

            scored_candidates.append((score, stable_int(persona_id), row))
            scored_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
            if len(scored_candidates) > 20:
                scored_candidates.pop()

        if scored_candidates:
            best_score = scored_candidates[0][0]
            best_pool = [row for score, _, row in scored_candidates if score == best_score]
            selected_row = rng.choice(best_pool)
            matched_by_keywords = True
            match_score = best_score
        elif fallback_reservoir:
            selected_row = rng.choice(fallback_reservoir)
            matched_by_keywords = False
            match_score = 0.0
        else:
            raise RuntimeError(
                f"Failed to find a persona candidate for record={record_id}, speaker_index={speaker_index}"
            )

        compact = compact_persona_record(selected_row)
        return {
            "speaker_index": speaker_index,
            "speaker_name_en": speaker.get("name_en"),
            "source_speaker_profile": speaker,
            "selection_metadata": {
                "selected_random_gender": context.selected_random_gender,
                "selected_random_age_group": context.selected_random_age,
                "matched_by_keywords": matched_by_keywords,
                "match_score": match_score,
                "candidate_gender": context.gender_key,
                "candidate_age_buckets": context.age_bucket_keys,
                "keyword_hints": context.keyword_hints,
            },
            "retrieved_persona": compact,
        }

    def select_many(
        self,
        speakers: list[dict[str, Any]],
        record_id: str,
        base_seed: int,
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        excluded_persona_ids: set[str] = set()
        for speaker_index, speaker in enumerate(ensure_list(speakers)):
            result = self.select_one(
                speaker=speaker,
                record_id=record_id,
                speaker_index=speaker_index,
                base_seed=base_seed,
                excluded_persona_ids=excluded_persona_ids,
            )
            persona_id = str((result.get("retrieved_persona") or {}).get("persona_id") or "")
            if persona_id:
                excluded_persona_ids.add(persona_id)
            selected.append(result)
        return selected

    def sample_random_personas(
        self,
        count: int,
        record_id: str,
        base_seed: int,
        excluded_persona_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        excluded_persona_ids = excluded_persona_ids or set()
        rng = random.Random(base_seed + stable_int(f"{record_id}:fallback"))
        reservoir: list[dict[str, Any]] = []
        seen = 0
        for row in self._iter_all_rows():
            persona_id = str(row.get("persona_id") or "")
            if persona_id in excluded_persona_ids:
                continue
            seen += 1
            if len(reservoir) < max(count * 16, 64):
                reservoir.append(row)
            else:
                replace_idx = rng.randint(0, seen - 1)
                if replace_idx < len(reservoir):
                    reservoir[replace_idx] = row

        if not reservoir:
            return []

        rng.shuffle(reservoir)
        chosen_rows = reservoir[:count]
        return [build_fallback_persona_entry(row, speaker_index=i) for i, row in enumerate(chosen_rows)]

    def sample_random_personas_for_speakers(
        self,
        speakers: list[dict[str, Any]],
        record_id: str,
        base_seed: int,
    ) -> list[dict[str, Any]]:
        rng = random.Random(base_seed + stable_int(f"{record_id}:fallback"))
        reservoir: list[dict[str, Any]] = []
        seen = 0
        for row in self._iter_all_rows():
            seen += 1
            if len(reservoir) < max(len(speakers) * 16, 64):
                reservoir.append(row)
            else:
                replace_idx = rng.randint(0, seen - 1)
                if replace_idx < len(reservoir):
                    reservoir[replace_idx] = row

        if not reservoir:
            return []

        rng.shuffle(reservoir)
        chosen_rows = reservoir[: len(speakers)]
        return [
            build_fallback_persona_entry(
                row,
                speaker_index=index,
                speaker_name_en=str((speaker or {}).get("name_en") or ""),
                source_speaker_profile=speaker,
            )
            for index, (speaker, row) in enumerate(zip(ensure_list(speakers), chosen_rows, strict=False))
        ]


def format_persona_prompt_context(personas: list[dict[str, Any]]) -> str:
    if not personas:
        try:
            fallback_retriever = PersonaRetriever(DEFAULT_PERSONA_DIR)
            personas = fallback_retriever.sample_random_personas(
                count=2,
                record_id="format_persona_prompt_context",
                base_seed=0,
            )
        except Exception:
            return "선택된 persona 없음"
        if not personas:
            return "선택된 persona 없음"

    blocks: list[str] = []
    for item in personas:
        retrieved = item.get("retrieved_persona") or {}
        selection = item.get("selection_metadata") or {}
        speaker_profile = item.get("source_speaker_profile") or {}
        blocks.append(
            "\n".join(
                [
                    f"[Speaker {item.get('speaker_index')}] {item.get('speaker_name_en')}",
                    f"- source role: {speaker_profile.get('role_in_scene', '')}",
                    f"- source register: {speaker_profile.get('register', '')}",
                    f"- source occupation_hint: {speaker_profile.get('occupation_hint', '') or '없음'}",
                    (
                        f"- selected persona: {retrieved.get('name', '')} / {retrieved.get('sex', '')} / "
                        f"{retrieved.get('age', '')}세 / {retrieved.get('occupation', '') or '직업정보없음'} / "
                        f"{retrieved.get('normalized_location', '')}"
                    ),
                    f"- persona summary: {retrieved.get('persona', '')}",
                    f"- cultural background: {retrieved.get('cultural_background', '')}",
                    f"- skills: {retrieved.get('skills_and_expertise', '')}",
                    f"- hobbies: {retrieved.get('hobbies_and_interests', '')}",
                    (
                        f"- retrieval notes: random_gender={selection.get('selected_random_gender', False)}, "
                        f"random_age_group={selection.get('selected_random_age_group', False)}, "
                        f"matched_by_keywords={selection.get('matched_by_keywords', False)}, "
                        f"keyword_hints={selection.get('keyword_hints', [])}"
                    ),
                ]
            )
        )
    return "\n\n".join(blocks)
