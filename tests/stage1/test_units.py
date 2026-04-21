"""Pure-Python unit tests for stage 1 helpers. No NVIDIA calls."""

from __future__ import annotations

from nemos_dream.schemas import CulturalRef, MappedRef
from nemos_dream.stage1_decompose_map._ref_filter import filter_refs
from nemos_dream.stage1_decompose_map._validator import _check_rules, validate_ref
from nemos_dream.stage1_decompose_map.tools import dict_lookup


def test_filter_refs_drops_generic_terms():
    refs = [
        CulturalRef(type="food", term="dinner"),
        CulturalRef(type="food", term="pumpkin spice latte"),
        CulturalRef(type="other", term="friend"),
        CulturalRef(type="brand", term="starbucks"),
    ]
    kept = filter_refs(refs)
    terms = {r.term for r in kept}
    assert terms == {"pumpkin spice latte", "starbucks"}


def test_validator_flags_empty_ko():
    ref = MappedRef(term="thanksgiving", ko="", type="holiday", source="web+llm")
    flags = _check_rules(ref)
    codes = {f["code"] for f in flags}
    assert "ko_empty_or_same" in codes


def test_validator_flags_echoed_term():
    ref = MappedRef(term="clown", ko="clown", type="other", source="web+llm")
    flags = _check_rules(ref)
    assert flags and flags[0]["code"] == "ko_empty_or_same"


def test_validator_flags_non_hangul_brand_as_error():
    # A brand mapped to a non-Korean token — not an echo of the source term.
    ref = MappedRef(term="chipotle", ko="Hom", type="brand", source="web+llm")
    flags = _check_rules(ref)
    assert any(f["code"] == "non_hangul_ko" and f["severity"] == "error" for f in flags)


def test_validator_flags_non_hangul_pop_culture_as_warn():
    # pop_culture stage names like IU/BTS are legitimately Latin; warn, not error.
    # ko must differ from term (otherwise ko_empty_or_same short-circuits).
    ref = MappedRef(term="aiu", ko="IU", type="pop_culture", source="dict")
    flags = _check_rules(ref)
    pop_flags = [f for f in flags if f["code"] == "non_hangul_ko"]
    assert pop_flags and pop_flags[0]["severity"] == "warn"


def test_validator_flags_corrupted_underscore_token():
    ref = MappedRef(
        term="4th of july",
        ko="광복절",
        type="holiday",
        source="retriever",
        notes="대한민국이 일본의 식민지_rule을 해방한 날",
    )
    flags = _check_rules(ref)
    assert any(f["code"] == "corrupted_token" for f in flags)


def test_validator_clean_mapping_passes():
    ref = MappedRef(
        term="thanksgiving", ko="추석", type="holiday", source="dict",
        notes="가족 모임 맥락",
    )
    assert _check_rules(ref) == []


def test_validate_ref_copies_flags_onto_model():
    ref = MappedRef(term="clown", ko="clown", type="other", source="web+llm")
    validated = validate_ref(ref)
    assert len(validated.validation) >= 1
    # Original is not mutated.
    assert ref.validation == []


def test_dict_lookup_returns_seed_entry():
    hit = dict_lookup.lookup("thanksgiving")
    assert hit is not None
    assert hit["ko"] == "추석"
    assert hit["type"] == "holiday"


def test_dict_lookup_normalizes_case_and_punctuation():
    # Normalizer lowercases and strips punctuation before matching.
    assert dict_lookup.lookup("Thanksgiving!") is not None
    assert dict_lookup.lookup("  STARBUCKS  ") is not None
