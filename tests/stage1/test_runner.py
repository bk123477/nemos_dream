"""Runner orchestration test: stubs out decompose + map_refs so this runs
without any NVIDIA credentials and still validates the end-to-end wiring."""

from __future__ import annotations

from pathlib import Path

from nemos_dream.io_utils import read_jsonl, write_jsonl
from nemos_dream.schemas import (
    CulturalRef,
    DialogueDecomposed,
    Emotion,
    MappedRef,
    RawInput,
    Scene,
    Speaker,
    Stage1Output,
    Turn,
)
from nemos_dream.stage1_decompose_map import cultural_map, runner
from nemos_dream.stage1_decompose_map.decompose import DecomposeResult


def _fake_decompose_result(row: RawInput) -> DecomposeResult:
    turns = [
        Turn(index=i, speaker=sp, text=utt)
        for i, (sp, utt) in enumerate(zip(row.speakers, row.dialogue, strict=False))
    ]
    speakers = []
    seen: set[str] = set()
    for sp in row.speakers:
        if sp in seen:
            continue
        seen.add(sp)
        speakers.append(Speaker(
            name_en=sp,
            role_in_scene="friend",
            register="casual",
            dominant_emotion=Emotion(type="neutral", intensity=3),
        ))
    scene = Scene(
        narrative_en=row.narrative,
        setting="home",
        relationship_type="friendship",
        topics=["test"],
    )
    # Pick a term that substring-matches the dialogue to exercise map_refs.
    refs = []
    dialogue_text = " ".join(row.dialogue).lower()
    for candidate in ("thanksgiving", "starbucks", "pizza"):
        if candidate in dialogue_text:
            refs.append(CulturalRef(type="other", term=candidate))
            break
    dd = DialogueDecomposed(
        overall_register="casual",
        overall_emotion=Emotion(type="neutral", intensity=3),
        speech_acts=["statement"],
        cultural_refs=refs,
    )
    return DecomposeResult(turns=turns, speakers=speakers, scene=scene, dialogue_decomposed=dd)


def test_runner_end_to_end_with_stubs(tmp_path: Path, monkeypatch):
    # Stub decompose + map_refs so the runner never hits the network.
    import nemos_dream.stage1_decompose_map.runner as runner_mod

    def fake_decompose(rows):
        return [_fake_decompose_result(r) for r in rows]

    def fake_map_refs(refs, *, dialogue=None):
        return [
            MappedRef(term=r.term, ko="테스트", type=r.type, source="dict", retrieved=False)
            for r in refs
        ]

    monkeypatch.setattr(runner_mod, "decompose", fake_decompose)
    monkeypatch.setattr(runner_mod, "map_refs", fake_map_refs)
    monkeypatch.setattr(cultural_map, "map_refs", fake_map_refs)

    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "stage1" / "in.jsonl"

    rows_in = [
        RawInput(
            original_index=1,
            dialogue=["Happy thanksgiving!", "You too!"],
            speakers=["Alex", "Bea"],
            narrative="Two friends greeting on thanksgiving.",
        ),
        RawInput(
            id="custom-id-2",
            original_index=2,
            dialogue=["Grabbed a coffee at starbucks", "nice"],
            speakers=["Sam", "Lee"],
            narrative="Coffee small talk.",
        ),
    ]
    write_jsonl(input_path, rows_in)

    n = runner.run(input_path, output_path)
    assert n == 2
    assert output_path.exists()

    rows_out = list(read_jsonl(output_path, Stage1Output))
    assert len(rows_out) == 2

    # ID convention: missing id → soda-<original_index>; existing id preserved.
    assert rows_out[0].id == "soda-1"
    assert rows_out[1].id == "custom-id-2"

    # Stage1Output invariant: cultural_ref terms must appear verbatim in the source.
    for row_out in rows_out:
        joined = " ".join(t.text for t in row_out.source_dialogue).lower()
        for ref in row_out.dialogue_decomposed.cultural_refs:
            assert ref.term in joined

    # mapped_refs were populated by our stub.
    assert all(m.ko == "테스트" for r in rows_out for m in r.mapped_refs)


def test_runner_requires_input_and_output():
    import pytest

    with pytest.raises(ValueError):
        runner.run(None, "x")
    with pytest.raises(ValueError):
        runner.run("x", None)
