from __future__ import annotations

from pathlib import Path

import pandas as pd

from nemos_dream.stage2_translate_rewrite import run_step3, runner
from nemos_dream.stage2_translate_rewrite.pipeline_modes import (
    default_stage2_output_path,
    default_step3_output_path,
)
from nemos_dream.stage2_translate_rewrite.run_step4 import mirror_step3_to_final
from tests.fixtures.sample_rows import sample_stage1, sample_stage2


def test_mode_specific_default_paths() -> None:
    base_dir = Path("data/stage2")
    assert default_step3_output_path(base_dir, "default") == base_dir / "stage2_rewrite.jsonl"
    assert default_step3_output_path(base_dir, "direct") == base_dir / "stage2_direct.jsonl"
    assert default_step3_output_path(base_dir, "naive_persona") == base_dir / "stage2_naive_persona.jsonl"
    assert default_stage2_output_path(base_dir, "default") == base_dir / "out.jsonl"
    assert default_stage2_output_path(base_dir, "direct") == base_dir / "out.direct.jsonl"
    assert default_stage2_output_path(base_dir, "naive_persona") == base_dir / "out.naive_persona.jsonl"


def test_runner_forwards_mode_specific_paths(tmp_path, monkeypatch) -> None:
    calls: list[tuple[str, Path, Path | None, str, int | None, bool]] = []

    def fake_run_step3(
        input_path: str | Path,
        output_path: str | Path | None = None,
        *,
        pipeline_mode: str,
        num_records: int | None = None,
        resume: bool = True,
        **_: object,
    ) -> Path:
        output = Path(output_path or "")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('{"id":"row-1"}\n', encoding="utf-8")
        calls.append(("step3", output, None, pipeline_mode, num_records, resume))
        return output

    def fake_run_step4(
        input_path: str | Path,
        output_path: str | Path | None = None,
        *,
        pipeline_mode: str,
        num_records: int | None = None,
        resume: bool = True,
        **_: object,
    ) -> Path:
        input_target = Path(input_path)
        output = Path(output_path or "")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('{"id":"row-1"}\n', encoding="utf-8")
        calls.append(("step4", input_target, output, pipeline_mode, num_records, resume))
        return output

    monkeypatch.setattr(runner, "run_step3", fake_run_step3)
    monkeypatch.setattr(runner, "run_step4", fake_run_step4)

    final_output = tmp_path / "custom.final.jsonl"
    count = runner.run(
        tmp_path / "input.jsonl",
        final_output,
        pipeline_mode="naive_persona",
        num_records=2,
        resume=False,
    )

    assert count == 1
    assert calls == [
        (
            "step3",
            tmp_path / "stage2_naive_persona.jsonl",
            None,
            "naive_persona",
            2,
            False,
        ),
        (
            "step4",
            tmp_path / "stage2_naive_persona.jsonl",
            final_output,
            "naive_persona",
            2,
            False,
        ),
    ]


def test_run_step3_direct_mode_sets_final_dialogue(monkeypatch) -> None:
    stage1_row = sample_stage1()

    def fake_run_data_designer(**_: object) -> pd.DataFrame:
        payload = {
            "step3_korean_dialogue": {
                "dialogue": [
                    {
                        "index": turn.index,
                        "speaker": turn.speaker,
                        "text": f"직역 {turn.index}",
                    }
                    for turn in stage1_row.source_dialogue
                ]
            }
        }
        return pd.DataFrame([payload])

    monkeypatch.setattr(run_step3, "run_data_designer", fake_run_data_designer)

    result = run_step3.run_single_row(
        stage1_row,
        api_key="test",
        model_name="test-model",
        endpoint="https://example.com",
        persona_dir="unused",
        artifact_root="unused",
        mode="preview",
        base_seed=7,
        dataset_name_prefix="test",
        pipeline_mode="direct",
    )

    assert result.persona == []
    assert result.final_dialogue == result.step3_korean_dialogue
    assert result.final_dialogue[0].speaker == stage1_row.source_dialogue[0].speaker
    assert result.translation_meta["pipeline_mode"] == "direct"
    assert result.translation_meta["step4_applied"] is False


def test_run_step3_naive_persona_uses_persona_names(monkeypatch) -> None:
    stage1_row = sample_stage1()
    stage2_row = sample_stage2()
    persona_payload = [item.model_dump() for item in stage2_row.persona]

    def fake_run_data_designer(**_: object) -> pd.DataFrame:
        payload = {
            "persona": persona_payload,
            "step3_korean_dialogue": {
                "dialogue": [
                    {
                        "index": turn.index,
                        "speaker": turn.speaker,
                        "text": f"번역 {turn.index}",
                    }
                    for turn in stage1_row.source_dialogue
                ]
            },
        }
        return pd.DataFrame([payload])

    monkeypatch.setattr(run_step3, "run_data_designer", fake_run_data_designer)

    result = run_step3.run_single_row(
        stage1_row,
        api_key="test",
        model_name="test-model",
        endpoint="https://example.com",
        persona_dir="unused",
        artifact_root="unused",
        mode="preview",
        base_seed=7,
        dataset_name_prefix="test",
        pipeline_mode="naive_persona",
    )

    assert result.persona
    assert result.final_dialogue == result.step3_korean_dialogue
    assert result.final_dialogue[0].speaker == "김수현"
    assert result.final_dialogue[1].speaker == "박윤지"
    assert result.translation_meta["pipeline_mode"] == "naive_persona"
    assert result.translation_meta["step4_applied"] is False


def test_mirror_step3_to_final_preserves_step3_dialogue() -> None:
    row = sample_stage2().model_copy(deep=True)
    row.final_dialogue = []
    mirrored = mirror_step3_to_final(row, "direct")

    assert mirrored.final_dialogue == row.step3_korean_dialogue
    assert mirrored.translation_meta["pipeline_mode"] == "direct"
    assert mirrored.translation_meta["step4_applied"] is False
