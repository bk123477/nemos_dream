"""Typed loader for ``configs/stage3/filter.yaml``.

The runner takes explicit kwargs so callers can inject test values without
touching disk, but the project-default config lives in YAML (per README —
"Judge + reward model IDs live here, not in code"). This module reads the
YAML and flattens it into the kwarg shape ``runner.run_async`` expects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

DEFAULT_PATH = Path("configs/stage3/filter.yaml")


@dataclass
class Stage3Config:
    axis_floor: int = 2
    aggregate_floor: float = 3.5
    ascii_ratio_max: float = 0.40
    intra_kr_coherence_floor: float = 0.55
    jaccard_threshold: float = 0.8
    num_perm: int = 128
    semantic_cosine_threshold: float = 0.92
    quality_weights: dict[str, float] = field(
        default_factory=lambda: {
            "property_preservation": 0.20,
            "naturalness": 0.20,
            "cultural_appropriateness": 0.35,
            "register_consistency": 0.125,
            "persona_style_consistency": 0.125,
        }
    )
    reward_weights: dict[str, float] = field(
        default_factory=lambda: {"correctness": 0.5, "coherence": 0.5}
    )
    self_verify_max_iter: int = 2
    self_verify_enabled_actions: list[str] = field(
        default_factory=lambda: [
            "stage1_redecompose",
            "maps_ref_redo",
            "stage2_rewrite",
            "websearch_cultural",
        ]
    )

    def as_runner_kwargs(self) -> dict[str, Any]:
        return {
            "axis_floor": self.axis_floor,
            "aggregate_floor": self.aggregate_floor,
            "coherence_floor": self.intra_kr_coherence_floor,
            "rules_cfg": {"ascii_ratio_max": self.ascii_ratio_max},
        }


def load(path: str | Path | None = None) -> Stage3Config:
    """Load the stage-3 config YAML, with graceful fallback to defaults.

    Missing file or missing PyYAML both return :class:`Stage3Config` with
    dataclass defaults — the runner is always usable.
    """

    cfg = Stage3Config()
    p = Path(path) if path is not None else DEFAULT_PATH
    if not p.is_file() or yaml is None:
        return cfg

    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    thr = raw.get("thresholds", {}) or {}
    dedup = raw.get("dedup", {}) or {}
    weights = raw.get("quality_weights", {}) or {}
    reward = (raw.get("reward", {}) or {}).get("weights", {}) or {}
    sv = raw.get("self_verify", {}) or {}

    if "axis_floor" in thr:
        cfg.axis_floor = int(thr["axis_floor"])
    if "aggregate" in thr:
        cfg.aggregate_floor = float(thr["aggregate"])
    if "ascii_ratio_max" in thr:
        cfg.ascii_ratio_max = float(thr["ascii_ratio_max"])
    if "intra_kr_coherence_floor" in thr:
        cfg.intra_kr_coherence_floor = float(thr["intra_kr_coherence_floor"])

    if "jaccard_threshold" in dedup:
        cfg.jaccard_threshold = float(dedup["jaccard_threshold"])
    if "num_perm" in dedup:
        cfg.num_perm = int(dedup["num_perm"])
    if "semantic_cosine_threshold" in dedup:
        cfg.semantic_cosine_threshold = float(dedup["semantic_cosine_threshold"])

    if weights:
        # Reference config uses "property" / "cultural" / "register" short keys;
        # normalise to our 5-axis names. Unknown keys are ignored.
        key_map = {
            "property": "property_preservation",
            "naturalness": "naturalness",
            "cultural": "cultural_appropriateness",
            "register": "register_consistency",
            "persona_style": "persona_style_consistency",
        }
        normalised: dict[str, float] = {}
        for k, v in weights.items():
            normalised[key_map.get(k, k)] = float(v)
        if normalised:
            cfg.quality_weights = normalised

    if reward:
        cfg.reward_weights = {k: float(v) for k, v in reward.items()}

    if "max_iter" in sv:
        cfg.self_verify_max_iter = int(sv["max_iter"])
    if "enabled_actions" in sv:
        cfg.self_verify_enabled_actions = list(sv["enabled_actions"])

    return cfg
