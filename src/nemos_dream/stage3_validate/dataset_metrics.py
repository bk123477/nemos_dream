"""Absolute (non-reference) dataset-health metrics for stage 3.

All metrics are computed locally with no NIM calls unless an ``embed_fn`` is
passed — when absent, embedding-diversity falls back to a distinct-bigram
proxy so the metric stays numeric in offline mode.

Contract: ``compute(accepted, rejected, *, embed_fn=None) -> dict``.

The returned dict is JSON-serialisable end-to-end and is written to
``dataset_metrics.json`` by the runner.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from statistics import mean

from nemos_dream.schemas import Stage3Output

EmbedFn = Callable[[list[str]], list[list[float]]]


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return float(s[k])


def _stats(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "n": 0}
    return {
        "mean": round(float(mean(xs)), 4),
        "p50": round(_percentile(xs, 50), 4),
        "p95": round(_percentile(xs, 95), 4),
        "n": len(xs),
    }


def _entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    h = 0.0
    for v in counter.values():
        p = v / total
        if p > 0:
            h -= p * math.log2(p)
    return round(h, 4)


def _tokenise(text: str) -> list[str]:
    return [tok for tok in text.replace("\n", " ").split(" ") if tok]


def _row_text(row: Stage3Output) -> str:
    turns = row.final_dialogue or row.korean_dialogue
    return " \n ".join(t.text for t in turns)


def _distinct_n(texts: list[str], n: int) -> float:
    total = 0
    seen: set[tuple[str, ...]] = set()
    for t in texts:
        toks = _tokenise(t)
        if len(toks) < n:
            continue
        for i in range(len(toks) - n + 1):
            gram = tuple(toks[i : i + n])
            seen.add(gram)
            total += 1
    if total == 0:
        return 0.0
    return round(len(seen) / total, 4)


def _reward_distribution(rows: list[Stage3Output]) -> dict[str, dict[str, float]]:
    keys: set[str] = set()
    for r in rows:
        if r.quality.reward:
            keys.update(r.quality.reward.keys())
    out: dict[str, dict[str, float]] = {}
    for k in sorted(keys):
        vals = [r.quality.reward[k] for r in rows if r.quality.reward and k in r.quality.reward]
        out[k] = _stats(vals)
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=True))
    da = sum(x * x for x in a) ** 0.5
    db = sum(y * y for y in b) ** 0.5
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def _embedding_diversity(
    rows: list[Stage3Output], embed_fn: EmbedFn | None
) -> dict[str, float]:
    texts = [_row_text(r) for r in rows if _row_text(r).strip()]
    if len(texts) < 2:
        return {"mean_pairwise_distance": 0.0, "source": "insufficient"}
    if embed_fn is not None:
        embeds = embed_fn(texts)
        pairs = 0
        acc = 0.0
        for i in range(len(embeds)):
            for j in range(i + 1, len(embeds)):
                acc += 1.0 - _cosine(embeds[i], embeds[j])
                pairs += 1
        return {
            "mean_pairwise_distance": round(acc / max(pairs, 1), 4),
            "source": "nv_embed",
        }
    # Distinct-bigram proxy: inverse overlap over corpus-wide bigram sets.
    bigrams: list[set[tuple[str, str]]] = []
    for t in texts:
        toks = _tokenise(t)
        bigrams.append({(toks[i], toks[i + 1]) for i in range(len(toks) - 1)})
    pairs = 0
    acc = 0.0
    for i in range(len(bigrams)):
        for j in range(i + 1, len(bigrams)):
            a, b = bigrams[i], bigrams[j]
            if not a or not b:
                continue
            jacc = len(a & b) / len(a | b)
            acc += 1.0 - jacc
            pairs += 1
    return {
        "mean_pairwise_distance": round(acc / max(pairs, 1), 4),
        "source": "bigram_jaccard_stub",
    }


def _decomposed_coverage_entropy(rows: list[Stage3Output]) -> dict[str, float]:
    register: Counter = Counter()
    emotion: Counter = Counter()
    speech: Counter = Counter()
    setting: Counter = Counter()
    relationship: Counter = Counter()
    age: Counter = Counter()
    speaker_register: Counter = Counter()
    for r in rows:
        register[r.dialogue_decomposed.overall_register] += 1
        emotion[r.dialogue_decomposed.overall_emotion.type] += 1
        for sa in r.dialogue_decomposed.speech_acts:
            speech[sa] += 1
        setting[r.scene.setting] += 1
        relationship[r.scene.relationship_type] += 1
        for sp in r.speakers:
            age[sp.age_group_hint] += 1
            speaker_register[sp.register] += 1
    return {
        "overall_register": _entropy(register),
        "overall_emotion": _entropy(emotion),
        "speech_acts": _entropy(speech),
        "scene_setting": _entropy(setting),
        "relationship_type": _entropy(relationship),
        "speaker_age_group": _entropy(age),
        "speaker_register": _entropy(speaker_register),
    }


def _cultural_ref_diversity(rows: list[Stage3Output]) -> dict[str, float]:
    """Distribution over ``dialogue_decomposed.cultural_refs`` (source-side)."""
    terms: set[str] = set()
    total = 0
    per_type: Counter = Counter()
    for r in rows:
        for c in r.dialogue_decomposed.cultural_refs:
            terms.add(c.term)
            total += 1
            per_type[c.type] += 1
    return {
        "total_refs": total,
        "distinct_terms": len(terms),
        "diversity_ratio": round(len(terms) / max(total, 1), 4),
        "type_entropy": _entropy(per_type),
    }


def _mapped_ref_stats(rows: list[Stage3Output]) -> dict[str, object]:
    total = 0
    distinct_ko: set[str] = set()
    retrieved = 0
    per_source: Counter = Counter()
    per_type: Counter = Counter()
    for r in rows:
        for m in r.mapped_refs:
            total += 1
            if m.ko:
                distinct_ko.add(m.ko)
            if m.retrieved:
                retrieved += 1
            per_source[m.source] += 1
            per_type[m.type] += 1
    return {
        "total_mapped": total,
        "distinct_ko": len(distinct_ko),
        "retrieved_share": round(retrieved / max(total, 1), 4),
        "by_source": dict(sorted(per_source.items())),
        "by_type": dict(sorted(per_type.items())),
    }


def _length_stats(rows: list[Stage3Output]) -> dict[str, dict[str, float]]:
    char_counts: list[float] = []
    token_counts: list[float] = []
    turn_counts: list[float] = []
    ref_counts: list[float] = []
    for r in rows:
        t = _row_text(r)
        char_counts.append(float(len(t)))
        token_counts.append(float(len(_tokenise(t))))
        turn_counts.append(float(len(r.source_dialogue)))
        ref_counts.append(float(len(r.mapped_refs)))
    return {
        "chars": _stats(char_counts),
        "tokens": _stats(token_counts),
        "turns": _stats(turn_counts),
        "mapped_refs": _stats(ref_counts),
    }


def _reject_breakdown(rows: list[Stage3Output]) -> dict[str, int]:
    c: Counter = Counter()
    for r in rows:
        for rr in r.reject_reasons:
            key = f"{rr.stage}:{rr.rule or 'unknown'}"
            c[key] += 1
    return dict(sorted(c.items()))


def _retry_stats(rows: list[Stage3Output]) -> dict[str, object]:
    iter_hist: Counter = Counter()
    action_counts: Counter = Counter()
    valid_after_retry = 0
    retried = 0
    for r in rows:
        iter_hist[r.iter] += 1
        if r.iter > 0:
            retried += 1
            if r.valid:
                valid_after_retry += 1
        for a in r.retry_actions:
            action_counts[a.action] += 1
    return {
        "iter_histogram": {str(k): v for k, v in sorted(iter_hist.items())},
        "action_counts": dict(sorted(action_counts.items())),
        "valid_after_retry_rate": round(valid_after_retry / retried, 4) if retried else 0.0,
        "retried_rows": retried,
    }


def compute(
    accepted: list[Stage3Output],
    rejected: list[Stage3Output],
    *,
    embed_fn: EmbedFn | None = None,
) -> dict:
    """Compute dataset-health metrics over the stage-3 output population.

    Diversity-style metrics (distinct-n, embedding-diversity, entropy) run on
    the accepted set only — those are the rows the SFT pipeline will actually
    see. Reject/retry breakdowns cover the full population so stage 4 can
    surface failure modes.
    """
    all_rows = accepted + rejected
    kr_texts = [_row_text(r) for r in accepted]

    return {
        "counts": {
            "accepted": len(accepted),
            "rejected": len(rejected),
            "total": len(all_rows),
        },
        "reward_distribution": _reward_distribution(accepted),
        "embedding_diversity": _embedding_diversity(accepted, embed_fn),
        "distinct_n": {
            "1": _distinct_n(kr_texts, 1),
            "2": _distinct_n(kr_texts, 2),
            "3": _distinct_n(kr_texts, 3),
        },
        "decomposed_coverage_entropy": _decomposed_coverage_entropy(accepted),
        "cultural_ref_diversity": _cultural_ref_diversity(accepted),
        "mapped_ref_stats": _mapped_ref_stats(accepted),
        "length_stats": _length_stats(accepted),
        "reject_breakdown": _reject_breakdown(all_rows),
        "retry_stats": _retry_stats(all_rows),
    }
