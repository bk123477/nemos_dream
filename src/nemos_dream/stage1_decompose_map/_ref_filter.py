"""Generic-term blocklist applied at ``map_refs`` entry.

The decomposer's system prompt carries the primary rule that truly generic
nouns (``dinner``, ``friend``, ``school``) should not appear in
``cultural_refs`` — this module is the backup for when the LLM ignores it.
"""

from __future__ import annotations

from nemos_dream.schemas import CulturalRef

GENERIC_BLOCKLIST: frozenset[str] = frozenset({
    "god", "heaven", "hell",
    "birthday", "wedding", "funeral", "anniversary",
    "dinner", "lunch", "breakfast", "brunch", "meal", "snack",
    "friend", "family", "mom", "dad", "brother", "sister", "parent",
    "weekend", "morning", "evening", "night",
    "phone", "car", "house", "school", "work", "office", "home",
    "holiday",
})


def filter_refs(refs: list[CulturalRef]) -> list[CulturalRef]:
    return [r for r in refs if r.term.strip().lower() not in GENERIC_BLOCKLIST]
