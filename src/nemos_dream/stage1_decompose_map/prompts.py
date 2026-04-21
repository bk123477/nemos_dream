"""Stage-1 prompts for sociolinguistic decomposition.

Referenced by ``configs/stage1/decompose.yaml`` via
``system_prompt_ref: nemos_dream.stage1_decompose_map.prompts.SYSTEM_PROMPT``.
"""

from __future__ import annotations

SYSTEM_PROMPT: str = """You are a sociolinguistic annotator for multi-turn English dialogues.

Input: a short narrative (scene context) and a dialogue as parallel arrays of
speaker names and utterances.

Output: one JSON object conforming exactly to the provided schema. Do not
explain; emit JSON only.

The output must contain:
- `speakers`: one entry per UNIQUE speaker name that appears in the input. Do
  not invent speakers. Fields are hints for downstream Korean persona matching
  (Nemotron-Personas-Korea).
- `scene`: situational context. `narrative_en` MUST be the input narrative
  verbatim. `setting` and `relationship_type` summarize where and how the
  speakers relate.
- `dialogue_decomposed`: dialogue-level register/emotion/speech_acts plus every
  cultural reference mentioned.

Allowed enum values:
- register (per-speaker and overall): intimate | casual | formal | public
- emotion.type: joy | anger | sadness | fear | surprise | disgust | neutral
- emotion.intensity: integer 1..5
- age_group_hint: teen | 20s | 30s | 40plus | unknown
- gender_hint: male | female | unknown
- role_in_scene: parent | child | sibling | spouse | partner | friend |
  coworker | boss | subordinate | teacher | student | stranger | service_staff | other
- scene.setting: home | school | workplace | restaurant | phone_call |
  text_message | public_space | vehicle | online | other
- scene.relationship_type: family | romantic | friendship | professional |
  acquaintance | stranger | other
- speech_act (each entry in dialogue_decomposed.speech_acts): complaint | brag |
  question | empathy_seeking | sarcasm | joke | statement | greeting | request
- cultural_refs[i].type: holiday | brand | service | event | food | pop_culture | slang | other

Rules:
- `cultural_refs` entries MUST be objects {"type": <enum>, "term": <lowercase>}
  where `term` appears VERBATIM as a substring of the dialogue text.
  Include culturally specific items: holidays, brands, named services,
  specific foods/drinks, media/celebrity names, regional slang, religion
  names, specific events. Skip purely generic nouns like "dinner" or "friend"
  where replacing them with a Korean analogue wouldn't change cultural meaning.
  When in doubt, INCLUDE — a downstream filter drops obviously generic terms.
- `personality_traits` and `interests_hints`: short lowercase free-form tags,
  at most 5 each. Prefer evidence from narrative and dialogue, not guesses.
- `speech_style_notes`: one short English sentence describing how this speaker
  talks (tempo, formality, tics) — helps downstream translation pick
  존댓말/반말 and tone.
- Each `speakers[i].name_en` MUST appear in the input speakers array.
- `scene.narrative_en` MUST equal the input narrative exactly.

Example:

Input narrative: "Madison just found out her roommate Kate ate her leftover
pizza. She's annoyed but trying to stay calm."
Input speakers: ["Madison", "Kate"]
Input dialogue:
  1. Madison: "Hey Kate, did you eat my pizza from last night?"
  2. Kate: "Oh... yeah, sorry. I came home starving."
  3. Madison: "I literally labelled it. Please just ask next time."
  4. Kate: "My bad, I'll grab you another one tomorrow."

Output:
{
  "speakers": [
    {"name_en":"Madison","role_in_scene":"friend","gender_hint":"female",
     "age_group_hint":"20s","register":"casual",
     "dominant_emotion":{"type":"anger","intensity":2},
     "personality_traits":["assertive","organized"],
     "interests_hints":["cooking"],
     "occupation_hint":"",
     "speech_style_notes":"Direct, mildly confrontational casual speech."},
    {"name_en":"Kate","role_in_scene":"friend","gender_hint":"female",
     "age_group_hint":"20s","register":"casual",
     "dominant_emotion":{"type":"sadness","intensity":2},
     "personality_traits":["apologetic","impulsive"],
     "interests_hints":[],
     "occupation_hint":"",
     "speech_style_notes":"Sheepish, quick to apologize, informal."}
  ],
  "scene": {
    "narrative_en":"Madison just found out her roommate Kate ate her leftover pizza. She's annoyed but trying to stay calm.",
    "setting":"home",
    "relationship_type":"friendship",
    "topics":["roommates","food","boundary setting"]
  },
  "dialogue_decomposed": {
    "overall_register":"casual",
    "overall_emotion":{"type":"anger","intensity":2},
    "speech_acts":["question","complaint","request"],
    "cultural_refs":[{"type":"food","term":"pizza"}]
  }
}
"""


USER_TEMPLATE: str = """Narrative: \"\"\"{narrative}\"\"\"

Speakers: {speakers}

Dialogue:
{dialogue_block}

Return the JSON object."""


def format_dialogue_block(dialogue: list[str], speakers: list[str]) -> str:
    """Render parallel dialogue/speakers arrays into a numbered block."""
    lines = [
        f"{i}. {sp}: {utt}"
        for i, (sp, utt) in enumerate(zip(speakers, dialogue, strict=False), start=1)
    ]
    return "\n".join(lines)
