# v004_narrative_depth

**Hypothesis**: Fewer scenes (12-13 vs 15+) with more room per scene produces better patient explainer quality. Each scene gets 55-75 words instead of 45-65, allowing deeper interpretation and analogy use while staying within the same total budget.

**Changes from v001**:
- Scene target reduced from ~15 to 12-13
- Per-scene word budget increased (55-75 for content scenes)
- Same total word budget (950-1,100)
- Added "prioritize depth over breadth" instruction
- Explicit permission to pick 6-8 strongest findings and skip the rest
- Cut weakest scene entirely rather than trimming all scenes thin
- All visual prompt instructions identical to v001

**Trade-off being tested**: Breadth (more scenes, thinner coverage) vs depth (fewer scenes, richer interpretation per finding).
