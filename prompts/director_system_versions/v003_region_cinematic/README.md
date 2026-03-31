# v003_region_cinematic

**Hypothesis**: Replace electrode-code complexity in visual prompts with richer artistic composition complexity, maintaining the total "difficulty budget" that drives Qwen's MoE quality routing.

**Changes from v001**:
- Same word count enforcement as v002 (hard per-scene limits)
- Visual prompt style section overhauled: explicit instruction to pour complexity into lighting, materials, depth, motion, and color rather than electrode-level neuroscience
- Added spatial-visual language examples for brain regions (avoiding electrode codes)
- Added instruction to keep visual prompts LONG and DETAILED (3-4 sentences minimum)
- Maintained all other visual prompt rules (self-contained, mandatory text, etc.)

**Key insight being tested**: Qwen allocates more compute/effort to complex prompts. When you simplify by removing numbers/electrode codes, quality drops. This variant keeps prompts equally complex but shifts the complexity from neuroscience specifics to cinematic composition language.
