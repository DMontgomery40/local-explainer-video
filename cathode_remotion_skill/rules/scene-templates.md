# Scene Template Catalog

Each scene in the storyboard must specify `composition.family` (one of the names below) and `composition.props` (the props object for that family).

## cover_hook

Opening title card with headline, optional subtitle and kicker.

```json
{
  "headline": "A Brain's Journey Back",
  "subtitle": "Following Recovery Through Real-Time Data",
  "kicker": "QEEG ANALYSIS",
  "backgroundImage": "cover_hook.png"
}
```

Required: `headline`
Optional: `subtitle`, `kicker`, `backgroundImage`

---

## brain_region_focus

Topdown brain diagram with labeled electrode regions. Each region is placed at its anatomical 10-20 position automatically.

```json
{
  "headline": "Region Activity Changes",
  "regions": [
    { "name": "Frontal", "value": "+15%", "status": "improved" },
    { "name": "Parietal", "value": "stable", "status": "stable" },
    { "name": "Temporal", "value": "-8%", "status": "declined" }
  ],
  "caption": "Session 1 vs Session 3",
  "backgroundImage": "brain_region_focus_topdown.png"
}
```

Required: `headline`, `regions` (array of `{name, value?, status?}`)
Optional: `caption`, `backgroundImage`

Status colors: `improved` = teal, `stable` = blue, `declined` = amber, `flagged` = red

Supported region names (fuzzy matched): frontal, prefrontal, central, parietal, central-parietal, temporal, occipital, fp1, fp2, f3, f4, fz, c3, c4, cz, t3, t4, p3, p4, pz, o1, o2, oz

---

## metric_card

Before/after display for a single metric with optional delta.

```json
{
  "headline": "Executive Function",
  "metricName": "Trail Making Test B",
  "beforeValue": "161s",
  "afterValue": "80s",
  "delta": "50% improvement",
  "caption": "Measured across 3 sessions"
}
```

Required: `headline`, `beforeValue`, `afterValue`
Optional: `metricName`, `delta`, `caption`, `backgroundImage`

---

## metric_comparison

Side-by-side comparison of two groups, sessions, or conditions.

```json
{
  "headline": "Session Comparison",
  "leftLabel": "Session 1",
  "rightLabel": "Session 3",
  "leftValue": "4.8",
  "rightValue": "2.1",
  "caption": "Theta/Beta Ratio"
}
```

Required: `headline`, `leftLabel`, `rightLabel`, `leftValue`, `rightValue`
Optional: `caption`, `backgroundImage`

---

## timeline_progression

Horizontal timeline with labeled markers for sessions or dates.

```json
{
  "headline": "Treatment Timeline",
  "markers": [
    { "label": "June", "sublabel": "Baseline" },
    { "label": "September", "sublabel": "Mid-treatment" },
    { "label": "November", "sublabel": "Follow-up" }
  ]
}
```

Required: `headline`, `markers` (array of `{label, sublabel?}`)
Optional: `backgroundImage`

---

## bullet_stack

Numbered list for roadmaps, key findings, or sequential items.

```json
{
  "headline": "Key Findings",
  "items": [
    "Cognitive function improved significantly",
    "Brain adapted through neurocompensation",
    "Session 3 anomaly explained by drowsiness",
    "Overall trajectory strongly positive"
  ],
  "caption": "Summary of evidence"
}
```

Required: `headline`, `items` (string array)
Optional: `caption`, `backgroundImage`

---

## data_stage

Horizontal bar chart for ranked/quantitative data.

```json
{
  "headline": "Alpha Power by Region",
  "items": [
    { "label": "Frontal", "value": 12.5, "displayValue": "12.5 µV²" },
    { "label": "Central", "value": 9.8, "displayValue": "9.8 µV²" },
    { "label": "Parietal", "value": 14.2, "displayValue": "14.2 µV²" }
  ],
  "unit": "µV²",
  "caption": "Absolute power at baseline"
}
```

Required: `headline`, `items` (array of `{label, value (number), displayValue?}`)
Optional: `unit`, `caption`, `backgroundImage`

---

## analogy_split

Two-panel metaphor comparison (software vs hardware, before vs after concepts).

```json
{
  "headline": "Software vs Hardware",
  "left": {
    "title": "Software (Function)",
    "items": ["Task switching", "Working memory", "Executive function"],
    "accent": "teal"
  },
  "right": {
    "title": "Hardware (Biology)",
    "items": ["Alpha frequency", "Processing speed", "Raw neural power"],
    "accent": "blue"
  },
  "caption": "Two sides of the same coin"
}
```

Required: `headline`, `left{title, items[]}`, `right{title, items[]}`
Optional: `left.accent`, `right.accent`, `caption`, `backgroundImage`

Accent colors: `teal`, `amber`, `blue`, `green`

---

## closing_cta

Final recommendation with optional bullet points and sign-off.

```json
{
  "headline": "What This Means For You",
  "bullets": [
    "Continue current treatment protocol",
    "Schedule follow-up in 8 weeks",
    "Track sleep quality between sessions"
  ],
  "signoff": "We're excited about this progress and look forward to seeing even more gains ahead."
}
```

Required: `headline`
Optional: `bullets` (string array), `signoff`, `backgroundImage`

---

## narration_slide

Generic fallback for scenes where no structured template fits. Text over a background.

```json
{
  "headline": "Understanding Your Results",
  "body": "Let's walk through what your brain data is telling us...",
  "caption": "Section overview"
}
```

Required: `headline`
Optional: `body`, `caption`, `backgroundImage`
