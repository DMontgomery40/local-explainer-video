# Brain Region Mapping (10-20 System)

The `brain_region_focus` composition template places region labels at anatomical positions on a topdown brain image (1664x928 pixels, brain centered at approximately 832, 400).

## Supported Regions

### Lobe-level regions
| Name | Position | Description |
|------|----------|-------------|
| frontal | upper-center | Frontal lobe mass |
| prefrontal | top | Forehead area, very top of brain |
| central | mid-brain | Central sulcus area |
| parietal | lower-center | Behind central sulcus |
| central-parietal | between central and parietal | Transition zone |
| temporal | left side | Temporal lobe (lateral) |
| occipital | bottom-back | Lowest visible lobe |

### Individual 10-20 electrodes
| Electrode | Position | Description |
|-----------|----------|-------------|
| Fp1, Fp2 | top-left, top-right | Frontal poles |
| F3, F4 | upper-left, upper-right | Frontal sites |
| Fz | upper-center midline | Midline frontal |
| C3, C4 | mid-left, mid-right | Central sites |
| Cz | center midline | Vertex |
| T3, T4 | far-left, far-right | Temporal sites |
| P3, P4 | lower-left, lower-right | Parietal sites |
| Pz | lower-center midline | Midline parietal |
| O1, O2 | bottom-left, bottom-right | Occipital sites |
| Oz | bottom-center midline | Midline occipital |

## Usage in storyboard

Use plain region names in the `regions` array. The renderer fuzzy-matches names to positions:

```json
{
  "regions": [
    { "name": "Frontal", "value": "+15%", "status": "improved" },
    { "name": "F3", "value": "12.5 µV²", "status": "stable" },
    { "name": "Occipital", "value": "-8%", "status": "declined" }
  ]
}
```

## Status colors

- `improved` — teal (#5eead4) — positive change
- `stable` — blue (#60a5fa) — no significant change
- `declined` — amber (#fbbf24) — negative change
- `flagged` — red (#f87171) — requires attention

## Background images

Available brain backgrounds:
- `brain_region_focus_topdown.png` — standard topdown view (default)
- `brain_region_focus_frontal.png` — frontal perspective
- `brain_region_focus_heatmap.png` — heatmap style
- `brain_region_focus.png` — simplified view

When the director omits `backgroundImage`, the template defaults to `brain_region_focus_topdown.png`.
