#!/usr/bin/env python3
"""Generate Neuro-Luminance and LUMIT logos using Qwen Image."""

import sys
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import replicate
import requests

LOGOS_DIR = Path(__file__).parent / ".logos"
LOGOS_DIR.mkdir(exist_ok=True)

LOGOS = [
    {
        "name": "neuro_luminance_light",
        "prompt": """Logo for 'Neuro-Luminance' brain health clinic on transparent background. Top-down brain silhouette in deep navy blue, minimal detail, recognizable shape. From the center of the brain, a warm amber-to-gold gradient light source radiates outward - not cartoonish rays, but a soft photon-scatter effect like infrared light diffusing through tissue. The light should feel like it's coming from within the brain, not shining on it. Neural pathway lines in thin gold trace from the light source toward the outer edges of the brain, suggesting activation spreading outward.
Text 'NEURO-LUMINANCE' below in clean geometric sans-serif (think Montserrat or similar), 'NEURO-' in the same deep navy as the brain, '-LUMINANCE' in the amber-gold gradient matching the light source. Subtle letter-spacing for elegance. Small caps tagline 'BRAIN HEALTH CENTERS' beneath in 40% gray, understated.
Overall: medical credibility meets warmth and hope. The feeling of light reaching places that were dark. Works at business card size. Transparent PNG with alpha channel.""",
    },
    {
        "name": "neuro_luminance_dark",
        "prompt": """Logo for 'Neuro-Luminance' brain health clinic on transparent background, designed for dark surfaces. Top-down brain silhouette rendered in soft silver with subtle blue undertone - not harsh white, more like moonlight on skin. The brain has a faint outer glow so it doesn't disappear on pure black backgrounds.
From the brain's center, warm amber-gold light radiates outward with more intensity than the light-mode version - here the glow can really bloom since the background is dark. The light diffuses like infrared wavelengths scattering through tissue, slightly more dramatic luminosity. Thin neural pathway traces in bright gold arc from center toward the cortex edges, suggesting healing activation.
Text 'NEURO-LUMINANCE' below in clean geometric sans-serif, 'NEURO-' in silver-white matching the brain, '-LUMINANCE' in glowing amber-gold gradient with soft outer glow for legibility. Small caps tagline 'BRAIN HEALTH CENTERS' beneath in 60% warm gray.
Overall: same brand, but the light pops against darkness - metaphor intentional. The feeling of illumination in a dark place. Transparent PNG with alpha channel.""",
    },
    {
        "name": "lumit_light",
        "prompt": """Wordmark logo for 'LUMIT' medical laser therapy on transparent background, designed for light surfaces. Bold geometric sans-serif letters, substantial weight, in deep charcoal gray (#2D2D2D).
The letter 'I' is transformed: the vertical stroke is a beam of warm amber-to-gold gradient light, and the dot of the 'I' is replaced by a small stylized brain icon viewed from above - the beam appears to be penetrating downward into the brain. The brain icon is tiny but recognizable, rendered in the same charcoal with a subtle amber glow where the beam enters.
Beneath the wordmark in small lightweight text: 'MULTI-WATT INFRARED THERAPY' in 50% gray, generous letter-spacing.
Overall: clinical precision meets the warmth of healing light. The 'I' tells the whole story - light entering brain. Simple enough to embroider on scrubs or etch on equipment. Transparent PNG.""",
    },
    {
        "name": "lumit_dark",
        "prompt": """Wordmark logo for 'LUMIT' medical laser therapy on transparent background, designed for dark surfaces. Bold geometric sans-serif letters, substantial weight, in off-white (#F5F5F5) with a barely-perceptible warm tint.
The letter 'I' transformed: the vertical stroke is a beam of warm amber-gold light with visible glow/bloom effect against the dark background - the light should feel active, almost pulsing. The dot of the 'I' replaced by small top-down brain icon in silver-white, with the amber beam appearing to enter the brain from above. Where beam meets brain, a subtle corona of amber light suggests penetration and activation.
Beneath the wordmark: 'MULTI-WATT INFRARED THERAPY' in 60% warm gray, generous letter-spacing, slight glow for legibility on very dark backgrounds.
Overall: the beam of light becomes the hero - dramatic, hopeful, technological. You immediately understand 'light goes into brain.' Scales down to app icon (just the illuminated 'I' with brain dot). Transparent PNG with alpha channel.""",
    },
]


def generate_logo(logo: dict, run_number: int) -> Path:
    """Generate a single logo."""
    client = replicate.Client(timeout=120)

    timestamp = datetime.now().strftime("%H%M%S")
    filename = f"{logo['name']}_run{run_number}_{timestamp}.png"
    output_path = LOGOS_DIR / filename

    print(f"  Generating {logo['name']}...", file=sys.stderr)

    output = client.run(
        "qwen/qwen-image-2512",
        input={
            "prompt": logo["prompt"],
            "aspect_ratio": "1:1",
            "output_format": "png",
            "go_fast": False,
            "guidance": 4,
            "num_inference_steps": 40,
        },
    )

    # Handle output
    if isinstance(output, list):
        image_url = output[0]
    elif hasattr(output, "url"):
        image_url = output.url
    else:
        image_url = str(output)

    # Download
    response = requests.get(image_url, timeout=(5, 60))
    response.raise_for_status()
    output_path.write_bytes(response.content)

    print(f"    Saved: {output_path}", file=sys.stderr)
    return output_path


def main():
    run_number = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print(f"\n=== Logo Generation Run {run_number} ===\n", file=sys.stderr)

    for logo in LOGOS:
        try:
            generate_logo(logo, run_number)
        except Exception as e:
            print(f"  ERROR generating {logo['name']}: {e}", file=sys.stderr)

    print(f"\nRun {run_number} complete. Logos saved to: {LOGOS_DIR}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
