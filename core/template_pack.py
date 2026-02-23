"""Deterministic non-brain scene rendering using patient-derived template pack assets."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image, ImageDraw, ImageFilter, ImageFont

TARGET_WIDTH = 1664
TARGET_HEIGHT = 928

_TEMPLATE_MARKER_VERSION = 1


@dataclass(frozen=True)
class TemplatePackEntry:
    """Single template image entry loaded from manifest."""

    template_id: str
    archetype: str
    path: Path
    source_project: str
    source_scene_id: int


@dataclass(frozen=True)
class TemplatePackManifest:
    """Manifest object for deterministic template selection."""

    version: int
    entries: tuple[TemplatePackEntry, ...]


def default_template_pack_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "templates" / "patient_pack"


def default_template_manifest_path() -> Path:
    return default_template_pack_dir() / "manifest.json"


def load_template_manifest(path: Path | None = None) -> TemplatePackManifest:
    """Load template pack manifest; return empty manifest when missing."""
    manifest_path = path or default_template_manifest_path()
    if not manifest_path.exists():
        return TemplatePackManifest(version=_TEMPLATE_MARKER_VERSION, entries=tuple())

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries: list[TemplatePackEntry] = []
    base = manifest_path.parent
    for item in raw.get("entries") or []:
        rel_or_abs = str(item.get("path") or "").strip()
        if not rel_or_abs:
            continue
        candidate = Path(rel_or_abs)
        if not candidate.is_absolute():
            candidate = (base / candidate).resolve()
        entries.append(
            TemplatePackEntry(
                template_id=str(item.get("template_id") or "").strip() or candidate.stem,
                archetype=str(item.get("archetype") or "general_panel").strip().lower(),
                path=candidate,
                source_project=str(item.get("source_project") or ""),
                source_scene_id=int(item.get("source_scene_id") or 0),
            )
        )

    return TemplatePackManifest(version=int(raw.get("version") or _TEMPLATE_MARKER_VERSION), entries=tuple(entries))


def classify_scene_archetype(scene: Mapping[str, Any]) -> str:
    """Classify a non-brain scene into a deterministic template archetype."""
    text = " ".join(
        str(scene.get(k) or "")
        for k in ("title", "subtitle", "visual_prompt", "narration", "footer", "scene_type")
    ).lower()

    # Most specific first.
    if any(k in text for k in ("timeline", "roadmap", "phase", "session", "milestone", "journey")):
        return "timeline_panel"
    if any(k in text for k in ("compare", "versus", "vs", "before", "after", "left", "right", "split")):
        return "split_compare"
    if any(k in text for k in ("network", "connectivity", "coherence", "pair", "edge", "pathway")):
        return "network_panel"
    if any(k in text for k in ("trend", "increase", "decrease", "shift", "evolution", "trajectory")):
        return "trend_panel"
    if any(k in text for k in ("gauge", "meter", "speed", "target", "zone", "kpi", "metric", "score")):
        return "metric_dashboard"

    if _extract_numeric_tokens(text):
        return "metric_dashboard"

    return "general_panel"


def select_template_for_scene(
    scene: Mapping[str, Any],
    manifest: TemplatePackManifest,
) -> TemplatePackEntry | None:
    """Select template deterministically by archetype and scene fingerprint."""
    if not manifest.entries:
        return None

    explicit_id = str(scene.get("template_id") or "").strip()
    if explicit_id:
        for entry in manifest.entries:
            if entry.template_id == explicit_id:
                return entry

    archetype = str(scene.get("template_archetype") or "").strip().lower() or classify_scene_archetype(scene)

    candidates = [entry for entry in manifest.entries if entry.archetype == archetype]
    if not candidates:
        candidates = [entry for entry in manifest.entries if entry.archetype == "general_panel"]
    if not candidates:
        candidates = list(manifest.entries)

    key = "::".join(
        [
            str(scene.get("uid") or ""),
            str(scene.get("id") or ""),
            str(scene.get("title") or ""),
            str(scene.get("visual_prompt") or ""),
            archetype,
        ]
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    index = int(digest[:8], 16) % len(candidates)
    return candidates[index]


def render_non_brain_scene(
    scene: Mapping[str, Any],
    project_dir: Path,
    *,
    manifest_path: Path | None = None,
    fallback_mode: str | None = None,
) -> Path:
    """
    Render a deterministic non-brain scene image.

    Runtime policy:
    - prefer patient-derived template pack background
    - apply deterministic overlay text/value badges
    - fallback to deterministic "snazzy" canvas when no template pack exists
    """
    scene_id = int(scene.get("id") or 0)
    output_path = project_dir / "images" / f"scene_{scene_id:03d}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = load_template_manifest(manifest_path)
    template = select_template_for_scene(scene, manifest)

    canvas = _load_background_canvas(template)
    mode = (fallback_mode or os.getenv("NON_BRAIN_FALLBACK_MODE") or "canvas").strip().lower()

    if template is None and mode in {"canvas", "snazzy", "d3ish"}:
        canvas = _build_snazzy_canvas_background(size=canvas.size)

    rendered = _render_overlay(canvas, scene, template)
    rendered.save(output_path)
    return output_path


def _load_background_canvas(template: TemplatePackEntry | None) -> Image.Image:
    if template and template.path.exists():
        img = Image.open(template.path).convert("RGB")
        return img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)

    # Default neutral background if no template pack exists yet.
    return _build_snazzy_canvas_background(size=(TARGET_WIDTH, TARGET_HEIGHT))


def _build_snazzy_canvas_background(size: tuple[int, int]) -> Image.Image:
    """Deterministic visually-rich fallback canvas for non-brain scenes."""
    width, height = size
    bg = Image.new("RGB", (width, height), (7, 26, 44))
    draw = ImageDraw.Draw(bg)

    # Vertical gradient bands.
    for y in range(height):
        t = y / max(1, height - 1)
        r = int(6 + 10 * (1 - t))
        g = int(20 + 30 * (1 - t))
        b = int(40 + 55 * (1 - t))
        draw.line((0, y, width, y), fill=(r, g, b))

    # Add subtle radial glows for depth.
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    o = ImageDraw.Draw(overlay)
    centers = [
        (int(width * 0.2), int(height * 0.25), 380, (255, 165, 80, 42)),
        (int(width * 0.72), int(height * 0.35), 430, (80, 190, 255, 36)),
        (int(width * 0.5), int(height * 0.78), 300, (120, 255, 210, 26)),
    ]
    for cx, cy, radius, color in centers:
        o.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=24))
    bg = Image.alpha_composite(bg.convert("RGBA"), overlay).convert("RGB")

    # "Glass" card region where text and values are placed.
    card = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(card)
    margin = 52
    d.rounded_rectangle(
        (margin, 128, width - margin, height - 52),
        radius=28,
        fill=(8, 23, 44, 188),
        outline=(116, 187, 255, 120),
        width=2,
    )
    card = card.filter(ImageFilter.GaussianBlur(radius=0.3))
    return Image.alpha_composite(bg.convert("RGBA"), card).convert("RGB")


def _render_overlay(
    image: Image.Image,
    scene: Mapping[str, Any],
    template: TemplatePackEntry | None,
) -> Image.Image:
    out = image.convert("RGBA")
    draw = ImageDraw.Draw(out)

    title = str(scene.get("title") or f"Scene {scene.get('id', '')}").strip() or "Scene"
    subtitle = str(scene.get("subtitle") or "").strip()
    if not subtitle:
        subtitle = _first_summary_line(scene)

    body_lines = _body_lines(scene)
    values = _extract_numeric_tokens(" ".join(body_lines + [str(scene.get("visual_prompt") or "")]))

    font_title = _load_font(size=64, bold=True)
    font_subtitle = _load_font(size=38, bold=False)
    font_body = _load_font(size=32, bold=False)
    font_badge = _load_font(size=30, bold=True)

    # Header plate
    draw.rounded_rectangle((62, 34, out.width - 62, 118), radius=20, fill=(8, 24, 43, 205), outline=(132, 206, 255, 160), width=2)
    draw.text((84, 45), _truncate(title, 72), font=font_title, fill=(232, 243, 255, 255))

    if subtitle:
        draw.text((84, 134), _truncate(subtitle, 88), font=font_subtitle, fill=(182, 217, 248, 255))

    y = 204
    max_lines = 8
    for line in body_lines[:max_lines]:
        draw.text((102, y), line, font=font_body, fill=(218, 235, 247, 245))
        y += 54

    # Deterministic metric badges near bottom.
    badge_y = out.height - 148
    badge_x = 92
    badge_h = 56
    for token in values[:6]:
        text = token.strip()
        if not text:
            continue
        tw = draw.textlength(text, font=font_badge)
        badge_w = int(max(140, tw + 52))
        draw.rounded_rectangle(
            (badge_x, badge_y, badge_x + badge_w, badge_y + badge_h),
            radius=18,
            fill=(17, 54, 82, 208),
            outline=(138, 208, 255, 175),
            width=2,
        )
        draw.text((badge_x + 24, badge_y + 11), text, font=font_badge, fill=(224, 244, 255, 255))
        badge_x += badge_w + 16

    if template:
        credit = f"template: {template.template_id}"
        draw.text((out.width - 380, out.height - 44), credit, font=_load_font(size=20, bold=False), fill=(145, 178, 205, 180))

    return out.convert("RGB")


def _load_font(*, size: int, bold: bool) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates: list[Path] = []

    repo_root = Path(__file__).resolve().parent.parent
    candidates.extend(
        [
            repo_root / "templates" / "fonts" / ("NotoSans-Bold.ttf" if bold else "NotoSans-Regular.ttf"),
            repo_root / "blender_pipeline" / "assets" / "fonts" / "NotoSans-Regular.ttf",
            Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
            Path("/Library/Fonts/Arial Bold.ttf"),
            Path("/Library/Fonts/Arial.ttf"),
        ]
    )

    for path in candidates:
        try:
            if path.exists():
                return ImageFont.truetype(str(path), size=size)
        except Exception:
            continue

    return ImageFont.load_default()


def _truncate(text: str, max_chars: int) -> str:
    t = " ".join(text.split())
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1].rstrip() + "..."


def _first_summary_line(scene: Mapping[str, Any]) -> str:
    prompt = str(scene.get("visual_prompt") or "")
    if not prompt:
        return ""
    parts = re.split(r"[\n.;]", prompt)
    for part in parts:
        s = " ".join(part.split())
        if s:
            return _truncate(s, 88)
    return ""


def _body_lines(scene: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []

    visual_prompt = str(scene.get("visual_prompt") or "")
    for part in re.split(r"[\n;]", visual_prompt):
        s = " ".join(part.split())
        if not s:
            continue
        lines.append(_truncate(s, 88))
        if len(lines) >= 8:
            break

    if not lines:
        narration = str(scene.get("narration") or "")
        for sentence in re.split(r"[.!?]", narration):
            s = " ".join(sentence.split())
            if not s:
                continue
            lines.append(_truncate(s, 88))
            if len(lines) >= 8:
                break

    return lines


def _extract_numeric_tokens(text: str) -> list[str]:
    # Includes common qEEG units and percentages.
    pattern = re.compile(
        r"(?<!\w)-?\d+(?:\.\d+)?\s*(?:%|hz|ms|uv|\u00b5v|sec|s)?",
        flags=re.IGNORECASE,
    )
    tokens = []
    seen = set()
    for match in pattern.finditer(text or ""):
        token = " ".join(match.group(0).split())
        if not token or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def write_template_manifest(path: Path, entries: Iterable[TemplatePackEntry]) -> None:
    """Utility used by extraction scripts to save manifest."""
    payload = {
        "version": _TEMPLATE_MARKER_VERSION,
        "entries": [
            {
                "template_id": e.template_id,
                "archetype": e.archetype,
                "path": str(e.path),
                "source_project": e.source_project,
                "source_scene_id": e.source_scene_id,
            }
            for e in entries
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
