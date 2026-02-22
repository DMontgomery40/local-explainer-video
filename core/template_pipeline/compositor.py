"""Deterministic text compositor for template backgrounds."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont

try:
    import cairo  # type: ignore

    _HAS_CAIRO = True
except Exception:
    cairo = None
    _HAS_CAIRO = False


TARGET_WIDTH = 1664
TARGET_HEIGHT = 928
DEFAULT_BACKEND = "pillow"


class CompositorError(ValueError):
    """Raised for deterministic compositor failures."""


@dataclass(frozen=True)
class TextStyle:
    font_family: str
    font_weight: str
    font_size: int
    min_font_size: int
    color: str
    tracking: float
    shadow: dict[str, Any] | None
    glow: dict[str, Any] | None
    line_height: float


@dataclass(frozen=True)
class AnchorDef:
    anchor_id: str
    box: tuple[int, int, int, int]
    align: str
    valign: str
    binding: dict[str, Any]
    style: TextStyle


@dataclass(frozen=True)
class BarOverlayDef:
    overlay_id: str
    box: tuple[int, int, int, int]
    source: str
    color: str
    base_alpha: float


@dataclass(frozen=True)
class AnchorLayout:
    template_id: str
    dimensions: tuple[int, int]
    anchors: tuple[AnchorDef, ...]
    bars: tuple[BarOverlayDef, ...]


def _parse_box(raw: Any) -> tuple[int, int, int, int]:
    if not isinstance(raw, list) or len(raw) != 4:
        raise CompositorError("box must be [x, y, w, h]")
    x, y, w, h = [int(v) for v in raw]
    if w <= 0 or h <= 0:
        raise CompositorError(f"Invalid box dimensions: {raw}")
    return (x, y, w, h)


def _parse_style(raw: dict[str, Any]) -> TextStyle:
    style = raw or {}
    return TextStyle(
        font_family=str(style.get("font_family") or "Inter"),
        font_weight=str(style.get("font_weight") or "bold").lower(),
        font_size=int(style.get("font_size") or 52),
        min_font_size=int(style.get("min_font_size") or 18),
        color=str(style.get("color") or "#FFFFFF"),
        tracking=float(style.get("tracking") or 0.0),
        shadow=style.get("shadow") if isinstance(style.get("shadow"), dict) else None,
        glow=style.get("glow") if isinstance(style.get("glow"), dict) else None,
        line_height=float(style.get("line_height") or 1.1),
    )


def _load_anchor_layout(path: Path) -> AnchorLayout:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CompositorError(f"Failed parsing anchor layout: {path}") from exc

    template_id = str(payload.get("template_id") or "").strip()
    if not template_id:
        raise CompositorError(f"{path}: template_id is required")

    dims = payload.get("dimensions") or [TARGET_WIDTH, TARGET_HEIGHT]
    if not isinstance(dims, list) or len(dims) != 2:
        raise CompositorError(f"{path}: dimensions must be [w, h]")
    dimensions = (int(dims[0]), int(dims[1]))

    anchors_raw = payload.get("anchors")
    if not isinstance(anchors_raw, list):
        raise CompositorError(f"{path}: anchors must be a list")

    anchors: list[AnchorDef] = []
    for idx, raw in enumerate(anchors_raw):
        if not isinstance(raw, dict):
            raise CompositorError(f"{path}: anchors[{idx}] must be an object")
        anchor_id = str(raw.get("id") or "").strip()
        if not anchor_id:
            raise CompositorError(f"{path}: anchors[{idx}] missing id")
        box = _parse_box(raw.get("box"))
        align = str(raw.get("align") or "center").lower()
        valign = str(raw.get("valign") or "middle").lower()
        binding = raw.get("binding")
        if not isinstance(binding, dict):
            raise CompositorError(f"{path}: anchor {anchor_id} missing binding object")
        anchors.append(
            AnchorDef(
                anchor_id=anchor_id,
                box=box,
                align=align,
                valign=valign,
                binding=binding,
                style=_parse_style(raw.get("style") if isinstance(raw.get("style"), dict) else {}),
            )
        )

    bars_raw = payload.get("bars") or []
    if not isinstance(bars_raw, list):
        raise CompositorError(f"{path}: bars must be a list")
    bars: list[BarOverlayDef] = []
    for idx, raw in enumerate(bars_raw):
        if not isinstance(raw, dict):
            raise CompositorError(f"{path}: bars[{idx}] must be an object")
        overlay_id = str(raw.get("id") or f"bar_{idx+1}")
        source = str(raw.get("source") or "").strip()
        if not source:
            raise CompositorError(f"{path}: bars[{idx}] missing source path")
        bars.append(
            BarOverlayDef(
                overlay_id=overlay_id,
                box=_parse_box(raw.get("box")),
                source=source,
                color=str(raw.get("color") or "#8CC8FF"),
                base_alpha=float(raw.get("base_alpha") or 0.22),
            )
        )

    return AnchorLayout(
        template_id=template_id,
        dimensions=dimensions,
        anchors=tuple(anchors),
        bars=tuple(bars),
    )


def _flatten_dict(payload: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(payload, dict):
        for k, v in payload.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out[key] = v
            out.update(_flatten_dict(v, key))
    elif isinstance(payload, list):
        for i, v in enumerate(payload):
            key = f"{prefix}.{i}" if prefix else str(i)
            out[key] = v
            out.update(_flatten_dict(v, key))
    return out


def _resolve_path(payload: Any, path: str) -> Any:
    cur: Any = payload
    for token in path.split("."):
        if not token:
            continue
        if isinstance(cur, dict):
            if token not in cur:
                return None
            cur = cur[token]
            continue
        if isinstance(cur, list):
            try:
                idx = int(token)
            except Exception:
                return None
            if idx < 0 or idx >= len(cur):
                return None
            cur = cur[idx]
            continue
        return None
    return cur


_PLACEHOLDER_RE = re.compile(r"\{([^{}]+)\}")


def _render_template_text(template: str, payload: dict[str, Any]) -> str:
    def _replace(match: re.Match[str]) -> str:
        expr = match.group(1).strip()
        if not expr:
            return ""
        value = _resolve_path(payload, expr)
        if value is None:
            return ""
        if isinstance(value, float):
            if abs(value) >= 1000:
                return f"{value:,.1f}"
            if value.is_integer():
                return str(int(value))
            return f"{value:.1f}".rstrip("0").rstrip(".")
        if isinstance(value, list):
            return ", ".join(str(v) for v in value)
        return str(value)

    return _PLACEHOLDER_RE.sub(_replace, template)


def _binding_text(binding: dict[str, Any], payload: dict[str, Any]) -> str:
    mode = str(binding.get("mode") or "path").lower()
    fallback = str(binding.get("fallback") or "")
    if mode == "constant":
        return str(binding.get("value") or fallback)
    if mode == "template":
        template = str(binding.get("template") or "")
        rendered = _render_template_text(template, payload).strip()
        return rendered or fallback
    if mode == "path":
        path = str(binding.get("path") or "").strip()
        if not path:
            return fallback
        value = _resolve_path(payload, path)
        if value is None:
            return fallback
        if isinstance(value, float):
            fmt = str(binding.get("format") or "")
            if fmt:
                try:
                    return format(value, fmt)
                except Exception:
                    pass
            if value.is_integer():
                return str(int(value))
            return f"{value:.1f}".rstrip("0").rstrip(".")
        if isinstance(value, list):
            return ", ".join(str(v) for v in value)
        return str(value)
    if mode == "join_path":
        path = str(binding.get("path") or "").strip()
        if not path:
            return fallback
        value = _resolve_path(payload, path)
        if not isinstance(value, list):
            return fallback
        field = str(binding.get("field") or "").strip()
        sep = str(binding.get("separator") or "\n")
        prefix = str(binding.get("prefix") or "")
        max_items = int(binding.get("max_items") or len(value))
        items: list[str] = []
        for entry in value[:max_items]:
            if field and isinstance(entry, dict):
                raw = entry.get(field)
            else:
                raw = entry
            if raw is None:
                continue
            text = str(raw).strip()
            if not text:
                continue
            items.append(f"{prefix}{text}" if prefix else text)
        return sep.join(items) if items else fallback
    if mode == "first_non_empty":
        paths = binding.get("paths") or []
        if isinstance(paths, list):
            for p in paths:
                path = str(p).strip()
                if not path:
                    continue
                value = _resolve_path(payload, path)
                if value is None:
                    continue
                text = str(value).strip()
                if text:
                    return text
        return fallback
    if mode == "summary":
        path = str(binding.get("path") or "structured_data").strip()
        value = _resolve_path(payload, path)
        if value is None:
            return fallback
        max_lines = max(1, int(binding.get("max_lines") or 8))
        max_chars = max(40, int(binding.get("max_chars") or 260))
        lines: list[str] = []

        def _friendly_key(key: str) -> str:
            text = str(key).replace("_", " ").strip()
            if text.endswith(" label"):
                text = text[: -len(" label")]
            return text

        def _push(text: str) -> None:
            cleaned = " ".join(text.split()).strip()
            if not cleaned:
                return
            if len(cleaned) > max_chars:
                cleaned = cleaned[: max_chars - 1].rstrip() + "…"
            lines.append(cleaned)

        def _fmt_scalar(raw: Any) -> str:
            if isinstance(raw, float):
                if raw.is_integer():
                    return str(int(raw))
                if abs(raw) >= 1000:
                    return f"{raw:,.1f}"
                return f"{raw:.1f}".rstrip("0").rstrip(".")
            return str(raw).strip()

        def _fmt_entry(item: dict[str, Any], parent_key: str = "") -> str:
            scalar: dict[str, Any] = {}
            for sk, sv in item.items():
                if isinstance(sv, (str, int, float)) and str(sv).strip():
                    scalar[str(sk)] = sv
            if not scalar:
                return ""

            unit = _fmt_scalar(scalar.get("unit")) if "unit" in scalar else ""
            lead_key = next(
                (k for k in ("label", "name", "id", "session", "metric") if k in scalar and _fmt_scalar(scalar[k])),
                "",
            )
            lead = _fmt_scalar(scalar.get(lead_key)) if lead_key else ""

            if "kind" in scalar and "severity" in scalar and len(scalar) <= 4:
                kind = _friendly_key(_fmt_scalar(scalar["kind"]))
                severity = _fmt_scalar(scalar["severity"])
                return f"{kind} ({severity})"

            if "from" in scalar and "to" in scalar:
                trend = f"{_fmt_scalar(scalar['from'])} -> {_fmt_scalar(scalar['to'])}"
                if unit:
                    trend = f"{trend} {unit}"
                return f"{lead}: {trend}" if lead else trend

            for value_key in ("value", "mean", "score", "ratio", "min", "max"):
                if value_key in scalar and lead:
                    text = f"{lead}: {_fmt_scalar(scalar[value_key])}"
                    if unit:
                        text = f"{text} {unit}"
                    return text

            if "from" in scalar and lead:
                text = f"{lead}: {_fmt_scalar(scalar['from'])}"
                if unit:
                    text = f"{text} {unit}"
                return text

            if lead and len(scalar) == 1:
                return lead

            parts: list[str] = []
            if lead:
                parts.append(lead)
            for sk, sv in scalar.items():
                if sk == lead_key or sk == "unit":
                    continue
                val = _fmt_scalar(sv)
                if sk == "kind":
                    val = _friendly_key(val)
                parts.append(f"{_friendly_key(sk)} {val}")
                if len(parts) >= 3:
                    break
            if not parts:
                return ""
            text = "; ".join(parts)
            if unit and "from" not in scalar and "to" not in scalar:
                text = f"{text} {unit}"
            return text

        if isinstance(value, dict):
            for k, v in value.items():
                if len(lines) >= max_lines:
                    break
                fk = _friendly_key(k)
                if isinstance(v, list):
                    if not v:
                        continue
                    if isinstance(v[0], dict):
                        _push(f"{fk}:")
                        for item in v[: max(1, min(4, max_lines - len(lines)))]:
                            if len(lines) >= max_lines:
                                break
                            text = _fmt_entry(item, parent_key=fk)
                            if text:
                                _push(f"• {text}")
                    else:
                        joined = ", ".join(_fmt_scalar(x) for x in v[:6])
                        _push(f"{fk}: {joined}")
                elif isinstance(v, dict):
                    entry = _fmt_entry(v, parent_key=fk)
                    if entry:
                        _push(f"{fk}: {entry}")
                    else:
                        _push(f"{fk}:")
                else:
                    _push(f"{fk}: {_fmt_scalar(v)}")
        elif isinstance(value, list):
            for i, item in enumerate(value[:max_lines], start=1):
                if isinstance(item, dict):
                    text = _fmt_entry(item)
                    _push(f"• {text}" if text else f"• {item}")
                else:
                    _push(f"• {_fmt_scalar(item)}")
        else:
            _push(_fmt_scalar(value))
        return "\n".join(lines) if lines else fallback
    return fallback


def _hex_to_rgba(color: str, alpha_scale: float = 1.0) -> tuple[int, int, int, int]:
    rgba = ImageColor.getcolor(color, "RGBA")
    return (rgba[0], rgba[1], rgba[2], max(0, min(255, int(rgba[3] * alpha_scale))))


def _font_search_dirs() -> tuple[Path, ...]:
    paths = [
        Path(__file__).resolve().parents[2] / "templates" / "fonts",
        Path.home() / "Library" / "Fonts",
        Path("/Library/Fonts"),
        Path("/System/Library/Fonts"),
        Path("/System/Library/Fonts/Supplemental"),
    ]
    env_dir = (os.getenv("TEMPLATE_FONT_DIR") or "").strip()
    if env_dir:
        paths.insert(0, Path(env_dir).expanduser())
    return tuple(p for p in paths if p.exists())


@lru_cache(maxsize=1)
def _font_index() -> list[Path]:
    files: list[Path] = []
    for root in _font_search_dirs():
        for ext in ("*.ttf", "*.otf", "*.ttc"):
            files.extend(root.rglob(ext))
    return files


def _font_candidates(family: str, weight: str) -> list[str]:
    fam = family.lower().strip()
    bold = weight in {"bold", "semibold", "heavy", "black"}
    names: list[str] = []
    if fam in {"inter", "inter var"}:
        names.extend(["Inter-Bold.ttf" if bold else "Inter-Regular.ttf", "Inter.ttf"])
    elif fam in {"montserrat"}:
        names.extend(
            ["Montserrat-Bold.ttf" if bold else "Montserrat-Regular.ttf", "Montserrat.ttf"]
        )
    elif fam in {"manrope"}:
        names.extend(["Manrope-Bold.ttf" if bold else "Manrope-Regular.ttf"])
    elif fam in {"source sans 3", "source sans", "sourcesans"}:
        names.extend(["SourceSans3-Bold.ttf" if bold else "SourceSans3-Regular.ttf"])
    names.extend(
        [
            "SFNS.ttf",
            "Helvetica.ttc",
            "Arial Bold.ttf" if bold else "Arial.ttf",
            "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        ]
    )
    return names


@lru_cache(maxsize=256)
def _resolve_font_path(family: str, weight: str) -> str:
    preferred = _font_candidates(family, weight)
    indexed = _font_index()
    by_name = {p.name.lower(): p for p in indexed}
    for candidate in preferred:
        hit = by_name.get(candidate.lower())
        if hit:
            return str(hit)

    fam = family.lower().replace(" ", "")
    weight_key = "bold" if weight in {"bold", "semibold", "heavy", "black"} else "regular"
    ranked: list[tuple[int, Path]] = []
    for p in indexed:
        stem = p.stem.lower().replace(" ", "")
        score = 0
        if fam and fam in stem:
            score += 10
        if weight_key == "bold" and any(k in stem for k in ("bold", "semi", "heavy", "black")):
            score += 4
        if weight_key == "regular" and "regular" in stem:
            score += 2
        if score > 0:
            ranked.append((score, p))
    if ranked:
        ranked.sort(key=lambda x: x[0], reverse=True)
        return str(ranked[0][1])

    return "DejaVuSans.ttf"


def _load_font(style: TextStyle, size_px: int) -> ImageFont.FreeTypeFont:
    path = _resolve_font_path(style.font_family, style.font_weight)
    return ImageFont.truetype(path, size=size_px)


def _measure_multiline(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, spacing: int) -> tuple[int, int]:
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="left")
    width = int(bbox[2] - bbox[0])
    height = int(bbox[3] - bbox[1])
    return width, height


def _wrap_text_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> str:
    paragraphs = text.splitlines() or [text]
    wrapped_lines: list[str] = []
    for para in paragraphs:
        words = para.split()
        if not words:
            wrapped_lines.append("")
            continue
        line = words[0]
        for word in words[1:]:
            candidate = f"{line} {word}"
            bbox = draw.textbbox((0, 0), candidate, font=font)
            if (bbox[2] - bbox[0]) <= max_width:
                line = candidate
            else:
                wrapped_lines.append(line)
                line = word
        wrapped_lines.append(line)
    return "\n".join(wrapped_lines)


def _draw_text_layer(
    layer: Image.Image,
    text: str,
    pos: tuple[float, float],
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int, int],
    *,
    spacing: int,
    tracking: float,
) -> None:
    draw = ImageDraw.Draw(layer, "RGBA")
    x, y = pos
    if abs(tracking) < 0.01:
        draw.multiline_text((x, y), text, font=font, fill=fill, spacing=spacing, align="left")
        return

    for line_idx, line in enumerate(text.splitlines()):
        cx = x
        cy = y + line_idx * int(font.size * 1.1 + spacing)
        for ch in line:
            draw.text((cx, cy), ch, font=font, fill=fill)
            ch_bbox = draw.textbbox((0, 0), ch, font=font)
            ch_w = ch_bbox[2] - ch_bbox[0]
            cx += ch_w + tracking


def _fit_font_size(
    text: str,
    style: TextStyle,
    box_w: int,
    box_h: int,
    scale: int,
) -> tuple[ImageFont.FreeTypeFont, int, str]:
    test = Image.new("RGBA", (max(4, box_w), max(4, box_h)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(test, "RGBA")
    start = max(style.min_font_size, style.font_size)
    spacing = int(max(0, (style.line_height - 1.0) * start))
    for size in range(start, style.min_font_size - 1, -1):
        font = _load_font(style, size * scale)
        spacing = int(max(0, (style.line_height - 1.0) * size * scale))
        wrapped = _wrap_text_to_width(draw, text, font, max_width=box_w * scale)
        width, height = _measure_multiline(draw, wrapped, font, spacing=spacing)
        if width <= box_w * scale and height <= box_h * scale:
            return font, spacing, wrapped
    min_font = _load_font(style, style.min_font_size * scale)
    min_spacing = int(max(0, (style.line_height - 1.0) * style.min_font_size * scale))
    wrapped = _wrap_text_to_width(draw, text, min_font, max_width=box_w * scale)
    return min_font, min_spacing, wrapped


def _render_text_anchor_pillow(base: Image.Image, anchor: AnchorDef, text: str, payload: dict[str, Any]) -> None:
    del payload  # reserved for future style overrides
    x, y, w, h = anchor.box
    if w < 2 or h < 2:
        return

    scale = 2
    text = text.strip()
    if not text:
        return
    font, spacing, wrapped_text = _fit_font_size(text, anchor.style, w, h, scale)

    overlay = Image.new("RGBA", (w * scale, h * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    text_w, text_h = _measure_multiline(draw, wrapped_text, font, spacing=spacing)

    if anchor.align == "left":
        tx = 0
    elif anchor.align == "right":
        tx = max(0, overlay.width - text_w)
    else:
        tx = max(0, (overlay.width - text_w) // 2)

    if anchor.valign == "top":
        ty = 0
    elif anchor.valign == "bottom":
        ty = max(0, overlay.height - text_h)
    else:
        ty = max(0, (overlay.height - text_h) // 2)

    main_color = _hex_to_rgba(anchor.style.color)

    # Glow pass.
    if anchor.style.glow:
        glow_color = _hex_to_rgba(str(anchor.style.glow.get("color") or "#A8D8FF66"))
        glow_radius = float(anchor.style.glow.get("radius") or 8.0) * scale
        glow_layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
        _draw_text_layer(
            glow_layer,
            wrapped_text,
            (float(tx), float(ty)),
            font,
            glow_color,
            spacing=spacing,
            tracking=anchor.style.tracking * scale,
        )
        glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=glow_radius))
        overlay.alpha_composite(glow_layer)

    # Shadow pass.
    if anchor.style.shadow:
        sh = anchor.style.shadow
        sx = float(sh.get("dx", 2.0)) * scale
        sy = float(sh.get("dy", 2.0)) * scale
        blur = float(sh.get("blur", 4.0)) * scale
        sh_color = _hex_to_rgba(str(sh.get("color") or "#00000099"))
        shadow_layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
        _draw_text_layer(
            shadow_layer,
            wrapped_text,
            (float(tx + sx), float(ty + sy)),
            font,
            sh_color,
            spacing=spacing,
            tracking=anchor.style.tracking * scale,
        )
        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=blur))
        overlay.alpha_composite(shadow_layer)

    # Main text pass.
    _draw_text_layer(
        overlay,
        wrapped_text,
        (float(tx), float(ty)),
        font,
        main_color,
        spacing=spacing,
        tracking=anchor.style.tracking * scale,
    )

    downscaled = overlay.resize((w, h), Image.Resampling.LANCZOS)
    base.alpha_composite(downscaled, dest=(x, y))


def _bars_from_layout(layout: AnchorLayout, payload: dict[str, Any]) -> list[tuple[BarOverlayDef, float]]:
    values: list[tuple[BarOverlayDef, float]] = []
    for bar in layout.bars:
        raw = _resolve_path(payload, bar.source)
        if raw is None:
            continue
        try:
            values.append((bar, float(raw)))
        except Exception:
            continue
    return values


def _render_bar_overlays(base: Image.Image, layout: AnchorLayout, payload: dict[str, Any]) -> None:
    bars = _bars_from_layout(layout, payload)
    if not bars:
        return
    peak = max(abs(v) for _, v in bars) or 1.0
    draw = ImageDraw.Draw(base, "RGBA")
    for bar, value in bars:
        x, y, w, h = bar.box
        color_rgba = _hex_to_rgba(bar.color)
        base_alpha = max(0.0, min(1.0, bar.base_alpha))
        draw.rounded_rectangle(
            [x, y, x + w, y + h],
            radius=max(4, int(w * 0.15)),
            fill=(color_rgba[0], color_rgba[1], color_rgba[2], int(255 * base_alpha)),
            outline=(255, 255, 255, 22),
            width=1,
        )
        ratio = min(1.0, max(0.0, abs(value) / peak))
        fill_h = max(2, int(h * ratio))
        fy0 = y + h - fill_h
        fill_layer = Image.new("RGBA", (w, fill_h), (0, 0, 0, 0))
        grad = Image.new("L", (1, fill_h))
        for i in range(fill_h):
            # Slight vertical glow to mimic luminous bars.
            grad.putpixel((0, i), min(255, int(170 + 85 * (i / max(1, fill_h - 1)))))
        alpha = grad.resize((w, fill_h))
        solid = Image.new("RGBA", (w, fill_h), color_rgba)
        solid.putalpha(alpha)
        fill_layer.alpha_composite(solid)
        base.alpha_composite(fill_layer, (x, fy0))


def _anchor_payload(scene: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "scene": scene,
        "scene_id": scene.get("id"),
        "title": scene.get("title"),
        "narration": scene.get("narration"),
        "scene_type": scene.get("scene_type"),
        "structured_data": scene.get("structured_data") or {},
    }
    payload["_flat"] = _flatten_dict(payload)
    return payload


def _render_with_pillow(
    template_path: Path,
    layout: AnchorLayout,
    scene: dict[str, Any],
    output_path: Path,
) -> Path:
    base = Image.open(template_path).convert("RGBA")
    if base.size != layout.dimensions:
        base = base.resize(layout.dimensions, Image.Resampling.LANCZOS)

    payload = _anchor_payload(scene)
    _render_bar_overlays(base, layout, payload)
    for anchor in layout.anchors:
        text = _binding_text(anchor.binding, payload)
        _render_text_anchor_pillow(base, anchor, text, payload)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    base.save(output_path, format="PNG")
    return output_path


def _render_with_cairo(
    template_path: Path,
    layout: AnchorLayout,
    scene: dict[str, Any],
    output_path: Path,
) -> Path:
    if not _HAS_CAIRO:
        raise CompositorError(
            "Cairo backend requested but pycairo is not installed. "
            "Install pycairo or use backend='pillow'."
        )

    # Cairo path uses Pillow bar overlays + Cairo text for evaluation parity.
    base = Image.open(template_path).convert("RGBA")
    if base.size != layout.dimensions:
        base = base.resize(layout.dimensions, Image.Resampling.LANCZOS)

    payload = _anchor_payload(scene)
    _render_bar_overlays(base, layout, payload)

    # Convert to Cairo surface (BGRA byte order).
    rgba = base.tobytes("raw", "BGRA")
    buf = bytearray(rgba)
    surface = cairo.ImageSurface.create_for_data(
        buf, cairo.FORMAT_ARGB32, base.width, base.height, base.width * 4
    )
    ctx = cairo.Context(surface)

    for anchor in layout.anchors:
        text = _binding_text(anchor.binding, payload).strip()
        if not text:
            continue
        x, y, w, h = anchor.box
        style = anchor.style
        font_size = style.font_size
        min_size = style.min_font_size
        chosen = font_size
        for size in range(font_size, min_size - 1, -1):
            ctx.select_font_face(
                style.font_family,
                cairo.FONT_SLANT_NORMAL,
                cairo.FONT_WEIGHT_BOLD
                if style.font_weight in {"bold", "semibold", "heavy", "black"}
                else cairo.FONT_WEIGHT_NORMAL,
            )
            ctx.set_font_size(size)
            ext = ctx.text_extents(text)
            if ext.width <= w and ext.height <= h:
                chosen = size
                break
        ctx.set_font_size(chosen)
        ext = ctx.text_extents(text)
        if anchor.align == "left":
            tx = x
        elif anchor.align == "right":
            tx = x + w - ext.width - ext.x_bearing
        else:
            tx = x + (w - ext.width) / 2 - ext.x_bearing
        if anchor.valign == "top":
            ty = y - ext.y_bearing
        elif anchor.valign == "bottom":
            ty = y + h - ext.height - ext.y_bearing
        else:
            ty = y + (h - ext.height) / 2 - ext.y_bearing

        if style.shadow:
            sh = style.shadow
            sh_color = _hex_to_rgba(str(sh.get("color") or "#00000099"))
            ctx.set_source_rgba(
                sh_color[0] / 255.0,
                sh_color[1] / 255.0,
                sh_color[2] / 255.0,
                sh_color[3] / 255.0,
            )
            ctx.move_to(tx + float(sh.get("dx", 2.0)), ty + float(sh.get("dy", 2.0)))
            ctx.show_text(text)

        color = _hex_to_rgba(style.color)
        ctx.set_source_rgba(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, color[3] / 255.0)
        ctx.move_to(tx, ty)
        ctx.show_text(text)

    surface.flush()
    out = Image.frombytes("RGBA", (base.width, base.height), bytes(buf), "raw", "BGRA")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path, format="PNG")
    return output_path


def compose_scene(
    *,
    template_path: Path,
    anchors_path: Path,
    scene: dict[str, Any],
    output_path: Path,
    backend: str = DEFAULT_BACKEND,
) -> Path:
    layout = _load_anchor_layout(anchors_path)
    backend_norm = str(backend or DEFAULT_BACKEND).strip().lower()
    if backend_norm == "cairo":
        return _render_with_cairo(template_path, layout, scene, output_path)
    if backend_norm == "pillow":
        return _render_with_pillow(template_path, layout, scene, output_path)
    raise CompositorError(f"Unknown compositor backend: {backend}")
