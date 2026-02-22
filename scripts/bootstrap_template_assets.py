#!/usr/bin/env python3
"""Create starter deterministic template assets (backgrounds + anchors + manifest)."""

from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

WIDTH = 1664
HEIGHT = 928


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _bg_dir() -> Path:
    p = _repo_root() / "templates" / "backgrounds"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _anchor_dir() -> Path:
    p = _repo_root() / "templates" / "anchors"
    p.mkdir(parents=True, exist_ok=True)
    return p


def gradient_bg(c1: tuple[int, int, int], c2: tuple[int, int, int]) -> Image.Image:
    img = Image.new("RGBA", (WIDTH, HEIGHT), c1 + (255,))
    draw = ImageDraw.Draw(img, "RGBA")
    for y in range(HEIGHT):
        t = y / max(1, HEIGHT - 1)
        r = int(c1[0] * (1 - t) + c2[0] * t)
        g = int(c1[1] * (1 - t) + c2[1] * t)
        b = int(c1[2] * (1 - t) + c2[2] * t)
        draw.line([(0, y), (WIDTH, y)], fill=(r, g, b, 255))
    return img


def add_glow(img: Image.Image, xy: tuple[int, int], radius: int, color: tuple[int, int, int, int]) -> None:
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    x, y = xy
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    layer = layer.filter(ImageFilter.GaussianBlur(radius=radius * 0.45))
    img.alpha_composite(layer)


def draw_panel(img: Image.Image, box: tuple[int, int, int, int], fill: tuple[int, int, int, int], outline: tuple[int, int, int, int], radius: int = 22) -> None:
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=2)


def save_image(name: str, img: Image.Image) -> str:
    path = _bg_dir() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    return str(path.relative_to(_repo_root()))


def save_anchor(name: str, payload: dict) -> str:
    path = _anchor_dir() / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path.relative_to(_repo_root()))


def build_generic_data_panel() -> tuple[str, str]:
    img = gradient_bg((14, 22, 40), (8, 54, 86))
    add_glow(img, (300, 180), 180, (120, 190, 255, 90))
    add_glow(img, (1320, 760), 230, (255, 180, 80, 80))
    draw_panel(img, (72, 52, 1592, 166), (8, 20, 36, 130), (180, 220, 255, 80))
    draw_panel(img, (72, 198, 1592, 860), (9, 18, 32, 96), (165, 220, 255, 70))
    image_rel = save_image("generic_data_panel_v1.png", img)

    anchor = {
        "template_id": "generic_data_panel_v1",
        "dimensions": [WIDTH, HEIGHT],
        "anchors": [
            {
                "id": "scene_title",
                "box": [110, 66, 1444, 86],
                "align": "center",
                "valign": "middle",
                "binding": {
                    "mode": "first_non_empty",
                    "paths": ["scene.title", "structured_data.title", "structured_data.metric"],
                    "fallback": "Brain Signal Update",
                },
                "style": {
                    "font_family": "Inter",
                    "font_weight": "bold",
                    "font_size": 62,
                    "min_font_size": 34,
                    "color": "#F5FBFF",
                    "shadow": {"dx": 2, "dy": 2, "blur": 4, "color": "#00000099"},
                    "glow": {"radius": 6, "color": "#8EC9FF66"},
                },
            },
            {
                "id": "headline",
                "box": [120, 228, 1424, 100],
                "align": "left",
                "valign": "middle",
                "binding": {
                    "mode": "first_non_empty",
                    "paths": [
                        "structured_data.verdict_title",
                        "structured_data.headline",
                        "structured_data.metric",
                        "structured_data.interpretation",
                        "structured_data.status",
                        "scene.title",
                    ],
                },
                "style": {
                    "font_family": "Inter",
                    "font_weight": "bold",
                    "font_size": 44,
                    "min_font_size": 24,
                    "color": "#D8EEFF",
                    "shadow": {"dx": 1, "dy": 2, "blur": 3, "color": "#00000088"},
                },
            },
            {
                "id": "details",
                "box": [120, 336, 1424, 500],
                "align": "left",
                "valign": "top",
                "binding": {
                    "mode": "summary",
                    "path": "structured_data",
                    "max_lines": 10,
                    "max_chars": 180,
                },
                "style": {
                    "font_family": "Inter",
                    "font_weight": "regular",
                    "font_size": 32,
                    "min_font_size": 20,
                    "color": "#C7E0F5",
                    "line_height": 1.25,
                    "shadow": {"dx": 1, "dy": 1, "blur": 2, "color": "#00000077"},
                },
            },
        ],
    }
    anchor_rel = save_anchor("generic_data_panel_v1.json", anchor)
    return image_rel, anchor_rel


def build_title_card() -> tuple[str, str]:
    img = gradient_bg((8, 16, 30), (20, 40, 70))
    add_glow(img, (832, 420), 260, (110, 170, 255, 105))
    add_glow(img, (260, 760), 190, (255, 186, 94, 80))
    add_glow(img, (1420, 140), 140, (120, 200, 255, 66))
    draw_panel(img, (110, 170, 1554, 740), (9, 16, 30, 110), (180, 225, 255, 70), radius=28)
    image_rel = save_image("title_card_v1.png", img)

    anchor = {
        "template_id": "title_card_v1",
        "dimensions": [WIDTH, HEIGHT],
        "anchors": [
            {
                "id": "title",
                "box": [120, 260, 1424, 220],
                "align": "center",
                "valign": "middle",
                "binding": {
                    "mode": "first_non_empty",
                    "paths": ["structured_data.title", "scene.title"],
                    "fallback": "A Brain's Journey",
                },
                "style": {
                    "font_family": "Inter",
                    "font_weight": "bold",
                    "font_size": 82,
                    "min_font_size": 38,
                    "color": "#F8FCFF",
                    "line_height": 1.1,
                    "shadow": {"dx": 2, "dy": 2, "blur": 4, "color": "#00000099"},
                    "glow": {"radius": 10, "color": "#8EC9FF77"},
                },
            },
            {
                "id": "subtitle",
                "box": [200, 520, 1264, 180],
                "align": "center",
                "valign": "top",
                "binding": {
                    "mode": "first_non_empty",
                    "paths": ["structured_data.subtitle", "structured_data.caption", "scene.narration"],
                    "fallback": "",
                },
                "style": {
                    "font_family": "Inter",
                    "font_weight": "regular",
                    "font_size": 36,
                    "min_font_size": 22,
                    "color": "#CDE5F8",
                    "line_height": 1.2,
                    "shadow": {"dx": 1, "dy": 1, "blur": 3, "color": "#00000088"},
                },
            },
        ],
    }
    anchor_rel = save_anchor("title_card_v1.json", anchor)
    return image_rel, anchor_rel


def build_bar_panel(count: int) -> tuple[str, str]:
    img = gradient_bg((10, 20, 36), (11, 40, 62))
    add_glow(img, (350, 780), 220, (96, 164, 255, 80))
    add_glow(img, (1280, 720), 240, (255, 180, 92, 84))
    draw_panel(img, (72, 46, 1592, 162), (10, 20, 34, 140), (170, 220, 255, 80))
    draw_panel(img, (72, 188, 1592, 850), (9, 18, 32, 95), (170, 220, 255, 64))

    margin = 220
    usable = WIDTH - margin * 2
    bar_w = 170 if count == 3 else 220
    gap = (usable - bar_w * count) // (count - 1 if count > 1 else 1)
    x_positions: list[int] = []
    x = margin
    for _ in range(count):
        x_positions.append(x)
        x += bar_w + gap

    for bx in x_positions:
        draw_panel(
            img,
            (bx, 300, bx + bar_w, 760),
            (20, 60, 92, 60),
            (195, 230, 255, 90),
            radius=28,
        )

    name = f"bar_volume_chart_{count}panel_v1.png"
    image_rel = save_image(name, img)

    anchors = [
        {
            "id": "metric_title",
            "box": [120, 62, 1424, 86],
            "align": "center",
            "valign": "middle",
            "binding": {
                "mode": "first_non_empty",
                "paths": ["structured_data.metric", "structured_data.title", "scene.title"],
                "fallback": "Signal Strength",
            },
            "style": {
                "font_family": "Inter",
                "font_weight": "bold",
                "font_size": 58,
                "min_font_size": 30,
                "color": "#F4FBFF",
                "shadow": {"dx": 2, "dy": 2, "blur": 4, "color": "#00000099"},
            },
        },
        {
            "id": "subtitle",
            "box": [120, 798, 1424, 72],
            "align": "center",
            "valign": "middle",
            "binding": {
                "mode": "template",
                "template": "Target: {structured_data.target_band.min}–{structured_data.target_band.max} {structured_data.target_band.unit}",
                "fallback": "",
            },
            "style": {
                "font_family": "Inter",
                "font_weight": "regular",
                "font_size": 30,
                "min_font_size": 20,
                "color": "#BDD9EF",
            },
        },
    ]
    bars = []
    for i, bx in enumerate(x_positions):
        idx = i + 1
        anchors.append(
            {
                "id": f"bar_{idx}_label",
                "box": [bx - 40, 242, bar_w + 80, 52],
                "align": "center",
                "valign": "middle",
                "binding": {"mode": "path", "path": f"structured_data.bars.{i}.label", "fallback": f"Session {idx}"},
                "style": {
                    "font_family": "Inter",
                    "font_weight": "regular",
                    "font_size": 28,
                    "min_font_size": 16,
                    "color": "#CFE7F9",
                },
            }
        )
        anchors.append(
            {
                "id": f"bar_{idx}_value",
                "box": [bx - 50, 700, bar_w + 100, 78],
                "align": "center",
                "valign": "middle",
                "binding": {
                    "mode": "template",
                    "template": f"{{structured_data.bars.{i}.value}} {{structured_data.bars.{i}.unit}}",
                    "fallback": "",
                },
                "style": {
                    "font_family": "Inter",
                    "font_weight": "bold",
                    "font_size": 46,
                    "min_font_size": 20,
                    "color": "#E6F5FF",
                    "shadow": {"dx": 2, "dy": 2, "blur": 3, "color": "#00000099"},
                    "glow": {"radius": 5, "color": "#82C5FF66"},
                },
            }
        )
        bars.append(
            {
                "id": f"bar_{idx}_fill",
                "box": [bx + 16, 320, bar_w - 32, 420],
                "source": f"structured_data.bars.{i}.value",
                "color": "#8CCBFF" if i < count - 1 else "#F4C97B",
                "base_alpha": 0.2,
            }
        )

    anchor = {
        "template_id": f"bar_volume_chart_{count}panel_v1",
        "dimensions": [WIDTH, HEIGHT],
        "anchors": anchors,
        "bars": bars,
    }
    anchor_rel = save_anchor(f"bar_volume_chart_{count}panel_v1.json", anchor)
    return image_rel, anchor_rel


def build_split_panel() -> tuple[str, str]:
    img = gradient_bg((12, 22, 36), (14, 44, 68))
    add_glow(img, (420, 740), 190, (90, 180, 255, 85))
    add_glow(img, (1240, 740), 190, (255, 166, 99, 85))
    draw_panel(img, (72, 48, 1592, 162), (9, 18, 32, 130), (182, 225, 255, 80))
    draw_panel(img, (72, 198, 804, 850), (8, 20, 36, 96), (130, 205, 255, 78))
    draw_panel(img, (860, 198, 1592, 850), (26, 19, 16, 86), (255, 202, 128, 78))
    image_rel = save_image("split_opposing_trends_v1.png", img)

    anchor = {
        "template_id": "split_opposing_trends_v1",
        "dimensions": [WIDTH, HEIGHT],
        "anchors": [
            {
                "id": "scene_title",
                "box": [120, 62, 1424, 84],
                "align": "center",
                "valign": "middle",
                "binding": {"mode": "first_non_empty", "paths": ["scene.title", "structured_data.headline"]},
                "style": {
                    "font_family": "Inter",
                    "font_weight": "bold",
                    "font_size": 58,
                    "min_font_size": 30,
                    "color": "#F8FCFF",
                    "shadow": {"dx": 2, "dy": 2, "blur": 4, "color": "#00000099"},
                },
            },
            {
                "id": "left_metric",
                "box": [120, 238, 636, 72],
                "align": "center",
                "valign": "middle",
                "binding": {"mode": "path", "path": "structured_data.left.metric", "fallback": ""},
                "style": {"font_family": "Inter", "font_weight": "bold", "font_size": 38, "min_font_size": 20, "color": "#D9EEFF"},
            },
            {
                "id": "left_values",
                "box": [120, 336, 636, 188],
                "align": "center",
                "valign": "top",
                "binding": {
                    "mode": "template",
                    "template": "{structured_data.left.from} → {structured_data.left.to} {structured_data.left.unit}",
                    "fallback": "",
                },
                "style": {
                    "font_family": "Inter",
                    "font_weight": "bold",
                    "font_size": 58,
                    "min_font_size": 24,
                    "color": "#B9DCFF",
                    "line_height": 1.15,
                },
            },
            {
                "id": "right_metric",
                "box": [908, 238, 636, 72],
                "align": "center",
                "valign": "middle",
                "binding": {"mode": "path", "path": "structured_data.right.metric", "fallback": ""},
                "style": {"font_family": "Inter", "font_weight": "bold", "font_size": 38, "min_font_size": 20, "color": "#FFE8CB"},
            },
            {
                "id": "right_values",
                "box": [908, 336, 636, 188],
                "align": "center",
                "valign": "top",
                "binding": {
                    "mode": "template",
                    "template": "{structured_data.right.from} → {structured_data.right.to} {structured_data.right.unit}",
                    "fallback": "",
                },
                "style": {
                    "font_family": "Inter",
                    "font_weight": "bold",
                    "font_size": 58,
                    "min_font_size": 24,
                    "color": "#FFD9AE",
                    "line_height": 1.15,
                },
            },
            {
                "id": "takeaway",
                "box": [120, 630, 1424, 180],
                "align": "center",
                "valign": "top",
                "binding": {
                    "mode": "first_non_empty",
                    "paths": ["structured_data.takeaway", "structured_data.headline", "scene.narration"],
                },
                "style": {
                    "font_family": "Inter",
                    "font_weight": "regular",
                    "font_size": 32,
                    "min_font_size": 20,
                    "color": "#DFEBF5",
                    "line_height": 1.2,
                },
            },
        ],
    }
    anchor_rel = save_anchor("split_opposing_trends_v1.json", anchor)
    return image_rel, anchor_rel


def build_waveform_panel() -> tuple[str, str]:
    img = gradient_bg((10, 18, 30), (10, 38, 60))
    draw = ImageDraw.Draw(img, "RGBA")
    add_glow(img, (1080, 200), 160, (120, 185, 255, 85))
    draw_panel(img, (72, 48, 1592, 162), (8, 18, 30, 130), (180, 220, 255, 80))
    draw_panel(img, (72, 198, 1592, 850), (8, 18, 30, 96), (170, 220, 255, 75))
    # grid
    for x in range(130, 1536, 80):
        draw.line([(x, 260), (x, 810)], fill=(170, 210, 240, 30), width=1)
    for y in range(260, 810, 60):
        draw.line([(120, y), (1544, y)], fill=(170, 210, 240, 30), width=1)
    # decorative waveform
    pts = []
    for i in range(0, 1410):
        x = 120 + i
        y = 520 + int(70 * math.sin(i / 42.0))
        pts.append((x, y))
    draw.line(pts, fill=(145, 205, 255, 150), width=3)
    image_rel = save_image("waveform_voltage_panel_v1.png", img)

    anchor = {
        "template_id": "waveform_voltage_panel_v1",
        "dimensions": [WIDTH, HEIGHT],
        "anchors": [
            {
                "id": "metric",
                "box": [120, 62, 1424, 82],
                "align": "center",
                "valign": "middle",
                "binding": {"mode": "first_non_empty", "paths": ["structured_data.metric", "scene.title"]},
                "style": {"font_family": "Inter", "font_weight": "bold", "font_size": 58, "min_font_size": 30, "color": "#F3FAFF"},
            },
            {
                "id": "trace_summary",
                "box": [120, 212, 1424, 120],
                "align": "left",
                "valign": "top",
                "binding": {
                    "mode": "summary",
                    "path": "structured_data.traces",
                    "max_lines": 3,
                    "max_chars": 160,
                },
                "style": {"font_family": "Inter", "font_weight": "regular", "font_size": 28, "min_font_size": 18, "color": "#C7E0F5"},
            },
            {
                "id": "details",
                "box": [120, 700, 1424, 140],
                "align": "left",
                "valign": "top",
                "binding": {"mode": "summary", "path": "structured_data", "max_lines": 3, "max_chars": 180},
                "style": {"font_family": "Inter", "font_weight": "regular", "font_size": 26, "min_font_size": 16, "color": "#D7EAF9"},
            },
        ],
    }
    anchor_rel = save_anchor("waveform_voltage_panel_v1.json", anchor)
    return image_rel, anchor_rel


def build_radial_panel() -> tuple[str, str]:
    img = gradient_bg((8, 20, 34), (14, 38, 58))
    draw = ImageDraw.Draw(img, "RGBA")
    add_glow(img, (832, 470), 220, (130, 190, 255, 90))
    draw_panel(img, (72, 48, 1592, 162), (8, 18, 30, 130), (180, 220, 255, 80))
    draw_panel(img, (72, 198, 1592, 850), (8, 18, 30, 90), (170, 220, 255, 70))
    center = (832, 510)
    radii = [220, 170, 120]
    for i, r in enumerate(radii):
        color = (140, 205, 255, 80 - i * 12)
        draw.ellipse((center[0] - r, center[1] - r, center[0] + r, center[1] + r), outline=color, width=10)
    image_rel = save_image("radial_kpi_ring_v1.png", img)

    anchor = {
        "template_id": "radial_kpi_ring_v1",
        "dimensions": [WIDTH, HEIGHT],
        "anchors": [
            {
                "id": "title",
                "box": [120, 62, 1424, 82],
                "align": "center",
                "valign": "middle",
                "binding": {"mode": "first_non_empty", "paths": ["structured_data.title", "scene.title"]},
                "style": {"font_family": "Inter", "font_weight": "bold", "font_size": 58, "min_font_size": 30, "color": "#F6FCFF"},
            },
            {
                "id": "ring_1",
                "box": [120, 270, 520, 70],
                "align": "left",
                "valign": "middle",
                "binding": {
                    "mode": "template",
                    "template": "{structured_data.rings.0.label}: {structured_data.rings.0.value} {structured_data.rings.0.unit}",
                    "fallback": "",
                },
                "style": {"font_family": "Inter", "font_weight": "regular", "font_size": 32, "min_font_size": 18, "color": "#D5EAFA"},
            },
            {
                "id": "ring_2",
                "box": [120, 350, 520, 70],
                "align": "left",
                "valign": "middle",
                "binding": {
                    "mode": "template",
                    "template": "{structured_data.rings.1.label}: {structured_data.rings.1.value} {structured_data.rings.1.unit}",
                    "fallback": "",
                },
                "style": {"font_family": "Inter", "font_weight": "regular", "font_size": 30, "min_font_size": 18, "color": "#D5EAFA"},
            },
            {
                "id": "ring_3",
                "box": [120, 430, 520, 70],
                "align": "left",
                "valign": "middle",
                "binding": {
                    "mode": "template",
                    "template": "{structured_data.rings.2.label}: {structured_data.rings.2.value} {structured_data.rings.2.unit}",
                    "fallback": "",
                },
                "style": {"font_family": "Inter", "font_weight": "regular", "font_size": 30, "min_font_size": 18, "color": "#D5EAFA"},
            },
            {
                "id": "center",
                "box": [672, 445, 320, 130],
                "align": "center",
                "valign": "middle",
                "binding": {
                    "mode": "first_non_empty",
                    "paths": ["structured_data.center_label", "structured_data.rings.0.value"],
                    "fallback": "",
                },
                "style": {
                    "font_family": "Inter",
                    "font_weight": "bold",
                    "font_size": 52,
                    "min_font_size": 22,
                    "color": "#F8FCFF",
                    "shadow": {"dx": 2, "dy": 2, "blur": 3, "color": "#00000099"},
                },
            },
        ],
    }
    anchor_rel = save_anchor("radial_kpi_ring_v1.json", anchor)
    return image_rel, anchor_rel


def build_roadmap_panel() -> tuple[str, str]:
    img = gradient_bg((10, 20, 35), (13, 44, 68))
    draw = ImageDraw.Draw(img, "RGBA")
    add_glow(img, (280, 620), 220, (110, 185, 255, 75))
    draw_panel(img, (72, 48, 1592, 162), (8, 18, 30, 130), (180, 225, 255, 80))
    draw_panel(img, (72, 198, 1592, 850), (8, 18, 30, 90), (170, 220, 255, 70))
    draw.line((230, 300, 230, 760), fill=(170, 220, 255, 130), width=4)
    for y in [340, 420, 500, 580, 660]:
        draw.ellipse((216, y - 14, 244, y + 14), fill=(166, 214, 255, 210))
    image_rel = save_image("roadmap_agenda_v1.png", img)

    anchor = {
        "template_id": "roadmap_agenda_v1",
        "dimensions": [WIDTH, HEIGHT],
        "anchors": [
            {
                "id": "title",
                "box": [120, 62, 1424, 84],
                "align": "center",
                "valign": "middle",
                "binding": {"mode": "first_non_empty", "paths": ["structured_data.title", "scene.title"], "fallback": "What We'll Explore"},
                "style": {"font_family": "Inter", "font_weight": "bold", "font_size": 58, "min_font_size": 30, "color": "#F7FCFF"},
            },
            {
                "id": "items",
                "box": [280, 300, 1180, 470],
                "align": "left",
                "valign": "top",
                "binding": {
                    "mode": "join_path",
                    "path": "structured_data.items",
                    "field": "label",
                    "separator": "\n",
                    "prefix": "• ",
                    "max_items": 8,
                    "fallback": "",
                },
                "style": {"font_family": "Inter", "font_weight": "regular", "font_size": 42, "min_font_size": 22, "color": "#D6EBFA", "line_height": 1.22},
            },
        ],
    }
    anchor_rel = save_anchor("roadmap_agenda_v1.json", anchor)
    return image_rel, anchor_rel


def main() -> None:
    generic_img, generic_anchor = build_generic_data_panel()
    title_img, title_anchor = build_title_card()
    bar3_img, bar3_anchor = build_bar_panel(3)
    bar2_img, bar2_anchor = build_bar_panel(2)
    split_img, split_anchor = build_split_panel()
    wave_img, wave_anchor = build_waveform_panel()
    radial_img, radial_anchor = build_radial_panel()
    roadmap_img, roadmap_anchor = build_roadmap_panel()

    templates = [
        # BAR variants
        {
            "template_id": "bar_volume_chart_3panel_v1",
            "scene_types": ["bar_volume_chart"],
            "template_path": bar3_img,
            "anchors_path": bar3_anchor,
            "priority": 10,
            "selector": {"bar_count_eq": 3},
            "tags": ["bars", "3panel"],
        },
        {
            "template_id": "bar_volume_chart_2panel_v1",
            "scene_types": ["bar_volume_chart"],
            "template_path": bar2_img,
            "anchors_path": bar2_anchor,
            "priority": 11,
            "selector": {"bar_count_eq": 2},
            "tags": ["bars", "2panel"],
        },
        # Highly-specific scene templates
        {
            "template_id": "split_opposing_trends_v1",
            "scene_types": ["split_opposing_trends"],
            "template_path": split_img,
            "anchors_path": split_anchor,
            "priority": 20,
            "selector": {},
            "tags": ["split"],
        },
        {
            "template_id": "roadmap_agenda_v1",
            "scene_types": ["roadmap_agenda"],
            "template_path": roadmap_img,
            "anchors_path": roadmap_anchor,
            "priority": 20,
            "selector": {},
            "tags": ["roadmap"],
        },
        {
            "template_id": "waveform_voltage_panel_v1",
            "scene_types": ["waveform_voltage_panel"],
            "template_path": wave_img,
            "anchors_path": wave_anchor,
            "priority": 20,
            "selector": {},
            "tags": ["waveform"],
        },
        {
            "template_id": "radial_kpi_ring_v1",
            "scene_types": ["radial_kpi_ring"],
            "template_path": radial_img,
            "anchors_path": radial_anchor,
            "priority": 20,
            "selector": {},
            "tags": ["radial"],
        },
        {
            "template_id": "title_card_v1",
            "scene_types": [
                "atmospheric_title_card",
                "atmospheric_metaphor_scene",
                "atmospheric_mechanism_scene",
            ],
            "template_path": title_img,
            "anchors_path": title_anchor,
            "priority": 20,
            "selector": {},
            "tags": ["title", "atmospheric"],
        },
        # Generic fallback for all remaining scene types
        {
            "template_id": "generic_data_panel_v1",
            "scene_types": [
                "multi_session_trend",
                "verdict_summary",
                "session_timeline",
                "gauge_ratio_meter",
                "coherence_network_map",
                "hemispheric_compare",
                "state_flexibility_rest_task",
                "baseline_target_split",
                "measurement_primer",
                "future_projection",
                "line_trajectory",
                "coherence_progression_sequence",
                "dotplot_variability",
                "pathway_hub_synthesis",
                "regional_frequency_map",
                "table_dashboard",
                "quality_alert",
                "bar_volume_chart",
            ],
            "template_path": generic_img,
            "anchors_path": generic_anchor,
            "priority": 999,
            "selector": {},
            "tags": ["fallback", "generic"],
        },
    ]

    for template in templates:
        template.setdefault("selector_metadata", {"strategy": "bootstrap_default", "notes": "scaffold template"})
        template.setdefault("origin", "scaffold_only")
        template.setdefault("curation_status", "pending")
        template.setdefault("approved_by", None)
        template.setdefault("approved_at", None)
        if template.get("template_id") == "generic_data_panel_v1":
            template.setdefault("operational_status", "dev_only")
            template.setdefault("production_ready", False)
        else:
            template.setdefault("operational_status", "dev_only")
            template.setdefault("production_ready", False)

    manifest = {"manifest_version": "1.1.0", "templates": templates}
    manifest_path = _repo_root() / "templates" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
