#!/usr/bin/env python3
"""Build hand-rolled animated HTML scene artifacts for a qEEG patient project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COMMON_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Fraunces:opsz,wght@9..144,700&display=swap');
* { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg: #0a0f1b;
  --panel: rgba(15, 24, 41, 0.78);
  --panel-soft: rgba(20, 30, 54, 0.58);
  --line: rgba(255,255,255,0.08);
  --text: #eef2ff;
  --muted: rgba(221, 230, 255, 0.62);
  --teal: #42d9cc;
  --teal-soft: rgba(66, 217, 204, 0.18);
  --amber: #f2b649;
  --amber-soft: rgba(242, 182, 73, 0.16);
  --violet: #b38cff;
  --violet-soft: rgba(179, 140, 255, 0.18);
  --blue: #6ab6ff;
  --blue-soft: rgba(106, 182, 255, 0.18);
  --red: #ef6b73;
}
html, body {
  width: 100%;
  height: 100%;
  overflow: hidden;
  background: radial-gradient(circle at 20% 20%, rgba(64, 114, 255, 0.1), transparent 35%),
              radial-gradient(circle at 80% 0%, rgba(66, 217, 204, 0.08), transparent 30%),
              linear-gradient(180deg, #0b1020 0%, #070b16 100%);
  color: var(--text);
  font-family: 'Inter', sans-serif;
}
body::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  background:
    linear-gradient(rgba(255,255,255,0.015) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.015) 1px, transparent 1px);
  background-size: 40px 40px;
  opacity: 0.28;
}
.cinematic-shell {
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 2;
  overflow: hidden;
}
.cinematic-matte {
  position: absolute;
  inset: 0;
  background:
    radial-gradient(circle at 50% 50%, transparent 56%, rgba(4, 8, 16, 0.18) 74%, rgba(4, 8, 16, 0.55) 100%),
    linear-gradient(180deg, rgba(5, 10, 18, 0.84) 0%, rgba(5, 10, 18, 0.22) 12%, rgba(5, 10, 18, 0.18) 88%, rgba(5, 10, 18, 0.88) 100%);
}
.cinematic-vignette {
  position: absolute;
  inset: 0;
  background:
    radial-gradient(circle at 50% 50%, transparent 50%, rgba(4, 8, 16, 0.15) 70%, rgba(4, 8, 16, 0.54) 100%),
    radial-gradient(circle at 50% 0%, rgba(242, 182, 73, 0.08), transparent 34%),
    radial-gradient(circle at 50% 100%, rgba(66, 217, 204, 0.06), transparent 36%);
}
.cinematic-frame-outer,
.cinematic-frame-inner,
.cinematic-frame-core {
  position: absolute;
  border-radius: 36px;
}
.cinematic-frame-outer {
  inset: 14px;
  border: 1px solid rgba(255,255,255,0.06);
  box-shadow:
    inset 0 0 0 1px rgba(255,255,255,0.015),
    0 0 0 1px rgba(10, 18, 34, 0.65),
    0 36px 90px rgba(0, 0, 0, 0.35);
}
.cinematic-frame-inner {
  inset: 22px;
  border: 1px solid rgba(255,255,255,0.045);
  box-shadow:
    inset 0 0 40px rgba(66, 217, 204, 0.025),
    inset 0 0 22px rgba(242, 182, 73, 0.03);
}
.cinematic-frame-core {
  inset: 30px;
  border: 1px solid rgba(255,255,255,0.022);
  box-shadow:
    inset 0 0 0 1px rgba(255,255,255,0.02),
    inset 0 0 120px rgba(255,255,255,0.015);
}
.cinematic-edge-glow {
  position: absolute;
  inset: 18px;
  border-radius: 34px;
  border: 1px solid rgba(255,255,255,0.04);
  box-shadow:
    inset 0 0 0 1px rgba(255,255,255,0.02),
    inset 0 0 100px rgba(66, 217, 204, 0.045),
    inset 0 0 70px rgba(242, 182, 73, 0.03),
    0 0 60px rgba(66, 217, 204, 0.03);
}
.cinematic-edge-glow::before,
.cinematic-edge-glow::after {
  content: "";
  position: absolute;
  left: 28px;
  right: 28px;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(242,182,73,0.25), rgba(66,217,204,0.25), transparent);
}
.cinematic-edge-glow::before { top: 18px; }
.cinematic-edge-glow::after { bottom: 18px; }
.cinematic-side-rail {
  position: absolute;
  top: 82px;
  bottom: 82px;
  width: 26px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.05);
  background:
    linear-gradient(180deg, rgba(255,255,255,0.1), rgba(255,255,255,0.02) 18%, rgba(255,255,255,0.02) 82%, rgba(255,255,255,0.08)),
    linear-gradient(180deg, rgba(242,182,73,0.12), rgba(66,217,204,0.08));
  box-shadow:
    inset 0 0 14px rgba(255,255,255,0.04),
    0 0 30px rgba(66,217,204,0.06);
}
.cinematic-side-rail::before {
  content: "";
  position: absolute;
  inset: 9px;
  border-radius: 999px;
  background:
    repeating-linear-gradient(180deg, rgba(255,255,255,0.08) 0 1px, transparent 1px 10px),
    linear-gradient(180deg, rgba(66,217,204,0.16), rgba(242,182,73,0.12));
  opacity: 0.8;
}
.cinematic-side-rail.left { left: 26px; }
.cinematic-side-rail.right { right: 26px; }
.cinematic-top-plate,
.cinematic-bottom-plate {
  position: absolute;
  left: 50%;
  width: 420px;
  height: 34px;
  transform: translateX(-50%);
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.06);
  background:
    linear-gradient(90deg, rgba(242,182,73,0.12), rgba(255,255,255,0.02) 18%, rgba(255,255,255,0.02) 82%, rgba(66,217,204,0.12)),
    rgba(8, 14, 26, 0.58);
  box-shadow:
    inset 0 0 18px rgba(255,255,255,0.03),
    0 0 22px rgba(66,217,204,0.05);
}
.cinematic-top-plate { top: 24px; }
.cinematic-bottom-plate { bottom: 24px; width: 360px; }
.cinematic-top-plate::before,
.cinematic-bottom-plate::before {
  content: "";
  position: absolute;
  left: 22px;
  right: 22px;
  top: 50%;
  height: 1px;
  transform: translateY(-50%);
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.24), transparent);
}
.cinematic-top-plate::after,
.cinematic-bottom-plate::after {
  content: "";
  position: absolute;
  top: 50%;
  left: 50%;
  width: 68px;
  height: 8px;
  transform: translate(-50%, -50%);
  border-radius: 999px;
  background: linear-gradient(90deg, rgba(242,182,73,0.35), rgba(66,217,204,0.35));
  box-shadow: 0 0 18px rgba(66,217,204,0.12);
}
.cinematic-corners {
  position: absolute;
  inset: 24px;
}
.corner {
  position: absolute;
  width: 120px;
  height: 120px;
  opacity: 0.85;
}
.corner::before,
.corner::after {
  content: "";
  position: absolute;
  background: linear-gradient(90deg, rgba(242,182,73,0.8), rgba(66,217,204,0.75));
  box-shadow: 0 0 18px rgba(66,217,204,0.16);
}
.corner::before {
  width: 72px;
  height: 2px;
}
.corner::after {
  width: 2px;
  height: 72px;
}
.corner.tl { top: 0; left: 0; }
.corner.tl::before, .corner.tl::after { top: 0; left: 0; }
.corner.tr { top: 0; right: 0; transform: scaleX(-1); }
.corner.tr::before, .corner.tr::after { top: 0; left: 0; }
.corner.bl { bottom: 0; left: 0; transform: scaleY(-1); }
.corner.bl::before, .corner.bl::after { top: 0; left: 0; }
.corner.br { bottom: 0; right: 0; transform: scale(-1, -1); }
.corner.br::before, .corner.br::after { top: 0; left: 0; }
.cinematic-node {
  position: absolute;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(255,255,255,0.95) 0%, rgba(66,217,204,0.55) 42%, rgba(66,217,204,0) 72%);
  box-shadow: 0 0 18px rgba(66,217,204,0.2);
  opacity: 0.92;
}
.cinematic-node.top-left { top: 54px; left: 182px; }
.cinematic-node.top-right { top: 54px; right: 182px; }
.cinematic-node.bottom-left { bottom: 54px; left: 182px; }
.cinematic-node.bottom-right { bottom: 54px; right: 182px; }
.cinematic-sweep {
  position: absolute;
  inset: -10%;
  background: linear-gradient(100deg, transparent 42%, rgba(255,255,255,0.045) 50%, transparent 58%);
  transform: translateX(-55%) rotate(4deg);
  opacity: 0.45;
  animation: sweepFrame 11s linear infinite;
}
.cinematic-noise {
  position: absolute;
  inset: 0;
  opacity: 0.035;
  mix-blend-mode: screen;
  background-image:
    radial-gradient(circle at 20% 30%, rgba(255,255,255,.75) 0 0.7px, transparent 0.8px),
    radial-gradient(circle at 65% 55%, rgba(255,255,255,.5) 0 0.7px, transparent 0.8px),
    radial-gradient(circle at 80% 18%, rgba(255,255,255,.7) 0 0.7px, transparent 0.8px),
    radial-gradient(circle at 40% 72%, rgba(255,255,255,.55) 0 0.7px, transparent 0.8px);
  background-size: 120px 120px, 160px 160px, 200px 200px, 140px 140px;
}
.page {
  position: relative;
  width: 100vw;
  height: 100vh;
  padding: 48px 60px;
  z-index: 3;
}
.kicker {
  font-size: 12px;
  letter-spacing: 0.32em;
  text-transform: uppercase;
  color: rgba(66, 217, 204, 0.72);
  margin-bottom: 10px;
}
.headline {
  font-family: 'Fraunces', serif;
  font-size: 54px;
  line-height: 0.95;
  letter-spacing: -0.03em;
  margin-bottom: 14px;
  text-shadow: 0 10px 40px rgba(0, 0, 0, 0.35);
}
.subhead {
  font-size: 21px;
  color: var(--muted);
}
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 28px;
  box-shadow: 0 30px 80px rgba(0, 0, 0, 0.35);
  backdrop-filter: blur(14px);
}
.caption-box {
  position: absolute;
  left: 60px;
  right: 60px;
  bottom: 44px;
  padding: 18px 22px;
  border-left: 3px solid rgba(66, 217, 204, 0.4);
  background: rgba(10, 18, 34, 0.72);
  border-radius: 0 18px 18px 0;
  color: rgba(220, 231, 255, 0.72);
  font-size: 16px;
  line-height: 1.55;
  font-style: italic;
}
.metric-chip {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 10px 16px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.1);
  background: rgba(255,255,255,0.04);
  font-size: 14px;
  font-weight: 700;
}
.teal { color: var(--teal); }
.amber { color: var(--amber); }
.violet { color: var(--violet); }
.blue { color: var(--blue); }
.fade-in { opacity: 0; animation: fadeIn .8s ease forwards; }
.slide-up { opacity: 0; transform: translateY(24px); animation: slideUp .8s cubic-bezier(.2,.8,.2,1) forwards; }
@keyframes fadeIn { to { opacity: 1; } }
@keyframes slideUp { to { opacity: 1; transform: translateY(0); } }
@keyframes pulseGlow {
  0%, 100% { box-shadow: 0 0 0 rgba(66,217,204,0.0); }
  50% { box-shadow: 0 0 30px rgba(66,217,204,0.22); }
}
@keyframes sweepFrame {
  0% { transform: translateX(-55%) rotate(4deg); }
  100% { transform: translateX(55%) rotate(4deg); }
}
"""


def page_html(title: str, body: str, *, css: str = "", script: str = "", p5: bool = False) -> str:
    scripts = ""
    runtime_css = ""
    if p5:
        scripts += '<script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.9.0/p5.min.js"></script>'
        runtime_css += """
canvas {
  position: fixed !important;
  inset: 0 !important;
  width: 100vw !important;
  height: 100vh !important;
  display: block !important;
  z-index: 1 !important;
  pointer-events: none !important;
}
"""
    shell = """
<div class="cinematic-shell">
  <div class="cinematic-matte"></div>
  <div class="cinematic-vignette"></div>
  <div class="cinematic-frame-outer"></div>
  <div class="cinematic-frame-inner"></div>
  <div class="cinematic-frame-core"></div>
  <div class="cinematic-edge-glow"></div>
  <div class="cinematic-side-rail left"></div>
  <div class="cinematic-side-rail right"></div>
  <div class="cinematic-top-plate"></div>
  <div class="cinematic-bottom-plate"></div>
  <div class="cinematic-corners">
    <div class="corner tl"></div>
    <div class="corner tr"></div>
    <div class="corner bl"></div>
    <div class="corner br"></div>
  </div>
  <div class="cinematic-node top-left"></div>
  <div class="cinematic-node top-right"></div>
  <div class="cinematic-node bottom-left"></div>
  <div class="cinematic-node bottom-right"></div>
  <div class="cinematic-sweep"></div>
  <div class="cinematic-noise"></div>
</div>
"""
    return (
        "<!DOCTYPE html><html lang=\"en\"><head>"
        "<meta charset=\"UTF-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">"
        f"<title>{title}</title>"
        f"{scripts}<style>{COMMON_CSS}{runtime_css}{css}</style></head><body>{shell}{body}{script}</body></html>"
    )


def scene_01_title_card() -> str:
    body = """
<div class="page" style="display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center;">
  <div class="fade-in" style="animation-delay:.2s;">
    <div class="kicker">LUMIT PATIENT DATA</div>
    <div class="headline" style="font-size:82px; max-width:1200px;">A Brain Finding Its Rhythm</div>
    <div class="subhead" style="font-size:28px;">Three sessions. Sixty-five days. One remarkable neurological story.</div>
  </div>
  <div style="position:absolute; inset:0; pointer-events:none;">
    <div style="position:absolute; left:50%; top:50%; width:580px; height:580px; transform:translate(-50%, -52%); border-radius:50%; border:1px solid rgba(66,217,204,.18); animation:pulseOrbit 5s ease-in-out infinite;"></div>
    <div style="position:absolute; left:50%; top:50%; width:760px; height:760px; transform:translate(-50%, -52%); border-radius:50%; border:1px solid rgba(106,182,255,.12); animation:pulseOrbit 7s ease-in-out infinite reverse;"></div>
    <svg viewBox="0 0 1664 928" style="position:absolute; inset:0; width:100%; height:100%; opacity:.24;">
      <path d="M595 435 C610 250 786 148 924 178 C1035 203 1138 300 1150 430 C1160 542 1098 688 986 748 C880 804 710 794 620 688 C571 630 588 523 595 435Z"
            fill="none" stroke="rgba(230,240,255,.42)" stroke-width="3"/>
      <path d="M640 360 C700 320 760 318 808 346 C860 377 910 383 960 360" fill="none" stroke="rgba(66,217,204,.4)" stroke-width="2"/>
      <path d="M664 504 C726 454 794 450 862 468 C926 484 988 484 1030 456" fill="none" stroke="rgba(106,182,255,.32)" stroke-width="2"/>
    </svg>
  </div>
</div>
"""
    css = """
@keyframes pulseOrbit {
  0%, 100% { opacity: .35; transform: translate(-50%, -52%) scale(0.98); }
  50% { opacity: .8; transform: translate(-50%, -52%) scale(1.02); }
}
"""
    return page_html("Scene 1 - Title Card", body, css=css)


def scene_02_roadmap() -> str:
    body = """
<div class="page">
  <div class="fade-in" style="animation-delay:.15s; text-align:center; margin-bottom:28px;">
    <div class="headline" style="font-size:72px;">What We Will Explore</div>
    <div class="subhead">Four threads of evidence from three brain recordings</div>
  </div>
  <div style="position:relative; width:760px; margin:0 auto; padding-left:46px;">
    <div style="position:absolute; left:14px; top:16px; bottom:16px; width:2px; background:linear-gradient(180deg, rgba(66,217,204,.42), rgba(66,217,204,.02));"></div>
    <div class="panel slide-up" style="animation-delay:.35s; padding:22px 24px; margin-bottom:16px;">
      <div class="metric-chip blue" style="margin-bottom:10px;">Attention Signals <span style="color:#9fc6ff;">Nearly 2× Stronger</span></div>
      <div style="font-size:18px; color:var(--muted);">How the brain's electrical response to important sounds grew dramatically.</div>
    </div>
    <div class="panel slide-up" style="animation-delay:.55s; padding:22px 24px; margin-bottom:16px;">
      <div class="metric-chip teal" style="margin-bottom:10px;">Processing Speed <span style="color:#9af1e8;">From lagging to landing</span></div>
      <div style="font-size:18px; color:var(--muted);">Why diagnostic delay moved into the healthy target window.</div>
    </div>
    <div class="panel slide-up" style="animation-delay:.75s; padding:22px 24px; margin-bottom:16px;">
      <div class="metric-chip amber" style="margin-bottom:10px;">Network Reorganization <span style="color:#ffd581;">The orchestra finds its rhythm</span></div>
      <div style="font-size:18px; color:var(--muted);">A dominant baseline connection quiets while healthier links strengthen.</div>
    </div>
    <div class="panel slide-up" style="animation-delay:.95s; padding:22px 24px;">
      <div class="metric-chip violet" style="margin-bottom:10px;">The Frequency Puzzle <span style="color:#d2b6ff;">Not every region changed the same way</span></div>
      <div style="font-size:18px; color:var(--muted);">Some rhythms normalized beautifully while others raised a new question.</div>
    </div>
  </div>
  <div class="caption-box slide-up" style="animation-delay:1.15s; width:760px; left:50%; transform:translateX(-50%); right:auto;">
    "We’ll start at baseline, then follow signal strength, processing speed, network reorganization, and the one puzzle piece that kept this story honest."
  </div>
</div>
"""
    return page_html("Scene 2 - Roadmap", body)


def scene_03_timeline() -> str:
    body = """
<div class="page">
  <div class="fade-in" style="animation-delay:.2s; margin-bottom:28px;">
    <div class="kicker">PATIENT CONTEXT</div>
    <div class="headline">The Patient and Baseline</div>
    <div class="subhead">Age seventy-one. Mild cognitive impairment. Three sessions over sixty-five days.</div>
  </div>
  <div class="panel slide-up" style="animation-delay:.45s; padding:42px 48px; width:1260px; margin:0 auto;">
    <div style="display:flex; justify-content:space-between; align-items:center; position:relative; height:220px;">
      <div style="position:absolute; left:120px; right:120px; top:88px; height:4px; background:linear-gradient(90deg, rgba(106,182,255,.28), rgba(66,217,204,.4), rgba(242,182,73,.32)); border-radius:999px;"></div>
      <div style="position:absolute; left:33%; top:76px; width:140px; font-size:14px; text-align:center; color:var(--muted);">35 days</div>
      <div style="position:absolute; left:66.2%; top:76px; width:140px; font-size:14px; text-align:center; color:var(--muted);">30 days</div>
      <div class="slide-up" style="animation-delay:.7s; position:relative; width:260px; text-align:center;">
        <div style="width:32px; height:32px; margin:0 auto 18px; border-radius:50%; background:rgba(106,182,255,.2); border:2px solid rgba(106,182,255,.8); box-shadow:0 0 26px rgba(106,182,255,.26);"></div>
        <div style="font-size:22px; font-weight:800;">Session 1</div>
        <div style="font-size:18px; color:var(--muted);">December 2025</div>
        <div style="margin-top:12px; font-size:14px; color:#9fc6ff; letter-spacing:.2em; text-transform:uppercase;">Baseline</div>
      </div>
      <div class="slide-up" style="animation-delay:.95s; position:relative; width:260px; text-align:center;">
        <div style="width:32px; height:32px; margin:0 auto 18px; border-radius:50%; background:rgba(66,217,204,.2); border:2px solid rgba(66,217,204,.84); box-shadow:0 0 26px rgba(66,217,204,.22);"></div>
        <div style="font-size:22px; font-weight:800;">Session 2</div>
        <div style="font-size:18px; color:var(--muted);">January 2026</div>
        <div style="margin-top:12px; font-size:14px; color:#9af1e8; letter-spacing:.2em; text-transform:uppercase;">Follow-up</div>
      </div>
      <div class="slide-up" style="animation-delay:1.2s; position:relative; width:260px; text-align:center;">
        <div style="width:32px; height:32px; margin:0 auto 18px; border-radius:50%; background:rgba(242,182,73,.2); border:2px solid rgba(242,182,73,.84); box-shadow:0 0 26px rgba(242,182,73,.22);"></div>
        <div style="font-size:22px; font-weight:800;">Session 3</div>
        <div style="font-size:18px; color:var(--muted);">February 2026</div>
        <div style="margin-top:12px; font-size:14px; color:#ffd581; letter-spacing:.2em; text-transform:uppercase;">Current snapshot</div>
      </div>
    </div>
  </div>
  <div class="caption-box slide-up" style="animation-delay:1.35s;">
    "What matters is not one isolated spike. It’s that the same brain was measured three times across sixty-five days, so we can watch the trajectory instead of guessing."
  </div>
</div>
"""
    return page_html("Scene 3 - The Patient and Baseline", body)


def scene_04_orchestra_intro() -> str:
    body = """
<div id="overlay" style="position:fixed; inset:0; pointer-events:none; z-index:10; padding:44px 52px; display:flex; flex-direction:column; justify-content:space-between;">
  <div class="fade-in" style="animation-delay:.2s; text-align:center;">
    <div class="headline" style="font-size:70px;">The Brain as an Orchestra</div>
    <div class="subhead">Each region has a section. Beautiful music depends on timing, balance, and coordination.</div>
  </div>
  <div style="display:flex; justify-content:space-between; padding:0 120px; font-size:18px; color:rgba(232,240,255,.82); font-weight:700;">
    <div class="panel slide-up" style="animation-delay:1s; padding:14px 22px;">Front</div>
    <div class="panel slide-up" style="animation-delay:1.15s; padding:14px 22px;">Center</div>
    <div class="panel slide-up" style="animation-delay:1.3s; padding:14px 22px;">Left</div>
    <div class="panel slide-up" style="animation-delay:1.45s; padding:14px 22px;">Right</div>
    <div class="panel slide-up" style="animation-delay:1.6s; padding:14px 22px;">Rear</div>
  </div>
  <div class="caption-box" style="position:relative; left:auto; right:auto; bottom:auto; margin:0 auto; max-width:980px;">
    "This analogy unlocks the rest of the report. We’re not asking whether one number changed. We’re asking whether the whole ensemble learned to play together."
  </div>
</div>
"""
    script = """
<script>
const clusters = [
  {x: 832, y: 240, n: 22, color: [66,217,204], label: 'front'},
  {x: 832, y: 420, n: 26, color: [106,182,255], label: 'center'},
  {x: 520, y: 420, n: 18, color: [242,182,73], label: 'left'},
  {x: 1144, y: 420, n: 18, color: [242,182,73], label: 'right'},
  {x: 832, y: 660, n: 22, color: [179,140,255], label: 'rear'},
];
let particles = [];
function setup() {
  createCanvas(windowWidth, windowHeight);
  pixelDensity(1);
  noStroke();
  clusters.forEach((cluster, idx) => {
    for (let i = 0; i < cluster.n; i++) {
      particles.push({
        cluster: idx,
        angle: random(TWO_PI),
        radius: random(18, 110),
        speed: random(0.003, 0.012),
        size: random(4, 10),
      });
    }
  });
}
function draw() {
  background(10, 15, 27, 35);
  stroke(255,255,255,16);
  noFill();
  ellipse(width/2, height/2, 720, 540);
  clusters.forEach((cluster, idx) => {
    const [r,g,b] = cluster.color;
    fill(r,g,b,22);
    noStroke();
    ellipse(cluster.x, cluster.y, 160, 160);
    particles.filter(p => p.cluster === idx).forEach((p, i) => {
      const t = frameCount * p.speed + p.angle;
      const x = cluster.x + cos(t * 1.1) * p.radius;
      const y = cluster.y + sin(t * 0.9) * p.radius * 0.66;
      fill(r, g, b, 38);
      ellipse(x, y, p.size * 3.2, p.size * 3.2);
      fill(250, 252, 255, 160);
      ellipse(x, y, p.size, p.size);
    });
  });
}
</script>
"""
    return page_html("Scene 4 - The Brain as an Orchestra", body, script=script, p5=True)


def scene_05_signal_strength() -> str:
    body = """
<div class="page" style="display:flex; flex-direction:column; align-items:center;">
  <div class="fade-in" style="animation-delay:.2s; text-align:center; margin-top:10px;">
    <div class="kicker">P300 SIGNAL VOLTAGE</div>
    <div class="headline">Signal Strength: Nearly Doubled</div>
    <div class="subhead">The brain’s detection response climbed from barely normal to clearly robust.</div>
  </div>
  <div class="panel slide-up" style="animation-delay:.45s; width:1240px; margin-top:34px; padding:46px 50px 34px;">
    <div style="position:relative; height:430px; display:flex; align-items:flex-end; justify-content:center; gap:120px;">
      <div style="position:absolute; left:0; right:0; top:110px; height:76px; background:rgba(66,217,204,.08); border-top:1px dashed rgba(66,217,204,.22); border-bottom:1px dashed rgba(66,217,204,.22);"></div>
      <div style="position:absolute; right:16px; top:134px; font-size:14px; color:#8ce9de;">Target: 6–14 µV</div>
      <div class="metric-group" data-value="13.1" data-max="26">
        <div class="metric-value blue">0.0</div>
        <div class="metric-bar-wrap"><div class="metric-bar blue-fill"></div></div>
        <div class="metric-label">Session 1</div>
        <div class="metric-note">Baseline</div>
      </div>
      <div class="metric-group" data-value="22.9" data-max="26">
        <div class="metric-value teal">0.0</div>
        <div class="metric-bar-wrap"><div class="metric-bar teal-fill"></div></div>
        <div class="metric-label">Session 2</div>
        <div class="metric-note">Midpoint</div>
      </div>
      <div class="metric-group" data-value="24.0" data-max="26">
        <div class="metric-value amber">0.0</div>
        <div class="metric-bar-wrap"><div class="metric-bar amber-fill"></div></div>
        <div class="metric-label">Session 3</div>
        <div class="metric-note">Follow-up</div>
      </div>
    </div>
    <div style="display:flex; justify-content:center; margin-top:30px;">
      <div class="metric-chip teal" id="signal-delta">+0%</div>
    </div>
  </div>
  <div class="caption-box slide-up" style="animation-delay:2.4s;">
    "By the third session, the P300 signal had risen from thirteen point one to twenty-four microvolts. That’s the brain saying, ‘I’m noticing what matters more powerfully now.’"
  </div>
</div>
"""
    css = """
.metric-group { width: 170px; text-align:center; }
.metric-value { font-size: 44px; font-weight: 800; margin-bottom: 18px; letter-spacing: -.04em; }
.metric-bar-wrap { height: 250px; position: relative; background: rgba(255,255,255,.03); border-radius: 18px 18px 0 0; overflow: hidden; border:1px solid rgba(255,255,255,.04); }
.metric-bar { position: absolute; left:0; right:0; bottom:0; height: 0%; border-radius: 18px 18px 0 0; transition: height 2.1s cubic-bezier(.2,.8,.2,1); }
.metric-label { margin-top: 18px; font-size: 20px; font-weight: 700; }
.metric-note { margin-top: 6px; color: var(--muted); font-size: 15px; }
.blue-fill { background: linear-gradient(180deg, rgba(106,182,255,.95), rgba(62,120,212,.88)); box-shadow: 0 0 30px rgba(106,182,255,.18); }
.teal-fill { background: linear-gradient(180deg, rgba(66,217,204,.95), rgba(27,136,129,.88)); box-shadow: 0 0 34px rgba(66,217,204,.18); }
.amber-fill { background: linear-gradient(180deg, rgba(242,182,73,.96), rgba(203,118,23,.92)); box-shadow: 0 0 34px rgba(242,182,73,.22); }
"""
    script = """
<script>
document.querySelectorAll('.metric-group').forEach((group, idx) => {
  const value = Number(group.dataset.value);
  const max = Number(group.dataset.max);
  const bar = group.querySelector('.metric-bar');
  const valueEl = group.querySelector('.metric-value');
  setTimeout(() => {
    bar.style.height = ((value / max) * 100).toFixed(1) + '%';
    const start = performance.now();
    const step = (now) => {
      const p = Math.min((now - start) / 1800, 1);
      valueEl.textContent = (value * (1 - Math.pow(1 - p, 3))).toFixed(1);
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }, 650 + idx * 250);
});
setTimeout(() => {
  const el = document.getElementById('signal-delta');
  const start = performance.now();
  const step = (now) => {
    const p = Math.min((now - start) / 1200, 1);
    el.textContent = '+' + Math.round(83 * (1 - Math.pow(1 - p, 3))) + '%';
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}, 2200);
</script>
"""
    return page_html("Scene 5 - Signal Strength", body, css=css, script=script)


def scene_06_processing_speed() -> str:
    body = """
<div class="page">
  <div class="fade-in" style="animation-delay:.2s; text-align:center;">
    <div class="kicker">P300 PROCESSING SPEED</div>
    <div class="headline">From Lagging to Landing</div>
    <div class="subhead">Latency started diagnostic, then moved decisively into the target window.</div>
  </div>
  <div class="panel slide-up" style="animation-delay:.45s; margin:36px auto 0; width:1380px; padding:34px 40px 38px;">
    <div class="latency-band-intro">
      <div class="latency-band-copy">Healthy timing window</div>
      <div class="latency-band-value">302–393 ms</div>
    </div>
    <div class="latency-grid">
      <div class="latency-card latency-delay">
        <div class="latency-card-top">
          <div class="latency-session blue">Session 1</div>
          <div class="latency-status-pill red-pill">Delayed</div>
        </div>
        <div class="latency-card-value red">452 ms</div>
        <div class="latency-card-copy">Baseline latency was slow enough to support the clinical concern.</div>
        <div class="latency-meter" style="--marker: 92%;">
          <div class="latency-meter-track"></div>
          <div class="latency-meter-target"></div>
          <div class="latency-meter-marker red-marker"></div>
        </div>
        <div class="latency-meter-scale">
          <span>250</span><span>300</span><span>350</span><span>400</span><span>450</span>
        </div>
      </div>
      <div class="latency-card latency-fast">
        <div class="latency-card-top">
          <div class="latency-session amber">Session 2</div>
          <div class="latency-status-pill amber-pill">Fast Rebound</div>
        </div>
        <div class="latency-card-value amber">292 ms</div>
        <div class="latency-card-copy">The second session swung below the target band, showing a dramatic correction.</div>
        <div class="latency-meter" style="--marker: 19%;">
          <div class="latency-meter-track"></div>
          <div class="latency-meter-target"></div>
          <div class="latency-meter-marker amber-marker"></div>
        </div>
        <div class="latency-meter-scale">
          <span>250</span><span>300</span><span>350</span><span>400</span><span>450</span>
        </div>
      </div>
      <div class="latency-card latency-landing">
        <div class="latency-card-top">
          <div class="latency-session teal">Session 3</div>
          <div class="latency-status-pill teal-pill">Landed In Range</div>
        </div>
        <div class="latency-card-value teal">336 ms</div>
        <div class="latency-card-copy">By the third session the timing settled squarely inside the healthy window.</div>
        <div class="latency-meter" style="--marker: 39%;">
          <div class="latency-meter-track"></div>
          <div class="latency-meter-target"></div>
          <div class="latency-meter-marker teal-marker"></div>
        </div>
        <div class="latency-meter-scale">
          <span>250</span><span>300</span><span>350</span><span>400</span><span>450</span>
        </div>
      </div>
    </div>
    <div class="latency-journey">
      <div class="latency-journey-step"><span class="journey-dot red-glow"></span><strong>452</strong><em>start</em></div>
      <div class="latency-journey-line"></div>
      <div class="latency-journey-step"><span class="journey-dot amber-glow"></span><strong>292</strong><em>overshoot</em></div>
      <div class="latency-journey-line"></div>
      <div class="latency-journey-step"><span class="journey-dot teal-glow"></span><strong>336</strong><em>landed</em></div>
    </div>
  </div>
  <div class="caption-box slide-up" style="animation-delay:1.1s;">
    "At baseline the signal arrived late enough to support a diagnosis of mild cognitive impairment. By the third session, it was comfortably inside the healthy target range."
  </div>
</div>
"""
    css = """
.latency-band-intro {
  display:flex;
  justify-content:space-between;
  align-items:center;
  margin-bottom:22px;
  padding:0 8px;
}
.latency-band-copy {
  font-size:14px;
  letter-spacing:.24em;
  text-transform:uppercase;
  color:rgba(221,230,255,.58);
}
.latency-band-value {
  font-size:20px;
  font-weight:800;
  color:#9af1e8;
  letter-spacing:.08em;
}
.latency-grid {
  display:grid;
  grid-template-columns:repeat(3, minmax(0, 1fr));
  gap:22px;
}
.latency-card {
  position:relative;
  overflow:hidden;
  min-height:330px;
  padding:24px 24px 22px;
  border-radius:28px;
  border:1px solid rgba(255,255,255,.08);
  background:
    radial-gradient(circle at 20% 0%, rgba(255,255,255,.04), transparent 36%),
    rgba(11, 19, 36, 0.9);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.02);
}
.latency-card::after {
  content:'';
  position:absolute;
  inset:0;
  pointer-events:none;
  background:linear-gradient(180deg, rgba(255,255,255,.025), transparent 28%);
}
.latency-delay { box-shadow: inset 0 0 0 1px rgba(239,107,115,.05), 0 20px 48px rgba(0,0,0,.18); }
.latency-fast { box-shadow: inset 0 0 0 1px rgba(242,182,73,.05), 0 20px 48px rgba(0,0,0,.18); }
.latency-landing { box-shadow: inset 0 0 0 1px rgba(66,217,204,.05), 0 20px 48px rgba(0,0,0,.18); }
.latency-card-top {
  display:flex;
  justify-content:space-between;
  align-items:center;
  margin-bottom:18px;
}
.latency-session {
  font-size:14px;
  font-weight:800;
  letter-spacing:.16em;
  text-transform:uppercase;
}
.latency-status-pill {
  padding:8px 12px;
  border-radius:999px;
  font-size:12px;
  font-weight:800;
  letter-spacing:.12em;
  text-transform:uppercase;
  border:1px solid rgba(255,255,255,.08);
  background:rgba(255,255,255,.03);
}
.red-pill { color:#ff9aa0; background:rgba(239,107,115,.08); }
.amber-pill { color:#ffd894; background:rgba(242,182,73,.08); }
.teal-pill { color:#9af1e8; background:rgba(66,217,204,.08); }
.latency-card-value {
  font-size:56px;
  line-height:.95;
  font-weight:800;
  letter-spacing:-.05em;
  margin-bottom:12px;
  text-shadow:0 10px 32px rgba(0,0,0,.32);
}
.latency-card-copy {
  min-height:58px;
  font-size:17px;
  line-height:1.5;
  color:var(--muted);
  margin-bottom:28px;
}
.latency-meter {
  position:relative;
  height:98px;
  margin-bottom:10px;
}
.latency-meter-track {
  position:absolute;
  left:0;
  right:0;
  top:38px;
  height:20px;
  border-radius:999px;
  background:rgba(255,255,255,.05);
}
.latency-meter-target {
  position:absolute;
  left:24%;
  width:41%;
  top:24px;
  height:48px;
  border-radius:999px;
  background:rgba(66,217,204,.12);
  border:1px solid rgba(66,217,204,.22);
  box-shadow:0 0 24px rgba(66,217,204,.08);
}
.latency-meter-marker {
  position:absolute;
  left:var(--marker);
  top:10px;
  width:2px;
  height:70px;
  transform:translateX(-50%);
  border-radius:999px;
}
.latency-meter-marker::before {
  content:'';
  position:absolute;
  left:50%;
  top:-2px;
  width:20px;
  height:20px;
  transform:translateX(-50%);
  border-radius:50%;
  background:currentColor;
  box-shadow:0 0 22px currentColor;
}
.latency-meter-marker::after {
  content:'';
  position:absolute;
  left:50%;
  top:12px;
  width:14px;
  height:14px;
  transform:translateX(-50%);
  border-radius:50%;
  background:#f7fbff;
}
.red-marker { color: rgba(239,107,115,.95); }
.amber-marker { color: rgba(242,182,73,.95); }
.teal-marker { color: rgba(66,217,204,.95); }
.latency-meter-scale {
  display:flex;
  justify-content:space-between;
  font-size:12px;
  letter-spacing:.12em;
  text-transform:uppercase;
  color:rgba(221,230,255,.46);
}
.latency-journey {
  display:flex;
  align-items:center;
  justify-content:center;
  gap:18px;
  margin-top:24px;
  padding-top:18px;
  border-top:1px solid rgba(255,255,255,.06);
}
.latency-journey-step {
  display:flex;
  align-items:center;
  gap:10px;
  color:rgba(238,242,255,.88);
}
.latency-journey-step strong {
  font-size:24px;
  letter-spacing:-.03em;
}
.latency-journey-step em {
  font-style:normal;
  color:var(--muted);
  text-transform:uppercase;
  letter-spacing:.14em;
  font-size:12px;
}
.journey-dot {
  width:12px;
  height:12px;
  border-radius:50%;
  background:currentColor;
}
.red-glow { color: rgba(239,107,115,.95); box-shadow:0 0 18px rgba(239,107,115,.28); }
.amber-glow { color: rgba(242,182,73,.95); box-shadow:0 0 18px rgba(242,182,73,.28); }
.teal-glow { color: rgba(66,217,204,.95); box-shadow:0 0 18px rgba(66,217,204,.28); }
.latency-journey-line {
  width:70px;
  height:2px;
  background:linear-gradient(90deg, rgba(239,107,115,.35), rgba(242,182,73,.35), rgba(66,217,204,.35));
}
.red { color: var(--red); } .amber { color: var(--amber); } .teal { color: var(--teal); }
"""
    return page_html("Scene 6 - Processing Speed", body, css=css)


def scene_07_reaction_time() -> str:
    body = """
<div class="page">
  <div class="fade-in" style="animation-delay:.2s; text-align:center;">
    <div class="kicker">PHYSICAL REACTION TIME</div>
    <div class="headline">Faster and Steadier</div>
    <div class="subhead">Average speed improved, but the real visual win is how much the spread tightened.</div>
  </div>
  <div class="panel slide-up" style="animation-delay:.45s; margin:34px auto 0; width:1260px; padding:36px 52px 42px;">
    <div style="position:relative; height:380px; display:flex; align-items:flex-end; justify-content:center; gap:140px;">
      <div style="position:absolute; left:100px; right:100px; top:118px; height:92px; background:rgba(66,217,204,.08); border-top:1px dashed rgba(66,217,204,.22); border-bottom:1px dashed rgba(66,217,204,.22);"></div>
      <div style="position:absolute; right:26px; top:150px; color:#8ce9de; font-size:14px;">Target: 275–395 ms</div>
      <div class="range-col" style="--range-top: 8%; --range-height: 68%; --avg-top: 24%;">
        <div class="range-value red">378ms</div>
        <div class="range-track"><div class="range-band red-fill"></div><div class="avg-dot red-dot"></div></div>
        <div class="range-meta">±106ms spread</div><div class="range-label">Session 1</div>
      </div>
      <div class="range-col" style="--range-top: 20%; --range-height: 48%; --avg-top: 36%;">
        <div class="range-value amber">313ms</div>
        <div class="range-track"><div class="range-band amber-fill"></div><div class="avg-dot amber-dot"></div></div>
        <div class="range-meta">±67ms spread</div><div class="range-label">Session 2</div>
      </div>
      <div class="range-col" style="--range-top: 30%; --range-height: 30%; --avg-top: 42%;">
        <div class="range-value teal">284ms</div>
        <div class="range-track"><div class="range-band teal-fill"></div><div class="avg-dot teal-dot"></div></div>
        <div class="range-meta">±40ms spread</div><div class="range-label">Session 3</div>
      </div>
    </div>
  </div>
  <div class="caption-box slide-up" style="animation-delay:1.3s;">
    "This is one of those scenes where steadiness matters as much as speed. The average got faster, but the huge baseline variability tightened into something much more reliable."
  </div>
</div>
"""
    css = """
.range-col { width: 180px; text-align:center; }
.range-value { font-size: 34px; font-weight: 800; margin-bottom: 14px; }
.range-track { position:relative; height: 280px; width: 40px; margin: 0 auto; background: rgba(255,255,255,.04); border-radius: 999px; border:1px solid rgba(255,255,255,.06); }
.range-band { position:absolute; left:7px; right:7px; top: var(--range-top); height: var(--range-height); border-radius:999px; }
.avg-dot { position:absolute; left:50%; top: var(--avg-top); transform:translate(-50%,-50%); width: 22px; height: 22px; border-radius:50%; border:3px solid rgba(255,255,255,.9); background: #0d1322; box-shadow:0 0 20px rgba(255,255,255,.08); }
.range-meta { margin-top: 14px; color: var(--muted); font-size: 15px; }
.range-label { margin-top: 8px; font-size: 18px; font-weight: 700; }
.red-fill { background: linear-gradient(180deg, rgba(239,107,115,.86), rgba(145,54,62,.86)); }
.amber-fill { background: linear-gradient(180deg, rgba(242,182,73,.9), rgba(156,102,23,.86)); }
.teal-fill { background: linear-gradient(180deg, rgba(66,217,204,.9), rgba(23,136,129,.86)); }
.red-dot { border-color: rgba(239,107,115,.95); } .amber-dot { border-color: rgba(242,182,73,.95); } .teal-dot { border-color: rgba(66,217,204,.95); }
.red { color: var(--red);} .amber { color: var(--amber);} .teal { color: var(--teal);}
"""
    return page_html("Scene 7 - Reaction Time", body, css=css)


def scene_08_out_of_sync() -> str:
    body = """
<div class="page">
  <div class="fade-in" style="animation-delay:.2s; text-align:center;">
    <div class="headline">The Orchestra Was Out of Sync</div>
    <div class="subhead">Baseline coherence was dominated by one rear-temporal pathway while frontal coordination stayed weak.</div>
  </div>
  <div class="caption-box" style="max-width:920px; left:50%; transform:translateX(-50%); right:auto; bottom:32px;">
    "At baseline, one left temporal connection was so dominant that it drowned out the rest of the ensemble. The front of the brain just wasn’t coordinating yet."
  </div>
</div>
"""
    script = """
<script>
let nodes = {
  frontalL: {x: 630, y: 275},
  frontalC: {x: 832, y: 248},
  frontalR: {x: 1035, y: 278},
  temporalL: {x: 534, y: 502},
  centralL: {x: 690, y: 438},
  centralC: {x: 832, y: 448},
  temporalR: {x: 1130, y: 510},
  rear: {x: 832, y: 650},
};
function setup(){ createCanvas(windowWidth, windowHeight); }
function draw(){
  background(10,15,27,34);
  noFill();
  stroke(255,255,255,34);
  strokeWeight(3);
  beginShape();
  curveVertex(590,220); curveVertex(590,220); curveVertex(730,170); curveVertex(930,178); curveVertex(1070,228); curveVertex(1130,365); curveVertex(1090,610); curveVertex(966,744); curveVertex(706,742); curveVertex(572,610); curveVertex(530,368); curveVertex(590,220); curveVertex(590,220);
  endShape();
  const pulse = (sin(frameCount*0.06)+1)/2;
  stroke(239,107,115, 140 + pulse*80); strokeWeight(14); line(nodes.temporalL.x, nodes.temporalL.y, nodes.centralL.x, nodes.centralL.y);
  stroke(239,107,115, 255); strokeWeight(5); line(nodes.temporalL.x, nodes.temporalL.y, nodes.centralL.x, nodes.centralL.y);
  stroke(255,255,255,35); strokeWeight(4); line(nodes.frontalL.x, nodes.frontalL.y, nodes.frontalC.x, nodes.frontalC.y);
  line(nodes.frontalC.x, nodes.frontalC.y, nodes.frontalR.x, nodes.frontalR.y);
  line(nodes.frontalL.x, nodes.frontalL.y, nodes.centralL.x, nodes.centralL.y);
  for (const [key, p] of Object.entries(nodes)) {
    const active = ['temporalL','centralL'].includes(key);
    noStroke();
    fill(active ? 'rgba(239,107,115,0.18)' : 'rgba(255,255,255,0.05)');
    circle(p.x, p.y, active ? 42 + pulse*10 : 24);
    fill(active ? 'rgba(239,107,115,0.95)' : 'rgba(215,225,255,0.72)');
    circle(p.x, p.y, active ? 14 : 9);
  }
  noStroke();
  fill(239,107,115,235);
  textSize(26); textStyle(BOLD); text('0.89 — Dominant', 415, 525);
  fill(200,210,230,145);
  textSize(18); text('Front regions were still whispering', 635, 198);
}
</script>
"""
    return page_html("Scene 8 - Out of Sync", body, script=script, p5=True)


def scene_09_finds_rhythm() -> str:
    body = """
<div class="page">
  <div class="fade-in" style="animation-delay:.2s; text-align:center;">
    <div class="headline">The Orchestra Finds Its Rhythm</div>
    <div class="subhead">The over-dominant temporal line quieted down while left-to-right frontal coordination finally lit up.</div>
  </div>
  <div class="caption-box" style="max-width:920px; left:50%; transform:translateX(-50%); right:auto; bottom:32px;">
    "By the third session, the loudest baseline connection had settled. Meanwhile, the lines that were barely whispering at the front of the brain finally started carrying the melody."
  </div>
</div>
"""
    script = """
<script>
let nodes = {
  frontalL: {x: 630, y: 275},
  frontalC: {x: 832, y: 248},
  frontalR: {x: 1035, y: 278},
  temporalL: {x: 534, y: 502},
  centralL: {x: 690, y: 438},
  centralC: {x: 832, y: 448},
  temporalR: {x: 1130, y: 510},
  rear: {x: 832, y: 650},
};
function setup(){ createCanvas(windowWidth, windowHeight); }
function draw(){
  background(10,15,27,30);
  noFill();
  stroke(255,255,255,32);
  strokeWeight(3);
  beginShape();
  curveVertex(590,220); curveVertex(590,220); curveVertex(730,170); curveVertex(930,178); curveVertex(1070,228); curveVertex(1130,365); curveVertex(1090,610); curveVertex(966,744); curveVertex(706,742); curveVertex(572,610); curveVertex(530,368); curveVertex(590,220); curveVertex(590,220);
  endShape();
  const beat = (sin(frameCount*0.085)+1)/2;
  const lines = [
    ['frontalL','frontalC', [66,217,204], 7],
    ['frontalC','frontalR', [66,217,204], 7],
    ['frontalL','centralL', [106,182,255], 5],
    ['centralL','centralC', [179,140,255], 4],
    ['temporalL','centralL', [242,182,73], 2.5]
  ];
  lines.forEach(([a,b,c,w]) => {
    const [r,g,bv] = c;
    stroke(r,g,bv, 90 + beat*120);
    strokeWeight(w + beat*1.2);
    line(nodes[a].x, nodes[a].y, nodes[b].x, nodes[b].y);
  });
  for (const [key, p] of Object.entries(nodes)) {
    const active = ['frontalL','frontalC','frontalR','centralL'].includes(key);
    noStroke();
    fill(active ? 'rgba(66,217,204,0.18)' : 'rgba(255,255,255,0.05)');
    circle(p.x, p.y, active ? 40 + beat*8 : 22);
    fill(active ? 'rgba(66,217,204,0.95)' : 'rgba(215,225,255,0.68)');
    circle(p.x, p.y, active ? 12 : 8);
  }
  noStroke();
  fill(66,217,204,235);
  textSize(24); textStyle(BOLD); text('0.36 — Settled', 418, 525);
  fill(143,239,227,220);
  text('0.87 — Now in Sync', 692, 190);
}
</script>
"""
    return page_html("Scene 9 - Finds Rhythm", body, script=script, p5=True)


def scene_10_building_session_by_session() -> str:
    body = """
<div class="page">
  <div class="fade-in" style="animation-delay:.2s; text-align:center;">
    <div class="headline">Fronto-Central Coordination: Building Session by Session</div>
    <div class="subhead">This was not a sudden leap. The same pathway strengthened steadily across all three recordings.</div>
  </div>
  <div style="display:flex; justify-content:center; gap:32px; margin-top:36px;">
    <div class="panel slide-up mini-brain" style="animation-delay:.45s;">
      <div class="mini-kicker">Session 1</div>
      <div class="mini-status blue">Weak but present</div>
      <svg viewBox="0 0 340 320" class="mini-svg">
        <ellipse cx="170" cy="156" rx="118" ry="108" fill="none" stroke="rgba(255,255,255,.14)" stroke-width="3"/>
        <line x1="132" y1="126" x2="176" y2="154" stroke="rgba(106,182,255,.55)" stroke-width="12" stroke-linecap="round"/>
        <circle cx="132" cy="126" r="16" fill="rgba(106,182,255,.18)"/><circle cx="176" cy="154" r="16" fill="rgba(106,182,255,.18)"/>
        <circle cx="132" cy="126" r="10" fill="#6ab6ff"/><circle cx="176" cy="154" r="10" fill="#6ab6ff"/>
        <text x="98" y="108" fill="rgba(221,230,255,.62)" font-size="12" letter-spacing="2">FRONT</text>
        <text x="178" y="182" fill="rgba(221,230,255,.62)" font-size="12" letter-spacing="2">CENTER</text>
      </svg>
      <div class="metric-chip blue mini-value-chip">0.43</div>
      <div class="mini-copy">The pathway was visible, but still too quiet to anchor executive control.</div>
    </div>
    <div class="panel slide-up mini-brain" style="animation-delay:.7s;">
      <div class="mini-kicker">Session 2</div>
      <div class="mini-status teal">Clearly building</div>
      <svg viewBox="0 0 340 320" class="mini-svg">
        <ellipse cx="170" cy="156" rx="118" ry="108" fill="none" stroke="rgba(255,255,255,.14)" stroke-width="3"/>
        <line x1="132" y1="126" x2="176" y2="154" stroke="rgba(66,217,204,.72)" stroke-width="16" stroke-linecap="round"/>
        <circle cx="132" cy="126" r="19" fill="rgba(66,217,204,.18)"/><circle cx="176" cy="154" r="19" fill="rgba(66,217,204,.18)"/>
        <circle cx="132" cy="126" r="10" fill="#42d9cc"/><circle cx="176" cy="154" r="10" fill="#42d9cc"/>
        <text x="98" y="108" fill="rgba(221,230,255,.62)" font-size="12" letter-spacing="2">FRONT</text>
        <text x="178" y="182" fill="rgba(221,230,255,.62)" font-size="12" letter-spacing="2">CENTER</text>
      </svg>
      <div class="metric-chip teal mini-value-chip">0.68</div>
      <div class="mini-copy">The line thickened into a meaningful coordination bridge by the midpoint.</div>
    </div>
    <div class="panel slide-up mini-brain" style="animation-delay:.95s;">
      <div class="mini-kicker">Session 3</div>
      <div class="mini-status amber">Strong connection</div>
      <svg viewBox="0 0 340 320" class="mini-svg">
        <ellipse cx="170" cy="156" rx="118" ry="108" fill="none" stroke="rgba(255,255,255,.14)" stroke-width="3"/>
        <line x1="132" y1="126" x2="176" y2="154" stroke="rgba(242,182,73,.92)" stroke-width="20" stroke-linecap="round"/>
        <circle cx="132" cy="126" r="22" fill="rgba(242,182,73,.18)"/><circle cx="176" cy="154" r="22" fill="rgba(242,182,73,.18)"/>
        <circle cx="132" cy="126" r="10" fill="#f2b649"/><circle cx="176" cy="154" r="10" fill="#f2b649"/>
        <text x="98" y="108" fill="rgba(221,230,255,.62)" font-size="12" letter-spacing="2">FRONT</text>
        <text x="178" y="182" fill="rgba(221,230,255,.62)" font-size="12" letter-spacing="2">CENTER</text>
      </svg>
      <div class="metric-chip amber mini-value-chip">0.81</div>
      <div class="mini-copy">By the final scan the same path read as an obvious high-confidence coordination lane.</div>
    </div>
  </div>
  <div class="caption-box slide-up" style="animation-delay:1.15s;">
    "The same front-to-center pathway climbed from point four three to point six eight to point eight one. This is exactly what a real trajectory looks like."
  </div>
</div>
"""
    css = """
.mini-brain { width: 360px; padding: 24px 24px 28px; text-align:center; }
.mini-svg { width: 100%; height: 220px; display:block; margin-top: 4px; }
.mini-kicker { font-size: 14px; letter-spacing: .2em; text-transform: uppercase; color: var(--muted); margin-bottom: 10px; }
.mini-status { font-size: 13px; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; margin-bottom: 8px; }
.mini-value-chip { font-size: 18px; padding: 12px 20px; margin-bottom: 14px; }
.mini-copy { font-size: 15px; line-height: 1.5; color: var(--muted); max-width: 250px; margin: 0 auto; }
"""
    return page_html("Scene 10 - Session by Session", body, css=css)


def scene_11_alpha_balance() -> str:
    body = """
<div class="page">
  <div class="fade-in" style="animation-delay:.2s; text-align:center;">
    <div class="headline">Frontal Alpha Balance: Left-Right Harmony</div>
    <div class="subhead">A tilted baseline ratio moved back into healthy left-right balance by the second session and stayed there.</div>
    <div class="balance-target-banner">Target band: 0.9–1.1</div>
  </div>
  <div style="display:flex; justify-content:center; gap:34px; margin-top:38px;">
    <div class="panel slide-up balance-card" style="animation-delay:.45s;">
      <div class="balance-title">Session 1</div>
      <div class="balance-status amber">Tilted left</div>
      <div class="scale">
        <div class="beam" style="transform:rotate(-10deg);"></div>
        <div class="fulcrum"></div>
        <div class="pan left-pan heavy"></div>
        <div class="pan right-pan"></div>
        <div class="pan-label left-label">LEFT</div>
        <div class="pan-label right-label">RIGHT</div>
      </div>
      <div class="metric-chip amber balance-chip">0.7</div>
      <div class="balance-note">Below the healthy 1.0 balance target</div>
      <div class="balance-copy">Baseline alpha was skewed enough that one frontal side was carrying more quiet-idle activity than the other.</div>
    </div>
    <div class="panel slide-up balance-card" style="animation-delay:.7s;">
      <div class="balance-title">Session 2</div>
      <div class="balance-status teal">Centered</div>
      <div class="scale">
        <div class="beam" style="transform:rotate(0deg);"></div>
        <div class="fulcrum"></div>
        <div class="pan left-pan"></div>
        <div class="pan right-pan"></div>
        <div class="pan-label left-label">LEFT</div>
        <div class="pan-label right-label">RIGHT</div>
      </div>
      <div class="metric-chip teal balance-chip">1.0</div>
      <div class="balance-note">Returned to the healthy midpoint</div>
      <div class="balance-copy">By the second session the left and right frontal regions were carrying comparable alpha load again.</div>
    </div>
    <div class="panel slide-up balance-card" style="animation-delay:.95s;">
      <div class="balance-title">Session 3</div>
      <div class="balance-status teal">Held steady</div>
      <div class="scale">
        <div class="beam" style="transform:rotate(6deg);"></div>
        <div class="fulcrum"></div>
        <div class="pan left-pan"></div>
        <div class="pan right-pan heavy-right"></div>
        <div class="pan-label left-label">LEFT</div>
        <div class="pan-label right-label">RIGHT</div>
      </div>
      <div class="metric-chip teal balance-chip">1.1</div>
      <div class="balance-note">Held at the upper edge of the target band</div>
      <div class="balance-copy">The balance stayed inside range, but now with a slight rightward lean exactly where the data pack places it.</div>
    </div>
  </div>
  <div class="caption-box slide-up" style="animation-delay:1.15s;">
    "This is a quieter finding, but it matters. The frontal balance that started tilted returned to center quickly and then stayed there."
  </div>
</div>
"""
    css = """
.balance-card { width: 340px; text-align:center; padding: 26px 24px 28px; }
.balance-target-banner { margin-top: 12px; font-size: 14px; letter-spacing: .16em; text-transform: uppercase; color: #9af1e8; }
.balance-title { font-size: 18px; font-weight: 800; margin-bottom: 8px; }
.balance-status { font-size: 13px; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; margin-bottom: 10px; }
.scale { position: relative; width: 240px; height: 182px; margin: 12px auto 18px; }
.scale::before { content:''; position:absolute; left:118px; top:34px; width:4px; height:96px; background:rgba(255,255,255,.16); }
.beam { position:absolute; left:48px; top:38px; width:144px; height:8px; background:linear-gradient(90deg, rgba(242,182,73,.78), rgba(66,217,204,.74)); border-radius:999px; transform-origin:center; box-shadow:0 0 16px rgba(255,255,255,.05); }
.fulcrum { position:absolute; left:107px; top:110px; width:26px; height:20px; background:rgba(255,255,255,.08); clip-path: polygon(50% 0, 100% 100%, 0 100%); }
.pan { position:absolute; top:76px; width:58px; height:58px; border-radius:50%; border:2px solid rgba(255,255,255,.12); background:rgba(255,255,255,.04); box-shadow: inset 0 1px 0 rgba(255,255,255,.04); }
.left-pan { left:30px; } .right-pan { right:30px; }
.heavy { box-shadow: 0 0 0 12px rgba(242,182,73,.12); border-color: rgba(242,182,73,.42); }
.heavy-right { box-shadow: 0 0 0 12px rgba(66,217,204,.12); border-color: rgba(66,217,204,.42); }
.pan-label { position:absolute; top:58px; font-size:11px; letter-spacing:.16em; text-transform:uppercase; color:rgba(221,230,255,.48); font-weight:700; }
.left-label { left:24px; } .right-label { right:18px; }
.balance-chip { font-size: 18px; padding: 12px 18px; }
.balance-note { margin-top: 12px; color: var(--muted); font-size: 15px; }
.balance-copy { margin-top: 12px; color: var(--muted); font-size: 15px; line-height: 1.5; }
"""
    return page_html("Scene 11 - Alpha Balance", body, css=css)


def scene_12_frequency_puzzle() -> str:
    body = """
<div class="page">
  <div class="fade-in" style="animation-delay:.2s; text-align:center;">
    <div class="headline">The Frequency Puzzle</div>
    <div class="subhead">One region slowed beautifully into range. Another stayed stubbornly fast.</div>
  </div>
  <div style="display:flex; justify-content:center; gap:32px; margin-top:36px;">
    <div class="panel slide-up region-card" style="animation-delay:.45s;">
      <div class="region-title teal">Central-Parietal Region</div>
      <div class="region-sub">11.4 Hz → 9.1 Hz</div>
      <svg viewBox="0 0 420 220" class="region-svg">
        <line x1="50" y1="180" x2="360" y2="180" stroke="rgba(255,255,255,.08)" stroke-width="2"/>
        <rect x="50" y="104" width="310" height="44" rx="12" fill="rgba(66,217,204,.10)" stroke="rgba(66,217,204,.2)"/>
        <text x="205" y="132" text-anchor="middle" fill="rgba(140,233,222,.88)" font-size="15">Target range: 8.5–10.5 Hz</text>
        <polyline points="90,54 205,95 320,139" fill="none" stroke="rgba(66,217,204,.9)" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>
        <circle cx="90" cy="54" r="9" fill="#6ab6ff"/><circle cx="205" cy="95" r="9" fill="#42d9cc"/><circle cx="320" cy="139" r="9" fill="#42d9cc"/>
      </svg>
      <div class="region-foot">Normalized into the healthy idle-speed window</div>
    </div>
    <div class="panel slide-up region-card" style="animation-delay:.7s;">
      <div class="region-title amber">Occipital Region</div>
      <div class="region-sub">10.5 Hz → 11.5 Hz</div>
      <svg viewBox="0 0 420 220" class="region-svg">
        <line x1="50" y1="180" x2="360" y2="180" stroke="rgba(255,255,255,.08)" stroke-width="2"/>
        <rect x="50" y="104" width="310" height="44" rx="12" fill="rgba(242,182,73,.08)" stroke="rgba(242,182,73,.18)"/>
        <text x="205" y="132" text-anchor="middle" fill="rgba(255,213,129,.85)" font-size="15">Target range: 9.0–11.0 Hz</text>
        <polyline points="90,90 205,74 320,52" fill="none" stroke="rgba(242,182,73,.92)" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>
        <circle cx="90" cy="90" r="9" fill="#f2b649"/><circle cx="205" cy="74" r="9" fill="#f2b649"/><circle cx="320" cy="52" r="9" fill="#f2b649"/>
      </svg>
      <div class="region-foot">Drifted slightly above the occipital target ceiling by Session 3</div>
    </div>
  </div>
  <div class="caption-box slide-up" style="animation-delay:1s;">
    "This is why the report feels honest. Some systems normalized cleanly. Others improved, but not all the way. That tension makes the story more believable, not less."
  </div>
</div>
"""
    css = """
.region-card { width: 520px; padding: 28px; }
.region-title { font-size: 28px; font-weight: 800; margin-bottom: 6px; }
.region-sub { font-size: 20px; color: var(--muted); margin-bottom: 8px; }
.region-svg { width: 100%; height: 220px; display:block; }
.region-foot { color: var(--muted); font-size: 16px; margin-top: 8px; }
"""
    return page_html("Scene 12 - Frequency Puzzle", body, css=css)


def scene_13_state_ratios() -> str:
    body = """
<div class="page">
  <div class="fade-in" style="animation-delay:.2s; text-align:center;">
    <div class="kicker">THETA-BETA RATIO</div>
    <div class="headline">State Ratios: Quiet Normalization</div>
    <div class="subhead">Small movement, but in the right direction — and into the healthy floor of the target zone.</div>
  </div>
  <div class="panel slide-up ratio-panel" style="animation-delay:.45s;">
    <div class="ratio-left">
      <div class="ratio-kicker">Quiet but consistent</div>
      <div class="ratio-big">0.5 → 0.4 → 0.6</div>
      <div class="ratio-copy">This marker dipped further below range in Session 2, then recovered to the exact floor of the healthy band by Session 3.</div>
      <div class="ratio-target-callout">
        <span class="target-dot"></span>
        Target band <strong>0.6–1.5</strong>; floor begins at <strong>0.6</strong>
      </div>
    </div>
    <div class="ratio-right">
      <div class="ratio-track-wrap">
        <div class="ratio-axis">
          <span>0.3</span><span>0.4</span><span>0.5</span><span>0.6</span><span>0.7</span><span>0.8</span>
        </div>
        <div class="ratio-track">
          <div class="ratio-target-band"></div>
          <div class="ratio-marker amber-marker" style="left:40%;">
            <div class="ratio-marker-label">S1</div>
            <div class="ratio-marker-value">0.5</div>
          </div>
          <div class="ratio-marker blue-marker" style="left:20%;">
            <div class="ratio-marker-label">S2</div>
            <div class="ratio-marker-value">0.4</div>
          </div>
          <div class="ratio-marker teal-marker" style="left:60%;">
            <div class="ratio-marker-label">S3</div>
            <div class="ratio-marker-value">0.6</div>
          </div>
          <svg viewBox="0 0 620 120" class="ratio-svg">
            <polyline points="248,60 124,84 372,36" fill="none" stroke="rgba(221,230,255,.38)" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </div>
      </div>
      <div class="ratio-session-row">
        <div class="ratio-session-card amber-glass">
          <div class="ratio-session-title">Session 1</div>
          <div class="ratio-session-status">Below range</div>
        </div>
        <div class="ratio-session-card blue-glass">
          <div class="ratio-session-title">Session 2</div>
          <div class="ratio-session-status">Dipped lower</div>
        </div>
        <div class="ratio-session-card teal-glass">
          <div class="ratio-session-title">Session 3</div>
          <div class="ratio-session-status">Reached threshold</div>
        </div>
      </div>
    </div>
  </div>
  <div class="caption-box slide-up" style="animation-delay:1.05s;">
    "The theta-beta ratio wasn’t the headline, but it moved in the right direction and reached the exact floor of the healthy target band by Session Three."
  </div>
</div>
"""
    css = """
.ratio-panel {
  width: 1180px;
  margin: 40px auto 0;
  padding: 36px 40px 34px;
  display: grid;
  grid-template-columns: 320px 1fr;
  gap: 34px;
  align-items: center;
}
.ratio-kicker {
  font-size: 13px;
  letter-spacing: .18em;
  text-transform: uppercase;
  color: rgba(221,230,255,.52);
  margin-bottom: 12px;
}
.ratio-big {
  font-size: 48px;
  line-height: .96;
  font-weight: 800;
  letter-spacing: -.05em;
  margin-bottom: 16px;
}
.ratio-copy {
  color: var(--muted);
  font-size: 18px;
  line-height: 1.55;
  margin-bottom: 22px;
}
.ratio-target-callout {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 12px 16px;
  border-radius: 999px;
  border: 1px solid rgba(66,217,204,.18);
  background: rgba(66,217,204,.07);
  color: rgba(238,242,255,.84);
}
.target-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: #42d9cc;
  box-shadow: 0 0 18px rgba(66,217,204,.24);
}
.ratio-track-wrap {
  position: relative;
  padding-top: 26px;
}
.ratio-axis {
  display: flex;
  justify-content: space-between;
  color: rgba(221,230,255,.46);
  font-size: 12px;
  letter-spacing: .12em;
  text-transform: uppercase;
  margin-bottom: 12px;
}
.ratio-track {
  position: relative;
  height: 120px;
  border-radius: 24px;
  background: linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.015));
  border: 1px solid rgba(255,255,255,.06);
  overflow: hidden;
}
.ratio-target-band {
  position: absolute;
  left: 60%;
  right: 0;
  top: 0;
  bottom: 0;
  background:
    linear-gradient(180deg, rgba(66,217,204,.12), rgba(66,217,204,.05)),
    repeating-linear-gradient(90deg, rgba(255,255,255,.05) 0 1px, transparent 1px 22px);
  border-left: 2px dashed rgba(66,217,204,.35);
}
.ratio-svg {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
}
.ratio-marker {
  position: absolute;
  top: 18px;
  transform: translateX(-50%);
  text-align: center;
  z-index: 2;
}
.ratio-marker::before {
  content: '';
  position: absolute;
  left: 50%;
  top: 34px;
  width: 3px;
  height: 42px;
  transform: translateX(-50%);
  background: currentColor;
  border-radius: 999px;
  opacity: .85;
}
.ratio-marker::after {
  content: '';
  position: absolute;
  left: 50%;
  top: 24px;
  width: 18px;
  height: 18px;
  transform: translateX(-50%);
  border-radius: 50%;
  background: currentColor;
  box-shadow: 0 0 18px currentColor;
}
.ratio-marker-label {
  font-size: 12px;
  letter-spacing: .12em;
  text-transform: uppercase;
  font-weight: 800;
}
.ratio-marker-value {
  margin-top: 4px;
  font-size: 28px;
  font-weight: 800;
  letter-spacing: -.04em;
}
.ratio-session-row {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
  margin-top: 18px;
}
.ratio-session-card {
  padding: 14px 16px;
  border-radius: 18px;
  border: 1px solid rgba(255,255,255,.06);
  background: rgba(255,255,255,.03);
}
.ratio-session-title {
  font-size: 13px;
  letter-spacing: .14em;
  text-transform: uppercase;
  color: rgba(221,230,255,.5);
  margin-bottom: 8px;
}
.ratio-session-status {
  font-size: 18px;
  font-weight: 700;
}
.amber-glass { box-shadow: inset 0 0 0 1px rgba(242,182,73,.04); }
.blue-glass { box-shadow: inset 0 0 0 1px rgba(106,182,255,.04); }
.teal-glass { box-shadow: inset 0 0 0 1px rgba(66,217,204,.04); }
"""
    return page_html("Scene 13 - State Ratios", body, css=css)


def scene_14_verdict() -> str:
    body = """
<div class="page">
  <div class="fade-in" style="animation-delay:.2s; text-align:center;">
    <div class="headline">Verdict: The Data Speaks</div>
    <div class="subhead">Across sixty-five days, the gains show up in power, speed, timing, and network behavior.</div>
  </div>
  <div class="panel slide-up" style="animation-delay:.45s; width:1260px; margin:38px auto 0; padding:34px 38px 30px;">
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:18px;">
      <div class="summary-row"><span class="summary-dot teal"></span><div><strong>Signal Strength</strong><span> P300 voltage climbed from 13.1 to 24.0 µV.</span></div></div>
      <div class="summary-row"><span class="summary-dot teal"></span><div><strong>Processing Speed</strong><span> Latency moved from 452 ms into the healthy target zone.</span></div></div>
      <div class="summary-row"><span class="summary-dot teal"></span><div><strong>Reaction Time</strong><span> Faster averages with far tighter spread.</span></div></div>
      <div class="summary-row"><span class="summary-dot amber"></span><div><strong>Network Rhythm</strong><span> Dominant baseline coherence quieted while better pathways strengthened.</span></div></div>
      <div class="summary-row"><span class="summary-dot amber"></span><div><strong>Alpha Balance</strong><span> Left-right frontal ratio returned to range and stayed there.</span></div></div>
      <div class="summary-row"><span class="summary-dot violet"></span><div><strong>The Honest Puzzle</strong><span> Occipital frequency improved, but not all the way — a real open question remains.</span></div></div>
    </div>
    <div style="margin-top:28px; display:flex; justify-content:center;">
      <div class="metric-chip teal" style="font-size:16px; padding:14px 22px;">Overall read: consistent, multi-layered positive change</div>
    </div>
  </div>
  <div class="caption-box slide-up" style="animation-delay:1s;">
    "The most encouraging part is not one miracle number. It’s that multiple systems moved together in the same positive direction. That’s what makes the story feel real."
  </div>
</div>
"""
    css = """
.summary-row { display:flex; gap:14px; align-items:flex-start; padding:16px 18px; border-radius:18px; background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.05); font-size:18px; line-height:1.45; color:var(--muted); }
.summary-row strong { color: var(--text); font-weight:800; }
.summary-dot { width:14px; height:14px; border-radius:50%; margin-top:7px; flex-shrink:0; box-shadow:0 0 16px currentColor; }
.summary-dot.teal { color: var(--teal); background: var(--teal); }
.summary-dot.amber { color: var(--amber); background: var(--amber); }
.summary-dot.violet { color: var(--violet); background: var(--violet); }
"""
    return page_html("Scene 14 - Verdict", body, css=css)


def build_project(project_dir: Path) -> list[Path]:
    scene_dir = project_dir / "scene_artifacts"
    scene_dir.mkdir(parents=True, exist_ok=True)

    scenes = [
        ("scene_01_title_card.html", scene_01_title_card()),
        ("scene_02_roadmap.html", scene_02_roadmap()),
        ("scene_03_timeline_baseline.html", scene_03_timeline()),
        ("scene_04_brain_orchestra.html", scene_04_orchestra_intro()),
        ("scene_05_p300_signal_strength.html", scene_05_signal_strength()),
        ("scene_06_processing_speed_latency.html", scene_06_processing_speed()),
        ("scene_07_reaction_time_range.html", scene_07_reaction_time()),
        ("scene_08_orchestra_out_of_sync.html", scene_08_out_of_sync()),
        ("scene_09_orchestra_finds_rhythm.html", scene_09_finds_rhythm()),
        ("scene_10_fronto_central_coordination.html", scene_10_building_session_by_session()),
        ("scene_11_frontal_alpha_balance.html", scene_11_alpha_balance()),
        ("scene_12_frequency_puzzle.html", scene_12_frequency_puzzle()),
        ("scene_13_state_ratios_gauge.html", scene_13_state_ratios()),
        ("scene_14_full_picture.html", scene_14_verdict()),
    ]

    written: list[Path] = []
    for name, html in scenes:
        path = scene_dir / name
        path.write_text(html)
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hand-rolled HTML scene artifacts for a patient project.")
    parser.add_argument("project_dir", type=Path)
    args = parser.parse_args()
    project_dir = args.project_dir.resolve()
    if not (project_dir / "plan.json").exists():
        raise SystemExit(f"Missing plan.json in {project_dir}")
    written = build_project(project_dir)
    print(f"Wrote {len(written)} scene artifacts to {project_dir / 'scene_artifacts'}")
    for path in written:
        print(path.name)


if __name__ == "__main__":
    main()
