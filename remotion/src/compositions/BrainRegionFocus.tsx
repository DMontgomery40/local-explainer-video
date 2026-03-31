import React from "react";
import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import type { BrainRegionFocusProps } from "../types";
import { FONT_BODY, FONT_DATA, COLOR_ACCENT } from "../types";
import { SceneLayout, Headline, Caption, useReveal } from "./shared";

const REGION_POSITIONS: Record<string, { top: number; left: number }> = {
  frontal:            { top: 200, left: 832 },
  prefrontal:         { top: 140, left: 832 },
  central:            { top: 310, left: 832 },
  parietal:           { top: 420, left: 832 },
  "central-parietal": { top: 365, left: 832 },
  temporal:           { top: 380, left: 480 },
  occipital:          { top: 580, left: 832 },
  fp1: { top: 135, left: 680 },  fp2: { top: 135, left: 984 },
  f3:  { top: 210, left: 620 },  f4:  { top: 210, left: 1044 },
  fz:  { top: 195, left: 832 },
  c3:  { top: 310, left: 560 },  c4:  { top: 310, left: 1100 },
  cz:  { top: 290, left: 832 },
  t3:  { top: 350, left: 440 },  t4:  { top: 350, left: 1220 },
  p3:  { top: 430, left: 620 },  p4:  { top: 430, left: 1040 },
  pz:  { top: 420, left: 832 },
  o1:  { top: 560, left: 700 },  o2:  { top: 560, left: 964 },
  oz:  { top: 580, left: 832 },
};

const STATUS_COLORS: Record<string, string> = {
  improved: COLOR_ACCENT,
  stable: "#60a5fa",
  declined: "#fbbf24",
  flagged: "#f87171",
};

function resolvePosition(name: string, fallbackIndex: number) {
  const key = name.toLowerCase().replace(/[\s_-]+/g, "");
  const match = Object.entries(REGION_POSITIONS).find(
    ([k]) => key.includes(k) || k.includes(key),
  );
  return match?.[1] ?? { top: 300 + fallbackIndex * 60, left: 400 + fallbackIndex * 80 };
}

export const BrainRegionFocus: React.FC<BrainRegionFocusProps> = ({
  headline,
  caption,
  regions,
  backgroundImage,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const pulse = Math.sin(frame * 0.12) * 0.3 + 0.7;

  return (
    <SceneLayout
      backgroundImage={backgroundImage ?? "backgrounds/brain_region_focus_topdown.png"}
      brightness={0.45}
      scrimGradient="linear-gradient(180deg, rgba(3,5,10,0.72) 0%, rgba(3,5,10,0.2) 35%, rgba(3,5,10,0.2) 65%, rgba(3,5,10,0.72) 100%)"
    >
      <Headline text={headline} fontSize={48} />

      {regions.map((reg, i) => {
        const regionReveal = spring({
          frame: Math.max(frame - 14 - i * 8, 0),
          fps,
          config: { damping: 14, stiffness: 100, mass: 0.8 },
        });
        const pos = resolvePosition(reg.name, i);
        const color = STATUS_COLORS[reg.status ?? "stable"] ?? "#60a5fa";
        const glowIntensity = regionReveal * pulse;

        return (
          <div key={i} style={{ position: "absolute", top: pos.top, left: pos.left, transform: "translate(-50%, -50%)" }}>
            {/* Pulsing glow dot */}
            <div style={{
              position: "absolute", top: "50%", left: "50%",
              transform: "translate(-50%, -50%)",
              width: 12, height: 12, borderRadius: "50%",
              background: color,
              boxShadow: `0 0 ${20 * glowIntensity}px ${10 * glowIntensity}px ${color}88, 0 0 ${40 * glowIntensity}px ${20 * glowIntensity}px ${color}44`,
              opacity: regionReveal,
            }} />
            {/* Label card */}
            <div style={{
              display: "flex", flexDirection: "column", alignItems: "center",
              gap: 2, marginTop: 18,
              opacity: regionReveal,
              transform: `translateY(${interpolate(regionReveal, [0, 1], [8, 0])}px)`,
            }}>
              <div style={{
                fontFamily: FONT_BODY, fontSize: 24, fontWeight: 700, color: "#fff",
                textShadow: "0 2px 8px rgba(0,0,0,0.7), 0 0 4px rgba(0,0,0,0.5)",
                whiteSpace: "nowrap",
              }}>
                {reg.name}
              </div>
              {reg.value && (
                <div style={{
                  fontFamily: FONT_DATA, fontSize: 20, color,
                  textShadow: `0 1px 6px rgba(0,0,0,0.6), 0 0 12px ${color}44`,
                  whiteSpace: "nowrap",
                }}>
                  {reg.value}
                </div>
              )}
            </div>
          </div>
        );
      })}

      {caption && <Caption text={caption} />}
    </SceneLayout>
  );
};
