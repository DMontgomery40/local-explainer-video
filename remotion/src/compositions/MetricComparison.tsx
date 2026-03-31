import React from "react";
import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import type { MetricComparisonProps } from "../types";
import { FONT_DATA, COLOR_ACCENT } from "../types";
import { SceneLayout, Headline, Caption, useReveal } from "./shared";

export const MetricComparison: React.FC<MetricComparisonProps> = ({
  headline,
  leftLabel,
  rightLabel,
  leftValue,
  rightValue,
  caption,
  backgroundImage,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const reveal = useReveal();
  const panelReveal = spring({
    frame: Math.max(frame - 10, 0),
    fps,
    config: { damping: 18, stiffness: 100, mass: 0.9 },
  });

  const panels = [
    { label: leftLabel, value: leftValue, color: "#60a5fa" },
    { label: rightLabel, value: rightValue, color: COLOR_ACCENT },
  ];

  return (
    <SceneLayout backgroundImage={backgroundImage}>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", gap: 36 }}>
        <Headline text={headline} />
        <div style={{ display: "flex", gap: 64, alignItems: "stretch" }}>
          {panels.map((p, i) => (
            <div
              key={i}
              style={{
                flex: 1,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: 12,
                padding: "32px 48px",
                background: "rgba(255,255,255,0.04)",
                borderRadius: 16,
                opacity: panelReveal,
                transform: `translateX(${interpolate(panelReveal, [0, 1], [i === 0 ? -24 : 24, 0])}px)`,
              }}
            >
              <div style={{ fontFamily: FONT_DATA, fontSize: 16, letterSpacing: "0.1em", textTransform: "uppercase", color: p.color }}>
                {p.label}
              </div>
              <div style={{ fontFamily: FONT_DATA, fontSize: 52, fontWeight: 700, color: "#fff" }}>
                {p.value}
              </div>
            </div>
          ))}
        </div>
        {caption && <Caption text={caption} />}
      </div>
    </SceneLayout>
  );
};
