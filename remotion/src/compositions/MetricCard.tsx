import React from "react";
import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import type { MetricCardProps } from "../types";
import { FONT_HEADLINE, FONT_DATA, FONT_BODY, COLOR_ACCENT } from "../types";
import { SceneLayout, Headline, Caption, useReveal } from "./shared";

export const MetricCard: React.FC<MetricCardProps> = ({
  headline,
  metricName,
  beforeValue,
  afterValue,
  delta,
  caption,
  backgroundImage,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const reveal = useReveal();

  const beforeReveal = spring({
    frame: Math.max(frame - 12, 0),
    fps,
    config: { damping: 200 },
  });
  const afterReveal = spring({
    frame: Math.max(frame - 20, 0),
    fps,
    config: { damping: 16, stiffness: 120, mass: 0.8 },
  });
  const deltaReveal = spring({
    frame: Math.max(frame - 30, 0),
    fps,
    config: { damping: 200 },
  });

  const afterGlow = interpolate(afterReveal, [0.5, 1], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

  return (
    <SceneLayout backgroundImage={backgroundImage} brightness={0.3}>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", gap: 24 }}>
        <Headline text={headline} />
        {metricName && (
          <div style={{
            fontFamily: FONT_DATA, fontSize: 18, letterSpacing: "0.1em",
            textTransform: "uppercase", color: COLOR_ACCENT,
            opacity: interpolate(reveal, [0.1, 0.5], [0, 1]),
          }}>
            {metricName}
          </div>
        )}
        <div style={{ display: "flex", gap: 64, alignItems: "center", marginTop: 8 }}>
          {/* Before panel */}
          <div style={{
            display: "flex", flexDirection: "column", alignItems: "center", gap: 8,
            opacity: beforeReveal,
            transform: `translateX(${interpolate(beforeReveal, [0, 1], [-20, 0])}px)`,
          }}>
            <div style={{ fontFamily: FONT_DATA, fontSize: 14, letterSpacing: "0.12em", textTransform: "uppercase", color: "rgba(255,244,234,0.4)" }}>
              Before
            </div>
            <div style={{ fontFamily: FONT_DATA, fontSize: 60, fontWeight: 700, color: "rgba(255,244,234,0.7)" }}>
              {beforeValue}
            </div>
          </div>
          {/* Arrow */}
          <div style={{
            fontFamily: FONT_DATA, fontSize: 36, color: COLOR_ACCENT,
            opacity: interpolate(afterReveal, [0, 0.5], [0, 1]),
            transform: `translateX(${interpolate(afterReveal, [0, 1], [-8, 0])}px)`,
          }}>
            →
          </div>
          {/* After panel */}
          <div style={{
            display: "flex", flexDirection: "column", alignItems: "center", gap: 8,
            opacity: afterReveal,
            transform: `scale(${interpolate(afterReveal, [0, 1], [0.85, 1])})`,
          }}>
            <div style={{ fontFamily: FONT_DATA, fontSize: 14, letterSpacing: "0.12em", textTransform: "uppercase", color: COLOR_ACCENT }}>
              After
            </div>
            <div style={{
              fontFamily: FONT_DATA, fontSize: 68, fontWeight: 700, color: "#fff",
              textShadow: `0 0 ${40 * afterGlow}px ${COLOR_ACCENT}44, 0 0 ${80 * afterGlow}px ${COLOR_ACCENT}22`,
            }}>
              {afterValue}
            </div>
          </div>
        </div>
        {delta && (
          <div style={{
            fontFamily: FONT_BODY, fontSize: 26, color: COLOR_ACCENT, fontWeight: 600,
            opacity: deltaReveal,
            transform: `translateY(${interpolate(deltaReveal, [0, 1], [12, 0])}px)`,
            textShadow: `0 0 20px ${COLOR_ACCENT}33`,
          }}>
            {delta}
          </div>
        )}
        {caption && <Caption text={caption} />}
      </div>
    </SceneLayout>
  );
};
