import React from "react";
import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import type { AnalogySplitProps } from "../types";
import { FONT_BODY, FONT_DATA, COLOR_ACCENT } from "../types";
import { SceneLayout, Headline, Caption, useReveal, useStaggeredReveal } from "./shared";

const ACCENT_MAP: Record<string, string> = {
  teal: COLOR_ACCENT,
  amber: "#fbbf24",
  blue: "#60a5fa",
  green: "#4ade80",
};

const Panel: React.FC<{
  title: string;
  items: string[];
  accent?: string;
  fromX: number;
}> = ({ title, items, accent = "teal", fromX }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const panelReveal = spring({
    frame: Math.max(frame - 8, 0),
    fps,
    config: { damping: 18, stiffness: 100, mass: 0.9 },
  });
  const color = ACCENT_MAP[accent] ?? COLOR_ACCENT;

  return (
    <div
      style={{
        flex: 1,
        padding: "28px 24px",
        display: "flex",
        flexDirection: "column",
        gap: 14,
        transform: `translateX(${interpolate(panelReveal, [0, 1], [fromX, 0])}px)`,
        opacity: panelReveal,
      }}
    >
      <div style={{ fontFamily: FONT_DATA, fontSize: 22, letterSpacing: "0.1em", textTransform: "uppercase", color, fontWeight: 700 }}>
        {title}
      </div>
      {items.map((item, i) => {
        const itemReveal = useStaggeredReveal(i);
        return (
          <div
            key={i}
            style={{
              fontFamily: FONT_BODY,
              fontSize: 24,
              lineHeight: 1.45,
              opacity: itemReveal,
              transform: `translateY(${interpolate(itemReveal, [0, 1], [8, 0])}px)`,
            }}
          >
            {item}
          </div>
        );
      })}
    </div>
  );
};

export const AnalogySplit: React.FC<AnalogySplitProps> = ({
  headline,
  left,
  right,
  caption,
  backgroundImage,
}) => (
  <SceneLayout backgroundImage={backgroundImage}>
    <div style={{ display: "flex", flexDirection: "column", gap: 28, height: "100%" }}>
      <Headline text={headline} />
      <div style={{ display: "flex", gap: 24, flex: 1, alignItems: "stretch" }}>
        <Panel title={left.title} items={left.items} accent={left.accent} fromX={-32} />
        <Panel title={right.title} items={right.items} accent={right.accent} fromX={32} />
      </div>
      {caption && <Caption text={caption} />}
    </div>
  </SceneLayout>
);
